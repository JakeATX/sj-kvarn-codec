// Bit-exactness harness against a llama.cpp tree's CUDA kernels: the tree's CUDA set_rows_tq6_rotated (staging
// ring + fp16 sink) and kvarn_seal_dyn (sealer) run on the GPU through ggml-backend; the bytes are compared with
// sj_kvarn.h on the host. Two ubatches with a wrapping ring and a seal before each, for every body config.
// Staging rows are compared as stored. A staging row that differs is then replaced by this library's row, so
// the record check measures the sealer on identical input. (ggml-cuda builds with -use_fast_math and sums the
// tq6_0 norm in a tree order, so its staging can differ from the CPU reference in the fp16 norm's last bit.)
//
// Build: tests/build_tree_harness.sh <tree root> <tree build dir> cuda
// Run:   tests/tree_cuda_harness [seeds]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#define SJ_KVARN_IMPLEMENTATION
#include "../sj_kvarn.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct cfg_t { int bk, bv, body; const char * name; };

const cfg_t CONFIGS[] = {
    {4, 4, SJKVARN_BODY_SCALAR,  "4/4 scalar"},
    {3, 3, SJKVARN_BODY_TRELLIS, "3/3 trellis"},
    {3, 2, SJKVARN_BODY_TRELLIS, "3/2 trellis"},
    {2, 2, SJKVARN_BODY_TRELLIS, "2/2 trellis"},
    {4, 3, SJKVARN_BODY_SCALAR,  "4/3 scalar"},
    {3, 3, SJKVARN_BODY_SCALAR,  "3/3 scalar"},
    {4, 2, SJKVARN_BODY_SCALAR,  "4/2 scalar"},
    {2, 4, SJKVARN_BODY_SCALAR,  "2/4 scalar"},
    {3, 2, SJKVARN_BODY_SCALAR,  "3/2 scalar"},
    {2, 2, SJKVARN_BODY_SCALAR,  "2/2 scalar"},
};

struct rng_t {
    uint64_t s;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t) (s >> 11); }
    float uni() { return (float) (next() & 0xFFFFFF) / 16777216.0f; }
    float gauss() { float a = 0; for (int i = 0; i < 12; ++i) a += uni(); return a - 6.0f; }
};

void synth(rng_t & r, std::vector<float> & x, int n_pos, int n_head, int D, float outlier) {
    std::vector<float> ch((size_t) n_head*D);
    for (auto & c : ch) c = r.uni() < 0.03f ? outlier*(0.5f + r.uni()) : 0.3f + r.uni();
    x.resize((size_t) n_pos*n_head*D);
    for (int p = 0; p < n_pos; ++p) {
        const float ts = 0.5f + 1.5f*r.uni();
        for (int i = 0; i < n_head*D; ++i) x[(size_t) p*n_head*D + i] = r.gauss()*ch[i]*ts + 0.05f*ch[i];
    }
}

struct res_t { long stage = 0, stage_bad = 0, sink = 0, sink_bad = 0, rec = 0, rec_bad = 0; };

// run a graph of one node on the backend; inputs are uploaded by the caller after allocation
struct graph_ctx {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    graph_ctx() { ggml_init_params ip = { 64*ggml_tensor_overhead() + 4*ggml_graph_overhead(), nullptr, true }; ctx = ggml_init(ip); }
    void alloc(ggml_backend_t be) { buf = ggml_backend_alloc_ctx_tensors(ctx, be); }
    ~graph_ctx() { if (buf) ggml_backend_buffer_free(buf); ggml_free(ctx); }
};

void compute(ggml_backend_t be, ggml_context * ctx, ggml_tensor * t) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "graph failed\n"); exit(2); }
    ggml_backend_synchronize(be);
}

res_t run_cfg(ggml_backend_t be, const cfg_t & c, uint64_t seed) {
    const int D = 256, HKV = 2, G = 128;
    const uint32_t S = 128, CAP = 384;          // ring smaller than the run: the second ubatch wraps
    const uint32_t N1 = S + 2*G + 40, N2 = S + 4*G + 30; // ubatch 1 = [0,N1), ubatch 2 = [N1,N2)
    res_t res;

    rng_t r{seed};
    std::vector<float> k, v;
    synth(r, k, N2, HKV, D, 12.0f);
    synth(r, v, N2, HKV, D, 3.0f);
    for (size_t i = 0; i < k.size(); i += D) { sj_kvarn_hadamard(k.data() + i, D); sj_kvarn_hadamard(v.data() + i, D); }

    // reference (host)
    sj_kvarn_config cfg = sj_kvarn_config_default(D, HKV, N2 + G, 256);
    cfg.bits_k = c.bk; cfg.bits_v = c.bv; cfg.body = c.body;
    cfg.sink = S; cfg.tail = 256; cfg.tail_max = 0;
    sj_kvarn_layer L;
    if (sj_kvarn_layer_init(&L, &cfg) != 0) { fprintf(stderr, "layer init failed\n"); exit(2); }
    const size_t rec_bytes = L.lay.bytes;

    // tree cache tensors on the GPU (ring rows S..S+CAP-1, fp16 sink appended after row S+CAP)
    const size_t row_bytes = ggml_row_size(GGML_TYPE_TQ6_0, (int64_t) D*HKV);
    const size_t sink_rows = ((size_t) S*HKV*D*2 + row_bytes - 1)/row_bytes;
    const int64_t n_rows = S + CAP + sink_rows;
    const int64_t n_groups = 4;
    graph_ctx cache;
    ggml_tensor * kc = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_TQ6_0, (int64_t) D*HKV, n_rows);
    ggml_tensor * vc = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_TQ6_0, (int64_t) D*HKV, n_rows);
    ggml_tensor * body = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_I8, (int64_t) n_groups*HKV*rec_bytes);
    cache.alloc(be);
    ggml_backend_buffer_clear(cache.buf, 0);
    const int32_t body_type = c.body == SJKVARN_BODY_TRELLIS ? GGML_TYPE_I16 : GGML_TYPE_F32;
    body->op_params[7] = body_type;

    std::vector<uint8_t> kh(ggml_nbytes(kc)), vh(ggml_nbytes(vc)), bh(ggml_nbytes(body));
    uint32_t B = S;
    const uint32_t ub[2][2] = {{0, N1}, {N1, N2}};
    for (int u = 0; u < 2; ++u) {
        const uint32_t pos0 = ub[u][0], pos1 = ub[u][1];
        // seal every whole group below pos0 that is not yet sealed
        const uint32_t target = pos0 > S ? S + G*((pos0 - S)/G) : S;
        if (target > B) {
            graph_ctx g;
            ggml_tensor * desc = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, GGML_KVARN_DESC_N_ENTRIES);
            ggml_tensor * s = ggml_kvarn_seal_dyn(g.ctx, body, kc, vc, desc, D, G, c.bk, c.bv, 16, (int32_t) ((target - B)/G));
            g.alloc(be);
            int32_t dv[GGML_KVARN_DESC_N_ENTRIES] = {0};
            dv[GGML_KVARN_DESC_S] = S; dv[GGML_KVARN_DESC_CAP] = CAP; dv[GGML_KVARN_DESC_G] = G; dv[GGML_KVARN_DESC_D] = D;
            dv[GGML_KVARN_DESC_RECBYTES] = (int32_t) rec_bytes; dv[GGML_KVARN_DESC_HKV] = HKV;
            dv[GGML_KVARN_DESC_TYPE_K] = GGML_TYPE_TQ6_0; dv[GGML_KVARN_DESC_TYPE_V] = GGML_TYPE_TQ6_0;
            dv[GGML_KVARN_DESC_BODY_TYPE] = body_type == GGML_TYPE_I16 ? GGML_TYPE_I16 : 0;
            dv[GGML_KVARN_DESC_SINK_TYPE] = GGML_TYPE_F16;
            dv[GGML_KVARN_DESC_B_OLD] = (int32_t) B; dv[GGML_KVARN_DESC_B] = (int32_t) target; dv[GGML_KVARN_DESC_N] = (int32_t) pos0;
            ggml_backend_tensor_set(desc, dv, 0, sizeof(dv));
            compute(be, g.ctx, s);
            sj_kvarn_layer_seal(&L, B, target);
            ggml_backend_tensor_get(body, bh.data(), 0, bh.size());
            for (uint32_t gi = (B - S)/G; gi < (target - S)/G; ++gi) {
                for (int h = 0; h < HKV; ++h) {
                    const size_t off = ((size_t) gi*HKV + h)*rec_bytes;
                    res.rec++;
                    if (memcmp(bh.data() + off, L.body + off, rec_bytes) != 0) {
                        res.rec_bad++;
                        size_t i = 0; while (bh[off + i] == L.body[off + i]) ++i;
                        fprintf(stderr, "  %s: record g%u h%d differs first at byte %zu\n", c.name, gi, h, i);
                    }
                }
            }
            B = target;
        }
        // store this ubatch
        {
            const uint32_t n = pos1 - pos0;
            graph_ctx g;
            ggml_tensor * kin = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, (int64_t) D*HKV, n);
            ggml_tensor * vin = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, (int64_t) D*HKV, n);
            ggml_tensor * idx = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, n);
            ggml_tensor * sk = ggml_set_rows_tq6_rotated(g.ctx, kc, kin, idx);
            ggml_tensor * sv = ggml_set_rows_tq6_rotated(g.ctx, vc, vin, idx);
            for (ggml_tensor * t : {sk, sv}) { t->op_params[2] = S; t->op_params[3] = S + CAP; }
            g.alloc(be);
            std::vector<int64_t> iv(n);
            for (uint32_t i = 0; i < n; ++i) { const uint32_t p = pos0 + i; iv[i] = p < S ? p : S + (p - S) % CAP; }
            ggml_backend_tensor_set(kin, k.data() + (size_t) pos0*D*HKV, 0, (size_t) n*D*HKV*4);
            ggml_backend_tensor_set(vin, v.data() + (size_t) pos0*D*HKV, 0, (size_t) n*D*HKV*4);
            ggml_backend_tensor_set(idx, iv.data(), 0, n*8);
            compute(be, g.ctx, sk);
            compute(be, g.ctx, sv);
            ggml_backend_tensor_get(kc, kh.data(), 0, kh.size());
            ggml_backend_tensor_get(vc, vh.data(), 0, vh.size());
            bool resync = false;
            for (uint32_t p = pos0; p < pos1; ++p) {
                sj_kvarn_layer_store(&L, p, k.data() + (size_t) p*D*HKV, v.data() + (size_t) p*D*HKV);
                for (int h = 0; h < HKV; ++h) {
                    for (int isv = 0; isv < 2; ++isv) {
                        const uint8_t * cc = isv ? vh.data() : kh.data();
                        if (p < S) {
                            const uint8_t * a = cc + (size_t) (S + CAP)*row_bytes + ((size_t) p*HKV + h)*D*2;
                            const uint8_t * b = L.sink + (((size_t) p*HKV + h)*2 + isv)*D*2;
                            res.sink++; res.sink_bad += memcmp(a, b, D*2) != 0;
                        } else {
                            const size_t sb = L.stage_bytes;
                            const uint8_t * a = cc + (size_t) (S + (p - S) % CAP)*row_bytes + h*sb;
                            const uint8_t * b = L.ring + (((size_t) sj_kvarn_ring_slot(S, L.cap, p)*HKV + h)*2 + isv)*sb;
                            res.stage++;
                            if (memcmp(a, b, sb) != 0) {
                                res.stage_bad++;
                                if (getenv("SJK_DIAG")) {
                                    fprintf(stderr, "  %s: staging p%u h%d %s differs at bytes", c.name, p, h, isv ? "V" : "K");
                                    for (size_t i = 0; i < sb; ++i) if (a[i] != b[i]) fprintf(stderr, " %zu(%02x/%02x)", i, a[i], b[i]);
                                    fprintf(stderr, "\n");
                                }
                                // later seals compare on identical staged input: copy this library's row into the tree cache
                                memcpy((isv ? vh.data() : kh.data()) + (a - cc), b, sb);
                                resync = true;
                            }
                        }
                    }
                }
            }
            if (resync) {
                ggml_backend_tensor_set(kc, kh.data(), 0, kh.size());
                ggml_backend_tensor_set(vc, vh.data(), 0, vh.size());
            }
        }
    }
    sj_kvarn_layer_free(&L);
    return res;
}

} // namespace

int main(int argc, char ** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 1;
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no CUDA backend\n"); return 2; }
    int fails = 0;
    printf("tree CUDA kernels vs sj_kvarn.h (D=256, 2 KV heads, ring wraps)\n");
    printf("%-12s %16s %14s %14s\n", "config", "records bad", "staging bad", "sink bad");
    for (const cfg_t & c : CONFIGS) {
        res_t t;
        for (int s = 0; s < seeds; ++s) {
            res_t r = run_cfg(be, c, 0x9E3779B97F4A7C15ULL ^ (uint64_t) (s + 1)*7919);
            t.rec += r.rec; t.rec_bad += r.rec_bad; t.stage += r.stage; t.stage_bad += r.stage_bad; t.sink += r.sink; t.sink_bad += r.sink_bad;
        }
        const bool ok = t.rec_bad == 0 && t.stage_bad == 0 && t.sink_bad == 0 && t.rec > 0;
        fails += !ok;
        printf("%-12s %8ld/%-7ld %7ld/%-6ld %7ld/%-6ld %s\n", c.name, t.rec_bad, t.rec, t.stage_bad, t.stage, t.sink_bad, t.sink, ok ? "ok" : "FAIL");
    }
    ggml_backend_free(be);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
