// Bit-exactness harness: runs the reference ggml CPU graph of a llama.cpp tree that carries the SJ-KVaRN cache (llamAmpere)
// (turbo_wht rotation -> set_rows_tq6_rotated staging with fp16 sink -> sj_kvarn_seal_dyn -> flash_attn_ext with
// the SJ-KVaRN region descriptor) next to sj_kvarn.h, through the real adaptive-tail schedule, and compares bytes.
//
// Build: tests/build_tree_harness.sh <tree root> <tree build dir>
// Run:   tests/tree_harness [--kvd FILE] [--kvd-groups N] [--vectors OUT]
//
// The tree's own header (ggml-sjkvarn.h) is compiled into this file with -ffp-contract=off for the decode
// comparison; the seal, staging and attention results come from the tree's shared libraries unchanged.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-sjkvarn.h"

#define SJ_KVARN_IMPLEMENTATION
#include "../sj_kvarn.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void quantize_row_tq6_0_rotated_ref(const float * x, void * y, int64_t k);
extern "C" void dequantize_row_tq6_0(const void * x, float * y, int64_t k);

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

// deterministic synthetic activations: per-channel outliers and per-token scales, like real K/V
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

ggml_context * mk_ctx(size_t bytes) {
    ggml_init_params ip = { bytes, nullptr, false };
    return ggml_init(ip);
}

void run(ggml_context * ctx, ggml_tensor * t, int nth = 4) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    if (ggml_graph_compute_with_ctx(ctx, gf, nth) != GGML_STATUS_SUCCESS) { fprintf(stderr, "graph failed\n"); exit(2); }
}

struct result_t {
    long rot_bad = 0, stage_bad = 0, sink_bad = 0, rec_total = 0, rec_bad = 0, dec_rows = 0, dec_bad = 0;
    long attn_vals = 0, attn_equal = 0;
    double attn_max_rel = 0.0;
    long policy_bad = 0;
};

uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// one end-to-end cache run of one config through the adaptive-tail schedule
result_t run_cache(const cfg_t & c, uint64_t seed, bool verbose) {
    const int D = 256, HKV = 2, HQ = 4, G = 128;
    const uint32_t S = 128, TAIL = 256, TAIL_MAX = 512, NUB = 64;
    // ubatch schedule: one 192-token prefill chunk (above n_ubatch it would be split; keep it <= 2*NUB), then decode-ish ubatches
    std::vector<uint32_t> ub;
    ub.push_back(128); ub.push_back(64);
    uint32_t total = 192;
    rng_t sr{seed*7 + 3};
    while (total < 1700) { uint32_t n = 1 + sr.next() % NUB; ub.push_back(n); total += n; }
    const uint32_t NPOS = total;

    result_t res;
    rng_t r{seed};
    std::vector<float> kraw, vraw, qraw;
    synth(r, kraw, NPOS, HKV, D, 12.0f);
    synth(r, vraw, NPOS, HKV, D, 3.0f);
    synth(r, qraw, NPOS, HQ, D, 4.0f);

    // ---- rotation: tree turbo_wht(256) vs sj_kvarn_hadamard
    std::vector<float> krot(kraw.size()), vrot(vraw.size()), qrot(qraw.size());
    {
        ggml_context * ctx = mk_ctx((size_t) 256 << 20);
        auto rot = [&](const std::vector<float> & src, std::vector<float> & dst, int nh) {
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) D*nh, NPOS);
            memcpy(a->data, src.data(), src.size()*4);
            ggml_tensor * o = ggml_turbo_wht(ctx, a, 0, 256, nullptr);
            run(ctx, o);
            memcpy(dst.data(), o->data, dst.size()*4);
            std::vector<float> mine(src);
            for (size_t i = 0; i < mine.size(); i += D) sj_kvarn_hadamard(mine.data() + i, D);
            for (size_t i = 0; i < mine.size(); ++i) res.rot_bad += f2u(mine[i]) != f2u(dst[i]);
        };
        rot(kraw, krot, HKV); rot(vraw, vrot, HKV); rot(qraw, qrot, HQ);
        ggml_free(ctx);
    }

    // ---- reference layer (sj_kvarn.h)
    sj_kvarn_config cfg = sj_kvarn_config_default(D, HKV, NPOS + G, NUB);
    cfg.bits_k = c.bk; cfg.bits_v = c.bv; cfg.body = c.body;
    cfg.sink = S; cfg.tail = TAIL; cfg.tail_max = TAIL_MAX;
    sj_kvarn_layer L;
    if (sj_kvarn_layer_init(&L, &cfg) != 0) { fprintf(stderr, "layer init failed for %s\n", c.name); exit(2); }
    sj_kvarn_policy pol;
    sj_kvarn_policy_init(&pol, S, TAIL, TAIL_MAX, G, 0);
    const uint32_t cap = L.cap;

    // ---- tree cache tensors
    const ggml_sj_kvarn::layout tl = ggml_sj_kvarn::make_layout(D, G, c.bk, c.bv);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_TQ6_0, (int64_t) D*HKV);
    const size_t sink_rows = ((size_t) S*HKV*D*2 + row_bytes - 1)/row_bytes;
    const int64_t n_rows = S + cap + sink_rows;
    const int64_t n_groups = (NPOS - S + G - 1)/G + 1;
    ggml_context * pctx = mk_ctx((size_t) 2*n_rows*row_bytes + (size_t) n_groups*HKV*tl.bytes + (1 << 20));
    ggml_tensor * kc = ggml_new_tensor_2d(pctx, GGML_TYPE_TQ6_0, (int64_t) D*HKV, n_rows);
    ggml_tensor * vc = ggml_new_tensor_2d(pctx, GGML_TYPE_TQ6_0, (int64_t) D*HKV, n_rows);
    ggml_tensor * body = ggml_new_tensor_1d(pctx, GGML_TYPE_I8, (int64_t) n_groups*HKV*tl.bytes);
    memset(kc->data, 0, ggml_nbytes(kc)); memset(vc->data, 0, ggml_nbytes(vc)); memset(body->data, 0, ggml_nbytes(body));
    const int32_t body_type = c.body == SJKVARN_BODY_TRELLIS ? GGML_TYPE_I16 : GGML_TYPE_F32;
    body->op_params[7] = body_type;
    if (L.lay.bytes != tl.bytes) { fprintf(stderr, "record size mismatch %u vs %zu\n", L.lay.bytes, (size_t) tl.bytes); exit(2); }

    // the tree's policy, transcribed (llama_kv_cache::apply_ubatch), as a cross-check of sj_kvarn_policy
    uint32_t tB = S, tB_pending = S; bool tdrain = false;

    uint32_t pos0 = 0;
    for (size_t u = 0; u < ub.size(); ++u) {
        const uint32_t n = ub[u];
        // policy
        const uint32_t pend = sj_kvarn_policy_begin_ubatch(&pol, pos0, n);
        {
            tB_pending = tB;
            const bool due = TAIL_MAX == 0 || tdrain || (pos0 > tB && pos0 - tB >= TAIL_MAX);
            tdrain = false;
            if (due && pos0 > S + TAIL) {
                uint32_t target = std::max(tB, S + G*((pos0 - TAIL - S)/G));
                tB_pending = target;
            }
        }
        res.policy_bad += pend != tB_pending;
        const uint32_t B_old = pol.B;
        // seal: tree (sj_kvarn_seal_dyn on the ring) and ours
        int32_t desc_v[GGML_SJKVARN_DESC_N_ENTRIES] = {0};
        desc_v[GGML_SJKVARN_DESC_S] = S; desc_v[GGML_SJKVARN_DESC_CAP] = cap;
        desc_v[GGML_SJKVARN_DESC_G] = G; desc_v[GGML_SJKVARN_DESC_D] = D;
        desc_v[GGML_SJKVARN_DESC_RECBYTES] = tl.bytes; desc_v[GGML_SJKVARN_DESC_HKV] = HKV;
        desc_v[GGML_SJKVARN_DESC_TYPE_K] = GGML_TYPE_TQ6_0; desc_v[GGML_SJKVARN_DESC_TYPE_V] = GGML_TYPE_TQ6_0;
        desc_v[GGML_SJKVARN_DESC_BODY_TYPE] = body_type == GGML_TYPE_I16 ? GGML_TYPE_I16 : 0;
        desc_v[GGML_SJKVARN_DESC_SINK_TYPE] = GGML_TYPE_F16;
        if (pend > B_old) {
            ggml_context * ctx = mk_ctx(16 << 20);
            ggml_tensor * desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GGML_SJKVARN_DESC_N_ENTRIES);
            desc_v[GGML_SJKVARN_DESC_B_OLD] = B_old; desc_v[GGML_SJKVARN_DESC_B] = pend;
            memcpy(desc->data, desc_v, sizeof(desc_v));
            ggml_tensor * s = ggml_sj_kvarn_seal_dyn(ctx, body, kc, vc, desc, D, G, c.bk, c.bv, 16, (int32_t) ((pend - B_old)/G));
            run(ctx, s, 8);
            ggml_free(ctx);
            sj_kvarn_layer_seal(&L, B_old, pend);
            for (uint32_t g = (B_old - S)/G; g < (pend - S)/G; ++g) {
                for (int h = 0; h < HKV; ++h) {
                    const uint8_t * a = (const uint8_t *) body->data + ((size_t) g*HKV + h)*tl.bytes;
                    const uint8_t * b = L.body + ((size_t) g*HKV + h)*L.lay.bytes;
                    res.rec_total++;
                    if (memcmp(a, b, tl.bytes) != 0) {
                        res.rec_bad++;
                        if (verbose && res.rec_bad <= 3) {
                            size_t i = 0; while (a[i] == b[i]) ++i;
                            fprintf(stderr, "  %s: record g%u h%d differs first at byte %zu (tree %02x ours %02x)\n", c.name, g, h, i, a[i], b[i]);
                        }
                    }
                    // decode: tree header (contraction off) vs ours, every row of the record
                    std::vector<float> ta(D), tb(D);
                    for (int t = 0; t < G; ++t) {
                        for (int isv = 0; isv < 2; ++isv) {
                            if (isv) { ggml_sj_kvarn::decode_v_row(a, tl, t, ta.data(), c.body == SJKVARN_BODY_TRELLIS); sj_kvarn_decode_v_row(&L.lay, a, t, tb.data()); }
                            else     { ggml_sj_kvarn::decode_k_row(a, tl, t, ta.data(), c.body == SJKVARN_BODY_TRELLIS); sj_kvarn_decode_k_row(&L.lay, a, t, tb.data()); }
                            res.dec_rows++;
                            res.dec_bad += memcmp(ta.data(), tb.data(), D*4) != 0;
                        }
                    }
                }
            }
        }
        sj_kvarn_policy_commit(&pol);
        tB = tB_pending;

        // store: tree set_rows_tq6_rotated (sink rows as fp16) and ours
        {
            ggml_context * ctx = mk_ctx(32 << 20);
            ggml_tensor * kin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) D*HKV, n);
            ggml_tensor * vin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) D*HKV, n);
            ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n);
            memcpy(kin->data, krot.data() + (size_t) pos0*D*HKV, (size_t) n*D*HKV*4);
            memcpy(vin->data, vrot.data() + (size_t) pos0*D*HKV, (size_t) n*D*HKV*4);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t p = pos0 + i;
                ((int64_t *) idx->data)[i] = p < S ? p : S + (p - S) % cap;
            }
            ggml_tensor * sk = ggml_set_rows_tq6_rotated(ctx, kc, kin, idx);
            ggml_tensor * sv = ggml_set_rows_tq6_rotated(ctx, vc, vin, idx);
            for (ggml_tensor * t : {sk, sv}) { t->op_params[2] = S; t->op_params[3] = S + cap; }
            run(ctx, sk); run(ctx, sv);
            ggml_free(ctx);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t p = pos0 + i;
                sj_kvarn_layer_store(&L, p, krot.data() + (size_t) p*D*HKV, vrot.data() + (size_t) p*D*HKV);
                for (int h = 0; h < HKV; ++h) {
                    for (int isv = 0; isv < 2; ++isv) {
                        const ggml_tensor * cc = isv ? vc : kc;
                        if (p < S) {
                            const uint8_t * a = (const uint8_t *) cc->data + (size_t) (S + cap)*row_bytes + ((size_t) p*HKV + h)*D*2;
                            const uint8_t * b = L.sink + (((size_t) p*HKV + h)*2 + isv)*D*2;
                            res.sink_bad += memcmp(a, b, D*2) != 0;
                        } else {
                            const size_t sb = L.stage_bytes;
                            const uint8_t * a = (const uint8_t *) cc->data + (size_t) (S + (p - S) % cap)*row_bytes + h*sb;
                            const uint8_t * b = L.ring + (((size_t) sj_kvarn_ring_slot(S, cap, p)*HKV + h)*2 + isv)*sb;
                            res.stage_bad += memcmp(a, b, sb) != 0;
                        }
                    }
                }
            }
        }

        // attend: tree flash_attn_ext (SJ-KVaRN reference, causal without mask) vs ours, rotated basis
        {
            const uint32_t N = pos0 + n;
            ggml_context * ctx = mk_ctx(64 << 20);
            ggml_tensor * desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GGML_SJKVARN_DESC_N_ENTRIES);
            desc_v[GGML_SJKVARN_DESC_B_OLD] = B_old; desc_v[GGML_SJKVARN_DESC_B] = pol.B;
            desc_v[GGML_SJKVARN_DESC_N] = N; desc_v[GGML_SJKVARN_DESC_QPOS0] = pos0;
            memcpy(desc->data, desc_v, sizeof(desc_v));
            ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n, HQ);
            for (uint32_t i = 0; i < n; ++i)
                for (int h = 0; h < HQ; ++h)
                    memcpy((float *) q->data + ((size_t) h*n + i)*D, qrot.data() + ((size_t) (pos0 + i)*HQ + h)*D, D*4);
            ggml_tensor * kv = ggml_view_3d(ctx, kc, D, S + cap, HKV, row_bytes, ggml_row_size(GGML_TYPE_TQ6_0, D), 0);
            ggml_tensor * vv = ggml_view_3d(ctx, vc, D, S + cap, HKV, row_bytes, ggml_row_size(GGML_TYPE_TQ6_0, D), 0);
            const float scale = 1.0f/sqrtf((float) D);
            ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, kv, vv, nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_sj_kvarn(fa, body, desc, c.bk, c.bv, (int32_t) GGML_PAD(N, 64));
            run(ctx, fa, 8);
            std::vector<float> mine(D);
            for (uint32_t i = 0; i < n; ++i) {
                for (int h = 0; h < HQ; ++h) {
                    const float * ref = (const float *) fa->data + ((size_t) i*HQ + h)*D;
                    sj_kvarn_layer_attend_row(&L, pol.B, N, qrot.data() + ((size_t) (pos0 + i)*HQ + h)*D, h/(HQ/HKV), pos0 + i, scale, mine.data());
                    double nr = 0, nd = 0;
                    for (int d = 0; d < D; ++d) {
                        nr += (double) ref[d]*ref[d]; nd += (double) (ref[d] - mine[d])*(ref[d] - mine[d]);
                        res.attn_vals++; res.attn_equal += f2u(ref[d]) == f2u(mine[d]);
                    }
                    const double rel = sqrt(nd/(nr > 0 ? nr : 1));
                    if (rel > res.attn_max_rel) res.attn_max_rel = rel;
                }
            }
            ggml_free(ctx);
        }
        pos0 += n;
    }
    ggml_free(pctx);
    sj_kvarn_layer_free(&L);
    return res;
}

// real seal inputs (rotated fp16 K/V groups captured from a model): tree seal_group vs ours
struct kvd_group { int D, G, head; std::vector<uint16_t> K, V; };

std::vector<kvd_group> load_kvd(const char * path, int max_groups) {
    std::vector<kvd_group> out;
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return out; }
    while ((int) out.size() < max_groups) {
        int32_t hdr[4]; char name[64];
        if (fread(hdr, 4, 4, f) != 4) break;
        if (hdr[0] != 0x31445643 || fread(name, 1, 64, f) != 64) break;
        kvd_group g; g.D = hdr[1]; g.G = hdr[2]; g.head = hdr[3];
        g.K.resize((size_t) g.D*g.G); g.V.resize((size_t) g.D*g.G);
        if (fread(g.K.data(), 2, g.K.size(), f) != g.K.size() || fread(g.V.data(), 2, g.V.size(), f) != g.V.size()) break;
        out.push_back(std::move(g));
    }
    fclose(f);
    return out;
}

// FNV-1a 64 over bytes
uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

} // namespace

// shared with tests/test_vectors.c: deterministic seal inputs from integer arithmetic only
#include "vector_inputs.h"

int main(int argc, char ** argv) {
    const char * kvd = nullptr; int kvd_groups = 16; const char * vec_out = nullptr; int seeds = 2;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--kvd") && i + 1 < argc) kvd = argv[++i];
        else if (!strcmp(argv[i], "--kvd-groups") && i + 1 < argc) kvd_groups = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vectors") && i + 1 < argc) vec_out = argv[++i];
        else if (!strcmp(argv[i], "--seeds") && i + 1 < argc) seeds = atoi(argv[++i]);
    }
    ggml_cpu_init();
    int fails = 0;

    if (seeds > 0) printf("== end-to-end cache runs (D=256, 2 KV heads, 4 Q heads, sink 128, tail 256->512, %d seeds)\n", seeds);
    if (seeds > 0) printf("%-12s %7s %6s %6s %6s %9s %9s %10s %12s\n", "config", "records", "rec!=", "stage!=", "sink!=", "dec rows", "dec!=", "attn==%", "attn max rel");
    for (const cfg_t & c : CONFIGS) {
        if (seeds <= 0) break;
        result_t tot;
        for (int s = 0; s < seeds; ++s) {
            result_t r = run_cache(c, 0x5EED0000ULL + 977*s + c.bk*10 + c.bv + 100*c.body, true);
            tot.rot_bad += r.rot_bad; tot.stage_bad += r.stage_bad; tot.sink_bad += r.sink_bad;
            tot.rec_total += r.rec_total; tot.rec_bad += r.rec_bad; tot.dec_rows += r.dec_rows; tot.dec_bad += r.dec_bad;
            tot.attn_vals += r.attn_vals; tot.attn_equal += r.attn_equal; tot.policy_bad += r.policy_bad;
            if (r.attn_max_rel > tot.attn_max_rel) tot.attn_max_rel = r.attn_max_rel;
        }
        printf("%-12s %7ld %6ld %6ld %6ld %9ld %9ld %9.2f%% %12.2e%s%s\n", c.name, tot.rec_total, tot.rec_bad, tot.stage_bad,
               tot.sink_bad, tot.dec_rows, tot.dec_bad, 100.0*tot.attn_equal/tot.attn_vals, tot.attn_max_rel,
               tot.rot_bad ? "  ROTATION MISMATCH" : "", tot.policy_bad ? "  POLICY MISMATCH" : "");
        // attention is a float reference, not a format: the tree build may contract to FMA, so only a tolerance applies
        if (tot.rec_bad || tot.stage_bad || tot.sink_bad || tot.dec_bad || tot.rot_bad || tot.policy_bad || tot.attn_max_rel > 1e-4) fails++;
    }

    if (kvd) {
        std::vector<kvd_group> gs = load_kvd(kvd, kvd_groups);
        printf("== real seal inputs: %zu groups from %s\n", gs.size(), kvd);
        for (const cfg_t & c : CONFIGS) {
            long bad = 0;
            for (const kvd_group & g : gs) {
                const ggml_sj_kvarn::layout tl = ggml_sj_kvarn::make_layout(g.D, g.G, c.bk, c.bv);
                sj_kvarn_layout l;
                if (sj_kvarn_layout_init(&l, g.D, g.G, c.bk, c.bv, c.body) != 0) { bad++; continue; }
                std::vector<uint8_t> a(tl.bytes), b(l.bytes), work(sj_kvarn_seal_workspace_bytes(&l));
                ggml_sj_kvarn::seal_group((const ggml_fp16_t *) g.K.data(), (const ggml_fp16_t *) g.V.data(), g.D, tl, 16, a.data(), c.body == SJKVARN_BODY_TRELLIS);
                sj_kvarn_seal_group(&l, g.K.data(), g.V.data(), g.D, 16, b.data(), work.data());
                bad += a != b;
            }
            printf("%-12s %zu groups, %ld differ\n", c.name, gs.size(), bad);
            if (bad) fails++;
        }
    }

    if (vec_out) {
        // expected hashes for tests/test_vectors.c (tree encoder output on the shared deterministic inputs)
        FILE * f = fopen(vec_out, "w");
        if (!f) { perror(vec_out); return 2; }
        fprintf(f, "# FNV-1a 64 of the tree's output on tests/vector_inputs.h inputs; regenerate with tree_harness --vectors\n");
        for (int D : {128, 256}) {
            std::vector<uint16_t> K((size_t) D*128), V((size_t) D*128);
            sjv_fill_group(D, 128, 1, K.data(), V.data());
            for (const cfg_t & c : CONFIGS) {
                const ggml_sj_kvarn::layout tl = ggml_sj_kvarn::make_layout(D, 128, c.bk, c.bv);
                std::vector<uint8_t> a(tl.bytes);
                ggml_sj_kvarn::seal_group((const ggml_fp16_t *) K.data(), (const ggml_fp16_t *) V.data(), D, tl, 16, a.data(), c.body == SJKVARN_BODY_TRELLIS);
                uint64_t hd = 1469598103934665603ULL;
                std::vector<float> row(D);
                for (int t = 0; t < 128; ++t) {
                    ggml_sj_kvarn::decode_k_row(a.data(), tl, t, row.data(), c.body == SJKVARN_BODY_TRELLIS); hd = fnv(row.data(), D*4, hd);
                    ggml_sj_kvarn::decode_v_row(a.data(), tl, t, row.data(), c.body == SJKVARN_BODY_TRELLIS); hd = fnv(row.data(), D*4, hd);
                }
                fprintf(f, "seal D=%d bk=%d bv=%d body=%d bytes=%zu rec=%016" PRIx64 " dec=%016" PRIx64 "\n", D, c.bk, c.bv, c.body, (size_t) tl.bytes, fnv(a.data(), a.size()), hd);
            }
        }
        {
            std::vector<float> x((size_t) 64*256);
            sjv_fill_rows(64, 256, x.data());
            std::vector<uint8_t> q((size_t) 64*2*98);
            for (int r = 0; r < 64; ++r) quantize_row_tq6_0_rotated_ref(x.data() + (size_t) r*256, q.data() + (size_t) r*196, 256);
            std::vector<float> y(x.size());
            for (int r = 0; r < 64; ++r) dequantize_row_tq6_0(q.data() + (size_t) r*196, y.data() + (size_t) r*256, 256);
            fprintf(f, "tq6 rows=64 n=256 enc=%016" PRIx64 " dec=%016" PRIx64 "\n", fnv(q.data(), q.size()), fnv(y.data(), y.size()*4));
        }
        fclose(f);
        printf("wrote %s\n", vec_out);
    }

    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
