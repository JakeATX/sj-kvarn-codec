// CUDA reference kernels vs the host codec: seal records, decoded rows and staging bytes compared bit for
// bit; attention compared with a tolerance (device expf differs from the host libm in the last bits).
#include "../sj_kvarn_cuda.cuh"
#include "vector_inputs.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main() {
    static const int cfg[][3] = { {4,4,0}, {3,3,1}, {3,2,1}, {2,2,1}, {3,3,0}, {4,2,0} };
    const int D = 256, G = 128, H = 2, NG = 4;
    int fails = 0;
    (void) sjv_fill_rows;
    // seal inputs: NG groups x H heads, token-major rows of H*D fp16 (like a KV cache row)
    std::vector<uint16_t> K((size_t) NG*G*H*D), V(K.size());
    {
        std::vector<uint16_t> k1((size_t) G*D), v1((size_t) G*D);
        for (int g = 0; g < NG; ++g)
            for (int h = 0; h < H; ++h) {
                sjv_fill_group(D, G, (uint32_t) (g*H + h + 11), k1.data(), v1.data());
                for (int t = 0; t < G; ++t) {
                    memcpy(&K[((size_t) (g*G + t)*H + h)*D], &k1[(size_t) t*D], D*2);
                    memcpy(&V[((size_t) (g*G + t)*H + h)*D], &v1[(size_t) t*D], D*2);
                }
            }
    }
    uint16_t * dK, * dV;
    CK(cudaMalloc(&dK, K.size()*2)); CK(cudaMalloc(&dV, V.size()*2));
    CK(cudaMemcpy(dK, K.data(), K.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, V.data(), V.size()*2, cudaMemcpyHostToDevice));
    for (auto & c : cfg) {
        sj_kvarn_layout l;
        sj_kvarn_layout_init(&l, D, G, c[0], c[1], c[2]);
        const int n_rec = NG*H;
        std::vector<uint8_t> host((size_t) n_rec*l.bytes), dev(host.size());
        std::vector<uint8_t> work(sj_kvarn_seal_workspace_bytes(&l));
        for (int r = 0; r < n_rec; ++r) {
            const int g = r / H, h = r % H;
            sj_kvarn_seal_group(&l, &K[(size_t) g*G*H*D + (size_t) h*D], &V[(size_t) g*G*H*D + (size_t) h*D], (size_t) H*D, 16,
                                host.data() + (size_t) r*l.bytes, work.data());
        }
        uint8_t * dB, * dW; float * dk, * dv;
        CK(cudaMalloc(&dB, host.size()));
        CK(cudaMalloc(&dW, (size_t) n_rec*sj_kvarn_cuda_work_stride(&l)));
        CK(sj_kvarn_cuda_seal(&l, dK, dV, (size_t) H*D, H, NG, 16, dB, dW, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(dev.data(), dB, dev.size(), cudaMemcpyDeviceToHost));
        int rec_bad = 0;
        for (int r = 0; r < n_rec; ++r) rec_bad += memcmp(&host[(size_t) r*l.bytes], &dev[(size_t) r*l.bytes], l.bytes) != 0;
        // decode on device from the host records
        CK(cudaMemcpy(dB, host.data(), host.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dk, (size_t) n_rec*G*D*4)); CK(cudaMalloc(&dv, (size_t) n_rec*G*D*4));
        sj_kvarn_cuda_decode_kernel<<<(n_rec*G + 63)/64, 64>>>(l, dB, n_rec, dk, dv);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> ok((size_t) n_rec*G*D), ov(ok.size()), hk(D), hv(D);
        CK(cudaMemcpy(ok.data(), dk, ok.size()*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ov.data(), dv, ov.size()*4, cudaMemcpyDeviceToHost));
        long dec_bad = 0;
        for (int r = 0; r < n_rec; ++r)
            for (int t = 0; t < G; ++t) {
                sj_kvarn_decode_k_row(&l, &host[(size_t) r*l.bytes], t, hk.data());
                sj_kvarn_decode_v_row(&l, &host[(size_t) r*l.bytes], t, hv.data());
                dec_bad += memcmp(hk.data(), &ok[((size_t) r*G + t)*D], D*4) != 0;
                dec_bad += memcmp(hv.data(), &ov[((size_t) r*G + t)*D], D*4) != 0;
            }
        printf("%d/%d %-7s seal %d/%d records bit-exact, decode %ld/%d rows differ\n", c[0], c[1], c[2] ? "trellis" : "scalar",
               n_rec - rec_bad, n_rec, dec_bad, n_rec*G*2);
        fails += rec_bad != 0 || dec_bad != 0;
        cudaFree(dB); cudaFree(dW); cudaFree(dk); cudaFree(dv);
    }

    // staging + attention through a small cache built on the host
    {
        sj_kvarn_config cfg0 = sj_kvarn_config_default(D, H, 2048, 64);
        cfg0.bits_k = 3; cfg0.bits_v = 3; cfg0.tail = 256; cfg0.tail_max = 512;
        sj_kvarn_layer L;
        sj_kvarn_layer_init(&L, &cfg0);
        sj_kvarn_policy pol;
        sj_kvarn_policy_init(&pol, cfg0.sink, cfg0.tail, cfg0.tail_max, G, 0);
        const uint32_t NPOS = 1200;
        std::vector<float> x((size_t) NPOS*H*D), q((size_t) NPOS*4*D);
        uint64_t s = 99;
        for (auto & e : x) e = (float) sjv_gauss(&s) / 1024.0f;
        for (auto & e : q) e = (float) sjv_gauss(&s) / 1024.0f;
        for (uint32_t p = 0; p < NPOS; p += 16) {
            if (sj_kvarn_policy_begin_ubatch(&pol, p, 16) > pol.B) sj_kvarn_layer_seal(&L, pol.B, pol.B_pending);
            sj_kvarn_policy_commit(&pol);
            for (uint32_t i = p; i < p + 16; ++i) sj_kvarn_layer_store(&L, i, &x[(size_t) i*H*D], &x[(size_t) ((i*7) % NPOS)*H*D]);
        }
        // staging bytes on device
        float * dx; uint8_t * dst; std::vector<uint8_t> st((size_t) NPOS*H*L.stage_bytes), sh(st.size());
        CK(cudaMalloc(&dx, x.size()*4)); CK(cudaMalloc(&dst, st.size()));
        CK(cudaMemcpy(dx, x.data(), x.size()*4, cudaMemcpyHostToDevice));
        sj_kvarn_cuda_stage_kernel<<<(NPOS*H + 63)/64, 64>>>(SJKVARN_STAGING_TQ6_0, dx, dst, D, L.stage_bytes, NPOS*H);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(st.data(), dst, st.size(), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < NPOS*H; ++i) sj_kvarn_stage_row(SJKVARN_STAGING_TQ6_0, &x[(size_t) i*D], &sh[(size_t) i*L.stage_bytes], D);
        const bool stage_ok = st == sh;
        printf("tq6_0 staging %u rows %s\n", NPOS*H, stage_ok ? "bit-exact" : "MISMATCH");
        fails += !stage_ok;
        // attention: last 16 positions, 4 query heads
        sj_kvarn_cuda_view cv;
        cv.lay = L.lay; cv.n_head_kv = H; cv.staging = L.cfg.staging; cv.sink_type = L.cfg.sink_type;
        cv.S = L.cfg.sink; cv.cap = L.cap; cv.stage_bytes = L.stage_bytes;
        const size_t sink_b = (size_t) L.cfg.sink*H*2*D*2, ring_b = (size_t) L.cap*H*2*L.stage_bytes, body_b = (size_t) L.n_groups*H*L.lay.bytes;
        uint8_t * dsink, * dring, * dbody; float * dq, * dout, * dscr;
        CK(cudaMalloc(&dsink, sink_b)); CK(cudaMalloc(&dring, ring_b)); CK(cudaMalloc(&dbody, body_b));
        CK(cudaMemcpy(dsink, L.sink, sink_b, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dring, L.ring, ring_b, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbody, L.body, body_b, cudaMemcpyHostToDevice));
        cv.sink = dsink; cv.ring = dring; cv.body = dbody;
        const int NQ = 16, HQ = 4; const uint32_t qpos0 = NPOS - NQ;
        CK(cudaMalloc(&dq, (size_t) NQ*HQ*D*4)); CK(cudaMalloc(&dout, (size_t) NQ*HQ*D*4)); CK(cudaMalloc(&dscr, (size_t) NQ*HQ*3*D*4));
        CK(cudaMemcpy(dq, &q[(size_t) qpos0*HQ*D], (size_t) NQ*HQ*D*4, cudaMemcpyHostToDevice));
        const float scale = 1.0f/16.0f;
        sj_kvarn_cuda_attend_kernel<<<(NQ*HQ + 31)/32, 32>>>(cv, pol.B, NPOS, qpos0, NQ, HQ, dq, scale, dout, dscr);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> out((size_t) NQ*HQ*D), ref(D);
        CK(cudaMemcpy(out.data(), dout, out.size()*4, cudaMemcpyDeviceToHost));
        double maxrel = 0; long eq = 0;
        for (int i = 0; i < NQ*HQ; ++i) {
            sj_kvarn_layer_attend_row(&L, pol.B, NPOS, &q[(size_t) qpos0*HQ*D + (size_t) i*D], (i % HQ)/(HQ/H), qpos0 + i/HQ, scale, ref.data());
            double n2 = 0, e2 = 0;
            for (int d = 0; d < D; ++d) { n2 += (double) ref[d]*ref[d]; e2 += (double) (ref[d] - out[(size_t) i*D + d])*(ref[d] - out[(size_t) i*D + d]); eq += f2u(ref[d]) == f2u(out[(size_t) i*D + d]); }
            maxrel = fmax(maxrel, sqrt(e2/(n2 > 0 ? n2 : 1)));
        }
        printf("attention (B=%u, N=%u, 3/3 trellis body + tq6 tail + f16 sink): %.1f%% values bit-equal, max rel err %.2e\n",
               pol.B, NPOS, 100.0*eq/(NQ*HQ*D), maxrel);
        fails += maxrel > 1e-5;
        sj_kvarn_layer_free(&L);
    }
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
