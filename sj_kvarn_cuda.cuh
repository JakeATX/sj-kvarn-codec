// sj_kvarn_cuda.cuh - optional CUDA reference kernels for SJ-KVaRN (Staged, Journaled KVarN).
//
// Correctness first: one thread per record for the seal, one thread per row for decode and staging, one
// thread per (query row, query head) for attention. These kernels are a portable starting point and a
// device-side check of the format; they are not tuned for speed. They produce the same record bytes as the
// host code in sj_kvarn.h (the codec routines are compiled for the device from the same source, with
// single-rounding intrinsics in place of the host's volatile helpers).
//
// Usage: include this file in exactly one .cu translation unit (it carries the implementation of
// sj_kvarn.h with internal linkage). Requires CUDA 11+ and a device with double precision.
//
// MIT License, see sj_kvarn.h.
#ifndef SJ_KVARN_CUDA_CUH
#define SJ_KVARN_CUDA_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef SJ_KVARN_IMPLEMENTATION
#define SJ_KVARN_IMPLEMENTATION
#endif
#ifndef SJ_KVARN_STATIC
#define SJ_KVARN_STATIC
#endif
#include "sj_kvarn.h"

// Device view of one layer's cache, using the memory layout of sj_kvarn_layer:
//   sink  [S][n_head_kv][K D | V D] fp16           (sink_type f16)
//   ring  [cap][n_head_kv][K row | V row]          (staging rows, sj_kvarn_stage_row_bytes each)
//   body  [n_groups][n_head_kv][record]            (sj_kvarn_layout.bytes each)
// Positions: p < S -> sink, S <= p < B -> body record (p-S)/G row (p-S)%G, p >= B -> ring slot (p-S)%cap.
typedef struct sj_kvarn_cuda_view {
    sj_kvarn_layout lay;
    int      n_head_kv;
    int      staging;      // sj_kvarn_staging
    int      sink_type;    // sj_kvarn_sink_type
    uint32_t S, cap;
    size_t   stage_bytes;
    const uint8_t * sink;
    const uint8_t * ring;
    const uint8_t * body;
} sj_kvarn_cuda_view;

// ---- seal ------------------------------------------------------------------------------------------------
// Seal n_rec records. Record r = g*n_head_kv + h reads G rows of K/V fp16 for head h starting at
// k + (size_t) g*G*row_stride + h*D (row_stride in elements between consecutive tokens).
// work: n_rec * sj_kvarn_seal_workspace_bytes(&lay) bytes of device memory (8-byte aligned slices).
__global__ void sj_kvarn_cuda_seal_kernel(sj_kvarn_layout lay, const uint16_t * k, const uint16_t * v, size_t row_stride,
                                          int n_head_kv, int n_rec, int iters, uint8_t * body, uint8_t * work, size_t work_stride) {
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= n_rec) return;
    const int g = r / n_head_kv, h = r % n_head_kv;
    const size_t off = (size_t) g*lay.G*row_stride + (size_t) h*lay.D;
    sj_kvarn_seal_group(&lay, k + off, v + off, row_stride, iters, body + (size_t) r*lay.bytes, work + (size_t) r*work_stride);
}

// ---- decode (dequantize) ------------------------------------------------------------------------------------
// out_k/out_v: [n_rec][G][D] floats, rotated basis
__global__ void sj_kvarn_cuda_decode_kernel(sj_kvarn_layout lay, const uint8_t * body, int n_rec, float * out_k, float * out_v) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_rec*lay.G) return;
    const int r = i / lay.G, t = i % lay.G;
    const uint8_t * rec = body + (size_t) r*lay.bytes;
    sj_kvarn_decode_k_row(&lay, rec, t, out_k + (size_t) i*lay.D);
    sj_kvarn_decode_v_row(&lay, rec, t, out_v + (size_t) i*lay.D);
}

// ---- staging -------------------------------------------------------------------------------------------------
// Stage n_rows rotated rows of D floats each (one thread per row).
__global__ void sj_kvarn_cuda_stage_kernel(int staging, const float * x, uint8_t * dst, int D, size_t dst_row_bytes, int n_rows) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    sj_kvarn_stage_row(staging, x + (size_t) i*D, dst + (size_t) i*dst_row_bytes, D);
}

__global__ void sj_kvarn_cuda_unstage_kernel(int staging, const uint8_t * src, size_t src_row_bytes, float * x, int D, int n_rows) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    sj_kvarn_unstage_row(staging, src + (size_t) i*src_row_bytes, x + (size_t) i*D, D);
}

// ---- attention ------------------------------------------------------------------------------------------------
// Row resolver shared by the attention kernel.
static __device__ void sj_kvarn_cuda_kv_row(const sj_kvarn_cuda_view & c, uint32_t B, uint32_t p, int h, float * k, float * v) {
    const int D = c.lay.D, H = c.n_head_kv;
    if (p >= c.S && p < B) {
        const uint32_t g = (p - c.S)/(uint32_t) c.lay.G, t = (p - c.S)%(uint32_t) c.lay.G;
        const uint8_t * rec = c.body + ((size_t) g*H + h)*c.lay.bytes;
        sj_kvarn_decode_k_row(&c.lay, rec, (int) t, k);
        sj_kvarn_decode_v_row(&c.lay, rec, (int) t, v);
    } else if (p < c.S && c.sink_type == SJKVARN_SINK_F16) {
        const uint16_t * kd = (const uint16_t *) c.sink + ((size_t) p*H + h)*2*D;
        for (int d = 0; d < D; ++d) { k[d] = sj_kvarn_f16_to_f32(kd[d]); v[d] = sj_kvarn_f16_to_f32(kd[D + d]); }
    } else {
        const uint8_t * row = p < c.S ? c.sink + ((size_t) p*H + h)*2*c.stage_bytes
                                      : c.ring + ((size_t) sj_kvarn_ring_slot(c.S, c.cap, p)*H + h)*2*c.stage_bytes;
        sj_kvarn_unstage_row(c.staging, row, k, D);
        sj_kvarn_unstage_row(c.staging, row + c.stage_bytes, v, D);
    }
}

// Causal attention for n_q query rows at positions qpos0.. (one thread per (row, query head)), positions
// [0, min(N, qpos+1)), online softmax in position order, rotated basis in and out.
// q, out: [n_q][n_head][D]. scratch: n_q*n_head*3*D floats of device memory (k row, v row, accumulator).
__global__ void sj_kvarn_cuda_attend_kernel(sj_kvarn_cuda_view c, uint32_t B, uint32_t N, uint32_t qpos0, int n_q, int n_head,
                                            const float * q, float scale, float * out, float * scratch) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_q*n_head) return;
    const int D = c.lay.D, iq = i / n_head, hq = i % n_head, h = hq / (n_head / c.n_head_kv);
    const uint32_t qpos = qpos0 + (uint32_t) iq;
    const uint32_t end = N < qpos + 1 ? N : qpos + 1;
    const float * qr = q + (size_t) i*D;
    float * krow = scratch + (size_t) i*3*D, * vrow = krow + D, * acc = vrow + D;
    float M = -INFINITY, Ssum = 0.0f;
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;
    for (uint32_t p = 0; p < end; ++p) {
        sj_kvarn_cuda_kv_row(c, B, p, h, krow, vrow);
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s = __fadd_rn(s, __fmul_rn(qr[d], krow[d]));
        s = __fmul_rn(s, scale);
        float ms = 1.0f, vs = 1.0f;
        if (s > M) {
            const float Mold = M;
            M = s;
            ms = expf(Mold - M);
            for (int d = 0; d < D; ++d) acc[d] = __fmul_rn(acc[d], ms);
        } else {
            vs = expf(s - M);
        }
        for (int d = 0; d < D; ++d) acc[d] = __fadd_rn(acc[d], __fmul_rn(vrow[d], vs));
        Ssum = __fadd_rn(__fmul_rn(Ssum, ms), vs);
    }
    const float inv = Ssum == 0.0f ? 0.0f : 1.0f/Ssum;
    for (int d = 0; d < D; ++d) out[(size_t) i*D + d] = acc[d]*inv;
}

// ---- host helpers -----------------------------------------------------------------------------------------------
static inline size_t sj_kvarn_cuda_work_stride(const sj_kvarn_layout * lay) {
    return (sj_kvarn_seal_workspace_bytes(lay) + 255) & ~(size_t) 255;
}

static inline cudaError_t sj_kvarn_cuda_seal(const sj_kvarn_layout * lay, const uint16_t * d_k, const uint16_t * d_v, size_t row_stride,
                                             int n_head_kv, int n_groups, int iters, uint8_t * d_body, uint8_t * d_work, cudaStream_t st) {
    const int n_rec = n_groups*n_head_kv;
    if (n_rec <= 0) return cudaSuccess;
    // the seal is deeply recursive-free but uses a large per-thread frame for the trellis search
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);
    sj_kvarn_cuda_seal_kernel<<<(n_rec + 31)/32, 32, 0, st>>>(*lay, d_k, d_v, row_stride, n_head_kv, n_rec, iters, d_body,
                                                              d_work, sj_kvarn_cuda_work_stride(lay));
    return cudaGetLastError();
}

#endif // SJ_KVARN_CUDA_CUH
