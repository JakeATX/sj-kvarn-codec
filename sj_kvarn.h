/*
 * sj_kvarn.h - SJ-KVaRN KV-cache codec, single-header C99 reference library.
 *
 * SJ-KVaRN = "Staged, Journaled KVarN".
 *
 *   Staged:    every new K/V row first enters an intermediate-precision staging
 *              ring (tq6_0: 6-bit codes + one fp16 norm per 128 values,
 *              6.125 bits per element). The staging ring is the exact tail that
 *              attention reads for recent tokens.
 *   Journaled: like a write-ahead log, recent tokens live in that exact tail and
 *              are committed ("sealed") in batches of whole groups into the
 *              compact body. With the default adaptive tail, the tail grows from
 *              4096 to 8192 positions and is then committed back down to 4096 in
 *              one batch, so sealing happens rarely and in large chunks.
 *
 * The sealed body is the KVarN method of huawei-csl/KVarN
 * (https://github.com/huawei-csl/KVarN, Apache-2.0): Hadamard-rotated K/V,
 * log-domain variance balancing of each (head, group) tile, asymmetric per-row
 * quantization and scale absorption. This file is an independent
 * implementation; it contains no code from that repository. Differences from
 * the KVarN reference are listed in README.md ("Format notes").
 *
 * Body codecs in this file:
 *   4/4 scalar   4-bit K, 4-bit V, round-to-nearest codes          4.28125 bits/element
 *   3/3 trellis  3-bit K, 3-bit V, trellis-coded (Viterbi) codes    3.28125 bits/element
 *   3/2 trellis  3-bit K, 2-bit V, trellis-coded codes              2.78125 bits/element
 *   (scalar bodies at any of 2/3/4 bits per side, and the 2/2 trellis, also work)
 *
 * Usage (stb style): in exactly one C or C++ file
 *     #define SJ_KVARN_IMPLEMENTATION
 *     #include "sj_kvarn.h"
 * and include it without the define everywhere else. Define SJ_KVARN_STATIC
 * as well to make every function static to that file. CUDA translation units
 * include sj_kvarn_cuda.cuh instead (it compiles the codec for host and device).
 * Link with the C math library (-lm). No other dependencies.
 *
 * Numerics: every operation whose rounding affects the stored bytes goes
 * through explicit single-rounding helpers, so the record bytes do not depend
 * on -ffp-contract or FMA availability. Requirements: IEEE-754 binary32 and
 * binary64 with round-to-nearest-even, no flush-to-zero, FLT_EVAL_METHOD 0
 * (any x86-64, AArch64 or CUDA target; not 32-bit x87). Do not build the
 * implementation with -ffast-math.
 *
 * ---------------------------------------------------------------------------
 * Integration API (what an engine calls, and when). See README.md for a guide.
 *
 *   setup     sj_kvarn_config_default(), sj_kvarn_layer_init() per attention
 *             layer, sj_kvarn_policy_init() once per sequence/cache.
 *   append    rotate the new K/V rows (sj_kvarn_rotate_heads) and store them:
 *             sj_kvarn_layer_store(layer, pos, k_rot, v_rot). Positions below
 *             the sink go to the fp16 sink, the rest to the tq6_0 staging ring.
 *   seal      before computing a ubatch that starts at position pos0:
 *               sj_kvarn_policy_begin_ubatch(&pol, pos0, n_tokens);
 *               for each layer: sj_kvarn_layer_seal(layer, pol.B, pol.B_pending);
 *               sj_kvarn_policy_commit(&pol);
 *             (sealing reads only positions below pos0, which were written by
 *             earlier ubatches, so it can run before this ubatch is stored).
 *   attend    rotate Q with the same Hadamard, attend over positions [0, N)
 *             (sj_kvarn_layer_attend_row: sink -> fp16, [sink, B) -> body,
 *             [B, N) -> staging ring), and un-rotate the output with the same
 *             Hadamard (it is its own inverse).
 *   idle      optional: sj_kvarn_policy_idle(&pol, end) + seal + commit
 *             compresses a grown adaptive tail while the engine is idle.
 * ---------------------------------------------------------------------------
 *
 * MIT License. Copyright (c) 2026 Jake K. Full text at the end of this file.
 */
#ifndef SJ_KVARN_H
#define SJ_KVARN_H

#include <stddef.h>
#include <stdint.h>

#define SJKVARN_VERSION_MAJOR 1
#define SJKVARN_VERSION_MINOR 0
#define SJKVARN_FORMAT_VERSION 1 /* bumps whenever stored bytes change */

#ifndef SJ_KVARN_HD
#if defined(__CUDACC__)
#define SJ_KVARN_HD __host__ __device__
#else
#define SJ_KVARN_HD
#endif
#endif

#ifndef SJKVARN_DEF
#ifdef SJ_KVARN_STATIC
#define SJKVARN_DEF static
#else
#define SJKVARN_DEF extern
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* Format constants                                                          */
/* ------------------------------------------------------------------------- */

#define SJKVARN_GROUP_DEFAULT     128   /* tokens per sealed group (G)                    */
#define SJKVARN_SINK_DEFAULT      128   /* leading positions kept exact (fp16)            */
#define SJKVARN_TAIL_DEFAULT      4096  /* exact tail kept after a commit                 */
#define SJKVARN_TAIL_MAX_DEFAULT  8192  /* tail length that triggers a commit (adaptive)  */
#define SJKVARN_ITERS_DEFAULT     16    /* variance-balancing iterations                  */
#define SJKVARN_RED_LANES         8     /* reduction contract, see sj_kvarn__sample_std   */

#define SJKVARN_TQ6_QK            128   /* values per tq6_0 staging block                 */
#define SJKVARN_TQ6_BLOCK_BYTES   98    /* fp16 norm + 64 B low nibbles + 32 B high bits  */

/* trellis bodies (3-bit and 2-bit sides): one Viterbi sequence per channel along
 * the 128 tokens of a group; window = current code over the 6 history bits */
#define SJKVARN_TRELLIS_SEQ       128
#define SJKVARN_TRELLIS_HIST_BITS 6
#define SJKVARN_TRELLIS_NSTATE    64    /* 2^6 */

typedef enum sj_kvarn_body {
    SJKVARN_BODY_SCALAR  = 0,  /* round-to-nearest codes                     */
    SJKVARN_BODY_TRELLIS = 1,  /* trellis-coded codes (3/3, 3/2, 2/2 pairs)  */
    SJKVARN_BODY_AUTO    = 2   /* trellis for 3/3, 3/2, 2/2; scalar otherwise */
} sj_kvarn_body;

typedef enum sj_kvarn_staging {
    SJKVARN_STAGING_TQ6_0 = 0, /* default: 6.125 bits per element                    */
    SJKVARN_STAGING_F16   = 1  /* opt-in comparison arm: exact fp16 tail             */
} sj_kvarn_staging;

typedef enum sj_kvarn_sink_type {
    SJKVARN_SINK_F16     = 0,  /* default: separate fp16 sink                         */
    SJKVARN_SINK_INHERIT = 1   /* sink rows use the staging type                      */
} sj_kvarn_sink_type;

/* tq6_0 staging block: 128 values in the rotated basis.
 * value[i] = TQ6_CENTROID[code_i] * norm, code_i = (qs nibble i) | (qh 2-bit field i) << 4
 * qs[i/2] holds code bits 0..3 of value i at bit 4*(i%2); qh[i/4] holds bits 4..5 at bit 2*(i%4).
 * norm = ||x|| / ||centroid vector|| (norm-corrected), fp16. */
typedef struct sj_kvarn_tq6_block {
    uint16_t norm;     /* fp16 bits */
    uint8_t  qs[64];
    uint8_t  qh[32];
} sj_kvarn_tq6_block;

/* One sealed record = one (group of G tokens, KV head). Byte layout, in order:
 *   K payload   G*D*bits_k/8 bytes
 *   V payload   G*D*bits_v/8 bytes
 *   Kscale[D]   fp16  (per channel, in K channel order, see sj_kvarn_k_ch_idx)
 *   Kzero[D]    fp16
 *   Ktok[G]     fp16  (per token)
 *   Vch[D]      fp16  (per channel, in V channel order, see sj_kvarn_v_ch_idx)
 *   Vscale[G]   fp16  (per token)
 *   Vzero[G]    fp16
 * Reconstruction (rotated basis):
 *   K[t,d] = (qK[t,d] * Kscale[d] + Kzero[d]) * Ktok[t]
 *   V[t,d] = (qV[t,d] * Vscale[t] + Vzero[t]) * Vch[d]
 * where q is the integer code (scalar body) or the codebook value of the code's
 * trellis window (trellis body). Payload bit order: see sj_kvarn_code_bit. */
typedef struct sj_kvarn_layout {
    int D, G, bits_k, bits_v, body;
    uint32_t k_payload, v_payload;   /* byte offsets */
    uint32_t k_scale, k_zero, k_tok, v_ch, v_scale, v_zero;
    uint32_t bytes;                  /* record size */
} sj_kvarn_layout;

/* ---- fp16 ---- */
SJKVARN_DEF SJ_KVARN_HD uint16_t sj_kvarn_f32_to_f16(float f);    /* round to nearest even */
SJKVARN_DEF SJ_KVARN_HD float    sj_kvarn_f16_to_f32(uint16_t h);

/* ---- rotation ---- */
/* Orthonormal Sylvester-Hadamard transform in place (n a power of two). It is
 * its own inverse. The codec rotates every head of Q, K and V with n = head_dim. */
SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_hadamard(float * x, int n);
SJKVARN_DEF void sj_kvarn_rotate_heads(float * x, int n_heads, int head_dim);

/* ---- record layout ---- */
SJKVARN_DEF int      sj_kvarn_body_resolve(int body, int bits_k, int bits_v); /* AUTO -> SCALAR/TRELLIS */
SJKVARN_DEF SJ_KVARN_HD int sj_kvarn_layout_init(sj_kvarn_layout * l, int D, int G, int bits_k, int bits_v, int body);
SJKVARN_DEF size_t   sj_kvarn_record_bytes(int D, int G, int bits_k, int bits_v);
SJKVARN_DEF double   sj_kvarn_body_bits_per_element(const sj_kvarn_layout * l);
/* bit offset of code (token t, channel d) inside the K (is_v = 0) or V payload */
SJKVARN_DEF SJ_KVARN_HD uint32_t sj_kvarn_code_bit(const sj_kvarn_layout * l, int t, int d, int is_v);
SJKVARN_DEF SJ_KVARN_HD int sj_kvarn_k_ch_idx(const sj_kvarn_layout * l, int d);
SJKVARN_DEF SJ_KVARN_HD int sj_kvarn_v_ch_idx(const sj_kvarn_layout * l, int d);

/* ---- seal (encode) ---- */
/* Scratch needed by sj_kvarn_seal_group (no allocation inside the codec). */
SJKVARN_DEF SJ_KVARN_HD size_t sj_kvarn_seal_workspace_bytes(const sj_kvarn_layout * l);
/* Seal one (group, head). K and V are G rotated fp16 rows of D values; row t starts
 * at K + t*row_stride (in elements). Writes l->bytes bytes to rec. work must hold
 * sj_kvarn_seal_workspace_bytes(l) bytes, aligned to 8. iters = 16 by default. */
SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_seal_group(const sj_kvarn_layout * l, const uint16_t * K, const uint16_t * V,
                                                 size_t row_stride, int iters, uint8_t * rec, void * work);

/* ---- decode ---- */
SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_decode_k_row(const sj_kvarn_layout * l, const uint8_t * rec, int t, float * out);
SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_decode_v_row(const sj_kvarn_layout * l, const uint8_t * rec, int t, float * out);

/* ---- staging (tq6_0) ---- */
/* x is already rotated; n % 128 == 0; writes n/128 blocks */
SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_tq6_quantize_row(const float * x, sj_kvarn_tq6_block * y, int n);
SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_tq6_dequantize_row(const sj_kvarn_tq6_block * y, float * x, int n);
SJKVARN_DEF size_t sj_kvarn_stage_row_bytes(int staging, int D); /* bytes for one head row */
SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_stage_row(int staging, const float * x_rot, void * dst, int D);
SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_unstage_row(int staging, const void * src, float * x_rot, int D);

/* ---- reference attention helpers (rotated basis) ---- */
/* scores[t] = dot(q, K[t]) for the first n tokens of a record */
SJKVARN_DEF void sj_kvarn_record_scores(const sj_kvarn_layout * l, const uint8_t * rec, const float * q, int n, float * scores);
/* acc[d] += sum_t w[t] * V[t,d] for the first n tokens of a record */
SJKVARN_DEF void sj_kvarn_record_accum_v(const sj_kvarn_layout * l, const uint8_t * rec, const float * w, int n, float * acc);

/* ---- adaptive tail / journal policy ---- */
typedef struct sj_kvarn_policy {
    uint32_t sink, tail, tail_max, group, flush_chunk; /* tail_max 0 = fixed tail */
    uint32_t B;          /* sealed end: positions [sink, B) are in the body           */
    uint32_t B_prev;     /* B before the last commit                                  */
    uint32_t B_pending;  /* seal target of the current ubatch, B <= B_pending          */
    int      draining;   /* a chunked flush is still in progress (flush_chunk > 0)    */
} sj_kvarn_policy;

SJKVARN_DEF void     sj_kvarn_policy_init(sj_kvarn_policy * p, uint32_t sink, uint32_t tail, uint32_t tail_max,
                                          uint32_t group, uint32_t flush_chunk);
SJKVARN_DEF void     sj_kvarn_policy_reset(sj_kvarn_policy * p);
/* Call before a ubatch whose first position is pos0 (all positions below pos0 are
 * stored). Returns B_pending; seal [B, B_pending) and then call commit. */
SJKVARN_DEF uint32_t sj_kvarn_policy_begin_ubatch(sj_kvarn_policy * p, uint32_t pos0, uint32_t n_tokens);
SJKVARN_DEF void     sj_kvarn_policy_commit(sj_kvarn_policy * p);
/* Idle compression: end = number of stored positions. Returns the new target
 * (B_pending) or B when there is nothing to seal. Seal [B, B_pending), commit. */
SJKVARN_DEF uint32_t sj_kvarn_policy_idle(sj_kvarn_policy * p, uint32_t end);
/* staging ring rows needed so that no unsealed row is overwritten */
SJKVARN_DEF uint32_t sj_kvarn_ring_capacity(uint32_t tail, uint32_t tail_max, uint32_t group, uint32_t n_ubatch);
SJKVARN_DEF SJ_KVARN_HD uint32_t sj_kvarn_ring_slot(uint32_t sink, uint32_t cap, uint32_t pos); /* pos >= sink */

/* ---- reference per-layer cache (CPU, allocates) ---- */
typedef struct sj_kvarn_config {
    int      head_dim;      /* D: power of two, multiple of 128 (256 is the tested size) */
    int      n_head_kv;
    int      group;         /* G = 128                                                   */
    int      bits_k, bits_v;
    int      body;          /* sj_kvarn_body                                             */
    int      iters;
    int      staging;       /* sj_kvarn_staging                                          */
    int      sink_type;     /* sj_kvarn_sink_type                                        */
    uint32_t sink, tail, tail_max, flush_chunk;
    uint32_t n_ubatch;      /* largest ubatch the engine will submit                     */
    uint32_t n_ctx;         /* maximum positions                                         */
} sj_kvarn_config;

typedef struct sj_kvarn_layer {
    sj_kvarn_config cfg;
    sj_kvarn_layout lay;
    uint32_t cap;           /* staging ring rows */
    uint32_t n_groups;      /* body capacity in groups */
    size_t   stage_bytes;   /* bytes of one head row in the ring */
    uint8_t * ring;         /* [cap][n_head_kv][stage_bytes] */
    uint8_t * sink;         /* f16: [sink][n_head_kv][D] fp16; inherit: [sink][n_head_kv][stage_bytes] */
    uint8_t * body;         /* [n_groups][n_head_kv][lay.bytes] */
    void    * work;
    uint16_t * kstage, * vstage; /* [G][D] fp16 seal inputs */
} sj_kvarn_layer;

SJKVARN_DEF sj_kvarn_config sj_kvarn_config_default(int head_dim, int n_head_kv, uint32_t n_ctx, uint32_t n_ubatch);
SJKVARN_DEF int  sj_kvarn_layer_init(sj_kvarn_layer * L, const sj_kvarn_config * cfg);
SJKVARN_DEF void sj_kvarn_layer_free(sj_kvarn_layer * L);
/* store one position (all KV heads, rotated, n_head_kv*D floats each) */
SJKVARN_DEF void sj_kvarn_layer_store(sj_kvarn_layer * L, uint32_t pos, const float * k_rot, const float * v_rot);
/* seal positions [B_from, B_to) (multiples of G past the sink) out of the ring */
SJKVARN_DEF void sj_kvarn_layer_seal(sj_kvarn_layer * L, uint32_t B_from, uint32_t B_to);
/* resolve position pos of KV head h to rotated K/V rows given the sealed end B */
SJKVARN_DEF void sj_kvarn_layer_kv_row(const sj_kvarn_layer * L, uint32_t B, uint32_t pos, int h, float * k, float * v);
/* one query row of one query head mapped to KV head h: causal over [0, min(N, qpos+1)).
 * q_rot and out_rot are in the rotated basis; scale is usually 1/sqrt(D). */
SJKVARN_DEF void sj_kvarn_layer_attend_row(const sj_kvarn_layer * L, uint32_t B, uint32_t N, const float * q_rot,
                                           int h, uint32_t qpos, float scale, float * out_rot);

#ifdef __cplusplus
}
#endif
#endif /* SJ_KVARN_H */

/* ========================================================================= */
/* Implementation                                                            */
/* ========================================================================= */
#if defined(SJ_KVARN_IMPLEMENTATION) && !defined(SJ_KVARN_IMPLEMENTATION_DONE)
#define SJ_KVARN_IMPLEMENTATION_DONE

#include <math.h>
#include <string.h>
#include <stdlib.h>
#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- built-in tables ------------------------------------------------------ */

/* trained trellis codebooks (fp16 bits, value in units of the row step):
 * index = trellis window (current code in the top bits over 6 history bits) */
/* ---- BEGIN GENERATED CODEBOOKS (tools/gen_codebooks.py) ---- */
#define SJKVARN_CB3_SHA256 "0d6e6d63309612d0df48ec6ea2bacd00c56e97fe35a67848c53892eafa1ac078" /* sha256 of the uint16 K[] then V[] table file */
#define SJKVARN_CB2_SHA256 "c7a63f25c47604af04a5be15d4704204d2fd8c7235ce441b88fb4a2da692dba8" /* sha256 of the uint16 K[] then V[] table file */
#define SJKVARN_CB3_K_INIT { \
    0x37b0, 0x30bc, 0x3934, 0x30b8, 0x3a53, 0x322f, 0x3aa3, 0x349c, 0x3425, 0x32e0, 0x3c01, 0x34c8, 0x3c3f, 0x3259, 0x39af, 0x3cba, \
    0x3b63, 0x3a3a, 0x32de, 0x3c40, 0x34c2, 0x3c52, 0x3dac, 0x3628, 0x3458, 0x3bf3, 0x3dec, 0x3603, 0x3d74, 0x3b80, 0x338c, 0x3d09, \
    0x3c54, 0x3263, 0x3b94, 0x3d9b, 0x3351, 0x3eaf, 0x3c7a, 0x3514, 0x3549, 0x3afa, 0x3e57, 0x39c5, 0x3d36, 0x36a0, 0x3f1d, 0x3c7e, \
    0x3ce7, 0x3d48, 0x3335, 0x3dac, 0x36eb, 0x3d9e, 0x35a2, 0x3efb, 0x392c, 0x37c2, 0x3dce, 0x3989, 0x3f5f, 0x3b88, 0x3ede, 0x3a5a, \
    0x3d8f, 0x3afc, 0x3ddd, 0x3bd6, 0x3e58, 0x3ccb, 0x3e86, 0x3d65, 0x3c4d, 0x3ce8, 0x3ede, 0x3d55, 0x3f99, 0x3d63, 0x3ee0, 0x4014, \
    0x3eb6, 0x3e15, 0x3e56, 0x4030, 0x3e68, 0x3f98, 0x4087, 0x3ec2, 0x3df5, 0x3f77, 0x4057, 0x3f09, 0x4061, 0x4038, 0x3ee6, 0x4090, \
    0x3ffa, 0x3ddd, 0x3ffb, 0x40c5, 0x3f8f, 0x40f6, 0x40a0, 0x3f4a, 0x3e8e, 0x3f72, 0x40e7, 0x3fd0, 0x411c, 0x401a, 0x417c, 0x4096, \
    0x405e, 0x405e, 0x3ffa, 0x411a, 0x4012, 0x414a, 0x4072, 0x4177, 0x3f81, 0x3eca, 0x40ec, 0x4000, 0x418a, 0x4066, 0x41a4, 0x4086, \
    0x4044, 0x3dff, 0x404c, 0x3ea1, 0x4054, 0x3ff9, 0x406b, 0x4048, 0x3ebf, 0x3f95, 0x4052, 0x3fef, 0x40f3, 0x4023, 0x40b0, 0x4170, \
    0x4072, 0x4042, 0x4074, 0x415f, 0x409d, 0x4105, 0x41bc, 0x40bc, 0x407f, 0x4106, 0x4132, 0x4119, 0x415f, 0x41c3, 0x40e1, 0x41f5, \
    0x414e, 0x4055, 0x414e, 0x4213, 0x4100, 0x4205, 0x4234, 0x4142, 0x40ea, 0x4110, 0x422a, 0x4146, 0x427b, 0x4199, 0x42c4, 0x4217, \
    0x41e1, 0x417f, 0x4196, 0x42a0, 0x419b, 0x42d8, 0x4207, 0x42f2, 0x4189, 0x40fc, 0x4276, 0x4196, 0x42ff, 0x41fe, 0x4339, 0x4244, \
    0x419c, 0x4022, 0x4178, 0x40ae, 0x4110, 0x41a5, 0x4117, 0x41ca, 0x4095, 0x412c, 0x40de, 0x4172, 0x41c1, 0x41b0, 0x41aa, 0x42b1, \
    0x418a, 0x413f, 0x4185, 0x424d, 0x4216, 0x41ee, 0x42e1, 0x41f2, 0x41ba, 0x423d, 0x41cf, 0x4290, 0x41f7, 0x42f8, 0x4201, 0x431a, \
    0x4266, 0x418b, 0x4259, 0x4332, 0x4227, 0x42df, 0x437a, 0x428e, 0x425a, 0x4256, 0x4341, 0x4268, 0x43a9, 0x42e3, 0x43da, 0x436f, \
    0x432f, 0x4272, 0x42ed, 0x43e3, 0x42c3, 0x440c, 0x435b, 0x4419, 0x431f, 0x4277, 0x43da, 0x42f7, 0x4428, 0x435a, 0x444a, 0x43ba, \
    0x42fd, 0x4152, 0x4287, 0x423c, 0x41f6, 0x4322, 0x423d, 0x4336, 0x41c6, 0x4296, 0x41c7, 0x42f2, 0x426a, 0x4359, 0x42a6, 0x43f3, \
    0x42aa, 0x4232, 0x4280, 0x432a, 0x437f, 0x42c9, 0x43ff, 0x4329, 0x42f2, 0x4359, 0x42b0, 0x43b5, 0x42dc, 0x4405, 0x4323, 0x441e, \
    0x4372, 0x42ae, 0x435b, 0x4424, 0x433f, 0x43c1, 0x4446, 0x43be, 0x43a5, 0x4382, 0x4413, 0x4383, 0x445e, 0x440a, 0x445a, 0x4457, \
    0x4437, 0x4379, 0x4412, 0x4487, 0x43df, 0x449f, 0x4450, 0x44aa, 0x444e, 0x43e2, 0x4495, 0x441d, 0x44c5, 0x444a, 0x44e6, 0x4487, \
    0x4442, 0x4298, 0x4395, 0x43fb, 0x431b, 0x444d, 0x4391, 0x445c, 0x4310, 0x4400, 0x42df, 0x4424, 0x436b, 0x445c, 0x43de, 0x44a3, \
    0x43e6, 0x434d, 0x43a0, 0x441d, 0x4468, 0x43e9, 0x4495, 0x4436, 0x4427, 0x443c, 0x43b4, 0x446d, 0x43e8, 0x4490, 0x442a, 0x44b6, \
    0x4445, 0x43d9, 0x4436, 0x44b1, 0x4428, 0x4462, 0x44cc, 0x4478, 0x4473, 0x4468, 0x4473, 0x445a, 0x44e6, 0x44a3, 0x44b9, 0x44f4, \
    0x44da, 0x4449, 0x44ac, 0x4520, 0x447d, 0x4533, 0x44f9, 0x4532, 0x4502, 0x44a4, 0x4539, 0x44bf, 0x455e, 0x44e3, 0x4579, 0x452a, \
    0x4521, 0x43ff, 0x447b, 0x44e8, 0x4447, 0x4507, 0x4483, 0x452b, 0x4448, 0x44cc, 0x441c, 0x44de, 0x4460, 0x4514, 0x44ac, 0x4568, \
    0x44b0, 0x4456, 0x447f, 0x44d1, 0x451c, 0x44a0, 0x4546, 0x44ec, 0x4503, 0x44ef, 0x4473, 0x4517, 0x4488, 0x4539, 0x44d8, 0x456d, \
    0x44e1, 0x4499, 0x44e4, 0x456d, 0x44d0, 0x44f5, 0x4570, 0x4521, 0x4523, 0x452f, 0x44cd, 0x4509, 0x457b, 0x4547, 0x4534, 0x45a2, \
    0x4598, 0x44eb, 0x4551, 0x45c4, 0x4515, 0x45cd, 0x45ae, 0x45ab, 0x45c0, 0x4569, 0x45ee, 0x456f, 0x4608, 0x4587, 0x461a, 0x45dc, \
    0x464e, 0x44f9, 0x45aa, 0x46a4, 0x4564, 0x4642, 0x457a, 0x4650, 0x4546, 0x4683, 0x4507, 0x45f0, 0x455c, 0x46b7, 0x45b5, 0x46a6, \
    0x45cf, 0x4568, 0x45b4, 0x46c5, 0x4607, 0x45a0, 0x4679, 0x45d9, 0x46b1, 0x4684, 0x453a, 0x45dd, 0x4549, 0x46b5, 0x45db, 0x46ac, \
    0x45b2, 0x4597, 0x4614, 0x46c8, 0x4606, 0x45aa, 0x46b6, 0x4600, 0x4621, 0x46c2, 0x4577, 0x45f3, 0x46c7, 0x462a, 0x45f1, 0x46bf, \
    0x46ac, 0x45c5, 0x462f, 0x46cb, 0x45de, 0x46c0, 0x46b6, 0x464b, 0x46a0, 0x467c, 0x46d4, 0x464b, 0x46d9, 0x464b, 0x46d7, 0x46b1 \
}
#define SJKVARN_CB3_V_INIT { \
    0x3526, 0x3046, 0x3501, 0x3a83, 0x3382, 0x3bb3, 0x34a6, 0x39f2, 0x33ca, 0x3868, 0x3cb6, 0x3478, 0x3cf8, 0x35dc, 0x3d95, 0x3b96, \
    0x3816, 0x3cfc, 0x3a95, 0x3dc1, 0x3b5e, 0x3e00, 0x3b57, 0x3e41, 0x3c6a, 0x34a9, 0x3dc6, 0x346b, 0x3e67, 0x35e5, 0x3e55, 0x37d5, \
    0x352d, 0x3cee, 0x3b17, 0x3e19, 0x3c44, 0x3f6e, 0x3c80, 0x3f12, 0x3bda, 0x3dda, 0x3f57, 0x37db, 0x3e63, 0x3613, 0x3f18, 0x3bdc, \
    0x3e0d, 0x3647, 0x3d24, 0x3ff2, 0x3c43, 0x3ec6, 0x4091, 0x3ef2, 0x3a47, 0x3eae, 0x39b4, 0x400b, 0x3dcb, 0x405e, 0x3b85, 0x4089, \
    0x3d61, 0x3b4e, 0x3d42, 0x3eab, 0x3ccf, 0x3f6c, 0x3d4b, 0x3ef5, 0x3c66, 0x3e0c, 0x3f60, 0x3e46, 0x401a, 0x3e8d, 0x404f, 0x3fb6, \
    0x3de4, 0x3fde, 0x3eda, 0x4072, 0x3f44, 0x40b2, 0x3fa1, 0x40ba, 0x3f7e, 0x3ed6, 0x409f, 0x3f94, 0x40fc, 0x3fef, 0x4116, 0x4064, \
    0x3e82, 0x4027, 0x3fd7, 0x40f1, 0x4033, 0x4150, 0x4068, 0x4150, 0x3ff1, 0x4071, 0x4119, 0x4030, 0x4157, 0x4073, 0x417d, 0x40c5, \
    0x40bc, 0x3f9d, 0x40b7, 0x417f, 0x4092, 0x417c, 0x41d6, 0x41a1, 0x4015, 0x4135, 0x4012, 0x41b5, 0x40d7, 0x4241, 0x4107, 0x425b, \
    0x4041, 0x3e41, 0x3ff7, 0x4070, 0x3f75, 0x4106, 0x4007, 0x40e2, 0x3f06, 0x404c, 0x4081, 0x4075, 0x412a, 0x4087, 0x415c, 0x4135, \
    0x403f, 0x4107, 0x406b, 0x418d, 0x4098, 0x41d6, 0x40e5, 0x41e2, 0x40d3, 0x40b6, 0x41b3, 0x40d6, 0x4208, 0x410a, 0x4236, 0x41d7, \
    0x40ba, 0x414c, 0x4130, 0x4226, 0x414c, 0x4278, 0x419b, 0x4284, 0x414a, 0x4188, 0x4217, 0x4182, 0x4285, 0x41c1, 0x42bb, 0x4261, \
    0x4205, 0x4127, 0x4214, 0x427e, 0x41eb, 0x4300, 0x4297, 0x4312, 0x41a6, 0x428b, 0x417b, 0x42fc, 0x4228, 0x4389, 0x4278, 0x43af, \
    0x419a, 0x4035, 0x4141, 0x4135, 0x40f6, 0x4237, 0x413a, 0x422d, 0x408f, 0x4180, 0x40f7, 0x41d7, 0x41a8, 0x41ec, 0x4226, 0x425a, \
    0x4158, 0x41d0, 0x4164, 0x4285, 0x418c, 0x42d1, 0x41ec, 0x42db, 0x41af, 0x41dd, 0x42a0, 0x41cf, 0x4300, 0x4211, 0x432e, 0x430d, \
    0x41fd, 0x4214, 0x4252, 0x4318, 0x4241, 0x4373, 0x4299, 0x4384, 0x4257, 0x429c, 0x42de, 0x42b6, 0x437d, 0x42bd, 0x43ba, 0x43b8, \
    0x4317, 0x4243, 0x435c, 0x430e, 0x431c, 0x440e, 0x436e, 0x4423, 0x42fa, 0x43b1, 0x42af, 0x440f, 0x434b, 0x4457, 0x43af, 0x4469, \
    0x42df, 0x4146, 0x427b, 0x41d4, 0x4235, 0x435b, 0x425b, 0x436f, 0x419d, 0x42a0, 0x41d0, 0x42fa, 0x424e, 0x4339, 0x42aa, 0x436c, \
    0x427b, 0x4262, 0x4271, 0x438e, 0x428c, 0x43dd, 0x42f7, 0x43e4, 0x429a, 0x4308, 0x437c, 0x42c0, 0x43f8, 0x4306, 0x4411, 0x440d, \
    0x432d, 0x42d2, 0x436b, 0x43ef, 0x4332, 0x4436, 0x438d, 0x4441, 0x4356, 0x43d5, 0x436f, 0x4406, 0x4421, 0x43b6, 0x4450, 0x447c, \
    0x4410, 0x434a, 0x4444, 0x43b7, 0x4421, 0x4496, 0x442e, 0x44b2, 0x4423, 0x4462, 0x43d4, 0x4496, 0x442d, 0x44de, 0x4467, 0x44f4, \
    0x441b, 0x4264, 0x4398, 0x42ae, 0x4364, 0x4442, 0x4385, 0x445b, 0x42d3, 0x43d3, 0x42d1, 0x440c, 0x4328, 0x442e, 0x4392, 0x4441, \
    0x43bb, 0x432e, 0x438b, 0x444b, 0x4393, 0x4473, 0x4401, 0x448a, 0x437f, 0x441c, 0x441d, 0x439e, 0x4477, 0x43ea, 0x4489, 0x4477, \
    0x442b, 0x43b6, 0x4448, 0x445f, 0x440a, 0x44ba, 0x443f, 0x44be, 0x4428, 0x4488, 0x441b, 0x44b2, 0x4474, 0x4459, 0x44cb, 0x4512, \
    0x4496, 0x442c, 0x44cb, 0x444d, 0x44a7, 0x4519, 0x44ad, 0x453d, 0x44d2, 0x44f1, 0x447a, 0x4522, 0x44b0, 0x456b, 0x44f4, 0x4583, \
    0x44e3, 0x43c0, 0x4479, 0x43d6, 0x4461, 0x44f9, 0x4475, 0x4529, 0x4413, 0x4489, 0x43ff, 0x44a5, 0x441f, 0x44c6, 0x4451, 0x44e1, \
    0x4498, 0x4422, 0x446c, 0x44f5, 0x446a, 0x4512, 0x44a1, 0x4543, 0x4457, 0x44c8, 0x4499, 0x4461, 0x44ef, 0x4486, 0x4518, 0x44fb, \
    0x44de, 0x4461, 0x44f9, 0x44dd, 0x4486, 0x453b, 0x44c7, 0x455c, 0x44ae, 0x4514, 0x4499, 0x454d, 0x44ef, 0x44cf, 0x4542, 0x45b8, \
    0x4531, 0x44c7, 0x4551, 0x44d8, 0x4541, 0x45bb, 0x453e, 0x45d6, 0x458f, 0x4596, 0x451f, 0x45bc, 0x454a, 0x4603, 0x4592, 0x4615, \
    0x45fc, 0x44d1, 0x4591, 0x44b5, 0x4556, 0x466f, 0x4568, 0x4672, 0x450f, 0x4675, 0x4502, 0x45d6, 0x44f7, 0x45d5, 0x4513, 0x45d4, \
    0x45be, 0x4512, 0x459f, 0x46a9, 0x4570, 0x46a3, 0x4598, 0x46ac, 0x4535, 0x463c, 0x4518, 0x4553, 0x460c, 0x4579, 0x4628, 0x45bb, \
    0x462c, 0x4578, 0x4611, 0x45ba, 0x453c, 0x46b0, 0x459d, 0x465b, 0x4577, 0x46ac, 0x456e, 0x46bb, 0x45e8, 0x4577, 0x4625, 0x46d5, \
    0x4620, 0x4598, 0x4622, 0x458a, 0x4625, 0x46c4, 0x45de, 0x4683, 0x4686, 0x46ae, 0x4607, 0x46c4, 0x4613, 0x46d6, 0x4666, 0x46da \
}
#define SJKVARN_CB2_K_INIT { \
    0x3809, 0x30d9, 0x39a5, 0x358b, 0x3488, 0x317c, 0x3845, 0x3412, 0x39eb, 0x336d, 0x3a72, 0x3772, 0x3498, 0x3874, 0x3b98, 0x37ec, \
    0x32df, 0x3967, 0x3246, 0x3a29, 0x3981, 0x3892, 0x3ba0, 0x3a1d, 0x344e, 0x3a50, 0x3215, 0x3bcd, 0x3a52, 0x3ce3, 0x38c4, 0x3ca2, \
    0x38a3, 0x3bee, 0x37eb, 0x3c2d, 0x39f1, 0x3bc1, 0x3d0c, 0x38f7, 0x3b8c, 0x3c99, 0x3c0f, 0x3d9e, 0x357d, 0x3a71, 0x3cf5, 0x37af, \
    0x3ae4, 0x3537, 0x3b8b, 0x368f, 0x35e8, 0x3cb5, 0x3818, 0x3c55, 0x39bb, 0x3c37, 0x3deb, 0x3aeb, 0x3c3d, 0x3820, 0x3e25, 0x3c82, \
    0x3c30, 0x386f, 0x3cc8, 0x3b4a, 0x3a99, 0x38ed, 0x3c12, 0x3b69, 0x3cee, 0x3a96, 0x3cea, 0x3c90, 0x3ae0, 0x3cc2, 0x3da4, 0x3cd4, \
    0x39fd, 0x3c6d, 0x3a74, 0x3d53, 0x3c9b, 0x3c5e, 0x3d50, 0x3d72, 0x3afc, 0x3d43, 0x3aa2, 0x3dc4, 0x3cae, 0x3e9c, 0x3c67, 0x3e97, \
    0x3c69, 0x3dd3, 0x3bf1, 0x3e26, 0x3cd9, 0x3df8, 0x3ebb, 0x3cb9, 0x3d83, 0x3e98, 0x3dd7, 0x3f62, 0x3d35, 0x3d37, 0x3f28, 0x3d76, \
    0x3e0f, 0x3c0e, 0x3e33, 0x3d16, 0x3c97, 0x3ed8, 0x3d53, 0x3ebc, 0x3dba, 0x3e6f, 0x3fed, 0x3e4e, 0x3ea1, 0x3d87, 0x4015, 0x3f2f, \
    0x3e5f, 0x3c11, 0x3f09, 0x3d65, 0x3d0c, 0x3bfe, 0x3d7e, 0x3dad, 0x3f0f, 0x3d52, 0x3e34, 0x3eca, 0x3d81, 0x3f44, 0x3e8a, 0x3f75, \
    0x3cc5, 0x3e26, 0x3d4d, 0x3f6e, 0x3e36, 0x3ca4, 0x3e6b, 0x3f50, 0x3d46, 0x3ee6, 0x3d59, 0x3f43, 0x3e67, 0x4039, 0x3e62, 0x401a, \
    0x3e24, 0x3fb1, 0x3d89, 0x3f5b, 0x3e4d, 0x3f94, 0x4051, 0x3e77, 0x3f41, 0x3e48, 0x3f58, 0x4069, 0x3f9d, 0x3eb6, 0x4076, 0x3f51, \
    0x3ffd, 0x3de4, 0x3f84, 0x3fad, 0x3e9f, 0x407c, 0x3f70, 0x4048, 0x400e, 0x3fa4, 0x40c5, 0x3ff3, 0x400d, 0x3fea, 0x40d7, 0x4077, \
    0x409c, 0x3e28, 0x410d, 0x3fc8, 0x3f5a, 0x3dcf, 0x3f66, 0x4022, 0x412f, 0x3fdb, 0x3f80, 0x40b9, 0x4005, 0x4141, 0x404a, 0x412b, \
    0x3eca, 0x4064, 0x3f45, 0x410f, 0x4026, 0x3e6c, 0x4055, 0x4140, 0x3f0d, 0x4078, 0x3f28, 0x4078, 0x404b, 0x416d, 0x4036, 0x4145, \
    0x4048, 0x416e, 0x3f6b, 0x40a2, 0x4043, 0x40cb, 0x4199, 0x403a, 0x40b1, 0x4005, 0x4086, 0x417c, 0x4139, 0x4047, 0x4174, 0x4099, \
    0x413a, 0x3fc3, 0x4084, 0x4143, 0x4063, 0x41a3, 0x40db, 0x4158, 0x417e, 0x40b4, 0x41be, 0x40e5, 0x40dd, 0x4143, 0x41a5, 0x4173 \
}
#define SJKVARN_CB2_V_INIT { \
    0x389b, 0x3071, 0x3840, 0x398d, 0x3282, 0x38b4, 0x3987, 0x352b, 0x3a99, 0x3331, 0x39ad, 0x3bb4, 0x36b9, 0x3aac, 0x3c92, 0x38ad, \
    0x33c0, 0x3ab3, 0x349e, 0x3c67, 0x38bf, 0x3be3, 0x3993, 0x39dc, 0x353a, 0x3a54, 0x346e, 0x3c46, 0x3b49, 0x3d1b, 0x3951, 0x3d6b, \
    0x3b19, 0x383e, 0x3c3d, 0x38c0, 0x39e8, 0x3d13, 0x3c1f, 0x3c2e, 0x3bc4, 0x3d98, 0x3a64, 0x3d85, 0x3815, 0x3c3c, 0x3d0a, 0x3b37, \
    0x3c14, 0x34c3, 0x39ea, 0x3cfc, 0x389c, 0x3caa, 0x3cd3, 0x39de, 0x3c82, 0x3a1e, 0x3e1c, 0x3ccb, 0x3d85, 0x3a8f, 0x3d75, 0x3e56, \
    0x3c3f, 0x37fc, 0x3c5e, 0x3c75, 0x3984, 0x3b1f, 0x3c6e, 0x3c4c, 0x3d0f, 0x3a0b, 0x3cad, 0x3db0, 0x3bd3, 0x3d21, 0x3e5b, 0x3c85, \
    0x39e7, 0x3ccd, 0x3acc, 0x3deb, 0x3c2c, 0x3d90, 0x3c74, 0x3d13, 0x3b46, 0x3cf3, 0x3b30, 0x3dda, 0x3d81, 0x3e36, 0x3ce6, 0x3ed9, \
    0x3d57, 0x3bb0, 0x3dc7, 0x3c67, 0x3cc2, 0x3e7c, 0x3d81, 0x3e54, 0x3d7a, 0x3f15, 0x3d51, 0x3efe, 0x3c85, 0x3e65, 0x3e82, 0x3e68, \
    0x3e2b, 0x3b5d, 0x3d33, 0x3ed8, 0x3cbc, 0x3eb1, 0x3e51, 0x3d7f, 0x3e56, 0x3d1a, 0x3fef, 0x3eef, 0x3f53, 0x3d85, 0x3f57, 0x4007, \
    0x3e2d, 0x3bc9, 0x3e79, 0x3d41, 0x3c81, 0x3cef, 0x3df6, 0x3ea7, 0x3ea7, 0x3cbd, 0x3de1, 0x3f11, 0x3d92, 0x3ec2, 0x4019, 0x3e2a, \
    0x3cae, 0x3e44, 0x3d7f, 0x3f8d, 0x3dbb, 0x3c37, 0x3dd6, 0x3df2, 0x3d53, 0x3ea2, 0x3d50, 0x3f45, 0x3f5e, 0x3efe, 0x3ecf, 0x4024, \
    0x3ef3, 0x3d4f, 0x3efa, 0x3e2f, 0x3de2, 0x3ff1, 0x3e95, 0x3fda, 0x3eb8, 0x4013, 0x3ea8, 0x4005, 0x3e53, 0x4025, 0x3f7a, 0x402b, \
    0x3fe0, 0x3daf, 0x3f00, 0x4040, 0x3e50, 0x4036, 0x3f2d, 0x3f5c, 0x3fab, 0x3f04, 0x40b2, 0x4033, 0x4086, 0x3f3a, 0x406d, 0x40ca, \
    0x4057, 0x3dae, 0x40b3, 0x3f25, 0x3e8b, 0x3ec7, 0x3ff8, 0x40bd, 0x4082, 0x3e5a, 0x3f76, 0x4078, 0x3f77, 0x406b, 0x4157, 0x4006, \
    0x3e6e, 0x3fea, 0x3fde, 0x4113, 0x403a, 0x3e4e, 0x3fb8, 0x401c, 0x3f2f, 0x4066, 0x3f0b, 0x40ab, 0x40fc, 0x4012, 0x406e, 0x4170, \
    0x4075, 0x3edc, 0x4042, 0x3ff9, 0x3f63, 0x4147, 0x4062, 0x4144, 0x4004, 0x40e9, 0x3fbc, 0x408e, 0x4016, 0x414b, 0x4076, 0x4100, \
    0x410a, 0x3f7d, 0x408d, 0x4157, 0x4049, 0x4163, 0x4034, 0x40bd, 0x40cb, 0x4093, 0x41ab, 0x4127, 0x4188, 0x407d, 0x415f, 0x41b1 \
}
/* ---- END GENERATED CODEBOOKS ---- */

/* tq6_0: 64 Lloyd-Max centroids for N(0, 1/128) and the 63 decision midpoints */
#define SJKVARN_TQ6_CENTROIDS_INIT { \
    -0.330935f, -0.286417f, -0.257865f, -0.236198f, -0.218435f, -0.203203f, -0.189753f, -0.177626f, \
    -0.166522f, -0.156230f, -0.146600f, -0.137517f, -0.128895f, -0.120663f, -0.112765f, -0.105157f, \
    -0.097801f, -0.090663f, -0.083717f, -0.076940f, -0.070310f, -0.063809f, -0.057422f, -0.051133f, \
    -0.044929f, -0.038798f, -0.032729f, -0.026710f, -0.020733f, -0.014787f, -0.008863f, -0.002953f, \
     0.002953f,  0.008863f,  0.014787f,  0.020733f,  0.026710f,  0.032729f,  0.038798f,  0.044929f, \
     0.051133f,  0.057422f,  0.063809f,  0.070310f,  0.076940f,  0.083717f,  0.090663f,  0.097801f, \
     0.105157f,  0.112765f,  0.120663f,  0.128895f,  0.137517f,  0.146600f,  0.156230f,  0.166522f, \
     0.177626f,  0.189753f,  0.203203f,  0.218435f,  0.236198f,  0.257865f,  0.286417f,  0.330935f }
#define SJKVARN_TQ6_MID_INIT { \
    -0.308676f, -0.272141f, -0.247031f, -0.227316f, -0.210819f, -0.196478f, -0.183690f, -0.172074f, \
    -0.161376f, -0.151415f, -0.142059f, -0.133206f, -0.124779f, -0.116714f, -0.108961f, -0.101479f, \
    -0.094232f, -0.087190f, -0.080329f, -0.073625f, -0.067060f, -0.060616f, -0.054277f, -0.048031f, \
    -0.041864f, -0.035763f, -0.029719f, -0.023722f, -0.017760f, -0.011825f, -0.005908f,  0.000000f, \
     0.005908f,  0.011825f,  0.017760f,  0.023722f,  0.029719f,  0.035763f,  0.041864f,  0.048031f, \
     0.054277f,  0.060616f,  0.067060f,  0.073625f,  0.080329f,  0.087190f,  0.094232f,  0.101479f, \
     0.108961f,  0.116714f,  0.124779f,  0.133206f,  0.142059f,  0.151415f,  0.161376f,  0.172074f, \
     0.183690f,  0.196478f,  0.210819f,  0.227316f,  0.247031f,  0.272141f,  0.308676f }

static const uint16_t sj_kvarn__cb3_k_h[512] = SJKVARN_CB3_K_INIT;
static const uint16_t sj_kvarn__cb3_v_h[512] = SJKVARN_CB3_V_INIT;
static const uint16_t sj_kvarn__cb2_k_h[256] = SJKVARN_CB2_K_INIT;
static const uint16_t sj_kvarn__cb2_v_h[256] = SJKVARN_CB2_V_INIT;
static const float    sj_kvarn__tq6_c_h[64]  = SJKVARN_TQ6_CENTROIDS_INIT;
static const float    sj_kvarn__tq6_m_h[63]  = SJKVARN_TQ6_MID_INIT;
#if defined(__CUDACC__)
static __device__ const uint16_t sj_kvarn__cb3_k_d[512] = SJKVARN_CB3_K_INIT;
static __device__ const uint16_t sj_kvarn__cb3_v_d[512] = SJKVARN_CB3_V_INIT;
static __device__ const uint16_t sj_kvarn__cb2_k_d[256] = SJKVARN_CB2_K_INIT;
static __device__ const uint16_t sj_kvarn__cb2_v_d[256] = SJKVARN_CB2_V_INIT;
static __device__ const float    sj_kvarn__tq6_c_d[64]  = SJKVARN_TQ6_CENTROIDS_INIT;
static __device__ const float    sj_kvarn__tq6_m_d[63]  = SJKVARN_TQ6_MID_INIT;
#endif

static SJ_KVARN_HD const uint16_t * sj_kvarn__cb(int bits, int is_v) {
#if defined(__CUDA_ARCH__)
    return bits == 3 ? (is_v ? sj_kvarn__cb3_v_d : sj_kvarn__cb3_k_d) : (is_v ? sj_kvarn__cb2_v_d : sj_kvarn__cb2_k_d);
#else
    return bits == 3 ? (is_v ? sj_kvarn__cb3_v_h : sj_kvarn__cb3_k_h) : (is_v ? sj_kvarn__cb2_v_h : sj_kvarn__cb2_k_h);
#endif
}
static SJ_KVARN_HD const float * sj_kvarn__tq6_c(void) {
#if defined(__CUDA_ARCH__)
    return sj_kvarn__tq6_c_d;
#else
    return sj_kvarn__tq6_c_h;
#endif
}
static SJ_KVARN_HD const float * sj_kvarn__tq6_m(void) {
#if defined(__CUDA_ARCH__)
    return sj_kvarn__tq6_m_d;
#else
    return sj_kvarn__tq6_m_h;
#endif
}

/* ---- single-rounding arithmetic (no contraction on any compiler) ---------- */

#if defined(__CUDA_ARCH__)
#define SJ__FADD(a, b) __fadd_rn((a), (b))
#define SJ__FSUB(a, b) __fsub_rn((a), (b))
#define SJ__FMUL(a, b) __fmul_rn((a), (b))
#define SJ__FDIV(a, b) __fdiv_rn((a), (b))
#define SJ__DADD(a, b) __dadd_rn((a), (b))
#define SJ__DSUB(a, b) __dsub_rn((a), (b))
#define SJ__DMUL(a, b) __dmul_rn((a), (b))
#define SJ__DDIV(a, b) __ddiv_rn((a), (b))
#define SJ__RINT(x)    rintf(x)
#else
static float  sj_kvarn__fadd(float a, float b)   { volatile float  r = a + b; return r; }
static float  sj_kvarn__fsub(float a, float b)   { volatile float  r = a - b; return r; }
static float  sj_kvarn__fmul(float a, float b)   { volatile float  r = a * b; return r; }
static float  sj_kvarn__fdiv(float a, float b)   { volatile float  r = a / b; return r; }
static double sj_kvarn__dadd(double a, double b) { volatile double r = a + b; return r; }
static double sj_kvarn__dsub(double a, double b) { volatile double r = a - b; return r; }
static double sj_kvarn__dmul(double a, double b) { volatile double r = a * b; return r; }
static double sj_kvarn__ddiv(double a, double b) { volatile double r = a / b; return r; }
#define SJ__FADD(a, b) sj_kvarn__fadd((a), (b))
#define SJ__FSUB(a, b) sj_kvarn__fsub((a), (b))
#define SJ__FMUL(a, b) sj_kvarn__fmul((a), (b))
#define SJ__FDIV(a, b) sj_kvarn__fdiv((a), (b))
#define SJ__DADD(a, b) sj_kvarn__dadd((a), (b))
#define SJ__DSUB(a, b) sj_kvarn__dsub((a), (b))
#define SJ__DMUL(a, b) sj_kvarn__dmul((a), (b))
#define SJ__DDIV(a, b) sj_kvarn__ddiv((a), (b))
#define SJ__RINT(x)    nearbyintf(x) /* ties to even in the default rounding mode */
#endif

static SJ_KVARN_HD float sj_kvarn__minf(float a, float b) { return b < a ? b : a; }
static SJ_KVARN_HD float sj_kvarn__maxf(float a, float b) { return a < b ? b : a; }
static SJ_KVARN_HD double sj_kvarn__mind(double a, double b) { return b < a ? b : a; }
static SJ_KVARN_HD double sj_kvarn__maxd(double a, double b) { return a < b ? b : a; }

/* ---- fp16 ------------------------------------------------------------------ */

static SJ_KVARN_HD float sj_kvarn__bits_f(uint32_t w) { float f; memcpy(&f, &w, 4); return f; }
static SJ_KVARN_HD uint32_t sj_kvarn__f_bits(float f) { uint32_t w; memcpy(&w, &f, 4); return w; }

/* The same integer/float code runs on host and device. The device intrinsic
 * __float2half_rn gives the same value, but with CUDA 12.4 ptxas, a byte-wise
 * memcpy of its result was compiled as a numeric conversion (F2I.U8.F16), so the
 * low byte of every stored half was wrong. */
SJKVARN_DEF SJ_KVARN_HD uint16_t sj_kvarn_f32_to_f16(float f) {
    /* exact round-to-nearest-even conversion (subnormals, overflow to inf) */
    const float scale_to_inf  = sj_kvarn__bits_f(UINT32_C(0x77800000));
    const float scale_to_zero = sj_kvarn__bits_f(UINT32_C(0x08800000));
    volatile float base = SJ__FMUL(SJ__FMUL(fabsf(f), scale_to_inf), scale_to_zero);
    const uint32_t w      = sj_kvarn__f_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign   = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }
    const float b2 = SJ__FADD(sj_kvarn__bits_f((bias >> 1) + UINT32_C(0x07800000)), base);
    const uint32_t bits     = sj_kvarn__f_bits(b2);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t man_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign  = exp_bits + man_bits;
    return (uint16_t) ((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT32_C(0x7E00) : nonsign));
}

SJKVARN_DEF SJ_KVARN_HD float sj_kvarn_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t w;
    if (exp == 0x1fu) {
        w = sign | 0x7f800000u | (man << 13);
    } else if (exp != 0) {
        w = sign | ((exp + 112u) << 23) | (man << 13);
    } else if (man == 0) {
        w = sign;
    } else {
        /* subnormal: normalize */
        int e = -1;
        do { ++e; man <<= 1; } while ((man & 0x400u) == 0);
        w = sign | ((uint32_t) (112 - e) << 23) | ((man & 0x3ffu) << 13);
    }
    return sj_kvarn__bits_f(w);
}

/* ---- rotation -------------------------------------------------------------- */

SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_hadamard(float * x, int n) {
    int h, i, j;
    for (h = 1; h < n; h <<= 1) {
        for (i = 0; i < n; i += 2*h) {
            for (j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j]     = SJ__FADD(a, b);
                x[j + h] = SJ__FSUB(a, b);
            }
        }
    }
    {
        const float s = SJ__FDIV(1.0f, sqrtf((float) n));
        for (i = 0; i < n; ++i) {
            x[i] = SJ__FMUL(x[i], s);
        }
    }
}

SJKVARN_DEF void sj_kvarn_rotate_heads(float * x, int n_heads, int head_dim) {
    int h;
    for (h = 0; h < n_heads; ++h) {
        sj_kvarn_hadamard(x + (size_t) h*head_dim, head_dim);
    }
}

/* ---- layout ---------------------------------------------------------------- */

SJKVARN_DEF int sj_kvarn_body_resolve(int body, int bits_k, int bits_v) {
    if (body == SJKVARN_BODY_AUTO) {
        const int lowk = bits_k == 2 || bits_k == 3, lowv = bits_v == 2 || bits_v == 3;
        return (lowk && lowv && bits_v <= bits_k) ? SJKVARN_BODY_TRELLIS : SJKVARN_BODY_SCALAR;
    }
    return body;
}

SJKVARN_DEF SJ_KVARN_HD int sj_kvarn_layout_init(sj_kvarn_layout * l, int D, int G, int bits_k, int bits_v, int body) {
    if (D < 32 || (D & (D - 1)) != 0 || G <= 0 || G % 16 != 0) return -1;
    if (bits_k < 2 || bits_k > 4 || bits_v < 2 || bits_v > 4) return -2;
    if (body != SJKVARN_BODY_SCALAR && body != SJKVARN_BODY_TRELLIS) return -3;
    if (body == SJKVARN_BODY_TRELLIS) {
        /* trellis bodies: the 3/3, 3/2 and 2/2 pairs, token-axis sequences of exactly G = 128 */
        if (bits_k == 4 || bits_v == 4 || bits_v > bits_k || G != SJKVARN_TRELLIS_SEQ || D % SJKVARN_TRELLIS_SEQ != 0) return -4;
    }
    l->D = D; l->G = G; l->bits_k = bits_k; l->bits_v = bits_v; l->body = body;
    l->k_payload = 0;
    l->v_payload = l->k_payload + (uint32_t) (D*bits_k/8) * (uint32_t) G;
    l->k_scale   = l->v_payload + (uint32_t) (D*bits_v/8) * (uint32_t) G;
    l->k_zero    = l->k_scale + 2u*(uint32_t) D;
    l->k_tok     = l->k_zero  + 2u*(uint32_t) D;
    l->v_ch      = l->k_tok   + 2u*(uint32_t) G;
    l->v_scale   = l->v_ch    + 2u*(uint32_t) D;
    l->v_zero    = l->v_scale + 2u*(uint32_t) G;
    l->bytes     = l->v_zero  + 2u*(uint32_t) G;
    return 0;
}

SJKVARN_DEF size_t sj_kvarn_record_bytes(int D, int G, int bits_k, int bits_v) {
    return (size_t) G*D*bits_k/8 + (size_t) G*D*bits_v/8 + 2*(3*(size_t) D + 3*(size_t) G);
}

SJKVARN_DEF double sj_kvarn_body_bits_per_element(const sj_kvarn_layout * l) {
    return 8.0 * (double) l->bytes / (2.0 * (double) l->D * (double) l->G);
}

/* 4-bit scalar payloads use "fragment order" (the tensor-core operand order of a
 * 16x16 tile; see README "Payload order"); every other width, and every trellis
 * payload, uses a plain bit stream. */
static SJ_KVARN_HD int sj_kvarn__frag(const sj_kvarn_layout * l, int bits) {
    return l->body == SJKVARN_BODY_SCALAR && bits == 4 && l->D % 16 == 0 && l->G % 16 == 0;
}

SJKVARN_DEF SJ_KVARN_HD uint32_t sj_kvarn_code_bit(const sj_kvarn_layout * l, int t, int d, int is_v) {
    const int D = l->D, G = l->G, bits = is_v ? l->bits_v : l->bits_k;
    if (l->body == SJKVARN_BODY_TRELLIS) {
        /* channel-major stream: the 128-token sequence of channel d is contiguous */
        return ((uint32_t) d*(uint32_t) G + (uint32_t) t)*(uint32_t) bits;
    }
    if (!sj_kvarn__frag(l, bits)) {
        /* token-major stream */
        return (uint32_t) t*(uint32_t) (D*bits) + (uint32_t) d*(uint32_t) bits;
    }
    {
        const int s = t >> 4, tt = t & 15, c = d >> 4, dd = d & 15;
        int lane, r, e;
        uint32_t word;
        if (!is_v) {
            lane = 4*(tt & 7) + ((dd >> 1) & 3);
            r    = (tt >> 3) + 2*(dd >> 3);
            e    = dd & 1;
        } else {
            lane = 4*(dd & 7) + ((tt >> 1) & 3);
            r    = (dd >> 3) + 2*(tt >> 3);
            e    = tt & 1;
        }
        word = (uint32_t) (s*(D/16) + c)*32u + (uint32_t) lane;
        return word*32u + (uint32_t) (r + 4*e)*4u;
    }
}

SJKVARN_DEF SJ_KVARN_HD int sj_kvarn_k_ch_idx(const sj_kvarn_layout * l, int d) {
    if (!sj_kvarn__frag(l, l->bits_k)) return d;
    {
        const int c = d >> 4, dd = d & 15, jj = dd >> 1, e = dd & 1;
        return (jj & 3)*(l->D/4) + c*4 + (jj >> 2)*2 + e;
    }
}

SJKVARN_DEF SJ_KVARN_HD int sj_kvarn_v_ch_idx(const sj_kvarn_layout * l, int d) {
    if (!sj_kvarn__frag(l, l->bits_v)) return d;
    {
        const int c = d >> 4, dd = d & 15;
        return (dd & 7)*(l->D/8) + c*2 + (dd >> 3);
    }
}

static SJ_KVARN_HD void sj_kvarn__pack(uint8_t * p, uint32_t bit, int bits, uint32_t v) {
    const uint32_t byte = bit >> 3, shift = bit & 7;
    const uint32_t w = (v & ((1u << bits) - 1u)) << shift;
    p[byte] |= (uint8_t) w;
    if (shift + (uint32_t) bits > 8) {
        p[byte + 1] |= (uint8_t) (w >> 8);
    }
}

static SJ_KVARN_HD uint32_t sj_kvarn__unpack(const uint8_t * p, uint32_t bit, int bits) {
    const uint32_t byte = bit >> 3, shift = bit & 7;
    uint32_t w = p[byte];
    if (shift + (uint32_t) bits > 8) {
        w |= (uint32_t) p[byte + 1] << 8;
    }
    return (w >> shift) & ((1u << bits) - 1u);
}

static SJ_KVARN_HD void sj_kvarn__put_half(uint8_t * rec, uint32_t off, float v) {
    const uint16_t h = sj_kvarn_f32_to_f16(v);
    memcpy(rec + off, &h, 2);
}
static SJ_KVARN_HD float sj_kvarn__get_half(const uint8_t * rec, uint32_t off) {
    uint16_t h;
    memcpy(&h, rec + off, 2);
    return sj_kvarn_f16_to_f32(h);
}

/* Trellis window (codebook index) of the code at stream position (seq, i): the
 * code in the top `bits` bits over the 6 stream bits before it (older codes
 * lower); history that would cross the start of the sequence reads as zero. */
static SJ_KVARN_HD uint32_t sj_kvarn__window(const uint8_t * payload, int bits, int seq, int i, int seq_len) {
    const uint32_t bit  = ((uint32_t) seq*(uint32_t) seq_len + (uint32_t) i)*(uint32_t) bits;
    const uint32_t mask = (1u << (SJKVARN_TRELLIS_HIST_BITS + bits)) - 1u;
    const int hist = (SJKVARN_TRELLIS_HIST_BITS + bits - 1)/bits;
    const int h = i & (SJKVARN_TRELLIS_SEQ - 1);
    if (h >= hist) {
        const uint32_t bs = bit - SJKVARN_TRELLIS_HIST_BITS, byte = bs >> 3, shift = bs & 7;
        const uint32_t v = (uint32_t) payload[byte] | ((uint32_t) payload[byte + 1] << 8);
        return (v >> shift) & mask;
    }
    {
        uint32_t w = sj_kvarn__unpack(payload, bit, bits) << SJKVARN_TRELLIS_HIST_BITS;
        int k;
        for (k = 1; k <= h; ++k) {
            w |= sj_kvarn__unpack(payload, bit - (uint32_t) (k*bits), bits) << (SJKVARN_TRELLIS_HIST_BITS - k*bits);
        }
        return w;
    }
}

/* ---- seal ------------------------------------------------------------------ */

static SJ_KVARN_HD size_t sj_kvarn__align8(size_t x) { return (x + 7) & ~(size_t) 7; }

typedef struct sj_kvarn__ws {
    float * tile, * bal, * s_row, * s_col, * q_lo, * q_step, * y, * cbf, * M, * Mn;
    double * log_a, * log_b;
    uint8_t * codes, * arg;
} sj_kvarn__ws;

static SJ_KVARN_HD size_t sj_kvarn__ws_carve(const sj_kvarn_layout * l, void * base, sj_kvarn__ws * w) {
    const size_t DG = (size_t) l->D*l->G, R = (size_t) (l->D > l->G ? l->D : l->G);
    uint8_t * p = (uint8_t *) base;
    size_t off = 0;
#define SJ__CARVE(field, type, n) do { if (w) w->field = (type *) (p + off); off = sj_kvarn__align8(off + sizeof(type)*(n)); } while (0)
    SJ__CARVE(tile,   float,  DG);
    SJ__CARVE(bal,    float,  DG);
    SJ__CARVE(s_row,  float,  R);
    SJ__CARVE(s_col,  float,  R);
    SJ__CARVE(q_lo,   float,  R);
    SJ__CARVE(q_step, float,  R);
    SJ__CARVE(log_a,  double, R);
    SJ__CARVE(log_b,  double, R);
    SJ__CARVE(y,      float,  SJKVARN_TRELLIS_SEQ);
    SJ__CARVE(cbf,    float,  512);
    SJ__CARVE(M,      float,  SJKVARN_TRELLIS_NSTATE);
    SJ__CARVE(Mn,     float,  SJKVARN_TRELLIS_NSTATE);
    SJ__CARVE(codes,  uint8_t, SJKVARN_TRELLIS_SEQ);
    SJ__CARVE(arg,    uint8_t, (size_t) SJKVARN_TRELLIS_SEQ*SJKVARN_TRELLIS_NSTATE);
#undef SJ__CARVE
    return off;
}

SJKVARN_DEF SJ_KVARN_HD size_t sj_kvarn_seal_workspace_bytes(const sj_kvarn_layout * l) {
    return sj_kvarn__ws_carve(l, NULL, NULL);
}

/* Reduction contract: a vector of n values (n % 8 == 0) is split into 8
 * contiguous chunks, each chunk is summed sequentially in double, and the chunk
 * partials are combined as ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)). The second
 * moment uses an explicit double fma. GPU sealers must follow the same order. */
static SJ_KVARN_HD double sj_kvarn__reduce8(const double * p) {
    double q[SJKVARN_RED_LANES];
    int i, w;
    for (i = 0; i < SJKVARN_RED_LANES; ++i) q[i] = p[i];
    for (w = 1; w < SJKVARN_RED_LANES; w <<= 1) {
        for (i = 0; i < SJKVARN_RED_LANES; i += 2*w) {
            q[i] = SJ__DADD(q[i], q[i + w]);
        }
    }
    return q[0];
}

/* sample standard deviation (n - 1) of a strided sequence */
static SJ_KVARN_HD float sj_kvarn__sample_std(const float * x, int n, size_t stride) {
    const int chunk = n / SJKVARN_RED_LANES;
    double p[SJKVARN_RED_LANES], mean, m2;
    int l, i;
    for (l = 0; l < SJKVARN_RED_LANES; ++l) {
        double acc = 0.0;
        for (i = 0; i < chunk; ++i) {
            acc = SJ__DADD(acc, (double) x[(size_t) (l*chunk + i)*stride]);
        }
        p[l] = acc;
    }
    mean = SJ__DDIV(sj_kvarn__reduce8(p), (double) n);
    for (l = 0; l < SJKVARN_RED_LANES; ++l) {
        double acc = 0.0;
        for (i = 0; i < chunk; ++i) {
            const double d = SJ__DSUB((double) x[(size_t) (l*chunk + i)*stride], mean);
            acc = fma(d, d, acc);
        }
        p[l] = acc;
    }
    m2 = sj_kvarn__reduce8(p);
    return (float) sqrt(SJ__DDIV(m2, (double) (n > 1 ? n - 1 : 1)));
}

static SJ_KVARN_HD void sj_kvarn__recompute(const float * M, int R, int C, const float * s_row, const float * s_col, float * cur) {
    int r, c;
    for (r = 0; r < R; ++r) {
        for (c = 0; c < C; ++c) {
            cur[(size_t) r*C + c] = SJ__FDIV(SJ__FDIV(M[(size_t) r*C + c], s_col[c]), s_row[r]);
        }
    }
}

/* Log-domain variance balancing of a row-major [R][C] tile: exactly `iters`
 * alternating column/row passes. Afterwards cur = M / s_col / s_row. The final
 * scales are kept (the KVarN reference keeps the best iteration instead). */
static SJ_KVARN_HD void sj_kvarn__balance(const float * M, int R, int C, int iters, float * s_row, float * s_col,
                                         float * cur, double * log_r, double * log_c) {
    int it, r, c;
    for (c = 0; c < C; ++c) { s_col[c] = 1.0f; log_c[c] = 0.0; }
    for (r = 0; r < R; ++r) { s_row[r] = 1.0f; log_r[r] = 0.0; }
    sj_kvarn__recompute(M, R, C, s_row, s_col, cur);
    for (it = 0; it < iters; ++it) {
        for (c = 0; c < C; ++c) {
            const float sd = sj_kvarn__minf(sj_kvarn__maxf(sj_kvarn__sample_std(cur + c, R, (size_t) C), 1e-3f), 1e3f);
            log_c[c] = sj_kvarn__mind(sj_kvarn__maxd(SJ__DADD(log_c[c], log((double) sd)), -0.3), 10.0);
            s_col[c] = (float) exp(log_c[c]);
        }
        sj_kvarn__recompute(M, R, C, s_row, s_col, cur);
        for (r = 0; r < R; ++r) {
            const float sd = sj_kvarn__minf(sj_kvarn__maxf(sj_kvarn__sample_std(cur + (size_t) r*C, C, 1), 1e-3f), 1e3f);
            log_r[r] = sj_kvarn__mind(sj_kvarn__maxd(SJ__DADD(log_r[r], log((double) sd)), -0.3), 10.0);
            s_row[r] = (float) exp(log_r[r]);
        }
        sj_kvarn__recompute(M, R, C, s_row, s_col, cur);
    }
}

static SJ_KVARN_HD uint32_t sj_kvarn__rtn(float x, float lo, float step, uint32_t qmax) {
    float r = SJ__RINT(SJ__FDIV(SJ__FSUB(x, lo), step));
    if (r < 0.0f) r = 0.0f;
    if (r > (float) qmax) r = (float) qmax;
    return (uint32_t) r;
}

/* Viterbi search of one trellis sequence. y[n] are row-normalized targets, cbf
 * the codebook as float. Cost = sum of (y - cb[window])^2 with IEEE single
 * rounding per op, per-state minima with strict < over ascending branches (ties
 * pick the smaller code), start state 0, no termination. */
static SJ_KVARN_HD void sj_kvarn__viterbi(const float * y, int n, int bits, const float * cbf, uint8_t * codes,
                                         float * M, float * Mn, uint8_t * arg) {
    const int NS = SJKVARN_TRELLIS_NSTATE, NB = 1 << bits;
    int i, s, j;
    for (s = 0; s < NS; ++s) M[s] = 0.0f;
    for (i = n - 1; i >= 0; --i) {
        float * t;
        for (s = 0; s < NS; ++s) {
            float best = INFINITY;
            int bj = 0;
            for (j = 0; j < NB; ++j) {
                const int w = s + NS*j;
                const float d = SJ__FSUB(y[i], cbf[w]);
                const float e = SJ__FMUL(d, d);
                const float c = SJ__FADD(e, M[w >> bits]);
                if (c < best) { best = c; bj = j; }
            }
            Mn[s] = best;
            arg[(size_t) i*NS + s] = (uint8_t) bj;
        }
        t = M; M = Mn; Mn = t;
    }
    s = 0;
    for (i = 0; i < n; ++i) {
        j = arg[(size_t) i*NS + s];
        codes[i] = (uint8_t) j;
        s = (s + NS*j) >> bits;
    }
}

/* Trellis-code one side. bal is the balanced tile [R][C] (K: rows = channels,
 * V: rows = tokens); lo/step the per-row affine; one sequence per channel. */
static SJ_KVARN_HD void sj_kvarn__trellis_code(const sj_kvarn_layout * l, const float * bal, int C, const float * lo,
                                              const float * step, int is_v, uint8_t * payload, sj_kvarn__ws * w) {
    const int D = l->D, G = l->G, bits = is_v ? l->bits_v : l->bits_k, nwin = 1 << (SJKVARN_TRELLIS_HIST_BITS + bits);
    const uint16_t * cb = sj_kvarn__cb(bits, is_v);
    int d, t;
    for (d = 0; d < nwin; ++d) w->cbf[d] = sj_kvarn_f16_to_f32(cb[d]);
    for (d = 0; d < D; ++d) {
        for (t = 0; t < G; ++t) {
            const int r = is_v ? t : d, col = is_v ? d : t;
            w->y[t] = SJ__FDIV(SJ__FSUB(bal[(size_t) r*C + col], lo[r]), step[r]);
        }
        sj_kvarn__viterbi(w->y, G, bits, w->cbf, w->codes, w->M, w->Mn, w->arg);
        for (t = 0; t < G; ++t) {
            sj_kvarn__pack(payload, ((uint32_t) d*(uint32_t) G + (uint32_t) t)*(uint32_t) bits, bits, w->codes[t]);
        }
    }
}

/* Refit each row's affine after trellis coding by moment matching (mean and
 * standard deviation of the reconstruction equal those of the row). Sequential
 * double sums in column order. Degenerate rows keep the RTN affine. */
static SJ_KVARN_HD void sj_kvarn__trellis_refit(const sj_kvarn_layout * l, const float * bal, int R, int C,
                                               const float * s_row, int is_v, const uint8_t * payload, uint8_t * rec) {
    const int G = l->G, bits = is_v ? l->bits_v : l->bits_k;
    const uint16_t * cb = sj_kvarn__cb(bits, is_v);
    int r, c;
    for (r = 0; r < R; ++r) {
        double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, vy, vx, ad, bd, n = (double) C;
        for (c = 0; c < C; ++c) {
            const int t = is_v ? r : c, d = is_v ? c : r;
            const double x  = (double) bal[(size_t) r*C + c];
            const double yh = (double) sj_kvarn_f16_to_f32(cb[sj_kvarn__window(payload, bits, d, t, G)]);
            Sx  = SJ__DADD(Sx, x);
            Sy  = SJ__DADD(Sy, yh);
            Sxx = SJ__DADD(Sxx, SJ__DMUL(x, x));
            Syy = SJ__DADD(Syy, SJ__DMUL(yh, yh));
            /* (the least-squares variant also needs Sxy; moment matching does not) */
        }
        vy = SJ__DSUB(SJ__DMUL(n, Syy), SJ__DMUL(Sy, Sy));
        if (!(vy > 0.0)) continue;
        vx = SJ__DSUB(SJ__DMUL(n, Sxx), SJ__DMUL(Sx, Sx));
        if (!(vx > 0.0)) continue;
        ad = sqrt(SJ__DDIV(vx, vy));
        bd = SJ__DDIV(SJ__DSUB(Sx, SJ__DMUL(ad, Sy)), n);
        if (!(ad > 0.0) || !(bd == bd) || ad > 1e30 || fabs(bd) > 1e30) continue;
        {
            const float a = (float) ad, b = (float) bd;
            const int ri = is_v ? r : sj_kvarn_k_ch_idx(l, r);
            sj_kvarn__put_half(rec, (is_v ? l->v_scale : l->k_scale) + 2u*(uint32_t) ri, SJ__FMUL(s_row[r], a));
            sj_kvarn__put_half(rec, (is_v ? l->v_zero  : l->k_zero)  + 2u*(uint32_t) ri, SJ__FMUL(s_row[r], b));
        }
    }
}

/* Quantize one balanced side. K: bal [D][G], rows = channels. V: bal [G][D], rows = tokens. */
static SJ_KVARN_HD void sj_kvarn__quantize_side(const sj_kvarn_layout * l, int is_v, uint8_t * rec, sj_kvarn__ws * w) {
    const int D = l->D, G = l->G;
    const int R = is_v ? G : D, C = is_v ? D : G;
    const int bits = is_v ? l->bits_v : l->bits_k;
    const uint32_t qmax = (1u << bits) - 1u;
    const int trellis = l->body == SJKVARN_BODY_TRELLIS;
    uint8_t * payload = rec + (is_v ? l->v_payload : l->k_payload);
    int r, c;
    for (r = 0; r < R; ++r) {
        const float * row = w->bal + (size_t) r*C;
        float lo = row[0], hi = row[0], step;
        uint32_t ri;
        for (c = 1; c < C; ++c) { lo = sj_kvarn__minf(lo, row[c]); hi = sj_kvarn__maxf(hi, row[c]); }
        step = sj_kvarn__maxf(SJ__FDIV(SJ__FSUB(hi, lo), (float) qmax), 1e-10f);
        ri = (uint32_t) (is_v ? r : sj_kvarn_k_ch_idx(l, r));
        sj_kvarn__put_half(rec, (is_v ? l->v_scale : l->k_scale) + 2u*ri, SJ__FMUL(w->s_row[r], step));
        sj_kvarn__put_half(rec, (is_v ? l->v_zero  : l->k_zero)  + 2u*ri, SJ__FMUL(w->s_row[r], lo));
        if (trellis) {
            w->q_lo[r] = lo; w->q_step[r] = step;
            continue;
        }
        for (c = 0; c < C; ++c) {
            const int t = is_v ? r : c, d = is_v ? c : r;
            sj_kvarn__pack(payload, sj_kvarn_code_bit(l, t, d, is_v), bits, sj_kvarn__rtn(row[c], lo, step, qmax));
        }
    }
    if (trellis) {
        sj_kvarn__trellis_code(l, w->bal, C, w->q_lo, w->q_step, is_v, payload, w);
        sj_kvarn__trellis_refit(l, w->bal, R, C, w->s_row, is_v, payload, rec);
    }
    /* column scales: Ktok[t] (K columns = tokens), Vch[d] (V columns = channels) */
    for (c = 0; c < C; ++c) {
        const uint32_t off = is_v ? l->v_ch + 2u*(uint32_t) sj_kvarn_v_ch_idx(l, c) : l->k_tok + 2u*(uint32_t) c;
        sj_kvarn__put_half(rec, off, w->s_col[c]);
    }
}

SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_seal_group(const sj_kvarn_layout * l, const uint16_t * K, const uint16_t * V,
                                                 size_t row_stride, int iters, uint8_t * rec, void * work) {
    const int D = l->D, G = l->G;
    sj_kvarn__ws w;
    int t, d;
    sj_kvarn__ws_carve(l, work, &w);
    memset(rec, 0, l->bytes);

    /* K: tile [D][G] (rows = channels, cols = tokens) */
    for (t = 0; t < G; ++t) {
        for (d = 0; d < D; ++d) {
            w.tile[(size_t) d*G + t] = sj_kvarn_f16_to_f32(K[(size_t) t*row_stride + d]);
        }
    }
    sj_kvarn__balance(w.tile, D, G, iters, w.s_row, w.s_col, w.bal, w.log_a, w.log_b);
    sj_kvarn__quantize_side(l, 0, rec, &w);

    /* V: tile [G][D] (rows = tokens, cols = channels) */
    for (t = 0; t < G; ++t) {
        for (d = 0; d < D; ++d) {
            w.tile[(size_t) t*D + d] = sj_kvarn_f16_to_f32(V[(size_t) t*row_stride + d]);
        }
    }
    sj_kvarn__balance(w.tile, G, D, iters, w.s_row, w.s_col, w.bal, w.log_a, w.log_b);
    sj_kvarn__quantize_side(l, 1, rec, &w);
}

/* ---- decode ---------------------------------------------------------------- */

static SJ_KVARN_HD float sj_kvarn__q(const sj_kvarn_layout * l, const uint8_t * rec, int t, int d, int is_v) {
    const int bits = is_v ? l->bits_v : l->bits_k;
    const uint8_t * payload = rec + (is_v ? l->v_payload : l->k_payload);
    if (l->body == SJKVARN_BODY_TRELLIS) {
        return sj_kvarn_f16_to_f32(sj_kvarn__cb(bits, is_v)[sj_kvarn__window(payload, bits, d, t, l->G)]);
    }
    return (float) sj_kvarn__unpack(payload, sj_kvarn_code_bit(l, t, d, is_v), bits);
}

SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_decode_k_row(const sj_kvarn_layout * l, const uint8_t * rec, int t, float * out) {
    const float tok = sj_kvarn__get_half(rec, l->k_tok + 2u*(uint32_t) t);
    int d;
    for (d = 0; d < l->D; ++d) {
        const uint32_t di = (uint32_t) sj_kvarn_k_ch_idx(l, d);
        const float q = sj_kvarn__q(l, rec, t, d, 0);
        out[d] = SJ__FMUL(SJ__FADD(SJ__FMUL(q, sj_kvarn__get_half(rec, l->k_scale + 2u*di)),
                                   sj_kvarn__get_half(rec, l->k_zero + 2u*di)), tok);
    }
}

SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_decode_v_row(const sj_kvarn_layout * l, const uint8_t * rec, int t, float * out) {
    const float sc = sj_kvarn__get_half(rec, l->v_scale + 2u*(uint32_t) t);
    const float zp = sj_kvarn__get_half(rec, l->v_zero  + 2u*(uint32_t) t);
    int d;
    for (d = 0; d < l->D; ++d) {
        const float q = sj_kvarn__q(l, rec, t, d, 1);
        out[d] = SJ__FMUL(SJ__FADD(SJ__FMUL(q, sc), zp),
                          sj_kvarn__get_half(rec, l->v_ch + 2u*(uint32_t) sj_kvarn_v_ch_idx(l, d)));
    }
}

/* ---- staging --------------------------------------------------------------- */

static SJ_KVARN_HD int sj_kvarn__tq6_nearest(float v) {
    const float * mid = sj_kvarn__tq6_m();
    int lo = 0, hi = 63;
    while (lo < hi) {
        const int m = (lo + hi) / 2;
        if (v < mid[m]) hi = m; else lo = m + 1;
    }
    return lo;
}

SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_tq6_quantize_row(const float * x, sj_kvarn_tq6_block * y, int n) {
    const float * cent = sj_kvarn__tq6_c();
    int b, i;
    for (b = 0; b < n / SJKVARN_TQ6_QK; ++b) {
        const float * src = x + (size_t) b*SJKVARN_TQ6_QK;
        sj_kvarn_tq6_block * blk = y + b;
        float norm_sq = 0.0f, norm, rn_sq = 0.0f, rn, inv = 0.0f;
        uint8_t idx[SJKVARN_TQ6_QK];
        for (i = 0; i < SJKVARN_TQ6_QK; ++i) norm_sq = SJ__FADD(norm_sq, SJ__FMUL(src[i], src[i]));
        norm = sqrtf(norm_sq);
        if (norm > 1e-10f) inv = SJ__FDIV(1.0f, norm);
        for (i = 0; i < SJKVARN_TQ6_QK; ++i) {
            const float v = norm > 1e-10f ? SJ__FMUL(src[i], inv) : 0.0f;
            idx[i] = (uint8_t) sj_kvarn__tq6_nearest(v);
        }
        for (i = 0; i < SJKVARN_TQ6_QK; ++i) rn_sq = SJ__FADD(rn_sq, SJ__FMUL(cent[idx[i]], cent[idx[i]]));
        rn = sqrtf(rn_sq);
        blk->norm = sj_kvarn_f32_to_f16(rn > 1e-10f ? SJ__FDIV(norm, rn) : norm);
        memset(blk->qs, 0, sizeof(blk->qs));
        memset(blk->qh, 0, sizeof(blk->qh));
        for (i = 0; i < SJKVARN_TQ6_QK; ++i) {
            const uint8_t c = idx[i] & 0x3F;
            blk->qs[i / 2] |= (uint8_t) ((c & 0xF) << ((i % 2) * 4));
            blk->qh[i / 4] |= (uint8_t) (((c >> 4) & 0x3) << ((i % 4) * 2));
        }
    }
}

SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_tq6_dequantize_row(const sj_kvarn_tq6_block * y, float * x, int n) {
    const float * cent = sj_kvarn__tq6_c();
    int b, i;
    for (b = 0; b < n / SJKVARN_TQ6_QK; ++b) {
        const float norm = sj_kvarn_f16_to_f32(y[b].norm);
        for (i = 0; i < SJKVARN_TQ6_QK; ++i) {
            const uint8_t lo = (y[b].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            const uint8_t hi = (y[b].qh[i / 4] >> ((i % 4) * 2)) & 0x3;
            x[(size_t) b*SJKVARN_TQ6_QK + i] = SJ__FMUL(cent[lo | (hi << 4)], norm);
        }
    }
}

SJKVARN_DEF size_t sj_kvarn_stage_row_bytes(int staging, int D) {
    return staging == SJKVARN_STAGING_F16 ? (size_t) D*2 : (size_t) (D / SJKVARN_TQ6_QK)*SJKVARN_TQ6_BLOCK_BYTES;
}

SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_stage_row(int staging, const float * x_rot, void * dst, int D) {
    int i;
    if (staging == SJKVARN_STAGING_F16) {
        uint16_t * h = (uint16_t *) dst;
        for (i = 0; i < D; ++i) h[i] = sj_kvarn_f32_to_f16(x_rot[i]);
        return;
    }
    sj_kvarn_tq6_quantize_row(x_rot, (sj_kvarn_tq6_block *) dst, D);
}

SJKVARN_DEF SJ_KVARN_HD void sj_kvarn_unstage_row(int staging, const void * src, float * x_rot, int D) {
    int i;
    if (staging == SJKVARN_STAGING_F16) {
        const uint16_t * h = (const uint16_t *) src;
        for (i = 0; i < D; ++i) x_rot[i] = sj_kvarn_f16_to_f32(h[i]);
        return;
    }
    sj_kvarn_tq6_dequantize_row((const sj_kvarn_tq6_block *) src, x_rot, D);
}

/* ---- reference attention helpers -------------------------------------------- */

static float sj_kvarn__dot(const float * a, const float * b, int n) {
    float s = 0.0f;
    int i;
    for (i = 0; i < n; ++i) s = SJ__FADD(s, SJ__FMUL(a[i], b[i]));
    return s;
}

SJKVARN_DEF void sj_kvarn_record_scores(const sj_kvarn_layout * l, const uint8_t * rec, const float * q, int n, float * scores) {
    float row[1024];
    int t;
    for (t = 0; t < n; ++t) {
        sj_kvarn_decode_k_row(l, rec, t, row);
        scores[t] = sj_kvarn__dot(q, row, l->D);
    }
}

SJKVARN_DEF void sj_kvarn_record_accum_v(const sj_kvarn_layout * l, const uint8_t * rec, const float * w, int n, float * acc) {
    float row[1024];
    int t, d;
    for (t = 0; t < n; ++t) {
        sj_kvarn_decode_v_row(l, rec, t, row);
        for (d = 0; d < l->D; ++d) acc[d] = SJ__FADD(acc[d], SJ__FMUL(w[t], row[d]));
    }
}

/* ---- policy ------------------------------------------------------------------ */

SJKVARN_DEF void sj_kvarn_policy_init(sj_kvarn_policy * p, uint32_t sink, uint32_t tail, uint32_t tail_max,
                                      uint32_t group, uint32_t flush_chunk) {
    p->sink = sink; p->tail = tail; p->tail_max = tail_max; p->group = group; p->flush_chunk = flush_chunk;
    sj_kvarn_policy_reset(p);
}

SJKVARN_DEF void sj_kvarn_policy_reset(sj_kvarn_policy * p) {
    p->B = p->B_prev = p->B_pending = p->sink;
    p->draining = 0;
}

/* The sealed end advances from the ubatch START, so every query row keeps at
 * least `tail` exact positions behind it. Adaptive tail (tail_max > 0): a commit
 * becomes due when the unsealed span reaches tail_max and runs down to `tail`.
 * With flush_chunk > 0 a decode-sized ubatch (fewer than G tokens) seals at most
 * flush_chunk groups and leaves the rest "draining" for the next ubatches;
 * prefill-sized ubatches take the whole commit at once. */
SJKVARN_DEF uint32_t sj_kvarn_policy_begin_ubatch(sj_kvarn_policy * p, uint32_t pos0, uint32_t n_tokens) {
    int due;
    p->B_pending = p->B;
    due = p->tail_max == 0 || p->draining || (pos0 > p->B && pos0 - p->B >= p->tail_max);
    p->draining = 0;
    if (due && pos0 > p->sink + p->tail) {
        uint32_t target = p->sink + p->group*((pos0 - p->tail - p->sink)/p->group);
        if (target < p->B) target = p->B;
        if (p->tail_max > 0 && p->flush_chunk > 0 && n_tokens < p->group) {
            const uint32_t limit = p->B + p->group*p->flush_chunk;
            if (target > limit) {
                target = limit;
                p->draining = 1;
            }
        }
        p->B_pending = target;
    }
    return p->B_pending;
}

SJKVARN_DEF void sj_kvarn_policy_commit(sj_kvarn_policy * p) {
    p->B_prev = p->B;
    p->B = p->B_pending;
}

SJKVARN_DEF uint32_t sj_kvarn_policy_idle(sj_kvarn_policy * p, uint32_t end) {
    uint32_t target;
    p->B_pending = p->B;
    if (p->tail_max == 0 || end <= p->sink + p->tail) return p->B;
    target = p->sink + p->group*((end - p->tail - p->sink)/p->group);
    if (target <= p->B) { p->draining = 0; return p->B; }
    p->B_pending = target;
    p->draining = 0;
    return target;
}

SJKVARN_DEF uint32_t sj_kvarn_ring_capacity(uint32_t tail, uint32_t tail_max, uint32_t group, uint32_t n_ubatch) {
    const uint32_t need = (tail > tail_max ? tail : tail_max) + group + 2u*n_ubatch;
    return (need + 127u) / 128u * 128u;
}

SJKVARN_DEF SJ_KVARN_HD uint32_t sj_kvarn_ring_slot(uint32_t sink, uint32_t cap, uint32_t pos) {
    return (pos - sink) % cap;
}

/* ---- reference per-layer cache ----------------------------------------------- */

SJKVARN_DEF sj_kvarn_config sj_kvarn_config_default(int head_dim, int n_head_kv, uint32_t n_ctx, uint32_t n_ubatch) {
    sj_kvarn_config c;
    c.head_dim = head_dim; c.n_head_kv = n_head_kv;
    c.group = SJKVARN_GROUP_DEFAULT;
    c.bits_k = 4; c.bits_v = 4;
    c.body = SJKVARN_BODY_AUTO;
    c.iters = SJKVARN_ITERS_DEFAULT;
    c.staging = SJKVARN_STAGING_TQ6_0;
    c.sink_type = SJKVARN_SINK_F16;
    c.sink = SJKVARN_SINK_DEFAULT;
    c.tail = SJKVARN_TAIL_DEFAULT;
    c.tail_max = SJKVARN_TAIL_MAX_DEFAULT;
    c.flush_chunk = 0;
    c.n_ubatch = n_ubatch;
    c.n_ctx = n_ctx;
    return c;
}

SJKVARN_DEF int sj_kvarn_layer_init(sj_kvarn_layer * L, const sj_kvarn_config * cfg) {
    const int D = cfg->head_dim, H = cfg->n_head_kv;
    int body, rc;
    memset(L, 0, sizeof(*L));
    L->cfg = *cfg;
    body = sj_kvarn_body_resolve(cfg->body, cfg->bits_k, cfg->bits_v);
    L->cfg.body = body;
    if (D % SJKVARN_TQ6_QK != 0 || cfg->group % 16 != 0 || cfg->sink == 0 || cfg->sink % 64 != 0 ||
        cfg->tail % (uint32_t) cfg->group != 0 || (cfg->tail_max != 0 && (cfg->tail_max < cfg->tail || cfg->tail_max % 128 != 0))) {
        return -10;
    }
    rc = sj_kvarn_layout_init(&L->lay, D, cfg->group, cfg->bits_k, cfg->bits_v, body);
    if (rc != 0) return rc;
    L->cap = sj_kvarn_ring_capacity(cfg->tail, cfg->tail_max, (uint32_t) cfg->group, cfg->n_ubatch);
    L->n_groups = cfg->n_ctx > cfg->sink ? (cfg->n_ctx - cfg->sink + (uint32_t) cfg->group - 1)/(uint32_t) cfg->group : 1;
    L->stage_bytes = sj_kvarn_stage_row_bytes(cfg->staging, D);
    L->ring = (uint8_t *) calloc((size_t) L->cap*H, 2*L->stage_bytes);
    L->sink = (uint8_t *) calloc((size_t) cfg->sink*H, 2*(cfg->sink_type == SJKVARN_SINK_F16 ? (size_t) D*2 : L->stage_bytes));
    L->body = (uint8_t *) calloc((size_t) L->n_groups*H, L->lay.bytes);
    L->work = malloc(sj_kvarn_seal_workspace_bytes(&L->lay));
    L->kstage = (uint16_t *) malloc((size_t) cfg->group*D*2);
    L->vstage = (uint16_t *) malloc((size_t) cfg->group*D*2);
    if (!L->ring || !L->sink || !L->body || !L->work || !L->kstage || !L->vstage) {
        sj_kvarn_layer_free(L);
        return -11;
    }
    return 0;
}

SJKVARN_DEF void sj_kvarn_layer_free(sj_kvarn_layer * L) {
    free(L->ring); free(L->sink); free(L->body); free(L->work); free(L->kstage); free(L->vstage);
    L->ring = L->sink = L->body = NULL; L->work = NULL; L->kstage = L->vstage = NULL;
}

SJKVARN_DEF void sj_kvarn_layer_store(sj_kvarn_layer * L, uint32_t pos, const float * k_rot, const float * v_rot) {
    const int D = L->cfg.head_dim, H = L->cfg.n_head_kv;
    int h, i;
    for (h = 0; h < H; ++h) {
        const float * kr = k_rot + (size_t) h*D, * vr = v_rot + (size_t) h*D;
        if (pos < L->cfg.sink && L->cfg.sink_type == SJKVARN_SINK_F16) {
            /* sink layout: [pos][head][K D | V D] fp16 */
            uint16_t * kd = (uint16_t *) L->sink + ((size_t) pos*H + h)*2*D;
            uint16_t * vd = kd + D;
            for (i = 0; i < D; ++i) { kd[i] = sj_kvarn_f32_to_f16(kr[i]); vd[i] = sj_kvarn_f32_to_f16(vr[i]); }
        } else {
            uint8_t * row = pos < L->cfg.sink ? L->sink + ((size_t) pos*H + h)*2*L->stage_bytes
                                              : L->ring + ((size_t) sj_kvarn_ring_slot(L->cfg.sink, L->cap, pos)*H + h)*2*L->stage_bytes;
            sj_kvarn_stage_row(L->cfg.staging, kr, row, D);
            sj_kvarn_stage_row(L->cfg.staging, vr, row + L->stage_bytes, D);
        }
    }
}

SJKVARN_DEF void sj_kvarn_layer_seal(sj_kvarn_layer * L, uint32_t B_from, uint32_t B_to) {
    const int D = L->cfg.head_dim, H = L->cfg.n_head_kv, G = L->cfg.group;
    float row[1024];
    uint32_t p0;
    int h, t, i;
    for (p0 = B_from; p0 < B_to; p0 += (uint32_t) G) {
        const uint32_t g = (p0 - L->cfg.sink)/(uint32_t) G;
        for (h = 0; h < H; ++h) {
            for (t = 0; t < G; ++t) {
                const uint8_t * src = L->ring + ((size_t) sj_kvarn_ring_slot(L->cfg.sink, L->cap, p0 + (uint32_t) t)*H + h)*2*L->stage_bytes;
                /* seal input = staged row widened to float, rounded to fp16 */
                sj_kvarn_unstage_row(L->cfg.staging, src, row, D);
                for (i = 0; i < D; ++i) L->kstage[(size_t) t*D + i] = sj_kvarn_f32_to_f16(row[i]);
                sj_kvarn_unstage_row(L->cfg.staging, src + L->stage_bytes, row, D);
                for (i = 0; i < D; ++i) L->vstage[(size_t) t*D + i] = sj_kvarn_f32_to_f16(row[i]);
            }
            sj_kvarn_seal_group(&L->lay, L->kstage, L->vstage, (size_t) D, L->cfg.iters,
                                L->body + ((size_t) g*H + h)*L->lay.bytes, L->work);
        }
    }
}

SJKVARN_DEF void sj_kvarn_layer_kv_row(const sj_kvarn_layer * L, uint32_t B, uint32_t pos, int h, float * k, float * v) {
    const int D = L->cfg.head_dim, H = L->cfg.n_head_kv, G = L->cfg.group;
    int i;
    if (pos >= L->cfg.sink && pos < B) {
        const uint32_t g = (pos - L->cfg.sink)/(uint32_t) G, t = (pos - L->cfg.sink)%(uint32_t) G;
        const uint8_t * rec = L->body + ((size_t) g*H + h)*L->lay.bytes;
        sj_kvarn_decode_k_row(&L->lay, rec, (int) t, k);
        sj_kvarn_decode_v_row(&L->lay, rec, (int) t, v);
    } else if (pos < L->cfg.sink && L->cfg.sink_type == SJKVARN_SINK_F16) {
        const uint16_t * kd = (const uint16_t *) L->sink + ((size_t) pos*H + h)*2*D;
        for (i = 0; i < D; ++i) { k[i] = sj_kvarn_f16_to_f32(kd[i]); v[i] = sj_kvarn_f16_to_f32(kd[D + i]); }
    } else {
        const uint8_t * row = pos < L->cfg.sink ? L->sink + ((size_t) pos*H + h)*2*L->stage_bytes
                                                : L->ring + ((size_t) sj_kvarn_ring_slot(L->cfg.sink, L->cap, pos)*H + h)*2*L->stage_bytes;
        sj_kvarn_unstage_row(L->cfg.staging, row, k, D);
        sj_kvarn_unstage_row(L->cfg.staging, row + L->stage_bytes, v, D);
    }
}

SJKVARN_DEF void sj_kvarn_layer_attend_row(const sj_kvarn_layer * L, uint32_t B, uint32_t N, const float * q_rot,
                                           int h, uint32_t qpos, float scale, float * out_rot) {
    const int D = L->cfg.head_dim;
    const uint32_t end = N < qpos + 1 ? N : qpos + 1;
    float krow[1024], vrow[1024], acc[1024];
    float M = -INFINITY, S = 0.0f;
    uint32_t p;
    int d;
    for (d = 0; d < D; ++d) acc[d] = 0.0f;
    /* single-pass online softmax over positions in order */
    for (p = 0; p < end; ++p) {
        float s, ms = 1.0f, vs = 1.0f;
        sj_kvarn_layer_kv_row(L, B, p, h, krow, vrow);
        s = SJ__FMUL(sj_kvarn__dot(q_rot, krow, D), scale);
        if (s > M) {
            const float Mold = M;
            M = s;
            ms = expf(SJ__FSUB(Mold, M));
            for (d = 0; d < D; ++d) acc[d] = SJ__FMUL(acc[d], ms);
        } else {
            vs = expf(SJ__FSUB(s, M));
        }
        for (d = 0; d < D; ++d) acc[d] = SJ__FADD(acc[d], SJ__FMUL(vrow[d], vs));
        S = SJ__FADD(SJ__FMUL(S, ms), vs);
    }
    {
        const float inv = S == 0.0f ? 0.0f : SJ__FDIV(1.0f, S);
        for (d = 0; d < D; ++d) out_rot[d] = SJ__FMUL(acc[d], inv);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* SJ_KVARN_IMPLEMENTATION */

/*
 * MIT License
 *
 * Copyright (c) 2026 Jake K
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
