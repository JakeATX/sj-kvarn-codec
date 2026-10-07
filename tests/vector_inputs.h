/* Deterministic test inputs built from integer arithmetic only (exact on every IEEE platform), shared by the
 * tree harness (which writes the expected hashes) and tests/test_vectors.c (which checks them).
 * Needs sj_kvarn.h (for sj_kvarn_f32_to_f16) to be included first. */
#ifndef SJ_KVARN_VECTOR_INPUTS_H
#define SJ_KVARN_VECTOR_INPUTS_H
#include <stdint.h>

static uint32_t sjv_next(uint64_t * s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    return (uint32_t) (*s >> 11);
}

/* approximately Gaussian integer in [-2048, 2044] (sum of four uniforms) */
static int32_t sjv_gauss(uint64_t * s) {
    int32_t a = 0, i;
    for (i = 0; i < 4; ++i) a += (int32_t) (sjv_next(s) & 1023u) - 512;
    return a;
}

/* value = gauss * channel_factor * token_factor * 2^-12, channel factor 1..4 with 3% outliers at 24..31,
 * token factor 1..4: an outlier-channel, per-token-scaled tile like real rotated K/V */
static float sjv_value(uint64_t * s, int ch_factor, int tok_factor) {
    return (float) (sjv_gauss(s) * ch_factor * tok_factor) * (1.0f / 4096.0f);
}

/* one (group, head) seal input: G rows of D fp16 values for K and for V */
static void sjv_fill_group(int D, int G, uint32_t seed, uint16_t * K, uint16_t * V) {
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
    int t, d, side;
    for (side = 0; side < 2; ++side) {
        uint16_t * X = side ? V : K;
        int chf[1024];
        for (d = 0; d < D; ++d) chf[d] = (sjv_next(&s) % 100u) < 3u ? 24 + (int) (sjv_next(&s) & 7u) : 1 + (int) (sjv_next(&s) & 3u);
        for (t = 0; t < G; ++t) {
            const int tf = 1 + (int) (sjv_next(&s) & 3u);
            for (d = 0; d < D; ++d) X[(size_t) t*D + d] = sj_kvarn_f32_to_f16(sjv_value(&s, chf[d], tf));
        }
    }
}

/* rows of rotated float values for the staging codec */
static void sjv_fill_rows(int rows, int n, float * x) {
    uint64_t s = 0xD1B54A32D192ED03ULL;
    int r, i;
    for (r = 0; r < rows; ++r) {
        const int tf = 1 + (int) (sjv_next(&s) & 7u);
        for (i = 0; i < n; ++i) x[(size_t) r*n + i] = sjv_value(&s, 1, tf);
        if (r == rows - 1) for (i = 0; i < n; ++i) x[(size_t) r*n + i] = 0.0f; /* the zero-norm branch */
    }
}
#endif
