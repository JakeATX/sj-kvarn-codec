/* fp16 conversion: every half decodes and re-encodes to itself; float -> half rounds to nearest even
 * (checked against a double-precision reference on a dense sweep, including subnormals and overflow). */
#define SJ_KVARN_IMPLEMENTATION
#include "../sj_kvarn.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static uint16_t ref_f16(float f) {
    /* exhaustive-search reference: nearest half, ties to even mantissa */
    uint32_t w; uint16_t best = 0, sign; double x, bd = INFINITY; int h;
    memcpy(&w, &f, 4);
    if (f != f) return 0x7E00 | (uint16_t) ((w >> 16) & 0x8000);
    sign = (uint16_t) ((w >> 16) & 0x8000);
    x = fabs((double) f);
    if (x >= 65520.0) return sign | 0x7C00;
    for (h = 0; h < 0x7C00; ++h) {
        const double v = (double) sj_kvarn_f16_to_f32((uint16_t) h);
        const double dd = fabs(v - x);
        if (dd < bd || (dd == bd && (h & 1) == 0)) { bd = dd; best = (uint16_t) h; }
        if (v > x) break;
    }
    return sign | best;
}

int main(void) {
    int bad = 0;
    uint32_t h, i;
    for (h = 0; h < 65536; ++h) {
        const float f = sj_kvarn_f16_to_f32((uint16_t) h);
        if (((h & 0x7C00) == 0x7C00) && (h & 0x3FF)) continue; /* NaN payloads */
        if (sj_kvarn_f32_to_f16(f) != h) { if (bad < 5) printf("half %04x round trip -> %04x\n", h, sj_kvarn_f32_to_f16(f)); bad++; }
    }
    for (i = 0; i < 200000; ++i) {
        /* sweep float bit patterns across the half range, with extra density near ties */
        const uint32_t w = 0x33000000u + i*((0x47800000u - 0x33000000u)/200000u) + (i & 0x1FFFu);
        float f; memcpy(&f, &w, 4);
        if (sj_kvarn_f32_to_f16(f) != ref_f16(f)) { if (bad < 10) printf("f32 %08x -> %04x want %04x\n", w, sj_kvarn_f32_to_f16(f), ref_f16(f)); bad++; }
        if (sj_kvarn_f32_to_f16(-f) != ref_f16(-f)) bad++;
    }
    printf("fp16 conversion %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
