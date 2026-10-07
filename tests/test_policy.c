/* Adaptive-tail policy: fuzzed against an independent transcription of the reference cache's per-ubatch rule,
 * plus the invariants an engine relies on (tail kept, ring never overwritten, group alignment). */
#define SJ_KVARN_IMPLEMENTATION
#include "../sj_kvarn.h"

#include <stdio.h>

typedef struct { uint32_t B, pending; int draining; } ref_state;

static void ref_apply(ref_state * s, uint32_t sink, uint32_t tail, uint32_t tail_max, uint32_t G, uint32_t chunk,
                      uint32_t pos0, uint32_t n_tokens) {
    int due;
    s->pending = s->B;
    due = tail_max == 0 || s->draining || (pos0 > s->B && pos0 - s->B >= tail_max);
    s->draining = 0;
    if (due && pos0 > sink + tail) {
        uint32_t target = sink + G*((pos0 - tail - sink)/G);
        if (target < s->B) target = s->B;
        if (tail_max > 0 && chunk > 0 && n_tokens < G) {
            const uint32_t limit = s->B + G*chunk;
            if (target > limit) { target = limit; s->draining = 1; }
        }
        s->pending = target;
    }
}

static uint64_t rs = 88172645463325252ULL;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t) (rs >> 11); }

int main(void) {
    static const uint32_t tails[][2] = { {4096, 8192}, {4096, 0}, {256, 512}, {128, 128}, {0, 0}, {512, 1024} };
    const uint32_t G = 128, sink = 128;
    int bad = 0, cases = 0;
    unsigned ti, chunk, trial;
    for (ti = 0; ti < sizeof tails/sizeof tails[0]; ++ti) {
        for (chunk = 0; chunk < 3; ++chunk) {
            for (trial = 0; trial < 200; ++trial) {
                const uint32_t tail = tails[ti][0], tail_max = tails[ti][1];
                const uint32_t nub = 1u + rnd() % 512u;
                const uint32_t cap = sj_kvarn_ring_capacity(tail, tail_max, G, nub);
                sj_kvarn_policy p;
                ref_state r = { sink, sink, 0 };
                uint32_t pos = 0;
                sj_kvarn_policy_init(&p, sink, tail, tail_max, G, chunk);
                while (pos < 40000) {
                    /* mostly decode-sized ubatches, some prefill-sized, occasional idle compression */
                    const uint32_t n = (rnd() % 4u) ? 1u + rnd() % 8u : 1u + rnd() % nub;
                    const uint32_t B_prev = p.B;
                    const uint32_t pend = sj_kvarn_policy_begin_ubatch(&p, pos, n);
                    ref_apply(&r, sink, tail, tail_max, G, chunk, pos, n);
                    cases++;
                    if (pend != r.pending || p.draining != r.draining) { bad++; break; }
                    if (pend > B_prev) {
                        if ((pend - sink) % G != 0) { bad++; break; }       /* whole groups only     */
                        if (pos - pend < tail) { bad++; break; }            /* tail kept exact       */
                    }
                    sj_kvarn_policy_commit(&p);
                    r.B = r.pending;
                    if (pos + n > B_prev + cap) { bad++; printf("ring overflow\n"); break; } /* unsealed rows fit */
                    pos += n;
                    if (tail_max > 0 && rnd() % 64u == 0) {
                        const uint32_t t = sj_kvarn_policy_idle(&p, pos);
                        if (t > p.B) {
                            if (pos - t < tail || (t - sink) % G) { bad++; break; }
                            sj_kvarn_policy_commit(&p);
                        }
                        r.B = p.B; r.pending = p.B; r.draining = p.draining;
                    }
                }
            }
        }
    }
    {
        /* default schedule: one token at a time, the tail grows 4096 -> 8192 then commits back to 4096 */
        sj_kvarn_policy p;
        uint32_t pos, commits = 0, maxspan = 0;
        sj_kvarn_policy_init(&p, sink, SJKVARN_TAIL_DEFAULT, SJKVARN_TAIL_MAX_DEFAULT, G, 0);
        for (pos = 0; pos < 100000; ++pos) {
            if (sj_kvarn_policy_begin_ubatch(&p, pos, 1) > p.B) commits++;
            sj_kvarn_policy_commit(&p);
            if (pos > p.B && pos - p.B > maxspan) maxspan = pos - p.B;
        }
        printf("default decode schedule: %u commits over 100000 tokens, longest exact tail %u positions\n", commits, maxspan);
        if (maxspan > SJKVARN_TAIL_MAX_DEFAULT || commits == 0) bad++;
    }
    printf("%d policy steps, %s\n", cases, bad ? "FAIL" : "PASS");
    return bad != 0;
}
