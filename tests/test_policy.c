/* Adaptive-tail policy: fuzzed against an independent transcription of the reference cache's per-ubatch rule,
 * plus the invariants an engine relies on (tail kept, ring never overwritten, group alignment), with random
 * truncations (prompt-cache reuse) and idle steps with random keep-recent margins mixed in. */
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
    int bad = 0, cases = 0, trunc_cases = 0;
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
                        const uint32_t keep = (rnd() % 2u) ? 0u : rnd() % 12000u;
                        const uint32_t margin = keep > tail ? keep : tail;
                        const uint32_t t = sj_kvarn_policy_idle(&p, pos, keep);
                        if (t > p.B) {
                            if (pos - t < margin || (t - sink) % G) { bad++; printf("idle sealed inside the margin\n"); break; }
                            sj_kvarn_policy_commit(&p);
                        }
                        r.B = p.B; r.pending = p.B; r.draining = p.draining;
                    }
                    if (rnd() % 389u == 0) {
                        /* prompt-cache reuse: cut back to a random earlier position and re-prefill from the resume point */
                        /* look back up to 1500 positions (a re-rendered turn or an edited prompt suffix) */
                        const uint32_t back = rnd() % ((pos < 1500u ? pos : 1500u) + 1u);
                        const uint32_t cut = pos - back, B_before = p.B;
                        uint32_t dropped = 0;
                        const uint32_t floor_q = sj_kvarn_policy_trunc_floor(&p, cut);
                        const uint32_t res = sj_kvarn_policy_truncate(&p, cut, &dropped);
                        trunc_cases++;
                        if (res == SJKVARN_NO_POS || res != floor_q || res > cut || cut - res > G - 1u) { bad++; printf("truncate resume\n"); break; }
                        if (cut >= B_before || cut <= sink) {
                            if (res != cut) { bad++; printf("truncate moved an exact cut\n"); break; }
                        } else if ((res - sink) % G != 0) { bad++; printf("truncate off a group boundary\n"); break; }
                        if (cut >= B_before) {
                            if (p.B != B_before || dropped) { bad++; printf("tail cut changed B\n"); break; }
                        } else {
                            const uint32_t nb = res > sink ? res : sink;
                            if (p.B != nb || p.B_pending != nb || p.B_prev != nb || p.draining || dropped != (B_before - nb)/G) {
                                bad++; printf("body cut state\n"); break;
                            }
                        }
                        r.B = p.B; r.pending = p.B; r.draining = p.draining;
                        pos = res;
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
    {
        /* idle keep-recent margin, deterministic cases (sink 128, G 128, tail 256, tail_max 512) */
        static const struct { uint32_t B, end, keep, want; } ic[] = {
            { 128, 1000,    0,  640 },  /* keep 0 = tail: floor(1000 - 256) = 640                        */
            { 128, 1000,  100,  640 },  /* keep below the tail changes nothing                            */
            { 128, 1000,  256,  640 },
            { 128, 1000,  400,  512 },  /* floor(1000 - 400) = 512                                         */
            { 128, 1000,  872,  128 },  /* floor(128) = sink: nothing to seal                              */
            { 128, 1000, 5000,  128 },  /* margin past the start: nothing to seal                          */
            { 640, 1000,  400,  640 },  /* margin target below B: no-op, B is never lowered by idle       */
            { 128,  300,    0,  128 },  /* end <= sink + tail                                              */
        };
        unsigned i;
        for (i = 0; i < sizeof ic/sizeof ic[0]; ++i) {
            sj_kvarn_policy p;
            uint32_t got;
            sj_kvarn_policy_init(&p, 128, 256, 512, 128, 0);
            p.B = p.B_prev = p.B_pending = ic[i].B;
            got = sj_kvarn_policy_idle(&p, ic[i].end, ic[i].keep);
            if (got != ic[i].want) { printf("idle case %u: got %u want %u\n", i, got, ic[i].want); bad++; }
        }
        {
            /* fixed tail: idle never acts */
            sj_kvarn_policy p;
            sj_kvarn_policy_init(&p, 128, 256, 0, 128, 0);
            if (sj_kvarn_policy_idle(&p, 5000, 0) != p.B) { printf("idle acted with a fixed tail\n"); bad++; }
        }
        {
            /* a clamped idle step leaves a chunked drain pending; an unclamped one finishes it */
            sj_kvarn_policy p;
            sj_kvarn_policy_init(&p, 128, 256, 512, 128, 1);
            p.draining = 1;
            sj_kvarn_policy_idle(&p, 1000, 600); sj_kvarn_policy_commit(&p);
            if (!p.draining) { printf("clamped idle cleared the drain\n"); bad++; }
            sj_kvarn_policy_idle(&p, 1000, 0); sj_kvarn_policy_commit(&p);
            if (p.draining || p.B != 640) { printf("full idle did not finish the drain\n"); bad++; }
        }
    }
    {
        /* truncation, deterministic cases: B = 1024 (7 sealed groups past sink 128) */
        static const struct { uint32_t cut, res, B, dropped; } tc[] = {
            { 2000, 2000, 1024, 0 },   /* tail cut                                  */
            { 1024, 1024, 1024, 0 },   /* exactly at B: tail cut, nothing dropped    */
            { 1023,  896,  896, 1 },   /* one below B: last group dropped            */
            {  896,  896,  896, 1 },   /* on a boundary: resume there                */
            {  897,  896,  896, 1 },
            {  500,  384,  384, 5 },   /* across several groups                      */
            {  256,  256,  256, 6 },
            {  129,  128,  128, 7 },
            {  128,  128,  128, 7 },   /* at the sink end: everything sealed dropped */
            {   60,   60,  128, 7 },   /* inside the sink: exact, resume at the cut  */
            {    0,    0,  128, 7 },
        };
        unsigned i;
        for (i = 0; i < sizeof tc/sizeof tc[0]; ++i) {
            sj_kvarn_policy p;
            uint32_t d = 99, got;
            sj_kvarn_policy_init(&p, 128, 256, 512, 128, 0);
            p.B = p.B_prev = p.B_pending = 1024;
            got = sj_kvarn_policy_truncate(&p, tc[i].cut, &d);
            if (got != tc[i].res || p.B != tc[i].B || d != tc[i].dropped) {
                printf("truncate case %u: cut %u -> resume %u B %u dropped %u\n", i, tc[i].cut, got, p.B, d); bad++;
            }
        }
        {
            /* refused while a seal is pending */
            sj_kvarn_policy p;
            sj_kvarn_policy_init(&p, 128, 256, 512, 128, 0);
            sj_kvarn_policy_begin_ubatch(&p, 2000, 1);
            if (p.B_pending == p.B || sj_kvarn_policy_truncate(&p, 300, NULL) != SJKVARN_NO_POS || p.B != 128) {
                printf("truncate during a pending seal\n"); bad++;
            }
        }
        if (sj_kvarn_group_floor(128, 128, 100) != 100 || sj_kvarn_group_floor(128, 128, 383) != 256 ||
            sj_kvarn_group_floor(128, 128, 384) != 384) { printf("group_floor\n"); bad++; }
    }
    printf("%d policy steps, %d random truncations, %s\n", cases, trunc_cases, bad ? "FAIL" : "PASS");
    return bad != 0;
}
