/* Prompt caching and multiple sequences on the reference cache (sj_kvarn_seq):
 *   1. truncation above and below the sealed end B, at and across group boundaries, inside the sink;
 *      kept records stay bit-identical, and truncate + re-prefill (also of an edited suffix) equals a fresh build
 *   2. state round trips for 4/4, 3/3, 3/3t and 3/2t (plus an f16-staging / inherited-sink variant):
 *      write -> read -> write is byte-identical, attention is bit-identical, and both copies continue identically
 *   3. every config mismatch and every damaged stream is rejected with its own code, leaving the target unchanged
 *   4. idle steps with a keep-recent margin on a real cache
 *   5. two sequences sharing one paged pool give exactly the bytes and outputs they give alone
 *   6. golden FNV-1a hashes of the state streams (tests/expected_state.txt) pin the serialized layout
 * usage: tests/test_state [tests/expected_state.txt] [--print-golden] */
#define SJ_KVARN_IMPLEMENTATION
#include "../sj_kvarn.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D        128
#define H        2
#define NL       2
#define SINK     128
#define G        128
#define TAIL     256
#define TAILMAX  512
#define NCTX     2304
#define ROW      (NL*H*D)

static int bad = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); bad++; } } while (0)

/* ---- deterministic, position-addressable inputs ---- */
static uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
/* value of (seed, variant, pos, layer/head/channel index, side); integers scaled by 2^-12, exact in float */
static float val(uint32_t seed, uint32_t variant, uint32_t pos, uint32_t i, int side) {
    const uint64_t h = mix(((uint64_t) seed << 40) ^ ((uint64_t) variant << 36) ^ ((uint64_t) pos << 16) ^ ((uint64_t) i << 1) ^ (uint64_t) side);
    const int32_t g = (int32_t) (h & 1023u) + (int32_t) ((h >> 10) & 1023u) - 1024;
    const int scale = (i % 41u) == 0 ? 24 : 1 + (int) ((h >> 20) & 3u); /* a few outlier channels */
    return (float) (g*scale) * (1.0f/4096.0f);
}

typedef struct { uint32_t seed, edit_from, edit_variant; } src_t; /* positions >= edit_from use edit_variant */

static void row_of(const src_t * s, uint32_t pos, float * k, float * v) {
    const uint32_t var = pos >= s->edit_from ? s->edit_variant : 0u;
    uint32_t i;
    for (i = 0; i < ROW; ++i) { k[i] = val(s->seed, var, pos, i, 0); v[i] = val(s->seed, var, pos, i, 1); }
}

/* grow [from, to) in ubatches of a deterministic size pattern */
static int grow(sj_kvarn_seq * q, const src_t * s, uint32_t from, uint32_t to, uint32_t pattern) {
    static float k[ROW], v[ROW];
    uint32_t pos = from, step = 0;
    while (pos < to) {
        uint32_t n = pattern == 0 ? 1u : (step % 5u == 4u ? 1u + (pattern*7u + step*13u) % 64u : 1u + (step + pattern) % 4u);
        uint32_t i;
        int rc;
        if (pos + n > to) n = to - pos;
        rc = sj_kvarn_seq_begin_ubatch(q, pos, n);
        if (rc != SJKVARN_OK) return rc;
        for (i = 0; i < n; ++i) { row_of(s, pos + i, k, v); sj_kvarn_seq_store(q, pos + i, k, v); }
        pos += n; step++;
    }
    return SJKVARN_OK;
}

static sj_kvarn_config mkcfg(int bk, int bv, int body) {
    sj_kvarn_config c = sj_kvarn_config_default(D, H, NCTX, 64);
    c.bits_k = bk; c.bits_v = bv; c.body = body;
    c.sink = SINK; c.tail = TAIL; c.tail_max = TAILMAX;
    return c;
}

static uint8_t * dump(const sj_kvarn_seq * q, size_t * size) {
    size_t n = sj_kvarn_seq_state_size(q), w = 0;
    uint8_t * buf = (uint8_t *) malloc(n);
    int rc = sj_kvarn_seq_state_write(q, buf, n, &w);
    CHECK(rc == SJKVARN_OK && w == n, "state write rc %d (%s) wrote %zu of %zu", rc, sj_kvarn_strerror(rc), w, n);
    *size = n;
    return buf;
}

/* payload only: sink, records and tail rows, independent of B_prev / draining */
static int same_payload(const sj_kvarn_seq * a, const sj_kvarn_seq * b) {
    size_t na, nb;
    uint8_t * x = dump(a, &na), * y = dump(b, &nb);
    int eq = na == nb && a->n == b->n && a->pol.B == b->pol.B &&
             memcmp(x + SJKVARN_STATE_HEADER_BYTES, y + SJKVARN_STATE_HEADER_BYTES, na - SJKVARN_STATE_HEADER_BYTES - 8) == 0;
    free(x); free(y);
    return eq;
}

static int same_stream(const sj_kvarn_seq * a, const sj_kvarn_seq * b) {
    size_t na, nb;
    uint8_t * x = dump(a, &na), * y = dump(b, &nb);
    int eq = na == nb && memcmp(x, y, na) == 0;
    free(x); free(y);
    return eq;
}

/* attention of a few query rows per layer and head; bit-exact comparison */
static int same_attention(const sj_kvarn_seq * a, const sj_kvarn_seq * b) {
    float q[D], oa[D], ob[D];
    uint32_t qp[5];
    int il, h, j, d;
    if (a->n != b->n || a->n == 0) return a->n == b->n;
    qp[0] = a->n - 1; qp[1] = a->n/2; qp[2] = a->pol.B > 0 ? a->pol.B - 1 : 0; qp[3] = a->pol.B; qp[4] = 3;
    for (il = 0; il < NL; ++il) for (h = 0; h < H; ++h) for (j = 0; j < 5; ++j) {
        const uint32_t p = qp[j] < a->n ? qp[j] : a->n - 1;
        for (d = 0; d < D; ++d) q[d] = val(777, 0, p, (uint32_t) ((il*H + h)*D + d), 0);
        sj_kvarn_seq_attend_row(a, il, q, h, p, 0.0883883476f, oa);
        sj_kvarn_seq_attend_row(b, il, q, h, p, 0.0883883476f, ob);
        if (memcmp(oa, ob, sizeof oa) != 0) return 0;
    }
    return 1;
}

static int all_zero(const uint8_t * p, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (p[i]) return 0;
    return 1;
}

/* ---- 1. truncation ---------------------------------------------------------------------------------- */
static void test_truncate(int bk, int bv, int body, const char * name) {
    const sj_kvarn_config c = mkcfg(bk, bv, body);
    const uint32_t N = 1500, M = 1760;
    const src_t s0 = { 11, UINT32_MAX, 0 };
    sj_kvarn_seq ref;
    uint32_t B0, cuts[16];
    int nc = 0, i;
    CHECK(sj_kvarn_seq_init(&ref, &c, NL, NULL) == SJKVARN_OK, "init");
    grow(&ref, &s0, 0, N, 1);
    B0 = ref.pol.B;
    CHECK(B0 > SINK + 3*G, "%s: reference build sealed too little (B %u)", name, B0);
    cuts[nc++] = N + 50;        /* past the end           */
    cuts[nc++] = N - 1;         /* tail cut               */
    cuts[nc++] = B0 + 7;        /* tail cut just above B  */
    cuts[nc++] = B0;            /* exactly at B           */
    cuts[nc++] = B0 - 1;        /* one below B            */
    cuts[nc++] = B0 - G;        /* on a group boundary    */
    cuts[nc++] = B0 - G - 5;    /* inside a group         */
    cuts[nc++] = SINK + G + 1;  /* across many groups     */
    cuts[nc++] = SINK + 1;
    cuts[nc++] = SINK;          /* at the sink end        */
    cuts[nc++] = SINK - 10;     /* inside the sink        */
    cuts[nc++] = 0;
    for (i = 0; i < nc; ++i) {
        const uint32_t cut = cuts[i];
        const uint32_t want_r = (cut >= B0 || cut <= SINK) ? cut : SINK + G*((cut - SINK)/G);
        const uint32_t want_B = cut >= B0 ? B0 : (want_r > SINK ? want_r : SINK);
        const uint32_t edit = cut < N ? cut : N;
        const src_t s1 = { 11, edit, 1 };   /* the client edited everything from the cut on */
        sj_kvarn_seq a, f;
        uint32_t r, g;
        int h;
        sj_kvarn_seq_init(&a, &c, NL, NULL);
        grow(&a, &s0, 0, N, 1);
        r = sj_kvarn_seq_truncate(&a, cut);
        CHECK(r == want_r, "%s cut %u: resume %u, want %u", name, cut, r, want_r);
        CHECK(r <= cut && cut - r <= G - 1, "%s cut %u: resume %u not within G-1", name, cut, r);
        CHECK(a.pol.B == want_B, "%s cut %u: B %u, want %u", name, cut, a.pol.B, want_B);
        CHECK(a.n == (r < N ? r : N), "%s cut %u: stored end %u", name, cut, a.n);
        /* kept records bit-identical, dropped ones scrubbed */
        for (g = 0; g < (B0 - SINK)/G; ++g) for (h = 0; h < H; ++h) {
            const uint8_t * x = sj_kvarn_layer_record(&a.layers[1], g, h), * y = sj_kvarn_layer_record(&ref.layers[1], g, h);
            if (g < (a.pol.B - SINK)/G) CHECK(memcmp(x, y, a.layers[1].lay.bytes) == 0, "%s cut %u: kept record %u changed", name, cut, g);
            else CHECK(all_zero(x, a.layers[1].lay.bytes), "%s cut %u: dropped record %u not scrubbed", name, cut, g);
        }
        /* re-prefill from the resume point with the edited suffix and a different ubatch pattern */
        CHECK(grow(&a, &s1, a.n, M, 3) == SJKVARN_OK, "regrow");
        sj_kvarn_seq_init(&f, &c, NL, NULL);
        grow(&f, &s1, 0, M, 2);
        /* the two schedules seal at different times; an idle flush brings both to the same B */
        sj_kvarn_seq_idle(&a, 0);
        sj_kvarn_seq_idle(&f, 0);
        CHECK(a.pol.B == f.pol.B, "%s cut %u: B %u vs fresh %u after idle", name, cut, a.pol.B, f.pol.B);
        CHECK(same_payload(&a, &f), "%s cut %u: truncate + re-prefill differs from a fresh build", name, cut);
        CHECK(same_attention(&a, &f), "%s cut %u: attention differs from a fresh build", name, cut);
        sj_kvarn_seq_free(&a); sj_kvarn_seq_free(&f);
    }
    printf("truncate   %-8s %d cuts (B = %u)\n", name, nc, B0);
    sj_kvarn_seq_free(&ref);
}

/* ---- 2. round trips --------------------------------------------------------------------------------- */
static void test_roundtrip(sj_kvarn_config c, const char * name, uint64_t * golden) {
    static const uint32_t Ns[] = { 0, 77, 128, 700, 1500, 1990 };
    const src_t s0 = { 23, UINT32_MAX, 0 };
    unsigned ni;
    for (ni = 0; ni < sizeof Ns/sizeof Ns[0]; ++ni) {
        const uint32_t N = Ns[ni];
        sj_kvarn_seq a, b, p;
        sj_kvarn_pool pool;
        sj_kvarn_config c2 = c;
        size_t n1, n2, n3, w;
        uint8_t * s1, * s2, * s3;
        int rc;
        sj_kvarn_seq_init(&a, &c, NL, NULL);
        grow(&a, &s0, 0, N, 4);
        s1 = dump(&a, &n1);
        CHECK(n1 <= sj_kvarn_state_size_max(&c, NL), "%s N %u: size %zu above size_max", name, N, n1);
        {
            /* too-small buffer */
            uint8_t tiny[16];
            CHECK(sj_kvarn_seq_state_write(&a, tiny, sizeof tiny, &w) == SJKVARN_ERR_BUFFER, "%s: small buffer accepted", name);
        }
        /* restore into a cache with a different ring (n_ubatch) and context */
        c2.n_ubatch = 512; c2.n_ctx = NCTX + 1000;
        sj_kvarn_seq_init(&b, &c2, NL, NULL);
        rc = sj_kvarn_seq_state_read(&b, s1, n1);
        CHECK(rc == SJKVARN_OK, "%s N %u: read rc %d %s", name, N, rc, sj_kvarn_strerror(rc));
        s2 = dump(&b, &n2);
        CHECK(n1 == n2 && memcmp(s1, s2, n1) == 0, "%s N %u: write -> read -> write not byte-identical", name, N);
        CHECK(same_attention(&a, &b), "%s N %u: restored attention differs", name, N);
        /* restore into a pooled sequence: the same stream */
        sj_kvarn_pool_init(&pool, &c, NL, 40);
        sj_kvarn_seq_init(&p, &c, NL, &pool);
        { uint32_t blk = sj_kvarn_pool_alloc(&pool); (void) blk; } /* shift block ids */
        rc = sj_kvarn_seq_state_read(&p, s1, n1);
        CHECK(rc == SJKVARN_OK, "%s N %u: pooled read rc %d", name, N, rc);
        s3 = dump(&p, &n3);
        CHECK(n1 == n3 && memcmp(s1, s3, n1) == 0, "%s N %u: pooled round trip not byte-identical", name, N);
        CHECK(same_attention(&a, &p), "%s N %u: pooled attention differs", name, N);
        /* the restored copies continue exactly like the original (policy state included) */
        grow(&a, &s0, N, N + 300, 5); grow(&b, &s0, N, N + 300, 5); grow(&p, &s0, N, N + 300, 5);
        CHECK(same_stream(&a, &b) && same_stream(&a, &p), "%s N %u: continuation after restore differs", name, N);
        CHECK(same_attention(&a, &b) && same_attention(&a, &p), "%s N %u: continuation attention differs", name, N);
        if (N == 1500) {
            size_t i;
            uint64_t hsh = 1469598103934665603ULL;
            for (i = 0; i < n1; ++i) { hsh ^= s1[i]; hsh *= 1099511628211ULL; }
            *golden = hsh;
        }
        free(s1); free(s2); free(s3);
        sj_kvarn_seq_free(&a); sj_kvarn_seq_free(&b); sj_kvarn_seq_free(&p);
        CHECK(pool.n_free == pool.n_blocks - 1, "%s: pool leaked blocks (%u free)", name, pool.n_free);
        sj_kvarn_pool_free(&pool);
    }
    printf("roundtrip  %-8s %u sizes, byte-identical, pooled and continued\n", name, (unsigned) (sizeof Ns/sizeof Ns[0]));
}

/* ---- 3. mismatch and damage ------------------------------------------------------------------------- */
static void refix_checksum(uint8_t * s, size_t n) {
    size_t i;
    uint64_t h = 1469598103934665603ULL;
    for (i = 0; i + 8 < n; ++i) { h ^= s[i]; h *= 1099511628211ULL; }
    for (i = 0; i < 8; ++i) s[n - 8 + i] = (uint8_t) (h >> (8*i));
}

static void expect_reject(const sj_kvarn_config * c, int n_layers, const uint8_t * s, size_t n, int want, const char * what) {
    const src_t sx = { 5, UINT32_MAX, 0 };
    sj_kvarn_seq t;
    size_t nb, na;
    uint8_t * before, * after;
    int rc;
    if (sj_kvarn_seq_init(&t, c, n_layers, NULL) != SJKVARN_OK) { CHECK(0, "init for %s", what); return; }
    grow(&t, &sx, 0, 900, 1);      /* the target already holds a sequence: it must survive the failed restore */
    before = dump(&t, &nb);
    rc = sj_kvarn_seq_state_read(&t, s, n);
    CHECK(rc == want, "%s: rc %d (%s), want %d (%s)", what, rc, sj_kvarn_strerror(rc), want, sj_kvarn_strerror(want));
    after = dump(&t, &na);
    CHECK(nb == na && memcmp(before, after, nb) == 0, "%s: failed restore changed the target", what);
    free(before); free(after);
    sj_kvarn_seq_free(&t);
}

static void test_mismatch(void) {
    const sj_kvarn_config c = mkcfg(3, 3, SJKVARN_BODY_TRELLIS);
    const src_t s0 = { 31, UINT32_MAX, 0 };
    sj_kvarn_seq a;
    sj_kvarn_config m;
    sj_kvarn_state_info info;
    size_t n;
    uint8_t * s, * x;
    int k = 0;
    sj_kvarn_seq_init(&a, &c, NL, NULL);
    grow(&a, &s0, 0, 1300, 2);
    s = dump(&a, &n);
    x = (uint8_t *) malloc(n);
    CHECK(sj_kvarn_state_peek(s, n, &info) == SJKVARN_OK && info.n == 1300 && info.B == a.pol.B && info.n_layers == NL &&
          info.total_bytes == n && info.cfg.bits_k == 3 && info.cfg.body == SJKVARN_BODY_TRELLIS, "peek");

    m = c; m.head_dim = 256;                       expect_reject(&m, NL, s, n, SJKVARN_ERR_HEAD_DIM, "head_dim"); k++;
    m = c; m.n_head_kv = 1;                        expect_reject(&m, NL, s, n, SJKVARN_ERR_N_HEAD_KV, "n_head_kv"); k++;
    m = c; m.sink = 192;                           expect_reject(&m, NL, s, n, SJKVARN_ERR_SINK, "sink length"); k++;
    m = c; m.sink_type = SJKVARN_SINK_INHERIT;     expect_reject(&m, NL, s, n, SJKVARN_ERR_SINK, "sink type"); k++;
    m = c; m.bits_v = 2;                           expect_reject(&m, NL, s, n, SJKVARN_ERR_BITS, "bits 3/2 vs 3/3"); k++;
    m = c; m.bits_k = 4; m.bits_v = 4; m.body = SJKVARN_BODY_AUTO; expect_reject(&m, NL, s, n, SJKVARN_ERR_BITS, "bits 4/4 vs 3/3"); k++;
    m = c; m.body = SJKVARN_BODY_SCALAR;           expect_reject(&m, NL, s, n, SJKVARN_ERR_BODY, "scalar vs trellis"); k++;
    m = c; m.iters = 8;                            expect_reject(&m, NL, s, n, SJKVARN_ERR_ITERS, "iters"); k++;
    m = c; m.staging = SJKVARN_STAGING_F16;        expect_reject(&m, NL, s, n, SJKVARN_ERR_STAGING, "staging"); k++;
    m = c; m.tail = 384;                           expect_reject(&m, NL, s, n, SJKVARN_ERR_TAIL, "tail"); k++;
    m = c; m.tail_max = 1024;                      expect_reject(&m, NL, s, n, SJKVARN_ERR_TAIL, "tail_max"); k++;
    m = c; m.tail_max = 0;                         expect_reject(&m, NL, s, n, SJKVARN_ERR_TAIL, "fixed vs adaptive tail"); k++;
    m = c; m.flush_chunk = 2;                      expect_reject(&m, NL, s, n, SJKVARN_ERR_TAIL, "flush_chunk"); k++;
    expect_reject(&c, NL + 1, s, n, SJKVARN_ERR_N_LAYERS, "n_layers"); k++;
    m = c; m.n_ctx = 1024;                         expect_reject(&m, NL, s, n, SJKVARN_ERR_CAPACITY, "context too small"); k++;
    {
        /* group mismatch needs a scalar body (trellis is fixed at G = 128) */
        sj_kvarn_config g4 = mkcfg(4, 4, SJKVARN_BODY_SCALAR), g64;
        sj_kvarn_seq q;
        size_t nq;
        uint8_t * sq;
        sj_kvarn_seq_init(&q, &g4, NL, NULL);
        grow(&q, &s0, 0, 900, 1);
        sq = dump(&q, &nq);
        g64 = g4; g64.group = 64;
        expect_reject(&g64, NL, sq, nq, SJKVARN_ERR_GROUP, "group"); k++;
        free(sq);
        sj_kvarn_seq_free(&q);
    }
    /* header fields the config cannot express: rewrite them and fix the checksum */
    memcpy(x, s, n); x[72] = 2;  refix_checksum(x, n); expect_reject(&c, NL, x, n, SJKVARN_ERR_ROTATION, "rotation id"); k++;
    memcpy(x, s, n); x[76] = 64; refix_checksum(x, n); expect_reject(&c, NL, x, n, SJKVARN_ERR_ROTATION, "rotation size"); k++;
    memcpy(x, s, n); x[80] ^= 1; refix_checksum(x, n); expect_reject(&c, NL, x, n, SJKVARN_ERR_CODEBOOK, "codebook"); k++;
    memcpy(x, s, n); x[12] = 2;  refix_checksum(x, n); expect_reject(&c, NL, x, n, SJKVARN_ERR_FORMAT, "record format version"); k++;
    memcpy(x, s, n); x[8] = 2;                         expect_reject(&c, NL, x, n, SJKVARN_ERR_VERSION, "state version"); k++;
    memcpy(x, s, n); x[0] = 'X';                       expect_reject(&c, NL, x, n, SJKVARN_ERR_MAGIC, "magic"); k++;
    memcpy(x, s, n); x[n/2] ^= 0x10;                   expect_reject(&c, NL, x, n, SJKVARN_ERR_CORRUPT, "payload bit flip"); k++;
    memcpy(x, s, n); x[n - 1] ^= 1;                    expect_reject(&c, NL, x, n, SJKVARN_ERR_CORRUPT, "checksum"); k++;
    memcpy(x, s, n); x[96] = (uint8_t) (x[96] + 1); refix_checksum(x, n); expect_reject(&c, NL, x, n, SJKVARN_ERR_CORRUPT, "B off a group boundary"); k++;
    memcpy(x, s, n); x[92] = (uint8_t) (x[92] + 1); refix_checksum(x, n); expect_reject(&c, NL, x, n, SJKVARN_ERR_CORRUPT, "n vs payload length"); k++;
    expect_reject(&c, NL, s, n - 1, SJKVARN_ERR_TRUNCATED, "one byte short"); k++;
    expect_reject(&c, NL, s, 100, SJKVARN_ERR_TRUNCATED, "header only"); k++;
    expect_reject(&c, NL, s, 4, SJKVARN_ERR_TRUNCATED, "4 bytes"); k++;
    {
        /* pool without enough free blocks; the pooled target must stay intact */
        sj_kvarn_pool pool;
        sj_kvarn_seq q;
        int rc;
        sj_kvarn_pool_init(&pool, &c, NL, 4);
        sj_kvarn_seq_init(&q, &c, NL, &pool);
        rc = sj_kvarn_seq_state_read(&q, s, n);
        CHECK(rc == SJKVARN_ERR_POOL_FULL && q.n == 0 && pool.n_free == 4, "pool full: rc %d", rc); k++;
        sj_kvarn_seq_free(&q);
        sj_kvarn_pool_free(&pool);
    }
    CHECK(strcmp(sj_kvarn_strerror(SJKVARN_ERR_CODEBOOK), "unknown SJ-KVaRN status code") != 0, "strerror");
    printf("mismatch   %d rejections, target unchanged after each\n", k);
    free(s); free(x);
    sj_kvarn_seq_free(&a);
}

/* ---- 4. idle margin on a real cache ------------------------------------------------------------------ */
static void test_idle(void) {
    const sj_kvarn_config c = mkcfg(4, 4, SJKVARN_BODY_SCALAR);
    const src_t s0 = { 41, UINT32_MAX, 0 };
    sj_kvarn_seq a, b;
    uint32_t B_before, prompt_end;
    sj_kvarn_seq_init(&a, &c, NL, NULL);
    grow(&a, &s0, 0, 1100, 0);           /* decode-style: B = 640, the tail has grown to 460 */
    B_before = a.pol.B;
    prompt_end = 700;                    /* the last turn starts here: keep it exact */
    sj_kvarn_seq_idle(&a, a.n - prompt_end);
    CHECK(B_before == 640 && a.pol.B == 640, /* full target would be 768, the margin stops it at 640 <= 700 */
          "idle with margin sealed to %u (prompt end %u)", a.pol.B, prompt_end);
    /* the protected turn is still reachable exactly: a cut at prompt_end resumes at prompt_end */
    CHECK(sj_kvarn_policy_trunc_floor(&a.pol, prompt_end) == prompt_end, "turn not reachable after idle");
    sj_kvarn_seq_idle(&a, 0);
    CHECK(a.pol.B == SINK + G*((1100 - TAIL - SINK)/G), "full idle sealed to %u", a.pol.B);
    /* the margin never changes record bytes: groups sealed by idle equal those sealed by decode */
    sj_kvarn_seq_init(&b, &c, NL, NULL);
    grow(&b, &s0, 0, 1100, 2);
    sj_kvarn_seq_idle(&b, 0);
    CHECK(same_payload(&a, &b) && same_attention(&a, &b), "idle-sealed cache differs");
    printf("idle       margin kept [%u, 1100) exact, then full flush to %u\n", prompt_end, a.pol.B);
    sj_kvarn_seq_free(&a); sj_kvarn_seq_free(&b);
}

/* ---- 5. multi-sequence pool ---------------------------------------------------------------------------- */
static void test_pool(int bk, int bv, int body, const char * name) {
    const sj_kvarn_config c = mkcfg(bk, bv, body);
    const src_t sa = { 51, UINT32_MAX, 0 }, sb = { 52, UINT32_MAX, 0 }, sa2 = { 51, 900, 3 };
    sj_kvarn_pool pool;
    sj_kvarn_seq pa, pb, la, lb;
    uint32_t r1, r2, g;
    int nonmono = 0;
    CHECK(sj_kvarn_pool_init(&pool, &c, NL, 30) == SJKVARN_OK, "pool init");
    sj_kvarn_seq_init(&pa, &c, NL, &pool); sj_kvarn_seq_init(&pb, &c, NL, &pool);
    sj_kvarn_seq_init(&la, &c, NL, NULL);  sj_kvarn_seq_init(&lb, &c, NL, NULL);
    /* interleave the two sequences in steps of 200 positions so their blocks alternate in the pool */
    {
        uint32_t p;
        for (p = 0; p < 1400; p += 200) {
            grow(&pa, &sa, p, p + 200, 1); grow(&la, &sa, p, p + 200, 1);
            grow(&pb, &sb, p, p + 200, 3); grow(&lb, &sb, p, p + 200, 3);
        }
    }
    /* A is truncated (its blocks go back to the pool), B grows into them, A re-prefills an edited suffix */
    r1 = sj_kvarn_seq_truncate(&pa, 900); r2 = sj_kvarn_seq_truncate(&la, 900);
    CHECK(r1 == r2, "pool truncate resume %u vs %u", r1, r2);
    grow(&pb, &sb, 1400, 1900, 3); grow(&lb, &sb, 1400, 1900, 3);
    grow(&pa, &sa2, r1, 1700, 2);  grow(&la, &sa2, r2, 1700, 2);
    sj_kvarn_seq_idle(&pa, 300); sj_kvarn_seq_idle(&la, 300);
    for (g = 1; g < (pb.pol.B - SINK)/G; ++g) nonmono += pb.blocks[g] != pb.blocks[g - 1] + 1;
    CHECK(nonmono > 0, "%s: block table is contiguous, indirection not exercised", name);
    CHECK(same_stream(&pa, &la) && same_stream(&pb, &lb), "%s: pooled state differs from the sequence alone", name);
    CHECK(same_attention(&pa, &la) && same_attention(&pb, &lb), "%s: pooled attention differs from the sequence alone", name);
    CHECK(pool.n_free == pool.n_blocks - (pa.pol.B - SINK)/G - (pb.pol.B - SINK)/G, "%s: pool accounting (%u free)", name, pool.n_free);
    printf("pool       %-8s 2 sequences in %u blocks: bytes and outputs equal to alone (%d table jumps)\n", name, pool.n_blocks, nonmono);
    sj_kvarn_seq_free(&pa); sj_kvarn_seq_free(&pb); sj_kvarn_seq_free(&la); sj_kvarn_seq_free(&lb);
    CHECK(pool.n_free == pool.n_blocks, "%s: blocks not returned", name);
    {
        /* a full pool refuses the seal and leaves the sequence as it was */
        sj_kvarn_pool small;
        sj_kvarn_seq q;
        int rc = SJKVARN_OK;
        sj_kvarn_pool_init(&small, &c, NL, 2);
        sj_kvarn_seq_init(&q, &c, NL, &small);
        rc = grow(&q, &sa, 0, 1400, 1);
        CHECK(rc == SJKVARN_ERR_POOL_FULL && q.pol.B == SINK + 2u*G && q.pol.B_pending == q.pol.B, "%s: pool full rc %d B %u", name, rc, q.pol.B);
        sj_kvarn_seq_free(&q);
        sj_kvarn_pool_free(&small);
    }
    sj_kvarn_pool_free(&pool);
}

int main(int argc, char ** argv) {
    const char * gpath = "tests/expected_state.txt";
    int print = 0, i, ng = 0;
    struct { const char * name; sj_kvarn_config c; uint64_t h; } rt[5];
    for (i = 1; i < argc; ++i) { if (!strcmp(argv[i], "--print-golden")) print = 1; else gpath = argv[i]; }

    test_truncate(4, 4, SJKVARN_BODY_SCALAR, "4/4");
    test_truncate(3, 3, SJKVARN_BODY_TRELLIS, "3/3t");

    rt[0].name = "4/4";  rt[0].c = mkcfg(4, 4, SJKVARN_BODY_SCALAR);
    rt[1].name = "3/3";  rt[1].c = mkcfg(3, 3, SJKVARN_BODY_SCALAR);
    rt[2].name = "3/3t"; rt[2].c = mkcfg(3, 3, SJKVARN_BODY_TRELLIS);
    rt[3].name = "3/2t"; rt[3].c = mkcfg(3, 2, SJKVARN_BODY_TRELLIS);
    rt[4].name = "4/4f"; rt[4].c = mkcfg(4, 4, SJKVARN_BODY_SCALAR);   /* f16 staging, inherited sink, chunked flush */
    rt[4].c.staging = SJKVARN_STAGING_F16; rt[4].c.sink_type = SJKVARN_SINK_INHERIT; rt[4].c.flush_chunk = 1;
    rt[2].c.flush_chunk = 1;
    for (i = 0; i < 5; ++i) test_roundtrip(rt[i].c, rt[i].name, &rt[i].h);

    test_mismatch();
    test_idle();
    test_pool(4, 4, SJKVARN_BODY_SCALAR, "4/4");
    test_pool(3, 2, SJKVARN_BODY_TRELLIS, "3/2t");

    if (print) {
        printf("# FNV-1a 64 of the sequence state stream (state version %d) of tests/test_state.c's 1500-position build\n", SJKVARN_STATE_VERSION);
        for (i = 0; i < 5; ++i) printf("state %s v%d hash=%016" PRIx64 "\n", rt[i].name, SJKVARN_STATE_VERSION, rt[i].h);
    } else {
        FILE * f = fopen(gpath, "r");
        char line[256], nm[16];
        int ver;
        uint64_t h;
        if (!f) { perror(gpath); bad++; }
        while (f && fgets(line, sizeof line, f)) {
            if (sscanf(line, "state %15s v%d hash=%" SCNx64, nm, &ver, &h) != 3) continue;
            for (i = 0; i < 5; ++i) if (!strcmp(nm, rt[i].name) && ver == SJKVARN_STATE_VERSION) {
                CHECK(h == rt[i].h, "golden state %s: %016" PRIx64 " vs expected %016" PRIx64, nm, rt[i].h, h);
                ng++;
            }
        }
        if (f) fclose(f);
        CHECK(ng == 5, "found %d golden state hashes, want 5", ng);
        printf("golden     %d state stream hashes match\n", ng);
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
