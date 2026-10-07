/* KL sanity check: attention distributions over sealed groups vs the exact fp16 inputs.
 * usage: test_kl [file.kvd [max_groups]]   (.kvd = records of int32 {0x31445643, D, G, head}, char name[64],
 * K[G][D] fp16, V[G][D] fp16; rotated seal inputs). Without a file it uses 8 synthetic groups. */
#define SJ_KVARN_IMPLEMENTATION
#include "../sj_kvarn.h"
#include "vector_inputs.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NQ 64

typedef struct { int D, G; uint16_t * K, * V; } grp;

static int load_kvd(const char * path, int maxg, grp * out) {
    FILE * f = fopen(path, "rb");
    int n = 0;
    if (!f) { perror(path); return 0; }
    while (n < maxg) {
        int32_t h[4]; char name[64]; size_t e;
        if (fread(h, 4, 4, f) != 4 || h[0] != 0x31445643 || fread(name, 1, 64, f) != 64) break;
        e = (size_t) h[1]*h[2];
        out[n].D = h[1]; out[n].G = h[2];
        out[n].K = (uint16_t *) malloc(e*2); out[n].V = (uint16_t *) malloc(e*2);
        if (fread(out[n].K, 2, e, f) != e || fread(out[n].V, 2, e, f) != e) break;
        n++;
    }
    fclose(f);
    return n;
}

int main(int argc, char ** argv) {
    static const int cfg[][3] = { {4,4,0}, {3,3,1}, {3,2,1}, {2,2,1}, {3,3,0}, {3,2,0}, {2,2,0} };
    grp g[64];
    double kl_mean[16];
    int ng, i, c, bad = 0;
    uint64_t s = 12345;
    (void) sjv_fill_rows;
    if (argc > 1) {
        ng = load_kvd(argv[1], argc > 2 ? atoi(argv[2]) : 16, g);
        printf("%d real groups from %s\n", ng, argv[1]);
    } else {
        ng = 8;
        for (i = 0; i < ng; ++i) {
            g[i].D = 256; g[i].G = 128;
            g[i].K = (uint16_t *) malloc(256*128*2); g[i].V = (uint16_t *) malloc(256*128*2);
            sjv_fill_group(256, 128, (uint32_t) i + 7, g[i].K, g[i].V);
        }
        printf("%d synthetic groups (outlier channels, per-token scales)\n", ng);
    }
    if (ng == 0) return 2;
    printf("%-12s %6s %12s %12s %14s\n", "config", "bits", "mean KL", "max KL", "out rel err");
    for (c = 0; c < (int) (sizeof cfg/sizeof cfg[0]); ++c) {
        double kl_sum = 0, kl_max = 0, err_sum = 0;
        int nkl = 0;
        sj_kvarn_layout l;
        s = 12345; /* same queries for every config */
        for (i = 0; i < ng; ++i) {
            const int D = g[i].D, G = g[i].G;
            uint8_t * rec;
            void * work;
            float * Kx = (float *) malloc((size_t) G*D*4), * Vx = (float *) malloc((size_t) G*D*4);
            float * Kd = (float *) malloc((size_t) G*D*4), * Vd = (float *) malloc((size_t) G*D*4);
            int t, d, qi;
            if (sj_kvarn_layout_init(&l, D, G, cfg[c][0], cfg[c][1], cfg[c][2]) != 0) { bad++; break; }
            rec = (uint8_t *) malloc(l.bytes);
            work = malloc(sj_kvarn_seal_workspace_bytes(&l));
            sj_kvarn_seal_group(&l, g[i].K, g[i].V, (size_t) D, SJKVARN_ITERS_DEFAULT, rec, work);
            for (t = 0; t < G; ++t) {
                for (d = 0; d < D; ++d) { Kx[t*D + d] = sj_kvarn_f16_to_f32(g[i].K[t*D + d]); Vx[t*D + d] = sj_kvarn_f16_to_f32(g[i].V[t*D + d]); }
                sj_kvarn_decode_k_row(&l, rec, t, Kd + (size_t) t*D);
                sj_kvarn_decode_v_row(&l, rec, t, Vd + (size_t) t*D);
            }
            /* queries: a key of the group plus noise, so attention is peaked like real heads */
            for (qi = 0; qi < NQ; ++qi) {
                double q[1024], sx[1024], sd[1024], mx = -1e300, md = -1e300, zx = 0, zd = 0, kl = 0, on = 0, oe = 0;
                const int src = (int) (sjv_next(&s) % (uint32_t) G);
                for (d = 0; d < D; ++d) q[d] = Kx[src*D + d] + (double) sjv_gauss(&s)/600.0*fabs(Kx[src*D + d]) ;
                {
                    /* scale q so the exact logits have standard deviation 2.5 (a typical attention temperature) */
                    double m1 = 0, m2 = 0, sc;
                    for (t = 0; t < G; ++t) {
                        double a = 0;
                        for (d = 0; d < D; ++d) a += q[d]*Kx[t*D + d];
                        m1 += a; m2 += a*a;
                    }
                    m1 /= G; m2 = m2/G - m1*m1;
                    sc = m2 > 0 ? 2.5/sqrt(m2) : 1.0;
                    for (d = 0; d < D; ++d) q[d] *= sc;
                }
                for (t = 0; t < G; ++t) {
                    double a = 0, b = 0;
                    for (d = 0; d < D; ++d) { a += q[d]*Kx[t*D + d]; b += q[d]*Kd[t*D + d]; }
                    sx[t] = a; sd[t] = b;
                    if (sx[t] > mx) mx = sx[t];
                    if (sd[t] > md) md = sd[t];
                }
                for (t = 0; t < G; ++t) { sx[t] = exp(sx[t] - mx); zx += sx[t]; sd[t] = exp(sd[t] - md); zd += sd[t]; }
                for (t = 0; t < G; ++t) {
                    const double p = sx[t]/zx, r = sd[t]/zd;
                    if (p > 0) kl += p*log(p/(r > 1e-300 ? r : 1e-300));
                }
                for (d = 0; d < D; ++d) {
                    double ox = 0, od = 0;
                    for (t = 0; t < G; ++t) { ox += sx[t]/zx*Vx[t*D + d]; od += sd[t]/zd*Vd[t*D + d]; }
                    on += ox*ox; oe += (ox - od)*(ox - od);
                }
                kl_sum += kl; if (kl > kl_max) kl_max = kl; nkl++;
                err_sum += sqrt(oe/(on > 0 ? on : 1));
            }
            free(Kx); free(Vx); free(Kd); free(Vd); free(rec); free(work);
        }
        printf("%d/%d %-7s %6.3f %12.3e %12.3e %14.3e\n", cfg[c][0], cfg[c][1], cfg[c][2] ? "trellis" : "scalar",
               sj_kvarn_body_bits_per_element(&l), kl_sum/nkl, kl_max, err_sum/nkl);
        kl_mean[c] = kl_sum/nkl;
        if (!(kl_mean[c] >= 0.0 && kl_mean[c] < 10.0) || !(err_sum/nkl < 1.0)) bad++; /* finite, not degenerate */
    }
    /* sanity: fewer K bits never beats more K bits within the scalar body (rows 0, 4, 6 = 4/4, 3/3, 2/2) */
    if (!(kl_mean[0] < kl_mean[4] && kl_mean[4] < kl_mean[6])) { printf("KL not ordered by K bits\n"); bad++; }
    for (i = 0; i < ng; ++i) { free(g[i].K); free(g[i].V); }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
