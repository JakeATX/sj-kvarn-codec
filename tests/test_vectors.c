/* Checks sj_kvarn.h against expected_vectors.txt: FNV-1a 64 hashes of the reference tree's seal records,
 * decoded rows and tq6_0 staging bytes on the deterministic inputs of vector_inputs.h. */
#define SJ_KVARN_IMPLEMENTATION
#include "../sj_kvarn.h"
#include "vector_inputs.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv(const void * p, size_t n, uint64_t h) {
    const uint8_t * b = (const uint8_t *) p;
    size_t i;
    for (i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
#define FNV0 1469598103934665603ULL

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "tests/expected_vectors.txt";
    FILE * f = fopen(path, "r");
    char line[512];
    int n = 0, bad = 0;
    if (!f) { perror(path); return 2; }
    while (fgets(line, sizeof line, f)) {
        int D, bk, bv, body, t;
        size_t bytes;
        uint64_t erec, edec;
        if (sscanf(line, "seal D=%d bk=%d bv=%d body=%d bytes=%zu rec=%" SCNx64 " dec=%" SCNx64, &D, &bk, &bv, &body, &bytes, &erec, &edec) == 7) {
            sj_kvarn_layout l;
            uint16_t * K = (uint16_t *) malloc((size_t) D*128*2), * V = (uint16_t *) malloc((size_t) D*128*2);
            uint8_t * rec;
            void * work;
            float * row = (float *) malloc((size_t) D*4);
            uint64_t hr, hd = FNV0;
            if (sj_kvarn_layout_init(&l, D, 128, bk, bv, body) != 0 || l.bytes != bytes) { printf("layout mismatch: %s", line); bad++; continue; }
            rec = (uint8_t *) malloc(l.bytes);
            work = malloc(sj_kvarn_seal_workspace_bytes(&l));
            sjv_fill_group(D, 128, 1, K, V);
            sj_kvarn_seal_group(&l, K, V, (size_t) D, SJKVARN_ITERS_DEFAULT, rec, work);
            hr = fnv(rec, l.bytes, FNV0);
            for (t = 0; t < 128; ++t) {
                sj_kvarn_decode_k_row(&l, rec, t, row); hd = fnv(row, (size_t) D*4, hd);
                sj_kvarn_decode_v_row(&l, rec, t, row); hd = fnv(row, (size_t) D*4, hd);
            }
            printf("D=%-3d %d/%d %-7s record %s  decode %s\n", D, bk, bv, body ? "trellis" : "scalar",
                   hr == erec ? "bit-exact" : "MISMATCH", hd == edec ? "bit-exact" : "MISMATCH");
            bad += (hr != erec) + (hd != edec);
            n++;
            free(K); free(V); free(rec); free(work); free(row);
        } else {
            int rows, nn;
            uint64_t eenc, edq;
            if (sscanf(line, "tq6 rows=%d n=%d enc=%" SCNx64 " dec=%" SCNx64, &rows, &nn, &eenc, &edq) == 4) {
                float * x = (float *) malloc((size_t) rows*nn*4), * y = (float *) malloc((size_t) rows*nn*4);
                sj_kvarn_tq6_block * q = (sj_kvarn_tq6_block *) malloc((size_t) rows*(nn/128)*sizeof(sj_kvarn_tq6_block));
                int r;
                uint64_t he, hdq;
                sjv_fill_rows(rows, nn, x);
                for (r = 0; r < rows; ++r) sj_kvarn_tq6_quantize_row(x + (size_t) r*nn, q + (size_t) r*(nn/128), nn);
                for (r = 0; r < rows; ++r) sj_kvarn_tq6_dequantize_row(q + (size_t) r*(nn/128), y + (size_t) r*nn, nn);
                he = fnv(q, (size_t) rows*(nn/128)*sizeof(sj_kvarn_tq6_block), FNV0);
                hdq = fnv(y, (size_t) rows*nn*4, FNV0);
                printf("tq6_0 staging      encode %s  decode %s\n", he == eenc ? "bit-exact" : "MISMATCH", hdq == edq ? "bit-exact" : "MISMATCH");
                bad += (he != eenc) + (hdq != edq);
                n++;
                free(x); free(y); free(q);
            }
        }
    }
    fclose(f);
    if (sizeof(sj_kvarn_tq6_block) != SJKVARN_TQ6_BLOCK_BYTES) { printf("tq6 block is %zu bytes\n", sizeof(sj_kvarn_tq6_block)); bad++; }
    printf("%d vectors, %s\n", n, bad || n == 0 ? "FAIL" : "PASS");
    return bad || n == 0;
}
