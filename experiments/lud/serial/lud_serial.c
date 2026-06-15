/*
 * LU decomposition (lud) — serial reference implementation.
 *
 * Port of the Rodinia lud benchmark (openmp/lud/base/lud_base.c), made
 * self-contained and instrumented for the experiment harness. It performs a
 * non-pivoting (Doolittle) LU factorization of a dense N x N float matrix,
 * IN PLACE: on output the strictly-lower triangle holds L (unit diagonal of L
 * is implicit, not stored) and the upper triangle (including the diagonal)
 * holds U. There is no pivoting; the input matrix is chosen so the pivots
 * (diagonal elements) stay non-zero.
 *
 * The arithmetic and accumulation order reproduce lud_base.c EXACTLY. For each
 * step i (the outer loop), the row of U (a[i][j], j>=i) and the column of L
 * (a[j][i], j>i) are computed; each entry subtracts a dot product accumulated
 * over k=0..i-1 in increasing-k order, using a single `float sum` accumulator.
 * The k-loop accumulation order is the part that MUST be preserved to match a
 * bit-close serial golden under --fmad=false; a blocked GPU port will reorder
 * those accumulations and so only matches within tolerance.
 *
 * CLI:  lud_serial <input_file> <output_file>
 *   input_file  : matrix file (see format below).
 *   output_file : the factored N*N matrix, one line per cell
 *                 "<index>\t<value>\n", row-major, index 0..N*N-1.
 *
 * Input file format (Rodinia lud .dat):
 *   line 1   : N            (matrix dimension)
 *   then     : N*N floats   (the matrix, row-major) — whitespace separated.
 * Parsed exactly as openmp/lud/common/common.c::create_matrix_from_file:
 *   fscanf "%d\n" for the size, then fscanf "%f " for each of the N*N entries.
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* LU factorization, identical arithmetic & accumulation order to lud_base.c. */
static void lud_base(float *a, int size)
{
    int i, j, k;
    float sum;

    for (i = 0; i < size; i++) {
        for (j = i; j < size; j++) {
            sum = a[i * size + j];
            for (k = 0; k < i; k++) sum -= a[i * size + k] * a[k * size + j];
            a[i * size + j] = sum;
        }

        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            for (k = 0; k < i; k++) sum -= a[j * size + k] * a[k * size + i];
            a[j * size + i] = sum / a[i * size + i];
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    const char *input_f = argv[1];
    const char *ofile   = argv[2];

    FILE *fp = fopen(input_f, "rb");
    if (!fp) { fprintf(stderr, "Error reading matrix file %s\n", input_f); return 1; }

    int size = 0;
    if (fscanf(fp, "%d\n", &size) != 1 || size <= 0) {
        fprintf(stderr, "bad matrix size\n"); fclose(fp); return 1;
    }

    float *a = (float *)malloc(sizeof(float) * (size_t)size * (size_t)size);
    if (!a) { fprintf(stderr, "alloc failed\n"); fclose(fp); return 1; }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (fscanf(fp, "%f ", a + (size_t)i * size + j) != 1) {
                fprintf(stderr, "bad matrix entry at (%d,%d)\n", i, j);
                fclose(fp); free(a); return 1;
            }
        }
    }
    fclose(fp);

    double t0 = now_seconds();
    lud_base(a, size);
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    FILE *out = fopen(ofile, "w");
    if (!out) { fprintf(stderr, "cannot open %s\n", ofile); free(a); return 1; }
    size_t total = (size_t)size * (size_t)size;
    for (size_t idx = 0; idx < total; idx++)
        fprintf(out, "%zu\t%.6f\n", idx, a[idx]);
    fclose(out);

    free(a);
    return 0;
}
