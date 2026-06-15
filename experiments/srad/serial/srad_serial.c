/*
 * SRAD (Speckle Reducing Anisotropic Diffusion) — serial reference.
 *
 * Port of the Rodinia srad_v2 benchmark (openmp/srad/srad_v2/srad.cpp), made
 * self-contained and instrumented for the experiment harness. The input image
 * is generated deterministically from srand(7) (no external data file), then
 * SRAD is iterated `niter` times. Each iteration:
 *   1. compute ROI statistics (mean/variance) over a speckle box -> q0sqr
 *   2. PREPARE pass: per-pixel directional derivatives (dN/dS/dW/dE) and the
 *      diffusion coefficient c[k] (saturated to [0,1])
 *   3. UPDATE pass: divergence D from neighbour coefficients, then
 *      J[k] += 0.25*lambda*D
 *
 * The two passes are each data-parallel over pixels (the GPU port parallelizes
 * them); only the ROI reduction is a (small) reduction. Boundary handling uses
 * clamped neighbour-index arrays iN/iS/jW/jE exactly as the reference.
 *
 * IMPORTANT numerics: arrays are float, but the per-pixel arithmetic uses
 * DOUBLE-precision literals (0.5, 1.0/16.0, .25, 1.0, 0.25*lambda) exactly as
 * the reference, so every float store is rounded from a double expression. A
 * correct GPU port must reproduce this (compute in double, store float) to
 * match bit-closely under --fmad=false.
 *
 * CLI:  srad_serial <rows> <cols> <y1> <y2> <x1> <x2> <lambda> <niter> <output_file>
 *   rows,cols : image dimensions (multiples of 16)
 *   y1,y2,x1,x2 : speckle/ROI box (inclusive)
 *   lambda    : diffusion rate in (0,1)
 *   niter     : number of SRAD iterations
 *   output    : final J image, one line per pixel "<index>\t<value>\n",
 *               row-major over rows*cols pixels.
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void random_matrix(float *I, int rows, int cols) {
    srand(7);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            I[i * cols + j] = rand() / (float)RAND_MAX;
}

int main(int argc, char **argv)
{
    if (argc != 10) {
        fprintf(stderr,
            "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lambda> <niter> <output_file>\n",
            argv[0]);
        return 1;
    }
    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int r1 = atoi(argv[3]);
    int r2 = atoi(argv[4]);
    int c1 = atoi(argv[5]);
    int c2 = atoi(argv[6]);
    float lambda = (float)atof(argv[7]);
    int niter = atoi(argv[8]);
    const char *ofile = argv[9];

    if ((rows % 16 != 0) || (cols % 16 != 0)) {
        fprintf(stderr, "rows and cols must be multiples of 16\n");
        return 1;
    }

    int size_I = cols * rows;
    int size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

    float *I  = (float *)malloc(size_I * sizeof(float));
    float *J  = (float *)malloc(size_I * sizeof(float));
    float *c  = (float *)malloc(size_I * sizeof(float));
    float *dN = (float *)malloc(size_I * sizeof(float));
    float *dS = (float *)malloc(size_I * sizeof(float));
    float *dW = (float *)malloc(size_I * sizeof(float));
    float *dE = (float *)malloc(size_I * sizeof(float));
    int *iN = (int *)malloc(rows * sizeof(int));
    int *iS = (int *)malloc(rows * sizeof(int));
    int *jW = (int *)malloc(cols * sizeof(int));
    int *jE = (int *)malloc(cols * sizeof(int));
    if (!I || !J || !c || !dN || !dS || !dW || !dE || !iN || !iS || !jW || !jE) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    for (int i = 0; i < rows; i++) { iN[i] = i - 1; iS[i] = i + 1; }
    for (int j = 0; j < cols; j++) { jW[j] = j - 1; jE[j] = j + 1; }
    iN[0] = 0; iS[rows - 1] = rows - 1;
    jW[0] = 0; jE[cols - 1] = cols - 1;

    random_matrix(I, rows, cols);
    for (int k = 0; k < size_I; k++)
        J[k] = (float)exp(I[k]);

    double t0 = now_seconds();
    for (int iter = 0; iter < niter; iter++) {
        float sum = 0, sum2 = 0;
        for (int i = r1; i <= r2; i++) {
            for (int j = c1; j <= c2; j++) {
                float tmp = J[i * cols + j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }
        float meanROI = sum / size_R;
        float varROI = (sum2 / size_R) - meanROI * meanROI;
        float q0sqr = varROI / (meanROI * meanROI);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int k = i * cols + j;
                float Jc = J[k];

                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;

                float G2 = (dN[k] * dN[k] + dS[k] * dS[k]
                          + dW[k] * dW[k] + dE[k] * dE[k]) / (Jc * Jc);
                float L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

                float num = (0.5 * G2) - ((1.0 / 16.0) * (L * L));
                float den = 1 + (.25 * L);
                float qsqr = num / (den * den);

                den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
                c[k] = 1.0 / (1.0 + den);

                if (c[k] < 0) { c[k] = 0; }
                else if (c[k] > 1) { c[k] = 1; }
            }
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int k = i * cols + j;
                float cN = c[k];
                float cS = c[iS[i] * cols + j];
                float cW = c[k];
                float cE = c[i * cols + jE[j]];

                float D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                J[k] = J[k] + 0.25 * lambda * D;
            }
        }
    }
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int idx = 0; idx < size_I; idx++)
        fprintf(fp, "%d\t%.5f\n", idx, J[idx]);
    fclose(fp);

    free(I); free(J); free(c);
    free(dN); free(dS); free(dW); free(dE);
    free(iN); free(iS); free(jW); free(jE);
    return 0;
}
