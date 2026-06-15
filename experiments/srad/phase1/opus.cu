/*
 * SRAD (srad_v2) — Phase 1 basic CUDA port (global memory only).
 *
 * 3 kernels per iteration: ROI reduction, prepare pass, update pass.
 * Per-pixel math done in double, stored to float, to match the serial
 * golden bit-closely under --fmad=false.
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

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

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(1);} } while(0)

// ROI reduction: each block reduces a chunk of the ROI box (size_R elements),
// accumulating sum and sum2. Final reduction of partials done on host-side
// via a second small kernel pass would add complexity; instead we accumulate
// block partials into global arrays and finalize with a single-block kernel.

// ROI reduction. The serial reference accumulates float `sum`/`sum2` running
// totals in row-major order over the box (i in [r1,r2], j in [c1,c2]). Float
// addition is NOT associative, so a parallel tree reduction (or double accum)
// yields a slightly different q0sqr, which then perturbs EVERY pixel each
// iteration and compounds over niter=100. To match the golden bit-closely we
// reproduce the exact sequential float accumulation in a single thread. The box
// is small (e.g. 128x128) so this is cheap relative to the per-pixel passes.
__global__ void reduce_roi(const float *J, int cols,
                           int r1, int r2, int c1, int c2, int size_R,
                           float *q0sqr_out)
{
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = r1; i <= r2; i++) {
        for (int j = c1; j <= c2; j++) {
            float tmp = J[i * cols + j];
            sum  += tmp;
            sum2 += tmp * tmp;
        }
    }
    float meanROI = sum / size_R;
    float varROI  = (sum2 / size_R) - meanROI * meanROI;
    float q0sqr   = varROI / (meanROI * meanROI);
    *q0sqr_out = q0sqr;
}

__global__ void prepare_kernel(const float *J, float *c,
                               float *dN, float *dS, float *dW, float *dE,
                               int rows, int cols, const float *q0sqr_ptr)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int N = rows * cols;
    if (k >= N) return;

    int i = k / cols;
    int j = k - i * cols;

    int iN = (i == 0) ? 0 : i - 1;
    int iS = (i == rows - 1) ? rows - 1 : i + 1;
    int jW = (j == 0) ? 0 : j - 1;
    int jE = (j == cols - 1) ? cols - 1 : j + 1;

    float Jc = J[k];

    float vN = J[iN * cols + j] - Jc;
    float vS = J[iS * cols + j] - Jc;
    float vW = J[i * cols + jW] - Jc;
    float vE = J[i * cols + jE] - Jc;
    dN[k] = vN; dS[k] = vS; dW[k] = vW; dE[k] = vE;

    float G2 = (vN * vN + vS * vS + vW * vW + vE * vE) / (Jc * Jc);
    float L = (vN + vS + vW + vE) / Jc;

    float q0sqr = *q0sqr_ptr;

    // Match the serial C type promotion EXACTLY:
    //   float num  = (0.5 * G2) - ((1.0/16.0) * (L*L));  -> double expr, stored float
    //   float den  = 1 + (.25 * L);                      -> double expr, stored float
    //   float qsqr = num / (den * den);                  -> den*den is FLOAT, division FLOAT
    //   float den  = (qsqr - q0sqr)/(q0sqr*(1+q0sqr));    -> all FLOAT
    //   float ck   = 1.0/(1.0 + den);                     -> double expr, stored float
    float num  = (float)((0.5 * (double)G2) - ((1.0 / 16.0) * ((double)L * (double)L)));
    float den  = (float)(1.0 + (0.25 * (double)L));
    float qsqr = num / (den * den);

    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    float ck = (float)(1.0 / (1.0 + (double)den));

    if (ck < 0) ck = 0;
    else if (ck > 1) ck = 1;
    c[k] = ck;
}

__global__ void update_kernel(float *J, const float *c,
                              const float *dN, const float *dS,
                              const float *dW, const float *dE,
                              int rows, int cols, float lambda)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int N = rows * cols;
    if (k >= N) return;

    int i = k / cols;
    int j = k - i * cols;

    int iS = (i == rows - 1) ? rows - 1 : i + 1;
    int jE = (j == cols - 1) ? cols - 1 : j + 1;

    float cN = c[k];
    float cS = c[iS * cols + j];
    float cW = c[k];
    float cE = c[i * cols + jE];

    // Serial: float D = cN*dN + cS*dS + cW*dW + cE*dE;  (all float)
    //         J[k] = J[k] + 0.25*lambda*D;  (0.25*lambda is double -> double expr, stored float)
    float D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
    J[k] = (float)((double)J[k] + 0.25 * (double)lambda * (double)D);
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
    int boxW = (c2 - c1 + 1);
    int size_R = (r2 - r1 + 1) * boxW;

    float *I = (float *)malloc(size_I * sizeof(float));
    float *J = (float *)malloc(size_I * sizeof(float));
    if (!I || !J) { fprintf(stderr, "alloc failed\n"); return 1; }

    random_matrix(I, rows, cols);
    for (int k = 0; k < size_I; k++)
        J[k] = (float)exp(I[k]);

    float *d_J, *d_c, *d_dN, *d_dS, *d_dW, *d_dE;
    CK(cudaMalloc(&d_J,  size_I * sizeof(float)));
    CK(cudaMalloc(&d_c,  size_I * sizeof(float)));
    CK(cudaMalloc(&d_dN, size_I * sizeof(float)));
    CK(cudaMalloc(&d_dS, size_I * sizeof(float)));
    CK(cudaMalloc(&d_dW, size_I * sizeof(float)));
    CK(cudaMalloc(&d_dE, size_I * sizeof(float)));

    float *d_q0;
    CK(cudaMalloc(&d_q0, sizeof(float)));

    CK(cudaMemcpy(d_J, J, size_I * sizeof(float), cudaMemcpyHostToDevice));

    int PBLK = 256;
    int pgrid = (size_I + PBLK - 1) / PBLK;

    CK(cudaDeviceSynchronize());
    double t0 = now_seconds();
    for (int iter = 0; iter < niter; iter++) {
        reduce_roi<<<1, 1>>>(d_J, cols, r1, r2, c1, c2, size_R, d_q0);
        prepare_kernel<<<pgrid, PBLK>>>(d_J, d_c, d_dN, d_dS, d_dW, d_dE,
                                        rows, cols, d_q0);
        update_kernel<<<pgrid, PBLK>>>(d_J, d_c, d_dN, d_dS, d_dW, d_dE,
                                       rows, cols, lambda);
    }
    CK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    CK(cudaMemcpy(J, d_J, size_I * sizeof(float), cudaMemcpyDeviceToHost));

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int idx = 0; idx < size_I; idx++)
        fprintf(fp, "%d\t%.5f\n", idx, J[idx]);
    fclose(fp);

    cudaFree(d_J); cudaFree(d_c); cudaFree(d_dN); cudaFree(d_dS);
    cudaFree(d_dW); cudaFree(d_dE);
    cudaFree(d_q0);
    free(I); free(J);
    return 0;
}
