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

// Step 1: ROI reduction kernel. The serial reference accumulates `sum` and
// `sum2` as floats in row-major order; q0sqr is a single scalar broadcast to
// every pixel, so even a tiny float-order difference here perturbs every c[k]
// and compounds over the 100 iterations (the source of the prior ~5e-3 error).
// The ROI is tiny (128x128), so we reproduce the serial sequential float sum
// EXACTLY with a single-thread reduction — cheap and bit-faithful.
__global__ void kernel_roi_reduce(
    const float * __restrict__ J,
    int cols,
    int r1, int r2, int c1, int c2,
    float *g_sum, float *g_sum2)
{
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = r1; i <= r2; i++) {
        for (int j = c1; j <= c2; j++) {
            float tmp = J[i * cols + j];
            sum  += tmp;
            sum2 += tmp * tmp;
        }
    }
    *g_sum  = sum;
    *g_sum2 = sum2;
}

// Step 2: PREPARE pass — compute dN/dS/dW/dE and c[k] for each pixel.
// Arithmetic in double to match serial reference.
__global__ void kernel_prepare(
    const float * __restrict__ J,
    float *dN, float *dS, float *dW, float *dE,
    float *c,
    const int *iN, const int *iS, const int *jW, const int *jE,
    int rows, int cols,
    float q0sqr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= cols) return;

    int k = i * cols + j;
    float Jc = J[k];

    // Match the serial reference precision EXACTLY: every named intermediate is
    // a float (so each product/sum is rounded to float), with the same
    // double-precision literals that promote sub-expressions to double before
    // the result is rounded back to float on assignment.
    float dn = J[iN[i] * cols + j] - Jc;
    float ds = J[iS[i] * cols + j] - Jc;
    float dw = J[i * cols + jW[j]] - Jc;
    float de = J[i * cols + jE[j]] - Jc;

    dN[k] = dn;
    dS[k] = ds;
    dW[k] = dw;
    dE[k] = de;

    float G2 = (dn*dn + ds*ds + dw*dw + de*de) / (Jc * Jc);
    float L  = (dn + ds + dw + de) / Jc;

    float num = (0.5 * G2) - ((1.0 / 16.0) * (L * L));
    float den = 1 + (.25 * L);
    float qsqr = num / (den * den);

    den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
    float ck = 1.0 / (1.0 + den);

    if (ck < 0) ck = 0;
    else if (ck > 1) ck = 1;
    c[k] = ck;
}

// Step 3: UPDATE pass — compute divergence and update J[k].
__global__ void kernel_update(
    float *J,
    const float * __restrict__ dN,
    const float * __restrict__ dS,
    const float * __restrict__ dW,
    const float * __restrict__ dE,
    const float * __restrict__ c,
    const int *iS, const int *jE,
    int rows, int cols,
    float lambda)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= cols) return;

    int k = i * cols + j;
    float cN = c[k];
    float cS = c[iS[i] * cols + j];
    float cW = c[k];
    float cE = c[i * cols + jE[j]];

    // Serial: D is float (each product/sum rounded to float); the update
    // promotes via the 0.25 double literal then rounds the stored float.
    float D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

    J[k] = J[k] + 0.25 * lambda * D;
}

int main(int argc, char **argv)
{
    if (argc != 10) {
        fprintf(stderr,
            "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lambda> <niter> <output_file>\n",
            argv[0]);
        return 1;
    }
    int rows   = atoi(argv[1]);
    int cols   = atoi(argv[2]);
    int r1     = atoi(argv[3]);
    int r2     = atoi(argv[4]);
    int c1     = atoi(argv[5]);
    int c2     = atoi(argv[6]);
    float lambda = (float)atof(argv[7]);
    int niter  = atoi(argv[8]);
    const char *ofile = argv[9];

    if ((rows % 16 != 0) || (cols % 16 != 0)) {
        fprintf(stderr, "rows and cols must be multiples of 16\n");
        return 1;
    }

    int size_I = rows * cols;
    int size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

    // Host allocations
    float *h_I = (float *)malloc(size_I * sizeof(float));
    float *h_J = (float *)malloc(size_I * sizeof(float));
    int   *h_iN = (int *)malloc(rows * sizeof(int));
    int   *h_iS = (int *)malloc(rows * sizeof(int));
    int   *h_jW = (int *)malloc(cols * sizeof(int));
    int   *h_jE = (int *)malloc(cols * sizeof(int));

    for (int i = 0; i < rows; i++) { h_iN[i] = i - 1; h_iS[i] = i + 1; }
    for (int j = 0; j < cols; j++) { h_jW[j] = j - 1; h_jE[j] = j + 1; }
    h_iN[0] = 0; h_iS[rows-1] = rows-1;
    h_jW[0] = 0; h_jE[cols-1] = cols-1;

    random_matrix(h_I, rows, cols);
    for (int k = 0; k < size_I; k++)
        h_J[k] = (float)exp(h_I[k]);

    // Device allocations
    float *d_J, *d_dN, *d_dS, *d_dW, *d_dE, *d_c;
    float *d_sum, *d_sum2;
    int   *d_iN, *d_iS, *d_jW, *d_jE;

    cudaMalloc(&d_J,    size_I * sizeof(float));
    cudaMalloc(&d_dN,   size_I * sizeof(float));
    cudaMalloc(&d_dS,   size_I * sizeof(float));
    cudaMalloc(&d_dW,   size_I * sizeof(float));
    cudaMalloc(&d_dE,   size_I * sizeof(float));
    cudaMalloc(&d_c,    size_I * sizeof(float));
    cudaMalloc(&d_sum,  sizeof(float));
    cudaMalloc(&d_sum2, sizeof(float));
    cudaMalloc(&d_iN, rows * sizeof(int));
    cudaMalloc(&d_iS, rows * sizeof(int));
    cudaMalloc(&d_jW, cols * sizeof(int));
    cudaMalloc(&d_jE, cols * sizeof(int));

    cudaMemcpy(d_J,  h_J,  size_I * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iN, h_iN, rows * sizeof(int),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_iS, h_iS, rows * sizeof(int),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_jW, h_jW, cols * sizeof(int),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_jE, h_jE, cols * sizeof(int),     cudaMemcpyHostToDevice);

    // Grid for 2D kernels
    dim3 block2d(16, 16);
    dim3 grid2d((cols + 15) / 16, (rows + 15) / 16);

    double t0 = now_seconds();

    for (int iter = 0; iter < niter; iter++) {
        // Step 1: ROI reduction
        float zero = 0.0f;
        cudaMemcpy(d_sum,  &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sum2, &zero, sizeof(float), cudaMemcpyHostToDevice);

        kernel_roi_reduce<<<1, 1>>>(
            d_J, cols, r1, r2, c1, c2, d_sum, d_sum2);

        float h_sum, h_sum2;
        cudaMemcpy(&h_sum,  d_sum,  sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum2, d_sum2, sizeof(float), cudaMemcpyDeviceToHost);

        float meanROI = h_sum / (float)size_R;
        float varROI  = (h_sum2 / (float)size_R) - meanROI * meanROI;
        float q0sqr   = varROI / (meanROI * meanROI);

        // Step 2: PREPARE
        kernel_prepare<<<grid2d, block2d>>>(
            d_J, d_dN, d_dS, d_dW, d_dE, d_c,
            d_iN, d_iS, d_jW, d_jE,
            rows, cols, q0sqr);

        // Step 3: UPDATE
        kernel_update<<<grid2d, block2d>>>(
            d_J, d_dN, d_dS, d_dW, d_dE, d_c,
            d_iS, d_jE,
            rows, cols, lambda);
    }

    cudaDeviceSynchronize();
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    // Copy result back
    cudaMemcpy(h_J, d_J, size_I * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int idx = 0; idx < size_I; idx++)
        fprintf(fp, "%d\t%.5f\n", idx, h_J[idx]);
    fclose(fp);

    // Cleanup
    cudaFree(d_J); cudaFree(d_dN); cudaFree(d_dS); cudaFree(d_dW); cudaFree(d_dE);
    cudaFree(d_c); cudaFree(d_sum); cudaFree(d_sum2);
    cudaFree(d_iN); cudaFree(d_iS); cudaFree(d_jW); cudaFree(d_jE);
    free(h_I); free(h_J);
    free(h_iN); free(h_iS); free(h_jW); free(h_jE);
    return 0;
}
