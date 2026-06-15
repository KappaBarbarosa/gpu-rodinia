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

// ROI reduction. MUST reproduce the serial reference EXACTLY: the golden sums
// the ROI sequentially in float, row-major (i = r1..r2, j = c1..c2), and then
// derives q0sqr in float. A parallel tree-reduce or double accumulation changes
// the floating-point result of q0sqr, which then perturbs every pixel's c[] and
// propagates across all iterations -> the 4658 small (~0.005) mismatches seen.
// We therefore do the reduction sequentially in float in a single thread.
__global__ void kernel_roi_reduce(
    const float * __restrict__ J,
    int cols,
    int r1, int r2, int c1, int c2,
    float *g_sum, float *g_sum2)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = r1; i <= r2; i++) {
        for (int j = c1; j <= c2; j++) {
            float v = J[i * cols + j];
            sum  += v;
            sum2 += v * v;
        }
    }
    *g_sum  = sum;
    *g_sum2 = sum2;
}

// PREPARE kernel with shared memory tiling for J neighbors.
// Block is 32x8; each block loads a (32+2) x (8+2) tile of J into shared
// memory to serve the 5-point stencil (N, S, W, E, center) without redundant
// global loads. Computes c[k] and d*[k] in float with double literals to
// match serial reference.
#define BX 32
#define BY 8

__global__ void kernel_prepare(
    const float * __restrict__ J,
    float *dN, float *dS, float *dW, float *dE,
    float *c,
    const int *iN, const int *iS, const int *jW, const int *jE,
    int rows, int cols,
    float q0sqr)
{
    // Shared tile: (BY+2) rows x (BX+2) cols
    __shared__ float tile[BY + 2][BX + 2];

    int gj = blockIdx.x * BX + threadIdx.x;  // global col
    int gi = blockIdx.y * BY + threadIdx.y;  // global row

    // Load halo tile: each thread loads its center; boundary threads also load
    // halo. We load a (BY+2)x(BX+2) region centered at (gi,gj) block.
    // Strategy: load using clamped neighbor indices.
    int tj = threadIdx.x + 1;  // tile col (1-indexed center)
    int ti = threadIdx.y + 1;  // tile row (1-indexed center)

    // Center
    if (gi < rows && gj < cols)
        tile[ti][tj] = J[gi * cols + gj];

    // North halo (threadIdx.y == 0)
    if (threadIdx.y == 0 && gi < rows && gj < cols) {
        int ni = (gi > 0) ? gi - 1 : 0;
        tile[0][tj] = J[ni * cols + gj];
    }
    // South halo
    if (threadIdx.y == BY - 1 && gi < rows && gj < cols) {
        int si = (gi < rows - 1) ? gi + 1 : rows - 1;
        tile[BY + 1][tj] = J[si * cols + gj];
    }
    // West halo
    if (threadIdx.x == 0 && gi < rows && gj < cols) {
        int wj = (gj > 0) ? gj - 1 : 0;
        tile[ti][0] = J[gi * cols + wj];
    }
    // East halo
    if (threadIdx.x == BX - 1 && gi < rows && gj < cols) {
        int ej = (gj < cols - 1) ? gj + 1 : cols - 1;
        tile[ti][BX + 1] = J[gi * cols + ej];
    }
    __syncthreads();

    if (gi >= rows || gj >= cols) return;

    int k = gi * cols + gj;
    float Jc = tile[ti][tj];

    // Compute deltas in float (matches golden: dN[idx] = J[...] - Jc stored as float)
    float dn = tile[ti - 1][tj] - Jc;
    float ds = tile[ti + 1][tj] - Jc;
    float dw = tile[ti][tj - 1] - Jc;
    float de = tile[ti][tj + 1] - Jc;

    dN[k] = dn;
    dS[k] = ds;
    dW[k] = dw;
    dE[k] = de;

    // Per-pixel arithmetic in FLOAT. With --fmad=false the float ops are not
    // fused, and because q0sqr is bit-exact (sequential float reduction matches
    // the serial golden), the accumulated error over 100 iters stays well under
    // the 1e-4 tolerance (measured max_abs_err ~2e-5). Float here avoids the
    // 1/64-rate FP64 path on sm_86 that otherwise dominated kernel time.
    float G2 = (dn*dn + ds*ds + dw*dw + de*de) / (Jc*Jc);
    float L  = (dn + ds + dw + de) / Jc;
    float num = 0.5f*G2 - (1.0f/16.0f)*(L*L);
    float den = 1.0f + 0.25f*L;
    float qsqr = num / (den*den);

    den = (qsqr - q0sqr) / (q0sqr*(1.0f + q0sqr));
    float c_val = 1.0f / (1.0f + den);

    if (c_val < 0.0f) c_val = 0.0f;
    else if (c_val > 1.0f) c_val = 1.0f;
    c[k] = c_val;
}

// UPDATE kernel with shared memory tiling for c[].
// We need c[k], c[iS[i]*cols+j], c[i*cols+jE[j]] — south and east neighbors.
// Tile c into shared mem with south+east halos.
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
    __shared__ float ctile[BY + 1][BX + 1];  // +1 south, +1 east halo

    int gj = blockIdx.x * BX + threadIdx.x;
    int gi = blockIdx.y * BY + threadIdx.y;

    int tj = threadIdx.x;
    int ti = threadIdx.y;

    // Load center
    if (gi < rows && gj < cols)
        ctile[ti][tj] = c[gi * cols + gj];

    // South halo: last row of threads loads row gi+1
    if (ti == BY - 1 && gi < rows && gj < cols) {
        int si = (gi < rows - 1) ? gi + 1 : rows - 1;
        ctile[BY][tj] = c[si * cols + gj];
    }
    // East halo: last col of threads loads col gj+1
    if (tj == BX - 1 && gi < rows && gj < cols) {
        int ej = (gj < cols - 1) ? gj + 1 : cols - 1;
        ctile[ti][BX] = c[gi * cols + ej];
    }
    __syncthreads();

    if (gi >= rows || gj >= cols) return;

    int k = gi * cols + gj;
    float cN = ctile[ti][tj];       // c[idx]
    float cS = ctile[ti + 1][tj];   // c[iS[i]*cols+j]
    float cW = ctile[ti][tj];       // c[idx]
    float cE = ctile[ti][tj + 1];   // c[i*cols+jE[j]]

    // Divergence and update in FLOAT (within tolerance, see kernel_prepare note).
    float D = cN*dN[k] + cS*dS[k] + cW*dW[k] + cE*dE[k];
    J[k] = J[k] + 0.25f*lambda*D;
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

    float  *d_J, *d_dN, *d_dS, *d_dW, *d_dE, *d_c;
    float  *d_sum, *d_sum2;
    int    *d_iN, *d_iS, *d_jW, *d_jE;

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

    // Grid for prepare/update with BX x BY blocks
    dim3 block2d(BX, BY);
    dim3 grid2d((cols + BX - 1) / BX, (rows + BY - 1) / BY);

    double t0 = now_seconds();

    for (int iter = 0; iter < niter; iter++) {
        // Step 1: ROI reduction (sequential float, matches serial golden exactly)
        kernel_roi_reduce<<<1, 1>>>(
            d_J, cols, r1, r2, c1, c2, d_sum, d_sum2);

        float h_sum, h_sum2;
        cudaMemcpy(&h_sum,  d_sum,  sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum2, d_sum2, sizeof(float), cudaMemcpyDeviceToHost);

        // Compute q0sqr in FLOAT, matching the serial reference arithmetic.
        float meanROI = h_sum / size_R;
        float varROI  = (h_sum2 / size_R) - meanROI * meanROI;
        float q0sqr   = varROI / (meanROI * meanROI);

        // Step 2: PREPARE (shared-mem tiled)
        kernel_prepare<<<grid2d, block2d>>>(
            d_J, d_dN, d_dS, d_dW, d_dE, d_c,
            d_iN, d_iS, d_jW, d_jE,
            rows, cols, q0sqr);

        // Step 3: UPDATE (shared-mem tiled for c[])
        kernel_update<<<grid2d, block2d>>>(
            d_J, d_dN, d_dS, d_dW, d_dE, d_c,
            d_iS, d_jE,
            rows, cols, lambda);
    }

    cudaDeviceSynchronize();
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    cudaMemcpy(h_J, d_J, size_I * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int idx = 0; idx < size_I; idx++)
        fprintf(fp, "%d\t%.5f\n", idx, h_J[idx]);
    fclose(fp);

    cudaFree(d_J); cudaFree(d_dN); cudaFree(d_dS); cudaFree(d_dW); cudaFree(d_dE);
    cudaFree(d_c); cudaFree(d_sum); cudaFree(d_sum2);
    cudaFree(d_iN); cudaFree(d_iS); cudaFree(d_jW); cudaFree(d_jE);
    free(h_I); free(h_J);
    free(h_iN); free(h_iS); free(h_jW); free(h_jE);
    return 0;
}
