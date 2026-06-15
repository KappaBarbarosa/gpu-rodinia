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

// ROI reduction. The serial golden accumulates the ROI sum/sum2 as a single
// running float chain in row-major (i,j) order; q0sqr is chaotically sensitive
// to that exact rounding over 100 iterations (a tree/per-row split or even an
// exact double sum drifts past the 1e-4 tolerance). So we MUST keep the exact
// sequential add order. To kill the single-thread global-memory-latency stall
// of the original <<<1,1>>> kernel, the block cooperatively stages the ROI into
// shared memory with coalesced parallel loads (store order is irrelevant), then
// thread 0 runs the exact serial-order add chain out of shared memory.
// q0sqr is computed on-device to avoid a per-iteration host round-trip/sync.
__global__ void kernel_roi_reduce(
    const float * __restrict__ J,
    int cols,
    int r1, int r2, int c1, int c2,
    int size_R,
    float *g_q0sqr)
{
    int nrows = r2 - r1 + 1;
    int ncols = c2 - c1 + 1;
    extern __shared__ float roi[];   // nrows*ncols floats

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int total = nrows * ncols;
    // Coalesced parallel load (order irrelevant for storing)
    for (int idx = tid; idx < total; idx += nthreads) {
        int rr = idx / ncols;
        int cc = idx - rr * ncols;
        roi[idx] = __ldg(&J[(r1 + rr) * cols + (c1 + cc)]);
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f, sum2 = 0.0f;
        for (int idx = 0; idx < total; idx++) {
            float v = roi[idx];     // serial order == (i,j) row-major
            sum  += v;
            sum2 += v * v;
        }
        float meanROI = sum / size_R;
        float varROI  = (sum2 / size_R) - meanROI * meanROI;
        g_q0sqr[0] = varROI / (meanROI * meanROI);
    }
}

#define BX 32
#define BY 16

__global__ void kernel_prepare(
    const float * __restrict__ J,
    float * __restrict__ dN,
    float * __restrict__ dS,
    float * __restrict__ dW,
    float * __restrict__ dE,
    float * __restrict__ c,
    int rows, int cols,
    const float * __restrict__ g_q0sqr)
{
    __shared__ float tile[BY + 2][BX + 2];
    float q0sqr = g_q0sqr[0];

    int gj = blockIdx.x * BX + threadIdx.x;
    int gi = blockIdx.y * BY + threadIdx.y;

    int tj = threadIdx.x + 1;
    int ti = threadIdx.y + 1;

    if (gi < rows && gj < cols)
        tile[ti][tj] = __ldg(&J[gi * cols + gj]);

    if (threadIdx.y == 0 && gj < cols) {
        int ni = (gi > 0) ? gi - 1 : 0;
        if (ni < rows)
            tile[0][tj] = __ldg(&J[ni * cols + gj]);
        else
            tile[0][tj] = 0.0f;
    }
    if (threadIdx.y == BY - 1 && gj < cols) {
        int si = (gi < rows - 1) ? gi + 1 : rows - 1;
        if (si < rows && gi < rows)
            tile[BY + 1][tj] = __ldg(&J[si * cols + gj]);
        else if (gi < rows)
            tile[BY + 1][tj] = __ldg(&J[(rows-1) * cols + gj]);
        else
            tile[BY + 1][tj] = 0.0f;
    }
    if (threadIdx.x == 0 && gi < rows) {
        int wj = (gj > 0) ? gj - 1 : 0;
        tile[ti][0] = __ldg(&J[gi * cols + wj]);
    }
    if (threadIdx.x == BX - 1 && gi < rows) {
        int ej = (gj < cols - 1) ? gj + 1 : cols - 1;
        if (gj < cols)
            tile[ti][BX + 1] = __ldg(&J[gi * cols + ej]);
        else
            tile[ti][BX + 1] = __ldg(&J[gi * cols + (cols-1)]);
    }
    __syncthreads();

    if (gi >= rows || gj >= cols) return;

    int k = gi * cols + gj;
    float Jc = tile[ti][tj];

    float dn = tile[ti - 1][tj] - Jc;
    float ds = tile[ti + 1][tj] - Jc;
    float dw = tile[ti][tj - 1] - Jc;
    float de = tile[ti][tj + 1] - Jc;

    dN[k] = dn;
    dS[k] = ds;
    dW[k] = dw;
    dE[k] = de;

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

__global__ void kernel_update(
    float * __restrict__ J,
    const float * __restrict__ dN,
    const float * __restrict__ dS,
    const float * __restrict__ dW,
    const float * __restrict__ dE,
    const float * __restrict__ c,
    int rows, int cols,
    float lambda)
{
    __shared__ float ctile[BY + 1][BX + 1];

    int gj = blockIdx.x * BX + threadIdx.x;
    int gi = blockIdx.y * BY + threadIdx.y;

    int tj = threadIdx.x;
    int ti = threadIdx.y;

    if (gi < rows && gj < cols)
        ctile[ti][tj] = __ldg(&c[gi * cols + gj]);

    if (ti == BY - 1 && gj < cols) {
        int si = (gi < rows - 1) ? gi + 1 : rows - 1;
        if (gi < rows)
            ctile[BY][tj] = __ldg(&c[si * cols + gj]);
        else
            ctile[BY][tj] = 0.0f;
    }
    if (tj == BX - 1 && gi < rows) {
        int ej = (gj < cols - 1) ? gj + 1 : cols - 1;
        if (gj < cols)
            ctile[ti][BX] = __ldg(&c[gi * cols + ej]);
        else
            ctile[ti][BX] = __ldg(&c[gi * cols + (cols-1)]);
    }
    __syncthreads();

    if (gi >= rows || gj >= cols) return;

    int k = gi * cols + gj;
    float cN = ctile[ti][tj];
    float cS = ctile[ti + 1][tj];
    float cW = ctile[ti][tj];
    float cE = ctile[ti][tj + 1];

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

    float *h_I, *h_J;
    cudaMallocHost(&h_I, size_I * sizeof(float));
    cudaMallocHost(&h_J, size_I * sizeof(float));

    random_matrix(h_I, rows, cols);
    for (int k = 0; k < size_I; k++)
        h_J[k] = (float)exp(h_I[k]);

    float  *d_J, *d_dN, *d_dS, *d_dW, *d_dE, *d_c, *d_q0sqr;

    cudaMalloc(&d_J,    size_I * sizeof(float));
    cudaMalloc(&d_dN,   size_I * sizeof(float));
    cudaMalloc(&d_dS,   size_I * sizeof(float));
    cudaMalloc(&d_dW,   size_I * sizeof(float));
    cudaMalloc(&d_dE,   size_I * sizeof(float));
    cudaMalloc(&d_c,    size_I * sizeof(float));
    cudaMalloc(&d_q0sqr, sizeof(float));

    cudaMemcpy(d_J, h_J, size_I * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block2d(BX, BY);
    dim3 grid2d((cols + BX - 1) / BX, (rows + BY - 1) / BY);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaFuncSetAttribute(kernel_roi_reduce, cudaFuncAttributeMaxDynamicSharedMemorySize, size_R*sizeof(float));

    double t0 = now_seconds();

    for (int iter = 0; iter < niter; iter++) {
        kernel_roi_reduce<<<1, 256, size_R*sizeof(float), stream>>>(
            d_J, cols, r1, r2, c1, c2, size_R, d_q0sqr);

        kernel_prepare<<<grid2d, block2d, 0, stream>>>(
            d_J, d_dN, d_dS, d_dW, d_dE, d_c,
            rows, cols, d_q0sqr);

        kernel_update<<<grid2d, block2d, 0, stream>>>(
            d_J, d_dN, d_dS, d_dW, d_dE, d_c,
            rows, cols, lambda);
    }

    cudaStreamSynchronize(stream);
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    cudaMemcpy(h_J, d_J, size_I * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int idx = 0; idx < size_I; idx++)
        fprintf(fp, "%d\t%.5f\n", idx, h_J[idx]);
    fclose(fp);

    cudaStreamDestroy(stream);
    cudaFree(d_J); cudaFree(d_dN); cudaFree(d_dS); cudaFree(d_dW); cudaFree(d_dE);
    cudaFree(d_c); cudaFree(d_q0sqr);
    cudaFreeHost(h_I); cudaFreeHost(h_J);
    return 0;
}
