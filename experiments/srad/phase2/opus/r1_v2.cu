/*
 * SRAD (srad_v2) — Phase 2, round 1 (r1_v2).
 *
 * Verified correctness model (from measured feedback, 0 mismatches,
 * max_abs_err=2e-5):
 *   1. ROI reduction accumulates in float in exact serial row-major order
 *      (thread 0 accumulates sequentially in shared memory). Double accumulation
 *      diverges q0sqr from the CPU golden across iterations.
 *   2. q0sqr is float; the prepare math is done in float so the
 *      `den = (qsqr - q0sqr)/(q0sqr*(1+q0sqr))` line matches the serial float
 *      arithmetic. Promoting to double diverges c[k] over iterations.
 *
 * Phase-2 optimizations:
 *   - Fused prepare+update kernel with shared-memory J tile + c tile: avoids the
 *     dN/dS/dW/dE and c[] global arrays and their re-reads (~3x less traffic).
 *   - Thread-coarsening: BX=32, BY=8, RPT=4 → 32 output rows/block, amortizing
 *     c-halo recomputation.
 *   - Double-buffering J (d_J / d_J2) eliminates write-read hazards.
 *   - Reciprocal optimization: rJc = 1/Jc reused for G2 and L.
 *   - __ldg() broadcast read of q0sqr; __restrict__ on kernel pointers.
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

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"cuda error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

/*
 * ROI reduction: cooperatively load ROI into shared memory, then thread 0
 * accumulates sum/sum2 in float in exact row-major order, matching the serial.
 * Dynamic shared memory = roi_size * sizeof(float).
 */
__global__ void reduce_roi(const float * __restrict__ J,
                           int r1, int r2, int c1, int c2,
                           int cols, int size_R, float *d_q0sqr)
{
    extern __shared__ float sbuf[];
    int t  = threadIdx.x;
    int nt = blockDim.x;
    int roi_cols = c2 - c1 + 1;
    int roi_rows = r2 - r1 + 1;
    int roi_size = roi_rows * roi_cols;

    for (int k = t; k < roi_size; k += nt) {
        int rr = k / roi_cols;
        int cc = k % roi_cols;
        sbuf[k] = J[(r1 + rr) * cols + c1 + cc];
    }
    __syncthreads();

    if (t == 0) {
        float sum = 0.0f, sum2 = 0.0f;
        for (int k = 0; k < roi_size; k++) {
            float tmp = sbuf[k];
            sum  += tmp;
            sum2 += tmp * tmp;
        }
        float meanROI = sum / size_R;
        float varROI  = (sum2 / size_R) - meanROI * meanROI;
        *d_q0sqr = varROI / (meanROI * meanROI);
    }
}

/* ---- Fused prepare+update kernel with shared memory tiling ---- */

#define BX 32
#define BY 8
#define RPT 4
#define ROWS_BLK (RPT * BY)      /* 32 output rows per block */
#define HJ_W (BX + 3)
#define HJ_H (ROWS_BLK + 3)
#define CC_W (BX + 1)
#define CC_H (ROWS_BLK + 1)

__device__ __forceinline__ float compute_c(float Jc, float Jn, float Js,
                                           float Jw, float Je, float q0sqr)
{
    float dN = Jn - Jc, dS = Js - Jc, dW = Jw - Jc, dE = Je - Jc;
    float rJc  = 1.0f / Jc;
    float G2   = (dN*dN + dS*dS + dW*dW + dE*dE) * (rJc * rJc);
    float L    = (dN + dS + dW + dE) * rJc;
    float num  = 0.5f * G2 - (1.0f / 16.0f) * (L * L);
    float den  = 1.0f + 0.25f * L;
    float qsqr = num / (den * den);
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    float c_val = 1.0f / (1.0f + den);
    if (c_val < 0.0f) c_val = 0.0f;
    else if (c_val > 1.0f) c_val = 1.0f;
    return c_val;
}

__global__ void srad_kernel(const float * __restrict__ Jin,
                            float * __restrict__ Jout,
                            const float * __restrict__ d_q0sqr,
                            float lambda, int rows, int cols)
{
    __shared__ float Jt[HJ_H][HJ_W];
    __shared__ float ct[CC_H][CC_W];

    int tx = threadIdx.x, ty = threadIdx.y;
    int col0 = blockIdx.x * BX;
    int row0 = blockIdx.y * ROWS_BLK;

    for (int k = ty * BX + tx; k < HJ_H * HJ_W; k += BX * BY) {
        int lr = k / HJ_W;
        int lc = k % HJ_W;
        int gr = row0 + lr - 1;
        int gc = col0 + lc - 1;
        if (gr < 0) gr = 0; else if (gr > rows - 1) gr = rows - 1;
        if (gc < 0) gc = 0; else if (gc > cols - 1) gc = cols - 1;
        Jt[lr][lc] = Jin[gr * cols + gc];
    }
    __syncthreads();

    float q0sqr = __ldg(d_q0sqr);

    for (int k = ty * BX + tx; k < CC_H * CC_W; k += BX * BY) {
        int cr = k / CC_W;
        int cc = k % CC_W;
        int jr = cr + 1, jc = cc + 1;
        float Jc = Jt[jr][jc];
        float Jn = Jt[jr - 1][jc];
        float Js = Jt[jr + 1][jc];
        float Jw = Jt[jr][jc - 1];
        float Je = Jt[jr][jc + 1];
        ct[cr][cc] = compute_c(Jc, Jn, Js, Jw, Je, q0sqr);
    }
    __syncthreads();

    int col = col0 + tx;

    for (int rr = 0; rr < RPT; rr++) {
        int row = row0 + ty + rr * BY;
        if (row >= rows || col >= cols) continue;

        int jty = ty + rr * BY;

        float Jc = Jt[jty + 1][tx + 1];
        float Jn = Jt[jty    ][tx + 1];
        float Js = Jt[jty + 2][tx + 1];
        float Jw = Jt[jty + 1][tx    ];
        float Je = Jt[jty + 1][tx + 2];

        float dN = Jn - Jc, dS = Js - Jc, dW = Jw - Jc, dE = Je - Jc;

        float cN = ct[jty    ][tx    ];
        float cW = ct[jty    ][tx    ];
        float cS = ct[jty + 1][tx    ];
        float cE = ct[jty    ][tx + 1];

        float D = cN * dN + cS * dS + cW * dW + cE * dE;
        Jout[row * cols + col] = Jc + 0.25f * lambda * D;
    }
}

int main(int argc, char **argv) {
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

    int size_I = cols * rows;
    int size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

    float *I = (float *)malloc(size_I * sizeof(float));
    float *J = (float *)malloc(size_I * sizeof(float));
    if (!I || !J) { fprintf(stderr, "alloc failed\n"); return 1; }

    random_matrix(I, rows, cols);
    for (int k = 0; k < size_I; k++)
        J[k] = (float)exp(I[k]);

    float *d_J, *d_J2, *d_q0sqr;
    CK(cudaMalloc(&d_J,     size_I * sizeof(float)));
    CK(cudaMalloc(&d_J2,    size_I * sizeof(float)));
    CK(cudaMalloc(&d_q0sqr, sizeof(float)));

    CK(cudaMemcpy(d_J, J, size_I * sizeof(float), cudaMemcpyHostToDevice));

    int roi_cols = c2 - c1 + 1;
    int roi_rows = r2 - r1 + 1;
    int roi_size = roi_rows * roi_cols;
    int roi_threads = 256;
    size_t roi_shmem = (size_t)roi_size * sizeof(float);

    CK(cudaFuncSetAttribute(reduce_roi,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 100000));

    dim3 tblock(BX, BY);
    dim3 tgrid((cols + BX - 1) / BX,
               (rows + ROWS_BLK - 1) / ROWS_BLK);

    CK(cudaDeviceSynchronize());
    double t0 = now_seconds();

    for (int iter = 0; iter < niter; iter++) {
        reduce_roi<<<1, roi_threads, roi_shmem>>>(
            d_J, r1, r2, c1, c2, cols, size_R, d_q0sqr);

        srad_kernel<<<tgrid, tblock>>>(d_J, d_J2, d_q0sqr, lambda, rows, cols);

        float *tmp = d_J; d_J = d_J2; d_J2 = tmp;
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

    CK(cudaFree(d_J)); CK(cudaFree(d_J2)); CK(cudaFree(d_q0sqr));
    free(I); free(J);
    return 0;
}
