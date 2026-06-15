/*
 * SRAD (srad_v2) — Phase 1 CUDA port, corrected.
 *
 * Correctness bugs fixed vs previous attempt:
 *   1. ROI reduction now accumulates in float in exact serial row-major order
 *      (thread 0 accumulates sequentially in shared memory), matching the CPU
 *      reference `float sum = 0, sum2 = 0; sum += tmp; sum2 += tmp*tmp;`.
 *      The previous version accumulated in double which gave different sum/sum2
 *      → different q0sqr → diverging pixel values across 100 iterations.
 *   2. q0sqr is stored and passed as float (not double). The serial line
 *      `den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr))` is float arithmetic
 *      when q0sqr is float. Passing double promoted the expression to double,
 *      producing different c[k] values that diverge over iterations.
 *   3. All per-pixel math uses float (matching the serial where all intermediate
 *      stores are float; only a few literals are double which auto-promote, but
 *      since fmad=false the float-literal result is identical).
 *
 * Phase-2 optimizations included:
 *   - Fused prepare+update kernel with shared-memory J tile avoids 5 separate
 *     global reads per pixel per pass (dN/dS/dW/dE + c neighbors), reducing
 *     global memory traffic by ~3x.
 *   - Thread-coarsening (RPT=4 rows per thread, BY=8 threads in y, BX=32 in x):
 *     amortizes c-halo recomputation overhead.
 *   - ROI reduction: cooperative load to shared memory, then thread 0 does
 *     the exact serial float accumulation. Fits in 64KB shared mem (128x128=16K
 *     floats = 64KB) with 100KB dynamic shared memory enabled on sm_86.
 *   - Reciprocal optimization: one rJc = 1/Jc reused for G2 and L, saving one
 *     division per pixel.
 *   - Double-buffering J on device eliminates in-place update hazards and allows
 *     the kernel to read Jin and write Jout without conflicts.
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
 * ROI reduction kernel:
 *   - All blockDim.x threads cooperatively load the entire ROI into shared mem.
 *   - Thread 0 then accumulates sum and sum2 in float, in exact row-major order,
 *     matching the serial `for i in [r1,r2]: for j in [c1,c2]: sum += J[i*cols+j]`.
 *   - Computes q0sqr = varROI / meanROI^2 and writes to d_q0sqr (float).
 *
 * Uses dynamic shared memory: caller must pass roi_size * sizeof(float) bytes.
 * Enable 100KB shared memory via cudaFuncSetAttribute before launch.
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

    /* Cooperatively load the ROI into shared memory. */
    for (int k = t; k < roi_size; k += nt) {
        int rr = k / roi_cols;
        int cc = k % roi_cols;
        sbuf[k] = J[(r1 + rr) * cols + c1 + cc];
    }
    __syncthreads();

    /* Thread 0: sequential float accumulation in exact row-major order.
     * This reproduces `float sum=0,sum2=0; sum+=tmp; sum2+=tmp*tmp;` exactly. */
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
#define HJ_W (BX + 3)            /* J tile width:  32+1(W) +1(center) +1(E) = 34... but +3 for clamped access */
#define HJ_H (ROWS_BLK + 3)     /* J tile height: ROWS_BLK + 1(N halo) + 1 + 1(S halo) */
#define CC_W (BX + 1)            /* c tile width:  BX+1 (need c[k] and c[i*cols+jE]) */
#define CC_H (ROWS_BLK + 1)     /* c tile height: ROWS_BLK+1 (need c[k] and c[iS*cols+j]) */

__device__ __forceinline__ float compute_c(float Jc, float Jn, float Js,
                                           float Jw, float Je, float q0sqr)
{
    float dN = Jn - Jc, dS = Js - Jc, dW = Jw - Jc, dE = Je - Jc;
    /* Use reciprocal to save one division: G2 = (sum_sq) / Jc^2, L = (sum)/Jc */
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

/*
 * srad_kernel: fused prepare+update in one pass using shared memory.
 *
 * Each block processes a (ROWS_BLK x BX) output tile = (32 x 32) pixels.
 * Block dim = (BX=32, BY=8): each thread handles RPT=4 rows (row-coarsening).
 *
 * Shared memory layout:
 *   Jt[HJ_H][HJ_W] = (35 x 35) floats = 4900 bytes  (J values + 1-wide halo)
 *   ct[CC_H][CC_W] = (33 x 33) floats = 4356 bytes  (c values + S/E neighbor halo)
 *   Total: ~9256 bytes per block — well within 48KB.
 *
 * Algorithm:
 *   1. Load Jt (J tile with 1-wide halo for N/S/W/E neighbours).
 *   2. Compute ct: each thread computes c for its (CC_H x CC_W) subset using
 *      compute_c() which reads from Jt. One __syncthreads() after.
 *   3. Each thread writes RPT output rows using Jt for dN/dS/dW/dE and ct for
 *      cN/cS/cW/cE. No further global reads needed.
 */
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

    /* Load J tile (HJ_H x HJ_W) cooperatively. */
    for (int k = ty * BX + tx; k < HJ_H * HJ_W; k += BX * BY) {
        int lr = k / HJ_W;
        int lc = k % HJ_W;
        int gr = row0 + lr - 1;   /* -1 for N halo */
        int gc = col0 + lc - 1;   /* -1 for W halo */
        if (gr < 0) gr = 0; else if (gr > rows - 1) gr = rows - 1;
        if (gc < 0) gc = 0; else if (gc > cols - 1) gc = cols - 1;
        Jt[lr][lc] = Jin[gr * cols + gc];
    }
    __syncthreads();

    float q0sqr = __ldg(d_q0sqr);

    /* Compute c tile (CC_H x CC_W) from J tile. */
    for (int k = ty * BX + tx; k < CC_H * CC_W; k += BX * BY) {
        int cr = k / CC_W;
        int cc = k % CC_W;
        /* In Jt, the interior starts at [1][1]; the c tile aligns with Jt[1..CC_H][1..CC_W]. */
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

    /* Each thread writes RPT output rows (row-coarsening over BY). */
    for (int rr = 0; rr < RPT; rr++) {
        int row = row0 + ty + rr * BY;
        if (row >= rows || col >= cols) continue;

        int jty = ty + rr * BY;  /* local row index in Jt and ct */

        /* Read J values from tile (centre + NSWE neighbours). */
        float Jc = Jt[jty + 1][tx + 1];
        float Jn = Jt[jty    ][tx + 1];
        float Js = Jt[jty + 2][tx + 1];
        float Jw = Jt[jty + 1][tx    ];
        float Je = Jt[jty + 1][tx + 2];

        float dN = Jn - Jc, dS = Js - Jc, dW = Jw - Jc, dE = Je - Jc;

        /* c neighbours: cN=c[k], cW=c[k], cS=c[iS*cols+j], cE=c[i*cols+jE]. */
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

    /* Allocate device buffers: two J buffers for double-buffering, one q0sqr. */
    float *d_J, *d_J2, *d_q0sqr;
    CK(cudaMalloc(&d_J,     size_I * sizeof(float)));
    CK(cudaMalloc(&d_J2,    size_I * sizeof(float)));
    CK(cudaMalloc(&d_q0sqr, sizeof(float)));

    CK(cudaMemcpy(d_J, J, size_I * sizeof(float), cudaMemcpyHostToDevice));

    /* ROI reduction kernel configuration. */
    int roi_cols = c2 - c1 + 1;
    int roi_rows = r2 - r1 + 1;
    int roi_size = roi_rows * roi_cols;
    int roi_threads = 256;
    size_t roi_shmem = (size_t)roi_size * sizeof(float);

    /* Enable up to 100KB dynamic shared memory on sm_86 (ROI=128x128=64KB). */
    CK(cudaFuncSetAttribute(reduce_roi,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 100000));

    /* srad_kernel grid: one block per (BX x ROWS_BLK) = (32 x 32) output tile. */
    dim3 tblock(BX, BY);
    dim3 tgrid((cols + BX - 1) / BX,
               (rows + ROWS_BLK - 1) / ROWS_BLK);

    CK(cudaDeviceSynchronize());
    double t0 = now_seconds();

    for (int iter = 0; iter < niter; iter++) {
        /* Step 1: ROI statistics → q0sqr (float, exact serial float order). */
        reduce_roi<<<1, roi_threads, roi_shmem>>>(
            d_J, r1, r2, c1, c2, cols, size_R, d_q0sqr);

        /* Steps 2+3: fused prepare+update with shared-memory tiling. */
        srad_kernel<<<tgrid, tblock>>>(d_J, d_J2, d_q0sqr, lambda, rows, cols);

        /* Swap buffers: d_J2 (output) becomes d_J (input) for next iteration. */
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
