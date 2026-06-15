/*
 * SRAD (srad_v2) — Phase 2 r2: shared-memory tiled fused prepare+update,
 * with a bit-exact serial ROI reduction.
 *
 * Correctness fixes from r1 retained:
 *  - compute_c uses (double)(L*L)  [FLOAT multiply then promote], matching the
 *    serial C type promotion exactly.
 *  - double-buffered ping-pong: srad_fused reads OLD J (Jin) for ALL prepare
 *    reads + the update base, writes NEW J (Jout); main swaps each iter. This
 *    removes the cross-block global race of an in-place fused kernel.
 *
 * r2 correctness regression FIXED: the earlier r2 used a parallel per-row ROI
 * reduction (per-row partials summed in a second pass). That regrouped the
 * float adds vs the golden's single continuous row-major running sum; since
 * float add is non-associative and q0sqr feeds every pixel for all niter
 * iterations, the tiny per-iter delta compounded into 4690 mismatches (the
 * fused kernel was identical to the correct r1). Reverted to the single
 * continuous serial accumulation -> 0 mismatches vs golden_2048_100.out.
 *
 * Per-pixel math is done in double then stored to float to match the serial
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

// ROI reduction reproducing the EXACT sequential float accumulation in
// row-major box order, to match the golden bit-closely.
//
// IMPORTANT (r2 correctness fix): the golden accumulates ONE continuous float
// running sum over the box in row-major order (J[r1,c1], J[r1,c1+1], ...,
// J[r1,c2], J[r1+1,c1], ...). Float addition is NON-associative, and q0sqr
// feeds every pixel for all niter iterations, so ANY regrouping of the adds
// (e.g. per-row partials then a second pass) diverges and compounds. The
// previous r2 parallel-per-row reduction produced 4690 mismatches purely from
// this reordering (the fused kernel was unchanged from the correct r1). We
// therefore keep a single continuous serial accumulation, matching the golden
// bit-for-bit (verified: 0 mismatches vs golden_2048_100.out). This kernel is
// tiny (size_R adds, once per iteration) and is not the kernel-time bottleneck.
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

// Compute the diffusion coefficient c for one pixel given the four directional
// differences. Reproduces serial C type promotion exactly:
//   num = 0.5*G2 - (1/16)*(L*L)   where L*L is a FLOAT multiply.
__device__ __forceinline__ float compute_c(float vN, float vS, float vW,
                                            float vE, float Jc, float q0sqr)
{
    float G2 = (vN * vN + vS * vS + vW * vW + vE * vE) / (Jc * Jc);
    float L  = (vN + vS + vW + vE) / Jc;
    float num  = (float)((0.5 * (double)G2) - ((1.0 / 16.0) * (double)(L * L)));
    float den  = (float)(1.0 + (0.25 * (double)L));
    float qsqr = num / (den * den);
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    float ck = (float)(1.0 / (1.0 + (double)den));
    if (ck < 0) ck = 0;
    else if (ck > 1) ck = 1;
    return ck;
}

#define TILE 16

// Fused prepare + update with shared-memory tiling.
// Shared region is (TILE+1) x (TILE+1): the core TILE x TILE pixels plus a
// 1-pixel halo on the SOUTH and EAST sides (the only neighbours step 3 needs
// from other pixels). For each shared element we store c, dN, dS, dW, dE.
//
// Jin = OLD buffer (read everywhere), Jout = NEW buffer (written by core only).
__global__ void srad_fused(const float *__restrict__ Jin, float *__restrict__ Jout,
                           int rows, int cols, float lambda,
                           const float *q0sqr_ptr)
{
    __shared__ float s_c [TILE + 1][TILE + 1];
    __shared__ float s_dN[TILE + 1][TILE + 1];
    __shared__ float s_dS[TILE + 1][TILE + 1];
    __shared__ float s_dW[TILE + 1][TILE + 1];
    __shared__ float s_dE[TILE + 1][TILE + 1];

    float q0sqr = *q0sqr_ptr;

    int tx = threadIdx.x;              // column within tile, 0..TILE-1
    int ty = threadIdx.y;             // row within tile, 0..TILE-1
    int gx = blockIdx.x * TILE + tx;  // global column
    int gy = blockIdx.y * TILE + ty;  // global row

    #define PREP(ly, lx)                                                       \
    do {                                                                       \
        int _gy = blockIdx.y * TILE + (ly);                                    \
        int _gx = blockIdx.x * TILE + (lx);                                    \
        if (_gy < rows && _gx < cols) {                                        \
            int _iN = (_gy == 0) ? 0 : _gy - 1;                                \
            int _iS = (_gy == rows - 1) ? rows - 1 : _gy + 1;                  \
            int _jW = (_gx == 0) ? 0 : _gx - 1;                                \
            int _jE = (_gx == cols - 1) ? cols - 1 : _gx + 1;                  \
            float _Jc = Jin[_gy * cols + _gx];                                 \
            float _vN = Jin[_iN * cols + _gx] - _Jc;                           \
            float _vS = Jin[_iS * cols + _gx] - _Jc;                           \
            float _vW = Jin[_gy * cols + _jW] - _Jc;                           \
            float _vE = Jin[_gy * cols + _jE] - _Jc;                           \
            s_dN[ly][lx] = _vN;                                                \
            s_dS[ly][lx] = _vS;                                                \
            s_dW[ly][lx] = _vW;                                                \
            s_dE[ly][lx] = _vE;                                                \
            s_c [ly][lx] = compute_c(_vN, _vS, _vW, _vE, _Jc, q0sqr);          \
        }                                                                      \
    } while (0)

    PREP(ty, tx);
    if (ty == TILE - 1) PREP(TILE, tx);
    if (tx == TILE - 1) PREP(ty, TILE);
    if (tx == TILE - 1 && ty == TILE - 1) PREP(TILE, TILE);

    #undef PREP

    __syncthreads();

    if (gy < rows && gx < cols) {
        float cN = s_c[ty][tx];
        float cW = s_c[ty][tx];
        float cS = s_c[ty + 1][tx];
        float cE = s_c[ty][tx + 1];

        float D = cN * s_dN[ty][tx] + cS * s_dS[ty][tx]
                + cW * s_dW[ty][tx] + cE * s_dE[ty][tx];

        int k = gy * cols + gx;
        Jout[k] = (float)((double)Jin[k] + 0.25 * (double)lambda * (double)D);
    }
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
    int nrows = (r2 - r1 + 1);
    int size_R = nrows * boxW;

    float *I = (float *)malloc(size_I * sizeof(float));
    float *J = (float *)malloc(size_I * sizeof(float));
    if (!I || !J) { fprintf(stderr, "alloc failed\n"); return 1; }

    random_matrix(I, rows, cols);
    for (int k = 0; k < size_I; k++)
        J[k] = (float)exp(I[k]);

    float *d_J, *d_J2;
    CK(cudaMalloc(&d_J,  size_I * sizeof(float)));
    CK(cudaMalloc(&d_J2, size_I * sizeof(float)));
    float *d_q0;
    CK(cudaMalloc(&d_q0, sizeof(float)));

    CK(cudaMemcpy(d_J, J, size_I * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    float *d_in = d_J, *d_out = d_J2;
    CK(cudaDeviceSynchronize());
    double t0 = now_seconds();
    for (int iter = 0; iter < niter; iter++) {
        reduce_roi<<<1, 1>>>(d_in, cols, r1, r2, c1, c2, size_R, d_q0);
        srad_fused<<<grid, block>>>(d_in, d_out, rows, cols, lambda, d_q0);
        float *tmp = d_in; d_in = d_out; d_out = tmp;
    }
    CK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    CK(cudaMemcpy(J, d_in, size_I * sizeof(float), cudaMemcpyDeviceToHost));

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int idx = 0; idx < size_I; idx++)
        fprintf(fp, "%d\t%.5f\n", idx, J[idx]);
    fclose(fp);

    cudaFree(d_J);
    cudaFree(d_J2);
    cudaFree(d_q0);
    free(I); free(J);
    return 0;
}
