/*
 * SRAD (srad_v2) — Phase 2 r4: two-kernel-per-iteration, launch-overhead trimmed.
 *
 * Correctness baseline (from r3, verified): per-iteration
 *   reduce_roi<<<1,1>>>  — ONE thread, exact continuous left-to-right row-major
 *                          float running sum -> q0sqr (NOT parallelized; float
 *                          add is non-associative and q0sqr feeds every pixel).
 *   srad_fused<<<grid,block>>> — tiled fused prepare+update, per-pixel math in
 *                          double then stored to float, matching the serial
 *                          golden bit-closely under --fmad=false.
 *
 * r4 optimization (no numeric change): the per-iteration cost was dominated by
 * launch latency and the implicit serialization between the two kernels, not
 * arithmetic. We cannot grid-sync (16384 blocks >> 138 co-resident), and we
 * must not change the ROI float add order. So:
 *  - All kernels are enqueued on ONE explicit stream and the host NEVER
 *    synchronizes inside the niter loop. The two kernels per iteration and the
 *    reduce of iter+1 all queue back-to-back; the driver pipelines launch
 *    latency under kernel execution instead of stalling each iteration.
 *  - srad_fused is given the __launch_bounds__ hint and the d_q0 read is the
 *    only cross-kernel dependency, satisfied by in-stream ordering (no host
 *    sync needed). One cudaDeviceSynchronize() after the whole loop.
 *  - reduce_roi keeps <<<1,1>>>: its work is a strict serial dependency chain,
 *    so extra threads cannot help and would risk reordering the float adds.
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

// ROI reduction: ONE thread, exact continuous left-to-right row-major float
// running sum, then q0sqr. Bit-identical to r2/r3.
__global__ void reduce_roi(const float *__restrict__ Jin, int cols,
                           int r1, int r2, int c1, int c2, int size_R,
                           float *d_q0)
{
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = r1; i <= r2; i++) {
        for (int j = c1; j <= c2; j++) {
            float tmp = Jin[i * cols + j];
            sum  += tmp;
            sum2 += tmp * tmp;
        }
    }
    float meanROI = sum / size_R;
    float varROI  = (sum2 / size_R) - meanROI * meanROI;
    *d_q0 = varROI / (meanROI * meanROI);
}

__global__ void __launch_bounds__(TILE *TILE)
srad_fused(const float *__restrict__ Jin,
           float *__restrict__ Jout,
           int rows, int cols, float lambda,
           const float *__restrict__ d_q0)
{
    __shared__ float s_c [TILE + 1][TILE + 1];
    __shared__ float s_dN[TILE + 1][TILE + 1];
    __shared__ float s_dS[TILE + 1][TILE + 1];
    __shared__ float s_dW[TILE + 1][TILE + 1];
    __shared__ float s_dE[TILE + 1][TILE + 1];

    float q0sqr = *d_q0;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE + tx;
    int gy = blockIdx.y * TILE + ty;

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

    CK(cudaMemcpyAsync(d_J, J, size_I * sizeof(float), cudaMemcpyHostToDevice, 0));

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    // Single explicit stream; in-stream ordering enforces reduce_roi -> srad_fused
    // (d_q0 producer/consumer) and srad_fused -> next reduce_roi (J producer)
    // without any host synchronization inside the loop, so launch latency
    // pipelines under kernel execution.
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));

    CK(cudaStreamSynchronize(stream));
    CK(cudaDeviceSynchronize());
    double t0 = now_seconds();

    for (int iter = 0; iter < niter; iter++) {
        const float *Jin  = (iter & 1) ? d_J2 : d_J;
        float       *Jout = (iter & 1) ? d_J  : d_J2;

        reduce_roi<<<1, 1, 0, stream>>>(Jin, cols, r1, r2, c1, c2, size_R, d_q0);
        srad_fused<<<grid, block, 0, stream>>>(Jin, Jout, rows, cols, lambda, d_q0);
    }
    CK(cudaGetLastError());

    CK(cudaStreamSynchronize(stream));
    CK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    float *d_final = (niter & 1) ? d_J2 : d_J;
    CK(cudaMemcpy(J, d_final, size_I * sizeof(float), cudaMemcpyDeviceToHost));

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int idx = 0; idx < size_I; idx++)
        fprintf(fp, "%d\t%.5f\n", idx, J[idx]);
    fclose(fp);

    cudaStreamDestroy(stream);
    cudaFree(d_J);
    cudaFree(d_J2);
    cudaFree(d_q0);
    free(I); free(J);
    return 0;
}
