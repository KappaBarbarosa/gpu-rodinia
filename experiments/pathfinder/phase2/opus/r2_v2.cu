/*
 * PathFinder — CUDA phase 2 (temporal tiling / pyramidal blocking).
 *
 * CLI:  pathfinder <cols> <rows> <output_file>
 * Wall generated deterministically on host via srand(9), i (rows) outer,
 * j (cols) inner, wall[i][j] = rand() % 10.
 *
 * r2_v2: carries forward the proven r1_v3 kernel. Feedback established this
 * workload is DRAM-bandwidth-bound (~82% of 448 GB/s peak; hard floor ~0.89 ms
 * set by 381 MB irreducible wall reads across 14 launches). All further
 * approaches tested (PH tuning, higher unroll, warp shuffle, L2 policy, CUDA
 * graphs, all-threads-write, BLOCK_SIZE=256, flat shared mem) gave only noise
 * or regressions. Winning changes retained:
 *  - Register-cached center value (my_val) — 1 fewer shared read/iter.
 *  - __launch_bounds__(512, 3): 37 -> 19 regs/thread, no spills.
 *  - cur ^= 1 swap avoids a temp register.
 *  - PYRAMID_HEIGHT=72: 97.1% wave efficiency, 2 real waves on 46 SMs.
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define M_SEED 9

#define BLOCK_SIZE     512
#define PYRAMID_HEIGHT 72

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define IN_RANGE(x, mn, mx) ((x) >= (mn) && (x) <= (mx))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__
__launch_bounds__(BLOCK_SIZE, 3)
void dynproc(int iteration, const int *__restrict__ gpuWall,
             const int *__restrict__ src, int *__restrict__ dst,
             int cols, int startStep, int border)
{
    __shared__ int buf[2][BLOCK_SIZE];

    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int small_block_cols = BLOCK_SIZE - 2 * border;
    const int blkX    = small_block_cols * bx - border;
    const int blkXmax = blkX + BLOCK_SIZE - 1;
    const int xidx    = blkX + tx;

    const int validXmin = (blkX < 0) ? -blkX : 0;
    const int validXmax = (blkXmax > cols - 1)
                          ? BLOCK_SIZE - 1 - (blkXmax - (cols - 1))
                          : BLOCK_SIZE - 1;

    int W = (tx - 1 < validXmin) ? validXmin : tx - 1;
    int E = (tx + 1 > validXmax) ? validXmax : tx + 1;

    const bool isValid = IN_RANGE(tx, validXmin, validXmax);

    int cur = 0;

    if (IN_RANGE(xidx, 0, cols - 1))
        buf[0][tx] = __ldg(&src[xidx]);
    __syncthreads();

    int my_val = buf[0][tx];

    #pragma unroll 4
    for (int i = 0; i < iteration; i++) {
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
            int left   = buf[cur][W];
            int right  = buf[cur][E];
            int m      = MIN(left, MIN(my_val, right));
            int wval   = __ldg(&gpuWall[(size_t)(startStep + i + 1) * cols + xidx]);
            my_val     = wval + m;
            buf[cur ^ 1][tx] = my_val;
        }
        __syncthreads();
        cur ^= 1;
    }

    if (IN_RANGE(tx, border, BLOCK_SIZE - 1 - border) &&
        IN_RANGE(xidx, 0, cols - 1) && isValid) {
        dst[xidx] = buf[cur][tx];
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <cols> <rows> <output_file>\n", argv[0]);
        return 1;
    }
    int cols = atoi(argv[1]);
    int rows = atoi(argv[2]);
    const char *ofile = argv[3];
    if (cols <= 0 || rows <= 0) {
        fprintf(stderr, "invalid arguments\n");
        return 1;
    }

    int *wall   = (int *)malloc((size_t)rows * cols * sizeof(int));
    int *result = (int *)malloc((size_t)cols * sizeof(int));
    if (!wall || !result) { fprintf(stderr, "alloc failed\n"); return 1; }

    srand(M_SEED);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            wall[(size_t)i * cols + j] = rand() % 10;

    int *d_wall, *d_a, *d_b;
    cudaMalloc(&d_wall, (size_t)rows * cols * sizeof(int));
    cudaMalloc(&d_a,    (size_t)cols * sizeof(int));
    cudaMalloc(&d_b,    (size_t)cols * sizeof(int));
    cudaMemcpy(d_wall, wall, (size_t)rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_a, d_wall, (size_t)cols * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaFuncSetCacheConfig(dynproc, cudaFuncCachePreferL1);

    const int border          = PYRAMID_HEIGHT;
    const int small_block_cols = BLOCK_SIZE - 2 * border;
    const int gridDimX        = (cols + small_block_cols - 1) / small_block_cols;

    int *src = d_a;
    int *dst = d_b;

    const int total_steps = rows - 1;

    cudaDeviceSynchronize();
    double t0 = now_seconds();

    for (int step = 0; step < total_steps; step += PYRAMID_HEIGHT) {
        int iteration = MIN(PYRAMID_HEIGHT, total_steps - step);
        dynproc<<<gridDimX, BLOCK_SIZE>>>(iteration, d_wall, src, dst,
                                          cols, step, border);
        int *tmp = src; src = dst; dst = tmp;
    }

    cudaDeviceSynchronize();
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    cudaMemcpy(result, src, (size_t)cols * sizeof(int), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int n = 0; n < cols; n++)
        fprintf(fp, "%d\t%d\n", n, result[n]);
    fclose(fp);

    cudaFree(d_wall); cudaFree(d_a); cudaFree(d_b);
    free(wall); free(result);
    return 0;
}
