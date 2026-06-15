#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define M_SEED 9

/*
 * Pyramidal (temporal-blocking / ghost-zone) PathFinder.
 *
 * Round 3: same BS=512/PH=72 sweet spot, plus micro-optimizations:
 *   - Use __ldg for gpuSrc initial load (was already done for wall).
 *   - Precompute wallStep pointer once outside the loop.
 *   - Use registers for W/E neighbor values instead of re-indexing shared mem
 *     twice (reduces shared-memory bank conflicts in the min reduction).
 *   - Mark shared memory with volatile only where needed (avoid spurious fences).
 *   - Use warp-level min where the stencil width fits (for the boundary threads
 *     near the valid region, warp shuffles can replace shared-mem reads for
 *     left/right neighbors — avoids bank conflict and saves 2 smem reads/step).
 *   - The inner loop unroll pragma is kept at 4 (matches the PH=72 % 4 == 0
 *     structure so no remainder iteration).
 */

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef PYRAMID_HEIGHT
#define PYRAMID_HEIGHT 72
#endif

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define IN_RANGE(x, mn, mx) ((x) >= (mn) && (x) <= (mx))

__global__
__launch_bounds__(BLOCK_SIZE, 2)
void dp_pyramid(int iteration,
                const int * __restrict__ gpuWall,
                const int * __restrict__ gpuSrc,
                int       * __restrict__ gpuDst,
                int cols, int rows,
                int startStep,
                int border)
{
    __shared__ int buf[2][BLOCK_SIZE];

    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int small_block_cols = BLOCK_SIZE - iteration * 2;
    const int blkX    = small_block_cols * bx - border;
    const int blkXmax = blkX + BLOCK_SIZE - 1;
    const int xidx    = blkX + tx;

    const int validXmin = (blkX < 0)         ? -blkX                              : 0;
    const int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1)
                                              : BLOCK_SIZE - 1;

    /* Boundary-clamped W/E neighbour indices — fixed for entire pyramid. */
    const int W = (tx - 1 < validXmin) ? validXmin : tx - 1;
    const int E = (tx + 1 > validXmax) ? validXmax : tx + 1;

    int cur = 0;
    const int in_range = IN_RANGE(xidx, 0, cols - 1);

    /* Load initial DP row from global memory. */
    buf[cur][tx] = in_range ? __ldg(&gpuSrc[xidx]) : 0;
    __syncthreads();

    /* Precompute wall row pointer; stride by cols each iteration. */
    const int * __restrict__ wallStep = gpuWall + (long)(startStep + 1) * cols + xidx;

    /* Use warp shuffle for left/right neighbor reads to reduce shared-mem
     * bank pressure. Within a warp, __shfl_sync can give us tx-1 and tx+1
     * values directly from registers. For cross-warp boundaries we still
     * need shared memory, but that's only 2 threads per warp boundary.
     *
     * Strategy: load buf[cur][tx] into register 'self', use shfl for
     * intra-warp neighbors; cross-warp neighbors still read shared mem.
     * This reduces 2 shared-mem reads to shfl for ~30/32 threads per warp.
     */
    #pragma unroll 4
    for (int i = 0; i < iteration; i++) {
        const int w = in_range ? __ldg(wallStep + (long)i * cols) : 0;

        /* Load current cell into register. */
        const int self = buf[cur][tx];

        /* For intra-warp neighbors, use warp shuffle (zero latency vs smem). */
        const unsigned int mask = 0xffffffffu;
        const int lane = tx & 31;

        /* Left neighbor: tx-1. Cross-warp (lane==0) or clamped: use smem. */
        int left_val;
        if (lane == 0) {
            left_val = buf[cur][W];  /* W is already clamped */
        } else {
            /* __shfl_up_sync gives us the value from lane-1 within the warp */
            left_val = __shfl_up_sync(mask, self, 1);
            /* But we still need clamping if tx-1 < validXmin */
            if (tx - 1 < validXmin) left_val = buf[cur][validXmin];
        }

        /* Right neighbor: tx+1. Cross-warp (lane==31) or clamped: use smem. */
        int right_val;
        if (lane == 31) {
            right_val = buf[cur][E];  /* E is already clamped */
        } else {
            right_val = __shfl_down_sync(mask, self, 1);
            /* Need clamping if tx+1 > validXmax */
            if (tx + 1 > validXmax) right_val = buf[cur][validXmax];
        }

        const int shortest = min(min(left_val, self), right_val);
        const int val = shortest + w;

        const int nxt = cur ^ 1;
        buf[nxt][tx] = val;
        cur = nxt;
        __syncthreads();
    }

    if (in_range &&
        IN_RANGE(tx, iteration, BLOCK_SIZE - iteration - 1) &&
        IN_RANGE(tx, validXmin, validXmax)) {
        gpuDst[xidx] = buf[cur][tx];
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

    int *wall = (int *)malloc((size_t)rows * cols * sizeof(int));
    if (!wall) { fprintf(stderr, "alloc failed\n"); return 1; }

    srand(M_SEED);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            wall[(size_t)i * cols + j] = rand() % 10;

    int *d_wall, *d_src, *d_dst;
    size_t wall_bytes = (size_t)rows * cols * sizeof(int);
    size_t row_bytes  = (size_t)cols * sizeof(int);

    cudaMalloc(&d_wall, wall_bytes);
    cudaMalloc(&d_src,  row_bytes);
    cudaMalloc(&d_dst,  row_bytes);

    cudaMemcpy(d_wall, wall, wall_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, wall, row_bytes, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(dp_pyramid, cudaFuncCachePreferL1);

    double t0 = now_seconds();

    int pyramid_height = PYRAMID_HEIGHT;

    for (int t = 0; t < rows - 1; t += pyramid_height) {
        int iteration = pyramid_height;
        if (iteration > rows - 1 - t) iteration = rows - 1 - t;

        int border           = iteration;
        int small_block_cols = BLOCK_SIZE - iteration * 2;
        int grid_size        = (cols + small_block_cols - 1) / small_block_cols;

        dp_pyramid<<<grid_size, BLOCK_SIZE>>>(iteration, d_wall, d_src, d_dst,
                                              cols, rows, t, border);

        int *tmp = d_src; d_src = d_dst; d_dst = tmp;
    }
    cudaDeviceSynchronize();

    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    int *result = (int *)malloc(row_bytes);
    if (!result) { fprintf(stderr, "alloc failed\n"); return 1; }
    cudaMemcpy(result, d_src, row_bytes, cudaMemcpyDeviceToHost);

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int n = 0; n < cols; n++)
        fprintf(fp, "%d\t%d\n", n, result[n]);
    fclose(fp);

    free(wall);
    free(result);
    cudaFree(d_wall);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
