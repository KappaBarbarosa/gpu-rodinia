/*
 * PathFinder — CUDA phase 2 (temporal tiling / pyramidal blocking).
 *
 * CLI:  pathfinder <cols> <rows> <output_file>
 * Wall generated deterministically on host via srand(9), i (rows) outer,
 * j (cols) inner, wall[i][j] = rand() % 10.
 *
 * Each kernel launch advances PYRAMID_HEIGHT DP rows using double-buffered
 * shared memory. Each block covers BLOCK_SIZE columns with a halo of
 * PYRAMID_HEIGHT on each side; only the central owned columns are written
 * back to global memory. Reduces ~rows launches to ceil(rows-1 / PH).
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define M_SEED 9

#define BLOCK_SIZE 512
#define PYRAMID_HEIGHT 72

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define IN_RANGE(x, mn, mx) ((x) >= (mn) && (x) <= (mx))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

/*
 * dynproc: advance up to `iteration` DP rows starting from global row
 * `startStep` (0-based step index; src holds the accumulated cost after
 * step startStep, i.e. row startStep). Each block owns a contiguous chunk of
 * BLOCK_SIZE columns minus halo, but loads BLOCK_SIZE columns into shared mem.
 */
__global__ void dynproc(int iteration, const int *__restrict__ gpuWall,
                        const int *__restrict__ src, int *__restrict__ dst,
                        int cols, int startStep, int border)
{
    __shared__ int buf[2][BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    /* Global column index of this thread's slot in the loaded tile.
       The tile's left edge in global coords is blkX. The block "owns"
       columns [blkXmin .. blkXmax], with a halo of `border` on each side. */
    int small_block_cols = BLOCK_SIZE - 2 * border;
    int blkX = small_block_cols * bx - border;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    int xidx = blkX + tx;

    /* valid range of indices that this block is allowed to *compute* at the
       current shrinking pyramid level */
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - (cols - 1))
                                         : BLOCK_SIZE - 1;

    int W = tx - 1;
    int E = tx + 1;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    int cur = 0, nxt = 1;

    /* Load initial row (accumulated cost after startStep) into buf[cur]. */
    if (IN_RANGE(xidx, 0, cols - 1)) {
        buf[cur][tx] = __ldg(&src[xidx]);
    }
    __syncthreads();

    #pragma unroll 8
    for (int i = 0; i < iteration; i++) {
        bool computeIt = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid)
            computeIt = true;

        if (computeIt) {
            int left  = buf[cur][W];
            int center = buf[cur][tx];
            int right = buf[cur][E];
            int m = MIN(left, MIN(center, right));
            /* wall row for the DP step: startStep + i + 1 */
            int wval = __ldg(&gpuWall[(size_t)(startStep + i + 1) * cols + xidx]);
            buf[nxt][tx] = wval + m;
        }
        __syncthreads();
        /* swap */
        int t = cur; cur = nxt; nxt = t;
    }

    /* Write back owned central columns. After `iteration` shrink levels, the
       valid computed region for this block is [iteration .. BLOCK_SIZE-1-iteration]
       in tile coords, intersected with the global column range. We write the
       columns this block owns (those that started as central, i.e.
       tx in [border .. BLOCK_SIZE-1-border]) to dst. */
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

    int *wall = (int *)malloc((size_t)rows * cols * sizeof(int));
    int *result = (int *)malloc((size_t)cols * sizeof(int));
    if (!wall || !result) { fprintf(stderr, "alloc failed\n"); return 1; }

    srand(M_SEED);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            wall[(size_t)i * cols + j] = rand() % 10;

    int *d_wall, *d_a, *d_b;
    cudaMalloc(&d_wall, (size_t)rows * cols * sizeof(int));
    cudaMalloc(&d_a, (size_t)cols * sizeof(int));
    cudaMalloc(&d_b, (size_t)cols * sizeof(int));
    cudaMemcpy(d_wall, wall, (size_t)rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    /* initial accumulated-cost row = wall row 0 -> d_a holds "after step 0". */
    cudaMemcpy(d_a, d_wall, (size_t)cols * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaFuncSetCacheConfig(dynproc, cudaFuncCachePreferL1);

    int border = PYRAMID_HEIGHT;
    int small_block_cols = BLOCK_SIZE - 2 * border;
    int gridDimX = (cols + small_block_cols - 1) / small_block_cols;

    int *src = d_a;  /* accumulated cost after current startStep */
    int *dst = d_b;

    int total_steps = rows - 1;  /* number of DP transitions */

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

    /* result is in `src` after the final swap */
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
