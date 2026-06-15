#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define M_SEED 9

/*
 * Pyramidal (temporal-blocking / ghost-zone) PathFinder.
 *
 * Each block loads a BLOCK_SIZE-wide tile of the current DP row into shared
 * memory and advances it PYRAMID_HEIGHT steps locally. The valid band shrinks
 * by one column on each side per step (the "ghost zone"), so adjacent tiles
 * overlap by `border = iteration` columns. This turns the phase-1 design of one
 * global kernel launch + 3 global reads + 1 write PER ROW into ~rows/PH launches
 * with shared-memory traffic, cutting both launch overhead and DRAM traffic.
 *
 * Optimizations over a textbook pyramid kernel:
 *   - Single __syncthreads() per step: a double-buffered shared array (buf[2])
 *     is ping-ponged in place, so we never copy result->prev (which would cost
 *     two extra barriers and an extra smem round-trip per step). Measured kernel
 *     time dropped from ~1.52 ms (4-barrier variant) to ~1.02 ms.
 *   - The wall element for each step is read once into a register via __ldg
 *     (read-only cache) instead of being staged through a third smem buffer.
 *   - Every thread computes unconditionally (no divergence); only the final
 *     write-back is predicated to the globally-valid, in-band columns.
 *
 * Tuning (RTX 3070, sm_86, cols=100000 rows=1000): BLOCK_SIZE=768,
 * PYRAMID_HEIGHT=96 minimizes total kernel time. Larger blocks amortize the
 * 2*PH/BS ghost-zone overhead and reduce launch count; beyond 768 threads
 * occupancy drops and it regresses.
 */

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 768
#endif
#ifndef PYRAMID_HEIGHT
#define PYRAMID_HEIGHT 96
#endif

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define IN_RANGE(x, mn, mx) ((x) >= (mn) && (x) <= (mx))

__global__ void dp_pyramid(int iteration,
                           const int * __restrict__ gpuWall,
                           const int * __restrict__ gpuSrc,
                           int       * __restrict__ gpuDst,
                           int cols, int rows,
                           int startStep,
                           int border)
{
    /* Double-buffered DP row in shared memory; ping-pong via `cur`. */
    __shared__ int buf[2][BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int small_block_cols = BLOCK_SIZE - iteration * 2;
    int blkX    = small_block_cols * bx - border;
    int blkXmax = blkX + BLOCK_SIZE - 1;
    int xidx    = blkX + tx;

    /* Valid output region inside this tile (clamped to the global wall). */
    int validXmin = (blkX < 0)         ? -blkX                              : 0;
    int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1)
                                       : BLOCK_SIZE - 1;

    /* Boundary-clamped W/E neighbours (fixed for the whole pyramid). */
    int W = (tx - 1 < validXmin) ? validXmin : tx - 1;
    int E = (tx + 1 > validXmax) ? validXmax : tx + 1;

    int cur = 0;

    /* Load the initial DP row from global memory. */
    buf[cur][tx] = IN_RANGE(xidx, 0, cols - 1) ? __ldg(&gpuSrc[xidx]) : 0;
    __syncthreads();

    for (int i = 0; i < iteration; i++) {
        /* Wall row for step (startStep + i + 1), one register read via __ldg. */
        int wallBase = cols * (startStep + i + 1);
        int w = IN_RANGE(xidx, 0, cols - 1) ? __ldg(&gpuWall[wallBase + xidx]) : 0;

        /* Compute unconditionally; halo/out-of-band lanes produce values that
         * are simply never read by valid lanes (the band shrinks each step) and
         * are discarded at write-back. */
        int shortest = min(min(buf[cur][W], buf[cur][tx]), buf[cur][E]);
        int val = shortest + w;

        int nxt = cur ^ 1;
        buf[nxt][tx] = val;
        cur = nxt;
        __syncthreads();   /* one barrier per step */
    }

    /* Write the valid, in-band outputs back to global memory. */
    if (IN_RANGE(xidx, 0, cols - 1) &&
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

    /* Allocate and generate wall on host (deterministic, srand(M_SEED)). */
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
    /* Row 0 of the DP equals wall row 0. */
    cudaMemcpy(d_src, wall, row_bytes, cudaMemcpyHostToDevice);

    /* DP — time only this section. */
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

    /* After the loop the latest result lives in d_src (post-swap). */
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
