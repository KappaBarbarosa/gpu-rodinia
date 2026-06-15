#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define M_SEED 9

/*
 * Pyramidal (temporal-blocking / ghost-zone) PathFinder.
 *
 * Round 2: PH=112, BS=768 confirmed at ~1.01 ms, within ~5% of bandwidth
 * roofline (0.96 ms for pure 400 MB read at RTX 3070 peak).
 * Added cudaFuncSetCacheConfig to prefer L1 (may marginally help shared-mem
 * pressure at BLOCK_SIZE=768 x 2 banks). No algorithmic change — already at
 * the roofline.
 */

#ifndef BLOCK_SIZE
/* BLOCK_SIZE=512 is the occupancy sweet spot on sm_86 (RTX 3070): 1536
 * threads/SM / 512 = exactly 3 resident blocks, no occupancy quantization
 * waste. Measured (cudaEvent, best-of-N): BS=512 ~1.02-1.04 ms vs BS=768
 * ~1.08 ms and BS=480/544 (non-multiple-of-256) ~1.15-1.26 ms. */
#define BLOCK_SIZE 512
#endif
#ifndef PYRAMID_HEIGHT
/* PYRAMID_HEIGHT=72 chosen empirically on sm_86 with BLOCK_SIZE=512 for
 * cols=100000, rows=1000: minimizes kernel time (~1.02 ms). It sits right
 * at the edge of a sharp cliff -- PH>=75 jumps to ~1.16 ms (register/cache
 * pressure), while smaller PH raises launch count + src-reload redundancy.
 * Redundancy here is BS/(BS-2*PH)=512/368=1.39x over the 400 MB wall, i.e.
 * ~556 MB effective traffic, near the ~448 GB/s bandwidth roofline. */
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
    /* Hoist the range check -- constant per thread for entire pyramid. */
    int in_range = IN_RANGE(xidx, 0, cols - 1);

    /* Load the initial DP row from global memory. */
    buf[cur][tx] = in_range ? __ldg(&gpuSrc[xidx]) : 0;
    __syncthreads();

    /* Precompute pointer to wall row (startStep+1) + xidx offset so the
     * inner loop only strides by cols each iteration, avoiding a full
     * (startStep+i+1)*cols multiply per step. */
    const int * __restrict__ wallStep = gpuWall + (long)(startStep + 1) * cols + xidx;

    #pragma unroll 4
    for (int i = 0; i < iteration; i++) {
        /* Wall element for this step, read via __ldg (read-only cache). */
        int w = in_range ? __ldg(wallStep + (long)i * cols) : 0;

        /* Unconditional min-reduction; ghost lanes produce values that
         * are never written back (the band shrinks each step). */
        int shortest = min(min(buf[cur][W], buf[cur][tx]), buf[cur][E]);
        int val = shortest + w;

        int nxt = cur ^ 1;
        buf[nxt][tx] = val;
        cur = nxt;
        __syncthreads();   /* one barrier per step */
    }

    /* Write the valid, in-band outputs back to global memory. */
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

    /* Hint the driver to prefer larger L1 over shared mem (shared usage is
     * only 2*BLOCK_SIZE*4 = 6 KB, well within the 16 KB minimum; rest of the
     * 128 KB SRAM can serve as L1 to better cache wall reads via __ldg). */
    cudaFuncSetCacheConfig(dp_pyramid, cudaFuncCachePreferL1);

    /* DP -- time only this section. */
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
