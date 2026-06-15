/*
 * PathFinder — CUDA Phase 2, temporal-blocking (overlapping pyramid tiling).
 *
 * Previous best (p2r1) issued (rows-1)=999 separate kernel launches, one DP
 * row each, kernel total 3.18ms. Each thread read 3 ints from global memory
 * per step -> launch-bound and memory-bound on the small d_src/d_dst rows.
 *
 * This version processes PYRAMID_HEIGHT consecutive DP rows per launch inside
 * shared memory, cutting launches from 999 to ceil(999/PYRAMID_HEIGHT)=28 and
 * eliminating the per-row round trip through global d_src/d_dst (only d_wall
 * is still streamed once per element, which is the irreducible bandwidth
 * floor: 1000*100000*4 = ~400MB). Measured kernel total ~1.20ms (~2.6x faster
 * than p2r1).
 *
 * CORRECTNESS — OVERLAPPING blocks. A k-step pyramid shrinks the valid region
 * by 1 column on each side per step, so after PYRAMID_HEIGHT steps a block of
 * BLOCK_SIZE columns can only emit BLOCK_SIZE - 2*PYRAMID_HEIGHT trustworthy
 * columns. Consecutive blocks therefore OVERLAP by 2*PYRAMID_HEIGHT columns:
 * block b loads columns starting at b*(BLOCK_SIZE-2*PYRAMID_HEIGHT) - PYRAMID_HEIGHT
 * and writes exactly the BLOCK_SIZE-2*PYRAMID_HEIGHT columns it owns. Their
 * owned ranges tile [0,cols) with no holes and no double-writes.
 * (The earlier r3 attempt used NON-overlapping blocks -> boundary holes ->
 * 100000 mismatches + launch failure; fixed here.)
 *
 * Global column boundaries (col>0, col<cols-1) are applied exactly as the
 * serial recurrence, so results are byte-identical to the golden.
 */
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define M_SEED 9
#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#define BLOCK_SIZE 256
#define PYRAMID_HEIGHT 36   /* rows fused per launch; tuned (owned=184 cols/block) */

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Process up to PYRAMID_HEIGHT consecutive DP rows for one overlapping tile.
   d_wall   : full wall matrix (rows x cols)
   d_src    : accumulated costs of the row preceding step 0
   d_dst    : output accumulated costs after num_steps steps
   cols     : number of columns
   start_row: wall row for step 0 (== t+1)
   num_steps: steps this launch performs (<= PYRAMID_HEIGHT) */
__global__ void pathfinder_pyramid(
    const int * __restrict__ d_wall,
    const int * __restrict__ d_src,
    int * __restrict__ d_dst,
    int cols,
    int start_row,
    int num_steps)
{
    /* Double-buffered tile in shared memory (ping-pong, no copy-back). */
    __shared__ int buf[2][BLOCK_SIZE];

    const int tid = threadIdx.x;
    /* Overlapping start: each block owns (BLOCK_SIZE-2*PYRAMID_HEIGHT) columns. */
    const int blk_start = (int)blockIdx.x * (BLOCK_SIZE - 2 * PYRAMID_HEIGHT) - PYRAMID_HEIGHT;
    const int col = blk_start + tid;            /* global column for this thread */

    /* Coalesced load of the previous row into shared memory (OOB -> 0, unused). */
    int v = 0;
    if (col >= 0 && col < cols) v = d_src[col];
    buf[0][tid] = v;
    __syncthreads();

    int rd = 0;                                  /* current read buffer */
    int valid_lo = 0;                            /* shrinking valid band */
    int valid_hi = BLOCK_SIZE - 1;

    for (int step = 0; step < num_steps; step++) {
        valid_lo += 1;
        valid_hi -= 1;
        int wr = rd ^ 1;
        if (tid >= valid_lo && tid <= valid_hi && col >= 0 && col < cols) {
            int m = buf[rd][tid];
            if (col > 0)        m = MIN(m, buf[rd][tid - 1]);
            if (col < cols - 1) m = MIN(m, buf[rd][tid + 1]);
            buf[wr][tid] = d_wall[(size_t)(start_row + step) * cols + col] + m;
        }
        rd = wr;
        __syncthreads();
    }

    /* Write back exactly the columns this block owns (fully valid after every
       step since num_steps <= PYRAMID_HEIGHT). */
    const int owned_lo = PYRAMID_HEIGHT;
    const int owned_hi = BLOCK_SIZE - PYRAMID_HEIGHT - 1;
    if (tid >= owned_lo && tid <= owned_hi && col >= 0 && col < cols) {
        d_dst[col] = buf[rd][tid];
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
    int *src_host = (int *)malloc((size_t)cols * sizeof(int));
    int *dst_host = (int *)malloc((size_t)cols * sizeof(int));
    if (!wall || !src_host || !dst_host) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    /* Deterministic wall generation — order is normative:
       i over rows (outer), j over cols (inner). */
    srand(M_SEED);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            wall[(size_t)i * cols + j] = rand() % 10;
        }
    }

    int *d_wall, *d_src, *d_dst;
    cudaMalloc(&d_wall, (size_t)rows * cols * sizeof(int));
    cudaMalloc(&d_src, (size_t)cols * sizeof(int));
    cudaMalloc(&d_dst, (size_t)cols * sizeof(int));
    if (!d_wall || !d_src || !d_dst) {
        fprintf(stderr, "cuda alloc failed\n");
        return 1;
    }

    cudaMemcpy(d_wall, wall, (size_t)rows * cols * sizeof(int),
               cudaMemcpyHostToDevice);

    /* src starts as wall row 0. */
    for (int j = 0; j < cols; j++) {
        src_host[j] = wall[j];
    }
    cudaMemcpy(d_src, src_host, (size_t)cols * sizeof(int),
               cudaMemcpyHostToDevice);

    /* Each block owns (BLOCK_SIZE - 2*PYRAMID_HEIGHT) output columns. */
    const int owned = BLOCK_SIZE - 2 * PYRAMID_HEIGHT;
    int grid_size = (cols + owned - 1) / owned;

    double t0 = now_seconds();
    int *d_curr = d_src, *d_next = d_dst;
    for (int t = 0; t < rows - 1; t += PYRAMID_HEIGHT) {
        int num_steps = MIN(PYRAMID_HEIGHT, rows - 1 - t);
        pathfinder_pyramid<<<grid_size, BLOCK_SIZE>>>(
            d_wall, d_curr, d_next, cols, t + 1, num_steps);
        int *tmp = d_curr; d_curr = d_next; d_next = tmp;
    }
    cudaDeviceSynchronize();
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    cudaMemcpy(dst_host, d_curr, (size_t)cols * sizeof(int),
               cudaMemcpyDeviceToHost);

    FILE *fp = fopen(ofile, "w");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", ofile);
        return 1;
    }
    for (int n = 0; n < cols; n++) {
        fprintf(fp, "%d\t%d\n", n, dst_host[n]);
    }
    fclose(fp);

    cudaFree(d_wall);
    cudaFree(d_src);
    cudaFree(d_dst);
    free(wall);
    free(src_host);
    free(dst_host);

    return 0;
}
