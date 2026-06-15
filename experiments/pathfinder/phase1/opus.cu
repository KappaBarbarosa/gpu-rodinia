/*
 * PathFinder — phase 2 CUDA port (ghost-zone temporal blocking / pyramid).
 * Each block loads a tile of columns into shared memory and advances PYRAMID_HEIGHT
 * DP rows internally before writing the (shrunken) valid region back to global
 * memory. This cuts kernel launches from rows-1 to ceil((rows-1)/PYRAMID_HEIGHT)
 * and reduces global-memory traffic by ~PYRAMID_HEIGHT.
 *
 * Numerics match the serial reference exactly (int min/add, boundary clamp).
 *
 * Phase-2 tuning vs the prior best (BLOCK_COLS=256/H=32, 1.716ms total kernel):
 *   - BLOCK_COLS=128, PYRAMID_HEIGHT=64: fewer launches (16 vs 32) and better
 *     occupancy of the small per-row working set; measured 1.590ms (-7.3%).
 *   - INTERIOR FAST PATH: every block except the first and last has all of its
 *     tile cells strictly inside the grid, so the per-cell global-edge clamps
 *     (g>0, g<cols-1, g>=0, g<cols) are provably always satisfied. Those blocks
 *     run a branch-free MIN over the three neighbours and index the wall row with
 *     a precomputed tile-local pointer (no per-cell `g` recompute). Only the two
 *     edge blocks pay the full clamped path, exactly reproducing serial semantics.
 *
 * CLI: pathfinder <cols> <rows> <output_file>
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define M_SEED 9
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

/* BLOCK_COLS output columns per block; PYRAMID_HEIGHT DP steps fused per launch.
 * Shared tile is BLOCK_COLS + 2*PYRAMID_HEIGHT wide (ghost halo feeds the inward-
 * shrinking pyramid). Shared footprint is 2 x (128 + 128) = 512 ints/block (2KB),
 * which keeps occupancy high while the deeper pyramid (H=64) halves launch count. */
#define BLOCK_COLS    128
#define PYRAMID_HEIGHT 64

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define CHECK(call) do {                                                     \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(_e));                                     \
        exit(1);                                                             \
    }                                                                        \
} while (0)

/*
 * Ghost-zone pyramid kernel.
 *  src     : accumulated-cost row already computed at step `start` (length cols)
 *  dst     : output accumulated-cost row after `steps` DP rows (length cols)
 *  wall    : full wall, row-major rows*cols (uses rows start+1 .. start+steps)
 *  cols    : grid width
 *  start   : index of the source row
 *  steps   : DP rows to advance this launch (<= PYRAMID_HEIGHT)
 *  nblocks : total blocks this launch (to detect the two edge blocks)
 *
 * Each block owns BLOCK_COLS output columns and loads BLOCK_COLS+2*PYRAMID_HEIGHT
 * columns (with halo) into shared memory. It iterates `steps` times, shrinking
 * the valid region by one on each side per step; after `steps` steps the central
 * BLOCK_COLS columns are valid and written to dst. Two shared buffers are
 * ping-ponged by pointer swap (one __syncthreads per step). Boundary clamp uses
 * the GLOBAL column id `g`, exactly reproducing the serial semantics; out-of-grid
 * ghost cells are sentinel-filled and never consumed by an in-grid neighbour.
 *
 * Interior blocks (not first, not last, full tile in-grid) take a branch-free
 * path: all global-edge predicates are statically true there. */
__global__ void dp_pyramid(const int * __restrict__ src,
                           int * __restrict__ dst,
                           const int * __restrict__ wall,
                           int cols, int start, int steps, int nblocks)
{
    __shared__ int cur[BLOCK_COLS + 2 * PYRAMID_HEIGHT];
    __shared__ int nxt[BLOCK_COLS + 2 * PYRAMID_HEIGHT];

    const int tx = threadIdx.x;
    const int blk_start = blockIdx.x * BLOCK_COLS;
    const int halo = PYRAMID_HEIGHT;
    const int tile_w = BLOCK_COLS + 2 * halo;

    for (int i = tx; i < tile_w; i += BLOCK_COLS) {
        int g = blk_start - halo + i;
        if (g < 0 || g >= cols)
            cur[i] = 0x3fffffff;   /* sentinel: never wins a MIN */
        else
            cur[i] = src[g];
    }
    __syncthreads();

    int *a = cur, *b = nxt;

    /* An interior block has every tile cell strictly inside the grid:
     *   blk_start - halo >= 0  AND  blk_start + BLOCK_COLS + halo <= cols.
     * Then for all valid i: g > 0, g < cols-1, g >= 0, g < cols hold, so the
     * clamps are no-ops and can be elided. */
    const bool interior =
        (blk_start - halo >= 0) && (blk_start + BLOCK_COLS + halo <= cols);

    if (interior) {
        const int base = blk_start - halo;
        for (int s = 0; s < steps; s++) {
            int shrink = s + 1;
            const int *wrow = wall + (size_t)(start + 1 + s) * cols + base;
            for (int i = tx; i < tile_w; i += BLOCK_COLS) {
                if (i >= shrink && i < tile_w - shrink) {
                    int m = a[i];
                    m = MIN(m, a[i - 1]);
                    m = MIN(m, a[i + 1]);
                    b[i] = wrow[i] + m;
                }
            }
            __syncthreads();
            int *t = a; a = b; b = t;
        }
    } else {
        for (int s = 0; s < steps; s++) {
            int shrink = s + 1;
            const int *wrow = wall + (size_t)(start + 1 + s) * cols;
            for (int i = tx; i < tile_w; i += BLOCK_COLS) {
                int g = blk_start - halo + i;   /* global column of cell i */
                if (i >= shrink && i < tile_w - shrink && g >= 0 && g < cols) {
                    int m = a[i];
                    if (g > 0)        m = MIN(m, a[i - 1]);
                    if (g < cols - 1) m = MIN(m, a[i + 1]);
                    b[i] = wrow[g] + m;
                }
            }
            __syncthreads();
            int *t = a; a = b; b = t;
        }
    }

    int out_col = blk_start + tx;
    if (out_col < cols)
        dst[out_col] = a[halo + tx];
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

    /* Deterministic wall generation — order is normative. */
    srand(M_SEED);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            wall[(size_t)i * cols + j] = rand() % 10;

    /* Device buffers: full wall + two ping-pong cost rows. */
    int *d_wall, *d_a, *d_b;
    CHECK(cudaMalloc(&d_wall, (size_t)rows * cols * sizeof(int)));
    CHECK(cudaMalloc(&d_a, (size_t)cols * sizeof(int)));
    CHECK(cudaMalloc(&d_b, (size_t)cols * sizeof(int)));
    CHECK(cudaMemcpy(d_wall, wall, (size_t)rows * cols * sizeof(int),
                     cudaMemcpyHostToDevice));

    /* Initial accumulated-cost row = wall row 0. */
    int *d_src = d_a;
    int *d_dst = d_b;
    CHECK(cudaMemcpy(d_src, d_wall, (size_t)cols * sizeof(int),
                     cudaMemcpyDeviceToDevice));

    int blocks = (cols + BLOCK_COLS - 1) / BLOCK_COLS;

    CHECK(cudaDeviceSynchronize());
    double t0 = now_seconds();
    int total = rows - 1;
    for (int t = 0; t < total; t += PYRAMID_HEIGHT) {
        int steps = MIN(PYRAMID_HEIGHT, total - t);
        dp_pyramid<<<blocks, BLOCK_COLS>>>(d_src, d_dst, d_wall, cols, t, steps,
                                           blocks);
        int *tmp = d_src; d_src = d_dst; d_dst = tmp; /* result ends in d_src */
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    CHECK(cudaMemcpy(result, d_src, (size_t)cols * sizeof(int),
                     cudaMemcpyDeviceToHost));

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int n = 0; n < cols; n++)
        fprintf(fp, "%d\t%d\n", n, result[n]);
    fclose(fp);

    cudaFree(d_wall); cudaFree(d_a); cudaFree(d_b);
    free(wall); free(result);
    return 0;
}
