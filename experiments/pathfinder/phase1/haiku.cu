/*
 * PathFinder — CUDA (temporal blocking / ghost-zone pyramid tiling).
 * Each block processes a tile of columns for PYRAMID_HEIGHT rows in shared
 * memory before touching global memory again. This amortizes kernel launches
 * and slashes global-memory traffic (read/write of the whole cost vector
 * every single row in the naive version).
 */
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define M_SEED 9
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef PYRAMID_HEIGHT
#define PYRAMID_HEIGHT 8
#endif

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Temporal-blocking kernel.
 *
 * For `iteration` rows starting at global cost row `startStep` (already in
 * `src`), each block owns a contiguous chunk of BLOCK_SIZE output columns.
 * It loads BLOCK_SIZE + 2*iteration cost values (with a halo / ghost zone of
 * `iteration` on each side) into shared memory, then performs `iteration`
 * DP steps purely in shared memory. The ghost zone shrinks by one each step
 * (the classic pyramid), so only the central BLOCK_SIZE columns are written
 * back as fully-correct results.
 *
 *   src  : input cost vector (cost after row startStep)
 *   dst  : output cost vector (cost after row startStep + iteration)
 *   wall : full weight matrix (row-major, rows x cols)
 *   cols : number of columns
 *   rows : number of rows
 *   startStep  : index of the cost row currently in src (0 == wall row 0)
 *   iteration  : number of DP steps to perform this launch (<= PYRAMID_HEIGHT)
 */
__global__ void pathfinder_temporal(const int *src, int *dst, const int *wall,
                                    int cols, int rows, int startStep,
                                    int iteration)
{
    __shared__ int prev[BLOCK_SIZE + 2 * PYRAMID_HEIGHT];
    __shared__ int cur[BLOCK_SIZE + 2 * PYRAMID_HEIGHT];

    int tx = threadIdx.x;
    /* First output column owned by this block. */
    int blkStart = blockIdx.x * BLOCK_SIZE;
    /* Shared-memory region covers [blkStart - iteration, blkStart + BLOCK_SIZE + iteration). */
    int smStart = blkStart - iteration;
    int smWidth = BLOCK_SIZE + 2 * iteration;

    /* Load the ghost-zoned input into shared memory. Each thread may load
       more than one element since smWidth >= blockDim.x. */
    for (int i = tx; i < smWidth; i += BLOCK_SIZE) {
        int g = smStart + i;
        prev[i] = (g >= 0 && g < cols) ? src[g] : INT_MAX;
    }
    __syncthreads();

    int *sPrev = prev;
    int *sCur  = cur;

    for (int it = 0; it < iteration; it++) {
        int row = startStep + it + 1;          /* wall row being added */
        /* Valid (computable) shared range shrinks by one on each side. */
        int lo = it + 1;
        int hi = smWidth - (it + 1);
        for (int i = tx; i < smWidth; i += BLOCK_SIZE) {
            int g = smStart + i;               /* global column for this slot */
            if (i >= lo && i < hi && g >= 0 && g < cols) {
                int m = sPrev[i];
                /* Left/right neighbours: clamp at the global boundaries. */
                if (g > 0)        m = MIN(m, sPrev[i - 1]);
                if (g < cols - 1) m = MIN(m, sPrev[i + 1]);
                sCur[i] = wall[(size_t)row * cols + g] + m;
            } else {
                sCur[i] = sPrev[i];            /* carry halo forward unchanged */
            }
        }
        __syncthreads();
        int *t = sPrev; sPrev = sCur; sCur = t;
    }

    /* Write back the central BLOCK_SIZE columns (fully resolved). */
    int outCol = blkStart + tx;
    if (outCol < cols) {
        int idx = tx + iteration;              /* offset of this column in shared */
        dst[outCol] = sPrev[idx];
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

    /* Allocate host memory for wall and temporary vectors. */
    int *wall = (int *)malloc((size_t)rows * cols * sizeof(int));
    int *src_host  = (int *)malloc((size_t)cols * sizeof(int));
    int *dst_host  = (int *)malloc((size_t)cols * sizeof(int));
    if (!wall || !src_host || !dst_host) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    /* Deterministic wall generation on host — order is normative:
       i over rows (outer), j over cols (inner). */
    srand(M_SEED);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            wall[(size_t)i * cols + j] = rand() % 10;
        }
    }

    /* Allocate device memory. */
    int *d_wall, *d_src, *d_dst;
    cudaMalloc(&d_wall, (size_t)rows * cols * sizeof(int));
    cudaMalloc(&d_src, (size_t)cols * sizeof(int));
    cudaMalloc(&d_dst, (size_t)cols * sizeof(int));
    if (!d_wall || !d_src || !d_dst) {
        fprintf(stderr, "cuda alloc failed\n");
        return 1;
    }

    /* Copy wall to device. */
    cudaMemcpy(d_wall, wall, (size_t)rows * cols * sizeof(int),
               cudaMemcpyHostToDevice);

    /* Initialize src (accumulated cost vector) with wall row 0. */
    for (int j = 0; j < cols; j++) {
        src_host[j] = wall[j];
    }
    cudaMemcpy(d_src, src_host, (size_t)cols * sizeof(int),
               cudaMemcpyHostToDevice);

    int grid_size = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* DP — time the loop. Temporal blocking: each launch advances the cost
       vector by `iter` rows (<= PYRAMID_HEIGHT). */
    double t0 = now_seconds();
    int startStep = 0;
    while (startStep < rows - 1) {
        int iter = MIN(PYRAMID_HEIGHT, (rows - 1) - startStep);
        pathfinder_temporal<<<grid_size, BLOCK_SIZE>>>(d_src, d_dst, d_wall,
                                                       cols, rows, startStep,
                                                       iter);
        int *tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
        startStep += iter;
    }
    cudaDeviceSynchronize();
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    /* Copy result back to host. */
    cudaMemcpy(dst_host, d_src, (size_t)cols * sizeof(int),
               cudaMemcpyDeviceToHost);

    /* Write output file. */
    FILE *fp = fopen(ofile, "w");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", ofile);
        return 1;
    }
    for (int n = 0; n < cols; n++) {
        fprintf(fp, "%d\t%d\n", n, dst_host[n]);
    }
    fclose(fp);

    /* Cleanup. */
    cudaFree(d_wall);
    cudaFree(d_src);
    cudaFree(d_dst);
    free(wall);
    free(src_host);
    free(dst_host);

    return 0;
}
