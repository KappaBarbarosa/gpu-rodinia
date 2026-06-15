#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define M_SEED 9

#define BLOCK_SIZE 256
#define PYRAMID_HEIGHT 32

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN3(a, b, c) min(min((a), (b)), (c))

/*
 * Pyramid / temporal-blocking kernel.
 * Each block owns BLOCK_SIZE output columns. It loads BLOCK_SIZE + 2*iteration
 * columns (the halo needed to compute `iteration` rows) into shared memory,
 * advances `iteration` DP rows entirely in shared memory, then writes back the
 * BLOCK_SIZE valid results. This cuts global traffic and kernel launches by a
 * factor of `iteration` versus a one-row-per-launch scheme.
 */
__global__ void dp_pyramid(int iteration,
                           const int *__restrict__ gpuWall,
                           const int *__restrict__ gpuSrc,
                           int *__restrict__ gpuDst,
                           int cols, int rows,
                           int startStep,   /* current row already computed in gpuSrc */
                           int border)
{
    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    /* Global column range this block is responsible for (with halo). */
    int small_block_cols = BLOCK_SIZE - iteration * 2;
    int blkX = small_block_cols * bx - border;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    int xidx = blkX + tx;

    /* Valid output region inside this block's shared tile. */
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1)
                                         : BLOCK_SIZE - 1;

    int W = tx - 1;
    int E = tx + 1;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if (IN_RANGE(xidx, 0, cols - 1)) {
        prev[tx] = gpuSrc[xidx];
    }
    __syncthreads();

    bool computed;
    for (int i = 0; i < iteration; i++) {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
            computed = true;
            int left  = prev[W];
            int up    = prev[tx];
            int right = prev[E];
            int shortest = MIN3(left, up, right);
            int index = cols * (startStep + i + 1) + xidx;
            result[tx] = shortest + gpuWall[index];
        }
        __syncthreads();
        if (i == iteration - 1) break;
        if (computed) prev[tx] = result[tx];
        __syncthreads();
    }

    if (computed) {
        gpuDst[xidx] = result[tx];
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
    /* Row 0 of the DP equals wall row 0. */
    cudaMemcpy(d_src, wall, row_bytes, cudaMemcpyHostToDevice);

    double t0 = now_seconds();

    int pyramid_height = PYRAMID_HEIGHT;

    for (int t = 0; t < rows - 1; t += pyramid_height) {
        int iteration = pyramid_height;
        if (iteration > rows - 1 - t) iteration = rows - 1 - t;

        int border = iteration;                       /* halo each side */
        int small_block_cols = BLOCK_SIZE - iteration * 2;
        int grid_size = (cols + small_block_cols - 1) / small_block_cols;

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
