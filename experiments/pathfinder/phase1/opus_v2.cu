/*
 * PathFinder — CUDA phase 1 (global memory only, one thread per column).
 *
 * CLI:  pathfinder <cols> <rows> <output_file>
 * Wall generated deterministically on host via srand(9), i (rows) outer,
 * j (cols) inner, wall[i][j] = rand() % 10.
 * One kernel launch per DP step; one thread per column; ping-pong buffers.
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define M_SEED 9

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void dp_step(const int *src, int *dst, const int *wnext, int cols)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= cols) return;
    int m = src[n];
    if (n > 0)        { int l = src[n - 1]; if (l < m) m = l; }
    if (n < cols - 1) { int r = src[n + 1]; if (r < m) m = r; }
    dst[n] = wnext[n] + m;
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

    /* initial accumulated-cost row = wall row 0 -> place in d_a (acts as dst). */
    cudaMemcpy(d_a, d_wall, (size_t)cols * sizeof(int), cudaMemcpyDeviceToDevice);

    int threads = 256;
    int blocks = (cols + threads - 1) / threads;

    int *dst = d_a;  /* current result row */
    int *src = d_b;

    cudaDeviceSynchronize();
    double t0 = now_seconds();
    for (int t = 0; t < rows - 1; t++) {
        int *tmp = src; src = dst; dst = tmp;   /* src = prev row */
        const int *wnext = d_wall + (size_t)(t + 1) * cols;
        dp_step<<<blocks, threads>>>(src, dst, wnext, cols);
    }
    cudaDeviceSynchronize();
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    cudaMemcpy(result, dst, (size_t)cols * sizeof(int), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int n = 0; n < cols; n++)
        fprintf(fp, "%d\t%d\n", n, result[n]);
    fclose(fp);

    cudaFree(d_wall); cudaFree(d_a); cudaFree(d_b);
    free(wall); free(result);
    return 0;
}
