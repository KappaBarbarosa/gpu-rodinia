#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOSUM62_SIZE 24

const int BLOSUM62[BLOSUM62_SIZE][BLOSUM62_SIZE] = {
    { 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
    {-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
    {-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
    {-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
    { 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
    {-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
    {-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
    { 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
    {-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
    {-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
    {-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -1, -4},
    { 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0, -1, -4},
    { 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -3, -2,  0, -1, -1, -1, -4},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -3, 11,  2, -3, -4, -3, -2, -4},
    {-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
    { 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
    {-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
    {-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
    { 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -4},
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

__device__ int maximum(int a, int b, int c) {
    int max_val = a;
    if (b > max_val) max_val = b;
    if (c > max_val) max_val = c;
    return max_val;
}

__global__ void nw_kernel(int *M, int *ref, int dim, int penalty, int max_rows, int max_cols, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int min_i = max(1, d - (max_cols - 1));
    int max_i = min(max_rows - 1, d - 1);

    if (idx >= min_i || idx > max_i) return;

    int i = idx;
    int j = d - i;

    if (j < 1 || j >= max_cols) return;

    int match = M[(i-1)*max_cols + (j-1)] + ref[i*max_cols + j];
    int delete_val = M[(i-1)*max_cols + j] - penalty;
    int insert_val = M[i*max_cols + (j-1)] - penalty;

    M[i*max_cols + j] = maximum(match, delete_val, insert_val);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <dim> <penalty> <output_file>\n", argv[0]);
        return 1;
    }

    int dim = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char *out = argv[3];

    int max_rows = dim + 2;
    int max_cols = dim + 2;
    int size = max_rows * max_cols;

    int *M = (int *)malloc(size * sizeof(int));
    int *ref = (int *)malloc(size * sizeof(int));
    unsigned char *input = (unsigned char *)malloc(dim * sizeof(unsigned char));

    srand(7);

    for (int i = 0; i < dim; i++) {
        input[i] = rand() % 24;
    }

    for (int i = 0; i < size; i++) {
        M[i] = 0;
    }

    for (int i = 1; i <= dim; i++) {
        M[i * max_cols] = rand() % 10 + 1;
        M[i * max_cols] = -i * penalty;
    }

    for (int j = 1; j <= dim; j++) {
        M[j] = rand() % 10 + 1;
        M[j] = -j * penalty;
    }

    for (int i = 1; i <= dim; i++) {
        for (int j = 1; j <= dim; j++) {
            ref[i * max_cols + j] = BLOSUM62[input[i-1]][input[j-1]];
        }
    }

    int *d_M, *d_ref;
    cudaMalloc(&d_M, size * sizeof(int));
    cudaMalloc(&d_ref, size * sizeof(int));

    cudaMemcpy(d_M, M, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, ref, size * sizeof(int), cudaMemcpyHostToDevice);

    for (int d = 2; d <= 2 * dim; d++) {
        int min_i = max(1, d - (max_cols - 1));
        int max_i = min(max_rows - 1, d - 1);
        int num_cells = max_i - min_i + 1;

        if (num_cells > 0) {
            int block_size = 256;
            int grid_size = (num_cells + block_size - 1) / block_size;
            nw_kernel<<<grid_size, block_size>>>(d_M, d_ref, dim, penalty, max_rows, max_cols, d);
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(M, d_M, size * sizeof(int), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(out, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open output file\n");
        return 1;
    }

    for (int idx = 0; idx < size; idx++) {
        fprintf(fp, "%d\t%d\n", idx, M[idx]);
    }

    fclose(fp);

    cudaFree(d_M);
    cudaFree(d_ref);
    free(M);
    free(ref);
    free(input);

    return 0;
}
