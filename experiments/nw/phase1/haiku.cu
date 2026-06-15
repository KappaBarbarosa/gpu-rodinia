#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

static int maximum(int a, int b, int c) {
    int k = (a <= b) ? b : a;
    return (k <= c) ? c : k;
}

static int blosum62[24][24] = {
{ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4},
{-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4},
{-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4},
{-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4},
{ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4},
{-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4},
{-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4},
{ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4},
{-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4},
{-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4},
{-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4},
{-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4},
{-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4},
{-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4},
{-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4},
{ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4},
{ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4},
{-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4},
{-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4},
{ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4},
{-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4},
{-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4},
{ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4},
{-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1}};

__global__ void nw_kernel(int *input_itemsets, int *reference,
                          int max_rows, int max_cols, int penalty,
                          int diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = max_rows + max_cols - 2;

    if (idx >= num_cells) return;

    int i = idx + 1;
    int j = diagonal - i;

    if (i < 1 || i >= max_rows || j < 1 || j >= max_cols) return;

    int diag_val = input_itemsets[(i-1)*max_cols + (j-1)] + reference[i*max_cols + j];
    int left_val = input_itemsets[i*max_cols + (j-1)] - penalty;
    int up_val = input_itemsets[(i-1)*max_cols + j] - penalty;

    input_itemsets[i*max_cols + j] = maximum(diag_val, left_val, up_val);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <dimension> <penalty> <output_file>\n", argv[0]);
        return 1;
    }

    int dim = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char *output_file = argv[3];

    int max_rows = dim + 1;
    int max_cols = dim + 1;
    int total_cells = max_rows * max_cols;

    int *input_itemsets = (int *)malloc(total_cells * sizeof(int));
    int *reference = (int *)malloc(total_cells * sizeof(int));

    if (!input_itemsets || !reference) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    memset(input_itemsets, 0, total_cells * sizeof(int));

    srand(7);
    for (int i = 1; i < max_rows; i++) {
        input_itemsets[i*max_cols] = rand() % 10 + 1;
    }
    for (int j = 1; j < max_cols; j++) {
        input_itemsets[j] = rand() % 10 + 1;
    }

    for (int i = 1; i < max_rows; i++) {
        for (int j = 1; j < max_cols; j++) {
            reference[i*max_cols + j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
        }
    }

    for (int i = 1; i < max_rows; i++) {
        input_itemsets[i*max_cols] = -i * penalty;
    }
    for (int j = 1; j < max_cols; j++) {
        input_itemsets[j] = -j * penalty;
    }

    int *d_input_itemsets, *d_reference;
    cudaMalloc(&d_input_itemsets, total_cells * sizeof(int));
    cudaMalloc(&d_reference, total_cells * sizeof(int));

    cudaMemcpy(d_input_itemsets, input_itemsets, total_cells * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reference, reference, total_cells * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 256;

    for (int d = 2; d <= 2*dim; d++) {
        int min_i = (d > max_cols) ? (d - max_cols + 1) : 1;
        int max_i = (d > max_rows) ? (max_rows - 1) : (d - 1);

        if (min_i > max_i) continue;

        int num_cells = max_i - min_i + 1;
        int grid_size = (num_cells + block_size - 1) / block_size;

        nw_kernel<<<grid_size, block_size>>>(d_input_itemsets, d_reference,
                                             max_rows, max_cols, penalty, d);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(input_itemsets, d_input_itemsets, total_cells * sizeof(int), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open output file\n");
        return 1;
    }

    for (int idx = 0; idx < total_cells; idx++) {
        fprintf(fp, "%d\t%d\n", idx, input_itemsets[idx]);
    }

    fclose(fp);

    cudaFree(d_input_itemsets);
    cudaFree(d_reference);
    free(input_itemsets);
    free(reference);

    return 0;
}
