#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

__global__ void nw_wavefront_kernel(int *itemsets, int *reference, int max_cols, int penalty, int d)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // i ranges from max(1, d-(max_cols-1)) to min(max_rows-1, d-1)
    // max_rows == max_cols
    int i_min = d - (max_cols - 1);
    if (i_min < 1) i_min = 1;
    int i_max = d - 1;
    if (i_max > max_cols - 1) i_max = max_cols - 1;

    int i = i_min + tid;
    if (i > i_max) return;

    int j = d - i;
    if (j < 1 || j >= max_cols) return;

    int diag = itemsets[(i-1)*max_cols + (j-1)] + reference[i*max_cols + j];
    int left = itemsets[i*max_cols + (j-1)] - penalty;
    int up   = itemsets[(i-1)*max_cols + j] - penalty;

    int val = diag;
    if (left > val) val = left;
    if (up   > val) val = up;

    itemsets[i*max_cols + j] = val;
}

int main(int argc, char *argv[])
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <dimension> <penalty> <output_file>\n", argv[0]);
        return 1;
    }

    int dim     = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char *outfile = argv[3];

    int max_rows = dim + 1;
    int max_cols = dim + 1;
    int total    = max_rows * max_cols;

    int *referrence    = (int*)malloc(total * sizeof(int));
    int *input_itemsets = (int*)malloc(total * sizeof(int));

    // Step 1: zero itemsets
    srand(7);
    for (int i = 0; i < total; i++) input_itemsets[i] = 0;

    // Step 2: fill row boundaries with rand
    for (int i = 1; i < max_rows; i++)
        input_itemsets[i * max_cols] = rand() % 10 + 1;

    // Step 3: fill col boundaries with rand
    for (int j = 1; j < max_cols; j++)
        input_itemsets[j] = rand() % 10 + 1;

    // Step 4: fill reference from BLOSUM62
    for (int i = 1; i < max_cols; i++)
        for (int j = 1; j < max_rows; j++)
            referrence[i * max_cols + j] = blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];

    // Step 5: overwrite boundaries with penalty values
    for (int i = 1; i < max_rows; i++)
        input_itemsets[i * max_cols] = -i * penalty;
    for (int j = 1; j < max_cols; j++)
        input_itemsets[j] = -j * penalty;

    // Allocate GPU memory
    int *d_itemsets, *d_reference;
    cudaMalloc(&d_itemsets,  total * sizeof(int));
    cudaMalloc(&d_reference, total * sizeof(int));

    cudaMemcpy(d_itemsets,  input_itemsets, total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reference, referrence,     total * sizeof(int), cudaMemcpyHostToDevice);

    // Wavefront DP: anti-diagonal d from 2 to 2*dim
    int block_size = 256;
    for (int d = 2; d <= 2 * dim; d++) {
        int i_min = d - (max_cols - 1);
        if (i_min < 1) i_min = 1;
        int i_max = d - 1;
        if (i_max > max_rows - 1) i_max = max_rows - 1;

        int count = i_max - i_min + 1;
        if (count <= 0) continue;

        int grid_size = (count + block_size - 1) / block_size;
        nw_wavefront_kernel<<<grid_size, block_size>>>(d_itemsets, d_reference, max_cols, penalty, d);
        cudaDeviceSynchronize();
    }

    // Copy result back
    cudaMemcpy(input_itemsets, d_itemsets, total * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_itemsets);
    cudaFree(d_reference);

    // Write output
    FILE *fp = fopen(outfile, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", outfile); return 1; }
    for (int idx = 0; idx < total; idx++)
        fprintf(fp, "%d\t%d\n", idx, input_itemsets[idx]);
    fclose(fp);

    free(referrence);
    free(input_itemsets);
    return 0;
}
