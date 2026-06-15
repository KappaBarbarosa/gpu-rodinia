#include <stdlib.h>
#include <stdio.h>
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

__device__ __forceinline__ int maximum_d(int a,int b,int c){
    int k=(a<=b)?b:a; return (k<=c)?c:k;
}

// One thread per cell on anti-diagonal d. i in [i_lo, i_hi], j = d - i.
__global__ void nw_diag(int *items, const int *ref, int max_cols,
                        int d, int i_lo, int i_hi, int penalty){
    int i = i_lo + blockIdx.x * blockDim.x + threadIdx.x;
    if (i > i_hi) return;
    int j = d - i;
    int idx = i*max_cols + j;
    int diag = items[(i-1)*max_cols + (j-1)] + ref[idx];
    int left = items[i*max_cols + (j-1)] - penalty;
    int up   = items[(i-1)*max_cols + j] - penalty;
    items[idx] = maximum_d(diag, left, up);
}

int main(int argc, char **argv){
    if (argc != 4){
        fprintf(stderr, "usage: %s <dimension> <penalty> <output_file>\n", argv[0]);
        return 1;
    }
    int dim = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char *outfile = argv[3];

    int max_rows = dim + 1;
    int max_cols = dim + 1;
    size_t n = (size_t)max_rows * max_cols;

    int *referrence = (int*)malloc(n * sizeof(int));
    int *input_itemsets = (int*)malloc(n * sizeof(int));

    srand(7);
    for (size_t k = 0; k < n; k++) input_itemsets[k] = 0;

    for (int i = 1; i < max_rows; i++)
        input_itemsets[i*max_cols] = rand()%10 + 1;
    for (int j = 1; j < max_cols; j++)
        input_itemsets[j] = rand()%10 + 1;

    for (int i = 1; i < max_cols; i++)
        for (int j = 1; j < max_rows; j++)
            referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];

    for (int i = 1; i < max_rows; i++)
        input_itemsets[i*max_cols] = -i*penalty;
    for (int j = 1; j < max_cols; j++)
        input_itemsets[j] = -j*penalty;

    // GPU DP fill via anti-diagonal wavefront
    int *d_items, *d_ref;
    cudaMalloc(&d_items, n*sizeof(int));
    cudaMalloc(&d_ref, n*sizeof(int));
    cudaMemcpy(d_items, input_itemsets, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, referrence, n*sizeof(int), cudaMemcpyHostToDevice);

    const int TPB = 256;
    for (int d = 2; d <= 2*dim; d++){
        int i_lo = (1 > d-(max_cols-1)) ? 1 : d-(max_cols-1);
        int i_hi = (max_rows-1 < d-1) ? max_rows-1 : d-1;
        if (i_lo > i_hi) continue;
        int cnt = i_hi - i_lo + 1;
        int blocks = (cnt + TPB - 1) / TPB;
        nw_diag<<<blocks, TPB>>>(d_items, d_ref, max_cols, d, i_lo, i_hi, penalty);
    }

    cudaMemcpy(input_itemsets, d_items, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    FILE *fp = fopen(outfile, "w");
    if (!fp){ fprintf(stderr, "cannot open %s\n", outfile); return 1; }
    for (size_t idx = 0; idx < n; idx++)
        fprintf(fp, "%zu\t%d\n", idx, input_itemsets[idx]);
    fclose(fp);

    cudaFree(d_items);
    cudaFree(d_ref);
    free(referrence);
    free(input_itemsets);
    return 0;
}
