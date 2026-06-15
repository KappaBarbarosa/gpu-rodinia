#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// BLOSUM62 substitution matrix (verbatim, Rodinia)
static int blosum62[24][24] = {
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
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

__device__ __host__ inline int maximum(int a, int b, int c) {
    int k = a > b ? a : b;
    return k > c ? k : c;
}

// Block-diagonal wavefront kernel.
// Each thread-block processes one BLOCK_SIZE x BLOCK_SIZE tile of the interior.
// blk identifies which block-diagonal we are on; offset_r selects whether we're
// in the top-left (growing) or bottom-right (shrinking) pass.
__global__ void nw_kernel(int *reference, int *matrix, int dim, int penalty,
                          int blk, int block_width, int offset_r, int offset_c)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // block index in the block-grid for this diagonal
    int b_index_x = bx + offset_c;
    int b_index_y = blk - 1 - bx + offset_r;

    // top-left corner of this tile in the (dim x dim) padded matrix coords.
    // Interior cells are at indices [1..dim-1]; matrix is (dim) wide.
    int base = dim * (b_index_y * BLOCK_SIZE + 1) + b_index_x * BLOCK_SIZE + 1;

    // shared tile with halo: row 0 = top halo, col 0 = left halo
    __shared__ int sm[(BLOCK_SIZE + 1) * (BLOCK_SIZE + 1)];
    __shared__ int ref[BLOCK_SIZE * BLOCK_SIZE];

    // top-left diagonal halo corner: matrix[base - dim - 1]
    if (tx == 0)
        sm[0] = matrix[base - dim - 1];

    // load reference scores for this tile
    for (int i = 0; i < BLOCK_SIZE; ++i)
        ref[i * BLOCK_SIZE + tx] = reference[base + dim * i + tx];

    __syncthreads();

    // left column halo: matrix entry just left of each tile row
    sm[(tx + 1) * (BLOCK_SIZE + 1)] = matrix[base + dim * tx - 1];
    __syncthreads();

    // top row halo: matrix entry just above each tile column
    sm[tx + 1] = matrix[base - dim + tx];
    __syncthreads();

    // diagonal sweep: fill tile (rows/cols 1..BLOCK_SIZE in shared array)
    for (int m = 0; m < BLOCK_SIZE; ++m) {
        if (tx <= m) {
            int t_x = tx;
            int t_y = m - tx;
            int idx = (t_y + 1) * (BLOCK_SIZE + 1) + (t_x + 1);
            sm[idx] = maximum(
                sm[t_y * (BLOCK_SIZE + 1) + t_x] + ref[t_y * BLOCK_SIZE + t_x],
                sm[(t_y + 1) * (BLOCK_SIZE + 1) + t_x] - penalty,
                sm[t_y * (BLOCK_SIZE + 1) + (t_x + 1)] - penalty);
        }
        __syncthreads();
    }

    for (int m = BLOCK_SIZE - 2; m >= 0; --m) {
        if (tx <= m) {
            int t_x = tx + BLOCK_SIZE - m - 1;
            int t_y = BLOCK_SIZE - tx - 1;
            int idx = (t_y + 1) * (BLOCK_SIZE + 1) + (t_x + 1);
            sm[idx] = maximum(
                sm[t_y * (BLOCK_SIZE + 1) + t_x] + ref[t_y * BLOCK_SIZE + t_x],
                sm[(t_y + 1) * (BLOCK_SIZE + 1) + t_x] - penalty,
                sm[t_y * (BLOCK_SIZE + 1) + (t_x + 1)] - penalty);
        }
        __syncthreads();
    }

    // write tile back to global matrix
    for (int i = 0; i < BLOCK_SIZE; ++i)
        matrix[base + dim * i + tx] = sm[(i + 1) * (BLOCK_SIZE + 1) + (tx + 1)];
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <dimension> <penalty> <output_file>\n", argv[0]);
        return 1;
    }

    int max_rows = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char *outfile = argv[3];

    // matrix is (dim) x (dim) with dim = max_rows + 1
    int dim = max_rows + 1;
    int n = dim * dim;

    int *reference = (int *)malloc(n * sizeof(int));
    int *matrix = (int *)malloc(n * sizeof(int));
    int *input_itemsets = (int *)malloc(n * sizeof(int));
    int *seq_a = (int *)malloc(dim * sizeof(int));
    int *seq_b = (int *)malloc(dim * sizeof(int));

    if (!reference || !matrix || !input_itemsets || !seq_a || !seq_b) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    memset(input_itemsets, 0, n * sizeof(int));

    srand(7);
    // random sequences (1..20)
    for (int i = 1; i < dim; i++)
        seq_a[i] = rand() % 10 + 1;
    for (int j = 1; j < dim; j++)
        seq_b[j] = rand() % 10 + 1;

    // build reference scores from BLOSUM62
    for (int i = 1; i < dim; i++)
        for (int j = 1; j < dim; j++)
            reference[i * dim + j] = blosum62[seq_a[i]][seq_b[j]];

    // boundary init
    for (int i = 1; i < dim; i++)
        input_itemsets[i * dim] = -i * penalty;
    for (int j = 1; j < dim; j++)
        input_itemsets[j] = -j * penalty;

    // device buffers
    int *d_reference, *d_matrix;
    cudaMalloc((void **)&d_reference, n * sizeof(int));
    cudaMalloc((void **)&d_matrix, n * sizeof(int));
    cudaMemcpy(d_reference, reference, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, input_itemsets, n * sizeof(int), cudaMemcpyHostToDevice);

    // number of interior cells per side = max_rows; tiled into block_width tiles
    int block_width = (dim - 1) / BLOCK_SIZE;  // = max_rows / BLOCK_SIZE

    // Pass 1: growing block-diagonals over the top-left triangle.
    for (int blk = 1; blk <= block_width; blk++) {
        dim3 grid(blk, 1);
        dim3 threads(BLOCK_SIZE, 1);
        nw_kernel<<<grid, threads>>>(d_reference, d_matrix, dim, penalty,
                                     blk, block_width, 0, 0);
    }

    // Pass 2: shrinking block-diagonals over the bottom-right triangle.
    for (int blk = block_width - 1; blk >= 1; blk--) {
        dim3 grid(blk, 1);
        dim3 threads(BLOCK_SIZE, 1);
        // offset_r and offset_c shift the diagonal into the lower-right corner.
        nw_kernel<<<grid, threads>>>(d_reference, d_matrix, dim, penalty,
                                     blk, block_width,
                                     block_width - blk, block_width - blk);
    }

    cudaMemcpy(matrix, d_matrix, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // write FULL matrix, row-major, "%d\t%d\n" per cell
    FILE *fp = fopen(outfile, "w");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", outfile);
        return 1;
    }
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            fprintf(fp, "%d\t%d\n", i * dim + j, matrix[i * dim + j]);
    fclose(fp);

    cudaFree(d_reference);
    cudaFree(d_matrix);
    free(reference);
    free(matrix);
    free(input_itemsets);
    free(seq_a);
    free(seq_b);
    return 0;
}
