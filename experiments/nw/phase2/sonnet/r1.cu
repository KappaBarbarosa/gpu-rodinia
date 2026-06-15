// Needleman-Wunsch CUDA Phase 2
// Block-diagonal wavefront: ~2*num_blocks launches instead of ~2*dim
// Each thread-block computes one BLOCK_SIZE x BLOCK_SIZE tile using shared memory.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

static const int blosum62[24][24] = {
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

// Amino acid character -> BLOSUM62 index
__constant__ int d_blosum62[24][24];

// Kernel: each thread-block handles tile (bx, by) = (bx_start + blockIdx.x, diag - bx_start - blockIdx.x)
// dim3 block(BLOCK_SIZE, BLOCK_SIZE)
__global__ void nw_tile_kernel(
    const int* __restrict__ seqA,
    const int* __restrict__ seqB,
    int* __restrict__ matrix,
    int dim,
    int penalty,
    int bx_start,
    int diag,
    int num_blocks
) {
    // Tile indices
    int bx = bx_start + blockIdx.x;
    int by = diag - bx;

    // Bounds check (shouldn't be needed if launch is correct, but be safe)
    if (bx < 0 || bx >= num_blocks || by < 0 || by >= num_blocks)
        return;

    int tx = threadIdx.x; // column within tile
    int ty = threadIdx.y; // row within tile

    // Starting global matrix indices (1-based in the (dim+1)x(dim+1) matrix)
    int row0 = by * BLOCK_SIZE; // top halo row
    int col0 = bx * BLOCK_SIZE; // left halo col

    int stride = dim + 1;

    // Shared memory: (BS+1) x (BS+1)
    // sh[0][0..BS] = top halo
    // sh[0..BS][0] = left halo
    // sh[1..BS][1..BS] = tile cells to compute
    __shared__ int sh[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

    // Load top halo row: sh[0][tx+1] = matrix[row0][col0 + tx + 1]
    // Loaded by threads with ty == 0
    if (ty == 0) {
        int col = col0 + tx + 1;
        if (col <= dim)
            sh[0][tx + 1] = matrix[row0 * stride + col];
        else
            sh[0][tx + 1] = 0; // out of bounds won't be used
    }
    // Load left halo col: sh[ty+1][0] = matrix[row0 + ty + 1][col0]
    // Loaded by threads with tx == 0
    if (tx == 0) {
        int row = row0 + ty + 1;
        if (row <= dim)
            sh[ty + 1][0] = matrix[row * stride + col0];
        else
            sh[ty + 1][0] = 0;
    }
    // Load corner: sh[0][0] = matrix[row0][col0]
    if (tx == 0 && ty == 0) {
        sh[0][0] = matrix[row0 * stride + col0];
    }
    __syncthreads();

    // Wavefront sweep: diagonal steps k = 0 .. 2*(BLOCK_SIZE-1)
    // At step k, thread (tx, ty) is active when tx + ty == k
    for (int k = 0; k < 2 * BLOCK_SIZE - 1; k++) {
        if (tx + ty == k) {
            int row_g = row0 + ty + 1; // global row (1-indexed) = row0 + ty + 1
            int col_g = col0 + tx + 1; // global col
            int seqA_idx = row_g - 1;  // 0-indexed into seqA
            int seqB_idx = col_g - 1;  // 0-indexed into seqB

            int match_score = d_blosum62[seqA[seqA_idx]][seqB[seqB_idx]];

            int i = ty + 1;
            int j = tx + 1;
            int val = sh[i-1][j-1] + match_score;
            int up  = sh[i-1][j]   - penalty;
            int lft = sh[i][j-1]   - penalty;
            if (up  > val) val = up;
            if (lft > val) val = lft;
            sh[i][j] = val;
        }
        __syncthreads();
    }

    // Write tile back
    {
        int row = row0 + ty + 1;
        int col = col0 + tx + 1;
        if (row <= dim && col <= dim) {
            matrix[row * stride + col] = sh[ty + 1][tx + 1];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <dimension> <penalty> <output_file>\n", argv[0]);
        return 1;
    }

    int dim     = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char* outfile = argv[3];

    // Copy BLOSUM62 to constant memory
    cudaMemcpyToSymbol(d_blosum62, blosum62, sizeof(blosum62));

    // Allocate host arrays
    int* seqA   = (int*)malloc(dim * sizeof(int));
    int* seqB   = (int*)malloc(dim * sizeof(int));
    int  sz     = (dim + 1) * (dim + 1);
    int* matrix = (int*)malloc(sz * sizeof(int));

    // Generate sequences: srand(7), use amino alphabet
    static const char amino[] = "ABCDEFGHIKLMNOPQRSTUVWXYZ";
    static const int  amino_to_blosum[256] = {
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23, 0,20, 4, 3, 6,13, 7, 8, 9,23,10,11,12, 2,23,
        14, 5, 1,15,16,23,19,17,22,18,21,23,23,23,23,23,
        23, 0,20, 4, 3, 6,13, 7, 8, 9,23,10,11,12, 2,23,
        14, 5, 1,15,16,23,19,17,22,18,21,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,
        23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23
    };

    srand(7);
    for (int i = 0; i < dim; i++) {
        char c = amino[rand() % 24];
        seqA[i] = amino_to_blosum[(unsigned char)c];
    }
    for (int i = 0; i < dim; i++) {
        char c = amino[rand() % 24];
        seqB[i] = amino_to_blosum[(unsigned char)c];
    }

    // Boundary conditions
    int stride = dim + 1;
    for (int i = 0; i <= dim; i++) {
        matrix[i * stride + 0] = -i * penalty;
        matrix[0 * stride + i] = -i * penalty;
    }

    // GPU memory
    int* d_seqA;
    int* d_seqB;
    int* d_matrix;

    cudaMalloc(&d_seqA,   dim * sizeof(int));
    cudaMalloc(&d_seqB,   dim * sizeof(int));
    cudaMalloc(&d_matrix, sz  * sizeof(int));

    cudaMemcpy(d_seqA,   seqA,   dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seqB,   seqB,   dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, matrix, sz  * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Launch kernels along block-diagonals
    // block-diagonal d: bx + by == d, bx in [max(0, d-num_blocks+1), min(d, num_blocks-1)]
    for (int d = 0; d < 2 * num_blocks - 1; d++) {
        int bx_start = (d >= num_blocks) ? (d - num_blocks + 1) : 0;
        int bx_end   = (d < num_blocks)  ? d                    : (num_blocks - 1);
        int num_tb   = bx_end - bx_start + 1;

        dim3 grid(num_tb, 1);
        nw_tile_kernel<<<grid, block>>>(
            d_seqA, d_seqB, d_matrix,
            dim, penalty,
            bx_start, d, num_blocks
        );
    }
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(matrix, d_matrix, sz * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output: full (dim+1)x(dim+1) matrix, row-major, "%d\t%d\n" per cell
    FILE* fp = fopen(outfile, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open output file: %s\n", outfile);
        return 1;
    }
    for (int i = 0; i <= dim; i++) {
        for (int j = 0; j <= dim; j++) {
            fprintf(fp, "%d\t%d\n", i * stride + j, matrix[i * stride + j]);
        }
    }
    fclose(fp);

    cudaFree(d_seqA);
    cudaFree(d_seqB);
    cudaFree(d_matrix);
    free(seqA);
    free(seqB);
    free(matrix);

    return 0;
}
