// Needleman-Wunsch CUDA — block-diagonal wavefront, corrected reference indexing
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOSUM62_SIZE 24

static const int BLOSUM62[BLOSUM62_SIZE][BLOSUM62_SIZE] = {
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

// Tile size for shared memory kernel
#define TILE 32

// Each thread block handles one tile on one diagonal
// Kernel processes all tiles on the same anti-diagonal simultaneously
__global__ void nw_kernel(
    int* __restrict__ matrix,
    const int* __restrict__ ref,
    int max_cols,
    int penalty,
    int diag,       // current tile-diagonal index (0-based)
    int n_tiles     // total tiles per row/col = (max_cols-1)/TILE
) {
    // blockIdx.x = which tile on this anti-diagonal
    int tile_row = diag - blockIdx.x;
    int tile_col = blockIdx.x;

    if (tile_row < 0 || tile_col < 0 || tile_row >= n_tiles || tile_col >= n_tiles)
        return;

    // Global top-left corner of this tile in the interior (1-indexed)
    int gi0 = tile_row * TILE + 1;
    int gj0 = tile_col * TILE + 1;

    // Shared memory: tile plus left halo (col -1) and top halo (row -1)
    // Layout: (TILE+1) x (TILE+1), index [row][col], row0=halo row, col0=halo col
    __shared__ int sm[TILE+1][TILE+1];

    int tx = threadIdx.x; // col within tile (0..TILE-1)
    int ty = threadIdx.y; // row within tile (0..TILE-1)

    // Load halo: top-left corner cell (gi0-1, gj0-1)
    if (tx == 0 && ty == 0) {
        sm[0][0] = matrix[(gi0-1)*max_cols + (gj0-1)];
    }
    // Load top halo row (gi0-1, gj0..gj0+TILE-1)
    if (ty == 0) {
        int gj = gj0 + tx;
        if (gj < max_cols)
            sm[0][tx+1] = matrix[(gi0-1)*max_cols + gj];
        else
            sm[0][tx+1] = 0;
    }
    // Load left halo col (gi0..gi0+TILE-1, gj0-1)
    if (tx == 0) {
        int gi = gi0 + ty;
        if (gi < max_cols)
            sm[ty+1][0] = matrix[gi*max_cols + (gj0-1)];
        else
            sm[ty+1][0] = 0;
    }
    __syncthreads();

    // Process cells within tile sequentially along internal anti-diagonals
    // internal diag d: cells where tx+ty == d, d in 0..2*(TILE-1)
    for (int d = 0; d < 2*TILE - 1; d++) {
        // Check if this thread participates
        if (tx + ty == d) {
            int gi = gi0 + ty;
            int gj = gj0 + tx;
            if (gi < max_cols && gj < max_cols) {
                int top_left = sm[ty][tx];
                int top      = sm[ty][tx+1];
                int left     = sm[ty+1][tx];
                int r        = ref[gi * max_cols + gj];
                int val = top_left + r;
                int v2  = top  - penalty;
                int v3  = left - penalty;
                if (v2 > val) val = v2;
                if (v3 > val) val = v3;
                sm[ty+1][tx+1] = val;
                matrix[gi * max_cols + gj] = val;
            }
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <max_rows> <penalty>\n", argv[0]);
        return 1;
    }

    int max_rows = atoi(argv[1]) + 1;
    int max_cols = max_rows;
    int penalty  = atoi(argv[2]);

    size_t sz = (size_t)max_rows * max_cols * sizeof(int);

    int* input_itemsets = (int*)malloc(sz);
    int* output         = (int*)malloc(sz);
    int* ref_h          = (int*)malloc(sz);
    memset(input_itemsets, 0, sz);
    memset(ref_h, 0, sz);

    // Initialize sequences using srand(7)
    srand(7);
    for (int i = 1; i < max_rows; i++) {
        input_itemsets[i * max_cols + 0] = rand() % 10 + 1;
    }
    for (int j = 1; j < max_cols; j++) {
        input_itemsets[0 * max_cols + j] = rand() % 10 + 1;
    }

    // Compute reference matrix on host BEFORE overwriting boundaries
    // ref[i][j] = BLOSUM62[ row_symbol ][ col_symbol ]
    // row_symbol = input_itemsets[i * max_cols + 0]
    // col_symbol = input_itemsets[0 * max_cols + j]
    for (int i = 1; i < max_rows; i++) {
        for (int j = 1; j < max_cols; j++) {
            ref_h[i * max_cols + j] = BLOSUM62[ input_itemsets[i * max_cols + 0] ][ input_itemsets[0 * max_cols + j] ];
        }
    }

    // Initialize boundary conditions
    for (int i = 1; i < max_rows; i++) {
        input_itemsets[i * max_cols + 0] = -i * penalty;
    }
    for (int j = 1; j < max_cols; j++) {
        input_itemsets[0 * max_cols + j] = -j * penalty;
    }
    input_itemsets[0] = 0;

    // Copy to GPU
    int* d_matrix;
    int* d_ref;
    cudaMalloc(&d_matrix, sz);
    cudaMalloc(&d_ref, sz);
    cudaMemcpy(d_matrix, input_itemsets, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, ref_h, sz, cudaMemcpyHostToDevice);

    int n_tiles = (max_rows - 1 + TILE - 1) / TILE;
    dim3 block(TILE, TILE);

    // Sweep tile-diagonals: diag 0..2*n_tiles-2
    for (int diag = 0; diag < 2 * n_tiles - 1; diag++) {
        // Number of tiles on this anti-diagonal
        int n_active = diag + 1;
        if (n_active > n_tiles) n_active = 2 * n_tiles - 1 - diag;
        // Starting tile_col for this diagonal
        int start_col = (diag < n_tiles) ? 0 : (diag - n_tiles + 1);

        dim3 grid(n_active);
        nw_kernel<<<grid, block>>>(d_matrix, d_ref, max_cols, penalty, diag, n_tiles);
        cudaDeviceSynchronize();
    }

    // Copy result back
    cudaMemcpy(output, d_matrix, sz, cudaMemcpyDeviceToHost);

    // Print full matrix
    for (int i = 0; i < max_rows; i++) {
        for (int j = 0; j < max_cols; j++) {
            printf("%d\t%d\n", i * max_cols + j, output[i * max_cols + j]);
        }
    }

    free(input_itemsets);
    free(output);
    free(ref_h);
    cudaFree(d_matrix);
    cudaFree(d_ref);
    return 0;
}
