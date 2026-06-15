#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define PENALTY 10
#define LIMIT -999999

// BLOSUM62 matrix (23x23)
static const int BLOSUM62[23][23] = {
    {  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -1, -1,  1,  0,  0, -3, -2,  0, -2, -1 }, // A
    { -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2,  0, -1, -1, -3, -2, -3,  0, -1 }, // B
    { -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -1, -4, -2, -3,  3, -1 }, // C
    { -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -1, -4, -3, -3,  4, -1 }, // D
    {  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -1, -2, -2, -1, -3, -2 }, // E
    { -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -1, -2, -1, -3,  0, -1 }, // F
    { -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -1, -3, -2, -3,  1, -1 }, // G
    {  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -1, -2, -3, -3, -1, -2 }, // H
    { -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -1, -2, -2, -2,  0, -1 }, // I
    { -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -1, -3, -1,  0, -3, -1 }, // J
    { -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -1, -2, -1,  0, -3, -1 }, // K
    { -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -1, -3, -2, -2,  0, -1 }, // L
    { -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1, -1,  0, -2, -1 }, // M
    { -1, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2, -1,  2,  0, -3, -3, -1 }, // N
    { -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -2, -1, -4, -3, -2, -1, -2 }, // O
    {  1,  0,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -1, -3, -2,  0,  0,  0 }, // P
    {  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -2,  1,  5,  0, -2, -2,  0, -1,  0 }, // Q
    {  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1,  0,  5, -1, -2, -1, -1,  0 }, // R
    { -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  2, -4, -3, -2, -1,  11, -2, -2, -4, -3 }, // S
    { -2, -2, -2, -3, -2, -1, -2, -3, -2, -1, -1, -2, -1,  0, -3, -2, -2, -2, -2,  7, -1, -3, -2 }, // T
    {  0, -3, -3, -3, -1, -3, -3, -3, -2,  0,  0, -2,  0, -3, -2,  0,  0, -1, -2, -1,  4, -3, -2 }, // U
    { -2,  0,  3,  4, -3,  0,  1, -1,  0, -3, -3,  0, -2, -3, -1,  0, -1, -1, -4, -3, -3,  4, -1 }, // V
    { -1, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -1, -1, -1, -2,  0,  0,  0, -3, -2, -2, -1, -2 }, // W
};

// Device function for three-way maximum
__device__ int maximum(int a, int b, int c) {
    if (a >= b && a >= c) return a;
    if (b >= a && b >= c) return b;
    return c;
}

// Kernel: process one anti-diagonal
__global__ void nw_kernel(int *M, int *ref, int diag_idx, int max_rows, int max_cols, int penalty) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // For anti-diagonal diag_idx, compute all cells (i, j) where i + j == diag_idx
    // i ranges from 1 to max_rows-1, j ranges from 1 to max_cols-1

    int i = idx + 1;  // Start from row 1
    int j = diag_idx - i;

    // Check bounds: 1 <= i < max_rows and 1 <= j < max_cols
    if (i >= 1 && i < max_rows && j >= 1 && j < max_cols) {
        int diag_val = M[(i-1)*max_cols + (j-1)] + ref[i*max_cols + j];
        int left_val = M[i*max_cols + (j-1)] - penalty;
        int up_val = M[(i-1)*max_cols + j] - penalty;

        M[i*max_cols + j] = maximum(diag_val, left_val, up_val);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <dimension> <penalty> <output_file>\n", argv[0]);
        return 1;
    }

    int dimension = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char *output_file = argv[3];

    int max_rows = dimension + 1;
    int max_cols = dimension + 1;

    // Host memory allocation
    int *h_M = (int *)malloc(max_rows * max_cols * sizeof(int));
    int *h_ref = (int *)malloc(max_rows * max_cols * sizeof(int));
    unsigned char *h_input = (unsigned char *)malloc(max_rows * max_cols * sizeof(unsigned char));

    // Initialize
    srand(7);
    memset(h_M, 0, max_rows * max_cols * sizeof(int));
    memset(h_input, 0, max_rows * max_cols * sizeof(unsigned char));

    // Set first column of M (excluding [0][0])
    for (int i = 1; i < max_rows; i++) {
        h_M[i * max_cols + 0] = -i * penalty;
    }

    // Set first row of M (excluding [0][0])
    for (int j = 1; j < max_cols; j++) {
        h_M[0 * max_cols + j] = -j * penalty;
    }

    // Set input and ref (skip [0][*] and [*][0])
    for (int i = 1; i < max_rows; i++) {
        for (int j = 1; j < max_cols; j++) {
            h_input[i * max_cols + j] = (unsigned char)(rand() % 23);
            h_ref[i * max_cols + j] = BLOSUM62[h_input[i * max_cols + j]][h_input[0 * max_cols + j]];
        }
    }

    // Initialize first row input (column 0 has row indices, row 0 has column indices)
    for (int i = 1; i < max_rows; i++) {
        h_input[i * max_cols + 0] = (unsigned char)(rand() % 23);
    }
    for (int j = 1; j < max_cols; j++) {
        h_input[0 * max_cols + j] = (unsigned char)(rand() % 23);
    }

    // Recalculate ref based on input
    for (int i = 1; i < max_rows; i++) {
        for (int j = 1; j < max_cols; j++) {
            h_ref[i * max_cols + j] = BLOSUM62[h_input[i * max_cols + 0]][h_input[0 * max_cols + j]];
        }
    }

    // Device memory allocation
    int *d_M, *d_ref;
    cudaMalloc(&d_M, max_rows * max_cols * sizeof(int));
    cudaMalloc(&d_ref, max_rows * max_cols * sizeof(int));

    // Copy to device
    cudaMemcpy(d_M, h_M, max_rows * max_cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, h_ref, max_rows * max_cols * sizeof(int), cudaMemcpyHostToDevice);

    // Process anti-diagonals
    // Anti-diagonal diag_idx has cells where i + j == diag_idx
    // diag_idx ranges from 2 to 2*(dimension)
    for (int diag_idx = 2; diag_idx <= 2 * dimension; diag_idx++) {
        // Number of cells on this anti-diagonal
        int num_cells = (diag_idx < max_cols) ? (diag_idx - 1) : (2 * dimension + 2 - diag_idx - 1);
        if (num_cells <= 0) continue;

        int threads_per_block = 256;
        int num_blocks = (num_cells + threads_per_block - 1) / threads_per_block;

        nw_kernel<<<num_blocks, threads_per_block>>>(d_M, d_ref, diag_idx, max_rows, max_cols, penalty);
        cudaDeviceSynchronize();
    }

    // Copy result back
    cudaMemcpy(h_M, d_M, max_rows * max_cols * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for writing\n", output_file);
        return 1;
    }

    for (int idx = 0; idx < max_rows * max_cols; idx++) {
        fprintf(fp, "%d\t%d\n", idx, h_M[idx]);
    }

    fclose(fp);

    // Cleanup
    cudaFree(d_M);
    cudaFree(d_ref);
    free(h_M);
    free(h_ref);
    free(h_input);

    return 0;
}
