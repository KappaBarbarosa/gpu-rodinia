#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOSUM62_SIZE 24
static int BLOSUM62[24][24] = {
    {4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4},
    {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4},
    {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4},
    {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
    {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4},
    {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -3, 1, 4, -1, -4},
    {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4},
    {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -2, -4},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -2, -4},
    {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4},
    {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -2, -4},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -2, -4},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -2, -3, -1, -1, -2, -4},
    {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4},
    {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4},
    {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -2, -2, -2, 2, 7, -1, -3, -2, -1, -4},
    {0, -3, -3, -3, -1, -2, -3, -3, -3, 3, 1, -2, 1, -1, -3, -2, 0, -3, -1, 4, -3, -2, -1, -4},
    {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
    {0, -1, -1, -1, -2, -1, -1, -1, -1, -2, -2, -1, -2, -2, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4},
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}
};

static inline int char_to_index(char c) {
    if (c == 'A') return 0;
    if (c == 'R') return 1;
    if (c == 'N') return 2;
    if (c == 'D') return 3;
    if (c == 'C') return 4;
    if (c == 'Q') return 5;
    if (c == 'E') return 6;
    if (c == 'G') return 7;
    if (c == 'H') return 8;
    if (c == 'I') return 9;
    if (c == 'L') return 10;
    if (c == 'K') return 11;
    if (c == 'M') return 12;
    if (c == 'F') return 13;
    if (c == 'P') return 14;
    if (c == 'S') return 15;
    if (c == 'T') return 16;
    if (c == 'W') return 17;
    if (c == 'Y') return 18;
    if (c == 'V') return 19;
    if (c == 'B') return 20;
    if (c == 'Z') return 21;
    if (c == 'X') return 22;
    return 23;
}

__device__ static inline int maximum(int a, int b, int c) {
    if (a >= b && a >= c) return a;
    if (b >= a && b >= c) return b;
    return c;
}

__global__ void nw_kernel(int *M, char *seqA, char *seqB, int max_rows, int max_cols, int penalty, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // For diagonal d, we compute cells (i, j) where i + j = d
    // with constraints: 0 <= i < max_rows, 0 <= j < max_cols
    if (idx >= max_rows + max_cols - 1) return;

    int i = idx;
    int j = d - i;

    if (i < 0 || i >= max_rows || j < 0 || j >= max_cols) return;

    // Boundary conditions
    if (i == 0 && j == 0) {
        M[0] = 0;
        return;
    }

    if (i == 0) {
        M[j] = -j * penalty;
        return;
    }

    if (j == 0) {
        M[i * max_cols] = -i * penalty;
        return;
    }

    // DP recurrence
    char a = seqA[i - 1];
    char b = seqB[j - 1];
    int idx_a = char_to_index(a);
    int idx_b = char_to_index(b);
    int match_score = BLOSUM62[idx_a][idx_b];

    int diag = M[(i - 1) * max_cols + (j - 1)] + match_score;
    int up = M[(i - 1) * max_cols + j] - penalty;
    int left = M[i * max_cols + (j - 1)] - penalty;

    M[i * max_cols + j] = maximum(diag, up, left);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <dim> <penalty> <output_file>\n", argv[0]);
        return 1;
    }

    int dim = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char *output_file = argv[3];

    int max_rows = dim + 1;
    int max_cols = dim + 1;
    int total_cells = max_rows * max_cols;

    // Generate sequences internally
    srand(7);
    char *seqA = (char *)malloc(dim * sizeof(char));
    char *seqB = (char *)malloc(dim * sizeof(char));

    const char alphabet[] = "ACDEFGHIKLMNPQRSTVWY";
    for (int i = 0; i < dim; i++) {
        seqA[i] = alphabet[rand() % 20];
        seqB[i] = alphabet[rand() % 20];
    }

    // Allocate device memory
    int *d_M;
    char *d_seqA, *d_seqB;
    cudaMalloc(&d_M, total_cells * sizeof(int));
    cudaMalloc(&d_seqA, dim * sizeof(char));
    cudaMalloc(&d_seqB, dim * sizeof(char));

    // Copy sequences to device
    cudaMemcpy(d_seqA, seqA, dim * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seqB, seqB, dim * sizeof(char), cudaMemcpyHostToDevice);

    // Initialize M to zero
    cudaMemset(d_M, 0, total_cells * sizeof(int));

    // Wavefront DP: process diagonals d = 0, 1, 2, ..., 2*dim
    for (int d = 0; d <= 2 * dim; d++) {
        int num_cells = (d + 1 < max_rows) ? (d + 1) : max_rows;
        num_cells = (d + 1 < max_cols) ? num_cells : max_cols;
        if (d >= max_rows) num_cells = 2 * dim - d + 1;
        if (d >= max_cols) num_cells = 2 * dim - d + 1;

        int threads_per_block = 256;
        int blocks = (num_cells + threads_per_block - 1) / threads_per_block;

        nw_kernel<<<blocks, threads_per_block>>>(d_M, d_seqA, d_seqB, max_rows, max_cols, penalty, d);
        cudaDeviceSynchronize();
    }

    // Copy result back
    int *M = (int *)malloc(total_cells * sizeof(int));
    cudaMemcpy(M, d_M, total_cells * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    for (int idx = 0; idx < total_cells; idx++) {
        fprintf(fp, "%d\t%d\n", idx, M[idx]);
    }

    fclose(fp);

    // Cleanup
    free(seqA);
    free(seqB);
    free(M);
    cudaFree(d_M);
    cudaFree(d_seqA);
    cudaFree(d_seqB);

    return 0;
}
