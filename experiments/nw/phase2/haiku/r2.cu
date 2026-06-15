#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MATCH_SCORE 1
#define MISMATCH_SCORE -1
#define GAP_PENALTY 2

__global__ void nw_kernel(int *M, char *A, char *B, int penalty, int max_cols, int max_rows, int d) {
    int threadIdx_x = threadIdx.x;
    int blockIdx_x = blockIdx.x;
    int blockDim_x = blockDim.x;

    int idx = blockIdx_x * blockDim_x + threadIdx_x;

    // For diagonal d, valid i range is: max(1, d-(max_cols-1)) to min(max_rows-1, d-1)
    int i_min = max(1, d - (max_cols - 1));
    int i_max = min(max_rows - 1, d - 1);

    int i = i_min + idx;

    // Guard: only process if i is within the valid range
    if (i <= i_max) {
        int j = d - i;

        // Validate j is in bounds (should always be, but safety check)
        if (j >= 1 && j < max_cols) {
            int idx_diag = i * max_cols + j;
            int idx_diag_prev = (i - 1) * max_cols + (j - 1);
            int idx_left = i * max_cols + (j - 1);
            int idx_up = (i - 1) * max_cols + j;

            int match_score = (A[i] == B[j]) ? MATCH_SCORE : MISMATCH_SCORE;

            int score_diag = M[idx_diag_prev] + match_score;
            int score_left = M[idx_left] - penalty;
            int score_up = M[idx_up] - penalty;

            M[idx_diag] = max(score_diag, max(score_left, score_up));
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <penalty>\n", argv[0]);
        return 1;
    }

    int penalty = atoi(argv[1]);

    int max_rows, max_cols;
    if (scanf("%d %d", &max_rows, &max_cols) != 2) {
        fprintf(stderr, "Failed to read dimensions\n");
        return 1;
    }

    // Allocate and read sequences
    char *A = (char *)malloc(max_rows);
    char *B = (char *)malloc(max_cols);

    if (scanf("%s %s", A, B) != 2) {
        fprintf(stderr, "Failed to read sequences\n");
        return 1;
    }

    // Allocate GPU memory for M (rows x cols matrix)
    size_t M_size = max_rows * max_cols * sizeof(int);
    int *d_M;
    cudaMalloc(&d_M, M_size);

    // Allocate GPU memory for sequences
    char *d_A, *d_B;
    cudaMalloc(&d_A, max_rows);
    cudaMalloc(&d_B, max_cols);

    // Copy sequences to GPU
    cudaMemcpy(d_A, A, max_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, max_cols, cudaMemcpyHostToDevice);

    // Initialize M matrix on host
    int *M = (int *)calloc(max_rows * max_cols, sizeof(int));

    // Initialize first row: M[0][j] = -j * penalty
    for (int j = 0; j < max_cols; j++) {
        M[0 * max_cols + j] = -j * penalty;
    }

    // Initialize first column: M[i][0] = -i * penalty
    for (int i = 0; i < max_rows; i++) {
        M[i * max_cols + 0] = -i * penalty;
    }

    // Copy initialized matrix to GPU
    cudaMemcpy(d_M, M, M_size, cudaMemcpyHostToDevice);

    // Process anti-diagonals from d=2 to d=2*max_rows (covering all interior cells)
    // Actually, the last diagonal is d = (max_rows-1) + (max_cols-1), but we want up to cell [max_rows-1][max_cols-1]
    // For that cell: i = max_rows-1, j = max_cols-1, so d = i + j = max_rows + max_cols - 2
    // But we're iterating through diagonals where i+j=d and both i,j are in valid ranges
    // The maximum value of d is when i=max_rows-1 and j=max_cols-1, so d_max = max_rows + max_cols - 2

    int d_max = max_rows + max_cols - 2;

    for (int d = 2; d <= d_max; d++) {
        int i_min = max(1, d - (max_cols - 1));
        int i_max = min(max_rows - 1, d - 1);

        int num_threads = i_max - i_min + 1;

        if (num_threads > 0) {
            int blockSize = 256;
            int numBlocks = (num_threads + blockSize - 1) / blockSize;

            nw_kernel<<<numBlocks, blockSize>>>(d_M, d_A, d_B, penalty, max_cols, max_rows, d);
            cudaDeviceSynchronize();
        }
    }

    // Copy result back
    cudaMemcpy(M, d_M, M_size, cudaMemcpyDeviceToHost);

    // Output full matrix
    for (int i = 0; i < max_rows; i++) {
        for (int j = 0; j < max_cols; j++) {
            if (j > 0) printf("\t");
            printf("%d", M[i * max_cols + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(A);
    free(B);
    free(M);
    cudaFree(d_M);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
