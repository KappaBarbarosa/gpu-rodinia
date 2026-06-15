/*
 * Needleman-Wunsch CUDA implementation
 * Phase 2 Round 3
 * - Correct two-pass block-diagonal wavefront (all num_blocks*num_blocks tiles covered)
 * - Output written to argv[3] in full "idx\tvalue\n" format
 * - Sequences passed separately; border initialized with gap penalties
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__device__ static inline int max3(int a, int b, int c) {
    int t = a > b ? a : b;
    return t > c ? t : c;
}

/*
 * Kernel: compute one BLOCK_SIZE x BLOCK_SIZE tile at block position (blk_row, blk_col).
 * M       : full (max_rows x max_cols) DP matrix (row-major), borders pre-initialized.
 * seq1    : sequence 1 (length elements, 0-indexed), indexes rows 1..length
 * seq2    : sequence 2 (length elements, 0-indexed), indexes cols 1..length
 * sub     : 21x21 substitution matrix (row-major)
 * gap     : gap penalty (negative int)
 */
__global__ void nw_tile_kernel(int *M, const int *seq1, const int *seq2,
                               const int *sub, int gap,
                               int max_cols, int blk_row, int blk_col)
{
    // Shared memory: (BLOCK_SIZE+1) x (BLOCK_SIZE+1), includes left/top halo
    __shared__ int sm[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

    int tx = threadIdx.x; // 0..BLOCK_SIZE-1  -> local col
    int ty = threadIdx.y; // 0..BLOCK_SIZE-1  -> local row

    // Global (1-based) row and col for this thread's DP cell
    int grow = blk_row * BLOCK_SIZE + ty + 1;  // row in M
    int gcol = blk_col * BLOCK_SIZE + tx + 1;  // col in M

    // Load halo from global memory (already computed by previous tiles/border init)
    if (tx == 0 && ty == 0)
        sm[0][0] = M[(grow - 1) * max_cols + (gcol - 1)];
    if (ty == 0)
        sm[0][tx + 1] = M[(grow - 1) * max_cols + gcol];
    if (tx == 0)
        sm[ty + 1][0] = M[grow * max_cols + (gcol - 1)];
    __syncthreads();

    // Wavefront within tile: iterate over diagonals 0..2*BLOCK_SIZE-2
    for (int d = 0; d < 2 * BLOCK_SIZE - 1; d++) {
        if (ty + tx == d) {
            int match  = sm[ty][tx]     + sub[seq1[grow - 1] * 21 + seq2[gcol - 1]];
            int del_op = sm[ty][tx + 1] + gap;   // gap in seq2 (delete)
            int ins_op = sm[ty + 1][tx] + gap;   // gap in seq1 (insert)
            int val = max3(match, del_op, ins_op);
            sm[ty + 1][tx + 1] = val;
            M[grow * max_cols + gcol] = val;
        }
        __syncthreads();
    }
}

static void read_sub_matrix(const char *fname, int *sub) {
    FILE *fp = fopen(fname, "r");
    if (!fp) { fprintf(stderr, "Cannot open sub matrix: %s\n", fname); exit(1); }
    for (int i = 0; i < 21 * 21; i++) fscanf(fp, "%d", &sub[i]);
    fclose(fp);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input> <sub_matrix> <output>\n", argv[0]);
        return 1;
    }
    const char *in_file  = argv[1];
    const char *sub_file = argv[2];
    const char *out_file = argv[3];

    // --- Read substitution matrix ---
    int *sub_h = (int *)malloc(21 * 21 * sizeof(int));
    read_sub_matrix(sub_file, sub_h);

    // --- Read input file ---
    // Format: <length>\n<seq1 values...>\n<seq2 values...>\n<gap>
    FILE *fp = fopen(in_file, "r");
    if (!fp) { fprintf(stderr, "Cannot open input: %s\n", in_file); return 1; }
    int length;
    fscanf(fp, "%d", &length);
    int *seq1_h = (int *)malloc(length * sizeof(int));
    int *seq2_h = (int *)malloc(length * sizeof(int));
    for (int i = 0; i < length; i++) fscanf(fp, "%d", &seq1_h[i]);
    for (int i = 0; i < length; i++) fscanf(fp, "%d", &seq2_h[i]);
    int gap;
    fscanf(fp, "%d", &gap);
    fclose(fp);

    int max_rows = length + 1;
    int max_cols = length + 1;
    long long total = (long long)max_rows * max_cols;

    // --- Initialize host DP matrix ---
    int *M_h = (int *)calloc(total, sizeof(int));
    // Border: gap penalties
    for (int i = 0; i < max_rows; i++) M_h[i * max_cols + 0] = i * gap;
    for (int j = 0; j < max_cols; j++) M_h[0 * max_cols + j] = j * gap;

    // --- Allocate device memory ---
    int *M_d, *sub_d, *seq1_d, *seq2_d;
    cudaMalloc(&M_d,    total * sizeof(int));
    cudaMalloc(&sub_d,  21 * 21 * sizeof(int));
    cudaMalloc(&seq1_d, length * sizeof(int));
    cudaMalloc(&seq2_d, length * sizeof(int));

    cudaMemcpy(M_d,    M_h,    total * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(sub_d,  sub_h,  21 * 21 * sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(seq1_d, seq1_h, length * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(seq2_d, seq2_h, length * sizeof(int),   cudaMemcpyHostToDevice);

    int num_blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

    /*
     * Two-pass block-diagonal wavefront covering all num_blocks x num_blocks tiles.
     *
     * Pass 1 (growing, top-left triangle including main diagonal):
     *   diagonal d = 0 .. num_blocks-1
     *   tiles: bx = 0..d, by = d-bx   (bx+by == d)
     *
     * Pass 2 (shrinking, bottom-right triangle):
     *   diagonal d = num_blocks .. 2*num_blocks-2  (equivalently d' = 1..num_blocks-1)
     *   tiles: bx = d-num_blocks+1 .. num_blocks-1, by = d-bx
     *          i.e. bx+by == d
     *
     * Together these cover every (bx,by) with 0<=bx,by<num_blocks exactly once.
     */

    // Pass 1: diagonals 0..num_blocks-1
    for (int d = 0; d < num_blocks; d++) {
        // tiles on this diagonal: bx=0..d, by=d-bx
        for (int bx = 0; bx <= d; bx++) {
            int by = d - bx;
            nw_tile_kernel<<<1, block_dim>>>(M_d, seq1_d, seq2_d, sub_d, gap, max_cols, bx, by);
        }
        cudaDeviceSynchronize();
    }

    // Pass 2: diagonals num_blocks..2*num_blocks-2
    for (int d = num_blocks; d <= 2 * num_blocks - 2; d++) {
        // tiles: bx = d-num_blocks+1 .. num_blocks-1, by = d-bx
        for (int bx = d - num_blocks + 1; bx < num_blocks; bx++) {
            int by = d - bx;
            nw_tile_kernel<<<1, block_dim>>>(M_d, seq1_d, seq2_d, sub_d, gap, max_cols, bx, by);
        }
        cudaDeviceSynchronize();
    }

    // --- Copy result back ---
    cudaMemcpy(M_h, M_d, total * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Write output to argv[3] ---
    FILE *fp_out = fopen(out_file, "w");
    if (!fp_out) { fprintf(stderr, "Cannot open output: %s\n", out_file); return 1; }
    for (long long i = 0; i < total; i++) {
        fprintf(fp_out, "%lld\t%d\n", i, M_h[i]);
    }
    fclose(fp_out);

    // --- Cleanup ---
    free(M_h); free(sub_h); free(seq1_h); free(seq2_h);
    cudaFree(M_d); cudaFree(sub_d); cudaFree(seq1_d); cudaFree(seq2_d);

    return 0;
}
