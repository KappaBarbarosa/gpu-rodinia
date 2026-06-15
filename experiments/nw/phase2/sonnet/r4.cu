// Needleman-Wunsch CUDA implementation - Phase 2 Round 4
// Block-diagonal wavefront parallelization
// No input files - generates sequences in memory with srand(7)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// BLOSUM62 matrix (24x24)
static const int BLOSUM62[24][24] = {
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

#define BLOCK_SIZE 16

// Kernel for processing one anti-diagonal block
// Each thread computes one cell M[i][j]
__global__ void nw_kernel(int *M, const int *ref, int cols, int penalty,
                          int block_diag, int num_blocks_per_diag) {
    int bx = blockIdx.x;

    // Map block index to (block_row, block_col) on this diagonal
    // block_diag = block_row + block_col (0-indexed, starting at 1,1)
    // So block_row ranges from max(0, block_diag - (num_block_cols-1)) to min(block_diag, num_block_rows-1)
    // Here we just use bx directly as offset from top of diagonal

    int block_row = bx;
    int block_col = block_diag - bx;

    if (block_row < 0 || block_col < 0) return;

    // Compute the actual cell row/col range (1-indexed in M)
    int i_start = block_row * BLOCK_SIZE + 1;
    int j_start = block_col * BLOCK_SIZE + 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = i_start + ty;
    int j = j_start + tx;

    int rows = cols; // square matrix

    if (i >= rows || j >= cols) return;

    // We need to compute M[i][j] = max(M[i-1][j-1]+ref[i*cols+j], M[i-1][j]-penalty, M[i][j-1]-penalty)
    // But within a block, cells depend on each other diagonally
    // Use shared memory with padding to handle intra-block dependencies

    // Shared memory: (BLOCK_SIZE+1) x (BLOCK_SIZE+1) to include borders from prev blocks
    __shared__ int smem[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

    // Load border from global memory
    // Left column border: M[i_start-1+ty][j_start-1] for ty=0..BLOCK_SIZE
    // Top row border: M[i_start-1][j_start-1+tx] for tx=0..BLOCK_SIZE

    if (ty == 0 && tx == 0) {
        smem[0][0] = M[(i_start - 1) * cols + (j_start - 1)];
    }
    if (ty == 0) {
        // Load top border row
        int jj = j_start + tx - 1;
        if (jj >= 0 && jj < cols)
            smem[0][tx + 1] = M[(i_start - 1) * cols + jj + 1 - 1 + 1];
        // smem[0][tx+1] = M[(i_start-1)*cols + (j_start+tx)]
        smem[0][tx + 1] = M[(i_start - 1) * cols + (j_start + tx)];
    }
    if (tx == 0) {
        // Load left border column
        smem[ty + 1][0] = M[(i_start + ty) * cols + (j_start - 1)];
    }

    __syncthreads();

    // Now compute cells diagonally within the block
    // Cell (ty, tx) in shared mem is smem[ty+1][tx+1]
    // It depends on smem[ty][tx], smem[ty][tx+1], smem[ty+1][tx]

    for (int diag = 0; diag < 2 * BLOCK_SIZE - 1; diag++) {
        // Cells on this sub-diagonal: ty + tx == diag
        if (tx + ty == diag) {
            int gi = i_start + ty;
            int gj = j_start + tx;
            if (gi < rows && gj < cols) {
                int score = ref[gi * cols + gj];
                int val = smem[ty][tx] + score;
                int v2 = smem[ty][tx + 1] - penalty;
                int v3 = smem[ty + 1][tx] - penalty;
                if (v2 > val) val = v2;
                if (v3 > val) val = v3;
                smem[ty + 1][tx + 1] = val;
            }
        }
        __syncthreads();
    }

    // Write results back to global memory
    if (i < rows && j < cols) {
        M[i * cols + j] = smem[ty + 1][tx + 1];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <dim> <penalty> <outpath>\n", argv[0]);
        return 1;
    }

    int dim = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char *outpath = argv[3];

    int max_rows = dim + 1;
    int max_cols = dim + 1;
    long long total = (long long)max_rows * max_cols;

    // Allocate host memory
    int *input_seq = (int*)malloc(max_cols * sizeof(int));
    int *ref_h = (int*)malloc(total * sizeof(int));
    int *M_h = (int*)malloc(total * sizeof(int));

    if (!input_seq || !ref_h || !M_h) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Generate input sequence with srand(7), rand()%10+1
    srand(7);
    for (int i = 0; i < dim; i++) {
        input_seq[i] = rand() % 10 + 1;
    }

    // Build reference matrix: ref[i*max_cols+j] = BLOSUM62[input[i-1]][input[j-1]] for i,j>=1
    // Boundaries are not used in ref directly, just fill with 0
    memset(ref_h, 0, total * sizeof(int));
    for (int i = 1; i < max_rows; i++) {
        for (int j = 1; j < max_cols; j++) {
            ref_h[i * max_cols + j] = BLOSUM62[input_seq[i-1]][input_seq[j-1]];
        }
    }

    // Initialize M boundaries
    memset(M_h, 0, total * sizeof(int));
    // M[i][0] = -i*penalty
    for (int i = 0; i < max_rows; i++) {
        M_h[i * max_cols + 0] = -i * penalty;
    }
    // M[0][j] = -j*penalty
    for (int j = 0; j < max_cols; j++) {
        M_h[0 * max_cols + j] = -j * penalty;
    }

    // Allocate device memory
    int *M_d, *ref_d;
    cudaMalloc(&M_d, total * sizeof(int));
    cudaMalloc(&ref_d, total * sizeof(int));

    cudaMemcpy(M_d, M_h, total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ref_d, ref_h, total * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernels in block-diagonal wavefront order
    int num_block_rows = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_block_cols = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_diags = num_block_rows + num_block_cols - 1;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    for (int d = 0; d < total_diags; d++) {
        // Block (br, bc) is on diagonal d if br + bc == d
        // br ranges from 0 to num_block_rows-1
        // bc = d - br ranges from 0 to num_block_cols-1
        int br_min = (d - (num_block_cols - 1) > 0) ? d - (num_block_cols - 1) : 0;
        int br_max = (d < num_block_rows - 1) ? d : num_block_rows - 1;
        int num_blocks = br_max - br_min + 1;

        if (num_blocks <= 0) continue;

        // We'll launch num_blocks blocks, each block index maps to br = br_min + blockIdx.x
        // bc = d - br
        // But kernel uses bx as br directly (needs offset)
        // Let's pass br_min as offset
        // Actually rewrite: launch with blockIdx.x in [0, num_blocks),
        // and compute br = br_min + blockIdx.x, bc = d - br

        // For simplicity, pack into kernel args via a different approach:
        // Pass d and br_min, kernel computes br = br_min + blockIdx.x
        // Modify kernel call to use br_min offset

        // We need to pass br_min to kernel - but original kernel uses bx as br directly
        // Let's just offset: pass (d, br_min) and let kernel do br = br_min + bx, bc = d - br

        // Since we can't easily change kernel signature mid-loop without modifying,
        // let's use a corrected kernel call:

        // Use a lambda-style approach: re-examine what kernel expects
        // kernel(M, ref, cols, penalty, block_diag, num_blocks_per_diag)
        // In kernel: block_row = bx (blockIdx.x), block_col = block_diag - bx
        // This only works if bx = br directly, i.e. bx starts at 0 and we want br = br_min + bx
        // So we need to fix the kernel or pass br_min

        // Simplest fix: pass br_min as the "block_diag" trick won't work cleanly
        // Let's just pass adjusted values and fix the kernel:
        // Actually we already have the kernel above - let's use a corrected version
        // We'll call with a modified grid where blockIdx.x directly gives the correct br
        // by passing d and br_min via a second kernel or by adjusting

        // The cleanest solution: use the kernel with bx = br (absolute),
        // so launch with num_block_rows blocks but only valid ones will execute
        // That wastes some launches but is simple

        // Actually: launch br_min..br_max as contiguous blocks
        // kernel sees bx=0..num_blocks-1, needs to know br_min
        // => pass br_min as extra param

        // We'll use a slightly different approach: always launch with br as the absolute index
        // and skip invalid ones. Launch num_block_rows blocks, kernel checks validity.
        // For large dim this is still O(num_block_rows) per diagonal = O(dim/BLOCK_SIZE)
        // Total kernel launches = O(dim/BLOCK_SIZE) with O(dim/BLOCK_SIZE) blocks each = O((dim/BLOCK_SIZE)^2) total work = O(dim^2/BLOCK_SIZE^2)
        // That's fine.

        // For now, launch num_blocks blocks with offset encoded:
        // We need to fix this properly. Let me use a wrapper.
        // The simplest: just launch d+1 blocks and let kernel skip out-of-range ones.

        int launch_blocks = d + 1; // br goes 0..d, kernel checks bc validity
        if (launch_blocks > num_block_rows) launch_blocks = num_block_rows;
        // Also bc = d - br must be in [0, num_block_cols-1]
        // kernel already returns if block_row < 0 || block_col < 0
        // We also need block_col < num_block_cols -- kernel will check i/j bounds

        nw_kernel<<<launch_blocks, block>>>(M_d, ref_d, max_cols, penalty, d, num_blocks);
        cudaDeviceSynchronize();
    }

    // Copy result back
    cudaMemcpy(M_h, M_d, total * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output
    FILE *fp = fopen(outpath, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open output file: %s\n", outpath);
        return 1;
    }

    for (long long idx = 0; idx < total; idx++) {
        fprintf(fp, "%lld\t%d\n", idx, M_h[idx]);
    }
    fclose(fp);

    // Cleanup
    free(input_seq);
    free(ref_h);
    free(M_h);
    cudaFree(M_d);
    cudaFree(ref_d);

    return 0;
}
