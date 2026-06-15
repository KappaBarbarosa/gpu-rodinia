/*
 * Needleman-Wunsch — serial reference implementation.
 *
 * Global pairwise sequence alignment via dynamic programming. Builds a
 * (dim+1) x (dim+1) score matrix. The two "sequences" and the substitution
 * (reference) scores are generated deterministically from srand(7), exactly as
 * in the Rodinia benchmark, so no external data file is needed.
 *
 * DP recurrence (for i,j >= 1):
 *   M[i][j] = max( M[i-1][j-1] + ref[i][j],     // diagonal: match/mismatch
 *                  M[i][j-1]   - penalty,        // left: gap
 *                  M[i-1][j]   - penalty )       // up:   gap
 * Boundary: M[i][0] = -i*penalty, M[0][j] = -j*penalty, M[0][0] = 0.
 *
 * Dependency note: M[i][j] needs its up, left, and diagonal neighbours, so the
 * full matrix is NOT embarrassingly parallel. But all cells on one anti-diagonal
 * (i+j = const) depend only on earlier anti-diagonals, hence are mutually
 * independent — this is the wavefront parallelism a GPU port exploits.
 *
 * CLI:  nw_serial <dimension> <penalty> <output_file>
 *   dimension : sequence length (should be a multiple of 16); matrix is
 *               (dimension+1) x (dimension+1).
 *   penalty   : positive integer gap penalty.
 *   output    : full score matrix, one line per cell "<index>\t<value>\n",
 *               row-major over the (dimension+1)^2 cells.
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static int maximum(int a, int b, int c) {
    int k = (a <= b) ? b : a;
    return (k <= c) ? c : k;
}

/* BLOSUM62 substitution matrix (24x24), as in the Rodinia benchmark. */
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

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <dimension> <penalty> <output_file>\n", argv[0]);
        return 1;
    }
    int dim = atoi(argv[1]);
    int penalty = atoi(argv[2]);
    const char *ofile = argv[3];
    if (dim <= 0 || penalty <= 0) {
        fprintf(stderr, "invalid arguments\n");
        return 1;
    }

    int max_rows = dim + 1;
    int max_cols = dim + 1;

    int *referrence     = (int *)malloc(max_rows * max_cols * sizeof(int));
    int *input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
    if (!referrence || !input_itemsets) { fprintf(stderr, "alloc failed\n"); return 1; }

    srand(7);

    for (int i = 0; i < max_cols; i++)
        for (int j = 0; j < max_rows; j++)
            input_itemsets[i * max_cols + j] = 0;

    /* Generate the two sequences (same rand() call order as the reference). */
    for (int i = 1; i < max_rows; i++)
        input_itemsets[i * max_cols] = rand() % 10 + 1;
    for (int j = 1; j < max_cols; j++)
        input_itemsets[j] = rand() % 10 + 1;

    /* Substitution scores from BLOSUM62 indexed by the sequence symbols. */
    for (int i = 1; i < max_cols; i++)
        for (int j = 1; j < max_rows; j++)
            referrence[i * max_cols + j] =
                blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];

    /* DP boundary conditions overwrite the first row/column. */
    for (int i = 1; i < max_rows; i++)
        input_itemsets[i * max_cols] = -i * penalty;
    for (int j = 1; j < max_cols; j++)
        input_itemsets[j] = -j * penalty;

    /* Fill the score matrix (serial; each cell uses up/left/diagonal). */
    double t0 = now_seconds();
    for (int i = 1; i < max_rows; i++) {
        for (int j = 1; j < max_cols; j++) {
            input_itemsets[i * max_cols + j] = maximum(
                input_itemsets[(i - 1) * max_cols + (j - 1)] + referrence[i * max_cols + j],
                input_itemsets[i * max_cols + (j - 1)] - penalty,
                input_itemsets[(i - 1) * max_cols + j] - penalty);
        }
    }
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    /* Write the full score matrix, row-major. */
    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    int n = max_rows * max_cols;
    for (int idx = 0; idx < n; idx++)
        fprintf(fp, "%d\t%d\n", idx, input_itemsets[idx]);
    fclose(fp);

    free(referrence);
    free(input_itemsets);
    return 0;
}
