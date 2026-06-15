/*
 * PathFinder — serial reference implementation.
 *
 * Single-source minimum-cost path through a 2D grid of weights ("wall"),
 * solved by dynamic programming. Row 0 is the starting row; for each
 * subsequent row, each cell's accumulated cost is its own weight plus the
 * minimum accumulated cost among the up-to-three cells directly above it
 * (above-left, above, above-right). The final answer is the accumulated cost
 * row after processing the last row.
 *
 * The wall weights are generated deterministically from srand(9), exactly as
 * in the Rodinia benchmark, so no external data file is needed. You MUST
 * reproduce the generation order exactly (i over rows outer, j over cols
 * inner; wall[i][j] = rand() % 10) or the output will differ.
 *
 * DP recurrence (for step t = 0 .. rows-2, over column n = 0 .. cols-1):
 *   m = src[n];
 *   if (n > 0)        m = min(m, src[n-1]);
 *   if (n < cols-1)   m = min(m, src[n+1]);
 *   dst[n] = wall[t+1][n] + m;
 * with src = accumulated cost of the previous row, ping-ponged into dst.
 * The initial src row is wall[0][*].
 *
 * Dependency note: within one DP step every column cell depends only on the
 * previous (already complete) row, so all `cols` cells of a row are mutually
 * independent — data-parallel over columns. The `rows` steps are sequential.
 *
 * CLI:  pathfinder_serial <cols> <rows> <output_file>
 *   cols   : grid width (number of columns / parallel work width).
 *   rows   : number of DP steps (grid height / sequential depth).
 *   output : the FINAL accumulated-cost row, one line per cell
 *            "<index>\t<value>\n", index 0 .. cols-1.
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define M_SEED 9
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <cols> <rows> <output_file>\n", argv[0]);
        return 1;
    }
    int cols = atoi(argv[1]);
    int rows = atoi(argv[2]);
    const char *ofile = argv[3];
    if (cols <= 0 || rows <= 0) {
        fprintf(stderr, "invalid arguments\n");
        return 1;
    }

    /* wall stored as a flat rows*cols array, wall[i][j] = wall_flat[i*cols+j]. */
    int *wall = (int *)malloc((size_t)rows * cols * sizeof(int));
    int *src  = (int *)malloc((size_t)cols * sizeof(int));
    int *dst  = (int *)malloc((size_t)cols * sizeof(int));
    if (!wall || !src || !dst) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* Deterministic wall generation — order is normative:
       i over rows (outer), j over cols (inner). */
    srand(M_SEED);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            wall[(size_t)i * cols + j] = rand() % 10;

    /* Initial accumulated-cost row = wall row 0. */
    for (int j = 0; j < cols; j++)
        dst[j] = wall[j];

    /* DP — time ONLY this loop. */
    double t0 = now_seconds();
    for (int t = 0; t < rows - 1; t++) {
        int *tmp = src; src = dst; dst = tmp;   /* ping-pong: src = prev row */
        const int *wnext = wall + (size_t)(t + 1) * cols;
        for (int n = 0; n < cols; n++) {
            int m = src[n];
            if (n > 0)        m = MIN(m, src[n - 1]);
            if (n < cols - 1) m = MIN(m, src[n + 1]);
            dst[n] = wnext[n] + m;
        }
    }
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    /* dst now holds the final accumulated-cost row. */
    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int n = 0; n < cols; n++)
        fprintf(fp, "%d\t%d\n", n, dst[n]);
    fclose(fp);

    free(wall);
    free(src);
    free(dst);
    return 0;
}
