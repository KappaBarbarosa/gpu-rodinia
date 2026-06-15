/*
 * hotspot3D_serial.c — serial reference for AI-optimizes-Rodinia experiment.
 * Derived from openmp/hotspot3D/3D.c (OMP pragmas stripped; input generated
 * deterministically from srand(7) so no external data files are needed).
 *
 * CLI: <prog> <rows/cols> <layers> <iterations> <output_file>
 *   rows/cols  : grid dimension (square: nx = ny = rows/cols)
 *   layers     : z-dimension (nz)
 *   iterations : number of time steps
 *   output_file: path for the output temperature file
 *
 * Output format: one line per cell, "<index>\t<value>\n" (%g format),
 * in loop order (i=row outer, j=col middle, k=layer inner), matching
 * writeoutput() in the original Rodinia code.
 *
 * Compute time (excluding I/O) is printed to stderr as:
 *   compute_seconds: <float>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define MAX_PD        3.0e6f
#define PRECISION     0.001f
#define SPEC_HEAT_SI  1.75e6f
#define K_SI          100.0f
#define FACTOR_CHIP   0.5f

static const float t_chip      = 0.0005f;
static const float chip_height = 0.016f;
static const float chip_width  = 0.016f;
static const float amb_temp    = 80.0f;

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <output_file>\n",
                argv[0]);
        return 1;
    }

    int nx    = atoi(argv[1]);   /* cols = rows (square grid) */
    int ny    = nx;
    int nz    = atoi(argv[2]);   /* layers */
    int niter = atoi(argv[3]);
    const char *outfile = argv[4];

    /* --- physical parameters (identical to original Rodinia) --- */
    float dx  = chip_height / ny;
    float dy  = chip_width  / nx;
    float dz  = t_chip      / nz;
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx  = dy / (2.0f * K_SI * t_chip * dx);
    float Ry  = dx / (2.0f * K_SI * t_chip * dy);
    float Rz  = dz / (K_SI * dx * dy);
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt  = PRECISION / max_slope;

    /* --- derived stencil coefficients --- */
    float stepDivCap = dt / Cap;
    float ce = stepDivCap / Rx;
    float cw = ce;
    float cn = stepDivCap / Ry;
    float cs = cn;
    float ct = stepDivCap / Rz;
    float cb = ct;
    float cc = 1.0f - (2.0f*ce + 2.0f*cn + 3.0f*ct);
    float dtCap = dt / Cap;

    int size = nx * ny * nz;
    float *power = (float *)malloc(size * sizeof(float));
    float *buf0  = (float *)malloc(size * sizeof(float));
    float *buf1  = (float *)malloc(size * sizeof(float));
    if (!power || !buf0 || !buf1) { fprintf(stderr, "malloc failed\n"); return 1; }

    /*
     * Generate input deterministically (srand(7)).
     * Loop order matches readinput() in original: i=row outer, j=col middle, k=layer inner.
     * Storage index: idx = j + i*nx + k*nx*ny  (x + y*nx + z*nx*ny).
     * Two rand() calls per cell: first for power, second for initial temperature.
     *
     * Power range: 0–15 W per cell (representative of real chip power maps;
     * MAX_PD=3e6 is the physical ceiling but data files use far smaller values —
     * with dt/Cap ≈ 0.34 K/step/W, 15 W gives ~5 K/step, reaching ~580 K after
     * 100 iterations, which matches the original Rodinia output range of ~300–400 °C).
     * Temperature: ambient ± 5 °C (small spatial variation; stencil is stable
     * with ce ≈ 0.034, so small initial gradients are safe).
     */
    srand(7);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                power[idx] = (rand() / (float)RAND_MAX) * 15.0f;
                buf0[idx]  = amb_temp + (rand() / (float)RAND_MAX) * 5.0f;
            }

    /*
     * Time-stepping: ping-pong between buf0 (cur) and buf1 (nxt).
     * After all iterations, `cur` holds the final temperature field.
     */
    float *cur = buf0, *nxt = buf1;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int iter = 0; iter < niter; iter++) {
        for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
                for (int x = 0; x < nx; x++) {
                    int c = x + y*nx + z*nx*ny;
                    int w = (x == 0)      ? c : c - 1;
                    int e = (x == nx-1)   ? c : c + 1;
                    int n = (y == 0)      ? c : c - nx;
                    int s = (y == ny-1)   ? c : c + nx;
                    int b = (z == 0)      ? c : c - nx*ny;
                    int t = (z == nz-1)   ? c : c + nx*ny;
                    nxt[c] = cur[c]*cc + cur[n]*cn + cur[s]*cs
                           + cur[e]*ce + cur[w]*cw
                           + cur[t]*ct + cur[b]*cb
                           + dtCap * power[c] + ct * amb_temp;
                }
        float *tmp = cur; cur = nxt; nxt = tmp;
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    fprintf(stderr, "compute_seconds: %.6f\n", elapsed);

    /* Write output: same loop order and format as writeoutput() in original */
    FILE *fp = fopen(outfile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", outfile); return 1; }
    int index = 0;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                fprintf(fp, "%d\t%g\n", index++, cur[idx]);
            }
    fclose(fp);

    free(power); free(buf0); free(buf1);
    return 0;
}
