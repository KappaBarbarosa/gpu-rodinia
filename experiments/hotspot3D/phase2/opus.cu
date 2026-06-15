/*
 * hotspot3D — Phase 2 optimised CUDA kernel.
 *
 * Optimisations over phase 1 (global-memory only, 7.81 ms / 100 iters):
 *
 *  1. Z-sliding window with register-pipelined Z-neighbours.
 *     Each block sweeps all nz Z-layers in a single launch, keeping one
 *     XY smem tile active at all times.  The Z-bottom and Z-top neighbours
 *     for each cell are kept in scalar registers (vb_reg / vt_reg) and
 *     advanced as the loop walks from z=0 to z=nz-1.  This guarantees:
 *       - Every global cell is loaded from DRAM at most twice (once as cur,
 *         once as power), instead of being re-fetched by up to 7 stencil
 *         neighbours across multiple blocks.
 *       - No smem is wasted storing the two Z-adjacent planes; shared memory
 *         shrinks from 3*(BX+2)*(BY+2)*4 to 3*(BX+2)*4 bytes, leaving
 *         more room for concurrent blocks.
 *
 *  2. Block shape 256×1 (8 warps).
 *     - 256 threads load exactly one cache line per 4-float group, giving
 *       perfect X-coalescing for all global reads and writes.
 *     - smem = 3 rows × 258 floats × 4 B = 3,096 B per block.
 *       49152 / 3096 = 15 blocks per SM (limited to 6 by the 1536-thread cap
 *       on SM 8.6), yielding 48 warps/SM = maximum occupancy.
 *     - Grid = (nx/256) × ny — for nx=ny=512 this is 2×512 = 1,024 blocks,
 *       enough waves to keep all 46 SMs busy continuously.
 *
 *  3. Stencil coefficients in __constant__ memory: broadcast from constant
 *     cache, zero register cost, zero L1/L2 traffic.
 *
 *  4. __restrict__ on all pointer arguments: allows the compiler to emit
 *     independent (potentially overlapping) load instructions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/* ---- physical constants ---- */
#define MAX_PD        3.0e6f
#define PRECISION     0.001f
#define SPEC_HEAT_SI  1.75e6f
#define K_SI          100.0f
#define FACTOR_CHIP   0.5f

static const float t_chip      = 0.0005f;
static const float chip_height = 0.016f;
static const float chip_width  = 0.016f;
static const float amb_temp    = 80.0f;

/* ---- stencil coefficients in constant memory ---- */
__constant__ float d_cc, d_cn, d_cs, d_ce, d_cw, d_ct, d_cb, d_dtCap, d_ambct;

/* ---- tile dimensions ---- */
#define BX  256        /* threads in X per block — one full smem row of 256 */
#define SX  (BX + 2)   /* 258: interior + 1-cell west/east halo             */
/* BY = 1 always: one Y row per block; Y-neighbours loaded into smem halo   */
#define SY  3          /* rows in smem: north-halo, cur, south-halo          */

/*
 * stencil_kernel_smem
 *
 * Grid:   (nx / BX) × ny × 1   — each block handles one row of X for one Y.
 * Block:  BX × 1 × 1
 *
 * For each Z layer the kernel:
 *   1. Loads the current XY row (plus halo) into smem.
 *   2. Uses register-held vb_reg / vt_reg for the Z-bottom / Z-top values.
 *   3. Computes the full 7-point stencil and writes to nxt.
 *   4. Advances the register pipeline: vb_reg ← old vc; vt_reg ← cur[z+2].
 *
 * Boundary condition: clamp-to-edge (mirror / no-flux), identical to serial.
 */
__global__ void stencil_kernel_smem(
        const float * __restrict__ cur,
        float       * __restrict__ nxt,
        const float * __restrict__ power,
        int nx, int ny, int nz)
{
    /* Shared memory: north-halo row [0], current row [1], south-halo row [2] */
    __shared__ float smem[SY][SX];

    const int tx = threadIdx.x;             /* 0 .. BX-1  */
    const int gx = blockIdx.x * BX + tx;   /* global X   */
    const int gy = blockIdx.y;              /* global Y (blockDim.y == 1) */
    const int nxny = nx * ny;
    const bool valid = (gx < nx) && (gy < ny);

    /*
     * load_row — cooperative load of one Z-slice row into smem.
     * Interior: smem[1][tx+1] from src[gy*nx + gx].
     * West/East: one extra element loaded by the edge threads.
     * North/South halo rows (smem[0], smem[2]): loaded by all threads.
     * Corners: loaded by the two corner threads (tx==0, tx==BX-1).
     */
#define LOAD_ROW(src_ptr)  do {                                                 \
    const float *__src = (src_ptr);                                             \
    /* Interior */                                                               \
    if (valid) smem[1][tx + 1] = __src[gy * nx + gx];                         \
    /* West halo */                                                              \
    if (tx == 0 && gy < ny) {                                                   \
        int ghx = (gx > 0) ? gx - 1 : 0;                                       \
        smem[1][0] = __src[gy * nx + ghx];                                     \
    }                                                                            \
    /* East halo */                                                              \
    if (tx == BX - 1 && gy < ny) {                                              \
        int ghx = (gx < nx - 1) ? gx + 1 : nx - 1;                            \
        smem[1][SX - 1] = __src[gy * nx + ghx];                               \
    }                                                                            \
    /* North halo row */                                                         \
    if (gx < nx) {                                                               \
        int ghy = (gy > 0) ? gy - 1 : 0;                                       \
        smem[0][tx + 1] = __src[ghy * nx + gx];                               \
    }                                                                            \
    /* South halo row */                                                         \
    if (gx < nx) {                                                               \
        int ghy = (gy < ny - 1) ? gy + 1 : ny - 1;                            \
        smem[2][tx + 1] = __src[ghy * nx + gx];                               \
    }                                                                            \
    /* Northwest corner */                                                       \
    if (tx == 0) {                                                               \
        int ghx = (gx > 0)     ? gx - 1 : 0;                                   \
        int ghn = (gy > 0)     ? gy - 1 : 0;                                   \
        int ghs = (gy < ny-1)  ? gy + 1 : ny - 1;                             \
        smem[0][0] = __src[ghn * nx + ghx];                                    \
        smem[2][0] = __src[ghs * nx + ghx];                                    \
    }                                                                            \
    /* Northeast corner */                                                       \
    if (tx == BX - 1) {                                                          \
        int ghx = (gx < nx-1)  ? gx + 1 : nx - 1;                             \
        int ghn = (gy > 0)     ? gy - 1 : 0;                                   \
        int ghs = (gy < ny-1)  ? gy + 1 : ny - 1;                             \
        smem[0][SX - 1] = __src[ghn * nx + ghx];                              \
        smem[2][SX - 1] = __src[ghs * nx + ghx];                              \
    }                                                                            \
} while(0)

    /* ---- Prime the register pipeline ---- */
    /* vb_reg: the Z-bottom neighbour for z=0 is clamped to cur[z=0], so prime to 0; */
    /* we use a flag (z==0) to select vc instead of vb_reg.                          */
    /* vt_reg: the Z-top neighbour for z=0 is cur[z=1] (or clamped cur[0] if nz==1) */
    float vb_reg = 0.0f;   /* unused at z=0 (boundary clamp) */
    float vt_reg = 0.0f;
    if (valid) {
        vt_reg = (nz > 1) ? cur[1 * nxny + gy * nx + gx]
                           : cur[0 * nxny + gy * nx + gx];
    }

    /* ---- Main Z loop ---- */
    for (int z = 0; z < nz; z++) {
        /* Load current Z plane into smem */
        LOAD_ROW(cur + z * nxny);
        __syncthreads();

        if (valid) {
            const float vc = smem[1][tx + 1];
            const float vn = smem[0][tx + 1];   /* north = y-1 */
            const float vs = smem[2][tx + 1];   /* south = y+1 */
            const float vw = smem[1][tx    ];   /* west  = x-1 */
            const float ve = smem[1][tx + 2];   /* east  = x+1 */
            const float vb = (z == 0)      ? vc : vb_reg;  /* bottom clamp */
            const float vt = (z == nz - 1) ? vc : vt_reg;  /* top    clamp */
            const float pw = power[z * nxny + gy * nx + gx];

            nxt[z * nxny + gy * nx + gx] =
                vc * d_cc +
                vn * d_cn + vs * d_cs +
                ve * d_ce + vw * d_cw +
                vt * d_ct + vb * d_cb +
                pw * d_dtCap + d_ambct;

            /* Advance register pipeline */
            vb_reg = vc;                /* current centre becomes next iteration's bottom */
            /* vt_reg for next z is cur[z+2]; load it now while smem is free */
            if (z + 2 < nz)
                vt_reg = cur[(z + 2) * nxny + gy * nx + gx];
            else if (z + 1 < nz)
                vt_reg = cur[(z + 1) * nxny + gy * nx + gx];
            /* at z == nz-1: vt_reg is unused (boundary clamp to vc) */
        }
        __syncthreads();
    }

#undef LOAD_ROW
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <output_file>\n",
                argv[0]);
        return 1;
    }

    int nx    = atoi(argv[1]);
    int ny    = nx;
    int nz    = atoi(argv[2]);
    int niter = atoi(argv[3]);
    const char *outfile = argv[4];

    float dx  = chip_height / ny;
    float dy  = chip_width  / nx;
    float dz  = t_chip      / nz;
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx  = dy / (2.0f * K_SI * t_chip * dx);
    float Ry  = dx / (2.0f * K_SI * t_chip * dy);
    float Rz  = dz / (K_SI * dx * dy);
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt  = PRECISION / max_slope;

    float stepDivCap = dt / Cap;
    float ce = stepDivCap / Rx;
    float cw = ce;
    float cn = stepDivCap / Ry;
    float cs = cn;
    float ct = stepDivCap / Rz;
    float cb = ct;
    float cc = 1.0f - (2.0f*ce + 2.0f*cn + 3.0f*ct);
    float dtCap = dt / Cap;
    float ambct = ct * amb_temp;

    /* Upload coefficients to constant memory */
    cudaMemcpyToSymbol(d_cc,    &cc,    sizeof(float));
    cudaMemcpyToSymbol(d_cn,    &cn,    sizeof(float));
    cudaMemcpyToSymbol(d_cs,    &cs,    sizeof(float));
    cudaMemcpyToSymbol(d_ce,    &ce,    sizeof(float));
    cudaMemcpyToSymbol(d_cw,    &cw,    sizeof(float));
    cudaMemcpyToSymbol(d_ct,    &ct,    sizeof(float));
    cudaMemcpyToSymbol(d_cb,    &cb,    sizeof(float));
    cudaMemcpyToSymbol(d_dtCap, &dtCap, sizeof(float));
    cudaMemcpyToSymbol(d_ambct, &ambct, sizeof(float));

    int size = nx * ny * nz;
    float *power = (float *)malloc(size * sizeof(float));
    float *buf0  = (float *)malloc(size * sizeof(float));
    if (!power || !buf0) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* Deterministic input generation — must match serial reference */
    srand(7);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                power[idx] = (rand() / (float)RAND_MAX) * 15.0f;
                buf0[idx]  = amb_temp + (rand() / (float)RAND_MAX) * 5.0f;
            }

    float *d_power, *d_a, *d_b;
    cudaMalloc(&d_power, size * sizeof(float));
    cudaMalloc(&d_a,     size * sizeof(float));
    cudaMalloc(&d_b,     size * sizeof(float));
    cudaMemcpy(d_power, power, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a,     buf0,  size * sizeof(float), cudaMemcpyHostToDevice);

    /*
     * Grid: one block per (BX-wide X strip, Y row).
     * Each block sweeps all nz Z-layers internally.
     * For non-multiples of BX: ceiling division ensures full coverage.
     */
    dim3 block(BX, 1, 1);
    dim3 grid((nx + BX - 1) / BX, ny, 1);

    float *d_cur = d_a, *d_nxt = d_b;

    cudaDeviceSynchronize();
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int iter = 0; iter < niter; iter++) {
        stencil_kernel_smem<<<grid, block>>>(d_cur, d_nxt, d_power, nx, ny, nz);
        float *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
    }

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    fprintf(stderr, "compute_seconds: %.6f\n", elapsed);

    cudaMemcpy(buf0, d_cur, size * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(outfile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", outfile); return 1; }
    int index = 0;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                fprintf(fp, "%d\t%g\n", index++, buf0[idx]);
            }
    fclose(fp);

    cudaFree(d_power); cudaFree(d_a); cudaFree(d_b);
    free(power); free(buf0);
    return 0;
}
