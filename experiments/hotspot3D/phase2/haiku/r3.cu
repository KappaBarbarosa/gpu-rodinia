/*
 * hotspot3D_cuda.cu — CUDA port of hotspot3D thermal simulation.
 * Phase 2 r3: BY=2, power[] shared-memory cache, __launch_bounds__.
 *
 * Changes over r2 (BY=1, smem=[8][3][34], ~65 µs/iter):
 *
 *   1. Increase BY from 1 to 2.
 *      Block: (BX=32, BY=2, BZ=8) → 512 threads/block.
 *      Occupancy: 3 blocks/SM × 512 = 1536 = 100% (same as r2's 6×256=1536).
 *      Y-halo overhead: (BY+2)/BY = 3.0x (BY=1) → 2.0x (BY=2).
 *      Each halo row is shared between 2 consecutive output rows instead of 1.
 *      Reduces Y-direction global memory traffic by ~33%.
 *      smem_cur: [BZ][BY+2][BX+2] = [8][4][34] = 1088 floats = 4352 bytes.
 *
 *   2. Cache power[] in shared memory (smem_pwr[BZ][BY][BX]).
 *      power[] is read-only, constant across all iterations, and requires no halos.
 *      Loading it into smem before __syncthreads() eliminates global-memory reads
 *      for power during the stencil compute step.
 *      smem_pwr: [BZ][BY][BX] = [8][2][32] = 512 floats = 2048 bytes.
 *      Total smem per block: 4352 + 2048 = 6400 bytes.
 *      Blocks/SM by smem: 102400/6400 = 16; by threads: 1536/512 = 3 (binding).
 *      Occupancy unchanged at 100% (3 × 512 = 1536 = max threads/SM).
 *
 *   3. __launch_bounds__(BX*BY*BZ, 3): informs the compiler that at most
 *      3 blocks/SM are resident, allowing tighter register allocation and
 *      better instruction scheduling without changing computation.
 *
 *   Stencil formula: IDENTICAL to r2 (preserves bit-exact floating-point result).
 *   Separate c_cn, c_cs, c_ce, c_cw constants retained (changing to merged c_cxy
 *   would alter the FP evaluation order and break bit-exact correctness vs serial).
 *
 *   All other r2 optimisations retained:
 *   - BX=32 (one warp in x): coalesced global loads.
 *   - BZ=8 = nz: one block covers all z-layers → gridDim.z = 1.
 *   - No z-halo rows; z-boundary via clamped smem indices (tzB, tzT).
 *   - __constant__ stencil coefficients (constant-cache broadcast).
 *   - __restrict__ on all pointers (ld.global.nc for non-coherent loads).
 *   - Branch-free x/y boundary via min/max clamping.
 *   - c_ambct = c_ct * amb_temp pre-baked in constant memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define MAX_PD        3.0e6f
#define PRECISION     0.001f
#define SPEC_HEAT_SI  1.75e6f
#define K_SI          100.0f
#define FACTOR_CHIP   0.5f

static const float t_chip      = 0.0005f;
static const float chip_height = 0.016f;
static const float chip_width  = 0.016f;
static const float amb_temp    = 80.0f;

/*
 * Block tile dimensions.
 * BX=32 (one warp in x): perfectly coalesced global loads.
 * BY=2 : two y-rows per block; halves Y-halo redundancy vs BY=1,
 *         while preserving 100% occupancy (3 × 512 = 1536 = max threads/SM).
 * BZ=8 (= nz): one block covers all z-layers → gridDim.z = 1.
 */
#define BX 32
#define BY  2
#define BZ  8

/*
 * Constant memory: stencil coefficients + pre-baked ambient contribution.
 * Separate c_cn/c_cs and c_ce/c_cw are retained (even though cn==cs and ce==cw
 * when nx==ny) to preserve the exact floating-point evaluation order of r2.
 */
__constant__ float c_cc, c_cn, c_cs, c_ce, c_cw, c_ct, c_cb, c_dtCap, c_ambct;

/*
 * stencil_kernel — 3-D shared-memory tiled 7-point stencil.
 *
 * Shared memory:
 *   smem_cur[BZ][BY+2][BX+2] — temperature tile with x/y halos
 *   smem_pwr[BZ][BY  ][BX  ] — power tile (no halos; same (x,y,z) as center)
 *
 * Z-boundary: no z-halo rows; handled by clamped smem indices tzB/tzT.
 * Stencil formula is bit-identical to r2 (same FP evaluation order).
 */
__global__
__launch_bounds__(BX * BY * BZ, 3)
void stencil_kernel(
    const float * __restrict__ cur,
    float       * __restrict__ nxt,
    const float * __restrict__ power,
    int nx, int ny, int nz)
{
    __shared__ float smem_cur[BZ][BY + 2][BX + 2];
    __shared__ float smem_pwr[BZ][BY][BX];

    const int tx = threadIdx.x;   /* 0..BX-1 */
    const int ty = threadIdx.y;   /* 0..BY-1 */
    const int tz = threadIdx.z;   /* 0..BZ-1 */

    const int x = blockIdx.x * BX + tx;
    const int y = blockIdx.y * BY + ty;
    const int z = tz;   /* gridDim.z=1, blockIdx.z=0 → z == tz */

    /* Clamped coordinates for x/y boundary conditions */
    const int xc = min(x, nx - 1);
    const int yc = min(y, ny - 1);
    const int xL = max(x - 1, 0);
    const int xR = min(x + 1, nx - 1);
    const int yU = max(y - 1, 0);
    const int yD = min(y + 1, ny - 1);

    const int plane = nx * ny;
    const int ctr_idx = xc + yc * nx + z * plane;

    /* --- Load center tile + power into smem --- */
    smem_cur[tz][ty + 1][tx + 1] = cur[ctr_idx];
    smem_pwr[tz][ty][tx]         = power[ctr_idx];

    /* X halos: loaded by boundary threads in x */
    if (tx == 0)
        smem_cur[tz][ty + 1][0]      = cur[xL + yc * nx + z * plane];
    if (tx == BX - 1)
        smem_cur[tz][ty + 1][BX + 1] = cur[xR + yc * nx + z * plane];

    /* Y halos: loaded by boundary threads in y */
    if (ty == 0)
        smem_cur[tz][0][tx + 1]      = cur[xc + yU * nx + z * plane];
    if (ty == BY - 1)
        smem_cur[tz][BY + 1][tx + 1] = cur[xc + yD * nx + z * plane];

    __syncthreads();

    /* Skip out-of-bounds threads (for non-divisible grid sizes) */
    if (x >= nx || y >= ny) return;

    /* Clamped z smem indices for no-flux z-boundary:
     *   z==0:    tzB=0    → smem_cur[0]    == center (self-clamped)
     *   z==BZ-1: tzT=BZ-1 → smem_cur[BZ-1] == center (self-clamped) */
    const int tzB = max(tz - 1, 0);
    const int tzT = min(tz + 1, BZ - 1);

    const int c = x + y * nx + z * plane;

    /* Stencil: SAME formula as r2 (preserves FP evaluation order for bit-exact output) */
    nxt[c] = smem_cur[tz ][ty + 1][tx + 1] * c_cc
           + smem_cur[tz ][ty    ][tx + 1] * c_cn   /* north (y-1) */
           + smem_cur[tz ][ty + 2][tx + 1] * c_cs   /* south (y+1) */
           + smem_cur[tz ][ty + 1][tx + 2] * c_ce   /* east  (x+1) */
           + smem_cur[tz ][ty + 1][tx    ] * c_cw   /* west  (x-1) */
           + smem_cur[tzT][ty + 1][tx + 1] * c_ct   /* above (z+1) */
           + smem_cur[tzB][ty + 1][tx + 1] * c_cb   /* below (z-1) */
           + c_dtCap * smem_pwr[tz][ty][tx] + c_ambct;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <output_file>\n",
                argv[0]);
        return 1;
    }

    int nx    = atoi(argv[1]);
    int ny    = nx;           /* always square */
    int nz    = atoi(argv[2]);
    int niter = atoi(argv[3]);
    const char *outfile = argv[4];

    /* --- physical parameters --- */
    float dx  = chip_height / ny;
    float dy  = chip_width  / nx;
    float dz  = t_chip      / nz;
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx  = dy / (2.0f * K_SI * t_chip * dx);
    float Ry  = dx / (2.0f * K_SI * t_chip * dy);
    float Rz  = dz / (K_SI * dx * dy);
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt  = PRECISION / max_slope;

    /* --- derived stencil coefficients (same as r2) --- */
    float stepDivCap = dt / Cap;
    float h_ce = stepDivCap / Rx;
    float h_cw = h_ce;
    float h_cn = stepDivCap / Ry;
    float h_cs = h_cn;
    float h_ct = stepDivCap / Rz;
    float h_cb = h_ct;
    float h_cc = 1.0f - (2.0f*h_ce + 2.0f*h_cn + 3.0f*h_ct);
    float h_dtCap = dt / Cap;
    float h_ambct = h_ct * amb_temp;   /* pre-baked ambient contribution */

    cudaMemcpyToSymbol(c_cc,    &h_cc,    sizeof(float));
    cudaMemcpyToSymbol(c_cn,    &h_cn,    sizeof(float));
    cudaMemcpyToSymbol(c_cs,    &h_cs,    sizeof(float));
    cudaMemcpyToSymbol(c_ce,    &h_ce,    sizeof(float));
    cudaMemcpyToSymbol(c_cw,    &h_cw,    sizeof(float));
    cudaMemcpyToSymbol(c_ct,    &h_ct,    sizeof(float));
    cudaMemcpyToSymbol(c_cb,    &h_cb,    sizeof(float));
    cudaMemcpyToSymbol(c_dtCap, &h_dtCap, sizeof(float));
    cudaMemcpyToSymbol(c_ambct, &h_ambct, sizeof(float));

    int size = nx * ny * nz;

    /* --- host allocation and input generation --- */
    float *h_power = (float *)malloc(size * sizeof(float));
    float *h_buf0  = (float *)malloc(size * sizeof(float));
    if (!h_power || !h_buf0) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* Deterministic input: srand(7) — must match serial golden */
    srand(7);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                h_power[idx] = (rand() / (float)RAND_MAX) * 15.0f;
                h_buf0[idx]  = amb_temp + (rand() / (float)RAND_MAX) * 5.0f;
            }

    /* --- device allocation and H2D transfer --- */
    float *d_power, *d_buf0, *d_buf1;
    if (cudaMalloc(&d_power, size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_buf0,  size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_buf1,  size * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n"); return 1;
    }
    if (cudaMemcpy(d_power, h_power, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_buf0,  h_buf0,  size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (H2D) failed\n"); return 1;
    }

    /* Grid: BZ=8 covers full z-column → gridDim.z = 1.
     * BY=2 halves gridDim.y relative to r2 (4096 vs 8192 blocks total). */
    dim3 blockDim(BX, BY, BZ);
    dim3 gridDim(
        (nx + BX - 1) / BX,
        (ny + BY - 1) / BY,
        (nz + BZ - 1) / BZ   /* = 1 since BZ == nz == 8 */
    );

    /* --- time-stepping loop --- */
    float *d_cur = d_buf0, *d_nxt = d_buf1;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int iter = 0; iter < niter; iter++) {
        stencil_kernel<<<gridDim, blockDim>>>(
            d_cur, d_nxt, d_power, nx, ny, nz);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "kernel launch failed at iteration %d\n", iter);
            return 1;
        }
        float *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n"); return 1;
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    fprintf(stderr, "compute_seconds: %.6f\n", elapsed);

    /* --- D2H and write output --- */
    float *h_result = (float *)malloc(size * sizeof(float));
    if (!h_result) { fprintf(stderr, "malloc failed\n"); return 1; }
    if (cudaMemcpy(h_result, d_cur, size * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (D2H) failed\n"); return 1;
    }

    FILE *fp = fopen(outfile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", outfile); return 1; }
    int index = 0;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                fprintf(fp, "%d\t%g\n", index++, h_result[idx]);
            }
    fclose(fp);

    free(h_power); free(h_buf0); free(h_result);
    cudaFree(d_power); cudaFree(d_buf0); cudaFree(d_buf1);
    return 0;
}
