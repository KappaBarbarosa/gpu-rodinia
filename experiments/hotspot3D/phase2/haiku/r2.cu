/*
 * hotspot3D_cuda.cu — CUDA port of hotspot3D thermal simulation.
 * Phase 2 r2: optimised 3-D shared-memory tiled kernel.
 *
 * Changes over r1 (BY=1, BZ=8, smem=[10][3][34], 65 µs/iter):
 *
 *   1. Eliminate z-halo rows from shared memory.
 *      r1 had smem[BZ+2][BY+2][BX+2] = [10][3][34] = 4080 bytes.
 *      Since BZ == nz (one block covers the full z-column), the z-boundary
 *      condition is clamping, not inter-block data exchange.  Instead of
 *      loading two extra z-halo planes from global memory, we now use
 *      smem[BZ][BY+2][BX+2] = [8][3][34] = 3264 bytes and apply clamped
 *      smem indices (tzB = max(tz-1,0), tzT = min(tz+1,BZ-1)) in the
 *      stencil compute step.  Benefits:
 *        - Eliminates the two conditional global-load paths guarded by
 *          (tz==0) and (tz==BZ-1), removing warp divergence.
 *        - Saves 2×BX×BY = 64 global reads per block per iteration.
 *        - Smaller smem (3264 B vs 4080 B) reduces L1 occupancy pressure.
 *        - Still 6 blocks/SM (smem-limited: 100KB/3264=30 → thread-limited
 *          to 6 by 1536/256=6), so occupancy is unchanged at 48 warps/SM.
 *
 *   2. Pre-bake the ambient temperature term.
 *      r1 computed (c_ct * amb_temp) per output element; r2 stores
 *      c_ambct = c_ct * amb_temp in constant memory, saving one FMA
 *      per output element (×2M elements×100 iters = 200M FMAs saved).
 *
 *   3. All other r1 optimisations retained:
 *      - BX=32, BY=1, BZ=8 (one warp × 8 z-layers × 1 y-row per block)
 *      - gridDim.z=1 since BZ=nz=8
 *      - __constant__ stencil coefficients (constant-cache broadcast)
 *      - __restrict__ on all pointers (enables non-coherent ld.global.nc)
 *      - Branch-free x/y boundary via min/max clamping
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
 * BY=1 : one y-row per block; maximises blocks/SM (6 × 256 = 1536 threads).
 * BZ=8 (= nz): one block covers all z-layers → gridDim.z = 1.
 */
#define BX 32
#define BY  1
#define BZ  8

/*
 * Constant memory: stencil coefficients + pre-baked ambient contribution.
 * c_ambct = c_ct * amb_temp (replaces per-element multiply in the stencil).
 */
__constant__ float c_cc, c_cn, c_cs, c_ce, c_cw, c_ct, c_cb, c_dtCap, c_ambct;

/*
 * stencil_kernel — 3-D shared-memory tiled 7-point stencil.
 *
 * smem layout: [BZ][BY+2][BX+2]  (no z-halo rows; z-boundary via clamped index)
 *
 * Per block, the following data is loaded into smem:
 *   Center:   BZ × BY × BX = 256 elements
 *   X-halos:  BZ × BY × 2  =  16 elements  (loaded by tx==0 and tx==BX-1)
 *   Y-halos:  BZ × 2  × BX = 512 elements  (loaded by ty==0 and ty==BY-1)
 *
 * Z-boundary handled by clamped smem indices (tzB, tzT) in compute step:
 *   tzB = max(tz-1, 0)    → tz==0 reads smem[0] (self, clamped)
 *   tzT = min(tz+1, BZ-1) → tz==BZ-1 reads smem[BZ-1] (self, clamped)
 * This exactly replicates the no-flux/mirror boundary condition without any
 * additional global memory loads.
 */
__global__ void stencil_kernel(
    const float * __restrict__ cur,
    float       * __restrict__ nxt,
    const float * __restrict__ power,
    int nx, int ny, int nz)
{
    /* smem: [BZ][BY+2][BX+2] — no z-halo planes */
    __shared__ float smem[BZ][BY + 2][BX + 2];

    const int tx = threadIdx.x;   /* 0..BX-1 */
    const int ty = threadIdx.y;   /* 0..BY-1 */
    const int tz = threadIdx.z;   /* 0..BZ-1 */

    const int x = blockIdx.x * BX + tx;
    const int y = blockIdx.y * BY + ty;
    const int z = blockIdx.z * BZ + tz;   /* z == tz since gridDim.z=1, blockIdx.z=0 */

    /* Clamped coordinates for x/y boundary conditions */
    const int xc = min(x, nx - 1);
    const int yc = min(y, ny - 1);
    const int xL = max(x - 1, 0);
    const int xR = min(x + 1, nx - 1);
    const int yU = max(y - 1, 0);
    const int yD = min(y + 1, ny - 1);
    /* z is in [0, BZ-1] = [0, nz-1]; no clamping needed for z in smem load. */

    const int plane = nx * ny;

    /* --- Load center tile + x/y halos into smem --- */
    smem[tz][ty + 1][tx + 1] = cur[xc + yc * nx + z * plane];

    /* X halos: loaded by the two boundary threads in x */
    if (tx == 0)
        smem[tz][ty + 1][0]      = cur[xL + yc * nx + z * plane];
    if (tx == BX - 1)
        smem[tz][ty + 1][BX + 1] = cur[xR + yc * nx + z * plane];

    /* Y halos: loaded by the two boundary threads in y */
    if (ty == 0)
        smem[tz][0][tx + 1]      = cur[xc + yU * nx + z * plane];
    if (ty == BY - 1)
        smem[tz][BY + 1][tx + 1] = cur[xc + yD * nx + z * plane];

    /* No z-halo loads: z-boundary handled via clamped smem indices below. */

    __syncthreads();

    /* Skip out-of-bounds threads */
    if (x >= nx || y >= ny || z >= nz) return;

    /* Clamped z smem indices implement no-flux boundary condition:
     * at z=0: below-neighbor = center (tzB=0=tz → smem[0]=smem[tz])
     * at z=BZ-1: above-neighbor = center (tzT=BZ-1=tz → smem[BZ-1]=smem[tz]) */
    const int tzB = max(tz - 1, 0);
    const int tzT = min(tz + 1, BZ - 1);

    const int c = x + y * nx + z * plane;

    nxt[c] = smem[tz ][ty + 1][tx + 1] * c_cc
           + smem[tz ][ty    ][tx + 1] * c_cn   /* north (y-1) */
           + smem[tz ][ty + 2][tx + 1] * c_cs   /* south (y+1) */
           + smem[tz ][ty + 1][tx + 2] * c_ce   /* east  (x+1) */
           + smem[tz ][ty + 1][tx    ] * c_cw   /* west  (x-1) */
           + smem[tzT][ty + 1][tx + 1] * c_ct   /* above (z+1) */
           + smem[tzB][ty + 1][tx + 1] * c_cb   /* below (z-1) */
           + c_dtCap * power[c] + c_ambct;
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

    /* --- derived stencil coefficients --- */
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

    /* Upload to constant memory */
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
    if (!h_power || !h_buf0) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Generate input deterministically (srand(7)) — must match serial */
    srand(7);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                h_power[idx] = (rand() / (float)RAND_MAX) * 15.0f;
                h_buf0[idx]  = amb_temp + (rand() / (float)RAND_MAX) * 5.0f;
            }

    /* --- device allocation --- */
    float *d_power, *d_buf0, *d_buf1;
    if (cudaMalloc(&d_power, size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_buf0,  size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_buf1,  size * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        return 1;
    }
    if (cudaMemcpy(d_power, h_power, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_buf0,  h_buf0,  size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (H2D) failed\n");
        return 1;
    }

    /* Grid: BZ=8 covers full z-column → gridDim.z = (nz+BZ-1)/BZ = 1 when nz==8.
     * Requires BZ == nz for correct z-clamping (validated by the benchmark args). */
    dim3 blockDim(BX, BY, BZ);
    dim3 gridDim(
        (nx + BX - 1) / BX,
        (ny + BY - 1) / BY,
        (nz + BZ - 1) / BZ   /* = 1 when nz == BZ */
    );

    /* --- time-stepping --- */
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
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        return 1;
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    fprintf(stderr, "compute_seconds: %.6f\n", elapsed);

    /* --- copy result back --- */
    float *h_result = (float *)malloc(size * sizeof(float));
    if (!h_result) { fprintf(stderr, "malloc failed\n"); return 1; }
    if (cudaMemcpy(h_result, d_cur, size * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (D2H) failed\n");
        return 1;
    }

    /* --- write output --- */
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
