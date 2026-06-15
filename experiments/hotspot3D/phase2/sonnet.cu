/*
 * hotspot3D CUDA phase-2 — shared-memory XY tiling + z-sweep with sliding window.
 *
 * Optimizations vs phase-1 baseline:
 *   1. Shared-memory XY tiling with 1-cell halos: E/W/N/S neighbors served from smem.
 *   2. Z-sweep with 3-plane sliding window: each z-plane is read from global memory
 *      exactly once and reused as "above", "current", and "below" neighbor across
 *      three consecutive z-iterations → ~2x reduction in global z-neighbor reads.
 *   3. Divergence-free cooperative tile loading: all threads in a block participate
 *      uniformly in loading each smem plane (no branch-heavy halo code).
 *   4. Pre-baked ambct = ct * amb_temp: avoids one multiply per stencil evaluation.
 *   5. __ldg for power[] and cur[] reads (explicit read-only cache path).
 *   6. TILE_Y=4: 32×4 threads → 128-thread blocks, 4 warps, with smem=612 floats/plane
 *      (3×1836 B = 5508 B/block), allowing high SM occupancy.
 *
 * Grid: (ceil(nx/32), ceil(ny/4), 1)  — z is a loop inside the kernel.
 * Block: (32, 4, 1) = 128 threads, 4 warps.
 *
 * Shared memory per block: 3 planes × 34×6 = 612 floats = 2448 bytes.
 */

#include <stdio.h>
#include <stdlib.h>
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

#define TILE_X  32
#define TILE_Y  4
#define SMEM_X  (TILE_X + 2)   /* 34 */
#define SMEM_Y  (TILE_Y + 2)   /* 6  */
#define SMEM_N  (SMEM_X * SMEM_Y)  /* 204 elements per plane */

/*
 * Cooperative load of one XY z-plane into smem_plane (SMEM_Y × SMEM_X).
 *
 * All TILE_X*TILE_Y threads participate. Each thread loads one or two elements
 * by iterating over a strided set of linear smem indices. No branch-heavy halo
 * code — each thread computes its global (x, y) by offsetting from the tile origin.
 * Out-of-bounds coordinates are clamped (Neumann BC = same value as boundary cell).
 */
__device__ __forceinline__
void load_plane_coop(float smem_plane[][SMEM_X],
                     const float * __restrict__ cur_plane,
                     int tile_x0, int tile_y0,
                     int nx, int ny,
                     int tid)    /* linear thread id within block: ty*TILE_X + tx */
{
    /* Each thread covers indices: tid, tid + TILE_X*TILE_Y, ... */
    const int block_size = TILE_X * TILE_Y;
    #pragma unroll
    for (int s = tid; s < SMEM_N; s += block_size) {
        int sy = s / SMEM_X;
        int sx = s % SMEM_X;
        int gy = tile_y0 - 1 + sy;
        int gx = tile_x0 - 1 + sx;
        /* Clamp to [0, nx-1] × [0, ny-1] (Neumann BC) */
        int gx_c = max(0, min(gx, nx - 1));
        int gy_c = max(0, min(gy, ny - 1));
        smem_plane[sy][sx] = __ldg(&cur_plane[gy_c * nx + gx_c]);
    }
}

__global__ void stencil_kernel(
    const float * __restrict__ cur,
    float       * __restrict__ nxt,
    const float * __restrict__ power,
    int nx, int ny, int nz,
    float cc, float cn, float cs, float ce, float cw,
    float ct, float cb, float dtCap, float ambct)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_X + tx;
    const int gy = blockIdx.y * TILE_Y + ty;
    const int tile_x0 = blockIdx.x * TILE_X;
    const int tile_y0 = blockIdx.y * TILE_Y;

    const bool valid = (gx < nx) && (gy < ny);
    const int xy_stride = nx * ny;
    const int tid = ty * TILE_X + tx;   /* linear thread id in block */

    /*
     * Shared memory: 3 slots (ring buffer), each SMEM_Y × SMEM_X.
     * At z-iteration z:
     *   slot_cur   = z % 3
     *   slot_below = (z + 2) % 3
     *   slot_above = (z + 1) % 3
     *
     * Rotation: after computing z, load z+2 into slot_below
     * (which becomes the new slot_above for z+1).
     */
    __shared__ float smem[3][SMEM_Y][SMEM_X];

    /* ------------------------------------------------------------------ *
     * Initial load: slot0 = z=0, slot1 = z=1, slot2 = z=-1(clamped=0)  *
     * Optimization: load z=0 once into slot0, then copy to slot2         *
     * (avoids double global read of z=0 plane for the boundary clamp).   *
     * ------------------------------------------------------------------ */
    load_plane_coop(smem[0], cur + 0 * xy_stride,  tile_x0, tile_y0, nx, ny, tid); /* z=0  */
    load_plane_coop(smem[1], cur + 1 * xy_stride,  tile_x0, tile_y0, nx, ny, tid); /* z=1  */
    __syncthreads();
    /* Copy smem[0] (z=0) → smem[2] (z=-1 clamped): no extra global memory traffic */
    for (int s = tid; s < SMEM_N; s += TILE_X * TILE_Y) {
        int sy = s / SMEM_X, sx = s % SMEM_X;
        smem[2][sy][sx] = smem[0][sy][sx];
    }
    __syncthreads();

    /* ------------------------------------------------------------------ *
     * Z-sweep                                                             *
     * ------------------------------------------------------------------ */
    for (int z = 0; z < nz; z++) {
        const int s_cur   = z % 3;
        const int s_below = (z + 2) % 3;
        const int s_above = (z + 1) % 3;

        if (valid) {
            const int c = gx + gy * nx + z * xy_stride;

            const float vc = smem[s_cur  ][ty + 1][tx + 1];
            const float vn = smem[s_cur  ][ty    ][tx + 1];
            const float vs = smem[s_cur  ][ty + 2][tx + 1];
            const float vw = smem[s_cur  ][ty + 1][tx    ];
            const float ve = smem[s_cur  ][ty + 1][tx + 2];
            const float vb = smem[s_below][ty + 1][tx + 1];
            const float vt = smem[s_above][ty + 1][tx + 1];

            nxt[c] = vc * cc
                   + vn * cn + vs * cs
                   + vw * cw + ve * ce
                   + vb * cb + vt * ct
                   + dtCap * __ldg(&power[c]) + ambct;
        }

        /* Advance: load the next "above" plane (z+2) into the freed slot */
        if (z + 1 < nz) {
            __syncthreads();
            int next_above_z = min(z + 2, nz - 1);   /* clamp last z+1 to nz-1 (Neumann) */
            load_plane_coop(smem[s_below], cur + next_above_z * xy_stride,
                            tile_x0, tile_y0, nx, ny, tid);
            __syncthreads();
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <output_file>\n", argv[0]);
        return 1;
    }

    int nx    = atoi(argv[1]);
    int ny    = nx;
    int nz    = atoi(argv[2]);
    int niter = atoi(argv[3]);
    const char *outfile = argv[4];

    /* Derived coefficients (matches serial reference exactly) */
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
    float cc = 1.0f - (2.0f * ce + 2.0f * cn + 3.0f * ct);
    float dtCap = dt / Cap;
    float ambct = ct * amb_temp;

    int size = nx * ny * nz;
    float *power = (float *)malloc(size * sizeof(float));
    float *buf0  = (float *)malloc(size * sizeof(float));
    if (!power || !buf0) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* Generate input deterministically (same as reference) */
    srand(7);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i * nx + k * nx * ny;
                power[idx] = (rand() / (float)RAND_MAX) * 15.0f;
                buf0[idx]  = amb_temp + (rand() / (float)RAND_MAX) * 5.0f;
            }

    /* Allocate device memory */
    float *d_power, *d_buf0, *d_buf1;
    cudaMalloc(&d_power, size * sizeof(float));
    cudaMalloc(&d_buf0,  size * sizeof(float));
    cudaMalloc(&d_buf1,  size * sizeof(float));

    cudaMemcpy(d_power, power, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf0,  buf0,  size * sizeof(float), cudaMemcpyHostToDevice);

    /* Grid covers XY; z-loop is inside the kernel */
    dim3 block(TILE_X, TILE_Y, 1);
    dim3 grid((nx + TILE_X - 1) / TILE_X,
              (ny + TILE_Y - 1) / TILE_Y,
              1);

    float *d_cur = d_buf0, *d_nxt = d_buf1;

    /* Time only the stencil computation */
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int iter = 0; iter < niter; iter++) {
        stencil_kernel<<<grid, block>>>(
            d_cur, d_nxt, d_power,
            nx, ny, nz,
            cc, cn, cs, ce, cw, ct, cb, dtCap, ambct);
        float *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
    }
    cudaDeviceSynchronize();

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    fprintf(stderr, "compute_seconds: %.6f\n", elapsed);

    /* Copy result back */
    cudaMemcpy(buf0, d_cur, size * sizeof(float), cudaMemcpyDeviceToHost);

    /* Write output */
    FILE *fp = fopen(outfile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", outfile); return 1; }
    int index = 0;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i * nx + k * nx * ny;
                fprintf(fp, "%d\t%g\n", index++, buf0[idx]);
            }
    fclose(fp);

    cudaFree(d_power); cudaFree(d_buf0); cudaFree(d_buf1);
    free(power); free(buf0);
    return 0;
}
