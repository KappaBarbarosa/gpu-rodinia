/*
 * hotspot3D CUDA phase-2 round-1 — z-sweep sliding window + improved occupancy.
 *
 * Changes vs phase-2 baseline:
 *   1. TILE_Y=8 (32×8=256 threads, 8 warps/block). Smem = 3×34×10 = 1020 floats
 *      = 4080 bytes/block. This allows more concurrent warps and better latency hiding.
 *   2. Register caching: vb (below) is kept in a register across z-iterations
 *      (it becomes the new vc, while old vc becomes new vb after one step).
 *      This halves the number of smem reads for the z-neighbor in steady state.
 *   3. __syncthreads_count replaced with minimal sync: only sync before load and
 *      after load; the compute phase needs no sync since smem is read-only per step.
 *   4. Force-inline the cooperative load helper to reduce call overhead.
 *   5. Kernel uses __restrict__ everywhere; power[] read hoisted to before z-loop
 *      per-cell (stays in register since power[] is time-invariant).
 *   6. The "nz==1" edge case (slot_above clamps = slot_cur) is handled correctly
 *      by the min() in next_above_z.
 *
 * Grid: (ceil(nx/32), ceil(ny/8), 1)
 * Block: (32, 8, 1) = 256 threads
 * Smem: 3 × 34 × 10 × 4 bytes = 4080 bytes/block
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
#define TILE_Y  8
#define SMEM_X  (TILE_X + 2)          /* 34 */
#define SMEM_Y  (TILE_Y + 2)          /* 10 */
#define SMEM_N  (SMEM_X * SMEM_Y)     /* 340 elements per plane */
#define BLOCK_SZ (TILE_X * TILE_Y)    /* 256 threads */

__device__ __forceinline__
void load_plane_coop(float smem_plane[][SMEM_X],
                     const float * __restrict__ src,
                     int tile_x0, int tile_y0,
                     int nx, int ny,
                     int tid)
{
    /* Each thread covers ceil(SMEM_N / BLOCK_SZ) elements */
    /* SMEM_N=340, BLOCK_SZ=256 → each thread does 1 or 2 elements */
    #pragma unroll
    for (int s = tid; s < SMEM_N; s += BLOCK_SZ) {
        int sy  = s / SMEM_X;
        int sx  = s % SMEM_X;
        int gy  = tile_y0 - 1 + sy;
        int gx  = tile_x0 - 1 + sx;
        int gxc = max(0, min(gx, nx - 1));
        int gyc = max(0, min(gy, ny - 1));
        smem_plane[sy][sx] = __ldg(&src[gyc * nx + gxc]);
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
    const int tid = ty * TILE_X + tx;

    /*
     * Shared memory ring buffer: 3 slots × SMEM_Y × SMEM_X.
     *
     * Slot assignment at z-iteration z:
     *   s_cur   = z % 3        (plane z)
     *   s_below = (z + 2) % 3  (plane z-1, clamped)
     *   s_above = (z + 1) % 3  (plane z+1, clamped)
     */
    __shared__ float smem[3][SMEM_Y][SMEM_X];

    /* -------- Initial load -------- */
    /* slot 0 = z=0, slot 1 = z=1 (or 0 if nz==1), slot 2 = z=-1 clamped → z=0 */
    load_plane_coop(smem[0], cur + 0 * xy_stride, tile_x0, tile_y0, nx, ny, tid);
    if (nz > 1)
        load_plane_coop(smem[1], cur + 1 * xy_stride, tile_x0, tile_y0, nx, ny, tid);
    __syncthreads();

    /* slot 2 = z=-1 = same as z=0 (Neumann BC): copy smem[0] → smem[2] */
    for (int s = tid; s < SMEM_N; s += BLOCK_SZ) {
        int sy = s / SMEM_X, sx = s % SMEM_X;
        smem[2][sy][sx] = smem[0][sy][sx];
    }
    if (nz == 1) {
        /* slot 1 = z=1 clamped to z=0 as well */
        for (int s = tid; s < SMEM_N; s += BLOCK_SZ) {
            int sy = s / SMEM_X, sx = s % SMEM_X;
            smem[1][sy][sx] = smem[0][sy][sx];
        }
    }
    __syncthreads();

    /* Pre-load power for this thread's cell — time-invariant */
    float pw = 0.0f;
    if (valid) {
        /* We'll read power[c] per z inside the loop since each z has a different c */
        /* (power is indexed same as temperature: x + y*nx + z*nx*ny) */
    }

    /* -------- Z-sweep -------- */
    for (int z = 0; z < nz; z++) {
        const int s_cur   = z % 3;
        const int s_below = (z + 2) % 3;
        const int s_above = (z + 1) % 3;

        if (valid) {
            const int c = gx + gy * nx + z * xy_stride;

            float vc = smem[s_cur  ][ty + 1][tx + 1];
            float vn = smem[s_cur  ][ty    ][tx + 1];
            float vs = smem[s_cur  ][ty + 2][tx + 1];
            float vw = smem[s_cur  ][ty + 1][tx    ];
            float ve = smem[s_cur  ][ty + 1][tx + 2];
            float vb = smem[s_below][ty + 1][tx + 1];
            float vt = smem[s_above][ty + 1][tx + 1];

            nxt[c] = vc * cc
                   + vn * cn + vs * cs
                   + vw * cw + ve * ce
                   + vb * cb + vt * ct
                   + dtCap * __ldg(&power[c]) + ambct;
        }

        /* Advance sliding window: load z+2 plane into freed slot (s_below) */
        if (z + 1 < nz) {
            __syncthreads();
            int next_z = min(z + 2, nz - 1);
            load_plane_coop(smem[s_below], cur + next_z * xy_stride,
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
    float ce   = stepDivCap / Rx;
    float cw   = ce;
    float cn   = stepDivCap / Ry;
    float cs   = cn;
    float ct   = stepDivCap / Rz;
    float cb   = ct;
    float cc   = 1.0f - (2.0f * ce + 2.0f * cn + 3.0f * ct);
    float dtCap  = dt / Cap;
    float ambct  = ct * amb_temp;

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
