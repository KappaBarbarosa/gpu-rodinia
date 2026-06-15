/*
 * hotspot3D CUDA phase-2 round-3
 *
 * Improvements over r1:
 *   1. TILE_Y=4 (128 threads, SMEM_Y=6, smem=3*34*6*4=2448B/block)
 *      Better occupancy on sm_86 (8+ blocks/SM vs 6 for r1).
 *   2. load_plane_structured: structured coalesced halo load with no integer
 *      division - each thread loads its own interior cell; boundary threads
 *      additionally load halo cells. No scatter, no div/mod.
 *   3. stencil_nz8: fully-unrolled 8-level z-kernel (nz==8 fast path) with
 *      static compile-time slot assignments - eliminates all z%3 modulo ops.
 *   4. stencil_generic: uses structured load for arbitrary nz values.
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

#define TILE_X   32
#define TILE_Y   4
#define SMEM_X   (TILE_X + 2)
#define SMEM_Y   (TILE_Y + 2)
#define BLOCK_SZ (TILE_X * TILE_Y)

__device__ __forceinline__
void load_plane_structured(float smem_plane[][SMEM_X],
                            const float * __restrict__ src,
                            int gx, int gy,
                            int nx, int ny,
                            int tx, int ty)
{
    const int gxc = max(0, min(gx, nx - 1));
    const int gyc = max(0, min(gy, ny - 1));

    smem_plane[ty + 1][tx + 1] = __ldg(&src[gyc * nx + gxc]);

    if (tx == 0) {
        int hx = max(0, gx - 1);
        smem_plane[ty + 1][0] = __ldg(&src[gyc * nx + hx]);
    }
    if (tx == TILE_X - 1) {
        int hx = min(gx + 1, nx - 1);
        smem_plane[ty + 1][SMEM_X - 1] = __ldg(&src[gyc * nx + hx]);
    }
    if (ty == 0) {
        int hy = max(0, gy - 1);
        smem_plane[0][tx + 1] = __ldg(&src[hy * nx + gxc]);
    }
    if (ty == TILE_Y - 1) {
        int hy = min(gy + 1, ny - 1);
        smem_plane[SMEM_Y - 1][tx + 1] = __ldg(&src[hy * nx + gxc]);
    }
    if (tx == 0 && ty == 0) {
        int hx = max(0, gx - 1), hy = max(0, gy - 1);
        smem_plane[0][0] = __ldg(&src[hy * nx + hx]);
    }
    if (tx == TILE_X - 1 && ty == 0) {
        int hx = min(gx + 1, nx - 1), hy = max(0, gy - 1);
        smem_plane[0][SMEM_X - 1] = __ldg(&src[hy * nx + hx]);
    }
    if (tx == 0 && ty == TILE_Y - 1) {
        int hx = max(0, gx - 1), hy = min(gy + 1, ny - 1);
        smem_plane[SMEM_Y - 1][0] = __ldg(&src[hy * nx + hx]);
    }
    if (tx == TILE_X - 1 && ty == TILE_Y - 1) {
        int hx = min(gx + 1, nx - 1), hy = min(gy + 1, ny - 1);
        smem_plane[SMEM_Y - 1][SMEM_X - 1] = __ldg(&src[hy * nx + hx]);
    }
}

__device__ __forceinline__
void copy_smem_plane(float smem[][SMEM_Y][SMEM_X], int dst, int src_slot,
                     int tx, int ty)
{
    smem[dst][ty + 1][tx + 1] = smem[src_slot][ty + 1][tx + 1];
    if (tx == 0)
        smem[dst][ty + 1][0]          = smem[src_slot][ty + 1][0];
    if (tx == TILE_X - 1)
        smem[dst][ty + 1][SMEM_X - 1] = smem[src_slot][ty + 1][SMEM_X - 1];
    if (ty == 0)
        smem[dst][0][tx + 1]           = smem[src_slot][0][tx + 1];
    if (ty == TILE_Y - 1)
        smem[dst][SMEM_Y - 1][tx + 1] = smem[src_slot][SMEM_Y - 1][tx + 1];
    if (tx == 0 && ty == 0)
        smem[dst][0][0]                      = smem[src_slot][0][0];
    if (tx == TILE_X - 1 && ty == 0)
        smem[dst][0][SMEM_X - 1]             = smem[src_slot][0][SMEM_X - 1];
    if (tx == 0 && ty == TILE_Y - 1)
        smem[dst][SMEM_Y - 1][0]             = smem[src_slot][SMEM_Y - 1][0];
    if (tx == TILE_X - 1 && ty == TILE_Y - 1)
        smem[dst][SMEM_Y - 1][SMEM_X - 1]   = smem[src_slot][SMEM_Y - 1][SMEM_X - 1];
}

#define STENCIL_CELL(z_, sc_, sb_, sa_) \
    if (valid) { \
        float vc = smem[sc_][ty+1][tx+1]; \
        float vn = smem[sc_][ty  ][tx+1]; \
        float vs = smem[sc_][ty+2][tx+1]; \
        float vw = smem[sc_][ty+1][tx  ]; \
        float ve = smem[sc_][ty+1][tx+2]; \
        float vb = smem[sb_][ty+1][tx+1]; \
        float vt = smem[sa_][ty+1][tx+1]; \
        int c_ = gx + gy*nx + (z_)*xy_stride; \
        nxt[c_] = vc*cc + vn*cn + vs*cs + vw*cw + ve*ce \
                + vb*cb + vt*ct \
                + dtCap*__ldg(&power[c_]) + ambct; \
    }

__global__ void stencil_nz8(
    const float * __restrict__ cur,
    float       * __restrict__ nxt,
    const float * __restrict__ power,
    int nx, int ny,
    float cc, float cn, float cs, float ce, float cw,
    float ct, float cb, float dtCap, float ambct)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_X + tx;
    const int gy = blockIdx.y * TILE_Y + ty;
    const bool valid = (gx < nx) && (gy < ny);
    const int xy_stride = nx * ny;

    __shared__ float smem[3][SMEM_Y][SMEM_X];

    load_plane_structured(smem[0], cur + 0 * xy_stride, gx, gy, nx, ny, tx, ty);
    load_plane_structured(smem[1], cur + 1 * xy_stride, gx, gy, nx, ny, tx, ty);
    __syncthreads();
    copy_smem_plane(smem, 2, 0, tx, ty);
    __syncthreads();

    STENCIL_CELL(0, 0, 2, 1)
    __syncthreads();
    load_plane_structured(smem[2], cur + 2 * xy_stride, gx, gy, nx, ny, tx, ty);
    __syncthreads();

    STENCIL_CELL(1, 1, 0, 2)
    __syncthreads();
    load_plane_structured(smem[0], cur + 3 * xy_stride, gx, gy, nx, ny, tx, ty);
    __syncthreads();

    STENCIL_CELL(2, 2, 1, 0)
    __syncthreads();
    load_plane_structured(smem[1], cur + 4 * xy_stride, gx, gy, nx, ny, tx, ty);
    __syncthreads();

    STENCIL_CELL(3, 0, 2, 1)
    __syncthreads();
    load_plane_structured(smem[2], cur + 5 * xy_stride, gx, gy, nx, ny, tx, ty);
    __syncthreads();

    STENCIL_CELL(4, 1, 0, 2)
    __syncthreads();
    load_plane_structured(smem[0], cur + 6 * xy_stride, gx, gy, nx, ny, tx, ty);
    __syncthreads();

    STENCIL_CELL(5, 2, 1, 0)
    __syncthreads();
    load_plane_structured(smem[1], cur + 7 * xy_stride, gx, gy, nx, ny, tx, ty);
    __syncthreads();

    STENCIL_CELL(6, 0, 2, 1)
    __syncthreads();
    copy_smem_plane(smem, 2, 1, tx, ty);
    __syncthreads();

    STENCIL_CELL(7, 1, 0, 2)
}

#undef STENCIL_CELL

__global__ void stencil_generic(
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
    const bool valid = (gx < nx) && (gy < ny);
    const int xy_stride = nx * ny;

    __shared__ float smem[3][SMEM_Y][SMEM_X];

    load_plane_structured(smem[0], cur + 0 * xy_stride, gx, gy, nx, ny, tx, ty);
    if (nz > 1)
        load_plane_structured(smem[1], cur + 1 * xy_stride, gx, gy, nx, ny, tx, ty);
    __syncthreads();

    copy_smem_plane(smem, 2, 0, tx, ty);
    if (nz == 1)
        copy_smem_plane(smem, 1, 0, tx, ty);
    __syncthreads();

    for (int z = 0; z < nz; z++) {
        const int s_cur   = z % 3;
        const int s_below = (z + 2) % 3;
        const int s_above = (z + 1) % 3;

        if (valid) {
            float vc = smem[s_cur  ][ty + 1][tx + 1];
            float vn = smem[s_cur  ][ty    ][tx + 1];
            float vs = smem[s_cur  ][ty + 2][tx + 1];
            float vw = smem[s_cur  ][ty + 1][tx    ];
            float ve = smem[s_cur  ][ty + 1][tx + 2];
            float vb = smem[s_below][ty + 1][tx + 1];
            float vt = smem[s_above][ty + 1][tx + 1];
            int c = gx + gy * nx + z * xy_stride;
            nxt[c] = vc * cc + vn * cn + vs * cs + vw * cw + ve * ce
                   + vb * cb + vt * ct
                   + dtCap * __ldg(&power[c]) + ambct;
        }

        if (z + 1 < nz) {
            __syncthreads();
            int next_z = min(z + 2, nz - 1);
            load_plane_structured(smem[s_below], cur + next_z * xy_stride,
                                  gx, gy, nx, ny, tx, ty);
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

    (void)dz;

    int size = nx * ny * nz;
    float *power = (float *)malloc(size * sizeof(float));
    float *buf0  = (float *)malloc(size * sizeof(float));
    if (!power || !buf0) { fprintf(stderr, "malloc failed\n"); return 1; }

    srand(7);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i * nx + k * nx * ny;
                power[idx] = (rand() / (float)RAND_MAX) * 15.0f;
                buf0[idx]  = amb_temp + (rand() / (float)RAND_MAX) * 5.0f;
            }

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

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    if (nz == 8) {
        for (int iter = 0; iter < niter; iter++) {
            stencil_nz8<<<grid, block>>>(
                d_cur, d_nxt, d_power,
                nx, ny,
                cc, cn, cs, ce, cw, ct, cb, dtCap, ambct);
            float *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
        }
    } else {
        for (int iter = 0; iter < niter; iter++) {
            stencil_generic<<<grid, block>>>(
                d_cur, d_nxt, d_power,
                nx, ny, nz,
                cc, cn, cs, ce, cw, ct, cb, dtCap, ambct);
            float *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
        }
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
                int idx = j + i * nx + k * nx * ny;
                fprintf(fp, "%d\t%g\n", index++, buf0[idx]);
            }
    fclose(fp);

    cudaFree(d_power); cudaFree(d_buf0); cudaFree(d_buf1);
    free(power); free(buf0);
    return 0;
}
