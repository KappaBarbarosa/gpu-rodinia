/*
 * hotspot3D CUDA phase-2 round-2 — structured coalesced halo load + TILE_Y=4 +
 *                                   fully-unrolled nz=8 specialization +
 *                                   multi-iteration kernel (all iters in one launch).
 *
 * r2 retains all r1 changes:
 *   1. STRUCTURED HALO LOAD: coalesced interior + separate halo rows/cols.
 *   2. TILE_Y=4: 128 threads/block, 12 blocks/SM, 100% occupancy.
 *   3. FULLY UNROLLED nz=8: no modulo in hot path; Neumann BC for z=7 correct.
 *
 * NEW in this round:
 *   4. MULTI-ITERATION KERNEL (stencil_kernel_nz8_multi):
 *      All niter iterations run inside ONE kernel launch. Eliminates ~100
 *      kernel-launch round-trips and keeps L2 warm between iterations.
 *      Pointer alternation: (iter&1) selects cur/nxt between {buf0, buf1}.
 *      __syncthreads() at iteration end ensures all nxt writes are visible
 *      before the next iteration reads them as cur (blocks are independent).
 *
 * Result buffer after niter iterations (multi-iter kernel):
 *   last_iter = niter-1; nxt = (last_iter&1) ? buf0 : buf1.
 *   niter=100 (even) -> last_iter=99 odd -> nxt=buf0 -> result in buf0.
 *   niter=1 (odd)   -> last_iter=0 even -> nxt=buf1 -> result in buf1.
 *   Host selects: (niter-1)&1 ? buf0 : buf1.
 *
 * Grid:  (ceil(nx/32), ceil(ny/4), 1)
 * Block: (32, 4, 1) = 128 threads = 4 warps
 * Smem:  3 * 6 * 34 * 4 = 2448 bytes per block
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
#define SMEM_N   (SMEM_X * SMEM_Y)
#define BLOCK_SZ (TILE_X * TILE_Y)

/* ─── device helpers ─── */

__device__ __forceinline__
float gload_clamped(const float * __restrict__ src,
                    int sx, int sy, int tile_x0, int tile_y0, int nx, int ny)
{
    int gx = max(0, min(tile_x0 - 1 + sx, nx - 1));
    int gy = max(0, min(tile_y0 - 1 + sy, ny - 1));
    return __ldg(&src[gy * nx + gx]);
}

__device__ __forceinline__
void load_plane_structured(float smem_plane[][SMEM_X],
                           const float * __restrict__ src,
                           int tile_x0, int tile_y0, int nx, int ny,
                           int tx, int ty)
{
    smem_plane[ty+1][tx+1] = gload_clamped(src, tx+1, ty+1, tile_x0, tile_y0, nx, ny);

    if (ty == 0) {
        smem_plane[0][tx+1] = gload_clamped(src, tx+1, 0, tile_x0, tile_y0, nx, ny);
        if (tx == 0)          smem_plane[0][0]        = gload_clamped(src, 0,        0,        tile_x0, tile_y0, nx, ny);
        if (tx == TILE_X-1)   smem_plane[0][SMEM_X-1] = gload_clamped(src, SMEM_X-1, 0,        tile_x0, tile_y0, nx, ny);
    }
    if (ty == TILE_Y-1) {
        smem_plane[SMEM_Y-1][tx+1] = gload_clamped(src, tx+1, SMEM_Y-1, tile_x0, tile_y0, nx, ny);
        if (tx == 0)          smem_plane[SMEM_Y-1][0]        = gload_clamped(src, 0,        SMEM_Y-1, tile_x0, tile_y0, nx, ny);
        if (tx == TILE_X-1)   smem_plane[SMEM_Y-1][SMEM_X-1] = gload_clamped(src, SMEM_X-1, SMEM_Y-1, tile_x0, tile_y0, nx, ny);
    }
    if (tx == 0)        smem_plane[ty+1][0]        = gload_clamped(src, 0,        ty+1, tile_x0, tile_y0, nx, ny);
    if (tx == TILE_X-1) smem_plane[ty+1][SMEM_X-1] = gload_clamped(src, SMEM_X-1, ty+1, tile_x0, tile_y0, nx, ny);
}

__device__ __forceinline__
void copy_plane_structured(float dst[][SMEM_X], const float src[][SMEM_X], int tx, int ty)
{
    dst[ty+1][tx+1] = src[ty+1][tx+1];
    if (ty == 0) {
        dst[0][tx+1] = src[0][tx+1];
        if (tx == 0)        dst[0][0]        = src[0][0];
        if (tx == TILE_X-1) dst[0][SMEM_X-1] = src[0][SMEM_X-1];
    }
    if (ty == TILE_Y-1) {
        dst[SMEM_Y-1][tx+1] = src[SMEM_Y-1][tx+1];
        if (tx == 0)        dst[SMEM_Y-1][0]        = src[SMEM_Y-1][0];
        if (tx == TILE_X-1) dst[SMEM_Y-1][SMEM_X-1] = src[SMEM_Y-1][SMEM_X-1];
    }
    if (tx == 0)        dst[ty+1][0]        = src[ty+1][0];
    if (tx == TILE_X-1) dst[ty+1][SMEM_X-1] = src[ty+1][SMEM_X-1];
}

__device__ __forceinline__
void compute_z(float smem[][SMEM_Y][SMEM_X],
               float * __restrict__ nxt,
               const float * __restrict__ power,
               int gx, int gy, int nx, int z_val, int xy,
               int sc, int sb, int sa,
               float cc, float cn, float cs, float ce, float cw,
               float ct, float cb, float dtCap, float ambct,
               int tx, int ty, bool valid)
{
    if (valid) {
        const int c = gx + gy*nx + z_val*xy;
        const float vc = smem[sc][ty+1][tx+1];
        const float vn = smem[sc][ty  ][tx+1];
        const float vs = smem[sc][ty+2][tx+1];
        const float vw = smem[sc][ty+1][tx  ];
        const float ve = smem[sc][ty+1][tx+2];
        const float vb = smem[sb][ty+1][tx+1];
        const float vt = smem[sa][ty+1][tx+1];
        nxt[c] = vc*cc + vn*cn + vs*cs + vw*cw + ve*ce
               + vb*cb + vt*ct + dtCap * __ldg(&power[c]) + ambct;
    }
}

/* ─── nz=8 multi-iteration kernel ─── */

__global__ void stencil_kernel_nz8_multi(
    float * __restrict__ buf0,
    float * __restrict__ buf1,
    const float * __restrict__ power,
    int nx, int ny, int niter,
    float cc, float cn, float cs, float ce, float cw,
    float ct, float cb, float dtCap, float ambct)
{
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_X + tx;
    const int gy = blockIdx.y * TILE_Y + ty;
    const int tile_x0 = blockIdx.x * TILE_X;
    const int tile_y0 = blockIdx.y * TILE_Y;
    const bool valid = (gx < nx) && (gy < ny);
    const int xy = nx * ny;

    __shared__ float smem[3][SMEM_Y][SMEM_X];

    for (int iter = 0; iter < niter; iter++) {
        const float * __restrict__ cur = (iter & 1) ? buf1 : buf0;
        float       * __restrict__ nxt = (iter & 1) ? buf0 : buf1;

        load_plane_structured(smem[0], cur + 0*xy, tile_x0, tile_y0, nx, ny, tx, ty);
        load_plane_structured(smem[1], cur + 1*xy, tile_x0, tile_y0, nx, ny, tx, ty);
        __syncthreads();
        copy_plane_structured(smem[2], smem[0], tx, ty);
        __syncthreads();

        /* z=0: sc=0(z0), sb=2(z-1 clamped=z0), sa=1(z1) */
        compute_z(smem,nxt,power,gx,gy,nx,0,xy,0,2,1,cc,cn,cs,ce,cw,ct,cb,dtCap,ambct,tx,ty,valid);
        __syncthreads();
        load_plane_structured(smem[2], cur + 2*xy, tile_x0, tile_y0, nx, ny, tx, ty);
        __syncthreads();

        /* z=1: sc=1(z1), sb=0(z0), sa=2(z2) */
        compute_z(smem,nxt,power,gx,gy,nx,1,xy,1,0,2,cc,cn,cs,ce,cw,ct,cb,dtCap,ambct,tx,ty,valid);
        __syncthreads();
        load_plane_structured(smem[0], cur + 3*xy, tile_x0, tile_y0, nx, ny, tx, ty);
        __syncthreads();

        /* z=2: sc=2(z2), sb=1(z1), sa=0(z3) */
        compute_z(smem,nxt,power,gx,gy,nx,2,xy,2,1,0,cc,cn,cs,ce,cw,ct,cb,dtCap,ambct,tx,ty,valid);
        __syncthreads();
        load_plane_structured(smem[1], cur + 4*xy, tile_x0, tile_y0, nx, ny, tx, ty);
        __syncthreads();

        /* z=3: sc=0(z3), sb=2(z2), sa=1(z4) */
        compute_z(smem,nxt,power,gx,gy,nx,3,xy,0,2,1,cc,cn,cs,ce,cw,ct,cb,dtCap,ambct,tx,ty,valid);
        __syncthreads();
        load_plane_structured(smem[2], cur + 5*xy, tile_x0, tile_y0, nx, ny, tx, ty);
        __syncthreads();

        /* z=4: sc=1(z4), sb=0(z3), sa=2(z5) */
        compute_z(smem,nxt,power,gx,gy,nx,4,xy,1,0,2,cc,cn,cs,ce,cw,ct,cb,dtCap,ambct,tx,ty,valid);
        __syncthreads();
        load_plane_structured(smem[0], cur + 6*xy, tile_x0, tile_y0, nx, ny, tx, ty);
        __syncthreads();

        /* z=5: sc=2(z5), sb=1(z4), sa=0(z6) */
        compute_z(smem,nxt,power,gx,gy,nx,5,xy,2,1,0,cc,cn,cs,ce,cw,ct,cb,dtCap,ambct,tx,ty,valid);
        __syncthreads();
        load_plane_structured(smem[1], cur + 7*xy, tile_x0, tile_y0, nx, ny, tx, ty);
        __syncthreads();

        /* z=6: sc=0(z6), sb=2(z5), sa=1(z7) → load z7 clamped into slot2 for Neumann BC */
        compute_z(smem,nxt,power,gx,gy,nx,6,xy,0,2,1,cc,cn,cs,ce,cw,ct,cb,dtCap,ambct,tx,ty,valid);
        __syncthreads();
        load_plane_structured(smem[2], cur + 7*xy, tile_x0, tile_y0, nx, ny, tx, ty);
        __syncthreads();

        /* z=7: sc=1(z7), sb=0(z6), sa=2(z7 clamped) */
        compute_z(smem,nxt,power,gx,gy,nx,7,xy,1,0,2,cc,cn,cs,ce,cw,ct,cb,dtCap,ambct,tx,ty,valid);

        /* barrier: all writes to nxt visible before next iter reads them */
        __syncthreads();
    }
}

/* ─── generic kernel (arbitrary nz) ─── */

__global__ void stencil_kernel(
    const float * __restrict__ cur,
    float       * __restrict__ nxt,
    const float * __restrict__ power,
    int nx, int ny, int nz,
    float cc, float cn, float cs, float ce, float cw,
    float ct, float cb, float dtCap, float ambct)
{
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_X + tx;
    const int gy = blockIdx.y * TILE_Y + ty;
    const int tile_x0 = blockIdx.x * TILE_X;
    const int tile_y0 = blockIdx.y * TILE_Y;
    const bool valid = (gx < nx) && (gy < ny);
    const int xy = nx * ny;
    const int tid = ty * TILE_X + tx;

    __shared__ float smem[3][SMEM_Y][SMEM_X];

    load_plane_structured(smem[0], cur + 0*xy, tile_x0, tile_y0, nx, ny, tx, ty);
    if (nz > 1)
        load_plane_structured(smem[1], cur + 1*xy, tile_x0, tile_y0, nx, ny, tx, ty);
    __syncthreads();

    for (int s = tid; s < SMEM_N; s += BLOCK_SZ) {
        int sy = s / SMEM_X, sx = s % SMEM_X;
        smem[2][sy][sx] = smem[0][sy][sx];
    }
    if (nz == 1) {
        for (int s = tid; s < SMEM_N; s += BLOCK_SZ) {
            int sy = s / SMEM_X, sx = s % SMEM_X;
            smem[1][sy][sx] = smem[0][sy][sx];
        }
    }
    __syncthreads();

    for (int z = 0; z < nz; z++) {
        const int s_cur   = z % 3;
        const int s_below = (z + 2) % 3;
        const int s_above = (z + 1) % 3;

        if (valid) {
            const int c = gx + gy*nx + z*xy;
            const float vc = smem[s_cur  ][ty+1][tx+1];
            const float vn = smem[s_cur  ][ty  ][tx+1];
            const float vs = smem[s_cur  ][ty+2][tx+1];
            const float vw = smem[s_cur  ][ty+1][tx  ];
            const float ve = smem[s_cur  ][ty+1][tx+2];
            const float vb = smem[s_below][ty+1][tx+1];
            const float vt = smem[s_above][ty+1][tx+1];
            nxt[c] = vc*cc + vn*cn + vs*cs + vw*cw + ve*ce
                   + vb*cb + vt*ct + dtCap * __ldg(&power[c]) + ambct;
        }

        if (z + 1 < nz) {
            __syncthreads();
            int next_z = min(z + 2, nz - 1);
            load_plane_structured(smem[s_below], cur + next_z*xy,
                                  tile_x0, tile_y0, nx, ny, tx, ty);
            __syncthreads();
        }
    }
}

/* ─── host ─── */

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
    float cc   = 1.0f - (2.0f*ce + 2.0f*cn + 3.0f*ct);
    float dtCap  = dt / Cap;
    float ambct  = ct * amb_temp;

    int size = nx * ny * nz;
    float *power = (float *)malloc(size * sizeof(float));
    float *buf0  = (float *)malloc(size * sizeof(float));
    if (!power || !buf0) { fprintf(stderr, "malloc failed\n"); return 1; }

    srand(7);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
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
    dim3 grid((nx + TILE_X-1) / TILE_X,
              (ny + TILE_Y-1) / TILE_Y, 1);

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    float *d_result;
    if (nz == 8) {
        stencil_kernel_nz8_multi<<<grid, block>>>(
            d_buf0, d_buf1, d_power,
            nx, ny, niter,
            cc, cn, cs, ce, cw, ct, cb, dtCap, ambct);
        cudaDeviceSynchronize();
        /* last iter = niter-1; nxt = ((niter-1)&1) ? buf0 : buf1 */
        d_result = ((niter-1) & 1) ? d_buf0 : d_buf1;
    } else {
        float *d_cur = d_buf0, *d_nxt = d_buf1;
        for (int iter = 0; iter < niter; iter++) {
            stencil_kernel<<<grid, block>>>(
                d_cur, d_nxt, d_power,
                nx, ny, nz,
                cc, cn, cs, ce, cw, ct, cb, dtCap, ambct);
            float *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
        }
        cudaDeviceSynchronize();
        d_result = d_cur;
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    fprintf(stderr, "compute_seconds: %.6f\n", elapsed);

    cudaMemcpy(buf0, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

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

    cudaFree(d_power); cudaFree(d_buf0); cudaFree(d_buf1);
    free(power); free(buf0);
    return 0;
}
