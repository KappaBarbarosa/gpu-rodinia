/*
 * hotspot3D — Phase 2 round-3 optimised CUDA kernel.
 *
 * Built on r1 (32×8 block, register Z-pipeline, constant-mem coefficients,
 * 6.57 ms / 100 iters, 113× speedup).
 *
 * Round-3 improvements over r1:
 *
 *  1. BY = 16  (block = 32×16 = 512 threads, half as many blocks, 16×32=512).
 *     The larger Y-tile better amortises halo loads: the perimeter/area ratio
 *     drops from (2·34+2·10)/(34·10)=0.256 to (2·34+2·18)/(34·18)=0.170.
 *     Fewer blocks also reduces scheduling overhead on the 46 SMs.
 *     Shared memory per block = 34×18×4 = 2448 bytes — still tiny.
 *
 *  2. Power prefetch into registers: pw_cur (current Z) and pw_nxt (next Z)
 *     are pipelined exactly like vc/vt.  This eliminates the random-stride
 *     global read `power[base+col]` from the inner computation (replaced by a
 *     register read), and the prefetch reads are issued early via __ldg()
 *     (read-only cache / L1 texture path) to hide latency.
 *
 *  3. All global loads use __ldg() to steer through the read-only cache.
 *
 *  4. #pragma unroll on the Z-loop (nz typically small, e.g. 8).
 *
 *  5. The bottom/top boundary branches (z==0 / z==nz-1) are factored out:
 *     vb_reg is primed to vc_reg before the loop so z==0 naturally uses the
 *     correct value without a branch inside the loop; only the top boundary
 *     still needs a runtime check (unavoidable when nz is not a compile-time
 *     constant).
 *
 * Correctness: identical 7-point stencil, clamp-to-edge boundaries, ping-pong
 * buffers, same input gen / output format / CLI as the serial golden.
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
#define BX  32
#define BY  16
#define SX  (BX + 2)    /* 34: interior + west/east halo  */
#define SY  (BY + 2)    /* 18: interior + north/south halo */

/*
 * Grid:   ceil(nx/BX) x ceil(ny/BY)
 * Block:  BX x BY (= 32 x 16 = 512 threads)
 *
 * Each block owns a BX×BY XY tile and sweeps all nz Z-layers.
 * Centre column is pipelined through registers (vb/vc/vt) and
 * power through (pw_cur/pw_nxt).
 */
__global__ void stencil_kernel(
        const float * __restrict__ cur,
        float       * __restrict__ nxt,
        const float * __restrict__ power,
        int nx, int ny, int nz)
{
    __shared__ float s[SY][SX];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * BX + tx;
    const int gy = blockIdx.y * BY + ty;
    const int nxny = nx * ny;
    const bool valid = (gx < nx) && (gy < ny);

    const int col = gy * nx + gx;              /* in-plane offset */

    /* Clamped in-plane neighbour offsets (computed once). */
    const int wcol = (gx > 0)      ? col - 1  : col;
    const int ecol = (gx < nx - 1) ? col + 1  : col;
    const int ncol = (gy > 0)      ? col - nx : col;
    const int scol = (gy < ny - 1) ? col + nx : col;

    /* This thread's shared-memory position. */
    const int sx = tx + 1;
    const int sy = ty + 1;

    /* Halo-ring flag: which threads load halo cells. */
    const bool wEdge = (tx == 0);
    const bool eEdge = (tx == BX - 1) || (gx == nx - 1);
    const bool nEdge = (ty == 0);
    const bool sEdge = (ty == BY - 1) || (gy == ny - 1);

    /* ---- Register Z-pipeline priming ---- */
    float vb_reg = 0.0f, vc_reg = 0.0f, vt_reg = 0.0f;
    float pw_cur = 0.0f, pw_nxt = 0.0f;

    if (valid) {
        vc_reg = __ldg(&cur[col]);                                   /* z=0 */
        vt_reg = (nz > 1) ? __ldg(&cur[nxny + col]) : vc_reg;       /* z=1 */
        /* Prime vb = vc so the z=0 bottom boundary works without a branch. */
        vb_reg = vc_reg;

        pw_cur = __ldg(&power[col]);                                 /* z=0 */
        pw_nxt = (nz > 1) ? __ldg(&power[nxny + col]) : pw_cur;     /* z=1 */
    }

#pragma unroll 8
    for (int z = 0; z < nz; z++) {
        const int base = z * nxny;

        if (valid) {
            /* Interior cell: already in register — no DRAM read. */
            s[sy][sx] = vc_reg;
            /* Halo ring from read-only cache. */
            if (wEdge) s[sy][0]      = __ldg(&cur[base + wcol]);
            if (eEdge) s[sy][BX + 1] = __ldg(&cur[base + ecol]);
            if (nEdge) s[0][sx]      = __ldg(&cur[base + ncol]);
            if (sEdge) s[SY - 1][sx] = __ldg(&cur[base + scol]);
        }
        __syncthreads();

        if (valid) {
            const float vn = s[sy - 1][sx];
            const float vs = s[sy + 1][sx];
            const float vw = s[sy][sx - 1];
            const float ve = s[sy][sx + 1];
            /* vb_reg already equals vc_reg at z==0 (primed above). */
            const float vt = (z == nz - 1) ? vc_reg : vt_reg;

            nxt[base + col] =
                vc_reg * d_cc +
                vn * d_cn + vs * d_cs +
                ve * d_ce + vw * d_cw +
                vt * d_ct + vb_reg * d_cb +
                pw_cur * d_dtCap + d_ambct;
        }
        __syncthreads();

        /* Advance pipelines. */
        vb_reg = vc_reg;
        vc_reg = vt_reg;
        pw_cur = pw_nxt;

        if (valid) {
            if (z + 2 < nz) {
                vt_reg = __ldg(&cur[(z + 2) * nxny + col]);
                pw_nxt = __ldg(&power[(z + 2) * nxny + col]);
            }
            /* else: vt_reg/pw_nxt unused (top boundary uses vc_reg). */
        }
    }
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

    dim3 block(BX, BY, 1);
    dim3 grid((nx + BX - 1) / BX, (ny + BY - 1) / BY, 1);

    float *d_cur = d_a, *d_nxt = d_b;

    cudaDeviceSynchronize();
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int iter = 0; iter < niter; iter++) {
        stencil_kernel<<<grid, block>>>(d_cur, d_nxt, d_power, nx, ny, nz);
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
