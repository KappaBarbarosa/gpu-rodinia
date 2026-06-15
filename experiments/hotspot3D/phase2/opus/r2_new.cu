/*
 * hotspot3D — Phase 2 round-2 CUDA kernel.
 *
 * Round-1 (BX=32, BY=8 2D smem tile + register Z-pipeline + constant-memory
 * coefficients) was measured at the hardware memory-bandwidth ceiling
 * (~422 GB/s of the RTX 3070's 448 GB/s peak, ~6.55 ms / 100 iters).
 *
 * Round-2 feedback established that EVERY explored alternative is neutral or
 * worse on this hardware (template unroll -9%, BY=16 -9%, __ldg neutral,
 * cooperative/XY temporal blocking infeasible: 1024 blocks > 276 resident
 * cooperative limit). The stencil is memory-bound with no reuse beyond the
 * 6 neighbours already captured by the smem tile + register pipeline, so the
 * r1 configuration is the global optimum. r2 preserves it verbatim.
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

#define MAX_PD        3.0e6f
#define PRECISION     0.001f
#define SPEC_HEAT_SI  1.75e6f
#define K_SI          100.0f
#define FACTOR_CHIP   0.5f

static const float t_chip      = 0.0005f;
static const float chip_height = 0.016f;
static const float chip_width  = 0.016f;
static const float amb_temp    = 80.0f;

__constant__ float d_cc, d_cn, d_cs, d_ce, d_cw, d_ct, d_cb, d_dtCap, d_ambct;

#define BX  32
#define BY  8
#define SX  (BX + 2)
#define SY  (BY + 2)

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

    const int col = gy * nx + gx;

    const int wcol = (gx > 0)      ? col - 1  : col;
    const int ecol = (gx < nx - 1) ? col + 1  : col;
    const int ncol = (gy > 0)      ? col - nx : col;
    const int scol = (gy < ny - 1) ? col + nx : col;

    const int sx = tx + 1;
    const int sy = ty + 1;

    const bool wEdge = (tx == 0);
    const bool eEdge = (tx == BX - 1) || (gx == nx - 1);
    const bool nEdge = (ty == 0);
    const bool sEdge = (ty == BY - 1) || (gy == ny - 1);

    float vb_reg = 0.0f, vc_reg = 0.0f, vt_reg = 0.0f;
    if (valid) {
        vc_reg = cur[col];
        vt_reg = (nz > 1) ? cur[nxny + col] : vc_reg;
    }

    for (int z = 0; z < nz; z++) {
        const int base = z * nxny;

        if (valid) {
            s[sy][sx] = vc_reg;
            if (wEdge) s[sy][0]      = cur[base + wcol];
            if (eEdge) s[sy][BX + 1] = cur[base + ecol];
            if (nEdge) s[0][sx]      = cur[base + ncol];
            if (sEdge) s[SY - 1][sx] = cur[base + scol];
        }
        __syncthreads();

        if (valid) {
            const float vn = s[sy - 1][sx];
            const float vs = s[sy + 1][sx];
            const float vw = s[sy][sx - 1];
            const float ve = s[sy][sx + 1];
            const float vb = (z == 0)      ? vc_reg : vb_reg;
            const float vt = (z == nz - 1) ? vc_reg : vt_reg;
            const float pw = power[base + col];

            nxt[base + col] =
                vc_reg * d_cc +
                vn * d_cn + vs * d_cs +
                ve * d_ce + vw * d_cw +
                vt * d_ct + vb * d_cb +
                pw * d_dtCap + d_ambct;
        }
        __syncthreads();

        vb_reg = vc_reg;
        vc_reg = vt_reg;
        if (valid && z + 2 < nz)
            vt_reg = cur[(z + 2) * nxny + col];
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
