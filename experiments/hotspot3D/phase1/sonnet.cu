/*
 * hotspot3D CUDA phase-1 — global memory only, straightforward parallelization.
 * Each CUDA thread handles one cell; all cells in a time step are independent.
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

__global__ void stencil_kernel(
    const float * __restrict__ cur,
    float       * __restrict__ nxt,
    const float * __restrict__ power,
    int nx, int ny, int nz,
    float cc, float cn, float cs, float ce, float cw,
    float ct, float cb, float dtCap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int c = x + y * nx + z * nx * ny;
    int w = (x == 0)      ? c : c - 1;
    int e = (x == nx - 1) ? c : c + 1;
    int n = (y == 0)      ? c : c - nx;
    int s = (y == ny - 1) ? c : c + nx;
    int b = (z == 0)      ? c : c - nx * ny;
    int t = (z == nz - 1) ? c : c + nx * ny;

    nxt[c] = cur[c] * cc
           + cur[n] * cn + cur[s] * cs
           + cur[e] * ce + cur[w] * cw
           + cur[t] * ct + cur[b] * cb
           + dtCap * power[c] + ct * amb_temp;
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

    /* Derived coefficients */
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

    int size = nx * ny * nz;
    float *power = (float *)malloc(size * sizeof(float));
    float *buf0  = (float *)malloc(size * sizeof(float));
    if (!power || !buf0) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* Generate input deterministically */
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

    /* Launch config: 2D block in xy, 1 in z */
    dim3 block(16, 16, 1);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              nz);

    float *d_cur = d_buf0, *d_nxt = d_buf1;

    /* Time only the stencil computation */
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int iter = 0; iter < niter; iter++) {
        stencil_kernel<<<grid, block>>>(d_cur, d_nxt, d_power,
                                        nx, ny, nz,
                                        cc, cn, cs, ce, cw, ct, cb, dtCap);
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
