/*
 * hotspot3D_cuda.cu — CUDA port of hotspot3D thermal simulation.
 * Phase 1: basic parallelization (global memory only).
 *
 * Derived from openmp/hotspot3D/3D.c. Input generated deterministically
 * from srand(7); stencil update matches computeTempCPU exactly.
 *
 * CLI: <prog> <rows/cols> <layers> <iterations> <output_file>
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
 * Kernel: stencil update for one time step.
 * One thread per cell; reads from cur, writes to nxt.
 */
__global__ void stencil_kernel(
    const float *cur, float *nxt,
    const float *power,
    int nx, int ny, int nz,
    float cc, float cn, float cs, float ce, float cw,
    float ct, float cb, float dtCap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int c = x + y*nx + z*nx*ny;

    /* Neighbour indices with clamped boundary (mirror / no-flux) */
    int w = (x == 0)      ? c : c - 1;
    int e = (x == nx-1)   ? c : c + 1;
    int n = (y == 0)      ? c : c - nx;
    int s = (y == ny-1)   ? c : c + nx;
    int b = (z == 0)      ? c : c - nx*ny;
    int t = (z == nz-1)   ? c : c + nx*ny;

    /* Stencil update: verbatim from computeTempCPU */
    nxt[c] = cur[c]*cc + cur[n]*cn + cur[s]*cs
           + cur[e]*ce + cur[w]*cw
           + cur[t]*ct + cur[b]*cb
           + dtCap * power[c] + ct * amb_temp;
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
    float ce = stepDivCap / Rx;
    float cw = ce;
    float cn = stepDivCap / Ry;
    float cs = cn;
    float ct = stepDivCap / Rz;
    float cb = ct;
    float cc = 1.0f - (2.0f*ce + 2.0f*cn + 3.0f*ct);
    float dtCap = dt / Cap;

    int size = nx * ny * nz;

    /* --- host allocation and input generation --- */
    float *h_power = (float *)malloc(size * sizeof(float));
    float *h_buf0  = (float *)malloc(size * sizeof(float));
    if (!h_power || !h_buf0) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Generate input deterministically (srand(7)) */
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
        cudaMalloc(&d_buf0, size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_buf1, size * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        return 1;
    }

    /* Copy input data to device */
    if (cudaMemcpy(d_power, h_power, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_buf0, h_buf0, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (H2D) failed\n");
        return 1;
    }

    /* --- block and grid configuration --- */
    dim3 blockDim(16, 16, 1);  /* 256 threads per block */
    dim3 gridDim(
        (nx + blockDim.x - 1) / blockDim.x,
        (ny + blockDim.y - 1) / blockDim.y,
        (nz + blockDim.z - 1) / blockDim.z
    );

    /* --- time-stepping --- */
    float *d_cur = d_buf0, *d_nxt = d_buf1;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int iter = 0; iter < niter; iter++) {
        stencil_kernel<<<gridDim, blockDim>>>(
            d_cur, d_nxt, d_power,
            nx, ny, nz,
            cc, cn, cs, ce, cw, ct, cb, dtCap
        );
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "kernel launch failed at iteration %d\n", iter);
            return 1;
        }

        /* Swap buffers */
        float *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
    }

    /* Ensure kernel completes */
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        return 1;
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    fprintf(stderr, "compute_seconds: %.6f\n", elapsed);

    /* --- copy result back to host --- */
    float *h_result = (float *)malloc(size * sizeof(float));
    if (!h_result) {
        fprintf(stderr, "malloc failed for result\n");
        return 1;
    }
    if (cudaMemcpy(h_result, d_cur, size * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (D2H) failed\n");
        return 1;
    }

    /* --- write output --- */
    FILE *fp = fopen(outfile, "w");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", outfile);
        return 1;
    }
    int index = 0;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                fprintf(fp, "%d\t%g\n", index++, h_result[idx]);
            }
    fclose(fp);

    /* --- cleanup --- */
    free(h_power);
    free(h_buf0);
    free(h_result);
    cudaFree(d_power);
    cudaFree(d_buf0);
    cudaFree(d_buf1);

    return 0;
}
