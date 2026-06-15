/*
 * Hotspot – phase-2 round-4
 * Naive one-thread-per-cell global-memory kernel.
 * Output format: "%d\t%g\n" per cell (index TAB value).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* Physical constants matching Rodinia hotspot */
#define SPEC_HEAT_SI  703.0f
#define K_SI          100.0f
#define FACTOR_CHIP   0.5f
#define T_CHIP        0.0005f
#define CHIP_HEIGHT   0.016f
#define CHIP_WIDTH    0.016f
#define AMB_TEMP      80.0f
#define PRECISION_VAL 0.001f   /* the numeric constant called PRECISION in Rodinia */

/* ── kernel ── */
__global__ void hotspot_step(
    const float * __restrict__ temp_in,
    const float * __restrict__ power,
          float * __restrict__ temp_out,
    int rows, int cols,
    float Cap_1,
    float Rx_1, float Ry_1, float Rz_1)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;

    int idx    = row * cols + col;
    float ctr  = temp_in[idx];

    float left  = (col > 0)        ? temp_in[idx - 1]    : ctr;
    float right = (col < cols - 1) ? temp_in[idx + 1]    : ctr;
    float up    = (row > 0)        ? temp_in[idx - cols]  : ctr;
    float down  = (row < rows - 1) ? temp_in[idx + cols]  : ctr;

    float delta = Cap_1 * (
          power[idx]
        + (left  + right - 2.0f * ctr) * Rx_1
        + (up    + down  - 2.0f * ctr) * Ry_1
        + (AMB_TEMP - ctr)             * Rz_1
    );

    temp_out[idx] = ctr + delta;
}

/* ── I/O helpers ── */
static void read_matrix(const char *fname, float *buf, int n) {
    FILE *f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%f", &buf[i]) != 1) {
            fprintf(stderr, "Read error at %d in %s\n", i, fname); exit(1);
        }
    }
    fclose(f);
}

static void write_output(const char *fname, float *buf, int n) {
    FILE *f = fopen(fname, "w");
    if (!f) { fprintf(stderr, "Cannot open %s for write\n", fname); exit(1); }
    for (int i = 0; i < n; i++)
        fprintf(f, "%d\t%g\n", i, buf[i]);
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr,
            "Usage: %s <grid_rows> <grid_cols> <sim_time> "
            "<temp_file> <power_file> <output_file>\n", argv[0]);
        return 1;
    }

    int rows     = atoi(argv[1]);
    int cols     = atoi(argv[2]);
    int sim_time = atoi(argv[3]);
    const char *temp_file   = argv[4];
    const char *power_file  = argv[5];
    const char *output_file = argv[6];

    int n    = rows * cols;
    size_t sz = (size_t)n * sizeof(float);

    /* ── derive coefficients (Rodinia reference) ── */
    float dx  = CHIP_HEIGHT / rows;
    float dy  = CHIP_WIDTH  / cols;
    float dz  = T_CHIP;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * K_SI * dx * dy * dz;
    float Rx  = dy / (2.0f * K_SI * dx * dz);
    float Ry  = dx / (2.0f * K_SI * dy * dz);
    float Rz  = dz / (K_SI * dx * dy);

    /* max_slope and step exactly as in Rodinia hotspot.cu */
    float max_slope = Cap / (0.5f * (Rx + Ry) / ((Rx * Ry) / (Rx + Ry)));
    float step = PRECISION_VAL / max_slope / 1000.0f;

    float Cap_1 = step / Cap;
    float Rx_1  = 1.0f / Rx;
    float Ry_1  = 1.0f / Ry;
    float Rz_1  = 1.0f / Rz;

    /* ── host buffers ── */
    float *h_temp  = (float*)malloc(sz);
    float *h_power = (float*)malloc(sz);
    read_matrix(temp_file,  h_temp,  n);
    read_matrix(power_file, h_power, n);

    /* ── device buffers (ping-pong) ── */
    float *d_temp[2], *d_power;
    cudaMalloc(&d_temp[0], sz);
    cudaMalloc(&d_temp[1], sz);
    cudaMalloc(&d_power,   sz);

    cudaMemcpy(d_temp[0], h_temp,  sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_power,   h_power, sz, cudaMemcpyHostToDevice);

    dim3 block(32, 8);
    dim3 grid((cols + 31) / 32, (rows + 7) / 8);

    int src = 0, dst = 1;
    for (int t = 0; t < sim_time; t++) {
        hotspot_step<<<grid, block>>>(
            d_temp[src], d_power, d_temp[dst],
            rows, cols, Cap_1, Rx_1, Ry_1, Rz_1);
        src ^= 1; dst ^= 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_temp, d_temp[src], sz, cudaMemcpyDeviceToHost);
    write_output(output_file, h_temp, n);

    free(h_temp); free(h_power);
    cudaFree(d_temp[0]); cudaFree(d_temp[1]); cudaFree(d_power);
    return 0;
}
