/*
 * Hotspot Phase-2 Round-3: naive global-memory kernel with __ldg, __restrict__,
 * combined neighbour terms, 32x8 block shape for coalescing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define PRECISION   double
#define SPEC_HEAT_SI 1.0106976e6
#define K_SI         100.0
#define FACTOR_CHIP  0.5
#define SPEC_HEAT_CHIP (SPEC_HEAT_SI * FACTOR_CHIP)
#define K_CHIP       (K_SI * FACTOR_CHIP)
#define T_CHIP       0.0005
#define CHIP_HEIGHT  0.016
#define CHIP_WIDTH   0.016
#define AMB_TEMP     80.0

static void read_input(PRECISION *in, int size, const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%lf", &in[i]) != 1) {
            fprintf(stderr, "Read error at %d\n", i); exit(1);
        }
    }
    fclose(f);
}

static void write_output(PRECISION *out, int size, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
    for (int i = 0; i < size; i++)
        fprintf(f, "%.6f\n", out[i]);
    fclose(f);
}

// Naive global-memory kernel: one thread per cell, combined terms, __ldg, __restrict__
__global__ void hotspot_kernel(
    const PRECISION * __restrict__ temp,
    const PRECISION * __restrict__ power,
    PRECISION       * __restrict__ result,
    int rows, int cols,
    PRECISION Rx_1, PRECISION Ry_1, PRECISION Rz_1, PRECISION step)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;

    int idx = row * cols + col;
    PRECISION self = __ldg(&temp[idx]);

    // East/West (column direction)
    PRECISION left  = (col > 0)        ? __ldg(&temp[idx - 1])    : self;
    PRECISION right = (col < cols - 1) ? __ldg(&temp[idx + 1])    : self;
    // North/South (row direction)
    PRECISION up    = (row > 0)        ? __ldg(&temp[idx - cols])  : self;
    PRECISION down  = (row < rows - 1) ? __ldg(&temp[idx + cols])  : self;

    PRECISION pwr = __ldg(&power[idx]);

    PRECISION delta = step * (
        (left + right - 2.0 * self) * Rx_1 +
        (up   + down  - 2.0 * self) * Ry_1 +
        (AMB_TEMP - self)           * Rz_1 +
        pwr
    );

    result[idx] = self + delta;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no_of_blocks> <block_size> <input_temp> <input_power> -- but expected 6 args\n", argv[0]);
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
        return 1;
    }

    int rows     = atoi(argv[1]);
    int cols     = atoi(argv[2]);
    int sim_time = atoi(argv[3]);
    const char *temp_file  = argv[4];
    const char *power_file = argv[5];
    const char *out_file   = argv[6];

    int size = rows * cols;

    PRECISION *temp_h  = (PRECISION*)malloc(size * sizeof(PRECISION));
    PRECISION *power_h = (PRECISION*)malloc(size * sizeof(PRECISION));
    PRECISION *out_h   = (PRECISION*)malloc(size * sizeof(PRECISION));

    read_input(temp_h,  size, temp_file);
    read_input(power_h, size, power_file);

    // Coefficient derivation (match reference exactly)
    PRECISION dx = CHIP_WIDTH  / (PRECISION)cols;
    PRECISION dy = CHIP_HEIGHT / (PRECISION)rows;
    PRECISION dz = T_CHIP;

    PRECISION Cap = SPEC_HEAT_CHIP * K_SI * dx * dy * dz;
    PRECISION Rx  = dy / (2.0 * K_CHIP * dx * dz);
    PRECISION Ry  = dx / (2.0 * K_CHIP * dy * dz);
    PRECISION Rz  = dz / (K_CHIP * dx * dy);

    PRECISION max_slope = Cap / (fmin(fmin(Rx, Ry), Rz) * Cap * Cap);
    PRECISION step = PRECISION(PRECISION(1.0) / max_slope) / 1000.0;

    PRECISION Rx_1 = 1.0 / (Rx * Cap);
    PRECISION Ry_1 = 1.0 / (Ry * Cap);
    PRECISION Rz_1 = 1.0 / (Rz * Cap);
    PRECISION step_over_cap = step;  // step already incorporates 1/Cap via max_slope

    // Device allocation
    PRECISION *d_temp0, *d_temp1, *d_power;
    cudaMalloc(&d_temp0, size * sizeof(PRECISION));
    cudaMalloc(&d_temp1, size * sizeof(PRECISION));
    cudaMalloc(&d_power, size * sizeof(PRECISION));

    cudaMemcpy(d_temp0, temp_h,  size * sizeof(PRECISION), cudaMemcpyHostToDevice);
    cudaMemcpy(d_power, power_h, size * sizeof(PRECISION), cudaMemcpyHostToDevice);

    // 32x8 block: threads coalesce along x (columns)
    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    PRECISION *src = d_temp0, *dst = d_temp1;

    for (int t = 0; t < sim_time; t++) {
        hotspot_kernel<<<grid, block>>>(src, d_power, dst, rows, cols,
                                        Rx_1, Ry_1, Rz_1, step_over_cap);
        // ping-pong
        PRECISION *tmp = src; src = dst; dst = tmp;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(out_h, src, size * sizeof(PRECISION), cudaMemcpyDeviceToHost);

    write_output(out_h, size, out_file);

    free(temp_h); free(power_h); free(out_h);
    cudaFree(d_temp0); cudaFree(d_temp1); cudaFree(d_power);
    return 0;
}
