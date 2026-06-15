#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void hotspot_kernel(
    float *power,
    float *temp,
    float *temp_out,
    int nx, int ny,
    float Cap_1, float Rx_1, float Ry_1, float Rz_1,
    float amb)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < ny && j < nx) {
        int idx = i * nx + j;

        float center = temp[idx];
        float p = power[idx];

        float west = (j > 0) ? temp[idx - 1] : center;
        float east = (j < nx - 1) ? temp[idx + 1] : center;
        float north = (i > 0) ? temp[idx - nx] : center;
        float south = (i < ny - 1) ? temp[idx + nx] : center;

        float delta = Cap_1 * (
            p +
            (west + east - 2.0f * center) * Rx_1 +
            (north + south - 2.0f * center) * Ry_1 +
            (amb - center) * Rz_1
        );

        temp_out[idx] = center + delta;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file>\n", argv[0]);
        exit(1);
    }

    int grid_rows = atoi(argv[1]);
    int grid_cols = atoi(argv[2]);
    int sim_time = atoi(argv[3]);
    char *temp_file = argv[4];

    int total_cells = grid_rows * grid_cols;

    // Host arrays
    float *h_power = (float *)malloc(total_cells * sizeof(float));
    float *h_temp = (float *)malloc(total_cells * sizeof(float));
    float *h_temp_out = (float *)malloc(total_cells * sizeof(float));

    // Read temperature from file
    FILE *fp = fopen(temp_file, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", temp_file);
        exit(1);
    }

    for (int i = 0; i < total_cells; i++) {
        if (fscanf(fp, "%f", &h_temp[i]) != 1) {
            fprintf(stderr, "Error reading temperature file\n");
            exit(1);
        }
    }
    fclose(fp);

    // Initialize power to zero
    for (int i = 0; i < total_cells; i++) {
        h_power[i] = 0.0f;
    }

    // Physical parameters
    float t_chip = 0.0005f;
    float chip_width = 0.016f;
    float chip_height = 0.016f;
    float t_ambient = 80.0f;

    // Thermal resistance and capacitance
    float K = 100.0f;
    float d = t_chip;
    float A = chip_width * chip_height;
    float Cap = 3.287e6f * t_chip * A;

    float Rx = d / (2.0f * K * chip_height * (chip_width / grid_cols));
    float Ry = d / (2.0f * K * chip_width * (chip_height / grid_rows));
    float Rz = d / (K * A);

    float max_slope = 0.0f;
    for (int i = 0; i < total_cells; i++) {
        max_slope = fmaxf(max_slope, fabs(h_temp[i] - t_ambient));
    }
    if (max_slope == 0.0f) max_slope = 1.0f;

    float step = 0.001f / max_slope / 1000.0f;
    float PRECISION = 0.001f;
    step = PRECISION / max_slope / 1000.0f;

    float Cap_1 = step / Cap;
    float Rx_1 = step / Rx;
    float Ry_1 = step / Ry;
    float Rz_1 = step / Rz;

    // Device arrays
    float *d_power, *d_temp, *d_temp_out;
    cudaMalloc(&d_power, total_cells * sizeof(float));
    cudaMalloc(&d_temp, total_cells * sizeof(float));
    cudaMalloc(&d_temp_out, total_cells * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_power, h_power, total_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, h_temp, total_cells * sizeof(float), cudaMemcpyHostToDevice);

    // Grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((grid_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (grid_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Main simulation loop
    int iterations = sim_time;
    for (int iter = 0; iter < iterations; iter++) {
        hotspot_kernel<<<grid, block>>>(
            d_power, d_temp, d_temp_out,
            grid_cols, grid_rows,
            Cap_1, Rx_1, Ry_1, Rz_1,
            t_ambient);

        // Swap buffers
        float *temp = d_temp;
        d_temp = d_temp_out;
        d_temp_out = temp;
    }

    // Single synchronization before final copy
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_temp, d_temp, total_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // Output results
    for (int i = 0; i < total_cells; i++) {
        printf("%d\t%g\n", i, h_temp[i]);
    }

    // Cleanup
    free(h_power);
    free(h_temp);
    free(h_temp_out);
    cudaFree(d_power);
    cudaFree(d_temp);
    cudaFree(d_temp_out);

    return 0;
}
