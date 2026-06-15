#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Kernel parameters
float t_chip = 0.0005f;      // chip thickness in meters
float chip_height = 0.016f;  // chip height in meters
float chip_width = 0.016f;   // chip width in meters
float Ku = 100.0f;           // thermal conductivity (W/mK)
float K_t = 1e-5f;           // transient convection (W/mK)
float t_ambient = 80.0f;     // ambient temperature (Celsius)

__global__ void hotspot_kernel(float *temp_new, const float *temp_old,
                               const float *power,
                               int rows, int cols,
                               float Cap_1, float Rx_1, float Ry_1, float Rz_1)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols)
        return;

    // Shared memory for tile: (BLOCK_SIZE+2) x (BLOCK_SIZE+2) for halo
    extern __shared__ float tile[];
    int tile_cols = blockDim.x + 2;
    int local_row = threadIdx.y + 1;
    int local_col = threadIdx.x + 1;

    // Load center element
    tile[local_row * tile_cols + local_col] = temp_old[row * cols + col];

    // Load halo (north, south, east, west)
    // North halo
    if (threadIdx.y == 0) {
        int halo_row = row - 1;
        float val = (halo_row >= 0) ? temp_old[halo_row * cols + col] : temp_old[row * cols + col];
        tile[0 * tile_cols + local_col] = val;
    }
    // South halo
    if (threadIdx.y == blockDim.y - 1) {
        int halo_row = row + 1;
        float val = (halo_row < rows) ? temp_old[halo_row * cols + col] : temp_old[row * cols + col];
        tile[(blockDim.y + 1) * tile_cols + local_col] = val;
    }
    // West halo
    if (threadIdx.x == 0) {
        int halo_col = col - 1;
        float val = (halo_col >= 0) ? temp_old[row * cols + halo_col] : temp_old[row * cols + col];
        tile[local_row * tile_cols + 0] = val;
    }
    // East halo
    if (threadIdx.x == blockDim.x - 1) {
        int halo_col = col + 1;
        float val = (halo_col < cols) ? temp_old[row * cols + halo_col] : temp_old[row * cols + col];
        tile[local_row * tile_cols + (blockDim.x + 1)] = val;
    }

    __syncthreads();

    // Get neighbors from shared memory (using center value for out-of-domain)
    float center = tile[local_row * tile_cols + local_col];
    float north = tile[(local_row - 1) * tile_cols + local_col];
    float south = tile[(local_row + 1) * tile_cols + local_col];
    float west = tile[local_row * tile_cols + (local_col - 1)];
    float east = tile[local_row * tile_cols + (local_col + 1)];

    // Power from global memory
    float p = power[row * cols + col];

    // Update: delta = Cap_1 * (power + (west+east-2*center)*Rx_1 + (north+south-2*center)*Ry_1 + (amb_temp-center)*Rz_1)
    float delta = Cap_1 * (p + (west + east - 2.0f * center) * Rx_1 +
                           (north + south - 2.0f * center) * Ry_1 +
                           (t_ambient - center) * Rz_1);

    temp_new[row * cols + col] = center + delta;
}

int main(int argc, char *argv[])
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <power_file> <temp_file> <grid_rows> <grid_cols>\n", argv[0]);
        return 1;
    }

    const char *power_file = argv[1];
    const char *temp_file = argv[2];
    int grid_rows = atoi(argv[3]);
    int grid_cols = atoi(argv[4]);
    int n = grid_rows * grid_cols;

    // Read power (text, one float per line)
    float *power_h = (float *)malloc(n * sizeof(float));
    FILE *fp = fopen(power_file, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", power_file);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f", &power_h[i]) != 1) {
            fprintf(stderr, "Error reading power file at index %d\n", i);
            return 1;
        }
    }
    fclose(fp);

    // Read initial temp (text, one float per line)
    float *temp_h = (float *)malloc(n * sizeof(float));
    fp = fopen(temp_file, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", temp_file);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f", &temp_h[i]) != 1) {
            fprintf(stderr, "Error reading temp file at index %d\n", i);
            return 1;
        }
    }
    fclose(fp);

    // Compute coefficients
    float grid_height = chip_height / grid_rows;
    float grid_width = chip_width / grid_cols;
    float Cap = grid_width * grid_height * chip_height * 846.0f / 1e6f;
    float Rx = grid_width / (2.0f * Ku * grid_height * t_chip);
    float Ry = grid_height / (2.0f * Ku * grid_width * t_chip);
    float Rz = t_chip / (Ku * grid_width * grid_height);

    float max_slope = (Ku / (846.0f * grid_height) - (grid_width * grid_height) / (2.0f * t_chip * (Rx + Ry)));
    float PRECISION = 0.001f;
    float step = PRECISION / max_slope / 1000.0f;

    float Cap_1 = step / Cap;
    float Rx_1 = step / Rx;
    float Ry_1 = step / Ry;
    float Rz_1 = step / Rz;

    // Allocate device memory
    float *power_d, *temp_old_d, *temp_new_d;
    cudaMalloc(&power_d, n * sizeof(float));
    cudaMalloc(&temp_old_d, n * sizeof(float));
    cudaMalloc(&temp_new_d, n * sizeof(float));

    cudaMemcpy(power_d, power_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(temp_old_d, temp_h, n * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((grid_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (grid_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int shared_size = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(float);

    // Run kernel (single iteration as specified)
    hotspot_kernel<<<grid, block, shared_size>>>(temp_new_d, temp_old_d, power_d,
                                                   grid_rows, grid_cols,
                                                   Cap_1, Rx_1, Ry_1, Rz_1);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(temp_h, temp_new_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Output as TEXT: index<tab>temperature
    for (int i = 0; i < n; i++) {
        printf("%d\t%g\n", i, temp_h[i]);
    }

    // Cleanup
    free(power_h);
    free(temp_h);
    cudaFree(power_d);
    cudaFree(temp_old_d);
    cudaFree(temp_new_d);

    return 0;
}
