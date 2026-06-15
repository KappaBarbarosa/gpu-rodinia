#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BX 32
#define BY 32
#define FACTOR_CHIP 0.5f
#define SPEC_HEAT_SI 1.75e6f
#define K_SI 100.0f
#define MAX_PD 3.0e6f
#define PRECISION 0.001f

__global__ void compute_tran_temp_kernel(
    const float *power,
    const float *temp_in,
    float *temp_out,
    int num_rows,
    int num_cols,
    float Cap_1,
    float Rx_1,
    float Ry_1,
    float Rz_1,
    float amb_temp
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * BX + tx;
    int row = blockIdx.y * BY + ty;

    // Shared memory with halo: (BY+2) x (BX+2)
    __shared__ float s[BY+2][BX+2];
    int sr = ty + 1;
    int sc = tx + 1;

    // Load center cell
    if (row < num_rows && col < num_cols) {
        int idx = row * num_cols + col;
        s[sr][sc] = temp_in[idx];
    } else {
        s[sr][sc] = amb_temp;
    }

    // Load halos with bounds checking
    if (ty == 0) {  // top halo (row-1)
        int halo_row = row - 1;
        if (halo_row >= 0 && col < num_cols) {
            int idx = halo_row * num_cols + col;
            s[0][sc] = temp_in[idx];
        } else {
            s[0][sc] = amb_temp;
        }
    }
    if (ty == BY - 1) {  // bottom halo (row+1)
        int halo_row = row + 1;
        if (halo_row < num_rows && col < num_cols) {
            int idx = halo_row * num_cols + col;
            s[BY+1][sc] = temp_in[idx];
        } else {
            s[BY+1][sc] = amb_temp;
        }
    }
    if (tx == 0) {  // left halo (col-1)
        int halo_col = col - 1;
        if (row < num_rows && halo_col >= 0) {
            int idx = row * num_cols + halo_col;
            s[sr][0] = temp_in[idx];
        } else {
            s[sr][0] = amb_temp;
        }
    }
    if (tx == BX - 1) {  // right halo (col+1)
        int halo_col = col + 1;
        if (row < num_rows && halo_col < num_cols) {
            int idx = row * num_cols + halo_col;
            s[sr][BX+1] = temp_in[idx];
        } else {
            s[sr][BX+1] = amb_temp;
        }
    }

    __syncthreads();

    // Compute if in bounds
    if (row < num_rows && col < num_cols) {
        int idx = row * num_cols + col;
        float center = s[sr][sc];
        float north = s[sr-1][sc];
        float south = s[sr+1][sc];
        float west = s[sr][sc-1];
        float east = s[sr][sc+1];

        float delta = Cap_1 * (
            power[idx] +
            (west + east - 2.0f * center) * Rx_1 +
            (north + south - 2.0f * center) * Ry_1 +
            (amb_temp - center) * Rz_1
        );

        temp_out[idx] = center + delta;
    }
}

void compute_tran_temp(
    int chip_height,
    int chip_width,
    int grid_rows,
    int grid_cols,
    int sim_time,
    float *power,
    float *temp_in,
    float *temp_out,
    float *temp_device,
    float *power_device,
    float amb_temp
) {
    // Coefficients
    float grid_height = chip_height / (float)grid_rows;
    float grid_width = chip_width / (float)grid_cols;
    float t_chip = 0.0005f;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0f * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0f * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0f;

    float Cap_1 = step / Cap;
    float Rx_1 = 1.0f / Rx;
    float Ry_1 = 1.0f / Ry;
    float Rz_1 = 1.0f / Rz;

    int total_cells = grid_rows * grid_cols;
    size_t size = total_cells * sizeof(float);

    // Copy initial temp and power to device
    cudaMemcpy(temp_device, temp_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(power_device, power, size, cudaMemcpyHostToDevice);

    float *device_temp_ping = temp_device;
    float *device_temp_pong = temp_device + total_cells;

    dim3 block(BX, BY);
    dim3 grid((grid_cols + BX - 1) / BX, (grid_rows + BY - 1) / BY);

    // Simulation loop
    for (int iter = 0; iter < sim_time; iter++) {
        compute_tran_temp_kernel<<<grid, block>>>(
            power_device,
            device_temp_ping,
            device_temp_pong,
            grid_rows,
            grid_cols,
            Cap_1,
            Rx_1,
            Ry_1,
            Rz_1,
            amb_temp
        );

        // Swap buffers
        float *temp = device_temp_ping;
        device_temp_ping = device_temp_pong;
        device_temp_pong = temp;
    }

    // Copy result back
    cudaMemcpy(temp_out, device_temp_ping, size, cudaMemcpyDeviceToHost);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
        return 1;
    }

    int grid_rows = atoi(argv[1]);
    int grid_cols = atoi(argv[2]);
    int sim_time = atoi(argv[3]);
    char *temp_file = argv[4];
    char *power_file = argv[5];
    char *output_file = argv[6];

    int total_cells = grid_rows * grid_cols;
    size_t size = total_cells * sizeof(float);

    float *temp_in = (float *)malloc(size);
    float *power = (float *)malloc(size);
    float *temp_out = (float *)malloc(size);

    // Read input files
    FILE *fp = fopen(temp_file, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", temp_file);
        return 1;
    }
    fread(temp_in, sizeof(float), total_cells, fp);
    fclose(fp);

    fp = fopen(power_file, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", power_file);
        return 1;
    }
    fread(power, sizeof(float), total_cells, fp);
    fclose(fp);

    // Device memory: temp_in, temp_out, power
    float *temp_device = NULL;
    float *power_device = NULL;
    cudaMalloc(&temp_device, 2 * size);
    cudaMalloc(&power_device, size);

    float amb_temp = 80.0f;

    compute_tran_temp(
        16, 16,  // chip_height, chip_width (in mm, default 0.016m = 16mm)
        grid_rows, grid_cols,
        sim_time,
        power, temp_in, temp_out,
        temp_device, power_device,
        amb_temp
    );

    // Write output
    fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open %s for writing\n", output_file);
        return 1;
    }
    for (int i = 0; i < total_cells; i++) {
        fprintf(fp, "%d\t%g\n", i, temp_out[i]);
    }
    fclose(fp);

    cudaFree(temp_device);
    cudaFree(power_device);
    free(temp_in);
    free(power);
    free(temp_out);

    return 0;
}
