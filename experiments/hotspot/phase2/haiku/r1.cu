#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

typedef float FLOAT;

#define BLOCK_X 32
#define BLOCK_Y 32
#define HALO 1

__global__ void single_iteration_kernel_tiled(
    FLOAT *result, FLOAT *temp, FLOAT *power,
    int row, int col, FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1)
{
    __shared__ FLOAT s_temp[(BLOCK_Y + 2 * HALO) * (BLOCK_X + 2 * HALO)];

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int tr = threadIdx.y;
    int tc = threadIdx.x;
    int s_col = BLOCK_X + 2 * HALO;

    if (r < row && c < col) {
        s_temp[(tr + HALO) * s_col + (tc + HALO)] = temp[r * col + c];
    }
    if (tr < HALO && r >= HALO && c < col) {
        s_temp[tr * s_col + (tc + HALO)] = temp[(r - HALO) * col + c];
    }
    if (tr + BLOCK_Y < BLOCK_Y + HALO && r + HALO < row && c < col) {
        s_temp[(tr + BLOCK_Y + HALO) * s_col + (tc + HALO)] = temp[(r + BLOCK_Y) * col + c];
    }
    if (tc < HALO && r < row && c >= HALO) {
        s_temp[(tr + HALO) * s_col + tc] = temp[r * col + (c - HALO)];
    }
    if (tc + BLOCK_X < BLOCK_X + HALO && r < row && c + BLOCK_X < col) {
        s_temp[(tr + HALO) * s_col + (tc + BLOCK_X + HALO)] = temp[r * col + (c + BLOCK_X)];
    }

    __syncthreads();

    if (r >= row || c >= col) return;

    int s_idx = (tr + HALO) * s_col + (tc + HALO);
    FLOAT t_center = s_temp[s_idx];
    FLOAT t_right = s_temp[s_idx + 1];
    FLOAT t_left = s_temp[s_idx - 1];
    FLOAT t_down = s_temp[s_idx + s_col];
    FLOAT t_up = s_temp[s_idx - s_col];

    FLOAT delta = 0.0f;

    if (r == 0 && c == 0) {
        delta = (Rx_1 * (t_right - t_center) + Ry_1 * (t_down - t_center) + Rz_1 * (t_center - t_center)) * Cap_1 + power[r * col + c];
    } else if (r == 0 && c == col - 1) {
        delta = (Rx_1 * (t_left - t_center) + Ry_1 * (t_down - t_center) + Rz_1 * (t_center - t_center)) * Cap_1 + power[r * col + c];
    } else if (r == row - 1 && c == 0) {
        delta = (Rx_1 * (t_right - t_center) + Ry_1 * (t_up - t_center) + Rz_1 * (t_center - t_center)) * Cap_1 + power[r * col + c];
    } else if (r == row - 1 && c == col - 1) {
        delta = (Rx_1 * (t_left - t_center) + Ry_1 * (t_up - t_center) + Rz_1 * (t_center - t_center)) * Cap_1 + power[r * col + c];
    } else if (r == 0) {
        delta = (Rx_1 * (t_right + t_left - 2.0f * t_center) + Ry_1 * (t_down - t_center) + Rz_1 * (t_center - t_center)) * Cap_1 + power[r * col + c];
    } else if (r == row - 1) {
        delta = (Rx_1 * (t_right + t_left - 2.0f * t_center) + Ry_1 * (t_up - t_center) + Rz_1 * (t_center - t_center)) * Cap_1 + power[r * col + c];
    } else if (c == 0) {
        delta = (Rx_1 * (t_right - t_center) + Ry_1 * (t_down + t_up - 2.0f * t_center) + Rz_1 * (t_center - t_center)) * Cap_1 + power[r * col + c];
    } else if (c == col - 1) {
        delta = (Rx_1 * (t_left - t_center) + Ry_1 * (t_down + t_up - 2.0f * t_center) + Rz_1 * (t_center - t_center)) * Cap_1 + power[r * col + c];
    } else {
        delta = (Rx_1 * (t_right + t_left - 2.0f * t_center) + Ry_1 * (t_down + t_up - 2.0f * t_center) + Rz_1 * (t_center - t_center)) * Cap_1 + power[r * col + c];
    }

    result[r * col + c] = t_center + delta;
}

void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power, int row, int col)
{
    FLOAT Cap = 1395100.0f;
    FLOAT Rx = 0.000012f;
    FLOAT Ry = 0.000012f;
    FLOAT Rz = 0.000012f;
    FLOAT Cap_1 = 1.0f / Cap;
    FLOAT Rx_1 = Rx + Rx;
    FLOAT Ry_1 = Ry + Ry;
    FLOAT Rz_1 = Rz + Rz;

    size_t size = row * col * sizeof(FLOAT);
    FLOAT *d_temp, *d_result, *d_power;

    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_result, size);
    cudaMalloc(&d_power, size);

    cudaMemcpy(d_temp, temp, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_power, power, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((col + BLOCK_X - 1) / BLOCK_X, (row + BLOCK_Y - 1) / BLOCK_Y);

    for (int i = 0; i < num_iterations; i++) {
        single_iteration_kernel_tiled<<<grid, block>>>(d_result, d_temp, d_power, row, col, Cap_1, Rx_1, Ry_1, Rz_1);
        FLOAT *tmp = d_temp;
        d_temp = d_result;
        d_result = tmp;
    }

    cudaDeviceSynchronize();

    FLOAT *final = (num_iterations & 1) ? d_result : d_temp;
    cudaMemcpy(result, final, size, cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_result);
    cudaFree(d_power);
}

void writeoutput(FLOAT *result, int row, int col, FILE *fp) {
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            fprintf(fp, "%d\t%g\n", r * col + c, result[r * col + c]);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
        return 1;
    }

    int row = atoi(argv[1]);
    int col = atoi(argv[2]);
    int sim_time = atoi(argv[3]);
    const char *temp_file = argv[4];
    const char *power_file = argv[5];
    const char *output_file = argv[6];

    size_t size = row * col * sizeof(FLOAT);
    FLOAT *temp = (FLOAT *)malloc(size);
    FLOAT *power = (FLOAT *)malloc(size);
    FLOAT *result = (FLOAT *)malloc(size);

    FILE *fp = fopen(temp_file, "r");
    for (int i = 0; i < row * col; i++) {
        fscanf(fp, "%f", &temp[i]);
    }
    fclose(fp);

    fp = fopen(power_file, "r");
    for (int i = 0; i < row * col; i++) {
        fscanf(fp, "%f", &power[i]);
    }
    fclose(fp);

    compute_tran_temp(result, sim_time, temp, power, row, col);

    fp = fopen(output_file, "w");
    writeoutput(result, row, col, fp);
    fclose(fp);

    free(temp);
    free(power);
    free(result);

    return 0;
}
