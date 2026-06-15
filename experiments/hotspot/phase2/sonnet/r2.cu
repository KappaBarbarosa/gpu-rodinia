#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define FACTOR_CHIP   0.5f
#define SPEC_HEAT_SI  1.75e6f
#define K_SI          100.0f
#define t_chip        0.0005f
#define chip_height   0.016f
#define chip_width    0.016f
#define AMB_TEMP      80.0f
#define MAX_PD        3.0e6f
#define PRECISION     0.001f

#define TILE_W 32
#define TILE_H 32

__global__ void hotspot_kernel(float* __restrict__ result,
                                const float* __restrict__ temp,
                                const float* __restrict__ power,
                                int row, int col,
                                float Cap_1, float Rx_1, float Ry_1, float Rz_1,
                                float amb_temp)
{
    __shared__ float s_temp[(TILE_H+2)][(TILE_W+2)];

    int c  = blockIdx.x * TILE_W + threadIdx.x;
    int r  = blockIdx.y * TILE_H + threadIdx.y;
    int sc = threadIdx.x + 1;
    int sr = threadIdx.y + 1;

    float t_self = 0.0f;
    if (r < row && c < col) {
        t_self = temp[r * col + c];
        s_temp[sr][sc] = t_self;
    }

    // Load halo cells
    if (threadIdx.x == 0) {
        int hc = c - 1;
        s_temp[sr][0] = (hc >= 0 && r < row) ? temp[r * col + hc] : 0.0f;
    }
    if (threadIdx.x == TILE_W - 1) {
        int hc = c + 1;
        s_temp[sr][TILE_W+1] = (hc < col && r < row) ? temp[r * col + hc] : 0.0f;
    }
    if (threadIdx.y == 0) {
        int hr = r - 1;
        s_temp[0][sc] = (hr >= 0 && c < col) ? temp[hr * col + c] : 0.0f;
    }
    if (threadIdx.y == TILE_H - 1) {
        int hr = r + 1;
        s_temp[TILE_H+1][sc] = (hr < row && c < col) ? temp[hr * col + c] : 0.0f;
    }

    __syncthreads();

    if (r >= row || c >= col) return;

    int idx = r * col + c;

    float t_N = (r > 0)       ? s_temp[sr-1][sc] : t_self;
    float t_S = (r < row - 1) ? s_temp[sr+1][sc] : t_self;
    float t_W = (c > 0)       ? s_temp[sr][sc-1] : t_self;
    float t_E = (c < col - 1) ? s_temp[sr][sc+1] : t_self;

    float delta = Cap_1 * (power[idx]
                  + (t_S - t_self) * Ry_1
                  + (t_N - t_self) * Ry_1
                  + (t_W - t_self) * Rx_1
                  + (t_E - t_self) * Rx_1
                  + (amb_temp - t_self) * Rz_1);

    result[idx] = t_self + delta;
}

static void read_matrix(const char* fname, float* arr, int n)
{
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%f", &arr[i]) != 1) {
            fprintf(stderr, "Read error %s at %d\n", fname, i); exit(1);
        }
    }
    fclose(f);
}

static void write_output(const char* fname, float* arr, int n)
{
    FILE* f = fopen(fname, "w");
    if (!f) { fprintf(stderr, "Cannot open %s for write\n", fname); exit(1); }
    for (int i = 0; i < n; i++)
        fprintf(f, "%d\t%g\n", i, arr[i]);
    fclose(f);
}

int main(int argc, char** argv)
{
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
        return 1;
    }

    int row      = atoi(argv[1]);
    int col      = atoi(argv[2]);
    int sim_time = atoi(argv[3]);
    const char* temp_file   = argv[4];
    const char* power_file  = argv[5];
    const char* output_file = argv[6];

    int n = row * col;

    float* h_temp  = (float*)malloc(n * sizeof(float));
    float* h_power = (float*)malloc(n * sizeof(float));
    float* h_out   = (float*)malloc(n * sizeof(float));

    read_matrix(temp_file,  h_temp,  n);
    read_matrix(power_file, h_power, n);

    // Derive coefficients exactly as per reference
    float grid_height = chip_height / (float)row;
    float grid_width  = chip_width  / (float)col;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx  = grid_width  / (2.0f * K_SI * t_chip * grid_height);
    float Ry  = grid_height / (2.0f * K_SI * t_chip * grid_width);
    float Rz  = t_chip      / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step      = PRECISION / max_slope / 1000.0f;

    float Cap_1 = step / Cap;
    float Rx_1  = 1.0f / Rx;
    float Ry_1  = 1.0f / Ry;
    float Rz_1  = 1.0f / Rz;

    float* d_buf0;
    float* d_buf1;
    float* d_power;

    cudaMalloc(&d_buf0,  n * sizeof(float));
    cudaMalloc(&d_buf1,  n * sizeof(float));
    cudaMalloc(&d_power, n * sizeof(float));

    cudaMemcpy(d_buf0,  h_temp,  n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_power, h_power, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_W, TILE_H);
    dim3 grid((col + TILE_W - 1) / TILE_W, (row + TILE_H - 1) / TILE_H);

    float* src = d_buf0;
    float* dst = d_buf1;

    for (int t = 0; t < sim_time; t++) {
        hotspot_kernel<<<grid, block>>>(dst, src, d_power,
                                        row, col,
                                        Cap_1, Rx_1, Ry_1, Rz_1,
                                        AMB_TEMP);
        float* tmp = src; src = dst; dst = tmp;
    }

    // src now points to the last-written buffer
    cudaMemcpy(h_out, src, n * sizeof(float), cudaMemcpyDeviceToHost);

    write_output(output_file, h_out, n);

    cudaFree(d_buf0);
    cudaFree(d_buf1);
    cudaFree(d_power);
    free(h_temp);
    free(h_power);
    free(h_out);

    return 0;
}
