#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_W 32
#define TILE_H 32

__global__ void hotspot_kernel(
        float * __restrict__ result,
        const float * __restrict__ temp,
        const float * __restrict__ power,
        int row, int col,
        float Cap_1, float Rx_1, float Ry_1, float Rz_1,
        float amb_temp)
{
    __shared__ float s_temp[(TILE_H + 2)][(TILE_W + 2)];

    int c = blockIdx.x * TILE_W + threadIdx.x;
    int r = blockIdx.y * TILE_H + threadIdx.y;

    int sc = threadIdx.x + 1;
    int sr = threadIdx.y + 1;

    float t_self = 0.0f;
    if (r < row && c < col) {
        t_self = temp[r * col + c];
        s_temp[sr][sc] = t_self;
    }

    if (threadIdx.x == 0) {
        int hc = c - 1;
        s_temp[sr][0] = (hc >= 0 && r < row) ? temp[r * col + hc] : 0.0f;
    }
    if (threadIdx.x == TILE_W - 1) {
        int hc = c + 1;
        s_temp[sr][TILE_W + 1] = (hc < col && r < row) ? temp[r * col + hc] : 0.0f;
    }

    if (threadIdx.y == 0) {
        int hr = r - 1;
        s_temp[0][sc] = (hr >= 0 && c < col) ? temp[hr * col + c] : 0.0f;
    }
    if (threadIdx.y == TILE_H - 1) {
        int hr = r + 1;
        s_temp[TILE_H + 1][sc] = (hr < row && c < col) ? temp[hr * col + c] : 0.0f;
    }

    __syncthreads();

    if (r >= row || c >= col) return;

    int idx = r * col + c;

    float t_N = (r > 0)       ? s_temp[sr - 1][sc] : t_self;
    float t_S = (r < row - 1) ? s_temp[sr + 1][sc] : t_self;
    float t_W = (c > 0)       ? s_temp[sr][sc - 1] : t_self;
    float t_E = (c < col - 1) ? s_temp[sr][sc + 1] : t_self;

    float delta = Cap_1 * (
          power[idx]
        + (t_S - t_self) * Ry_1
        + (t_N - t_self) * Ry_1
        + (t_W - t_self) * Rx_1
        + (t_E - t_self) * Rx_1
        + (amb_temp - t_self) * Rz_1
    );

    result[idx] = t_self + delta;
}

static void fatal(const char *msg) { fprintf(stderr, "%s\n", msg); exit(1); }

static float *read_grid(const char *fname, int n)
{
    FILE *fp = fopen(fname, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) fatal("malloc");
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f", &buf[i]) != 1) {
            fprintf(stderr, "Read error at index %d in %s\n", i, fname);
            exit(1);
        }
    }
    fclose(fp);
    return buf;
}

static void write_output(const char *fname, const float *buf, int n)
{
    FILE *fp = fopen(fname, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s for writing\n", fname); exit(1); }
    for (int i = 0; i < n; i++)
        fprintf(fp, "%d\t%g\n", i, buf[i]);
    fclose(fp);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> "
                        "<temp_file> <power_file> <output_file>\n", argv[0]);
        return 1;
    }

    int row      = atoi(argv[1]);
    int col      = atoi(argv[2]);
    int sim_time = atoi(argv[3]);
    const char *temp_file   = argv[4];
    const char *power_file  = argv[5];
    const char *output_file = argv[6];

    int n = row * col;

    float t_chip      = 0.0005f;
    float chip_height = 0.016f;
    float chip_width  = 0.016f;
    float amb_temp    = 80.0f;

    float step = 0.001f;
    float K_Si = 100.0f;
    float K_heat = 0.1f;
    (void)K_heat;

    float dx   = chip_height / (float)row;
    float dy   = chip_width  / (float)col;

    float Cap = CUDART_PI_F * K_Si * t_chip;
    float FACTOR_CHIP  = 0.5f;
    float SPEC_HEAT_SI = 1750000.0f;
    Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;

    float Rx = dy / (2.0f * K_Si * t_chip * dx);
    float Ry = dx / (2.0f * K_Si * t_chip * dy);
    float Rz = t_chip / (K_Si * dx * dy);

    float Cap_1 = step / Cap;
    float Rx_1  = 1.0f / Rx;
    float Ry_1  = 1.0f / Ry;
    float Rz_1  = 1.0f / Rz;

    float *h_temp  = read_grid(temp_file,  n);
    float *h_power = read_grid(power_file, n);

    float *d_buf0, *d_buf1, *d_power;
    cudaMalloc((void **)&d_buf0,  n * sizeof(float));
    cudaMalloc((void **)&d_buf1,  n * sizeof(float));
    cudaMalloc((void **)&d_power, n * sizeof(float));

    cudaMemcpy(d_buf0,  h_temp,  n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_power, h_power, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_W, TILE_H);
    dim3 grid((col + TILE_W - 1) / TILE_W, (row + TILE_H - 1) / TILE_H);

    float *d_t = d_buf0;
    float *d_r = d_buf1;

    for (int t = 0; t < sim_time; t++) {
        hotspot_kernel<<<grid, block>>>(d_r, d_t, d_power,
                                        row, col,
                                        Cap_1, Rx_1, Ry_1, Rz_1,
                                        amb_temp);
        float *tmp = d_t; d_t = d_r; d_r = tmp;
    }

    float *h_result = (float *)malloc(n * sizeof(float));
    cudaMemcpy(h_result, d_t, n * sizeof(float), cudaMemcpyDeviceToHost);

    write_output(output_file, h_result, n);

    free(h_temp);
    free(h_power);
    free(h_result);
    cudaFree(d_buf0);
    cudaFree(d_buf1);
    cudaFree(d_power);

    return 0;
}
