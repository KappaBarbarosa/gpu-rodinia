#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STR_SIZE 256
#define MAX_PD       (3.0e6)
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100
#define FACTOR_CHIP  0.5

typedef float FLOAT;

const FLOAT t_chip      = 0.0005f;
const FLOAT chip_height = 0.016f;
const FLOAT chip_width  = 0.016f;
const FLOAT amb_temp    = 80.0f;

__global__ void hotspot_kernel(FLOAT *result, FLOAT *temp, FLOAT *power,
                               int row, int col,
                               FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= row || c >= col) return;

    int idx = r * col + c;
    FLOAT delta;

    if (r == 0 && c == 0) {
        /* Corner 1 */
        delta = Cap_1 * (power[0] +
            (temp[1]   - temp[0]) * Rx_1 +
            (temp[col] - temp[0]) * Ry_1 +
            (amb_temp  - temp[0]) * Rz_1);
    } else if (r == 0 && c == col - 1) {
        /* Corner 2 */
        delta = Cap_1 * (power[c] +
            (temp[c-1]   - temp[c]) * Rx_1 +
            (temp[c+col] - temp[c]) * Ry_1 +
            (amb_temp    - temp[c]) * Rz_1);
    } else if (r == row - 1 && c == col - 1) {
        /* Corner 3 */
        delta = Cap_1 * (power[r*col+c] +
            (temp[r*col+c-1]   - temp[r*col+c]) * Rx_1 +
            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 +
            (amb_temp          - temp[r*col+c]) * Rz_1);
    } else if (r == row - 1 && c == 0) {
        /* Corner 4 */
        delta = Cap_1 * (power[r*col] +
            (temp[r*col+1]   - temp[r*col]) * Rx_1 +
            (temp[(r-1)*col] - temp[r*col]) * Ry_1 +
            (amb_temp        - temp[r*col]) * Rz_1);
    } else if (r == 0) {
        /* Edge 1 (top) */
        delta = Cap_1 * (power[c] +
            (temp[c+1] + temp[c-1] - 2.0f*temp[c]) * Rx_1 +
            (temp[col+c]           - temp[c]) * Ry_1 +
            (amb_temp              - temp[c]) * Rz_1);
    } else if (c == col - 1) {
        /* Edge 2 (right) */
        delta = Cap_1 * (power[r*col+c] +
            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0f*temp[r*col+c]) * Ry_1 +
            (temp[r*col+c-1]                       - temp[r*col+c]) * Rx_1 +
            (amb_temp                              - temp[r*col+c]) * Rz_1);
    } else if (r == row - 1) {
        /* Edge 3 (bottom) */
        delta = Cap_1 * (power[r*col+c] +
            (temp[r*col+c+1] + temp[r*col+c-1] - 2.0f*temp[r*col+c]) * Rx_1 +
            (temp[(r-1)*col+c]                 - temp[r*col+c]) * Ry_1 +
            (amb_temp                          - temp[r*col+c]) * Rz_1);
    } else if (c == 0) {
        /* Edge 4 (left) */
        delta = Cap_1 * (power[r*col] +
            (temp[(r+1)*col] + temp[(r-1)*col] - 2.0f*temp[r*col]) * Ry_1 +
            (temp[r*col+1]                     - temp[r*col]) * Rx_1 +
            (amb_temp                          - temp[r*col]) * Rz_1);
    } else {
        /* Interior */
        delta = Cap_1 * (power[r*col+c] +
            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0f*temp[r*col+c]) * Ry_1 +
            (temp[r*col+c+1]   + temp[r*col+c-1]   - 2.0f*temp[r*col+c]) * Rx_1 +
            (amb_temp                              - temp[r*col+c]) * Rz_1);
    }

    result[idx] = temp[idx] + delta;
}

static void read_input(FLOAT *vect, int size, const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
    for (int i = 0; i < size; i++) {
        if (fscanf(fp, "%f", &vect[i]) != 1) {
            fprintf(stderr, "Error reading %s at index %d\n", filename, i);
            exit(1);
        }
    }
    fclose(fp);
}

static void writeoutput(FLOAT *vect, int rows, int cols, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s for writing\n", filename); exit(1); }
    int total = rows * cols;
    for (int i = 0; i < total; i++) {
        fprintf(fp, "%d\t%g\n", i, vect[i]);
    }
    fclose(fp);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
        return 1;
    }

    int rows      = atoi(argv[1]);
    int cols      = atoi(argv[2]);
    int sim_time  = atoi(argv[3]);
    const char *temp_file   = argv[4];
    const char *power_file  = argv[5];
    const char *output_file = argv[6];

    int total = rows * cols;
    size_t sz = total * sizeof(FLOAT);

    FLOAT *h_temp  = (FLOAT*)calloc(total, sizeof(FLOAT));
    FLOAT *h_power = (FLOAT*)calloc(total, sizeof(FLOAT));
    FLOAT *h_result = (FLOAT*)calloc(total, sizeof(FLOAT));

    if (!h_temp || !h_power || !h_result) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    read_input(h_temp,  total, temp_file);
    read_input(h_power, total, power_file);

    /* Compute coefficients exactly as compute_tran_temp does */
    FLOAT grid_height = chip_height / rows;
    FLOAT grid_width  = chip_width  / cols;
    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx  = grid_width  / (2.0f * K_SI * t_chip * grid_height);
    FLOAT Ry  = grid_height / (2.0f * K_SI * t_chip * grid_width);
    FLOAT Rz  = t_chip / (K_SI * grid_height * grid_width);
    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step      = PRECISION / max_slope / 1000.0f;
    FLOAT Rx_1  = 1.f / Rx;
    FLOAT Ry_1  = 1.f / Ry;
    FLOAT Rz_1  = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;

    /* Allocate two device buffers for ping-pong */
    FLOAT *d_buf0, *d_buf1, *d_power;
    cudaMalloc((void**)&d_buf0,  sz);
    cudaMalloc((void**)&d_buf1,  sz);
    cudaMalloc((void**)&d_power, sz);

    cudaMemcpy(d_buf0,  h_temp,  sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_power, h_power, sz, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    /* Ping-pong: d_t is the "current temp", d_r is the "result" buffer */
    FLOAT *d_t = d_buf0;
    FLOAT *d_r = d_buf1;

    for (int i = 0; i < sim_time; i++) {
        hotspot_kernel<<<grid, block>>>(d_r, d_t, d_power, rows, cols,
                                       Cap_1, Rx_1, Ry_1, Rz_1);
        /* Swap pointers */
        FLOAT *tmp = d_t; d_t = d_r; d_r = tmp;
    }

    cudaMemcpy(h_result, d_t, sz, cudaMemcpyDeviceToHost);

    writeoutput(h_result, rows, cols, output_file);

    cudaFree(d_buf0);
    cudaFree(d_buf1);
    cudaFree(d_power);
    free(h_temp);
    free(h_power);
    free(h_result);

    return 0;
}
