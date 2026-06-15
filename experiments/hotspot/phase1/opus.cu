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

#define BLOCK_X 16
#define BLOCK_Y 16

static void checkCuda(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

/* One GPU thread per grid cell. GLOBAL memory only. Reproduces the serial
   per-cell update exactly, including corner/edge/interior boundary cases. */
__global__ void single_iteration_kernel(FLOAT *result, const FLOAT *temp,
                                         const FLOAT *power, int row, int col,
                                         FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1,
                                         FLOAT Rz_1, FLOAT amb)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= row || c >= col) return;

    FLOAT delta;

    if ((r == 0) && (c == 0)) {                    /* Corner 1 */
        delta = (Cap_1) * (power[0] +
            (temp[1]   - temp[0]) * Rx_1 +
            (temp[col] - temp[0]) * Ry_1 +
            (amb       - temp[0]) * Rz_1);
    } else if ((r == 0) && (c == col - 1)) {       /* Corner 2 */
        delta = (Cap_1) * (power[c] +
            (temp[c-1]   - temp[c]) * Rx_1 +
            (temp[c+col] - temp[c]) * Ry_1 +
            (amb         - temp[c]) * Rz_1);
    } else if ((r == row - 1) && (c == col - 1)) { /* Corner 3 */
        delta = (Cap_1) * (power[r*col+c] +
            (temp[r*col+c-1]   - temp[r*col+c]) * Rx_1 +
            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 +
            (amb               - temp[r*col+c]) * Rz_1);
    } else if ((r == row - 1) && (c == 0)) {       /* Corner 4 */
        delta = (Cap_1) * (power[r*col] +
            (temp[r*col+1]   - temp[r*col]) * Rx_1 +
            (temp[(r-1)*col] - temp[r*col]) * Ry_1 +
            (amb             - temp[r*col]) * Rz_1);
    } else if (r == 0) {                           /* Edge 1 (top) */
        delta = (Cap_1) * (power[c] +
            (temp[c+1] + temp[c-1] - 2.0f*temp[c]) * Rx_1 +
            (temp[col+c]           - temp[c]) * Ry_1 +
            (amb                   - temp[c]) * Rz_1);
    } else if (c == col - 1) {                     /* Edge 2 (right) */
        delta = (Cap_1) * (power[r*col+c] +
            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0f*temp[r*col+c]) * Ry_1 +
            (temp[r*col+c-1]                       - temp[r*col+c]) * Rx_1 +
            (amb                                   - temp[r*col+c]) * Rz_1);
    } else if (r == row - 1) {                     /* Edge 3 (bottom) */
        delta = (Cap_1) * (power[r*col+c] +
            (temp[r*col+c+1] + temp[r*col+c-1] - 2.0f*temp[r*col+c]) * Rx_1 +
            (temp[(r-1)*col+c]                 - temp[r*col+c]) * Ry_1 +
            (amb                               - temp[r*col+c]) * Rz_1);
    } else if (c == 0) {                           /* Edge 4 (left) */
        delta = (Cap_1) * (power[r*col] +
            (temp[(r+1)*col] + temp[(r-1)*col] - 2.0f*temp[r*col]) * Ry_1 +
            (temp[r*col+1]                     - temp[r*col]) * Rx_1 +
            (amb                               - temp[r*col]) * Rz_1);
    } else {                                       /* Interior */
        delta = (Cap_1) * (power[r*col+c] +
            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0f*temp[r*col+c]) * Ry_1 +
            (temp[r*col+c+1]   + temp[r*col+c-1]   - 2.0f*temp[r*col+c]) * Rx_1 +
            (amb                                   - temp[r*col+c]) * Rz_1);
    }
    result[r*col+c] = temp[r*col+c] + delta;
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, const char *file)
{
    int i;
    FILE *fp = fopen(file, "r");
    if (!fp) { fprintf(stderr, "file %s could not be opened\n", file); exit(1); }
    char str[STR_SIZE];
    FLOAT val;
    int total = grid_rows * grid_cols;
    for (i = 0; i < total; i++) {
        if (fgets(str, STR_SIZE, fp) == NULL) {
            fprintf(stderr, "not enough lines in file %s\n", file); exit(1);
        }
        if (feof(fp)) { fprintf(stderr, "not enough lines in file %s\n", file); exit(1); }
        if (sscanf(str, "%f", &val) != 1) {
            fprintf(stderr, "invalid file format %s\n", file); exit(1);
        }
        vect[i] = val;
    }
    fclose(fp);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, const char *file)
{
    int i, j, index = 0;
    char str[STR_SIZE];
    FILE *fp = fopen(file, "w");
    if (!fp) { fprintf(stderr, "file %s could not be opened\n", file); exit(1); }
    for (i = 0; i < grid_rows; i++) {
        for (j = 0; j < grid_cols; j++) {
            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
            fputs(str, fp);
            index++;
        }
    }
    fclose(fp);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
        exit(1);
    }

    int grid_rows = atoi(argv[1]);
    int grid_cols = atoi(argv[2]);
    int sim_time  = atoi(argv[3]);
    const char *tfile = argv[4];
    const char *pfile = argv[5];
    const char *ofile = argv[6];

    if (grid_rows <= 0 || grid_cols <= 0 || sim_time < 0) {
        fprintf(stderr, "invalid arguments\n"); exit(1);
    }

    int size = grid_rows * grid_cols;
    size_t bytes = (size_t)size * sizeof(FLOAT);

    FLOAT *temp   = (FLOAT *)calloc(size, sizeof(FLOAT));
    FLOAT *power  = (FLOAT *)calloc(size, sizeof(FLOAT));
    FLOAT *result = (FLOAT *)calloc(size, sizeof(FLOAT));
    if (!temp || !power || !result) {
        fprintf(stderr, "calloc failed\n"); exit(1);
    }

    read_input(temp,  grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);

    /* Coefficients computed exactly as compute_tran_temp does. */
    FLOAT grid_height = chip_height / grid_rows;
    FLOAT grid_width  = chip_width  / grid_cols;
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

    /* Device buffers: ping-pong two temperature buffers, plus power. */
    FLOAT *d_temp, *d_result, *d_power;
    checkCuda(cudaMalloc((void **)&d_temp,   bytes), "malloc d_temp");
    checkCuda(cudaMalloc((void **)&d_result, bytes), "malloc d_result");
    checkCuda(cudaMalloc((void **)&d_power,  bytes), "malloc d_power");

    checkCuda(cudaMemcpy(d_temp,  temp,  bytes, cudaMemcpyHostToDevice), "copy temp");
    checkCuda(cudaMemcpy(d_power, power, bytes, cudaMemcpyHostToDevice), "copy power");

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((grid_cols + BLOCK_X - 1) / BLOCK_X,
              (grid_rows + BLOCK_Y - 1) / BLOCK_Y);

    /* t holds the "current" buffer, r the "next" buffer; ping-pong each step. */
    FLOAT *t = d_temp;
    FLOAT *r = d_result;
    for (int i = 0; i < sim_time; i++) {
        single_iteration_kernel<<<grid, block>>>(r, t, d_power, grid_rows, grid_cols,
                                                  Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
        checkCuda(cudaGetLastError(), "kernel launch");
        FLOAT *tmp = t; t = r; r = tmp;
    }
    checkCuda(cudaDeviceSynchronize(), "sync");

    /* After the loop, the final data lives in buffer `t`. */
    FLOAT *final_dev = t;
    FLOAT *final_host = (sim_time & 1) ? result : temp;
    checkCuda(cudaMemcpy(final_host, final_dev, bytes, cudaMemcpyDeviceToHost), "copy back");

    writeoutput(final_host, grid_rows, grid_cols, ofile);

    cudaFree(d_temp);
    cudaFree(d_result);
    cudaFree(d_power);
    free(temp);
    free(power);
    free(result);
    return 0;
}
