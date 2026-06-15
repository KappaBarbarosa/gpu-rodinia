#include <stdio.h>
#include <stdlib.h>

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

__global__ void single_iteration_kernel(FLOAT *result, FLOAT *temp, FLOAT *power,
                                         int row, int col, FLOAT Cap_1, FLOAT Rx_1,
                                         FLOAT Ry_1, FLOAT Rz_1)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= row || c >= col) return;

    int idx = r * col + c;
    FLOAT delta;

    if ((r == 0) && (c == 0)) {                    /* Corner 1 */
        delta = (Cap_1) * (power[0] +
            (temp[1]   - temp[0]) * Rx_1 +
            (temp[col] - temp[0]) * Ry_1 +
            (amb_temp  - temp[0]) * Rz_1);
    } else if ((r == 0) && (c == col - 1)) {       /* Corner 2 */
        delta = (Cap_1) * (power[c] +
            (temp[c-1]   - temp[c]) * Rx_1 +
            (temp[c+col] - temp[c]) * Ry_1 +
            (amb_temp    - temp[c]) * Rz_1);
    } else if ((r == row - 1) && (c == col - 1)) { /* Corner 3 */
        delta = (Cap_1) * (power[idx] +
            (temp[idx-1]     - temp[idx]) * Rx_1 +
            (temp[(r-1)*col+c]   - temp[idx]) * Ry_1 +
            (amb_temp            - temp[idx]) * Rz_1);
    } else if ((r == row - 1) && (c == 0)) {       /* Corner 4 */
        delta = (Cap_1) * (power[r*col] +
            (temp[r*col+1]     - temp[r*col]) * Rx_1 +
            (temp[(r-1)*col]   - temp[r*col]) * Ry_1 +
            (amb_temp          - temp[r*col]) * Rz_1);
    } else if (r == 0) {                           /* Edge 1 (top) */
        delta = (Cap_1) * (power[c] +
            (temp[c+1] + temp[c-1] - 2.0f*temp[c]) * Rx_1 +
            (temp[col+c]           - temp[c]) * Ry_1 +
            (amb_temp              - temp[c]) * Rz_1);
    } else if (c == col - 1) {                     /* Edge 2 (right) */
        delta = (Cap_1) * (power[idx] +
            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0f*temp[idx]) * Ry_1 +
            (temp[idx-1]                       - temp[idx]) * Rx_1 +
            (amb_temp                              - temp[idx]) * Rz_1);
    } else if (r == row - 1) {                     /* Edge 3 (bottom) */
        delta = (Cap_1) * (power[idx] +
            (temp[idx+1] + temp[idx-1] - 2.0f*temp[idx]) * Rx_1 +
            (temp[(r-1)*col+c]                 - temp[idx]) * Ry_1 +
            (amb_temp                          - temp[idx]) * Rz_1);
    } else if (c == 0) {                           /* Edge 4 (left) */
        delta = (Cap_1) * (power[r*col] +
            (temp[(r+1)*col] + temp[(r-1)*col] - 2.0f*temp[r*col]) * Ry_1 +
            (temp[r*col+1]                     - temp[r*col]) * Rx_1 +
            (amb_temp                          - temp[r*col]) * Rz_1);
    } else {                                       /* Interior */
        delta = (Cap_1) * (power[idx] +
            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0f*temp[idx]) * Ry_1 +
            (temp[idx+1]   + temp[idx-1]   - 2.0f*temp[idx]) * Rx_1 +
            (amb_temp                              - temp[idx]) * Rz_1);
    }
    result[idx] = temp[idx] + delta;
}

void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power,
                       int row, int col)
{
    FLOAT grid_height = chip_height / row;
    FLOAT grid_width  = chip_width  / col;
    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx  = grid_width  / (2.0f * K_SI * t_chip * grid_height);
    FLOAT Ry  = grid_height / (2.0f * K_SI * t_chip * grid_width);
    FLOAT Rz  = t_chip / (K_SI * grid_height * grid_width);
    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step      = PRECISION / max_slope / 1000.0f;
    FLOAT Rx_1  = 1.0f / Rx;
    FLOAT Ry_1  = 1.0f / Ry;
    FLOAT Rz_1  = 1.0f / Rz;
    FLOAT Cap_1 = step / Cap;

    FLOAT *d_temp, *d_power, *d_result;
    int size = row * col * sizeof(FLOAT);

    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_power, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_temp, temp, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_power, power, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((col + blockDim.x - 1) / blockDim.x,
                 (row + blockDim.y - 1) / blockDim.y);

    FLOAT *r = d_result;
    FLOAT *t = d_temp;
    for (int i = 0; i < num_iterations; i++) {
        single_iteration_kernel<<<gridDim, blockDim>>>(r, t, d_power, row, col,
                                                        Cap_1, Rx_1, Ry_1, Rz_1);
        cudaDeviceSynchronize();
        FLOAT *tmp = t;
        t = r;
        r = tmp;
    }

    FLOAT *final = (num_iterations & 1) ? d_result : d_temp;
    cudaMemcpy(result, final, size, cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_power);
    cudaFree(d_result);
}

void read_input(FLOAT *vect, int num, FILE *fp)
{
    int i;
    for (i = 0; i < num; i++) {
        fscanf(fp, "%f", &vect[i]);
    }
}

void writeoutput(FLOAT *vect, int row, int col, FILE *fp)
{
    int i, j;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            fprintf(fp, "%d\t%g\n", i*col+j, vect[i*col+j]);
        }
    }
}

int main(int argc, char *argv[])
{
    int grid_rows, grid_cols, sim_time;
    char temp_file[STR_SIZE], power_file[STR_SIZE], output_file[STR_SIZE];

    if (argc != 7) {
        fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n",
                argv[0]);
        return 1;
    }

    grid_rows = atoi(argv[1]);
    grid_cols = atoi(argv[2]);
    sim_time = atoi(argv[3]);
    strcpy(temp_file, argv[4]);
    strcpy(power_file, argv[5]);
    strcpy(output_file, argv[6]);

    int size = grid_rows * grid_cols;
    FLOAT *temp = (FLOAT *)malloc(size * sizeof(FLOAT));
    FLOAT *power = (FLOAT *)malloc(size * sizeof(FLOAT));
    FLOAT *result = (FLOAT *)malloc(size * sizeof(FLOAT));

    FILE *fp = fopen(temp_file, "r");
    read_input(temp, size, fp);
    fclose(fp);

    fp = fopen(power_file, "r");
    read_input(power, size, fp);
    fclose(fp);

    compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);

    fp = fopen(output_file, "w");
    FLOAT *final = (sim_time & 1) ? result : temp;
    writeoutput(final, grid_rows, grid_cols, fp);
    fclose(fp);

    free(temp);
    free(power);
    free(result);

    return 0;
}
