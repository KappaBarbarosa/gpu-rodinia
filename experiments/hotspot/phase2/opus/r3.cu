#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD  (3.0e6)
/* required precision in degrees */
#define PRECISION   0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor */
#define FACTOR_CHIP 0.5

/* chip parameters */
const double t_chip = 0.0005;
const double chip_height = 0.016;
const double chip_width = 0.016;
/* ambient temperature, assuming no package at all */
const double amb_temp = 80.0;

#define BLOCK_X 32
#define BLOCK_Y 8

/* Naive one-thread-per-cell global-memory 5-point stencil, single step/launch. */
__global__ void hotspot_kernel(const float * __restrict__ power,
                               const float * __restrict__ temp_src,
                               float * __restrict__ temp_dst,
                               int row, int col,
                               float Cap_1, float Rx_1, float Ry_1, float Rz_1,
                               float step, float amb)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; /* column, coalesced */
    int r = blockIdx.y * blockDim.y + threadIdx.y; /* row */

    if (r >= row || c >= col) return;

    int idx = r * col + c;

    float center = __ldg(&temp_src[idx]);
    float north, south, east, west;

    /* boundary ladder: corners / edges / interior */
    if (r == 0)        north = center; else north = __ldg(&temp_src[idx - col]);
    if (r == row - 1)  south = center; else south = __ldg(&temp_src[idx + col]);
    if (c == 0)        west  = center; else west  = __ldg(&temp_src[idx - 1]);
    if (c == col - 1)  east  = center; else east  = __ldg(&temp_src[idx + 1]);

    float delta = Cap_1 * (__ldg(&power[idx])
                  + (south + north - 2.0f * center) * Ry_1
                  + (east + west - 2.0f * center) * Rx_1
                  + (amb - center) * Rz_1);

    temp_dst[idx] = center + delta;
}

void fatal(const char *s) { fprintf(stderr, "Error: %s\n", s); exit(1); }

void readinput(float *vect, int grid_rows, int grid_cols, const char *file)
{
    int i, j;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if ((fp = fopen(file, "r")) == 0) fatal("file could not be opened for reading");
    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++) {
            if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
            if (feof(fp)) fatal("not enough lines in file");
            if ((sscanf(str, "%f", &val) != 1)) fatal("invalid file format");
            vect[i * grid_cols + j] = val;
        }
    fclose(fp);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, const char *file)
{
    int i, j, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0) printf("The file was not opened\n");
    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++) {
            sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }
    fclose(fp);
}

int main(int argc, char **argv)
{
    int grid_rows, grid_cols, sim_time;
    int size;
    char *tfile, *pfile, *ofile;

    if (argc != 7) {
        printf("Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
        exit(1);
    }

    grid_rows = atoi(argv[1]);
    grid_cols = atoi(argv[2]);
    sim_time  = atoi(argv[3]);
    tfile = argv[4];
    pfile = argv[5];
    ofile = argv[6];

    size = grid_rows * grid_cols;

    float *temp  = (float *)malloc(size * sizeof(float));
    float *power = (float *)malloc(size * sizeof(float));
    float *result= (float *)malloc(size * sizeof(float));
    if (!temp || !power || !result) fatal("unable to allocate memory");

    readinput(temp,  grid_rows, grid_cols, tfile);
    readinput(power, grid_rows, grid_cols, pfile);

    /* coefficient derivation (must match serial reference exactly) */
    double grid_height = chip_height / grid_rows;
    double grid_width  = chip_width  / grid_cols;

    double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    double Rx = grid_width  / (2.0 * K_SI * t_chip * grid_height);
    double Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    double Rz = t_chip / (K_SI * grid_height * grid_width);

    double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    double step = PRECISION / max_slope / 1000.0;

    float Cap_1 = (float)(step / Cap);
    float Rx_1  = (float)(1.0 / Rx);
    float Ry_1  = (float)(1.0 / Ry);
    float Rz_1  = (float)(1.0 / Rz);
    float amb   = (float)amb_temp;

    float *d_temp, *d_power, *d_dst;
    cudaMalloc((void **)&d_temp,  size * sizeof(float));
    cudaMalloc((void **)&d_power, size * sizeof(float));
    cudaMalloc((void **)&d_dst,   size * sizeof(float));

    cudaMemcpy(d_temp,  temp,  size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_power, power, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((grid_cols + BLOCK_X - 1) / BLOCK_X,
              (grid_rows + BLOCK_Y - 1) / BLOCK_Y);

    /* ping-pong, single time step per launch, one sync at the end */
    float *src = d_temp;
    float *dst = d_dst;
    for (int t = 0; t < sim_time; t++) {
        hotspot_kernel<<<grid, block>>>(d_power, src, dst,
                                        grid_rows, grid_cols,
                                        Cap_1, Rx_1, Ry_1, Rz_1, step, amb);
        float *tmp = src; src = dst; dst = tmp;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(result, src, size * sizeof(float), cudaMemcpyDeviceToHost);

    writeoutput(result, grid_rows, grid_cols, ofile);

    cudaFree(d_temp);
    cudaFree(d_power);
    cudaFree(d_dst);
    free(temp);
    free(power);
    free(result);

    return 0;
}
