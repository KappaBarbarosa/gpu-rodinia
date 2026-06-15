/*
 * HotSpot — serial reference implementation.
 *
 * Transient thermal simulation of a 2D chip. Given an initial temperature
 * grid and a per-cell power-dissipation grid, repeatedly advance the
 * temperature field by one discrete time step ("single_iteration") for
 * `sim_time` iterations, then write the final temperature grid.
 *
 * This is a single-threaded reference. Each output cell depends only on the
 * PREVIOUS iteration's temperature grid (a 5-point stencil plus a local power
 * term), so cells within one iteration are mutually independent.
 *
 * CLI:  hotspot_serial <grid_rows> <grid_cols> <sim_time> \
 *                      <temp_file> <power_file> <output_file>
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD       (3.0e6)
/* required precision in degrees */
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100
/* capacitance fitting factor */
#define FACTOR_CHIP  0.5

typedef float FLOAT;

/* chip parameters */
const FLOAT t_chip      = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width  = 0.016;
/* ambient temperature, assuming no package at all */
const FLOAT amb_temp    = 80.0;

void fatal(const char *s)
{
    fprintf(stderr, "error: %s\n", s);
    exit(1);
}

/* Single iteration of the transient solver: advance `temp` by one time step
 * into `result`. col = number of columns, row = number of rows. */
void single_iteration(FLOAT *result, FLOAT *temp, FLOAT *power, int row, int col,
                      FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1)
{
    int r, c;
    FLOAT delta;

    for (r = 0; r < row; ++r) {
        for (c = 0; c < col; ++c) {
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
                delta = (Cap_1) * (power[r*col+c] +
                    (temp[r*col+c-1]     - temp[r*col+c]) * Rx_1 +
                    (temp[(r-1)*col+c]   - temp[r*col+c]) * Ry_1 +
                    (amb_temp            - temp[r*col+c]) * Rz_1);
            } else if ((r == row - 1) && (c == 0)) {       /* Corner 4 */
                delta = (Cap_1) * (power[r*col] +
                    (temp[r*col+1]     - temp[r*col]) * Rx_1 +
                    (temp[(r-1)*col]   - temp[r*col]) * Ry_1 +
                    (amb_temp          - temp[r*col]) * Rz_1);
            } else if (r == 0) {                           /* Edge 1 (top) */
                delta = (Cap_1) * (power[c] +
                    (temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 +
                    (temp[col+c]           - temp[c]) * Ry_1 +
                    (amb_temp              - temp[c]) * Rz_1);
            } else if (c == col - 1) {                     /* Edge 2 (right) */
                delta = (Cap_1) * (power[r*col+c] +
                    (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) * Ry_1 +
                    (temp[r*col+c-1]                       - temp[r*col+c]) * Rx_1 +
                    (amb_temp                              - temp[r*col+c]) * Rz_1);
            } else if (r == row - 1) {                     /* Edge 3 (bottom) */
                delta = (Cap_1) * (power[r*col+c] +
                    (temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) * Rx_1 +
                    (temp[(r-1)*col+c]                 - temp[r*col+c]) * Ry_1 +
                    (amb_temp                          - temp[r*col+c]) * Rz_1);
            } else if (c == 0) {                           /* Edge 4 (left) */
                delta = (Cap_1) * (power[r*col] +
                    (temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) * Ry_1 +
                    (temp[r*col+1]                     - temp[r*col]) * Rx_1 +
                    (amb_temp                          - temp[r*col]) * Rz_1);
            } else {                                       /* Interior */
                delta = (Cap_1) * (power[r*col+c] +
                    (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) * Ry_1 +
                    (temp[r*col+c+1]   + temp[r*col+c-1]   - 2.0*temp[r*col+c]) * Rx_1 +
                    (amb_temp                              - temp[r*col+c]) * Rz_1);
            }
            result[r*col+c] = temp[r*col+c] + delta;
        }
    }
}

/* Transient solver driver: convert the heat-transfer ODE coefficients to
 * difference-equation form, then iterate `single_iteration`. */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power,
                       int row, int col)
{
    FLOAT grid_height = chip_height / row;
    FLOAT grid_width  = chip_width  / col;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx  = grid_width  / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry  = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz  = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step      = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1  = 1.f / Rx;
    FLOAT Ry_1  = 1.f / Ry;
    FLOAT Rz_1  = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;

    FLOAT *r = result;
    FLOAT *t = temp;
    for (int i = 0; i < num_iterations; i++) {
        single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1);
        FLOAT *tmp = t; t = r; r = tmp;
    }
    /* After the loop, the most recent result lives in `t` (due to the final
     * swap). The caller writes (sim_time & 1) ? result : temp accordingly. */
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, const char *file)
{
    int i;
    FILE *fp;
    char str[STR_SIZE];
    FLOAT val;

    fp = fopen(file, "r");
    if (!fp) fatal("file could not be opened for reading");

    for (i = 0; i < grid_rows * grid_cols; i++) {
        if (!fgets(str, STR_SIZE, fp)) fatal("not enough lines in file");
        if (sscanf(str, "%f", &val) != 1) fatal("invalid file format");
        vect[i] = val;
    }
    fclose(fp);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, const char *file)
{
    int i, j, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0) printf("The file was not opened\n");

    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++) {
            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
            fputs(str, fp);
            index++;
        }
    fclose(fp);
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
    exit(1);
}

int main(int argc, char **argv)
{
    int grid_rows, grid_cols, sim_time;
    FLOAT *temp, *power, *result;
    const char *tfile, *pfile, *ofile;

    if (argc != 7) usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 ||
        (grid_cols = atoi(argv[2])) <= 0 ||
        (sim_time  = atoi(argv[3])) <= 0)
        usage(argc, argv);

    temp   = (FLOAT *) calloc(grid_rows * grid_cols, sizeof(FLOAT));
    power  = (FLOAT *) calloc(grid_rows * grid_cols, sizeof(FLOAT));
    result = (FLOAT *) calloc(grid_rows * grid_cols, sizeof(FLOAT));
    if (!temp || !power || !result) fatal("unable to allocate memory");

    tfile = argv[4];
    pfile = argv[5];
    ofile = argv[6];

    read_input(temp,  grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);

    double t0 = now_seconds();
    compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    /* Ping-pong: after an even number of iterations the latest data is in
     * `temp`; after an odd number it is in `result`. */
    writeoutput((1 & sim_time) ? result : temp, grid_rows, grid_cols, ofile);

    free(temp);
    free(power);
    free(result);
    return 0;
}
