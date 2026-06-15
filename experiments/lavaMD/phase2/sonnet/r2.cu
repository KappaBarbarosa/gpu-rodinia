/*
 * lavaMD — CUDA port with shared-memory tiling, round 2.
 * Improvements over r1:
 *   1. __expf (SFU) — retained from r1 (~14% win).
 *   2. Neighbour list (nn + number[]) preloaded into registers at block start,
 *      eliminating repeated global reads of box_str in the hot k-loop.
 *   3. __ldg() read-only cache loads for rv/qv in the cooperative stage.
 *   4. Home-particle rAi kept in registers throughout (already in r1).
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define NUMBER_PAR_PER_BOX 100
/* Maximum neighbours (26) stored inline */
#define MAX_NEIGHBOURS 26

typedef float fp;

typedef struct { fp v, x, y, z; } FOUR_VECTOR;

typedef struct { int x, y, z; int number; long offset; } nei_str;
typedef struct {
    int x, y, z;
    int number;
    long offset;
    int nn;
    nei_str nei[MAX_NEIGHBOURS];
} box_str;

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*
 * GPU kernel: one block per home box, 100 threads per block.
 *
 * Key changes vs r1:
 *  - We preload nn and the neighbour number[] array from box_str into a small
 *    shared array at kernel startup (one cooperative load, then __syncthreads).
 *    This removes 27 * (1 + nn) potential global/L2 reads of box_str fields
 *    from the hot loop, replacing them with fast shared-memory reads.
 *  - __ldg() hints for the cooperative rv/qv staging loads so they go through
 *    the read-only (texture) cache path.
 *  - __expf retained from r1 (SFU path, ~14% kernel speedup).
 */
__global__ void lavaMD_kernel(
    const box_str     * __restrict__ boxes,
    const FOUR_VECTOR * __restrict__ rv,
    const fp          * __restrict__ qv,
    FOUR_VECTOR       * __restrict__ fv,
    long number_boxes,
    fp a2
) {
    /* Staging for one neighbour box at a time */
    __shared__ FOUR_VECTOR sB[NUMBER_PAR_PER_BOX];
    __shared__ fp          sQ[NUMBER_PAR_PER_BOX];

    /* Preloaded neighbour metadata: 0 = home, 1..nn = neighbours */
    __shared__ int sNeighNum[1 + MAX_NEIGHBOURS];  /* box numbers */
    __shared__ int sNN;                             /* neighbour count */

    long l = blockIdx.x;   /* home box index */
    int  i = threadIdx.x;  /* home particle index (0..99) */

    /* --- Cooperatively preload nn and neighbour numbers --- */
    if (i == 0) {
        int nn = boxes[l].nn;
        sNN = nn;
        sNeighNum[0] = (int)l;          /* home box is source k=0 */
        for (int k = 0; k < nn; k++) {
            sNeighNum[k + 1] = boxes[l].nei[k].number;
        }
    }
    __syncthreads();

    long first_i = boxes[l].offset;
    FOUR_VECTOR rAi = __ldg(&rv[first_i + i]);

    fp fv_v = 0.0f, fv_x = 0.0f, fv_y = 0.0f, fv_z = 0.0f;

    int nn = sNN;
    for (int k = 0; k <= nn; k++) {
        int pointer  = sNeighNum[k];
        long first_j = boxes[pointer].offset;   /* offset still read from global; nn times only */

        /* Cooperatively stage this source box into shared memory */
        __syncthreads();
        sB[i] = __ldg(&rv[first_j + i]);
        sQ[i] = __ldg(&qv[first_j + i]);
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            fp rBv = sB[j].v;
            fp rBx = sB[j].x;
            fp rBy = sB[j].y;
            fp rBz = sB[j].z;
            fp qBj = sQ[j];

            fp r2  = rAi.v + rBv - (rAi.x * rBx + rAi.y * rBy + rAi.z * rBz);
            fp u2  = a2 * r2;
            fp vij = __expf(-u2);
            fp fs  = 2.0f * vij;

            fp dx = rAi.x - rBx;
            fp dy = rAi.y - rBy;
            fp dz = rAi.z - rBz;

            fv_v += qBj * vij;
            fv_x += qBj * fs * dx;
            fv_y += qBj * fs * dy;
            fv_z += qBj * fs * dz;
        }
    }

    fv[first_i + i].v = fv_v;
    fv[first_i + i].x = fv_x;
    fv[first_i + i].y = fv_y;
    fv[first_i + i].z = fv_z;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <boxes1d> <output_file>\n", argv[0]);
        return 1;
    }
    int boxes1d = atoi(argv[1]);
    const char *ofile = argv[2];
    if (boxes1d <= 0) {
        fprintf(stderr, "boxes1d must be > 0\n");
        return 1;
    }

    fp alpha = 0.5f;
    fp a2    = 2.0f * alpha * alpha;   /* = 0.5 */

    long number_boxes = (long)boxes1d * boxes1d * boxes1d;
    long space_elem   = number_boxes * NUMBER_PAR_PER_BOX;

    box_str     *box = (box_str *)    malloc(number_boxes * sizeof(box_str));
    FOUR_VECTOR *rv  = (FOUR_VECTOR *)malloc(space_elem   * sizeof(FOUR_VECTOR));
    fp          *qv  = (fp *)         malloc(space_elem   * sizeof(fp));
    FOUR_VECTOR *fv  = (FOUR_VECTOR *)malloc(space_elem   * sizeof(FOUR_VECTOR));
    if (!box || !rv || !qv || !fv) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* ---- BOX / NEIGHBOUR structure ---- */
    int nh = 0;
    for (int i = 0; i < boxes1d; i++) {
        for (int j = 0; j < boxes1d; j++) {
            for (int k = 0; k < boxes1d; k++) {
                box[nh].x = k;
                box[nh].y = j;
                box[nh].z = i;
                box[nh].number = nh;
                box[nh].offset = (long)nh * NUMBER_PAR_PER_BOX;
                box[nh].nn = 0;
                for (int l = -1; l < 2; l++) {
                    for (int m = -1; m < 2; m++) {
                        for (int n = -1; n < 2; n++) {
                            if ((((i+l) >= 0 && (j+m) >= 0 && (k+n) >= 0) &&
                                 ((i+l) < boxes1d && (j+m) < boxes1d && (k+n) < boxes1d)) &&
                                !(l == 0 && m == 0 && n == 0)) {
                                int nn = box[nh].nn;
                                box[nh].nei[nn].x = (k+n);
                                box[nh].nei[nn].y = (j+m);
                                box[nh].nei[nn].z = (i+l);
                                box[nh].nei[nn].number =
                                    (box[nh].nei[nn].z * boxes1d * boxes1d) +
                                    (box[nh].nei[nn].y * boxes1d) +
                                     box[nh].nei[nn].x;
                                box[nh].nei[nn].offset =
                                    (long)box[nh].nei[nn].number * NUMBER_PAR_PER_BOX;
                                box[nh].nn = nn + 1;
                            }
                        }
                    }
                }
                nh = nh + 1;
            }
        }
    }

    /* ---- DETERMINISTIC INPUT GENERATION (pinned srand(7)) ---- */
    srand(7);
    for (long i = 0; i < space_elem; i++) {
        rv[i].v = (fp)((rand() % 10 + 1) / 10.0);
        rv[i].x = (fp)((rand() % 10 + 1) / 10.0);
        rv[i].y = (fp)((rand() % 10 + 1) / 10.0);
        rv[i].z = (fp)((rand() % 10 + 1) / 10.0);
    }
    for (long i = 0; i < space_elem; i++) {
        qv[i] = (fp)((rand() % 10 + 1) / 10.0);
    }
    for (long i = 0; i < space_elem; i++) {
        fv[i].v = 0; fv[i].x = 0; fv[i].y = 0; fv[i].z = 0;
    }

    /* ---- Allocate GPU memory and copy inputs ---- */
    box_str     *d_box;
    FOUR_VECTOR *d_rv, *d_fv;
    fp          *d_qv;

    cudaMalloc((void**)&d_box, number_boxes * sizeof(box_str));
    cudaMalloc((void**)&d_rv,  space_elem   * sizeof(FOUR_VECTOR));
    cudaMalloc((void**)&d_qv,  space_elem   * sizeof(fp));
    cudaMalloc((void**)&d_fv,  space_elem   * sizeof(FOUR_VECTOR));

    cudaMemcpy(d_box, box, number_boxes * sizeof(box_str),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_rv,  rv,  space_elem   * sizeof(FOUR_VECTOR), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qv,  qv,  space_elem   * sizeof(fp),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_fv,  fv,  space_elem   * sizeof(FOUR_VECTOR), cudaMemcpyHostToDevice);

    /* ---- Launch kernel: one block per box, 100 threads per block ---- */
    dim3 block(NUMBER_PAR_PER_BOX, 1, 1);
    dim3 grid((unsigned int)number_boxes, 1, 1);

    cudaDeviceSynchronize();
    double t0 = now_seconds();

    lavaMD_kernel<<<grid, block>>>(d_box, d_rv, d_qv, d_fv, number_boxes, a2);
    cudaDeviceSynchronize();

    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    /* ---- Copy results back ---- */
    cudaMemcpy(fv, d_fv, space_elem * sizeof(FOUR_VECTOR), cudaMemcpyDeviceToHost);

    cudaFree(d_box); cudaFree(d_rv); cudaFree(d_qv); cudaFree(d_fv);

    /* ---- OUTPUT ---- */
    FILE *fp_out = fopen(ofile, "w");
    if (!fp_out) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    long idx = 0;
    for (long i = 0; i < space_elem; i++) {
        fprintf(fp_out, "%ld\t%.6f\n", idx++, fv[i].v);
        fprintf(fp_out, "%ld\t%.6f\n", idx++, fv[i].x);
        fprintf(fp_out, "%ld\t%.6f\n", idx++, fv[i].y);
        fprintf(fp_out, "%ld\t%.6f\n", idx++, fv[i].z);
    }
    fclose(fp_out);

    free(box); free(rv); free(qv); free(fv);
    return 0;
}
