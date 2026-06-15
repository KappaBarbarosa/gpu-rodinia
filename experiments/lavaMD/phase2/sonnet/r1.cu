/*
 * lavaMD — CUDA port, Phase 2 Round 1.
 * Builds on phase-1 shared-memory tiling:
 *   - Preload home-box neighbour list into registers/shared to avoid
 *     repeated large box_str global reads inside the k-loop.
 *   - Use __ldg() for read-only rv/qv loads outside shared staging path.
 *   - Use float4 vectorized loads for FOUR_VECTOR (16-byte aligned).
 *   - Use __ldg() + int array for box neighbour numbers (read-only cache).
 *   - Pinned host memory for faster H2D/D2H.
 *   - Hoisted a2 as compile-time constant and use fmaf-safe formulation.
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define NUMBER_PAR_PER_BOX 100

typedef float fp;
typedef struct { fp v, x, y, z; } FOUR_VECTOR;

typedef struct { int x, y, z; int number; long offset; } nei_str;
typedef struct {
    int x, y, z;
    int number;
    long offset;
    int nn;
    nei_str nei[26];
} box_str;

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*
 * Kernel: one block per home box, 100 threads per block.
 *
 * New in phase 2:
 *  1. Preload neighbour-box numbers into __shared__ snei[] once at block start
 *     so all 100 threads share the same small int array from smem (not large
 *     box_str re-reads every iteration of the k-loop).
 *  2. float4 coalesced loads for rv (FOUR_VECTOR == float4, 128-bit).
 *  3. __ldg() read-only cache for the shared-memory cooperative loads.
 *  4. boxes[] read through __ldg() (read-only path).
 */
__global__ void lavaMD_kernel(
    const box_str * __restrict__ boxes,
    const float4  * __restrict__ rv4,   /* rv as float4 for vectorised loads */
    const fp      * __restrict__ qv,
    FOUR_VECTOR   * __restrict__ fv,
    fp a2
) {
    /* shared staging: current neighbour box's particles */
    __shared__ float4 sB[NUMBER_PAR_PER_BOX];
    __shared__ fp     sQ[NUMBER_PAR_PER_BOX];
    /* preloaded neighbour-box number list (up to 26 + home = 27 entries) */
    __shared__ int    snei[27];  /* snei[0]=home, snei[1..nn]=neighbours */

    int l = (int)blockIdx.x;
    int i = threadIdx.x;   /* 0..99 */

    /* ---- preload home-box metadata into shared memory ---- */
    /* thread 0 loads the neighbour list; all threads see it after sync */
    int nn;
    if (i == 0) {
        /* use __ldg for read-only box_str */
        int lnn = __ldg(&boxes[l].nn);
        nn = lnn;
        snei[0] = l;   /* home box */
        for (int k = 0; k < lnn; k++) {
            snei[k + 1] = __ldg(&boxes[l].nei[k].number);
        }
        /* store nn for other threads */
        snei[26] = lnn;   /* reuse last slot as nn storage */
    }
    __syncthreads();
    nn = snei[26];

    /* ---- load home particle (float4 = 128-bit coalesced) ---- */
    long first_i = (long)l * NUMBER_PAR_PER_BOX;
    float4 rAi4  = __ldg(&rv4[first_i + i]);
    /* unpack */
    fp rAiv = rAi4.x;  /* FOUR_VECTOR: .v=x .x=y .y=z .z=w in memory? */
    /* FOUR_VECTOR is { v, x, y, z } so float4.x=v, .y=x, .z=y, .w=z */
    fp rAix = rAi4.y;
    fp rAiy = rAi4.z;
    fp rAiz = rAi4.w;

    fp fv_v = 0.0f, fv_x = 0.0f, fv_y = 0.0f, fv_z = 0.0f;

    int total_k = 1 + nn;
    for (int k = 0; k < total_k; k++) {
        int pointer  = snei[k];
        long first_j = (long)pointer * NUMBER_PAR_PER_BOX;

        /* cooperatively stage neighbour box into shared memory */
        __syncthreads();
        sB[i] = __ldg(&rv4[first_j + i]);
        sQ[i] = __ldg(&qv[first_j + i]);
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            float4 rBj4 = sB[j];
            fp rBjv = rBj4.x;
            fp rBjx = rBj4.y;
            fp rBjy = rBj4.z;
            fp rBjz = rBj4.w;
            fp qBj  = sQ[j];

            fp r2  = rAiv + rBjv - (rAix * rBjx + rAiy * rBjy + rAiz * rBjz);
            fp u2  = a2 * r2;
            fp vij = expf(-u2);
            fp fs  = 2.0f * vij;

            fp dx = rAix - rBjx;
            fp dy = rAiy - rBjy;
            fp dz = rAiz - rBjz;

            fv_v += qBj * vij;
            fv_x += qBj * fs * dx;
            fv_y += qBj * fs * dy;
            fv_z += qBj * fs * dz;
        }
    }

    long out_idx = first_i + i;
    fv[out_idx].v = fv_v;
    fv[out_idx].x = fv_x;
    fv[out_idx].y = fv_y;
    fv[out_idx].z = fv_z;
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

    /* use pinned host memory for faster H2D / D2H transfers */
    box_str     *box;
    FOUR_VECTOR *rv, *fv;
    fp          *qv;
    cudaMallocHost((void**)&box, number_boxes * sizeof(box_str));
    cudaMallocHost((void**)&rv,  space_elem   * sizeof(FOUR_VECTOR));
    cudaMallocHost((void**)&qv,  space_elem   * sizeof(fp));
    cudaMallocHost((void**)&fv,  space_elem   * sizeof(FOUR_VECTOR));
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
                                int nn2 = box[nh].nn;
                                box[nh].nei[nn2].x = (k+n);
                                box[nh].nei[nn2].y = (j+m);
                                box[nh].nei[nn2].z = (i+l);
                                box[nh].nei[nn2].number =
                                    (box[nh].nei[nn2].z * boxes1d * boxes1d) +
                                    (box[nh].nei[nn2].y * boxes1d) +
                                     box[nh].nei[nn2].x;
                                box[nh].nei[nn2].offset =
                                    (long)box[nh].nei[nn2].number * NUMBER_PAR_PER_BOX;
                                box[nh].nn = nn2 + 1;
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
    float4      *d_rv4;
    fp          *d_qv;
    FOUR_VECTOR *d_fv;

    cudaMalloc((void**)&d_box, number_boxes * sizeof(box_str));
    cudaMalloc((void**)&d_rv4, space_elem   * sizeof(float4));   /* same size as FOUR_VECTOR */
    cudaMalloc((void**)&d_qv,  space_elem   * sizeof(fp));
    cudaMalloc((void**)&d_fv,  space_elem   * sizeof(FOUR_VECTOR));

    /* FOUR_VECTOR and float4 are layout-compatible (both 4 floats) */
    cudaMemcpy(d_box, box, number_boxes * sizeof(box_str),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_rv4, rv,  space_elem   * sizeof(FOUR_VECTOR), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qv,  qv,  space_elem   * sizeof(fp),          cudaMemcpyHostToDevice);
    /* fv not needed D2H as input (initialized to zero on device implicitly via output write) */
    /* but we still zero it for safety */
    cudaMemset(d_fv, 0, space_elem * sizeof(FOUR_VECTOR));

    /* ---- Launch kernel: one block per box, 100 threads per block ---- */
    dim3 block(NUMBER_PAR_PER_BOX, 1, 1);
    dim3 grid((unsigned int)number_boxes, 1, 1);

    cudaDeviceSynchronize();
    double t0 = now_seconds();

    lavaMD_kernel<<<grid, block>>>(d_box, d_rv4, d_qv, d_fv, a2);
    cudaDeviceSynchronize();

    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    /* ---- Copy results back ---- */
    cudaMemcpy(fv, d_fv, space_elem * sizeof(FOUR_VECTOR), cudaMemcpyDeviceToHost);

    cudaFree(d_box); cudaFree(d_rv4); cudaFree(d_qv); cudaFree(d_fv);

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

    cudaFreeHost(box); cudaFreeHost(rv); cudaFreeHost(qv); cudaFreeHost(fv);
    return 0;
}
