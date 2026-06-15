/*
 * lavaMD — Phase 2 CUDA port (shared-memory tiled).
 *
 * One thread-block per home box, one thread per home particle (block == 100).
 * For each interaction box (home + up to 26 neighbours) the 100 threads
 * cooperatively stage that box's rv/qv into shared memory, then every thread
 * reuses the 100 staged particles from shared memory instead of re-reading them
 * from global memory (100x reuse per box). The neighbour box offset is computed
 * directly (number*100) to avoid a second global load from the large box_str
 * array, and __expf replaces expf (verified within tolerance) for the per-pair
 * exponential.
 *
 * CLI:  lavaMD <boxes1d> <output_file>
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define NUMBER_PAR_PER_BOX 100
#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))

typedef float fp;

typedef struct { fp x, y, z; }       THREE_VECTOR;
typedef struct { fp v, x, y, z; }    FOUR_VECTOR;

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

#define CUDA_CHECK(call) do {                                            \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                cudaGetErrorString(_e), __FILE__, __LINE__);            \
        exit(1);                                                         \
    }                                                                    \
} while (0)

/* One block per home box, one thread per home particle (0..99). */
__global__ void lavaMD_kernel(long number_boxes,
                              fp a2,
                              const box_str * __restrict__ box,
                              const FOUR_VECTOR * __restrict__ rv,
                              const fp * __restrict__ qv,
                              FOUR_VECTOR * __restrict__ fv)
{
    long l = blockIdx.x;            /* home box index */
    if (l >= number_boxes) return;
    int i = threadIdx.x;            /* home particle index 0..99 */
    if (i >= NUMBER_PAR_PER_BOX) return;

    __shared__ FOUR_VECTOR sB[NUMBER_PAR_PER_BOX];
    __shared__ fp          sQ[NUMBER_PAR_PER_BOX];

    long first_i = box[l].offset;
    FOUR_VECTOR rA = rv[first_i + i];

    fp acc_v = 0.f, acc_x = 0.f, acc_y = 0.f, acc_z = 0.f;

    int nn = box[l].nn;
    for (int k = 0; k < (1 + nn); k++) {
        int pointer = (k == 0) ? (int)l : box[l].nei[k-1].number;
        long first_j = (long)pointer * NUMBER_PAR_PER_BOX;

        /* Cooperatively stage this interaction box into shared memory. */
        __syncthreads();
        sB[i] = rv[first_j + i];
        sQ[i] = qv[first_j + i];
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            FOUR_VECTOR rBj = sB[j];
            fp r2  = rA.v + rBj.v - DOT(rA, rBj);
            fp u2  = a2 * r2;
            fp vij = __expf(-u2);
            fp fs  = 2.0f * vij;
            fp dx = rA.x - rBj.x;
            fp dy = rA.y - rBj.y;
            fp dz = rA.z - rBj.z;
            fp qj = sQ[j];
            acc_v += qj * vij;
            acc_x += qj * (fs * dx);
            acc_y += qj * (fs * dy);
            acc_z += qj * (fs * dz);
        }
    }

    FOUR_VECTOR out;
    out.v = acc_v; out.x = acc_x; out.y = acc_y; out.z = acc_z;
    fv[first_i + i] = out;
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
    fp a2 = 2.0f * alpha * alpha;

    long number_boxes = (long)boxes1d * boxes1d * boxes1d;
    long space_elem   = number_boxes * NUMBER_PAR_PER_BOX;

    box_str     *box = (box_str *)    malloc(number_boxes * sizeof(box_str));
    FOUR_VECTOR *rv  = (FOUR_VECTOR *)malloc(space_elem   * sizeof(FOUR_VECTOR));
    fp          *qv  = (fp *)         malloc(space_elem   * sizeof(fp));
    FOUR_VECTOR *fv  = (FOUR_VECTOR *)malloc(space_elem   * sizeof(FOUR_VECTOR));
    if (!box || !rv || !qv || !fv) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* ---- BOX / NEIGHBOUR structure (verbatim from main.c) ---- */
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

    /* ---- Device allocations & copies ---- */
    box_str     *d_box;
    FOUR_VECTOR *d_rv, *d_fv;
    fp          *d_qv;
    CUDA_CHECK(cudaMalloc(&d_box, number_boxes * sizeof(box_str)));
    CUDA_CHECK(cudaMalloc(&d_rv,  space_elem   * sizeof(FOUR_VECTOR)));
    CUDA_CHECK(cudaMalloc(&d_qv,  space_elem   * sizeof(fp)));
    CUDA_CHECK(cudaMalloc(&d_fv,  space_elem   * sizeof(FOUR_VECTOR)));

    CUDA_CHECK(cudaMemcpy(d_box, box, number_boxes * sizeof(box_str), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rv,  rv,  space_elem   * sizeof(FOUR_VECTOR), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qv,  qv,  space_elem   * sizeof(fp), cudaMemcpyHostToDevice));

    /* ---- KERNEL (timed: interaction loops only) ---- */
    double t0 = now_seconds();
    dim3 grid((unsigned)number_boxes);
    dim3 block(NUMBER_PAR_PER_BOX);
    lavaMD_kernel<<<grid, block>>>(number_boxes, a2, d_box, d_rv, d_qv, d_fv);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    CUDA_CHECK(cudaMemcpy(fv, d_fv, space_elem * sizeof(FOUR_VECTOR), cudaMemcpyDeviceToHost));

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

    cudaFree(d_box); cudaFree(d_rv); cudaFree(d_qv); cudaFree(d_fv);
    free(box); free(rv); free(qv); free(fv);
    return 0;
}
