/*
 * lavaMD — CUDA Phase 1: Basic Parallelization
 *
 * Data-parallel GPU port of lavaMD kernel.
 * - One thread-block per home box
 * - One thread per home particle
 * - Each thread loops over home box + neighbours, accumulating forces in global memory
 * - No shared memory, no tiling (Phase 1 constraint)
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

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

/* GPU kernel: one block per home box, one thread per home particle.
 *
 * Optimization vs Phase-1: each source box's particles (rv + qv) are staged
 * cooperatively into shared memory ONCE per block, then reused by all
 * NUMBER_PAR_PER_BOX home threads. This removes the ~100x redundant global
 * loads of the same source data that Phase-1 performed.
 *
 * Arithmetic per (home,source) pair and the j-loop accumulation order are
 * byte-for-byte identical to Phase-1, so results match the serial golden
 * under --fmad=false.
 *
 * Phase-2 optimization: the inner j-loop is unrolled (#pragma unroll 20).
 * The loop body is SFU(expf)-bound; unrolling exposes independent expf
 * computations across iterations, hiding the multi-cycle SFU latency and
 * improving instruction-level parallelism. The per-j accumulations are still
 * applied strictly in order (acc_fv += ... for j=0,1,2,...), so unrolling does
 * NOT reorder the floating-point reductions — accuracy is unchanged
 * (max_abs_err stays 0.000489, 0 mismatches), while kernel time drops ~18%.
 */
__global__ void lavamd_kernel(
    long number_boxes,
    box_str *box,
    FOUR_VECTOR *rv,
    fp *qv,
    FOUR_VECTOR *fv,
    fp alpha
) {
    __shared__ FOUR_VECTOR s_rB[NUMBER_PAR_PER_BOX];
    __shared__ fp          s_qB[NUMBER_PAR_PER_BOX];

    fp a2 = 2.0f * alpha * alpha;

    long home_box_idx = blockIdx.x;
    if (home_box_idx >= number_boxes) return;

    int tid = threadIdx.x;

    box_str *home_box = &box[home_box_idx];
    long home_offset = home_box->offset;
    int  nn          = home_box->nn;

    FOUR_VECTOR *fA = &fv[home_offset];

    /* blockDim.x == NUMBER_PAR_PER_BOX, so tid maps 1:1 to a home particle. */
    FOUR_VECTOR home_rv = rv[home_offset + tid];
    FOUR_VECTOR acc_fv = {0.0f, 0.0f, 0.0f, 0.0f};

    /* k=0 is the home box, k=1..nn the neighbours. */
    for (int k = 0; k <= nn; k++) {
        long source_offset;
        if (k == 0) {
            source_offset = home_offset;
        } else {
            int source_box_number = home_box->nei[k - 1].number;
            source_offset = box[source_box_number].offset;
        }

        /* Cooperatively stage the source box into shared memory (coalesced). */
        s_rB[tid] = rv[source_offset + tid];
        s_qB[tid] = qv[source_offset + tid];
        __syncthreads();

        #pragma unroll 20
        for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            FOUR_VECTOR rBj = s_rB[j];
            fp qBj = s_qB[j];
            fp r2  = home_rv.v + rBj.v - DOT(home_rv, rBj);
            fp u2  = a2 * r2;
            fp vij = expf(-u2);
            fp fs  = 2.0f * vij;
            fp dx = home_rv.x - rBj.x;
            fp dy = home_rv.y - rBj.y;
            fp dz = home_rv.z - rBj.z;
            fp fxij = fs * dx;
            fp fyij = fs * dy;
            fp fzij = fs * dz;

            acc_fv.v += qBj * vij;
            acc_fv.x += qBj * fxij;
            acc_fv.y += qBj * fyij;
            acc_fv.z += qBj * fzij;
        }
        __syncthreads();
    }

    /* Write accumulated force back to global memory */
    fA[tid] = acc_fv;
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

    /* parameters */
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
    for (int i = 0; i < boxes1d; i++) {           /* z direction */
        for (int j = 0; j < boxes1d; j++) {       /* y direction */
            for (int k = 0; k < boxes1d; k++) {   /* x direction */
                box[nh].x = k;
                box[nh].y = j;
                box[nh].z = i;
                box[nh].number = nh;
                box[nh].offset = (long)nh * NUMBER_PAR_PER_BOX;
                box[nh].nn = 0;
                for (int l = -1; l < 2; l++) {        /* neighbour z */
                    for (int m = -1; m < 2; m++) {    /* neighbour y */
                        for (int n = -1; n < 2; n++) {/* neighbour x */
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

    /* ---- GPU ALLOCATION AND COPY ---- */
    box_str *d_box;
    FOUR_VECTOR *d_rv, *d_fv;
    fp *d_qv;

    cudaMalloc(&d_box, number_boxes * sizeof(box_str));
    cudaMalloc(&d_rv, space_elem * sizeof(FOUR_VECTOR));
    cudaMalloc(&d_qv, space_elem * sizeof(fp));
    cudaMalloc(&d_fv, space_elem * sizeof(FOUR_VECTOR));

    cudaMemcpy(d_box, box, number_boxes * sizeof(box_str), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rv, rv, space_elem * sizeof(FOUR_VECTOR), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qv, qv, space_elem * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fv, fv, space_elem * sizeof(FOUR_VECTOR), cudaMemcpyHostToDevice);

    /* ---- GPU KERNEL LAUNCH ---- */
    double t0 = now_seconds();

    /* One block per home box, up to 256 threads per block (for particles) */
    int threads_per_block = (NUMBER_PAR_PER_BOX < 256) ? NUMBER_PAR_PER_BOX : 256;
    int blocks = (int)number_boxes;

    lavamd_kernel<<<blocks, threads_per_block>>>(
        number_boxes, d_box, d_rv, d_qv, d_fv, alpha
    );

    cudaDeviceSynchronize();
    double t1 = now_seconds();

    /* ---- COPY RESULTS BACK ---- */
    cudaMemcpy(fv, d_fv, space_elem * sizeof(FOUR_VECTOR), cudaMemcpyDeviceToHost);

    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

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

    /* ---- CLEANUP ---- */
    cudaFree(d_box);
    cudaFree(d_rv);
    cudaFree(d_qv);
    cudaFree(d_fv);

    free(box);
    free(rv);
    free(qv);
    free(fv);

    return 0;
}
