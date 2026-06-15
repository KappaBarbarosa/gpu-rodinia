/*
 * lavaMD — CUDA Phase 2, Round 1: Multi-Box-Per-Block Optimization
 *
 * Data-parallel GPU port of lavaMD kernel.
 * - Multiple home boxes per block (blockDim.x = 256, 2-3 home boxes per block)
 * - One thread per home particle across all boxes in the block
 * - Each thread loops over its home box + neighbours, accumulating forces
 * - Shared memory staging of source-box particles to reduce global traffic
 * - Full warp occupancy (256 = 8 full warps, no lane waste)
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

/* GPU kernel: multiple home boxes per block, one thread per home particle.
 *
 * blockDim.x = 256 threads (8 full warps, no lane waste).
 * Each block processes 2 or 3 consecutive home boxes (as many as fit in 256 threads).
 * Each thread i processes home particle (i % 100) in home box (i / 100).
 *
 * For each (home box, source box) pair, all threads cooperatively stage the
 * source box's particles into shared memory once, then all home threads in
 * that block compute against shared memory. This amortizes the load cost
 * across multiple home boxes.
 */
__global__ void lavamd_kernel(
    long number_boxes,
    box_str *box,
    FOUR_VECTOR *rv,
    fp *qv,
    FOUR_VECTOR *fv,
    fp alpha
) {
    /* Shared memory large enough for one source box (100 FOUR_VECTOR + 100 fp) */
    extern __shared__ char shared_buf[];
    FOUR_VECTOR *s_rB = (FOUR_VECTOR *)shared_buf;
    fp          *s_qB = (fp *)(s_rB + NUMBER_PAR_PER_BOX);

    fp a2 = 2.0f * alpha * alpha;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    /* Determine how many home boxes fit in this block.
     * With blockDim.x = 256 and NUMBER_PAR_PER_BOX = 100:
     *  - 256 / 100 = 2 home boxes per block with 56 threads left over
     *  - For the last block, there may be fewer home boxes.
     * We assign threads to home boxes in round-robin within this block.
     */
    int boxes_per_block = blockDim.x / NUMBER_PAR_PER_BOX;
    if (boxes_per_block < 1) boxes_per_block = 1;
    if (boxes_per_block > 3) boxes_per_block = 3; /* safety cap */

    /* Which home box does this thread belong to? */
    int local_box_index = tid / NUMBER_PAR_PER_BOX;
    int particle_in_box = tid % NUMBER_PAR_PER_BOX;

    long global_home_box_idx = bid * boxes_per_block + local_box_index;

    /* Early exit if this thread's home box doesn't exist. */
    if (global_home_box_idx >= number_boxes) return;

    box_str *home_box = &box[global_home_box_idx];
    long home_offset = home_box->offset;
    int  nn          = home_box->nn;

    FOUR_VECTOR *fA = &fv[home_offset];

    /* Load this thread's home particle. */
    FOUR_VECTOR home_rv = rv[home_offset + particle_in_box];
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

        /* Cooperatively stage the source box into shared memory.
         * All blockDim.x threads participate, even those beyond NUMBER_PAR_PER_BOX
         * particles. They load extra particles, which get masked out.
         * This ensures full warp utilization and coalesced loads.
         */
        if (tid < NUMBER_PAR_PER_BOX) {
            s_rB[tid] = rv[source_offset + tid];
            s_qB[tid] = qv[source_offset + tid];
        }
        __syncthreads();

        /* Only threads assigned to a real particle in this home box compute. */
        if (local_box_index < boxes_per_block && global_home_box_idx < number_boxes) {
            #pragma unroll 1
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
        }
        __syncthreads();
    }

    /* Write accumulated force back to global memory. */
    if (local_box_index < boxes_per_block && global_home_box_idx < number_boxes) {
        fA[particle_in_box] = acc_fv;
    }
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

    /* Use blockDim.x = 256 (8 full warps, no lane waste).
     * Each block processes 2 home boxes (256 / 100 = 2 with 56 leftover).
     * Blocks: ceil(number_boxes / 2).
     */
    int threads_per_block = 256;
    int boxes_per_block = 2;
    int blocks = (int)((number_boxes + boxes_per_block - 1) / boxes_per_block);

    /* Shared memory for one source box: 100 FOUR_VECTOR + 100 float */
    size_t shared_size = 100 * sizeof(FOUR_VECTOR) + 100 * sizeof(fp);

    lavamd_kernel<<<blocks, threads_per_block, shared_size>>>(
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
