/*
 * lavaMD — serial reference.
 *
 * Port of the Rodinia lavaMD benchmark (openmp/lavaMD/{main.c,kernel/kernel_cpu.c}),
 * made self-contained, deterministic, and instrumented for the experiment harness.
 *
 * lavaMD is an N-body-style molecular-dynamics kernel. Space is partitioned into a
 * regular 3-D grid of boxes_1d^3 boxes; each box holds NUMBER_PAR_PER_BOX (=100)
 * particles. Each particle carries a 4-vector position/charge-distance rv = {v,x,y,z}
 * and a scalar charge qv. For every particle in every "home" box, we accumulate a
 * 4-vector force/potential fv = {v,x,y,z} from its pairwise interaction with every
 * particle in the home box and its (up to 26) neighbour boxes. The per-pair physics
 * (Gaussian-screened force) is reproduced VERBATIM from kernel_cpu.c.
 *
 * There is NO external data file: the input (positions/charges) is GENERATED. The
 * original seeds with srand(time(NULL)); this reference pins srand(7) so the input
 * — and therefore the golden — is fully reproducible. The exact generation order is
 * documented in ALGORITHM.md so a blind CUDA author reproduces the SAME input.
 *
 * Numerics: the original uses `fp = double`. This experiment standardizes on float32
 * for the GPU comparison, so ALL particle data and accumulation are `float` here.
 * a2 = 2*alpha*alpha with alpha=0.5 (so a2=0.5). The accumulation is a sum of bounded,
 * similar-magnitude contributions (well-conditioned), so a reordered GPU reduction
 * stays close to this serial order — see ALGORITHM.md and the manifest tolerance note.
 *
 * Each home box's particles are INDEPENDENT of every other box's output (fv is written
 * only for the home box; neighbours are read-only). This is the data-parallelism the
 * GPU port exploits (classic port: one thread-block per box).
 *
 * CLI:  lavaMD_serial <boxes1d> <output_file>
 *   boxes1d : number of boxes per dimension; total boxes = boxes1d^3,
 *             total particles = boxes1d^3 * 100.
 *   output  : for each particle (index 0..total-1, box-major) its fv vector, flattened
 *             as 4 lines per particle in order v,x,y,z. Line format "<index>\t<value>\n"
 *             with index 0..4*total-1, value printed %.6f.
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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

    /* parameters (verbatim from main.c / kernel_cpu.c) */
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

    /* ---- DETERMINISTIC INPUT GENERATION (pinned srand(7)) ----
     * Original uses srand(time(NULL)); we pin the seed. Generation ORDER matches
     * main.c EXACTLY: first all rv (4 rand() calls per particle, in v,x,y,z order),
     * then all qv (1 rand() call per particle). Each value = (rand()%10 + 1)/10.0,
     * i.e. uniform in {0.1,0.2,...,1.0}. */
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

    /* ---- KERNEL (verbatim interaction from kernel_cpu.c) ---- */
    double t0 = now_seconds();
    for (long l = 0; l < number_boxes; l++) {
        long first_i = box[l].offset;
        FOUR_VECTOR *rA = &rv[first_i];
        FOUR_VECTOR *fA = &fv[first_i];

        for (int k = 0; k < (1 + box[l].nn); k++) {
            int pointer = (k == 0) ? (int)l : box[l].nei[k-1].number;
            long first_j = box[pointer].offset;
            FOUR_VECTOR *rB = &rv[first_j];
            fp          *qB = &qv[first_j];

            for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {
                for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                    fp r2  = rA[i].v + rB[j].v - DOT(rA[i], rB[j]);
                    fp u2  = a2 * r2;
                    fp vij = expf(-u2);
                    fp fs  = 2.0f * vij;
                    THREE_VECTOR d;
                    d.x = rA[i].x - rB[j].x;
                    d.y = rA[i].y - rB[j].y;
                    d.z = rA[i].z - rB[j].z;
                    fp fxij = fs * d.x;
                    fp fyij = fs * d.y;
                    fp fzij = fs * d.z;

                    fA[i].v += qB[j] * vij;
                    fA[i].x += qB[j] * fxij;
                    fA[i].y += qB[j] * fyij;
                    fA[i].z += qB[j] * fzij;
                }
            }
        }
    }
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    /* ---- OUTPUT: flatten fv to one value per line, 4 per particle (v,x,y,z) ---- */
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
