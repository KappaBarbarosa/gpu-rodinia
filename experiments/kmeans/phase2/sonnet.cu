/*
 * K-Means CUDA Phase 2
 *
 * Key optimizations over phase 1 (global-memory, cluster-outer loop):
 *
 *   1. INVERTED LOOP ORDER: outer=feature, inner=cluster.
 *      Each feature value is loaded once and accumulated into per-cluster
 *      registers.  Phase 1 loaded each feature NC times (once per cluster);
 *      this version loads it exactly once.
 *
 *   2. CONSTANT MEMORY for centroids (5*34*4 = 680 bytes).
 *      The centroid array is small and read uniformly across the warp, so
 *      it lives in constant-memory broadcast cache rather than L2.
 *
 *   3. TRANSPOSED FEATURE LAYOUT [nfeatures][npoints].
 *      Thread i reads features_T[f*npoints + i]: adjacent threads touch
 *      adjacent floats -> fully coalesced 128-byte transactions.
 *
 *   4. FULLY UNROLLED INNER CLUSTER LOOP via template parameter NC_STATIC.
 *      The compiler unrolls the inner cluster loop completely, eliminating
 *      branches and enabling FMA chaining.  A runtime guard handles
 *      nclusters < NC_STATIC.
 *
 *   5. PARTIAL UNROLL (x4) of the feature loop to hide memory latency.
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * Constant memory for centroids.
 * Workload: 5 clusters * 34 features * 4 bytes = 680 bytes.
 * We size to MAX_CLUSTERS * MAX_FEATURES to be safe.
 */
#define MAX_CLUSTERS 64
#define MAX_FEATURES 64
__constant__ float c_clusters[MAX_CLUSTERS * MAX_FEATURES];

/*
 * NC_STATIC: compile-time upper bound on nclusters for loop unrolling.
 * At runtime we guard with (c < nclusters) for unused slots.
 * 8 is a reasonable static size; extend if needed.
 */
#define NC_STATIC 8

/*
 * Kernel: inverted-loop k-means assign step.
 *
 * For each point i:
 *   for f in 0..nfeatures:
 *     fv = features_T[f * npoints + i]     (coalesced load)
 *     for c in 0..NC_STATIC:
 *       accum[c] += (fv - centroid[c][f])^2
 *   membership[i] = argmin(accum)
 */
__global__ void assign_kernel(
    const float * __restrict__ features_T,   /* [nfeatures][npoints], transposed */
    int           * __restrict__ membership,
    int npoints,
    int nfeatures,
    int nclusters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;

    /* per-cluster distance accumulators, kept in registers */
    float accum[NC_STATIC];
    #pragma unroll
    for (int c = 0; c < NC_STATIC; c++) accum[c] = 0.0f;

    /* outer loop: one feature dimension at a time */
    #pragma unroll 4
    for (int f = 0; f < nfeatures; f++) {
        float fv = features_T[(size_t)f * npoints + i];  /* coalesced */
        /* inner loop fully unrolled; compiler emits FMAs */
        #pragma unroll
        for (int c = 0; c < NC_STATIC; c++) {
            float d = fv - c_clusters[c * nfeatures + f]; /* constant-mem broadcast */
            accum[c] += d * d;
        }
    }

    /* find argmin (slot 0 is always valid for nclusters >= 1) */
    float min_dist = accum[0];
    int   best     = 0;
    #pragma unroll
    for (int c = 1; c < NC_STATIC; c++) {
        if (c < nclusters && accum[c] < min_dist) {
            min_dist = accum[c];
            best     = c;
        }
    }
    membership[i] = best;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <npoints> <nfeatures> <nclusters> <max_iter> <output_file>\n", argv[0]);
        return 1;
    }
    int npoints   = atoi(argv[1]);
    int nfeatures = atoi(argv[2]);
    int nclusters = atoi(argv[3]);
    int max_iter  = atoi(argv[4]);
    const char *outfile = argv[5];

    if (nclusters > NC_STATIC) {
        fprintf(stderr, "Error: nclusters (%d) exceeds NC_STATIC (%d). Increase NC_STATIC.\n",
                nclusters, NC_STATIC);
        return 1;
    }
    if (nclusters > MAX_CLUSTERS || nfeatures > MAX_FEATURES) {
        fprintf(stderr, "Error: nclusters or nfeatures exceeds constant memory limits.\n");
        return 1;
    }

    /* generate input data (row-major: features[i * nfeatures + f]) */
    srand(7);
    float *features = (float *)malloc(npoints * nfeatures * sizeof(float));
    for (int i = 0; i < npoints * nfeatures; i++)
        features[i] = (float)rand() / RAND_MAX * 500.0f;

    /* init centroids: first nclusters points */
    float *clusters = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (int c = 0; c < nclusters; c++)
        for (int f = 0; f < nfeatures; f++)
            clusters[c * nfeatures + f] = features[c * nfeatures + f];

    /* transpose features to [nfeatures][npoints] for coalesced GPU reads */
    float *features_T = (float *)malloc((size_t)nfeatures * npoints * sizeof(float));
    for (int f = 0; f < nfeatures; f++)
        for (int i = 0; i < npoints; i++)
            features_T[(size_t)f * npoints + i] = features[i * nfeatures + f];

    int   *membership     = (int   *)malloc(npoints * sizeof(int));
    float *new_centers    = (float *)calloc(nclusters * nfeatures, sizeof(float));
    int   *new_center_len = (int   *)calloc(nclusters, sizeof(int));

    for (int i = 0; i < npoints; i++) membership[i] = -1;

    /* device allocations */
    float *d_features_T;
    int   *d_membership;

    cudaMalloc((void **)&d_features_T, (size_t)nfeatures * npoints * sizeof(float));
    cudaMalloc((void **)&d_membership, npoints * sizeof(int));

    /* copy transposed features once (invariant across iterations) */
    cudaMemcpy(d_features_T, features_T,
               (size_t)nfeatures * npoints * sizeof(float),
               cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size  = (npoints + block_size - 1) / block_size;

    double t0 = now();

    for (int iter = 0; iter < max_iter; iter++) {
        /* upload updated centroids to constant memory (680 bytes) */
        cudaMemcpyToSymbol(c_clusters, clusters, nclusters * nfeatures * sizeof(float));

        assign_kernel<<<grid_size, block_size>>>(
            d_features_T, d_membership,
            npoints, nfeatures, nclusters);

        cudaMemcpy(membership, d_membership, npoints * sizeof(int), cudaMemcpyDeviceToHost);

        /* --- update step: CPU (serial, exact oracle match) --- */
        for (int c = 0; c < nclusters * nfeatures; c++) new_centers[c] = 0.0f;
        for (int c = 0; c < nclusters; c++) new_center_len[c] = 0;

        for (int i = 0; i < npoints; i++) {
            int c = membership[i];
            new_center_len[c]++;
            for (int f = 0; f < nfeatures; f++)
                new_centers[c * nfeatures + f] += features[i * nfeatures + f];
        }
        for (int c = 0; c < nclusters; c++)
            if (new_center_len[c] > 0)
                for (int f = 0; f < nfeatures; f++)
                    clusters[c * nfeatures + f] = new_centers[c * nfeatures + f] / new_center_len[c];
    }

    double t1 = now();
    fprintf(stderr, "compute_seconds=%.6f\n", t1 - t0);

    /* write output */
    FILE *fp = fopen(outfile, "w");
    if (!fp) { perror("fopen"); return 1; }
    for (int i = 0; i < npoints; i++)
        fprintf(fp, "%d\t%d\n", i, membership[i]);
    fclose(fp);

    cudaFree(d_features_T);
    cudaFree(d_membership);
    free(features); free(features_T); free(clusters);
    free(membership); free(new_centers); free(new_center_len);
    return 0;
}
