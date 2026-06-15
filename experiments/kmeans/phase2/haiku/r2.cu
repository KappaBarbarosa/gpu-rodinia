/*
 * K-Means CUDA implementation (Phase 2 Round 3 optimized)
 *
 * Strategy: keep r1's two-phase approach (load features → register array,
 * then iterate clusters) which proved faster than the interleaved r2 approach.
 * Add what r1 was missing: template parameters so BOTH loops get compile-time
 * unrolling, and use NFEATURES as a compile-time constant in the centroid offset
 * computation to eliminate the runtime multiply c * nfeatures.
 *
 * Key changes vs r1:
 *   1. Template<NCLUSTERS=5, NFEATURES=34>: makes outer cluster loop a
 *      compile-time constant => `#pragma unroll` on the cluster loop works,
 *      eliminating all branch overhead and enabling better ILP scheduling.
 *   2. Centroid offset `c * NFEATURES + f` is now compile-time, removing
 *      the runtime multiply for cptr per cluster.
 *   3. Block size 128: with 34+1 floats/ints per thread in registers, 128
 *      threads/block gives more register headroom, reducing spilling vs 256.
 *   4. No __launch_bounds__: avoids artificially capping compiler register
 *      allocation (r2 showed this hurt performance).
 *
 * GPU: assign step (find nearest centroid for each point)
 * CPU: update step (recompute centroids, maintain oracle correctness)
 *
 * CLI: <prog> <npoints> <nfeatures> <nclusters> <max_iter> <output_file>
 * Output: one line per point: "<index>\t<cluster_id>\n"
 * Stderr: compute_seconds=<t>
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

/*
 * Constant memory for centroids: 5 clusters * 34 features = 170 floats = 680 bytes.
 * All warps reading the same centroid element get a broadcast (one transaction).
 */
#define MAX_CLUSTERS 5
#define MAX_FEATURES 34

__constant__ float c_clusters[MAX_CLUSTERS * MAX_FEATURES];

/*
 * GPU kernel: assign step
 *
 * Two-phase approach (proven faster than interleaved in r1 vs r2 comparison):
 *
 * Phase 1 – Load:
 *   Load all NFEATURES values for point i into register array fval[].
 *   Accesses are coalesced: consecutive threads in a warp load consecutive
 *   elements of each feature row ([nfeatures][npoints] layout).
 *
 * Phase 2 – Compute:
 *   For each cluster c, compute squared Euclidean distance entirely from
 *   registers (fval[]) and constant memory (c_clusters[]).
 *   Both loops have compile-time bounds via template params => both fully
 *   unrolled, no branch overhead, compiler can schedule across iterations.
 *
 * Template params make `c * NFEATURES` a compile-time constant, so the
 * compiler folds the centroid base address per cluster into an immediate.
 */
template<int NCLUSTERS, int NFEATURES>
__global__ void kmeans_assign(
    const float *__restrict__ features,   /* [NFEATURES][npoints] transposed */
    int         *__restrict__ membership, /* [npoints] output */
    int npoints
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;

    /* Phase 1: load all features into registers (coalesced, read-only cache) */
    float fval[NFEATURES];
    #pragma unroll
    for (int f = 0; f < NFEATURES; f++) {
        fval[f] = __ldg(features + f * npoints + i);
    }

    float min_dist = FLT_MAX;
    int best = 0;

    /* Phase 2: compute distance to each centroid from register values */
    #pragma unroll
    for (int c = 0; c < NCLUSTERS; c++) {
        float dist = 0.0f;
        /* c * NFEATURES is a compile-time constant with template params */
        #pragma unroll
        for (int f = 0; f < NFEATURES; f++) {
            float d = fval[f] - c_clusters[c * NFEATURES + f];
            dist += d * d;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best = c;
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

    if (nclusters > MAX_CLUSTERS || nfeatures > MAX_FEATURES) {
        fprintf(stderr, "Error: nclusters=%d > MAX_CLUSTERS=%d or nfeatures=%d > MAX_FEATURES=%d\n",
                nclusters, MAX_CLUSTERS, nfeatures, MAX_FEATURES);
        return 1;
    }

    /* --- generate input data: srand(7), features in [0, 500) --- */
    srand(7);

    /* Row-major [npoints][nfeatures] for CPU update step */
    float *features_row = (float *)malloc(npoints * nfeatures * sizeof(float));
    for (int i = 0; i < npoints * nfeatures; i++)
        features_row[i] = (float)rand() / RAND_MAX * 500.0f;

    /* Transposed [nfeatures][npoints] for GPU coalesced access */
    float *features_col = (float *)malloc(nfeatures * npoints * sizeof(float));
    for (int p = 0; p < npoints; p++)
        for (int f = 0; f < nfeatures; f++)
            features_col[f * npoints + p] = features_row[p * nfeatures + f];

    /* --- init centroids: first nclusters points --- */
    float *clusters = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (int c = 0; c < nclusters; c++)
        for (int f = 0; f < nfeatures; f++)
            clusters[c * nfeatures + f] = features_row[c * nfeatures + f];

    int   *membership     = (int   *)malloc(npoints  * sizeof(int));
    float *new_centers    = (float *)calloc(nclusters * nfeatures, sizeof(float));
    int   *new_center_len = (int   *)calloc(nclusters, sizeof(int));

    for (int i = 0; i < npoints; i++) membership[i] = -1;

    /* --- GPU memory allocation --- */
    float *d_features   = NULL;
    int   *d_membership = NULL;

    check_cuda(cudaMalloc(&d_features,   nfeatures * npoints * sizeof(float)), "malloc d_features");
    check_cuda(cudaMalloc(&d_membership, npoints   * sizeof(int)),              "malloc d_membership");

    /* Copy transposed features to GPU once (static across all iterations) */
    check_cuda(cudaMemcpy(d_features, features_col, nfeatures * npoints * sizeof(float),
                          cudaMemcpyHostToDevice), "copy transposed features to GPU");

    /* --- main iteration loop --- */
    double t0 = now();

    /*
     * 128 threads/block: with 34 floats in fval[] + dist + best in registers,
     * 128 threads gives more register budget per thread than 256, avoiding
     * spilling to local memory while still providing enough warps for latency hiding.
     */
    int threads_per_block = 128;
    int blocks = (npoints + threads_per_block - 1) / threads_per_block;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Upload updated centroids to constant memory (680 bytes) */
        check_cuda(cudaMemcpyToSymbol(c_clusters, clusters, nclusters * nfeatures * sizeof(float)),
                   "copy clusters to constant memory");

        /* GPU: assign step — template specialization for NCLUSTERS=5, NFEATURES=34 */
        kmeans_assign<5, 34><<<blocks, threads_per_block>>>(
            d_features, d_membership, npoints
        );
        check_cuda(cudaGetLastError(), "assign kernel");
        check_cuda(cudaDeviceSynchronize(), "sync after assign");

        /* Copy membership back to host */
        check_cuda(cudaMemcpy(membership, d_membership, npoints * sizeof(int),
                              cudaMemcpyDeviceToHost), "copy membership to host");

        /* CPU: update step (exact accumulation order as serial reference) */
        for (int c = 0; c < nclusters * nfeatures; c++) new_centers[c] = 0.0f;
        for (int c = 0; c < nclusters; c++) new_center_len[c] = 0;

        for (int i = 0; i < npoints; i++) {
            int c = membership[i];
            new_center_len[c]++;
            for (int f = 0; f < nfeatures; f++)
                new_centers[c * nfeatures + f] += features_row[i * nfeatures + f];
        }
        for (int c = 0; c < nclusters; c++)
            if (new_center_len[c] > 0)
                for (int f = 0; f < nfeatures; f++)
                    clusters[c * nfeatures + f] = new_centers[c * nfeatures + f] / new_center_len[c];
    }

    double t1 = now();
    fprintf(stderr, "compute_seconds=%.6f\n", t1 - t0);

    /* --- output: index\tcluster_id --- */
    FILE *fp = fopen(outfile, "w");
    if (!fp) { perror("fopen"); return 1; }
    for (int i = 0; i < npoints; i++)
        fprintf(fp, "%d\t%d\n", i, membership[i]);
    fclose(fp);

    /* --- cleanup --- */
    cudaFree(d_features);
    cudaFree(d_membership);
    free(features_row);
    free(features_col);
    free(clusters);
    free(membership);
    free(new_centers);
    free(new_center_len);

    return 0;
}
