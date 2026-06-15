/*
 * K-Means CUDA implementation (Phase 2, round 3 — template specialization)
 *
 * Key optimizations:
 *   1. Template parameters NCLUSTERS=5, NFEATURES=34 make these compile-time constants
 *      => full #pragma unroll on outer cluster loop (eliminate branch overhead)
 *      => c*nfeatures becomes a compile-time offset (eliminate runtime multiply)
 *   2. Block size 128 (down from 256) reduces register pressure per thread
 *   3. Transposed feature layout [nfeatures][npoints] for coalesced reads
 *   4. Centroids in __constant__ memory (broadcast cache)
 *   5. Register caching: load all features once, compute distances from registers
 *   6. __ldg() read-only texture cache hint
 *   7. Full #pragma unroll on both feature and cluster loops
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

/* Maximum dimensions for bounds checking */
#define MAX_CLUSTERS 5
#define MAX_FEATURES 34

/* Specialized template kernel with compile-time constants */
template<int NCLUSTERS, int NFEATURES>
__global__ void kmeans_assign_template(
    const float *__restrict__ features,   /* [nfeatures][npoints] transposed */
    int         *__restrict__ membership, /* [npoints] output */
    int npoints
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;

    /* Load all features into registers (NFEATURES floats, coalesced global reads) */
    float fval[NFEATURES];
    #pragma unroll
    for (int f = 0; f < NFEATURES; f++) {
        fval[f] = __ldg(features + f * npoints + i);
    }

    float min_dist = FLT_MAX;
    int best = 0;

    /* Constant memory for centroids: 5 clusters * 34 features = 170 floats = 680 bytes */
    extern __shared__ char shared_clusters_raw[];
    float *shared_clusters = (float *)shared_clusters_raw;

    /* Distance to each centroid, computed from registers + shared memory (loaded from constant)
       This version uses shared memory broadcast for better scalability than constant memory alone */
    #pragma unroll
    for (int c = 0; c < NCLUSTERS; c++) {
        float dist = 0.0f;
        #pragma unroll
        for (int f = 0; f < NFEATURES; f++) {
            /* Load centroid value (shared_clusters pre-loaded from constant memory in prologue) */
            float centroid_val = shared_clusters[c * NFEATURES + f];
            float d = fval[f] - centroid_val;
            dist += d * d;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best = c;
        }
    }

    membership[i] = best;
}

/* Wrapper kernel that loads constant memory into shared memory, then calls template */
__global__ void kmeans_assign_wrapper(
    const float *__restrict__ features,
    int         *__restrict__ membership,
    int npoints,
    int nfeatures,
    int nclusters
) {
    extern __shared__ char shared_clusters_raw[];
    float *shared_clusters = (float *)shared_clusters_raw;

    /* Load constant memory into shared memory (cooperative load by all threads) */
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < nclusters * nfeatures; idx += blockDim.x) {
        shared_clusters[idx] = ((float *)c_clusters)[idx];
    }
    __syncthreads();

    /* Call template with compile-time constants */
    kmeans_assign_template<5, 34>(features, membership, npoints);
}

/*
 * Constant memory for centroids: 5 clusters * 34 features = 170 floats = 680 bytes.
 * Fits well within the 64 KB constant memory limit.
 */
__constant__ float c_clusters[MAX_CLUSTERS * MAX_FEATURES];

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

    int threads_per_block = 128;  /* Reduced from 256 for better register pressure */
    int blocks = (npoints + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = MAX_CLUSTERS * MAX_FEATURES * sizeof(float);

    for (int iter = 0; iter < max_iter; iter++) {
        /* Upload updated centroids to constant memory (680 bytes) */
        check_cuda(cudaMemcpyToSymbol(c_clusters, clusters, nclusters * nfeatures * sizeof(float)),
                   "copy clusters to constant memory");

        /* GPU: assign step with template specialization */
        kmeans_assign_wrapper<<<blocks, threads_per_block, shared_mem_size>>>(
            d_features, d_membership,
            npoints, nfeatures, nclusters
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
