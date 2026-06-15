/*
 * K-Means — Phase 2 round 3 optimized CUDA.
 * GPU: assign step.
 * CPU: centroid update (serial order, exact numerics).
 * CLI: <npoints> <nfeatures> <nclusters> <max_iter> <output_file>
 *
 * Optimizations over r1:
 *  1. Template dispatch on nclusters: for small nclusters (≤8), the cluster
 *     loop is fully unrolled at compile time (NC known), eliminating loop
 *     overhead and reducing dist[] register allocation from 32 to NC floats.
 *     With nclusters=5 this saves 27 registers/thread, roughly doubling warp
 *     occupancy from ~24 to ~48 warps/SM, which hides memory latency better.
 *  2. Constant memory layout changed to [cluster][MAX_FEATURES] with compile-
 *     time stride MAX_FEATURES=64.  Address computation becomes a constant
 *     offset (no runtime multiply) and the compiler can pre-compute all
 *     centroid addresses during unrolling.
 *  3. __ldg() on features_T reads: routes through the read-only data cache
 *     (L1 texture path) on sm_35+, which does not conflict with regular L1.
 *  4. #pragma unroll 2 on the nfeatures loop: doubles the instruction window
 *     visible to the warp scheduler, enabling better latency hiding for
 *     back-to-back global loads 256KB apart in features_T.
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_CLUSTERS  32
#define MAX_FEATURES  64

/*
 * Centroid layout: d_clusters_const[c * MAX_FEATURES + f]
 * Stride MAX_FEATURES is compile-time, allowing the compiler to pre-compute
 * all centroid offsets during loop unrolling.  The warp still broadcasts
 * (all threads read the same address), so constant-cache hit rate is 100%.
 */
__constant__ float d_clusters_const[MAX_CLUSTERS * MAX_FEATURES];

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * Templated assign kernel — NC is the exact number of clusters.
 *
 * With NC known at compile time:
 *   - dist[NC] occupies exactly NC registers (vs MAX_CLUSTERS=32).
 *   - The inner cluster loop is fully unrolled (#pragma unroll with NC).
 *   - The min-find loop is fully unrolled.
 *   - No loop-control overhead; better instruction scheduling.
 *
 * nfeatures is still runtime (outer loop), but #pragma unroll 2 widens the
 * instruction window so the scheduler can overlap two back-to-back global
 * loads from different feature rows.
 */
template <int NC>
__global__ void assign_kernel_nc(const float * __restrict__ features_T,
                                  int * __restrict__ membership,
                                  int npoints, int nfeatures) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npoints) return;

    float dist[NC];
    #pragma unroll
    for (int c = 0; c < NC; c++) dist[c] = 0.0f;

    /* Unroll 2: two feature iterations per loop body for better pipelining. */
    #pragma unroll 2
    for (int f = 0; f < nfeatures; f++) {
        /* Read-only cache path: features_T[f*npoints + p], coalesced across warp. */
        float fv = __ldg(&features_T[(size_t)f * npoints + p]);

        /* Compile-time constant addresses: c*MAX_FEATURES+f for c=0..NC-1.
         * All 32 warp threads read the same address → broadcast from const cache. */
        #pragma unroll
        for (int c = 0; c < NC; c++) {
            float d = fv - d_clusters_const[c * MAX_FEATURES + f];
            dist[c] += d * d;
        }
    }

    float min_dist = dist[0];
    int index = 0;
    #pragma unroll
    for (int c = 1; c < NC; c++) {
        if (dist[c] < min_dist) { min_dist = dist[c]; index = c; }
    }
    membership[p] = index;
}

/*
 * Fallback kernel for nclusters > 8: same as r1 but with __ldg and
 * compile-time MAX_FEATURES stride for constant memory.
 */
__global__ void assign_kernel_generic(const float * __restrict__ features_T,
                                       int * __restrict__ membership,
                                       int npoints, int nfeatures, int nclusters) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npoints) return;

    float dist[MAX_CLUSTERS];
    for (int c = 0; c < nclusters; c++) dist[c] = 0.0f;

    for (int f = 0; f < nfeatures; f++) {
        float fv = __ldg(&features_T[(size_t)f * npoints + p]);
        for (int c = 0; c < nclusters; c++) {
            float d = fv - d_clusters_const[c * MAX_FEATURES + f];
            dist[c] += d * d;
        }
    }

    float min_dist = dist[0];
    int index = 0;
    for (int c = 1; c < nclusters; c++) {
        if (dist[c] < min_dist) { min_dist = dist[c]; index = c; }
    }
    membership[p] = index;
}

/* Launch the right template instantiation based on runtime nclusters. */
static void launch_assign(const float *d_features_T, int *d_membership,
                           int npoints, int nfeatures, int nclusters,
                           int blocks, int threads) {
    switch (nclusters) {
        case 1:  assign_kernel_nc<1><<<blocks, threads>>>(d_features_T, d_membership, npoints, nfeatures); break;
        case 2:  assign_kernel_nc<2><<<blocks, threads>>>(d_features_T, d_membership, npoints, nfeatures); break;
        case 3:  assign_kernel_nc<3><<<blocks, threads>>>(d_features_T, d_membership, npoints, nfeatures); break;
        case 4:  assign_kernel_nc<4><<<blocks, threads>>>(d_features_T, d_membership, npoints, nfeatures); break;
        case 5:  assign_kernel_nc<5><<<blocks, threads>>>(d_features_T, d_membership, npoints, nfeatures); break;
        case 6:  assign_kernel_nc<6><<<blocks, threads>>>(d_features_T, d_membership, npoints, nfeatures); break;
        case 7:  assign_kernel_nc<7><<<blocks, threads>>>(d_features_T, d_membership, npoints, nfeatures); break;
        case 8:  assign_kernel_nc<8><<<blocks, threads>>>(d_features_T, d_membership, npoints, nfeatures); break;
        default: assign_kernel_generic<<<blocks, threads>>>(d_features_T, d_membership, npoints, nfeatures, nclusters); break;
    }
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

    if (nclusters > MAX_CLUSTERS) {
        fprintf(stderr, "Error: nclusters=%d > MAX_CLUSTERS=%d\n", nclusters, MAX_CLUSTERS);
        return 1;
    }
    if (nfeatures > MAX_FEATURES) {
        fprintf(stderr, "Error: nfeatures=%d > MAX_FEATURES=%d\n", nfeatures, MAX_FEATURES);
        return 1;
    }

    srand(7);
    float *features = (float *)malloc((size_t)npoints * nfeatures * sizeof(float));
    for (int i = 0; i < npoints * nfeatures; i++)
        features[i] = (float)rand() / RAND_MAX * 500.0f;

    float *clusters = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (int c = 0; c < nclusters; c++)
        for (int f = 0; f < nfeatures; f++)
            clusters[c * nfeatures + f] = features[c * nfeatures + f];

    int   *membership     = (int   *)malloc(npoints * sizeof(int));
    float *new_centers    = (float *)calloc(nclusters * nfeatures, sizeof(float));
    int   *new_center_len = (int   *)calloc(nclusters, sizeof(int));
    for (int i = 0; i < npoints; i++) membership[i] = -1;

    /* Build transposed feature layout [feature][point] for coalesced device reads. */
    float *features_T = (float *)malloc((size_t)npoints * nfeatures * sizeof(float));
    for (int i = 0; i < npoints; i++)
        for (int f = 0; f < nfeatures; f++)
            features_T[(size_t)f * npoints + i] = features[(size_t)i * nfeatures + f];

    float *d_features_T;
    int   *d_membership;
    cudaMalloc(&d_features_T, (size_t)npoints * nfeatures * sizeof(float));
    cudaMalloc(&d_membership, npoints * sizeof(int));
    cudaMemcpy(d_features_T, features_T, (size_t)npoints * nfeatures * sizeof(float),
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (npoints + threads - 1) / threads;

    /*
     * Staging buffer for constant memory upload.
     * Layout: clusters_padded[c * MAX_FEATURES + f] = clusters[c][f].
     * Padded to MAX_FEATURES per cluster so stride is compile-time constant.
     */
    float clusters_padded[MAX_CLUSTERS * MAX_FEATURES];

    double t0 = now();

    for (int iter = 0; iter < max_iter; iter++) {
        /* Build padded centroid array with compile-time stride MAX_FEATURES. */
        memset(clusters_padded, 0, sizeof(clusters_padded));
        for (int c = 0; c < nclusters; c++)
            for (int f = 0; f < nfeatures; f++)
                clusters_padded[c * MAX_FEATURES + f] = clusters[c * nfeatures + f];

        /* Upload to constant memory (synchronous; 8 KB max). */
        cudaMemcpyToSymbol(d_clusters_const, clusters_padded,
                           nclusters * MAX_FEATURES * sizeof(float));

        launch_assign(d_features_T, d_membership, npoints, nfeatures, nclusters,
                      blocks, threads);

        cudaMemcpy(membership, d_membership, npoints * sizeof(int),
                   cudaMemcpyDeviceToHost);

        /* Centroid update on CPU (serial order, exact numerics). */
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
