/*
 * K-Means — Phase 2 round 2 optimized CUDA.
 * GPU: assign step.
 * CPU: centroid update (serial order, exact numerics).
 * CLI: <npoints> <nfeatures> <nclusters> <max_iter> <output_file>
 *
 * Optimizations over r1:
 *  1. Constant-memory layout changed to [feature][cluster] (stride MAX_CLUSTERS)
 *     so the inner cluster loop is stride-1 in constant memory (was stride-nfeatures).
 *     This gives sequential cache-line reuse within the 8KB constant cache.
 *  2. #pragma unroll 8 on cluster and feature loops for ILP.
 *  3. Pinned (page-locked) host memory for membership DtoH transfer.
 *  4. Async kernel + memcpy via CUDA stream.
 *  5. Use __ldg() for explicit L1/texture cache path on features_T reads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_CLUSTERS 32
#define MAX_FEATURES 64

/*
 * Constant memory layout: d_clusters_const[f * MAX_CLUSTERS + c]
 * For fixed f, iterating c=0..nclusters-1 is stride-1 (sequential).
 * MAX_CLUSTERS=32 as stride keeps layout compile-time fixed and
 * ensures 128-byte alignment between feature rows.
 */
__constant__ float d_clusters_const[MAX_FEATURES * MAX_CLUSTERS];

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * assign_kernel_v2:
 *   features_T[f*npoints + p] — transposed [feature][point], coalesced.
 *   Constant memory accessed as d_clusters_const[f*MAX_CLUSTERS + c]: stride-1
 *   over c, enabling sequential broadcast to all warp threads per cycle.
 *   dist[MAX_CLUSTERS] lives in registers; with 256 threads * 32 floats
 *   = 32 KB per block — fits in the 64-KB register file on Ampere/Volta.
 */
__global__ void assign_kernel(const float * __restrict__ features_T,
                               int * __restrict__ membership,
                               int npoints, int nfeatures, int nclusters) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npoints) return;

    float dist[MAX_CLUSTERS];
    #pragma unroll
    for (int c = 0; c < MAX_CLUSTERS; c++) dist[c] = 0.0f;

    for (int f = 0; f < nfeatures; f++) {
        /* __ldg: route through read-only/texture cache (L1 cached on sm_35+). */
        float fv = __ldg(&features_T[(size_t)f * npoints + p]);

        /* Stride-1 walk through constant memory for feature f. */
        const float *crow = &d_clusters_const[f * MAX_CLUSTERS];
        #pragma unroll 8
        for (int c = 0; c < nclusters; c++) {
            float d = fv - crow[c];
            dist[c] += d * d;
        }
    }

    float min_dist = dist[0];
    int index = 0;
    #pragma unroll 8
    for (int c = 1; c < nclusters; c++) {
        if (dist[c] < min_dist) { min_dist = dist[c]; index = c; }
    }
    membership[p] = index;
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

    srand(7);
    float *features = (float *)malloc(npoints * nfeatures * sizeof(float));
    for (int i = 0; i < npoints * nfeatures; i++)
        features[i] = (float)rand() / RAND_MAX * 500.0f;

    float *clusters = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (int c = 0; c < nclusters; c++)
        for (int f = 0; f < nfeatures; f++)
            clusters[c * nfeatures + f] = features[c * nfeatures + f];

    /* Pinned membership buffer for faster DtoH transfer. */
    int *membership;
    cudaHostAlloc(&membership, npoints * sizeof(int), cudaHostAllocDefault);

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
    cudaMemcpy(d_features_T, features_T, (size_t)npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (npoints + threads - 1) / threads;

    /*
     * Temporary buffer: transpose clusters from [cluster][feature] to
     * [feature][cluster] layout expected by constant memory.
     * Stack allocation is safe: MAX_FEATURES * MAX_CLUSTERS * 4 = 8 KB.
     */
    float clusters_fc[MAX_FEATURES * MAX_CLUSTERS];

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double t0 = now();

    for (int iter = 0; iter < max_iter; iter++) {
        /* Transpose clusters to [feature][cluster] layout. */
        for (int f = 0; f < nfeatures; f++)
            for (int c = 0; c < nclusters; c++)
                clusters_fc[f * MAX_CLUSTERS + c] = clusters[c * nfeatures + f];

        /* Upload to constant memory (async). */
        cudaMemcpyToSymbolAsync(d_clusters_const, clusters_fc,
                                nfeatures * MAX_CLUSTERS * sizeof(float),
                                0, cudaMemcpyHostToDevice, stream);

        assign_kernel<<<blocks, threads, 0, stream>>>(
            d_features_T, d_membership, npoints, nfeatures, nclusters);

        cudaMemcpyAsync(membership, d_membership, npoints * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

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

    cudaStreamDestroy(stream);
    cudaFree(d_features_T); cudaFree(d_membership);
    free(features); free(features_T); free(clusters);
    cudaFreeHost(membership);
    free(new_centers); free(new_center_len);
    return 0;
}
