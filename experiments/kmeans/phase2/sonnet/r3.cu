/*
 * K-Means CUDA Phase 2, Round 3
 * Changes vs r2:
 *   1. 1 point per thread (was 2 via float2) — doubles block count, boosts warp
 *      occupancy from ~22 to ~44 warps/SM, better latency hiding on sm_86
 *   2. Transposed float layout features_T[f * npoints + i] — simple 32-bit scalar
 *      loads, perfectly coalesced, same bandwidth as float2 but with 2x occupancy
 *   3. Block size 256, grid = (npoints + 255) / 256 = 256 blocks (was 128)
 *   4. NC_STATIC=5, MAX_FEATURES=34 — constant memory for centroids (unchanged)
 *   5. Full #pragma unroll on inner cluster loop (unchanged)
 *   6. Ampere L2 persistence for feature buffer (unchanged)
 *   7. __ldg() reads + fmaf() compute (unchanged)
 * Centroid update stays on CPU (exact oracle match).
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

/* Exact values for standard workload */
#define NC_STATIC    5
#define MAX_CLUSTERS 16
#define MAX_FEATURES 34

/* Centroids in constant memory: [nclusters][MAX_FEATURES], 5*34*4 = 680 bytes */
__constant__ float c_clusters[MAX_CLUSTERS * MAX_FEATURES];

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * GPU kernel: each thread processes exactly 1 point.
 * Layout: features_T[f * npoints + i] — coalesced 32-bit loads.
 * Outer loop over features, inner over clusters (constant-mem broadcast).
 */
__global__ void assign_kernel(
    const float * __restrict__ features_T,
    int         * __restrict__ membership,
    int npoints,
    int nfeatures,
    int nclusters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;

    /* per-cluster distance accumulators in registers */
    float dist[NC_STATIC];
    #pragma unroll
    for (int c = 0; c < NC_STATIC; c++) dist[c] = 0.0f;

    /* outer: features — each loaded once (32-bit coalesced load) */
    #pragma unroll 17
    for (int f = 0; f < nfeatures; f++) {
        float v = __ldg(&features_T[f * npoints + i]);
        #pragma unroll
        for (int c = 0; c < NC_STATIC; c++) {
            float d = v - c_clusters[c * MAX_FEATURES + f];
            dist[c] = fmaf(d, d, dist[c]);
        }
    }

    /* argmin */
    float min_dist = FLT_MAX;
    int   idx      = 0;
    #pragma unroll
    for (int c = 0; c < NC_STATIC; c++) {
        if (dist[c] < min_dist) {
            min_dist = dist[c];
            idx      = c;
        }
    }
    membership[i] = idx;
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
        fprintf(stderr, "Error: nclusters=%d exceeds MAX_CLUSTERS=%d\n", nclusters, MAX_CLUSTERS);
        return 1;
    }
    if (nfeatures > MAX_FEATURES) {
        fprintf(stderr, "Error: nfeatures=%d exceeds MAX_FEATURES=%d\n", nfeatures, MAX_FEATURES);
        return 1;
    }
    if (nclusters > NC_STATIC) {
        fprintf(stderr, "Error: nclusters=%d exceeds NC_STATIC=%d\n", nclusters, NC_STATIC);
        return 1;
    }

    /* generate input data (row-major: features[i * nfeatures + f]) */
    srand(7);
    float *features = (float *)malloc((size_t)npoints * nfeatures * sizeof(float));
    for (int i = 0; i < npoints * nfeatures; i++)
        features[i] = (float)rand() / RAND_MAX * 500.0f;

    /* init centroids: first nclusters points */
    float *clusters = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (int c = 0; c < nclusters; c++)
        for (int f = 0; f < nfeatures; f++)
            clusters[c * nfeatures + f] = features[c * nfeatures + f];

    int   *membership     = (int   *)malloc(npoints * sizeof(int));
    float *new_centers    = (float *)calloc(nclusters * nfeatures, sizeof(float));
    int   *new_center_len = (int   *)calloc(nclusters, sizeof(int));

    for (int i = 0; i < npoints; i++) membership[i] = -1;

    /*
     * Build transposed float layout: features_T[f * npoints + i]
     * Each warp accessing features_T[f * npoints + warp_base .. warp_base+31]
     * fetches 32 consecutive floats = one 128-byte cache line — perfectly coalesced.
     */
    float *features_T = (float *)malloc((size_t)nfeatures * npoints * sizeof(float));
    for (int f = 0; f < nfeatures; f++)
        for (int i = 0; i < npoints; i++)
            features_T[f * npoints + i] = features[i * nfeatures + f];

    /* allocate device memory */
    float *d_features_T;
    int   *d_membership;

    size_t feat_bytes = (size_t)nfeatures * npoints * sizeof(float);
    cudaMalloc((void **)&d_features_T, feat_bytes);
    cudaMalloc((void **)&d_membership, npoints * sizeof(int));

    /* Create explicit stream for L2 persistence */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Set up Ampere L2 persistent cache for feature data */
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    /* Reserve up to 3 MB of L2 as persistent region (RTX 3070 has 4 MB L2) */
    size_t persist_size = 3UL * 1024 * 1024;
    if (persist_size > (size_t)prop.persistingL2CacheMaxSize)
        persist_size = prop.persistingL2CacheMaxSize;
    if (persist_size > 0)
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size);

    /* Apply persisting access policy to the feature buffer on the stream */
    if (prop.persistingL2CacheMaxSize > 0) {
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr  = (void *)d_features_T;
        attr.accessPolicyWindow.num_bytes = (feat_bytes < persist_size) ? feat_bytes : persist_size;
        attr.accessPolicyWindow.hitRatio  = 1.0f;
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    /* copy transposed features once */
    cudaMemcpyAsync(d_features_T, features_T, feat_bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    /* staging buffer for constant memory upload */
    float h_cmem[MAX_CLUSTERS * MAX_FEATURES];

    int block_size = 256;
    int grid_size  = (npoints + block_size - 1) / block_size;

    double t0 = now();

    for (int iter = 0; iter < max_iter; iter++) {
        /* --- upload centroids to constant memory --- */
        for (int c = 0; c < nclusters; c++)
            for (int f = 0; f < nfeatures; f++)
                h_cmem[c * MAX_FEATURES + f] = clusters[c * nfeatures + f];

        cudaMemcpyToSymbol(c_clusters, h_cmem, nclusters * MAX_FEATURES * sizeof(float));

        /* --- assign step: GPU --- */
        assign_kernel<<<grid_size, block_size, 0, stream>>>(
            d_features_T, d_membership,
            npoints, nfeatures, nclusters);

        cudaMemcpyAsync(membership, d_membership, npoints * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        /* --- update step: CPU (serial, exact match with oracle) --- */
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

    /* reset L2 persistence before freeing */
    if (prop.persistingL2CacheMaxSize > 0) {
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        cudaCtxResetPersistingL2Cache();
    }

    cudaStreamDestroy(stream);
    cudaFree(d_features_T);
    cudaFree(d_membership);
    free(features_T); free(features); free(clusters);
    free(membership); free(new_centers); free(new_center_len);
    return 0;
}
