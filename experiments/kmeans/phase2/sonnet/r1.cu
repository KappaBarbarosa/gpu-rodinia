/*
 * K-Means CUDA Phase 2, Round 1
 * Optimizations applied (from feedback):
 *   1. Transposed feature layout [nfeatures][npoints] for coalesced access
 *   2. Constant memory for centroids (broadcast, zero L2 traffic)
 *   3. Inverted loop order: outer=feature, inner=cluster (load each feature once)
 *   4. Static inner cluster loop with #pragma unroll (NC_STATIC=8)
 *   5. Partial unroll x4 on outer feature loop
 * Centroid update stays on CPU (exact oracle match).
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>

/* Maximum supported clusters and features for constant memory */
#define NC_STATIC   8
#define MAX_CLUSTERS 16
#define MAX_FEATURES 64

/* Centroids in constant memory: [nclusters][nfeatures], max 16*64*4 = 4096 bytes */
__constant__ float c_clusters[MAX_CLUSTERS * MAX_FEATURES];

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * GPU kernel: transposed layout features_T[f * npoints + i]
 * Outer loop over features (load once), inner loop over clusters (in registers).
 * NC_STATIC must be >= nclusters; a runtime guard handles nc < NC_STATIC.
 */
__global__ void assign_kernel(
    const float * __restrict__ features_T,
    int           * __restrict__ membership,
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

    /* outer: features — each loaded once, coalesced */
    int f = 0;
    /* partial unroll x4 for load-latency hiding */
    for (; f <= nfeatures - 4; f += 4) {
        float v0 = features_T[(f+0) * npoints + i];
        float v1 = features_T[(f+1) * npoints + i];
        float v2 = features_T[(f+2) * npoints + i];
        float v3 = features_T[(f+3) * npoints + i];
        #pragma unroll
        for (int c = 0; c < NC_STATIC; c++) {
            float d0 = v0 - c_clusters[c * MAX_FEATURES + (f+0)];
            float d1 = v1 - c_clusters[c * MAX_FEATURES + (f+1)];
            float d2 = v2 - c_clusters[c * MAX_FEATURES + (f+2)];
            float d3 = v3 - c_clusters[c * MAX_FEATURES + (f+3)];
            dist[c] += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        }
    }
    /* remainder */
    for (; f < nfeatures; f++) {
        float v = features_T[f * npoints + i];
        #pragma unroll
        for (int c = 0; c < NC_STATIC; c++) {
            float d = v - c_clusters[c * MAX_FEATURES + f];
            dist[c] += d * d;
        }
    }

    /* argmin with runtime guard for nc < NC_STATIC */
    float min_dist = FLT_MAX;
    int   index    = 0;
    #pragma unroll
    for (int c = 0; c < NC_STATIC; c++) {
        if (c < nclusters && dist[c] < min_dist) {
            min_dist = dist[c];
            index    = c;
        }
    }
    membership[i] = index;
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

    /* generate input data (row-major) */
    srand(7);
    float *features = (float *)malloc(npoints * nfeatures * sizeof(float));
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

    /* transpose features: features_T[f][i] = features[i][f] */
    float *features_T = (float *)malloc(nfeatures * npoints * sizeof(float));
    for (int i = 0; i < npoints; i++)
        for (int f = 0; f < nfeatures; f++)
            features_T[f * npoints + i] = features[i * nfeatures + f];

    /* allocate device memory */
    float *d_features_T;
    int   *d_membership;

    cudaMalloc((void **)&d_features_T, (size_t)nfeatures * npoints * sizeof(float));
    cudaMalloc((void **)&d_membership, npoints * sizeof(int));

    /* copy transposed features once */
    cudaMemcpy(d_features_T, features_T, (size_t)nfeatures * npoints * sizeof(float), cudaMemcpyHostToDevice);

    /* staging buffer for constant memory upload (padded to MAX_FEATURES stride) */
    float h_cmem[MAX_CLUSTERS * MAX_FEATURES];

    int block_size = 256;
    int grid_size  = (npoints + block_size - 1) / block_size;

    double t0 = now();

    for (int iter = 0; iter < max_iter; iter++) {
        /* --- upload centroids to constant memory --- */
        /* pack into padded layout [c][MAX_FEATURES] */
        for (int c = 0; c < nclusters; c++)
            for (int f = 0; f < nfeatures; f++)
                h_cmem[c * MAX_FEATURES + f] = clusters[c * nfeatures + f];

        cudaMemcpyToSymbol(c_clusters, h_cmem, nclusters * MAX_FEATURES * sizeof(float));

        /* --- assign step: GPU --- */
        assign_kernel<<<grid_size, block_size>>>(
            d_features_T, d_membership,
            npoints, nfeatures, nclusters);

        cudaMemcpy(membership, d_membership, npoints * sizeof(int), cudaMemcpyDeviceToHost);

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

    cudaFree(d_features_T);
    cudaFree(d_membership);
    free(features_T); free(features); free(clusters);
    free(membership); free(new_centers); free(new_center_len);
    return 0;
}
