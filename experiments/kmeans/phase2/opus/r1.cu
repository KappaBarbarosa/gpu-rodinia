/*
 * K-Means — Phase 2 round 1 optimized CUDA.
 * GPU: assign step (coalesced transposed features + constant-memory centroids).
 * CPU: centroid update (serial order, exact numerics).
 * CLI: <npoints> <nfeatures> <nclusters> <max_iter> <output_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_CLUSTERS 32
#define MAX_FEATURES 64

/* Centroids live in constant memory: broadcast to all warp threads in one cycle. */
__constant__ float d_clusters_const[MAX_CLUSTERS * MAX_FEATURES];

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * features_T is stored transposed [feature][point]: features_T[f*npoints + p].
 * Adjacent threads (consecutive p) read stride-1 addresses -> coalesced loads.
 * Loop order: feature-outer / cluster-inner. Each feature value is loaded once
 * and used to accumulate squared distance to every centroid.
 * dist[] kept in registers (no #pragma unroll on the cluster loop to avoid spill).
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
        float fv = features_T[f * npoints + p];   /* coalesced */
        const float *crow = &d_clusters_const[f];  /* feature f across centroids: stride nfeatures */
        for (int c = 0; c < nclusters; c++) {
            float d = fv - crow[c * nfeatures];     /* broadcast from constant cache */
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
    cudaMemcpy(d_features_T, features_T, (size_t)npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (npoints + threads - 1) / threads;

    double t0 = now();

    for (int iter = 0; iter < max_iter; iter++) {
        /* upload centroids to constant memory (small: <= 32*64 floats) */
        cudaMemcpyToSymbol(d_clusters_const, clusters,
                           nclusters * nfeatures * sizeof(float));

        assign_kernel<<<blocks, threads>>>(d_features_T, d_membership,
                                            npoints, nfeatures, nclusters);
        cudaMemcpy(membership, d_membership, npoints * sizeof(int), cudaMemcpyDeviceToHost);

        /* update step on CPU (serial order) */
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

    cudaFree(d_features_T); cudaFree(d_membership);
    free(features); free(features_T); free(clusters); free(membership);
    free(new_centers); free(new_center_len);
    return 0;
}
