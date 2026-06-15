/*
 * K-Means CUDA Phase 1 — global memory only.
 * Assign step on GPU, update step on CPU (for exact oracle match).
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

/* GPU kernel: each thread handles one point */
__global__ void assign_kernel(
    const float * __restrict__ features,
    const float * __restrict__ clusters,
    int           * __restrict__ membership,
    int npoints,
    int nfeatures,
    int nclusters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;

    float min_dist = FLT_MAX;
    int   index    = 0;

    for (int c = 0; c < nclusters; c++) {
        float dist = 0.0f;
        for (int f = 0; f < nfeatures; f++) {
            float d = features[i * nfeatures + f] - clusters[c * nfeatures + f];
            dist += d * d;
        }
        if (dist < min_dist) {
            min_dist = dist;
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

    /* generate input data */
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

    /* allocate device memory */
    float *d_features, *d_clusters;
    int   *d_membership;

    cudaMalloc((void **)&d_features,   npoints   * nfeatures * sizeof(float));
    cudaMalloc((void **)&d_clusters,   nclusters * nfeatures * sizeof(float));
    cudaMalloc((void **)&d_membership, npoints   * sizeof(int));

    /* copy features once (they never change) */
    cudaMemcpy(d_features, features, npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size  = (npoints + block_size - 1) / block_size;

    double t0 = now();

    for (int iter = 0; iter < max_iter; iter++) {
        /* --- assign step: GPU --- */
        cudaMemcpy(d_clusters, clusters, nclusters * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

        assign_kernel<<<grid_size, block_size>>>(
            d_features, d_clusters, d_membership,
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

    cudaFree(d_features);
    cudaFree(d_clusters);
    cudaFree(d_membership);
    free(features); free(clusters); free(membership);
    free(new_centers); free(new_center_len);
    return 0;
}
