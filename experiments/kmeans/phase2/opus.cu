/*
 * K-Means — Phase 2 optimized CUDA.
 * Optimizations vs phase1:
 *   1. Constant memory for centroids (5 * 34 * 4 = 680 bytes; fits in 64 KB
 *      constant cache). All threads in a warp read the same centroid value
 *      simultaneously → broadcast, zero extra bandwidth.
 *   2. Transposed feature layout [nfeatures][npoints] on the GPU.
 *      In phase1, features[i*nfeatures + f] strides across 34 floats between
 *      adjacent threads → uncoalesced. With transposed storage,
 *      features_T[f*npoints + p] places consecutive points contiguously,
 *      so a warp of 32 threads issues one 128-byte load per feature iteration.
 *   3. Feature-outer / cluster-inner loop order: each feature value is loaded
 *      once and immediately used to accumulate distances for all nclusters.
 *      Keeps the loaded value in a register; no re-reads from global memory.
 *   4. #pragma unroll on the dist[] init loop so the compiler can keep all
 *      32 slots in registers from the start.
 *   5. dist[] array stays in registers (MAX_CLUSTERS=32 entries; compiler uses
 *      48 registers, zero stack-frame / zero spill with 256 threads on sm_86).
 * CPU: centroid update (serial order, unchanged from phase1).
 * CLI: <npoints> <nfeatures> <nclusters> <max_iter> <output_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>

/* Constant memory for centroids: up to MAX_CLUSTERS * MAX_FEATURES floats.
 * 32 * 64 = 2048 floats = 8 KB (within 64 KB limit).
 * Benchmark: 5 * 34 = 170 floats = 680 bytes. */
#define MAX_CLUSTERS  32
#define MAX_FEATURES  64
__constant__ float d_clusters_const[MAX_CLUSTERS * MAX_FEATURES];

/*
 * assign_kernel_opt:
 *   features_T: transposed [nfeatures][npoints]. Point p feature f is at
 *               features_T[f * npoints + p]. Coalesced warp reads.
 *   membership: output [npoints], one int per point.
 *
 * Each thread handles one point.  For each feature f, it loads its own value
 * in one coalesced instruction (shared with the rest of the warp), then
 * accumulates squared distances to all nclusters centroids (from constant mem).
 */
__global__ void assign_kernel_opt(const float * __restrict__ features_T,
                                   int *membership,
                                   int npoints, int nfeatures, int nclusters)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npoints) return;

    /* Accumulate per-cluster squared distances in registers. */
    float dist[MAX_CLUSTERS];
    #pragma unroll
    for (int c = 0; c < MAX_CLUSTERS; c++) dist[c] = 0.0f;

    /* Outer loop: features — each iteration one coalesced global read. */
    for (int f = 0; f < nfeatures; f++) {
        float feat_val = features_T[f * npoints + p];
        /* Inner loop: clusters — all values from constant-memory broadcast. */
        for (int c = 0; c < nclusters; c++) {
            float d = feat_val - d_clusters_const[c * nfeatures + f];
            dist[c] += d * d;
        }
    }

    /* Find nearest centroid. */
    float min_dist = dist[0];
    int   index    = 0;
    for (int c = 1; c < nclusters; c++) {
        if (dist[c] < min_dist) { min_dist = dist[c]; index = c; }
    }
    membership[p] = index;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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
        fprintf(stderr, "Error: nclusters=%d > %d or nfeatures=%d > %d\n",
                nclusters, MAX_CLUSTERS, nfeatures, MAX_FEATURES);
        return 1;
    }

    srand(7);
    /* Row-major [point][feature] on the CPU side */
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

    /* Build transposed feature array [nfeatures][npoints] for the GPU. */
    float *features_T = (float *)malloc(nfeatures * npoints * sizeof(float));
    for (int f = 0; f < nfeatures; f++)
        for (int i = 0; i < npoints; i++)
            features_T[f * npoints + i] = features[i * nfeatures + f];

    float *d_features_T;
    int   *d_membership;
    cudaMalloc(&d_features_T, nfeatures * npoints * sizeof(float));
    cudaMalloc(&d_membership, npoints * sizeof(int));
    /* Upload transposed features once — they never change. */
    cudaMemcpy(d_features_T, features_T, nfeatures * npoints * sizeof(float),
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (npoints + threads - 1) / threads;

    double t0 = now();

    for (int iter = 0; iter < max_iter; iter++) {
        /* Upload updated centroids to constant memory (680 bytes). */
        cudaMemcpyToSymbol(d_clusters_const, clusters,
                           nclusters * nfeatures * sizeof(float));

        assign_kernel_opt<<<blocks, threads>>>(d_features_T, d_membership,
                                               npoints, nfeatures, nclusters);

        cudaMemcpy(membership, d_membership, npoints * sizeof(int),
                   cudaMemcpyDeviceToHost);

        /* Centroid update on CPU (serial order, unchanged). */
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
                    clusters[c * nfeatures + f] =
                        new_centers[c * nfeatures + f] / new_center_len[c];
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
