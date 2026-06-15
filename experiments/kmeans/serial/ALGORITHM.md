# K-Means Clustering — Algorithm Specification

## Problem

Given N data points each with F features, partition them into K clusters by
minimizing within-cluster sum of squared Euclidean distances. This is standard
Lloyd's algorithm (K-means), run for a fixed number of iterations.

## Standard Workload

| Parameter   | Value   |
|-------------|---------|
| npoints (N) | 65536   |
| nfeatures (F)| 34     |
| nclusters (K)| 5      |
| max_iter    | 20      |

## Input Generation

Data is generated synthetically (no file input). Seed with `srand(7)`, then:

```c
for (int i = 0; i < npoints * nfeatures; i++)
    features[i] = (float)rand() / RAND_MAX * 500.0f;
```

Layout: `features[i * nfeatures + f]` = feature f of point i (row-major).

## Centroid Initialization

Use the first `nclusters` points as initial centroids (deterministic):

```c
for (int c = 0; c < nclusters; c++)
    for (int f = 0; f < nfeatures; f++)
        clusters[c * nfeatures + f] = features[c * nfeatures + f];
```

## Main Loop (exactly max_iter iterations, no convergence check)

### Assign Step (GPU target)

For each point i, find the nearest centroid by squared Euclidean distance:

```c
for (int i = 0; i < npoints; i++) {
    float min_dist = FLT_MAX;
    int index = 0;
    for (int c = 0; c < nclusters; c++) {
        float dist = 0.0f;
        for (int f = 0; f < nfeatures; f++) {
            float d = features[i*nfeatures+f] - clusters[c*nfeatures+f];
            dist += d * d;
        }
        if (dist < min_dist) { min_dist = dist; index = c; }
    }
    membership[i] = index;
}
```

This step is **embarrassingly parallel**: each point's assignment is independent.

### Update Step (keep on CPU / host)

Recompute centroids from the new assignments. **This step must be done on the
CPU (host) to ensure float accumulation order matches the serial reference,
making the oracle an exact integer comparison.**

```c
// zero accumulators
for (int c = 0; c < nclusters * nfeatures; c++) new_centers[c] = 0.0f;
for (int c = 0; c < nclusters; c++) new_center_len[c] = 0;

// accumulate
for (int i = 0; i < npoints; i++) {
    int c = membership[i];
    new_center_len[c]++;
    for (int f = 0; f < nfeatures; f++)
        new_centers[c*nfeatures+f] += features[i*nfeatures+f];
}

// divide
for (int c = 0; c < nclusters; c++)
    if (new_center_len[c] > 0)
        for (int f = 0; f < nfeatures; f++)
            clusters[c*nfeatures+f] = new_centers[c*nfeatures+f] / new_center_len[c];
```

## Output Format

After all iterations, write to `<output_file>` one line per point:

```
<index>\t<cluster_id>\n
```

Example (first 3 lines):
```
0	0
1	1
2	2
```

Exactly `npoints` lines.

## Timing

Report `compute_seconds=<t>` to stderr (wall time of the assign+update loop only,
excluding data generation, init, and file I/O).

## CLI

```
./kmeans <npoints> <nfeatures> <nclusters> <max_iter> <output_file>
```

Example:
```
./kmeans 65536 34 5 20 out.txt
```

## Oracle Note

Because the update step is performed on the CPU in serial order, the centroid
values are identical between the serial reference and any GPU implementation
that follows this spec. The final `membership[]` array is therefore
**exactly reproducible** and the oracle uses exact integer comparison
(rtol = atol = 0).

## Performance Characteristics

- **Dominant computation**: assign step — O(N × K × F) per iteration
- **Memory pattern**: each point reads K × F floats from the centroid array
  (small: 5 × 34 × 4B = 680B, fits in L1/constant memory) plus its own F
  features (row-major, coalesced if threads process consecutive points)
- **Parallelism**: assign step is embarrassingly parallel over N points
- **Phase 2 opportunities**: constant memory for centroids, shared memory
  tiling for feature loads, vectorized (float4) loads
