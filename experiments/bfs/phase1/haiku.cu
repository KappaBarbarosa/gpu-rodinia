#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define BLOCK_SIZE 128

__global__ void __launch_bounds__(BLOCK_SIZE) bfs_expand_kernel(
    int no_of_nodes,
    const int * __restrict__ graph_nodes_start,
    const int * __restrict__ graph_nodes_degree,
    const char * __restrict__ graph_mask,
    char * __restrict__ updating_graph_mask,
    int * __restrict__ cost,
    const int * __restrict__ graph_edges,
    const char * __restrict__ graph_visited)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= no_of_nodes) return;

    if (graph_mask[tid]) {
        int begin = graph_nodes_start[tid];
        int end = begin + graph_nodes_degree[tid];
        int new_cost = cost[tid] + 1;   /* hoisted out of the loop */
        for (int i = begin; i < end; i++) {
            int nid = graph_edges[i];
            if (!graph_visited[nid]) {
                cost[nid] = new_cost;
                updating_graph_mask[nid] = 1;
            }
        }
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE) bfs_commit_kernel(
    int no_of_nodes,
    char * __restrict__ graph_mask,
    char * __restrict__ updating_graph_mask,
    char * __restrict__ graph_visited,
    int * __restrict__ stop_flag)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= no_of_nodes) return;

    /* Clear the previous frontier so only newly-discovered nodes expand next
       level. Without this, every already-visited frontier node would re-scan
       its edges on every subsequent iteration (the dominant cost in the naive
       port). */
    graph_mask[tid] = 0;
    if (updating_graph_mask[tid]) {
        graph_mask[tid] = 1;
        graph_visited[tid] = 1;
        updating_graph_mask[tid] = 0;
        *stop_flag = 1;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    const char *input_f = argv[1];
    const char *ofile = argv[2];

    FILE *fp = fopen(input_f, "r");
    if (!fp) {
        fprintf(stderr, "Error reading graph file %s\n", input_f);
        return 1;
    }

    int no_of_nodes = 0;
    if (fscanf(fp, "%d", &no_of_nodes) != 1 || no_of_nodes <= 0) {
        fprintf(stderr, "bad node count\n");
        return 1;
    }

    int *h_graph_nodes_start = (int *)malloc(sizeof(int) * no_of_nodes);
    int *h_graph_nodes_degree = (int *)malloc(sizeof(int) * no_of_nodes);
    char *h_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
    char *h_updating_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
    char *h_graph_visited = (char *)malloc(sizeof(char) * no_of_nodes);
    int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);

    if (!h_graph_nodes_start || !h_graph_nodes_degree || !h_graph_mask ||
        !h_updating_graph_mask || !h_graph_visited || !h_cost) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    int start, edgeno;
    for (int i = 0; i < no_of_nodes; i++) {
        if (fscanf(fp, "%d %d", &start, &edgeno) != 2) {
            fprintf(stderr, "bad node line\n");
            return 1;
        }
        h_graph_nodes_start[i] = start;
        h_graph_nodes_degree[i] = edgeno;
        h_graph_mask[i] = 0;
        h_updating_graph_mask[i] = 0;
        h_graph_visited[i] = 0;
        h_cost[i] = -1;
    }

    int source = 0;
    if (fscanf(fp, "%d", &source) != 1) { /* source forced to 0 below */ }
    source = 0;

    int edge_list_size = 0;
    if (fscanf(fp, "%d", &edge_list_size) != 1 || edge_list_size <= 0) {
        fprintf(stderr, "bad edge count\n");
        return 1;
    }

    int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
    if (!h_graph_edges) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    int id, c;
    for (int i = 0; i < edge_list_size; i++) {
        if (fscanf(fp, "%d %d", &id, &c) != 2) {
            fprintf(stderr, "bad edge line\n");
            return 1;
        }
        h_graph_edges[i] = id;
    }
    fclose(fp);

    h_graph_mask[source] = 1;
    h_graph_visited[source] = 1;
    h_cost[source] = 0;

    /* Allocate device memory */
    int *d_graph_nodes_start;
    int *d_graph_nodes_degree;
    char *d_graph_mask;
    char *d_updating_graph_mask;
    char *d_graph_visited;
    int *d_cost;
    int *d_graph_edges;
    int *d_stop_flag;

    cudaMalloc(&d_graph_nodes_start, sizeof(int) * no_of_nodes);
    cudaMalloc(&d_graph_nodes_degree, sizeof(int) * no_of_nodes);
    cudaMalloc(&d_graph_mask, sizeof(char) * no_of_nodes);
    cudaMalloc(&d_updating_graph_mask, sizeof(char) * no_of_nodes);
    cudaMalloc(&d_graph_visited, sizeof(char) * no_of_nodes);
    cudaMalloc(&d_cost, sizeof(int) * no_of_nodes);
    cudaMalloc(&d_graph_edges, sizeof(int) * edge_list_size);
    cudaMalloc(&d_stop_flag, sizeof(int));

    /* Copy data to device */
    cudaMemcpy(d_graph_nodes_start, h_graph_nodes_start, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_nodes_degree, h_graph_nodes_degree, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(char) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(char) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(char) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);

    /* BFS on device */
    int grid_size = (no_of_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double t0 = now_seconds();

    int stop = 1;
    while (stop) {
        int h_stop = 0;
        cudaMemcpy(d_stop_flag, &h_stop, sizeof(int), cudaMemcpyHostToDevice);

        /* Phase 1: expand current frontier */
        bfs_expand_kernel<<<grid_size, BLOCK_SIZE>>>(
            no_of_nodes,
            d_graph_nodes_start,
            d_graph_nodes_degree,
            d_graph_mask,
            d_updating_graph_mask,
            d_cost,
            d_graph_edges,
            d_graph_visited);

        /* Phase 2: commit (clear old frontier, promote new one) */
        bfs_commit_kernel<<<grid_size, BLOCK_SIZE>>>(
            no_of_nodes,
            d_graph_mask,
            d_updating_graph_mask,
            d_graph_visited,
            d_stop_flag);

        /* Did any node get newly discovered this level? */
        cudaMemcpy(&h_stop, d_stop_flag, sizeof(int), cudaMemcpyDeviceToHost);
        stop = h_stop;
    }

    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    /* Copy result back */
    cudaMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);

    /* Write output */
    FILE *out = fopen(ofile, "w");
    if (!out) {
        fprintf(stderr, "cannot open %s\n", ofile);
        return 1;
    }
    for (int i = 0; i < no_of_nodes; i++) {
        fprintf(out, "%d\t%d\n", i, h_cost[i]);
    }
    fclose(out);

    /* Cleanup */
    cudaFree(d_graph_nodes_start);
    cudaFree(d_graph_nodes_degree);
    cudaFree(d_graph_mask);
    cudaFree(d_updating_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);
    cudaFree(d_graph_edges);
    cudaFree(d_stop_flag);

    free(h_graph_nodes_start);
    free(h_graph_nodes_degree);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
    free(h_graph_edges);

    return 0;
}
