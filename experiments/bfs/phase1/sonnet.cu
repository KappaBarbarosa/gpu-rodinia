#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct Node {
    int starting;
    int no_of_edges;
};

__global__ void kernel_expand(
    const struct Node *__restrict__ graph_nodes,
    const int         *__restrict__ graph_edges,
    char        *__restrict__ graph_mask,
    char        *__restrict__ updating_graph_mask,
    const char  *__restrict__ graph_visited,
    int         *__restrict__ cost,
    int          no_of_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= no_of_nodes) return;

    if (graph_mask[tid]) {
        graph_mask[tid] = 0;
        struct Node nd = graph_nodes[tid];
        int begin = nd.starting;
        int end   = begin + nd.no_of_edges;
        int my_cost = cost[tid] + 1;
        for (int i = begin; i < end; i++) {
            int nid = __ldg(&graph_edges[i]);
            if (!graph_visited[nid]) {
                cost[nid] = my_cost;
                updating_graph_mask[nid] = 1;
            }
        }
    }
}

__global__ void kernel_commit(
    char *__restrict__ graph_mask,
    char *__restrict__ updating_graph_mask,
    char *__restrict__ graph_visited,
    int  *__restrict__ d_stop,
    int   no_of_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= no_of_nodes) return;
    if (updating_graph_mask[tid]) {
        graph_mask[tid]          = 1;
        graph_visited[tid]       = 1;
        updating_graph_mask[tid] = 0;
        *d_stop = 1;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]); return 1; }
    const char *input_f = argv[1];
    const char *ofile   = argv[2];
    FILE *fp = fopen(input_f, "r");
    if (!fp) { fprintf(stderr, "Error reading graph file %s\n", input_f); return 1; }
    int no_of_nodes = 0;
    if (fscanf(fp, "%d", &no_of_nodes) != 1 || no_of_nodes <= 0) { fprintf(stderr, "bad node count\n"); return 1; }
    struct Node *h_graph_nodes   = (struct Node *)malloc(sizeof(struct Node) * no_of_nodes);
    char        *h_graph_mask    = (char *)malloc(sizeof(char) * no_of_nodes);
    char        *h_updating_mask = (char *)malloc(sizeof(char) * no_of_nodes);
    char        *h_graph_visited = (char *)malloc(sizeof(char) * no_of_nodes);
    int         *h_cost          = (int  *)malloc(sizeof(int)  * no_of_nodes);
    if (!h_graph_nodes || !h_graph_mask || !h_updating_mask || !h_graph_visited || !h_cost) { fprintf(stderr, "alloc failed\n"); return 1; }
    int start, edgeno;
    for (int i = 0; i < no_of_nodes; i++) {
        if (fscanf(fp, "%d %d", &start, &edgeno) != 2) { fprintf(stderr, "bad node line\n"); return 1; }
        h_graph_nodes[i].starting = start; h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i]=0; h_updating_mask[i]=0; h_graph_visited[i]=0; h_cost[i]=-1;
    }
    int source = 0; fscanf(fp, "%d", &source); source = 0;
    int edge_list_size = 0;
    if (fscanf(fp, "%d", &edge_list_size) != 1 || edge_list_size <= 0) { fprintf(stderr, "bad edge count\n"); return 1; }
    int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
    if (!h_graph_edges) { fprintf(stderr, "alloc failed\n"); return 1; }
    int id, c;
    for (int i = 0; i < edge_list_size; i++) {
        if (fscanf(fp, "%d %d", &id, &c) != 2) { fprintf(stderr, "bad edge line\n"); return 1; }
        h_graph_edges[i] = id;
    }
    fclose(fp);
    h_graph_mask[source]=1; h_graph_visited[source]=1; h_cost[source]=0;
    struct Node *d_graph_nodes; int *d_graph_edges; char *d_graph_mask; char *d_updating_mask; char *d_graph_visited; int *d_cost; int *d_stop;
    cudaMalloc((void **)&d_graph_nodes, sizeof(struct Node)*no_of_nodes);
    cudaMalloc((void **)&d_graph_edges, sizeof(int)*edge_list_size);
    cudaMalloc((void **)&d_graph_mask, sizeof(char)*no_of_nodes);
    cudaMalloc((void **)&d_updating_mask, sizeof(char)*no_of_nodes);
    cudaMalloc((void **)&d_graph_visited, sizeof(char)*no_of_nodes);
    cudaMalloc((void **)&d_cost, sizeof(int)*no_of_nodes);
    cudaMalloc((void **)&d_stop, sizeof(int));
    cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(struct Node)*no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_updating_mask, h_updating_mask, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);
    int block_size = 256;
    int grid_size = (no_of_nodes + block_size - 1) / block_size;
    double t0 = now_seconds();
    int h_stop;
    do {
        h_stop = 0;
        cudaMemcpy(d_stop, &h_stop, sizeof(int), cudaMemcpyHostToDevice);
        kernel_expand<<<grid_size, block_size>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_mask, d_graph_visited, d_cost, no_of_nodes);
        kernel_commit<<<grid_size, block_size>>>(d_graph_mask, d_updating_mask, d_graph_visited, d_stop, no_of_nodes);
        cudaMemcpy(&h_stop, d_stop, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_stop);
    cudaDeviceSynchronize();
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);
    cudaMemcpy(h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);
    FILE *out = fopen(ofile, "w");
    if (!out) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int i = 0; i < no_of_nodes; i++) fprintf(out, "%d\t%d\n", i, h_cost[i]);
    fclose(out);
    cudaFree(d_graph_nodes); cudaFree(d_graph_edges); cudaFree(d_graph_mask); cudaFree(d_updating_mask); cudaFree(d_graph_visited); cudaFree(d_cost); cudaFree(d_stop);
    free(h_graph_nodes); free(h_graph_mask); free(h_updating_mask); free(h_graph_visited); free(h_cost); free(h_graph_edges);
    return 0;
}
