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

// Phase 1: expand frontier - one thread per frontier node (compact queue)
// Each frontier node processes all its out-edges
__global__ void __launch_bounds__(128) kernel_expand(
    const struct Node * __restrict__ graph_nodes,
    const int * __restrict__ graph_edges,
    const int * __restrict__ frontier,          // compact list of frontier node IDs
    int   frontier_size,
    char *updating_graph_mask,
    const char * __restrict__ graph_visited,
    int  *cost)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    int tid = frontier[idx];
    struct Node n = graph_nodes[tid];
    int begin = n.starting;
    int end   = begin + n.no_of_edges;
    int c = cost[tid] + 1;
    for (int i = begin; i < end; i++) {
        int nid = __ldg(&graph_edges[i]);
        if (!__ldg(&graph_visited[nid]) && updating_graph_mask[nid] == 0) {
            cost[nid] = c;
            updating_graph_mask[nid] = 1;
        }
    }
}

// Phase 2: commit newly discovered nodes AND build next frontier queue
// One thread per node in updating_mask (scan all N nodes)
// Writes newly committed nodes into next_frontier via atomicAdd
__global__ void __launch_bounds__(128) kernel_commit(
    char *updating_graph_mask,
    char *graph_visited,
    int  *next_frontier,
    int  *next_frontier_size,
    int   no_of_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < no_of_nodes && updating_graph_mask[tid]) {
        graph_visited[tid]       = 1;
        updating_graph_mask[tid] = 0;
        int slot = atomicAdd(next_frontier_size, 1);
        next_frontier[slot] = tid;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    const char *input_f = argv[1];
    const char *ofile   = argv[2];

    FILE *fp = fopen(input_f, "r");
    if (!fp) { fprintf(stderr, "Error reading graph file %s\n", input_f); return 1; }

    int no_of_nodes = 0;
    if (fscanf(fp, "%d", &no_of_nodes) != 1 || no_of_nodes <= 0) {
        fprintf(stderr, "bad node count\n"); return 1;
    }

    struct Node *h_graph_nodes = (struct Node *)malloc(sizeof(struct Node) * no_of_nodes);
    char *h_graph_visited       = (char *)malloc(sizeof(char) * no_of_nodes);
    char *h_updating_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
    int  *h_cost                = (int  *)malloc(sizeof(int)  * no_of_nodes);

    int start, edgeno;
    for (int i = 0; i < no_of_nodes; i++) {
        fscanf(fp, "%d %d", &start, &edgeno);
        h_graph_nodes[i].starting    = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_updating_graph_mask[i] = 0;
        h_graph_visited[i]       = 0;
        h_cost[i]                = -1;
    }

    int source = 0;
    fscanf(fp, "%d", &source);
    source = 0;

    int edge_list_size = 0;
    fscanf(fp, "%d", &edge_list_size);
    int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
    int id, c;
    for (int i = 0; i < edge_list_size; i++) {
        fscanf(fp, "%d %d", &id, &c);
        h_graph_edges[i] = id;
    }
    fclose(fp);

    h_graph_visited[source] = 1;
    h_cost[source]          = 0;

    // GPU allocations
    struct Node *d_graph_nodes;
    int         *d_graph_edges;
    char        *d_updating_graph_mask, *d_graph_visited;
    int         *d_cost;
    // Two frontier queues (ping-pong): current and next
    int         *d_frontier_a, *d_frontier_b;
    int         *d_frontier_size;  // size of next frontier (atomicAdd target)

    cudaMalloc(&d_graph_nodes,          sizeof(struct Node) * no_of_nodes);
    cudaMalloc(&d_graph_edges,          sizeof(int) * edge_list_size);
    cudaMalloc(&d_updating_graph_mask,  sizeof(char) * no_of_nodes);
    cudaMalloc(&d_graph_visited,        sizeof(char) * no_of_nodes);
    cudaMalloc(&d_cost,                 sizeof(int) * no_of_nodes);
    // Frontier queues: worst case all nodes in frontier
    cudaMalloc(&d_frontier_a,   sizeof(int) * no_of_nodes);
    cudaMalloc(&d_frontier_b,   sizeof(int) * no_of_nodes);
    cudaMalloc(&d_frontier_size, sizeof(int));

    cudaMemcpy(d_graph_nodes,         h_graph_nodes,         sizeof(struct Node) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges,         h_graph_edges,         sizeof(int) * edge_list_size,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(char) * no_of_nodes,        cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_visited,       h_graph_visited,       sizeof(char) * no_of_nodes,        cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost,                h_cost,                sizeof(int) * no_of_nodes,         cudaMemcpyHostToDevice);

    // Seed frontier with source node
    int h_frontier_size = 1;
    int h_source = source;
    cudaMemcpy(d_frontier_a, &h_source, sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 128;
    int num_blocks_all = (no_of_nodes + block_size - 1) / block_size;

    double t0 = now_seconds();

    // current frontier is d_frontier_a with size h_frontier_size
    // next frontier goes into d_frontier_b
    int *d_cur_frontier  = d_frontier_a;
    int *d_next_frontier = d_frontier_b;

    while (h_frontier_size > 0) {
        // Reset next frontier size
        int zero = 0;
        cudaMemcpy(d_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // Expand: one thread per frontier node
        int expand_blocks = (h_frontier_size + block_size - 1) / block_size;
        kernel_expand<<<expand_blocks, block_size>>>(
            d_graph_nodes, d_graph_edges,
            d_cur_frontier, h_frontier_size,
            d_updating_graph_mask,
            d_graph_visited, d_cost);

        // Commit: scan all nodes, build next frontier queue
        kernel_commit<<<num_blocks_all, block_size>>>(
            d_updating_graph_mask,
            d_graph_visited,
            d_next_frontier,
            d_frontier_size,
            no_of_nodes);

        // Read back next frontier size
        cudaMemcpy(&h_frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap frontiers
        int *tmp = d_cur_frontier;
        d_cur_frontier  = d_next_frontier;
        d_next_frontier = tmp;
    }

    cudaDeviceSynchronize();
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    cudaMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);

    FILE *out = fopen(ofile, "w");
    if (!out) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int i = 0; i < no_of_nodes; i++)
        fprintf(out, "%d\t%d\n", i, h_cost[i]);
    fclose(out);

    cudaFree(d_graph_nodes); cudaFree(d_graph_edges);
    cudaFree(d_updating_graph_mask); cudaFree(d_graph_visited);
    cudaFree(d_cost); cudaFree(d_frontier_a); cudaFree(d_frontier_b);
    cudaFree(d_frontier_size);
    free(h_graph_nodes); free(h_graph_edges);
    free(h_updating_graph_mask); free(h_graph_visited); free(h_cost);
    return 0;
}
