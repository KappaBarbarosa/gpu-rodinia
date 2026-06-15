/*
 * Breadth-First Search (bfs) — CUDA phase-2, round 1.
 * Level-synchronous BFS, one thread per node, two kernels per level.
 *
 * Carried-over (verified) optimizations:
 *   - BLOCK_SIZE 128: swept minimum for the divergent, memory-bound
 *     frontier-expansion kernel1 (kernel1 786k->697k ns).
 *   - const __restrict__ on all read-only pointers -> read-only data
 *     cache + no aliasing assumptions (main kernel1 win).
 *   - kernel2 block-local shared stop flag: one thread/block writes the
 *     single global stop word instead of every updated thread.
 *
 * New this round (cheap, removes per-level host->device traffic):
 *   - Reset the stop flag on the DEVICE with cudaMemsetAsync instead of a
 *     blocking cudaMemcpy(HtoD) every iteration. The old code issued a
 *     synchronous 4-byte HtoD copy before each kernel1 launch, which
 *     serialized the stream and added a host roundtrip per level (11
 *     levels). cudaMemsetAsync on the default stream keeps the reset in
 *     the kernel stream and avoids the host sync, so the only host
 *     readback per level is the result copy. Numerics are unchanged.
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct Node {
    int starting;
    int no_of_edges;
};

__global__ void kernel1(const Node *__restrict__ g_nodes,
                        const int *__restrict__ g_edges,
                        char *__restrict__ g_mask, char *__restrict__ g_updating_mask,
                        const char *__restrict__ g_visited, int *__restrict__ g_cost,
                        int no_of_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < no_of_nodes && g_mask[tid]) {
        g_mask[tid] = 0;
        Node n = g_nodes[tid];
        int begin = n.starting;
        int end   = n.starting + n.no_of_edges;
        int cnext = g_cost[tid] + 1;
        for (int i = begin; i < end; i++) {
            int nid = g_edges[i];
            if (!g_visited[nid]) {
                g_cost[nid] = cnext;
                g_updating_mask[nid] = 1;
            }
        }
    }
}

__global__ void kernel2(char *__restrict__ g_mask, char *__restrict__ g_updating_mask,
                        char *__restrict__ g_visited, int *__restrict__ g_stop,
                        int no_of_nodes)
{
    __shared__ int s_any;
    if (threadIdx.x == 0) s_any = 0;
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < no_of_nodes && g_updating_mask[tid]) {
        g_mask[tid] = 1;
        g_visited[tid] = 1;
        g_updating_mask[tid] = 0;
        s_any = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_any) *g_stop = 1;
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

    Node *graph_nodes         = (Node *)malloc(sizeof(Node) * no_of_nodes);
    char *graph_mask          = (char *)malloc(sizeof(char) * no_of_nodes);
    char *updating_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
    char *graph_visited       = (char *)malloc(sizeof(char) * no_of_nodes);
    int  *cost                = (int  *)malloc(sizeof(int)  * no_of_nodes);
    if (!graph_nodes || !graph_mask || !updating_graph_mask || !graph_visited || !cost) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    int start, edgeno;
    for (int i = 0; i < no_of_nodes; i++) {
        if (fscanf(fp, "%d %d", &start, &edgeno) != 2) { fprintf(stderr, "bad node line\n"); return 1; }
        graph_nodes[i].starting     = start;
        graph_nodes[i].no_of_edges  = edgeno;
        graph_mask[i]          = 0;
        updating_graph_mask[i] = 0;
        graph_visited[i]       = 0;
        cost[i]                = -1;
    }

    int source = 0;
    fscanf(fp, "%d", &source);
    source = 0;

    int edge_list_size = 0;
    if (fscanf(fp, "%d", &edge_list_size) != 1 || edge_list_size <= 0) {
        fprintf(stderr, "bad edge count\n"); return 1;
    }
    int *graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
    if (!graph_edges) { fprintf(stderr, "alloc failed\n"); return 1; }
    int id, c;
    for (int i = 0; i < edge_list_size; i++) {
        if (fscanf(fp, "%d %d", &id, &c) != 2) { fprintf(stderr, "bad edge line\n"); return 1; }
        graph_edges[i] = id;
    }
    fclose(fp);

    graph_mask[source]    = 1;
    graph_visited[source] = 1;
    cost[source]          = 0;

    /* Device buffers. */
    Node *d_nodes; int *d_edges; char *d_mask, *d_updating_mask, *d_visited;
    int *d_cost, *d_stop;
    cudaMalloc((void**)&d_nodes, sizeof(Node) * no_of_nodes);
    cudaMalloc((void**)&d_edges, sizeof(int) * edge_list_size);
    cudaMalloc((void**)&d_mask, sizeof(char) * no_of_nodes);
    cudaMalloc((void**)&d_updating_mask, sizeof(char) * no_of_nodes);
    cudaMalloc((void**)&d_visited, sizeof(char) * no_of_nodes);
    cudaMalloc((void**)&d_cost, sizeof(int) * no_of_nodes);
    cudaMalloc((void**)&d_stop, sizeof(int));

    cudaMemcpy(d_nodes, graph_nodes, sizeof(Node) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, graph_mask, sizeof(char) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_updating_mask, updating_graph_mask, sizeof(char) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, graph_visited, sizeof(char) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);

    int num_blocks = (no_of_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double t0 = now_seconds();
    int stop;
    do {
        /* Reset the stop word on-device (no blocking HtoD host roundtrip). */
        cudaMemsetAsync(d_stop, 0, sizeof(int));
        kernel1<<<num_blocks, BLOCK_SIZE>>>(d_nodes, d_edges, d_mask,
            d_updating_mask, d_visited, d_cost, no_of_nodes);
        kernel2<<<num_blocks, BLOCK_SIZE>>>(d_mask, d_updating_mask,
            d_visited, d_stop, no_of_nodes);
        cudaMemcpy(&stop, d_stop, sizeof(int), cudaMemcpyDeviceToHost);
    } while (stop);
    cudaDeviceSynchronize();
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    cudaMemcpy(cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);

    FILE *out = fopen(ofile, "w");
    if (!out) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int i = 0; i < no_of_nodes; i++)
        fprintf(out, "%d\t%d\n", i, cost[i]);
    fclose(out);

    cudaFree(d_nodes); cudaFree(d_edges); cudaFree(d_mask);
    cudaFree(d_updating_mask); cudaFree(d_visited); cudaFree(d_cost); cudaFree(d_stop);
    free(graph_nodes); free(graph_mask); free(updating_graph_mask);
    free(graph_visited); free(cost); free(graph_edges);
    return 0;
}
