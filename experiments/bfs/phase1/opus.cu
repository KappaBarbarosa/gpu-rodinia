/*
 * Breadth-First Search (bfs) — CUDA phase-2.
 * Level-synchronous BFS, one thread per node, two kernels per level.
 *
 * Optimizations:
 *  - read-only __restrict__ pointers so the memory-bound frontier-expansion
 *    kernel uses the read-only data cache;
 *  - block-local stop flag in kernel2 so only one thread per block touches the
 *    global stop word (kills atomic/write contention on the single stop word);
 *  - VECTORIZED kernel2: the consolidation kernel scans the three 1M-byte mask
 *    arrays four bytes at a time via uchar4 loads (quarters launch width,
 *    improves coalescing).
 *  - DEFERRED COST WRITE (phase-2 r2): the scattered per-neighbour
 *    `cost[nid] = level` store was the dominant cost in kernel1 on the two wide
 *    levels (random 4-byte writes thrash the cache). It is removed from kernel1
 *    and folded into kernel2, which already scans every node and writes `cost`
 *    in a perfectly coalesced sweep (`cost[base+lane] = level`). A node is
 *    discovered exactly once (gated by `!visited`) and marked in updating_mask,
 *    so kernel2 stamps its BFS level. nsys: kernel1 663->378 us. With the cost
 *    write gone, the old `!g_updating_mask[nid]` dedup gate now costs more (an
 *    extra scattered read) than the duplicate 1-byte stores it saved, so it was
 *    dropped too. BLOCK_SIZE 256->128 was best for the now memory-latency-bound
 *    kernel1. Net: total GPU kernel ~785 us -> ~495 us (~37% faster, ~1.6x),
 *    exact match, memcheck-clean, no atomics in the hot path.
 *
 * A frontier-queue (worklist) variant was measured and REGRESSED hard
 * (atomicCAS on cost + atomicAdd on the queue counter -> ~2.9 ms, the wide
 * level alone ~1.46 ms): the average out-degree is ~6 and the wide levels touch
 * almost the whole graph, so atomic contention dwarfs the saved 1M scans. The
 * mask-based, atomic-free scheme is the right ceiling here.
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
                        const char *__restrict__ g_visited,
                        int no_of_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < no_of_nodes && g_mask[tid]) {
        g_mask[tid] = 0;
        Node n = g_nodes[tid];
        int begin = n.starting;
        int end   = n.starting + n.no_of_edges;
        for (int i = begin; i < end; i++) {
            int nid = g_edges[i];
            if (!g_visited[nid]) {
                g_updating_mask[nid] = 1;
            }
        }
    }
}

/* Consolidation pass, vectorized: each thread processes 4 contiguous nodes by
 * loading the updating-mask / mask / visited arrays as uchar4 words. The node
 * arrays are padded up to a multiple of 4 (padding bytes are 0 and never set),
 * so the extra lanes are inert. */
__global__ void kernel2(uchar4 *__restrict__ g_mask, uchar4 *__restrict__ g_updating_mask,
                        uchar4 *__restrict__ g_visited, int *__restrict__ g_cost,
                        int *__restrict__ g_stop, int level, int n4)
{
    __shared__ int s_any;
    if (threadIdx.x == 0) s_any = 0;
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n4) {
        uchar4 u = g_updating_mask[tid];
        if (u.x | u.y | u.z | u.w) {
            uchar4 m = g_mask[tid];
            uchar4 v = g_visited[tid];
            int base = tid * 4;
            if (u.x) { m.x = 1; v.x = 1; g_cost[base    ] = level; }
            if (u.y) { m.y = 1; v.y = 1; g_cost[base + 1] = level; }
            if (u.z) { m.z = 1; v.z = 1; g_cost[base + 2] = level; }
            if (u.w) { m.w = 1; v.w = 1; g_cost[base + 3] = level; }
            g_mask[tid] = m;
            g_visited[tid] = v;
            g_updating_mask[tid] = make_uchar4(0, 0, 0, 0);
            s_any = 1;
        }
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

    /* Pad the per-node byte arrays up to a multiple of 4 for the vectorized
     * kernel2 (padding entries stay 0 and are never marked). */
    int padded = (no_of_nodes + 3) & ~3;

    Node *graph_nodes         = (Node *)malloc(sizeof(Node) * no_of_nodes);
    char *graph_mask          = (char *)calloc(padded, 1);
    char *updating_graph_mask = (char *)calloc(padded, 1);
    char *graph_visited       = (char *)calloc(padded, 1);
    int  *cost                = (int  *)malloc(sizeof(int)  * no_of_nodes);
    if (!graph_nodes || !graph_mask || !updating_graph_mask || !graph_visited || !cost) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    int start, edgeno;
    for (int i = 0; i < no_of_nodes; i++) {
        if (fscanf(fp, "%d %d", &start, &edgeno) != 2) { fprintf(stderr, "bad node line\n"); return 1; }
        graph_nodes[i].starting     = start;
        graph_nodes[i].no_of_edges  = edgeno;
        cost[i]                = -1;
    }

    int source = 0;
    if (fscanf(fp, "%d", &source) != 1) { /* ignore: source forced to 0 below */ }
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

    /* Device buffers (mask arrays sized to the padded length). */
    Node *d_nodes; int *d_edges; char *d_mask, *d_updating_mask, *d_visited;
    int *d_cost, *d_stop;
    cudaMalloc((void**)&d_nodes, sizeof(Node) * no_of_nodes);
    cudaMalloc((void**)&d_edges, sizeof(int) * edge_list_size);
    cudaMalloc((void**)&d_mask, sizeof(char) * padded);
    cudaMalloc((void**)&d_updating_mask, sizeof(char) * padded);
    cudaMalloc((void**)&d_visited, sizeof(char) * padded);
    cudaMalloc((void**)&d_cost, sizeof(int) * padded);
    cudaMalloc((void**)&d_stop, sizeof(int));

    cudaMemcpy(d_nodes, graph_nodes, sizeof(Node) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, graph_mask, sizeof(char) * padded, cudaMemcpyHostToDevice);
    cudaMemcpy(d_updating_mask, updating_graph_mask, sizeof(char) * padded, cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, graph_visited, sizeof(char) * padded, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);

    int num_blocks = (no_of_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int n4         = padded / 4;
    int num_blocks2 = (n4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double t0 = now_seconds();
    int stop;
    int level = 1;
    do {
        stop = 0;
        cudaMemcpy(d_stop, &stop, sizeof(int), cudaMemcpyHostToDevice);
        kernel1<<<num_blocks, BLOCK_SIZE>>>(d_nodes, d_edges, d_mask,
            d_updating_mask, d_visited, no_of_nodes);
        kernel2<<<num_blocks2, BLOCK_SIZE>>>((uchar4*)d_mask, (uchar4*)d_updating_mask,
            (uchar4*)d_visited, d_cost, d_stop, level, n4);
        level++;
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
