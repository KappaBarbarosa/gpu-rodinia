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

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Work-efficient BFS expansion with load-balanced edge processing.
// One warp per frontier node: thread 0 reads node_id and edge range,
// then all threads in the warp collectively process edges in a loop.
// This reduces intra-warp divergence on the edge loop.
__global__ void bfs_expand_kernel(
    const int *__restrict__ graph_nodes_start,
    const int *__restrict__ graph_nodes_degree,
    const int *__restrict__ graph_edges,
    int *cost,
    int *visited,                 // 0 = unvisited, 1 = visited
    const int *__restrict__ frontier_in,
    int frontier_in_size,
    int *frontier_out,
    int *frontier_out_size,
    int *work_done_flag)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= frontier_in_size) return;

    int node_id = frontier_in[warp_id];
    int begin = graph_nodes_start[node_id];
    int end = begin + graph_nodes_degree[node_id];
    int new_cost = cost[node_id] + 1;

    // All lanes in the warp collectively process edges [begin, end).
    // Distribute edges across lanes in a loop.
    for (int i = begin + lane_id; i < end; i += WARP_SIZE) {
        int nid = graph_edges[i];
        // Atomically claim the node: only the first thread to flip visited
        // 0 -> 1 queues it, so the output frontier has no duplicates.
        if (visited[nid] == 0 && atomicCAS(&visited[nid], 0, 1) == 0) {
            cost[nid] = new_cost;
            int pos = atomicAdd(frontier_out_size, 1);
            frontier_out[pos] = nid;
            *work_done_flag = 1;  // Signal that work was done.
        }
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
    int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
    int *h_visited = (int *)malloc(sizeof(int) * no_of_nodes);

    if (!h_graph_nodes_start || !h_graph_nodes_degree || !h_cost || !h_visited) {
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
        h_visited[i] = 0;
        h_cost[i] = -1;
    }

    int source = 0;
    if (fscanf(fp, "%d", &source) != 1) {
        fprintf(stderr, "bad source\n");
        return 1;
    }

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

    h_visited[source] = 1;
    h_cost[source] = 0;

    /* Allocate device memory */
    int *d_graph_nodes_start;
    int *d_graph_nodes_degree;
    int *d_cost;
    int *d_graph_edges;
    int *d_visited;
    int *d_frontier_in;
    int *d_frontier_out;
    int *d_frontier_out_size;
    int *d_work_done_flag;

    // Pinned host memory for frontier size to enable async transfers.
    int *h_frontier_out_size;
    cudaHostAlloc(&h_frontier_out_size, sizeof(int), cudaHostAllocDefault);

    cudaMalloc(&d_graph_nodes_start, sizeof(int) * no_of_nodes);
    cudaMalloc(&d_graph_nodes_degree, sizeof(int) * no_of_nodes);
    cudaMalloc(&d_cost, sizeof(int) * no_of_nodes);
    cudaMalloc(&d_graph_edges, sizeof(int) * edge_list_size);
    cudaMalloc(&d_visited, sizeof(int) * no_of_nodes);
    cudaMalloc(&d_frontier_in, sizeof(int) * no_of_nodes);
    cudaMalloc(&d_frontier_out, sizeof(int) * no_of_nodes);
    cudaMalloc(&d_frontier_out_size, sizeof(int));
    cudaMalloc(&d_work_done_flag, sizeof(int));

    /* Copy data to device */
    cudaMemcpy(d_graph_nodes_start, h_graph_nodes_start, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_nodes_degree, h_graph_nodes_degree, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);

    // Initialize frontier with source node
    int h_source = source;
    cudaMemcpy(d_frontier_in, &h_source, sizeof(int), cudaMemcpyHostToDevice);

    /* BFS on device: host-driven, one kernel launch per level. */
    double t0 = now_seconds();

    int cur_frontier_size = 1;
    int zero = 0;

    while (cur_frontier_size > 0) {
        // Reset output frontier size and work flag on device.
        cudaMemcpy(d_frontier_out_size, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_work_done_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // Compute grid: one warp per frontier node.
        int threads_needed = cur_frontier_size * WARP_SIZE;
        int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;

        bfs_expand_kernel<<<grid, BLOCK_SIZE>>>(
            d_graph_nodes_start,
            d_graph_nodes_degree,
            d_graph_edges,
            d_cost,
            d_visited,
            d_frontier_in,
            cur_frontier_size,
            d_frontier_out,
            d_frontier_out_size,
            d_work_done_flag);

        // Asynchronously copy frontier size back to pinned host memory.
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(h_frontier_out_size, d_frontier_out_size, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        cur_frontier_size = *h_frontier_out_size;

        // Swap in/out frontier buffers for next level.
        int *tmp = d_frontier_in;
        d_frontier_in = d_frontier_out;
        d_frontier_out = tmp;
    }

    cudaDeviceSynchronize();
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
    cudaFree(d_cost);
    cudaFree(d_graph_edges);
    cudaFree(d_visited);
    cudaFree(d_frontier_in);
    cudaFree(d_frontier_out);
    cudaFree(d_frontier_out_size);
    cudaFree(d_work_done_flag);
    cudaFreeHost(h_frontier_out_size);

    free(h_graph_nodes_start);
    free(h_graph_nodes_degree);
    free(h_cost);
    free(h_visited);
    free(h_graph_edges);

    return 0;
}
