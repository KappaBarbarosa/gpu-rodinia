/*
 * Breadth-First Search (bfs) — serial reference implementation.
 *
 * Single-source BFS over a directed graph given in the Rodinia adjacency-array
 * format. Computes, for every node, the hop distance (cost) from the source;
 * unreachable nodes keep cost -1. Source is node 0 (matches the CUDA reference,
 * which forces source=0 regardless of the value stored in the file).
 *
 * This is the openmp/bfs algorithm with the OpenMP pragmas stripped: a level-
 * synchronous BFS. Each "round" first expands every node currently in the
 * frontier (mask) — relaxing its out-edges — then folds the freshly updated
 * set into the visited set and into the next frontier. The two-phase structure
 * (expand, then commit) is exactly what makes each phase data-parallel on a GPU,
 * but it is numerically identical to a plain queue-based BFS: every node is
 * settled at its true shortest hop distance because all edges have unit weight
 * and a node is only assigned a cost the first time it is discovered.
 *
 * CLI:  bfs_serial <input_file> <output_file>
 *   input_file  : Rodinia graph file (see format below).
 *   output_file : one line per node, "<index>\t<cost>\n", index 0..N-1.
 *
 * Input file format:
 *   line 1            : N  (number of nodes)
 *   next N lines      : <starting> <no_of_edges>   (CSR-style offset+degree
 *                       into the edge array, for node i)
 *   next line         : source        (ignored; source is forced to 0)
 *   next line         : E  (number of edges)
 *   next E lines      : <dest_id> <cost>           (cost is ignored — unit hops)
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct Node {
    int starting;
    int no_of_edges;
};

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

    struct Node *graph_nodes = (struct Node *)malloc(sizeof(struct Node) * no_of_nodes);
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
    fscanf(fp, "%d", &source);   /* read & discard the file's source */
    source = 0;                  /* forced to 0, as in the CUDA reference */

    int edge_list_size = 0;
    if (fscanf(fp, "%d", &edge_list_size) != 1 || edge_list_size <= 0) {
        fprintf(stderr, "bad edge count\n"); return 1;
    }
    int *graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
    if (!graph_edges) { fprintf(stderr, "alloc failed\n"); return 1; }
    int id, c;
    for (int i = 0; i < edge_list_size; i++) {
        if (fscanf(fp, "%d %d", &id, &c) != 2) { fprintf(stderr, "bad edge line\n"); return 1; }
        graph_edges[i] = id;     /* edge cost is ignored: BFS uses unit hops */
    }
    fclose(fp);

    graph_mask[source]    = 1;
    graph_visited[source] = 1;
    cost[source]          = 0;

    /* Level-synchronous BFS. */
    double t0 = now_seconds();
    int stop;
    do {
        stop = 0;
        /* Phase 1: expand the current frontier. */
        for (int tid = 0; tid < no_of_nodes; tid++) {
            if (graph_mask[tid]) {
                graph_mask[tid] = 0;
                int begin = graph_nodes[tid].starting;
                int end   = graph_nodes[tid].no_of_edges + graph_nodes[tid].starting;
                for (int i = begin; i < end; i++) {
                    int nid = graph_edges[i];
                    if (!graph_visited[nid]) {
                        cost[nid] = cost[tid] + 1;
                        updating_graph_mask[nid] = 1;
                    }
                }
            }
        }
        /* Phase 2: commit newly discovered nodes into the next frontier. */
        for (int tid = 0; tid < no_of_nodes; tid++) {
            if (updating_graph_mask[tid]) {
                graph_mask[tid]          = 1;
                graph_visited[tid]       = 1;
                stop                     = 1;
                updating_graph_mask[tid] = 0;
            }
        }
    } while (stop);
    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    FILE *out = fopen(ofile, "w");
    if (!out) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    for (int i = 0; i < no_of_nodes; i++)
        fprintf(out, "%d\t%d\n", i, cost[i]);
    fclose(out);

    free(graph_nodes); free(graph_mask); free(updating_graph_mask);
    free(graph_visited); free(cost); free(graph_edges);
    return 0;
}
