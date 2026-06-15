# Breadth-First Search (bfs) — Algorithm Specification

## Problem
Single-source breadth-first search over a directed graph. For every node,
compute its **hop distance** (`cost`) from the source node. The source is
**node 0**. Unreachable nodes have cost `-1`. All edges are unit weight (the
per-edge "cost" field in the file is ignored).

## Input file format (Rodinia adjacency array)
```
N                       # number of nodes
start_0  deg_0          # for node 0: offset into edge array + out-degree
start_1  deg_1
...
start_{N-1} deg_{N-1}
source                  # a source node id — IGNORE it, force source = 0
E                       # number of edges
dst_0  c_0              # edge 0: destination node id + a cost (IGNORE the cost)
dst_1  c_1
...
dst_{E-1} c_{E-1}
```
Node `i`'s outgoing edges are `edges[start_i .. start_i + deg_i - 1]`; each such
entry is the destination node id. The graph is given in CSR-like form
(offset + degree per node, then a flat destination array).

## Algorithm — LEVEL-SYNCHRONOUS BFS (the part to parallelize)
BFS proceeds in levels (waves). Maintain three per-node bit arrays — `mask`
(current frontier), `updating_mask` (nodes discovered this level), and
`visited` — plus the integer `cost` array. Initialise `mask[0]=visited[0]=1`,
`cost[0]=0`, all other `cost=-1`. Then repeat until no node is updated:

1. **Expand (data-parallel over all nodes):** for each node `tid` with
   `mask[tid]` set, clear `mask[tid]`, and for each out-edge destination `nid`:
   if `nid` is not visited, set `cost[nid] = cost[tid] + 1` and
   `updating_mask[nid] = 1`.
2. **Commit (data-parallel over all nodes):** for each node `tid` with
   `updating_mask[tid]` set, set `mask[tid]=1`, `visited[tid]=1`,
   clear `updating_mask[tid]`, and record that work happened (loop continues).

Stop when a full round commits nothing.

## Parallelism property
Within each level, **phase 1 is parallel across all nodes** (one thread per
node, reading its own frontier flag and relaxing its out-edges) and so is
**phase 2**. The two phases are separated by a global barrier (kernel boundary).
The classic Harish-et-al GPU port launches two kernels per level, one thread per
node, over `ceil(N/512)` blocks of up to 512 threads, looping the level pair on
the host until a "no-change" flag stays false. The number of levels equals the
eccentricity of the source.

### Note on the cost write (benign race)
Two frontier nodes may relax the same unvisited neighbour `nid` in the same
level. Both write `cost[nid]` — but to the **same value** (`level+1`), because
every node that reaches `nid` first does so at the same BFS level. So the
concurrent writes are not a correctness hazard. A node is only ever assigned a
cost while it is still unvisited, and it becomes visited at the end of the level
it is discovered, so its final cost is its true shortest hop distance.

## Numerical type
All integers. The golden output is matched with **zero tolerance (exact)**.

## I/O contract (must match exactly)
Command line:
```
<prog> <input_file> <output_file>
```
- `input_file`: a graph file in the format above.
- `output_file`: one line per node, `"<index>\t<cost>\n"`, `index` from 0 to
  `N-1` in order (`cost` is the integer hop distance, or `-1` if unreachable).
```
for i in 0..N-1: print  i, cost[i]
```
