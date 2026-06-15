# PathFinder (pathfinder) — Algorithm Specification

## Problem
Minimum-cost path through a 2D grid of integer weights, the "wall". Starting
from row 0, you walk down the grid one row at a time; from a cell in column `n`
you may step to column `n-1`, `n`, or `n+1` in the next row (boundary-clamped).
The DP computes, for every column, the minimum total weight to reach that column
in the current row from any starting column in row 0. The answer is the
accumulated-cost vector after the last row.

The wall weights are generated **deterministically** from `srand(9)` — there is
NO external input file. You MUST reproduce the generation exactly (same number
and order of `rand()` calls) or the output will differ.

## Deterministic input generation (reproduce EXACTLY)
The wall is a `rows x cols` integer matrix, `wall[i][j]` for row `i` in
`0..rows-1` and column `j` in `0..cols-1`.
1. `srand(9);`  (the seed is `#define M_SEED 9`).
2. Fill the wall in this **normative nested order** — `i` over rows is the OUTER
   loop, `j` over cols is the INNER loop:
   ```
   for i in 0..rows-1:
       for j in 0..cols-1:
           wall[i][j] = rand() % 10;     // weights are 0..9
   ```
   This exact `rand()` call order is normative: a blind port must produce the
   identical wall. (If you flatten the wall, use `wall[i*cols + j]`.)

This generation runs on the host. Only the DP below is parallelized on the GPU.

## DP recurrence (the part to parallelize)
Let `src` be the accumulated-cost vector of the previous row and `dst` that of
the current row (both length `cols`). The initial vector is row 0 of the wall:
```
dst[j] = wall[0][j]   for j in 0..cols-1
```
Then for each step `t = 0 .. rows-2`, ping-pong (`src` <- previous `dst`) and
compute, for every column `n = 0 .. cols-1`:
```
m = src[n];
if (n > 0)        m = min(m, src[n-1]);    // no left neighbour at column 0
if (n < cols-1)   m = min(m, src[n+1]);    // no right neighbour at column cols-1
dst[n] = wall[t+1][n] + m;
```
`min` is the 2-argument integer minimum. Note the neighbour reads are clamped at
the boundaries: column 0 has no left neighbour, column `cols-1` has no right
neighbour (equivalently, the original `IN_RANGE`/`CLAMP_RANGE` macros). After the
final step, `dst` holds the result row that is written out.

## Parallelism property — DATA-PARALLEL OVER COLUMNS
Within a single DP step, `dst[n]` depends only on `src[n-1]`, `src[n]`,
`src[n+1]` — i.e. only on the previous row, which is fully computed before the
step begins. Therefore all `cols` cells of a row are mutually **independent** and
can be computed in parallel; the `rows` steps are **sequential** (a global
barrier / kernel boundary separates them).

The straightforward GPU port (phase 1) launches one kernel per row (or loops
inside one kernel with a grid-wide barrier between rows), **one thread per
column**, each thread reading its three previous-row neighbours from global
memory and writing its own cell. More advanced ports (phase 2) use ghost-zone /
pyramidal blocking to process several rows per kernel from shared memory,
reducing kernel-launch and global-memory traffic — but they must compute the
identical result.

## Numerical type
All integers (`int`); weights are `0..9`, accumulated costs are sums of weights.
Results are **exact** — the golden is matched with zero tolerance (rtol 0,
atol 0). Integer min/add has no reordering hazard.

## I/O contract (must match exactly)
Command line:
```
<prog> <cols> <rows> <output_file>
```
- `cols`: grid width (number of columns; the parallel work width).
- `rows`: number of DP steps (grid height; the sequential depth).
- No input files: generate the wall as above.
- `output_file`: the FINAL accumulated-cost row, one line per column
  `"<index>\t<value>\n"`, `index` from 0 to `cols-1` in order:
  ```
  for n in 0..cols-1: print  n, dst[n]
  ```
