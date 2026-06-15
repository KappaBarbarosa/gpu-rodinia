# Needleman-Wunsch (nw) — Algorithm Specification

## Problem
Global pairwise sequence alignment by dynamic programming. Build a
`(dim+1) x (dim+1)` integer score matrix `M`. Both input "sequences" and the
substitution-score (reference) matrix are generated **deterministically** from
`srand(7)` — there is NO external input file. You MUST reproduce the generation
exactly (same `rand()` call order) or the output will differ.

## Deterministic input generation (reproduce EXACTLY)
Let `max_rows = max_cols = dim + 1`. Using the 24x24 `BLOSUM62` table (verbatim
from the serial reference):
1. `srand(7);`
2. Zero the whole `input_itemsets` matrix.
3. `for i in 1..max_rows-1: input_itemsets[i*max_cols] = rand()%10 + 1;`  (first column)
4. `for j in 1..max_cols-1: input_itemsets[j] = rand()%10 + 1;`            (first row)
   — this exact order of `rand()` calls is normative.
5. `for i in 1..max_cols-1: for j in 1..max_rows-1: ref[i*max_cols+j] = BLOSUM62[input_itemsets[i*max_cols]][input_itemsets[j]];`
6. Overwrite DP boundaries: `for i in 1..max_rows-1: input_itemsets[i*max_cols] = -i*penalty;`
   and `for j in 1..max_cols-1: input_itemsets[j] = -j*penalty;`  (so M[0][0]=0).

This setup (steps 1-6) is inherently serial and runs on the host. Only the DP
fill below is parallelized on the GPU.

## DP recurrence (the part to parallelize)
For `i,j >= 1`:
```
M[i][j] = max( M[i-1][j-1] + ref[i][j],   // diagonal
               M[i][j-1]   - penalty,      // from left  (gap)
               M[i-1][j]   - penalty )     // from above (gap)
```
`max` is the 3-argument integer maximum.

## Parallelism property — WAVEFRONT
`M[i][j]` depends on its up, left, and diagonal neighbours, so cells are NOT all
independent. However, every cell on a single **anti-diagonal** `i + j = d`
depends only on cells with smaller `i+j` (the two previous anti-diagonals).
Therefore all cells on one anti-diagonal can be computed in parallel, and the
anti-diagonals are processed in sequence `d = 2, 3, ..., 2*dim`. A naive GPU
port launches one kernel per anti-diagonal (one thread per cell on it), reading
neighbours from global memory.

## Numerical type
All integers (`int`). Results are exact — the golden is matched with zero
tolerance.

## I/O contract (must match exactly)
Command line:
```
<prog> <dimension> <penalty> <output_file>
```
- No input files: generate the matrices as above.
- `output_file`: the FULL final score matrix, one line per cell
  `"<index>\t<value>\n"`, row-major over all `(dim+1)*(dim+1)` cells, `index`
  from 0. (i.e. for idx in 0..(max_rows*max_cols-1): print idx, M[idx].)
