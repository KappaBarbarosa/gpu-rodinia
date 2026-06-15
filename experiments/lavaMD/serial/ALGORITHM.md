# lavaMD — Algorithm Specification

## Problem
lavaMD is an N-body-style molecular-dynamics kernel. 3-D space is partitioned
into a regular grid of `boxes1d^3` boxes; each box holds `NUMBER_PAR_PER_BOX =
100` particles. Every particle has a 4-vector `rv = {v, x, y, z}` (position +
a per-particle distance term `v`) and a scalar charge `qv`. For each particle in
each **home box**, we accumulate a 4-vector force/potential `fv = {v, x, y, z}`
from its pairwise interaction with every particle in the home box and in each of
its (up to 26) neighbour boxes. The per-pair physics is a Gaussian-screened
force/potential.

There is **NO external input file** — the positions and charges are GENERATED.
The original Rodinia code seeds with `srand(time(NULL))`; this reference pins
`srand(7)` so the input (and therefore the golden output) is fully reproducible.
A blind CUDA author MUST reproduce the same input exactly.

## Deterministic input generation (reproduce EXACTLY)
Constants: `NUMBER_PAR_PER_BOX = 100`, `alpha = 0.5`, `a2 = 2*alpha*alpha = 0.5`.
Total boxes `nb = boxes1d^3`; total particles `N = nb * 100`.

1. `srand(7);`  (called ONCE, before any generation)
2. For each particle `i = 0 .. N-1` (in flat box-major order), generate `rv[i]`
   with **four** `rand()` calls in this exact order:
   ```
   rv[i].v = (rand()%10 + 1) / 10.0
   rv[i].x = (rand()%10 + 1) / 10.0
   rv[i].y = (rand()%10 + 1) / 10.0
   rv[i].z = (rand()%10 + 1) / 10.0
   ```
3. THEN, in a separate loop over `i = 0 .. N-1`, generate the charge with **one**
   `rand()` call each:
   ```
   qv[i] = (rand()%10 + 1) / 10.0
   ```
4. Initialize all `fv[i] = {0,0,0,0}`.

Each value lies in `{0.1, 0.2, ..., 1.0}`. The generation ORDER matters: all
`rv` first (4 draws/particle), then all `qv` (1 draw/particle), using the SAME
`rand()` stream from the single `srand(7)`. To match the C golden bit-for-bit
you must use the platform C `rand()` (glibc); a portable trick is to dump `rv`/
`qv` from a tiny C program. (The harness, however, compares with tolerance, so a
faithful re-implementation of the same draws is sufficient.)

## Box / neighbour structure (reproduce EXACTLY)
Boxes are numbered in z→y→x order: box at grid `(z=i, y=j, x=k)` has
`number = i*boxes1d*boxes1d + j*boxes1d + k` and `offset = number*100` (its first
particle's flat index). Neighbours are the 26 surrounding boxes within grid
bounds (triple loop `l,m,n ∈ {-1,0,1}`, excluding `(0,0,0)`), enumerated in that
same `l,m,n` order; `nn` ≤ 26 is the neighbour count.

## Interaction kernel (verbatim physics)
For each home box `l`, let `rA/fA` be the home box's particle/force arrays
(`fA = &fv[offset_l]`). Iterate `k = 0 .. nn` over source boxes: `k==0` is the
home box itself, `k>=1` is neighbour `k-1`. Let `rB`, `qB` be the source box's
position and charge arrays. Then for each home particle `i` and each source
particle `j` (both `0..99`):
```
r2  = rA[i].v + rB[j].v - DOT(rA[i], rB[j])      // DOT = x*x'+y*y'+z*z'
u2  = a2 * r2
vij = exp(-u2)
fs  = 2.0 * vij
d.x = rA[i].x - rB[j].x
d.y = rA[i].y - rB[j].y
d.z = rA[i].z - rB[j].z
fxij = fs*d.x ;  fyij = fs*d.y ;  fzij = fs*d.z
fA[i].v += qB[j] * vij
fA[i].x += qB[j] * fxij
fA[i].y += qB[j] * fyij
fA[i].z += qB[j] * fzij
```
The home box is always processed first (`k==0`), then neighbours in enumeration
order; within a box, particle `j` runs `0..99`. This is the serial accumulation
order. (A GPU port reorders this sum — see "Numerical type" below.)

## Dependency / parallelism
- `fv` is written ONLY for the home box's particles; neighbour boxes are read
  only. So each home particle's 4-vector force sum is **independent** of every
  other particle's output — the whole computation is **data-parallel over
  particles** (equivalently over home boxes).
- The classic GPU port assigns **one thread-block per box** and one (or a strided
  set of) thread(s) per home particle; each thread loops over the home + neighbour
  boxes and accumulates its own `fv`. **Phase-1 should be a straightforward
  data-parallel port** (global-memory reads of `rv`/`qv`), NOT a shared-memory
  tiled version.
- There are no cross-particle write hazards and no iterative dependency: a single
  pass produces the final `fv`.

## Numerical type — float32, reordering is SAFE
All data and accumulation are **`float` (float32)**: `alpha`, `a2`, `rv`, `qv`,
`fv`, and the per-pair temporaries. `vij = expf(-u2)`.

The accumulation is a sum of bounded, similar-magnitude positive-ish
contributions (each `≈ q * exp(-u2) * O(1)`, well-conditioned). A GPU port will
sum each particle's ~2700 interactions in a DIFFERENT order than this serial
loop. Empirically (boxes1d=10, float32), a fully reversed box+particle ordering
differs from the serial order by **max rel ≈ 1.7e-4** (max abs ≈ 7.3e-3 on
values up to ~2600). float32-vs-float64 drift is **max rel ≈ 3.4e-4**. lavaMD is
therefore oracle-COMPATIBLE: a correct reordered float32 reduction stays within a
small tolerance of the golden. The manifest tolerance (`rtol=1e-3`, `atol=1e-2`)
sits comfortably above the observed drift while still catching real logic/indexing
bugs.

## I/O contract (must match exactly)
Command line:
```
<prog> <boxes1d> <output_file>
```
- No input files: generate `rv`/`qv` as above from `srand(7)`.
- `output_file`: every particle's final `fv`, flattened to **one value per line,
  4 lines per particle** in `v, x, y, z` order, particles in flat box-major order
  (`i = 0 .. N-1`). Line format `"<index>\t<value>\n"` with `index = 0 ..
  4*N-1`, value printed `%.6f`. Total lines = `4 * boxes1d^3 * 100`.
- `compute_seconds: %.6f` is printed to **stderr**, timing ONLY the interaction
  loops (not input generation or output).
