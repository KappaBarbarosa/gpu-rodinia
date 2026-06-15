# LU Decomposition (lud) — Algorithm Specification

## Problem
Compute the **non-pivoting (Doolittle) LU factorization** of a dense `N x N`
matrix `A` of `float`s, **in place**. On output, `A` is overwritten so that:
- the strictly-lower triangle holds `L` (`L[i][j]` for `i>j`); the unit diagonal
  of `L` (all 1s) is **implicit and NOT stored**;
- the upper triangle including the diagonal holds `U` (`U[i][j]` for `i<=j`).
So `A = L*U` with `L` unit-lower-triangular and `U` upper-triangular.

There is **no pivoting**. The input matrix is chosen so the pivots (the
diagonal elements `a[i][i]` at the moment they are used) stay non-zero; do not
add row swaps.

## Input file format (Rodinia lud `.dat`)
```
N                       # matrix dimension (first line)
v_0 v_1 v_2 ...         # then N*N floats, row-major, whitespace separated
```
Parsed exactly as `openmp/lud/common/common.c::create_matrix_from_file`:
read the size with `fscanf("%d\n", &N)`, then read each of the `N*N` entries
with `fscanf("%f ", ...)` in row-major order into `a[i*N+j]`.

The workload is `data/lud/2048.dat` → `N = 2048`.

## The LU recurrence (reproduce EXACTLY — this is `lud_base.c`)
The outer loop runs `i = 0 .. N-1`. At step `i`:

**Row of `U`** — for `j = i .. N-1`:
```
sum = a[i*N + j]
for k = 0 .. i-1:  sum -= a[i*N + k] * a[k*N + j]   # increasing k order
a[i*N + j] = sum
```

**Column of `L`** — for `j = i+1 .. N-1`:
```
sum = a[j*N + i]
for k = 0 .. i-1:  sum -= a[j*N + k] * a[k*N + i]   # increasing k order
a[j*N + i] = sum / a[i*N + i]                        # divide by the pivot
```

The `U` row is computed first, then the `L` column, because the `L` column
divides by `a[i*N+i]`, which the `U`-row step (at `j=i`) has just finalized as
the pivot for this step.

Each `sum` is a single `float` accumulator updated in **strictly increasing `k`
order** with one subtraction of a `float` product at a time. Reproducing this
accumulation order is what makes a port match the serial golden bit-closely
under `--fmad=false` (see "Numerical type").

## Dependency / parallelism property
The outer `i` loop is **strictly sequential**: step `i` reads rows/columns
`0 .. i-1` that earlier steps finalized, and step `i+1` depends on everything
step `i` wrote.

WITHIN a single step `i`, however, the work is **data-parallel**:
- the `U`-row entries `a[i][j]` (`j = i .. N-1`) are independent of each other —
  one thread per `j`;
- the `L`-column entries `a[j][i]` (`j = i+1 .. N-1`) are independent of each
  other — one thread per `j` — but they all read the pivot `a[i][i]`, so the
  `U`-row pass must fully complete (at least the `j=i` element) before the
  `L`-column pass starts.

### Phase-1 target: a STRAIGHTFORWARD data-parallel port (NOT blocked)
For phase 1, write the simple, direct port:
- Host loops `i = 0 .. N-1` sequentially.
- For each `i`, launch a kernel (or two) that computes the `U` row in parallel
  over `j` and then the `L` column in parallel over `j`, each thread running the
  **same increasing-`k` accumulation loop** shown above. Keep a single `float`
  accumulator per element and accumulate in increasing `k`.
- Do NOT implement the classic 3-kernel blocked diagonal/perimeter/internal
  scheme (the Rodinia `cuda/lud` design). Blocking reorders the `k`
  accumulations and will NOT match the serial golden bit-closely; it is a
  phase-2 optimization, not the phase-1 correctness target.

(For context only: the high-performance GPU design partitions the matrix into
`BS x BS` blocks and uses three kernels per diagonal block — factor the diagonal
block, update the perimeter row/column blocks, then update the trailing
internal blocks via shared-memory matrix multiply. That is intentionally out of
scope for the first correctness port.)

## Numerical type — CRITICAL
All data is `float` (single precision). The serial reference accumulates each
`sum` in a single `float` variable, subtracting one `float` product per `k` in
increasing-`k` order, with no fused multiply-add. To match the CPU golden
closely under the correctness build (`--fmad=false`), the port must:
- keep `float` storage,
- accumulate in **increasing `k` order** with a single `float` accumulator,
- avoid FMA contraction (the build passes `--fmad=false`).

Note: this particular `2048.dat` matrix is **ill-conditioned for non-pivoting
LU** (condition number ~1e14), so factor entries grow to large magnitudes
(up to ~6.6e4) in later rows and tiny rounding differences amplify there. A
faithful straightforward port matches the golden bit-closely; the comparison
tolerance is set to absorb only minor benign float-order wiggle, not a different
(e.g. blocked) accumulation scheme.

## I/O contract (must match exactly)
Command line:
```
<prog> <input_file> <output_file>
```
- `input_file`: a matrix `.dat` file in the format above.
- `output_file`: the full factored `N*N` matrix, one line per cell
  `"<index>\t<value>\n"`, row-major, `index` from `0` to `N*N-1`, value printed
  with `%.6f`:
```
for idx in 0 .. N*N-1: print  idx, a[idx]
```
Time ONLY the factorization (exclude file I/O) and print
`compute_seconds: %.6f\n` to **stderr**.
