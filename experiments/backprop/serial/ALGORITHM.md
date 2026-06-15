# Backpropagation (backprop) — Algorithm Specification

## Problem
Train a single three-layer fully-connected neural network for **exactly one
epoch** (one forward pass + one backprop pass) and emit the adjusted
input→hidden weight matrix. This is the Rodinia `backprop` benchmark
(`openmp/backprop`), with the OpenMP pragmas stripped and made self-contained.

Network structure (the `16` and `1` are FIXED in Rodinia and must not change):
- `n_input` input units (+1 index-0 bias/threshold unit) → size `n_input+1`
- `16` hidden units (+1 bias) → size `17`
- `1` output unit (+1 bias) → size `2`

There is **NO external input file** — the input units and all initial weights
are regenerated deterministically from `srand(7)`. You MUST reproduce this
generation exactly.

## Deterministic state generation (reproduce EXACTLY)
RNG is the C library `rand()`. Values use `(float) rand() / RAND_MAX`
(NOTE: the Rodinia `drnd()`/`dpn1()` helpers and the `BIGRND` divisor are
**not** used for these arrays — `bpnn_randomize_weights` and `load` divide by
`RAND_MAX`). Seed and call order, matching `setup → bpnn_initialize(7) →
bpnn_create → load`:

1. `srand(7);`
2. **input_weights** `[i][j]`, `i=0..n_input`, `j=0..16` (row-major):
   `(n_input+1)*17` calls to `rand()`, each `= (float)rand()/RAND_MAX`.
3. **hidden_weights** `[i][j]`, `i=0..16`, `j=0..1`: `17*2` calls, same formula.
4. **input_prev_weights** = all `0.0`  (no rand).
5. **hidden_prev_weights** = all `0.0` (no rand).
6. **target**`[j] = 0.1` for `j=0..1` (no rand). (Only `target[1]` is used.)
7. **input_units**`[k] = (float)rand()/RAND_MAX` for `k=1..n_input`:
   `n_input` calls. `input_units[0]` is NOT set here — it is forced to `1.0`
   (the bias) at the start of the forward pass.

The total `rand()` count is `(n_input+1)*17 + 34 + n_input`. Getting the count
and order right is essential: a blind author who regenerates the same stream
gets bit-identical initial state.

## Constants (from `backprop.h`)
```
ETA      = 0.3    (learning rate)
MOMENTUM = 0.3
```
Squashing function (sigmoid), verbatim from `backprop.c`:
```
squash(x) = 1.0 / (1.0 + exp(-x))
```

## Train kernel — the ONE epoch (this is the timed region)
All arrays are `float`. The forward sums accumulate into a `float`.

1. **Forward input→hidden** (`l1=input_units`, `l2=hidden_units`,
   `conn=input_weights`):
   ```
   input_units[0] = 1.0
   for j in 1..16:
       sum = 0.0
       for k in 0..n_input:  sum += input_weights[k][j] * input_units[k]
       hidden_units[j] = squash(sum)
   ```
2. **Forward hidden→output** (`conn=hidden_weights`):
   ```
   hidden_units[0] = 1.0
   for j in 1..1:
       sum = 0.0
       for k in 0..16:  sum += hidden_weights[k][j] * hidden_units[k]
       output_units[j] = squash(sum)
   ```
3. **Output delta** (`target[j] = 0.1`):
   ```
   for j in 1..1:
       o = output_units[j]
       output_delta[j] = o*(1.0-o)*(target[j]-o)
   ```
4. **Hidden delta**:
   ```
   for j in 1..16:
       h = hidden_units[j]
       sum = 0.0
       for k in 1..1:  sum += output_delta[k]*hidden_weights[j][k]
       hidden_delta[j] = h*(1.0-h)*sum
   ```
5. **Adjust hidden→output weights** (`prev` starts at 0):
   ```
   hidden_units[0] = 1.0
   for j in 1..1:
       for k in 0..16:
           new_dw = ETA*output_delta[j]*hidden_units[k] + MOMENTUM*hidden_prev_weights[k][j]
           hidden_weights[k][j]      += new_dw
           hidden_prev_weights[k][j]  = new_dw
   ```
6. **Adjust input→hidden weights** (the data-parallel hot loop, `prev`=0):
   ```
   input_units[0] = 1.0
   for j in 1..16:
       for k in 0..n_input:
           new_dw = ETA*hidden_delta[j]*input_units[k] + MOMENTUM*input_prev_weights[k][j]
           input_weights[k][j]      += new_dw
           input_prev_weights[k][j]  = new_dw
   ```

## Parallelism property
The dominant work is over the `(n_input+1) x 17` input→hidden weight matrix:
- **Step 1** (forward input→hidden): for each hidden node `j`, a reduction sum
  over all `n_input+1` input units. The 16 hidden nodes are independent. A GPU
  port parallelizes this and will **REORDER** the per-node summation (e.g. tree
  / warp-shuffle reduction).
- **Step 6** (input→hidden weight adjust): fully data-parallel over the
  `(n_input+1) x 16` matrix — one independent element-wise update per weight,
  no reduction.

Phase-1 expectation: a straightforward data-parallel port (one thread per
output element; a per-node reduction for step 1) — NOT a shared-memory /
reduction-tree-optimized implementation. Float type throughout.

## Numerical / oracle notes — READ THIS (important)
This benchmark's input scaling is **degenerate at large `n_input`**, and that
is a property of Rodinia itself, not of this port:

- Inputs `~U(0,1)` and weights `~U(0,1)` make the input→hidden sum grow like
  `0.5 * n_input`. For `n_input >= ~32` the sum is large (tens to thousands),
  so `squash()` saturates to exactly `1.0`. Then `h*(1-h) = 0`, so every
  `hidden_delta[j] = 0`, so the step-6 weight adjustment is **identically 0**.
- Therefore, at the workload size `n_input = 65536`, the adjusted input→hidden
  weight matrix **equals the initial random weights** (to float rounding).
- Consequence for the reduction REORDER: because the only thing the reordered
  forward sum feeds into (the output weights) is `weight + tiny_adjustment`, and
  the adjustment is `0` (or, in the non-saturated regime, `<= ~1e-5` against a
  `~0.5` weight), the reorder perturbation falls **below the float32 ULP of the
  output and rounds away**. Measured reorder relative diff on the output is
  exactly `0` for ascending vs reversed vs pairwise-tree summation, at every
  size tested (8 … 65536). The chosen oracle is reorder-invariant → backprop is
  **oracle-compatible**.
- The remaining variation is float32-vs-float64 / library-`exp` drift. An
  independent float32 reimplementation matched the C golden to max abs `5e-7`
  (that residual is just the `%.6f` print precision).

A correct GPU port must still get the forward pass, `squash`, the output/hidden
deltas, the weight-update arithmetic, the RNG stream, and the row-major output
layout all correct — those are what the oracle verifies. The saturation simply
means the "right answer" happens to equal the initial weights at this size.

## I/O contract (must match exactly)
Command line:
```
backprop_serial <n_input> <output_file>
```
- No input files: regenerate input units + weights from `srand(7)` as above.
- `output_file`: the adjusted **input→hidden weight matrix**
  `input_weights[i][j]`, `i=0..n_input`, `j=0..16`, **row-major**, one value per
  line `"<index>\t<value>\n"`, `index` from 0, value printed `%.6f`.
  Line count = `(n_input+1) * 17`.
- Timing: print `compute_seconds: %.6f` to **stderr**, wrapping ONLY the train
  kernel (steps 1–6), not the state generation or file write.
