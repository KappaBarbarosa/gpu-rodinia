# pathfinder — AI-optimizes-Rodinia results

**Benchmark:** PathFinder — minimum-cost grid path DP. Grid is `rows × cols`; each
cell holds a cost `wall[r][c]` (rand()%10). DP: `dst[c] = wall[r][c] + min(src[c-1],
src[c], src[c+1])` for each row r. Output = final row values (integer, exact).
Workload: **cols=100,000 × rows=1,000**. Parallelism = all cols are independent
within each row; rows must be processed sequentially (each depends on the previous).

**Serial baseline:** `experiments/pathfinder/serial/pathfinder_serial.c`,
compute ≈ 162 ms. Golden: `golden_100000x1000.out` (100k integers), verified by
independent Python DP (max diff = 0). Tolerance: **exact integer** (rtol=atol=0).

**Harness:** `harness/manifests/pathfinder.json`; nsys kernel time; GPU-pinned pool;
revert-to-best + no-premature-stop; **3 rounds** (token budget).

## Results (fastest correct per model)

| Model | Phase1 | Phase2 best | Kernel | Speedup | Trajectory |
|-------|--------|-------------|--------|---------|------------|
| **Sonnet** | OK (3.19ms) | p2r2 | **1.098 ms** | **120.9×** | p1→r1(1.18)→**r2(1.10)**→r3 WRONG |
| **Haiku** | OK (4.03ms) | p2r2 | 1.207 ms | 108.7× | p1→r1(3.18)→**r2(1.21)**→r3 WRONG |
| **Opus** | OK (1.72ms) | p2r1 | 1.684 ms | 78.3× | p1→**r1(1.68)** (plateau stop) |

All correct entries: 0 mismatches / 100,000, max_abs_err = 0.0, memcheck-clean.

## Key findings

- **All three models produced a correct phase-1 — pathfinder is straightforwardly
  parallel.** Each column in a row is independent; a naive "one thread per column"
  kernel launched 1,000 times (once per row) is correct and easy to write.

- **Sonnet wins this benchmark at 1.098ms / 120.9×.** Both Sonnet and Haiku found
  the key optimization at r2: shared-memory tiling that amortizes the wall-value
  loads and reduces the 1,000-launch overhead. The ~3× gain from phase1 to best
  (3.19ms → 1.10ms for Sonnet; 4.03ms → 1.21ms for Haiku) matches the expected
  benefit of temporal tiling over a per-row launch loop.

- **Opus stopped early due to a strong phase-1.** Its phase-1 was already 1.72ms
  (vs 3.19ms for Sonnet, 4.03ms for Haiku), so p2r1 only improved 2.3% (1.72→1.68ms)
  — below the 5% plateau threshold — and the harness correctly stopped the loop.
  The headroom from Opus's phase-1 was too small to trigger further rounds, even
  though Sonnet/Haiku had room to close 3–4× from worse baselines. Ironically,
  Opus's stronger phase-1 *prevented* it from being pushed to find the faster
  tiled solution.

- **r3 broke both Sonnet and Haiku after a good r2.** After finding the correct
  tiled solution at r2 (~1.1–1.2ms), both models' r3 attempts introduced bugs
  (WRONG output). Revert-to-best preserved the r2 result; the loop ended at
  MAX_ROUNDS=3 with r2 as the best. This is consistent with the cross-benchmark
  pattern: models sometimes regress on the last round trying to over-optimize.

- **Phase-1 baseline quality determines how much optimization pressure the model
  faces.** A model that happens to write a fast naive port will plateau-stop early;
  a model with a slower naive port gets more phase-2 iterations and (if it lands
  the tiled version) can finish faster. This introduces a path-dependence: the
  final ranking partly reflects phase-1 luck, not just phase-2 optimization skill.

## Caveat (methodology)
Single run. The phase-1 kernel times varied widely (Opus 1.72ms vs Haiku 4.03ms)
from the same algorithm spec — pure LLM generation variance (e.g. block size choice,
loop structure) that determines whether the plateau stop fires early or late.
