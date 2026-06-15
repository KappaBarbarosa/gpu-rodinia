# bfs — AI-optimizes-Rodinia results

**Benchmark:** Breadth-first search, single-source hop distance over `graph1MW_6.txt`
(1,000,000 nodes / 5,999,970 edges; source = node 0; all reachable; max BFS level
= 11). Output = integer cost per node, **exact** match required.

**Serial baseline:** `experiments/bfs/serial/bfs_serial.c`, compute ≈ 0.077 s
(level-synchronous; openmp algorithm with pragmas stripped). Golden independently
verified by a from-scratch queue-based Python BFS (0 mismatches / 1M).

**Harness:** `harness/manifests/bfs.json`; metric = nsys GPU kernel time.

## bfs was run THREE times — treat it as a variance study, not a single ranking

bfs sits at a **memory/launch-bound ceiling** where the kernel (~0.8–1.2 ms) is
dwarfed by the one-time HtoD copy (~4.1–4.4 ms, the 24 MB edge array). At that
ceiling, blind-generation + diagnoser stochasticity swings the numbers more than
any real model/optimization difference. Three runs (the current on-disk artifacts
are the third):

| Run (config) | Opus best | Sonnet best | Haiku best |
|---|---|---|---|
| #1 parallel, contended timing (old harness) | 0.916 (phase1) | 0.850 (p2r1) | 0.922 (p2r1) |
| #2 sequential, clean, revert-to-best | 0.837 (p2r1) | **0.823 (p2r2)** | **FAIL (no correct version)** |
| #3 GPU-pool, clean, revert-to-best *(on disk)* | **0.785 (phase1)** | 0.923 (phase1) | 1.227 (phase1) |

All "correct" entries are exact (0 mismatches / 1,000,000) and memcheck-clean.
Kernel times in ms. Speedups vs serial compute range ~62–100×.

The spread is large and **not** model-ranking signal: Haiku alone went
fail → 0.92 → 1.23 across the three runs; Opus 0.785–0.916; Sonnet 0.823–0.923.
**Single-run rankings on bfs are not meaningful.**

## Robust conclusions (hold across all three runs)

- **bfs is launch/transfer-bound; the naive port is already near-optimal.** Every
  run, the best version was a plain two-kernels-per-level global-memory BFS at
  ~0.8–1.2 ms; HtoD dominates. There is almost no compute to accelerate.

- **Phase-2 hardware optimization yields no durable gain.** Across runs phase-2
  either matched phase-1 (the diagnoser declared convergence and stopped) or
  *regressed* — frontier-worklist compaction and warp-cooperative expansion both
  cost more than scanning all N nodes saves on this shallow (11-level), wide,
  avg-degree-6 graph. The only time phase-2 beat phase-1 meaningfully was run #2's
  Sonnet (0.823 ms), and that margin is within the cross-run spread.

- **bfs has the easiest parallelism in the suite.** It is the only benchmark so
  far where every model produced a correct naive port in *some* run; when Haiku
  compiles, it is correct. (Haiku still failed entirely in run #2 — build errors
  incl. `warpSize` used in a host/linkage context — underscoring its instability.)

- **Opus is the most consistent at producing a fast, correct phase-1.** Sonnet is
  algorithmically adventurous (tries warp-coop / frontier queues) which on a
  ceiling-bound kernel mostly hurts; Haiku is unreliable run-to-run.

## Validated harness behaviour (from run #2)
The **revert-to-best** policy worked in production: Sonnet's winning 0.823 ms came
right after an r1 that regressed to 3.76 ms; the old "any non-improvement = stop"
rule would have halted it at phase-1, but the new policy reverted to best, told it
the warp approach regressed, and let it find the better version. (On a
headroom-bound benchmark this matters more than it did here.)

## Methodological takeaway
For ceiling-bound benchmarks (bfs, hotspot) run-to-run noise exceeds model
differences, so the conclusion is "all models tie at the naive ceiling; phase-2
buys nothing." The benchmarks with real compute headroom (nw, srad, lud, lavaMD)
are where signal dominates noise and the per-model numbers actually separate.
