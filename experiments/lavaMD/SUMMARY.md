# lavaMD — AI-optimizes-Rodinia results

**Benchmark:** lavaMD — N-body molecular-dynamics potential/force. Particles live in
a 3-D grid of boxes; each particle interacts with all particles in its home box +
26 neighbour boxes (Lennard-Jones-like `exp(-α·r²)` kernel). Workload **boxes1d=10**
→ 1000 boxes × 100 particles = **100,000 particles**, ~270 M pairwise interactions.
Output = per-particle `{v,x,y,z}` → 400,000 floats. Input generated deterministically
from `srand(7)` (documented in ALGORITHM.md so the blind author reproduces it).

**Serial baseline:** `experiments/lavaMD/serial/lavaMD_serial.c`, compute ≈ 2.1 s.
Verbatim port of `openmp/lavaMD/kernel/kernel_cpu.c` physics. Golden reproducible
exactly; cross-checked with an independent numpy reimplementation (max_rel 3.4e-4,
float32 drift). Tolerance rtol 1e-3 / atol 1e-2 (reorder-tolerance verified:
reversed-order float32 sum drifts only 1.7e-4 — oracle-compatible).

**Harness:** `harness/manifests/lavaMD.json`; nsys kernel time; GPU-pinned pool;
revert-to-best + no-premature-stop phase-2; **single run, 3 rounds** (token budget).

## Results (fastest correct per model)

| Model  | Best correct | Kernel | Speedup | Trajectory (kernel ms) |
|--------|--------------|--------|---------|------------------------|
| **Opus**   | phase2 r3 | **0.697 ms** | **2891×** | p1 OK(1.30) → 0.79 → 0.72 → **0.70** |
| **Haiku**  | phase2 r3 | 0.753 ms | 2826× | p1 OK(1.30) → r1 WRONG → 0.89 → **0.75** |
| **Sonnet** | phase1    | 1.293 ms | 1600× | p1 OK(1.29) → r1 WRONG → r2 BUILDFAIL |

All "OK" exact within tolerance (0 mismatches / 400,000, max_abs ≈ 4.9e-4),
memcheck-clean.

## Key findings

- **lavaMD is a strong headroom benchmark and a clean parallelization target.** All
  three models produced a correct naive phase-1 port (one thread per particle, force
  sum over the 27-box neighbourhood — embarrassingly parallel per particle), and the
  speedups are enormous: 1600× naive → ~2900× optimized. Real compute-bound work
  (O(particles × neighbours)), the opposite of bfs/hotspot.

- **Opus optimized cleanly and monotonically — the no-premature-stop fix worked.**
  It improved every round (0.79 → 0.72 → 0.70 ms) using a `__shared__` float4-packed
  staging of partner records (x,y,z,v in 16-byte vector loads) + `__launch_bounds__`
  occupancy tuning, and only stopped when the remaining gain fell below 5% (a genuine
  plateau — exactly the intended behaviour, vs the old harness which would have halted
  it after r1).

- **Haiku again recovered via revert-to-best.** Its r1 was wrong (170 k mismatches),
  but the loop kept its correct phase-1 as best, fed back the failure, and Haiku found
  a correct shared-memory version at r2 (0.89) then improved to 0.75 ms — landing
  essentially tied with Opus. Strong showing for Haiku here.

- **Sonnet thrashed in phase-2 (consistent with its cross-benchmark pattern).** Correct
  phase-1 (1.29 ms), then r1 wrong (170 k mismatches) and r2 failed to build (2 compile
  errors), so its best stayed at the naive phase-1. Algorithmically it reached the same
  naive solution as the others but could not land a correct optimization in the
  (reduced) round budget.

## Caveat (methodology)
Single run. Per the srad finding, blind-generation variance is large, so treat the
per-model ranking as one sample — though here all three at least *reached correct
phase-1*, and the ~0.70–0.75 ms ceiling (Opus, Haiku) vs 1.29 ms (Sonnet, phase-1
only) reflects whether each landed a correct shared-memory optimization this run, not
a stable capability gap. The robust takeaway: lavaMD has ~2900× headroom and the fast
solution is shared-memory partner-staging; getting there is one correct phase-2 round
away, which Opus and Haiku managed and Sonnet did not this run.
