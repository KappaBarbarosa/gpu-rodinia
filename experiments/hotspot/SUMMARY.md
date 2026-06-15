# HotSpot — results (Opus / Sonnet / Haiku, blind)

Workload: 1024×1024 grid, sim_time=1000. Metric: nsys GPU kernel time (median over the run).
Serial compute baseline: ~5.77 s. Correctness: bit-close vs serial golden (rtol 1e-5, atol 1e-4),
correctness build uses `nvcc --fmad=false` (naive correct ports are bit-exact). ncu unavailable
(admin-only counters) → optimization guided by nsys timing.

## Final per-model result (fastest CORRECT version across phase1+phase2)

| Model  | Phase1 (naive)        | Phase2 best correct      | Speedup vs serial | Phase2 rounds used |
|--------|-----------------------|--------------------------|-------------------|--------------------|
| Opus   | correct, 34.5 ms      | **33.7 ms** (r3)         | 173.6×            | 3 (converged)      |
| Sonnet | correct, 34.5 ms      | **33.5 ms** (r4)         | 173.1×            | 4 (recovered late) |
| Haiku  | INCORRECT (bug)       | **none correct**         | —                 | 4 (cap, no success)|

## Trajectories
- **Opus**: p1 correct → p2r1 wrong (`step` missing `/1000` → 1000× timestep) → p2r2 correct but slow (pyramid 37.3 ms) → p2r3 correct & fast (reverted to naive + `__restrict__`/`__ldg`/32×8, 33.7 ms). Clean, fast convergence.
- **Sonnet**: p1 correct → p2r1 build fail (`CUDART_PI_F` no header) → p2r2 correct but slow (32×32 tile, 48 ms) → p2r3 regressed (broke output format, values drifted, 90 ms) → p2r4 recovered (clean naive + restrict, correct, 33.5 ms). Needed all 4 rounds.
- **Haiku**: p1 INCORRECT (double buffer-parity selection wrote initial field) → p2r1 wrong → p2r2 wrong (binary `fread` of TEXT input; hardcoded nonsense coeffs; dropped ambient term) → p2r3 build fail (host var `t_ambient` in device code) → p2r4 CLI arg-count check wrong → exits with usage, no output. A different distinct bug each round; never produced a correct program.

## Key findings
- **Naive is near-optimal here.** HotSpot is a memory-bound 5-point stencil; on Ampere (large L2) naive global-memory neighbour reads are largely cached, so shared-memory tiling and temporal/pyramid blocking added overhead without net speedup. Best results ≈ "clean naive + `__restrict__`/`__ldg`", only ~2–3% under naive (within noise).
- **Phase1 already discriminates models** (Opus/Sonnet correct & bit-exact; Haiku buggy) — enabled only by the tight tolerance (correct `--fmad=false` ports are bit-exact, so logic bugs show even though this gentle workload barely evolves the field).
- **Iteration value differs sharply by model**: Opus fixed its single bug and optimized within 3 rounds; Sonnet thrashed (build/format/perf regressions) but recovered by round 4; Haiku never converged in 4 rounds — markedly weaker at blind CUDA generation on this task.
