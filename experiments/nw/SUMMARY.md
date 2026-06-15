# Needleman-Wunsch (nw) — results (in progress)

Workload: dim=2048, penalty=10 → (2049×2049) integer score matrix. Metric: nsys GPU kernel time.
Serial compute ~9–14 ms (very fast integer DP). Correctness: EXACT integer match (zero tolerance).
Input generated internally (srand(7)); no dataset. Parallelism = anti-diagonal WAVEFRONT.

## Final standings (fastest CORRECT version; phase2 cap = 4 rounds)

| Model  | Phase1 (per-diagonal naive) | Phase2 (block-diagonal)     | Best correct kernel | Speedup vs serial |
|--------|-----------------------------|------------------------------|---------------------|-------------------|
| Opus   | correct, 11.0 ms            | **correct, 1.46 ms (r1)**    | **1.46 ms**         | **6.24×**         |
| Sonnet | correct, 11.0 ms            | r1 ref-index wrong → r2 output-to-wrong-place+zeros → r3 opened penalty as filename → r4 runs but values still wrong | 11.0 ms (phase1) | 1.3× |
| Haiku  | BUILD FAIL (`maximum` not `__device__`) | r1 coverage gap → r2 "failed to read dimensions" → r3 size off-by-one (2050²) → r4 build fail again | **none correct** | — |

Opus got the block-diagonal optimization correct in ONE round (6.24×). Sonnet's block-diagonal was never correct across 4 rounds — a different host/IO/algorithm bug each round (best = its correct phase1 naive). Haiku never produced a correct or compiling version at all (phase1 build fail + 4 failed phase2 rounds).

## Notes
- **nw has real phase2 headroom** (unlike hotspot): naive does ~2*dim≈4096 kernel launches → launch-overhead-bound (GPU ≈ serial). Block-diagonal (BLOCK_SIZE tiles, ~2*dim/BS launches + shared memory) is the Rodinia optimization. **Opus implemented it correctly in one round → 6.24× / 7.5× kernel speedup.**
- Sonnet phase2 r1 bug: reference matrix value wrong at the first interior cell (ref indexing/generation). r2 regressed elsewhere.
- Haiku: phase1 didn't compile; phase2 r1 incomplete anti-diagonal coverage (bottom-right never computed); r2 broke input/arg handling.
- **Cross-benchmark pattern (hotspot + nw): Opus converges reliably (often fast); Sonnet is correct-but-thrashy (new bug each round, sometimes recovers); Haiku is weak (frequently never reaches a correct version within the round cap).**
