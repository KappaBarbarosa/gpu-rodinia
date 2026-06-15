# srad — AI-optimizes-Rodinia results

**Benchmark:** SRAD v2 (Speckle Reducing Anisotropic Diffusion), iterative image
denoising stencil. Workload 2048×2048, ROI (0,0)–(127,127), λ=0.5, **niter=100**.
Output = final J image (4,194,304 floats), tolerance rtol/atol 1e-4 (correct
`--fmad=false` ports match the CPU golden to ~2e-5).

**Serial baseline:** `experiments/srad/serial/srad_serial.c`, compute ≈ 6.7 s.
Faithful port of `openmp/srad/srad_v2/srad.cpp` (same `srand(7)` input, `J=exp(I)`,
prepare/update math). Golden independently verified: reproducible from the serial;
algorithm matches the OpenMP reference.

**Harness:** `harness/manifests/srad.json`; metric = nsys GPU kernel time; GPU-pinned
pool (no inter-kernel contention) + revert-to-best phase-2. srad was run **twice**
under clean conditions (v1 = `_run_v1_gpupool/`; v2 = current on-disk, with the
diagnoser-stop fix). A still-earlier run was contaminated by a concurrent workflow
and is discarded (`_archive_contaminated/`).

## Two clean runs — the ranking FLIPPED

| Model  | Run v1 best | Run v2 best | Best-of-2 |
|--------|-------------|-------------|-----------|
| **Sonnet** | **17.15 ms / 389×** (r4) | 62.00 ms / 109× (r3) | ~17 ms |
| **Haiku**  | 93.96 ms / 71× (r1)     | **16.08 ms / 414×** (r3) | ~16 ms |
| **Opus**   | 34.19 ms / 196× (r1)    | 165.45 ms / 40× (r4) | ~34 ms |

v2 trajectories (kernel ms): Opus p1 WRONG→WRONG→WRONG→WRONG→**165** (only correct
at r4); Sonnet p1 WRONG→WRONG→94→**62**→62; Haiku p1 240→18.1→16.5→**16.1**.
All "best" entries verified exact (0 mismatches / 4.19 M, max_abs ≈ 2e-5,
memcheck-clean).

## The headline finding: run-to-run variance dominates, even here

srad is a genuine **headroom** benchmark — phase-2 optimization legitimately buys
4–15× over the naive port (shared-mem ROI reduction, fused prepare+update kernels,
thread-coarsening, float4 vectorization all appear in the fast versions). We
expected the per-model numbers to be meaningful here (unlike the
memory/launch-bound bfs/hotspot). **They are not, at N=1.** Across the two runs:

- Haiku swung **94 → 16 ms** (worst to best);
- Opus swung **34 → 165 ms** (middle to worst — and failed correctness in 4 of 5
  v2 candidates);
- Sonnet swung **17 → 62 ms**.

The winner is different in each run. The variance comes from **blind generation**:
each spawn is an independent draw that varies in *both* correctness (does the
kernel compile and match the golden?) and quality (which optimizations it lands).
The harness fixes (revert-to-best, ignore premature diagnoser-stop) are correct and
help a model *compound* gains within a run, but they cannot remove the across-run
randomness of the initial draws — e.g. v2 Opus's problem was correctness, not early
stopping, so the stop-fix couldn't help it.

**Implication:** single-run per-benchmark numbers are noise-dominated even for
headroom benchmarks. A meaningful per-model ranking needs **best-of-N** (or a
distribution) per benchmark. The most defensible single statistic is the
**best-of-N kernel time = the ceiling a model could reach**; by that measure srad is
Sonnet ≈ Haiku (~16–17 ms) > Opus (~34 ms) — but even that rests on only N=2.

## What is robust (holds across both runs)
- srad has large real compute headroom; the fast solutions all use shared memory +
  kernel fusion + vectorization, so phase-2 iteration genuinely matters here.
- 388–414× over serial is achievable; the naive phase-1 port is ~120–240 ms and a
  well-optimized version is ~16–17 ms.
- The earlier "Haiku beats Opus" inversion from the contaminated run was an
  artifact, but ironically v2 reproduces Haiku < Opus *legitimately* — underlining
  that at N=1 you cannot read model capability off one run.

## Caveat
All numbers are 1–2 samples. Treat the per-model srad ranking as unsettled; the
durable takeaways are the achievable speedup range and the dominance of run-to-run
variance over model differences.
