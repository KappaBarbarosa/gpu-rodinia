export const meta = {
  name: 'rodinia-optimize-bm',
  description: 'AI-optimizes-Rodinia: run phase1 (blind naive) + phase2 (iterative optimize, <=4 rounds) for ONE benchmark across Opus/Sonnet/Haiku, keeping all code artifacts in the workflow runtime. Requires the benchmark to already have harness/manifests/<bm>.json + serial + golden.',
  phases: [
    { title: 'Load' },
    { title: 'Phase1' },
    { title: 'Phase2' },
    { title: 'Summary' },
  ],
}

// ---- inputs ----
const REPO = '/home/marl2025/pp_final/gpu-rodinia'
const BM = (args && args.bm) ? args.bm : 'hotspot3D'
const MODELS = (args && args.models) ? args.models : ['opus', 'sonnet', 'haiku']
const MAX_ROUNDS = (args && args.maxRounds) ? args.maxRounds : 3   // phase2 cap; cut 4->3 to save tokens

// ---- schemas ----
const LOAD_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    serial_code: { type: 'string', description: 'Full contents of the serial_source file.' },
    algorithm_doc: { type: 'string', description: 'Full contents of the algorithm_doc (ALGORITHM.md).' },
    cli_and_io: { type: 'string', description: 'One-paragraph restatement of the exact CLI and output format from the algorithm doc / serial code.' },
  },
  required: ['serial_code', 'algorithm_doc', 'cli_and_io'],
}
const EVAL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    build_ok: { type: 'boolean' },
    verify_ok: { type: 'boolean' },
    mismatches: { type: 'integer', description: '-1 if unknown/not computed' },
    kernel_ms: { type: 'number', description: 'nsys total GPU kernel time in ms, -1 if unknown' },
    speedup: { type: 'number', description: 'kernel speedup vs serial compute, -1 if unknown' },
    htod_ms: { type: 'number', description: 'nsys HtoD transfer time in ms, -1 if unknown' },
    dtoh_ms: { type: 'number', description: 'nsys DtoH transfer time in ms, -1 if unknown' },
    memcheck_clean: { type: 'boolean' },
    error_excerpt: { type: 'string', description: 'Short build/runtime error or verify reason; empty if none.' },
    kernel_breakdown: {
      type: 'array',
      description: 'Per-kernel timing from nsys; [] if not available.',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          name: { type: 'string' },
          count: { type: 'integer' },
          avg_us: { type: 'number' },
          total_ms: { type: 'number' },
        },
        required: ['name', 'count', 'avg_us', 'total_ms'],
      },
    },
  },
  required: ['build_ok', 'verify_ok', 'mismatches', 'kernel_ms', 'speedup', 'htod_ms', 'dtoh_ms', 'memcheck_clean', 'error_excerpt', 'kernel_breakdown'],
}
const DIAG_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    prev_code: { type: 'string', description: 'Full current contents of the candidate .cu being iterated.' },
    diagnosis: { type: 'string', description: 'Precise, actionable feedback: what is wrong and exactly how to fix it (cite the failing metric).' },
    stop: { type: 'boolean', description: 'True if the candidate is correct AND further <5% kernel improvement is expected (converged) — then no more rounds.' },
  },
  required: ['prev_code', 'diagnosis', 'stop'],
}

// --- GPU pool ---
// Pin each timing run to a distinct GPU so at most NGPU candidates are measured at
// once, each ALONE on its device → clean nsys timing AND both RTX 3070s utilised.
// CPU-bound gen/diag agents overlap freely; only the GPU eval is gated by this pool.
const GPUS = (args && args.gpus) ? args.gpus : [0, 1]
const _freeGpus = [...GPUS]
const _gpuWaiters = []
function acquireGpu() {
  if (_freeGpus.length) return Promise.resolve(_freeGpus.shift())
  return new Promise(res => _gpuWaiters.push(res))
}
function releaseGpu(d) {
  const w = _gpuWaiters.shift()
  if (w) w(d)
  else _freeGpus.push(d)
}

function evalCmd(cand, out, label, device) {
  return `Run this exact command and then report the parsed result by reading the JSON it writes:\n` +
    `cd ${REPO} && CUDA_VISIBLE_DEVICES=${device} python3 harness/evaluate.py --manifest harness/manifests/${BM}.json --candidate ${cand} --out ${out} --label ${label}\n` +
    `Then read ${out} and extract: build_correctness.ok -> build_ok; verify.ok -> verify_ok; verify.mismatches (or -1) -> mismatches; timing.nsys.kernel_s*1000 -> kernel_ms (-1 if missing); speedup_kernel_vs_serial_compute -> speedup (-1 if missing); timing.nsys.htod_s*1000 -> htod_ms (-1 if missing); timing.nsys.dtoh_s*1000 -> dtoh_ms (-1 if missing); memcheck.clean -> memcheck_clean; a short build_correctness.stderr tail OR verify.reason -> error_excerpt; timing.nsys.kernel_breakdown -> kernel_breakdown ([] if missing). Use Bash and Read only. Do NOT modify the candidate.`
}

const P1_CONSTRAINT =
  `PHASE 1 — BASIC PARALLELIZATION ONLY. Identify the parallelism in the serial code and write a correct, straightforward CUDA port. Use GLOBAL memory only. Do NOT use shared memory, tiling, temporal/blocking, or other hardware-specific optimizations (those are phase 2). Match the EXACT CLI and output format. Correctness (matching the serial golden) is the hard requirement.`

async function genPhase1(model, loaded) {
  const cand = `${REPO}/experiments/${BM}/phase1/${model}.cu`
  await agent(
    `${P1_CONSTRAINT}\n\nWrite the full .cu to: ${cand}\n\n=== ALGORITHM SPEC ===\n${loaded.algorithm_doc}\n\n=== CLI/IO ===\n${loaded.cli_and_io}\n\n=== SERIAL REFERENCE ===\n${loaded.serial_code}`,
    { agentType: 'blind-cuda-gen', model, phase: 'Phase1', label: `gen:${BM}:${model}:p1` }
  )
  return cand
}

async function evaluate(cand, out, label, phaseTitle) {
  // Acquire a GPU for the duration of this measurement; release it even on error
  // so a failed eval never leaks a device out of the pool.
  const device = await acquireGpu()
  try {
    return await agent(evalCmd(cand, out, label, device),
      { agentType: 'general-purpose', schema: EVAL_SCHEMA, phase: phaseTitle, label: `eval:${label}@gpu${device}` })
  } finally {
    releaseGpu(device)
  }
}

// Run one model through phase1 + up to MAX_ROUNDS of phase2. Returns a record.
//
// REVERT-TO-BEST policy: phase 2 always iterates from the model's BEST CORRECT
// version so far, never from the latest attempt. A round that regresses (correct
// but slower), is wrong, or fails to build does NOT stop the loop and is NOT
// carried forward — the next round restarts from the best and is explicitly told
// what just failed so it tries a DIFFERENT optimization. The loop stops only when:
//   (a) the diagnoser declares convergence while a correct best exists AND the
//       model is not on a steep improvement curve (improvedFast guard), OR
//   (b) MAX_ROUNDS is reached.
// There is NO automatic plateau stop — every model always runs all MAX_ROUNDS so
// results are directly comparable regardless of phase-1 baseline quality.
async function runModel(model, loaded) {
  // phase 1
  const p1 = await genPhase1(model, loaded)
  const p1eval = await evaluate(p1, `${REPO}/experiments/${BM}/phase1/${model}.result.json`, `${model}_p1`, 'Phase1')
  const history = [{ stage: 'phase1', file: p1, ...p1eval }]
  // best = best CORRECT version with a known kernel time; null until one exists.
  let best = (p1eval.verify_ok && p1eval.kernel_ms > 0)
    ? { file: p1, kernel_ms: p1eval.kernel_ms, speedup: p1eval.speedup, stage: 'phase1', eval: p1eval }
    : null
  let lastFile = p1, lastEval = p1eval, lastStage = 'phase1'
  // Did the most recent phase-2 round improve the best by >=5%? While this is true
  // the model is still on a steep curve, so we IGNORE a diagnoser stop=true — it has
  // no way to know the true ceiling (it cannot see the reference) and tends to call
  // "converged" after one big win when much more headroom remains (e.g. Opus stopping
  // at 34ms on srad while Sonnet reached 17ms). We only let stop=true end the loop
  // after at least one phase-2 round AND once improvement has flattened.
  let improvedFast = false

  for (let r = 1; r <= MAX_ROUNDS; r++) {
    // Iterate from the best correct version if we have one; otherwise from the
    // most recent attempt (still chasing first correctness).
    const baseFile = best ? best.file : lastFile
    const baseEval = best ? best.eval : lastEval
    const baseDesc = best
      ? `the BEST CORRECT version so far (${best.stage}, kernel ${best.kernel_ms} ms, ${best.speedup}x). Make it FASTER while staying correct.`
      : `your most recent attempt (${lastStage}), which is NOT yet correct. Fix correctness first.`
    // If the most recent attempt is not the base, tell the diagnoser what it did,
    // so it avoids repeating a move that just regressed or broke.
    const lastNote = (lastFile !== baseFile)
      ? `\nIMPORTANT — your most recent attempt (${lastStage}) is NOT the file above. It ` +
        (lastEval.verify_ok
          ? `was correct but SLOWER: ${lastEval.kernel_ms} ms, a regression vs the best ${best.kernel_ms} ms.`
          : (lastEval.build_ok
              ? `was INCORRECT (${lastEval.mismatches} mismatches): ${lastEval.error_excerpt}`
              : `FAILED TO BUILD: ${lastEval.error_excerpt}`)) +
        ` Do NOT repeat that change — diagnose the BEST version above and propose a DIFFERENT optimization.`
      : ''
    const diag = await agent(
      `You are reviewing a CUDA candidate for benchmark "${BM}" in an iterative optimization loop.\n` +
      `Read the candidate file ${baseFile} (use Read). This is ${baseDesc}\nIts measured evaluation:\n${JSON.stringify(baseEval)}\n${lastNote}\n\n` +
      `Phase-2 goal: keep it CORRECT (must match the serial golden), then minimize GPU kernel time; hardware optimizations (shared memory, tiling, temporal/block-diagonal, coalescing, occupancy, fewer launches, etc.) are ALLOWED. ` +
      `Do NOT read any reference/hand-tuned/original implementation — diagnose only from the candidate and its metrics.\n` +
      `Return: prev_code = the FULL current contents of ${baseFile}; diagnosis = precise actionable feedback (what to change and exactly how, cite the metric); stop = true ONLY if it is correct AND you are confident NO optimization will yield >5% further kernel improvement.`,
      { agentType: 'general-purpose', schema: DIAG_SCHEMA, phase: 'Phase2', label: `diag:${BM}:${model}:r${r}` }
    )
    // Honor a diagnoser stop only after >=1 phase-2 round and once we've stopped
    // improving fast — never while still on a steep optimization curve.
    if (diag.stop && best && r > 1 && !improvedFast) break

    const rfile = `${REPO}/experiments/${BM}/phase2/${model}/r${r}.cu`
    await agent(
      `PHASE 2, round ${r} for benchmark "${BM}". Hardware optimizations are allowed; correctness (match the serial golden) is mandatory; keep the EXACT CLI and output format.\n\n` +
      `Measured feedback on the version you are improving:\n${diag.diagnosis}\n\n` +
      `Write the full corrected/optimized .cu to: ${rfile}\n\n=== ALGORITHM SPEC ===\n${loaded.algorithm_doc}\n\n=== VERSION TO IMPROVE (start from this) ===\n${diag.prev_code}`,
      { agentType: 'blind-cuda-gen', model, phase: 'Phase2', label: `gen:${BM}:${model}:r${r}` }
    )
    const reval = await evaluate(rfile, `${REPO}/experiments/${BM}/phase2/${model}/r${r}.result.json`, `${model}_p2r${r}`, 'Phase2')
    history.push({ stage: `p2r${r}`, file: rfile, ...reval })
    lastFile = rfile; lastEval = reval; lastStage = `p2r${r}`
    improvedFast = false   // reset; set true only if THIS round improved best >=5%

    // Only a CORRECT round with a real kernel time can update the best.
    if (reval.verify_ok && reval.kernel_ms > 0) {
      const comparable = best && best.kernel_ms > 0
      if (!comparable || reval.kernel_ms < best.kernel_ms) {
        const improvedFrac = comparable ? (best.kernel_ms - reval.kernel_ms) / best.kernel_ms : 1
        best = { file: rfile, kernel_ms: reval.kernel_ms, speedup: reval.speedup, stage: `p2r${r}`, eval: reval }
        if (improvedFrac >= 0.05) improvedFast = true  // still on a steep curve — block diagnoser premature stop
      }
      // correct but not faster than best (regression): keep best, do NOT stop
    }
    // wrong / build-fail / regression: do NOT stop; next round reverts to best
  }

  return { model, bestCorrect: best ? { file: best.file, kernel_ms: best.kernel_ms, speedup: best.speedup, stage: best.stage } : null, history }
}

// ---- run ----
phase('Load')
log(`>>> RUNNING BENCHMARK: ${BM}  (models: ${MODELS.join(', ')}, maxRounds: ${MAX_ROUNDS}) <<<`)
const loaded = await agent(
  `Use Read. Read ${REPO}/harness/manifests/${BM}.json. From it, read the file at its "phase1_inputs.serial_source" (relative to ${REPO}) and "phase1_inputs.algorithm_doc". ` +
  `Return serial_code = full serial source contents, algorithm_doc = full ALGORITHM.md contents, cli_and_io = a one-paragraph restatement of the exact command-line and output format. Return full contents, do not summarize.`,
  { agentType: 'general-purpose', schema: LOAD_SCHEMA, phase: 'Load', label: `load:${BM}` }
)

phase('Phase2')
// Models run CONCURRENTLY (full pipelines overlap), but every GPU measurement is
// gated by the GPU pool above — at most ${GPUS.length} candidates are timed at once,
// each pinned ALONE to its own device. This keeps nsys timing clean (no inter-kernel
// contention) while using both GPUs, so wall time ≈ sequential / NGPU.
log(`Running ${MODELS.length} models concurrently; GPU timing pinned to devices [${GPUS.join(', ')}].`)
const results = await parallel(MODELS.map(m => () => runModel(m, loaded)))

phase('Summary')
const summary = results.filter(Boolean).map(r => ({
  model: r.model,
  best_correct: r.bestCorrect,
  rounds: r.history.length - 1,
  trajectory: r.history.map(h => `${h.stage}:${h.verify_ok ? 'OK' : (h.build_ok ? 'WRONG' : 'BUILDFAIL')}${h.verify_ok && h.kernel_ms > 0 ? `(${h.kernel_ms.toFixed(2)}ms)` : ''}`).join(' -> '),
}))
log(`Benchmark ${BM} done. Summary:\n${JSON.stringify(summary, null, 2)}`)
return { benchmark: BM, summary }
