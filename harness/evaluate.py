#!/usr/bin/env python3
"""
evaluate.py — build, run, verify, and time a candidate CUDA implementation
against a benchmark manifest + golden output.

Produces a JSON report capturing: build success, correctness (vs golden, with
tolerance), wall-clock timing (median of N), speedup vs the serial reference,
and a compute-sanitizer memcheck pass (used as phase-2 feedback).

Usage:
  evaluate.py --manifest harness/manifests/hotspot.json \
              --candidate experiments/hotspot/phase1/cand.cu \
              --out experiments/hotspot/phase1/result.json [--label opus_p1]

Designed so the candidate is a single .cu implementing the manifest's CLI.
The build/run/verify plumbing is identical across phases and models; only the
candidate source differs.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rp(path):
    """Resolve a manifest path relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def build(candidate, flags, out_bin):
    cmd = ["nvcc", *flags, candidate, "-o", out_bin]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode == 0, " ".join(cmd), p.stderr


def nsys_profile(bin_path, args, timeout, workdir):
    """Profile one run with nsys and parse total kernel time and H2D/D2H copy
    time (seconds) from the --stats=true text tables. Returns dict or None."""
    rep = os.path.join(workdir, "prof")
    cmd = ["nsys", "profile", "--force-overwrite=true", "-o", rep,
           "--stats=true", bin_path, *args]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    text = p.stdout + p.stderr

    def section_total_ns(lines, header, row_filter=None):
        """Sum the 'Total Time (ns)' (2nd numeric column) of data rows under
        a section header, optionally filtered by a substring in the row."""
        total = 0.0
        found_hdr = False
        in_rows = False
        for ln in lines:
            if header in ln:
                found_hdr = True
                in_rows = False
                continue
            if found_hdr and set(ln.strip()) <= set("- "):
                if ln.strip():
                    in_rows = True
                    continue
            if in_rows:
                if not ln.strip():
                    break
                if row_filter and row_filter not in ln:
                    continue
                toks = ln.split()
                # toks[0]=pct, toks[1]=Total Time (ns) with commas
                try:
                    total += float(toks[1].replace(",", ""))
                except (IndexError, ValueError):
                    continue
        return total if found_hdr else None

    def parse_kernel_breakdown(ls):
        """Return list of per-kernel dicts from CUDA Kernel Statistics table."""
        kernels = []
        found_hdr = False
        in_rows = False
        for ln in ls:
            if "CUDA Kernel Statistics:" in ln:
                found_hdr = True
                in_rows = False
                continue
            if found_hdr and set(ln.strip()) <= set("- "):
                if ln.strip():
                    in_rows = True
                continue
            if in_rows:
                if not ln.strip():
                    break
                toks = ln.split()
                if len(toks) < 9:
                    continue
                try:
                    total_ns = float(toks[1].replace(",", ""))
                    count = int(toks[2].replace(",", ""))
                    avg_ns = float(toks[3].replace(",", ""))
                    name = " ".join(toks[8:])[:80]
                    kernels.append({
                        "name": name,
                        "count": count,
                        "avg_us": round(avg_ns / 1e3, 2),
                        "total_ms": round(total_ns / 1e6, 4),
                    })
                except (IndexError, ValueError):
                    continue
        return kernels

    lines = text.splitlines()
    kern_ns = section_total_ns(lines, "CUDA Kernel Statistics:")
    htod_ns = section_total_ns(lines, "CUDA Memory Operation Statistics (by time):", "HtoD")
    dtoh_ns = section_total_ns(lines, "CUDA Memory Operation Statistics (by time):", "DtoH")
    if kern_ns is None:
        return {"error": "could not parse nsys output", "raw_tail": text[-1500:]}
    return {
        "kernel_s": kern_ns / 1e9,
        "htod_s": (htod_ns or 0.0) / 1e9,
        "dtoh_s": (dtoh_ns or 0.0) / 1e9,
        "kernel_breakdown": parse_kernel_breakdown(lines),
    }


def run_once(bin_path, args, timeout):
    """Run once, return (ok, wall_seconds, stdout, stderr)."""
    t0 = time.monotonic()
    try:
        p = subprocess.run([bin_path, *args], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, float("inf"), "", "timeout"
    wall = time.monotonic() - t0
    return p.returncode == 0, wall, p.stdout, p.stderr


def parse_indexed_float(path):
    """Parse one temperature per line. Accepts both the spec format
    '<index>\\t<value>' and a bare '<value>' line by taking the LAST
    whitespace token as the temperature (robust to a missing index column).
    Returns (values, format_ok) where format_ok is False if any line lacked
    the expected 2-column '<index>\\t<value>' shape."""
    vals = []
    format_ok = True
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                format_ok = False
            try:
                vals.append(float(parts[-1]))
            except (ValueError, IndexError):
                format_ok = False
    return vals, format_ok


def compare(golden_vals, cand_vals, rtol, atol):
    if len(golden_vals) != len(cand_vals):
        return {"ok": False, "reason": f"length mismatch {len(cand_vals)} vs {len(golden_vals)}"}
    max_abs = 0.0
    max_rel = 0.0
    mismatches = 0
    worst_idx = -1
    for i, (g, c) in enumerate(zip(golden_vals, cand_vals)):
        ae = abs(c - g)
        re = ae / abs(g) if g != 0 else ae
        if ae > max_abs:
            max_abs = ae
        if re > max_rel:
            max_rel = re
        if ae > atol + rtol * abs(g):
            mismatches += 1
            if worst_idx < 0:
                worst_idx = i
    return {
        "ok": mismatches == 0,
        "n": len(golden_vals),
        "mismatches": mismatches,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "first_mismatch_index": worst_idx,
    }


def fmt_args(template, wl, output_file):
    ctx = dict(wl)
    ctx["output_file"] = output_file
    out = []
    for a in template:
        s = a
        for k, v in ctx.items():
            s = s.replace("{" + k + "}", str(v))
        # resolve data file paths relative to repo
        if k in ("temp_file", "power_file") or s.startswith("data/"):
            pass
        out.append(s)
    # resolve the data file args to absolute repo paths
    resolved = []
    for s in out:
        resolved.append(rp(s) if s.startswith("data/") else s)
    return resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    m = json.load(open(rp(args.manifest)))
    wl = m["workload"]
    tol = m["tolerance"]
    repeats = m["run"]["repeats"]
    timeout = m["run"]["timeout_seconds"]
    golden_path = rp(m["golden"])

    report = {"label": args.label, "benchmark": m["name"],
              "candidate": args.candidate}

    workdir = tempfile.mkdtemp(prefix="eval_")
    cand_out = os.path.join(workdir, "cand.out")
    args_run = fmt_args(m["args_template"], wl, cand_out)
    report["run_args"] = args_run

    # --- correctness build (FMA off) ---
    corr_bin = os.path.join(workdir, "cand_corr")
    ok, cmd, err = build(rp(args.candidate), m["build"]["correctness_flags"], corr_bin)
    report["build_correctness"] = {"ok": ok, "cmd": cmd, "stderr": err[-4000:]}
    if not ok:
        report["status"] = "build_failed"
        json.dump(report, open(rp(args.out), "w"), indent=2)
        print(json.dumps({k: report[k] for k in ("status", "build_correctness")}, indent=2))
        return

    # --- correctness run + compare ---
    ok, wall, out, rerr = run_once(corr_bin, args_run, timeout)
    report["correctness_run"] = {"ok": ok, "wall_s": wall, "stderr": rerr[-2000:]}
    if ok and os.path.exists(cand_out):
        cand_vals, fmt_ok = parse_indexed_float(cand_out)
        golden_vals, _ = parse_indexed_float(golden_path)
        report["verify"] = compare(golden_vals, cand_vals, tol["rtol"], tol["atol"])
        report["verify"]["output_format_ok"] = fmt_ok
    else:
        report["verify"] = {"ok": False, "reason": "no output produced"}

    # --- compute-sanitizer memcheck (phase-2 feedback) ---
    san = subprocess.run(["compute-sanitizer", "--tool", "memcheck", corr_bin, *args_run],
                         capture_output=True, text=True, timeout=timeout)
    clean = "ERROR SUMMARY: 0 errors" in (san.stdout + san.stderr)
    report["memcheck"] = {"clean": clean, "tail": (san.stdout + san.stderr)[-2000:]}

    # --- perf build (FMA on) + nsys kernel timing ---
    perf_bin = os.path.join(workdir, "cand_perf")
    ok, cmd, err = build(rp(args.candidate), m["build"]["perf_flags"], perf_bin)
    report["build_perf"] = {"ok": ok, "cmd": cmd, "stderr": err[-2000:]}
    if ok:
        # wall-clock (median of N) as a secondary, end-to-end figure
        times = []
        for _ in range(repeats):
            rok, w, _, _ = run_once(perf_bin, args_run, timeout)
            if rok:
                times.append(w)
        # nsys: isolate GPU kernel + transfer time (primary metric)
        prof = nsys_profile(perf_bin, args_run, timeout, workdir)
        report["timing"] = {
            "median_wall_s": statistics.median(times) if times else None,
            "nsys": prof,
        }
        # serial compute time (excludes I/O) for an apples-to-apples speedup
        serial_bin = rp(m["serial_binary"])
        serial_regex = m["run"].get("serial_compute_regex")
        if os.path.exists(serial_bin):
            serial_args = fmt_args(m["args_template"], wl, os.path.join(workdir, "serial.out"))
            sok, sw, sout, serr = run_once(serial_bin, serial_args, timeout)
            serial_compute = None
            if serial_regex:
                mt = re.search(serial_regex, sout + serr)
                if mt:
                    serial_compute = float(mt.group(1))
            report["serial_compute_s"] = serial_compute
            report["serial_wall_s"] = sw if sok else None
            if serial_compute and prof and prof.get("kernel_s"):
                report["speedup_kernel_vs_serial_compute"] = serial_compute / prof["kernel_s"]

    report["status"] = "ok" if report.get("verify", {}).get("ok") else "incorrect"
    json.dump(report, open(rp(args.out), "w"), indent=2)
    # concise console summary
    summary = {k: report.get(k) for k in
               ("status", "verify", "timing", "serial_compute_s",
                "speedup_kernel_vs_serial_compute", "memcheck")}
    if summary.get("memcheck"):
        summary["memcheck"] = {"clean": summary["memcheck"]["clean"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
