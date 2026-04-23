#!/usr/bin/env python3
"""
Run CPU and GPU op benchmarks and compare:
- CPU-only run
- GPU-only run
- CPU+GPU concurrent run (two subprocesses at once)

This answers:
1) What is the latency gap (CPU vs GPU) per op?
2) Does running them concurrently change either side's numbers?

We intentionally run benchmarks in *separate processes* to avoid Python GIL
and to mirror "two jobs on the same machine" contention (CPU threads, memory BW,
GPU SMs, PCIe, etc.).
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import math


_LINE_RE = re.compile(r"^(?P<name>.+?)\s+(?P<ms>\d+(?:\.\d+)?)\s+ms/iter\s*$")


@dataclass(frozen=True)
class RunResult:
    label: str
    returncode: int
    stdout: str
    stderr: str
    ms_by_op: Dict[str, float]


def _parse_ms(stdout: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in stdout.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        out[m.group("name").strip()] = float(m.group("ms"))
    return out


def _run_one(label: str, argv: List[str]) -> RunResult:
    p = subprocess.run(argv, capture_output=True, text=True)
    return RunResult(
        label=label,
        returncode=p.returncode,
        stdout=p.stdout,
        stderr=p.stderr,
        ms_by_op=_parse_ms(p.stdout),
    )


def _run_concurrent(
    cpu_label: str,
    cpu_argv: List[str],
    gpu_label: str,
    gpu_argv: List[str],
) -> Tuple[RunResult, RunResult]:
    cpu_p = subprocess.Popen(cpu_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpu_p = subprocess.Popen(gpu_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cpu_out, cpu_err = cpu_p.communicate()
    gpu_out, gpu_err = gpu_p.communicate()
    return (
        RunResult(cpu_label, cpu_p.returncode, cpu_out, cpu_err, _parse_ms(cpu_out)),
        RunResult(gpu_label, gpu_p.returncode, gpu_out, gpu_err, _parse_ms(gpu_out)),
    )


def _fmt_pct(delta: float) -> str:
    # delta is (new - old) / old
    return f"{delta * 100:+6.1f}%"

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _collect(store: Dict[str, List[float]], ms_by_op: Dict[str, float]) -> None:
    for k, v in ms_by_op.items():
        store.setdefault(k, []).append(v)


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare CPU/GPU op benchmark latency and concurrency impact")
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for both runs (default: current interpreter)",
    )
    ap.add_argument(
        "--mode",
        default="both",
        choices=["standalone", "concurrent", "both"],
        help="What to run: standalone only, concurrent only, or both (default)",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the selected mode(s) N times and report averaged results",
    )

    # Pass-through args for cpu/gpu benches.
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--threads", type=int, default=32, help="CPU threads for cpu_ops_bench.py")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", default="cuda", help="GPU device for gpu_ops_bench.py (e.g. cuda or cuda:0)")

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--num-heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=128_000)
    ap.add_argument("--inter", type=int, default=11008)
    ap.add_argument("--kv-blocks", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-blocks-per-seq", type=int, default=256)

    args = ap.parse_args()

    cpu_cmd = [
        args.python,
        "benchmarks/cpu_ops_bench.py",
        "--dtype",
        args.dtype,
        "--threads",
        str(args.threads),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--batch",
        str(args.batch),
        "--seq",
        str(args.seq),
        "--hidden",
        str(args.hidden),
        "--num-heads",
        str(args.num_heads),
        "--head-dim",
        str(args.head_dim),
        "--vocab",
        str(args.vocab),
        "--inter",
        str(args.inter),
        "--kv-blocks",
        str(args.kv_blocks),
        "--block-size",
        str(args.block_size),
        "--max-blocks-per-seq",
        str(args.max_blocks_per_seq),
    ]

    gpu_cmd = [
        args.python,
        "benchmarks/gpu_ops_bench.py",
        "--dtype",
        args.dtype,
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--device",
        args.device,
        "--batch",
        str(args.batch),
        "--seq",
        str(args.seq),
        "--hidden",
        str(args.hidden),
        "--num-heads",
        str(args.num_heads),
        "--head-dim",
        str(args.head_dim),
        "--vocab",
        str(args.vocab),
        "--inter",
        str(args.inter),
        "--kv-blocks",
        str(args.kv_blocks),
        "--block-size",
        str(args.block_size),
        "--max-blocks-per-seq",
        str(args.max_blocks_per_seq),
    ]

    print("== Compare CPU vs GPU ==")
    print(f"python={shlex.join([args.python])}")
    print(f"shape: B={args.batch} S={args.seq} H={args.hidden} heads={args.num_heads} head_dim={args.head_dim} dtype={args.dtype}")
    print(f"iters={args.iters} warmup={args.warmup} cpu_threads={args.threads} gpu_device={args.device}")
    print(f"mode={args.mode} repeats={args.repeats}")
    print()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    # Accumulators across repeats
    cpu_solo_all: Dict[str, List[float]] = {}
    gpu_solo_all: Dict[str, List[float]] = {}
    cpu_conc_all: Dict[str, List[float]] = {}
    gpu_conc_all: Dict[str, List[float]] = {}

    for i in range(args.repeats):
        rep_tag = f"[{i+1}/{args.repeats}]"
        if args.mode in ("standalone", "both"):
            _print_header(f"Standalone runs {rep_tag}")
            t0 = time.time()
            cpu_standalone = _run_one("CPU (standalone)", cpu_cmd)
            gpu_standalone = _run_one("GPU (standalone)", gpu_cmd)
            t1 = time.time()
            print(f"elapsed={t1 - t0:.1f}s")
            if cpu_standalone.returncode != 0:
                print("\n[CPU standalone stderr]\n" + cpu_standalone.stderr)
                raise SystemExit(cpu_standalone.returncode)
            if gpu_standalone.returncode != 0:
                print("\n[GPU standalone stderr]\n" + gpu_standalone.stderr)
                raise SystemExit(gpu_standalone.returncode)
            _collect(cpu_solo_all, cpu_standalone.ms_by_op)
            _collect(gpu_solo_all, gpu_standalone.ms_by_op)

        if args.mode in ("concurrent", "both"):
            _print_header(f"Concurrent run (CPU + GPU at the same time) {rep_tag}")
            t0 = time.time()
            cpu_conc, gpu_conc = _run_concurrent("CPU (concurrent)", cpu_cmd, "GPU (concurrent)", gpu_cmd)
            t1 = time.time()
            print(f"elapsed={t1 - t0:.1f}s")
            if cpu_conc.returncode != 0:
                print("\n[CPU concurrent stderr]\n" + cpu_conc.stderr)
                raise SystemExit(cpu_conc.returncode)
            if gpu_conc.returncode != 0:
                print("\n[GPU concurrent stderr]\n" + gpu_conc.stderr)
                raise SystemExit(gpu_conc.returncode)
            _collect(cpu_conc_all, cpu_conc.ms_by_op)
            _collect(gpu_conc_all, gpu_conc.ms_by_op)

    # Report averaged results
    if args.mode in ("standalone", "both") and cpu_solo_all and gpu_solo_all:
        _print_header("CPU vs GPU latency gap (standalone, averaged)")
        all_ops = sorted(set(cpu_solo_all) | set(gpu_solo_all))
        print(f"{'Op':36s} {'CPU(ms)':>10s} {'GPU(ms)':>10s} {'CPU/GPU':>10s} {'CPUσ':>8s} {'GPUσ':>8s}")
        for op in all_ops:
            cpu_vals = cpu_solo_all.get(op)
            gpu_vals = gpu_solo_all.get(op)
            if not cpu_vals or not gpu_vals:
                continue
            cpu_m = _mean(cpu_vals)
            gpu_m = _mean(gpu_vals)
            ratio = cpu_m / gpu_m if gpu_m > 0 else float("inf")
            print(f"{op:36s} {cpu_m:10.3f} {gpu_m:10.3f} {ratio:10.1f}x {_stdev(cpu_vals):8.3f} {_stdev(gpu_vals):8.3f}")

    if args.mode == "both" and cpu_solo_all and gpu_solo_all and cpu_conc_all and gpu_conc_all:
        _print_header("Concurrency impact (concurrent vs standalone, averaged)")
        print("CPU side:")
        print(f"{'Op':36s} {'solo':>10s} {'conc':>10s} {'delta':>8s} {'soloσ':>8s} {'concσ':>8s}")
        for op in sorted(set(cpu_solo_all) & set(cpu_conc_all)):
            solo_vals = cpu_solo_all[op]
            conc_vals = cpu_conc_all[op]
            solo_m = _mean(solo_vals)
            conc_m = _mean(conc_vals)
            print(f"{op:36s} {solo_m:10.3f} {conc_m:10.3f} {_fmt_pct((conc_m - solo_m) / solo_m):>8s} {_stdev(solo_vals):8.3f} {_stdev(conc_vals):8.3f}")

        print("\nGPU side:")
        print(f"{'Op':36s} {'solo':>10s} {'conc':>10s} {'delta':>8s} {'soloσ':>8s} {'concσ':>8s}")
        for op in sorted(set(gpu_solo_all) & set(gpu_conc_all)):
            solo_vals = gpu_solo_all[op]
            conc_vals = gpu_conc_all[op]
            solo_m = _mean(solo_vals)
            conc_m = _mean(conc_vals)
            print(f"{op:36s} {solo_m:10.3f} {conc_m:10.3f} {_fmt_pct((conc_m - solo_m) / solo_m):>8s} {_stdev(solo_vals):8.3f} {_stdev(conc_vals):8.3f}")


if __name__ == "__main__":
    main()

