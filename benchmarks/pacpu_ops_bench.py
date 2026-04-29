#!/usr/bin/env python3
"""
Benchmark pacpu SIMD attention operator used by NEO:
  torch.ops.pacpu.paged_attention_cpu(...)

This script focuses on operators that actually exist in pacpu. Today that is
the fused paged attention decode entrypoint.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List

import torch


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp16", "float16", "bf16", "bfloat16"):
        # pacpu currently expects at::Half inputs for q/k/v and caches.
        return torch.float16
    if s in ("fp32", "float32"):
        # Keep interface lenient; compute still uses fp16 inputs in pacpu.
        return torch.float16
    raise ValueError(f"Unsupported dtype: {s}")


def _bench(
    fn,
    iters: int,
    warmup: int,
) -> float:
    for _ in range(warmup):
        out = fn()
        _ = float(out.reshape(-1)[:1].sum().item()) if out.numel() else 0.0

    t0 = time.perf_counter()
    acc = 0.0
    for _ in range(iters):
        out = fn()
        acc += float(out.reshape(-1)[:1].sum().item()) if out.numel() else 0.0
    t1 = time.perf_counter()
    _ = acc
    return (t1 - t0) * 1e3 / iters


def _build_block_table(
    max_seqs: int,
    max_blocks_per_seq: int,
    kv_blocks: int,
) -> torch.Tensor:
    # Deterministic logical->physical mapping.
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    return torch.randint(
        low=0,
        high=kv_blocks,
        size=(max_seqs, max_blocks_per_seq),
        generator=g,
        dtype=torch.int32,
        device="cpu",
    )


def _build_seq_lengths(batch: int, seq: int, block_size: int) -> List[int]:
    # Keep lengths block-aligned for stable behavior and easy control.
    seq = max(block_size, (seq // block_size) * block_size)
    return [seq] * batch


def main() -> None:
    p = argparse.ArgumentParser(description="pacpu SIMD op benchmark")
    p.add_argument("--library-path", required=True, help="Path to libpacpu-*.so")
    p.add_argument("--threads", type=int, default=0, help="torch intraop threads (0=leave default)")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])

    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--num-heads", type=int, default=32, help="Q heads")
    p.add_argument("--num-kv-heads", type=int, default=32, help="KV heads")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--cur-layer", type=int, default=0)
    p.add_argument("--softmax-scale", type=float, default=None, help="Default: 1/sqrt(head_dim)")

    p.add_argument("--kv-blocks", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-seqs-in-block-table", type=int, default=4096)
    p.add_argument("--max-blocks-per-seq", type=int, default=256)
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(args.threads, 8)))

    # Load extension once.
    torch.ops.load_library(args.library_path)
    dtype = _dtype_from_str(args.dtype)

    if args.softmax_scale is None:
        args.softmax_scale = 1.0 / math.sqrt(args.head_dim)

    assert args.batch <= args.max_seqs_in_block_table, "batch must be <= max-seqs-in-block-table"

    seq_ids = list(range(args.batch))
    seq_lens = _build_seq_lengths(args.batch, args.seq, args.block_size)

    q = torch.randn((args.batch, args.num_heads, args.head_dim), dtype=dtype, device="cpu")
    k = torch.randn((args.batch, args.num_kv_heads, args.head_dim), dtype=dtype, device="cpu")
    v = torch.randn((args.batch, args.num_kv_heads, args.head_dim), dtype=dtype, device="cpu")

    k_cache = torch.zeros(
        (args.num_layers, args.kv_blocks, args.num_kv_heads, args.block_size, args.head_dim),
        dtype=dtype,
        device="cpu",
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = _build_block_table(
        args.max_seqs_in_block_table,
        args.max_blocks_per_seq,
        args.kv_blocks,
    )
    o = torch.zeros((args.batch, args.num_heads, args.head_dim), dtype=torch.float32, device="cpu")

    def op_pacpu_paged_attention() -> torch.Tensor:
        torch.ops.pacpu.paged_attention_cpu(
            args.cur_layer,
            args.softmax_scale,
            seq_ids,
            seq_lens,
            q,
            k,
            v,
            k_cache,
            v_cache,
            block_table,
            o,
        )
        return o

    print("== pacpu SIMD Op Bench ==")
    print(f"library={args.library_path}")
    print(
        "shape: "
        f"B={args.batch} seq={seq_lens[0]} q_heads={args.num_heads} kv_heads={args.num_kv_heads} "
        f"head_dim={args.head_dim} layers={args.num_layers} block_size={args.block_size}"
    )
    print(f"threads={torch.get_num_threads()} iters={args.iters} warmup={args.warmup}")
    print()

    ms = _bench(op_pacpu_paged_attention, args.iters, args.warmup)
    print(f"{'PagedAttentionCPU(pacpu-SIMD)':38s}  {ms:10.3f} ms/iter")


if __name__ == "__main__":
    main()
