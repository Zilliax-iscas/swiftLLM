#!/usr/bin/env python3
"""
CPU micro-benchmarks for common LLM inference ops.

Goals:
- Measure *operator-level* latency on CPU (not end-to-end model).
- Keep dependencies minimal: only PyTorch (CPU).
- Provide knobs for shapes / dtype / threads.

Note:
- Results depend heavily on CPU model, memory bandwidth, thread affinity, and PyTorch build (MKL/oneDNN).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Shapes:
    batch: int
    seq: int
    hidden: int
    num_heads: int
    head_dim: int
    vocab: int
    inter: int
    kv_blocks: int
    block_size: int


def _set_threads(threads: int) -> None:
    if threads > 0:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 8)))


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {s}")


def _bench(
    name: str,
    fn: Callable[[], torch.Tensor | None],
    iters: int,
    warmup: int,
) -> tuple[str, float]:
    # Warmup
    for _ in range(warmup):
        out = fn()
        if isinstance(out, torch.Tensor):
            # Prevent dead-code elimination
            _ = out.sum().item() if out.numel() else 0.0

    # Timed loop (simple wall-time). We intentionally avoid torch.utils.benchmark
    # because users often want a single-file script with no extra output formatting.
    import time

    t0 = time.perf_counter()
    acc = 0.0
    for _ in range(iters):
        out = fn()
        if isinstance(out, torch.Tensor):
            acc += float(out.reshape(-1)[:1].float().sum().item()) if out.numel() else 0.0
    t1 = time.perf_counter()
    _ = acc  # keep alive
    ms_per_iter = (t1 - t0) * 1e3 / iters
    return name, ms_per_iter


def _make_causal_mask(seq: int, device: torch.device) -> torch.Tensor:
    # True where we should mask (upper triangle).
    return torch.triu(torch.ones((seq, seq), dtype=torch.bool, device=device), diagonal=1)


def _kv_block_table(batch: int, max_blocks_per_seq: int, kv_blocks: int, device: torch.device) -> torch.Tensor:
    # [batch, max_blocks_per_seq] with values in [0, kv_blocks)
    # This mimics a paged-attention block table.
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    tbl = torch.randint(low=0, high=kv_blocks, size=(batch, max_blocks_per_seq), generator=g, device="cpu")
    return tbl.to(device=device, non_blocking=False)


def main() -> None:
    p = argparse.ArgumentParser(description="CPU micro-benchmarks for LLM ops")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--threads", type=int, default=0, help="torch intraop threads (0 = leave default)")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)

    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--vocab", type=int, default=128_000)
    p.add_argument("--inter", type=int, default=11008, help="FFN intermediate dim (e.g. LLaMA-7B)")

    p.add_argument("--kv-blocks", type=int, default=4096, help="num KV blocks in cache")
    p.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    p.add_argument("--max-blocks-per-seq", type=int, default=256)

    args = p.parse_args()

    _set_threads(args.threads)
    dtype = _dtype_from_str(args.dtype)
    device = torch.device("cpu")

    shapes = Shapes(
        batch=args.batch,
        seq=args.seq,
        hidden=args.hidden,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        vocab=args.vocab,
        inter=args.inter,
        kv_blocks=args.kv_blocks,
        block_size=args.block_size,
    )

    # Basic sanity for typical LLM shapes.
    assert shapes.hidden == shapes.num_heads * shapes.head_dim, (
        f"hidden({shapes.hidden}) must equal num_heads({shapes.num_heads})*head_dim({shapes.head_dim})"
    )

    # Common tensors
    # Embedding lookup
    embed_weight = torch.randn((shapes.vocab, shapes.hidden), device=device, dtype=dtype)
    input_ids = torch.randint(0, shapes.vocab, (shapes.batch, shapes.seq), device=device, dtype=torch.int64)

    # Elementwise / reductions
    x = torch.randn((shapes.batch, shapes.seq, shapes.hidden), device=device, dtype=dtype)
    y = torch.randn((shapes.batch, shapes.seq, shapes.hidden), device=device, dtype=dtype)

    # GEMM: [M, K] x [K, N]
    m = shapes.batch * shapes.seq
    k = shapes.hidden
    n = shapes.hidden
    a_gemm = torch.randn((m, k), device=device, dtype=dtype)
    w_gemm = torch.randn((k, n), device=device, dtype=dtype)
    bias = torch.randn((n,), device=device, dtype=dtype)

    # Attention shapes
    # Prefill-style: (B*H, S, D)
    bh = shapes.batch * shapes.num_heads
    q = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)
    k_t = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)
    v_t = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)
    causal_mask = _make_causal_mask(shapes.seq, device)

    # KV cache (paged) mock:
    # Store K/V for each block: [kv_blocks, block_size, H, D]
    # (layout is arbitrary for benchmarking read/write cost)
    k_cache = torch.empty((shapes.kv_blocks, shapes.block_size, shapes.num_heads, shapes.head_dim), device=device, dtype=dtype)
    v_cache = torch.empty((shapes.kv_blocks, shapes.block_size, shapes.num_heads, shapes.head_dim), device=device, dtype=dtype)
    block_table = _kv_block_table(shapes.batch, args.max_blocks_per_seq, shapes.kv_blocks, device=device)
    # Version-2: same paged IDs but sorted by physical block id (better locality, not logical order).
    block_table_sorted = torch.sort(block_table, dim=1).values
    # Version-3: fully contiguous physical blocks for each sequence (best locality baseline).
    contiguous_base = torch.arange(args.max_blocks_per_seq, dtype=torch.int64, device=device).unsqueeze(0).repeat(shapes.batch, 1)
    contiguous_offsets = (torch.arange(shapes.batch, dtype=torch.int64, device=device).unsqueeze(1) * args.max_blocks_per_seq) % shapes.kv_blocks
    block_table_contiguous = ((contiguous_base + contiguous_offsets) % shapes.kv_blocks).to(torch.int64)

    # Decoding-style vectors
    q_vec = torch.randn((bh, shapes.head_dim), device=device, dtype=dtype)  # [B*H, D]
    # We'll treat "context length" as seq; in practice per-seq length differs.
    k_mat = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)  # [B*H, S, D]
    v_mat = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)  # [B*H, S, D]

    results: list[tuple[str, float]] = []

    # 1) EmbeddingLookup
    def op_embedding() -> torch.Tensor:
        return F.embedding(input_ids, embed_weight)  # [B,S,H]

    # 2) ElementWise Add
    def op_add() -> torch.Tensor:
        return x + y

    # 3) ReduceMean / ReduceVar (per hidden, over seq)
    def op_reduce_mean() -> torch.Tensor:
        return x.mean(dim=1)

    def op_reduce_var() -> torch.Tensor:
        # Unbiased=False matches common layernorm/variance usage
        return x.var(dim=1, unbiased=False)

    # 4) GEMM + bias add
    def op_gemm_bias() -> torch.Tensor:
        return a_gemm @ w_gemm + bias

    # 5) GEMM(QK) prefill-style: [B*H, S, D] x [B*H, D, S] -> [B*H, S, S]
    def op_qk() -> torch.Tensor:
        return torch.matmul(q, k_t.transpose(-1, -2))

    # 6) Causal MaskFill
    def op_maskfill() -> torch.Tensor:
        scores = op_qk()
        # broadcast mask to (B*H, S, S)
        return scores.masked_fill(causal_mask, float("-inf"))

    # 7) Softmax (attention)
    def op_softmax() -> torch.Tensor:
        scores = op_maskfill()
        return torch.softmax(scores, dim=-1)

    # 8) GEMM(Attn×V) prefill-style: [B*H, S, S] x [B*H, S, D] -> [B*H, S, D]
    def op_attn_v() -> torch.Tensor:
        attn = op_softmax()
        return torch.matmul(attn, v_t)

    # 9) Split / Concat
    def op_split() -> torch.Tensor:
        a, b = torch.split(x, [shapes.hidden // 2, shapes.hidden - shapes.hidden // 2], dim=-1)
        return a + b  # force materialization

    def op_concat() -> torch.Tensor:
        a, b = torch.split(x, [shapes.hidden // 2, shapes.hidden - shapes.hidden // 2], dim=-1)
        return torch.cat([a, b], dim=-1)

    # 10) ReduceMax / ReduceSum
    def op_reduce_max() -> torch.Tensor:
        return x.max(dim=1).values

    def op_reduce_sum() -> torch.Tensor:
        return x.sum(dim=1)

    # 11) GEMV (QK) decoding-style: [B*H, S, D] · [B*H, D] -> [B*H, S]
    def op_gemv_qk() -> torch.Tensor:
        # einsum is often a decent proxy for batched GEMV
        return torch.einsum("bsd,bd->bs", k_mat, q_vec)

    # 12) KV Cache read/write (paged-like)
    # Write: scatter a block's K/V (simulate appending the last token in each seq's last block)
    def op_kv_write() -> torch.Tensor:
        # Choose last block for each seq in the batch
        blk_ids = block_table[:, -1]  # [B]
        # token offset within block (use a fixed offset to avoid extra randomness)
        off = shapes.block_size - 1
        # produce per-seq K/V for all heads
        k_tok = torch.randn((shapes.batch, shapes.num_heads, shapes.head_dim), device=device, dtype=dtype)
        v_tok = torch.randn((shapes.batch, shapes.num_heads, shapes.head_dim), device=device, dtype=dtype)
        k_cache[blk_ids, off, :, :] = k_tok
        v_cache[blk_ids, off, :, :] = v_tok
        return k_tok  # keep alive

    # Read V1: paged random-like (original behavior)
    def op_kv_read_paged_random() -> torch.Tensor:
        blk_ids = block_table  # [B, max_blocks_per_seq]
        gathered = k_cache.index_select(0, blk_ids.reshape(-1)).reshape(
            shapes.batch, args.max_blocks_per_seq, shapes.block_size, shapes.num_heads, shapes.head_dim
        )
        # force some reduction to keep tensor live
        return gathered.sum(dim=(1, 2))

    # Read V2: paged but physical block IDs are sorted (improves locality)
    def op_kv_read_paged_sorted() -> torch.Tensor:
        blk_ids = block_table_sorted
        gathered = k_cache.index_select(0, blk_ids.reshape(-1)).reshape(
            shapes.batch, args.max_blocks_per_seq, shapes.block_size, shapes.num_heads, shapes.head_dim
        )
        return gathered.sum(dim=(1, 2))

    # Read V3: fully contiguous physical blocks (best-case locality baseline)
    def op_kv_read_contiguous() -> torch.Tensor:
        blk_ids = block_table_contiguous
        gathered = k_cache.index_select(0, blk_ids.reshape(-1)).reshape(
            shapes.batch, args.max_blocks_per_seq, shapes.block_size, shapes.num_heads, shapes.head_dim
        )
        return gathered.sum(dim=(1, 2))

    # 13) Softmax + GEMV(Attn×V) decoding-style
    def op_softmax_decode() -> torch.Tensor:
        scores = op_gemv_qk() / math.sqrt(shapes.head_dim)  # [B*H, S]
        return torch.softmax(scores, dim=-1)

    def op_gemv_attn_v() -> torch.Tensor:
        attn = op_softmax_decode()  # [B*H, S]
        return torch.einsum("bs,bsd->bd", attn, v_mat)

    # 14) GELU (ElementWise) + exp/div style reductions
    def op_gelu() -> torch.Tensor:
        return F.gelu(x)

    def op_exp_sum_div() -> torch.Tensor:
        t = x.max(dim=-1, keepdim=True).values
        e = torch.exp((x - t).to(torch.float32))  # exp in fp32 is common
        s = e.sum(dim=-1, keepdim=True)
        return (e / s).to(dtype)

    ops: Iterable[tuple[str, Callable[[], torch.Tensor | None]]] = [
        ("EmbeddingLookup", op_embedding),
        ("ElementWiseAdd", op_add),
        ("ReduceMean(dim=seq)", op_reduce_mean),
        ("ReduceVar(dim=seq)", op_reduce_var),
        ("GEMM+BiasAdd", op_gemm_bias),
        ("GEMM(QK) [prefill]", op_qk),
        ("CausalMaskFill", op_maskfill),
        ("Softmax [prefill]", op_softmax),
        ("GEMM(Attn×V) [prefill]", op_attn_v),
        ("Split", op_split),
        ("Concat", op_concat),
        ("ReduceMax(dim=seq)", op_reduce_max),
        ("ReduceSum(dim=seq)", op_reduce_sum),
        ("GEMV(QK) [decode]", op_gemv_qk),
        ("KVCacheWrite(paged)", op_kv_write),
        ("KVCacheRead(paged-random)", op_kv_read_paged_random),
        ("KVCacheRead(paged-sorted-physical)", op_kv_read_paged_sorted),
        ("KVCacheRead(contiguous-physical)", op_kv_read_contiguous),
        ("Softmax [decode]", op_softmax_decode),
        ("GEMV(Attn×V) [decode]", op_gemv_attn_v),
        ("GELU(ElementWise)", op_gelu),
        ("ReduceMax+Exp+ReduceSum+Div(softmax core)", op_exp_sum_div),
    ]

    print("== CPU Op Bench ==")
    print(f"threads={torch.get_num_threads()} dtype={dtype} batch={shapes.batch} seq={shapes.seq} hidden={shapes.hidden} heads={shapes.num_heads} head_dim={shapes.head_dim}")
    print(f"kv_blocks={shapes.kv_blocks} block_size={shapes.block_size} max_blocks_per_seq={args.max_blocks_per_seq}")
    print(f"iters={args.iters} warmup={args.warmup}")
    if os.environ.get("OMP_NUM_THREADS"):
        print(f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    if os.environ.get("MKL_NUM_THREADS"):
        print(f"MKL_NUM_THREADS={os.environ['MKL_NUM_THREADS']}")
    print()

    for name, fn in ops:
        n, ms = _bench(name, fn, iters=args.iters, warmup=args.warmup)
        results.append((n, ms))
        print(f"{n:38s}  {ms:10.3f} ms/iter")


if __name__ == "__main__":
    main()

