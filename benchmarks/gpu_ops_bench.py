#!/usr/bin/env python3
"""
GPU micro-benchmarks for common LLM inference ops.

This mirrors `benchmarks/cpu_ops_bench.py` but runs everything on CUDA and uses
CUDA events for timing.

Requirements:
- PyTorch with CUDA support
"""

from __future__ import annotations

import argparse
import math
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


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {s}")


def _bench_cuda_events(
    name: str,
    fn: Callable[[], torch.Tensor | None],
    iters: int,
    warmup: int,
) -> tuple[str, float]:
    # Warmup
    for _ in range(warmup):
        out = fn()
        if isinstance(out, torch.Tensor):
            # Prevent dead-code elimination and ensure kernels launch.
            _ = out.reshape(-1)[:1].float().sum()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    acc = 0.0
    for _ in range(iters):
        out = fn()
        if isinstance(out, torch.Tensor):
            acc += float(out.reshape(-1)[:1].float().sum().item()) if out.numel() else 0.0
    end.record()
    end.synchronize()
    _ = acc
    ms_total = start.elapsed_time(end)
    return name, ms_total / iters


def _make_causal_mask(seq: int, device: torch.device) -> torch.Tensor:
    # True where we should mask (upper triangle).
    return torch.triu(torch.ones((seq, seq), dtype=torch.bool, device=device), diagonal=1)


def _kv_block_table(batch: int, max_blocks_per_seq: int, kv_blocks: int, device: torch.device) -> torch.Tensor:
    # Keep deterministic ids generated on CPU, then move to GPU.
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    tbl = torch.randint(low=0, high=kv_blocks, size=(batch, max_blocks_per_seq), generator=g, device="cpu")
    return tbl.to(device=device, non_blocking=True)


def main() -> None:
    p = argparse.ArgumentParser(description="GPU micro-benchmarks for LLM ops")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--device", default="cuda", help="CUDA device string, e.g. cuda or cuda:0")

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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch and run on a machine with GPU.")

    dtype = _dtype_from_str(args.dtype)
    device = torch.device(args.device)
    # torch.cuda.set_device requires an explicit device index.
    if device.type != "cuda":
        raise ValueError(f"--device must be a CUDA device (got {device})")
    torch.cuda.set_device(device.index if device.index is not None else 0)

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

    assert shapes.hidden == shapes.num_heads * shapes.head_dim, (
        f"hidden({shapes.hidden}) must equal num_heads({shapes.num_heads})*head_dim({shapes.head_dim})"
    )

    # Optional: reduce allocator noise for benchmarks.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Common tensors
    embed_weight = torch.randn((shapes.vocab, shapes.hidden), device=device, dtype=dtype)
    input_ids = torch.randint(0, shapes.vocab, (shapes.batch, shapes.seq), device=device, dtype=torch.int64)

    x = torch.randn((shapes.batch, shapes.seq, shapes.hidden), device=device, dtype=dtype)
    y = torch.randn((shapes.batch, shapes.seq, shapes.hidden), device=device, dtype=dtype)

    # GEMM: [M, K] x [K, N]
    m = shapes.batch * shapes.seq
    k = shapes.hidden
    n = shapes.hidden
    a_gemm = torch.randn((m, k), device=device, dtype=dtype)
    w_gemm = torch.randn((k, n), device=device, dtype=dtype)
    bias = torch.randn((n,), device=device, dtype=dtype)

    # Attention shapes (prefill-style)
    bh = shapes.batch * shapes.num_heads
    q = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)
    k_t = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)
    v_t = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)
    causal_mask = _make_causal_mask(shapes.seq, device)

    # KV cache mock on GPU
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
    q_vec = torch.randn((bh, shapes.head_dim), device=device, dtype=dtype)
    k_mat = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)
    v_mat = torch.randn((bh, shapes.seq, shapes.head_dim), device=device, dtype=dtype)

    # Ops
    def op_embedding() -> torch.Tensor:
        return F.embedding(input_ids, embed_weight)

    def op_add() -> torch.Tensor:
        return x + y

    def op_reduce_mean() -> torch.Tensor:
        return x.mean(dim=1)

    def op_reduce_var() -> torch.Tensor:
        return x.var(dim=1, unbiased=False)

    def op_gemm_bias() -> torch.Tensor:
        return a_gemm @ w_gemm + bias

    def op_qk() -> torch.Tensor:
        return torch.matmul(q, k_t.transpose(-1, -2))

    def op_maskfill() -> torch.Tensor:
        scores = op_qk()
        return scores.masked_fill(causal_mask, float("-inf"))

    def op_softmax() -> torch.Tensor:
        scores = op_maskfill()
        return torch.softmax(scores, dim=-1)

    def op_attn_v() -> torch.Tensor:
        attn = op_softmax()
        return torch.matmul(attn, v_t)

    def op_split() -> torch.Tensor:
        a, b = torch.split(x, [shapes.hidden // 2, shapes.hidden - shapes.hidden // 2], dim=-1)
        return a + b

    def op_concat() -> torch.Tensor:
        a, b = torch.split(x, [shapes.hidden // 2, shapes.hidden - shapes.hidden // 2], dim=-1)
        return torch.cat([a, b], dim=-1)

    def op_reduce_max() -> torch.Tensor:
        return x.max(dim=1).values

    def op_reduce_sum() -> torch.Tensor:
        return x.sum(dim=1)

    def op_gemv_qk() -> torch.Tensor:
        return torch.einsum("bsd,bd->bs", k_mat, q_vec)

    def op_kv_write() -> torch.Tensor:
        blk_ids = block_table[:, -1]
        off = shapes.block_size - 1
        k_tok = torch.randn((shapes.batch, shapes.num_heads, shapes.head_dim), device=device, dtype=dtype)
        v_tok = torch.randn((shapes.batch, shapes.num_heads, shapes.head_dim), device=device, dtype=dtype)
        k_cache[blk_ids, off, :, :] = k_tok
        v_cache[blk_ids, off, :, :] = v_tok
        return k_tok

    def op_kv_read_paged_random() -> torch.Tensor:
        blk_ids = block_table
        gathered = k_cache.index_select(0, blk_ids.reshape(-1)).reshape(
            shapes.batch, args.max_blocks_per_seq, shapes.block_size, shapes.num_heads, shapes.head_dim
        )
        return gathered.sum(dim=(1, 2))

    def op_kv_read_paged_sorted() -> torch.Tensor:
        blk_ids = block_table_sorted
        gathered = k_cache.index_select(0, blk_ids.reshape(-1)).reshape(
            shapes.batch, args.max_blocks_per_seq, shapes.block_size, shapes.num_heads, shapes.head_dim
        )
        return gathered.sum(dim=(1, 2))

    def op_kv_read_contiguous() -> torch.Tensor:
        blk_ids = block_table_contiguous
        gathered = k_cache.index_select(0, blk_ids.reshape(-1)).reshape(
            shapes.batch, args.max_blocks_per_seq, shapes.block_size, shapes.num_heads, shapes.head_dim
        )
        return gathered.sum(dim=(1, 2))

    def op_softmax_decode() -> torch.Tensor:
        scores = op_gemv_qk() / math.sqrt(shapes.head_dim)
        return torch.softmax(scores, dim=-1)

    def op_gemv_attn_v() -> torch.Tensor:
        attn = op_softmax_decode()
        return torch.einsum("bs,bsd->bd", attn, v_mat)

    def op_gelu() -> torch.Tensor:
        return F.gelu(x)

    def op_exp_sum_div() -> torch.Tensor:
        t = x.max(dim=-1, keepdim=True).values
        e = torch.exp((x - t).to(torch.float32))
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

    props = torch.cuda.get_device_properties(device)
    print("== GPU Op Bench ==")
    print(f"device={device} name={props.name} sm={props.major}.{props.minor}")
    print(f"dtype={dtype} batch={shapes.batch} seq={shapes.seq} hidden={shapes.hidden} heads={shapes.num_heads} head_dim={shapes.head_dim}")
    print(f"kv_blocks={shapes.kv_blocks} block_size={shapes.block_size} max_blocks_per_seq={args.max_blocks_per_seq}")
    print(f"iters={args.iters} warmup={args.warmup}")
    print()

    for name, fn in ops:
        n, ms = _bench_cuda_events(name, fn, iters=args.iters, warmup=args.warmup)
        print(f"{n:38s}  {ms:10.3f} ms/iter")


if __name__ == "__main__":
    main()

