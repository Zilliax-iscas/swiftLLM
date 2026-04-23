# CPU micro-benchmarks

This folder contains operator-level CPU benchmarks for common LLM inference ops (embedding, GEMM/GEMV, softmax, KV cache read/write, etc.).

KV-cache read now reports three variants:
- `KVCacheRead(paged-random)`: paged gather with unsorted physical block ids
- `KVCacheRead(paged-sorted-physical)`: same ids but sorted by physical id (better locality)
- `KVCacheRead(contiguous-physical)`: contiguous physical blocks (best-case locality baseline)

## Run

From repo root (`sllm/swiftLLM`):

```bash
python3 benchmarks/cpu_ops_bench.py --help
python3 benchmarks/cpu_ops_bench.py --dtype bf16 --threads 32 --batch 8 --seq 1024 --hidden 4096 --num-heads 32 --head-dim 128
```

### GPU version

```bash
python3 benchmarks/gpu_ops_bench.py --help
python3 benchmarks/gpu_ops_bench.py --dtype bf16 --batch 8 --seq 1024 --hidden 4096 --num-heads 32 --head-dim 128
```

### Run CPU + GPU and compare (standalone vs concurrent)

Run with the same Python environment that has both CPU/GPU PyTorch:

```bash
python3 benchmarks/compare_cpu_gpu.py --help
python3 benchmarks/compare_cpu_gpu.py --dtype bf16 --threads 32 --batch 8 --seq 1024 --hidden 4096 --num-heads 32 --head-dim 128
```

Repeat multiple times and report averages:

```bash
python3 benchmarks/compare_cpu_gpu.py --repeats 5 --dtype bf16 --threads 32 --batch 8 --seq 1024 --hidden 4096 --num-heads 32 --head-dim 128
```

### Notes

- For more stable results, pin thread counts via `--threads` and (optionally) environment variables:
  - `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
- `--dtype fp16` on CPU may be slow (or emulated) depending on your CPU/PyTorch build; `bf16` is usually preferred if supported.



