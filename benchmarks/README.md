# CPU 微基准测试

该目录包含常见 LLM 推理算子的 CPU 算子级基准测试（如 embedding、GEMM/GEMV、softmax、KV cache 读写等）。

KV cache 读取目前包含三种变体：
- `KVCacheRead(paged-random)`：分页 gather，物理 block id 未排序
- `KVCacheRead(paged-sorted-physical)`：同一组 id，但按物理 block id 排序（局部性更好）
- `KVCacheRead(contiguous-physical)`：物理 block 连续（最佳局部性基线）

## 运行方式

在仓库根目录（`sllm/swiftLLM`）执行：

```bash
python3 benchmarks/cpu_ops_bench.py --help
python3 benchmarks/cpu_ops_bench.py --dtype bf16 --threads 32 --batch 8 --seq 1024 --hidden 4096 --num-heads 32 --head-dim 128
```

### GPU 版本

```bash
python3 benchmarks/gpu_ops_bench.py --help
python3 benchmarks/gpu_ops_bench.py --dtype bf16 --batch 8 --seq 1024 --hidden 4096 --num-heads 32 --head-dim 128
```

### 联合运行 CPU + GPU 并对比（独立运行 vs 并发运行）

请使用同一个同时安装了 CPU/GPU 版 PyTorch 的 Python 环境：

```bash
python3 benchmarks/compare_cpu_gpu.py --help
python3 benchmarks/compare_cpu_gpu.py --dtype bf16 --threads 32 --batch 8 --seq 1024 --hidden 4096 --num-heads 32 --head-dim 128
```

### 对比 GPU CUDA 算子与 pacpu SIMD 算子

如果你希望 CPU 侧使用 pacpu 的 SIMD 内核（`torch.ops.pacpu.paged_attention_cpu`），而不是 PyTorch CPU 算子：

```bash
python3 benchmarks/compare_cpu_gpu.py \
  --cpu-backend pacpu \
  --pacpu-library-path /path/to/libpacpu-llama2_7b-tp1.so \
  --dtype fp16 \
  --threads 32 \
  --batch 8 --seq 1024 --num-heads 32 --head-dim 128 \
  --pacpu-num-kv-heads 32 --pacpu-num-layers 32
```

你也可以直接运行 pacpu 专用基准：

```bash
python3 benchmarks/pacpu_ops_bench.py --help
```

多次重复并输出平均结果：

```bash
python3 benchmarks/compare_cpu_gpu.py --repeats 5 --dtype bf16 --threads 32 --batch 8 --seq 1024 --hidden 4096 --num-heads 32 --head-dim 128
```

### 说明

- 为了结果更稳定，建议通过 `--threads` 以及（可选）环境变量固定线程数：
  - `OMP_NUM_THREADS`、`MKL_NUM_THREADS`
- `--dtype fp16` 在 CPU 上可能较慢（甚至被模拟实现），具体取决于 CPU 与 PyTorch 构建；若支持，通常更推荐 `bf16`。



