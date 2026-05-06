# SwiftLLM

*该项目仍在开发中。部分功能可能尚未实现，文档也可能不完整。*

一个小巧但强大的、面向研究用途的 LLM 推理系统。

仅用 2k 行代码（约为 vLLM 的 2%），即可实现与 vLLM 相当的性能。

## 为什么选择 SwiftLLM

开源的 LLM Serving 框架有很多，包括 [HuggingFace Transformers](https://github.com/huggingface/transformers)、[vLLM](https://github.com/vllm-project/vllm)、[LightLLM](https://github.com/ModelTC/lightllm)、[DistServe](https://github.com/LLMServe/DistServe) 和 [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)。为什么还需要 SwiftLLM？

原因在于，这些框架主要是为**生产环境**设计的，而不是为**研究**设计的。它们具备大量功能，例如支持 100+ 模型、各种硬件支持、LoRA、量化、多模态、前缀缓存、beam search 等等。虽然它们作为面向生产的一体化方案非常强大，但它们的代码库过于庞大且复杂，难以理解和修改（例如 vLLM 拥有超过 10 万行代码），这使得它们并不适合研究用途。此外，它们的历史包袱也是一个问题。

SwiftLLM 被设计成一个小巧但强大的 LLM 推理系统，专门面向**研究用途**。“小巧”意味着它只保留研究所必需的功能；“强大”意味着它在性能上不做妥协；而“Swift”则意味着它易于理解和修改。SwiftLLM 在支持基础功能（见下方列表）并能达到与 vLLM 相当性能的同时，整个代码库只有不到 **2k** 行代码（约为 vLLM 的 2%），使用 Python 和 [OpenAI Triton](https://github.com/openai/triton)（一种用于编写 CUDA kernel 的 DSL）编写，易于阅读、修改、调试、测试和扩展，也能够方便地与你新颖而出色的研究想法结合。

## 功能列表

目前，SwiftLLM 支持以下功能：

- 迭代式调度（Iterational Scheduling）和选择性批处理（Selective Batching）（由 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) 提出）
- PagedAttention（由 [vLLM](https://github.com/vllm-project/vllm) 提出，论文见 [paper](https://arxiv.org/abs/2309.06180)）
- LLaMA / LLaMA2 / LLaMA3 模型（[链接](https://llama.meta.com/)）及其变体
- 将 prefill 与 decoding 组合执行（Piggybacking）（由 [SARATHI](https://arxiv.org/abs/2308.16369) 提出）
- Flash Attention（由 [FlashAttention](https://arxiv.org/abs/2205.14135) 和 [FlashAttention-2](https://arxiv.org/abs/2307.08691) 提出）
- Paged Attention v2（也称 Flash-Decoding，介绍见 [这里](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)）

未来我们计划支持以下功能：

- 张量并行（Tensor Parallelism）和流水线并行（Pipeline Parallelism）

为了让代码库保持小巧，我们不会支持以下功能。如果你想在研究项目中使用它们，可能需要自行实现：

- 量化（Quantization）
- LoRA
- 多模态
- 不遵循 LLaMA 架构的模型
- 贪心采样以外的采样方法
- NVIDIA GPU 以外的硬件支持（不过只要 OpenAI Triton 支持，迁移到其他硬件应该并不困难）

请记住，SwiftLLM **不是** 面向生产的一体化方案。更适合把它看作你研究项目的“基础设施”，你可能仍需要自己实现一些功能。我们鼓励你，亲爱的研究者，去阅读代码、理解代码、修改代码，并按照你的研究需求进行扩展。

## 架构

SwiftLLM 的架构可以分为两个主要部分：*控制平面*（control plane）和 *数据平面*（data plane）。

简单来说，*控制平面*决定“算什么”以及“如何调度”，而 *数据平面*决定“如何计算”以及“如何实现”，并执行具体的计算过程。它们以 master-worker 的方式协同工作：控制平面像 master，负责高层调度与协调，并向数据平面下发任务；数据平面像 worker，负责底层计算。

控制平面的代码位于 `swiftllm/server` 目录中，包括 `Engine`、`Scheduler`、API server 和 `TokenizationEngine` 等组件。数据平面的代码位于 `swiftllm/worker` 目录中，包括计算图描述（位于 `swiftllm/worker/model.py`）、模型各层的实现（位于 `swiftllm/layers`），以及 OpenAI Triton kernel（你可以把 “kernel” 理解为在 GPU 上执行的函数，位于 `swiftllm/kernels`）。

我们以一个简化版 API server（位于 `swiftllm/server/api_server.py`）为例：

- 启动时，它会使用一个 `EngineConfig` 来创建 `Engine`。
- 随后通过 `Engine.initialize` 初始化引擎，其中会创建 `Scheduler`、`TokenizationEngine` 和一组 worker（目前由于尚不支持 Tensor Parallelism，因此只有一个 worker）。然后它会命令 worker 执行 `profile_num_blocks` 来计算 GPU block 数量，之后引擎会命令所有 worker 分配各自的 KV cache 和 KV swap。
- 最后，通过 `Engine.start_all_event_loops` 启动事件循环。在循环的每一步中，引擎会向调度器查询下一批需要计算的请求，命令 worker 执行 swap in/out，然后将该批请求发送给 worker 进行计算。
- API server 负责监听用户请求，并与引擎交互以完成这些请求。

目前控制平面（`Engine`）和数据平面（`LlamaModel`）都驻留在同一个节点上。在实现 Tensor Parallelism / Pipeline Parallelism 之后，数据平面可能会分布到多个节点上。

## 如何使用

我们提供两种使用 SwiftLLM 的方式：同时使用控制平面和数据平面，或者仅使用数据平面。

如果你的想法足够简单或足够优雅，能够无缝集成到现有控制平面中，那么你可以同时使用控制平面和数据平面。另一种情况下，如果你想实现一个非常有意思的新思路，也可以只复用数据平面，并自行实现一个新的控制平面。

## 构建与运行

首先配置环境：

- 你可以从一个全新的 conda 环境开始，要求 Python >= 3.9；也可以使用已有环境。如果你使用 conda，请不要忘记激活环境。
- 安装 [PyTorch](https://pytorch.org/)。请根据你的硬件选择正确的版本。
- 通过 `pip install packaging` 安装 `packaging`

然后开始安装：

- 使用 `git clone https://github.com/interestingLSY/swiftLLM.git` 克隆仓库
- 进入仓库目录（`cd swiftLLM`），并通过 `pip install -r requirements.txt` 安装其他依赖
- PyTorch 可能会顺带为你安装一个稳定版的 [OpenAI Triton](https://github.com/triton-lang/triton)。如果你想使用 nightly 版本以获得最前沿的性能、并接受潜在问题，可以卸载后改装 nightly 版本
- 运行 `pip install -e .`，将 SwiftLLM 安装到当前环境中
- 通过 `pip install -e csrc` 安装一些 C 绑定

下面是一些示例：

- 目前 SwiftLLM 还不支持自动从 HuggingFace 下载权重。你可能需要先从 HuggingFace 克隆或下载模型权重。支持 `.bin` 和 `.safetensors` 两种格式。假设你的模型权重存放在 `/data/to/weight/`
- 如果你想看一个离线推理示例，可以尝试 `python3 examples/offline.py --model-path /data/to/weight`。这个示例只使用数据平面。如果你打算在不使用控制平面的情况下使用 SwiftLLM，这是一个很好的起点
- 如果你想看一个使用 `Engine` 的在线服务示例，可以尝试 `python3 examples/online.py --model-path /data/to/weight`。如果你计划同时使用控制平面和数据平面，这是一个非常好的示例
- 如果你想看一个更复杂的示例，可以参考 `swiftllm/server/api_server.py`。它会启动一个 API server，并提供类似 vLLM 的在线服务接口

## 性能

尽管体积很小（小巧的东西也可以很可爱！），SwiftLLM 在性能上并没有妥协。我们在多个场景下对 SwiftLLM 进行了评估，结果表明，与 vLLM 相比，SwiftLLM 可以达到等效性能，甚至在某些情况下表现更好。

### 单次 Forward 操作

第一个场景是“单次 forward 操作”：我们向模型输入一批数据，并让它生成一个输出 token（等价于一次 “forward” 操作）。这是 LLM 推理（无论在线还是离线）的基础操作，因此其性能至关重要。

这里我们使用的是 FP16 精度下的 LLaMA-3 7B 模型，运行在 NVIDIA A100 80G PCIE / RTX 4090 GPU 上。结果如下（越低越好）：

![offline-llama-3-7b-a100](https://raw.githubusercontent.com/interestingLSY/swiftLLM/master/docs/assets/offline-llama-3-7b-a100.png)

![offline-llama-3-7b-4090](https://raw.githubusercontent.com/interestingLSY/swiftLLM/master/docs/assets/offline-llama-3-7b-4090.png)

可以看到，在相同设置下，SwiftLLM 能够达到与 vLLM 相当的性能，甚至在部分情况下更优。

### 在线服务

第二个场景是“在线服务”：我们启动一个 API server，从真实世界数据集中采样 prompt，并让模型生成补全结果。这是 LLM 在聊天机器人、代码补全等真实应用中的典型使用场景。

这里我们使用 [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) 数据集来采样 prompt，并使用不同 lambda 的泊松过程来模拟不同的请求到达速率。结果如下（越低越好）：

![online-llama-3-7b-a100](https://raw.githubusercontent.com/interestingLSY/swiftLLM/master/docs/assets/online-llama-3-7b-a100.png)

![online-llama-3-7b-4090](https://raw.githubusercontent.com/interestingLSY/swiftLLM/master/docs/assets/online-llama-3-7b-4090.png)

可以看到，在 A100 80G PCIE 上，SwiftLLM 能达到与 vLLM 相当的性能；而在 RTX 4090 上，SwiftLLM 明显优于 vLLM（主要原因是我们的控制平面开销更低）。
