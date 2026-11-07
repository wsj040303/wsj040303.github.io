# 流水线并行：把层分段，让 micro-batch 流动

张量并行切的是“一层之内”，流水线并行切的是“层与层之间”。它把模型的连续若干层放进一个 stage，每个 stage 放在一张卡或一组卡上；一个 batch 从 stage 0 流向 stage p-1，反向再流回来。如果一次只送一个 batch，任意时刻只有一个 stage 在计算，其余都在等——设备利用率只有 `1/p`。流水线并行要解决的核心问题，就是如何用 micro-batch 把这条流水线填满，以及填不满的部分（气泡）要付多少代价。

## 朴素模型并行：为什么需要 micro-batch

设模型有 `L` 层，切成 `p` 个 stage，每个 stage 负责 `L/p` 层。一个 mini-batch 的 forward 必须依次经过 stage 0、1、…、p-1，backward 再沿反方向传回。下面是不切 micro-batch 的时序：

```text
stage 0: [F............][B............]
stage 1:        [F............][B............]
stage 2:               [F............][B............]
stage 3:                      [F............][B............]
```

同一时刻只有一个 stage 在工作，其余 `p-1` 个 stage 空闲。这就是最原始的气泡（bubble）。把 mini-batch 切成 `m` 个 micro-batch 后，stage 0 算完第一个 micro-batch 就把激活传给 stage 1，自己继续算第二个；stage 1 收到后立刻开始算，流水线就被填起来了。

![GPipe 与 1F1B 的流水线时序：GPipe 先跑完所有 micro-batch 的前向再统一反向，1F1B 在稳态下一前一后交替，气泡相同但激活值显存更低。](images/pp-schedule.svg)

## GPipe：先全部前向，再全部反向

GPipe 的调度最直观：把 `m` 个 micro-batch 的 forward 全部跑完，再统一执行全部 backward。

```text
p=4, m=4   F_i = 第 i 个 micro-batch 的 forward，B_i = backward
stage 0: F1 F2 F3 F4 .  .  .  B4 B3 B2 B1
stage 1: .  F1 F2 F3 F4 .  .  B4 B3 B2 B1
stage 2: .  .  F1 F2 F3 F4 .  B4 B3 B2 B1
stage 3: .  .  .  F1 F2 F3 F4 B4 B3 B2 B1
                  ↑ 全部 forward 完成
```

气泡来自流水线头部填充和尾部排空：第一个 micro-batch 要经过 `p-1` 步才能填满流水线，最后一个 micro-batch 也要经过 `p-1` 步排空。设一个 micro-batch 的 forward 加 backward 时间为 `t_f + t_b`，理想总时间为 `m(t_f + t_b)`，气泡占比可以写成：

```text
bubble / total = (p - 1) / (m + p - 1) ≈ (p - 1) / m
```

这个式子说明两件事：`p` 越大，气泡越大；`m` 越大，气泡越小，但收益递减。`p=4, m=4` 时气泡约 43%；`p=8, m=32` 时约 18%；`p=16, m=64` 时仍有约 19%。所以 PP 通常要配一个较大的 global batch，让 `m` 远大于 `p`。[LLM Infra Atlas · PP](https://llm-infra-atlas.github.io/parallel/03_pp/)

GPipe 更严重的问题是激活值显存：backward 依赖 forward 的中间结果，而 GPipe 在做完所有 forward 之后才开始 backward，所以 `m` 个 micro-batch 的激活必须同时驻留，显存是 `O(m)`。想减小气泡就要增大 `m`，但增大 `m` 又会让显存上涨，这是 GPipe 的直接矛盾。

## 1F1B：尽早反向，尽早释放

PipeDream-Flush（通常叫 1F1B）改变了调度：一个 micro-batch 完成 forward 后，不必等其它 micro-batch，尽快执行它的 backward，然后立即释放它的激活。稳态下每个 stage 严格按“一次 forward、一次 backward”交替：

```text
stage 0: F1 F2 F3 F4 B1 B2 B3 B4
stage 1: .  F1 F2 F3 F4 B1 B2 B3 B4
stage 2: .  .  F1 F2 F3 F4 B1 B2 B3 B4
stage 3: .  .  .  F1 F2 F3 F4 B1 B2 B3 B4
```

气泡与 GPipe 相同，仍是 `(p-1)/(m+p-1)`，但每张卡同时只需要保存约 `p` 个 micro-batch 的激活，而不是 `m` 个。这是 PP 能用于大模型训练的前提：气泡用 `m` 压，显存用调度压，两件事被解耦了。Megatron-LM 的默认流水线调度就是 1F1B。

## 交错式流水线：用更多通信换更小气泡

交错式（interleaved，也叫 virtual pipeline）把每个物理 stage 再切成 `v` 个虚拟 stage，每个物理 rank 持有 `v` 段不连续的层。例如 8 张卡、`v=2`，rank 0 可能持有第 1–5 层和第 41–45 层，rank 1 持有第 6–10 层和第 46–50 层。调度器看到的是 16 个虚拟 stage，而不是 8 个。

气泡公式变成：

```text
bubble / total = (p - 1) / (v × m + p - 1) ≈ (p - 1) / (v × m)
```

`p=8, v=2, m=32` 时，气泡从 18% 降到约 10%。代价是每个 micro-batch 在 stage 之间多走 `v` 次，P2P 通信次数和调度复杂度都上升。Megatron 用 `--num-layers-per-virtual-pipeline-stage` 或 `--virtual-pipeline-model-parallel-size` 控制这个参数。[Megatron Core · Pipeline Parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/pipeline_parallel.html)

## DualPipe 与 zero-bubble

DeepSeek-V3 的 DualPipe 是另一条路线：它把 forward 和 backward 拆成更细的块，让两个方向的流水线同时流动，并把 MoE 的 All-to-All 通信塞进计算空隙里。官方说法是 DualPipe 比现有 PP 方法气泡更少，而且能重叠前向/反向的计算与通信，用来掩盖跨节点专家并行的开销。[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)

更一般地，zero-bubble pipeline 的思路是把 backward 拆成“计算权重的部分”和“计算输入梯度的部分”，把不依赖后续 stage 的部分提前执行，从而填掉气泡。这些调度都在同一个权衡空间里：气泡、激活值显存、通信次数三者不可能同时最优。

## PP 的通信与放置

PP 的通信只发生在 stage 边界：前向传激活，反向传梯度，都是相邻 rank 之间的 P2P Send/Recv。单个 micro-batch 的激活大小是 `micro_batch × seq × hidden × 字节数`，与 TP 每层 All-Reduce 的累计量相比小得多。

这就是“TP 放机内、PP 放机间”的由来。NVIDIA 在 MT-NLG 530B 的系统设计里明确写道：TP 通信开销最大，优先放节点内；PP 通信量最小，可以跨节点；DP 尽量放节点内或相邻节点。[Megatron-Turing NLG 530B](https://arxiv.org/html/2201.11990v3)

PP 的另一个优势是显存按 stage 均分。一个 `L` 层的模型切成 `p` 段，每段只保存 `L/p` 层的权重、梯度和优化器状态；如果再把每段内部用 TP 切开，就是“PP 管深度、TP 管宽度”的二维组合。

## 工程约束

PP 看起来简单，落地时有几处容易踩坑：

- **stage 负载不均**：`L` 不能被 `p` 整除时，有的 stage 多一层，气泡和显存都会偏向它。Embedding 和 LM Head 通常放在首尾 stage，也会造成额外不平衡。
- **micro-batch 数**：`m` 必须能整除 global batch，且通常要远大于 `p`。`m` 太小，气泡会吃掉大部分收益；`m` 太大，激活值和调度开销上升。
- **tied embedding**：输入 embedding 和输出 LM Head 共享权重时，首尾 stage 需要特殊处理，否则参数同步会出错。
- **PP 与 DP 的交互**：PP 组内的 stage 必须按顺序通信，DP 组则在每个 step 末尾同步梯度；两组通信可能争用同一张网卡，需要靠拓扑映射和通信重叠来缓解。
- **推理中的 PP**：PP 能跨节点放下大模型，但每个 token 要经过所有 stage，尾延迟增加。在线服务通常更倾向 TP/EP 或 PD 分离，PP 多用于离线吞吐或多机部署。

## 什么时候用 PP

适合 PP 的信号：

- 模型很深（50 层以上），单机 TP 后仍放不下；
- 集群跨节点，TP 已经受限于 NVLink 域；
- 能接受较大的 global batch，愿意用 `m` 压气泡；
- 层结构规整，stage 容易均分。

不适合 PP 的信号：

- 模型能靠 TP 或 ZeRO 放进单机；
- batch 很小、延迟敏感，气泡和 stage 间串行会成为瓶颈；
- 层数不能被 `p` 整除，且没有合适的虚拟 stage 方案；
- 训练框架不支持 1F1B 或交错调度，只能跑朴素 PP。

## 参考资料

- Huang et al.，[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- Narayanan et al.，[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/html/2104.04473v5)
- NVIDIA Megatron Core：[Pipeline Parallel package](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/pipeline_parallel.html)
- NVIDIA Megatron Core：[Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- LLM Infra Atlas：[Pipeline Parallelism（PP）](https://llm-infra-atlas.github.io/parallel/03_pp/)
- LLM Infra Atlas：[GPipe / 1F1B](https://llm-infra-atlas.github.io/parallel/03_pp/01_gpipe_1f1b/)
- 知乎：[论文解读系列第四篇：谷歌 GPipe 训练超大规模神经网络](https://zhuanlan.zhihu.com/p/113233933)
- 技术栈：[流水线并行 PP（NaivePP/GPipe/F-then-B/PipeDream/1F1B）](https://jishuzhan.net/article/1940984006488666114)
