# 数据并行：模型复制、梯度同步与状态分片

数据并行（Data Parallelism，DP）不切分模型，只切分训练数据。每张 GPU 保留一份完整模型副本，各自处理不同的 batch 分片，反向传播后同步梯度，使各副本在参数更新后保持一致。本文说明 DDP 的梯度同步过程、训练状态的显存构成、ZeRO 的三个分片阶段和 FSDP 的实现关系，并给出 7B 模型的显存计算示例。

## 问题定位

数据并行的适用前提是单卡能够容纳完整模型状态和一份 micro-batch 的激活值。满足该前提时，DP 的计算效率最高：每张卡执行完整的矩阵乘法，不需要为切分后的张量付出额外同步；每个训练 step 只在末尾同步一次梯度。

模型放不下时，DP 本身无法解决显存问题，因为它复制了全部参数、梯度和优化器状态。ZeRO 和 FSDP 在保持数据并行计算结构的同时，把这些状态分片到各卡，用通信换取显存。理解这三者的关系，需要先分别计算状态占用和通信量。

## DDP 的基本模型

设数据并行组有 `N` 张 GPU，全局 batch 被分成 `N` 份。每张 GPU 用本地 batch 执行一次前向和反向，得到本地梯度。Distributed Data Parallel（DDP）在每个 step 末尾对所有卡的梯度执行全规约（All-Reduce），使每张卡得到完整梯度的平均值，然后各自执行优化器更新。初始参数相同、同步后的梯度相同，因此更新后的参数也相同，副本不会漂移。

All-Reduce 的语义是“所有 rank 的输入求和，每个 rank 都得到完整结果”。在 DDP 中，输入是各卡的本地梯度；求和后除以 `N` 得到平均梯度。每张卡都保留完整梯度，因此优化器可以直接更新完整参数。

## 显存构成与 ZeRO 分片

混合精度加 Adam 的训练状态可以按参数逐项计算：

| 状态 | 精度 | 每参数字节 | 7B 模型 |
| --- | --- | --- | --- |
| 模型权重 | FP16/BF16 | 2 | 14 GB |
| 梯度 | FP16/BF16 | 2 | 14 GB |
| FP32 主权重 | FP32 | 4 | 28 GB |
| Adam 一阶动量 | FP32 | 4 | 28 GB |
| Adam 二阶动量 | FP32 | 4 | 28 GB |
| 合计 |  | 16 | 112 GB |

7B 模型需要 112 GB，单张 80 GB 卡无法容纳；该数字还不包含激活值、通信缓冲区和临时 workspace。拆法与[知乎的 ZeRO 图解](https://zhuanlan.zhihu.com/p/656639603)和 [Datawhale 的显存计算教程](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html)一致。不同框架的精度策略会改变系数，但优化器状态占比最大这一结论不变。

Zero Redundancy Optimizer（ZeRO）按阶段把训练状态分片到数据并行组。分片后的梯度规约使用规约散射（Reduce-Scatter），参数聚合使用全收集（All-Gather）：

| 方案 | 分片对象 | 每卡模型状态（7B，N=8） | 主要通信 |
| --- | --- | --- | --- |
| DDP / ZeRO-0 | 不分片 | 112 GB | 梯度 All-Reduce |
| ZeRO-1 | 优化器状态 | 2 + 2 + 12/8 = 5.5 B/参数 ≈ 38.5 GB | Reduce-Scatter + All-Gather |
| ZeRO-2 | 优化器状态 + 梯度 | 2 + 14/8 = 3.75 B/参数 ≈ 26.3 GB | Reduce-Scatter + All-Gather |
| ZeRO-3 / FSDP | 优化器状态 + 梯度 + 参数 | 16/8 = 2 B/参数 = 14 GB | 每层 All-Gather 参数 + Reduce-Scatter 梯度 |

ZeRO-1 的收益最大：只分片优化器状态，就把 112 GB 降到 38.5 GB，通信量与 DDP 处于同一量级。ZeRO-2 继续分片梯度，显存进一步下降，但每张卡仍保存完整参数。ZeRO-3 把参数也分片，显存最省，代价是每次前向都要把整层参数 All-Gather 回来；由于参数在前向和反向各聚合一次，并额外执行一次梯度 Reduce-Scatter，其总通信量约为 DDP 的 1.5 倍。[ZeRO 论文](https://arxiv.org/abs/1910.02054) 层越小、切得越细，这一开销越明显。

![DDP 与 ZeRO 的显存对比：DDP 每张卡复制参数、梯度和优化器状态；ZeRO-1/2/3 依次把优化器状态、梯度、参数分片到各卡。](images/dp-zero.svg)

示意图中四行分别对应 DDP、ZeRO-1、ZeRO-2 和 ZeRO-3，四列对应四张 GPU。绿色、橙色、蓝色横条分别表示参数、梯度和优化器状态；深色段表示该卡实际保存的分片，浅灰段表示该卡不保存。DDP 行中三种横条都横跨整张卡，表示每卡复制全部状态；ZeRO-1 只有蓝色横条被等分为四段，每卡保留一段；ZeRO-2 的橙色横条也按同样方式分片；ZeRO-3 的绿色横条同样分片。行左侧的数字是按 7B 模型和 8 卡估算的单卡占用，用于比较分片比例，不是实测值。

## FSDP 的实现关系

Fully Sharded Data Parallel（FSDP）是 PyTorch 对 ZeRO-3 的实现。它的一次前向和反向按以下顺序执行：

1. 构造时把参数切成 shard，每张卡只保留自己那一片。
2. 进入某一层前，All-Gather 该层的完整参数。
3. 执行前向计算，随后丢弃非本卡参数，释放显存。
4. 反向时再次 All-Gather 参数，计算梯度。
5. 反向结束后 Reduce-Scatter 梯度，每张卡留下自己负责的梯度分片。
6. 优化器只更新本地参数分片和对应的优化器状态。

PyTorch 文档强调，FSDP 应按层或按 Transformer block 调用 `fully_shard`，使 All-Gather 和释放以层为单位发生。若把整个模型当成一个 FSDP 单元，前向开始前就要聚合全部参数，显存收益会消失。[torch.distributed.fsdp.fully_shard](https://docs.pytorch.org/docs/main/distributed/fsdp.fully_shard.html)

FSDP2 使用 DTensor 表示分片参数，并提供 `ShardingStrategy.SHARD_GRAD_OP`（等价于 ZeRO-2，只切梯度和优化器状态）等策略。选择依据是当前瓶颈：显存不足时使用 FULL_SHARD，通信不足时退到 SHARD_GRAD_OP。

## 示例：7B 模型的四档显存

上述表格以一个 7B 模型、8 路 DP 为例，说明不同方案的单卡状态占用：

```text
模型参数 P = 7 × 10^9
每参数状态 = 权重 2 + 梯度 2 + FP32 主权重 4 + Adam m 4 + Adam v 4 = 16 B
单卡状态（不分片）= 7 × 10^9 × 16 B = 112 GB

ZeRO-1：参数 2 B + 梯度 2 B + 优化器状态 12/8 B = 5.5 B/参数 → 38.5 GB
ZeRO-2：参数 2 B + (梯度 2 + 优化器 12)/8 B = 3.75 B/参数 → 26.3 GB
ZeRO-3：16/8 B/参数 = 2 B/参数 → 14 GB
```

该算例是解释性构造，用于说明分片比例，不代表某个框架在特定硬件上的实测值。实际显存还要加上激活值、通信缓冲区和 allocator 保留空间。

## 梯度累积与全局 batch

数据并行的有效 batch 由三个量决定：

```text
global_batch = micro_batch × grad_accum × DP_size
```

`micro_batch` 是单卡一次前向的样本数，`grad_accum` 是在一次参数更新前累积的 micro-batch 数，`DP_size` 是数据并行组大小。显存不足时，减小 `micro_batch`、增大 `grad_accum`，可以在不改变全局 batch 的情况下降低激活值峰值。梯度累积不减少模型状态的显存，也不减少每个 step 的梯度通信次数；它把一次大 batch 拆成多次小 batch 计算，再把梯度累加。

调整全局 batch 时，学习率通常需要同步调整。线性缩放规则以 `global_batch` 的变化比例调整学习率，但大 batch 训练还需要 warmup 和梯度裁剪。只修改其中一个量，容易造成训练不稳定。

## 限制与适用条件

数据并行适合以下条件：

- 模型状态加一份 micro-batch 的激活值能装进单卡；
- 集群扩展主要依靠增加模型副本，而不是切分模型；
- 通信预算允许每个 step 同步一次梯度。

数据并行不适合以下条件：

- 单卡无法容纳权重；
- 激活值随序列长度增长，成为主要显存瓶颈；
- 优化器状态已经占满显存，而通信带宽不足以支撑 ZeRO-3 的频繁参数聚合。

数据并行只分摊模型状态，不分摊激活值。每张卡仍需保存本地 batch 的完整激活。长序列或大 batch 下，激活值可能超过模型状态，此时需要激活重计算、张量并行、序列并行或流水线并行。推理场景中，DP 复制模型并切分请求，可以提高吞吐；但每个副本独立保存键值缓存（KV Cache），显存不会因为 DP 而减少。

## 结论

DDP 通过梯度 All-Reduce 保持模型副本一致，计算效率高，但完整复制训练状态。ZeRO-1、ZeRO-2、ZeRO-3 依次分片优化器状态、梯度和参数，用额外的聚合通信换取显存；FSDP 是 PyTorch 中按层执行的 ZeRO-3 实现。选择哪一档，取决于显存与通信哪一项先成为瓶颈。

## 参考资料

- NVIDIA NeMo：[Parallelisms · Data Parallelism](https://docs.nvidia.com/nemo-framework/user-guide/25.11/nemotoolkit/features/parallelisms.html)
- NVIDIA Megatron Core：[Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- PyTorch：[Getting Started with Fully Sharded Data Parallel (FSDP2)](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- PyTorch：[torch.distributed.fsdp.fully_shard](https://docs.pytorch.org/docs/main/distributed/fsdp.fully_shard.html)
- Rajbhandari et al.，[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- 知乎：[图解大模型训练之：数据并行上篇（DP、DDP 与 ZeRO）](https://zhuanlan.zhihu.com/p/656639603)
- 知乎：[图解大模型训练之：数据并行下篇（DeepSpeed ZeRO）](https://zhuanlan.zhihu.com/p/654910335)
- 博客：[DeepSpeed 完全图解：ZeRO / 3D 并行 / Offload 从零讲清楚](https://yaoyuanzhou.github.io/topics/notes-deepspeed.html)
