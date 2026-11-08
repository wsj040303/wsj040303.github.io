# 并行组合与选型：约束、配置与排障

五种并行维度在真实集群中很少单独出现。一个 2048 卡的训练任务通常同时使用流水线并行（Pipeline Parallelism，PP）、专家并行（Expert Parallelism，EP）和数据并行（Data Parallelism，DP）；一个 70B 模型的在线推理可能使用张量并行（Tensor Parallelism，TP）和数据并行。上下文并行（Context Parallelism，CP）用于长序列。本文给出维度组合的约束、选型所依据的通信成本、训练与推理的差异，以及从显存和拓扑推导配置的示例。

## 问题定位

并行配置需要同时满足三项约束：每个维度的并行度是整数，各维度乘积等于总卡数；每张卡的显存不超过容量；切分后引入的通信能在目标时间内完成。前两项决定配置是否可行，第三项决定配置是否高效。

Megatron 用一条乘法关系描述维度组合：

```text
world_size = TP × PP × CP × EP × DP
```

数据并行还可以乘以梯度累积，形成全局 batch：

```text
global_batch = micro_batch × grad_accum × DP
```

该关系的第一项作用是约束：各维度必须整除总卡数。第二项作用是分配通信域：TP 组最小、通信最频繁，应压在一个 NVLink 域内；PP 组按 stage 顺序通信，可以跨节点；EP 组的 All-to-All 跨节点，但需要高带宽；DP 组最大，通常跨节点。

## 选型依据：通信成本与拓扑

选型顺序由通信成本决定。NVIDIA NeMo 的性能调优指南给出的顺序是：先使用 DP；模型放不下时增加 TP；TP 仍不足时增加 PP；序列很长时增加 CP；MoE 模型再增加 EP。[Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/25.09/performance/performance-guide.html)

该顺序对应以下通信特征：

1. **DP**：每个 step 同步一次梯度，每卡执行完整 GEMM，计算效率高于模型并行；模型放得下时，DP 通常作为首选。
2. **TP**：每层都要 All-Reduce，通信频繁，但能同时降低权重和激活；限制在单机 NVLink 域内。
3. **PP**：只传 stage 边界的激活，通信量小，但有气泡，需要较大的 batch。
4. **CP**：序列 8K 以上、激活值成为瓶颈时使用；通信量通常低于 TP。
5. **EP**：只适用于 MoE，All-to-All 通信量最大，但能把专家分布到多个节点。

该顺序不是固定规则。DeepSeek-V3 跳过 TP，直接使用 PP + EP + DP；原因是其 MoE 结构和 H800 的带宽比例使 EP 的通信开销低于 TP。选型的实质是比较“哪种切法能以最少的通信满足显存约束”，而不是照搬某组参数。

## 五种维度的对照

| 维度 | 切分对象 | 显存收益 | 主要通信 | 通信域 | 典型并行度 | 主要代价 |
| --- | --- | --- | --- | --- | --- | --- |
| DP | 全局 batch | 无（复制模型） | 梯度 All-Reduce | 跨节点 | 数十到数千 | 模型状态冗余 |
| TP | 层内 hidden/head | 权重与激活按 TP 下降 | 每层 All-Reduce | 节点内 NVLink | 2–8 | 小 GEMM、通信频繁 |
| PP | 层 | 按 stage 下降 | 点对点（P2P）激活 | 跨节点 | 2–35 | 气泡、负载不均 |
| EP | MoE（混合专家）专家 | 专家权重按 EP 下降 | 全对全（All-to-All） | 跨节点 | 8–320 | 路由不均、延迟 |
| CP | 序列 | 激活按 CP 下降 | P2P / All-to-All | 节点内或相邻节点 | 2–8 | 注意力内核复杂、负载不均 |

表中的典型并行度来自公开配置，不是硬性上限。`TP=16` 跨两台机器时，All-Reduce 会经过 InfiniBand；`PP=32` 而 global batch 只有 32 时，气泡会占去大部分迭代时间；`EP=320` 而专家只有 256 个时，需要检查冗余专家和放置策略。配置是否合理，取决于这些量是否与拓扑和 batch 匹配。

## 公开配置分析

**GPT-3 175B。** NVIDIA 的 Megatron 扩展实践使用 8 路 TP 加 8 路 PP，在 1024 张 A100 上训练，剩余维度由 DP 补齐。TP 压在单机 8 卡内，PP 跨节点。[Scaling Language Model Training to a Trillion Parameters](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron.md/)

**MT-NLG 530B。** 每个模型副本占 280 张 A100：节点内 8 路 TP，跨节点 35 路 PP；再用 DP 扩展到数千卡。NVIDIA 说明，这样放置是因为 TP 通信开销最大、PP 通信量最小。[Megatron-Turing NLG 530B](https://arxiv.org/html/2201.11990v3)

**DeepSeek-V3 671B。** 2048 张 H800，16 路 PP、64 路 EP（跨 8 节点）、零冗余优化器（ZeRO-1）DP，TP 为 1。训练用 DualPipe 重叠计算与 All-to-All；推理时预填充（prefill）使用 EP32，解码（decode）使用 EP144，TP 均为 1。[DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)

三个配置的共同点是：TP 从未超过单机 8 卡；PP 和 EP 承担跨节点扩展；DP 负责把剩余卡数补齐。差异在 MoE 模型把 EP 放在核心位置，稠密模型主要依靠 TP + PP + DP。

## 训练与推理的差异

训练和推理的瓶颈不同，并行配置不能直接套用。

| 关注点 | 训练 | 推理 |
| --- | --- | --- |
| 主要显存 | 参数、梯度、优化器状态、激活 | 权重、键值缓存（KV Cache）、临时激活 |
| 梯度同步 | 每次更新需要 | 不需要 |
| TP 的作用 | 切分权重、梯度、优化器状态、激活 | 切分权重和 KV Cache |
| PP 的作用 | 分摊层，跨节点容纳模型 | 跨节点容纳模型，但增加尾延迟 |
| EP 的作用 | 分摊专家，配合 All-to-All 重叠 | 分摊专家，decode 阶段常用大 EP |
| 关键指标 | 吞吐（tokens/s）、模型浮点利用率（MFU） | 首 token 延迟、每输出 token 时间（TPOT）、吞吐 |

推理的显存计算需要把权重和 KV Cache 分开。以 Llama-3 70B 为例，FP16 权重约 140 GB，TP=8 时每卡约 17.5 GB。128K 上下文的 KV Cache 按 `2 × layers × kv_heads × head_dim × seq × batch × 2 bytes` 估算，单条序列约 43 GB；TP=8 后每卡约 5.4 GB。加上并发请求和激活峰值，80 GB 卡的余量并不宽裕。推理 TP 度不能只按权重估算，需要把 KV Cache 一并计入。

## 示例：64 卡配置推导

假设有 8 台机器、每台 8 张 80 GB H100，共 64 卡，训练一个 70B 稠密模型。推导过程如下：

1. **纯 DP**：70B 的训练状态按每参数 16 字节计约 1.12 TB，64 卡平均每卡 17.5 GB，但每张卡仍需保存完整模型状态，单卡放不下，纯 DP 不可行。
2. **TP=8**：单卡模型状态约 140 GB，仍超过 80 GB；TP=16 会跨节点，不划算。
3. **PP=4 + TP=8**：单卡模型状态约 35 GB；剩余卡数为 `64/(4×8)=2`，即 2 路 DP。每卡还需容纳激活值，35 GB 的余量足够。
4. **气泡**：PP=4 时，micro-batch 数 `m` 建议至少 16，气泡约 `3/(16+3)≈16%`；global batch 允许时把 `m` 增加到 32，气泡降到约 9%。
5. **通信**：TP 组在节点内，走 NVLink；PP 组跨节点，传激活；DP 只有 2 路，梯度同步压力较小。

该配置不是唯一解，但展示了选型的实际过程：先按显存确定 TP/PP，再用 DP 补齐卡数，最后用 `m` 控制气泡。算例为解释性构造，未在对应硬件上实测。

## 排障顺序

并行配置运行失败或性能低于预期时，按以下顺序检查：

1. **整除关系**：`world_size` 能否被各维度整除；层数能否被 PP 整除；头数能否被 TP 整除；专家数能否被 EP 整除。
2. **拓扑**：TP 组是否跨节点；EP 组跨了多少节点；PP 的 stage 是否落在带宽足够的链路上。
3. **显存峰值**：权重、优化器状态、激活或 KV Cache 哪一项先超出容量；对应选择 TP、ZeRO、激活重计算、CP 或 PagedAttention。
4. **气泡与通信**：用 profiler 查看 PP 气泡占比、All-Reduce/All-to-All 时间，以及计算与通信是否重叠。
5. **负载均衡**：PP 各 stage 的层数和耗时是否一致；EP 各 rank 收到的 token 数是否倾斜；CP 各 rank 的注意力计算量是否被 causal mask 拉偏。
6. **数值与语义**：TP 的 All-Reduce 精度、ZeRO 的分片边界、PP 的 tied embedding、CP 的 online softmax 是否与单卡实现一致。

前两步检查配置是否正确，后四步检查配置是否高效。整除关系和拓扑错误通常在启动阶段或第一轮迭代中暴露，不需要进入性能分析。

## 限制与适用条件

本文的配置推导基于稠密模型和混合精度训练。MoE 模型需要把 EP 纳入显存估算，并单独评估 All-to-All 的带宽；长序列训练需要把 CP 的 KV 交换计入通信时间；推理需要把 KV Cache 和请求并发纳入显存估算。

公开配置的数字来自各模型的技术报告或官方博客，本文只做算术推导，未在对应集群上复现。不同框架对参数切分顺序、通信重叠和显存分配的实现不同，同样的并行度可能得到不同的性能结果。

## 结论

并行配置是在卡数、显存和通信三者之间的约束优化。DP 计算效率最高，TP 解决单层放不下，PP 跨节点分摊层，CP 处理长序列激活，EP 处理 MoE 专家。选型顺序由通信成本决定，最终配置需要与网络拓扑、batch 大小和框架实现共同匹配。排障应先检查整除关系和拓扑，再分析显存峰值、气泡和负载均衡。

## 参考资料

- NVIDIA Megatron Core：[Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- NVIDIA NeMo：[Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/25.09/performance/performance-guide.html)
- NVIDIA Technical Blog：[Scaling Language Model Training to a Trillion Parameters Using Megatron](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron.md/)
- Narayanan et al.，[Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B](https://arxiv.org/html/2201.11990v3)
- DeepSeek-AI，[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)
- DeepSeek-AI，[DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- LLM Infra Atlas：[大规模训练的并行策略总览](https://llm-infra-atlas.github.io/parallel/)
- 知乎：[LLM 训练各种并行策略](https://saicat.github.io/4cd8532f.html)
