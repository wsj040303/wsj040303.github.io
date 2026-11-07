# 组合与选型：3D/4D 并行怎样落到集群

前面六篇分别讲了五种并行维度。真实集群里，它们很少单独出现：一个 2048 卡的训练任务通常同时用 PP、EP、DP，一个 70B 模型的在线推理可能用 TP 加 DP。这一篇把五个维度拼回去，回答两个问题：给定模型和集群，应该先开哪个、后开哪个；以及看到一组并行参数时，怎样判断它是否合理。

## 一条公式，五个自由度

Megatron 用一条乘法关系描述并行维度的组合：

```text
world_size = TP × PP × CP × EP × DP
```

其中 DP 通常还会再乘以梯度累积，形成全局 batch：

```text
global_batch = micro_batch × grad_accum × DP
```

这条公式的第一层作用是约束：每个维度都必须是整数，且乘积要等于总卡数。第二层作用是分配通信域：TP 组最小、通信最频繁，应该压在一个 NVLink 域内；PP 组按 stage 顺序通信，可以跨节点；EP 组的 All-to-All 跨节点，但需要高带宽；DP 组最大，通常跨节点。

以 2048 张 H800 为例，DeepSeek-V3 取 `PP=16, EP=64, TP=1, CP=1`，于是 `DP=2`。这不是随便选的：H800 的节点内带宽和模型形状让 TP 不划算，PP 负责把 61 层分到 16 个 stage，EP 负责把 256 个专家分到 64 个 rank，剩下的 2 路 DP 用 ZeRO-1 分摊优化器状态。[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)

## 选型顺序：从最便宜的并行开始

NVIDIA NeMo 的性能调优指南给了一个明确的顺序：先用 DP；模型放不下再加 TP；TP 还不够再加 PP；序列很长时加 CP；MoE 模型再加 EP。[Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/25.09/performance/performance-guide.html)

这个顺序背后的逻辑是“通信成本从低到高”：

1. **DP 最先**：每步只同步一次梯度，计算效率最高；只要模型放得下，DP 几乎总是最优。
2. **TP 其次**：每层都要 All-Reduce，但能同时降低权重和激活；限制在单机 NVLink 域内。
3. **PP 再次**：只传 stage 边界的激活，通信便宜，但有气泡，需要较大的 batch。
4. **CP 补充**：序列 8K 以上、激活值爆炸时才需要；通信量通常低于 TP。
5. **EP 最后**：只适用于 MoE，All-to-All 最贵，但能把专家摊到很多节点。

这条顺序不是绝对的。DeepSeek-V3 就跳过了 TP，直接用 PP + EP + DP；原因是它的 MoE 结构和 H800 的带宽比例让 EP 比 TP 更划算。选型的本质不是背顺序，而是比较“哪种切法能用最少的通信，换来足够的显存”。

## 五种维度的横向对照

| 维度 | 切什么 | 显存收益 | 主要通信 | 通信域 | 典型并行度 | 主要代价 |
| --- | --- | --- | --- | --- | --- | --- |
| DP | 全局 batch | 无（复制模型） | 梯度 All-Reduce | 跨节点 | 数十到数千 | 模型状态冗余 |
| TP | 层内 hidden/head | 权重与激活按 TP 下降 | 每层 All-Reduce | 节点内 NVLink | 2–8 | 小 GEMM、通信频繁 |
| PP | 层 | 按 stage 下降 | P2P 激活 | 跨节点 | 2–35 | 气泡、负载不均 |
| EP | MoE 专家 | 专家权重按 EP 下降 | All-to-All | 跨节点 | 8–320 | 路由不均、延迟 |
| CP | 序列 | 激活按 CP 下降 | P2P / All-to-All | 节点内或相邻节点 | 2–8 | 注意力内核复杂、负载不均 |

这张表可以直接用来排查配置。例如看到 `TP=16` 跨了两台机器，就要怀疑 All-Reduce 是否走了 InfiniBand；看到 `PP=32` 但 global batch 只有 32，就要怀疑气泡是否吃掉了大部分算力；看到 `EP=320` 但专家只有 256 个，就要检查冗余专家和放置策略。

## 三个真实配置

**GPT-3 175B。** NVIDIA 的 Megatron 扩展实践使用 8 路 TP 加 8 路 PP，在 1024 张 A100 上训练，剩余维度由 DP 补齐。TP 压在单机 8 卡内，PP 跨节点，这是最经典的 3D 并行配置。[Scaling Language Model Training to a Trillion Parameters](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron.md/)

**MT-NLG 530B。** 每个模型副本占 280 张 A100：节点内 8 路 TP，跨节点 35 路 PP；再用 DP 扩到数千卡。NVIDIA 明确说明，这样放置是因为 TP 通信开销最大、PP 通信量最小。[Megatron-Turing NLG 530B](https://arxiv.org/html/2201.11990v3)

**DeepSeek-V3 671B。** 2048 张 H800，16 路 PP、64 路 EP（跨 8 节点）、ZeRO-1 DP，TP 为 1。训练用 DualPipe 重叠计算与 All-to-All；推理时 prefill 用 EP32，decode 用 EP144，TP 均为 1。[DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)

三个配置的共同点是：TP 从来没有超过单机 8 卡；PP 和 EP 承担跨节点扩展；DP 负责把剩余卡数补齐。不同点是 MoE 模型把 EP 放到了核心位置，稠密模型则主要靠 TP + PP + DP。

## 训练与推理：同一组维度，不同的账

训练和推理的并行选型不能直接套用，因为瓶颈不同。

| 关注点 | 训练 | 推理 |
| --- | --- | --- |
| 主要显存 | 参数、梯度、优化器状态、激活 | 权重、KV Cache、临时激活 |
| 是否需要梯度同步 | 是，每次更新 | 否 |
| TP 的作用 | 切权重、梯度、优化器状态、激活 | 切权重和 KV Cache |
| PP 的作用 | 分摊层，跨节点放模型 | 跨节点放模型，但增加尾延迟 |
| EP 的作用 | 分摊专家，配合 All-to-All 重叠 | 分摊专家，decode 阶段常用大 EP |
| 关键指标 | 吞吐（tokens/s）、MFU | 首 token 延迟、TPOT、吞吐 |

一个具体的推理账本：Llama-3 70B 的 FP16 权重约 140 GB，TP=8 时每卡约 17.5 GB，看起来 80 GB 卡很宽裕。但 128K 上下文的 KV Cache 按 `2 × layers × kv_heads × head_dim × seq × batch × 2 bytes` 估算，单条序列约 43 GB；TP=8 后每卡约 5.4 GB。如果再考虑并发请求和激活峰值，显存并不富裕。这说明推理的 TP 度不能只看权重，要把 KV Cache 一起算进去。

## 一个 64 卡的配置练习

假设有 8 台机器、每台 8 张 80 GB H100，共 64 卡，要训练一个 70B 稠密模型。可以这样推：

1. **先试纯 DP**：70B 的训练状态约 1.1 TB（按 16 字节/参数），64 卡平均每卡 17.5 GB，但每张卡仍要放完整模型状态，单卡放不下，纯 DP 不可行。
2. **加 TP**：TP=8 把单卡模型状态降到约 140 GB，仍然超过 80 GB；TP=16 会跨节点，不划算。
3. **加 PP**：PP=4、TP=8，单卡模型状态约 35 GB；剩下 `64/(4×8)=2` 路 DP。每卡还要放激活值，35 GB 的余量足够，这个配置可行。
4. **检查气泡**：PP=4 时，micro-batch 数 `m` 建议至少 16，气泡约 `3/(16+3)≈16%`；如果 global batch 允许，把 `m` 加到 32，气泡降到约 9%。
5. **检查通信**：TP 组在节点内，走 NVLink；PP 组跨节点，传激活；DP 组只有 2 路，梯度同步压力小。

这组配置不是唯一解，但它展示了选型的实际过程：先算显存，再定 TP/PP，最后用 DP 补齐卡数，并用 `m` 控制气泡。

## 排障顺序

并行配置跑不起来或跑得慢时，按下面的顺序检查，通常比盲目调参有效：

1. **整除关系**：`world_size` 能否被各维度整除；层数能否被 PP 整除；头数能否被 TP 整除；专家数能否被 EP 整除。
2. **拓扑**：TP 组是否跨节点；EP 组跨了多少节点；PP 的 stage 是否落在带宽足够的链路上。
3. **显存峰值**：是权重、优化器状态、激活还是 KV Cache 先爆；对应地选择 TP、ZeRO、激活重计算、CP 或 PagedAttention。
4. **气泡与通信**：用 profiler 看 PP 气泡占比、All-Reduce/All-to-All 时间、计算与通信是否重叠。
5. **负载均衡**：PP 各 stage 的层数和耗时是否一致；EP 各 rank 收到的 token 数是否倾斜；CP 各 rank 的注意力计算量是否被 causal mask 拉偏。
6. **数值与语义**：TP 的 All-Reduce 精度、ZeRO 的分片边界、PP 的 tied embedding、CP 的 online softmax 是否与单卡实现一致。

前两步是“配置是否正确”，后四步是“配置是否高效”。很多问题在第一步就能发现，不需要上 profiler。

## 一张决策清单

最后把选型压缩成一张清单，方便在真实项目里逐条对照：

- 模型能装进单卡：先用 DP，把卡数用于扩大 batch。
- 单卡装不下权重：先试 ZeRO/FSDP；仍不够，上 TP，TP 不超过单机 GPU 数。
- TP 后仍装不下：加 PP，保证 `m` 远大于 `p`，并检查 stage 是否均分。
- 序列 8K 以上且激活值先爆：加 CP，确认注意力内核支持 causal mask 和负载均衡。
- MoE 模型：在 PP/DP 之外加 EP，优先保证专家放置的节点局部性，再考虑扩大 EP。
- 推理：先把权重和 KV Cache 分开算；权重放不下用 TP/PP，KV Cache 放不下用 TP/EP 或分页管理；decode 阶段谨慎使用 CP。
- 任何时候：先画拓扑，再算通信量，最后才调参数。

并行不是“开得越多越快”，而是“用最合适的通信换最需要的显存”。把这句话记住，再看任何一组并行配置，都能问出正确的问题。

## 参考资料

- NVIDIA Megatron Core：[Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- NVIDIA NeMo：[Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/25.09/performance/performance-guide.html)
- NVIDIA Technical Blog：[Scaling Language Model Training to a Trillion Parameters Using Megatron](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron.md/)
- Narayanan et al.，[Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B](https://arxiv.org/html/2201.11990v3)
- DeepSeek-AI，[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)
- DeepSeek-AI，[DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- LLM Infra Atlas：[大规模训练的并行策略总览](https://llm-infra-atlas.github.io/parallel/)
- 知乎：[LLM 训练各种并行策略](https://saicat.github.io/4cd8532f.html)
