# 大模型并行的五个维度与通信约束

本文讨论大模型训练和推理中的五种并行维度：数据并行、张量并行、流水线并行、专家并行和上下文并行。每种维度切分不同的对象，使用不同的集合通信原语，并因此适合放在不同的网络层级。框架参数和源码实现不在本文范围内。

## 问题定位

一个模型能否在多卡上运行，取决于两项约束：单卡显存是否容纳得下所需状态，以及切分后引入的通信是否能在可接受的时间内完成。显存决定可行性，通信决定效率。

训练状态的显存可以按参数逐项估算。以混合精度和 Adam 为例，每个参数需要保存 FP16 或 BF16 权重（2 字节）、同精度梯度（2 字节）、FP32 主权重（4 字节）、一阶动量（4 字节）和二阶动量（4 字节），合计 16 字节。一个 7B 模型仅这些状态就需要约 112 GB，超过一张 80 GB H100 的容量；70B 模型的 FP16 权重为 140 GB，单卡无法容纳权重本身。该拆法与 [ZeRO 的图解](https://zhuanlan.zhihu.com/p/656639603)和 [Datawhale 的显存计算教程](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html)一致。

上述估算不包含激活值、通信缓冲区和临时 workspace。激活值随 batch、序列长度和层数增长；长上下文训练时，单层注意力的 Q、K、V 和输出中间结果可达 GB 级。推理没有梯度和优化器状态，但需要保存随并发请求数和上下文长度线性增长的键值缓存（KV Cache）。后续的序列并行、上下文并行和 KV Cache 分页管理分别处理这两类状态。

## 五种并行维度

Megatron Core 将并行策略归纳为五个正交维度：数据并行（Data Parallelism，DP）切分 batch；张量并行（Tensor Parallelism，TP）切分单层内的权重和激活；流水线并行（Pipeline Parallelism，PP）切分模型深度；上下文并行（Context Parallelism，CP）切分序列长度；专家并行（Expert Parallelism，EP）切分混合专家（Mixture-of-Experts，MoE）模型的专家。[Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

![五种并行维度：DP 切 batch，TP 切层内张量，PP 切层，EP 切 MoE 专家，CP 切序列。](images/parallel-axes.svg)

示意图中五块面板对应五个切分对象。DP 面板中的四张 GPU 各自保存一份完整模型，只处理不同的 batch 分片；TP 面板把同一个权重矩阵沿列或行分到多张 GPU；PP 面板把连续层划分为 stage；EP 面板把 token 路由到持有目标专家的 GPU；CP 面板把一条序列切成多段。每个面板底部的通信标注说明了该维度需要的主要集合通信。

| 维度 | 切分对象 | 每张卡保留的状态 | 主要通信 | 典型放置 |
| --- | --- | --- | --- | --- |
| DP | 全局 batch | 一份完整模型副本 | 梯度全规约（All-Reduce），或规约散射（Reduce-Scatter）加全收集（All-Gather） | 跨节点，规模最大 |
| TP | 单层权重与激活的 hidden 或 head 维 | 每层的一片权重 | All-Reduce、All-Gather、Reduce-Scatter | 节点内 NVLink |
| PP | 连续的层 | 模型的一段 | 相邻 stage 的点对点（P2P）Send/Recv | 跨节点 |
| EP | MoE 的专家 | 一部分专家 | 全对全（All-to-All） | 跨节点，需高带宽 |
| CP | 序列维度 | 一段 token 的激活 | P2P 或 All-to-All 交换 KV | 节点内或相邻节点 |

表中的“通信域”比“切什么”更能决定实际配置。TP 在每一层都需要通信，通信频率最高，因此通常限制在 NVLink 域内；PP 只在 stage 边界传递激活，通信量最小，可以跨节点；DP 在每个训练 step 同步一次梯度，频率低于 TP；EP 的 All-to-All 同时经过 NVLink 和 InfiniBand，是 MoE 训练中带宽需求最高的一项。DeepSeek-V3 的技术报告给出其 H800 集群的带宽：节点内 NVLink 约 160 GB/s，跨节点 InfiniBand 约 50 GB/s，相差约 3.2 倍，这一比例直接决定了各维度的放置位置。[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)

## 通信原语与开销

五种维度最终使用同一组集合通信原语。理解原语的语义，可以判断某种并行是否值得开启。

| 原语 | 语义 | 典型使用位置 |
| --- | --- | --- |
| All-Reduce | 所有 rank 的输入求和，每个 rank 都得到完整结果 | DDP 梯度同步、TP 行并行的输出合并 |
| All-Gather | 每个 rank 贡献一片，所有 rank 得到拼接后的完整张量 | FSDP 前向取回参数、Ulysses 换回头维 |
| Reduce-Scatter | 先求和，再按片散回各 rank | ZeRO 梯度分片、TP 序列并行的激活规约 |
| All-to-All | 每个 rank 向其他每个 rank 发送不同分片 | MoE 的 token 分发与回收、Ulysses |
| Send/Recv | 点对点传输 | PP 的 stage 边界、Ring Attention 的 KV 传递 |

All-Reduce 可以分解为 Reduce-Scatter 加 All-Gather，这一等价关系是理解 ZeRO 和 FSDP 的基础。DDP 使用一次 All-Reduce 同步梯度，每张卡最终保留完整梯度；ZeRO 把它拆成两次通信，使优化器状态和梯度可以分片保存，代价是前向或反向时需要额外的参数聚合。

环形算法下，一次 All-Reduce 在每个 rank 上的通信量约为 `2 × (N-1)/N × S`，其中 `N` 是组内 rank 数，`S` 是张量字节数。组增大时，单卡通信量趋近 `2S`，不会无限增长；但同步延迟和带宽争用会随组增大而恶化。这一差异解释了为什么 TP 通常不超过单机 GPU 数，而 DP 可以扩展到数百甚至数千卡。

## 拓扑约束与放置

并行维度到硬件的映射遵循一条原则：通信量大的维度放在带宽高的链路，通信量小的维度放在跨节点链路上。NVIDIA 在 Megatron-Turing NLG 530B 的系统设计中给出了对应说明：三种策略中 TP 的通信开销最大，因此优先把 TP worker 放在同一节点内；PP 的通信量最小，可以把 stage 跨节点放置；DP 尽量放在节点内或相邻节点。[Megatron-Turing NLG 530B](https://arxiv.org/html/2201.11990v3)

真实配置重复了这条原则。GPT-3 175B 使用 8 路 TP 和 8 路 PP，再叠加 DP，在 1024 张 A100 上训练；MT-NLG 530B 的每个模型副本占 280 张 A100，节点内 8 路 TP、跨节点 35 路 PP，再用 DP 扩展到数千卡。两套配置都把 TP 限制在 8 卡以内，把 PP 放在节点之间。[Scaling Language Model Training to a Trillion Parameters](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron.md/)

## 组合关系

Megatron 用一条乘法关系描述各维度的组合：

```text
world_size = TP × PP × CP × EP × DP
```

以 2048 张卡、PP=16、EP=64 为例，若 TP=1、CP=1，则 DP=2。DeepSeek-V3 采用 16 路 PP、64 路 EP 和 ZeRO-1 DP，在 2048 张 H800 上训练，未使用 TP。其技术报告说明，H800 的节点内带宽和模型形状使 TP 在该配置下不划算。[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)

该公式也用于排查配置错误。`world_size` 不能被某个维度整除、stage 层数分配不均、EP 组跨越过多节点，都会在训练开始前或第一轮迭代中暴露。检查时应先核对整除关系和拓扑映射，再调整框架参数。

## 限制与适用条件

本文给出的显存数字基于混合精度加 Adam 的通用估算，实际框架的精度策略、分布式优化器和激活重计算会改变系数。通信量公式采用环形算法，NCCL 在不同拓扑和消息大小下会选择其他算法。真实配置的数字来自各模型的技术报告或官方博客，本文只做算术推导，未在对应集群上复现。

五种并行并非互斥，也不是开得越多越好。每种维度都用通信换取显存或计算资源，选择依据是比较“哪种切法能以最少的通信满足显存约束”。后续文章分别展开每种维度的切分对象、通信模式、显存收益和适用条件。

## 结论

大模型并行沿五个正交维度切分：DP 切 batch，TP 切层内张量，PP 切层，EP 切专家，CP 切序列。五者的通信频率和通信量依次决定其网络放置：TP 留在 NVLink 域内，PP 跨节点，DP 负责大规模扩展，EP 需要高带宽的 All-to-All，CP 用于长序列的激活值。

## 参考资料

- NVIDIA Megatron Core：[Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- NVIDIA NeMo Framework：[Parallelisms](https://docs.nvidia.com/nemo-framework/user-guide/25.11/nemotoolkit/features/parallelisms.html)
- NVIDIA Technical Blog：[Scaling Language Model Training to a Trillion Parameters Using Megatron](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron.md/)
- Shoeybi et al.，[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/html/1909.08053v4)
- Narayanan et al.，[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/html/2104.04473v5)
- DeepSeek-AI，[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)
- 知乎：[图解大模型训练之：数据并行（DP、DDP 与 ZeRO）](https://zhuanlan.zhihu.com/p/656639603)
- CSDN：[数据并行（DP）、张量模型并行（TP）、流水线并行（PP）](https://blog.csdn.net/yuanyuanxingxing/article/details/132412847)
- LLM Infra Atlas：[大规模训练的并行策略总览](https://llm-infra-atlas.github.io/parallel/)
