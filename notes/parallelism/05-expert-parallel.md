# 专家并行：MoE 路由、All-to-All 与负载均衡

专家并行（Expert Parallelism，EP）把混合专家（Mixture-of-Experts，MoE）模型的路由专家分散到不同 GPU。每个 token 由路由器选出少量专家，被发送到持有这些专家的 GPU 上计算，再返回原 GPU 加权求和。EP 降低单卡专家权重的占用，代价是两次全对全（All-to-All）通信，以及由路由结果引起的负载不均衡。本文说明 MoE 层的结构、EP 的分发与回收过程、负载均衡机制，以及 DeepSeek-V3 的公开配置。

## 问题定位

稠密模型的并行切分同一个矩阵；MoE 模型的并行切分“哪些参数被激活”。DeepSeek-V3 有 671B 总参数，每个 token 只激活 37B。若每张 GPU 都保存完整模型，FP16 权重约 1.3 TB，单个 80 GB GPU 无法容纳；但每张 GPU 只需保存一部分专家，就能把模型摊到数十张卡上。

EP 的显存收益来自专家分片，通信代价来自 token 路由。token 不再固定留在某张卡上，它必须离开原 GPU，前往持有目标专家的 GPU，计算后再返回。这一次往返构成 MoE 训练和推理中最昂贵的通信。

## MoE 层的基本结构

一个 MoE 层包含一个路由器和若干专家。对每个 token，路由器计算它到各专家的分数，选出 top-k 个专家；每个被选中的专家对 token 执行一次前馈网络（FFN）；最后按路由权重把各专家的输出加权求和。

DeepSeek-V3 的配置是 256 个路由专家加 1 个共享专家，每个 token 激活 8 个路由专家，总参数 671B、激活 37B。共享专家对所有 token 生效，路由专家由路由器选择。[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)

## 专家并行的分发与回收

设专家并行度为 `EP`，模型有 `E` 个路由专家，则每张卡持有 `E/EP` 个专家。以 DeepSeek-V3 的 256 个专家、`EP=64` 为例，每张卡持有 4 个专家；若启用冗余专家，每张卡持有的数量相应增加。

![专家并行的一次分发与回收：token 经路由器选出专家后，通过 Dispatch All-to-All 发送到持有对应专家的 GPU，专家计算后再通过 Combine All-to-All 返回并按权重相加。](images/ep-alltoall.svg)

示意图中左侧是各数据并行 rank 的本地 token，中间是路由器，右侧是持有专家的 EP rank。箭头从本地 token 指向路由器，再从路由器指向专家所在的 GPU；这一阶段称为 Dispatch All-to-All。专家计算完成后，结果沿虚线箭头返回原 GPU，称为 Combine All-to-All。底部流程说明每张卡的执行顺序：本地 token 记录路由目标、按目标专家分组发送、专家执行 FFN、结果送回并按路由权重相加。

Dispatch 与 Combine 的数据量都与 `token 数 × top-k × hidden × 字节数` 成正比。`top-k=8` 意味着同一个 token 要被复制到 8 个目的地，通信量是稠密模型激活传输的数倍。这决定了 EP 需要与计算重叠，并依赖高带宽网络。

## EP 与 DP 的关系

在 Megatron 的并行布局中，EP 通常叠加在 DP 之上：Attention 部分按 DP 复制，MoE 部分按 EP 切分。一个 DP 组内的多张卡，在 MoE 层里组成一组 EP rank；Attention 仍然各自持有完整副本。

该结构使 EP 可以在不复制全部专家的前提下扩展显存。代价是 token 在 EP 组内往返，All-to-All 成为新的瓶颈。EP 只作用于 MoE 层；Attention、LayerNorm、Embedding 仍需要 DP、TP 或 PP 处理。因此真实配置通常是“Attention 用 DP/TP，MoE 用 EP”，再叠加 PP 负责模型深度。

## 负载均衡

EP 的通信量由路由结果决定。某个专家过载时，持有它的 GPU 收到大量 token，成为 straggler（拖慢整体进度的慢节点）；其他 GPU 在等待它。路由随训练数据动态变化，负载不均衡会反复出现。

常用的三类机制如下：

**辅助负载均衡损失。** 在训练损失中加入一项，惩罚专家负载的方差，鼓励路由器把 token 分得更均匀。该做法自 GShard、Switch Transformer 起被广泛使用，但辅助损失可能影响语言建模目标。

**无辅助损失的偏置调整。** DeepSeek-V3 为每个专家维护一个偏置项，根据负载动态调整路由分数，而不在损失函数中增加惩罚。这样既平衡负载，又避免辅助损失对语言建模目标的干扰。

**冗余专家与重排。** 对高负载专家复制多份，把副本打包到不同 GPU。DeepSeek 开源的 EPLB（Expert Parallelism Load Balancer）根据估计的专家负载，先按节点做分层打包，保证节点间负载均衡，再在节点内复制和放置专家；得益于 group-limited routing，它还会尽量把同一组专家放在同一节点，减少跨节点流量。[EPLB](https://github.com/deepseek-ai/EPLB)

DeepSeek-V3 使用 node-limited routing：256 个路由专家分成 8 组，每组 32 个，部署时每组放在一个节点上，算法保证每个 token 最多被路由到 4 个节点。这样跨节点 All-to-All 的流量有上界，不会因路由到过多节点而失控。[Insights into DeepSeek-V3](https://arxiv.org/pdf/2505.09343)

## 通信实现：DeepEP

DeepSeek 开源的 DeepEP 是面向 MoE 的通信库，目标是降低 EP 的 All-to-All 开销：

- 同时支持 NVLink 和 RDMA，节点内走 NVLink，节点间走 InfiniBand；
- 为 prefill 提供高吞吐内核，为 decode 提供低延迟内核；
- 原生支持 FP8 调度，降低通信带宽需求；
- 通过 SM 分配和通信/计算重叠，减少 All-to-All 对计算资源的占用。

这些优化对应 EP 的两个阶段。prefill 阶段 token 多、吞吐优先；decode 阶段每步 token 少、延迟优先。两者的最优内核和并行度不同，生产部署通常将 prefill 与 decode 分开配置。

## 示例：DeepSeek-V3 的 EP 配置

DeepSeek-V3 的训练使用 16 路 PP、64 路 EP（跨 8 个节点）和 ZeRO-1 DP，在 2048 张 H800 上完成，未使用 TP。官方在后续系统分析中说明，H800 的节点内带宽和模型形状使 TP 在该配置下不划算，而 EP 能把专家分散到更多节点，配合 DualPipe 把 All-to-All 藏入计算。[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)

推理阶段采用不同的并行度。公开的推理系统说明给出：prefill 使用路由专家 EP32，MLA 和共享专家 DP32，一个部署单元跨 4 个节点，每张卡负责 9 个路由专家加 1 个共享专家；decode 使用 EP144，一个部署单元跨 18 个节点，每张卡负责 2 个路由专家加 1 个共享专家。两者均为 TP1。[DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)

decode 使用更大 EP 的原因是每步 token 少。单卡专家过多时，每个专家分到的 batch 太小，通用矩阵乘（GEMM）效率低；把专家摊到更多卡上，每张卡只保留少量专家，可以让每个专家获得合适的 batch，同时降低单卡显存。代价是 All-to-All 的跨节点跳数增加，需要低延迟内核和双 micro-batch 重叠来掩盖。

## 限制与适用条件

EP 的并行度受以下条件约束：

- 专家总数必须能被 EP 整除；冗余专家的放置还需额外处理；
- EP 组越大，All-to-All 的跨节点流量越大，网络越容易成为瓶颈；
- 每张卡上的专家太少时，单个专家的 GEMM 变小，计算效率下降；
- 路由负载不均时，EP 度越大，straggler 的影响越明显；
- EP 只作用于 MoE 层，Attention 和 Embedding 仍需其他并行维度处理。

EP 适合专家数量多、单卡无法容纳全部专家、且具备高带宽跨节点网络的 MoE 模型。稠密模型没有专家可切，EP 不适用。专家数量很少、或者 batch 小到每个专家只能收到少量 token 时，EP 的通信开销会超过显存收益。

## 结论

专家并行把 MoE 的路由专家分散到多张 GPU，通过 Dispatch 和 Combine 两次 All-to-All 完成 token 往返。它按 EP 比例降低专家权重的单卡占用，但通信量随 top-k 和 EP 度增长，并受路由负载影响。负载均衡依赖辅助损失、无辅助损失的偏置调整、冗余专家和节点局部性放置；DeepEP 等实现通过专用内核和计算通信重叠降低开销。

## 参考资料

- DeepSeek-AI，[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437)
- DeepSeek-AI，[Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://arxiv.org/pdf/2505.09343)
- DeepSeek-AI，[DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- DeepSeek-AI，[EPLB: Expert Parallelism Load Balancer](https://github.com/deepseek-ai/EPLB)
- NVIDIA Megatron Core：[Parallelism Strategies Guide · Expert Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- 博客：[专家并行（EP）深度解析——MoE 时代的第五种并行维度](https://forceinjection.github.io/09_inference_system/parallelism/expert_parallelism_deep_dive.html)
- 火山引擎开发者社区：[DeepSeek-V3 技术报告（中文）](https://developer.volcengine.com/articles/7542491735755063342)
