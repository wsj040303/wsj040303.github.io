# 序列与上下文并行：把长序列切开

前面几种并行都在切模型：DP 切 batch，TP 切层内矩阵，PP 切层，EP 切专家。序列并行换了一个维度：模型不动，把一条长序列切成几段，每张卡只处理其中一段。它解决的是长上下文带来的激活值爆炸——当序列从 4K 涨到 128K，单层注意力的中间结果可能从几十 MB 涨到 GB 级，单卡无论怎么切模型都放不下。

“序列并行”这个词有两个常见含义，先分清：

- **Megatron 序列并行（SP）**：TP 的配套优化，切的是 LayerNorm、Dropout 等逐点操作的激活，目的是减少 TP 区域外的激活冗余。
- **序列/上下文并行（CP）**：把整条序列切到多张卡上，每张卡负责一段 token 的完整 Transformer 计算，注意力通过交换 KV 完成。

这一篇主要讲后者，最后再回到前者，说明两者在真实配置里怎样配合。

## 为什么长序列需要单独切

先看一笔注意力激活账。设 `batch=1`、`seq=128K`、`64` 个 Q 头、`8` 个 KV 头、`head_dim=128`、BF16，单层注意力需要保存：

| 张量 | 形状 | 大小 |
| --- | --- | --- |
| Q | `[1, 128K, 64, 128]` | 2 GB |
| K | `[1, 128K, 8, 128]` | 0.5 GB |
| V | `[1, 128K, 8, 128]` | 0.5 GB |
| O | `[1, 128K, 64, 128]` | 2 GB |

单层就要 5 GB，几十层堆起来，80 GB 卡很快见底。这还不算 MLP 的中间激活和反向所需的保存。把序列按 CP 切成 8 段，每张卡只保存 `1/8` 的 Q/K/V/O，激活值直接降到原来的八分之一。[飞桨开源社区 · 大模型训练中的序列并行](https://pfcc.blog/posts/context-parallel-sharing)

但注意力不是逐点操作。每个 query token 都要和序列里所有 key 做点积，如果每张卡只持有自己那段 KV，它就无法独立完成注意力。所以 CP 的核心问题是：怎样在切分序列的同时，让每张卡看到完整的 KV，并且通信量可控。

## Ring Attention：把 KV 块沿着环传递

Ring Attention 的做法很直接：每张卡持有自己那段序列的 Q、K、V，然后沿着一个环，把自己的 KV 块依次传给下一张卡；每收到一块 KV，就用它和本地的 Q 计算一部分注意力，直到所有 KV 块都路过一遍。

```text
rank 0 持有 Q0 K0 V0
rank 1 持有 Q1 K1 V1
rank 2 持有 Q2 K2 V2
rank 3 持有 Q3 K3 V3

第 1 步：每张卡用本地 KV 算局部 attention
第 2 步：K/V 传给下一张卡，继续算
第 3 步：再传一圈，直到所有 KV 块都参与过
```

难点在于 softmax 不能简单地分段相加。Ring Attention 使用 online softmax：每算完一块 KV，就更新当前的最大值和归一化因子，把新块的贡献合并进来。这样不需要保存完整的 `S × S` 注意力矩阵，显存从 `O(S²)` 降到 `O(S)`，计算和通信还能重叠——当前块的注意力在算，下一块 KV 已经在路上。

环形传递的通信量是每张卡把本地 KV 传 `CP-1` 次，数据量与 `S/CP × hidden` 成正比；由于是点对点传输，可以拆成小块与计算重叠，不需要全局同步。这是 Ring Attention 在超长序列上仍然可用的原因。[Ring Attention 论文](https://arxiv.org/abs/2310.01889)

## DeepSpeed Ulysses：用 All-to-All 换头维

Ulysses 走另一条路。它不传 KV，而是重新排列 Q/K/V 的布局：先把序列维切分、头维完整，做一次 All-to-All，变成“序列完整、头维切分”；每张卡只计算一部分头，但能看到完整序列；算完再 All-to-All 换回原来的布局。

```text
切分前：每张卡 [S/CP, H]        —— 序列切分，头完整
All-to-All：交换序列块与头块
切分后：每张卡 [S, H/CP]        —— 序列完整，头切分
计算注意力：每张卡负责 H/CP 个头
All-to-All：换回 [S/CP, H]
```

这样注意力的计算本身不需要跨卡，每张卡在自己的头上做完整的 `S × S` 注意力；通信集中在两次 All-to-All 上，一次换过去、一次换回来。Ulysses 的通信量比 Ring Attention 更集中，适合有高带宽互联的集群；缺点是头数必须能被 CP 整除，而且 All-to-All 是全局同步，重叠难度更高。[DeepSpeed Ulysses 论文](https://arxiv.org/abs/2309.14509)

![Ring Attention 与 Ulysses 的对比：Ring 沿环传递 KV 块并与计算重叠，Ulysses 用两次 All-to-All 在序列维和头维之间重排。](images/cp-ring.svg)

Megatron 的上下文并行更接近 Ring Attention：沿序列维切分，注意力通过 P2P 交换 KV，支持 causal mask，并可以和 TP、PP、DP 组合。NVIDIA 的文档把 CP 定位为“序列长度 8K 以上”的并行维度，并指出 CP 的通信量与 TP 同量级但通常更低。[Megatron Core · Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

## 负载均衡：causal mask 带来的麻烦

自回归注意力的 causal mask 让计算量呈三角形分布：前面的 token 要 attend 到更多位置，后面的 token 更少。如果简单地把序列切成连续几段，第一段负责的 query 少、KV 多，最后一段 query 多、KV 少，各 rank 的计算量并不相等。

常见解法是 zigzag 切分：把序列按块编号，rank 0 拿第 1 块和最后 1 块，rank 1 拿第 2 块和倒数第 2 块，以此类推，使每张卡分到的“query 数 × KV 数”大致相等。这个细节在 Ring Attention 和 Megatron CP 的实现里都存在，长序列训练时不做负载均衡，CP 的加速比会明显低于线性。

## CP 与 TP 的关系

两者都在切激活，但切的维度不同：

| 维度 | 切什么 | 注意力是否需要跨卡 | 主要通信 | 适用场景 |
| --- | --- | --- | --- | --- |
| TP | hidden / head 维 | 是，每个头被切开 | 每层 All-Reduce | 层放不下，hidden 大 |
| Megatron SP | 序列维（仅逐点操作） | 否 | All-Gather / Reduce-Scatter | 配合 TP 降激活 |
| CP | 序列维（整个 Transformer） | 是，交换 KV | P2P 或 All-to-All | 序列很长（8K+） |

NVIDIA NeMo 的性能指南给了一个有用的判断：当序列长度显著大于 hidden size 时，激活值会先爆，这时 CP 比继续加大 TP 更有效；同样大小的张量，CP 的通信量通常低于 TP。[Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/25.09/performance/performance-guide.html)

Megatron SP 和 CP 可以同时使用：SP 在 TP 组内切逐点激活，CP 在更外层切整条序列。真实的长上下文训练配置里，常见的是 `TP × CP × PP × DP` 的组合，CP 通常放在节点内或相邻节点，因为 KV 交换的带宽需求较高。

## 推理中的长上下文

训练里的 CP 主要解决激活值；推理里的瓶颈换成了 KV Cache。长上下文推理时，KV Cache 随 `seq × batch` 线性增长，可能比模型权重还大。这时有几种思路：

- **TP/EP 切 KV Cache**：把 KV 头或层切到多张卡上，每张卡只保存一部分 KV；
- **CP 切 prefill**：长 prompt 的 prefill 阶段把序列切开，并行计算注意力，缩短首 token 延迟；
- **PagedAttention 管理 KV**：不切并行度，而是把 KV 分页存储，减少碎片和重复。

decode 阶段每步只生成一个 token，序列维的并行收益有限，反而增加通信；所以生产系统通常只在 prefill 或超长上下文场景使用 CP，decode 更依赖 TP/EP 和 KV Cache 管理。这个取舍与 DeepSeek 把 prefill、decode 分开部署的思路一致。

## 什么时候用 CP

适合 CP 的信号：

- 序列长度 8K 以上，激活值成为显存瓶颈；
- hidden size 不够大，继续加 TP 的收益已经很低；
- 有支持 CP 的注意力内核（Ring Attention、Megatron CP、Ulysses）；
- 序列长度能被 CP 整除，且能处理 causal mask 的负载不均。

不适合 CP 的信号：

- 序列很短，CP 的通信和调度开销超过收益；
- 头数太少，Ulysses 的 All-to-All 无法均分；
- 框架不支持 CP 的注意力实现，只能退化成低效的全局 All-Gather；
- 推理 decode 阶段，序列维并行没有足够的工作量。

## 参考资料

- Liu et al.，[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
- Jacobs et al.，[DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)
- Korthikanti et al.，[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- NVIDIA Megatron Core：[Parallelism Strategies Guide · Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- NVIDIA Megatron Core：[Context Parallel Overview](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- NVIDIA NeMo：[Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/25.09/performance/performance-guide.html)
- 飞桨开源社区：[大模型训练中的序列并行技术与优化策略](https://pfcc.blog/posts/context-parallel-sharing)
- Hugging Face 博客：[Ulysses + Ring-Attention 技术原理与实现](https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention)
- LLM Infra Atlas：[Tensor Parallelism（TP）与 Sequence Parallelism（SP）](https://llm-infra-atlas.github.io/parallel/02_tp_sp/)
