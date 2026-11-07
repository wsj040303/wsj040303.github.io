# 张量并行：在同一层内切开矩阵

数据并行解决“副本太多”，张量并行解决“一层都放不下”。它的做法不是把模型按层分给不同卡，而是把同一层里的权重矩阵沿某个维度切开，每张卡只算矩阵的一部分，再用集合通信把部分结果拼成完整输出。Megatron-LM 的关键贡献，是找到了列并行与行并行的组合方式，使一个 Transformer 层在前后各只需一次通信，而不是每切一刀就同步一次。

## 从一个 MLP 开始：列并行接行并行

Transformer 的 MLP 可以写成 `Y = GeLU(XA) · B`，其中 `X` 的形状是 `[batch, seq, hidden]`，`A` 把 hidden 映射到中间维，`B` 再映射回 hidden。要把这个计算切到两张卡上，先看 `A` 怎么切。

如果按行切 `A`，得到 `X1A1 + X2A2`，必须先把两个部分和加起来才能过 GeLU，因为 GeLU 是非线性的：`GeLU(X1A1 + X2A2) ≠ GeLU(X1A1) + GeLU(X2A2)`。这意味着 GeLU 之前就要一次同步，通信被夹在两层中间。

Megatron 选择按列切 `A`：

```text
A = [A1, A2]
Y = XA = [X A1, X A2] = [Y1, Y2]
GeLU(Y) = [GeLU(Y1), GeLU(Y2)]
```

每个 rank 独立计算自己那组输出列，GeLU 可以逐元素作用，不需要任何通信。接着把 `B` 按行切：

```text
B = [B1; B2]
Z = YB = Y1 B1 + Y2 B2
```

每个 rank 得到一个部分和，最后做一次 All-Reduce 求和，得到完整的 `Z`。这就是 Megatron 的 `f` 与 `g` 两个共轭算子：`f` 在前向是恒等、反向是 All-Reduce；`g` 在前向是 All-Reduce、反向是恒等。MLP 的前向只有一次 All-Reduce，反向也只有一次。[Megatron-LM 论文](https://arxiv.org/html/1909.08053v4)

![张量并行的 MLP：第一层按列切分，GeLU 各卡独立计算，第二层按行切分，最后一次 All-Reduce 求和。](images/tp-mlp.svg)

| 线性层 | 切权重的方式 | 输入 | 输出 | 前向通信 | 反向通信 |
| --- | --- | --- | --- | --- | --- |
| ColumnParallelLinear | 沿输出维（列）切 | 完整复制 | 按列切分 | 无 | 对输入梯度 All-Reduce |
| RowParallelLinear | 沿输入维（行）切 | 按列切分 | 部分和 | All-Reduce 求和 | 无 |

这张表的“共轭”关系值得记住：列并行把通信推到反向，行并行把通信推到前向。把两者串起来，通信就落在 MLP 的两端，而不是中间。

## Attention：按头切分

Self-Attention 天然适合张量并行，因为多头注意力的每个头本来就是独立的。Megatron 把 Q、K、V 的投影矩阵按列切分，每个 rank 负责一部分头；每个头在自己的卡上完成 `QK^T`、softmax 和 `AV`；输出投影矩阵再按行切分，最后 All-Reduce 一次。

这样一来，一个 Transformer 层的前向有两次 All-Reduce：一次在 Attention 的输出投影后，一次在 MLP 的输出投影后。反向对应两次。合起来就是常说的“每层前向 2 次、反向 2 次 All-Reduce”。[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/html/2104.04473v5)

每次 All-Reduce 的张量大小是 `batch × seq × hidden`。以 `batch=1, seq=4096, hidden=8192`、BF16 为例，一次通信约 64 MB；环形 All-Reduce 在每个 rank 上的通信量约为 `2 × (N-1)/N × 64 MB`。`N=8` 时约 112 MB，`N=16` 时约 120 MB。单看数字增长不大，但 All-Reduce 是同步操作，延迟和带宽争用会随组增大而恶化，这就是 TP 通常不超过单机 8 卡的原因。

## 一个具体的显存例子

Llama-3 70B 的一层 FFN 里，`W1` 的形状是 `8192 × 28672`。FP16 下这个矩阵约 470 MB；如果再加上 FP32 主权重和 Adam 的 `m`、`v`，这一层带来的训练状态接近 5 GB。80 层堆起来，纯 DDP 显然放不下。[LLM Infra Atlas · TP/SP](https://llm-infra-atlas.github.io/parallel/02_tp_sp/)

切到 8 路 TP 后，每张卡只保留 `8192 × 3584` 的一片，权重降到约 59 MB，优化器状态也按同一比例缩小。代价是每层多出两次 All-Reduce，以及更小的 GEMM。TP 的收益和代价都来自这个“切得更细”：显存线性下降，单卡矩阵乘的 `M` 或 `N` 也线性下降，算术强度随之降低，通信占比上升。

## 序列并行：把 LayerNorm 和 Dropout 也切开

标准 TP 只切了 Attention 和 MLP 的矩阵乘，LayerNorm、Dropout、残差相加这些逐点操作仍然在每个 rank 上保存完整的 `[batch, seq, hidden]` 激活。序列并行（Megatron Sequence Parallelism）把这些区域沿序列维切开：每个 rank 只保存 `1/TP` 的 token，LayerNorm 和 Dropout 在自己的分片上独立计算。

代价是 TP 区域的边界要换一种通信方式。原本行并行输出的一次 All-Reduce，被替换成 Reduce-Scatter（把完整激活规约并散回各卡）加 All-Gather（进入 TP 区域前把序列拼回来）。通信量在理论上与 All-Reduce 相当，但激活值显存下降了，反向时能保存更多激活，或者用更大的 batch。Megatron 的 `--sequence-parallel` 打开的就是这个优化。[Megatron Core · Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

这里要注意命名。Megatron 的“序列并行”是 TP 的配套优化，切的是 TP 区域内没被切到的逐点操作；第 6 篇讲的“上下文并行 / 长序列并行”切的是整个注意力的序列维，两者不是同一件事。

## 词表并行

Embedding 和最后的 LM Head 在词表维上很大。TP 还会把词表切开：每个 rank 只保存一部分 token 的 embedding，输出 logits 也只算自己那一段，最后对 logits 做 All-Reduce 或 All-Gather，再计算交叉熵。这样做的好处是 embedding 和 LM Head 的显存也随 TP 下降；坏处是词表大小必须能被 TP 整除，而且最后的 logits 通信量与词表大小成正比，词表很大时不容忽视。

## TP 在推理中的样子

推理没有梯度和优化器状态，TP 的显存收益主要来自权重和 KV Cache 的切分。vLLM 的 `tensor_parallel_size` 会把每层的权重按输出维切开，注意力头也按 rank 分配；每个 rank 只保存自己那部分头的 KV Cache。

这带来一个实践约束：注意力头数必须能被 TP 整除。对 GQA/MQA 模型，KV 头数可能少于 Q 头数，当 TP 大于 KV 头数时，部分 rank 没有 KV 头可分，框架通常会把 KV 头复制到多个 rank，或者要求 TP 不超过 KV 头数。部署前先检查模型的 `num_attention_heads` 和 `num_key_value_heads`，比调参更重要。

推理 TP 还会影响延迟。TP 越大，单卡 GEMM 越小，All-Reduce 的相对开销越高；在小 batch 的 decode 阶段，通信延迟可能盖过计算收益。常见做法是 prefill 用较大的 TP 提高吞吐，decode 用较小的 TP 或改用 EP/DP，这也是 DeepSeek 分离 prefill 与 decode 部署的动机之一。

## 什么时候用 TP

适合 TP 的信号：

- 单层权重或单层激活放不进单卡；
- 模型 hidden size 较大（4096 以上），切分后每卡仍有足够大的 GEMM；
- 有 NVLink/NVSwitch，TP 组能压在一个节点内。

不适合 TP 的信号：

- 模型能靠 DP 或 ZeRO 放下，TP 只会增加通信；
- TP 度超过单机 GPU 数，All-Reduce 被迫走 InfiniBand；
- 注意力头数或词表大小不能被 TP 整除；
- batch 很小、序列很短，GEMM 已经很小，再切就只剩通信。

## 参考资料

- Shoeybi et al.，[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/html/1909.08053v4)
- Narayanan et al.，[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/html/2104.04473v5)
- NVIDIA Megatron Core：[Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- NVIDIA NeMo：[Parallelisms · Tensor Parallelism](https://docs.nvidia.com/nemo-framework/user-guide/25.11/nemotoolkit/features/parallelisms.html)
- LLM Infra Atlas：[Tensor Parallelism（TP）与 Sequence Parallelism（SP）](https://llm-infra-atlas.github.io/parallel/02_tp_sp/)
- CSDN：[图解大模型分布式训练：张量并行 Megatron-LM 方法](https://blog.csdn.net/Antai_ZHU/article/details/144409187)
- CSDN：[Megatron 的并行与优化原理](https://happyprince.blog.csdn.net/article/details/163314037)
- 博客园：[模型并行分布式训练 Megatron（3）——模型并行实现](https://www.cnblogs.com/rossiXYZ/p/15871062.html)
