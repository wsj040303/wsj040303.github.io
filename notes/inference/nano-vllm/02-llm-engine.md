# LLMEngine：请求入口与执行循环

`LLMEngine` 是 Nano-vLLM 的控制入口：它创建模型执行器和调度器，接收请求，再反复执行“调度 → 模型计算 → 状态更新”。本文以 [Nano-vLLM 提交 `bb823b3` 的 `llm_engine.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/llm_engine.py) 为准，解释单卡主路径和多卡分支。手敲练习稿可以按同一控制流逐段核对；运行结果与性能数据不从尚未验证的练习稿推断。

## 类的职责与初始化顺序

对外导出的 [`LLM`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/llm.py) 只继承 `LLMEngine`，没有重写方法。调用 `LLM(model_path, ...)` 就会进入 `LLMEngine.__init__()`。其中 `model_path` 指向包含模型配置、权重和分词器文件的目录；它既用于加载 Qwen3 权重，也用于加载匹配的 tokenizer。

初始化的顺序决定了后续组件能否取得完整配置：

| 步骤 | 代码动作 | 后续依赖 |
| --- | --- | --- |
| 1 | 从 `kwargs` 中筛出 [`Config`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/config.py) 的字段并建立配置 | 模型目录、最大批量 token 数、并行度等参数固定下来 |
| 2 | 创建 `ModelRunner`，加载权重、预热并分配 GPU KV 张量 | 执行器算出 `config.num_kvcache_blocks` |
| 3 | 从 `config.model` 加载 `AutoTokenizer`，读取 `eos_token_id` | 调度器知道何时因 EOS 结束请求 |
| 4 | 创建 `Scheduler(config)` | 块管理器取得真实块数和 EOS |

这里的“同一模型目录”强调**权重与分词器配套**。分词器把文本转换为 token ID，模型的嵌入层按这些 ID 读取权重；如果二者不匹配，同一个 ID 的语义会错位。`use_fast=True` 请求 Transformers 使用快速分词器实现，但不改变两者必须配套的关系。[`ModelRunner` 权重加载](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py)

## 多卡进程与 rank

**rank** 是并行进程的编号，从 0 到 `tensor_parallel_size - 1`。这个项目让每个 rank 对应一张 GPU，共同计算同一批请求的模型分片；它不是把不同请求简单分派给不同 GPU。rank 0 在当前进程运行，其他 rank 通过子进程启动。[`ModelRunner` 的设备与进程组初始化](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py)

```python
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)
self.model_runner = ModelRunner(config, 0, self.events)
```

`get_context("spawn")` 取得用于创建子进程和同步对象的上下文；这一行本身没有启动进程。`process.start()` 才让编号为 `i` 的 worker 开始运行。循环从 1 开始，是因为循环后的 `ModelRunner(config, 0, ...)` 已在当前进程创建 rank 0。单卡配置下 `range(1, 1)` 为空，不产生额外 worker。

每个子进程的 `Event` 用来通知它处理新命令，`self.ps` 保存进程对象，退出时可以逐个 `join()` 等待它们结束。rank 0 在当前进程中可以直接调用自己的 `ModelRunner`，但它仍参加同一组模型计算。控制消息由共享内存和事件传递；张量并行计算所需的结果通过分布式通信完成。这两类通信承担不同工作。[`ModelRunner.call()` 与共享内存路径](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py)

## 请求创建与入队

`add_request(prompt, sampling_params)` 只负责**提交**一个请求：字符串 prompt 经 tokenizer 转成 token ID；已经传入的 token ID 列表直接使用；接着构造 [`Sequence`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/sequence.py)，调用 `scheduler.add(seq)` 将它加入 `waiting` 队列。此方法没有立即运行模型，也没有返回新建 `Sequence` 的 ID。

`generate()` 允许输入一个 `SamplingParams` 对象或与 prompts 对应的参数列表。若只传一个对象，代码先把它重复成与 prompts 等长的列表。随后使用 `zip(prompts, sampling_params)` **按位置配对**：第一个 prompt 配第一组参数，第二个配第二组。每一对交给 `add_request()`。若调用者直接给长度不等的两个列表，Python 的 `zip` 在较短列表结束时停止；当前实现没有额外检查，因此剩余 prompt 不会入队。

```python
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

这段 `for` 在生成循环开始前完成，所以 `generate()` 的这一批请求都先到达等待队列。它能展示批量调度，但不能直接代表线上请求按不同时间到达的情况。

## 单轮调度与模型执行

`step()` 表示一次调度与模型计算。`schedule()` 返回两个值：`seqs` 是本轮选中的 `Sequence` 对象列表，`is_prefill` 表示本轮处理 prompt 还是执行 decode。当前调度器会先尝试 prefill；若选中 prefill，本轮不会混入 decode。[`Scheduler.schedule()` 源码](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py)

```python
seqs, is_prefill = self.scheduler.schedule()
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
```

`ModelRunner.run()` 按阶段准备输入、执行 Qwen3 前向并采样 token；`postprocess()` 把本轮已计算的 token 计入缓存进度，必要时追加采样 token，并在结束时释放该请求的 KV 块。`step()` 只把**本轮已完成的请求**放进 `outputs`。运行中的请求即使刚生成首 token，也暂时不在返回列表中。

这里的 `num_tokens` 用**正负号作阶段标记**：prefill 返回本轮计划计算的 prompt token 总数；decode 每个选中请求计算一个 token，于是返回 `-len(seqs)`。负号不代表“负数个 token”，它让后面的进度条能按符号区分 prefill 和 decode，分别计算显示速度。要记录实际每个请求的首 token 时间，需要在现有返回值之外增加逐请求事件。

## 生成循环与结果顺序

`generate()` 反复调用 `step()`，直到调度器的等待队列和运行队列都为空。每轮用 `perf_counter()` 量 `step()` 的耗时，再根据上述符号更新最近一轮的 prefill 或 decode 吞吐显示。这里显示的是**当前轮次的处理速度**，并非每个请求的首 token 延迟或尾延迟。

完成的请求以 `seq_id` 为键放进结果字典。最后代码按 `seq_id` 排序，再把生成 token ID 解码成文本，所以返回顺序与请求加入顺序一致。实际返回的每项是 `{"text": ..., "token_ids": ...}` 字典；当前源码把方法返回类型写为 `list[str]`，与实际返回结构不一致，阅读和手敲时应以第 89–90 行的返回语句为准。[返回路径源码](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/llm_engine.py)

## 两个请求的执行示例

设请求 A 与 B 都在 `generate()` 开始时加入，均配置 `max_tokens=2`、`ignore_eos=True`，且 token 预算与 KV 块足以让二者同处一次 prefill batch。下面是按当前代码推导的控制流；采样出的具体 token 受随机过程影响。

| 时刻 | 等待与运行状态 | 计算与返回 |
| --- | --- | --- |
| 入队完成 | A、B 都在 `waiting` | 尚未执行模型 |
| 第一次 `step()` | 调度器安排两者完整 prefill，移入 `running` | 算 prompt、各采样首 token；尚未满足两个输出 token，因此 `step()` 不返回完成请求 |
| 第二次 `step()` | A、B 在 `running` | 各输入上一步采样的 token，执行一次 decode；达到 `max_tokens=2`，释放其块并返回完成结果 |
| 循环结束 | 两个队列都为空 | 按 `seq_id` 排序，解码并返回 A、B 的结果 |

若 prompt 需要分块 prefill、资源不足或 EOS 提前结束，轮数和队列变化会不同。这个小例子只用于核对 `generate → step → schedule → run → postprocess` 的基本关系。

## 阅读结论与下一步

`LLMEngine` 自己不实现 Attention 算法或 KV 块分配细节。它的关键职责是**保持顺序与边界**：先完成资源配置，再接收请求；每轮先调度，再执行，再更新状态；所有请求结束后才按提交顺序返回。掌握这条控制流后，阅读 [`Sequence`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/sequence.py) 的计数，再进入 [`Scheduler`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py) 的 prefill/decode 分支，才能判断调度改动会影响哪些请求。
