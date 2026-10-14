# scheduler.py: 选择每轮请求并管理 KV 块资源

```python
from collections import deque
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # self.waiting - 变量名（实例属性）deque[Sequence] - 类型注解- deque 是容器类型（双端队列）- [Sequence] 说明这个 deque 里存放的是 Sequence 类型的对象- 相当于告诉别人："这是一个装着 Sequence 对象的 deque"
        self.waiting: deque[Sequence] = deque()# 等待队列，包含新请求和需要继续 prefill 或抢占后重新进入的请求。
        self.running: deque[Sequence] = deque()# 运行队列，包含完整 prompt 已被安排处理、后续可能进行 decode 的请求。

    def is_finished(self):   #供 LLMEngine.generate() 结束循环。
        return not self.waiting and not self.running

    def add(self,seq: Sequence):
        self.waiting.append(seq)

        # schedule() 返回 list[Sequence] 与 bool。布尔值为 True 表示本轮走 prefill，False 表示 decode。
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens  # 本轮还可以计算多少 prefill token。
            if remaining == 0:
                break
            # If: 减去的是前缀缓存命中（来自其他请求的共享前缀）。
            # Else: 减去的是本请求已计算部分；其中已包含共享前缀和 chunked prefill 的进度。
            if not seq.block_table:     # 没有块表意味着这是第一次为它分配 KV 块，或者抢占后块表被清空。
                num_cached_blocks = self.block_manager.can_allocate(seq)   # 查询可复用前缀块数，并同时确认块数是否足够；返回 -1 表示当前不能分配。
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size    # 第一次分配时，待计算 token 数等于当前序列长度减去已命中的完整前缀块长度。
            else:          # 已有块表表示某个分块 prefill 尚未完成；待计算量改为当前序列总长度减去已算好的 token 数。
                num_tokens = seq.num_tokens - seq.num_cached_tokens  #为什么else分支不去剪掉共享前缀？因为在 else 分支时，共享前缀已经被处理过了。（之前必然经历过if分支）
            if remaining < num_tokens and scheduled_seqs:          # 当前请求放不进剩余 token 预算，且本轮已经安排过其他请求时，先停止。因此当前实现只允许本轮第一个请求使用不完整的 prefill chunk。
                break
            # 关联共享前缀块的目的是实现 KV cache 复用，避免重复计算。
            # 假设块大小为 16，两请求分词后的前 32 个 token 完全相同，且各需 4 块：
            # 请求 A（已处理）: "你是一个 helpful assistant。请帮我..."
            #                  └────── 前 32 个 token ──────┘
            #                  已计算，KV cache 存储在块 [5, 6]。
            # 请求 B（新请求）: "你是一个 helpful assistant。请翻译..."
            #                  └────── 前 32 个 token ──────┘
            # 不关联的情况：
            # 请求 A 的块表：seq_A.block_table = [5, 6, 7, 8]
            # 请求 B 的块表：seq_B.block_table = [9, 10, 11, 12]
            # 请求 B 需要重新计算前 32 个 token 的 KV cache。
            # 关联后的情况：
            # 请求 A 的块表：seq_A.block_table = [5, 6, 7, 8]
            # 请求 B 的块表：seq_B.block_table = [5, 6, 11, 12]
            # 请求 B 的前两个块复用 A 已计算的 KV cache。
            if not seq.block_table:        # 仅在没有块表时执行真实的块编号分配与前缀块关联。已分块处理过的请求继续使用原块表。
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens  # 一个批次包含一次运算一起要进行的token数num_batched_tokens；一个批次为了提高计算效率，会把多个请求的序列一起计算，这一个请求的token就是scheduledtokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:   # 若已缓存进度加本轮计划量等于当前序列长度，说明完整 prompt 已被安排在本轮算完。
                seq.status = SequenceStatus.RUNNING     # 把请求标成运行中。这个状态变化发生在本轮模型计算之前
                self.waiting.popleft()     # 从 waiting 队头移除，并追加到 running 队尾。
                self.running.append(seq)
            scheduled_seqs.append(seq)# 无论 prompt 本轮是否算完，只要安排了一个 chunk，就加入本轮执行列表。

        if scheduled_seqs:
            return scheduled_seqs, True

        # 只有没有 prefill 请求被选中时，控制流才到这里。
        # decode
        # 从运行队列选请求，最多选 max_num_seqs 个。
        # A 若需要新块而空闲池为零，等待 B、C、D 自然结束可能使 A 长时间无响应。
        # 抢占会产生重新进入 prefill 等开销，但能先让队头请求继续运行。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft() # deque.popleft() 从队头取出请求；deque.pop() 从队尾取出请求，两者都能高效地操作队列两端。
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop()) #抢占：腾出显存资源，让其他请求能继续运行
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1      # Decode 对这个请求每轮计算一个 token。
                seq.is_prefill = False             # 标记它进入 decode 输入准备路径，也影响多卡传输时只发送最后一个 token。
                self.block_manager.may_append(seq)   # 如跨过块边界，正式给请求追加一个物理 KV 块。
                scheduled_seqs.append(seq)   # 将当前请求放进本轮 decode batch。
        assert scheduled_seqs             # 要求本轮最终至少有一个请求可执行；
        self.running.extendleft(reversed(scheduled_seqs))# 把选中的请求按原顺序放回运行队列左端，等待模型执行和后处理；反转是为了抵消 extendleft 逐个插入时的顺序变化。
        return scheduled_seqs, False
    def preempt(self, seq: Sequence):         # 定义抢占：把请求从当前运行条件撤回。调用者负责从 running 中取出它。
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True           # 下次调度需走 prefill 恢复路径；若之前的 KV 块可通过前缀缓存命中，可减少重算。
        self.block_manager.deallocate(seq)    # 释放该请求对 KV 块的引用并清空块表、缓存进度。
        self.waiting.appendleft(seq)   # 抢占请求放在等待队列前端，比普通新请求更早再次尝试。

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):# postprocess() 接收本轮请求列表、模型为它们采样的 token ID 列表和本轮路径标记。
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)            # 给本轮刚算完的完整 KV 块登记内容哈希，使后续请求有机会复用。
            seq.num_cached_tokens += seq.num_scheduled_tokens     # 将本轮已完成的模型计算计入缓存进度。
            seq.num_scheduled_tokens = 0             # 清除本轮计划量，下一轮由 schedule() 重新设置。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:    #  意思是prefill还没完全看完，第一个token都没有生成。如果本轮只是长 prompt 的中间 chunk，就忽略此次采样 token；prompt 尚未完整，不能把它当成真正的生成结果。
                continue
            seq.append_token(token_id)                      # 对完整 prefill 或 decode，把采样 token 追加到序列末尾，更新总长度和 last_token。这个新 token 的 KV 要在下一轮 decode 才计算。
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens: # ignore_eos=False 且采样到 EOS，或已生成数恰好达到 max_tokens 时，结束请求。
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

## KV 块不足时的抢占

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)
        break
```

[`Scheduler.schedule()`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py) 在 decode 分支从 `running` 中选择请求。这些请求已经完成 prompt 处理，正在逐个生成新 token。生成时，模型要把新 token 的 Key 和 Value 写入 KV Cache。`BlockManager.can_append(seq)` 检查当前请求在下一步是否需要一个新的物理 KV 块，以及空闲块是否足够。它的条件是 `len(seq) % block_size == 1`：只有刚跨过块边界才需要新块；若当前块还有位置，即使空闲池为空，这一步也能继续。[`can_append()` 源码](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py)

假设 `running` 依次是 A、B、C、D，A 正好需要新块，但空闲块为零。只等待 B、C、D 自然生成到结束，A 的下一步可能要等很久；当它们的输出很长，或系统持续有请求占用块时，A 会长时间没有响应。B、C、D 如果还能继续生成并最终结束，这个例子属于**资源等待造成的请求饥饿或长尾延迟**；如果多个请求都在块边界等新块，则可能出现无人能推进的资源停滞。代码因此把队尾 D 抢占，解除 D 对块的引用，再检查 A；还不够就继续从队尾撤回请求。它优先让队头请求取得资源，并没有保证每个请求都能立即前进。

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

抢占的代价从这四行可以看出：D 离开运行队列，块表与 `num_cached_tokens` 被清空，下次从等待队列重新走 prefill。`BlockManager.deallocate()` 解除 D 对块的引用；若某个块还被其他请求共享，它的引用计数不会降到零，因此这次抢占也未必马上腾出一个空闲块。[`deallocate()` 源码](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py) 释放后的完整块若仍在哈希索引中，D 恢复时可能再次命中；如果块已被重新分配给不同内容，相应 token 就要重新计算。抢占用额外的状态切换与可能的重算，换取队头请求及时继续执行。

如果其他运行请求都已撤回，当前请求仍拿不到块，`else` 会连当前请求也抢占。这份实现随后要求 `scheduled_seqs` 非空；若本轮一个请求都没选中，`assert scheduled_seqs` 会报错。因此，这段逻辑解决的是常见的资源竞争，不能把它理解为对所有块容量配置都能完成请求的保证。

## 共享前缀与分块 prefill 的剩余计算量

```python
if remaining == 0:
    break
if not seq.block_table:
    num_cached_blocks = self.block_manager.can_allocate(seq)
    if num_cached_blocks == -1:
        break
    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
else:
    num_tokens = seq.num_tokens - seq.num_cached_tokens
if remaining < num_tokens and scheduled_seqs:
    break
if not seq.block_table:
    self.block_manager.allocate(seq, num_cached_blocks)
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

`remaining` 是本轮还能计算的 token 数；`num_tokens` 是当前请求**还需要**计算的 token 数。两者分开，才能决定这轮是否放得下当前请求。`block_table` 保存这个请求按顺序使用的物理 KV 块编号。它为空时，说明还没有为当前执行阶段建立块映射：新请求会如此；被抢占后，`deallocate()` 清空块表，也会重新进入这条分支。

第一次分配时，`can_allocate(seq)` 按 token 内容和前缀哈希查找连续、完整、已经算过的 KV 块，并同时检查块容量。返回的 `num_cached_blocks` 是**块数**，乘上 `block_size` 才是可以跳过计算的 token 数。真正将这些共享块编号写进 `seq.block_table` 的是后面的 `allocate(seq, num_cached_blocks)`；它还会设置 `seq.num_cached_tokens = num_cached_blocks * block_size`。[`can_allocate()` 与 `allocate()` 源码](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py)

例如，设每块容纳 16 个 token，请求 A 和 B 分词后的前 32 个 token 完全相同，后面的 token 不同，且两者各需 4 块。A 已算好前缀并使用 `[5, 6, 7, 8]`。B 第一次进入时可命中前两块，于是它的块表可以是 `[5, 6, 11, 12]`：`5`、`6` 保存两者共用的前缀 KV，`11`、`12` 留给 B 的不同部分。若 B 总长 64 个 token，`num_cached_blocks = 2`，这一分支得到 `num_tokens = 64 - 2 × 16 = 32`。这些编号是解释性示例，实际编号由空闲块池决定；复用要求 token ID 按块连续一致，文本看起来相似并不足够。

如果 B 的 32 个待计算 token 本轮只得到 20 个预算，`seq.num_scheduled_tokens` 就是 20。模型处理完后，`postprocess()` 把 20 加到 `num_cached_tokens`，此时它是 52。B 的 `block_table` 仍在；下一轮进入 `else`，计算 `64 - 52 = 12`。这里无需再减一次 32 个共享前缀 token，因为 `num_cached_tokens` 已经把复用的 32 和新算好的 20 都记入进度。抢占会清空块表和进度；恢复时重新走 `if`，再次查询仍可用的共享前缀。

`if remaining < num_tokens and scheduled_seqs` 规定：本轮已经选了其他请求，当前请求又放不进剩余预算，就先停下。只有本轮第一个被选中的请求可以使用不完整的 prefill chunk。这个条件使一次 prefill batch 不会同时放入多个仅处理了部分 prompt 的请求。

## 运行队列为何先取出再放回

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())
        else:
            self.preempt(seq)
            break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))
return scheduled_seqs, False
```

`popleft()` 从队头取出准备调度的请求，体现这里优先考虑较早进入 `running` 的请求。`pop()` 同样是 `deque` 的高效两端操作；代码用它从队尾挑选被抢占者，保留队头的优先权。先把当前请求取出还有一个作用：资源不足时，`self.running.pop()` 取到的是**其他**运行请求，不会误把当前正在争取资源的请求当作第一个牺牲对象。

取出不代表运行请求已经完成。选中请求仍需留在 `running`，因为模型执行后 `postprocess()` 可能调用 `self.running.remove(seq)` 移除已结束的请求。`scheduled_seqs` 是本轮交给模型计算的列表；`running` 则保存跨轮次继续生成的请求。调度完成后，`extendleft(reversed(scheduled_seqs))` 把选中的请求放回运行队头。

设 `running = [A, B, C]`，本轮最多选两个请求，三者都不需要抢占。`popleft()` 先后取出 A、B，队列暂时是 `[C]`，`scheduled_seqs = [A, B]`。`extendleft()` 会逐个从左侧插入，所以先把列表反转为 `[B, A]`，插入 B 再插入 A，最终恢复为 `[A, B, C]`。这也说明它不是把 A、B 轮换到队尾：本实现会让已选中的队头请求保持靠前，直到结束、被抢占，或受到其他调度条件限制。
