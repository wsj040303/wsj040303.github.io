# CUDA 性能测试与调优

## 测试目标与计时口径

CUDA 性能调优从固定工作负载开始：输入形状、数据类型、内存布局、数值范围、精度要求和输出校验都要明确。同一 Kernel 在小输入时可能主要受启动延迟影响，在大输入时可能主要受内存或计算吞吐限制。比较两个实现时，应保持输入、编译选项和计时范围一致。

**端到端耗时**从调用方开始到结果可用，可能包含内存分配、Host（CPU）与 Device（GPU）间复制、Kernel 启动和同步；**Kernel 区间时间**只覆盖指定 Stream 中包围 Kernel 的测量区间。前者决定程序是否真正变快，后者帮助解释设备端改动。Kernel 启动是异步的：若用 CPU 时钟测量，结束计时前必须同步；CUDA Event 可直接测量 GPU Stream 中的区间。相关计时方法见 [CUDA C++ Best Practices Guide · Timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#timing)。

| 目标 | 使用的测量 | 用于回答的问题 |
| --- | --- | --- |
| 版本比较 | 未启用分析器的重复计时 | 同一工作负载下究竟快了多少 |
| 程序级定位 | `nsys` 的 CUDA API、复制和 Kernel 时间线 | 时间消耗发生在调用、传输、等待还是计算 |
| Kernel 级定位 | `ncu` 的吞吐、访存和资源指标 | 哪种硬件行为支持当前瓶颈假设 |

分析器会改变被测程序的执行条件，因此先建立未插桩基线，再用 `nsys` 和 `ncu` 解释原因，最后回到同一基线方法复测。记录 GPU、驱动、CUDA Toolkit、编译选项、输入规模、线程块配置、热身次数和统计方法，才能在另一台机器上解释结果差异。

## 设备与编译环境

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
nvcc --version
```

| 命令部分 | 含义 | 迁移到自己的设备 |
| --- | --- | --- |
| `nvidia-smi` | 查询 NVIDIA 驱动可见的 GPU。 | 多卡机器应确认程序实际使用的是哪张卡。 |
| `--query-gpu=...` | 选择输出字段；逗号分隔，不能在字段间加入空格。`name` 是型号，`compute_cap` 是计算能力，`driver_version` 是驱动版本，`memory.total` 是显存总量。 | 按需增删字段；计算能力用于选择 `nvcc -arch`。 |
| `--format=csv` | 按逗号分隔值输出，便于保存测试环境。 | 加 `noheader` 可去掉表头，但文章示例保留表头便于辨认单位。 |
| `nvcc --version` | 显示 CUDA 编译器版本；它与驱动版本是不同信息。 | 换机器后重新记录；同一源码在不同编译器或目标架构下可能生成不同指令。 |

例如 `compute_cap` 显示 `12.0` 时，本机 CUDA Toolkit 13.4 的编译目标使用 `-arch=sm_120`。目标架构还必须受到所安装 Toolkit 支持；不能仅按 GPU 型号猜测。硬件环境记录本身不构成性能结论。

## 正确性与 CUDA Event 基线

先运行参考实现并逐项检查输出，再测量性能版本。边界形状应覆盖小于一个 Block、恰好整除 Block 和有尾部线程的情况。Kernel 启动后用 `cudaGetLastError()` 检查启动错误，在需要结果时同步以发现执行期错误。若浮点归约改变运算顺序，正确性比较应使用由应用误差要求确定的容差。

CUDA Stream 是 GPU 工作按提交顺序执行的序列；Event 记录到指定 Stream 中，GPU 执行到该位置时才写入时间戳。把开始 Event、Kernel 和结束 Event 排入同一个 Stream，才能使区间对应这次启动。

```cuda
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args...);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
float milliseconds = 0.0f;
cudaEventElapsedTime(&milliseconds, start, stop);
float microseconds = milliseconds * 1000.0f;
```

| 代码或变量 | 作用与单位 | 替换到自己的算子时 |
| --- | --- | --- |
| `start`、`stop` | 通过 `cudaEventCreate` 创建的两个 GPU Event；分别记录区间起点和终点。 | 计时结束后用 `cudaEventDestroy` 释放。 |
| `stream` | 被测工作所在的 CUDA Stream；Event 与 Kernel 要放在同一个 Stream。 | 使用默认 Stream 时可写 `0`；其他 Stream 用实际句柄。 |
| `kernel<<<grid, block, 0, stream>>>(args...)` | 四个启动配置依次是 Grid 尺寸、每个 Block 的线程数、动态共享内存字节数、Stream；圆括号内是 Kernel 实参。 | 把 `kernel` 和 `args...` 换成自己的函数与设备指针；有动态共享内存时把第三项 `0` 换成所需字节数。 |
| `cudaEventSynchronize(stop)` | 等待 GPU 记录结束 Event 后再读取时间。 | 同步只用于取得已完成的测量结果；不要把同步调用本身当成 Kernel 时间。 |
| `&milliseconds`、`microseconds` | `&milliseconds` 把输出变量的地址交给 `cudaEventElapsedTime`；函数写入毫秒数，乘 `1000` 得到微秒。 | 统一报告单位，避免把毫秒直接写成微秒。 |

先热身数次，排除初始化和首次加载的影响，再采集多次结果并报告中位数与独立运行之间的波动。对极短 Kernel，Event 区间可能受到启动提交间隙影响，应同时观察实际调用路径。CUDA Event 的记录与返回单位见 [CUDA Runtime API · Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)。

对于访存算子，可计算**算法有效带宽**：`有效带宽 = (按算法计的读取字节数 + 写入字节数) / (Kernel 秒数 × 10^9)`，单位为 GB/s。字节数按算法真实读写次数计算；缓存命中时，有效带宽不是 DRAM 计数器测得的流量。详见 [CUDA C++ Best Practices Guide · Bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#bandwidth)。

## Nsight Systems 程序级时间线

**Nsight Systems**（`nsys`）同时记录 CPU 发出的 CUDA API 调用以及 GPU 上的 Kernel 和复制。先看时间线与汇总，判断程序是否把大量时间花在分配、数据传输、同步等待或 Kernel 之间的空隙，再决定是否深入单个 Kernel。

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --stats=true -o /tmp/cuda_perf_profile ./cuda_perf_bench
nsys stats --report cuda_gpu_trace /tmp/cuda_perf_profile.sqlite
```

| 命令部分 | 具体含义 | 迁移到自己的程序 |
| --- | --- | --- |
| `nsys profile` | 启动目标程序并采集一次时间线；第一行行尾的 `\` 只表示 Shell 命令延续到下一行。 | 被测程序放在所有 `nsys` 选项之后；程序自己的参数再放在程序名之后。 |
| `--trace=cuda` | 记录 CUDA API、GPU Kernel 和 CUDA 内存操作。 | 分析 NVTX 标记时可改成 `--trace=cuda,nvtx`。 |
| `--sample=none`、`--cpuctxsw=none` | 本例不采 CPU 指令采样和 CPU 线程切换，只保留 CUDA 时间线。 | 若问题来自 CPU 调度或数据准备，再开启相应采集；这两个选项不影响本例的 CUDA 汇总。 |
| `--stats=true` | 采集结束后输出默认统计表；本机版本同时生成供 `nsys stats` 使用的 SQLite 文件。 | 先从汇总表找时间较多的操作，再看逐条时间线。 |
| `-o /tmp/cuda_perf_profile` | 指定报告路径前缀，不是被测程序路径；生成 `.nsys-rep` 报告及本例的 `.sqlite` 文件。 | 换成可写且每次采集不冲突的路径；重复使用同名前缀时可加 `--force-overwrite=true`。 |
| `./cuda_perf_bench` | 被 `nsys` 启动的可执行程序。 | 换成自己的可执行文件，例如 `./my_operator --size 4096`。 |
| `nsys stats --report cuda_gpu_trace ...sqlite` | 第二行离线读取既有采集数据；`cuda_gpu_trace` 逐条列出 GPU 操作的起点、持续时间、Grid、Block 和名称。 | 将文件名改成前一行生成的 SQLite 路径；若只保留 `.nsys-rep`，也可传入报告文件让工具导出 SQLite。 |

默认统计中的 `cuda_api_sum` 按 CPU 侧 API 名称汇总调用次数和耗时；`cuda_gpu_kern_sum` 汇总 GPU Kernel 启动；`cuda_gpu_mem_time_sum` 和 `cuda_gpu_mem_size_sum` 分别汇总 GPU 复制时间与字节数。`Total Time` 是该类别各次调用之和，`Avg` 是每次平均，`Instances` 或 `Num Calls` 是次数。API 时间与 GPU Kernel 时间可能重叠，不能简单相加成端到端墙钟时间。报告命令见 [Nsight Systems User Guide · Stats](https://docs.nvidia.com/nsight-systems/UserGuide/#example-stats-command-sequences)。

## Nsight Compute Kernel 指标

**Nsight Compute**（`ncu`）针对选中的 Kernel 收集硬件指标。`SpeedOfLight` 显示计算和存储层次的吞吐；`MemoryWorkloadAnalysis` 补充访存信息。下列命令把目标限制为一个已热身的 Kernel，使采集成本和结果解释更清楚。

```bash
ncu --list-sections
ncu --query-metrics --query-metrics-mode all
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
  -k regex:coalesced_add -s 6 -c 1 ./cuda_perf_bench
```

| 命令部分 | 具体含义 | 迁移到自己的 Kernel |
| --- | --- | --- |
| `ncu --list-sections` | 列出当前安装版本支持的报告章节及其准确名称。 | 根据要验证的问题挑选章节；章节越多，重放与采集开销通常越大。 |
| `ncu --query-metrics --query-metrics-mode all` | 列出可查询指标及其完整后缀变体；`all` 使下文用到的 `.sum` 名称也出现在结果中。 | 指标名与硬件或工具版本有关，复制前先核对；输出较长，可在终端内搜索关键字。 |
| `--section SpeedOfLight` | 采集计算、DRAM、L1/TEX、L2 等部件相对峰值的吞吐；百分比对应分析器当前采集条件。 | 先看哪条通路接近上限，再决定进一步采集什么。 |
| `--section MemoryWorkloadAnalysis` | 在同一次命令中增加内存工作负载章节；`--section` 可重复指定多个章节。 | 分析访存时保留；其他问题可换成 `LaunchStats` 或 `WarpStateStats`。 |
| `-k regex:coalesced_add` | `-k` 是 Kernel 名过滤器；`regex:` 表示后面的表达式做部分正则匹配。 | 改成自己的 Kernel 名或表达式；先从 `nsys` 的 Kernel 名称确认拼写。含 Shell 特殊字符时给表达式加引号。 |
| `-s 6` | 跳过前 6 次**匹配过滤器**的启动，再开始采集。本例对应 1 次正确性检查和 5 次热身。 | 按自己程序的同名 Kernel 调用顺序计算；若只想采第一次，可写 `-s 0`。 |
| `-c 1` | 只采 1 次匹配的启动，避免把 20 次计时启动全部做指标采集。 | 只有确实需要比较多次采样时才增加数量。 |
| `./cuda_perf_bench` | 让 `ncu` 启动并运行整个程序，过滤器只决定哪些 Kernel 被分析。 | 换成自己的可执行程序及参数。 |

若连续访存假设需要更直接的证据，可采集 L1/TEX 全局加载的请求数与 sector 数。一个 **sector** 是对齐的 32 字节内存片段；对 32 个活跃线程各读一个连续 32 位数值的 Warp，理想情况下一个加载请求涉及 4 个 sector。`sectors / requests` 越高，通常表示一次请求触及更多片段；也要结合参与线程数和元素宽度判断。定义见 [Nsight Compute Profiling Guide · Quantities](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#quantities)。

```bash
ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  -k regex:coalesced_add -s 6 -c 1 ./cuda_perf_bench
```

| 指标片段 | 含义 |
| --- | --- |
| `--metrics A,B` | 以逗号连接两个指标名；逗号后不加空格，行尾 `\` 将两行接成一个参数。 |
| `l1tex__` | L1/TEX 单元的指标前缀。 |
| `t_requests` / `t_sectors` | 分别计全局加载请求与这些请求触及的 sector。 |
| `pipe_lsu_mem_global_op_ld` | 限定为 Load/Store Unit 路径上的全局内存加载；不会把本例的全局写回混入分子。 |
| `.sum` | 把被选中 Kernel 的对应事件计数求和；这两项都是计数，不是百分比。 |

`ncu` 可能多次重放 Kernel，且默认时钟与缓存控制、Kernel 串行化方式可与普通运行不同。因此它的 `Duration` 和分析器运行时程序打印的 Event 数值用于诊断，不与未插桩基线直接计算加速比。低 DRAM 百分比也不能单独证明计算受限：请求可能主要在 L2 服务，或者受访存形态、并行度等限制。参见 [Nsight Compute Profiling Guide · Workload Durations](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#workload-durations)。

## 调优决策与复测

诊断顺序为“正确性 → 未插桩基线 → 程序时间线 → 单个 Kernel 指标 → 一项改动 → 同口径复测”。工具给出可检验的假设，是否真正变快仍由正确性和基线时间决定。

| 观察 | 下一步核查 | 候选改动与验证 |
| --- | --- | --- |
| 端到端时间远大于 Kernel 时间 | 用 `nsys` 查看分配、复制、同步和 Kernel 间隙。 | 复用设备内存、减少不必要的往返复制或同步，再测端到端时间。 |
| Kernel 内存通路繁忙 | 检查相邻线程地址、sector/request、缓存及 DRAM 吞吐。 | 先调整线程映射；有数据复用时再评估分块或共享内存，并测 Kernel 时间。 |
| SM 忙但内存吞吐不高 | 结合 `SpeedOfLight`、`LaunchStats` 和必要时的 `WarpStateStats` 判断指令、资源或依赖等待。 | 一次只改 Block、每线程工作量或同步方式中的一项；比较编译资源用量和实际耗时。 |

`WarpStateStats` 的等待原因需要结合调度器是否能持续发射指令解读；等待计数本身不等于等量性能损失。Occupancy 是活跃 Warp 占可驻留 Warp 的比例，提高它也不保证缩短时间。相关定义见 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) 和 [CUDA C++ Best Practices Guide · Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#occupancy)。

迁移到自己的算子时，固定四组内容：可执行程序及其真实输入、`-k` 所匹配的 Kernel 名、`-s` 前的同名启动次数、以及与瓶颈假设相对应的章节或指标。改动后仍用相同输入、正确性规则、编译配置和未插桩计时方法比较。下面用一个完整程序展示这一流程。

## RTX 5070 访存映射实测

例子是向量加法 `c[i] = a[i] + b[i]`。两个 Kernel 执行相同计算、覆盖相同元素并使用相同 Grid 和 Block，差异只有线程到元素下标的映射。连续版让一个 Warp 的相邻线程读相邻地址；跨步版让相邻线程相隔 `gridDim.x` 个元素。本例 `gridDim.x = 65,536`，相邻线程的地址相隔 `65,536 × 4 = 262,144` 字节。NVIDIA 的合并访存规则说明了为何需要检查这种映射；性能效果仍由下文实测确定。见 [CUDA C++ Best Practices Guide · Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory)。

### 完整 CUDA C++ 程序

下面是可直接编译的完整程序，也可[下载 `cuda_perf_bench.cu`](../examples/cuda_perf_bench.cu)。两个输入都初始化为 `1.0f`，每个输出应精确等于 `2.0f`。程序先逐元素校验两个版本，再分别热身 5 次、采样 20 次，并打印 Event 时间中位数。

<!-- CUDA_SOURCE -->
```cuda
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t result, const char* step) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

// 两个版本计算相同的 c[i] = a[i] + b[i]，只改变线程到元素的映射。
__global__ void coalesced_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void strided_add(const float* a, const float* b, float* c, int n) {
  int i = threadIdx.x * gridDim.x + blockIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

using Kernel = void (*)(const float*, const float*, float*, int);

static void launch(Kernel kernel, const float* a, const float* b, float* c,
                   int n, int blocks, int threads) {
  kernel<<<blocks, threads>>>(a, b, c, n);
  check(cudaGetLastError(), "kernel launch");
}

static void verify(Kernel kernel, const float* a, const float* b, float* c,
                   int n, int blocks, int threads, std::vector<float>& output) {
  launch(kernel, a, b, c, n, blocks, threads);
  check(cudaDeviceSynchronize(), "kernel execution");
  check(cudaMemcpy(output.data(), c, output.size() * sizeof(float),
                   cudaMemcpyDeviceToHost), "copy result");
  for (int i = 0; i < n; ++i) {
    if (output[i] != 2.0f) {
      std::fprintf(stderr, "mismatch at %d: %.8g\n", i, output[i]);
      std::exit(EXIT_FAILURE);
    }
  }
}

static float median_us(Kernel kernel, const float* a, const float* b, float* c,
                       int n, int blocks, int threads) {
  constexpr int warmup = 5;
  constexpr int repeat = 20;
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "create start event");
  check(cudaEventCreate(&stop), "create stop event");
  for (int i = 0; i < warmup; ++i) launch(kernel, a, b, c, n, blocks, threads);
  check(cudaDeviceSynchronize(), "warmup");

  std::vector<float> times;
  times.reserve(repeat);
  for (int i = 0; i < repeat; ++i) {
    check(cudaEventRecord(start), "record start");
    launch(kernel, a, b, c, n, blocks, threads);
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "synchronize stop");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");
    times.push_back(milliseconds * 1000.0f);
  }
  std::sort(times.begin(), times.end());
  check(cudaEventDestroy(start), "destroy start event");
  check(cudaEventDestroy(stop), "destroy stop event");
  return (times[repeat / 2 - 1] + times[repeat / 2]) / 2.0f;
}

int main() {
  constexpr int n = 1 << 24;
  constexpr int threads = 256;
  constexpr int blocks = (n + threads - 1) / threads;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> input(n, 1.0f), output(n);
  float *a = nullptr, *b = nullptr, *c = nullptr;
  check(cudaMalloc(&a, bytes), "allocate a");
  check(cudaMalloc(&b, bytes), "allocate b");
  check(cudaMalloc(&c, bytes), "allocate c");
  check(cudaMemcpy(a, input.data(), bytes, cudaMemcpyHostToDevice), "copy a");
  check(cudaMemcpy(b, input.data(), bytes, cudaMemcpyHostToDevice), "copy b");

  verify(coalesced_add, a, b, c, n, blocks, threads, output);
  verify(strided_add, a, b, c, n, blocks, threads, output);
  const float contiguous_us = median_us(coalesced_add, a, b, c, n, blocks, threads);
  const float strided_us = median_us(strided_add, a, b, c, n, blocks, threads);
  const double gb = 3.0 * bytes / 1e9;
  std::printf("verified %d elements, block=%d, 5 warmup + 20 samples per kernel\n",
              n, threads);
  std::printf("coalesced: median %.1f us, effective %.1f GB/s\n",
              contiguous_us, gb / (contiguous_us * 1e-6));
  std::printf("strided:   median %.1f us, effective %.1f GB/s\n",
              strided_us, gb / (strided_us * 1e-6));

  check(cudaFree(a), "free a");
  check(cudaFree(b), "free b");
  check(cudaFree(c), "free c");
}
```

| 程序变量或函数 | 含义与改动位置 |
| --- | --- |
| `n`、`threads`、`blocks` | `n = 1 << 24` 为 16,777,216 个元素；每 Block 256 线程；`blocks = (n + threads - 1) / threads` 向上取整以覆盖尾部。换规模时修改 `n` 并重新编译。 |
| `bytes`、`a/b/c` | `bytes = n × sizeof(float)` 为单数组字节数；`a`、`b`、`c` 是三块设备内存。`input/output` 是 Host 侧校验缓冲区。 |
| `coalesced_add`、`strided_add` | 仅改变 `i` 的计算；`if (i < n)` 保证尾部不越界。换自己的算子时保留可比较的输入与输出语义。 |
| `Kernel`、`check`、`launch`、`verify` | `Kernel` 是两个 Kernel 共用的函数指针类型；后三个函数分别检查 CUDA API 返回码、启动 Kernel 并检查启动错误、同步后逐元素核对结果。错误检查失败时程序退出。 |
| `median_us`、`warmup`、`repeat`、`times` | `median_us` 负责 Event 计时；各版本 5 次热身、20 次采样；`times` 排序后取中间两个数的平均作为中位数。改变热身次数时需同步重算 `ncu -s`。 |
| `gb` | 每元素两次读取加一次写入，所以算法字节数为 `3 × bytes`；除以 `10^9` 转成 GB，再除以秒数得到 GB/s。 |

### 编译、校验与未插桩基线

从下载文件所在目录执行下列命令；用仓库源码时先进入 `notes/examples/`。先记录本机环境，再编译。测试期间未手动锁定 GPU 时钟。

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
nvcc --version
```

本次输出确认 NVIDIA GeForce RTX 5070、计算能力 12.0、驱动 595.91.07、显存 12,227 MiB，以及 CUDA Toolkit / `nvcc` 13.4；分析工具版本为 Nsight Systems 2026.3.2、Nsight Compute 2026.3.1。

```bash
nvcc -O3 -std=c++17 -arch=sm_120 cuda_perf_bench.cu -o cuda_perf_bench
./cuda_perf_bench
```

| 编译命令部分 | 含义与替换方法 |
| --- | --- |
| `nvcc` | CUDA C++ 编译器；需在 PATH 中可调用。 |
| `-O3` | 启用较高优化级别；比较版本时保持一致。 |
| `-std=c++17` | 采用 C++17 语法规则；源码需要的标准若不同则修改。 |
| `-arch=sm_120` | 生成 RTX 5070 所用计算能力 12.0 的设备代码；其他 GPU 需查询 `compute_cap` 后替换，且 Toolkit 必须支持该目标。 |
| `cuda_perf_bench.cu` | 输入源码；换成自己的 `.cu` 文件。 |
| `-o cuda_perf_bench` | 指定输出可执行文件名；下一行的 `./cuda_perf_bench` 表示运行当前目录里的该文件。 |

程序的第一行 `verified 16777216 elements` 表示两个实现的每个输出都通过校验。下表是三次独立、未启用分析器的运行；每格为该次运行中 20 个 Event 时间的中位数。

本次大输入恰好是 256 的倍数。为额外验证尾部边界，将源码中的 `n` 暂改为 `1001` 并重新编译，本机得到 `verified 1001 elements`；随后恢复原规模进行性能测量。小输入的时间处于微秒量级，仅用于正确性检查，不与下表的大输入结果比较。

| 运行 | 连续版中位数 | 跨步版中位数 |
| --- | --- | --- |
| 1 | 335.5 µs | 894.1 µs |
| 2 | 335.4 µs | 894.1 µs |
| 3 | 335.3 µs | 894.2 µs |

每个数组有 `16,777,216 × 4 = 67,108,864` 字节，即 64 MiB；三块设备数组合计 192 MiB。算法每次读取 `a`、`b` 并写入 `c`，合计 `201,326,592` 字节。以第一次运行为例，连续版有效带宽约为 `201,326,592 / (335.5 × 10^-6) / 10^9 = 600.1 GB/s`；跨步版约为 `225.2 GB/s`。同一工作负载下，连续版约快 `894.1 / 335.5 = 2.67` 倍。这里比较的是 Kernel 区间；输入复制与结果回传不计入这组时间。

### Nsight Systems 时间线与汇总

在同一个可执行程序上运行前述 `nsys` 命令，报告前缀改为 `/tmp/cuda_doc5070`；第二条命令读取它生成的 SQLite 文件。

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --stats=true -o /tmp/cuda_doc5070 ./cuda_perf_bench
nsys stats --report cuda_gpu_trace /tmp/cuda_doc5070.sqlite
```

| 本次报告项 | 实测内容 | 据此得到的判断 |
| --- | --- | --- |
| `cuda_gpu_kern_sum` | 两个 Kernel 各 26 次启动；连续版平均 331.3 µs，跨步版平均 875.0 µs。 | 26 = 1 次校验 + 5 次热身 + 20 次采样；时间量级与未插桩基线一致，但汇总包含校验和热身。 |
| `cuda_api_sum` | `cudaMalloc` 3 次，`cudaMemcpy` 4 次，`cudaEventSynchronize` 40 次。 | 内存分配、两次输入复制、两次校验回传和每样本同步都出现在程序调用路径；这些 CPU API 总时间不属于 Event 测得的 Kernel 区间。 |
| `cuda_gpu_trace` | 逐条列出两次 Host→Device 复制、校验 Kernel、Device→Host 回传，以及后续热身和采样 Kernel。 | 复制和校验确实位于计时循环之前；时间线可核对测量范围。 |

例如 `cuda_gpu_trace` 中两次 Host→Device 复制各约 3.56–4.06 ms，而采样的连续版 Kernel 约 0.33 ms。若应用每次调用都要重新复制输入，端到端优化重点会不同；本例刻意测量输入已经驻留设备后的 Kernel。分析器运行中的程序也打印了 Event 时间，但最终版本比较使用上面的未插桩三次运行。

### Nsight Compute 吞吐与访存计数

两个版本分别采集一次。每次运行程序时，`-k` 只匹配其中一个 Kernel；该 Kernel 的第 1 次启动用于校验，第 2–6 次用于热身，所以 `-s 6 -c 1` 选中第 7 次启动。

```bash
ncu --list-sections
ncu --query-metrics --query-metrics-mode all
```

本机列表包含 `SpeedOfLight`、`MemoryWorkloadAnalysis`，也包含后面使用的 `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` 与 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`。完成名称核对后运行下列采集命令。

```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
  -k regex:coalesced_add -s 6 -c 1 ./cuda_perf_bench
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
  -k regex:strided_add -s 6 -c 1 ./cuda_perf_bench
```

| Kernel | DRAM Throughput | L2 Cache Throughput | Compute (SM) Throughput |
| --- | --- | --- | --- |
| `coalesced_add` | 91.67% | 27.18% | 17.97% |
| `strided_add` | 26.78% | 88.05% | 4.82% |

连续版在这次分析器采集中表现为高 DRAM 吞吐；跨步版则显示较高的 L2 吞吐，提示两者对存储层次施加了不同压力。为了直接检查访问是否合并，再运行与前文相同的两个指标，第二次只改 `-k` 后的 Kernel 名。

```bash
ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  -k regex:coalesced_add -s 6 -c 1 ./cuda_perf_bench
ncu --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  -k regex:strided_add -s 6 -c 1 ./cuda_perf_bench
```

| Kernel | 加载请求数 | 加载 sector 数 | sector / 请求 |
| --- | --- | --- | --- |
| `coalesced_add` | 1,048,576 | 4,194,304 | 4 |
| `strided_add` | 1,048,576 | 33,554,432 | 32 |

两个实现执行相同数量的全局加载请求；连续版每请求平均 4 个 sector，符合 32 个活跃线程读取连续 `float` 的预期，跨步版每请求平均 32 个 sector，说明相邻线程触及不同片段。这组计数器给出了访存映射影响加载请求效率的直接证据。写回也使用同一线程映射，因此本次减速不能全部归因于加载。加载 sector 数相差 8 倍，未插桩时间约相差 2.67 倍；缓存、硬件并行性和其他开销使二者不能按同一比例换算。`ncu` 的 Duration 也不用于计算这 2.67 倍加速。

这套命令迁移到归约、GEMV 或融合算子时，先替换源码与可执行文件，再用时间线确认 Kernel 名和调用顺序，最后按已观察到的瓶颈选择章节或指标。只有在输入、正确性规则和计时范围不变时，不同版本的数字才具有可比性。
