// Edit this list to add reading records. Sample entries are public papers, not personal reading claims.
window.PAPER_CATEGORIES = [
  { id: "inference", name: "推理引擎" },
  { id: "operators", name: "算子与内存" },
  { id: "compiler", name: "编译与编程模型" },
  { id: "distributed", name: "分布式与通信" }
];

window.PAPER_ENTRIES = [
  {
    title: "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration",
    shortName: "AWQ",
    aliases: ["激活感知权重量化", "TinyChat", "W4A16", "权重低比特量化"],
    category: "inference",
    year: "2024",
    venue: "MLSys",
    rating: 3.5,
    summary: "解释激活感知缩放、量化误差条件与 TinyChat 的设备端加速证据。",
    articleUrl: "awq/",
    url: "https://arxiv.org/abs/2306.00978"
  },
  {
    title: "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention",
    shortName: "vAttention",
    aliases: ["连续虚拟 KV 缓存", "CUDA VMM", "动态显存管理"],
    category: "inference",
    year: "2025",
    venue: "ASPLOS",
    rating: 4.5,
    summary: "用连续虚拟张量和按需物理页映射管理 KV 缓存，并核对长上下文性能证据。",
    articleUrl: "vattention/",
    url: "https://arxiv.org/abs/2405.04437"
  },
  {
    title: "SGLang: Efficient Execution of Structured Language Model Programs",
    shortName: "SGLang",
    aliases: ["SGLang: Efficient Execution of Structured Language Model Programs", "RadixAttention", "结构化语言模型程序", "压缩有限状态机"],
    category: "inference",
    year: "2024",
    venue: "NeurIPS",
    rating: 3.5,
    summary: "从多调用程序出发，解释 KV 缓存复用、约束解码和 API 推测执行。",
    articleUrl: "sglang/",
    url: "https://arxiv.org/abs/2312.07104"
  },
  {
    title: "Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions",
    shortName: "Tensor Comprehensions / TC",
    aliases: ["Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions", "张量理解", "算子自动调优", "张量编译器"],
    category: "compiler",
    year: "2018",
    venue: "FAIR 技术报告",
    rating: 2.5,
    summary: "从张量表达式、多面体 GPU 编译和实测调优入手，建立算子自动调优的基础概念。",
    articleUrl: "tensor-comprehensions/",
    url: "https://arxiv.org/abs/1802.04730"
  },
  {
    title: "Efficient Memory Management for Large Language Model Serving with PagedAttention",
    shortName: "vLLM / PagedAttention",
    aliases: ["大模型推理显存管理", "KV 缓存", "分页注意力", "写时复制"],
    category: "inference",
    year: "2023",
    venue: "SOSP",
    rating: 4.5,
    summary: "解析 KV 缓存分页、块表共享与抢占调度，并核对吞吐实验的适用边界。",
    articleUrl: "pagedattention/",
    url: "https://arxiv.org/abs/2309.06180"
  },
  {
    title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
    shortName: "FlashAttention-2",
    aliases: ["FA2", "FlashAttention 2", "序列并行注意力", "Warp 工作划分"],
    category: "operators",
    year: "2023",
    venue: "arXiv",
    summary: "解释在线 Softmax 改写、序列维度并行、反向梯度归约与 Warp 分工，并核对 A100 实验。",
    articleUrl: "flashattention-2/",
    url: "https://arxiv.org/abs/2307.08691"
  },
  {
    title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
    shortName: "FlashAttention",
    aliases: ["闪存注意力"],
    category: "operators",
    year: "2022",
    venue: "NeurIPS",
    rating: 5,
    summary: "从分块、在线 Softmax 和反向重算出发，核对精确注意力的 IO 复杂度与实验结果。",
    articleUrl: "flashattention/",
    url: "https://arxiv.org/abs/2205.14135"
  },
  {
    title: "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations",
    shortName: "Triton",
    aliases: ["张量块编译器"],
    category: "compiler",
    year: "2019",
    venue: "MAPL",
    summary: "从张量块编程模型出发，连接自定义算子与编译器设计。",
    url: "https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf",
    sample: true
  }
];
