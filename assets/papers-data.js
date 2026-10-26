// Edit this list to add reading records. Sample entries are public papers, not personal reading claims.
window.PAPER_CATEGORIES = [
  { id: "evaluation", name: "评测与可靠性" },
  { id: "training", name: "训练与对齐" },
  { id: "inference", name: "推理引擎" },
  { id: "operators", name: "算子与内存" },
  { id: "compiler", name: "编译与编程模型" },
  { id: "distributed", name: "分布式与通信" }
];

window.PAPER_ENTRIES = [
  {
    title: "Can You Trust the Confidence? ConfBench for Vision-Language Models on Document Extraction",
    shortName: "ConfBench：文档抽取置信度",
    aliases: ["置信度校准", "VLM", "文档抽取", "OCR", "AUROC", "ECARB", "first-token"],
    category: "evaluation",
    year: "2026",
    venue: "arXiv",
    summary: "结论先行，围绕输入模态、模型能力、绝对校准与首 token 排序的四项发现解读文档抽取置信度。",
    articleUrl: "confbench/",
    url: "https://arxiv.org/abs/2608.01792v1"
  },
  {
    title: "PiKV: KV Cache Management System for Mixture of Experts",
    shortName: "PiKV",
    aliases: ["MoE", "专家分片", "KV 缓存", "稀疏路由", "缓存调度", "FPGA"],
    category: "inference",
    year: "2025",
    venue: "ES-FoMo III workshop / arXiv",
    summary: "聚焦路由、压缩和调度，解释三者如何减少 KV 存储与搬运开销。",
    articleUrl: "pikv/",
    url: "https://arxiv.org/abs/2508.06526"
  },
  {
    title: "Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A",
    shortName: "聊天模型概率校准与正确性预测",
    aliases: ["MSP", "AUROC", "ECE", "calibration", "概率校准", "正确性预测", "选择性弃权"],
    category: "evaluation",
    year: "2025",
    venue: "TMLR",
    summary: "从答案 token 的概率解释校准与正确性排序的差别，结合原图、算例和统计证据分析选择性弃权。",
    articleUrl: "chat-llm-calibration/",
    url: "https://arxiv.org/abs/2402.13213v4"
  },
  {
    title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
    shortName: "DeepSeek-R1",
    aliases: ["DeepSeek-R1-Zero", "GRPO", "组相对策略优化", "长链推理", "强化学习"],
    category: "training",
    year: "2025",
    venue: "arXiv",
    rating: 4.5,
    summary: "从论文的训练问题出发，解释 GRPO、规则与模型奖励、语言混合及实验证据。",
    articleUrl: "deepseek-r1/",
    url: "https://arxiv.org/abs/2501.12948"
  },
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
