// Edit this list to add reading records. Sample entries are public papers, not personal reading claims.
window.PAPER_CATEGORIES = [
  { id: "inference", name: "推理引擎" },
  { id: "operators", name: "算子与内存" },
  { id: "compiler", name: "编译与编程模型" },
  { id: "distributed", name: "分布式与通信" }
];

window.PAPER_ENTRIES = [
  {
    title: "Tensor Comprehensions 论文阅读报告",
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
    aliases: ["大模型推理显存管理"],
    category: "inference",
    year: "2023",
    venue: "SOSP",
    summary: "从 KV Cache 的显存管理入手，理解大模型推理服务的吞吐问题。",
    url: "https://arxiv.org/abs/2309.06180",
    sample: true
  },
  {
    title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
    shortName: "FlashAttention",
    aliases: ["闪存注意力"],
    category: "operators",
    year: "2022",
    venue: "NeurIPS",
    summary: "用分块和 IO 开销的视角理解注意力算子的实现。",
    url: "https://arxiv.org/abs/2205.14135",
    sample: true
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
