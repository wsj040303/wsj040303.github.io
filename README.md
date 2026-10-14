# wsj040303.github.io
my notes-especially for ai-infra

## 论文公式

论文报告的数学符号和行内表达式写作 `$...$`，需要独立展示的公式写作 `$$` 块；重要公式可在 TeX 末尾加 `\tag{1}` 等编号。反引号仅用于代码、命令和配置名。构建脚本使用 KaTeX 将数学 Markdown 预渲染成 HTML，网页使用 `assets/math/` 中的本地样式与字体。修改任一 `papers/*/article.md` 后，运行对应的 `npm run build:*` 和 `npm run check:*`。

## 编辑 Tensor Comprehensions 阅读报告

正文源文件是 [papers/tensor-comprehensions/article.md](papers/tensor-comprehensions/article.md)，图片放在同目录的 images/。修改 Markdown 后，通知 Codex 检查并发布；博客页面 index.html 会由 Markdown 生成。

本地生成与核对命令：npm ci、npm run build:tc、npm run check:tc。只修改 Markdown 不会更新已发布的 HTML 正文。

## 编辑 SGLang 阅读报告

正文源文件是 [papers/sglang/article.md](papers/sglang/article.md)。修改后运行 `npm run build:sglang` 生成页面，再运行 `npm run check:sglang` 核对 HTML 与 Markdown 一致。

## 编辑 PagedAttention 阅读报告

正文源文件是 [papers/pagedattention/article.md](papers/pagedattention/article.md)，图片放在同目录的 `images/`。修改后运行 `npm run build:pagedattention` 生成页面，再运行 `npm run check:pagedattention` 核对 HTML 与 Markdown 一致。

## 编辑 CUDA 性能测试与调优

正文源文件是 [notes/cuda/performance.md](notes/cuda/performance.md)。直接修改段落、表格和代码块；标有 `<!-- CUDA_SOURCE -->` 的完整 CUDA 代码块也会生成下载用的 `notes/examples/cuda_perf_bench.cu`，因此只需改 Markdown 一份。

修改后通知 Codex 检查并上线。若要自己预览生成结果，在仓库根目录执行 `npm ci`、`npm run build:cuda`，再打开 `notes/cuda/performance.html`；执行 `npm run check:cuda` 可检查 HTML 和 `.cu` 是否与 Markdown 一致。只修改 Markdown 不会更新已发布页面。

## 编辑 vAttention 阅读报告

正文源文件是 [papers/vattention/article.md](papers/vattention/article.md)，原文图放在同目录的 images/。修改后运行 `npm run build:vattention` 生成 HTML，再运行 `npm run check:vattention` 核对。

## 编辑 AWQ 阅读报告

正文源文件是 [papers/awq/article.md](papers/awq/article.md)，原文图放在同目录的 `images/`。修改后运行 `npm run build:awq` 生成 HTML，再运行 `npm run check:awq` 核对。

## 编辑 FlashAttention-2 阅读报告

正文源文件是 [papers/flashattention-2/article.md](papers/flashattention-2/article.md)，原文图放在同目录的 `images/`。修改后运行 `npm run build:flashattention2` 生成 HTML，再运行 `npm run check:flashattention2` 核对。

## 编辑推理引擎编写专题

Nano-vLLM 解读的正文源文件是 [第 1 章](notes/inference/nano-vllm/01-overview.md)、[llm_engine.py 解析](notes/inference/nano-vllm/02-llm-engine.md)、[sequence.py 解析](notes/inference/nano-vllm/03-sequence.md) 和 [scheduler.py 解析](notes/inference/nano-vllm/04-scheduler.md)，组件示意图放在同目录的 `images/`。编辑 Markdown 后，运行 `npm run build:inference` 生成对应 HTML；运行 `npm run check:inference` 检查生成页是否与 Markdown 一致。专题入口位于 `ai-infra/index.html → notes/inference/index.html → notes/inference/nano-vllm/index.html`。
