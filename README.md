# wsj040303.github.io
my notes-especially for ai-infra

## 编辑 Tensor Comprehensions 阅读报告

正文源文件是 [papers/tensor-comprehensions/article.md](papers/tensor-comprehensions/article.md)，图片放在同目录的 images/。修改 Markdown 后，通知 Codex 检查并发布；博客页面 index.html 会由 Markdown 生成。

本地生成与核对命令：npm ci、npm run build:tc、npm run check:tc。只修改 Markdown 不会更新已发布的 HTML 正文。

## 编辑 CUDA 性能测试与调优

正文源文件是 [notes/cuda/performance.md](notes/cuda/performance.md)。直接修改段落、表格和代码块；标有 `<!-- CUDA_SOURCE -->` 的完整 CUDA 代码块也会生成下载用的 `notes/examples/cuda_perf_bench.cu`，因此只需改 Markdown 一份。

修改后通知 Codex 检查并上线。若要自己预览生成结果，在仓库根目录执行 `npm ci`、`npm run build:cuda`，再打开 `notes/cuda/performance.html`；执行 `npm run check:cuda` 可检查 HTML 和 `.cu` 是否与 Markdown 一致。只修改 Markdown 不会更新已发布页面。
