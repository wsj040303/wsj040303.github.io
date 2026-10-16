import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const articleDir = path.join(repo, "notes", "inference", "nano-vllm");
const checkOnly = process.argv.includes("--check");
const chapters = [
  {
    stem: "01-overview",
    description: "Nano-vLLM 项目概览、组件思维导图与源码阅读顺序。",
    deck: "第 1 章 · 项目范围、请求路径与阅读顺序",
    previous: null,
    next: { href: "02-llm-engine.html", text: "下一章：llm_engine.py" },
  },
  {
    stem: "02-llm-engine",
    sourceFile: "llm_engine.py",
    description: "llm_engine.py 完整注释代码及初始化、请求入队、单轮执行与批量生成的源码解析。",
    deck: "第 2 章 · 完整注释代码与执行路径",
    previous: { href: "01-overview.html", text: "上一章：项目概览" },
    next: { href: "03-sequence.html", text: "下一章：sequence.py" },
  },
  {
    stem: "03-sequence",
    sourceFile: "sequence.py",
    description: "sequence.py 完整注释代码及请求状态、token 计数、KV 块视图与跨进程序列化的源码解析。",
    deck: "第 3 章 · 完整注释代码与请求状态",
    previous: { href: "02-llm-engine.html", text: "上一章：llm_engine.py" },
    next: { href: "04-scheduler.html", text: "下一章：scheduler.py" },
  },
  {
    stem: "04-scheduler",
    sourceFile: "scheduler.py",
    description: "scheduler.py 完整注释代码及 KV 块抢占、共享前缀、分块 prefill 和运行队列操作的源码解析。",
    deck: "第 4 章 · 完整注释代码与调度决策",
    previous: { href: "03-sequence.html", text: "上一章：sequence.py" },
    next: { href: "05-block-manager.html", text: "下一章：block_manager.py" },
  },
  {
    stem: "05-block-manager",
    sourceFile: "block_manager.py",
    description: "block_manager.py 完整注释代码及引用计数、前缀哈希和 KV 块边界的源码解析。",
    deck: "第 5 章 · 完整注释代码与 KV 块管理",
    previous: { href: "04-scheduler.html", text: "上一章：scheduler.py" },
    next: { href: "06-model-runner.html", text: "下一章：model_runner.py" },
  },
  {
    stem: "06-model-runner",
    sourceFile: "model_runner.py",
    description: "model_runner.py 完整注释代码及执行模式、进程消息、KV 显存预算与 CUDA Graph 的加速和档位取舍的源码解析。",
    deck: "第 6 章 · 完整注释代码与模型执行",
    previous: { href: "05-block-manager.html", text: "上一章：block_manager.py" },
    next: { href: "07-qwen3.html", text: "下一章：qwen3.py" },
  },
  {
    stem: "07-qwen3",
    sourceFile: "qwen3.py",
    description: "qwen3.py 完整注释代码、Qwen3 Dense 整体与单层结构图，以及注意力展平、残差传递和合并权重加载的源码解析。",
    deck: "第 7 章 · 完整注释代码与 Qwen3 模型结构",
    previous: { href: "06-model-runner.html", text: "上一章：model_runner.py" },
    next: null,
  },
];

function escapeHtml(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

const pythonKeywords = new Set([
  "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
  "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
  "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
  "with", "yield",
]);
const pythonBuiltins = new Set(["bool", "dict", "int", "isinstance", "len", "list", "next", "range", "str", "sum", "zip"]);

function highlightPython(escapedCode) {
  const code = escapedCode.replace(/&(?:amp|lt|gt|quot|#39|#x27);/g, (entity) => ({
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&#x27;": "'",
  })[entity]);
  const span = (kind, value) => '<span class="tok-' + kind + '">' + escapeHtml(value) + '</span>';
  let result = "";
  let i = 0;
  while (i < code.length) {
    const start = i;
    const char = code[i];
    if (char === "#") {
      while (i < code.length && code[i] !== "\n") i++;
      result += span("comment", code.slice(start, i));
    } else if (char === '"' || char === "'") {
      const quote = code.startsWith(char.repeat(3), i) ? char.repeat(3) : char;
      i += quote.length;
      while (i < code.length) {
        if (code[i] === "\\") {
          i += Math.min(2, code.length - i);
        } else if (code.startsWith(quote, i)) {
          i += quote.length;
          break;
        } else {
          i++;
        }
      }
      result += span("string", code.slice(start, i));
    } else if (/[A-Za-z_]/.test(char)) {
      i++;
      while (i < code.length && /[A-Za-z_0-9]/.test(code[i])) i++;
      const word = code.slice(start, i);
      const kind = pythonKeywords.has(word) ? "keyword" :
        ["True", "False", "None"].includes(word) ? "constant" :
          pythonBuiltins.has(word) ? "builtin" : "";
      result += kind ? span(kind, word) : escapeHtml(word);
    } else if (/[0-9]/.test(char)) {
      i++;
      while (i < code.length && /[0-9_.]/.test(code[i])) i++;
      result += span("number", code.slice(start, i));
    } else {
      result += escapeHtml(char);
      i++;
    }
  }
  return result;
}

function renderMarkdown(input, sourceFile = "") {
  const prepared = input.trim();
  if (!prepared) return "";
  let html = marked.parse(prepared, { gfm: true });
  html = html.replaceAll("<h3>", "<h4>").replaceAll("</h3>", "</h4>");
  html = html.replace(/<pre><code(?: class="language-([^"]+)")?>([\s\S]*?)<\/code><\/pre>/g,
    (_, language, content) => {
      const label = language === "python" ? "Python" : language || "Code";
      const complete = Boolean(sourceFile) && content.split("\n").length > 70;
      const className = complete ? "engine-code is-complete" : "engine-code";
      const caption = complete ? "完整代码 · " + sourceFile : "代码片段";
      return '<figure class="' + className + '"><figcaption>' + escapeHtml(caption) + ' <span>' +
        escapeHtml(label) + '</span></figcaption><pre><code>' +
        (language === "python" ? highlightPython(content.trimEnd()) : content.trimEnd()) + '</code></pre></figure>';
    });
  html = html.replaceAll("<table>", '<div class="table-scroll"><table>')
    .replaceAll("</table>", "</table></div>");
  html = html.replace(/<a href="([^"]+)">/g, (_, href) => {
    const external = /^https?:\/\//.test(href);
    const attrs = external ? ' target="_blank" rel="noopener noreferrer"' : "";
    return '<a class="inline-link" href="' + href + '"' + attrs + '>';
  });
  html = html.replace(/<p><img src="([^"]+)" alt="([^"]*)"><\/p>/g, (_, src, alt) => {
    if (!fs.existsSync(path.join(articleDir, src))) {
      throw new Error("图像文件不存在：" + src);
    }
    return '<figure class="engine-figure"><a href="' + src +
      '" target="_blank" rel="noopener noreferrer" aria-label="打开完整示意图">' +
      '<img src="' + src + '" alt="' + alt + '"></a><figcaption>' +
      alt + ' 点击可打开完整图片。</figcaption></figure>';
  });
  return html.trimEnd();
}

function renderChapter(chapter) {
  const sourcePath = path.join(articleDir, chapter.stem + ".md");
  const source = fs.readFileSync(sourcePath, "utf8");
  const titleMatch = source.match(/^# (.+)$/m);
  if (!titleMatch) throw new Error(sourcePath + " 必须以一级标题开头");
  const title = titleMatch[1];
  const markdown = source.slice(titleMatch[0].length).trimStart();
  const matches = [...markdown.matchAll(/^## (.+)$/gm)];
  if (!matches.length) throw new Error(sourcePath + " 至少需要一个二级标题");
  const sections = matches.map((match, index) => ({
    name: match[1],
    id: "section-" + (index + 1),
    start: match.index,
    bodyStart: match.index + match[0].length,
    end: index + 1 < matches.length ? matches[index + 1].index : markdown.length,
  }));
  const intro = renderMarkdown(markdown.slice(0, sections[0].start), chapter.sourceFile);
  const body = [
    intro,
    ...sections.map((section) => '<section id="' + section.id + '"><h3>' +
      escapeHtml(section.name) + '</h3>\n' +
      renderMarkdown(markdown.slice(section.bodyStart, section.end)) + '\n</section>'),
  ].filter(Boolean).join("\n");
  const toc = sections.map((section) =>
    '<li><a href="#' + section.id + '">' + escapeHtml(section.name) + '</a></li>'
  ).join("\n");
  const actions = [
    '<a href="index.html">返回 Nano-vLLM 章节目录</a>',
    chapter.previous ? '<a href="' + chapter.previous.href + '">' + chapter.previous.text + '</a>' : "",
    chapter.next ? '<a href="' + chapter.next.href + '">' + chapter.next.text + '</a>' : "",
  ].filter(Boolean).join("\n");

  return [
    '<!doctype html>',
    '<html lang="zh-CN">',
    '  <head>',
    '    <meta charset="utf-8">',
    '    <meta name="viewport" content="width=device-width, initial-scale=1">',
    '    <meta name="description" content="' + escapeHtml(chapter.description) + '">',
    '    <meta name="theme-color" content="#0b2015">',
    '    <title>' + escapeHtml(title) + ' · WSJ</title>',
    '    <link rel="stylesheet" href="../../../assets/site.css?v=20261008-layout">',
    '    <link rel="stylesheet" href="../../../assets/garden.css">',
    '    <link rel="stylesheet" href="../../../assets/inference.css?v=20261014-reader-layout">',
    '    <script src="../../../assets/site.js" defer></script>',
    '  </head>',
    '  <body>',
    '    <a class="skip-link" href="#main">跳转到内容</a>',
    '    <div class="site-frame">',
    '      <header class="site-header">',
    '        <a class="brand" href="../../../index.html" aria-label="WSJ 首页">WSJ<span>.</span></a>',
    '        <button class="nav-toggle" type="button" aria-controls="site-nav" aria-expanded="false">菜单 <span aria-hidden="true">☰</span></button>',
    '        <nav class="site-nav" id="site-nav" aria-label="主导航">',
    '          <a href="../../../index.html">首页</a><a href="../../../ai-infra/index.html" aria-current="page">AI Infra</a><a href="../../../papers/index.html">论文阅读</a><a href="../../../projects/index.html">GitHub 项目</a><a href="../../../resources/index.html">推荐资源</a>',
    '        </nav>',
    '        <a class="star-link" href="https://github.com/wsj040303/wsj040303.github.io" target="_blank" rel="noopener noreferrer" aria-label="打开 GitHub 仓库点 Star">',
    '          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 2.5 2.9 6 6.6 1-4.8 4.7 1.1 6.6-5.8-3.1-5.8 3.1 1.1-6.6-4.8-4.7 6.6-1z"/></svg>',
    '          <span>Star on GitHub</span>',
    '        </a>',
    '      </header>',
    '      <main class="reader-main" id="main">',
    '        <nav class="breadcrumb" aria-label="面包屑导航"><a href="../../../index.html">首页</a><span>/</span><a href="../../../ai-infra/index.html">AI Infra</a><span>/</span><a href="../index.html">推理引擎编写</a><span>/</span><a href="index.html">Nano-vLLM 解读</a><span>/</span>' + escapeHtml(title) + '</nav>',
    '        <div class="reader-heading"><h1>' + escapeHtml(title) + '</h1><p class="article-deck">' + escapeHtml(chapter.deck) + '</p></div>',
    '        <div class="reader-grid">',
    '          <aside class="chapter-sidebar">',
    '            <details class="chapter-panel" open><summary>本文章节</summary>',
    '              <nav aria-label="本文章节目录"><ol class="chapter-list">',
    toc,
    '              </ol></nav>',
    '            </details>',
    '          </aside>',
    '          <article class="reader-content engine-article">',
    body,
    '            <div class="reader-actions">' + actions + '</div>',
    '          </article>',
    '        </div>',
    '      </main>',
    '    </div>',
    '  </body>',
    '</html>',
    '',
  ].join("\n");
}

let stale = false;
for (const chapter of chapters) {
  const pagePath = path.join(articleDir, chapter.stem + ".html");
  const html = renderChapter(chapter);
  if (checkOnly) {
    if (!fs.existsSync(pagePath) || fs.readFileSync(pagePath, "utf8") !== html) {
      console.error(pagePath + " 与 Markdown 源文件不一致；运行 npm run build:inference");
      stale = true;
    }
  } else {
    fs.writeFileSync(pagePath, html);
    console.log("已生成 " + pagePath);
  }
}
if (stale) process.exitCode = 1;
