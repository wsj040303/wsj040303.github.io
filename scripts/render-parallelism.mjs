import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const articleDir = path.join(repo, "notes", "parallelism");
const checkOnly = process.argv.includes("--check");

const chapters = [
  {
    stem: "01-overview",
    title: "大模型并行的五个维度与通信约束",
    description: "数据并行、张量并行、流水线并行、专家并行和上下文并行分别切分什么，使用哪些集合通信原语，以及各维度与网络拓扑的对应关系。",
    deck: "第 1 篇 · 五种维度、通信原语与拓扑约束",
  },
  {
    stem: "02-data-parallel",
    title: "数据并行：模型复制、梯度同步与状态分片",
    description: "DDP 的梯度同步、训练状态的显存构成、ZeRO-1/2/3 的分片阶段与 FSDP 的按层参数聚合。",
    deck: "第 2 篇 · DDP、ZeRO 与 FSDP 的显存取舍",
  },
  {
    stem: "03-tensor-parallel",
    title: "张量并行：层内切分、通信模式与适用边界",
    description: "线性层的列并行与行并行、MLP 与 Attention 的通信位置、每层两次 All-Reduce，以及序列并行与词表并行的配套约束。",
    deck: "第 3 篇 · 列并行、行并行与每层通信",
  },
  {
    stem: "04-pipeline-parallel",
    title: "流水线并行：调度、气泡与激活值显存",
    description: "GPipe、1F1B 与交错式流水线的时序、气泡公式、激活值显存阶数，以及 stage 划分与跨节点放置。",
    deck: "第 4 篇 · GPipe、1F1B 与交错式调度",
  },
  {
    stem: "05-expert-parallel",
    title: "专家并行：MoE 路由、All-to-All 与负载均衡",
    description: "MoE 层的路由与专家结构、Dispatch/Combine 两次 All-to-All、EPLB 的冗余专家放置，以及 DeepSeek-V3 的训练与推理配置。",
    deck: "第 5 篇 · 稀疏激活、专家放置与通信",
  },
  {
    stem: "06-context-parallel",
    title: "上下文并行：长序列切分与注意力通信",
    description: "Megatron 序列并行与长序列上下文并行的区别、Ring Attention 的 KV 传递、Ulysses 的 All-to-All 重排，以及 causal mask 的负载均衡。",
    deck: "第 6 篇 · Ring Attention、Ulysses 与 Megatron CP",
  },
  {
    stem: "07-combination",
    title: "并行组合与选型：约束、配置与排障",
    description: "DP、TP、PP、EP、CP 的组合公式、通信成本与拓扑匹配、GPT-3/MT-NLG/DeepSeek-V3 的公开配置，以及配置排障顺序。",
    deck: "第 7 篇 · 组合公式、真实配置与排障",
  },
].map((chapter, index, all) => ({
  ...chapter,
  previous: index > 0 ? { href: all[index - 1].stem + ".html", text: "上一篇：" + all[index - 1].title } : null,
  next: index + 1 < all.length ? { href: all[index + 1].stem + ".html", text: "下一篇：" + all[index + 1].title } : null,
}));

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

function renderMarkdown(input) {
  const prepared = input.trim();
  if (!prepared) return "";
  let html = marked.parse(prepared, { gfm: true });
  html = html.replaceAll("<h3>", "<h4>").replaceAll("</h3>", "</h4>");
  html = html.replace(/<pre><code(?: class="language-([^"]+)")?>([\s\S]*?)<\/code><\/pre>/g,
    (_, language, content) => {
      const label = language === "python" ? "Python" : language === "bash" ? "Shell" : language || "Code";
      return '<figure class="engine-code"><figcaption>代码片段 <span>' + escapeHtml(label) +
        '</span></figcaption><pre><code>' +
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
  const sections = matches.map((match, sectionIndex) => ({
    name: match[1],
    id: "section-" + (sectionIndex + 1),
    start: match.index,
    bodyStart: match.index + match[0].length,
    end: sectionIndex + 1 < matches.length ? matches[sectionIndex + 1].index : markdown.length,
  }));
  const intro = renderMarkdown(markdown.slice(0, sections[0].start));
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
    '<a href="index.html">返回并行专题目录</a>',
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
    '    <link rel="stylesheet" href="../../assets/site.css?v=20261008-layout">',
    '    <link rel="stylesheet" href="../../assets/garden.css">',
    '    <link rel="stylesheet" href="../../assets/inference.css?v=20261014-reader-layout">',
    '    <link rel="stylesheet" href="../../assets/parallelism.css">',
    '    <script src="../../assets/site.js" defer></script>',
    '  </head>',
    '  <body>',
    '    <a class="skip-link" href="#main">跳转到内容</a>',
    '    <div class="site-frame">',
    '      <header class="site-header">',
    '        <a class="brand" href="../../index.html" aria-label="WSJ 首页">WSJ<span>.</span></a>',
    '        <button class="nav-toggle" type="button" aria-controls="site-nav" aria-expanded="false">菜单 <span aria-hidden="true">☰</span></button>',
    '        <nav class="site-nav" id="site-nav" aria-label="主导航">',
    '          <a href="../../index.html">首页</a><a href="../../ai-infra/index.html" aria-current="page">AI Infra</a><a href="../../papers/index.html">论文阅读</a><a href="../../projects/index.html">GitHub 项目</a><a href="../../resources/index.html">推荐资源</a>',
    '        </nav>',
    '        <a class="star-link" href="https://github.com/wsj040303/wsj040303.github.io" target="_blank" rel="noopener noreferrer" aria-label="打开 GitHub 仓库点 Star">',
    '          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 2.5 2.9 6 6.6 1-4.8 4.7 1.1 6.6-5.8-3.1-5.8 3.1 1.1-6.6-4.8-4.7 6.6-1z"/></svg>',
    '          <span>Star on GitHub</span>',
    '        </a>',
    '      </header>',
    '      <main class="reader-main" id="main">',
    '        <nav class="breadcrumb" aria-label="面包屑导航"><a href="../../index.html">首页</a><span>/</span><a href="../../ai-infra/index.html">AI Infra</a><span>/</span><a href="index.html">并行计算</a><span>/</span>' + escapeHtml(title) + '</nav>',
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
chapters.forEach((chapter, index) => {
  const pagePath = path.join(articleDir, chapter.stem + ".html");
  const html = renderChapter(chapter);
  if (checkOnly) {
    if (!fs.existsSync(pagePath) || fs.readFileSync(pagePath, "utf8") !== html) {
      console.error(pagePath + " 与 Markdown 源文件不一致；运行 npm run build:parallelism");
      stale = true;
    }
  } else {
    fs.writeFileSync(pagePath, html);
    console.log("已生成 " + pagePath);
  }
});
if (stale) process.exitCode = 1;
