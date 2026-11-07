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
    title: "为什么需要并行：五种切法与一张通信地图",
    description: "从显存与算力的两道墙出发，用数据并行、张量并行、流水线并行、专家并行和上下文并行解释大模型怎样切到多张 GPU 上。",
    deck: "第 1 篇 · 并行维度、通信原语与阅读顺序",
  },
  {
    stem: "02-data-parallel",
    title: "数据并行：从 DDP 到 ZeRO 与 FSDP",
    description: "数据并行怎样复制模型、切分 batch 并同步梯度；ZeRO-1/2/3 与 FSDP 又怎样把参数、梯度和优化器状态分片。",
    deck: "第 2 篇 · 复制模型，切分数据，分摊训练状态",
  },
  {
    stem: "03-tensor-parallel",
    title: "张量并行：在同一层内切开矩阵",
    description: "Megatron-LM 的列并行与行并行、Attention 的按头切分、每层四次 All-Reduce，以及张量并行为什么通常留在 NVLink 域内。",
    deck: "第 3 篇 · 列并行、行并行与每层通信账本",
  },
  {
    stem: "04-pipeline-parallel",
    title: "流水线并行：把层分段，让 micro-batch 流动",
    description: "GPipe、1F1B 与交错式流水线的时间线、气泡公式、激活值显存和跨节点放置原则。",
    deck: "第 4 篇 · 气泡、micro-batch 与调度演进",
  },
  {
    stem: "05-expert-parallel",
    title: "专家并行：MoE 的 All-to-All 与负载均衡",
    description: "MoE 怎样把专家分散到多卡、token 怎样经 All-to-All 路由、DeepSeek-V3 的 EPLB 与 DeepEP 在解决什么问题。",
    deck: "第 5 篇 · 稀疏激活、All-to-All 与专家放置",
  },
  {
    stem: "06-context-parallel",
    title: "序列与上下文并行：把长序列切开",
    description: "Megatron 序列并行、Ring Attention 与 DeepSpeed Ulysses 的区别，长上下文训练中激活显存和通信量的取舍。",
    deck: "第 6 篇 · 序列切分、环形注意力和头维 All-to-All",
  },
  {
    stem: "07-combination",
    title: "组合与选型：3D/4D 并行怎样落到集群",
    description: "把 DP、TP、PP、EP、CP 组合成 world size，解释训练与推理的不同选型，并给出从单卡到集群的决策顺序。",
    deck: "第 7 篇 · 组合公式、真实配置与排障顺序",
  },
  {
    stem: "08-for-readers",
    title: "写给博客读者：大模型并行到底在解决什么",
    description: "用餐厅、医院和长文章的类比，讲清五种并行各自解决什么问题，以及一条训练配置里的数字该怎么读。",
    deck: "第 8 篇 · 给非 infra 读者的通俗总结",
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

function renderChapter(chapter, index) {
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
  const series = chapters.map((item, itemIndex) =>
    '<li' + (itemIndex === index ? ' class="is-current"' : '') + '><a href="' +
    item.stem + '.html"' + (itemIndex === index ? ' aria-current="page"' : '') + '>' +
    escapeHtml(item.title) + '</a></li>'
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
    '            <details class="chapter-panel series-panel" open><summary>本系列</summary>',
    '              <nav aria-label="并行专题文章列表"><ol class="series-list">',
    series,
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
  const html = renderChapter(chapter, index);
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
