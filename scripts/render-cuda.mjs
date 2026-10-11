import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const articleDir = path.join(repo, "notes", "cuda");
const markdownPath = path.join(articleDir, "performance.md");
const pagePath = path.join(articleDir, "performance.html");
const examplePath = path.join(repo, "notes", "examples", "cuda_perf_bench.cu");
const checkOnly = process.argv.includes("--check");

const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

const sectionIds = new Map([
  ["测试目标与计时口径", "scope"],
  ["设备与编译环境", "environment"],
  ["正确性与 CUDA Event 基线", "baseline"],
  ["Nsight Systems 程序级时间线", "system"],
  ["Nsight Compute Kernel 指标", "kernel"],
  ["调优决策与复测", "loop"],
  ["RTX 5070 访存映射实测", "experiment"],
]);

const source = fs.readFileSync(markdownPath, "utf8");
const titleMatch = source.match(/^# (.+)$/m);
if (!titleMatch || titleMatch.index !== 0) {
  throw new Error("performance.md 必须以一级标题开头");
}
const title = titleMatch[1];
const markdown = source.slice(titleMatch[0].length).trimStart();
const headings = [...markdown.matchAll(/^## (.+)$/gm)];
if (!headings.length) throw new Error("performance.md 至少需要一个二级标题");

const codeMatches = [...source.matchAll(/<!-- CUDA_SOURCE -->\r?\n```cuda\r?\n([\s\S]*?)\r?\n```/g)];
if (codeMatches.length !== 1) {
  throw new Error("必须恰好有一个 <!-- CUDA_SOURCE --> 标记和紧随其后的 cuda 代码块");
}
const example = codeMatches[0][1].replaceAll("\r\n", "\n") + "\n";

const usedIds = new Set();
const sections = headings.map((match, index) => {
  const name = match[1];
  const baseId = sectionIds.get(name) || `section-${index + 1}`;
  let id = baseId;
  let suffix = 2;
  while (usedIds.has(id)) id = `${baseId}-${suffix++}`;
  usedIds.add(id);
  return { name, id, index: match.index, length: match[0].length };
});

function renderMarkdown(input) {
  const prepared = input.trim().replace("<!-- CUDA_SOURCE -->", "");
  if (!prepared) return "";
  let html = marked.parse(prepared, { gfm: true });
  html = html.replaceAll("<h3>", "<h4>").replaceAll("</h3>", "</h4>");
  html = html.replace(
    /<pre><code(?: class="language-([^"]+)")?>([\s\S]*?)<\/code><\/pre>/g,
    (_, language, content) => {
      const fullSource = language === "cuda" && content.split("\n").length > 60;
      const label = language === "bash" ? "Shell" : language === "cuda" ? "CUDA C++" : language || "Code";
      const caption = fullSource ? "cuda_perf_bench.cu · 完整程序，可滚动查看" :
        language === "bash" ? "终端命令" : "代码片段";
      const className = fullSource ? "code-block is-long" : "code-block";
      return `<figure class="${className}"><figcaption><span>${escapeHtml(caption)}</span>` +
        `<span class="code-lang">${escapeHtml(label)}</span></figcaption>` +
        `<pre><code>${content.trimEnd()}</code></pre></figure>`;
    }
  );
  html = html.replaceAll("<table>", '<div class="table-scroll"><table>')
    .replaceAll("</table>", "</table></div>");
  html = html.replace(/<a href="([^"]+)">/g, (_, href) => {
    const external = /^https?:\/\//.test(href);
    const attrs = external ? ' target="_blank" rel="noopener noreferrer"' :
      href.endsWith(".cu") ? " download" : "";
    return `<a class="inline-link" href="${href}"${attrs}>`;
  });
  return html.trimEnd();
}

const intro = renderMarkdown(markdown.slice(0, sections[0].index));
const article = [intro, ...sections.map((item, index) => {
  const end = index + 1 < sections.length ? sections[index + 1].index : markdown.length;
  const body = markdown.slice(item.index + item.length, end);
  return `<section id="${item.id}"><h3>${escapeHtml(item.name)}</h3>\n` +
    renderMarkdown(body) + "\n</section>";
})].filter(Boolean).join("\n");
const toc = sections.map((item) =>
  `<li><a href="#${item.id}">${escapeHtml(item.name)}</a></li>`
).join("\n");

function replaceBetween(page, start, end, content) {
  const a = page.indexOf(start);
  const b = page.indexOf(end, a + start.length);
  if (a < 0 || b < 0) throw new Error(`performance.html 中缺少生成标记：${start} / ${end}`);
  return page.slice(0, a + start.length) + "\n" + content + "\n" + page.slice(b);
}

const current = fs.readFileSync(pagePath, "utf8");
let next = current.replace(
  /<title>[^<]* · WSJ<\/title>/,
  `<title>${escapeHtml(title)} · WSJ</title>`
).replace(
  /(<div class="reader-heading"><h1>)[^<]*(<\/h1>)/,
  (_, before, after) => before + escapeHtml(title) + after
);
next = replaceBetween(next, "<!-- CUDA_TOC_START -->", "<!-- CUDA_TOC_END -->", toc);
next = replaceBetween(next, "<!-- CUDA_ARTICLE_START -->", "<!-- CUDA_ARTICLE_END -->", article);

if (checkOnly) {
  if (next !== current || example !== fs.readFileSync(examplePath, "utf8")) {
    console.error("生成文件与 performance.md 不一致；运行 npm run build:cuda");
    process.exitCode = 1;
  } else {
    console.log("performance.html 和 cuda_perf_bench.cu 与 performance.md 一致");
  }
} else {
  fs.writeFileSync(pagePath, next);
  if (example !== fs.readFileSync(examplePath, "utf8")) fs.writeFileSync(examplePath, example);
  console.log("已从 notes/cuda/performance.md 生成 HTML 与 CUDA 示例");
}
