import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "./math-markdown.mjs";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const articleDir = path.join(repo, "papers", "pikv");
const sourcePath = path.join(articleDir, "article.md");
const pagePath = path.join(articleDir, "index.html");
const checkOnly = process.argv.includes("--check");
const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

const sectionIds = new Map([
  ["主要工作与优化目标", "research-question"],
  ["路由：缩小本次查询的 KV 访问范围", "routing"],
  ["压缩：减少每条 KV 的存储与搬运量", "compression"],
  ["调度：让有限容量优先保留有用的 KV 页", "scheduling"],
  ["FPGA：可选扩展", "fpga"],
]);

const source = fs.readFileSync(sourcePath, "utf8");
const titleMatch = source.match(/^# (.+)$/m);
if (!titleMatch || titleMatch.index !== 0) {
  throw new Error("article.md must begin with a level-one title");
}
const title = titleMatch[1];
const markdown = source.slice(titleMatch[0].length).trimStart();
const headings = [...markdown.matchAll(/^## (.+)$/gm)];
if (!headings.length) throw new Error("article.md needs at least one level-two section");

for (const match of source.matchAll(/!\[[^\]]*\]\((images\/[^)]+)\)/g)) {
  if (!fs.existsSync(path.join(articleDir, match[1]))) {
    throw new Error("Missing article image: " + match[1]);
  }
}

const usedIds = new Set();
const items = headings.map((match, index) => {
  const name = match[1];
  const baseId = sectionIds.get(name) || "section-" + (index + 1);
  let id = baseId;
  let suffix = 2;
  while (usedIds.has(id)) id = baseId + "-" + suffix++;
  usedIds.add(id);
  return { name, id, index: match.index, length: match[0].length };
});

items.forEach((item, index) => {
  const end = index + 1 < items.length ? items[index + 1].index : markdown.length;
  const body = markdown.slice(item.index + item.length, end);
  item.subheadings = marked.lexer(body)
    .filter((token) => token.type === "heading" && token.depth === 3)
    .map((token, subindex) => ({ name: token.text, id: item.id + "-" + (subindex + 1) }));
});

function renderMarkdown(input, subheadings = []) {
  const prepared = input.trim().replace(
    /!\[([^\]]+)\]\((images\/[^)]+)\)\n\n\*([^\n]+)\*/g,
    (_, alt, image, caption) => {
      const captionHtml = marked.parseInline(caption);
      return '<figure class="paper-figure"><a href="' + escapeHtml(image) +
        '" aria-label="打开原尺寸论文图片"><img src="' + escapeHtml(image) +
        '" alt="' + escapeHtml(alt) +
        '" loading="lazy" decoding="async"></a><figcaption>' +
        captionHtml + "</figcaption></figure>";
    }
  );
  let html = marked.parse(prepared, { gfm: true });
  let subindex = 0;
  html = html.replace(/<h3>([\s\S]*?)<\/h3>/g, (_, content) => {
    const heading = subheadings[subindex++];
    if (!heading) throw new Error("Unexpected level-three heading in article");
    return '<h4 id="' + escapeHtml(heading.id) + '">' + content + "</h4>";
  });
  if (subindex !== subheadings.length) {
    throw new Error("Missing rendered level-three heading in article");
  }
  html = html.replace(
    /<pre><code(?: class="[^"]*")?>([\s\S]*?)<\/code><\/pre>/g,
    (_, content) => '<figure class="code-block"><pre><code>' + content + "</code></pre></figure>"
  );
  html = html.replaceAll("<table>", '<div class="table-scroll"><table>')
    .replaceAll("</table>", "</table></div>")
    .replaceAll("<a href=", '<a class="inline-link" href=');
  return html;
}

const intro = renderMarkdown(markdown.slice(0, items[0].index));
const sections = items.map((item, index) => {
  const end = index + 1 < items.length ? items[index + 1].index : markdown.length;
  const body = markdown.slice(item.index + item.length, end);
  return '<section id="' + item.id + '"><h3>' + escapeHtml(item.name) +
    "</h3>\n" + renderMarkdown(body, item.subheadings) + "</section>";
}).join("\n");
const toc = items.map((item) => {
  const sublist = item.subheadings.length
    ? '<ol class="chapter-sublist">' + item.subheadings.map((heading) =>
      '<li><a href="#' + heading.id + '">' + escapeHtml(heading.name) + "</a></li>"
    ).join("") + "</ol>"
    : "";
  return '<li><a href="#' + item.id + '">' + escapeHtml(item.name) +
    "</a>" + sublist + "</li>";
}).join("\n");

function replaceBetween(page, start, end, content) {
  const a = page.indexOf(start);
  const b = page.indexOf(end, a + start.length);
  if (a < 0 || b < 0) throw new Error("Missing generated-content markers in index.html");
  return page.slice(0, a + start.length) + "\n" + content + "\n" + page.slice(b);
}

const current = fs.readFileSync(pagePath, "utf8");
let next = current.replace(
  /<title>[^<]* · WSJ<\/title>/,
  () => "<title>" + escapeHtml(title) + " · WSJ</title>"
).replace(
  /(<div class="reader-heading"><h1>)[^<]*(<\/h1>)/,
  (_, before, after) => before + escapeHtml(title) + after
);
next = replaceBetween(next, "<!-- PIKV_TOC_START -->", "<!-- PIKV_TOC_END -->", toc);
next = replaceBetween(
  next, "<!-- PIKV_ARTICLE_START -->", "<!-- PIKV_ARTICLE_END -->",
  '<div class="paper-intro">' + intro + "</div>\n" + sections
);

if (checkOnly) {
  if (next !== current) {
    console.error("index.html is out of date; run npm run build:pikv");
    process.exitCode = 1;
  } else {
    console.log("index.html matches article.md");
  }
} else {
  fs.writeFileSync(pagePath, next);
  console.log("Rendered " + path.relative(repo, pagePath));
}
