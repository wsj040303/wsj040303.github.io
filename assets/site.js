const menuButton = document.querySelector(".nav-toggle");
const menu = document.querySelector("#site-nav");

if (menuButton && menu) {
  menuButton.addEventListener("click", () => {
    const expanded = menuButton.getAttribute("aria-expanded") !== "true";
    menuButton.setAttribute("aria-expanded", String(expanded));
    menu.dataset.open = String(expanded);
  });
  menu.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      menuButton.setAttribute("aria-expanded", "false");
      menu.dataset.open = "false";
    });
  });
}

const chapterPanel = document.querySelector(".chapter-panel");
if (chapterPanel && window.matchMedia("(max-width: 650px)").matches) {
  chapterPanel.open = false;
}

const chapterLinks = document.querySelectorAll(".chapter-list a");
if (chapterLinks.length) {
  const updateCurrent = () => {
    chapterLinks.forEach((link) => {
      link.classList.toggle("is-current", link.hash === window.location.hash);
    });
  };
  window.addEventListener("hashchange", updateCurrent);
  updateCurrent();
}

// Highlight the CUDA C++ snippets locally, without loading a third-party script.
const codeKeywords = new Set([
  "alignas", "auto", "break", "case", "class", "const", "constexpr", "continue",
  "decltype", "default", "delete", "do", "else", "enum", "explicit", "extern",
  "false", "for", "if", "inline", "namespace", "nullptr", "operator", "private",
  "protected", "public", "return", "sizeof", "static", "static_assert", "struct",
  "switch", "template", "this", "true", "typedef", "typename", "using", "virtual",
  "volatile", "while", "__device__", "__global__", "__host__", "__shared__",
  "__forceinline__", "__restrict__", "__syncthreads", "__launch_bounds__"
]);
const codeTypes = new Set([
  "bool", "char", "double", "float", "half", "half2", "int", "long", "short", "signed",
  "size_t", "unsigned", "void", "dim3", "cudaError_t", "cudaStream_t", "uint32_t",
  "uint64_t", "int32_t", "int64_t", "__half", "__half2"
]);
const codeToken = /\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|^[ \t]*#[^\n]*|\b(?:0[xX][\da-fA-F']+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)[uUlLfF]*\b|\b[A-Za-z_]\w*\b/gm;

document.querySelectorAll(".code-block pre code").forEach((code) => {
  const source = code.textContent;
  const fragment = document.createDocumentFragment();
  let cursor = 0;
  for (const match of source.matchAll(codeToken)) {
    const token = match[0];
    const position = match.index;
    if (position > cursor) fragment.append(document.createTextNode(source.slice(cursor, position)));
    let kind;
    if (token.startsWith("//") || token.startsWith("/*")) kind = "comment";
    else if (/^[ \t]*#/.test(token)) kind = "preprocessor";
    else if (token.startsWith('"') || token.startsWith("'")) kind = "string";
    else if (/^\d/.test(token)) kind = "number";
    else if (codeKeywords.has(token)) kind = "keyword";
    else if (codeTypes.has(token)) kind = "type";
    else if (/^\s*\(/.test(source.slice(position + token.length))) kind = "function";
    if (kind) {
      const span = document.createElement("span");
      span.className = `tok-${kind}`;
      span.textContent = token;
      fragment.append(span);
    } else {
      fragment.append(document.createTextNode(token));
    }
    cursor = position + token.length;
  }
  fragment.append(document.createTextNode(source.slice(cursor)));
  code.replaceChildren(fragment);
});
