import { Marked } from "marked";
import katex from "katex";

const render = (tex, displayMode) => katex.renderToString(tex, {
  displayMode,
  output: "htmlAndMathml",
  throwOnError: true,
});

// Render TeX while parsing Markdown so formulas inside paragraphs are never
// mistaken for code spans or processed by the later HTML replacements.
export const marked = new Marked({
  gfm: true,
  extensions: [
    {
      name: "mathDisplay",
      level: "block",
      start: (source) => source.indexOf("$$"),
      tokenizer(source) {
        const match = /^\$\$[ \t]*\n([\s\S]+?)\n\$\$(?=\n|$)/.exec(source);
        if (match) return { type: "mathDisplay", raw: match[0], tex: match[1] };
      },
      renderer(token) {
        return '<div class="math-display">' + render(token.tex, true) + "</div>\n";
      },
    },
    {
      name: "mathInline",
      level: "inline",
      start: (source) => source.indexOf("$"),
      tokenizer(source) {
        const match = /^\$(?!\$)((?:\\.|[^$\\\n])+?)\$(?!\$)/.exec(source);
        if (match && match[1].trim()) {
          return { type: "mathInline", raw: match[0], tex: match[1] };
        }
      },
      renderer(token) {
        return '<span class="math-inline">' + render(token.tex, false) + "</span>";
      },
    },
  ],
});
