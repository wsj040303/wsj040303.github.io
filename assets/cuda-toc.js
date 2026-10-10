const toc = document.querySelector(".chapter-list[data-recursive-toc]");
const article = document.querySelector(".reader-content.cuda-article");

if (toc && article && article.querySelector("h2, h3, h4, h5, h6")) {
  const headings = [...article.querySelectorAll("h2, h3, h4, h5, h6")];
  const generated = document.createElement("ol");
  const stack = [{ level: 0, list: generated }];
  let generatedId = 0;

  for (const heading of headings) {
    const level = Number(heading.tagName.slice(1));
    while (stack.length > 1 && stack[stack.length - 1].level >= level) {
      stack.pop();
    }

    const section = heading.closest("section[id]");
    const sectionHeading = section?.querySelector("h2, h3, h4, h5, h6");
    let id = heading.id || (sectionHeading === heading ? section.id : "");
    if (!id) {
      do {
        id = `${section?.id || "article"}-heading-${++generatedId}`;
      } while (document.getElementById(id));
      heading.id = id;
    }

    const parent = stack[stack.length - 1];
    if (!parent.list) {
      parent.list = document.createElement("ol");
      parent.list.className = "chapter-sublist";
      parent.item.append(parent.list);
    }

    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = `#${id}`;
    link.textContent = heading.textContent.trim();
    link.style.setProperty("--toc-indent", `${(stack.length - 1) * 16}px`);
    item.append(link);
    parent.list.append(item);
    stack.push({ level, item, list: null });
  }

  toc.replaceChildren(...generated.children);
}
