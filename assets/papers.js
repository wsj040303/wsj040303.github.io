const paperSearch = document.querySelector("#paper-search");
const paperList = document.querySelector("#paper-list");

if (paperSearch && paperList) {
  const categories = window.PAPER_CATEGORIES || [];
  const papers = window.PAPER_ENTRIES || [];
  const categoryList = document.querySelector("#paper-categories");
  const resultCount = document.querySelector("#paper-count");
  const emptyState = document.querySelector("#paper-empty");
  const clearSearch = document.querySelector("#paper-search-clear");
  const resetButton = document.querySelector("#paper-reset");
  const categoryNames = new Map(categories.map((item) => [item.id, item.name]));
  let activeCategory = "all";

  const normalize = (value) => String(value || "").normalize("NFKC").toLocaleLowerCase().trim();

  const addText = (parent, tag, className, value) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = value;
    parent.append(element);
    return element;
  };

  const renderCategories = () => {
    categoryList.replaceChildren();
    for (const category of [{ id: "all", name: "全部论文" }, ...categories]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "paper-category";
      button.dataset.category = category.id;
      button.setAttribute("aria-pressed", String(category.id === activeCategory));
      addText(button, "span", "paper-category-name", category.name);
      const count = category.id === "all"
        ? papers.length
        : papers.filter((paper) => paper.category === category.id).length;
      addText(button, "span", "paper-category-count", String(count));
      button.addEventListener("click", () => {
        activeCategory = category.id;
        renderCategories();
        renderPapers();
      });
      categoryList.append(button);
    }
  };

  const makeCard = (paper) => {
    const card = document.createElement("article");
    card.className = "paper-card";
    const meta = addText(card, "div", "paper-card-meta", categoryNames.get(paper.category) || "其他");
    addText(meta, "span", "paper-year", paper.year);
    addText(card, "h2", "", paper.title);
    if (paper.shortName) addText(card, "p", "paper-short-name", paper.shortName);
    if (paper.summary) addText(card, "p", "paper-summary", paper.summary);
    const footer = document.createElement("div");
    footer.className = "paper-card-footer";
    if (paper.sample) addText(footer, "span", "paper-sample", "示例条目");
    const link = addText(footer, "a", "paper-link", paper.articleUrl ? "阅读报告 →" : "查看论文 ↗");
    link.href = paper.articleUrl || paper.url;
    if (paper.articleUrl && paper.url) {
      const source = addText(footer, "a", "paper-source-link", "论文原文 ↗");
      source.href = paper.url;
      source.target = "_blank";
      source.rel = "noopener noreferrer";
    } else {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
    card.append(footer);
    return card;
  };

  const renderPapers = () => {
    const query = normalize(paperSearch.value);
    const matches = papers.filter((paper) => {
      const inCategory = activeCategory === "all" || paper.category === activeCategory;
      const inTitle = normalize(`${paper.title} ${paper.shortName} ${(paper.aliases || []).join(" ")}`).includes(query);
      return inCategory && inTitle;
    });
    paperList.replaceChildren(...matches.map(makeCard));
    emptyState.hidden = matches.length > 0;
    resultCount.textContent = query || activeCategory !== "all"
      ? `找到 ${matches.length} 篇论文`
      : `共 ${matches.length} 篇论文`;
    clearSearch.hidden = !paperSearch.value;
  };

  paperSearch.addEventListener("input", renderPapers);
  clearSearch.addEventListener("click", () => {
    paperSearch.value = "";
    paperSearch.focus();
    renderPapers();
  });
  resetButton.addEventListener("click", () => {
    activeCategory = "all";
    paperSearch.value = "";
    renderCategories();
    renderPapers();
    paperSearch.focus();
  });

  renderCategories();
  renderPapers();
}
