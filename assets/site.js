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
