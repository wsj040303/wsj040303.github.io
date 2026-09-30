const menuButton = document.querySelector(".nav-toggle");
const menu = document.querySelector("#site-nav");

if (menuButton && menu) {
  menuButton.addEventListener("click", () => {
    const open = menuButton.getAttribute("aria-expanded") !== "true";
    menuButton.setAttribute("aria-expanded", String(open));
    menu.dataset.open = String(open);
  });
  menu.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      menuButton.setAttribute("aria-expanded", "false");
      menu.dataset.open = "false";
    });
  });
}

document.querySelectorAll("[data-year]").forEach((element) => {
  element.textContent = new Date().getFullYear();
});

const copyButton = document.querySelector("[data-copy-link]");
const copyStatus = document.querySelector("[data-copy-status]");

if (copyButton && copyStatus) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      copyStatus.textContent = "链接已复制";
    } catch {
      copyStatus.textContent = "复制失败，请从地址栏复制";
    }
  });
}
