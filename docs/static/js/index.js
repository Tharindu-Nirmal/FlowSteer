// FlowSteer website scripts
document.addEventListener("DOMContentLoaded", function () {
  // ----- Navbar burger (mobile menu) -----
  const burger = document.querySelector(".navbar-burger");
  const menu = document.querySelector(".navbar-menu");

  if (burger && menu) {
    burger.addEventListener("click", function () {
      burger.classList.toggle("is-active");
      menu.classList.toggle("is-active");
    });
  }

  // ----- Results: task tabs -----
  const tabs = document.querySelectorAll(".tabs ul li[data-task]");
  const panels = document.querySelectorAll(".task-panel");

  if (tabs.length && panels.length) {
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const task = tab.getAttribute("data-task");

        // Toggle active tab
        tabs.forEach((t) => t.classList.remove("is-active"));
        tab.classList.add("is-active");

        // Show the matching panel, hide others
        panels.forEach((panel) => {
          if (panel.getAttribute("data-task") === task) {
            panel.classList.add("is-active");
          } else {
            panel.classList.remove("is-active");
          }
        });
      });
    });
  }

  // ----- Before/after sliders -----
  const sliders = document.querySelectorAll(".ba-slider");

  sliders.forEach((slider) => {
    const input = slider.querySelector(".ba-slider-input");
    const afterWrapper = slider.querySelector(".ba-img-after-wrapper");

    if (!input || !afterWrapper) return;

    // Set initial width based on initial input value
    afterWrapper.style.width = input.value + "%";

    input.addEventListener("input", () => {
      const val = input.value; // 0–100
      afterWrapper.style.width = val + "%";
    });
  });
});
