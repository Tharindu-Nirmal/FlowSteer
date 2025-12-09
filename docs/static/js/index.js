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

        // Show the matching panel, hide the others
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

  // ----- Before/after sliders (clip-path + vertical handle) -----
  const sliders = document.querySelectorAll(".ba-slider");

  sliders.forEach((slider) => {
    const input = slider.querySelector(".ba-slider-input");
    const afterWrapper = slider.querySelector(".ba-img-after-wrapper");
    const handle = slider.querySelector(".ba-handle");

    if (!input || !afterWrapper || !handle) return;

    function updateSlider() {
      const val = parseFloat(input.value) || 0; // 0–100
      // Reveal val% of the after image from the left
      // We clip the right side by (100 - val)%
      afterWrapper.style.clipPath = `inset(0 ${100 - val}% 0 0)`;
      // Move the visible handle line
      handle.style.left = `${val}%`;
    }

    // Update on drag
    input.addEventListener("input", updateSlider);
    // And once at start
    updateSlider();
  });
});
