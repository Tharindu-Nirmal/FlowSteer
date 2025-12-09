window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})

// Task tabs: toggle panels
document.addEventListener("DOMContentLoaded", function () {
  const tabs = document.querySelectorAll(".tabs ul li[data-task]");
  const panels = document.querySelectorAll(".task-panel");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const task = tab.getAttribute("data-task");

      // Update tab active state
      tabs.forEach(t => t.classList.remove("is-active"));
      tab.classList.add("is-active");

      // Show the corresponding panel
      panels.forEach(panel => {
        if (panel.getAttribute("data-task") === task) {
          panel.classList.add("is-active");
        } else {
          panel.classList.remove("is-active");
        }
      });
    });
  });

  // Before/after sliders
  const sliders = document.querySelectorAll(".ba-slider");
  sliders.forEach(slider => {
    const input = slider.querySelector(".ba-slider-input");
    const afterWrapper = slider.querySelector(".ba-img-after-wrapper");

    if (input && afterWrapper) {
      input.addEventListener("input", () => {
        const val = input.value; // 0–100
        afterWrapper.style.width = val + "%";
      });
    }
  });
});