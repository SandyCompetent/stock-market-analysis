/**
 * Portfolio Website Interactive Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Mobile Menu Toggle
  const mobileToggle = document.getElementById("mobileToggle");
  const navMenu = document.getElementById("navMenu");

  if (mobileToggle && navMenu) {
    mobileToggle.addEventListener("click", () => {
      navMenu.classList.toggle("open");
    });

    // Close mobile menu on link click
    document.querySelectorAll(".nav-link").forEach((link) => {
      link.addEventListener("click", () => {
        navMenu.classList.remove("open");
      });
    });
  }

  // 2. Interactive Device Mockup Switcher (Desktop vs Mobile)
  const btnDesktop = document.getElementById("btnDesktop");
  const btnMobile = document.getElementById("btnMobile");
  const mockupFrame = document.getElementById("mockupFrame");
  const mockupImage = document.getElementById("mockupImage");
  const mockupUrl = document.getElementById("mockupUrl");

  if (btnDesktop && btnMobile && mockupFrame && mockupImage) {
    btnDesktop.addEventListener("click", () => {
      btnDesktop.classList.add("active");
      btnMobile.classList.remove("active");
      mockupFrame.classList.remove("mobile-mode");
      mockupFrame.classList.add("desktop-mode");
      mockupImage.src = "assets/img/streamlit-desktop.png";
      mockupImage.alt = "Desktop Streamlit Web Application";
      if (mockupUrl) mockupUrl.textContent = "stock-market-analysis-sandy.streamlit.app (Desktop 1440x900)";
    });

    btnMobile.addEventListener("click", () => {
      btnMobile.classList.add("active");
      btnDesktop.classList.remove("active");
      mockupFrame.classList.remove("desktop-mode");
      mockupFrame.classList.add("mobile-mode");
      mockupImage.src = "assets/img/streamlit-mobile.png";
      mockupImage.alt = "Mobile Streamlit Web Application";
      if (mockupUrl) mockupUrl.textContent = "stock-market-analysis-sandy.streamlit.app (Mobile 390x844)";
    });
  }

  // 3. Image Lightbox Modal
  const modal = document.getElementById("imageModal");
  const modalImg = document.getElementById("modalImg");
  const modalClose = document.getElementById("modalClose");

  const zoomableImages = document.querySelectorAll(".zoomable, .mockup-img");

  zoomableImages.forEach((img) => {
    img.addEventListener("click", () => {
      if (modal && modalImg) {
        modalImg.src = img.src;
        modal.classList.add("active");
      }
    });
  });

  if (modalClose && modal) {
    modalClose.addEventListener("click", () => {
      modal.classList.remove("active");
    });

    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        modal.classList.remove("active");
      }
    });
  }

  // 4. Smooth Anchor Scrolling & Header Shadows
  window.addEventListener("scroll", () => {
    const navbar = document.querySelector(".navbar");
    if (navbar) {
      if (window.scrollY > 30) {
        navbar.style.borderBottomColor = "rgba(99, 102, 241, 0.35)";
      } else {
        navbar.style.borderBottomColor = "rgba(99, 102, 241, 0.18)";
      }
    }
  });
});
