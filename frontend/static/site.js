(function () {
  const FALLBACK_PRODUCT_NAME = "ResearchPigeon";

  const SOCIAL_ICON_SVGS = {
    x:
      '<svg class="site-footer-icon" aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">' +
      '<path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"></path>' +
      "</svg>",
    bluesky:
      '<svg class="site-footer-icon" aria-hidden="true" width="20" height="20" viewBox="0 0 16 16" fill="currentColor">' +
      '<path d="M3.468 1.948C5.303 3.325 7.276 6.118 8 7.616c.725-1.498 2.698-4.29 4.532-5.668C13.855.955 16 .186 16 2.632c0 .489-.28 4.105-.444 4.692-.572 2.04-2.653 2.561-4.504 2.246 3.236.551 4.06 2.375 2.281 4.2-3.376 3.464-4.852-.87-5.23-1.98-.07-.204-.103-.3-.103-.218 0-.081-.033.014-.102.218-.379 1.11-1.855 5.444-5.231 1.98-1.778-1.825-.955-3.65 2.28-4.2-1.85.315-3.932-.205-4.503-2.246C.28 6.737 0 3.12 0 2.632 0 .186 2.145.955 3.468 1.948"></path>' +
      "</svg>",
  };

  const SOCIAL_LINK_LABELS = {
    x: "ResearchPigeon on X",
    bluesky: "ResearchPigeon on Bluesky",
  };

  function applyProductName(productName) {
    const name = productName || FALLBACK_PRODUCT_NAME;
    document.querySelectorAll(".product-name-text, .product-name-slot").forEach(function (el) {
      el.textContent = name;
    });
    const suffix = document.body.getAttribute("data-document-title");
    document.title = suffix ? suffix + " - " + name : name;
  }

  function renderSocialFooter(socialLinks) {
    if (!socialLinks || typeof socialLinks !== "object") {
      return;
    }

    const entries = Object.keys(SOCIAL_LINK_LABELS).filter(function (key) {
      return typeof socialLinks[key] === "string" && socialLinks[key].length > 0;
    });
    if (entries.length === 0) {
      return;
    }

    const footer = document.createElement("footer");
    footer.className = "site-footer";

    const nav = document.createElement("nav");
    nav.className = "site-footer-links";
    nav.setAttribute("aria-label", "Social links");

    entries.forEach(function (key) {
      const link = document.createElement("a");
      link.className = "site-footer-link";
      link.href = socialLinks[key];
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.setAttribute("aria-label", SOCIAL_LINK_LABELS[key]);
      link.innerHTML = SOCIAL_ICON_SVGS[key];
      nav.appendChild(link);
    });

    footer.appendChild(nav);
    document.body.classList.add("has-site-footer");
    document.body.appendChild(footer);
  }

  function markActiveTopbarLink() {
    const nav = document.querySelector(".topbar-nav");
    if (!nav) {
      return;
    }

    const path = window.location.pathname.replace(/\/+$/, "") || "/";
    const activeHrefByPath = {
      "/": null,
      "/profiles": "/profiles",
      "/preferences": "/profiles",
      "/papers": "/papers",
      "/feedback": "/papers",
      "/digest": "/digest",
      "/about": "/about",
    };
    const activeHref = Object.prototype.hasOwnProperty.call(activeHrefByPath, path)
      ? activeHrefByPath[path]
      : null;

    nav.querySelectorAll(".link-btn").forEach(function (link) {
      const linkPath = (link.getAttribute("href") || "").replace(/\/+$/, "");
      const isActive = Boolean(activeHref && linkPath === activeHref);
      link.classList.toggle("is-active", isActive);
      if (isActive) {
        link.setAttribute("aria-current", "page");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  }

  async function initSiteBranding() {
    try {
      const response = await fetch("/site-config");
      if (!response.ok) {
        applyProductName(FALLBACK_PRODUCT_NAME);
        return;
      }
      const payload = await response.json();
      applyProductName(payload.product_name);
      renderSocialFooter(payload.social_links);
    } catch (_error) {
      applyProductName(FALLBACK_PRODUCT_NAME);
    }
  }

  markActiveTopbarLink();
  initSiteBranding();
})();
