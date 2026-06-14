function setPageStatus(statusEl, message, isError) {
  statusEl.textContent = message || "";
  if (!message) {
    statusEl.style.removeProperty("color");
    return;
  }
  statusEl.style.color = isError ? "#b91c1c" : "#6b7280";
}

function setDebugControlsVisible(canAccess) {
  document.querySelectorAll("[data-debug-admin]").forEach(function (el) {
    el.classList.toggle("hidden", !canAccess);
  });
}

async function checkAuthenticatedSession({ sessionLabelEl, authGateEl, appEl }) {
  const session = await apiRequest("/auth/session", "GET");
  setDebugControlsVisible(Boolean(session.can_debug_access));
  if (!session.authenticated) {
    bindSessionMenu(sessionLabelEl, { authenticated: false });
    authGateEl.classList.remove("hidden");
    appEl.classList.add("hidden");
    return false;
  }
  bindSessionMenu(sessionLabelEl, {
    authenticated: true,
    email: session.email,
  });
  authGateEl.classList.add("hidden");
  appEl.classList.remove("hidden");
  return session;
}

function bindSessionMenu(sessionLabelEl, { authenticated, email }) {
  const root = sessionLabelEl.closest(".session-menu");
  const panel = root ? root.querySelector(".session-menu-panel") : null;
  const logoutBtn = root ? root.querySelector(".session-logout-btn") : null;
  if (!root || !panel || !logoutBtn) {
    sessionLabelEl.textContent = authenticated ? email : "Not signed in";
    return;
  }

  if (!authenticated) {
    sessionLabelEl.disabled = true;
    sessionLabelEl.textContent = "Not signed in";
    panel.classList.add("hidden");
    return;
  }

  sessionLabelEl.disabled = false;
  sessionLabelEl.textContent = email;

  if (root.dataset.bound === "1") {
    return;
  }
  root.dataset.bound = "1";

  sessionLabelEl.addEventListener("click", function (event) {
    event.stopPropagation();
    panel.classList.toggle("hidden");
  });

  logoutBtn.addEventListener("click", async function () {
    try {
      await apiRequest("/auth/logout", "POST");
      window.location.href = "/";
    } catch (error) {
      panel.classList.add("hidden");
      window.alert(String(error.message || error));
    }
  });

  if (!document.documentElement.dataset.sessionMenuDismissBound) {
    document.addEventListener("click", function () {
      document.querySelectorAll(".session-menu-panel").forEach(function (el) {
        el.classList.add("hidden");
      });
    });
    document.documentElement.dataset.sessionMenuDismissBound = "1";
  }
}

function bindMagicLinkForm({ formEl, statusEl, linkWrapEl, linkEl, nextPath = "" }) {
  formEl.addEventListener("submit", async (event) => {
    event.preventDefault();
    const email = document.getElementById("auth-email").value.trim();
    statusEl.textContent = "Sending magic link...";
    linkWrapEl.classList.add("hidden");
    try {
      const payload = await apiRequest("/auth/magic-link/request", "POST", { email });
      statusEl.textContent = "Check your inbox for the confirmation link.";
      if (payload.magic_link) {
        linkEl.href = nextPath
          ? `${payload.magic_link}&next=${encodeURIComponent(nextPath)}`
          : payload.magic_link;
        linkWrapEl.classList.remove("hidden");
      }
    } catch (error) {
      statusEl.textContent = String(error.message || error);
    }
  });
}

async function refreshDebugAccess() {
  try {
    const session = await apiRequest("/auth/session", "GET");
    setDebugControlsVisible(Boolean(session.can_debug_access));
  } catch (_error) {
    setDebugControlsVisible(false);
  }
}

refreshDebugAccess();
