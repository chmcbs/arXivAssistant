const authGate = document.getElementById("auth-gate");
const prefsApp = document.getElementById("prefs-app");
const authStatus = document.getElementById("auth-status");
const authLinkWrap = document.getElementById("auth-link-wrap");
const authLink = document.getElementById("auth-link");
const sessionLabel = document.getElementById("session-label");
const prefsStatus = document.getElementById("prefs-status");
const profilesGrid = document.getElementById("profiles-grid");
const addProfileBtn = document.getElementById("add-profile-btn");
const cardTemplate = document.getElementById("profile-card-template");

let categories = [];
let profiles = [];

function setStatus(message, isError) {
  if (!isError) {
    prefsStatus.textContent = "";
    return;
  }
  prefsStatus.textContent = message;
  prefsStatus.style.color = "#b91c1c";
}

async function checkSession() {
  const session = await apiRequest("/auth/session", "GET");
  if (!session.authenticated) {
    sessionLabel.textContent = "Not signed in";
    authGate.classList.remove("hidden");
    prefsApp.classList.add("hidden");
    return false;
  }
  sessionLabel.textContent = session.email;
  authGate.classList.add("hidden");
  prefsApp.classList.remove("hidden");
  return true;
}

async function loadCategories() {
  const payload = await apiRequest("/categories", "GET");
  categories = payload.categories || [];
}

function getSelectedProfileIds() {
  return profiles
    .filter((item) => item.digest_enabled && !item.is_draft)
    .map((item) => item.profile_id);
}

function setSummaryLine(element, label, value) {
  element.innerHTML = "";
  const strong = document.createElement("strong");
  strong.textContent = `${label}: `;
  element.appendChild(strong);
  element.appendChild(document.createTextNode(value || "-"));
}

function formatFeedbackDate(value) {
  if (!value) {
    return "";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "";
  }
  return parsed.toLocaleDateString("en-GB", {
    year: "2-digit",
    month: "2-digit",
    day: "2-digit",
  });
}

function normalizeKeyword(rawKeyword, currentKeywords) {
  const keyword = rawKeyword.trim().toLowerCase();
  if (!keyword) {
    return { error: "Keyword cannot be empty." };
  }
  if (keyword.length > 24) {
    return { error: "Keyword max length is 24 characters." };
  }
  if ((currentKeywords || []).includes(keyword)) {
    return { error: "Keyword already exists on this profile." };
  }
  if ((currentKeywords || []).length >= 20) {
    return { error: "Keyword limit reached (20)." };
  }
  return { keyword };
}

async function updateDigestSelection() {
  await apiRequest("/profiles/digest-selection", "PUT", {
    profile_ids: getSelectedProfileIds(),
  });
}

async function loadProfiles() {
  const payload = await apiRequest("/profiles", "GET");
  profiles = payload.profiles || [];
  renderProfiles();
}

function renderProfiles() {
  profilesGrid.innerHTML = "";
  if (!profiles.length) {
    profilesGrid.innerHTML = "<p class='muted'>No profiles yet. Add your first profile.</p>";
    return;
  }

  profiles.forEach((profile) => {
    const isDraft = Boolean(profile.is_draft);
    const isEditing = isDraft || Boolean(profile.is_editing);
    const node = cardTemplate.content.firstElementChild.cloneNode(true);
    const profileTitleInput = node.querySelector(".profile-title-input");
    const categoryLabel = node.querySelector(".category-label");
    const categorySelect = node.querySelector(".category-select");
    const interestLabel = node.querySelector(".interest-label");
    const interestText = node.querySelector(".interest-text");
    const summaryBlock = node.querySelector(".profile-summary");
    const summaryCategory = node.querySelector(".summary-category");
    const summaryDescription = node.querySelector(".summary-description");
    const summaryFeedback = node.querySelector(".summary-feedback");
    const feedbackToggle = node.querySelector(".feedback-toggle");
    const feedbackPanel = node.querySelector(".feedback-panel");
    const feedbackList = node.querySelector(".feedback-list");
    const digestCheckbox = node.querySelector(".digest-checkbox");
    const keywordList = node.querySelector(".keyword-list");
    const saveBtn = node.querySelector(".save-btn");
    const cancelEditBtn = node.querySelector(".cancel-edit-btn");
    const deleteBtn = node.querySelector(".delete-btn");
    const editBtn = node.querySelector(".edit-btn");

    profile.keywords = Array.isArray(profile.keywords) ? profile.keywords : [];
    profile.feedback_items = Array.isArray(profile.feedback_items) ? profile.feedback_items : [];
    profile.is_adding_keyword = Boolean(profile.is_adding_keyword);
    profile.feedback_expanded = Boolean(profile.feedback_expanded);
    profileTitleInput.value = profile.profile_name || "";
    profileTitleInput.disabled = !isEditing;
    interestText.value = profile.interest_sentence;
    setSummaryLine(summaryCategory, "Category", profile.category);
    setSummaryLine(summaryDescription, "Description", profile.interest_sentence);
    function updateFeedbackSummary() {
      const likedCount = (profile.liked_arxiv_ids || []).length;
      const dislikedCount = (profile.disliked_arxiv_ids || []).length;
      summaryFeedback.textContent = `\ud83d\udc4d ${likedCount}   \ud83d\udc4e ${dislikedCount}`;
    }
    updateFeedbackSummary();
    feedbackToggle.textContent = profile.feedback_expanded
      ? "Hide feedback"
      : "View feedback";

    const showDraftFields = isDraft && isEditing;
    categoryLabel.classList.toggle("hidden", !showDraftFields);
    categorySelect.classList.toggle("hidden", !showDraftFields);
    interestLabel.classList.toggle("hidden", !showDraftFields);
    interestText.classList.toggle("hidden", !showDraftFields);
    summaryBlock.classList.toggle("hidden", isDraft);
    feedbackToggle.classList.toggle("hidden", isDraft);
    feedbackPanel.classList.toggle("hidden", isDraft || !profile.feedback_expanded);
    interestText.readOnly = !isDraft;
    interestText.placeholder = isDraft ? "Describe your research interests." : "";
    digestCheckbox.checked = profile.digest_enabled;
    saveBtn.textContent = isDraft ? "Create profile" : "Save profile";
    deleteBtn.textContent = isDraft ? "Cancel" : "Delete";
    saveBtn.classList.toggle("hidden", !isEditing);
    cancelEditBtn.classList.toggle("hidden", !isEditing || isDraft);
    deleteBtn.classList.toggle("hidden", !isEditing);
    keywordList.classList.remove("hidden");
    editBtn.classList.toggle("hidden", isDraft || isEditing);
    editBtn.textContent = "Edit profile";
    profileTitleInput.addEventListener("input", () => {
      profile.profile_name = profileTitleInput.value;
    });

    categories.forEach((category) => {
      const option = document.createElement("option");
      option.value = category;
      option.textContent = category;
      if (category === profile.category) {
        option.selected = true;
      }
      categorySelect.appendChild(option);
    });

    function drawKeywords(values) {
      keywordList.innerHTML = "";
      values.forEach((value) => {
        const chip = document.createElement("span");
        chip.className = "keyword-chip";
        chip.textContent = value;
        if (isEditing) {
          const removeBtn = document.createElement("button");
          removeBtn.className = "chip-btn";
          removeBtn.textContent = "x";
          removeBtn.type = "button";
          removeBtn.addEventListener("click", async () => {
            if (isDraft) {
              profile.keywords = (profile.keywords || []).filter((item) => item !== value);
              drawKeywords(profile.keywords);
              setStatus("Keyword removed.", false);
              return;
            }
            try {
              const payload = await apiRequest(`/profiles/${profile.profile_id}/keywords`, "DELETE", { keyword: value });
              profile.keywords = Array.isArray(payload.keywords) ? payload.keywords : [];
              drawKeywords(profile.keywords);
              setStatus("Keyword removed.", false);
            } catch (error) {
              setStatus(String(error.message || error), true);
            }
          });
          chip.appendChild(removeBtn);
        }
        keywordList.appendChild(chip);
      });

      if (!isEditing) {
        return;
      }

      const addChip = document.createElement("button");
      addChip.className = "keyword-chip keyword-add-chip";
      addChip.type = "button";
      addChip.textContent = "+keyword";
      addChip.addEventListener("click", () => {
        profile.is_adding_keyword = true;
        drawKeywords(profile.keywords || []);
      });
      keywordList.appendChild(addChip);

      if (!profile.is_adding_keyword) {
        return;
      }

      const inputChip = document.createElement("span");
      inputChip.className = "keyword-chip keyword-input-chip";
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = "e.g. retrieval";
      input.className = "keyword-inline-input";

      const confirmBtn = document.createElement("button");
      confirmBtn.className = "chip-btn";
      confirmBtn.type = "button";
      confirmBtn.textContent = "✓";

      const cancelBtn = document.createElement("button");
      cancelBtn.className = "chip-btn";
      cancelBtn.type = "button";
      cancelBtn.textContent = "✕";

      async function addKeywordFromInlineInput() {
        const normalized = normalizeKeyword(input.value, profile.keywords || []);
        if (normalized.error) {
          setStatus(normalized.error, true);
          return;
        }
        if (isDraft) {
          profile.keywords = [...(profile.keywords || []), normalized.keyword];
          profile.is_adding_keyword = false;
          drawKeywords(profile.keywords);
          setStatus("Keyword added.", false);
          return;
        }
        try {
          const payload = await apiRequest(`/profiles/${profile.profile_id}/keywords`, "POST", {
            keyword: normalized.keyword,
          });
          profile.keywords = Array.isArray(payload.keywords) ? payload.keywords : [];
          profile.is_adding_keyword = false;
          drawKeywords(profile.keywords);
          setStatus("Keyword added.", false);
        } catch (error) {
          setStatus(String(error.message || error), true);
        }
      }

      confirmBtn.addEventListener("click", addKeywordFromInlineInput);
      cancelBtn.addEventListener("click", () => {
        profile.is_adding_keyword = false;
        drawKeywords(profile.keywords || []);
      });
      input.addEventListener("keydown", async (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          await addKeywordFromInlineInput();
        } else if (event.key === "Escape") {
          profile.is_adding_keyword = false;
          drawKeywords(profile.keywords || []);
        }
      });

      inputChip.appendChild(input);
      inputChip.appendChild(confirmBtn);
      inputChip.appendChild(cancelBtn);
      keywordList.appendChild(inputChip);
      input.focus();
    }

    function drawFeedbackItems() {
      feedbackList.innerHTML = "";
      if (isDraft) {
        return;
      }

      if (!profile.feedback_items.length) {
        const emptyRow = document.createElement("p");
        emptyRow.className = "summary-line";
        emptyRow.textContent = "No feedback yet.";
        feedbackList.appendChild(emptyRow);
        return;
      }

      const likedItems = profile.feedback_items.filter((item) => item.label === "like");
      const dislikedItems = profile.feedback_items.filter((item) => item.label === "dislike");

      function drawFeedbackGroup(label, symbol, items) {
        if (!items.length) {
          return;
        }
        const heading = document.createElement("p");
        heading.className = "summary-line feedback-group-title";
        heading.textContent = `${symbol} ${label}`;
        feedbackList.appendChild(heading);

        items.forEach((item) => {
          const row = document.createElement("div");
          row.className = "feedback-row";

          const leftSide = document.createElement("div");
          leftSide.className = "feedback-row-left";

          const text = document.createElement("a");
          text.textContent = item.title;
          text.className = "feedback-item-text";
          text.href = `https://arxiv.org/pdf/${item.arxiv_id}`;
          text.target = "_blank";
          text.rel = "noopener noreferrer";

          const rightSide = document.createElement("div");
          rightSide.className = "feedback-row-actions";

          const dateLabel = document.createElement("span");
          dateLabel.className = "feedback-date";
          dateLabel.textContent = formatFeedbackDate(item.created_at);

          rightSide.appendChild(dateLabel);
          if (isEditing) {
            const removeBtn = document.createElement("button");
            removeBtn.className = "chip-btn";
            removeBtn.type = "button";
            removeBtn.textContent = "x";
            removeBtn.addEventListener("click", async () => {
              try {
                await apiRequest("/feedback", "DELETE", {
                  profile_id: profile.profile_id,
                  arxiv_id: item.arxiv_id,
                });
                profile.feedback_items = (profile.feedback_items || []).filter(
                  (entry) => entry.arxiv_id !== item.arxiv_id
                );
                if (item.label === "like") {
                  profile.liked_arxiv_ids = (profile.liked_arxiv_ids || []).filter(
                    (id) => id !== item.arxiv_id
                  );
                } else {
                  profile.disliked_arxiv_ids = (profile.disliked_arxiv_ids || []).filter(
                    (id) => id !== item.arxiv_id
                  );
                }
                updateFeedbackSummary();
                setStatus("Feedback removed.", false);
                drawFeedbackItems();
              } catch (error) {
                setStatus(String(error.message || error), true);
              }
            });
            rightSide.appendChild(removeBtn);
          }

          leftSide.appendChild(text);
          row.appendChild(leftSide);
          row.appendChild(rightSide);
          feedbackList.appendChild(row);
        });
      }

      drawFeedbackGroup("Likes", "\ud83d\udc4d", likedItems);
      drawFeedbackGroup("Dislikes", "\ud83d\udc4e", dislikedItems);
    }

    drawKeywords(profile.keywords || []);
    drawFeedbackItems();
    feedbackToggle.addEventListener("click", () => {
      profile.feedback_expanded = !profile.feedback_expanded;
      renderProfiles();
    });

    saveBtn.addEventListener("click", async () => {
      if (isDraft) {
        const interestSentence = interestText.value.trim();
        if (!interestSentence) {
          setStatus("Interest sentence is required when creating a profile.", true);
          interestText.focus();
          return;
        }
        try {
          const createPayload = await apiRequest("/profiles", "POST", {
            profile_name: profileTitleInput.value.trim() || `Profile ${profiles.length}`,
            category: categorySelect.value,
            interest_sentence: interestSentence,
          });
          let createdProfile = createPayload.profile;
          const draftKeywords = [...(profile.keywords || [])];
          for (const keyword of draftKeywords) {
            const keywordPayload = await apiRequest(
              `/profiles/${createdProfile.profile_id}/keywords`,
              "POST",
              { keyword }
            );
            createdProfile.keywords = keywordPayload.keywords;
          }
          const updatePayload = await apiRequest(`/profiles/${createdProfile.profile_id}`, "PUT", {
            profile_name: profileTitleInput.value,
            category: categorySelect.value,
            digest_enabled: digestCheckbox.checked,
          });
          createdProfile = updatePayload.profile;
          await updateDigestSelection();
          await loadProfiles();
          setStatus("Profile created.", false);
        } catch (error) {
          setStatus(String(error.message || error), true);
        }
        return;
      }
      try {
        const updatePayload = await apiRequest(`/profiles/${profile.profile_id}`, "PUT", {
          profile_name: profileTitleInput.value,
          digest_enabled: digestCheckbox.checked,
        });
        profile.profile_name = updatePayload.profile.profile_name;
        profile.digest_enabled = updatePayload.profile.digest_enabled;
        profile.is_editing = false;
        await updateDigestSelection();
        renderProfiles();
        setStatus("Profile saved.", false);
      } catch (error) {
        setStatus(String(error.message || error), true);
      }
    });

    editBtn.addEventListener("click", async () => {
      profile.is_editing = true;
      renderProfiles();
      setStatus("Edit mode enabled.", false);
    });
    cancelEditBtn.addEventListener("click", async () => {
      profile.is_editing = false;
      await loadProfiles();
      setStatus("Edit cancelled.", false);
    });

    digestCheckbox.addEventListener("change", async () => {
      profile.digest_enabled = digestCheckbox.checked;
      if (isDraft) {
        return;
      }
      try {
        await apiRequest(`/profiles/${profile.profile_id}`, "PUT", {
          profile_name: profile.profile_name,
          digest_enabled: profile.digest_enabled,
        });
        await updateDigestSelection();
        setStatus("Active state updated.", false);
      } catch (error) {
        profile.digest_enabled = !digestCheckbox.checked;
        digestCheckbox.checked = profile.digest_enabled;
        setStatus(String(error.message || error), true);
      }
    });

    deleteBtn.addEventListener("click", async () => {
      if (isDraft) {
        profiles = profiles.filter((item) => item.profile_id !== profile.profile_id);
        renderProfiles();
        setStatus("Profile creation cancelled.", false);
        return;
      }
      try {
        await apiRequest(`/profiles/${profile.profile_id}`, "DELETE", {});
        profiles = profiles.filter((item) => item.profile_id !== profile.profile_id);
        await updateDigestSelection();
        renderProfiles();
        setStatus("Profile deleted.", false);
      } catch (error) {
        setStatus(String(error.message || error), true);
      }
    });

    profilesGrid.appendChild(node);
  });
}

document.getElementById("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const email = document.getElementById("auth-email").value.trim();
  authStatus.textContent = "Sending magic link...";
  authLinkWrap.classList.add("hidden");
  try {
    const payload = await apiRequest("/auth/magic-link/request", "POST", { email });
    authStatus.textContent = "Check your inbox for the confirmation link.";
    if (payload.magic_link) {
      authLink.href = payload.magic_link;
      authLinkWrap.classList.remove("hidden");
    }
  } catch (error) {
    authStatus.textContent = String(error.message || error);
  }
});

addProfileBtn.addEventListener("click", async () => {
  if (profiles.length >= 3) {
    setStatus("Profile limit reached (3).", true);
    return;
  }
  if (profiles.some((item) => item.is_draft)) {
    setStatus("Finish creating the current new profile first.", true);
    return;
  }
  profiles.unshift({
    profile_id: `draft-${Date.now()}`,
    profile_slot: profiles.length + 1,
    profile_name: `Profile ${profiles.length + 1}`,
    category: categories[0] || "cs.AI",
    interest_sentence: "",
    digest_enabled: true,
    keywords: [],
    is_draft: true,
  });
  renderProfiles();
  setStatus("Fill in the new profile card and click Create profile.", false);
});

async function init() {
  try {
    const authenticated = await checkSession();
    if (!authenticated) {
      return;
    }
    await loadCategories();
    await loadProfiles();
  } catch (error) {
    setStatus(String(error.message || error), true);
  }
}

init();
