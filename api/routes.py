"""
HTTP route definitions for the API service
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from api.dependencies import (
    add_profile_keyword_payload,
    create_profile_payload,
    generate_daily_picks_payload,
    get_daily_picks_payload,
    get_debug_daily_picks_payload,
    get_metrics_payload,
    list_profile_keywords_payload,
    list_profiles_payload,
    remove_profile_keyword_payload,
    save_feedback_payload,
    update_digest_selection_payload,
)
from api.schemas import (
    CreateProfileRequest,
    CreateProfileResponse,
    DailyPicksResponse,
    DebugDailyPicksResponse,
    FeedbackRequest,
    FeedbackResponse,
    GenerateDailyPicksRequest,
    GenerateDailyPicksResponse,
    ListProfilesResponse,
    ManageProfileKeywordRequest,
    ManageProfileKeywordResponse,
    UpdateDigestSelectionRequest,
    UpdateDigestSelectionResponse,
)
from core.config import DEFAULT_USER_ID

app = FastAPI(title="arXiv Assistant API")


@app.get("/validate", response_class=HTMLResponse)
def validate() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Validation UI</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 20px;
      line-height: 1.4;
      color: #1f2937;
    }
    h1 {
      margin-top: 0;
    }
    .muted {
      color: #4b5563;
      margin-bottom: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
    }
    .card {
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px;
      background: #fafafa;
    }
    .card h2 {
      margin: 0 0 8px 0;
      font-size: 16px;
    }
    label {
      display: block;
      margin: 6px 0 3px;
      font-size: 12px;
      color: #374151;
    }
    input, select, button {
      width: 100%;
      box-sizing: border-box;
      margin-bottom: 8px;
      padding: 8px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      font-size: 14px;
    }
    button {
      background: #111827;
      color: #fff;
      border: 0;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover {
      background: #0b1220;
    }
    pre {
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      padding: 10px;
      border-radius: 6px;
      min-height: 140px;
      overflow: auto;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <h1>Validation UI</h1>
  <p class="muted">
    Internal development surface for quickly validating API behavior and DB side effects.
  </p>

  <div class="grid">
    <section class="card">
      <h2>GET /profiles</h2>
      <label for="profiles-user-id">user_id</label>
      <input id="profiles-user-id" value="default">
      <button id="profiles-btn">Load Profiles</button>
      <pre id="profiles-out">Click "Load Profiles".</pre>
    </section>

    <section class="card">
      <h2>PUT /profiles/digest-selection</h2>
      <label for="digest-user-id">user_id</label>
      <input id="digest-user-id" value="default">
      <label for="digest-profile-ids">profile_ids (comma-separated)</label>
      <input id="digest-profile-ids" placeholder="profile-1,profile-2">
      <button id="digest-btn">Update Selection</button>
      <pre id="digest-out">Set digest profile IDs.</pre>
    </section>

    <section class="card">
      <h2>POST /daily-picks/generate</h2>
      <label for="generate-user-id">user_id</label>
      <input id="generate-user-id" value="default">
      <label for="generate-profile-id">profile_id (optional)</label>
      <input id="generate-profile-id" placeholder="profile-1">
      <label for="generate-max-results">max_results</label>
      <input id="generate-max-results" type="number" value="150" min="1">
      <label for="generate-embedding-limit">embedding_limit</label>
      <input id="generate-embedding-limit" type="number" value="600" min="1">
      <button id="generate-btn">Run Generate</button>
      <pre id="generate-out">Trigger full generation pipeline.</pre>
    </section>

    <section class="card">
      <h2>GET /daily-picks</h2>
      <label for="daily-user-id">user_id</label>
      <input id="daily-user-id" value="default">
      <label for="daily-profile-id">profile_id (optional)</label>
      <input id="daily-profile-id" placeholder="profile-1">
      <button id="daily-btn">Load Daily Picks</button>
      <pre id="daily-out">Load the public picks payload.</pre>
    </section>

    <section class="card">
      <h2>GET /daily-picks/debug</h2>
      <label for="debug-user-id">user_id</label>
      <input id="debug-user-id" value="default">
      <label for="debug-profile-id">profile_id (optional)</label>
      <input id="debug-profile-id" placeholder="profile-1">
      <button id="debug-btn">Load Debug Picks</button>
      <pre id="debug-out">Load scoring/debug fields.</pre>
    </section>

    <section class="card">
      <h2>POST /feedback</h2>
      <label for="feedback-user-id">user_id</label>
      <input id="feedback-user-id" value="default">
      <label for="feedback-profile-id">profile_id (optional)</label>
      <input id="feedback-profile-id" placeholder="profile-1">
      <label for="feedback-arxiv-id">arxiv_id</label>
      <input id="feedback-arxiv-id" placeholder="2601.00001">
      <label for="feedback-label">label</label>
      <select id="feedback-label">
        <option value="like">like</option>
        <option value="dislike">dislike</option>
      </select>
      <button id="feedback-btn">Submit Feedback</button>
      <pre id="feedback-out">Submit feedback for a paper.</pre>
    </section>

    <section class="card">
      <h2>GET /metrics</h2>
      <label for="metrics-limit">latest_runs_limit</label>
      <input id="metrics-limit" type="number" value="10" min="1">
      <button id="metrics-btn">Load Metrics</button>
      <pre id="metrics-out">Inspect run and recommendation metrics.</pre>
    </section>
  </div>

  <script>
    function output(id, value) {
      document.getElementById(id).textContent =
        typeof value === "string" ? value : JSON.stringify(value, null, 2);
    }

    function optionalString(value) {
      var trimmed = (value || "").trim();
      return trimmed.length ? trimmed : null;
    }

    async function request(url, method, body) {
      var options = { method: method || "GET", headers: { "Content-Type": "application/json" } };
      if (body) {
        options.body = JSON.stringify(body);
      }
      var response = await fetch(url, options);
      var payload;
      try {
        payload = await response.json();
      } catch (err) {
        payload = { detail: "No JSON response body" };
      }
      if (!response.ok) {
        throw { status: response.status, payload: payload };
      }
      return payload;
    }

    async function run(outId, action) {
      output(outId, "Loading...");
      try {
        var result = await action();
        output(outId, result);
      } catch (err) {
        output(outId, {
          error: true,
          status: err.status || 500,
          detail: err.payload || err.message || "Unexpected error"
        });
      }
    }

    document.getElementById("profiles-btn").addEventListener("click", function () {
      run("profiles-out", async function () {
        var userId = encodeURIComponent(document.getElementById("profiles-user-id").value.trim() || "default");
        return request("/profiles?user_id=" + userId, "GET");
      });
    });

    document.getElementById("digest-btn").addEventListener("click", function () {
      run("digest-out", async function () {
        var userId = document.getElementById("digest-user-id").value.trim() || "default";
        var profileIds = document
          .getElementById("digest-profile-ids")
          .value
          .split(",")
          .map(function (value) { return value.trim(); })
          .filter(function (value) { return value.length > 0; });
        return request("/profiles/digest-selection", "PUT", {
          user_id: userId,
          profile_ids: profileIds
        });
      });
    });

    document.getElementById("generate-btn").addEventListener("click", function () {
      run("generate-out", async function () {
        var body = {
          user_id: document.getElementById("generate-user-id").value.trim() || "default",
          max_results: Number(document.getElementById("generate-max-results").value) || 150,
          embedding_limit: Number(document.getElementById("generate-embedding-limit").value) || 600
        };
        var profileId = optionalString(document.getElementById("generate-profile-id").value);
        if (profileId) {
          body.profile_id = profileId;
        }
        return request("/daily-picks/generate", "POST", body);
      });
    });

    document.getElementById("daily-btn").addEventListener("click", function () {
      run("daily-out", async function () {
        var userId = encodeURIComponent(document.getElementById("daily-user-id").value.trim() || "default");
        var profileId = optionalString(document.getElementById("daily-profile-id").value);
        var query = "/daily-picks?user_id=" + userId;
        if (profileId) {
          query += "&profile_id=" + encodeURIComponent(profileId);
        }
        return request(query, "GET");
      });
    });

    document.getElementById("debug-btn").addEventListener("click", function () {
      run("debug-out", async function () {
        var userId = encodeURIComponent(document.getElementById("debug-user-id").value.trim() || "default");
        var profileId = optionalString(document.getElementById("debug-profile-id").value);
        var query = "/daily-picks/debug?user_id=" + userId;
        if (profileId) {
          query += "&profile_id=" + encodeURIComponent(profileId);
        }
        return request(query, "GET");
      });
    });

    document.getElementById("feedback-btn").addEventListener("click", function () {
      run("feedback-out", async function () {
        var body = {
          arxiv_id: document.getElementById("feedback-arxiv-id").value.trim(),
          label: document.getElementById("feedback-label").value,
          user_id: document.getElementById("feedback-user-id").value.trim() || "default"
        };
        var profileId = optionalString(document.getElementById("feedback-profile-id").value);
        if (profileId) {
          body.profile_id = profileId;
        }
        return request("/feedback", "POST", body);
      });
    });

    document.getElementById("metrics-btn").addEventListener("click", function () {
      run("metrics-out", async function () {
        var limit = Number(document.getElementById("metrics-limit").value) || 10;
        return request("/metrics?latest_runs_limit=" + encodeURIComponent(limit), "GET");
      });
    });
  </script>
</body>
</html>
"""


@app.get("/daily-picks", response_model=DailyPicksResponse)
def daily_picks(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
) -> dict:
    return get_daily_picks_payload(user_id=user_id, profile_id=profile_id)


@app.get("/daily-picks/debug", response_model=DebugDailyPicksResponse)
def daily_picks_debug(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
) -> dict:
    return get_debug_daily_picks_payload(user_id=user_id, profile_id=profile_id)


@app.post("/daily-picks/generate", response_model=GenerateDailyPicksResponse)
def daily_picks_generate(request: GenerateDailyPicksRequest) -> dict:
    return generate_daily_picks_payload(request)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> dict:
    return save_feedback_payload(request)


@app.post("/profiles", response_model=CreateProfileResponse)
def profiles_create(request: CreateProfileRequest) -> dict:
    return create_profile_payload(request)


@app.get("/profiles", response_model=ListProfilesResponse)
def profiles_list(user_id: str = DEFAULT_USER_ID) -> dict:
    return list_profiles_payload(user_id=user_id)


@app.get("/profiles/{profile_id}/keywords", response_model=ManageProfileKeywordResponse)
def profiles_keywords_list(
    profile_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> dict:
    return list_profile_keywords_payload(profile_id=profile_id, user_id=user_id)


@app.post(
    "/profiles/{profile_id}/keywords", response_model=ManageProfileKeywordResponse
)
def profiles_keywords_add(
    profile_id: str,
    request: ManageProfileKeywordRequest,
) -> dict:
    return add_profile_keyword_payload(profile_id=profile_id, request=request)


@app.delete(
    "/profiles/{profile_id}/keywords", response_model=ManageProfileKeywordResponse
)
def profiles_keywords_remove(
    profile_id: str,
    request: ManageProfileKeywordRequest,
) -> dict:
    return remove_profile_keyword_payload(profile_id=profile_id, request=request)


@app.put("/profiles/digest-selection", response_model=UpdateDigestSelectionResponse)
def profiles_digest_selection_update(request: UpdateDigestSelectionRequest) -> dict:
    return update_digest_selection_payload(request)


@app.get("/metrics")
def metrics(latest_runs_limit: int = 10) -> dict:
    if latest_runs_limit < 1:
        raise HTTPException(status_code=400, detail="latest_runs_limit must be >= 1")

    return get_metrics_payload(latest_runs_limit=latest_runs_limit)
