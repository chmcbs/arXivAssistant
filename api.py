"""
FastAPI service for personalised arXiv paper recommendations
"""

from datetime import datetime
from typing import Literal

import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import DEFAULT_USER_ID, get_arxiv_categories
from db_helper import get_database_url
from preferences import save_feedback, update_preference_embedding


app = FastAPI(title="arXiv Assistant API")


LATEST_DAILY_PICKS_SQL = """
WITH latest_run AS (
    SELECT
        run_id,
        MAX(generated_at) AS generated_at
    FROM recommendations
    WHERE user_id = %s
    GROUP BY run_id
    ORDER BY MAX(generated_at) DESC
    LIMIT 1
)
SELECT
    rec.rank,
    p.arxiv_id,
    p.title,
    COALESCE(p.abstract, '') AS abstract,
    p.pdf_url,
    rec.run_id::text,
    r.category,
    rec.generated_at,
    rec.final_score,
    rec.candidate_window,
    rec.fallback_stage
FROM latest_run lr
JOIN recommendations rec
  ON rec.run_id = lr.run_id
 AND rec.user_id = %s
JOIN papers p ON p.arxiv_id = rec.arxiv_id
JOIN runs r ON r.run_id = rec.run_id
ORDER BY rec.rank ASC;
"""

RUN_STATUS_COUNTS_SQL = """
SELECT status, COUNT(*)
FROM runs
GROUP BY status
ORDER BY status ASC;
"""

LATEST_RUNS_SQL = """
SELECT
    run_id::text,
    status,
    category,
    max_results,
    fetched_count,
    saved_count,
    started_at,
    finished_at,
    error_message
FROM runs
ORDER BY started_at DESC
LIMIT %s;
"""

RECOMMENDATION_TOTAL_SQL = """
SELECT COUNT(*)
FROM recommendations;
"""

RECOMMENDATIONS_BY_USER_SQL = """
SELECT user_id, COUNT(*)
FROM recommendations
GROUP BY user_id
ORDER BY user_id ASC;
"""


class PublicPick(BaseModel):
    rank: int
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str | None


class DebugPick(PublicPick):
    run_id: str
    category: str
    generated_at: datetime
    final_score: float
    candidate_window: str
    fallback_stage: int


class DailyPicksResponse(BaseModel):
    user_id: str
    needs_generation: bool
    picks: list[PublicPick]


class DebugDailyPicksResponse(BaseModel):
    user_id: str
    needs_generation: bool
    run_id: str | None = None
    category: str | None = None
    generated_at: datetime | None = None
    picks: list[DebugPick]


class GenerateDailyPicksRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    max_results: int = Field(default=150, ge=1)
    embedding_limit: int = Field(default=600, ge=1)


class GenerateDailyPicksResponse(BaseModel):
    user_id: str
    run_ids: list[str]
    embedded_count: int
    recommendation_counts: dict[str, int]
    needs_generation: bool
    picks: list[PublicPick]


class FeedbackRequest(BaseModel):
    arxiv_id: str
    label: Literal["like", "dislike"]
    user_id: str = DEFAULT_USER_ID


class FeedbackResponse(BaseModel):
    feedback_id: str
    user_id: str
    arxiv_id: str
    label: Literal["like", "dislike"]
    preference_updated: bool


def _fetch_latest_picks(user_id: str) -> list[tuple]:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(LATEST_DAILY_PICKS_SQL, (user_id, user_id))
            return cur.fetchall()


def _public_pick(row: tuple) -> dict:
    return {
        "rank": int(row[0]),
        "arxiv_id": row[1],
        "title": row[2],
        "abstract": row[3],
        "pdf_url": row[4],
    }


def _debug_pick(row: tuple) -> dict:
    return {
        **_public_pick(row),
        "run_id": row[5],
        "category": row[6],
        "generated_at": row[7],
        "final_score": float(row[8]),
        "candidate_window": row[9],
        "fallback_stage": int(row[10]),
    }


def get_daily_picks_payload(user_id: str = DEFAULT_USER_ID) -> dict:
    rows = _fetch_latest_picks(user_id)

    return {
        "user_id": user_id,
        "needs_generation": not rows,
        "picks": [_public_pick(row) for row in rows],
    }


def get_debug_daily_picks_payload(user_id: str = DEFAULT_USER_ID) -> dict:
    rows = _fetch_latest_picks(user_id)

    payload = {
        "user_id": user_id,
        "needs_generation": not rows,
        "run_id": None,
        "category": None,
        "generated_at": None,
        "picks": [_debug_pick(row) for row in rows],
    }

    if rows:
        payload["run_id"] = rows[0][5]
        payload["category"] = rows[0][6]
        payload["generated_at"] = rows[0][7]

    return payload


def _ensure_single_category_mvp() -> None:
    categories = get_arxiv_categories()
    if len(categories) != 1:
        raise HTTPException(
            status_code=400,
            detail="API MVP supports exactly one configured arXiv category",
        )


def generate_daily_picks_payload(request: GenerateDailyPicksRequest) -> dict:
    _ensure_single_category_mvp()

    from pipeline import run_pipeline

    summary = run_pipeline(
        user_id=request.user_id,
        max_results=request.max_results,
        embedding_limit=request.embedding_limit,
    )
    picks_payload = get_daily_picks_payload(request.user_id)

    return {
        "user_id": request.user_id,
        "run_ids": summary["run_ids"],
        "embedded_count": summary["embedded_count"],
        "recommendation_counts": {
            run_id: len(recommendations)
            for run_id, recommendations in summary["recommendations_by_run"].items()
        },
        "needs_generation": picks_payload["needs_generation"],
        "picks": picks_payload["picks"],
    }


def save_feedback_payload(request: FeedbackRequest) -> dict:
    try:
        feedback_id = save_feedback(
            arxiv_id=request.arxiv_id,
            label=request.label,
            user_id=request.user_id,
        )
        update_preference_embedding(user_id=request.user_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "feedback_id": feedback_id,
        "user_id": request.user_id,
        "arxiv_id": request.arxiv_id,
        "label": request.label,
        "preference_updated": True,
    }


def get_metrics_payload(latest_runs_limit: int = 10) -> dict:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(RUN_STATUS_COUNTS_SQL)
            run_status_counts = {
                status: int(count)
                for status, count in cur.fetchall()
            }

            cur.execute(LATEST_RUNS_SQL, (latest_runs_limit,))
            latest_runs = [
                {
                    "run_id": row[0],
                    "status": row[1],
                    "category": row[2],
                    "max_results": int(row[3]),
                    "fetched_count": int(row[4] or 0),
                    "saved_count": int(row[5] or 0),
                    "started_at": row[6],
                    "finished_at": row[7],
                    "error_message": row[8],
                }
                for row in cur.fetchall()
            ]

            cur.execute(RECOMMENDATION_TOTAL_SQL)
            total_recommendations = int(cur.fetchone()[0])

            cur.execute(RECOMMENDATIONS_BY_USER_SQL)
            recommendations_by_user = {
                user_id: int(count)
                for user_id, count in cur.fetchall()
            }

    return {
        "run_status_counts": run_status_counts,
        "latest_runs": latest_runs,
        "total_recommendations": total_recommendations,
        "recommendations_by_user": recommendations_by_user,
    }


@app.get("/daily-picks", response_model=DailyPicksResponse)
def daily_picks(user_id: str = DEFAULT_USER_ID) -> dict:
    return get_daily_picks_payload(user_id)


@app.get("/daily-picks/debug", response_model=DebugDailyPicksResponse)
def daily_picks_debug(user_id: str = DEFAULT_USER_ID) -> dict:
    return get_debug_daily_picks_payload(user_id)


@app.post("/daily-picks/generate", response_model=GenerateDailyPicksResponse)
def daily_picks_generate(request: GenerateDailyPicksRequest) -> dict:
    return generate_daily_picks_payload(request)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> dict:
    return save_feedback_payload(request)


@app.get("/metrics")
def metrics(latest_runs_limit: int = 10) -> dict:
    if latest_runs_limit < 1:
        raise HTTPException(status_code=400, detail="latest_runs_limit must be >= 1")

    return get_metrics_payload(latest_runs_limit=latest_runs_limit)
