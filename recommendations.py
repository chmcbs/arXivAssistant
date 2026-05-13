"""
Generates top-K recommendations per (run_id, profile_id)
"""

import uuid
import psycopg
from config import DEFAULT_USER_ID, get_daily_picks_k, get_keyword_boost_cap
from db_helper import get_database_url
from keyword_search import SEARCH_DICTIONARY, paper_search_vector_sql
from profiles import get_or_create_default_profile

FETCH_RUN_SQL = """
SELECT run_id, category, max_results
FROM runs
WHERE run_id = %s
  AND status = 'completed';
"""

FETCH_EFFECTIVE_K_SQL = """
SELECT COALESCE(daily_k, %s)
FROM profile_preferences
WHERE profile_id = %s;
"""

RANK_CANDIDATES_SQL = f"""
WITH run_context AS (
    SELECT category, max_results
    FROM runs
    WHERE run_id = %s
      AND status = 'completed'
),
preference_context AS (
    SELECT preference_embedding
    FROM profile_preferences
    WHERE profile_id = %s
),
feedback_excluded AS (
    SELECT DISTINCT arxiv_id
    FROM paper_feedback
    WHERE profile_id = %s
      AND label IN ('like', 'dislike')
),
seen_papers AS (
    SELECT DISTINCT arxiv_id
    FROM recommendations
    WHERE profile_id = %s
),
base_papers AS (
    SELECT
        p.arxiv_id,
        p.title,
        p.abstract,
        p.published_at,
        e.embedding
    FROM papers p
    JOIN paper_embeddings e ON e.arxiv_id = p.arxiv_id
    JOIN run_context rc ON rc.category = ANY(p.categories)
    LEFT JOIN feedback_excluded fe ON fe.arxiv_id = p.arxiv_id
    WHERE fe.arxiv_id IS NULL
),
run_window AS (
    SELECT
        bp.*,
        ROW_NUMBER() OVER (
            ORDER BY bp.published_at DESC NULLS LAST, bp.arxiv_id ASC
        ) AS run_rank
    FROM base_papers bp
),
stage0 AS (
    SELECT
        rw.arxiv_id,
        rw.title,
        rw.abstract,
        rw.published_at,
        rw.embedding,
        0::smallint AS fallback_stage,
        'run'::text AS candidate_window
    FROM run_window rw
    CROSS JOIN run_context rc
    LEFT JOIN seen_papers sp ON sp.arxiv_id = rw.arxiv_id
    WHERE rw.run_rank <= rc.max_results
      AND sp.arxiv_id IS NULL
),
stage1 AS (
    SELECT
        bp.arxiv_id,
        bp.title,
        bp.abstract,
        bp.published_at,
        bp.embedding,
        1::smallint AS fallback_stage,
        '7d'::text AS candidate_window
    FROM base_papers bp
    LEFT JOIN seen_papers sp ON sp.arxiv_id = bp.arxiv_id
    WHERE bp.published_at >= NOW() - INTERVAL '7 days'
      AND sp.arxiv_id IS NULL
),
stage2 AS (
    SELECT
        bp.arxiv_id,
        bp.title,
        bp.abstract,
        bp.published_at,
        bp.embedding,
        2::smallint AS fallback_stage,
        '30d'::text AS candidate_window
    FROM base_papers bp
    LEFT JOIN seen_papers sp ON sp.arxiv_id = bp.arxiv_id
    WHERE bp.published_at >= NOW() - INTERVAL '30 days'
      AND sp.arxiv_id IS NULL
),
stage3 AS (
    SELECT
        bp.arxiv_id,
        bp.title,
        bp.abstract,
        bp.published_at,
        bp.embedding,
        3::smallint AS fallback_stage,
        '1y'::text AS candidate_window
    FROM base_papers bp
    LEFT JOIN seen_papers sp ON sp.arxiv_id = bp.arxiv_id
    WHERE bp.published_at >= NOW() - INTERVAL '1 year'
      AND sp.arxiv_id IS NULL
),
stage4 AS (
    SELECT
        bp.arxiv_id,
        bp.title,
        bp.abstract,
        bp.published_at,
        bp.embedding,
        4::smallint AS fallback_stage,
        'all'::text AS candidate_window
    FROM base_papers bp
    LEFT JOIN seen_papers sp ON sp.arxiv_id = bp.arxiv_id
    WHERE sp.arxiv_id IS NULL
),
stage5 AS (
    SELECT
        bp.arxiv_id,
        bp.title,
        bp.abstract,
        bp.published_at,
        bp.embedding,
        5::smallint AS fallback_stage,
        'all_seen_neutral'::text AS candidate_window
    FROM base_papers bp
    JOIN seen_papers sp ON sp.arxiv_id = bp.arxiv_id
),
all_candidates AS (
    SELECT * FROM stage0
    UNION ALL
    SELECT * FROM stage1
    UNION ALL
    SELECT * FROM stage2
    UNION ALL
    SELECT * FROM stage3
    UNION ALL
    SELECT * FROM stage4
    UNION ALL
    SELECT * FROM stage5
),
scored AS (
    SELECT
        c.arxiv_id,
        c.title,
        c.abstract,
        c.published_at,
        c.fallback_stage,
        c.candidate_window,
        1 - (c.embedding <=> pc.preference_embedding) AS base_dense_score,
        LEAST(
            COALESCE(keyword_scores.keyword_score_sum, 0.0),
            %s
        ) AS keyword_boost,
        (1 - (c.embedding <=> pc.preference_embedding)) + LEAST(
            COALESCE(keyword_scores.keyword_score_sum, 0.0),
            %s
        ) AS final_score
    FROM all_candidates c
    CROSS JOIN preference_context pc
    LEFT JOIN LATERAL (
        SELECT
            SUM(
                ts_rank_cd(
                    {paper_search_vector_sql("c")},
                    websearch_to_tsquery('{SEARCH_DICTIONARY}', pk.keyword)
                )
            ) AS keyword_score_sum
        FROM profile_keywords pk
        WHERE pk.profile_id = %s
          AND {paper_search_vector_sql("c")} @@ websearch_to_tsquery('{SEARCH_DICTIONARY}', pk.keyword)
    ) AS keyword_scores ON TRUE
),
deduped AS (
    SELECT DISTINCT ON (arxiv_id)
        arxiv_id,
        title,
        abstract,
        published_at,
        fallback_stage,
        candidate_window,
        base_dense_score,
        keyword_boost,
        final_score
    FROM scored
    ORDER BY
        arxiv_id,
        fallback_stage ASC,
        final_score DESC,
        published_at DESC NULLS LAST
),
prioritized AS (
    SELECT
        arxiv_id,
        title,
        abstract,
        published_at,
        fallback_stage,
        candidate_window,
        base_dense_score,
        keyword_boost,
        final_score
    FROM deduped
    ORDER BY
        fallback_stage ASC,
        final_score DESC,
        published_at DESC NULLS LAST,
        arxiv_id ASC
    LIMIT %s
),
ranked AS (
    SELECT
        ROW_NUMBER() OVER (
            ORDER BY
                fallback_stage ASC,
                final_score DESC,
                published_at DESC NULLS LAST,
                arxiv_id ASC
        ) AS rank,
        arxiv_id,
        title,
        abstract,
        fallback_stage,
        candidate_window,
        base_dense_score,
        keyword_boost,
        final_score
    FROM prioritized
)
SELECT
    rank,
    arxiv_id,
    title,
    abstract,
    fallback_stage,
    candidate_window,
    base_dense_score,
    keyword_boost,
    final_score
FROM ranked
ORDER BY rank ASC;
"""

DELETE_EXISTING_SQL = """
DELETE FROM recommendations
WHERE run_id = %s
  AND profile_id = %s;
"""

INSERT_RECOMMENDATION_SQL = """
INSERT INTO recommendations (
    recommendation_id,
    run_id,
    profile_id,
    arxiv_id,
    rank,
    base_dense_score,
    keyword_boost,
    final_score,
    candidate_window,
    fallback_stage
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
);
"""

def _resolve_profile_id(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
) -> str:
    if profile_id:
        return profile_id

    profile = get_or_create_default_profile(user_id=user_id)
    return str(profile["profile_id"])

# Resolve the number of items to return (override > user preference > default)
def _get_effective_k(cur, profile_id: str, k_override: int | None) -> int:
    if k_override is not None:
        if k_override < 1:
            raise ValueError("k_override must be >= 1")
        return k_override

    default_k = get_daily_picks_k()
    cur.execute(FETCH_EFFECTIVE_K_SQL, (default_k, profile_id))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No preference profile found for profile_id={profile_id}")

    return int(row[0])

def _ensure_completed_run(cur, run_id: str) -> tuple[str, str, int]:
    cur.execute(FETCH_RUN_SQL, (run_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Run {run_id} must exist and be completed")
    return str(row[0]), str(row[1]), int(row[2])

# Rank papers for a completed run and persist as recommendations
def generate_recommendations(
    run_id: str,
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
    k_override: int | None = None,
) -> list[dict]:
    resolved_profile_id = _resolve_profile_id(user_id=user_id, profile_id=profile_id)

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            _ensure_completed_run(cur, run_id)
            effective_k = _get_effective_k(cur, resolved_profile_id, k_override)

            cur.execute(
                RANK_CANDIDATES_SQL,
                (
                    run_id,
                    resolved_profile_id,
                    resolved_profile_id,
                    resolved_profile_id,
                    get_keyword_boost_cap(),
                    get_keyword_boost_cap(),
                    resolved_profile_id,
                    effective_k,
                ),
            )
            rows = cur.fetchall()

            cur.execute(DELETE_EXISTING_SQL, (run_id, resolved_profile_id))

            inserts = [
                (
                    str(uuid.uuid4()),
                    run_id,
                    resolved_profile_id,
                    row[1],
                    int(row[0]),
                    float(row[6]),
                    float(row[7]),
                    float(row[8]),
                    row[5],
                    int(row[4]),
                )
                for row in rows
            ]

            if inserts:
                cur.executemany(INSERT_RECOMMENDATION_SQL, inserts)

    return [
        {
            "rank": int(row[0]),
            "arxiv_id": row[1],
            "title": row[2],
            "abstract": row[3] or "",
            "fallback_stage": int(row[4]),
            "candidate_window": row[5],
            "base_dense_score": float(row[6]),
            "keyword_boost": float(row[7]),
            "final_score": float(row[8]),
        }
        for row in rows
    ]

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python recommendations.py <run_id> [user_id] [profile_id]")

    cli_run_id = sys.argv[1]
    cli_user_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_USER_ID
    cli_profile_id = sys.argv[3] if len(sys.argv) > 3 else None
    generated = generate_recommendations(
        cli_run_id,
        user_id=cli_user_id,
        profile_id=cli_profile_id,
    )

    for row in generated:
        print(
            f"{row['rank']}. {row['arxiv_id']} "
            f"score={row['final_score']:.4f} "
            f"stage={row['fallback_stage']} "
            f"window={row['candidate_window']}"
        )
