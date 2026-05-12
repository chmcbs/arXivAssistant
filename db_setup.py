"""
Creates postgres table for storing fetched papers
"""

import psycopg
from db_helper import get_database_url

CREATE_PAPERS_TABLE = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT[],
    published_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    pdf_url TEXT,
    entry_url TEXT,
    categories TEXT[],
    inserted_at TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    category TEXT NOT NULL,
    max_results INTEGER NOT NULL,
    fetched_count INTEGER DEFAULT 0,
    saved_count INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    error_message TEXT
);
"""

CREATE_VECTOR_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

CREATE_PAPER_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS paper_embeddings (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_PAPERS_KEYWORD_INDEX = """
CREATE INDEX IF NOT EXISTS papers_keyword_idx
ON papers
USING GIN (
    (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
    )
);
"""

CREATE_USER_PREFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY,
    initial_interest_embedding vector(384) NOT NULL,
    preference_embedding vector(384) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

ALTER_USER_PREFERENCES_ADD_DAILY_K = """
ALTER TABLE user_preferences
ADD COLUMN IF NOT EXISTS daily_k INTEGER CHECK (daily_k >= 1);
"""

CREATE_PAPER_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS paper_feedback (
    feedback_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES user_preferences(user_id) ON DELETE CASCADE,
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    label TEXT NOT NULL CHECK (label IN ('like', 'dislike')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_PAPER_FEEDBACK_USER_PAPER_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS paper_feedback_user_paper_idx
ON paper_feedback (user_id, arxiv_id);
"""

CREATE_RECOMMENDATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES user_preferences(user_id) ON DELETE CASCADE,
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    rank INTEGER NOT NULL CHECK (rank >= 1),
    final_score DOUBLE PRECISION NOT NULL,
    candidate_window TEXT NOT NULL,
    fallback_stage SMALLINT NOT NULL DEFAULT 0,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id, user_id, rank),
    UNIQUE(run_id, user_id, arxiv_id)
);
"""

CREATE_RECOMMENDATIONS_USER_GENERATED_INDEX = """
CREATE INDEX IF NOT EXISTS recommendations_user_generated_idx
ON recommendations (user_id, generated_at DESC);
"""

CREATE_RECOMMENDATIONS_USER_PAPER_GENERATED_INDEX = """
CREATE INDEX IF NOT EXISTS recommendations_user_paper_generated_idx
ON recommendations (user_id, arxiv_id, generated_at DESC);
"""

def main():
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_VECTOR_EXTENSION)
            cur.execute(CREATE_PAPERS_TABLE)
            cur.execute(CREATE_RUNS_TABLE)
            cur.execute(CREATE_PAPER_EMBEDDINGS_TABLE)
            cur.execute(CREATE_PAPERS_KEYWORD_INDEX)
            cur.execute(CREATE_USER_PREFERENCES_TABLE)
            cur.execute(ALTER_USER_PREFERENCES_ADD_DAILY_K)
            cur.execute(CREATE_PAPER_FEEDBACK_TABLE)
            cur.execute(CREATE_PAPER_FEEDBACK_USER_PAPER_INDEX)
            cur.execute(CREATE_RECOMMENDATIONS_TABLE)
            cur.execute(CREATE_RECOMMENDATIONS_USER_GENERATED_INDEX)
            cur.execute(CREATE_RECOMMENDATIONS_USER_PAPER_GENERATED_INDEX)

if __name__ == "__main__":
    main()