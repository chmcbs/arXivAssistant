"""
Creates postgres table for storing fetched papers

"""

import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

def get_database_url() -> str:
    return os.environ["DATABASE_URL"]

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

def main():
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_PAPERS_TABLE)
            cur.execute(CREATE_RUNS_TABLE)

if __name__ == "__main__":
    main()