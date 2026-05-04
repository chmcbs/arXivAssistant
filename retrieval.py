"""
Searches stored papers using Postgres full-text keyword retrieval
"""

import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

# Give title matches more importance than abstract matches
PAPER_SEARCH_VECTOR_SQL = """
setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
"""

def get_database_url() -> str:
    return os.environ["DATABASE_URL"]

def keyword_query(query: str) -> str:
    return query.strip()

def search_keyword_papers(query: str, limit: int = 10) -> list[dict]:
    cleaned_query = keyword_query(query)

    if not cleaned_query:
        return []

    sql = f"""
    SELECT
        arxiv_id,
        title,
        abstract,
        ts_rank_cd(
            {PAPER_SEARCH_VECTOR_SQL},
            websearch_to_tsquery('english', %s)
        ) AS keyword_score
    FROM papers
    WHERE {PAPER_SEARCH_VECTOR_SQL} @@ websearch_to_tsquery('english', %s)
    ORDER BY keyword_score DESC, published_at DESC
    LIMIT %s;
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (cleaned_query, cleaned_query, limit))
            rows = cur.fetchall()

    return [
        {
            "arxiv_id": row[0],
            "title": row[1],
            "abstract": row[2] or "",
            "keyword_score": float(row[3]),
        }
        for row in rows
    ]

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "language agents"
    results = search_keyword_papers(query, limit=5)

    for result in results:
        print(f"{result['keyword_score']:.4f} {result['arxiv_id']} {result['title']}")