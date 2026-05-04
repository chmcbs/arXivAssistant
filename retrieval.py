"""
Searches stored papers using hybrid retrieval
"""

import os
import psycopg
from dotenv import load_dotenv
from embeddings import embed_texts

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

# Convert the list of floats into string format for pgvector
def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"

# Query paper_embeddings and return papers sorted by cosine similarity to the query vector
def search_dense_papers(query: str, limit: int = 10) -> list[dict]:
    cleaned_query = keyword_query(query)

    if not cleaned_query:
        return []

    query_vector = vector_literal(embed_texts([cleaned_query])[0])

    sql = """
    SELECT
        p.arxiv_id,
        p.title,
        p.abstract,
        1 - (e.embedding <=> %s::vector) AS dense_score
    FROM paper_embeddings e
    JOIN papers p ON p.arxiv_id = e.arxiv_id
    ORDER BY e.embedding <=> %s::vector
    LIMIT %s;
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_vector, query_vector, limit))
            rows = cur.fetchall()

    return [
        {
            "arxiv_id": row[0],
            "title": row[1],
            "abstract": row[2] or "",
            "dense_score": float(row[3]),
        }
        for row in rows
    ]

# Make keyword scores and dense scores comparable
def normalize_score(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0

    return score / max_score

# Use 60/40 weighting for dense/keyword
def search_hybrid_papers(
    query: str,
    limit: int = 10,
    dense_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> list[dict]:
    keyword_results = search_keyword_papers(query, limit=limit)
    dense_results = search_dense_papers(query, limit=limit)

    max_keyword_score = max(
        [result["keyword_score"] for result in keyword_results],
        default=0.0,
    )
    max_dense_score = max(
        [result["dense_score"] for result in dense_results],
        default=0.0,
    )

    merged: dict[str, dict] = {}

    for result in keyword_results:
        merged[result["arxiv_id"]] = {
            **result,
            "keyword_score": result["keyword_score"],
            "dense_score": 0.0,
        }

    for result in dense_results:
        existing = merged.get(result["arxiv_id"], {})
        merged[result["arxiv_id"]] = {
            **result,
            **existing,
            "dense_score": result["dense_score"],
            "keyword_score": existing.get("keyword_score", 0.0),
        }

    ranked_results = []

    for result in merged.values():
        keyword_score_norm = normalize_score(
            result["keyword_score"],
            max_keyword_score,
        )
        dense_score_norm = normalize_score(
            result["dense_score"],
            max_dense_score,
        )
        final_score = (
            dense_weight * dense_score_norm
            + keyword_weight * keyword_score_norm
        )

        ranked_results.append(
            {
                **result,
                "dense_score_norm": dense_score_norm,
                "keyword_score_norm": keyword_score_norm,
                "final_score": final_score,
            }
        )

    return sorted(
        ranked_results,
        key=lambda result: result["final_score"],
        reverse=True,
    )[:limit]

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "language agents"
    results = search_hybrid_papers(query, limit=5)

    for result in results:
        print(
            f"{result['final_score']:.4f} "
            f"dense={result['dense_score_norm']:.4f} "
            f"keyword={result['keyword_score_norm']:.4f} "
            f"{result['arxiv_id']} {result['title']}"
        )