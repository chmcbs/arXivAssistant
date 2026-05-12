"""
Searches stored papers using hybrid retrieval
"""

import psycopg
from embeddings import embed_texts
from config import get_hybrid_weights
from db_helper import get_database_url
from vector_helper import vector_literal

########################################
############# KEYWORD ##################
########################################

# Match papers_keyword_idx expression in db_setup.py so Postgres can use the GIN index
PAPER_SEARCH_VECTOR_SQL = """
setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
"""

def clean_query(query: str) -> str:
    return query.strip()

# Convert keywords to a Postgres tsquery, then compare to each paper's combined title/abstract tsvector and return the most relevant
def search_keyword_papers(query: str, limit: int = 10) -> list[dict]:
    cleaned_query = clean_query(query)

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

########################################
############### DENSE ##################
########################################

# Query paper_embeddings and return papers sorted by cosine similarity to the query vector
def search_dense_papers(query: str, limit: int = 10) -> list[dict]:
    cleaned_query = clean_query(query)

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

########################################
############### HYBRID #################
########################################

# Make keyword scores and dense scores comparable
def normalize_score(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0

    return score / max_score

# Combine search methods
def search_hybrid_papers(
    query: str,
    limit: int = 10,
    dense_weight: float | None = None,
    keyword_weight: float | None = None,
) -> list[dict]:
    if dense_weight is None and keyword_weight is None:
        dense_weight, keyword_weight = get_hybrid_weights()
    elif dense_weight is None or keyword_weight is None:
        raise ValueError("dense_weight and keyword_weight must be provided together")

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
