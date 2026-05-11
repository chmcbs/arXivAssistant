"""
Searches stored papers using hybrid retrieval
"""

import psycopg
from embeddings import embed_texts
from config import DEFAULT_USER_ID, get_hybrid_weights
from db_helper import get_database_url
from vector_helper import vector_literal

########################################
############### SPARSE #################
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

########################################
########## PERSONALIZATION #############
########################################

def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_magnitude = sum(value * value for value in left) ** 0.5
    right_magnitude = sum(value * value for value in right) ** 0.5

    if left_magnitude == 0 or right_magnitude == 0:
        return 0.0

    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    ) / (left_magnitude * right_magnitude)

def normalize_cosine_score(score: float) -> float:
    return (score + 1) / 2

def get_user_preference_vector(user_id: str = DEFAULT_USER_ID) -> list[float] | None:
    sql = """
    SELECT preference_embedding
    FROM user_preferences
    WHERE user_id = %s;
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            row = cur.fetchone()

    if row is None:
        return None

    return list(row[0])

def get_paper_embedding_map(arxiv_ids: list[str]) -> dict[str, list[float]]:
    if not arxiv_ids:
        return {}

    sql = """
    SELECT arxiv_id, embedding
    FROM paper_embeddings
    WHERE arxiv_id = ANY(%s);
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (arxiv_ids,))
            rows = cur.fetchall()

    return {
        arxiv_id: list(embedding)
        for arxiv_id, embedding in rows
    }

def search_personalized_papers(
    query: str,
    user_id: str = DEFAULT_USER_ID,
    limit: int = 10,
    candidate_multiplier: int = 3,
    hybrid_weight: float = 0.7,
    preference_weight: float = 0.3,
) -> list[dict]:
    if hybrid_weight < 0 or preference_weight < 0:
        raise ValueError("Personalized ranking weights must be non-negative")

    total_weight = hybrid_weight + preference_weight
    if total_weight == 0:
        raise ValueError("At least one personalized ranking weight must be greater than zero")

    hybrid_weight = hybrid_weight / total_weight
    preference_weight = preference_weight / total_weight

    candidates = search_hybrid_papers(
        query,
        limit=limit * candidate_multiplier,
    )

    if not candidates:
        return []

    preference_vector = get_user_preference_vector(user_id)

    if preference_vector is None:
        return candidates[:limit]

    embedding_map = get_paper_embedding_map(
        [candidate["arxiv_id"] for candidate in candidates]
    )

    personalized_results = []

    for candidate in candidates:
        paper_embedding = embedding_map.get(candidate["arxiv_id"])

        if paper_embedding is None:
            preference_similarity = 0.0
        else:
            preference_similarity = cosine_similarity(
                preference_vector,
                paper_embedding,
            )

        preference_score = normalize_cosine_score(preference_similarity)
        personalized_score = (
            hybrid_weight * candidate["final_score"]
            + preference_weight * preference_score
        )

        personalized_results.append(
            {
                **candidate,
                "preference_similarity": preference_similarity,
                "preference_score": preference_score,
                "personalized_score": personalized_score,
            }
        )

    return sorted(
        personalized_results,
        key=lambda result: result["personalized_score"],
        reverse=True,
    )[:limit]

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "language agents"
    results = search_personalized_papers(query, limit=5)

    for result in results:
        print(
            f"{result.get('personalized_score', result['final_score']):.4f} "
            f"hybrid={result['final_score']:.4f} "
            f"dense={result['dense_score_norm']:.4f} "
            f"keyword={result['keyword_score_norm']:.4f} "
            f"{result['arxiv_id']} {result['title']}"
        )