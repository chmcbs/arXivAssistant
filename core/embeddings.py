"""
Reads papers without embeddings, generates vectors, and stores them in the paper_embeddings table
"""

import psycopg
from core.db import get_database_url
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FETCH_PAPERS_MISSING_EMBEDDINGS_SQL = """
SELECT p.arxiv_id, p.title, p.abstract
FROM papers p
LEFT JOIN paper_embeddings e ON p.arxiv_id = e.arxiv_id
WHERE e.arxiv_id IS NULL
ORDER BY p.inserted_at ASC
LIMIT %s;
"""

UPSERT_EMBEDDING_SQL = """
INSERT INTO paper_embeddings (
    arxiv_id,
    embedding,
    model_name
)
VALUES (
    %(arxiv_id)s,
    %(embedding)s,
    %(model_name)s
)
ON CONFLICT (arxiv_id)
DO UPDATE SET
    embedding = EXCLUDED.embedding,
    model_name = EXCLUDED.model_name,
    embedded_at = NOW();
"""

def get_papers_missing_embeddings(limit: int) -> list[dict]:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(FETCH_PAPERS_MISSING_EMBEDDINGS_SQL, (limit,))
            rows = cur.fetchall()

    return [
        {
            "arxiv_id": row[0],
            "title": row[1],
            "abstract": row[2] or "",
        }
        for row in rows
    ]

def paper_text(paper: dict) -> str:
    return f"{paper['title']}\n\n{paper['abstract']}"

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True)

    return embeddings.tolist()

def save_embeddings(papers: list[dict], embeddings: list[list[float]]) -> int:
    rows = [
        {
            "arxiv_id": paper["arxiv_id"],
            "embedding": embedding,
            "model_name": MODEL_NAME,
        }
        for paper, embedding in zip(papers, embeddings)
    ]

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_EMBEDDING_SQL, rows)

    return len(rows)

def run_embeddings(limit: int = 600) -> int:
    papers = get_papers_missing_embeddings(limit)

    if not papers:
        print("No papers missing embeddings")
        return 0

    texts = [paper_text(paper) for paper in papers]
    embeddings = embed_texts(texts)
    saved_count = save_embeddings(papers, embeddings)

    print(f"Saved embeddings for {saved_count} papers")
    return saved_count

if __name__ == "__main__":
    run_embeddings()