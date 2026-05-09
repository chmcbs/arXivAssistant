import os
import psycopg
from dotenv import load_dotenv
from embeddings import embed_texts
import uuid


load_dotenv()

def get_database_url() -> str: # TODO: Move to a shared database helper module
    return os.environ["DATABASE_URL"]

def vector_literal(vector: list[float]) -> str: # TODO: Move to a shared vector helper module
    return "[" + ",".join(str(value) for value in vector) + "]"

DEFAULT_USER_ID = "default"

def initialize_preference_embedding(interest_text: str, user_id: str = DEFAULT_USER_ID) -> None:
    preference_vector = vector_literal(embed_texts([interest_text])[0])

    # TODO: Move to a helper function
    sql = """
    INSERT INTO user_preferences (
        user_id,
        preference_embedding
    )
    VALUES (
        %s,
        %s::vector
    )
    ON CONFLICT (user_id)
    DO UPDATE SET
        preference_embedding = EXCLUDED.preference_embedding,
        updated_at = NOW();
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, preference_vector))

def save_feedback(
    arxiv_id: str,
    label: str,
    user_id: str = DEFAULT_USER_ID,
) -> str:
    if label not in {"like", "dislike"}:
        raise ValueError("label must be 'like' or 'dislike'")

    feedback_id = str(uuid.uuid4())

    sql = """
    INSERT INTO paper_feedback (
        feedback_id,
        user_id,
        arxiv_id,
        label
    )
    VALUES (%s, %s, %s, %s);
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (feedback_id, user_id, arxiv_id, label))

    return feedback_id

def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []

    dimension = len(vectors[0])

    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimension)
    ]

def compute_preference_vector(
    liked_vectors: list[list[float]],
    disliked_vectors: list[list[float]],
    dislike_weight: float = 0.5,
) -> list[float]:
    liked_mean = mean_vector(liked_vectors)

    if not liked_mean:
        raise ValueError("At least one liked paper is required to update preferences")

    disliked_mean = mean_vector(disliked_vectors)

    if not disliked_mean:
        return liked_mean

    return [
        liked_value - dislike_weight * disliked_value
        for liked_value, disliked_value in zip(liked_mean, disliked_mean)
    ]

def update_preference_embedding(user_id: str = DEFAULT_USER_ID) -> None:
    sql = """
    SELECT e.embedding, f.label
    FROM paper_feedback f
    JOIN paper_embeddings e ON e.arxiv_id = f.arxiv_id
    WHERE f.user_id = %s;
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()

    liked_vectors = [
        list(embedding)
        for embedding, label in rows
        if label == "like"
    ]
    disliked_vectors = [
        list(embedding)
        for embedding, label in rows
        if label == "dislike"
    ]

    preference_vector = compute_preference_vector(
        liked_vectors,
        disliked_vectors,
    )
    preference_vector_literal = vector_literal(preference_vector)

    # TODO: Move to a helper function
    save_sql = """
    INSERT INTO user_preferences (
        user_id,
        preference_embedding
    )
    VALUES (
        %s,
        %s::vector
    )
    ON CONFLICT (user_id)
    DO UPDATE SET
        preference_embedding = EXCLUDED.preference_embedding,
        updated_at = NOW();
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(save_sql, (user_id, preference_vector_literal))