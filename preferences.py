"""
Preference embedding and feedback handling
"""

import psycopg
from embeddings import embed_texts
import uuid
from config import DEFAULT_USER_ID
from db_helper import get_database_url
from vector_helper import vector_literal

# Convert pgvector strings to lists
def coerce_vector(raw_vector) -> list[float]:
    if isinstance(raw_vector, str):
        cleaned = raw_vector.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]

        if not cleaned:
            return []

        return [float(value.strip()) for value in cleaned.split(",") if value.strip()]

    return [float(value) for value in raw_vector]

# Cold start preference embedding
def initialize_preference_embedding(interest_text: str, user_id: str = DEFAULT_USER_ID) -> None:
    preference_vector = vector_literal(embed_texts([interest_text])[0])

    sql = """
    INSERT INTO user_preferences (
        user_id,
        initial_interest_embedding,
        preference_embedding
    )
    VALUES (
        %s,
        %s::vector,
        %s::vector
    )
    ON CONFLICT (user_id)
    DO UPDATE SET
        initial_interest_embedding = EXCLUDED.initial_interest_embedding,
        preference_embedding = EXCLUDED.preference_embedding,
        updated_at = NOW();
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, preference_vector, preference_vector))

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
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (user_id, arxiv_id)
    DO UPDATE SET
        label = EXCLUDED.label,
        created_at = NOW()
    RETURNING feedback_id;
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (feedback_id, user_id, arxiv_id, label))
            row = cur.fetchone()

    return str(row[0])

def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []

    dimension = len(vectors[0])

    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimension)
    ]

# Mix feedback with the initial preference embedding
def blend_vectors(
    initial_vector: list[float],
    feedback_vector: list[float],
    alpha: float,
) -> list[float]:
    return [
        alpha * initial_value + (1 - alpha) * feedback_value
        for initial_value, feedback_value in zip(initial_vector, feedback_vector)
    ]

# Decay initial weight
def feedback_alpha(num_feedbacks: int) -> float:
    return 1 / (1 + num_feedbacks)

# Normalise preference embedding so cosine distance compares direction rather than magnitude during ranking
def normalize_vector(vector: list[float]) -> list[float]:
    magnitude = sum(value * value for value in vector) ** 0.5

    if magnitude == 0:
        return vector

    return [value / magnitude for value in vector]

# Summarise overall feedback as a single vector
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
    SELECT
        up.initial_interest_embedding,
        e.embedding,
        f.label
    FROM user_preferences up
    LEFT JOIN paper_feedback f ON f.user_id = up.user_id
    LEFT JOIN paper_embeddings e ON e.arxiv_id = f.arxiv_id
    WHERE up.user_id = %s;
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No preference profile found for user_id={user_id}")

    initial_vector = coerce_vector(rows[0][0])

    liked_vectors = [
        coerce_vector(embedding)
        for _, embedding, label in rows
        if embedding is not None and label == "like"
    ]
    disliked_vectors = [
        coerce_vector(embedding)
        for _, embedding, label in rows
        if embedding is not None and label == "dislike"
    ]

    num_feedbacks = len(liked_vectors) + len(disliked_vectors)

    if not liked_vectors:
        preference_vector = initial_vector
    else:
        feedback_vector = compute_preference_vector(
            liked_vectors,
            disliked_vectors,
        )
        alpha = feedback_alpha(num_feedbacks)
        preference_vector = blend_vectors(
            initial_vector,
            feedback_vector,
            alpha,
        )

    preference_vector = normalize_vector(preference_vector)
    preference_vector_literal = vector_literal(preference_vector)

    save_sql = """
    UPDATE user_preferences
    SET preference_embedding = %s::vector,
        updated_at = NOW()
    WHERE user_id = %s;
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(save_sql, (preference_vector_literal, user_id))