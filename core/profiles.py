"""
User profile model and helpers
"""

import uuid
import psycopg
from core.config import DEFAULT_INTEREST_TEXT, DEFAULT_USER_ID, get_arxiv_categories
from core.db import get_database_url
from core.keyword_search import MAX_KEYWORDS_PER_PROFILE, normalize_keyword

MAX_PROFILES_PER_USER = 3

def _validate_interest_sentence(interest_sentence: str) -> str:
    value = interest_sentence.strip()
    if not value:
        raise ValueError("interest_sentence must not be empty")
    return value

def _validate_category(category: str) -> str:
    value = category.strip()
    if not value:
        raise ValueError("category must not be empty")
    allowed_categories = set(get_arxiv_categories())
    if value not in allowed_categories:
        raise ValueError(
            "category must be one of configured ARXIV_CATEGORIES: "
            + ", ".join(sorted(allowed_categories))
        )
    return value

def _pick_next_available_slot(occupied_slots: set[int]) -> int:
    for slot in range(1, MAX_PROFILES_PER_USER + 1):
        if slot not in occupied_slots:
            return slot
    raise ValueError(f"user has reached the {MAX_PROFILES_PER_USER}-profile cap")

def create_profile(
    user_id: str = DEFAULT_USER_ID,
    category: str | None = None,
    interest_sentence: str = DEFAULT_INTEREST_TEXT,
) -> str:
    validated_interest = _validate_interest_sentence(interest_sentence)
    validated_category = _validate_category(
        category or get_arxiv_categories()[0]
    )
    profile_id = str(uuid.uuid4())

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT profile_slot
                FROM user_profiles
                WHERE user_id = %s
                ORDER BY profile_slot ASC
                FOR UPDATE;
                """,
                (user_id,),
            )
            occupied_slots = {int(row[0]) for row in cur.fetchall()}

            profile_slot = _pick_next_available_slot(occupied_slots)

            cur.execute(
                """
                INSERT INTO user_profiles (
                    profile_id,
                    user_id,
                    profile_slot,
                    category,
                    interest_sentence
                )
                VALUES (%s, %s, %s, %s, %s);
                """,
                (
                    profile_id,
                    user_id,
                    profile_slot,
                    validated_category,
                    validated_interest,
                ),
            )

    return profile_id

def list_profiles(user_id: str = DEFAULT_USER_ID) -> list[dict]:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    profile_id::text,
                    user_id,
                    profile_slot,
                    category,
                    interest_sentence,
                    created_at,
                    digest_enabled
                FROM user_profiles
                WHERE user_id = %s
                ORDER BY profile_slot ASC;
                """,
                (user_id,),
            )
            rows = cur.fetchall()

    return [
        {
            "profile_id": row[0],
            "user_id": row[1],
            "profile_slot": int(row[2]),
            "category": row[3],
            "interest_sentence": row[4],
            "created_at": row[5],
            "digest_enabled": bool(row[6]),
        }
        for row in rows
    ]

def get_profile(profile_id: str) -> dict | None:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    profile_id::text,
                    user_id,
                    profile_slot,
                    category,
                    interest_sentence,
                    created_at,
                    digest_enabled
                FROM user_profiles
                WHERE profile_id = %s;
                """,
                (profile_id,),
            )
            row = cur.fetchone()

    if row is None:
        return None

    return {
        "profile_id": row[0],
        "user_id": row[1],
        "profile_slot": int(row[2]),
        "category": row[3],
        "interest_sentence": row[4],
        "created_at": row[5],
        "digest_enabled": bool(row[6]),
    }

def get_or_create_default_profile(
    user_id: str = DEFAULT_USER_ID,
    category: str | None = None,
    interest_sentence: str = DEFAULT_INTEREST_TEXT,
) -> dict:
    profiles = list_profiles(user_id=user_id)
    if profiles:
        return profiles[0]

    profile_id = create_profile(
        user_id=user_id,
        category=category,
        interest_sentence=interest_sentence,
    )
    profile = get_profile(profile_id)
    if profile is None:
        raise ValueError("failed to create default profile")
    return profile

def list_profile_keywords(
    profile_id: str,
    user_id: str = DEFAULT_USER_ID,
) -> list[str]:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM user_profiles
                WHERE profile_id = %s
                  AND user_id = %s;
                """,
                (profile_id, user_id),
            )
            if cur.fetchone() is None:
                raise ValueError("profile not found for user")

            cur.execute(
                """
                SELECT keyword
                FROM profile_keywords
                WHERE profile_id = %s
                ORDER BY keyword ASC;
                """,
                (profile_id,),
            )
            rows = cur.fetchall()

    return [row[0] for row in rows]

def add_profile_keyword(
    profile_id: str,
    keyword: str,
    user_id: str = DEFAULT_USER_ID,
) -> list[str]:
    normalized_keyword = normalize_keyword(keyword)

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM user_profiles
                WHERE profile_id = %s
                  AND user_id = %s;
                """,
                (profile_id, user_id),
            )
            if cur.fetchone() is None:
                raise ValueError("profile not found for user")

            cur.execute(
                """
                SELECT COUNT(*)
                FROM profile_keywords
                WHERE profile_id = %s;
                """,
                (profile_id,),
            )
            current_count = int(cur.fetchone()[0])

            cur.execute(
                """
                INSERT INTO profile_keywords (profile_id, keyword)
                VALUES (%s, %s)
                ON CONFLICT (profile_id, keyword) DO NOTHING
                RETURNING keyword;
                """,
                (profile_id, normalized_keyword),
            )
            inserted = cur.fetchone()
            if inserted is not None and current_count >= MAX_KEYWORDS_PER_PROFILE:
                raise ValueError(
                    f"profile keyword cap reached ({MAX_KEYWORDS_PER_PROFILE})"
                )

            cur.execute(
                """
                SELECT keyword
                FROM profile_keywords
                WHERE profile_id = %s
                ORDER BY keyword ASC;
                """,
                (profile_id,),
            )
            rows = cur.fetchall()

    return [row[0] for row in rows]

def remove_profile_keyword(
    profile_id: str,
    keyword: str,
    user_id: str = DEFAULT_USER_ID,
) -> list[str]:
    normalized_keyword = normalize_keyword(keyword)

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM user_profiles
                WHERE profile_id = %s
                  AND user_id = %s;
                """,
                (profile_id, user_id),
            )
            if cur.fetchone() is None:
                raise ValueError("profile not found for user")

            cur.execute(
                """
                DELETE FROM profile_keywords
                WHERE profile_id = %s
                  AND keyword = %s;
                """,
                (profile_id, normalized_keyword),
            )

            cur.execute(
                """
                SELECT keyword
                FROM profile_keywords
                WHERE profile_id = %s
                ORDER BY keyword ASC;
                """,
                (profile_id,),
            )
            rows = cur.fetchall()

    return [row[0] for row in rows]

def list_digest_selected_profile_ids(user_id: str = DEFAULT_USER_ID) -> list[str]:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT profile_id::text
                FROM user_profiles
                WHERE user_id = %s
                  AND digest_enabled = TRUE
                ORDER BY profile_slot ASC;
                """,
                (user_id,),
            )
            rows = cur.fetchall()

    return [row[0] for row in rows]

def set_digest_profile_selection(
    profile_ids: list[str],
    user_id: str = DEFAULT_USER_ID,
) -> list[str]:
    requested_profile_ids = list(dict.fromkeys(profile_ids))
    if not requested_profile_ids:
        raise ValueError("at least one profile must be selected for digest generation")

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT profile_id::text
                FROM user_profiles
                WHERE user_id = %s
                  AND profile_id = ANY(%s::uuid[]);
                """,
                (user_id, requested_profile_ids),
            )
            matched_ids = {row[0] for row in cur.fetchall()}
            missing_ids = sorted(set(requested_profile_ids) - matched_ids)
            if missing_ids:
                raise ValueError("some profile_ids do not belong to user")

            cur.execute(
                """
                UPDATE user_profiles
                SET digest_enabled = FALSE
                WHERE user_id = %s;
                """,
                (user_id,),
            )
            cur.execute(
                """
                UPDATE user_profiles
                SET digest_enabled = TRUE
                WHERE user_id = %s
                  AND profile_id = ANY(%s::uuid[]);
                """,
                (user_id, requested_profile_ids),
            )

            cur.execute(
                """
                SELECT profile_id::text
                FROM user_profiles
                WHERE user_id = %s
                  AND digest_enabled = TRUE
                ORDER BY profile_slot ASC;
                """,
                (user_id,),
            )
            rows = cur.fetchall()

    selected = [row[0] for row in rows]
    if not selected:
        raise ValueError("at least one profile must be selected for digest generation")
    return selected
