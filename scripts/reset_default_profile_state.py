"""
Resets local recommendation state for the default profile
"""

import psycopg
from config import DEFAULT_INTEREST_TEXT, DEFAULT_USER_ID
from db_helper import get_database_url
from preferences import initialize_preference_embedding
from profiles import get_or_create_default_profile

def reset_default_profile_state(
    user_id: str = DEFAULT_USER_ID,
    interest_text: str = DEFAULT_INTEREST_TEXT,
) -> None:
    profile = get_or_create_default_profile(
        user_id=user_id,
        interest_sentence=interest_text,
    )
    profile_id = str(profile["profile_id"])

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM paper_feedback WHERE profile_id = %s;", (profile_id,))
            cur.execute("DELETE FROM recommendations WHERE profile_id = %s;", (profile_id,))

    initialize_preference_embedding(
        interest_text,
        user_id=user_id,
        profile_id=profile_id,
    )

    print(f"Reset feedback and recommendations for profile_id={profile_id!r}")
    print(f"Initialized preference embedding from: {interest_text!r}")

if __name__ == "__main__":
    reset_default_profile_state()