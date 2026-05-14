"""
Resets local recommendation state for the default profile
"""

import psycopg
from core.config import DEFAULT_INTEREST_TEXT, DEFAULT_USER_ID
from core.db import get_database_url
from core.preferences import initialize_preference_embedding
from core.profiles import get_or_create_default_profile

DELETE_PROFILE_FEEDBACK_SQL = """
DELETE FROM paper_feedback WHERE profile_id = %s;
"""

DELETE_PROFILE_RECOMMENDATIONS_SQL = """
DELETE FROM recommendations WHERE profile_id = %s;
"""

def reset_default_profile_state(
    user_id: str = DEFAULT_USER_ID,
    interest_text: str = DEFAULT_INTEREST_TEXT,
) -> None:
    profile = get_or_create_default_profile(
        user_id=user_id,
        interest_sentence=interest_text,
    )
    profile_id = str(profile.profile_id)

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(DELETE_PROFILE_FEEDBACK_SQL, (profile_id,))
            cur.execute(DELETE_PROFILE_RECOMMENDATIONS_SQL, (profile_id,))

    initialize_preference_embedding(
        interest_text,
        user_id=user_id,
        profile_id=profile_id,
    )

    print(f"Reset feedback and recommendations for profile_id={profile_id!r}")
    print(f"Initialized preference embedding from: {interest_text!r}")

if __name__ == "__main__":
    reset_default_profile_state()