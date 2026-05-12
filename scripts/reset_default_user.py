"""
Resets the local default user's recommendation state
"""

import psycopg
from config import DEFAULT_INTEREST_TEXT, DEFAULT_USER_ID
from db_helper import get_database_url
from preferences import initialize_preference_embedding

def reset_user_state(
    user_id: str = DEFAULT_USER_ID,
    interest_text: str = DEFAULT_INTEREST_TEXT,
) -> None:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM paper_feedback WHERE user_id = %s;", (user_id,))
            cur.execute("DELETE FROM recommendations WHERE user_id = %s;", (user_id,))

    initialize_preference_embedding(interest_text, user_id=user_id)

    print(f"Reset feedback and recommendations for user_id={user_id!r}")
    print(f"Initialized preference embedding from: {interest_text!r}")

if __name__ == "__main__":
    reset_user_state()