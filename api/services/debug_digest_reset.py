"""
Debug-only helpers for clearing application data
"""

from typing import Any

from core.preferences import reset_all_preference_embeddings


def reset_papers_and_runs(conn: Any) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.runs")
        runs_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM public.papers")
        papers_count = int(cur.fetchone()[0])
        cur.execute("DELETE FROM public.runs")
        cur.execute("DELETE FROM public.papers")
    reset_preference_embeddings = reset_all_preference_embeddings(conn)
    conn.commit()
    return {
        "deleted_runs": runs_count,
        "deleted_papers": papers_count,
        "reset_preference_embeddings": reset_preference_embeddings,
    }


def reset_user_profiles(conn: Any) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.user_profiles")
        profiles_count = int(cur.fetchone()[0])
        cur.execute("DELETE FROM public.user_profiles")
    conn.commit()
    return {
        "deleted_profiles": profiles_count,
    }
