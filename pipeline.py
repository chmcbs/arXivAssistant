"""
Runs the end-to-end recommendation pipeline
"""

from config import DEFAULT_USER_ID
from db_setup import main as setup_database
from embeddings import run_embeddings
from ingestion import run_ingestion
from recommendations import generate_recommendations

def run_pipeline(
    user_id: str = DEFAULT_USER_ID,
    profile_id: str | None = None,
    max_results: int = 150,
    embedding_limit: int = 600,
) -> dict:
    print("1/4 Setting up database schema...")
    setup_database()

    print("2/4 Running ingestion...")
    run_ids = run_ingestion(max_results=max_results)
    print(f"Ingestion produced {len(run_ids)} run(s)")

    print("3/4 Generating embeddings...")
    embedded_count = run_embeddings(limit=embedding_limit)
    print(f"Embedded {embedded_count} paper(s)")

    print("4/4 Generating recommendations...")
    recommendations_by_run = {}
    for run_id in run_ids:
        try:
            recommendations = generate_recommendations(
                run_id,
                user_id=user_id,
                profile_id=profile_id,
            )
            recommendations_by_run[run_id] = recommendations
            print(f"Run {run_id}: saved {len(recommendations)} recommendation(s)")
        except Exception as error:
            recommendations_by_run[run_id] = []
            print(f"Run {run_id}: recommendation step failed: {error}")

    return {
        "run_ids": run_ids,
        "embedded_count": embedded_count,
        "recommendations_by_run": recommendations_by_run,
    }

if __name__ == "__main__":
    run_pipeline()
