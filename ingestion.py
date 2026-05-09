"""
Fetches papers from arXiv and stores them idempotently
"""

import os
import re
import arxiv
import psycopg
from dotenv import load_dotenv
import uuid

load_dotenv()

def get_database_url() -> str:
    return os.environ["DATABASE_URL"]

def fetch_papers(
    category: str = 'cs.AI',
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    ):
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    return list(client.results(search))

def clean_id(arxiv_id: str) -> str:
    """
    Convert IDs so different versions update to the same paper row
    """
    return re.sub(r"v\d+$", "", arxiv_id)

def save_papers(papers: list[arxiv.Result]) -> int:
    """
    Insert or update papers by canonical arXiv ID
    """
    query = """
    INSERT INTO papers (
        arxiv_id,
        title,
        abstract,
        authors,
        published_at,
        updated_at,
        pdf_url,
        entry_url,
        categories
    )
    VALUES (
        %(arxiv_id)s,
        %(title)s,
        %(abstract)s,
        %(authors)s,
        %(published_at)s,
        %(updated_at)s,
        %(pdf_url)s,
        %(entry_url)s,
        %(categories)s
    )
    ON CONFLICT (arxiv_id)
    DO UPDATE SET
        title = EXCLUDED.title,
        abstract = EXCLUDED.abstract,
        authors = EXCLUDED.authors,
        published_at = EXCLUDED.published_at,
        updated_at = EXCLUDED.updated_at,
        pdf_url = EXCLUDED.pdf_url,
        entry_url = EXCLUDED.entry_url,
        categories = EXCLUDED.categories;
    """
    rows = [
        {
            "arxiv_id": clean_id(paper.get_short_id()),
            "title": paper.title,
            "abstract": paper.summary,
            "authors": [str(author) for author in paper.authors],
            "published_at": paper.published,
            "updated_at": paper.updated,
            "pdf_url": paper.pdf_url,
            "entry_url": paper.entry_id,
            "categories": paper.categories,
        }
        for paper in papers
    ]
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, rows)
    return len(rows)

def start_run(category: str, max_results: int) -> str:
    run_id = str(uuid.uuid4())

    query = """
    INSERT INTO runs (run_id, status, category, max_results)
    VALUES (%s, 'running', %s, %s);
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (run_id, category, max_results))

    return run_id

def complete_run(run_id: str, fetched_count: int, saved_count: int) -> None:
    query = """
    UPDATE runs
    SET status = 'completed',
        fetched_count = %s,
        saved_count = %s,
        finished_at = NOW()
    WHERE run_id = %s;
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (fetched_count, saved_count, run_id))

def fail_run(run_id: str, error_message: str) -> None:
    query = """
    UPDATE runs
    SET status = 'failed',
        finished_at = NOW(),
        error_message = %s
    WHERE run_id = %s;
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (error_message, run_id))

def run_ingestion(category: str = "cs.AI", max_results: int = 10) -> str:
    run_id = start_run(category, max_results)

    try:
        papers = fetch_papers(category=category, max_results=max_results)
        saved_count = save_papers(papers)
        complete_run(run_id, len(papers), saved_count)
        print(f"Run {run_id} saved {saved_count} papers")
        return run_id
    except Exception as error:
        fail_run(run_id, str(error))
        raise

if __name__ == "__main__":
    run_ingestion()