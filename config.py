"""
Loads shared settings from environment variables
"""

import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default")
DEFAULT_DAILY_K = int(os.getenv("DAILY_PICKS_K", "3"))

def get_hybrid_weights() -> tuple[float, float]:
    dense = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.6"))
    keyword = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4"))

    if dense < 0 or keyword < 0:
        raise ValueError("Hybrid weights must be non-negative")

    total = dense + keyword
    if total == 0:
        raise ValueError("At least one hybrid weight must be greater than zero")

    return dense / total, keyword / total

def get_daily_picks_k() -> int:
    if DEFAULT_DAILY_K < 1:
        raise ValueError("DAILY_PICKS_K must be >= 1")

    return DEFAULT_DAILY_K

def get_arxiv_categories() -> list[str]:
    raw = os.getenv("ARXIV_CATEGORIES", "cs.AI")
    categories = [c.strip() for c in raw.split(",") if c.strip()]

    if not categories:
        raise ValueError("At least one arXiv category must be configured")

    return categories