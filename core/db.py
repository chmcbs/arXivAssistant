"""
Database connection utilities
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_database_url() -> str:
    return os.environ["DATABASE_URL"]
