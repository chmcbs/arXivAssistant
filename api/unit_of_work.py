"""
Request-scoped unit-of-work object for API interactions
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
import psycopg
from core.db import get_database_url

@dataclass
class ApiUnitOfWork:
    conn: Any
    generated_run_ids: list[str] = field(default_factory=list)

    def set_generated_run_ids(self, run_ids: list[str]) -> None:
        self.generated_run_ids = list(dict.fromkeys(run_ids))

@contextmanager
def open_api_unit_of_work(
    uow: ApiUnitOfWork | None = None,
    conn=None,
):
    if uow is not None and conn is not None:
        raise ValueError("provide either uow or conn, not both")

    if uow is not None:
        yield uow
        return

    if conn is not None:
        yield ApiUnitOfWork(conn=conn)
        return

    with psycopg.connect(get_database_url()) as owned_conn:
        yield ApiUnitOfWork(conn=owned_conn)
