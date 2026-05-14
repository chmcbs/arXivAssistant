"""
Service helpers for metrics endpoint
"""

from typing import Callable
from api.mappers import to_metrics_payload

def get_metrics_payload(
    latest_runs_limit: int,
    fetch_metrics_rows: Callable[[int], object],
) -> dict:
    metrics_rows = fetch_metrics_rows(latest_runs_limit)
    return to_metrics_payload(metrics_rows)
