"""
Progress monitoring for core pipeline
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

_PROGRESS_USER: ContextVar[str | None] = ContextVar("pipeline_progress_user", default=None)

STEP_LABELS: dict[str, str] = {
    "starting": "Starting…",
    "ingestion": "Fetching papers…",
    "embeddings": "Generating embeddings…",
    "recommendations": "Ranking papers…",
    "descriptions": "Writing descriptions…",
    "finishing": "Preparing digest…",
}


@dataclass(frozen=True)
class PipelineProgressSnapshot:
    active: bool
    step: str | None = None
    label: str | None = None
    detail: str | None = None
    updated_at: datetime | None = None

    def as_dict(self) -> dict:
        payload = {
            "active": self.active,
            "step": self.step,
            "label": self.label,
            "detail": self.detail,
        }
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at.isoformat()
        return payload


@dataclass
class _ProgressEntry:
    step: str
    label: str
    detail: str | None
    updated_at: datetime


_lock = Lock()
_by_user: dict[str, _ProgressEntry] = {}


def _label_for_step(step: str) -> str:
    return STEP_LABELS.get(step, step.replace("_", " ").capitalize() + "…")


def begin(user_id: str, *, step: str = "starting", detail: str | None = None) -> None:
    set_step(step, detail=detail, user_id=user_id)


def set_step(
    step: str,
    *,
    detail: str | None = None,
    user_id: str | None = None,
) -> None:
    resolved_user_id = user_id or _PROGRESS_USER.get()
    if not resolved_user_id:
        return

    entry = _ProgressEntry(
        step=step,
        label=_label_for_step(step),
        detail=detail,
        updated_at=datetime.now(UTC),
    )
    with _lock:
        _by_user[resolved_user_id] = entry


def clear(user_id: str) -> None:
    with _lock:
        _by_user.pop(user_id, None)


def get_progress(user_id: str) -> PipelineProgressSnapshot:
    with _lock:
        entry = _by_user.get(user_id)

    if entry is None:
        return PipelineProgressSnapshot(active=False)

    return PipelineProgressSnapshot(
        active=True,
        step=entry.step,
        label=entry.label,
        detail=entry.detail,
        updated_at=entry.updated_at,
    )


@contextmanager
def track_pipeline(user_id: str):
    token = _PROGRESS_USER.set(user_id)
    begin(user_id)
    try:
        yield
    finally:
        _PROGRESS_USER.reset(token)
        clear(user_id)
