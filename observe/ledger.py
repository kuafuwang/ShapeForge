"""Spend totals stored in the observe index. A later reading may raise a total. It never lowers one."""

from __future__ import annotations

import time


def operator_name(task: str | None) -> str:
    """The operator a search belongs to. A tile variant stays with its operator."""
    name = (task or "").strip()
    if not name:
        return "unknown"
    return name.split("@", 1)[0]


def month_of(started_at: float | None) -> str:
    """``YYYY-MM`` in local time. A missing start is ``unknown``."""
    if started_at is None:
        return "unknown"
    try:
        return time.strftime("%Y-%m", time.localtime(float(started_at)))
    except (TypeError, ValueError, OSError):
        return "unknown"
