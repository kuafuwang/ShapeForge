"""Per-model totals from the billed agent rows in ``transcript.jsonl``."""

from __future__ import annotations

import math
from pathlib import Path

from .reads import iter_jsonl


def _ratio(numerator, denominator, *, scale: float = 1.0) -> float | None:
    """Return a finite, non-negative ratio, or ``None`` for invalid inputs."""
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        return None
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    try:
        numerator_float = float(numerator)
        denominator_float = float(denominator)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(numerator_float) or not math.isfinite(denominator_float):
        return None
    if numerator_float < 0 or denominator_float <= 0:
        return None
    value = numerator_float * scale / denominator_float
    return round(value, 6) if math.isfinite(value) else None


def efficiency_fields(*, tokens_cache_read, tokens_output, cost_usd, sessions=None) -> dict:
    """Derive session/model cost efficiency without changing the recorded totals.

    ``sessions`` is supplied only for aggregate rows. Passing zero still returns the
    aggregate field with a null value; omitting it leaves that field out entirely.
    """
    fields = {
        "cache_read_per_output": _ratio(tokens_cache_read, tokens_output),
        "usd_per_1m_output": _ratio(cost_usd, tokens_output, scale=1_000_000),
    }
    if sessions is not None:
        fields["cost_per_session_usd"] = _ratio(cost_usd, sessions)
    return fields


class Totals:
    """增量聚合 agent 行：每个 provider/model/committee 一行，不保留输入事件。

    Streaming callers (``taic.brief``) feed one event at a time; ``from_events`` is the
    convenience wrapper for an iterable/sequence.
    """

    def __init__(self) -> None:
        """@brief 建一个空的聚合器。"""
        self._rows: dict[tuple[str, str, str], dict] = {}

    def add(self, event) -> None:
        """@brief 收一条事件；非 agent 行直接忽略。

        @param event 转录事件。
        """
        if not isinstance(event, dict) or event.get("event") != "agent":
            return
        provider = str(event.get("provider") or "")
        model = str(event.get("model") or "")
        committee = str(event.get("committee") or "")
        key = (provider, model, committee)
        row = self._rows.setdefault(key, {
            "provider": provider, "model": model, "committee": committee,
            "sessions": 0, "cost_usd": 0.0, "_cost_values": 0,
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
        })
        row["sessions"] += 1
        cost = event.get("cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(float(cost)):
            row["cost_usd"] += float(cost)
            row["_cost_values"] += 1
        tokens = event.get("tokens") if isinstance(event.get("tokens"), dict) else {}
        for key_name, token_name in (
            ("input_tokens", "input"), ("output_tokens", "output"),
            ("cache_read_tokens", "cache_read"), ("cache_write_tokens", "cache_write"),
        ):
            value = tokens.get(token_name, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                row[key_name] += int(value)

    def result(self) -> list[dict]:
        """@brief 结算成排好序的模型费用行。

        @return 每个 provider/model/committee 一行的列表。
        """
        rows = list(self._rows.values())
        for row in rows:
            row["cost_usd"] = round(row["cost_usd"], 6)
            cost_values = row.pop("_cost_values")
            row["sessions_with_cost"] = cost_values
            cost = row["cost_usd"] if cost_values else None
            row.update(efficiency_fields(
                tokens_cache_read=row["cache_read_tokens"], tokens_output=row["output_tokens"],
                cost_usd=cost, sessions=row["sessions"],
            ))
        return sorted(rows, key=lambda row: (-row["cost_usd"], -row["sessions"], row["provider"], row["model"], row["committee"]))


_TOKEN_KEYS = (
    ("input_tokens", "input"), ("output_tokens", "output"),
    ("cache_read_tokens", "cache_read"), ("cache_write_tokens", "cache_write"),
)


def _finite_or_none(value):
    """@brief 有限非负数字返回 float，其余（含 bool）返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def session_rows(path: Path) -> list[dict]:
    """@brief 逐条读出计费会话行，不做跨会话合并。

    这是 :func:`from_events` 的明细版本：``from_events`` 按 provider/model/committee 聚合，
    这里保留每个 ``event=agent`` / ``event=chat`` 行本身，供需要逐场对账的调用方使用。
    费用缺失或非有限时 ``cost_usd`` 保持 ``None``，由调用方决定是否 fail closed；
    本函数不改动既有聚合语义，也不替调用方补 0。

    @param path ``transcript.jsonl`` 路径。
    @return 按文件顺序的会话行列表；文件不可读时为空列表。
    """
    rows = []
    for event in iter_jsonl(path):
        if not isinstance(event, dict) or event.get("event") not in {"agent", "chat"}:
            continue
        cost = _finite_or_none(event.get("cost_usd"))
        tokens = event.get("tokens") if isinstance(event.get("tokens"), dict) else {}
        counts = {}
        for key_name, token_name in _TOKEN_KEYS:
            counts[key_name] = _finite_or_none(tokens.get(token_name)) or 0
        rows.append({
            "event": str(event.get("event")),
            "label": str(event.get("label") or ""),
            "provider": str(event.get("provider") or ""),
            "model": str(event.get("model") or ""),
            "committee": str(event.get("committee") or ""),
            "cost_usd": cost if cost is not None and cost >= 0 else None,
            "cost_incomplete": event.get("cost_incomplete"),
            "outcome": str(event.get("outcome") or ""),
            **{name: int(value) for name, value in counts.items()},
        })
    return rows


def from_events(events) -> list[dict]:
    """Aggregate one row per provider/model/committee from agent events.

    Retry scheduling rows and chat rows are not sessions; only ``event=agent`` carries
    the billed token counts. A non-finite cost is ignored rather than poisoning the total.
    """
    totals = Totals()
    for event in events:
        totals.add(event)
    return totals.result()


def from_path(path: Path) -> list[dict]:
    """Read a transcript path and aggregate its agent rows; missing/invalid files yield []."""
    return from_events(iter_jsonl(path))
