"""Per-model totals from the billed agent rows in ``transcript.jsonl``."""

from __future__ import annotations

import json
import math
from pathlib import Path


def from_events(events) -> list[dict]:
    """Aggregate one row per provider/model/committee from agent events.

    Retry scheduling rows and chat rows are not sessions; only ``event=agent`` carries
    the billed token counts. A non-finite cost is ignored rather than poisoning the total.
    """
    totals: dict[tuple[str, str, str], dict] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("event") != "agent":
            continue
        provider = str(event.get("provider") or "")
        model = str(event.get("model") or "")
        committee = str(event.get("committee") or "")
        key = (provider, model, committee)
        row = totals.setdefault(key, {
            "provider": provider, "model": model, "committee": committee,
            "sessions": 0, "cost_usd": 0.0,
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
        })
        row["sessions"] += 1
        cost = event.get("cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(float(cost)):
            row["cost_usd"] += float(cost)
        tokens = event.get("tokens") if isinstance(event.get("tokens"), dict) else {}
        for key_name, token_name in (
            ("input_tokens", "input"), ("output_tokens", "output"),
            ("cache_read_tokens", "cache_read"), ("cache_write_tokens", "cache_write"),
        ):
            value = tokens.get(token_name, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                row[key_name] += int(value)
    rows = list(totals.values())
    for row in rows:
        row["cost_usd"] = round(row["cost_usd"], 6)
    return sorted(rows, key=lambda row: (-row["cost_usd"], -row["sessions"], row["provider"], row["model"], row["committee"]))


def from_path(path: Path) -> list[dict]:
    """Read a transcript path and aggregate its agent rows; missing/invalid files yield []."""
    events = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return from_events(events)
