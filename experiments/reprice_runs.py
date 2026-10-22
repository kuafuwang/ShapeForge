"""Recompute what a finished hmz run actually cost, from its transcript token counts.

The live ledger for a coding-agent session records the dollars it was charged but, before this
change, wrote ``hit=miss=out=0`` for the token columns. This script is the historical repair:
it reads ``transcript.jsonl`` ``event=agent`` rows, prices each session with the model's own
``cost`` table from ``~/.pi/agent/models.json``, and writes ``cost_recomputed.json`` next to
``result.json``.

It never changes ``result.json.spent_usd``. That number is what the budget gate used while the
run was live; rewriting it would make a finished run look different from the one that stopped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RUNS = (
    "runs/apple-m4-max/flash_attention-20261022-2",
    "runs/apple-m4-max/conv2d-feedback-20261021",
    "runs/apple-m4-max/conv2d-flame-20261020",
)
_KEYS = ("input", "output", "cache_read", "cache_write")


def load_prices(path: Path | None = None) -> dict[str, dict]:
    """model id -> USD per million tokens, from the same file the session used."""
    source = Path(path) if path else Path.home() / ".pi" / "agent" / "models.json"
    doc = json.loads(source.read_text(encoding="utf-8"))
    table: dict[str, dict] = {}
    providers = doc.get("providers") if isinstance(doc, dict) else None
    for _name, provider in (providers or {}).items():
        if not isinstance(provider, dict):
            continue
        for model in provider.get("models") or []:
            if not isinstance(model, dict) or not model.get("id"):
                continue
            cost = model.get("cost") or {}
            table[str(model["id"])] = {
                "input": float(cost.get("input", 0.30)),
                "output": float(cost.get("output", 1.20)),
                "cache_read": float(cost.get("cacheRead", cost.get("input", 0.30))),
                "cache_write": float(cost.get("cacheWrite", cost.get("input", 0.30)) or 0.0),
            }
    return table


def cost_of(tokens: dict, prices: dict) -> float:
    """USD for one session's token dict (``input``/``output``/``cache_read``/``cache_write``)."""
    total = 0.0
    for key in _KEYS:
        value = tokens.get(key, 0)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value) / 1_000_000 * float(prices.get(key, 0.0))
    return total


def recompute_run(run_dir: Path, table: dict[str, dict], *, write: bool = True) -> dict:
    """The per-session and total recomputation for one run. Writes ``cost_recomputed.json`` when ``write``."""
    run = Path(run_dir)
    result_path = run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    recorded = result.get("spent_usd") if isinstance(result, dict) else None
    sessions = []
    missing = set()
    transcript = run / "transcript.jsonl"
    if transcript.is_file():
        for line in transcript.open(encoding="utf-8", errors="replace"):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("event") != "agent":
                continue
            model = str(event.get("model") or "")
            prices = table.get(model)
            tokens = event.get("tokens") if isinstance(event.get("tokens"), dict) else {}
            if prices is None:
                missing.add(model)
                prices = {key: 0.0 for key in _KEYS}
            sessions.append({
                "label": event.get("label"),
                "model": model,
                "provider": event.get("provider"),
                "tokens": {key: int(tokens.get(key, 0) or 0) for key in _KEYS},
                "recorded_usd": event.get("cost_usd"),
                "recomputed_usd": round(cost_of(tokens, prices), 9),
                "prices": prices,
            })
    recomputed = sum(row["recomputed_usd"] for row in sessions)
    doc = {
        "schema_version": 1,
        "run": str(run),
        "recorded_usd": recorded,
        "recomputed_usd": round(recomputed, 9),
        "delta_usd": None if not isinstance(recorded, (int, float)) else round(recomputed - float(recorded), 9),
        "sessions": sessions,
        "missing_models": sorted(missing),
        "rule": "sum(input*price_in + output*price_out + cache_read*price_read + cache_write*price_write) / 1e6; spent_usd is left unchanged",
    }
    if write:
        (run / "cost_recomputed.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.reprice_runs")
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    parser.add_argument("--models", default=None, help="models.json path (default: ~/.pi/agent/models.json)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    table = load_prices(Path(args.models) if args.models else None)
    for name in args.runs:
        run = Path(name)
        if not run.is_dir():
            print(json.dumps({"run": name, "error": "not a directory"}, ensure_ascii=False))
            continue
        doc = recompute_run(run, table, write=not args.dry_run)
        print(json.dumps({
            "run": doc["run"], "recorded_usd": doc["recorded_usd"], "recomputed_usd": doc["recomputed_usd"],
            "delta_usd": doc["delta_usd"], "sessions": len(doc["sessions"]), "missing_models": doc["missing_models"],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
