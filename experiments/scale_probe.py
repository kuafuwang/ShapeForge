"""Measure the observe index and retrieval paths on synthetic runs.

The 10^5 extrapolation in the data-layer review had not been measured. This script creates
minimal run directories in a temporary root, indexes them with the real ``observe.index.Index``
and times the request paths. It never touches ``runs/``, the GPU or the model.

    uv run python -m experiments.scale_probe --counts 1000,10000,100000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

_STATE = json.dumps({
    "task": "relu", "dtype": "fp16", "size": "small", "phase": "done",
    "started_at": 1, "updated_at": 1, "spent_usd": 1.0, "budget_usd": 5.0,
    "model": "model-a", "best_speedup": 1.0,
}).encode()
_RESULT = json.dumps({
    "task": "relu", "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
    "spent_usd": 1.0, "budget_usd": 5.0, "started_at": 1, "best_speedup": 1.0,
    "confirmed_speedup": 1.0, "confirmed_status": "ok", "publication": "confirmed",
    "published_speedup": 1.0, "protocol_version": "2",
}).encode()


def _make_root(root: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    state = root / ".state.template"
    result = root / ".result.template"
    state.write_bytes(_STATE)
    result.write_bytes(_RESULT)
    width = max(6, len(str(count)))
    for index in range(count):
        directory = root / f"run_{index:0{width}d}"
        directory.mkdir()
        os.link(state, directory / "state.json")
        os.link(result, directory / "result.json")
    state.unlink()
    result.unlink()


def _reset_singletons() -> None:
    import observe.index as index
    import observe.query as query

    index._SINGLE = None
    query._HISTORY["key"] = None
    query._HISTORY["rows"] = []
    query._FEATURE_CACHE.clear()
    query._TICK["at"] = 0.0


def measure(count: int) -> dict:
    from observe.index import Index
    from observe import query
    from taic.brief import summarize_run

    root = Path(tempfile.mkdtemp(prefix=f"taic-scale-{count}-"))
    db = root / ".index.sqlite"
    try:
        t0 = time.perf_counter()
        _make_root(root / "runs", count)
        create_s = time.perf_counter() - t0
        os.environ["AI_LOWERLING_RUNS_DIR"] = str(root / "runs")
        os.environ["OBSERVE_DB"] = str(db)
        os.environ["AI_LOWERLING_BOOKS"] = str(root / "none.sqlite")
        os.environ["OBSERVE_TICK_INTERVAL"] = "0"
        _reset_singletons()
        t0 = time.perf_counter()
        index = Index(db)
        index.sync_known()
        sync_s = time.perf_counter() - t0
        rows = index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

        _reset_singletons()
        t0 = time.perf_counter()
        history = query.history()
        history_s = time.perf_counter() - t0

        _reset_singletons()
        t0 = time.perf_counter()
        index.tick()
        tick_s = time.perf_counter() - t0

        _reset_singletons()
        t0 = time.perf_counter()
        first = query.runs_page(limit=50)
        page_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        second = query.runs_page(limit=50, cursor=first.get("next"))
        page2_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        by_speed = query.runs_page(limit=50, sort="speedup")
        speed_page_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        best = query.best_confirmed(limit=20)
        best_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        brief = summarize_run(root / "runs" / "run_000000")
        brief_s = time.perf_counter() - t0
        return {
            "count": count, "indexed": rows, "create_s": round(create_s, 3),
            "sync_s": round(sync_s, 3), "history_s": round(history_s, 3), "tick_s": round(tick_s, 3),
            "page_s": round(page_s, 3), "page2_s": round(page2_s, 3), "speed_page_s": round(speed_page_s, 3),
            "best_s": round(best_s, 3), "brief_s": round(brief_s, 3),
            "history_rows": len(history), "page_rows": len(first.get("runs") or []),
            "page2_rows": len(second.get("runs") or []), "speed_page_rows": len(by_speed.get("runs") or []), "best_rows": len(best.get("best") or []),
            "brief_bytes": len(json.dumps(brief, ensure_ascii=False)),
            "db_bytes": db.stat().st_size if db.exists() else 0,
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.scale_probe")
    parser.add_argument("--counts", default="1000,10000,100000")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    counts = [int(item) for item in args.counts.split(",") if item.strip()]
    rows = []
    for count in counts:
        row = measure(count)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
