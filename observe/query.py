"""Read the observe index. Listing, filtering and arm summaries do not open candidate reports."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from experiments.common import suite_metrics
from experiments.contract_menu import judge
from observe import detail
from observe.budget import summarize
from observe.index import borrow, get_index
from taic.plan_ledger import nodes_from_trace
from taic.policy import feature_signature

PAGE_LIMIT = 100
_LIVE = {"last": None, "n": 0}
_SENT = 0
_GRID_CACHE: dict[tuple, list[dict]] = {}


def gpu_snapshot() -> dict:
    return detail.gpu_snapshot()


pid_alive = detail.pid_alive


def refresh() -> None:
    """Pick up new directories and edits to runs the index already knows. Used by reads that must match the disk."""
    get_index().sync_known()


def tick() -> None:
    """Idle frame: do not re-read finished runs."""
    get_index().tick()


def history() -> list[dict]:
    refresh()
    rows = []
    with borrow() as index:
        fetched = [item["row_json"] for item in index.conn.execute("SELECT row_json FROM runs WHERE status != 'missing'")]
    for text in fetched:
        if not text:
            continue
        rows.append(json.loads(text))
    rows.sort(key=lambda row: float(row.get("started_at") or 0), reverse=True)
    return rows


def budget_report(rows: list[dict] | None = None) -> dict:
    return summarize(history() if rows is None else rows)


def resolve_run(run_id: str) -> Path | None:
    return get_index().resolve(run_id)


def run_detail(run_id: str, *, stamp: str = "") -> dict | None:
    directory = resolve_run(run_id)
    if directory is None:
        return None
    current = _detail_stamp(run_id)
    if stamp and stamp == current:
        return {"same": True, "id": run_id, "detail_stamp": current}
    detail_doc = detail.run_detail(run_id)
    if detail_doc is None:
        return None
    config = detail._load(directory / "run_config.json")
    detail_doc["feature_grid"] = feature_grid(directory, detail_doc.get("result"))
    detail_doc["plan_outcomes"] = plan_outcomes(directory, config, detail_doc.get("state"), detail_doc.get("result"))
    detail_doc["detail_stamp"] = _detail_stamp(run_id)
    return detail_doc


def _detail_stamp(run_id: str) -> str:
    """Identity of the detail payload. A transcript append does not change it."""
    row = get_index().conn.execute(
        "SELECT fp_state, fp_result, fp_config, fp_confirm, fp_extra FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return ""
    return "|".join(str(row[key] or "") for key in ("fp_state", "fp_result", "fp_config", "fp_confirm", "fp_extra"))


def feature_grid(directory: Path, result: dict | None) -> list[dict]:
    """One cell per source signature. The cell keeps the fastest candidate seen during the search.

    Identical source is signed once. The grid is cached until ``state.json`` or ``result.json`` changes.
    """
    winner = (result or {}).get("confirm_winner_label")
    from experiments.common import conclusion_of_run

    published_ok = conclusion_of_run(directory, result).get("publication") == "confirmed"
    key = (
        str(directory.resolve()),
        _file_signature(directory / "state.json"),
        _file_signature(directory / "result.json"),
        str(winner or ""),
        published_ok,
    )
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    root = directory / "candidates"
    if not root.is_dir():
        _GRID_CACHE[key] = []
        return []
    best: dict[tuple, dict] = {}
    signed: dict[str, tuple] = {}
    order: list[tuple] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        source_path = path / "kernel.metal"
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError:
            continue
        report = detail._load(path / "report.json") or {}
        speed = report.get("speedup")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            speed = None
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()
        signature = signed.get(digest)
        if signature is None:
            signature = feature_signature(source)
            signed[digest] = signature
        label = path.name.split("_", 1)[1] if "_" in path.name else path.name
        current = best.get(signature)
        if current is None or (speed is not None and (current["speedup"] is None or speed > current["speedup"])):
            if current is None:
                order.append(signature)
            best[signature] = {
                "signature": list(signature),
                "label": label,
                "speedup": speed,
                "published": published_ok and label == winner,
            }
    rows = [best[signature] for signature in order if signature in best]
    for stale in [item for item in _GRID_CACHE if item[0] == key[0] and item != key]:
        _GRID_CACHE.pop(stale, None)
    _GRID_CACHE[key] = rows
    return rows


def plan_outcomes(directory: Path, config: dict | None, state: dict | None, result: dict | None) -> list[dict]:
    """Empty unless this run turned ``plan_ledger`` on. No priority and no confidence."""
    search = (config or {}).get("search") or {}
    if search.get("plan_ledger") is not True:
        return []
    trace = (state or {}).get("trace") or (result or {}).get("trace") or []
    history_lines = (state or {}).get("history") or (result or {}).get("history") or []
    if not isinstance(history_lines, list):
        history_lines = []
    lines = [line for line in history_lines if isinstance(line, str)]
    nodes = nodes_from_trace(trace if isinstance(trace, list) else [], lines)
    return [
        {
            "round": node.round_id,
            "label": node.label,
            "title": node.title,
            "outcome": node.outcome_kind,
            "speedup": node.speedup,
        }
        for node in nodes
    ]


def runs_page(*, experiment_id: str | None = None, task: str | None = None, dtype: str | None = None, model: str | None = None, status: str | None = None, limit: int = 50, offset: int = 0) -> dict:
    tick()
    limit = max(1, min(int(limit or 50), PAGE_LIMIT))
    offset = max(0, int(offset or 0))
    clauses = ["status != 'missing'"]
    args: list = []
    for column, value in (("experiment_id", experiment_id), ("task", task), ("dtype", dtype), ("model", model), ("status", status)):
        if value:
            clauses.append(f"{column} = ?")
            args.append(value)
    where = " AND ".join(clauses)
    with borrow() as index:
        total = index.conn.execute(f"SELECT COUNT(*) FROM runs WHERE {where}", args).fetchone()[0]
        rows = [
            dict(row)
            for row in index.conn.execute(
                f"SELECT run_id, experiment_id, arm_id, trial, task, dtype, model, status, publication, published_speedup, spent_usd, budget_usd, source, binding, started_at FROM runs WHERE {where} ORDER BY started_at DESC, run_id LIMIT ? OFFSET ?",
                [*args, limit, offset],
            )
        ]
    return {"total": total, "limit": limit, "offset": offset, "runs": rows}


def experiments() -> list[dict]:
    tick()
    rows = []
    with borrow() as index:
        items = list(index.conn.execute("SELECT * FROM experiments ORDER BY created_at DESC"))
        for item in items:
            arms = [
                dict(arm)
                for arm in index.conn.execute(
                    "SELECT arm_id, planned_n FROM arms WHERE experiment_id = ? ORDER BY arm_id",
                    (item["experiment_id"],),
                )
            ]
            rows.append({**dict(item), "arms": arms})
    return rows


def experiment_jobs() -> list[dict]:
    """Queues the stop button can see. Reads the state file of each known experiment, not the whole tree."""
    refresh()
    with borrow() as index:
        ids = [item["experiment_id"] for item in index.conn.execute("SELECT experiment_id FROM experiments")]
    jobs = []
    root = detail.runs_root()
    for experiment_id in ids:
        directory = root / experiment_id
        job = detail.experiment_job(directory)
        if job is not None:
            jobs.append(job)
    jobs.sort(key=lambda row: float(row.get("updated_at") or 0), reverse=True)
    return jobs


def compare(experiment_id: str) -> dict:
    refresh()
    with borrow() as index:
        exp = index.conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        exp = dict(exp) if exp is not None else None
        runs = [
            dict(row)
            for row in index.conn.execute(
                "SELECT * FROM runs WHERE experiment_id = ? AND status != 'missing'",
                (experiment_id,),
            )
        ]
        arms = [dict(arm) for arm in index.conn.execute("SELECT * FROM arms WHERE experiment_id = ? ORDER BY arm_id", (experiment_id,))]
    if exp is None and not runs:
        return {"experiment_id": experiment_id, "found": False, "verdict": "样本不足以判定", "arms": []}
    if exp is None:
        exp = {"kind": runs[0]["kind"], "config_hash": "", "frozen_json": "{}"}
    frozen = json.loads(exp["frozen_json"] or "{}")
    drift = [row["run_id"] for row in runs if row["binding"] == "drift"]
    inferred = [row["run_id"] for row in runs if row["source"] == "inferred"]
    usable = [row for row in runs if row["binding"] != "drift" and row["source"] != "inferred"]
    hashes = {row_hash(row, exp["config_hash"]) for row in usable}
    if len(hashes) > 1:
        usable = [row for row in usable if row_hash(row, exp["config_hash"]) == exp["config_hash"]]
    by_arm: dict[str, list[dict]] = {}
    for row in usable:
        by_arm.setdefault(row["arm_id"] or "", []).append(row)
    arm_rows = []
    judge_rows = []
    for arm in arms:
        members = by_arm.get(arm["arm_id"], [])
        speeds = [row["published_speedup"] for row in members if isinstance(row["published_speedup"], (int, float))]
        metrics = suite_metrics([
            {"published_speedup": row["published_speedup"], "spent_usd": row["spent_usd"]}
            for row in members
        ])
        spent = metrics["spent_usd"]
        gain = _gain_per_usd(members, spent)
        arm_rows.append({
            "arm_id": arm["arm_id"],
            "n": len(members),
            "published_n": len(speeds),
            "median": _median(speeds),
            "min": min(speeds) if speeds else None,
            "max": max(speeds) if speeds else None,
            "confirmed_n": sum(1 for row in members if row["publication"] == "confirmed"),
            "fast": metrics["fast"],
            "spent_usd": spent,
            "gain_per_usd": gain,
        })
        for row in members:
            judge_rows.append({"slug": arm["arm_id"], "published_speedup": row["published_speedup"], "skipped": None})
    if inferred and not usable:
        verdict, reason, winner = "不下判定", "这些运行是从旧目录推断的", None
    elif exp["kind"] == "contract_menu":
        judged = judge(judge_rows, [row["arm_id"] for row in arm_rows])
        verdict, reason, winner = judged["verdict"], judged["reason"], judged["winner"]
    elif not any(row["published_n"] >= 2 for row in arm_rows):
        verdict, reason, winner = "样本不足以判定", "发布数字少于 2 个", None
    else:
        verdict, reason, winner = "样本不足以判定", "只给出各臂的数字，不从最高的一次定胜负", None
    return {
        "experiment_id": experiment_id,
        "found": True,
        "kind": exp["kind"],
        "config_hash": exp["config_hash"],
        "frozen": frozen,
        "verdict": verdict,
        "reason": reason,
        "winner": winner,
        "arms": arm_rows,
        "drift": drift,
        "inferred": inferred,
    }


def row_hash(row: dict, fallback: str) -> str:
    return fallback


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _gain_per_usd(members: list[dict], spent: float) -> float | None:
    if spent <= 0:
        return None
    pieces = []
    for row in members:
        published = row.get("published_speedup")
        naive = row.get("naive_speedup")
        if isinstance(published, (int, float)) and not isinstance(published, bool) and isinstance(naive, (int, float)) and not isinstance(naive, bool) and naive:
            pieces.append(published - naive)
    if not pieces:
        return None
    return sum(pieces) / spent


def overview() -> dict:
    from tasks import list_tasks

    rows = history()
    live = [row for row in rows if row.get("status") == "running"]
    progress = None
    if live:
        directory = resolve_run(live[0]["id"])
        if directory is not None:
            progress = detail._age(detail._load(directory / "eval_progress.json"))
    evals = detail.live_evals()
    beat = detail._heartbeat(progress) or (detail._heartbeat(evals[0]) if evals else None)
    return {
        "gpu": detail.gpu_snapshot(),
        "live_runs": live,
        "live_evals": evals,
        "heartbeat": beat,
        "tasks": list(list_tasks(include_phase2=True)),
        "experiments": experiment_jobs(),
        "now": time.time(),
    }


def frame(run_id: str | None) -> dict:
    hist = history()
    chosen = run_id or ""
    if chosen and resolve_run(chosen) is None:
        chosen = ""
    if not chosen:
        live = next((row for row in hist if row.get("status") == "running"), None)
        chosen = live["id"] if live else (hist[0]["id"] if hist else "")
    return {
        "overview": overview(),
        "history": hist,
        "budget": budget_report(hist),
        "selected": chosen,
        "run": run_detail(chosen) if chosen else None,
    }


def ledger_report() -> dict:
    refresh()
    with borrow() as index:
        by_operator = [
            {"operator": row["operator"], "runs": row["runs"], "spent_usd": round(row["spent"], 6), "budget_usd": round(row["budget"], 6)}
            for row in index.conn.execute(
                "SELECT operator, COUNT(*) AS runs, COALESCE(SUM(spent_max_usd), 0) AS spent, COALESCE(SUM(budget_usd), 0) AS budget FROM ledger GROUP BY operator ORDER BY spent DESC, operator"
            )
        ]
        by_month = []
        for row in index.conn.execute(
            "SELECT month, COUNT(*) AS runs, COALESCE(SUM(spent_max_usd), 0) AS spent FROM ledger GROUP BY month ORDER BY month"
        ):
            operators = [
                {"operator": item["operator"], "runs": item["runs"], "spent_usd": round(item["spent"], 6)}
                for item in index.conn.execute(
                    "SELECT operator, COUNT(*) AS runs, COALESCE(SUM(spent_max_usd), 0) AS spent FROM ledger WHERE month = ? GROUP BY operator ORDER BY spent DESC, operator",
                    (row["month"],),
                )
            ]
            by_month.append({"month": row["month"], "runs": row["runs"], "spent_usd": round(row["spent"], 6), "operators": operators})
        counted = index.conn.execute("SELECT COUNT(*), COALESCE(SUM(budget_usd), 0) FROM ledger").fetchone()
        spent = index.total_usd()
        carry = _carry_total(index)
    return {
        "runs": counted[0],
        "spent_usd": spent,
        "budget_usd": round(counted[1], 6),
        "by_operator": by_operator,
        "by_month": by_month,
        "carryover_usd": carry,
    }


def _carry_total(index) -> float:
    row = index.conn.execute("SELECT total_usd FROM ledger_carryover LIMIT 1").fetchone()
    return 0.0 if row is None else float(row["total_usd"] or 0)


def live_files() -> list[Path]:
    from observe.publish import gpu_dir, observe_dir

    paths = [gpu_dir(), observe_dir()]
    directory = observe_dir()
    if directory.is_dir():
        paths.extend(sorted(directory.glob("tcenv-*.json"))[:5])
    return paths


def live_revision() -> int:
    signature = tuple(_file_signature(path) for path in live_files())
    if signature != _LIVE["last"]:
        _LIVE["last"] = signature
        _LIVE["n"] += 1
    return int(_LIVE["n"])


def _file_signature(path: Path) -> tuple:
    try:
        if path.is_dir():
            stat = path.stat()
            return (str(path), stat.st_mtime_ns, len(list(path.iterdir())))
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), 0, 0)


def stream_revision() -> int:
    return poll_stream()["revision"]


def poll_stream() -> dict:
    """One idle frame. The detail revision and the turn revision move independently."""
    global _SENT
    index = get_index()
    rev, changed = index.tick()
    _SENT = max(_SENT, rev, live_revision())
    return {
        "index_revision": _SENT,
        "turns_revision": index.turns_revision(),
        "revision": _SENT,
        "runs": changed,
    }


def changed_runs() -> list[str]:
    return list(get_index().changed)
