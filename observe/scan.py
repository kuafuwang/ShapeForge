"""Turn run directories and the GPU cache into the dashboard's current frame."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from observe.publish import gpu_dir, gpu_events_path, observe_dir

STALE_S = 15.0
EVENT_TAIL = 40

REPO_ROOT = Path(__file__).resolve().parent.parent


def runs_root() -> Path:
    env = os.environ.get("AI_LOWERLING_RUNS_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "runs"


def pid_alive(pid: object) -> bool:
    try:
        number = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if number <= 0:
        return False
    try:
        os.kill(number, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _load(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _age(doc: dict | None) -> dict | None:
    if not doc:
        return None
    out = dict(doc)
    at = out.get("heartbeat_at")
    if isinstance(at, (int, float)):
        out["heartbeat_age_s"] = round(max(0.0, time.time() - float(at)), 1)
    return out


def resolve_run(run_id: str) -> Path | None:
    """A directory under ``runs/`` that holds a search snapshot. Rejects anything that climbs out."""
    if not run_id or "\x00" in run_id:
        return None
    rel = Path(run_id)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    root = runs_root().resolve()
    try:
        path = (root / rel).resolve()
    except OSError:
        return None
    if path != root and root not in path.parents:
        return None
    if not path.is_dir():
        return None
    if not (path / "state.json").exists() and not (path / "result.json").exists():
        return None
    return path


def _rel(path: Path) -> str:
    return path.resolve().relative_to(runs_root().resolve()).as_posix()


def liveness(state: dict | None, result: dict | None) -> str:
    if result or (state and state.get("phase") == "done"):
        return "done"
    if not state:
        return "unknown"
    updated = float(state.get("updated_at") or 0)
    if time.time() - updated > STALE_S and not pid_alive(state.get("pid")):
        return "interrupted"
    return "running"


def _search_dirs() -> list[tuple[Path, dict | None, dict | None]]:
    root = runs_root()
    if not root.is_dir():
        return []
    dirs: set[Path] = set()
    for name in ("state.json", "result.json"):
        for path in root.rglob(name):
            dirs.add(path.parent)
    found: list[tuple[Path, dict | None, dict | None]] = []
    for directory in dirs:
        state = _load(directory / "state.json")
        result = _load(directory / "result.json")
        if state and state.get("script") and not result:
            continue
        if result or (state and state.get("task") and not state.get("script")):
            found.append((directory, state, result))
    return found


def _speedup(state: dict | None, result: dict | None) -> float | None:
    if result:
        value = result.get("confirmed_speedup")
        if value is None:
            value = result.get("best_speedup")
        return value
    if state:
        return state.get("best_speedup")
    return None


def _when(directory: Path, state: dict | None) -> float:
    if state and isinstance(state.get("started_at"), (int, float)):
        return float(state["started_at"])
    try:
        return (directory / "state.json").stat().st_mtime if (directory / "state.json").exists() else directory.stat().st_mtime
    except OSError:
        return 0.0


def history() -> list[dict]:
    rows = []
    for directory, state, result in _search_dirs():
        status = liveness(state, result)
        task = (result or state or {}).get("task") or ""
        dtype = (result or state or {}).get("dtype") or ""
        phase = (state or {}).get("phase") or ""
        stop = (result or {}).get("stop_reason") or (state or {}).get("stop_reason") or ""
        spent = (result or {}).get("spent_usd")
        if spent is None and state:
            spent = state.get("spent_usd")
        rows.append({
            "id": _rel(directory),
            "started_at": _when(directory, state),
            "task": task,
            "dtype": dtype,
            "phase": phase,
            "stop_reason": stop,
            "speedup": _speedup(state, result),
            "spent_usd": spent,
            "budget_usd": (result or state or {}).get("budget_usd"),
            "round": (state or {}).get("round"),
            "max_rounds": (state or {}).get("max_rounds"),
            "inflight_label": (state or {}).get("inflight_label"),
            "status": status,
        })
    rows.sort(key=lambda row: row["started_at"], reverse=True)
    return rows


def _tail_events() -> list[dict]:
    path = gpu_events_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-EVENT_TAIL:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def gpu_snapshot() -> dict:
    holders: list[dict] = []
    waiters: list[dict] = []
    directory = gpu_dir()
    if directory.is_dir():
        for path in directory.glob("*.json"):
            doc = _load(path)
            if not doc or not pid_alive(doc.get("pid")):
                continue
            if path.name.endswith(".wait.json") or doc.get("role") == "waiter":
                waiters.append(doc)
            else:
                holders.append(doc)
    return {"holders": holders, "waiters": waiters, "events": _tail_events()}


def live_evals() -> list[dict]:
    directory = observe_dir()
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("tcenv-*.json")):
        doc = _age(_load(path))
        if not doc:
            continue
        try:
            pid = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        doc["pid"] = pid
        if doc.get("finished") or not pid_alive(pid):
            continue
        out.append(doc)
    return out


def experiment_jobs() -> list[dict]:
    root = runs_root()
    if not root.is_dir():
        return []
    rows = []
    for path in root.rglob("state.json"):
        state = _load(path)
        if not state or not state.get("script"):
            continue
        phase = state.get("phase") or "run"
        if phase in ("done", "aborted"):
            status = phase
        else:
            status = "interrupted" if time.time() - float(state.get("updated_at") or 0) > STALE_S and not pid_alive(state.get("pid")) else "running"
        rows.append({
            "id": _rel(path.parent),
            "script": state.get("script"),
            "phase": phase,
            "status": status,
            "current": state.get("current"),
            "queue": state.get("queue") or [],
            "done": state.get("done") or [],
            "updated_at": state.get("updated_at"),
        })
    rows.sort(key=lambda row: float(row.get("updated_at") or 0), reverse=True)
    return rows


def _cases(report: dict) -> list[dict]:
    cases = (report.get("correctness") or {}).get("cases") or []
    out = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        out.append({"case": case.get("case"), "ok": case.get("ok"), "message": (case.get("message") or "")[:160]})
    return out


def _excerpt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")[:400]
    except OSError:
        return ""


def _candidates(directory: Path) -> list[dict]:
    root = directory / "candidates"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        report = _load(path / "report.json") or {}
        prefix, _, label = path.name.partition("_")
        try:
            idx = int(prefix)
        except ValueError:
            idx = None
            label = path.name
        roof = report.get("roofline") or {}
        bench = report.get("benchmark") or {}
        rows.append({
            "idx": idx,
            "label": label,
            "status": report.get("status"),
            "stage": report.get("stage"),
            "mode": report.get("mode"),
            "speedup": report.get("speedup"),
            "candidate_us": report.get("candidate_us"),
            "message": (report.get("message") or "")[:300],
            "cases": _cases(report),
            "roofline": {key: roof.get(key) for key in ("achieved_tflops", "achieved_gbps", "compute_fraction", "bandwidth_fraction", "bound")},
            "benchmark": {key: bench.get(key) for key in ("noisy", "attempts", "noise_cv")},
            "excerpt": _excerpt(path / "kernel.metal"),
        })
    return rows


def _heartbeat(progress: dict | None) -> dict | None:
    if not progress or not progress.get("heartbeat"):
        return None
    return {"letter": progress.get("heartbeat"), "age_s": progress.get("heartbeat_age_s"), "stage": progress.get("stage")}


def overview() -> dict:
    from tasks import list_tasks

    hist = history()
    live = [row for row in hist if row["status"] == "running"]
    progress = None
    if live:
        directory = resolve_run(live[0]["id"])
        if directory is not None:
            progress = _age(_load(directory / "eval_progress.json"))
    evals = live_evals()
    beat = _heartbeat(progress) or (_heartbeat(evals[0]) if evals else None)
    return {
        "gpu": gpu_snapshot(),
        "live_runs": live,
        "live_evals": evals,
        "heartbeat": beat,
        "tasks": list(list_tasks(include_phase2=True)),
        "experiments": experiment_jobs(),
        "now": time.time(),
    }


def run_detail(run_id: str) -> dict | None:
    directory = resolve_run(run_id)
    if directory is None:
        return None
    state = _load(directory / "state.json")
    result = _load(directory / "result.json")
    trace = (state or {}).get("trace") or (result or {}).get("trace") or []
    confirmations = (state or {}).get("confirmations") or (result or {}).get("confirmations") or []
    return {
        "id": _rel(directory),
        "status": liveness(state, result),
        "state": state,
        "result": result,
        "trace": trace,
        "confirmations": confirmations,
        "candidates": _candidates(directory),
        "eval_progress": _age(_load(directory / "eval_progress.json")),
        "llm": _load(directory / "llm.json"),
    }


def frame(run_id: str | None) -> dict:
    hist = history()
    chosen = run_id or ""
    if chosen and resolve_run(chosen) is None:
        chosen = ""
    if not chosen:
        live = next((row for row in hist if row["status"] == "running"), None)
        chosen = live["id"] if live else (hist[0]["id"] if hist else "")
    return {
        "overview": overview(),
        "history": hist,
        "selected": chosen,
        "run": run_detail(chosen) if chosen else None,
    }
