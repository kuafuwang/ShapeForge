"""Read-only facts for the next reader of a run. No model call and no GPU kernel.

With no directory, ``run_brief`` prints a short orientation. With a run directory it prints
one compact JSON object built only from files on disk.
"""

# 失败时先看这份摘要，而不是整份 transcript.jsonl、state.json 或看板页面。
# 只读已知文件。消息正文、内核源码、密钥都不进入输出。缺文件就是 null，不补一个数。

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SECRET = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")

CAP = 8
_TEXT = 80
_GPU_TAIL = 262_144

ORIENTATION = """\
taic brief
Read-only. No model request, no GPU kernel, no lock taken.

Where a search starts
  uv run python -m taic.cli run TASK
  taic/cli.py cmd_run builds the config and calls Orchestrator.run in taic/orchestrator.py.
  Layers: taic.toml, then a preset (flag --preset, else TAIC_PRESET, else paper = taic.paper.toml).
  --config uses that file alone. --budget, when given, replaces budget_usd.
  uv run python -m taic.cli resume RUN_DIR continues a directory and does not resend billed requests.
  --target names a profile under targets/. The default is apple-m4-max. A new run is stored in runs/<target>/.
  uv run python -m taic.cli brief --kernels lists confirmed kernels under kernels/<target>/.
  uv run python -m taic.cli migrate-runs moves flat run directories into runs/apple-m4-max/.

Two committees
  Every search is exactly two committees, named first and second. Each has one member (provider and model).
  Another count or other names is refused before any model call. A longer member list is not reduced to the first.
  If the config has no committees, both copy the [llm] model. --model assigns that one model to both.
  The same model on both seats still starts a new session. There is no handoff sentence in TASK.md.
  The initial draft and round 1 bind first. From round 1, round r binds (r - 1) % 2. The switch is only at a round boundary.
  Each round is one new hmz session. The task file carries the kernel, the scorecard, and the last failure. It does not carry the previous chat.
  A session that does not leave a new kernel is agent_failed. The next session goes to the other committee.
  Two agent_failed sessions in a row stop the search. An incorrect kernel, or a file identical to the seed, clears that count.
  A session that ends without a token bill stops the whole search (cost_incomplete).
  Both share one ledger, one transcript.jsonl, and one budget. Keys and endpoints stay out of the run directory.

Evaluation
  tcenv order: screen, compile, ABI, guards and inputs, numeric check, benchmark. verify() runs first and stays unsupported.
  Evaluations inside one search take an asyncio lock in Orchestrator, so those measurements do not overlap.
  The machine lock is a flock in tcenv/gpulock.py. Timing holds it exclusively; compile and correctness hold it shared.
  Default path: ~/.cache/ai_lowerling/gpu.lock (TCENV_GPU_LOCK_FILE overrides it).
  A Metal kernel cannot be preempted and also drives the display. max_hangs is 1, so the first timeout stops the run.
  The next seed must be safe, correct, and faster by min_improvement and by noise_z standard errors.
  A within_noise candidate stays in the trace. The seed stays the previous kernel.

Publication gate
  experiments.common.conclusion_of_run decides the published speedup.
  The number is the remeasure, and only when that confirmation is bound to the same kernel and the timing is quiet.
  best_speedup stays on search_best. An empty gate leaves published null.

GPU lock cache
  Events append to ~/.cache/ai_lowerling/observe/gpu_events.jsonl.
  AI_LOWERLING_OBSERVE_DIR replaces the cache root. This command prints the last events only when that file is present.

Open next
  taic/cli.py
  taic/committees.py
  taic/orchestrator.py: _guard_shape, _one_session, _rounds, _eval, _confirm
  taic/kernel_agent.py: render_contract, run_kernel_agent
  experiments/common.py: conclusion_of_run
  tcenv/gpulock.py
  tcenv/evaluate.py
  On a failure, run: uv run python -m taic.cli brief RUN_DIR
  Read that JSON before transcript.jsonl, state.json, or the HTML dashboard.
"""


def run_brief(run_dir: str | None) -> int:
    """Print the orientation, or one JSON object. ``2`` when ``run_dir`` is not a directory."""
    if not run_dir:
        print(ORIENTATION, end="")
        return 0
    path = Path(run_dir)
    if not path.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 2
    print(json.dumps(summarize_run(path), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def summarize_run(run_dir: Path) -> dict:
    """One JSON-ready object. Missing files are null. Message text and kernel source are left out."""
    run = Path(run_dir)
    state = _read_obj(run / "state.json")
    result, result_err = _load_result(run)
    docs = [d for d in (state, result) if isinstance(d, dict)]
    published, publication = _published(run, result, result_err)
    chat, retry, transcript_stage = _transcript_counts(run / "transcript.jsonl")
    doc = {
        "task": _first(docs, "task"),
        "dtype": _first(docs, "dtype"),
        "size": _first(docs, "size"),
        "committee": _first(docs, "committee"),
        "model": _first(docs, "model"),
        "phase": _first(docs, "phase"),
        "stop_reason": _first(docs, "stop_reason"),
        "round": _round_of(state, result),
        "spent_usd": _first(docs, "spent_usd"),
        "budget_usd": _first(docs, "budget_usd"),
        "published": published,
        "publication": publication,
        "search_best": _first(docs, "best_speedup"),
        "cands": _cands(run, state, result),
        "counts": {"chat": chat, "retry": retry, "truncation_retries": _truncation_retries(run / "transcript.jsonl"), "eval_stage": _eval_stage(run, transcript_stage)},
        "paths": _paths(run),
    }
    gpu = _gpu_events()
    if gpu is not None:
        doc["gpu"] = gpu
    return doc


def _first(docs: list[dict], key: str):
    for doc in docs:
        if key in doc:
            return doc[key]
    return None


def _round_of(state: dict | None, result: dict | None):
    if isinstance(state, dict) and "round" in state:
        return state["round"]
    if isinstance(result, dict) and "round" in result:
        return result["round"]
    if isinstance(result, dict) and "rounds" in result:
        return result["rounds"]
    if isinstance(state, dict) and "rounds" in state:
        return state["rounds"]
    return None


def _load_result(run: Path) -> tuple[dict | None, str | None]:
    path = run / "result.json"
    if not path.is_file():
        return None, None
    from experiments.common import read_result
    from taic.artifacts import NeedsInspection

    try:
        doc = read_result(run)
    except NeedsInspection:
        return None, "unreadable"
    if not isinstance(doc, dict):
        return None, "unreadable"
    return doc, None


def _published(run: Path, result: dict | None, result_err: str | None):
    if result_err:
        return None, result_err
    if result is None:
        return None, None
    from experiments.common import conclusion_of_run

    got = conclusion_of_run(run, result)
    if not isinstance(got, dict):
        return None, "unreadable"
    speed = got.get("speedup")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        speed = None
    return speed, got.get("publication")


def _cands(run: Path, state: dict | None, result: dict | None) -> list[dict]:
    trace = _trace(state, result)
    if trace is not None:
        rows = [_cand_from_trace(item) for item in trace if isinstance(item, dict)]
        return rows[-CAP:]
    root = run / "candidates"
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if _cand_dir(p)]
    dirs.sort(key=lambda p: p.name)
    return [_cand_from_dir(p) for p in dirs[-CAP:]]


def _trace(state: dict | None, result: dict | None) -> list | None:
    for doc in (state, result):
        if isinstance(doc, dict) and isinstance(doc.get("trace"), list):
            return doc["trace"]
    return None


def _cand_dir(path: Path) -> bool:
    name = path.name
    return path.is_dir() and len(name) > 4 and name[:3].isdigit() and name[3] == "_"


def _cand_from_trace(item: dict) -> dict:
    recheck = item.get("differential_recheck")
    if isinstance(recheck, dict):
        recheck = recheck.get("class")
    return {
        "label": _text(item.get("label")),
        "status": _text(item.get("status")),
        "speedup": _number(item.get("speedup")),
        "within_noise": item.get("within_noise") if isinstance(item.get("within_noise"), bool) else None,
        "differential_recheck": _text(recheck),
    }


def _cand_from_dir(path: Path) -> dict:
    report = _read_obj(path / "report.json") or {}
    recheck = _read_obj(path / "differential_recheck.json") or {}
    return {
        "label": _text(path.name[4:]),
        "status": _text(report.get("status")),
        "speedup": _number(report.get("speedup")),
        "within_noise": None,
        "differential_recheck": _text(recheck.get("class")),
    }


def _transcript_counts(path: Path) -> tuple[int | None, int | None, str | None]:
    if not path.is_file():
        return None, None, None
    chat = retry = 0
    stage = None
    for ev in _jsonl(path):
        kind = ev.get("event")
        if kind == "chat":
            chat += 1
        elif kind == "retry":
            retry += 1
        found = ev.get("stage")
        if isinstance(found, str):
            stage = _text(found)
    return chat, retry, stage


def _truncation_retries(path: Path) -> int | None:
    """Replies the transcript marked as cut off by max_tokens. Missing file is null."""
    if not path.is_file():
        return None
    n = 0
    for ev in _jsonl(path):
        if ev.get("event") == "warning" and ev.get("message") == "reply truncated by max_tokens":
            n += 1
        elif ev.get("finish_reason") == "length":
            n += 1
    return n


def _eval_stage(run: Path, transcript_stage: str | None):
    path = run / "eval_progress.json"
    if path.is_file():
        doc = _read_obj(path)
        if isinstance(doc, dict):
            return _text(doc.get("stage"))
        return None
    return transcript_stage


def _gpu_events() -> list[dict] | None:
    from tcenv.events import gpu_events_path

    path = gpu_events_path()
    if not path.is_file():
        return None
    rows = []
    for ev in _jsonl(path, tail=True):
        what = ev.get("what")
        rows.append({
            "t": _number(ev.get("t")),
            "pid": ev.get("pid") if isinstance(ev.get("pid"), int) and not isinstance(ev.get("pid"), bool) else None,
            "kind": _text(ev.get("kind")),
            "mode": _text(ev.get("mode")),
            "what": _text(what),
        })
    return rows[-CAP:]


def _paths(run: Path) -> dict:
    names = {
        "result": run / "result.json",
        "confirm": run / "confirm_report.json",
        "metal": run / "best.metal",
        "transcript": run / "transcript.jsonl",
    }
    return {key: (str(path.resolve()) if path.is_file() else None) for key, path in names.items()}


def _read_obj(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _jsonl(path: Path, *, tail: bool = False) -> list[dict]:
    try:
        if tail:
            text = _tail_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            rows.append(ev)
    return rows


def _tail_text(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, size - _GPU_TAIL))
        data = handle.read().decode("utf-8", errors="replace")
    if size > _GPU_TAIL:
        cut = data.find("\n")
        if cut >= 0:
            data = data[cut + 1 :]
    return data


def _text(value) -> str | None:
    if not isinstance(value, str) or "\n" in value or len(value) > _TEXT:
        return None
    return _SECRET.sub("<redacted>", value)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value
