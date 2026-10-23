"""Continue a run from what its directory holds: after a crash, a closed window, or a stop for budget.

``python -m taic.cli resume <run_dir> [--add-budget USD | --budget USD] [--dry-run]`` rebuilds the search state from the files and goes on with the
rounds that were not finished. Nothing here spends money or touches the GPU except ``Orchestrator.run()`` after ``restore()``.

What the directory holds, and what is rebuilt from it
-----------------------------------------------------
* cost ledger        every model request is in ``transcript.jsonl`` with its usage, its cost and the running total, so the money spent so far is exact
                     (``ledger.json`` exists only when a run ended, ``state.json`` only knows the total as of its last write).
* candidates         ``candidates/NNN_<label>/{kernel.metal,report.json}``; the best kernel, the evaluation cache and the trace are rebuilt from them
                     (``state.json`` keeps the trace too; the selection is replayed and compared with it).
* finished rounds    ``state.json`` keeps the history lines and the round number. A round that already has a candidate is finished.
                     A round with no candidate starts again as a new session. The previous conversation is not continued.
* settings           ``run_config.json`` must say ``loop_protocol: flame_chase_1``. A directory without that mark is refused.
                     The system prompt is read back from ``system_prompt.txt`` when that file is there.

What cannot be recovered: a request that was in flight when the process died (the provider may have billed it, the ledger never saw it),
Instruments traces that failed (not listed), the ledger's per-request wall-clock times of older sessions (``t`` is ``null`` for them).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import control
from .artifacts import LEDGER_SCHEMA_VERSION, NeedsInspection, RESULT_SCHEMA_VERSION, STATE_SCHEMA_VERSION, read_versioned
from .policy import candidate_eligible, comparison_speedup, relation_to_best

STOP_BUDGET = "budget_exhausted"
SUGGESTED_AMOUNTS = (3.0, 5.0, 15.0)
LABEL_RE = re.compile(r"^(?P<base>r(?P<round>\d+)\.p(?P<plan>\d+))(?:\.fix(?P<fix>\d+))?(?:#t(?P<turn>\d+))?$")
CAND_DIR_RE = re.compile(r"^(?P<idx>\d+)_(?P<label>.+)$")
PROFILE_RE = re.compile(r"^r(?P<round>\d+)_(?P<label>.+)\.(?P<ext>json|txt)$")


class ResumeError(RuntimeError):
    """The run directory cannot be continued (the message says why and what to do)."""


# ---------------------------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------------------------


@dataclass
class BranchResume:
    """Where a part-way branch (one plan of a round) stood when the process stopped."""

    attempt: int  # repair attempt in progress (0 = the first answer)
    messages: list  # the conversation to go on from
    turn: int = 0  # tool-loop turn of the last assistant message (with ``pending``) or of the next request (without)
    pending: list = field(default_factory=list)  # tool calls of the last assistant message that have no result yet
    reply_text: str | None = None  # a final answer that had been received but not evaluated
    quick_used: int = 0  # quick_check calls this request already used
    note: str = ""


@dataclass
class RoundPartial:
    """The last round that was entered; it may be finished, part-way or not yet started."""

    rnd: int
    seed_label: str
    plans: list | None  # None: the planner has to be asked
    done: dict = field(default_factory=dict)  # branch index -> label of the candidate it returned (None: it ended without a valid kernel)
    resume: dict = field(default_factory=dict)  # branch index -> BranchResume
    plan_source: str = ""  # how the plans were found, for the report


@dataclass
class Snapshot:
    run_dir: str
    task: str
    dtype: str
    size: str
    model: str
    requests: list
    candidates: list  # dicts: idx, label, source, report (EvalReport), plan
    trace: list
    history: list
    best_label: str | None
    rounds_done: int
    next_round: int
    stagnant: int
    partial: RoundPartial | None
    hangs: int = 0
    evaluations: int = 0
    crashes_in_a_row: int = 0
    profiles: list = field(default_factory=list)
    profile_bodies: dict = field(default_factory=dict)
    system: str | None = None
    elapsed_s: float = 0.0
    started_at: float | None = None
    initial_budget_usd: float | None = None
    resumes: int = 0
    agent_failures: int = 0
    notes: list = field(default_factory=list)  # what was rebuilt, what is approximate
    checks: dict = field(default_factory=dict)  # consistency checks between the sources

    @property
    def spent_usd(self) -> float:
        return float(self.requests[-1]["total_usd"]) if self.requests else 0.0


# ---------------------------------------------------------------------------------------------
# reading the directory (light: no tcenv, no openai)
# ---------------------------------------------------------------------------------------------


def _first(*values):
    return next((v for v in values if v is not None), None)


def _doc(path: Path) -> dict | None:
    d = control.read_json(path)
    return d if isinstance(d, dict) else None


def read_transcript(run_dir: Path) -> tuple[list[dict], int]:
    """(events, number of damaged lines). A damaged last line is what a kill in the middle of a write leaves behind."""
    path = Path(run_dir) / "transcript.jsonl"
    events: list[dict] = []
    bad = 0
    if not path.exists():
        return events, bad
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(e, dict):
                events.append(e)
    return events, bad


def has_best(state: dict | None, result: dict | None) -> bool:
    for doc in (state, result):
        if doc and (doc.get("best_label") or any(t.get("improved") for t in doc.get("trace") or [] if isinstance(t, dict))):
            return True
    return False


def _same_pid(a, b) -> bool:
    try:
        return int(a) > 0 and int(a) == int(b)
    except (TypeError, ValueError):
        return False


def _launching_us(holder: dict) -> bool:
    """The lock was written by the process that spawned this resume (the dashboard's ``uv run``), or it names this process.
    That is our own start, not another resume."""
    if not isinstance(holder, dict):
        return False
    return _same_pid(holder.get("pid"), os.getpid()) or _same_pid(holder.get("pid"), os.getppid())


def _count_stop(stop: str) -> bool:
    """A stop that is a count of rounds: the hard round cap, or patience (rounds without a new best)."""
    return stop == "max_rounds" or stop.startswith("converged (")


def inspect_run(run_dir: Path, *, pid_check=control.pid_alive, allow_count_stops: bool = False) -> dict:
    """Whether and how ``run_dir`` can be continued, from its small files only (the dashboard calls this for every click).

    ``kind``: ``running`` (a live process owns it; ``can_topup_live`` says whether it polls the control file), ``resuming`` (a resume holds the lock),
    ``interrupted`` (no result, no live process), ``budget_exhausted`` (finished, stopped for money), ``finished`` (any other stop), ``unknown``."""
    run_dir = Path(run_dir)
    try:
        state, result = read_versioned(run_dir / "state.json", STATE_SCHEMA_VERSION), read_versioned(run_dir / "result.json", RESULT_SCHEMA_VERSION)
    except NeedsInspection as e:
        info: dict = {
            "run_dir": str(run_dir), "kind": "unknown", "stop_reason": "", "phase": "", "resumable": False, "reason": str(e), "needs_budget": False,
            "can_topup_live": False, "live_reason": "", "spent_usd": None, "budget_usd": None, "initial_budget_usd": None,
            "pid": None, "round": None, "max_rounds": None, "best_speedup": None, "control_supported": False, "history": control.budget_history(run_dir, None),
            "suggested_amounts": list(SUGGESTED_AMOUNTS), "task": None, "dtype": None,
        }
        return info
    ledger = (state or {}).get("ledger") or {}
    spent = _first((result or {}).get("spent_usd"), (state or {}).get("spent_usd"), ledger.get("total_usd"))
    budget = _first((result or {}).get("budget_usd"), (state or {}).get("budget_usd"), ledger.get("budget_usd"))
    stop = (result or {}).get("stop_reason") or (state or {}).get("stop_reason") or ""
    phase = (state or {}).get("phase") or ""
    pid = (state or {}).get("pid")
    holder = control.lock_holder(run_dir)
    hist = control.budget_history(run_dir, budget)
    info: dict = {
        "run_dir": str(run_dir), "kind": "unknown", "stop_reason": stop, "phase": phase, "resumable": False, "reason": "", "needs_budget": False,
        "can_topup_live": False, "live_reason": "", "spent_usd": spent, "budget_usd": budget, "initial_budget_usd": hist["initial_budget_usd"],
        "pid": pid, "round": (state or {}).get("round"), "max_rounds": (state or {}).get("max_rounds"),
        "best_speedup": _first((result or {}).get("confirmed_speedup"), (result or {}).get("best_speedup"), (state or {}).get("best_speedup")),
        "control_supported": bool((state or {}).get("control_supported")), "history": hist, "suggested_amounts": list(SUGGESTED_AMOUNTS),
        "task": (result or state or {}).get("task"), "dtype": (result or state or {}).get("dtype"),
    }
    try:
        info["needs_budget"] = (
            bool((result or {}).get("session_budget_stop"))
            or (spent is not None and budget is not None and float(spent) >= float(budget) - 1e-9)
        )
    except (TypeError, ValueError):
        pass
    if holder and not _same_pid(holder.get("pid"), pid) and not _launching_us(holder):
        info.update(kind="resuming", reason=f"already being resumed by pid {holder.get('pid')}")
        return info
    if state is None and result is None:
        info.update(reason="no state.json or result.json in this directory: not a search run")
        return info
    finished = result is not None or phase == "done"
    if not finished:
        if pid_check(pid):
            info.update(kind="running", reason=f"still running (pid {pid})")
            if info["control_supported"]:
                info["can_topup_live"] = True
            else:
                info["live_reason"] = "started by an older version that does not read the control file: wait for it to stop, then resume it"
            return info
        info["kind"] = "interrupted"
        if not has_best(state, result):
            info["reason"] = "no valid kernel was found before the process stopped: there is nothing to continue from, start a new run"
            return info
        if _refuse_old_protocol(info, run_dir):
            return info
        info["resumable"] = True
        return info
    if stop == STOP_BUDGET:
        info["kind"] = "budget_exhausted"
        if not has_best(state, result):
            info["reason"] = "stopped for budget without a valid kernel: start a new run"
            return info
        if _refuse_old_protocol(info, run_dir):
            return info
        info["resumable"] = True
        return info
    if allow_count_stops and _count_stop(stop):
        # 仪表盘「重置」才打开这条路：轮数或耐心次数用完的运行可以续，显卡挂死和模型报错仍然不行。
        info["kind"] = "count_limited"
        if not has_best(state, result):
            info["reason"] = "stopped on a round limit without a valid kernel: start a new run"
            return info
        if _refuse_old_protocol(info, run_dir):
            return info
        info["resumable"] = True
        return info
    info["kind"] = "finished"
    why = stop or "unknown reason"
    info["reason"] = f"stopped for '{why}', not for budget: only runs that ran out of budget (or were interrupted) can be continued"
    return info


# ---------------------------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------------------------


def infer_settings(run_dir: Path, events: list[dict] | None = None) -> dict:
    """The settings of the run: ``{"search": {...}, "llm": {...}, "provenance": {key: how}, "system": str | None}``.
    ``run_config.json`` is exact; for older runs the values are read from ``state.json`` and the recorded prompts."""
    run_dir = Path(run_dir)
    prov: dict[str, str] = {}
    cfg = _doc(run_dir / control.CONFIG_FILE)
    if events is None:
        events, _ = read_transcript(run_dir)
    chats = [e for e in events if e.get("event") == "chat"]
    system = None
    if chats:
        msgs = chats[0].get("messages") or []
        if msgs and msgs[0].get("role") == "system":
            system = msgs[0].get("content")
    if cfg:
        search = dict(cfg.get("search") or {})
        llm = dict(cfg.get("llm") or {})
        for k in search:
            prov[f"search.{k}"] = "run_config.json"
        for k in llm:
            prov[f"llm.{k}"] = "run_config.json"
        return {"search": search, "llm": llm, "provenance": prov, "system": system}
    state = _doc(run_dir / "state.json") or {}
    result = _doc(run_dir / "result.json") or {}
    search: dict = {}
    llm: dict = {}
    first_budget = _first((result.get("initial_budget_usd")), control.budget_history(run_dir)["initial_budget_usd"], result.get("budget_usd"), state.get("initial_budget_usd"), state.get("budget_usd"))
    if first_budget is not None:
        search["budget_usd"], prov["search.budget_usd"] = float(first_budget), "state.json / result.json"
    if state.get("max_rounds") is not None:
        search["max_rounds"], prov["search.max_rounds"] = int(state["max_rounds"]), "state.json"
    if state.get("model") or result.get("model"):
        llm["model"], prov["llm.model"] = state.get("model") or result.get("model"), "state.json"
    if system:
        llm["use_tools"] = "# Skill 3: MSL reference (retrieval)" in system
        prov["llm.use_tools"] = "recorded system prompt"
        # profile_poor: Skill 2b is in the system prompt of runs started after it existed; earlier runs show it in the planner feedback or in profiles/
        seen = "# Skill 2b:" in system or (run_dir / "profiles").is_dir() and any((run_dir / "profiles").iterdir()) or any(
            "Profiler evidence for this kernel" in m.get("content", "") for e in chats if str(e.get("label", "")).startswith("r") for m in e.get("messages", []) if m.get("role") == "user"
        )
        search["profile_poor"], prov["search.profile_poor"] = bool(seen), "recorded prompts / profiles/"
        search["hw_feedback"], prov["search.hw_feedback"] = "# Skill 2c:" in system, "recorded system prompt"  # Skill 2c is only in the prompt of runs started with search.hw_feedback
    # planner prompt: number of plans, and whether the last plan is the local one
    for e in chats:
        if str(e.get("label", "")).startswith("r1.plan"):
            users = [m.get("content", "") for m in e.get("messages", []) if m.get("role") == "user"]
            text = users[0] if users else ""
            m = re.search(r"propose (\d+) DIFFERENT", text)
            if m:
                search["branches"], prov["search.branches"] = int(m.group(1)), "recorded plan prompt"
            search["plan_mix"], prov["search.plan_mix"] = "LOCAL, low-risk change" in text, "recorded plan prompt"
            break
    # repair prompt: "repair attempt k/N"
    for e in chats:
        for msg in e.get("messages", []):
            m = re.search(r"repair attempt \d+/(\d+)", msg.get("content", "") if msg.get("role") == "user" else "")
            if m:
                search["max_repair_tries"], prov["search.max_repair_tries"] = int(m.group(1)), "recorded repair prompt"
                break
        if "max_repair_tries" in search:
            break
    # initial samples: the parallel first attempts are labelled init, init.s2, init.s3, ...
    labels = {str(e.get("label", "")) for e in chats}
    samples = {int(m.group(1)) for lab in labels for m in [re.match(r"^init(?:\d+)?(?:\.s(\d+))?(?:\.fix\d+)?(?:#t\d+)?$", lab)] if m and m.group(1)}
    if labels and any(l.startswith("init") for l in labels):
        search["init_samples"], prov["search.init_samples"] = (max(samples) if samples else 1), "recorded labels"
    return {"search": search, "llm": llm, "provenance": prov, "system": system}


# ---------------------------------------------------------------------------------------------
# rebuilding the state
# ---------------------------------------------------------------------------------------------


def _requests_from_transcript(events: list[dict], notes: list[str]) -> list[dict]:
    from .llm import split_usage, token_counts

    reqs: list[dict] = []
    running = 0.0
    drift = 0.0
    for e in events:
        if e.get("event") == "agent":
            cost = e.get("cost_usd")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                continue
            hit, miss, write, out = token_counts(e.get("tokens"))
            running += float(cost)
            reqs.append({
                "t": None, "label": e.get("label", ""), "hit": hit, "miss": miss, "write": write, "out": out,
                "cost_usd": float(cost), "total_usd": running,
            })
            continue
        if e.get("event") != "chat":
            continue
        hit, miss, write, out = split_usage(e.get("usage") if isinstance(e.get("usage"), dict) else {})
        cost = float(e.get("cost_usd") or 0.0)
        running += cost
        total = float(e.get("total_usd", running))
        drift = max(drift, abs(total - running))
        reqs.append({"t": None, "label": e.get("label", ""), "hit": hit, "miss": miss, "write": write, "out": out, "cost_usd": cost, "total_usd": total})
    if drift > 1e-6:
        notes.append(f"transcript totals differ from the sum of request costs by up to ${drift:.6f} (an earlier session's transcript is missing?): the recorded running totals are used")
    return reqs


def _label_info(label: str) -> dict | None:
    m = LABEL_RE.match(label or "")
    if not m:
        return None
    return {"base": m["base"], "round": int(m["round"]), "plan": int(m["plan"]), "attempt": int(m["fix"] or 0), "turn": int(m["turn"]) if m["turn"] is not None else None}


def _read_candidates(run_dir: Path, notes: list[str]) -> list[dict]:
    from tcenv.evaluate import EvalReport
    from tcenv.schema import SchemaError

    out = []
    root = Path(run_dir) / "candidates"
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        m = CAND_DIR_RE.match(d.name)
        if not m or not (d / "kernel.metal").exists():
            continue
        report_path = d / "report.json"
        if not report_path.exists():
            notes.append(f"candidate {d.name} has no report.json: skipped")
            continue
        try:
            rep = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise NeedsInspection(f"{report_path} is damaged and needs human inspection") from e
        try:
            report = EvalReport.from_json(rep)
        except SchemaError as e:
            raise NeedsInspection(f"{report_path} needs human inspection: {e}") from e
        rejected = False
        recheck_path = d / "differential_recheck.json"
        if recheck_path.is_file():
            try:
                rec = json.loads(recheck_path.read_text(encoding="utf-8"))
                rejected = bool(isinstance(rec, dict) and rec.get("counts_as_kernel_error"))
            except (OSError, ValueError):
                rejected = False
        out.append({"idx": int(m["idx"]), "label": m["label"], "source": (d / "kernel.metal").read_text(), "report": report, "plan": None, "recheck_rejected": rejected})
    out.sort(key=lambda c: c["idx"])
    return out


def _eligible(report, correctness_only: bool) -> bool:
    if report is None:
        return False
    return candidate_eligible(bool(report.eligible), report.speedup, correctness_only=correctness_only)


def _speedup(report, correctness_only: bool) -> float:
    recorded = None if report is None else report.speedup
    return comparison_speedup(_eligible(report, correctness_only), recorded)


def _replay(cands: list[dict], search) -> list[dict]:
    """The selection the orchestrator makes (``Orchestrator._eval``), replayed over the stored reports."""
    from .orchestrator import speedup_se

    best = None
    out = []
    for c in cands:
        rep = c["report"]
        sp = _speedup(rep, search.correctness_only)
        eligible = _eligible(rep, search.correctness_only)
        best_sp = None if best is None else best["sp"]
        best_se = 0.0 if best is None else speedup_se(best["report"])
        if c.get("recheck_rejected"):
            relation = "keep"
        else:
            relation = relation_to_best(
                eligible, sp, best_sp, speedup_se(rep), best_se,
                min_improvement=search.min_improvement, noise_z=search.noise_z,
            )
        improved = relation == "replace"
        within = relation == "tie"
        if improved:
            best = {"report": rep, "sp": sp, "label": c["label"]}
        out.append({"idx": c["idx"], "label": c["label"], "status": rep.status, "speedup": rep.speedup, "best_speedup": best["sp"] if best else None, "improved": improved, "within_noise": within})
    return out



def _refuse_old_protocol(info: dict, run_dir: Path) -> bool:
    """True when this directory predates flame_chase_1, so ``load_snapshot`` would refuse it.

    The directory stays readable (brief, dashboard, run details); only the resume button must not appear.
    Without this, the dashboard reads ``resumable: true`` from a run that ``taic resume`` always rejects.
    """
    protocol = _loop_protocol(run_dir)
    if protocol == "flame_chase_1":
        return False
    info.update(
        kind="old_protocol",
        reason=f"loop_protocol is {protocol!r}: this code continues only flame_chase_1 runs, so this one can be read but not resumed",
    )
    return True


def _loop_protocol(run_dir: Path) -> str | None:
    doc = control.read_json(Path(run_dir) / "run_config.json") or {}
    value = doc.get("loop_protocol") if isinstance(doc, dict) else None
    return value if isinstance(value, str) else None


def _load_flame_chase(run_dir: Path, search, kernel_name: str) -> Snapshot:
    """Resume a ``flame_chase_1`` run. A round that already has a candidate is finished; an open round starts a new session.

    There is no partial conversation to continue. ``kernel_name`` is unused: the kernel text is whatever the candidate file holds.
    """
    del kernel_name
    notes: list[str] = []
    state, result = read_versioned(run_dir / "state.json", STATE_SCHEMA_VERSION), read_versioned(run_dir / "result.json", RESULT_SCHEMA_VERSION)
    if state is None and result is None:
        raise ResumeError(f"{run_dir} has neither state.json nor result.json")
    events, bad = read_transcript(run_dir)
    if bad:
        notes.append(f"{bad} damaged line(s) in transcript.jsonl skipped")
    ref = result or state or {}
    task, dtype, size, model = ref.get("task", ""), ref.get("dtype", ""), ref.get("size", ""), ref.get("model", "")
    requests = _requests_from_transcript(events, notes)
    ledger_doc = read_versioned(run_dir / "ledger.json", LEDGER_SCHEMA_VERSION) or {}
    if isinstance(ledger_doc.get("requests"), list) and ledger_doc["requests"]:
        if not requests:
            from .artifacts import NeedsInspection

            raise NeedsInspection(f"{run_dir / 'ledger.json'} has a request list but transcript.jsonl does not; a checkpoint cannot replace the billed transcript and needs human inspection")
        notes.append("ledger.json holds a request list; it was not applied. transcript.jsonl is the billed-request record")
    from_state = _first((state or {}).get("spent_usd"), ((state or {}).get("ledger") or {}).get("total_usd"))
    total = float(requests[-1]["total_usd"]) if requests else 0.0
    if isinstance(from_state, (int, float)) and not isinstance(from_state, bool) and float(from_state) > total + 1e-6:
        gap = float(from_state) - total
        notes.append(f"state.json reports ${float(from_state):.4f} but the transcript only accounts for ${total:.4f}: the ${gap:.4f} difference is carried as one 'unaccounted' entry so the spend is never under-counted")
        requests.append({"t": None, "label": "unaccounted", "hit": 0, "miss": 0, "out": 0, "cost_usd": gap, "total_usd": float(from_state)})
    cands = _read_candidates(run_dir, notes)
    cand_root = run_dir / "candidates"
    if cand_root.is_dir() and any(cand_root.iterdir()) and not cands:
        raise ResumeError("no candidates/ were stored: nothing to continue from, start a new run")
    recorded = max([d.get("trace") for d in (state, result) if d and isinstance(d.get("trace"), list)] or [[]], key=len)
    trace = [dict(t) for t in recorded if isinstance(t, dict)]
    best_label = (state or {}).get("search_seed_label") or (state or {}).get("best_label") or (result or {}).get("search_seed_label")
    if best_label and not any(c["label"] == best_label for c in cands):
        best_label = next((t["label"] for t in reversed(trace) if t.get("improved")), None)
    seen = {t.get("label") for t in trace}
    for cand in cands:
        if cand["label"] in seen:
            continue
        report = cand.get("report")
        speed = getattr(report, "speedup", None)
        if getattr(report, "status", None) == "ok" and isinstance(speed, (int, float)):
            from .orchestrator import speedup_se

            best_sp = None
            best_report = None
            for earlier in cands:
                if earlier["label"] == best_label:
                    best_report = earlier.get("report")
                    break
            for row in reversed(trace):
                if row.get("label") == best_label and isinstance(row.get("speedup"), (int, float)):
                    best_sp = float(row["speedup"])
                    break
            relation = relation_to_best(
                True, float(speed), best_sp, speedup_se(report), speedup_se(best_report),
                min_improvement=search.min_improvement, noise_z=search.noise_z,
            )
            improved = relation == "replace"
            notes.append(f"{cand['label']} was rebuilt by replaying the selection")
            if improved:
                notes.append(
                    f"search_seed_label {best_label} is older than {cand['label']} evaluated after state.json: "
                    "the resume continues from the newer kernel"
                )
                best_label = cand["label"]
            trace.append({
                "idx": cand["idx"], "label": cand["label"], "improved": improved, "within_noise": relation == "tie",
                "speedup": speed, "status": "ok", "cost_usd": total, "t_s": 0.0,
            })
            seen.add(cand["label"])
    rnd = int(_first((state or {}).get("round"), (result or {}).get("rounds"), 0) or 0)
    finished = any(c["label"] == f"r{rnd}.p0" or c["label"].startswith(f"r{rnd}.p0.") for c in cands)
    if rnd <= 0:
        next_round = 1
    elif finished:
        next_round = rnd + 1
    else:
        next_round = rnd
        notes.append(f"round {rnd} has no candidate; it starts again as a new session")
    history = list((state or {}).get("history") or [])
    stagnant = int((state or {}).get("stagnant") or 0)
    agent_failures = int((state or {}).get("agent_failures") or 0)
    hangs = int(_first((state or {}).get("hangs"), 0) or 0)
    evaluations = int(_first((state or {}).get("evaluations"), len(trace)) or 0)
    crashes = int((state or {}).get("crashes_in_a_row") or 0)
    started = _first((state or {}).get("started_at"))
    elapsed = max([float(t.get("t_s") or 0.0) for t in trace] + [0.0])
    system_path = run_dir / "system_prompt.txt"
    system = system_path.read_text() if system_path.is_file() else None
    if system is None:
        notes.append("system_prompt.txt is missing; the continued run keeps the prompt this process builds")
    resumes = int(_first((state or {}).get("resumes"), 0) or 0)
    hist = control.budget_history(run_dir)
    initial = _first(hist["initial_budget_usd"], (state or {}).get("initial_budget_usd"), (state or {}).get("budget_usd"))
    return Snapshot(
        run_dir=str(run_dir), task=task, dtype=dtype, size=size, model=model, requests=requests, candidates=cands, trace=trace,
        history=history, best_label=best_label, rounds_done=max(0, next_round - 1), next_round=next_round, stagnant=stagnant,
        partial=None, hangs=hangs, evaluations=evaluations, crashes_in_a_row=crashes, system=system, elapsed_s=elapsed,
        started_at=started, initial_budget_usd=initial, resumes=resumes, agent_failures=agent_failures, notes=notes,
        checks={"loop_protocol": "flame_chase_1"},
    )


def load_snapshot(run_dir: Path, search, kernel_name: str, *, settings: dict | None = None) -> Snapshot:
    """Rebuild the search state of ``run_dir`` (read-only).

    A directory without ``loop_protocol: flame_chase_1`` is refused. The chat-and-branches protocol is not continued.
    """
    del settings
    run_dir = Path(run_dir)
    protocol = _loop_protocol(run_dir)
    if protocol != "flame_chase_1":
        raise ResumeError(
            f"{run_dir} cannot be resumed: loop_protocol is {protocol!r}. This code only continues flame_chase_1. The search protocol changed."
        )
    return _load_flame_chase(run_dir, search, kernel_name)



def describe(snap: Snapshot) -> dict:
    """What was rebuilt, as a small JSON-able report (the dry run prints it)."""
    p = snap.partial
    return {
        "run_dir": snap.run_dir, "task": snap.task, "dtype": snap.dtype, "model": snap.model,
        "spent_usd": round(snap.spent_usd, 6), "requests": len(snap.requests), "candidates": len(snap.candidates), "trace_entries": len(snap.trace),
        "best_label": snap.best_label, "best_speedup": next((t["best_speedup"] for t in reversed(snap.trace) if t.get("best_speedup") is not None), None),
        "rounds_completed": snap.rounds_done, "history_lines": len(snap.history), "stagnant_rounds": snap.stagnant,
        "evaluations": snap.evaluations, "hangs": snap.hangs, "profiles": len(snap.profiles), "elapsed_s": round(snap.elapsed_s, 1),
        "system_prompt_restored": bool(snap.system),
        "continues_in_round": p.rnd if p else snap.next_round,
        "round": None if not p else {
            "seed": p.seed_label, "plans": [x["title"] for x in (p.plans or [])] or None, "plans_from": p.plan_source,
            "finished_branches": {f"p{i}": lab for i, lab in p.done.items()},
            "continuing_branches": {f"p{i}": {"attempt": b.attempt, "turn": b.turn, "pending_tool_calls": len(b.pending), "reply_to_evaluate": b.reply_text is not None, "why": b.note} for i, b in p.resume.items()},
            "branches_to_start": [f"p{i}" for i in range(len(p.plans or [])) if i not in p.done and i not in p.resume],
        },
        "notes": snap.notes, "checks": snap.checks,
    }


# ---------------------------------------------------------------------------------------------
# preparing the directory for the continued run
# ---------------------------------------------------------------------------------------------


def _next_index(run_dir: Path, stem: str) -> int:
    n = 1
    while any(run_dir.glob(f"{stem}.before_topup_{n}.json")) or (run_dir / f"{stem}.before_resume_{n}.json").exists():
        n += 1
    return n


def snapshot_results(run_dir: Path) -> list[str]:
    """Before a continued run can overwrite them: keep ``result.json`` and ``confirm_report.json`` as ``<name>.before_topup_<n>.json`` (and the old
    ``state.json`` as ``state.before_resume_<n>.json``). The files are *moved*, so a directory without ``result.json`` is a run in progress."""
    run_dir = Path(run_dir)
    moved = []
    n = _next_index(run_dir, "result")
    for name in ("result", "confirm_report"):
        src = run_dir / f"{name}.json"
        if src.exists():
            dst = run_dir / f"{name}.before_topup_{n}.json"
            control.write_json_atomic(dst, json.loads(src.read_text()))
            src.unlink()
            moved.append(dst.name)
    st = run_dir / "state.json"
    if st.exists():
        dst = run_dir / f"state.before_resume_{n}.json"
        try:
            control.write_json_atomic(dst, json.loads(st.read_text()))
            moved.append(dst.name)
        except ValueError:
            pass
    return moved


def restore_snapshot(run_dir: Path, moved: list[str]) -> None:
    """Put ``result.json`` and ``confirm_report.json`` back when startup fails after ``snapshot_results`` moved them."""
    run_dir = Path(run_dir)
    for name in moved:
        if name.startswith("state."):
            continue
        src = run_dir / name
        stem, marker, _rest = name.partition(".before_topup_")
        if not marker or not src.is_file():
            continue
        dest = run_dir / f"{stem}.json"
        if not dest.exists():
            src.replace(dest)


def effective_budget(info: dict, *, add_usd: float | None, budget_usd: float | None, control_doc: dict | None = None) -> float:
    """The total budget of the continued run: ``--budget``, else recorded + ``--add-budget``, else the larger of the recorded budget and a request still
    waiting in control.json. Must exceed what is spent."""
    if info.get("needs_budget") and add_usd is None and budget_usd is None:
        pending = control.parse_control(control_doc)[0] if control_doc else None
        if pending is None:
            raise ResumeError(
                "the run stopped with too little budget for another normal session: "
                "add budget with --add-budget USD or --budget TOTAL"
            )
    spent = float(info.get("spent_usd") or 0.0)
    recorded = info.get("budget_usd")
    recorded = float(recorded) if recorded is not None else None
    if budget_usd is not None:
        new = float(budget_usd)
    elif add_usd is not None:
        if recorded is None:
            raise ResumeError("the run does not record its budget: give the new total with --budget")
        new = recorded + float(add_usd)
    else:
        pending = control.parse_control(control_doc)[0] if control_doc else None
        new = max([x for x in (recorded, pending) if x is not None], default=None)
        if new is None:
            raise ResumeError("the run does not record its budget: give it with --budget")
    if not math.isfinite(new) or new <= 0:
        raise ResumeError("the budget must be a positive number")
    if new <= spent + 1e-9:
        raise ResumeError(f"budget ${new:.2f} does not exceed what the run has already spent (${spent:.4f}): add budget with --add-budget USD or --budget TOTAL")
    return new
