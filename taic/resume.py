"""Continue a run from what its directory holds: after a crash, a closed window, or a stop for budget.

``python -m taic.cli resume <run_dir> [--add-budget USD | --budget USD] [--dry-run]`` rebuilds the search state from the files and goes on with the
rounds that were not finished. Nothing here spends money or touches the GPU except ``Orchestrator.run()`` after ``restore()``.

What the directory holds, and what is rebuilt from it
-----------------------------------------------------
* cost ledger        every model request is in ``transcript.jsonl`` with its usage, its cost and the running total, so the money spent so far is exact
                     (``ledger.json`` exists only when a run ended, ``state.json`` only knows the total as of its last write).
* candidates         ``candidates/NNN_<label>/{kernel.metal,report.json}``; the best kernel, the evaluation cache and the trace are rebuilt from them
                     (``state.json`` keeps the trace too; the selection is replayed and compared with it).
* finished rounds    the hypotheses tried (``history``) are rebuilt from the planner's replies in the transcript plus the candidates of each branch.
* the round that was interrupted
                     its plans come from the transcript (no new planner request), branches that had finished stay finished, and a branch that was
                     part-way continues from its recorded conversation (pending tool calls are answered locally; the next request is the one that was due).
* settings           ``run_config.json`` (new runs). Older runs: budget and round limit from ``state.json``; tool use, profiler skill, plan mix, branches and
                     repair tries from the recorded prompts; the system prompt itself is reused from the transcript. ``patience`` and ``reasoning_effort``
                     are not recorded by older runs: give them on the command line if they differed from the preset.

What cannot be recovered: a request that was in flight when the process died (the provider may have billed it, the ledger never saw it), the
wording of the history line of a branch whose model request failed (rebuilt as "no valid kernel after repair"), Instruments traces that failed (not
listed), the ledger's per-request wall-clock times of older sessions (``t`` is ``null`` for them).
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

FEEDBACK_CHARS = 6000  # as in the orchestrator
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


def inspect_run(run_dir: Path, *, pid_check=control.pid_alive) -> dict:
    """Whether and how ``run_dir`` can be continued, from its small files only (the dashboard calls this for every click).

    ``kind``: ``running`` (a live process owns it; ``can_topup_live`` says whether it polls the control file), ``resuming`` (a resume holds the lock),
    ``interrupted`` (no result, no live process), ``budget_exhausted`` (finished, stopped for money), ``finished`` (any other stop), ``unknown``."""
    run_dir = Path(run_dir)
    state, result = _doc(run_dir / "state.json"), _doc(run_dir / "result.json")
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
        info["needs_budget"] = spent is not None and budget is not None and float(spent) >= float(budget) - 1e-9
    except (TypeError, ValueError):
        pass
    if holder:
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
        info["resumable"] = True
        return info
    if stop == STOP_BUDGET:
        info["kind"] = "budget_exhausted"
        if not has_best(state, result):
            info["reason"] = "stopped for budget without a valid kernel: start a new run"
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
    from .llm import split_usage

    reqs: list[dict] = []
    running = 0.0
    drift = 0.0
    for e in events:
        if e.get("event") != "chat":
            continue
        hit, miss, out = split_usage(e.get("usage") if isinstance(e.get("usage"), dict) else {})
        cost = float(e.get("cost_usd") or 0.0)
        running += cost
        total = float(e.get("total_usd", running))
        drift = max(drift, abs(total - running))
        reqs.append({"t": None, "label": e.get("label", ""), "hit": hit, "miss": miss, "out": out, "cost_usd": cost, "total_usd": total})
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

    out = []
    root = Path(run_dir) / "candidates"
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        m = CAND_DIR_RE.match(d.name)
        if not m or not (d / "kernel.metal").exists():
            continue
        rep = control.read_json(d / "report.json")
        if not isinstance(rep, dict):
            notes.append(f"candidate {d.name} has no readable report.json: skipped")
            continue
        out.append({"idx": int(m["idx"]), "label": m["label"], "source": (d / "kernel.metal").read_text(), "report": EvalReport.from_json(rep), "plan": None})
    out.sort(key=lambda c: c["idx"])
    return out


def _eligible(report, correctness_only: bool) -> bool:
    if report is None or not report.eligible:
        return False
    return True if correctness_only else report.speedup is not None


def _speedup(report, correctness_only: bool) -> float:
    if not _eligible(report, correctness_only):
        return 0.0
    return report.speedup if report.speedup is not None else 1.0


def _replay(cands: list[dict], search) -> list[dict]:
    """The selection the orchestrator makes (``Orchestrator._eval``), replayed over the stored reports."""
    from .orchestrator import speedup_se

    best = None
    out = []
    for c in cands:
        rep = c["report"]
        sp = _speedup(rep, search.correctness_only)
        improved = within = False
        if _eligible(rep, search.correctness_only):
            if best is None:
                improved = True
            else:
                gain = max(search.min_improvement, 1.0 + search.noise_z * math.hypot(speedup_se(rep), speedup_se(best["report"])))
                if sp > best["sp"] * gain:
                    improved = True
                elif sp > best["sp"]:
                    within = True
        if improved:
            best = {"report": rep, "sp": sp, "label": c["label"]}
        out.append({"idx": c["idx"], "label": c["label"], "status": rep.status, "speedup": rep.speedup, "best_speedup": best["sp"] if best else None, "improved": improved, "within_noise": within})
    return out


def _round_plans(events: list[dict], rnd: int, k: int) -> tuple[list[dict] | None, str]:
    """The plans of round ``rnd`` as the planner's replies in the transcript give them (``parse_plans`` is what the orchestrator used)."""
    from .parse import parse_plans

    evs = [e for e in events if e.get("event") == "chat" and e.get("label") in (f"r{rnd}.plan", f"r{rnd}.plan.retry")]
    for e in evs:
        _, plans = parse_plans((e.get("reply") or {}).get("content") or "", k)
        if plans:
            return plans, f"transcript ({e['label']})"
    if len(evs) >= 2:
        return None, "fallback"  # both planner attempts were unusable: the orchestrator used its generic plans
    return None, "missing" if not evs else "unparsed"


def _quick_used(conv: list[dict]) -> int:
    """quick_check calls already made in the request that ``conv`` (ending with an assistant message) belongs to."""
    n = 0
    for m in reversed(conv[:-1]):
        if m.get("role") == "user":
            break
        if m.get("role") == "assistant":
            n += sum(1 for t in (m.get("tool_calls") or []) if (t.get("function") or {}).get("name") == "quick_check")
    return n


def _branch(rnd: int, i: int, chats: list[dict], cands: list[dict], search, kernel_name: str) -> tuple[str, str | None, BranchResume | None]:
    """State of branch ``i`` of round ``rnd``: ``("done", label | None, None)``, ``("partial", None, BranchResume)`` or ``("notstarted", None, None)``."""
    from tcenv.report import format_feedback

    from . import prompts
    from .orchestrator import nudge_user
    from .parse import extract_metal

    base = f"r{rnd}.p{i}"
    evs = [e for e in chats if (_label_info(e.get("label", "")) or {}).get("base") == base]
    cs = [c for c in cands if c["label"] == base or c["label"].startswith(base + ".fix")]
    if cs and _eligible(cs[-1]["report"], search.correctness_only):
        return "done", cs[-1]["label"], None
    if not evs:
        return ("done", None, None) if cs else ("notstarted", None, None)
    last = evs[-1]
    li = _label_info(last["label"])
    attempt, turn = li["attempt"], li["turn"] or 0
    conv = list(last.get("messages") or []) + [last.get("reply") or {"role": "assistant", "content": ""}]
    reply = conv[-1]
    max_try = search.max_repair_tries
    tcs = reply.get("tool_calls") or []
    if tcs:
        pending = [{"id": t["id"], "name": t["function"]["name"], "arguments": t["function"]["arguments"]} for t in tcs]
        return "partial", None, BranchResume(attempt, conv, turn, pending, None, _quick_used(conv), f"waiting for {len(pending)} tool result(s) of {last['label']}")
    text = reply.get("content") or ""
    tag = base if attempt == 0 else f"{base}.fix{attempt}"
    cand = next((c for c in cs if c["label"] == tag), None)
    if extract_metal(text, kernel_name) is None:
        if attempt >= max_try:
            return "done", None, None
        return "partial", None, BranchResume(attempt + 1, conv + [{"role": "user", "content": nudge_user(kernel_name)}], 0, [], None, 0, f"reply of {last['label']} had no code block: next is the reminder")
    if cand is None:
        return "partial", None, BranchResume(attempt, conv, turn, [], text, 0, f"the reply of {last['label']} was received but never evaluated")
    if cand["report"].status == "timeout" or attempt >= max_try:
        return "done", None, None
    fb = format_feedback(cand["report"])[:FEEDBACK_CHARS]
    return "partial", None, BranchResume(attempt + 1, conv + [{"role": "user", "content": prompts.repair_user(fb, attempt + 1, max_try)}], 0, [], None, 0, f"{tag} failed evaluation: next is repair attempt {attempt + 1}")


def load_snapshot(run_dir: Path, search, kernel_name: str, *, settings: dict | None = None) -> Snapshot:
    """Rebuild the search state of ``run_dir`` (read-only). ``search`` is the ``SearchConfig`` the continued run will use (it fixes ``branches``,
    ``max_repair_tries`` and the selection margins that the rebuild replays); ``kernel_name`` is the task's kernel."""
    from .orchestrator import FALLBACK_PLANS

    run_dir = Path(run_dir)
    notes: list[str] = []
    checks: dict = {}
    state, result = _doc(run_dir / "state.json"), _doc(run_dir / "result.json")
    if state is None and result is None:
        raise ResumeError(f"{run_dir} has neither state.json nor result.json")
    events, bad = read_transcript(run_dir)
    if bad:
        notes.append(f"{bad} damaged line(s) in transcript.jsonl skipped (a write that was cut off by the kill)")
    chats = [e for e in events if e.get("event") == "chat"]
    ref = result or state or {}
    task, dtype, size, model = ref.get("task", ""), ref.get("dtype", ""), ref.get("size", ""), ref.get("model", "")

    # --- ledger -------------------------------------------------------------------------------
    requests = _requests_from_transcript(events, notes)
    ledger_doc = _doc(run_dir / "ledger.json") or {}
    from_state = _first((state or {}).get("spent_usd"), ((state or {}).get("ledger") or {}).get("total_usd"))
    from_ledger = _first(((ledger_doc.get("summary") or {}).get("total_usd")), (result or {}).get("spent_usd"))
    if not requests and isinstance(ledger_doc.get("requests"), list) and ledger_doc["requests"]:
        requests = [dict(r) for r in ledger_doc["requests"]]
        notes.append("transcript.jsonl is missing: the ledger comes from ledger.json")
    total = float(requests[-1]["total_usd"]) if requests else 0.0
    checks["ledger_requests"] = len(requests)
    checks["ledger_total_usd"] = total
    checks["state_total_usd"] = from_state
    best_known = max([x for x in (from_state, from_ledger) if x is not None], default=0.0)
    if best_known > total + 1e-6:
        gap = best_known - total
        notes.append(f"state.json / ledger.json report ${best_known:.4f} but the transcript only accounts for ${total:.4f}: the ${gap:.4f} difference is carried as one 'unaccounted' entry so the spend is never under-counted")
        requests.append({"t": None, "label": "unaccounted", "hit": 0, "miss": 0, "out": 0, "cost_usd": gap, "total_usd": best_known})
    elif from_state is not None and total > from_state + 1e-6:
        notes.append(f"the transcript holds ${total - from_state:.4f} more than state.json (requests answered after its last write): the transcript is exact")

    # --- candidates, trace, best --------------------------------------------------------------
    cands = _read_candidates(run_dir, notes)
    if not cands:
        raise ResumeError("no candidates/ were stored: nothing to continue from, start a new run")
    recorded = max([d.get("trace") for d in (state, result) if d and isinstance(d.get("trace"), list)] or [[]], key=len)
    replayed = _replay(cands, search)
    n_rec = min(len(recorded), len(cands))
    mism = sum(1 for a, b in zip(recorded[:n_rec], replayed[:n_rec]) if bool(a.get("improved")) != b["improved"] or a.get("label") != b["label"])
    checks["trace_recorded"], checks["candidates"], checks["selection_replay_mismatches"] = len(recorded), len(cands), mism
    if mism:
        notes.append(f"replaying the selection over the stored reports disagrees with the recorded trace on {mism} candidate(s): the recorded trace is trusted")
    trace = [dict(t) for t in recorded[:n_rec]]
    last_cost = trace[-1]["cost_usd"] if trace else 0.0
    last_t = trace[-1]["t_s"] if trace else 0.0
    for r in replayed[n_rec:]:
        tag = r["label"]
        cost = next((float(e["total_usd"]) for e in reversed(chats) if e.get("label") == tag or str(e.get("label", "")).startswith(tag + "#")), last_cost)
        last_cost = max(last_cost, cost)
        trace.append({**r, "cost_usd": last_cost, "t_s": last_t})
    if len(trace) > n_rec:
        notes.append(f"{len(trace) - n_rec} candidate(s) were evaluated after state.json was last written: their trace entries are rebuilt by replaying the selection")
    if len(recorded) > len(cands):
        notes.append(f"state.json lists {len(recorded) - len(cands)} more candidate(s) than candidates/ holds: those entries are dropped")
    best_label = next((t["label"] for t in reversed(trace) if t.get("improved")), None)
    checks["best_label_state"] = (state or {}).get("best_label")
    checks["best_label"] = best_label
    if state and state.get("best_label") and state["best_label"] != best_label and not (result and result.get("stop_reason")):
        notes.append(f"best kernel per state.json is {state['best_label']} but the trace says {best_label}: the trace is used")

    # --- rounds, plans, history ---------------------------------------------------------------
    label_rounds = [li["round"] for c in cands for li in [_label_info(c["label"])] if li]
    plan_rounds = [int(m.group(1)) for e in chats for m in [re.match(r"^r(\d+)\.plan", str(e.get("label", "")))] if m]
    last_round = max([int((state or {}).get("round") or 0), int((result or {}).get("rounds") or 0), *label_rounds, *plan_rounds], default=0)
    k = search.branches
    by_label = {c["label"]: c for c in cands}
    history: list[str] = []
    stagnant = 0
    best_at_end: dict[int, str | None] = {}
    cur = None
    entries_by_round: dict[int, list[dict]] = {}
    for t in trace:
        li = _label_info(t["label"])
        r = li["round"] if li else 0
        entries_by_round.setdefault(r, []).append(t)
    for r in range(0, last_round + 1):
        for t in entries_by_round.get(r, []):
            if t.get("improved"):
                cur = t["label"]
        best_at_end[r] = cur
    plans_by_round: dict[int, list[dict] | None] = {}
    for r in range(1, last_round):  # complete rounds
        plans, how = _round_plans(events, r, k)
        if plans is None and how == "fallback":
            plans = [dict(p) for p in FALLBACK_PLANS[:k]]  # both planner replies were unusable: the orchestrator used its generic plans
        elif plans is None:
            # no planner reply in the transcript (it is missing or damaged): the titles stored with the candidates in result.json are the next best thing
            titles: dict[int, str] = {}
            for c in (result or {}).get("candidates") or []:
                li = _label_info(str(c.get("label", "")))
                if li and li["round"] == r and c.get("plan"):
                    titles.setdefault(li["plan"], str(c["plan"]))
            plans = [{"title": titles.get(i, f"(plan {i + 1} of round {r}: title not recorded)"), "change": "", "expected_gain": ""} for i in range(k)]
            notes.append(f"round {r}: the planner's reply is not in the transcript ({how}); plan titles are taken from result.json" if titles else f"round {r}: the planner's reply is not in the transcript ({how}) and result.json has no plan titles: the history line of that round is approximate")
        plans_by_round[r] = plans
        for i, p in enumerate(plans):
            st, lab, _ = _branch(r, i, chats, cands, search, kernel_name)
            if st == "done" and lab:
                cand = by_label[lab]
                t = next(t for t in trace if t["label"] == lab)
                tag = f"{_speedup(cand['report'], search.correctness_only):.3f}x" + (" (new best)" if best_at_end[r] == lab else " (within measurement noise of the best)" if t.get("within_noise") else " (not better)")
            else:
                tag = "no valid kernel after repair"
            history.append(f"round {r}: {p['title']} -> {tag}")
        if any(t.get("improved") for t in entries_by_round.get(r, [])):
            stagnant = 0
        else:
            stagnant += 1
    for r, plans in plans_by_round.items():
        for i, p in enumerate(plans):
            for c in cands:
                li = _label_info(c["label"])
                if li and li["round"] == r and li["plan"] == i:
                    c["plan"] = p["title"]

    partial = None
    if last_round >= 1:
        r = last_round
        first_idx = next((j for j, t in enumerate(trace) if (_label_info(t["label"]) or {}).get("round") == r), None)
        seed_label = best_at_end.get(r - 1) if first_idx is None else next((t["label"] for t in reversed(trace[:first_idx]) if t.get("improved")), None)
        seed_label = seed_label or best_label
        plans, how = _round_plans(events, r, k)
        any_branch = any((_label_info(c["label"]) or {}).get("round") == r for c in cands) or any((_label_info(e.get("label", "")) or {}).get("round") == r for e in chats)
        if plans is None and how == "fallback" or (plans is None and any_branch):
            plans, how = [dict(p) for p in FALLBACK_PLANS[:k]], "generic plans (the planner's replies were unusable)"
        done: dict[int, str | None] = {}
        resume: dict[int, BranchResume] = {}
        if plans:
            for i in range(len(plans)):
                st, lab, br = _branch(r, i, chats, cands, search, kernel_name)
                if st == "done":
                    done[i] = lab
                elif st == "partial":
                    resume[i] = br
            for c in cands:
                li = _label_info(c["label"])
                if li and li["round"] == r and li["plan"] < len(plans):
                    c["plan"] = plans[li["plan"]]["title"]
        partial = RoundPartial(r, seed_label, plans, done, resume, how)
    # --- profiler evidence --------------------------------------------------------------------
    profiles: list[dict] = []
    bodies: dict[str, str] = {}
    pdir = run_dir / "profiles"
    if pdir.is_dir():
        for f in sorted(pdir.iterdir()):
            m = PROFILE_RE.match(f.name)
            if not m:
                continue
            if m["ext"] == "txt":
                bodies[m["label"]] = f.read_text()
            else:
                doc = control.read_json(f) or {}
                cand = by_label.get(m["label"])
                profiles.append({"round": int(m["round"]), "seed": m["label"], "seed_speedup": _speedup(cand["report"], search.correctness_only) if cand else None,
                                 "spilled_bytes": doc.get("spilled_bytes"), "shader_code_bytes": doc.get("shader_code_bytes"), "gpu_active_fraction": doc.get("gpu_active_fraction")})
    profiles.sort(key=lambda p: (p["round"], p["seed"]))

    # --- counters, time, system prompt --------------------------------------------------------
    hangs = int(_first((state or {}).get("hangs"), (result or {}).get("hangs"), sum(1 for t in trace if t.get("status") == "timeout")) or 0)
    evaluations = int(_first((state or {}).get("evaluations"), (result or {}).get("evaluations"), len(trace)) or 0)
    crashes = 0
    for t in reversed(trace):
        if t.get("status") != "crash":
            break
        crashes += 1
    started = _first((state or {}).get("started_at"))
    elapsed = max([float(t.get("t_s") or 0.0) for t in trace] + [float((state or {}).get("updated_at") or 0.0) - float(started or 0.0) if started else 0.0])
    system = None
    if chats:
        msgs = chats[0].get("messages") or []
        if msgs and msgs[0].get("role") == "system":
            system = msgs[0].get("content")
    inflight = [v for v in ((control.read_json(run_dir / "llm.json") or {}).get("inflight") or {}).values() if isinstance(v, dict)]
    if inflight:
        notes.append(f"{len(inflight)} model request(s) were in flight when the process stopped ({', '.join(str(v.get('label')) for v in inflight)}): their cost is not in the ledger and cannot be recovered")
    resumes = int(_first((state or {}).get("resumes"), (result or {}).get("resumes"), sum(1 for e in events if e.get("event") == "resume")) or 0)
    hist = control.budget_history(run_dir)
    initial = _first(hist["initial_budget_usd"], (result or {}).get("initial_budget_usd"), (state or {}).get("initial_budget_usd"), (state or {}).get("budget_usd"), (result or {}).get("budget_usd"))
    return Snapshot(
        run_dir=str(run_dir), task=task, dtype=dtype, size=size, model=model, requests=requests, candidates=cands, trace=trace, history=history, best_label=best_label,
        rounds_done=(last_round - 1) if partial else 0, next_round=last_round + 1 if not partial else last_round, stagnant=stagnant, partial=partial, hangs=hangs,
        evaluations=evaluations, crashes_in_a_row=crashes, profiles=profiles, profile_bodies=bodies, system=system, elapsed_s=elapsed, started_at=started, initial_budget_usd=initial,
        resumes=resumes, notes=notes, checks=checks,
    )


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


def effective_budget(info: dict, *, add_usd: float | None, budget_usd: float | None, control_doc: dict | None = None) -> float:
    """The total budget of the continued run: ``--budget``, else recorded + ``--add-budget``, else the larger of the recorded budget and a request still
    waiting in control.json. Must exceed what is spent."""
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
