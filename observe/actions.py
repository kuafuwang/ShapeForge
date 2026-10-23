"""The dashboard's write paths: adding budget to one run, and resetting every run.

What a click can do (``POST /api/runs/<run>/budget``):

* a run that is still running and polls its control file (``state.json`` says ``control_supported``): write ``control.json`` (live top-up);
* a run that stopped for budget, or whose process died: start ``python -m taic.cli resume`` for it as a detached process;
* anything else: refused with the reason (a run that converged, hit its round limit or a GPU hang is not continued; there is no override).

``POST /api/budget/reset`` is the one-click reset: every search that can continue gets $15 added and its round limit and patience lifted.
A run that hung the GPU, failed in the model, or never produced a kernel is skipped. The experiment ``cap.json``, when there is one, is raised by the same amount so the add is not refused.

``POST /api/stop`` sends SIGTERM to every live search and experiment queue in this repo. It signals that process group, and also every process group of a descendant and of a process whose command names that run's directory. ``hmz`` is started in its own session, so a group signal to the search alone leaves ``pi`` running. The dashboard process is never a target.

This spends real money and can start GPU work, so every POST needs: a loopback ``Host``, a same-origin ``Origin`` (when the browser sends one), the
per-process random token that the served page carries (``X-Observe-Token``), a positive amount under ``OBSERVE_MAX_TOPUP_USD`` (default 25), a run
that resolves inside ``runs/`` (no ``..``, no symlink out) and at most one request in flight per run. ``OBSERVE_CONTROL_DRYRUN=1`` does all of the
checks, records the command that *would* run in ``budget_events.json`` and starts nothing. The reset amount is fixed at $15 and is not taken from the request body.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from observe import detail, query
from taic import control, resume

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKEN = secrets.token_urlsafe(32)  # one per server process; the page that this process serves carries it
TOKEN_HEADER = "x-observe-token"
LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}
DEFAULT_MAX_TOPUP_USD = 25.0
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


class Refused(Exception):
    """A request that is not carried out; ``status`` is the HTTP status and ``message`` is shown to the user."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


def max_topup_usd() -> float:
    try:
        v = float(os.environ.get("OBSERVE_MAX_TOPUP_USD", DEFAULT_MAX_TOPUP_USD))
    except ValueError:
        return DEFAULT_MAX_TOPUP_USD
    return v if math.isfinite(v) and v > 0 else DEFAULT_MAX_TOPUP_USD


def dry_run() -> bool:
    return os.environ.get("OBSERVE_CONTROL_DRYRUN", "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------------------------
# who may ask
# ---------------------------------------------------------------------------------------------


def _host_name(netloc: str) -> str:
    netloc = (netloc or "").strip().lower()
    if netloc.startswith("["):
        return netloc.split("]")[0] + "]"
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def host_is_loopback(headers) -> bool:
    return _host_name(headers.get("host", "")) in LOOPBACK


def check_origin(headers) -> None:
    """DNS rebinding and cross-site requests: the Host must be loopback and an Origin, when there is one, must be the page's own origin."""
    if not host_is_loopback(headers):
        raise Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
    origin = headers.get("origin")
    if origin is not None:
        o = urlsplit(origin)
        if o.scheme != "http" or (o.netloc or "").lower() != (headers.get("host") or "").lower():
            raise Refused(403, "refused: cross-origin request")


def check_token(headers) -> None:
    given = headers.get(TOKEN_HEADER, "")
    if not given or not hmac.compare_digest(given.encode(), TOKEN.encode()):
        raise Refused(403, "refused: missing or wrong X-Observe-Token (reload the page)")


def page_token(headers) -> str:
    """The token to embed in the served page: only for a loopback Host (a rebound hostname never gets it)."""
    return TOKEN if host_is_loopback(headers) else ""


# ---------------------------------------------------------------------------------------------
# what is possible for a run
# ---------------------------------------------------------------------------------------------


def _run_dir(run_id: str) -> Path:
    path = query.resolve_run(run_id)
    if path is None:
        raise Refused(404, "no such run")
    return path


def _warnings(run_dir: Path, info: dict) -> list[str]:
    out: list[str] = []
    if info.get("resumable") and not (run_dir / control.CONFIG_FILE).exists():
        out.append("this run was started by an older version: its settings are inferred from its prompts; 'patience' is not recorded, the continued run uses taic.toml's value")
    return out


def describe(run_id: str) -> dict:
    """``GET /api/runs/<run>/control``: what a click on 'add budget' would do, and why not when it cannot."""
    run_dir = _run_dir(run_id)
    info = resume.inspect_run(run_dir)
    live = bool(info["can_topup_live"])
    can_resume = bool(info["resumable"])
    reason = "" if (live or can_resume) else (info["live_reason"] or info["reason"] or "this run cannot be continued")
    hist = info["history"]
    return {
        "run": run_id,
        "kind": info["kind"],
        "stop_reason": info["stop_reason"],
        "can_topup_live": live,
        "can_resume": can_resume,
        "can_add_budget": live or can_resume,
        "action": "topup_live" if live else "resume" if can_resume else None,
        "reason_if_not": reason,
        "live_reason": info["live_reason"],
        "budget": {"current_usd": info["budget_usd"], "initial_usd": hist["initial_budget_usd"], "spent_usd": info["spent_usd"], "needs_budget": info["needs_budget"]},
        "best_speedup": info["best_speedup"],
        "history": {"topped_up": hist["topped_up"], "events": hist["events"], "applied": hist["applied"], "pending": hist["pending"]},
        "suggested_amounts": info["suggested_amounts"],
        "max_topup_usd": max_topup_usd(),
        "dry_run": dry_run(),
        "warnings": _warnings(run_dir, info),
        "pid": info["pid"],
    }


# ---------------------------------------------------------------------------------------------
# asking for budget
# ---------------------------------------------------------------------------------------------


def _number(value, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise Refused(400, f"{name} must be a number")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise Refused(400, f"{name} must be a number") from None
    if not math.isfinite(v) or v <= 0:
        raise Refused(400, f"{name} must be a positive number")
    return v


def _amount_and_total(body: dict, current: float | None, spent: float) -> tuple[float, float]:
    """(amount to add, new total budget) from ``amount_usd`` or ``new_budget_usd``; the increase is capped, the total must exceed what is spent."""
    if not isinstance(body, dict):
        raise Refused(400, "the body must be a JSON object")
    has_amount, has_total = body.get("amount_usd") is not None, body.get("new_budget_usd") is not None
    if has_amount == has_total:
        raise Refused(400, "give exactly one of amount_usd (USD to add) or new_budget_usd (the new total)")
    if has_amount:
        amount = _number(body["amount_usd"], "amount_usd")
        if current is None:
            raise Refused(409, "the run does not record its budget: send new_budget_usd")
        total = current + amount
    else:
        total = _number(body["new_budget_usd"], "new_budget_usd")
        amount = total - (current if current is not None else spent)
    cap = max_topup_usd()
    if amount > cap + 1e-9:
        raise Refused(400, f"refused: adding ${amount:.2f} exceeds the cap of ${cap:.2f} per request (OBSERVE_MAX_TOPUP_USD)")
    if current is not None and total <= current + 1e-9:
        raise Refused(400, f"the new budget ${total:.2f} must exceed the current ${current:.2f}")
    if total <= spent + 1e-9:
        raise Refused(400, f"the new budget ${total:.2f} must exceed what is already spent (${spent:.4f})")
    return amount, total


def promote_harness(run_id: str, proposal_key: str, *, root: Path | None = None) -> dict:
    """Write one reviewed proposal into that hardware's experience file. Same step as ``taic harness apply --confirm``."""
    from taic.harness import ProposalError, apply_patch, board

    key = str(proposal_key or "").strip()
    if not key:
        raise Refused(400, "缺少提案")
    run_dir = _run_dir(run_id)
    doc = board(run_dir)
    row = next((item for item in doc["proposals"] if item.get("proposal_key") == key and item.get("status") == "valid"), None)
    if row is None:
        raise Refused(404, "没有这条有效提案")
    if not any(item.get("proposal_key") == key and item.get("useful") is True for item in doc["reviews"]):
        raise Refused(400, "这条还没有被独立 review 为有用，不能固化")
    if any(item.get("event") == "promotion" and item.get("proposal_key") == key for item in doc["decisions"]):
        raise Refused(409, "这条已经固化到本地")
    scope = (row.get("proposal") or {}).get("scope") if isinstance(row.get("proposal"), dict) else {}
    target_id = str((scope or {}).get("target_id") or "").strip()
    if not target_id:
        raise Refused(400, "提案没有目标硬件")
    with _lock_for(run_id):
        try:
            return apply_patch(run_dir, target_id=target_id, proposal_key=key, root=root, applied_by="dashboard")
        except ProposalError as exc:
            raise Refused(400, str(exc)) from exc


def convert_harness(run_id: str, seq: int, digest: str, body: dict) -> dict:
    """Human-confirmed conversion of one free-text suggestion into a structured proposal.

    The human click is the gate. The conversion itself is local and deterministic; it does
    not call an LLM. ``mode=ai`` only records that an AI filled the fields; the person still
    confirms the write.
    """
    from taic.harness import ProposalError, convert_feedback, free_text_suggestions

    run_dir = _run_dir(run_id)
    if body.get("confirm") is not True:
        raise Refused(400, "conversion needs explicit human confirmation")
    suggestions = free_text_suggestions(run_dir)
    source = next((row for row in suggestions if int(row.get("seq") or 0) == int(seq) and str(row.get("digest") or "") == str(digest)), None)
    if source is None:
        raise Refused(404, "no unconverted free-text suggestion with that seq/digest")
    run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) if (run_dir / "run_config.json").is_file() else {}
    scope = run_cfg.get("target") if isinstance(run_cfg.get("target"), dict) else {}
    target_id = str(body.get("target_id") or scope.get("id") or "apple-m4-max")
    task = str(body.get("task") or run_cfg.get("task") or "")
    dtype = str(body.get("dtype") or run_cfg.get("dtype") or "")
    if not task:
        raise Refused(400, "run has no task; cannot convert")
    mode = str(body.get("mode") or "manual")
    if mode not in ("manual", "ai"):
        raise Refused(400, "mode must be manual or ai")
    try:
        return convert_feedback(
            run_dir, seq=seq, digest=digest, target_id=target_id, task=task, dtype=dtype,
            surface=str(body.get("surface") or ""), action=str(body.get("action") or ""),
            probe=str(body.get("probe") or ""), measurement=str(body.get("measurement") or ""),
            rationale=str(body.get("rationale") or ""), by=str(body.get("by") or "dashboard"),
            confirm=True, mode=mode, supersede=str(body.get("supersede") or ""),
        )
    except ProposalError as exc:
        raise Refused(400, str(exc)) from exc


def review_harness(run_id: str, proposal_key: str, body: dict) -> dict:
    """Run one explicit independent review for a structured (usually converted) proposal.

    This spends a model call only after a person clicks the button; it never reviews free
    text implicitly. The existing review budget and trust semantics are unchanged.
    """
    from taic import trust as trust_mod
    from taic.committees import load_committees
    from taic.kernel_agent import Prices
    from taic.llm import load_config, load_raw_config
    from taic import review as review_mod
    from taic import harness

    run_dir = _run_dir(run_id)
    proposals = [row for row in harness.board(run_dir)["proposals"] if row.get("status") == "valid" and row.get("proposal_key") == proposal_key]
    if not proposals:
        raise Refused(404, "no valid proposal with that key")
    proposal = proposals[0]
    if any(row.get("event") == "review" and row.get("proposal_key") == proposal_key for row in harness.board(run_dir)["reviews"]):
        raise Refused(409, "this proposal already has a review")
    run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) if (run_dir / "run_config.json").is_file() else {}
    recorded = run_cfg.get("committees") if isinstance(run_cfg.get("committees"), list) else []
    provider, model = str(body.get("provider") or ""), str(body.get("model") or "")
    if not provider or not model:
        proposal_committee = harness.proposal_committee(proposal)
        peers = [c for c in recorded if isinstance(c, dict) and c.get("name") != proposal_committee]
        members = (peers[0].get("members") if peers else None) or []
        if not members or not isinstance(members[0], dict):
            raw = load_raw_config(None, "base")
            cfg, _ = load_config(None, preset="base")
            configured = load_committees(raw, active=cfg)
            peers = [c for c in configured if c.name != proposal_committee] or configured
            members = [{"provider": m.provider, "model": m.model} for c in peers for m in c.members]
        if members and isinstance(members[0], dict):
            independent = [
                member for member in members
                if isinstance(member, dict)
                and not harness.reviewer_conflicts_with_group(proposals, str(member.get("provider") or ""), str(member.get("model") or ""))
            ]
            members = independent or members
        if not members or not isinstance(members[0], dict):
            raise Refused(400, "cannot choose a reviewer; pass provider and model")
        provider, model = str(members[0].get("provider") or ""), str(members[0].get("model") or "")
    raw = load_raw_config(None, "base")
    cfg, _ = load_config(None, preset="base")
    committees = load_committees(raw, active=cfg)
    member = next((m for c in committees for m in c.members if m.provider == provider and m.model == model), None)
    if member is None:
        raise Refused(400, f"reviewer {provider}/{model} is not in the configured committees")
    if harness.reviewer_conflicts_with_group(proposals, provider, model):
        raise Refused(400, f"reviewer {provider}/{model} is an author of this claim; choose an independent model")
    prices = Prices(member.cfg.price_in, member.cfg.price_out, member.cfg.price_cache_read, member.cfg.price_cache_write)
    work = run_dir / "harness_reviews" / proposal_key
    bundle = review_mod.evidence_bundle(run_dir, proposal)
    prompt = review_mod.build_review_prompt(proposal.get("proposal") or {}, run_id=str(proposal.get("run_id") or run_dir.name), bundle=bundle)
    plan = review_mod.estimate_review_budget(prices, prompt)
    try:
        verdict = review_mod.run_review(
            work, prompt=prompt, provider=provider, model=model,
            effort=member.cfg.reasoning_effort, prices=prices,
        )
    except review_mod.ReviewError as exc:
        raise Refused(502, str(exc)) from exc
    review_mod.append_review(run_dir, proposal, verdict)
    settled = harness.settle(
        run_dir, target_id=str(proposal.get("target_id") or ""), task=str(proposal.get("task") or ""),
        protocol_version=str(run_cfg.get("protocol_version") or trust_mod.current_protocol_version()),
        enabled=True, shadow=True,
    )
    return {"review": verdict, "plan": plan, "settle": settled}


def review_feedback(run_id: str, seq: int, digest: str, body: dict) -> dict:
    """Record one claim review and refresh the index before returning the updated claim."""
    from observe.feedback import apply_review
    from observe.index import get_index

    if not isinstance(body, dict):
        raise Refused(400, "the body must be an object")
    if not isinstance(digest, str) or not digest.strip():
        raise Refused(400, "the claim digest is empty")
    run_dir = _run_dir(run_id)
    with _lock_for(run_id):
        try:
            stored = apply_review(run_dir, int(seq), digest, body)
        except ValueError as exc:
            raise Refused(400, str(exc)) from exc
        index = get_index()
        index.refresh_feedback(run_id, run_dir)
        found = index.conn.execute(
            "SELECT 1 FROM feedback_claims WHERE run_id = ? AND seq = ? AND digest = ?",
            (run_id, int(seq), digest),
        ).fetchone()
    if found is None:
        raise Refused(404, "no such feedback claim")
    return stored


def review_feedback_batch(run_id: str, rows: list[dict], *, dry_run: bool = False, method: str | None = None, batch_id: str = "") -> dict:
    """Preview or atomically write a batch of claim reviews.

    Every item is validated and its ``expected_status`` is checked before the
    review file changes. A dry run reads and validates only.
    """
    from observe.feedback import apply_reviews, preview_reviews
    from observe.index import get_index

    if not isinstance(rows, list) or not rows:
        raise Refused(400, "the batch must be a non-empty JSON array")
    if method and method not in ("human", "agent", "script", "api"):
        raise Refused(400, "method must be human, agent, script or api")
    run_dir = _run_dir(run_id)
    with _lock_for(run_id):
        planned = preview_reviews(run_dir, rows)
        index = get_index()
        for item in planned:
            found = index.conn.execute(
                "SELECT 1 FROM feedback_claims WHERE run_id = ? AND seq = ? AND digest = ?",
                (run_id, int(item["seq"]), str(item["digest"])),
            ).fetchone()
            if found is None:
                raise Refused(404, f"no such feedback claim: {run_id}#{item['seq']}#{item['digest']}")
        if dry_run:
            return {"dry_run": True, "items": planned}
        try:
            stored = apply_reviews(run_dir, rows, method=method, batch_id=batch_id)
        except ValueError as exc:
            raise Refused(400, str(exc)) from exc
        index.refresh_feedback(run_id, run_dir)
    return {"dry_run": False, "items": stored, "batch_id": str(batch_id or (stored[0].get("batch_id") if stored else ""))}


def _lock_for(run_id: str) -> threading.Lock:
    with _guard:
        return _locks.setdefault(run_id, threading.Lock())


def _note_manifest_budget(run_dir: Path, total: float) -> None:
    """A raised budget is an extension on the experiment manifest. Frozen conditions stay put."""
    from experiments.manifest import ManifestError, extend_budget

    current = run_dir.resolve()
    for _ in range(8):
        if current.parent == current:
            return
        current = current.parent
        if not (current / "experiment.json").is_file():
            continue
        try:
            extend_budget(current, run=run_dir.resolve().relative_to(current).as_posix(), new_cap=total)
        except (ManifestError, ValueError, OSError):
            return
        return


def _resume_command(run_dir: Path, total: float) -> list[str]:
    uv = shutil.which("uv")
    base = [uv, "run", "python"] if uv else [sys.executable]
    return [*base, "-m", "taic.cli", "resume", str(run_dir), "--budget", f"{total:.6f}", "--who", "dashboard"]


def _next_log(run_dir: Path) -> Path:
    n = 1
    while (run_dir / f"resume_{n}.log").exists():
        n += 1
    return run_dir / f"resume_{n}.log"


def _enforce_experiment_cap(run_dir: Path, total: float) -> None:
    """Refuse a top-up that would reserve more than the experiment's cap.json allows. No cap file means no extra limit."""
    from experiments.common import cap_allows

    ok, why = cap_allows(run_dir, total)
    if not ok:
        raise Refused(400, why)


def add_budget(run_id: str, body: dict) -> dict:
    """``POST /api/runs/<run>/budget`` after the origin and token checks. Returns what was done (or, in dry-run mode, what would be)."""
    run_dir = _run_dir(run_id)
    with _lock_for(run_id):
        info = resume.inspect_run(run_dir)
        spent = float(info["spent_usd"] or 0.0)
        current = float(info["budget_usd"]) if info["budget_usd"] is not None else None
        if info["kind"] == "resuming":
            raise Refused(409, info["reason"])
        if info["kind"] == "running":
            if not info["can_topup_live"]:
                raise Refused(409, info["live_reason"] or info["reason"])
            pending = info["history"]["pending"]
            if pending:
                raise Refused(409, "a previous top-up of this run has not been read by the run yet; wait a moment")
            amount, total = _amount_and_total(body, current, spent)
            _enforce_experiment_cap(run_dir, total)
            if dry_run():
                ev = control.record_event(run_dir, "dryrun", current, total, who="dashboard", note="would write control.json", action="topup_live")
                return {"ok": True, "dry_run": True, "action": "topup_live", "old_budget_usd": current, "new_budget_usd": total, "amount_usd": amount, "event": ev}
            doc = control.write_control(run_dir, total, who="dashboard", amount_usd=amount, old_budget_usd=current)
            _note_manifest_budget(run_dir, total)
            return {"ok": True, "dry_run": False, "action": "topup_live", "old_budget_usd": current, "new_budget_usd": total, "amount_usd": amount, "control": doc}
        if not info["resumable"]:
            raise Refused(409, info["reason"] or "this run cannot be continued")
        amount, total = _amount_and_total(body, current, spent)
        _enforce_experiment_cap(run_dir, total)
        cmd = _resume_command(run_dir, total)
        if dry_run():
            ev = control.record_event(run_dir, "dryrun", current, total, who="dashboard", note="would start: " + " ".join(cmd), action="resume", command=cmd)
            return {"ok": True, "dry_run": True, "action": "resume", "old_budget_usd": current, "new_budget_usd": total, "amount_usd": amount, "command": cmd, "event": ev}
        control.record_event(run_dir, "requested", current, total, who="dashboard", note="resume requested", action="resume")
        _note_manifest_budget(run_dir, total)
        log = _next_log(run_dir)
        with open(log, "ab") as out:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        # claim the run for the child at once, so that a second click before it has started is refused (the child, or the python that ``uv run`` starts
        # for it, recognises this lock as its own: ``starting`` and its parent's pid)
        try:
            control.write_json_atomic(control.lock_path(run_dir), {"pid": proc.pid, "started_at": time.time(), "who": "dashboard", "starting": True})
        except Exception:  # noqa: BLE001 - the child takes the lock itself
            pass
        return {"ok": True, "dry_run": False, "action": "resume", "old_budget_usd": current, "new_budget_usd": total, "amount_usd": amount, "pid": proc.pid, "log": log.name, "command": cmd}


# 一键重置：每个能继续的搜索在现有预算上加这么多，并把轮数上限和耐心次数抬到下面两个数。次数上限实际上不再拦住搜索，钱花完仍会停。
RESET_ADD_USD = 15.0
RESET_MAX_ROUNDS = 100_000
RESET_PATIENCE = 100_000


def _reset_plan_for(run_id: str) -> dict:
    """What a reset would do for one search. Does not write."""
    run_dir = _run_dir(run_id)
    info = resume.inspect_run(run_dir, allow_count_stops=True)
    current = float(info["budget_usd"]) if info["budget_usd"] is not None else None
    spent = float(info["spent_usd"] or 0.0)
    item = {"id": run_id, "kind": info["kind"], "stop_reason": info["stop_reason"], "old_budget_usd": current, "action": "skip", "reason": "", "new_budget_usd": None}
    if current is None:
        item["reason"] = "没有记下预算"
        return item
    # 已经超支时，在现有预算上加 15 仍可能低于已花费。取预算和已花费里较高的那个再加 15，这样一定多出 15 美元可用。
    item["new_budget_usd"] = max(current, spent) + RESET_ADD_USD
    if info["kind"] == "resuming":
        item["reason"] = info["reason"] or "正在被另一个进程续跑"
        return item
    if info["kind"] == "running":
        if not info["can_topup_live"]:
            item["reason"] = info["live_reason"] or "运行中的进程不读控制文件"
            return item
        if info["history"]["pending"]:
            item["reason"] = "上一笔加预算还没被这次搜索读到"
            return item
        item["action"] = "topup_live"
        return item
    if info["resumable"]:
        item["action"] = "resume"
        return item
    item["new_budget_usd"] = None
    item["reason"] = info["reason"] or "不能重置"
    return item


def reset_preview() -> dict:
    """``GET /api/budget/reset``: which searches would be reset, and which would be skipped. Writes nothing."""
    rows = [_reset_plan_for(row["id"]) for row in query.history()]
    return {
        "add_usd": RESET_ADD_USD,
        "max_rounds": RESET_MAX_ROUNDS,
        "patience": RESET_PATIENCE,
        "dry_run": dry_run(),
        "runs": [row for row in rows if row["action"] != "skip"],
        "skipped": [row for row in rows if row["action"] == "skip"],
    }


def _raise_experiment_caps(rows: list[dict]) -> list[dict]:
    """Raise each experiment cap by the extra dollars this reset will reserve, so the add is not refused.

    A finished run is reserved at what it spent, while the reset reserves its new budget. The difference can be more than $15."""
    from experiments.common import find_cap, run_commitment, write_cap

    grouped: dict[Path, dict] = {}
    for row in rows:
        run_dir = _run_dir(row["id"])
        found = find_cap(run_dir)
        if found is None:
            continue
        exp_dir, cap = found
        this = run_commitment(run_dir)
        extra = max(0.0, max(this, float(row["new_budget_usd"])) - this)
        slot = grouped.setdefault(exp_dir, {"old": cap, "add": 0.0, "n": 0})
        slot["add"] += extra
        slot["n"] += 1
    out = []
    for exp_dir, slot in grouped.items():
        new = slot["old"] + slot["add"]
        if not dry_run() and slot["add"] > 0:
            write_cap(exp_dir, new)
        out.append({"experiment": str(exp_dir), "old_cap_usd": slot["old"], "new_cap_usd": new, "n_runs": slot["n"], "added_usd": slot["add"]})
    return out


def _reset_resume_command(run_dir: Path, total: float) -> list[str]:
    cmd = _resume_command(run_dir, total)
    return [*cmd[:-2], "--max-rounds", str(RESET_MAX_ROUNDS), "--patience", str(RESET_PATIENCE), "--lift-count-limits", *cmd[-2:]]


def _apply_reset(item: dict) -> dict:
    try:
        return _apply_reset_unchecked(item)
    except Refused as e:
        return {**item, "action": "skip", "reason": e.message, "new_budget_usd": None}


def _apply_reset_unchecked(item: dict) -> dict:
    run_dir = _run_dir(item["id"])
    current, total = item["old_budget_usd"], item["new_budget_usd"]
    note = "reset: add $15 and lift the round limit and patience"
    with _lock_for(item["id"]):
        if item["action"] == "topup_live":
            if not dry_run():
                _enforce_experiment_cap(run_dir, total)
            if dry_run():
                ev = control.record_event(run_dir, "dryrun", current, total, who="dashboard", note="would write control.json; " + note, action="reset")
                return {**item, "dry_run": True, "event": ev}
            doc = control.write_control(
                run_dir, total, who="dashboard", amount_usd=total - current, old_budget_usd=current, note=note,
                max_rounds=RESET_MAX_ROUNDS, patience=RESET_PATIENCE,
            )
            _note_manifest_budget(run_dir, total)
            return {**item, "dry_run": False, "control": doc}
        cmd = _reset_resume_command(run_dir, total)
        if not dry_run():
            _enforce_experiment_cap(run_dir, total)
        if dry_run():
            ev = control.record_event(run_dir, "dryrun", current, total, who="dashboard", note="would start: " + " ".join(cmd), action="reset", command=cmd)
            return {**item, "dry_run": True, "command": cmd, "event": ev}
        control.record_event(run_dir, "requested", current, total, who="dashboard", note=note, action="reset")
        _note_manifest_budget(run_dir, total)
        log = _next_log(run_dir)
        with open(log, "ab") as out:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            control.write_json_atomic(control.lock_path(run_dir), {"pid": proc.pid, "started_at": time.time(), "who": "dashboard", "starting": True})
        except Exception:  # noqa: BLE001 - the child takes the lock itself
            pass
        return {**item, "dry_run": False, "pid": proc.pid, "log": log.name, "command": cmd}


def reset_budgets() -> dict:
    """``POST /api/budget/reset``. Adds $15 to every search that can continue and lifts its round limit and patience."""
    plan = reset_preview()
    caps = _raise_experiment_caps(plan["runs"])
    results = [_apply_reset(row) if row["action"] != "skip" else row for row in plan["runs"] + plan["skipped"]]
    # preview split them; apply keeps skips unchanged and replaces the ones it carried out
    done = [row for row in results if row["action"] != "skip"]
    skipped = [row for row in results if row["action"] == "skip"]
    return {"ok": True, "add_usd": RESET_ADD_USD, "max_rounds": RESET_MAX_ROUNDS, "patience": RESET_PATIENCE, "dry_run": dry_run(), "caps": caps, "runs": done, "skipped": skipped}


def _parse_ps(text: str) -> list[tuple[int, str]]:
    """``(pid, command)`` lines from ``ps -ax -o pid=,command=`` that are a search or an experiment script of this repo."""
    found = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if _is_search_command(cmd):
            found.append((pid, cmd.strip()))
    return found


def _is_search_command(cmd: str) -> bool:
    """A process this button may stop. The dashboard itself is not one of them."""
    if "-m observe" in cmd or cmd.rstrip().endswith("observe"):
        return False
    return "-m taic.cli" in cmd or "-m experiments." in cmd


def _ps_commands() -> list[tuple[int, str]]:
    try:
        text = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    return _parse_ps(text)


def _stop_groups() -> tuple[list[dict], list[dict]]:
    """Process groups to signal, and commands that were seen but refused. One group is signaled once even if several searches share it."""
    labels: dict[int, list[str]] = {}
    commands: dict[int, str] = {}
    for row in query.history():
        if row.get("status") != "running":
            continue
        directory = query.resolve_run(row["id"])
        state = detail._load(directory / "state.json") if directory is not None else None
        pid = (state or {}).get("pid")
        if not query.pid_alive(pid):
            continue
        labels.setdefault(int(pid), []).append(row["id"])
    for job in query.experiment_jobs():
        if job.get("status") != "running":
            continue
        directory = detail.resolve_run(job["id"])
        state = detail._load(directory / "state.json") if directory is not None else None
        pid = (state or {}).get("pid")
        if not query.pid_alive(pid):
            continue
        labels.setdefault(int(pid), []).append(f"{job['id']}（{job.get('script') or '队列'}）")
    for pid, cmd in _ps_commands():
        if query.pid_alive(pid):
            commands[pid] = cmd
            labels.setdefault(pid, [])
    own_pgid = os.getpgrp()
    groups: dict[int, dict] = {}
    refused: list[dict] = []
    for pid, names in labels.items():
        if pid <= 1 or pid in (os.getpid(), os.getppid()):
            refused.append({"pid": pid, "ids": names, "reason": "这是看板自己的进程"})
            continue
        try:
            pgid = os.getpgid(pid)
        except OSError:
            refused.append({"pid": pid, "ids": names, "reason": "进程已经不在"})
            continue
        if pgid == own_pgid:
            refused.append({"pid": pid, "ids": names, "reason": "和看板在同一个进程组"})
            continue
        cmd = commands.get(pid) or _command_of(pid)
        if not _is_search_command(cmd):
            refused.append({"pid": pid, "ids": names, "reason": "不是本仓库的搜索进程", "command": cmd})
            continue
        slot = groups.setdefault(pgid, {"pgid": pgid, "pids": [], "ids": [], "command": "", "roots": []})
        slot["pids"].append(pid)
        for name in names:
            if name not in slot["ids"]:
                slot["ids"].append(name)
            root = _root_of(name)
            if root and root not in slot["roots"]:
                slot["roots"].append(root)
        if cmd and not slot["command"]:
            slot["command"] = cmd
    _attach_related(list(groups.values()), own_pgid)
    return list(groups.values()), refused


def _root_of(name: str) -> str:
    """Absolute run directory for a history id. Experiment rows append ``（脚本）`` after the id."""
    run_id = name.split("（", 1)[0]
    directory = query.resolve_run(run_id)
    if directory is None:
        directory = detail.resolve_run(run_id)
    if directory is None:
        return ""
    try:
        return str(directory.resolve())
    except OSError:
        return str(directory)


def _process_table() -> list[dict]:
    """Every process: pid, parent, process group, command. Empty when ``ps`` cannot be read."""
    try:
        text = subprocess.check_output(["ps", "-ax", "-o", "pid=,ppid=,pgid=,command="], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid, ppid, pgid = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        rows.append({"pid": pid, "ppid": ppid, "pgid": pgid, "command": parts[3] if len(parts) > 3 else ""})
    return rows


def _descendants(root: int, by_parent: dict[int, list[int]]) -> set[int]:
    found: set[int] = set()
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in by_parent.get(parent, []):
            if child in found or child == root:
                continue
            found.add(child)
            stack.append(child)
    return found


def _attach_related(groups: list[dict], own_pgid: int) -> None:
    """Add the process groups of children and of processes that name this run.

    ``hmz exec`` is started with a new session, so it is not in the search's group. A child that has
    already been reparented still names ``agent_work`` under the run directory in its command.
    """
    table = _process_table()
    by_parent: dict[int, list[int]] = {}
    for row in table:
        by_parent.setdefault(row["ppid"], []).append(row["pid"])
    for group in groups:
        pids = set(group["pids"])
        for pid in list(pids):
            pids |= _descendants(pid, by_parent)
        roots = [root for root in group.get("roots", []) if root]
        if roots:
            for row in table:
                if any(root in row["command"] for root in roots):
                    pids.add(row["pid"])
                    pids |= _descendants(row["pid"], by_parent)
        pgids = []
        seen = {int(group["pgid"])}
        by_pid = {row["pid"]: row for row in table}
        for pid in pids:
            row = by_pid.get(pid)
            pgid = row["pgid"] if row else None
            if pgid is None:
                try:
                    pgid = os.getpgid(pid)
                except OSError:
                    continue
            if pgid <= 1 or pgid == own_pgid or pgid in seen:
                continue
            seen.add(pgid)
            pgids.append(pgid)
        group["related_pgids"] = pgids


def _command_of(pid: int) -> str:
    try:
        text = subprocess.check_output(["ps", "-o", "command=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return text.strip()


def stop_preview() -> dict:
    """``GET /api/stop``: which live searches would be stopped. Writes nothing and signals nothing."""
    groups, refused = _stop_groups()
    return {"groups": groups, "refused": refused, "dry_run": dry_run()}


def _signal_group(pgid: int) -> str | None:
    """SIGTERM one process group. ``None`` when the signal was delivered."""
    try:
        os.kill(-int(pgid), signal.SIGTERM)
    except ProcessLookupError:
        return None
    except OSError as e:
        return str(e)
    return None


def stop_runs() -> dict:
    """``POST /api/stop``. SIGTERM each live search, its process group, and the groups of processes it started.

    ``hmz`` and ``pi`` run in their own session. Signaling only the search group leaves them calling the model.
    """
    groups, refused = _stop_groups()
    stopped = []
    for group in groups:
        search_pgid = int(group["pgid"])
        # 先停 hmz / pi 那一组，再停搜索自己。反过来的话，搜索先死，hmz 变成孤儿并继续调模型。
        pgids = [int(item) for item in group.get("related_pgids") or [] if int(item) != search_pgid]
        pgids.append(search_pgid)
        if dry_run():
            stopped.append({**group, "pgids": pgids, "signaled": False, "dry_run": True})
            continue
        sent = []
        for pgid in pgids:
            error = _signal_group(pgid)
            if error:
                reason = f"信号没有送到：{error}" if pgid == search_pgid else f"关联进程组的信号没有送到：{error}"
                refused.append({**group, "pgid": pgid, "reason": reason})
                continue
            sent.append(pgid)
        if sent:
            stopped.append({**group, "pgids": sent, "signaled": True, "dry_run": False})
    return {"ok": True, "dry_run": dry_run(), "stopped": stopped, "refused": refused}


def parse_body(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8") or "null")
    except (UnicodeDecodeError, ValueError):
        raise Refused(400, "the body is not valid JSON") from None
