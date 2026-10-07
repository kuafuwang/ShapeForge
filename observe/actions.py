"""The dashboard's one write path: adding budget to a run. Everything else in ``observe`` stays read-only.

What a click can do (``POST /api/runs/<run>/budget``):

* a run that is still running and polls its control file (``state.json`` says ``control_supported``): write ``control.json`` (live top-up);
* a run that stopped for budget, or whose process died: start ``python -m taic.cli resume`` for it as a detached process;
* anything else: refused with the reason (a run that converged, hit its round limit or a GPU hang is not continued; there is no override).

This spends real money and can start GPU work, so every POST needs: a loopback ``Host``, a same-origin ``Origin`` (when the browser sends one), the
per-process random token that the served page carries (``X-Observe-Token``), a positive amount under ``OBSERVE_MAX_TOPUP_USD`` (default 25), a run
that resolves inside ``runs/`` (no ``..``, no symlink out) and at most one request in flight per run. ``OBSERVE_CONTROL_DRYRUN=1`` does all of the
checks, records the command that *would* run in ``budget_events.json`` and starts nothing.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from observe import scan
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
    path = scan.resolve_run(run_id)
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


def _lock_for(run_id: str) -> threading.Lock:
    with _guard:
        return _locks.setdefault(run_id, threading.Lock())


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


def parse_body(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8") or "null")
    except (UnicodeDecodeError, ValueError):
        raise Refused(400, "the body is not valid JSON") from None
