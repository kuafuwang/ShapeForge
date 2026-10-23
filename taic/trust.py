"""Per-target model trust grants, their append-only audit, and the checks a proposal needs.

Trust is deliberately *not* a model's self-report. A person grants it for one
(target, task, provider, model, effort) tuple, for named scopes, with the prompt and
measurement protocol it was granted against. A later prompt or protocol change makes the
grant stale instead of silently carrying it over. The registry lives with the target:

    targets/<target_id>/trust.toml
    targets/<target_id>/trust_audit.jsonl

No credentials are stored here. The file is small enough to review with the device data.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import time
import tomllib
from pathlib import Path

from tcenv.schema import PROTOCOL_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
SCOPES = frozenset({"overlay:tip", "overlay:premise", "propose:skill", "propose:task_wording"})
STATUSES = frozenset({"active", "suspended", "revoked"})
_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrustError(ValueError):
    """A grant or lookup that cannot be honoured."""


def target_root(target_id: str, root: Path | None = None) -> Path:
    """The target's directory. ``root`` is for tests and alternate repositories."""
    if not _TARGET.fullmatch(str(target_id or "")):
        raise TrustError(f"invalid target id {target_id!r}")
    base = Path(root) if root is not None else REPO_ROOT / "targets"
    return base / target_id


def trust_path(target_id: str, root: Path | None = None) -> Path:
    return target_root(target_id, root) / "trust.toml"


def audit_path(target_id: str, root: Path | None = None) -> Path:
    return target_root(target_id, root) / "trust_audit.jsonl"


_PROMPT_FILES = (
    "taic/prompts.py",
    "taic/kernel_agent.py",
    "taic/harness.py",
    "targets/apple-m4-max/experience.md",
    "taic/flows/kernel_once/skills/taic-tools/SKILL.md",
)


def current_prompt_sha1() -> str:
    """A stable fingerprint of the prompt/TASK surface. Edits make old grants stale.

    ``kernel_agent.py`` owns the TASK contract and ``harness.py`` owns HARNESS.md; the
    target experience file and the agent skill are injected as prompt text too. A trust grant
    must not silently survive a change to the text the model is asked to follow.
    """
    digest = hashlib.sha256()
    for name in _PROMPT_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((REPO_ROOT / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def current_protocol_version() -> str:
    return str(PROTOCOL_VERSION)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    # JSON basic strings use the same escaping TOML basic strings accept for our values.
    return json.dumps(str(value), ensure_ascii=False)


def _dump(grants: list[dict]) -> str:
    lines = ["version = 1", ""]
    for grant in grants:
        lines.append("[[grants]]")
        for key in (
            "id", "target_id", "task", "provider", "model", "effort", "prompt_sha1",
            "protocol_version", "status", "granted_by", "granted_at", "reason",
            "evidence_run", "expires_at", "scopes",
        ):
            if key not in grant:
                continue
            lines.append(f"{key} = {_toml_value(grant[key])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _grant_id(target_id: str, task: str, provider: str, model: str, effort: str) -> str:
    payload = "\0".join((str(target_id), str(task), str(provider), str(model), str(effort)))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_grants(target_id: str, root: Path | None = None) -> list[dict]:
    path = trust_path(target_id, root)
    if not path.is_file():
        return []
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TrustError(f"cannot read {path}: {exc}") from exc
    rows = doc.get("grants")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _write(target_id: str, grants: list[dict], root: Path | None) -> None:
    path = trust_path(target_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        tmp.write_text(_dump(grants), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _audit(target_id: str, event: dict, root: Path | None) -> None:
    path = audit_path(target_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    lock = path.with_name(".trust_audit.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as out:
                out.write(line)
                out.flush()
                os.fsync(out.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_audit(target_id: str, root: Path | None = None) -> list[dict]:
    path = audit_path(target_id, root)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def grant(
    target_id: str,
    task: str,
    provider: str,
    model: str,
    *,
    effort: str = "",
    scopes: list[str] | tuple[str, ...] = ("overlay:tip",),
    granted_by: str = "cli",
    reason: str = "",
    evidence_run: str = "",
    expires_at: float | None = None,
    prompt_sha1: str | None = None,
    protocol_version: str | None = None,
    root: Path | None = None,
) -> dict:
    """Grant or replace one (target, task, model) trust entry. A revoked entry can be re-granted."""
    target_id, task = str(target_id).strip(), str(task).strip()
    provider, model, effort = str(provider).strip(), str(model).strip(), str(effort or "").strip()
    if not target_id or not task or not provider or not model:
        raise TrustError("target, task, provider and model are required")
    clean_scopes = sorted(set(str(scope).strip() for scope in scopes if str(scope).strip()))
    if not clean_scopes or any(scope not in SCOPES for scope in clean_scopes):
        raise TrustError(f"scopes must be a non-empty subset of {sorted(SCOPES)}")
    expiry = _finite(expires_at)
    if expires_at is not None and (expiry is None or expiry <= time.time()):
        raise TrustError("expires_at must be a future finite timestamp")
    ident = _grant_id(target_id, task, provider, model, effort)
    now = time.time()
    entry = {
        "id": ident,
        "target_id": target_id,
        "task": task,
        "provider": provider,
        "model": model,
        "effort": effort,
        "prompt_sha1": str(prompt_sha1 if prompt_sha1 is not None else current_prompt_sha1()),
        "protocol_version": str(protocol_version if protocol_version is not None else current_protocol_version()),
        "status": "active",
        "granted_by": str(granted_by or "cli"),
        "granted_at": now,
        "reason": str(reason or ""),
        "evidence_run": str(evidence_run or ""),
        "expires_at": expiry if expiry is not None else 0.0,
        "scopes": clean_scopes,
    }
    grants = load_grants(target_id, root)
    replaced = None
    for index, row in enumerate(grants):
        if row.get("id") == ident:
            replaced = row
            grants[index] = entry
            break
    else:
        grants.append(entry)
    _write(target_id, grants, root)
    _audit(target_id, {"t": now, "event": "grant", "grant": entry, "previous": replaced}, root)
    return entry


def set_status(
    target_id: str,
    *,
    task: str,
    provider: str,
    model: str,
    effort: str = "",
    status: str,
    by: str = "cli",
    reason: str = "",
    root: Path | None = None,
) -> dict:
    """Suspend or revoke one entry. Returns the updated entry; raises when it does not exist."""
    if status not in ("suspended", "revoked"):
        raise TrustError("set_status only suspends or revokes; grant creates an active entry")
    ident = _grant_id(target_id, task, provider, model, effort)
    grants = load_grants(target_id, root)
    for row in grants:
        if row.get("id") != ident:
            continue
        previous = dict(row)
        row["status"] = status
        _write(target_id, grants, root)
        _audit(
            target_id,
            {"t": time.time(), "event": status, "by": by, "reason": reason, "previous": previous, "grant": row},
            root,
        )
        return dict(row)
    raise TrustError(f"no trust entry for {target_id}/{task} {provider}/{model} effort={effort!r}")


def revoke(target_id: str, **kwargs) -> dict:
    kwargs["status"] = "revoked"
    return set_status(target_id, **kwargs)


def suspend(target_id: str, **kwargs) -> dict:
    kwargs["status"] = "suspended"
    return set_status(target_id, **kwargs)


def check(
    entry: dict,
    *,
    target_id: str,
    task: str,
    provider: str,
    model: str,
    effort: str = "",
    scope: str = "overlay:tip",
    protocol_version: str | None = None,
    prompt_sha1: str | None = None,
    now: float | None = None,
) -> tuple[bool, str]:
    """Whether one grant authorises this proposal author and scope right now."""
    if not isinstance(entry, dict):
        return False, "missing trust entry"
    if entry.get("status") != "active":
        return False, f"trust is {entry.get('status') or 'not active'}"
    if entry.get("target_id") != target_id:
        return False, "trust is for another target"
    if entry.get("task") != task:
        return False, "trust is for another task"
    if (entry.get("provider"), entry.get("model"), str(entry.get("effort") or "")) != (provider, model, str(effort or "")):
        return False, "trust is for another provider/model/effort"
    if scope not in (entry.get("scopes") or []):
        return False, f"trust does not include {scope}"
    expiry = _finite(entry.get("expires_at"))
    if expiry is not None and expiry > 0 and (now if now is not None else time.time()) >= expiry:
        return False, "trust has expired"
    if str(entry.get("protocol_version") or "") != str(protocol_version if protocol_version is not None else current_protocol_version()):
        return False, "trust was granted for another measurement protocol"
    expected_prompt = str(entry.get("prompt_sha1") or "")
    if expected_prompt and expected_prompt != str(prompt_sha1 if prompt_sha1 is not None else current_prompt_sha1()):
        return False, "trust was granted for another prompt"
    return True, ""


def is_trusted(
    target_id: str,
    *,
    task: str,
    provider: str,
    model: str,
    effort: str = "",
    scope: str = "overlay:tip",
    root: Path | None = None,
    **kwargs,
) -> tuple[bool, str]:
    for entry in load_grants(target_id, root):
        ok, why = check(
            entry, target_id=target_id, task=task, provider=provider, model=model,
            effort=effort, scope=scope, **kwargs,
        )
        if ok:
            return True, ""
    return False, "no active grant"
