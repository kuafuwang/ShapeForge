"""Deployment grant files for the external evaluator.

The raw session token appears only in this operator-controlled file and in the
host token file; the job store keeps a SHA-256 lookup digest plus subject/scope.
"""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .identity import SessionGrant

SCHEMA_VERSION = 1


class GrantFileError(ValueError):
    """A deployment grant file is malformed or unreadable."""


def _rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def grant_document(grant: SessionGrant) -> dict[str, Any]:
    return {
        "token": grant.token,
        "run_id": grant.run_id,
        "session_id": grant.session_id,
        "target_id": grant.target_id,
        "kinds": sorted(grant.allowed_kinds),
        "modes": sorted(grant.allowed_modes),
        "subject": grant.subject,
        "max_timed": grant.max_timed,
        "max_gpu_s": grant.max_gpu_s,
        "expires_at": None if grant.expires_at is None else _rfc3339(datetime.fromtimestamp(grant.expires_at, timezone.utc)),
    }


def parse_grant(document: object) -> SessionGrant:
    if not isinstance(document, Mapping):
        raise GrantFileError("each grant must be an object")
    allowed = {"token", "run_id", "session_id", "target_id", "kinds", "modes", "subject", "max_timed", "max_gpu_s", "expires_at"}
    unknown = set(document) - allowed
    if unknown:
        raise GrantFileError(f"unknown grant fields: {sorted(unknown)}")
    for name in ("token", "run_id", "session_id", "target_id"):
        value = document.get(name)
        if not isinstance(value, str) or not value:
            raise GrantFileError(f"grant {name} is required")
    kinds = document.get("kinds", ["formal_eval", "selfcheck"])
    modes = document.get("modes", [])
    if not isinstance(kinds, list) or not kinds or any(not isinstance(kind, str) for kind in kinds):
        raise GrantFileError("grant kinds must be a non-empty list of strings")
    if not isinstance(modes, list) or any(not isinstance(mode, str) for mode in modes):
        raise GrantFileError("grant modes must be a list of strings")
    expires_at = document.get("expires_at")
    expires_seconds: float | None
    if expires_at is None:
        expires_seconds = None
    elif isinstance(expires_at, str):
        try:
            expires_seconds = datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        except ValueError as exc:
            raise GrantFileError("grant expires_at must be UTC RFC 3339 with second precision") from exc
    else:
        raise GrantFileError("grant expires_at must be a timestamp string or null")
    try:
        return SessionGrant(
            token=str(document["token"]),
            run_id=str(document["run_id"]),
            session_id=str(document["session_id"]),
            target_id=str(document["target_id"]),
            allowed_kinds=frozenset(kinds),
            allowed_modes=frozenset(modes),
            subject=str(document.get("subject") or ""),
            max_timed=int(document.get("max_timed", 20)),
            max_gpu_s=float(document.get("max_gpu_s", 120.0)),
            expires_at=expires_seconds,
        )
    except (TypeError, ValueError) as exc:
        raise GrantFileError(f"grant is invalid: {exc}") from exc


def parse_grants(document: object) -> list[SessionGrant]:
    if not isinstance(document, Mapping) or set(document) != {"schema_version", "grants"}:
        raise GrantFileError("grant file must contain schema_version and grants")
    if document["schema_version"] != SCHEMA_VERSION:
        raise GrantFileError("grant file has an unsupported schema")
    if not isinstance(document["grants"], list) or not document["grants"]:
        raise GrantFileError("grant file must contain at least one grant")
    grants = [parse_grant(entry) for entry in document["grants"]]
    tokens = [grant.token for grant in grants]
    if len(set(tokens)) != len(tokens):
        raise GrantFileError("grant tokens must be unique")
    return grants


def load_grants_file(path: Path | str) -> list[SessionGrant]:
    path = Path(path)
    try:
        info = path.stat()
    except OSError as exc:
        raise GrantFileError(f"cannot read grant file: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise GrantFileError("grant file must be a regular file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise GrantFileError("grant file must not be readable by group or others (mode 0600)")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GrantFileError(f"cannot read grant file: {exc}") from exc
    return parse_grants(document)


def dump_grants(grants: list[SessionGrant]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "grants": [grant_document(grant) for grant in grants]}


def write_grants_file(path: Path | str, grants: list[SessionGrant]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(dump_grants(grants), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path
