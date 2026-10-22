"""Version boundary for a worker request and an EvalReport.

Version 0 is a historical document with no ``schema_version`` key. A newer version than this process knows, or a field this version does not list, is refused. Dropping that field and still building a report is how an older parent mis-reads a newer file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPORT_SCHEMA_VERSION = 2
# Measurement protocol, separate from the file schema. Bump when the timing window,
# confirmation batches or target limits change, so aggregates can group by epoch.
PROTOCOL_VERSION = "2"
REQUEST_SCHEMA_VERSION = 4  # 4 requires target_id.

# ``eligible`` is computed from ``status`` and written for older readers. It is not a stored field.
REPORT_DERIVED = frozenset({"eligible"})
REQUEST_FIELDS = frozenset({
    "schema_version", "task", "dtype", "size", "kernel_path", "mode", "seed", "reps", "cases", "out", "heartbeat", "stage",
    "fast_bench", "require_contract_launch", "target_id",
})
REQUEST_REQUIRED = frozenset({"task", "dtype", "size", "kernel_path", "mode", "seed", "reps", "out", "target_id"})


class SchemaError(ValueError):
    """This document cannot be read by this process. The file needs human inspection."""

    def __init__(self, message: str) -> None:
        text = message.rstrip(".")
        if "needs human inspection" not in text:
            text += "; needs human inspection"
        super().__init__(text)


def document_version(doc: object, *, max_version: int, what: str) -> int:
    """The document's version. A missing ``schema_version`` is version 0. A newer version is refused."""
    if not isinstance(doc, dict):
        raise SchemaError(f"{what} is not a JSON object")
    if "schema_version" not in doc:
        return 0
    version = doc["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise SchemaError(f"{what} schema_version {version!r} is not a version number")
    if version > max_version:
        raise SchemaError(f"{what} schema_version {version} is newer than this process ({max_version})")
    return version


def migrate_report(doc: dict, known_fields: set[str]) -> dict:
    """Copy a version 0 or 1 report into constructor fields.

    Version 0 and version 1 store the same fields. The only migration is to drop the derived ``eligible`` flag.
    Any other field is refused, including on a version 0 file, so it is not silently ignored.
    """
    document_version(doc, max_version=REPORT_SCHEMA_VERSION, what="report")
    out = dict(doc)
    eligible = out.pop("eligible", None)
    unknown = set(out) - set(known_fields) - {"schema_version"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise SchemaError(f"report has fields this process does not know ({names})")
    out.pop("schema_version", None)
    missing = [name for name in ("task", "dtype", "size", "mode") if name not in out]
    if missing:
        raise SchemaError(f"report is missing {missing}")
    # 老文件有时只写了 eligible，没有 status。eligible 为真就当成当时的 ok，否则读出来会变成 pending、不能当种子。
    if "status" not in out and eligible is True:
        out["status"] = "ok"
    return out


def read_request(doc: dict) -> dict:
    """A worker request this process can run. Version 0 is the historical shape, without ``schema_version``."""
    document_version(doc, max_version=REQUEST_SCHEMA_VERSION, what="worker request")
    unknown = set(doc) - REQUEST_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise SchemaError(f"worker request has fields this process does not know ({names})")
    missing = sorted(REQUEST_REQUIRED - set(doc))
    if missing:
        raise SchemaError(f"worker request is missing {missing}")
    return doc


def write_json_atomic(path: Path, doc) -> None:
    """Write ``doc`` beside ``path`` and replace it into place. A reader sees the old file or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
