"""Read the live test anchors; closed verifiers and production code must not import this module.

A prompt or task-pack change updates ``live_anchors.json`` and appends an epoch.
Closed packages continue to validate against their own historical records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANCHORS = Path(__file__).with_suffix(".json")


def load_live_anchors(root: Path = ROOT) -> dict:
    document = json.loads((root / "tests/support/live_anchors.json").read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unknown live anchors schema")
    return document


def measure_pack_closure(root: Path = ROOT) -> tuple[tuple[str, ...], str]:
    from tcenv.taskcontract.pack import code_import_closure

    files = tuple(code_import_closure(root))
    digest = hashlib.sha256(b"".join((root / name).read_bytes() for name in files)).hexdigest()
    return files, digest
