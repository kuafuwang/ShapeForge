"""Versioned run files other than the transcript.

``transcript.jsonl`` is the billed-request record and is append-only. ``state.json``, ``result.json`` and ``ledger.json`` are checkpoints: they can speed up a resume, and they must not be written back over the transcript. A file that is present but damaged, or newer than this process, needs human inspection. It is not treated as work that was never done.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.schema import SchemaError, document_version

RESULT_SCHEMA_VERSION = 1
# The measurement protocol, separate from the file schema. Bump when the timing window,
# confirmation batches or target limits change, so aggregates can group by epoch.
PROTOCOL_VERSION = "2"
STATE_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1


class NeedsInspection(RuntimeError):
    """A stored report or checkpoint is present but cannot be trusted."""


def read_versioned(path: Path, max_version: int) -> dict | None:
    """The JSON object at ``path``, or ``None`` when the file is absent.

    A present file that is not a readable object of a known version raises ``NeedsInspection``.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise NeedsInspection(f"{path} is damaged and needs human inspection") from e
    try:
        document_version(doc, max_version=max_version, what=str(path))
    except SchemaError as e:
        raise NeedsInspection(str(e)) from e
    return doc
