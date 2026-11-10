"""Write fresh B-migration observations without overwriting closed phase evidence.

The archived ``B*/evidence/gpu-*.json`` files are historical acceptance records, not
pytest output directories.  A fresh Metal run gets a new directory for every call by
default.  To keep an intentional re-recording epoch, set
``SHAPEFORGE_B_EVIDENCE_RUN_DIR`` to a *new* directory outside the phase's closed
``evidence/`` directory and archive that directory separately after review.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

RUN_DIR_ENV = "SHAPEFORGE_B_EVIDENCE_RUN_DIR"


def write_gpu_observation(phase_dir: Path, task: str, record: Mapping[str, Any]) -> Path:
    """Write one fresh GPU observation and return its path.

    ``record`` is written unchanged.  The function never writes into the phase's
    closed evidence directory, including when an operator supplies a run directory.
    """
    closed = (phase_dir / "evidence").resolve()
    configured = os.environ.get(RUN_DIR_ENV)
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = Path(tempfile.mkdtemp(prefix=f"shapeforge-{phase_dir.name}-gpu-")).resolve()
    if root == closed or closed in root.parents:
        raise ValueError(f"{RUN_DIR_ENV} must be outside the closed evidence directory: {closed}")
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"gpu-{task}.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


__all__ = ["RUN_DIR_ENV", "write_gpu_observation"]
