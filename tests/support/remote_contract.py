"""Locations and loader for backend-neutral external evaluator contract goldens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORT = Path(__file__).resolve().parent
GOLDEN = SUPPORT / "golden" / "remote" / "evaluator_v1.json"
GO_VERIFIER = SUPPORT / "remotecontract"


def load_evaluator_golden() -> dict[str, Any]:
    """Load the frozen cross-language contract vectors without recording them."""

    return json.loads(GOLDEN.read_text(encoding="utf-8"))
