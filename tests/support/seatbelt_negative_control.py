#!/usr/bin/env python3
"""Independent outside-read negative control for the current Seatbelt installation.

Creates and reads an artificial canary outside the attempt, then verifies that the same path is
unreadable through the production HmzFenceExecutor profile. No frozen probe is edited and no real
credential, GPU, NPU or model is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from tcenv.worker_isolation import AttemptSpec, HmzFenceExecutor, ResourceLimits, inspect_hmz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="shapeforge-seatbelt-negative-control-") as directory:
        root = Path(directory)
        outside = root / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("artificial-read-canary\n")
        negative_control = secret.read_text() == "artificial-read-canary\n"
        source = (
            "import json, os\n"
            "from pathlib import Path\n"
            "try:\n"
            "    value = Path(os.environ['PROBE_OUTSIDE']).read_text()\n"
            "    outcome = {'result': 'succeeded', 'value': value}\n"
            "except Exception as exc:\n"
            "    outcome = {'result': 'denied', 'error': type(exc).__name__}\n"
            "(Path(__file__).resolve().parents[1] / 'build' / 'artifacts' / 'result.json').write_text(json.dumps(outcome))\n"
        )
        installation = inspect_hmz()
        executor = HmzFenceExecutor(installation_loader=lambda: installation)
        result = executor.run(AttemptSpec(
            attempt_root=root / "attempt",
            argv=(sys.executable, str(root / "attempt" / "input" / "probe.py")),
            input_files={"probe.py": source.encode()},
            artifacts=("result.json",),
            extra_env={"PROBE_OUTSIDE": str(secret)},
            limits=ResourceLimits(wall_seconds=5, cpu_seconds=3),
        ))
        observed = json.loads((root / "attempt" / "build" / "artifacts" / "result.json").read_text())
        document = {
            "schema_id": "shapeforge.seatbelt-negative-read-control.1",
            "status": "passed" if negative_control and observed["result"] == "denied" else "failed",
            "negative_control_outside_read_succeeded": negative_control,
            "fenced_read": observed,
            "hmz": installation.document(),
            "publication_allowed": False,
        }
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return 0 if document["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
