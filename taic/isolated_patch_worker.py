"""Trusted entry point executed inside the isolated patch-test container.

The container invokes this module from the read-only harness mount. It applies
only the supplied patch to a fresh source snapshot, then runs host-registered
checks. pytest subprocesses use the patched snapshot as PYTHONPATH; this module
and the check definitions remain on the trusted read-only mount.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m taic.isolated_patch_worker")
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict) or not isinstance(request.get("check_items"), list):
            raise ValueError("isolated worker request has no check_items")
        applied = subprocess.run(["git", "apply", "--check", str(args.patch)], cwd=args.work, capture_output=True, text=True)
        if applied.returncode != 0:
            raise RuntimeError((applied.stderr or applied.stdout or "patch does not apply")[-2000:])
        applied = subprocess.run(["git", "apply", str(args.patch)], cwd=args.work, capture_output=True, text=True)
        if applied.returncode != 0:
            raise RuntimeError((applied.stderr or applied.stdout or "patch apply failed")[-2000:])
        from .acceptance_checks import run_checks

        results = run_checks(request["check_items"], worktree=args.work, project=Path("/opt/taic"))
        sys.stdout.write(json.dumps(results, ensure_ascii=False, separators=(",", ":")) + "\n")
        return 0
    except Exception as exc:  # noqa: BLE001 - the outer runner turns nonzero into a failed verification
        print(f"isolated patch worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
