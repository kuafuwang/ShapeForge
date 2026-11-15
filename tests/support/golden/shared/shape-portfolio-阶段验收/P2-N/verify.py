#!/usr/bin/env python3
"""P2-N acceptance: preserve the small stop decision and the unmodified baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2N = Path(__file__).resolve().parent
MANIFEST = P2N / "manifest.sha256"
BASELINE = P2N / "acceptance-baseline.json"
CPU_TEST = "tests/shared/test_conv2d_small_stop.py"
DELIVERABLES = (
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-N/N.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-N/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-N/pack-closure.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-N/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-N/evidence/decision.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-N/evidence/weight-cache-report.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-N/evidence/weight-cache.metal",
    "tests/shared/test_conv2d_small_stop.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    MANIFEST.write_text("\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n", encoding="utf-8")
    print(f"OK: wrote P2-N manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-N manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-N manifest mismatch: {name}")
    print(f"OK: P2-N manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-n-conv2d-small-stop-1", "wrong P2-N epoch")
    require(document["parent_head"] == "a9c1dccce636107c2852995578b05611bf19865b", "P2-N parent HEAD drifted")
    require(document["publication_allowed"] is False, "P2-N must not publish")
    closure = json.loads((P2N / "pack-closure.json").read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files), "P2-N task-pack closure drifted")
    require(closure["sha256"] == digest == document["pack_closure_sha256"], "P2-N task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-N pack file count drifted")
    print(f"OK: P2-N baseline (79 pack files, digest={digest[:12]}…)")


def verify_history() -> None:
    for phase, count in (("P2-I", 71), ("P2-J", 22), ("P2-K", 13), ("P2-L", 12), ("P2-M", 21)):
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/{phase}/manifest.sha256"
        entries = [line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        require(len(entries) == count, f"{phase} historical manifest count drifted")
        for digest, name in entries:
            require(sha256(ROOT / name) == digest, f"{phase} historical file changed: {name}")
        print(f"OK: {phase} historical manifest remains byte-identical ({count} files)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_baseline()
    verify_history()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", CPU_TEST],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dict(os.environ),
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0 and "1 passed" in output, f"P2-N stop decision failed:\n{output}")
    print("OK: P2-N stop decision (1 CPU; no model/search/publication)")


if __name__ == "__main__":
    main()
