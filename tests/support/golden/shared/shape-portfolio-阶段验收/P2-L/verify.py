#!/usr/bin/env python3
"""P2-L acceptance: reproduce the Split-K negative decision without adding a candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2L = Path(__file__).resolve().parent
MANIFEST = P2L / "manifest.sha256"
BASELINE = P2L / "acceptance-baseline.json"
CPU_TEST = "tests/shared/test_gemm_splitk_probe.py"
GPU_TEST = "tests/backends/metal/test_gemm_splitk_probe_gpu.py"
DELIVERABLES = (
    "docs/架构/gemm-splitk-headroom.md",
    "experiments/gemm_splitk_probe.py",
    "tests/support/gemm_splitk.py",
    "tests/shared/test_gemm_splitk_probe.py",
    "tests/backends/metal/test_gemm_splitk_probe_gpu.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/L.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/pack-closure.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/record.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/evidence/launch-floor.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/evidence/analysis.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    MANIFEST.write_text("\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n", encoding="utf-8")
    print(f"OK: wrote P2-L manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-L manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-L manifest mismatch: {name}")
    print(f"OK: P2-L manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-l-gemm-splitk-negative-1", "wrong P2-L epoch")
    require(document["parent_head"] == "521d9d5322058fee14478a066388be3410960b60", "P2-L parent HEAD drifted")
    require(document["publication_allowed"] is False, "P2-L must not publish")
    closure = json.loads((P2L / "pack-closure.json").read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files), "P2-L task-pack closure drifted")
    require(closure["sha256"] == digest == document["pack_closure_sha256"], "P2-L task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-L pack file count drifted")
    print(f"OK: P2-L baseline (79 pack files, digest={digest[:12]}…)")


def verify_history() -> None:
    for phase, count in (("P2-I", 71), ("P2-J", 22), ("P2-K", 13)):
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/{phase}/manifest.sha256"
        entries = [line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        require(len(entries) == count, f"{phase} historical manifest count drifted")
        for digest, name in entries:
            require(sha256(ROOT / name) == digest, f"{phase} historical file changed: {name}")
        print(f"OK: {phase} historical manifest remains byte-identical ({count} files)")


def pytest_nodes(arguments: list[str], *, where: str, expected: int) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dict(os.environ),
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"{where} failed:\n{output}")
    require(f"{expected} passed" in output, f"{where}: expected {expected} passed:\n{output}")
    require(" skipped" not in output and "xfail" not in output and "deselected" not in output, f"{where}: skip/xfail/deselect is not accepted:\n{output}")
    print(f"OK: {where} ({expected} passed)")


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
    from tests.support.gemm_splitk import assert_negative

    assert_negative()
    print("OK: P2-L negative analysis and no-implementation boundary")
    pytest_nodes([CPU_TEST], where="P2-L CPU analysis", expected=1)
    pytest_nodes([GPU_TEST], where="P2-L real Metal negative replay", expected=1)
    print("OK: P2-L acceptance (1 CPU + 1 real Metal negative replay; no candidate, no publication)")


if __name__ == "__main__":
    main()
