#!/usr/bin/env python3
"""P2-K acceptance: portfolio-level fresh replay, validated only, never activated."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2K = Path(__file__).resolve().parent
MANIFEST = P2K / "manifest.sha256"
BASELINE = P2K / "acceptance-baseline.json"
CPU_TEST = "tests/shared/test_conv2d_portfolio_fresh.py"
GPU_TEST = "tests/backends/metal/test_conv2d_portfolio_fresh_gpu.py"
DELIVERABLES = (
    "docs/架构/conv2d-portfolio-fresh.md",
    "tests/support/conv2d_portfolio_fresh.py",
    "tests/backends/metal/conv2d_portfolio_fresh_record.py",
    "tests/shared/test_conv2d_portfolio_fresh.py",
    "tests/backends/metal/test_conv2d_portfolio_fresh_gpu.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-K/K.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-K/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-K/pack-closure.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-K/record.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-K/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-K/evidence/fresh-suite.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-K/evidence/validated-manifest.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-K/evidence/packed-half-probe.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    MANIFEST.write_text("\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n", encoding="utf-8")
    print(f"OK: wrote P2-K manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-K manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-K manifest mismatch: {name}")
    print(f"OK: P2-K manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-k-conv2d-portfolio-fresh-1", "wrong P2-K epoch")
    require(document["parent_head"] == "1365473c2f3d26ed882946a2fffce133d2128df0", "P2-K parent HEAD drifted")
    require(document["publication_allowed"] is False, "P2-K must not publish")
    closure = json.loads((P2K / "pack-closure.json").read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files), "P2-K task-pack file closure drifted")
    require(closure["sha256"] == digest == document["pack_closure_sha256"], "P2-K task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-K pack file count drifted")
    print(f"OK: P2-K baseline (79 pack files, digest={digest[:12]}…)")


def verify_history() -> None:
    for phase, count in (("P2-I", 71), ("P2-J", 22)):
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
    from tests.support.conv2d_portfolio_fresh import assert_fresh_suite, assert_packed_probe_stays_unadopted

    assert_fresh_suite()
    assert_packed_probe_stays_unadopted()
    print("OK: P2-K fresh suite, validated manifest and packed-half negative result")
    pytest_nodes([CPU_TEST], where="P2-K CPU replay", expected=2)
    pytest_nodes([GPU_TEST], where="P2-K real Metal fresh replay", expected=1)
    print("OK: P2-K acceptance (2 CPU + 1 real Metal replay; validated only, no activation/publication)")


if __name__ == "__main__":
    main()
