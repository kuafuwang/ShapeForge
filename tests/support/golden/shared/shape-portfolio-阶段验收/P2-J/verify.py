#!/usr/bin/env python3
"""P2-J acceptance: small single-stage MMA, exact portfolio dispatch and fresh replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2J = Path(__file__).resolve().parent
MANIFEST = P2J / "manifest.sha256"
BASELINE = P2J / "acceptance-baseline.json"
CPU_TEST = "tests/shared/test_conv2d_small_mma.py"
LAUNCH_TEST = "tests/backends/metal/test_conv2d_small_mma_gpu.py::test_small_is_one_stage_and_paper_keeps_two_stages"
GPU_TESTS = (
    "tests/backends/metal/test_conv2d_small_mma_gpu.py::test_small_specialized_mma_passes_all_six_cases",
    "tests/backends/metal/test_conv2d_small_mma_gpu.py::test_paper_specialized_mma_still_passes_all_six_cases",
    "tests/backends/metal/test_conv2d_small_mma_gpu.py::test_full_eval_replays_both_candidates_and_the_latency_gate",
)
DELIVERABLES = (
    "docs/架构/conv2d-small-mma-headroom.md",
    "tests/support/conv2d_small_mma.py",
    "tests/backends/metal/conv2d_small_mma_record.py",
    "tests/shared/test_conv2d_small_mma.py",
    "tests/backends/metal/test_conv2d_small_mma_gpu.py",
    "tests/support/golden/metal/conv2d_small_mma/coverage.json",
    "tests/support/golden/metal/conv2d_small_mma/fallback/kernel.metal",
    "tests/support/golden/metal/conv2d_small_mma/variants/paper/kernel.metal",
    "tests/support/golden/metal/conv2d_small_mma/variants/small/kernel.metal",
    "tests/support/golden/metal/conv2d_small_mma/full-eval.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/J.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/pack-closure.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/record.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/launch-floor.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/direct-mma-report.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/initial-recompute-report.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/headroom.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/small-confirmation.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/small-run.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/paper-run.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    MANIFEST.write_text(
        "\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n",
        encoding="utf-8",
    )
    print(f"OK: wrote P2-J manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-J manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-J manifest mismatch: {name}")
    print(f"OK: P2-J manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-j-conv2d-small-mma-1", "wrong P2-J epoch")
    require(document["parent_head"] == "2462ef4e64e923cdd5653f22c0500301c036b6c6", "P2-J parent HEAD drifted")
    require(document["publication_allowed"] is False, "P2-J must not publish")
    closure = json.loads((P2J / "pack-closure.json").read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files), "P2-J task-pack file closure drifted")
    require(closure["sha256"] == digest == document["pack_closure_sha256"], "P2-J task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-J pack file count drifted")
    print(f"OK: P2-J baseline (79 pack files, digest={digest[:12]}…)")


def verify_p2i_unchanged() -> None:
    historical = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/manifest.sha256"
    entries = [line.split("  ", 1) for line in historical.read_text(encoding="utf-8").splitlines()]
    require(len(entries) == 71, "P2-I historical manifest file count drifted")
    for digest, name in entries:
        require(sha256(ROOT / name) == digest, f"P2-I historical file changed: {name}")
    print("OK: P2-I historical manifest remains byte-identical (71 files)")


def verify_evidence() -> None:
    from taic.portfolio_confirmation import verify_subject_document
    from tests.support.conv2d_small_mma import SMALL_CONFIRMATION, SMALL_SHA256, assert_recorded_portfolio

    assert_recorded_portfolio()
    subject = verify_subject_document(json.loads(SMALL_CONFIRMATION.read_text(encoding="utf-8")))
    require(subject["status"] == "confirmed", "small fresh confirmation is not confirmed")
    require(subject["identity"]["kind"] == "match", "small fresh identity mismatch")
    require(subject["candidate_sha256"] == SMALL_SHA256, "small fresh candidate sha drifted")
    require(subject["search_seed"] == 7 and subject["confirmation_seed"] != 7, "small fresh seed drifted")
    report = subject["confirmation_report"]
    require(report["status"] == "ok" and report["safety"]["ok"] is True, "small fresh correctness/safety failed")
    require(report["speedups"]["torch_mps"] > 1.10, "small fresh lost its Torch-MPS margin")
    require(report["candidate_us"] <= 1.10 * report["baseline_us"]["mlx"], "small fresh exceeds the 1.10 MLX gate")
    headroom = json.loads((P2J / "evidence/headroom.json").read_text(encoding="utf-8"))
    decision = headroom["decision"]
    require(decision["fresh_gate_passed"] is True, "headroom gate summary is stale")
    require(decision["single_stage_required"] is True, "small two-stage route was incorrectly revived")
    require(decision["two_stage_no_fill_floor_us"] > decision["fresh_candidate_us"], "two-stage floor no longer supports the decision")
    require(decision["production_speedup_published"] is False, "small probe became a publication")
    print("OK: P2-J small fresh identity, six cases, 1.10 gate and headroom decision")


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
    verify_p2i_unchanged()
    verify_evidence()
    pytest_nodes([CPU_TEST, LAUNCH_TEST], where="P2-J CPU and launch nodes", expected=6)
    pytest_nodes(list(GPU_TESTS), where="P2-J real Metal nodes", expected=3)
    print("OK: P2-J acceptance (6 CPU + 3 real Metal nodes; no model/search; publication_allowed=false)")


if __name__ == "__main__":
    main()
