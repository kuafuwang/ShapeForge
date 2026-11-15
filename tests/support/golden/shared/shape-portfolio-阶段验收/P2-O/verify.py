#!/usr/bin/env python3
"""P2-O acceptance: gemm_silu REG kept, global fp32 scratch rejected by measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2O = Path(__file__).resolve().parent
MANIFEST = P2O / "manifest.sha256"
BASELINE = P2O / "acceptance-baseline.json"
CPU_TEST = "tests/shared/test_gemm_silu_scratch_decision.py"
GPU_TEST = "tests/backends/metal/test_gemm_silu_scratch_gpu.py"
DELIVERABLES = (
    "docs/架构/gemm-silu-scratch-headroom.md",
    "experiments/gemm_silu_scratch_probe.py",
    "tests/support/gemm_silu_scratch.py",
    "tests/support/golden/metal/gemm_silu_reg/variants/small/kernel.metal",
    "tests/support/golden/metal/gemm_silu_reg/variants/paper/kernel.metal",
    "tests/backends/metal/gemm_silu_scratch_record.py",
    "tests/backends/metal/test_gemm_silu_scratch_gpu.py",
    "tests/shared/test_gemm_silu_scratch_decision.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/O.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/pack-closure.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/record.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/decision.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/scratch-probe.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/reg-small-confirmation.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/reg-paper-confirmation.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/reg-paper-confirmation-occupied.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/plain-gemm-small-control.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/plain-gemm-paper-control.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    MANIFEST.write_text("\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n", encoding="utf-8")
    print(f"OK: wrote P2-O manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-O manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-O manifest mismatch: {name}")
    print(f"OK: P2-O manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-o-gemm-silu-reg-vs-global-1", "wrong P2-O epoch")
    require(document["parent_head"] == "a9c1dccce636107c2852995578b05611bf19865b", "P2-O parent HEAD drifted")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    require(head == document["parent_head"], "the working tree is not on the recorded parent HEAD")
    require(document["publication_allowed"] is False, "P2-O must not publish")
    require(document["npu_simulated_only"] is True, "P2-O must keep NPU simulation-only")
    require(document["prompt_sha1"] == current_prompt_sha1(), "P2-O changed the prompt fingerprint")
    require(
        document["trust_toml_sha256"] == hashlib.sha256((ROOT / "targets/apple-m4-max/trust.toml").read_bytes()).hexdigest(),
        "P2-O changed the trust file",
    )
    closure = json.loads((P2O / "pack-closure.json").read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files), "P2-O task-pack closure drifted")
    require(closure["sha256"] == digest == document["pack_closure_sha256"], "P2-O task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-O pack file count drifted")
    print(f"OK: P2-O baseline (79 pack files, prompt={document['prompt_sha1'][:12]}…, trust pinned)")


def verify_history() -> None:
    counts = json.loads(BASELINE.read_text(encoding="utf-8"))["history_manifests"]
    for phase, count in counts.items():
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/{phase}/manifest.sha256"
        entries = [line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        require(len(entries) == count, f"{phase} historical manifest count drifted")
        for digest, name in entries:
            require(sha256(ROOT / name) == digest, f"{phase} historical file changed: {name}")
        print(f"OK: {phase} historical manifest remains byte-identical ({count} files)")


def verify_evidence() -> None:
    from tests.support import gemm_silu_scratch as scratch

    scratch.assert_source_provenance()
    scratch.assert_no_global_candidate()
    subjects = {
        "small": scratch.load_json(scratch.SMALL_CONFIRMATION),
        "paper": scratch.load_json(scratch.PAPER_CONFIRMATION),
    }
    controls = {
        "small": scratch.load_json(scratch.SMALL_CONTROL),
        "paper": scratch.load_json(scratch.PAPER_CONTROL),
    }
    probe = scratch.load_json(scratch.PROBE)
    scratch.assert_probe(probe)
    for size, subject in subjects.items():
        scratch.assert_subject(subject, size)
        require(subject["status"] == "confirmed", f"{size} is not confirmed")
        require(subject["identity"]["kind"] == "match", f"{size} identity is not a match")
    for size, control in controls.items():
        scratch.assert_control(control, size)
    decision = scratch.load_json(scratch.DECISION)
    scratch.assert_decision(decision, subjects, controls, probe)
    require(decision["adopted_route"] == "REG" and decision["decision"] == "reg_only", "P2-O did not keep REG")
    require(decision["implementation"]["global_candidate_source_added"] is False, "P2-O added a GLOBAL candidate")
    require(decision["implementation"]["same_algorithm_ab_run"] is False, "P2-O claims an A/B it did not run")
    for size, one in decision["sizes"].items():
        require(one["global_would_win"] is False, f"{size}: a GLOBAL route would have won")
        require(one["hard_lower_bound_us"] >= one["reg_candidate_us"], f"{size}: hard lower bound is below REG")
        require(one["best_probe_global_estimate_us"] >= one["reg_candidate_us"], f"{size}: probe extrapolation is below REG")
        require(one["probe_estimates_are_not_candidates"] is True, f"{size}: probe extrapolation is mislabelled")
        require(one["fusion_cost_fraction_of_reg"] < 0.01, f"{size}: the fusion cost is not negligible")
    occupied = scratch.load_json(scratch.EVIDENCE / "reg-paper-confirmation-occupied.json")
    require(occupied["schema_id"] == "shapeforge.gemm-silu-confirmation-retry-occupied.1", "retry wrapper drifted")
    require(
        scratch.stable_identity(occupied["subject"]) == scratch.stable_identity(subjects["paper"]),
        "the occupied retry is not the same candidate/identity",
    )
    require(
        occupied["subject"]["confirmation_report"]["candidate_us"]
        > subjects["paper"]["confirmation_report"]["candidate_us"],
        "the retry note does not describe a slower sample",
    )
    print("OK: P2-O REG sources, six-case fresh subjects, control, probe and negative decision")


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
    verify_evidence()
    pytest_nodes([CPU_TEST], where="P2-O CPU decision", expected=6)
    pytest_nodes(["-m", "gpu", GPU_TEST], where="P2-O real Metal replay", expected=6)
    print("OK: P2-O acceptance (6 CPU + 6 real Metal nodes; REG kept, GLOBAL rejected, publication_allowed=false)")


if __name__ == "__main__":
    main()
