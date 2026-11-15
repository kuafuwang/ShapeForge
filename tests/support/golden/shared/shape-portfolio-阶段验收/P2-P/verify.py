#!/usr/bin/env python3
"""P2-P acceptance: bitdelta_gemm REG kept, every legal SHARED form measured slower."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2P = Path(__file__).resolve().parent
MANIFEST = P2P / "manifest.sha256"
BASELINE = P2P / "acceptance-baseline.json"
CPU_TEST = "tests/shared/test_bitdelta_unpack_decision.py"
GPU_TEST = "tests/backends/metal/test_bitdelta_unpack_gpu.py"
DELIVERABLES = (
    "docs/架构/bitdelta-unpack-headroom.md",
    "experiments/bitdelta_unpack_probe.py",
    "tests/support/bitdelta_unpack.py",
    "tests/support/golden/metal/bitdelta_reg/variants/small/kernel.metal",
    "tests/support/golden/metal/bitdelta_reg/variants/paper/kernel.metal",
    "tests/backends/metal/bitdelta_unpack_record.py",
    "tests/backends/metal/test_bitdelta_unpack_gpu.py",
    "tests/shared/test_bitdelta_unpack_decision.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/P.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/pack-closure.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/record.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/decision.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/unpack-probe.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/unpack-probe-replay.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/unpack-probe-r1-superseded.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/unpack-probe-r2-superseded.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/unpack-probe-r3-superseded.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/unpack-probe-r4-superseded.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/unpack-probe-replay-r4-superseded.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/reg-small-confirmation.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/reg-paper-confirmation.json",
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
    print(f"OK: wrote P2-P manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-P manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-P manifest mismatch: {name}")
    print(f"OK: P2-P manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-p-bitdelta-reg-vs-shared-1", "wrong P2-P epoch")
    require(document["parent_head"] == "a9c1dccce636107c2852995578b05611bf19865b", "P2-P parent HEAD drifted")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    require(head == document["parent_head"], "the working tree is not on the recorded parent HEAD")
    require(document["publication_allowed"] is False, "P2-P must not publish")
    require(document["npu_simulated_only"] is True, "P2-P must keep NPU simulation-only")
    require(document["prompt_sha1"] == current_prompt_sha1(), "P2-P changed the prompt fingerprint")
    require(
        document["trust_toml_sha256"]
        == hashlib.sha256((ROOT / "targets/apple-m4-max/trust.toml").read_bytes()).hexdigest(),
        "P2-P changed the trust file",
    )
    closure = json.loads((P2P / "pack-closure.json").read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files), "P2-P task-pack closure drifted")
    require(closure["sha256"] == digest == document["pack_closure_sha256"], "P2-P task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-P pack file count drifted")
    print(f"OK: P2-P baseline (79 pack files, prompt={document['prompt_sha1'][:12]}…, trust pinned)")


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
    from tests.support import bitdelta_unpack as unpack

    unpack.assert_source_provenance()
    unpack.assert_no_shared_candidate()
    unpack.assert_superseded()
    subjects = {
        "small": unpack.load_json(unpack.SMALL_CONFIRMATION),
        "paper": unpack.load_json(unpack.PAPER_CONFIRMATION),
    }
    probe = unpack.load_json(unpack.PROBE)
    unpack.assert_probe(probe)
    unpack.assert_replay(unpack.load_json(unpack.PROBE_REPLAY))
    for size, subject in subjects.items():
        unpack.assert_subject(subject, size)
        require(subject["status"] == "confirmed", f"{size} is not confirmed")
        require(subject["identity"]["kind"] == "match", f"{size} identity is not a match")
    decision = unpack.load_json(unpack.DECISION)
    unpack.assert_decision(decision, subjects, probe)
    require(decision["adopted_route"] == "REG" and decision["decision"] == "reg_only", "P2-P did not keep REG")
    require(decision["implementation"]["shared_candidate_source_added"] is False, "P2-P added a SHARED candidate")
    require(decision["implementation"]["same_algorithm_ab_run"] is False, "P2-P claims an A/B it did not run")
    for size, one in decision["sizes"].items():
        require(one["shared_would_win"] is False, f"{size}: a legal SHARED form would win")
        require(one["meets_adoption_bar"] is False, f"{size}: a legal SHARED form beats its REG cost")
        require(one["adoption_bar_ratio_over_reg"] == 1.0, f"{size}: the adoption bar is not 1.0x REG")
        require(
            one["best_legal_shared_ratio_over_reg"] > 1.0,
            f"{size}: the best legal SHARED form is not slower than REG",
        )
        require(
            one["best_legal_shared_estimate_us"] > one["reg_candidate_us"],
            f"{size}: the best legal SHARED estimate is not slower than REG",
        )
        require(one["probe_estimates_are_not_candidates"] is True, f"{size}: probe extrapolation is mislabelled")
        geometry = unpack.source_geometry(size)
        require(
            one["reg_tiling"]["row_blocks"] == geometry["row_blocks"]
            and one["reg_tiling"]["output_threadgroups"] == geometry["output_threadgroups"]
            and one["reg_tiling"]["tile_m"] == geometry["tile_m"],
            f"{size}: the recorded REG tiling is not the candidate source geometry",
        )
        require(one["shared_tiling"] == unpack.shared_geometry(size), f"{size}: SHARED tiling drifted")
        require(one["reg_decode_repeats"] == geometry["row_blocks"], f"{size}: decode repeats drifted")
        require(
            one["reg_decoded_elements"] == geometry["row_blocks"] * one["decode_elements"],
            f"{size}: decoded element count is not repeats x K x N",
        )
        require(one["measured_no_fill_fp16_us"] > one["roofline_lower_bound_us"], f"{size}: no-fill arm below the floor")
        require("best_legal_shared_measured_slower" in one["stop_reasons"], f"{size}: stop reason missing")
        require(
            "every_measured_legal_shared_form_slower" in one["stop_reasons"],
            f"{size}: the every-form stop reason is missing",
        )
    paper = decision["sizes"]["paper"]
    require(
        paper["recoverable_decode_headroom_fraction"] < 0.10,
        "paper: the recoverable decode headroom is not a single digit",
    )
    # The decision must not pretend the registered-baseline gate is the discriminator: at paper the
    # cheapest SHARED estimate would pass that gate (about 0.97x of the fastest baseline).
    require(
        paper["best_legal_shared_would_pass_registered_baseline_gate"] is True,
        "paper: the baseline-gate context is wrong; the SHARED estimate should pass it",
    )
    require(
        decision["sizes"]["small"]["best_legal_shared_would_pass_registered_baseline_gate"] is False,
        "small: the baseline-gate context is wrong; the SHARED estimate should fail it",
    )
    require("not the harness gate" in decision["adoption_bar"], "the adoption bar is mislabeled as the gate")
    require(paper["fp16_fill_marginal_us"] > 0, "paper: the fp16 fill is not a positive marginal cost")
    require(paper["subtraction_error_us"] > 0, "paper: subtracting the isolated fill arm is not shown to be wrong")
    require(paper["reg_over_roofline"] < 1.10, "paper: REG is not near the hardware floor")
    print("OK: P2-P REG sources, six-case fresh subjects, probe, superseded probes and negative decision")


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
    require(
        " skipped" not in output and "xfail" not in output and "deselected" not in output,
        f"{where}: skip/xfail/deselect is not accepted:\n{output}",
    )
    print(f"OK: {where} ({expected} passed)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--static", action="store_true", help="skip the real-Metal nodes")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_baseline()
    verify_history()
    verify_evidence()
    pytest_nodes([CPU_TEST], where="P2-P CPU decision", expected=8)
    if args.static:
        print("OK: P2-P static acceptance (8 CPU nodes; real Metal nodes skipped on purpose)")
        return
    pytest_nodes(["-m", "gpu", GPU_TEST], where="P2-P real Metal replay", expected=5)
    print(
        "OK: P2-P acceptance (8 CPU + 5 real Metal nodes; REG kept, every legal SHARED form slower, "
        "publication_allowed=false)"
    )


if __name__ == "__main__":
    main()
