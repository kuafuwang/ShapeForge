#!/usr/bin/env python3
"""P2-Q acceptance: SEQ state handoff is safe, and candidate scratch buys it nothing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2Q = Path(__file__).resolve().parent
MANIFEST = P2Q / "manifest.sha256"
BASELINE = P2Q / "acceptance-baseline.json"
CPU_TEST = "tests/shared/test_seq_handoff_decision.py"
GPU_TEST = "tests/backends/metal/test_seq_state_handoff_gpu.py"
DELIVERABLES = (
    "docs/架构/seq-state-handoff-headroom.md",
    "experiments/seq_handoff_probe.py",
    "tests/support/seq_handoff.py",
    "tests/backends/metal/seq_handoff_record.py",
    "tests/backends/metal/test_seq_state_handoff_gpu.py",
    "tests/shared/test_seq_handoff_decision.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/Q.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/pack-closure.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/review.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence/decision.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence/seq-handoff-measurements.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence/seq-handoff-probe.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence/seq-handoff-scratch-attempt.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence/confirmations.json",
)
CPU_NODES = 18
GPU_NODES = 27


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
    print(f"OK: wrote P2-Q manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-Q manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-Q manifest mismatch: {name}")
    print(f"OK: P2-Q manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-q-seq-state-handoff-1", "wrong P2-Q epoch")
    require(
        document["parent_head"] == "a9c1dccce636107c2852995578b05611bf19865b",
        "P2-Q parent HEAD drifted",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    require(head == document["parent_head"], "the working tree is not on the recorded parent HEAD")
    require(document["publication_allowed"] is False, "P2-Q must not publish")
    require(document["npu_simulated_only"] is True, "P2-Q must keep NPU simulation-only")
    require(document["prompt_sha1"] == current_prompt_sha1(), "P2-Q changed the prompt fingerprint")
    require(
        document["trust_toml_sha256"]
        == hashlib.sha256((ROOT / "targets/apple-m4-max/trust.toml").read_bytes()).hexdigest(),
        "P2-Q changed the trust file",
    )
    closure = json.loads((P2Q / "pack-closure.json").read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files), "P2-Q task-pack closure drifted")
    require(
        closure["sha256"] == digest == document["pack_closure_sha256"],
        "P2-Q task-pack digest drifted",
    )
    require(len(files) == document["pack_closure_files"] == 79, "P2-Q pack file count drifted")
    print(f"OK: P2-Q baseline (79 pack files, prompt={document['prompt_sha1'][:12]}…, trust pinned)")


def verify_history() -> None:
    counts = json.loads(BASELINE.read_text(encoding="utf-8"))["history_manifests"]
    for phase, count in counts.items():
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/{phase}/manifest.sha256"
        entries = [line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        require(len(entries) == count, f"{phase} historical manifest count drifted")
        for digest, name in entries:
            require(sha256(ROOT / name) == digest, f"{phase} historical file changed: {name}")
        print(f"OK: {phase} historical manifest remains byte-identical ({count} files)")


def verify_review() -> None:
    """The independent review trail must stay attached to the phase it signed off."""
    document = json.loads((P2Q / "review.json").read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-q-seq-state-handoff-1", "the review belongs to another epoch")
    require(document["done"] is True, "the review did not sign off")
    require(document["final_verdict_round"] == 3, "the signed-off round is not the last recorded one")
    require([item["verdict"] for item in document["prior_rounds"]] == ["done=false", "done=false"], "round trail drifted")
    require(document["reviewer"]["rounds"] == 3, "the reviewer round count drifted")
    blocking = [text for item in document["prior_rounds"] for text in item.get("blocking", [])]
    require(len(blocking) == 2, "the two blocking findings must stay on record")
    require(any("two-buffer ping-pong" in text for text in blocking), "the retracted claim is not on record")
    require(any("doubles the storage" in text for text in blocking), "the storage error is not on record")
    require(document["publication_allowed"] is False, "the review must not publish")
    require(document["npu_simulated_only"] is True, "the review must keep NPU simulation-only")
    print("OK: P2-Q review trail (3 rounds, 2 blocking findings retained, done=true)")


def verify_evidence() -> None:
    from tests.support import seq_handoff as sh

    measurements = sh.load_json(sh.MEASUREMENTS)
    probe = sh.load_json(sh.PROBE)
    scratch = sh.load_json(sh.SCRATCH_ATTEMPT)
    confirmations = sh.load_json(sh.CONFIRMATIONS)
    decision = sh.load_json(sh.DECISION)
    sh.assert_measurements(measurements)
    sh.assert_confirmations(confirmations)
    sh.assert_scratch_attempt(scratch)
    sh.assert_decision(decision)
    require(decision == sh.decision_document(measurements, probe, scratch), "the decision is not a recomputation")
    require(decision["adopted_route"] == "existing_abi_workspace_serial_stages", "the route moved")
    require(decision["scratch_spec_added"] is False, "P2-Q added a ScratchSpec")
    require(decision["implementation_changed"] is False, "P2-Q changed the implementation")

    # The three arms must be the same algorithm: only the Host Program's piece count and the
    # candidate's own stage count change.
    shipped_planner = sh.KDA_PLANNER.read_text(encoding="utf-8")
    for arm in sh.KDA_ARMS:
        planner = sh.kda_planner_source(arm)
        if arm == "seq_three_piece":
            require(planner == shipped_planner, "the shipped arm is not the shipped planner")
        else:
            changed = [
                (left, right)
                for left, right in zip(shipped_planner.splitlines(), planner.splitlines())
                if left != right
            ]
            require(
                changed == [(sh.KDA_PIECE_LINE, "    const uint32_t pieces = 1u;")],
                f"{arm}: the planner difference is not only the piece count",
            )
    shipped_varlen = sh.VARLEN_DEVICE.read_text(encoding="utf-8")
    require(sh.varlen_source("pieces_three") == shipped_varlen, "the shipped varlen arm is not the fixture")
    require(
        sh.varlen_source("pieces_one") == shipped_varlen.replace(sh.VARLEN_PIECE_LINE, "    uint pieces = 1u;"),
        "the one-piece varlen arm is not the fixture with a single cut",
    )

    # Fresh confirmations: every arm, never the search seed, identity match.
    require(len(confirmations["subjects"]) == len(sh.KDA_ARMS) * 2 + len(sh.VARLEN_ARMS), "confirmation count")
    for key, subject in confirmations["subjects"].items():
        require(subject["status"] == "confirmed", f"{key} is not confirmed")
        require(subject["identity"]["kind"] == "match", f"{key} identity is {subject['identity']['kind']}")
        require(int(subject["confirmation_seed"]) != 7, f"{key} used the search seed")

    # The scratch route is refused, not merely unused.
    poison = scratch["attempts"]["uninitialized_poison"]
    require(poison["accepted"] is False, "poison scratch was accepted for a recurrent state")
    require("write-only" in poison["detail"], poison["detail"])
    require(scratch["scratch_candidate_added"] is False, "a SEQ scratch candidate was added")

    # The probe must be a measurement, not a candidate, and must show the width effect.
    require(probe["not_a_candidate"] is True and probe["publication_allowed"] is False, "probe is mislabelled")
    narrow = probe["handoff"]["handoff_state_fp32_D64"]
    wide = probe["handoff"]["handoff_state_fp32_D64_wide"]
    require(narrow["state_bytes"] == wide["state_bytes"], "the two copies do not move the same bytes")
    require(narrow["threads"] < wide["threads"], "the width comparison is not a width comparison")
    require(narrow["handoff_over_control_us"] > 10.0, "the narrow copy is not measurably slow")
    require(wide["handoff_over_control_us"] < 5.0, "the wide copy is not measurably cheap")
    require(probe["kda_stage_cost"]["small"]["pipeline_stage_floor_us"] > 0.0, "the stage floor is not positive")

    # Every measured size must show the SEQ path as the slowest of its arms.
    for name, one in decision["sizes"].items():
        require(
            one["seq_three_piece_us"] >= one["single_stage_us"],
            f"{name}: the SEQ path is not the slowest arm",
        )
    require(
        decision["sizes"]["kda_small"]["seq_overhead_fraction_of_single_stage"] > 0.15,
        "small: cutting the chain is not a material cost",
    )
    require(
        decision["sizes"]["kda_paper"]["seq_overhead_fraction_of_single_stage"] < 0.01,
        "paper: the cut is not a fixed cost",
    )
    require(
        decision["sizes"]["varlen_mixed"]["seq_overhead_fraction_of_single_stage"] > 0.40,
        "varlen mixed: the packed handoff is not the dominant extra cost",
    )
    print(
        "OK: P2-Q SEQ arms, fresh subjects, scratch refusal, probe and route decision "
        "(existing ABI workspace kept, no ScratchSpec)"
    )


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
    verify_review()
    verify_evidence()
    pytest_nodes([CPU_TEST], where="P2-Q CPU decision", expected=CPU_NODES)
    if args.static:
        print(f"OK: P2-Q static acceptance ({CPU_NODES} CPU nodes; real Metal nodes skipped on purpose)")
        return
    pytest_nodes(["-m", "gpu", GPU_TEST], where="P2-Q real Metal SEQ path", expected=GPU_NODES)
    print(
        f"OK: P2-Q acceptance ({CPU_NODES} CPU + {GPU_NODES} real Metal nodes; SEQ protocol safe, "
        "handoff not worth a ScratchSpec, publication_allowed=false)"
    )


if __name__ == "__main__":
    main()
