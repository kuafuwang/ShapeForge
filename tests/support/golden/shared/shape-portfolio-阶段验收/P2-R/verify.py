#!/usr/bin/env python3
"""P2-R acceptance: worker geometry identity keeps non-empty scratch."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2R = Path(__file__).resolve().parent
P2Q_PACK = P2R.parent / "P2-Q" / "pack-closure.json"
MANIFEST = P2R / "manifest.sha256"
BASELINE = P2R / "acceptance-baseline.json"
CPU_TEST = "tests/shared/test_bundle_scratch_geometry_identity.py"
GPU_TEST = "tests/backends/metal/test_bundle_scratch_geometry_gpu.py"
OLD_PACK = "5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f"
# P2-I's closed manifest still records the pre-fix worker.  This phase changes that file and
# does not rewrite the manifest.  P2-J through P2-Q do not list it.
P2I_WORKER_PIN = "39ce59320d35066489d88db4eac213c65764e6c57f8932b16fbd6e84a01c370c"
LIVE_WORKER_SHA = "a083a36272e31bb9f9edaa60f5df9a57a927facf150582e7b51f087ac65cbe33"
DELIVERABLES = (
    "docs/架构/bundle-scratch-worker-geometry.md",
    "tests/support/bundle_scratch_identity.py",
    "tests/shared/test_bundle_scratch_geometry_identity.py",
    "tests/backends/metal/bundle_scratch_seq.py",
    "tests/backends/metal/test_bundle_scratch_geometry_gpu.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/R.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/pack-closure.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/review-r1-rejected.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/review-r3-rejected.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/review.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/evidence/pre-fix-identity.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/evidence/identity-delta.json",
)
CPU_NODES = 11
BACKEND_GEOMETRY_NODES = (
    "test_registered_kda_scratch_rows_are_four_wide",
    "test_a_single_read_write_poison_buffer_is_still_refused",
)
METAL_EVAL_NODES = (
    "test_texture_pingpong_executes_every_declared_case",
    "test_a_nonzero_host_program_is_incorrect_and_is_not_an_identity_mismatch",
    "test_device_pointer_pingpong_fails_reflection_after_identity_matches",
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
    print(f"OK: wrote P2-R manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-R manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-R manifest mismatch: {name}")
    print(f"OK: P2-R manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-r-bundle-scratch-geometry-1", "wrong P2-R epoch")
    require(
        document["parent_head"] == "a9c1dccce636107c2852995578b05611bf19865b",
        "P2-R parent HEAD drifted",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    require(head == document["parent_head"], "the working tree is not on the recorded parent HEAD")
    require(document["publication_allowed"] is False, "P2-R must not publish")
    require(document["npu_simulated_only"] is True, "P2-R must keep NPU simulation-only")
    require(document["prompt_sha1"] == current_prompt_sha1(), "P2-R changed the prompt fingerprint")
    require(
        document["trust_toml_sha256"]
        == hashlib.sha256((ROOT / "targets/apple-m4-max/trust.toml").read_bytes()).hexdigest(),
        "P2-R changed the trust file",
    )
    closure = json.loads((P2R / "pack-closure.json").read_text(encoding="utf-8"))
    previous = json.loads(P2Q_PACK.read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files) == list(previous["files"]), "P2-R task-pack file set drifted")
    require(closure["sha256"] == digest == document["pack_closure_sha256"], "P2-R task-pack digest drifted")
    require(digest != OLD_PACK == document["previous_pack_closure_sha256"] == previous["sha256"], "old pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-R pack file count drifted")
    require(closure["changed_files"] == ["tcenv/scratch/spec.py"], "the recorded pack delta is not spec.py")
    changed = []
    for name in files:
        old = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True, check=True)
        if old.stdout != (ROOT / name).read_bytes():
            changed.append(name)
    require(changed == ["tcenv/scratch/spec.py"], f"pack bytes changed outside spec.py: {changed}")
    require("tcenv/v2_worker.py" not in files, "v2_worker.py entered the task-pack closure")
    print(f"OK: P2-R baseline (79 pack files, digest {digest[:12]}…, prompt and trust pinned)")


def verify_history() -> None:
    """Historical manifests stay as written.  P2-I still pins the pre-fix worker; the live file moved."""
    counts = json.loads(BASELINE.read_text(encoding="utf-8"))["history_manifests"]
    for phase, count in counts.items():
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/{phase}/manifest.sha256"
        entries = [line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        require(len(entries) == count, f"{phase} historical manifest count drifted")
        mismatched = [name for digest, name in entries if sha256(ROOT / name) != digest]
        if phase == "P2-I":
            require(mismatched == ["tcenv/v2_worker.py"], f"P2-I live drift is not only the worker: {mismatched}")
            pinned = next(digest for digest, name in entries if name == "tcenv/v2_worker.py")
            require(pinned == P2I_WORKER_PIN, "P2-I's worker pin was rewritten")
            require(sha256(ROOT / "tcenv/v2_worker.py") == LIVE_WORKER_SHA, "the live worker hash drifted")
            print(f"OK: P2-I manifest unchanged ({count} files); live worker differs from the recorded pin")
            continue
        require(mismatched == [], f"{phase} historical file changed: {mismatched}")
        print(f"OK: {phase} historical manifest remains byte-identical ({count} files)")


def verify_evidence() -> None:
    from tests.support.bundle_scratch_identity import identity_delta, kda_contract, pre_fix_record
    from tests.support.seq_handoff import kda_source, pingpong_source

    recorded = json.loads((P2R / "evidence" / "pre-fix-identity.json").read_text(encoding="utf-8"))
    require(recorded == pre_fix_record(), "the pre-fix identity record is not a recomputation")
    require(recorded["pre_fix"]["pingpong_documents_equal"] is False, "the pre-fix failure was not reproduced")
    require(recorded["pre_fix"]["device_zero_documents_equal"] is False, "device_zero pre-fix failure was not reproduced")
    require(recorded["post_fix"]["pingpong_documents_equal"] is True, "the ping-pong document does not round-trip")
    require(recorded["post_fix"]["empty_scratch_key_absent"] is True, "empty scratch grew a key")
    require(recorded["post_fix"]["historical_scratch_key_absent"] is True, "a historical document grew a scratch key")
    require(recorded["publication_allowed"] is False, "the identity record publishes")
    delta = json.loads((P2R / "evidence" / "identity-delta.json").read_text(encoding="utf-8"))
    fresh = identity_delta(kda_source("seq_three_piece"), pingpong_source())
    require(delta == fresh, "the identity delta is not a recomputation")
    require(delta["execution_changed"] and delta["lifecycle_changed"], "scratch did not move execution identity")
    require(delta["artifact_changed"] and delta["native_changed"], "scratch did not move artifact identity")
    require(delta["evaluation_changed"] is True, "scratch did not move TaskEvaluationResult.identity")
    require(delta["evaluation_identity"] == "TaskEvaluationResult.identity", "evaluation identity is not the result property")
    require(delta["empty_evaluation"] != delta["scratch_evaluation"], "the two evaluation summaries are the same")
    require(len(delta["empty_evaluation"]) == 64 and len(delta["scratch_evaluation"]) == 64, "evaluation summary is not a sha256")
    require(delta["empty_candidate_sha256"] == delta["scratch_candidate_sha256"], "the evaluation comparison swapped candidates")
    require(delta["empty_task_pack_digest"] == delta["scratch_task_pack_digest"], "the evaluation comparison swapped task packs")
    require("task_pack_digest" in delta["evaluation_unchanged_fields"], "task_pack_digest was recorded as a scratch difference")
    require("candidate_sha256" in delta["evaluation_unchanged_fields"], "candidate bytes were not held fixed")
    require(
        set(delta["evaluation_changed_fields"])
        >= {"geometry_identity", "bundle_digest", "artifact_set_digest", "native_digest", "report"},
        "evaluation field movement was dropped",
    )
    require(delta["evaluation_differences"], "evaluation differences were omitted")
    require(
        any("execution_geometry.json" in item["path"] for item in delta["evaluation_differences"]),
        "the frozen geometry artifact difference was omitted",
    )
    require(delta["evaluation_device_verdict"] is False, "the CPU identity comparison claims a device verdict")
    require(
        delta["empty_report_status"] == delta["scratch_report_status"] == "rejected",
        "the CPU report shell changed or claims a successful device run",
    )
    require(delta["empty_resolved_has_scratch"] is False and delta["scratch_resolved_has_scratch"] is True, "scratch presence drifted")
    runner = (ROOT / "tcenv/taskcontract/runner.py").read_text(encoding="utf-8")
    require("lifecycle.geometry.document() != chosen.geometry.document()" in runner, "the identity comparison was removed")
    require("the lifecycle was registered for a different resolved geometry" in runner, "the identity error was removed")
    require(kda_contract().instance.tensor("handoff_states").space == "workspace", "handoff_states left the OpSpec workspace")
    print("OK: P2-R pre-fix mismatch, post-fix round-trip, identity delta and kept comparison")


def verify_review() -> None:
    rejected = json.loads((P2R / "review-r1-rejected.json").read_text(encoding="utf-8"))
    require(rejected["done"] is False, "round 1 was not a rejection")
    require(rejected["epoch"] == "p2-r-bundle-scratch-geometry-1", "round 1 belongs to another epoch")
    blocking = " ".join(rejected["blocking"])
    require("tcenv/v2_worker.py" in blocking and P2I_WORKER_PIN in blocking, "round 1's worker-pin finding was dropped")
    reopened = json.loads((P2R / "review-r3-rejected.json").read_text(encoding="utf-8"))
    require(reopened["done"] is False, "round 3 was not a rejection")
    require(reopened["round"] == 3, "round 3's rejection record moved")
    reopened_text = " ".join(reopened["blocking"])
    require("TaskEvaluationResult.identity" in reopened_text, "round 3's evaluation-identity finding was dropped")
    require("preflight_submission" in reopened_text and "evaluate_candidate" in reopened_text, "round 3's node split was dropped")
    document = json.loads((P2R / "review.json").read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-r-bundle-scratch-geometry-1", "the review belongs to another epoch")
    require(document["done"] is True, "the review did not sign off")
    require(document["reproduced_pre_fix_failure"] is True, "the reviewer did not reproduce the pre-fix failure")
    require(document["old_geometry_compatible"] is True, "the reviewer did not check old geometry bytes")
    require(document["evaluation_identity_checked"] is True, "the reviewer did not check evaluation identity")
    require(document["reviewer"]["edited_tree"] is False, "the reviewer edited the tree")
    require("gpu_passed" not in document, "the review still collapses backend and device nodes into gpu_passed")
    require(document["cpu_passed"] == CPU_NODES, "the review's CPU count drifted")
    require(document["backend_geometry_passed"] == len(BACKEND_GEOMETRY_NODES), "the review's backend/geometry count drifted")
    require(document["metal_evaluation_passed"] == len(METAL_EVAL_NODES), "the review's Metal evaluation count drifted")
    evidence = json.loads((P2R / "evidence" / "identity-delta.json").read_text(encoding="utf-8"))
    require(document["evaluation_changed"] is True, "the review does not record evaluation_changed")
    require(document["empty_evaluation"] == evidence["empty_evaluation"], "the review's empty evaluation summary drifted")
    require(document["scratch_evaluation"] == evidence["scratch_evaluation"], "the review's scratch evaluation summary drifted")
    rendered = json.dumps(document, ensure_ascii=False)
    require("5 real Metal" not in rendered and "真实 Metal 5" not in rendered, "the review still calls five nodes real Metal")
    require(document["publication_allowed"] is False, "the review must not publish")
    require(document["npu_simulated_only"] is True, "the review must keep NPU simulation-only")
    require(document["blocking"] == [], "the signed-off review still has blocking findings")
    print("OK: P2-R review (evaluation identity checked, nodes split, done=true)")


def classify_gpu_tests() -> tuple[list[str], list[str]]:
    """Split the gpu-marked file by whether the test calls ``_evaluate``."""
    tree = ast.parse((ROOT / GPU_TEST).read_text(encoding="utf-8"))
    backend: list[str] = []
    metal: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        calls_evaluate = any(
            isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "_evaluate"
            for item in ast.walk(node)
        )
        (metal if calls_evaluate else backend).append(node.name)
    return backend, metal


def verify_node_classes() -> None:
    backend, metal = classify_gpu_tests()
    require(tuple(backend) == BACKEND_GEOMETRY_NODES, f"backend/geometry nodes drifted: {backend}")
    require(tuple(metal) == METAL_EVAL_NODES, f"real Metal evaluation nodes drifted: {metal}")
    print(
        f"OK: P2-R gpu file is {len(backend)} backend/geometry + {len(metal)} real Metal evaluation"
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
    require(
        re.search(rf"(?<!\d){expected} passed(?!\d)", output) is not None,
        f"{where}: expected {expected} passed:\n{output}",
    )
    require(
        " skipped" not in output and "xfail" not in output and "deselected" not in output,
        f"{where}: skip/xfail/deselect is not accepted:\n{output}",
    )
    print(f"OK: {where} ({expected} passed)")


def _gpu_node_ids(names: tuple[str, ...]) -> list[str]:
    return [f"{GPU_TEST}::{name}" for name in names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument(
        "--static",
        action="store_true",
        help="skip the gpu-marked backend/geometry nodes and the real Metal evaluation nodes",
    )
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_baseline()
    verify_history()
    verify_evidence()
    verify_node_classes()
    verify_review()
    pytest_nodes([CPU_TEST], where="P2-R CPU identity", expected=CPU_NODES)
    if args.static:
        print(
            f"OK: P2-R static acceptance ({CPU_NODES} CPU nodes; "
            f"{len(BACKEND_GEOMETRY_NODES)} backend/geometry and "
            f"{len(METAL_EVAL_NODES)} real Metal evaluation nodes skipped on purpose)"
        )
        return
    pytest_nodes(
        ["-m", "gpu", *_gpu_node_ids(BACKEND_GEOMETRY_NODES)],
        where="P2-R backend/geometry",
        expected=len(BACKEND_GEOMETRY_NODES),
    )
    pytest_nodes(
        ["-m", "gpu", *_gpu_node_ids(METAL_EVAL_NODES)],
        where="P2-R real Metal evaluation",
        expected=len(METAL_EVAL_NODES),
    )
    print(
        f"OK: P2-R acceptance ({CPU_NODES} CPU + {len(BACKEND_GEOMETRY_NODES)} backend/geometry + "
        f"{len(METAL_EVAL_NODES)} real Metal evaluation nodes; "
        "worker scratch identity restored, publication_allowed=false)"
    )


if __name__ == "__main__":
    main()
