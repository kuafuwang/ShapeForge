#!/usr/bin/env python3
"""P2-T acceptance: treatment-only overlay injection, unpaid render, no live search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2T = Path(__file__).resolve().parent
P2S = P2T.parent / "P2-S"
P2R_PACK = P2T.parent / "P2-R" / "pack-closure.json"
P2M_EPOCH = P2T.parent / "P2-M" / "prompt-epoch.json"
MANIFEST = P2T / "manifest.sha256"
BASELINE = P2T / "acceptance-baseline.json"
EVIDENCE = P2T / "evidence"
EXPERIENCE = ROOT / "targets/apple-m4-max/experience.md"
OVERLAY = P2S / "evidence" / "experience-overlay.md"
LIVE_WORKER_SHA = "a083a36272e31bb9f9edaa60f5df9a57a927facf150582e7b51f087ac65cbe33"
TESTS = (
    "tests/shared/test_experience_overlay_injection.py",
    "tests/shared/test_experience_search_prep.py",
    "tests/shared/test_search_yield_ab.py",
    "tests/shared/test_experience_staging.py",
    "tests/shared/test_orchestrator.py::test_a_new_run_freezes_guidance_and_an_old_run_still_copies_the_repo",
)
CPU_NODES = 37
DELIVERABLES = (
    "docs/架构/experience-overlay-injection.md",
    "experiments/common.py",
    "experiments/search_yield_ab.py",
    "taic/experience_overlay.py",
    "taic/kernel_agent.py",
    "tests/shared/test_experience_overlay_injection.py",
    "tests/shared/test_experience_search_prep.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/T.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/review-r1-rejected.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/review.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/evidence/approval.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/evidence/dry-run-plan.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/evidence/identity.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/evidence/injection.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T/evidence/render-comparison.json",
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
    print(f"OK: wrote P2-T manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-T manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-T manifest mismatch: {name}")
    print(f"OK: P2-T manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-t-experience-overlay-injection-1", "wrong P2-T epoch")
    require(document["parent_head"] == "a9c1dccce636107c2852995578b05611bf19865b", "P2-T parent HEAD drifted")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    require(head == document["parent_head"], "the working tree is not on the recorded parent HEAD")
    require(document["publication_allowed"] is False, "P2-T must not publish")
    require(document["npu_simulated_only"] is True, "P2-T must keep NPU simulation-only")
    require(document["plan_3_complete"] is False and document["plan_4_started"] is False, "P2-T claims a later plan")
    require(document["paid_search"] == "未授权/未执行", "P2-T must not claim the search ran")
    require(document["injection_ready"] is True, "P2-T lost the injection path")
    require(document["prompt_changed"] is False, "P2-T records a prompt change")
    live_prompt = current_prompt_sha1()
    require(document["prompt_sha1"] == document["previous_prompt_sha1"] == live_prompt, "P2-T prompt fingerprint moved")
    epoch = json.loads(P2M_EPOCH.read_text(encoding="utf-8"))
    require(live_prompt == epoch["current_prompt_sha1"], "P2-T moved off the P2-M prompt fingerprint")
    require(live_prompt != epoch["previous_prompt_sha1"], "P2-M's previous prompt sha was rewritten")
    require(sha256(ROOT / "taic/orchestrator.py") == document["orchestrator_sha256"], "orchestrator.py left the P2-I pin")
    require(
        document["trust_toml_sha256"] == hashlib.sha256((ROOT / "targets/apple-m4-max/trust.toml").read_bytes()).hexdigest(),
        "P2-T changed the trust file",
    )
    require(sha256(P2S / "manifest.sha256") == document["p2s_manifest_sha256"], "P2-S manifest bytes changed")
    previous = json.loads(P2R_PACK.read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(files) == list(previous["files"]), "P2-T task-pack file set drifted")
    require(digest == document["pack_closure_sha256"] == previous["sha256"], "P2-T task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-T pack file count drifted")
    require("experience_overlay" not in (ROOT / "taic/orchestrator.py").read_text(encoding="utf-8"), "orchestrator gained an overlay hook")
    print(f"OK: P2-T baseline (prompt unchanged, orchestrator pinned, 79 pack files, digest {digest[:12]}…)")


def verify_history() -> None:
    """Historical manifests stay as written. P2-I still pins the pre-fix worker. P2-S keeps its two drifted files."""
    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    counts = document["history_manifests"]
    require(
        set(counts) == {f"P2-{name}" for name in ("I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S")},
        "history set drifted",
    )
    drift = list(document["p2s_live_drift"])
    for phase, count in counts.items():
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/{phase}/manifest.sha256"
        entries = [line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        require(len(entries) == count, f"{phase} historical manifest count drifted")
        mismatched = [name for digest, name in entries if sha256(ROOT / name) != digest]
        if phase == "P2-I":
            require(mismatched == ["tcenv/v2_worker.py"], f"P2-I live drift is not only the worker: {mismatched}")
            pinned = next(digest for digest, name in entries if name == "tcenv/v2_worker.py")
            require(pinned == document["p2i_worker_pin"], "P2-I's worker pin was rewritten")
            require(sha256(ROOT / "tcenv/v2_worker.py") == LIVE_WORKER_SHA, "the live worker hash drifted")
            print(f"OK: P2-I manifest unchanged ({count} files); live worker differs from the recorded pin")
            continue
        if phase == "P2-S":
            require(sorted(mismatched) == sorted(drift), f"P2-S live drift is not the two injection files: {mismatched}")
            print(f"OK: P2-S manifest unchanged ({count} files); live drift is {', '.join(sorted(mismatched))}")
            continue
        require(mismatched == [], f"{phase} historical file changed: {mismatched}")
        print(f"OK: {phase} historical manifest remains byte-identical ({count} files)")


def _comparison_from_manifest(manifest: dict) -> dict:
    arms = {}
    for cell in manifest["cells"]:
        arms.setdefault(cell["arm"], cell)
    control, treatment = arms["control"], arms["treatment"]
    sidecar = treatment["experience_overlay"]
    from experiments.experience_catalog import OVERLAY_MARK

    return {
        "schema_id": "shapeforge.p2t-render-comparison.1",
        "injection_ready": True,
        "produces_candidates": False,
        "calls_model": False,
        "is_search_result": False,
        "paid_search": "未授权/未执行",
        "publication_allowed": False,
        "search_eval_seed": manifest["search_eval_seed"],
        "committees": manifest["committees"],
        "system_prompt_sha256": manifest["system_prompt_sha256"],
        "system_prompt_contains_overlay": False,
        "normalized_task_equal_across_arms": len({cell["normalized_task_sha256"] for cell in manifest["cells"]}) == 1,
        "overlay_copies": {"control": control["overlay_copies"], "treatment": treatment["overlay_copies"]},
        "control_experience_sha256": control["experience_sha256"],
        "control_experience_bytes": control["experience_bytes"],
        "treatment_experience_sha256": treatment["experience_sha256"],
        "treatment_experience_bytes": treatment["experience_bytes"],
        "overlay_path": sidecar["overlay_path"],
        "overlay_sha256": sidecar["overlay_sha256"],
        "rendered_experience_sha256": sidecar["rendered_experience_sha256"],
        "authoritative_task_rules": sidecar["authoritative_task_rules"],
        "not_a_task_contract": sidecar["not_a_task_contract"],
        "installed_in_global_experience": sidecar["installed_in_global_experience"],
        "sidecar_contains_overlay_mark": OVERLAY_MARK in json.dumps(sidecar, ensure_ascii=False),
    }


def verify_evidence() -> None:
    from experiments.experience_catalog import OVERLAY_MARK
    from experiments.search_yield_ab import build_experience_plan, render_experience_plan
    from taic.experience_overlay import injection_document
    from taic.trust import current_prompt_sha1

    approval = json.loads((EVIDENCE / "approval.json").read_text(encoding="utf-8"))
    for key in ("approved_by", "delta", "tolerable_overrun", "execution_record", "experiment_id", "gpu_window"):
        require(approval[key] is None, f"approval field {key} was filled")
    require(approval["executed"] is False and approval["paid_search"] == "未授权/未执行", "approval claims execution")
    require(approval["injection_ready"] is True and approval["live_search_authorized"] is False, "approval lost the unpaid boundary")
    identity = json.loads((EVIDENCE / "identity.json").read_text(encoding="utf-8"))
    require(identity["prompt_sha1"] == current_prompt_sha1(), "identity prompt sha is stale")
    require(identity["prompt_changed"] is False, "identity records a prompt change")
    require(identity["overlay_installed_in_global_experience"] is False, "identity says the overlay was installed")
    require(OVERLAY_MARK not in EXPERIENCE.read_text(encoding="utf-8"), "the overlay was installed into experience.md")
    require(sha256(EXPERIENCE) == identity["experience_file_sha256"], "experience.md hash moved")
    require(json.loads((EVIDENCE / "injection.json").read_text(encoding="utf-8")) == injection_document(), "injection document drifted")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    plan = build_experience_plan(
        overlay_sha256=sha256(OVERLAY),
        overlay_path=OVERLAY.relative_to(ROOT).as_posix(),
        git_head=head,
    )
    stored_plan = json.loads((EVIDENCE / "dry-run-plan.json").read_text(encoding="utf-8"))
    require(stored_plan == plan, "dry-run plan does not match the live builder")
    require(stored_plan["injection_ready"] is True and stored_plan["executed"] is False, "dry-run plan claims execution")
    require(stored_plan["paid_search"] == "未授权/未执行", "dry-run plan claims a paid search")
    require("没有注入" not in json.dumps(stored_plan, ensure_ascii=False), "dry-run still lists a missing injection path")
    out = ROOT / "runs" / "_p2t_verify_render"
    if out.exists():
        shutil.rmtree(out)
    try:
        manifest = render_experience_plan(out, plan)
        require(manifest["produces_candidates"] is False and manifest["calls_model"] is False, "render-only claims a search product")
        require(manifest["is_search_result"] is False and manifest["paid_search"] == "未授权/未执行", "render-only claims a search result")
        comparison = _comparison_from_manifest(manifest)
        stored = json.loads((EVIDENCE / "render-comparison.json").read_text(encoding="utf-8"))
        require(comparison == stored, "live render does not match the archived comparison")
        control = out / "render-only" / "control" / "gemm_silu_small_rep1" / "agent_work" / "init" / "experience.md"
        treatment = out / "render-only" / "treatment" / "gemm_silu_small_rep1" / "agent_work" / "init" / "experience.md"
        require(control.read_text(encoding="utf-8").count(OVERLAY_MARK) == 0, "control render contains the overlay")
        require(treatment.read_text(encoding="utf-8").count(OVERLAY_MARK) == 1, "treatment render lost the single overlay")
        require(list(out.rglob("result.json")) == [], "render-only wrote a search result")
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print("OK: P2-T blank approval, live render, and unchanged prompt")


def verify_review() -> None:
    rejected = json.loads((P2T / "review-r1-rejected.json").read_text(encoding="utf-8"))
    require(rejected["done"] is False and rejected["prior_done"] is True, "the premature signature is still done")
    rejected_text = json.dumps(rejected, ensure_ascii=False)
    require("brief.repo" in rejected_text and "rendered_experience_sha256" in rejected_text, "the rejection dropped a blocking finding")
    document = json.loads((P2T / "review.json").read_text(encoding="utf-8"))
    comparison = json.loads((EVIDENCE / "render-comparison.json").read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-t-experience-overlay-injection-1", "the review belongs to another epoch")
    require(document["done"] is True, "the independent signature is not done")
    require(document.get("awaiting_independent_signature", False) is False, "the signed review still awaits a signature")
    require(document["supersedes"] == "review-r1-rejected.json", "the review does not supersede the rejected signature")
    require(document["execute_not_invoked"] is True, "the review invoked a live search")
    require(document["publication_allowed"] is False and document["npu_simulated_only"] is True, "the review publishes or opens NPU")
    require(document["paid_search"] == "未授权/未执行", "the review claims the search ran")
    require(document["plan_3_complete"] is False and document["plan_4_started"] is False, "the review claims a later plan")
    require(document["prompt_changed"] is False, "the review records a prompt change")
    require(document["reviewer"]["edited_tree"] is False, "the reviewer edited the tree")
    require(document["reviewer"]["paid_model"] is False and document["reviewer"]["cost_usd"] == 0, "the review spent model budget")
    require(document["blocking"] == [], "the signed review still lists a blocker")
    for key in (
        "approval_blank",
        "render_replayed",
        "binding_and_rendered_sha_refusal_checked",
        "missing_overlay_key_refusal_checked",
        "pair_matrix_checked",
        "committees_checked",
        "lightweight_brief_checked",
    ):
        require(document[key] is True, f"the review did not check {key}")
    require(document["shared_run_seed"] == 7, "the shared run seed is not 7")
    require(
        document["control_experience_bytes"] == 13415 == comparison["control_experience_bytes"],
        "control experience bytes do not match the live render",
    )
    require(
        document["treatment_experience_bytes"] == 25469 == comparison["treatment_experience_bytes"],
        "treatment experience bytes do not match the live render",
    )
    require(
        document["overlay_copies"] == {"control": 0, "treatment": 1} == comparison["overlay_copies"],
        "overlay copy counts do not match the live render",
    )
    require(
        document["rendered_experience_sha256"] == comparison["rendered_experience_sha256"],
        "rendered experience sha does not match render-comparison.json",
    )
    require(str(document["rendered_experience_sha256"]).startswith("8e5adbec"), "rendered experience sha prefix drifted")
    require(document["prompt_sha1"] == baseline["prompt_sha1"], "prompt sha1 does not match acceptance-baseline.json")
    require(str(document["prompt_sha1"]).startswith("e9d28ff"), "prompt sha1 prefix drifted")
    require(
        document["pack_closure_sha256"] == baseline["pack_closure_sha256"],
        "pack digest does not match acceptance-baseline.json",
    )
    require(str(document["pack_closure_sha256"]).startswith("b5b74185"), "pack digest prefix drifted")
    require(
        document["remaining_blockers"] == [
            "approved_by is blank.",
            "delta is blank, so a paired result would be inconclusive.",
            "Tolerable overrun past the $12 start-next-cell commitment cap is not approved.",
            "Paid search is 未授权/未执行.",
            "Plan 3 is not complete and Plan 4 has not started.",
            "publication_allowed is false.",
            "NPU stays simulation-only.",
            "The frozen overlay must stay out of targets/apple-m4-max/experience.md, because installing it would move current_prompt_sha1.",
            "The P2-S manifest is not rewritten. Live experiments/search_yield_ab.py and tests/shared/test_experience_search_prep.py differ from that pin.",
        ],
        "remaining blockers changed",
    )
    notes = document["notes"]
    require("13415" in notes and "25469" in notes and "8e5adbec" in notes, "notes omit the measured render")
    rendered = json.dumps(document, ensure_ascii=False)
    require("没有注入路径" not in rendered and "no prompt-injection" not in rendered, "the review still lists a missing injection path")
    print("OK: P2-T review (r1 signature void; independent signature done=true; paid search 未授权/未执行)")


def pytest_nodes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *TESTS],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dict(os.environ),
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"P2-T CPU tests failed:\n{output}")
    require(__import__("re").search(rf"(?<!\d){CPU_NODES} passed(?!\d)", output) is not None, f"expected {CPU_NODES} passed:\n{output}")
    require(" skipped" not in output and "xfail" not in output and "deselected" not in output, f"skip/xfail/deselect is not accepted:\n{output}")
    print(f"OK: P2-T CPU tests ({CPU_NODES} passed)")


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
    verify_review()
    pytest_nodes()
    print("OK: P2-T acceptance (injection ready; paid search 未授权/未执行; plan_3_complete=false)")


if __name__ == "__main__":
    main()
