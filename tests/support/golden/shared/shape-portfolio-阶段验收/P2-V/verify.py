#!/usr/bin/env python3
"""P2-V acceptance: task-scoped compact overlay, unpaid render, no live search.

This script recomputes the projection, the dry-run and one real render. It does
not start ``taic run``, ``taic resume``, ``taic track-explore``, a GPU benchmark,
or a paid call. A compact ``--execute`` is required to return before any runner.
"""

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
P2V = Path(__file__).resolve().parent
P2U = P2V.parent / "P2-U"
P2M_EPOCH = P2V.parent / "P2-M" / "prompt-epoch.json"
MANIFEST = P2V / "manifest.sha256"
BASELINE = P2V / "acceptance-baseline.json"
EVIDENCE = P2V / "evidence"
EXPERIENCE = ROOT / "targets/apple-m4-max/experience.md"
DOC = ROOT / "docs/架构/compact-experience-overlay.md"
AGENTS = ROOT / "AGENTS.md"
COMPACT = EVIDENCE / "compact-overlay.md"
FULL = P2V.parent / "P2-S" / "evidence" / "experience-overlay.md"
LIVE_WORKER_SHA = "a083a36272e31bb9f9edaa60f5df9a57a927facf150582e7b51f087ac65cbe33"
PARENT_HEAD = "d3388497aabaf2a687f3f1d65801c87edd9fdd9c"
PREVIOUS_PARENT_HEAD = "a9c1dccce636107c2852995578b05611bf19865b"
ORCHESTRATOR_SHA = "458cd29093fd36fd87fb2f500d8bcbe935bfa2d1d6a0fb583bde4b7c868795bd"
PREVIOUS_ORCHESTRATOR_SHA = "35832a11e22bf3a5a222f541157ac85e02a5a2d17d2ad925dc575fb2511e9ef1"
EXPERIENCE_OVERLAY_SHA = "55ebc524f4c0b0feb6c23325b1e3cc77512ad0519b83ab122cc46b3cb8211ec5"
P2T_EXPERIENCE_OVERLAY_PIN = "982439a7233b1f3261a7cacd342dee877d3cbce37247c9c1c592e2408f7df4db"
P2M_PROMPT_TEST_PIN = "1ecefb12391d0faf02083afaa11d02050d1a8eec802fb6a9fa131245bd31bf6b"
NORMALIZED_TASK_SHA = "7ad7dfe8a3be4ed0245f07ffa8bd27abc50e52507a41bb47f2d8b7fb73d028bb"
SUPERSEDED_REVIEW = "review-r1-superseded.json"
SUPERSEDED_REVIEW_SHA = "525c1237c89c949b4639d463b53b1322061cc401e9d6fa46c8aef0ddc48fe8c5"
P2I_LIVE_DRIFT = ["taic/orchestrator.py", "tcenv/v2_worker.py"]
P2M_LIVE_DRIFT = ["tests/shared/test_search_v2_bundle_prompt.py"]
P2T_LIVE_DRIFT = ["experiments/search_yield_ab.py", "taic/experience_overlay.py"]
TESTS = ("tests/shared/test_experience_compact_overlay.py",)
CPU_NODES = 11
COMPACT_REL = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/evidence/compact-overlay.md"
FULL_REL = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/evidence/experience-overlay.md"
BANNED = (
    "conv2d",
    "bitdelta",
    "kda",
    "varlen",
    "split-k",
    "splitk",
    "paper",
    "1160",
    "1.6721",
    "1.0071",
    "simdgroup",
    "steel",
    "bm=",
    "bk=",
    "grid=",
)
REQUIRED = (
    "一次性 SiLU 中间结果留在寄存器",
    "不省 FLOPs",
    "fp32 写读",
    "barrier",
    "producer/consumer",
    "dispatch",
    "1.8252",
    "8.7713",
    "not_a_candidate=true",
    "不是发布 speedup",
    "不指定候选 tiling",
    "fresh identity",
    "publication_allowed=false",
    "overlay_kind: compact",
)
REMAINING = [
    "approved_by is blank.",
    "delta is blank.",
    "Tolerable overrun is blank.",
    "P2-U approval (kuafu, delta 0.01, cap raised to $15) does not authorize this experiment.",
    "Filling allow-spend, allow-gpu, approved-by and a positive finite delta still does not execute a compact overlay.",
    "Paid search is 未授权/未执行.",
    "adopt_compact_overlay is false.",
    "Plan 4 has not started.",
    "publication_allowed is false.",
    "NPU stays simulation-only.",
    "The compact overlay must stay out of targets/apple-m4-max/experience.md, because installing it would move current_prompt_sha1.",
    "Compared with the executed P2-U sessions, TASK.md also changed, so overlay length is not the only variable.",
    "P2-I through P2-U manifests are not rewritten. Re-running the P2-T or P2-U verifier fails as historical anchor drift: those baselines still require parent HEAD a9c1dccce636107c2852995578b05611bf19865b and orchestrator 35832a11e22bf3a5a222f541157ac85e02a5a2d17d2ad925dc575fb2511e9ef1.",
]
DELIVERABLES = (
    "docs/架构/compact-experience-overlay.md",
    "experiments/experience_compact.py",
    "tests/shared/test_experience_compact_overlay.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/V.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/review-r1-superseded.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/review.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/evidence/approval.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/evidence/compact-overlay.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/evidence/dry-run-plan.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/evidence/projection.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/evidence/render-comparison.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), f"{path} is not a JSON object")
    return document


def write_manifest() -> None:
    MANIFEST.write_text(
        "\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n",
        encoding="utf-8",
    )
    print(f"OK: wrote P2-V manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-V manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-V manifest mismatch: {name}")
    print(f"OK: P2-V manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    document = load(BASELINE)
    require(document["epoch"] == "p2v-compact-experience-overlay-1", "wrong P2-V epoch")
    require(document["parent_head"] == PARENT_HEAD, "P2-V parent HEAD drifted")
    require(document["previous_parent_head"] == PREVIOUS_PARENT_HEAD, "P2-V dropped the previous parent HEAD")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    require(head == document["parent_head"], "the working tree is not on the recorded parent HEAD")
    require(document["publication_allowed"] is False, "P2-V must not publish")
    require(document["npu_simulated_only"] is True, "P2-V must keep NPU simulation-only")
    require(document["plan_4_started"] is False, "P2-V claims Plan 4 started")
    require(document["adopt_compact_overlay"] is False, "P2-V adopts the compact overlay")
    require(document["inherits_p2u_approval"] is False, "P2-V inherits the P2-U approval")
    require(document["paid_search"] == "未授权/未执行", "P2-V claims the search ran")
    require(document["prompt_changed"] is False, "P2-V records a prompt change")
    live_prompt = current_prompt_sha1()
    require(document["prompt_sha1"] == document["previous_prompt_sha1"] == live_prompt, "P2-V prompt fingerprint moved")
    epoch = json.loads(P2M_EPOCH.read_text(encoding="utf-8"))
    require(live_prompt == epoch["current_prompt_sha1"], "P2-V moved off the P2-M prompt fingerprint")
    require(
        sha256(ROOT / "taic/orchestrator.py") == document["orchestrator_sha256"] == ORCHESTRATOR_SHA,
        "orchestrator.py left the recorded P2-V pin",
    )
    require(document["previous_orchestrator_sha256"] == PREVIOUS_ORCHESTRATOR_SHA, "P2-V dropped the previous orchestrator pin")
    require(document["previous_orchestrator_sha256"] != document["orchestrator_sha256"], "P2-V collapsed the orchestrator change")
    require(document["p2i_orchestrator_pin"] == PREVIOUS_ORCHESTRATOR_SHA, "P2-I orchestrator pin was rewritten in the baseline")
    require(sha256(ROOT / "taic/experience_overlay.py") == document["experience_overlay_sha256"] == EXPERIENCE_OVERLAY_SHA, "experience_overlay.py left the P2-V pin")
    require(document["experience_render_changed"] is False, "P2-V claims the experience render changed")
    require(document["task_text_shared_by_both_arms"] is True, "P2-V no longer shares one TASK.md across arms")
    require(document["task_text_requires_eval_v2_validate"] is False, "P2-V still requires eval-v2 --validate")
    require(document["cross_experiment_only_variable_is_overlay_length"] is False, "P2-V claims overlay length is the only cross-experiment variable")
    require(document["normalized_task_sha256"] == NORMALIZED_TASK_SHA, "normalized TASK.md sha drifted")
    require(document["p2i_live_drift"] == P2I_LIVE_DRIFT, "P2-I live drift classification changed")
    require(document["p2m_prompt_test_pin"] == P2M_PROMPT_TEST_PIN, "P2-M prompt-test pin was rewritten")
    require(document["p2m_live_drift"] == P2M_LIVE_DRIFT, "P2-M live drift classification changed")
    require(document["p2t_experience_overlay_pin"] == P2T_EXPERIENCE_OVERLAY_PIN, "P2-T experience_overlay pin was rewritten")
    require(document["p2t_live_drift"] == P2T_LIVE_DRIFT, "P2-T live drift classification changed")
    require(sha256(ROOT / "targets/apple-m4-max/trust.toml") == document["trust_toml_sha256"], "P2-V changed the trust file")
    require(sha256(ROOT / "experiments/search_yield_ab.py") == document["search_yield_ab_sha256"], "search_yield_ab.py left the P2-V pin")
    previous = json.loads((P2V.parent / "P2-R" / "pack-closure.json").read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(files) == list(previous["files"]), "P2-V task-pack file set drifted")
    require(digest == document["pack_closure_sha256"] == previous["sha256"], "P2-V task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-V pack file count drifted")
    require(sha256(ROOT / "tcenv/v2_worker.py") == document["live_worker_sha256"] == LIVE_WORKER_SHA, "the live worker hash drifted")
    print(f"OK: P2-V baseline (prompt unchanged, 79 pack files, digest {digest[:12]}…)")


def verify_history() -> None:
    """Historical manifests stay as written. P2-T may differ only in search_yield_ab.py."""
    document = load(BASELINE)
    counts = document["history_manifests"]
    pins = document["history_manifest_sha256"]
    names = [f"P2-{name}" for name in "IJKLMNOPQRSTU"]
    require(list(counts) == names and list(pins) == names, "history set drifted")
    for phase in names:
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/{phase}/manifest.sha256"
        require(sha256(path) == pins[phase], f"{phase} manifest bytes were rewritten")
        entries = [line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        require(len(entries) == counts[phase], f"{phase} historical manifest count drifted")
        mismatched = [name for digest, name in entries if sha256(ROOT / name) != digest]
        if phase == "P2-I":
            require(mismatched == P2I_LIVE_DRIFT, f"P2-I live drift changed: {mismatched}")
            worker_pin = next(digest for digest, name in entries if name == "tcenv/v2_worker.py")
            orchestrator_pin = next(digest for digest, name in entries if name == "taic/orchestrator.py")
            require(worker_pin == document["p2i_worker_pin"], "P2-I's worker pin was rewritten")
            require(orchestrator_pin == document["p2i_orchestrator_pin"] == PREVIOUS_ORCHESTRATOR_SHA, "P2-I's orchestrator pin was rewritten")
            require(sha256(ROOT / "taic/orchestrator.py") != orchestrator_pin, "live orchestrator still matches the P2-I pin")
            print(f"OK: P2-I manifest unchanged ({counts[phase]} files); live drift is {', '.join(mismatched)}")
            continue
        if phase == "P2-M":
            require(mismatched == P2M_LIVE_DRIFT, f"P2-M live drift changed: {mismatched}")
            pinned = next(digest for digest, name in entries if name == "tests/shared/test_search_v2_bundle_prompt.py")
            require(pinned == document["p2m_prompt_test_pin"] == P2M_PROMPT_TEST_PIN, "P2-M's prompt-test pin was rewritten")
            print(f"OK: P2-M manifest unchanged ({counts[phase]} files); live drift is {mismatched[0]}")
            continue
        if phase == "P2-S":
            require(sorted(mismatched) == sorted(document["p2s_live_drift"]), f"P2-S live drift changed: {mismatched}")
            print(f"OK: P2-S manifest unchanged ({counts[phase]} files); live drift is {', '.join(sorted(mismatched))}")
            continue
        if phase == "P2-T":
            require(mismatched == P2T_LIVE_DRIFT, f"P2-T live drift changed: {mismatched}")
            search_pin = next(digest for digest, name in entries if name == "experiments/search_yield_ab.py")
            overlay_pin = next(digest for digest, name in entries if name == "taic/experience_overlay.py")
            require(search_pin == document["p2t_search_yield_pin"], "P2-T's search_yield pin was rewritten")
            require(overlay_pin == document["p2t_experience_overlay_pin"] == P2T_EXPERIENCE_OVERLAY_PIN, "P2-T's experience_overlay pin was rewritten")
            require(sha256(ROOT / "experiments/search_yield_ab.py") == document["search_yield_ab_sha256"], "live search_yield hash drifted")
            require(sha256(ROOT / "taic/experience_overlay.py") == EXPERIENCE_OVERLAY_SHA != overlay_pin, "live experience_overlay hash drifted or still matches the P2-T pin")
            kernel_pin = next(digest for digest, name in entries if name == "taic/kernel_agent.py")
            require(sha256(ROOT / "taic/kernel_agent.py") == kernel_pin, "taic/kernel_agent.py left the P2-T pin")
            print(f"OK: P2-T manifest unchanged ({counts[phase]} files); live drift is {', '.join(mismatched)}")
            continue
        require(mismatched == [], f"{phase} historical file changed: {mismatched}")
        print(f"OK: {phase} historical manifest remains byte-identical ({counts[phase]} files)")


def verify_historical_anchors() -> None:
    """P2-T and P2-U still pin the pre-d338849 HEAD and orchestrator. Do not rewrite them."""
    for phase in ("P2-T", "P2-U"):
        directory = P2V.parent / phase
        baseline = load(directory / "acceptance-baseline.json")
        source = (directory / "verify.py").read_text(encoding="utf-8")
        require(baseline["parent_head"] == PREVIOUS_PARENT_HEAD, f"{phase} parent HEAD was rewritten")
        require(baseline["orchestrator_sha256"] == PREVIOUS_ORCHESTRATOR_SHA, f"{phase} orchestrator pin was rewritten")
        require(PREVIOUS_PARENT_HEAD in source, f"{phase} verifier no longer names the old parent HEAD")
        require("orchestrator.py left the P2-I pin" in source, f"{phase} verifier dropped its orchestrator pin check")
        require(PARENT_HEAD not in source, f"{phase} verifier was retargeted to the new HEAD")
        require(ORCHESTRATOR_SHA not in source and ORCHESTRATOR_SHA not in json.dumps(baseline), f"{phase} absorbed the new orchestrator pin")
    stored = load(P2V.parent / "P2-T" / "evidence" / "dry-run-plan.json")
    require(stored["git_head"] == PREVIOUS_PARENT_HEAD, "P2-T dry-run git_head was rewritten")
    print("OK: P2-T and P2-U keep the pre-d338849 HEAD and orchestrator pins")


def verify_overlay_text() -> None:
    from experiments.experience_catalog import OVERLAY_MARK
    from experiments.experience_compact import project_compact_overlay, read_compact_overlay

    text = read_compact_overlay()
    require(text == project_compact_overlay() == COMPACT.read_text(encoding="utf-8"), "compact file drifted from the projection")
    encoded = text.encode("utf-8")
    require(len(encoded) == 974, f"compact overlay is {len(encoded)} bytes")
    require(hashlib.sha256(encoded).hexdigest() == "fd72abf078d9d5a7cd2118ee5263a3c83e5695c044c0bc0cc6ca86758a0aa3d1", "compact overlay sha drifted")
    lowered = text.casefold()
    for banned in BANNED:
        require(banned.casefold() not in lowered, f"compact overlay contains {banned}")
    for phrase in REQUIRED:
        require(phrase in text, f"compact overlay lost {phrase!r}")
    require(text.count(OVERLAY_MARK) == 1 and text.count("## ") == 1, "compact overlay marker or heading drifted")
    require(sha256(FULL) == "8e3b9f975cb3988e03af09a12e131bfdb8a699631c8780d544d3836f4bdb97cc", "full overlay bytes changed")
    print("OK: compact overlay is 974 bytes and stays on the gemm_silu small lesson")


def verify_evidence() -> None:
    from experiments.experience_catalog import OVERLAY_MARK
    from experiments.experience_compact import (
        CompactProjectionError,
        approval_document,
        assert_compact_execution_allowed,
        build_compact_plan,
        execute_compact_plan,
        measure_render,
        p2u_boundary,
        projection_document,
        render_comparison,
    )
    from experiments.search_yield_ab import build_experience_plan, main as search_main, render_experience_plan
    from experiments.experience_compact import main as compact_main

    projection = projection_document()
    require(load(EVIDENCE / "projection.json") == projection, "projection.json does not match the live projection")
    require(projection["lesson_id"] == "gemm-silu-reg" and projection["size"] == "small", "projection left the small lesson")
    require(projection["reg_fresh_us_text"] == "1.8252" and projection["two_stage_floor_us_text"] == "8.7713", "formatted P2-O numbers drifted")
    require(projection["confirmation_seed"] == 3775173618909035 and projection["identity"] == "match", "fresh identity drifted")
    require(projection["search_seed"] == 7 and projection["not_a_candidate"] is True, "probe boundary drifted")
    require(projection["omitted_size"] == "paper" and projection["candidate_tiling_specified"] is False, "paper or tiling leaked into the projection")
    require("conv2d-paper-shared-u" in projection["omitted_lesson_ids"], "conv2d lesson was not omitted by id")
    require("bitdelta-reg" in projection["omitted_lesson_ids"] and "seq-serial-handoff" in projection["omitted_lesson_ids"], "unrelated lessons were not omitted")
    require(projection["speedup_published"] is False and projection["publication_allowed"] is False, "projection publishes a speedup")
    approval = approval_document()
    require(load(EVIDENCE / "approval.json") == approval, "approval.json does not match the live blank approval")
    for key in ("approved_by", "delta", "tolerable_overrun", "execution_record"):
        require(approval[key] is None, f"approval field {key} was filled")
    require(approval["executed"] is False and approval["inherits_p2u_approval"] is False, "approval claims execution or reuse")
    prior = p2u_boundary()
    require(prior["verdict"] == "no_gain" and prior["adopt_overlay"] is False and prior["reused"] is False, "P2-U boundary was reused")
    require(prior["delta"] == 0.01 and prior["approved_by"] == "kuafu", "P2-U approval record drifted")
    plan = build_compact_plan()
    require(load(EVIDENCE / "dry-run-plan.json") == plan, "dry-run plan does not match the live builder")
    require(plan["new_experiment"] is True and plan["not_plan_4"] is True and plan["plan_4_started"] is False, "dry-run is not a new experiment")
    require(plan["inherits_p2u_approval"] is False and plan["adopt_compact_overlay"] is False, "dry-run adopts an overlay")
    require(plan["approved_by"] is None and plan["delta"] is None and plan["tolerable_overrun"] is None, "dry-run copied an approval")
    require(plan["executed"] is False and plan["paid_search"] == "未授权/未执行", "dry-run claims execution")
    require(plan["p2u_verdict"] == "no_gain" and plan["p2u_adopt_overlay"] is False, "dry-run rewrote the P2-U verdict")
    require(plan["overlay_kind"] == "compact" and plan["compact_overlay_sha256"] != plan["full_overlay_sha256"], "compact and full overlays are not distinguished")
    inner = plan["experience_plan"]
    require(inner["executed"] is False and inner["delta"] is None and inner["approved_by"] is None, "inner plan is no longer unapproved")
    require(inner["tasks"] == ["gemm_silu"] and inner["dtype"] == "fp16" and inner["size"] == "small", "inner plan left the projected shape")
    conclusion = load(P2U / "evidence" / "conclusion.json")
    require(conclusion["verdict"] == "no_gain" and conclusion["adopt_overlay"] is False, "P2-U conclusion was rewritten")
    require(conclusion["publication_allowed"] is False, "P2-U conclusion now publishes")
    sizes = measure_render()
    require(sizes["control_bytes"] == 13415 and sizes["full_treatment_bytes"] == 25469, "control or full treatment bytes drifted")
    require(sizes["compact_treatment_bytes"] == 14389 and sizes["compact_overlay_bytes"] == 974, "compact render bytes drifted")
    require(sizes["full_added_bytes"] == 12054 and sizes["added_bytes_removed"] == 11080, "reduction integers drifted")
    require(repr(sizes["added_bytes_removed"] / sizes["full_added_bytes"]) == "0.9191969470715116", "reduction ratio drifted")
    out = ROOT / "runs" / "_p2v_verify_render"
    if out.exists():
        shutil.rmtree(out)
    try:
        manifest = render_experience_plan(out, inner)
        comparison = render_comparison(manifest)
        require(load(EVIDENCE / "render-comparison.json") == comparison, "live render does not match the archived comparison")
        require(comparison["overlay_copies"] == {"control": 0, "treatment": 1}, "copy counts drifted")
        require(comparison["rendered_experience_sha256"] == sizes["compact_treatment_sha256"], "rendered sha drifted")
        require(comparison["system_prompt_contains_overlay"] is False, "the system prompt contains the overlay")
        control = out / "render-only" / "control" / "gemm_silu_small_rep1" / "agent_work" / "init" / "experience.md"
        treatment = out / "render-only" / "treatment" / "gemm_silu_small_rep1" / "agent_work" / "init" / "experience.md"
        require(control.read_text(encoding="utf-8").count(OVERLAY_MARK) == 0, "control render contains the overlay")
        require(treatment.read_text(encoding="utf-8").count("overlay_kind: compact") == 1, "treatment render lost the compact marker")
        require(list(out.rglob("result.json")) == [], "render-only wrote a search result")
        task_shas = [cell["normalized_task_sha256"] for cell in manifest["cells"]]
        require(len(set(task_shas)) == 1 and task_shas[0] == NORMALIZED_TASK_SHA, "normalized TASK.md is not shared")
        control_task = (control.parent / "TASK.md").read_text(encoding="utf-8")
        treatment_task = (treatment.parent / "TASK.md").read_text(encoding="utf-8")
        require("eval-v2" in control_task and "eval-v2" in treatment_task, "TASK.md lost eval-v2")
        require("--validate" not in control_task and "--validate" not in treatment_task, "TASK.md still requires --validate")
        require(comparison["control_bytes"] == 13415 and comparison["compact_treatment_bytes"] == 14389, "experience bytes moved with the task text")
        require(comparison["rendered_experience_sha256"] == "93b65a30a3b1de5addaabda1e1249cf72295e7396d6627981baaa5511c748c81", "experience render sha moved")
    finally:
        shutil.rmtree(out, ignore_errors=True)

    def explode(*_args, **_kwargs):
        raise AssertionError("runner was called")

    refused = ROOT / "runs" / "_p2v_verify_execute"
    wrong = ROOT / "runs" / "_p2v_verify_wrong"
    paper = ROOT / "runs" / "_p2v_verify_paper"
    for path in (refused, wrong, paper):
        shutil.rmtree(path, ignore_errors=True)
    try:
        try:
            execute_compact_plan(runner=explode, approved_by="kuafu", delta=0.01)
        except CompactProjectionError as exc:
            require("not approved" in str(exc), "compact execution refusal lost its reason")
        else:
            raise AssertionError("execute_compact_plan returned")
        require(
            compact_main([
                "--out", str(refused), "--execute", "--allow-spend", "--allow-gpu",
                "--approved-by", "kuafu", "--delta", "0.01",
            ]) == 2,
            "compact --execute did not refuse",
        )
        require(not refused.exists(), "compact --execute created a directory")
        require(
            search_main([
                "--out", str(refused), "--execute", "--allow-spend", "--allow-gpu",
                "--approved-by", "kuafu", "--delta", "0.01",
                "--factor", "experience-overlay", "--overlay", str(COMPACT),
                "--tasks", "gemm_silu", "--dtype", "fp16", "--size", "small",
            ], runner=explode) == 2,
            "shared --execute did not refuse the compact overlay",
        )
        require(not refused.exists(), "shared --execute created a directory")
        require(
            search_main([
                "--out", str(wrong), "--dry-run", "--factor", "experience-overlay",
                "--overlay", str(COMPACT), "--tasks", "conv2d", "--size", "small",
            ], runner=explode) == 2,
            "wrong task was accepted",
        )
        require(not wrong.exists(), "wrong task created a directory")
        require(
            search_main([
                "--out", str(paper), "--render-only", "--factor", "experience-overlay",
                "--overlay", str(COMPACT), "--tasks", "gemm_silu", "--dtype", "fp16", "--size", "paper",
            ], runner=explode) == 2,
            "paper size was accepted",
        )
        require(not paper.exists(), "paper size created a directory")
        full_plan = build_experience_plan(
            overlay_sha256=sha256(FULL),
            overlay_path=FULL_REL,
            git_head="abc",
            pairs=1,
        )
        assert_compact_execution_allowed(full_plan)
    finally:
        for path in (refused, wrong, paper):
            shutil.rmtree(path, ignore_errors=True)
    require(OVERLAY_MARK not in EXPERIENCE.read_text(encoding="utf-8"), "the overlay was installed into experience.md")
    print("OK: blank approval, live compact render, and unpaid execute refusal")


def verify_docs() -> None:
    text = DOC.read_text(encoding="utf-8")
    for phrase in (
        "gemm_silu",
        "fp16",
        "small",
        "apple-m4-max",
        "gemm-silu-reg",
        "1.8252",
        "8.7713",
        "3775173618909035",
        "974",
        "fd72abf078d9d5a7cd2118ee5263a3c83e5695c044c0bc0cc6ca86758a0aa3d1",
        "13415",
        "25469",
        "14389",
        "93b65a30a3b1de5addaabda1e1249cf72295e7396d6627981baaa5511c748c81",
        "12054",
        "12059",
        "11080",
        "0.9191969470715116",
        "adopt_compact_overlay=false",
        "publication_allowed=false",
        "simulation-only",
        "未授权/未执行",
        "Plan 4 未开始",
        "inherits_p2u_approval=false",
        "prepare_session",
        "不指定候选 tiling",
        "1ce450fb9804e1efdd624bb3d51542d10f6ea35fd68d0051170e4a46686cd7f8",
        "413df71af15af76524aaea28ed80e581543b56365b3ff0be4185ed25734fbca6",
        PARENT_HEAD,
        PREVIOUS_PARENT_HEAD,
        ORCHESTRATOR_SHA,
        PREVIOUS_ORCHESTRATOR_SHA,
        EXPERIENCE_OVERLAY_SHA,
        P2T_EXPERIENCE_OVERLAY_PIN,
        P2M_PROMPT_TEST_PIN,
        NORMALIZED_TASK_SHA,
        "selfcheck_allow_validate",
        "不能把 overlay 长度说成唯一变量",
        "历史锚点漂移",
        "review-r1-superseded.json",
    ):
        require(phrase in text, f"contract lost {phrase!r}")
    for phrase in ("adopt_compact_overlay=true", "publication_allowed=true", "Plan 4 已经开始", "verdict=treatment_wins"):
        require(phrase not in text, f"contract claims {phrase}")
    bullets = [line for line in AGENTS.read_text(encoding="utf-8").splitlines() if line.startswith("- **P2-V：")]
    require(len(bullets) == 1, "AGENTS.md must contain exactly one P2-V status bullet")
    bullet = bullets[0]
    for phrase in (
        "974",
        "14389",
        "11080",
        "0.9191969470715116",
        "fd72abf078d9d5a7cd2118ee5263a3c83e5695c044c0bc0cc6ca86758a0aa3d1",
        "93b65a30a3b1de5addaabda1e1249cf72295e7396d6627981baaa5511c748c81",
        "adopt_compact_overlay=false",
        "未授权/未执行",
        "docs/架构/compact-experience-overlay.md",
        "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/verify.py",
        PARENT_HEAD,
        ORCHESTRATOR_SHA,
        "selfcheck_allow_validate=False",
        "不能把 overlay 长度说成唯一变量",
        "历史锚点漂移",
        "review-r1-superseded.json",
    ):
        require(phrase in bullet, f"AGENTS.md P2-V bullet lost {phrase!r}")
    require("adopt_compact_overlay=true" not in bullet, "AGENTS.md adopts the compact overlay")
    print("OK: contract and AGENTS.md record the unpaid compact overlay")


def verify_review() -> None:
    superseded_path = P2V / SUPERSEDED_REVIEW
    require(sha256(superseded_path) == SUPERSEDED_REVIEW_SHA, "the pre-d338849 review bytes were rewritten")
    superseded = load(superseded_path)
    require(superseded["done"] is True, "the preserved pre-d338849 signature is no longer the original done=true")
    require(PARENT_HEAD not in json.dumps(superseded), "the preserved signature was edited to cover the new HEAD")
    require(ORCHESTRATOR_SHA not in json.dumps(superseded), "the preserved signature was edited to cover the new orchestrator")
    document = load(P2V / "review.json")
    comparison = load(EVIDENCE / "render-comparison.json")
    baseline = load(BASELINE)
    conclusion = load(P2U / "evidence" / "conclusion.json")
    require(document["epoch"] == "p2v-compact-experience-overlay-1", "the review belongs to another epoch")
    require(document["supersedes"] == SUPERSEDED_REVIEW, "the review does not supersede the pre-d338849 signature")
    require(document["historical_signature_covers_this_head"] is False, "the old done=true still covers this HEAD")
    require(document["parent_head"] == baseline["parent_head"] == PARENT_HEAD, "the review is not on the new HEAD")
    require(document["orchestrator_sha256"] == baseline["orchestrator_sha256"] == ORCHESTRATOR_SHA, "the review orchestrator pin drifted")
    require(document["normalized_task_sha256"] == NORMALIZED_TASK_SHA, "the review task sha drifted")
    require(document["experience_render_changed"] is False, "the review says the experience render changed")
    require(document["task_text_shared_by_both_arms"] is True, "the review splits the task text")
    require(document["task_text_requires_eval_v2_validate"] is False, "the review still requires eval-v2 --validate")
    require(document["cross_experiment_only_variable_is_overlay_length"] is False, "the review claims overlay length is the only cross-experiment variable")
    require(document["done"] is True, "the independent signature is not done")
    require(document["blocking"] == [], "the signed review still lists a blocker")
    require(document["execute_not_invoked"] is True and document["started_search"] is False, "the review started a search")
    require(document["unpaid_execute_refused"] is True and document["render_replayed"] is True, "the review skipped a check")
    require(document["projection_checked"] is True, "the review did not check the projection")
    require(document["publication_allowed"] is False and document["npu_simulated_only"] is True, "the review publishes or opens NPU")
    require(document["paid_search"] == "未授权/未执行", "the review claims the search ran")
    require(document["plan_4_started"] is False and document["adopt_compact_overlay"] is False, "the review adopts the overlay or starts Plan 4")
    require(document["inherits_p2u_approval"] is False, "the review reuses the P2-U approval")
    require(document["p2u_verdict"] == conclusion["verdict"] == "no_gain", "the review rewrote the P2-U verdict")
    require(document["p2u_adopt_overlay"] is conclusion["adopt_overlay"] is False, "the review rewrote adopt_overlay")
    require(document["reviewer"]["edited_tree"] is False, "the reviewer edited the tree")
    require(document["reviewer"]["paid_model"] is False and document["reviewer"]["cost_usd"] == 0, "the review spent model budget")
    require(document["reviewer"]["method"] == "verifier recompute", "the review is not a verifier recompute")
    require(document["remaining_blockers"] == REMAINING, "remaining blockers changed")
    require(document["control_experience_bytes"] == 13415 == comparison["control_bytes"], "control bytes do not match the live render")
    require(document["full_treatment_bytes"] == 25469 == comparison["full_treatment_bytes"], "full treatment bytes do not match")
    require(document["compact_treatment_bytes"] == 14389 == comparison["compact_treatment_bytes"], "compact treatment bytes do not match")
    require(document["compact_overlay_bytes"] == 974 == comparison["compact_overlay_bytes"], "compact overlay bytes do not match")
    require(document["full_added_bytes"] == 12054 and document["added_bytes_removed"] == 11080, "reduction integers do not match")
    require(document["added_bytes_reduction"] == "0.9191969470715116", "reduction ratio does not match")
    require(document["overlay_copies"] == {"control": 0, "treatment": 1} == comparison["overlay_copies"], "copy counts do not match")
    require(document["rendered_experience_sha256"] == comparison["rendered_experience_sha256"], "rendered sha does not match")
    require(document["compact_overlay_sha256"] == comparison["compact_overlay_sha256"], "compact sha does not match")
    require(document["prompt_sha1"] == baseline["prompt_sha1"], "prompt sha1 does not match the baseline")
    require(document["pack_closure_sha256"] == baseline["pack_closure_sha256"], "pack digest does not match the baseline")
    notes = document["notes"]
    for phrase in (
        "13415", "25469", "14389", "974", "11080", "0.9191969470715116", "93b65a30", "fd72abf0",
        "no_gain", "1.8252", "8.7713", PARENT_HEAD, ORCHESTRATOR_SHA, NORMALIZED_TASK_SHA,
        "overlay length is not the only variable", "review-r1-superseded.json",
    ):
        require(phrase in notes, f"notes omit {phrase}")
    print("OK: P2-V review (verifier recompute on d338849, done=true, adopt_compact_overlay=false)")


def pytest_nodes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *TESTS],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dict(os.environ),
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"P2-V CPU tests failed:\n{output}")
    require(__import__("re").search(rf"(?<!\d){CPU_NODES} passed(?!\d)", output) is not None, f"expected {CPU_NODES} passed:\n{output}")
    require(" skipped" not in output and "xfail" not in output and "deselected" not in output, f"skip/xfail/deselect is not accepted:\n{output}")
    print(f"OK: P2-V CPU tests ({CPU_NODES} passed)")


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
    verify_historical_anchors()
    verify_overlay_text()
    verify_evidence()
    verify_docs()
    verify_review()
    pytest_nodes()
    print("OK: P2-V acceptance (compact overlay unpaid; adopt_compact_overlay=false; publication_allowed=false)")


if __name__ == "__main__":
    main()
