#!/usr/bin/env python3
"""P2-S acceptance: verified experience catalog and an unexecuted paired dry-run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2S = Path(__file__).resolve().parent
P2R_PACK = P2S.parent / "P2-R" / "pack-closure.json"
P2M_EPOCH = P2S.parent / "P2-M" / "prompt-epoch.json"
MANIFEST = P2S / "manifest.sha256"
BASELINE = P2S / "acceptance-baseline.json"
EVIDENCE = P2S / "evidence"
EXPERIENCE = ROOT / "targets/apple-m4-max/experience.md"
# P2-I's closed manifest still records the pre-fix worker. Later phases do not rewrite it.
P2I_WORKER_PIN = "39ce59320d35066489d88db4eac213c65764e6c57f8932b16fbd6e84a01c370c"
LIVE_WORKER_SHA = "a083a36272e31bb9f9edaa60f5df9a57a927facf150582e7b51f087ac65cbe33"
TESTS = (
    "tests/shared/test_experience_search_prep.py",
    "tests/shared/test_search_yield_ab.py",
)
CPU_NODES = 19
DELIVERABLES = (
    "docs/架构/verified-experience-search-prep.md",
    "experiments/experience_catalog.py",
    "experiments/search_yield_ab.py",
    "tests/shared/test_experience_search_prep.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/S.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/acceptance-baseline.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/verify.py",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/review-r1-rejected.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/review.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/evidence/catalog.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/evidence/experience-overlay.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/evidence/prompt-boundary.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/evidence/approval.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S/evidence/dry-run-plan.json",
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
    print(f"OK: wrote P2-S manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-S manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-S manifest mismatch: {name}")
    print(f"OK: P2-S manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-s-experience-search-prep-1", "wrong P2-S epoch")
    require(document["parent_head"] == "a9c1dccce636107c2852995578b05611bf19865b", "P2-S parent HEAD drifted")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    require(head == document["parent_head"], "the working tree is not on the recorded parent HEAD")
    require(document["publication_allowed"] is False, "P2-S must not publish")
    require(document["npu_simulated_only"] is True, "P2-S must keep NPU simulation-only")
    require(document["plan_3_complete"] is False, "P2-S must not claim Plan 3 is complete")
    require(document["paid_search"] == "未授权/未执行", "P2-S must not claim the search ran")
    require(document["prompt_changed"] is False, "P2-S records a prompt change")
    live_prompt = current_prompt_sha1()
    require(document["prompt_sha1"] == document["previous_prompt_sha1"] == live_prompt, "P2-S prompt fingerprint moved")
    epoch = json.loads(P2M_EPOCH.read_text(encoding="utf-8"))
    require(live_prompt == epoch["current_prompt_sha1"], "P2-S moved off the P2-M prompt fingerprint")
    require(live_prompt != epoch["previous_prompt_sha1"], "P2-M's previous prompt sha was rewritten")
    require(
        document["trust_toml_sha256"]
        == hashlib.sha256((ROOT / "targets/apple-m4-max/trust.toml").read_bytes()).hexdigest(),
        "P2-S changed the trust file",
    )
    previous = json.loads(P2R_PACK.read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(files) == list(previous["files"]), "P2-S task-pack file set drifted from P2-R")
    require(
        digest == document["pack_closure_sha256"] == previous["sha256"],
        "P2-S task-pack digest drifted from P2-R",
    )
    require(len(files) == document["pack_closure_files"] == 79, "P2-S pack file count drifted")
    print(f"OK: P2-S baseline (prompt unchanged, 79 pack files, digest {digest[:12]}…)")


def verify_history() -> None:
    """Historical manifests stay as written. P2-I still pins the pre-fix worker."""
    counts = json.loads(BASELINE.read_text(encoding="utf-8"))["history_manifests"]
    require(set(counts) == {f"P2-{name}" for name in ("I", "J", "K", "L", "M", "N", "O", "P", "Q", "R")}, "history set drifted")
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
    catalog = json.loads((EVIDENCE / "catalog.json").read_text(encoding="utf-8"))
    overlay = (EVIDENCE / "experience-overlay.md").read_text(encoding="utf-8")
    boundary = json.loads((EVIDENCE / "prompt-boundary.json").read_text(encoding="utf-8"))
    approval = json.loads((EVIDENCE / "approval.json").read_text(encoding="utf-8"))
    plan = json.loads((EVIDENCE / "dry-run-plan.json").read_text(encoding="utf-8"))
    require(catalog["teachable_ids"] == [
        "conv2d-paper-shared-u",
        "conv2d-small-single-stage",
        "gemm-silu-reg",
        "bitdelta-reg",
        "gemm-splitk-stop",
        "seq-serial-handoff",
    ], "teachable set drifted")
    require(catalog["publication_allowed"] is False and catalog["paid_search"] == "未授权/未执行", "catalog claims a search or a publication")
    require("这份文字是 P2-S 的冻结叠加层" in overlay, "overlay lost its uninstalled marker")
    require("这份文字是 P2-S 的冻结叠加层" not in EXPERIENCE.read_text(encoding="utf-8"), "the overlay was installed into experience.md")
    require("同一份 device 源码" not in overlay, "the overlay still collapses KDA and varlen into one device source")
    require("same device source" not in json.dumps(catalog), "the catalog still claims one device source")
    require(catalog["overlay_context_rule"] in overlay, "the overlay dropped the counterexample rule")
    controls = next(item for item in catalog["lessons"] if item["id"] == "seq-serial-handoff")["controls"]
    require(controls["kda"]["device_bytes_identical"] is True, "KDA device bytes were split")
    require(controls["varlen"]["device_bytes_identical"] is False, "varlen device bytes were collapsed")
    require(controls["varlen"]["host_program_present"] is False, "varlen gained a host program")
    require("17622.82" in overlay, "the scalar counterexample left the overlay")
    for token in ("2.765", "1315", "unrunnable", "best_speedup"):
        require(token not in overlay, f"overlay contains excluded token {token}")
    require(boundary["prompt_changed"] is False, "prompt boundary records a change")
    require(boundary["current_prompt_sha1"] == boundary["previous_prompt_sha1"], "prompt boundary shas differ")
    require(boundary["system_prompt_identical_across_arms"] is True, "system prompts differ across arms")
    require(all(boundary["p2m_rendered_prompts_match"].values()), "P2-M render anchor does not match")
    require(sha256(EXPERIENCE) == boundary["experience_file_sha256"], "experience.md no longer matches the recorded hash")
    require(sha256(P2M_EPOCH) == boundary["p2m_prompt_epoch_sha256"], "P2-M prompt-epoch.json was rewritten")
    for key in ("approved_by", "delta", "experiment_id", "execution_record", "gpu_window", "owner_reviewed_on"):
        require(approval[key] is None, f"approval field {key} was filled")
    require(approval["paid_search"] == "未授权/未执行" and approval["executed"] is False, "approval claims execution")
    require(plan["factor"] == "experience-overlay" and plan["executed"] is False, "dry-run plan claims execution")
    require(plan["paid_search"] == "未授权/未执行" and plan["delta"] is None and plan["approved_by"] is None, "dry-run plan was approved")
    require(plan["tasks"] == ["gemm_silu"] and plan["dtype"] == "fp16" and plan["size"] == "small", "experiment task drifted")
    require(plan["pairs"] == 2 and len(plan["cells"]) == 4, "pair count drifted")
    require(plan["nominal_total_usd"] == plan["max_total_usd"] == 12.0, "budget cap drifted")
    require(plan["control"] == plan["treatment"], "search switches differ between arms")
    require(plan["search_eval_seed"] == 7, "search seed drifted")
    require([cell["order"] for cell in plan["cells"]] == ["AB", "AB", "BA", "BA"], "AB/BA order drifted")
    treatment = [cell["experience_overlay_sha256"] for cell in plan["cells"] if cell["arm"] == "treatment"]
    control = [cell["experience_overlay_sha256"] for cell in plan["cells"] if cell["arm"] == "control"]
    require(control == [None, None] and treatment == [plan["overlay_sha256"], plan["overlay_sha256"]], "overlay assignment drifted")
    require(plan["overlay_sha256"] == sha256(EVIDENCE / "experience-overlay.md"), "plan overlay hash does not match the file")
    meaning = plan["stop"]["hard_cap_usd_meaning"]
    require("later cell" in meaning and "not a ceiling the bill cannot pass" in meaning, "hard cap is still described as a bill ceiling")
    require(plan["stop"]["tolerable_overrun_approved"] is None, "tolerable overrun was approved")
    require(plan["stop"]["in_flight_request_may_exceed_soft_budget"] is True, "the soft budget was turned into a bill ceiling")
    seq = next(item for item in catalog["lessons"] if item["id"] == "seq-serial-handoff")
    require(seq["fresh_subject_count"] == 8, "a fresh subject was skipped")
    for row in seq["confirmed_rows"]:
        if row["name"].startswith("kda_"):
            require(row["ac_is_isolated_handoff"] is False and row["same_input_paired_run"] is False, f"{row['name']} still calls A−C an isolated handoff")
            require(len({row["fresh_a_seed"], row["fresh_b_seed"], row["fresh_c_seed"]}) == 3, f"{row['name']} confirmation seeds collapsed")
    bit = next(item for item in catalog["lessons"] if item["id"] == "bitdelta-reg")
    small_ratio = f"{bit['sizes']['small']['best_legal_shared_ratio_over_reg']:.4f}"
    paper_ratio = f"{bit['sizes']['paper']['best_legal_shared_ratio_over_reg']:.4f}"
    require(bit["sizes"]["small"]["exact_ratio_taught"] is False and small_ratio not in overlay, "the unstable small bitdelta ratio was taught")
    require(paper_ratio in overlay, "the paper bitdelta ratio left the overlay")
    require("不是字节流量比" in overlay and "split2_partial_write_read_us_at_peak" in overlay, "Split-K lost its time-equivalent unit")
    require("不能解释为隔离的 handoff 成本" in overlay, "A−C is still presented as isolated handoff")
    require("kda single 不在 decision.sizes" in overlay and "出现在测量决策里" not in overlay, "kda single is still called a decision row")
    require("decision.json" in overlay and "tests/support/seq_handoff.py" in overlay, "a rendered number lost its source file")
    require(overlay.count("\n本机实测:") == 6, "a verified section lost its measurement line")
    contract = (ROOT / "docs/架构/verified-experience-search-prep.md").read_text(encoding="utf-8")
    require("5b23ac934eb5d5e1fe9a73902a8ebf31b8873bab18864d8c6d9dde76205293bf" not in contract, "the contract still cites the stale treatment digest")
    require("20227 字节" not in contract, "the contract still cites the stale treatment length")
    treatment = re.findall(r"过滤后的处理经验 \| (\d+) 字节，`([0-9a-f]{64})`", contract)
    control = re.findall(r"过滤后的对照经验 \| (\d+) 字节，`([0-9a-f]{64})`", contract)
    require(treatment == [(str(boundary["treatment_filtered_bytes"]), boundary["treatment_filtered_experience_sha256"])], "contract treatment digest does not match prompt-boundary.json")
    require(control == [(str(boundary["control_filtered_bytes"]), boundary["control_filtered_experience_sha256"])], "contract control digest does not match prompt-boundary.json")
    require("是否启动后续 cell 的承诺上限" in contract and "硬上限" not in contract, "the contract still calls $12 a hard bill ceiling")
    print("OK: P2-S catalog, uninstalled overlay, blank approval and four-cell dry-run")


def verify_rejected_review() -> None:
    document = json.loads((P2S / "review-r1-rejected.json").read_text(encoding="utf-8"))
    require(document["done"] is False, "the voided signature is still done")
    require(document["prior_done"] is True, "the rejection does not identify the voided signature")
    require(len(document["blocking"]) >= 8, "the rejection dropped findings")
    rendered = json.dumps(document, ensure_ascii=False)
    for token in ("kda_single", "handoff", "本机实测", "5b23ac", "承诺上限"):
        require(token in rendered, f"the rejection record dropped {token}")
    require(document["paid_search"] == "未授权/未执行" and document["plan_3_complete"] is False, "the rejection claims a finished plan")
    print("OK: P2-S r1 signature stays rejected")


def verify_review() -> None:
    document = json.loads((P2S / "review.json").read_text(encoding="utf-8"))
    require(document.get("supersedes") == "review-r1-rejected.json", "the new review does not supersede the rejected signature")
    require(document["epoch"] == "p2-s-experience-search-prep-1", "the review belongs to another epoch")
    require(document["done"] is True, "the preparation review did not sign off")
    require(document["plan_3_complete"] is False, "the review claims Plan 3 is complete")
    require(document["paid_search"] == "未授权/未执行", "the review claims the search ran")
    require(document["publication_allowed"] is False, "the review must not publish")
    require(document["npu_simulated_only"] is True, "the review must keep NPU simulation-only")
    require(document["prompt_changed"] is False and document["overlay_installed"] is False, "the review records a prompt edit")
    require(document["experience_evidence_checked"] is True, "the review did not check experience evidence")
    require(document["prompt_boundary_checked"] is True, "the review did not check the prompt boundary")
    require(document["dry_run_checked"] is True, "the review did not check the dry-run")
    require(document["approval_blank"] is True and document["execute_refused"] is True, "the review missed a blank or the execute refusal")
    require(document["reviewer"]["edited_tree"] is False, "the reviewer edited the tree")
    require(document["reviewer"]["paid_model"] is False and document["reviewer"]["cost_usd"] == 0, "the review spent model budget")
    require(document["blocking"] == [], "the signed-off review still has blocking findings")
    rendered = json.dumps(document, ensure_ascii=False)
    require("未授权/未执行" in rendered, "the review dropped the unexecuted statement")
    require(document["remaining_blockers"], "the review dropped the remaining blockers")
    print("OK: P2-S review (preparation only, search not executed)")


def pytest_nodes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *TESTS],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dict(os.environ),
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"P2-S CPU tests failed:\n{output}")
    require(re.search(rf"(?<!\d){CPU_NODES} passed(?!\d)", output) is not None, f"expected {CPU_NODES} passed:\n{output}")
    require(" skipped" not in output and "xfail" not in output and "deselected" not in output, f"skip/xfail/deselect is not accepted:\n{output}")
    print(f"OK: P2-S CPU tests ({CPU_NODES} passed)")


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
    verify_rejected_review()
    verify_review()
    pytest_nodes()
    print("OK: P2-S acceptance (preparation only; paid search 未授权/未执行; plan_3_complete=false)")


if __name__ == "__main__":
    main()
