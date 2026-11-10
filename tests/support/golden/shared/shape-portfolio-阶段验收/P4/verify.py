#!/usr/bin/env python3
"""P4 acceptance: offline shape-portfolio dispatch over the existing fake-NPU protocols.

P4 starts no model, no search, no GPU and no real NPU.  It checks the archived simulation,
replays it through the fixed CPU nodes, and re-runs the existing remote conformance file.
Skipped fixed nodes fail.  A skipped Go conformance client is accepted only when Go or
openssl is absent; a present toolchain must pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from tcenv.track import REPO_ROOT
from tests.support.boundaries import _AcceptanceRoot

ROOT = _AcceptanceRoot(REPO_ROOT)
P4 = Path(__file__).resolve().parent
MANIFEST = P4 / "manifest.sha256"
EVIDENCE = P4 / "evidence" / "simulation.json"

SIMULATOR = "tests/support/npu_sim/shape_portfolio.py"
TEST = "tests/shared/test_shape_portfolio_npu_sim.py"
SUPPORT_README = "tests/support/README.md"
PLAN = "docs/架构/shape-portfolio实施计划.md"
DOC = "docs/架构/shape-portfolio-阶段验收/P4.md"
VERIFIER = "docs/架构/shape-portfolio-阶段验收/P4/verify.py"
EVIDENCE_NAME = "docs/架构/shape-portfolio-阶段验收/P4/evidence/simulation.json"
CONFORMANCE = "tests/shared/test_remote_conformance.py"
GO_CONFORMANCE = "tests/shared/test_remote_go_conformance.py"

DELIVERABLES = (
    SIMULATOR,
    TEST,
    SUPPORT_README,
    PLAN,
    DOC,
    VERIFIER,
    EVIDENCE_NAME,
)
NODES = (
    "test_p4_exact_dispatch_covers_the_finite_domain_and_keeps_fallback_uncredited",
    "test_p4_source_and_bundle_receipts_bind_the_selected_projection",
    "test_p4_evidence_stays_simulated_and_drops_timing_speedup_and_publication",
    "test_p4_bundle_tamper_and_forged_manifest_digest_are_rejected",
    "test_p4_outside_domain_is_not_submitted_and_does_not_fall_back",
    "test_p4_candidate_view_hides_unseen_keys_and_fresh_seeds",
    "test_p4_document_only_manifest_cannot_be_activated",
    "test_p4_replay_matches_the_archived_evidence",
)
EXPECTED = {
    "tail": ("candidate", "variant-c-tail", True),
    "below": ("fallback", None, False),
    "threshold": ("candidate", "variant-a-bundle", True),
    "above": ("candidate", "variant-a-bundle", True),
    "large": ("candidate", "variant-b-single", True),
}
FORBIDDEN = frozenset(
    {
        "speedup",
        "speedups",
        "candidate_us",
        "elapsed_s",
        "elapsed_ns",
        "device_elapsed_ns",
        "timing",
        "publishable",
        "confirmed",
        "best_speedup",
        "latency_ratio",
    }
)
_PASSED_COUNT = re.compile(r"(\d+) passed\b")


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
    print(f"OK: wrote {MANIFEST.relative_to(REPO_ROOT)} ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P4 manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P4 manifest mismatch: {name}")
    print(f"OK: P4 manifest ({len(entries)} files)")


def _forbid(document: Any, *, where: str = "report") -> None:
    if isinstance(document, Mapping):
        found = FORBIDDEN & set(document)
        require(not found, f"{where} carries forbidden fields {sorted(found)}")
        for key, value in document.items():
            _forbid(value, where=f"{where}.{key}")
    elif isinstance(document, list):
        for index, value in enumerate(document):
            _forbid(value, where=f"{where}[{index}]")


def verify_evidence() -> None:
    document = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    require(document["schema_id"] == "local.shape-portfolio-npu-simulation.1", "wrong P4 schema")
    require(document["simulated"] is True, "P4 evidence is not marked simulated")
    require(document["publication_allowed"] is False, "P4 evidence allows publication")
    npu = document["npu_p0"]
    require(npu["status"] == "blocked", "P4 cleared NPU-P0")
    require(npu["real_device_available"] is False, "P4 claims a real NPU")
    require(npu["correctness_on_device_proven"] is False, "P4 claims device correctness")
    require(npu["performance_on_device_proven"] is False, "P4 claims device performance")
    _forbid(document)
    portfolio = document["portfolio"]
    require(portfolio["op_name"] == "portfolio_pointwise", "P4 used another task")
    require(portfolio["target_id"] == "apple-m4-max", "P4 domain target changed")
    require(portfolio["tune_shards"] == ["fp16/large/benchmark", "fp16/large/golden"], "tune shards drifted")
    require(portfolio["unseen_count"] == 8 and portfolio["required_unseen_count"] == 8, "unseen coverage drifted")
    view = portfolio["candidate_view"]
    require(set(view) == {"schema_id", "portfolio_freeze_digest", "tune_shards"}, "candidate view grew")
    require(view["tune_shards"] == portfolio["tune_shards"], "candidate view is not the tune set")
    protocol = document["protocol"]
    require(protocol["source_api_contract_id"] == "tcenv.remote.npu-source.evaluator-api.1", "source contract changed")
    require(protocol["bundle_api_contract_id"] == "tcenv.remote.npu-bundle.evaluator-api.1", "bundle contract changed")
    require(protocol["fixture_task"] == "sem_attention_odd_scale", "protocol fixture task changed")
    subjects = protocol["subjects"]
    require(
        {item["label"] for item in subjects} == {"variant-a-bundle", "variant-b-single", "variant-c-tail", "domain-fallback"},
        "protocol subjects drifted",
    )
    rows = {row["instance_key"]: row for item in subjects for row in item["dispatches"]}
    require(set(rows) == {f"fp16/{size}/{usage}" for size in EXPECTED for usage in ("golden", "benchmark")}, "domain rows drifted")
    for key, row in rows.items():
        provider, variant_key, credits = EXPECTED[key.split("/")[1]]
        require(row["provider"] == provider and row["variant_key"] == variant_key, f"{key} dispatch drifted")
        require(row["credits_variant"] is credits, f"{key} credit drifted")
        if provider == "fallback":
            require(row["outcome"] == "fallback" and row["variant_id"] is None, f"{key} fallback shape drifted")
        else:
            require(row["outcome"] == "resolved" and row["fallback_id"] is None, f"{key} variant shape drifted")
    for item in subjects:
        require(item["source"]["candidate_digest"]["value"] == item["submitted_sha256"], "source receipt is unbound")
        require(item["bundle"]["candidate_digest"]["value"] == item["bundle"]["artifact_set_digest"], "bundle receipt is unbound")
        require(item["source"]["measurement_kind"] == "simulated", "source result is not simulated")
        require(item["bundle"]["publication_allowed"] is False, "bundle result is publishable")
        require(item["source"]["device_group_id"] == "vendor-a-group-0", "source device group drifted")
        require(item["bundle"]["device_group_id"] == "vendor-b-group-0", "bundle device group drifted")
    omitted = set(document["omitted_protocol_fields"])
    require({"evidence.timing", "report.speedup", "report.candidate_us", "report.elapsed_s"} <= omitted, "protocol timings were not dropped")
    manifest = document["manifest"]
    require(manifest["status"] == "validated" and manifest["activation_generation"] == 0, "manifest is not document-only")
    require(manifest["derived_from_confirmations"] is False, "P4 treated a document as confirmation")
    require(manifest["stable_confirmation_digest_is_fresh_confirmation"] is False, "P4 claims fresh confirmation")
    negatives = document["negatives"]
    require(negatives["bundle_tamper_rejected"] is True, "bundle tamper was not rejected")
    require(negatives["outside_domain"]["protocol_submitted"] is False, "an outside-domain request was submitted")
    require(negatives["outside_domain"]["fallback_id"] is None, "an outside-domain request fell back")
    require("without confirmed sidecars" in negatives["activation_rejected"], "activation failure changed")
    require("not its canonical projection" in negatives["forged_manifest_digest_rejected"], "forged digest failure changed")
    require("must not credit" in negatives["fallback_credit_rejected"], "fallback credit failure changed")
    require("not in the frozen portfolio" in negatives["identity_swap_rejected"], "identity swap failure changed")
    print("OK: P4 archived simulation")


def _passed_count(summary: str, *, where: str) -> int:
    matches = _PASSED_COUNT.findall(summary)
    require(len(matches) == 1, f"{where}: expected exactly one passed count in {summary!r}")
    return int(matches[0])


def _pytest(arguments: list[str], *, where: str, expect_passed: int) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = (result.stdout or "") + (result.stderr or "")
    require(result.returncode == 0, f"{where} failed:\n{output[-8000:]}")
    summary = [line for line in output.splitlines() if re.search(r"\b(passed|failed|error)\b", line)]
    require(summary, f"{where}: no pytest summary")
    last = summary[-1]
    require(" skipped" not in last and "xfailed" not in last and " failed" not in last, f"{where} did not stay green: {last}")
    require(_passed_count(last, where=where) == expect_passed, f"{where}: {last}")
    print(f"OK: {where} ({last})")


def verify_nodes() -> None:
    source = (ROOT / TEST).read_text(encoding="utf-8")
    for name in NODES:
        require(f"def {name}(" in source, f"missing fixed node {name}")
    _pytest([f"{TEST}::{name}" for name in NODES], where="P4 simulation", expect_passed=len(NODES))


def verify_conformance() -> None:
    require(shutil.which("openssl") is not None, "openssl is required for remote conformance")
    _pytest([CONFORMANCE], where="remote conformance", expect_passed=5)
    if shutil.which("go") is None:
        print("OK: Go conformance not run; go is absent")
        return
    _pytest([GO_CONFORMANCE], where="Go remote conformance", expect_passed=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_evidence()
    verify_nodes()
    verify_conformance()
    print("OK: P4 offline NPU shape-portfolio boundary")


if __name__ == "__main__":
    main()
