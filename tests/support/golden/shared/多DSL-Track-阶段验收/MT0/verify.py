#!/usr/bin/env python3
"""Independent structural checks for the read-only MT0 evidence package."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TEST_NODE = "tests/shared/test_mt0_dsl_selection.py"
TEST_NODES = tuple(
    json.loads((HERE / "nodes.json").read_text())["positive"]
    + json.loads((HERE / "nodes.json").read_text())["negative"]
)


def _load(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def verify_document(document: dict) -> None:
    if document.get("schema_id") != "shapeforge.mt0.dsl-selection.1":
        raise ValueError("wrong MT0 schema")
    if document.get("status") != "blocked":
        raise ValueError("MT0 cannot pass without two qualified DSLs")
    if document.get("audited_candidate_count") != 2 or len(document.get("candidates", [])) != 2:
        raise ValueError("exactly two concrete candidates must be audited")
    if document.get("qualified_candidate_count") != 0 or document.get("qualified_candidate_ids"):
        raise ValueError("this round has no qualified candidate")
    expected = {
        "tilelang": ("0.1.15", "a35f8ddf45eba16c21211ec8822d56ce5363036f"),
        "halide": ("22.1.0", "a8c592b056c06eeeeb064778e6f83e3c6002338f"),
    }
    for row in document["candidates"]:
        dsl_id = row["dsl_id"]
        if (row["version"], row["commit"]) != expected[dsl_id]:
            raise ValueError(f"unpinned or unexpected DSL identity: {dsl_id}")
        if row["candidate_execution_model"] != "embedded_host_code" or row["availability"] != "interface_only":
            raise ValueError("embedded candidates cannot be marked available without isolation evidence")
        if row["qualified"] is not False or not row["blockers"]:
            raise ValueError("an unqualified candidate is being hidden")
        if row["reuses_current_msl_single_or_bundle_contract"] or row["reuses_current_msl_single_or_bundle_adapter"]:
            raise ValueError("an existing MSL Track is being reused as the alleged independent route")
        if row["build_artifacts"]["local_observed"] or row["local_frontend"]["installed"]:
            raise ValueError("local frontend/build evidence was fabricated")
        for key in ("frontend", "optimizer_scheduler", "delivery_identity", "shared_substrate_boundary"):
            if not row["independence_evidence"].get(key):
                raise ValueError(f"missing independent-route evidence: {dsl_id}.{key}")
        if not row["build_artifacts"]["upstream_observed"] or not row["device_execution_path"]:
            raise ValueError("missing real upstream artifact/execution evidence")
    for stage in ("MT1b", "MT2", "MT3", "MT4", "MT6", "MT7"):
        if document["next_stages"].get(stage) != "blocked":
            raise ValueError(f"{stage} must remain blocked")
    if document["next_stages"].get("MT1a") != "allowed_independent_work" or document["next_stages"].get("MT5") != "allowed_independent_work":
        raise ValueError("MT0 blocked must not block the independent MT1a/MT5 packages")
    boundary = document["evidence_boundary"]
    for key in ("local_kernel_compilation", "gpu_or_npu_execution", "model_or_paid_search", "publication_allowed"):
        if boundary.get(key) is not False:
            raise ValueError(f"unauthorized evidence claim: {key}")
    if boundary.get("performance_claim") is not None:
        raise ValueError("MT0 has no performance result")
    if boundary.get("local_frontend_help_or_version_probe") is not False:
        raise ValueError("no installed DSL frontend produced a version/help probe")


def verify_upstream() -> None:
    ledger = _load("upstream-map.json")
    if len(ledger["sources"]) < 10:
        raise ValueError("upstream ledger is incomplete")
    for row in ledger["sources"]:
        if hashlib.sha256(row["evidence_excerpt"].encode()).hexdigest() != row["evidence_excerpt_sha256"]:
            raise ValueError(f"source excerpt hash mismatch: {row['source_id']}")
        if not row["retrieval_reference"]:
            raise ValueError("source retrieval reference is missing")


def verify_manifest() -> None:
    for line in (HERE / "manifest.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = HERE / relative
        if not path.is_file():
            raise ValueError(f"missing manifest member: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"manifest digest mismatch: {relative}")


def self_test() -> None:
    original = _load("dsl-selection.json")
    mutations = []
    available = copy.deepcopy(original)
    available["candidates"][0]["availability"] = "available"
    mutations.append(available)
    alias = copy.deepcopy(original)
    alias["candidates"][0]["reuses_current_msl_single_or_bundle_adapter"] = True
    mutations.append(alias)
    passing = copy.deepcopy(original)
    passing["status"] = "complete"
    mutations.append(passing)
    fake_local = copy.deepcopy(original)
    fake_local["candidates"][0]["build_artifacts"]["local_observed"] = ["invented.metallib"]
    mutations.append(fake_local)
    graph = copy.deepcopy(original)
    graph["next_stages"]["MT2"] = "allowed"
    mutations.append(graph)
    for index, changed in enumerate(mutations):
        try:
            verify_document(changed)
        except ValueError:
            continue
        raise AssertionError(f"verifier self-test mutation {index} was accepted")


def main() -> int:
    document = _load("dsl-selection.json")
    verify_document(document)
    verify_upstream()
    self_test()
    if "--check" in sys.argv:
        verify_manifest()
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-n", "0", *TEST_NODES], cwd=ROOT, check=False
        )
        if completed.returncode != 0:
            raise ValueError(f"fixed MT0 pytest node failed with exit {completed.returncode}; read logs/pytest/triage/triage_summary.md")
    print(json.dumps({
        "verifier_ok": True,
        "stage": "MT0",
        "status": "blocked",
        "audited_dsls": ["tilelang@0.1.15", "halide@22.1.0"],
        "qualified_dsl_count": 0,
        "fixed_nodes": {TEST_NODE: 11},
        "self_test_negative_mutations": 5,
        "device_or_paid_work": False,
        "next_authorization": "confirm candidates and isolation, then offline pinned frontend probes before any MT1b decision",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
