"""CPU identity, dispatch and archived-evidence checks for P2-J."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from taic.portfolio_confirmation import verify_subject_document
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest, ExactDispatchResolver
from tcenv.portfolio_assembly import check_parent, parent_entries_from_tree
from tcenv.remote.artifact_bundle import MAX_BUNDLE_INLINE_ENTRY_BYTES
from tcenv.target import load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track.expr import evaluate, parse_term
from tests.support.conv2d_small_mma import (
    EVIDENCE,
    FULL_EVAL_SCHEMA,
    PAPER_SHA256,
    REPO_ROOT,
    SMALL_CONFIRMATION,
    SMALL_SHA256,
    TREE,
    assert_recorded_portfolio,
    assembled,
    full_eval_document,
    load_json,
)


def _resolve(size: str, usage: str = "golden"):
    domain, portfolio = assembled()
    contract = load_task_contract(REPO_ROOT / "tasks" / "conv2d", dtype="fp16", size="small")
    instance = instantiate_v2(contract.spec, "fp16", size)
    point = domain.plan.find("fp16", size, usage)
    request = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage)
    evidence = {item.variant_id: "validated" for item in portfolio.variants}
    evidence[portfolio.fallback.fallback_id] = "validated"
    resolver = ExactDispatchResolver(
        domain_plan=domain.plan,
        portfolio=portfolio,
        evidence=evidence,
        spec=contract.spec,
        target=load_target("apple-m4-max"),
        instances=domain.instances,
    )
    return resolver.resolve(request)


def test_small_and_both_use_their_own_specialized_candidates():
    domain, portfolio = assembled()
    assert [item.variant_key for item in portfolio.variants] == ["paper", "small"]
    for size, sha in (("small", SMALL_SHA256), ("paper", PAPER_SHA256)):
        for usage in ("golden", "benchmark"):
            record = _resolve(size, usage)
            assert record.provider == "candidate" and record.outcome == "resolved"
            assert record.variant_id == next(item.variant_id for item in portfolio.variants if item.variant_key == size)
            assert record.fallback_id is None and record.fallback_reason is None
        source = (TREE / "variants" / size / "kernel.metal").read_bytes()
        assert hashlib.sha256(source).hexdigest() == sha
        assert len(source) <= MAX_BUNDLE_INLINE_ENTRY_BYTES
    small = (TREE / "variants" / "small" / "kernel.metal").read_text(encoding="utf-8")
    assert "launch conv2d_wgrid" not in small and "scratch uwin" not in small
    assert "recompute_u" in small and "simdgroup_float8x8" in small
    assert portfolio.fallback.kind == "registered_implementation"


def test_guards_are_exact_and_a_weak_small_guard_overlaps_paper(tmp_path: Path):
    _domain, portfolio = assembled()
    document = load_json(TREE / "coverage.json")
    guards = {item["variant_key"]: parse_term(item["guard"], item["variant_key"]) for item in document["variants"]}
    symbols = {
        "small": {"N": 4, "H": 18, "W": 18, "C": 32, "OC": 64, "KH": 3, "KW": 3, "OH": 16, "OW": 16},
        "paper": {"N": 32, "H": 56, "W": 56, "C": 64, "OC": 128, "KH": 3, "KW": 3, "OH": 54, "OW": 54},
    }
    for key, guard in guards.items():
        assert evaluate(guard, symbols[key]) is True
        other = "paper" if key == "small" else "small"
        assert evaluate(guard, symbols[other]) is False
    root = tmp_path / "tree"
    shutil.copytree(TREE, root)
    weakened = load_json(root / "coverage.json")
    next(item for item in weakened["variants"] if item["variant_key"] == "small")["guard"] = ["eq", "KH", 3]
    (root / "coverage.json").write_text(json.dumps(weakened), encoding="utf-8")
    domain, _portfolio = assembled()
    report = check_parent(parent_entries_from_tree(root, domain), domain)
    assert report.ok is False
    assert any("两个以上变体同时命中" in problem for problem in report.problems)


def test_a_single_source_byte_changes_variant_identity(tmp_path: Path):
    _domain, portfolio = assembled()
    root = tmp_path / "tree"
    shutil.copytree(TREE, root)
    kernel = root / "variants" / "small" / "kernel.metal"
    original = "constant constexpr uint K_TOTAL=KH*KW*C;"
    assert original in kernel.read_text(encoding="utf-8")
    kernel.write_text(kernel.read_text(encoding="utf-8").replace(original, original + " ", 1), encoding="utf-8")
    domain, _portfolio = assembled()
    changed = parse_portfolio(parent_entries_from_tree(root, domain), domain_plan=domain.plan)
    assert next(item for item in changed.variants if item.variant_key == "small").variant_id != next(item for item in portfolio.variants if item.variant_key == "small").variant_id


def test_archived_runs_and_summary_replay_stable_identity_and_gates():
    assert_recorded_portfolio()
    summary = load_json(TREE / "full-eval.json")
    assert summary["schema_id"] == FULL_EVAL_SCHEMA
    assert summary["publication_allowed"] is False and summary["fresh_confirmation"] is False
    small = load_json(EVIDENCE / "small-run.json")
    paper = load_json(EVIDENCE / "paper-run.json")
    assert summary == full_eval_document(small, paper)


def test_small_fresh_subject_is_confirmed_and_matches_the_variant():
    subject = verify_subject_document(load_json(SMALL_CONFIRMATION))
    assert subject["status"] == "confirmed" and subject["identity"]["kind"] == "match"
    assert subject["candidate_sha256"] == SMALL_SHA256
    assert subject["search_seed"] == 7 and subject["confirmation_seed"] != 7
    report = subject["confirmation_report"]
    assert report["status"] == "ok" and report["safety"]["ok"] is True
    cases = report["correctness"]["cases"]
    assert [case["case"] for case in cases] == ["normal", "wide_range", "large", "tiny", "outlier", "ieee"]
    assert all(case["ok"] for case in cases)
    assert report["speedups"]["torch_mps"] > 1.10
    assert report["candidate_us"] <= 1.10 * report["baseline_us"]["mlx"]
    headroom = load_json(EVIDENCE / "headroom.json")
    assert headroom["decision"]["fresh_gate_passed"] is True
    assert headroom["decision"]["two_stage_no_fill_floor_us"] > headroom["decision"]["fresh_candidate_us"] * 1.10
    assert headroom["decision"]["single_stage_floor_us"] < headroom["decision"]["fresh_candidate_us"]
    assert headroom["decision"]["best_baseline_speedup_greater_than_one"] is False
    assert headroom["decision"]["production_speedup_published"] is False
