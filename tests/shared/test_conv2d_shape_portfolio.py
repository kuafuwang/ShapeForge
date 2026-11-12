"""CPU contract for the conv2d small+paper portfolio.

The candidate tree owns the coverage plan.  The harness tests only parse, exact dispatch,
projection bytes and the recorded small-MMA blocker.  Nothing here launches a device.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio, project_provider
from tcenv.portfolio.dispatch import DispatchRequest, ExactDispatchResolver
from tcenv.portfolio_assembly import check_parent, domain_for_sizes, encode_parent, parent_entries_from_tree
from tcenv.remote.artifact_bundle import MAX_BUNDLE_INLINE_ENTRY_BYTES
from tcenv.target import load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track.expr import evaluate, parse_term
from tests.support.conv2d_portfolio import (
    CONFIRMED_PAPER_SHA256,
    PAPER_FIXTURE,
    PAPER_RUN,
    PAPER_SYMBOLS,
    REPO_ROOT,
    SMALL_RUN,
    SMALL_SYMBOLS,
    TREE,
    assert_recorded_portfolio,
    assert_timing_gate,
    load_json,
    stable_identity,
)

TARGET_ID = "apple-m4-max"


def _assembled():
    """@brief 用仓库里的候选树打包 small+paper 域。

    @return ``(contract, domain, entries, portfolio, report)``。
    """
    contract = load_task_contract(REPO_ROOT / "tasks" / "conv2d", dtype="fp16", size="small")
    target = load_target(TARGET_ID)
    domain = domain_for_sizes(contract, target, ("small", "paper"))
    entries = parent_entries_from_tree(TREE, domain)
    portfolio = parse_portfolio(entries, domain_plan=domain.plan)
    report = check_parent(entries, domain)
    return contract, target, domain, entries, portfolio, report


def _evidence(portfolio) -> dict[str, str]:
    """@brief 把每个 provider 标成已验证，供 CPU dispatch 使用。

    @param portfolio 已解析的父包。
    @return provider 身份到证据状态。
    """
    evidence = {variant.variant_id: "validated" for variant in portfolio.variants}
    evidence[portfolio.fallback.fallback_id] = "validated"
    return evidence


def _resolve(contract, target, domain, portfolio, size: str, usage: str):
    """@brief 对一个域点做精确 dispatch，不预检几何。

    @param contract conv2d 合同。
    @param target 目标档案。
    @param domain 组装域。
    @param portfolio 已解析父包。
    @param size ``small`` 或 ``paper``。
    @param usage ``golden`` 或 ``benchmark``。
    @return dispatch 记录。
    """
    instance = instantiate_v2(contract.spec, "fp16", size)
    point = domain.plan.find("fp16", size, usage)
    request = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage)
    resolver = ExactDispatchResolver(
        domain_plan=domain.plan,
        portfolio=portfolio,
        evidence=_evidence(portfolio),
        spec=contract.spec,
        target=target,
        instances=domain.instances,
    )
    return resolver.resolve(request)


def _entry(entries, suffix: str):
    """@brief 按路径后缀取一条父包 entry。

    @param entries 父 entries。
    @param suffix 路径后缀。
    @return 那一条 entry。
    """
    found = [entry for entry in entries if entry.path.endswith(suffix)]
    assert len(found) == 1
    return found[0]


def test_conv2d_portfolio_dispatches_paper_variant_and_small_fallback():
    """paper 命中唯一 MMA 变体；small 的两个用途都落到独立 fallback。"""
    contract, target, domain, entries, portfolio, report = _assembled()
    assert report.ok, report.problems
    assert [variant.variant_key for variant in portfolio.variants] == ["paper"]
    rows = {(row.size, row.usage): row for row in report.rows}
    assert set(rows) == {("small", "golden"), ("small", "benchmark"), ("paper", "golden"), ("paper", "benchmark")}
    for usage in ("golden", "benchmark"):
        small = rows[("small", usage)]
        paper = rows[("paper", usage)]
        assert small.provider == "fallback" and small.variant_key is None
        assert paper.provider == "candidate" and paper.variant_key == "paper"
        small_record = _resolve(contract, target, domain, portfolio, "small", usage)
        paper_record = _resolve(contract, target, domain, portfolio, "paper", usage)
        assert small_record.outcome == "fallback"
        assert small_record.provider == "fallback"
        assert small_record.fallback_reason == "outside_candidate_coverage"
        assert small_record.variant_id is None
        assert paper_record.outcome == "resolved"
        assert paper_record.provider == "candidate"
        assert paper_record.variant_id == portfolio.variants[0].variant_id
        assert paper_record.fallback_id is None
    again = parent_entries_from_tree(TREE, domain)
    again_portfolio = parse_portfolio(again, domain_plan=domain.plan)
    assert again_portfolio.digest == portfolio.digest
    assert again_portfolio.variants[0].variant_id == portfolio.variants[0].variant_id
    assert again_portfolio.fallback.fallback_id == portfolio.fallback.fallback_id
    envelope = encode_parent(entries)
    assert len(envelope.encode("utf-8")) < 1_000_000
    assert _entry(entries, "variants/paper/kernel.metal").data == PAPER_FIXTURE.read_bytes()


def test_projected_paper_bytes_stay_the_confirmed_fixture():
    """投影不得改 paper 源码，也不得把 fallback 编成同一份 MMA。"""
    _contract, _target, domain, entries, portfolio, report = _assembled()
    assert report.ok, report.problems
    paper = project_provider(portfolio, portfolio.variants[0], authority=domain.authority)
    fallback = project_provider(portfolio, portfolio.fallback, authority=domain.authority)
    paper_sources = [entry.data for entry in paper.candidate.entries if entry.path.endswith(".metal")]
    fallback_sources = [entry.data for entry in fallback.candidate.entries if entry.path.endswith(".metal")]
    assert paper_sources == [PAPER_FIXTURE.read_bytes()]
    assert fallback_sources == [(TREE / "fallback" / "kernel.metal").read_bytes()]
    assert hashlib.sha256(paper_sources[0]).hexdigest() == CONFIRMED_PAPER_SHA256
    assert paper.track_id == "msl-single"
    assert fallback.track_id == "msl-single"
    assert paper.candidate.candidate_sha256 != portfolio.parent_sha256
    assert fallback.candidate.candidate_sha256 != portfolio.parent_sha256
    assert paper.candidate.candidate_sha256 != fallback.candidate.candidate_sha256
    assert b"simdgroup_float8x8" in paper_sources[0]
    assert b"simdgroup_float8x8" not in fallback_sources[0]
    assert b"bool direct" in fallback_sources[0]
    assert b"scratch y" not in paper_sources[0]
    assert b"scratch y" not in fallback_sources[0]
    assert b"scratch uwin fp32 36*C*OC zero" in fallback_sources[0]
    for entry in entries:
        assert len(entry.data) <= MAX_BUNDLE_INLINE_ENTRY_BYTES
    assert MAX_BUNDLE_INLINE_ENTRY_BYTES == 262144


def test_paper_guard_is_exact_and_a_weak_guard_would_steal_small(tmp_path: Path):
    """九个符号都相等才命中 paper。只判断 KH 会把 small 错派给 MMA 变体。"""
    contract, target, domain, _entries, portfolio, _report = _assembled()
    guard = parse_term(json.loads((TREE / "coverage.json").read_text(encoding="utf-8"))["variants"][0]["guard"], "paper")
    assert evaluate(guard, PAPER_SYMBOLS) is True
    assert evaluate(guard, SMALL_SYMBOLS) is False
    stolen = dict(PAPER_SYMBOLS)
    stolen["OC"] = 64
    assert evaluate(guard, stolen) is False
    root = _copy_tree(tmp_path)
    document = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    document["variants"][0]["guard"] = ["eq", "KH", 3]
    (root / "coverage.json").write_text(json.dumps(document), encoding="utf-8")
    entries = parent_entries_from_tree(root, domain)
    weakened = parse_portfolio(entries, domain_plan=domain.plan)
    record = _resolve(contract, target, domain, weakened, "small", "golden")
    assert record.provider == "candidate"
    assert record.variant_id == weakened.variants[0].variant_id
    real = _resolve(contract, target, domain, portfolio, "small", "golden")
    assert real.provider == "fallback"


def test_overlap_and_refuse_are_rejected_without_guessing(tmp_path: Path):
    """两个相同 guard 是 overlap；拿掉 fallback 后 small 无人接。"""
    _contract, _target, domain, _entries, portfolio, report = _assembled()
    assert report.ok, report.problems
    overlap_root = _copy_tree(tmp_path)
    document = json.loads((overlap_root / "coverage.json").read_text(encoding="utf-8"))
    document["variants"].append({"variant_key": "other", "guard": document["variants"][0]["guard"]})
    (overlap_root / "coverage.json").write_text(json.dumps(document), encoding="utf-8")
    destination = overlap_root / "variants" / "other"
    destination.mkdir()
    shutil.copyfile(overlap_root / "fallback" / "kernel.metal", destination / "kernel.metal")
    overlap = check_parent(parent_entries_from_tree(overlap_root, domain), domain)
    assert overlap.ok is False
    assert any("两个以上变体同时命中" in problem for problem in overlap.problems)

    refuse_root = _copy_tree(tmp_path / "refuse")
    shutil.rmtree(refuse_root / "fallback")
    refused = json.loads((refuse_root / "coverage.json").read_text(encoding="utf-8"))
    refused["fallback"] = {"kind": "refuse", "reasons": ["outside_candidate_coverage"]}
    (refuse_root / "coverage.json").write_text(json.dumps(refused), encoding="utf-8")
    missing = check_parent(parent_entries_from_tree(refuse_root, domain), domain)
    assert missing.ok is False
    assert any("没人接" in problem for problem in missing.problems)

    mutated = _copy_tree(tmp_path / "mutated")
    kernel = mutated / "variants" / "paper" / "kernel.metal"
    kernel.write_text(kernel.read_text(encoding="utf-8").replace("using namespace metal;", "using namespace metal; ", 1), encoding="utf-8")
    changed = parse_portfolio(parent_entries_from_tree(mutated, domain), domain_plan=domain.plan)
    assert changed.variants[0].variant_id != portfolio.variants[0].variant_id


def test_small_mma_blocker_does_not_reuse_the_paper_confirmation():
    """small 没有合法的 MMA 性能变体。paper 的确认数字不能当成 small 或发布。"""
    blocker = json.loads((TREE / "small-mma-blocker.json").read_text(encoding="utf-8"))
    assert blocker["schema_id"] == "shapeforge.conv2d-portfolio-small-mma-blocker.1"
    assert blocker["small_specialized_mma"] is False
    assert blocker["publication_allowed"] is False
    assert blocker["paper_confirmation_applies_to_small"] is False
    assert blocker["paper_candidate_sha256"] == CONFIRMED_PAPER_SHA256
    assert blocker["measurement_file"] == "full-eval.json"
    assert blocker["measurement_runs"] == [
        "tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-portfolio-small.json",
        "tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-portfolio-paper.json",
    ]
    assert "C=64" in blocker["reason"]
    assert "617.29" in blocker["reason"]
    assert "full-eval.json" in blocker["reason"]
    assert SMALL_SYMBOLS["OH"] % 4 == 0
    assert PAPER_SYMBOLS["OH"] % 4 == 2
    assert SMALL_SYMBOLS["C"] * SMALL_SYMBOLS["OC"] % 64 == 0
    assert SMALL_SYMBOLS["N"] * SMALL_SYMBOLS["OC"] % 64 == 0
    assert PAPER_SYMBOLS["C"] * PAPER_SYMBOLS["OC"] % 64 == 0
    assert PAPER_SYMBOLS["N"] * PAPER_SYMBOLS["OC"] % 64 == 0


def test_recorded_search_seed_eval_keeps_small_outside_the_gate():
    """两份归档 run 决定摘要。身份和 1.10 门可重放，墙钟微秒不逐值钉死。"""
    assert_recorded_portfolio()


def test_replay_does_not_require_equal_wall_clock_samples():
    """改动归档墙钟后，稳定身份不变；把 small 改到门内才会被拒绝。"""
    recorded = load_json(SMALL_RUN)
    shifted = json.loads(json.dumps(recorded))
    shifted["timing"]["candidate_us"] = float(recorded["timing"]["candidate_us"]) + 25.0
    assert stable_identity(shifted) == stable_identity(recorded)
    assert_timing_gate(shifted)
    fast = json.loads(json.dumps(recorded))
    fast["timing"]["candidate_us"] = float(fast["timing"]["baseline_us"]["torch_mps"]) / 2.0
    fast["timing"]["baseline_us"]["mlx"] = fast["timing"]["candidate_us"] * 2.0
    with pytest.raises(AssertionError, match="small fallback latency ratio"):
        assert_timing_gate(fast)
    assert load_json(PAPER_RUN)["candidate_sha256"] == CONFIRMED_PAPER_SHA256
    assert load_json(SMALL_RUN)["provider"] == "fallback"


def _copy_tree(root: Path) -> Path:
    """@brief 把候选树拷进测试临时目录。

    @param root 临时父目录。
    @return 拷贝后的树。
    """
    copied = root / "tree"
    shutil.copytree(TREE, copied)
    return copied
