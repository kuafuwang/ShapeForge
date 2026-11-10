"""CPU tests for eval-portfolio.  Every point is an injected run_point; nothing touches a GPU."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from taic.portfolio_confirmation import SEARCH_SEED
from taic.portfolio_search import run_portfolio_search
from taic.task_binding import bind_task
from tcenv.backends.metal.portfolio_eval import EVALUATION_SCHEMA_ID, PortfolioEvaluation, evaluate_parent
from tcenv.cli import main
from tcenv.contract_errors import TaskContractError
from tcenv.portfolio_assembly import (
    benchmark_sizes,
    domain_for_sizes,
    encode_parent,
    parent_entries_from_tree,
)
from tcenv.target import load_target
from tcenv.track.plane import CandidateEntry
from tests.support.portfolio_assembly import (
    PAPER_GUARD,
    SMALL_GUARD,
    gemm_variant_source,
    write_gemm_tree,
)

_TARGET = "apple-m4-max"


@lru_cache(maxsize=1)
def _gemm():
    """@brief 绑定一次 gemm，并造出 paper+small 的域。

    @return ``(contract, target, domain)``。
    """
    contract = bind_task("gemm", "fp16", "paper", target_id=_TARGET).locator.load()
    target = load_target(_TARGET)
    sizes = benchmark_sizes(contract.spec, "fp16")
    return contract, target, domain_for_sizes(contract, target, sizes)


def _locator(contract):
    """@brief 一个只负责把缓存的合同交回去的定位器。

    @param contract 任务合同。
    @return 带 ``load`` 的对象。
    """
    return SimpleNamespace(load=lambda: contract)


def _parent(tmp_path: Path):
    """@brief 打包一份覆盖 gemm 两个 benchmark 档的父包。

    @param tmp_path 测试临时目录。
    @return ``(locator, entries, target, sizes)``。
    """
    contract, target, domain = _gemm()
    root = tmp_path / "src"
    write_gemm_tree(root, contract)
    entries = parent_entries_from_tree(root, domain)
    return _locator(contract), entries, target, domain.sizes


def _overlap(tmp_path: Path):
    """@brief 两个变体的 guard 都接住同一个点。

    @param tmp_path 测试临时目录。
    @return 父 entries。
    """
    contract, _target, domain = _gemm()
    root = tmp_path / "overlap"
    for size in ("small", "paper"):
        destination = root / "variants" / size / "kernel.metal"
        destination.parent.mkdir(parents=True)
        destination.write_text(gemm_variant_source(contract, size), encoding="utf-8")
    document = {
        "schema_id": "tcenv.portfolio-assembly.coverage.1",
        "planner": {"id": "overlap", "version": 1},
        "variants": [
            {"variant_key": "small", "guard": ["ge", "M", 1]},
            {"variant_key": "paper", "guard": ["ge", "M", 1]},
        ],
        "fallback": {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    }
    (root / "coverage.json").write_text(json.dumps(document) + "\n", encoding="utf-8")
    return parent_entries_from_tree(root, domain)


def _quiet_report(candidate_us: float, baseline_us: float, *, noisy: bool = False) -> dict:
    """@brief 一份安静（或故意吵）的计时报告。

    @param candidate_us 候选中位数。
    @param baseline_us 基线中位数。
    @param noisy 是否标成吵。
    @return 报告字典。
    """
    return {
        "status": "ok",
        "message": "",
        "correctness": {"ok": True, "cases": []},
        "evaluation_fingerprint": "b" * 64,
        "candidate_us": candidate_us,
        "baseline_us": {"mlx": baseline_us},
        "speedup": baseline_us / candidate_us,
        "benchmark": {
            "noisy": noisy,
            "best_baseline": "mlx",
            "candidate": {"cv": 0.01, "reps": 15},
            "baselines": {"mlx": {"cv": 0.01, "reps": 15}},
        },
    }


def _ok_report() -> dict:
    """@brief 一份只有正确性的报告。

    @return 报告字典。
    """
    return {"status": "ok", "message": "", "correctness": {"ok": True}}


def _run(report: dict, *, provider: str = "candidate", label: str = "paper"):
    """@brief 造一个带 record、projection、result、evaluation 的假运行。

    @param report ``result.report``。
    @param provider ``candidate`` 或 ``fallback``。
    @param label 投影标签，fallback 时不用。
    @return 简单对象。
    """
    return SimpleNamespace(
        record=SimpleNamespace(
            provider=provider,
            outcome=report.get("status", "ok"),
            variant_id="variant",
            fallback_id=None,
            fallback_reason=None,
        ),
        projection=None if provider != "candidate" else SimpleNamespace(label=label),
        result=SimpleNamespace(report=report),
        evaluation={
            "dispatch": [1, 1, 1],
            "expected_dispatch": [1, 1, 1],
            "track": {"track_id": "msl-single"},
            "evaluation_fingerprint": "b" * 64,
        },
    )


def _scripted(ratios: dict, *, providers: dict | None = None, noisy_sizes: set | None = None, timeout_first: bool = False, fail_first: bool = False):
    """@brief 按 size 回放计时数字的 ``run_point``。

    正确性调用回一份 ok 报告。计时调用按 ``ratios[size] = (candidate_us, baseline_us)`` 回放。

    @param ratios 每个 size 的一对微秒数。
    @param providers 覆盖某个 size 的 provider。
    @param noisy_sizes 这些 size 的计时标成吵。
    @param timeout_first 第一次调用返回 timeout。
    @param fail_first 第一次调用抛任务合同错误。
    @return ``(run_point, calls)``。
    """
    chosen = providers or {}
    noisy = noisy_sizes or set()
    calls = []

    def run_point(locator, plan, portfolio, request, evidence, *, mode, seed, reps):
        calls.append({"size": request.size, "usage": request.usage, "mode": mode, "seed": seed, "reps": reps})
        if fail_first and len(calls) == 1:
            raise TaskContractError("bad point")
        if timeout_first and len(calls) == 1:
            return _run({"status": "timeout", "message": "hung"}, label=request.size)
        if mode != "full":
            return _run(_ok_report(), label=request.size)
        candidate_us, baseline_us = ratios[request.size]
        return _run(
            _quiet_report(candidate_us, baseline_us, noisy=request.size in noisy),
            provider=chosen.get(request.size, "candidate"),
            label=request.size,
        )

    return run_point, calls


def _evaluate(tmp_path: Path, run_point, *, mode: str = "timed", before_point=None, entries=None):
    """@brief 用缓存的 gemm 域评测一份父包。

    @param tmp_path 测试临时目录。
    @param run_point 注入的单点函数。
    @param mode ``quick`` 或 ``timed``。
    @param before_point 开跑前的回调。
    @param entries 父 entries；缺省是合法的 gemm 父包。
    @return 评测结果。
    """
    locator, packed, target, sizes = _parent(tmp_path)
    return evaluate_parent(
        locator,
        packed if entries is None else entries,
        target=target,
        sizes=sizes,
        mode=mode,
        run_point=run_point,
        before_point=before_point,
    )


def test_two_quiet_shards_at_0_8_and_1_0_are_ok(tmp_path):
    """@brief 两个分片比值 0.8 和 1.0 时状态是 ok，专用账有加速比。"""
    run_point, calls = _scripted({"paper": (80.0, 100.0), "small": (100.0, 100.0)})
    evaluation = _evaluate(tmp_path, run_point)
    assert evaluation.status == "ok"
    assert evaluation.performance["specialized_hit_speedup"] == pytest.approx(1 / 0.9)
    assert evaluation.performance["publishable"] is True
    assert [item["mode"] for item in calls if item["usage"] == "golden"] == ["quick", "quick"]
    assert [item["mode"] for item in calls if item["usage"] == "benchmark"] == ["full", "full"]
    assert {item["seed"] for item in calls} == {SEARCH_SEED}
    document = evaluation.document()
    assert document["schema_id"] == EVALUATION_SCHEMA_ID
    assert document["evidence_kind"] == "search_time"
    assert document["publication_allowed"] is False


def test_a_1_2_shard_is_rejected_even_when_the_other_is_0_5(tmp_path):
    """@brief 一个分片比值 1.2 就拒绝，另一个 0.5 的分片救不了它。"""
    run_point, _calls = _scripted({"paper": (120.0, 100.0), "small": (50.0, 100.0)})
    evaluation = _evaluate(tmp_path, run_point)
    assert evaluation.status == "rejected"
    reasons = evaluation.performance["failure_reasons"]
    assert any("fp16/paper/benchmark" in reason and "ratio>1.10" in reason for reason in reasons)
    assert {
        "fp16/paper/benchmark:effective_ratio>1.10",
        "fp16/paper/benchmark:specialized_ratio>1.10",
    } <= set(evaluation.problems)
    assert evaluation.performance["specialized_hit_speedup"] is None
    small = next(shard for shard in evaluation.shards if shard["instance_key"] == "fp16/small/benchmark")
    assert small["latency_ratio"] == pytest.approx(0.5)


def test_one_noisy_shard_is_timing_uncertain(tmp_path):
    """@brief 只有计时不确定这一种失败原因时，状态是 timing_uncertain。"""
    run_point, _calls = _scripted(
        {"paper": (80.0, 100.0), "small": (100.0, 100.0)},
        noisy_sizes={"paper"},
    )
    evaluation = _evaluate(tmp_path, run_point)
    assert evaluation.status == "timing_uncertain"
    assert evaluation.performance["failure_reasons"] == ["fp16/paper/benchmark:timing_uncertain"]
    assert evaluation.problems == ()


def test_timeout_on_the_first_point_hangs_after_one_call(tmp_path):
    """@brief 第一个点返回 timeout 时状态是 hang，而且只调用了一次。"""
    run_point, calls = _scripted({}, timeout_first=True)
    evaluation = _evaluate(tmp_path, run_point, mode="quick")
    assert evaluation.status == "hang"
    assert len(calls) == 1
    assert evaluation.correctness[0]["status"] == "timeout"


def test_a_failed_check_does_not_call_run_point(tmp_path):
    """@brief 检查不过的父包不会跑任何一个点。"""
    run_point, calls = _scripted({"paper": (80.0, 100.0), "small": (100.0, 100.0)})
    evaluation = _evaluate(tmp_path, run_point, entries=_overlap(tmp_path))
    assert evaluation.status == "rejected"
    assert calls == []
    assert evaluation.problems


def test_a_fallback_shard_counts_only_in_the_effective_book(tmp_path):
    """@brief fallback 分片的时间进端到端账，不进专用账。"""
    run_point, _calls = _scripted(
        {"paper": (80.0, 100.0), "small": (50.0, 100.0)},
        providers={"small": "fallback"},
    )
    evaluation = _evaluate(tmp_path, run_point)
    assert evaluation.status == "ok"
    small = next(shard for shard in evaluation.shards if shard["instance_key"] == "fp16/small/benchmark")
    assert small["provider"] == "fallback"
    assert evaluation.performance["specialized_hit_latency_ratio"] == pytest.approx(0.8)
    assert evaluation.performance["effective_end_to_end_latency_ratio"] == pytest.approx(0.65)
    assert evaluation.performance["specialized_hit_speedup"] == pytest.approx(1.25)
    assert evaluation.performance["effective_end_to_end_speedup"] == pytest.approx(1 / 0.65)


def test_quick_mode_has_no_shards_and_no_performance(tmp_path):
    """@brief quick 只跑正确性，不产生分片，也不算两本账。"""
    run_point, calls = _scripted({})
    evaluation = _evaluate(tmp_path, run_point, mode="quick")
    assert evaluation.status == "ok"
    assert evaluation.shards == ()
    assert evaluation.performance is None
    assert len(calls) == 4
    assert {item["mode"] for item in calls} == {"quick"}


def test_before_point_reason_stops_before_any_run(tmp_path):
    """@brief ``before_point`` 返回原因时停下，原因出现在 problems 里。"""
    run_point, calls = _scripted({"paper": (80.0, 100.0), "small": (100.0, 100.0)})
    evaluation = _evaluate(tmp_path, run_point, before_point=lambda: "stop-before-gpu")
    assert calls == []
    assert "stop-before-gpu" in evaluation.problems
    assert evaluation.status == "rejected"


def test_a_contract_error_rejects_that_point_and_continues(tmp_path):
    """@brief 一个点抛任务合同错误时该行 rejected，后面的点继续跑。"""
    run_point, calls = _scripted({}, fail_first=True)
    evaluation = _evaluate(tmp_path, run_point, mode="quick")
    assert len(calls) == 4
    assert evaluation.status == "rejected"
    assert any(row["status"] == "rejected" and row["message"] == "bad point" for row in evaluation.correctness)
    rejected = next(row for row in evaluation.correctness if row["status"] == "rejected")
    assert f"{rejected['instance_key']}: bad point" in evaluation.problems


def test_real_evaluation_ratio_problems_reach_the_next_assemble_task(tmp_path):
    """@brief 驱动重试题面必须收到真实 evaluate_parent 的两条 ratio 拒绝原因。"""
    contract, _target, domain = _gemm()
    texts = []
    evaluations = []

    def shard_runner(size, shard_dir):
        shard_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "best_label": size,
            "best_speedup": 1.5,
            "confirmed_speedup": 1.5,
            "spent_usd": 0.1,
        }
        (shard_dir / "result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
        (shard_dir / "best.metal").write_text(gemm_variant_source(contract, size), encoding="utf-8")

    def session_runner(*, attempt, work, task_text, submission):
        texts.append(task_text)
        document = {
            "schema_id": "tcenv.portfolio-assembly.coverage.1",
            "planner": {"id": "integration", "version": 1},
            "variants": [
                {"variant_key": "small", "guard": SMALL_GUARD},
                {"variant_key": "paper", "guard": PAPER_GUARD},
            ],
            "fallback": {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
        }
        (work / "src" / "coverage.json").write_text(json.dumps(document) + "\n", encoding="utf-8")
        source = encode_parent(parent_entries_from_tree(work / "src", domain))
        return SimpleNamespace(
            source=source,
            cost_usd=0.01,
            cost_incomplete=False,
            spent={"output": 1},
            provider="fake",
            model="fake",
        )

    def evaluate(locator, entries, *, target, sizes, mode):
        ratios = (
            {"paper": (120.0, 100.0), "small": (50.0, 100.0)}
            if not evaluations
            else {"paper": (80.0, 100.0), "small": (50.0, 100.0)}
        )
        run_point, _calls = _scripted(ratios)
        outcome = evaluate_parent(
            locator, entries, target=target, sizes=sizes, mode=mode, run_point=run_point,
        )
        evaluations.append(outcome)
        return outcome

    def confirm(**kwargs):
        return {
            "status": "confirmed",
            "reason": "injected",
            "coverage_kind": "tune_confirmation",
            "instance_key": kwargs["request"].instance_key,
        }

    code = run_portfolio_search(
        "gemm",
        "fp16",
        run_dir=tmp_path / "run",
        budget=1.0,
        assemble_budget=1.0,
        assemble_attempts=2,
        seed=11,
        shard_runner=shard_runner,
        session_runner=session_runner,
        evaluate=evaluate,
        confirm=confirm,
    )
    assert code == 0
    assert len(evaluations) == 2
    expected = {
        "fp16/paper/benchmark:effective_ratio>1.10",
        "fp16/paper/benchmark:specialized_ratio>1.10",
    }
    assert expected <= set(evaluations[0].problems)
    assert len(texts) == 2
    for reason in expected:
        assert reason in texts[1]


def test_eval_portfolio_requires_exactly_one_mode():
    """@brief 没有 ``--quick`` 或 ``--timed`` 时退出 2，而且不读文件。"""
    assert main(["eval-portfolio", "gemm", "missing-parent", "--dtype", "fp16"]) == 2


def test_eval_portfolio_exit_code_follows_status(monkeypatch, tmp_path):
    """@brief 换掉 ``evaluate_parent`` 后，ok 退出 0，其余退出 1。"""
    parent = tmp_path / "parent.txt"
    entry = CandidateEntry(
        path="coverage_plan/portfolio.json",
        role="launch",
        media_type="application/json",
        data=b"{}\n",
    )
    parent.write_text(encode_parent((entry,)), encoding="utf-8")

    def fake(locator, entries, *, target, sizes, mode, seed=SEARCH_SEED, reps=15, run_point=None, before_point=None, after_point=None):
        return _canned(fake.status, mode)

    monkeypatch.setattr("tcenv.backends.metal.portfolio_eval.evaluate_parent", fake)
    fake.status = "ok"
    assert main(["eval-portfolio", "gemm", str(parent), "--dtype", "fp16", "--quick", "--json"]) == 0
    fake.status = "rejected"
    assert main(["eval-portfolio", "gemm", str(parent), "--dtype", "fp16", "--timed", "--json"]) == 1


def _canned(status: str, mode: str) -> PortfolioEvaluation:
    """@brief 命令行测试用的评测对象，字段够文本和 JSON 两条路打印。

    @param status 最终状态。
    @param mode ``quick`` 或 ``timed``。
    @return 评测对象。
    """
    return PortfolioEvaluation(
        task="gemm",
        dtype="fp16",
        target_id=_TARGET,
        sizes=("paper", "small"),
        mode=mode,
        seed=SEARCH_SEED,
        domain_plan_digest="d" * 64,
        portfolio_digest="p" * 64,
        parent_sha256="s" * 64,
        check={
            "ok": status == "ok",
            "problems": [],
            "rows": [],
            "domain_plan_digest": "d" * 64,
            "portfolio_digest": "p" * 64,
            "parent_sha256": "s" * 64,
        },
        correctness=(),
        shards=(),
        performance=None,
        status=status,
        problems=(),
    )
