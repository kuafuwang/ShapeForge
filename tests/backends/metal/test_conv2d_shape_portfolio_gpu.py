"""Real-Metal correctness for the conv2d small+paper portfolio.

paper executes the confirmed F(4×4) MMA fixture.  small executes the independent general
F(4×4) fallback.  A separate run of that same fallback source on paper checks the 2-wide
tail.  The slow node remeasures both benchmark points at the search seed and checks the
1.10 gate without publishing.
"""

from __future__ import annotations

import pytest

from tcenv.backends.metal.launch import parse_launch_spec
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.target import load_target
from tcenv.taskcontract.portfolio_runner import locator_for_instance
from tests.backends.metal.conv2d_portfolio_record import (
    conv2d_locator,
    project_portfolio_run,
    run_conv2d_portfolio,
)
from tests.support.conv2d_portfolio import (
    CONFIRMED_PAPER_SHA256,
    PAPER_RUN,
    PAPER_SYMBOLS,
    SMALL_RUN,
    SMALL_SYMBOLS,
    TREE,
    assert_timing_gate,
    load_json,
    stable_identity,
)

CASES = ("normal", "wide_range", "large", "tiny", "outlier", "ieee")
gpu = pytest.mark.gpu


def _task_locator():
    """@brief 仓库里的 conv2d locator。

    @return 默认档 locator。
    """
    return conv2d_locator()


def _run_portfolio(size: str, *, usage: str = "golden", mode: str = "quick", reps: int = 1):
    """@brief 用真实 Metal 适配器跑一个域点的全部声明 case。

    @param size ``small`` 或 ``paper``。
    @param usage ``golden`` 或 ``benchmark``。
    @param mode ``quick`` 或 ``full``。
    @param reps 计时重复次数。
    @return portfolio 运行结果。
    """
    return run_conv2d_portfolio(size, usage=usage, mode=mode, reps=reps)


def _why(report: dict) -> str:
    """@brief 把失败报告收成一行。

    @param report schema-4 报告。
    @return 诊断文本。
    """
    parts = [f"status={report.get('status')!r}", f"message={report.get('message')!r}"]
    errors = (report.get("compile") or {}).get("errors")
    if errors:
        parts.append(str(errors)[:800])
    for key in ("abi", "safety"):
        violations = (report.get(key) or {}).get("violations") or []
        if violations:
            parts.append(key + ": " + "; ".join(str(item) for item in violations[:4]))
    nondet = (report.get("safety") or {}).get("nondeterminism") or []
    if nondet:
        parts.append("nondeterminism: " + "; ".join(str(item) for item in nondet[:4]))
    return " | ".join(parts)


def _assert_cases(report: dict) -> None:
    """@brief 六个声明 case、安全轮次和输出初值无关性都通过。

    @param report schema-4 报告。
    """
    assert report["status"] == "ok", _why(report)
    cases = report["correctness"]["cases"]
    assert [case["case"] for case in cases] == list(CASES)
    assert all(case["ok"] for case in cases)
    assert report["safety"]["ok"] is True
    assert report["safety"]["nondeterminism"] == []


def test_general_launch_grids_match_both_declared_shapes():
    """通用核的网格表达式在 small 和 paper 上都能整除，scratch 不借用 y。"""
    source = (TREE / "fallback" / "kernel.metal").read_text(encoding="utf-8")
    small = parse_launch_spec(source, SMALL_SYMBOLS)
    paper = parse_launch_spec(source, PAPER_SYMBOLS)
    assert small is not None and paper is not None
    assert small.stages[0].grid == (32, 1, 1)
    assert small.stages[1].grid == (4, 1, 1)
    assert paper.stages[0].grid == (128, 1, 1)
    assert paper.stages[1].grid == (64, 1, 1)
    assert [item.name for item in small.scratches] == ["uwin"]
    assert small.scratches[0].init_policy == "device_zero"
    assert small.scratches[0].nbytes == 36 * SMALL_SYMBOLS["C"] * SMALL_SYMBOLS["OC"] * 4
    assert paper.scratches[0].nbytes == 36 * PAPER_SYMBOLS["C"] * PAPER_SYMBOLS["OC"] * 4
    confirmed = (TREE / "variants" / "paper" / "kernel.metal").read_text(encoding="utf-8")
    parsed = parse_launch_spec(confirmed, PAPER_SYMBOLS)
    assert parsed is not None
    assert parsed.stages[0].grid == (288, 1, 1)
    assert parsed.stages[1].grid == (392, 2, 1)


@gpu
def test_portfolio_paper_keeps_the_mma_fast_path():
    """paper 仍执行已确认的 MMA 变体，六个 case 和 out1/out2/out3 都通过。"""
    run = _run_portfolio("paper")
    assert run.record.provider == "candidate"
    assert run.record.outcome == "resolved"
    assert run.record.fallback_id is None
    report = run.result.report
    _assert_cases(report)
    document = run.document()
    assert document["fallback_credits_variant"] is False
    assert document["performance_claim"] is None
    assert document["active_manifest"] is None


@gpu
def test_portfolio_small_uses_the_general_fallback():
    """small 没有 MMA 变体。通用 F(4×4) 在六个 case 和三轮初值下正确，时间不算变体加速。"""
    run = _run_portfolio("small")
    assert run.record.provider == "fallback"
    assert run.record.outcome == "fallback"
    assert run.record.fallback_reason == "outside_candidate_coverage"
    assert run.record.variant_id is None
    report = run.result.report
    _assert_cases(report)
    document = run.document()
    assert document["fallback_credits_variant"] is False
    assert document["performance_claim"] is None


@gpu
def test_general_kernel_covers_the_paper_partial_tail():
    """同一份通用核在 paper 上覆盖 OH=54 的 2 宽尾块，含 IEEE 直接卷积回退。"""
    source = (TREE / "fallback" / "kernel.metal").read_text(encoding="utf-8")
    locator = locator_for_instance(_task_locator(), "fp16", "paper")
    report = evaluate_candidate(
        locator, source, target=load_target("apple-m4-max"), mode="quick", reps=1, seed=7
    ).result.report
    _assert_cases(report)


@gpu
def test_general_kernel_does_not_repair_a_missed_u_element():
    """把最后一个 U 写成 0 后，普通输入仍走 Winograd，结果是 incorrect 而不是被直接卷积补上。"""
    source = (TREE / "fallback" / "kernel.metal").read_text(encoding="utf-8")
    original = "uwin[(e * C + c) * OC + oc] = sum;"
    patched = (
        "float stored_u = sum; "
        "if (e + 1u == 36u && c + 1u == C && oc + 1u == OC) stored_u = 0.0f; "
        "uwin[(e * C + c) * OC + oc] = stored_u;"
    )
    assert source.count(original) == 1
    locator = locator_for_instance(_task_locator(), "fp16", "small")
    report = evaluate_candidate(
        locator,
        source.replace(original, patched, 1),
        target=load_target("apple-m4-max"),
        mode="quick",
        reps=1,
        seed=7,
        cases=("normal",),
    ).result.report
    assert report["status"] == "incorrect", _why(report)


@gpu
@pytest.mark.slow
def test_search_seed_full_eval_keeps_paper_margin_and_small_outside_the_gate():
    """重跑两档正式计时，重放归档身份和 1.10 门，不要求墙钟微秒逐值相等。"""
    paper = project_portfolio_run(
        _run_portfolio("paper", usage="benchmark", mode="full", reps=15),
        reps=15,
        usage="benchmark",
    )
    small = project_portfolio_run(
        _run_portfolio("small", usage="benchmark", mode="full", reps=15),
        reps=15,
        usage="benchmark",
    )
    archived_paper = load_json(PAPER_RUN)
    archived_small = load_json(SMALL_RUN)
    assert stable_identity(paper) == stable_identity(archived_paper)
    assert stable_identity(small) == stable_identity(archived_small)
    assert_timing_gate(paper)
    assert_timing_gate(small)
    assert_timing_gate(archived_paper)
    assert_timing_gate(archived_small)
    assert paper["candidate_sha256"] == CONFIRMED_PAPER_SHA256
    assert small["candidate_sha256"] != CONFIRMED_PAPER_SHA256
    assert paper["timing"]["candidate_us"] != small["timing"]["candidate_us"]
