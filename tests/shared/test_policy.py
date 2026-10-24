"""The seed rule lives in taic.policy. The same fake reports must get the same replace/tie/keep trace from the pure functions, from resume replay, and from the orchestrator with the second-seed gate off."""

from __future__ import annotations

import asyncio

from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, speedup_se
from taic.policy import batches_agree, candidate_eligible, choose_confirmed, comparison_speedup, in_confirm_pool, next_stagnant, patience_exhausted, relation_to_best
from taic.resume import _replay
from tcenv.evaluate import EvalReport

from ..support.fakes import FakeClient

# Frozen under min_improvement 1.005 and noise_z 2.
# No timing stats: the floor is 1.005, so 1.004 ties 1.0 and 1.006 replaces it.
# cv 0.16 over 3 reps against a seed with no recorded error needs about 1.335x; 1.20 ties and 1.40 replaces.
TIMED = [
    ("bad0", "incorrect", 9.0, None, None),
    ("a", "ok", 1.0, None, None),
    ("b", "ok", 1.004, None, None),
    ("c", "ok", 1.006, None, None),
    ("d", "incorrect", 5.0, None, None),
    ("e", "ok", 0.5, None, None),
    ("f", "ok", 1.006, None, None),
    ("g", "ok", None, None, None),
    ("h", "ok", 1.20, 0.16, 3),
    ("i", "ok", 1.40, 0.16, 3),
]
TIMED_TRACE = ["keep", "replace", "tie", "replace", "keep", "keep", "keep", "keep", "tie", "replace"]

MIN_IMPROVEMENT = 1.005
NOISE_Z = 2.0


def _report(status: str, speedup: float | None, cv: float | None, reps: int | None) -> EvalReport:
    """@brief 构造一个带加速比和 benchmark 统计的 EvalReport。

    @param status 评估状态。
    @param speedup 加速比，None 表示未测出。
    @param cv 变异系数，None 表示不带 benchmark 统计。
    @param reps benchmark 重复次数。
    @return 构造好的 EvalReport。
    """
    report = EvalReport(task="relu", dtype="fp16", size="small", mode="full", status=status, stage="done", seed=1, message=status, speedup=speedup)
    if speedup is not None:
        report.speedups = {"mlx": speedup}
        report.candidate_us = 10.0
        report.baseline_us = {"mlx": 12.0}
    if cv is not None:
        report.benchmark = {
            "candidate": {"median_us": 10.0, "cv": cv, "reps": reps},
            "baselines": {"mlx": {"median_us": 12.0, "cv": cv, "reps": reps}},
            "best_baseline": "mlx",
        }
    return report


def _cases(rows):
    """@brief 把 TIMED 风格的表格行转换成 (label, EvalReport) 列表。

    @param rows (label, status, speedup, cv, reps) 元组序列。
    @return (label, EvalReport) 列表。
    """
    return [(label, _report(status, speedup, cv, reps)) for label, status, speedup, cv, reps in rows]


def _pure_trace(cases, *, correctness_only: bool) -> list[str]:
    """@brief 直接调用 policy 纯函数，得到一批用例的 replace/tie/keep 轨迹。

    @param cases 已构造的 (label, EvalReport) 用例。
    @param correctness_only 是否只看正确性、忽略加速比。
    @return 关系字符串列表。
    """
    best_speedup = None
    best_se = 0.0
    best_report = None
    out = []
    for _label, report in cases:
        eligible = candidate_eligible(bool(report.eligible), report.speedup, correctness_only=correctness_only)
        speedup = comparison_speedup(eligible, report.speedup)
        if best_report is not None:
            best_se = speedup_se(best_report)
        relation = relation_to_best(
            eligible, speedup, best_speedup, speedup_se(report), best_se,
            min_improvement=MIN_IMPROVEMENT, noise_z=NOISE_Z,
        )
        out.append(relation)
        if relation == "replace":
            best_speedup = speedup
            best_report = report
    return out


def _replay_trace(cases, *, correctness_only: bool) -> list[str]:
    """@brief 用 resume._replay 重放同一批报告，得到 replace/tie/keep 轨迹。

    @param cases 已构造的 (label, EvalReport) 用例。
    @param correctness_only 是否只看正确性、忽略加速比。
    @return 关系字符串列表。
    """
    search = SearchConfig(min_improvement=MIN_IMPROVEMENT, noise_z=NOISE_Z, correctness_only=correctness_only)
    stored = [{"idx": i, "label": label, "report": report} for i, (label, report) in enumerate(cases)]
    out = []
    for row in _replay(stored, search):
        if row["improved"]:
            out.append("replace")
        elif row["within_noise"]:
            out.append("tie")
        else:
            out.append("keep")
    return out


def test_fake_reports_have_one_replace_tie_keep_trace(tmp_path):
    """@brief 同一批 fake 报告下，纯函数、resume 重放和关闭二次种子门的 orchestrator 得到同一条轨迹。

    @param tmp_path pytest 提供的临时目录，用作 run 目录。
    """
    cases = _cases(TIMED)
    pure = _pure_trace(cases, correctness_only=False)
    assert pure == TIMED_TRACE
    assert _replay_trace(cases, correctness_only=False) == TIMED_TRACE

    by_source = {f"kernel void relu_kernel() {{ /* {label} */ }}\n": report for label, report in cases}
    calls = []

    def evaluate(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
        calls.append(mode)
        return by_source[source]

    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(
        budget_usd=5.0, branches=1, max_rounds=1, eval_reps=3, confirm_top=4,
        min_improvement=MIN_IMPROVEMENT, noise_z=NOISE_Z, differential_recheck=False,
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False,
    )
    orch = Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / "run", client=FakeClient([]), evaluate_fn=evaluate, log=lambda _s: None, seed=1)

    async def evaluate_all():
        for label, _report in cases:
            await orch._eval(f"kernel void relu_kernel() {{ /* {label} */ }}\n", label)

    asyncio.run(evaluate_all())
    live = []
    for row in orch.trace:
        assert row["differential_recheck"] is None
        if row["improved"]:
            live.append("replace")
        elif row["within_noise"]:
            live.append("tie")
        else:
            live.append("keep")
    assert live == TIMED_TRACE
    assert calls == ["full"] * len(cases)
    assert not (orch.run_dir / "rechecks").exists()
    assert [c.label for c in orch._confirm_pool()] == ["i", "h"]


def test_a_correctness_only_replay_uses_the_same_eligibility():
    """@brief correctness_only 模式下纯函数与重放使用同一可用性规则：没有加速比也算 replace。
    """
    rows = [
        ("p", "ok", None, None, None),
        ("q", "ok", None, None, None),
        ("r", "ok", 1.1, None, None),
        ("s", "incorrect", 2.0, None, None),
    ]
    cases = _cases(rows)
    expected = ["replace", "keep", "replace", "keep"]
    assert _pure_trace(cases, correctness_only=True) == expected
    assert _replay_trace(cases, correctness_only=True) == expected
    assert candidate_eligible(True, None, correctness_only=False) is False
    assert comparison_speedup(False, 2.0) == 0.0
    assert comparison_speedup(True, None) == 1.0


def test_confirm_pool_includes_a_tie_and_patience_resets_only_on_a_new_seed():
    """@brief 确认池包含打平项；只有出现新最优时才重置 patience。
    """
    assert in_confirm_pool(1.0, 1.0, 1.005) is True
    assert in_confirm_pool(2.0, 1.0, 1.005) is False
    assert next_stagnant(False, 0) == 1
    assert next_stagnant(True, 3) == 0
    assert patience_exhausted(3, 4) is False
    assert patience_exhausted(4, 4) is True


def test_confirmation_keeps_the_search_best_inside_the_input_batch_floor():
    """@brief 确认选择在输入批次内保留 search best，除非另一候选超过噪声阈值；批次不一致时 batches_agree 为 False。
    """
    quiet = {"median": 1.0, "se": 0.0, "search_best": True}
    nudge = {"median": 1.01, "se": 0.0, "search_best": False}
    clear = {"median": 1.20, "se": 0.0, "search_best": False}
    assert choose_confirmed([quiet, nudge], min_improvement=1.005, noise_z=2.0) is quiet
    assert choose_confirmed([quiet, clear], min_improvement=1.005, noise_z=2.0) is clear
    assert choose_confirmed([nudge, clear], min_improvement=1.005, noise_z=2.0) is clear
    assert choose_confirmed(
        [{"median": 1.0, "se": 0.0, "search_best": False}, {"median": 1.01, "se": 0.0, "search_best": False}],
        min_improvement=1.005, noise_z=2.0,
    ) is None
    assert batches_agree([1.10, 1.40, 1.20]) is False
    assert batches_agree([1.20, 1.21, 1.19]) is True


def test_recent_session_cost_uses_the_median_and_ignores_empty_history():
    """@brief session_budget_floor 忽略空、None 和负数历史，按中位数取近期会话成本。
    """
    from taic.policy import session_budget_floor

    assert session_budget_floor([]) is None
    assert session_budget_floor([0.0, -1.0, None, 1.0]) == 1.0
    assert session_budget_floor([0.2, 0.4, 1.0]) == 0.4
    assert session_budget_floor([0.2, 0.4, 1.0, 3.0]) == 1.0


def test_a_degenerate_gain_floor_never_promotes_a_slower_kernel():
    """@brief 退化的增益下限至少为 1.0，慢于当前最优的候选不会被提升。
    """
    from taic.policy import compare_to_best, required_gain

    gain = required_gain(0.5, -2.0, 0.0, 0.0)
    assert gain >= 1.0
    assert compare_to_best(True, 0.6, 1.0, gain) == "keep"

