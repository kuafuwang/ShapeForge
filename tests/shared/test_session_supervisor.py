"""Pure session-policy tests: no hmz, model, GPU or orchestrator state machine."""

from __future__ import annotations

import asyncio

from taic.kernel_agent import AgentResult
from taic.session_supervisor import ProviderRetryController, classify_result, session_origin


def _result(source=None, *, status="ok", failure_class=""):
    """@brief 造一个只含结果分类所需字段的 ``AgentResult``。

    @param source 会话交回的内核源码；None 表示没有提交。
    @param status 结果状态，例如 ``ok`` / ``timeout`` / ``failed``。
    @param failure_class 诊断里的失败类别。
    @return 供 ``classify_result`` 使用的 ``AgentResult``。
    """
    return AgentResult(
        source=source, cost_usd=0.01, cost_incomplete=False, status=status,
        diagnostics={"failure_class": failure_class},
    )


def test_classify_result_separates_provider_failure_from_normal_no_submission():
    """@brief 验证结果分类把供应商失败、正常未提交、超时未提交、未改动与已评测区分开。"""
    assert classify_result(_result(failure_class="turn_failed"), None) == "provider_failed"
    assert classify_result(_result(failure_class="no_submission_after_result"), None) == "agent_no_submission"
    assert classify_result(_result(), None) == "agent_failed"
    assert classify_result(_result(status="timeout", failure_class="turn_failed"), None) == "agent_failed"
    assert classify_result(_result("seed"), "seed") == "unchanged"
    assert classify_result(_result("new"), "seed") == "evaluated"
    assert classify_result(_result("seed", status="failed"), "seed") == "agent_failed"


def test_session_origin_is_json_shaped_and_attempt_is_explicit():
    """@brief 验证 ``session_origin`` 产出的 JSON 形状固定，并显式记录重试序号与候选标签。"""
    assert session_origin(provider="p", model="m", committee="first", session_label="r1.p0.retry1",
                          candidate_label="r1.p0", attempt=1) == {
        "origin": "session", "provider": "p", "model": "m", "committee": "first",
        "session_label": "r1.p0.retry1", "candidate_label": "r1.p0", "attempt": 1,
    }


def test_retry_controller_retries_then_clears_the_marker():
    """@brief 验证重试控制器首次失败后换标签重开一场会话，成功返回后清掉该候选的重试计数。"""
    calls, scheduled, sleeps, publishes = [], [], [], []

    async def attempt(tag, number):
        calls.append((tag, number))
        return "provider_failed" if number == 0 else "evaluated"

    async def nap(delay):
        sleeps.append(delay)

    controller = ProviderRetryController(1)
    outcome = asyncio.run(controller.run(
        "r1.p0", attempt_fn=attempt, publish=publishes.append, sleep=nap,
        on_scheduled=lambda failed, candidate, number, delay: scheduled.append((failed, candidate, number, delay)),
    ))
    assert outcome == "evaluated"
    assert calls == [("r1.p0", 0), ("r1.p0.retry1", 1)]
    assert scheduled == [("r1.p0", "r1.p0", 1, 5.0)] and sleeps == [5.0]
    assert controller.attempts == {}


def test_a_terminal_marker_stops_without_opening_another_session():
    """@brief 验证重试用尽后留下终态标记：带该标记恢复的控制器不再开新会话。"""
    calls, sleeps, publishes = [], [], []

    async def fail(tag, number):
        calls.append((tag, number))
        return "provider_failed"

    async def nap(delay):
        sleeps.append(delay)

    controller = ProviderRetryController(1)
    assert asyncio.run(controller.run("r1.p0", attempt_fn=fail, publish=publishes.append, sleep=nap, on_scheduled=lambda *_: None)) == "provider_failed"
    assert calls == [("r1.p0", 0), ("r1.p0.retry1", 1)]
    assert controller.attempts == {"r1.p0": 2} and sleeps == [5.0]

    async def should_not_run(*_args):
        raise AssertionError("terminal marker replayed")

    resumed = ProviderRetryController(1, controller.attempts)
    assert asyncio.run(resumed.run("r1.p0", attempt_fn=should_not_run, publish=publishes.append, sleep=nap, on_scheduled=lambda *_: None)) == "provider_failed"
    assert resumed.attempts == {"r1.p0": 2}
