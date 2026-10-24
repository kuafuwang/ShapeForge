"""Phase 0a local runner interface tests. No hmz, model, GPU or Docker."""

from dataclasses import replace
from pathlib import Path

import pytest

from taic.kernel_agent import AgentResult, Prices, SessionBrief
from taic.session_runtime import (
    HmzSessionPlacement,
    LocalHmzKernelSessionRunner,
    SelfCheckBinding,
    SessionExecution,
    SessionPaths,
)


def _brief(work: Path) -> SessionBrief:
    """Build the minimal session brief accepted by the local runner."""
    return SessionBrief(
        repo=work.parent,
        task_name="relu",
        dtype="fp16",
        size="small",
        kernel_name="relu_kernel",
        triton_source="def relu(x): return x",
        description="relu",
        contract_text="kernel relu_kernel",
        abi_stub="kernel void relu_kernel()",
        rtol=1e-3,
        atol_rel=1e-3,
    )


def _execution(work: Path) -> SessionExecution:
    """Build one local execution with distinctive values for assertion."""
    return SessionExecution(
        placement=HmzSessionPlacement(),
        paths=SessionPaths.local(work),
        prices=Prices(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4),
        turn_output_tokens=1234,
        turn_minutes=17,
        session_cost_cap_usd=1.25,
        timeout_s=42.0,
        hmz_binary="/tmp/fake-hmz",
        selfcheck=SelfCheckBinding(mode="local_cli"),
    )


def test_session_paths_defaults_to_the_local_canonical_workdir(tmp_path):
    """@brief SessionPaths 默认让 agent 看到与 host_work 相同的路径。"""
    paths = SessionPaths.local(tmp_path / "agent_work" / "r1.p0")
    assert paths.agent_work == str(paths.host_work)
    assert paths.host_work == tmp_path / "agent_work" / "r1.p0"


def test_local_runner_forwards_existing_run_kernel_agent_parameters(tmp_path):
    """@brief 本地 runner 必须完整转发现有 run_kernel_agent 的全部执行参数。"""
    work = tmp_path / "agent_work" / "r1.p0"
    brief = _brief(work)
    execution = _execution(work)
    expected = AgentResult(source="kernel", cost_usd=0.5, cost_incomplete=False, status="ok")
    captured: dict = {}

    def fake_run(brief_arg, **kwargs):
        captured["brief"] = brief_arg
        captured.update(kwargs)
        return expected

    runner = LocalHmzKernelSessionRunner(run_fn=fake_run)
    result = runner.execute(
        brief,
        execution=execution,
        provider="deepseek",
        model="deepseek-flash",
        effort="high",
    )

    assert result is expected
    assert captured["brief"] is brief
    assert captured["work"] == work.resolve()
    assert captured["provider"] == "deepseek"
    assert captured["model"] == "deepseek-flash"
    assert captured["effort"] == "high"
    assert captured["prices"] == execution.prices
    assert captured["turn_output_tokens"] == 1234
    assert captured["turn_minutes"] == 17
    assert captured["session_cost_cap_usd"] == 1.25
    assert captured["hmz"] == "/tmp/fake-hmz"
    assert captured["timeout_s"] == 42.0


def test_local_runner_fills_an_empty_brief_workdir(tmp_path):
    """@brief 空 work_dir 必须补成 host_work，避免 TASK 命令回退到当前目录。"""
    work = tmp_path / "agent_work" / "r1.p0"
    brief = _brief(work)
    runner = LocalHmzKernelSessionRunner(run_fn=lambda *args, **kwargs: AgentResult(source=None, cost_usd=0.0, cost_incomplete=False, status="ok"))
    runner.execute(brief, execution=_execution(work), provider="p", model="m", effort=None)
    assert brief.work_dir == str(work.resolve())


def test_local_runner_validates_the_host_bridge_binding(tmp_path):
    """@brief execution 与 brief 的 bridge 必须一致，不能一边声明一边实际走另一条路径。"""
    work = tmp_path / "agent_work" / "r1.p0"
    brief = replace(_brief(work), selfcheck_socket="/host/session.sock", selfcheck_token_file="/host/session.token")
    execution = replace(
        _execution(work),
        selfcheck=SelfCheckBinding(mode="host_bridge", endpoint_id="/host/session.sock", token_ref="/host/session.token"),
    )
    runner = LocalHmzKernelSessionRunner(run_fn=lambda *args, **kwargs: AgentResult(source=None, cost_usd=0.0, cost_incomplete=False, status="ok"))
    runner.execute(brief, execution=execution, provider="p", model="m", effort=None)

    mismatched = replace(execution, selfcheck=SelfCheckBinding(mode="local_cli"))
    with pytest.raises(ValueError, match="conflicts"):
        runner.execute(brief, execution=mismatched, provider="p", model="m", effort=None)


def test_local_runner_refuses_a_nonlocal_placement(tmp_path):
    """@brief 本地 runner 不能把远程 placement 悄悄当成本地执行。"""
    work = tmp_path / "agent_work" / "r1.p0"
    execution = SessionExecution(
        placement=HmzSessionPlacement(harness_placement="env", workspace_env="ssh"),
        paths=SessionPaths.local(work),
        prices=Prices(0.1, 0.2, 0.3, 0.4),
    )
    runner = LocalHmzKernelSessionRunner(run_fn=lambda *args, **kwargs: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="local harness and workspace"):
        runner.execute(_brief(work), execution=execution, provider="p", model="m", effort=None)


def test_local_runner_refuses_a_brief_that_points_at_another_workdir(tmp_path):
    """@brief brief.work_dir 与 host_work 不一致时，TASK 命令会指向别处，必须拒绝。"""
    work = tmp_path / "agent_work" / "r1.p0"
    brief = replace(_brief(work), work_dir=str(tmp_path / "elsewhere"))
    runner = LocalHmzKernelSessionRunner(run_fn=lambda *args, **kwargs: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="brief.work_dir"):
        runner.execute(brief, execution=_execution(work), provider="p", model="m", effort=None)


def test_local_runner_refuses_mismatched_agent_visible_paths(tmp_path):
    """@brief 本地 agent 可见路径与 host canonical 路径不一致时必须失败。"""
    work = tmp_path / "agent_work" / "r1.p0"
    execution = SessionExecution(
        placement=HmzSessionPlacement(),
        paths=SessionPaths(host_work=work, agent_work=str(tmp_path / "elsewhere")),
        prices=Prices(0.1, 0.2, 0.3, 0.4),
    )
    runner = LocalHmzKernelSessionRunner(run_fn=lambda *args, **kwargs: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="host_work"):
        runner.execute(_brief(work), execution=execution, provider="p", model="m", effort=None)
