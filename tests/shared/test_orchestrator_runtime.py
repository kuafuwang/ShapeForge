"""Compatibility tests for the local evaluator/session adapters in the search loop."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from taic.kernel_agent import AgentResult
from tcenv.evaluation import EvaluationUnavailable
from ..support.orchestrator import code, llm_script, make_orch, run






class RecordingRunner:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, brief, *, execution, provider, model, effort):
        self.calls.append((brief, execution, provider, model, effort))
        text = code("speed=1.2")
        source = text.split("```metal\n", 1)[1].rsplit("```", 1)[0]
        return AgentResult(source=source + "\n", cost_usd=0.01, cost_incomplete=False, status="ok")




def test_search_can_use_an_injected_session_runner(tmp_path):
    """@brief 测试注入的 agent_runner 仍优先；没有它时可用 HmzKernelSessionRunner。"""
    runner = RecordingRunner()
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, session_runner=runner)
    orch.agent_runner = None
    res = run(orch)
    assert res.best_speedup == 1.2
    assert len(runner.calls) == 1
    _brief, execution, provider, model, effort = runner.calls[0]
    assert execution.placement.harness_placement == "local"
    assert execution.paths.host_work == (orch.run_dir / "agent_work" / "init").resolve()
    assert provider == "deepseek" or provider
    assert model


def test_fixed_selfcheck_socket_is_bound_as_host_bridge(tmp_path):
    """@brief 固定 bridge 没有 factory 时，execution 元数据也必须匹配实际自测路径。"""
    runner = RecordingRunner()
    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0,
        session_runner=runner,
    )
    orch.selfcheck_socket = str(tmp_path / "fixed.sock")
    orch.selfcheck_token_file = str(tmp_path / "fixed.token")
    orch.agent_runner = None
    run(orch)
    execution = runner.calls[0][1]
    assert execution.selfcheck.mode == "host_bridge"
    assert execution.selfcheck.endpoint_id == str(tmp_path / "fixed.sock")
    assert execution.selfcheck.token_ref == str(tmp_path / "fixed.token")










def test_evaluation_request_id_is_stable_for_resume(tmp_path):
    """@brief 同 run/session/候选/请求体的 request_id 必须确定，避免恢复时重复评测。"""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0)
    orch._last_session_id = "init"
    first = orch._evaluation_request("kernel void demo() {}", mode="quick", seed=3, reps=2, label="init")
    second = orch._evaluation_request("kernel void demo() {}", mode="quick", seed=3, reps=2, label="init")
    assert first.request_id == second.request_id
    assert first.request_digest == second.request_digest


def test_selfcheck_bridge_factory_is_scoped_to_one_session(tmp_path):
    """@brief 每场会话使用独立 bridge，并在反馈归档后停止它。"""
    class Bridge:
        def __init__(self):
            from types import SimpleNamespace
            self.socket_path = tmp_path / "selfcheck.sock"
            self.config = SimpleNamespace(reps=7, min_improvement=1.2, noise_z=3.0, max_timed=1, max_gpu_s=2.5, allow_validate=False)
            self.stopped = False
        def stop(self):
            self.stopped = True

    bridges = []
    def factory(session_id, work, evaluator):
        del work, evaluator
        bridge = Bridge()
        bridges.append((session_id, bridge))
        return bridge, str(tmp_path / "selfcheck.token")

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, selfcheck_bridge_factory=factory)
    res = run(orch)
    assert res.best_speedup == 1.2
    assert len(bridges) == 1 and bridges[0][0] == "init"
    assert bridges[0][1].stopped is True
    brief = orch._session_briefs[0]
    assert brief.selfcheck_token_file.endswith("selfcheck.token")
    assert (brief.reps, brief.min_improvement, brief.noise_z) == (7, 1.2, 3.0)
    assert (brief.selfcheck_max_timed, brief.selfcheck_max_gpu_s, brief.selfcheck_allow_validate) == (1, 2.5, False)






def test_execution_manifest_records_observed_evaluator_identity(tmp_path):
    """@brief execution.json 必须保留实际 evaluator 身份，而不只保存用户配置。"""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0)
    run(orch)
    document = json.loads((orch.run_dir / "execution.json").read_text())
    observed = document["observed"]
    assert observed["site_id"] == "local"
    assert observed["target_id"] == "apple-m4-max"
    assert observed["epoch"]
    assert observed["worker_build"] == ""


def test_execution_manifest_rejects_observed_identity_drift(tmp_path):
    """@brief 同一 run 内 worker build 变化必须被识别为 evaluator identity drift。"""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0)
    request = orch._evaluation_request("kernel void demo() {}", mode="quick", seed=3, reps=1, label="init")
    provenance = orch.evaluator._provenance(request)
    orch._observe_evaluator(provenance)
    with pytest.raises(EvaluationUnavailable, match="identity"):
        orch._observe_evaluator(replace(provenance, worker_build="different-build"))
