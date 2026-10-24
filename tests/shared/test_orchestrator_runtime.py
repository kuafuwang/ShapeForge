"""Compatibility tests for the local evaluator/session adapters in the search loop."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from taic.kernel_agent import AgentResult
from tcenv.remote.client import RemoteError
from tcenv.evaluation import EvaluationUnavailable
from tcenv.evaluation import EvaluationResult, local_provenance, receipt_id_for

from ..support.orchestrator import code, fake_eval, llm_script, make_orch, run


class RecordingEvaluator:
    def __init__(self) -> None:
        self.requests = []
        self.base = fake_eval([])

    def formal(self, request):
        self.requests.append(request)
        report = self.base(request.task, request.dtype, request.size, request.source, mode=request.mode, seed=request.seed, reps=request.reps)
        return EvaluationResult(report=report, provenance=local_provenance(request), evidence={"candidate_sha256": request.candidate_sha256}, receipt={"receipt_id": receipt_id_for(request), "request_id": request.request_id, "request_digest": request.request_digest, "measurement_kind": "measured"})

    def selfcheck(self, request):
        raise AssertionError("search must use formal evaluation")


class RecordingRunner:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, brief, *, execution, provider, model, effort):
        self.calls.append((brief, execution, provider, model, effort))
        text = code("speed=1.2")
        source = text.split("```metal\n", 1)[1].rsplit("```", 1)[0]
        return AgentResult(source=source + "\n", cost_usd=0.01, cost_incomplete=False, status="ok")


def test_search_uses_the_injected_evaluator_with_a_unified_request(tmp_path):
    """@brief `_eval` 的搜索评测通过统一 executor，并带上 run/task fingerprint。"""
    evaluator = RecordingEvaluator()
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=evaluator, publishable_site_ids={"local"})
    res = run(orch)
    assert res.best_speedup == 1.2
    assert evaluator.requests
    request = evaluator.requests[0]
    assert request.kind == "formal_eval" and request.mode in {"quick", "full"}
    assert request.run_id == orch.run_dir.name
    assert request.session_id == "init"
    assert request.task_fingerprint == orch.task_fingerprint


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


class FailingEvaluator:
    def formal(self, request):
        raise RemoteError("transport_closed", "simulated disconnect")

    def selfcheck(self, request):
        raise AssertionError


class UnpublishedEvaluator(RecordingEvaluator):
    def formal(self, request):
        result = super().formal(request)
        return EvaluationResult(report=result.report, provenance=replace(result.provenance, publication_allowed=False), evidence=result.evidence, receipt=result.receipt)


def test_remote_failure_finishes_with_evaluation_unavailable_and_outbox(tmp_path):
    """@brief 远程断连不写假报告，保留 result 和 unavailable outbox，且不计 GPU 故障。"""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=FailingEvaluator())
    res = run(orch)
    assert res.stop_reason == "evaluation_unavailable"
    assert (orch.run_dir / "result.json").is_file()
    rows = list((orch.run_dir / "evaluation_outbox").glob("*.json"))
    assert len(rows) == 1
    import json
    document = json.loads(rows[0].read_text())
    assert document["status"] == "unavailable"
    assert document["error"]["code"] == "transport_closed"
    assert orch.hangs == 0 and orch.crashes_in_a_row == 0


def test_publication_false_is_rejected_before_confirmation(tmp_path):
    """@brief publication_allowed=False 的报告不能进入 best 或 confirmed。"""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=UnpublishedEvaluator(), publishable_site_ids={"local"})
    res = run(orch)
    assert res.stop_reason == "evaluation_unavailable"
    assert res.best_speedup is None and res.confirmed_speedup is None
    rows = list((orch.run_dir / "evaluation_outbox").glob("*.json"))
    import json
    assert json.loads(rows[0].read_text())["error"]["code"] == "publication_refused"


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
            self.socket_path = tmp_path / "selfcheck.sock"
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
    assert orch._session_briefs[0].selfcheck_token_file.endswith("selfcheck.token")


def test_resume_rejects_removing_epoch_allowlist_even_through_direct_orchestrator(tmp_path):
    """@brief 库入口也必须拒绝 remote resume 时把 epoch 白名单改成不限制。"""
    from taic.orchestrator import RunStartError
    from tests.support.fakes import FakeClient

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0)
    (orch.run_dir / "execution.json").write_text(json.dumps({
        "schema_version": 1, "evaluation": "remote",
        "publishable_site_ids": ["local"], "publishable_epochs": ["e" * 64],
    }))
    with pytest.raises(RunStartError, match="epoch allowlist"):
        type(orch)(
            orch.task_name, orch.dtype, orch.size, orch.cfg, orch.search,
            run_dir=orch.run_dir, resuming=True, client=FakeClient([]),
            evaluate_fn=fake_eval([]), evaluator=RecordingEvaluator(),
            publishable_site_ids={"local"}, publishable_epochs=None,
        )


def test_execution_manifest_records_observed_evaluator_identity(tmp_path):
    """@brief execution.json 必须保留实际 evaluator 身份，而不只保存用户配置。"""
    evaluator = RecordingEvaluator()
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=evaluator, publishable_site_ids={"local"})
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
    from tcenv.evaluation import local_provenance
    provenance = local_provenance(request)
    orch._observe_evaluator(provenance)
    with pytest.raises(EvaluationUnavailable, match="identity"):
        orch._observe_evaluator(replace(provenance, worker_build="different-build"))
