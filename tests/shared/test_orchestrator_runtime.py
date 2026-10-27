"""Compatibility tests for the local evaluator/session adapters in the search loop."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from taic.kernel_agent import AgentResult
from tcenv.evaluation import EvaluationUnavailable
from tcenv.evaluation import EvaluationResult, local_provenance, receipt_id_for

from ..support.orchestrator import code, fake_eval, llm_script, make_orch, run


class RecordingEvaluator:
    transport_kind = "grpc"

    def __init__(self) -> None:
        self.requests = []
        self.base = fake_eval([])

    def formal(self, request):
        self.requests.append(request)
        report = self.base(request.task, request.dtype, request.size, request.source, mode=request.mode, seed=request.seed, reps=request.reps)
        return EvaluationResult(report=report, provenance=local_provenance(request), evidence={"candidate_sha256": request.candidate_sha256}, receipt={"receipt_id": receipt_id_for(request), "request_id": request.request_id, "request_digest": request.request_digest, "measurement_kind": "measured"})

    def selfcheck(self, request):
        raise AssertionError("search must use formal evaluation")


def _test_epoch() -> str:
    from tcenv.evaluation import EvaluationRequest

    return local_provenance(EvaluationRequest(task="relu", dtype="fp16", size="small", source="kernel void demo() {}", target_id="apple-m4-max")).epoch


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
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=evaluator, publishable_site_ids={"local"}, publishable_epochs={_test_epoch()})
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


class FailingEvaluator:
    transport_kind = "grpc"

    def formal(self, request):
        raise EvaluationUnavailable("transport_closed", "simulated disconnect")

    def selfcheck(self, request):
        raise AssertionError


class UnpublishedEvaluator(RecordingEvaluator):
    def formal(self, request):
        result = super().formal(request)
        return EvaluationResult(report=result.report, provenance=replace(result.provenance, publication_allowed=False), evidence=result.evidence, receipt=result.receipt)


def test_remote_failure_finishes_with_evaluation_unavailable_and_outbox(tmp_path):
    """@brief 远程断连不写假报告，保留 result 和 unavailable outbox，且不计 GPU 故障。"""
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=FailingEvaluator(), publishable_epochs={_test_epoch()})
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
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=UnpublishedEvaluator(), publishable_site_ids={"local"}, publishable_epochs={_test_epoch()})
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


def test_remote_orchestrator_requires_an_explicit_epoch_allowlist(tmp_path):
    """@brief remote 不得用 publishable_epochs=None 兼容行为绕过精确发布绑定。"""
    from taic.orchestrator import RunStartError

    with pytest.raises(RunStartError, match="evaluator epoch allowlist"):
        make_orch(
            tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0,
            evaluator=RecordingEvaluator(), publishable_site_ids={"local"},
        )


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
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=evaluator, publishable_site_ids={"local"}, publishable_epochs={_test_epoch()})
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


class GrpcShapedEvaluator(RecordingEvaluator):
    """A recording evaluator that exposes the gRPC host-adapter metadata."""

    transport_kind = "grpc"
    supports_explicit_retry = True

    def __init__(self, *, fail_formal: bool = False) -> None:
        super().__init__()
        self.fail_formal = fail_formal
        self.resume_calls = 0
        self._provenance = None

    def execution_metadata(self) -> dict:
        return {
            "transport": "grpc",
            "api_contract_id": "tcenv.remote.v1.evaluator-api.1",
            "evidence_contract_id": "tcenv.remote.v1.evidence.1",
            "request_digest_algorithm": "sha256-rfc8785-v1",
            "result_signature_algorithm": "hmac-sha256-v1",
            "result_signature_key_ids": ["sim-site/key-1"],
        }

    @property
    def provenance(self):
        return self._provenance

    def _result(self, request):
        result = super().formal(request)
        result.receipt["attempt_id"] = "attempt-1"
        result.receipt["signature_key_id"] = "sim-site/key-1"
        result.receipt["signature_algorithm"] = "hmac-sha256-v1"
        self._provenance = result.provenance
        return result

    def formal(self, request):
        if self.fail_formal:
            self.requests.append(request)
            raise EvaluationUnavailable("interrupted", "durable attempt was interrupted", retryable=True)
        return self._result(request)

    def formal_resume(self, request):
        self.resume_calls += 1
        return self._result(request)


def test_execution_manifest_and_outbox_record_grpc_contract_metadata(tmp_path):
    """@brief gRPC run 必须把 transport、两个合约、attempt 与 key_id 写进证据。"""
    evaluator = GrpcShapedEvaluator()
    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0,
        evaluator=evaluator, publishable_site_ids={"local"}, publishable_epochs={_test_epoch()},
    )
    res = run(orch)
    assert res.best_speedup == 1.2
    document = json.loads((orch.run_dir / "execution.json").read_text())
    assert document["transport"] == "grpc"
    assert document["api_contract_id"] == "tcenv.remote.v1.evaluator-api.1"
    assert document["evidence_contract_id"] == "tcenv.remote.v1.evidence.1"
    observed = document["observed"]
    assert observed["transport"] == "grpc"
    assert observed["api_contract_id"] == "tcenv.remote.v1.evaluator-api.1"
    rows = list((orch.run_dir / "evaluation_outbox").glob("*.json"))
    state = json.loads(next(path for path in rows if not path.name.endswith(".result.json") and path.name != "receipts.json").read_text())
    assert state["status"] == "completed"
    assert state["transport"] == "grpc"
    assert state["api_contract_id"] == "tcenv.remote.v1.evaluator-api.1"
    assert state["attempt_id"] == "attempt-1"
    assert state["signature_key_id"] == "sim-site/key-1"
    assert state["signature_algorithm"] == "hmac-sha256-v1"


def test_resume_uses_explicit_retry_only_for_a_pending_outbox_row(tmp_path):
    """@brief 只有持久 pending/unavailable 行才驱动 Retry；普通 Submit 不会自动重试。"""
    evaluator = GrpcShapedEvaluator(fail_formal=True)
    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0,
        evaluator=evaluator, publishable_site_ids={"local"}, publishable_epochs={_test_epoch()},
    )
    source = "// speed=1.2\nkernel void demo() {}"
    with pytest.raises(EvaluationUnavailable):
        orch._evaluate_report(source, mode="quick", seed=3, reps=1, label="init")
    assert evaluator.resume_calls == 0
    report = orch._evaluate_report(source, mode="quick", seed=3, reps=1, label="init")
    assert report.status == "ok"
    assert evaluator.resume_calls == 1


def test_resume_rejects_a_grpc_run_without_its_contract_identities(tmp_path):
    """@brief 新 gRPC run 缺 api/evidence contract 时不能伪装成可续跑。"""
    from taic.orchestrator import RunStartError
    from tests.support.fakes import FakeClient

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0)
    (orch.run_dir / "execution.json").write_text(json.dumps({
        "schema_version": 1, "evaluation": "remote", "transport": "grpc",
        "publishable_site_ids": ["local"], "publishable_epochs": ["e" * 64],
    }))
    with pytest.raises(RunStartError, match="api_contract_id"):
        type(orch)(
            orch.task_name, orch.dtype, orch.size, orch.cfg, orch.search,
            run_dir=orch.run_dir, resuming=True, client=FakeClient([]),
            evaluate_fn=fake_eval([]), evaluator=GrpcShapedEvaluator(),
            publishable_site_ids={"local"}, publishable_epochs={"e" * 64},
        )


class NoTransportEvaluator(RecordingEvaluator):
    transport_kind = ""


def test_a_remote_evaluator_must_declare_its_transport(tmp_path):
    """@brief 远程 evaluator 不声明 transport 时失败，不能再默认成 legacy-ssh。"""
    from taic.orchestrator import RunStartError

    with pytest.raises(RunStartError, match="must declare its transport"):
        make_orch(
            tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0,
            evaluator=NoTransportEvaluator(), publishable_site_ids={"local"},
            publishable_epochs={_test_epoch()},
        )


def test_resume_rejects_a_legacy_ssh_manifest_without_rewriting_it(tmp_path):
    """@brief 旧 legacy-ssh execution.json 的 resume 明确失败且不改写 manifest。"""
    from taic.orchestrator import RunStartError
    from tests.support.fakes import FakeClient

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0)
    manifest = orch.run_dir / "execution.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "evaluation": "remote", "transport": "legacy-ssh",
        "publishable_site_ids": ["local"], "publishable_epochs": ["e" * 64],
    }))
    before = manifest.read_text()
    with pytest.raises(RunStartError, match="legacy-ssh"):
        type(orch)(
            orch.task_name, orch.dtype, orch.size, orch.cfg, orch.search,
            run_dir=orch.run_dir, resuming=True, client=FakeClient([]),
            evaluate_fn=fake_eval([]), evaluator=GrpcShapedEvaluator(),
            publishable_site_ids={"local"}, publishable_epochs={"e" * 64},
        )
    assert manifest.read_text() == before


def test_resume_rejects_a_remote_manifest_without_a_transport(tmp_path):
    """@brief 缺 transport 的旧远程 manifest 按 legacy 解释后拒绝，不换成 gRPC。"""
    from taic.orchestrator import RunStartError
    from tests.support.fakes import FakeClient

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0)
    manifest = orch.run_dir / "execution.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "evaluation": "remote",
        "publishable_site_ids": ["local"], "publishable_epochs": ["e" * 64],
    }))
    before = manifest.read_text()
    with pytest.raises(RunStartError, match="no explicit evaluator transport"):
        type(orch)(
            orch.task_name, orch.dtype, orch.size, orch.cfg, orch.search,
            run_dir=orch.run_dir, resuming=True, client=FakeClient([]),
            evaluate_fn=fake_eval([]), evaluator=GrpcShapedEvaluator(),
            publishable_site_ids={"local"}, publishable_epochs={"e" * 64},
        )
    assert manifest.read_text() == before


class NpuShapedEvaluator(RecordingEvaluator):
    """Host-adapter double for the npu-source contract (no device, no gRPC)."""

    WIRE_DIGEST = "3" * 64
    DEVICE_GROUP = {
        "device_group_id": "vendor-a-group-0",
        "vendor_id": "vendor-a",
        "profile_digest": "1" * 64,
        "topology_digest": "2" * 64,
        "resource_generation": "1",
    }

    def __init__(self) -> None:
        super().__init__()
        self.wire_calls = 0
        self._provenance = None

    def execution_metadata(self) -> dict:
        from tcenv.remote import contracts

        return {
            "transport": "grpc",
            "api_contract_id": contracts.NPU_SOURCE_API_CONTRACT_ID,
            "evidence_contract_id": contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
            "request_digest_algorithm": contracts.REQUEST_DIGEST_ALGORITHM,
            "result_signature_algorithm": contracts.RESULT_SIGNATURE_ALGORITHM,
            "result_signature_key_ids": ["sim-site/key-1"],
        }

    def wire_request_digest(self, request) -> str:
        self.wire_calls += 1
        return self.WIRE_DIGEST

    @property
    def provenance(self):
        return self._provenance

    def formal(self, request):
        from tcenv.remote import contracts
        from tcenv.remote.contracts import NPU_TIMING_EVIDENCE_SCHEMA_ID

        assert request.contract_request_digest == self.WIRE_DIGEST, "host must bind the wire digest first"
        self.requests.append(request)
        report = self.base(
            request.task, request.dtype, request.size, request.source,
            mode=request.mode, seed=request.seed, reps=request.reps,
        )
        evidence = {
            "schema_id": NPU_TIMING_EVIDENCE_SCHEMA_ID,
            "candidate_digest": {"algorithm": "sha256", "value": request.candidate_sha256},
            "device_group": dict(self.DEVICE_GROUP),
            "timing": {
                "source": "DEVICE_EVENT", "clock_domain": "host-double-clock",
                "synchronization_completed": True, "device_elapsed_ns": 1_000,
                "host_api_elapsed_ns": 2_000, "launches": 1,
            },
            "measurement_kind": "measured",
            "publication_allowed": True,
            "complete": True,
        }
        candidate = {"algorithm": "sha256", "value": request.candidate_sha256}
        attempt_id, issued_at = "attempt-1", "2026-01-01T00:00:00Z"
        # Mirror the real host adapter: the signed wire receipt plus host-only
        # envelope metadata.  The receipt id must bind the NPU identity.
        receipt = {
            "receipt_id": contracts.attempt_receipt_id(
                request_id=request.request_id, attempt_id=attempt_id,
                request_digest=request.request_digest, candidate_digest=candidate,
                device_group=self.DEVICE_GROUP, dialect=contracts.NPU_SOURCE_DIALECT,
            ),
            "request_id": request.request_id, "attempt_id": attempt_id,
            "request_digest": request.request_digest,
            "candidate_digest": candidate,
            "device_group": dict(self.DEVICE_GROUP),
            "lease_fence": "1", "session_sequence": "1", "issued_at": issued_at,
            "measurement_kind": "measured",
            "signature_key_id": "sim-site/key-1",
            "signature_algorithm": "hmac-sha256-v1",
        }
        provenance = replace(
            local_provenance(request), publication_allowed=True,
            request_digest=request.request_digest,
        )
        result = EvaluationResult(report=report, provenance=provenance, evidence=evidence, receipt=receipt)
        self._provenance = provenance
        return result


def test_npu_contract_run_binds_the_wire_digest_before_the_durable_row(tmp_path):
    """@brief host 编排必须用 `wire_request_digest` 绑定 NPU 行、Submit 与 resume 同一身份。"""
    evaluator = NpuShapedEvaluator()
    orch = make_orch(
        tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0,
        evaluator=evaluator, publishable_site_ids={"local"}, publishable_epochs={_test_epoch()},
    )
    source = "// speed=1.2\nkernel void demo() {}"
    report = orch._evaluate_report(source, mode="quick", seed=3, reps=1, label="init")
    assert report.status == "ok"
    assert evaluator.wire_calls == 1
    assert evaluator.requests[0].request_digest == NpuShapedEvaluator.WIRE_DIGEST

    rows = [
        path for path in (orch.run_dir / "evaluation_outbox").glob("*.json")
        if not path.name.endswith(".result.json") and path.name != "receipts.json"
    ]
    assert len(rows) == 1
    state = json.loads(rows[0].read_text())
    assert state["schema_version"] == 2
    assert state["request_digest"] == NpuShapedEvaluator.WIRE_DIGEST
    assert state["candidate_digest"]["value"] == evaluator.requests[0].candidate_sha256
    assert state["device_group"] == NpuShapedEvaluator.DEVICE_GROUP
    assert state["status"] == "completed"

    # Resume replays the retained envelope under the same identity: no second
    # Submit and no silently different digest.
    replay = orch._evaluate_report(source, mode="quick", seed=3, reps=1, label="init")
    assert replay.status == "ok"
    assert len(evaluator.requests) == 1
