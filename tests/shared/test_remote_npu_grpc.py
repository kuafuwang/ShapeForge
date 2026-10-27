"""P0 NPU source contract over the real six gRPC RPCs with fake vendor A."""

from __future__ import annotations

import secrets
import time
from pathlib import Path

import grpc
import pytest
from grpc_status import rpc_status

from tcenv.remote import contracts
from tcenv.remote import grpc_contract as gc
from tcenv.remote.v1 import evaluator_pb2 as pb
from tests.support.job_engine import (
    NPU_FEATURES,
    NPU_TARGET,
    NPU_TOKEN,
    npu_engine,
    npu_identity,
    npu_request,
)
from tests.support.vendor_sim.harness import NpuGrpcHarness
from tests.support.vendor_sim.ledger import VendorLedger

SOURCE = b"npu-kernel relu_kernel\n"
METADATA = (("authorization", f"Bearer {NPU_TOKEN}"),)


def _expected_pb(harness: NpuGrpcHarness) -> pb.ExpectedEvaluator:
    expected = harness.engine.expected_identities.evaluator
    return pb.ExpectedEvaluator(
        site_id=expected.site_id, device_id=expected.device_id,
        target_id=expected.target_id, worker_build=expected.worker_build,
        task_pack_digest=expected.task_pack_digest,
        evaluator_epoch=expected.evaluator_epoch,
    )


def _capabilities(harness: NpuGrpcHarness, *, features=NPU_FEATURES):
    return harness.factory.stub.GetCapabilities(
        pb.CapabilitiesRequest(
            api_contract_ids=[contracts.NPU_SOURCE_API_CONTRACT_ID],
            evidence_contract_ids=[contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID],
            required_features=list(features),
            client_nonce=secrets.token_hex(16),
            expected=_expected_pb(harness),
        ),
        metadata=METADATA, timeout=10,
    )


def _submit(
    harness: NpuGrpcHarness, capability, request_id: str, *, source: bytes = SOURCE,
    candidate_timeout_ms: int = 3000,
):
    external = npu_request(request_id, source=source, candidate_timeout_ms=candidate_timeout_ms)
    snapshot = harness.factory.stub.Submit(
        pb.SubmitRequest(
            request=gc.request_to_proto(external, dialect=contracts.NPU_SOURCE_DIALECT),
            request_digest=external.request_digest,
            expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
            capability_id=capability.capability_id,
            required_features=list(NPU_FEATURES),
        ),
        metadata=METADATA, timeout=10,
    )
    return external, snapshot


def _get_job(harness: NpuGrpcHarness, external):
    return harness.factory.stub.GetJob(
        pb.JobRef(
            request_id=external.request_id, request_digest=external.request_digest,
            expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
        ),
        metadata=METADATA, timeout=10,
    )


def _wait_state(harness: NpuGrpcHarness, external, states: set[str], timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    snapshot = _get_job(harness, external)
    while gc.state_from_proto(snapshot.state) not in states and time.monotonic() < deadline:
        time.sleep(0.02)
        snapshot = _get_job(harness, external)
    assert gc.state_from_proto(snapshot.state) in states, (
        f"timed out waiting for {states}: {gc.state_from_proto(snapshot.state)}"
    )
    return snapshot


def _error_detail(exc: grpc.RpcError):
    status = rpc_status.from_call(exc)
    assert status is not None
    for item in status.details:
        if item.Is(pb.EvaluationErrorDetail.DESCRIPTOR):
            detail = pb.EvaluationErrorDetail()
            item.Unpack(detail)
            return detail
    return None


def test_capabilities_submit_getjob_watch_and_simulated_publication(tmp_path):
    with NpuGrpcHarness(tmp_path) as harness:
        capability = _capabilities(harness)
        assert capability.api_contract_id == contracts.NPU_SOURCE_API_CONTRACT_ID
        assert capability.evidence_contract_id == contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID
        assert "npu-source-artifact" in capability.supported_features
        assert len(capability.targets) == 1
        target = capability.targets[0]
        assert list(target.artifact_kinds) == ["source"]
        assert list(target.launch_kinds) == ["KERNEL"]
        assert list(target.launch_subtypes) == ["SIMT"]
        assert list(target.timing_sources) == ["DEVICE_EVENT"]
        assert len(target.device_groups) == 1
        assert target.device_groups[0].device_group_id == "vendor-a-group-0"
        assert target.device_groups[0].state == "available"
        assert capability.provenance.device_group.device_group_id == "vendor-a-group-0"
        assert capability.provenance.measurement_kind == pb.MEASUREMENT_KIND_SIMULATED
        assert capability.provenance.publication_allowed is False

        external, submitted = _submit(harness, capability, "npu-six-rpc-1")
        assert submitted.request_id == external.request_id
        assert submitted.request_digest == external.request_digest
        snapshot = _wait_state(harness, external, {"completed"})
        assert snapshot.HasField("result")
        receipt = gc.receipt_from_proto(
            snapshot.result.receipt, dialect=contracts.NPU_SOURCE_DIALECT,
        )
        assert receipt["candidate_digest"] == external.candidate_digest
        assert receipt["device_group"]["device_group_id"] == "vendor-a-group-0"
        assert snapshot.provenance.measurement_kind == pb.MEASUREMENT_KIND_SIMULATED
        assert snapshot.provenance.publication_allowed is False

        # Watch streams the terminal snapshot from the retained beginning.
        events = list(harness.factory.stub.Watch(
            pb.WatchRequest(
                job=pb.JobRef(
                    request_id=external.request_id,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                ),
                after_sequence=0,
            ),
            metadata=METADATA, timeout=10,
        ))
        assert events
        terminal = [event for event in events if event.HasField("snapshot")]
        assert terminal
        assert gc.state_from_proto(terminal[-1].snapshot.state) == "completed"
        assert terminal[-1].snapshot.HasField("result")


def test_explicit_cancel_reaches_terminal_and_releases_device_resources(tmp_path):
    with NpuGrpcHarness(tmp_path, worker_env={"TCENV_VENDOR_A_SLOW": "1"}) as harness:
        capability = _capabilities(harness)
        external, submitted = _submit(harness, capability, "npu-cancel-1")
        ledger = VendorLedger(harness.ledger_path)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ledger.live_resources():
            time.sleep(0.02)
        assert ledger.live_resources(), "fake vendor A never prepared device resources"
        snapshot = harness.factory.stub.Cancel(
            pb.CancelRequest(
                job=pb.JobRef(
                    request_id=external.request_id,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                ),
                reason="p0 explicit cancel",
            ),
            metadata=METADATA, timeout=10,
        )
        assert snapshot.cancel_requested is True
        final = _wait_state(harness, external, {"cancelled"})
        assert not final.HasField("result")
        assert ledger.live_resources() == []


def test_cancel_after_the_terminal_state_does_not_touch_the_device_lease(tmp_path):
    with NpuGrpcHarness(tmp_path) as harness:
        capability = _capabilities(harness)
        external, _ = _submit(harness, capability, "npu-cancel-terminal-1")
        final = _wait_state(harness, external, {"completed"})
        attempt_id = final.current_attempt.attempt_id
        fence = final.current_attempt.lease_fence
        ledger = VendorLedger(harness.ledger_path)
        assert ledger.read()["cancelled"] == []
        # Cancelling an already finished job is a durable no-op: the released
        # attempt/fence must not be marked cancelled a second time.
        harness.factory.stub.Cancel(
            pb.CancelRequest(
                job=pb.JobRef(
                    request_id=external.request_id,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                ),
                reason="cancel after completion",
            ),
            metadata=METADATA, timeout=10,
        )
        assert ledger.is_cancelled(attempt_id, fence) is False
        assert _get_job(harness, external).HasField("result")
        # The same attempt fed straight to the controller *would* be marked; the
        # terminal-state guard in the Cancel path is what prevents it.
        harness.engine.device_controller.cancel(attempt_id, fence)
        assert ledger.is_cancelled(attempt_id, fence) is True


def test_retry_interrupted_attempt_uses_new_capability_and_fence(tmp_path):
    with NpuGrpcHarness(
        tmp_path, worker_env={"TCENV_VENDOR_A_ABORT_AFTER_LAUNCH": "1"},
    ) as harness:
        capability = _capabilities(harness)
        external, _ = _submit(harness, capability, "npu-retry-1")
        interrupted = _wait_state(harness, external, {"interrupted"})
        assert interrupted.HasField("current_attempt")
        old_attempt = interrupted.current_attempt
        assert old_attempt.lease_fence == 1
        assert interrupted.failure.retryable is True
        # The worker was killed after launch; the controller proved the exact
        # attempt/fence resources were released, so Retry is allowed.
        assert VendorLedger(harness.ledger_path).live_resources() == []

        fresh = _capabilities(harness)
        retry = harness.factory.stub.Retry(
            pb.RetryRequest(
                job=pb.JobRef(
                    request_id=external.request_id,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                ),
                interrupted_attempt_id=old_attempt.attempt_id,
                capability_id=fresh.capability_id,
                required_features=list(NPU_FEATURES),
            ),
            metadata=METADATA, timeout=10,
        )
        assert retry.current_attempt.attempt_id != old_attempt.attempt_id
        assert retry.current_attempt.lease_fence > old_attempt.lease_fence
        final = _wait_state(harness, external, {"completed"})
        assert final.current_attempt.lease_fence == retry.current_attempt.lease_fence
        assert final.HasField("result")
        receipt = gc.receipt_from_proto(
            final.result.receipt, dialect=contracts.NPU_SOURCE_DIALECT,
        )
        assert receipt["attempt_id"] == retry.current_attempt.attempt_id
        assert receipt["lease_fence"] == str(retry.current_attempt.lease_fence)


def test_stale_attempt_fence_result_is_rejected(tmp_path):
    with NpuGrpcHarness(
        tmp_path, worker_env={"TCENV_VENDOR_A_ABORT_AFTER_LAUNCH": "1"},
    ) as harness:
        capability = _capabilities(harness)
        external, _ = _submit(harness, capability, "npu-stale-1")
        interrupted = _wait_state(harness, external, {"interrupted"})
        old_attempt = harness.engine.store._attempt(
            harness.engine.store._conn.execute(
                "SELECT * FROM attempts WHERE attempt_id=?",
                (interrupted.current_attempt.attempt_id,),
            ).fetchone()
        )
        fresh = _capabilities(harness)
        harness.factory.stub.Retry(
            pb.RetryRequest(
                job=pb.JobRef(
                    request_id=external.request_id,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                ),
                interrupted_attempt_id=interrupted.current_attempt.attempt_id,
                capability_id=fresh.capability_id,
                required_features=list(NPU_FEATURES),
            ),
            metadata=METADATA, timeout=10,
        )
        # A late result from the old attempt/fence must not commit against the
        # new attempt.  The store records a stale-result diagnostic instead.
        stale = harness.engine.store.complete_attempt(
            external.request_id, old_attempt,
            report={
                "task": "relu", "dtype": "fp16", "size": "small", "mode": "quick",
                "seed": 3, "status": "ok", "stage": "done", "candidate_us": 1.0,
                "speedup": 1.0, "elapsed_s": 0.0, "target_id": NPU_TARGET,
                "protocol_version": "3", "simulated": True,
            },
            evidence=None,
        )
        assert stale.result is None
        final = _wait_state(harness, external, {"completed", "interrupted", "failed", "cancelled"})
        assert final.result is None or final.current_attempt.attempt_id != old_attempt.attempt_id


def test_feature_contract_and_digest_fail_closed_before_create(tmp_path):
    with NpuGrpcHarness(tmp_path) as harness:
        capability = _capabilities(harness)
        external = npu_request("npu-negative-1")
        # Unsupported required feature is rejected by GetCapabilities.
        with pytest.raises(grpc.RpcError) as excinfo:
            _capabilities(harness, features=(*NPU_FEATURES, "unsupported-npu-feature"))
        assert excinfo.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        detail = _error_detail(excinfo.value)
        assert detail is not None and "unsupported-npu-feature" in detail.missing_features

        # A digest mismatch is rejected before durable acceptance.
        with pytest.raises(grpc.RpcError) as excinfo:
            harness.factory.stub.Submit(
                pb.SubmitRequest(
                    request=gc.request_to_proto(external, dialect=contracts.NPU_SOURCE_DIALECT),
                    request_digest="1" * 64,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                    capability_id=capability.capability_id,
                    required_features=list(NPU_FEATURES),
                ),
                metadata=METADATA, timeout=10,
            )
        assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        # A legacy contract pair cannot select this NPU target.
        with pytest.raises(grpc.RpcError) as excinfo:
            harness.factory.stub.GetCapabilities(
                pb.CapabilitiesRequest(
                    api_contract_ids=[contracts.API_CONTRACT_ID],
                    evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
                    required_features=list(NPU_FEATURES),
                    client_nonce=secrets.token_hex(16),
                    expected=_expected_pb(harness),
                ),
                metadata=METADATA, timeout=10,
            )
        assert excinfo.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert harness.engine.store._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_npu_contract_rejects_legacy_or_other_device_group_submission(tmp_path):
    with NpuGrpcHarness(tmp_path) as harness:
        capability = _capabilities(harness)
        external = npu_request("npu-group-1")
        from dataclasses import replace
        from tcenv.remote.job_types import (
            CandidateDigest, DeviceGroupBinding, NpuExecutionSpec, NpuSourceRequest,
        )

        original = external.npu
        wrong_group = DeviceGroupBinding(
            device_group_id=original.execution.device_group.device_group_id,
            vendor_id=original.execution.device_group.vendor_id,
            profile_digest=original.execution.device_group.profile_digest,
            topology_digest=original.execution.device_group.topology_digest,
            resource_generation="2",
        )
        wrong = replace(external, npu=NpuSourceRequest(
            candidate_digest=original.candidate_digest,
            execution=NpuExecutionSpec(device_group=wrong_group),
        ))
        with pytest.raises(grpc.RpcError) as excinfo:
            harness.factory.stub.Submit(
                pb.SubmitRequest(
                    request=gc.request_to_proto(wrong, dialect=contracts.NPU_SOURCE_DIALECT),
                    request_digest=wrong.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                    capability_id=capability.capability_id,
                    required_features=list(NPU_FEATURES),
                ),
                metadata=METADATA, timeout=10,
            )
        assert excinfo.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert harness.engine.store._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0

        # A legacy-shaped request is not silently interpreted as NPU.
        legacy_message = pb.EvaluationRequest(
            kind=pb.EVALUATION_KIND_FORMAL_EVAL,
            request_id="legacy-shape-1",
            run_id="npu-run-1",
            session_id="npu-session-1",
            task="relu",
            dtype="fp16",
            size="small",
            target_id=NPU_TARGET,
            task_fingerprint=external.task_fingerprint,
            candidate_sha256=external.candidate_digest["value"],
            source=SOURCE,
            mode=pb.EVALUATION_MODE_QUICK,
            reps=2,
            queue_timeout_ms=3000,
            candidate_timeout_ms=3000,
        )
        with pytest.raises(grpc.RpcError) as excinfo:
            harness.factory.stub.Submit(
                pb.SubmitRequest(
                    request=legacy_message,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                    capability_id=capability.capability_id,
                    required_features=list(NPU_FEATURES),
                ),
                metadata=METADATA, timeout=10,
            )
        assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_npu_candidate_timeout_completes_a_diagnostic_timeout_report(tmp_path):
    """@brief 候选超时保持 timeout 语义，且 NPU 证据是诚实的诊断行。"""
    with NpuGrpcHarness(tmp_path, worker_env={"TCENV_VENDOR_A_SLOW": "1"}) as harness:
        capability = _capabilities(harness)
        external, _ = _submit(harness, capability, "npu-timeout-1", candidate_timeout_ms=200)
        final = _wait_state(harness, external, {"completed"})
        assert final.HasField("result")
        report = gc.report_json(gc.blob_from_proto(final.result.report, allow_empty=False)[0])
        assert report["status"] == "timeout"
        assert report["stage"] == "timeout"
        evidence = gc.evidence_json(gc.blob_from_proto(final.result.evidence, allow_empty=True)[0])
        assert evidence["complete"] is False
        assert evidence["timing"]["synchronization_completed"] is False
        assert evidence["timing"]["device_elapsed_ns"] == 0
        assert evidence["timing"]["launches"] == 0
        assert evidence["timing"]["clock_domain"] == contracts.NPU_UNOBSERVED_CLOCK_DOMAIN
        assert evidence["publication_allowed"] is False
        receipt = gc.receipt_from_proto(
            final.result.receipt, dialect=contracts.NPU_SOURCE_DIALECT,
        )
        assert receipt["candidate_digest"] == external.candidate_digest
        assert evidence["candidate_digest"] == receipt["candidate_digest"]
        assert final.provenance.publication_allowed is False
        # The job is terminal, so the result is durable and replayable rather
        # than an interrupt that needs an explicit Retry.
        assert not final.HasField("failure")


def test_npu_selfcheck_hang_still_records_a_timeout_report(tmp_path):
    """@brief selfcheck 的 hang 变成可记账的 timeout 报告，而不是协议失败。"""
    with NpuGrpcHarness(tmp_path, worker_env={"TCENV_VENDOR_A_SLOW": "1"}) as harness:
        capability = _capabilities(harness)
        external = npu_request(
            "npu-selfcheck-timeout-1", source=SOURCE, kind="selfcheck", mode="quick",
            candidate_timeout_ms=200,
        )
        harness.factory.stub.Submit(
            pb.SubmitRequest(
                request=gc.request_to_proto(external, dialect=contracts.NPU_SOURCE_DIALECT),
                request_digest=external.request_digest,
                expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                capability_id=capability.capability_id,
                required_features=list(NPU_FEATURES),
            ),
            metadata=METADATA, timeout=10,
        )
        final = _wait_state(harness, external, {"completed"})
        report = gc.report_json(gc.blob_from_proto(final.result.report, allow_empty=False)[0])
        assert report["status"] == "timeout"
        assert report["simulated"] is True
        evidence = gc.evidence_json(gc.blob_from_proto(final.result.evidence, allow_empty=True)[0])
        assert evidence["complete"] is False and evidence["publication_allowed"] is False
