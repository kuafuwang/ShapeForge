"""P0 NPU source contract: canonical digest, receipt, signature and outbox chain."""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tcenv.evaluation import EvaluationRequest, EvaluationUnavailable
from tcenv.remote import contracts
from tcenv.remote.job_types import JobError, StateCorrupt
from tests.support.job_engine import (
    NPU_TARGET,
    NPU_TASK_FINGERPRINT,
    NPU_TOKEN,
    npu_engine,
    npu_get,
    npu_request,
    npu_submit,
)
from tests.support.vendor_sim.vendor_a import FakeVendorAController
from tests.support.vendor_sim.harness import (
    NPU_HOST_SIGNATURE_KEY,
    NPU_SIGNING_KEY_ID,
    NpuGrpcHarness,
)
from taic.evaluation_outbox import EvaluationOutbox, bind_wire_digest

SOURCE = b"npu-kernel relu_kernel\n"


def _legacy_request(request_id: str = "npu-outbox-1", *, source: bytes = SOURCE) -> EvaluationRequest:
    return EvaluationRequest(
        kind="formal_eval", request_id=request_id, run_id="npu-run-1",
        session_id="npu-session-1", task="relu", dtype="fp16", size="small",
        target_id=NPU_TARGET, task_fingerprint=NPU_TASK_FINGERPRINT,
        source=source.decode("utf-8"), mode="quick", seed=3, reps=2,
        queue_timeout_s=3.0, candidate_timeout_s=3.0,
    )


def _npu_outbox(root: Path, *, signature_key: bytes | None = None) -> EvaluationOutbox:
    return EvaluationOutbox(
        root,
        signature_key=NPU_HOST_SIGNATURE_KEY if signature_key is None else signature_key,
        transport="grpc",
        api_contract_id=contracts.NPU_SOURCE_API_CONTRACT_ID,
        evidence_contract_id=contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
    )


def test_npu_canonical_request_binds_source_device_group_and_new_document_type():
    request = npu_request("npu-digest-1", source=SOURCE)
    document = request.digest_document()
    assert document["document_type"] == contracts.NPU_SOURCE_REQUEST_DOCUMENT_TYPE
    assert document["document_version"] == contracts.NPU_SOURCE_DOCUMENT_VERSION
    assert "candidate_digest" in document["request"]
    assert "candidate_sha256" not in document["request"]
    assert document["request"]["candidate_digest"] == request.candidate_digest
    assert document["request"]["execution"]["device_group"]["device_group_id"] == "vendor-a-group-0"

    changed_source = npu_request("npu-digest-1", source=SOURCE + b"\n")
    assert changed_source.request_digest != request.request_digest

    execution = request.npu.execution
    from tcenv.remote.job_types import DeviceGroupBinding, NpuExecutionSpec, NpuSourceRequest

    changed_group = DeviceGroupBinding(
        device_group_id=execution.device_group.device_group_id,
        vendor_id=execution.device_group.vendor_id,
        profile_digest=execution.device_group.profile_digest,
        topology_digest=execution.device_group.topology_digest,
        resource_generation="2",
    )
    changed = replace(request, npu=NpuSourceRequest(
        candidate_digest=request.npu.candidate_digest,
        execution=NpuExecutionSpec(device_group=changed_group),
    ))
    assert changed.request_digest != request.request_digest


def test_npu_contract_rejects_legacy_or_unsupported_execution_shape():
    request = npu_request("npu-shape-1", source=SOURCE)
    fields = request.digest_fields()
    fields["execution"] = {**fields["execution"], "artifact_kind": "bundle"}
    with pytest.raises(contracts.ContractError):
        contracts.request_digest_document(
            fields, source_bytes=SOURCE, dialect=contracts.NPU_SOURCE_DIALECT,
        )
    fields = request.digest_fields()
    fields["execution"] = {**fields["execution"], "required_timing_sources": ["HOST_WALL"]}
    with pytest.raises(contracts.ContractError):
        contracts.request_digest_document(
            fields, source_bytes=SOURCE, dialect=contracts.NPU_SOURCE_DIALECT,
        )
    with pytest.raises(contracts.ContractError):
        contracts.dialect_for(
            contracts.NPU_SOURCE_API_CONTRACT_ID, contracts.EVIDENCE_CONTRACT_ID,
        )


def test_npu_result_signature_rejects_every_tampered_chain_link(tmp_path):
    with NpuGrpcHarness(tmp_path) as harness:
        request = bind_wire_digest(_legacy_request(), harness.factory, _npu_outbox(tmp_path / "bind"))
        external = npu_request(request.request_id, source=SOURCE)
        assert request.request_digest == external.request_digest
        result = harness.factory.formal(request)
        snapshot = harness.engine.get_job(
            request_id=external.request_id, request_digest=external.request_digest,
            expected=harness.engine.expected_identities, grant_token=NPU_TOKEN,
        )
        stored = snapshot.result
        assert stored is not None
        document = {
            "provenance": stored.provenance,
            "report": stored.report_descriptor,
            "evidence": stored.evidence_descriptor,
            "receipt": stored.receipt,
        }
        signature = stored.signature["signature"]
        assert contracts.verify_result_signature(
            document, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
            signature=signature, dialect=contracts.NPU_SOURCE_DIALECT,
        )
        assert result.receipt["candidate_digest"] == external.candidate_digest
        assert "candidate_sha256" not in result.receipt
        assert result.provenance.measurement_kind == "simulated"
        assert result.evidence["publication_allowed"] is False

        tampered_receipt = copy.deepcopy(document)
        tampered_receipt["receipt"]["candidate_digest"]["value"] = "0" * 64
        with pytest.raises(contracts.ContractError):
            contracts.verify_result_signature(
                tampered_receipt, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
                signature=signature, dialect=contracts.NPU_SOURCE_DIALECT,
            )
        with pytest.raises(contracts.ContractError):
            contracts.receipt_document(
                tampered_receipt["receipt"], dialect=contracts.NPU_SOURCE_DIALECT,
            )

        tampered_group = copy.deepcopy(document)
        tampered_group["provenance"]["device_group"]["resource_generation"] = "2"
        with pytest.raises(contracts.ContractError):
            contracts.verify_result_signature(
                tampered_group, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
                signature=signature, dialect=contracts.NPU_SOURCE_DIALECT,
            )

        tampered_evidence = copy.deepcopy(document)
        tampered_evidence["evidence"]["sha256"] = "1" * 64
        assert not contracts.verify_result_signature(
            tampered_evidence, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
            signature=signature, dialect=contracts.NPU_SOURCE_DIALECT,
        )

        # The legacy verifier must not accept a NPU document merely because the
        # HMAC key happens to match.
        with pytest.raises(contracts.ContractError):
            contracts.verify_result_signature(
                document, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
                signature=signature, dialect=contracts.LEGACY_DIALECT,
            )


def test_npu_outbox_row_version_digest_and_resume_tamper(tmp_path):
    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        legacy = _legacy_request()
        external = npu_request(legacy.request_id, source=SOURCE)
        wire_request = replace(legacy, contract_request_digest=external.request_digest)
        outbox = _npu_outbox(root)
        assert outbox.begin(wire_request, epoch=harness.factory.epoch) is None
        result = harness.factory.formal(wire_request)
        outbox.complete(wire_request, result)

        state_path = outbox._state_path(wire_request.request_id)
        row = json.loads(state_path.read_text(encoding="utf-8"))
        assert row["schema_version"] == 2
        assert row["candidate_digest"] == external.candidate_digest
        assert row["device_group"] == result.receipt["device_group"]
        assert row["request_digest"] == external.request_digest

        resumed = _npu_outbox(root)
        replay = resumed.begin(wire_request, epoch=harness.factory.epoch)
        assert replay is not None
        assert replay.receipt["candidate_digest"] == external.candidate_digest

        # Tamper the durable result envelope: resume must fail closed.
        result_path = outbox._result_path(wire_request.request_id)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["receipt"]["candidate_digest"]["value"] = "2" * 64
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _npu_outbox(root).begin(wire_request, epoch=harness.factory.epoch)

        # Restore the result, then tamper the outbox row's CandidateDigest.
        payload["receipt"]["candidate_digest"]["value"] = external.candidate_digest["value"]
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        row = json.loads(state_path.read_text(encoding="utf-8"))
        row["candidate_digest"]["value"] = "3" * 64
        state_path.write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _npu_outbox(root).begin(wire_request, epoch=harness.factory.epoch)


def test_npu_request_rejects_a_dual_candidate_identity_and_legacy_mapping():
    from tcenv.remote.job_types import ExternalEvaluationRequest, JobError
    from tcenv.remote import grpc_contract as gc

    request = npu_request("npu-dual-1", source=SOURCE)
    with pytest.raises(JobError, match="must not also carry"):
        ExternalEvaluationRequest(
            kind=request.kind, request_id=request.request_id, run_id=request.run_id,
            session_id=request.session_id, task=request.task, dtype=request.dtype,
            size=request.size, target_id=request.target_id,
            task_fingerprint=request.task_fingerprint,
            candidate_sha256=request.npu.candidate_digest.value,
            source=request.source, mode=request.mode, seed=request.seed,
            reps=request.reps, queue_timeout_ms=request.queue_timeout_ms,
            candidate_timeout_ms=request.candidate_timeout_ms, cases=None,
            npu=request.npu,
        )
    # The NPU contract never silently downgrades to the legacy worker/evaluator
    # request, and it cannot be announced under the legacy contract id.
    with pytest.raises(JobError, match="no legacy request mapping"):
        request.to_legacy()
    with pytest.raises(JobError, match="request contract differs"):
        gc.request_to_proto(request, dialect=contracts.LEGACY_DIALECT)


def test_npu_outbox_row_device_group_tamper_fails_closed(tmp_path):
    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        legacy = _legacy_request("npu-outbox-group-1")
        wire = harness.factory.wire_request_digest(legacy)
        external = npu_request(legacy.request_id, source=SOURCE)
        assert wire == external.request_digest
        request = replace(legacy, contract_request_digest=wire)
        outbox = _npu_outbox(root)
        assert outbox.begin(request, epoch=harness.factory.epoch) is None
        result = harness.factory.formal(request)
        outbox.complete(request, result)

        state_path = outbox._state_path(request.request_id)
        row = json.loads(state_path.read_text(encoding="utf-8"))
        assert row["device_group"] == result.receipt["device_group"]

        # A durable row whose device group was rewritten must not replay the
        # retained result: the row is part of the CandidateDigest chain.
        row["device_group"]["resource_generation"] = "9"
        state_path.write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _npu_outbox(root).begin(request, epoch=harness.factory.epoch)

        # A truncated group is corrupt rather than "absent".
        row["device_group"] = {"device_group_id": "vendor-a-group-0"}
        state_path.write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _npu_outbox(root).begin(request, epoch=harness.factory.epoch)


def test_npu_stale_durable_request_digest_fails_before_submit(tmp_path):
    with NpuGrpcHarness(tmp_path) as harness:
        legacy = _legacy_request("npu-stale-wire-1")
        stale = replace(legacy, contract_request_digest="0" * 64)
        with pytest.raises(EvaluationUnavailable, match="durable host request digest"):
            harness.factory.formal(stale)
        # Nothing was durably accepted: a changed device group or request can
        # never be resumed into a silently different job.
        assert harness.engine.store._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0

        fresh = replace(legacy, contract_request_digest=harness.factory.wire_request_digest(legacy))
        assert harness.factory.formal(fresh).receipt["candidate_digest"]["value"] == (
            hashlib.sha256(SOURCE).hexdigest()
        )


def test_legacy_outbox_stays_version_one_and_has_no_candidate_digest(tmp_path):
    request = _legacy_request("legacy-outbox-1")
    outbox = EvaluationOutbox(tmp_path, transport="grpc")
    outbox.begin(request, epoch="e" * 64)
    row = json.loads(outbox._state_path(request.request_id).read_text(encoding="utf-8"))
    assert row["schema_version"] == 1
    assert row["candidate_sha256"] == request.candidate_sha256
    assert "candidate_digest" not in row


def test_npu_outbox_and_client_require_the_host_wire_digest(tmp_path):
    """@brief 没有 wire digest 的 NPU 请求不能静默退回旧 digest，也不能落盘或 Submit。"""
    with NpuGrpcHarness(tmp_path) as harness:
        legacy = _legacy_request("npu-no-wire-1")
        assert legacy.contract_request_digest == ""
        outbox = _npu_outbox(tmp_path / "run")
        with pytest.raises(EvaluationUnavailable, match="wire request digest"):
            outbox.begin(legacy, epoch=harness.factory.epoch)
        with pytest.raises(EvaluationUnavailable, match="wire request digest"):
            harness.factory.formal(legacy)
        with pytest.raises(EvaluationUnavailable, match="wire request digest"):
            harness.factory.get_job(legacy)
        assert not outbox._state_path(legacy.request_id).exists()
        assert harness.engine.store._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        # A legacy-only host keeps its historical digest; the binding helper is
        # a no-op when the outbox is not the NPU contract.
        assert bind_wire_digest(legacy, harness.factory, None) is legacy
        legacy_outbox = EvaluationOutbox(tmp_path / "legacy", transport="grpc")
        assert bind_wire_digest(legacy, object(), legacy_outbox) is legacy


def test_npu_outbox_resume_replays_the_bound_wire_identity(tmp_path):
    """@brief 生产接线（bind_wire_digest）下 begin→formal→complete→resume 身份一致且可重放。"""
    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        outbox = _npu_outbox(root)
        request = bind_wire_digest(_legacy_request("npu-resume-1"), harness.factory, outbox)
        external = npu_request(request.request_id, source=SOURCE)
        assert request.request_digest == external.request_digest == harness.factory.wire_request_digest(request)

        assert outbox.begin(request, epoch=harness.factory.epoch) is None
        result = harness.factory.formal(request)
        assert result.provenance.request_digest == request.request_digest
        assert result.receipt["request_digest"] == request.request_digest
        outbox.complete(request, result)

        row = json.loads(outbox._state_path(request.request_id).read_text(encoding="utf-8"))
        assert row["request_digest"] == external.request_digest
        assert row["candidate_digest"] == external.candidate_digest
        replay = _npu_outbox(root).begin(request, epoch=harness.factory.epoch)
        assert replay is not None
        assert replay.receipt["request_digest"] == external.request_digest
        assert replay.provenance.request_digest == external.request_digest

        # A legacy-keyed row for the same request id is a different job: the
        # durable row must refuse it instead of replaying under the wrong id.
        with pytest.raises(EvaluationUnavailable):
            _npu_outbox(root).begin(_legacy_request("npu-resume-1"), epoch=harness.factory.epoch)


def test_npu_timing_evidence_requires_honest_completion(tmp_path):
    """@brief ok 报告必须带完整设备事件；timeout 报告不得伪造已完成的设备事件。"""
    request = npu_request("npu-evidence-1", source=SOURCE)
    complete = {
        "schema_id": contracts.NPU_TIMING_EVIDENCE_SCHEMA_ID,
        "candidate_digest": request.candidate_digest,
        "device_group": request.npu.execution.device_group.to_dict(),
        "timing": {
            "source": "DEVICE_EVENT",
            "clock_domain": "vendor-a-device-event-clock-v1",
            "synchronization_completed": True,
            "device_elapsed_ns": 1_000,
            "host_api_elapsed_ns": 2_000,
            "launches": 1,
        },
        "measurement_kind": "simulated",
        "publication_allowed": False,
        "complete": True,
    }
    contracts.npu_timing_evidence_document(complete, report_status="ok")

    diagnostic = copy.deepcopy(complete)
    diagnostic["complete"] = False
    diagnostic["timing"].update({
        "synchronization_completed": False,
        "device_elapsed_ns": 0,
        "launches": 0,
        "clock_domain": contracts.NPU_UNOBSERVED_CLOCK_DOMAIN,
    })
    contracts.npu_timing_evidence_document(diagnostic, report_status="timeout")
    # An ok report may not be backed by a row that never observed the device.
    with pytest.raises(contracts.ContractError, match="ok report"):
        contracts.npu_timing_evidence_document(diagnostic, report_status="ok")
    # An incomplete row may only be diagnostic: no device time, no launch and no
    # observed clock domain.
    with pytest.raises(contracts.ContractError, match="device time"):
        contracts.npu_timing_evidence_document(
            {**diagnostic, "timing": {**diagnostic["timing"], "device_elapsed_ns": 5}},
            report_status="timeout",
        )
    with pytest.raises(contracts.ContractError, match="completed synchronization"):
        contracts.npu_timing_evidence_document(
            {**diagnostic, "timing": {**diagnostic["timing"], "synchronization_completed": True}},
            report_status="timeout",
        )
    with pytest.raises(contracts.ContractError, match="unobserved clock"):
        contracts.npu_timing_evidence_document(
            {**diagnostic, "timing": {**diagnostic["timing"], "clock_domain": "observed"}},
            report_status="timeout",
        )
    # A timeout report must not claim a completed device event.
    with pytest.raises(contracts.ContractError, match="timeout report"):
        contracts.npu_timing_evidence_document(complete, report_status="timeout")


def test_npu_outbox_rejects_a_rewritten_wire_receipt_on_replay(tmp_path):
    """@brief retained result 的 wire receipt 被改写（含 receipt_id）后，replay 必须拒绝。"""
    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        outbox = _npu_outbox(root)
        request = bind_wire_digest(_legacy_request("npu-receipt-tamper-1"), harness.factory, outbox)
        assert outbox.begin(request, epoch=harness.factory.epoch) is None
        result = harness.factory.formal(request)
        outbox.complete(request, result)
        assert _npu_outbox(root).begin(request, epoch=harness.factory.epoch) is not None

        result_path = outbox._result_path(request.request_id)
        original = json.loads(result_path.read_text(encoding="utf-8"))

        # 1. A rewritten receipt_id is refused even though every other field is
        #    intact: the id must recompute from request/attempt/candidate/group.
        tampered = copy.deepcopy(original)
        tampered["receipt"]["receipt_id"] = "not-a-valid-receipt"
        result_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable, match="receipt"):
            _npu_outbox(root).begin(request, epoch=harness.factory.epoch)

        # 2. Re-signing the tampered receipt does not help: the wire fields are
        #    validated independently of the host signature.
        tampered["signature"] = ""
        result_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable, match="receipt"):
            _npu_outbox(root).begin(request, epoch=harness.factory.epoch)

        # 3. A host-only metadata rewrite is still accepted (it is not wire data).
        metadata_only = copy.deepcopy(original)
        metadata_only["receipt"]["signature_key_id"] = "sim-npu-site/key-1"
        metadata_only["receipt"]["measurement_kind"] = "simulated"
        result_path.write_text(json.dumps(metadata_only), encoding="utf-8")
        assert _npu_outbox(root).begin(request, epoch=harness.factory.epoch) is not None

        # 4. The wire receipt's normative fields are validated too (the fence
        #    must stay a decimal string; the fence-to-attempt cross-check lives
        #    in the durable job store, which has the attempt row).
        malformed_fence = copy.deepcopy(original)
        malformed_fence["receipt"]["lease_fence"] = "not-a-fence"
        result_path.write_text(json.dumps(malformed_fence), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable, match="receipt"):
            _npu_outbox(root).begin(request, epoch=harness.factory.epoch)

        result_path.write_text(json.dumps(original), encoding="utf-8")
        assert _npu_outbox(root).begin(request, epoch=harness.factory.epoch) is not None


def test_npu_outbox_requires_a_host_signature_key(tmp_path):
    """@brief NPU outbox 没有宿主 HMAC key 就构造失败，不能靠 receipt_id 当认证。"""
    with pytest.raises(ValueError, match="host result signature key"):
        EvaluationOutbox(
            tmp_path / "unsigned",
            transport="grpc",
            api_contract_id=contracts.NPU_SOURCE_API_CONTRACT_ID,
            evidence_contract_id=contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
        )


def test_npu_replay_rejects_a_forged_but_self_consistent_envelope(tmp_path):
    """@brief 合法重算 receipt_id 的伪造 attempt/fence、report、evidence、provenance 都必须被拒。"""
    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        outbox = _npu_outbox(root)
        request = bind_wire_digest(_legacy_request("npu-forge-1"), harness.factory, outbox)
        assert outbox.begin(request, epoch=harness.factory.epoch) is None
        result = harness.factory.formal(request)
        outbox.complete(request, result)

        result_path = outbox._result_path(request.request_id)
        original = json.loads(result_path.read_text(encoding="utf-8"))
        assert _npu_outbox(root).begin(request, epoch=harness.factory.epoch) is not None

        def replay(payload):
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                return _npu_outbox(root).begin(request, epoch=harness.factory.epoch)
            finally:
                result_path.write_text(json.dumps(original), encoding="utf-8")

        # 1. Forge the attempt/fence, then recompute a *valid* receipt id from the
        #    forged fields.  Only the host HMAC can tell this envelope apart.
        forged = copy.deepcopy(original)
        forged["receipt"]["attempt_id"] = "attempt-forged"
        forged["receipt"]["lease_fence"] = "999"
        forged["receipt"]["receipt_id"] = contracts.attempt_receipt_id(
            request_id=forged["receipt"]["request_id"],
            attempt_id="attempt-forged", request_digest=forged["receipt"]["request_digest"],
            candidate_digest=forged["receipt"]["candidate_digest"],
            device_group=forged["receipt"]["device_group"],
            dialect=contracts.NPU_SOURCE_DIALECT,
        )
        wire_fields = (
            "receipt_id", "request_id", "attempt_id", "request_digest", "candidate_digest",
            "device_group", "lease_fence", "session_sequence", "issued_at",
        )
        contracts.receipt_document(
            {key: forged["receipt"][key] for key in wire_fields},
            dialect=contracts.NPU_SOURCE_DIALECT,
        )
        with pytest.raises(EvaluationUnavailable, match="signature"):
            replay(forged)

        # 2. Rewrite a signed report field that the timing rule does not cover:
        #    only the host HMAC can reject it.
        faster = copy.deepcopy(original)
        faster["report"]["speedup"] = 9.9
        with pytest.raises(EvaluationUnavailable, match="signature"):
            replay(faster)

        # 3. Promote the evidence to publishable/measured after the fact.
        promoted = copy.deepcopy(original)
        promoted["evidence"]["measurement_kind"] = "measured"
        promoted["evidence"]["publication_allowed"] = True
        with pytest.raises(EvaluationUnavailable, match="signature"):
            replay(promoted)

        # 4. Rewrite the provenance identity.
        provenance = copy.deepcopy(original)
        provenance["provenance"]["site_id"] = "another-site"
        with pytest.raises(EvaluationUnavailable, match="signature"):
            replay(provenance)

        # 5. A different (rotated or revoked) host key cannot authenticate the
        #    retained envelope either.
        result_path.write_text(json.dumps(original), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable, match="signature"):
            _npu_outbox(root, signature_key=b"other-host-envelope-key-32-bytes!").begin(
                request, epoch=harness.factory.epoch,
            )


def test_npu_completed_evidence_requires_an_observed_clock_and_device_time(tmp_path):
    """@brief complete/ok 不能同时声明未观测时钟或零设备耗时。"""
    request = npu_request("npu-clock-1", source=SOURCE)
    evidence = {
        "schema_id": contracts.NPU_TIMING_EVIDENCE_SCHEMA_ID,
        "candidate_digest": request.candidate_digest,
        "device_group": request.npu.execution.device_group.to_dict(),
        "timing": {
            "source": "DEVICE_EVENT",
            "clock_domain": contracts.NPU_UNOBSERVED_CLOCK_DOMAIN,
            "synchronization_completed": True,
            "device_elapsed_ns": 1_000,
            "host_api_elapsed_ns": 2_000,
            "launches": 1,
        },
        "measurement_kind": "simulated",
        "publication_allowed": False,
        "complete": True,
    }
    with pytest.raises(contracts.ContractError, match="observed clock domain"):
        contracts.npu_timing_evidence_document(evidence, report_status="ok")

    zero_time = copy.deepcopy(evidence)
    zero_time["timing"].update({
        "clock_domain": "vendor-a-device-event-clock-v1", "device_elapsed_ns": 0,
    })
    with pytest.raises(contracts.ContractError, match="positive device elapsed"):
        contracts.npu_timing_evidence_document(zero_time, report_status="ok")

    # A diagnostic row may keep the unobserved clock, but it cannot be "complete".
    evidence["complete"] = False
    evidence["timing"].update({
        "synchronization_completed": False, "launches": 0, "device_elapsed_ns": 0,
    })
    contracts.npu_timing_evidence_document(evidence, report_status="timeout")


def test_npu_ok_report_candidate_time_must_come_from_the_device_event(tmp_path):
    """@brief ok 报告的 candidate_us 必须来自 device event，不能写成 host 时间。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = FakeVendorAController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-timing-bind-1", source=SOURCE)
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while snapshot.state != "completed" and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        assert snapshot.state == "completed"
        evidence = snapshot.result.evidence_json()
        assert evidence["complete"] is True
        report = snapshot.result.report_json()
        assert report["status"] == "ok"
        assert report["candidate_us"] == evidence["timing"]["device_elapsed_ns"] / 1000.0

        # A report whose candidate time is a host wall time is rejected by the
        # store, and the device time is not silently preferred instead.
        forged = dict(report)
        forged["candidate_us"] = 50_000.0
        with pytest.raises(JobError, match="device-event elapsed time"):
            engine.store.validate_worker_result(
                request=request, report=forged, evidence=evidence,
                provenance=snapshot.result.provenance,
            )
        # The host-side replay check applies the same rule.
        with pytest.raises(contracts.ContractError, match="device-event elapsed time"):
            contracts.npu_report_timing_check(evidence, forged)
    finally:
        engine.close(wait=5)


def _completed_npu_job(tmp_path, request_id: str):
    """Complete one NPU job through the fake gRPC loop and return its state dir."""

    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    state = root / "state"
    controller = FakeVendorAController(ledger_path)
    engine = npu_engine(
        state, ledger_path=ledger_path, controller=controller,
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    request = npu_request(request_id, source=SOURCE)
    npu_submit(engine, request)
    deadline = time.monotonic() + 10
    snapshot = npu_get(engine, request)
    while snapshot.state != "completed" and time.monotonic() < deadline:
        time.sleep(0.02)
        snapshot = npu_get(engine, request)
    assert snapshot.state == "completed" and snapshot.result is not None
    return engine, request, state, ledger_path


def _sql_update(state: Path, statement: str, parameters: tuple) -> None:
    connection = sqlite3.connect(state / "jobs.sqlite3")
    try:
        connection.execute(statement, parameters)
        connection.commit()
    finally:
        connection.close()


def _rewrite_json_column(state: Path, table: str, column: str, request_id: str, mutate) -> None:
    connection = sqlite3.connect(state / "jobs.sqlite3")
    try:
        row = connection.execute(
            f"SELECT {column} FROM {table} WHERE request_id=?", (request_id,),
        ).fetchone()
        document = json.loads(row[0])
        mutate(document)
        connection.execute(
            f"UPDATE {table} SET {column}=? WHERE request_id=?",
            (json.dumps(document, sort_keys=True), request_id),
        )
        connection.commit()
    finally:
        connection.close()


def _tamper_request_task(document):
    document["task"] = "forged-task"


def _tamper_request_execution(document):
    document["execution"]["launch_kind"] = "GRAPH"


def _tamper_request_dual_identity(document):
    document["candidate_sha256"] = document["candidate_digest"]["value"]


def _tamper_job_provenance_generation(document):
    document["device_group"]["resource_generation"] = "999"


def _tamper_job_provenance_launch(document):
    document["launch_kind"] = "GRAPH"


def _tamper_binding_generation(document):
    document["resource_generation"] = "999"


@pytest.mark.parametrize("mutate", [
    _tamper_request_task,
    _tamper_request_execution,
    _tamper_request_dual_identity,
    _tamper_job_provenance_generation,
    _tamper_job_provenance_launch,
])
def test_a_tampered_npu_durable_identity_is_never_served(tmp_path, mutate):
    """@brief 持久化 request/provenance 被篡改后，直接读取与重开都必须 fail closed。"""
    engine, request, state, ledger_path = _completed_npu_job(tmp_path, "npu-durable-tamper-1")
    try:
        if mutate is _tamper_request_task or mutate is _tamper_request_execution \
                or mutate is _tamper_request_dual_identity:
            _rewrite_json_column(state, "jobs", "request_json", request.request_id, mutate)
        else:
            _rewrite_json_column(state, "jobs", "provenance_json", request.request_id, mutate)

        with pytest.raises(StateCorrupt):
            npu_get(engine, request)
        with pytest.raises(StateCorrupt):
            engine.store._job_by_id(request.request_id)
    finally:
        engine.close(wait=5)

    with pytest.raises(StateCorrupt):
        npu_engine(
            state, ledger_path=ledger_path, controller=FakeVendorAController(ledger_path),
            signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
        )


def test_a_tampered_attempt_device_binding_is_never_served(tmp_path):
    """@brief attempt lease 的 device binding 必须与签名 result 的绑定一致。"""
    engine, request, state, ledger_path = _completed_npu_job(tmp_path, "npu-binding-tamper-1")
    try:
        _rewrite_json_column(state, "attempts", "device_binding_json", request.request_id,
                             _tamper_binding_generation)
        with pytest.raises(StateCorrupt):
            npu_get(engine, request)
        with pytest.raises(StateCorrupt):
            engine.store._job_by_id(request.request_id)
    finally:
        engine.close(wait=5)

    with pytest.raises(StateCorrupt):
        npu_engine(
            state, ledger_path=ledger_path, controller=FakeVendorAController(ledger_path),
            signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
        )


def test_a_stored_request_with_two_candidate_identities_is_rejected():
    """@brief from_stored 不得静默丢弃 legacy candidate_sha256 来接受混合身份。"""
    from tcenv.remote.job_types import ExternalEvaluationRequest, JobError

    request = npu_request("npu-stored-dual-1", source=SOURCE)
    document = {"request_id": request.request_id, **request.digest_fields()}
    document["candidate_sha256"] = request.candidate_digest["value"]
    with pytest.raises(JobError, match="also carries candidate_sha256"):
        ExternalEvaluationRequest.from_stored(document, request.source)
