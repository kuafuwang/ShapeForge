"""P1c: the BuildProvenance is an independent outbox/resume identity.

The build document is *not* folded into ``artifact_set_digest``; it is a field of
the canonical request digest and its own row field in the durable outbox.  These
tests bind a real prebuilt request through the real gRPC evaluator and then
exercise the outbox/resume path, so a swapped build identity cannot be resumed
behind an otherwise equal candidate.
"""

from __future__ import annotations

import json

import pytest

from taic.evaluation_outbox import EvaluationOutbox, bind_wire_digest
from tcenv.evaluation import EvaluationRequest, EvaluationUnavailable
from tcenv.remote import contracts
from tests.support.job_engine import NPU_TASK_FINGERPRINT
from tests.support.vendor_sim.harness import (
    NPU_HOST_SIGNATURE_KEY,
    ModuleBuildGrpcHarness,
)

SOURCE = b"npu-kernel relu_kernel\n  tx_thread_id\n"


def _host_request(harness, request_id: str) -> EvaluationRequest:
    return EvaluationRequest(
        kind="formal_eval", request_id=request_id, run_id="npu-run-1",
        session_id="npu-session-1", task="relu", dtype="fp16", size="small",
        target_id=harness.engine.identity.target_id,
        task_fingerprint=NPU_TASK_FINGERPRINT, source="",
        mode="quick", seed=3, reps=2, queue_timeout_s=10.0, candidate_timeout_s=10.0,
    )


def _outbox(tmp_path) -> EvaluationOutbox:
    return EvaluationOutbox(
        tmp_path / "run", signature_key=NPU_HOST_SIGNATURE_KEY, transport="grpc",
        api_contract_id=contracts.NPU_BUILD_API_CONTRACT_ID,
        evidence_contract_id=contracts.NPU_BUILD_EVIDENCE_CONTRACT_ID,
    )


def _state_rows(tmp_path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "run" / "evaluation_outbox").glob("*.json")
        if not path.name.endswith(".result.json") and path.name != "receipts.json"
    ]


def _prepared(harness, request_id: str):
    outcome = harness.host_build(SOURCE)
    harness.upload_module(outcome)
    harness.set_prebuilt(outcome, outcome.bundle, SOURCE)
    host = _host_request(harness, request_id)
    evaluator = harness.factory._for_request(host)
    return outcome, host, evaluator, bind_wire_digest(host, evaluator, npu_contract=True)


def test_the_build_provenance_is_its_own_row_field_and_resume_identity(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome, _host, _evaluator, request = _prepared(harness, "build-outbox-1")
        provenance = harness.build.to_dict()
        assert request.wire_build_provenance == provenance
        assert provenance["mode"] == "prebuilt"
        assert provenance["provenance"]["build_host"] == "control_host"
        assert request.wire_candidate_digest == outcome.bundle.artifact_set_digest
        # The provenance is independent of the artifact set digest: it names the
        # builder/toolchain/profile and the source digest.
        assert "provenance" not in json.dumps(outcome.bundle.to_dict())
        assert request.wire_build_provenance["provenance"]["artifact_set_digest"] == (
            request.wire_candidate_digest
        )

        outbox = _outbox(tmp_path)
        assert outbox.begin(request) is None
        rows = _state_rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["status"] == "pending"
        assert rows[0]["build_provenance"] == provenance
        assert rows[0]["candidate_digest"]["value"] == outcome.bundle.artifact_set_digest

        # A resume that claims another build identity under the same candidate is
        # a conflict, not a silent reuse of the stored provenance.
        from dataclasses import replace

        swapped = replace(
            request,
            wire_build_provenance={
                **provenance,
                "provenance": {**provenance["provenance"], "builder_version": "9.9.9"},
            },
        )
        with pytest.raises(EvaluationUnavailable, match="BuildProvenance"):
            outbox.begin(swapped)
        # The row itself is untouched.
        assert _state_rows(tmp_path) == rows


def test_a_build_row_cannot_be_created_without_a_build_provenance(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        _outcome, _host, _evaluator, request = _prepared(harness, "build-outbox-2")
        from dataclasses import replace

        with pytest.raises(EvaluationUnavailable, match="BuildProvenance"):
            _outbox(tmp_path).begin(replace(request, wire_build_provenance=None))
        assert _state_rows(tmp_path) == []


def test_the_request_digest_changes_when_only_the_build_identity_changes(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome, host, evaluator, first = _prepared(harness, "build-outbox-3")
        # Same artifact bytes, same candidate; another registered builder
        # version.  The canonical request digest must change.
        from tests.support.vendor_sim import module_builder as mb

        harness.build = mb.request_build(
            mb.control_host_profile(builder_version="1.4.1"), source=SOURCE,
            mode="prebuilt", artifact_set_digest=outcome.bundle.artifact_set_digest,
        )
        second = bind_wire_digest(host, evaluator, npu_contract=True)
        assert first.wire_candidate_digest == second.wire_candidate_digest
        assert first.wire_build_provenance != second.wire_build_provenance
        assert first.request_digest != second.request_digest
