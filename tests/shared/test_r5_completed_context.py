"""R5: the read-only ``load_completed_context`` used by a resume (D43, F-30/F-36/F-41).

CPU only.  The legacy dialect cases use a hand-built row + result envelope; the NPU cases use the
vendor-sim gRPC harness (a fake evaluator, never a real device).  Both assert the same contract:
a missing or pending row is an **unknown** context (``None``), and anything that does not add up —
row, result, receipt, evidence, HMAC, receipt index, or the caller's digest — is
``NeedsInspection``/``result_invalid``, never a silent unknown and never a rewrite.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from taic.artifacts import NeedsInspection
from taic.evaluation_outbox import EvaluationOutbox
from tcenv.evaluate import EvalReport
from tcenv.evaluation import EvaluationRequest, EvaluationResult, local_provenance, receipt_id_for, sign_result
from tcenv.remote import contracts

from ..support.job_engine import NPU_TARGET, NPU_TASK_FINGERPRINT, npu_request
from ..support.vendor_sim.harness import NPU_HOST_SIGNATURE_KEY, NpuGrpcHarness
from tests.support.v1_tasks import V1_TASK

KEY = b"host-result-signature-key-0123456789"


# ------------------------------------------------------------------------------ legacy (Metal/source) dialect


def _request(request_id: str = "ctx-legacy-1") -> EvaluationRequest:
    return EvaluationRequest(
        task=V1_TASK, dtype="fp16", size="small", source="kernel void demo() {}",
        mode="quick", seed=3, reps=2, run_id="run-1", session_id="session-1",
        task_fingerprint="a" * 64, cache_policy="disabled", request_id=request_id,
    )


def _result(request: EvaluationRequest) -> EvaluationResult:
    # A completed outbox row retains a whole schema-4 report (status ``ok`` requires its build
    # record), so the fixture is one too instead of an ``EvalReport`` with no build.
    from ..support.report_build import report_for_request

    report = report_for_request(
        {"task": request.task, "dtype": request.dtype, "size": request.size,
         "mode": request.mode, "seed": 3, "target_id": request.target_id or "apple-m4-max"},
        source=request.source.encode("utf-8"), build_kind="native",
    )
    result = EvaluationResult(
        report=report, provenance=local_provenance(request),
        evidence={"candidate_sha256": request.candidate_sha256},
        receipt={
            "receipt_id": receipt_id_for(request), "request_id": request.request_id,
            "request_digest": request.request_digest, "measurement_kind": "measured",
        },
    )
    return result


def _completed_legacy(tmp_path, *, signature_key: bytes | None = None):
    request = _request()
    root = tmp_path / "run"
    outbox = EvaluationOutbox(root, signature_key=signature_key)
    outbox.begin(request)
    result = _result(request)
    if signature_key is not None:
        result.signature = sign_result(result, signature_key)
    outbox.complete(request, result)
    return root, request, result


def test_a_completed_legacy_row_loads_with_the_source_bytes_it_was_measured_with(tmp_path):
    """The row/result/receipt/evidence chain is revalidated and the source is rehashed when given."""
    root, request, result = _completed_legacy(tmp_path)
    context = EvaluationOutbox(root).load_completed_context(
        request.request_id, request.request_digest, source_bytes=request.source.encode("utf-8"),
    )
    assert context is not None
    assert context.request_id == request.request_id
    assert context.request_digest == request.request_digest
    assert context.row["status"] == "completed"
    assert context.result.provenance.request_digest == request.request_digest
    assert context.source_hash == "matched"


def test_a_completed_row_without_source_bytes_is_unreviewed_not_passed(tmp_path):
    """Without the bytes the inline source hash cannot be rechecked, and the result says so."""
    root, request, _result_doc = _completed_legacy(tmp_path)
    context = EvaluationOutbox(root).load_completed_context(request.request_id, request.request_digest)
    assert context is not None
    assert context.source_hash == "unreviewed"


def test_the_source_bytes_must_match_the_row_identity(tmp_path):
    """Bytes that are not the measured source make the context damage, not a wiped check."""
    root, request, _result_doc = _completed_legacy(tmp_path)
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root).load_completed_context(
            request.request_id, request.request_digest, source_bytes=b"kernel void other() {}\n",
        )


@pytest.mark.parametrize("status", ["pending", "unavailable"])
def test_a_row_that_is_not_completed_is_an_unknown_context(tmp_path, status):
    """pending/unavailable rows are unknown; they are not damage and nothing is rewritten."""
    root = tmp_path / "run"
    request = _request()
    outbox = EvaluationOutbox(root)
    outbox.begin(request)
    if status == "unavailable":
        from tcenv.evaluation import EvaluationUnavailable

        outbox.unavailable(request, EvaluationUnavailable("transport_closed", "offline"))
    before = sorted(path.name for path in (root / "evaluation_outbox").iterdir())
    assert outbox.load_completed_context(request.request_id, request.request_digest) is None
    assert sorted(path.name for path in (root / "evaluation_outbox").iterdir()) == before


def test_a_missing_row_is_unknown_and_creates_nothing(tmp_path):
    """No row at all is unknown: loading a context never writes a row."""
    root = tmp_path / "run"
    outbox = EvaluationOutbox(root)
    assert outbox.load_completed_context("never-begun", "ab" * 32) is None
    assert list((root / "evaluation_outbox").iterdir()) == []


def test_a_row_epoch_that_differs_from_the_retained_result_is_damage(tmp_path):
    """评审复现：``epoch=""`` 下也要强制行与 retained result 的历史 epoch 一致（F-41）。

    ``epoch=""`` only means "do not compare with *today's* evaluator"; the row's own historical epoch
    is part of the context, and a result written under another epoch was never this row's result.
    """
    root, request, _result_doc = _completed_legacy(tmp_path, signature_key=KEY)
    row_path = EvaluationOutbox.row_path(root, request.request_id)
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["evaluator_epoch"] = "f" * 64
    row_path.write_text(json.dumps(row), encoding="utf-8")
    for source in (request.source.encode("utf-8"), None):
        with pytest.raises(NeedsInspection, match="result_invalid"):
            EvaluationOutbox(root, signature_key=KEY).load_completed_context(
                request.request_id, request.request_digest, source_bytes=source,
            )


def test_a_row_candidate_that_differs_from_the_measured_source_is_damage(tmp_path):
    """评审复现：generic 行持久化的 candidate_sha256 必须等于重算源码/receipt/evidence。

    The row was ignored before, so rewriting its candidate identity still produced a context the
    confirmation gate would trust.  With the bytes it is a direct mismatch; without them the row
    still has to agree with the receipt/evidence the result carries.
    """
    root, request, _result_doc = _completed_legacy(tmp_path, signature_key=KEY)
    row_path = EvaluationOutbox.row_path(root, request.request_id)
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["candidate_sha256"] = "f" * 64
    row_path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root, signature_key=KEY).load_completed_context(
            request.request_id, request.request_digest, source_bytes=request.source.encode("utf-8"),
        )
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root, signature_key=KEY).load_completed_context(
            request.request_id, request.request_digest,
        )


def test_a_row_whose_candidate_matches_still_loads(tmp_path):
    """Positive control: the untouched row still loads with and without the source bytes."""
    root, request, _result_doc = _completed_legacy(tmp_path, signature_key=KEY)
    for source in (request.source.encode("utf-8"), None):
        context = EvaluationOutbox(root, signature_key=KEY).load_completed_context(
            request.request_id, request.request_digest, source_bytes=source,
        )
        assert context is not None
        assert context.row["candidate_sha256"] == request.candidate_sha256


def test_a_retained_report_that_is_not_schema_4_is_damage(tmp_path):
    """评审复现：保留 result 里的报告改成 schema 3 时，只读加载必须把它当损坏。

    ``EvaluationResult.from_dict`` keeps ``report`` as the parsed dictionary, so without an explicit
    closure an unvalidated (schema-3) report was accepted as a completed context.  The envelope is
    unsigned here, so the schema check is the only gate — and the original text is not rewritten.
    """
    root, request, _result_doc = _completed_legacy(tmp_path)
    result_path = EvaluationOutbox.result_path(root, request.request_id)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["report"]["schema_version"] = 3
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    for source in (request.source.encode("utf-8"), None):
        with pytest.raises(NeedsInspection, match="result_invalid"):
            EvaluationOutbox(root).load_completed_context(
                request.request_id, request.request_digest, source_bytes=source,
            )


def _duplicate_report_key(root, request) -> bytes:
    """Write two legit ``schema_version`` keys into the retained report; return the file bytes."""
    result_path = EvaluationOutbox.result_path(root, request.request_id)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["report"]["schema_version"] = 4
    text = json.dumps(payload).replace('"report": {', '"report": {"schema_version": 4, ', 1)
    result_path.write_text(text, encoding="utf-8")
    return result_path.read_bytes()


def test_a_retained_report_with_a_duplicate_key_is_damage(tmp_path):
    """评审复现：retained report 里两个合法 ``schema_version`` 时，只读加载必须拒绝且不改写文件。

    ``json.loads`` keeps the last value and forgets the repetition, so the schema closure saw an
    intact document and accepted it.  The envelope is parsed with ``loads_strict`` now (§6.4: every
    report read refuses duplicate keys before ``EvalReport.from_json``), and the reader never writes.
    """
    root, request, _result_doc = _completed_legacy(tmp_path)
    before = _duplicate_report_key(root, request)
    for source in (request.source.encode("utf-8"), None):
        with pytest.raises(NeedsInspection, match="result_invalid"):
            EvaluationOutbox(root).load_completed_context(
                request.request_id, request.request_digest, source_bytes=source,
            )
    assert EvaluationOutbox.result_path(root, request.request_id).read_bytes() == before


def test_a_completed_replay_with_a_duplicate_report_key_is_refused(tmp_path):
    """``begin()`` 的 completed 重放走同一个读盘函数：重复键同样拒绝，文件不变。"""
    from tcenv.evaluation import EvaluationUnavailable

    root, request, _result_doc = _completed_legacy(tmp_path)
    before = _duplicate_report_key(root, request)
    with pytest.raises(EvaluationUnavailable) as caught:
        EvaluationOutbox(root).begin(request)
    assert caught.value.code == "outbox_corrupt"
    assert "repeats the key" in str(caught.value)
    assert EvaluationOutbox.result_path(root, request.request_id).read_bytes() == before
    assert json.loads(EvaluationOutbox.row_path(root, request.request_id).read_text(encoding="utf-8"))["status"] == "completed"


def test_a_retained_report_with_a_missing_required_field_is_damage(tmp_path):
    """The closure is the whole schema: a report missing a required key is refused too."""
    root, request, _result_doc = _completed_legacy(tmp_path)
    result_path = EvaluationOutbox.result_path(root, request.request_id)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["report"].pop("evaluation_fingerprint", None)
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root).load_completed_context(request.request_id, request.request_digest)


@pytest.mark.parametrize("value", ["absent", None, ""])
def test_a_completed_context_without_a_historical_epoch_is_damage(tmp_path, value):
    """评审复现：两侧同时缺失/``None``/空串不再是「相等」，没有可比较的 epoch 就不是已校验上下文。"""
    root, request, _result_doc = _completed_legacy(tmp_path)
    row_path = EvaluationOutbox.row_path(root, request.request_id)
    result_path = EvaluationOutbox.result_path(root, request.request_id)
    row = json.loads(row_path.read_text(encoding="utf-8"))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    for target, key in ((row, "evaluator_epoch"), (payload["provenance"], "epoch")):
        if value == "absent":
            target.pop(key, None)
        else:
            target[key] = value
    row_path.write_text(json.dumps(row), encoding="utf-8")
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root).load_completed_context(request.request_id, request.request_digest)


def test_a_completed_context_records_a_comparable_historical_epoch(tmp_path):
    """Positive control: the untouched context states the same non-empty epoch on both sides."""
    root, request, result = _completed_legacy(tmp_path)
    context = EvaluationOutbox(root).load_completed_context(request.request_id, request.request_digest)
    assert context is not None
    epoch = context.row["evaluator_epoch"]
    assert isinstance(epoch, str) and epoch
    assert epoch == context.result.provenance.epoch == result.provenance.epoch


def test_a_caller_digest_that_differs_from_the_row_is_damage(tmp_path):
    """The projection's digest and the row's digest must name one request."""
    root, request, _result_doc = _completed_legacy(tmp_path)
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root).load_completed_context(request.request_id, "cd" * 32)


def test_a_row_recorded_for_another_request_id_is_damage(tmp_path):
    """A row whose own ``request_id`` disagrees with its file name was rewritten."""
    root, request, _result_doc = _completed_legacy(tmp_path)
    row_path = EvaluationOutbox.row_path(root, request.request_id)
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["request_id"] = "someone-else"
    row_path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root).load_completed_context(request.request_id, request.request_digest)


def test_a_missing_or_escaping_result_path_is_damage(tmp_path):
    """A completed row whose result is gone (or points outside the outbox) is not "unknown"."""
    root, request, _result_doc = _completed_legacy(tmp_path)
    row_path = EvaluationOutbox.row_path(root, request.request_id)
    row = json.loads(row_path.read_text(encoding="utf-8"))
    EvaluationOutbox.result_path(root, request.request_id).unlink()
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root).load_completed_context(request.request_id, request.request_digest)
    row["report_path"] = "../../escaped.result.json"
    row_path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root).load_completed_context(request.request_id, request.request_digest)


def test_a_tampered_result_envelope_fails_the_host_signature(tmp_path):
    """The HMAC covers report/provenance/evidence/receipt: any edit fails the reload."""
    root, request, _result_doc = _completed_legacy(tmp_path, signature_key=KEY)
    result_path = EvaluationOutbox.result_path(root, request.request_id)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["report"]["speedup"] = 9.99
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root, signature_key=KEY).load_completed_context(
            request.request_id, request.request_digest, source_bytes=request.source.encode("utf-8"),
        )


def test_a_receipt_id_reused_for_another_digest_is_damage(tmp_path):
    """The receipt index is part of the evidence: a reused id is not a fresh completed context."""
    root, request, result = _completed_legacy(tmp_path)
    receipt_id = result.receipt["receipt_id"]
    (root / "evaluation_outbox" / "receipts.json").write_text(
        json.dumps({receipt_id: "ef" * 32}), encoding="utf-8",
    )
    with pytest.raises(NeedsInspection, match="result_invalid"):
        EvaluationOutbox(root).load_completed_context(
            request.request_id, request.request_digest, source_bytes=request.source.encode("utf-8"),
        )


def test_an_npu_retained_report_that_is_not_schema_4_is_damage(tmp_path):
    """The NPU dialect gets the same schema-4 closure (re-signed, so only the schema can refuse it)."""
    from taic.evaluation_outbox import bind_wire_digest
    from tcenv.evaluation import sign_result

    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        external = npu_request("ctx-npu-schema", source=SOURCE)
        host = _legacy_request("ctx-npu-schema", SOURCE)
        request = bind_wire_digest(
            replace(host, contract_request_digest=external.request_digest),
            harness.factory._for_request(host), npu_contract=True,
        )
        outbox = _npu_outbox(root)
        outbox.begin(request, epoch=harness.factory.epoch)
        outbox.complete(request, harness.factory.formal(request))

        assert _npu_outbox(root).load_completed_context(request.request_id, request.request_digest) is not None

        result_path = EvaluationOutbox.result_path(root, request.request_id)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["report"]["schema_version"] = 3
        tampered = EvaluationResult.from_dict(payload)
        tampered.signature = sign_result(tampered, NPU_HOST_SIGNATURE_KEY)
        result_path.write_text(json.dumps(tampered.to_dict()), encoding="utf-8")
        with pytest.raises(NeedsInspection, match="result_invalid"):
            _npu_outbox(root).load_completed_context(request.request_id, request.request_digest)


# --------------------------------------------------------------------------------- NPU (inline source) dialect


def _npu_outbox(root) -> EvaluationOutbox:
    return EvaluationOutbox(
        root, signature_key=NPU_HOST_SIGNATURE_KEY, transport="grpc",
        api_contract_id=contracts.NPU_SOURCE_API_CONTRACT_ID,
        evidence_contract_id=contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
    )


def _legacy_request(request_id: str, source: bytes) -> EvaluationRequest:
    return EvaluationRequest(
        kind="formal_eval", request_id=request_id, run_id="npu-run-1",
        session_id="npu-session-1", task=V1_TASK, dtype="fp16", size="small",
        target_id=NPU_TARGET, task_fingerprint=NPU_TASK_FINGERPRINT,
        source=source.decode("utf-8"), mode="quick", seed=3, reps=2,
        queue_timeout_s=3.0, candidate_timeout_s=3.0,
    )


SOURCE = b"npu-kernel relu_kernel\n"


def test_an_npu_completed_row_reloads_its_receipt_and_evidence_binding(tmp_path):
    """The NPU row's CandidateDigest/device group/receipt/evidence must all still agree."""
    from taic.evaluation_outbox import bind_wire_digest

    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        external = npu_request("ctx-npu-1", source=SOURCE)
        host = _legacy_request("ctx-npu-1", SOURCE)
        request = bind_wire_digest(replace(host, contract_request_digest=external.request_digest), harness.factory._for_request(host), npu_contract=True)
        outbox = _npu_outbox(root)
        outbox.begin(request, epoch=harness.factory.epoch)
        outbox.complete(request, harness.factory.formal(request))

        resumed = _npu_outbox(root)
        context = resumed.load_completed_context(
            request.request_id, request.request_digest, source_bytes=SOURCE,
        )
        assert context is not None
        assert context.source_hash == "matched"
        assert context.row["candidate_digest"] == external.candidate_digest
        assert context.result.receipt["candidate_digest"] == external.candidate_digest
        assert context.result.evidence["device_group"] == context.row["device_group"]
        # Historical epoch: loading it is not a comparison against today's evaluator epoch.
        assert context.result.provenance.epoch == harness.factory.epoch

        unreviewed = resumed.load_completed_context(request.request_id, request.request_digest)
        assert unreviewed is not None and unreviewed.source_hash == "unreviewed"

        with pytest.raises(NeedsInspection, match="result_invalid"):
            resumed.load_completed_context(request.request_id, request.request_digest, source_bytes=b"other\n")


def test_an_npu_row_whose_candidate_digest_was_rewritten_is_damage(tmp_path):
    """A rewritten CandidateDigest is outbox damage, not an unknown context."""
    from taic.evaluation_outbox import bind_wire_digest

    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        external = npu_request("ctx-npu-2", source=SOURCE)
        host = _legacy_request("ctx-npu-2", SOURCE)
        request = bind_wire_digest(replace(host, contract_request_digest=external.request_digest), harness.factory._for_request(host), npu_contract=True)
        outbox = _npu_outbox(root)
        outbox.begin(request, epoch=harness.factory.epoch)
        outbox.complete(request, harness.factory.formal(request))
        row_path = EvaluationOutbox.row_path(root, request.request_id)
        row = json.loads(row_path.read_text(encoding="utf-8"))
        row["candidate_digest"]["value"] = "9" * 64
        row_path.write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(NeedsInspection, match="result_invalid"):
            _npu_outbox(root).load_completed_context(request.request_id, request.request_digest)
