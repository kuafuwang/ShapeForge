"""Durable evaluation outbox tests; no device or model is used."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from taic.evaluation_outbox import EvaluationOutbox
from tcenv.evaluation import EvaluationRequest, EvaluationResult, EvaluationUnavailable, local_provenance, receipt_id_for, sign_result
from tcenv.evaluate import EvalReport


def _request() -> EvaluationRequest:
    return EvaluationRequest(
        task="relu", dtype="fp16", size="small", source="kernel void demo() {}",
        mode="quick", seed=3, reps=2, run_id="run-1", session_id="session-1",
        task_fingerprint="a" * 64, cache_policy="disabled",
    )


def _result(request: EvaluationRequest) -> EvaluationResult:
    # A completed outbox row retains a whole schema-4 report (status ``ok`` requires its build
    # record); the fixture is one so the read boundary can validate it.
    from ..support.report_build import report_for_request

    report = report_for_request(
        {"task": request.task, "dtype": request.dtype, "size": request.size,
         "mode": request.mode, "seed": 3, "target_id": request.target_id or "apple-m4-max"},
        source=request.source.encode("utf-8"), build_kind="native",
    )
    return EvaluationResult(report=report, provenance=local_provenance(request), evidence={"candidate_sha256": request.candidate_sha256}, receipt={"receipt_id": receipt_id_for(request), "request_id": request.request_id, "request_digest": request.request_digest, "measurement_kind": "measured"})


def test_outbox_persists_pending_then_replays_completed_result(tmp_path):
    """@brief 请求先写 pending，完成后同 request_id 直接读回结果，不重复调用执行器。"""
    request = _request()
    outbox = EvaluationOutbox(tmp_path)
    assert outbox.begin(request) is None
    outbox.complete(request, _result(request))
    replayed = outbox.begin(request)
    assert replayed is not None
    assert replayed.provenance.request_digest == request.request_digest
    assert replayed.evidence["candidate_sha256"] == request.candidate_sha256
    rows = list((tmp_path / "evaluation_outbox").glob("*.json"))
    assert len(rows) == 3  # state + result envelope + receipt index
    state = json.loads(next(path for path in rows if not path.name.endswith(".result.json") and path.name != "receipts.json").read_text())
    assert state["status"] == "completed" and state["report_path"]


def test_outbox_records_unavailable_without_a_report(tmp_path):
    """@brief 基础设施失败保留错误，不生成候选报告。"""
    request = _request()
    outbox = EvaluationOutbox(tmp_path)
    outbox.begin(request)
    outbox.unavailable(request, EvaluationUnavailable("transport_closed", "offline", retryable=True))
    rows = outbox.pending()
    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["error"]["code"] == "transport_closed"
    assert rows[0]["report_path"] == ""


def test_outbox_rejects_request_id_reuse_with_different_body(tmp_path):
    """@brief 同 request_id 不同 request digest 必须 fail closed。"""
    request = _request()
    outbox = EvaluationOutbox(tmp_path)
    outbox.begin(request)
    changed = EvaluationRequest(
        task="relu", dtype="fp16", size="small", source="kernel void other() {}",
        mode="quick", seed=3, reps=2, run_id="run-1", session_id="session-1",
        task_fingerprint="a" * 64, cache_policy="disabled", request_id=request.request_id,
    )
    with pytest.raises(EvaluationUnavailable, match="different request"):
        outbox.begin(changed)


def test_outbox_refuses_to_replay_a_result_from_another_epoch(tmp_path):
    """@brief 恢复时 evaluator epoch 变化必须停止，不能混用旧测量。"""
    request = _request()
    outbox = EvaluationOutbox(tmp_path)
    result = _result(request)
    outbox.begin(request, epoch=result.provenance.epoch)
    outbox.complete(request, result)
    with pytest.raises(EvaluationUnavailable, match="epoch"):
        outbox.begin(request, epoch="f" * 64)


def test_outbox_refuses_a_completed_replay_signed_by_a_revoked_key(tmp_path):
    """@brief 撤销外部 HMAC key 后，completed outbox 只能 fail closed，不能改写结果或继续发布。"""
    request = _request()
    outbox = EvaluationOutbox(tmp_path)
    outbox.begin(request)
    result = _result(request)
    result.receipt["signature_key_id"] = "site/key-1"
    outbox.complete(request, result)
    revoked = EvaluationOutbox(tmp_path, external_trusted_key_ids=frozenset({"site/key-2"}))
    with pytest.raises(EvaluationUnavailable, match="no longer trusts") as excinfo:
        revoked.begin(request)
    # This is the orchestrator's failure path: the completed row stays intact
    # and only records the revoked-key diagnostic.
    revoked.unavailable(request, excinfo.value)
    state = json.loads(next(path for path in (tmp_path / "evaluation_outbox").glob("*.json") if not path.name.endswith(".result.json") and path.name != "receipts.json").read_text())
    assert state["status"] == "completed"
    assert state["report_path"]
    assert state["error"]["code"] == "signature_key_revoked"
    rotated = EvaluationOutbox(tmp_path, external_trusted_key_ids=frozenset({"site/key-1", "site/key-2"}))
    assert rotated.begin(request) is not None


def test_outbox_rejects_reusing_a_receipt_for_another_digest(tmp_path):
    """@brief host receipt 索引不允许同 receipt 绑定另一份请求体。"""
    first = _request()
    second = EvaluationRequest(
        task=first.task, dtype=first.dtype, size=first.size, source=first.source,
        mode="full", seed=first.seed, reps=first.reps,
        run_id=first.run_id, session_id=first.session_id,
        task_fingerprint=first.task_fingerprint, cache_policy="disabled",
    )
    outbox = EvaluationOutbox(tmp_path)
    outbox.begin(first)
    result = _result(first)
    outbox.complete(first, result)
    outbox.begin(second)
    replay = _result(second)
    replay.receipt = dict(replay.receipt, receipt_id=result.receipt["receipt_id"])
    with pytest.raises(EvaluationUnavailable, match="receipt"):
        outbox.complete(second, replay)


def test_outbox_verifies_a_configured_result_signature(tmp_path):
    """@brief outbox 也必须验证配置密钥对应的签名，不能接受错签结果。"""
    request = _request()
    key = b"outbox-signing-key"
    result = _result(request)
    result.signature = sign_result(result, key)
    outbox = EvaluationOutbox(tmp_path, signature_key=key)
    outbox.begin(request)
    outbox.complete(request, result)
    assert outbox.begin(request) is not None
    with pytest.raises(EvaluationUnavailable, match="signature"):
        EvaluationOutbox(tmp_path, signature_key=b"wrong-key").begin(request)
