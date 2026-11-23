"""The production remote worker entry point uses a backend object, not a bare function."""

from __future__ import annotations

import hashlib
import io
from types import SimpleNamespace

from tcenv.remote.protocol import EvaluationRequest, encode_frame, read_frame
from tcenv.remote import worker


def _request() -> EvaluationRequest:
    source = "kernel void demo() {}"
    return EvaluationRequest(
        kind="formal_eval", request_id="worker-entry-1", run_id="run-1", session_id="session-1",
        task="relu", dtype="fp16", size="small", target_id="apple-m4-max",
        task_fingerprint="a" * 64, candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source=source, mode="quick", seed=3, reps=1, queue_timeout_s=1.0, candidate_timeout_s=1.0,
        cache_policy="disabled",
    )


def test_worker_main_passes_a_production_backend_object(monkeypatch):
    """@brief main 不再把 evaluate_request 函数当 backend.evaluate 调用。"""
    calls = []

    class FakeBackend:
        def evaluate(self, request):
            calls.append(request)
            return {"report": {"status": "ok"}, "evidence": {}, "receipt": {}}

    monkeypatch.setattr("tcenv.remote.production.ProductionBackend", FakeBackend)
    request = _request()
    output = io.BytesIO()
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(encode_frame("request", request.request_id, request.to_dict()))))
    monkeypatch.setattr(worker.sys, "stdout", SimpleNamespace(buffer=output))
    assert worker.main() == 0
    output.seek(0)
    frame = read_frame(output)
    assert frame.frame_type == "result"
    assert frame.payload["report"]["status"] == "ok"
    assert calls == [request]
