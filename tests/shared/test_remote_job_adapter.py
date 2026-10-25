"""Legacy-to-v1 request adaptation and transport-neutral evidence checks."""

from __future__ import annotations

import hashlib

import pytest

from tcenv.evaluation import EvaluationRequest, request_digest as legacy_request_digest
from tcenv.remote.job_types import ExternalEvaluationRequest, JobError


def test_legacy_adapter_normalizes_ms_once_and_preserves_exact_source_and_legacy_digest():
    source = "kernel void demo() {}\r\n// trailing space \r\n"
    legacy = EvaluationRequest(
        kind="formal_eval", request_id="adapter-request", run_id="run-1",
        session_id="session-1", task="relu", dtype="fp16", size="small",
        target_id="sim-target", task_fingerprint="a" * 64, source=source,
        candidate_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        mode="quick", seed=2**63 + 1, reps=15,
        queue_timeout_s=1.2344, candidate_timeout_s=0.0001,
        cache_policy="disabled",
    )
    old_digest = legacy.request_digest
    external = ExternalEvaluationRequest.from_legacy(legacy)
    assert external.source == source.encode("utf-8")
    assert external.queue_timeout_ms == 1235
    assert external.candidate_timeout_ms == 1
    assert external.seed == str(2**63 + 1)
    assert external.request_digest != old_digest
    assert legacy_request_digest(legacy) == old_digest

    round_trip = external.to_legacy()
    assert round_trip.source == source
    assert round_trip.source_bytes == external.source
    assert round_trip.candidate_sha256 == legacy.candidate_sha256
    assert round_trip.queue_timeout_s == 1.235
    assert round_trip.candidate_timeout_s == 0.001
    # Normalization belongs to the new request; it never mutates or rewrites
    # the original legacy object's digest behavior.
    assert legacy.request_digest == old_digest


def test_external_adapter_rejects_missing_task_identity_and_unsupported_seed():
    legacy = EvaluationRequest(
        kind="formal_eval", request_id="adapter-invalid", run_id="run-1",
        session_id="session-1", task="relu", dtype="fp16", size="small",
        target_id="sim-target", task_fingerprint="", source="kernel void x() {}",
        mode="quick", seed=2**64,
    )
    with pytest.raises(JobError, match="task_fingerprint"):
        ExternalEvaluationRequest.from_legacy(legacy)

    legacy = EvaluationRequest(
        kind="formal_eval", request_id="adapter-seed", run_id="run-1",
        session_id="session-1", task="relu", dtype="fp16", size="small",
        target_id="sim-target", task_fingerprint="a" * 64,
        source="kernel void x() {}", mode="quick", seed=2**64,
    )
    with pytest.raises(JobError, match="seed"):
        ExternalEvaluationRequest.from_legacy(legacy)
