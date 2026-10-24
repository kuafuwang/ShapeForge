"""Remote evaluator framing and request validation."""

from __future__ import annotations

import hashlib

import pytest

from tcenv.remote.protocol import (
    MAX_FRAME_PAYLOAD,
    EvaluationRequest,
    FrameDecoder,
    ProtocolError,
    encode_frame,
    read_frame,
)


TASK_FP = "a" * 64
SITE_FP = "b" * 64


def _request(**overrides):
    source = overrides.pop("source", "kernel void demo() {}")
    values = {
        "kind": "formal_eval",
        "request_id": "req-1",
        "run_id": "run-1",
        "session_id": "session-1",
        "task": "relu",
        "dtype": "fp16",
        "size": "small",
        "target_id": "sim-npu-test",
        "task_fingerprint": TASK_FP,
        "candidate_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source": source,
        "mode": "quick",
        "seed": 7,
        "reps": 3,
        "queue_timeout_s": 2.0,
        "candidate_timeout_s": 2.0,
        "cases": ("domain",),
    }
    values.update(overrides)
    return EvaluationRequest(**values)


def test_frames_survive_fragmentation_coalescing_and_exact_bytes():
    first = encode_frame("hello", "nonce-1", {"hello": "world", "value": 1})
    second = encode_frame("request", "req-1", {"source": "héllo\r\n", "n": 2})
    decoder = FrameDecoder()
    assert decoder.feed(first[:2]) == []
    assert decoder.feed(first[2:] + second[:5])[0].payload == {"hello": "world", "value": 1}
    frames = decoder.feed(second[5:])
    assert len(frames) == 1
    assert frames[0].payload == {"source": "héllo\r\n", "n": 2}
    assert decoder._buffer == bytearray()


def test_half_frame_is_rejected():
    raw = encode_frame("hello", "nonce-1", {"x": 1})
    decoder = FrameDecoder()
    assert decoder.feed(raw[:-1]) == []
    with pytest.raises(ProtocolError, match="partial"):
        decoder.finish()


def test_bad_payload_digest_is_rejected():
    raw = bytearray(encode_frame("hello", "nonce-1", {"x": 1}))
    raw[-1] ^= 1
    decoder = FrameDecoder()
    with pytest.raises(ProtocolError, match="digest"):
        decoder.feed(bytes(raw))


def test_oversize_payload_is_refused_before_encoding():
    with pytest.raises(ProtocolError, match="limit"):
        encode_frame("hello", "nonce-1", {"x": "x" * (MAX_FRAME_PAYLOAD + 1)})


def test_request_digest_is_stable_and_source_bytes_are_not_normalized():
    source = "line1\r\nline2  \u00e9"
    req = _request(source=source)
    assert req.candidate_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert req.request_digest == req.request_digest
    assert EvaluationRequest.from_dict(req.to_dict()).request_digest == req.request_digest
    assert req.source_bytes == source.encode("utf-8")


def test_candidate_digest_must_match_actual_source_bytes():
    with pytest.raises(ProtocolError, match="candidate_sha256"):
        _request(candidate_sha256="c" * 64)


def test_unknown_request_fields_and_invalid_modes_are_refused():
    req = _request()
    document = req.to_dict()
    document["command"] = "rm -rf /"
    with pytest.raises(ProtocolError, match="unknown"):
        EvaluationRequest.from_dict(document)
    with pytest.raises(ProtocolError, match="mode"):
        _request(kind="selfcheck", mode="full")


def test_cases_and_request_identity_are_bounded():
    with pytest.raises(ProtocolError, match="cases"):
        _request(cases=[""])
    with pytest.raises(ProtocolError, match="request_id"):
        _request(request_id="bad id with spaces")
    with pytest.raises(ProtocolError, match="reps"):
        _request(reps=0)


def test_read_frame_reads_one_exact_frame_from_stream():
    from io import BytesIO

    raw = encode_frame("hello", "nonce-1", {"ok": True})
    stream = BytesIO(raw + encode_frame("cancel", "req-2", {"request_id": "req-2"}))
    first = read_frame(stream)
    second = read_frame(stream)
    assert (first.frame_type, first.request_id, first.payload["ok"]) == ("hello", "nonce-1", True)
    assert (second.frame_type, second.request_id) == ("cancel", "req-2")
