"""Wire protocol for the remote evaluator service.

The transport is a length-prefixed frame stream.  A frame has a four-byte
big-endian JSON-header length, a canonical JSON header, and an opaque JSON
payload.  The header binds the frame type, request id, payload length and
payload digest.  This is intentionally transport-neutral: stdin/stdout,
a socket or an SSH forced command can carry the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO

SCHEMA_VERSION = 1
PROTOCOL_VERSION = "1"
MAX_HEADER_BYTES = 65_536
MAX_FRAME_PAYLOAD = 2_000_000
MAX_SOURCE_BYTES = 1_000_000
MAX_TIMEOUT_S = 3_600.0
MAX_REPS = 10_000

_HEADER_KEYS = frozenset(
    {"schema_version", "frame_type", "request_id", "payload_length", "payload_sha256"}
)
_FRAME_TYPES = frozenset({"hello", "request", "result", "error", "cancel", "progress"})
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CACHE_POLICIES = frozenset({"reuse", "refresh", "disabled"})
_KINDS = frozenset({"formal_eval", "selfcheck"})


from tcenv.evaluation import EvaluationError as ProtocolError


def canonical_json(document: Any) -> bytes:
    """Canonical bytes used for digests and frame headers."""
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Frame:
    """A decoded frame; ``payload`` is always a JSON object."""

    frame_type: str
    request_id: str
    payload: dict[str, Any]
    payload_length: int
    payload_sha256: str


def encode_frame(
    frame_type: str, request_id: str, payload: dict[str, Any], *,
    max_payload: int = MAX_FRAME_PAYLOAD,
) -> bytes:
    """Encode one frame, rejecting oversize headers and payloads.

    The default is the frozen legacy limit.  A newer private worker protocol
    may explicitly choose a different bounded limit without changing legacy
    wire behavior.
    """
    if frame_type not in _FRAME_TYPES:
        raise ProtocolError(f"unknown frame type {frame_type!r}")
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise ProtocolError("request_id must be a bounded opaque identifier")
    try:
        raw = canonical_json(payload)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"frame payload is not canonical JSON: {exc}") from exc
    if isinstance(max_payload, bool) or not isinstance(max_payload, int) or not 0 < max_payload <= 64_000_000:
        raise ProtocolError("frame payload limit is invalid")
    if len(raw) > max_payload:
        raise ProtocolError("frame payload exceeds the limit")
    header = {
        "schema_version": SCHEMA_VERSION,
        "frame_type": frame_type,
        "request_id": request_id,
        "payload_length": len(raw),
        "payload_sha256": sha256_hex(raw),
    }
    encoded_header = canonical_json(header)
    if len(encoded_header) > MAX_HEADER_BYTES:
        raise ProtocolError("frame header exceeds the limit")
    return struct.pack(">I", len(encoded_header)) + encoded_header + raw


def _header(bytes_: bytes, *, max_payload: int = MAX_FRAME_PAYLOAD) -> dict[str, Any]:
    try:
        header = json.loads(bytes_.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("frame header is not UTF-8 JSON") from exc
    if not isinstance(header, dict) or set(header) != _HEADER_KEYS:
        raise ProtocolError("frame header has unexpected fields")
    if header.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("unsupported frame schema_version")
    if header.get("frame_type") not in _FRAME_TYPES:
        raise ProtocolError("unknown frame type")
    if not isinstance(header.get("request_id"), str) or not _REQUEST_ID.fullmatch(header["request_id"]):
        raise ProtocolError("invalid frame request_id")
    length = header.get("payload_length")
    if isinstance(length, bool) or not isinstance(length, int) or not 0 <= length <= max_payload:
        raise ProtocolError("invalid frame payload_length")
    digest = header.get("payload_sha256")
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise ProtocolError("invalid frame payload_sha256")
    return header


def _payload(raw: bytes, header: dict[str, Any]) -> dict[str, Any]:
    if len(raw) != header["payload_length"]:
        raise ProtocolError("frame payload length does not match the header")
    if sha256_hex(raw) != header["payload_sha256"]:
        raise ProtocolError("frame payload digest does not match the header")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("frame payload is not UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("frame payload must be an object")
    return payload


class FrameDecoder:
    """Incremental decoder for a stream that may fragment or coalesce frames."""

    def __init__(self, *, max_header: int = MAX_HEADER_BYTES, max_payload: int = MAX_FRAME_PAYLOAD) -> None:
        self._buffer = bytearray()
        self._max_header = max_header
        self._max_payload = max_payload

    def feed(self, data: bytes) -> list[Frame]:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ProtocolError("frame input must be bytes")
        self._buffer.extend(data)
        frames: list[Frame] = []
        while len(self._buffer) >= 4:
            header_len = struct.unpack(">I", self._buffer[:4])[0]
            if header_len <= 0 or header_len > self._max_header:
                raise ProtocolError("frame header length is outside the limit")
            if len(self._buffer) < 4 + header_len:
                break
            header_raw = bytes(self._buffer[4 : 4 + header_len])
            header = _header(header_raw, max_payload=self._max_payload)
            payload_len = int(header["payload_length"])
            if payload_len > self._max_payload:
                raise ProtocolError("frame payload length is outside the limit")
            end = 4 + header_len + payload_len
            if len(self._buffer) < end:
                break
            raw = bytes(self._buffer[4 + header_len : end])
            payload = _payload(raw, header)
            frames.append(
                Frame(
                    frame_type=str(header["frame_type"]),
                    request_id=str(header["request_id"]),
                    payload=payload,
                    payload_length=payload_len,
                    payload_sha256=str(header["payload_sha256"]),
                )
            )
            del self._buffer[:end]
        if len(self._buffer) > self._max_header + self._max_payload:
            raise ProtocolError("frame buffer exceeds the limit")
        return frames

    def finish(self) -> None:
        if self._buffer:
            raise ProtocolError("stream ended with a partial frame")


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    out = bytearray()
    while len(out) < size:
        chunk = stream.read(size - len(out))
        if not chunk:
            raise ProtocolError("stream ended in the middle of a frame")
        out.extend(chunk)
    return bytes(out)


def read_frame(stream: BinaryIO, *, max_payload: int = MAX_FRAME_PAYLOAD) -> Frame:
    """Read exactly one frame from a blocking binary stream."""
    header_len_raw = _read_exact(stream, 4)
    header_len = struct.unpack(">I", header_len_raw)[0]
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ProtocolError("frame header length is outside the limit")
    header = _header(_read_exact(stream, header_len), max_payload=max_payload)
    payload_len = int(header["payload_length"])
    if payload_len > max_payload:
        raise ProtocolError("frame payload length is outside the limit")
    raw = _read_exact(stream, payload_len)
    return Frame(
        frame_type=str(header["frame_type"]),
        request_id=str(header["request_id"]),
        payload=_payload(raw, header),
        payload_length=payload_len,
        payload_sha256=str(header["payload_sha256"]),
    )


def write_frame(
    stream: BinaryIO, frame_type: str, request_id: str, payload: dict[str, Any], *,
    max_payload: int = MAX_FRAME_PAYLOAD,
) -> None:
    stream.write(encode_frame(frame_type, request_id, payload, max_payload=max_payload))
    stream.flush()


from tcenv.evaluation import EvaluationRequest, EvaluatorProvenance, EvaluationResult


def provenance_epoch(**facts: str) -> str:
    """Stable epoch over the fields that make an evaluator environment."""
    return sha256_hex(canonical_json(facts))


@dataclass(frozen=True)
class HelloRequest:
    """First frame on an authenticated connection."""

    protocol_version: str
    client_nonce: str
    session_token: str
    expected_site_id: str = ""
    expected_worker_build: str = ""
    expected_task_pack_digest: str = ""
    target_id: str = ""

    def __post_init__(self) -> None:
        for name in ("protocol_version", "client_nonce", "session_token"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ProtocolError(f"{name} must be a non-empty string")
        if len(self.client_nonce) > 128 or len(self.session_token) > 512:
            raise ProtocolError("hello nonce or token is too long")

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> "HelloRequest":
        if not isinstance(document, dict):
            raise ProtocolError("hello must be an object")
        fields = {"protocol_version", "client_nonce", "session_token", "expected_site_id", "expected_worker_build", "expected_task_pack_digest", "target_id"}
        unknown = set(document) - fields
        if unknown:
            raise ProtocolError(f"unknown hello fields: {sorted(unknown)}")
        missing = {"protocol_version", "client_nonce", "session_token"} - set(document)
        if missing:
            raise ProtocolError(f"missing hello fields: {sorted(missing)}")
        return cls(**document)
