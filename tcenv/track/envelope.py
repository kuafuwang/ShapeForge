"""Single-submission segmented envelope for multi-file candidates (A3).

The agent/evaluator chain submits one text.  A ``msl-bundle`` candidate is several files plus its
geometry, so it travels as one envelope built from the *fixed* candidate manifest and one
``taic-file`` segment per entry.  The envelope is transport only:

* after decoding it is an ordinary ``CandidateArtifact`` with the same entry bytes;
* ``candidate_sha256`` is the candidate-entry manifest digest (:attr:`CandidateArtifact.manifest_bytes`),
  never ``sha256(envelope)``;
* re-ordering files, the author, the model, the committee or the session changes the envelope text
  but not the candidate identity; changing one byte of one entry changes it.

The format is length-delimited, not delimiter-delimited: a file body may contain any text
(including a line that looks like a marker), because the parser reads exactly the declared number
of bytes.  A body must be UTF-8 (candidate sources are text); non-UTF-8 bytes fail closed instead
of being silently escaped.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from typing import Any

from ..contract_errors import TaskContractError
from ..schema import SchemaError, loads_strict
from .plane import CandidateArtifact, CandidateEntry

ENVELOPE_SCHEMA_ID = "tcenv.track.candidate-envelope.1"
ENVELOPE_MAGIC = b"taic-envelope 1"
ENVELOPE_END = b"taic-end-envelope"
FILE_MARKER = b"taic-file"
MAX_ENVELOPE_BYTES = 1_000_000
MAX_ENVELOPE_ENTRIES = 64
_TRACK_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")
_ROLE = re.compile(r"[a-z_]{1,32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class EnvelopeError(TaskContractError):
    """@brief 分段 envelope 不是同一次候选提交。"""

    code = "candidate_envelope_invalid"


def is_candidate_envelope(text: str) -> bool:
    """@brief 文本是否是完整的分段候选 envelope（只做标记检查，不解析）。

    @param text 提交文本。
    @return 含起止标记时为 True。
    """
    if not isinstance(text, str):
        return False
    return ENVELOPE_MAGIC.decode("ascii") in text.splitlines() and ENVELOPE_END.decode("ascii") in text


def _b64(data: bytes, what: str) -> str:
    """@brief 把字节编码成无换行的标准 base64。

    @param data 字节。
    @param what 诊断名称。
    @return ASCII 文本。
    """
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str, what: str) -> bytes:
    """@brief 解码一段标准 base64。

    @param text ASCII 文本。
    @param what 诊断名称。
    @return 字节。
    @raises EnvelopeError 非法 base64 或非 ASCII。
    """
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise EnvelopeError(f"{what} is not ASCII base64") from exc
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EnvelopeError(f"{what} is not valid base64: {exc}") from exc


def encode_envelope(candidate: CandidateArtifact) -> str:
    """@brief 把多文件候选编码成一份可提交文本。

    @param candidate 候选（不能是单文件提交；msl-single 直接提交文件字节）。
    @return envelope 文本。
    @raises EnvelopeError 单文件候选、路径/媒体类型/角色非法、字节不是 UTF-8 或超出上限。
    """
    if not isinstance(candidate, CandidateArtifact):
        raise EnvelopeError("encode_envelope needs a CandidateArtifact")
    if candidate.single_source:
        raise EnvelopeError(
            "a single-source candidate is submitted as its file bytes; the segmented envelope is only for a "
            "multi-file bundle"
        )
    if not _TRACK_ID.fullmatch(candidate.track_id):
        raise EnvelopeError(f"candidate track id {candidate.track_id!r} is not a transport identifier")
    if len(candidate.entries) > MAX_ENVELOPE_ENTRIES:
        raise EnvelopeError(f"a candidate envelope carries at most {MAX_ENVELOPE_ENTRIES} entries")
    manifest = candidate.manifest_bytes
    parts: list[bytes] = [
        ENVELOPE_MAGIC + b"\n",
        b"taic-track " + candidate.track_id.encode("ascii") + b"\n",
        b"taic-manifest " + _b64(manifest, "manifest").encode("ascii") + b"\n",
    ]
    for entry in sorted(candidate.entries, key=lambda item: item.path):
        body = bytes(entry.data)
        try:
            body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EnvelopeError(
                f"candidate entry {entry.path!r} is not UTF-8 text and cannot travel in the text envelope: {exc}"
            ) from exc
        if not _ROLE.fullmatch(entry.role):
            raise EnvelopeError(f"candidate entry {entry.path!r} has an illegal transport role {entry.role!r}")
        header = (
            FILE_MARKER
            + b" "
            + _b64(entry.path.encode("utf-8"), "path").encode("ascii")
            + b" "
            + entry.role.encode("ascii")
            + b" "
            + _b64(entry.media_type.encode("utf-8"), "media type").encode("ascii")
            + b" "
            + str(len(body)).encode("ascii")
            + b" "
            + entry.sha256.encode("ascii")
        )
        parts.append(header + b"\n")
        parts.append(body)
        parts.append(b"\ntaic-end-file\n")
    parts.append(ENVELOPE_END + b"\n")
    raw = b"".join(parts)
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise EnvelopeError(f"candidate envelope is {len(raw)} bytes; the transport limit is {MAX_ENVELOPE_BYTES}")
    return raw.decode("utf-8")


def decode_envelope(text: str) -> CandidateArtifact:
    """@brief 从一份提交文本解出候选；身份是 manifest digest，不是 envelope 的哈希。

    @param text envelope 文本。
    @return 候选。
    @raises EnvelopeError 头/段/长度/摘要/manifest 与 entries 不一致。
    """
    if not isinstance(text, str):
        raise EnvelopeError("a candidate envelope must be text")
    try:
        remaining = text.encode("utf-8")
    except UnicodeEncodeError as exc:  # pragma: no cover - str is always encodable
        raise EnvelopeError(f"candidate envelope is not UTF-8 text: {exc}") from exc
    if len(remaining) > MAX_ENVELOPE_BYTES:
        raise EnvelopeError(f"candidate envelope is {len(remaining)} bytes; the transport limit is {MAX_ENVELOPE_BYTES}")
    if not remaining.startswith(ENVELOPE_MAGIC + b"\n"):
        raise EnvelopeError(f"candidate envelope must start with `{ENVELOPE_MAGIC.decode('ascii')}`")

    def line(prefix: bytes | None = None) -> bytes:
        nonlocal remaining
        index = remaining.find(b"\n")
        if index < 0:
            raise EnvelopeError("candidate envelope ended inside a header line")
        value, remaining = remaining[:index], remaining[index + 1 :]
        if prefix is not None and not value.startswith(prefix):
            raise EnvelopeError(f"candidate envelope expected a line starting with {prefix!r}")
        return value if prefix is None else value[len(prefix) :]

    remaining = remaining[len(ENVELOPE_MAGIC) + 1 :]
    track = line(b"taic-track ")
    try:
        track_id = track.decode("ascii")
    except UnicodeDecodeError as exc:
        raise EnvelopeError("candidate envelope track id is not ASCII") from exc
    if not _TRACK_ID.fullmatch(track_id):
        raise EnvelopeError(f"candidate envelope track id {track_id!r} is not a transport identifier")
    manifest_bytes = _unb64(line(b"taic-manifest ").decode("ascii"), "manifest")
    try:
        manifest = loads_strict(manifest_bytes.decode("utf-8"), what="candidate manifest")
    except (UnicodeDecodeError, SchemaError) as exc:
        raise EnvelopeError(f"candidate envelope manifest is not strict JSON: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"schema_id", "entries"}:
        raise EnvelopeError("candidate manifest must contain exactly schema_id and entries")
    manifest_entries = manifest["entries"]
    if not isinstance(manifest_entries, list) or not manifest_entries:
        raise EnvelopeError("candidate manifest must list at least one entry")
    if len(manifest_entries) > MAX_ENVELOPE_ENTRIES:
        raise EnvelopeError(
            f"a candidate envelope carries at most {MAX_ENVELOPE_ENTRIES} entries"
        )
    by_path: dict[str, dict[str, Any]] = {}
    for item in manifest_entries:
        if not isinstance(item, dict) or set(item) != {"path", "role", "media_type", "size_bytes", "sha256"}:
            raise EnvelopeError("each candidate manifest entry must contain path/role/media_type/size_bytes/sha256")
        path = item["path"]
        if not isinstance(path, str) or path in by_path:
            raise EnvelopeError(f"candidate manifest has a duplicate/invalid path {path!r}")
        by_path[path] = item

    entries: list[CandidateEntry] = []
    while True:
        if remaining.startswith(ENVELOPE_END + b"\n"):
            remaining = remaining[len(ENVELOPE_END) + 1 :]
            break
        if not remaining.startswith(FILE_MARKER + b" "):
            raise EnvelopeError("candidate envelope has trailing or unexpected text before the end marker")
        header = line(FILE_MARKER + b" ").split(b" ")
        if len(header) != 5:
            raise EnvelopeError("a taic-file header must contain path/role/media_type/size/sha256")
        raw_path, raw_role, raw_media, raw_size, raw_sha = header
        try:
            path = _unb64(raw_path.decode("ascii"), "path").decode("utf-8")
            media_type = _unb64(raw_media.decode("ascii"), "media type").decode("utf-8")
            role = raw_role.decode("ascii")
            size = int(raw_size.decode("ascii"))
            header_sha = raw_sha.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise EnvelopeError(f"a taic-file header is not decodable: {exc}") from exc
        if not _SHA256.fullmatch(header_sha):
            raise EnvelopeError(f"a taic-file header has an invalid sha256 {header_sha!r}")
        if size < 0 or size > MAX_ENVELOPE_BYTES:
            raise EnvelopeError("a taic-file body size is outside the transport budget")
        if len(remaining) < size + len(b"\ntaic-end-file\n"):
            raise EnvelopeError("candidate envelope ended inside a taic-file body")
        body = remaining[:size]
        remaining = remaining[size:]
        if not remaining.startswith(b"\ntaic-end-file\n"):
            raise EnvelopeError("a taic-file body is not followed by its end marker")
        remaining = remaining[len(b"\ntaic-end-file\n") :]
        try:
            data = body.decode("utf-8").encode("utf-8")
        except UnicodeDecodeError as exc:
            raise EnvelopeError(f"candidate entry {path!r} is not UTF-8 text: {exc}") from exc
        declared = by_path.get(path)
        if declared is None:
            raise EnvelopeError(f"candidate envelope carries an undeclared entry {path!r}")
        actual_sha = hashlib.sha256(data).hexdigest()
        if header_sha != declared["sha256"] or header_sha != actual_sha:
            raise EnvelopeError(
                f"candidate entry {path!r}: the taic-file header sha256 is not the manifest/body digest"
            )
        if (
            role != declared["role"]
            or media_type != declared["media_type"]
            or size != declared["size_bytes"]
            or actual_sha != declared["sha256"]
        ):
            raise EnvelopeError(f"candidate entry {path!r} does not match its manifest")
        if len(entries) >= MAX_ENVELOPE_ENTRIES:
            raise EnvelopeError(
                f"a candidate envelope carries at most {MAX_ENVELOPE_ENTRIES} entries"
            )
        entries.append(CandidateEntry(path=path, role=role, media_type=media_type, data=data))
        if len(entries) > len(by_path):
            raise EnvelopeError("candidate envelope carries more files than its manifest")

    if remaining not in (b"", b"\n"):
        raise EnvelopeError("candidate envelope has trailing text after its end marker")
    if len(entries) != len(by_path):
        missing = sorted(set(by_path) - {entry.path for entry in entries})
        raise EnvelopeError(f"candidate envelope is missing declared entries {missing}")
    candidate = CandidateArtifact(track_id=track_id, entries=tuple(entries))
    if candidate.manifest_bytes != manifest_bytes:
        raise EnvelopeError(
            "the embedded candidate manifest is not the canonical manifest of the carried entries; the envelope "
            "and the candidate identity disagree"
        )
    return candidate


__all__ = [
    "ENVELOPE_MAGIC",
    "ENVELOPE_SCHEMA_ID",
    "MAX_ENVELOPE_BYTES",
    "MAX_ENVELOPE_ENTRIES",
    "EnvelopeError",
    "decode_envelope",
    "encode_envelope",
    "is_candidate_envelope",
]
