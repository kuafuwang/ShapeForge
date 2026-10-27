"""Independent content-addressed blob transfer channel (P0b).

The six-RPC Evaluator control plane carries metadata only.  Candidate bytes for
a large model or graph travel through this module instead: a chunked,
content-addressed, grant-bound store with an explicit state machine.  Nothing in
this module is Protobuf, gRPC, or vendor specific, and it never serializes the
model: the bytes stay opaque and are verified against a declared size and
SHA-256.

Design rules (see ``docs/架构/多厂商NPU评测通信抽象-plan.md`` §4.5):

* a blob is addressed by ``content_id = "blob-sha256-v1:" + sha256``; the
  declared digest must equal the real bytes at commit time;
* writes are strictly sequential (``offset == received_bytes``), so an
  out-of-order, replayed-with-different-bytes, oversized or over-quota chunk
  fails closed instead of silently corrupting the artifact;
* every operation is bound to one :class:`BlobScope` (site/device/target and the
  grant's subject/run/session) and, once known, to one attempt/fence;
* crash recovery resumes from the manifest's verified chunks; nothing past them
  is ever trusted;
* terminal cleanup releases the request lease, and a cleanup failure marks the
  blob ``quarantine`` instead of pretending it was deleted.

Only the single-active frontend process writes this store, so the manifest
read-modify-write is guarded by a process-local lock.  Multi-process writers are
explicitly out of scope.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from . import contracts

BLOB_MANIFEST_NAME = "manifest.json"
BLOB_CHUNKS_DIR = "chunks"
BLOB_MANIFEST_SCHEMA_VERSION = 1
#: Streaming hash block.  Reading a blob never loads more than one block (or one
#: requested range, which is itself capped by ``chunk_size``).
BLOB_HASH_BLOCK_BYTES = 64 * 1024

STATE_ACCEPTED = "accepted"
STATE_UPLOADING = "uploading"
STATE_READY = "ready"
STATE_FAILED = "failed"
STATE_QUARANTINE = "quarantine"
BLOB_STATES = frozenset({STATE_ACCEPTED, STATE_UPLOADING, STATE_READY, STATE_FAILED, STATE_QUARANTINE})
CLEANUP_PENDING = "pending"
CLEANUP_DONE = "done"
CLEANUP_QUARANTINE = "quarantine"

_IDENTIFIER = re.compile(r"[A-Za-z0-9._:-]{1,160}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class BlobError(Exception):
    """A blob-channel operation was refused or failed.

    ``code`` is the transport-neutral classification a gRPC adapter maps to a
    status: ``not_found``, ``permission_denied``, ``failed_precondition``,
    ``resource_exhausted``, ``out_of_range`` or ``unavailable``.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _require_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise BlobError("failed_precondition", f"{name} must be a bounded opaque identifier")
    return value


@dataclass(frozen=True)
class BlobScope:
    """The authorization scope a blob belongs to.

    It is derived from the per-RPC session grant plus the deployment identity, so
    a caller can never widen it.  A blob recorded under one scope is invisible to
    every other scope even when the caller knows its ``content_id``.
    """

    site_id: str
    device_id: str
    target_id: str
    subject: str
    run_id: str
    session_id: str

    def __post_init__(self) -> None:
        for name in ("site_id", "device_id", "target_id", "subject", "run_id", "session_id"):
            _require_identifier(getattr(self, name), f"scope.{name}")

    def document(self) -> dict[str, str]:
        return {
            "site_id": self.site_id,
            "device_id": self.device_id,
            "target_id": self.target_id,
            "subject": self.subject,
            "run_id": self.run_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_document(cls, value: Any) -> "BlobScope":
        if not isinstance(value, Mapping) or set(value) != {
            "site_id", "device_id", "target_id", "subject", "run_id", "session_id",
        }:
            raise BlobError("failed_precondition", "blob scope fields are missing or unknown")
        return cls(**{name: value[name] for name in (
            "site_id", "device_id", "target_id", "subject", "run_id", "session_id",
        )})


@dataclass(frozen=True)
class BlobAttempt:
    """The attempt/fence that owns an upload or commit.

    A stale attempt can neither finish another attempt's upload nor clean it up.
    """

    attempt_id: str
    lease_fence: int

    def __post_init__(self) -> None:
        _require_identifier(self.attempt_id, "attempt.attempt_id")
        if isinstance(self.lease_fence, bool) or not isinstance(self.lease_fence, int) or self.lease_fence < 1:
            raise BlobError("failed_precondition", "attempt.lease_fence must be a positive integer")

    def document(self) -> dict[str, Any]:
        return {"attempt_id": self.attempt_id, "lease_fence": self.lease_fence}

    @classmethod
    def from_document(cls, value: Any) -> "BlobAttempt":
        if not isinstance(value, Mapping) or set(value) != {"attempt_id", "lease_fence"}:
            raise BlobError("failed_precondition", "blob attempt fields are missing or unknown")
        return cls(attempt_id=value["attempt_id"], lease_fence=value["lease_fence"])


@dataclass(frozen=True)
class BlobLimits:
    """Policy limits for one store.

    They are deliberately explicit: a size or chunk limit that is not configured
    is a deployment decision, not an implicit ``unlimited``.
    """

    max_size_bytes: int = 1 << 30
    min_chunk_size: int = 1
    max_chunk_size: int = contracts.MAX_BLOB_CHUNK_BYTES
    max_chunks: int = 1 << 16
    max_uploads_per_scope: int = 4
    max_scope_bytes: int = 2 << 30
    max_total_bytes: int = 8 << 30
    #: An unfinished upload expires after this long.
    partial_ttl_ms: int = 15 * 60 * 1000
    #: A committed blob expires after this long without a lease.
    ready_ttl_ms: int = 24 * 60 * 60 * 1000
    #: Whether releasing the last lease deletes the bytes.
    cleanup_on_release: bool = True

    def __post_init__(self) -> None:
        for name in ("max_size_bytes", "min_chunk_size", "max_chunk_size", "max_chunks",
                     "max_uploads_per_scope", "max_scope_bytes", "max_total_bytes",
                     "partial_ttl_ms", "ready_ttl_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"blob limit {name} must be a positive integer")
        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError("blob min_chunk_size exceeds max_chunk_size")
        if self.max_chunk_size > contracts.MAX_BLOB_CHUNK_BYTES:
            raise ValueError("blob max_chunk_size exceeds the protocol ceiling")

    def document(self) -> dict[str, Any]:
        return {
            "max_size_bytes": self.max_size_bytes,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "max_chunks": self.max_chunks,
            "max_uploads_per_scope": self.max_uploads_per_scope,
            "max_scope_bytes": self.max_scope_bytes,
            "max_total_bytes": self.max_total_bytes,
            "partial_ttl_ms": self.partial_ttl_ms,
            "ready_ttl_ms": self.ready_ttl_ms,
            "cleanup_on_release": self.cleanup_on_release,
        }


@dataclass(frozen=True)
class BlobState:
    """The durable view of one blob (safe to return to the caller)."""

    descriptor: dict[str, Any]
    state: str
    received_bytes: int
    verified_chunks: tuple[int, ...]
    created_at_ms: int
    updated_at_ms: int
    expires_at_ms: int
    scope: dict[str, str]
    attempt: dict[str, Any] | None
    leases: tuple[str, ...]
    cleanup: dict[str, Any]
    quarantine: dict[str, Any] | None

    @property
    def content_id(self) -> str:
        return self.descriptor["content_id"]

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY

    def document(self) -> dict[str, Any]:
        return {
            "descriptor": dict(self.descriptor),
            "state": self.state,
            "received_bytes": self.received_bytes,
            "verified_chunks": list(self.verified_chunks),
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "scope": dict(self.scope),
            "attempt": None if self.attempt is None else dict(self.attempt),
            "leases": list(self.leases),
            "cleanup": dict(self.cleanup),
            "quarantine": None if self.quarantine is None else dict(self.quarantine),
        }


class BlobIO:
    """The filesystem boundary, so tests can inject real I/O failures.

    Production uses :class:`LocalBlobIO`.  A test subclass raises ``OSError`` (or
    a crash signal) from one of these methods; the channel must then fail the
    operation and quarantine the artifact rather than report success.
    """

    def prepare_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def read(self, path: Path, limit: int) -> bytes:
        with path.open("rb") as handle:
            return handle.read(limit)

    def write_atomic(self, path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def remove(self, path: Path) -> None:
        path.unlink()

    def remove_tree(self, path: Path) -> None:
        for child in sorted(path.iterdir()):
            if child.is_dir():
                self.remove_tree(child)
            else:
                child.unlink()
        path.rmdir()

    def exists(self, path: Path) -> bool:
        return path.exists()

    def listdir(self, path: Path) -> list[str]:
        return sorted(entry.name for entry in path.iterdir())

    def size(self, path: Path) -> int:
        return path.stat().st_size


class LocalBlobIO(BlobIO):
    """The production filesystem implementation."""


def _content_digest(content_id: str) -> str:
    if not isinstance(content_id, str) or not content_id.startswith(contracts.BLOB_CONTENT_ID_PREFIX):
        raise BlobError("failed_precondition", "blob content id has an unsupported algorithm prefix")
    digest = content_id[len(contracts.BLOB_CONTENT_ID_PREFIX):]
    if not _HEX64.fullmatch(digest):
        raise BlobError("failed_precondition", "blob content id must end with 64 lowercase hex characters")
    return digest


@dataclass
class _BlobFiles:
    """Path/manifest plumbing shared by the writer and the trusted reader."""

    root: Path
    io: BlobIO

    def directory(self, digest: str) -> Path:
        return self.root / "blobs" / digest[:2] / digest

    def manifest_path(self, digest: str) -> Path:
        return self.directory(digest) / BLOB_MANIFEST_NAME

    def chunk_path(self, digest: str, index: int) -> Path:
        return self.directory(digest) / BLOB_CHUNKS_DIR / f"{index:08d}.chunk"

    def load(self, digest: str) -> dict[str, Any] | None:
        path = self.manifest_path(digest)
        if not self.io.exists(path):
            return None
        try:
            document = json.loads(self.io.read(path, contracts.MAX_CANONICAL_BYTES).decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise BlobError("unavailable", f"blob manifest is unreadable: {exc}") from exc
        if not isinstance(document, dict):
            raise BlobError("unavailable", "blob manifest is not an object")
        if document.get("schema_version") != BLOB_MANIFEST_SCHEMA_VERSION:
            raise BlobError("unavailable", "blob manifest has an unsupported schema version")
        return document

    def store(self, digest: str, document: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(document), sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.io.write_atomic(self.manifest_path(digest), payload)

    def iter_blob_digests(self) -> Iterator[str]:
        base = self.root / "blobs"
        if not self.io.exists(base):
            return
        for prefix in self.io.listdir(base):
            directory = base / prefix
            if not directory.is_dir():
                continue
            for name in self.io.listdir(directory):
                if _HEX64.fullmatch(name) and name[:2] == prefix:
                    yield name

    def chunks(self, digest: str) -> list[Path]:
        path = self.directory(digest) / BLOB_CHUNKS_DIR
        if not self.io.exists(path):
            return []
        return [path / name for name in self.io.listdir(path)]

    def iter_file_blocks(self, path: Path, *, block: int = BLOB_HASH_BLOCK_BYTES) -> Iterator[bytes]:
        with path.open("rb") as handle:
            while True:
                data = handle.read(block)
                if not data:
                    return
                yield data


class BlobStore:
    """The writer/reader side of the content-addressed blob channel."""

    def __init__(
        self, root: Path | str, *, limits: BlobLimits | None = None, io: BlobIO | None = None,
        clock: Callable[[], float] | None = None, cleanup_on_release: bool | None = None,
    ) -> None:
        self.root = Path(root)
        self.io = io or LocalBlobIO()
        self.limits = limits or BlobLimits()
        if cleanup_on_release is not None:
            self.limits = dataclasses.replace(self.limits, cleanup_on_release=bool(cleanup_on_release))
        self._clock = clock or (lambda: 0.0)
        self._lock = threading.RLock()
        self._files = _BlobFiles(self.root, self.io)
        self.io.prepare_dir(self.root / "blobs")

    # -- small helpers -----------------------------------------------------
    def now_ms(self) -> int:
        return int(self._clock() * 1000)

    @property
    def files(self) -> _BlobFiles:
        return self._files

    def _normalize_descriptor(self, descriptor: Any) -> dict[str, Any]:
        try:
            normalized = contracts.blob_descriptor_document(descriptor)
        except contracts.ContractError as exc:
            raise BlobError("failed_precondition", str(exc)) from exc
        if normalized["size_bytes"] > self.limits.max_size_bytes:
            raise BlobError("resource_exhausted", "blob exceeds the configured per-blob byte limit")
        chunk_size = normalized["chunk_size"]
        if not self.limits.min_chunk_size <= chunk_size <= self.limits.max_chunk_size:
            raise BlobError("resource_exhausted", "blob chunk_size is outside the configured bounds")
        chunks = 0 if normalized["size_bytes"] == 0 else -(-normalized["size_bytes"] // chunk_size)
        if chunks > self.limits.max_chunks:
            raise BlobError("resource_exhausted", "blob would need more chunks than the configured limit")
        return normalized

    @staticmethod
    def _require_scope(manifest: Mapping[str, Any], scope: BlobScope) -> None:
        if manifest.get("scope") != scope.document():
            raise BlobError("permission_denied", "blob belongs to another grant scope")

    @staticmethod
    def _require_attempt(manifest: Mapping[str, Any], attempt: BlobAttempt | None) -> None:
        recorded = manifest.get("attempt")
        if recorded is None:
            return
        if attempt is None or recorded != attempt.document():
            raise BlobError("permission_denied", "blob upload belongs to another attempt/fence")

    def _require_live(
        self, manifest: Mapping[str, Any], *, now: int, allow_failed: bool = False,
    ) -> None:
        if not allow_failed and manifest.get("state") in {STATE_FAILED, STATE_QUARANTINE}:
            if manifest["state"] == STATE_QUARANTINE:
                raise BlobError("failed_precondition", "blob is quarantined and cannot be reused")
            raise BlobError("failed_precondition", "blob is failed")
        expires = manifest.get("expires_at_ms")
        if isinstance(expires, int) and now >= expires:
            raise BlobError("not_found", "blob has expired")

    @staticmethod
    def _state(manifest: Mapping[str, Any]) -> BlobState:
        return BlobState(
            descriptor=dict(manifest["descriptor"]),
            state=manifest["state"],
            received_bytes=int(manifest["received_bytes"]),
            verified_chunks=tuple(int(index) for index in manifest["verified_chunks"]),
            created_at_ms=int(manifest["created_at_ms"]),
            updated_at_ms=int(manifest["updated_at_ms"]),
            expires_at_ms=int(manifest["expires_at_ms"]),
            scope=dict(manifest["scope"]),
            attempt=None if manifest.get("attempt") is None else dict(manifest["attempt"]),
            leases=tuple(sorted(entry["request_id"] for entry in manifest["leases"])),
            cleanup=dict(manifest["cleanup"]),
            quarantine=None if manifest.get("quarantine") is None else dict(manifest["quarantine"]),
        )

    def _load_owned(
        self, scope: BlobScope, content_id: str, *, now: int, allow_failed: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        digest = _content_digest(content_id)
        manifest = self._files.load(digest)
        if manifest is None:
            raise BlobError("not_found", "blob does not exist")
        if manifest["descriptor"] != contracts.blob_descriptor_document(manifest["descriptor"]):
            raise BlobError("unavailable", "blob manifest descriptor is inconsistent")
        self._require_scope(manifest, scope)
        self._require_live(manifest, now=now, allow_failed=allow_failed)
        return digest, manifest

    @staticmethod
    def _charge(manifest: Mapping[str, Any], on_disk: int) -> int:
        """Return the bytes this blob occupies against the quotas.

        A declared-but-unfinished upload reserves its **whole declared size**, not
        just the chunks already on disk: otherwise two concurrent declarations
        could both pass the check and then both write past the quota.  A ready
        blob is charged its declared (verified) bytes.  A failed/quarantined blob
        is charged whatever residue still exists on disk, because that space is
        not free yet.
        """

        declared = int(manifest["descriptor"]["size_bytes"])
        if manifest["state"] in {STATE_ACCEPTED, STATE_UPLOADING, STATE_READY}:
            return declared
        return on_disk

    @staticmethod
    def _scope_key(scope: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(scope.items()))

    def _ledger_locked(self, *, byte_stats: bool = False) -> dict[str, Any]:
        """One pass over the store: declared charges, reservations and totals.

        ``quota_bytes`` is what the quota checks use: the declared size of every
        ready or unfinished blob plus the residual bytes of failed artifacts.

        The write path deliberately asks without ``byte_stats``: a chunk write
        must not stat every other artifact, and for ready/unfinished blobs the
        declared charge (which is what the quota reserves) is already exact.  The
        ``*_bytes`` fields are the on-disk view used by ``usage()`` and tests.
        """

        ledger: dict[str, Any] = {
            "blobs": 0, "ready_bytes": 0, "partial_bytes": 0, "open_uploads": 0,
            "reserved_bytes": 0, "quota_bytes": 0,
            "scope_charge": {}, "scope_uploads": {},
        }
        for digest in self._files.iter_blob_digests():
            manifest = self._files.load(digest)
            if manifest is None:
                continue
            ledger["blobs"] += 1
            state = manifest["state"]
            if state in {STATE_READY, STATE_ACCEPTED, STATE_UPLOADING}:
                on_disk = 0
                if byte_stats:
                    on_disk = sum(self.io.size(path) for path in self._files.chunks(digest))
                    if state == STATE_READY:
                        ledger["ready_bytes"] += on_disk
                    else:
                        ledger["partial_bytes"] += on_disk
                charge = int(manifest["descriptor"]["size_bytes"])
            else:
                # A failed/quarantined artifact may still hold residual bytes
                # (cleanup failed): that space is not free yet.
                on_disk = sum(self.io.size(path) for path in self._files.chunks(digest))
                ledger["partial_bytes"] += on_disk
                charge = on_disk
            ledger["quota_bytes"] += charge
            key = self._scope_key(manifest["scope"])
            ledger["scope_charge"][key] = ledger["scope_charge"].get(key, 0) + charge
            if state in {STATE_ACCEPTED, STATE_UPLOADING}:
                ledger["open_uploads"] += 1
                ledger["reserved_bytes"] += charge
                ledger["scope_uploads"][key] = ledger["scope_uploads"].get(key, 0) + 1
        return ledger

    def usage(self) -> dict[str, Any]:
        """Return the store ledger (used by quota checks and the tests)."""

        with self._lock:
            return self._ledger_locked(byte_stats=True)

    def _check_quota_locked(
        self, scope: BlobScope, digest: str | None, declared: int,
    ) -> None:
        """Refuse a declaration or a write that the quota cannot hold.

        ``digest`` names the blob being checked (``None`` for a new declaration);
        its own existing charge is excluded before the new declaration is added,
        so a replayed write only fails when the *rest* of the store grew or the
        limits shrank.
        """

        ledger = self._ledger_locked()
        own = 0
        if digest is not None:
            manifest = self._files.load(digest)
            if manifest is not None:
                on_disk = 0
                if manifest["state"] in {STATE_FAILED, STATE_QUARANTINE}:
                    on_disk = sum(self.io.size(path) for path in self._files.chunks(digest))
                own = self._charge(manifest, on_disk)
        key = self._scope_key(scope.document())
        if ledger["scope_charge"].get(key, 0) - own + declared > self.limits.max_scope_bytes:
            raise BlobError("resource_exhausted", "grant scope byte quota is exhausted")
        if ledger["quota_bytes"] - own + declared > self.limits.max_total_bytes:
            raise BlobError("resource_exhausted", "blob store byte quota is exhausted")

    # -- upload ------------------------------------------------------------
    def begin(
        self, scope: BlobScope, descriptor: Mapping[str, Any], *, attempt: BlobAttempt | None = None,
    ) -> BlobState:
        """Declare an upload (size/digest/chunk size) before any byte is stored.

        Re-declaring an existing content id is an idempotent resume only when the
        whole descriptor is identical; a different size, digest, media type or
        chunk size is a conflict, never an overwrite.
        """

        normalized = self._normalize_descriptor(descriptor)
        digest = _content_digest(normalized["content_id"])
        now = self.now_ms()
        with self._lock:
            manifest = self._files.load(digest)
            if manifest is not None:
                self._require_scope(manifest, scope)
                if manifest["descriptor"] != normalized:
                    raise BlobError(
                        "failed_precondition",
                        "content id was already declared with another size/digest/chunk size",
                    )
                self._require_live(manifest, now=now)
                if manifest["state"] == STATE_READY:
                    return self._state(manifest)
                self._require_attempt(manifest, attempt)
                # A resume keeps the durable state: `accepted` only becomes
                # `uploading` when the first chunk is acknowledged.
                return self._state(manifest)
            ledger = self._ledger_locked()
            key = self._scope_key(scope.document())
            if ledger["scope_uploads"].get(key, 0) >= self.limits.max_uploads_per_scope:
                raise BlobError("resource_exhausted", "scope has too many concurrent uploads")
            # The declared size is reserved now, before any byte is written, so a
            # second declaration cannot slip in and later overflow the quota.
            self._check_quota_locked(scope, None, normalized["size_bytes"])
            manifest = {
                "schema_version": BLOB_MANIFEST_SCHEMA_VERSION,
                "descriptor": normalized,
                # Declared but no byte stored yet; the first chunk moves it to
                # `uploading`.  Both states are durable and resumable.
                "state": STATE_ACCEPTED,
                "received_bytes": 0,
                "verified_chunks": [],
                "created_at_ms": now,
                "updated_at_ms": now,
                "expires_at_ms": now + self.limits.partial_ttl_ms,
                "scope": scope.document(),
                "attempt": None if attempt is None else attempt.document(),
                "leases": [],
                "cleanup": {"state": CLEANUP_PENDING, "attempts": 0, "last_error": None},
                "quarantine": None,
            }
            self.io.prepare_dir(self._files.directory(digest) / BLOB_CHUNKS_DIR)
            self._files.store(digest, manifest)
            return self._state(manifest)

    def write_chunk(
        self, scope: BlobScope, content_id: str, offset: int, data: bytes | bytearray | memoryview,
        *, attempt: BlobAttempt | None = None,
    ) -> BlobState:
        """Append one chunk at the only admissible offset (``received_bytes``).

        A replay of an already verified chunk is an idempotent acknowledgement
        only when the bytes are identical; a different payload at the same
        offset is a conflict.  Out-of-order chunks are refused so a hole can
        never be counted as progress.
        """

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise BlobError("failed_precondition", "chunk data must be bytes-like")
        payload = bytes(data)
        now = self.now_ms()
        with self._lock:
            digest, manifest = self._load_owned(scope, content_id, now=now)
            if manifest["state"] not in {STATE_ACCEPTED, STATE_UPLOADING}:
                raise BlobError("failed_precondition", f"blob is {manifest['state']}; only uploading accepts chunks")
            self._require_attempt(manifest, attempt)
            descriptor = manifest["descriptor"]
            if isinstance(offset, bool) or not isinstance(offset, int):
                raise BlobError("failed_precondition", "chunk offset must be an integer")
            if not payload:
                raise BlobError("failed_precondition", "chunk must not be empty")
            if len(payload) > descriptor["chunk_size"]:
                raise BlobError("resource_exhausted", "chunk exceeds the declared chunk_size")
            received = int(manifest["received_bytes"])
            index = len(manifest["verified_chunks"])
            if offset < received:
                # A replay is addressed by the declared chunk boundary, not by
                # "the last chunk": re-sending any acknowledged chunk (not only
                # the newest) is an idempotent ack when the bytes are identical.
                chunk_size = descriptor["chunk_size"]
                if offset % chunk_size:
                    raise BlobError(
                        "failed_precondition", "replayed offset is not a declared chunk boundary",
                    )
                replayed = offset // chunk_size
                if replayed >= len(manifest["verified_chunks"]):
                    raise BlobError(
                        "failed_precondition", "replayed offset is beyond the acknowledged chunks",
                    )
                expected = min(chunk_size, descriptor["size_bytes"] - offset)
                path = self._files.chunk_path(digest, replayed)
                if not self.io.exists(path):
                    raise BlobError("unavailable", "already acknowledged chunk is missing")
                stored = self.io.read(path, chunk_size + 1)
                if len(stored) != expected or len(payload) != expected:
                    raise BlobError(
                        "failed_precondition",
                        "replayed chunk length differs from the acknowledged chunk",
                    )
                if stored != payload:
                    raise BlobError("failed_precondition", "replayed offset carries different bytes")
                return self._state(manifest)
            if offset != received:
                raise BlobError("failed_precondition", "chunk offset must equal the verified upload length")
            # Re-check the quota on the write path as well: a shrunken limit or a
            # corrupted reservation must stop the upload instead of overshooting.
            self._check_quota_locked(scope, digest, descriptor["size_bytes"])
            if received + len(payload) > descriptor["size_bytes"]:
                raise BlobError("out_of_range", "chunk would exceed the declared blob size")
            expected_index = -(-descriptor["size_bytes"] // descriptor["chunk_size"]) - 1
            final = received + len(payload) == descriptor["size_bytes"]
            if not final and len(payload) != descriptor["chunk_size"]:
                raise BlobError("failed_precondition", "a non-final chunk must be exactly chunk_size bytes")
            if index > expected_index:
                raise BlobError("failed_precondition", "blob already has all of its declared chunks")
            path = self._files.chunk_path(digest, index)
            try:
                self.io.write_atomic(path, payload)
            except OSError as exc:
                raise BlobError("unavailable", f"chunk could not be stored: {exc}") from exc
            manifest["state"] = STATE_UPLOADING
            manifest["verified_chunks"].append(index)
            manifest["received_bytes"] = received + len(payload)
            manifest["updated_at_ms"] = now
            manifest["expires_at_ms"] = now + self.limits.partial_ttl_ms
            try:
                self._files.store(digest, manifest)
            except OSError as exc:
                # The chunk file is written but not acknowledged: recovery
                # rewrites it instead of trusting the unverified bytes.
                raise BlobError("unavailable", f"chunk manifest could not be updated: {exc}") from exc
            return self._state(manifest)

    def commit(
        self, scope: BlobScope, content_id: str, *, attempt: BlobAttempt | None = None,
    ) -> BlobState:
        """Verify the received bytes against the declared size and digest.

        Verification streams the chunk files; it never holds the whole blob.  A
        mismatch fails the blob closed (``failed``) and frees its chunks, so it
        can never be submitted as a complete artifact.
        """

        now = self.now_ms()
        with self._lock:
            digest, manifest = self._load_owned(scope, content_id, now=now)
            self._require_attempt(manifest, attempt)
            if manifest["state"] == STATE_READY:
                return self._state(manifest)
            if manifest["state"] not in {STATE_ACCEPTED, STATE_UPLOADING}:
                raise BlobError("failed_precondition", f"blob is {manifest['state']}; only uploading can commit")
            descriptor = manifest["descriptor"]
            if int(manifest["received_bytes"]) != descriptor["size_bytes"]:
                raise BlobError("failed_precondition", "blob is incomplete; committed size differs from the declaration")
            expected_chunks = 0 if descriptor["size_bytes"] == 0 else -(-descriptor["size_bytes"] // descriptor["chunk_size"])
            if len(manifest["verified_chunks"]) != expected_chunks:
                raise BlobError("failed_precondition", "blob is missing declared chunks")
            computed = hashlib.sha256()
            total = 0
            try:
                for index in range(expected_chunks):
                    path = self._files.chunk_path(digest, index)
                    if not self.io.exists(path):
                        raise BlobError("unavailable", "declared chunk is missing from the store")
                    for block in self._files.iter_file_blocks(path):
                        total += len(block)
                        computed.update(block)
            except OSError as exc:
                raise BlobError("unavailable", f"blob verification failed: {exc}") from exc
            actual = computed.hexdigest()
            if total != descriptor["size_bytes"] or actual != descriptor["sha256"]:
                self._fail_locked(digest, manifest, now, "digest or size verification failed")
                raise BlobError(
                    "failed_precondition",
                    f"blob digest verification failed (declared {descriptor['sha256']} / "
                    f"{descriptor['size_bytes']} bytes, received {actual} / {total} bytes)",
                )
            manifest["state"] = STATE_READY
            manifest["updated_at_ms"] = now
            manifest["expires_at_ms"] = now + self.limits.ready_ttl_ms
            self._files.store(digest, manifest)
            return self._state(manifest)

    def _fail_locked(self, digest: str, manifest: dict[str, Any], now: int, detail: str) -> None:
        manifest["state"] = STATE_FAILED
        manifest["failure"] = detail
        manifest["updated_at_ms"] = now
        manifest["verified_chunks"] = []
        manifest["received_bytes"] = 0
        manifest["expires_at_ms"] = now + self.limits.partial_ttl_ms
        for path in self._files.chunks(digest):
            try:
                self.io.remove(path)
            except OSError:
                manifest["state"] = STATE_QUARANTINE
                manifest["quarantine"] = {"reason": f"failure cleanup failed for {detail}", "at_ms": now}
                break
        self._files.store(digest, manifest)

    def abort(
        self, scope: BlobScope, content_id: str, *, attempt: BlobAttempt | None = None,
    ) -> BlobState:
        """Give up on an unfinished upload and free its bytes.

        Idempotent: aborting an already failed blob returns the same state.  A
        cleanup failure quarantines the blob instead of reporting a clean abort.
        """

        now = self.now_ms()
        with self._lock:
            digest, manifest = self._load_owned(scope, content_id, now=now, allow_failed=True)
            self._require_attempt(manifest, attempt)
            if manifest["state"] in {STATE_READY, STATE_FAILED}:
                return self._state(manifest)
            if manifest["state"] == STATE_ACCEPTED:
                manifest["state"] = STATE_FAILED
                manifest["failure"] = "aborted before any chunk"
                manifest["updated_at_ms"] = now
                manifest["expires_at_ms"] = now + self.limits.partial_ttl_ms
                self._files.store(digest, manifest)
                return self._state(manifest)
            self._fail_locked(digest, manifest, now, "aborted upload")
            return self._state(manifest)

    # -- read --------------------------------------------------------------
    def state(self, scope: BlobScope, content_id: str) -> BlobState:
        """Read the durable state, including a failed or quarantined artifact.

        Inspection must be able to report the failure that made the blob
        unusable; the read/write/attach paths still refuse it.
        """

        now = self.now_ms()
        with self._lock:
            _digest, manifest = self._load_owned(
                scope, content_id, now=now, allow_failed=True,
            )
            return self._state(manifest)

    def read_chunk(self, scope: BlobScope, content_id: str, offset: int, length: int) -> bytes:
        """Read one bounded range of a committed blob.

        ``length`` is capped by the declared ``chunk_size`` and the range must lie
        inside the blob, so a caller cannot ask the server to read a whole model
        into memory in one call.
        """

        now = self.now_ms()
        with self._lock:
            digest, manifest = self._load_owned(scope, content_id, now=now)
            return self._read_locked(digest, manifest, offset, length)

    def _read_locked(self, digest: str, manifest: Mapping[str, Any], offset: int, length: int) -> bytes:
        if manifest["state"] != STATE_READY:
            raise BlobError("failed_precondition", f"blob is {manifest['state']}; no bytes are readable")
        descriptor = manifest["descriptor"]
        if isinstance(offset, bool) or not isinstance(offset, int) or isinstance(length, bool) or not isinstance(length, int):
            raise BlobError("failed_precondition", "offset and length must be integers")
        if offset < 0 or length < 1:
            raise BlobError("out_of_range", "offset/length are outside the blob")
        if length > descriptor["chunk_size"]:
            raise BlobError("out_of_range", "read length exceeds the declared chunk_size")
        if offset + length > descriptor["size_bytes"]:
            raise BlobError("out_of_range", "read range exceeds the declared blob size")
        first = offset // descriptor["chunk_size"]
        last = (offset + length - 1) // descriptor["chunk_size"]
        collected = bytearray()
        for index in range(first, last + 1):
            path = self._files.chunk_path(digest, index)
            if not self.io.exists(path):
                raise BlobError("unavailable", "declared chunk is missing from the store")
            chunk = self.io.read(path, descriptor["chunk_size"] + 1)
            start_in_chunk = max(0, offset - index * descriptor["chunk_size"])
            end_in_chunk = min(len(chunk), offset + length - index * descriptor["chunk_size"])
            collected += chunk[start_in_chunk:end_in_chunk]
        data = bytes(collected)
        if len(data) != length:
            raise BlobError("unavailable", "stored chunk is shorter than its declared boundary")
        return data

    def iter_chunks(
        self, scope: BlobScope, content_id: str, *, offset: int = 0,
    ) -> Iterator[bytes]:
        """Yield verified chunks in order without buffering the whole blob.

        The final chunk verifies the complete digest; a corrupted store therefore
        raises after the last chunk instead of handing the worker a truncated or
        altered artifact.
        """

        now = self.now_ms()
        with self._lock:
            digest, manifest = self._load_owned(scope, content_id, now=now)
        if manifest["state"] != STATE_READY:
            raise BlobError("failed_precondition", f"blob is {manifest['state']}; no bytes are readable")
        yield from _iter_verified_files(self._files, digest, manifest, offset)

    # -- leases and lifecycle ----------------------------------------------
    def attach(
        self, scope: BlobScope, content_id: str, request_id: str, *,
        attempt: BlobAttempt | None = None,
    ) -> BlobState:
        """Bind a committed blob to one accepted request.

        The lease is what lets a terminal job release (or safely keep) the bytes;
        an upload that is not ``ready`` can never be attached, which is also what
        keeps an incomplete blob out of Submit.
        """

        _require_identifier(request_id, "request_id")
        now = self.now_ms()
        with self._lock:
            digest, manifest = self._load_owned(scope, content_id, now=now)
            if manifest["state"] != STATE_READY:
                raise BlobError("failed_precondition", "only a ready blob can be attached to a request")
            self._require_attempt(manifest, attempt)
            entry = {
                "request_id": request_id,
                "attempt_id": None if attempt is None else attempt.attempt_id,
                "lease_fence": None if attempt is None else attempt.lease_fence,
            }
            existing = [item for item in manifest["leases"] if item["request_id"] == request_id]
            if existing:
                if existing[0] != entry:
                    raise BlobError("failed_precondition", "request lease is bound to another attempt/fence")
                return self._state(manifest)
            manifest["leases"].append(entry)
            manifest["updated_at_ms"] = now
            manifest["expires_at_ms"] = now + self.limits.ready_ttl_ms
            self._files.store(digest, manifest)
            return self._state(manifest)

    def leases(self) -> dict[str, list[str]]:
        """Return ``request_id -> [content_id, ...]`` for reconciliation."""

        result: dict[str, list[str]] = {}
        with self._lock:
            for digest in self._files.iter_blob_digests():
                manifest = self._files.load(digest)
                if manifest is None:
                    continue
                for entry in manifest["leases"]:
                    result.setdefault(entry["request_id"], []).append(manifest["descriptor"]["content_id"])
        return result

    def detach(
        self, scope: BlobScope, content_id: str, request_id: str,
    ) -> dict[str, Any]:
        """Drop one request's lease on one blob without touching its bytes.

        This is the narrow rollback a rejected multi-entry Submit needs: it must
        not release another blob's lease for the same request (a request may
        already own a live lease from an earlier admission) and it must not
        delete bytes the caller may reference again.  ``release`` remains the
        whole-request cleanup used at a terminal state.
        """

        _require_identifier(request_id, "request_id")
        now = self.now_ms()
        with self._lock:
            digest, manifest = self._load_owned(scope, content_id, now=now, allow_failed=True)
            remaining = [item for item in manifest["leases"] if item["request_id"] != request_id]
            released = len(remaining) != len(manifest["leases"])
            if released:
                manifest["leases"] = remaining
                manifest["updated_at_ms"] = now
                self._files.store(digest, manifest)
            return {
                "content_id": manifest["descriptor"]["content_id"],
                "released": released,
                "cleanup": CLEANUP_PENDING,
                "remaining_leases": len(remaining),
            }

    def release(self, request_id: str, *, cleanup: bool | None = None) -> list[dict[str, Any]]:
        """Drop one request's leases and clean up what no longer has an owner.

        Returns one record per blob that was touched.  A cleanup failure leaves an
        explicit ``cleanup=quarantine`` record: the caller must treat that as an
        infrastructure failure, not as a successful cleanup.  ``cleanup=False``
        only drops the lease, which is what a rejected Submit needs: the upload
        itself is still a valid artifact the caller may reference again.
        """

        _require_identifier(request_id, "request_id")
        remove = self.limits.cleanup_on_release if cleanup is None else bool(cleanup)
        now = self.now_ms()
        records: list[dict[str, Any]] = []
        with self._lock:
            for digest in self._files.iter_blob_digests():
                manifest = self._files.load(digest)
                if manifest is None:
                    continue
                remaining = [item for item in manifest["leases"] if item["request_id"] != request_id]
                if len(remaining) == len(manifest["leases"]):
                    continue
                manifest["leases"] = remaining
                manifest["updated_at_ms"] = now
                record = {
                    "content_id": manifest["descriptor"]["content_id"],
                    "released": True,
                    "cleanup": CLEANUP_PENDING,
                    "remaining_leases": len(remaining),
                }
                if not remaining and remove:
                    record["cleanup"] = self._delete_locked(digest, manifest, now, "released")
                    if record["cleanup"] != CLEANUP_DONE:
                        record["error"] = manifest["cleanup"]["last_error"]
                else:
                    self._files.store(digest, manifest)
                records.append(record)
            return records

    def _delete_locked(self, digest: str, manifest: dict[str, Any], now: int, reason: str) -> str:
        try:
            for path in self._files.chunks(digest):
                self.io.remove(path)
            chunks_dir = self._files.directory(digest) / BLOB_CHUNKS_DIR
            if self.io.exists(chunks_dir):
                self.io.remove_tree(chunks_dir)
            if self.io.exists(self._files.manifest_path(digest)):
                self.io.remove(self._files.manifest_path(digest))
        except OSError as exc:
            manifest["state"] = STATE_QUARANTINE
            manifest["quarantine"] = {"reason": f"cleanup failed ({reason}): {exc}", "at_ms": now}
            manifest["cleanup"] = {"state": CLEANUP_QUARANTINE, "attempts": int(manifest["cleanup"]["attempts"]) + 1, "last_error": str(exc)}
            manifest["updated_at_ms"] = now
            try:
                self._files.store(digest, manifest)
            except OSError:
                pass
            return CLEANUP_QUARANTINE
        return CLEANUP_DONE

    def reconcile(self, active_request_ids: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """Release leases of requests that are neither active nor recoverable.

        The engine calls this on start (after restart) and whenever a job reaches
        a terminal state, so a crash between commit and cleanup cannot leak bytes
        forever.  A request in the active set keeps its lease.
        """

        active = set(active_request_ids)
        result: dict[str, list[dict[str, Any]]] = {}
        with self._lock:
            for request_id in sorted(self.leases()):
                if request_id in active:
                    continue
                result[request_id] = self.release(request_id)
        return result

    def sweep(self) -> list[dict[str, Any]]:
        """Delete expired uploads and unleased committed blobs.

        Returns the per-blob outcome so a caller can assert that expiry really
        happened and that a failing cleanup was reported as quarantine.
        """

        now = self.now_ms()
        records: list[dict[str, Any]] = []
        with self._lock:
            for digest in list(self._files.iter_blob_digests()):
                manifest = self._files.load(digest)
                if manifest is None:
                    continue
                if now < int(manifest["expires_at_ms"]):
                    continue
                if manifest["leases"]:
                    continue
                records.append({
                    "content_id": manifest["descriptor"]["content_id"],
                    "cleanup": self._delete_locked(digest, manifest, now, "expired"),
                    "state": manifest["state"],
                })
            return records

    def quarantine(self, content_id: str, reason: str) -> BlobState:
        """Explicitly quarantine a blob (operator/test path; never automatic)."""

        digest = _content_digest(content_id)
        now = self.now_ms()
        with self._lock:
            manifest = self._files.load(digest)
            if manifest is None:
                raise BlobError("not_found", "blob does not exist")
            manifest["state"] = STATE_QUARANTINE
            manifest["quarantine"] = {"reason": str(reason)[:512], "at_ms": now}
            manifest["updated_at_ms"] = now
            self._files.store(digest, manifest)
            return self._state(manifest)


class BlobReader:
    """Trusted worker-side reader for one content id.

    The frontend materializes nothing and hands the worker only a store root it
    owns (never a path from the agent).  The reader still re-verifies the whole
    digest while streaming, so a corrupted store cannot feed a device.
    """

    def __init__(self, root: Path | str, *, io: BlobIO | None = None) -> None:
        self._files = _BlobFiles(Path(root), io or LocalBlobIO())

    def state(self, content_id: str) -> dict[str, Any]:
        digest = _content_digest(content_id)
        manifest = self._files.load(digest)
        if manifest is None:
            raise BlobError("not_found", "blob does not exist")
        return BlobStore._state(manifest)

    def iter_chunks(self, content_id: str, *, offset: int = 0) -> Iterator[bytes]:
        digest = _content_digest(content_id)
        manifest = self._files.load(digest)
        if manifest is None:
            raise BlobError("not_found", "blob does not exist")
        if manifest["state"] != STATE_READY:
            raise BlobError("failed_precondition", f"blob is {manifest['state']}; no bytes are readable")
        yield from _iter_verified_files(self._files, digest, manifest, offset)

    def digest(self, content_id: str) -> tuple[str, int]:
        """Stream the blob and return ``(sha256, size)`` without buffering it."""

        computed = hashlib.sha256()
        size = 0
        for chunk in self.iter_chunks(content_id):
            computed.update(chunk)
            size += len(chunk)
        return computed.hexdigest(), size


def _iter_verified_files(
    files: _BlobFiles, digest: str, manifest: Mapping[str, Any], offset: int,
) -> Iterator[bytes]:
    descriptor = manifest["descriptor"]
    if offset < 0 or offset > descriptor["size_bytes"]:
        raise BlobError("out_of_range", "offset is outside the blob")
    computed = hashlib.sha256()
    total = 0
    for index in range(len(manifest["verified_chunks"])):
        path = files.chunk_path(digest, index)
        if not files.io.exists(path):
            raise BlobError("unavailable", "declared chunk is missing from the store")
        try:
            chunk = files.io.read(path, descriptor["chunk_size"] + 1)
        except OSError as exc:
            raise BlobError("unavailable", f"chunk could not be read: {exc}") from exc
        if len(chunk) > descriptor["chunk_size"]:
            raise BlobError("unavailable", "stored chunk exceeds its declared boundary")
        computed.update(chunk)
        total += len(chunk)
        if total <= offset:
            continue
        yield chunk[max(0, offset - (total - len(chunk))):]
    if total != descriptor["size_bytes"] or computed.hexdigest() != descriptor["sha256"]:
        raise BlobError("unavailable", "blob content does not verify against its descriptor")
