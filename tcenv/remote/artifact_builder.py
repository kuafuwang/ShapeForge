"""Unified ``ArtifactBuilder``: source -> executable artifact set.

This is the Backend/task-semantics layer that turns a candidate source into the
executable artifact set a device can load.  It deliberately knows nothing about
the content-addressed blob channel, the vendor driver or the transport: a
builder produces an :class:`ArtifactBundle` plus a
:class:`BuildProvenance`/``BuildEvidence`` pair, and placement decides *where*
the build runs:

* ``prebuilt`` — the control host builds before Submit; the evaluator only
  verifies and loads the artifact set (it needs no compiler, git or LLM
  credentials);
* ``evaluator`` — the attempt builds the source it received and loads the
  result in the same attempt.

Both timings share one ``BuildOutcome`` shape, one error classification and one
canonical provenance/evidence contract.  A builder may only be selected by a
*registered* build profile: nothing here accepts a shell command, a host path or
an arbitrary build tool.

The third placement — build and execute on the control host, as Metal does — has
no portable artifact set to transport.  Its executable is expressed as a
backend-native ``NativeExecutable`` (a live handle plus a canonical descriptor
and digest) carried on the local-only ``ArtifactBundle``; such a bundle refuses
every request/worker document, so a local pipeline can never be smuggled into a
remote artifact set.  ``artifact_set_digest`` keeps describing only the manifest
and bytes in all three placements.  The provenance is independent of
``artifact_set_digest`` (which keeps describing only the artifact manifest and
bytes), so a changed build environment is visible as a different provenance
instead of silently reusing an old measurement baseline.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from . import contracts
from .artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable, payload_bytes
from .contracts import canonical_digest_document

#: The shared error classification.  A typed ``compile`` failure is a candidate
#: failure and becomes a ``compile_error`` report; ``toolchain_profile`` is
#: refused before Submit/create/start; ``deadline`` means the build did not fit
#: its budget.  An untyped builder crash is infrastructure, never a compile
#: failure, and is therefore not mapped into this classification.
BUILD_ERROR_CLASSES = contracts.BUILD_ERROR_CLASSES


class BuildError(ValueError):
    """A typed build outcome that is not a successful artifact set."""

    def __init__(self, error_class: str, detail: str, *, log: bytes = b"", duration_ms: int = 0) -> None:
        if error_class not in BUILD_ERROR_CLASSES:
            raise ValueError(f"unknown build error class {error_class!r}")
        super().__init__(detail)
        self.error_class = error_class
        self.detail = detail
        self.log = bytes(log)
        self.duration_ms = int(duration_ms)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.error_class}: {self.detail}"


@dataclass(frozen=True)
class BuildProfile:
    """One registered build profile and the toolchain identity it pins.

    The registry — not a request — decides which profiles exist.  The fields are
    a closed identity: builder id/version, the host that builds, the profile
    name, the toolchain/SDK versions and the bounded compile options.
    """

    build_profile: str
    builder_id: str
    builder_version: str
    build_host: str
    toolchain_id: str
    toolchain_version: str
    sdk_version: str
    compile_profile: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.build_host not in set(contracts.BUILD_HOSTS.values()):
            raise ValueError("build profile build_host must be control_host or evaluator")
        # Validate the identity through the same contract later used on the wire.
        contracts.build_provenance_document(self.provenance_document(
            source_sha256="0" * 64, artifact_set_digest=None,
        ))

    @property
    def toolchain(self) -> dict[str, str]:
        return {
            "id": self.toolchain_id,
            "version": self.toolchain_version,
            "sdk_version": self.sdk_version,
        }

    def provenance_document(self, *, source_sha256: str, artifact_set_digest: str | None) -> dict[str, Any]:
        return {
            "schema_id": contracts.NPU_BUILD_PROVENANCE_SCHEMA_ID,
            "builder_id": self.builder_id,
            "builder_version": self.builder_version,
            "build_host": self.build_host,
            "build_profile": self.build_profile,
            "toolchain": self.toolchain,
            "compile_profile": dict(self.compile_profile),
            "source_sha256": source_sha256,
            "artifact_set_digest": artifact_set_digest,
        }

    def capability_document(self) -> dict[str, Any]:
        """The identity a probe advertises (no source or artifact digest)."""

        return {
            "builder_id": self.builder_id,
            "builder_version": self.builder_version,
            "build_host": self.build_host,
            "build_profile": self.build_profile,
            "toolchain": self.toolchain,
            "compile_profile": dict(self.compile_profile),
        }


@dataclass(frozen=True)
class BuildCapabilities:
    """What a builder probe reports; ``available`` gates Submit."""

    profile: BuildProfile
    available: bool = True
    detail: str = ""

    @property
    def build_profile(self) -> str:
        return self.profile.build_profile

    def identity(self) -> dict[str, Any]:
        return self.profile.capability_document()

    def matches(self, provenance: Mapping[str, Any]) -> str:
        """Return why ``provenance`` is not this profile (empty = it matches)."""

        try:
            document = contracts.build_provenance_document(provenance)
        except contracts.ContractError as exc:
            return f"build provenance is invalid: {exc}"
        expected = self.profile.capability_document()
        for name, value in expected.items():
            if document[name] != value:
                return f"build provenance {name} differs from the registered profile"
        return ""


@dataclass(frozen=True)
class BuildProvenance:
    """The provenance one build produced (or the identity it must produce)."""

    profile: BuildProfile
    source_sha256: str
    artifact_set_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        try:
            return contracts.build_provenance_document(self.profile.provenance_document(
                source_sha256=self.source_sha256,
                artifact_set_digest=self.artifact_set_digest,
            ))
        except contracts.ContractError as exc:
            raise BuildError("toolchain_profile", f"build provenance is invalid: {exc}") from exc

    def request_document(self, *, mode: str) -> dict[str, Any]:
        return {
            "schema_id": contracts.NPU_BUILD_REQUEST_SCHEMA_ID,
            "mode": mode,
            "provenance": self.to_dict(),
        }


@dataclass(frozen=True)
class BuildArtifact:
    """One produced artifact: the manifest entry without any bytes."""

    path: str
    role: str
    media_type: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_entry(cls, entry: BundleEntry) -> "BuildArtifact":
        return cls(
            path=entry.path, role=entry.role, media_type=entry.media_type,
            size_bytes=entry.size_bytes, sha256=entry.sha256,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "path": self.path, "role": self.role, "media_type": self.media_type,
            "size_bytes": self.size_bytes, "sha256": self.sha256,
        }


@dataclass(frozen=True)
class BuildEvidence:
    """One build outcome as the signed evidence document describes it.

    ``reused`` is a prebuilt artifact set that this attempt verified and loaded;
    ``ok`` is this attempt's own successful build; ``compile_failed`` is a typed
    candidate compile failure; ``not_run`` is a diagnostic row (the attempt was
    terminated before the build).  A typed value models a producer's successful
    outcome (``ok``) or a received prebuilt set (``reused``/``not_run``); the
    signed ``compile_failed`` row is a frontend/worker document validated by the
    same contract, not a ``BuildOutcome``.

    ``placement`` decides which of those combinations is coherent: a control-host
    producer may report ``prebuilt``/``ok`` because it really compiled the set,
    while a *received* prebuilt row must be ``reused`` and repeat the producer's
    build-evidence link.  The remote NPU rule is never relaxed by the producer
    placement.

    ``native`` records a backend-native executable that was produced and kept in
    this process (the Metal pipeline).  It never enters the wire document: the
    NPU contract only accepts a portable artifact set, so ``to_dict`` stays the
    exact shared shape and ``native_document`` is the local record.
    """

    mode: str
    status: str
    provenance: BuildProvenance
    artifacts: tuple[BuildArtifact, ...] = ()
    error_class: str | None = None
    detail: str = ""
    duration_ms: int = 0
    log: bytes = b""
    loaded: tuple[tuple[str, str], ...] = ()
    native: NativeExecutable | None = None
    #: Where this row was produced: ``control_host`` (fresh producer build),
    #: ``evaluator`` (fresh build in an attempt) or ``received`` (a prebuilt
    #: artifact set this attempt verified and loaded).  It is neutral model
    #: semantics, not a wire field: the remote validator derives the same
    #: placement from the accepted request's mode.
    placement: str = ""
    #: The producer build-evidence link a ``received`` row repeats.  It is the
    #: control host's own record (status/duration/log digest/evidence digest),
    #: authenticated by the request digest; the NPU never receives the log bytes.
    producer: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode not in contracts.BUILD_MODES:
            raise ValueError("build evidence mode must be prebuilt or evaluator")
        if self.status not in contracts.BUILD_STATUSES:
            raise ValueError("build evidence status is not a supported build outcome")
        if self.error_class is not None and self.error_class not in BUILD_ERROR_CLASSES:
            raise ValueError("build evidence error class is not supported")
        if self.duration_ms < 0:
            raise ValueError("build evidence duration must not be negative")
        if self.status in {"reused", "ok"}:
            if not self.artifacts:
                raise ValueError("a successful build must list its artifacts")
        elif self.artifacts:
            raise ValueError("a failed or unrun build must not claim artifacts")
        placement = self.placement
        if not placement:
            placement = (
                "received" if self.mode == "prebuilt" and self.status in {"reused", "not_run"}
                else "control_host" if self.mode == "prebuilt"
                else "evaluator"
            )
            object.__setattr__(self, "placement", placement)
        if placement not in contracts.BUILD_PLACEMENTS:
            raise ValueError("build evidence placement is not supported")
        if placement == "control_host":
            if self.mode != "prebuilt" or self.status != "ok":
                raise ValueError(
                    "a control-host producer build must be a fresh prebuilt compile (ok)"
                )
        elif placement == "evaluator":
            if self.mode != "evaluator" or self.status != "ok":
                raise ValueError(
                    "an evaluator producer build must be a fresh evaluator compile (ok)"
                )
        else:
            if self.mode != "prebuilt" or self.status not in {"reused", "not_run"}:
                raise ValueError(
                    "a received placement only describes a reused (or not_run) prebuilt set"
                )
        if placement == "received" and self.status == "reused":
            if self.producer is None:
                raise ValueError("a received prebuilt artifact requires the producer link")
            try:
                normalized = contracts.build_producer_record(self.producer)
            except contracts.ContractError as exc:
                raise ValueError(f"producer build-evidence link is invalid: {exc}") from exc
            object.__setattr__(self, "producer", normalized)
        elif self.producer is not None:
            raise ValueError("only a received prebuilt artifact repeats a producer link")
        if self.native is not None:
            if not isinstance(self.native, NativeExecutable):
                raise ValueError("build evidence native executable must be typed")
            if self.status != "ok":
                # A live handle can only come from a build that ran here.
                raise ValueError("a backend-native executable requires a fresh successful build")
            if not any(
                artifact.path == self.native.entry_path
                and artifact.sha256 == self.native.entry_sha256
                for artifact in self.artifacts
            ):
                raise ValueError("the backend-native executable is not built from a listed artifact")

    def native_document(self) -> dict[str, Any] | None:
        """The local (never transmitted) backend-native executable record."""

        if self.native is None:
            return None
        return {**self.native.document(), "digest": self.native.digest}

    def producer_record(self) -> dict[str, Any]:
        """The producer link a prebuilt request carries (control-host build).

        It binds the *whole* producer evidence document by digest and repeats the
        status, duration and log identity.  A received row must repeat this
        record unchanged, which is what closes the original build record over
        the signed result without shipping the log bytes to the NPU.
        """

        if self.placement != "control_host" or self.status != "ok":
            raise ValueError("only a successful control-host producer build has a producer link")
        document = self.to_dict()
        try:
            contracts.build_producer_evidence_document(document, placement="control_host")
        except contracts.ContractError as exc:
            raise ValueError(f"producer build evidence is invalid: {exc}") from exc
        record = {
            "schema_id": contracts.NPU_BUILD_PRODUCER_RECORD_SCHEMA_ID,
            "status": self.status,
            "mode": self.mode,
            "duration_ms": self.duration_ms,
            "log": document["log"],
            "evidence_sha256": hashlib.sha256(canonical_digest_document(document)).hexdigest(),
        }
        try:
            return contracts.build_producer_record(record)
        except contracts.ContractError as exc:  # pragma: no cover - defensive
            raise ValueError(f"producer build-evidence link is invalid: {exc}") from exc

    def matches_producer_record(self, record: Mapping[str, Any]) -> bool:
        """True only for the link this exact producer evidence would emit."""

        try:
            return contracts.build_producer_record(record) == self.producer_record()
        except (ValueError, contracts.ContractError):
            return False

    def artifacts_document(self) -> dict[str, Any] | None:
        if not self.artifacts:
            return None
        from .artifact_bundle import BUNDLE_SCHEMA_ID, artifact_set_digest as manifest_digest

        entries = [artifact.manifest() for artifact in self.artifacts]
        return {
            "schema_id": BUNDLE_SCHEMA_ID,
            "entries": entries,
            "artifact_set_digest": manifest_digest(BUNDLE_SCHEMA_ID, entries),
        }

    def to_dict(self) -> dict[str, Any]:
        """The raw BuildEvidence document (validated by :meth:`verified`)."""

        artifacts = self.artifacts_document()
        document = {
            "schema_id": contracts.NPU_BUILD_EVIDENCE_SCHEMA_ID,
            "mode": self.mode,
            "status": self.status,
            "error_class": self.error_class,
            "provenance": self.provenance.to_dict(),
            "artifacts": artifacts,
            "duration_ms": self.duration_ms,
            "log": (
                None if not self.log
                else {
                    "sha256": hashlib.sha256(self.log).hexdigest(),
                    "size_bytes": len(self.log),
                }
            ),
            "loaded": [{"path": path, "sha256": digest} for path, digest in self.loaded],
        }
        if self.producer is not None:
            document["producer"] = dict(self.producer)
        return document

    def verified(
        self, *, expected_build: Mapping[str, Any] | None = None,
        request_artifact: Mapping[str, Any] | None = None,
        report_status: str | None = None, complete: bool | None = None,
    ) -> dict[str, Any]:
        """Validate this row under its own placement and return the document.

        A producer row (control host or evaluator) is validated with the local
        producer rule, which is what makes ``prebuilt``/``ok`` coherent for the
        machine that compiled it.  A received row is validated with the remote
        NPU rule (prebuilt must be ``reused`` and repeat the producer link).
        """

        try:
            if self.placement in ("control_host", "evaluator"):
                if expected_build is not None or request_artifact is not None:
                    raise contracts.ContractError(
                        "a producer build evidence is not a received artifact receipt"
                    )
                return contracts.build_producer_evidence_document(
                    self.to_dict(), placement=self.placement,
                )
            return contracts.npu_build_evidence_document(
                self.to_dict(), expected_build=expected_build,
                request_artifact=request_artifact, report_status=report_status, complete=complete,
            )
        except contracts.ContractError as exc:
            raise BuildError("toolchain_profile", f"build evidence is invalid: {exc}") from exc


def make_blob_entry(
    *, path: str, role: str, media_type: str, data: bytes, chunk_size: int,
) -> BundleEntry:
    """A content-addressed bundle entry for an artifact above the inline limit.

    The builder knows the exact bytes, so it can state the content identity
    without touching the channel: ``content_id``/``sha256``/``size``/media type
    describe the bytes and the transport uploads them.  Submitting before the
    upload commits fails closed in the engine (the artifact is not ready), so a
    descriptor is never mistaken for a delivered artifact.
    """

    from .artifact_bundle import MAX_BUNDLE_INLINE_ENTRY_BYTES

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("a blob-backed artifact requires bytes")
    raw = bytes(data)
    if len(raw) <= MAX_BUNDLE_INLINE_ENTRY_BYTES:
        raise ValueError("an artifact within the inline limit must not be blob-referenced")
    digest = hashlib.sha256(raw).hexdigest()
    return BundleEntry(
        path=path, role=role, media_type=media_type, size_bytes=len(raw), sha256=digest,
        blob_ref={
            "content_id": contracts.blob_content_id(digest),
            "sha256": digest,
            "size_bytes": len(raw),
            "media_type": media_type,
            "chunk_size": int(chunk_size),
        },
    )


@dataclass(frozen=True)
class BuildOutcome:
    """One successful build: the artifact set, its provenance and its evidence.

    ``payloads`` carries the exact bytes of blob-backed entries so the transport
    layer can upload them; it is never the transport protocol itself.
    """

    bundle: ArtifactBundle
    provenance: BuildProvenance
    evidence: BuildEvidence
    payloads: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, ArtifactBundle):
            raise ValueError("a build outcome requires a typed ArtifactBundle")
        if not isinstance(self.provenance, BuildProvenance):
            raise ValueError("a build outcome requires typed provenance")
        if not isinstance(self.evidence, BuildEvidence):
            raise ValueError("a build outcome requires typed evidence")
        if self.evidence.native != self.bundle.native:
            raise ValueError(
                "the build evidence native executable is not the artifact set's executable"
            )
        if self.evidence.native is not None and (
            self.evidence.native.handle is not self.bundle.native.handle
        ):
            # The descriptor equality above cannot see the live object: the
            # handle is excluded from equality and from every digest (it is a
            # process-local capability, not a document field).  Inside one
            # outcome the two records must still name the *same* live
            # executable, otherwise the bundle would carry one pipeline and the
            # evidence another while both claim the same build.
            raise ValueError(
                "the build evidence native executable is not the artifact set's live executable"
            )
        self.bundle.verify_bytes()
        if self.provenance.artifact_set_digest != self.bundle.artifact_set_digest:
            raise ValueError("build provenance artifact_set_digest does not bind the artifact set")
        manifest = [entry.manifest() for entry in self.bundle.entries]
        produced = [artifact.manifest() for artifact in self.evidence.artifacts]
        if produced and produced != manifest:
            raise ValueError("build evidence artifacts are not the built artifact set")
        if self.evidence.provenance.to_dict() != self.provenance.to_dict():
            raise ValueError("build evidence provenance differs from the build outcome provenance")
        payloads = dict(self.payloads)
        object.__setattr__(self, "payloads", payloads)
        for entry in self.bundle.entries:
            data = payloads.get(entry.path)
            if entry.blob_ref is None:
                if data is not None:
                    raise ValueError("an inline artifact does not need a transport payload")
                continue
            if data is None:
                raise ValueError(f"a blob-backed artifact {entry.path!r} requires its exact bytes")
            if len(data) != entry.size_bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
                raise ValueError(f"a blob-backed artifact {entry.path!r} does not match its descriptor")
        if set(payloads) - {entry.path for entry in self.bundle.entries}:
            raise ValueError("a transport payload names an artifact the build did not produce")

    @property
    def artifact_set_digest(self) -> str:
        return self.bundle.artifact_set_digest

    def payload(self) -> bytes:
        """The exact persisted bytes of the portable artifact set container.

        A local-only artifact set (a backend-native executable) has no portable
        container and therefore no transport payload at all.
        """

        return payload_bytes(self.bundle)


@dataclass(frozen=True)
class SourceBundle:
    """The exact source a build consumes (never a host path)."""

    source: bytes
    media_type: str = "text/plain; charset=utf-8"
    path: str = "kernel.tx"
    role: str = "SOURCE"

    def __post_init__(self) -> None:
        if not isinstance(self.source, (bytes, bytearray, memoryview)):
            raise ValueError("source bundle requires bytes")
        object.__setattr__(self, "source", bytes(self.source))
        if not self.source:
            raise ValueError("source bundle must not be empty")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.source).hexdigest()

    def entry(self) -> BundleEntry:
        return BundleEntry(
            path=self.path, role=self.role, media_type=self.media_type,
            size_bytes=len(self.source), sha256=self.sha256, data=self.source,
        )


@runtime_checkable
class ArtifactBuilder(Protocol):
    """One registered builder (probe + deterministic build)."""

    def probe(self) -> BuildCapabilities: ...

    def build(
        self, source_bundle: SourceBundle, *, task_contract: Any,
        deadline_monotonic: float | None = None,
    ) -> BuildOutcome: ...


class ArtifactBuilderRegistry:
    """The registered builders a frontend may use, keyed by build profile.

    Registration is composition-root configuration: a request can name a
    profile, but it can never add one, supply a command or point at a host path.
    ``probe`` is re-run on demand so a toolchain that disappeared is visible as an
    unavailable profile instead of a stale capability.
    """

    def __init__(self, builders: Iterable[ArtifactBuilder] = ()) -> None:
        self._builders: dict[str, ArtifactBuilder] = {}
        for builder in builders:
            capabilities = builder.probe()
            profile = capabilities.build_profile
            if profile in self._builders:
                raise ValueError(f"build profile {profile!r} is registered twice")
            self._builders[profile] = builder

    def __bool__(self) -> bool:
        return bool(self._builders)

    @property
    def profiles(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def capabilities(self) -> tuple[BuildCapabilities, ...]:
        return tuple(builder.probe() for _profile, builder in sorted(self._builders.items()))

    def require_available(self) -> None:
        if not self._builders:
            raise BuildError("toolchain_profile", "no artifact builder is registered on this frontend")
        unavailable = [
            f"{capabilities.build_profile}: {capabilities.detail or 'unavailable'}"
            for capabilities in self.capabilities() if not capabilities.available
        ]
        if unavailable:
            raise BuildError("toolchain_profile", "; ".join(unavailable)[:400])

    def builder_for(self, provenance: Mapping[str, Any]) -> ArtifactBuilder:
        """Return the registered builder for this provenance or fail closed."""

        try:
            document = contracts.build_provenance_document(provenance)
        except contracts.ContractError as exc:
            raise BuildError("toolchain_profile", f"build provenance is invalid: {exc}") from exc
        builder = self._builders.get(document["build_profile"])
        if builder is None:
            raise BuildError(
                "toolchain_profile",
                f"build profile {document['build_profile']!r} is not registered on this frontend",
            )
        capabilities = builder.probe()
        if not capabilities.available:
            raise BuildError(
                "toolchain_profile",
                f"build profile {document['build_profile']!r} is unavailable: {capabilities.detail}",
            )
        mismatch = capabilities.matches(document)
        if mismatch:
            raise BuildError("toolchain_profile", mismatch)
        return builder

    def build(
        self, *, source_bundle: SourceBundle, task_contract: Mapping[str, Any],
        provenance: Mapping[str, Any], deadline_monotonic: float | None = None,
    ) -> BuildOutcome:
        """Build through the registered profile and check the outcome identity."""

        builder = self.builder_for(provenance)
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise BuildError("deadline", "the build deadline expired before the build started")
        outcome = builder.build(
            source_bundle, task_contract=task_contract, deadline_monotonic=deadline_monotonic,
        )
        expected = contracts.build_provenance_document(provenance)
        produced = outcome.provenance.to_dict()
        for name in (
            "builder_id", "builder_version", "build_host", "build_profile",
            "toolchain", "compile_profile", "source_sha256",
        ):
            if produced[name] != expected[name]:
                raise BuildError(
                    "toolchain_profile",
                    f"the builder produced a different {name} than the registered profile",
                )
        if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
            raise BuildError("deadline", "the build did not finish before the attempt deadline")
        return outcome


def bundle_document(bundle: ArtifactBundle) -> dict[str, Any]:
    """The content-free request document for one artifact set."""

    return bundle.to_dict()


__all__ = [
    "ArtifactBuilder",
    "ArtifactBuilderRegistry",
    "BUILD_ERROR_CLASSES",
    "BuildArtifact",
    "BuildCapabilities",
    "BuildError",
    "BuildEvidence",
    "BuildOutcome",
    "BuildProfile",
    "BuildProvenance",
    "SourceBundle",
    "make_blob_entry",
]
