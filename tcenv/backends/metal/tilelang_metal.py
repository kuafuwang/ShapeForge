"""MT2 TileLang Metal builder and independent generated-artifact audit.

The candidate-owned Python DSL is lowered by the pinned frontend inside the lower-level hmz
Profile.  The emitted MSL is then screened and compiled by the same ``MetalSourceBuilder`` used
by Metal kernel pipelines, still inside the bounded v2 worker.  This module never calls the
msl-single/msl-bundle candidate contract, adapter or HostTilingPlan.

A successful build carries both identities: the DSL source/frontend/source-only identity and the
actual Apple Metal assembled-source/native identity.  ``audit_evaluator_build`` lets the control
host rebuild the closed generated-artifact set and the source-only identity from the worker's
schema-4 record before accepting the build it did not compile itself.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping

from ...remote.artifact_builder import (
    BuildArtifact,
    BuildCapabilities,
    BuildError,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
    SourceBundle,
)
from ...remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
from ...target import DEFAULT_TARGET_ID
from ...track import tilelang as contract
from ...track import tilelang_confirmation as fresh_contract
from ...track.registry import TrackSpec, load_track_registry
from .kernel_pipeline import metal_contract
from .launch import launch_math
from .screen import screen_v2_source
from .source_builder import MetalSourceBuilder
from .tilelang_build import (
    FRONTEND_ENTRY_SHA256,
    TileLangSourceOnlyBuild,
    observe_staged_environment,
    observe_staged_frontend,
)

METAL_BUILDER_ID = "tilelang-metal-track-builder"
METAL_BUILD_PROFILE = "tilelang-metal-worker-v2"
METAL_DESCRIPTOR_SCHEMA_ID = "tcenv.track.tilelang.metal-build.1"
METAL_FRESH_DESCRIPTOR_SCHEMA_ID = "tcenv.track.tilelang.metal-build.2"


class TileLangCapabilityBuildError(BuildError):
    """@brief staged frontend/hmz 缺失：typed 环境能力结果，不归咎于候选。"""

    error_code = "backend_capability_unavailable"

    def __init__(self, detail: str) -> None:
        super().__init__("toolchain_profile", detail[:800])


def _entry(path: str, media_type: str, data: bytes, *, role: str = "AUXILIARY") -> BundleEntry:
    return BundleEntry(
        path=path,
        role=role,
        media_type=media_type,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        data=bytes(data),
    )


def _candidate_entries(candidate: Any) -> tuple[BundleEntry, ...]:
    return tuple(
        _entry(entry.path, entry.media_type, bytes(entry.data)) for entry in candidate.entries
    )


def _frontend_failure_profile(
    candidate: Any, *, track_version: int = contract.TILELANG_METAL_TRACK_VERSION
) -> BuildProfile:
    """@brief frontend 在 Metal probe 前失败时的真实、有限构建环境身份。"""
    return BuildProfile(
        build_profile=METAL_BUILD_PROFILE,
        builder_id=METAL_BUILDER_ID,
        builder_version=str(track_version),
        build_host="control_host",
        toolchain_id="tilelang-metal",
        toolchain_version=contract.FRONTEND_VERSION,
        sdk_version=platform.mac_ver()[0] or "unknown",
        compile_profile={
            "language": "tilelang-python",
            "frontend": contract.FRONTEND_ID,
            "frontend_version": contract.FRONTEND_VERSION,
            "track_version": int(track_version),
            "device": "not_probed",
        },
    )


def _metal_profile(
    base: BuildProfile,
    identity: contract.SourceOnlyBuildIdentity | None = None,
    *,
    track_version: int = contract.TILELANG_METAL_TRACK_VERSION,
    candidate: Any = None,
    fresh_subject: str | None = None,
) -> BuildProfile:
    """Apple Metal probe plus the Track-versioned stable or attempt-specific subject.

    /2 preserves its historical attempt-specific BuildProfile.  /3 puts only the stable fresh
    subject in provenance; the full attempt/artifact identity remains in the native descriptor.
    """
    compile_profile = dict(base.compile_profile)
    compile_profile.update(
        {
            "frontend": contract.FRONTEND_ID,
            "frontend_version": contract.FRONTEND_VERSION,
            "frontend_options_sha256": contract.options_sha256(),
            "track_version": int(track_version),
        }
    )
    if track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
        if fresh_subject is None:
            if identity is None or candidate is None:
                raise contract.TileLangContractError("/3 Metal profile needs a stable fresh subject")
            fresh_subject = fresh_contract.expected_fresh_subject(
                candidate=candidate, environment=identity.environment
            ).cache_key
        compile_profile["tilelang_fresh_subject"] = fresh_subject
    else:
        if identity is None:
            raise contract.TileLangContractError("/2 Metal profile needs its source-only identity")
        compile_profile["tilelang_source_only_identity"] = identity.digest
    return BuildProfile(
        build_profile=METAL_BUILD_PROFILE,
        builder_id=METAL_BUILDER_ID,
        builder_version=str(track_version),
        build_host=base.build_host,
        toolchain_id=base.toolchain_id,
        toolchain_version=base.toolchain_version,
        sdk_version=base.sdk_version,
        compile_profile=compile_profile,
    )


def _descriptor(
    outcome: Any,
    *,
    track_version: int = contract.TILELANG_METAL_TRACK_VERSION,
    fresh_subject: str | None = None,
) -> dict[str, Any]:
    identity = outcome.identity
    document = {
        "schema_id": METAL_DESCRIPTOR_SCHEMA_ID,
        "track": {
            "track_id": contract.TILELANG_TRACK_ID,
            "track_version": int(track_version),
        },
        "candidate_sha256": identity.candidate_sha256,
        "dsl_source_sha256": identity.dsl_source_sha256,
        "candidate_entries": [dict(row) for row in identity.candidate_entries],
        "frontend": dict(identity.frontend),
        "options": contract.options_document(mode=contract.SOURCE_ONLY_BUILD_MODE),
        "options_sha256": contract.options_sha256(),
        "environment": dict(identity.environment),
        "attempt": dict(identity.attempt),
        "artifacts": [dict(row) for row in identity.artifacts],
        "artifact_set_sha256": identity.artifact_set_sha256,
        "cache_key": identity.cache_key,
        "source_only_identity_digest": identity.digest,
        "frontend_entry_sha256": outcome.entry_sha256,
    }
    if track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
        if not fresh_subject:
            raise contract.TileLangContractError("/3 native descriptor needs its stable fresh subject")
        document["schema_id"] = METAL_FRESH_DESCRIPTOR_SCHEMA_ID
        document["fresh_subject"] = fresh_subject
        # native_digest is a two-period confirmation field.  Attempt facts therefore cannot live
        # in this descriptor; they are preserved verbatim in BuildEvidence.log instead.
        document.pop("attempt")
        document.pop("source_only_identity_digest")
    return document


def _fresh_attempt_log(identity: contract.SourceOnlyBuildIdentity) -> bytes:
    return json.dumps(
        identity.document(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _codegen_failure(
    candidate: Any,
    exc: Exception,
    started: float,
    *,
    track_version: int = contract.TILELANG_METAL_TRACK_VERSION,
) -> BuildError:
    source = next(entry for entry in candidate.entries if entry.role == contract.SOURCE_ROLE)
    profile = _frontend_failure_profile(candidate, track_version=track_version)
    provenance = BuildProvenance(
        profile=profile,
        source_sha256=candidate.candidate_sha256,
        artifact_set_digest=None,
    )
    message = f"{type(exc).__name__}: {exc}"
    return BuildError(
        "compile",
        message[:500],
        report_detail=message[:2000],
        provenance=provenance,
        # The frontend compiler consumed these exact DSL bytes; no Metal assembly exists yet.
        assembled_sha256=hashlib.sha256(bytes(source.data)).hexdigest(),
        log=(message + "\n").encode("utf-8", "replace")[-64 * 1024 :],
        duration_ms=max(1, int(round((time.monotonic() - started) * 1000))),
    )


class TileLangMetalBuilder:
    """@brief /2 的真实 Metal builder：fenced DSL lowering + shared Apple Metal compilation。"""

    build_kind = "native"

    def __init__(
        self,
        spec: TrackSpec,
        *,
        target_id: str = DEFAULT_TARGET_ID,
        math: str | None = None,
        attempt_base: Path | None = None,
        staging: Path | None = None,
        executor: Any = None,
        limits: Any = None,
    ) -> None:
        contract.audit_registration(spec)
        if spec.track_version != contract.TILELANG_METAL_TRACK_VERSION:
            raise contract.TileLangBuildModeUnavailable(
                f"Metal build requires tilelang-pipeline/{contract.TILELANG_METAL_TRACK_VERSION}, got {spec.key}"
            )
        self.spec = spec
        # /2 uses the same spec for the outer Metal build and nested source-only lowering.
        # /3 overrides only the outer spec; its nested frontend contract remains explicitly /2.
        self.source_spec = spec
        self.target_id = target_id
        self.math = math
        self._attempt_base = attempt_base
        self._staging = staging
        self._executor = executor
        self._limits = limits

    def prepare(self, request: Any) -> None:
        """@brief 控制机只核验合同/几何/ABI；不 import frontend、不打开设备。"""
        track = request.track_selection.track
        if track.key != self.spec.key or track.document() != self.spec.document():
            raise contract.TileLangContractError("the TileLang build request is not the frozen outer TrackSpec")
        if request.candidate.track_id != self.spec.track_id:
            raise contract.TileLangContractError("the candidate does not belong to the TileLang Track")
        contract.validate_candidate(request.candidate, self.source_spec)
        geometry = request.selection.geometry
        if geometry.source != "candidate" or not geometry.active_stages:
            raise contract.TileLangContractError("TileLang /2 needs active candidate-owned execution geometry")
        metal_contract(
            task_contract=request.contract,
            abi=request.abi,
            geometry=geometry,
            target_id=self.target_id,
        )

    def build(self, request: Any) -> BuildOutcome:
        """@brief 在 bounded worker 内完成首次 frontend 与 Metal 编译。"""
        self.prepare(request)
        candidate = request.candidate
        geometry = request.selection.geometry
        kernel_names = tuple(dict.fromkeys(stage.kernel for stage in geometry.active_stages))
        started = time.monotonic()
        try:
            source_build = TileLangSourceOnlyBuild(
                self.source_spec,
                attempt_base=self._attempt_base,
                staging=self._staging,
                executor=self._executor,
                limits=self._limits,
            ).build(candidate, expected_kernel_names=kernel_names)
        except contract.TileLangFrontendUnavailable as exc:
            raise TileLangCapabilityBuildError(str(exc)) from exc
        except contract.TileLangIdentityDrift as exc:
            raise BuildError("toolchain_profile", f"the pinned TileLang frontend drifted: {exc}") from exc
        except (contract.TileLangCodegenError, contract.TileLangContractError) as exc:
            raise _codegen_failure(candidate, exc, started, track_version=self.spec.track_version) from exc

        artifact_dir = source_build.attempt_root / "build" / "artifacts"
        generated: dict[str, bytes] = {}
        for name in contract.SOURCE_ONLY_ARTIFACT_PATHS:
            path = artifact_dir / name
            if not path.is_file() or path.is_symlink():
                raise BuildError("toolchain_profile", f"the verified source-only artifact is missing: {name}")
            generated[name] = path.read_bytes()
        try:
            kernel_text = generated["kernel.metal"].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _codegen_failure(candidate, exc, started, track_version=self.spec.track_version) from exc
        refused = screen_v2_source(kernel_text)
        if refused:
            raise _codegen_failure(
                candidate,
                contract.TileLangContractError("generated MSL refused before Metal compilation: " + refused[0]),
                started,
                track_version=self.spec.track_version,
            )
        task_contract = metal_contract(
            task_contract=request.contract,
            abi=request.abi,
            geometry=geometry,
            target_id=self.target_id,
        )
        emitted_math = launch_math(kernel_text)
        if (
            self.spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION
            and emitted_math not in (None, "safe")
        ):
            raise _codegen_failure(
                candidate,
                contract.TileLangContractError(
                    f"/3 freezes Metal math='safe'; generated MSL requests {emitted_math!r}"
                ),
                started,
                track_version=self.spec.track_version,
            )
        math = self.math or (
            "safe"
            if self.spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION
            else emitted_math
        ) or "safe"
        fresh_subject = (
            fresh_contract.expected_fresh_subject(
                candidate=candidate, environment=source_build.identity.environment
            ).cache_key
            if self.spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION
            else None
        )
        builder = MetalSourceBuilder(
            kernel_contract=task_contract,
            math=math,
            kernel_names=kernel_names,
        )
        source_bundle = SourceBundle(
            generated["kernel.metal"],
            media_type=contract.SOURCE_ONLY_ARTIFACT_MEDIA_TYPES["kernel.metal"],
            path="kernel.metal",
        )
        try:
            compiled = builder.build(source_bundle, task_contract=task_contract)
        except BuildError as exc:
            if exc.error_class != "compile":
                raise
            profile = _metal_profile(
                exc.provenance.profile,
                source_build.identity,
                track_version=self.spec.track_version,
                candidate=candidate,
                fresh_subject=fresh_subject,
            )
            raise BuildError(
                "compile",
                exc.detail,
                report_detail=exc.report_detail,
                provenance=BuildProvenance(
                    profile=profile,
                    source_sha256=candidate.candidate_sha256,
                    artifact_set_digest=None,
                ),
                assembled_sha256=exc.assembled_sha256,
                log=exc.log,
                duration_ms=exc.duration_ms,
            ) from exc

        native = compiled.bundle.native
        if native is None:
            raise BuildError("toolchain_profile", "the Apple Metal builder returned no native executable")
        descriptor = dict(native.descriptor)
        descriptor["tilelang"] = _descriptor(
            source_build, track_version=self.spec.track_version, fresh_subject=fresh_subject
        )
        new_native = NativeExecutable(
            backend_id=native.backend_id,
            kind=native.kind,
            entry_path="kernel.metal",
            entry_sha256=native.entry_sha256,
            descriptor=descriptor,
            handle=native.handle,
        )
        entries = [
            *_candidate_entries(candidate),
            *(
                _entry(path, contract.SOURCE_ONLY_ARTIFACT_MEDIA_TYPES[path], generated[path])
                for path in contract.SOURCE_ONLY_ARTIFACT_PATHS
            ),
        ]
        bundle = ArtifactBundle(entries=tuple(entries), native=new_native)
        profile = _metal_profile(
            compiled.provenance.profile,
            source_build.identity,
            track_version=self.spec.track_version,
            candidate=candidate,
            fresh_subject=fresh_subject,
        )
        provenance = BuildProvenance(
            profile=profile,
            source_sha256=candidate.candidate_sha256,
            artifact_set_digest=bundle.artifact_set_digest,
        )
        artifacts = tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries)
        elapsed_ms = max(1, int(round((time.monotonic() - started) * 1000)))
        log = (
            f"tilelang: source_only_identity={source_build.identity.digest} "
            f"artifact_set={source_build.identity.artifact_set_sha256} "
            f"frontend_entry={source_build.entry_sha256}\n"
        ).encode("utf-8")
        if self.spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
            log += b"tilelang_attempt_evidence=" + _fresh_attempt_log(source_build.identity) + b"\n"
        log += compiled.evidence.log
        evidence = BuildEvidence(
            mode="prebuilt",
            status="ok",
            provenance=provenance,
            artifacts=artifacts,
            duration_ms=elapsed_ms,
            log=log,
            native=new_native,
            placement="control_host",
        )
        return BuildOutcome(
            bundle=bundle,
            provenance=provenance,
            evidence=evidence,
            payloads={},
            assembled_sha256=compiled.assembled_sha256,
        )

    def audit_evaluator_build(
        self,
        *,
        entries: list[Mapping[str, Any]],
        candidate_entries: list[Mapping[str, Any]],
        manifest_entry: Mapping[str, Any],
        geometry_entry: Mapping[str, Any],
        native: Mapping[str, Any],
        canonical_bundle: Any,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """@brief 独立核对 worker 生成物闭集、frontend/source-only 身份与 native entry。"""
        from ...remote.execution_geometry import EvaluatorBuildError

        by_path: dict[str, Mapping[str, Any]] = {}
        for entry in entries:
            by_path.setdefault(str(entry["path"]), entry)
        generated_rows: list[dict[str, Any]] = []
        generated_entries: list[dict[str, Any]] = []
        for path in contract.SOURCE_ONLY_ARTIFACT_PATHS:
            entry = by_path.get(path)
            expected_media = contract.SOURCE_ONLY_ARTIFACT_MEDIA_TYPES[path]
            if entry is None or entry.get("role") != "AUXILIARY" or entry.get("media_type") != expected_media:
                raise EvaluatorBuildError(f"the TileLang generated artifact {path!r} is missing or mistyped")
            generated_entries.append(dict(entry))
            generated_rows.append(
                {"path": path, "bytes": int(entry["size_bytes"]), "sha256": str(entry["sha256"])}
            )
        expected_entries = sorted(
            [*candidate_entries, manifest_entry, geometry_entry, *generated_entries],
            key=lambda item: item["path"],
        )
        document = native.get("descriptor", {}).get("tilelang")
        if not isinstance(document, Mapping):
            raise EvaluatorBuildError("the TileLang native descriptor has no frontend/build identity")
        candidate = canonical_bundle.candidate
        try:
            expected_fields = {
                "schema_id", "track", "candidate_sha256", "dsl_source_sha256", "candidate_entries",
                "frontend", "options", "options_sha256", "environment", "attempt", "artifacts",
                "artifact_set_sha256", "cache_key", "source_only_identity_digest", "frontend_entry_sha256",
            }
            expected_schema = METAL_DESCRIPTOR_SCHEMA_ID
            if self.spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
                expected_fields.difference_update({"attempt", "source_only_identity_digest"})
                expected_fields.add("fresh_subject")
                expected_schema = METAL_FRESH_DESCRIPTOR_SCHEMA_ID
            if set(document) != expected_fields or document["schema_id"] != expected_schema:
                raise EvaluatorBuildError("the TileLang native descriptor fields are not the closed Metal-build schema")
            if document["track"] != {
                "track_id": contract.TILELANG_TRACK_ID,
                "track_version": self.spec.track_version,
            }:
                raise EvaluatorBuildError("the TileLang native descriptor is for another Track version")
            if document["candidate_sha256"] != candidate.candidate_sha256:
                raise EvaluatorBuildError("the TileLang native descriptor belongs to another candidate")
            source = next(entry for entry in candidate.entries if entry.role == contract.SOURCE_ROLE)
            if document["dsl_source_sha256"] != source.sha256:
                raise EvaluatorBuildError("the TileLang native descriptor binds another DSL source")
            manifest_entries = [dict(row) for row in candidate.manifest()["entries"]]
            if [dict(row) for row in document["candidate_entries"]] != manifest_entries:
                raise EvaluatorBuildError("the TileLang native descriptor candidate manifest drifted")
            frontend = contract.verify_frontend_observation(document["frontend"])
            if dict(document["frontend"]) != frontend:
                raise EvaluatorBuildError("the TileLang native descriptor frontend is not canonical")
            if document["options"] != contract.options_document() or document["options_sha256"] != contract.options_sha256():
                raise EvaluatorBuildError("the TileLang frontend options drifted")
            environment = document["environment"]
            expected = contract.expected_build_subject(
                candidate=candidate,
                environment=environment,
                track_version=self.source_spec.track_version,
            )
            if document["cache_key"] != expected.cache_key:
                raise EvaluatorBuildError("the TileLang source-only cache key does not match the control-host subject")
            if document["artifacts"] != generated_rows:
                raise EvaluatorBuildError("the TileLang native descriptor artifact rows drifted")
            artifact_digest = contract.artifact_set_sha256(generated_rows)
            if document["artifact_set_sha256"] != artifact_digest:
                raise EvaluatorBuildError("the TileLang source-only artifact set digest drifted")
            if document["frontend_entry_sha256"] != FRONTEND_ENTRY_SHA256:
                raise EvaluatorBuildError("the harness-owned TileLang frontend entry drifted")
            if self.spec.track_version == contract.TILELANG_METAL_TRACK_VERSION:
                attempt = contract.verify_attempt(document["attempt"])
                rebuilt = contract.SourceOnlyBuildIdentity(
                    candidate_sha256=candidate.candidate_sha256,
                    dsl_source_sha256=source.sha256,
                    candidate_entries=tuple(manifest_entries),
                    frontend=frontend,
                    environment=dict(environment),
                    attempt=attempt,
                    artifacts=tuple(generated_rows),
                    artifact_set_sha256_value=artifact_digest,
                    cache_key_value=expected.cache_key,
                    full_build_value=True,
                    track_version=self.source_spec.track_version,
                )
                if document["source_only_identity_digest"] != rebuilt.digest:
                    raise EvaluatorBuildError("the TileLang source-only identity digest is not independently reproducible")
            if self.spec.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
                fresh_subject = fresh_contract.expected_fresh_subject(
                    candidate=candidate, environment=environment
                ).cache_key
                if document["fresh_subject"] != fresh_subject:
                    raise EvaluatorBuildError("the TileLang fresh subject is not independently reproducible")
        except EvaluatorBuildError:
            raise
        except Exception as exc:
            raise EvaluatorBuildError(f"the TileLang native descriptor is invalid: {exc}") from exc
        return expected_entries, {"kernel.metal"}


class TileLangFreshMetalBuilder(TileLangMetalBuilder):
    """Unregistered /3 builder with fresh-stable provenance and an explicit /2 lowering step."""

    def __init__(
        self,
        spec: TrackSpec,
        *,
        target_id: str = DEFAULT_TARGET_ID,
        math: str | None = None,
        attempt_base: Path | None = None,
        staging: Path | None = None,
        executor: Any = None,
        limits: Any = None,
    ) -> None:
        source_spec = load_track_registry().get(
            contract.TILELANG_TRACK_ID, version=contract.TILELANG_METAL_TRACK_VERSION
        )
        fresh_contract.audit_fresh_registration(spec, source_track=source_spec)
        if math not in (None, "safe"):
            raise ValueError("TileLang /3 freezes Metal math='safe' so probe and build identities agree")
        super().__init__(
            source_spec,
            target_id=target_id,
            math=math,
            attempt_base=attempt_base,
            staging=staging,
            executor=executor,
            limits=limits,
        )
        self.spec = spec
        self.source_spec = source_spec

    def probe(self, candidate: Any, geometry: Any) -> BuildCapabilities:
        """Rebuild the /3 expected profile without importing or executing the candidate.

        The staged frontend is observed and hashed, but no source-only lowering occurs here.  The
        real candidate build still runs later inside the bounded worker.  The Metal probe itself
        uses only the fixed safe-math toolchain configuration.
        """
        contract.validate_candidate(candidate, self.source_spec)
        if geometry.source != "candidate" or not geometry.active_stages:
            raise contract.TileLangContractError("/3 probe needs candidate-owned active execution geometry")
        frontend = observe_staged_frontend(self._staging)
        environment = observe_staged_environment(frontend)
        subject = fresh_contract.expected_fresh_subject(
            candidate=candidate, environment=environment
        ).cache_key
        names = tuple(dict.fromkeys(stage.kernel for stage in geometry.active_stages))
        capabilities = MetalSourceBuilder(
            kernel_contract=None, math="safe", kernel_names=names
        ).probe()
        if not capabilities.available:
            return capabilities
        return BuildCapabilities(
            profile=_metal_profile(
                capabilities.profile,
                track_version=self.spec.track_version,
                fresh_subject=subject,
            ),
            available=True,
            detail=capabilities.detail,
        )


__all__ = [
    "METAL_BUILDER_ID",
    "METAL_BUILD_PROFILE",
    "METAL_DESCRIPTOR_SCHEMA_ID",
    "METAL_FRESH_DESCRIPTOR_SCHEMA_ID",
    "TileLangCapabilityBuildError",
    "TileLangFreshMetalBuilder",
    "TileLangMetalBuilder",
]
