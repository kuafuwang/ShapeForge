"""MT6 /3 builder/probe CPU tests; no Metal device or frontend execution."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from types import SimpleNamespace

import pytest

from tcenv.remote.artifact_builder import BuildCapabilities, BuildProfile
from tcenv.track import load_track_registry
from tcenv.track import tilelang as contract
from tcenv.track import tilelang_confirmation as fresh
from tcenv.track.delivery import candidate_geometry
from tcenv.backends.metal import tilelang_metal
from tests.shared.test_track_mt1b_tilelang_contract import (
    ATTEMPT,
    GOOD_ENVIRONMENT,
    GOOD_FRONTEND,
    R5_CANDIDATE,
    _production_artifacts,
    _source_candidate,
)


def _source_spec():
    return load_track_registry().get(contract.TILELANG_TRACK_ID, version=2)


def _fresh_spec():
    source = _source_spec()
    return dataclasses.replace(
        source,
        track_version=fresh.TILELANG_CONFIRMATION_TRACK_VERSION,
        status="interface_only",
        capabilities=tuple(item for item in source.capabilities if item != "metal_dispatch"),
        build_adapter=dataclasses.replace(source.build_adapter, implementation_version=3),
        evidence=(),
    )


def _base_profile() -> BuildProfile:
    return BuildProfile(
        build_profile="tilelang-metal-worker-v2",
        builder_id="metal-source-builder",
        builder_version="1",
        build_host="control_host",
        toolchain_id="apple-metal",
        toolchain_version="4.0",
        sdk_version="macos-test",
        compile_profile={
            "device": "Apple M4 Max",
            "gpu_family": "Apple9",
            "language": "msl",
            "math": "safe",
            "unified_memory": True,
        },
    )


def _identities():
    source = R5_CANDIDATE.read_bytes()
    candidate = _source_candidate(source=source)
    artifacts = _production_artifacts(source)
    first = contract.source_only_build_identity(
        candidate=candidate,
        observed_frontend=GOOD_FRONTEND,
        environment=GOOD_ENVIRONMENT,
        artifacts=artifacts,
        attempt=ATTEMPT,
        expected_kernel_names=("scale_kernel",),
        track_version=2,
    )
    second = contract.source_only_build_identity(
        candidate=candidate,
        observed_frontend=GOOD_FRONTEND,
        environment=GOOD_ENVIRONMENT,
        artifacts=artifacts,
        attempt=dict(ATTEMPT, request_sha256="c" * 64),
        expected_kernel_names=("scale_kernel",),
        track_version=2,
    )
    return candidate, first, second


def test_mt6_fresh_profile_is_attempt_stable_but_v2_profile_is_unchanged():
    candidate, first, second = _identities()
    base = _base_profile()
    fresh_first = tilelang_metal._metal_profile(base, first, track_version=3, candidate=candidate)
    fresh_second = tilelang_metal._metal_profile(base, second, track_version=3, candidate=candidate)
    assert fresh_first == fresh_second
    assert fresh_first.builder_version == "3"
    assert fresh_first.compile_profile["tilelang_fresh_subject"] == fresh.expected_fresh_subject(
        candidate=candidate, environment=GOOD_ENVIRONMENT
    ).cache_key
    assert "tilelang_source_only_identity" not in fresh_first.compile_profile

    legacy_first = tilelang_metal._metal_profile(base, first)
    legacy_second = tilelang_metal._metal_profile(base, second)
    assert legacy_first.builder_version == "2"
    assert legacy_first.compile_profile["tilelang_source_only_identity"] == first.digest
    assert legacy_second.compile_profile["tilelang_source_only_identity"] == second.digest
    assert legacy_first != legacy_second


def test_mt6_probe_rebuilds_the_same_stable_profile_without_device_or_candidate_execution(monkeypatch):
    candidate = _source_candidate()
    geometry = candidate_geometry(candidate)
    resolved = SimpleNamespace(source=geometry.source, active_stages=geometry.stages)
    captured = {}

    class _Probe:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def probe(self):
            return BuildCapabilities(profile=_base_profile(), available=True)

    monkeypatch.setattr(tilelang_metal, "MetalSourceBuilder", _Probe)
    monkeypatch.setattr(tilelang_metal, "observe_staged_frontend", lambda staging: object())
    monkeypatch.setattr(tilelang_metal, "observe_staged_environment", lambda frontend: GOOD_ENVIRONMENT)
    builder = tilelang_metal.TileLangFreshMetalBuilder(_fresh_spec())
    capabilities = builder.probe(candidate, resolved)
    assert capabilities.available
    assert builder.spec.key == ("tilelang-pipeline", 3)
    assert builder.source_spec.key == ("tilelang-pipeline", 2)
    assert captured["math"] == "safe"
    assert captured["kernel_names"] == ("scale_kernel",)
    assert capabilities.profile.compile_profile["tilelang_fresh_subject"] == fresh.expected_fresh_subject(
        candidate=candidate, environment=GOOD_ENVIRONMENT
    ).cache_key
    assert "tilelang_source_only_identity" not in capabilities.profile.compile_profile


def test_mt6_native_descriptor_is_attempt_stable_and_log_keeps_attempt_evidence():
    candidate, first, second = _identities()
    subject = fresh.expected_fresh_subject(candidate=candidate, environment=GOOD_ENVIRONMENT).cache_key
    first_outcome = SimpleNamespace(identity=first, entry_sha256=tilelang_metal.FRONTEND_ENTRY_SHA256)
    second_outcome = SimpleNamespace(identity=second, entry_sha256=tilelang_metal.FRONTEND_ENTRY_SHA256)
    document = tilelang_metal._descriptor(first_outcome, track_version=3, fresh_subject=subject)
    second_document = tilelang_metal._descriptor(second_outcome, track_version=3, fresh_subject=subject)
    assert document == second_document
    assert document["schema_id"] == tilelang_metal.METAL_FRESH_DESCRIPTOR_SCHEMA_ID
    assert document["track"] == {"track_id": "tilelang-pipeline", "track_version": 3}
    assert "source_only_identity_digest" not in document
    assert "attempt" not in document
    assert document["fresh_subject"] == subject
    attempt_log = tilelang_metal._fresh_attempt_log(first)
    assert b'"attempt"' in attempt_log
    assert first.document()["attempt"] == ATTEMPT
    legacy = tilelang_metal._descriptor(first_outcome)
    assert legacy["track"]["track_version"] == 2
    assert legacy["source_only_identity_digest"] == first.digest
    assert legacy["attempt"] == ATTEMPT
    assert "fresh_subject" not in legacy


def test_mt6_artifact_audit_recomputes_and_rejects_a_drifting_fresh_subject():
    from tcenv.remote.execution_geometry import EvaluatorBuildError

    candidate, first, _ = _identities()
    subject = fresh.expected_fresh_subject(candidate=candidate, environment=GOOD_ENVIRONMENT).cache_key
    outcome = SimpleNamespace(identity=first, entry_sha256=tilelang_metal.FRONTEND_ENTRY_SHA256)
    document = tilelang_metal._descriptor(outcome, track_version=3, fresh_subject=subject)
    artifacts = _production_artifacts(R5_CANDIDATE.read_bytes())
    generated = [
        {
            "path": path,
            "role": "AUXILIARY",
            "media_type": contract.SOURCE_ONLY_ARTIFACT_MEDIA_TYPES[path],
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for path, data in artifacts.items()
    ]
    kwargs = {
        "entries": generated,
        "candidate_entries": [{"path": entry.path} for entry in candidate.entries],
        "manifest_entry": {"path": "candidate.manifest.json"},
        "geometry_entry": {"path": contract.GEOMETRY_PATH},
        "native": {"descriptor": {"tilelang": document}},
        "canonical_bundle": SimpleNamespace(candidate=candidate),
    }
    builder = tilelang_metal.TileLangFreshMetalBuilder(_fresh_spec())
    entries, entry_paths = builder.audit_evaluator_build(**kwargs)
    assert {entry["path"] for entry in entries} >= set(artifacts)
    assert entry_paths == {"kernel.metal"}
    drifted = copy.deepcopy(kwargs)
    drifted["native"]["descriptor"]["tilelang"]["fresh_subject"] = "0" * 64
    with pytest.raises(EvaluatorBuildError, match="fresh subject"):
        builder.audit_evaluator_build(**drifted)


def test_mt6_v3_uses_the_stable_native_barrier_without_changing_v2():
    from tcenv.backends.metal.tilelang_runtime import _configure_native_batch

    original = object()
    ctx = SimpleNamespace(_native_batch=original)
    _configure_native_batch(ctx, 3)
    assert ctx._native_batch is original
    assert ctx.tilelang_barrier_interval == 2
    _configure_native_batch(ctx, 2)
    assert ctx._native_batch is original
    assert ctx.tilelang_barrier_interval == 0


def test_mt6_track_explore_reads_taic_best_source(tmp_path):
    from taic.track_explore import _best_source_path

    (tmp_path / "best.taic").write_text("taic-envelope 1\n", encoding="utf-8")
    assert _best_source_path(tmp_path) == tmp_path / "best.taic"
    (tmp_path / "best.taic").unlink()
    (tmp_path / "best.metal").write_text("kernel void k() {}", encoding="utf-8")
    assert _best_source_path(tmp_path) == tmp_path / "best.metal"


def test_mt6_track_explore_allows_retry_attempts_not_batches():
    from taic.track_explore import TrackExploreError, _validate_confirmation_attempts

    _validate_confirmation_attempts(
        {"confirmation_attempts": 2}, {"benchmark": {"attempts": 2}}, 1
    )
    _validate_confirmation_attempts(
        {"confirmation_attempts": 3}, {"benchmark": {"attempts": 1}}, 1
    )
    with pytest.raises(TrackExploreError, match="confirmation_batch_mismatch"):
        _validate_confirmation_attempts(
            {"confirmation_attempts": 4}, {"benchmark": {"attempts": 4}}, 1
        )


def test_mt6_build_cost_uses_complete_build_evidence_duration():
    from taic.track_explore import _compile_seconds

    report = {
        "compile": {"compile_s": 0.0002, "pipeline_s": 0.0003},
        "build": {"evidence": {"duration_ms": 3408}},
    }
    assert _compile_seconds(report) == 3.408
    assert _compile_seconds({"compile": {"compile_s": 0.001, "pipeline_s": 0.002}}) == 0.003


def test_mt6_fresh_registration_is_interface_only_and_version_checked():
    source = _source_spec()
    spec = _fresh_spec()
    audit = fresh.audit_fresh_registration(spec, source_track=source)
    assert audit["track"] == "tilelang-pipeline/3"
    assert audit["source_only_track"] == "tilelang-pipeline/2"
    with pytest.raises(contract.TileLangContractError, match="available|interface_only"):
        fresh.audit_fresh_registration(dataclasses.replace(spec, status="available"), source_track=source)
    with pytest.raises(contract.TileLangContractError, match="metal_dispatch"):
        fresh.audit_fresh_registration(
            dataclasses.replace(spec, capabilities=(*spec.capabilities, "metal_dispatch")), source_track=source
        )
    with pytest.raises(contract.TileLangIdentityDrift, match="implementation_version=3"):
        fresh.audit_fresh_registration(
            dataclasses.replace(spec, build_adapter=source.build_adapter), source_track=source
        )
    with pytest.raises(ValueError, match="math='safe'"):
        tilelang_metal.TileLangFreshMetalBuilder(spec, math="fast")
