"""MT6 /3 confirmation and activation routing tests; all CPU, no real build/dispatch."""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import pytest

from tcenv.backends.metal import portfolio_confirmation as confirmation
from tcenv.backends.metal import search_candidate as search
from tcenv.backends.metal import tilelang_metal
from tcenv.contract_errors import UnsupportedContractCapability
from tcenv.portfolio.candidate import project_provider
from tcenv.remote.artifact_builder import BuildCapabilities, BuildProvenance
from tcenv.target import load_target
from tcenv.track import tilelang_confirmation as fresh
from tcenv.track.envelope import encode_envelope
from tcenv.track.registry import TrackRegistry, load_track_registry
from tests.backends.metal.test_tilelang_mt6_builder import _base_profile, _fresh_spec
from tests.shared.test_track_mt1b_tilelang_contract import GOOD_ENVIRONMENT
from tests.support.tilelang_mt2 import tilelang_candidate, write_mt2_task


def _registry() -> TrackRegistry:
    base = load_track_registry()
    spec = _fresh_spec()
    available = dataclasses.replace(
        spec,
        status="available",
        capabilities=(*spec.capabilities, "metal_dispatch"),
        evidence=("tests/backends/metal/test_tilelang_mt6_confirmation.py",),
    )
    tracks = tuple(track for track in base.tracks if track.key != spec.key)
    return TrackRegistry(
        backend_id=base.backend_id,
        tracks=(*tracks, available),
        schema_version=base.schema_version,
    )


@pytest.fixture
def fresh_registry(monkeypatch):
    registry = _registry()
    import taic.track_prompt as track_prompt

    monkeypatch.setattr(track_prompt, "load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr(search, "load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr(confirmation, "load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr(fresh, "load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr("tcenv.portfolio.candidate.load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr("tcenv.track.registry.load_track_registry", lambda *args, **kwargs: registry)
    confirmation._EXPECTATION_CACHE.clear()
    return registry


def _frozen(tmp_path):
    locator = write_mt2_task(tmp_path)
    source = encode_envelope(tilelang_candidate(worklist=False))
    frozen = search.confirmation_input(
        locator,
        source,
        target=load_target("apple-m4-max"),
        guidance_digest="0" * 64,
        master_seed=b"0123456789abcdef",
        track_id="tilelang-pipeline",
        track_version=3,
    )
    return locator, source, frozen


def _projection(frozen):
    portfolio = frozen.freeze.portfolio()
    return project_provider(
        portfolio,
        portfolio.variants[0],
        authority=frozen.freeze.authority,
        track_id="tilelang-pipeline",
        track_version=3,
    )


def test_mt6_confirmation_input_derives_dsl_epoch_without_authorizing_gpu(fresh_registry, tmp_path):
    _, _, frozen = _frozen(tmp_path)
    assert frozen.freeze.protocol_epoch == search.DSL_PROTOCOL_EPOCH == 3
    assert frozen.request.usage == "benchmark"
    assert frozen.validation.tune_shards == ("fp32/small/benchmark",)
    assert frozen.validation.unseen_instance_keys == ("fp32/small/golden",)
    # This is a test-only available registry fixture for CPU routing.  The production registry
    # remains without /3; the separate interface-only test below fails before any run.


def test_mt6_expected_uses_fresh_builder_probe_not_msl_source(fresh_registry, tmp_path, monkeypatch):
    locator, _, frozen = _frozen(tmp_path)
    contract = locator.load()
    projection = _projection(frozen)
    resolved = confirmation._resolved(contract, projection, load_target("apple-m4-max"))
    subject = fresh.expected_fresh_subject(
        candidate=projection.candidate, environment=GOOD_ENVIRONMENT
    ).cache_key
    profile = tilelang_metal._metal_profile(
        _base_profile(), track_version=3, fresh_subject=subject
    )
    captured = {}

    class _Builder:
        def __init__(self, spec, *, target_id):
            captured["spec"] = spec.key
            captured["target_id"] = target_id

        def probe(self, candidate, geometry):
            captured["candidate"] = candidate.candidate_sha256
            captured["geometry"] = geometry
            return BuildCapabilities(profile=profile, available=True)

    monkeypatch.setattr(tilelang_metal, "TileLangFreshMetalBuilder", _Builder)
    monkeypatch.setattr(confirmation, "MetalBackend", lambda: SimpleNamespace(version="test-metal"))
    monkeypatch.setattr(
        confirmation,
        "combined_candidate_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("MSL source path was reached")),
    )
    expected = confirmation.confirmation_expected(
        contract, projection, resolved, locator=locator, target_id="apple-m4-max"
    )
    compile_profile = expected["build"]["provenance"]["compile_profile"]
    assert compile_profile["tilelang_fresh_subject"] == subject
    assert "tilelang_source_only_identity" not in compile_profile
    assert expected["build"]["candidate_sha256"] == projection.candidate.candidate_sha256
    assert expected["evaluation"]["target_id"] == "apple-m4-max"
    assert captured["spec"] == ("tilelang-pipeline", 3)
    assert captured["candidate"] == projection.candidate.candidate_sha256
    assert captured["geometry"] is resolved


def test_mt6_activation_expectation_replays_dsl_dispatch_and_track(fresh_registry, tmp_path, monkeypatch):
    locator, _, frozen = _frozen(tmp_path)
    sentinel = {"expected": "independent-fresh-/3"}
    monkeypatch.setattr(confirmation, "confirmation_expected", lambda *args, **kwargs: sentinel)
    record, expected = confirmation.activation_expectation(
        locator=locator,
        freeze=frozen.freeze,
        validation=frozen.validation,
        document={"instance_key": "fp32/small/benchmark"},
        track_version=3,
        track_id="tilelang-pipeline",
    )
    assert record.outcome == "resolved"
    assert record.provider == "candidate"
    assert expected is sentinel


def test_mt6_dsl_evaluator_uses_registered_adapter_and_preserves_cases(fresh_registry, monkeypatch):
    captured = {}
    sentinel = object()

    class _Builder:
        def __init__(self, **kwargs):
            captured["builder"] = kwargs

    class _Evaluator:
        def __init__(self, **kwargs):
            captured["evaluator"] = kwargs

    class _Adapter:
        def resolve(self, name):
            return {
                "builder": lambda **kwargs: _Builder(**kwargs),
                "evaluator": _Evaluator,
                "geometry_resolver": "candidate-geometry",
            }[name]

    monkeypatch.setattr(confirmation, "adapter_for_track", lambda track: _Adapter())
    monkeypatch.setattr(
        confirmation,
        "load_track_registry",
        lambda: SimpleNamespace(require_available=lambda *_args, **_kwargs: _fresh_spec()),
    )
    def _run(**kwargs):
        captured["run"] = kwargs
        return sentinel

    monkeypatch.setattr(confirmation, "run_portfolio_task", _run)
    locator = SimpleNamespace(load=lambda: SimpleNamespace(instance=SimpleNamespace(case_domain=SimpleNamespace(cases=("canonical", "edge")))))
    result = confirmation._evaluate(
        locator,
        object(),
        object(),
        object(),
        object(),
        mode="full",
        seed=7,
        reps=15,
        target_id="apple-m4-max",
        track_version=3,
        track_id="tilelang-pipeline",
    )
    assert result is sentinel
    assert captured["run"]["track_id"] == "tilelang-pipeline"
    assert captured["run"]["track_version"] == 3
    assert captured["run"]["geometry_resolver_factory"](None) == "candidate-geometry"
    assert captured["evaluator"]["cases"] == ("canonical", "edge")


def test_mt6_v2_local_evaluator_reuses_the_fresh_expected_route(fresh_registry, tmp_path, monkeypatch):
    from taic.task_binding import TaskBinding
    from taic.v2_evaluation import V2LocalEvaluator

    locator, source, _ = _frozen(tmp_path)
    binding = TaskBinding(
        mode="v2",
        name=locator.op_name,
        dtype="fp32",
        size="small",
        target_id="apple-m4-max",
        locator=locator,
        contract=locator.load(),
        track_id="tilelang-pipeline",
        track_version=3,
    )
    sentinel = {"expected": "v2-local-fresh-/3"}
    monkeypatch.setattr(confirmation, "confirmation_expected", lambda *args, **kwargs: sentinel)
    result = V2LocalEvaluator(binding).confirmation_expected(
        task=locator.op_name,
        dtype="fp32",
        size="small",
        target_id="apple-m4-max",
        source=source,
    )
    assert result is sentinel


def test_mt6_confirm_candidate_forwards_the_fresh_track_to_real_eval(fresh_registry, tmp_path, monkeypatch):
    locator, source, _ = _frozen(tmp_path)
    captured = {}

    class _Stop(Exception):
        pass

    def _stop(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop()

    monkeypatch.setattr(confirmation, "_evaluate", _stop)
    with pytest.raises(_Stop):
        search.confirm_candidate(
            locator,
            source,
            target=load_target("apple-m4-max"),
            guidance_digest="0" * 64,
            master_seed=b"0123456789abcdef",
            track_id="tilelang-pipeline",
            track_version=3,
        )
    assert captured["track_id"] == "tilelang-pipeline"
    assert captured["track_version"] == 3
    assert captured["seed"] == search.SEARCH_SEED


def test_mt6_coordinated_report_forgery_cannot_override_independent_expected(fresh_registry, tmp_path):
    from taic.policy import compare_confirmation_identity

    _, _, frozen = _frozen(tmp_path)
    projection = _projection(frozen)
    subject = fresh.expected_fresh_subject(
        candidate=projection.candidate, environment=GOOD_ENVIRONMENT
    ).cache_key
    profile = tilelang_metal._metal_profile(_base_profile(), track_version=3, fresh_subject=subject)
    expected = {
        "build": {
            "candidate_sha256": projection.candidate.candidate_sha256,
            "opspec_digest": frozen.freeze.authority.spec.digest,
            "lowering_profile_digest": frozen.freeze.authority.contract.fingerprint(),
            "provenance": BuildProvenance(
                profile=profile,
                source_sha256=projection.candidate.candidate_sha256,
                artifact_set_digest=None,
            ).to_dict(),
        },
        "evaluation": {
            "evaluation_fingerprint": "eval",
            "baseline_implementation_digest": "baseline",
            "baseline_environment_digest": "environment",
            "protocol_version": 4,
            "target_id": "apple-m4-max",
            "backend": "metal",
            "backend_version": "test",
            "device_name": "Apple M4 Max",
            "gpu_family": "Apple9",
        },
    }

    def report(artifact="a" * 64):
        return {
            **copy.deepcopy(expected["evaluation"]),
            "build": {
                **copy.deepcopy(expected["build"]),
                "assembled_sha256": "b" * 64,
                "artifact_set_digest": artifact,
                "native_digest": "c" * 64,
            },
        }

    search_report = report()
    confirmation_report = report()
    matched = compare_confirmation_identity(
        candidate_sha256=projection.candidate.candidate_sha256,
        expected=expected,
        search=search_report,
        confirmation=confirmation_report,
        scope="local",
    )
    assert matched["kind"] == "match"

    for forged in (search_report, confirmation_report):
        forged["build"]["provenance"]["compile_profile"]["tilelang_fresh_subject"] = "0" * 64
    coordinated = compare_confirmation_identity(
        candidate_sha256=projection.candidate.candidate_sha256,
        expected=expected,
        search=search_report,
        confirmation=confirmation_report,
        scope="local",
    )
    assert coordinated["kind"] == "build_identity_mismatch"
    assert any("compile_profile" in item["field"] for item in coordinated["differences"])

    smuggled_search = report()
    smuggled_confirmation = report()
    for forged in (smuggled_search, smuggled_confirmation):
        forged["build"]["provenance"]["compile_profile"]["tilelang_source_only_identity"] = "d" * 64
    smuggled = compare_confirmation_identity(
        candidate_sha256=projection.candidate.candidate_sha256,
        expected=expected,
        search=smuggled_search,
        confirmation=smuggled_confirmation,
        scope="local",
    )
    assert smuggled["kind"] != "match"

    outcome = compare_confirmation_identity(
        candidate_sha256=projection.candidate.candidate_sha256,
        expected=expected,
        search=report("a" * 64),
        confirmation=report("e" * 64),
        scope="local",
    )
    assert outcome["kind"] == "build_identity_mismatch"


def test_mt6_available_adapter_replays_confirmation_seed_and_guidance(fresh_registry, tmp_path):
    from taic.portfolio_confirmation import confirmation_seed_for
    from tcenv.backends.metal.track_factory import adapter_for_track

    locator, source, frozen = _frozen(tmp_path)
    adapter = adapter_for_track(fresh_registry.get("tilelang-pipeline", version=3))
    derived = adapter.resolve("confirmation")(
        locator,
        source,
        target=load_target("apple-m4-max"),
        guidance_digest="0" * 64,
        master_seed=b"0123456789abcdef",
        instance_key="fp32/small/benchmark",
    )
    assert derived["portfolio_freeze_digest"] == frozen.freeze.digest
    assert derived["validation_plan_digest"] == frozen.validation.digest
    assert derived["confirmation_seed"] == confirmation_seed_for(frozen.validation, "fp32/small/benchmark")
    assert adapter.resolve("guidance")("system", "statement")


def test_mt6_interface_only_evaluator_fails_before_any_run(fresh_registry, monkeypatch):
    called = []
    base = load_track_registry()
    fresh_spec = _fresh_spec()
    tracks = tuple(track for track in base.tracks if track.key != fresh_spec.key)
    interface_registry = TrackRegistry(
        backend_id=base.backend_id,
        tracks=(*tracks, fresh_spec),
        schema_version=base.schema_version,
    )
    monkeypatch.setattr(confirmation, "load_track_registry", lambda: interface_registry)
    monkeypatch.setattr(confirmation, "run_portfolio_task", lambda **kwargs: called.append(kwargs))
    with pytest.raises(UnsupportedContractCapability, match="interface_only|not selectable|available"):
        confirmation._evaluate(
            SimpleNamespace(),
            object(),
            object(),
            object(),
            object(),
            mode="quick",
            seed=7,
            reps=3,
            target_id="apple-m4-max",
            track_version=3,
            track_id="tilelang-pipeline",
        )
    assert called == []
