"""MT6 /3 real Metal evidence; requires explicit GPU authorization, never a model call."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tcenv.backends.metal import portfolio_confirmation as confirmation
from tcenv.backends.metal import search_candidate as search
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.target import load_target
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.dispatch import DispatchRequest
from taic.task_binding import bind_task
from tcenv.track import tilelang_confirmation as fresh
from tcenv.track.registry import load_track_registry
from tests.backends.metal.test_tilelang_mt6_builder import _fresh_spec

ROOT = Path(__file__).resolve().parents[3]
MT4_RUN = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT4/evidence/deepseek-relu-r1"


def _available_spec():
    spec = _fresh_spec()
    return dataclasses.replace(
        spec,
        status="available",
        capabilities=(*spec.capabilities, "metal_dispatch"),
        evidence=("tests/backends/metal/test_tilelang_mt6_gpu.py",),
    )


def _registry_file(tmp_path: Path) -> Path:
    document = json.loads((ROOT / "backends/metal/tracks.json").read_text(encoding="utf-8"))
    available = _available_spec().document()
    document["tracks"] = [
        track
        for track in document["tracks"]
        if (track["track_id"], track["track_version"]) != (available["track_id"], available["track_version"])
    ]
    document["tracks"].append(available)
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_track_registry(path)
    return path


def _source() -> str:
    return (MT4_RUN / "best.taic").read_text(encoding="utf-8")


def _locator():
    return bind_task("relu", "fp16", "small").locator


@pytest.mark.gpu
def test_mt6_real_metal_two_independent_builds_have_stable_identity(tmp_path, monkeypatch):
    registry = _registry_file(tmp_path)
    selected = load_track_registry(registry)
    monkeypatch.setattr("tcenv.portfolio.candidate.load_track_registry", lambda *args, **kwargs: selected)
    monkeypatch.setattr(search, "load_track_registry", lambda *args, **kwargs: selected)
    monkeypatch.setattr(confirmation, "load_track_registry", lambda *args, **kwargs: selected)
    monkeypatch.setattr(fresh, "load_track_registry", lambda *args, **kwargs: selected)
    target = load_target("apple-m4-max")
    locator = _locator()
    source = _source()
    first = evaluate_candidate(
        locator,
        source,
        target=target,
        mode="quick",
        seed=7,
        reps=1,
        track_registry_path=registry,
        track_id="tilelang-pipeline",
        track_version=3,
    )
    second = evaluate_candidate(
        locator,
        source,
        target=target,
        mode="quick",
        seed=123,
        reps=1,
        track_registry_path=registry,
        track_id="tilelang-pipeline",
        track_version=3,
    )
    assert first.result is not None and second.result is not None
    left = first.result.report
    right = second.result.report
    assert left["status"] == right["status"] == "ok"
    assert left["build"]["native_digest"] == right["build"]["native_digest"]
    assert left["build"]["artifacts"]["artifact_set_digest"] == right["build"]["artifacts"]["artifact_set_digest"]
    assert left["build"]["assembled_sha256"] == right["build"]["assembled_sha256"]
    left_profile = left["build"]["provenance"]["compile_profile"]
    right_profile = right["build"]["provenance"]["compile_profile"]
    assert left_profile == right_profile
    assert left_profile["tilelang_fresh_subject"]
    assert "tilelang_source_only_identity" not in left_profile
    descriptor = left["build"]["native"]["descriptor"]["tilelang"]
    assert descriptor["schema_id"] == "tcenv.track.tilelang.metal-build.2"
    assert descriptor["track"] == {"track_id": "tilelang-pipeline", "track_version": 3}
    assert "attempt" not in descriptor and "source_only_identity_digest" not in descriptor
    assert any(row["kernel"] == "relu_kernel_kernel" for row in first.result.execution["dispatch"])


@pytest.mark.gpu
def test_mt6_real_fresh_confirmation_benchmark_and_unseen_golden(tmp_path, monkeypatch):
    registry = _registry_file(tmp_path)
    selected = load_track_registry(registry)
    monkeypatch.setattr("tcenv.portfolio.candidate.load_track_registry", lambda *args, **kwargs: selected)
    monkeypatch.setattr(search, "load_track_registry", lambda *args, **kwargs: selected)
    monkeypatch.setattr(confirmation, "load_track_registry", lambda *args, **kwargs: selected)
    monkeypatch.setattr(fresh, "load_track_registry", lambda *args, **kwargs: selected)
    monkeypatch.setattr("tcenv.taskcontract.runner.load_track_registry", lambda *args, **kwargs: selected)
    locator = _locator()
    target = load_target("apple-m4-max")
    run_config = json.loads((MT4_RUN / "run_config.json").read_text(encoding="utf-8"))
    frozen = search.confirmation_input(
        locator,
        _source(),
        target=target,
        guidance_digest=run_config["confirmation_guidance_digest"],
        master_seed=b"0123456789abcdef",
        track_id="tilelang-pipeline",
        track_version=3,
    )
    assert frozen.freeze.protocol_epoch == 3
    documents = {}
    for usage in ("benchmark", "golden"):
        if usage == "benchmark":
            request = frozen.request
        else:
            contract = locator.load()
            instance = instantiate_v2(contract.spec, "fp16", "small")
            point = frozen.freeze.domain_plan.find("fp16", "small", "golden")
            request = DispatchRequest.from_instance(
                instance, target_abi_digest=point.target_abi_digest, usage="golden"
            )
        document = confirmation.confirm_subject(
            locator=locator,
            domain_plan=frozen.freeze.domain_plan,
            portfolio=frozen.freeze.portfolio(),
            request=request,
            evidence=frozen.evidence,
            validation=frozen.validation,
            freeze=frozen.freeze,
            track_version=3,
            track_id="tilelang-pipeline",
        )
        documents[usage] = document
        (tmp_path / f"fresh-{usage}.json").write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        print("FRESH_VERDICT", usage, document["status"], document["reason"], document["identity"])
        # This is the mechanism test, not a quiet-performance publication gate.  A noisy benchmark
        # is preserved as a typed rejection; correctness/identity must still match and golden must
        # confirm.  MT6 cannot become available while this outcome is unstable.
        if usage == "golden":
            assert document["status"] == "confirmed", document
        else:
            assert document["status"] in ("confirmed", "rejected"), document
            if document["status"] == "rejected":
                assert document["reason"] == "timing_uncertain", document
        assert document["identity"]["kind"] == "match"
        assert document["provider"] == "candidate"
        assert document["credits_variant"] is True
        assert document["confirmation_seed"] != document["search_seed"]
        search_report = document["search_report"]
        fresh_report = document["confirmation_report"]
        assert search_report["status"] == fresh_report["status"] == "ok"
        assert search_report["build"]["native_digest"] == fresh_report["build"]["native_digest"]
        assert (
            search_report["build"]["provenance"]["compile_profile"]
            == fresh_report["build"]["provenance"]["compile_profile"]
        )
        assert "tilelang_fresh_subject" in search_report["build"]["provenance"]["compile_profile"]
    assert set(documents) == {"benchmark", "golden"}
    print(json.dumps({
        usage: {
            "status": document["status"],
            "attempts": document["confirmation_attempts"],
            "candidate_us": document["confirmation_report"].get("candidate_us"),
            "speedup": document["confirmation_report"].get("speedup"),
        }
        for usage, document in documents.items()
    }, sort_keys=True))