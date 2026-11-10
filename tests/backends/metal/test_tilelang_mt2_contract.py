"""MT2 TileLang /2 contract and adapter tests (CPU/source-only; no device dispatch)."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tcenv.backends.metal.tilelang_build import TileLangSourceOnlyBuild
from tcenv.backends.metal.tilelang_runtime import TileLangMetalExecutor, TileLangTrackEvaluator, launch_spec_from_geometry
from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
from tcenv.track import (
    CanonicalBundle, CandidateEntry, abi_limits_from_target, candidate_geometry,
    geometry_pair, load_track_registry,
)
from tcenv.target import load_target
from tcenv.track import tilelang as contract
from tcenv.track.abi import derive_tensor_memory_contract
from tests.support.tilelang_mt2 import (
    TRACK_ID, TRACK_VERSION, tilelang_candidate, write_mt2_registry, write_mt2_task,
)


def _spec(path: Path):
    return load_track_registry(path).get(TRACK_ID, version=TRACK_VERSION)


def _resolved(locator, candidate):
    task = locator.load()
    target = load_target("apple-m4-max")
    abi = derive_tensor_memory_contract(task.instance, target=target)
    limits = abi_limits_from_target(target)
    pair = geometry_pair(candidate_geometry(candidate), None, instance=task.instance, contract=abi, limits=limits)
    return task, abi, limits, pair, CanonicalBundle(candidate=candidate, pair=pair)


def test_mt2_v2_coexists_with_the_frozen_v1_and_has_a_distinct_contract(tmp_path):
    registry = load_track_registry(write_mt2_registry(tmp_path / "tracks.json"))
    v1 = registry.get(TRACK_ID, version=1)
    v2 = registry.get(TRACK_ID, version=2)
    assert v1.status == "interface_only" and not v1.selectable
    assert v2.status == "available" and v2.selectable and v2.supports("metal_dispatch")
    assert registry.get(TRACK_ID).track_version == 2
    assert registry.require_available(TRACK_ID).track_version == 2
    assert contract.frozen_contract() == contract.frozen_contract(track_version=1)
    assert contract.frozen_contract(track_version=2)["build"]["implemented_modes"] == ["source_only", "metal_build"]
    from tcenv.identity import fingerprint

    assert fingerprint(contract.frozen_contract()) == "689f86802ed8e040a37461677c4ffb922b9be1b8ea4c3e73b18faeeed9100022"
    assert fingerprint(contract.frozen_contract(track_version=2)) != fingerprint(contract.frozen_contract())


def test_mt2_frontend_or_adapter_drift_is_typed(tmp_path):
    spec = _spec(write_mt2_registry(tmp_path / "tracks.json"))
    drifted = dataclasses.replace(spec, frontend=dataclasses.replace(spec.frontend, frontend_version="0.1.16"))
    with pytest.raises(contract.TileLangIdentityDrift, match="frontend.frontend_version"):
        contract.audit_registration(drifted)
    mismatched = dataclasses.replace(spec, build_adapter=dataclasses.replace(spec.build_adapter, implementation_version=1))
    with pytest.raises(contract.TileLangContractError, match="frozen contract is version 1"):
        contract.audit_registration(mismatched)
    unknown = dataclasses.replace(spec, track_version=3)
    with pytest.raises(contract.TileLangContractError, match="contract versions"):
        contract.audit_registration(unknown)


def test_mt2_geometry_projects_directly_to_the_shared_launch_spec_without_taic_launch(tmp_path):
    locator = write_mt2_task(tmp_path)
    candidate = tilelang_candidate(worklist=True)
    _task, abi, _limits, pair, _canonical = _resolved(locator, candidate)
    resolved = pair.candidate
    spec = launch_spec_from_geometry(resolved, abi)
    assert spec.math == "safe"
    assert [(stage.kernel, stage.grid, stage.threads, stage.workspaces) for stage in spec.stages] == [
        ("scale_kernel_kernel", (8, 1, 1), (128, 1, 1), ("tmp",)),
        ("bias_kernel_kernel", (8, 1, 1), (128, 1, 1), ("tmp",)),
    ]
    assert [(item.name, item.nbytes, item.zero) for item in spec.workspaces] == [("tmp", 4096, False)]
    assert "taic-launch" not in spec.digest()


def test_mt2_executor_screens_generated_msl_not_just_the_python_candidate(tmp_path):
    locator = write_mt2_task(tmp_path)
    candidate = tilelang_candidate(worklist=False)
    task, abi, limits, pair, canonical = _resolved(locator, candidate)
    source = b'#include "/etc/passwd"\nkernel void scale_kernel_kernel() {}\n'
    entry = BundleEntry(
        path="kernel.metal", role="AUXILIARY", media_type="text/x-metal-source",
        size_bytes=len(source), sha256=hashlib.sha256(source).hexdigest(), data=source,
    )
    native = NativeExecutable(
        backend_id="metal", kind="compute-pipeline", entry_path="kernel.metal",
        entry_sha256=entry.sha256,
        descriptor={"tilelang": {"schema_id": "tcenv.track.tilelang.metal-build.1"}},
    )
    request = SimpleNamespace(
        build_request=SimpleNamespace(
            candidate=candidate, contract=task, abi=abi, limits=limits,
            selection=SimpleNamespace(geometry=pair.candidate),
        ),
        geometry=pair.candidate,
        canonical_bundle=canonical,
        build=SimpleNamespace(bundle=ArtifactBundle(entries=(entry,), native=native)),
    )
    executor = TileLangMetalExecutor(request)
    assert executor.screen() and "include" in executor.screen()[0]


def test_mt2_timeout_or_crash_never_fabricates_a_full_build():
    request = SimpleNamespace(
        build_request=SimpleNamespace(
            contract=SimpleNamespace(op_name="mt2_scale", instance=SimpleNamespace(dtype="fp32", size="small"))
        )
    )
    evaluator = TileLangTrackEvaluator()
    for status in ("timeout", "crash"):
        report = evaluator._synthetic_failure(request, status, "fixed MT2 negative")
        assert report["status"] == status and report["build"] is None
        assert evaluator.evaluated_build is None


def test_mt2_adapters_do_not_import_the_msl_track_adapter_or_host_tiling_plan():
    root = Path(__file__).resolve().parents[3]
    for relative in (
        "tcenv/backends/metal/kernel_pipeline.py",
        "tcenv/backends/metal/tilelang_metal.py",
        "tcenv/backends/metal/tilelang_runtime.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any("track_adapter" in name or "HostTilingPlan" in name for name in imported), (relative, imported)


def test_mt2_multi_function_module_has_one_preamble_and_two_real_kernels(tmp_path):
    registry = write_mt2_registry(tmp_path / "tracks.json")
    spec = _spec(registry)
    candidate = tilelang_candidate(worklist=True)
    try:
        outcome = TileLangSourceOnlyBuild(spec, attempt_base=tmp_path).build(
            candidate, expected_kernel_names=("scale_kernel_kernel", "bias_kernel_kernel")
        )
    except contract.TileLangFrontendUnavailable as exc:
        # Hosts without the explicitly pinned frontend/hmz still fail closed; the real positive is
        # the authorized Metal GPU node, not an unfenced fallback here.
        assert exc.code == "backend_capability_unavailable"
        return
    source = (outcome.attempt_root / "build/artifacts/kernel.metal").read_text(encoding="utf-8")
    assert source.count("union __TVMArgUnion {") == 1
    assert source.count("#include <metal_stdlib>") == 1
    assert source.count("void scale_kernel_kernel(") == 1
    assert source.count("void bias_kernel_kernel(") == 1
    assert outcome.identity.track_version == 2 and outcome.identity.full_build
    assert outcome.identity.document()["device_dispatch_calls"] == 0


def test_mt2_repeated_source_only_artifacts_are_attempt_path_independent(tmp_path):
    registry = write_mt2_registry(tmp_path / "tracks.json")
    spec = _spec(registry)
    candidate = tilelang_candidate(worklist=False)
    outcomes = []
    try:
        for _ in range(2):
            outcomes.append(
                TileLangSourceOnlyBuild(spec, attempt_base=tmp_path).build(
                    candidate, expected_kernel_names=("scale_kernel_kernel",)
                )
            )
    except contract.TileLangFrontendUnavailable as exc:
        assert exc.code == "backend_capability_unavailable"
        return
    first, second = outcomes
    assert first.attempt_root != second.attempt_root
    assert first.identity.artifact_set_sha256 == second.identity.artifact_set_sha256
    assert first.identity.cache_key == second.identity.cache_key
    first_ir = (first.attempt_root / "build/artifacts/device_ir.json").read_bytes()
    second_ir = (second.attempt_root / "build/artifacts/device_ir.json").read_bytes()
    assert first_ir == second_ir
    assert str(first.attempt_root).encode() not in first_ir
    assert str(second.attempt_root).encode() not in second_ir
    # The terminal attempt is fresh evidence, not part of the deterministic artifact bytes.
    assert first.identity.digest != second.identity.digest
