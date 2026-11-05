"""A missing native Host Program toolchain is a typed environment outcome, never a worker crash.

The native C path needs standard Clang and `nm`.  When Clang is absent:

* ``MetalTrackBuilder.build`` fails before compiling MSL with a
  ``backend_capability_unavailable`` typed error (not a candidate ``compile`` error);
* the v2 worker projects that into a build-free ``rejected`` report with the code in its message;
* the linear ``host/tiling.json`` path never calls the host toolchain and still builds;
* an executor cache miss that must recompile the module maps the same refusal into a
  ``rejected`` report (the already-built full record is preserved), not an exception.

These nodes exercise the real builder/worker result boundary; they do not dispatch a kernel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.track_adapter import (
    HostCapabilityBuildError,
    MetalTrackBuilder,
    MetalTrackEvaluator,
)
from tcenv.backends.metal.host_program import MetalHostProgram
from tcenv.hostprogram.errors import HostCapabilityUnavailable
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.taskcontract.runner import evaluate_task, select_track
from tcenv.target import load_target
from tcenv.track import CandidateLifecycle, HostStep, PrepareDeclaration, candidate_geometry
from tcenv.track.envelope import decode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry
from tcenv.v2_worker import capability_failure_result

from ...support.hostprogram_fixtures import PLANNER_SOURCE, kda_bundle_entries, kda_bundle_source

pytestmark = pytest.mark.gpu

TARGET = load_target("apple-m4-max")
TASK = REPO_ROOT / "tasks" / "varlen_kda"


class _StopAfterBuild(Exception):
    def __init__(self, request) -> None:
        self.request = request
        super().__init__("stop after build")


class _CaptureRequest:
    """@brief 不执行 device，只要求 evaluate_task 走到 build 之后。"""

    owns_build = False

    def evaluate(self, request):
        raise _StopAfterBuild(request)


def _linear_bundle_source() -> str:
    """@brief 同一 device/geometry，但 host 侧是既有 linear JSON（不是 C）。"""
    plan = {
        "schema_id": "tcenv.metal.host-tiling.1",
        "schema_version": 1,
        "writes": [
            {
                "workspace": "work_items",
                "kind": "linear",
                "start": 0,
                "step": 0,
                "count": 0,
                "stride": 0,
                "offset": 0,
                "dtype": "u32",
            }
        ],
    }
    entries = tuple(entry for entry in kda_bundle_entries() if entry.role != "host_tiling") + (
        CandidateEntry(
            path="host/tiling.json",
            role="host_tiling",
            media_type="application/json",
            data=json.dumps(plan).encode("utf-8"),
        ),
    )
    from tcenv.track.envelope import encode_envelope

    return encode_envelope(CandidateArtifact(track_id="msl-bundle", entries=entries))


def _lifecycle_factory(source: str, *, program: bool):
    contract = load_task_contract(TASK, dtype="fp16", size="mixed")
    candidate = decode_envelope(source)

    def factory(chosen):
        return CandidateLifecycle(
            track_id="msl-bundle",
            geometry=chosen,
            host_steps=(
                HostStep(
                    PrepareDeclaration(
                        name="host_tiling",
                        input_dependent=program,
                        preprocesses_inputs=program,
                        inside_timing=program,
                    ),
                    role="host_tiling",
                ),
                HostStep(
                    PrepareDeclaration(name="schedule", input_dependent=False, inside_timing=False),
                    role="schedule",
                ),
            ),
            candidate_sha256=candidate.candidate_sha256,
            target_id=TARGET.id,
            instance_key=contract.fingerprint(),
        )

    return factory


def _run_to_build(source: str, evaluator, *, program: bool):
    locator = bind_task("varlen_kda", "fp16", "mixed").locator
    candidate = decode_envelope(source)
    geometry = candidate_geometry(candidate)
    return evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate=candidate,
        builder=MetalTrackBuilder(target_id=TARGET.id),
        evaluator=evaluator,
        geometry_resolver=lambda _candidate: geometry,
        lifecycle_factory=_lifecycle_factory(source, program=program),
        target_id=TARGET.id,
    )


@pytest.fixture
def missing_toolchain(monkeypatch):
    """@brief 模拟 native 唯一必需的工具 clang 缺失。"""
    import tcenv.hostprogram.native_toolchain as toolchain

    real_which = toolchain.shutil.which
    real_is_file = Path.is_file
    monkeypatch.setattr(
        toolchain.shutil,
        "which",
        lambda name: None if name == "clang" else real_which(name),
    )
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: False if self.name == "clang" else real_is_file(self),
    )
    monkeypatch.delenv("SF_HOST_CLANG", raising=False)


def test_a_missing_host_toolchain_is_a_typed_build_refusal(missing_toolchain):
    """@brief C 路径在 build 边界得到 ``backend_capability_unavailable``，不是候选 compile。"""
    with pytest.raises(HostCapabilityBuildError) as caught:
        _run_to_build(kda_bundle_source(), _CaptureRequest(), program=True)
    error = caught.value
    assert error.error_class == "toolchain_profile"
    assert error.error_code == "backend_capability_unavailable"

    locator = bind_task("varlen_kda", "fp16", "mixed").locator
    selection = select_track("msl-bundle")
    resolved = {
        "mode": "quick",
        "seed": 7,
        "target_id": TARGET.id,
        "frozen_track": selection.frozen_document(presentation_sha256="a" * 64),
    }
    result = capability_failure_result(locator, resolved, error)
    assert result["report"]["status"] == "rejected"
    assert result["report"]["stage"] == "host_toolchain"
    assert "backend_capability_unavailable" in result["report"]["message"]
    assert "host_program_invalid" not in result["report"]["message"]
    assert result["build"] is None
    assert result["execution"]["provider"] == "bounded-v2-worker"
    assert result["execution"]["dispatch"] == []


def test_the_linear_json_path_does_not_need_the_host_toolchain(missing_toolchain):
    """@brief 同一缺依赖下，host/tiling.json 仍能构建（JSON linear 不受影响）。"""
    with pytest.raises(_StopAfterBuild) as caught:
        _run_to_build(_linear_bundle_source(), _CaptureRequest(), program=False)
    request = caught.value.request
    assert request.build is not None
    assert request.build.bundle.native is not None
    assert request.build.bundle.native.descriptor.get("host_program") is None


def test_an_executor_cache_miss_without_the_toolchain_is_a_rejected_report(monkeypatch):
    """@brief 执行期重编译缺依赖：``rejected`` + 已构建的完整 build，而不是异常/crash。"""

    def unavailable(*_args, **_kwargs):
        raise HostCapabilityUnavailable("native clang is missing in this executor process")

    monkeypatch.setattr(MetalHostProgram, "from_descriptor", unavailable)
    locator = bind_task("varlen_kda", "fp16", "mixed").locator
    candidate = decode_envelope(kda_bundle_source())
    geometry = candidate_geometry(candidate)
    evaluator = MetalTrackEvaluator(target_id=TARGET.id, mode="quick", cases=("canonical",), worker_child=True)
    result = evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate=candidate,
        builder=MetalTrackBuilder(target_id=TARGET.id),
        evaluator=evaluator,
        geometry_resolver=lambda _candidate: geometry,
        lifecycle_factory=_lifecycle_factory(kda_bundle_source(), program=True),
        target_id=TARGET.id,
    )
    assert result.report["status"] == "rejected"
    assert result.report["stage"] == "host_toolchain"
    assert "backend_capability_unavailable" in result.report["message"]
    assert result.build is not None
    assert result.execution["provider"] == "bounded-v2-worker"
    assert result.execution["dispatch"] == []


@pytest.mark.parametrize("module", ("tcenv.hostprogram.toolchain", "tcenv.hostprogram.runtime", "wasmtime"))
def test_native_host_program_does_not_import_wasm_path(module, monkeypatch):
    """@brief native build 不得 import 旧 Wasm toolchain/runtime/engine。"""
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == module:
            raise AssertionError(f"native build imported retired Wasm path {module}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(_StopAfterBuild) as caught:
        _run_to_build(kda_bundle_source(), _CaptureRequest(), program=True)
    descriptor = caught.value.request.build.bundle.native.descriptor["host_program"]
    assert descriptor["execution_backend"] == "native-c99"
    assert "native_sha256" in descriptor


def test_from_descriptor_recompiles_a_comment_only_identity_instead_of_reusing_the_wrong_build():
    """@brief A 已缓存、B 只有注释差异（同 native bytes、不同 assembled）时必须重编译 B。"""
    from tcenv.hostprogram.contract import host_contract_spec
    from tcenv.hostprogram.native_toolchain import compile_native_host_program, remember_native_build
    from tcenv.track.abi import derive_tensor_memory_contract

    contract = load_task_contract(TASK, dtype="fp16", size="mixed")
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
    build_a = compile_native_host_program(PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c")
    remember_native_build(build_a)
    source_b = PLANNER_SOURCE + "\n/* prebuilt B: comment-only identity */\n"
    build_b = compile_native_host_program(source_b, spec=spec, entry_path="host/tiling.c")
    assert build_a.native_sha256 == build_b.native_sha256
    assert build_a.assembled_sha256 != build_b.assembled_sha256

    candidate = CandidateArtifact(track_id="msl-bundle", entries=kda_bundle_entries(planner_source=source_b))
    program = MetalHostProgram.from_descriptor(
        candidate,
        instance=contract.instance,
        tensors=abi.tensors,
        target=TARGET,
        target_id=TARGET.id,
        descriptor=build_b.descriptor(),
        native_digest="a" * 64,
        inside_timing=True,
        input_dependent=True,
        preprocesses_inputs=True,
    )
    assert program is not None
    assert program.entry.sha256 == build_b.program_sha256
    assert program.build.assembled_sha256 == build_b.assembled_sha256
