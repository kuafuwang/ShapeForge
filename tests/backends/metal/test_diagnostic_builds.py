"""Diagnostic builds (probes, calibration, internal fill) go through the registered factory. CPU.

These are CPU-only: no device is opened and no pipeline is created.  They pin the *call shape* —
``backend.artifact_builder(diagnostic_profile=...)`` then ``build`` once, with
``deadline_monotonic=None`` and raw-source assemble (no contract header) — while the real
pipeline behaviour is covered by the ``gpu`` cases in ``test_build_path_gpu.py``.

They live in the Metal layer because they import ``tcenv.backends.metal``
(``tests/shared`` may not, see ``tests/shared/test_structure.py``).
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TCENV = ROOT / "tcenv"

from tests.support.v1_tasks import V1_TASK, v1_kernel_name

V1_KERNEL = v1_kernel_name()


def test_diagnostic_profiles_are_registered_and_refuse_unknown_names():
    """@brief 诊断构建只用**登记**的 profile 名：未知名直接拒绝，不能由调用方拼选项。"""
    import pytest

    from tcenv.backends.metal.source_builder import (
        DIAGNOSTIC_BUILD_PROFILES,
        DiagnosticContract,
        MetalSourceBuilder,
    )

    assert {"metal.raw-source-probe", "metal.raw-source-calibration", "metal.internal-fill"} <= set(
        DIAGNOSTIC_BUILD_PROFILES
    )
    for name, entry in DIAGNOSTIC_BUILD_PROFILES.items():
        assert entry["assemble"] == "raw_source", name
        assert entry["math"] in ("fast", "safe", "relaxed"), name
    with pytest.raises(ValueError, match="unregistered diagnostic build profile"):
        MetalSourceBuilder(backend=object(), diagnostic_profile="metal.not-registered")
    # The diagnostic contract is not a task contract: it assembles raw source with no header.
    contract = DiagnosticContract(kernel_name="probe_kernel")
    assert contract.assemble("kernel void probe_kernel() {}\n") == ("kernel void probe_kernel() {}\n", 0)
    assert contract.render_header() == ""


def test_the_runtime_never_compiles_its_own_fill_kernel():
    """@brief 内部清零核必须由组合根注入；runtime 里没有 ``self.compile`` 的惰性路径。"""
    runtime = ast.parse((TCENV / "backends" / "metal" / "runtime.py").read_text(encoding="utf-8"))
    fill = next(
        node
        for node in ast.walk(runtime)
        if isinstance(node, ast.FunctionDef) and node.name == "_fill_kernel"
    )
    called = {node.func.attr for node in ast.walk(fill) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "compile" not in called, "_fill_kernel must not compile; the composition root injects it"
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "inject_fill_kernel"
        for node in ast.walk(runtime)
    )


def test_the_composition_roots_inject_the_internal_kernel_through_the_factory():
    """@brief evaluate/xcworker 在拿到 context 后立刻经工厂注入内部核。"""
    for relative in ("tcenv/evaluate.py", "tcenv/xcworker.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert '"prepare_context"' in text, relative
        assert text.index("prepare_context") < text.index("artifact_builder("), relative


def test_the_bandwidth_calibration_is_one_outcome_through_the_factory(monkeypatch):
    """@brief 带宽标定：一次 ``build`` 得到 copy/read 两个 pipeline，并只用 outcome 的 handle。

    不碰 GPU：上下文与 builder 都是替身，验证的是**调用形状**（工厂 → build → handles），
    真实 pipeline 由 ``tests/backends/metal/`` 的 GPU 用例覆盖。
    """
    from tcenv.backends.metal import profile as profile_module

    calls: list[dict] = []

    class FakeKernel:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeOutcome:
        def __init__(self, names) -> None:
            self._handles = tuple(FakeKernel(name) for name in names)

        def executable_handles(self):
            return self._handles

    class FakeBuilder:
        def __init__(self, names) -> None:
            self.names = names

        def build(self, source_bundle, *, task_contract, deadline_monotonic=None):
            calls.append(
                {
                    "sha256": source_bundle.sha256,
                    "bytes": source_bundle.source,
                    "kernel_name": task_contract.kernel_name,
                    "deadline": deadline_monotonic,
                }
            )
            return FakeOutcome(self.names)

    class FakeBackend:
        def artifact_builder(self, **kwargs):
            calls.append({"factory": kwargs})
            return FakeBuilder(kwargs["kernel_names"])

    class FakeTiming:
        gpu_per_iter_s = 1e-3

    class FakeContext:
        def new_buffer(self, nbytes):
            return object()

        def view(self, buffer):
            class _View:
                def __setitem__(self, key, value):
                    pass

            return _View()

        def run(self, kernel, args, grid, threads, iters=1):
            return FakeTiming()

    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: FakeBackend())
    out = profile_module.measure_peak_bandwidth(ctx=FakeContext(), nbytes=4096, iters=1)

    assert set(out) == {"copy4_gbps", "read4_gbps"}
    builds = [call for call in calls if "sha256" in call]
    factories = [call for call in calls if "factory" in call]
    assert len(builds) == 1, "calibration must be one build outcome, not one per kernel"
    assert len(factories) == 1
    assert factories[0]["factory"]["diagnostic_profile"] == "metal.raw-source-calibration"
    assert factories[0]["factory"]["kernel_names"] == ("copy4", "read4")
    assert builds[0]["deadline"] is None, "a local build passes deadline_monotonic=None"
    assert builds[0]["kernel_name"] == "copy4"
    assert builds[0]["bytes"] == profile_module._COPY_SRC.encode("utf-8")


def test_a_candidate_compile_failure_carries_the_projection_attachments():
    """@brief §5.1：候选 compile 失败带 typed provenance（无 artifact set）+ assembled 摘要。

    这里用 FakeNpu 的确定性失败路径 + Metal builder 的**装配前**失败路径（非 UTF-8 字节），
    两者都在 CPU 上可跑；真实 Metal 编译器拒绝内核的附件由 GPU 用例覆盖。
    """
    import hashlib

    import pytest

    from tcenv.remote.artifact_builder import (
        BuildError,
        SourceBundle,
        require_build_error_projection,
    )
    from tests.support.npu_sim.fake_backend import FakeNpuBackend

    backend = FakeNpuBackend()
    builder = backend.artifact_builder(kernel_contract=None)
    contract = type("C", (), {"kernel_name": "npu_main"})()
    with pytest.raises(BuildError) as excinfo:
        builder.build(SourceBundle(b"COMPILE_FAIL npu-kernel npu_main\n"), task_contract=contract)
    error = excinfo.value
    require_build_error_projection(error)
    assert error.error_class == "compile"
    assert error.provenance is not None and error.provenance.artifact_set_digest is None
    assert error.provenance.source_sha256 == hashlib.sha256(b"COMPILE_FAIL npu-kernel npu_main\n").hexdigest()
    assert error.assembled_sha256 == error.provenance.source_sha256

    # A failure that never assembled (non-UTF-8 bytes at the builder API) keeps the digest null.
    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.backends.metal.source_builder import DiagnosticContract

    metal = MetalBackend()
    metal_builder = metal.artifact_builder(
        kernel_contract=DiagnosticContract(kernel_name="probe"),
        diagnostic_profile="metal.raw-source-probe",
    )
    metal_builder._device_info = lambda: {
        "name": "test", "gpu_family": "test", "language_version": "3.2", "os_version": "test",
    }
    metal_builder._profile_for = lambda info: metal_builder.__class__.__mro__  # not used: patched below
    with pytest.raises(BuildError) as non_text:
        metal_builder.build(
            SourceBundle(b"\xff\xfe not utf-8"), task_contract=DiagnosticContract(kernel_name="probe")
        )
    # Either the toolchain probe refuses first (no device on this host) or the compile error
    # carries provenance with a null assembled digest; both are infrastructure-vs-candidate
    # distinctions the conformance check must accept.
    if non_text.value.error_class == "compile":
        require_build_error_projection(non_text.value)
        assert non_text.value.assembled_sha256 is None
    else:
        assert non_text.value.error_class in ("toolchain_profile", "rejected")


def test_the_metal_builder_attaches_provenance_to_a_compile_failure(monkeypatch):
    """@brief Metal builder 的 compile 失败分支：附件与报告文本都来自同一次构建（CPU 替身）。

    不需要 GPU：把设备探测与 runtime 编译替换成替身，只验证 builder 自己的失败分支。真实编译器
    的诊断与行号重写由 ``tests/backends/metal/test_build_path_gpu.py`` 的 gpu 用例覆盖。
    """
    import hashlib

    import pytest

    from tasks import get_task
    from tcenv.remote.artifact_builder import BuildError, SourceBundle, require_build_error_projection
    from tcenv.backends.metal.backend import MetalBackend

    backend = MetalBackend()
    task = get_task(V1_TASK, "fp16", "small")
    contract = task.contract
    builder = backend.artifact_builder(kernel_contract=contract, specialization=task.specialization)
    assert builder.lowering_identity == {
        "opspec_digest": task.specialization.opspec_digest,
        "lowering_profile_digest": task.specialization.lowering_profile_digest,
    }
    info = {"name": "fake-device", "gpu_family": "fake", "language_version": "3.2"}
    monkeypatch.setattr(builder, "_device_info", lambda: info)
    monkeypatch.setattr(builder, "_profile_for", lambda _info: builder.probe().profile)
    monkeypatch.setattr(builder, "_check_probed_identity", lambda _profile: None)
    monkeypatch.setattr(builder, "probe", lambda: type(
        "Caps", (), {"profile": type("P", (), {"toolchain_version": "3.2", "to_dict": lambda self: {}})()}
    )())
    monkeypatch.setattr(backend, "context", lambda: object())
    error_class = backend.compile_error()

    def fail(_context, _assembled, _names):
        raise error_class("program_source:3: error: use of undeclared identifier 'x'")

    monkeypatch.setattr(builder, "_compile_assembled", fail)
    source = task.naive_msl.encode("utf-8")
    with pytest.raises(BuildError) as excinfo:
        builder.build(SourceBundle(source), task_contract=contract, deadline_monotonic=None)
    error = excinfo.value
    require_build_error_projection(error)
    assert error.error_class == "compile"
    assert error.provenance is not None
    assert error.provenance.source_sha256 == hashlib.sha256(source).hexdigest()
    assert error.provenance.artifact_set_digest is None
    assembled, _header = contract.assemble(source.decode("utf-8"))
    assert error.assembled_sha256 == hashlib.sha256(assembled.encode("utf-8")).hexdigest()
    assert error.assembled_sha256 != hashlib.sha256(source).hexdigest(), "the header is part of the assembly"
    assert error.duration_ms >= 1, "a build that ran is never reported as duration 0"
    assert "undeclared identifier" in error.report_detail


def test_a_source_only_or_non_compile_failure_is_not_required_to_carry_the_projection():
    """@brief 只有 ``compile`` 类要求附件；``toolchain_profile``/``source_only`` 不被误判。"""
    from tcenv.remote.artifact_builder import BuildError, require_build_error_projection

    from tcenv.remote.artifact_builder import BUILD_ERROR_CLASSES

    assert set(BUILD_ERROR_CLASSES) == {"compile", "deadline", "toolchain_profile"}
    for error_class in ("toolchain_profile", "deadline"):
        require_build_error_projection(BuildError(error_class, "not the candidate's fault"))


def test_the_typed_probe_builds_through_the_factory_and_never_compiles_itself(monkeypatch):
    """@brief typed probe：工厂 → build → outcome handle；raw-source profile，deadline=None。

    CPU-only: the device, the lock and the runtime are doubles, so what is pinned here is the call
    shape, not a measurement.  A real probe run needs the GPU and is out of scope by authorization.
    """
    from tcenv.backends.metal import probes as probes_module

    calls: list[dict] = []

    class FakeKernel:
        name = "probe_kernel"
        max_total_threads = 1024
        bindings = (type("B", (), {"kind": "buffer", "index": 0, "name": "p"})(),)

    class FakeOutcome:
        def executable_handles(self):
            return (FakeKernel(),)

    class FakeBuilder:
        def build(self, source_bundle, *, task_contract, deadline_monotonic=None):
            calls.append(
                {
                    "sha256": source_bundle.sha256,
                    "bytes": source_bundle.source,
                    "kernel_name": task_contract.kernel_name,
                    "deadline": deadline_monotonic,
                }
            )
            return FakeOutcome()

    class FakeBackend:
        def artifact_builder(self, **kwargs):
            calls.append({"factory": kwargs})
            return FakeBuilder()

    class FakeTiming:
        wall_s = 0.002

    class FakeContext:
        def info(self):
            return {"name": "Fake Metal", "gpu_family": "fake"}

        def new_buffer(self, nbytes):
            return object()

        def run(self, kernel, args, grid, threads, iters=1):
            return FakeTiming()

    class FakeTarget:
        backend = "metal"
        id = "apple-m4-max"

    class Manifest:
        tool_id = "metal.time_msl"
        version = "1"
        digest = "0" * 64
        handler_source_sha256 = "1" * 64

    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: FakeBackend())
    monkeypatch.setattr("tcenv.backends.metal.runtime.get_context", lambda: FakeContext())
    monkeypatch.setattr(probes_module, "gpu_lock", lambda *a, **k: __import__("contextlib").nullcontext())
    monkeypatch.setattr("tcenv.backends.metal.screen.screen_source", lambda source: [])
    monkeypatch.setattr("tcenv.target.require_active", lambda target_id: FakeTarget())
    monkeypatch.setattr("tcenv.target.device_matches", lambda target, info: True)

    request_source = (
        "kernel void probe_kernel(device uint* p [[buffer(0)]], uint i [[thread_position_in_grid]]) {}\n"
    )
    result = probes_module.time_msl(
        {
            "target_id": "apple-m4-max",
            "source": request_source,
            "kernel_name": "probe_kernel",
            "grid": [1, 1, 1],
            "threads": [1, 1, 1],
            "buffer_bytes": 4096,
            "iters": 2,
        },
        manifest=Manifest(),
    )
    assert result["status"] == "ok", result["error"]
    factories = [call for call in calls if "factory" in call]
    builds = [call for call in calls if "sha256" in call]
    assert len(factories) == 1 and len(builds) == 1
    assert factories[0]["factory"]["diagnostic_profile"] == "metal.raw-source-probe"
    assert builds[0]["kernel_name"] == "probe_kernel"
    assert builds[0]["deadline"] is None
    # The probe's compilation unit is exactly the request bytes (no contract header).
    assert builds[0]["bytes"] == request_source.encode("utf-8")


def test_a_candidate_build_is_bound_to_the_specialization(monkeypatch):
    """@brief §5.1/§5.3-2：候选工厂收到 Specialization，契约与 lowering 身份同源。

    负例在 ``build`` 真正触碰设备之前就被拒绝：缺 Specialization 是组合根错误
    （``toolchain_profile``，不归责候选），契约不是那份 Specialization 的契约同样拒绝。
    """
    import pytest

    from tasks import get_task
    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.remote.artifact_builder import BuildError, SourceBundle

    backend = MetalBackend()
    task = get_task(V1_TASK, "fp16", "small")
    other = get_task("sem_int_5", "fp16", "small")

    # Positive: the factory carries the very Specialization the evaluator lowered.
    builder = backend.artifact_builder(kernel_contract=task.contract, specialization=task.specialization)
    assert builder.specialization is task.specialization
    assert builder.lowering_identity == {
        "opspec_digest": task.specialization.opspec_digest,
        "lowering_profile_digest": task.specialization.lowering_profile_digest,
    }
    # A diagnostic build has no task, so it must not carry a Specialization.
    with pytest.raises(ValueError, match="not both"):
        backend.artifact_builder(
            specialization=task.specialization, diagnostic_profile="metal.raw-source-probe"
        )
    # Two sources for the contract are refused at construction.
    with pytest.raises(ValueError, match="one source"):
        backend.artifact_builder(kernel_contract=other.contract, specialization=task.specialization)

    # A contract-only builder (synthetic build-contract tests) has no lowering identity, so it
    # cannot claim a task identity it does not have.
    bare = backend.artifact_builder(kernel_contract=task.contract)
    assert bare.lowering_identity is None

    # Negative: the build contract is not the Specialization's contract.
    smuggled = backend.artifact_builder(specialization=task.specialization)
    with pytest.raises(BuildError) as mismatch:
        smuggled.build(SourceBundle(task.naive_msl.encode("utf-8")), task_contract=other.contract)
    assert mismatch.value.error_class == "toolchain_profile"
    assert "two sources" in mismatch.value.detail


def test_the_candidate_call_sites_pass_the_specialization():
    """@brief evaluate / xcworker 的候选工厂调用必须带 ``specialization=task.specialization``。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    for relative in ("tcenv/evaluate.py", "tcenv/xcworker.py"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "specialization=task.specialization" in text, relative


def test_every_registered_candidate_builder_carries_the_projection_on_compile_failure():
    """@brief §5.1 的 conformance：**每个**登记候选 builder 的 compile 失败都带齐附件。

    逐个覆盖：生产 Metal builder（设备/编译用替身，CPU）、`tests/support` 的 FakeNpu 与 FakeTsm
    （模拟 builder 也不可豁免）。``source_only`` 从不编译，故不在此列。
    """
    import pytest

    from tests.support.npu_sim.fake_backend import FakeNpuBackend
    from tests.support.vendor_sim import module_builder as mb
    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.remote.artifact_builder import (
        BuildError,
        SourceBundle,
        require_build_error_projection,
    )

    errors: list[BuildError] = []

    # 1) FakeNpu: a simulated candidate whose compiler rejects the kernel.
    class _Contract:
        kernel_name = "npu_main"

    with pytest.raises(BuildError) as npu:
        FakeNpuBackend().artifact_builder(kernel_contract=None).build(
            SourceBundle(b"COMPILE_FAIL npu-kernel npu_main\n"), task_contract=_Contract()
        )
    errors.append(npu.value)

    # 2) FakeTsm: the vendor-sim builder used by the remote build contract tests.
    tsm = mb.FakeTsmModuleBuilder(mb.evaluator_profile())
    with pytest.raises(BuildError) as vendor:
        tsm.build(
            SourceBundle(b"TXCC_FAIL\nnpu-kernel npu_main\n"),
            task_contract={"kernel_name": "npu_main"},
        )
    errors.append(vendor.value)

    # 3) Production Metal builder: real failure branch, device/compiler replaced by doubles.
    #    A candidate build is bound to the task's Specialization (as in the evaluator).
    from tasks import get_task

    task = get_task(V1_TASK, "fp16", "small")
    backend = MetalBackend()
    metal_builder = backend.artifact_builder(
        kernel_contract=task.contract, specialization=task.specialization
    )
    metal_builder._device_info = lambda: {"name": "fake", "gpu_family": "fake", "language_version": "3.2"}
    metal_builder.probe = lambda: type(
        "Caps", (), {"profile": type("P", (), {"toolchain_version": "3.2", "to_dict": lambda self: {}})()}
    )()
    metal_builder._profile_for = lambda _info: metal_builder.probe().profile
    metal_builder._check_probed_identity = lambda _profile: None
    backend.context = lambda: object()
    error_class = backend.compile_error()
    metal_builder._compile_assembled = lambda *_a: (_ for _ in ()).throw(error_class("program_source:1:1: error"))
    with pytest.raises(BuildError) as metal:
        metal_builder.build(
            SourceBundle(task.naive_msl.encode("utf-8")), task_contract=task.contract
        )
    errors.append(metal.value)

    assert len(errors) == 3
    for error in errors:
        assert error.error_class == "compile", error.detail
        require_build_error_projection(error)  # raises if anything is missing
        assert error.provenance is not None and error.provenance.artifact_set_digest is None
        assert error.assembled_sha256 is not None


def test_the_evaluator_refuses_to_blame_the_candidate_for_an_unprojectable_compile_error(
    monkeypatch, tmp_path
):
    """@brief 报告映射前强制检查：附件缺失的 ``compile`` 失败落成基础设施失败，不是 compile_error。"""
    import pytest

    from tcenv import evaluate as ev
    from tcenv.evaluation import EvaluationError
    from tcenv.remote.artifact_builder import BuildError
    from tests.support.npu_sim.fake_backend import FakeNpuBackend

    backend = FakeNpuBackend()
    from dataclasses import replace

    from tcenv.target import load_target

    profile = replace(load_target("apple-m4-max"), backend="sim-npu", id="sim-npu-test", backend_version="1")
    monkeypatch.setattr("tcenv.target.require_active", lambda target_id: profile)
    monkeypatch.setattr("tcenv.target.load_target", lambda target_id: profile)
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: backend)
    monkeypatch.setenv("TCENV_GPU_LOCK_FILE", str(tmp_path / "device.lock"))

    real_builder = backend.artifact_builder

    def bare_builder(**kwargs):
        builder = real_builder(**kwargs)

        def build(source_bundle, *, task_contract, deadline_monotonic=None):
            # No provenance, no assembled digest: not projectable.
            raise BuildError("compile", "the fake compiler rejected the kernel")

        builder.build = build
        return builder

    monkeypatch.setattr(backend, "artifact_builder", bare_builder)
    with pytest.raises(BuildError) as excinfo:
        ev._evaluate_in_process(
            V1_TASK, "fp16", "small", f"npu-kernel {V1_KERNEL}\n",
            mode="quick", seed=7, reps=1, measure_serial=False, target_id=profile.id,
        )
    # Infrastructure, not the candidate: the class changed and the reason says why.
    assert excinfo.value.error_class == "toolchain_profile"
    assert "not projectable" in excinfo.value.detail


def test_the_evaluator_hands_the_task_specialization_to_the_factory(monkeypatch, tmp_path):
    """@brief 行为证据：评测器把 ``task.specialization`` 交给候选工厂（CPU，FakeNpu）。

    这比字符串检查强：记录的是工厂实际收到的关键字参数，且与同一次 build 的 contract 同源。
    """
    from dataclasses import replace

    from tcenv import evaluate as ev
    from tcenv.target import load_target
    from tests.support.npu_sim.fake_backend import FakeNpuBackend

    backend = FakeNpuBackend()
    profile = replace(load_target("apple-m4-max"), backend="sim-npu", id="sim-npu-test", backend_version="1")
    monkeypatch.setattr("tcenv.target.require_active", lambda target_id: profile)
    monkeypatch.setattr("tcenv.target.load_target", lambda target_id: profile)
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: backend)
    monkeypatch.setenv("TCENV_GPU_LOCK_FILE", str(tmp_path / "device.lock"))

    seen: list[dict] = []
    real = backend.artifact_builder

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(backend, "artifact_builder", spy)
    report = ev._evaluate_in_process(
        V1_TASK, "fp16", "small", f"npu-kernel {V1_KERNEL}\n",
        mode="quick", seed=7, reps=1, measure_serial=False, target_id=profile.id,
    )
    assert report.status == "ok", report.message
    candidate = [call for call in seen if not call.get("diagnostic_profile")]
    assert candidate, seen
    from tasks import get_task

    task = get_task(V1_TASK, "fp16", "small")
    assert candidate[0]["specialization"] is task.specialization
    assert candidate[0]["kernel_contract"] is task.contract


def _fake_npu_evaluator(monkeypatch, tmp_path, error_factory):
    """@brief 在 FakeNpu 上装一个按需失败的候选 builder，并返回 ``_evaluate_in_process``。

    @param monkeypatch pytest 的 monkeypatch。
    @param tmp_path 临时目录（GPU 锁文件）。
    @param error_factory 接收 builder 的闭包，返回要抛出的 ``BuildError``。
    @return ``_evaluate_in_process`` 函数。
    """
    from dataclasses import replace

    from tcenv import evaluate as ev
    from tcenv.target import load_target
    from tests.support.npu_sim.fake_backend import FakeNpuBackend

    backend = FakeNpuBackend()
    profile = replace(load_target("apple-m4-max"), backend="sim-npu", id="sim-npu-test", backend_version="1")
    monkeypatch.setattr("tcenv.target.require_active", lambda target_id: profile)
    monkeypatch.setattr("tcenv.target.load_target", lambda target_id: profile)
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: backend)
    monkeypatch.setenv("TCENV_GPU_LOCK_FILE", str(tmp_path / "device.lock"))
    real = backend.artifact_builder

    def failing(**kwargs):
        builder = real(**kwargs)

        def build(source_bundle, *, task_contract, deadline_monotonic=None):
            raise error_factory(builder)

        builder.build = build
        return builder

    monkeypatch.setattr(backend, "artifact_builder", failing)
    return ev._evaluate_in_process, profile


def _provenance_for(builder):
    """@brief 造一份合法的失败 provenance（与成功路径同一类型、无 artifact set）。

    @param builder 候选 builder（提供 build profile）。
    @return ``BuildProvenance``。
    """
    from tcenv.remote.artifact_builder import BuildProvenance

    return BuildProvenance(
        profile=builder.probe().profile, source_sha256="a" * 64, artifact_set_digest=None,
    )


def test_a_compile_failure_without_an_assembled_digest_is_not_blamed_on_the_candidate(
    monkeypatch, tmp_path
):
    """@brief 合法 provenance + **null** assembled digest：候选报告映射必须判为基础设施失败。

    这条路径的源码从 Python ``str`` 编码而来（`_evaluate_in_process` 自己 encode），因此
    ``assembled_sha256=None`` 只可能是 builder 缺陷；把它当成候选 ``compile_error`` 会伪造一份
    §6.2 的 partial 投影。``require_build_error_projection(..., require_assembled=True)`` 负责拦住它。
    """
    import pytest

    from tcenv.remote.artifact_builder import BuildError

    def error(builder):
        return BuildError(
            "compile", "the fake compiler rejected the kernel",
            provenance=_provenance_for(builder), assembled_sha256=None, duration_ms=1,
        )

    evaluate_in_process, profile = _fake_npu_evaluator(monkeypatch, tmp_path, error)
    with pytest.raises(BuildError) as excinfo:
        evaluate_in_process(
            V1_TASK, "fp16", "small", f"npu-kernel {V1_KERNEL}\n",
            mode="quick", seed=7, reps=1, measure_serial=False, target_id=profile.id,
        )
    assert excinfo.value.error_class == "toolchain_profile"
    assert "not projectable" in excinfo.value.detail
    assert "assembled_sha256" in excinfo.value.detail


def test_a_compile_failure_with_an_assembled_digest_is_a_candidate_compile_error(
    monkeypatch, tmp_path
):
    """@brief 对照：同一失败带上非空 assembled digest 时，仍映射为候选 ``compile_error``。"""
    from tcenv.remote.artifact_builder import BuildError

    def error(builder):
        return BuildError(
            "compile", "the fake compiler rejected the kernel",
            report_detail="sim-npu: kernel rejected at line 1",
            provenance=_provenance_for(builder), assembled_sha256="b" * 64, duration_ms=1,
        )

    evaluate_in_process, profile = _fake_npu_evaluator(monkeypatch, tmp_path, error)
    report = evaluate_in_process(
        V1_TASK, "fp16", "small", f"npu-kernel {V1_KERNEL}\n",
        mode="quick", seed=7, reps=1, measure_serial=False, target_id=profile.id,
    )
    assert (report.status, report.stage) == ("compile_error", "compile")
    assert report.compile == {"ok": False, "errors": "sim-npu: kernel rejected at line 1"}


def test_the_projection_check_keeps_the_null_digest_legal_at_the_builder_api():
    """@brief 直接 API 的非 UTF-8 预装配失败仍然合法：默认模式接受 ``assembled_sha256=None``。"""
    import pytest

    from tcenv.remote.artifact_builder import (
        BuildError,
        BuildProvenance,
        require_build_error_projection,
    )
    from tests.support.vendor_sim import module_builder as mb

    profile = mb.evaluator_profile()
    error = BuildError(
        "compile", "source is not UTF-8 text",
        provenance=BuildProvenance(profile=profile, source_sha256="c" * 64, artifact_set_digest=None),
        assembled_sha256=None,
    )
    require_build_error_projection(error)  # legal: nothing was assembled
    with pytest.raises(ValueError, match="assembled_sha256"):
        require_build_error_projection(error, require_assembled=True)


def test_prepare_context_builds_the_internal_fill_through_the_factory_exactly_once():
    """@brief 组合根行为 spy（CPU）：未注入时工厂+build 各一次，注入同一个 handle；已注入时不再构建。

    这是 §5.2/§5.3-2 对“内部 fill 也经工厂 → build”的行为证据，不是文本顺序检查：伪造的 builder
    与伪造的 context 记录实际收到的参数和注入的对象。
    """
    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.backends.metal.runtime import MetalContext

    kernel = object()  # the live pipeline handle the builder would return

    class FakeOutcome:
        def executable_handles(self):
            return (kernel,)

    class FakeBuilder:
        def __init__(self) -> None:
            self.builds: list[dict] = []

        def build(self, source_bundle, *, task_contract, deadline_monotonic=None):
            self.builds.append(
                {
                    "bytes": source_bundle.source,
                    "sha256": source_bundle.sha256,
                    "kernel_name": task_contract.kernel_name,
                    "deadline": deadline_monotonic,
                }
            )
            return FakeOutcome()

    class FakeContext:
        # The production constants: the assertion below must compare the built bytes with
        # ``MetalContext.FILL_KERNEL_SOURCE``, not with a copy this fake carries (that would be
        # self-证: any source would pass as long as the fake agrees with itself).
        FILL_KERNEL_NAME = MetalContext.FILL_KERNEL_NAME
        FILL_KERNEL_SOURCE = MetalContext.FILL_KERNEL_SOURCE

        def __init__(self) -> None:
            self._fill = None
            self.injected: list[object] = []

        def inject_fill_kernel(self, handle) -> None:
            self.injected.append(handle)
            self._fill = handle

    backend = MetalBackend()
    factory_calls: list[dict] = []
    builder = FakeBuilder()

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return builder

    backend.artifact_builder = factory  # type: ignore[method-assign]

    context = FakeContext()
    backend.prepare_context(context)

    # Exactly one factory call and exactly one build, with the registered diagnostic profile.
    assert len(factory_calls) == 1, factory_calls
    assert factory_calls[0]["diagnostic_profile"] == "metal.internal-fill"
    assert factory_calls[0]["kernel_contract"].kernel_name == MetalContext.FILL_KERNEL_NAME
    assert factory_calls[0].get("specialization") is None
    assert len(builder.builds) == 1, builder.builds
    build = builder.builds[0]
    assert build["bytes"] == MetalContext.FILL_KERNEL_SOURCE.encode("utf-8"), "the production fixed source"
    assert build["kernel_name"] == MetalContext.FILL_KERNEL_NAME
    assert build["deadline"] is None, "a local build passes deadline_monotonic=None"
    # The injected handle is the outcome's handle (one outcome, one handle).
    assert context.injected == [kernel]
    assert context._fill is kernel

    # Idempotent: an already-injected context is not rebuilt and does not ask the factory again.
    backend.prepare_context(context)
    assert len(factory_calls) == 1, "prepare_context rebuilt an injected fill kernel"
    assert len(builder.builds) == 1
    assert context.injected == [kernel]

    # A context that cannot accept an injection has no internal kernels (test double).
    backend.prepare_context(object())
    assert len(factory_calls) == 1


def test_the_spied_backend_must_be_the_one_the_evaluator_obtains(monkeypatch, tmp_path):
    """@brief 接线自检（CPU）：spy 必须装在 ``registry.load_backend`` 返回的**同一个实例**上。

    ``load_backend`` 每次都新建实例，所以给一个自己 new 的 backend 装 spy 会记录不到任何东西（真机上
    表现为 candidate 记录为空、fresh context 的 diagnostic 计数也对不上）。本用例先复现那个失败形态，
    再证明“把 registry 指向被 spy 的实例”之后记录才出现。
    """
    from dataclasses import replace

    from tcenv import evaluate as ev
    from tcenv.target import load_target
    from tests.support.npu_sim.fake_backend import FakeNpuBackend

    profile = replace(load_target("apple-m4-max"), backend="sim-npu", id="sim-npu-test", backend_version="1")
    monkeypatch.setattr("tcenv.target.require_active", lambda target_id: profile)
    monkeypatch.setattr("tcenv.target.load_target", lambda target_id: profile)
    monkeypatch.setenv("TCENV_GPU_LOCK_FILE", str(tmp_path / "device.lock"))

    def run(handed_out):
        seen: list[str] = []
        monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: handed_out)

        real = handed_out.artifact_builder

        def spy(**kwargs):
            seen.append("factory")
            return real(**kwargs)

        monkeypatch.setattr(handed_out, "artifact_builder", spy)
        report = ev._evaluate_in_process(
            V1_TASK, "fp16", "small", f"npu-kernel {V1_KERNEL}\n",
            mode="quick", seed=7, reps=1, measure_serial=False, target_id=profile.id,
        )
        assert report.status == "ok", report.message
        return seen

    # The failure mode: the recorder sits on an instance the evaluator never gets.
    assert run(FakeNpuBackend()) == ["factory"]  # control: the handed-out instance is spied

    visible = FakeNpuBackend()
    hidden = FakeNpuBackend()
    seen_on_hidden: list[str] = []
    real_hidden = hidden.artifact_builder

    def hidden_spy(**kwargs):
        seen_on_hidden.append("factory")
        return real_hidden(**kwargs)

    monkeypatch.setattr(hidden, "artifact_builder", hidden_spy)
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: visible)
    report = ev._evaluate_in_process(
        V1_TASK, "fp16", "small", f"npu-kernel {V1_KERNEL}\n",
        mode="quick", seed=7, reps=1, measure_serial=False, target_id=profile.id,
    )
    assert report.status == "ok", report.message
    assert seen_on_hidden == [], "a spy on an instance the evaluator never receives cannot record"
