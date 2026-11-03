"""Real-Mac checks for the single build path (GPU). Moved out of ``tests/shared`` in R3.

The structural checks (AST/call-graph, no GPU) live in
``tests/shared/test_build_path_structure.py``; the *behaviour* of the composition-root hook and of
the diagnostic builds (factory → build → handle, one outcome, `deadline_monotonic=None`) is pinned
on CPU in ``test_diagnostic_builds.py``.  These two cases only add what needs the real compiler: a
live pipeline for the candidate, the real compile error, and the diagnostic line rewrite.

Both record diagnostic and candidate builds **separately** and assert exact counts.  ``get_context()``
is a process singleton and ``prepare_context`` skips an already-injected fill kernel, so the
expected diagnostic count comes from the state *before* the call (0 or 1), never from a widened
tolerance: a ``prepare_context`` that never reaches the factory would fail these tests.
"""

from __future__ import annotations

import hashlib

import pytest


def _fill_already_injected() -> bool:
    """@brief 本次评测开始前，进程单例 context 是否已注入清零核。

    @return 已注入为 True。
    """
    from tcenv.backends.metal.runtime import get_context

    return getattr(get_context(), "_fill", None) is not None


def _install_spy(monkeypatch, diagnostic: list[dict], candidate: list[dict]):
    """@brief 在评测器**实际使用**的 backend 实例上记录两类 build。

    ``tcenv.evaluate`` 通过 ``registry.load_backend(profile.backend)`` 取 backend，而
    ``load_backend`` 每次都会新建实例。因此不能 spy 一个自己 new 出来的 ``MetalBackend()``：那个实例
    永远不会被评测器调用（记录为空，真机上会让 candidate 断言与 fresh-context 的 diagnostic 计数一起
    失败）。这里把 registry 的工厂指向要 spy 的实例，让“被 spy 的 backend”与“评测器拿到的 backend”是
    同一个对象。

    @param monkeypatch pytest 的 monkeypatch。
    @param diagnostic 诊断 build 记录列表。
    @param candidate 候选 build 记录列表。
    @return 被 spy 的 backend 实例（也是评测器将拿到的那个）。
    """
    from tcenv.backends.registry import load_backend

    backend = load_backend("metal")
    real = backend.artifact_builder

    def spy(**kwargs):
        builder = real(**kwargs)
        original = builder.build
        target = diagnostic if kwargs.get("diagnostic_profile") else candidate

        def build(source_bundle, **build_kwargs):
            target.append(
                {
                    "kwargs": kwargs,
                    "bytes": source_bundle.source,
                    "sha256": source_bundle.sha256,
                    "task_contract": build_kwargs.get("task_contract"),
                    "deadline": build_kwargs.get("deadline_monotonic"),
                }
            )
            return original(source_bundle, **build_kwargs)

        builder.build = build
        return builder

    monkeypatch.setattr(backend, "artifact_builder", spy)
    # The evaluator asks the registry, not this module: make the registry hand out the spied
    # instance so both sides provably share it.
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: backend)
    return backend


def _assert_internal_fill(diagnostic: list[dict], *, expected: int) -> None:
    """@brief 断言内部清零核的 build 次数与身份（组合根 → 工厂 → build）。

    @param diagnostic 诊断 build 记录。
    @param expected 期望次数（未注入时 1，已注入时 0）。
    """
    from tcenv.backends.metal.runtime import MetalContext

    assert len(diagnostic) == expected, diagnostic
    for record in diagnostic:
        assert record["kwargs"]["diagnostic_profile"] == "metal.internal-fill"
        assert record["kwargs"]["kernel_contract"].kernel_name == MetalContext.FILL_KERNEL_NAME
        assert record["kwargs"].get("specialization") is None, "a diagnostic build has no task"
        assert record["task_contract"].kernel_name == MetalContext.FILL_KERNEL_NAME
        assert record["bytes"] == MetalContext.FILL_KERNEL_SOURCE.encode("utf-8")
        assert record["deadline"] is None


@pytest.mark.gpu
def test_the_real_evaluator_builds_through_the_registered_builder(monkeypatch):
    """@brief 真实 Mac：评测器经登记的 builder 构建一次，candidate_sha256 仍是候选原文。"""
    from ...support.v1_tasks import V1_TASK, v1_task
    from tcenv.backends.metal.runtime import get_context
    from tcenv.evaluate import _evaluate_in_process

    task = v1_task()
    source = task.naive_msl
    # The composition root injects the runtime's internal kernel before the candidate is built, so
    # the factory is asked for two different things.  They are recorded and asserted separately.
    already_injected = _fill_already_injected()
    diagnostic: list[dict] = []
    candidate: list[dict] = []
    backend = _install_spy(monkeypatch, diagnostic, candidate)

    report = _evaluate_in_process(
        V1_TASK, "fp16", "small", source,
        mode="quick", seed=7, reps=2, measure_serial=False,
        target_id="apple-m4-max",
    )
    assert report.status == "ok", report.message

    # 1) The internal kernel: registered diagnostic profile, fixed source, one build when the
    #    context was fresh (and none when an earlier test already injected it).
    _assert_internal_fill(diagnostic, expected=0 if already_injected else 1)

    # 2) The candidate: exactly one build, bound to this task's Specialization and contract.
    assert len(candidate) == 1, candidate
    assert candidate[0]["kwargs"]["kernel_contract"] is task.contract
    assert candidate[0]["kwargs"]["specialization"] is task.specialization
    assert candidate[0]["kwargs"].get("diagnostic_profile") is None
    assert candidate[0]["task_contract"] is task.contract
    assert candidate[0]["kwargs"]["kernel_names"] is None
    assert candidate[0]["deadline"] is None
    assert candidate[0]["sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert candidate[0]["bytes"] == source.encode("utf-8")

    legacy = get_context().compile(
        task.contract.assemble(source)[0], task.contract.kernel_name, math="fast",
    )
    assert report.compile["ok"] is True
    assert report.compile["language_version"] == legacy.language_version
    assert report.compile["math_mode"] == "fast"
    assert report.compile["launch"] is None
    assert report.compile["compile_s"] > 0
    assert report.compile["pipeline_s"] > 0


@pytest.mark.gpu
def test_an_invalid_kernel_reaches_compile_error_through_the_builder(monkeypatch):
    """@brief 真实 Mac：无效 MSL 经 builder 变成候选 ``compile_error``，文本形状不变。"""
    from ...support.v1_tasks import V1_TASK, v1_task
    from tcenv.backends.metal.runtime import MetalCompileError, get_context
    from tcenv.evaluate import _evaluate_in_process

    task = v1_task()
    broken = task.naive_msl + "\nthis is not valid metal;\n"
    already_injected = _fill_already_injected()
    diagnostic: list[dict] = []
    candidate: list[dict] = []
    backend = _install_spy(monkeypatch, diagnostic, candidate)

    report = _evaluate_in_process(
        V1_TASK, "fp16", "small", broken,
        mode="quick", seed=7, reps=2, measure_serial=False,
        target_id="apple-m4-max",
    )
    assert report.status == "compile_error", report.message
    assert report.stage == "compile"

    # The internal fill kernel is still built through the factory (never inside the runtime), and
    # the broken candidate is built exactly once — a single shared counter would conflate them.
    _assert_internal_fill(diagnostic, expected=0 if already_injected else 1)
    assert len(candidate) == 1, candidate
    assert candidate[0]["kwargs"]["kernel_contract"] is task.contract
    assert candidate[0]["kwargs"]["specialization"] is task.specialization
    assert candidate[0]["kwargs"].get("diagnostic_profile") is None
    assert candidate[0]["kwargs"]["kernel_names"] is None
    assert candidate[0]["task_contract"] is task.contract
    assert candidate[0]["deadline"] is None
    assert candidate[0]["sha256"] == hashlib.sha256(broken.encode("utf-8")).hexdigest()
    assert candidate[0]["bytes"] == broken.encode("utf-8")

    assembled, header_lines = task.contract.assemble(broken)
    with pytest.raises(MetalCompileError) as legacy:
        get_context().compile(assembled, task.contract.kernel_name, math="fast")
    expected = task.contract.rewrite_compiler_lines(str(legacy.value), header_lines)
    assert report.compile["ok"] is False
    assert report.compile["errors"] == expected
    assert "kernel.metal:" in expected
    assert not expected.startswith("MetalCompileError:")


@pytest.mark.gpu
def test_one_real_multi_stage_build_returns_two_live_pipelines():
    """@brief 真实 Mac：一次多段构建产生同一 outcome 下的两个 live pipeline。"""
    from ...support.v1_tasks import v1_task
    from tcenv.backends.registry import load_backend
    from tcenv.remote.artifact_builder import SourceBundle

    task = v1_task()
    names = ("first_kernel", "second_kernel")
    source = "kernel void first_kernel() {}\nkernel void second_kernel() {}\n"
    builder = load_backend("metal").artifact_builder(
        kernel_contract=task.contract, kernel_names=names,
    )
    outcome = builder.build(
        SourceBundle(source.encode("utf-8")), task_contract=task.contract,
    )

    handles = outcome.executable_handles()
    assert len(handles) == 2
    assert handles[0] is not handles[1]
    assert all(handle.pso is not None for handle in handles)

    native = outcome.bundle.native
    assert native is not None and native.handle is not None
    assert native.descriptor["kernel_names"] == list(names)
    assert [item["name"] for item in native.descriptor["kernels"]] == list(names)
    assembled, _ = task.contract.assemble(source)
    assert native.descriptor["assembled_sha256"] == hashlib.sha256(
        assembled.encode("utf-8"),
    ).hexdigest()
    assert native.entry_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert outcome.provenance.source_sha256 == native.entry_sha256
    assert outcome.bundle.native_digest == native.digest
    assert outcome.evidence.native_document()["digest"] == native.digest
