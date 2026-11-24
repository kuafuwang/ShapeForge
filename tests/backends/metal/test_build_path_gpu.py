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
def test_one_real_multi_stage_build_returns_two_live_pipelines():
    """@brief 真实 Mac：一次多段构建产生同一 outcome 下的两个 live pipeline。"""
    from ...support.v2_specimen import V2_TASK, v2_candidate_source, v2_pack_registered_task
    from tcenv.backends.registry import load_backend
    from tcenv.remote.artifact_builder import SourceBundle

    task = v2_pack_registered_task(V2_TASK, "fp16", "small")
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
    # 多阶段 descriptor 不把 kernel 0 的 bindings 再存一遍：顶层已经完整记录它们；
    # 其余阶段各自保留自己的 bindings，信息不丢但 native descriptor 不再重复膨胀。
    assert "bindings" not in native.descriptor["kernels"][0]
    assert native.descriptor["kernels"][1]["bindings"] == []
    assembled, _ = task.contract.assemble(source)
    assert native.descriptor["assembled_sha256"] == hashlib.sha256(
        assembled.encode("utf-8"),
    ).hexdigest()
    assert native.entry_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert outcome.provenance.source_sha256 == native.entry_sha256
    assert outcome.bundle.native_digest == native.digest
    assert outcome.evidence.native_document()["digest"] == native.digest
