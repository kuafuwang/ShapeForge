"""Real-Metal conv2d scratch loop: negatives first, then F(2x2) and F(4x4).

Paper shape only for the six-case smoke.  OH = OW = 54 is not a multiple of 4, so the
F(4x4) kernel has a 2-wide tail.  Quick mode is correctness, including the IEEE case.
Timing belongs to the later quiet-window confirmation, not to this smoke.  The native
eligibility check uses the real small-shape texture pipeline, single and bundle.
"""

from __future__ import annotations

import numpy as np
import pytest

from tcenv.backends.metal.kernel_pipeline import metal_contract
from tcenv.backends.metal.native_batch import eligible, prepare_native_batch
from tcenv.backends.metal.runtime import TextureArg, get_context
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.backends.metal.stage_binding import bind_pipeline_arguments
from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver
from tcenv.backends.metal.v2_runtime import V2ArgSet, reflection_for_kernel
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import locator_for_instance
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.track import (
    abi_limits_from_target,
    candidate_from_single_source,
    derive_tensor_memory_contract,
    resolve_geometry,
)
from tcenv.track.delivery import parse_candidate_geometry

from .conv2d_scratch_kernel import conv2d_winograd, launch_counts, metal_source

pytestmark = pytest.mark.gpu


def _locator():
    found = next(
        item
        for item in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT)
        if item.op_name == "conv2d"
    )
    return locator_for_instance(found, "fp16", "paper")


def _symbols() -> dict[str, int]:
    return {str(name): int(value) for name, value in _locator().load().instance.symbols.items()}


def _why(report: dict) -> str:
    parts = [f"status={report.get('status')!r}"]
    if report.get("message"):
        parts.append(str(report["message"]))
    errors = (report.get("compile") or {}).get("errors")
    if errors:
        parts.append(f"compile: {errors[:800]}")
    for key in ("abi", "safety"):
        violations = (report.get(key) or {}).get("violations") or []
        if violations:
            parts.append(key + ": " + "; ".join(str(item) for item in violations[:3]))
    return " | ".join(parts)


def _evaluate(source: str, *, cases: tuple[str, ...] | None = None):
    return evaluate_candidate(
        _locator(),
        source,
        target=load_target("apple-m4-max"),
        mode="quick",
        reps=1,
        cases=cases,
    )


def test_a_missed_last_oc_write_is_refused():
    """@brief Stage 0 漏写最后一个 OC 块时，三轮填充不一致，安全门判 unsafe。

    out1/out3 把没写的 ``uwin`` 留成 0xFF，out2 留成 0。Stage 1 读到不同的 U，写出的
    ``y`` 跟着变。评测先把「结果随初值变化」定为 unsafe，数值比较不会把它改判成 ok。
    """
    report = _evaluate(metal_source(_symbols(), 4, omit_last_oc=True), cases=("normal",)).result.report
    assert report["status"] == "unsafe", _why(report)
    assert "not independent of the initial write-output" in str(report.get("message", ""))


def test_a_partial_u_write_is_incorrect():
    """@brief 每个 U 的最后一个元素写成 0 时，三轮结果一致但和参考不同，判 incorrect。"""
    report = _evaluate(metal_source(_symbols(), 4, zero_last_element=True), cases=("normal",)).result.report
    assert report["status"] == "incorrect", _why(report)


def test_an_out_of_bounds_uwin_store_is_unsafe():
    """@brief 越过 uwin 末尾一个 float 必须撞上 0xCD 保护带并判 unsafe。"""
    report = _evaluate(metal_source(_symbols(), 4, oob=True), cases=("normal",)).result.report
    assert report["status"] == "unsafe", _why(report)


def test_f22_paper_passes_every_case_including_ieee():
    """@brief F(2x2, 3x3) 在 paper 的 6 个 case 上正确，含 IEEE 回退。"""
    report = _evaluate(metal_source(_symbols(), 2)).result.report
    assert report["status"] == "ok", _why(report)


def test_f44_paper_passes_every_case_including_the_partial_tail():
    """@brief F(4x4, 3x3) 在 paper 的 6 个 case 上正确，覆盖 OH=54 的 2 宽尾块。"""
    report = _evaluate(metal_source(_symbols(), 4)).result.report
    assert report["status"] == "ok", _why(report)


def _small_contract():
    contract = load_task_contract(REPO_ROOT / "tasks" / "conv2d", dtype="fp16", size="small")
    target = load_target(DEFAULT_TARGET_ID)
    abi = derive_tensor_memory_contract(contract.instance, target=target)
    limits = abi_limits_from_target(target)
    symbols = {str(name): int(value) for name, value in contract.instance.symbols.items()}
    return contract, abi, limits, symbols


def _execute(ctx, kernels, resolved, bindings, abi, image, weight) -> None:
    """用真实 stage 参数跑一轮原生 pipeline，并核对输出。"""
    argset = V2ArgSet(ctx, abi, resolved)
    argset.buffers["x"].write(np.ascontiguousarray(image).tobytes())
    argset.buffers["w"].write(np.ascontiguousarray(weight).tobytes())
    args = [argset.stage_args(binding) for binding in bindings]
    assert any(isinstance(arg, TextureArg) for _slot, arg in args[0])
    assert any(isinstance(arg, TextureArg) for _slot, arg in args[1])
    stages = list(kernels)
    assert eligible(stages=stages, arg_sets=[args], fills=[[]]) is True
    before = int(ctx._native_batch.calls)
    grids = [tuple(int(v) for v in stage.grid) for stage in resolved.stages]
    threads = [tuple(int(v) for v in stage.threads) for stage in resolved.stages]
    ctx.run_pipeline(stages, grids, threads, [0, 0], [args], [[]], iters=2)
    assert int(ctx._native_batch.calls) == before + 1
    out = np.frombuffer(argset.buffers["y"].read(), dtype=np.float16).reshape(image.shape[0], image.shape[1] - 2, image.shape[2] - 2, weight.shape[0])
    ref = conv2d_winograd(image.astype(np.float64), weight.astype(np.float64), tile=2)
    assert np.allclose(out.astype(np.float64), ref, rtol=2e-2, atol=2e-2)


def test_real_single_and_bundle_texture_pipelines_are_native_and_correct():
    """@brief small 的真实纹理参数在 single 和 bundle 上都 eligible，并且原生编码器真的跑了。"""
    contract, abi, limits, symbols = _small_contract()
    source = metal_source(symbols, 2)
    ctx = get_context()
    prepare_native_batch(ctx)
    assert ctx._native_batch is not None
    single = resolve_geometry(
        msl_single_geometry_resolver(contract)(candidate_from_single_source(source)),
        instance=contract.instance,
        contract=abi,
        limits=limits,
    )
    assembled, _header = metal_contract(
        task_contract=contract, abi=abi, geometry=single, target_id=DEFAULT_TARGET_ID
    ).assemble(source)
    kernels = ctx.compile_all(assembled, ["conv2d_wgrid", "conv2d_kernel"], math="safe")
    reflection = {
        kernels[0].name: tuple(reflection_for_kernel(kernels[0])),
        kernels[1].name: tuple(reflection_for_kernel(kernels[1])),
    }
    assert any(item.kind == "texture" and item.access == "write" for item in reflection["conv2d_wgrid"])
    assert any(item.kind == "texture" and item.access == "read" for item in reflection["conv2d_kernel"])
    stage0, stage1 = launch_counts(symbols, 2)
    bundle = resolve_geometry(
        parse_candidate_geometry(
            {
                "schema_version": 1,
                "source": "candidate",
                "scratch": [{"name": "uwin", "dtype": "fp32", "size_terms": ["16", "C", "OC"]}],
                "stages": [
                    {
                        "kernel": "conv2d_wgrid",
                        "grid": [stage0, 1, 1],
                        "threads": [64, 1, 1],
                        "scratch": [{"name": "uwin", "access": "write"}],
                    },
                    {
                        "kernel": "conv2d_kernel",
                        "grid": [stage1, 1, 1],
                        "threads": [64, 1, 1],
                        "scratch": [{"name": "uwin", "access": "read"}],
                    },
                ],
            }
        ),
        instance=contract.instance,
        contract=abi,
        limits=limits,
    )
    generator = np.random.default_rng(5)
    image = generator.standard_normal((symbols["N"], symbols["H"], symbols["W"], symbols["C"])).astype(np.float16)
    weight = generator.standard_normal((symbols["OC"], 3, 3, symbols["C"])).astype(np.float16)
    for resolved in (single, bundle):
        bindings = bind_pipeline_arguments(resolved, contract=abi, reflection=reflection, limits=limits)
        _execute(ctx, kernels, resolved, bindings, abi, image, weight)
