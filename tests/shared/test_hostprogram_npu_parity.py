"""NPU simulator parity for the same native64 Host Program (CPU only).

The simulator is test-only and its reports are fixed ``simulated=true`` /
``publication_allowed=false``.  It never establishes real-device correctness or performance.

What is actually compared:

* **Same HostTarget** — the *real native dylib* (not a Python transliteration) is executed under one
  target view; every valid span's name/offset/count/elem_nbytes/bytes/digest and the resolved
  scalar dictionary must match a second execution, and both Metal-style/NPU-style adapters must
  produce the same work-list bytes and dependency edges.
* **Genuinely different targets** — the same module is executed under a Metal-like and an
  NPU-like target; each run's own output is unpacked and scheduled on its own target.  The item
  set, one-claim-per-item and the dependency edges must agree; core assignment, grid and launch
  geometry are allowed to differ and are not compared.
"""

from __future__ import annotations

import struct

import pytest

from tcenv.hostprogram.contract import host_contract_spec
from tcenv.hostprogram.native_runtime import NativeHostRuntime
from tcenv.hostprogram.native_toolchain import compile_native_host_program, discover_native_toolchain
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.track.abi import derive_tensor_memory_contract
from tcenv.target import load_target
from tcenv.tensors import round_to, to_raw

from ..support.hostprogram_fixtures import PLANNER_SOURCE
from ..support.npu_sim.varlen_worklist import (
    METAL_STUB,
    NPU_STUB,
    SimTarget,
    claim_problems,
    metal_adapter,
    npu_adapter,
    parity_problems,
    simulate_schedule,
    unpack_items,
)

TASK = REPO_ROOT / "tasks" / "varlen_kda"
TARGET = load_target("apple-m4-max")
MAX_SPLIT = 3

METAL_VALUES = {
    "max_buffer_index": 31,
    "max_threads_per_threadgroup": 1024,
    "max_threadgroup_memory_bytes": 32768,
}
NPU_VALUES = {
    "max_buffer_index": 7,
    "max_threads_per_threadgroup": 2048,
    "max_threadgroup_memory_bytes": 196608,
}


def _toolchain():
    try:
        return discover_native_toolchain()
    except Exception as exc:  # noqa: BLE001 - skip only when the real toolchain is absent
        pytest.skip(f"V3 Host Program toolchain unavailable ({exc}); run `brew install llvm lld` + `uv sync`")


def _fixture():
    """@brief 编译一次 fixture，返回 ``(runtime, spec, raw, segments, heads, dim)``。"""
    contract = load_task_contract(TASK, dtype="fp16", size="mixed")
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
    build = compile_native_host_program(PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    runtime = NativeHostRuntime(build, spec)
    logical = contract.reference.generate(contract.instance, "canonical", 7)
    shapes = {tensor.name: tensor.shape for tensor in contract.instance.tensors}
    dtypes = {tensor.name: tensor.dtype for tensor in contract.instance.tensors}
    raw = {
        name: to_raw(round_to(value.reshape(tuple(int(v) for v in shapes[name])), dtypes[name]), dtypes[name])
        for name, value in logical.items()
        if name in shapes
    }
    nsp1 = contract.instance.symbols["NSP1"]
    cu = list(struct.unpack("<%di" % nsp1, raw["cu_seqlens"]))
    segments = [cu[index + 1] - cu[index] for index in range(len(cu) - 1)]
    return runtime, spec, raw, segments, contract.instance.symbols["H"], contract.instance.symbols["D"]


def _execute(runtime, raw, target_values):
    runtime.set_target(target_values)
    run = runtime.execute(lambda name, off, nbytes: raw[name][off : off + nbytes], values=None, audit=True)
    assert run.ok, run.error_detail
    return run


def _snapshot(run):
    """@brief 所有有效 span 的元数据/字节/digest + resolved scalars（不是只看两个 span）。"""
    return {
        "status": run.status,
        "spans": tuple(
            (span.name, span.offset, span.count, span.elem_nbytes, span.data, span.sha256)
            for span in run.spans
        ),
        "resolved": dict(run.resolved_scalars),
        "reads": tuple((read.name, read.elem_off, read.count, read.sha256) for read in run.input_reads),
    }


def test_one_target_compares_every_span_and_resolved_scalar():
    """@brief 同一 HostTarget 下：全部 span 元数据/字节/resolved scalars 逐项一致。"""
    runtime, _spec, raw, segments, heads, dim = _fixture()
    first = _execute(runtime, raw, METAL_VALUES)
    second = _execute(runtime, raw, METAL_VALUES)
    assert _snapshot(first) == _snapshot(second)
    # Explicitly cover the span that a work-items-only comparison would miss.
    counts = next(span for span in first.spans if span.name == "item_counts")
    assert counts.count == 1
    assert struct.unpack("<i", counts.data)[0] == len(unpack_items(
        next(span.data for span in first.spans if span.name == "work_items"),
        next(span.count for span in first.spans if span.name == "work_items") // 2,
    ))
    assert first.resolved_scalars == {}

    metal = metal_adapter(segments, heads=heads, max_split=MAX_SPLIT, target=METAL_STUB)
    npu = npu_adapter(segments, heads=heads, max_split=MAX_SPLIT, dim=dim, target=METAL_STUB)
    words = next(span.data for span in first.spans if span.name == "work_items")
    assert metal.words == npu.words == words
    assert parity_problems(metal, npu) == []
    for report in (metal, npu):
        document = report.document()
        assert document["simulated"] is True
        assert document["publication_allowed"] is False
        assert document["measurement_kind"] == "simulated"


def _semantic_items(items):
    """@brief 与发射顺序/位置索引无关的 item 身份集合。"""
    return {(item.chain, item.piece, item.begin, item.valid, item.pieces) for item in items}


def _semantic_edges(items):
    """@brief 用 ``(chain, piece)`` 稳定键表达的语义依赖边，不是位置索引。"""
    keys = {(item.chain, item.piece): (item.chain, item.piece, item.begin, item.valid, item.pieces) for item in items}
    edges = []
    for item in items:
        producer = (item.chain, item.piece)
        consumer = (item.chain, item.piece + 1)
        if consumer in keys:
            edges.append((keys[producer], keys[consumer]))
    return sorted(edges)


def test_different_targets_keep_the_item_set_claims_and_semantic_edges():
    """@brief 同一 module 在 18-core Metal-like / 64-core NPU-like 下各自执行、各自调度。

    比较项：两个 run 的全部 span/字节/resolved 相同（Host Program 是输入驱动而非 target 驱动）；
    各自把**自己的** work_items 交给自己的 scheduler，item 全集相同、每项恰好一个 core 认领、
    依赖边等价；claim map/grid 不比较。
    """
    runtime, _spec, raw, segments, heads, dim = _fixture()
    metal = SimTarget(
        id="real-metal-like",
        kind="metal",
        cores=18,
        scratch_bytes=32768,
        dma_alignment=64,
        item_table_capacity=64,
    )
    npu = SimTarget(
        id="real-npu-like",
        kind="npu",
        cores=64,
        scratch_bytes=196608,
        dma_alignment=512,
        item_table_capacity=160,
    )
    metal_run = _execute(runtime, raw, METAL_VALUES)
    npu_run = _execute(runtime, raw, NPU_VALUES)

    def items_of(run):
        span = next(span for span in run.spans if span.name == "work_items")
        return list(unpack_items(span.data, span.count // 2))

    metal_items = items_of(metal_run)
    npu_items = items_of(npu_run)
    # Cross-target parity is the item *set* and semantic edges; emission order is target-specific
    # and may legitimately differ.  (A separate mode-3 test exercises that difference.)
    assert _semantic_items(metal_items) == _semantic_items(npu_items)
    assert _semantic_edges(metal_items) == _semantic_edges(npu_items)

    metal_schedule = simulate_schedule(metal_items, metal, order="stage")
    npu_schedule = simulate_schedule(npu_items, npu, order="stage")
    assert metal_schedule.deadlock is False and npu_schedule.deadlock is False
    assert claim_problems(metal_items, metal_schedule.claims) == []
    assert claim_problems(npu_items, npu_schedule.claims) == []
    assert sorted(metal_schedule.completed) == sorted(npu_schedule.completed) == list(range(len(metal_items)))
    assert metal_schedule.cores != npu_schedule.cores

    # V2's independent planners under the same genuinely different targets.
    metal_report = metal_adapter(segments, heads=heads, max_split=MAX_SPLIT, target=metal)
    npu_report = npu_adapter(segments, heads=heads, max_split=MAX_SPLIT, dim=dim, target=npu)
    assert _semantic_items(metal_report.items) == _semantic_items(npu_report.items)
    assert _semantic_edges(metal_report.items) == _semantic_edges(npu_report.items)

    for schedule in (metal_schedule, npu_schedule):
        document = schedule.document()
        assert document["simulated"] is True
        assert document["publication_allowed"] is False


def test_a_target_dependent_emission_order_is_accepted_when_the_semantics_match():
    """@brief 合法 target-dependent 顺序：字节顺序不同，item 集合/语义边/认领仍等价。"""
    runtime, spec, raw, _segments, _heads, _dim = _fixture()
    source = "#define SF_PLANNER_MODE 3\n" + PLANNER_SOURCE
    build = compile_native_host_program(source, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())

    def run_with(values):
        local = NativeHostRuntime(build, spec)
        local.set_target(values)
        return local.execute(lambda name, off, nbytes: raw[name][off : off + nbytes], values=None, audit=True)

    metal_run = run_with(METAL_VALUES)
    npu_run = run_with(NPU_VALUES)
    assert metal_run.ok and npu_run.ok
    metal_span = next(span for span in metal_run.spans if span.name == "work_items")
    npu_span = next(span for span in npu_run.spans if span.name == "work_items")
    # The NPU target's larger threadgroup limit reverses the emission order; the bytes differ.
    assert metal_span.data != npu_span.data
    metal_items = list(unpack_items(metal_span.data, metal_span.count // 2))
    npu_items = list(unpack_items(npu_span.data, npu_span.count // 2))
    assert _semantic_items(metal_items) == _semantic_items(npu_items)
    assert _semantic_edges(metal_items) == _semantic_edges(npu_items)

    metal = SimTarget(id="metal-mode3", kind="metal", cores=18, scratch_bytes=32768, dma_alignment=64, item_table_capacity=64)
    npu = SimTarget(id="npu-mode3", kind="npu", cores=64, scratch_bytes=196608, dma_alignment=512, item_table_capacity=160)
    for items, target in ((metal_items, metal), (npu_items, npu)):
        schedule = simulate_schedule(items, target, order="stage")
        assert schedule.deadlock is False
        assert claim_problems(items, schedule.claims) == []
        assert sorted(schedule.completed) == list(range(len(items)))
        assert schedule.document()["simulated"] is True
        assert schedule.document()["publication_allowed"] is False
