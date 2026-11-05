"""V3: NPU simulator parity for the same Wasm Host Program module (CPU only).

The simulator is test-only and its reports are fixed ``simulated=true`` /
``publication_allowed=false``.  It never establishes real-device correctness or performance.

Two parity layers are checked here:

1. **Same HostTarget stub** — the real Wasm module (not a Python transliteration) produces the
   same ``work_items`` span as both the Metal-style and NPU-style adapters, and those adapters'
   work-list bytes and dependency edges agree.  This is the V3 extension of the V2 parity.
2. **Genuinely different targets** — the same module under a Metal-like target (18 cores) and an
   NPU-like target (64 cores, coarser DMA) yields the same item set, every item is claimed exactly
   once, and the dependency edges are equivalent.  Core assignment and launch geometry are allowed
   to differ and are not compared.
"""

from __future__ import annotations

import struct

import pytest

from tcenv.hostprogram.contract import host_contract_spec
from tcenv.hostprogram.runtime import HostRuntime
from tcenv.hostprogram.toolchain import compile_host_program, discover_toolchain
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
    dependency_edges,
    metal_adapter,
    npu_adapter,
    parity_problems,
    simulate_schedule,
    unpack_items,
)

TASK = REPO_ROOT / "tasks" / "varlen_kda"
TARGET = load_target("apple-m4-max")
MAX_SPLIT = 3


def _toolchain():
    try:
        return discover_toolchain()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"V3 Host Program toolchain unavailable ({exc}); run `brew install llvm lld` + `uv sync`")


def _wasm_work_list():
    """@brief 用真实 Wasm module 跑 canonical 分段，返回 ``(segments, words, flags)``。"""
    contract = load_task_contract(TASK, dtype="fp16", size="mixed")
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
    build = compile_host_program(PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    runtime = HostRuntime(build, spec)
    logical = contract.reference.generate(contract.instance, "canonical", 7)
    shapes = {tensor.name: tensor.shape for tensor in contract.instance.tensors}
    dtypes = {tensor.name: tensor.dtype for tensor in contract.instance.tensors}
    raw = {
        name: to_raw(round_to(value.reshape(tuple(int(v) for v in shapes[name])), dtypes[name]), dtypes[name])
        for name, value in logical.items()
        if name in shapes
    }
    nsp1 = contract.instance.symbols["NSP1"]
    headers = {
        "metal-stub": {
            "max_buffer_index": 30,
            "max_threads_per_threadgroup": METAL_STUB.scratch_bytes,
            "max_threadgroup_memory_bytes": METAL_STUB.scratch_bytes,
        },
        "npu-stub": {
            "max_buffer_index": 7,
            "max_threads_per_threadgroup": NPU_STUB.scratch_bytes,
            "max_threadgroup_memory_bytes": NPU_STUB.scratch_bytes,
        },
    }
    outputs = {}
    for name, target in headers.items():
        runtime.set_target(target)
        run = runtime.execute(
            lambda key, off, nbytes, _raw=raw: _raw[key][off : off + nbytes], values=None, audit=True
        )
        assert run.ok, run.error_detail
        spans = {span.name: span for span in run.spans}
        outputs[name] = (spans["work_items"].data, spans["handoff_flags"].data)
    cu = list(struct.unpack("<%di" % nsp1, raw["cu_seqlens"]))
    segments = [cu[index + 1] - cu[index] for index in range(len(cu) - 1)]
    # The module is input-driven: two different HostTarget views produce the same bytes.
    assert outputs["metal-stub"] == outputs["npu-stub"]
    return segments, *outputs["metal-stub"], contract.instance.symbols["H"], contract.instance.symbols["D"]


def test_the_wasm_work_list_matches_both_adapters_under_one_target_stub():
    """@brief 同一 HostTarget 桩：Wasm span 与 Metal/NPU adapter 逐字节一致。"""
    segments, words, _flags, heads, dim = _wasm_work_list()
    metal = metal_adapter(segments, heads=heads, max_split=MAX_SPLIT, target=METAL_STUB)
    npu = npu_adapter(segments, heads=heads, max_split=MAX_SPLIT, dim=dim, target=METAL_STUB)
    assert metal.words == words
    assert npu.words == words
    assert parity_problems(metal, npu) == []
    # Both adapters still refuse a report without the simulation markers.
    for report in (metal, npu):
        document = report.document()
        assert document["simulated"] is True
        assert document["publication_allowed"] is False
        assert document["measurement_kind"] == "simulated"


def test_real_different_targets_keep_the_item_set_and_dependencies_but_not_the_schedule():
    """@brief 异 target：item 全集/唯一认领/依赖边等价；core 分配与 launch 允许不同。"""
    segments, words, _flags, heads, dim = _wasm_work_list()
    items = list(unpack_items(words, len(words) // 8))
    assert len(items) == len(unpack_items(words, len(words) // 8))
    # Same Wasm work list, two genuinely different device constraints.
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
    # Two independent planners (Metal-style vs NPU-style) on genuinely different targets must
    # still produce the same item table and dependency edges; the Wasm module's bytes already
    # agreed with the Metal-style adapter above.
    metal_report = metal_adapter(segments, heads=heads, max_split=MAX_SPLIT, target=metal)
    npu_report = npu_adapter(segments, heads=heads, max_split=MAX_SPLIT, dim=dim, target=npu)
    def fields(entry):
        return (entry.chain, entry.piece, entry.begin, entry.valid, entry.pieces)

    assert [fields(item) for item in metal_report.items] == [fields(item) for item in npu_report.items]
    assert metal_report.edges == npu_report.edges == dependency_edges(items)

    metal_schedule = simulate_schedule(items, metal, order="stage")
    npu_schedule = simulate_schedule(items, npu, order="stage")
    assert metal_schedule.deadlock is False
    assert npu_schedule.deadlock is False
    assert claim_problems(items, metal_schedule.claims) == []
    assert claim_problems(items, npu_schedule.claims) == []
    # Every item completes on both targets (a different core count may change the wave map).
    assert metal_schedule.completed == npu_schedule.completed == tuple(range(len(items)))
    assert metal_schedule.cores != npu_schedule.cores
    # Claim maps are target-specific evidence, and both must be marked simulated.
    for schedule in (metal_schedule, npu_schedule):
        document = schedule.document()
        assert document["simulated"] is True
        assert document["publication_allowed"] is False
