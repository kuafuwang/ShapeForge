"""V3: the candidate-owned C99 KDA planner, differentially checked against a Python oracle.

The C fixture is the candidate; the Python code in ``tests/support/hostprogram_oracle.py`` is a
separate implementation used **only** here.  Nothing in this file enters a task pack, a candidate
or the timed path.

* mode 1 emits the canonical balanced piece table; its bytes must equal the independent V2
  ``plan_pieces`` derivation for every L1 case (the Dyn-0 device plan is the other witness).
* mode 2 exercises the fixed-point cost model, range-cut and LPT grouping.  Its emitted bytes are
  compared with the oracle's independent range-cut + heapq LPT output, and its makespan must not
  exceed a round-robin chain assignment on a skewed probe tier.
* invalid boundaries and determinism are checked through the same Wasm runtime the evaluator uses.
"""

from __future__ import annotations

import json
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
from ..support.hostprogram_oracle import (
    coverage_problems,
    emitted_items,
    lpt_assignment,
    range_cut_pieces,
    round_robin_assignment,
)
from ..support.npu_sim.varlen_worklist import pack_items, plan_pieces
from ..support.varlen_probe import write_probe_task

TASK = REPO_ROOT / "tasks" / "varlen_kda"
TARGET = load_target("apple-m4-max")
MAX_SPLIT = 3


def _toolchain():
    try:
        return discover_toolchain()
    except Exception as exc:  # noqa: BLE001 - skip only when the real toolchain is absent
        pytest.skip(f"V3 Host Program toolchain unavailable ({exc}); run `brew install llvm lld` + `uv sync`")


def _spec(contract):
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    return host_contract_spec(instance=contract.instance, tensors=abi.tensors)


def _run(contract, source: str, *, case: str = "canonical", seed: int = 7, target_values=None):
    spec = _spec(contract)
    build = compile_host_program(source, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    runtime = HostRuntime(build, spec)
    runtime.set_target(target_values or {"max_threads_per_threadgroup": 1024})
    logical = contract.reference.generate(contract.instance, case, seed)
    shapes = {tensor.name: tensor.shape for tensor in contract.instance.tensors}
    dtypes = {tensor.name: tensor.dtype for tensor in contract.instance.tensors}
    raw = {
        name: to_raw(round_to(value.reshape(tuple(int(v) for v in shapes[name])), dtypes[name]), dtypes[name])
        for name, value in logical.items()
        if name in shapes
    }
    run = runtime.execute(lambda name, off, nbytes: raw[name][off : off + nbytes], values=None, audit=True)
    return spec, raw, run


def _segments(raw: dict[str, bytes], count: int) -> list[int]:
    values = list(struct.unpack("<%di" % count, raw["cu_seqlens"]))
    return [values[index + 1] - values[index] for index in range(len(values) - 1)]


def _decode_items(span_data: bytes) -> list[tuple[int, int, int, int, int]]:
    words = list(struct.unpack("<%dI" % (len(span_data) // 4), span_data))
    return [
        (word1 & 0xFF, (word1 >> 8) & 0xFF, word0 & 0xFFFF, word0 >> 16, (word1 >> 16) & 0xFF)
        for word0, word1 in zip(words[0::2], words[1::2])
    ]


def _span(run, name: str) -> bytes:
    return next(span.data for span in run.spans if span.name == name)


def _count(run, name: str) -> int:
    return next(span.count for span in run.spans if span.name == name)


CASES = ("canonical", "permuted", "skewed", "all_unaligned_primes", "max_handoff_stress", "chunk_boundary", "zero_length", "single_sequence")


@pytest.mark.parametrize("case", CASES)
def test_mode_one_work_items_match_the_independent_device_plan_byte_for_byte(case):
    """@brief 同一分段下 C99 规划与独立 V2 ``plan_pieces`` 逐字节一致。"""
    contract = load_task_contract(TASK, dtype="fp16", size="mixed")
    _spec_, raw, run = _run(contract, PLANNER_SOURCE, case=case)
    assert run.ok, run.error_detail
    segments = _segments(raw, contract.instance.symbols["NSP1"])
    items = plan_pieces(segments, contract.instance.symbols["H"], MAX_SPLIT)
    assert _span(run, "work_items") == pack_items(items)
    assert _count(run, "item_counts") == 1
    assert struct.unpack("<i", _span(run, "item_counts"))[0] == len(items)
    # Every possible boundary of every chain is reset, including empty chains.
    flags = _span(run, "handoff_flags")
    assert _count(run, "handoff_flags") == contract.instance.symbols["num_seqs"] * contract.instance.symbols["H"] * (MAX_SPLIT - 1)
    assert flags == bytes(len(flags))


def test_mode_two_range_cut_and_lpt_match_the_independent_oracle(tmp_path):
    """@brief mode 2 的 range-cut + LPT 发射顺序与 Python oracle（heapq）逐字节一致。"""
    write_probe_task(
        TASK,
        tmp_path / "varlen_kda",
        {"skew": [64] + [1] * 39},
    )
    contract = load_task_contract(tmp_path / "varlen_kda", dtype="fp16", size="skew")
    source = "#define SF_PLANNER_MODE 2\n" + PLANNER_SOURCE
    _spec_, raw, run = _run(contract, source)
    assert run.ok, run.error_detail
    segments = _segments(raw, contract.instance.symbols["NSP1"])
    heads = contract.instance.symbols["H"]
    chains = len(segments) * heads
    workers = min(8, chains)
    pieces = range_cut_pieces(segments, heads, MAX_SPLIT)
    assignment, lpt_makespan = lpt_assignment(pieces, workers)
    expected = pack_items(emitted_items(pieces, assignment, workers))
    assert _span(run, "work_items") == expected
    decoded = _decode_items(_span(run, "work_items"))
    assert coverage_problems(decoded, segments, heads, max_split=MAX_SPLIT) == []
    assert len(decoded) == len(pieces)
    # LPT is not worse than a naive chain round-robin on the same cost model.
    _rr_assignment, rr_makespan = round_robin_assignment(pieces, workers)
    assert lpt_makespan <= rr_makespan
    # The long chain (64 tokens) is the heaviest; LPT gives it its own worker.
    long_chain = 0
    others = [chain for chain in assignment if chain != long_chain]
    assert all(assignment[chain] != assignment[long_chain] for chain in others[: workers - 1])


def test_the_planner_is_deterministic_and_target_independent():
    """@brief 同一模块/输入可重放；不同 target 参数不改变输入驱动的 item 表。"""
    contract = load_task_contract(TASK, dtype="fp16", size="mixed")
    _spec_, raw, first = _run(contract, PLANNER_SOURCE, target_values={"max_threads_per_threadgroup": 1024})
    assert first.ok
    _spec_, _raw, second = _run(
        contract,
        PLANNER_SOURCE,
        target_values={"max_threads_per_threadgroup": 256, "max_buffer_index": 7},
    )
    assert second.ok
    assert [span.data for span in first.spans] == [span.data for span in second.spans]


def test_invalid_boundaries_are_refused_without_a_span():
    """@brief 非 0 首项 / 非单调 / 末项不等于 total 都是 typed candidate failure。"""
    contract = load_task_contract(TASK, dtype="fp16", size="mixed")
    spec = _spec(contract)
    build = compile_host_program(PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    runtime = HostRuntime(build, spec)
    runtime.set_target({"max_threads_per_threadgroup": 1024})
    logical = contract.reference.generate(contract.instance, "canonical", 7)
    shapes = {tensor.name: tensor.shape for tensor in contract.instance.tensors}
    dtypes = {tensor.name: tensor.dtype for tensor in contract.instance.tensors}
    raw = {
        name: to_raw(round_to(value.reshape(tuple(int(v) for v in shapes[name])), dtypes[name]), dtypes[name])
        for name, value in logical.items()
        if name in shapes
    }
    nsp1 = contract.instance.symbols["NSP1"]
    good = list(struct.unpack("<%di" % nsp1, raw["cu_seqlens"]))
    for bad in ([1] + good[1:], [good[0], good[2], good[1]] + good[3:], good[:-1] + [good[-1] + 1]):
        assert len(bad) == nsp1
        broken = dict(raw)
        broken["cu_seqlens"] = struct.pack("<%di" % nsp1, *bad)
        run = runtime.execute(lambda name, off, nbytes, _r=broken: _r[name][off : off + nbytes], values=None, audit=True)
        assert run.status != 0, bad
        assert run.spans == ()
        assert run.error_code == "host_program_invalid"


def test_the_quick_host_probe_cli_compiles_and_runs_the_fixture(capsys):
    """@brief 候选可单独编译/试跑 Host Program，不必先编译 device kernel 或占用 GPU。"""
    from tcenv.cli import main

    source = REPO_ROOT / "tests" / "support" / "golden" / "shared" / "hostprogram" / "kda_planner.c"
    code = main(["host-probe", "varlen_kda", "--dtype", "fp16", "--size", "mixed", "--source", str(source)])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "ok"
    assert document["fuel_used"] > 0 and document["fuel_used"] < document["fuel_budget"]
    assert document["used_workspaces"] == ["work_items", "item_counts", "handoff_flags"]
    assert {item["name"] for item in document["input_reads"]} >= {"cu_seqlens"}
    assert document["engine"] == "wasmtime"
