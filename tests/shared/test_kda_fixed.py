"""Fixed-length KDA standard task and candidate-owned native planner (no GPU)."""

from __future__ import annotations

import json
import math
import shutil
import struct

import pytest
import torch

from tcenv.hostprogram.contract import host_contract_spec
from tcenv.hostprogram.native_runtime import NativeHostRuntime
from tcenv.hostprogram.native_toolchain import compile_native_host_program, discover_native_toolchain
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.baselines import load_baselines
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.tensors import round_to, to_raw
from tcenv.track.abi import derive_tensor_memory_contract
from tcenv.target import load_target

from ..support.kda_fixed_bundle import PLANNER_SOURCE

TASK = REPO_ROOT / "tasks" / "kda"
TARGET = load_target("apple-m4-max")


def _contract(size="small"):
    return load_task_contract(TASK, dtype="fp16", size=size)


def _planner_run(contract, *, target_values=None):
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
    build = compile_native_host_program(
        PLANNER_SOURCE,
        spec=spec,
        entry_path="host/tiling.c",
        toolchain=discover_native_toolchain(),
    )
    runtime = NativeHostRuntime(build, spec)
    runtime.set_target(target_values or {"max_threads_per_threadgroup": 1024})
    logical = contract.reference.generate(contract.instance, "normal", 7)
    shapes = {tensor.name: tensor.shape for tensor in contract.instance.tensors}
    dtypes = {tensor.name: tensor.dtype for tensor in contract.instance.tensors}
    raw = {
        name: to_raw(round_to(value.reshape(tuple(int(v) for v in shapes[name])), dtypes[name]), dtypes[name])
        for name, value in logical.items()
        if name in shapes
    }
    return runtime.execute(lambda name, off, nbytes: raw[name][off : off + nbytes], values=None, audit=True)


def _span(run, name):
    return next(span.data for span in run.spans if span.name == name)


def test_fixed_kda_has_an_independent_fixed_shape_abi():
    contract = _contract()
    instance = contract.instance
    assert contract.spec.op_name == "kda"
    assert instance.cu_seqlens is None
    assert "cu_seqlens" not in {tensor.name for tensor in instance.tensors}
    assert instance.symbols == {"N": 14, "H": 2, "D": 8}
    assert instance.tensor("q").shape == (14, 2, 8)
    assert instance.tensor("initial_state").shape == (2, 8, 8)
    assert instance.tensor("final_state").shape == (2, 8, 8)
    assert instance.tensor("work_items").shape == (3, 2)
    assert instance.tensor("handoff_states").shape == (2, 2, 8, 8)


def test_oracle_decays_the_initial_state_before_the_first_residual(tmp_path):
    task = tmp_path / "kda"
    shutil.copytree(TASK, task)
    document = json.loads((task / "opspec.json").read_text(encoding="utf-8"))
    document["sizes"]["single"]["H"] = 1
    document["sizes"]["single"]["D"] = 1
    (task / "opspec.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    contract = load_task_contract(task, dtype="fp16", size="single")
    inputs = {
        "q": torch.ones((1, 1, 1)),
        "k": torch.ones((1, 1, 1)),
        "v": torch.ones((1, 1, 1)),
        "g": torch.ones((1, 1, 1)),
        "beta": torch.ones((1, 1)),
        "A_log": torch.zeros((1,)),
        "dt_bias": torch.zeros((1,)),
        "initial_state": torch.full((1, 1, 1), 2.0),
        "scale": torch.ones((1,)),
    }
    result = contract.reference.execute(contract.instance, inputs, {"LOWER_BOUND": -5})
    normalized = 1.0 / math.sqrt(1.0 + 1e-6)
    sigmoid = 1.0 / (1.0 + math.exp(-1.0))
    decayed = 2.0 * math.exp(-5.0 * sigmoid)
    residual = (1.0 - decayed * normalized) * sigmoid
    expected_state = decayed + residual * normalized
    expected_output = expected_state * normalized
    assert result["final_state"].item() == pytest.approx(expected_state, rel=1e-12)
    assert result["output"].item() == pytest.approx(expected_output, rel=1e-12)


def test_task_listing_uses_each_tasks_default_when_paper_is_absent(capsys):
    from tcenv.cli import main

    assert main(["tasks"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("kda ") and "size=paper" in line for line in lines)
    assert any(line.startswith("varlen_kda ") and "size=mixed" in line for line in lines)


def test_fixed_kda_baseline_package_loads_and_closes():
    loaded = load_baselines(TASK, op_name="kda")
    assert set(loaded.runners) == {"torch_mps"}
    assert callable(loaded.runners["torch_mps"])


def test_fixed_cases_have_distinct_replayable_inputs_and_nonzero_initial_state():
    contract = _contract()
    first = contract.reference.generate(contract.instance, "normal", 7)
    second = contract.reference.generate(contract.instance, "wide_range", 7)
    replay = contract.reference.generate(contract.instance, "normal", 7)
    assert torch.equal(first["q"], replay["q"])
    assert not torch.equal(first["q"], second["q"])
    assert not torch.equal(first["initial_state"], second["initial_state"])
    assert torch.count_nonzero(first["initial_state"]) > 0
    assert first["initial_state"].shape == (2, 8, 8)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("single", [(0, 1)]),
        ("small", [(0, 5), (5, 5), (10, 4)]),
        ("tail", [(0, 6), (6, 6), (12, 5)]),
    ],
)
def test_native_planner_publishes_exact_balanced_work_spans(size, expected):
    run = _planner_run(_contract(size))
    assert run.ok, run.error_detail
    words = struct.unpack(f"<{2 * len(expected)}I", _span(run, "work_items"))
    assert list(zip(words[0::2], words[1::2])) == expected
    assert struct.unpack("<i", _span(run, "item_counts"))[0] == len(expected)
    assert {span.name for span in run.spans} == {"work_items", "item_counts"}


def test_native_planner_is_deterministic_across_target_views():
    first = _planner_run(_contract(), target_values={"max_threads_per_threadgroup": 1024})
    second = _planner_run(_contract(), target_values={"max_threads_per_threadgroup": 256})
    assert first.ok and second.ok
    assert [(span.name, span.offset, span.count, span.data) for span in first.spans] == [
        (span.name, span.offset, span.count, span.data) for span in second.spans
    ]
