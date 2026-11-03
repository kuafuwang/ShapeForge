"""read_write / workspace / guard / timing protocol of the v2 plane (A2, CPU).

This is the machine-readable version of the plan's §3 rules: read_write initial values are
restored before every correctness trial, read_write buffers are never poisoned and never counted
as unwritten, guard bands are ``GUARD_PATTERN``, workspace clearing only touches declared-zero
regions, consecutive benchmark repetitions check only guards/out-of-bounds/hardware state, and
fresh-rotation timing is never mixed with a registered stateful run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcenv.contract_errors import GeometryError, TaskContractError
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    CORRECTNESS_TRIALS,
    GUARD_BYTES,
    GUARD_PATTERN,
    POISON,
    POISON_PATTERN,
    RESTORE_INITIAL,
    UNTOUCHED,
    ZERO,
    ExecutionGeometry,
    GeometryStage,
    PrepareDeclaration,
    TimingDiscipline,
    TargetAbiLimits,
    benchmark_launch_plan,
    check_prepare_timing,
    correctness_launch_plan,
    derive_tensor_memory_contract,
    resolve_geometry,
    unwritten_check_applies,
    workspace_fills,
)

from ..support.v2_tasks import (
    rmsnorm_read_write_opspec,
    rmsnorm_workspace_opspec,
    write_task,
)

LIMITS = TargetAbiLimits(max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)


def _fixture(tmp_path: Path, *, kind: str = "read_write"):
    """@brief 构造 ``(contract, geometry)``。

    @param tmp_path 临时目录。
    @param kind ``read_write`` 或 ``workspace``。
    @return 二元组。
    """
    if kind == "workspace":
        task = load_task_contract(write_task(tmp_path / "ws", op_name="rmsnorm_ws", opspec=rmsnorm_workspace_opspec()))
        stage = GeometryStage(
            kernel="scratch_kernel",
            grid=({"ceil_div": ["M*H", 256]}, 1, 1),
            threads=(256, 1, 1),
            workspace=({"name": "scratch", "zero": True},),
        )
    else:
        task = load_task_contract(write_task(tmp_path / "rw", op_name="rmsnorm_rw", opspec=rmsnorm_read_write_opspec()))
        stage = GeometryStage(kernel="rw_kernel", grid=({"ceil_div": ["M*H", 256]}, 1, 1), threads=(256, 1, 1))
    abi = derive_tensor_memory_contract(task.instance, target=load_target(DEFAULT_TARGET_ID))
    geometry = resolve_geometry(
        ExecutionGeometry(source="candidate", stages=(stage,)),
        instance=task.instance,
        contract=abi,
        limits=LIMITS,
    )
    return abi, geometry


def test_read_write_is_restored_on_every_trial_and_never_poisoned(tmp_path):
    """@brief out1/out2/out3 前都恢复 read_write 初值；纯 write 输出毒化/清零/毒化。"""
    abi, geometry = _fixture(tmp_path)
    read_write = set(abi.tensor(name).name for name in ("y", "acc"))
    for trial in CORRECTNESS_TRIALS:
        plan = correctness_launch_plan(abi, geometry, trial)
        assert plan.phase == f"correctness:{trial}"
        assert plan.guard_pattern == GUARD_PATTERN == 0xCD
        assert plan.guard_bytes == GUARD_BYTES
        for name in read_write:
            assert plan.action(name) == RESTORE_INITIAL, trial
            assert plan.action(name) != POISON
            assert unwritten_check_applies(abi, name) is False
        assert plan.action("x") == RESTORE_INITIAL
        assert unwritten_check_applies(abi, "x") is False
    assert correctness_launch_plan(abi, geometry, "out1").action("y") == RESTORE_INITIAL
    with pytest.raises(TaskContractError, match="unknown correctness trial"):
        correctness_launch_plan(abi, geometry, "out4")


def test_a_pure_write_output_is_poisoned_zeroed_and_poisoned(tmp_path):
    """@brief 纯 write 输出在 out1/out3 毒化、out2 清零，并参与 unwritten 统计。"""
    abi, geometry = _fixture(tmp_path)
    from tcenv.track.plane import TensorMemoryArg, TensorMemoryContract

    contract = TensorMemoryContract(
        abi_id=abi.abi_id,
        backend_id=abi.backend_id,
        target_id=abi.target_id,
        tensors=(
            TensorMemoryArg(name="x", role="input", dtype="fp16", space="global", access="read", shape=(4,), strides=(1,)),
            TensorMemoryArg(name="y", role="output", dtype="fp16", space="global", access="write", shape=(4,), strides=(1,)),
        ),
        acc_dtype="fp32",
        rtol=0.0,
        atol_rel=0.0,
        exact=False,
    )
    assert correctness_launch_plan(contract, geometry, "out1").action("y") == POISON
    assert correctness_launch_plan(contract, geometry, "out2").action("y") == ZERO
    assert correctness_launch_plan(contract, geometry, "out3").action("y") == POISON
    assert unwritten_check_applies(contract, "y") is True
    assert POISON_PATTERN == 0xFF


def test_workspace_clearing_only_touches_declared_zero_regions(tmp_path):
    """@brief benchmark 只清声明 zero 的 workspace；correctness 三个 trial 全都处理。"""
    abi, geometry = _fixture(tmp_path, kind="workspace")
    declared = workspace_fills(abi, geometry, phase="benchmark")
    assert [use.name for use in declared] == ["scratch"]
    assert all(use.zero for use in declared)
    plan = benchmark_launch_plan(abi, geometry, discipline=TimingDiscipline.fresh_rotation(3))
    assert plan.action("scratch") == ZERO
    assert plan.action("scratch") == ZERO

    # Removing the declaration leaves the workspace uncleared inside the timed window.
    undeclared = resolve_geometry(
        ExecutionGeometry(
            source="candidate",
            stages=(
                GeometryStage(
                    kernel="scratch_kernel",
                    grid=({"ceil_div": ["M*H", 256]}, 1, 1),
                    threads=(256, 1, 1),
                    workspace=("scratch",),
                ),
            ),
        ),
        instance=load_task_contract(
            write_task(tmp_path / "ws2", op_name="rmsnorm_ws", opspec=rmsnorm_workspace_opspec())
        ).instance,
        contract=abi,
        limits=LIMITS,
    )
    assert workspace_fills(abi, undeclared, phase="benchmark") == ()
    assert benchmark_launch_plan(abi, undeclared, discipline=TimingDiscipline.fresh_rotation()).action("scratch") == UNTOUCHED
    # Correctness still exercises both directions so a dependence on garbage is caught.
    assert correctness_launch_plan(abi, undeclared, "out1").action("scratch") == POISON
    assert correctness_launch_plan(abi, undeclared, "out2").action("scratch") == ZERO
    with pytest.raises(TaskContractError, match="unknown phase"):
        workspace_fills(abi, undeclared, phase="warmup")


def test_benchmark_repetitions_check_only_guards_bounds_and_hardware_state(tmp_path):
    """@brief 连续 reps 后只做 guard/越界/硬件状态检查；fresh 与 stateful 不串口径。"""
    abi, geometry = _fixture(tmp_path)
    fresh = TimingDiscipline.fresh_rotation(4)
    assert fresh.post_rep_checks() == ("guard_bands", "out_of_bounds", "hardware_state")
    assert "value_diff" not in fresh.post_rep_checks()
    plan = benchmark_launch_plan(abi, geometry, discipline=fresh)
    assert plan.action("y") == RESTORE_INITIAL and plan.action("acc") == RESTORE_INITIAL

    stateful = TimingDiscipline.stateful("acc is an accumulating state across reps")
    stateful_plan = benchmark_launch_plan(abi, geometry, discipline=stateful)
    assert stateful_plan.action("y") == UNTOUCHED and stateful_plan.action("acc") == UNTOUCHED
    assert stateful_plan.action("x") == UNTOUCHED
    assert stateful_plan.document()["phase"] == "benchmark:stateful"

    with pytest.raises(TaskContractError, match="stateful timing must register"):
        TimingDiscipline(mode="stateful")
    with pytest.raises(GeometryError, match="fresh rotation and a stateful"):
        TimingDiscipline(mode="fresh_rotation", stateful_declaration="mixed")
    with pytest.raises(GeometryError, match="both fresh-rotation and stateful"):
        TimingDiscipline.resolve(fresh_rotation=True, stateful_declaration="mixed")
    with pytest.raises(TaskContractError, match="must be registered"):
        TimingDiscipline.resolve(fresh_rotation=False)
    assert TimingDiscipline.resolve(fresh_rotation=True, rotation_sets=2).rotation_sets == 2
    with pytest.raises(TaskContractError, match="rotation_sets"):
        TimingDiscipline.fresh_rotation(0)
    with pytest.raises(TaskContractError, match="unknown timing discipline"):
        TimingDiscipline(mode="wall_clock")


def test_input_dependent_prepare_must_be_inside_the_timed_window(tmp_path):
    """@brief Host Tiling/prepare 不能把输入相关工作藏到计时外，也不能替内核算数学。"""
    ok = check_prepare_timing(
        [
            PrepareDeclaration(name="host_tiling", input_dependent=False),
            PrepareDeclaration(name="bind", input_dependent=True, inside_timing=True),
            # Reusable, input-independent preparation is the only kind that may be cached out
            # of the timed window, and it must name the scope it belongs to.
            PrepareDeclaration(
                name="compile", input_dependent=False, cached=True, cache_identity="candidate+target+instance"
            ),
        ]
    )
    assert len(ok) == 3
    # Input-dependent work is always billed to the run: a "cache" that exists because the
    # inputs were read once is still input work hidden from the measurement.
    with pytest.raises(GeometryError, match="input-dependent and outside the timed window"):
        check_prepare_timing([PrepareDeclaration(name="prep", input_dependent=True)])
    with pytest.raises(GeometryError, match="input-dependent and outside the timed window"):
        check_prepare_timing(
            [
                PrepareDeclaration(
                    name="plan_cache", input_dependent=True, cached=True, cache_identity="candidate+target+instance"
                )
            ]
        )
    with pytest.raises(GeometryError, match="preprocesses inputs outside"):
        check_prepare_timing([PrepareDeclaration(name="prep", input_dependent=True, preprocesses_inputs=True)])
    with pytest.raises(GeometryError, match="no cache identity"):
        check_prepare_timing([PrepareDeclaration(name="compile", input_dependent=False, cached=True)])
    with pytest.raises(GeometryError, match="computes the reference"):
        check_prepare_timing([PrepareDeclaration(name="oracle", input_dependent=False, computes_reference=True)])


def test_the_protocol_has_no_per_repetition_value_diff_and_no_hidden_fill(tmp_path):
    """@brief 协议本身不提供"每轮重复终值差分"或"计时外清零核"的入口。"""
    import dataclasses

    fields = {field.name for field in dataclasses.fields(TimingDiscipline)}
    assert "value_diff" not in fields and "overflow" not in fields
    abi, geometry = _fixture(tmp_path)
    plan = benchmark_launch_plan(abi, geometry, discipline=TimingDiscipline.fresh_rotation())
    document = plan.document()
    assert document["guard_pattern"] == GUARD_PATTERN
    assert {entry["action"] for entry in document["tensors"]} <= {RESTORE_INITIAL, ZERO, POISON, UNTOUCHED}
