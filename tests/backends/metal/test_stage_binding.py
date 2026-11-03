"""MSL-reflection per-stage argument binding for v2 candidates (A2, CPU).

No device is opened: the reflection records here are the plain data the Metal build adapter
converts ``MTLDataType`` into at the edge.  The rules under test are the ones a candidate cannot
be allowed to negotiate at run time — contiguous per-stage slots, name-based binding of tensors
and ``Derived Scalars``, access/width checks, and at least one active stage writing every result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcenv.backends.metal.stage_binding import (
    ReflectedArgument,
    bind_pipeline_arguments,
    bind_stage_arguments,
)
from tcenv.contract_errors import GeometryError
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    ExecutionGeometry,
    GeometryStage,
    TargetAbiLimits,
    derive_tensor_memory_contract,
    resolve_geometry,
)

from ...support.v2_tasks import rmsnorm_opspec, rmsnorm_read_write_opspec, rmsnorm_workspace_opspec, write_task

LIMITS = TargetAbiLimits(max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)


def _resolved(tmp_path: Path, *, kind: str = "plain", stages=None):
    """@brief 构造 ``(contract, geometry)``。

    @param tmp_path 临时目录。
    @param kind ``plain``/``workspace``/``read_write``。
    @param stages 覆盖 stage 列表。
    @return 二元组。
    """
    if kind == "workspace":
        task = load_task_contract(write_task(tmp_path / "ws", op_name="rmsnorm_ws", opspec=rmsnorm_workspace_opspec()))
    elif kind == "read_write":
        task = load_task_contract(write_task(tmp_path / "rw", op_name="rmsnorm_rw", opspec=rmsnorm_read_write_opspec()))
    elif kind == "attr":
        document = rmsnorm_opspec("rmsnorm_hint")
        document["attrs"]["SPLIT_FACTOR"] = {"dtype": "int", "default": 4}
        task = load_task_contract(write_task(tmp_path / "attr", op_name="rmsnorm_hint", opspec=document))
    else:
        task = load_task_contract(write_task(tmp_path / "plain"))
    abi = derive_tensor_memory_contract(task.instance, target=load_target(DEFAULT_TARGET_ID))
    stage_list = stages or (GeometryStage(kernel="main", grid=({"ceil_div": ["M*H", 256]}, 1, 1), threads=(256, 1, 1)),)
    geometry = resolve_geometry(
        ExecutionGeometry(source="candidate", stages=tuple(stage_list)),
        instance=task.instance,
        contract=abi,
        limits=LIMITS,
    )
    return abi, geometry


def _buffers(*items: tuple[int, str, str, str, int]) -> tuple[ReflectedArgument, ...]:
    """@brief 由 ``(slot, name, access, scalar, elem_size)`` 构造反射记录。

    @param items 记录。
    @return 反射元组。
    """
    return tuple(
        ReflectedArgument(index=slot, kind="buffer", name=name, access=access, scalar_type=scalar, elem_size=elem)
        for slot, name, access, scalar, elem in items
    )


def test_a_stage_binds_tensors_and_derived_scalars_by_name_into_contiguous_slots(tmp_path):
    """@brief 张量与派生标量按名绑定到连续的 stage 槽位；派生标量进入 setBytes。"""
    abi, geometry = _resolved(
        tmp_path,
        stages=(
            GeometryStage(
                kernel="main",
                grid=({"ceil_div": ["M*H", 256]}, 1, 1),
                threads=(256, 1, 1),
                derived_scalars=(("ROW_COUNT", 4), ("STRIDE", "H")),
            ),
        ),
    )
    stage = geometry.active_stages[0]
    binding = bind_stage_arguments(
        stage,
        contract=abi,
        reflection=_buffers(
            (0, "x", "read", "half", 2),
            (1, "w", "read", "half", 2),
            (2, "y", "write", "half", 2),
            (3, "ROW_COUNT", "read", "int", 4),
            (4, "STRIDE", "read", "uint", 4),
        ),
        limits=LIMITS,
    )
    assert [argument.slot for argument in binding.arguments] == [0, 1, 2, 3, 4]
    assert binding.set_bytes == ((3, "ROW_COUNT", 4), (4, "STRIDE", abi.tensor("x").shape[1]))
    document = binding.document()
    assert [entry["role"] for entry in document["arguments"]] == [
        "tensor",
        "tensor",
        "tensor",
        "derived_scalar",
        "derived_scalar",
    ]
    assert document["arguments"][3]["value"] == 4


def test_a_vectorized_pointer_is_allowed_but_the_base_scalar_must_match(tmp_path):
    """@brief ``half4*`` 是合法向量化；``float*`` 覆盖 half 缓冲则不合法。"""
    abi, geometry = _resolved(tmp_path)
    stage = geometry.active_stages[0]
    ok = bind_stage_arguments(
        stage,
        contract=abi,
        reflection=_buffers((0, "x", "read", "half", 8), (1, "w", "read", "half", 2), (2, "y", "write", "half", 4)),
        limits=LIMITS,
    )
    from tcenv.dtypes import get_dtype

    x = abi.tensor("x")
    elements = 1
    for dim in x.shape:
        elements *= int(dim)
    assert ok.arguments[0].nbytes == elements * get_dtype(x.dtype).nbytes
    with pytest.raises(GeometryError, match="base scalar type"):
        bind_stage_arguments(
            stage,
            contract=abi,
            reflection=_buffers((0, "x", "read", "float", 4), (1, "w", "read", "half", 2), (2, "y", "write", "half", 2)),
            limits=LIMITS,
        )
    with pytest.raises(GeometryError, match="not a `half` scalar"):
        bind_stage_arguments(
            stage,
            contract=abi,
            reflection=_buffers((0, "x", "read", "half", 2 + 1), (1, "w", "read", "half", 2), (2, "y", "write", "half", 2)),
            limits=LIMITS,
        )


def test_access_permission_and_contiguity_are_enforced(tmp_path):
    """@brief 输入必须只读、read_write 必须可写、槽位必须从 0 连续且不重复。"""
    abi, geometry = _resolved(tmp_path, kind="read_write")
    stage = geometry.active_stages[0]
    with pytest.raises(GeometryError, match="input must stay"):
        bind_stage_arguments(
            stage,
            contract=abi,
            reflection=_buffers((0, "x", "readwrite", "half", 2), (1, "w", "read", "half", 2), (2, "y", "readwrite", "half", 2), (3, "acc", "readwrite", "half", 2)),
            limits=LIMITS,
        )
    with pytest.raises(GeometryError, match="read_write tensor"):
        bind_stage_arguments(
            stage,
            contract=abi,
            reflection=_buffers((0, "x", "read", "half", 2), (1, "w", "read", "half", 2), (2, "y", "write", "half", 2), (3, "acc", "write", "half", 2)),
            limits=LIMITS,
        )
    with pytest.raises(GeometryError, match="contiguous"):
        bind_stage_arguments(
            stage,
            contract=abi,
            reflection=_buffers((0, "x", "read", "half", 2), (1, "w", "read", "half", 2), (2, "y", "readwrite", "half", 2), (4, "acc", "readwrite", "half", 2)),
            limits=LIMITS,
        )
    with pytest.raises(GeometryError, match="share buffer slot"):
        bind_stage_arguments(
            stage,
            contract=abi,
            reflection=_buffers((0, "x", "read", "half", 2), (0, "w", "read", "half", 2), (1, "y", "readwrite", "half", 2), (2, "acc", "readwrite", "half", 2)),
            limits=LIMITS,
        )
    with pytest.raises(GeometryError, match="neither an ABI tensor nor a derived scalar"):
        bind_stage_arguments(
            stage,
            contract=abi,
            reflection=_buffers((0, "x", "read", "half", 2), (1, "w", "read", "half", 2), (2, "y", "readwrite", "half", 2), (3, "mystery", "read", "half", 2)),
            limits=LIMITS,
        )


def test_derived_scalar_width_and_read_only_are_enforced(tmp_path):
    """@brief 派生标量必须是 4/8 字节只读整数；声明了却没绑定同样拒绝。"""
    abi, geometry = _resolved(
        tmp_path,
        stages=(
            GeometryStage(
                kernel="main",
                grid=({"ceil_div": ["M*H", 256]}, 1, 1),
                threads=(256, 1, 1),
                derived_scalars=(("ROW_COUNT", 4),),
            ),
        ),
    )
    stage = geometry.active_stages[0]
    good = _buffers((0, "x", "read", "half", 2), (1, "w", "read", "half", 2), (2, "y", "write", "half", 2))
    with pytest.raises(GeometryError, match="integer scalar"):
        bind_stage_arguments(
            stage, contract=abi, reflection=good + _buffers((3, "ROW_COUNT", "read", "half", 2)), limits=LIMITS
        )
    with pytest.raises(GeometryError, match="is 4 bytes"):
        bind_stage_arguments(
            stage, contract=abi, reflection=good + _buffers((3, "ROW_COUNT", "read", "int", 8)), limits=LIMITS
        )
    with pytest.raises(GeometryError, match="read-only"):
        bind_stage_arguments(
            stage, contract=abi, reflection=good + _buffers((3, "ROW_COUNT", "readwrite", "int", 4)), limits=LIMITS
        )
    with pytest.raises(GeometryError, match="not bound by the kernel"):
        bind_stage_arguments(stage, contract=abi, reflection=good, limits=LIMITS)


def test_inactive_stages_are_not_bound_and_a_stage_may_bind_a_subset(tmp_path):
    """@brief 未激活 stage 不参与 ABI；单段可以只绑定张量子集。"""
    abi, geometry = _resolved(
        tmp_path,
        kind="workspace",
        stages=(
            GeometryStage(
                kernel="fill",
                grid=({"ceil_div": ["M*H", 256]}, 1, 1),
                threads=(256, 1, 1),
                workspace=({"name": "scratch", "zero": True},),
            ),
            GeometryStage(
                kernel="reduce",
                grid=({"ceil_div": ["M*H", 256]}, 1, 1),
                threads=(256, 1, 1),
                when=("gt", "M", 10**9),
            ),
        ),
    )
    first = geometry.active_stages[0]
    binding = bind_stage_arguments(
        first,
        contract=abi,
        reflection=_buffers((0, "x", "read", "half", 2), (1, "scratch", "write", "half", 2)),
        limits=LIMITS,
    )
    assert [argument.name for argument in binding.arguments] == ["x", "scratch"]
    inactive = geometry.stages[1]
    with pytest.raises(GeometryError, match="inactive stage"):
        bind_stage_arguments(inactive, contract=abi, reflection=(), limits=LIMITS)


def test_a_pipeline_needs_every_result_written_by_an_active_stage(tmp_path):
    """@brief 没有任何激活 stage 写某个结果张量时在 GPU 之前拒绝。"""
    abi, geometry = _resolved(tmp_path)
    reflection = {"main": _buffers((0, "x", "read", "half", 2), (1, "w", "read", "half", 2), (2, "y", "write", "half", 2))}
    bindings = bind_pipeline_arguments(geometry, contract=abi, reflection=reflection, limits=LIMITS)
    assert len(bindings) == 1
    with pytest.raises(GeometryError, match="no active stage writes the result tensors"):
        bind_pipeline_arguments(
            geometry,
            contract=abi,
            reflection={"main": _buffers((0, "x", "read", "half", 2), (1, "w", "read", "half", 2))},
            limits=LIMITS,
        )
    with pytest.raises(GeometryError, match="no reflection"):
        bind_pipeline_arguments(geometry, contract=abi, reflection={}, limits=LIMITS)


def test_an_over_slot_budget_binding_is_refused(tmp_path):
    """@brief 槽位号超出目标上限即拒绝。"""
    abi, geometry = _resolved(tmp_path)
    stage = geometry.active_stages[0]
    tight = TargetAbiLimits(max_buffer_index=2, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)
    with pytest.raises(GeometryError, match="exceeds the target limit"):
        bind_stage_arguments(
            stage,
            contract=abi,
            reflection=_buffers((0, "x", "read", "half", 2), (1, "w", "read", "half", 2), (3, "y", "write", "half", 2)),
            limits=tight,
        )


def test_binding_is_deterministic_and_has_no_author_field(tmp_path):
    """@brief 同字节候选 + 同几何必得同一绑定；绑定记录没有作者/会话字段。"""
    import dataclasses

    abi, geometry = _resolved(tmp_path)
    stage = geometry.active_stages[0]
    reflection = _buffers((0, "x", "read", "half", 2), (1, "w", "read", "half", 2), (2, "y", "write", "half", 2))
    first = bind_stage_arguments(stage, contract=abi, reflection=reflection, limits=LIMITS).document()
    second = bind_stage_arguments(stage, contract=abi, reflection=reflection, limits=LIMITS).document()
    assert first == second
    names = {field.name for field in dataclasses.fields(ReflectedArgument)}
    for forbidden in ("author", "model", "committee", "session"):
        assert forbidden not in names


def test_a_stage_refuses_a_workspace_it_did_not_declare(tmp_path):
    """@brief 真实绑定时 workspace 必须属于该 stage 的 workspace 计划，不能借 ABI 偷渡。"""
    abi, geometry = _resolved(
        tmp_path,
        kind="workspace",
        stages=(
            GeometryStage(
                kernel="main",
                grid=({"ceil_div": ["M*H", 256]}, 1, 1),
                threads=(256, 1, 1),
                workspace=(),
            ),
        ),
    )
    stage = geometry.active_stages[0]
    with pytest.raises(GeometryError, match="not declared by this stage"):
        bind_stage_arguments(
            stage,
            contract=abi,
            reflection=_buffers((0, "scratch", "write", "half", 2)),
            limits=LIMITS,
        )
    declared = GeometryStage(
        kernel="main",
        grid=({"ceil_div": ["M*H", 256]}, 1, 1),
        threads=(256, 1, 1),
        workspace=({"name": "scratch", "zero": True},),
    )
    abi2, geometry2 = _resolved(tmp_path / "declared", kind="workspace", stages=(declared,))
    binding = bind_stage_arguments(
        geometry2.active_stages[0],
        contract=abi2,
        reflection=_buffers((0, "scratch", "write", "half", 2)),
        limits=LIMITS,
    )
    assert [argument.name for argument in binding.arguments] == ["scratch"]
