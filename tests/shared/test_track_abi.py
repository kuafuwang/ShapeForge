"""``TargetAbiDeriver``: one derivation of the ABI from instance + registered target (A2, CPU).

The ABI locks names/roles/dtypes/spaces/access/shape/strides, the accumulator dtype, the numeric
tolerances and the target's buffer-slot budget.  It locks no tile, grid, threads or fallback
geometry, and it never invents a device limit: a target without the registered limits is refused.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from tcenv.contract_errors import GeometryError, TaskContractError
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    TargetAbiDeriver,
    TargetAbiLimits,
    abi_limits_from_target,
    derive_tensor_memory_contract,
    derived_abi_digest,
    tensor_span_bytes,
)
from tcenv.track.abi import REQUIRED_TARGET_LIMITS

from ..support.v2_tasks import rmsnorm_read_write_opspec, rmsnorm_workspace_opspec, write_task


def _contract(tmp_path: Path, *, kind: str = "plain"):
    """@brief 装配一个 v2 合同。

    @param tmp_path 临时目录。
    @param kind ``plain``/``workspace``/``read_write``。
    @return 任务合同。
    """
    if kind == "workspace":
        return load_task_contract(write_task(tmp_path / "ws", op_name="rmsnorm_ws", opspec=rmsnorm_workspace_opspec()))
    if kind == "read_write":
        return load_task_contract(write_task(tmp_path / "rw", op_name="rmsnorm_rw", opspec=rmsnorm_read_write_opspec()))
    return load_task_contract(write_task(tmp_path / "plain"))


def test_the_deriver_projects_the_instance_and_locks_no_geometry(tmp_path):
    """@brief 派生结果逐字段对应实例；文档里没有 tile/grid/threads/fallback。"""
    contract = _contract(tmp_path)
    instance = contract.instance
    abi = derive_tensor_memory_contract(instance, target=load_target(DEFAULT_TARGET_ID))
    assert abi.backend_id == "metal"
    assert abi.target_id == DEFAULT_TARGET_ID
    assert abi.acc_dtype == instance.acc_dtype
    assert (abi.rtol, abi.atol_rel, abi.exact) == (
        instance.tolerances.rtol,
        instance.tolerances.atol_rel,
        instance.tolerances.exact,
    )
    assert [tensor.name for tensor in abi.tensors] == [tensor.name for tensor in instance.tensors]
    for tensor, binding in zip(abi.tensors, instance.tensors):
        assert (tensor.role, tensor.dtype, tensor.space, tensor.access) == (
            binding.role,
            binding.dtype,
            binding.space,
            binding.access,
        )
        assert tensor.shape == tuple(binding.shape)
        assert tensor.strides == tuple(binding.strides)
    document = abi.document()
    assert set(document) == {"schema_version", "abi_id", "backend_id", "target_id", "tensors", "acc_dtype", "tolerances"}
    blob = repr(document).lower()
    for forbidden in ("tile", "grid", "threads", "num_warps", "warps=", "fallback"):
        assert forbidden not in blob
    assert abi.abi_id == "abi.rmsnorm.fp16.small.apple-m4-max"
    assert derived_abi_digest(abi) == derived_abi_digest(
        derive_tensor_memory_contract(instance, target=load_target(DEFAULT_TARGET_ID))
    )
    other = _contract(tmp_path / "ws_other", kind="workspace").instance
    assert derived_abi_digest(abi) != derived_abi_digest(
        derive_tensor_memory_contract(other, target=load_target(DEFAULT_TARGET_ID))
    )


def test_a_workspace_tensor_is_an_abi_argument_but_not_a_reference_tensor(tmp_path):
    """@brief workspace 进 ABI，不进 reference 初值/结果（A2 的 workspace 能力）。"""
    contract = _contract(tmp_path, kind="workspace")
    instance = contract.instance
    abi = derive_tensor_memory_contract(instance, target=load_target(DEFAULT_TARGET_ID))
    scratch = abi.tensor("scratch")
    assert scratch.space == "workspace" and scratch.access == "write"
    assert "scratch" not in instance.reference_inputs
    assert "scratch" not in instance.reference_outputs
    again = _contract(tmp_path / "again", kind="workspace")
    assert again.instance.reference_inputs == instance.reference_inputs


def test_read_write_tensors_keep_their_initial_and_final_contract(tmp_path):
    """@brief read_write 张量在 ABI 里是 read_write，且同时属于初值与结果合同。"""
    contract = _contract(tmp_path, kind="read_write")
    instance = contract.instance
    abi = derive_tensor_memory_contract(instance, target=load_target(DEFAULT_TARGET_ID))
    for name in instance.read_write_tensors:
        assert abi.tensor(name).access == "read_write"
        assert name in instance.reference_inputs
        assert name in instance.reference_outputs


def test_the_slot_budget_buffer_length_and_capabilities_come_from_the_target(tmp_path):
    """@brief 槽位、缓冲长度与缺项都按目标拒绝，派生器不猜默认值。"""
    instance = _contract(tmp_path).instance
    profile = load_target(DEFAULT_TARGET_ID)
    tight = TargetAbiLimits(
        max_buffer_index=1, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768
    )
    with pytest.raises(TaskContractError, match="buffer slots"):
        TargetAbiDeriver(backend_id="metal", target_id=DEFAULT_TARGET_ID, limits=tight).derive(instance)
    small = TargetAbiLimits(
        max_buffer_index=30,
        max_threads_per_threadgroup=1024,
        max_threadgroup_memory_bytes=32768,
        max_buffer_length=16,
    )
    with pytest.raises(TaskContractError, match="device buffer length"):
        TargetAbiDeriver(backend_id="metal", target_id=DEFAULT_TARGET_ID, limits=small).derive(instance)
    broken = SimpleNamespace(id="no-limits", backend="metal", raw_limits={})
    with pytest.raises(TaskContractError, match="does not register"):
        abi_limits_from_target(broken)
    assert REQUIRED_TARGET_LIMITS == (
        "max_buffer_index",
        "max_threads_per_threadgroup",
        "max_threadgroup_memory_bytes",
    )
    assert abi_limits_from_target(profile).max_buffer_length is None
    assert abi_limits_from_target(profile, max_buffer_length=1 << 20).max_buffer_length == 1 << 20
    with pytest.raises(TaskContractError, match="version"):
        TargetAbiDeriver(backend_id="metal", target_id=DEFAULT_TARGET_ID, limits=tight, version=2)
    with pytest.raises(TaskContractError, match="v2 SemanticInstance"):
        TargetAbiDeriver(backend_id="metal", target_id=DEFAULT_TARGET_ID, limits=tight).derive(object())


def test_the_abi_never_accepts_a_symbolic_or_negative_dimension(tmp_path):
    """@brief 派生要求具体维度；符号/负值不被静默求值。"""
    from tcenv.track.abi import _concrete

    assert _concrete(4, "x") == 4
    for bad in ("M", 1.0, True, None):
        with pytest.raises(GeometryError, match="concrete integer"):
            _concrete(bad, "x")
    with pytest.raises(GeometryError, match="non-negative"):
        _concrete(-1, "x")


def test_the_deriver_has_no_per_op_or_tiling_input(tmp_path):
    """@brief 派生器只接受 instance/target/limits：签名里没有 op 名、tile、grid 或 profile。"""
    names = {field.name for field in dataclasses.fields(TargetAbiDeriver)}
    assert names == {"backend_id", "target_id", "limits", "version"}
    import inspect

    signature = inspect.signature(TargetAbiDeriver.derive)
    assert list(signature.parameters) == ["self", "instance"]


def test_a_strided_tensor_is_measured_by_the_span_it_addresses():
    """@brief 非连续张量按最大偏移算字节跨度，不用 numel 低估。"""
    assert tensor_span_bytes((4,), (1,), 2) == 8
    assert tensor_span_bytes((3, 2), (8, 1), 2) == (2 * 8 + 1 + 1) * 2
    assert tensor_span_bytes((0,), (1,), 2) == 0
    assert tensor_span_bytes((), (), 2) == 0
    with pytest.raises(GeometryError, match="one stride per dimension"):
        tensor_span_bytes((2,), (1, 2), 2)
    for shape, strides, size in (((2,), (-1,), 2), ((2,), (1,), 0), ((-1,), (1,), 2)):
        with pytest.raises(GeometryError):
            tensor_span_bytes(shape, strides, size)
