"""V2 Dyn-0 的 workspace 容量与分配范围（CPU；不开设备、不编译）。

V2 把 ``work_items``/``item_counts``/``handoff_flags``/``handoff_states``/``token_stats`` 作为
真实的 OpSpec workspace 声明，于是 ABI 里出现了一批候选可能用、也可能完全不用的大 scratch
（``log_decay_tab`` 在 T=8192/H=96/D=128 下约 384 MiB）。

这里在 CPU 上钉住三件事：

* 分配范围是**全部可能 stage 的并集**（含 ``when`` 为假者），而不是 ABI 里所有 workspace：
  分配必须发生在任何 dispatch 之前，所以“本 case 没激活”不能作为不分配的理由；完全不可能被任何
  stage 绑定的 workspace 不分配。同一 workspace 的 zero 策略必须跨 possible stages 唯一。
* launch 块声明的 workspace 字节必须与 ABI span 逐字节一致；未声明但被 stage 使用、或声明了 ABI
  没有的 workspace 都被拒绝。
* Dyn-0 没有 ``runtime_scalars``/``LaunchGeometry``：几何是构建期静态整数，唯一“envelope”就是
  静态 grid 上限（``MAX_THREADGROUPS``），所以“resolved grid 超 envelope”这一条负例在 Dyn-0 下
  不适用，它由静态 launch 解析拒绝代替。
"""

from __future__ import annotations

import pytest

from tcenv.backends.metal.launch import LaunchRejected, parse_launch_spec
from tcenv.backends.metal.v2_runtime import possible_workspace_uses, workspace_size_problems
from tcenv.contract_errors import GeometryError
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.track import (
    ExecutionGeometry,
    GeometryStage,
    TargetAbiLimits,
    derive_tensor_memory_contract,
    resolve_geometry,
)
from tcenv.track.abi import abi_limits_from_target
from tcenv.target import load_target

KDA_DIR = REPO_ROOT / "tasks" / "varlen_kda"
LIMITS = TargetAbiLimits(max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)


def _resolved(size: str = "mixed", stages=()):
    """@brief 用 varlen_kda 的真实 ABI 求值一份 CPU 几何。

    @param size 档位名。
    @param stages ``GeometryStage`` 序列。
    @return ``(contract, abi, geometry)``。
    """
    contract = load_task_contract(KDA_DIR, dtype="fp16", size=size)
    abi = derive_tensor_memory_contract(contract.instance, target=load_target("apple-m4-max"))
    geometry = resolve_geometry(
        ExecutionGeometry(source="candidate", stages=tuple(stages)),
        instance=contract.instance,
        contract=abi,
        limits=LIMITS,
    )
    return contract, abi, geometry


def _stage(kernel: str, workspace=(), *, when=None) -> GeometryStage:
    """@brief 一个最小的静态 stage（grid=1、threads=1）。

    @param kernel 核名。
    @param workspace ``{"name": ..., "zero": ...}`` 序列。
    @param when 可选的激活条件。
    @return stage 声明。
    """
    return GeometryStage(kernel=kernel, grid=(1, 1, 1), threads=(1, 1, 1), workspace=tuple(workspace), when=when)


def test_the_possible_workspace_union_includes_inactive_stages():
    """@brief 并集包含未激活 stage 的 workspace：分配点必须在任何 dispatch 之前。"""
    _, _, geometry = _resolved(
        stages=(
            _stage("recurrence", [{"name": "token_stats", "zero": False}]),
            _stage("handoff", [{"name": "handoff_flags", "zero": False}], when=0),
        )
    )

    assert [use.name for use in geometry.workspace_uses] == ["token_stats"]
    assert [use.name for use in possible_workspace_uses(geometry)] == ["token_stats", "handoff_flags"]


def test_a_workspace_no_stage_can_bind_is_not_possible():
    """@brief 没有任何 stage 声明的 workspace（例如 naive 候选的 ``log_decay_tab``）不进入分配集合。"""
    contract, _, geometry = _resolved(stages=(_stage("varlen_kda"),))

    declared = [tensor.name for tensor in contract.instance.tensors if tensor.space == "workspace"]
    assert declared == ["log_decay_tab", "token_stats", "work_items", "item_counts", "handoff_flags", "handoff_states"]
    assert possible_workspace_uses(geometry) == ()


def test_a_zero_policy_conflict_across_possible_stages_is_refused():
    """@brief 同一 workspace 被不同 possible stage 声明 zero 与非 zero 时拒绝（含未激活 stage）。

    现有 ``resolve_geometry`` 只对**激活** stage 做这条检查；V2 的分配并集把未激活 stage 也算
    进“可能使用”，所以这里的冲突必须在分配前拒绝，否则“什么时候清零”会随 case 改变。
    """
    _, _, geometry = _resolved(
        stages=(
            _stage("recurrence", [{"name": "token_stats", "zero": True}]),
            _stage("handoff", [{"name": "token_stats", "zero": False}], when=0),
        )
    )

    with pytest.raises(GeometryError, match="declared both zero and not-zero"):
        possible_workspace_uses(geometry)


def test_a_launch_must_declare_the_exact_abi_workspace_span():
    """@brief launch 的 workspace 字节必须等于 ABI span；漏声明或多声明都被拒绝。"""
    contract, abi, geometry = _resolved(
        stages=(_stage("recurrence", [{"name": "token_stats", "zero": False}]),)
    )
    tokens, heads, dim = (contract.instance.symbols[name] for name in ("total_tokens", "H", "D"))
    spans = {tensor.name: 1 for tensor in abi.tensors if tensor.space == "workspace"}
    assert spans, "the varlen_kda ABI must declare workspace capacities"
    exact_span = tokens * heads * 3 * 4

    ok = parse_launch_spec(
        "/* taic-launch\nmath safe\n"
        f"workspace token_stats {tokens}*{heads}*3*4\n"
        "launch recurrence grid=1,1,1 threads=1,1,1 workspace=token_stats\n*/\n",
        contract.instance.symbols,
    )
    assert workspace_size_problems(abi, geometry, ok) == []

    undersized = parse_launch_spec(
        "/* taic-launch\nmath safe\n"
        "workspace token_stats 4\n"
        "launch recurrence grid=1,1,1 threads=1,1,1 workspace=token_stats\n*/\n",
        contract.instance.symbols,
    )
    problems = workspace_size_problems(abi, geometry, undersized)
    assert problems and "launch declares 4 bytes" in problems[0], problems
    assert exact_span != 4

    missing = parse_launch_spec(
        "/* taic-launch\nmath safe\nlaunch recurrence grid=1,1,1 threads=1,1,1\n*/\n",
        contract.instance.symbols,
    )
    problems = workspace_size_problems(abi, geometry, missing)
    assert problems and "resolved geometry uses workspace 'token_stats'" in problems[0], problems

    unknown = parse_launch_spec(
        "/* taic-launch\nmath safe\nworkspace not_in_abi 16\n"
        "launch recurrence grid=1,1,1 threads=1,1,1 workspace=not_in_abi\n*/\n",
        contract.instance.symbols,
    )
    problems = workspace_size_problems(abi, geometry, unknown)
    assert problems and "the ABI does not contain" in problems[0], problems


def test_dyn0_geometry_is_static_and_has_no_runtime_scalars():
    """@brief Dyn-0 的几何只有静态整数：没有 runtime scalar 绑定，也没有逐 case LaunchGeometry。

    “resolved grid 超 envelope”是 Dyn-1 的负例。Dyn-0 的对应事实是：grid 在构建期就是确定整数，
    越过静态上限的 grid 在 launch 解析/ABI 校验期被拒绝，而不是在运行期比较 envelope。
    """
    _, abi, geometry = _resolved(stages=(_stage("recurrence", [{"name": "token_stats", "zero": False}]),))

    document = geometry.document()
    assert set(document) == {"resolution_version", "source", "stages"}
    assert "runtime_scalars" not in document and "dyn_level" not in document
    assert "launch_geometry" not in document
    for stage in document["stages"]:
        assert all(isinstance(value, int) for value in stage["grid"])
        assert all(isinstance(value, int) for value in stage["threads"])

    # 静态 envelope：相邻的“超上限”负例是 launch 解析期的 MAX_THREADGROUPS 拒绝。
    limit = 1 << 24
    with pytest.raises(LaunchRejected, match="threadgroups"):
        parse_launch_spec(
            f"/* taic-launch\nmath safe\nlaunch recurrence grid={limit + 1},1,1 threads=1,1,1\n*/\n",
            {},
        )


def test_the_real_target_registers_the_buffer_slots_the_new_workspaces_need():
    """@brief 6 个 workspace 加 12 个数据张量仍在目标槽位预算内（ABI 派生本身就会拒绝越界）。"""
    contract = load_task_contract(KDA_DIR, dtype="fp16", size="mixed")
    abi = derive_tensor_memory_contract(contract.instance, target=load_target("apple-m4-max"))
    limits = abi_limits_from_target(load_target("apple-m4-max"))
    assert len(abi.tensors) == 18
    assert len(abi.tensors) <= limits.buffer_slots
