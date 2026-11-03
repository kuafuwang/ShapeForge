"""Typed ExecutionGeometry resolution, symbolic evaluation, when/derived scalars (A2, CPU).

These are the A2 geometry rules: expressions evaluate exactly, ``when`` gates stages, named
``Derived Scalars`` depend only on symbols/attrs, workspace uses must exist, resources must fit
the target, and the independent fallback is a separate geometry that enters the execution
identity.  The A1 field sets are untouched, so an A1 checkout still sees a field projection only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcenv.contract_errors import GeometryError, TaskContractError
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    ExecutionGeometry,
    GeometryStage,
    ResolvedGeometry,
    TargetAbiDeriver,
    TargetAbiLimits,
    abi_limits_from_target,
    derive_tensor_memory_contract,
    dispatch_parity,
    geometry_pair,
    resolve_geometry,
    select_geometry,
    selection_parity,
)

from ..support.v2_tasks import (
    rmsnorm_opspec,
    rmsnorm_read_write_opspec,
    rmsnorm_workspace_opspec,
    write_task,
)

LIMITS = TargetAbiLimits(
    max_buffer_index=30,
    max_threads_per_threadgroup=1024,
    max_threadgroup_memory_bytes=32768,
    max_buffer_length=1 << 30,
)


def _instance(tmp_path: Path, *, workspace: bool = False, read_write: bool = False):
    """@brief 构造一个 v2 实例（默认 RMSNorm，可选 workspace/read_write 变体）。

    @param tmp_path 临时目录。
    @param workspace 是否使用 workspace 夹具。
    @param read_write 是否使用 read_write 夹具。
    @return ``(instance, contract, limits)``。
    """
    if workspace:
        contract = load_task_contract(
            write_task(tmp_path / "ws", op_name="rmsnorm_ws", opspec=rmsnorm_workspace_opspec())
        )
    elif read_write:
        contract = load_task_contract(
            write_task(tmp_path / "rw", op_name="rmsnorm_rw", opspec=rmsnorm_read_write_opspec())
        )
    else:
        contract = load_task_contract(write_task(tmp_path / "plain"))
    instance = contract.instance
    abi = derive_tensor_memory_contract(instance, target=load_target(DEFAULT_TARGET_ID))
    return instance, abi, LIMITS


def _stage(**overrides) -> GeometryStage:
    """@brief 一个绑定全部张量的 stage 的默认字段。

    @param overrides 覆盖项。
    @return stage。
    """
    fields = {
        "kernel": "rmsnorm_kernel",
        "grid": ("ceil_div(M*H, 256)", 1, 1),
        "threads": (256, 1, 1),
    }
    fields.update(overrides)
    return GeometryStage(**fields)


def test_symbolic_geometry_evaluates_exactly_and_text_agrees_with_json(tmp_path):
    """@brief 符号几何按 OpSpec symbol 求值；文本与 JSON term 得出同一结果。"""
    instance, contract, limits = _instance(tmp_path)
    text = ExecutionGeometry(source="candidate", stages=(_stage(grid=("M*H/256", 1, 1)),))
    tree = ExecutionGeometry(source="candidate", stages=(_stage(grid=({"ceil_div": ["M*H", 256]}, 1, 1)),))
    resolved_text = resolve_geometry(text, instance=instance, contract=contract, limits=limits)
    resolved_tree = resolve_geometry(tree, instance=instance, contract=contract, limits=limits)
    want = instance.symbols["M"] * instance.symbols["H"]
    assert want % 256 == 0
    assert resolved_text.stages[0].grid == (want // 256, 1, 1)
    assert resolved_tree.stages[0].grid == resolved_text.stages[0].grid
    # ``ceil_div`` and exact ``/`` agree when the division is exact; an inexact division is refused
    # rather than rounded, so no evaluation can silently disagree with a launch block.
    inexact = ExecutionGeometry(source="candidate", stages=(_stage(grid=("M*H/255", 1, 1)),))
    with pytest.raises(GeometryError, match="divide evenly"):
        resolve_geometry(inexact, instance=instance, contract=contract, limits=limits)


def test_an_attribute_may_size_the_launch_but_a_tensor_may_not(tmp_path):
    """@brief attrs 是语义参数，可以参与几何；张量名与未知名字不能。"""
    document = rmsnorm_opspec("rmsnorm_attr")
    document["attrs"]["SPLIT_FACTOR"] = {"dtype": "int", "default": 4}
    contract = load_task_contract(write_task(tmp_path / "attr", op_name="rmsnorm_attr", opspec=document))
    instance = contract.instance
    abi = derive_tensor_memory_contract(instance, target=load_target(DEFAULT_TARGET_ID))
    contract, limits = abi, LIMITS
    assert instance.resolved_attrs["SPLIT_FACTOR"] == 4
    assert instance.resolved_attrs["EPS"] == pytest.approx(1e-06)
    attr_name = "SPLIT_FACTOR"
    stage = _stage(threads=(attr_name, 1, 1))
    resolved = resolve_geometry(
        ExecutionGeometry(source="candidate", stages=(stage,)), instance=instance, contract=contract, limits=limits
    )
    assert resolved.stages[0].threads[0] == int(instance.resolved_attrs[attr_name])
    # A float attr is a semantic parameter, not a launch size: referencing it is an undeclared
    # name, never a silent coercion.
    with pytest.raises(GeometryError, match="undeclared name"):
        resolve_geometry(
            ExecutionGeometry(source="candidate", stages=(_stage(threads=("EPS", 1, 1)),)),
            instance=instance,
            contract=contract,
            limits=limits,
        )
    for bad in ("x", "NO_SUCH_SYMBOL"):
        with pytest.raises(GeometryError, match="undeclared name"):
            resolve_geometry(
                ExecutionGeometry(source="candidate", stages=(_stage(threads=(bad, 1, 1)),)),
                instance=instance,
                contract=contract,
                limits=limits,
            )


def test_when_gates_stages_and_a_zero_active_stage_geometry_is_refused(tmp_path):
    """@brief ``when`` 让 stage 在该实例不激活；全部不激活时拒绝，不伪造空 dispatch。"""
    instance, contract, limits = _instance(tmp_path)
    conditional = ExecutionGeometry(
        source="candidate",
        stages=(
            _stage(kernel="first_kernel"),
            _stage(kernel="second_kernel", when=("gt", "M", 0)),
            _stage(kernel="third_kernel", when=("gt", "M", 10**9)),
        ),
    )
    resolved = resolve_geometry(conditional, instance=instance, contract=contract, limits=limits)
    assert [stage.kernel for stage in resolved.active_stages] == ["first_kernel", "second_kernel"]
    assert resolved.stages[2].active is False and resolved.stages[2].when is False
    assert "third_kernel" in [stage.kernel for stage in resolved.stages]

    inactive = ExecutionGeometry(source="candidate", stages=(_stage(when=("gt", "M", 10**9)),))
    with pytest.raises(GeometryError, match="inactive"):
        resolve_geometry(inactive, instance=instance, contract=contract, limits=limits)


def test_derived_scalars_are_named_by_the_candidate_and_depend_only_on_symbols_and_attrs(tmp_path):
    """@brief 派生标量按名求值；不能依赖另一个派生标量/张量，也不能重名或遮蔽。"""
    instance, contract, limits = _instance(tmp_path)
    stage = _stage(
        derived_scalars=(
            ("ELEMENTS", "M*H"),
            ("BLOCKS", {"ceil_div": ["ELEMENTS", 256]}),
        )
    )
    with pytest.raises(GeometryError, match="only on symbols/attrs"):
        resolve_geometry(
            ExecutionGeometry(source="candidate", stages=(stage,)), instance=instance, contract=contract, limits=limits
        )
    ok = _stage(derived_scalars=(("ELEMENTS", "M*H"), ("BLOCKS", {"ceil_div": ["M*H", 256]})))
    resolved = resolve_geometry(
        ExecutionGeometry(source="candidate", stages=(ok,)), instance=instance, contract=contract, limits=limits
    )
    assert dict(resolved.stages[0].derived_scalars) == {
        "ELEMENTS": instance.symbols["M"] * instance.symbols["H"],
        "BLOCKS": (instance.symbols["M"] * instance.symbols["H"]) // 256,
    }
    assert resolved.stages[0].document()["derived_scalars"][0]["name"] == "ELEMENTS"

    for bad, message in (
        ((("ELEMENTS", 1), ("ELEMENTS", 2)), "duplicate"),
        ((("x", 1),), "derived-scalar name"),
        ((("M", 1),), "shadows"),
        ((("ELEMENTS", "NO_SUCH_SYMBOL"),), "only on symbols/attrs"),
    ):
        with pytest.raises(GeometryError, match=message):
            resolve_geometry(
                ExecutionGeometry(source="candidate", stages=(_stage(derived_scalars=bad),)),
                instance=instance,
                contract=contract,
                limits=limits,
            )
    # A derived scalar is a named integer, not a second symbol space: the launch shape and the
    # stage predicate may use it exactly as they may use a symbol, and both sides evaluate to the
    # same value that ``setBytes`` binds.
    used = _stage(
        grid=("BLOCKS", 1, 1),
        threads=("ELEMENTS", 1, 1),
        threadgroup_bytes={"*": ["BLOCKS", 64]},
        when=("gt", "BLOCKS", 0),
        derived_scalars=(("ELEMENTS", "H"), ("BLOCKS", {"ceil_div": ["M*H", 256]})),
    )
    resolved_used = resolve_geometry(
        ExecutionGeometry(source="candidate", stages=(used,)), instance=instance, contract=contract, limits=limits
    )
    elements = instance.symbols["H"]
    blocks = (instance.symbols["M"] * instance.symbols["H"]) // 256
    assert resolved_used.stages[0].grid[0] == blocks
    assert resolved_used.stages[0].threads[0] == elements
    assert resolved_used.stages[0].threadgroup_bytes == blocks * 64
    assert resolved_used.stages[0].when is True


def test_workspace_uses_must_be_declared_and_have_one_clearing_policy(tmp_path):
    """@brief workspace 只在 ABI 里声明过才可用；同一 workspace 的 zero 策略必须唯一。"""
    instance, contract, limits = _instance(tmp_path, workspace=True)
    ok = ExecutionGeometry(
        source="candidate",
        stages=(_stage(kernel="stage_one", workspace=({"name": "scratch", "zero": True},)),),
    )
    resolved = resolve_geometry(ok, instance=instance, contract=contract, limits=limits)
    assert resolved.workspace_uses[0].name == "scratch"
    assert resolved.workspace_uses[0].zero is True
    # A bare name means "not cleared by the harness".
    bare = resolve_geometry(
        ExecutionGeometry(source="candidate", stages=(_stage(workspace=("scratch",)),)),
        instance=instance,
        contract=contract,
        limits=limits,
    )
    assert bare.workspace_uses[0].zero is False
    with pytest.raises(GeometryError, match="not a workspace tensor"):
        resolve_geometry(
            ExecutionGeometry(source="candidate", stages=(_stage(workspace=("y",)),)),
            instance=instance,
            contract=contract,
            limits=limits,
        )
    conflicting = ExecutionGeometry(
        source="candidate",
        stages=(
            _stage(kernel="a", workspace=({"name": "scratch", "zero": True},)),
            _stage(kernel="b", workspace=({"name": "scratch", "zero": False},)),
        ),
    )
    pair = geometry_pair(conflicting, instance=instance, contract=contract, limits=limits)
    # One workspace must have one clearing policy and resolving is where that is refused, not a
    # property a later consumer may forget to read.
    assert pair.candidate is None and "both zero and not-zero" in (pair.candidate_error or "")
    with pytest.raises(GeometryError, match="both zero and not-zero"):
        resolve_geometry(conflicting, instance=instance, contract=contract, limits=limits)


def test_grid_threads_and_threadgroup_memory_must_fit_the_target(tmp_path):
    """@brief 线程数、负值与 threadgroup 内存都按目标上限校验。"""
    instance, contract, limits = _instance(tmp_path)
    with pytest.raises(GeometryError, match="threads per group"):
        resolve_geometry(
            ExecutionGeometry(source="candidate", stages=(_stage(threads=(2048, 1, 1)),)),
            instance=instance,
            contract=contract,
            limits=limits,
        )
    with pytest.raises(GeometryError, match="must be >= 1"):
        resolve_geometry(
            ExecutionGeometry(source="candidate", stages=(_stage(grid=(0, 1, 1)),)),
            instance=instance,
            contract=contract,
            limits=limits,
        )
    with pytest.raises(GeometryError, match="threadgroup_bytes"):
        resolve_geometry(
            ExecutionGeometry(source="candidate", stages=(_stage(threadgroup_bytes=65536),)),
            instance=instance,
            contract=contract,
            limits=limits,
        )
    with pytest.raises(GeometryError, match="three terms"):
        resolve_geometry(
            ExecutionGeometry(source="candidate", stages=(_stage(grid=(1, 1)),)),
            instance=instance,
            contract=contract,
            limits=limits,
        )


def test_the_fallback_is_independent_and_enters_the_execution_identity(tmp_path):
    """@brief fallback 独立求值、必须显式允许，且永远同时进入执行身份。"""
    instance, contract, limits = _instance(tmp_path)
    candidate = ExecutionGeometry(source="candidate", stages=(_stage(grid=({"ceil_div": ["M*H", 256]}, 1, 1)),))
    fallback = ExecutionGeometry(source="fallback", stages=(_stage(kernel="fallback_kernel", threads=(64, 1, 1)),))
    pair = geometry_pair(candidate, fallback, instance=instance, contract=contract, limits=limits)
    assert pair.candidate is not None and pair.fallback is not None
    selection = select_geometry(pair)
    assert selection.provider == "candidate" and selection.used_fallback is False
    assert selection.identity == pair.identity

    other_fallback = ExecutionGeometry(source="fallback", stages=(_stage(kernel="other_fallback", threads=(32, 1, 1)),))
    other_pair = geometry_pair(candidate, other_fallback, instance=instance, contract=contract, limits=limits)
    assert other_pair.identity != pair.identity, "a fallback change must move the execution identity"
    without_fallback = geometry_pair(candidate, instance=instance, contract=contract, limits=limits)
    assert without_fallback.identity != pair.identity

    # A fallback is never silently used: an unusable candidate needs an explicit opt-in.
    broken = ExecutionGeometry(source="candidate", stages=(_stage(threads=(4096, 1, 1)),))
    broken_pair = geometry_pair(broken, fallback, instance=instance, contract=contract, limits=limits)
    assert broken_pair.candidate is None and broken_pair.candidate_error
    with pytest.raises(GeometryError, match="explicit"):
        select_geometry(broken_pair)
    chosen = select_geometry(broken_pair, allow_fallback=True)
    assert chosen.used_fallback is True and chosen.provider == "fallback"
    assert chosen.identity == broken_pair.identity
    # ... and a fallback cannot pretend to be the candidate.
    with pytest.raises(GeometryError, match="source"):
        geometry_pair(fallback, instance=instance, contract=contract, limits=limits)
    with pytest.raises(GeometryError, match="expected 'fallback'"):
        geometry_pair(
            candidate,
            ExecutionGeometry(source="candidate", stages=(_stage(),)),
            instance=instance,
            contract=contract,
            limits=limits,
        )


def test_dispatch_parity_catches_a_worker_that_drifted_from_the_recorded_geometry(tmp_path):
    """@brief 逐段 dispatch 与记录几何不一致即 mismatch（A4 的 worker 检查在此可测）。"""
    instance, contract, limits = _instance(tmp_path)
    geometry = ExecutionGeometry(
        source="candidate",
        stages=(
            _stage(kernel="one", grid=(4, 1, 1), threads=(64, 1, 1)),
            _stage(kernel="two", grid=(2, 1, 1), threads=(32, 1, 1)),
        ),
    )
    fallback = ExecutionGeometry(
        source="fallback", stages=(_stage(kernel="fb", grid=(1, 1, 1), threads=(32, 1, 1)),)
    )
    resolved: ResolvedGeometry = resolve_geometry(geometry, instance=instance, contract=contract, limits=limits)
    observed = [(stage.kernel, stage.grid, stage.threads, 0, ()) for stage in resolved.active_stages]
    assert dispatch_parity(resolved, observed) == []
    drifted = [observed[0], ("two", (2, 1, 1), (64, 1, 1), 0, ())]
    assert dispatch_parity(resolved, drifted)
    assert dispatch_parity(resolved, observed[:1])
    with pytest.raises(GeometryError, match="dispatch record"):
        dispatch_parity(resolved, [("one", (4, 1, 1), (64, 1, 1))])

    # A worker that runs the independent fallback must be a mismatch even when the stages it
    # dispatched happen to be legal; the provider is compared before the dispatch records.
    pair = geometry_pair(geometry, fallback, instance=instance, contract=contract, limits=limits)
    selection = select_geometry(pair)
    assert selection_parity(selection, provider="candidate", observed=observed) == []
    wrong_provider = selection_parity(selection, provider="fallback", observed=observed)
    assert wrong_provider and "fallback" in wrong_provider[0]
    ran_fallback = [(stage.kernel, stage.grid, stage.threads, 0, ()) for stage in pair.fallback.active_stages]
    assert selection_parity(selection, provider="candidate", observed=ran_fallback)


def test_the_geometry_plane_has_no_style_gate_or_sdk_feature_whitelist():
    """@brief A2 删除了循环/goto 风格审查：几何平面不扫描内核写法，也不设 SDK 白名单。"""
    source = (Path(__file__).resolve().parents[2] / "tcenv" / "track" / "geometry.py").read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]
    for forbidden in ("goto", "while (", "while(", "for (", "persistent", "whitelist", "allowlist"):
        assert forbidden not in body.lower(), f"the geometry plane reviews kernel style: {forbidden}"
    assert "screen" not in body
    # A kernel name is an identifier, not a pattern: any name the compiler accepts is the
    # candidate's business.
    import dataclasses

    names = {field.name for field in dataclasses.fields(ExecutionGeometry)}
    assert names == {"source", "stages", "schema_version"}


def test_the_a1_abi_deriver_uses_registered_limits_only():
    """@brief 派生器只读 target 档案登记的通用上限，缺项时拒绝而不是用默认值。"""
    profile = load_target(DEFAULT_TARGET_ID)
    limits = abi_limits_from_target(profile)
    assert limits.max_buffer_index == profile.raw_limits["max_buffer_index"]
    assert limits.buffer_slots == profile.raw_limits["max_buffer_index"] + 1
    from types import SimpleNamespace

    broken = SimpleNamespace(id="no-limits", backend="metal", raw_limits={"counter_sample_us": 0})
    with pytest.raises(TaskContractError, match="does not register"):
        TargetAbiDeriver.for_target(broken)
    with pytest.raises(TaskContractError, match=">= 1"):
        TargetAbiLimits(max_buffer_index=0, max_threads_per_threadgroup=1, max_threadgroup_memory_bytes=1)
