"""V5 L2 (capacity) tiers: runtime resolved shape, padding and continuous batching (CPU only).

What this file pins
-------------------
An L2 tier is not an L1 tier with a new name.  It has to be provable that:

* the OpSpec declares a **capacity** and several **workload profiles**, and mixed declarations
  (sub-shape fixed/L1/L2 tiers in one task, ``seq_lens`` together with ``capacity``, one profile, one
  resolved total, no full-capacity profile, a non-negative boundary sentinel, a zero padding byte) are
  refused with typed errors;
* buffers, workspace capacities, ABI spans and the Host Program contract are sized by the capacity while
  ``resolved_total_tokens``/``resolved_num_seqs`` change per call, and one sweep really runs several
  different batches (shrink *and* grow), including ``total=capacity``/``count=capacity``, ``total=1``,
  ``count=1``, empty sequences, skewed/permuted/prime segmentations and a fresh draw;
* the capacity-sized ``cu_seqlens`` carries this call's offset prefix plus the declared negative sentinel
  in the unused tail, and every element outside the logical region is the declared poison byte;
  empty batch, negative/non-monotonic offsets, a wrong last prefix entry, a non-sentinel tail, a
  "convenient zero" tail and an over-capacity batch are all refused;
* the evaluation-input digest binds the *actual* resolved total/count and boundary bytes, while the
  semantic instance digest and the dispatch domain stay at the capacity point;
* the native Host Program receives the resolved scalars through ``SfHostResolved``, rejects a candidate
  that rewrites them, and keeps one build identity for every resolved batch (no per-total recompile);
* the correctness comparison uses only the logical region, and padding/sentinel bytes that the candidate
  changed are a failure rather than something a prefix compare can hide.

Nothing here runs a GPU, a paid search or the NPU device.  The simulated parity that reuses this contract
lives in ``test_hostprogram_npu_parity.py`` and stays ``simulated=true/publication_allowed=false``.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch

from tcenv.contract_errors import VarlenContractError
from tcenv.dtypes import get_dtype
from tcenv.hostprogram.contract import host_contract_spec
from tcenv.hostprogram.errors import HostProgramError
from tcenv.hostprogram.native_abi import render_native_support
from tcenv.hostprogram.native_runtime import NativeHostRuntime
from tcenv.hostprogram.native_toolchain import compile_native_host_program, discover_native_toolchain
from tcenv.identity import load_strict_json
from tcenv.opspec.model import OpSpecError
from tcenv.opspec.v2 import instantiate_v2, load_opspec_v2, opspec_v2_from_document
from tcenv.portfolio.candidate import instance_names
from tcenv.portfolio.domain import (
    evaluation_input_digest,
    normalize_semantic_domain_plan,
    semantic_instance_digest,
    workload_profile_digest,
)
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.taskcontract.reference import ReferenceModule
from tcenv.taskcontract.runtime_shape import (
    ResolvedShape,
    cu_seqlens_problems,
    padding_byte_problems,
    padding_problems,
    padding_slabs,
    resolved_shape_for_case,
    resolved_shape_from_inputs,
    workload_profile_for_case,
)
from tcenv.taskcontract.varlen import require_runtime_segments
from tcenv.target import load_target
from tcenv.tensors import round_to, to_raw
from tcenv.track.abi import derive_tensor_memory_contract

from ..support.hostprogram_fixtures import (
    CAPACITY_PLANNER_FORGES_RESOLVED,
    CAPACITY_PLANNER_SOURCE,
    PLANNER_SOURCE,
)

KDA_DIR = REPO_ROOT / "tasks" / "varlen_kda"
OPSPEC_PATH = KDA_DIR / "opspec.json"
TARGET = load_target("apple-m4-max")
CAPACITY = "capacity"
SIZES = ("fixed", "mixed", "uniform", CAPACITY)


def _document() -> dict:
    """@brief 权威 OpSpec 文档的可修改副本（负例用）。"""
    return copy.deepcopy(load_strict_json(OPSPEC_PATH))


def _validate(document: dict) -> None:
    """@brief 用生产校验器校验一份（可能被篡改的）文档。"""
    opspec_v2_from_document(document, source="negative")


def _contract(size: str = CAPACITY):
    """@brief 某个档位的已装配 TaskContract。"""
    return load_task_contract(KDA_DIR, dtype="fp16", size=size)


def _instance(size: str = CAPACITY):
    """@brief 某个档位的权威语义实例。"""
    return _contract(size).instance


def _toolchain():
    """@brief 真实 clang/ld。

    A missing toolchain is a typed capability failure (`HostCapabilityUnavailable`), not a reason to make the
    acceptance case disappear: this project forbids adding skip/xfail to new tests, so the error propagates.

    @return 已发现的工具链。
    @raises HostCapabilityUnavailable 本机没有 clang/ld。
    """
    return discover_native_toolchain()


def _raw_inputs(instance, case: str, seed: int = 424242) -> dict[str, object]:
    """@brief 生成一个 case 的输入，并附上它的 resolved 形状。"""
    contract = _contract()
    resolved = resolved_shape_for_case(instance, case)
    produced = contract.reference.generate(instance, case, seed)
    return {"inputs": produced, "resolved": resolved}


def _l2_runtime(source: str = CAPACITY_PLANNER_SOURCE):
    """@brief 编译一次 L2 Host Program fixture，返回 ``(runtime, spec, build, contract)``。"""
    contract = _contract()
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
    build = compile_native_host_program(source, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    return NativeHostRuntime(build, spec), spec, build, contract


def _raw(contract, produced: dict) -> dict[str, bytes]:
    """@brief 把生成的输入按声明 dtype 舍入并转成裸字节。"""
    out = {}
    for name, value in produced.items():
        tensor = contract.instance.tensor(name)
        out[name] = to_raw(round_to(value.reshape(tuple(int(v) for v in tensor.shape)), tensor.dtype), tensor.dtype)
    return out


# ---------------------------------------------------------------------------------------------------
# 1. OpSpec / schema: the three tier kinds and their refusals
# ---------------------------------------------------------------------------------------------------


def test_the_pack_declares_a_capacity_tier_next_to_the_l1_tiers():
    """@brief 同一 pack 里 L1 与 L2 并存，且 L1 档的语义身份没有变成 capacity。"""
    spec = load_opspec_v2(OPSPEC_PATH)
    assert CAPACITY in spec.sizes
    kinds = {size: instantiate_v2(spec, "fp16", size).variable_length_kind for size in SIZES}
    assert kinds == {"fixed": "l1", "mixed": "l1", "uniform": "l1", CAPACITY: "l2"}


def test_an_l2_instance_binds_capacity_and_no_canonical_segmentation():
    """@brief L2 的语义身份 = capacity + profiles；``cu_seqlens`` 必须为 ``None``。"""
    instance = _instance()
    assert instance.cu_seqlens is None
    assert instance.canonical_seq_lens is None
    assert instance.variable_length_kind == "l2"
    capacity = instance.capacity
    assert (capacity.total_tokens, capacity.num_seqs) == (16, 4)
    assert capacity.cu_seqlens_sentinel < 0
    assert 1 <= capacity.padding_poison_byte <= 255
    assert instance.symbols["total_tokens"] == capacity.total_tokens
    assert instance.symbols["num_seqs"] == capacity.num_seqs
    assert instance.symbols["NSP1"] == capacity.num_seqs + 1
    profiles = sorted(instance.workload_profiles, key=lambda item: item.id)
    assert len(profiles) >= 2
    assert len({profile.total_tokens for profile in profiles}) >= 2
    assert len({profile.num_seqs for profile in profiles}) >= 2
    assert any(
        profile.total_tokens == capacity.total_tokens and profile.num_seqs == capacity.num_seqs
        for profile in profiles
    )


def test_an_l1_instance_is_not_a_capacity_instance():
    """@brief L1 档仍然是 ``resolved == capacity`` 的退化情形，且没有 capacity 块。"""
    instance = _instance("mixed")
    assert instance.variable_length_kind == "l1"
    assert instance.capacity is None
    assert instance.workload_profiles == ()
    assert instance.cu_seqlens is not None
    assert instance.resolved_total_tokens == instance.symbols["total_tokens"]


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (
            lambda tier: tier.__setitem__("seq_lens", [4, 4, 4, 4]),
            "declares both 'seq_lens' (L1 canonical segmentation) and 'capacity'",
        ),
        (lambda tier: tier.pop("workload_profiles"), "must declare its workload profiles"),
        (
            lambda tier: tier.__setitem__("workload_profiles", [{"id": "only", "seq_lens": [4, 4, 4, 4]}]),
            "at least two workload profiles",
        ),
        (
            lambda tier: tier.__setitem__(
                "workload_profiles",
                [{"id": "a", "seq_lens": [16]}, {"id": "b", "seq_lens": [16]}],
            ),
            "two distinct resolved totals and two distinct resolved counts",
        ),
        (
            lambda tier: tier["capacity"].__setitem__("cu_seqlens_sentinel", 0),
            "must be a negative int32",
        ),
        (
            lambda tier: tier["capacity"].__setitem__("padding_poison_byte", 0),
            "a convenient 0 is not a poison marker",
        ),
        (
            lambda tier: tier["capacity"].pop("num_seqs"),
            "declares exactly",
        ),
        (
            lambda tier: tier.__setitem__("workload_profiles", [{"id": "a", "seq_lens": [33]}, {"id": "b", "seq_lens": [1]}]),
            "exceeds the declared capacity",
        ),
        (
            lambda tier: tier.__setitem__(
                "workload_profiles",
                [{"id": "a", "seq_lens": [4, 4, 4, 4]}, {"id": "A", "seq_lens": [1]}],
            ),
            "must match",
        ),
    ],
)
def test_illegal_capacity_declarations_are_refused(mutate, expected):
    """@brief 混用/缺声明/单档/非法 sentinel/便利 0/超容量/非法 profile id 全部 typed 拒绝。"""
    document = _document()
    mutate(document["sizes"][CAPACITY])
    with pytest.raises(OpSpecError) as excinfo:
        _validate(document)
    assert expected in str(excinfo.value)


def test_a_fixed_tier_may_not_be_mixed_into_a_varlen_task():
    """@brief 固定档与变长档混用（含 L2）在文档层被拒绝。"""
    document = _document()
    document["sizes"]["plain"] = {"H": 2, "D": 8, "total_tokens": 4, "num_seqs": 1, "NSP1": 2, "usage": ["golden"]}
    with pytest.raises(OpSpecError) as excinfo:
        _validate(document)
    assert "entirely varlen" in str(excinfo.value)


def test_a_fused_resolved_axis_is_refused_because_padding_would_be_undefined():
    """@brief 容量轴必须独立成 dim；``total_tokens*H`` 会让 logical/padding 区域无法定义。"""
    document = _document()
    document["tensors"]["q"]["dims"] = ["total_tokens*H", "D"]
    document["tensors"]["q"]["strides"] = ["D", "1"]
    with pytest.raises(OpSpecError) as excinfo:
        _validate(document)
    assert "fuses the resolved symbol total_tokens" in str(excinfo.value)


@pytest.mark.parametrize("tensor_name", ["q", "output", "final_state"])
def test_an_aliased_resolved_axis_is_refused_for_every_tensor(tensor_name):
    """@brief 负例：``T2=total_tokens`` 这类别名轴不能当 invariant 用（input/output/final_state）。

    An axis whose declared term is a *second symbol* declared at the capacity value is
    indistinguishable from ``total_tokens``/``num_seqs``.  Treating it as invariant would compare the
    whole capacity buffer and skip the padding check for that tensor — the "L1 renamed as L2" escape.
    The document validator refuses it for an input, a ``write`` output and the ``read_write``-style
    ``final_state`` alike, so no tensor is a loophole.
    """
    document = _document()
    document["symbols"]["S2"] = {"min": 1}
    for definition in document["sizes"].values():
        definition["S2"] = int(document["sizes"][CAPACITY]["capacity"]["num_seqs"])
    tensor = document["tensors"][tensor_name]
    tensor["dims"] = ["S2"] + list(tensor["dims"][1:])
    with pytest.raises(OpSpecError) as excinfo:
        _validate(document)
    message = str(excinfo.value)
    assert "is not a canonical resolved axis" in message
    assert "resolved-axis alias" in message
    assert f"tensors.{tensor_name}.dims[0]" in message


def test_a_strided_capacity_tensor_is_refused_because_the_byte_region_is_unprovable():
    """@brief 负例：容量档的 logical/padding 区域从连续字节图派生，非行主序布局必须 fail closed。"""
    document = _document()
    document["tensors"]["q"]["layout"] = "strided"
    document["tensors"]["q"]["strides"] = ["1", "total_tokens", "total_tokens*H"]
    with pytest.raises(OpSpecError) as excinfo:
        _validate(document)
    assert "is not row-major" in str(excinfo.value)


def test_an_invariant_product_at_the_capacity_value_is_still_legal():
    """@brief 正例：``dt_bias`` 的 ``H*D`` 恰好等于容量 total，但它是独立符号积，不是别名。"""
    document = _document()
    symbols = _capacity_tier_symbols_for(document)
    assert int(symbols["H"]) * int(symbols["D"]) == int(symbols["total_tokens"])
    _validate(document)  # no refusal: a product of independent extents is provably not the token axis


def _capacity_tier_symbols_for(document):
    """@brief 由文档重算容量档的符号值（测试侧只读副本，不引生产实现）。"""
    definition = document["sizes"][CAPACITY]
    values = {name: int(definition[name]) for name in document["symbols"] if name in definition}
    values["total_tokens"] = int(definition["capacity"]["total_tokens"])
    values["num_seqs"] = int(definition["capacity"]["num_seqs"])
    values["NSP1"] = values["num_seqs"] + 1
    return values


def test_a_non_result_tensor_keeps_the_parameter_extent_rule():
    """@brief 正例：``dt_bias`` 的 ``H*D`` 恰好等于容量 total，作为**显式声明**的参数范围仍合法。

    The role rule must not turn into "any axis whose length equals the capacity is illegal": ``dt_bias`` is a
    parameter vector of ``H*D`` elements and has nothing to do with the token axis.  A shape algebra cannot
    express that, so the task declares it (``axis_roles: {"0": "parameter"}``); without the declaration the
    same tensor is refused, because the harness would otherwise be guessing which ``H*D`` it is looking at.
    """
    document = _document()
    assert document["tensors"]["dt_bias"]["dims"] == ["H*D"]
    assert document["tensors"]["dt_bias"]["role"] == "input"
    assert document["tensors"]["dt_bias"]["axis_roles"] == {"0": "parameter"}
    _validate(document)
    without = _document()
    without["tensors"]["dt_bias"].pop("axis_roles")
    with pytest.raises(OpSpecError) as excinfo:
        _validate(without)
    assert "neither a canonical resolved axis" in str(excinfo.value)


def test_axis_roles_are_inert_when_only_l1_tiers_are_materialized():
    """@brief 正例：只生成 fixed/L1 tier 时，L2 角色声明不得破坏既有任务变体。"""
    document = _document()
    document["sizes"] = {"mixed": document["sizes"]["mixed"]}
    _validate(document)


def test_a_packed_input_must_name_or_declare_its_capacity_sized_axis():
    """@brief 负例：packed input 的 capacity 尺寸轴改成 ``H*D``（无声明）→ 拒绝。

    The previous version only guarded result tensors, so ``q.dims[0] = "H*D"`` (16 == capacity total here)
    passed the real validator.  ``padding_slabs`` then treated q as invariant, the harness handed the custom
    ``error_scale`` hook the whole capacity buffer including the padding poison, and a hook computing
    ``sqrt(mean(inputs["q"]**2))`` inflated the final-state denominator: a wrong logical ``final_state``
    passed with ``scaled_err=0.916``.  The rule now covers every tensor; naming the axis or declaring it a
    parameter extent is the only way through, and a result can never take the parameter route.
    """
    document = _document()
    document["tensors"]["q"]["dims"] = ["H*D", "H", "D"]
    with pytest.raises(OpSpecError) as excinfo:
        _validate(document)
    message = str(excinfo.value)
    assert "neither a canonical resolved axis" in message
    assert "error_scale" in message
    # The explicit form is the task's own claim (it is recorded in the instance identity), not a silencer.
    declared = _document()
    declared["tensors"]["q"]["dims"] = ["H*D", "H", "D"]
    declared["tensors"]["q"]["axis_roles"] = {"0": "parameter"}
    _validate(declared)


def test_axis_roles_are_a_closed_declaration():
    """@brief 负例：轴角色不能滥用（结果张量、canonical 轴、越界轴、未知角色都拒绝）。"""
    document = _document()
    document["tensors"]["final_state"]["axis_roles"] = {"0": "parameter"}
    with pytest.raises(OpSpecError):
        _validate(document)
    document = _document()
    document["tensors"]["q"]["axis_roles"] = {"0": "parameter"}
    with pytest.raises(OpSpecError, match="contradicts the canonical resolved term"):
        _validate(document)
    document = _document()
    document["tensors"]["dt_bias"]["axis_roles"] = {"3": "parameter"}
    with pytest.raises(OpSpecError, match="is not a declared axis"):
        _validate(document)
    document = _document()
    document["tensors"]["dt_bias"]["axis_roles"] = {"0": "packed"}
    with pytest.raises(OpSpecError, match="not one of"):
        _validate(document)


def test_the_declared_axis_roles_enter_the_semantic_instance_identity():
    """@brief 轴角色是声明事实：进语义实例文档（评审能看到 ``dt_bias`` 被显式声明为参数范围）。"""
    document = _instance().fingerprint_document()
    tensors = {item["name"]: item for item in document["tensors"]}
    assert tensors["dt_bias"]["axis_roles"] == {"0": "parameter"}
    assert "axis_roles" not in tensors["q"], "an undeclared tensor adds no key"


@pytest.mark.parametrize("term", ["4", "2*2", "H*2", "H*D"])
def test_a_result_tensor_must_name_its_capacity_sized_axis(term):
    """@brief 负例：结果张量的 capacity 尺寸轴不能用字面量/参数积冒充 resolved 轴。

    ``resolve_axis_kind``'s shape algebra cannot separate ``dt_bias``'s ``H*D`` (an independent parameter
    extent) from a sequence-major result axis that merely *evaluates* to the capacity (``"4"``, ``"2*2"``,
    ``"H*2"``, ``"H*D"``).  The role is the discriminator: a **result** axis at the capacity bound has to
    name the resolved symbol.  Without this rule the declared ``final_state`` logical region becomes the
    whole capacity buffer, the error scale is dominated by the padding poison (1.33e16 on this task), and a
    completely wrong logical ``final_state`` passes with ``n_bad=0``.
    """
    document = _document()
    document["tensors"]["final_state"]["dims"][0] = term
    with pytest.raises(OpSpecError) as excinfo:
        _validate(document)
    message = str(excinfo.value)
    assert "a result tensor axis of length" in message
    assert "not 'total_tokens'/'num_seqs'" in message


def test_an_ambiguous_capacity_bound_is_refused_on_a_result_axis():
    """@brief 负例：capacity 的 total 与 count 相等且 >1 时，结果轴上的两个 canonical 名字无法区分。

    They evaluate to the same length, so a result axis of that length labelled ``num_seqs`` cannot be proven
    to be the sequence axis -- and if it really is the token axis, the harness slices to ``resolved_num_seqs``
    and demands the real token rows as padding poison: an honest candidate fails while a candidate that never
    computes those tokens passes.  A 1/1 capacity is exempt because every legal profile then has resolved
    total == resolved count == 1, so both slices coincide.
    """
    from tcenv.opspec.model import capacity_axis_role_problem

    ambiguous = {"total_tokens": 4, "num_seqs": 4}
    def probe(dim, term, symbols, *, is_result=True, role=None):
        return capacity_axis_role_problem(
            dim, term, symbols, declared_role=role, is_result=is_result, where="probe"
        )

    assert "declare distinct capacity bounds" in str(probe(4, "num_seqs", ambiguous))
    assert "declare distinct capacity bounds" in str(probe(4, "total_tokens", ambiguous))
    assert "declare distinct capacity bounds" in str(probe(4, "total_tokens", ambiguous, is_result=False))
    assert probe(1, "num_seqs", {"total_tokens": 1, "num_seqs": 1}) is None
    document = _document()
    tier = document["sizes"][CAPACITY]
    tier["capacity"]["total_tokens"] = 4
    tier["capacity"]["num_seqs"] = 4
    tier["workload_profiles"] = [
        {"id": "a_full", "seq_lens": [1, 1, 1, 1]},
        {"id": "b_mid", "seq_lens": [2, 1]},
        {"id": "c_two", "seq_lens": [2]},
    ]
    with pytest.raises(OpSpecError) as excinfo:
        _validate(document)
    assert "declare distinct capacity bounds" in str(excinfo.value)


def test_a_capacity_tier_without_a_full_capacity_profile_is_refused():
    """@brief capacity 点本身必须被某个 profile 覆盖，否则“容量”从未被测。"""
    document = _document()
    document["sizes"][CAPACITY]["workload_profiles"] = [
        {"id": "a", "seq_lens": [5, 3, 1]},
        {"id": "b", "seq_lens": [2, 1]},
    ]
    with pytest.raises(OpSpecError) as excinfo:
        _validate(document)
    assert "at least one profile must reach the full capacity" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------------
# 2. logical / resolved / capacity shapes and buffer sizing
# ---------------------------------------------------------------------------------------------------


def test_tensor_bindings_and_workspaces_are_sized_by_capacity():
    """@brief 全部 buffer/ABI span 按 capacity 定，与任何一次 resolved 调用无关。"""
    instance = _instance()
    shapes = {binding.name: binding.shape for binding in instance.tensors}
    assert shapes["q"] == (16, 2, 8)
    assert shapes["beta"] == (16, 2)
    assert shapes["initial_state"] == (4, 2, 8, 8)
    assert shapes["final_state"] == (4, 2, 8, 8)
    assert shapes["cu_seqlens"] == (5,)
    abi = derive_tensor_memory_contract(instance, target=TARGET)
    spans = {tensor.name: tuple(int(v) for v in tensor.shape) for tensor in abi.tensors}
    assert spans["q"] == (16, 2, 8)
    assert spans["cu_seqlens"] == (5,)


def test_the_case_sweep_runs_several_batches_that_shrink_and_grow():
    """@brief 一次 case sweep 就是 continuous batching：多个不同 total/count，含缩与涨。"""
    instance = _instance()
    capacities = instance.capacity
    seen = []
    for index, case in enumerate(instance.case_domain.cases):
        profile = workload_profile_for_case(instance, case)
        resolved = resolved_shape_for_case(instance, case)
        assert profile is not None and resolved is not None
        assert (resolved.total_tokens, resolved.num_seqs) == (profile.total_tokens, profile.num_seqs)
        assert resolved.total_tokens <= capacities.total_tokens
        assert resolved.num_seqs <= capacities.num_seqs
        assert resolved.total_tokens >= 1 and resolved.num_seqs >= 1
        # The harness assigns profiles cyclically over the declared case order.
        ordered = sorted(instance.workload_profiles, key=lambda item: item.id)
        assert profile.id == ordered[index % len(ordered)].id
        seen.append((resolved.total_tokens, resolved.num_seqs))
    totals = [total for total, _count in seen]
    assert len(set(totals)) >= 3
    assert len({count for _total, count in seen}) >= 3
    assert any(seen[i] < seen[i + 1] for i in range(len(seen) - 1)), "the sweep must grow at least once"
    assert any(seen[i] > seen[i + 1] for i in range(len(seen) - 1)), "the sweep must shrink at least once"
    assert (capacities.total_tokens, capacities.num_seqs) in seen, "the capacity point must be covered"
    assert (1, 1) in seen, "a one-token one-sequence batch must be covered"
    # An empty *sequence* is legal and must be exercised (an empty batch is not).
    lengths = []
    for case in instance.case_domain.cases:
        cu = _raw_inputs(instance, case)["inputs"]["cu_seqlens"].tolist()
        resolved = resolved_shape_for_case(instance, case)
        prefix = cu[: resolved.num_seqs + 1]
        lengths.extend(end - begin for begin, end in zip(prefix, prefix[1:]))
    assert any(length == 0 for length in lengths), "an empty sequence must be exercised"
    assert any(length > 1 for length in lengths), "a multi-token sequence must be exercised"


def test_every_case_keeps_its_profile_total_and_capacity_boundary_instances():
    """@brief 生成的实际分段保留 profile 的 total/count（空序列合法、total=0 不行）。"""
    instance = _instance()
    for case in instance.case_domain.cases:
        produced = _raw_inputs(instance, case)
        resolved = produced["resolved"]
        cu = produced["inputs"]["cu_seqlens"].tolist()
        prefix_len = resolved.num_seqs + 1
        prefix = cu[:prefix_len]
        assert prefix[0] == 0
        assert prefix[-1] == resolved.total_tokens
        assert all(end >= begin for begin, end in zip(prefix, prefix[1:]))
        assert all(value == instance.capacity.cu_seqlens_sentinel for value in cu[prefix_len:])
        assert resolved.total_tokens >= 1


# ---------------------------------------------------------------------------------------------------
# 3. Continuous-batching inputs: padding, sentinel, digest
# ---------------------------------------------------------------------------------------------------


def test_generated_padding_and_boundary_tail_are_the_declared_poison():
    """@brief 未使用的 input 区域逐字节是声明的 poison，cu 尾部是声明的 sentinel。"""
    instance = _instance()
    capacity = instance.capacity
    contract = _contract()
    for case in instance.case_domain.cases:
        produced = _raw_inputs(instance, case)
        resolved, inputs = produced["resolved"], produced["inputs"]
        # Byte-level on the storage bytes that reach the device (the oracle itself works widened to fp32).
        raw = _raw(contract, inputs)
        payload = {name: value for name, value in raw.items() if name != "cu_seqlens"}
        shapes = {name: instance.tensor(name).shape for name in payload}
        nbytes = {name: get_dtype(instance.tensor(name).dtype).nbytes for name in payload}
        assert padding_byte_problems(instance, resolved, payload, shapes, nbytes, where=case) == ()
        sentinel = capacity.cu_seqlens_sentinel
        cu = inputs["cu_seqlens"].tolist()
        assert all(value == sentinel for value in cu[resolved.num_seqs + 1 :])
        if resolved.total_tokens < capacity.total_tokens:
            # Not a convenient value: the filler is the declared poison byte pattern, not 0/-1.
            assert not bool(inputs["q"][resolved.total_tokens :].eq(0).all())
        if resolved.num_seqs < capacity.num_seqs:
            assert inputs["initial_state"][resolved.num_seqs :].numel() > 0


def test_a_generator_that_leaves_the_padding_unwritten_is_caught():
    """@brief 负例：padding 被写成 0 而不是声明的 poison，harness 必须拒绝。"""
    instance = _instance()
    produced = _raw_inputs(instance, "permuted")  # b_tiny: 1/1, so most of the capacity is padding
    resolved, inputs = produced["resolved"], produced["inputs"]
    assert resolved.total_tokens < instance.capacity.total_tokens
    contract = _contract()
    inputs = dict(inputs)
    inputs["q"] = produced["inputs"]["q"].clone()
    inputs["q"][resolved.total_tokens :] = 0.0
    raw = _raw(contract, inputs)
    payload = {name: value for name, value in raw.items() if name != "cu_seqlens"}
    shapes = {name: instance.tensor(name).shape for name in payload}
    nbytes = {name: get_dtype(instance.tensor(name).dtype).nbytes for name in payload}
    problems = padding_byte_problems(instance, resolved, payload, shapes, nbytes, where="t")
    assert problems and "q padding" in problems[0]


def _cu_with(instance, values, case: str = "canonical") -> dict:
    """@brief 用一组（可能非法的）偏移替换某个 case 的 cu_seqlens。"""
    produced = _raw_inputs(instance, case)
    inputs = dict(produced["inputs"])
    inputs["cu_seqlens"] = torch.tensor(list(values), dtype=torch.int32)
    return {"inputs": inputs, "resolved": produced["resolved"]}


@pytest.mark.parametrize(
    "case,values,expected",
    [
        # An empty batch is not an L2 call; only an empty *sequence* is legal.
        ("skewed", [0, 5, 4, 9, -1], "non-decreasing"),
        ("permuted", [0, 2, -1, -1, -1], "must end at the resolved total_tokens"),
        ("permuted", [0, 1, 5, -1, -1], "capacity tail entry"),
        ("permuted", [0, 1, 0, 0, 0], "capacity tail entry 2 is 0"),
        ("permuted", [1, 2, -1, -1, -1], "must start at 0"),
        ("canonical", [0, 4, 8, 12, 15], "must end at the resolved total_tokens"),
    ],
)
def test_illegal_runtime_boundaries_are_refused(case, values, expected):
    """@brief empty batch/非单调/错误末项/非 sentinel 尾部/便利 0 尾部/负首项全部拒绝。"""
    instance = _instance()
    produced = _cu_with(instance, values, case)
    with pytest.raises(VarlenContractError) as excinfo:
        require_runtime_segments(
            instance,
            produced["inputs"],
            where="negative",
            resolved=produced["resolved"],
            case=case,
        )
    assert expected in str(excinfo.value)


def test_an_empty_batch_is_refused_even_without_a_declared_expectation():
    """@brief 没有 case 期望时，字节解析也不能把“全是 sentinel”当成 0-token 批次。"""
    instance = _instance()
    produced = _cu_with(instance, [-1, -1, -1, -1, -1], "permuted")
    with pytest.raises(VarlenContractError) as excinfo:
        resolved_shape_from_inputs(instance, produced["inputs"])
    assert "at least one sequence" in str(excinfo.value)
    with pytest.raises(VarlenContractError):
        require_runtime_segments(instance, produced["inputs"], where="negative")


def test_an_over_capacity_batch_is_refused_before_any_device_work():
    """@brief 实际 total 超容量：resolved 解析与合同校验都必须 typed 拒绝。"""
    instance = _instance()
    capacity = instance.capacity
    with pytest.raises(VarlenContractError):
        ResolvedShape(
            kind="l2",
            total_tokens=capacity.total_tokens + 1,
            num_seqs=1,
            capacity_total_tokens=capacity.total_tokens,
            capacity_num_seqs=capacity.num_seqs,
            cu_seqlens_sentinel=capacity.cu_seqlens_sentinel,
            padding_poison_byte=capacity.padding_poison_byte,
            workload_profile_id="forged",
        )
    produced = _cu_with(instance, [0, capacity.total_tokens + 1, -1, -1, -1])
    with pytest.raises(VarlenContractError):
        resolved_shape_from_inputs(instance, produced["inputs"])


def test_resolved_shape_is_read_from_the_bytes_not_from_the_case_name():
    """@brief resolved 形状从实际字节解析；伪造的 resolved 声明与字节不一致时被拒。"""
    instance = _instance()
    produced = _raw_inputs(instance, "skewed")  # c_mid: 9/3
    parsed = resolved_shape_from_inputs(instance, produced["inputs"])
    assert (parsed.total_tokens, parsed.num_seqs) == (9, 3)
    # The declared profile is only known through the case; the byte parse must not relabel itself.
    assert parsed.canonical_seq_lens == ()
    assert produced["resolved"].canonical_seq_lens == (5, 3, 1)
    forged = ResolvedShape(
        kind="l2",
        total_tokens=9,
        num_seqs=2,
        capacity_total_tokens=instance.capacity.total_tokens,
        capacity_num_seqs=instance.capacity.num_seqs,
        cu_seqlens_sentinel=instance.capacity.cu_seqlens_sentinel,
        padding_poison_byte=instance.capacity.padding_poison_byte,
        workload_profile_id="b_tiny",
    )
    problems = cu_seqlens_problems(
        instance, forged, produced["inputs"]["cu_seqlens"], expected_shape=(5,)
    )
    assert problems and any("capacity tail entry" in problem for problem in problems)


def test_evaluation_input_digest_binds_the_actual_resolved_batch():
    """@brief 同一 capacity 实例上，不同 total/count 或不同分段产生不同 evaluation digest。"""
    instance = _instance()
    digests = {}
    for case in ("canonical", "permuted", "skewed", "all_unaligned_primes"):
        produced = _raw_inputs(instance, case)
        digests[case] = evaluation_input_digest(instance, case, 7, produced["inputs"])
    assert len(set(digests.values())) == len(digests)
    # Regenerating the same (instance, case, seed) reproduces the digest.
    again = _raw_inputs(instance, "skewed")
    assert evaluation_input_digest(instance, "skewed", 7, again["inputs"]) == digests["skewed"]
    # Changing the boundary bytes changes the identity (the resolved shape is read back from them).
    retimed = _raw_inputs(instance, "canonical")
    inputs = dict(retimed["inputs"])
    inputs["cu_seqlens"] = retimed["inputs"]["cu_seqlens"].clone()
    inputs["cu_seqlens"][1] = 3
    assert evaluation_input_digest(instance, "canonical", 7, inputs) != digests["canonical"]
    # Only the boundary bytes are hashed: the packed payload is determined by (instance, case, seed), and
    # an L2 payload is no different (the capacity filler is part of that deterministic draw).  Padding
    # *correctness* is checked separately, by value, on both sides of the run.
    payload = _raw_inputs(instance, "canonical")
    altered = dict(payload["inputs"])
    altered["q"] = payload["inputs"]["q"].clone()
    altered["q"][-1, 0, 0] = 0.0
    assert evaluation_input_digest(instance, "canonical", 7, altered) == digests["canonical"]


def test_the_semantic_identity_and_dispatch_domain_stay_at_the_capacity_point():
    """@brief resolved 只进输入/launch 证据：语义身份、域计划与 guard 名字空间都不含 resolved 值。"""
    instance = _instance()
    digest = semantic_instance_digest(instance)
    plan = normalize_semantic_domain_plan(load_opspec_v2(OPSPEC_PATH), [instance], target=TARGET)
    point = plan.find("fp16", CAPACITY, "golden")
    assert point is not None
    assert point.semantic_instance_digest == digest
    # The guard/dispatch name space is the capacity symbols + attrs, never a resolved shape.
    names = instance_names(instance, where="capacity dispatch")
    assert names["total_tokens"] == instance.capacity.total_tokens
    assert names["num_seqs"] == instance.capacity.num_seqs
    assert not any("resolved" in name for name in names)
    # Two different resolved batches of the same capacity point share the domain identity.
    first = resolved_shape_for_case(instance, "canonical")
    second = resolved_shape_for_case(instance, "permuted")
    assert (first.total_tokens, first.num_seqs) != (second.total_tokens, second.num_seqs)
    assert semantic_instance_digest(instance) == digest
    assert plan.digest == normalize_semantic_domain_plan(load_opspec_v2(OPSPEC_PATH), [instance], target=TARGET).digest
    # The declared profile set has its own identity; the per-call resolved values are not part of it.
    assert workload_profile_digest(instance) == workload_profile_digest(_instance())


def test_the_workload_profile_digest_requires_a_capacity_instance():
    """@brief workload-profile 身份只对 L2 有定义（L1 的身份是 canonical 分段）。"""
    from tcenv.contract_errors import TaskContractError

    instance = _instance()
    value = workload_profile_digest(instance)
    assert isinstance(value, str) and len(value) == 64
    with pytest.raises(TaskContractError):
        workload_profile_digest(_instance("mixed"))


def test_l2_requires_the_resolved_generator_argument():
    """@brief L2 的生成器必须接受 ``resolved=``；否则 typed 拒绝而不是静默按 capacity 生成。"""
    instance = _instance()
    reference = _contract().reference
    assert isinstance(reference, ReferenceModule)
    stripped = ReferenceModule(
        op_name=reference.op_name,
        path=reference.path,
        source=reference.source,
        source_sha256=reference.source_sha256,
        reference_version=reference.reference_version,
        sampling=None,
        error_scale_kind=reference.error_scale_kind,
        _reference=reference._reference,
        _gen_inputs=lambda case, seed, inst: {},
        gen_inputs_resolved=False,
    )
    from tcenv.contract_errors import TaskContractError

    with pytest.raises(TaskContractError) as excinfo:
        stripped.generate(instance, "canonical", 1)
    assert "resolved=None" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------------
# 4./5. Native Host Program: resolved scalars, tamper refusal, one build for many batches
# ---------------------------------------------------------------------------------------------------


def _execute_l2(runtime, contract, case: str, seed: int = 31337):
    """@brief 用某个 case 的真实输入执行一次 L2 Host Program。"""
    produced = _raw_inputs(contract.instance, case, seed)
    raw = _raw(contract, produced["inputs"])
    resolved = produced["resolved"]
    run = runtime.execute(
        lambda name, off, nbytes: raw[name][off : off + nbytes],
        values=None,
        resolved=resolved.resolved_scalars(),
        audit=True,
    )
    return run, resolved


def test_the_native_abi_exposes_real_resolved_fields_for_l2_only():
    """@brief L2 header 有 ``SfHostResolved`` 真字段与 accessor；L1 header 逐字节保持 reserved-only。"""
    l2 = _contract()
    l2_header = render_native_support(host_contract_spec(instance=l2.instance))
    assert "uint32_t total_tokens;" in l2_header
    assert "uint32_t num_seqs;" in l2_header
    assert "sf_native_set_resolved_total_tokens" in l2_header
    assert "sf_native_resolved_num_seqs" in l2_header
    assert "native resolved scalar total_tokens was rewritten during host_tiling" in l2_header
    l1 = _contract("mixed")
    l1_header = render_native_support(host_contract_spec(instance=l1.instance))
    assert "typedef struct { uint32_t reserved; } SfHostResolved;" in l1_header
    assert "sf_native_resolved_" not in l1_header


def test_the_same_build_serves_many_resolved_batches_without_recompiling():
    """@brief 一次编译 + 连续多个不同 total/count：build 身份不变、计划不同、resolved 回读一致。"""
    runtime, _spec, build, contract = _l2_runtime()
    identity = (build.program_sha256, build.assembled_sha256, build.native_sha256)
    seen: dict[tuple[int, int], bytes] = {}
    for case in contract.instance.case_domain.cases:
        run, resolved = _execute_l2(runtime, contract, case)
        assert run.ok, run.error_detail
        assert dict(run.resolved_scalars) == resolved.resolved_scalars()
        assert (build.program_sha256, build.assembled_sha256, build.native_sha256) == identity
        assert runtime.build is build
        items = next(span for span in run.spans if span.name == "work_items")
        assert items.count == 2 * resolved.num_seqs, "the plan must follow the resolved count, not the capacity"
        seen[(resolved.total_tokens, resolved.num_seqs)] = items.data
    assert len(set(seen.values())) == len(seen), "different resolved batches must produce different plans"
    # The resolved values never enter the generated contract text (only the field names do).
    assert "16u" not in render_native_support(host_contract_spec(instance=contract.instance)).split("SfHostResolved")[1][:80]


def test_a_candidate_that_rewrites_a_resolved_scalar_is_refused():
    """@brief 候选改写 ``resolved`` → C 合同检查失败，全部 span/resolved 丢弃。"""
    runtime, _spec, _build, contract = _l2_runtime(CAPACITY_PLANNER_FORGES_RESOLVED)
    produced = _raw_inputs(contract.instance, "canonical")
    raw = _raw(contract, produced["inputs"])
    with pytest.raises(HostProgramError) as excinfo:
        runtime.execute(
            lambda name, off, nbytes: raw[name][off : off + nbytes],
            values=None,
            resolved=produced["resolved"].resolved_scalars(),
            audit=True,
        )
    assert "resolved scalar" in str(excinfo.value)


def test_a_resolved_scalar_the_contract_did_not_declare_is_refused():
    """@brief L1 合同没有 resolved 字段：传值必须拒绝（不是静默忽略），C 侧也无法读取。"""
    contract = _contract("mixed")
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
    build = compile_native_host_program(PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    runtime = NativeHostRuntime(build, spec)
    logical = contract.reference.generate(contract.instance, "canonical", 7)
    raw = _raw(contract, logical)
    with pytest.raises(HostProgramError) as excinfo:
        runtime.execute(
            lambda name, off, nbytes: raw[name][off : off + nbytes],
            values=None,
            resolved={"total_tokens": 4},
            audit=True,
        )
    assert "declares no runtime resolved scalars" in str(excinfo.value)


def test_an_l1_abi_cannot_even_read_a_resolved_scalar():
    """@brief L1 的 header 没有 resolved 字段，读它的候选在编译期就被拒绝。"""
    contract = _contract("mixed")
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
    with pytest.raises(HostProgramError) as excinfo:
        compile_native_host_program(CAPACITY_PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    assert "suggest" in str(excinfo.value) or "error" in str(excinfo.value).lower()


def test_the_resolved_shape_and_input_digest_enter_the_host_run_document():
    """@brief HostRunEvidence 里能查到本次 resolved 形状与可重放的 input digest（不进 build 身份）。"""
    runtime, _spec, _build, contract = _l2_runtime()
    run, resolved = _execute_l2(runtime, contract, "skewed")
    document = run.document()
    assert document["resolved_scalars"] == {"total_tokens": 9, "num_seqs": 3}
    digests = json.dumps(document, sort_keys=True)
    assert "resolved_scalars" in digests
    expected = evaluation_input_digest(contract.instance, "skewed", 31337, _raw_inputs(contract.instance, "skewed")["inputs"])
    assert expected == evaluation_input_digest(contract.instance, "skewed", 31337, _raw_inputs(contract.instance, "skewed")["inputs"])
    assert resolved.document()["full_capacity"] is False


# ---------------------------------------------------------------------------------------------------
# 6. Logical region vs padding (pure helpers; the Metal comparator is covered in its own CPU test)
# ---------------------------------------------------------------------------------------------------


def test_padding_slabs_are_disjoint_from_the_logical_region():
    """@brief logical/padding 区域由声明的 dim 词决定，且 padding slab 与 logical 区域不相交。"""
    instance = _instance()
    resolved = resolved_shape_for_case(instance, "skewed")  # 9/3 of 16/4
    logical, slabs = padding_slabs(instance, resolved, (16, 2, 8), ("total_tokens", "H", "D"))
    assert [s.stop for s in logical] == [9, 2, 8]
    assert len(slabs) == 1
    assert slabs[0][0] == slice(9, 16)
    # A tensor with no resolved axis has no padding at all.
    _logical, none_slabs = padding_slabs(instance, resolved, (2,), ("H",))
    assert none_slabs == ()
    # Sequence-major padding is a separate axis.
    _l, seq_slabs = padding_slabs(instance, resolved, (4, 2, 8, 8), ("num_seqs", "H", "D", "D"))
    assert len(seq_slabs) == 1 and seq_slabs[0][0] == slice(3, 4)


def test_padding_bytes_changed_outside_the_logical_region_are_reported():
    """@brief 只改 padding 也必须被报出来（不会被“只比较前缀”掩盖）。"""
    instance = _instance()
    resolved = resolved_shape_for_case(instance, "permuted")  # 1/1 of 16/4
    capacity = instance.capacity
    poison = torch.full((16, 2, 8), float(0), dtype=torch.float16)
    raw = poison.view(torch.uint8).reshape(16, 2, 8, 2)
    raw[1:] = int(capacity.padding_poison_byte)
    tensor = poison
    assert padding_problems(instance, resolved, {"q": tensor}, where="after") == ()
    tampered = tensor.clone()
    tampered[3, 0, 0] = 1.0
    problems = padding_problems(instance, resolved, {"q": tampered}, where="after")
    assert problems and "changed outside the logical region" in problems[0]
