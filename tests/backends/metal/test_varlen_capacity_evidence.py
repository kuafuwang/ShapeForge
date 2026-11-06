"""V5 L2 release blockers: logical-only scales, alias refusal, profile-bound digests, call evidence.

This file pins the five things a reviewer found missing in the first V5 slice.  Each section is a
*negative* that used to pass and must now fail, so the tests are only meaningful if they fail on the
pre-fix code:

1. the L2 error scale is computed on the **logical region only** — ``output`` entirely correct with
   ``final_state`` entirely wrong is a failure, and changing the padding poison cannot change the
   logical verdict or the scale (a poison-inflated denominator hid exactly that mistake);
2. a resolved-axis *alias* (a second symbol declared at the capacity value and used as a row axis) is
   refused at OpSpec validation and at prepare/compare time — there is no "compare the whole capacity
   buffer" fallback for inputs, outputs or ``final_state``;
3. ``evaluation_input_digest`` binds the *declared* ``workload_profile_id`` and its canonical
   ``seq_lens``, verifies that the actual ``cu_seqlens`` bytes derive the profile's total/count, and
   states its scope as boundary+identity rather than claiming a byte-complete hash;
4. ``HostRunEvidence`` keeps the same field set on success and failure, carries each ordered call's
   resolved shape and input digest, and consecutive calls are not collapsed to the last record or
   dropped by static dispatch dedup;
5. the timed benchmark runs the declared continuous-batching mixture in declared order (not always
   ``cases[0]`` at full capacity), the mixture is machine-checked, a fallback provider cannot be mixed
   into the variant's mixture, and resolved values never become a second dispatch domain point.

Nothing here opens a Metal device, spends money or touches the NPU device.
"""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from tcenv.backends.metal import track_adapter, v2_runtime
from tcenv.backends.metal.host_program import MetalHostProgram
from tcenv.backends.metal.v2_runtime import (
    V2Benchmark,
    V2Case,
    _check_v2_case,
    benchmark_datasets,
    benchmark_mixture,
    benchmark_mixture_problems,
    prepare_v2_case,
)
from tcenv.contract_errors import TaskContractError, VarlenContractError
from tcenv.hostprogram.results import HostRun
from tcenv.identity import load_strict_json
from tcenv.opspec.model import OpSpecError
from tcenv.opspec.v2 import opspec_v2_from_document
from tcenv.portfolio import domain as domain_module
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest, ExactDispatchResolver
from tcenv.portfolio.domain import evaluation_input_digest, normalize_semantic_domain_plan
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.taskcontract.reference import broadcasts_scale
from tcenv.taskcontract.runtime_shape import (
    capacity_axis_problems,
    logical_region,
    padding_slabs,
    require_capacity_axes,
    resolved_shape_for_case,
    workload_profile_for_case,
)
from tcenv.dtypes import get_dtype
from tcenv.target import load_target
from tcenv.tensors import from_raw, to_raw
from tcenv.track.abi import derive_tensor_memory_contract
from tests.support.shape_portfolio import (
    evidence_all_valid,
    parent_entries,
    portfolio_document,
)

KDA_DIR = REPO_ROOT / "tasks" / "varlen_kda"
OPSPEC_PATH = KDA_DIR / "opspec.json"
TARGET = load_target("apple-m4-max")


def _contract(size: str = "capacity"):
    return load_task_contract(KDA_DIR, dtype="fp16", size=size)


def _instance(size: str = "capacity"):
    return _contract(size).instance


def _abi(instance=None):
    return derive_tensor_memory_contract(instance or _instance(), target=TARGET)


def _candidate_bytes(
    contract,
    data: V2Case,
    *,
    final_state_fill: float | None = None,
    poison: int | None = None,
    logical_offset: dict[str, float] | None = None,
) -> dict[str, bytes]:
    """@brief 用 reference 终值构造一份候选输出（容量尺寸，logical 之外是声明 poison）。"""
    instance = contract.instance
    resolved = data.resolved_shape
    capacity = bool(instance.is_capacity and resolved is not None)
    expected = 0
    if capacity:
        expected = int(instance.capacity.padding_poison_byte if poison is None else poison)
    out: dict[str, bytes] = {}
    for name in instance.reference_outputs:
        binding = instance.tensor(name)
        shape = tuple(int(dim) for dim in binding.shape)
        nbytes = get_dtype(binding.dtype).nbytes
        value = data.ref[name].to(torch.float32).reshape(shape).clone()
        logical = (
            logical_region(instance, resolved, name)
            if capacity
            else tuple(slice(0, int(dim)) for dim in shape)
        )
        if name in (logical_offset or {}):
            value[logical] = value[logical] + float(logical_offset[name])
        if final_state_fill is not None and name == "final_state":
            value[logical] = float(final_state_fill)
        raw = bytearray(to_raw(value, binding.dtype))
        array = torch.frombuffer(raw, dtype=torch.uint8).reshape(shape + (nbytes,))
        if capacity:
            _logical, slabs = padding_slabs(instance, resolved, shape, tuple(binding.dim_terms))
            for slab in slabs:
                array[tuple(slab) + (slice(0, nbytes),)] = expected
        out[name] = bytes(raw)
    return out


# ===================================================================================================
# 1. Logical-region-only error scale
# ===================================================================================================


def test_output_correct_but_final_state_entirely_wrong_fails():
    """@brief 负例（blocker 1）：``output`` 全对、``final_state`` 全错必须是失败。

    On the pre-fix code the default RMS scale for ``final_state`` was averaged over the whole capacity
    buffer, whose unused rows hold the padding poison.  That inflated the denominator by ~1e16 and let a
    completely wrong logical ``final_state`` pass with a tiny scaled error.
    """
    contract = _contract()
    abi = _abi()
    data = prepare_v2_case(contract, "permuted", 907)  # b_tiny: 1 token / 1 sequence of 16/4
    assert data.resolved_shape is not None and not data.resolved_shape.is_full_capacity
    # The scale of the capacity buffer is dominated by the poison rows if anyone averages over them.
    assert float(data.scale["final_state"]) < 10.0, "the L2 scale must come from the logical region only"

    honest = _check_v2_case(contract, abi, data, _candidate_bytes(contract, data), "permuted")
    assert honest.ok, honest.message

    wrong = _check_v2_case(
        contract, abi, data, _candidate_bytes(contract, data, final_state_fill=0.0), "permuted"
    )
    assert not wrong.ok
    assert wrong.scaled_err > 1.0
    assert "elements out of tolerance" in wrong.message
    assert any(item["output"] == "final_state" for item in [wrong.worst or {}])


def test_the_logical_verdict_does_not_depend_on_the_padding_poison():
    """@brief 负例（blocker 1）：改 padding poison 只应改 padding 判决，不应当改 logical 误差。

    The honest candidate's padding is the declared poison; the tampered one holds a different byte
    pattern.  ``scaled_err``/``n_bad`` are logical statistics and must be identical, while ``ok`` is
    allowed to differ *only* because of the padding bytes — a verdict that moves with the poison value
    is a verdict that is partly measuring the poison.
    """
    contract = _contract()
    abi = _abi()
    data = prepare_v2_case(contract, "skewed", 4242)  # 9/3 of 16/4
    honest = _check_v2_case(contract, abi, data, _candidate_bytes(contract, data), "skewed")
    assert honest.ok, honest.message

    # A different poison byte outside the logical region: the logical numbers must not move.
    other_poison = 0x33
    assert other_poison != int(contract.instance.capacity.padding_poison_byte)
    tampered = _check_v2_case(
        contract, abi, data, _candidate_bytes(contract, data, poison=other_poison), "skewed"
    )
    assert not tampered.ok
    assert "padding byte(s) outside the logical region" in tampered.message
    assert tampered.scaled_err == honest.scaled_err
    assert tampered.n_bad == honest.n_bad
    assert tampered.n_elems == honest.n_elems

    # And a *different declared poison* produces the same logical verdict: re-declare the capacity
    # poison, have the generator follow the declaration, and compare the two full evaluations.
    other = _contract_with_poison(other_poison)
    other_data = prepare_v2_case(other, "skewed", 4242)
    moved = _check_v2_case(other, _abi(other.instance), other_data, _candidate_bytes(other, other_data), "skewed")
    assert moved.ok, moved.message
    assert moved.scaled_err == honest.scaled_err
    assert moved.n_bad == honest.n_bad


def _contract_with_poison(poison: int):
    """@brief 同一个 L2 题、只把声明的 padding poison 换成 ``poison`` 的测试合同。

    The task generator writes its own declaration (the OpSpec constant), so a test that wants to vary
    the *declared* poison has to wrap ``gen_inputs`` and re-poison the capacity tensors after the real
    generator ran.  That keeps the test honest: the declaration and the bytes agree, which is exactly
    what the harness checks.
    """
    base = _contract()
    instance = dataclasses.replace(
        base.instance,
        capacity=dataclasses.replace(base.instance.capacity, padding_poison_byte=int(poison)),
    )
    original = base.reference._gen_inputs

    def gen(case, seed, inst, resolved=None):
        produced = original(case, seed, inst, resolved=resolved)
        if resolved is None:
            return produced
        patched = {}
        for name, value in produced.items():
            if name == "cu_seqlens":
                patched[name] = value
                continue
            binding = inst.tensor(name)
            shape = tuple(int(dim) for dim in binding.shape)
            nbytes = get_dtype(binding.dtype).nbytes
            raw = bytearray(to_raw(value.reshape(shape), binding.dtype))
            array = torch.frombuffer(raw, dtype=torch.uint8).reshape(shape + (nbytes,))
            _logical, slabs = padding_slabs(inst, resolved, shape, tuple(binding.dim_terms))
            for slab in slabs:
                array[tuple(slab) + (slice(0, nbytes),)] = int(inst.capacity.padding_poison_byte)
            patched[name] = from_raw(bytes(raw), binding.dtype, shape)
        return patched

    reference = dataclasses.replace(base.reference, _gen_inputs=gen)
    return dataclasses.replace(base, instance=instance, reference=reference)


def _contract_with_error_scale(seen: list[tuple[str, tuple[int, ...]]], *, full_capacity: bool = False):
    """@brief 一个带自定义 ``error_scale`` 钩子的 L2 合同（钩子记录它看到的形状）。"""
    base = _contract()
    instance = base.instance

    def hook(inputs, reference_outputs, inst, tolerances):  # noqa: ANN001 - reference hook signature
        for name in sorted(reference_outputs):
            seen.append((f"out:{name}", tuple(int(dim) for dim in reference_outputs[name].shape)))
        for name in sorted(inputs):
            seen.append((f"in:{name}", tuple(int(dim) for dim in inputs[name].shape)))
        resolved = resolved_shape_for_case(inst, "permuted")
        target = (int(resolved.total_tokens),) if full_capacity is False else tuple(
            int(dim) for dim in inst.tensor("output").shape
        )
        assert target == tuple(int(dim) for dim in reference_outputs["output"].shape[:1])
        # A per-row scale (broadcast over the trailing axes) is the case the old ``scale[logical]``
        # slicing got wrong for L2: a 1-D scale cannot be indexed with a 3-D logical slice.
        rows = tuple(int(dim) for dim in reference_outputs["output"].shape)[0]
        return {
            "output": torch.linspace(0.5, 1.5, rows, dtype=torch.float64),
            "final_state": torch.tensor(0.25, dtype=torch.float64),
        }

    reference = dataclasses.replace(base.reference, error_scale_kind="custom", _error_scale=hook)
    return dataclasses.replace(base, reference=reference)


def test_a_custom_error_scale_is_handed_the_logical_region_only():
    """@brief 负例（blocker 1）：自定义尺度钩子只能看到 logical 区域，且按行尺度能正确广播。

    The hook is written for L2: it asserts it sees ``total_tokens = 1`` rows, not the capacity's 16.
    The pre-fix code handed it the full capacity tensors (padding poison included) and then sliced a
    per-row scale with a 3-D index, which either raised or re-introduced the poison denominator.
    """
    seen: list[tuple[str, tuple[int, ...]]] = []
    contract = _contract_with_error_scale(seen)
    data = prepare_v2_case(contract, "permuted", 907)
    resolved = data.resolved_shape
    assert resolved is not None
    shapes = dict(seen)
    assert shapes["out:output"] == (int(resolved.total_tokens), 2, 8)
    assert shapes["in:q"] == (int(resolved.total_tokens), 2, 8)
    assert shapes["out:final_state"] == (int(resolved.num_seqs), 2, 8, 8)
    assert shapes["in:cu_seqlens"] == (5,), "the boundary tensor keeps its capacity length"
    assert data.scale["output"].shape == (int(resolved.total_tokens),)
    assert bool(broadcasts_scale(tuple(data.scale["output"].shape), (int(resolved.total_tokens), 2, 8)))

    abi = _abi()
    honest = _check_v2_case(contract, abi, data, _candidate_bytes(contract, data), "permuted")
    assert honest.ok, honest.message
    wrong = _check_v2_case(
        contract, abi, data, _candidate_bytes(contract, data, final_state_fill=-3.0), "permuted"
    )
    assert not wrong.ok, "a wrong logical final_state must fail under a per-row custom scale too"


def test_a_custom_error_scale_that_covers_the_capacity_buffer_is_refused():
    """@brief 负例（blocker 1）：返回整缓冲区形状的尺度不能通过（不能把 poison 塞进分母）。"""
    base = _contract()

    def hook(inputs, reference_outputs, inst, tolerances):  # noqa: ANN001 - reference hook signature
        shape = tuple(int(dim) for dim in inst.tensor("final_state").shape)
        return {"final_state": torch.ones(shape, dtype=torch.float64), "output": torch.tensor(1.0)}

    contract = dataclasses.replace(
        base,
        reference=dataclasses.replace(base.reference, error_scale_kind="custom", _error_scale=hook),
    )
    with pytest.raises(TaskContractError) as excinfo:
        prepare_v2_case(contract, "permuted", 1)
    assert "does not broadcast to the logical result shape" in str(excinfo.value)


# ===================================================================================================
# 2. Resolved-axis aliases
# ===================================================================================================

ALIAS_CASES = ("q", "output", "final_state")


def _aliased_instance(instance, name: str):
    """@brief 把 ``name`` 的首轴换成“容量值相同的第二符号”别名（绕过文档校验的手工实例）。"""
    symbol = "S2"
    value = int(instance.symbols["num_seqs"])
    symbols = {**dict(instance.symbols), symbol: value}
    tensors = []
    for binding in instance.tensors:
        if binding.name == name:
            terms = (symbol,) + tuple(binding.dim_terms[1:])
            binding = dataclasses.replace(binding, dim_terms=terms)
        tensors.append(binding)
    return dataclasses.replace(instance, symbols=symbols, tensors=tuple(tensors))


@pytest.mark.parametrize("name", ALIAS_CASES)
def test_an_aliased_resolved_axis_is_refused_at_runtime(name):
    """@brief 负例（blocker 2）：input/output/final_state 的别名轴在运行时被 typed 拒绝。"""
    instance = _instance()
    aliased = _aliased_instance(instance, name)
    problems = capacity_axis_problems(aliased)
    assert problems and "cannot prove this axis is not a resolved-axis alias" in problems[0]
    with pytest.raises(VarlenContractError) as excinfo:
        require_capacity_axes(aliased)
    assert "refusing rather than comparing the whole capacity buffer" in str(excinfo.value)
    resolved = resolved_shape_for_case(instance, "permuted")
    binding = aliased.tensor(name)
    with pytest.raises(VarlenContractError) as excinfo:
        padding_slabs(aliased, resolved, binding.shape, binding.dim_terms)
    assert "resolved-axis alias" in str(excinfo.value)


def test_the_comparator_refuses_a_forged_resolved_declaration():
    """@brief 负例（blocker 4/2）：比较器重新核对 resolved 声明、输入身份与真实字节，不信调用方对象。

    ``_check_v2_case`` is the last gate before a verdict.  A ``V2Case`` carrying a resolved shape from
    another case (or another ``input_digest``) would move the logical region and mislabel the evidence, so
    it is refused instead of being compared.
    """
    contract = _contract()
    abi = _abi()
    instance = contract.instance
    data = prepare_v2_case(contract, "permuted", 907)  # b_tiny 1/1
    honest = _candidate_bytes(contract, data)
    other = resolved_shape_for_case(instance, "canonical")  # a_full 16/4
    assert (other.total_tokens, other.num_seqs) != (data.resolved_shape.total_tokens, data.resolved_shape.num_seqs)

    for forged in (
        dataclasses.replace(data, resolved_shape=other),
        dataclasses.replace(data, resolved=other.resolved_scalars()),
        dataclasses.replace(data, resolved_document=other.document()),
        dataclasses.replace(data, resolved_document=None),
        dataclasses.replace(data, resolved_shape=None),
        dataclasses.replace(data, input_digest="0" * 64),
    ):
        with pytest.raises(VarlenContractError):
            _check_v2_case(contract, abi, forged, honest, "permuted")

    # The same bytes under a different case name are not that case's batch: the declared profile wins.
    with pytest.raises(VarlenContractError) as excinfo:
        _check_v2_case(contract, abi, data, honest, "canonical")
    assert "are not the declared profile" in str(excinfo.value)


def test_a_case_without_a_resolved_shape_cannot_compare_the_whole_capacity_buffer():
    """@brief 负例（blocker 2/3）：``resolved_shape=None`` 不能跳过边界/padding/digest 校验。

    Pre-fix the comparator gated every L2 check on ``data.resolved_shape is not None``, so deleting the
    field fell through to a whole-capacity comparison that ran no boundary validation, no digest recompute
    and no padding check.  The reference's ``output`` padding is zeros (the oracle never writes it), so the
    reference bytes fed back verbatim returned ``ok=True`` with ``resolved_shape=None``.  The gate is now
    the instance kind, and the missing value is its own typed refusal.
    """
    contract = _contract()
    abi = _abi()
    data = prepare_v2_case(contract, "permuted", 907)
    forged = dataclasses.replace(data, resolved_shape=None)
    # ``_candidate_bytes`` only poisons the padding of a *resolved* case, so these are the reference bytes
    # themselves (``output`` padding = 0), exactly the payload the bypass used to accept.
    reference_bytes = _candidate_bytes(contract, forged)
    with pytest.raises(VarlenContractError) as excinfo:
        _check_v2_case(contract, abi, forged, reference_bytes, "permuted")
    assert "carries no resolved shape" in str(excinfo.value)
    # The same bytes under the real case are refused for the *right* reason: their padding is not the
    # declared poison.  That is what the forged case used to skip.
    real = _check_v2_case(contract, abi, data, reference_bytes, "permuted")
    assert not real.ok and "padding byte(s) outside the logical region" in real.message


def test_a_result_axis_that_does_not_name_the_capacity_is_refused_before_a_verdict():
    """@brief 负例（blocker 1/2）：结果轴的 capacity 尺寸未命名 → prepare/compare 都拒绝，不能变成整缓冲区比较。

    ``"4"``/``"2*2"``/``"H*2"``/``"H*D"`` all pass the shape algebra (and the real OpSpec validator used to
    accept them on ``final_state``).  With that declaration the logical region is the whole capacity
    buffer: the scale becomes 1.3306672482102278e16 (poison-dominated) and a wrong logical ``final_state``
    with intact padding returns ``ok=True, n_bad=0``.  The rule is role-based, so ``dt_bias``'s legitimate
    ``H*D`` on a *non-result* tensor is untouched (see the shared test).
    """
    contract = _contract()
    for term in ("4", "2*2", "H*2", "H*D"):
        tensors = [
            dataclasses.replace(binding, dim_terms=(term,) + tuple(binding.dim_terms[1:]))
            if binding.name == "final_state"
            else binding
            for binding in contract.instance.tensors
        ]
        forged_instance = dataclasses.replace(contract.instance, tensors=tuple(tensors))
        problems = capacity_axis_problems(forged_instance)
        assert problems and "a result tensor axis of length" in problems[0]
        with pytest.raises(VarlenContractError):
            require_capacity_axes(forged_instance)
        forged_contract = dataclasses.replace(contract, instance=forged_instance)
        with pytest.raises(VarlenContractError):
            prepare_v2_case(forged_contract, "permuted", 907)
        # The comparator refuses the same instance before any byte is compared.
        abi = derive_tensor_memory_contract(forged_instance, target=TARGET)
        data = prepare_v2_case(contract, "permuted", 907)
        with pytest.raises(VarlenContractError):
            _check_v2_case(
                forged_contract, abi, data, _candidate_bytes(contract, data), "permuted"
            )


def test_a_packed_input_axis_cannot_feed_padding_to_a_custom_error_scale():
    """@brief 负例：未声明的 q=H*D 必须在 custom scale 看到任何输入前被拒绝。"""
    seen: list[tuple[str, tuple[int, ...]]] = []
    contract = _contract_with_error_scale(seen)
    tensors = [
        dataclasses.replace(binding, dim_terms=("H*D", "H", "D"))
        if binding.name == "q"
        else binding
        for binding in contract.instance.tensors
    ]
    forged = dataclasses.replace(contract.instance, tensors=tuple(tensors))
    problems = capacity_axis_problems(forged)
    assert any("neither a canonical resolved axis" in problem for problem in problems)
    forged_contract = dataclasses.replace(contract, instance=forged)
    with pytest.raises(VarlenContractError):
        prepare_v2_case(forged_contract, "permuted", 907)
    assert seen == [], "the custom hook must not see a full-capacity input with poison padding"
    abi = derive_tensor_memory_contract(forged, target=TARGET)
    data = prepare_v2_case(contract, "permuted", 907)
    seen.clear()  # discard the valid prepare's scale; only the forged comparator may not run it
    with pytest.raises(VarlenContractError):
        _check_v2_case(forged_contract, abi, data, _candidate_bytes(contract, data), "permuted")
    assert seen == []


def test_a_strided_capacity_binding_fails_closed_at_runtime():
    """@brief 负例：手搓实例的 strided 布局同样不能进入整缓冲区比较。"""
    instance = _instance()
    tensors = []
    for binding in instance.tensors:
        if binding.name == "output":
            binding = dataclasses.replace(
                binding, layout="strided", strides=(1, 16, 32)
            )
        tensors.append(binding)
    aliased = dataclasses.replace(instance, tensors=tuple(tensors))
    problems = capacity_axis_problems(aliased)
    assert any("layout 'strided' is not row-major/contiguous" in item for item in problems)
    with pytest.raises(VarlenContractError):
        require_capacity_axes(aliased)


@pytest.mark.parametrize("name", ALIAS_CASES)
def test_a_prepared_case_with_an_aliased_axis_fails_closed(name, monkeypatch):
    """@brief 负例（blocker 2）：``prepare_v2_case`` 不会退回整缓冲区比较，而是拒绝。"""
    contract = _contract()
    aliased = _aliased_instance(contract.instance, name)
    monkeypatch.setattr(
        v2_runtime, "prepare_v2_case", lambda *a, **k: prepare_v2_case(contract, "permuted", 1)
    )
    with pytest.raises(VarlenContractError):
        require_capacity_axes(aliased)
    # The comparator path uses the same rule; a hand-built V2Case with an aliased instance is refused
    # before any byte is compared.
    data = prepare_v2_case(contract, "permuted", 907)
    abi = derive_tensor_memory_contract(aliased, target=TARGET)
    with pytest.raises(VarlenContractError):
        _check_v2_case(
            dataclasses.replace(contract, instance=aliased), abi, data, _candidate_bytes(contract, data), "permuted"
        )


# ===================================================================================================
# 3. Profile-bound evaluation input identity
# ===================================================================================================


def _inputs_with_cu(instance, values, case="canonical"):
    """@brief 用一组偏移替换某个 case 的 ``cu_seqlens``（保持容量形状）。"""
    contract = _contract()
    produced = contract.reference.generate(instance, case, 7)
    inputs = dict(produced)
    inputs["cu_seqlens"] = torch.tensor(list(values), dtype=torch.int32)
    return inputs


def test_the_input_digest_binds_the_declared_workload_profile(monkeypatch):
    """@brief blocker 3：digest 显式绑定声明的 profile id/seq_lens，并声明自己的覆盖范围。"""
    instance = _instance()
    captured: list[dict] = []

    def capture(document):
        captured.append(copy.deepcopy(document))
        return "0" * 64

    monkeypatch.setattr(domain_module, "fingerprint", capture)
    inputs = _inputs_with_cu(instance, [0, 1, -1, -1, -1], "permuted")  # b_tiny 1/1
    domain_module.evaluation_input_digest(instance, "permuted", 11, inputs)
    document = captured[-1]
    assert document["identity_scope"] == "boundary-bytes+shape-metadata+declared-profile"
    assert document["workload_profile_id"] == "b_tiny", "the declared profile of case 1 is b_tiny"
    assert document["declared_profile"] == {"id": "b_tiny", "seq_lens": [1]}
    assert document["resolved"] == {"total_tokens": 1, "num_seqs": 1, "cu_seqlens_prefix_sha256": document["resolved"]["cu_seqlens_prefix_sha256"]}
    assert document["workload_profile_digest"] == domain_module.workload_profile_digest(instance)
    assert "canonical_seq_lens" not in document["resolved"], "resolved values are byte-derived, not the profile"


def test_the_input_digest_binds_the_canonical_profile_even_when_totals_match():
    """@brief blocker 3：total/count 相同但声明 profile 的 canonical 分段不同 → digest 必须不同。"""
    instance = _instance()
    inputs = _inputs_with_cu(instance, [0, 5, 7, 9, -1], "skewed")  # c_mid 9/3
    baseline = evaluation_input_digest(instance, "skewed", 5, inputs)
    profiles = tuple(instance.workload_profiles)
    swapped = tuple(
        dataclasses.replace(profile, seq_lens=(4, 4, 1)) if profile.id == "c_mid" else profile for profile in profiles
    )
    changed = dataclasses.replace(instance, workload_profiles=swapped)
    moved = evaluation_input_digest(changed, "skewed", 5, inputs)
    assert moved != baseline, "a different declared canonical profile must change the identity"
    assert evaluation_input_digest(instance, "skewed", 5, inputs) == baseline


def test_a_batch_that_is_not_the_declared_profile_is_refused():
    """@brief blocker 3：实际边界推出的 total/count 不是声明 profile → typed 拒绝。"""
    instance = _instance()
    assert workload_profile_for_case(instance, "canonical").total_tokens == 16
    inputs = _inputs_with_cu(instance, [0, 4, 8, 12, 12], "canonical")  # 12/4, not 16/4
    with pytest.raises(TaskContractError) as excinfo:
        evaluation_input_digest(instance, "canonical", 3, inputs)
    assert "declares the workload profile" in str(excinfo.value)


def test_the_generator_cannot_shrink_the_batch_under_a_case_name():
    """@brief blocker 3：任务生成器写出的 cu 与声明 profile 不一致时，``prepare`` 先拒绝。"""
    base = _contract()
    instance = base.instance

    def gen(case, seed, inst, resolved=None):
        produced = base.reference._gen_inputs(case, seed, inst, resolved=resolved)
        patched = dict(produced)
        patched["cu_seqlens"] = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32)
        return patched

    contract = dataclasses.replace(
        base, reference=dataclasses.replace(base.reference, _gen_inputs=gen)
    )
    with pytest.raises(VarlenContractError) as excinfo:
        prepare_v2_case(contract, "canonical", 1)
    assert "must end at the resolved total_tokens (16)" in str(excinfo.value)
    assert instance.case_domain.cases[0] == "canonical"


def test_the_input_digest_is_not_a_byte_complete_hash():
    """@brief blocker 3：payload 字节不在 digest 范围内（scope 已写明），改动由数值/padding 检查抓。"""
    instance = _instance()
    contract = _contract()
    produced = contract.reference.generate(instance, "canonical", 9)
    baseline = evaluation_input_digest(instance, "canonical", 9, produced)
    altered = dict(produced)
    altered["q"] = produced["q"].clone()
    altered["q"][0, 0, 0] = altered["q"][0, 0, 0] + 1.0
    assert evaluation_input_digest(instance, "canonical", 9, altered) == baseline
    # The digest does move with the boundary bytes and with the case/seed it names.
    with_cu = dict(produced)
    with_cu["cu_seqlens"] = produced["cu_seqlens"].clone()
    with_cu["cu_seqlens"][1] = 3
    assert evaluation_input_digest(instance, "canonical", 9, with_cu) != baseline
    assert evaluation_input_digest(instance, "canonical", 10, produced) != baseline


def test_the_comparator_revalidates_the_storage_boundary_not_only_the_rounded_claim():
    """@brief 负例：实际 initial=full、rounded/ref/digest=tiny 时必须拒绝，不能重标批次。"""
    contract = _contract()
    abi = _abi()
    tiny = prepare_v2_case(contract, "permuted", 907)
    full = prepare_v2_case(contract, "canonical", 907)
    forged_initial = dict(tiny.initial)
    forged_initial["cu_seqlens"] = full.initial["cu_seqlens"]
    forged = dataclasses.replace(tiny, initial=forged_initial)
    with pytest.raises(VarlenContractError, match="initial.*storage|initial.*boundary|not the boundary"):
        _check_v2_case(contract, abi, forged, _candidate_bytes(contract, tiny), "permuted")


# ===================================================================================================
# 4. HostRunEvidence for every ordered call (success and failure)
# ===================================================================================================


class _FakeHostProgram:
    """@brief 只用真实 ``MetalHostProgram._run_document`` 字段集的 Host Program 替身。"""

    candidate_sha256 = "a" * 64
    target_id = "apple-m4-max"
    native_digest = "b" * 64
    inside_timing = True
    _zero_flags = {"work_items": False, "item_counts": False}
    entry = SimpleNamespace(path="host/tiling.c")
    build = SimpleNamespace(
        program_sha256="c" * 64,
        assembled_sha256="d" * 64,
        native_sha256="e" * 64,
        toolchain=SimpleNamespace(
            execution_backend="native64", platform="macos", clang="clang", clang_version="18"
        ),
    )
    failure_document = MetalHostProgram.failure_document
    _run_document = MetalHostProgram._run_document

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, argset, *, audit, profile=False):  # noqa: ANN001 - runtime protocol
        run = HostRun(
            status=0,
            error_code=None,
            error_detail="",
            host_wall_ns=7,
            input_reads=(),
            spans=(),
            resolved_scalars=dict(argset.resolved_scalars),
            audit=True,
        )
        document = self._run_document(
            run,
            writes=[],
            policy={},
            timing_mode="audit",
            typed_error=None,
            host_wall_ns=run.host_wall_ns,
            resolved_shape=argset.resolved_shape_document,
            evaluation_input_digest=argset.evaluation_input_digest or None,
        )
        self.calls.append(document)
        return document


class _StubArgSet:
    """@brief ``MetalV2Runtime`` 用的 argset 替身（输入字节 + 本 case 的 resolved 上下文）。"""

    def __init__(self) -> None:
        self.resolved_scalars: dict[str, int] = {}
        self.resolved_shape_document = None
        self.evaluation_input_digest = ""
        self.initial: dict[str, bytes] = {}
        self.padding_calls: list = []
        self.candidate: dict[str, bytes] = {}
        self.workspace_zero = {}

    def load_initial(self, initial):  # noqa: ANN001 - runtime protocol
        self.initial = dict(initial)

    def apply_fill_plan(self, plan, *, reset_guards=True):  # noqa: ANN001 - runtime protocol
        return None

    def load_capacity_padding(self, instance, resolved):  # noqa: ANN001 - runtime protocol
        self.padding_calls.append((instance.size, None if resolved is None else resolved.total_tokens))

    def read_results(self):  # noqa: ANN001 - runtime protocol
        return dict(self.candidate)

    def safety_violations(self):  # noqa: ANN001 - runtime protocol
        return []


def _fake_runtime(monkeypatch, cases, *, fail_on=None, determinism=True):
    """@brief 构造一个不碰 Metal 的 ``MetalV2Runtime``，只用真实 correctness 累积逻辑。"""
    contract = _contract()
    runtime = object.__new__(v2_runtime.MetalV2Runtime)
    runtime.contract = contract
    runtime.abi = _abi()
    runtime.geometry = SimpleNamespace(active_stages=())
    runtime.provider = "candidate"
    runtime.host_plan = None
    runtime.host_inside_timing = True
    runtime.selection = SimpleNamespace(identity="0" * 64)
    runtime.argset = _StubArgSet()
    runtime.last_host_run = None
    runtime.launch_runs = []
    runtime.benchmark_batches = []
    runtime.host_program = _FakeHostProgram()
    prepared: dict[str, V2Case] = {}

    def fake_run_once(*, benchmark, host_audit=True, **_kwargs):  # noqa: ANN001 - runtime protocol
        case = prepared["current"]
        data = prepared[case]
        index = prepared["index"]
        if fail_on is not None and index == fail_on:
            raise v2_runtime.HostProgramFailed(
                HostRun(
                    status=-2,
                    error_code="host_program_invalid",
                    error_detail="the candidate Host Program rejected this batch",
                    host_wall_ns=11,
                    input_reads=(),
                    spans=(),
                    resolved_scalars={},
                    audit=True,
                )
            )
        runtime.last_host_run = runtime.host_program.execute(runtime.argset, audit=host_audit)
        runtime.argset.candidate = _candidate_bytes(contract, data)
        return []

    runtime.run_once = fake_run_once
    original_prepare = v2_runtime.prepare_v2_case

    def spy_prepare(contract_, case, seed):  # noqa: ANN001 - spy
        data = original_prepare(contract_, case, seed)
        prepared[case] = data
        prepared["current"] = case
        prepared["index"] = list(cases).index(case)
        return data

    monkeypatch.setattr(v2_runtime, "prepare_v2_case", spy_prepare)
    result = runtime.correctness(cases=cases, seed=77, determinism=determinism)
    return runtime, result


def test_every_ordered_call_keeps_its_host_run_and_launch_record(monkeypatch):
    """@brief 负例（blocker 4）：连续多个 case 的 HostRun/launch 证据全部保留，不只最后一个。"""
    cases = tuple(_instance().case_domain.cases[:3])
    runtime, (results, safety, nondet, dispatch, host_runs) = _fake_runtime(monkeypatch, cases)
    assert all(result.ok for result in results), [result.message for result in results if not result.ok]
    trials = len(v2_runtime.CORRECTNESS_TRIALS)
    # One HostRunEvidence per *distinct* call identity: the three correctness trials of a case feed the
    # same read-only inputs and resolved shape, so they collapse to one document instead of three copies
    # — but every case keeps its own, and no case keeps only the last one.
    assert isinstance(host_runs, list) and len(host_runs) == len(cases)
    resolved = [resolved_shape_for_case(runtime.contract.instance, case) for case in cases]
    for index, (document, shape) in enumerate(zip(host_runs, resolved)):
        assert document["evaluation_case"] == cases[index]
        assert document["evaluation_case_ordinal"] == index
        assert document["resolved_shape"] == shape.document()
        assert document["resolved_scalars"] == shape.resolved_scalars()
        assert len(document["evaluation_input_digest"]) == 64
    assert len({document["evaluation_input_digest"] for document in host_runs}) == len(cases)
    assert len({(document["resolved_shape"]["total_tokens"]) for document in host_runs}) >= 2
    # Launch evidence is per call, in order, and not deduplicated by the (frozen, identical) dispatch
    # signature: one entry per ordered case with its own resolved shape/digest.
    assert [item["ordinal"] for item in runtime.launch_runs] == list(range(len(cases) * trials))
    assert [item["case"] for item in runtime.launch_runs] == [case for case in cases for _ in range(trials)]
    assert [item["case_ordinal"] for item in runtime.launch_runs] == [index for index in range(len(cases)) for _ in range(trials)]
    assert len({item["evaluation_input_digest"] for item in runtime.launch_runs}) == len(cases)
    assert all(item["provider"] == runtime.provider for item in runtime.launch_runs)

    # The executor's record() merges the whole list and the per-call launch records exactly once.
    executor = object.__new__(track_adapter.MetalV2Executor)
    executor.evidence = {"dispatch": [], "host_runs": []}
    executor.runtime = runtime
    executor.record(dispatch=dispatch, host=host_runs)
    executor.record(dispatch=dispatch, host=host_runs)
    assert len(executor.evidence["host_runs"]) == len(cases)
    assert [item["ordinal"] for item in executor.evidence["launch_runs"]] == list(range(len(cases) * trials))


def test_a_failed_call_keeps_the_same_document_fields_as_a_success(monkeypatch):
    """@brief 负例（blocker 4）：失败 HostRun 也带 resolved 形状/输入身份，字段集不缩水。"""
    cases = tuple(_instance().case_domain.cases[:2])
    runtime, (results, safety, nondet, dispatch, host_runs) = _fake_runtime(monkeypatch, cases, fail_on=1)
    assert [result.ok for result in results] == [True, False]
    assert len(host_runs) == 2, "the successful case and the failed call must both be recorded"
    success, failure = host_runs[0], host_runs[-1]
    assert success["typed_error"] is None and success["host_status"] == 0
    assert failure["typed_error"] == "host_program_invalid"
    assert failure["workspace_writes"] == []
    # Same field set, including the two L2 fields the failure used to drop.
    assert set(failure) == set(success)
    assert failure["resolved_shape"] == resolved_shape_for_case(runtime.contract.instance, cases[1]).document()
    assert failure["evaluation_input_digest"] == results[1].input_sha256
    # A failed run publishes no resolved scalars (the whole run is discarded), but the *declared* batch
    # it was invoked for is still recoverable from the resolved shape — that is the point of the field.
    assert failure["resolved_scalars"] == {}
    assert failure["resolved_shape"]["total_tokens"] == resolved_shape_for_case(
        runtime.contract.instance, cases[1]
    ).total_tokens
    assert failure["evaluation_case"] == cases[1]
    # The failure is also a launch (it happened), so the ordered sequence is still complete.
    assert runtime.launch_runs[-1]["case"] == cases[1]
    assert runtime.launch_runs[-1]["ordinal"] == len(runtime.launch_runs) - 1


def test_a_plain_l1_run_keeps_the_historical_evidence_shape(monkeypatch):
    """@brief L1/fixed 证据不新增键：``host_runs`` 仍是单条，``launch_runs`` 不增长。"""
    contract = _contract("mixed")
    runtime = object.__new__(v2_runtime.MetalV2Runtime)
    runtime.contract = contract
    runtime.abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    runtime.geometry = SimpleNamespace(active_stages=())
    runtime.provider = "candidate"
    runtime.host_plan = None
    runtime.host_inside_timing = True
    runtime.selection = SimpleNamespace(identity="0" * 64)
    runtime.argset = _StubArgSet()
    runtime.last_host_run = None
    runtime.launch_runs = []
    runtime.benchmark_batches = []
    runtime.host_program = _FakeHostProgram()
    prepared: dict[str, V2Case] = {}

    def fake_run_once(*, benchmark, host_audit=True, **_kwargs):  # noqa: ANN001 - runtime protocol
        data = prepared["current"]
        runtime.last_host_run = runtime.host_program.execute(runtime.argset, audit=host_audit)
        runtime.argset.candidate = _candidate_bytes(contract, data)
        return []

    runtime.run_once = fake_run_once
    original = v2_runtime.prepare_v2_case

    def spy(contract_, case, seed):  # noqa: ANN001 - spy
        data = original(contract_, case, seed)
        prepared["current"] = data
        return data

    monkeypatch.setattr(v2_runtime, "prepare_v2_case", spy)
    results, _safety, _nondet, _dispatch, host_runs = runtime.correctness(
        cases=("canonical",), seed=5, determinism=False
    )
    assert results and results[0].ok
    assert isinstance(host_runs, list) and len(host_runs) == 1
    assert host_runs[0]["resolved_shape"] is None
    assert host_runs[0]["evaluation_input_digest"]
    assert runtime.launch_runs == [], "a non-capacity instance must not grow the L2 launch list"


# ===================================================================================================
# 5. Continuous-batching benchmark mixture and dispatch/fallback negatives
# ===================================================================================================


def test_the_benchmark_mixture_follows_the_declared_profile_order():
    """@brief 负例（blocker 5）：计时 mixture 按声明顺序跑多个 resolved 批次，不只 cases[0]。"""
    instance = _instance()
    cases = tuple(instance.case_domain.cases)
    mixture = benchmark_mixture(instance, cases, rotation=1)
    assert len(mixture) >= len(cases)
    assert mixture[0] == cases[0]
    assert list(mixture[: len(cases)]) == list(cases)
    shapes = [resolved_shape_for_case(instance, case) for case in mixture]
    assert len({(shape.total_tokens, shape.num_seqs) for shape in shapes}) >= 2
    assert any(shape.is_full_capacity for shape in shapes)
    # A fixed/L1 instance keeps the historical single-case rotation.
    l1 = _instance("mixed")
    assert set(benchmark_mixture(l1, l1.case_domain.cases, rotation=3)) == {l1.case_domain.cases[0]}

    batches = [
        {
            "ordinal": index,
            "case": case,
            "provider": "candidate",
            "resolved_shape": resolved_shape_for_case(instance, case).document(),
            "evaluation_input_digest": "f" * 64,
        }
        for index, case in enumerate(mixture)
    ]
    assert benchmark_mixture_problems(instance, batches, cases=cases) == ()


def test_every_rotation_set_carries_its_own_resolved_batch():
    """@brief 负例（blocker 5）：rotation set 不再共享 ``cases[0]`` 的输入，而是各自的 resolved 批次。

    The pre-fix benchmark prepared one ``V2Case`` (``cases[0]``, the full-capacity profile) and copied it
    into every rotation set, so a candidate that only handled the capacity point was never timed on a
    smaller batch.  The mixture builder must bind each set to its own dataset, in declared order.
    """
    contract = _contract()
    cases = tuple(contract.instance.case_domain.cases)
    plan, datasets = benchmark_datasets(contract, cases, rotation=1, seed=11)
    assert list(plan[: len(cases)]) == list(cases)
    assert set(datasets) == set(plan)
    shapes = {case: datasets[case].resolved for case in datasets}
    assert len({(value["total_tokens"], value["num_seqs"]) for value in shapes.values()}) >= 2
    assert len({datasets[case].input_digest for case in datasets}) == len(datasets)
    assert all(datasets[case].resolved_shape is not None for case in datasets)
    # Fixed/L1 keeps the historical single-case rotation.
    l1 = _contract("mixed")
    l1_cases = tuple(l1.instance.case_domain.cases)
    l1_plan, l1_datasets = benchmark_datasets(l1, l1_cases, rotation=3, seed=11)
    assert set(l1_plan) == {l1_cases[0]}
    assert set(l1_datasets) == {l1_cases[0]}


def test_a_capacity_only_or_misattributed_mixture_is_refused():
    """@brief 负例（blocker 5）：只跑满 capacity、乱序、伪造 resolved、fallback 混入全部拒绝。"""
    instance = _instance()
    cases = tuple(instance.case_domain.cases)
    full = resolved_shape_for_case(instance, cases[0])
    assert full.is_full_capacity
    single = [
        {
            "ordinal": 0,
            "case": cases[0],
            "provider": "candidate",
            "resolved_shape": full.document(),
            "evaluation_input_digest": "1" * 64,
        }
    ]
    problems = benchmark_mixture_problems(instance, single, cases=cases)
    assert any("never timed declared workload profile(s)" in item for item in problems)
    assert any("the declaration covers" in item for item in problems)
    # The one batch that did run *was* the full-capacity profile, so that specific complaint must not fire.
    assert not any("full-capacity" in item for item in problems)

    mixture = benchmark_mixture(instance, cases, rotation=2)
    batches = [
        {
            "ordinal": index,
            "case": case,
            "provider": "candidate",
            "resolved_shape": resolved_shape_for_case(instance, case).document(),
            "evaluation_input_digest": "2" * 64,
        }
        for index, case in enumerate(mixture)
    ]
    # out of order
    reordered = copy.deepcopy(batches)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert any("declared continuous-batching order" in item for item in benchmark_mixture_problems(instance, reordered, cases=cases))
    # forged resolved shape (capacity full shape under a smaller case)
    forged = copy.deepcopy(batches)
    forged[1]["resolved_shape"] = full.document()
    assert any("declares" in item for item in benchmark_mixture_problems(instance, forged, cases=cases))
    # fallback time mixed into the variant's mixture
    fallback = copy.deepcopy(batches)
    fallback[1]["provider"] = "fallback"
    problems = benchmark_mixture_problems(instance, fallback, cases=cases)
    assert any("fallback's time is not the generated variant's" in item for item in problems)
    # missing input identity
    anonymous = copy.deepcopy(batches)
    anonymous[1]["evaluation_input_digest"] = None
    assert any("carries no evaluation_input_digest" in item for item in benchmark_mixture_problems(instance, anonymous, cases=cases))
    # a non-capacity instance cannot publish a mixture at all
    l1 = _instance("mixed")
    assert benchmark_mixture_problems(l1, batches, cases=l1.case_domain.cases)


def test_a_mixture_benchmark_is_not_reported_as_a_single_shape_speedup():
    """@brief 负例（blocker 5）：mixture 的 measurement scope 与单形状基线不可比。"""
    instance = _instance()
    cases = tuple(instance.case_domain.cases)
    mixture = benchmark_mixture(instance, cases, rotation=1)
    batches = [
        {
            "ordinal": index,
            "case": case,
            "provider": "candidate",
            "resolved_shape": resolved_shape_for_case(instance, case).document(),
            "evaluation_input_digest": "3" * 64,
        }
        for index, case in enumerate(mixture)
    ]
    stats = SimpleNamespace(median_s=1.0, median_us=1e6)
    mixed = V2Benchmark(
        stats=stats,
        baselines={},
        rotation_sets=len(batches),
        dispatch=[],
        host=None,
        guard_violations=[],
        noise_cv=0.0,
        attempts=1,
        batches=batches,
    )
    assert mixed.measurement_scope == "continuous_batching_mixture"
    single = V2Benchmark(
        stats=stats,
        baselines={},
        rotation_sets=1,
        dispatch=[],
        host=None,
        guard_violations=[],
        noise_cv=0.0,
        attempts=1,
        batches=batches[:1],
    )
    assert single.measurement_scope == "single_shape"


def test_a_mixture_measurement_never_publishes_a_speedup():
    """@brief 负例（blocker 5）：mixture 与单形状基线的比值不发布，单形状测量照常发布。"""
    from tcenv.evaluate import speedup_policy

    speedups = {"torch_mps": 2.0}
    baselines = {"torch_mps": 100.0}
    mixture = speedup_policy(speedups, baselines, scope="continuous_batching_mixture")
    assert mixture["speedup"] is None and mixture["speedups"] == {}
    assert mixture["best_baseline"] is None
    assert "not like-for-like" in mixture["withheld"]
    single = speedup_policy(speedups, baselines, scope="single_shape")
    assert single["speedup"] == 2.0
    assert single["speedups"] == {"torch_mps": 2.0}
    assert single["best_baseline"] == "torch_mps"
    assert single["withheld"] is None
    assert speedup_policy({}, {}, scope="single_shape") == {
        "speedups": {},
        "speedup": None,
        "best_baseline": None,
        "withheld": None,
    }


def test_resolved_values_are_not_a_dispatch_domain_point():
    """@brief 负例（blocker 5）：伪造 resolved symbols / 未知 size 的请求都不会被 dispatch 接受。"""
    contract = _contract()
    instance = contract.instance
    plan = normalize_semantic_domain_plan(contract.spec, [instance], target=TARGET)
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    resolver = ExactDispatchResolver(
        domain_plan=plan,
        portfolio=portfolio,
        evidence=evidence_all_valid(portfolio),
        spec=contract.spec,
        target=TARGET,
        instances=(instance,),
    )
    point = plan.find("fp16", "capacity", "golden")
    assert point is not None
    request = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage="golden")
    bound = resolver.resolve(request)
    assert bound.outcome != "semantic_unresolved", bound.detail
    assert "guard is not decidable" in bound.detail, "the request itself binds; only the fixture guards are unknown"

    forged = copy.deepcopy(request.document())
    forged["symbols"] = dict(forged["symbols"], total_tokens=1, num_seqs=1)
    refused = resolver.resolve(forged)
    assert refused.outcome == "semantic_unresolved"
    assert "contradicts the authoritative semantic instance" in refused.detail

    unknown = copy.deepcopy(request.document())
    unknown["size"] = "resolved_batch"
    unknown["instance_key"] = "fp16/resolved_batch/golden"
    unmatched = resolver.resolve(unknown)
    assert unmatched.outcome == "unsupported_contract_capability"


def test_a_single_profile_tier_is_not_a_declarable_boundary():
    """@brief 结论撤回：单 profile **不是**可声明的合法边界（文档层 typed 拒绝）。

    ``_validate_capacity_tier`` requires at least two workload profiles with two distinct resolved totals
    *and* two distinct counts, so no production OpSpec can declare a single-profile L2 tier — a tier that
    only ever runs one resolved shape is an L1 tier under another name.  The earlier version of this test
    built one with ``dataclasses.replace`` and called it a "legal boundary"; that was a fabricated
    declaration layer, so the claim is withdrawn here and the real validator is the assertion.

    What the runtime still guarantees is *defensive*, and is pinned below: if such an in-memory instance
    ever reaches the benchmark, it is measured as one shape (no invented mixture, no spurious failure, no
    withheld speedup).  That is a robustness property, not an accepted declaration.
    """
    document = copy.deepcopy(load_strict_json(OPSPEC_PATH))
    document["sizes"]["capacity"]["workload_profiles"] = [{"id": "a_full", "seq_lens": [4, 4, 4, 4]}]
    with pytest.raises(OpSpecError, match="at least two workload profiles"):
        opspec_v2_from_document(document, source="single-profile")

    # Defensive runtime path for an instance no valid document can produce: every batch carries the same
    # declared shape, so the mixture self-check must pass as a *single shape* and the report must not
    # pretend it timed a mixture.
    instance = dataclasses.replace(
        _instance(),
        workload_profiles=tuple(sorted(_instance().workload_profiles, key=lambda profile: profile.id))[:1],
    )
    cases = tuple(instance.case_domain.cases)
    mixture = benchmark_mixture(instance, cases, rotation=2)
    assert len(mixture) >= len(cases)
    batches = [
        {
            "ordinal": index,
            "case": case,
            "provider": "candidate",
            "resolved_shape": resolved_shape_for_case(instance, case).document(),
            "evaluation_input_digest": "3" * 64,
        }
        for index, case in enumerate(mixture)
    ]
    assert benchmark_mixture_problems(instance, batches, cases=cases) == ()
    scope = V2Benchmark(
        stats=None, baselines={}, rotation_sets=len(mixture), dispatch=[], host=None,
        guard_violations=[], noise_cv=0.0, attempts=0, batches=batches,
    ).measurement_scope
    assert scope == "single_shape"


def test_the_benchmark_evidence_names_every_timed_batch_and_its_scope():
    """@brief 负例（blocker 5）：证据里能查到每个 timed 批次的 resolved 形状/输入身份与 scope。"""
    instance = _instance()
    cases = tuple(instance.case_domain.cases)
    # ``rotation > len(cases)`` on purpose: the second cycle repeats the first cases, which is exactly where
    # the per-call ``ordinal`` must not be reused as ``case_ordinal``.
    mixture = benchmark_mixture(instance, cases, rotation=16)
    runtime = object.__new__(v2_runtime.MetalV2Runtime)
    runtime.contract = _contract()
    runtime.provider = "candidate"
    runtime.launch_runs = []
    runtime.benchmark_batches = []
    runtime.benchmark_mixture_notes = {}
    runtime.host_plan = None
    runtime.host_program = None
    runtime.argset = _StubArgSet()
    runtime.abi = _abi()
    runtime.geometry = SimpleNamespace(active_stages=())
    runtime.selection = SimpleNamespace(identity="0" * 64)
    runtime.last_host_run = None
    runtime.last_host_timing = None
    runtime.stage_args_builds = 0
    runtime.read_only_restores = 0
    runtime.candidate_command_buffers = 0
    runtime.benchmark_diagnostics = {}
    runtime.workspace_probe = {}
    runtime.limits = None
    runtime.bindings = ()
    datasets = {case: prepare_v2_case(runtime.contract, case, 3) for case in set(mixture)}
    # The *production* record path (``benchmark`` calls the same method before timing).
    runtime._record_benchmark_mixture(cases, mixture, datasets)
    assert benchmark_mixture_problems(instance, runtime.benchmark_batches, cases=cases) == ()
    assert all(batch["evaluation_input_digest"] for batch in runtime.benchmark_batches)
    assert len(runtime.launch_runs) == len(mixture)
    assert all(item["phase"] == "benchmark" for item in runtime.launch_runs)
    assert [item["audited"] for item in runtime.launch_runs].count(True) == 1
    # ordinals are per call (0..15), while case_ordinal is the *declared* index (0..8, then 0..6 again).
    assert [item["ordinal"] for item in runtime.launch_runs] == list(range(len(mixture)))
    assert [item["case"] for item in runtime.launch_runs] == list(mixture)
    assert [item["case_ordinal"] for item in runtime.launch_runs] == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6
    ]
    second = runtime.launch_runs[len(cases):]
    assert len(second) > 0
    assert [item["case"] for item in second] == list(cases[: len(second)])
    assert [item["case_ordinal"] for item in second] == list(range(len(second)))
    # The mixture is strictly ordered and covers the capacity point plus at least one smaller batch.
    totals = [batch["resolved_shape"]["total_tokens"] for batch in runtime.benchmark_batches]
    assert any(batch["resolved_shape"]["full_capacity"] for batch in runtime.benchmark_batches)
    assert len(set(totals)) >= 2
