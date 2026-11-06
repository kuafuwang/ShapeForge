"""V5 L2 correctness comparison: logical prefix only, padding checked separately (CPU; no device).

The L2 comparator has two jobs that a full-buffer comparison cannot do at once:

* compare ``output[:resolved_total]`` and ``final_state[:resolved_num_seqs]`` — the logical result;
* refuse a candidate that wrote *outside* that region, even when every logical element is in tolerance.

These tests pin the byte-level helpers the Metal correctness path uses, plus the report shape: a fixed/L1
case result keeps its historical keys, and only an L2 case adds ``resolved_shape``.  Nothing here opens a
Metal device or times anything.
"""

from __future__ import annotations

import torch

from tcenv.backends.metal.testing import CaseResult
from tcenv.backends.metal.v2_runtime import (
    POISON_PATTERN,
    _capacity_padding_bytes,
    _prefix_lengths,
    _unwritten_bytes_region,
    prepare_v2_case,
)
from tcenv.portfolio.domain import evaluation_input_digest
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.taskcontract.runtime_shape import padding_slabs, resolved_shape_for_case

KDA_DIR = REPO_ROOT / "tasks" / "varlen_kda"
POISON_ELEMENT = 0x5A


def _contract(size: str = "capacity"):
    return load_task_contract(KDA_DIR, dtype="fp16", size=size)


def _tensor_with_poison_padding(instance, resolved, shape, dtype=torch.float16, fill=0.0):
    """@brief 构造一个 logical 区域填充 ``fill``、padding 逐字节为声明 poison 的张量。"""
    tensor = torch.full(shape, float(fill), dtype=dtype)
    byte_view = tensor.view(torch.uint8).reshape(shape + (-1,))
    terms = {
        binding.name: tuple(binding.dim_terms)
        for binding in instance.tensors
    }["q"]
    _, slabs = padding_slabs(instance, resolved, shape, terms)
    for slab in slabs:
        byte_view[tuple(slab) + (slice(0, int(byte_view.shape[-1])),)] = int(instance.capacity.padding_poison_byte)
    return tensor


def test_padding_bytes_are_counted_outside_the_logical_region_only():
    """@brief padding 完好时 0；只改 padding 也必须被计数；改 logical 区域不计入 padding。"""
    instance = _contract().instance
    resolved = resolved_shape_for_case(instance, "permuted")  # 1/1 of 16/4
    shape = (16, 2, 8)
    tensor = _tensor_with_poison_padding(instance, resolved, shape)
    raw = tensor.numpy().tobytes()
    _logical, slabs = padding_slabs(instance, resolved, shape, ("total_tokens", "H", "D"))
    assert _capacity_padding_bytes(raw, shape, 2, slabs, instance.capacity.padding_poison_byte) == 0

    tampered = tensor.clone()
    tampered[5, 1, 3] = 1.0
    assert _capacity_padding_bytes(tampered.numpy().tobytes(), shape, 2, slabs, POISON_ELEMENT) > 0

    logical_write = tensor.clone()
    logical_write[0, 0, 0] = 3.0
    assert _capacity_padding_bytes(logical_write.numpy().tobytes(), shape, 2, slabs, POISON_ELEMENT) == 0


def test_unwritten_count_is_restricted_to_the_logical_region():
    """@brief 纯 write 输出“仍然 poison”的元素只在 logical 区域内统计。"""
    instance = _contract().instance
    resolved = resolved_shape_for_case(instance, "permuted")  # 1/1
    shape = (16, 2, 8)
    tensor = torch.full(shape, -1.0, dtype=torch.float16)
    byte_view = tensor.view(torch.uint8).reshape(shape + (-1,))
    byte_view[1:] = 0x5A  # padding still poison, logical rows not
    _logical, _slabs = padding_slabs(instance, resolved, shape, ("total_tokens", "H", "D"))
    logical = (slice(0, 1), slice(0, 2), slice(0, 8))
    assert _unwritten_bytes_region(tensor.numpy().tobytes(), shape, 2, logical) == 0
    empty = torch.full(shape, -1.0, dtype=torch.float16)
    empty.view(torch.uint8).reshape(shape + (-1,))[0] = POISON_PATTERN
    assert _unwritten_bytes_region(empty.numpy().tobytes(), shape, 2, logical) == 16


def test_prefix_lengths_stop_at_the_capacity_sentinel():
    """@brief 覆盖证据取实际分段，不把 capacity 尾部当序列。"""
    instance = _contract().instance
    assert _prefix_lengths(torch.tensor([0, 5, 7, 9, -1], dtype=torch.int32)) == (5, 2, 2)
    assert _prefix_lengths(torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)) == (4, 4, 4, 4)


def test_prepare_v2_case_binds_the_resolved_shape_and_input_digest():
    """@brief 每个 case 的 V2Case 带自己的 resolved 形状与可重放 digest（固定/L1 仍为空）。"""
    contract = _contract()
    seen = []
    for case in contract.instance.case_domain.cases:
        data = prepare_v2_case(contract, case, 600 + len(seen))
        resolved = resolved_shape_for_case(contract.instance, case)
        assert data.resolved == resolved.resolved_scalars()
        assert data.resolved_document is not None
        assert data.resolved_document["workload_profile_id"] == resolved.workload_profile_id
        assert data.input_digest == evaluation_input_digest(contract.instance, case, data.seed, data.rounded)
        seen.append((data.resolved["total_tokens"], data.resolved["num_seqs"]))
    assert len(set(seen)) >= 3, "the case sweep must cover several resolved batches"

    l1 = _contract("mixed")
    l1_case = prepare_v2_case(l1, "canonical", 7)
    assert l1_case.resolved == {}
    assert l1_case.resolved_document is None
    assert l1_case.input_digest, "an L1 case keeps its evaluation-input identity"


def test_a_case_result_only_gains_resolved_shape_for_l2():
    """@brief 固定/L1 报告形状不变；L2 报告多一个 resolved_shape 键。"""
    base = dict(
        case="canonical",
        seed=1,
        ok=True,
        scaled_err=0.1,
        allow=1.0,
        baseline_scaled_err=None,
        n_bad=0,
        n_elems=1,
        nan_count=0,
        unwritten=0,
        worst=None,
    )
    assert "resolved_shape" not in CaseResult(**base).to_json()
    document = {"kind": "l2", "total_tokens": 4, "num_seqs": 2}
    assert CaseResult(**base, resolved_shape=document).to_json()["resolved_shape"] == document


def _candidate_outputs(contract, data, *, padding_tamper: bool = False, logical_offset: float = 0.0):
    """@brief 用 reference 结果构造一份“候选输出”的存储字节（capacity 尺寸，padding 为声明 poison）。

    @param contract capacity 档的 TaskContract。
    @param data 该 case 的 ``V2Case``（提供 reference 终值与 resolved 形状）。
    @param padding_tamper 是否在 padding slab 里写一个 0（负例）。
    @param logical_offset 是否只给 logical 区域加一个常数（数值负例）。
    @return 张量名 → 存储字节。
    """
    import numpy as np

    from tcenv.dtypes import get_dtype
    from tcenv.tensors import to_raw

    instance = contract.instance
    out: dict[str, bytes] = {}
    for name in instance.reference_outputs:
        tensor = instance.tensor(name)
        shape = tuple(int(dim) for dim in tensor.shape)
        nbytes = get_dtype(tensor.dtype).nbytes
        value = data.ref[name].to(torch.float32).reshape(shape)
        logical, slabs = padding_slabs(instance, data.resolved_shape, shape, tuple(tensor.dim_terms))
        if logical_offset:
            value = value.clone()
            value[tuple(logical)] = value[tuple(logical)] + float(logical_offset)
        raw = bytearray(to_raw(value, tensor.dtype))
        array = np.frombuffer(raw, dtype=np.uint8).reshape(shape + (nbytes,))
        for slab in slabs:
            array[tuple(slab)] = int(instance.capacity.padding_poison_byte)
        if padding_tamper and slabs:
            array[tuple(slabs[0])] = 0
        out[name] = bytes(raw)
    return out


def test_the_l2_comparator_compares_the_logical_prefix_and_checks_padding():
    """@brief 真正的 Metal correctness 比较器（CPU 直接调用）：逻辑区域对≈通过，仅改 padding=失败。"""
    from tcenv.backends.metal.v2_runtime import _check_v2_case
    from tcenv.track.abi import derive_tensor_memory_contract
    from tcenv.target import load_target

    contract = _contract()
    abi = derive_tensor_memory_contract(contract.instance, target=load_target("apple-m4-max"))
    data = prepare_v2_case(contract, "permuted", 907)
    assert data.resolved_shape is not None and not data.resolved_shape.is_full_capacity

    honest = _check_v2_case(contract, abi, data, _candidate_outputs(contract, data), "permuted")
    assert honest.ok, honest.message
    assert honest.resolved_shape is not None
    assert honest.resolved_shape["total_tokens"] == data.resolved["total_tokens"]

    tampered = _check_v2_case(contract, abi, data, _candidate_outputs(contract, data, padding_tamper=True), "permuted")
    assert not tampered.ok
    assert "padding byte(s) outside the logical region" in tampered.message

    wrong = _check_v2_case(contract, abi, data, _candidate_outputs(contract, data, logical_offset=1.0), "permuted")
    assert not wrong.ok
    assert "elements out of tolerance" in wrong.message
