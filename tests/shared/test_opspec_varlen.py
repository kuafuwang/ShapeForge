"""varlen (packed sequence) instances: ``seq_lens`` derivation, identity and its negatives.

CPU only, backend free.  A varlen tier declares ``seq_lens`` and instantiates exactly like a fixed
tier: ``total_tokens``/``num_seqs``/``NSP1`` are derived from the distribution, tensors keep their
declared dims/strides, and the fixed tier stays the degenerate single-segment case.  The segment
boundaries are part of the instance identity, so two distributions with the same token total are
different problems.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tcenv.identity import fingerprint
from tcenv.opspec import OpSpecError, instantiate_v2, load_opspec


def _document(seq_lens: list[int] | None, *, heads: int = 2, dim: int = 4) -> dict:
    """@brief 一份最小的 varlen v2 文档；``seq_lens=None`` 时是固定 shape 档。

    @param seq_lens 分段长度分布（``None`` 表示固定 shape 档）。
    @param heads 头数。
    @param dim 单头维度。
    @return v2 文档。
    """
    tier: dict = {"H": heads, "D": dim, "usage": ["golden"]}
    if seq_lens is not None:
        tier["seq_lens"] = seq_lens
    else:
        tier.update({"total_tokens": 8, "num_seqs": 1, "NSP1": 2})
    return {
        "spec_version": 2,
        "op_name": "varlen_probe",
        "category": "elementwise",
        "formula": "y[t] = SCALE * x[t]",
        "type_vars": {"T": ["fp16"]},
        "symbols": {
            "H": {"min": 1},
            "D": {"min": 1},
            "total_tokens": {"min": 1},
            "num_seqs": {"min": 1},
            "NSP1": {"min": 1},
        },
        "attrs": {"SCALE": {"dtype": "int", "default": 3}},
        "tensors": {
            "x": {
                "role": "input",
                "dims": ["total_tokens", "H", "D"],
                "dtype": "T",
                "layout": "contiguous",
                "strides": ["H*D", "D", "1"],
                "space": "global",
                "access": "read",
            },
            "y": {
                "role": "output",
                "dims": ["total_tokens", "H", "D"],
                "dtype": "T",
                "layout": "contiguous",
                "strides": ["H*D", "D", "1"],
                "space": "global",
                "access": "write",
            },
            "cu_seqlens": {
                "role": "input",
                "dims": ["NSP1"],
                "dtype": "i32",
                "layout": "contiguous",
                "strides": ["1"],
                "space": "global",
                "access": "read",
            },
        },
        "sizes": {"main": tier},
        "spec_params": ["H", "D"],
        "numerics": {
            "acc_dtype": "fp32",
            "rounding": "single_round_at_output",
            "tolerances": {"fp16": {"rtol": 0.002, "atol_rel": 0.001}},
            "exact": False,
        },
        "cases": {"names": ["normal"], "inputs": {}},
        "work_accounting": {
            "flops": {"terms": [{"coefficient": 1, "symbols": ["total_tokens", "H", "D"]}]},
            "min_bytes": {"terms": [{"tensor": "x"}, {"tensor": "y"}]},
        },
        "discovery": {"phase": 1, "default_instance": {"dtype": "fp16", "size": "main"}},
    }


def _load(tmp_path: Path, document: dict, name: str = "varlen_probe"):
    """@brief 把文档写进标准目录并走公开 loader。

    @param tmp_path 临时目录。
    @param document v2 文档。
    @param name 目录名（等于 op_name）。
    @return 通过校验的 :class:`OpSpec`。
    """
    task = tmp_path / name
    task.mkdir(parents=True, exist_ok=True)
    (task / "opspec.json").write_text(json.dumps(document), encoding="utf-8")
    return load_opspec(task / "opspec.json")


def test_seq_lens_derive_the_runtime_symbols(tmp_path):
    """@brief ``seq_lens`` 派生 total_tokens/num_seqs/NSP1，张量照常按它们求值。"""
    spec = _load(tmp_path, _document([3, 5, 2]))
    instance = instantiate_v2(spec, "fp16", "main")

    assert instance.symbols["total_tokens"] == 10
    assert instance.symbols["num_seqs"] == 3
    assert instance.symbols["NSP1"] == 4
    assert instance.cu_seqlens == (0, 3, 8, 10)
    assert instance.tensor("x").shape == (10, 2, 4)
    assert instance.tensor("cu_seqlens").shape == (4,)


def test_a_fixed_tier_is_the_degenerate_single_segment_case(tmp_path):
    """@brief 固定 shape 档没有 cu_seqlens，仍是单段全长。"""
    spec = _load(tmp_path, _document(None))
    instance = instantiate_v2(spec, "fp16", "main")

    assert instance.cu_seqlens is None
    assert instance.symbols["total_tokens"] == 8
    assert instance.symbols["num_seqs"] == 1
    assert instance.tensor("x").shape == (8, 2, 4)


def test_a_uniform_distribution_agrees_with_the_fixed_total(tmp_path):
    """@brief 等长分段与固定档的 token 总数一致时，只有分段边界不同。"""
    uniform = instantiate_v2(_load(tmp_path / "u", _document([4, 4])), "fp16", "main")
    fixed = instantiate_v2(_load(tmp_path / "f", _document(None)), "fp16", "main")

    assert uniform.symbols["total_tokens"] == fixed.symbols["total_tokens"]
    assert uniform.tensor("x").shape == fixed.tensor("x").shape
    assert uniform.cu_seqlens == (0, 4, 8)
    assert fixed.cu_seqlens is None


def test_the_segment_boundaries_are_part_of_the_instance_identity(tmp_path):
    """@brief 同 token 总数、不同分段 = 不同问题，指纹必须不同。"""
    left = instantiate_v2(_load(tmp_path / "a", _document([4, 4, 2])), "fp16", "main")
    right = instantiate_v2(_load(tmp_path / "b", _document([2, 4, 4])), "fp16", "main")

    assert left.symbols == right.symbols
    assert left.cu_seqlens != right.cu_seqlens
    assert fingerprint(left.fingerprint_document()) != fingerprint(right.fingerprint_document())


def test_the_derived_symbols_are_carried_by_the_fingerprint(tmp_path):
    """@brief 派生符号与分段边界都进冻结投影，证据身份可复核。"""
    instance = instantiate_v2(_load(tmp_path, _document([3, 5, 2])), "fp16", "main")
    document = instance.fingerprint_document()

    assert document["symbols"]["total_tokens"] == 10
    assert document["cu_seqlens"] == [0, 3, 8, 10]
    assert [tensor["shape"] for tensor in document["tensors"] if tensor["name"] == "cu_seqlens"] == [[4]]


@pytest.mark.parametrize(
    "seq_lens",
    [pytest.param([], id="empty"), pytest.param([0], id="zero"), pytest.param([-1, 4], id="negative"),
     pytest.param([4, 4.0], id="float"), pytest.param([4, True], id="bool")],
)
def test_invalid_seq_lens_are_refused(tmp_path, seq_lens):
    """@brief 非法分布（空/零/负/非整数）在结构校验阶段就拒绝。"""
    with pytest.raises(OpSpecError, match="seq_lens"):
        _load(tmp_path, _document(seq_lens))


def test_a_varlen_tier_may_not_also_declare_the_derived_symbols(tmp_path):
    """@brief 派生符号由 seq_lens 算出，档里再写一遍是自相矛盾的重复声明。"""
    document = _document([3, 5])
    document["sizes"]["main"]["total_tokens"] = 999
    with pytest.raises(OpSpecError, match="derives .*total_tokens"):
        _load(tmp_path, document)


def test_seq_lens_is_a_tier_declaration_and_not_a_symbol(tmp_path):
    """@brief 分布本身不上符号表：实例符号只有张量真用到的维度。"""
    spec = _load(tmp_path, _document([3, 5, 2]))
    instance = instantiate_v2(spec, "fp16", "main")

    assert "seq_lens" not in instance.symbols
    assert set(instance.symbols) == {"H", "D", "total_tokens", "num_seqs", "NSP1"}
    assert spec.sizes["main"]["seq_lens"] == [3, 5, 2]


def test_a_copy_of_the_document_does_not_alias_the_instance(tmp_path):
    """@brief 文档可变，实例的分段边界不可变。"""
    document = _document([3, 5, 2])
    spec = _load(tmp_path, document)
    instance = instantiate_v2(spec, "fp16", "main")
    original = copy.deepcopy(document)
    document["sizes"]["main"]["seq_lens"].append(99)

    assert instance.cu_seqlens == (0, 3, 8, 10)
    assert document != original
