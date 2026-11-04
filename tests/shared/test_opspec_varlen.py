"""varlen (packed sequence) instances: ``seq_lens`` derivation, identity and its negatives.

CPU only, backend free.  A varlen tier declares ``seq_lens`` and instantiates exactly like a fixed
tier: ``total_tokens``/``num_seqs``/``NSP1`` are derived from the distribution, tensors keep their
declared dims/strides, and the fixed tier stays the degenerate single-segment case.  The segment
boundaries are part of the instance identity, so two distributions with the same token total are
different problems.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from tcenv.identity import fingerprint
from tcenv.opspec import VARLEN_CASE_DOMAIN, OpSpecError, instantiate_v2, load_opspec
from tcenv.portfolio.domain import semantic_instance_digest
from tcenv.taskcontract.locator import REPO_ROOT

#: 归档的 v1 仓库树（v1 合同样本，本测试不修改它，只拿它的文档做反例）。
V1_ARCHIVE = REPO_ROOT / "tests" / "support" / "golden" / "shared" / "v1-track"
V1_DOCUMENT = V1_ARCHIVE / "tasks" / "sem_attention_odd_scale" / "opspec.json"


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
        # A varlen document must declare the L1 correctness-case floor; the fixed variant does not.
        "cases": {"names": list(VARLEN_CASE_DOMAIN) if seq_lens is not None else ["normal"], "inputs": {}},
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


def test_a_fixed_instance_keeps_the_pre_varlen_projection(tmp_path):
    """@brief 固定 shape 实例的冻结投影不带 varlen 键，已有证据身份逐位不变。"""
    fixed = instantiate_v2(_load(tmp_path, _document(None)), "fp16", "main")
    varlen = instantiate_v2(_load(tmp_path / "v", _document([8])), "fp16", "main")

    assert "cu_seqlens" not in fixed.fingerprint_document()
    assert varlen.fingerprint_document()["cu_seqlens"] == [0, 8]


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


def test_a_fixed_tier_missing_a_symbol_is_refused_at_validation_time(tmp_path):
    """@brief 固定档缺符号必须在校验期报 ``OpSpecError``，不能延后成实例化的 ``KeyError``。

    派生符号豁免只属于合法的 v2 varlen 档；固定档没有这个豁免。
    """
    for symbol in ("total_tokens", "num_seqs", "NSP1"):
        document = _document(None)
        del document["sizes"]["main"][symbol]
        with pytest.raises(OpSpecError, match="missing symbols"):
            _load(tmp_path / symbol, document)


def test_a_varlen_task_must_declare_the_cu_seqlens_input(tmp_path):
    """@brief varlen 题没有 ``cu_seqlens`` 输入就是 ABI 缺陷，结构校验期拒绝。"""
    document = _document([3, 5])
    del document["tensors"]["cu_seqlens"]
    with pytest.raises(OpSpecError, match="cu_seqlens"):
        _load(tmp_path, document)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        pytest.param({"dtype": "fp32"}, "dtype", id="fp32"),
        pytest.param({"dtype": "u32"}, "dtype", id="u32"),
        pytest.param({"dims": ["H"], "strides": ["1"]}, "dims", id="wrong-dim"),
        pytest.param({"strides": ["2"]}, "strides", id="wrong-stride"),
        pytest.param({"access": "read_write"}, "access", id="read-write"),
        pytest.param({"space": "workspace"}, "space", id="workspace"),
        pytest.param({"role": "output", "access": "write"}, "role", id="output"),
    ],
)
def test_an_illegal_varlen_boundary_abi_is_refused(tmp_path, mutation, reason):
    """@brief ``cu_seqlens`` 必须是 ``[NSP1]`` 的 i32 只读 input；其他形态都是 typed refusal。"""
    document = _document([3, 5, 2])
    document["tensors"]["cu_seqlens"].update(mutation)
    with pytest.raises(OpSpecError, match="illegal varlen ABI") as caught:
        _load(tmp_path / reason, document)
    assert reason in str(caught.value)


def test_mixing_fixed_and_varlen_tiers_is_refused(tmp_path):
    """@brief 一题里既有 fixed 档又有 varlen 档会需要两套 ABI，因此明确拒绝而不是按实例猜。"""
    document = _document([3, 5])
    document["sizes"]["fixed"] = {
        "H": 2,
        "D": 4,
        "total_tokens": 8,
        "num_seqs": 1,
        "NSP1": 2,
        "usage": ["golden"],
    }
    with pytest.raises(OpSpecError, match="entirely varlen"):
        _load(tmp_path, document)


def test_a_varlen_total_that_overflows_the_int32_abi_is_refused(tmp_path):
    """@brief 边界 ABI 是 i32：静态分布的总 token 数超出 i32 就在校验期拒绝。"""
    document = _document([2**31 - 1, 1])
    with pytest.raises(OpSpecError, match="i32 range"):
        _load(tmp_path, document)


def test_a_varlen_tier_must_satisfy_the_declared_symbol_minimum(tmp_path):
    """@brief 派生符号也要满足 symbols 里声明的 ``min``。"""
    document = _document([4, 4])
    document["symbols"]["num_seqs"]["min"] = 3
    with pytest.raises(OpSpecError, match="below the declared min"):
        _load(tmp_path, document)


def test_a_varlen_task_must_declare_the_l1_case_domain(tmp_path):
    """@brief varlen 题必须覆盖 §5.1 的 correctness case，不能只声明 canonical 就宣称支持任意分段。"""
    document = _document([3, 5, 2])
    document["cases"]["names"] = ["canonical", "fresh_distribution"]
    with pytest.raises(OpSpecError, match="must cover the L1 correctness cases") as caught:
        _load(tmp_path, document)
    for missing in ("permuted", "zero_length", "max_handoff_stress", "single_sequence"):
        assert missing in str(caught.value)


def test_a_varlen_task_must_put_canonical_first(tmp_path):
    """@brief 运行时用第一个 case 做 benchmark/validate：canonical 必须是它。"""
    document = _document([3, 5, 2])
    document["cases"]["names"] = list(VARLEN_CASE_DOMAIN[1:]) + ["canonical"]
    with pytest.raises(OpSpecError, match="must declare 'canonical' first"):
        _load(tmp_path, document)


def test_a_varlen_task_may_declare_extra_cases_after_the_floor(tmp_path):
    """@brief 下限是“至少覆盖”，题目可以追加自己的 case；追加不改变 L1 语义。"""
    document = _document([3, 5, 2])
    document["cases"]["names"] = list(VARLEN_CASE_DOMAIN) + ["extra_probe"]
    spec = _load(tmp_path, document)
    assert spec.digest
    instance = instantiate_v2(spec, "fp16", "main")
    assert tuple(instance.case_domain.cases)[-1] == "extra_probe"


def test_a_fixed_shape_task_keeps_its_own_case_names(tmp_path):
    """@brief L1 case 下限只约束 varlen 档；固定 shape 题不受影响。"""
    spec = _load(tmp_path, _document(None))
    assert tuple(instantiate_v2(spec, "fp16", "main").case_domain.cases) == ("normal",)


def test_a_v1_archive_refuses_seq_lens(tmp_path):
    """@brief ``_validate_size`` 由 v1/v2 共用，但 ``seq_lens`` 只能是 v2 档：v1 必须拒绝。"""
    document = json.loads(V1_DOCUMENT.read_text(encoding="utf-8"))
    assert document["spec_version"] == 1
    # 原样归档仍然可加载（拒绝只针对 seq_lens，不是归档本身不可用）。
    pristine = _load(tmp_path / "pristine", document, name=document["op_name"])
    assert pristine.spec_version == 1

    document["sizes"]["small"]["seq_lens"] = [128]
    with pytest.raises(OpSpecError, match="seq_lens") as caught:
        _load(tmp_path / "varlen", document, name=document["op_name"])
    assert "v1 document declares fixed-shape tiers only" in str(caught.value)


def test_the_canonical_boundaries_are_the_only_difference_inside_one_opspec(tmp_path):
    """@brief 同一 ``opspec_digest`` 下只改 cu 字段：冻结投影只差这一个键，身份随即不同。

    这是隔离性测试：它不靠“两份不同的 opspec 文档”间接证明，而是直接在同一实例上换分段。
    """
    instance = instantiate_v2(_load(tmp_path, _document([3, 5, 2])), "fp16", "main")
    reordered = dataclasses.replace(instance, cu_seqlens=(0, 5, 8, 10))
    left = instance.fingerprint_document()
    right = reordered.fingerprint_document()

    assert left["opspec_digest"] == right["opspec_digest"]
    assert set(left) == set(right)
    assert {key for key in left if left[key] != right[key]} == {"cu_seqlens"}
    assert fingerprint(left) != fingerprint(right)
    assert semantic_instance_digest(instance) != semantic_instance_digest(reordered)
    assert instance.symbols == reordered.symbols


#: 加 varlen 之前就已冻结的固定 shape 身份（opspec_digest, 实例投影, semantic_instance_digest）。
#: V0/V1 只允许 varlen 实例新增键：这些固定 shape 值必须逐位不变，变了就是身份回归。
FROZEN_FIXED_IDENTITY = {
    ("gemm", "fp16", "paper"): (
        "d745f02cb981f467c8de0f3c43351b7f101fe616ba9fa64aca16d50147f84056",
        "621646d9fdd25719b1ea0c874507752416ef95bf4e8110b62449fbc2e1717e61",
        "842b071f3c5f733b6bcc506edbdd71e405036de51ed82b47a06b09b3e7309feb",
    ),
    ("rmsnorm", "fp16", "small"): (
        "280b0ab81fee284603b8e8e97e41d9c0f3b09c712ceb41cadb901185b6d321e3",
        "91c363187d7d5d9b7ecc634382ec790c241b96886655ce7648812399c1916680",
        "9009a1aad92d39d5ddbb422aa1fa43ce88d67ca503614e5926df9bbf998a458a",
    ),
    ("flash_attention", "fp16", "paper"): (
        "27a3c274ac82fe22b1005980a95c6f1e6cb8c55f1e266cafd9585f85baddf5ad",
        "f3ea6a2916bec3b53f916dbddd6aa2895b405e4d881a2a1ac789a8ecad627cff",
        "9ee414dcb54fe1fba67ae2d1e308eee0ea42560293fe709544261a50b1e9ea58",
    ),
}


@pytest.mark.parametrize("combo", sorted(FROZEN_FIXED_IDENTITY), ids=lambda value: "/".join(value))
def test_a_fixed_shape_instance_keeps_its_frozen_identity(combo):
    """@brief 固定 shape 题的 opspec/实例/语义身份逐位不变（varlen 不污染旧证据）。"""
    op, dtype, size = combo
    spec = load_opspec(REPO_ROOT / "tasks" / op / "opspec.json")
    instance = instantiate_v2(spec, dtype, size)
    assert instance.cu_seqlens is None
    assert "cu_seqlens" not in instance.fingerprint_document()
    assert spec.digest == FROZEN_FIXED_IDENTITY[combo][0]
    assert fingerprint(instance.fingerprint_document()) == FROZEN_FIXED_IDENTITY[combo][1]
    assert semantic_instance_digest(instance) == FROZEN_FIXED_IDENTITY[combo][2]
