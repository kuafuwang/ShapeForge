"""Harness-owned runtime ``cu_seqlens`` contract for L1 (packed varlen) instances.

CPU only, backend free.  A varlen tier fixes ``total_tokens``/``num_seqs`` but not the
segmentation, so correctness must accept every legal boundary set and refuse every illegal one
*before* any host/device work.  The three layers this file pins:

* the **static** declaration (``sizes.<tier>.seq_lens``) requires every segment >= 1 — pinned by
  ``tests/shared/test_opspec_varlen.py``;
* the **runtime** tensor may contain an empty sequence, but the batch must carry >= 1 token, the
  offsets must be ``i32``, start at 0, be non-decreasing and end at ``total_tokens``;
* the check is the **harness's**, and it is the same funnel for ``generate`` and ``execute``: a
  task's ``gen_inputs`` cannot certify its own boundaries, and a replayed/hand-built input set
  cannot bypass the check.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import pytest
import torch

from tcenv.contract_errors import TaskContractError, VarlenContractError
from tcenv.opspec.v2 import VARLEN_CASE_DOMAIN, instantiate_v2, opspec_v2_from_document
from tcenv.portfolio.domain import evaluation_input_digest, semantic_instance_digest
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.varlen import (
    INT32_MAX,
    is_varlen,
    require_runtime_segments,
    runtime_segment_problems,
    segment_lengths,
)

#: The sparse COO fixture trips torch's invariant-check advisory; it is a fixture detail, not a
#: harness warning.  Filtering it here keeps the suite output about the contract under test.
pytestmark = pytest.mark.filterwarnings("ignore:Sparse invariant checks")

#: Canonical distribution of the synthetic probe task (total_tokens = 10, num_seqs = 3).
SEQ_LENS = (3, 5, 2)
TOTAL_TOKENS = sum(SEQ_LENS)


def _document(seq_lens: tuple[int, ...] | None = SEQ_LENS) -> dict:
    """@brief 一份最小的 varlen v2 文档（``seq_lens=None`` 时是固定 shape 档）。

    @param seq_lens 分段长度分布（``None`` 表示固定 shape 档）。
    @return v2 文档。
    """
    tier: dict = {"H": 2, "D": 4, "usage": ["golden"]}
    if seq_lens is not None:
        tier["seq_lens"] = list(seq_lens)
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
        "cases": {"names": list(VARLEN_CASE_DOMAIN), "inputs": {}},
        "work_accounting": {
            "flops": {"terms": [{"coefficient": 1, "symbols": ["total_tokens", "H", "D"]}]},
            "min_bytes": {"terms": [{"tensor": "x"}, {"tensor": "y"}]},
        },
        "discovery": {"phase": 1, "default_instance": {"dtype": "fp16", "size": "main"}},
    }


#: A reference whose ``gen_inputs`` builds the boundaries from the instance's canonical profile.
GOOD_REFERENCE = '''"""y = SCALE * x over the packed batch."""

import torch


def reference(inputs, attrs):
    x = inputs["x"].to(torch.float64)
    scale = float(attrs["SCALE"])
    cu = [int(value) for value in inputs["cu_seqlens"].tolist()]
    y = torch.zeros_like(x)
    for seq in range(len(cu) - 1):
        y[cu[seq] : cu[seq + 1]] = scale * x[cu[seq] : cu[seq + 1]]
    return {"y": y}


def gen_inputs(case, seed, instance):
    generator = torch.Generator().manual_seed(int(seed))
    total_tokens = int(instance.symbols["total_tokens"])
    heads = int(instance.symbols["H"])
    dim = int(instance.symbols["D"])
    x = torch.randn((total_tokens, heads, dim), generator=generator)
    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)
    return {"x": x, "cu_seqlens": cu}
'''

#: The same task, but ``gen_inputs`` emits an illegal boundary set (last offset != total_tokens).
BAD_GENERATOR_REFERENCE = GOOD_REFERENCE.replace(
    '    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)',
    '    cu = torch.tensor([0, *list(instance.cu_seqlens)[1:-1], total_tokens + 1], dtype=torch.int32)',
)

#: ``gen_inputs`` 变体：每种都必须在 ``generate`` 漏斗上得到 ``VarlenContractError``。
BAD_GENERATORS = {
    "wrong-length": GOOD_REFERENCE.replace(
        "    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)",
        "    cu = torch.tensor([0, total_tokens], dtype=torch.int32)",
    ),
    "wrong-rank": GOOD_REFERENCE.replace(
        "    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)",
        "    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32).unsqueeze(-1)",
    ),
    "nan-offset": GOOD_REFERENCE.replace(
        "    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)",
        '    cu = torch.tensor([0.0, float("nan"), 8.0, 10.0], dtype=torch.float32)',
    ),
    "inf-offset": GOOD_REFERENCE.replace(
        "    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)",
        '    cu = torch.tensor([0.0, float("inf"), 8.0, 10.0], dtype=torch.float32)',
    ),
    "sparse-coo": GOOD_REFERENCE.replace(
        "    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)",
        "    cu = torch.sparse_coo_tensor(torch.tensor([[0, 1, 2, 3]]), "
        "torch.tensor([0, 3, 8, 10], dtype=torch.int32), (4,), dtype=torch.int32)",
    ),
    "meta": GOOD_REFERENCE.replace(
        "    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)",
        '    cu = torch.empty(4, dtype=torch.int32, device="meta")',
    ),
}


def _unreadable_boundary(layout: str) -> torch.Tensor:
    """@brief 构造 shape/dtype 合法但内容不可读的边界张量（sparse COO / meta）。

    @param layout ``sparse-coo`` 或 ``meta``。
    @return 边界张量。
    """
    if layout == "sparse-coo":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.sparse_coo_tensor(
                torch.tensor([[0, 1, 2, 3]]),
                torch.tensor([0, 3, 8, 10], dtype=torch.int32),
                (4,),
                dtype=torch.int32,
            )
    return torch.empty(4, dtype=torch.int32, device="meta")


def _task(tmp_path: Path, reference_source: str, *, document: dict | None = None):
    """@brief 把一份 varlen 探针题写进 tmp 目录并装载。

    @param tmp_path 临时目录。
    @param reference_source reference.py 源文本。
    @param document 可选的 v2 文档（缺省用 :func:`_document`）。
    @return ``TaskContract``。
    """
    task = tmp_path / "varlen_probe"
    task.mkdir(parents=True, exist_ok=True)
    (task / "opspec.json").write_text(json.dumps(document or _document()), encoding="utf-8")
    (task / "reference.py").write_text(reference_source, encoding="utf-8")
    return load_task_contract(task, dtype="fp16", size="main")


def _oracle(contract):
    """@brief 探针题的 reference oracle（合同自带的那一份）。

    @param contract 任务合同。
    @return :class:`ReferenceModule`。
    """
    return contract.reference


def test_the_probe_instance_is_a_varlen_instance_with_its_own_boundaries():
    """@brief 探针题确实是 L1 实例：边界与派生符号一致。"""
    instance = instantiate_v2(opspec_v2_from_document(_document(), source="<varlen_probe>"), "fp16", "main")
    assert is_varlen(instance)
    assert instance.cu_seqlens == (0, 3, 8, 10)
    assert instance.symbols["total_tokens"] == TOTAL_TOKENS
    assert instance.symbols["num_seqs"] == len(SEQ_LENS)


def test_segment_lengths_reads_tensors_and_sequences(tmp_path):
    """@brief 边界读取同时支持 torch 张量、list 与 numpy 风格数组。"""
    values = [0, 3, 8, 10]
    assert segment_lengths(torch.tensor(values, dtype=torch.int32)) == (3, 5, 2)
    assert segment_lengths(values) == (3, 5, 2)
    with pytest.raises(VarlenContractError):
        segment_lengths(object())


def test_a_fixed_shape_instance_has_no_runtime_boundary_contract():
    """@brief 固定 shape 实例不进入 varlen 检查（它的输入里没有边界张量）。"""
    from tcenv.opspec import load_opspec
    from tcenv.taskcontract.locator import REPO_ROOT

    instance = instantiate_v2(load_opspec(REPO_ROOT / "tasks" / "gemm" / "opspec.json"), "fp16", "paper")
    assert not is_varlen(instance)
    assert runtime_segment_problems(instance, {}) == ()


def test_the_harness_accepts_a_legal_boundary_set_including_an_empty_sequence(tmp_path):
    """@brief 合法分段（含空序列）通过：空段是运行时事实，整批仍 >= 1 token。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    instance = contract.instance
    inputs = _oracle(contract).generate(instance, "canonical", 7)

    assert runtime_segment_problems(instance, inputs) == ()

    legal = dict(inputs)
    legal["cu_seqlens"] = torch.tensor([0, 0, 5, 10], dtype=torch.int32)
    assert runtime_segment_problems(instance, legal) == ()
    assert segment_lengths(legal["cu_seqlens"]) == (0, 5, 5)


@pytest.mark.parametrize(
    "where,values",
    [
        ("first offset is not zero", [1, 3, 8, 10]),
        ("last offset is not total_tokens", [0, 3, 8, 11]),
        ("offsets decrease", [0, 5, 3, 10]),
        ("an offset is negative", [0, -1, 3, 10]),
        ("an offset exceeds int32", [0, 3, 8, INT32_MAX + 1]),
    ],
)
def test_an_illegal_boundary_set_is_a_typed_refusal(tmp_path, where, values):
    """@brief 端点/单调/i32 非法都是 ``varlen_contract_invalid``，不是静默接受。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    inputs = dict(_oracle(contract).generate(contract.instance, "canonical", 7))
    inputs["cu_seqlens"] = torch.tensor(values, dtype=torch.int64 if values[-1] > INT32_MAX else torch.int32)

    problems = runtime_segment_problems(contract.instance, inputs)
    assert problems, where
    with pytest.raises(VarlenContractError) as caught:
        require_runtime_segments(contract.instance, inputs, where="test")
    assert caught.value.code == "varlen_contract_invalid"
    assert "cu_seqlens" in str(caught.value)


def test_a_wrong_length_or_dtype_boundary_tensor_is_refused(tmp_path):
    """@brief 长度必须 ``num_seqs + 1``，dtype 必须是 i32。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    good = dict(_oracle(contract).generate(contract.instance, "canonical", 7))

    short = dict(good)
    short["cu_seqlens"] = torch.tensor([0, 10], dtype=torch.int32)
    assert runtime_segment_problems(contract.instance, short)

    floating = dict(good)
    floating["cu_seqlens"] = torch.tensor([0, 3, 8, 10], dtype=torch.float32)
    assert any("i32" in problem for problem in runtime_segment_problems(contract.instance, floating))

    # 非整数偏移的浮点张量即使数值恰好是整数，也必须按 ABI 拒绝。
    with pytest.raises(VarlenContractError):
        require_runtime_segments(contract.instance, floating, where="test")


@pytest.mark.parametrize(
    "where,cu",
    [
        ("wrong length", torch.tensor([0, 10], dtype=torch.int32)),
        ("wrong rank", torch.tensor([[0], [3], [8], [10]], dtype=torch.int32)),
        ("not i32", torch.tensor([0.0, 3.0, 8.0, 10.0], dtype=torch.float32)),
        ("nan offset", torch.tensor([0.0, float("nan"), 8.0, 10.0], dtype=torch.float32)),
        ("inf offset", torch.tensor([0.0, float("inf"), 8.0, 10.0], dtype=torch.float32)),
    ],
)
def test_the_execute_funnel_refuses_bad_boundaries_as_a_typed_error(tmp_path, where, cu):
    """@brief ``execute`` 对长度/rank/dtype/NaN/Inf 都抛 ``varlen_contract_invalid``，不是裸异常。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    oracle = contract.reference
    good = dict(oracle.generate(contract.instance, "canonical", 7))
    bad = dict(good)
    bad["cu_seqlens"] = cu

    problems = runtime_segment_problems(contract.instance, bad)
    assert problems, where
    with pytest.raises(VarlenContractError) as caught:
        oracle.execute(contract.instance, bad)
    assert caught.value.code == "varlen_contract_invalid"


def test_the_execute_funnel_refuses_a_missing_boundary_as_a_typed_error(tmp_path):
    """@brief ``execute`` 缺 cu 时也是 typed refusal（不是 SemanticUnresolved/KeyError）。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    oracle = contract.reference
    good = dict(oracle.generate(contract.instance, "canonical", 7))
    del good["cu_seqlens"]
    with pytest.raises(VarlenContractError) as caught:
        oracle.execute(contract.instance, good)
    assert caught.value.code == "varlen_contract_invalid"


@pytest.mark.parametrize("variant", sorted(BAD_GENERATORS), ids=str)
def test_the_generate_funnel_refuses_bad_boundaries_as_a_typed_error(tmp_path, variant):
    """@brief ``generate`` 对长度/rank/dtype/NaN/Inf 都抛 ``varlen_contract_invalid``。"""
    contract = _task(tmp_path, BAD_GENERATORS[variant])
    with pytest.raises(VarlenContractError) as caught:
        contract.reference.generate(contract.instance, "canonical", 7)
    assert caught.value.code == "varlen_contract_invalid"
    assert "gen_inputs" in str(caught.value)


@pytest.mark.parametrize("layout", ["sparse-coo", "meta"])
def test_an_unreadable_boundary_layout_is_a_typed_refusal(tmp_path, layout):
    """@brief sparse COO / meta 等无存储布局的 cu 不能抛裸异常：helper 与两个入口都 typed refusal。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    oracle = contract.reference
    good = dict(oracle.generate(contract.instance, "canonical", 7))
    bad = dict(good)
    bad["cu_seqlens"] = _unreadable_boundary(layout)

    problems = runtime_segment_problems(contract.instance, bad)
    assert problems and any("not readable" in problem for problem in problems), problems
    with pytest.raises(VarlenContractError) as caught:
        require_runtime_segments(contract.instance, bad, where="test")
    assert caught.value.code == "varlen_contract_invalid"
    with pytest.raises(VarlenContractError) as caught:
        oracle.execute(contract.instance, bad)
    assert caught.value.code == "varlen_contract_invalid"


@pytest.mark.parametrize("variant", ["sparse-coo", "meta"])
def test_the_generate_funnel_refuses_an_unreadable_boundary_layout(tmp_path, variant):
    """@brief ``gen_inputs`` 产出 sparse/meta 边界时，``generate`` 也只给 typed refusal。"""
    contract = _task(tmp_path, BAD_GENERATORS[variant])
    assert contract.instance is not None
    with pytest.raises(VarlenContractError) as caught:
        contract.reference.generate(contract.instance, "canonical", 7)
    assert caught.value.code == "varlen_contract_invalid"


def test_a_zero_token_batch_is_refused_at_runtime(tmp_path):
    """@brief 运行时 ``total_tokens == 0``（空批次）被拒绝；空序列不等同于空批次。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    instance = contract.instance
    good = dict(contract.reference.generate(instance, "canonical", 7))
    zero = dataclasses.replace(instance, symbols={**dict(instance.symbols), "total_tokens": 0})

    problems = runtime_segment_problems(zero, good)
    assert any("total_tokens must be >= 1" in problem for problem in problems), problems
    with pytest.raises(VarlenContractError) as caught:
        require_runtime_segments(zero, good, where="test")
    assert caught.value.code == "varlen_contract_invalid"
    with pytest.raises(VarlenContractError):
        contract.reference.execute(zero, good)


def test_a_missing_boundary_tensor_is_refused(tmp_path):
    """@brief 缺 ``cu_seqlens`` 时给出明确 refusal（不是 KeyError/None 崩溃）。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    problems = runtime_segment_problems(contract.instance, {})
    assert problems and "missing" in problems[0]


def test_the_task_generator_cannot_certify_its_own_boundaries(tmp_path):
    """@brief ``gen_inputs`` 自己产出的非法边界由 harness 漏斗拒绝（同一 generate 路径）。"""
    contract = _task(tmp_path, BAD_GENERATOR_REFERENCE)
    with pytest.raises(VarlenContractError) as caught:
        _oracle(contract).generate(contract.instance, "canonical", 7)
    assert caught.value.code == "varlen_contract_invalid"
    assert "gen_inputs" in str(caught.value)


def test_the_execute_funnel_refuses_a_replayed_illegal_boundary_set(tmp_path):
    """@brief 重放/手搭的输入也走同一漏斗：oracle 不会读到非法分段。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    oracle = _oracle(contract)
    inputs = dict(oracle.generate(contract.instance, "canonical", 7))
    inputs["cu_seqlens"] = torch.tensor([0, 3, 8, 9], dtype=torch.int32)

    with pytest.raises(VarlenContractError) as caught:
        oracle.execute(contract.instance, inputs)
    assert caught.value.code == "varlen_contract_invalid"
    assert "inputs" in str(caught.value)


def test_the_static_profile_still_refuses_a_zero_length_segment(tmp_path):
    """@brief 静态 ``seq_lens`` 每段 >= 1；零长度只能由运行时 cu 构造。"""
    from tcenv.opspec import OpSpecError

    document = _document((3, 0, 7))
    task = tmp_path / "varlen_probe"
    task.mkdir(parents=True, exist_ok=True)
    (task / "opspec.json").write_text(json.dumps(document), encoding="utf-8")
    (task / "reference.py").write_text(GOOD_REFERENCE, encoding="utf-8")

    with pytest.raises(OpSpecError, match="positive integer"):
        load_task_contract(task, dtype="fp16", size="main")


def test_the_evaluation_input_digest_is_replayable_and_case_scoped(tmp_path):
    """@brief 实际输入身份可重放，且按 case/seed 隔离（不是语义实例身份）。"""
    contract = _task(tmp_path, GOOD_REFERENCE)
    instance = contract.instance
    oracle = contract.reference
    first = oracle.generate(instance, "canonical", 5)
    digest = evaluation_input_digest(instance, "canonical", 5, first)

    # 重放：同一 (instance, case, seed) 重新生成得到同一身份。
    assert evaluation_input_digest(instance, "canonical", 5, oracle.generate(instance, "canonical", 5)) == digest
    # 不同 seed / 不同 case 是不同的输入身份。
    assert evaluation_input_digest(instance, "canonical", 6, first) != digest
    alternative = oracle.generate(instance, "permuted", 5)
    assert evaluation_input_digest(instance, "permuted", 5, alternative) != digest
    # 三层身份互不等价：实际输入身份不是语义实例身份。
    assert digest != semantic_instance_digest(instance)

    with pytest.raises(TaskContractError, match="not declared"):
        evaluation_input_digest(instance, "missing", 5, first)
    with pytest.raises(TaskContractError, match="do not match"):
        evaluation_input_digest(instance, "canonical", 5, {"x": first["x"]})


def test_the_input_digest_binds_the_actual_boundary_bytes_not_the_canonical_copy(tmp_path):
    """@brief 同一语义实例下换合法分段：语义身份不变，实际输入身份改变。

    permuted/skewed/fresh 是 evaluation case，不是新的 domain point；它们由实际 cu 字节区分，
    因此这里不需要（也不允许）伪造一个新的实例身份。
    """
    contract = _task(tmp_path, GOOD_REFERENCE)
    instance = contract.instance
    canonical = dict(contract.reference.generate(instance, "canonical", 5))
    permuted = dict(canonical)
    permuted["cu_seqlens"] = torch.tensor([0, 5, 8, 10], dtype=torch.int32)

    left = evaluation_input_digest(instance, "canonical", 5, canonical)
    right = evaluation_input_digest(instance, "canonical", 5, permuted)
    assert left != right
    # 两份输入属于同一个 domain point：语义实例身份（含 canonical cu）没有变。
    assert semantic_instance_digest(instance) == semantic_instance_digest(
        instantiate_v2(opspec_v2_from_document(_document(), source="<varlen_probe>"), "fp16", "main")
    )
