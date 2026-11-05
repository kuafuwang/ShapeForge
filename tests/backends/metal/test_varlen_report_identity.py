"""varlen 评测报告里的实际输入身份（CPU；只跑 reference/投影，不编译、不启动 kernel）。

L1 的 correctness 必须能回答“这次到底跑了哪组分段”。`prepare_v2_case` 是 Metal v2 runtime 里
把 reference 输入变成设备初值和比较参考的那一步，所以它的 `input_digest` 必须在每道 varlen 题、
每个 case 上都等于 `portfolio.domain.evaluation_input_digest`（case + seed + 实际 cu 字节 + 输入
张量身份），并最终进入 ``CaseResult.input_sha256``。固定 shape 实例保持空串：它们的输入身份已经
由实例指纹和既有证据字段表达，不新增键。
"""

from __future__ import annotations

import math

import pytest

from tcenv.backends.metal.v2_runtime import _check_v2_case, prepare_v2_case
from tcenv.dtypes import get_dtype
from tcenv.evaluate import case_coverage
from tcenv.opspec.v2 import VARLEN_CASE_DOMAIN
from tcenv.portfolio.domain import evaluation_input_digest
from tcenv.target import load_target
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.taskcontract.varlen import segment_lengths
from tcenv.taskcontract.varlen_cases import case_realization
from tcenv.tensors import to_raw
from tcenv.track.abi import derive_tensor_memory_contract
from tests.support.varlen_probe import write_probe_task

#: ``op_name -> (task directory, declared sizes)``；只列 varlen 题。
TASKS = {
    "varlen_kda": (REPO_ROOT / "tasks" / "varlen_kda", ("fixed", "mixed", "uniform")),
    "varlen_scale": (REPO_ROOT / "tasks" / "varlen_scale", ("mixed", "uniform", "single")),
}
COMBOS = tuple((op, size) for op, (_, sizes) in TASKS.items() for size in sizes)
SEED = 7


@pytest.mark.parametrize("op,size", COMBOS, ids=lambda value: str(value))
@pytest.mark.parametrize("case", VARLEN_CASE_DOMAIN)
def test_every_case_reports_the_digest_of_the_inputs_it_actually_prepared(op, size, case):
    """@brief ``prepare_v2_case`` 记录的输入身份与实际生成的 cu 字节一致（失败也能重放）。"""
    directory, _ = TASKS[op]
    contract = load_task_contract(directory, dtype="fp16", size=size)
    data = prepare_v2_case(contract, case, SEED)

    expected = evaluation_input_digest(contract.instance, case, SEED, data.rounded)
    assert data.input_digest == expected
    assert len(data.input_digest) == 64
    # 同一 (instance, case, seed) 重放得到同一身份；换 case 得到不同身份。
    assert prepare_v2_case(contract, case, SEED).input_digest == data.input_digest


def test_a_fixed_shape_instance_keeps_the_pre_varlen_report_shape():
    """@brief 固定 shape 实例不新增输入身份键（空串），varlen 之外证据形状不变。"""
    contract = load_task_contract(REPO_ROOT / "tasks" / "lion", dtype="fp16", size="small")
    assert contract.instance.cu_seqlens is None
    data = prepare_v2_case(contract, "tiny", SEED)
    assert data.input_digest == ""


def _result(contract, case: str):
    """@brief 用 CPU 参考值当候选输出，跑一次 v2 的 case 比较（不编译、不启动 kernel）。

    @param contract 任务合同。
    @param case 用例名。
    @return ``CaseResult``。
    """
    data = prepare_v2_case(contract, case, SEED)
    abi = derive_tensor_memory_contract(contract.instance, target=load_target("apple-m4-max"))
    out_raw = {
        tensor.name: to_raw(data.ref[tensor.name], tensor.dtype)
        for tensor in abi.tensors
        if tensor.space != "workspace" and tensor.access in ("write", "read_write")
    }
    return _check_v2_case(contract, abi, data, out_raw, case)


@pytest.mark.parametrize("op,size", COMBOS, ids=lambda value: str(value))
@pytest.mark.parametrize("case", VARLEN_CASE_DOMAIN)
def test_the_correctness_result_carries_the_case_realization(op, size, case):
    """@brief 报告里的每个 case 都带 realization（由实际分段独立算出），而不是只有成败。"""
    directory, _ = TASKS[op]
    contract = load_task_contract(directory, dtype="fp16", size=size)
    result = _result(contract, case)
    expected = case_realization(
        case,
        segment_lengths(prepare_v2_case(contract, case, SEED).rounded["cu_seqlens"]),
        tuple(int(value) for value in contract.instance.canonical_seq_lens),
    )

    assert result.ok is True, result.message
    assert result.case_realization == expected
    assert result.to_json()["case_realization"] == expected


def test_the_aggregate_coverage_separates_partial_from_full_cases():
    """@brief 验收统计把 partial/not_constructible 单列，不把它们算成完整覆盖。"""
    multi = load_task_contract(TASKS["varlen_kda"][0], dtype="fp16", size="mixed")
    summary = case_coverage([_result(multi, case) for case in VARLEN_CASE_DOMAIN])
    assert summary["cases"] == len(VARLEN_CASE_DOMAIN)
    # KDA mixed：chunk_boundary 的槽位放不下整组 BT 目标 → partial；其余（含真正的 single_sequence）exact。
    assert summary["partial"] == ["chunk_boundary"]
    assert summary["not_constructible"] == []
    assert summary["deviates"] == []
    assert set(summary["exact"]) == set(VARLEN_CASE_DOMAIN) - {"chunk_boundary"}

    single = load_task_contract(TASKS["varlen_scale"][0], dtype="fp16", size="single")
    summary = case_coverage([_result(single, case) for case in VARLEN_CASE_DOMAIN])
    assert set(summary["exact"]) == {"canonical", "single_sequence"}
    assert set(summary["not_constructible"]) == set(VARLEN_CASE_DOMAIN) - {"canonical", "single_sequence"}


def test_a_fixed_shape_case_keeps_the_pre_varlen_report_shape():
    """@brief 固定 shape 实例没有 realization 键；聚合统计返回 None（旧证据形状不变）。"""
    contract = load_task_contract(REPO_ROOT / "tasks" / "lion", dtype="fp16", size="small")
    result = _result(contract, "tiny")

    assert result.case_realization is None
    assert "case_realization" not in result.to_json()
    assert case_coverage([result]) is None


@pytest.mark.parametrize("op", sorted(TASKS))
@pytest.mark.parametrize(
    "probe,lens,case,kind",
    [
        ("all-prime-impossible", (1, 2), "all_unaligned_primes", "not_constructible"),
        ("all-ones", (1, 1), "all_unaligned_primes", "not_constructible"),
        ("all-ones", (1, 1), "max_handoff_stress", "not_constructible"),
        ("all-twos", (2, 2), "skewed", "partial"),
        ("full-chunk", (5, 5, 5, 5, 5, 4), "chunk_boundary", "exact"),
        ("last-bt", (5, 5, 4), "chunk_boundary", "partial"),
        ("all-ones", (1, 1), "chunk_boundary", "partial"),
    ],
)
def test_a_probe_tier_reaches_the_report_with_its_realization(op, probe, lens, case, kind, tmp_path):
    """@brief 探针档上的退化/不可构造 case 带着真实 realization 进报告，不计完整覆盖。"""
    task = write_probe_task(TASKS[op][0], tmp_path / op, {probe: lens})
    contract = load_task_contract(task, dtype="fp16", size=probe)

    result = _result(contract, case)
    assert result.ok is True, result.message
    assert result.case_realization["kind"] == kind, (op, probe, case, result.case_realization)
    assert result.to_json()["case_realization"]["kind"] == kind

    summary = case_coverage([_result(contract, name) for name in VARLEN_CASE_DOMAIN])
    assert case in summary[kind], (op, probe, case, summary)
    if kind != "exact":
        assert case not in summary["exact"]


def test_a_nan_candidate_report_stays_canonicalizable():
    """@brief 候选输出 NaN 时报告仍能做身份指纹：应判 incorrect，而不是 evaluator 崩溃。

    poison 残留、漏写 stage 或错误递推都会给出 NaN 输出。报告是评测身份的一部分
    （``TaskEvaluationResult.document()`` 含 ``report``），而 ``tcenv.identity`` 拒绝非有限浮点，
    所以最坏元素诊断值必须像 ``scaled_err`` 一样用 ``"inf"``/``"nan"`` 拼写。否则一个错误候选会让
    整个评测抛 ``IdentityError``，而不是返回可审计的 ``incorrect``。
    """
    from tcenv.identity import fingerprint

    contract = load_task_contract(TASKS["varlen_kda"][0], dtype="fp16", size="mixed")
    data = prepare_v2_case(contract, "canonical", SEED)
    abi = derive_tensor_memory_contract(contract.instance, target=load_target("apple-m4-max"))
    # 0xFF 是 harness 的 poison 图案：浮点解释为 NaN，整数为负数。
    out_raw = {
        tensor.name: bytes([0xFF]) * (math.prod(int(v) for v in tensor.shape) * get_dtype(tensor.dtype).nbytes)
        for tensor in abi.tensors
        if tensor.space != "workspace" and tensor.access in ("write", "read_write")
    }

    result = _check_v2_case(contract, abi, data, out_raw, "canonical")
    document = result.to_json()

    assert result.ok is False
    assert document["worst"]["scaled_err"] == "inf"
    assert document["worst"]["got"] == "nan"
    assert len(fingerprint(document)) == 64
