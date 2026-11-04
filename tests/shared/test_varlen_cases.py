"""L1 (packed varlen) evaluation cases: one ABI, many legal ``cu_seqlens``.

CPU only, backend free.  A varlen tier fixes ``total_tokens``/``num_seqs`` (so the packed tensor
shapes, the ABI and every buffer stay compile-time constants) but **not** the segmentation:
``sizes.<tier>.seq_lens`` is only the *canonical* profile published for scheduling and benchmarking.
Correctness must accept every legal boundary set under the same total/count, which is what these
cases generate and what the harness independently validates.

This file pins the case-generation contract:

* every declared case keeps the tier's total/count and produces an ``i32`` ``[NSP1]`` tensor that
  starts at 0, is monotonically non-decreasing and ends at ``total_tokens``;
* an empty segment exists only in ``zero_length`` (exactly one) and ``single_sequence`` (one
  non-empty sequence, the rest empty); the static profile never has one and the batch always
  carries at least one token;
* each case has the shape its name promises where the tier can host it, and every case carries a
  harness-computed realization record that separates ``exact`` from ``partial``/
  ``not_constructible`` — an unrealizable shape is never counted as full coverage;
* a single-sequence tier has exactly one legal segmentation, so the seven multi-segment cases are
  recorded as ``not_constructible`` there — the count is never changed to manufacture a case;
* the fresh draw never repeats a published canonical profile;
* the oracle is driven by the *actual* boundary tensor (an empty segment keeps its initial state),
  and the same actual bytes reach the evaluation report as the input digest.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from tcenv.opspec.v2 import VARLEN_CASE_DOMAIN
from tcenv.portfolio.domain import evaluation_input_digest, semantic_instance_digest
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.taskcontract.reference import load_reference
from tcenv.taskcontract.varlen import segment_lengths
from tcenv.taskcontract.varlen_cases import (
    CASE_REALIZATION_KINDS,
    EXACT,
    NOT_CONSTRUCTIBLE,
    PARTIAL,
    VARLEN_CHUNK,
    VARLEN_CHUNK_TARGETS,
    case_realization,
)

KDA_DIR = REPO_ROOT / "tasks" / "varlen_kda"
SCALE_DIR = REPO_ROOT / "tasks" / "varlen_scale"

#: ``op_name -> (task directory, declared sizes)``.
TASKS: dict[str, tuple] = {
    "varlen_kda": (KDA_DIR, ("fixed", "mixed", "uniform")),
    "varlen_scale": (SCALE_DIR, ("mixed", "uniform", "single")),
}
COMBOS = tuple((op, size) for op, (_, sizes) in TASKS.items() for size in sizes)

CASES = VARLEN_CASE_DOMAIN
SEED = 7
#: Cases that must reshape the segmentation on a multi-sequence tier.
RESHAPED = ("permuted", "skewed", "all_unaligned_primes", "max_handoff_stress", "chunk_boundary", "zero_length")


def _contract(op: str, size: str):
    """@brief 装配一道 varlen 题的语义实例。

    @param op 题目名。
    @param size 档位名。
    @return :class:`TaskContract`。
    """
    return load_task_contract(TASKS[op][0], dtype="fp16", size=size)


def _oracle(op: str, size: str):
    """@brief 装载一道 varlen 题的 reference oracle。

    @param op 题目名。
    @param size 档位名。
    @return :class:`ReferenceModule`。
    """
    contract = _contract(op, size)
    return load_reference(TASKS[op][0], op_name=op, instance=contract.instance)


def _inputs(op: str, size: str, case: str, seed: int = SEED) -> dict:
    """@brief 生成某题/档/用例的合同输入（走真实 loader + reference 钩子）。

    @param op 题目名。
    @param size 档位名。
    @param case 用例名。
    @param seed 输入种子。
    @return 输入字典。
    """
    contract = _contract(op, size)
    return contract.reference.generate(contract.instance, case, seed)


def _lengths(op: str, size: str, case: str, seed: int = SEED) -> tuple[int, ...]:
    """@brief 某个用例生成的实际每段长度。

    @param op 题目名。
    @param size 档位名。
    @param case 用例名。
    @param seed 输入种子。
    @return 长度元组。
    """
    return segment_lengths(_inputs(op, size, case, seed)["cu_seqlens"])


def _count(op: str, size: str) -> int:
    """@brief 该档声明的序列条数。

    @param op 题目名。
    @param size 档位名。
    @return ``num_seqs``。
    """
    return int(_contract(op, size).instance.symbols["num_seqs"])


def _is_prime(value: int) -> bool:
    """@brief 测试侧独立素数判定（不复用题目的私有 helper）。

    @param value 整数。
    @return 是否质数。
    """
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _multi_combos() -> tuple[tuple[str, str], ...]:
    """@brief 有多条序列的档位（单段档只有一种合法分段）。

    @return ``(op, size)`` 元组集。
    """
    return tuple((op, size) for op, size in COMBOS if _count(op, size) > 1)


def _single_combos() -> tuple[tuple[str, str], ...]:
    """@brief 只声明了一条序列的档位。

    @return ``(op, size)`` 元组集。
    """
    return tuple((op, size) for op, size in COMBOS if _count(op, size) == 1)


@pytest.mark.parametrize("op,size", COMBOS, ids=lambda value: str(value))
def test_every_tier_declares_the_l1_case_domain_with_canonical_first(op, size):
    """@brief 每档声明完整的 L1 case 域，且 ``canonical`` 排第一（benchmark/validate 用它）。

    “声明完整”不等于“每档都能实现”：能否实现由 realization 记录逐 case 说明。
    """
    instance = _contract(op, size).instance
    assert tuple(instance.case_domain.cases) == CASES
    assert instance.case_domain.cases[0] == "canonical"
    assert instance.cu_seqlens is not None
    assert instance.canonical_seq_lens is not None


@pytest.mark.parametrize("op,size", COMBOS, ids=lambda value: str(value))
@pytest.mark.parametrize("case", CASES)
def test_every_case_keeps_the_packed_shape_and_a_legal_boundary_set(op, size, case):
    """@brief 任何 case 都不改 total/count，且边界满足 L1 合同（首 0、单调不减、末 total、i32）。"""
    instance = _contract(op, size).instance
    total_tokens = int(instance.symbols["total_tokens"])
    num_seqs = int(instance.symbols["num_seqs"])
    inputs = _inputs(op, size, case)
    cu = inputs["cu_seqlens"]

    assert cu.dtype == torch.int32
    assert tuple(cu.shape) == (num_seqs + 1,)
    values = [int(value) for value in cu.tolist()]
    lengths = [end - begin for begin, end in zip(values, values[1:])]

    assert values[0] == 0
    assert values[-1] == total_tokens
    assert all(begin <= end for begin, end in zip(values, values[1:]))
    assert len(lengths) == num_seqs
    assert sum(lengths) == total_tokens
    assert total_tokens >= 1
    if case not in ("zero_length", "single_sequence"):
        # 静态声明的每段 >= 1；零长度只能由运行时 cu 构造（zero_length 与 single_sequence 会用到）。
        assert min(lengths) >= 1
    assert min(instance.canonical_seq_lens) >= 1
    if case == "canonical":
        assert tuple(values) == instance.cu_seqlens
    # 打包张量的形状不随 case 改变（ABI 是编译期常量）。
    assert inputs["q" if instance.op_name == "varlen_kda" else "x"].shape[0] == total_tokens
    if instance.op_name == "varlen_kda":
        assert inputs["initial_state"].shape == (num_seqs, instance.symbols["H"], instance.symbols["D"], instance.symbols["D"])


@pytest.mark.parametrize("op,size", _multi_combos(), ids=lambda value: str(value))
def test_only_the_declared_cases_may_contain_an_empty_segment(op, size):
    """@brief 只有 ``zero_length``（恰好一个空段）与 ``single_sequence``（其余全空）可以有空段。"""
    lengths = _lengths(op, size, "zero_length")
    assert lengths.count(0) == 1
    assert sum(lengths) >= 1
    assert lengths != _contract(op, size).instance.canonical_seq_lens

    single = _lengths(op, size, "single_sequence")
    assert single.count(0) == _count(op, size) - 1
    assert sum(1 for value in single if value > 0) == 1

    for case in CASES:
        if case in ("zero_length", "single_sequence"):
            continue
        assert min(_lengths(op, size, case)) >= 1, case


@pytest.mark.parametrize("op,size", _single_combos(), ids=lambda value: str(value))
def test_a_single_sequence_tier_has_exactly_one_legal_segmentation(op, size):
    """@brief 单段档的 packed 形状只允许一种分段：所有 case 都是它（数量不能变）。"""
    instance = _contract(op, size).instance
    assert instance.cu_seqlens == (0, instance.symbols["total_tokens"])
    for case in CASES:
        assert _lengths(op, size, case) == instance.canonical_seq_lens, case


@pytest.mark.parametrize("op,size", _multi_combos(), ids=lambda value: str(value))
def test_the_correctness_cases_actually_reshape_the_segmentation(op, size):
    """@brief 多段档的 correctness case 必须给出不同的合法分布，而不是重放 canonical。"""
    canonical = _contract(op, size).instance.canonical_seq_lens
    layouts = {case: _lengths(op, size, case) for case in CASES}
    for case in RESHAPED:
        assert layouts[case] != canonical, case
    # 小档（total 8~16、2~3 段）的合法组成本来就少，不同 case 可能碰巧落到同一分布；
    # 但“整组 case 只是 canonical 换名字”必须被这条断言拦住。
    assert len({layouts[case] for case in RESHAPED}) >= 4
    # single_sequence 在多段档上是真实布局：一条非空序列 + 若干空序列，不是 canonical 重放。
    assert layouts["single_sequence"] != canonical
    assert layouts["single_sequence"].count(0) == _count(op, size) - 1
    assert sum(1 for value in layouts["single_sequence"] if value > 0) == 1


@pytest.mark.parametrize("op,size", _multi_combos(), ids=lambda value: str(value))
def test_the_prime_case_uses_prime_lengths_on_every_declared_tier(op, size):
    """@brief ``all_unaligned_primes`` 在所有多段档上给出全质数分段（连续 partial tail 紧邻）。"""
    lengths = _lengths(op, size, "all_unaligned_primes")
    assert lengths != _contract(op, size).instance.canonical_seq_lens
    assert all(_is_prime(length) for length in lengths), lengths


@pytest.mark.parametrize("op,size", _multi_combos(), ids=lambda value: str(value))
def test_the_handoff_and_skewed_cases_have_their_declared_shape(op, size):
    """@brief handoff = 一条最长链 + 全 1 碎片；skewed = 一条长序列 + 多条 2 token 短序列。"""
    instance = _contract(op, size).instance
    total_tokens = int(instance.symbols["total_tokens"])
    num_seqs = int(instance.symbols["num_seqs"])

    handoff = _lengths(op, size, "max_handoff_stress")
    assert handoff.count(1) == num_seqs - 1
    assert handoff[0] == total_tokens - (num_seqs - 1)

    skewed = _lengths(op, size, "skewed")
    assert sorted(skewed[1:]) == [2] * (num_seqs - 1)
    assert skewed[0] == total_tokens - 2 * (num_seqs - 1)
    assert handoff != skewed


@pytest.mark.parametrize("op,size", _multi_combos(), ids=lambda value: str(value))
def test_the_chunk_boundary_case_plans_around_the_declared_chunk_size(op, size):
    """@brief chunk case 从 ``BT-1`` 起按 chunk 目标铺段，末段吸收该档余量。"""
    from tasks.varlen_kda.reference import CHUNK, CHUNK_TARGETS

    # 两个任务的 case planner 是各自独立的，但 chunk 契约必须一致，且等于 harness 的核对口径。
    from tasks.varlen_scale.reference import CHUNK as SCALE_CHUNK
    from tasks.varlen_scale.reference import CHUNK_TARGETS as SCALE_TARGETS

    assert (SCALE_CHUNK, SCALE_TARGETS) == (CHUNK, CHUNK_TARGETS) == (VARLEN_CHUNK, VARLEN_CHUNK_TARGETS)

    lengths = _lengths(op, size, "chunk_boundary")
    assert lengths != _contract(op, size).instance.canonical_seq_lens
    assert lengths[0] == CHUNK - 1  # BT-1：partial tail 紧邻 chunk 边界
    assert min(lengths) <= CHUNK  # 至少一段落在 chunk 大小以内（partial tail）
    assert lengths[-1] >= 1
    assert sum(lengths) == int(_contract(op, size).instance.symbols["total_tokens"])


@pytest.mark.parametrize("op,size", _multi_combos(), ids=lambda value: str(value))
def test_the_fresh_case_never_repeats_the_canonical_profile(op, size):
    """@brief fresh 分布随 seed 变化，且从不与公开的 canonical 档重合（tune/unseen 不相交）。"""
    canonical = _contract(op, size).instance.canonical_seq_lens
    layouts = set()
    for seed in range(10):
        lengths = _lengths(op, size, "fresh_distribution", seed)
        assert lengths != canonical, seed
        layouts.add(lengths)
    assert len(layouts) > 1


@pytest.mark.parametrize("op,size", COMBOS, ids=lambda value: str(value))
def test_the_oracle_segments_by_the_actual_boundaries(op, size):
    """@brief oracle 按实际 ``cu_seqlens`` 分段：空段保持状态，非空段推进状态；elementwise 全token被覆盖。"""
    contract = _contract(op, size)
    instance = contract.instance
    case = "zero_length" if _count(op, size) > 1 else "canonical"
    inputs = contract.reference.generate(instance, case, SEED)
    result = contract.reference.execute(instance, inputs)
    lengths = segment_lengths(inputs["cu_seqlens"])

    if op == "varlen_kda":
        initial = inputs["initial_state"].to(torch.float64)
        final = result["final_state"]
        assert final.shape == initial.shape
        for seq, length in enumerate(lengths):
            if length == 0:
                # 空序列：没有 output 行，终态必须等于初态（不得推进状态）。
                assert torch.equal(final[seq], initial[seq])
            else:
                assert not torch.allclose(final[seq], initial[seq])
        assert result["output"].shape[0] == instance.symbols["total_tokens"]
    else:
        # elementwise：只有被 cu 覆盖的 token 存在，且每个 token 恰好一次。
        assert tuple(lengths) == tuple(segment_lengths(inputs["cu_seqlens"]))
        assert sum(lengths) == instance.symbols["total_tokens"]
        assert torch.equal(result["y"], 3 * inputs["x"].to(torch.float64))


@pytest.mark.parametrize("op,size", _multi_combos(), ids=lambda value: str(value))
def test_a_reshaped_case_really_changes_the_oracle_result(op, size):
    """@brief 把实际分段换成 canonical：stateful oracle 的结果必须变，elementwise 的 y 不变。"""
    contract = _contract(op, size)
    instance = contract.instance
    inputs = dict(contract.reference.generate(instance, "permuted", SEED))
    actual = contract.reference.execute(instance, inputs)
    canonical = dict(inputs)
    canonical["cu_seqlens"] = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)
    copy = contract.reference.execute(instance, canonical)

    if op == "varlen_kda":
        # 分段真的驱动了递推：换回 canonical 边界后逐段终态不同。
        assert not torch.allclose(actual["final_state"], copy["final_state"])
    else:
        # elementwise 的数学与分段无关：任何合法分段覆盖的 token 集合相同，y 必须逐位一致。
        assert torch.equal(actual["y"], copy["y"])


@pytest.mark.parametrize("size", TASKS["varlen_kda"][1])
def test_each_sequence_gets_a_distinct_non_zero_initial_state(size):
    """@brief ``initial_state`` 逐序列不同且非零：串段或零状态续传不能靠运气通过。"""
    contract = _contract("varlen_kda", size)
    instance = contract.instance
    for case in CASES:
        state = contract.reference.generate(instance, case, SEED)["initial_state"]
        assert state.shape[0] == instance.symbols["num_seqs"]
        for seq in range(state.shape[0]):
            assert float(state[seq].abs().max()) > 0.0, (case, seq)
        for left, right in itertools.combinations(range(state.shape[0]), 2):
            assert not torch.equal(state[left], state[right]), (case, left, right)


@pytest.mark.parametrize("op,size", COMBOS, ids=lambda value: str(value))
def test_the_cases_are_evaluation_inputs_of_one_domain_point(op, size):
    """@brief permuted/skewed/fresh 不是新 domain point：语义身份不变，实际输入身份不同。"""
    contract = _contract(op, size)
    instance = contract.instance
    oracle = contract.reference
    canonical = oracle.generate(instance, "canonical", SEED)
    other_case = "permuted" if _count(op, size) > 1 else "fresh_distribution"
    other = oracle.generate(instance, other_case, SEED)

    semantic = semantic_instance_digest(instance)
    left = evaluation_input_digest(instance, "canonical", SEED, canonical)
    right = evaluation_input_digest(instance, other_case, SEED, other)
    assert left != right
    assert left != semantic and right != semantic
    # 重放：同一 (instance, case, seed) 重新生成得到同一身份。
    assert evaluation_input_digest(instance, "canonical", SEED, oracle.generate(instance, "canonical", SEED)) == left


def _record(op: str, size: str, case: str) -> dict:
    """@brief 某题/档/用例的 realization 记录（由实际分段独立计算）。

    @param op 题目名。
    @param size 档位名。
    @param case 用例名。
    @return 闭合记录。
    """
    instance = _contract(op, size).instance
    return case_realization(case, _lengths(op, size, case), instance.canonical_seq_lens)


@pytest.mark.parametrize("op,size", COMBOS, ids=lambda value: str(value))
@pytest.mark.parametrize("case", CASES)
def test_every_case_records_what_it_actually_realized(op, size, case):
    """@brief 每个 case 都有一条闭合的 realization 记录，且不会把不可构造/部分实现记成完整覆盖。"""
    record = _record(op, size, case)
    lengths = _lengths(op, size, case)

    assert record["case"] == case
    assert record["kind"] in CASE_REALIZATION_KINDS
    assert record["segments"] == list(lengths)
    assert record["reason"]
    assert record["kind"] != "deviates"

    if _count(op, size) == 1 and case not in ("canonical", "single_sequence"):
        # 单段档只有一种合法分段：这些 case 的形状不存在，显式记为不可构造。
        assert record["kind"] == NOT_CONSTRUCTIBLE, case
        assert record["missing"]
    elif case == "chunk_boundary":
        # 小档的槽位放不下整组 BT 目标：partial，且缺失项逐条列出。
        assert record["kind"] == PARTIAL
        assert record["missing"]
    elif case == "permuted" and len(set(_contract(op, size).instance.canonical_seq_lens)) == 1:
        # 均匀 canonical 没有“不同顺序”，只能用同 total/count 的另一合法切分。
        assert record["kind"] == PARTIAL
        assert record["missing"] == ["a permutation of the canonical multiset"]
    else:
        assert record["kind"] == EXACT, (op, size, case, record)


@pytest.mark.parametrize("op,size", _single_combos(), ids=lambda value: str(value))
def test_a_single_sequence_tier_reports_its_coverage_honestly(op, size):
    """@brief 单段档的 7 个 case 显式不可构造；只有 canonical 与 single_sequence 算精确覆盖。"""
    kinds = {case: _record(op, size, case)["kind"] for case in CASES}
    assert kinds["canonical"] == EXACT
    assert kinds["single_sequence"] == EXACT
    assert {case for case, kind in kinds.items() if kind == NOT_CONSTRUCTIBLE} == set(CASES) - {
        "canonical",
        "single_sequence",
    }


@pytest.mark.parametrize("op,size", _multi_combos(), ids=lambda value: str(value))
def test_the_chunk_case_lists_the_bt_targets_the_tier_cannot_host(op, size):
    """@brief chunk_boundary 的缺失项就是该档槽位放不下的 BT 目标，可逐条审计。"""
    record = _record(op, size, "chunk_boundary")
    assert record["kind"] == PARTIAL
    assert record["realized"][0] == f"BT target {VARLEN_CHUNK - 1}"

    count = _count(op, size)
    lengths = _lengths(op, size, "chunk_boundary")
    planned = [VARLEN_CHUNK_TARGETS[index % len(VARLEN_CHUNK_TARGETS)] for index in range(count - 1)]
    expected_missing = [f"BT target {target}" for index, target in enumerate(planned) if lengths[index] != target]
    expected_missing += [f"BT target {target}" for target in VARLEN_CHUNK_TARGETS[len(planned) :]]
    assert record["missing"] == expected_missing
