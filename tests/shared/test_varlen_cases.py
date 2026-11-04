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
    DEVIATES,
    EXACT,
    NOT_CONSTRUCTIBLE,
    PARTIAL,
    VARLEN_CHUNK,
    VARLEN_CHUNK_TARGETS,
    case_realization,
)
from tests.support.varlen_probe import write_probe_task

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
        # 小档的槽位放不下整组 BT 目标：partial 且缺失项逐条列出；六段全中才是 exact。
        assert record["kind"] in (PARTIAL, EXACT)
        assert bool(record["missing"]) == (record["kind"] == PARTIAL)
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


#: 每个 shipped 多段档的 chunk 分段与本次 realized/missing（显式预期，不由实现公式反推）。
#: ``missing`` 只表示本次分段未实现，不是“该档无法承载”（同档的其它合法分段可能实现）。
CHUNK_EXPECTATIONS = {
    ("varlen_kda", "mixed"): ((3, 7, 4), (3, 7, 4), (5, 1, 9)),
    ("varlen_kda", "uniform"): ((3, 13), (3,), (7, 4, 5, 1, 9)),
    ("varlen_scale", "mixed"): ((3, 6, 1), (3, 1), (7, 4, 5, 9)),
    ("varlen_scale", "uniform"): ((3, 5), (3, 5), (7, 4, 1, 9)),
}


@pytest.mark.parametrize("op,size", _multi_combos(), ids=lambda value: str(value))
def test_the_chunk_case_lists_the_bt_targets_this_layout_does_not_realize(op, size):
    """@brief chunk_boundary 按实际分段（含末段余量）核算本次 realized/missing，逐条可审计。"""
    record = _record(op, size, "chunk_boundary")
    segments, realized, missing = CHUNK_EXPECTATIONS[(op, size)]

    assert record["segments"] == list(segments)
    assert record["kind"] == PARTIAL
    assert record["realized"] == [f"BT target {target}" for target in realized]
    assert record["missing"] == [f"BT target {target}" for target in missing]


#: 边界探针档：shipped tiers 覆盖不到、但 planner 必须优雅处理的 total/count。
PROBES: dict[str, tuple[int, ...]] = {
    "all-prime-impossible": (1, 2),
    "all-ones": (1, 1),
    "all-twos": (2, 2),
    "all-twos-three": (2, 2, 2),
    "full-chunk": (5, 5, 5, 5, 5, 4),
    "last-bt": (5, 5, 4),
    "last-bt-plus": (5, 5, 5),
    "last-two-bt-plus": (5, 5, 9),
}


def _probe_contract(op: str, probe: str, tmp_path):
    """@brief 用 shipped planner + 探针档位装配一个合同。

    @param op 题目名。
    @param probe 探针档名。
    @param tmp_path 临时目录。
    @return :class:`TaskContract`。
    """
    # 标准入口必须是 tasks/<op_name>/：目录名与 op_name 不一致会被 loader 拒绝。
    task = write_probe_task(TASKS[op][0], tmp_path / op, {probe: PROBES[probe]})
    return load_task_contract(task, dtype="fp16", size=probe)


def _probe_record(contract, case: str) -> dict:
    """@brief 探针档上的 realization 记录（经公开 generate 取实际分段）。

    @param contract 任务合同。
    @param case 用例名。
    @return 闭合记录。
    """
    inputs = contract.reference.generate(contract.instance, case, SEED)
    return case_realization(
        case, segment_lengths(inputs["cu_seqlens"]), contract.instance.canonical_seq_lens
    )


@pytest.mark.parametrize("op", sorted(TASKS))
@pytest.mark.parametrize("probe", sorted(PROBES))
def test_a_probe_tier_generates_every_case_through_the_public_funnel(op, probe, tmp_path):
    """@brief ``[1,2]``/``[1,1]`` 这类边界档上，9 个 case 都必须能生成合法输入（不能裸中止）。"""
    contract = _probe_contract(op, probe, tmp_path)
    instance = contract.instance
    total = int(instance.symbols["total_tokens"])
    count = int(instance.symbols["num_seqs"])

    for case in CASES:
        inputs = contract.reference.generate(instance, case, SEED)
        values = [int(value) for value in inputs["cu_seqlens"].tolist()]
        assert values[0] == 0, (op, probe, case)
        assert values[-1] == total, (op, probe, case)
        assert len(values) == count + 1, (op, probe, case)
        lengths = segment_lengths(inputs["cu_seqlens"])
        assert len(lengths) == count and sum(lengths) == total, (op, probe, case)


@pytest.mark.parametrize("op", sorted(TASKS))
def test_a_tier_without_an_all_prime_segmentation_falls_back_without_aborting(op, tmp_path):
    """@brief ``[1,2]`` 不存在全素数分段：planner 退回 canonical，报告记 not_constructible。"""
    contract = _probe_contract(op, "all-prime-impossible", tmp_path)
    record = _probe_record(contract, "all_unaligned_primes")

    assert record["kind"] == NOT_CONSTRUCTIBLE
    assert record["missing"] == ["prime segments"]
    # 退回的是一个合法分段（不是非质数冒充 exact，也没有中止整次 correctness）。
    assert record["segments"] == list(contract.instance.canonical_seq_lens)
    assert record["kind"] != EXACT


@pytest.mark.parametrize("op", sorted(TASKS))
def test_an_all_ones_tier_marks_the_impossible_cases_not_constructible(op, tmp_path):
    """@brief ``[1,1]``：permuted/fresh/skewed/primes/handoff 不可构造，chunk 仍实现目标 1。"""
    contract = _probe_contract(op, "all-ones", tmp_path)
    expected_not_constructible = {
        "permuted",
        "skewed",
        "all_unaligned_primes",
        "max_handoff_stress",
        "fresh_distribution",
    }
    kinds = {case: _probe_record(contract, case) for case in CASES}

    for case in CASES:
        record = kinds[case]
        if case in expected_not_constructible:
            assert record["kind"] == NOT_CONSTRUCTIBLE, case
            assert record["missing"], case
        elif case == "chunk_boundary":
            # [1,1] 实际包含申报目标 1：只能记 partial，且 missing 里不能再出现 1。
            assert record["kind"] == PARTIAL, record
            assert record["realized"] == ["BT target 1"], record
            assert "BT target 1" not in record["missing"], record
        else:
            # canonical、single_sequence（[2,0]）与 zero_length（[2,0]）成立；handoff 的
            # [1,1] 只是公式退化的产物，没有长链，不能算完整覆盖。
            assert record["kind"] == EXACT, (case, record)
    # 全 1 档上“重放 canonical”不是 planner 失职，而是没有别的合法分段：不能记 deviates。
    assert all(record["kind"] != DEVIATES for record in kinds.values())


@pytest.mark.parametrize("op", sorted(TASKS))
def test_a_one_two_tier_keeps_the_realizable_cases_realizable(op, tmp_path):
    """@brief ``[1,2]``：primes 不可构造；chunk 只实现末段目标 1；skewed 部分实现。"""
    contract = _probe_contract(op, "all-prime-impossible", tmp_path)
    kinds = {case: _probe_record(contract, case)["kind"] for case in CASES}

    assert kinds["all_unaligned_primes"] == NOT_CONSTRUCTIBLE
    # (2,1) 的末段就是申报目标 1：不能把它记成不可构造，也不该说目标 1 缺失。
    assert kinds["chunk_boundary"] == PARTIAL
    assert kinds["skewed"] == PARTIAL
    for case in ("canonical", "permuted", "max_handoff_stress", "zero_length", "single_sequence", "fresh_distribution"):
        assert kinds[case] == EXACT, (case, kinds)
    assert DEVIATES not in kinds.values()


def test_the_classifier_separates_impossible_shapes_from_planner_misses():
    """@brief 不可构造（形状不存在）与 deviates（本可实现却重放）必须分开判定。"""
    # 全 1 档：只有一种正长度分段 → permuted/fresh/skewed 都是形状不存在。
    for case in ("permuted", "fresh_distribution", "skewed"):
        record = case_realization(case, (1, 1), (1, 1))
        assert record["kind"] == NOT_CONSTRUCTIBLE, (case, record)
        assert record["missing"], case
        assert record["kind"] != DEVIATES

    # chunk 例外：全 1 分段本身就包含申报目标 1 → 只能 partial，不能说 1 缺失。
    chunk = case_realization("chunk_boundary", (1, 1), (1, 1))
    assert chunk["kind"] == PARTIAL, chunk
    assert chunk["realized"] == ["BT target 1"], chunk
    assert "BT target 1" not in chunk["missing"], chunk

    # 可实现档上重放 canonical 才是 planner 失职。
    for case in ("permuted", "fresh_distribution"):
        record = case_realization(case, (2, 2), (2, 2))
        assert record["kind"] == DEVIATES, (case, record)
        assert record["missing"] == []

    # total == 2*count：长段与 2-token 碎片同长，构不成偏斜；退回 1-token 碎片后记 partial。
    skewed = case_realization("skewed", (3, 1), (2, 2))
    assert skewed["kind"] == PARTIAL, skewed
    assert skewed["missing"] == ["two-token fragments"]
    # 全 1 档的 handoff 没有长链，不能算完整 stress 覆盖。
    handoff = case_realization("max_handoff_stress", (1, 1), (1, 1))
    assert handoff["kind"] == NOT_CONSTRUCTIBLE, handoff
    assert handoff["missing"], handoff

    # chunk：一个目标都没出现是 planner 漏实现（同档 (1,3) 可承载目标 1/3），不是不可构造。
    witness = case_realization("chunk_boundary", (1, 3), (1, 3))
    assert witness["kind"] == PARTIAL, witness
    assert witness["realized"] == ["BT target 3", "BT target 1"], witness
    miss = case_realization("chunk_boundary", (2, 2), (2, 2))
    assert miss["kind"] == DEVIATES, miss
    assert "can host one" in miss["reason"], miss
    one = case_realization("chunk_boundary", (2, 1), (1, 2))
    assert one["kind"] == PARTIAL, one
    assert one["realized"] == ["BT target 1"], one
    assert case_realization("chunk_boundary", (3, 13), (8, 8))["kind"] == PARTIAL


@pytest.mark.parametrize("op", sorted(TASKS))
@pytest.mark.parametrize("probe", ("all-twos", "all-twos-three"))
def test_an_all_twos_tier_builds_a_real_skew_and_never_calls_it_exact(op, probe, tmp_path):
    """@brief ``total == 2*num_seqs``：长段与碎片同为 2 不成偏斜，必须换成真实偏斜并记 partial。"""
    contract = _probe_contract(op, probe, tmp_path)
    count = int(contract.instance.symbols["num_seqs"])
    record = _probe_record(contract, "skewed")

    assert record["segments"] == [count + 1, *([1] * (count - 1))], (op, probe, record)
    assert record["kind"] == PARTIAL, (op, probe, record)
    assert "two-token fragments" in record["missing"], (op, probe, record)
    assert record["kind"] != EXACT


@pytest.mark.parametrize("op", sorted(TASKS))
@pytest.mark.parametrize(
    "probe,expected",
    (
        ("last-bt", (3, 7, 4)),
        ("last-bt-plus", (3, 7, 5)),
        ("last-two-bt-plus", (3, 7, 9)),
    ),
)
def test_the_remainder_segment_counts_toward_chunk_coverage(op, probe, expected, tmp_path):
    """@brief 末段余量恰好是 BT / BT+1 / 2BT+1 时，它就是已实现的目标，不能记进 missing。"""
    contract = _probe_contract(op, probe, tmp_path)
    record = _probe_record(contract, "chunk_boundary")

    assert record["segments"] == list(expected)
    assert record["kind"] == PARTIAL, (op, probe, record)
    target = f"BT target {expected[-1]}"
    assert target in record["realized"], (op, probe, record)
    assert target not in record["missing"], (op, probe, record)


@pytest.mark.parametrize("op", sorted(TASKS))
def test_a_six_slot_tier_realizes_every_chunk_target(op, tmp_path):
    """@brief 六段档 (3,7,4,5,1,9) 覆盖全部申报目标 → exact，missing 为空。"""
    contract = _probe_contract(op, "full-chunk", tmp_path)
    record = _probe_record(contract, "chunk_boundary")

    assert record["segments"] == list(VARLEN_CHUNK_TARGETS)
    assert record["kind"] == EXACT, (op, record)
    assert record["missing"] == []
    assert len(record["realized"]) == len(VARLEN_CHUNK_TARGETS)


def test_chunk_realization_is_exact_only_when_every_declared_target_appears():
    """@brief 分类器按实际分段全集判定：六个目标缺一不可，末段同样计入。"""
    exact = case_realization("chunk_boundary", VARLEN_CHUNK_TARGETS, (5, 5, 5, 5, 5, 4))
    assert exact["kind"] == EXACT
    assert exact["missing"] == []

    without_last = case_realization("chunk_boundary", (3, 7, 4, 5, 1, 8), (5, 5, 5, 5, 5, 3))
    assert without_last["kind"] == PARTIAL
    assert without_last["missing"] == ["BT target 9"]


def test_permuted_partial_fallback_only_applies_to_a_uniform_canonical():
    """@brief 非均匀 canonical 本可真正排列；给出既非重放也非排列的布局是 deviates，不是 partial。"""
    miss = case_realization("permuted", (1, 6, 7), (4, 8, 2))
    assert miss["kind"] == DEVIATES, miss
    assert "uniform" not in miss["reason"], miss

    uniform = case_realization("permuted", (6, 4), (5, 5))
    assert uniform["kind"] == PARTIAL, uniform
    assert uniform["missing"] == ["a permutation of the canonical multiset"], uniform
