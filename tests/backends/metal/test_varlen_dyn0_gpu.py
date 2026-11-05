"""V2 Dyn-0：真实 Metal 上的 work list、两类多阶段 fixture 与 handoff 负例。

本文件只证明 **机制**，不证明 KDA 性能：

* fixture A（``prep_chain.metal``）：stage 1 预处理 per-token norm/log-decay/beta 并**在 device 上
  按实际 ``cu_seqlens`` 建 work list**，stage 2 每个 ``(seq, head)`` 一个 threadgroup 顺序递推。
  它是“预处理 + 顺序链”，**不冒充**跨 piece handoff：整条链始终在同一个 threadgroup 里跑完。
* fixture B（``piece_handoff.metal``）：stage 0 规划 piece，stage 1/2/3 分别处理首/中/末 piece；
  中/末 piece 的初值来自 ``handoff_states``，由 ``handoff_flags`` 的 ready/consumed 协议管理。
  stage 之间是真实 Metal serial encoder + ``memoryBarrierWithScope_(MTLBarrierScopeBuffers)``，
  没有任何同 launch 跨 threadgroup 自旋。

Dyn-0 的几何是静态的：grid、stage 与参数槽位在构建期确定，多余 threadgroup / stage 读运行时
count 后退出。两类 fixture 都只使用 OpSpec 声明的 workspace；未被任何 stage 绑定的 workspace
（例如 fixture B 的 ``log_decay_tab`` 与 ``token_stats``）不会被分配。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.target import load_target

from ...support.varlen_probe import write_probe_task

pytestmark = pytest.mark.gpu

#: 固定的 fixture 源码（backend 专属数据，放 ``tests/support/golden/metal/varlen/``）。
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "support" / "golden" / "metal" / "varlen"
PREP_CHAIN = (FIXTURE_DIR / "prep_chain.metal").read_text(encoding="utf-8")
PIECE_HANDOFF = (FIXTURE_DIR / "piece_handoff.metal").read_text(encoding="utf-8")

#: mixed 档的派生量，用来把证据里的 grid/字节数与公式对上。
MIXED_CHAINS = 3 * 2  # num_seqs * H
MAX_SPLIT = 3


def _evaluate(source: str, *, size: str = "mixed", cases: tuple[str, ...] | None = None, locator=None):
    """@brief 在真实 Metal 上评估一个候选（quick：只跑正确性，不计时）。

    @param source 候选源码。
    @param size 长度分布档。
    @param cases 用例子集；None 跑合同的全部用例。
    @param locator 覆盖任务 locator（探针档用）。
    @return 组合运行结果。
    """
    if locator is None:
        locator = bind_task("varlen_kda", "fp16", size).locator
    return evaluate_candidate(
        locator, source, target=load_target("apple-m4-max"), mode="quick", reps=3, cases=cases
    )


def _mutate(source: str, needle: str, replacement: str) -> str:
    """@brief 唯一子串替换（fixture 负例的唯一构造方式）。

    @param source 原源码。
    @param needle 必须恰好出现一次的片段。
    @param replacement 替换文本。
    @return 新源码。
    """
    assert source.count(needle) == 1, needle
    return source.replace(needle, replacement)


def _assert_refused(run) -> dict:
    """@brief 断言候选被正确性/安全检查拒绝，并返回报告。

    @param run 组合运行结果。
    @return 报告字典。
    """
    report = run.result.report
    assert report["status"] in ("incorrect", "unsafe"), report["summary"]
    if report["status"] == "incorrect":
        assert report["correctness"]["ok"] is False
    return report


# --- fixture A: 预处理 + 顺序链 -------------------------------------------------------


def test_fixture_a_preprocessing_then_one_sequential_chain_per_sequence_and_head():
    """@brief fixture A 在每个 L1 case 上正确，且只分配它真正绑定的 workspace。"""
    run = _evaluate(PREP_CHAIN)
    report = run.result.report

    assert report["status"] == "ok", report["summary"]
    assert report["correctness"]["ok"] is True
    cases = report["correctness"]["cases"]
    assert [case["case"] for case in cases] == list(
        load_task_contract(REPO_ROOT / "tasks" / "varlen_kda", dtype="fp16", size="mixed").instance.case_domain.cases
    )

    symbols = load_task_contract(REPO_ROOT / "tasks" / "varlen_kda", dtype="fp16", size="mixed").instance.symbols
    tokens, heads, dim = symbols["total_tokens"], symbols["H"], symbols["D"]
    evidence = run.evaluation
    # 只有 fixture 真正绑定的 4 个 workspace 被分配；handoff_* 虽然进了 ABI，但没有 stage 使用。
    assert evidence["workspace_bytes"] == {
        "token_stats": tokens * heads * 3 * 4,
        "log_decay_tab": tokens * heads * dim * 4,
        "work_items": MIXED_CHAINS * MAX_SPLIT * 2 * 4,
        "item_counts": MIXED_CHAINS * MAX_SPLIT * 4,
    }
    dispatch = evidence["dispatch"]
    assert [item["kernel"] for item in dispatch] == ["varlen_prep", "varlen_chain"]
    assert dispatch[0]["workspace"] == ["token_stats", "log_decay_tab", "work_items", "item_counts"]
    assert dispatch[1]["threads"] == [32, 1, 1]
    # 静态 grid 是 ABI 容量（每条链 3 个 piece 的位置），实际只有 num_seqs*H 个 item：
    # 多出来的 threadgroup 读 item_counts[0] 后退出。
    assert dispatch[1]["grid"][0] == MIXED_CHAINS * MAX_SPLIT
    assert dispatch[1]["grid"][0] > MIXED_CHAINS


def test_fixture_a_an_empty_sequence_must_not_advance_state(tmp_path):
    """@brief 空序列不得推进状态（也不得把下一条序列的 token 算进自己的终态）。"""
    source = _mutate(
        PREP_CHAIN,
        "int end = begin + int(valid);",
        "int end = begin + max(int(valid), 1);",
    )
    run = _evaluate(source, cases=("canonical", "zero_length"))

    _assert_refused(run)


def test_fixture_a_a_consumer_dispatched_before_its_producer_is_refused():
    """@brief 把 recurrence 段排在预处理段之前（consumer-first）必须被拒绝。"""
    prep = (
        "launch varlen_prep grid=total_tokens*H,1,1 threads=1,1,1 "
        "workspace=token_stats,log_decay_tab,work_items,item_counts"
    )
    chain = (
        "launch varlen_chain grid=num_seqs*H*3,1,1 threads=32,1,1 "
        "workspace=work_items,item_counts,token_stats,log_decay_tab"
    )
    source = _mutate(PREP_CHAIN, f"{prep}\n{chain}", f"{chain}\n{prep}")
    run = _evaluate(source, cases=("canonical",))

    _assert_refused(run)


def test_fixture_a_a_partial_tail_that_rounds_into_the_next_sequence_is_refused():
    """@brief 把段尾向上取整到 chunk 边界（读进下一条序列）必须被拒绝。"""
    source = _mutate(
        PREP_CHAIN,
        "int end = begin + int(valid);",
        "int end = begin + int(valid == 0u ? 0u : ((valid + 3u) & ~3u));",
    )
    run = _evaluate(source, cases=("chunk_boundary",))

    _assert_refused(run)


def test_fixture_a_sharing_one_state_across_heads_is_refused():
    """@brief 跨 head 共用一份 state（丢掉 head 维度）必须被拒绝。"""
    source = _mutate(
        PREP_CHAIN,
        "state[vi * D + ki] = initial_state[(seq * H + head) * D * D + vi * D + ki];",
        "state[vi * D + ki] = initial_state[seq * D * D + vi * D + ki];",
    )
    run = _evaluate(source, cases=("canonical",))

    _assert_refused(run)


def test_fixture_a_a_work_list_write_beyond_the_declared_capacity_is_unsafe():
    """@brief device 侧 work list 写出声明容量必须被 guard 抓到（非崩溃、非静默通过）。"""
    source = _mutate(PREP_CHAIN, "if (i < num_seqs * H) {", "if (i < total_tokens * H) {")
    run = _evaluate(source, cases=("canonical",))

    report = _assert_refused(run)
    assert report["status"] == "unsafe", report["summary"]
    assert any("work_items" in item for item in report["safety"]["violations"]), report["safety"]


def test_fixture_a_reads_a_large_item_table_from_global(tmp_path):
    """@brief item table 超过 threadgroup 容量（64 项）时仍正确：走 global fallback。

    探针档保留 shipped ``reference.py`` 与全部 ABI 字段，只换分段：40 条 2-token 序列 × H=2 是
    80 条链 > 64 项缓存容量，所以这一档只有 global 读表能算对。
    """
    # 探针目录必须叫 op_name（自动发现按 tasks/<op>/ 的身份校验目录名）。
    write_probe_task(REPO_ROOT / "tasks" / "varlen_kda", tmp_path / "varlen_kda", {"wide": [2] * 40})
    locator = discover_standard_tasks(tmp_path, code_root=REPO_ROOT)[0]
    contract = load_task_contract(tmp_path / "varlen_kda", dtype="fp16", size="wide")
    assert contract.instance.symbols["num_seqs"] * contract.instance.symbols["H"] == 80

    run = _evaluate(PREP_CHAIN, locator=locator)
    report = run.result.report

    assert report["status"] == "ok", report["summary"]
    assert report["correctness"]["ok"] is True
    assert run.evaluation["workspace_bytes"]["work_items"] == 80 * MAX_SPLIT * 2 * 4


# --- fixture B: 真实 piece handoff ----------------------------------------------------


def test_fixture_b_real_piece_handoff_passes_every_case():
    """@brief fixture B 在每个 L1 case 上正确；handoff 存储与 OpSpec 容量公式一致。"""
    run = _evaluate(PIECE_HANDOFF)
    report = run.result.report

    assert report["status"] == "ok", report["summary"]
    assert report["correctness"]["ok"] is True
    assert {case["case"] for case in report["correctness"]["cases"]} >= {"canonical", "max_handoff_stress", "zero_length"}

    symbols = load_task_contract(REPO_ROOT / "tasks" / "varlen_kda", dtype="fp16", size="mixed").instance.symbols
    dim = symbols["D"]
    evidence = run.evaluation
    # 只分配 handoff 机制本身使用的 4 个 workspace：per-token 表在 fixture B 里并不存在。
    assert evidence["workspace_bytes"] == {
        "work_items": MIXED_CHAINS * MAX_SPLIT * 2 * 4,
        "item_counts": MIXED_CHAINS * MAX_SPLIT * 4,
        "handoff_flags": MIXED_CHAINS * 4 * 4,
        "handoff_states": MIXED_CHAINS * (MAX_SPLIT - 1) * dim * dim * 4,
    }
    assert [item["kernel"] for item in evidence["dispatch"]] == [
        "varlen_plan",
        "varlen_piece0",
        "varlen_piece1",
        "varlen_piece2",
    ]
    for item in evidence["dispatch"][1:]:
        assert item["grid"][0] == MIXED_CHAINS * MAX_SPLIT


def test_fixture_b_continuation_starting_from_the_initial_state_is_refused():
    """@brief 中/末 piece 从 ``initial_state`` 续传（丢掉 handoff）必须被拒绝。"""
    source = _mutate(
        PIECE_HANDOFF,
        "state[vi * D + ki] = handoff_states[slot * D * D + vi * D + ki];",
        "state[vi * D + ki] = initial_state[(seq * H + head) * D * D + vi * D + ki];",
    )
    run = _evaluate(source, cases=("max_handoff_stress",))

    _assert_refused(run)


def test_fixture_b_continuation_starting_from_zero_is_refused():
    """@brief 中/末 piece 从零状态续传必须被拒绝。"""
    source = _mutate(
        PIECE_HANDOFF,
        "state[vi * D + ki] = handoff_states[slot * D * D + vi * D + ki];",
        "state[vi * D + ki] = 0.0f;",
    )
    run = _evaluate(source, cases=("max_handoff_stress",))

    _assert_refused(run)


def test_fixture_b_a_producer_that_never_publishes_the_flag_is_refused():
    """@brief producer 只写 state 不发布 ready flag：consumer 不得读到“碰巧正确”的状态。"""
    source = _mutate(PIECE_HANDOFF, "        handoff_flags[slot] = FLAG_READY;", "        (void)0;")
    run = _evaluate(source, cases=("max_handoff_stress",))

    _assert_refused(run)


def test_fixture_b_a_consumer_that_never_clears_the_flag_is_refused():
    """@brief consumer 未清 flag：链尾的协议后置条件必须拒绝该终态。"""
    source = _mutate(PIECE_HANDOFF, "            handoff_flags[slot] = FLAG_EMPTY;", "            (void)0;")
    run = _evaluate(source, cases=("max_handoff_stress",))

    _assert_refused(run)


def test_fixture_b_a_handoff_slot_shared_across_chains_is_refused():
    """@brief 跨 (seq, head) 共用 handoff slot（只按 piece 索引）必须被拒绝。"""
    source = _mutate(
        _mutate(
            PIECE_HANDOFF,
            "uint slot = chain * (MAX_SPLIT - 1u) + piece - 1u;",
            "uint slot = piece - 1u;",
        ),
        "uint slot = chain * (MAX_SPLIT - 1u) + piece;",
        "uint slot = piece;",
    )
    run = _evaluate(source, cases=("max_handoff_stress",))

    _assert_refused(run)


def test_fixture_b_a_partial_tail_that_rounds_into_the_next_sequence_is_refused():
    """@brief piece 内最后一个 chunk 未按 ``valid`` 掩码（读进下一条序列）必须被拒绝。"""
    source = _mutate(PIECE_HANDOFF, "int hi = min(base + int(BT), t1);", "int hi = base + int(BT);")
    run = _evaluate(source, cases=("chunk_boundary",))

    _assert_refused(run)


def test_fixture_b_a_consumer_dispatched_before_its_producer_is_refused():
    """@brief 把 piece1 排在 piece0 之前（consumer-first）必须被拒绝。"""
    piece0 = (
        "launch varlen_piece0 grid=num_seqs*H*3,1,1 threads=1,1,1 "
        "workspace=work_items,item_counts,handoff_flags,handoff_states"
    )
    piece1 = (
        "launch varlen_piece1 grid=num_seqs*H*3,1,1 threads=1,1,1 "
        "workspace=work_items,item_counts,handoff_flags,handoff_states"
    )
    source = _mutate(PIECE_HANDOFF, f"{piece0}\n{piece1}", f"{piece1}\n{piece0}")
    run = _evaluate(source, cases=("max_handoff_stress",))

    _assert_refused(run)


def test_fixture_b_an_empty_sequence_keeps_its_initial_state():
    """@brief 空序列的终态必须等于初态（规划段不写它就会留下 poison）。"""
    source = _mutate(
        PIECE_HANDOFF,
        "final_state[(seq * H + head) * D * D + vi * D + ki] =\n"
        "                        initial_state[(seq * H + head) * D * D + vi * D + ki];",
        "final_state[(seq * H + head) * D * D + vi * D + ki] = 0.0f;",
    )
    run = _evaluate(source, cases=("canonical", "zero_length"))

    _assert_refused(run)


def test_fixture_b_a_candidate_that_relies_on_zero_initialised_flags_is_refused():
    """@brief 依赖 workspace 初始为零（不自己初始化 flag）的候选在 poison trial 上必须失败。

    正确性协议对 out1/out3 **毒化**每个 workspace、对 out2 清零；``zero`` 声明只影响计时路径的
    清零，不改变这条协议。所以候选必须在每次 trial 自己初始化协议状态，不能假设初始为零。
    """
    source = _mutate(
        PIECE_HANDOFF,
        "            handoff_flags[chain * (MAX_SPLIT - 1u) + slot] = FLAG_EMPTY;",
        "            (void)0;",
    )
    run = _evaluate(source, cases=("max_handoff_stress",))

    _assert_refused(run)
