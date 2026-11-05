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
import json
import struct

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.runtime import BufferArg, MetalContext
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.remote.artifact_bundle import MAX_NATIVE_DESCRIPTOR_BYTES
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.target import load_target

from ...support.npu_sim.varlen_worklist import (
    NPU_STUB,
    dependency_edges,
    metal_adapter,
    npu_adapter,
    pack_items,
    unpack_items,
)
from ...support.varlen_probe import write_probe_task

pytestmark = pytest.mark.gpu

#: 固定的 fixture 源码（backend 专属数据，放 ``tests/support/golden/metal/varlen/``）。
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "support" / "golden" / "metal" / "varlen"
PREP_CHAIN = (FIXTURE_DIR / "prep_chain.metal").read_text(encoding="utf-8")
PIECE_HANDOFF = (FIXTURE_DIR / "piece_handoff.metal").read_text(encoding="utf-8")

#: mixed 档的派生量，用来把证据里的 grid/字节数与公式对上。
MIXED_CHAINS = 3 * 2  # num_seqs * H
MAX_SPLIT = 3


def _evaluate(
    source: str,
    *,
    size: str = "mixed",
    cases: tuple[str, ...] | None = None,
    locator=None,
):
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


def _why(report: dict) -> str:
    """@brief 从 schema-4 报告里取一句可读原因（报告没有、也不该有 ``summary`` 字段）。

    断言消息只能用报告真实存在的字段。这里取 ``status`` + ``message``，编译失败时补编译错误，
    有安全违例时补违例前几条。早先的版本读 ``report["summary"]``，一旦断言不成立就抛
    ``KeyError: 'summary'``：失败仍然发生，但真实原因（哪个 stage、哪条 message）被一个假"字段不存在"
    盖掉了，负例到底是真拒绝还是基础设施错误无从判断。

    @param report schema-4 报告字典（`EvalReport.to_json()` 的投影）。
    @return 一行原因，永不因缺字段抛异常。
    """
    parts = [f"status={report.get('status')!r}"]
    if report.get("stage"):
        parts.append(f"stage={report['stage']!r}")
    if report.get("message"):
        parts.append(str(report["message"]))
    errors = (report.get("compile") or {}).get("errors")
    if errors:
        parts.append(f"compile: {errors[:600]}")
    violations = (report.get("safety") or {}).get("violations") or []
    if violations:
        parts.append("safety: " + "; ".join(str(item) for item in violations[:3]))
    return " | ".join(parts)


def _assert_refused(run) -> dict:
    """@brief 断言候选被正确性/安全检查拒绝，并返回报告。

    @param run 组合运行结果。
    @return 报告字典。
    """
    report = run.result.report
    assert report["status"] in ("incorrect", "unsafe"), _why(report)
    if report["status"] == "incorrect":
        assert (report.get("correctness") or {}).get("ok") is False, _why(report)
    return report


# --- fixture A: 预处理 + 顺序链 -------------------------------------------------------


def test_fixture_a_preprocessing_then_one_sequential_chain_per_sequence_and_head():
    """@brief fixture A 在每个 L1 case 上正确，且只分配它真正绑定的 workspace。"""
    run = _evaluate(PREP_CHAIN)
    report = run.result.report

    assert report["status"] == "ok", _why(report)
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
    assert report["status"] == "unsafe", _why(report)
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

    assert report["status"] == "ok", _why(report)
    assert report["correctness"]["ok"] is True
    assert run.evaluation["workspace_bytes"]["work_items"] == 80 * MAX_SPLIT * 2 * 4


# --- fixture B: 真实 piece handoff ----------------------------------------------------


def test_fixture_b_real_piece_handoff_passes_every_case():
    """@brief fixture B 在每个 L1 case 上正确；handoff 存储与 OpSpec 容量公式一致。"""
    run = _evaluate(PIECE_HANDOFF)
    report = run.result.report

    assert report["status"] == "ok", _why(report)
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

    # 四段 × 15 绑定已经占掉 native descriptor 闭集上限的 97%：``kernels`` 表把 kernel 0 的
    # bindings 在顶层又写了一遍。再添一个 stage/绑定就会让*合法*候选在 worker 里 crash 成
    # “descriptor is too large”（基础设施错误冒充结论）。这里把它钉在上限之下，并留下实测值：
    # 需要更大头部空间时应改 descriptor 形状（新 epoch），不是放宽限额或跳过这条负例。
    native = (report.get("build") or {}).get("native") or {}
    descriptor = native.get("descriptor") or {}
    assert descriptor, _why(report)
    assert descriptor["kernel_names"] == [
        "varlen_plan",
        "varlen_piece0",
        "varlen_piece1",
        "varlen_piece2",
    ]
    descriptor_bytes = len(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert descriptor_bytes <= MAX_NATIVE_DESCRIPTOR_BYTES, (
        f"the 4-stage fixture's native descriptor is {descriptor_bytes} B of {MAX_NATIVE_DESCRIPTOR_BYTES} B; "
        "a real candidate with one more binding would fail as an infrastructure crash"
    )


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


# --- 设备 work list 与 NPU adapter 的真实差分 ------------------------------------------


def _standalone_plan_source(contract) -> str:
    """@brief fixture 源码 + 一份只声明实例符号的测试前置（harness assembler 注入同样整数）。

    @param contract varlen_kda 合同。
    @return 可直接交给 ``MetalContext.compile_all`` 的完整 MSL。
    """
    symbols = contract.instance.symbols
    preamble = (
        "// Test-only standalone preamble: the harness assembler injects the same integers from the\n"
        "// task instance.  The kernel bodies below are the fixture's own bytes.\n"
        f"constant constexpr uint num_seqs = {symbols['num_seqs']};\n"
        f"constant constexpr uint H = {symbols['H']};\n"
        f"constant constexpr uint D = {symbols['D']};\n"
    )
    marker = "using namespace metal;\n"
    assert PIECE_HANDOFF.count(marker) == 1
    return PIECE_HANDOFF.replace(marker, marker + preamble, 1)


def _device_work_list(contract, segments: tuple[int, ...]):
    """@brief 在真实设备上只跑 fixture B 的规划 stage，读回它写出的 work list。

    这是“设备实际字节”而不是 Python 转写：同一份 ``piece_handoff.metal`` 规划 kernel、同一
    ``cu_seqlens``，只把 harness 注入的整数常量补上；用声明容量与 poison 初值，未写满的项可辨认。

    @param contract varlen_kda 合同。
    @param segments 本次实际分段。
    @return ``(items, count, raw_words)``。
    """
    symbols = contract.instance.symbols
    heads, dim = symbols["H"], symbols["D"]
    chains = len(segments) * heads
    ctx = MetalContext()
    kernel = ctx.compile_all(_standalone_plan_source(contract), ["varlen_plan"])[0]

    def buffer(nbytes: int, payload: bytes = b"") -> object:
        buf = ctx.new_buffer(max(nbytes, 16))
        view = ctx.view(buf)
        view[:] = b"\xff" * len(view)
        if payload:
            view[: len(payload)] = payload
        return buf

    prefix = [0]
    for length in segments:
        prefix.append(prefix[-1] + length)
    initial_values = [0.5] * (chains * dim * dim)
    initial = buffer(chains * dim * dim * 4, struct.pack(f"<{len(initial_values)}f", *initial_values))
    cu_seqlens = buffer(len(prefix) * 4, struct.pack(f"<{len(prefix)}i", *prefix))
    final = buffer(chains * dim * dim * 4)
    words = buffer(chains * MAX_SPLIT * 2 * 4)
    counts = buffer(4)
    flags = buffer(chains * (MAX_SPLIT - 1) * 4)

    ctx.run(
        kernel,
        [
            [
                (0, BufferArg(initial)),
                (1, BufferArg(cu_seqlens)),
                (2, BufferArg(final)),
                (3, BufferArg(words)),
                (4, BufferArg(counts)),
                (5, BufferArg(flags)),
            ]
        ],
        (1, 1, 1),
        (1, 1, 1),
    )
    raw = bytes(ctx.view(words))
    count = struct.unpack_from("<i", bytes(ctx.view(counts)), 0)[0]
    return unpack_items(raw, count), count, raw


def test_the_device_work_list_matches_the_npu_adapter_byte_for_byte():
    """@brief 设备规划 stage 实际写出的 work list 与 NPU adapter 逐字节、逐边一致。

    这是跨“真实 Metal 设备字节”与“host-neutral NPU 规划”的差分，而不是同一函数换标签：
    设备侧是 ``piece_handoff.metal`` 的 ``varlen_plan`` 在 GPU 上跑出的 buffer，NPU 侧是
    ``npu_adapter`` 的独立规划实现。
    """
    contract = load_task_contract(REPO_ROOT / "tasks" / "varlen_kda", dtype="fp16", size="mixed")
    heads, dim = contract.instance.symbols["H"], contract.instance.symbols["D"]

    for segments in ((4, 8, 2), (12, 1, 1)):
        device, count, raw = _device_work_list(contract, segments)
        npu = npu_adapter(segments, heads=heads, max_split=MAX_SPLIT, dim=dim, target=NPU_STUB)
        metal = metal_adapter(segments, heads=heads, max_split=MAX_SPLIT, target=NPU_STUB)

        assert count == len(npu.items) == len(device)
        assert device == npu.items
        assert pack_items(device) == npu.words == raw[: 8 * count]
        assert dependency_edges(device) == npu.edges
        # 设备写出的每一项都非空：没有 valid=0 的空写回。
        assert all(item.valid > 0 for item in device)
        # Metal 侧的设备常量转写同样落在同一字节上。
        assert metal.words == npu.words

    # canonical 的 len=4 链在真实设备上是 2/1/1（不是旧 span 公式的 2/2/0）。
    device, _, _ = _device_work_list(contract, (4, 8, 2))
    assert [(item.begin, item.valid) for item in device if item.chain == 0] == [(0, 2), (2, 1), (3, 1)]
    assert [(item.begin, item.valid) for item in device if item.chain == 4] == [(12, 1), (13, 1)]


# --- fixture B: 旧 READY flag / 上一轮 handoff state ----------------------------------


#: 在首 piece 里留下“上一轮”的 READY + 终态：只对真的有三段的链生效（只有第三段会读 slot 1；
#: 段数更少的链，末 piece 是 piece1，它会把任何非 EMPTY 的 flag 判成未清）。旧值由 fixture 自己
#: 写出（``initial_state + 0.5``），不是 workspace 的 poison/零初值，所以这个实验不依赖 harness 的
#: 清零策略，也不靠“读到了没被写过的残留字节”来制造差异。
STALE_SEED = (
    "        // Negative-fixture only: a previous round left a READY flag and its terminal state in the\n"
    "        // slot a *later* piece will consume (slot 1 needs a 3-piece chain).  The value is written\n"
    "        // here instead of inherited from the workspace's poison/zero content, so the experiment\n"
    "        // does not depend on the harness's clearing policy.\n"
    "        if (pieces == MAX_SPLIT) {\n"
    "            uint stale_slot = chain * (MAX_SPLIT - 1u) + 1u;\n"
    "            for (uint vi = 0; vi < D; vi++) {\n"
    "                for (uint ki = 0; ki < D; ki++) {\n"
    "                    handoff_states[stale_slot * D * D + vi * D + ki] =\n"
    "                        initial_state[(seq * H + head) * D * D + vi * D + ki] + STALE_OFFSET;\n"
    "                }\n"
    "            }\n"
    "            handoff_flags[stale_slot] = FLAG_READY;\n"
    "        }\n"
)

#: 首 piece 的 ``if (piece == 0u)`` 分支结束、中/末 piece 的 handoff 分支开始处。
_SEED_ANCHOR = "    } else {\n        uint slot = chain * (MAX_SPLIT - 1u) + piece - 1u;"
#: 中 piece 的“发布终态 + READY”分支。
_PUBLISH_ANCHOR = "    if (piece + 1u < pieces) {"


def _seeded(offset: str) -> str:
    """@brief fixture B + 一段旧 READY/旧 state（构造负例的唯一入口）。

    @param offset 旧 state 相对 ``initial_state`` 的运行时偏移表达式（例如 ``0.5f``）。
    @return 变异后的源码。
    """
    seeded = _mutate(PIECE_HANDOFF, _SEED_ANCHOR, STALE_SEED + _SEED_ANCHOR)
    return _mutate(
        seeded,
        "constant constexpr float PROTOCOL_POISON = 1.0e30f;",
        f"constant constexpr float PROTOCOL_POISON = 1.0e30f;\nconstant constexpr float STALE_OFFSET = {offset};",
    )


def _first_case(report: dict) -> dict:
    """@brief 取报告里第一个 case 结论，并断言它不因缺字段而逃过检查。

    @param report schema-4 报告。
    @return case 字典。
    """
    cases = (report.get("correctness") or {}).get("cases") or []
    assert cases, _why(report)
    return cases[0]


def test_fixture_b_a_stale_ready_flag_and_previous_state_are_refused():
    """@brief 上一轮留下的 READY flag + 旧 handoff state 被 consumer 读走：必须数值拒绝。

    构造（只改 fixture B 的 kernel 体，不动 ABI/workspace 声明）：首 piece 在 slot 1 留下“上一轮”
    的 ``FLAG_READY`` 与 ``initial_state + 0.5``；中 piece 本轮**不发布**自己的终态（既不写
    ``handoff_states`` 也不发 flag），于是末 piece 通过 ``READY`` 检查读到那份旧 state。

    三个控制实验，缺一不可：

    * 控制 1：同一份旧 READY/旧 state，但中 piece 正常发布（旧值被覆盖）→ 必须 ``ok``。这说明
      “伪造的 flag/旧值存在”本身不是失败原因。
    * 控制 2：负例必须在 ``scaled_err`` 上失败，且是**有限中等量级**、无 NaN。协议 poison 路径
      （``PROTOCOL_POISON = 1e30``）会给出 1e30 量级（实测溢出成 ``"inf"``）的误差，所以这条把
      “旧 state”与“未发布/未清 flag”区分开：链尾的 flag 后置条件是满足的（两个 slot 都被消费并
      清空）。
    * 控制 3：只改旧 state 的偏移（0.5 → 0.25），同一负例的 ``scaled_err`` 必须跟着变。这证明末
      piece 真的读了这份旧字节，而不是失败与旧值内容无关。
    """
    seeded = _seeded("0.5f")

    # 控制 1：旧值存在但被本轮正常发布覆盖 → 必须通过。
    control = _evaluate(seeded, cases=("canonical",))
    control_report = control.result.report
    assert control_report["status"] == "ok", _why(control_report)

    # 负例：中 piece 不发布本轮状态，末 piece 只能看到旧 READY + 旧 state。
    negative = _mutate(seeded, _PUBLISH_ANCHOR, "    if (piece + 1u < pieces && piece != 1u) {")
    run = _evaluate(negative, cases=("canonical",))
    report = _assert_refused(run)

    assert report["status"] == "incorrect", _why(report)
    first = _first_case(report)
    assert first["case"] == "canonical", first
    assert first["ok"] is False, first
    # 有限且中等量级的偏差：走的是“旧 state”路径，不是 PROTOCOL_POISON（1e30）路径。
    assert first["scaled_err"] != "inf", first["message"]
    stale_err = float(first["scaled_err"])
    assert 1e-3 < stale_err < 1e6, first["message"]
    assert first["nan_count"] == 0, first["message"]

    # 控制 3：只换旧 state 的内容，误差必须跟着变 → 末 piece 真的读了这份旧字节。
    shifted = _mutate(_seeded("0.25f"), _PUBLISH_ANCHOR, "    if (piece + 1u < pieces && piece != 1u) {")
    shifted_report = _assert_refused(_evaluate(shifted, cases=("canonical",)))
    shifted_case = _first_case(shifted_report)
    assert shifted_case["ok"] is False, shifted_case
    shifted_err = float(shifted_case["scaled_err"])
    assert 1e-3 < shifted_err < 1e6, shifted_case["message"]
    assert abs(shifted_err - stale_err) > 0.1 * max(stale_err, shifted_err), (stale_err, shifted_err)
