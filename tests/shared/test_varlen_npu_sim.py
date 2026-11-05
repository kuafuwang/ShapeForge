"""V2 NPU 语义模拟：同一 OpSpec 的 work list、状态协议、对抗调度与 DMA 对齐踩踏（CPU）。

这里**不执行 MSL，也不声称 NPU 真机正确性**。模拟器（``tests/support/npu_sim/varlen_worklist.py``）
复用同一 OpSpec/分段 case 与同一 work-list 规则，把工作项映射到模拟 core，把 handoff 映射成显式
依赖边，并按 target 的 DMA 对齐粒度检查 partial tail 写回是否会踩到下一条 TND 序列。

四类断言：

* 规约：piece 划分、首/中/末 piece 的状态来源、依赖边与 OpSpec 的 ``MAX_SPLIT_PER_CHAIN`` 容量一致；
* 对抗：有限 core 槽位下 consumer-first 必然形成等待环（死锁），逆序但拓扑合法的调度仍能完成；
* 对齐：粗粒度 DMA 写回在未声明 masked write 时被判为踩踏，声明后不再判缺陷；
* parity：两个**独立** adapter（Metal 侧转写 device 规划 + 静态 grid；NPU 侧 host-neutral 规划 + 有限
  core）在同一 HostTarget 桩下 work list 逐字节一致；真实异 target 下 item 全集相同、两侧各自每项恰好
  一次认领、依赖边等价，但 core 分配与 launch 允许不同。另有让打包/依赖边/规划漂移的负对照，证明
  parity 检查真的会抓到漂移，而不是比较同一个函数。

所有报告固定 ``simulated=true`` / ``publication_allowed=false``，不产生时间或 speedup，也不能进入
发布/确认/catalog/brief。清微 NPU-P0 仍 blocked。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct

import pytest

from tcenv.opspec.v2 import VARLEN_CASE_DOMAIN
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tests.support.npu_sim.varlen_worklist import (
    EVIDENCE_SCOPE,
    MEASUREMENT_KIND,
    METAL_STUB,
    NPU_STUB,
    ORDERS,
    PUBLICATION_ALLOWED,
    Piece,
    SIMULATED,
    alignment_clobbers,
    claim_problems,
    dependency_edges,
    empty_chains,
    evidence_markers,
    metal_adapter,
    metal_device_plan,
    npu_adapter,
    npu_host_plan,
    pack_items,
    parity_problems,
    plan_pieces,
    simulate,
    simulate_schedule,
    state_sources,
    work_list_digest,
)

KDA_DIR = REPO_ROOT / "tasks" / "varlen_kda"
MAX_SPLIT = 3
#: mixed 档（seq_lens=[4,8,2]）的 canonical 分段。
MIXED_SEGMENTS = (4, 8, 2)
#: 有限 core 槽位的对抗目标：2 个槽位不足以“先占满所有 consumer”。
TWO_CORE_NPU = replace(NPU_STUB, id="sim-npu-two-cores", cores=2)


def _actual_segments(case: str, seed: int = 7) -> tuple[int, ...]:
    """@brief 由题目的 reference 生成器拿到该 case 的**实际**分段（与 Metal 评测同一份输入）。

    @param case L1 correctness case。
    @param seed 输入种子。
    @return 每条序列的 token 数。
    """
    contract = load_task_contract(KDA_DIR, dtype="fp16", size="mixed")
    inputs = contract.reference.generate(contract.instance, case, seed)
    cu = [int(value) for value in inputs["cu_seqlens"].tolist()]
    return tuple(end - begin for begin, end in zip(cu, cu[1:]))


def test_the_piece_plan_and_state_roles_cover_every_sequence_and_head():
    """@brief piece 划分恰好覆盖每链全部 token，首/中/末 piece 的状态来源正确。"""
    items = plan_pieces(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT)
    roles = state_sources(items)

    # 每链 (sequence, head) 的 token 覆盖必须等于该段长度。
    for seq, length in enumerate(MIXED_SEGMENTS):
        for head in range(2):
            chain = seq * 2 + head
            covered = sum(item.valid for item in items if item.chain == chain)
            assert covered == length
            pieces = {item.piece for item in items if item.chain == chain}
            assert len(pieces) == min(MAX_SPLIT, length)

    first = [role for role in roles if role["piece"] == 0]
    middle = [role for role in roles if role["piece"] > 0 and role["state_sink"] == "handoff"]
    last = [role for role in roles if role["state_sink"] == "final_state"]
    assert first and all(role["state_source"] == "initial_state" for role in first)
    assert middle and all(role["state_source"] == "handoff" for role in middle)
    assert last and all(role["state_source"] == "handoff" for role in last)
    # 只有 3 个 piece 的链才有中间 piece：seq0(4) 与 seq1(8) × 两个 head = 4 个。
    assert len(middle) == 4

    # 两字打包与 Metal fixture 的布局一致：word0 = begin | valid<<16。
    words = [item.words() for item in items]
    assert pack_items(items) == b"".join(
        int.to_bytes(word, 4, "little") for pair in words for word in pair
    )
    assert len(set(words)) == len(words)


def test_one_dependency_edge_per_piece_boundary_and_nothing_else():
    """@brief 依赖图只有同一链相邻 piece 的边：没有跨链、跨 head 的隐藏依赖。"""
    items = plan_pieces(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT)
    edges = dependency_edges(items)

    expected = 0
    for length in MIXED_SEGMENTS:
        expected += (min(MAX_SPLIT, length) - 1) * 2  # 两个 head
    assert len(edges) == expected
    for producer, consumer in edges:
        left, right = items[producer], items[consumer]
        assert left.chain == right.chain
        assert right.piece == left.piece + 1


@pytest.mark.parametrize("case", VARLEN_CASE_DOMAIN)
def test_the_simulator_plans_from_the_real_case_segmentation(case):
    """@brief 每个 L1 case 的实际分段都能规划：总 token 覆盖不变，空段不产生 piece。"""
    segments = _actual_segments(case)
    items = plan_pieces(segments, heads=2, max_split=MAX_SPLIT)

    assert sum(segments) == 14
    for seq, length in enumerate(segments):
        for head in range(2):
            covered = sum(item.valid for item in items if item.chain == seq * 2 + head)
            assert covered == length
    assert set(empty_chains(segments, 2)) == {
        seq * 2 + head for seq, length in enumerate(segments) if length == 0 for head in range(2)
    }


def test_a_limited_core_slot_consumer_first_schedule_deadlocks():
    """@brief 有限 core 槽位 + consumer-first 必然形成等待环；拓扑顺序仍能完成。"""
    items = plan_pieces(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT)

    for order in ("stage", "reverse_stage"):
        schedule = simulate_schedule(items, TWO_CORE_NPU, order=order)
        assert schedule.deadlock is False
        assert len(schedule.completed) == len(items)
        assert claim_problems(items, schedule.claims) == []
        assert schedule.waits == 0  # 拓扑顺序从不先占 consumer

    adversarial = simulate_schedule(items, TWO_CORE_NPU, order="consumer_first")
    assert adversarial.deadlock is True
    assert adversarial.waits > 0
    assert len(adversarial.completed) < len(items)

    report = simulate(
        MIXED_SEGMENTS, heads=2, dim=8, max_split=MAX_SPLIT, target=TWO_CORE_NPU, order="consumer_first"
    )
    assert report["ok"] is False
    assert any("deadlocked" in defect for defect in report["defects"])


def test_a_coarse_dma_granule_can_clobber_the_next_tnd_sequence():
    """@brief 未声明 masked write 时，512B DMA 写回会在 partial tail 处踩到下一条序列。"""
    segments = (4, 8, 2)
    items = plan_pieces(segments, heads=2, max_split=MAX_SPLIT)
    clobbered = alignment_clobbers(
        items, segments, heads=2, dim=8, elem_bytes=2, alignment=512, masked_write=False
    )
    assert clobbered
    assert all(entry["clobber_bytes"] > 0 for entry in clobbered)
    # 踩到的确实**不是**本 piece 所在序列（跨 TND 边界）。
    for entry in clobbered:
        assert entry["clobbers_sequence"] != entry["sequence"]

    # 声明 masked write / RMW 后不再判缺陷；byte-addressable 目标（Metal 的谓词化写）也不踩。
    assert alignment_clobbers(
        items, segments, heads=2, dim=8, elem_bytes=2, alignment=512, masked_write=True
    ) == []
    assert alignment_clobbers(
        items, segments, heads=2, dim=8, elem_bytes=2, alignment=4, masked_write=False
    ) == []

    report = simulate(segments, heads=2, dim=8, max_split=MAX_SPLIT, target=NPU_STUB)
    assert any("cross a TND sequence boundary" in defect for defect in report["defects"])
    masked = simulate(
        segments, heads=2, dim=8, max_split=MAX_SPLIT, target=replace(NPU_STUB, masked_write=True)
    )
    assert masked["ok"] is True, masked["defects"]
    assert masked["alignment"]["clobbers"] == []
    # aligned 目标（模拟 stub）也不会被判踩踏。
    assert simulate(segments, heads=2, dim=8, max_split=MAX_SPLIT, target=METAL_STUB)["ok"] is True


def test_the_scratch_budget_and_item_table_location_follow_the_target(tmp_path):
    """@brief 小 item table 进 scratch，大表走 global；per-core scratch 超预算时判缺陷。"""
    items = plan_pieces(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT)
    report = simulate(MIXED_SEGMENTS, heads=2, dim=8, max_split=MAX_SPLIT, target=NPU_STUB)

    assert report["scratch"]["item_table_location"] == "scratch"
    assert report["scratch"]["peak_bytes"] == 8 * 8 * 4 + len(items) * 8
    assert report["scratch"]["fits"] is True

    # 80 条链（40 段 × 2 head）在 64 项缓存容量之上 -> global fallback。
    wide = simulate((2,) * 40, heads=2, dim=8, max_split=MAX_SPLIT, target=NPU_STUB)
    assert wide["scratch"]["item_table_location"] == "global"
    assert wide["scratch"]["peak_bytes"] == 8 * 8 * 4

    tiny = simulate(
        MIXED_SEGMENTS, heads=2, dim=8, max_split=MAX_SPLIT,
        target=replace(NPU_STUB, id="sim-npu-tiny", scratch_bytes=128),
    )
    assert tiny["scratch"]["fits"] is False
    assert any("scratch needs" in defect for defect in tiny["defects"])


def test_the_balanced_split_never_emits_an_empty_piece():
    """@brief len=4 的链被切成 2/1/1，而不是 2/2/0；三条独立实现给出同一 item 全集。"""
    assert [(it.begin, it.valid) for it in plan_pieces((4, 8, 2), 2, MAX_SPLIT) if it.chain == 0] == [
        (0, 2),
        (2, 1),
        (3, 1),
    ]
    assert [(it.begin, it.valid) for it in plan_pieces((4, 8, 2), 2, MAX_SPLIT) if it.chain == 2] == [
        (4, 3),
        (7, 3),
        (10, 2),
    ]
    assert [(it.begin, it.valid) for it in plan_pieces((4, 8, 2), 2, MAX_SPLIT) if it.chain == 4] == [
        (12, 1),
        (13, 1),
    ]

    for segments in (MIXED_SEGMENTS, (4,), (1,), (16,), (2, 2, 2, 2), (5, 3), (0, 4, 0)):
        reference = plan_pieces(segments, 2, MAX_SPLIT)
        assert metal_device_plan(segments, 2, MAX_SPLIT) == reference
        assert npu_host_plan(segments, 2, MAX_SPLIT) == reference
        assert all(item.valid > 0 for item in reference), segments
        # 每链的 piece 首尾相接，覆盖该序列的整个 token 区间。
        for seq, length in enumerate(segments):
            chain_pieces = [item for item in reference if item.chain == seq * 2]
            if length == 0:
                assert chain_pieces == []
                continue
            assert sum(item.valid for item in chain_pieces) == length
            cursor = chain_pieces[0].begin
            for item in chain_pieces:
                assert item.begin == cursor
                cursor += item.valid


def test_an_empty_writeback_has_no_dma_span_and_cannot_clobber():
    """@brief valid=0 不写任何字节：没有 DMA span，也就不能把邻居判成被踩踏。"""
    segments = (4, 8, 2)
    empty = Piece(chain=0, piece=2, begin=3, valid=0, pieces=3)
    span = dict(heads=2, dim=8, elem_bytes=2, alignment=512, masked_write=False)
    assert alignment_clobbers([empty], segments, **span) == []
    # 对照：同一起点写出一个 token 时确实会踩到下一条序列，证明上面的空集不是因为对齐把 span 缩没了。
    assert alignment_clobbers([replace(empty, valid=1)], segments, **span)

    report = simulate(segments, heads=2, dim=8, max_split=MAX_SPLIT, target=NPU_STUB)
    assert report["alignment"]["empty_writebacks"] == []
    assert all(item["valid"] > 0 for item in report["work_list"]["items"])
    assert all(entry["clobber_bytes"] > 0 for entry in report["alignment"]["clobbers"])


def test_the_two_adapters_agree_byte_for_byte_on_one_host_target_stub():
    """@brief 同一 HostTarget 桩（只有 kind 不同）下，Metal/NPU 两条 adapter 的 work list 逐字节一致。"""
    metal = metal_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, target=METAL_STUB)
    same_parameters = replace(METAL_STUB, kind="npu", id="sim-npu-same-stub")
    npu = npu_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, dim=8, target=same_parameters)

    # 两个 adapter 各自有独立的规划实现，但同一组 target 参数下字节必须相同。
    assert metal.items == npu.items
    assert metal.words == npu.words
    assert metal.edges == npu.edges
    assert parity_problems(metal, npu) == []
    # 两侧各自“每项恰好一次认领”；空槽位不算认领。
    assert metal.claim_problems() == []
    assert npu.claim_problems() == []

    # kind 不参与规划/打包：只改 kind 的 target 不改变任何一侧的 work list。
    for kind in ("metal", "npu"):
        assert (
            metal_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, target=replace(METAL_STUB, kind=kind)).words
            == metal.words
        )
        assert (
            npu_adapter(
                MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, dim=8, target=replace(same_parameters, kind=kind)
            ).words
            == npu.words
        )


def test_two_real_targets_share_the_work_list_but_not_the_core_assignment():
    """@brief 真实异 target：work list/依赖边等价、两侧唯一认领；core 分配/launch 不同且不参与比较。"""
    four_cores = replace(NPU_STUB, id="sim-npu-four-cores", cores=4)
    metal = metal_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, target=METAL_STUB)
    npu = npu_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, dim=8, target=four_cores)

    assert parity_problems(metal, npu) == []
    assert metal.words == npu.words
    assert metal.edges == npu.edges
    assert metal.target.cores != npu.target.cores
    assert npu.launch["cores"] == 4 and npu.launch["waves"] == -(-len(npu.items) // 4) > 1

    # Dyn-0 几何：grid 是 ABI 容量（每条链 max_split 个槽位），不是运行时 item 数；空槽位读 count 退出。
    chains = len(MIXED_SEGMENTS) * 2
    assert metal.launch["grid"] == chains * MAX_SPLIT
    assert len(metal.items) < metal.launch["grid"]
    assert metal.launch["runtime_items"] == len(metal.items)
    assert metal.launch["idle_slots"] == metal.launch["grid"] - len(metal.items)
    assert metal.launch["stages"] == 4  # plan + piece0/1/2，静态阶段数
    assert len({item.piece for item in metal.items}) == MAX_SPLIT
    # 空槽位不认领任何项，但它们仍然在几何里（不是把 grid 缩到 item 数）。
    empty_slots = [slot for slot, cases in metal.claims if not cases]
    assert len(empty_slots) == metal.launch["idle_slots"] and empty_slots

    # 同一份 work list 被映射到不同的执行几何：Metal 每个 item 一个静态 TG 槽位，NPU 先把 4 个核占满。
    metal_assignment = [cases for _, cases in metal.claims if cases]
    npu_assignment = [cases for _, cases in npu.claims if cases]
    assert metal_assignment != npu_assignment
    # 不要求 grid 相同，也从不把它写进 parity 条件。
    assert metal.launch["grid"] != npu.launch["cores"]
    # 两侧都恰好一次认领，且覆盖 item 全集。
    assert metal.claim_problems() == []
    assert npu.claim_problems() == []
    for adapter in (metal, npu):
        claimed = [item for _, cases in adapter.claims for item in cases]
        assert sorted(claimed) == list(range(len(adapter.items)))
    # target 特有的约束缺陷允许只出现在一侧（粗粒度 DMA 只会在 NPU 侧报踩踏）。
    assert npu.defects and metal.defects == ()


def test_the_parity_check_catches_a_drifted_work_list_edge_or_claim():
    """@brief 负对照：打包/依赖边/规划/认领任一漂移都必须被 parity 抓住。"""
    metal = metal_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, target=METAL_STUB)
    npu = npu_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, dim=8, target=NPU_STUB)
    assert parity_problems(metal, npu) == []

    # (a) 打包漂移：两字顺序对调（item 集合没变，字节变了）。
    swapped = replace(
        npu, words=b"".join(struct.pack("<II", *reversed(item.words())) for item in npu.items)
    )
    assert any("work list bytes differ" in problem for problem in parity_problems(metal, swapped))
    # 只翻转一个 bit 也抓到。
    flipped = bytearray(npu.words)
    flipped[0] ^= 1
    assert any("work list bytes differ" in problem for problem in parity_problems(metal, replace(npu, words=bytes(flipped))))
    # (b) 依赖边漂移：丢掉一条边。
    dropped = replace(npu, edges=npu.edges[1:])
    assert any("dependency edges differ" in problem for problem in parity_problems(metal, dropped))
    # (c) 规划漂移（真实输入不同，不是构造漂移）：一侧用 max_split+1。
    drifted = npu_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT + 1, dim=8, target=NPU_STUB)
    assert any("work list bytes differ" in problem for problem in parity_problems(metal, drifted))
    # (d) 认领漂移：同一项被两个 core 认领；两侧都要被检查。
    double = replace(npu, claims=((0, (0,)), (1, (0,))) + npu.claims[2:])
    assert any("npu: an item was claimed by more than one core" in p for p in parity_problems(metal, double))
    metal_double = replace(metal, claims=((0, (0,)), (1, (0,))) + metal.claims[2:])
    assert any("metal: an item was claimed by more than one core" in p for p in parity_problems(metal_double, npu))
    # 有项没被任何 core 认领。
    unclaimed = replace(npu, claims=tuple((slot, ()) for slot, _ in npu.claims))
    assert any("npu: items never claimed" in problem for problem in parity_problems(metal, unclaimed))
    # (e) core 分配本身不属于 parity：换一种合法的认领顺序仍然通过。
    permuted = replace(npu, claims=tuple(reversed(npu.claims)))
    assert permuted.claim_problems() == []
    assert parity_problems(metal, permuted) == []


@pytest.mark.parametrize("order", ORDERS)
def test_every_simulated_report_is_marked_simulated_and_not_publishable(order):
    """@brief 所有模拟报告固定 ``simulated=true``/``publication_allowed=false``，且没有时间/加速比。"""
    report = simulate(MIXED_SEGMENTS, heads=2, dim=8, max_split=MAX_SPLIT, target=NPU_STUB, order=order)

    assert report["simulated"] is SIMULATED is True
    assert report["publication_allowed"] is PUBLICATION_ALLOWED is False
    assert report["measurement_kind"] == MEASUREMENT_KIND == "simulated"
    assert report["evidence_scope"] == EVIDENCE_SCOPE
    for field in ("latency", "speedup", "confirmed", "catalog", "end_to_end_wall_ns", "device_kernel_ns"):
        assert field not in report, field
    # 模拟报告没有“静默通过”：缺陷必须显式列出，ok 与 defects 一致。
    assert report["ok"] == (not report["defects"])


def test_every_adapter_report_document_is_marked_simulated_and_not_publishable():
    """@brief adapter 自己的 ``document()`` 也必须带固定标记，不靠 ``simulate()`` 包装。

    ``metal_adapter``/``npu_adapter`` 都是 Python 模拟（前者只是设备规划 stage 的转写，没有设备
    执行）。直接拿 adapter 的文档时标记不能消失——这正是只测 ``simulate()`` 时会漏掉的路径。
    """
    metal = metal_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, target=METAL_STUB)
    npu = npu_adapter(MIXED_SEGMENTS, heads=2, max_split=MAX_SPLIT, dim=8, target=NPU_STUB)
    wrapper = simulate(MIXED_SEGMENTS, heads=2, dim=8, max_split=MAX_SPLIT, target=NPU_STUB)

    for adapter in (metal, npu):
        document = adapter.document()
        assert document["adapter"] == adapter.name
        # 先用“缺哪个键”报错，而不是直接下标：缺失时不能以一个 KeyError 结束。
        missing = [key for key in evidence_markers() if key not in document]
        assert not missing, f"{adapter.name} adapter document is missing {missing}"
        assert document["simulated"] is SIMULATED is True, adapter.name
        assert document["publication_allowed"] is PUBLICATION_ALLOWED is False, adapter.name
        assert document["measurement_kind"] == MEASUREMENT_KIND == "simulated", adapter.name
        assert document["evidence_scope"] == EVIDENCE_SCOPE, adapter.name
        for field in ("latency", "speedup", "confirmed", "catalog", "end_to_end_wall_ns", "device_kernel_ns"):
            assert field not in document, (adapter.name, field)
        # 两条报告路径同源：adapter 文档与 wrapper 报告的标记不能各写一份。
        for key in evidence_markers():
            assert document[key] == wrapper[key] == evidence_markers()[key], (adapter.name, key)

    # adapter 文档不是另一份规划：它就是 wrapper 里的同一份 work list 证据。
    assert npu.document()["work_list"]["digest"] == wrapper["work_list"]["digest"]
    assert npu.document()["work_list"]["count"] == wrapper["work_list"]["count"]
    assert npu.document()["work_list"]["edges"] == wrapper["work_list"]["edges"]


def test_a_schedule_document_taken_directly_is_marked_simulated_and_not_publishable():
    """@brief 直接取 ``simulate_schedule(...).document()`` 也不能得到未标记的调度结果。

    ``Schedule`` 是调用方可以自己拿到的公开返回类型，所以“它只是 ``simulate()`` 的片段”是解释，
    不是约束：标记必须写在文档里。这里对全部调度顺序直接验证，不经 wrapper。
    """
    items = plan_pieces(MIXED_SEGMENTS, 2, MAX_SPLIT)
    for order in ORDERS:
        schedule = simulate_schedule(items, TWO_CORE_NPU, order=order)
        document = schedule.document()
        missing = [key for key in evidence_markers() if key not in document]
        assert not missing, f"schedule document ({order}) is missing {missing}"
        assert document["simulated"] is SIMULATED is True, order
        assert document["publication_allowed"] is PUBLICATION_ALLOWED is False, order
        assert document["measurement_kind"] == MEASUREMENT_KIND == "simulated", order
        assert document["evidence_scope"] == EVIDENCE_SCOPE, order
        for field in ("latency", "speedup", "confirmed", "catalog", "end_to_end_wall_ns", "device_kernel_ns"):
            assert field not in document, (order, field)
        # 标记是加在调度事实上而不是替换它：事件/等待/死锁/完成集仍在。
        assert document["order"] == schedule.order and document["cores"] == schedule.cores
        assert document["waits"] == schedule.waits and document["deadlock"] == schedule.deadlock
        assert document["completed"] == list(schedule.completed)
        assert len(document["events"]) == len(schedule.events)

    # wrapper 里的 ``schedule`` 与直接取的是同一份文档（同一个 target/order/规划）。
    wrapper = simulate(MIXED_SEGMENTS, heads=2, dim=8, max_split=MAX_SPLIT, target=TWO_CORE_NPU, order="stage")
    assert wrapper["schedule"] == simulate_schedule(items, TWO_CORE_NPU, order="stage").document()


def test_production_packages_do_not_reference_the_simulator():
    """@brief 生产组合根不 import 模拟器：模拟结果不可能进入发布/确认/catalog/brief。"""
    needles = ("varlen_worklist", "npu_sim.varlen")
    scanned = 0
    for package in ("tcenv", "taic", "tasks"):
        for path in (REPO_ROOT / package).rglob("*"):
            if not path.is_file() or path.suffix not in (".py", ".json", ".toml", ".md"):
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in needles:
                assert needle not in text, f"{path} references the test-only simulator"
    # 正向对照：扫描路径拼错时不能靠“没扫到任何文件”静默通过。
    assert scanned > 100, f"only {scanned} production files were scanned"

    module = Path(__file__).resolve().parents[1] / "support" / "npu_sim" / "varlen_worklist.py"
    source = module.read_text(encoding="utf-8")
    assert "tcenv.backends" not in source and "evaluate_candidate" not in source
    assert work_list_digest(()) == work_list_digest(())
