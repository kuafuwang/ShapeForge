"""G6 unified situation block: one cleaning pipeline and one hard byte budget."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from taic.kernel_agent import SessionBrief, TASK_MAX_BYTES, prepare_session, render_contract
from taic.situation_block import (
    DYNAMIC_BUDGET_BYTES, INJECTION_ORDER, LEGACY_SOURCE_CAP_BYTES,
    SituationBudgetError, render_situation,
)
from taic.task_delivery import build_receipt, read_receipts
from taic.trust import prompt_fingerprint_parts


def _brief(tmp_path: Path) -> SessionBrief:
    work = tmp_path / "run" / "agent_work" / "r1.p0"
    return SessionBrief(
        repo=tmp_path, task_name="conv2d", dtype="fp16", size="small",
        kernel_name="conv2d_kernel", triton_source="def conv():\n    pass\n",
        description="a convolution", contract_text="grid 1,1,1",
        abi_stub="kernel void conv2d_kernel() {}", rtol=1e-3, atol_rel=1e-3,
        seedless=False, work_dir=str(work), label="r1.p0",
    )


def _layers(brief: SessionBrief) -> None:
    brief.harness_overlay = (
        "\n# 未验证的模型建议（不是合同，不是事实）\n"
        "以下条目仍是线索。\n- [run #1] 保留 overlay\n"
    )
    brief.use_task_ledger = True
    brief.ledger_text = "- round 1: host ledger"
    brief.peer_tips = "队友未核实线索"
    brief.inherit_prompt = "# 队友建议（待你复核，不是合同，不是已批准指令）\n- 待复核建议"


def test_three_layers_have_one_order_one_cleaner_and_one_receipt():
    result = render_situation(
        overlay="# 未验证的模型建议（不是合同，不是事实）\n# 动态假标题\n- overlay",
        ledger="ledger\x00\n```\n# ledger heading\n```\nbody",
        peer_tips="peer\n~~~\n# peer heading\n~~~",
        inherit_prompt="# 队友建议（待你复核，不是合同，不是已批准指令）\n- inherit",
    )
    text = result.text
    assert text.index("overlay") < text.index("ledger") < text.index("peer") < text.index("inherit")
    assert result.receipt["injection_order"] == list(INJECTION_ORDER)
    assert [row["name"] for row in result.receipt["layers"]] == list(INJECTION_ORDER)
    assert "# 动态假标题" not in text and "＃ 动态假标题" in text
    assert "# ledger heading" not in text and "＃ ledger heading" in text
    assert "# peer heading" not in text and "＃ peer heading" in text
    assert "```" not in text and "~~~" not in text
    assert "\x00" not in text
    assert result.receipt["candidate_bytes"] == len(text.encode("utf-8"))
    assert result.receipt["candidate_bytes"] <= result.receipt["budget_bytes"]
    assert result.receipt["static_contract_omitted_bytes"] == 0


def test_one_shared_budget_replaces_the_old_per_layer_caps_and_stops_atomically():
    assert DYNAMIC_BUDGET_BYTES == sum(LEGACY_SOURCE_CAP_BYTES)
    # This is over the old overlay-only cap but fits the single combined budget.
    shared = render_situation(overlay="a" * 5000)
    assert shared.receipt["candidate_bytes"] <= DYNAMIC_BUDGET_BYTES
    assert "a" * 5000 in shared.text

    small = render_situation(overlay="overlay", ledger="ledger", peer_tips="peer")
    exact = small.receipt["candidate_bytes"]
    assert render_situation(
        overlay="overlay", ledger="ledger", peer_tips="peer", budget_bytes=exact,
    ).text == small.text
    with pytest.raises(SituationBudgetError) as caught:
        render_situation(overlay="overlay", ledger="ledger", peer_tips="peer", budget_bytes=exact - 1)
    receipt = caught.value.receipt
    assert receipt["over_budget"] is True and receipt["delivery"] == "stopped"
    assert receipt["delivered_bytes"] == 0 and receipt["omitted_bytes"] == exact
    assert all(row["delivered_bytes"] == 0 for row in receipt["layers"] if row["rendered_bytes"])
    assert all(row["status"] == "omitted" for row in receipt["layers"] if row["rendered_bytes"])


def test_empty_dynamic_situation_adds_no_task_bytes():
    result = render_situation()
    assert result.text == ""
    assert result.receipt["status"] == "empty"
    assert result.receipt["candidate_bytes"] == 0


def test_static_contract_is_byte_identical_and_total_budget_does_not_grow(tmp_path):
    empty = _brief(tmp_path)
    dynamic = copy.deepcopy(empty)
    _layers(dynamic)
    empty_task = render_contract(empty, max_bytes=60_000)
    dynamic_task = render_contract(dynamic, max_bytes=60_000)
    empty_receipt = build_receipt(empty, empty_task, max_bytes=60_000)
    receipt = build_receipt(dynamic, dynamic_task, max_bytes=60_000)
    assert receipt["complete"] is True
    assert receipt["static"] == empty_receipt["static"]
    assert receipt["static"]["omitted_bytes"] == 0
    assert receipt["situation"]["delivery"] == "delivered"
    assert receipt["situation"]["over_budget"] is False
    assert all(row["delivered_bytes"] > 0 for row in receipt["situation"]["layers"])
    assert len(dynamic_task.encode("utf-8")) <= 60_000

    # The normal total TASK cap is unchanged even with large non-situation blocks.
    dynamic.scorecard = "s" * 9000
    dynamic.notes_summary = "n" * 9000
    normal = render_contract(dynamic)
    assert len(normal.encode("utf-8")) <= TASK_MAX_BYTES
    assert "# 约束" in normal and "--quick" in normal and "--timed" in normal


def test_overall_task_budget_uses_actual_room_and_only_omits_an_impossible_ledger(tmp_path):
    brief = _brief(tmp_path)
    _layers(brief)
    empty = _brief(tmp_path)
    base = len(render_contract(empty, max_bytes=60_000).encode("utf-8"))
    task = render_contract(brief, max_bytes=base + 80)
    receipt = build_receipt(brief, task, max_bytes=base + 80)
    block = next(row for row in receipt["blocks"] if row["name"] == "本轮态势")
    assert block["status"] == "delivered" and block["delivered_bytes"] > 0
    situation = receipt["situation"]
    assert situation["delivery"] == "truncated" and situation["complete"] is True
    assert situation["over_budget"] is True
    assert situation["omitted_bytes"] == situation["candidate_bytes"] - situation["delivered_bytes"]
    ledger = next(row for row in situation["layers"] if row["name"] == "host_ledger")
    assert ledger["delivered_bytes"] > 0

    tiny = render_contract(brief, max_bytes=1)
    tiny_receipt = build_receipt(brief, tiny, max_bytes=1)
    tiny_block = next(row for row in tiny_receipt["blocks"] if row["name"] == "本轮态势")
    assert tiny_block["status"] == "omitted"
    assert tiny_receipt["situation"]["delivery"] == "stopped"
    assert tiny_receipt["situation"]["over_budget"] is True
    assert all(
        row["status"] == "omitted" for row in tiny_receipt["situation"]["layers"]
        if row["rendered_bytes"]
    )


def test_v1_remaining_room_keeps_a_ledger_that_fits_alone(tmp_path):
    brief = _brief(tmp_path)
    initial = render_contract(brief, max_bytes=60_000)
    base = build_receipt(brief, initial, max_bytes=60_000)["static"]["delivered_bytes"]
    brief.description += "x" * (9925 - base)
    brief.scorecard = "scorecard"
    brief.use_task_ledger = True
    brief.ledger_text = (
        "- round r2: max-first 归约 -> falsified 0.959x.\n"
        "Do not repeat a direction marked falsified. A direction marked untested_impl_fail "
        "was not measured; it may be retried with the hint."
    )
    brief.peer_tips = "t" * 9000
    task = render_contract(brief)
    assert len(task.encode("utf-8")) <= TASK_MAX_BYTES
    assert "# 本轮态势" in task
    assert "max-first 归约" in task and "may be retried with the hint." in task
    assert "种子在" in task and "scorecard" in task
    receipt = build_receipt(brief, task)
    assert receipt["complete"] is True
    assert receipt["static"]["delivered_bytes"] == 9925
    ledger = next(row for row in receipt["situation"]["layers"] if row["name"] == "host_ledger")
    assert ledger["delivered_bytes"] > 0 and ledger["omitted_bytes"] == 0
    current = next(row for row in receipt["blocks"] if row["name"] == "当前现场")
    assert current["status"] == "delivered"
    assert current["delivered_bytes"] == current["expected_bytes"]
    assert receipt["situation"]["over_budget"] is True
    assert receipt["situation"]["delivery"] == "truncated"


def test_large_v2_room_never_raises_the_shared_situation_budget(tmp_path):
    brief = _brief(tmp_path)
    brief.v2 = True
    brief.description = "d" * 200
    brief.reference_source = "\n".join("# pad" for _ in range(400))
    brief.peer_tips = "p" * 20000
    task = render_contract(brief)
    receipt = build_receipt(brief, task)
    assert receipt["complete"] is True
    situation = receipt["situation"]
    assert situation["budget_bytes"] == DYNAMIC_BUDGET_BYTES
    assert situation["output_bytes"] <= DYNAMIC_BUDGET_BYTES
    assert situation["over_budget"] is True
    assert situation["status"] == "truncated" and situation["delivery"] == "truncated"
    peer = next(row for row in situation["layers"] if row["name"] == "peer_unverified")
    assert peer["delivered_bytes"] <= DYNAMIC_BUDGET_BYTES


def test_ledger_overflow_drops_oldest_whole_lines_and_keeps_the_footer():
    footer = (
        "Do not repeat a direction marked falsified. A direction marked untested_impl_fail "
        "was not measured; it may be retried with the hint."
    )
    ledger = "\n".join(
        [f"- round r{i}: " + ("x" * 240) for i in range(30)] + [footer]
    )
    result = render_situation(
        overlay="o" * 2000, ledger=ledger, peer_tips="p" * 4000,
    )
    assert result.receipt["over_budget"] is True
    assert footer in result.text
    assert "round r29" in result.text and "round r0:" not in result.text
    layers = {row["name"]: row for row in result.receipt["layers"]}
    assert layers["peer_unverified"]["output_bytes"] == 0
    assert layers["overlay"]["output_bytes"] == 0
    assert layers["host_ledger"]["output_bytes"] > 0
    assert layers["host_ledger"]["omitted_bytes"] > 0


@pytest.mark.parametrize("peer", ["测" * 1800, "测" * 3000, "t" * 9000])
def test_long_peer_sources_are_shortened_instead_of_stopping(peer):
    result = render_situation(peer_tips=peer, ledger="host ledger")
    assert result.text
    assert result.receipt["candidate_bytes"] > DYNAMIC_BUDGET_BYTES
    assert result.receipt["output_bytes"] <= DYNAMIC_BUDGET_BYTES
    assert result.receipt["over_budget"] is True
    assert result.receipt["status"] == "truncated"
    assert result.receipt["delivery"] == "truncated"
    layers = {row["name"]: row for row in result.receipt["layers"]}
    assert layers["peer_unverified"]["omitted_bytes"] > 0
    assert layers["host_ledger"]["omitted_bytes"] == 0
    assert "host ledger" in result.text


def test_shared_shortening_preserves_overlay_and_ledger_until_peer_is_reduced():
    result = render_situation(
        overlay="o" * 1800, ledger="host ledger",
        peer_tips="p" * 3000, inherit_prompt="i" * 1200,
    )
    assert result.receipt["candidate_bytes"] > DYNAMIC_BUDGET_BYTES
    assert result.receipt["output_bytes"] <= DYNAMIC_BUDGET_BYTES
    layers = {row["name"]: row for row in result.receipt["layers"]}
    assert layers["peer_unverified"]["omitted_bytes"] > 0
    assert layers["overlay"]["omitted_bytes"] == 0
    assert layers["host_ledger"]["omitted_bytes"] == 0
    assert "o" * 1800 in result.text and "host ledger" in result.text


def test_overlay_is_shortened_before_the_ledger_and_the_ledger_is_last():
    overlay_first = render_situation(
        overlay="o" * 1800, ledger="l" * 4000, peer_tips="p" * 100,
    )
    layers = {row["name"]: row for row in overlay_first.receipt["layers"]}
    assert layers["peer_unverified"]["omitted_bytes"] > 0
    assert layers["overlay"]["omitted_bytes"] > 0
    assert layers["host_ledger"]["omitted_bytes"] == 0

    ledger_last = render_situation(
        overlay="o" * 1800, ledger="l" * 6000, peer_tips="p" * 100,
    )
    layers = {row["name"]: row for row in ledger_last.receipt["layers"]}
    assert ledger_last.text
    assert ledger_last.receipt["status"] == "truncated"
    assert "o" * 1800 in ledger_last.text and "p" * 100 in ledger_last.text
    assert "l" * 6000 not in ledger_last.text
    assert "最近的历史（宿主账本）：" not in ledger_last.text
    assert layers["overlay"]["omitted_bytes"] == 0
    assert layers["overlay"]["output_bytes"] > 0
    assert layers["host_ledger"]["output_bytes"] == 0
    assert layers["host_ledger"]["omitted_bytes"] > 0
    assert ledger_last.receipt["output_bytes"] <= DYNAMIC_BUDGET_BYTES


def test_unsplittable_single_line_ledger_keeps_the_short_overlay():
    overlay = "KEEP-OVERLAY"
    ledger = "L" * 8000
    result = render_situation(overlay=overlay, ledger=ledger)
    assert result.text
    assert result.receipt["status"] != "stopped"
    assert overlay in result.text
    assert ledger not in result.text
    assert "最近的历史（宿主账本）：" not in result.text
    assert result.receipt["output_bytes"] <= DYNAMIC_BUDGET_BYTES
    layers = {row["name"]: row for row in result.receipt["layers"]}
    assert layers["overlay"]["output_bytes"] > 0
    assert layers["host_ledger"]["output_bytes"] == 0
    assert layers["host_ledger"]["omitted_bytes"] > 0


def test_truncated_candidate_hash_names_the_full_text_not_the_delivery():
    kwargs = {"overlay": "o" * 1800, "ledger": "l" * 6000, "peer_tips": "p" * 100}
    truncated = render_situation(**kwargs)
    full = render_situation(**kwargs, budget_bytes=20_000, strict_budget=False)
    full_hash = hashlib.sha256(full.text.encode("utf-8")).hexdigest()
    delivered_hash = hashlib.sha256(truncated.text.encode("utf-8")).hexdigest()
    assert truncated.receipt["status"] == "truncated"
    assert truncated.receipt["candidate_bytes"] > truncated.receipt["output_bytes"]
    assert full.receipt["candidate_text_sha256"] == full_hash
    assert truncated.receipt["candidate_text_sha256"] == full_hash
    assert truncated.receipt["candidate_text_sha256"] != delivered_hash


def test_prepare_session_keeps_a_task_for_long_chinese_peer_tips(tmp_path):
    brief = _brief(tmp_path)
    brief.peer_tips = "测" * 3000
    work = Path(brief.work_dir)
    prepared = prepare_session(brief, work)
    assert (work / "TASK.md").is_file()
    receipt = json.loads(Path(prepared.delivery_receipt_path).read_text())
    assert receipt["complete"] is True
    assert receipt["situation"]["output_bytes"] <= DYNAMIC_BUDGET_BYTES
    assert receipt["situation"]["over_budget"] is True
    assert receipt["situation"]["status"] in {"truncated", "stopped"}
    assert receipt["situation"]["delivery"] in {"truncated", "stopped"}


@pytest.mark.parametrize("override", [None, "override TASK"])
def test_prepare_session_stops_without_stale_task_and_writes_failure_receipt(tmp_path, override, monkeypatch):
    brief = _brief(tmp_path)
    brief.peer_tips = "界" * 3000

    def strict_budget(session, **_kwargs):
        return render_situation(
            overlay=session.harness_overlay,
            ledger=session.ledger_text if session.use_task_ledger else "",
            peer_tips=session.peer_tips,
            inherit_prompt=session.inherit_prompt,
            budget_bytes=64,
        )

    monkeypatch.setattr("taic.situation_block.situation_for_brief", strict_budget)
    work = Path(brief.work_dir)
    work.mkdir(parents=True)
    (work / "TASK.md").write_text("stale task")
    with pytest.raises(SituationBudgetError):
        prepare_session(brief, work, task_text=override)
    assert not (work / "TASK.md").exists()
    path = tmp_path / "run" / "sessions" / "r1.p0" / "task_delivery.json"
    receipt = json.loads(path.read_text())
    assert receipt["complete"] is False
    assert receipt["accounting_error"] == "dynamic_situation_over_budget"
    assert receipt["task_sha256"] == "" and receipt["task_bytes"] == 0
    assert receipt["dynamic"]["over_budget"] is True
    assert receipt["situation"]["delivery"] == "stopped"
    assert read_receipts(tmp_path / "run")[0]["task_bytes"] == 0


def test_seedless_delivers_only_peer_tips_even_with_a_generous_task_budget(tmp_path):
    brief = _brief(tmp_path)
    brief.seedless = True
    _layers(brief)
    task = render_contract(brief, max_bytes=60_000)
    assert "队友未核实线索" in task
    assert "保留 overlay" not in task
    assert "host ledger" not in task
    assert "待复核建议" not in task
    receipt = build_receipt(brief, task, max_bytes=60_000)
    layers = {row["name"]: row for row in receipt["situation"]["layers"]}
    assert layers["overlay"]["rendered_bytes"] == 0
    assert layers["host_ledger"]["rendered_bytes"] == 0
    assert layers["peer_unverified"]["status"] == "delivered"
    assert layers["peer_unverified"]["delivered_bytes"] > 0


def test_situation_renderer_source_is_part_of_the_prompt_fingerprint():
    parts = dict(prompt_fingerprint_parts())
    assert "taic/situation_block.py" in parts
    assert parts["taic/situation_block.py"] == Path("taic/situation_block.py").read_bytes()
