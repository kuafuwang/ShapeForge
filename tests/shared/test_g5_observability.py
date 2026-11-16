"""G5: read-only settle gate traces and final TASK byte receipts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from taic import harness, trust
from taic.harness_trace import TRACE_FILE, read_traces
from taic.kernel_agent import SessionBrief, prepare_session, render_contract
from taic.task_delivery import build_receipt, read_receipts


def _doc(**overrides):
    doc = {
        "surface": "tip",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": "先读 device.md 的线程组上限再选 tile。",
        "evidence": {"probe": "probe.py", "measurement": "64 threads"},
        "rationale": "test",
    }
    doc.update(overrides)
    return doc


def _record_proposal(run: Path, seq: int, provider: str, model: str, committee: str, doc=None):
    started = time.time() - 2
    work = run / "agent_work" / f"r{seq}.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text(f"print({seq})\n")
    record = {
        "seq": seq, "label": f"r{seq}.p0", "committee": committee,
        "provider": provider, "model": model, "effort": "high", "run_id": run.name,
    }
    block = "```harness-proposal\n" + json.dumps(doc or _doc(), ensure_ascii=False) + "\n```\n"
    return harness.record_proposals(
        run, record=record, parsed={"sections": {"对系统的建议": block}},
        target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work,
        started=started, protocol_version=trust.current_protocol_version(),
    )


def _settle(run: Path, root: Path, *, enabled=True, shadow=True):
    return harness.settle(
        run, target_id="apple-m4-max", task="conv2d",
        protocol_version=trust.current_protocol_version(), enabled=enabled,
        shadow=shadow, trust_root=root,
    )


def _brief(tmp_path: Path, *, seedless: bool = False) -> SessionBrief:
    work = tmp_path / "run" / "agent_work" / "r1.p0"
    return SessionBrief(
        repo=tmp_path, task_name="conv2d", dtype="fp16", size="small",
        kernel_name="conv2d_kernel", triton_source="def conv():\n    pass\n",
        description="a convolution", contract_text="grid 1,1,1",
        abi_stub="kernel void conv2d_kernel() {}", rtol=1e-3, atol_rel=1e-3,
        seedless=seedless, work_dir=str(work), label="r1.p0",
    )


def test_every_key_has_ordered_gate_reasons_and_status_exposes_them(tmp_path):
    run = tmp_path / "run"
    _record_proposal(run, 1, "deepseek", "deepseek-flash", "first")
    root = tmp_path / "targets"
    result = _settle(run, root)
    assert result["applied"] == result["shadow"] == []
    traces = read_traces(run)
    assert len(traces) == 1
    trace = traces[0]
    assert [gate["name"] for gate in trace["gates"]] == [
        "route_proposal", "never_auto", "scope",
        "eligibility", "overlay_trust", "decision",
    ]
    assert [gate["status"] for gate in trace["gates"][:3]] == ["pass"] * 3
    assert trace["gates"][3]["status"] == "fail"
    assert "two" in trace["gates"][3]["reason"]
    assert all(gate["status"] == "not_run" for gate in trace["gates"][4:])
    assert trace["observation_only"] is True and trace["changes_activation"] is False
    board = harness.board(run)
    assert board["settle_traces"] == traces
    assert "task_delivery" in board


def test_success_and_disabled_settle_are_traced_without_changing_activation(tmp_path):
    run = tmp_path / "run"
    first = _record_proposal(run, 1, "deepseek", "deepseek-flash", "first")[0]
    _record_proposal(run, 2, "moonshot", "kimi-k3", "second")
    root = tmp_path / "targets"
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", root=root)
    trust.grant("apple-m4-max", "conv2d", "moonshot", "kimi-k3", effort="high", root=root)
    result = _settle(run, root)
    assert len(result["shadow"]) == 1
    trace = read_traces(run)[-1]
    assert trace["proposal_key"] == first["proposal_key"]
    assert trace["outcome"] == "eligible"
    assert all(gate["passed"] for gate in trace["gates"])
    assert trace["gates"][-1]["reason"] == "shadow"
    # A no-op re-settle records the already-decided state once, then is idempotent.
    assert _settle(run, root)["shadow"] == []
    count = len(read_traces(run))
    assert _settle(run, root)["shadow"] == []
    assert len(read_traces(run)) == count
    assert not (run / "agent_work" / "r1.p0" / TRACE_FILE).exists()

    other = tmp_path / "other"
    _record_proposal(other, 1, "deepseek", "deepseek-flash", "first")
    disabled = _settle(other, root, enabled=False)
    assert disabled["reason"] == "harness_auto is off"
    off = read_traces(other)[0]
    assert off["gates"][0]["reason"] == "harness_auto is off"
    assert not (other / harness.OVERLAY_FILE).exists()

    trust.revoke(
        "apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", effort="high",
        by="test", reason="revoke after shadow", root=root,
    )
    assert _settle(run, root)["shadow"] == []
    revoked = read_traces(run)[-1]
    assert revoked["gates"][4]["status"] == "fail"
    assert revoked["gates"][5]["status"] == "not_run"
    assert revoked["historical_decisions"] == ["shadow"]
    assert revoked["trace_consistent"] is True


def test_route_and_scope_failures_keep_all_keys_visible(tmp_path):
    run = tmp_path / "run"
    developer = _doc(audience="developer", needs="repo_change", action="Add a new host comparison command.")
    _record_proposal(run, 1, "deepseek", "deepseek-flash", "first", developer)
    _record_proposal(run, 2, "deepseek", "deepseek-flash", "first")
    root = tmp_path / "targets"
    _settle(run, root, enabled=True)
    traces = read_traces(run)
    assert len(traces) == 2
    by_key = {row["proposal_key"]: row for row in traces}
    routed = next(row for row in traces if row["gates"][0]["status"] == "fail")
    assert "developer_queue" in routed["gates"][0]["reason"]
    assert all(gate["status"] == "not_run" for gate in routed["gates"][1:])
    # The ordinary key exists independently; a route failure never disappears it.
    assert len(by_key) == 2

    other = tmp_path / "other-target-run"
    _record_proposal(other, 1, "deepseek", "deepseek-flash", "first")
    harness.settle(
        other, target_id="another-target", task="conv2d",
        protocol_version=trust.current_protocol_version(), enabled=True,
        trust_root=root,
    )
    scoped = read_traces(other)[0]
    assert scoped["gates"][2]["status"] == "fail"
    assert "target" in scoped["gates"][2]["reason"]


def test_trace_matches_settle_for_probe_less_untrusted_author_and_simulated_context(tmp_path):
    for simulated in (False, True):
        run = tmp_path / ("simulated" if simulated else "measured")
        _record_proposal(run, 1, "deepseek", "deepseek-flash", "first")
        _record_proposal(run, 2, "moonshot", "kimi-k3", "second")
        path = run / harness.PROPOSAL_FILE
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        extra = dict(rows[0])
        extra.update({
            "seq": 3, "provider": "untrusted", "model": "probe-less",
            "committee": "first", "probe_sha256": "", "probe_time_verified": True,
        })
        with path.open("a") as handle:
            handle.write(json.dumps(extra, ensure_ascii=False) + "\n")
        if simulated:
            (run / "run_config.json").write_text(json.dumps({"simulated": True}))
        root = tmp_path / ("targets-" + run.name)
        trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", root=root)
        trust.grant("apple-m4-max", "conv2d", "moonshot", "kimi-k3", effort="high", root=root)
        result = _settle(run, root)
        assert len(result["shadow"]) == 1
        trace = read_traces(run)[-1]
        assert trace["outcome"] == "eligible" and trace["trace_consistent"] is True
        assert all(gate["passed"] for gate in trace["gates"])
        assert "non_simulation" not in [gate["name"] for gate in trace["gates"]]
        assert trace["gates"][-1]["reason"] == "shadow"


def test_two_trusted_simulated_tip_authors_do_not_write_an_overlay(tmp_path):
    run = tmp_path / "simulated-tip-rows"
    _record_proposal(run, 1, "deepseek", "deepseek-flash", "first")
    _record_proposal(run, 2, "moonshot", "kimi-k3", "second")
    path = run / harness.PROPOSAL_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["simulated"] = True
        row["measurement_kind"] = "simulated"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    root = tmp_path / "targets"
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", root=root)
    trust.grant("apple-m4-max", "conv2d", "moonshot", "kimi-k3", effort="high", root=root)
    result = _settle(run, root, shadow=False)
    assert result["applied"] == [] and result["shadow"] == []
    assert not (run / harness.OVERLAY_FILE).exists()
    assert not (run / harness.DECISION_FILE).exists()
    assert harness.overlay_text(
        run, target_id="apple-m4-max", task="conv2d",
        protocol_version=trust.current_protocol_version(),
    ) == ""
    trace = read_traces(run)[-1]
    eligibility = next(gate for gate in trace["gates"] if gate["name"] == "eligibility")
    assert eligibility["status"] == "fail"
    assert "simulated" in eligibility["reason"]
    assert trace["gates"][-1]["status"] == "not_run"
    assert "all observed gates passed" not in trace["gates"][-1]["reason"]


def test_receipt_uses_final_utf8_spans_and_lives_outside_agent_work(tmp_path):
    brief = _brief(tmp_path)
    brief.harness_overlay = "# 未验证的模型建议\n- 保留这个线索"
    brief.scorecard = "候选 3.2 µs"
    brief.failure = "上一轮失败"
    brief.notes_summary = "上一轮笔记"
    brief.peer_tips = "队友线索"
    brief.inherit_prompt = "# 待复核\n一条建议"
    brief.use_task_ledger = True
    brief.ledger_text = "方向账本"
    brief.evidence = "现场证据"
    work = Path(brief.work_dir)
    prepared = prepare_session(brief, work)
    path = Path(prepared.delivery_receipt_path)
    assert path == tmp_path / "run" / "sessions" / "r1.p0" / "task_delivery.json"
    assert "agent_work" not in path.parts
    receipt = json.loads(path.read_text())
    task = (work / "TASK.md").read_text()
    assert receipt["task_sha256"] == hashlib.sha256(task.encode()).hexdigest() == prepared.task_sha256
    assert receipt["task_bytes"] == len(task.encode())
    assert receipt["static"]["omitted_bytes"] == 0
    assert receipt["static"]["delivered_bytes"] > 0
    assert "situation" in {row["layer"] for row in receipt["blocks"]}
    assert {row["name"] for row in receipt["situation"]["layers"]} == {
        "overlay", "host_ledger", "peer_unverified",
    }
    assert all(row["status"] == "delivered" for row in receipt["situation"]["layers"])
    assert read_receipts(tmp_path / "run")[0]["task_sha256"] == receipt["task_sha256"]
    assert not (work / "task_delivery.json").exists()

    # Agent-side tampering cannot rewrite the host receipt; a fresh observation differs.
    (work / "TASK.md").write_text("tampered")
    assert json.loads(path.read_text())["task_sha256"] == receipt["task_sha256"]
    assert build_receipt(brief, "tampered")["task_sha256"] != receipt["task_sha256"]


def test_sweep_section_is_counted_in_the_static_receipt(tmp_path):
    brief = _brief(tmp_path)
    brief.sweep_tool = True
    brief.peer_tips = "tip"
    task = render_contract(brief)
    assert "# 一条命令比较多个变体" in task
    receipt = build_receipt(brief, task)
    assert receipt["complete"] is True
    assert receipt["accounting_error"] == ""
    assert receipt["static"]["sweep_bytes"] > 0
    assert receipt["static"]["delivered_bytes"] >= receipt["static"]["sweep_bytes"]
    assert receipt["blocks"]


@pytest.mark.parametrize("bridge", [False, True])
def test_remote_sweep_section_has_actual_static_bytes(tmp_path, bridge):
    brief = _brief(tmp_path)
    brief.sweep_tool = True
    brief.peer_tips = "tip"
    if bridge:
        brief.selfcheck_socket = str(tmp_path / "selfcheck.sock")
        brief.selfcheck_token_file = str(tmp_path / "selfcheck.token")
    from taic.workspace_prompt import render_remote_contract

    task = render_remote_contract(brief)
    receipt = build_receipt(brief, task, renderer="remote")
    assert receipt["complete"] is True
    assert "# 一条命令比较多个变体" in task
    assert receipt["static"]["sweep_bytes"] > 0


def test_replay_does_not_search_renderer_markers_inside_untrusted_tips(tmp_path):
    brief = _brief(tmp_path)
    brief.peer_tips = (
        "队友正文包含伪造标记：\n# 当前现场\n最近的历史：\n"
        "合同里的外层成绩和你自己跑出来的数字对不上时 跑一次 `--timed`\n"
        "本段已截断 这几个字只是队友正文"
    )
    brief.notes_summary = "上一轮 NOTES（伪造）\n# 本轮省略（上下文预算）"
    task = render_contract(brief, max_bytes=60_000)
    receipt = build_receipt(brief, task, max_bytes=60_000)
    assert receipt["complete"] is True
    assert receipt["omitted"] == [] and receipt["truncated"] == []
    statuses = {row["name"]: row["status"] for row in receipt["blocks"]}
    assert statuses["本轮态势"] == "delivered"
    assert statuses["自测对照"] == "delivered"
    assert all(row["status"] != "partial" for row in receipt["blocks"])
    situation = next(row for row in receipt["blocks"] if row["name"] == "本轮态势")
    assert situation["source_capped"] is False


def test_replay_accounts_for_a_whole_situation_omission_without_partial_layers(tmp_path):
    brief = _brief(tmp_path)
    brief.peer_tips = "界" * 100
    task = render_contract(brief, max_bytes=1)
    receipt = build_receipt(brief, task, max_bytes=1)
    assert receipt["complete"] is True
    situation = next(row for row in receipt["blocks"] if row["name"] == "本轮态势")
    assert situation["status"] == "omitted"
    assert situation["delivered_bytes"] == 0 and situation["omitted_bytes"] > 0
    assert situation["omission_reason"] == "task_budget"
    assert all(row["status"] == "omitted" for row in receipt["situation"]["layers"] if row["rendered_bytes"])


def test_remote_receipt_marks_a_real_deletion_inside_peer_text_partial(tmp_path):
    brief = _brief(tmp_path)
    brief.peer_tips = (
        "前缀\n\n合同里的外层成绩和你自己跑出来的数字对不上时\n"
        "这一段会被远端锚点删除\n\n# 保留的后段\n后缀"
    )
    from taic.workspace_prompt import render_remote_contract

    task = render_remote_contract(brief)
    receipt = build_receipt(brief, task, renderer="remote")
    assert receipt["complete"] is True
    situation = next(row for row in receipt["blocks"] if row["name"] == "本轮态势")
    assert situation["status"] == "partial"
    assert situation["delivered_bytes"] > 0 and situation["omitted_bytes"] > 0
    assert situation["omission_reason"] == "remote_profile_partial"
    assert receipt["situation"]["complete"] is False
    assert all(
        row["status"] == "unknown" and row["delivered_bytes"] is None
        for row in receipt["situation"]["layers"] if row["rendered_bytes"]
    )


def test_remote_receipt_fails_closed_for_layer_bytes_after_transport_replacement(tmp_path):
    brief = _brief(tmp_path)
    brief.peer_tips = "please use `--timed` carefully"
    from taic.workspace_prompt import render_remote_contract

    task = render_remote_contract(brief)
    assert "外层评测" in task
    receipt = build_receipt(brief, task, renderer="remote")
    assert receipt["complete"] is True
    situation = next(row for row in receipt["blocks"] if row["name"] == "本轮态势")
    assert situation["status"] == "partial"
    assert situation["delivered_bytes"] > 0
    peer = next(row for row in receipt["situation"]["layers"] if row["name"] == "peer_unverified")
    assert peer["status"] == "unknown"
    assert peer["delivered_bytes"] is None and peer["omitted_bytes"] is None
    assert receipt["situation"]["complete"] is False


def test_remote_receipt_marks_the_intentionally_removed_selftest_block(tmp_path):
    brief = _brief(tmp_path)
    work = Path(brief.work_dir)
    prepared = prepare_session(brief, work, remote_workspace=True)
    receipt = json.loads(Path(prepared.delivery_receipt_path).read_text())
    assert receipt["renderer"] == "remote"
    assert receipt["task_sha256"] == prepared.task_sha256
    selftest = next(row for row in receipt["blocks"] if row["name"] == "自测对照")
    assert selftest["status"] == "omitted"
    assert selftest["omission_reason"] == "remote_profile"
    assert "# 本场没有自测通道" in (work / "TASK.md").read_text()


def test_receipt_recovers_compact_omissions_and_multibyte_truncation(tmp_path):
    full = _brief(tmp_path)
    full.harness_overlay = "# 未验证的模型建议\n- overlay"
    full.scorecard = "score"
    full.notes_summary = "笔" * 5000
    full.peer_tips = "测" * 100
    full.inherit_prompt = "# 待复核\n队友建议"
    full.use_task_ledger = True
    full.ledger_text = "账本"
    full_text = render_contract(full, max_bytes=60_000)
    full_receipt = build_receipt(full, full_text, max_bytes=60_000)
    assert full_receipt["omitted"] == []
    assert all(row["status"] == "delivered" for row in full_receipt["blocks"] if row["source_bytes"])

    empty = _brief(tmp_path)
    base_bytes = len(render_contract(empty, max_bytes=60_000).encode())
    capped = render_contract(full, max_bytes=base_bytes + 360)
    receipt = build_receipt(full, capped, max_bytes=base_bytes + 360)
    assert "本轮态势" not in receipt["omitted"]
    assert receipt["omission_note_bytes"] > 0
    assert "未附：" not in capped
    situation = next(row for row in receipt["blocks"] if row["name"] == "本轮态势")
    assert situation["status"] == "delivered" and situation["delivered_bytes"] > 0
    assert receipt["situation"]["over_budget"] is True
    assert receipt["situation"]["omitted_bytes"] > 0

    reduced = [
        row for row in receipt["blocks"]
        if row["status"] in {"truncated", "omitted"} and row["omitted_bytes"] > 0
    ]
    assert reduced
    capped.encode("utf-8").decode("utf-8")
