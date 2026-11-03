"""Offline routing/injection fixtures: useful developer requests never become overlays.

Each case fixes an intent, the host-derived recipient and the expected disposition.
No model call, no GPU, no repository writes outside tmp_path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from taic import harness, trust
from taic.proposal_routing import route_proposal

CASES = Path(__file__).resolve().parents[1] / "support" / "injection_cases.json"
B413 = "Add a tcenv.cli ab subcommand that compiles several kernel files, times them interleaved in one process over rotating buffer sets, and prints per-file median wall time and cv."
PEER = "先读 device.md 的线程组上限再选 tile。"


def _case_ids():
    return [case["id"] for case in json.loads(CASES.read_text(encoding="utf-8"))["cases"]]


@pytest.mark.parametrize("case_id", _case_ids())
def test_routing_fixture(case_id):
    """@brief 每个语料都固定接收者、仓库改动需求、当前可执行性与最终 disposition。"""
    case = next(c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"] if c["id"] == case_id)
    got = route_proposal(case)
    for key, expected in case["expect"].items():
        assert got[key] == expected, (case_id, key, got)


def test_validate_proposal_keeps_declaration_but_records_host_route(tmp_path):
    """@brief 模型自报 peer 会留痕，宿主仍判 developer 并标记 intent_conflict。"""
    doc = {
        "surface": "tip",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": B413,
        "audience": "peer",
        "needs": "none",
        "peer_action": "use the new subcommand",
        "evidence": {"reproduction": "run the proposed command", "expected": "per-file median and cv"},
        "rationale": "useful tool",
    }
    proposal = harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=None)
    assert proposal["audience"] == "peer"  # 自报留痕，方便人核对意图
    assert proposal["routing"]["recipient"] == "developer"
    assert proposal["routing"]["requires_repo_change"] is True
    assert proposal["routing"]["actionable_within_session"] is False
    assert proposal["routing"]["disposition"] == "developer_queue"
    assert proposal["routing"]["intent_conflict"] is True


def _row(seq, provider, model, committee, action, probe_sha, *, key="k", task="conv2d", peer_action=""):
    return {
        "schema_version": 1, "event": "proposal", "seq": seq, "status": "valid",
        "proposal_key": key, "run_id": "r", "target_id": "apple-m4-max", "task": task,
        "provider": provider, "model": model, "committee": committee,
        "probe_time_verified": True, "probe_sha256": probe_sha,
        "proposal": {"surface": "tip", "action": action, "peer_action": peer_action,
                     "scope": {"target_id": "apple-m4-max", "task": task, "dtype": "fp16"}, "evidence": {}},
    }


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _settle(run: Path):
    return harness.settle(
        run, target_id="apple-m4-max", task="conv2d",
        protocol_version=trust.current_protocol_version(), enabled=True, shadow=False,
        trust_root=run, prompt_sha1="test",
    )


def test_developer_request_cannot_reach_quorum_overlay(tmp_path, monkeypatch):
    """@brief b413 即使有两个委员会、两个模型、两份探针与信任，也零 overlay。"""
    monkeypatch.setattr(harness, "_trusted", lambda *args, **kwargs: (True, ""))
    _write_jsonl(tmp_path / "harness_proposals.jsonl", [
        _row(1, "deepseek", "deepseek-flash", "first", B413, "a"),
        _row(2, "moonshot", "kimi-k3", "second", B413, "b"),
    ])
    result = _settle(tmp_path)
    assert result == {"applied": [], "shadow": []}
    assert not (tmp_path / "harness_overlay.jsonl").exists()


def test_a_real_peer_tip_still_reaches_quorum_overlay(tmp_path, monkeypatch):
    """@brief 合法对照：同样的信任与证据下，当前可执行的写核提示仍可生效。"""
    monkeypatch.setattr(harness, "_trusted", lambda *args, **kwargs: (True, ""))
    _write_jsonl(tmp_path / "harness_proposals.jsonl", [
        _row(1, "deepseek", "deepseek-flash", "first", PEER, "a"),
        _row(2, "moonshot", "kimi-k3", "second", PEER, "b"),
    ])
    result = _settle(tmp_path)
    assert len(result["applied"]) == 1
    assert result["applied"][0]["text"] == PEER


def test_useful_review_cannot_activate_developer_request(tmp_path, monkeypatch):
    """@brief 单来源 + useful review 也不能把 b413 激活为 reviewed_active。"""
    monkeypatch.setattr(harness, "_trusted", lambda *args, **kwargs: (True, ""))
    _write_jsonl(tmp_path / "harness_proposals.jsonl", [_row(1, "deepseek", "deepseek-flash", "first", B413, "a")])
    _write_jsonl(tmp_path / "harness_reviews.jsonl", [{
        "event": "review", "proposal_key": "k", "useful": True,
        "protocol_version": trust.current_protocol_version(), "source_seq": 1,
        "reviewer": {"provider": "moonshot", "model": "kimi-k3", "effort": "high"},
    }])
    result = _settle(tmp_path)
    assert result == {"applied": [], "shadow": []}
    assert not (tmp_path / "harness_overlay.jsonl").exists()


def test_replay_real_b413_and_28c_records():
    """@brief 回放真实提案：b413 与全部 reporting/文档/self-check 请求进开发者队列，28c 仍是写核候选。"""
    expected = {
        "b4131be13c4750c8": "developer_queue",
        "43c462b0b4f70803": "developer_queue",
        "992da5f11b71aa14": "developer_queue",
        "f51b80b21f40375e": "developer_queue",
        "d62f8472ec2a324a": "developer_queue",
        "20ac07894968f2fe": "developer_queue",
        "28c247a285f8a1e3": "overlay_candidate",
    }
    rows = []
    for path in Path("runs").glob("*/*/harness_proposals.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(key in line for key in expected):
                rows.append(json.loads(line))
    if not rows:
        pytest.skip("historical proposal records are not present")
    by_key = {row["proposal_key"]: row for row in rows}
    assert set(expected) <= set(by_key)
    for key, disposition in expected.items():
        assert route_proposal(by_key[key]["proposal"])["disposition"] == disposition, key


def test_developer_request_is_not_sent_to_automatic_review(tmp_path):
    """@brief developer 提案留在队列里，不占自动 overlay 评审流量与费用。"""
    rows = [_row(1, "deepseek", "deepseek-flash", "first", B413, "a"),
            _row(2, "moonshot", "kimi-k3", "second", PEER, "b")]
    pending = harness.pending_tip_reviews(tmp_path, rows)
    assert [row["proposal"]["action"] for row in pending] == [PEER]


def test_developer_request_cannot_be_promoted_into_experience(tmp_path):
    """@brief 人工 promotion 也不能把工具请求写进写核经验库；它应走 P2 代码路径。"""
    proposal = {"surface": "tip", "action": B413, "rationale": "developer tool"}
    with pytest.raises(harness.ProposalError, match="developer or unknown"):
        harness.candidate_patch(proposal, target_id="apple-m4-max", proposal_key="b4131be13c4750c8", root=tmp_path)
    patch, rel = harness.candidate_patch({"surface": "tip", "action": PEER}, target_id="apple-m4-max", proposal_key="k", root=tmp_path)
    assert rel == "targets/apple-m4-max/experience.md" and PEER in patch


def test_inherit_prompt_offers_peer_tip_without_activating_it(tmp_path):
    """@brief 有用 review 可作为 advisory 摘要进入下一次复核，但不生成 overlay。"""
    key = "0123456789abcdef"
    _write_jsonl(tmp_path / "harness_proposals.jsonl", [
        _row(1, "deepseek", "deepseek-flash", "first", PEER, "a", key=key, peer_action="先读 device.md"),
        _row(2, "moonshot", "kimi-k3", "second", B413, "b", key="fedcba0987654321"),
    ])
    _write_jsonl(tmp_path / "harness_reviews.jsonl", [{
        "event": "review", "proposal_key": key, "useful": True, "reason": "可当场核对",
        "reviewer": {"provider": "moonshot", "model": "kimi-k3"},
    }])
    text = harness.inherit_prompt_text(tmp_path, target_id="apple-m4-max", task="conv2d")
    assert "待你复核" in text and PEER in text and "先读 device.md" in text
    assert "review=useful" in text and "评审摘要(仅供参考): 可当场核对" in text
    assert "tip-review <key> adopt|reject|unknown" in text
    assert B413 not in text  # developer 请求不占复核上下文
    assert not (tmp_path / "harness_overlay.jsonl").exists()


def test_inherit_prompt_skips_overlay_and_already_reviewed_tips(tmp_path):
    """@brief 已有 overlay 或 NOTES 已记录 adopt/reject/unknown 的提案不重复送审。"""
    key = "0123456789abcdef"
    _write_jsonl(tmp_path / "harness_proposals.jsonl", [_row(1, "deepseek", "deepseek-flash", "first", PEER, "a", key=key)])
    _write_jsonl(tmp_path / "harness_overlay.jsonl", [{"event": "overlay", "proposal_key": key}])
    assert harness.inherit_prompt_text(tmp_path, target_id="apple-m4-max", task="conv2d") == ""
    (tmp_path / "harness_overlay.jsonl").unlink()
    notes = tmp_path / "agent_work" / "r1"
    notes.mkdir(parents=True)
    (notes / "NOTES.md").write_text(f"tip-review {key} reject: 不适用当前 shape\n", encoding="utf-8")
    assert harness.inherit_prompt_text(tmp_path, target_id="apple-m4-max", task="conv2d") == ""


def test_inherit_prompt_is_bounded_and_prefers_actionable_peer_fields(tmp_path):
    """@brief 最多两条；带 peer_action 的细分建议优先，避免把长 rationale 全塞进 TASK。"""
    rows = [
        _row(1, "a", "a", "first", PEER, "a", key="0000000000000001"),
        _row(2, "b", "b", "second", PEER + "", "b", key="0000000000000002", peer_action="跑现有探针"),
        _row(3, "c", "c", "first", PEER, "c", key="0000000000000003"),
    ]
    _write_jsonl(tmp_path / "harness_proposals.jsonl", rows)
    text = harness.inherit_prompt_text(tmp_path, target_id="apple-m4-max", task="conv2d", limit=1200, max_items=2)
    assert text.count("- [tip ") == 2
    assert "0000000000000002" in text and "跑现有探针" in text
    assert len(text.encode("utf-8")) <= 1400  # 包含两行静态说明，动态部分受 limit 约束


def test_inherit_prompt_reaches_next_task_without_extra_reviewer(tmp_path):
    """@brief harness_auto 关闭时也能作为待复核提示进入 TASK；seedless 首场不注入。"""
    from ..support.orchestrator import code, llm_script, make_orch
    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [])
    run = orch.run_dir
    run.mkdir(parents=True, exist_ok=True)
    orch.search.inherit_prompt = True
    _write_jsonl(run / "harness_proposals.jsonl", [
        _row(1, "deepseek", "deepseek-flash", "first", PEER, "a", key="0123456789abcdef", task=orch.task_name),
    ])
    brief = orch._session_brief(seedless=False)
    assert PEER in brief.inherit_prompt
    from taic.kernel_agent import render_contract
    task_text = render_contract(brief)
    assert "队友建议（待你复核" in task_text and PEER in task_text
    seedless = render_contract(orch._session_brief(seedless=True))
    assert "队友建议（待你复核" not in seedless


def test_template_teaches_the_two_intent_formats():
    """@brief 模板不只有字段名：还教 peer/developer 的选择规则、对照示例与禁止项。"""
    text = harness.TEMPLATE
    assert "Choose ``audience`` before writing ``action``" in text
    assert '"audience": "peer"' in text and '"needs": "none"' in text
    assert '"audience": "developer"' in text and '"needs": "repo_change"' in text
    assert "Never name a command, file" in text
    assert "Add a tcenv.cli ab subcommand" in text  # developer 反例不是抽象口号


def test_unknown_tip_can_be_reviewed_but_never_activated(tmp_path, monkeypatch):
    """@brief 不确定但不越权的建议可给后续 LLM 复核，仍不能自动 overlay。"""
    action = "Consider the overall approach carefully."
    key = "0123456789abcdef"
    assert route_proposal({"action": action})["disposition"] == "review_candidate"
    _write_jsonl(tmp_path / "harness_proposals.jsonl", [
        _row(1, "deepseek", "deepseek-flash", "first", action, "a", key=key),
        _row(2, "moonshot", "kimi-k3", "second", action, "b", key=key),
    ])
    monkeypatch.setattr(harness, "_trusted", lambda *args, **kwargs: (True, ""))
    assert _settle(tmp_path) == {"applied": [], "shadow": []}
    text = harness.inherit_prompt_text(tmp_path, target_id="apple-m4-max", task="conv2d")
    assert "intent=unknown" in text and action in text
    assert "不是否决" in text and "改写成可执行探针" in text


def test_record_proposals_preserves_declared_intent_and_host_route(tmp_path):
    """@brief 真实落盘不能丢 audience/needs/peer_action，否则重算路由会反转。"""
    action = "Measure the candidate with the existing probe and compare cv."
    doc = {
        "surface": "tip",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": action,
        "audience": "developer",
        "needs": "new_tool",
        "peer_action": "先读 device.md",
        "evidence": {"probe": "probe.py", "measurement": "10 us"},
    }
    work = tmp_path / "agent_work" / "r1"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n", encoding="utf-8")
    parsed = {"sections": {"对系统的建议": "```harness-proposal\n" + json.dumps(doc) + "\n```"}}
    rows = harness.record_proposals(
        tmp_path,
        record={"seq": 1, "run_id": "r", "label": "r1", "provider": "deepseek", "model": "deepseek-flash", "effort": "high", "committee": "first"},
        parsed=parsed, target_id="apple-m4-max", task="conv2d", dtype="fp16",
        work=work, started=time.time() - 1, protocol_version=trust.current_protocol_version(),
    )
    assert len(rows) == 1
    proposal = rows[0]["proposal"]
    assert proposal["audience"] == "developer" and proposal["needs"] == "new_tool"
    assert proposal["peer_action"] == "先读 device.md"
    assert proposal["routing"]["disposition"] == "developer_queue"
    assert route_proposal(proposal)["disposition"] == "developer_queue"
    assert action not in harness.inherit_prompt_text(tmp_path, target_id="apple-m4-max", task="conv2d")


def test_peer_action_cannot_upgrade_or_inject_a_proposal(tmp_path):
    """@brief 自报 peer_action 不能把 unknown 升级为 overlay，也不能携带 shell/仓库请求。"""
    vague = {"action": "Consider the overall approach carefully.", "audience": "peer", "peer_action": "do it"}
    assert route_proposal(vague)["disposition"] == "review_candidate"
    assert route_proposal({**vague, "peer_action": "Add a tcenv.cli ab subcommand"})["disposition"] == "developer_queue"
    doc = {
        "surface": "tip", "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": PEER, "audience": "peer", "needs": "none",
        "peer_action": "Add a tcenv.cli ab subcommand && rm -rf /",
        "evidence": {"reproduction": "r", "expected": "e"},
    }
    with pytest.raises(harness.ProposalError, match="peer_action"):
        harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=None)


def test_reporting_documentation_and_selfcheck_requests_are_developer_tasks():
    """@brief 报告、文档、self-check 与同义工具请求不属于写核 workdir 动作。"""
    for action in (
        "在评测报告里打印每个 stage 的编译器寄存器用量。",
        "Print the fixed per-iteration pipeline overhead beside the candidate time.",
        "Fix the threadelements() example in mslreference.md.",
        "Document the working IR dump command in the taic-tools skill.",
        "In the timed self-check, also report a paired candidate-vs-seed ratio.",
        "给 tcenv.cli 加一个 ab 子命令，交替计时多个 kernel 文件并输出 cv。",
        "Provide a tcenv.cli ab subcommand to compare two kernel files.",
    ):
        assert route_proposal({"action": action})["disposition"] == "developer_queue", action


def test_inherit_prompt_deduplicates_keys_but_allows_shadow_candidates(tmp_path):
    """@brief 同 key 只占一个名额；shadow 未激活的候选仍可给后续 LLM 复核。"""
    key = "0123456789abcdef"
    _write_jsonl(tmp_path / "harness_proposals.jsonl", [
        _row(1, "deepseek", "deepseek-flash", "first", PEER, "a", key=key),
        _row(2, "moonshot", "kimi-k3", "second", PEER, "b", key=key),
    ])
    _write_jsonl(tmp_path / "harness_overlay.jsonl", [{"event": "overlay", "proposal_key": key, "shadow": True}])
    text = harness.inherit_prompt_text(tmp_path, target_id="apple-m4-max", task="conv2d")
    assert text.count("- [tip ") == 1 and PEER in text


def test_historical_developer_overlay_is_evidence_not_an_injection(tmp_path):
    """@brief 历史 b413 行仍可留在盘上审计，但 overlay_text 不得再把它送进 TASK。"""
    _write_jsonl(tmp_path / "harness_overlay.jsonl", [{
        "schema_version": 1, "event": "overlay", "target_id": "apple-m4-max", "task": "conv2d",
        "protocol_version": trust.current_protocol_version(), "source_run": "r", "source_seq": 2,
        "proposal_key": "b4131be13c4750c8", "surface": "tip", "text": B413,
        "shadow": False, "reviewed": True,
    }])
    assert harness.overlay_rows(tmp_path, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), shadow=False)
    assert harness.overlay_text(tmp_path, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version()) == ""


def test_repository_requests_are_blocked_but_session_notes_are_allowed():
    """@brief 通用仓库请求必须拦截；写 NOTES/FEEDBACK 则是会话内合法工作。"""
    assert route_proposal({
        "action": "Use the existing probe to compare two tile sizes.",
        "audience": "peer", "needs": "none", "peer_action": "Modify the repository.",
    })["disposition"] == "developer_queue"
    for action in (
        "Measure the tile timing and document it in NOTES.md.",
        "Measure the tile timing and report it in FEEDBACK.md.",
    ):
        assert route_proposal({"action": action, "audience": "peer"})["disposition"] == "overlay_candidate", action


def test_overlay_text_does_not_claim_two_independent_authors(tmp_path):
    """@brief 渲染措辞承认单来源+评审的可能，不再固定写成两个模型各自提出。"""
    _write_jsonl(tmp_path / "harness_overlay.jsonl", [{
        "schema_version": 1, "event": "overlay", "target_id": "apple-m4-max", "task": "conv2d",
        "protocol_version": trust.current_protocol_version(), "source_run": "r", "source_seq": 1,
        "proposal_key": "k", "surface": "tip", "text": PEER, "shadow": False, "reviewed": True,
    }])
    text = harness.overlay_text(tmp_path, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version())
    assert "两个模型在本硬件上各自提出" not in text
    assert "单一来源" in text and PEER in text
