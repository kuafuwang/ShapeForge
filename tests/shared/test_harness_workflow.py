"""A synthetic feedback -> proposal -> trust -> active overlay -> TASK workflow. No model or GPU."""

from __future__ import annotations

import json
import time
from pathlib import Path

from taic import harness, trust
from taic.kernel_agent import SessionBrief, render_contract
from tasks import get_task


def _doc(action: str):
    return {
        "surface": "tip",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": action,
        "evidence": {"probe": "probe.py", "measurement": "1.0 us"},
        "rationale": "workflow test",
    }


def _session(run: Path, seq: int, provider: str, model: str, committee: str, action: str, probe: str):
    work = run / "agent_work" / f"r{seq}.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text(probe)
    parsed = {"sections": {"对系统的建议": "```harness-proposal\n" + json.dumps(_doc(action), ensure_ascii=False) + "\n```\n"}}
    harness.record_proposals(
        run, record={"seq": seq, "label": f"r{seq}.p0", "committee": committee, "provider": provider, "model": model, "effort": "high", "run_id": "workflow"},
        parsed=parsed, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work, started=time.time() - 1,
    )


def test_an_active_tip_reaches_the_next_task_and_disable_removes_it(tmp_path):
    run = tmp_path / "run"
    _session(run, 1, "deepseek", "deepseek-flash", "first", "先读 device.md 的线程组上限。", "print(1)\n")
    _session(run, 2, "moonshot", "kimi-k3", "second", "先读 device.md 的线程组上限。", "print(2)\n")
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", root=tmp_path / "targets")
    trust.grant("apple-m4-max", "conv2d", "moonshot", "kimi-k3", effort="high", root=tmp_path / "targets")
    result = harness.settle(
        run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(),
        enabled=True, shadow=False, trust_root=tmp_path / "targets",
    )
    assert result["applied"]
    assert "先读 device.md" in harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version())

    task = get_task("conv2d", "fp16", "small")
    brief = SessionBrief(
        repo=tmp_path, task_name="conv2d", dtype="fp16", size="small",
        kernel_name=task.contract.kernel_name, triton_source=task.triton_source, description=task.description,
        contract_text=task.contract.describe(), abi_stub=task.contract.abi_stub(), rtol=task.rtol, atol_rel=task.atol_rel,
        work_dir=str(tmp_path / "work"), seedless=False, seed_source="kernel void conv2d_kernel() {}",
        harness_overlay=harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version()),
    )
    text = render_contract(brief, max_bytes=20_000)
    assert "未验证的模型建议" in text and "先读 device.md" in text

    harness.set_overlay_control(run, disabled=True, by="alice", reason="workflow stop")
    assert harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version()) == ""
    assert harness.overlay_text(run, target_id="other-target", task="conv2d", protocol_version=trust.current_protocol_version()) == ""


def test_a_premise_is_shadow_only_until_the_trusted_record_gate_exists(tmp_path):
    run = tmp_path / "run"
    work = run / "agent_work" / "r1.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n")
    parsed = {"sections": {"对系统的建议": "```harness-proposal\n" + json.dumps({**_doc("先测再决定 tile。"), "surface": "premise"}, ensure_ascii=False) + "\n```\n"}}
    harness.record_proposals(
        run, record={"seq": 1, "label": "r1.p0", "committee": "first", "provider": "deepseek", "model": "deepseek-flash", "effort": "high", "run_id": "workflow"},
        parsed=parsed, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work, started=time.time() - 1,
    )
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", scopes=["overlay:premise"], root=tmp_path / "targets")
    result = harness.settle(
        run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(),
        enabled=True, shadow=False, trust_root=tmp_path / "targets",
    )
    assert result["applied"] == []
    assert harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version()) == ""


def test_a_reviewed_active_overlay_is_in_the_next_session_brief_even_when_shadow_is_on(tmp_path):
    from ..support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [])
    run = orch.run_dir
    run.mkdir(parents=True, exist_ok=True)
    import json as _json

    (run / "harness_overlay.jsonl").write_text(_json.dumps({
        "schema_version": 1, "event": "overlay", "target_id": "apple-m4-max", "task": "relu",
        "protocol_version": trust.current_protocol_version(), "source_run": "r", "source_seq": 1,
        "proposal_key": "k", "surface": "tip", "text": "先读 device.md 再选 tile。",
        "evidence_sha256": "", "shadow": False, "reviewed": True,
        "reviewer": {"provider": "moonshot", "model": "kimi-k3"}, "active_from": 1, "active_until": None,
    }) + "\n")
    orch.search.harness_auto = True
    orch.search.harness_shadow = True
    brief = orch._session_brief(seedless=False)
    assert "先读 device.md" in brief.harness_overlay
