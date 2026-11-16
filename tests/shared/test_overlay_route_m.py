"""Route M: host paired comparisons can shadow a P1 line. Nominations cannot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from taic import harness, trust
from taic.overlay_route_m import GATE_ORDER, TRACE_FILE, settle_route_m

CHIP = "apple-m4-max"
TASK = "conv2d"
PROVIDER = "deepseek"
MODEL = "deepseek-flash"
KERNEL_A = (
    "kernel void a(device float *y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] = 1.0f; }\n"
)
KERNEL_B = (
    "kernel void b(device float *y [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
    "#pragma unroll\n"
    " for (int k = 0; k < 4; ++k) { y[i] = y[i] + 1.0f; }\n"
    "}\n"
)
RATIO = "9.876"
INTERVAL = "1.234"


def _digest(text: str) -> str:
    """@brief 源码字节的 SHA-256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _settle(run: Path, root: Path, *, task: str = TASK, chip: str = CHIP, shadow: bool = True, enabled: bool = True) -> dict:
    """@brief 用临时信任表跑一次 settle。"""
    return harness.settle(
        run, target_id=chip, task=task, protocol_version=trust.current_protocol_version(),
        enabled=enabled, shadow=shadow, trust_root=root,
    )


def _grant(root: Path, *, task: str = TASK, chip: str = CHIP, provider: str = PROVIDER, model: str = MODEL) -> None:
    """@brief 给提名作者一条 active overlay:tip。"""
    trust.grant(chip, task, provider, model, effort="high", scopes=["overlay:tip"], root=root)


def _plant(
    tmp: Path,
    *,
    conclusion: str = "no_diff",
    direction: str | None = None,
    applied: bool = True,
    gpu: bool = True,
    task: str = TASK,
    chip: str = CHIP,
    backend: str = "metal",
    simulated: bool = False,
    provider: str = PROVIDER,
    model: str = MODEL,
    sources: bool = False,
    nomination_claim: str = "",
    nomination_conclusion: str = "",
    premise: str = "",
    fence_action: str = "",
    extra: dict | None = None,
    run_backend: str | None = "metal",
    label: str = "r2.p0",
) -> tuple[Path, Path, str, str]:
    """@brief 造一场带宿主比较文件的 run。返回 run、信任根和两颗 SHA。"""
    run = tmp / "run"
    root = tmp / "targets"
    if sources:
        sha_a, sha_b = _digest(KERNEL_A), _digest(KERNEL_B)
    else:
        sha_a, sha_b = _digest("kernel-a"), _digest("kernel-b")
    folder = run / "candidates" / f"002_{label}"
    folder.mkdir(parents=True)
    document = {
        "schema": "taic.paired_comparison.1",
        "applied": applied,
        "gpu": gpu,
        "backend": backend,
        "sha_a": sha_a,
        "sha_b": sha_b,
        "conclusion": conclusion,
        "direction": direction,
        "changes_seed": False,
        "ratios": [float(RATIO)],
        "interval": [float(INTERVAL), 5.678],
        "median_a_s": 0.111,
        "median_b_s": 0.222,
    }
    if not applied:
        document["reason"] = "no current seed to pair with"
        document["gpu"] = False
    if extra:
        document.update(extra)
    comparison = folder / "paired_comparison.json"
    comparison.write_text(json.dumps(document), encoding="utf-8")
    (folder / "report.json").write_text(json.dumps({"task": task, "dtype": "fp16"}), encoding="utf-8")
    if sources:
        (folder / "kernel.metal").write_text(KERNEL_A, encoding="utf-8")
        paired = run / "paired-sources"
        paired.mkdir()
        (paired / "b.metal").write_text(KERNEL_B, encoding="utf-8")
    feedback = {
        "label": label, "seq": 3, "provider": provider, "model": model, "effort": "high",
        "task": task, "target_id": chip, "backend": backend, "simulated": simulated,
        "measurement_kind": "simulated" if simulated else "measured",
    }
    (run / "feedback.jsonl").write_text(json.dumps(feedback) + "\n", encoding="utf-8")
    if run_backend is not None:
        (run / "run_config.json").write_text(
            json.dumps({"target": {"id": chip, "backend": run_backend}, "task": task}),
            encoding="utf-8",
        )
    if nomination_claim or nomination_conclusion:
        work = run / "agent_work" / label / ".selfcheck"
        work.mkdir(parents=True)
        (work / "paired_comparisons.jsonl").write_text(json.dumps({
            "schema": "taic.paired_nomination.1",
            "sha_a": sha_a, "sha_b": sha_b,
            "conclusion": nomination_conclusion,
            "claim": nomination_claim,
            "source": "model",
            "host_evidence": True,
        }) + "\n", encoding="utf-8")
    if premise:
        (run / "agent_work" / label).mkdir(parents=True, exist_ok=True)
        (run / "agent_work" / label / "FEEDBACK.md").write_text(
            f"# 被否决的前提\n- {premise}\n", encoding="utf-8",
        )
    if fence_action:
        (run / harness.PROPOSAL_FILE).write_text(json.dumps({
            "status": "valid",
            "label": label,
            "proposal": {"surface": "tip", "action": fence_action},
        }) + "\n", encoding="utf-8")
    return run, root, sha_a, sha_b


def _traces(run: Path) -> list[dict]:
    """@brief 读路线 M 留痕。"""
    return [json.loads(line) for line in (run / TRACE_FILE).read_text(encoding="utf-8").splitlines()]


def _overlay(run: Path) -> list[dict]:
    """@brief 读 overlay 行。"""
    path = run / harness.OVERLAY_FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_no_diff_writes_a_shadow_line_and_keeps_the_model_ratio_out(tmp_path: Path):
    """@brief 没差可以送达。实测句和模型说法分开，比值留在提名记录里。"""
    run, root, sha_a, sha_b = _plant(tmp_path, conclusion="no_diff", nomination_conclusion="diff")
    work = run / "agent_work" / "r2.p0"
    work.mkdir(parents=True, exist_ok=True)
    (work / "FEEDBACK.md").write_text(
        f"# 被否决的前提\n- {sha_a} versus {sha_b} differs by {RATIO}\n",
        encoding="utf-8",
    )
    _grant(root)
    result = _settle(run, root)
    assert result == {"applied": [], "shadow": []}
    assert "route_m" not in result
    traces = _traces(run)
    assert len(traces) == 1
    trace = traces[0]
    assert [gate["name"] for gate in trace["gates"]] == list(GATE_ORDER)
    assert all(gate["status"] == "pass" for gate in trace["gates"])
    assert trace["shadow_written"] is True
    assert trace["changes_activation"] is False
    assert trace["delivery"]["delivered"] is False
    assert trace["delivery"]["reason"] == "shadow"
    assert trace["delivery"]["injected_bytes"] == 0
    assert trace["delivery"]["text_bytes"] == len(trace["task_line"].encode("utf-8"))
    assert trace["evidence_role"] == "host_paired_comparison"
    line = trace["task_line"]
    assert line.startswith(trace["measured_text"])
    assert "模型说法（提名）" in line
    assert "diff" in trace["model_claim_in_line"]
    assert f"芯片 {CHIP}" in line and f"题目 {TASK}" in line
    assert "没差" in trace["measured_text"]
    assert "不点名特征差" in trace["measured_text"]
    assert RATIO in trace["model_claim_raw"]
    assert RATIO not in line and INTERVAL not in line
    assert not re.search(r"\d+\.\d+", line)
    rows = _overlay(run)
    assert len(rows) == 1
    assert rows[0]["shadow"] is True and rows[0]["route"] == "M"
    assert rows[0]["changes_activation"] is False
    assert rows[0]["text"] == line
    host_bytes = (run / "candidates" / "002_r2.p0" / "paired_comparison.json").read_bytes()
    assert rows[0]["evidence_sha256"] == hashlib.sha256(host_bytes).hexdigest()
    assert harness.overlay_text(
        run, target_id=CHIP, task=TASK, protocol_version=trust.current_protocol_version(),
    ) == ""
    assert harness.board(run)["route_m"][0]["gates"][0]["name"] == "route_proposal"
    again = _settle(run, root)
    assert again == {"applied": [], "shadow": []}
    assert len(_traces(run)) == 1
    assert len(_overlay(run)) == 1


def test_a_structural_diff_names_the_feature_and_not_the_ratio(tmp_path: Path):
    """@brief 有差只写两侧 kernel 的特征差和较快的一侧，不写单点比值。"""
    run, root, sha_b_expected, _sha_a = _plant(
        tmp_path, conclusion="diff", direction="b_faster", sources=True,
    )
    # _plant assigns sha_a to kernel A and sha_b to kernel B when sources are on.
    sha_b = _digest(KERNEL_B)
    assert sha_b_expected == _digest(KERNEL_A)
    _grant(root)
    _settle(run, root)
    trace = _traces(run)[0]
    assert trace["shadow_written"] is True
    assert trace["structural_attribution"]["only_b"] == ["unroll"]
    line = trace["task_line"]
    assert "有差" in line and "unroll" in line
    assert f"较快的是 {sha_b[:8]} 侧" in line
    assert RATIO not in line and not re.search(r"\d+\.\d+", line)
    assert harness.overlay_text(
        run, target_id=CHIP, task=TASK, protocol_version=trust.current_protocol_version(),
    ) == ""


def test_diff_without_a_feature_delta_is_not_delivered(tmp_path: Path):
    """@brief 有差但说不出可复用的特征差时，不写影子行。"""
    run, root, _sha_a, _sha_b = _plant(tmp_path, conclusion="diff", direction="a_faster")
    _grant(root)
    _settle(run, root)
    trace = _traces(run)[0]
    assert trace["shadow_written"] is False
    assert trace["gates"][0]["status"] == "not_run"
    assert "no reusable structural attribution" in trace["gates"][0]["reason"]
    assert _overlay(run) == []


def test_too_noisy_is_recorded_and_not_delivered(tmp_path: Path):
    """@brief 噪声太大留痕，不写成可以送达的结论。"""
    run, root, _sha_a, _sha_b = _plant(tmp_path, conclusion="too_noisy")
    _grant(root)
    _settle(run, root)
    trace = _traces(run)[0]
    assert trace["shadow_written"] is False
    assert trace["conclusion"] == "too_noisy"
    assert all(gate["status"] == "not_run" for gate in trace["gates"])
    assert "too_noisy" in trace["gates"][0]["reason"]
    assert _overlay(run) == []


def test_a_model_nomination_alone_does_not_become_evidence(tmp_path: Path):
    """@brief 工作目录里的提名，哪怕自报 host_evidence，也过不了证据这道。"""
    run = tmp_path / "run"
    work = run / "agent_work" / "r2.p0" / ".selfcheck"
    work.mkdir(parents=True)
    sha_a, sha_b = _digest("left"), _digest("right")
    (work / "paired_comparisons.jsonl").write_text(json.dumps({
        "schema": "taic.paired_nomination.1",
        "sha_a": sha_a, "sha_b": sha_b, "conclusion": "no_diff",
        "claim": "the kernels have no difference",
        "source": "model", "host_evidence": True,
    }) + "\n", encoding="utf-8")
    (work / "host_paired_comparisons.jsonl").write_text(json.dumps({
        "schema": "taic.paired_comparison.1", "applied": True, "gpu": True,
        "sha_a": sha_a, "sha_b": sha_b, "conclusion": "no_diff", "host_evidence": True,
    }) + "\n", encoding="utf-8")
    _grant(tmp_path / "targets")
    _settle(run, tmp_path / "targets")
    traces = _traces(run)
    assert traces and all(row["evidence_role"] == "nomination" for row in traces)
    assert all(row["shadow_written"] is False for row in traces)
    assert _overlay(run) == []


def test_trust_uses_the_session_record_and_not_the_comparison_file(tmp_path: Path):
    """@brief 比较文件里写一个受信任的模型，不能代替宿主会话记录里的作者。"""
    run, root, _sha_a, _sha_b = _plant(
        tmp_path, provider="nobody", model="untrusted-model",
        extra={"provider": "moonshot", "model": "kimi-k3"},
    )
    trust.grant(CHIP, TASK, "moonshot", "kimi-k3", effort="high", scopes=["overlay:tip"], root=root)
    _settle(run, root)
    trace = _traces(run)[0]
    names = [gate["name"] for gate in trace["gates"]]
    assert names == list(GATE_ORDER)
    assert [gate["status"] for gate in trace["gates"][:4]] == ["pass", "pass", "pass", "pass"]
    trust_gate = trace["gates"][4]
    assert trust_gate["name"] == "overlay_tip_trust"
    assert trust_gate["passed"] is False
    assert trace["shadow_written"] is False
    assert _overlay(run) == []


def test_simulated_measurement_stops_at_the_simulation_gate(tmp_path: Path):
    """@brief 模拟测量在非模拟门停下，前面的门照样留结果。"""
    run, root, _sha_a, _sha_b = _plant(tmp_path, simulated=True)
    _grant(root)
    _settle(run, root)
    trace = _traces(run)[0]
    assert [gate["status"] for gate in trace["gates"]] == ["pass", "pass", "pass", "fail", "pass"]
    assert trace["gates"][3]["reason"] == "simulated measurement"
    assert trace["shadow_written"] is False
    assert _overlay(run) == []


def test_a_task_name_that_hits_never_auto_does_not_shadow(tmp_path: Path):
    """@brief 题目里带 tasks/ 时，never-auto 拒绝这一行。"""
    blocked = "tasks/conv2d"
    run, root, _sha_a, _sha_b = _plant(tmp_path, task=blocked)
    _grant(root, task=blocked)
    _settle(run, root, task=blocked)
    trace = _traces(run)[0]
    assert trace["gates"][0]["passed"] is True
    assert trace["gates"][1]["name"] == "never_auto"
    assert trace["gates"][1]["passed"] is False
    assert trace["shadow_written"] is False
    assert _overlay(run) == []


def test_scope_mismatch_is_its_own_gate(tmp_path: Path):
    """@brief 比较属于 conv2d，settle 却按 gemm 调用时，作用域门失败。"""
    run, root, _sha_a, _sha_b = _plant(tmp_path)
    _grant(root, task="gemm")
    _settle(run, root, task="gemm")
    trace = _traces(run)[0]
    assert trace["gates"][2]["name"] == "scope"
    assert trace["gates"][2]["passed"] is False
    assert "conv2d" in trace["gates"][2]["reason"]
    assert trace["shadow_written"] is False


def test_an_unapplied_host_file_is_not_a_shadow(tmp_path: Path):
    """@brief 宿主记下了没配成对的原因，这不是一条可送达的比较。"""
    run, root, _sha_a, _sha_b = _plant(tmp_path, applied=False)
    _grant(root)
    _settle(run, root)
    trace = _traces(run)[0]
    assert trace["qualified"] if "qualified" in trace else trace["evidence_role"] == "not_host_evidence"
    assert "no current seed" in trace["gates"][0]["reason"]
    assert trace["shadow_written"] is False
    assert _overlay(run) == []


def test_a_model_sentence_that_asks_for_a_tool_stays_out_of_the_line(tmp_path: Path):
    """@brief 提名里要求新增 tcenv 命令时，任务书一行只保留宿主实测。"""
    run, root, _sha_a, _sha_b = _plant(
        tmp_path,
        nomination_claim="Add a tcenv command for the kernel",
        nomination_conclusion="no_diff",
    )
    _grant(root)
    _settle(run, root)
    trace = _traces(run)[0]
    assert trace["shadow_written"] is True
    assert trace["model_claim_withheld"] is True
    assert "tcenv" not in trace["task_line"]
    assert "Add" not in trace["task_line"]
    assert "no_diff" in trace["model_claim_in_line"]
    assert "没差" in trace["measured_text"]


def test_active_settle_still_leaves_route_m_in_shadow(tmp_path: Path):
    """@brief settle 的 active 开关不把路线 M 写成生效 overlay。"""
    run, root, _sha_a, _sha_b = _plant(tmp_path)
    _grant(root)
    result = _settle(run, root, shadow=False)
    assert result["applied"] == []
    assert result["shadow"] == []
    row = _overlay(run)[0]
    assert row["shadow"] is True and row["route"] == "M"
    assert harness.overlay_text(
        run, target_id=CHIP, task=TASK, protocol_version=trust.current_protocol_version(),
    ) == ""


def test_a_hand_edited_route_m_row_is_not_injected(tmp_path: Path):
    """@brief 把路线 M 的 shadow 标志改成 false，任务书仍然不收这一行。"""
    run = tmp_path / "run"
    run.mkdir()
    protocol = trust.current_protocol_version()
    (run / harness.OVERLAY_FILE).write_text(json.dumps({
        "event": "overlay", "route": "M", "shadow": False, "surface": "tip",
        "target_id": CHIP, "task": TASK, "protocol_version": protocol,
        "proposal_key": "route-m-key", "text": "在芯片 apple-m4-max 的题目 conv2d 上对比 kernel。",
        "source_run": "run", "source_seq": 1,
    }) + "\n", encoding="utf-8")
    assert harness.overlay_rows(
        run, target_id=CHIP, task=TASK, protocol_version=protocol, shadow=False,
    ) == []
    assert harness.overlay_text(run, target_id=CHIP, task=TASK, protocol_version=protocol) == ""


def test_route_m_does_not_replace_a_quorum_overlay(tmp_path: Path):
    """@brief 同一场里，双来源 tip 仍按原规则生效；路线 M 继续只留影子。"""
    run, root, _sha_a, _sha_b = _plant(tmp_path)
    _grant(root)
    started = 1.0
    action = "先读 device.md 的线程组上限再选 tile。"
    for seq, provider, model, committee, body in (
        (1, "deepseek", "deepseek-flash", "first", "print(1)\n"),
        (2, "moonshot", "kimi-k3", "second", "print(2)\n"),
    ):
        work = run / "agent_work" / f"r{seq}.p0"
        work.mkdir(parents=True)
        (work / "probe.py").write_text(body, encoding="utf-8")
        doc = {
            "surface": "tip",
            "scope": {"target_id": CHIP, "task": TASK, "dtype": "fp16"},
            "action": action,
            "evidence": {"probe": "probe.py", "measurement": "64 threads"},
            "rationale": "test",
        }
        parsed = {"sections": {"对系统的建议": "```harness-proposal\n" + json.dumps(doc) + "\n```\n"}}
        harness.record_proposals(
            run, record={"seq": seq, "label": f"r{seq}.p0", "committee": committee, "provider": provider, "model": model, "effort": "high", "run_id": "run"},
            parsed=parsed, target_id=CHIP, task=TASK, dtype="fp16", work=work, started=started,
            protocol_version=trust.current_protocol_version(),
        )
    trust.grant(CHIP, TASK, "moonshot", "kimi-k3", effort="high", scopes=["overlay:tip"], root=root)
    result = _settle(run, root, shadow=False)
    assert len(result["applied"]) == 1
    assert result["applied"][0].get("route") != "M"
    assert action in harness.overlay_text(
        run, target_id=CHIP, task=TASK, protocol_version=trust.current_protocol_version(),
    )
    route_rows = [row for row in _overlay(run) if row.get("route") == "M"]
    assert len(route_rows) == 1 and route_rows[0]["shadow"] is True
    assert "没差" not in harness.overlay_text(
        run, target_id=CHIP, task=TASK, protocol_version=trust.current_protocol_version(),
    )


def test_disabled_settle_does_not_write_route_m(tmp_path: Path):
    """@brief harness 总开关关闭时，路线 M 也不写文件。"""
    run, root, _sha_a, _sha_b = _plant(tmp_path)
    _grant(root)
    result = _settle(run, root, enabled=False)
    assert result["reason"] == "harness_auto is off"
    assert not (run / TRACE_FILE).exists()
    assert not (run / harness.OVERLAY_FILE).exists()


def test_route_m_does_not_move_the_prompt_fingerprint():
    """@brief 路线 M 不改提示面指纹。"""
    from tests.support.live_anchors import load_live_anchors

    anchors = load_live_anchors()
    assert trust.current_prompt_sha1() == anchors["prompt_sha1"]


def _selection_document(sha_a: str, sha_b: str, path_a: Path, path_b: Path) -> dict:
    """@brief 一份宿主选择记录：测量是证据，路径要再核对。"""
    return {
        "schema": "taic.paired_host_selection.1",
        "selected": [{
            "sha_a": sha_a,
            "sha_b": sha_b,
            "sha_pair": sorted((sha_a, sha_b)),
            "path_a": str(path_a),
            "path_b": str(path_b),
            "conclusion": "diff",
            "features": ["model_invented_feature"],
        }],
        "measurements": [{
            "schema": "taic.paired_comparison.1",
            "applied": True,
            "gpu": True,
            "backend": "metal",
            "sha_a": sha_a,
            "sha_b": sha_b,
            "conclusion": "diff",
            "direction": "b_faster",
            "changes_seed": False,
            "features": ["model_invented_feature"],
            "ratios": [float(RATIO)],
        }],
    }


def _plant_selection(tmp: Path, *, where: str, mode: str) -> tuple[Path, Path]:
    """@brief 种一份 candidate 或 session 的宿主选择记录。

    ``mode`` 为 ``match``、``tamper`` 或 ``outside``。后两种的路径不能提供特征。
    """
    run = tmp / "run"
    root = tmp / "targets"
    sha_a, sha_b = _digest(KERNEL_A), _digest(KERNEL_B)
    if where == "candidate":
        folder = run / "candidates" / "002_r2.p0"
        label = "r2.p0"
    else:
        folder = run / "sessions" / "init"
        label = "init"
    folder.mkdir(parents=True)
    owned = run / "agent_work" / label
    owned.mkdir(parents=True)
    path_a = owned / "a.metal"
    path_b = owned / "b.metal"
    path_a.write_text(KERNEL_A, encoding="utf-8")
    path_b.write_text(KERNEL_B, encoding="utf-8")
    if mode == "tamper":
        path_b.write_text(KERNEL_B + "\n// tampered\n", encoding="utf-8")
    elif mode == "outside":
        escaped = tmp / "outside"
        escaped.mkdir()
        path_a = escaped / "a.metal"
        path_b = escaped / "b.metal"
        path_a.write_text(KERNEL_A, encoding="utf-8")
        path_b.write_text(KERNEL_B, encoding="utf-8")
    (folder / "paired_host_selection.json").write_text(
        json.dumps(_selection_document(sha_a, sha_b, path_a, path_b)),
        encoding="utf-8",
    )
    (run / "feedback.jsonl").write_text(json.dumps({
        "label": label, "seq": 1, "provider": PROVIDER, "model": MODEL, "effort": "high",
        "task": TASK, "target_id": CHIP, "backend": "metal", "simulated": False,
        "measurement_kind": "measured",
    }) + "\n", encoding="utf-8")
    (run / "run_config.json").write_text(
        json.dumps({"target": {"id": CHIP, "backend": "metal"}, "task": TASK}),
        encoding="utf-8",
    )
    return run, root


def test_host_selection_rehashes_sources_for_a_structural_diff(tmp_path: Path):
    """@brief candidate 和 session 的选择记录都要按字节核对源码后才能写有差。"""
    for where in ("candidate", "session"):
        run, root = _plant_selection(tmp_path / where, where=where, mode="match")
        _grant(root)
        _settle(run, root)
        trace = _traces(run)[0]
        assert trace["evidence_path"].endswith("paired_host_selection.json")
        assert trace["shadow_written"] is True
        assert trace["structural_attribution"]["only_b"] == ["unroll"]
        assert "unroll" in trace["task_line"]
        assert "model_invented_feature" not in trace["task_line"]
        assert RATIO not in trace["task_line"]
        assert _overlay(run)[0]["route"] == "M" and _overlay(run)[0]["shadow"] is True


def test_a_tampered_or_escaped_host_selection_source_is_not_attributed(tmp_path: Path):
    """@brief 源码被改过，或路径跑到 run 外面，有差不能靠模型路径送达。"""
    for where in ("candidate", "session"):
        for mode in ("tamper", "outside"):
            run, root = _plant_selection(tmp_path / f"{where}-{mode}", where=where, mode=mode)
            # A same-SHA copy already on the host index must not fill in a failed re-read.
            paired = run / "paired-sources"
            paired.mkdir()
            (paired / "a.metal").write_text(KERNEL_A, encoding="utf-8")
            (paired / "b.metal").write_text(KERNEL_B, encoding="utf-8")
            _grant(root)
            _settle(run, root)
            trace = _traces(run)[0]
            assert trace["shadow_written"] is False
            assert trace["structural_attribution"]["kind"] == "unknown"
            assert "no reusable structural attribution" in trace["gates"][0]["reason"]
            assert _overlay(run) == []


def test_a_code_marked_nomination_keeps_only_the_conclusion_word(tmp_path: Path):
    """@brief 反引号和 Shell 语法不能进任务书；控制字符按 P1 清掉。"""
    run, root, _sha_a, _sha_b = _plant(
        tmp_path / "code",
        nomination_claim="`keep unroll`",
        nomination_conclusion="diff",
    )
    _grant(root)
    _settle(run, root)
    trace = _traces(run)[0]
    assert trace["shadow_written"] is True
    assert trace["model_claim_withheld"] is True
    assert trace["model_claim_raw"] == "`keep unroll`"
    assert "`" not in trace["task_line"]
    assert "keep unroll" not in trace["task_line"]
    assert "diff" in trace["model_claim_in_line"]
    assert "not a plain sentence" in trace["withheld_reason"]

    plain, plain_root, _left, _right = _plant(
        tmp_path / "plain",
        nomination_claim="\x00keep the unroll pragma",
        nomination_conclusion="no_diff",
    )
    _grant(plain_root)
    _settle(plain, plain_root)
    kept = _traces(plain)[0]
    assert kept["model_claim_withheld"] is False
    assert "\x00" not in kept["task_line"]
    assert "keep the unroll pragma" in kept["task_line"]


def test_a_session_without_proposals_still_shadows_the_host_comparison(tmp_path: Path, monkeypatch):
    """@brief 没有围栏提案时，失败和成功会话都在证据与作者记录齐备后写路线 M。"""
    import asyncio

    from taic.kernel_agent import AgentError, AgentResult
    from taic.paired_compare import append_nomination
    from tests.support.orchestrator import code, llm_script, make_orch

    def run_session(directory: Path, *, fail: bool) -> None:
        orch = make_orch(directory, llm_script(code("speed=1.0")), [], max_rounds=1)
        orch.search.harness_auto = True
        orch._trust_root = directory / "trust"
        member = orch.committees[0].members[0]
        trust.grant(
            orch.target_id, orch.task_name, member.provider, member.model,
            effort=member.cfg.reasoning_effort or "",
            scopes=["overlay:tip"], root=orch._trust_root,
        )
        host_dir = directory / "host-authority"
        monkeypatch.setattr(
            "taic.paired_compare.host_evidence_path",
            lambda _checked: host_dir / "host_paired_comparisons.jsonl",
        )
        selection = orch.run_dir / "sessions" / "init" / "paired_host_selection.json"
        order: list[str] = []
        real_feedback = orch._record_feedback
        real_select = orch._select_paired_nominations

        def feedback(*args, **kwargs):
            order.append("feedback")
            if fail:
                assert not selection.exists()
            else:
                assert selection.is_file()
            return real_feedback(*args, **kwargs)

        async def select(*args, **kwargs):
            order.append("select")
            return await real_select(*args, **kwargs)

        orch._record_feedback = feedback
        orch._select_paired_nominations = select
        orch._paired_runner_binder = lambda _path_a, _path_b: (lambda _k: 1e-4, lambda _k: 1e-4)

        def agent(brief, _tag):
            work = Path(brief.work_dir)
            left = work / "a.metal"
            right = work / "b.metal"
            left.write_text(KERNEL_A, encoding="utf-8")
            right.write_text(KERNEL_B, encoding="utf-8")
            append_nomination(work / ".selfcheck", {
                "sha_a": _digest(KERNEL_A), "sha_b": _digest(KERNEL_B),
                "path_a": str(left), "path_b": str(right), "conclusion": "no_diff",
            })
            (work / "FEEDBACK.md").write_text("# 对系统的建议\n- no fence here\n", encoding="utf-8")
            if fail:
                raise AgentError("no kernel")
            return AgentResult(
                source=KERNEL_A + "\n// speed=1.0\n", cost_usd=0.01,
                cost_incomplete=False, status="ok",
            )

        orch.agent_runner = agent
        outcome = asyncio.run(orch._one_session_attempt("init", seedless=True, candidate_label="init", attempt=0))
        assert outcome == ("agent_failed" if fail else "evaluated")
        assert order == (["feedback", "select"] if fail else ["select", "feedback"])
        assert not (orch.run_dir / harness.PROPOSAL_FILE).exists()
        traces = _traces(orch.run_dir)
        host = [row for row in traces if str(row.get("evidence_path") or "").endswith("paired_host_selection.json")]
        assert len(host) == 1
        assert host[0]["shadow_written"] is True
        assert host[0]["conclusion"] == "no_diff"
        assert "unroll" in host[0]["task_line"]
        assert "没差" in host[0]["task_line"]
        assert not re.search(r"\d+\.\d+", host[0]["task_line"])
        rows = [row for row in _overlay(orch.run_dir) if row.get("route") == "M"]
        assert len(rows) == 1 and rows[0]["shadow"] is True
        assert harness.overlay_text(
            orch.run_dir, target_id=orch.target_id, task=orch.task_name,
            protocol_version=trust.current_protocol_version(),
        ) == ""

    run_session(tmp_path / "failed", fail=True)
    run_session(tmp_path / "ok", fail=False)


def test_an_untrusted_other_session_nomination_stays_out_of_the_shadow_line(tmp_path: Path):
    """@brief 另一场不受信任的提名不能借宿主测量会话的 grant 进影子行。"""
    run, root, sha_a, sha_b = _plant(tmp_path)
    stolen = "keep unroll for this kernel"
    with (run / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "label": "r1.p0", "seq": 1, "provider": "nobody", "model": "untrusted-local",
            "effort": "high", "task": TASK, "target_id": CHIP, "backend": "metal",
        }) + "\n")
    other = run / "agent_work" / "r1.p0" / ".selfcheck"
    other.mkdir(parents=True)
    nomination = json.dumps({
        "schema": "taic.paired_nomination.1",
        "sha_a": sha_a, "sha_b": sha_b, "conclusion": "diff", "claim": stolen, "source": "model",
    }) + "\n"
    (other / "paired_comparisons.jsonl").write_text(nomination, encoding="utf-8")
    nested = run / "agent_work" / "r1.p0" / "r2.p0" / ".selfcheck"
    nested.mkdir(parents=True)
    (nested / "paired_comparisons.jsonl").write_text(nomination, encoding="utf-8")
    (run / harness.PROPOSAL_FILE).write_text("".join(json.dumps(row) + "\n" for row in (
        {
            "status": "valid", "label": "r1.p0",
            "proposal": {"surface": "tip", "action": f"{stolen} {sha_a} {sha_b}"},
        },
        {
            "status": "invalid", "label": "r2.p0",
            "proposal": {"surface": "tip", "action": f"do not quote this invalid fence {sha_a} {sha_b}"},
        },
    )), encoding="utf-8")
    _grant(root)
    _settle(run, root)
    traces = _traces(run)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["shadow_written"] is True
    assert trace["session_label"] == "r2.p0"
    assert trace["nomination_session"] == ""
    assert trace["author"]["provider"] == PROVIDER
    assert trace["author"]["model"] == MODEL
    assert trace["author"]["session_label"] == "r2.p0"
    assert stolen not in trace["task_line"]
    assert stolen not in trace["model_claim_raw"]
    assert stolen not in trace["model_claim_in_line"]
    assert "do not quote this invalid fence" not in trace["model_claim_raw"]
    assert trace["gates"][4]["name"] == "overlay_tip_trust"
    assert trace["gates"][4]["passed"] is True


def test_a_provider_retry_binds_the_session_that_measured(tmp_path: Path):
    """@brief 重试会话的选择记录不沿用候选目录上的旧标签。"""
    sha_a, sha_b = _digest("kernel-a"), _digest("kernel-b")
    measurement = {
        "schema": "taic.paired_comparison.1",
        "applied": True, "gpu": True, "backend": "metal",
        "sha_a": sha_a, "sha_b": sha_b, "conclusion": "no_diff",
        "direction": None, "changes_seed": False, "ratios": [float(RATIO)],
    }
    old = "keep unroll for this kernel"
    fresh = "keep the unroll pragma"

    def feedback(label: str, provider: str, model: str, seq: int) -> dict:
        return {
            "label": label, "seq": seq, "provider": provider, "model": model, "effort": "high",
            "task": TASK, "target_id": CHIP, "backend": "metal", "simulated": False,
        }

    def nominate(run: Path, label: str, claim: str) -> None:
        work = run / "agent_work" / label / ".selfcheck"
        work.mkdir(parents=True)
        (work / "paired_comparisons.jsonl").write_text(json.dumps({
            "schema": "taic.paired_nomination.1",
            "sha_a": sha_a, "sha_b": sha_b, "conclusion": "no_diff", "claim": claim, "source": "model",
        }) + "\n", encoding="utf-8")

    def plant_run(directory: Path) -> tuple[Path, Path]:
        run = directory / "run"
        root = directory / "targets"
        run.mkdir(parents=True)
        (run / "run_config.json").write_text(
            json.dumps({"target": {"id": CHIP, "backend": "metal"}, "task": TASK}),
            encoding="utf-8",
        )
        (run / "feedback.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (
            feedback("r2.p0", "nobody", "untrusted-local", 2),
            feedback("r2.p0.retry1", PROVIDER, MODEL, 4),
        )), encoding="utf-8")
        nominate(run, "r2.p0", old)
        nominate(run, "r2.p0.retry1", fresh)
        (run / harness.PROPOSAL_FILE).write_text(json.dumps({
            "status": "valid", "label": "r2.p0",
            "proposal": {"surface": "tip", "action": f"{old} {sha_a} {sha_b}"},
        }) + "\n", encoding="utf-8")
        _grant(root)
        return run, root

    ordered, ordered_root = plant_run(tmp_path / "order")
    session = ordered / "sessions" / "r2.p0.retry1"
    session.mkdir(parents=True)
    (session / "paired_host_selection.json").write_text(json.dumps({
        "schema": "taic.paired_host_selection.1",
        "session_label": "r2.p0",
        "measurements": [measurement],
    }), encoding="utf-8")
    candidate = ordered / "candidates" / "002_r2.p0"
    candidate.mkdir(parents=True)
    (candidate / "paired_host_selection.json").write_text(json.dumps({
        "schema": "taic.paired_host_selection.1",
        "measurements": [measurement],
    }), encoding="utf-8")
    _settle(ordered, ordered_root)
    trace = _traces(ordered)[0]
    assert len(_traces(ordered)) == 1
    assert trace["evidence_path"].startswith("sessions/r2.p0.retry1/")
    assert trace["shadow_written"] is True
    assert trace["session_label"] == "r2.p0.retry1"
    assert trace["nomination_session"] == "r2.p0.retry1"
    assert trace["author"]["model"] == MODEL
    assert trace["author"]["session_label"] == "r2.p0.retry1"
    assert fresh in trace["task_line"]
    assert old not in trace["task_line"]
    assert old not in trace["model_claim_raw"]

    saved, saved_root = plant_run(tmp_path / "saved")
    only = saved / "candidates" / "002_r2.p0"
    only.mkdir(parents=True)
    (only / "paired_host_selection.json").write_text(json.dumps({
        "schema": "taic.paired_host_selection.1",
        "session_label": "r2.p0.retry1",
        "measurements": [measurement],
    }), encoding="utf-8")
    _settle(saved, saved_root)
    copied = _traces(saved)[0]
    assert copied["evidence_path"].startswith("candidates/002_r2.p0/")
    assert copied["shadow_written"] is True
    assert copied["session_label"] == "r2.p0.retry1"
    assert copied["author"]["provider"] == PROVIDER
    assert fresh in copied["task_line"]
    assert old not in copied["model_claim_raw"]


def test_a_missing_backend_does_not_write_a_shadow(tmp_path: Path):
    """@brief 本次后端或证据后端缺一边时不写影子行。两边都有但不同也不写。"""
    absent, absent_root, _sha_a, _sha_b = _plant(tmp_path / "absent", backend="", run_backend=None)
    _grant(absent_root)
    _settle(absent, absent_root)
    trace = _traces(absent)[0]
    assert trace["gates"][2]["name"] == "scope"
    assert trace["gates"][2]["passed"] is False
    assert "scope backend is missing" in trace["gates"][2]["reason"]
    assert trace["shadow_written"] is False
    assert _overlay(absent) == []

    evidence, evidence_root, _left, _right = _plant(tmp_path / "evidence", run_backend=None)
    _grant(evidence_root)
    _settle(evidence, evidence_root)
    evidence_trace = _traces(evidence)[0]
    assert evidence_trace["backend"] == "metal"
    assert evidence_trace["gates"][2]["passed"] is False
    assert "scope backend is missing" in evidence_trace["gates"][2]["reason"]
    assert evidence_trace["shadow_written"] is False

    caller, caller_root, _one, _two = _plant(tmp_path / "caller", backend="", run_backend=None)
    _grant(caller_root)
    settle_route_m(
        caller, target_id=CHIP, task=TASK, protocol_version=trust.current_protocol_version(),
        trust_root=caller_root, backend="metal",
    )
    caller_trace = _traces(caller)[0]
    assert caller_trace["backend"] == ""
    assert caller_trace["gates"][2]["passed"] is False
    assert "scope backend is missing" in caller_trace["gates"][2]["reason"]
    assert caller_trace["shadow_written"] is False
    assert _overlay(caller) == []

    mismatch, mismatch_root, _a, _b = _plant(tmp_path / "mismatch")
    _grant(mismatch_root)
    settle_route_m(
        mismatch, target_id=CHIP, task=TASK, protocol_version=trust.current_protocol_version(),
        trust_root=mismatch_root, backend="cuda",
    )
    mismatch_trace = _traces(mismatch)[0]
    assert mismatch_trace["gates"][2]["passed"] is False
    assert "cuda" in mismatch_trace["gates"][2]["reason"]
    assert mismatch_trace["shadow_written"] is False
    assert _overlay(mismatch) == []


def test_store_paired_binds_the_attempt_session_label(tmp_path: Path):
    """@brief 宿主写 paired comparison 时记下 origin 里的实际会话，不用候选目录名。"""
    from tests.support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)

    class Retry:
        idx = 2
        label = "r2.p0"
        origin = {"origin": "session", "session_label": "r2.p0.retry1", "candidate_label": "r2.p0", "attempt": 1}

    orch._store_paired(Retry(), {
        "schema": "taic.paired_comparison.1", "applied": False, "changes_seed": False,
        "session_label": "r2.p0", "reason": "no current seed",
    })
    saved = json.loads((orch.run_dir / "candidates" / "002_r2.p0" / "paired_comparison.json").read_text(encoding="utf-8"))
    assert saved["session_label"] == "r2.p0.retry1"
    assert saved["reason"] == "no current seed"

    class Old:
        idx = 0
        label = "init"
        origin = {}

    orch._store_paired(Old(), {"schema": "taic.paired_comparison.1", "applied": False, "changes_seed": False})
    old = json.loads((orch.run_dir / "candidates" / "000_init" / "paired_comparison.json").read_text(encoding="utf-8"))
    assert "session_label" not in old


def test_a_retry_paired_comparison_cannot_borrow_the_candidate_grant(tmp_path: Path):
    """@brief 重试会话的 paired_comparison 不能用候选目录上的旧提名和 grant。"""
    stolen = "keep unroll for this kernel"
    run, root, sha_a, sha_b = _plant(tmp_path / "retry", extra={"session_label": "r2.p0.retry1"})
    with (run / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "label": "r2.p0.retry1", "seq": 4, "provider": "nobody", "model": "untrusted-retry",
            "effort": "high", "task": TASK, "target_id": CHIP, "backend": "metal", "simulated": False,
        }) + "\n")
    old = run / "agent_work" / "r2.p0" / ".selfcheck"
    old.mkdir(parents=True)
    (old / "paired_comparisons.jsonl").write_text(json.dumps({
        "schema": "taic.paired_nomination.1",
        "sha_a": sha_a, "sha_b": sha_b, "conclusion": "diff", "claim": stolen, "source": "model",
    }) + "\n", encoding="utf-8")
    (run / harness.PROPOSAL_FILE).write_text(json.dumps({
        "status": "valid", "label": "r2.p0",
        "proposal": {"surface": "tip", "action": f"{stolen} {sha_a} {sha_b}"},
    }) + "\n", encoding="utf-8")
    _grant(root)
    _settle(run, root)
    trace = _traces(run)[0]
    assert trace["session_label"] == "r2.p0.retry1"
    assert trace["author"]["provider"] == "nobody"
    assert trace["author"]["model"] == "untrusted-retry"
    assert trace["author"]["session_label"] == "r2.p0.retry1"
    assert [gate["status"] for gate in trace["gates"][:4]] == ["pass", "pass", "pass", "pass"]
    assert trace["gates"][4]["name"] == "overlay_tip_trust"
    assert trace["gates"][4]["passed"] is False
    assert trace["shadow_written"] is False
    assert stolen not in trace["task_line"]
    assert stolen not in trace["model_claim_raw"]
    assert stolen not in trace["model_claim_in_line"]
    assert _overlay(run) == []

    forged, forged_root, left, right = _plant(tmp_path / "forged", extra={"session_label": "../r2.p0"})
    forged_work = forged / "agent_work" / "r2.p0" / ".selfcheck"
    forged_work.mkdir(parents=True)
    (forged_work / "paired_comparisons.jsonl").write_text(json.dumps({
        "schema": "taic.paired_nomination.1",
        "sha_a": left, "sha_b": right, "conclusion": "diff", "claim": stolen, "source": "model",
    }) + "\n", encoding="utf-8")
    _grant(forged_root)
    _settle(forged, forged_root)
    refused = _traces(forged)[0]
    assert refused["session_label"] == ""
    assert refused["author"] is None
    assert refused["shadow_written"] is False
    assert stolen not in refused["model_claim_raw"]
    assert _overlay(forged) == []
