"""Real CLI module entrypoints in isolated subprocesses; no GPU or model calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def _invoke(tmp_path: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    """@brief 用临时 runs/index 环境执行真实 ``python -m taic.cli``。"""
    env = os.environ.copy()
    env.update({
        "AI_LOWERLING_RUNS_DIR": str(tmp_path / "runs"),
        "AI_LOWERLING_OBSERVE_DIR": str(tmp_path / "observe"),
        "OBSERVE_DB": str(tmp_path / "observe.sqlite"),
        "AI_LOWERLING_BOOKS": str(tmp_path / "books.sqlite"),
        "TAIC_REVIEW_EVENTS": str(tmp_path / "review_events.jsonl"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return subprocess.run(
        [sys.executable, "-m", "taic.cli", *args],
        cwd=ROOT,
        env=env,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_brief_module_entrypoint_returns_json(tmp_path):
    """@brief brief 不只进程内可调用：真实模块入口返回可解析 JSON。"""
    run = tmp_path / "runs" / "conv"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "size": "small", "phase": "done",
        "model": "scripted", "round": 0, "spent_usd": 0, "budget_usd": 1,
        "best_speedup": 1.25, "trace": [],
    }), encoding="utf-8")
    result = _invoke(tmp_path, "brief", str(run))
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["task"] == "conv2d"
    assert document["search_best"] == 1.25
    assert document["paths"]["result"] is None


def test_note_module_entrypoint_writes_a_structured_annotation(tmp_path):
    """@brief note 的真实 CLI 子进程写结构化 annotation，不触碰模型或索引中的真实 runs。"""
    work = tmp_path / "work"
    work.mkdir()
    result = _invoke(
        tmp_path, "note", "--work", str(work), "--section", "note", "--text", "subprocess annotation",
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["schema"] == "taic.note.1"
    rows = (work / ".notes" / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["text"] == "subprocess annotation"


def test_feedback_dry_run_stdin_is_a_real_no_write_subprocess(tmp_path):
    """@brief feedback 从 stdin 预览批次：真实入口返回 0，且不写 review 文件。"""
    run = tmp_path / "runs" / "conv"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "trace": [],
    }), encoding="utf-8")
    from taic.agent_feedback import parse_feedback

    parsed = parse_feedback("# 对系统的建议\n- need a probe\n")
    (run / "feedback.jsonl").write_text(json.dumps({
        "format": "agent_feedback", "v": 1, "seq": 1, "t": 1,
        "task": "conv2d", "dtype": "fp16", "label": "init", "work_dir": "agent_work/init",
        "items": parsed["items"], "unverified_premises": parsed["unverified_premises"],
        "sections": parsed["sections"], "kinds": parsed["kinds"], "raw": parsed["raw"],
    }) + "\n", encoding="utf-8")
    digest = parsed["items"][0]["digest"]
    batch = json.dumps([{
        "seq": 1, "digest": digest, "status": "info",
        "verification": "unchecked", "expected_status": "pending",
    }])
    result = _invoke(tmp_path, "feedback", "conv", "--file", "-", "--dry-run", stdin=batch)
    assert result.returncode == 0, result.stderr
    assert not (run / "feedback_review.json").exists()
