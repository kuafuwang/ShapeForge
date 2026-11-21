"""Controlled NOTES replacement changes only the intended span."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.omission_controlled_experiment import prepare, replace_notes


def test_replace_notes_preserves_every_other_byte():
    task = "before\n上一轮 NOTES（硬件上限类断言未附实测数字的，按未验证处理）：\nold\n…（本段已截断；全文见本目录文件）\n\n# 本轮省略（上下文预算）\nafter\n"
    result = replace_notes(task, "new notes")
    assert result.startswith("before\n")
    assert result.endswith("# 本轮省略（上下文预算）\nafter\n")
    assert "new notes" in result and "old" not in result


def test_prepare_produces_full_and_curated_tasks_with_identical_outside(tmp_path):
    run = tmp_path / "run-a"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"trace": [
        {"idx": 1, "label": "r1.p0", "improved": True},
        {"idx": 2, "label": "r2.p0", "improved": False},
    ]}), encoding="utf-8")
    folder = run / "candidates" / "001_r1.p0"
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps({"candidate_us": 1.0, "status": "ok"}), encoding="utf-8")
    (folder / "NOTES.md").write_text("full old notes", encoding="utf-8")
    source = "外层实测：1 µs，状态 ok，换了种子\nfull old notes".strip()
    (run / "sessions" / "r2.p0").mkdir(parents=True)
    (run / "sessions" / "r2.p0" / "task_delivery.json").write_text(json.dumps({
        "blocks": [{"name": "上一轮 NOTES", "source_bytes": len(source.encode()), "omitted_bytes": 1}],
    }), encoding="utf-8")
    (run / "agent_work" / "r2.p0").mkdir(parents=True)
    (run / "agent_work" / "r2.p0" / "TASK.md").write_text(
        "before\n上一轮 NOTES（硬件上限类断言未附实测数字的，按未验证处理）：\nold\n# 本轮省略（上下文预算）\nafter\n",
        encoding="utf-8",
    )
    document = prepare(run, "r2.p0", "curated")
    assert document["outside_notes_identical"] is True
    assert "full old notes" in document["control_task"]
    assert "curated" in document["treatment_task"]
    assert document["control_task"] != document["treatment_task"]
    assert document["model_calls"] == 0 and document["gpu_calls"] == 0
