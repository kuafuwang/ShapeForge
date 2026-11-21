"""Prepare a controlled retain/delete TASK experiment without calling a model.

Only the previous-NOTES span may change. The control restores the full original;
the treatment keeps an explicitly supplied, smaller set of current constraints.
No production prompt or run evidence is modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from taic.omission_content_audit import _outer
from taic.orchestrator import _notes_for_next

SCHEMA = "shapeforge.omission-controlled-experiment.1"
NOTES_HEADING = "上一轮 NOTES（硬件上限类断言未附实测数字的，按未验证处理）：\n"
OMISSION_HEADING = "# 本轮省略（上下文预算）"
TRUNCATED = "…（本段已截断；全文见本目录文件）"


def replace_notes(task: str, notes: str) -> str:
    """@brief 只替换 TASK 中上一轮 NOTES 的内容，其他字节保持不变。

    @param task 已交付的 TASK 文本。
    @param notes 要放入该段的完整文本。
    @return 新的 TASK 文本。
    @raises ValueError 找不到 NOTES 标题时抛出。
    """
    start = task.find(NOTES_HEADING)
    if start < 0:
        raise ValueError("the delivered TASK has no previous-NOTES heading")
    content_start = start + len(NOTES_HEADING)
    end = task.find(OMISSION_HEADING, content_start)
    if end < 0:
        end = len(task)
    return task[:content_start] + notes.strip() + "\n" + TRUNCATED + "\n\n" + task[end:]


def prepare(run_dir: Path, label: str, curated_notes: str) -> dict[str, Any]:
    """@brief 为指定会话生成 full-control 与 curated-treatment 两份 TASK。

    @param run_dir 包含 result.json、candidates 和 agent_work 的 run 目录。
    @param label 会话标签。
    @param curated_notes 处理臂保留的信息。
    @return 两份文本、摘要和校验信息。
    """
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    trace = result.get("trace") or []
    index = next((i for i, row in enumerate(trace) if row.get("label") == label), None)
    if not index:
        raise ValueError("the session has no previous candidate")
    previous = trace[index - 1]
    folder = run_dir / "candidates" / f"{previous['idx']:03d}_{previous['label']}"
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    source = _notes_for_next(_outer(previous, report), (folder / "NOTES.md").read_text(encoding="utf-8", errors="replace"))
    delivered_path = run_dir / "sessions" / label / "task_delivery.json"
    delivery = json.loads(delivered_path.read_text(encoding="utf-8"))
    block = next((row for row in delivery.get("blocks") or [] if row.get("name") == "上一轮 NOTES"), None)
    if block is None or len(source.encode("utf-8")) != block["source_bytes"]:
        raise ValueError("the reconstructed NOTES do not match the receipt source_bytes")
    delivered_task = (run_dir / "agent_work" / label / "TASK.md").read_text(encoding="utf-8")
    control = replace_notes(delivered_task, source)
    treatment = replace_notes(delivered_task, curated_notes)
    def outside(text: str) -> tuple[str, str]:
        start = text.find(NOTES_HEADING)
        end = text.find(OMISSION_HEADING, start + len(NOTES_HEADING))
        return text[:start], text[end:]
    assert outside(control) == outside(treatment) == outside(delivered_task)
    return {
        "schema": SCHEMA,
        "run_id": str(run_dir),
        "session_label": label,
        "source_bytes": block["source_bytes"],
        "control_notes": source,
        "treatment_notes": curated_notes.strip(),
        "control_task": control,
        "treatment_task": treatment,
        "outside_notes_identical": True,
        "control_task_sha256": hashlib.sha256(control.encode()).hexdigest(),
        "treatment_task_sha256": hashlib.sha256(treatment.encode()).hexdigest(),
        "model_calls": 0,
        "gpu_calls": 0,
    }


__all__ = ["SCHEMA", "replace_notes", "prepare"]
