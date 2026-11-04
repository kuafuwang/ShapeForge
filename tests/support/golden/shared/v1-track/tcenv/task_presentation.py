"""Deterministic presentation of a task contract (no ABI derivation, no tiling).

Both the v2 task-contract loader and the TrackRenderer use this module, so the statement a
candidate reads has exactly one renderer.  It formats *declared facts only*: tensors, attrs,
cases, tolerances and work accounting.  It never reads a track, a grid, a tile, a per-task
lowering profile or an author identity, which is what makes the same contract produce the
same statement for a human, an LLM, a committee or a session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskPresentation:
    """@brief 一个 TrackRenderer 的确定性输出。"""

    track_id: str
    track_version: int
    statement: str
    reference_source: str
    abi_document: Mapping[str, Any]

    def document(self) -> dict[str, Any]:
        """@brief 展示文档（不包含作者/会话/委员会）。

        @return 字段字典。
        """
        return {
            "track_id": self.track_id,
            "track_version": int(self.track_version),
            "statement": self.statement,
            "reference_source": self.reference_source,
            "abi": dict(self.abi_document),
        }


def render_statement(instance: Any, display: Any) -> str:
    """@brief 由 OpSpec 事实与表示覆盖生成确定性题面。

    @param instance ``SemanticInstanceV2``（鸭子类型，避免模块循环）。
    @param display ``DisplayOverrides``（鸭子类型）。
    @return 题面文本。
    """
    spec_text = str(getattr(display, "spec_text", "") or "")
    if spec_text:
        return spec_text
    description = str(getattr(display, "description", "") or "")
    formula = str(getattr(display, "formula", "") or getattr(instance, "formula", "") or "")
    lines: list[str] = []
    if description:
        lines.extend((description, ""))
    lines.append(f"# Task: {instance.op_name} ({instance.category})")
    lines.append(f"# Instance: dtype={instance.dtype} size={instance.size}")
    if formula:
        lines.append(f"# Summary: {formula}")
    lines.append("")
    lines.append("## Tensors (declaration order)")
    for binding in instance.tensors:
        shape = "[" + ", ".join(str(dim) for dim in binding.shape) + "]"
        lines.append(
            f"- {binding.name}: role={binding.role} dtype={binding.dtype} shape={shape} "
            f"layout={binding.layout} strides={list(binding.strides)} space={binding.space} access={binding.access}"
        )
    lines.append("")
    lines.append("## Attrs (resolved)")
    if instance.resolved_attrs:
        lines.append(", ".join(f"{name}={value!r}" for name, value in instance.resolved_attrs.items()))
    else:
        lines.append("(none)")
    lines.append("")
    lines.append(f"## Cases: {', '.join(instance.case_domain.cases)}")
    lines.append(
        f"## Tolerances: rtol={instance.tolerances.rtol!r} atol_rel={instance.tolerances.atol_rel!r} "
        f"exact={instance.tolerances.exact}"
    )
    lines.append(
        f"## Work accounting: flops={instance.work_accounting.flops!r} "
        f"min_bytes={instance.work_accounting.min_bytes!r}"
    )
    discovery = getattr(instance, "discovery", None)
    if discovery:
        lines.append(f"## Discovery: {dict(discovery)!r}")
    return "\n".join(lines)


__all__ = ["TaskPresentation", "render_statement"]
