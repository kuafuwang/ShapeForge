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
    # varlen（打包变长序列）实例：token 轴是一批不等长序列的拼接，分段边界是运行时数据。
    # 固定 shape 实例不渲染这一节，题面与加 varlen 之前逐字相同。
    cu_seqlens = getattr(instance, "cu_seqlens", None)
    if cu_seqlens is not None:
        symbols = dict(getattr(instance, "symbols", {}) or {})
        tensor_names = {binding.name for binding in instance.tensors}
        lines.append("## Varlen (packed sequence)")
        lines.append(
            "The leading tensor axis is a *packed batch*: one token axis holding "
            f"{symbols.get('num_seqs')} complete sequences of different lengths, one after another. "
            "`cu_seqlens` holds the boundaries: it has "
            f"{symbols.get('NSP1')} entries (`num_seqs + 1`), starts at 0 and ends at "
            f"`total_tokens` ({symbols.get('total_tokens')}); sequence `s` is the half-open token range "
            "`[cu_seqlens[s], cu_seqlens[s+1])`. Read the boundaries from the `cu_seqlens` tensor; "
            "they are runtime data, not compiler constants."
        )
        profile = getattr(instance, "canonical_seq_lens", None)
        if profile is not None:
            lines.append(
                "Canonical length profile, published so you can choose a schedule: "
                f"seq_lens = {list(profile)} over {symbols.get('num_seqs')} sequences and "
                f"{symbols.get('total_tokens')} tokens. It is *one* legal segmentation, not the only one: "
                "correctness reshapes the same total/count into other legal boundaries (permuted, skewed, "
                "chunk-boundary, prime-length, one long chain, zero-length segments and fresh draws). "
                "A segment may be empty (`cu_seqlens[s] == cu_seqlens[s+1]`); a tier that declares a "
                "single sequence is the degenerate single-sequence case. Anything you derive from the "
                "profile is a heuristic; correctness always comes from the values in `cu_seqlens`."
            )
        # 逐序列状态合同只对真的声明了状态张量的题打印；elementwise 的 varlen 题没有这个义务。
        if {"initial_state", "final_state"} <= tensor_names:
            lines.append(
                "Per-sequence work: any recurrent or stateful quantity is independent per sequence. "
                "A per-sequence state starts from that sequence's slice of `initial_state` and is written "
                "back to `final_state` when its segment ends; it must never be carried across a boundary "
                "and must never be assumed to start at zero."
            )
        lines.append(
            "`total_tokens`, `num_seqs` and `NSP1` are the total/as-count/shape symbols and ARE compile-time "
            "constants for this instance. A single-sequence instance is the ordinary fixed-length case."
        )
        lines.append("")
        # varlen 题额外展示受限 C99/native64 Host Program 的完整 ABI 头与子集/返回码。固定 shape
        # 题不渲染这一节，题面与加 Host Program 之前逐字相同。
        from .hostprogram.contract import host_program_statement_section

        lines.append(host_program_statement_section(instance))
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
