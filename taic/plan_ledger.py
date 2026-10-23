"""Structured record of what a plan actually established.

An ABI error, a compile error, or a reply with no kernel does not show that the direction is slow.
Only a kernel that is safe and correct, and not faster, falsifies the direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PlanNode:
    """@brief 计划账本里的一条记录：某轮某分支真正确立了什么。

    ``outcome_kind`` 是 ``validated_win``/``validated_tie``/``falsified``/``untested_impl_fail`` 之一；
    ``stage`` 是原始状态或 ``history``；``constraint_hint`` 是重试该方向时的约束提示。
    """

    round_id: int
    label: str
    title: str
    outcome_kind: str
    stage: str
    speedup: float | None
    parent_speedup: float | None
    constraint_hint: str


def classify_outcome(status: str, message: str, *, improved: bool, within_noise: bool, speedup: float | None, parent_speedup: float | None) -> tuple[str, str]:
    """@brief 把一次评测结果归类为计划账本的结果类型。

    ABI 错误、编译错误或没有交出内核，不能说明方向慢；只有安全、正确且没变快的内核才算否证。

    @param status 报告的 ``status``。
    @param message 失败信息或历史行标签，用于识别「没有内核」这类情况。
    @param improved 轨迹里记录的当时是否换掉了种子。
    @param within_noise 候选是否落在噪声带内。
    @param speedup 本候选的加速比；未计时时为 ``None``。
    @param parent_speedup 评测时的种子加速比；没有种子时为 ``None``。
    @return ``(outcome_kind, constraint_hint)``；``untested_impl_fail`` 表示这个方向没被计时。
    """
    text = message or ""
    if status in ("abi_error", "compile_error", "rejected", "no_code") or "no code" in text or "no valid kernel" in text:
        hint = ""
        if "32768" in text or "threadgroup" in text.lower():
            hint = "static threadgroup memory exceeded 32768 B; a retry of this direction must shrink the tile or reuse one buffer"
        elif status in ("no_code",) or "no code" in text:
            hint = "no kernel was returned, so this direction was not measured"
        elif status == "compile_error":
            hint = "the kernel did not compile, so this direction was not measured"
        else:
            hint = "the kernel did not reach a timed measurement"
        return "untested_impl_fail", hint
    if status in ("incorrect", "unsafe", "runtime_error"):
        return "untested_impl_fail", "the kernel failed before a timing that could judge the direction"
    if status == "ok" and improved:
        return "validated_win", ""
    if status == "ok" and within_noise:
        return "validated_tie", ""
    if status == "ok" and speedup is None:
        # 核对了、安全了，但没有计时（--correctness-only）。这不是「这个方向慢」，
        # 不能让规划器读到「不要重试这个方向」。
        return "untested_impl_fail", "the kernel was correct but was not timed, so the direction was not judged"
    if status == "ok" and parent_speedup is not None and speedup is not None and speedup <= parent_speedup:
        return "falsified", ""
    if status == "ok":
        return "falsified", ""
    return "untested_impl_fail", "this attempt has no timing"


def nodes_from_trace(trace: list[dict], history: list[str]) -> list[PlanNode]:
    """@brief 从历史行与每个 ``rN.pI`` 分支的最后一份报告重建账本节点。

    同一轮里后一个分支可能换掉种子，历史行会写成 ``not better``，但轨迹里 ``improved`` 仍是当时的结果；
    赢过的方向不会再被标成已证伪。第二次种子上算错的分支仍报 ``ok``，那是正确性没过，不是「写对了但没变快」。

    @param trace 运行轨迹列表，每项含 ``label``/``status``/``speedup`` 等字段。
    @param history 规划器可读的历史行，形如 ``round N: title -> tag``。
    @return 按历史行顺序排列的 ``PlanNode`` 列表。
    """
    by_branch: dict[tuple[int, int], dict] = {}
    for row in trace:
        match = re.match(r"r(\d+)\.p(\d+)", str(row.get("label") or ""))
        if match:
            by_branch[(int(match.group(1)), int(match.group(2)))] = row
    counts: dict[int, int] = {}
    nodes: list[PlanNode] = []
    for line in history:
        match = re.match(r"round (\d+): (.*) -> (.*)", line)
        if not match:
            continue
        rnd = int(match.group(1))
        title, tag = match.group(2), match.group(3)
        index = counts.get(rnd, 0)
        counts[rnd] = index + 1
        row = by_branch.get((rnd, index), {})
        status = str(row.get("status") or "")
        if not status and ("no code" in tag or "no valid kernel" in tag):
            status = "no_code"
        if not status and tag.startswith("abi"):
            status = "abi_error"
        parent = row.get("best_speedup")
        parent_speed = float(parent) if isinstance(parent, (int, float)) and not isinstance(parent, bool) else None
        speed = row.get("speedup")
        speed_f = float(speed) if isinstance(speed, (int, float)) and not isinstance(speed, bool) else None
        kind, hint = classify_outcome(status, tag, improved=bool(row.get("improved")), within_noise=bool(row.get("within_noise")), speedup=speed_f, parent_speedup=parent_speed)
        recheck = row.get("differential_recheck")
        # 同一轮里后一个分支把种子换掉后，历史行会写成 not better，但轨迹里 improved 仍是当时的结果。赢过的方向不能再标成已经证伪。
        # 第二次种子上算错的，搜索报告仍是 ok。那是正确性没过，不是「写对了但没变快」。
        if isinstance(recheck, str) and recheck not in ("", "timeout", "crash") and status == "ok":
            kind, hint = "untested_impl_fail", "a second seed failed correctness, so this direction was not timed as a loss"
        elif "not better" in tag and status == "ok" and not row.get("improved"):
            kind, hint = "falsified", ""
        nodes.append(PlanNode(rnd, f"r{rnd}.p{index}", title, kind, status or "history", speed_f, parent_speed, hint))
    return nodes


def format_ledger_for_planner(nodes: list[PlanNode], limit: int = 24) -> str:
    """@brief 生成给规划器看的账本文本，未计时的失败不被列为要避开的方向。

    @param nodes 账本节点列表。
    @param limit 只列出最后多少个节点。
    @return 多行文本，末尾附上「已证伪不要重复、未计时可带提示重试」的说明。
    """
    lines = []
    for node in nodes[-limit:]:
        extra = f" {node.speedup:.3f}x" if isinstance(node.speedup, float) else ""
        hint = f" Hint: {node.constraint_hint}." if node.constraint_hint else ""
        lines.append(f"- round {node.round_id}: {node.title} -> {node.outcome_kind}{extra}.{hint}")
    body = "\n".join(lines) or "- (none yet)"
    return body + "\nDo not repeat a direction marked falsified. A direction marked untested_impl_fail was not measured; it may be retried with the hint."
