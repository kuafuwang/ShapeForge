"""Structured record of what a plan actually established.

An ABI error, a compile error, or a reply with no kernel does not show that the direction is slow.
Only a kernel that is safe and correct, and not faster, falsifies the direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PlanNode:
    round_id: int
    label: str
    title: str
    outcome_kind: str
    stage: str
    speedup: float | None
    parent_speedup: float | None
    constraint_hint: str


def classify_outcome(status: str, message: str, *, improved: bool, within_noise: bool, speedup: float | None, parent_speedup: float | None) -> tuple[str, str]:
    """``(outcome_kind, constraint_hint)``. ``untested_impl_fail`` means the direction was not timed."""
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
    if status == "ok" and parent_speedup is not None and speedup is not None and speedup <= parent_speedup:
        return "falsified", ""
    if status == "ok":
        return "falsified", ""
    return "untested_impl_fail", "this attempt has no timing"


def nodes_from_trace(trace: list[dict], history: list[str]) -> list[PlanNode]:
    """Rebuild nodes from the history lines and the last report of each ``rN.pI`` branch."""
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
    """Planner text. An untested failure is not listed as a direction to avoid."""
    lines = []
    for node in nodes[-limit:]:
        extra = f" {node.speedup:.3f}x" if isinstance(node.speedup, float) else ""
        hint = f" Hint: {node.constraint_hint}." if node.constraint_hint else ""
        lines.append(f"- round {node.round_id}: {node.title} -> {node.outcome_kind}{extra}.{hint}")
    body = "\n".join(lines) or "- (none yet)"
    return body + "\nDo not repeat a direction marked falsified. A direction marked untested_impl_fail was not measured; it may be retried with the hint."
