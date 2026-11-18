"""Observe final TASK.md bytes without changing the prompt renderer.

For local tasks this module replays only the renderer's dynamic-block budget
selection.  The replay must equal the final TASK byte-for-byte before any block
receipt is emitted.  It never searches for renderer-like markers inside
untrusted NOTES or peer tips.  Remote receipts map those verified spans through
the deterministic transport edit script.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import shlex
import copy
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "taic.task_delivery.1"
RECEIPT_NAME = "task_delivery.json"


@dataclass(frozen=True)
class _Block:
    name: str
    layer: str
    source: str
    expected: str
    source_capped: bool = False


def _block(name: str, layer: str, source: str, expected: str, *, cap_limit: int | None = None, source_capped: bool = False) -> _Block:
    capped = len((source or "").strip().encode("utf-8")) > cap_limit if cap_limit is not None else source_capped
    return _Block(name, layer, source or "", expected or "", bool(capped))


def _blocks(brief) -> list[_Block]:
    # Local imports avoid a module cycle; prepare_session calls us after rendering.
    from . import kernel_agent as ka
    from .situation_block import situation_for_brief

    cap = ka._cap_task
    blocks: list[_Block] = []
    situation = situation_for_brief(brief)
    if situation.text:
        source = "\n\n".join(part for part in (
            brief.harness_overlay,
            brief.ledger_text if brief.use_task_ledger else "",
            brief.peer_tips,
            brief.inherit_prompt,
        ) if part)
        blocks.append(_block("本轮态势", "situation", source, situation.text))
    if not brief.seedless:
        submit = ka.submission_name(brief.task_name, brief.dtype, brief.size, brief.label, brief.file_ext)
        context = ["", "# 当前现场", f"种子在 `{brief.seed_name}`。交卷写 `{submit}`，不要改种子文件。"]
        if brief.scorecard:
            context += ["", "外层成绩单：", cap(brief.scorecard, 1000)]
        if brief.bound_line:
            context += ["", cap(brief.bound_line, 300)]
        blocks.append(_block(
            "当前现场", "other", "\n".join(x for x in (brief.scorecard, brief.bound_line) if x),
            "\n".join(context) + "\n",
            source_capped=(
                len(brief.scorecard.strip().encode("utf-8")) > 1000
                or len(brief.bound_line.strip().encode("utf-8")) > 300
            ),
        ))
        if brief.failure:
            blocks.append(_block(
                "上一轮外层失败反馈", "other", brief.failure,
                "\n\n上一轮外层失败反馈：\n" + cap(brief.failure, 1000), cap_limit=1000,
            ))
        if brief.notes_summary:
            blocks.append(_block(
                "上一轮 NOTES", "other", brief.notes_summary,
                "\n\n上一轮 NOTES（硬件上限类断言未附实测数字的，按未验证处理）：\n" + cap(brief.notes_summary, 1800), cap_limit=1800,
            ))
        if brief.plateau:
            plateau_source = (
                "这一场不在当前结构上微调。从没被计时否定的方向里另选算法族或启动几何。"
                "先在 NOTES 写下假设是什么、依赖哪几条记录、什么观察会否定它。"
            )
            blocks.append(_block("平台期", "other", plateau_source, "\n\n# 平台期\n" + cap(plateau_source, 900), cap_limit=900))
        if not brief.use_task_ledger and brief.history:
            blocks.append(_block("最近的历史", "other", brief.history, "\n\n最近的历史：\n" + cap(brief.history, 400), cap_limit=400))
        if brief.evidence:
            blocks.append(_block("现场证据", "other", brief.evidence, "\n\n" + cap(brief.evidence, 600), cap_limit=600))
        selftest_source = (
            "合同里的外层成绩和你自己跑出来的数字对不上时，先把评测命令里的交卷文件换成 "
            + shlex.quote(str(ka._work_path(brief) / brief.seed_name))
            + " 跑一次 `--timed`，再改。交卷前的自测用上面的命令，文件名是交卷文件。"
        )
        blocks.append(_block("自测对照", "other", selftest_source, "\n\n" + cap(selftest_source, 400), cap_limit=400))
    return blocks


def _sweep_suffix(brief) -> str:
    """The static optional sweep section between the contract and dynamic blocks.

    This is a deliberate second copy.  ``current_prompt_sha1`` hashes the source of the
    registered ``kernel_agent.render_contract``, so moving the prose into a helper shared
    by both modules would let the task book change without moving the prompt fingerprint
    and an old grant would silently stay valid (``taic/trust.py`` is the authority here and
    is not ours to extend).  Keeping an independent expression also makes the receipt a
    real check instead of comparing the renderer with itself.

    Only the prose repeats: the commands come from ``kernel_agent._sweep_command`` and
    ``kernel_agent._ab_command``, and ``tests/shared/test_taskbook_ab.py`` pins the bytes
    of both copies so a one-sided edit fails there rather than silently degrading every
    receipt to ``complete=false``.
    """
    from . import kernel_agent as ka

    if brief.sweep_tool and brief.selfcheck_socket:
        lines = [
            "",
            "# 一条命令比较多个变体",
            "这场的计时评测走 host bridge。这里不提供 `sweep-v2`，以免绕过远端评测和 host 权威配额。",
        ]
    elif brief.sweep_tool:
        ab_available = brief.v2 and brief.dtype in ka.AB_DTYPES
        rank_line = (
            "名次按各自对最强基线的加速比。噪声用加速比的标准误和本场的 `noise_z`。缺统计量标成未知。"
            "这一版不把多个候选放进同一个进程里交错计时"
        )
        # 与渲染器同一条规则：没有下面的 `ab` 段，就不写指向它的句子。
        rank_line += "；要在一段窗口里排先后，用下面那条 `ab`。" if ab_available else "。"
        lines = [
            "",
            "# 一条命令比较多个变体",
            "把最多 16 个完整候选文件放进一个目录。文件由你自己生成。框架不提供模板，也不替你选参数。",
            "每个文件走和 `--timed` 相同的外层评测。候选目录只放文件；自测行记在本场 `.selfcheck`，和普通 `--timed` 共用 20 次、120 秒和挂死记录。",
            "任何一个 timeout 或 crash 会立刻停下，之后的计时，包括普通 `--timed`，都会被拒绝。",
            rank_line,
            "```",
            ka._sweep_command(brief),
            "```",
        ]
        if ab_available:
            lines += [
                "",
                "# 同一段窗口里比较多个变体",
                "`sweep-v2` 每份文件各跑一次外层评测，名次里带着窗口之间的漂移。要看同一段 GPU 窗口里谁快，用 `ab`：给 2 到 16 份完整候选文件，它交错计时，第一份是对照。它读的是融合命令缓冲的 GPU 时间戳，和 `--timed` 报的 `candidate_gpu_us_median` 同一个量；每份给 median、min、cv，再给每份相对对照的逐轮配对比，`--json` 打 `tcenv.ab.1`。",
                "**只读**：不写 `.selfcheck`、不占 `--timed` 的 20 次/120 秒配额、文档里 `promotable` 是 `false`。这是你自己的工地测量，不是发布证据，也不能代替交卷前那次 `--timed`。",
                "正确性问本题自己的 quick 评测：算错的文件仍会计时（消融要看那个时间），但标成不可排序、不进名次；不安全的文件不进计时循环；第一份必须算对，否则整次拒绝。",
                "`--control` 不是 `ab` 的参数，它加在计时那条 `eval-v2` 上：`eval-v2 ... --timed --control`。同一段窗口里把同一份字节再启动一次，报告里 `benchmark.control.resolvable_relative` 就是这段窗口能分辨的最小相对差别，同一份字节本该读 1.000。它默认关，计时成本翻倍。这版快了 1% 而它大于 1%，就是这段窗口分不清这个差。",
                "```",
                ka._ab_command(brief),
                "```",
            ]
    else:
        return ""
    return "\n".join(lines) + "\n"


def _static_prefix(task: str, brief) -> str | None:
    from .kernel_agent import submission_name

    submit = submission_name(brief.task_name, brief.dtype, brief.size, brief.label, brief.file_ext)
    tail = f"交卷文件 `{submit}` 里一直放目前最好的一版。外层只读这个文件。\n"
    found = task.find(tail)
    if found < 0:
        return None
    end = found + len(tail)
    suffix = _sweep_suffix(brief)
    if not task.startswith(suffix, end):
        return None
    return task[: end + len(suffix)]


def _public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _row(spec: _Block, status: str, delivered: str, *, reason: str = "", start: int = 0, end: int = 0) -> dict:
    expected_raw = spec.expected.encode("utf-8")
    delivered_raw = delivered.encode("utf-8")
    complete = status == "delivered"
    return {
        "name": spec.name,
        "layer": spec.layer,
        "status": status,
        "source_capped": spec.source_capped,
        "source_bytes": len(spec.source.encode("utf-8")),
        "expected_bytes": len(expected_raw),
        "delivered_bytes": len(delivered_raw),
        "omitted_bytes": 0 if complete else max(0, len(expected_raw) - len(delivered_raw)),
        "sha256": hashlib.sha256(delivered_raw).hexdigest() if delivered_raw else "",
        "omission_reason": reason,
        "_start": start,
        "_end": end,
        "_complete": complete,
    }


def _simulate(brief, static: str, max_bytes: int) -> tuple[str, list[dict], int, int, int]:
    """Replay kernel_agent.render_contract's dynamic selection, not its static text."""
    from . import kernel_agent as ka

    specs = _blocks(brief)
    out = static
    base_bytes = len(static.encode("utf-8"))
    situation_text = next((spec.expected for spec in specs if spec.name == "本轮态势"), "")
    overlay_present = bool(brief.harness_overlay)
    effective_max = (
        max(max_bytes, base_bytes + 6000)
        if brief.v2 and max_bytes == ka.TASK_MAX_BYTES and base_bytes + 2400 > max_bytes
        else max_bytes
    )
    limit = max(0, effective_max - (200 if situation_text else 420))
    reserve_after = [0] * len(specs)
    has_situation = any(spec.name == "本轮态势" for spec in specs)
    for index in range(len(specs) - 2, -1, -1):
        following = specs[index + 1]
        keep = following.name == "平台期" and brief.plateau
        reserve = len(following.expected.encode("utf-8")) if keep else 0
        if has_situation and following.name == "当前现场":
            reserve = min(420, len(following.expected.encode("utf-8")))
        reserve_after[index] = reserve_after[index + 1] + reserve
    rows: list[dict] = []
    omitted: list[str] = []
    for index, spec in enumerate(specs):
        text = spec.expected
        room = limit - len(out.encode("utf-8")) - reserve_after[index]
        if spec.name == "本轮态势":
            from .situation_block import DYNAMIC_BUDGET_BYTES, situation_for_brief

            fitted = situation_for_brief(
                brief, budget_bytes=min(DYNAMIC_BUDGET_BYTES, max(0, room)), strict_budget=False,
            )
            if fitted.text and len(fitted.text.encode("utf-8")) <= room:
                fitted_spec = _Block(spec.name, spec.layer, spec.source, fitted.text)
                start = len(out)
                out += fitted.text
                delivered_row = _row(fitted_spec, "delivered", fitted.text, start=start, end=len(out))
                delivered_row["_situation"] = fitted.receipt
                rows.append(delivered_row)
                continue
            omitted.append(spec.name)
            omitted_row = _row(spec, "omitted", "", reason="task_budget")
            omitted_row["_situation"] = fitted.receipt
            rows.append(omitted_row)
            continue
        if spec.name == "当前现场":
            if len(text.encode("utf-8")) <= room:
                start = len(out)
                out += text
                rows.append(_row(spec, "delivered", text, start=start, end=len(out)))
                continue
            cap = max(0, room - 120)
            if cap >= 80:
                trimmed = ka._cap_task(text, cap)
                if trimmed:
                    start = len(out)
                    out += trimmed
                    status = "delivered" if trimmed == text.strip() else "truncated"
                    rows.append(_row(spec, status, trimmed, reason="" if status == "delivered" else "task_budget_truncated", start=start, end=len(out)))
                    continue
            omitted.append(spec.name)
            rows.append(_row(spec, "omitted", "", reason="task_budget"))
            continue
        if room <= 0:
            omitted.append(spec.name)
            rows.append(_row(spec, "omitted", "", reason="task_budget"))
            continue
        if len(text.encode("utf-8")) <= room:
            start = len(out)
            out += text
            rows.append(_row(spec, "delivered", text, start=start, end=len(out)))
            continue
        if spec.name == "上一轮 NOTES" and room >= 80:
            trimmed = ka._cap_within(text, room)
            start = len(out)
            out += trimmed
            status = "delivered" if trimmed == text else "truncated"
            rows.append(_row(spec, status, trimmed, reason="" if status == "delivered" else "task_budget_truncated", start=start, end=len(out)))
            continue
        omitted.append(spec.name)
        rows.append(_row(spec, "omitted", "", reason="task_budget"))
    note_start = -1
    if omitted:
        full_note = (
            "\n\n# 本轮省略（上下文预算）\n"
            f"以下段落因 {effective_max} 字节预算未附：" + "、".join(omitted) + "。\n"
            "全文在本目录的 NOTES.md、FEEDBACK.md、feedback.jsonl 和 state.json；需要时读取，不要假设它们为空。\n"
        )
        compact_note = (
            "\n\n# 本轮省略（上下文预算）\n"
            "全文在本目录的 NOTES.md、FEEDBACK.md、feedback.jsonl 和 state.json；需要时读取，不要假设它们为空。\n"
        )
        note = compact_note if overlay_present else full_note
        room = effective_max - len(out.encode("utf-8"))
        if len(note.encode("utf-8")) > room:
            note = compact_note
        if room > 0:
            note_start = len(out)
            out += ka._clip_bytes(note, room)
    return out, rows, note_start, effective_max, base_bytes


def _unverified(task: str, renderer: str, reason: str) -> dict:
    raw = task.encode("utf-8")
    return {
        "schema": SCHEMA,
        "event": "task_delivery",
        "renderer": renderer,
        "complete": False,
        "accounting_error": reason,
        "task_sha256": hashlib.sha256(raw).hexdigest(),
        "task_bytes": len(raw),
        "static": {"delivered_bytes": 0, "omitted_bytes": 0, "sha256": "", "complete": False},
        "dynamic": {"delivered_bytes": 0, "omitted_bytes": 0, "budget_bytes": None, "over_budget": False},
        "blocks": [],
        "situation": None,
        "omitted": [],
        "truncated": [],
        "omission_note_bytes": 0,
        "observation_only": True,
    }


def _apply_situation_delivery(info: dict, row: dict | None) -> dict:
    result = copy.deepcopy(info)
    if row is None or result.get("status") == "empty":
        return result
    status = row["status"]
    overflow = bool(result.get("over_budget")) or int(result.get("candidate_bytes") or 0) > int(result.get("budget_bytes") or 0)
    final_status = (
        "truncated" if overflow and status == "delivered"
        else "stopped" if overflow and status == "omitted"
        else "truncated" if overflow
        else status
    )
    result["delivery"] = final_status
    result["status"] = final_status
    result["over_budget"] = overflow
    result["delivered_bytes"] = row["delivered_bytes"]
    result["omitted_bytes"] = max(0, int(result.get("candidate_bytes") or 0) - row["delivered_bytes"])
    result["text_sha256"] = row["sha256"]
    if status == "delivered":
        result["complete"] = True
        for layer in result["layers"]:
            if layer["rendered_bytes"]:
                layer["delivered_bytes"] = layer.get("output_bytes", layer["rendered_bytes"])
                layer["omitted_bytes"] = layer["rendered_bytes"] - layer["delivered_bytes"]
                layer["status"] = "truncated" if layer["omitted_bytes"] else "delivered"
    elif status == "omitted":
        result["complete"] = True
        for layer in result["layers"]:
            if layer["rendered_bytes"]:
                layer["status"] = "omitted"
                layer["delivered_bytes"] = 0
                layer["omitted_bytes"] = layer["rendered_bytes"]
    else:
        # Transport edits can cut across generated layer boundaries. Keep the
        # exact whole-block bytes, but fail closed on per-layer attribution.
        result["complete"] = False
        result["accounting_error"] = "per-layer remote delivery cannot be proven"
        for layer in result["layers"]:
            if layer["rendered_bytes"]:
                layer["status"] = "unknown"
                layer["delivered_bytes"] = None
                layer["omitted_bytes"] = None
    return result


def _situation_delivery(brief, rows: list[dict]) -> dict:
    from .situation_block import situation_for_brief

    row = next((row for row in rows if row["name"] == "本轮态势"), None)
    info = (
        copy.deepcopy(row["_situation"])
        if row is not None and isinstance(row.get("_situation"), dict)
        else situation_for_brief(brief).receipt
    )
    return _apply_situation_delivery(info, row)


def _local_receipt(brief, task: str, renderer: str, max_bytes: int) -> tuple[dict, list[dict], int]:
    static = _static_prefix(task, brief)
    if static is None:
        return _unverified(task, renderer, "the generated static TASK boundary is missing"), [], -1
    predicted, rows, note_start, effective_max, base_bytes = _simulate(brief, static, max_bytes)
    if predicted != task:
        return _unverified(task, renderer, "dynamic budget replay does not equal the final TASK"), [], -1
    dynamic_delivered = sum(row["delivered_bytes"] for row in rows)
    dynamic_omitted = sum(row["omitted_bytes"] for row in rows)
    note = task[note_start:] if note_start >= 0 else ""
    budget = max(0, effective_max - base_bytes)
    receipt = {
        "schema": SCHEMA,
        "event": "task_delivery",
        "renderer": renderer,
        "complete": True,
        "accounting_error": "",
        "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
        "task_bytes": len(task.encode("utf-8")),
        "static": {
            "delivered_bytes": len(static.encode("utf-8")),
            "omitted_bytes": 0,
            "sweep_bytes": len(_sweep_suffix(brief).encode("utf-8")),
            "sha256": hashlib.sha256(static.encode("utf-8")).hexdigest(),
            "complete": True,
        },
        "dynamic": {
            "delivered_bytes": dynamic_delivered,
            "omitted_bytes": dynamic_omitted,
            "budget_bytes": budget,
            "over_budget": dynamic_delivered + len(note.encode("utf-8")) > budget,
        },
        "blocks": [_public_row(row) for row in rows],
        "situation": _situation_delivery(brief, rows),
        "omitted": [row["name"] for row in rows if row["status"] == "omitted"],
        "truncated": [row["name"] for row in rows if row["status"] == "truncated"],
        "omission_note_bytes": len(note.encode("utf-8")),
        "observation_only": True,
    }
    return receipt, rows, note_start


def _position_mapper(opcodes, remote_length: int):
    def map_position(position: int, *, right: bool) -> int:
        for tag, i1, i2, j1, j2 in opcodes:
            if position < i1:
                return max(0, j1 - (i1 - position))
            if i1 <= position <= i2:
                if tag == "equal":
                    return j1 + position - i1
                if tag == "insert":
                    return j2 if right else j1
                if position == i1:
                    return j1
                if position == i2:
                    return j2
                return j2 if right else j1
        return remote_length
    return map_position


def _remote_receipt(brief, task: str, renderer: str) -> dict:
    from . import kernel_agent as ka
    from .workspace_prompt import render_remote_contract

    local_task = ka.render_contract(brief)
    local, rows, note_start = _local_receipt(brief, local_task, "local", ka.TASK_MAX_BYTES)
    if not local["complete"] or render_remote_contract(brief) != task:
        return _unverified(task, renderer, "remote transport verification failed")
    matcher = difflib.SequenceMatcher(a=local_task, b=task, autojunk=False)
    opcodes = matcher.get_opcodes()
    map_position = _position_mapper(opcodes, len(task))
    mapped_rows = []
    for row in rows:
        if row["status"] == "omitted":
            mapped_rows.append(_public_row(row))
            continue
        start = map_position(row["_start"], right=False)
        end = map_position(row["_end"], right=True)
        start = max(0, min(start, len(task)))
        end = max(start, min(end, len(task)))
        delivered = task[start:end]
        spec = _Block(row["name"], row["layer"], "", "")
        if not delivered and row["delivered_bytes"]:
            mapped = _row(spec, "omitted", "", reason="remote_profile")
            mapped.update({
                "layer": row["layer"], "source_capped": row["source_capped"],
                "source_bytes": row["source_bytes"], "expected_bytes": row["expected_bytes"],
                "omitted_bytes": row["delivered_bytes"],
            })
        else:
            status = row["status"]
            # Any transport edit inside a delivered block makes per-layer byte
            # attribution unprovable. Whole-block bytes remain exact; callers
            # must not reuse local layer lengths after a replacement or deletion.
            edited = False
            for tag, i1, i2, _j1, _j2 in opcodes:
                if tag == "equal":
                    continue
                if tag == "insert":
                    touched = row["_start"] < i1 < row["_end"]
                else:
                    touched = max(0, min(i2, row["_end"]) - max(i1, row["_start"])) > 0
                edited = edited or touched
            if edited:
                status = "partial"
            mapped = _row(spec, status, delivered, reason="remote_profile_partial" if status == "partial" else row["omission_reason"])
            mapped.update({
                "layer": row["layer"], "source_capped": row["source_capped"],
                "source_bytes": row["source_bytes"], "expected_bytes": row["expected_bytes"],
                "omitted_bytes": max(0, row["delivered_bytes"] - len(delivered.encode("utf-8"))),
            })
        mapped_rows.append(_public_row(mapped))
    # Recompute the static boundary from mapped spans without searching text.
    delivered_positions = []
    for row in rows:
        if row["status"] != "omitted":
            position = map_position(row["_start"], right=False)
            mapped_end = map_position(row["_end"], right=True)
            if mapped_end > position:
                delivered_positions.append(position)
    static_end = min(delivered_positions) if delivered_positions else (map_position(note_start, right=False) if note_start >= 0 else len(task))
    static = task[:static_end]
    local_static = _static_prefix(local_task, brief) or ""
    sweep_suffix = _sweep_suffix(brief)
    if sweep_suffix and local_static.endswith(sweep_suffix):
        local_sweep_start = len(local_static) - len(sweep_suffix)
        remote_sweep_start = map_position(local_sweep_start, right=False)
        remote_sweep_end = map_position(len(local_static), right=True)
        sweep_bytes = len(task[remote_sweep_start:remote_sweep_end].encode("utf-8"))
    else:
        sweep_bytes = 0
    remote_note = map_position(note_start, right=False) if note_start >= 0 else -1
    note = task[remote_note:] if remote_note >= 0 else ""
    dynamic_delivered = sum(row["delivered_bytes"] for row in mapped_rows)
    dynamic_omitted = sum(row["omitted_bytes"] for row in mapped_rows)
    budget = local["dynamic"]["budget_bytes"]
    return {
        **{key: value for key, value in local.items() if key not in {"renderer", "task_sha256", "task_bytes", "static", "dynamic", "blocks", "omitted", "truncated", "omission_note_bytes"}},
        "renderer": renderer,
        "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
        "task_bytes": len(task.encode("utf-8")),
        "static": {
            "delivered_bytes": len(static.encode("utf-8")), "omitted_bytes": 0,
            "sweep_bytes": sweep_bytes,
            "sha256": hashlib.sha256(static.encode("utf-8")).hexdigest(), "complete": True,
        },
        "dynamic": {
            "delivered_bytes": dynamic_delivered, "omitted_bytes": dynamic_omitted,
            "budget_bytes": budget, "over_budget": dynamic_delivered + len(note.encode("utf-8")) > budget,
        },
        "blocks": mapped_rows,
        "situation": _apply_situation_delivery(
            local["situation"], next((row for row in mapped_rows if row["name"] == "本轮态势"), None),
        ),
        "omitted": [row["name"] for row in mapped_rows if row["status"] == "omitted"],
        "truncated": [row["name"] for row in mapped_rows if row["status"] in {"truncated", "partial"}],
        "omission_note_bytes": len(note.encode("utf-8")),
    }


def build_receipt(brief, task_text: str, *, renderer: str = "local", max_bytes: int | None = None) -> dict:
    """Return verified final-TASK spans and UTF-8 byte counts.

    If replay cannot prove the spans, ``complete=false`` and no block verdict is
    emitted.  ``source_capped`` distinguishes an existing per-block source cap
    from an additional whole-TASK budget truncation.
    """
    from . import kernel_agent as ka

    task = task_text or ""
    limit = ka.TASK_MAX_BYTES if max_bytes is None else int(max_bytes)
    if renderer == "remote":
        return _remote_receipt(brief, task, renderer)
    if renderer == "override":
        if task == ka.render_contract(brief):
            return _local_receipt(brief, task, "local", ka.TASK_MAX_BYTES)[0]
        from .workspace_prompt import render_remote_contract
        if task == render_remote_contract(brief):
            return _remote_receipt(brief, task, "remote")
        return _unverified(task, renderer, "override TASK is not a known local or remote rendering")
    return _local_receipt(brief, task, renderer, limit)[0]


def host_receipt_path(work: Path) -> Path | None:
    """Return the host-owned session receipt path for a canonical agent workdir."""
    work = Path(work)
    if work.parent.name != "agent_work":
        return None
    run_dir = work.parent.parent
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", work.name)
    return run_dir / "sessions" / label / RECEIPT_NAME


def record_receipt(brief, work: Path, task_text: str, *, renderer: str, task_sha256: str) -> tuple[dict, Path | None]:
    """Build a receipt and atomically write it outside the model-writable workdir."""
    from . import control

    receipt = build_receipt(brief, task_text, renderer=renderer)
    if receipt["task_sha256"] != str(task_sha256):
        raise ValueError("the delivery receipt TASK digest differs from prompt_surface_digest")
    path = host_receipt_path(work)
    if path is not None:
        control.write_json_atomic(path, receipt)
    return receipt, path


def record_situation_failure(work: Path, situation: dict, *, renderer: str) -> tuple[dict, Path | None]:
    """Persist an atomic over-budget stop; no final TASK digest is invented."""
    from . import control

    candidate = int(situation.get("candidate_bytes") or 0)
    receipt = {
        "schema": SCHEMA,
        "event": "task_delivery",
        "renderer": renderer,
        "complete": False,
        "accounting_error": "dynamic_situation_over_budget",
        "task_sha256": "",
        "task_bytes": 0,
        "static": {"delivered_bytes": 0, "omitted_bytes": 0, "sha256": "", "complete": False},
        "dynamic": {
            "delivered_bytes": 0, "omitted_bytes": candidate,
            "budget_bytes": situation.get("budget_bytes"), "over_budget": True,
        },
        "blocks": [{
            "name": "本轮态势", "layer": "situation", "status": "omitted",
            "source_capped": False, "source_bytes": situation.get("source_bytes") or 0,
            "expected_bytes": candidate, "delivered_bytes": 0, "omitted_bytes": candidate,
            "sha256": "", "omission_reason": "dynamic_situation_over_budget",
        }],
        "situation": copy.deepcopy(situation),
        "omitted": ["本轮态势"], "truncated": [], "omission_note_bytes": 0,
        "observation_only": True,
    }
    path = host_receipt_path(work)
    if path is not None:
        control.write_json_atomic(path, receipt)
    return receipt, path


def read_receipts(run_dir: Path) -> list[dict]:
    """Read host receipts for ``harness status``; ignore partial/non-object JSON."""
    from . import control

    rows = []
    for path in sorted(Path(run_dir).glob("sessions/*/task_delivery.json")):
        doc = control.read_json(path)
        if isinstance(doc, dict) and doc.get("schema") == SCHEMA:
            row = dict(doc)
            row["session_label"] = path.parent.name
            row["receipt_path"] = str(path)
            rows.append(row)
    return rows
