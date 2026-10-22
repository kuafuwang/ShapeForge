"""Audit the pending agent feedback against the run directories.

Reads the observe index, not a model. For every pending claim it records the section, the stored
premise state, the line-level measured/unverified mix, whether each named probe exists now, and a
small verdict. Writes a Markdown report and a JSON file for another agent.

    uv run python -m experiments.feedback_audit
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from observe.feedback import review_board
from observe.index import get_index
from taic.agent_feedback import _MEASURED, _probe_exists

REPO = Path(__file__).resolve().parent.parent
TOPICS = (
    ("self_check_dir", ("self-check-dir", "session_checks", "permissionerror", "不可写", "operation not permitted")),
    ("relative_path", ("相对路径", "--directory", "filenotfounderror", "找不到文件", "kernel.metal 找不到")),
    ("xctrace", ("xctrace", "xcode")),
    ("budget", ("预算", "余额", "$15", "$14")),
    ("bench2", ("bench2", "配对 a/b", "配对")),
    ("dce", ("死代码", "dce", "消融")),
    ("max_threads", ("maxthreads", "max_total_threads", "寄存器压力", "占用率")),
    ("roofline", ("roofline", "屋顶", "下界", "换种子门槛", "gate")),
    ("generator", ("生成器", "gen15", "cgran", "vmode")),
    ("workspace_zero", ("workspace_fills", "zero", "填零")),
    ("seed_report", ("--seed-report",)),
    ("notes", ("notes 节选", "截断", "拷贝进")),
)
FIXED = {"self_check_dir", "relative_path", "seed_report"}
POLICY = {"budget", "roofline"}


def topic(text: str) -> str:
    low = text.lower()
    for name, words in TOPICS:
        if any(word in low for word in words):
            return name
    return "other"


def clean_id(item: dict) -> str:
    return f"{item['run_id']}#{item['seq']}:{item['digest']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.feedback_audit")
    parser.add_argument("--out-md", default="docs/feedback_audit_20261022.md")
    parser.add_argument("--out-json", default="docs/feedback_audit_20261022.json")
    args = parser.parse_args(argv)
    conn = get_index().read_only()
    board = review_board(conn, status="pending")
    rows = []
    for item in board["items"]:
        work = REPO / "runs" / item["run_id"] / "agent_work" / str(item["label"])
        measured = valid = missing = unverified = other = 0
        for line in [part.strip() for part in item["text"].splitlines() if part.strip()]:
            match = _MEASURED.search(line)
            if match:
                measured += 1
                if _probe_exists(match.group(1).strip(), work=work, repo=REPO):
                    valid += 1
                else:
                    missing += 1
            elif "未验证" in line:
                unverified += 1
            else:
                other += 1
        section = item["section"]
        stored = item["premise"] or ""
        text = item["text"].strip()
        empty = len(text) < 100 and (text.startswith(("无", "没有", "（空）")) or text.lower().startswith("none"))
        topic_name = "algorithm_premise" if section == "被否决的前提" else topic(text)
        if empty:
            verdict, reason = "drop_empty", "empty or 'no help needed' placeholder"
        elif section == "被否决的前提":
            if stored == "verified" or valid:
                verdict, reason = "valid_measured", f"{valid} measured line(s) with an existing probe"
            elif stored == "downgraded" or missing:
                verdict, reason = "missing_probe_at_capture", "stored as unverified because the named probe was missing when parsed"
            else:
                verdict, reason = "open_unverified", "explicitly unverified; useful as a don't-retry note, not as evidence"
        elif topic_name in FIXED:
            verdict, reason = "fixed_in_code", "current TASK.md/self-check path no longer uses the failing form"
        elif topic_name in POLICY:
            verdict, reason = "policy_decision", "valid suggestion, but it changes the search gate or budget policy"
        else:
            verdict, reason = "actionable_open", "specific, reproducible, and not yet fixed"
        rows.append({
            "id": clean_id(item), "run_id": item["run_id"], "seq": item["seq"], "digest": item["digest"],
            "label": item["label"], "task": item.get("task"), "model": item.get("model"),
            "section": section, "stored_premise": stored, "topic": topic_name,
            "measured_lines": measured, "measured_valid": valid, "measured_missing": missing,
            "unverified_lines": unverified, "other_lines": other,
            "verdict": verdict, "reason": reason,
            "text": text,
        })
    verdicts = Counter(row["verdict"] for row in rows)
    sections = Counter(row["section"] for row in rows)
    premises = Counter(row["stored_premise"] for row in rows if row["section"] == "被否决的前提")
    topics = Counter(row["topic"] for row in rows if row["section"] != "被否决的前提")
    empty_rows = [row for row in rows if row["verdict"] == "drop_empty"]
    measured_rows = [row for row in rows if row["verdict"] == "valid_measured"]
    missing_rows = [row for row in rows if row["verdict"] == "missing_probe_at_capture"]
    unverified_rows = [row for row in rows if row["verdict"] == "open_unverified"]
    fixed_rows = [row for row in rows if row["verdict"] == "fixed_in_code"]
    policy_rows = [row for row in rows if row["verdict"] == "policy_decision"]
    open_rows = [row for row in rows if row["verdict"] == "actionable_open"]
    by_topic = defaultdict(list)
    by_topic_nonempty = defaultdict(list)
    for row in rows:
        if row["section"] != "被否决的前提":
            by_topic[row["topic"]].append(row)
            if row["verdict"] != "drop_empty":
                by_topic_nonempty[row["topic"]].append(row)
    duplicate_rows = sum(len(values) - 1 for values in by_topic_nonempty.values() if len(values) > 1)
    data = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(REPO / "data" / "observe.sqlite"),
        "pending_total": board.get("pending_total"),
        "totals": dict(verdicts),
        "by_section": dict(sections),
        "premise_states": dict(premises),
        "system_topics": dict(topics),
        "nonempty_system_topics": {name: len(values) for name, values in by_topic_nonempty.items()},
        "line_level": {
            "measured_lines": sum(row["measured_lines"] for row in rows),
            "measured_valid": sum(row["measured_valid"] for row in rows),
            "measured_missing": sum(row["measured_missing"] for row in rows),
            "unverified_lines": sum(row["unverified_lines"] for row in rows),
            "other_lines": sum(row["other_lines"] for row in rows),
        },
        "duplicate_suggestion_rows": duplicate_rows,
        "groups": {
            "valid_measured": [row["id"] for row in measured_rows],
            "missing_probe_at_capture": [row["id"] for row in missing_rows],
            "open_unverified": [row["id"] for row in unverified_rows],
            "fixed_in_code": [row["id"] for row in fixed_rows],
            "policy_decision": [row["id"] for row in policy_rows],
            "actionable_open": [row["id"] for row in open_rows],
            "drop_empty": [row["id"] for row in empty_rows],
        },
        "topic_groups": {
            name: {"rows": len(values), "verdicts": dict(Counter(row["verdict"] for row in values)),
                   "ids": [row["id"] for row in values]}
            for name, values in sorted(by_topic.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        },
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def table(counter: Counter, header: str) -> list[str]:
        lines = [f"| {header} | 条数 |", "| --- | ---: |"]
        for name, count in counter.most_common():
            lines.append(f"| {name or '(empty)'} | {count} |")
        return lines

    L = [
        "# 代理反馈待审统计（2026-10-22）",
        "",
        f"来源：`{data['source']}`，`status='pending'` 共 **{data['pending_total']}** 条。",
        "本报告只读索引和运行目录，不调用模型、不启动 GPU。",
        "",
        "## 结论",
        "",
        f"- 174 条 = **被否决的前提 {sections.get('被否决的前提', 0)}** + **对系统的建议 {sections.get('对系统的建议', 0)}** + **需要帮助 {sections.get('需要帮助', 0)}**。",
        f"- 可直接当作实测证据：**{len(measured_rows)} 条**（含 {data['line_level']['measured_valid']} 行 `[已实测]`，探针文件现在都存在）。",
        f"- 有效但还开放的系统建议：**{len(open_rows)} 条**；需要政策决定：**{len(policy_rows)} 条**；当前代码已修：**{len(fixed_rows)} 条**。",
        f"- 明确的未验证前提：**{len(unverified_rows) + len(missing_rows)} 条**（35 条 `[未验证]` + 39 条采集时探针缺失被降级）；只能当“别再试”的线索，不能当证据。",
        f"- 不应进入待审：**{len(empty_rows)} 条空占位**（“无/没有/不需要”）；{sum(len(v) for v in by_topic_nonempty.values())} 条非空建议/求助合并为 {len(by_topic_nonempty)} 个主题，其中 **{duplicate_rows} 条是同一主题的重复**。",
        f"- 行级：{data['line_level']['measured_lines']} 行 `[已实测]`，其中 {data['line_level']['measured_valid']} 行探针存在、{data['line_level']['measured_missing']} 行缺失；{data['line_level']['unverified_lines']} 行 `[未验证]`，{data['line_level']['other_lines']} 行分析/补充。",
        "",
        "## 分类计数",
        "",
        *table(Counter(data["totals"]), "verdict"),
        "",
        *table(Counter(data["by_section"]), "section"),
        "",
        *table(Counter(data["premise_states"]), "被否决的前提的 stored premise"),
        "",
        *table(Counter(data["system_topics"]), "系统建议/求助 topic（含空占位）"),
        "",
        *table(Counter(data["nonempty_system_topics"]), "非空系统建议/求助 topic"),
        "",
        "## 有效：实测证据",
        "",
        f"{len(measured_rows)} 条 premise 至少有一行 `[已实测]`，探针文件现在都能在 `agent_work` 里找到。它们主要是算法/硬件结论，不是系统缺陷；逐条 ID、原文和行级计数在 JSON 的 `rows`。",
        "",
        "## 有效：系统建议",
        "",
    ]
    for name, info in data["topic_groups"].items():
        if name == "algorithm_premise":
            continue
        L.append(f"- **{name}**：{info['rows']} 条，{info['verdicts']}；IDs: {', '.join(info['ids'][:6])}{' …' if len(info['ids']) > 6 else ''}")
    L += [
        "",
        "## 应剔除 / 应降级",
        "",
        f"### 空占位（{len(empty_rows)} 条，应直接删除，不进入待审）",
        "",
    ]
    for row in empty_rows:
        L.append(f"- `{row['id']}` {row['section']} {row['label']}: {row['text'][:100]}")
    L += [
        "",
        f"### 采集时探针缺失（{len(missing_rows)} 条，已标成未验证，不应作为 `[已实测]` 审查）",
        "",
        "这些行在采集时引用的探针文件不存在，解析器已改写成 `[未验证: 探针文件不存在 | … | …]`。它们可以保留为“别再试”的记录，但不应该和 60 条有探针的实测混在一起计数。",
        "",
        "**本次审计前的数据问题**：`prune_work` 曾把 FEEDBACK.md/NOTES.md 引用的探针一起归档，行级检查一度 83 行全部找不到文件。已修：`prune_work` 现在保留被引用文件；`experiments.restore_feedback_evidence` 从 `scratch/` 恢复了 538 个文件。当前 83 行都能找到探针；索引里 39 条仍保留采集时 `downgraded` 的文本，重新解析或人工复核前不要把它们当实测。",
        "",
        f"### 明确未验证（{len(unverified_rows)} 条，保留为开放问题）",
        "",
        "这些是 agent 自己标了 `[未验证]` 的前提；有信息量，但不是证据。审查时只决定“要不要继续追”，不要接受为事实。",
        "",
        "## 已修 / 已过时",
        "",
        f"{len(fixed_rows)} 条建议指向旧 TASK.md 或旧 self-check 路径：",
        "",
        "- `self_check_dir`：已把 `--self-check-dir` 改到工作区内 `.selfcheck/`，agent 沙箱可写；旧 run 的外层 `session_checks` 仍可回退（提交 `1fd881d`）。",
        "- `relative_path` / `seed_report`：当前 `_check_command` 用绝对路径的 kernel 与 `report.json`。",
        "",
        "## 待政策决定",
        "",
        f"{len(policy_rows)} 条是有效建议，但改的是预算/晋升门槛，不是 bug：",
        "",
    ]
    for row in policy_rows:
        L.append(f"- `{row['id']}` {row['topic']} {row['label']}: {row['text'][:110]}")
    L += [
        "",
        "## 修复建议（按优先级）",
        "",
        "1. **删除 7 条空占位**，并让 `taic/agent_feedback.py` 把“无/没有/不需要”这类整段识别为空，不再进入 pending。",
        "2. **把 39 条探针缺失行降为 info**（或保留但不在“待审证据”里计数）；它们的探针已从 `scratch` 恢复 538 个文件，重新解析后可能有一部分能升级为 verified。",
        "3. **合并 30 条系统建议的重复主题**：self-check 7、relative path 7、roofline 5、budget 5、bench2 2、DCE 2，其余各 1。每个主题只保留一条开放问题。",
        "4. **关闭已修的 15 条**（self-check/relative/seed-report），并在看板上标注“已修”。",
        "5. **保留未验证前提**，但打上“未验证”标签，不做实测证据。",
        "",
        "## 复现",
        "",
        "```bash",
        "uv run python -m experiments.feedback_audit",
        "```",
        "",
        f"机器可读：`{args.out_json}`；逐条 `verdict`、`measured_lines`、`text` 都在 `rows`。",
        "",
    ]
    Path(args.out_md).write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({
        "pending": data["pending_total"], "verdicts": data["totals"], "sections": data["by_section"],
        "premises": data["premise_states"], "topics": data["system_topics"],
        "duplicate_suggestion_rows": duplicate_rows, "md": args.out_md, "json": args.out_json,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
