"""Read-only yield of finished searches. This script does not start a search, call a model, or use a GPU.

- lift: median over runs of ``best_speedup / first_ok_init_speedup - 1``. An init label matches
  ``^init(\\d+)?(\\.s\\d+)?$``. A run with no positive init speedup is left out of this median only.
- unchanged fraction: one pooled ratio, unchanged agent sessions divided by all agent sessions
  in the included runs. With no agent events, that run contributes its trace rows the same way.
  A run with neither is left out of the pool. ``unchanged_fraction_per_run_median`` is the older
  median of per-run ratios and is not the primary number.
- spend after best: median over runs of ``(final_spend - cost_usd on the last improved row) / final_spend``.
  ``final_spend`` is ``result.json`` ``spent_usd`` when that number is present and positive.
  A paid session that wrote no trace row is still in ``spent_usd``. Older results without
  ``spent_usd`` fall back to the last trace row's ``cost_usd``. Both are running totals.
  No improved row, or a final total that is not positive, leaves the run out.
- timed self-checks: one count per agent session, then the median of those counts.
  A session uses ``agent_work/<label>/.selfcheck/self_checks.jsonl`` when that file exists,
  otherwise ``agent_work/<label>/self_checks.jsonl``, otherwise
  ``candidates/<idx>_<label>/self_checks.jsonl``, otherwise the agent event's ``self_checks``.
  One old file does not switch the whole run. ``self_checks_per_run_median`` is the median of
  per-run sums and is not the primary number.

The median of an even count is the average of the two middle values. An empty root prints nulls and ``n`` 0.
Runs are every directory that contains ``result.json``, found recursively. Symlinks are not followed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

_INIT_LABEL = re.compile(r"^init(\d+)?(\.s\d+)?$")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _load(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _run_dirs(root: Path) -> list[Path]:
    """@brief 递归找出每个含 ``result.json`` 的运行目录。

    根目录自己有结果时也算一次，并继续找更深层的结果。不跟随符号链接。

    @param root 搜索根。
    @return 按路径排序的运行目录。
    """
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        if "result.json" in filenames:
            found.append(Path(dirpath))
    return sorted(found)


def _positive(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number <= 0:
        return None
    return number


def _agent_events(run_dir: Path) -> list[dict]:
    path = run_dir / "transcript.jsonl"
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event") == "agent":
            events.append(event)
    return events


def _first_init_speedup(doc: dict) -> float | None:
    for cand in doc.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        if not _INIT_LABEL.match(str(cand.get("label") or "")):
            continue
        if cand.get("status") != "ok":
            continue
        speed = _positive(cand.get("speedup"))
        if speed is not None:
            return speed
    return None


def _lift(doc: dict) -> float | None:
    first = _first_init_speedup(doc)
    best = _positive(doc.get("best_speedup"))
    if first is None or best is None:
        return None
    return best / first - 1


def _unchanged_counts(run_dir: Path, doc: dict) -> tuple[int, int] | None:
    """@brief 一次运行里原样交回的场次和总场次。

    有 agent 事件时用事件。没有时用 trace 行。两边都没有则不进汇总。

    @param run_dir 运行目录。
    @param doc 该次 ``result.json``。
    @return ``(unchanged, sessions)``；无法计数时是 None。
    """
    events = _agent_events(run_dir)
    if events:
        return sum(1 for event in events if event.get("outcome") == "unchanged"), len(events)
    trace = [row for row in (doc.get("trace") or []) if isinstance(row, dict)]
    if not trace:
        return None
    return sum(1 for row in trace if row.get("status") == "unchanged"), len(trace)


def _spend_after_best(doc: dict) -> float | None:
    trace = [row for row in (doc.get("trace") or []) if isinstance(row, dict)]
    last_improved = None
    for row in trace:
        if not row.get("improved"):
            continue
        cost = row.get("cost_usd")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            continue
        last_improved = float(cost)
    if last_improved is None:
        return None
    # 最后几场付费失败可能不写 trace。优先用结果里的累计花费，旧记录才退回最后一行。
    final = _positive(doc.get("spent_usd"))
    if final is None and trace:
        final = _positive(trace[-1].get("cost_usd"))
    if final is None:
        return None
    return (final - last_improved) / final


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def _offer(chosen: dict[str, tuple[int, float]], label: str, priority: int, count: float) -> None:
    if not label:
        return
    current = chosen.get(label)
    if current is None or priority < current[0]:
        chosen[label] = (priority, float(count))


def _session_self_checks(run_dir: Path) -> list[float]:
    """@brief 每个 agent session 一条自测次数。同一场只留优先级最高的一份记录。

    新目录 ``.selfcheck/self_checks.jsonl`` 优先于旧的会话文件，旧文件优先于
    ``candidates/<idx>_<label>/`` 归档，归档优先于 agent 事件上的 ``self_checks``。
    一场有旧文件不会让同一次运行里的其他场改用别的来源。

    @param run_dir 运行目录。
    @return 按标签排序的每场次数。没有记录时是空列表。
    """
    chosen: dict[str, tuple[int, float]] = {}
    work = run_dir / "agent_work"
    if work.is_dir():
        for session in sorted(path for path in work.iterdir() if path.is_dir() and not path.name.startswith(".")):
            fresh = session / ".selfcheck" / "self_checks.jsonl"
            old = session / "self_checks.jsonl"
            if fresh.is_file():
                _offer(chosen, session.name, 0, _line_count(fresh))
            elif old.is_file():
                _offer(chosen, session.name, 1, _line_count(old))
    cand_root = run_dir / "candidates"
    if cand_root.is_dir():
        for slot in sorted(path for path in cand_root.iterdir() if path.is_dir() and not path.name.startswith(".")):
            path = slot / "self_checks.jsonl"
            if not path.is_file():
                nested = slot / ".selfcheck" / "self_checks.jsonl"
                path = nested if nested.is_file() else path
            if not path.is_file():
                continue
            label = slot.name.split("_", 1)[1] if "_" in slot.name else slot.name
            _offer(chosen, label, 2, _line_count(path))
    for event in _agent_events(run_dir):
        count = event.get("self_checks")
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            continue
        _offer(chosen, str(event.get("label") or ""), 3, float(count))
    return [item[1] for _, item in sorted(chosen.items())]


def summarize(root: str | Path) -> dict:
    """@brief 汇总一个目录下各次搜索的产出。

    抬升和最优之后的花费仍是各次运行的中位数。原样交回是纳入场次的汇总比例。
    自测次数是每场 session 的中位数。

    @param root 运行根。没有 ``result.json`` 时中位数是 None，``n`` 为 0。
    @return 含运行数、汇总比例、中位数和对应样本数的字典。
    """
    lifts = []
    unchanged_values = []
    spend = []
    checks = []
    per_run_checks = []
    unchanged_count = 0
    unchanged_sessions = 0
    runs = 0
    for run_dir in _run_dirs(Path(root)):
        doc = _load(run_dir / "result.json")
        if doc is None:
            continue
        runs += 1
        lift = _lift(doc)
        if lift is not None:
            lifts.append(lift)
        pair = _unchanged_counts(run_dir, doc)
        if pair is not None:
            unchanged_count += pair[0]
            unchanged_sessions += pair[1]
            unchanged_values.append(pair[0] / pair[1])
        after = _spend_after_best(doc)
        if after is not None:
            spend.append(after)
        session_counts = _session_self_checks(run_dir)
        if session_counts:
            checks.extend(session_counts)
            per_run_checks.append(float(sum(session_counts)))
    return {
        "n": runs,
        "lift_median": _median(lifts),
        "lift_n": len(lifts),
        "unchanged_fraction": (unchanged_count / unchanged_sessions) if unchanged_sessions else None,
        "unchanged_count": unchanged_count,
        "unchanged_sessions": unchanged_sessions,
        "unchanged_fraction_per_run_median": _median(unchanged_values),
        "spend_after_best_median": _median(spend),
        "spend_n": len(spend),
        "self_checks_median": _median(checks),
        "self_checks_n": len(checks),
        "self_checks_per_run_median": _median(per_run_checks),
    }


def main(argv: list[str] | None = None) -> int:
    """@brief 把 ``summarize`` 打成 JSON。不启动搜索。

    @param argv 命令行；None 时用进程参数。
    @return 恒为 0。空目录也是 0。
    """
    parser = argparse.ArgumentParser(description="Read-only search yield. Does not start a search.")
    parser.add_argument("--root", default="runs", help="directory of finished runs (default: runs)")
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
