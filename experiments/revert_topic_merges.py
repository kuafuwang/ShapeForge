"""Revert the explicit topic-level duplicate merges from the 2026-10-22 audit.

A topic is a triage bucket, not an equivalence relation. This script touches only review
entries whose status is ``obsolete`` and whose note says ``duplicate of``; empty-placeholder
obsolete entries are left alone. The original note is kept in history, and the new note says
the merge was reverted.

    uv run python -m experiments.revert_topic_merges --dry-run
    uv run python -m experiments.revert_topic_merges --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from observe.feedback import apply_review
from taic import control

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = REPO / "runs"


def _review_files(root: Path) -> list[Path]:
    """@brief 找出根目录下所有的 review 文件。

    @param root 存放运行的根目录。
    @return 排好序的 ``feedback_review.json`` 路径列表。
    """
    return sorted(Path(root).rglob("feedback_review.json"))


def _load(path: Path) -> dict:
    """@brief 读一份 review 文件，读不出来时返回空壳。

    @param path ``feedback_review.json`` 路径。
    @return 文档字典；文件缺失或不是字典时返回 ``{"v": 1, "claims": {}}``。
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"v": 1, "claims": {}}
    return doc if isinstance(doc, dict) else {"v": 1, "claims": {}}


def find(root: Path) -> list[dict]:
    """@brief 找出所有被标成``主题重复``而置为 obsolete 的条目。

    空占位等其他 obsolete 条目不列入。

    @param root 存放运行的根目录。
    @return 每项的所属运行目录、``seq``、``digest``、``topic`` 与旧备注。
    """
    out = []
    for path in _review_files(root):
        doc = _load(path)
        claims = doc.get("claims") if isinstance(doc.get("claims"), dict) else {}
        for key, claim in claims.items():
            if not isinstance(claim, dict):
                continue
            note = str(claim.get("note") or "")
            if claim.get("status") != "obsolete" or "duplicate of" not in note.lower():
                continue
            seq_text, _, digest = str(key).partition(":")
            try:
                seq = int(seq_text)
            except ValueError:
                continue
            out.append({
                "run_dir": path.parent,
                "seq": seq,
                "digest": digest,
                "topic": str(claim.get("topic") or ""),
                "note": note,
                "key": key,
            })
    return out


def revert(rows: list[dict], *, apply: bool, backup_dir: Path | None = None) -> dict:
    """@brief 把主题合并改回 pending，可选先备份 review 文件。

    @param rows ``find`` 返回的待回退条目。
    @param apply 为真时真的写文件；为假时只列计划。
    @param backup_dir 备份目录；为 ``None`` 时不备份。
    @return 统计字典：行数、写入数、错误数与逐条明细。
    """
    result = {"mode": "apply" if apply else "dry-run", "rows": len(rows), "written": 0, "errors": 0, "backups": [], "details": []}
    if apply and backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        for row in rows:
            run_dir = row["run_dir"]
            if run_dir in seen:
                continue
            seen.add(run_dir)
            source = run_dir / "feedback_review.json"
            target = backup_dir / run_dir.name / "feedback_review.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            result["backups"].append(str(target))
    for row in rows:
        detail = {"run": row["run_dir"].name, "seq": row["seq"], "digest": row["digest"], "topic": row["topic"], "old_note": row["note"]}
        if not apply:
            result["details"].append(detail)
            continue
        body = {
            "status": "pending",
            "verification": "unchecked",
            "topic": row["topic"],
            "note": "audit 2026-10-22: possible same topic; duplicate merge reverted because topic is not action-equivalent",
        }
        try:
            control.acquire_run_lock(row["run_dir"], who="revert-topic-merges")
            try:
                apply_review(row["run_dir"], row["seq"], row["digest"], body)
            finally:
                control.release_run_lock(row["run_dir"])
        except (OSError, ValueError, control.RunLockError) as exc:
            result["errors"] += 1
            detail["error"] = str(exc)
        else:
            result["written"] += 1
        result["details"].append(detail)
    return result


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m experiments.revert_topic_merges`` 的入口，默认干跑。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码；有错时返回 2。
    """
    parser = argparse.ArgumentParser(prog="experiments.revert_topic_merges")
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup-dir", default=None, help="copy each touched feedback_review.json here before applying")
    args = parser.parse_args(argv)
    rows = find(Path(args.runs))
    backup = Path(args.backup_dir) if args.backup_dir else (Path("/tmp") / f"taic-topic-revert-{int(time.time())}" if args.apply else None)
    result = revert(rows, apply=args.apply, backup_dir=backup)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
