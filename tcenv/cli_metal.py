"""Metal-specific ``tcenv`` subcommands.

Only commands that cannot mean anything for another backend live here, so a
non-Metal target never sees them:

* ``paired-compare`` nominates a two-file comparison and writes it into the
  self-check ledger.
* ``ab`` ranks 2-16 files in one GPU window and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _compare_pair(backend: str, path_a: Path, path_b: Path, **kwargs) -> dict:
    """@brief 把两份源码交给后端。Metal 会在本进程里绑定并交错计时。"""
    from tcenv.backends.metal.paired_compare import compare_for_backend

    return compare_for_backend(backend, path_a=path_a, path_b=path_b, **kwargs)


def cmd_paired_compare(a: argparse.Namespace) -> int:
    """@brief 比较两个候选文件。两个 SHA 写进工作目录的提名文件，不写进权威自测行。

    支持的后端在执行前先查配额。一旦进入执行，成功、失败、超时和崩溃都占一次
    timed 次数；超时和崩溃进入既有挂死记录。未知耗时不写成 GPU 秒数。
    任务书里的说明等 E1，这里只提供命令。
    """
    import hashlib

    from taic.paired_compare import (
        PairedCompareError,
        append_nomination,
        backend_capability,
        record_supported_attempt,
    )
    from tcenv.selfcheck import budget_reason, earlier_hang

    path_a = Path(a.left)
    path_b = Path(a.right)
    try:
        sha_a = hashlib.sha256(path_a.read_bytes()).hexdigest()
        sha_b = hashlib.sha256(path_b.read_bytes()).hexdigest()
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    directory = Path(a.self_check_dir) if a.self_check_dir else path_a.parent / ".selfcheck"
    supported = bool(backend_capability(a.backend)["supported"])
    if supported:
        blocked = earlier_hang(directory) or budget_reason(directory)
        if blocked:
            print(blocked, file=sys.stderr)
            return 2
    row = {
        "sha_a": sha_a,
        "sha_b": sha_b,
        "path_a": str(path_a),
        "path_b": str(path_b),
        "conclusion": "",
    }
    try:
        document = _compare_pair(
            a.backend, path_a, path_b, sha_a=sha_a, sha_b=sha_b, seed=a.seed,
            reps=int(getattr(a, "reps", 16)),
            task=getattr(a, "task", None), dtype=getattr(a, "dtype", None), size=getattr(a, "size", None),
            track_id=getattr(a, "track", None) or "msl-single",
        )
    except Exception as exc:
        # 已经进入支持后端之后，普通异常也是一次崩溃。不捕获 BaseException，
        # 键盘中断和 SystemExit 仍然直接离开，也不补配额。
        if supported:
            try:
                record_supported_attempt(directory, row, {"status": "crash", "reason": str(exc)})
            except Exception:
                pass
        print(str(exc), file=sys.stderr)
        return 2
    if not isinstance(document, dict):
        document = {"applied": False, "status": "crash", "reason": "paired compare returned no document"}
    row["conclusion"] = str(document.get("conclusion") or "")
    try:
        if supported:
            record_supported_attempt(directory, row, document)
        else:
            append_nomination(directory, row)
    except (PairedCompareError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0 if document.get("applied") else 2


# 子命令分发。eval 走外部核，naive 走任务自带参考核，profile 先 quick 再跟踪，tasks 只列契约。


def cmd_ab(a: argparse.Namespace) -> int:
    """@brief 在一段 GPU 窗口里交错计时 2-16 份候选，只读，不占 ``--timed`` 配额。

    只属于 Metal：它绑定 MSL 候选的 v2 runtime 并读融合命令缓冲的 GPU 时间戳，
    也就是 ``eval-v2 --timed`` 报 ``candidate_gpu_us_median`` 用的同一个物理量。
    排序因此不会像手抄探针那样和评测器打架。结果不是发布证据。
    """
    from taic.paired_compare import PairedCompareError
    from tcenv.backends.metal.ab_bench import MAX_FILES, bench_paths

    backend = getattr(a, "backend", "metal")
    if backend != "metal":
        print(f"tcenv ab is Metal-only; backend {backend!r} is not supported", file=sys.stderr)
        return 2
    files = list(getattr(a, "files", []) or [])
    if len(files) < 2:
        print("ab needs at least two candidate files", file=sys.stderr)
        return 2
    if len(files) > MAX_FILES:
        print(f"ab takes at most {MAX_FILES} candidate files, got {len(files)}", file=sys.stderr)
        return 2
    if not getattr(a, "task", None) or not getattr(a, "dtype", None) or not getattr(a, "size", None):
        print("ab needs --task, --dtype and --size", file=sys.stderr)
        return 2
    try:
        document = bench_paths(
            files, task=a.task, dtype=a.dtype, size=a.size,
            seed=int(getattr(a, "seed", 7)),
            track_id=str(getattr(a, "track", None) or "msl-single"),
            reps=int(getattr(a, "reps", 16)),
            target_id=getattr(a, "target", "apple-m4-max"),
        )
    except PairedCompareError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if getattr(a, "json", False):
        print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    else:
        print(_format_ab(document))
    return 0 if document.get("rows") and not document.get("status") else 2


def _format_ab(document: dict) -> str:
    """@brief 把 ``tcenv.ab.1`` 文档排成一张表。控制行是第一份文件。"""
    if not document.get("rows"):
        return str(document.get("reason") or "ab produced no measurement")
    control = document.get("control") or ""
    lines = [
        f"timed by: {document.get('timed_by')}  control: {control}  "
        f"reps: {document.get('reps')}  warmup: {document.get('warmup_reps')}",
        f"{'file':<40}{'median us':>12}{'min us':>11}{'cv':>8}{'vs control':>12}",
    ]
    for row in document["rows"]:
        ratio = row.get("paired_ratio_to_first")
        ratio_text = "control" if row.get("label") == control else (f"{ratio:.4f}" if ratio else "-")
        label = str(row.get("label"))
        if not row.get("rankable"):
            # 算错的行照旧计时，但排在表外，不给「哪版更快」增加证据。
            err = row.get("worst_scaled_err")
            label = f"[{row.get('status') or 'wrong'}] {label}"
            ratio_text = f"{err:.3g} over" if isinstance(err, (int, float)) else "not ranked"
        lines.append(
            f"{label[:39]:<40}{row.get('median_us', 0.0):>12.3f}"
            f"{row.get('min_us', 0.0):>11.3f}{row.get('cv', 0.0):>8.3f}{ratio_text:>12}"
        )
    for rejected in document.get("rejected") or []:
        lines.append(f"REJECT {rejected.get('file')}: {rejected.get('reason')}")
    ranking = document.get("ranking") or {}
    if ranking.get("order"):
        lines.append("fastest first (correct candidates only): " + " < ".join(ranking["order"]))
    elif ranking.get("reason"):
        lines.append(f"no ranking: {ranking['reason']}")
    if document.get("timed_by") != "gpu_timestamp":
        lines.append("note: wall clock fallback; do not compare with candidate_gpu_us_median")
    lines.append("read-only ranking; not publication evidence")
    return "\n".join(lines)
