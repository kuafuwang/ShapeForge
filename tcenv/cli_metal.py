"""Metal-specific CLI commands (paired-compare, ab)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _compare_pair(backend: str, path_a: Path, path_b: Path, **kwargs) -> dict:
    """@brief 把两份源码交给后端。Metal 会在本进程里绑定并交错计时。"""
    if backend != "metal":
        return {
            "applied": False,
            "supported": False,
            "backend": backend,
            "reason": f"paired compare is Metal-only; backend {backend!r} is not supported",
        }
    from tcenv.backends.metal.paired_compare import compare_for_backend

    return compare_for_backend(backend, path_a=path_a, path_b=path_b, **kwargs)


def cmd_ab(a: argparse.Namespace) -> int:
    """@brief 同进程多文件计时工具。只属于 Metal，其他 backend 返回"不支持"。

    把现有的 paired_compare.py 暴露成 CLI 子命令。不重复造轮子。
    """
    backend = getattr(a, "backend", "metal")
    if backend != "metal":
        print(f"tcenv.cli ab is Metal-only; backend {backend!r} is not supported", file=sys.stderr)
        return 2
    from tcenv.backends.metal.paired_compare import compare_for_backend

    path_a = Path(a.left)
    path_b = Path(a.right)
    try:
        sha_a = hashlib.sha256(path_a.read_bytes()).hexdigest()
        sha_b = hashlib.sha256(path_b.read_bytes()).hexdigest()
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        document = compare_for_backend(
            backend, path_a=path_a, path_b=path_b, sha_a=sha_a, sha_b=sha_b,
            seed=getattr(a, "seed", 7),
            task=getattr(a, "task", None), dtype=getattr(a, "dtype", None), size=getattr(a, "size", None),
            track_id=getattr(a, "track", None) or "msl-single",
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0 if document.get("applied") else 2


def cmd_paired_compare(a: argparse.Namespace) -> int:
    """@brief 比较两个候选文件。两个 SHA 写进工作目录的提名文件，不写进权威自测行。

    支持的后端在执行前先查配额。一旦进入执行，成功、失败、超时和崩溃都占一次
    timed 次数；超时和崩溃进入既有挂死记录。未知耗时不写成 GPU 秒数。
    任务书里的说明等 E1，这里只提供命令。
    """
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
