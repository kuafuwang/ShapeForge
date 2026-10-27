"""§9.1 内容寻址 blob 的保留策略：默认只报告，``--apply`` 才删除。

blob 是**持久证据**，不是派生可删数据：它一旦生成就不能从 ``runs/`` + ``kernels/`` 重建（源会被
30 天 TTL 删掉），所以保留期必须长于 ``deleted`` 的 30 天。删除依据是 **blob 文件自身 mtime**，
不是 ``blob_refs.created_at``——没有引用时根本没有 ref 行可读。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from taic.kernel_catalog import blob_path, blob_root

BLOB_NAME = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_RETENTION_DAYS = 365.0


def retention_days() -> float:
    """@brief 保留天数：``TAIC_BLOB_RETENTION_DAYS`` 覆盖，默认 365。

    @return 保留天数（非负）。
    """
    try:
        return max(0.0, float(os.environ.get("TAIC_BLOB_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))))
    except ValueError:
        return DEFAULT_RETENTION_DAYS


def _files(root: Path) -> list[Path]:
    """@brief 枚举 blob 文件（只认 ``<2 hex>/<64 hex>`` 的普通文件，不跟符号链接）。

    @param root blob 根目录。
    @return 按路径排序的文件列表。
    """
    if not root.is_dir():
        return []
    out: list[Path] = []
    try:
        buckets = sorted(root.iterdir())
    except OSError:
        return []
    for bucket in buckets:
        if not bucket.is_dir() or bucket.is_symlink() or not re.fullmatch(r"[0-9a-f]{2}", bucket.name):
            continue
        try:
            items = sorted(bucket.iterdir())
        except OSError:
            continue
        for item in items:
            if not item.is_file() or item.is_symlink():
                continue
            if item.name[:2] != bucket.name or not BLOB_NAME.match(item.name):
                continue
            out.append(item)
    return out


def scan(root: Path | None = None, *, older_than_days: float | None = None,
         now: float | None = None) -> list[dict]:
    """@brief 只读扫描：每个 blob 一行，``eligible`` 表示文件 mtime 已到保留线。

    不需要 ``blob_refs`` 行：无引用的 blob 同样要能列出并清理。

    @param root blob 根目录；None 表示 ``taic.kernel_catalog.blob_root()``。
    @param older_than_days 保留天数；None 表示 ``retention_days()``。
    @param now 当前时间戳；None 表示 ``time.time()``。
    @return 每行含 ``sha256``/``path``/``bytes``/``age_days``/``eligible``。
    """
    days = retention_days() if older_than_days is None else max(0.0, float(older_than_days))
    moment = time.time() if now is None else now
    rows = []
    for path in _files(Path(root) if root is not None else blob_root()):
        try:
            stat = path.stat()
        except OSError:
            continue
        age = (moment - stat.st_mtime) / 86400.0
        rows.append({
            "sha256": path.name,
            "path": str(path),
            "bytes": stat.st_size,
            "age_days": round(age, 3),
            "eligible": age >= days,
        })
    return rows


def apply(rows: list[dict]) -> dict:
    """@brief 删除 ``eligible`` 为真的 blob；不碰其它行，删空的分桶目录顺手收掉。

    @param rows ``scan`` 返回的行列表。
    @return 含 ``removed``/``bytes`` 的字典。
    """
    removed: list[str] = []
    freed = 0
    for row in rows:
        if not row.get("eligible"):
            continue
        path = Path(str(row.get("path") or ""))
        try:
            freed += path.stat().st_size
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            continue
        removed.append(str(path))
        try:
            path.parent.rmdir()
        except OSError:
            # 分桶里还有别的 blob，或者目录不可删；都不影响已删的这一份。
            pass
    return {"removed": removed, "bytes": freed}


def read_source(sha256: str, root: Path | None = None) -> str | None:
    """@brief 读回一份 blob 源码；摘要非法或文件不存在时返回 None。

    @param sha256 十六进制摘要（只接受 64 位小写）。
    @param root blob 根目录；None 表示 ``taic.kernel_catalog.blob_root()``。
    @return 源码文本，或 None。
    """
    if not isinstance(sha256, str) or not BLOB_NAME.match(sha256):
        return None
    base = Path(root) if root is not None else blob_root()
    path = blob_path(sha256) if root is None else base / sha256[:2] / sha256
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
