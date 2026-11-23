"""Fail-closed reads for the files a run directory contributes.

The index, the brief and the publication gate must rebuild their facts from the run tree itself.
A file leaf that is a symbolic link — or a directory component *below* the anchor that is one —
is refused instead of followed: otherwise a planted ``result.json -> /outside/secret.json`` would
ingest bytes from outside ``runs/`` and a report could quote a number the run never produced.

The leaf is opened once with ``O_NOFOLLOW`` and every parent component is walked from the anchor
with ``O_DIRECTORY|O_NOFOLLOW`` relative to the already-open parent; the bytes are read from that
same descriptor.  Resolving the path first and then opening it again would leave a window in which
a swap could redirect the open.

The anchor is the ``runs/`` root for anything lexically under it, and otherwise the nearest
enclosing directory that holds a ``state.json`` or ``result.json`` (so an explicit run directory
outside ``runs/`` still refuses links below it).  With neither, the file's own parent is the anchor
and only the leaf is checked — those reads are the caller's explicitly named global files.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path

#: ``O_NOFOLLOW`` is absent on some platforms; 0 keeps the walk working there (the component loop
#: still refuses a symlink reported by lstat).  This repository only ships Metal on macOS.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)

#: How far up an explicit path is searched for its enclosing run directory.
_ANCESTOR_LIMIT = 64


class RunFileRefused(OSError):
    """A run file was a link or left its anchor; no byte from it was read."""


def refused(path) -> RunFileRefused:
    """@brief 构造「拒绝读取」的 ``OSError``（``ELOOP``）。

    @param path 被拒绝的路径。
    @return 带路径的异常。
    """
    return RunFileRefused(errno.ELOOP, "refusing a linked or out-of-tree run file", os.fspath(path))


def _absolute(path) -> Path:
    """@brief 规整成绝对路径文本，不碰文件系统。

    @param path 任意路径。
    @return ``normpath`` 后的绝对 ``Path``。
    """
    text = os.fspath(path)
    if not os.path.isabs(text):
        text = os.path.join(os.getcwd(), text)
    return Path(os.path.normpath(text))


def _looks_like_run(directory: Path) -> bool:
    """@brief 该目录是否像一次运行：含 ``state.json`` 或 ``result.json`` 目录项（不跟随链接）。

    @param directory 候选目录。
    @return 含任一快照名时为 True。
    """
    return any(os.path.lexists(os.path.join(os.fspath(directory), name)) for name in ("state.json", "result.json"))


def anchor_for(path) -> Path:
    """@brief 选择符号链接检查的起点。

    ``runs/`` 内的路径以根为起点（逐级拒绝链接，根内目录别名必须先解析成 canonical 路径）；
    根外路径以最近的、含快照的祖先目录为起点，找不到时退回文件自己的父目录。

    @param path 待读路径。
    @return 绝对起点目录。
    """
    target = _absolute(path)
    try:
        from observe.detail import root_path

        root = _absolute(root_path()[0])
    except Exception:  # noqa: BLE001 - 根不可用时只做叶子检查，读取仍要工作
        root = None
    text, root_text = os.fspath(target), os.fspath(root) if root is not None else ""
    if root_text and (text == root_text or text.startswith(root_text + os.sep)) and root is not None:
        return root
    current = target.parent
    for _step in range(_ANCESTOR_LIMIT):
        if _looks_like_run(current):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return target.parent


def _parts(path, anchor) -> tuple[str, ...]:
    """@brief 把 ``path`` 拆成相对 ``anchor`` 的分量，越界即拒绝。

    @param path 待读路径。
    @param anchor 已规整的起点目录。
    @return 分量元组（至少一项）。
    @raises RunFileRefused 路径不在起点内、等于起点或含 ``.``/``..`` 时。
    """
    target, base = _absolute(path), _absolute(anchor)
    target_text, base_text = os.fspath(target), os.fspath(base)
    if target_text == base_text or not target_text.startswith(base_text + os.sep):
        raise refused(path)
    parts = tuple(target_text[len(base_text) + 1:].split(os.sep))
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise refused(path)
    return parts


def open_file(path, *, anchor=None) -> int:
    """@brief 打开一个普通文件；叶子与起点以下的父目录都不得是符号链接。

    父目录逐级用 ``O_DIRECTORY|O_NOFOLLOW`` 相对已打开的父 FD 打开，叶子用 ``O_NOFOLLOW``；
    返回的 FD 指向通过 ``fstat`` 检查的普通文件，调用方从同一个 FD 读取。

    @param path 文件路径。
    @param anchor 检查起点；None 时用 :func:`anchor_for`。
    @return 打开的只读 FD（调用方负责关闭）。
    @raises OSError 路径越界、任一分量是符号链接或叶子不是普通文件时（``RunFileRefused`` 是 ``OSError``）。
    """
    base = _absolute(anchor if anchor is not None else anchor_for(path))
    parts = _parts(path, base)
    base_fd = os.open(os.fspath(base), os.O_RDONLY | _DIRECTORY)
    current = base_fd
    try:
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=current)
            if current != base_fd:
                os.close(current)
            current = following
        fd = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW | _NONBLOCK, dir_fd=current)
    finally:
        if current != base_fd:
            os.close(current)
        os.close(base_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise refused(path)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _leaf_stat(path, *, anchor=None) -> os.stat_result | None:
    """@brief 逐级 ``lstat`` 后返回叶子自身的 stat；越界、链接或缺失时为 None。

    只看元数据（指纹/存在性），不打开文件。父目录在检查与后续读取之间的竞态由
    :func:`open_file` 的逐级 FD 打开兜住；这里读出的 stat 只用于“要不要重读”的判断。

    @param path 路径。
    @param anchor 检查起点；None 时用 :func:`anchor_for`。
    @return ``lstat`` 结果或 None。
    """
    base = _absolute(anchor if anchor is not None else anchor_for(path))
    try:
        parts = _parts(path, base)
    except OSError:
        return None
    current = base
    for part in parts[:-1]:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError:
            return None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            return None
    try:
        info = os.lstat(_absolute(path))
    except OSError:
        return None
    return None if stat.S_ISLNK(info.st_mode) else info


def stat_leaf(path, *, anchor=None) -> os.stat_result | None:
    """@brief 文件或目录的非链接 ``lstat``；任何链接分量都返回 None。

    @param path 路径。
    @param anchor 检查起点；None 时用 :func:`anchor_for`。
    @return stat 或 None。
    """
    return _leaf_stat(path, anchor=anchor)


def is_regular_file(path, *, anchor=None) -> bool:
    """@brief 该路径是一个不以符号链接出现的普通文件。

    @param path 路径。
    @param anchor 检查起点。
    @return 是普通文件时为 True。
    """
    info = _leaf_stat(path, anchor=anchor)
    return info is not None and stat.S_ISREG(info.st_mode)


def is_directory(path, *, anchor=None) -> bool:
    """@brief 该路径是一个不以符号链接出现的目录。

    @param path 路径。
    @param anchor 检查起点。
    @return 是目录时为 True。
    """
    info = _leaf_stat(path, anchor=anchor)
    return info is not None and stat.S_ISDIR(info.st_mode)


def read_bytes(path, *, anchor=None, limit: int | None = None) -> bytes:
    """@brief 从同一个 ``O_NOFOLLOW`` FD 读取字节。

    @param path 文件路径。
    @param anchor 检查起点。
    @param limit 最多读多少字节；None 表示整份。
    @return 读到的字节。
    @raises OSError 被拒绝、缺失或不可读时。
    """
    fd = open_file(path, anchor=anchor)
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = -1
            return handle.read() if limit is None else handle.read(limit)
    finally:
        if fd >= 0:
            os.close(fd)


def read_text(path, *, anchor=None, encoding: str = "utf-8", errors: str = "strict",
              limit: int | None = None) -> str:
    """@brief 从同一个 FD 读文本。

    @param path 文件路径。
    @param anchor 检查起点。
    @param encoding 文本编码。
    @param errors 解码错误策略。
    @param limit 最多读多少字节；None 表示整份。
    @return 解码后的文本。
    @raises OSError 被拒绝、缺失或不可读时；``UnicodeDecodeError`` 按 ``errors`` 处理。
    """
    return read_bytes(path, anchor=anchor, limit=limit).decode(encoding, errors=errors)


def load_json(path, *, anchor=None) -> dict | None:
    """@brief 安全地读一个 JSON 对象；缺失、被拒绝、读坏或根节点不是对象时返回 None。

    @param path JSON 文件路径。
    @param anchor 检查起点。
    @return 字典或 None。
    """
    try:
        text = read_text(path, anchor=anchor)
    except (OSError, UnicodeDecodeError):
        return None
    try:
        document = json.loads(text)
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


def iter_jsonl(path, *, anchor=None):
    """@brief 安全地逐行产出 JSON 对象；坏行跳过。

    @param path JSONL 文件路径。
    @param anchor 检查起点。
    @return 字典生成器；文件不可读或被拒绝时什么也不产出。
    """
    from taic.reads import _row

    try:
        fd = open_file(path, anchor=anchor)
    except OSError:
        return
    try:
        handle = os.fdopen(fd, "r", encoding="utf-8", errors="replace", closefd=True)
        fd = -1
    finally:
        if fd >= 0:
            os.close(fd)
    with handle:
        for line in handle:
            item = _row(line)
            if item is not None:
                yield item


def iter_lines(path, *, anchor=None):
    """@brief 安全地逐行产出文本；文件不可读或被拒绝时什么也不产出。

    @param path 文本文件路径。
    @param anchor 检查起点。
    @return 行文本生成器。
    """
    try:
        fd = open_file(path, anchor=anchor)
    except OSError:
        return
    try:
        handle = os.fdopen(fd, "r", encoding="utf-8", errors="replace", closefd=True)
        fd = -1
    finally:
        if fd >= 0:
            os.close(fd)
    with handle:
        for line in handle:
            yield line


def head_text(path, *, anchor=None, limit: int) -> str:
    """@brief 安全地读文件开头 ``limit`` 字节。

    @param path 文件路径。
    @param anchor 检查起点。
    @param limit 最多读多少字节。
    @return 解码文本；不可读或被拒绝时为空串。
    """
    try:
        return read_text(path, anchor=anchor, limit=max(0, int(limit)), errors="replace")
    except OSError:
        return ""


def tail_text(path, *, anchor=None, limit: int, trim_partial: bool = True) -> str:
    """@brief 安全地从同一个 FD 读文件末尾至多 ``limit`` 字节。

    与 :func:`taic.reads.tail_text` 的语义一致：``read(limit)`` 是硬上限，``trim_partial`` 为真时
    丢掉从中间切开的第一行（只有确实被截断时才丢）。被拒绝或不可读时返回空串。

    @param path 文件路径。
    @param anchor 检查起点。
    @param limit 最多读多少字节。
    @param trim_partial 是否丢掉第一个半行。
    @return 文本内容；不可读时为 ``""``。
    """
    try:
        fd = open_file(path, anchor=anchor)
    except OSError:
        return ""
    try:
        size = os.fstat(fd).st_size
        start = max(0, size - max(0, int(limit)))
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = -1
            handle.seek(start)
            data = handle.read(max(0, int(limit)))
            at_line_start = True
            if trim_partial and size > limit:
                if start > 0:
                    # ``pread`` 不移动共享的读取偏移：探测前一个字节不会吃掉这次要读的第一行。
                    at_line_start = os.pread(handle.fileno(), 1, start - 1) in (b"\n", b"\r")
                else:
                    at_line_start = True
    finally:
        if fd >= 0:
            os.close(fd)
    text = data.decode("utf-8", errors="replace")
    if not at_line_start:
        cut = text.find("\n")
        if cut >= 0:
            text = text[cut + 1:]
    return text
