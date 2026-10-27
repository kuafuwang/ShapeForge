"""有界读取追加日志：要么一次流式遍历，要么只读尾部一段，不把整份文件读进内存。

这里只放读法，不放任何业务判断；``brief``、``observe.detail`` 与测试日志共用同一份实现。
"""

from __future__ import annotations

import json
from pathlib import Path

# GPU 事件日志、epic 日志与测试日志的默认尾读上限。实测 gpu_events.jsonl 2.1MB/15,600 行，
# 但显示只要最后几十行，读 256KB 足够覆盖并留出余量。
TAIL_BYTES = 262_144


def _row(line: str) -> dict | None:
    """@brief 把一行 JSONL 解析成字典；空行、坏行与非对象返回 None。

    不 ``strip`` 整行：``json.loads`` 本来接受行首行尾空白，而转录里的一行可以有几 MB，
    ``strip`` 会为它多复制一份，白白抬高流式遍历的峰值内存。

    @param line 一行文本。
    @return 解析后的字典，或 None。
    """
    if not line or line.isspace():
        return None
    try:
        row = json.loads(line)
    except ValueError:
        return None
    return row if isinstance(row, dict) else None


def iter_jsonl(path: Path):
    """@brief 逐行读一份 JSONL；跳过坏行，不构造全量列表。

    文件不可读时什么也不产出。进程被杀时尾巴上的半行按坏行丢掉，前面完整的事件仍然留下。

    @param path JSONL 文件路径。
    @return 生成器，逐条产出字典。
    """
    try:
        handle = Path(path).open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            row = _row(line)
            if row is not None:
                yield row


def parse_jsonl(text: str) -> list[dict]:
    """@brief 把一段 JSONL 文本解析成字典列表，坏行跳过。

    @param text JSONL 文本。
    @return 解析成功的字典列表。
    """
    rows = []
    for line in text.splitlines():
        row = _row(line)
        if row is not None:
            rows.append(row)
    return rows


def tail_text(path: Path, *, limit: int = TAIL_BYTES, trim_partial: bool = True) -> str:
    """@brief 读文件末尾至多 ``limit`` 字节。

    ``read(limit)`` 是硬上限：stat 之后文件继续被追加也读不出超过 ``limit`` 的字节。
    ``trim_partial`` 为 True 时丢掉第一个半行（从中间 byte 切开的那一行）；整体小于上限时不丢。

    @param path 文件路径。
    @param limit 最多读多少字节。
    @param trim_partial 是否丢掉第一个半行。
    @return 文本内容；文件不可读时为空字符串。
    """
    try:
        size = Path(path).stat().st_size
        with Path(path).open("rb") as handle:
            handle.seek(max(0, size - limit))
            data = handle.read(limit)
    except OSError:
        # 目录、权限、刚被删掉的路径都按“不可读”返回空串，与调用方约定一致。
        return ""
    text = data.decode("utf-8", errors="replace")
    if trim_partial and size > limit:
        # 只有切点确实在一行中间时才丢第一段；恰好落在行边界上时它是完整行。
        at_line_start = False
        try:
            with Path(path).open("rb") as probe:
                probe.seek(max(0, size - limit) - 1)
                at_line_start = probe.read(1) in (b"\n", b"\r")
        except OSError:
            at_line_start = False
        if not at_line_start:
            cut = text.find("\n")
            if cut >= 0:
                text = text[cut + 1 :]
    return text


def tail_lines(path: Path, *, keep: int, chunk: int = 65_536, max_bytes: int = TAIL_BYTES) -> list[str]:
    """@brief 后向读取最后 ``keep`` 个物理行，保持旧实现的“先取行、再解析”语义，且有硬字节预算。

    从文件尾按块往回读，凑够 ``keep`` 个换行就停；``max_bytes`` 是硬上限，单个超长行不会让这里
    退化成整读。预算先用完时，从中间切开的那一行（可能只是超长行的尾部）整行丢掉，返回预算内
    完整的行；一个完整行都装不下时返回空列表，而不是把整份文件读进来。

    文件以换行结尾时，末尾的空段不算一行；写到一半的最后一行照旧返回，交给解析器丢掉。

    @param path 文件路径。
    @param keep 最多返回多少行。
    @param chunk 每次回读的字节数。
    @param max_bytes 回读的硬字节上限。
    @return 行文本列表（按文件顺序）；文件不可读时为空。
    """
    keep = max(0, int(keep))
    chunk = max(1, int(chunk))
    max_bytes = max(0, int(max_bytes))
    if keep == 0 or max_bytes == 0:
        return []
    try:
        size = Path(path).stat().st_size
        handle = Path(path).open("rb")
    except OSError:
        return []
    budget = min(size, max_bytes)
    at_line_start = False
    with handle:
        data = b""
        position = size
        limited = False
        # 回读够不够用 CR/LF 分隔符计数判断（`\r\n` 算一个），不要只数 `\n`：CR-only 的日志
        # 会被当成一行而整份读完。最后仍交给 ``str.splitlines()``，分隔符语义与旧实现一致。
        # 多要一个分隔符，因为第一段可能是从行中间切开的半边，末尾可能是写了一半的行。
        while position > 0:
            remaining = budget - (size - position)
            if remaining <= 0:
                limited = True
                break
            step = min(chunk, position, remaining)
            position -= step
            handle.seek(position)
            data = handle.read(step) + data
            breaks = data.count(b"\n") + data.count(b"\r") - data.count(b"\r\n")
            if breaks > keep:
                break
            if size - position >= budget:
                limited = True
                break
        if limited and position > 0:
            # 只有截断点确实在一行中间时才丢第一段；恰好落在行边界上时它是完整行。
            try:
                handle.seek(position - 1)
                at_line_start = handle.read(1) in (b"\n", b"\r")
            except OSError:
                at_line_start = False
    lines = data.decode("utf-8", errors="replace").splitlines()
    if limited and position > 0 and not at_line_start and lines:
        lines = lines[1:]
    return lines[-keep:]


def head_text(path: Path, *, limit: int) -> str:
    """@brief 只读文件开头的 ``limit`` 字节；用来取内核片段，不整份读源码。

    @param path 文件路径。
    @param limit 最多读多少字节。
    @return 解码后的文本；文件不可读时为空字符串。
    """
    try:
        with Path(path).open("rb") as handle:
            data = handle.read(max(0, int(limit)))
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def tail_text_across(paths, *, limit: int = TAIL_BYTES) -> str:
    """@brief 读一组有序文件（旧→新）的逻辑尾文本，跨文件共用字节预算。

    只有最旧的那份可能被截断（预算是从新往旧花的），所以截断点是否落在行边界只看那一份的
    前一个字节：恰好落在行首时第一行是完整的，不能丢（``1\n2\n3\n`` 的 4 字节尾巴是
    ``2\n3\n``，不是一个 JSON 事件时同理）。

    @param paths 逻辑顺序的文件列表。
    @param limit 总字节上限。
    @return 拼接后的尾部文本；被截断的那份从行中间切开时丢掉第一个半行。
    """
    budget = max(0, int(limit))
    chunks: list[bytes] = []
    cut_path: Path | None = None
    cut_at = 0
    for path in reversed(list(paths)):
        if budget <= 0:
            break
        try:
            size = Path(path).stat().st_size
        except OSError:
            continue
        take = min(size, budget)
        if take <= 0:
            continue
        try:
            with Path(path).open("rb") as handle:
                handle.seek(size - take)
                data = handle.read(take)
        except OSError:
            continue
        if size > take:
            cut_path = Path(path)
            cut_at = size - take
        chunks.insert(0, data)
        budget -= take
    text = b"".join(chunks).decode("utf-8", errors="replace")
    if cut_path is not None and cut_at > 0:
        try:
            with cut_path.open("rb") as probe:
                probe.seek(cut_at - 1)
                at_line_start = probe.read(1) in (b"\n", b"\r")
        except OSError:
            at_line_start = False
        if not at_line_start:
            newline = text.find("\n")
            if newline >= 0:
                text = text[newline + 1 :]
    return text


def tail_valid_rows_across(paths, *, keep: int = 8, max_bytes: int = TAIL_BYTES) -> list[dict]:
    """@brief 跨有序文件取“最后 keep 条**有效**事件”（坏行/半行不占名额）。

    与 ``tail_rows_across`` 的“最后 keep 个物理行”不同：``brief`` 的旧行为是先解析再取尾部。

    @param paths 逻辑顺序的文件列表。
    @param keep 最多返回多少条可解析事件。
    @param max_bytes 跨所有文件的读入上限。
    @return 事件字典列表，按逻辑顺序。
    """
    rows = parse_jsonl(tail_text_across(paths, limit=max_bytes))
    return rows[-max(0, int(keep)):]


def tail_rows_across(paths, *, keep: int = 40, chunk: int = 65_536, max_bytes: int = TAIL_BYTES) -> list[dict]:
    """@brief 跨多个有序文件（旧→新）读逻辑尾：轮转后的分片 + 活动文件也有界。

    从最新的文件往回读，凑够 ``keep`` 行或共用字节预算用完就停；每个文件只读自己的尾巴，
    不会整读任何一份。分片是完整的，活动文件末尾的半行照旧交给解析器丢掉。

    @param paths 逻辑顺序（旧→新）的文件列表。
    @param keep 最多返回多少条可解析事件。
    @param chunk 单次回读块大小。
    @param max_bytes 跨所有文件的读入硬上限。
    @return 事件字典列表，按逻辑顺序。
    """
    budget = max(0, int(max_bytes))
    lines: list[str] = []
    for path in reversed(list(paths)):
        if len(lines) >= keep or budget <= 0:
            break
        try:
            size = Path(path).stat().st_size
        except OSError:
            continue
        take = min(size, budget)
        if take <= 0:
            continue
        lines = tail_lines(path, keep=keep - len(lines), chunk=chunk, max_bytes=take) + lines
        budget -= take
    return parse_jsonl("\n".join(lines[-max(0, int(keep)):]))


def tail_rows(path: Path, *, keep: int = 40, chunk: int = 65_536, max_bytes: int = TAIL_BYTES) -> list[dict]:
    """@brief 取最后 ``keep`` 个物理行再解析；坏行与半行照旧占一个名额。

    这与旧实现的 ``read_text().splitlines()[-keep:]`` 同义：不是“最后 keep 条有效事件”。

    @param path JSONL 文件路径。
    @param keep 取最后多少个物理行。
    @param chunk 回读块大小。
    @param max_bytes 回读的硬字节上限。
    @return 解析成功的字典列表（少于 ``keep`` 条是正常的）。
    """
    return parse_jsonl("\n".join(tail_lines(path, keep=keep, chunk=chunk, max_bytes=max_bytes)))
