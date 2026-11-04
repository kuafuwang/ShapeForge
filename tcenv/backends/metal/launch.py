"""The optional ``/* taic-launch */`` block: several kernels, each with its own grid.

Absent, evaluation is the single contract kernel on the contract grid. Present, it is the
only description of how to launch. One parser, used by the reply extractor, the evaluator,
feature analysis and the profiler, so those stages cannot disagree about the same bytes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..base import LaunchRejected

# 段数上限的用处是给解析和 ABI 检查一个有限的输入，不是性能：编码实测 1.42 µs/段（1000 段 1.42 ms），
# 而每段的 dispatch 都在计时窗内，多段只会自罚时间。真实的段数本来就被回复长度压住（max_tokens 48000，
# 一行 launch 约 50 字节）。32 的依据是经验区间：Winograd F(2x2,3x3) 是 4 段，split-K 是 k+1 段（本仓库
# 见过的 k 不超过 16），32 给到 2 倍余量。
MAX_STAGES = 32
# 线程组上限按核自己能寻址的规模定：契约的线程索引是 32 位 uint，所以 2^32 是硬界。
# 2^24 组 × 每组最多 1024 线程 = 1.7e10 线程，是仓库最大张量（paper 的 2^24 元素）的 1000 倍；
# 65536 组（2^24 元素、每组 256 线程）必须放行。再大只可能是病态核，不值得为它保留解析路径。
MAX_THREADGROUPS = 1 << 24
MARKER = "taic-launch"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class WorkspaceDecl:
    name: str
    nbytes: int
    zero: bool


@dataclass(frozen=True)
class StageDecl:
    kernel: str
    grid: tuple[int, int, int]
    threads: tuple[int, int, int]
    threadgroup_bytes: int
    workspaces: tuple[str, ...]


@dataclass(frozen=True)
class LaunchSpec:
    math: str
    workspaces: tuple[WorkspaceDecl, ...]
    stages: tuple[StageDecl, ...]

    def digest(self) -> str:
        """@brief 生成描述整份启动规格的稳定文本。

        @return 逐行展开的规格摘要，可用于哈希。
        """
        parts = [f"math={self.math}"]
        for w in self.workspaces:
            parts.append(f"workspace {w.name} {w.nbytes}{' zero' if w.zero else ''}")
        for s in self.stages:
            ws = ",".join(s.workspaces)
            parts.append(
                f"launch {s.kernel} grid={s.grid[0]},{s.grid[1]},{s.grid[2]} "
                f"threads={s.threads[0]},{s.threads[1]},{s.threads[2]} "
                f"threadgroup={s.threadgroup_bytes} workspace={ws}"
            )
        return "\n".join(parts)


def unique_launch_block(source: str) -> str | None:
    """@brief 源码中唯一的 ``taic-launch`` 块正文（不做尺寸求值）。

    The structural half of :func:`parse_launch_spec`, reused by callers that need to know whether
    a source really carries one launch block without evaluating its grid: the marker must appear
    exactly once, inside a block comment, and nowhere else.  A line comment, a string literal or a
    second block is rejected.

    @param source MSL 源码。
    @return 启动块正文；源码没有启动块时为 None。
    @raises LaunchRejected 标记出现在块注释之外或出现多次。
    """
    blocks, stray = _scan(source or "")
    launches: list[str] = []
    for body in blocks:
        tokens = body.split()
        if tokens and tokens[0] == MARKER:
            launches.append(body)
        elif MARKER in body:
            raise LaunchRejected(
                "taic-launch must appear exactly once, as the first token of its own /* taic-launch */ block; "
                "a prefix such as taic-launch-fake, a non-first token, a second block, a line comment, or a string "
                "containing the marker is rejected"
            )
    if stray or len(launches) > 1:
        raise LaunchRejected(
            "taic-launch must appear exactly once, as its own /* taic-launch */ block; "
            "a second block, a line comment, or a string containing the marker is rejected"
        )
    return launches[0] if launches else None


def parse_launch_spec(source: str, constants: dict[str, int]) -> LaunchSpec | None:
    """@brief 解析源码里唯一的 ``taic-launch`` 块。

    @param source MSL 源码。
    @param constants 表达式里可用的整数常量。
    @return 启动规格；源码没有启动块时为 None。
    @raises LaunchRejected 当标记出现在块注释之外、出现多次或块内容非法。

    The one launch block in ``source``, or ``None`` when the source has none.

    Raises ``LaunchRejected`` when the marker appears anywhere except exactly one block comment
    whose first token is ``taic-launch``, or when that block does not parse.
    """
    body = unique_launch_block(source)
    if body is None:
        return None
    return _parse_body(body, constants)


def declared_stage_kernels(source: str) -> list[str] | None:
    """@brief 只取 ``launch`` 行上的核名，不求值尺寸。

    @param source MSL 源码。
    @return 核名列表；源码不是唯一启动块时为 None。

    Kernel names on ``launch`` lines, without evaluating sizes.

    ``None`` when the source is not exactly one launch block. Feature analysis uses this so an
    unused kernel in the same file is not credited.
    """
    try:
        body = unique_launch_block(source)
    except LaunchRejected:
        return None
    if body is None:
        return None
    names: list[str] = []
    for raw in body.splitlines():
        parts = raw.split()
        if parts and parts[0] == "launch" and len(parts) > 1 and _NAME.fullmatch(parts[1]):
            names.append(parts[1])
    return names or None


def launch_math(source: str) -> str | None:
    """@brief 从源码判断该用 ``fast`` 还是 ``safe`` 数学模式。

    @param source MSL 源码。
    @return ``fast`` 或 ``safe``；没有启动块时为 None。

    ``fast`` or ``safe`` when the source has one launch block, else ``None``.

    A block that fails later (unknown constants, bad sizes) still reports the math line, so a
    ``math safe`` candidate is not compiled as fast-math.
    """
    try:
        spec = parse_launch_spec(source, {})
    except LaunchRejected:
        spec = None
    if spec is not None:
        return spec.math
    names = declared_stage_kernels(source)
    if names is None and "taic-launch" not in (source or ""):
        return None
    if re.search(r"(?m)^\s*math\s+safe\s*$", source or ""):
        return "safe"
    if re.search(r"(?m)^\s*math\s+relaxed\s*$", source or ""):
        return "relaxed"
    if "taic-launch" in (source or ""):
        return "fast"
    return None


def check_launch_limits(
    spec: LaunchSpec,
    *,
    contract_names: set[str],
    max_threads_xyz: tuple[int, int, int],
    max_threads_product: int,
    max_threadgroup_memory_bytes: int,
    recommended_working_set_bytes: int,
    max_buffer_length: int,
    rotation_sets: int,
) -> None:
    """@brief 核对启动规格是否越过设备与预算上限。

    @param spec 启动规格。
    @param contract_names 契约参数名集合，用于查重名。
    @param max_threads_xyz 设备每组线程数三维上限。
    @param max_threads_product 设备每组线程总数上限。
    @param max_threadgroup_memory_bytes 设备 threadgroup 内存上限。
    @param recommended_working_set_bytes 推荐工作集大小，用于 workspace 预算。
    @param max_buffer_length 单块缓冲长度上限。
    @param rotation_sets 计时轮转组数，用于放大 workspace 预算。
    @raises LaunchRejected 当段数、重名、尺寸或 workspace 预算越界。

    Device and size limits. Raises ``LaunchRejected`` before any allocation.
    """
    if len(spec.stages) > MAX_STAGES:
        raise LaunchRejected(f"taic-launch has {len(spec.stages)} stages; the limit is {MAX_STAGES}")
    names = {w.name for w in spec.workspaces}
    if names & contract_names:
        hit = ", ".join(sorted(names & contract_names))
        raise LaunchRejected(f"workspace name collides with a contract argument: {hit}")
    total = 0
    for w in spec.workspaces:
        if w.nbytes > max_buffer_length:
            raise LaunchRejected(f"workspace `{w.name}` is {w.nbytes} bytes; the device buffer limit is {max_buffer_length}")
        total += w.nbytes
    # 每块工作区两侧各有 16384 字节护栏，和输出缓冲同一宽度。预算按分配长度算，不按声明的有效载荷算。
    guard = 16384
    padded = total + len(spec.workspaces) * 2 * guard
    copies = max(1, int(rotation_sets))
    cap = int(0.25 * recommended_working_set_bytes)
    kernel_names = {stage.kernel for stage in spec.stages}
    if kernel_names & names:
        hit = ", ".join(sorted(kernel_names & names))
        raise LaunchRejected(f"workspace name collides with a launched kernel: {hit}")
    if padded * copies > cap:
        raise LaunchRejected(
            f"workspaces use {padded} bytes including guard bands, times {copies} rotation sets, above the {cap} byte workspace budget"
        )
    for stage in spec.stages:
        product = 1
        for i, n in enumerate(stage.threads):
            if n > max_threads_xyz[i]:
                raise LaunchRejected(
                    f"stage `{stage.kernel}` threads component {n} exceeds the device limit {max_threads_xyz[i]}"
                )
            product *= n
        if product > max_threads_product:
            raise LaunchRejected(
                f"stage `{stage.kernel}` launches {product} threads per group; the device allows {max_threads_product}"
            )
        if stage.threadgroup_bytes > max_threadgroup_memory_bytes:
            raise LaunchRejected(
                f"stage `{stage.kernel}` asks for {stage.threadgroup_bytes} B of dynamic threadgroup memory; "
                f"the device limit is {max_threadgroup_memory_bytes} B"
            )


def _scan(source: str) -> tuple[list[str], bool]:
    """@brief 扫出块注释正文，并检测标记是否出现在块注释之外。

    @param source MSL 源码。
    @return (块注释正文列表, 是否在块注释之外出现标记)。
    @raises LaunchRejected 当块注释没有闭合。

    Block-comment bodies, and whether ``taic-launch`` appears outside a block comment.
    """
    blocks: list[str] = []
    stray = False
    i, n = 0, len(source)
    in_line = False
    in_string = False
    quote = ""
    escape = False
    block_at: int | None = None
    while i < n:
        if in_line:
            if source[i] == "\n":
                in_line = False
            elif source.startswith(MARKER, i):
                stray = True
            i += 1
            continue
        if in_string:
            if escape:
                escape = False
            elif source[i] == "\\":
                escape = True
            elif source[i] == quote:
                in_string = False
            elif source.startswith(MARKER, i):
                stray = True
            i += 1
            continue
        if block_at is not None:
            if source.startswith("*/", i):
                blocks.append(source[block_at + 2 : i])
                block_at = None
                i += 2
                continue
            i += 1
            continue
        if source.startswith("//", i):
            in_line = True
            i += 2
            continue
        if source.startswith("/*", i):
            block_at = i
            i += 2
            continue
        if source[i] in "\"'":
            in_string = True
            quote = source[i]
            escape = False
        i += 1
    if block_at is not None:
        raise LaunchRejected("taic-launch block comment is not closed")
    return blocks, stray


def _parse_body(body: str, constants: dict[str, int]) -> LaunchSpec:
    """@brief 解析启动块正文，生成启动规格。

    @param body 块注释正文。
    @param constants 表达式里可用的整数常量。
    @return 解析出的启动规格。
    @raises LaunchRejected 当某一行或某个尺寸不合法。
    """
    text = body.strip()
    if not text.startswith(MARKER):
        raise LaunchRejected("launch block does not start with taic-launch")
    rest = text[len(MARKER) :].lstrip()
    if rest.startswith("*/"):
        raise LaunchRejected("taic-launch block is empty")
    math = "fast"
    seen_math = False
    workspaces: list[WorkspaceDecl] = []
    seen_ws: set[str] = set()
    stages: list[StageDecl] = []
    for raw in rest.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "math":
            if seen_math or len(parts) != 2 or parts[1] not in ("fast", "safe", "relaxed"):
                raise LaunchRejected("math must be a single `math fast`, `math safe` or `math relaxed` line")
            math = parts[1]
            seen_math = True
        elif kind == "workspace":
            if len(parts) < 3:
                raise LaunchRejected(f"bad workspace line: {line}")
            name = parts[1]
            zero = parts[-1] == "zero"
            expr = " ".join(parts[2:-1] if zero else parts[2:])
            if not expr or (zero and parts[-1] != "zero"):
                raise LaunchRejected(f"bad workspace line: {line}")
            if not _NAME.fullmatch(name) or name in seen_ws:
                raise LaunchRejected(f"workspace name `{name}` is missing or repeated")
            nbytes = _eval_expr(expr, constants)
            if nbytes < 4 or nbytes % 4 != 0:
                raise LaunchRejected(f"workspace `{name}` must be a multiple of 4 bytes and at least 4")
            seen_ws.add(name)
            workspaces.append(WorkspaceDecl(name, nbytes, zero))
        elif kind == "launch":
            stages.append(_parse_launch_line(line, constants, seen_ws))
        else:
            raise LaunchRejected(f"unknown taic-launch line: {line}")
    if not stages:
        raise LaunchRejected("taic-launch has no launch line")
    if len(stages) > MAX_STAGES:
        raise LaunchRejected(f"taic-launch has {len(stages)} stages; the limit is {MAX_STAGES}")
    return LaunchSpec(math, tuple(workspaces), tuple(stages))


_NEXT_FIELD = re.compile(r"\s+(grid|threads|threadgroup|workspace)\s*=")


def _parse_launch_line(line: str, constants: dict[str, int], workspaces: set[str]) -> StageDecl:
    """@brief 解析一行 ``launch`` 声明。

    @param line 该行原文。
    @param constants 表达式里可用的整数常量。
    @param workspaces 已声明的 workspace 名集合。
    @return 该段的声明。
    @raises LaunchRejected 当字段缺失、重复或取值非法。
    """
    rest = line[len("launch") :].strip()
    name = re.match(r"[A-Za-z_][A-Za-z0-9_]*", rest)
    if not name:
        raise LaunchRejected("launch line needs a kernel name")
    kernel = name.group(0)
    rest = rest[name.end() :]
    fields: dict[str, str] = {}
    while rest.strip():
        rest = rest.lstrip()
        key_at = re.match(r"(grid|threads|threadgroup|workspace)\s*=\s*", rest)
        if not key_at:
            word = rest.split()[0]
            raise LaunchRejected(f"launch `{kernel}` has a bare word `{word}`")
        key = key_at.group(1)
        if key in fields:
            raise LaunchRejected(f"launch `{kernel}` has a bad or repeated field `{key}`")
        value, rest = _read_field_value(rest[key_at.end() :])
        fields[key] = value
    if "grid" not in fields or "threads" not in fields:
        raise LaunchRejected(f"launch `{kernel}` needs grid= and threads=")
    grid = _triple(fields["grid"], constants, f"launch `{kernel}` grid")
    threads = _triple(fields["threads"], constants, f"launch `{kernel}` threads")
    groups = grid[0] * grid[1] * grid[2]
    if groups > MAX_THREADGROUPS:
        raise LaunchRejected(f"launch `{kernel}` has {groups} threadgroups; the limit is {MAX_THREADGROUPS}")
    tg = 0
    if "threadgroup" in fields:
        tg = _eval_expr(fields["threadgroup"], constants)
        if tg < 0:
            raise LaunchRejected(f"launch `{kernel}` threadgroup byte count is negative")
    used: list[str] = []
    if "workspace" in fields and fields["workspace"]:
        for name in fields["workspace"].split(","):
            if name not in workspaces or name in used:
                raise LaunchRejected(f"launch `{kernel}` workspace `{name}` is unknown or repeated")
            used.append(name)
    return StageDecl(kernel, grid, threads, tg, tuple(used))


def _read_field_value(text: str) -> tuple[str, str]:
    """@brief 读出 ``key=`` 之后的字段值。

    @param text 等号之后的文本。
    @return (字段值, 剩余文本)。

    The text after ``key=``. Spaces and a wrapping ``( ... )`` stay part of the value.

    ``grid=(16, 1, 1)`` and ``grid=16,1,1`` are the same three numbers. The next
    ``threads=`` / ``grid=`` / ``threadgroup=`` / ``workspace=`` ends the value.
    """
    if text.startswith("("):
        end = _closing_paren(text)
        if end is not None and (_NEXT_FIELD.match(text[end + 1 :]) or not text[end + 1 :].strip()):
            return text[1:end].strip(), text[end + 1 :]
    for match in _NEXT_FIELD.finditer(text):
        if _paren_depth(text[: match.start()]) == 0:
            return text[: match.start()].strip(), text[match.start() :]
    return text.strip(), ""


def _closing_paren(text: str) -> int | None:
    """@brief 找到与第一个字符处左括号配对的右括号。

    @param text 以左括号开头的文本。
    @return 右括号下标；未闭合时为 None。
    """
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _paren_depth(text: str) -> int:
    """@brief 返回文本的括号净值。

    @param text 待统计文本。
    @return 左括号数减右括号数。
    """
    return text.count("(") - text.count(")")


def _triple(text: str, constants: dict[str, int], what: str) -> tuple[int, int, int]:
    """@brief 解析逗号分隔的三元组整数。

    @param text 字段值文本。
    @param constants 表达式里可用的整数常量。
    @param what 出错信息里的字段名。
    @return 三个分量。
    @raises LaunchRejected 当分量不足三个或含有小于 1 的值。
    """
    bits = text.split(",")
    if len(bits) != 3:
        raise LaunchRejected(f"{what} needs three comma-separated integers")
    values = tuple(_eval_expr(bit, constants) for bit in bits)
    if any(v < 1 for v in values):
        raise LaunchRejected(f"{what} components must be >= 1")
    return values  # type: ignore[return-value]


def _eval_expr(text: str, constants: dict[str, int]) -> int:
    """@brief 求值启动块里的整数表达式。

    @param text 表达式文本。
    @param constants 可用的整数常量。
    @return 求值结果。
    @raises LaunchRejected 当表达式为空、有未知名字、非法字符、除不尽或有多余输入。
    """
    tokens = _tokenize(text.strip())
    if not tokens:
        raise LaunchRejected("empty integer expression")
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def eat(kind: str | None = None) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise LaunchRejected(f"integer expression `{text}` ended early")
        tok = tokens[pos]
        if kind is not None and tok != kind:
            raise LaunchRejected(f"integer expression `{text}` expected `{kind}`")
        pos += 1
        return tok

    def parse_expr() -> int:
        value = parse_term()
        while peek() in ("+", "-"):
            op = eat()
            rhs = parse_term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def parse_term() -> int:
        value = parse_factor()
        while peek() in ("*", "/"):
            op = eat()
            rhs = parse_factor()
            if op == "*":
                value *= rhs
            else:
                if rhs == 0 or value % rhs != 0:
                    raise LaunchRejected(f"integer expression `{text}` does not divide evenly")
                value //= rhs
        return value

    def parse_factor() -> int:
        tok = peek()
        if tok == "-":
            eat("-")
            return -parse_factor()
        if tok == "(":
            eat("(")
            value = parse_expr()
            eat(")")
            return value
        eat()
        if tok is None:
            raise LaunchRejected(f"integer expression `{text}` ended early")
        if tok.isdigit():
            try:
                return int(tok)
            except ValueError as exc:
                # A decimal literal longer than the interpreter's digit limit is a candidate typo,
                # not an infrastructure failure: keep it inside the launch closed set.
                raise LaunchRejected(f"integer expression `{text}` has an unreadable number literal: {exc}") from exc
        if tok not in constants:
            raise LaunchRejected(f"integer expression `{text}` uses unknown name `{tok}`")
        return int(constants[tok])

    value = parse_expr()
    if pos != len(tokens):
        raise LaunchRejected(f"integer expression `{text}` has trailing input")
    return value


def _tokenize(text: str) -> list[str]:
    """@brief 把整数表达式切成数字、名字与运算符。

    @param text 表达式文本。
    @return 记号列表。
    @raises LaunchRejected 当出现非法字符。
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "+-*/()":
            out.append(ch)
            i += 1
            continue
        if ch.isdigit():
            j = i + 1
            while j < n and text[j].isdigit():
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            out.append(text[i:j])
            i = j
            continue
        raise LaunchRejected(f"integer expression `{text}` has an illegal character")
    return out
