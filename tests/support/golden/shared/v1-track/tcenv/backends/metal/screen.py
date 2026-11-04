"""Static pre-screen that runs *before* a candidate reaches the GPU.

A kernel that never terminates monopolizes the GPU, which on Apple silicon also drives the display: the
window server stops being serviced and the machine can freeze for tens of seconds or need a hard reset.
Nothing can preempt a running Metal kernel, so the cheap, reliable defence is to refuse the obviously
non-terminating constructs up front and to bound every launch with an external kill timer (see evaluate.py).
"""
# 判卷流水线的第 0 关，在 compile 之前，也在子进程和 GPU 之前。
# evaluate() 先调用这里；被拒绝则 status 为 rejected，编译器不会看到这份源码。
# 这里拒绝：while(true) / while(1)（多包一层括号或写成 true == true 也算）、条件为空的 for、以及 <metal_...> 以外的 #include。
# 有上界的 for、while(n) 不在这里判，留给单次启动的杀死时器。
# 要在编译前拦，是因为编译器接受死循环，而 Metal 核一旦在 GPU 上跑起来，进程内部抢占不了；这块 GPU 还负责显示。外面的杀死时器只能在数秒后补救。

from __future__ import annotations

import ast
import re

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
# A string literal is data, not code: `"for (;;)"` reaches a buffer, not the GPU's loop counter.
_STRING = re.compile(r'"(?:\\.|[^"\\])*"')
# `goto` can form a loop the condition scan cannot bound. Kernels that use it are rare enough that
# refusing it up front is cheaper than a runaway launch.
_GOTO = re.compile(r"\bgoto\b")

# 条件外面再套一层括号、或写成 true == true，仍然是不会停的循环。空的 for 条件同样拒绝。
_WHILE = re.compile(r"\bwhile\s*\(")
_FOR = re.compile(r"\bfor\s*\(")
_TRUE = re.compile(r"(true|1u|1U|1)==\1")
# 只放行尖括号里的 Metal 标准头，例如 <metal_stdlib>。引号头和 <stdio.h> 这类都拒绝。
_INCLUDE = re.compile(r"^\s*#\s*include\s*(.*)$", re.MULTILINE)
_ALLOWED_INCLUDE = re.compile(r"^<metal_[a-z_0-9]+>\s*$")


# 规则搜的是会执行的文本。注释里写出的 while(true) 不会跑。块注释换成同等数量的换行，避免把上下两行粘成一行。
def strip_comments(src: str) -> str:
    """@brief 去掉源码里的行注释与块注释。

    @param src 原始源码。
    @return 去掉注释的文本，块注释换成等量换行以保留行号。
    """
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), src))


def strip_strings(src: str) -> str:
    """@brief 把字符串字面量换成空串，避免把数据当代码扫描。

    @param src 原始源码。
    @return 替换后的源码。

    Remove string literals before the loop scan. Strings first, then comments: a `//` inside a
    string is not a comment, and a quote inside a comment is not a string.
    """
    return _STRING.sub('""', src)


def _matching_paren(code: str, open_at: int) -> int | None:
    """@brief 找到与 ``open_at`` 处左括号配对的右括号。

    @param code 待扫描文本。
    @param open_at 左括号下标。
    @return 右括号下标；括号未闭合时为 None。

    Index of the ``)`` that closes the ``(`` at ``open_at``, or ``None`` when the header is cut off.
    """
    depth = 0
    for i in range(open_at, len(code)):
        if code[i] == "(":
            depth += 1
        elif code[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _outer_parens(text: str) -> bool:
    """@brief 判断整段文本是否被一对最外层括号完整包住。

    @param text 待判断文本。
    @return 是则 True。
    """
    if len(text) < 2 or text[0] != "(" or text[-1] != ")":
        return False
    return _matching_paren(text, 0) == len(text) - 1


def _unconditional(cond: str) -> bool:
    """@brief 判断循环条件是否恒为真。

    @param cond 括号内的条件文本。
    @return 恒真（空、``true``、``1`` 或同一常量自比）时为 True。

    True when ``cond`` cannot become false: empty, ``true`` / ``1``, or the same constant compared with itself.
    """
    text = cond.strip()
    while _outer_parens(text):
        text = text[1:-1].strip()
    compact = re.sub(r"\s+", "", text)
    if compact in {"", "true", "1", "1u", "1U"}:
        return True
    if _TRUE.fullmatch(compact) is not None:
        return True
    return _constant_true(text)


# Nodes a constant-folded loop condition may contain. Anything else (a call, an attribute, a name)
# makes the condition non-constant and is left to the compiler and the launch watchdog.
_CONST_NODES = (
    ast.Expression, ast.Constant, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.Load,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def _safe_pow(tree: ast.AST) -> bool:
    """A constant ``**`` is only evaluated when the result cannot explode in time or memory."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)):
            continue
        exponent, base = node.right, node.left
        if not (isinstance(exponent, ast.Constant) and isinstance(exponent.value, int) and not isinstance(exponent.value, bool) and -4096 <= exponent.value <= 64):
            return False
        if not (isinstance(base, ast.Constant) and isinstance(base.value, (int, float)) and not isinstance(base.value, bool) and abs(float(base.value)) <= 1e6):
            return False
    return True


def _constant_true(cond: str) -> bool:
    """True when ``cond`` is a constant expression that evaluates truthy.

    ``while (0 == 0)`` and ``while (2)`` never terminate, but the literal ``true``/``1`` patterns
    did not see them. Evaluation is restricted to literals and arithmetic/comparison operators;
    anything with a name or call is not constant here and is not rejected by this rule.
    """
    py = cond.replace("&&", " and ").replace("||", " or ")
    py = re.sub(r"!(?!=)", " not ", py)
    py = re.sub(r"\btrue\b", "True", py, flags=re.I)
    py = re.sub(r"\bfalse\b", "False", py, flags=re.I)
    py = re.sub(r"(\d)[uUlL]+\b", r"\1", py).strip()
    try:
        tree = ast.parse(py, mode="eval")
    except SyntaxError:
        return False
    if any(not isinstance(node, _CONST_NODES) for node in ast.walk(tree)):
        return False
    if not _safe_pow(tree):
        return False
    try:
        return bool(eval(compile(tree, "<loop-condition>", "eval"), {"__builtins__": {}}, {}))
    except Exception:  # noqa: BLE001 - division by zero and the like mean "not a safe constant"
        return False


def _split_semicolons(text: str) -> list[str]:
    """@brief 按顶层分号切分文本，忽略括号内的分号。

    @param text 待切分文本。
    @return 各段文本列表（保留空段）。
    """
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == ";" and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _unbounded(code: str) -> list[str]:
    """@brief 扫出条件恒真的 while/for 构造。

    @param code 已去注释与字符串的源码。
    @return 拒绝理由列表。
    """
    reasons = []
    for match in _WHILE.finditer(code):
        end = _matching_paren(code, match.end() - 1)
        if end is not None and _unconditional(code[match.end():end]):
            reasons.append("`while (true)` / `while (1)` is an unconditional loop")
    for match in _FOR.finditer(code):
        end = _matching_paren(code, match.end() - 1)
        if end is None:
            continue
        parts = _split_semicolons(code[match.end():end])
        if len(parts) >= 3 and _unconditional(parts[1]):
            reasons.append("`for (...; ; ...)` has no loop condition")
    return reasons


def screen_v2_source(src: str) -> list[str]:
    """@brief v2 的首道边界：只拒绝宿主路径/include 逃逸，不审查循环或 goto。

    v2 candidates own their loops, goto, persistent kernels and state machines; the bounded
    worker/watchdog owns runaway execution.  The static boundary still has to stop a source from
    asking the compiler to read an arbitrary host path, so only ``<metal_...>`` headers pass.

    @param src 原始 MSL 源码。
    @return 拒绝理由列表；空列表表示可以进入首次编译。
    """
    code = strip_comments(strip_strings(src))
    problems: list[str] = []
    for match in _INCLUDE.finditer(code):
        target = match.group(1).strip()
        if not _ALLOWED_INCLUDE.match(target):
            problems.append(
                f"`#include {target}` is not allowed in a v2 candidate; only Metal standard headers "
                "such as <metal_stdlib> may be included, and host paths are outside the candidate boundary"
            )
    return problems


# 返回拒绝理由；空列表表示可以交给编译器。能在这里看出来的就不要编译，更不要提交到 GPU。
def screen_source(src: str) -> list[str]:
    """@brief 静态预筛源码，列出拒绝上 GPU 的理由。

    @param src 原始 MSL 源码。
    @return 拒绝理由列表，空列表表示可以编译。

    Return reasons to refuse the source (empty list: fine to compile and run).
    """
    code = strip_comments(strip_strings(src))
    problems = []
    if _GOTO.search(code):
        problems.append(
            "`goto` is refused: a backward jump can form a loop the screen cannot bound, and a runaway GPU kernel "
            "can freeze the machine's display. Use a bounded `for` loop instead."
        )
    for why in _unbounded(code):
        problems.append(
            f"{why}. The harness refuses to launch kernels with unbounded loops (a runaway GPU kernel can freeze the "
            "machine's display). Rewrite it as a `for` loop with a bounded trip count."
        )
    # 整个 include 目标都要匹配 <metal_...>。多一个路径，或换成引号，就拒绝。
    for m in _INCLUDE.finditer(code):
        target = m.group(1).strip()
        if not _ALLOWED_INCLUDE.match(target):
            problems.append(f"`#include {target}` is not allowed; only Metal standard headers such as <metal_stdlib> may be included.")
    return problems
