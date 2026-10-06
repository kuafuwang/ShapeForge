"""Static pre-screen that runs *before* a candidate reaches the GPU.

A kernel that never terminates monopolizes the GPU, which on Apple silicon also drives the display: the
window server stops being serviced and the machine can freeze for tens of seconds or need a hard reset.
Nothing can preempt a running Metal kernel, so the cheap, reliable defence is to refuse the obviously
non-terminating constructs up front and to bound every launch with an external kill timer (see evaluate.py).
"""
# 判卷流水线的第 0 关，在 compile 之前，也在子进程和 GPU 之前。
# evaluate() 先调用这里；被拒绝则 status 为 rejected，编译器不会看到这份源码。
# 这里拒绝三类东西：while(true) / while(1) / while(1u) / while(1U)、没有条件的 for(;;)、以及 <metal_...> 以外的 #include。
# 有上界的 for、while(n) 不在这里判，留给单次启动的杀死时器。
# 要在编译前拦，是因为编译器接受死循环，而 Metal 核一旦在 GPU 上跑起来，进程内部抢占不了；这块 GPU 还负责显示。外面的杀死时器只能在数秒后补救。

from __future__ import annotations

import re

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

# 只拦两种一眼就不会停的循环：while(true) / while(1) / while(1u) / while(1U)，以及条件为空的 for(;;)。
_RULES = [
    (re.compile(r"\bwhile\s*\(\s*(true|1|1u|1U)\s*\)"), "`while (true)` / `while (1)` is an unconditional loop"),
    (re.compile(r"\bfor\s*\([^;()]*;\s*;[^()]*\)"), "`for (...; ; ...)` has no loop condition"),
]
# 只放行尖括号里的 Metal 标准头，例如 <metal_stdlib>。引号头和 <stdio.h> 这类都拒绝。
_INCLUDE = re.compile(r"^\s*#\s*include\s*(.*)$", re.MULTILINE)
_ALLOWED_INCLUDE = re.compile(r"^<metal_[a-z_0-9]+>\s*$")


# 规则搜的是会执行的文本。注释里写出的 while(true) 不会跑。块注释换成同等数量的换行，避免把上下两行粘成一行。
def strip_comments(src: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), src))


# 返回拒绝理由；空列表表示可以交给编译器。能在这里看出来的就不要编译，更不要提交到 GPU。
def screen_source(src: str) -> list[str]:
    """Return reasons to refuse the source (empty list: fine to compile and run)."""
    code = strip_comments(src)
    problems = []
    for rx, why in _RULES:
        if rx.search(code):
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
