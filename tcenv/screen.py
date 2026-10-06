"""Static pre-screen that runs *before* a candidate reaches the GPU.

A kernel that never terminates monopolizes the GPU, which on Apple silicon also drives the display: the
window server stops being serviced and the machine can freeze for tens of seconds or need a hard reset.
Nothing can preempt a running Metal kernel, so the cheap, reliable defence is to refuse the obviously
non-terminating constructs up front and to bound every launch with an external kill timer (see evaluate.py).
"""

from __future__ import annotations

import re

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

_RULES = [
    (re.compile(r"\bwhile\s*\(\s*(true|1|1u|1U)\s*\)"), "`while (true)` / `while (1)` is an unconditional loop"),
    (re.compile(r"\bfor\s*\([^;()]*;\s*;[^()]*\)"), "`for (...; ; ...)` has no loop condition"),
]
_INCLUDE = re.compile(r"^\s*#\s*include\s*(.*)$", re.MULTILINE)
_ALLOWED_INCLUDE = re.compile(r"^<metal_[a-z_0-9]+>\s*$")


def strip_comments(src: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), src))


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
    for m in _INCLUDE.finditer(code):
        target = m.group(1).strip()
        if not _ALLOWED_INCLUDE.match(target):
            problems.append(f"`#include {target}` is not allowed; only Metal standard headers such as <metal_stdlib> may be included.")
    return problems
