"""Static preflight of a restricted C99 Host Program source (projection-time only).

This module never compiles anything: the real ``clang`` invocation happens only inside the bounded
worker's ``build()``.  Projection only has to answer "is this a candidate *source file* that can
be handed to that compiler at all": it refuses host includes, libc/process/file/network access,
inline assembly, function pointers, varargs and environment reads.  A real compile error (bad
types, wrong signature, missing return) is the compiler's job and is reported as a typed
``host_program_invalid`` with the compiler's own diagnostic text.

The rules are a curated identifier blacklist rather than a C parser.  They are deliberately
over-approximate where it is cheap (any ``#include`` is refused) and only approximate where a
real language front-end is needed (a cast that copies a function pointer), because native code is
not a memory sandbox: the security/runaway boundary is the external bounded worker, its process
watchdog and the absence of implicit host imports — not this text scan.
"""

from __future__ import annotations

import re
from typing import Iterator, Sequence

#: ``clang`` is invoked with ``-nostdlib`` and this list is refused by name as well, so a
#: candidate cannot rely on an ambient libc symbol being resolved by a future toolchain change.
FORBIDDEN_CALLS: tuple[str, ...] = (
    "malloc", "calloc", "realloc", "free", "aligned_alloc", "alloca",
    "printf", "fprintf", "sprintf", "snprintf", "vprintf", "vsnprintf", "puts", "putchar", "perror",
    "fopen", "fclose", "fread", "fwrite", "fgetc", "fputc", "fgets", "fputs", "fflush", "fseek",
    "open", "close", "read", "write", "lseek", "unlink", "remove", "rename", "mkdir",
    "time", "clock", "gettimeofday", "clock_gettime", "getenv", "setenv", "putenv", "unsetenv",
    "rand", "srand", "random", "srandom", "system", "popen", "pclose", "exec", "execl", "execv",
    "fork", "wait", "waitpid", "sleep", "usleep", "nanosleep", "kill", "raise", "signal",
    "setjmp", "longjmp", "sigsetjmp", "siglongjmp", "abort", "exit", "atexit", "atexit",
    "dlopen", "dlsym", "dlerror", "syscall", "mmap", "munmap", "mprotect",
    "getpid", "getppid", "getuid", "geteuid", "gethostname", "socket", "connect", "send", "recv",
)
#: Preprocessor/compiler escape hatches that are matched as substrings (they are not
#: identifiers).  ``#include`` is refused because the harness prepends the generated ABI header and
#: nothing else may enter the translation unit.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "__attribute__((import", "__attribute__((constructor", "__attribute__((destructor",
    "__attribute__((export", "wasi_snapshot_preview1", "__wasi", "emscripten",
)
#: Escape hatches that look like identifiers and are matched with word boundaries.
FORBIDDEN_IDENTIFIERS = frozenset(
    {
        "asm", "__asm", "__asm__", "_Complex", "_Atomic", "_Thread_local", "thread_local",
        "va_list", "va_start", "va_arg", "va_end", "__builtin_va",
        "__builtin_return_address", "__builtin_frame_address",
        "FILE", "stdin", "stdout", "stderr", "errno", "environ",
    }
)
_PREPROCESSOR = re.compile(r"#\s*(?:include|pragma|import)\b")

_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
#: Aliasing the workspaces parameter (``SfHostWorkspaces *w = workspaces``) hides which
#: workspace the source really writes from the name scan that decides allocation and the
#: host/device-zero ownership.  The contract is "use the entry parameter directly"; taking its
#: address or assigning to it is refused instead of silently depending on one spelling.
_WORKSPACE_ALIAS = re.compile(r"\bSfHostWorkspaces\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*=")
_WORKSPACE_ASSIGN = re.compile(r"\bworkspaces\s*=(?!=)")
_WORKSPACE_ADDRESS = re.compile(r"&\s*workspaces\b(?!\s*(?:->|\.))")
#: A function-pointer declaration/call: ``(*name)(`` / ``(*name[``.
_FUNCTION_POINTER = re.compile(r"\(\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*(?:\[[^\]]*\])?\s*\)\s*\(")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NATIVE_INTERNAL = re.compile(r"\bsf_native_[A-Za-z0-9_]*\b")


def strip_comments_and_literals(source: str) -> tuple[str, list[int]]:
    """@brief 去掉注释与字符串/字符字面量，保留换行以维持行号。

    @param source 原始源码。
    @return ``(stripped, line_of_index)``；第二项把 stripped 的下标映射回原行号。
    """
    out: list[str] = []
    lines: list[int] = []
    index = 0
    line = 1
    length = len(source)
    state = "code"
    while index < length:
        char = source[index]
        nxt = source[index + 1] if index + 1 < length else ""
        if char == "\n":
            line += 1
        if state == "code":
            if char == "/" and nxt == "/":
                state = "line_comment"
                out.append(" ")
                lines.append(line)
                index += 2
                continue
            if char == "/" and nxt == "*":
                state = "block_comment"
                out.append(" ")
                lines.append(line)
                index += 2
                continue
            if char == '"':
                state = "string"
                out.append(" ")
                lines.append(line)
                index += 1
                continue
            if char == "'":
                state = "char"
                out.append(" ")
                lines.append(line)
                index += 1
                continue
            out.append(char)
            lines.append(line)
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                state = "code"
                out.append("\n")
                lines.append(line)
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and nxt == "/":
                state = "code"
                index += 2
                continue
            if char == "\n":
                out.append("\n")
                lines.append(line)
            index += 1
            continue
        # string/char literals
        if char == "\\":
            index += 2
            continue
        if (state == "string" and char == '"') or (state == "char" and char == "'"):
            state = "code"
            index += 1
            continue
        if char == "\n":
            # An unterminated literal is a compiler error; do not let it swallow the rest.
            state = "code"
            out.append("\n")
            lines.append(line)
        index += 1
    return "".join(out), lines


def _line_at(lines: Sequence[int], index: int) -> int:
    if not lines:
        return 1
    return lines[min(max(index, 0), len(lines) - 1)]


def static_precheck(source: str) -> tuple[str, ...]:
    """@brief 投影期的危险语法/越界能力预检。

    @param source 候选 C99 源码。
    @return 问题列表（空表示没有静态拒绝项）。
    """
    if not isinstance(source, str):
        return ("host program source must be text",)
    problems: list[str] = []
    stripped, lines = strip_comments_and_literals(source)
    if not stripped.strip():
        return ("host program source is empty",)
    if "\x00" in source:
        problems.append("host program source contains a NUL byte")
    lowered = stripped
    for match in _PREPROCESSOR.finditer(lowered):
        problems.append(
            f"line {_line_at(lines, match.start())}: `{match.group(0).strip()}` is not allowed; the harness "
            "prepends the generated host_contract.h"
        )
    for token in FORBIDDEN_SUBSTRINGS:
        position = lowered.find(token)
        if position >= 0:
            problems.append(
                f"line {_line_at(lines, position)}: `{token}` is not allowed in a restricted C99 Host Program"
            )
    for match in _CALL.finditer(lowered):
        name = match.group(1)
        if name in FORBIDDEN_CALLS:
            problems.append(
                f"line {_line_at(lines, match.start())}: `{name}(` is a forbidden libc/process/file/network call"
            )
    for match in _IDENTIFIER.finditer(lowered):
        name = match.group(0)
        if name in FORBIDDEN_IDENTIFIERS:
            problems.append(
                f"line {_line_at(lines, match.start())}: `{name}` is a forbidden escape hatch/host object"
            )
    for match in _NATIVE_INTERNAL.finditer(lowered):
        problems.append(
            f"line {_line_at(lines, match.start())}: `{match.group(0)}` is a harness-internal native "
            "entry, not candidate API; use only the generated SfHost* fields and sf_input_read_* callbacks"
        )
    for match in _FUNCTION_POINTER.finditer(lowered):
        problems.append(
            f"line {_line_at(lines, match.start())}: function pointers are not allowed in a restricted C99 Host Program"
        )
    for pattern, why in (
        (_WORKSPACE_ALIAS, "a local SfHostWorkspaces alias hides which workspace the source writes"),
        (_WORKSPACE_ASSIGN, "assigning to the workspaces parameter hides which workspace the source writes"),
        (_WORKSPACE_ADDRESS, "taking the address of workspaces hides which workspace the source writes"),
    ):
        match = pattern.search(lowered)
        if match:
            problems.append(
                f"line {_line_at(lines, match.start())}: `{match.group(0).strip()}` is not allowed; {why}. "
                "Use `workspaces->name` directly so allocation and host/device-zero ownership stay auditable"
            )
    # Stable, de-duplicated diagnostics: the same source must produce the same refusal.
    return tuple(dict.fromkeys(problems))


def require_static_precheck(source: str) -> None:
    """@brief 预检并抛出带位置的问题；供 projection/quick 路径复用。

    @param source 候选 C99 源码。
    @raises ValueError 至少一条静态问题。
    """
    problems = static_precheck(source)
    if problems:
        raise ValueError("; ".join(problems[:6]))


def iter_forbidden_calls(source: str) -> Iterator[tuple[str, int]]:
    """@brief 迭代被拒绝的调用（诊断/测试用，含行号）。

    @param source 源码。
    @return ``(name, line)`` 迭代器。
    """
    stripped, lines = strip_comments_and_literals(source)
    for match in _CALL.finditer(stripped):
        if match.group(1) in FORBIDDEN_CALLS:
            yield match.group(1), _line_at(lines, match.start())


__all__ = [
    "FORBIDDEN_CALLS",
    "FORBIDDEN_IDENTIFIERS",
    "FORBIDDEN_SUBSTRINGS",
    "iter_forbidden_calls",
    "require_static_precheck",
    "static_precheck",
    "strip_comments_and_literals",
]
