"""Which hardware features does a generated Metal kernel use? (static detector core, no GPU dispatch)

The core of the hardware-feature detector, importable from the search loop (``taic.hwfeedback``) and from the report CLI
(``experiments/hw_features.py``, which adds run discovery, the validation set, win-rate aggregation and REPORT.md on top of it).

The paper attributes the speed of LLM-lowered kernels to hardware-specific features (Tensor Core MMA, shared-memory staging and
swizzling, cp.async / TMA pipelines, warp shuffles, packed / vectorized accesses, ...; see ``docs/hardware_features.md`` table 1). This module
detects their Apple-GPU analogues at three evidence levels that are never mixed:

SOURCE   comments and string literals are blanked first, function-like macros are expanded, helper functions that the kernel calls are inlined
         (argument names substituted) and constants (contract header, ``constant constexpr`` definitions) are evaluated, so ``simdgroup_matrix`` in a
         comment does not count and tile counts / threadgroup bytes are real numbers.
COMPILED the candidate is assembled with the task's contract header and compiled with the Metal toolchain to AIR text (``metal -S -emit-llvm``,
         the same front end and MSL version the harness uses; nothing is loaded on the GPU and nothing is dispatched). The IR confirms that the
         matrix intrinsics, simd operations, barriers, vector loads / stores and threadgroup allocations survived the optimizer. Limits: the AIR is
         what the GPU back end consumes, not the final machine code; call-site counts are static (a loop body counts once).
MEASURED numbers the harness already stored for the candidate (``report.json``: roofline, pipeline reflection, speedup) compared with the
         ceiling of the same kernel *without* the feature: the task's own naive scalar kernel measured by the same protocol (fallback: the
         scalar-FMA ceiling as a fraction of the matrix peak for compute-bound tasks, 87 % of the measured bandwidth for memory-bound ones).

Verdicts per feature: ``used and effective`` | ``present but not effective`` | ``expected but missing`` | ``not applicable`` (plus
``present, not measured`` when the candidate has no benchmark). Presence is verified; the *speedup attributable to one feature* is not: that
needs an ablation (see ``ablation_plan``), so features other than the primary one of a task family carry ``attribution: joint``.

Only imports from ``tcenv`` / ``tasks`` happen lazily and read-only (contract header, task constants); nothing is written outside the cache directory
that the caller passes to ``compile_air``.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO = Path(__file__).resolve().parent.parent

# ======================================================================================================================================
# 1. lexical helpers
# ======================================================================================================================================


def strip_comments_and_strings(src: str) -> str:
    """Blank ``//`` and ``/* */`` comments and the inside of string / char literals (length and newlines are preserved)."""
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"' and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j, n - 1)
            seg = src[i : j + 1]
            out.append('"' + re.sub(r"[^\n]", " ", seg[1:-1]) + ('"' if seg.endswith('"') and len(seg) > 1 else ""))
            i = j + 1
        elif c == "'":
            m = re.match(r"'(?:\\.[^'\n]{0,3}|[^'\\\n])'", src[i : i + 8])
            if m:
                out.append("'" + " " * (m.end() - 2) + "'")
                i += m.end()
            else:  # digit separator such as 1'000
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def normalize(code: str) -> str:
    """Drop the ``metal::`` qualifier (same length, so offsets still match) so ``metal::threadgroup_barrier`` is seen as ``threadgroup_barrier``."""
    return re.sub(r"\bmetal\s*::", lambda m: " " * len(m.group(0)), code)


def match_close(s: str, i: int, open_ch: str, close_ch: str) -> int:
    """Index of the bracket closing the one at ``s[i]`` (-1 if unbalanced)."""
    depth = 0
    for j in range(i, len(s)):
        ch = s[j]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return j
    return -1


def split_top(s: str, sep: str = ",") -> list[str]:
    """Split ``s`` at ``sep`` outside (), [], {} and simple template brackets ``<type, 8, 8>``."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "<" and i > 0 and (s[i - 1].isalnum() or s[i - 1] == "_"):
            k = s.find(">", i)
            if k > 0 and re.fullmatch(r"[\w\s,:*&<>]*", s[i + 1 : k]) and depth >= 0:
                cur.append(s[i : k + 1])
                i = k + 1
                continue
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, max(pos, 0)) + 1


_ROOT_RX = re.compile(r"^[\s(&*]*(?:thread|device|threadgroup|constant|const|auto)?\s*([A-Za-z_]\w*)")


def root_name(expr: str) -> str | None:
    m = re.match(r"^[\s(&*]*([A-Za-z_]\w*)", expr)
    return m.group(1) if m else None


# ======================================================================================================================================
# 2. constants and sizes
# ======================================================================================================================================

SCALAR_BYTES = {"char": 1, "uchar": 1, "bool": 1, "short": 2, "ushort": 2, "half": 2, "bfloat": 2, "int": 4, "uint": 4, "float": 4, "long": 8, "ulong": 8, "size_t": 8,
                "atomic_uint": 4, "atomic_int": 4, "atomic_float": 4, "atomic_bool": 4}
_VEC_RX = re.compile(r"(packed_)?(char|uchar|short|ushort|half|bfloat|int|uint|float|long|ulong)([234])?$")
CAST_WORDS = r"(?:u?int|ulong|long|ushort|short|size_t|uchar|char|float|half|bool)"


def type_bytes(t: str) -> tuple[int, int, int] | None:
    """(bytes of one element, lanes, bytes of the whole type) for MSL scalar / vector types; None if unknown."""
    t = t.strip()
    t = re.sub(r"^(?:const|constant|device|threadgroup|thread)\s+", "", t)
    if t in SCALAR_BYTES:
        return SCALAR_BYTES[t], 1, SCALAR_BYTES[t]
    m = _VEC_RX.match(t)
    if m:
        eb = SCALAR_BYTES[m.group(2)]
        lanes = int(m.group(3) or 1)
        return eb, lanes, eb * (4 if (lanes == 3 and not m.group(1)) else lanes)
    m = re.fullmatch(r"simdgroup_(half|float|bfloat)8x8", t)
    if m:
        return SCALAR_BYTES[m.group(1)], 64, SCALAR_BYTES[m.group(1)] * 64
    m = re.fullmatch(r"vec\s*<\s*(\w+)\s*,\s*(\d+)\s*>", t)
    if m and m.group(1) in SCALAR_BYTES:
        return SCALAR_BYTES[m.group(1)], int(m.group(2)), SCALAR_BYTES[m.group(1)] * int(m.group(2))
    return None


class _Eval(ast.NodeVisitor):
    def __init__(self, names: dict[str, float]):
        self.names = names

    def visit_Expression(self, node):  # noqa: N802
        return self.visit(node.body)

    def visit_Constant(self, node):  # noqa: N802
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("const")

    def visit_Name(self, node):  # noqa: N802
        if node.id in self.names:
            return self.names[node.id]
        raise KeyError(node.id)

    def visit_UnaryOp(self, node):  # noqa: N802
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Invert):
            return ~int(v)
        raise ValueError("unary")

    def visit_BinOp(self, node):  # noqa: N802
        a, b = self.visit(node.left), self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            if isinstance(a, int) and isinstance(b, int):
                return int(a / b) if b else 0
            return a / b
        if isinstance(op, ast.FloorDiv):
            return a // b
        if isinstance(op, ast.Mod):
            return int(math.fmod(a, b))
        if isinstance(op, ast.LShift):
            return int(a) << int(b)
        if isinstance(op, ast.RShift):
            return int(a) >> int(b)
        if isinstance(op, ast.BitAnd):
            return int(a) & int(b)
        if isinstance(op, ast.BitOr):
            return int(a) | int(b)
        if isinstance(op, ast.BitXor):
            return int(a) ^ int(b)
        raise ValueError("binop")

    def visit_Call(self, node):  # noqa: N802
        fn = node.func.id if isinstance(node.func, ast.Name) else ""
        args = [self.visit(a) for a in node.args]
        if fn in ("min", "fmin"):
            return min(args)
        if fn in ("max", "fmax"):
            return max(args)
        if fn in ("int", "uint", "ulong", "long", "ushort", "short", "size_t", "uchar", "char"):
            return int(args[0])
        if fn in ("float", "half"):
            return float(args[0])
        raise ValueError("call")

    def generic_visit(self, node):
        raise ValueError(type(node).__name__)


def eval_expr(expr: str, consts: dict[str, float]) -> float | int | None:
    """Evaluate a C-like constant expression (+ - * / % << >> & | ^ ~, min/max, casts, sizeof, literals); None if any name is unknown."""
    e = expr.strip().rstrip(";")
    if not e or "?" in e:
        return None
    e = re.sub(r"sizeof\s*\(\s*([\w ]+?)\s*\)", lambda m: str((type_bytes(m.group(1)) or (0, 0, 0))[2]), e)
    e = re.sub(rf"\(\s*{CAST_WORDS}\s*\)", "", e)
    e = re.sub(r"\b0[xX]([0-9a-fA-F]+)[uUlL]*\b", lambda m: str(int(m.group(1), 16)), e)
    e = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", e)
    e = re.sub(r"\b(\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+|\d+[eE][-+]?\d+)[fFhH]?\b", r"\1", e)
    e = re.sub(r"\b(\d+)[fFhH]\b", r"\1.0", e)
    e = e.replace("::", "_")
    try:
        tree = ast.parse(e, mode="eval")
        v = _Eval(consts).visit(tree)
    except (SyntaxError, KeyError, ValueError, ZeroDivisionError, TypeError, OverflowError):
        return None
    if isinstance(v, float) and v.is_integer() and abs(v) < 2**53 and "." not in e:
        return int(v)
    return v


_CONST_DECL_RX = re.compile(
    r"(?:\bstatic\s+)?(?:\bconstant\s+constexpr|\bconstexpr\s+constant|\bconstexpr|\bconstant|\bconst)\s+(?:\bconstant\s+)?"
    r"(?:u?int|ulong|long|ushort|short|size_t|uchar|char|float|half|bool)\s+([^;{}()]*(?:\([^;{}]*\)[^;{}()]*)*);"
)


def collect_constants(code: str, base: dict[str, float] | None = None) -> dict[str, float]:
    """Fixed point over every ``constexpr`` / ``constant`` / ``#define`` integer or float definition that evaluates with the known names."""
    consts: dict[str, float] = dict(base or {})
    pending: list[tuple[str, str]] = []
    for m in _CONST_DECL_RX.finditer(code):
        for decl in split_top(m.group(1)):
            mm = re.match(r"([A-Za-z_]\w*)\s*(?:=\s*(.+)|\{(.+)\})\s*$", decl, re.S)
            if mm:
                pending.append((mm.group(1), (mm.group(2) or mm.group(3)).strip()))
    for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+([^\n]+)$", code, re.M):
        pending.append((m.group(1), m.group(2).strip()))
    for _ in range(8):
        progress = False
        for name, expr in pending:
            if name in consts and name not in (base or {}):
                continue
            v = eval_expr(expr, consts)
            if v is not None and consts.get(name) != v:
                consts[name] = v
                progress = True
        if not progress:
            break
    return consts


# ======================================================================================================================================
# 3. structure: functions, loops, statements, helper inlining
# ======================================================================================================================================


@dataclass
class Func:
    name: str
    header: str
    params: str
    body: str
    is_kernel: bool


_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "else", "do", "catch", "static_assert", "decltype", "alignas"}


def parse_functions(code: str) -> list[Func]:
    depth = [0] * (len(code) + 1)
    d = 0
    for i, ch in enumerate(code):
        depth[i] = d
        if ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
    funcs: list[Func] = []
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", code):
        if depth[m.start()] != 0 or m.group(1) in _KEYWORDS:
            continue
        po = m.end() - 1
        pc = match_close(code, po, "(", ")")
        if pc < 0:
            continue
        j = pc + 1
        while True:
            while j < len(code) and code[j].isspace():
                j += 1
            if code.startswith("const", j):
                j += 5
                continue
            if code.startswith("[[", j):
                k = code.find("]]", j)
                j = k + 2 if k > 0 else len(code)
                continue
            break
        if j >= len(code) or code[j] != "{":
            continue
        pb = match_close(code, j, "{", "}")
        if pb < 0:
            continue
        hs = max(code.rfind(";", 0, m.start()), code.rfind("}", 0, m.start()), -1) + 1
        header = code[hs : m.start()]
        funcs.append(Func(m.group(1), header, code[po + 1 : pc], code[j + 1 : pb], bool(re.search(r"\bkernel\b", header))))
    return funcs


def _stmt_end(code: str, i: int) -> int:
    """End (exclusive) of the statement starting at ``i`` (a braced block, a control statement or an expression statement)."""
    n = len(code)
    while i < n and code[i].isspace():
        i += 1
    if i >= n:
        return n
    if code[i] == "{":
        j = match_close(code, i, "{", "}")
        return n if j < 0 else j + 1
    m = re.match(r"(for|while|if|switch)\s*\(", code[i:])
    if m:
        pc = match_close(code, i + m.end() - 1, "(", ")")
        if pc < 0:
            return n
        e = _stmt_end(code, pc + 1)
        if m.group(1) == "if":
            k = e
            while k < n and code[k].isspace():
                k += 1
            if code.startswith("else", k) and not (k + 4 < n and (code[k + 4].isalnum() or code[k + 4] == "_")):
                return _stmt_end(code, k + 4)
        return e
    if code.startswith("#pragma", i):
        k = code.find("\n", i)
        return _stmt_end(code, n if k < 0 else k + 1)
    depth = 0
    for j in range(i, n):
        ch = code[j]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == ";" and depth <= 0:
            return j + 1
    return n


def find_loops(code: str) -> list[tuple[int, int, int]]:
    """(start, body_start, end) of every ``for`` / ``while`` loop in ``code``."""
    loops = []
    for m in re.finditer(r"\b(for|while)\s*\(", code):
        po = m.end() - 1
        pc = match_close(code, po, "(", ")")
        if pc < 0:
            continue
        loops.append((m.start(), pc + 1, _stmt_end(code, pc + 1)))
    return loops


def iter_statements(code: str) -> Iterator[tuple[int, str]]:
    i, n = 0, len(code)
    while i < n:
        while i < n and (code[i].isspace() or code[i] in "{};"):
            i += 1
        if i >= n:
            break
        if code[i] == "#":
            j = code.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        m = re.match(r"(?:(?:for|while|if|switch)\s*\(|else\b|do\b)", code[i:])
        if m:
            if m.group(0).endswith("("):
                pc = match_close(code, i + m.end() - 1, "(", ")")
                i = n if pc < 0 else pc + 1
            else:
                i += m.end()
            continue
        j, depth = i, 0
        while j < n:
            ch = code[j]
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif depth <= 0 and ch in ";{}":
                break
            j += 1
        yield i, code[i:j]
        i = j + 1


def _assign_split(st: str) -> tuple[str, str] | None:
    """(lhs, rhs) of the first top-level assignment (``=`` or a compound ``op=``) of a statement; None for non-assignments."""
    depth = 0
    for i, ch in enumerate(st):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            nxt = st[i + 1] if i + 1 < len(st) else ""
            prev = st[i - 1] if i > 0 else ""
            if nxt == "=" or prev in "=!":
                continue
            if prev in "<>" and not (st[max(0, i - 2) : i] in ("<<", ">>")):
                continue
            lhs = st[:i].rstrip()
            while lhs and lhs[-1] in "+-*/|&^%<>":
                lhs = lhs[:-1].rstrip()
            return lhs, st[i + 1 :].strip()
    return None


def _param_names(params: str) -> list[str]:
    names = []
    for p in split_top(params):
        p = re.sub(r"\[\[.*?\]\]", "", p).strip()
        m = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])*\s*$", p)
        names.append(m.group(1) if m else "")
    return names


def expand_calls(body: str, helpers: dict[str, Func], depth: int = 4) -> str:
    """Inline statement-level calls of helper functions (formal parameters replaced by the actual argument text).

    Only the *order* of events (load, MMA, store, barrier) matters for the analyses that use the result, so no renaming of locals is done."""
    if depth <= 0 or not helpers:
        return body
    pat = re.compile(r"\b(" + "|".join(map(re.escape, helpers)) + r")\s*\(")
    out = body
    for m in reversed(list(pat.finditer(body))):
        h = helpers[m.group(1)]
        pc = match_close(out, m.end() - 1, "(", ")") if out is body else match_close(body, m.end() - 1, "(", ")")
        if pc < 0:
            continue
        before = body[: m.start()].rstrip()
        after = body[pc + 1 :].lstrip()
        if not (after.startswith(";") and (not before or before[-1] in ";{}:)")):
            continue  # value-returning use inside an expression: not inlined
        args = split_top(body[m.end() : pc])
        names = _param_names(h.params)
        inner = expand_calls(h.body, {k: v for k, v in helpers.items() if k != h.name}, depth - 1)
        for nm, arg in zip(names, args):
            if nm:
                inner = re.sub(rf"\b{re.escape(nm)}\b", arg if re.fullmatch(r"[\w.]+", arg) else f"({arg})", inner)
        out = out[: m.start()] + "{ " + inner + " }" + out[pc + 1 :] if out is body else out
        body = out
    return body


# ======================================================================================================================================
# 4. source analysis
# ======================================================================================================================================

_SIMD_NAMES = ("sum", "max", "min", "product", "and", "or", "xor", "prefix_inclusive_sum", "prefix_exclusive_sum", "prefix_inclusive_product",
               "prefix_exclusive_product", "shuffle", "shuffle_xor", "shuffle_down", "shuffle_up", "shuffle_rotate_down", "shuffle_rotate_up",
               "broadcast", "broadcast_first", "ballot", "all", "any", "is_first")
_SIMD_REDUCE = {"sum", "max", "min", "product", "and", "or", "xor", "prefix_inclusive_sum", "prefix_exclusive_sum", "prefix_inclusive_product", "prefix_exclusive_product"}
_SIMD_SHUFFLE = {"shuffle", "shuffle_xor", "shuffle_down", "shuffle_up", "shuffle_rotate_down", "shuffle_rotate_up", "broadcast", "broadcast_first"}
_MATH_FNS = ("exp", "exp2", "exp10", "log", "log2", "log10", "sin", "cos", "tan", "tanh", "sinh", "cosh", "sqrt", "rsqrt", "fma", "pow", "erf",
             "divide", "recip", "sigmoid", "abs", "fabs", "max", "fmax", "min", "fmin", "clamp", "select", "saturate", "floor", "ceil", "round")
_EPI_MATH = {"exp", "exp2", "exp10", "log", "log2", "log10", "tanh", "sin", "cos", "sqrt", "rsqrt", "pow", "erf", "divide", "recip", "sigmoid"}


_DEF_RX = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)\(([^)\n]*)\)[ \t]*((?:[^\n\\]|\\[ \t]*\n|\\.)*)", re.M)


def expand_macros(code: str, depth: int = 3) -> str:
    """Expand function-like ``#define`` macros (kernels written with macros keep their MMA / loads inside the macro body)."""
    macros: dict[str, tuple[list[str], str]] = {}

    def grab(m: re.Match) -> str:
        params = [p.strip() for p in m.group(2).split(",") if p.strip()]
        macros[m.group(1)] = (params, re.sub(r"\\[ \t]*\n", " ", m.group(3)))
        return re.sub(r"[^\n]", " ", m.group(0))

    out = _DEF_RX.sub(grab, code)
    if not macros:
        return code
    pat = re.compile(r"\b(" + "|".join(map(re.escape, macros)) + r")\s*\(")
    for _ in range(depth):
        pieces: list[str] = []
        pos, changed = 0, False
        while True:
            m = pat.search(out, pos)
            if not m:
                break
            pc = match_close(out, m.end() - 1, "(", ")")
            if pc < 0:
                break
            params, body = macros[m.group(1)]
            args = split_top(out[m.end() : pc])
            if params and len(args) >= len(params):
                sub = dict(zip(params, args))
                body = re.sub(r"\b(" + "|".join(map(re.escape, params)) + r")\b", lambda mm: sub[mm.group(1)], body)
            pieces.append(out[pos : m.start()] + body)
            pos, changed = pc + 1, True
        pieces.append(out[pos:])
        out = "".join(pieces)
        if not changed:
            break
    return out


def _num(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) and v == int(v) else None


def _space_of(expr: str, tg_names: set[str], dev_names: set[str]) -> str:
    ids = set(re.findall(r"[A-Za-z_]\w*", expr))
    if ids & tg_names:
        return "threadgroup"
    if ids & dev_names:
        return "device"
    return "unknown"


def analyze_source(source: str, *, consts: dict[str, float] | None = None, kernel_name: str | None = None, kernel_names: list[str] | None = None) -> dict:
    """Source-level evidence (everything the text proves). ``consts`` are the contract constants (BLOCK_M, N, ...).

    ``kernel_names`` limits the scan to those kernels and the helpers they reach. An unused kernel in the
    same file does not count. Without it, the named kernel, or else the first kernel, is the scan.
    """
    raw_code = strip_comments_and_strings(source)
    code = expand_macros(normalize(raw_code))
    funcs = parse_functions(code)
    kernels = [f for f in funcs if f.is_kernel]
    by_name = {f.name: f for f in kernels}
    if kernel_names:
        chosen = [by_name[n] for n in dict.fromkeys(kernel_names) if n in by_name]
    elif kernel_name and kernel_name in by_name:
        chosen = [by_name[kernel_name]]
    else:
        chosen = kernels[:1]
    helpers = {f.name: f for f in funcs if not f.is_kernel}
    cst = collect_constants(code, consts)
    ev: dict[str, Any] = {
        "kernel": ",".join(f.name for f in chosen) if chosen else None,
        "n_kernels": len(kernels),
        "lines": source.count("\n") + 1,
    }
    if kernel_names:
        ev["launched_kernels"] = list(dict.fromkeys(kernel_names))
    if not chosen:
        ev["error"] = "no `kernel` function found"
        scan, kbody, params = code, code, ""
        used_helpers: list[str] = []
    else:
        params = "\n".join(f.params for f in chosen)
        kbody = "\n".join(f.body for f in chosen)
        reach, todo = set(), [kbody]
        while todo:
            txt = todo.pop()
            for h in helpers:
                if h not in reach and re.search(rf"\b{re.escape(h)}\s*\(", txt):
                    reach.add(h)
                    todo.append(helpers[h].body)
        used_helpers = sorted(reach)
        scan = params + "\n" + kbody + "\n" + "\n".join(helpers[h].params + "\n" + helpers[h].body for h in used_helpers)
        ev["helpers_used"] = used_helpers
        ev["helpers_unused"] = sorted(set(helpers) - reach)
    expanded = expand_calls(kbody, {h: helpers[h] for h in used_helpers}) if chosen else code
    ev["n_constants_evaluated"] = len(cst)

    # ---- address spaces -----------------------------------------------------------------------------------------------------------
    tg_arrays = []
    for m in re.finditer(r"\bthreadgroup\s+(?:const\s+)?([A-Za-z_][\w:]*(?:\s*<[^>;]*>)?)\s+([A-Za-z_]\w*)\s*((?:\[[^\]]+\])*)\s*(?=[;=\{\[]|__attribute__|\[\[)", scan):
        ty, name, dims = m.group(1).strip(), m.group(2), re.findall(r"\[([^\]]+)\]", m.group(3))
        sizes = [eval_expr(d, cst) for d in dims]
        elems = math.prod(int(s) for s in sizes) if sizes and all(s is not None for s in sizes) else (1 if not dims else None)
        tb = type_bytes(ty)
        tg_arrays.append({"name": name, "type": ty, "dims": dims, "sizes": [None if s is None else int(s) for s in sizes], "elems": elems,
                          "bytes": (elems * tb[2]) if (elems is not None and tb) else None})
    tg_names = {a["name"] for a in tg_arrays}
    tg_ptrs = set(re.findall(r"\bthreadgroup\s+(?:const\s+)?[\w:<>, ]+?\s*\*\s*(?:(?:const|__restrict__|__restrict|restrict)\s+)*([A-Za-z_]\w*)", scan))
    dev_names: set[str] = set()
    typed_ptrs: dict[str, tuple[str, str]] = {}  # name -> (space, element type)
    for m in re.finditer(r"\b(device|constant|threadgroup)\s+(?:const\s+)?([A-Za-z_][\w:]*)\s*\*\s*(?:(?:const|__restrict__|__restrict|restrict)\s+)*([A-Za-z_]\w*)", scan):
        sp, ty, nm = m.groups()
        typed_ptrs[nm] = (sp, ty)
        if sp in ("device", "constant"):
            dev_names.add(nm)
        else:
            tg_ptrs.add(nm)
    for m in re.finditer(r"\b(?:const\s+)?auto\s+([A-Za-z_]\w*)\s*=\s*reinterpret_cast\s*<\s*(?:const\s+)?(device|constant|threadgroup)\s+(?:const\s+)?([\w:]+)\s*\*\s*>", scan):
        typed_ptrs[m.group(1)] = (m.group(2), m.group(3))
        (dev_names if m.group(2) != "threadgroup" else tg_ptrs).add(m.group(1))
    tg_all = tg_names | tg_ptrs
    # pointer aliases of device buffers: `device const half* qb = q + off;`
    for _ in range(2):
        for m in re.finditer(r"(?:device|constant)\s+(?:const\s+)?[\w:]+\s*\*\s*(?:(?:const|__restrict__|__restrict|restrict)\s+)*([A-Za-z_]\w*)\s*=\s*([^;]+);", scan):
            if set(re.findall(r"[A-Za-z_]\w*", m.group(2))) & dev_names:
                dev_names.add(m.group(1))
    in_bufs = [m.group(1) for m in re.finditer(r"device\s+const\s+[\w:]+\s*\*\s*(?:(?:__restrict__|__restrict|restrict)\s+)?([A-Za-z_]\w*)", params)]
    out_bufs = [m.group(1) for m in re.finditer(r"device\s+(?!const)[\w:]+\s*\*\s*(?:(?:__restrict__|__restrict|restrict)\s+)?([A-Za-z_]\w*)", params)]

    # ---- matrix unit ---------------------------------------------------------------------------------------------------------------
    mat_decl: dict[str, dict] = {}
    for m in re.finditer(r"\bsimdgroup_(half|float|bfloat)8x8\b|\bsimdgroup_matrix\s*<\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>", scan):
        rest = scan[m.end() :]
        if rest.lstrip().startswith("(") or scan[max(0, m.start() - 12) : m.start()].rstrip().endswith(("make_filled_", "<")):
            continue
        et = m.group(1) or m.group(2)
        end = len(rest)
        depth = 0
        for k, ch in enumerate(rest):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                if depth == 0:
                    end = k
                    break
                depth -= 1
            elif depth == 0 and ch in ";{":
                end = k
                break
        for decl in split_top(rest[:end]):
            dm = re.match(r"(?:thread\s+|threadgroup\s+|const\s+)*(&|\*)?\s*([A-Za-z_]\w*)\s*((?:\[[^\]]+\])*)", decl)
            if not dm or not dm.group(2):
                continue
            ref = bool(dm.group(1))
            dims = re.findall(r"\[([^\]]+)\]", dm.group(3))
            sizes = [eval_expr(d, cst) for d in dims]
            count = 0 if ref else (math.prod(int(s) for s in sizes) if all(s is not None for s in sizes) else None)
            if dm.group(2) not in mat_decl:
                mat_decl[dm.group(2)] = {"elem": et, "dims": [None if s is None else int(s) for s in sizes], "count": count, "ref": ref}
    acc_roots: list[str] = []
    opr_roots: list[str] = []
    mma_sites = 0
    for m in re.finditer(r"\bsimdgroup_multiply_accumulate\s*\(", scan):
        pc = match_close(scan, m.end() - 1, "(", ")")
        args = split_top(scan[m.end() : pc])
        if len(args) >= 4:
            mma_sites += 1
            acc_roots += [r for r in (root_name(args[0]), root_name(args[3])) if r]
            opr_roots += [r for r in (root_name(args[1]), root_name(args[2])) if r]
    for m in re.finditer(r"\bsimdgroup_multiply\s*\(", scan):
        pc = match_close(scan, m.end() - 1, "(", ")")
        args = split_top(scan[m.end() : pc])
        if len(args) >= 3:
            mma_sites += 1
            acc_roots += [r for r in (root_name(args[0]),) if r]
            opr_roots += [r for r in (root_name(args[1]), root_name(args[2])) if r]
    acc_set, opr_set = list(dict.fromkeys(acc_roots)), list(dict.fromkeys(opr_roots))
    acc_tiles = 0
    acc_unknown = False
    for r in acc_set:
        d = mat_decl.get(r)
        if d is None or d["count"] is None:
            acc_unknown = True
        elif not d["ref"]:
            acc_tiles += d["count"]
    # accumulator aliases through helper parameters (`thread simdgroup_float8x8& c`): the caller's declaration is the real one
    if acc_set and acc_tiles == 0:
        acc_tiles = sum(d["count"] or 0 for d in mat_decl.values() if not d["ref"] and d["elem"] == "float") or 0
    acc_types = sorted({mat_decl[r]["elem"] for r in acc_set if r in mat_decl})
    opr_types = sorted({mat_decl[r]["elem"] for r in opr_set if r in mat_decl})
    big = max((mat_decl[r] for r in acc_set if r in mat_decl and mat_decl[r]["count"]), key=lambda d: d["count"], default=None)
    loads, stores = [], []
    for fn, lst in (("simdgroup_load", loads), ("simdgroup_store", stores)):
        for m in re.finditer(rf"\b{fn}\s*\(", scan):
            pc = match_close(scan, m.end() - 1, "(", ")")
            args = split_top(scan[m.end() : pc])
            if len(args) >= 2:
                lst.append(_space_of(args[1], tg_all, dev_names))
    te_roots = [m.group(1) for m in re.finditer(r"([A-Za-z_]\w*)(?:\s*\[[^\]]*\])*\s*\.\s*thread_elements\s*\(", scan)]
    matrix = {
        "present": mma_sites > 0, "mma_sites": mma_sites, "acc_vars": acc_set, "operand_vars": opr_set, "acc_tiles": acc_tiles if not acc_unknown or acc_tiles else None,
        "acc_dims": big["dims"] if big else None, "acc_types": acc_types, "operand_types": opr_types, "loads_from": sorted(set(loads)), "load_sites": len(loads),
        "store_to": sorted(set(stores)), "store_sites": len(stores), "thread_elements_sites": len(te_roots),
        "thread_elements_on_operands": sorted({r for r in te_roots if r in opr_set and r not in acc_set}),
        "thread_elements_on_accumulators": sorted({r for r in te_roots if r in acc_set}),
        "tensor_api": bool(re.search(r"\b(?:cooperative_tensor|matmul2d|tensor_ops|mpp::|tensor\s*<\s*(?:device|threadgroup)|metal_tensor)", code)
                           or re.search(r"#\s*include\s*<(?:metal_tensor|metal_cooperative_tensor|MetalPerformancePrimitives[^>]*)>", raw_code)),
    }
    ev["matrix"] = matrix

    # ---- loops, statements, device / threadgroup accesses --------------------------------------------------------------------------
    loops_scan = find_loops(scan)
    loops_exp = find_loops(expanded)
    stmts = list(iter_statements(expanded))

    def in_loop(pos: int, loops=loops_exp) -> bool:
        return any(s <= pos < e for s, _, e in loops)

    barriers = []
    for m in re.finditer(r"\b(threadgroup_barrier|simdgroup_barrier)\s*\(([^)]*)\)", scan):
        barriers.append({"kind": m.group(1), "flags": sorted(set(re.findall(r"\bmem_(none|threadgroup_imageblock|threadgroup|device|texture|object_data)\b", m.group(2)))) or ["?"], "pos": m.start()})
    barr_exp = [(m.start(), m.group(1)) for m in re.finditer(r"\b(threadgroup_barrier|simdgroup_barrier)\s*\(", expanded)]
    tg_barriers = [b for b in barriers if b["kind"] == "threadgroup_barrier"]
    tg_barr_in_loop_exp = [p for p, k in barr_exp if k == "threadgroup_barrier" and in_loop(p)]
    # the main loop: the outermost loop holding the most matrix / barrier activity
    main = None
    if loops_exp:
        def activity(lp):
            t = expanded[lp[0] : lp[2]]
            return t.count("simdgroup_multiply") * 4 + t.count("threadgroup_barrier") * 2 + t.count("simdgroup_load")

        outer = [lp for lp in loops_exp if not any(o != lp and o[0] <= lp[0] and lp[2] <= o[2] for o in loops_exp)]
        best = max(outer, key=activity)
        main = best if activity(best) > 0 else None
        # step into a single wrapper loop (tile loop around the k loop) when it holds the same barriers
    barrier_main = len([1 for p, k in barr_exp if k == "threadgroup_barrier" and main and main[0] <= p < main[2]]) if main else None
    ev["sync"] = {"threadgroup_barrier_sites": len(tg_barriers), "simdgroup_barrier_sites": len(barriers) - len(tg_barriers),
                  "flags": sorted({f for b in barriers for f in b["flags"]}), "barrier_sites_in_loops_expanded": len(tg_barr_in_loop_exp),
                  "main_loop_barrier_sites": barrier_main, "has_loop": bool(loops_exp)}

    # device / threadgroup vector accesses -----------------------------------------------------------------------------------------
    sites: list[dict] = []

    def is_store_after(txt: str, j: int) -> bool:
        k = j
        while k < len(txt) and (txt[k].isspace() or txt[k] == ")"):  # `*(cast<T*>(p)) = v`: the closing parens belong to the deref
            k += 1
        if k < len(txt) and txt[k] == "[":  # cast(...)[i]
            e = match_close(txt, k, "[", "]")
            k = e + 1 if e > 0 else k
            while k < len(txt) and txt[k].isspace():
                k += 1
        return bool(re.match(r"(?:[-+*/|&^]|<<|>>)?=(?!=)", txt[k : k + 3]))

    def deref_before(txt: str, i: int) -> bool:
        k = i - 1
        while k >= 0 and (txt[k].isspace() or txt[k] == "("):
            k -= 1
        if k >= 0 and txt[k] == "*":
            p = k - 1
            while p >= 0 and txt[p].isspace():
                p -= 1
            if p < 0:
                return True
            w = re.search(r"([A-Za-z_]\w*)$", txt[: p + 1])
            if w and w.group(1) in ("return", "else", "case", "do"):
                return True
            return not (txt[p].isalnum() or txt[p] in "_)]")
        return False

    def add_site(sp: str, ty: str, kind: str, via: str, pos: int):
        tb = type_bytes(ty)
        if tb and sp in ("device", "constant", "threadgroup"):
            sites.append({"space": "device" if sp == "constant" else sp, "kind": kind, "bytes": tb[2], "type": ty, "via": via, "pos": pos})

    for m in re.finditer(r"reinterpret_cast\s*<\s*(?:const\s+)?(device|constant|threadgroup)?\s*(?:const\s+)?([\w:]+)\s*\*\s*>\s*\(", scan):
        pc = match_close(scan, m.end() - 1, "(", ")")
        if pc > 0 and deref_before(scan, m.start()) or (pc > 0 and scan[pc + 1 : pc + 3].lstrip().startswith("[")):
            add_site(m.group(1) or "thread", m.group(2), "store" if is_store_after(scan, pc + 1) else "load", "reinterpret_cast", m.start())
    for m in re.finditer(r"\(\s*(?:const\s+)?(device|constant|threadgroup)\s+(?:const\s+)?([\w:]+)\s*\*\s*\)", scan):
        j = m.end()
        while j < len(scan) and scan[j].isspace():
            j += 1
        if j < len(scan) and scan[j] == "&":
            j += 1
        k = j
        if k < len(scan) and scan[k] == "(":
            e = match_close(scan, k, "(", ")")
            k = (e + 1) if e > 0 else k
        else:
            mm = re.match(r"[A-Za-z_]\w*(?:\s*(?:\[[^\]]*\]|\.\w+|->\w+))*", scan[k:])
            if mm:
                k += mm.end()
        # `*(T*)&x[i]` or `((T*)x)[i]`
        if deref_before(scan, m.start()) or re.match(r"\s*\)\s*\[", scan[k : k + 8]):
            add_site(m.group(1), m.group(2), "store" if is_store_after(scan, k + (scan[k : k + 8].find(")") + 1 if scan[k : k + 8].lstrip().startswith(")") else 0)) else "load", "c-cast", m.start())
    # typed pointers are scoped per function (a helper's `pa` may be a threadgroup pointer while the kernel's `pa` is a device pointer)
    scopes = ([params + "\n" + kbody] + [helpers[h].params + "\n" + helpers[h].body for h in used_helpers]) if chosen else [scan]
    ptr_rx = re.compile(r"\b(device|constant|threadgroup)\s+(?:const\s+)?([A-Za-z_][\w:]*)\s*\*\s*(?:(?:const|__restrict__|__restrict|restrict)\s+)*([A-Za-z_]\w*)")
    auto_rx = re.compile(r"\b(?:const\s+)?auto\s+([A-Za-z_]\w*)\s*=\s*reinterpret_cast\s*<\s*(?:const\s+)?(device|constant|threadgroup)\s+(?:const\s+)?([\w:]+)\s*\*\s*>")
    for sc in scopes:
        local: dict[str, tuple[str, str]] = {}
        for m in ptr_rx.finditer(sc):
            local[m.group(3)] = (m.group(1), m.group(2))
        for m in auto_rx.finditer(sc):
            local[m.group(1)] = (m.group(2), m.group(3))
        for nm, (sp, ty) in local.items():
            for m in re.finditer(rf"(?<![\w.]){re.escape(nm)}\s*\[(?!\[)", sc):  # `x[i]`, not the attribute `x [[buffer(0)]]`
                e = match_close(sc, m.end() - 1, "[", "]")
                if e > 0:
                    add_site(sp, ty, "store" if is_store_after(sc, e + 1) else "load", "typed-pointer", m.start())
            for m in re.finditer(rf"(?<![\w.]){re.escape(nm)}\b(?!\s*\[(?!\[))", sc):
                if deref_before(sc, m.start()):  # `*p`, `*(p + i)`, `x = *p++`
                    add_site(sp, ty, "store" if is_store_after(sc, m.end()) else "load", "typed-pointer-deref", m.start())
    for m in re.finditer(r"__builtin_nontemporal_(store|load)\s*\(", scan):  # streaming accesses through the builtin
        pc = match_close(scan, m.end() - 1, "(", ")")
        cm = re.search(r"(device|threadgroup)\s+(?:const\s+)?([\w:]+)\s*\*", scan[m.end() : pc])
        if cm:
            add_site(cm.group(1), cm.group(2), m.group(1), "nontemporal", m.start())
    ev["memory"] = _summarize_sites(sites)

    # input read passes (static) ------------------------------------------------------------------------------------------------
    passes = {}
    for b in in_bufs:
        names = {b} | {a for a in dev_names if re.search(rf"\b{re.escape(a)}\s*=[^;]*\b{re.escape(b)}\b", scan)}
        loop_ids, straight = set(), 0
        for m in re.finditer(r"\b(" + "|".join(map(re.escape, names)) + r")\b", expanded):
            pre = expanded[max(0, m.start() - 60) : m.start()]
            if re.search(r"(?:device|constant)\s+(?:const\s+)?[\w:]+\s*\*\s*(?:const\s+)?$", pre):
                continue
            if re.match(r"\s*=[^=]", expanded[m.end() : m.end() + 3]) and re.search(r"(?:device|constant)[^;]*$", pre):
                continue
            outer = [lp for lp in loops_exp if lp[0] <= m.start() < lp[2] and not any(o != lp and o[0] <= lp[0] and lp[2] <= o[2] and o[0] <= m.start() < o[2] for o in loops_exp)]
            if outer:
                loop_ids.add(outer[0][0])
            else:
                straight += 1
        if names and (loop_ids or straight):
            passes[b] = len(loop_ids) if loop_ids else 1
    ev["global_read_passes"] = passes

    # ---- statement-level dataflow: loaded registers, threadgroup stores, prefetch ------------------------------------------------
    loaded_at: dict[str, int] = {}
    tg_store_uses: list[tuple[int, set[str]]] = []
    mma_pos = [m.start() for m in re.finditer(r"\bsimdgroup_multiply(?:_accumulate)?\s*\(", expanded)]
    dev_core = r"reinterpret_cast\s*<\s*(?:const\s+)?device|\(\s*(?:const\s+)?device\s+[\w:]+\s*\*\s*\)|\b(?:" + "|".join(map(re.escape, sorted(dev_names))) + r")\b\s*(?:\[|\+)" if dev_names else r"reinterpret_cast\s*<\s*(?:const\s+)?device|\(\s*(?:const\s+)?device\s+[\w:]+\s*\*\s*\)"
    dev_helpers = [h for h in used_helpers if re.search(dev_core, helpers[h].body)] if chosen else []
    dev_rx = re.compile(dev_core + (r"|\b(?:" + "|".join(map(re.escape, dev_helpers)) + r")\s*\(" if dev_helpers else ""))
    tg_store_rx = re.compile(r"^\s*\*\s*\(?\s*reinterpret_cast\s*<\s*(?:const\s+)?threadgroup\b|^\s*\*?\s*\(\s*(?:const\s+)?threadgroup\b")
    for pos, st in stmts:
        eq = _assign_split(st)
        if eq is None:
            continue
        lhs, rhs = eq
        lroot = root_name(lhs.lstrip("*& (")) or ""
        if tg_store_rx.match(st) or (lroot in tg_all and "[" in lhs):
            tg_store_uses.append((pos, set(re.findall(r"[A-Za-z_]\w*", rhs))))
            continue
        lvar = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]|\.\s*\w+)*\s*$", lhs)
        lname = lvar.group(1) if lvar else lroot
        if not lname or lname in dev_names or lname in tg_all or lhs.lstrip().startswith("*"):
            continue
        if dev_rx.search(rhs):
            loaded_at[lname] = pos
        else:
            src_vars = [v for v in re.findall(r"[A-Za-z_]\w*", rhs) if v in loaded_at]
            if src_vars and re.fullmatch(r"\s*[\w.\[\]]+\s*", rhs):
                loaded_at[lname] = pos  # register copy `k0 = nk0;`: the data is in k0 from here on
    prefetch = False
    prefetch_why = ""
    for spos, used in tg_store_uses:
        for v in used & set(loaded_at):
            lpos = loaded_at[v]
            lp = next((l for l in loops_exp if l[1] <= spos < l[2] and l[1] <= lpos < l[2]), None)
            if lp is None:
                continue
            carried = lpos > spos
            over_mma = any(lpos < mp < spos for mp in mma_pos if lp[1] <= mp < lp[2])
            if carried or over_mma:
                prefetch, prefetch_why = True, f"`{v}` is loaded from device memory {'after the threadgroup store that consumes it (one iteration ahead)' if carried else 'before the MMAs and stored to threadgroup memory after them'}"
                break
        if prefetch:
            break

    # ---- threadgroup memory -------------------------------------------------------------------------------------------------------
    total_tg = sum(a["bytes"] or 0 for a in tg_arrays)
    pad_ev: list[str] = []
    for nm, expr in re.findall(r"\b([A-Za-z_]\w*)\s*=\s*([^;,]+)", "\n".join(m.group(0) for m in _CONST_DECL_RX.finditer(code))):
        mm = re.fullmatch(r"\s*([A-Za-z_]\w*|\d+u?)\s*\+\s*([A-Za-z_]\w*|\d+u?)\s*", expr)
        if not mm:
            continue
        pad = eval_expr(mm.group(2), cst)  # `D + 8` and `D + PAD` (PAD a constant)
        used_as_stride = bool(tg_arrays) and bool(re.search(rf"\*\s*{re.escape(nm)}\b|\b{re.escape(nm)}\s*\*", scan))
        if isinstance(pad, int) and 1 <= pad <= 16 and (re.search(r"(?i)stride|pitch|^ld|^lda|^ldb|^ldc|pad|row", nm) or used_as_stride):
            pad_ev.append(f"{nm} = {mm.group(1)} + {mm.group(2)}")
    for nm, val in cst.items():
        if isinstance(val, int) and val > 1 and val % 2 == 1 and re.search(r"(?i)stride|pitch|^fp$|^ld|pad", nm) and nm in code:
            if re.search(rf"\*\s*{re.escape(nm)}\b|\b{re.escape(nm)}\s*\*", scan):
                pad_ev.append(f"{nm} = {val} (odd stride)")
    for m in re.finditer(r"\bthreadgroup\b[^;=]*\b\w+\s*\[[^\]]*\+\s*\d+u?\s*\]\s*;", scan):
        pad_ev.append(m.group(0)[:60].strip())
    # pad of a padded TG array declared like  As[BLOCK_M * (BLOCK_K + 8)]
    for a in tg_arrays:
        for d in a["dims"]:
            if re.search(r"\(\s*\w+\s*\+\s*\d{1,2}u?\s*\)|\*\s*\w+\s*\+\s*\d{1,2}u?\s*$", d.strip()):
                pad_ev.append(f"{a['name']}[{d.strip()}]")
    parity_arith = bool(re.search(r"\bswap\s*\(|\^\s*=?\s*1u?\b|\(\s*\w+\s*&\s*1u?\s*\)\s*(?:\?|\*|==|!=)|\b\w+\s*&\s*1u?\s*(?:\?|==|!=|\])|%\s*2u?\b|\b(?:stage|phase|parity|cur|curr|nxt|buf_idx)\w*\b", scan))
    parity_idx = parity_arith
    stem = lambda n: re.sub(r"_?(?:0|1|a|b|A|B|ping|pong)$", "", n)  # noqa: E731
    sibling = []
    for a in tg_arrays:
        for b in tg_arrays:
            if a["name"] < b["name"] and a["elems"] and a["elems"] == b["elems"] and a["type"] == b["type"] and a["elems"] >= 32 and stem(a["name"]) == stem(b["name"]) and stem(a["name"]):
                sibling.append((a["name"], b["name"]))
    slot_ptrs: dict[str, list[str]] = {}
    for m in re.finditer(r"\bthreadgroup\s+(?:const\s+)?[\w:]+\s*\*\s*(?:const\s+)?([A-Za-z_]\w*)\s*=\s*(?:\(\s*threadgroup[^)]*\)\s*)?([A-Za-z_]\w*)\s*(?:\+[^;]*)?;", scan):
        if m.group(2) in tg_names:
            slot_ptrs.setdefault(m.group(2), []).append(m.group(1))
    multi_slot = [a for a, ps in slot_ptrs.items() if len(ps) >= 2]
    stage_dim = [a["name"] for a in tg_arrays if len(a["sizes"]) >= 2 and a["sizes"][0] in (2, 3, 4) and (a["elems"] or 0) >= 64]
    multi_dim = [a["name"] for a in tg_arrays if any(re.search(r"(?:^|[^\w])[234]u?\s*\*|\*\s*[234]u?(?:$|[^\w])|\bSTAGES\b|\bNBUF\b|\bNSTAGE", d) for d in a["dims"]) and (a["elems"] or 0) >= 64]
    ternary_sel = [(a, b) for a, b in sibling if re.search(rf"\?\s*{a}\s*:\s*{b}\b|\?\s*{b}\s*:\s*{a}\b", scan)]
    if (stage_dim and parity_arith) or ternary_sel or (sibling and parity_arith) or (multi_dim and parity_arith) or (multi_slot and loops_exp):
        dbl = "explicit"
    elif sibling or stage_dim or multi_dim:
        dbl = "likely"
    else:
        dbl = "no"
    tile_like = [a for a in tg_arrays if (a["bytes"] or 0) >= 512]
    staging = bool(tile_like and (tg_barriers or "threadgroup" in loads))
    ev["threadgroup"] = {"arrays": [{k: a[k] for k in ("name", "type", "dims", "elems", "bytes")} for a in tg_arrays], "declared_bytes": total_tg, "tile_arrays": [a["name"] for a in tile_like],
                         "staging": staging, "padding_evidence": sorted(set(pad_ev)), "padded": bool(pad_ev), "double_buffer": dbl, "double_buffer_evidence": {"stage_dim_arrays": stage_dim, "sibling_arrays": sibling,
                         "parity_index": parity_idx, "multi_stage_dim_arrays": multi_dim, "slot_pointer_arrays": {a: slot_ptrs[a] for a in multi_slot}, "ternary_selection": ternary_sel}, "register_prefetch": prefetch, "register_prefetch_evidence": prefetch_why,
                         "mma_reads_from_threadgroup": "threadgroup" in loads}

    # ---- simd operations -----------------------------------------------------------------------------------------------------------
    simd = {}
    for m in re.finditer(r"\b(?:simd|quad)_(" + "|".join(_SIMD_NAMES) + r")\s*\(", scan):
        key = m.group(0).split("(")[0].strip()
        simd[key] = simd.get(key, 0) + 1
    ev["simd"] = {"ops": simd, "reduce": sorted(k for k in simd if k.split("_", 1)[1] in _SIMD_REDUCE), "shuffle": sorted(k for k in simd if k.split("_", 1)[1] in _SIMD_SHUFFLE),
                  "simdgroup_index_used": bool(re.search(r"\bsimdgroup_index_in_threadgroup\b", params)) or bool(re.search(r"\bsimdgroup_index_in_threadgroup\b", scan.replace(params, "")))}

    # ---- math, fast-math, unroll, atomics -------------------------------------------------------------------------------------------
    ns_calls = sorted(set(re.findall(r"\b(?:fast|precise)::\w+", scan)))
    fns: dict[str, int] = {}
    for m in re.finditer(r"\b(?:(?:fast|precise)::)?(" + "|".join(_MATH_FNS) + r")\s*\(", scan):
        fns[m.group(1)] = fns.get(m.group(1), 0) + 1
    log2e = bool(re.search(r"1\.44269\d*|M_LOG2E", code))
    ev["math"] = {"functions": fns, "fast_namespace": [c for c in ns_calls if c.startswith("fast::")], "precise_namespace": [c for c in ns_calls if c.startswith("precise::")],
                  "exp2_with_log2e_fold": bool(fns.get("exp2") and log2e)}
    ev["unroll"] = {"pragmas": len(re.findall(r"#\s*pragma\s+(?:unroll|clang\s+loop\s+unroll)", raw_code)), "loops": len(loops_scan)}
    ev["atomics"] = {"calls": len(re.findall(r"\batomic_\w+\s*\(", scan)), "types": sorted(set(re.findall(r"\batomic_(?:uint|int|float|bool)\b", scan)))}
    ev["async_copy"] = bool(re.search(r"\bsimdgroup_event\b|\bsimdgroup_async_copy\b|__metal_simdgroup_async|\basync_copy\b|\bwait_for_event\b", code))
    ev["launch"] = {"max_total_threads_attr": bool(re.search(r"max_total_threads_per_threadgroup", raw_code))}

    # ---- online softmax ------------------------------------------------------------------------------------------------------------
    osm = {"present": False, "evidence": ""}
    for m in re.finditer(r"(?:\bfast::|\bprecise::)?\b(?:[A-Za-z0-9]+_)*exp2?(?:_[A-Za-z0-9]+)*\s*\(", scan):  # also helper wrappers such as exp2_acc / fast_exp
        pc = match_close(scan, m.end() - 1, "(", ")")
        arg = scan[m.end() : pc]
        mm = re.search(r"([A-Za-z_]\w*(?:\[[^\]]*\])?)\s*-\s*([A-Za-z_]\w*(?:\[[^\]]*\])?)", arg)
        if not mm:
            continue
        p, q = mm.group(1), mm.group(2)
        mx_fn = r"(?:fast::|precise::)?[A-Za-z_]*max[A-Za-z_]*"  # max, fmax, nan_max, fmaxn, ...
        upd = re.search(rf"\b{re.escape(q)}\s*=\s*{mx_fn}\s*\(([^;]*)\)", scan) or re.search(rf"\b{re.escape(p)}\s*=\s*{mx_fn}\s*\(([^;]*)\)", scan)
        if not upd or not (re.search(rf"\b{re.escape(p)}\b", upd.group(1)) or re.search(rf"\b{re.escape(q)}\b", upd.group(1))):
            continue
        lhs = re.search(r"([A-Za-z_]\w*)\s*=\s*[^;=]*$", scan[max(0, m.start() - 80) : m.start()])
        alpha = lhs.group(1) if lhs else None
        use = False
        if alpha:
            use = bool(re.search(rf"(\b\w+(?:\[[^\]]*\])*)\s*=\s*\1\s*\*\s*{re.escape(alpha)}\b|\*=\s*{re.escape(alpha)}\b|\b{re.escape(alpha)}\s*\*\s*\w|\w\s*\*\s*{re.escape(alpha)}\b|fma\([^;]*\b{re.escape(alpha)}\b", scan))
        if use or alpha is None:
            osm = {"present": bool(use), "evidence": f"exp of ({p} - {q}) with `{q} = max({p}, ...)`, rescale factor `{alpha}` applied to the running state"}
            if use:
                break
    ev["online_softmax"] = osm

    # ---- epilogue --------------------------------------------------------------------------------------------------------------------
    epi = {"region": False, "on_fragments": False, "math": [], "store_spaces": [], "via_threadgroup": False}
    if mma_pos:
        last = mma_pos[-1]
        outer = [lp for lp in loops_exp if lp[0] <= last < lp[2] and not any(o != lp and o[0] <= lp[0] and lp[2] <= o[2] and o[0] <= last < o[2] for o in loops_exp)]
        start = outer[0][2] if outer else last
        region = expanded[start:]
        epi["region"] = True
        epi["on_fragments"] = bool(re.search(r"\.\s*thread_elements\s*\(", region))
        epi["math"] = sorted({m.group(1) for m in re.finditer(r"\b(?:fast::|precise::)?(" + "|".join(sorted(_EPI_MATH)) + r")\s*\(", region)})
        epi["store_spaces"] = sorted({_space_of(split_top(region[m.end() : match_close(region, m.end() - 1, "(", ")")])[1], tg_all, dev_names) for m in re.finditer(r"\bsimdgroup_store\s*\(", region) if len(split_top(region[m.end() : match_close(region, m.end() - 1, "(", ")")])) > 1})
        epi["via_threadgroup"] = bool(re.search(r"threadgroup\b[^;]*=|\bsimdgroup_store\s*\([^;]*\b(?:" + "|".join(map(re.escape, sorted(tg_all))) + r")\b" if tg_all else r"(?!x)x", region))
        epi["region_chars"] = len(region)
    ev["epilogue"] = epi

    # ---- bit packing of loaded data ------------------------------------------------------------------------------------------------
    bitops = 0
    for v in loaded_at:
        vr = rf"\b{re.escape(v)}(?:\s*\.\s*\w+|\s*\[[^\]]*\])*"
        pat = rf"{vr}\s*(?:>>|<<|&(?!&)|\|(?!\|)|\^)|(?:>>|<<|&(?!&)|\|(?!\|)|\^)\s*\(?\s*{vr}\b"
        bitops += len(re.findall(pat, expanded))
    ev["bit_packing"] = {"sites": bitops, "as_type_calls": len(re.findall(r"\bas_type\s*<", scan)), "extract_bits": len(re.findall(r"\b(?:extract_bits|insert_bits|popcount|rotate)\s*\(", scan)),
                         "present": bitops > 0 or bool(re.search(r"\b(?:extract_bits|popcount)\s*\(", scan))}

    # ---- registers -----------------------------------------------------------------------------------------------------------------
    regs = []
    for m in re.finditer(r"(?:^|[;{}\s])(?:const\s+)?(?:thread\s+)?(u?int|float|half|bfloat|[ui]?short|uchar|float[234]|half[234]|u?int[234]|bfloat[234])\s+([A-Za-z_]\w*)\s*((?:\[[^\]]+\])+)\s*(?:=|;)", scan):
        dims = [eval_expr(d, cst) for d in re.findall(r"\[([^\]]+)\]", m.group(3))]
        n = math.prod(int(d) for d in dims) if all(d is not None for d in dims) else None
        tb = type_bytes(m.group(1))
        regs.append({"name": m.group(2), "type": m.group(1), "elems": n, "bytes": n * tb[2] if (n is not None and tb) else None})
    ev["registers"] = {"arrays": regs, "array_bytes": sum(r["bytes"] or 0 for r in regs)}
    ev["buffers"] = {"inputs": in_bufs, "outputs": out_bufs}
    return ev


def _summarize_sites(sites: list[dict]) -> dict:
    def agg(space: str, kind: str):
        bs = [s["bytes"] for s in sites if s["space"] == space and s["kind"] == kind]
        return {"sites": len(bs), "max_bytes": max(bs) if bs else 0, "widths": {str(b): bs.count(b) for b in sorted(set(bs))}}

    return {"device_load": agg("device", "load"), "device_store": agg("device", "store"), "threadgroup_load": agg("threadgroup", "load"), "threadgroup_store": agg("threadgroup", "store")}


# ======================================================================================================================================
# 5. compiled evidence (AIR text, no GPU)
# ======================================================================================================================================

_TOOLCHAIN: dict[str, Any] = {}
PARSER_VERSION = 7  # bump when parse_air changes: it is part of the compile-cache key


def metal_toolchain() -> str | None:
    """Path of the Metal compiler driver (None if the toolchain component is not installed)."""
    if "metal" not in _TOOLCHAIN:
        try:
            r = subprocess.run(["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True, text=True, timeout=20)
            _TOOLCHAIN["metal"] = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
        except (OSError, subprocess.SubprocessError):
            _TOOLCHAIN["metal"] = None
    return _TOOLCHAIN["metal"]


_LL_BYTES = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "half": 2, "bfloat": 2, "float": 4, "double": 8}


def _ll_type_bytes(ty: str) -> int:
    m = re.fullmatch(r"<(\d+) x (\w+)>", ty.strip())
    if m:
        return int(m.group(1)) * _LL_BYTES.get(m.group(2), 0)
    return _LL_BYTES.get(ty.strip(), 0)


def _ll_size(ty: str) -> int:
    """Byte size of an LLVM type such as ``[64 x [80 x half]]``, ``<4 x i32>`` or ``float``."""
    ty = ty.strip()
    m = re.match(r"\[(\d+) x (.+)\]$", ty)
    if m:
        return int(m.group(1)) * _ll_size(m.group(2))
    return _ll_type_bytes(ty)


def parse_air(text: str) -> dict:
    """Facts from the AIR (LLVM IR) text of one compiled kernel."""
    calls = re.findall(r"\bcall\b[^\n@]*@(air\.[\w.]+)\(", text)
    intr: dict[str, int] = {}
    for c in calls:
        intr[c] = intr.get(c, 0) + 1
    mat = {"multiply_accumulate": 0, "multiply": 0, "load": 0, "store": 0, "init": 0, "init_f32": 0, "acc_types": set(), "operand_types": set(), "load_spaces": set()}
    simd: dict[str, int] = {}
    math_fns: dict[str, int] = {}
    barrier_flags: list[int] = []
    for name, n in intr.items():
        m = re.match(r"air\.simdgroup_matrix_8x8_(multiply_accumulate|multiply|load|store|init_diag|init_filled)\.(.*)$", name)
        if m:
            op, suffix = m.groups()
            key = "init" if op.startswith("init") else op
            mat[key] += n
            if key == "init" and suffix.startswith("v64f32"):
                mat["init_f32"] += n
            parts = suffix.split(".")
            if op == "multiply_accumulate" and len(parts) >= 4:
                mat["acc_types"].add(parts[0].lstrip("v0123456789"))
                mat["operand_types"].add(parts[1].lstrip("v0123456789"))
            elif op == "multiply" and len(parts) >= 3:
                mat["acc_types"].add(parts[0].lstrip("v0123456789"))
                mat["operand_types"].add(parts[1].lstrip("v0123456789"))
            if op in ("load", "store") and len(parts) >= 2:
                mat["load_spaces"].add({"p1": "device", "p3": "threadgroup"}.get(parts[1][:2], parts[1]))
            continue
        if name.startswith("air.simd_") or name.startswith("air.quad_"):
            simd[name.split(".")[1]] = simd.get(name.split(".")[1], 0) + n
        elif re.match(r"air\.(fast_|precise_)?(exp|exp2|log|log2|sin|cos|tanh|sqrt|rsqrt|fma|pow|fmax|fmin|divide|sigmoid)\b", name):
            math_fns[name[4:].split(".")[0]] = math_fns.get(name[4:].split(".")[0], 0) + n
    for m in re.finditer(r"@air\.wg\.barrier\(i32 (\d+), i32 (\d+)\)", text):
        barrier_flags.append(int(m.group(1)))
    n_simd_barrier = len(re.findall(r"@air\.simdgroup\.barrier", text))
    loads: dict[tuple[str, int], int] = {}
    stores: dict[tuple[str, int], int] = {}
    for m in re.finditer(r"\bload (?:volatile )?(<\d+ x \w+>|\w+),\s*(?:<\d+ x \w+>|\w+) addrspace\((\d)\)\*", text):
        key = ({"1": "device", "2": "constant", "3": "threadgroup", "0": "thread"}.get(m.group(2), m.group(2)), _ll_type_bytes(m.group(1)))
        loads[key] = loads.get(key, 0) + 1
    for m in re.finditer(r"\bstore (?:volatile )?(<\d+ x \w+>|\w+) [^,\n]+,\s*(?:<\d+ x \w+>|\w+) addrspace\((\d)\)\*", text):
        key = ({"1": "device", "2": "constant", "3": "threadgroup", "0": "thread"}.get(m.group(2), m.group(2)), _ll_type_bytes(m.group(1)))
        stores[key] = stores.get(key, 0) + 1

    def widths(d, space):
        return {str(b): n for (s, b), n in sorted(d.items()) if s in space and b}

    tg_globals = []
    for m in re.finditer(r"^@(\S+) = [^\n]*?addrspace\(3\) global (.+?)(?: undef| zeroinitializer| poison)?(?:,[^\n]*)?$", text, re.M):
        nbytes = _ll_size(m.group(2).strip())
        tg_globals.append({"name": m.group(1).split("E", 1)[-1][-24:], "bytes": nbytes})
    inner_of = dict(re.findall(r'^(%"struct\.metal::simdgroup_matrix(?:\.\d+)?") = type \{ (%"struct\.metal::_simdgroup_matrix(?:\.\d+)?") \}', text, re.M))
    elem_of = dict(re.findall(r'^(%"struct\.metal::_simdgroup_matrix(?:\.\d+)?") = type \{ <64 x (\w+)> \}', text, re.M))
    tys = {o: elem_of.get(i) for o, i in inner_of.items()}
    frag_arrays: list[dict] = []
    for m in re.finditer(r"alloca ((?:\[\d+ x )+)(%\"struct\.metal::simdgroup_matrix(?:\.\d+)?\")((?:\])+)", text):
        dims = [int(x) for x in re.findall(r"\[(\d+) x", m.group(1))]
        frag_arrays.append({"elem": tys.get(m.group(2)) or "?", "dims": dims, "tiles": math.prod(dims)})
    for m in re.finditer(r"alloca (%\"struct\.metal::simdgroup_matrix(?:\.\d+)?\")[,\s]", text):
        frag_arrays.append({"elem": tys.get(m.group(1)) or "?", "dims": [1], "tiles": 1})
    fast_flags = len(re.findall(r"\b(?:fadd|fmul|fsub|fdiv|fcmp|call) fast\b", text))
    nonfast_fp = len(re.findall(r"\b(?:fadd|fmul|fsub|fdiv) (?!fast)[a-z ]*(?:float|half|<)", text))
    return {
        "intrinsics": dict(sorted(intr.items())), "matrix": {**{k: v for k, v in mat.items() if not isinstance(v, set)}, **{k: sorted(v) for k, v in mat.items() if isinstance(v, set)}},
        "simd": simd, "math": math_fns, "barriers": {"workgroup": len(barrier_flags), "flags": sorted(set(barrier_flags)), "simdgroup": n_simd_barrier},
        "device_load_bytes": widths(loads, {"device", "constant"}), "device_store_bytes": widths(stores, {"device"}), "threadgroup_load_bytes": widths(loads, {"threadgroup"}),
        "threadgroup_store_bytes": widths(stores, {"threadgroup"}), "tg_globals": tg_globals, "tg_global_bytes": sum(g["bytes"] for g in tg_globals),
        "fragment_arrays": frag_arrays, "fragment_tiles_f32": sum(a["tiles"] for a in frag_arrays if a["elem"] in ("float",)), "fast_flags": fast_flags, "nonfast_fp_ops": nonfast_fp,
        "atomics": len(re.findall(r"\batomicrmw\b|@air\.atomic", text)), "ir_lines": text.count("\n"),
    }


def compile_air(full_source: str, *, std: str = "metal4.0", fast_math: bool = True, timeout: float = 90.0, cache_dir: Path | None = None) -> dict:
    """Compile ``full_source`` (contract header + candidate) to AIR text and parse it. Never touches the GPU."""
    key = hashlib.sha1((f"v{PARSER_VERSION}|{std}|{fast_math}|" + full_source).encode()).hexdigest()
    cf = cache_dir / f"{key}.json" if cache_dir else None
    if cf and cf.exists():
        try:
            return json.loads(cf.read_text())
        except (OSError, ValueError):
            pass
    tool = metal_toolchain()
    if tool is None:
        return {"ok": False, "error": "Metal toolchain not available (`xcrun -sdk macosx --find metal` failed)", "skipped": True}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hwf_") as td:
        src, out = Path(td) / "k.metal", Path(td) / "k.ll"
        src.write_text(full_source)
        res = None
        for stdv in (std, "metal3.2", "metal3.1"):
            cmd = ["nice", "-n", "15", "xcrun", "-sdk", "macosx", "metal", f"-std={stdv}", "-w", "-S", "-emit-llvm", "-o", str(out), str(src)]
            cmd.insert(8, "-ffast-math" if fast_math else "-fno-fast-math")
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": f"compile timed out after {timeout:.0f} s"}
            if res.returncode == 0 or "invalid value" not in (res.stderr or ""):
                std_used = stdv
                break
        if res is None or res.returncode != 0 or not out.exists():
            err = (res.stderr if res else "")
            first = next((ln for ln in err.splitlines() if "error:" in ln), err.strip()[:300])
            ev = {"ok": False, "error": re.sub(r"^.*?k\.metal:", "k.metal:", first)[:300], "seconds": round(time.perf_counter() - t0, 2)}
        else:
            ev = {"ok": True, "std": std_used, "seconds": round(time.perf_counter() - t0, 2), **parse_air(out.read_text())}
    if cf:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps(ev))
        except OSError:
            pass
    return ev


# ======================================================================================================================================
# 6. features, expectations, verdicts
# ======================================================================================================================================

FEATURES: dict[str, dict] = {
    "matrix_unit": {"title": "matrix unit (simdgroup_matrix 8x8 MMA)", "paper": "P1 Tensor Core MMA (mma.sync / WGMMA / tcgen05)", "family": "compute"},
    "float_accumulate": {"title": "float accumulators for the MMA", "paper": "P1 (FP32 accumulation of FP16/FP8 MMA)", "family": "compute"},
    "register_blocking": {"title": "register blocking (>= 4 accumulator fragments per SIMD group)", "paper": "P2 larger tiles per instruction group (m64n256 vs m64n64), tile kept in registers", "family": "compute"},
    "simdgroup_tiling": {"title": "work split per SIMD group (simdgroup_index_in_threadgroup)", "paper": "P13 thread mapping / warp-independent slices", "family": "any"},
    "tg_staging": {"title": "threadgroup-memory staging of tiles", "paper": "P6 shared-memory staging, packed SMEM stores", "family": "compute"},
    "tg_padding": {"title": "padded / odd threadgroup strides", "paper": "P5 XOR / offset swizzle (bank-conflict avoidance)", "family": "compute"},
    "double_buffering": {"title": "multi-stage threadgroup buffers (double buffering)", "paper": "P7 double buffering / multi-stage pipelines", "family": "compute"},
    "pipelining": {"title": "software pipelining of loads and compute (double buffer or register prefetch)", "paper": "P7/P8 double buffering, cp.async / TMA pipelines", "family": "compute"},
    "register_prefetch": {"title": "global loads in flight across barrier / MMA (register prefetch)", "paper": "P8 cp.async / TMA overlap of copy and compute (Metal analogue)", "family": "compute"},
    "async_copy": {"title": "hardware async copy (simdgroup_event / async_copy)", "paper": "P8 cp.async / TMA + mbarrier", "family": "compute"},
    "vector_access": {"title": "vectorized device loads / stores (>= 8 B per access)", "paper": "P10 packed / vectorized accesses, 32-bit loads, vectorized global stores", "family": "memory"},
    "simd_reduce": {"title": "SIMD-group reductions (simd_sum / simd_max / prefix)", "paper": "P9 warp shuffle reductions, redux.sync", "family": "memory"},
    "simd_shuffle": {"title": "SIMD-group shuffles / broadcasts", "paper": "P9 warp-level shuffles", "family": "any"},
    "online_softmax": {"title": "online softmax (running max + rescale)", "paper": "P11 softmax kept in registers, normalization overlapped with MMA", "family": "compute"},
    "operand_in_registers": {"title": "MMA operand fragments built in registers (thread_elements)", "paper": "P4 decode directly into Tensor Core operands", "family": "compute"},
    "epilogue_on_fragments": {"title": "epilogue applied on accumulator registers", "paper": "P12 epilogue / layout conversion (stmatrix, one sync, vector stores)", "family": "compute"},
    "bit_packing": {"title": "bit-level packing / decoding of loaded data", "paper": "P14 funnel shifts, byte permutes, lop3, packed formats", "family": "any"},
    "single_pass": {"title": "one pass over every input in global memory", "paper": "(skills) move every byte once", "family": "memory"},
    "few_barriers": {"title": "few threadgroup barriers (<= 2 per main-loop iteration / kernel)", "paper": "P15 fewer synchronization points (3 -> 1)", "family": "any"},
    "unroll": {"title": "explicit loop unrolling (#pragma unroll)", "paper": "(loop overhead, FlashSinkhorn/BitDelta unrolled steps)", "family": "any"},
    "fast_math": {"title": "fast math / exp2 with folded log2(e)", "paper": "P16 target-specific instructions (tanh, exp2)", "family": "any"},
    "tensor_api": {"title": "Metal 4 tensor / cooperative-tensor API", "paper": "P3 (Blackwell tensor memory + descriptor MMA) - nearest Apple analogue", "family": "compute"},
}

# paper feature (docs/hardware_features.md table 1) -> detector features that implement its Apple analogue (any of them present counts)
PAPER_TO_DETECTOR: dict[str, tuple[str, tuple[str, ...]]] = {
    "P1": ("Tensor Core MMA (mma.sync / ldmatrix)", ("matrix_unit",)),
    "P2": ("larger MMA tiles per instruction group (WGMMA m64n256)", ("register_blocking",)),
    "P3": ("tcgen05 + tensor memory, descriptor MMA", ("tensor_api",)),
    "P4": ("decode directly into MMA operands", ("operand_in_registers",)),
    "P5": ("shared-memory swizzle (bank conflicts)", ("tg_padding",)),
    "P6": ("shared-memory staging", ("tg_staging",)),
    "P7": ("double buffering / multi-stage pipeline", ("double_buffering",)),
    "P8": ("cp.async / TMA copy-compute overlap", ("register_prefetch", "async_copy")),
    "P9": ("warp shuffles and reductions", ("simd_reduce", "simd_shuffle")),
    "P10": ("packed / vectorized accesses", ("vector_access",)),
    "P11": ("softmax kept in registers (online)", ("online_softmax",)),
    "P12": ("epilogue / layout conversion on registers", ("epilogue_on_fragments",)),
    "P13": ("thread mapping per warp", ("simdgroup_tiling",)),
    "P14": ("funnel shift / byte permute / lop3 / packed formats", ("bit_packing",)),
    "P15": ("fewer synchronization points", ("few_barriers",)),
    "P16": ("target-specific math (tanh, exp2)", ("fast_math",)),
}

# Expected features per task: "required" (an optimal kernel has it), "recommended" (the paper / the repo's skills say it pays), absent = not applicable.
_ELEMENTWISE = {"vector_access": "required", "single_pass": "required"}
_ROWWISE = {"vector_access": "required", "single_pass": "required", "simd_reduce": "required", "few_barriers": "recommended"}
_GEMMLIKE = {"matrix_unit": "required", "float_accumulate": "required", "register_blocking": "required", "tg_staging": "required", "vector_access": "recommended",
             "tg_padding": "recommended", "pipelining": "recommended"}
EXPECTED: dict[str, dict[str, str]] = {
    "sigmoid": {**_ELEMENTWISE, "fast_math": "recommended"},
    "relu": dict(_ELEMENTWISE),
    "silu": {**_ELEMENTWISE, "fast_math": "recommended"},
    "swiglu": {**_ELEMENTWISE, "fast_math": "recommended"},
    "lion": dict(_ELEMENTWISE),
    "rmsnorm": {**_ROWWISE},
    "softmax": {**_ROWWISE, "fast_math": "recommended"},
    "reduction_sum": {**_ROWWISE},
    "rope": {"vector_access": "required", "single_pass": "required"},
    "gemv": {"vector_access": "required", "single_pass": "required", "simd_reduce": "required"},
    "gemm": dict(_GEMMLIKE),
    "gemm_silu": {**_GEMMLIKE, "epilogue_on_fragments": "required", "fast_math": "recommended"},
    "conv2d": {**_GEMMLIKE, "vector_access": "required"},
    "flash_attention": {**_GEMMLIKE, "tg_staging": "recommended", "online_softmax": "required", "simd_shuffle": "recommended", "operand_in_registers": "recommended", "few_barriers": "recommended", "fast_math": "recommended"},
    "bitdelta_gemm": {**_GEMMLIKE, "bit_packing": "required", "operand_in_registers": "recommended"},
    "mamba2_chunk_scan": {**{k: v for k, v in _GEMMLIKE.items() if k not in ("tg_padding",)}, "matrix_unit": "recommended", "float_accumulate": "recommended", "register_blocking": "recommended", "tg_staging": "recommended",
                          "fast_math": "recommended"},
}
TASK_FAMILY = {**{t: "elementwise" for t in ("sigmoid", "relu", "silu", "swiglu", "lion", "rope")}, **{t: "rowwise" for t in ("rmsnorm", "softmax", "reduction_sum", "gemv")},
               **{t: "compute" for t in ("gemm", "gemm_silu", "conv2d", "flash_attention", "bitdelta_gemm", "mamba2_chunk_scan")}}
PRIMARY = {"compute": {"matrix_unit"}, "elementwise": {"vector_access"}, "rowwise": {"vector_access", "simd_reduce"}}
SCALAR_OVER_MATRIX = 5.5 / 14.8  # documented in taic/skills/apple_gpu_optimization.md ("measured": scalar-FMA GEMM ~5.5 vs matrix ~14.8 TFLOP/s)
NAIVE_FRACTION_OF_PEAK_BW = 0.87  # same file: "naive scalar versions already reach 80-87 % of the measured peak"
EFFECT_MARGIN = {"compute": 1.10, "memory": 1.02}

VERDICTS = ("used and effective", "present but not effective", "expected but missing", "not applicable")


def base_task(task: str | None) -> str | None:
    return task.split("@")[0] if task else None


def detect_features(src: dict, comp: dict | None, pipeline: dict | None = None) -> dict[str, dict]:
    """Per feature: present (source), compiled confirmation, detail."""
    mat, mem, tg, simd = src["matrix"], src["memory"], src["threadgroup"], src["simd"]
    cm = (comp or {}).get("matrix", {}) if comp and comp.get("ok") else None
    ok = bool(comp and comp.get("ok"))
    out: dict[str, dict] = {}

    def put(fid: str, present: bool, compiled: bool | None, detail: str):
        out[fid] = {"present": bool(present), "source": bool(present), "compiled": compiled, "detail": detail}

    put("matrix_unit", mat["present"], (cm["multiply_accumulate"] + cm["multiply"] > 0) if ok else None,
        (f"{mat['mma_sites']} MMA site(s), operands {','.join(mat['operand_types']) or '?'} -> acc {','.join(mat['acc_types']) or '?'}" + (f"; AIR: {cm['multiply_accumulate']} multiply_accumulate call site(s)" if ok else "")) if mat["present"] else "no simdgroup_multiply(_accumulate)")
    facc = bool(mat["present"] and mat["acc_types"] and all(t == "float" for t in mat["acc_types"]))
    put("float_accumulate", facc, ("f32" in (cm or {}).get("acc_types", [])) if ok and mat["present"] else None, f"accumulator types {mat['acc_types']}" if mat["present"] else "no matrix unit")
    tiles = mat["acc_tiles"] or 0
    ctiles = max((comp or {}).get("fragment_tiles_f32", 0), cm.get("init_f32", 0)) if ok else None
    put("register_blocking", mat["present"] and tiles >= 4, (ctiles >= 4) if ok and mat["present"] else None, f"{tiles} accumulator fragment(s) ({mat['acc_dims']})" + (f"; AIR f32 fragment arrays: {ctiles}" if ok else ""))
    put("simdgroup_tiling", simd["simdgroup_index_used"], None, "simdgroup_index_in_threadgroup used" if simd["simdgroup_index_used"] else "no per-SIMD-group index")
    put("tg_staging", tg["staging"], (comp["tg_global_bytes"] >= 512) if ok else None, f"{tg['declared_bytes']} B declared in {len(tg['arrays'])} array(s); tiles: {tg['tile_arrays']}" + (f"; AIR threadgroup globals {comp['tg_global_bytes']} B" if ok else ""))
    put("tg_padding", tg["staging"] and tg["padded"], None, "; ".join(tg["padding_evidence"]) or "no padded / odd stride found (heuristic)")
    put("double_buffering", tg["double_buffer"] == "explicit", None, f"{tg['double_buffer']}: {tg['double_buffer_evidence']}")
    put("register_prefetch", tg["register_prefetch"], None, tg["register_prefetch_evidence"] or "no loop-carried register staging found (heuristic)")
    put("pipelining", tg["double_buffer"] == "explicit" or tg["register_prefetch"], None,
        "; ".join(x for x in (("multi-stage threadgroup buffers" if tg["double_buffer"] == "explicit" else ""), ("register prefetch" if tg["register_prefetch"] else "")) if x) or "single-stage: load, barrier, compute, barrier")
    put("async_copy", src["async_copy"], None, "simdgroup_event / async_copy referenced (the public toolchain has no such header)" if src["async_copy"] else "none")
    dl, ds = mem["device_load"], mem["device_store"]
    cl = max((int(b) for b in (comp or {}).get("device_load_bytes", {})), default=0) if ok else None
    cs = max((int(b) for b in (comp or {}).get("device_store_bytes", {})), default=0) if ok else None
    vec = max(dl["max_bytes"], ds["max_bytes"]) >= 8
    put("vector_access", vec, (max(cl or 0, cs or 0) >= 8) if ok else None, f"source: load {dl['max_bytes'] or 'scalar'} B / store {ds['max_bytes'] or 'scalar'} B (sites {dl['sites']}/{ds['sites']})" + (f"; AIR: widest load {cl} B / store {cs} B" if ok else ""))
    csimd = (comp or {}).get("simd", {}) if ok else None
    put("simd_reduce", bool(simd["reduce"]), bool(csimd and any(k.split("_", 1)[1] in _SIMD_REDUCE or k in ("simd_sum", "simd_max", "simd_min") or k.startswith("simd_prefix") for k in csimd)) if ok else None, ", ".join(simd["reduce"]) or "none")
    put("simd_shuffle", bool(simd["shuffle"]), bool(csimd and any("shuffle" in k or "broadcast" in k for k in csimd)) if ok else None, ", ".join(simd["shuffle"]) or "none")
    put("online_softmax", src["online_softmax"]["present"], None, src["online_softmax"]["evidence"] or "no running-max / rescale pattern")
    opr = mat["thread_elements_on_operands"]
    put("operand_in_registers", bool(opr), None, f"thread_elements() on operand fragment(s) {opr}" if opr else "operands come from simdgroup_load only")
    epi = src["epilogue"]
    put("epilogue_on_fragments", mat["present"] and epi["on_fragments"], None, f"post-MMA thread_elements; math {epi['math']}; stores to {epi['store_spaces']}" if epi["on_fragments"] else "no thread_elements() after the last MMA")
    bp = src["bit_packing"]
    put("bit_packing", bp["present"], None, f"{bp['sites']} shift/mask statement(s) on loaded data, {bp['as_type_calls']} as_type")
    passes = src["global_read_passes"]
    sp = bool(passes) and max(passes.values()) <= 1
    put("single_pass", sp, None, f"static read passes per input: {passes}")
    sy = src["sync"]
    nb = sy["main_loop_barrier_sites"] if sy["main_loop_barrier_sites"] is not None else sy["threadgroup_barrier_sites"]
    put("few_barriers", nb <= 2, None, f"{sy['threadgroup_barrier_sites']} barrier site(s), {nb} in the main loop / kernel" + (f"; AIR {comp['barriers']['workgroup']}" if ok else ""))
    put("unroll", src["unroll"]["pragmas"] > 0, None, f"{src['unroll']['pragmas']} unroll pragma(s) over {src['unroll']['loops']} loop(s)")
    m = src["math"]
    fast_air = bool(ok and comp["math"] and any(k.startswith("fast_") or k.startswith("exp2") for k in comp["math"]))
    put("fast_math", bool(m["fast_namespace"]) or m["exp2_with_log2e_fold"] or fast_air, fast_air if ok else None, f"{m['fast_namespace'] or ''} exp2+log2e fold: {m['exp2_with_log2e_fold']}; AIR intrinsics {sorted((comp or {}).get('math', {}))}" if ok else f"{m['fast_namespace']} exp2+log2e fold: {m['exp2_with_log2e_fold']}")
    put("tensor_api", mat["tensor_api"], None, "metal_tensor / cooperative_tensor / matmul2d referenced" if mat["tensor_api"] else "none")
    return out


@dataclass
class Ceilings:
    """The no-feature performance ceilings a candidate of a task is compared against."""

    naive_tflops: float | None = None
    naive_gbps: float | None = None
    peak_tflops: float | None = None
    peak_gbps: float | None = None
    source: str = ""

    def scalar_tflops(self) -> tuple[float | None, str]:
        if self.naive_tflops:
            return self.naive_tflops, "task's naive scalar kernel"
        if self.peak_tflops:
            return SCALAR_OVER_MATRIX * self.peak_tflops, f"{SCALAR_OVER_MATRIX:.2f} x measured matrix peak"
        return None, ""

    def memory_gbps(self) -> tuple[float | None, str]:
        if self.naive_gbps:
            return self.naive_gbps, "task's naive scalar kernel"
        if self.peak_gbps:
            return NAIVE_FRACTION_OF_PEAK_BW * self.peak_gbps, f"{NAIVE_FRACTION_OF_PEAK_BW:.2f} x measured bandwidth peak"
        return None, ""


def perf_test(task: str | None, measured: dict | None, ceil: Ceilings | None) -> dict:
    """Is the measured performance clearly above what the kernel reaches without the task family's primary feature?"""
    fam = TASK_FAMILY.get(base_task(task) or "", None)
    if not measured or measured.get("status") != "ok" or not measured.get("candidate_us"):
        return {"result": "unmeasured", "why": f"no benchmark (status {measured.get('status') if measured else 'n/a'})"}
    if fam == "compute":
        ach = measured.get("achieved_tflops")
        ref, how = (ceil.scalar_tflops() if ceil else (None, ""))
        if ach is None or ref is None:
            return {"result": "unknown", "why": "no roofline numbers"}
        r = ach / ref
        return {"result": "above" if r >= EFFECT_MARGIN["compute"] else "at", "kind": "compute", "achieved": round(ach, 2), "ceiling": round(ref, 2), "ratio": round(r, 2), "ceiling_basis": how, "unit": "TFLOP/s"}
    if fam in ("elementwise", "rowwise"):
        ach = measured.get("achieved_gbps")
        ref, how = (ceil.memory_gbps() if ceil else (None, ""))
        if ach is None or ref is None:
            return {"result": "unknown", "why": "no roofline numbers"}
        r = ach / ref
        return {"result": "above" if r >= EFFECT_MARGIN["memory"] else "at", "kind": "memory", "achieved": round(ach), "ceiling": round(ref), "ratio": round(r, 3), "ceiling_basis": how, "unit": "GB/s"}
    return {"result": "unknown", "why": "task family unknown"}


def judge(task: str | None, feats: dict[str, dict], pt: dict) -> tuple[dict[str, dict], list[str]]:
    """Attach expectation, verdict and attribution to every feature; return (features, 'expected but missing' ids, required first)."""
    exp = EXPECTED.get(base_task(task) or "", {})
    fam = TASK_FAMILY.get(base_task(task) or "", "")
    prim = PRIMARY.get(fam, set())
    missing_req, missing_rec = [], []
    for fid, f in feats.items():
        level = exp.get(fid, "n/a")
        f["expectation"] = level
        f["attribution"] = "primary" if fid in prim else "joint"
        if f["present"]:
            if pt["result"] == "above":
                v = "used and effective"
            elif pt["result"] == "at":
                v = "present but not effective"
            elif pt["result"] == "unmeasured":
                v = "present, not measured"
            else:
                v = "present, effect unknown"
            if level == "n/a" and fid not in prim:
                v = "present, not expected" if v != "present, not measured" else v
            f["verdict"] = v
            if f["compiled"] is False:
                f["note"] = "in the source but not in the compiled AIR (dead code or removed by the optimizer)"
        elif level in ("required", "recommended"):
            f["verdict"] = "expected but missing"
            (missing_req if level == "required" else missing_rec).append(fid)
        else:
            f["verdict"] = "not applicable"
    return feats, missing_req + missing_rec


def summary_line(src: dict, comp: dict | None, feats: dict[str, dict], missing: list[str]) -> str:
    mat, mem, tg = src["matrix"], src["memory"], src["threadgroup"]
    parts = []
    if mat["present"]:
        acc = "/".join(mat["acc_types"]) or "?"
        opr = "/".join(mat["operand_types"]) or "?"
        tiles = mat["acc_tiles"]
        shape = "x".join(str(d) for d in mat["acc_dims"]) if mat["acc_dims"] else ""
        parts.append(f"matrix unit: yes (8x8 {opr}->{acc} acc, {tiles if tiles is not None else '?'} tile(s){' [' + shape + ']' if shape else ''})")
    else:
        parts.append("matrix unit: no")
    if tg["staging"]:
        bits = [f"threadgroup {tg['declared_bytes'] / 1024:.1f} KiB"]
        if feats["tg_padding"]["present"]:
            bits.append("padded")
        if tg["double_buffer"] == "explicit":
            bits.append("double-buffered")
        elif tg["double_buffer"] == "likely":
            bits.append("2 buffers?")
        if tg["register_prefetch"]:
            bits.append("register prefetch")
        parts.append("staging: " + ", ".join(bits))
    else:
        parts.append("staging: none")
    dl, ds = mem["device_load"]["max_bytes"], mem["device_store"]["max_bytes"]
    st_txt = f"{ds}B" if ds >= 4 else ("simdgroup_store" if "device" in mat["store_to"] else "scalar")
    ld_txt = f"{dl}B" if dl >= 4 else ("simdgroup_load" if "device" in mat["loads_from"] else "scalar")
    parts.append(f"vector: {ld_txt} ld / {st_txt} st")
    simd = src["simd"]["ops"]
    if simd:
        parts.append("simd: " + ",".join(sorted(k.replace("simd_", "") for k in simd)))
    extra = [n for n, k in (("online softmax", "online_softmax"), ("operand fragments in regs", "operand_in_registers"), ("epilogue on regs", "epilogue_on_fragments"), ("bit packing", "bit_packing"), ("unroll", "unroll")) if feats[k]["present"]]
    if extra:
        parts.append(", ".join(extra))
    s = " | ".join(parts)
    if missing:
        s += " | missing: " + ", ".join(m.replace("_", " ") for m in missing[:5])
    return s


@dataclass
class HardwareFeatureReport:
    label: str
    task: str | None
    dtype: str | None
    source: dict
    compiled: dict | None
    measured: dict | None
    performance_test: dict
    features: dict[str, dict]
    missing: list[str]
    summary: str
    partial: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=lambda o: sorted(o) if isinstance(o, set) else str(o))


def build_report(source: str, *, label: str, task: str | None = None, dtype: str | None = None, size: str = "paper", measured: dict | None = None, ceilings: Ceilings | None = None,
                 compile_evidence: bool = True, cache_dir: Path | None = None, partial: bool = False, extra: dict | None = None, task_obj: Any = None,
                 compile_timeout: float = 90.0) -> HardwareFeatureReport:
    """The whole pipeline for one kernel source: source evidence -> compiled evidence -> measured evidence -> verdicts.
    ``task_obj`` (a ``tcenv.task.Task``) replaces the lookup of ``task`` by name when the caller already holds it."""
    consts, header, kname = {}, "", None
    if task:
        try:
            t = task_obj if task_obj is not None else _get_task(task, dtype or "fp16", size)
            consts = {n: v for n, _, v in t.contract.constants}
            kname = t.contract.kernel_name
            header = t.contract.render_header()
        except Exception as e:  # noqa: BLE001 - an unresolvable task only disables header-dependent evidence
            extra = {**(extra or {}), "task_error": str(e)[:200]}
    from tcenv.launch import declared_stage_kernels, launch_math

    launched = declared_stage_kernels(source)
    src = analyze_source(source, consts=consts, kernel_name=None if launched else kname, kernel_names=launched)
    comp = None
    if compile_evidence:
        if task and header:
            math = launch_math(source)
            if "taic-launch" in source and math is None:
                comp = {"ok": False, "skipped": True, "error": "launch block did not parse; AIR evidence suppressed"}
            else:
                comp = compile_air(header + source, cache_dir=cache_dir, timeout=compile_timeout, fast_math=math != "safe")
        else:
            comp = {"ok": False, "skipped": True, "error": "no contract header (task unknown): not compiled"}
    feats = detect_features(src, comp)
    pt = perf_test(task, measured, ceilings)
    feats, missing = judge(task, feats, pt)
    return HardwareFeatureReport(label=label, task=task, dtype=dtype, source=src, compiled=comp, measured=measured, performance_test=pt, features=feats, missing=missing,
                                 summary=summary_line(src, comp, feats, missing), partial=partial, extra=extra or {})


_TASK_CACHE: dict[tuple, Any] = {}


def _get_task(name: str, dtype: str, size: str):
    key = (name, dtype, size)
    if key not in _TASK_CACHE:
        sys.path.insert(0, str(REPO))
        from tasks import get_task  # lazy: heavy (torch) and not needed for pure source analysis

        _TASK_CACHE[key] = get_task(name, dtype, size)
    return _TASK_CACHE[key]


ablation_plan = """\
OPTIONAL ablation (not run by this tool; it needs GPU time under the exclusive lock): to attribute a speedup to one feature, build a variant of the
best kernel with only that feature removed (matrix unit -> scalar FMA loop over the same tile; vector loads -> scalar loads of the same addresses; register
blocking -> 1 fragment per SIMD group; double buffer / register prefetch -> load, barrier, compute, barrier; padding -> pad 0; epilogue on fragments ->
store to threadgroup first), keep ABI and tolerance, and measure it in the same interleaved protocol (``tcenv.evaluate.evaluate(task, dtype, size, source)``).
The ratio variant / original is the feature's contribution *in that kernel*; features interact, so the contributions do not add up.
"""


# ======================================================================================================================================
# 7. measured evidence
# ======================================================================================================================================


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def measured_from_report(rep: dict | None, task_obj: Any = None, calib: dict | None = None) -> dict | None:
    """Flatten a harness ``report.json`` (or a naive-kernel summary) into the numbers the verdicts need."""
    if not rep:
        return None
    m: dict[str, Any] = {"status": rep.get("status"), "speedup": rep.get("speedup"), "candidate_us": rep.get("candidate_us"), "baseline_us": rep.get("baseline_us")}
    r = rep.get("roofline") or {}
    if r:
        m.update({k: r.get(k) for k in ("bound", "achieved_tflops", "compute_fraction", "achieved_gbps", "bandwidth_fraction", "time_over_lower_bound", "peak_tflops_measured", "peak_gbps_measured")})
    elif task_obj is not None and m["candidate_us"]:
        us = m["candidate_us"]
        m["achieved_gbps"] = task_obj.min_bytes / (us * 1e-6) / 1e9
        m["achieved_tflops"] = task_obj.flops / (us * 1e-6) / 1e12 if task_obj.flops else None
        if calib:
            m["peak_gbps_measured"] = calib.get("peak_gbps")
            m["peak_tflops_measured"] = (calib.get("peak_tflops") or {}).get(task_obj.dtype)
            m["bandwidth_fraction"] = m["achieved_gbps"] / m["peak_gbps_measured"] if m["peak_gbps_measured"] else None
            m["compute_fraction"] = m["achieved_tflops"] / m["peak_tflops_measured"] if (m["achieved_tflops"] and m["peak_tflops_measured"]) else None
        m["derived_from"] = "candidate_us and the task's flops / min_bytes (naive summary has no roofline)"
    p = rep.get("pipeline") or {}
    if p:
        m["max_total_threads"] = p.get("max_total_threads_per_threadgroup")
        m["static_threadgroup_bytes"] = p.get("static_threadgroup_bytes")
        stages = p.get("stages") if isinstance(p.get("stages"), list) else None
        if stages:
            m["launch_threads"] = None
            m["stages"] = [
                {
                    "kernel": s.get("kernel"),
                    "max_total_threads": s.get("max_total_threads_per_threadgroup"),
                    "launch_threads": s.get("launch_threads_per_threadgroup"),
                }
                for s in stages
            ]
        else:
            m["launch_threads"] = p.get("launch_threads_per_threadgroup")
    b = rep.get("benchmark") or {}
    if b:
        m["cv"] = (b.get("candidate") or {}).get("cv")
        m["noisy"] = b.get("noisy")
    return m

