"""One integer/boolean expression evaluator for the v2 geometry plane (A2).

The plan requires the OpSpec JSON *term* and the launch *text* expression to be evaluated by
the **same** rules, so a candidate cannot get one answer from ``ExecutionGeometry.grid`` and a
different answer from the rendered ``taic-launch`` line.  This module is that single evaluator:

* a JSON term may be an integer, a name, a single-key mapping (``{"ceil_div": ["N", 256]}``),
  a list/tuple in prefix form (``["gt", "K", 0]``), or a text expression (``"ceil_div(N,256)+1"``);
* text expressions use the arithmetic grammar of :mod:`tcenv.backends.metal.launch`
  (``+ - * / ( )``, exact integer division, identifiers), plus the same closed function set.

Nothing here is Metal specific: the function set is ``ceil_div``/``min``/``max``/``trunc_div``/
``rem`` for integers, ``gt/ge/lt/le/eq/ne`` for comparisons and ``and/or/not`` for predicates.
There is no ``tile``/``threads`` constant, no op name and no per-task table: an expression only
sees the names its caller supplies.

``/`` keeps the A2 semantics: it is exact integer division and an inexact (or zero) divisor is a
``GeometryError``.  The shape-portfolio guard layer adds ``trunc_div(a, b)`` and ``rem(a, b)``
explicitly instead of reinterpreting ``/``: both truncate toward zero, the remainder has the
sign of the dividend (``rem(-7, 2) == -1``) and a zero divisor is refused.  Tail/alignment
predicates must use ``rem``; nothing here silently rounds a launch size.

A term that cannot be evaluated, that divides inexactly, that calls an unregistered function or
that uses a name the instance does not declare is a ``GeometryError``.  Evaluation never
guesses and never falls back to a default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..contract_errors import GeometryError

INT = "int"
BOOL = "bool"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Closed function set.  ``(min_args, max_args)``; ``None`` means "unbounded".
FUNCTIONS: Mapping[str, tuple[int, int | None]] = {
    "ceil_div": (2, 2),
    "trunc_div": (2, 2),
    "rem": (2, 2),
    "min": (1, None),
    "max": (1, None),
    "gt": (2, 2),
    "ge": (2, 2),
    "lt": (2, 2),
    "le": (2, 2),
    "eq": (2, 2),
    "ne": (2, 2),
    "and": (2, None),
    "or": (2, None),
    "not": (1, 1),
}
COMPARISONS = frozenset({"gt", "ge", "lt", "le", "eq", "ne"})
BOOLEAN_FUNCTIONS = frozenset({"and", "or", "not"}) | COMPARISONS
ARITHMETIC_OPERATORS = frozenset({"+", "-", "*", "/"})


@dataclass(frozen=True)
class Const:
    """@brief 一个整数字面量。"""

    value: int


@dataclass(frozen=True)
class Var:
    """@brief 一个名字引用（symbol、attr 或属性名）。"""

    name: str


@dataclass(frozen=True)
class Call:
    """@brief 一次函数/运算符应用。"""

    func: str
    args: tuple[Any, ...]


Node = Const | Var | Call


def parse_term(term: Any, where: str) -> Node:
    """@brief 把一个 JSON/text 术语解析成表达式树。

    @param term 术语（整数、名字、文本表达式、单键映射或前缀列表）。
    @param where 诊断位置。
    @return 表达式树。
    @raises GeometryError 术语不是可求值的表达式。
    """
    if isinstance(term, bool):
        raise GeometryError(f"{where}: a geometry term must not be a boolean literal; use a predicate function")
    if isinstance(term, int):
        return Const(int(term))
    if isinstance(term, str):
        text = term.strip()
        if not text:
            raise GeometryError(f"{where}: a geometry term string must be non-empty")
        if _IDENTIFIER.fullmatch(text):
            return Var(text)
        return _parse_text(text, where)
    if isinstance(term, Mapping):
        if len(term) != 1:
            raise GeometryError(f"{where}: a function term must have exactly one key, got {sorted(term)}")
        (func, args), = term.items()
        return _call(str(func), args, where)
    if isinstance(term, (list, tuple)):
        if not term:
            raise GeometryError(f"{where}: an empty list is not an expression")
        head, rest = term[0], term[1:]
        if not isinstance(head, str):
            raise GeometryError(f"{where}: a prefix expression needs an operator/function name first")
        return _call(head, list(rest), where)
    raise GeometryError(f"{where}: unsupported geometry term type {type(term).__name__}")


def _call(func: str, args: Any, where: str) -> Call:
    """@brief 构造一次已登记的应用。

    @param func 函数/运算符名。
    @param args 参数（单个值或序列）。
    @param where 诊断位置。
    @return 应用节点。
    @raises GeometryError 函数未登记或参数个数不对。
    """
    if func in ARITHMETIC_OPERATORS:
        # Normalise first so a single operand works whether the author wrote ``{"-": "K"}`` or
        # ``{"-": ["K"]}``; one function-application path, not two spellings with two rules.
        if not isinstance(args, (list, tuple)):
            args = [args]
        if func == "-" and len(args) == 1:
            return Call(func, (parse_term(args[0], where),))
        if func == "+" and len(args) == 1:
            return parse_term(args[0], where)
        arity = (2, 2)
    elif func in FUNCTIONS:
        arity = FUNCTIONS[func]
        if not isinstance(args, (list, tuple)):
            args = [args]
    else:
        raise GeometryError(f"{where}: unknown geometry function {func!r}; registered functions are {sorted(FUNCTIONS)}")
    low, high = arity
    if len(args) < low or (high is not None and len(args) > high):
        raise GeometryError(f"{where}: {func}() takes {low}..{high if high is not None else 'n'} arguments, got {len(args)}")
    # ``_parse_text`` already builds child nodes; a JSON term still holds raw values.  Accepting
    # both keeps one function-application path instead of two parsers.
    return Call(
        func,
        tuple(
            item if isinstance(item, (Const, Var, Call)) else parse_term(item, f"{where}.{func}[{index}]")
            for index, item in enumerate(args)
        ),
    )


def kind_of(node: Node) -> str:
    """@brief 表达式的静态类型（``int`` 或 ``bool``）。

    @param node 表达式树。
    @return 类型名。
    @raises GeometryError 类型不适配（例如 ``and`` 收到整数、``+`` 收到谓词）。
    """
    if isinstance(node, Const):
        return INT
    if isinstance(node, Var):
        return INT
    if node.func in COMPARISONS:
        for index, arg in enumerate(node.args):
            if kind_of(arg) != INT:
                raise GeometryError(f"{node.func}() argument {index} must be an integer, not a predicate")
        return BOOL
    if node.func in ("and", "or", "not"):
        for index, arg in enumerate(node.args):
            if kind_of(arg) != BOOL:
                raise GeometryError(f"{node.func}() argument {index} must be a predicate, not an integer")
        return BOOL
    for index, arg in enumerate(node.args):
        if kind_of(arg) != INT:
            raise GeometryError(f"{node.func}() argument {index} must be an integer, not a predicate")
    return INT


def names_of(node: Node) -> set[str]:
    """@brief 表达式引用的全部名字。

    @param node 表达式树。
    @return 名字集合。
    """
    if isinstance(node, Var):
        return {node.name}
    if isinstance(node, Const):
        return set()
    found: set[str] = set()
    for arg in node.args:
        found |= names_of(arg)
    return found


def evaluate(node: Node, names: Mapping[str, int]) -> int | bool:
    """@brief 求值一个表达式。

    @param node 表达式树。
    @param names 可用名字到整数的映射。
    @return 整数或谓词。
    @raises GeometryError 未知名字、除不尽、除零或参数非法。
    """
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Var):
        if node.name not in names:
            raise GeometryError(f"geometry expression uses undeclared name {node.name!r}; declared names are {sorted(names)}")
        value = names[node.name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise GeometryError(f"geometry name {node.name!r} is not an integer")
        return value
    values = [evaluate(arg, names) for arg in node.args]
    func = node.func
    if func in COMPARISONS:
        left, right = values
        if isinstance(left, bool) or isinstance(right, bool):
            raise GeometryError(f"{func}() needs integers")
        return {
            "gt": left > right,
            "ge": left >= right,
            "lt": left < right,
            "le": left <= right,
            "eq": left == right,
            "ne": left != right,
        }[func]
    if func in ("and", "or"):
        bools = []
        for index, value in enumerate(values):
            if not isinstance(value, bool):
                raise GeometryError(f"{func}() argument {index} must be a predicate")
            bools.append(value)
        return all(bools) if func == "and" else any(bools)
    if func == "not":
        if not isinstance(values[0], bool):
            raise GeometryError("not() needs a predicate")
        return not values[0]
    ints = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise GeometryError(f"{func}() argument {index} must be an integer")
        ints.append(value)
    if func == "+":
        return ints[0] + ints[1]
    if func == "-":
        return ints[0] - ints[1] if len(ints) == 2 else -ints[0]
    if func == "*":
        return ints[0] * ints[1]
    if func == "/":
        if ints[1] == 0 or ints[0] % ints[1] != 0:
            raise GeometryError(f"integer division {ints[0]}/{ints[1]} does not divide evenly")
        return ints[0] // ints[1]
    if func == "ceil_div":
        numerator, denominator = ints
        if denominator <= 0:
            raise GeometryError("ceil_div() needs a positive denominator")
        return -((-numerator) // denominator)
    if func == "trunc_div":
        numerator, denominator = ints
        if denominator == 0:
            raise GeometryError("trunc_div() divisor is zero; integer division has no value")
        quotient = abs(numerator) // abs(denominator)
        return -quotient if (numerator < 0) != (denominator < 0) else quotient
    if func == "rem":
        numerator, denominator = ints
        if denominator == 0:
            raise GeometryError("rem() divisor is zero; the remainder has no value")
        quotient = abs(numerator) // abs(denominator)
        if (numerator < 0) != (denominator < 0):
            quotient = -quotient
        return numerator - quotient * denominator
    if func == "min":
        return min(ints)
    if func == "max":
        return max(ints)
    raise GeometryError(f"unreachable geometry function {func!r}")  # pragma: no cover - parse_term closes the set


def evaluate_term(term: Any, names: Mapping[str, int], *, where: str, expect: str = INT) -> int | bool:
    """@brief 解析并求值一个术语，同时校验静态类型。

    @param term 术语。
    @param names 可用名字到整数的映射。
    @param where 诊断位置。
    @param expect 期望静态类型（``int``/``bool``）。
    @return 整数或谓词。
    @raises GeometryError 非法术语、类型不符或求值失败。
    """
    node = parse_term(term, where)
    found = kind_of(node)
    if found != expect:
        raise GeometryError(f"{where}: expected a {expect} expression, got {found}")
    return evaluate(node, names)


def _parse_text(text: str, where: str) -> Node:
    """@brief 解析一个文本表达式（与 ``taic-launch`` 相同的算术语法）。

    @param text 表达式文本。
    @param where 诊断位置。
    @return 表达式树。
    @raises GeometryError 词法/语法非法。
    """
    tokens = _tokenize(text, where)
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def eat(expected: str | None = None) -> str:
        nonlocal position
        if position >= len(tokens):
            raise GeometryError(f"{where}: expression `{text}` ended early")
        token = tokens[position]
        if expected is not None and token != expected:
            raise GeometryError(f"{where}: expression `{text}` expected {expected!r}, got {token!r}")
        position += 1
        return token

    def parse_expression() -> Node:
        left = parse_multiplicative()
        while peek() in ("+", "-"):
            operator = eat()
            left = Call(operator, (left, parse_multiplicative()))
        return left

    def parse_multiplicative() -> Node:
        left = parse_unary()
        while peek() in ("*", "/"):
            operator = eat()
            left = Call(operator, (left, parse_unary()))
        return left

    def parse_unary() -> Node:
        if peek() == "-":
            eat("-")
            return Call("-", (parse_unary(),))
        return parse_primary()

    def parse_primary() -> Node:
        token = peek()
        if token is None:
            raise GeometryError(f"{where}: expression `{text}` ended early")
        if token == "(":
            eat("(")
            node = parse_expression()
            eat(")")
            return node
        eat()
        if token.isdigit():
            return Const(int(token))
        if not _IDENTIFIER.fullmatch(token):
            raise GeometryError(f"{where}: expression `{text}` has an illegal token {token!r}")
        if peek() == "(":
            eat("(")
            args: list[Node] = []
            if peek() != ")":
                while True:
                    args.append(parse_expression())
                    if peek() == ",":
                        eat(",")
                        continue
                    break
            eat(")")
            return _call(token, args, where)
        return Var(token)

    node = parse_expression()
    if position != len(tokens):
        raise GeometryError(f"{where}: expression `{text}` has trailing input {tokens[position:]}")
    return node


def _tokenize(text: str, where: str) -> list[str]:
    """@brief 把文本表达式切成记号。

    @param text 表达式文本。
    @param where 诊断位置。
    @return 记号列表。
    @raises GeometryError 出现非法字符。
    """
    out: list[str] = []
    index, size = 0, len(text)
    while index < size:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "+-*/(),":
            out.append(char)
            index += 1
            continue
        if char.isdigit() and char.isascii():
            end = index + 1
            while end < size and text[end].isdigit() and text[end].isascii():
                end += 1
            out.append(text[index:end])
            index = end
            continue
        if char.isalpha() and char.isascii() or char == "_":
            end = index + 1
            while end < size and ((text[end].isalnum() and text[end].isascii()) or text[end] == "_"):
                end += 1
            out.append(text[index:end])
            index = end
            continue
        raise GeometryError(f"{where}: expression `{text}` has an illegal character {char!r}")
    return out


__all__ = [
    "ARITHMETIC_OPERATORS",
    "BOOLEAN_FUNCTIONS",
    "BOOL",
    "COMPARISONS",
    "FUNCTIONS",
    "INT",
    "Call",
    "Const",
    "Node",
    "Var",
    "evaluate",
    "evaluate_term",
    "kind_of",
    "names_of",
    "parse_term",
]
