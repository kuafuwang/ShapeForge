"""Structured term grammar shared by lowerings and OpSpec evaluation domains.

A *term* is a closed JSON value, never free text:

* ``3``                                       integer constant;
* ``{"symbol": "M", "factor": 1}``            symbol value floored by a positive factor;
* ``{"product": ["N", "OH", "OW"]}``          product of symbols;
* ``{"ceil_div": {"numerator": t, "denominator": t}}``;
* ``{"floor_div": {"numerator": t, "denominator": t}}``;
* ``{"max": [t, t]}``                         two-term maximum.

The lowering registry uses it for a profile's ``grid`` terms and the OpSpec evaluation
domain uses it for a size-dependent ``ieee`` injection cap (``max(8, floor(M_TOTAL/100))``).
Both run through the same validator and evaluator, so neither can hide an expression the
other cannot check.
"""

from __future__ import annotations

from typing import Any, Mapping


class TermError(ValueError):
    """@brief 词非法：形态未知、字段多余、引用未声明符号或除法非法。"""


def normalize_term(term: Any, *, where: str) -> None:
    """@brief 校验一个结构化词的文法。

    @param term 待校验的词。
    @param where 诊断位置。
    @raises TermError 形态未知或字段多余。
    """
    if isinstance(term, int) and not isinstance(term, bool):
        if term < 0:
            raise TermError(f"{where}: {term!r} must not be negative")
        return
    if not isinstance(term, Mapping):
        raise TermError(f"{where}: unsupported term {term!r}")
    keys = set(term)
    if keys == {"symbol"} or keys == {"symbol", "factor"}:
        if not isinstance(term["symbol"], str) or not term["symbol"]:
            raise TermError(f"{where}: symbol term needs a non-empty name")
        if int(term.get("factor", 1)) <= 0:
            raise TermError(f"{where}: factor must be positive")
        return
    if keys == {"product"}:
        names = term["product"]
        if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) and n for n in names):
            raise TermError(f"{where}: product needs a non-empty list of symbol names")
        return
    if keys in ({"ceil_div"}, {"floor_div"}):
        inner = term[next(iter(keys))]
        if not isinstance(inner, Mapping) or set(inner) != {"numerator", "denominator"}:
            raise TermError(f"{where}: division needs numerator and denominator")
        normalize_term(inner["numerator"], where=f"{where}.numerator")
        normalize_term(inner["denominator"], where=f"{where}.denominator")
        return
    if keys == {"max"}:
        pair = term["max"]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TermError(f"{where}: max needs exactly two terms")
        normalize_term(pair[0], where=f"{where}.max[0]")
        normalize_term(pair[1], where=f"{where}.max[1]")
        return
    raise TermError(f"{where}: unsupported term {term!r}")


def term_document(term: Any) -> Any:
    """@brief 把词汇规范化成 JSON 闭集表示（进入身份文档）。

    @param term 词。
    @return int 或只含 symbol/factor/product/ceil_div/floor_div/max 的字典。
    """
    if isinstance(term, int):
        return int(term)
    if "symbol" in term:
        return {"symbol": str(term["symbol"]), "factor": int(term.get("factor", 1))}
    if "product" in term:
        return {"product": [str(name) for name in term["product"]]}
    if "max" in term:
        return {"max": [term_document(term["max"][0]), term_document(term["max"][1])]}
    key = "ceil_div" if "ceil_div" in term else "floor_div"
    return {
        key: {
            "numerator": term_document(term[key]["numerator"]),
            "denominator": term_document(term[key]["denominator"]),
        }
    }


def eval_term(term: Any, symbols: Mapping[str, int], *, where: str) -> int:
    """@brief 在符号值上求值一个词。

    @param term 词。
    @param symbols 已解析符号值。
    @param where 诊断位置。
    @return 具体整数。
    @raises TermError 引用未声明符号、除零或非法 factor。
    """
    if isinstance(term, int):
        return int(term)
    if isinstance(term, Mapping) and "max" in term:
        return max(
            eval_term(term["max"][0], symbols, where=f"{where}.max[0]"),
            eval_term(term["max"][1], symbols, where=f"{where}.max[1]"),
        )
    if "symbol" in term:
        name = str(term["symbol"])
        if name not in symbols:
            raise TermError(f"{where}: symbol {name!r} is not in the semantic instance")
        factor = int(term.get("factor", 1))
        if factor <= 0:
            raise TermError(f"{where}: factor must be positive")
        return int(symbols[name]) // factor
    if "product" in term:
        value = 1
        for name in term["product"]:
            if str(name) not in symbols:
                raise TermError(f"{where}: symbol {name!r} is not in the semantic instance")
            value *= int(symbols[str(name)])
        return value
    key = "ceil_div" if "ceil_div" in term else "floor_div"
    numerator = eval_term(term[key]["numerator"], symbols, where=f"{where}.{key}.numerator")
    denominator = eval_term(term[key]["denominator"], symbols, where=f"{where}.{key}.denominator")
    if denominator <= 0:
        raise TermError(f"{where}: denominator must be positive, got {denominator}")
    if key == "ceil_div":
        return -(-numerator // denominator)
    return numerator // denominator


__all__ = ["TermError", "eval_term", "normalize_term", "term_document"]
