"""Materialise the OpSpec document of one ``sem_int_<seed>`` task.

The document is the *only* semantic contract of the generated task: it carries the closed
statement program **and its numeric rule table** (wrap set, hash/loop shifts, mod/index
masks, shift domains), the per-block reduction size, the int32 tensor domain, the exact
tolerance mode, the two sizes and the work accounting.  The registered Metal profile
supplies the launch facts (threads/warps and the matching ``BLOCK`` constant, cross-checked
at assembly) and the runtime scalar's value comes from the ``SCALAR_K`` attr, so neither is
duplicated here.
"""

from __future__ import annotations

from . import program

#: Size domain of the integer family (the frozen implementation's ``INT_N``).
INT_N = {"paper": 1 << 16, "small": 1 << 12}

#: Index derivation used by the generated prompt's final line and by the reference.  The mask is
#: *not* written here: it is interpolated from the program document's rule table, so the formula
#: has no second copy of it that a stale view could disagree with.
FORMULA_TEMPLATE = (
    "out[i] = (f0 ^ f1 ^ f2 ^ f3) + (idx[i] & {index_mask}), where f0..f3 are the program's final block "
    "values; every statement is int32 two's-complement with wraparound"
)


def formula_for(rules: dict) -> str:
    """@brief 由程序规则表渲染题面公式（``index_mask`` 只有一个来源）。

    @param rules 程序文档的 ``rules`` 块。
    @return 公式文本。
    """
    return FORMULA_TEMPLATE.format(index_mask=int(rules["index_mask"]))


def opspec_document(seed: int, size: str) -> dict:
    """@brief 由 ``(seed, size)`` 物化一份 OpSpec 文档。

    @param seed 题目名里的整数种子。
    @param size 尺寸档位（``paper``/``small``）。
    @return 通过 ``opspec_from_document`` 校验的文档。
    @raises KeyError 未登记的 size。
    """
    n = INT_N[size]
    extra = {"doc": "runtime scalar argument K"}
    program_document = program.program_document(seed, block=program.BLOCK)
    return {
        "spec_version": 1,
        "op_name": f"sem_int_{seed}",
        "category": "integer_program",
        "formula": formula_for(program_document["rules"]),
        "uses": ["SCALAR_K"],
        "type_vars": {"T": ["i32"]},
        "symbols": {"N": {"min": 1}},
        "tensors": {
            "x": {
                "role": "input",
                "dims": ["N"],
                "dtype": "T",
                "layout": "contiguous",
                "strides": ["1"],
                "space": "global",
                "access": "read",
            },
            "y": {
                "role": "input",
                "dims": ["N"],
                "dtype": "T",
                "layout": "contiguous",
                "strides": ["1"],
                "space": "global",
                "access": "read",
            },
            "out": {
                "role": "output",
                "dims": ["N"],
                "dtype": "T",
                "layout": "contiguous",
                "strides": ["1"],
                "space": "global",
                "access": "write",
            },
        },
        "numerics": {
            "acc_dtype": "i32",
            "rounding": "exact_integer",
            "tolerances": {"i32": {"rtol": 0.0, "atol_rel": 0.0}},
            "exact": True,
        },
        "attrs": {
            "SCALAR_K": {
                "dtype": "int",
                "default": program.SCALAR_K,
                "doc": extra["doc"],
            }
        },
        "cases": {"names": ["random", "small", "edge"]},
        "sizes": {
            "paper": {"N": INT_N["paper"], "usage": ["golden", "benchmark"]},
            "small": {"N": INT_N["small"], "usage": ["golden", "benchmark"]},
        },
        "spec_params": ["N"],
        "work_accounting": {
            "flops": {"terms": [{"literal": 0}]},
            "min_bytes": {"terms": [{"tensor": "x"}, {"tensor": "y"}, {"tensor": "out"}]},
        },
        "program": program_document,
    }


__all__ = ["FORMULA_TEMPLATE", "INT_N", "formula_for", "opspec_document"]


# ---------------------------------------------------------------------------
# Registered domains (B4: the dynamic assembly entry ``task.py`` retired with the v1 registry;
# these are the *authoring tool's* domains, kept here so the generator stays self-describing).
INT_PREFIX = "sem_int_"

#: Explicit dtype aliases: integer tasks materialise as i32; the requested dtype only decides
#: which spellings are accepted.  The frozen implementation ignored the request and always
#: materialised i32; accepting ``fp32`` was an accident of that path, not a declared alias.
DTYPE_ALIASES = {"fp16": "i32", "bf16": "i32", "i32": "i32"}

#: Registered seed domain: non-negative, at most 18 ASCII decimal digits.  The bound keeps
#: ``op_name``/task names bounded; it is checked on the *spelling*, not only the value, so a
#: longer zero-padded spelling is out of domain even when it parses to the same integer.
SEED_DIGITS_MAX = 18


def parse_seed(name: str) -> int:
    """@brief 从题目名里取种子并校验登记的种子域。

    @param name 形如 ``sem_int_<seed>`` 的题目名。
    @return ``0 <= seed < 10**18`` 的种子。
    @raises ValueError 名字不以 ``sem_int_`` 开头、后缀不是十进制整数或超出种子域。
    """
    if not name.startswith(INT_PREFIX):
        raise ValueError(f"not an integer program task: {name!r}")
    suffix = name[len(INT_PREFIX):]
    # ASCII only, explicitly: ``str.isdigit`` is true for Unicode decimal digits
    # (``"١".isdigit()``), so it would accept spellings this registered domain never declared.
    if not suffix or any(character not in "0123456789" for character in suffix):
        raise ValueError(
            f"{name!r}: the seed must be a non-negative ASCII decimal integer (registered seed domain)"
        )
    if len(suffix) > SEED_DIGITS_MAX:
        raise ValueError(
            f"{name!r}: the seed spelling must be at most {SEED_DIGITS_MAX} ASCII decimal digits "
            f"(registered seed domain), got {len(suffix)}"
        )
    seed = int(suffix)
    if seed >= 10**SEED_DIGITS_MAX:
        raise ValueError(f"{name!r}: the seed must be below 10**{SEED_DIGITS_MAX} (registered seed domain)")
    return seed
