"""``display.json``: representation-only overrides for a v2 task statement.

The display layer may change *how* a task is described, never *what* is verified.  Its
closed schema contains ``description``/``spec_text``/``formula`` and nothing else, so a
display file cannot carry cases, tolerances, tensors, an ABI or launch facts.

``spec_text`` is reserved for the G0-frozen ``sem_*`` fixtures whose prompt text is
deliberately different from the code they must implement.  A normal operator may not
replace its statement: that would turn the author-facing text into a second, unverifiable
semantic source, so it is refused as ``semantic_unresolved``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..identity import IdentityError, load_strict_json
from ..opspec.validator import SchemaFailure, validate
from .errors import SemanticUnresolved, TaskContractError

DISPLAY_SCHEMA_VERSION = 1
DISPLAY_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "display.v1.schema.json"
DISPLAY_FILE = "display.json"

#: The four fixed semantic-recovery fixtures frozen by G0.
_FIXED_SEM = (
    "sem_gemm_neg",
    "sem_relu_gelu_comment",
    "sem_gelu_named_relu",
    "sem_attention_odd_scale",
)
#: The dynamic G0 integer family (``sem_int_<seed>``) is a frozen fixture family.
_SEM_INT_PREFIX = "sem_int_"
#: The registered dynamic family accepts at most this many ASCII decimal seed digits.
SEM_INT_SEED_DIGITS_MAX = 18
_SEM_PREFIXES = (_SEM_INT_PREFIX,)
#: Public allowlist: the only names whose statement text may be overridden.  The
#: ``sem_int_`` entry is a family prefix; :func:`spec_text_override_allowed` validates the
#: exact registered suffix grammar rather than accepting an arbitrary non-empty suffix.
G0_SEM_SPEC_TEXT_ALLOWLIST: tuple[str, ...] = _FIXED_SEM + _SEM_PREFIXES


def _is_registered_sem_int_name(op_name: str) -> bool:
    """@brief 判断名字是否是 ``sem_int_<seed>`` 的登记拼写。

    This mirrors the registered family's syntax without importing ``tasks``: one to
    ``SEM_INT_SEED_DIGITS_MAX`` ASCII decimal digits, no sign/exponent/whitespace/Unicode
    digits.  Bounded spelling length also bounds the task name; leading zeroes are part of
    the registered spelling (the frozen family accepts ``sem_int_007``).

    @param op_name 算子名。
    @return 属于登记动态族时为 True。
    """
    if not op_name.startswith(_SEM_INT_PREFIX):
        return False
    suffix = op_name[len(_SEM_INT_PREFIX) :]
    return 1 <= len(suffix) <= SEM_INT_SEED_DIGITS_MAX and all(character in "0123456789" for character in suffix)


def spec_text_override_allowed(op_name: str) -> bool:
    """@brief 该算子是否属于 G0 冻结的 ``sem_*`` 白名单（允许题面文本覆盖）。

    @param op_name 算子名。
    @return 允许时为 True。
    """
    if op_name in _FIXED_SEM:
        return True
    return _is_registered_sem_int_name(op_name)


def _normalise_formula(text: str) -> str:
    """@brief 规范化公式文本，只用于表示一致性比较。

    @param text 公式文本。
    @return 去首尾空白并压缩内部空白后的文本。
    """
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class DisplayOverrides:
    """@brief 通过校验的表示层覆盖（空串表示未覆盖）。"""

    description: str = ""
    spec_text: str = ""
    formula: str = ""

    @property
    def is_empty(self) -> bool:
        """@brief 是否没有任何覆盖。

        @return 三个字段都为空时为 True。
        """
        return not (self.description or self.spec_text or self.formula)

    def document(self) -> dict[str, str]:
        """@brief 只含已声明覆盖的 JSON 文档。

        @return 字段字典。
        """
        return {
            key: value
            for key, value in (
                ("description", self.description),
                ("spec_text", self.spec_text),
                ("formula", self.formula),
            )
            if value
        }


def display_from_document(document: Mapping[str, Any], *, op_name: str, opspec_formula: str = "") -> DisplayOverrides:
    """@brief 校验一份 display 文档并返回覆盖对象。

    @param document display 文档。
    @param op_name 该算子名（用于 whitelist）。
    @param opspec_formula OpSpec 的摘要公式（用于表示一致性比较）。
    @return 覆盖对象。
    @raises SemanticUnresolved 普通算子覆盖题面，或冗余公式与 OpSpec 不一致。
    @raises TaskContractError 文档结构非法。
    """
    try:
        schema = load_strict_json(DISPLAY_SCHEMA_PATH)
    except IdentityError as exc:  # pragma: no cover - schema is shipped
        raise TaskContractError(f"registered display schema is unreadable: {exc}") from exc
    try:
        validate(document, schema)
    except SchemaFailure as exc:
        raise TaskContractError(f"{DISPLAY_FILE}: {exc}") from exc

    spec_text = str(document.get("spec_text", ""))
    if spec_text and not spec_text_override_allowed(op_name):
        raise SemanticUnresolved(
            f"{DISPLAY_FILE}: {op_name!r} may not override its statement text; display.spec_text is "
            f"limited to the G0-frozen sem_* allowlist {list(G0_SEM_SPEC_TEXT_ALLOWLIST)}"
        )
    formula = str(document.get("formula", ""))
    if formula and opspec_formula and _normalise_formula(formula) != _normalise_formula(opspec_formula):
        raise SemanticUnresolved(
            f"{DISPLAY_FILE}: display.formula disagrees with the OpSpec summary formula; the two "
            "representations of one task must agree, and neither side is auto-selected"
        )
    return DisplayOverrides(
        description=str(document.get("description", "")),
        spec_text=spec_text,
        formula=formula,
    )


def load_display(
    task_dir: str | Path,
    *,
    op_name: str,
    opspec_formula: str = "",
) -> DisplayOverrides:
    """@brief 读取可选 ``display.json``；不存在时返回空覆盖。

    @param task_dir 任务目录。
    @param op_name 算子名。
    @param opspec_formula OpSpec 摘要公式。
    @return 覆盖对象。
    @raises SemanticUnresolved 表示冲突。
    @raises TaskContractError 文档非法。
    """
    path = Path(task_dir) / DISPLAY_FILE
    if not path.exists():
        return DisplayOverrides()
    try:
        document = load_strict_json(path)
    except IdentityError as exc:
        raise TaskContractError(f"{path}: {exc}") from exc
    return display_from_document(document, op_name=op_name, opspec_formula=opspec_formula)


__all__ = [
    "DISPLAY_FILE",
    "DISPLAY_SCHEMA_PATH",
    "DISPLAY_SCHEMA_VERSION",
    "DisplayOverrides",
    "G0_SEM_SPEC_TEXT_ALLOWLIST",
    "SEM_INT_SEED_DIGITS_MAX",
    "display_from_document",
    "load_display",
    "spec_text_override_allowed",
]
