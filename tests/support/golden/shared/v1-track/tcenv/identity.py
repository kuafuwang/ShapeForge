"""Canonical identities: value encoding, strict JSON loading and RFC 8785 digests.

Plan §3.1 freezes one rule for every new identity document in this repository:

* the document is first *normalised* (``canonical_value``) so that it contains only
  the JSON subset ``tcenv.remote.contracts.canonical_digest_document`` accepts;
* the normalised document is then serialised and hashed by that existing helper —
  no second RFC 8785 serialiser is allowed to appear here (D40).

The normalisation rules are exactly the plan's, and nothing else:

* object keys must be non-empty ASCII strings;
* integers stay integers and must be inside the IEEE-754 safe range;
* every non-integer number becomes its shortest round-trip decimal *string*
  (``repr``), so ``1e-06`` and ``0.000001`` are one value while ``0.0`` and
  ``-0.0`` are two; NaN and ±Inf are refused;
* strings are hashed by codepoint — no Unicode normalisation, so NFC and NFD
  spellings of the same visible text are different identities;
* bools and null pass through unchanged.

``load_strict_json`` is the only loader identity documents may use: it refuses
duplicate keys instead of letting "the last value wins" decide an identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .remote.contracts import MAX_CANONICAL_BYTES, ContractError, canonical_digest_document

#: Largest integer that survives an IEEE-754 double round trip (``2**53 - 1``).
MAX_SAFE_INTEGER = 9_007_199_254_740_991


class IdentityError(ValueError):
    """@brief 身份文档结构非法：重复键、未知类型、非有限浮点或越界整数。"""


def float_text(value: float) -> str:
    """@brief 把有限浮点编码成最短往返十进制字符串（§3.1 的唯一 float 编码）。

    @param value 有限浮点。
    @return ``repr(value)``，且已检查 ``float(text) == value`` 与 ``repr(float(text)) == text``。
    @raises IdentityError 值不是有限浮点，或往返检查失败。
    """
    if not isinstance(value, float) or isinstance(value, bool):
        raise IdentityError("float_text needs a float")
    if not math.isfinite(value):
        raise IdentityError("NaN and ±Inf are not canonical values")
    text = repr(value)
    if float(text) != value or repr(float(text)) != text:
        raise IdentityError(f"float {text!r} does not round trip")
    return text


def canonical_value(value: Any) -> Any:
    """@brief 递归规范化一个身份文档的值，使其可送入既有 canonicaliser。

    @param value 任意 Python 值。
    @return 只含 null/bool/int/str/array/object 的等价文档。
    @raises IdentityError 出现重复以外的非法类型、非 ASCII 键、非有限浮点或越界整数。
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            raise IdentityError("canonical integers must be IEEE-754 safe")
        return value
    if isinstance(value, float):
        return float_text(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise IdentityError("canonical field names must be non-empty strings")
            if not key.isascii():
                raise IdentityError(f"canonical field name {key!r} must be ASCII")
            out[key] = canonical_value(item)
        return out
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    raise IdentityError(f"unsupported canonical value: {type(value).__name__}")


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    """@brief 规范化后取 RFC 8785 字节。

    @param document 身份文档。
    @return 规范化文档的 canonical 字节串。
    """
    return canonical_digest_document(canonical_value(document))


def fingerprint(document: Mapping[str, Any]) -> str:
    """@brief 计算身份文档的 64 位小写十六进制 SHA-256。

    @param document 身份文档。
    @return ``SHA256(RFC8785(canonical_value(document)))`` 的十六进制表示。
    """
    digest = hashlib.sha256(canonical_bytes(document)).hexdigest()
    assert len(digest) == 64
    return digest


def load_strict_json(path: str | Path) -> Any:
    """@brief 读取一个身份/语义 JSON 文件，拒绝重复键与非有限字面量。

    @param path JSON 文件路径。
    @return 解析后的 Python 值。
    @raises IdentityError 出现重复键、``NaN``/``Infinity`` 字面量、非 UTF-8 或非 JSON。
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IdentityError(f"{path}: not UTF-8: {exc}") from exc
    return loads_strict_json(text, origin=str(path))


def loads_strict_json(text: str, *, origin: str = "<string>") -> Any:
    """@brief 解析身份/语义 JSON 文本，拒绝重复键与非有限字面量。

    @param text JSON 文本。
    @param origin 出错信息里使用的来源名。
    @return 解析后的 Python 值。
    @raises IdentityError 结构非法。
    """
    from .schema import SchemaError, loads_strict

    try:
        return loads_strict(text, what=origin, reject_constants=True)
    except SchemaError as exc:
        raise IdentityError(str(exc).split("; needs human inspection")[0]) from exc
    except IdentityError:
        raise
    except (ValueError, RecursionError) as exc:
        raise IdentityError(f"{origin}: invalid JSON: {exc}") from exc


def is_hex64(value: Any) -> bool:
    """@brief 判断一个值是否是身份摘要的规范写法（64 位小写十六进制）。

    @param value 待检查的值。
    @return 形如 ``[0-9a-f]{64}`` 时为 True。
    """
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


__all__ = [
    "MAX_CANONICAL_BYTES",
    "MAX_SAFE_INTEGER",
    "ContractError",
    "IdentityError",
    "canonical_bytes",
    "canonical_value",
    "fingerprint",
    "float_text",
    "is_hex64",
    "load_strict_json",
    "loads_strict_json",
]
