"""Canonical identity encoding (plan §3.1): one value encoding, one canonicaliser.

CPU only.  The R0 baseline package froze seven canonical documents plus numeric
encoding examples with an *independent* stdlib serializer; this module recomputes them
through the repository helper, so "the helper agrees with the frozen vectors" is a real
check rather than a self-consistency tautology.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcenv.identity import (
    IdentityError,
    canonical_value,
    fingerprint,
    float_text,
    loads_strict_json,
)
from tcenv.remote.contracts import ContractError, canonical_digest_document

R0 = Path(__file__).resolve().parents[2] / "docs" / "架构" / "构建执行与证据语义升级-R0"


def _r0_json(relative: str):
    """@brief 读取 R0 基线包里的一份 JSON。

    @param relative 相对 R0 目录的路径。
    @return 解析结果。
    """
    return json.loads((R0 / relative).read_text(encoding="utf-8"))


def test_the_r0_identity_vectors_recompute_with_the_repository_helper():
    """@brief 7 个 R0 canonical 向量用仓库 helper 重算后与冻结摘要一致。"""
    recorded = _r0_json("vectors/digests.json")
    assert len(recorded) == 7
    for name, entry in recorded.items():
        document = _r0_json(entry["document"])
        digest = fingerprint(document)
        assert digest == entry["digest"], name
        assert len(digest) == 64 and digest == digest.lower()


def test_the_r0_numeric_encoding_examples_round_trip():
    """@brief R0 的 float/integer 编码样例逐条可复算（1e-06 与 0.000001 同一值）。"""
    examples = _r0_json("vectors/encoding_examples.json")
    for example in examples["float_examples"]:
        value = float(example["literal"])
        assert float_text(value) == example["canonical"], example["literal"]
        assert float(example["canonical"]) == value
    for example in examples["integer_examples"]:
        assert canonical_value(int(example["literal"])) == example["canonical_value"]


def test_key_order_and_whitespace_do_not_change_the_identity():
    """@brief 同一文档只改键顺序或 JSON 空白时摘要不变。"""
    left = loads_strict_json('{"a": 1, "b": [2, {"c": "x"}]}')
    right = loads_strict_json('{\n  "b": [2, {"c": "x"}],\n  "a": 1\n}')
    assert left != right or list(left) != list(right)
    assert fingerprint(left) == fingerprint(right)


def test_a_float_literal_has_exactly_one_canonical_spelling():
    """@brief ``1e-06`` 与 ``0.000001`` 是同一个 canonical 值；``0.002`` 不是 ``0.0020``。"""
    assert float_text(1e-06) == "1e-06"
    assert float_text(0.002) == "0.002"
    assert fingerprint({"x": 1e-06}) == fingerprint({"x": 0.000001})
    assert fingerprint({"x": 0.002}) == fingerprint({"x": float("0.0020")})


def test_negative_zero_and_integer_valued_floats_are_distinct():
    """@brief ``0.0``/``-0.0`` 与 ``1``/``1.0``、``16777216``/``16777216.0`` 都是不同身份。"""
    pairs = [
        ({"x": 0.0}, {"x": -0.0}),
        ({"x": 1}, {"x": 1.0}),
        ({"x": 16777216}, {"x": 16777216.0}),
    ]
    for left, right in pairs:
        assert canonical_value(left) != canonical_value(right)
        assert fingerprint(left) != fingerprint(right)
    assert canonical_value({"x": -0.0}) == {"x": "-0.0"}


def test_non_finite_and_unsafe_numbers_are_refused():
    """@brief NaN/±Inf 与超出 IEEE-754 安全范围的整数都不是合法 canonical 值。"""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(IdentityError):
            canonical_value({"x": bad})
    with pytest.raises(IdentityError):
        canonical_value({"x": 2**53})
    with pytest.raises(IdentityError):
        loads_strict_json('{"x": NaN}')
    with pytest.raises(IdentityError):
        loads_strict_json('{"x": Infinity}')


def test_unicode_text_is_not_normalised():
    """@brief 同一可见文本的 NFC/NFD 两种写法是不同身份（不做 Unicode 归一化）。"""
    nfc = {"doc": "caf\u00e9"}
    nfd = {"doc": "cafe\u0301"}
    assert nfc["doc"] != nfd["doc"]  # equal on screen, different codepoints
    assert fingerprint(nfc) != fingerprint(nfd)


def test_duplicate_keys_and_non_ascii_keys_are_refused():
    """@brief 重复键由严格 loader 拒绝；非 ASCII 键不能进入身份文档。"""
    with pytest.raises(IdentityError):
        loads_strict_json('{"a": 1, "a": 2}')
    with pytest.raises(IdentityError):
        loads_strict_json('{"outer": {"b": 1, "b": 2}}')
    with pytest.raises(IdentityError):
        canonical_value({"é": 1})
    with pytest.raises(IdentityError):
        canonical_value({"": 1})


def test_the_existing_canonicaliser_still_refuses_a_bare_float():
    """@brief N-12：把未规范化的 float 直接送进既有 canonicaliser 必须失败。"""
    with pytest.raises(ContractError):
        canonical_digest_document({"x": 1.0})
    with pytest.raises(ContractError):
        canonical_digest_document({"x": 0.0})
    canonical_digest_document({"x": "1.0"})
