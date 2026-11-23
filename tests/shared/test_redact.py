"""Credential-shaped diagnostic text must not survive any of the readers we added."""

from __future__ import annotations

import json

from observe.detail import _clip
from observe.index import _diagnostic
from taic.brief import _text
from taic.redact import redact_text


TEXT = (
    "api_key=API-SECRET-123456\n"
    "Authorization: Bearer BEARER-SECRET-123456\n"
    "password=hunter2 refresh_token=REFRESH-SECRET-123456\n"
    "sk-abcdefghijklmnop"
)

#: The JSON spelling and the token prefixes the audit found unhandled.  None of these may survive.
JSON_AND_PREFIXES = (
    '{"api_key": "JSON-SECRET-123", "token": "ghp_abcdefghijklmnop"}',
    "{\"api_key\":\"COMPACT-SECRET-123\"}",
    "'api-key' : 'SPACED-SECRET-123'",
    '{"password":"PASSWORD-SECRET-123"}',
    "?token=QUERY-SECRET-123",
    "github_pat_ABCDEFGHIJKLMNOPQRST",
    "xoxb-1234567890abcdef",
    "AKIAABCDEFGHIJKLMNOP",
    "eyJhbGciOiJIUzI1NiJ9.payload",
    r'{\"api_key\": \"ESCAPED-SECRET-123\"}',
    r'{\\\"token\\\": \\\"DOUBLE-ESCAPED-SECRET-123\\\"}',
)


def test_shared_redaction_covers_key_value_and_bearer_forms():
    """@brief 验证共享脱敏覆盖键值对、Authorization Bearer、password/refresh_token 与 sk- 前缀密钥，并按 limit 截断。
    """
    cleaned = redact_text(TEXT)
    for secret in ("API-SECRET", "BEARER-SECRET", "hunter2", "REFRESH-SECRET", "abcdefghijklmnop"):
        assert secret not in cleaned
    assert cleaned.count("<redacted>") >= 4
    assert len(redact_text("x" * 100, limit=12)) == 12


def test_brief_sqlite_and_detail_readers_use_the_same_policy():
    """@brief 验证 brief、SQLite 索引与 detail 三个读取器沿用同一脱敏策略，且索引输出压平换行、brief 对干净长文本不返回内容。
    """
    short = "api_key=API-SECRET Authorization: Bearer BEARER-SECRET"
    for cleaned in (_text(short), _diagnostic(TEXT), _clip(TEXT, 500)):
        assert cleaned is not None
        for secret in ("API-SECRET", "BEARER-SECRET", "hunter2", "REFRESH-SECRET", "abcdefghijklmnop"):
            assert secret not in cleaned
        assert "<redacted>" in cleaned
    assert "\n" not in _diagnostic(TEXT)
    assert _text("x" * 100) is None


def test_json_keys_and_registered_token_prefixes_are_redacted():
    """@brief JSON 键形态与登记过的 token 前缀都要清掉，包括引号、空白与紧凑写法。"""
    for text in JSON_AND_PREFIXES:
        cleaned = redact_text(text)
        for needle in (
            "JSON-SECRET-123", "COMPACT-SECRET-123", "SPACED-SECRET-123", "PASSWORD-SECRET-123",
            "QUERY-SECRET-123", "ghp_abcdefghijklmnop", "github_pat_", "xoxb-", "AKIAABCDEFGHIJKLMNOP",
            "eyJhbGciOiJIUzI1NiJ9", "ESCAPED-SECRET-123", "DOUBLE-ESCAPED-SECRET-123",
        ):
            assert needle not in cleaned, (text, cleaned)
    # 幂等：脱敏过的文本再过一遍不变，``"<redacted>"`` 形态不会被二次切坏。
    once = redact_text(JSON_AND_PREFIXES[0])
    assert redact_text(once) == once and '"<redacted>"' in once


def test_the_widened_patterns_do_not_eat_ordinary_prose():
    """@brief 放宽后仍不能把普通文本当成密钥：只认键值分隔符与登记前缀。"""
    for text in (
        "the secret sauce is extra tempo",
        "token budget is 20 per session",
        '{"status": "ok", "total_tokens": 500}',
        "converged (12 rounds without improvement)",
        "api_key is the name of the setting, not its value",
    ):
        assert redact_text(text) == text
    # 既有语义不能回退：短值仍然被当成键值对待。
    assert "hunter2" not in redact_text("password=hunter2")
    assert "OPAQUE123" not in redact_text("?access_token=OPAQUE123")


def test_redaction_at_a_json_boundary_keeps_the_document_parseable():
    """@brief 在序列化边界脱敏也不能吃掉 JSON 的引号，转义一份的键值同样要清掉。"""
    documents = (
        {"content": "Authorization: Bearer BEARER-SECRET-123"},
        {"content": '{"api_key": "NESTED-SECRET-123"}'},
        {"content": r'{\"api_key\": \"ESCAPED-SECRET-123\"}'},
        {"content": json.dumps(json.dumps({"api_key": "DEEP-SECRET-123"}))},
        {"Authorization": "Bearer BEARER-JSON-SECRET-123"},
    )
    for document in documents:
        cleaned = redact_text(json.dumps(document))
        parsed = json.loads(cleaned)  # still valid JSON or JSONL
        text = json.dumps(parsed)
        for needle in (
            "BEARER-SECRET-123", "NESTED-SECRET-123", "ESCAPED-SECRET-123", "DEEP-SECRET-123",
            "BEARER-JSON-SECRET-123",
        ):
            assert needle not in text, (document, cleaned)
        assert "<redacted>" in text
