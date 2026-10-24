"""Credential-shaped diagnostic text must not survive any of the readers we added."""

from __future__ import annotations

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
