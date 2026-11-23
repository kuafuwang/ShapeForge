"""Shared redaction for diagnostics that can reach logs, SQLite or the dashboard.

These strings come from provider errors, hmz stderr and occasionally a model's final answer.
Keep one policy here instead of letting each reader invent a narrower regex.

This is a **shape** filter for the credential forms this repository has registered (key/value
pairs including their JSON spelling, ``Authorization: Bearer``, and a known list of token
prefixes).  It is not a complete secret detector: an arbitrary secret that looks like an ordinary
identifier cannot be recognised by a pattern, and nothing here should be described as if it could.
For provider keys the harness also knows by value, ``taic/llm.py`` replaces those first.
"""

from __future__ import annotations

import re

#: Character cap for a ``stop_reason`` that is stored in the index and shown by ``brief``/``--wide``.
#: The longest real sample is 224 characters (an HTTP 402 provider error); the narrow ``stop_code``
#: stays at 24.  This is a display/store bound, not a redaction boundary.
STOP_REASON_CHARS = 400

#: The key names whose value is a credential.  ``token``/``secret`` cover the JSON spelling; the
#: compound names are listed explicitly so a match cannot start in the middle of
#: ``aws_secret_access_key`` and leave the rest behind.
_KEY_NAMES = (
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"private[_-]?key|aws[_-]?secret[_-]?access[_-]?key|authorization|password|secret|token"
)
#: Normalized suffixes that make a *structured* dictionary key hold a credential.  A key such as
#: ``api_key`` is not itself a secret shape, but its value is one whatever it looks like; the
#: same names are used here so the JSONL writer and the text filter agree.  Matching is on a
#: normalized suffix, so ``x-api-key``/``openai_api_key``/``hf_token`` are covered while
#: ``total_tokens``/``token_count`` are not.
_CREDENTIAL_KEY_SUFFIXES = (
    "api_key", "apikey", "access_token", "refresh_token", "auth_token", "client_secret",
    "private_key", "aws_secret_access_key", "authorization", "password", "secret", "token",
)
#: Quotes may be backslash-escaped any number of times: the same JSON spelling can have been
#: serialized once or several times before the text reaches the filter.
_ESCAPED_QUOTE = r"(?:\\*[\"'])?"
# Key/value forms: api_key=..., "api_key": "...", 'api-key' : '...', Authorization: ...
# The lookbehind allows the separators a real diagnostic puts in front of a key (``?``/``&``/
# ``-``/``/``) while still refusing to fire in the middle of a word (``total_tokens``).
_KV = re.compile(
    r"(?i)(?<![\w])(" + _ESCAPED_QUOTE + r"(?:" + _KEY_NAMES + r")" + _ESCAPED_QUOTE + r"\s*[:=]\s*" + _ESCAPED_QUOTE + r")"
    r"([^\s,;\"'\\]+)"
)
# "Authorization: Bearer VALUE" is common and the value often has punctuation.  Quotes and
# backslashes end the value so a redacted JSON/JSONL document keeps its delimiters.
_BEARER = re.compile(r"(?i)\b(bearer)\s+([^\s,;\"'\\]+)")
# Registered token prefixes.  Each one carries a minimum length so prose that merely names the
# prefix ("ghp_ is a GitHub prefix") is left alone; the JWT shape is the same one
# ``tcenv/worker_isolation/hmz_fence.py`` already uses.
_TOKENS = re.compile(
    r"\b(?:"
    r"sk-[A-Za-z0-9_\-]{12,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|glpat-[A-Za-z0-9_\-]{16,}"
    r"|AIza[0-9A-Za-z_\-]{30,}"
    r"|npm_[A-Za-z0-9]{30,}"
    r"|pypi-[A-Za-z0-9_\-]{20,}"
    r"|eyJ[A-Za-z0-9_\-]{12,}\."
    r")"
)


def redact_text(value, *, limit: int | None = None, one_line: bool = False) -> str:
    """Redact the registered credential shapes and optionally bound/collapse the text.

    @param value Text to clean; non-strings are treated as empty.
    @param limit Optional maximum character count after redaction.
    @param one_line Collapse whitespace before redacting (useful for JSONL/SQL cells).
    @return A redacted, optionally bounded string. The caller owns any ellipsis.
    """
    if not isinstance(value, str) or not value:
        return ""
    text = " ".join(value.split()) if one_line else value
    text = _BEARER.sub(lambda match: f"{match.group(1)} <redacted>", text)
    text = _KV.sub(lambda match: f"{match.group(1)}<redacted>", text)
    text = _TOKENS.sub("<redacted>", text)
    return text[:limit] if limit is not None else text


def redact_stop_reason(value) -> str:
    """@brief 一条停止原因的入库/展示形态：登记形状脱敏、压成一行、再限长。

    A stop reason is free text from a provider or an exception, so it can carry the same credential
    shapes every other diagnostic column is cleaned of.  The narrow ``stop_code`` is derived from
    this cleaned text, so a reason that starts with a token cannot smuggle the token into the
    ``<=24`` character column either.

    @param value 原始停止原因，可为 ``None``。
    @return 脱敏并限长后的单行文本。
    """
    return redact_text(value, limit=STOP_REASON_CHARS, one_line=True)


def credential_key(name) -> bool:
    """@brief 判断一个结构化键名是否“它的值就是凭据”。

    A JSON object such as ``{"api_key": "some-readable-word"}`` carries a secret that no text
    shape can recognise from the value alone.  The key itself is the context, so callers that
    serialise a structure (``Transcript``) redact that value wholesale.  Matching is on a
    normalized suffix so ``x-api-key``/``openai_api_key``/``hf_token`` are covered while
    ``total_tokens``/``token_count`` are not.

    @param name 字典键名，可以是任意对象（非字符串一律为 False）。
    @return 该键的值应当整体脱敏时为 True。
    """
    if not isinstance(name, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not normalized:
        return False
    return any(
        normalized == suffix or normalized.endswith("_" + suffix)
        for suffix in _CREDENTIAL_KEY_SUFFIXES
    )
