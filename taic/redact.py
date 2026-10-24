"""Shared redaction for diagnostics that can reach logs, SQLite or the dashboard.

These strings come from provider errors, hmz stderr and occasionally a model's final answer.
Keep one policy here instead of letting each reader invent a narrower regex.
"""

from __future__ import annotations

import re

# Key/value forms: api_key=..., Authorization: ..., password: ...
_KV = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"authorization|password|secret)\b\s*[:=]\s*)([^\s,;]+)"
)
# "Authorization: Bearer VALUE" is common and the value often has punctuation.
_BEARER = re.compile(r"(?i)\b(bearer)\s+([^\s,;]+)")
# Provider keys without a key= prefix.
_SK = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")


def redact_text(value, *, limit: int | None = None, one_line: bool = False) -> str:
    """Redact common credential shapes and optionally bound/collapse the text.

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
    text = _SK.sub("<redacted>", text)
    return text[:limit] if limit is not None else text
