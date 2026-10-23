"""Small helpers for scripted LLM tests."""

from __future__ import annotations

try:  # the SDK's HTTP backend differs between versions
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx


def status_error(cls, code, headers=None, message="boom"):
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    return cls(message, response=httpx.Response(code, request=req, headers=headers or {}), body=None)
