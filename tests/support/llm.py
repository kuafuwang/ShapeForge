"""Small helpers for scripted LLM tests."""

from __future__ import annotations

try:  # the SDK's HTTP backend differs between versions
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx


def status_error(cls, code, headers=None, message="boom"):
    """@brief 构造一个携带指定 HTTP 状态码与响应头的 LLM SDK 异常，用于脚本化错误测试。

    @param cls 要实例化的 SDK 异常类。
    @param code HTTP 状态码。
    @param headers 可选的响应头。
    @param message 异常消息，作为响应体置空时的兜底文本。
    @return 该异常类的实例。
    """
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    return cls(message, response=httpx.Response(code, request=req, headers=headers or {}), body=None)
