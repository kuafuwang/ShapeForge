"""A scripted stand-in for ``openai.AsyncOpenAI`` so that LLM/orchestrator logic is testable offline."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable


class FakeUsage(SimpleNamespace):
    def model_dump(self) -> dict:
        return dict(self.__dict__)


def make_usage(hit: int = 0, miss: int = 1000, out: int = 500) -> FakeUsage:
    return FakeUsage(prompt_tokens=hit + miss, completion_tokens=out, total_tokens=hit + miss + out, prompt_cache_hit_tokens=hit, prompt_cache_miss_tokens=miss)


def make_response(content: str = "", *, reasoning: str | None = None, tool_calls: list[tuple[str, str, dict]] | None = None, usage: FakeUsage | None = None, finish: str | None = None):
    tcs = None
    if tool_calls:
        tcs = [SimpleNamespace(id=i, type="function", function=SimpleNamespace(name=n, arguments=json.dumps(a))) for i, n, a in tool_calls]
    msg = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tcs)
    # 带工具调用时默认结束原因是 tool_calls。显式传入 finish（例如 length）时保留，用来表示这次调用被截断。
    finish_reason = "tool_calls" if tool_calls and finish is None else (finish or "stop")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], usage=usage or make_usage())


class FakeClient:
    """``script`` items are responses, exceptions (raised), or callables ``f(messages, **kw) -> response``."""

    def __init__(self, script: list[Any] | Callable | None = None) -> None:
        self.script = script
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            async def create(self, **kw):
                outer.calls.append(kw)
                if callable(outer.script):
                    return outer.script(kw["messages"], **{k: v for k, v in kw.items() if k != "messages"})
                item = outer.script.pop(0)
                if isinstance(item, BaseException):
                    raise item
                if callable(item):
                    return item(kw["messages"], **{k: v for k, v in kw.items() if k != "messages"})
                return item

        self.chat = SimpleNamespace(completions=_Completions())
