from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import openai
import pytest

try:  # the SDK's HTTP backend differs between versions
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from taic.llm import LLM, BudgetExhausted, CostLedger, LLMConfig, LLMError, Transcript, load_config, split_usage
from taic.parse import extract_json, extract_metal, parse_plans
from taic.tools import SkillIndex, make_handlers

from .fakes import FakeClient, make_response, make_usage

SECRET = "sk-test-SECRET-0123456789abcdef"


def cfg(**kw) -> LLMConfig:
    base = dict(base_url="http://x", api_key=SECRET, model="m", reasoning_effort="high", echo_reasoning_content=True, price_in=0.30, price_out=1.20, price_cache_read=0.006)
    base.update(kw)
    return LLMConfig(**base)


def run(coro):
    return asyncio.run(coro)


def status_error(cls, code, headers=None, message="boom"):
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    return cls(message, response=httpx.Response(code, request=req, headers=headers or {}), body=None)


# ----------------------------------------------------------------------------- cost


def test_cost_is_cache_aware():
    led = CostLedger(cfg())
    c = led.add(make_usage(hit=5000, miss=1000, out=2000))
    assert c == pytest.approx((5000 * 0.006 + 1000 * 0.30 + 2000 * 1.20) / 1e6)
    assert led.summary()["input_tokens_cache_hit"] == 5000


def test_usage_without_deepseek_fields_falls_back_to_cached_tokens():
    u = {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 600}}
    assert split_usage(u) == (600, 400, 0, 10)


def test_cache_write_tokens_are_not_billed_as_a_cache_miss():
    u = {"prompt_tokens": 9012, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 6400, "cache_write_tokens": 2560}}
    assert split_usage(u) == (6400, 52, 2560, 10)
    led = CostLedger(cfg(price_cache_write=0.0))
    cost = led.add(u)
    assert cost == pytest.approx((6400 * 0.006 + 52 * 0.30 + 10 * 1.20) / 1e6)


def test_budget_gate_is_checked_before_the_request():
    client = FakeClient([make_response("a", usage=make_usage(miss=10_000_000, out=0)), make_response("b")])
    llm = LLM(cfg(), CostLedger(cfg(), budget_usd=1.0), client=client)
    run(llm.chat([{"role": "user", "content": "x"}]))  # costs $3 > budget: in-flight request may overshoot
    assert llm.ledger.total_usd > 1.0
    with pytest.raises(BudgetExhausted):
        run(llm.chat([{"role": "user", "content": "y"}]))
    assert len(client.calls) == 1  # the second request was never sent


# ----------------------------------------------------------------------------- retries


def test_rate_limit_and_connection_errors_are_retried(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = FakeClient([status_error(openai.RateLimitError, 429, {"retry-after": "1"}), openai.APIConnectionError(request=httpx.Request("POST", "http://x")), make_response("ok")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    assert run(llm.chat([{"role": "user", "content": "x"}])).text() == "ok"
    assert len(client.calls) == 3


def test_bad_request_is_not_retried():
    client = FakeClient([status_error(openai.BadRequestError, 400)])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    with pytest.raises(LLMError):
        run(llm.chat([{"role": "user", "content": "x"}]))
    assert len(client.calls) == 1


def test_quota_on_429_is_not_retried():
    client = FakeClient([status_error(openai.RateLimitError, 429, message="insufficient_quota")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    with pytest.raises(LLMError, match="quota"):
        run(llm.chat([{"role": "user", "content": "x"}]))
    assert len(client.calls) == 1


def test_an_empty_retry_is_billed_and_a_kernel_in_reasoning_is_kept(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    usage = make_usage(miss=1_000_000, out=0)
    client = FakeClient([make_response("", reasoning="kernel void k() {}", finish="stop", usage=usage)])
    llm = LLM(cfg(max_retries=0), CostLedger(cfg()), client=client)
    reply = run(llm.chat([{"role": "user", "content": "x"}]))
    assert "kernel void" in reply.text()
    assert llm.ledger.total_usd == pytest.approx(0.30)
    assert len(client.calls) == 1


def test_resume_ledger_includes_an_agent_turn():
    from taic.resume import _requests_from_transcript

    notes: list[str] = []
    reqs = _requests_from_transcript([
        {"event": "chat", "label": "plan", "cost_usd": 0.5, "total_usd": 0.5, "usage": {"prompt_tokens": 10, "completion_tokens": 1, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 10}},
        {"event": "agent", "label": "r1.p0", "cost_usd": 1.2},
    ], notes)
    assert sum(r["cost_usd"] for r in reqs) == pytest.approx(1.7)


def test_empty_stop_is_sent_again(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = FakeClient([make_response("", finish="stop"), make_response("kernel void k() {}")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    assert "kernel void" in run(llm.chat([{"role": "user", "content": "x"}])).text()
    assert len(client.calls) == 2


def test_a_stream_that_ends_without_finish_is_discarded_and_sent_again(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    async def half():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="half", reasoning_content=None, tool_calls=None), finish_reason=None)],
            usage=None,
        )

    client = FakeClient([half(), make_response("done")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    assert run(llm.chat([{"role": "user", "content": "x"}])).text() == "done"
    assert len(client.calls) == 2


def test_stream_chunks_fold_into_one_reply():
    async def chunks():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hel", reasoning_content="think", tool_calls=None), finish_reason=None)],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content="lo",
                    reasoning_content=None,
                    tool_calls=[SimpleNamespace(index=0, id="c", function=SimpleNamespace(name="lookup_msl", arguments='{"topic":"x"}'))],
                ),
                finish_reason="tool_calls",
            )],
            usage=make_usage(),
        )

    client = FakeClient([chunks()])
    reply = run(LLM(cfg(), CostLedger(cfg()), client=client).chat([{"role": "user", "content": "x"}]))
    assert reply.text() == "hello" and reply.reasoning_content == "think"
    assert reply.tool_calls == [{"id": "c", "name": "lookup_msl", "arguments": '{"topic":"x"}'}]
    assert reply.finish_reason == "tool_calls"


def test_context_overflow_drops_reasoning_and_shortens_the_tool_result_once():
    client = FakeClient([
        status_error(openai.BadRequestError, 400, message="maximum context length exceeded"),
        make_response("ok"),
    ])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "a", "reasoning_content": "secret"},
        {"role": "tool", "tool_call_id": "t", "content": "Z" * 5000},
    ]
    assert run(llm.chat(messages)).text() == "ok"
    sent = client.calls[1]["messages"]
    assert all("reasoning_content" not in m for m in sent)
    assert "omitted after a context-window error" in sent[-1]["content"]
    assert llm.cfg.echo_reasoning_content is True


def test_a_tool_call_cut_off_by_length_is_not_executed():
    ran = []

    def handler(_args):
        ran.append(1)
        return "ran"

    first = make_response("partial", tool_calls=[("c1", "lookup_msl", {"topic": "x"})], finish="length")
    client = FakeClient([first, make_response("kernel void k() {}")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    reply, msgs = run(llm.chat_with_tools([{"role": "user", "content": "go"}], [], {"lookup_msl": handler}))
    assert ran == []
    assert any(m["role"] == "tool" and "cut off by the output limit" in m["content"] for m in msgs)
    assert "kernel void" in reply.text()


def test_gives_up_after_max_retries(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = FakeClient([status_error(openai.InternalServerError, 500) for _ in range(10)])
    llm = LLM(cfg(max_retries=2), CostLedger(cfg()), client=client)
    with pytest.raises(LLMError):
        run(llm.chat([{"role": "user", "content": "x"}]))
    assert len(client.calls) == 3


# ----------------------------------------------------------------------------- request shape


def test_reasoning_effort_and_thinking_off():
    client = FakeClient(lambda m, **kw: make_response("ok"))
    run(LLM(cfg(reasoning_effort="max"), CostLedger(cfg()), client=client).chat([{"role": "user", "content": "x"}]))
    assert client.calls[-1]["reasoning_effort"] == "max" and "extra_body" not in client.calls[-1]
    run(LLM(cfg(reasoning_effort="off"), CostLedger(cfg()), client=client).chat([{"role": "user", "content": "x"}]))
    assert "reasoning_effort" not in client.calls[-1] and client.calls[-1]["extra_body"]["thinking"] == {"type": "disabled"}


# ----------------------------------------------------------------------------- tool loop + reasoning echo


@pytest.mark.parametrize("echo", [True, False])
def test_tool_loop_echoes_reasoning_content_only_when_configured(echo):
    seen = {}

    def second(messages, **kw):
        seen["messages"] = messages
        return make_response("```metal\nkernel void k() {}\n```")

    client = FakeClient([make_response("", reasoning="thinking about simd_sum", tool_calls=[("call_1", "lookup_msl", {"topic": "simd_sum"})]), second])
    llm = LLM(cfg(echo_reasoning_content=echo), CostLedger(cfg()), client=client)
    handlers = make_handlers(SkillIndex(), None, 3)
    from taic.tools import LOOKUP_TOOL

    reply, msgs = run(llm.chat_with_tools([{"role": "user", "content": "go"}], [LOOKUP_TOOL], handlers, label="t"))
    assistant = [m for m in seen["messages"] if m["role"] == "assistant"][0]
    assert ("reasoning_content" in assistant) == echo
    tool_msg = [m for m in seen["messages"] if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "call_1" and "simd_sum" in tool_msg["content"]
    assert "kernel void" in reply.text()


def test_tools_are_withdrawn_at_the_turn_limit():
    client = FakeClient(lambda m, **kw: make_response("", tool_calls=[("c", "lookup_msl", {"topic": "x"})]) if kw.get("tools") else make_response("final answer"))
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    from taic.tools import LOOKUP_TOOL

    reply, _ = run(llm.chat_with_tools([{"role": "user", "content": "go"}], [LOOKUP_TOOL], make_handlers(SkillIndex(), None, 3), max_turns=2))
    assert reply.text() == "final answer"


# ----------------------------------------------------------------------------- red: the quota-exhausted branch
# 这两个用例是 test 的"红"状态：今天跑会抛 LLMError("unreachable")。修好后它们必须转绿，
# 且不是靠改这两条断言。见 docs/架构/搜索.md 的工具额度一段。


def test_a_tool_call_after_the_tools_were_withdrawn_returns_instead_of_raising():
    """Some providers echo the history's tool_calls even when ``tools`` is not offered.

    The loop must return that reply (no kernel, empty text) so the caller takes its usual
    "no code" path, instead of raising ``LLMError("unreachable")`` and failing the whole branch.
    """
    client = FakeClient(lambda m, **kw: make_response("", tool_calls=[("c", "lookup_msl", {"topic": "x"})]))
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    from taic.tools import LOOKUP_TOOL

    reply, msgs = run(llm.chat_with_tools([{"role": "user", "content": "go"}], [LOOKUP_TOOL], make_handlers(SkillIndex(), None, 3), max_turns=2))
    assert len(client.calls) == 3  # max_turns + 1 requests: the last one is the forced answer
    assert "tools" not in client.calls[-1]
    assert reply.text() == ""
    # 会话不能以悬空的 tool_calls 结尾，否则 resume 会把它当 pending，续跑时真的去执行这些被撤回的调用。
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["tool_calls"]
    assert msgs[-1]["role"] == "tool" and "not available" in msgs[-1]["content"]
    assert msgs[-1]["tool_call_id"] == msgs[-2]["tool_calls"][0]["id"]


def test_a_length_cut_on_the_forced_answer_returns_instead_of_raising():
    """A reply truncated by ``max_tokens`` on the last turn must not end in ``unreachable``.

    Truncated replies are 32.5% of the model spend, so this path is not hypothetical.
    """

    def script(messages, **kw):
        if kw.get("tools"):
            return make_response("", tool_calls=[("c", "lookup_msl", {"topic": "x"})])
        return make_response("", tool_calls=[("c2", "lookup_msl", {"topic": "x"})], finish="length")

    client = FakeClient(script)
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    from taic.tools import LOOKUP_TOOL

    reply, _ = run(llm.chat_with_tools([{"role": "user", "content": "go"}], [LOOKUP_TOOL], make_handlers(SkillIndex(), None, 3), max_turns=2))
    assert len(client.calls) == 3
    assert reply.text() == ""


def test_unknown_tool_and_handler_errors_go_back_to_the_model():
    client = FakeClient([make_response("", tool_calls=[("c1", "nope", {})]), lambda m, **kw: make_response("done")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    _, msgs = run(llm.chat_with_tools([{"role": "user", "content": "go"}], [], {}))
    assert any(m["role"] == "tool" and "unknown tool" in m["content"] for m in msgs)


# ----------------------------------------------------------------------------- transcript


def test_transcript_never_contains_the_api_key(tmp_path):
    tr = Transcript(tmp_path / "t.jsonl")
    llm = LLM(cfg(), CostLedger(cfg()), tr, client=FakeClient([make_response(f"my key is {SECRET}")]))
    run(llm.chat([{"role": "system", "content": f"Authorization: Bearer {SECRET}"}, {"role": "user", "content": "x"}], label="redact"))
    text = (tmp_path / "t.jsonl").read_text()
    assert SECRET not in text and "<redacted>" in text
    assert json.loads(text.splitlines()[0])["label"] == "redact"


# ----------------------------------------------------------------------------- config


def test_load_config_reads_credentials_and_prices_from_models_json(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    mj = tmp_path / "models.json"
    mj.write_text(json.dumps({"providers": {"deepseek": {"baseUrl": "https://api.example", "apiKey": "k-123456789", "models": [{"id": "deepseek-flash", "cost": {"input": 0.3, "output": 1.2, "cacheRead": 0.006}}]}}}))
    toml = tmp_path / "taic.toml"
    toml.write_text(f'[llm]\ncredentials_file = "{mj}"\nprovider = "deepseek"\nmodel = "deepseek-flash"\n[search]\nbudget_usd = 3.5\nbranches = 2\n')
    c, s = load_config(toml)
    assert (c.base_url, c.api_key, c.price_cache_read) == ("https://api.example", "k-123456789", 0.006)
    assert (s.budget_usd, s.branches) == (3.5, 2)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-123456")
    assert load_config(toml)[0].api_key == "env-key-123456"
    assert "k-123456789" not in repr(load_config(toml)[0]) and "env-key" not in repr(load_config(toml)[0])


def test_output_cap_defaults_to_48000_and_a_config_file_can_replace_it(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    assert load_config(preset="base")[0].max_tokens == 48000  # taic.toml. 96000 and half of 384K were considered the same day and set back.
    toml = tmp_path / "taic.toml"
    toml.write_text('[llm]\nmodel = "deepseek-flash"\nmax_tokens = 256000\n[search]\nbudget_usd = 1\n')
    assert load_config(toml)[0].max_tokens == 256000


def test_repo_config_loads_with_local_credentials():
    c, s = load_config()
    assert c.model == "deepseek-flash" and c.price_out == pytest.approx(1.2) and s.budget_usd > 0


# ----------------------------------------------------------------------------- parsing and retrieval


def test_extract_metal_prefers_the_last_block_that_defines_the_kernel():
    text = "idea:\n```metal\nfloat helper() { return 1; }\n```\nfinal:\n```metal\nkernel void my_k(device half* y [[buffer(0)]]) {}\n```\n"
    assert "kernel void my_k" in extract_metal(text, "my_k")
    assert extract_metal("no code here", "my_k") is None
    assert extract_metal("```metal\nkernel void other() {}\n```", "my_k") is None
    staged = "```metal\n/* taic-launch\nlaunch other grid=1,1,1 threads=32,1,1\n*/\nkernel void other() {}\n```\n"
    assert "taic-launch" in extract_metal(staged, "my_k")


def test_extract_json_handles_fences_nesting_and_braces_in_strings():
    t = 'Sure.\n```json\n{"bottleneck": "uses } brace", "plans": [{"title": "a", "change": "b {c}"}]}\n```'
    assert extract_json(t)["plans"][0]["change"] == "b {c}"
    assert extract_json('prefix {"a": {"b": 1}} suffix') == {"a": {"b": 1}}
    assert extract_json("nothing") is None
    analysis, plans = parse_plans('{"bottleneck":"mem","analysis":"x","plans":["vectorize loads",{"title":"t","change":"c"},{}]}', 5)
    assert analysis.startswith("mem") and [p["title"] for p in plans] == ["vectorize loads", "t"]


def test_skill_lookup_finds_the_right_section():
    idx = SkillIndex()
    assert "simd_shuffle_xor" in idx.lookup("simd_sum")
    assert "simdgroup_multiply_accumulate" in idx.lookup("simdgroup_matrix mma")
    assert "thread_elements" in idx.lookup("epilogue thread_elements")
    assert "no section matches" in idx.lookup("zzzz qqqq")


def test_aclose_closes_the_http_client_and_tolerates_clients_without_close():
    closed = []

    class Closable:
        async def close(self):
            closed.append("async")

    class SyncClosable:
        def close(self):
            closed.append("sync")

    class Failing:
        async def close(self):
            raise RuntimeError("Event loop is closed")

    for client in (Closable(), SyncClosable(), Failing(), object()):
        c = cfg()
        run(LLM(c, CostLedger(c), client=client).aclose())  # never raises
    assert closed == ["async", "sync"]


def test_paper_preset_has_the_paper_budget_and_a_longer_patience_while_base_is_untouched():
    from taic.llm import load_config

    _, paper = load_config(preset="paper")
    _, base = load_config(preset="base")
    assert paper.budget_usd == 15.0 and paper.max_rounds == 100 and paper.patience == 12
    assert base.patience == 6 and base.budget_usd != 15.0  # taic.toml alone, which the running experiments keep reading


def _stream_chunk(text: str = "", *, reasoning: str | None = None, finish: str | None = None):
    delta = SimpleNamespace(content=text or None, reasoning_content=reasoning, tool_calls=None)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=finish, delta=delta)])


def test_streaming_preview_is_throttled_and_cleared_when_the_request_ends(tmp_path, monkeypatch):
    import time

    clock = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    path = tmp_path / "llm.json"
    c = cfg()
    llm = LLM(c, CostLedger(c), client=FakeClient([]))
    llm.progress_path = path
    seen = []

    async def spy(label, status, **fields):
        if "preview" in fields:
            seen.append((clock["t"], fields["preview"], fields.get("thinking") or ""))
        await orig(label, status, **fields)

    orig = llm._note_llm
    llm._note_llm = spy

    async def chunks():
        for index in range(25):
            yield _stream_chunk("a" * 100, reasoning="sk-SUPERSECRET" if index == 0 else None)
        clock["t"] = 1002.0
        for _ in range(50):
            yield _stream_chunk("b" * 100)
        yield _stream_chunk(finish="stop")

    run(llm._read_completion(chunks(), "lab"))
    assert [item[0] for item in seen] == [1000.0, 1002.0]
    assert all(len(item[1]) <= 4096 and len(item[2]) <= 4096 for item in seen)
    stored = path.read_text(encoding="utf-8")
    assert "SUPERSECRET" not in stored
    assert "messages" not in json.loads(stored)["inflight"]["lab"]
    run(llm._note_llm("lab", "ok"))
    assert json.loads(path.read_text(encoding="utf-8"))["inflight"] == {}
    run(llm._note_llm("lab", "error", error="boom"))
    assert json.loads(path.read_text(encoding="utf-8"))["inflight"] == {}
