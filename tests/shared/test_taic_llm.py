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

from ..support.fakes import FakeClient, make_response, make_usage
from ..support.llm import status_error

SECRET = "sk-test-SECRET-0123456789abcdef"


def cfg(**kw) -> LLMConfig:
    """@brief 构造测试用的 LLMConfig，默认指向假地址和假密钥，并给出固定价格。

    @param kw 覆盖默认字段的配置项。
    @return 构造好的测试用 LLMConfig。
    """
    base = dict(base_url="http://x", api_key=SECRET, model="m", reasoning_effort="high", echo_reasoning_content=True, price_in=0.30, price_out=1.20, price_cache_read=0.006)
    base.update(kw)
    return LLMConfig(**base)


def run(coro):
    """@brief 在同步测试里执行一个协程。

    @param coro 待执行的协程对象。
    @return 协程的返回值。
    """
    return asyncio.run(coro)


# ----------------------------------------------------------------------------- cost


def test_cost_is_cache_aware():
    """@brief 计费区分缓存命中、缓存未命中和输出 token，并汇总缓存命中数。
    """
    led = CostLedger(cfg())
    c = led.add(make_usage(hit=5000, miss=1000, out=2000))
    assert c == pytest.approx((5000 * 0.006 + 1000 * 0.30 + 2000 * 1.20) / 1e6)
    assert led.summary()["input_tokens_cache_hit"] == 5000


def test_usage_without_deepseek_fields_falls_back_to_cached_tokens():
    """@brief 缺少 DeepSeek 专用字段时，用 prompt_tokens_details.cached_tokens 拆分命中与未命中。
    """
    u = {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 600}}
    assert split_usage(u) == (600, 400, 0, 10)


def test_cache_write_tokens_are_not_billed_as_a_cache_miss():
    """@brief 缓存写入 token 从输入里扣除，不计为缓存未命中，写入单价为零时不计费。
    """
    u = {"prompt_tokens": 9012, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 6400, "cache_write_tokens": 2560}}
    assert split_usage(u) == (6400, 52, 2560, 10)
    led = CostLedger(cfg(price_cache_write=0.0))
    cost = led.add(u)
    assert cost == pytest.approx((6400 * 0.006 + 52 * 0.30 + 10 * 1.20) / 1e6)


def test_budget_gate_is_checked_before_the_request():
    """@brief 预算在发请求前检查：在途请求可以超预算，但下一次调用直接失败且不发请求。
    """
    client = FakeClient([make_response("a", usage=make_usage(miss=10_000_000, out=0)), make_response("b")])
    llm = LLM(cfg(), CostLedger(cfg(), budget_usd=1.0), client=client)
    run(llm.chat([{"role": "user", "content": "x"}]))  # costs $3 > budget: in-flight request may overshoot
    assert llm.ledger.total_usd > 1.0
    with pytest.raises(BudgetExhausted):
        run(llm.chat([{"role": "user", "content": "y"}]))
    assert len(client.calls) == 1  # the second request was never sent


# ----------------------------------------------------------------------------- retries


def test_rate_limit_and_connection_errors_are_retried(monkeypatch):
    """@brief 429 和连接错误会重试，重试成功后共发出三次请求。

    @param monkeypatch pytest 补丁工具，用来去掉重试间的真实 sleep。
    """
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = FakeClient([status_error(openai.RateLimitError, 429, {"retry-after": "1"}), openai.APIConnectionError(request=httpx.Request("POST", "http://x")), make_response("ok")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    assert run(llm.chat([{"role": "user", "content": "x"}])).text() == "ok"
    assert len(client.calls) == 3


def test_bad_request_is_not_retried():
    """@brief 400 错误不重试，直接转成 LLMError，只发一次请求。
    """
    client = FakeClient([status_error(openai.BadRequestError, 400)])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    with pytest.raises(LLMError):
        run(llm.chat([{"role": "user", "content": "x"}]))
    assert len(client.calls) == 1


def test_quota_on_429_is_not_retried():
    """@brief 429 里带 insufficient_quota 属于额度耗尽，不重试。
    """
    client = FakeClient([status_error(openai.RateLimitError, 429, message="insufficient_quota")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    with pytest.raises(LLMError, match="quota"):
        run(llm.chat([{"role": "user", "content": "x"}]))
    assert len(client.calls) == 1


def test_an_empty_retry_is_billed_and_a_kernel_in_reasoning_is_kept(monkeypatch):
    """@brief 空回复仍按 usage 计费，且 reasoning 里的 kernel 会作为文本保留。

    @param monkeypatch pytest 补丁工具，用来去掉重试间的真实 sleep。
    """
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
    """@brief resume 重建账本时把 agent 事件的花费也算进去。
    """
    from taic.resume import _requests_from_transcript

    notes: list[str] = []
    reqs = _requests_from_transcript([
        {"event": "chat", "label": "plan", "cost_usd": 0.5, "total_usd": 0.5, "usage": {"prompt_tokens": 10, "completion_tokens": 1, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 10}},
        {"event": "agent", "label": "r1.p0", "cost_usd": 1.2},
    ], notes)
    assert sum(r["cost_usd"] for r in reqs) == pytest.approx(1.7)


def test_empty_stop_is_sent_again(monkeypatch):
    """@brief finish=stop 但正文为空时重发一次请求。

    @param monkeypatch pytest 补丁工具，用来去掉重试间的真实 sleep。
    """
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = FakeClient([make_response("", finish="stop"), make_response("kernel void k() {}")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    assert "kernel void" in run(llm.chat([{"role": "user", "content": "x"}])).text()
    assert len(client.calls) == 2


def test_a_stream_that_ends_without_finish_is_discarded_and_sent_again(monkeypatch):
    """@brief 流式响应缺少 finish_reason 就结束视为截断，丢弃并重发。

    @param monkeypatch pytest 补丁工具，用来去掉重试间的真实 sleep。
    """
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
    """@brief 多个流式 chunk 的正文、reasoning 和 tool_calls 会合并成一条完整回复。
    """
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
    """@brief 上下文超限时丢掉历史 reasoning 并裁剪一次工具结果后重试成功，且不改变 echo 配置。
    """
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
    """@brief finish=length 截断的工具调用不执行，改把截断原因作为工具结果回给模型。
    """
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
    """@brief 500 错误重试到 max_retries 后抛 LLMError，共发 max_retries+1 次请求。

    @param monkeypatch pytest 补丁工具，用来去掉重试间的真实 sleep。
    """
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
    """@brief reasoning_effort 原样透传；设为 off 时省略该字段并发送 thinking disabled。
    """
    client = FakeClient(lambda m, **kw: make_response("ok"))
    run(LLM(cfg(reasoning_effort="max"), CostLedger(cfg()), client=client).chat([{"role": "user", "content": "x"}]))
    assert client.calls[-1]["reasoning_effort"] == "max" and "extra_body" not in client.calls[-1]
    run(LLM(cfg(reasoning_effort="off"), CostLedger(cfg()), client=client).chat([{"role": "user", "content": "x"}]))
    assert "reasoning_effort" not in client.calls[-1] and client.calls[-1]["extra_body"]["thinking"] == {"type": "disabled"}


# ----------------------------------------------------------------------------- tool loop + reasoning echo


@pytest.mark.parametrize("echo", [True, False])
def test_tool_loop_echoes_reasoning_content_only_when_configured(echo):
    """@brief 只有 echo_reasoning_content 打开时工具循环才回传 reasoning，工具结果按 id 关联。

    @param echo 是否回传 reasoning_content。
    """
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
    """@brief 到达轮次上限后不再提供 tools，模型必须给出最终答案。
    """
    client = FakeClient(lambda m, **kw: make_response("", tool_calls=[("c", "lookup_msl", {"topic": "x"})]) if kw.get("tools") else make_response("final answer"))
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    from taic.tools import LOOKUP_TOOL

    reply, _ = run(llm.chat_with_tools([{"role": "user", "content": "go"}], [LOOKUP_TOOL], make_handlers(SkillIndex(), None, 3), max_turns=2))
    assert reply.text() == "final answer"


# ----------------------------------------------------------------------------- red: the quota-exhausted branch
# 这两个用例是 test 的"红"状态：今天跑会抛 LLMError("unreachable")。修好后它们必须转绿，
# 且不是靠改这两条断言。见 docs/架构/搜索.md 的工具额度一段。


def test_a_tool_call_after_the_tools_were_withdrawn_returns_instead_of_raising():
    """@brief 撤回 tools 后仍回显的 tool_call 必须原样返回，而不是抛 LLMError("unreachable")。

    有些 provider 即使没有提供 ``tools`` 也会回显历史里的 tool_calls。循环必须返回这条回复
    （没有 kernel、文本为空），让调用方走惯常的“无代码”路径，而不是抛 ``LLMError("unreachable")``
    让整条分支失败。会话也不能以悬空的 tool_calls 结尾，否则 resume 会把它当 pending，
    续跑时真的去执行这些被撤回的调用。
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
    """@brief 最后一轮被 max_tokens 截断时返回空回复，而不是以 unreachable 结束。

    被截断的回复占模型花费的 32.5%，所以这条路径不是假设。
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
    """@brief 未知工具和 handler 错误作为 tool 消息回给模型，而不是中断循环。
    """
    client = FakeClient([make_response("", tool_calls=[("c1", "nope", {})]), lambda m, **kw: make_response("done")])
    llm = LLM(cfg(), CostLedger(cfg()), client=client)
    _, msgs = run(llm.chat_with_tools([{"role": "user", "content": "go"}], [], {}))
    assert any(m["role"] == "tool" and "unknown tool" in m["content"] for m in msgs)


# ----------------------------------------------------------------------------- transcript


def test_transcript_never_contains_the_api_key(tmp_path):
    """@brief transcript 里的消息和回复都不含 API key，并保留 label。

    @param tmp_path pytest 提供的临时目录，用于写 transcript。
    """
    tr = Transcript(tmp_path / "t.jsonl")
    llm = LLM(cfg(), CostLedger(cfg()), tr, client=FakeClient([make_response(f"my key is {SECRET}")]))
    run(llm.chat([{"role": "system", "content": f"Authorization: Bearer {SECRET}"}, {"role": "user", "content": "x"}], label="redact"))
    text = (tmp_path / "t.jsonl").read_text()
    assert SECRET not in text and "<redacted>" in text
    assert json.loads(text.splitlines()[0])["label"] == "redact"


# ----------------------------------------------------------------------------- config


def test_load_config_reads_credentials_and_prices_from_models_json(tmp_path, monkeypatch):
    """@brief 从 models.json 读取地址、密钥和价格；环境变量优先，repr 不泄露密钥。

    @param tmp_path pytest 提供的临时目录，用于写 models.json 和 taic.toml。
    @param monkeypatch pytest 补丁工具，用于增删环境变量。
    """
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
    """@brief 默认输出上限是 48000 token，配置文件可以覆盖它。

    @param tmp_path pytest 提供的临时目录，用于写 taic.toml。
    @param monkeypatch pytest 补丁工具，用于设置最小环境变量。
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    assert load_config(preset="base")[0].max_tokens == 48000  # taic.toml. 96000 and half of 384K were considered the same day and set back.
    toml = tmp_path / "taic.toml"
    toml.write_text('[llm]\nmodel = "deepseek-flash"\nmax_tokens = 256000\n[search]\nbudget_usd = 1\n')
    assert load_config(toml)[0].max_tokens == 256000


def test_repo_config_loads_with_local_credentials():
    """@brief 仓库自带配置能加载，模型与输出价格符合预期且搜索预算为正。
    """
    c, s = load_config()
    assert c.model == "deepseek-flash" and c.price_out == pytest.approx(1.2) and s.budget_usd > 0


# ----------------------------------------------------------------------------- parsing and retrieval


def test_extract_metal_prefers_the_last_block_that_defines_the_kernel():
    """@brief extract_metal 取最后一个定义目标 kernel 的代码块；找不到或名字不匹配时返回 None，带 taic-launch 的暂存块仍会取出。
    """
    text = "idea:\n```metal\nfloat helper() { return 1; }\n```\nfinal:\n```metal\nkernel void my_k(device half* y [[buffer(0)]]) {}\n```\n"
    assert "kernel void my_k" in extract_metal(text, "my_k")
    assert extract_metal("no code here", "my_k") is None
    assert extract_metal("```metal\nkernel void other() {}\n```", "my_k") is None
    staged = "```metal\n/* taic-launch\nlaunch other grid=1,1,1 threads=32,1,1\n*/\nkernel void other() {}\n```\n"
    assert "taic-launch" in extract_metal(staged, "my_k")


def test_extract_json_handles_fences_nesting_and_braces_in_strings():
    """@brief extract_json 处理围栏、嵌套和字符串里的花括号；parse_plans 兼容字符串形式的 plan。
    """
    t = 'Sure.\n```json\n{"bottleneck": "uses } brace", "plans": [{"title": "a", "change": "b {c}"}]}\n```'
    assert extract_json(t)["plans"][0]["change"] == "b {c}"
    assert extract_json('prefix {"a": {"b": 1}} suffix') == {"a": {"b": 1}}
    assert extract_json("nothing") is None
    analysis, plans = parse_plans('{"bottleneck":"mem","analysis":"x","plans":["vectorize loads",{"title":"t","change":"c"},{}]}', 5)
    assert analysis.startswith("mem") and [p["title"] for p in plans] == ["vectorize loads", "t"]


def test_skill_lookup_finds_the_right_section():
    """@brief 技能检索能按关键词找到对应章节，未命中时给出提示。
    """
    idx = SkillIndex()
    assert "simd_shuffle_xor" in idx.lookup("simd_sum")
    assert "simdgroup_multiply_accumulate" in idx.lookup("simdgroup_matrix mma")
    assert "thread_elements" in idx.lookup("epilogue thread_elements")
    assert "no section matches" in idx.lookup("zzzz qqqq")


def test_aclose_closes_the_http_client_and_tolerates_clients_without_close():
    """@brief aclose 关闭异步和同步客户端，对无 close 或 close 抛错的对象不抛异常。
    """
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
    """@brief paper preset 使用 15 美元预算、100 轮和 12 轮 patience，base 仍是 taic.toml 的 6 轮。
    """
    from taic.llm import load_config

    _, paper = load_config(preset="paper")
    _, base = load_config(preset="base")
    assert paper.budget_usd == 15.0 and paper.max_rounds == 100 and paper.patience == 12
    assert base.patience == 6 and base.budget_usd != 15.0  # taic.toml alone, which the running experiments keep reading


def _stream_chunk(text: str = "", *, reasoning: str | None = None, finish: str | None = None):
    """@brief 构造一个模拟的 OpenAI 流式 chunk。

    @param text 增量正文。
    @param reasoning 增量 reasoning 内容。
    @param finish 该 chunk 的 finish_reason。
    @return 带 usage、choices 和 delta 的流式 chunk。
    """
    delta = SimpleNamespace(content=text or None, reasoning_content=reasoning, tool_calls=None)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=finish, delta=delta)])


def test_streaming_preview_is_throttled_and_cleared_when_the_request_ends(tmp_path, monkeypatch):
    """@brief 流式预览按时间节流、截断到 4096 字符并脱敏密钥，请求结束或出错后 inflight 清空。

    @param tmp_path pytest 提供的临时目录，用于写进度文件。
    @param monkeypatch pytest 补丁工具，用于固定 monotonic 时钟。
    """
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


def test_extract_source_accepts_an_explicit_fence_set():
    """@brief extract_source 可以指定围栏标签集合，默认不识别 xlang 围栏。
    """
    from taic.parse import extract_source

    text = "```xlang\nkernel void my_k() {}\n```"
    assert extract_source(text, "my_k", tags={"xlang"}).startswith("kernel void my_k")
    assert extract_source(text, "my_k") is None


def test_the_preview_redacts_the_configured_key():
    """@brief 预览会替换配置里的密钥并截断超长文本，普通文本原样返回。
    """
    from taic.llm import _scrub_preview

    key = "prox-0123456789abcdef"
    out = _scrub_preview("x" * 3000 + key, [key])
    assert key not in out and "pro…" in out
    assert _scrub_preview("plain text") == "plain text"

