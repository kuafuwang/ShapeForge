"""OpenAI-compatible chat client with retries, a cache-aware USD ledger, transcripts and a tool-calling loop.

Paper correspondence: this is the "model" box of TAIC (GPT-6 Astra there; DeepSeek V4.1 Flash here). All
requests go through ``chat.completions.create`` because that is the interface every OpenAI-compatible endpoint supports.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent


class LLMError(RuntimeError):
    """A request failed in a way retrying cannot fix (bad request, auth, ...)."""


class BudgetExhausted(RuntimeError):
    """The USD budget was used up; raised *before* a request is issued (the paper's budget gate)."""


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------


@dataclass
class LLMConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    reasoning_effort: str | None = "high"
    echo_reasoning_content: bool = True
    use_tools: bool = True
    extra_body: dict = field(default_factory=dict)
    max_tokens: int = 48000
    request_timeout_s: float = 900.0
    max_retries: int = 6
    # USD per million tokens
    price_in: float = 0.30
    price_out: float = 1.20
    price_cache_read: float = 0.006


# The paper's per-run API budget: "a maximum API budget of $15 per run for convolution, matrix multiplication, and FlashAttention"
# (Section 4.3, Table 1; $1 for the elementwise SwiGLU kernel) and "under a $15 budget" for the ten published kernels (Section 4.4, Figure 5).
PAPER_BUDGET_USD = 15.0
# Settings layered over taic.toml when no explicit config file is given: ``taic.<preset>.toml`` (only the keys it lists are replaced).
# The paper preset holds the paper-level budget and a round limit high enough that the budget, not the round count, ends a run.
# Select another one with ``--preset NAME`` or ``TAIC_PRESET=NAME``; ``base`` means taic.toml alone.
DEFAULT_PRESET = "paper"


@dataclass
class SearchConfig:
    budget_usd: float = PAPER_BUDGET_USD
    branches: int = 3
    max_repair_tries: int = 3
    max_rounds: int = 12
    max_quick_checks: int = 3
    min_improvement: float = 1.005
    noise_z: float = 2.0  # a replacement must also beat the best by this many standard errors of the speedup comparison
    confirm_top: int = 2  # re-measure up to this many statistically tied top candidates at the end and keep the fastest
    eval_reps: int = 15
    max_hangs: int = 1
    correctness_only: bool = False  # semantic-recovery tests: judge correctness only (quick evaluation, no benchmark, no confirmation)
    profile_best: bool = False  # run an Instruments trace of the winning kernel at the end
    patience: int = 4  # stop after this many rounds without a new best
    plan_mix: bool = False  # planner: the last plan of a round is a local, low-risk (instruction- or parameter-level) change, the others are structural (paper Section 3.3 plans "range from local instruction-level edits to global changes")
    init_samples: int = 1  # initial kernels sampled in parallel; the best eligible one seeds the search (repeated sampling, Brown et al. 2024)
    profile_poor: bool = False  # when a round's seed is slower than the strongest baseline, trace it with Instruments and give the findings to the planner and the patch authors
    profile_poor_below: float = 1.0  # "slower": speedup of the seed over the strongest baseline below this
    profile_max: int = 4  # traces per run at most (each takes tens of seconds under the exclusive GPU lock)


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text()) if path.exists() else {}


def load_raw_config(path: str | Path | None = None, preset: str | None = None) -> dict:
    """The parsed settings. An explicit ``path`` is used as it is. Otherwise ``taic.toml`` is read and the preset file
    ``taic.<preset>.toml`` (``preset``, else ``$TAIC_PRESET``, else ``DEFAULT_PRESET``) replaces the keys it lists, section by section.
    ``base`` (or ``none``) selects ``taic.toml`` alone."""
    if path:
        return _read_toml(Path(path))
    raw = _read_toml(REPO_ROOT / "taic.toml")
    name = preset or os.environ.get("TAIC_PRESET") or DEFAULT_PRESET
    if name in ("base", "none"):
        return raw
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise LLMError(f"bad config preset name {name!r}")
    overlay = REPO_ROOT / f"taic.{name}.toml"
    if not overlay.exists():
        raise LLMError(f"config preset '{name}' not found: {overlay} does not exist (use --preset base for taic.toml alone)")
    for section, values in _read_toml(overlay).items():
        if isinstance(values, dict):
            raw[section] = {**raw.get(section, {}), **values}
        else:
            raw[section] = values
    return raw


def load_config(path: str | Path | None = None, *, model: str | None = None, preset: str | None = None, **search_overrides) -> tuple[LLMConfig, SearchConfig]:
    raw = load_raw_config(path, preset)
    llm = dict(raw.get("llm", {}))
    search = dict(raw.get("search", {}))
    search.update({k: v for k, v in search_overrides.items() if v is not None})
    model = model or llm.get("model", "deepseek-flash")

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    price = {}
    cred_file = Path(os.path.expanduser(llm.get("credentials_file", "~/.pi/agent/models.json")))
    if cred_file.exists():
        providers = json.loads(cred_file.read_text()).get("providers", {})
        prov = providers.get(llm.get("provider", "deepseek"), {})
        base_url = base_url or prov.get("baseUrl")
        api_key = api_key or prov.get("apiKey")
        for m in prov.get("models", []):
            if m.get("id") == model:
                price = m.get("cost", {})
    if not base_url or not api_key:
        raise LLMError(f"no credentials: set OPENAI_API_KEY/OPENAI_BASE_URL or provide provider '{llm.get('provider')}' in {cred_file}")

    cfg = LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort=llm.get("reasoning_effort", "xhigh"),
        echo_reasoning_content=bool(llm.get("echo_reasoning_content", True)),
        use_tools=bool(llm.get("use_tools", True)),
        extra_body=dict(llm.get("extra_body", {})),
        max_tokens=int(llm.get("max_tokens", 48000)),
        request_timeout_s=float(llm.get("request_timeout_s", 900)),
        max_retries=int(llm.get("max_retries", 6)),
        price_in=float(price.get("input", 0.30)),
        price_out=float(price.get("output", 1.20)),
        price_cache_read=float(price.get("cacheRead", price.get("input", 0.30))),
    )
    allowed = SearchConfig.__dataclass_fields__.keys()
    return cfg, SearchConfig(**{k: v for k, v in search.items() if k in allowed})


# ---------------------------------------------------------------------------------------------
# cost ledger
# ---------------------------------------------------------------------------------------------


def split_usage(usage: Any) -> tuple[int, int, int]:
    """(cache-hit input tokens, cache-miss input tokens, output tokens). DeepSeek reports hit/miss explicitly;
    other endpoints expose ``prompt_tokens_details.cached_tokens``."""

    def g(obj, name, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    prompt = int(g(usage, "prompt_tokens", 0) or 0)
    out = int(g(usage, "completion_tokens", 0) or 0)
    hit = g(usage, "prompt_cache_hit_tokens")
    miss = g(usage, "prompt_cache_miss_tokens")
    if hit is None or miss is None:
        hit = g(g(usage, "prompt_tokens_details"), "cached_tokens", 0) or 0
        miss = max(prompt - int(hit), 0)
    return int(hit), int(miss), out


class CostLedger:
    def __init__(self, cfg: LLMConfig, budget_usd: float | None = None) -> None:
        self.cfg = cfg
        self.budget_usd = budget_usd
        self.total_usd = 0.0
        self.in_tokens_hit = 0
        self.in_tokens_miss = 0
        self.out_tokens = 0
        self.requests: list[dict] = []
        self.t0 = time.time()

    def cost_of(self, usage: Any) -> float:
        hit, miss, out = split_usage(usage)
        c = self.cfg
        return (hit * c.price_cache_read + miss * c.price_in + out * c.price_out) / 1e6

    def add(self, usage: Any, label: str = "") -> float:
        hit, miss, out = split_usage(usage)
        cost = self.cost_of(usage)
        self.total_usd += cost
        self.in_tokens_hit += hit
        self.in_tokens_miss += miss
        self.out_tokens += out
        self.requests.append({"t": time.time() - self.t0, "label": label, "hit": hit, "miss": miss, "out": out, "cost_usd": cost, "total_usd": self.total_usd})
        return cost

    @property
    def remaining(self) -> float | None:
        return None if self.budget_usd is None else self.budget_usd - self.total_usd

    def check(self) -> None:
        """The budget gate: evaluated before every request. A request that is already in flight may overshoot it."""
        if self.budget_usd is not None and self.total_usd >= self.budget_usd:
            raise BudgetExhausted(f"budget ${self.budget_usd:.2f} exhausted (spent ${self.total_usd:.4f})")

    def summary(self) -> dict:
        return {
            "total_usd": self.total_usd,
            "budget_usd": self.budget_usd,
            "requests": len(self.requests),
            "input_tokens_cache_hit": self.in_tokens_hit,
            "input_tokens_cache_miss": self.in_tokens_miss,
            "output_tokens": self.out_tokens,
        }


# ---------------------------------------------------------------------------------------------
# transcript
# ---------------------------------------------------------------------------------------------


class Transcript:
    """Append-only JSONL log of every request/response. Secrets are redacted."""

    def __init__(self, path: Path | None, secrets: list[str] | None = None) -> None:
        self.path = path
        self._secrets = [s for s in (secrets or []) if s and len(s) >= 8]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def redact(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, "<redacted>")
        return re.sub(r"(sk-[A-Za-z0-9_\-]{12,})", "<redacted>", text)

    def log(self, event: dict) -> None:
        if not self.path:
            return
        line = self.redact(json.dumps(event, ensure_ascii=False, default=str))
        with open(self.path, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------------------------


@dataclass
class Reply:
    content: str
    reasoning_content: str | None
    tool_calls: list[dict]
    finish_reason: str | None
    usage: Any
    cost_usd: float
    message: dict  # assistant message ready to be appended to the conversation

    def text(self) -> str:
        return self.content or ""


ToolHandler = Callable[[dict], "str | Awaitable[str]"]


class LLM:
    def __init__(self, cfg: LLMConfig, ledger: CostLedger, transcript: Transcript | None = None, *, client: Any = None, max_concurrency: int = 4) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.transcript = transcript or Transcript(None)
        self.transcript._secrets = list({*self.transcript._secrets, cfg.api_key}) if cfg.api_key else self.transcript._secrets
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.request_timeout_s, max_retries=0)
        self.client = client
        self._sem = asyncio.Semaphore(max_concurrency)
        self.progress_path: Path | None = None
        self._progress_lock = asyncio.Lock()

    async def aclose(self) -> None:
        """Close the HTTP client (scripted test clients have nothing to close)."""
        close = getattr(self.client, "close", None)
        if close is None:
            return
        try:
            r = close()
            if asyncio.iscoroutine(r):
                await r
        except Exception:  # closing is best effort; never fail a finished run because of it
            pass

    # -- request construction ----------------------------------------------------------------
    def _params(self, tools: list[dict] | None, max_tokens: int | None) -> dict:
        c = self.cfg
        p: dict[str, Any] = {"model": c.model, "max_tokens": max_tokens or c.max_tokens}
        extra = dict(c.extra_body)
        if c.reasoning_effort in (None, "off"):
            extra.setdefault("thinking", {"type": "disabled"})
        else:
            p["reasoning_effort"] = c.reasoning_effort
        if extra:
            p["extra_body"] = extra
        if tools:
            p["tools"] = tools
        return p

    async def _note_llm(self, label: str, status: str, **fields) -> None:
        """In-flight model request, for the dashboard. Does not record prompts or keys."""
        path = self.progress_path
        if path is None:
            return
        try:
            async with self._progress_lock:
                doc: dict = {}
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    doc = {}
                inflight = dict(doc.get("inflight") or {})
                now = time.time()
                entry = {"label": label, "status": status, "updated_at": now, **fields}
                if status in ("ok", "error"):
                    inflight.pop(label, None)
                    doc["last"] = entry
                else:
                    prev = inflight.get(label) or {}
                    entry["started_at"] = prev.get("started_at", now)
                    inflight[label] = entry
                doc["inflight"] = inflight
                doc["updated_at"] = now
                from observe.publish import publish

                publish(path, doc)
        except Exception:
            return

    def _assistant_message(self, msg: Any) -> dict:
        d: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        rc = getattr(msg, "reasoning_content", None)
        if rc and self.cfg.echo_reasoning_content:
            d["reasoning_content"] = rc
        if getattr(msg, "tool_calls", None):
            d["tool_calls"] = [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in msg.tool_calls
            ]
        return d

    # -- single request with retries ---------------------------------------------------------
    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None, label: str = "", max_tokens: int | None = None) -> Reply:
        import openai

        self.ledger.check()  # budget gate before the request is issued
        params = self._params(tools, max_tokens)
        attempt = 0
        await self._note_llm(label, "inflight")
        async with self._sem:
            while True:
                try:
                    t0 = time.time()
                    resp = await self.client.chat.completions.create(messages=messages, **params)
                    break
                except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                    attempt += 1
                    if attempt > self.cfg.max_retries:
                        await self._note_llm(label, "error", error=f"{type(e).__name__}: {e}"[:200])
                        raise LLMError(f"{label}: gave up after {attempt - 1} retries: {type(e).__name__}: {e}") from e
                    delay = min(60.0, 2.0**attempt) * (0.5 + random.random())
                    ra = None
                    resp_obj = getattr(e, "response", None)
                    if resp_obj is not None:
                        try:
                            ra = float(resp_obj.headers.get("retry-after"))
                        except (TypeError, ValueError):
                            ra = None
                    if ra is not None:
                        delay = max(delay, ra)
                    self.transcript.log({"event": "retry", "label": label, "attempt": attempt, "error": f"{type(e).__name__}: {e}", "sleep_s": delay})
                    await self._note_llm(label, "retry", attempt=attempt, error=f"{type(e).__name__}"[:200])
                    await asyncio.sleep(delay)
                except openai.APIStatusError as e:  # 4xx: not retryable
                    self.transcript.log({"event": "error", "label": label, "status": e.status_code, "error": str(e)})
                    if e.status_code == 400 and "reasoning_content" in str(e) and any("reasoning_content" in m for m in messages):
                        # the endpoint does not accept echoed reasoning on this kind of turn: drop it and try once more
                        messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
                        self.cfg.echo_reasoning_content = False
                        await self._note_llm(label, "retry", attempt=attempt, error="HTTP 400")
                        continue
                    await self._note_llm(label, "error", error=f"HTTP {e.status_code}"[:200])
                    raise LLMError(f"{label}: HTTP {e.status_code}: {e}") from e
        choice = resp.choices[0]
        msg = choice.message
        cost = self.ledger.add(resp.usage, label)
        reply = Reply(
            content=msg.content or "",
            reasoning_content=getattr(msg, "reasoning_content", None),
            tool_calls=[{"id": t.id, "name": t.function.name, "arguments": t.function.arguments} for t in (msg.tool_calls or [])],
            finish_reason=choice.finish_reason,
            usage=resp.usage,
            cost_usd=cost,
            message=self._assistant_message(msg),
        )
        self.transcript.log(
            {
                "event": "chat",
                "label": label,
                "model": self.cfg.model,
                "seconds": time.time() - t0,
                "messages": messages,
                "reply": reply.message,
                "finish_reason": reply.finish_reason,
                "usage": resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else str(resp.usage),
                "cost_usd": cost,
                "total_usd": self.ledger.total_usd,
            }
        )
        if reply.finish_reason == "length":
            self.transcript.log({"event": "warning", "label": label, "message": "reply truncated by max_tokens"})
        await self._note_llm(label, "ok", cost_usd=cost)
        return reply

    # -- tool loop -----------------------------------------------------------------------------
    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        handlers: dict[str, ToolHandler],
        *,
        label: str = "",
        max_turns: int = 6,
        max_tokens: int | None = None,
    ) -> tuple[Reply, list[dict]]:
        """Run the model until it answers without calling a tool. Returns (final reply, full message list).
        When the turn limit is reached tools are withdrawn so the model must produce an answer."""
        msgs = list(messages)
        for turn in range(max_turns + 1):
            use = tools if turn < max_turns else None
            reply = await self.chat(msgs, tools=use, label=f"{label}#t{turn}", max_tokens=max_tokens)
            msgs.append(reply.message)
            if not reply.tool_calls:
                return reply, msgs
            for tc in reply.tool_calls:
                handler = handlers.get(tc["name"])
                try:
                    args = json.loads(tc["arguments"] or "{}")
                    if handler is None:
                        out = f"error: unknown tool '{tc['name']}'"
                    else:
                        r = handler(args)
                        out = await r if asyncio.iscoroutine(r) else r
                except Exception as e:  # noqa: BLE001 - tool errors go back to the model
                    out = f"error: {type(e).__name__}: {e}"
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(out)})
        raise LLMError("unreachable")
