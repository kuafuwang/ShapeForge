"""OpenAI-compatible chat client with retries, a cache-aware USD ledger, transcripts and a tool-calling loop.

Paper correspondence: this is the "model" box of TAIC (GPT-6 Astra there; DeepSeek V4.1 Flash here). All
requests go through ``chat.completions.create`` because that is the interface every OpenAI-compatible endpoint supports.
"""

# 流水线里的位置：编排器负责一轮轮问模型、评测内核。本文件是模型这一侧——每次对话怎么换成美元、
# 发请求之前预算够不够、密钥怎样从对话记录里抹掉。预算、耐心、轮数三个停机旋钮定义在这里，拿它们结束循环的是编排器。

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


# 请求本身失败（鉴权、坏参数、重试耗尽，或不可重试的 4xx）。编排器里：规划失败改用内置备用方案，搜索继续；
# 并行的初始样本或一轮分支里，失败的那一支丢掉，其余继续；只有全部失败，或没有并行的那一次失败，才结束运行，停因是 llm_error。
class LLMError(RuntimeError):
    """A request failed in a way retrying cannot fix (bad request, auth, ...)."""


# 已经入账的美元达到预算。只在发请求之前抛，不会从一次已经发出的 HTTP 里抛出。
# 编排器把它当成致命：并行里只要有一支抛出，整次搜索就停，停因是 budget_exhausted。不会像 LLMError 那样丢掉一支再继续。
# 两种停法之后仍会复测赢家；那是 GPU 计时，不再问模型。
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


# 三个旋钮截的不是同一件事：
# budget_usd 在本文件的 CostLedger.check 里截还没发出的模型请求——已花费达到它就抛 BudgetExhausted，一轮中途也会停。
# patience 截「没有换上新的最优内核」的轮数：一轮全部评完后才加一，达到就停，停因是 converged。这个比较在编排器里。
# max_rounds 截优化轮数；初始内核在这个循环之外，循环走完则停因是 max_rounds。这个比较也在编排器里。
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
    hw_feedback: bool = False  # append a short static hardware-feature analysis (taic/hwfeedback.py) to the feedback of every correct seed and teach it as Skill 2c; off = prompts and feedback byte-identical to before


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


# 把 toml 和密钥文件收成模型配置和搜索配置。缓存命中单价若密钥文件没写 cacheRead，就按输入单价，而不是数据类默认的 0.006。
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
        # 没有 cacheRead 时回落到 input（再没有才是 0.30）。命中缓存不会自动按数据类默认的 0.006 计。
        price_cache_read=float(price.get("cacheRead", price.get("input", 0.30))),
    )
    allowed = SearchConfig.__dataclass_fields__.keys()
    return cfg, SearchConfig(**{k: v for k, v in search.items() if k in allowed})


# ---------------------------------------------------------------------------------------------
# cost ledger
# ---------------------------------------------------------------------------------------------


# 把用量拆成缓存命中的输入、未命中的输入、输出。三种 token 单价不同，不能都按 prompt_tokens 计。
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
    # 两边都有就直接用，不再拿 prompt_tokens 重算。缺任意一边时，命中改从 cached_tokens 取，未命中 = 输入总量减去命中；只给出的那一边作废。
    if hit is None or miss is None:
        hit = g(g(usage, "prompt_tokens_details"), "cached_tokens", 0) or 0
        miss = max(prompt - int(hit), 0)
    return int(hit), int(miss), out


# 一次运行共用的美元账。budget_usd 为 None 时不设门。费用在应答回来之后才加上，发请求前还不知道这一次要花多少。
class CostLedger:
    def __init__(self, cfg: LLMConfig, budget_usd: float | None = None) -> None:
        self.cfg = cfg
        self.budget_usd = budget_usd
        self.initial_budget_usd = budget_usd
        self.total_usd = 0.0
        self.in_tokens_hit = 0
        self.in_tokens_miss = 0
        self.out_tokens = 0
        self.requests: list[dict] = []
        self.t0 = time.time()
        # live top-up: a small JSON file the dashboard (or ``taic.cli``) writes into the run directory, read before every budget gate
        self.control_path: Path | None = None
        self.on_budget_change: Callable[[float, float, dict], None] | None = None
        self.on_control_problem: Callable[[str], None] | None = None
        self._control_sig: tuple[int, int] | None = None

    # -- live budget control ------------------------------------------------------------------
    # 开始盯运行目录里的 control.json。先记住文件现在的修改时间和大小：启动时已经在的那一份先不当成新请求，
    # 要等它再变一次才采纳，避免把上一次会话留下的数字又加一遍。
    def watch_control(self, path: Path | None) -> None:
        """Start reading ``path`` (``<run_dir>/control.json``) before every budget gate. A file that already exists is ignored until it changes:
        it describes an earlier session, and whoever started this one has already folded it into the budget."""
        self.control_path = Path(path) if path else None
        self._control_sig = self._signature()

    # 只看修改时间和长度，不读内容。文件不在就是没有新请求。
    def _signature(self) -> tuple[int, int] | None:
        if self.control_path is None:
            return None
        try:
            st = self.control_path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    # 活进程加钱的地方：文件变了才读。新总预算 = max(文件里的数, 已花费)。可以比当前预算高，于是继续问模型；
    # 也可以比当前预算低，但不会低于已经花掉的美元。文件里的数若低于已花费，预算被抬到恰好等于已花费，紧接着的比较就会判定花完。
    def refresh_budget(self) -> bool:
        """Take a new total budget from the control file if it changed since the last look. Never lowers the budget below what is already spent
        and ignores (with a note through ``on_control_problem``) a damaged file. Returns whether the budget changed."""
        if self.control_path is None or self.budget_usd is None:
            return False
        sig = self._signature()
        # 文件没出现，或修改时间与长度都没变：不重读。
        if sig is None or sig == self._control_sig:
            return False
        # 先记下这次的签名。文件内容坏掉时预算不动，也不会在之后每次发请求时重复报警。
        self._control_sig = sig
        from . import control

        doc = control.read_json(self.control_path)
        requested, problem = control.parse_control(doc)
        if requested is None:
            if self.on_control_problem:
                self.on_control_problem(f"control file ignored: {problem}")
            return False
        # 下界是已花费，不是当前预算。文件里的数更小，预算就变成已花费，而不是那个更小的请求。
        new = max(requested, self.total_usd)
        old = self.budget_usd
        # 和当前预算几乎相同则不当成一次变更。
        if abs(new - old) < 1e-12:
            return False
        self.budget_usd = new
        if self.on_budget_change:
            self.on_budget_change(old, new, doc)
        return True

    # 接着跑时把上一份请求账搬过来。已花费用最后一条里的累计值；账是空的就从 0 开始。否则预算门会以为还没花过钱。
    def restore(self, requests: list[dict]) -> None:
        """Carry over the requests of an earlier session (from a transcript or ``ledger.json``): totals, token counts and the request list."""
        self.requests = [dict(r) for r in requests]
        self.total_usd = float(self.requests[-1]["total_usd"]) if self.requests else 0.0
        self.in_tokens_hit = sum(int(r.get("hit", 0)) for r in self.requests)
        self.in_tokens_miss = sum(int(r.get("miss", 0)) for r in self.requests)
        self.out_tokens = sum(int(r.get("out", 0)) for r in self.requests)

    # 单价是美元 / 百万 token：命中 × price_cache_read，未命中 × price_in，输出 × price_out。
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

    # 先看控制文件有没有改总预算，再比较已经入账的总额。不预估下一次，所以刚放行的请求回来后总额可以高于预算。
    def check(self) -> None:
        """The budget gate: evaluated before every request. A request that is already in flight may overshoot it.
        A raised budget in the run's control file is picked up here first (see ``watch_control``)."""
        self.refresh_budget()
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


# 运行目录里的 transcript.jsonl。密钥不能落盘：已知秘密整段替换，再按 sk- 形态扫一遍。
class Transcript:
    """Append-only JSONL log of every request/response. Secrets are redacted."""

    # 短于 8 个字符的秘密不进表，避免把普通短词替换掉。
    def __init__(self, path: Path | None, secrets: list[str] | None = None) -> None:
        self.path = path
        self._secrets = [s for s in (secrets or []) if s and len(s) >= 8]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    # 先按表里的原文替换；sk- 开头且后面至少 12 个字符的形态再扫一遍，盖住没放进表里的密钥。
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
        # 无论调用方有没有把密钥放进 secrets，打码表都带上这把 api_key（含短于 8 字符的）。
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
    # 一次补全。预算门在并发信号量之外、也在 HTTP 之前：好几次可能都看到「还没超」然后一起发出去。
    # 费用要等响应才入账，所以已经发出的请求可以让总额略超预算。过了门的这一次，后面的重试不再查预算。
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
                        # 这次 continue 不增加 attempt，并且把回显推理关掉，后面的请求也不再带 reasoning_content。
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
        start_turn: int = 0,
        pending: list[dict] | None = None,
    ) -> tuple[Reply, list[dict]]:
        """Run the model until it answers without calling a tool. Returns (final reply, full message list).
        When the turn limit is reached tools are withdrawn so the model must produce an answer.

        Resuming an interrupted conversation: ``messages`` already hold the turns before ``start_turn`` and the assistant message of turn ``start_turn``;
        ``pending`` lists that message's tool calls whose results were never obtained. They are answered first and the loop goes on at ``start_turn + 1``,
        so request labels (``<label>#t<turn>``) continue where the interrupted session stopped."""
        # 模型可能一直叫工具。轮号用尽后把工具撤掉，逼它给出正文。中断后续跑时，缺结果的工具调用先补上，轮号不从头计。
        msgs = list(messages)
        first = start_turn
        if pending:
            # 这些调用已经问过模型，只补工具结果，不再发一次请求。
            await self._answer_tools(msgs, pending, handlers)
            first = start_turn + 1
        for turn in range(first, max_turns + 1):
            # 前 max_turns 轮可以叫工具；到达上限的那一轮不再提供工具。
            use = tools if turn < max_turns else None
            reply = await self.chat(msgs, tools=use, label=f"{label}#t{turn}", max_tokens=max_tokens)
            msgs.append(reply.message)
            if not reply.tool_calls:
                return reply, msgs
            await self._answer_tools(msgs, reply.tool_calls, handlers)
        raise LLMError("unreachable")

    @staticmethod
    async def _answer_tools(msgs: list[dict], tool_calls: list[dict], handlers: dict[str, ToolHandler]) -> None:
        """Run the tool calls of an assistant message and append their results to ``msgs``."""
        for tc in tool_calls:
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
