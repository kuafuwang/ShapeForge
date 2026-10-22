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
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
_PREVIEW_CHARS = 4096
_PREVIEW_INTERVAL_S = 1.0
_PREVIEW_GROWTH = 2048
_PREVIEW_SECRET = re.compile(r"(?i)(sk-)(\S{0,80})")
# 一次请求最多几轮（含最后一轮不给工具的那次）。resume 用同一个数判断某条 assistant 的 tool_calls 还算不算 pending。
MAX_TOOL_TURNS = 6


def _scrub_preview(text: str) -> str:
    """The latest slice of an in-flight reply. A key-shaped token is not written to llm.json."""
    shown = text[-_PREVIEW_CHARS:]
    return _PREVIEW_SECRET.sub(r"\1…", shown)


# 请求本身失败（鉴权、坏参数、重试耗尽，或不可重试的 4xx）。编排器里：规划失败改用内置备用方案，搜索继续；
# 并行的初始样本或一轮分支里，失败的那一支丢掉，其余继续；只有全部失败，或没有并行的那一次失败，才结束运行，停因是 llm_error。
class LLMError(RuntimeError):
    """A request failed in a way retrying cannot fix (bad request, auth, ...)."""


class _Transient(Exception):
    """An empty or unfinished completion. The same messages are sent again; the partial text is not."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def _is_quota(status: int | None, text: str) -> bool:
    """402, or a 429 whose body is a balance problem rather than a rate limit."""
    if status == 402:
        return True
    return bool(_QUOTA_TEXT.search(text or ""))


def _is_context(status: int | None, text: str) -> bool:
    """The provider rejected the prompt for size. A 429 that says 'too many tokens' is a rate limit, not this."""
    if status == 413:
        return True
    if status == 429:
        return False
    return bool(_CONTEXT_TEXT.search(text or ""))


def _shrink_for_context(messages: list[dict]) -> tuple[list[dict], bool]:
    """Drop echoed reasoning and shorten long tool results. Returns the new list and whether reasoning was removed."""
    out = []
    dropped = False
    for message in messages:
        item = dict(message)
        if "reasoning_content" in item:
            item.pop("reasoning_content")
            dropped = True
        content = item.get("content")
        if item.get("role") == "tool" and isinstance(content, str) and len(content) > _TOOL_SHRINK_AT:
            item["content"] = content[:2400] + "\n\n[... tool result middle omitted after a context-window error ...]\n\n" + content[-800:]
        out.append(item)
    return out, dropped


# 已经入账的美元达到预算。只在发请求之前抛，不会从一次已经发出的 HTTP 里抛出。
# 编排器把它当成致命：并行里只要有一支抛出，整次搜索就停，停因是 budget_exhausted。不会像 LLMError 那样丢掉一支再继续。
# 两种停法之后仍会复测赢家；那是 GPU 计时，不再问模型。
class BudgetExhausted(RuntimeError):
    """The USD budget was used up; raised *before* a request is issued (the paper's budget gate)."""


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------


# 单次输出上限，思考和正文都算。2026-10-07 考虑过 96000 和接口最大值的一半（196608，写满大约 17 分钟），当天改回 48000。
DEFAULT_MAX_TOKENS = 48000
# 工具结果超过这么多字符，并且服务商报了上下文超窗，才把中间换成省略号再请求一次。
_TOOL_SHRINK_AT = 4096
_QUOTA_TEXT = re.compile(r"insufficient[_\s-]?quota|insufficient balance|quota exceeded|exceeded your current quota|billing|credit balance", re.I)
_CONTEXT_TEXT = re.compile(r"context length|context window|maximum context|prompt is too long|request entity too large|payload too large", re.I)


@dataclass
class LLMConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    reasoning_effort: str | None = "high"
    echo_reasoning_content: bool = True
    use_tools: bool = True
    extra_body: dict = field(default_factory=dict)
    max_tokens: int = DEFAULT_MAX_TOKENS
    request_timeout_s: float = 900.0  # idle gap between streamed chunks. A reply may run longer while chunks keep arriving
    max_retries: int = 6
    truncation_retry: bool = True  # a length-capped reply with no kernel is asked again once, without the echoed reasoning, and does not spend a repair turn
    truncation_retry_effort: str | None = "low"  # effort on that one retry: "low", "off", or "same" (keep reasoning_effort). Default "low" because a high-effort retry was still cut off 14/20 times
    plan_reasoning_effort: str | None = None  # planner requests only; None keeps reasoning_effort
    patch_reasoning_effort: str | None = None  # patch and repair requests; None keeps reasoning_effort. A truncation retry still uses truncation_retry_effort
    # USD per million tokens
    price_in: float = 0.30
    price_out: float = 1.20
    price_cache_read: float = 0.006
    price_cache_write: float = 0.0  # USD / million tokens. 0 matches providers that do not charge for a cache fill


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
    branches: int = 1  # flame chase is one session a round; a start with any other value is refused
    max_repair_tries: int = 0  # unused: a failed outer evaluation is handed to the next session
    max_rounds: int = 12
    max_quick_checks: int = 0  # unused: the coding agent times itself with tcenv eval --timed
    min_improvement: float = 1.005
    noise_z: float = 2.0  # a replacement must also beat the best by this many standard errors of the speedup comparison
    confirm_top: int = 2  # re-measure up to this many near-ties. The search seed stays unless another complete remeasure clears the noise gate
    # 确认阶段对每一颗候选跑几批独立输入，报中位数而不是某一批的最大值。
    # 一批的采样误差已经很小（30 次重复、cv≈2% → SE≈0.4%），但换一批输入会带来 1–2% 的数据相关差异；
    # 见 runs/apple-m4-max/conv2d-feedback-20261021：搜索值 1.7276（9 颗里挑最大，向上偏约 0.9%），单批确认 1.6896，
    # 其中候选在确认那批输入上慢 1.8%。1 保留旧的单批行为。
    confirm_batches: int = 3
    eval_reps: int = 15
    max_hangs: int = 1
    correctness_only: bool = False  # semantic-recovery tests: judge correctness only (quick evaluation, no benchmark, no confirmation)
    profile_best: bool = False  # run an Instruments trace of the winning kernel at the end
    patience: int = 6  # stop after this many rounds without a new best; one session is one round, so 6 gives each committee three turns
    plan_mix: bool = False  # unused: the session chooses its own direction
    init_samples: int = 1  # a start with any other value is refused
    profile_poor: bool = False  # when a round's seed is slower than the strongest baseline, trace it with Instruments and give the findings to the planner and the patch authors
    profile_poor_below: float = 1.0  # "slower": speedup of the seed over the strongest baseline below this
    profile_max: int = 4  # traces per run at most (each takes tens of seconds under the exclusive GPU lock)
    hw_feedback: bool = False  # append a short static hardware-feature analysis (taic/hwfeedback.py) to the feedback of every correct seed and teach it as Skill 2c; off = prompts and feedback byte-identical to before
    stop_at_roofline: bool = True  # end the search when the seed is already inside the noise gate of its roofline and no baseline beats that bound
    abi_guard: bool = True  # stop a branch that repeats the same ABI failure, and state the threadgroup-memory limit in the repair prompt
    target_note: bool = True  # when a baseline is faster than the roofline bound, say so and give that baseline's time as the target
    rich_history: bool = True  # a failed branch's history line names the failure (abi / incorrect / no code) instead of one generic sentence
    status_lines: bool = True  # the planner and the patch author are told the noise floor and how much budget, round count and patience remain
    differential_recheck: bool = False  # before a candidate becomes the seed or a within-noise tie, recheck correctness on an independent seed (quick, no timing). Off leaves the old selection.
    plan_ledger: bool = True  # next plan sees a short record of each try. A compile or ABI failure is not marked as a direction that was timed and lost. Off keeps the 24 text lines
    diversity_archive: bool = False  # after two stagnant rounds, the last branch of a round patches a feature-different eligible kernel. Off keeps every branch on the seed
    fast_bench: bool = False  # search-stage timing skips the serial dispatch and baselines other than the fastest one seen. Confirmation always times every baseline
    experience_index: str | None = None  # path of a read-only hint file from taic.experience. None adds nothing to the planner prompt
    kernel_agent: str = "pi"  # hmz + pi writes every kernel. Any other value is refused at start.
    relay_budget_usd: float = 2.0  # output-token guide for one session. A higher actual cost is logged; the run budget stops the next session
    turn_minutes: int = 25  # wall clock of the main turn inside kernel_once; the process timeout is longer and only a backstop
    require_contract_launch: bool = False  # contract-menu and variant runs reject a taic-launch block so the launch stays the one the menu chose
    peer_tips: bool = True  # put this run's recent peer tips into the next TASK.md. Off leaves that block empty. Tips never cross runs.


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
        reasoning_effort=llm.get("reasoning_effort", "high"),
        echo_reasoning_content=bool(llm.get("echo_reasoning_content", True)),
        use_tools=bool(llm.get("use_tools", True)),
        extra_body=dict(llm.get("extra_body", {})),
        max_tokens=int(llm.get("max_tokens", DEFAULT_MAX_TOKENS)),
        request_timeout_s=float(llm.get("request_timeout_s", 900)),
        max_retries=int(llm.get("max_retries", 6)),
        price_in=float(price.get("input", 0.30)),
        price_out=float(price.get("output", 1.20)),
        # 没有 cacheRead 时回落到 input（再没有才是 0.30）。命中缓存不会自动按数据类默认的 0.006 计。
        price_cache_read=float(price.get("cacheRead", price.get("input", 0.30))),
        price_cache_write=float(price.get("cacheWrite", price.get("input", 0.30)) or 0.0),
    )
    if "truncation_retry" in search:
        cfg.truncation_retry = bool(search["truncation_retry"])
    for key in ("truncation_retry_effort", "plan_reasoning_effort", "patch_reasoning_effort"):
        if key in search and search[key] is not None:
            setattr(cfg, key, search[key])
        elif key in llm and llm[key] is not None:
            setattr(cfg, key, llm[key])
    allowed = SearchConfig.__dataclass_fields__.keys()
    return cfg, SearchConfig(**{k: v for k, v in search.items() if k in allowed})


# ---------------------------------------------------------------------------------------------
# cost ledger
# ---------------------------------------------------------------------------------------------


# 把用量拆成缓存命中的输入、未命中的输入、输出。三种 token 单价不同，不能都按 prompt_tokens 计。
def split_usage(usage: Any) -> tuple[int, int, int, int]:
    """(cache-hit input tokens, cache-miss input tokens, cache-write tokens, output tokens).

    DeepSeek reports hit and miss explicitly; those two already partition the prompt, so a cache-write
    field is not subtracted again. Other endpoints expose ``prompt_tokens_details.cached_tokens`` and,
    for Kimi, ``cache_write_tokens``. A cache fill is not a cache miss.
    """

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
    details = g(usage, "prompt_tokens_details")
    # 两边都有就直接用，不再拿 prompt_tokens 重算。缺任意一边时，命中改从 cached_tokens 取，
    # 缓存写入单独计，未命中 = 输入总量减去命中再减去写入。
    if hit is None or miss is None:
        hit = g(details, "cached_tokens", 0) or 0
        write = int(g(details, "cache_write_tokens", 0) or 0)
        miss = max(prompt - int(hit) - write, 0)
    else:
        write = 0
    return int(hit), int(miss), write, out


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
        self.on_control: Callable[[dict], None] | None = None
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
        # 轮数和耐心次数跟预算写在同一份文件里。预算数字没变时也要通知，否则只抬上限的请求会被丢掉。
        if self.on_control:
            self.on_control(doc)
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
        hit, miss, write, out = split_usage(usage)
        c = self.cfg
        return (hit * c.price_cache_read + write * c.price_cache_write + miss * c.price_in + out * c.price_out) / 1e6

    def add(self, usage: Any, label: str = "") -> float:
        hit, miss, _write, out = split_usage(usage)
        cost = self.cost_of(usage)
        self.total_usd += cost
        self.in_tokens_hit += hit
        self.in_tokens_miss += miss
        self.out_tokens += out
        self.requests.append({"t": time.time() - self.t0, "label": label, "hit": hit, "miss": miss, "out": out, "cost_usd": cost, "total_usd": self.total_usd})
        return cost

    def charge(self, usd: float, label: str = "") -> float:
        """Add a cost that did not come from this process's own chat completion, such as one coding-agent turn."""
        cost = float(usd)
        self.total_usd += cost
        self.requests.append({"t": time.time() - self.t0, "label": label, "hit": 0, "miss": 0, "out": 0, "cost_usd": cost, "total_usd": self.total_usd})
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
    echo_reasoning: bool = True  # later turns of this same conversation may keep echoing reasoning_content

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
            try:
                import httpx2 as _http
            except ImportError:  # the OpenAI SDK in this environment vendors HTTP as httpx2
                import httpx as _http
            from openai import AsyncOpenAI

            # read 是两块数据之间的空闲，不是整次回复的墙钟。思考过程只要还在往下流，就不会被 900 秒整段掐断。
            timeout = _http.Timeout(connect=30.0, read=cfg.request_timeout_s, write=30.0, pool=30.0)
            client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout, max_retries=0)
        self.client = client
        self._send_stream_usage = True
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
    def _params(self, tools: list[dict] | None, max_tokens: int | None, reasoning_effort: str | None = None) -> dict:
        c = self.cfg
        p: dict[str, Any] = {"model": c.model, "max_tokens": max_tokens or c.max_tokens}
        extra = dict(c.extra_body)
        effort = c.reasoning_effort if reasoning_effort is None else reasoning_effort
        if effort in (None, "off"):
            extra.setdefault("thinking", {"type": "disabled"})
        else:
            p["reasoning_effort"] = effort
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
                from tcenv.events import publish

                publish(path, doc)
        except Exception:
            return

    def _assistant_message(self, msg: Any, *, echo: bool | None = None) -> dict:
        d: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        rc = getattr(msg, "reasoning_content", None)
        keep = self.cfg.echo_reasoning_content if echo is None else echo
        if rc and keep:
            d["reasoning_content"] = rc
        if getattr(msg, "tool_calls", None):
            d["tool_calls"] = [
                {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in msg.tool_calls
            ]
        return d

    @staticmethod
    def _retry_delay(attempt: int, error: Any) -> float:
        """Seconds to wait. A numeric Retry-After is a floor under the local backoff."""
        delay = min(60.0, 2.0**attempt) * (0.5 + random.random())
        response = getattr(error, "response", None)
        if response is None:
            return delay
        try:
            after = float(response.headers.get("retry-after"))
        except (TypeError, ValueError):
            return delay
        return max(delay, after)

    def _request_params(self, params: dict) -> dict:
        """Ask for a stream. Usage on the last chunk is optional; a provider that rejects it is remembered."""
        out = dict(params)
        out["stream"] = True
        if self._send_stream_usage:
            out["stream_options"] = {"include_usage": True}
        return out

    async def _read_completion(self, resp: Any, label: str) -> tuple[Any, str | None, Any, bool]:
        """A complete chat object, or a stream folded into one message. The bool is true when bytes arrived as a stream."""
        if not hasattr(resp, "__aiter__"):
            choice = resp.choices[0]
            return choice.message, choice.finish_reason, resp.usage, False
        content: list[str] = []
        reasoning: list[str] = []
        tools: dict[int, dict[str, str]] = {}
        finish = None
        usage = None
        noted = False
        last_preview_at = 0.0
        last_preview_size = 0

        async def _flush_preview() -> None:
            nonlocal last_preview_at, last_preview_size
            text = "".join(content)
            thought = "".join(reasoning)
            size = len(text) + len(thought)
            if size - last_preview_size < _PREVIEW_GROWTH:
                return
            now = time.monotonic()
            if last_preview_at and now - last_preview_at < _PREVIEW_INTERVAL_S:
                return
            last_preview_at = now
            last_preview_size = size
            await self._note_llm(label, "streaming", preview=_scrub_preview(text), thinking=_scrub_preview(thought))

        async for chunk in resp:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            piece = getattr(delta, "content", None)
            thought = getattr(delta, "reasoning_content", None)
            deltas = list(getattr(delta, "tool_calls", None) or [])
            if (piece or thought or deltas) and not noted:
                noted = True
                await self._note_llm(label, "streaming")
            if piece:
                content.append(piece)
            if thought:
                reasoning.append(thought)
            if piece or thought:
                await _flush_preview()
            for call in deltas:
                slot = tools.setdefault(int(getattr(call, "index", 0) or 0), {"id": "", "name": "", "arguments": ""})
                if getattr(call, "id", None):
                    slot["id"] = call.id
                fn = getattr(call, "function", None)
                if fn is None:
                    continue
                if getattr(fn, "name", None):
                    slot["name"] += fn.name
                if getattr(fn, "arguments", None):
                    slot["arguments"] += fn.arguments
        calls = None
        if tools:
            calls = [
                SimpleNamespace(
                    id=slot["id"] or f"call_{index}",
                    type="function",
                    function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]),
                )
                for index, slot in sorted(tools.items())
            ]
        message = SimpleNamespace(content="".join(content), reasoning_content="".join(reasoning) or None, tool_calls=calls)
        return message, finish, usage, True

    # -- single request with retries ---------------------------------------------------------
    # 一次补全。预算门在并发信号量之外、也在 HTTP 之前：好几次可能都看到「还没超」然后一起发出去。
    # 费用要等响应才入账，所以已经发出的请求可以让总额略超预算。过了门的这一次，后面的重试不再查预算。
    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None, label: str = "", max_tokens: int | None = None, reasoning_effort: str | None = None, echo_reasoning: bool | None = None) -> Reply:
        import openai

        self.ledger.check()  # budget gate before the request is issued
        params = self._request_params(self._params(tools, max_tokens, reasoning_effort))
        attempt = 0
        context_shrunk = False
        echo = self.cfg.echo_reasoning_content if echo_reasoning is None else echo_reasoning
        usage = None
        msg = SimpleNamespace(content="", reasoning_content=None, tool_calls=None)
        finish = None
        last_billed = 0.0
        await self._note_llm(label, "inflight")
        async with self._sem:
            while True:
                try:
                    t0 = time.time()
                    resp = await self.client.chat.completions.create(messages=messages, **params)
                    msg, finish, usage, streamed = await self._read_completion(resp, label)
                    # 流干净地结束但没有 finish：半截丢掉，同一份消息再发。已经流出的字不写进下一次请求。
                    if streamed and finish is None:
                        raise _Transient("stream ended before a finish reason")
                    # 正常 stop 却什么都没有：同一份消息再发。长度封顶的空回复留给截断重试，不在这里消耗次数。
                    if finish in (None, "stop") and not (msg.content or "").strip() and not msg.tool_calls:
                        raise _Transient("empty response")
                    break
                except _Transient as e:
                    # 这次已经向服务商生成过 token。重发之前先入账，预算门才能看见。
                    if usage is not None:
                        last_billed = self.ledger.add(usage, label)
                        usage = None
                    attempt += 1
                    if attempt > self.cfg.max_retries:
                        if e.kind == "empty response":
                            reasoning = getattr(msg, "reasoning_content", None) or ""
                            content = (getattr(msg, "content", None) or "").strip()
                            # 正文是空的、核写在思考里时，把思考交给调用方。结束原因不再改写成一次干净的 stop 之前先保住这段文字。
                            if not content and "kernel void" in reasoning:
                                msg = SimpleNamespace(content=reasoning, reasoning_content=reasoning, tool_calls=None)
                            else:
                                msg = SimpleNamespace(content=content, reasoning_content=reasoning or None, tool_calls=None)
                            finish = finish or "stop"
                            break
                        await self._note_llm(label, "error", error=e.kind[:200])
                        raise LLMError(f"{label}: gave up after {attempt - 1} retries: {e.kind}") from e
                    delay = self._retry_delay(attempt, None)
                    self.transcript.log({"event": "retry", "label": label, "attempt": attempt, "error": e.kind, "sleep_s": delay})
                    await self._note_llm(label, "retry", attempt=attempt, error=e.kind[:200])
                    await asyncio.sleep(delay)
                except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                    # 429 上的余额不足不重试。限流才读 Retry-After。
                    if isinstance(e, openai.RateLimitError) and _is_quota(getattr(e, "status_code", 429), str(e)):
                        self.transcript.log({"event": "error", "label": label, "status": getattr(e, "status_code", 429), "error": str(e)})
                        await self._note_llm(label, "error", error="quota"[:200])
                        raise LLMError(f"{label}: quota: {e}") from e
                    attempt += 1
                    if attempt > self.cfg.max_retries:
                        await self._note_llm(label, "error", error=f"{type(e).__name__}: {e}"[:200])
                        raise LLMError(f"{label}: gave up after {attempt - 1} retries: {type(e).__name__}: {e}") from e
                    delay = self._retry_delay(attempt, e)
                    self.transcript.log({"event": "retry", "label": label, "attempt": attempt, "error": f"{type(e).__name__}: {e}", "sleep_s": delay})
                    await self._note_llm(label, "retry", attempt=attempt, error=f"{type(e).__name__}"[:200])
                    await asyncio.sleep(delay)
                except openai.APIStatusError as e:  # 4xx: not retryable, except one context shrink and a rejected stream_options field
                    self.transcript.log({"event": "error", "label": label, "status": e.status_code, "error": str(e)})
                    if e.status_code == 400 and "stream_options" in str(e).lower() and "stream_options" in params and not _is_context(e.status_code, str(e)):
                        params.pop("stream_options", None)
                        self._send_stream_usage = False
                        await self._note_llm(label, "retry", attempt=attempt, error="HTTP 400 stream_options")
                        continue
                    text = str(e)
                    shrink = _is_context(e.status_code, text) or (e.status_code == 400 and "reasoning_content" in text)
                    if shrink and not context_shrunk:
                        rewritten, dropped_reasoning = _shrink_for_context(messages)
                        if rewritten != messages:
                            # 超窗或服务商拒收回显的思考：只改这一次的消息再发。去掉思考后，后面的请求也不再回传。
                            messages = rewritten
                            context_shrunk = True
                            if dropped_reasoning:
                                # 只停掉这一次对话后面的回传。别的分支仍按配置把思考带回去。
                                echo = False
                            await self._note_llm(label, "retry", attempt=attempt, error=f"HTTP {e.status_code} context")
                            continue
                    await self._note_llm(label, "error", error=f"HTTP {e.status_code}"[:200])
                    raise LLMError(f"{label}: HTTP {e.status_code}: {e}") from e
        cost = self.ledger.add(usage, label) if usage is not None else last_billed
        reply = Reply(
            content=msg.content or "",
            reasoning_content=getattr(msg, "reasoning_content", None),
            tool_calls=[{"id": t.id, "name": t.function.name, "arguments": t.function.arguments} for t in (msg.tool_calls or [])],
            finish_reason=finish,
            usage=usage,
            cost_usd=cost,
            message=self._assistant_message(msg, echo=echo),
            echo_reasoning=echo,
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
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else (None if usage is None else str(usage)),
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
        max_turns: int = MAX_TOOL_TURNS,
        max_tokens: int | None = None,
        start_turn: int = 0,
        pending: list[dict] | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[Reply, list[dict]]:
        """Run the model until it answers without calling a tool. Returns (final reply, full message list).
        When the turn limit is reached tools are withdrawn so the model must produce an answer.

        Resuming an interrupted conversation: ``messages`` already hold the turns before ``start_turn`` and the assistant message of turn ``start_turn``;
        ``pending`` lists that message's tool calls whose results were never obtained. They are answered first and the loop goes on at ``start_turn + 1``,
        so request labels (``<label>#t<turn>``) continue where the interrupted session stopped."""
        # 模型可能一直叫工具。轮号用尽后把工具撤掉，逼它给出正文。中断后续跑时，缺结果的工具调用先补上，轮号不从头计。
        msgs = list(messages)
        first = start_turn
        echo: bool | None = None
        if pending:
            # 这些调用已经问过模型，只补工具结果，不再发一次请求。
            await self._answer_tools(msgs, pending, handlers)
            first = start_turn + 1
        for turn in range(first, max_turns + 1):
            # 前 max_turns 轮可以叫工具；到达上限的那一轮不再提供工具。
            use = tools if turn < max_turns else None
            reply = await self.chat(msgs, tools=use, label=f"{label}#t{turn}", max_tokens=max_tokens, reasoning_effort=reasoning_effort, echo_reasoning=echo)
            if reply.echo_reasoning is False:
                echo = False
            msgs.append(reply.message)
            if not reply.tool_calls:
                return reply, msgs
            # 这一轮已经不再提供工具。模型仍返回工具调用时不执行，避免撤工具之后还去跑 quick_check。
            # 但要把每个调用都补上一条工具结果：会话若以悬空的 tool_calls 结尾，续跑会把它当成
            # "等待工具结果"，真的去执行这些被撤回的调用（见 taic/resume.py 的 _branch）。
            if use is None:
                for tc in reply.tool_calls:
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "error: tools are not available on this turn; answer with the kernel",
                    })
                return reply, msgs
            # 输出顶到上限时，工具参数可能只写了一半。不执行，把这句写回，让模型重发或直接给内核。
            if reply.finish_reason == "length":
                for tc in reply.tool_calls:
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "error: this tool call was cut off by the output limit before it could run. Resend the complete call, or answer with the kernel.",
                    })
                continue
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
