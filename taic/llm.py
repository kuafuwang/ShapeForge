"""OpenAI-compatible chat client with retries, a cache-aware USD ledger, transcripts and a tool-calling loop.

Paper correspondence: this is the "model" box of TAIC (GPT-6 Astra there; DeepSeek V4.1 Flash here). All
requests go through ``chat.completions.create`` because that is the interface every OpenAI-compatible endpoint supports.
"""

# 流水线里的位置：编排器负责一轮轮问模型、评测内核。本文件是模型这一侧——每次对话怎么换成美元、
# 发请求之前预算够不够、密钥怎样从对话记录里抹掉。预算、耐心、轮数三个停机旋钮定义在这里，拿它们结束循环的是编排器。

from __future__ import annotations

import asyncio
import hashlib
import json
import math
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


def _scrub_preview(text: str, secrets: list[str] | tuple[str, ...] = ()) -> str:
    """@brief 取流式回复的末尾一段，并抹掉形似密钥的片段。

    只看最后 ``_PREVIEW_CHARS`` 个字符；``sk-`` 开头的 token 替换成 ``sk-…``，
    保证密钥不会写进 ``llm.json``。这个预览只给仪表盘看，不参与任何判定。

    @param text 已经流出的正文，或思考内容
    @return 脱敏后的末尾切片，最多 ``_PREVIEW_CHARS`` 个字符
    """
    shown = text[-_PREVIEW_CHARS:]
    for secret in secrets:
        if isinstance(secret, str) and len(secret) >= 8:
            shown = shown.replace(secret, secret[:3] + "…")
    return _PREVIEW_SECRET.sub(r"\1…", shown)


# 请求本身失败（鉴权、坏参数、重试耗尽，或不可重试的 4xx）。编排器里：规划失败改用内置备用方案，搜索继续；
# 并行的初始样本或一轮分支里，失败的那一支丢掉，其余继续；只有全部失败，或没有并行的那一次失败，才结束运行，停因是 llm_error。
class LLMError(RuntimeError):
    """@brief 重试也修不好的一次请求失败。

    鉴权错误、坏参数、重试耗尽、不可重试的 4xx 都走这个类型。编排器的处理见类上方的注释：
    规划失败改用内置备用方案，搜索继续；并行的初始样本或一轮分支里，失败的那一支丢掉，其余继续；
    只有全部失败，或没有并行的那一次失败，才结束运行，停因是 ``llm_error``。
    """


class _Transient(Exception):
    """@brief 一次空回复或没写完的回复，值得用同一份消息再发一次。

    只在 ``chat`` 内部使用：捕获后先入账、退避，再用同一份消息重发。
    已经流出的半截文字不会带进下一次请求。
    """

    def __init__(self, kind: str) -> None:
        """@brief 记下失败种类，并把它同时放进异常的 args。

        @param kind 失败种类名（``"empty response"`` / ``"stream ended before a finish reason"``），
                    ``chat`` 在重试耗尽时靠它判断要不要把思考内容当正文交回
        """
        super().__init__(kind)
        self.kind = kind


def _is_quota(status: int | None, text: str) -> bool:
    """@brief 判断这次失败是不是余额/额度问题，而不是限流。

    402 一律算余额问题。429 只有在正文命中 ``_QUOTA_TEXT``（余额不足、账单、额度用尽一类字样）
    时才算，否则仍按限流处理，可以退避重试。

    @param status HTTP 状态码，可能为 None
    @param text 错误正文，可能为空串
    @return 是否需要按余额不足处理（是则不重试，直接抛 ``LLMError``）
    """
    if status == 402:
        return True
    return bool(_QUOTA_TEXT.search(text or ""))


def _is_context(status: int | None, text: str) -> bool:
    """@brief 判断服务商是不是因为提示过长而拒绝了这次请求。

    413 一律算。429 一律不算：正文写着 "too many tokens" 的 429 是限流，不是上下文超窗。
    其它状态码看正文是否命中 ``_CONTEXT_TEXT``。

    @param status HTTP 状态码，可能为 None
    @param text 错误正文，可能为空串
    @return 是否值得先收缩上下文再重发一次
    """
    if status == 413:
        return True
    if status == 429:
        return False
    return bool(_CONTEXT_TEXT.search(text or ""))


def _shrink_for_context(messages: list[dict]) -> tuple[list[dict], bool]:
    """@brief 去掉回显的思考内容、截短过长的工具结果，给超窗的请求瘦身。

    ``role == "tool"`` 且正文超过 ``_TOOL_SHRINK_AT`` 的消息，中间换成省略提示，
    只留前 2400 和后 800 个字符。不改动传入的列表，返回的是逐条拷贝出的新列表。

    @param messages 当前对话的消息列表
    @return ``(新消息列表, 是否删掉了 reasoning_content)``；后者为真则停掉后续请求的思考回显
    """
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
    """@brief 已经入账的美元达到预算，在发请求之前抛出（论文里的预算门）。

    只会从 ``CostLedger.check`` 里抛出，不会从一次已经发出的 HTTP 里抛出。编排器把它当成致命：
    并行里只要有一支抛出，整次搜索就停，停因是 ``budget_exhausted``，不会像 ``LLMError`` 那样丢掉一支再继续。
    两种停法之后仍会复测赢家；那是 GPU 计时，不再问模型。
    """


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
    """@brief 一个 OpenAI 兼容端点的连接方式与计价方式。

    逐字段：
    - ``base_url``：端点根地址。
    - ``api_key``：密钥；``repr`` 里不显示，``Transcript`` 会把它打码。
    - ``model``：模型名，原样放进请求。
    - ``reasoning_effort``：思考强度，默认 ``high``；None 或 ``off`` 表示在 ``extra_body`` 里
      关掉思考。
    - ``echo_reasoning_content``：后续轮次是否把上一条的 ``reasoning_content`` 回显回去。
    - ``use_tools``：是否给工具（实际传哪些 tools 由调用方决定）。
    - ``keep_transcript_bodies``：直接构造时保留审计正文；``load_config`` 给 live run 关掉它。
    - ``extra_body``：原样塞进请求的额外字段，``thinking`` 之类端点私有开关走这里。
    - ``max_tokens``：单次输出上限，思考和正文都算，默认 ``DEFAULT_MAX_TOKENS``。
    - ``request_timeout_s``：流式两块数据之间的空闲上限；只要 chunk 还在到，
      一次回复可以更久。
    - ``max_retries``：同一个请求最多重发几次。
    - ``truncation_retry``：被长度截断且没有内核的回复再问一次，不带回显的思考，
      且不花掉一次修复轮。
    - ``truncation_retry_effort``：那次截断重试的思考强度：``low`` / ``off`` / ``same``（沿用 ``reasoning_effort``）；
      默认 ``low``，因为高强度重试 20 次里仍有 14 次被截断。
    - ``plan_reasoning_effort``：只作用于规划请求；None 表示沿用 ``reasoning_effort``。
    - ``patch_reasoning_effort``：作用于 patch 与修复请求；None 表示沿用 ``reasoning_effort``；
      截断重试仍用 ``truncation_retry_effort``。
    - ``price_in`` / ``price_out`` / ``price_cache_read`` / ``price_cache_write``：USD per 1M token 单价，
      依次是未命中输入 / 输出 / 缓存读 / 缓存写；缓存写为 0 对应不为缓存填充收费的服务商。
    """

    base_url: str
    api_key: str = field(repr=False)
    model: str
    reasoning_effort: str | None = "high"
    echo_reasoning_content: bool = True
    use_tools: bool = True
    keep_transcript_bodies: bool = True  # direct construction keeps the audit body; load_config turns it off for live runs
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
    """@brief 搜索的停机旋钮、确认策略与提示增强开关。

    三个停机旋钮截的不是同一件事：``budget_usd`` 在 ``CostLedger.check`` 里截还没发出的模型请求——已花费达到它就抛
    ``BudgetExhausted``，一轮中途也会停；``patience`` 截“没有换上新的最优内核”的轮数，一轮全部评完后才加一，
    达到就停，停因是 ``converged``，这个比较在编排器里；``max_rounds`` 截优化轮数，初始内核在这个循环之外，
    循环走完则停因是 ``max_rounds``，这个比较也在编排器里。

    逐字段：
    - ``budget_usd``：一次运行的 API 预算，默认 ``PAPER_BUDGET_USD``。
    - ``branches``：并行分支数；flame chase 每轮一个 session，启动时给其它值会被拒绝。
    - ``max_repair_tries``：未使用：外层评估失败的那一支交给下一个 session。
    - ``max_rounds``：优化轮数上限。
    - ``max_quick_checks``：未使用：编码 agent 用 ``tcenv eval --timed`` 自己计时。
    - ``min_improvement``：新内核要快过当前最优的最小倍数。
    - ``noise_z``：替换还得比最优快出这么多个倍速比的标准误。
    - ``confirm_top``：最多复测这么多颗接近平手的候选；除非另一颗完整的复测过了噪声门，
      搜索种子不变。
    - ``confirm_batches``：确认阶段每颗候选跑几批独立输入，报中位数而不是某一批的最大值。
      一批的采样误差已很小（30 次重复、cv≈2% → SE≈0.4%），但换一批输入会带来 1–2% 的数据相关差异；
      见 ``runs/apple-m4-max/conv2d-feedback-20261021``：搜索值 1.7276（9 颗里挑最大，向上偏约 0.9%），单批确认 1.6896，
      其中候选在确认那批输入上慢 1.8%。1 保留旧的单批行为。
    - ``eval_reps``：单次计时重复几次。
    - ``max_hangs``：允许挂死几次。
    - ``correctness_only``：语义恢复测试用：只看正确性（快速评估，不跑基准、不做确认）。
    - ``profile_best``：结尾对赢家内核跑一次 Instruments 追踪。
    - ``patience``：连续这么多轮没出新最优就停；一个 session 算一轮，所以 6 给两个委员会各三次机会。
    - ``plan_mix``：未使用：session 自己选方向。
    - ``init_samples``：初始样本数；启动时给其它值会被拒绝。
    - ``profile_poor``：某轮种子比最强基线慢时，用 Instruments 追踪它，并把结论交给 planner 与 patch 作者。
    - ``profile_poor_below``：“更慢”的阈值：种子相对最强基线的倍速比低于此值。
    - ``profile_max``：一次运行最多追踪几次（每次在独占 GPU 锁下要几十秒）。
    - ``hw_feedback``：给每个正确种子的反馈追加一段静态硬件特征分析（``taic/hwfeedback.py``）并作为 Skill 2c 教它；
      关则 prompt 与反馈跟以前逐字节相同。
    - ``stop_at_roofline``：种子已落在其 roofline 的噪声门内、且没有基线能超过这个界时结束搜索。
    - ``abi_guard``：同一支重复同一 ABI 失败就停，并在修复 prompt 里写明 threadgroup memory 上限。
    - ``target_note``：基线比 roofline 界还快时明说，并把这个基线的时间当目标。
    - ``rich_history``：失败分支的历史行点名失败类型（abi / incorrect / no code），而不是一句笼统的话。
    - ``status_lines``：告诉 planner 与 patch 作者噪声地板，以及还剩多少预算、轮数与耐心。
    - ``differential_recheck``：候选成为种子或落入噪声内平手之前，用另一颗独立种子复查正确性
      （快速，不计时）；关则保留旧选择。
    - ``plan_ledger``：下一份 plan 能看到每次尝试的简短记录。编译或 ABI 失败不记成“测过但输了的方向”；
      关则保留 24 行文本。
    - ``diversity_archive``：连续两轮停滞之后，该轮最后一支去 patch 一颗特征不同的合格内核；关则所有分支都从种子出发。
    - ``fast_bench``：搜索阶段的计时跳过严格串行派发，也跳过除见过的最快者以外的基线；确认阶段总是给每个基线计时。
    - ``measure_serial``：额外那次严格串行派发不被选择/发布读取；2026-10-22 的 A/B 保持关。
    - ``eval_cache``：复用完全相同的（内核、任务、种子、模式、协议）报告；不同种子或故意复测仍会占 GPU。
    - ``experience_index``：``taic.experience`` 的只读提示文件路径；None 则不给 planner prompt 加任何东西。
    - ``kernel_agent``：hmz + pi 写每个内核；其它值在启动时被拒绝。
    - ``relay_budget_usd``：一个 session 的输出 token 参考值；实际花费更高只记录，由运行预算去停下个 session。
    - ``turn_minutes``：``kernel_once`` 里主轮的墙钟上限；进程超时更长，只是兜底。
    - ``provider_retries``：结构化 hmz ``turn_failed`` 且无 submission 时，在同一 committee、同一候选内最多重试几次；
      默认 0。正常回答但未交卷、timeout、进程失败和已有 checkpoint 不重试。
    - ``session_cost_cap_usd``：一次 hmz attempt 的外层预算上限（美元），默认 None（不改变总预算门）；
      每次 provider 重试独立计算，并受运行剩余预算约束。
    - ``require_contract_launch``：合约菜单与变体运行拒收 taic-launch 块，保证 launch 就是菜单选的那个。
    - ``peer_tips``：把本次运行的近期同伴提示放进下一份 TASK.md；关则那块留空；提示从不跨运行。
    - ``harness_auto``：结构化 harness 提案可以落入本次运行的 overlay；默认关。
    - ``harness_shadow``：未经审阅的 quorum 条目留在影子层；审阅过的提示可以为本次运行激活。
    - ``harness_review``：让另一个委员会模型在下一个 session 之前审阅单来源提示。
    """

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
    measure_serial: bool = False  # the extra strictly serial dispatch is unread by selection/publication; the 2026-10-22 A/B kept it off
    eval_cache: bool = True  # reuse an identical (kernel, task, seed, mode, protocol) report; different seeds/deliberate remeasures still hit the GPU
    experience_index: str | None = None  # path of a read-only hint file from taic.experience. None adds nothing to the planner prompt
    kernel_agent: str = "pi"  # hmz + pi writes every kernel. Any other value is refused at start.
    relay_budget_usd: float = 2.0  # output-token guide for one session. A higher actual cost is logged; the run budget stops the next session
    turn_minutes: int = 25  # wall clock of the main turn inside kernel_once; the process timeout is longer and only a backstop
    provider_retries: int = 0  # retries of a structured provider turn failure within the same candidate; never retries a normal no-submission answer
    session_cost_cap_usd: float | None = None  # optional per-attempt outer hmz budget; None leaves the existing total-budget gate
    require_contract_launch: bool = False  # contract-menu and variant runs reject a taic-launch block so the launch stays the one the menu chose
    peer_tips: bool = True  # put this run's recent peer tips into the next TASK.md. Off leaves that block empty. Tips never cross runs.
    harness_auto: bool = False  # structured harness proposals may settle into this run's overlay; off by default
    harness_shadow: bool = True  # hold unreviewed quorum entries in shadow; a reviewed tip may activate for this run
    harness_review: bool = True  # let the other committee model review a single-source tip before the next session
    harness_experiment_id: str = ""  # explicit A/B binding; ordinary searches have no experiment identity
    harness_experiment_arm: str = ""  # control or treatment, from an authorized experiment plan
    harness_change_sha256: str = ""  # digest of the treatment patch, empty for control
    harness_change_path: str = ""  # optional repository-relative patch applied by the experiment driver


def _read_toml(path: Path) -> dict:
    """@brief 读一个 TOML 文件，文件不存在时返回空字典。

    @param path TOML 文件路径
    @return 解析后的字典；文件不存在时是 ``{}``
    """
    return tomllib.loads(path.read_text()) if path.exists() else {}


def load_raw_config(path: str | Path | None = None, preset: str | None = None) -> dict:
    """@brief 读出设置字典，并按预设文件做分层覆盖。

    显式给的 ``path`` 原样使用，不做预设覆盖。否则先读 ``taic.toml``，再把预设文件 ``taic.<preset>.toml`` 里列出的键
    逐 section 覆盖上去（没列出的键保留原值）。预设名取 ``preset``，其次 ``$TAIC_PRESET``，最后 ``DEFAULT_PRESET``；
    ``base``（或 ``none``）表示只用 ``taic.toml``。预设名只允许字母、数字、下划线、连字符，
    预设文件必须在仓库根目录存在。

    @param path 显式配置文件路径；给定时直接用它
    @param preset 预设名；None 表示按 ``$TAIC_PRESET`` → ``DEFAULT_PRESET`` 的顺序取
    @return 合并后的原始设置字典（``llm`` / ``search`` 等 section）
    @raises LLMError 当预设名含非法字符，或预设文件不存在
    """
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
    """@brief 把 TOML 与密钥文件收成 ``LLMConfig`` 和 ``SearchConfig``。

    优先级：环境变量 ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` 覆盖密钥文件；密钥文件由 ``llm.credentials_file`` 给出
    （默认 ``~/.pi/agent/models.json``），按 ``llm.provider`` 取 ``providers.<provider>`` 的 ``baseUrl`` / ``apiKey``，
    并按模型 id 取 ``cost``。输入/输出单价缺省 0.30 / 1.20；缓存命中单价若密钥文件没写 ``cacheRead``，就按输入单价，
    而不是数据类默认的 0.006，命中缓存不会自动按 0.006 计。``truncation_retry`` 从 ``search`` 段读；
    ``truncation_retry_effort`` / ``plan_reasoning_effort`` / ``patch_reasoning_effort`` 先从 ``search`` 段取，
    取不到再从 ``llm`` 段取。传给 ``SearchConfig`` 的只留它认识的键，值为 None 的覆盖值被忽略。
    不经 TOML 直接构造 ``LLMConfig`` 时 ``keep_transcript_bodies`` 默认开，而这里给 live run 改成默认关。

    @param path 显式配置文件路径；None 表示 ``taic.toml``（可再叠预设层）
    @param model 覆盖模型名；None 时用 ``llm.model``，再无则 ``deepseek-flash``
    @param preset 预设名，含义见 ``load_raw_config``
    @param search_overrides 逐个 ``SearchConfig`` 字段的覆盖值；None 的条目被丢掉
    @return ``(模型配置, 搜索配置)``
    @raises LLMError 当最终既没有 base_url 也没有 api_key
    """
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
        keep_transcript_bodies=bool(llm.get("keep_transcript_bodies", False)),
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
    """@brief 把一次应答的用量拆成缓存命中 / 未命中 / 缓存写入 / 输出。

    三种 token 单价不同，不能都按 ``prompt_tokens`` 计。DeepSeek 明确给出 hit 与 miss，这两项已经划分了整个 prompt，
    所以不再减一次 cache-write 字段。其它端点走 ``prompt_tokens_details.cached_tokens``，
    Kimi 另有 ``cache_write_tokens``。缓存填充不算缓存未命中。
    缺任意一边时，未命中 = 输入总量 − 命中 − 写入（不小于 0）。

    @param usage ``chat.completions`` 返回的 usage 对象或同名 dict；可为 None
    @return ``(缓存命中的输入 token, 未命中的输入 token, 缓存写入 token, 输出 token)``
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


def token_counts(tokens: Any) -> tuple[int, int, int, int]:
    """@brief 把 hmz 记录的 token 字典换算成四列计数。

    编码 agent 的流用 ``input`` / ``output`` / ``cache_read`` / ``cache_write`` 四个键。形状不认识或为空时返回全零
    而不是报错：没报 token 的会话已经由 ``cost_incomplete`` 停下了。非整数（含 bool）的单个值当 0。

    @param tokens hmz 会话的 token 字典；非 dict 时按全零处理
    @return ``(缓存读, 未命中输入, 缓存写, 输出)``
    """
    if not isinstance(tokens, dict):
        return 0, 0, 0, 0

    def get(name: str) -> int:
        value = tokens.get(name, 0)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    return get("cache_read"), get("input"), get("cache_write"), get("output")


# 一次运行共用的美元账。budget_usd 为 None 时不设门。费用在应答回来之后才加上，发请求前还不知道这一次要花多少。
def _finite_cost(value: float) -> float:
    """A billed amount must be a finite, non-negative number. NaN silently disabled the budget gate."""
    cost = float(value)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError(f"a cost must be finite and non-negative, got {value!r}")
    return cost


class CostLedger:
    """@brief 一次运行共用的美元账，并充当发请求前的预算门。

    ``budget_usd`` 为 None 时不设门。费用在应答回来之后才加上，发请求前还不知道这一次要花多少，
    所以已经放行的请求可以让总额略超预算。单价统一是 USD per 1M tokens。
    """

    def __init__(self, cfg: LLMConfig, budget_usd: float | None = None) -> None:
        """@brief 建一份空账：计数清零、计时开始，并留好预算控制文件的回调。

        ``budget_usd`` 同时存进 ``initial_budget_usd``，方便事后对比中途追加过多少。控制文件初始未设置，
        需要 ``watch_control`` 才会盯运行目录里的 ``control.json``。

        @param cfg 提供四档单价（输入 / 输出 / 缓存读 / 缓存写）的模型配置
        @param budget_usd 总预算（美元）；None 表示不设预算门
        """
        self.cfg = cfg
        self.budget_usd = budget_usd
        self.initial_budget_usd = budget_usd
        self.total_usd = 0.0
        self.in_tokens_hit = 0
        self.in_tokens_miss = 0
        self.in_tokens_write = 0
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
        """@brief 开始盯运行目录里的 ``control.json``，每次预算门之前读一次。

        先记住文件现在的修改时间和大小：启动时已经在的那一份先不当成新请求，要等它再变一次才采纳，
        避免把上一次会话留下的数字又加一遍。

        @param path 控制文件路径，通常是 ``<run_dir>/control.json``；None 表示不盯
        """
        self.control_path = Path(path) if path else None
        self._control_sig = self._signature()

    # 只看修改时间和长度，不读内容。文件不在就是没有新请求。
    def _signature(self) -> tuple[int, int] | None:
        """@brief 取控制文件的修改时间与长度，当作“有没有更新”的签名。

        只看 stat，不读内容。文件不在或读不了就是没有新请求。

        @return ``(mtime_ns, size)``；未设置控制文件或 stat 失败时为 None
        """
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
        """@brief 控制文件变了就把新的总预算接过来。

        活进程加钱的地方：文件变了才读。新总预算 = ``max(文件里的数, 已花费)``。可以比当前预算高，于是继续问模型；
        也可以比当前预算低，但不会低于已经花掉的美元——文件里的数若低于已花费，预算被抬到恰好等于已花费，
        紧接着的比较就会判定花完。文件坏掉时不改预算，只通过 ``on_control_problem`` 报一次，
        不会在之后每次发请求时重复报警。
        轮数与耐心次数跟预算写在同一个文件里，所以即使预算数字没变也会调一次 ``on_control``。

        @return 预算是否真的变了（变化小于 1e-12 不算）；未设控制文件或未设预算时总是 False
        """
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
        """@brief 接着跑时把上一份请求账搬过来。

        从 transcript 或 ``ledger.json`` 恢复请求列表、累计花费与四列 token 计数。已花费用最后一条里的累计值；
        账是空的就从 0 开始，否则预算门会以为还没花过钱。

        @param requests 上一条会话的请求记录，逐条拷贝，调用方的列表不被改动
        """
        self.requests = [dict(r) for r in requests]
        self.total_usd = float(self.requests[-1]["total_usd"]) if self.requests else 0.0
        self.in_tokens_hit = sum(int(r.get("hit", 0)) for r in self.requests)
        self.in_tokens_miss = sum(int(r.get("miss", 0)) for r in self.requests)
        self.in_tokens_write = sum(int(r.get("write", 0)) for r in self.requests)
        self.out_tokens = sum(int(r.get("out", 0)) for r in self.requests)

    def price_basis(self) -> dict:
        """@brief 这份账实际使用的 USD per 1M tokens 单价。

        每次入账都把它存到那条请求的 ``basis`` 里，事后可以确认这笔钱是按哪套价格算的。

        @return ``input`` / ``output`` / ``cache_read`` / ``cache_write`` 四档单价
        """
        c = self.cfg
        return {
            "input": c.price_in,
            "output": c.price_out,
            "cache_read": c.price_cache_read,
            "cache_write": c.price_cache_write,
        }

    # 单价是美元 / 百万 token：命中 × price_cache_read，未命中 × price_in，输出 × price_out。
    def cost_of(self, usage: Any) -> float:
        """@brief 按单价把一次用量折成美元。

        单价是美元 / 百万 token：命中 × ``price_cache_read``，写入 × ``price_cache_write``，未命中 × ``price_in``，
        输出 × ``price_out``，最后除以 1e6。

        @param usage ``chat.completions`` 的 usage 对象或同名 dict
        @return 这一次的美元花费
        """
        hit, miss, write, out = split_usage(usage)
        c = self.cfg
        return (hit * c.price_cache_read + write * c.price_cache_write + miss * c.price_in + out * c.price_out) / 1e6

    def add(self, usage: Any, label: str = "") -> float:
        """@brief 记一笔来自本进程 chat 补全的花费，并累加四列 token 计数。

        条目里包括相对 ``t0`` 的时刻、标签、四列 token、本次花费、累计花费和当时的单价表。

        @param usage ``chat.completions`` 的 usage 对象或同名 dict
        @param label 这笔花费来自哪个请求，如 ``planner#t0``
        @return 本次花费的美元
        """
        hit, miss, write, out = split_usage(usage)
        cost = _finite_cost(self.cost_of(usage))
        self.total_usd += cost
        self.in_tokens_hit += hit
        self.in_tokens_miss += miss
        self.in_tokens_write += write
        self.out_tokens += out
        self.requests.append({
            "t": time.time() - self.t0, "label": label, "hit": hit, "miss": miss, "write": write, "out": out,
            "cost_usd": cost, "total_usd": self.total_usd, "basis": self.price_basis(),
        })
        return cost

    def charge(self, usd: float, label: str = "", *, tokens: Any = None, basis: dict | None = None) -> float:
        """@brief 记一笔不是本进程 chat 补全产生的花费，比如一次编码 agent 回合。

        ``tokens`` 是 hmz 会话的 token 字典（``input`` / ``output`` / ``cache_read`` / ``cache_write``），
        记下来能让账的 token 列保持有意义；老调用方传空字典也可以。

        @param usd 这笔花费的美元数（由调用方算好）
        @param label 这笔花费的标签，如 ``kernel_once``
        @param tokens hmz 会话的 token 字典；None 或非 dict 时四列按全零记
        @param basis 这笔花费依据的单价表；None 时用 ``price_basis()``
        @return 实际入账的美元（即 ``usd``）
        """
        cost = _finite_cost(usd)
        hit, miss, write, out = token_counts(tokens)
        self.total_usd += cost
        self.in_tokens_hit += hit
        self.in_tokens_miss += miss
        self.in_tokens_write += write
        self.out_tokens += out
        self.requests.append({
            "t": time.time() - self.t0, "label": label, "hit": hit, "miss": miss, "write": write, "out": out,
            "cost_usd": cost, "total_usd": self.total_usd, "basis": dict(basis) if isinstance(basis, dict) else self.price_basis(),
        })
        return cost

    @property
    def remaining(self) -> float | None:
        """@brief 预算减去已花费的余额。

        @return 剩余美元；没有预算时为 None
        """
        return None if self.budget_usd is None else self.budget_usd - self.total_usd

    # 先看控制文件有没有改总预算，再比较已经入账的总额。不预估下一次，所以刚放行的请求回来后总额可以高于预算。
    def check(self) -> None:
        """@brief 预算门：每次发请求之前走一遍。

        先看控制文件有没有改总预算（拉高预算在这里最先被读到，见 ``watch_control``），再比较已经入账的总额。
        不预估下一次，所以已经在飞的请求回来后总额可以高于预算。

        @raises BudgetExhausted 当已入账美元达到预算
        """
        self.refresh_budget()
        if self.budget_usd is not None and self.total_usd >= self.budget_usd:
            raise BudgetExhausted(f"budget ${self.budget_usd:.2f} exhausted (spent ${self.total_usd:.4f})")

    def summary(self) -> dict:
        """@brief 给日志与仪表盘用的账目快照。

        @return 含 ``total_usd`` / ``budget_usd`` / ``requests`` / 四列 token 计数 / ``cost_basis`` 的字典
        """
        return {
            "total_usd": self.total_usd,
            "budget_usd": self.budget_usd,
            "requests": len(self.requests),
            "input_tokens_cache_hit": self.in_tokens_hit,
            "input_tokens_cache_miss": self.in_tokens_miss,
            "input_tokens_cache_write": self.in_tokens_write,
            "output_tokens": self.out_tokens,
            "cost_basis": self.price_basis(),
        }


# ---------------------------------------------------------------------------------------------
# transcript
# ---------------------------------------------------------------------------------------------




def chat_transcript_event(
    *, label: str, model: str, seconds: float, messages: list, reply_message: dict,
    finish_reason, usage, cost_usd: float, total_usd: float, keep_bodies: bool,
) -> dict:
    """@brief 造一行 chat transcript 记录。

    正文默认不落盘：只存消息与回复的 sha256 和字节数，避免像旧 run 那样重复存下 760 MB 的 prompt。
    ``keep_bodies`` 为真时保留旧形状，供明确需要完整审计的调用方使用。hmz 搜索不走这条 live 路径，
    但真跑起来时也不该再造出那些重复正文。

    @param label 请求标签，如 ``planner#t0``
    @param model 模型名
    @param seconds 这次请求耗掉的秒数
    @param messages 发出去的消息列表
    @param reply_message 回填的 assistant 消息
    @param finish_reason 结束原因（``stop`` / ``length`` / ``tool_calls`` 等），可能为 None
    @param usage 用量对象；有 ``model_dump`` 就用它，None 记 None，其它转成字符串
    @param cost_usd 这次请求的美元花费
    @param total_usd 入账之后的累计美元
    @param keep_bodies 为真则存完整正文，否则只存 ``*_sha256`` 与 ``*_bytes``
    @return 一行 dict，可直接写进 ``transcript.jsonl``
    """
    event = {
        "event": "chat", "label": label, "model": model, "seconds": seconds,
        "finish_reason": finish_reason,
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else (None if usage is None else str(usage)),
        "cost_usd": cost_usd, "total_usd": total_usd,
    }
    texts = (
        ("messages", json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)),
        ("reply", json.dumps(reply_message, ensure_ascii=False, sort_keys=True, default=str)),
    )
    if keep_bodies:
        event["messages"] = messages
        event["reply"] = reply_message
    else:
        for name, text in texts:
            event[f"{name}_sha256"] = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
            event[f"{name}_bytes"] = len(text.encode("utf-8", "replace"))
    return event


# 运行目录里的 transcript.jsonl。密钥不能落盘：已知秘密整段替换，再按 sk- 形态扫一遍。
class Transcript:
    """@brief 运行目录里 ``transcript.jsonl`` 的追加式日志，每次请求/应答写一行。

    密钥不能落盘：写进文件的每一行都先脱敏。``path`` 为 None 时所有写入都静默跳过。
    """

    # 短于 8 个字符的秘密不进表，避免把普通短词替换掉。
    def __init__(self, path: Path | None, secrets: list[str] | None = None) -> None:
        """@brief 记下落盘路径与已知秘密表，并确保父目录存在。

        短于 8 个字符的秘密不进表，避免把普通短词替换掉。

        @param path transcript 文件路径；None 表示不落盘
        @param secrets 已知秘密（api key 等）；None 视为空表
        """
        self.path = path
        self._secrets = [s for s in (secrets or []) if s and len(s) >= 8]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    # 先按表里的原文替换；sk- 开头且后面至少 12 个字符的形态再扫一遍，盖住没放进表里的密钥。
    def redact(self, text: str) -> str:
        """@brief 把已知秘密和形如密钥的片段替换成 ``<redacted>``。

        先按表里的原文整段替换；``sk-`` 开头且后面至少 12 个字符的形态再扫一遍，盖住没放进表里的密钥。

        @param text 待脱敏的文本（通常是一行 JSON）
        @return 脱敏后的文本
        """
        for s in self._secrets:
            text = text.replace(s, "<redacted>")
        return re.sub(r"(sk-[A-Za-z0-9_\-]{12,})", "<redacted>", text)

    def log(self, event: dict) -> None:
        """@brief 追加一行事件到 transcript。

        先序列化成一行 JSON（非 JSON 值转成字符串）再脱敏，然后追加写入。没有路径时什么也不做。

        @param event 要记录的事件字典
        """
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
    """@brief 一次补全的结果，既含解析后的字段，也含可直接回填对话的消息。

    - ``content``：正文。
    - ``reasoning_content``：服务商返回的思考内容，可能为 None。
    - ``tool_calls``：``{id, name, arguments}`` 列表，没调工具时为空。
    - ``finish_reason``：结束原因，``stop`` / ``length`` / ``tool_calls`` 等。
    - ``usage``：用量对象，用于记账。
    - ``cost_usd``：这一次的美元花费。
    - ``message``：可以直接追加到对话里的 assistant 消息。
    - ``echo_reasoning``：这段对话后续轮次是否仍回显 ``reasoning_content``；上下文收缩后会变成 False。
    """

    content: str
    reasoning_content: str | None
    tool_calls: list[dict]
    finish_reason: str | None
    usage: Any
    cost_usd: float
    message: dict  # assistant message ready to be appended to the conversation
    echo_reasoning: bool = True  # later turns of this same conversation may keep echoing reasoning_content

    def text(self) -> str:
        """@brief 取正文，空回复时给空字符串而不是 None。

        @return ``content``；空时是 ``""``
        """
        return self.content or ""


ToolHandler = Callable[[dict], "str | Awaitable[str]"]


class LLM:
    """@brief 一个带重试、记账、transcript 和工具循环的 chat 客户端。

    所有请求都走 ``chat.completions.create``，因为那是每个 OpenAI 兼容端点都支持的接口。预算门在并发信号量之外、
    也在 HTTP 之前：好几次可能都看到“还没超”然后一起发出去。
    """

    def __init__(self, cfg: LLMConfig, ledger: CostLedger, transcript: Transcript | None = None, *, client: Any = None, max_concurrency: int = 4) -> None:
        """@brief 建好客户端、打码表和并发闸门。

        无论调用方有没有把密钥放进 ``secrets``，打码表都带上这把 ``api_key``（含短于 8 字符的）。
        ``client`` 为 None 时自建 ``AsyncOpenAI``：``read`` 超时是两块数据之间的空闲，不是整次回复的墙钟，且 SDK 自带的
        重试关掉（由本类负责）。此时思考过程只要还在往下流，就不会被 ``request_timeout_s`` 整段掐断。
        ``progress_path`` 留给调用方设，给仪表盘写在飞状态用。

        @param cfg 模型配置
        @param ledger 预算与花费账
        @param transcript 落盘日志；None 表示不落盘
        @param client 现成的异步客户端（脚本化测试给的那种）；None 表示自建
        @param max_concurrency 同时在飞的请求上限
        """
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
        """@brief 关掉 HTTP 客户端，尽力而为。

        没有 ``close`` 的脚本化测试客户端直接跳过；关闭时抛的异常被吞掉，不会让一个已经跑完的 run 失败。
        """
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
        """@brief 拼出一次请求的参数。

        强度为 None 或 ``off`` 时不发 ``reasoning_effort``，而是在 ``extra_body`` 里补
        ``thinking: {"type": "disabled"}``；``extra_body`` 不为空才带上，``tools`` 为真才带上。

        @param tools 工具定义列表；None 或空表示不给工具
        @param max_tokens 覆盖单次输出上限；None 时用 ``cfg.max_tokens``
        @param reasoning_effort 覆盖思考强度；None 时用 ``cfg.reasoning_effort``
        @return 传给 ``chat.completions.create`` 的关键字参数
        """
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
        """@brief 把在飞的模型请求写进 progress 文件，给仪表盘看；不记 prompt 与密钥。

        写文件在锁内完成。``status`` 为 ``ok`` / ``error`` 时该请求从 ``inflight`` 移除并写进 ``last``，否则更新
        ``inflight`` 里的条目并保留首次的 ``started_at``。未设 ``progress_path``、文件读坏或写失败一律安静失败：
        这只是观测，不能影响搜索。

        @param label 请求标签，同时在 ``inflight`` 里当键
        @param status 状态：``inflight`` / ``streaming`` / ``retry`` / ``ok`` / ``error``
        @param fields 附加字段，如 ``preview``、``thinking``、``attempt``、``cost_usd``
        """
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
        """@brief 把补全返回的消息转成可以回填进对话的 assistant 消息。

        正文为空时补空字符串；``echo``（None 时取 ``cfg.echo_reasoning_content``）为真才带回 ``reasoning_content``；
        有 ``tool_calls`` 时按 OpenAI 的形状重写成 ``id`` / ``type`` / ``function.name`` / ``function.arguments``。

        @param msg ``_read_completion`` 收好的消息对象
        @param echo 是否回显思考内容；None 表示按配置
        @return 可直接 ``append`` 进消息列表的 dict
        """
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
        """@brief 算出这次重试该睡多久。

        本地退避是 ``min(60, 2**attempt)`` 再乘一个 0.5–1.5 的随机系数；响应头里的 ``Retry-After`` 若是个数字，
        就当作本地退避的下界。``Retry-After`` 缺失或不是数字时只用本地退避。

        @param attempt 已经重试过的次数（从 1 开始）
        @param error 触发重试的异常；None 或没有响应时只看本地退避
        @return 需要睡眠的秒数
        """
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
        """@brief 在参数上追加流式开关。

        总是要 ``stream=True``；还没被服务商拒过时再要 ``stream_options.include_usage``，好让最后一块带着用量回来。
        用量是可选的：被拒过的端点由 ``chat`` 里置位 ``_send_stream_usage`` 记住，之后不再要。

        @param params ``_params`` 拼好的请求参数
        @return 加过流式字段的新 dict，不改动传入的字典
        """
        out = dict(params)
        out["stream"] = True
        if self._send_stream_usage:
            out["stream_options"] = {"include_usage": True}
        return out

    async def _read_completion(self, resp: Any, label: str) -> tuple[Any, str | None, Any, bool]:
        """@brief 收下一次补全：整块对象直接取，流则折叠成一条消息。

        流式时按 chunk 的 index 拼工具调用，``id`` / ``name`` / ``arguments`` 都是累加的片段，
        没给 id 时用 ``call_<index>``。行进中按 ``_PREVIEW_GROWTH`` 与 ``_PREVIEW_INTERVAL_S`` 节流，
        把 ``_scrub_preview`` 处理过的正文与思考写给仪表盘。

        @param resp 客户端返回的应答，可能是整块对象，也可能是异步迭代器
        @param label 请求标签，写仪表盘用
        @return ``(消息对象, finish_reason, usage, 是否流式)``；消息对象带 ``content`` /
                ``reasoning_content`` / ``tool_calls``
        """
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
            await self._note_llm(label, "streaming", preview=_scrub_preview(text, self.transcript._secrets), thinking=_scrub_preview(thought, self.transcript._secrets))

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
        """@brief 发一次补全（可带工具定义），中途重试、超窗收缩、记账并写 transcript。

        预算门在并发信号量之外、也在 HTTP 之前；好几次可能都看到“还没超”然后一起发出去。费用要等响应才入账，
        所以已经发出的请求可以让总额略超预算；过了门的这一次，后面的重试不再查预算。

        重试分类（``_retry_delay`` 决定退避，``Retry-After`` 是下界）：
        - ``_Transient``：空回复，或流在 ``finish_reason`` 之前结束。重发前先把这次已经产生的用量入账，预算门才看得见。
          用尽 ``max_retries`` 后不再抛错，而是把最后的文字当结果交回：正文为空而思考里有 ``kernel void`` 时
          交回思考内容；结束原因为空则记成 ``stop``。
        - 429 / 连接错 / 超时 / 5xx：可重试。429 的正文若是余额问题而不限量流，则不重试，直接抛 ``LLMError``。
        - 其它 4xx：不可重试，但两种情形各放行一次——``stream_options`` 被拒时去掉该字段再发；上下文超窗
          （或服务商拒收重传的 ``reasoning_content``）时用 ``_shrink_for_context`` 瘦身再发，去掉过思考后，
          本次对话后续请求也不再回传。
        用尽重试抛 ``LLMError``，抛之前把 ``label`` 标成 ``error`` 写进仪表盘。

        成功后记账、写一行 chat transcript，``finish_reason == "length"`` 时再补一条截断告警，
        最后把 ``label`` 标成 ``ok``。返回值只是单轮结果，工具循环在 ``chat_with_tools`` 里。

        @param messages 发给模型的消息列表；上下文收缩时内部会换成瘦身后的副本
        @param tools 工具定义列表；None 表示不带工具
        @param label 请求标签，进 transcript 与仪表盘
        @param max_tokens 覆盖单次输出上限；None 时用 ``cfg.max_tokens``
        @param reasoning_effort 覆盖思考强度；None 时用 ``cfg.reasoning_effort``
        @param echo_reasoning 覆盖是否回显思考内容；None 时用 ``cfg.echo_reasoning_content``
        @return 解析好的 ``Reply``（含 ``message``、用量、本次花费与折算后的回显开关）
        @raises BudgetExhausted 当发请求之前已入账金额已达到预算
        @raises LLMError 当 4xx 不可重试、余额不足，或重试次数用尽
        """
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
        self.transcript.log(chat_transcript_event(
            label=label, model=self.cfg.model, seconds=time.time() - t0, messages=messages,
            reply_message=reply.message, finish_reason=reply.finish_reason, usage=usage,
            cost_usd=cost, total_usd=self.ledger.total_usd, keep_bodies=self.cfg.keep_transcript_bodies,
        ))
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
        """@brief 反复问模型，直到它在不调用工具的情况下给出答复。

        轮号小于 ``max_turns`` 的轮次提供工具；轮号到达 ``max_turns`` 的那一轮把工具撤掉，逼模型直接给正文。
        撤掉工具后模型仍返回工具调用时不执行，但每个调用都补一条错误结果：会话若以悬空的 ``tool_calls`` 结尾，
        续跑会把它当成“等待工具结果”，真的去执行这些被撤回的调用。``finish_reason == "length"`` 时
        工具参数可能只写了一半，同样不执行，把错误写回让模型重发或直接给内核。

        中断后续跑：``messages`` 里已经有 ``start_turn`` 之前的轮次和 ``start_turn`` 那一轮的 assistant 消息，
        ``pending`` 列出那条消息里结果没拿到的工具调用。它们先被补上（不再发请求），循环从 ``start_turn + 1`` 继续，
        请求标签（``<label>#t<turn>``）接着原会话往下数。

        @param messages 已有的消息列表；内部会拷贝，调用方的列表不被改动
        @param tools 工具定义列表
        @param handlers 工具名到实现的映射；实现可以返回字符串或 awaitable
        @param label 标签前缀，实际请求标签是 ``<label>#t<turn>``
        @param max_turns 轮号上限（含最后一轮不给工具的那次）
        @param max_tokens 每轮的单次输出上限；None 时用 ``cfg.max_tokens``
        @param start_turn 续跑时的起始轮号
        @param pending 结果尚未取得的工具调用；None 表示没有
        @param reasoning_effort 思考强度覆盖；None 时用 ``cfg.reasoning_effort``
        @return ``(最后一轮答复, 含全部工具结果与 assistant 消息的完整列表)``
        @raises LLMError 当循环意外走到末尾（正常路径都在循环内返回，所以实际上是不可达的兵底）
        """
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
        """@brief 执行一条 assistant 消息里的工具调用，并把结果追加到 ``msgs``。

        参数按 JSON 解析，空串当 ``{}``。未知工具、解析失败或 handler 抛异常都变成一条 ``error: ...`` 的工具结果
        写回模型，不向上抛；handler 返回 awaitable 时会先 await。

        @param msgs 消息列表，工具结果就地追加
        @param tool_calls ``{id, name, arguments}`` 形式的调用列表
        @param handlers 工具名到实现的映射
        """
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
