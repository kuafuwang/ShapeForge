"""TAIC: the diagnose -> plan -> branch -> evaluate loop (paper Section 3.3, Figure 2).

  1. initial generation (+ compile/ABI/safety/correctness repair turns)
  2. diagnose the dominant bottleneck from the measured feedback and plan K different optimizations
  3. K patch requests run concurrently (LLM calls overlap; GPU evaluations are serialized behind one lock so that
     every measurement runs on an otherwise idle GPU)
  4. only candidates that are safe, correct AND faster than the current best by more than the measurement noise
     become the next seed
  5. repeat until the USD budget (checked before every request), the round limit or a stagnation limit is hit
  6. the winner and the candidates tied with it within noise are re-measured with a fresh seed and more repetitions;
     the fastest re-measured kernel is the result (removes the winner's-curse bias of picking the maximum of several
     noisy measurements)
"""

# 整条搜索是「诊断 → 规划 → 分支补丁 → 评测」。先做出一颗合格种子，之后每轮根据测量反馈找出瓶颈、规划 K 份不同改法，模型并行写补丁，GPU 上逐个量。
# 从上往下先看 run 与 _search 的轮次，再看 _eval：只有安全、算对、并且快过测量噪声的候选才能换成下一轮种子；一次完整评测超时会停掉整次搜索。
# 美元预算不在本文件里比较。每次模型请求前，LLM.chat 里的 CostLedger.check 先用 refresh_budget 看 control.json 有没有加预算，再拿已花费和 budget_usd 比；不够就抛 BudgetExhausted。已经在飞的请求可能略超。
# 每次搜索都是两个委员会轮流接手。初始稿和第 1 轮是 first，之后第 r 轮是 (r - 1) % 2。只在轮次边界换委员会。同一条候选的修补不换。
# 结束时 _confirm 用新种子重测优胜者和噪声范围内的并列者。中断后续跑走 restore / _search_resumed，不把已评过的内核再评一遍。

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from tasks import get_task
from tcenv.evaluate import EvalReport, evaluate
from tcenv.task import task_fingerprint
from tcenv.schema import SchemaError
from tcenv.report import format_feedback

from . import control, prompts
from .artifacts import LEDGER_SCHEMA_VERSION, PROTOCOL_VERSION, RESULT_SCHEMA_VERSION, RUN_CONFIG_SCHEMA_VERSION, STATE_SCHEMA_VERSION
from .feedback_extra import abi_signature, at_roofline, failure_phrase, status_lines, target_note, threadgroup_limit_note
from .llm import LLM, BudgetExhausted, CostLedger, LLMConfig, LLMError, SearchConfig, Transcript
from .policy import candidate_eligible, choose_confirmed, comparison_speedup, in_confirm_pool, next_stagnant, patience_exhausted, relation_to_best, required_gain, session_budget_floor
from .committees import committees_record, default_committees, handoff_for, order_committees
from .parse import parse_plans
from .symptom import bound_of

REPO_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_CHARS = 6000
INITIAL_RESTARTS = 2


def _git_identity(root: Path) -> tuple[str, bool | None]:
    """@brief 读取仓库的当前提交号与工作区是否脏。

    git 读不到时返回 ``("unknown", None)``。本函数不接触 GPU。

    @param root 仓库根目录（``git`` 命令的 ``cwd``）。
    @return ``(commit hash, dirty)``；git 读不到时为 ``("unknown", None)``。
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        if not commit:
            return "unknown", None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5, check=True).stdout
        return commit, bool(dirty.strip())
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


_EVAL_PHASES = ("search", "confirm", "recheck", "domain")


def eval_seed(run_seed: int, phase: str, label: str, source: str) -> int:
    """@brief 由运行种子和候选身份派生一次评测用的 48 位种子。

    ``source`` 是拼接 contract 头之前的内核正文。``phase`` 取 ``search``、``confirm``、``recheck`` 或 ``domain``；
    加入 ``domain`` 不改变其它三个 phase 的 payload。``label`` 用评测之前就存在的那个（``r2.p1.fix1``、``init.s2``、``prior``），
    不是 ``Candidate.idx``。相同输入永远返回同一个种子。它不管模型服务端的采样，也不管 GPU 温度。

    @param run_seed 整次运行的种子（``run_seed``）。
    @param phase 评测阶段：``search``、``confirm``、``recheck`` 或 ``domain``。
    @param label 评测之前就已确定的候选标签。
    @param source 拼接 contract 头之前的内核正文。
    @return 48 位整数种子。
    @raises ValueError 当 ``phase`` 不在 ``_EVAL_PHASES`` 里时。
    """
    if phase not in _EVAL_PHASES:
        raise ValueError(f"phase must be one of {_EVAL_PHASES}")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    payload = json.dumps(["ai_lowerling/eval_seed/v1", int(run_seed), phase, label, source_sha256], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:6], "big")


def _valid_run_seed(value) -> int | None:
    """@brief 校验一个候选 ``run_seed`` 是否可用。

    bool 不算整数；只接受 ``0 <= value < 2**64``。

    @param value 从配置或命令行读进来、看起来像种子的值。
    @return 合法时原样返回该整数；否则返回 None。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 0 <= value < 2**64:
        return value
    return None


def _load_run_seed(run_dir: Path) -> int | None:
    """@brief 读出这次运行记录的 ``run_seed``。

    早于 ``run_seed`` 字段的运行返回 None，这些运行的评测保持随机。先读 ``run_seed``，没有就回退到旧的 ``seed``。

    @param run_dir 运行目录（读其中的 ``run_config.json``）。
    @return 合法种子；目录没记录或值不合法时为 None。
    """
    doc = control.read_json(Path(run_dir) / control.CONFIG_FILE) or {}
    recorded = doc.get("run_seed")
    if recorded is None:
        recorded = doc.get("seed")
    return _valid_run_seed(recorded)


_RECHECK_KEYS = ("class", "task", "dtype", "size", "label", "source_sha256", "search_seed", "recheck_seed", "case", "tolerance", "speed_relation", "counts_as_kernel_error", "retest_command", "human_permission_required")


def _strip_counterexample(doc):
    """@brief 递归删掉已存报告里任何位置的 ``VerifyResult.counterexample``。

    复查文件不允许带这个字段。

    @param doc 已序列化的报告（dict/list/标量）。
    @return 去掉 ``counterexample`` 后的同构结构。
    """
    if isinstance(doc, dict):
        return {key: _strip_counterexample(value) for key, value in doc.items() if key != "counterexample"}
    if isinstance(doc, list):
        return [_strip_counterexample(value) for value in doc]
    return doc


def _report_without_counterexample(report: EvalReport | None):
    """@brief 把一份 ``EvalReport`` 转成不带 counterexample 的 JSON 文档。

    @param report 评测报告，可为 None。
    @return 清洗后的 JSON 文档；``report`` 为 None 时也返回 None。
    """
    if report is None:
        return None
    return _strip_counterexample(report.to_json())


def _contains_counterexample(doc) -> bool:
    """@brief 判断已序列化的文档里是否还残留 ``counterexample``。

    @param doc 任意已序列化的结构（dict/list/标量）。
    @return 任一层级（含嵌套 dict 与 list）出现该键时为 True。
    """
    if isinstance(doc, dict):
        if "counterexample" in doc:
            return True
        return any(_contains_counterexample(value) for value in doc.values())
    if isinstance(doc, list):
        return any(_contains_counterexample(value) for value in doc)
    return False


def check_recheck_record(doc: dict) -> list[str]:
    """@brief 检查一份差分复查记录缺了什么。

    返回空列表表示这份记录完整到可以离线阅读。本函数不跑内核。

    @param doc 待检查的复查记录。
    @return 问题描述列表；空列表表示完整。
    """
    if not isinstance(doc, dict):
        return ["the record is not an object"]
    missing = [key for key in _RECHECK_KEYS if key not in doc]
    problems = [f"missing {key}" for key in missing]
    if "counterexample" in doc or _contains_counterexample(doc):
        problems.append("a differential failure is not a VerifyResult.counterexample")
    kind = doc.get("class")
    if kind in ("timeout", "crash") and doc.get("counts_as_kernel_error") is True:
        problems.append(f"{kind} is an environment failure and must not be stored as a kernel error")
    if doc.get("human_permission_required") is False:
        problems.append("the retest command requires a person to allow it")
    return problems


# GPU 挂死或 worker 连续崩溃时立刻结束整次搜索。捕获之后不再开新的评测；默认一次 timeout 就够触发它。
class StopRun(Exception):
    """@brief 整次搜索必须立刻结束时抛出的异常（GPU 挂死熔断）。

    抛出之后不再开新的评测；默认 ``max_hangs`` 为 1，一次完整评测 timeout 就够触发它。
    """


class RunStartError(Exception):
    """@brief 新运行在任何模型请求之前被拒绝时抛出。

    拒绝发生在这两种情形：目录里已经有一次运行（拒绝覆盖既有文件），或 ``run_config.json`` 写不出去。
    """


# 一颗被评过（或即将评）的内核。能否成为下一轮种子不看标签，只看 eligible 加上是否快过测量噪声。
@dataclass
class Candidate:
    """@brief 一颗被评过（或即将评）的内核及其评测结果。

    能否成为下一轮种子不看标签，只看 ``eligible`` 加上是否快过测量噪声（``_eval`` 里的 ``_required_gain``）。
    字段：

    - ``idx``：在本场运行中的评测序号，同时是 candidates 目录名的前缀。
    - ``label``：评测前就确定的候选标签（``r2.p1.fix1``、``init.s2``、``prior``）。
    - ``source``：交给评测的内核正文。
    - ``report``：``EvalReport``；还没评过时为 None。
    - ``plan``：写这颗内核时用的规划文本，可为 None。
    - ``cost_usd``：评到这颗时的累计花费（美元）。
    - ``t_s``：评到这颗时的整场搜索耗时（秒）。
    - ``is_best``：是否已成为当前种子。
    - ``within_noise``：比当时的种子快，但优势没有超过测量噪声，不能成为下一轮种子；结束时仍可能和优胜者一起进入复测。
    - ``recheck_rejected``：第二种子复查判了核错。搜索报告仍可能是 ok，但不能进确认池，也不能当发布胜者。
    - ``correctness_only``：只做正确性恢复时为真：合格不看加速比，结束时也不做 ``_confirm``。
    """

    idx: int
    label: str
    source: str
    report: EvalReport | None = None
    plan: str | None = None
    cost_usd: float = 0.0
    t_s: float = 0.0
    is_best: bool = False
    # 比当时的种子快，但优势没有超过测量噪声，不能成为下一轮种子；结束时仍可能和优胜者一起进入复测。
    within_noise: bool = False  # measured faster than the best at the time, but not by more than the measurement noise
    # 第二种子复查判了核错。搜索报告仍可能是 ok，但不能进确认池，也不能当发布胜者。
    recheck_rejected: bool = False

    # 只做正确性恢复时为真：合格不看加速比，结束时也不做 _confirm。
    correctness_only: bool = False

    # 安全且数值正确（EvalReport.status 为 ok）才算合格。正式搜索还要求报告里真有 speedup；缺计时的候选不能换种子。规则在 policy.candidate_eligible。
    @property
    def eligible(self) -> bool:
        """@brief 这颗内核是否合格。

        安全且数值正确（``EvalReport.status`` 为 ok）才算合格。正式搜索还要求报告里真有 speedup；
        缺计时的候选不能换种子。规则在 ``policy.candidate_eligible``，``correctness_only`` 时跳过加速比要求。

        @return 可参与种子比较时为 True。
        """
        report = self.report
        return candidate_eligible(
            report is not None and bool(report.eligible),
            None if report is None else report.speedup,
            correctness_only=self.correctness_only,
        )

    # 不合格时加速比按 0，缺失的计时不可能把当前种子挤掉。和续跑回放共用 policy.comparison_speedup。
    @property
    def speedup(self) -> float:
        """@brief 用于比较的加速比。

        不合格时按 0 算，缺失的计时不可能把当前种子挤掉；合格但缺时间的按 1 算。
        和续跑回放共用 ``policy.comparison_speedup``。

        @return 合格的加速比（缺时间为 1.0），不合格为 0.0。
        """
        report = self.report
        return comparison_speedup(self.eligible, None if report is None else report.speedup)

    def brief(self) -> dict:
        """@brief 汇总成 ``RunResult.candidates`` 里的一行。

        @return 含 idx、label、plan、status、speedup、candidate_us、cost_usd、t_s、is_best 与截断到 300 字的 message 的字典；
            没评过时 status 为 ``unevaluated``。
        """
        r = self.report
        return {
            "idx": self.idx, "label": self.label, "plan": self.plan, "status": r.status if r else "unevaluated",
            "speedup": r.speedup if r else None, "candidate_us": r.candidate_us if r else None,
            "cost_usd": self.cost_usd, "t_s": self.t_s, "is_best": self.is_best,
            "message": (r.message[:300] if r else ""),
        }


# 整次搜索的收尾记录。best_speedup 是被选中时的测量；confirmed_speedup 是结束时新种子、更多重复测出来的。budget_usd 是结束时的上限，中途加过预算后会高于 initial_budget_usd。
@dataclass
class RunResult:
    """@brief 整次搜索的收尾记录。

    ``best_speedup`` 是被选中时的测量；``confirmed_speedup`` 是结束时新种子、更多重复测出来的。
    ``budget_usd`` 是结束时的上限，中途加过预算后会高于 ``initial_budget_usd``。
    字段：

    - ``run_dir``/``task``/``dtype``/``size``/``model``：运行目录与题目、模型。
    - ``stop_reason``：停止原因（``budget_exhausted``、``max_rounds``、``converged (...)``、``gpu_hang_circuit_breaker`` 等）。
    - ``rounds``：跑到第几轮。
    - ``spent_usd``：结束时已花费的美元。
    - ``budget_usd``：结束时的上限（每次加预算都会抬高）。
    - ``best_label``：作为结果的那颗内核的标签。
    - ``best_speedup``：选中那一刻的加速比，不是复测值；复测若改了优胜者，这里仍是该内核在搜索过程中的测量。
    - ``confirmed_speedup``：新种子、两倍重复次数下的加速比，用来压掉「多次噪声测量里挑最大值」。
    - ``confirmed_status``：确认阶段的状态（``ok``、``uncertain`` 或失败状态）。
    - ``confirmed_values``：确认阶段每一批的值。
    - ``confirmed_worst``/``confirmed_best``：确认阶段最差与最好的那一批；发布的是中位数，这两个字段让人看见区间。
    - ``confirmations``：每一颗被重测的候选（优胜者与噪声范围内的并列者）。
    - ``baseline_us``/``candidate_us``：确认那份报告的基线与候选微秒数。
    - ``evaluations``：上过 GPU 的评测次数。
    - ``hangs``：挂死次数。
    - ``trace``：每个评过的候选一行，按顺序。
    - ``candidates``：每个候选的 ``Candidate.brief()``。
    - ``ledger``：账本摘要。
    - ``profiles``：搜索期间对慢种子抓的 Instruments 痕迹（``search.profile_poor``）。
    - ``initial_budget_usd``：开跑时的预算（与结束时的 ``budget_usd`` 在上调后不同）。
    - ``budget_events``：每一次生效的预算变动（实时加预算与续跑），从旧到新。
    - ``resumes``：中断或预算停止之后续跑了几次。
    - ``search_seed_label``：搜索停下时正在被改动的那颗内核；确认可能发布另一颗。
    - ``search_seed_speedup``：那颗内核在搜索过程中的加速比。
    - ``confirm_winner_label``：新种子复测留下来的内核；确认没跑时为 None。
    - ``confirmation_source_sha256``：被确认优胜者的内核正文 sha256；确认没绑定源码时为 None。
    - ``target_id``/``backend``/``backend_version``/``protocol_version``：目标与协议标识。
    - ``session_budget_stop``：停止是因为剩余预算不足以再跑一场最近规模的会话；这是预算停止，但 control 文件里加预算不应默默继续它。
    """

    run_dir: str
    task: str
    dtype: str
    size: str
    model: str
    stop_reason: str = ""
    rounds: int = 0
    spent_usd: float = 0.0
    budget_usd: float | None = None
    best_label: str | None = None
    # 选中那一刻的加速比，不是复测值。复测若改了优胜者，这里仍是该内核在搜索过程中的测量。
    best_speedup: float | None = None  # as measured when selected
    # 新种子、两倍重复次数下的加速比，用来压掉「多次噪声测量里挑最大值」。
    confirmed_speedup: float | None = None  # fresh seed, more repetitions
    confirmed_status: str | None = None
    # 确认阶段每一批的值与最差那批：发布的是中位数，这两个字段让人看见区间。
    confirmed_values: list = field(default_factory=list)
    confirmed_worst: float | None = None
    confirmed_best: float | None = None
    confirmations: list = field(default_factory=list)  # every re-measured candidate (the best and its ties within noise)
    baseline_us: dict = field(default_factory=dict)
    candidate_us: float | None = None
    evaluations: int = 0
    hangs: int = 0
    trace: list = field(default_factory=list)  # one entry per evaluated candidate, in order
    candidates: list = field(default_factory=list)
    ledger: dict = field(default_factory=dict)
    profiles: list = field(default_factory=list)  # Instruments traces taken of slow seeds during the search (search.profile_poor)
    initial_budget_usd: float | None = None  # the budget the run started with (``budget_usd`` is the budget it ended with: they differ after a top-up)
    budget_events: list = field(default_factory=list)  # every budget change that took effect (live top-ups and resumes), oldest first
    resumes: int = 0  # how many times the run was continued after an interruption or a budget stop
    search_seed_label: str | None = None  # the kernel the search was mutating when it stopped; confirm may publish a different one
    search_seed_speedup: float | None = None  # that kernel's speedup as measured during the search
    confirm_winner_label: str | None = None  # the kernel the fresh-seed remeasure kept; None when confirmation did not run
    confirmation_source_sha256: str | None = None  # sha256 of the confirmed winner's kernel text; None when confirmation did not bind a source
    target_id: str = ""
    backend: str = ""
    backend_version: str = ""
    protocol_version: str = ""
    # The run stopped because the remaining budget could not cover another recent-sized
    # session; this is a budget stop, but a control-file top-up should not silently continue it.
    session_budget_stop: bool = False


# 源码的内容身份：评测缓存、复测去重、profiler 缓存都用它。同一份文本不会被当成两个内核。
_KERNEL_DECL = re.compile(r"\bkernel\s+void\b")


def _sha(source: str) -> str:
    """@brief 源码的内容身份。

    评测缓存、复测去重、profiler 缓存都用它。同一份文本不会被当成两个内核。

    @param source 内核正文。
    @return sha1 十六进制串。
    """
    return hashlib.sha1(source.encode()).hexdigest()


# 回复里抽不出带约定内核名的 Metal 块时追问的那一句。抽失败只消耗这一问，不会产生候选，也不会上 GPU。
def nudge_user(kernel_name: str, *, source_fence: str) -> str:
    """@brief 回复里抽不出带约定内核名的 fenced 块时送去追问的那一句话。

    抽失败只消耗这一问，不会产生候选，也不会上 GPU。

    @param kernel_name 契约要求的内核名（``kernel void {kernel_name}``）。
    @param source_fence 后端要求的代码块语言标记（例如 ``metal``）。
    @return 追问文本。
    """
    return (
        f"Your reply did not contain a ```{source_fence} code block defining `kernel void {kernel_name}`, "
        f"or a ```{source_fence} block with a /* taic-launch */ section and at least one `kernel void`. "
        f"Reply again with the complete kernel in exactly one ```{source_fence} block."
    )


def _strip_reasoning(messages: list[dict]) -> list[dict]:
    """@brief 返回一份去掉了 ``reasoning_content`` 的 ``messages`` 副本。

    截断的思考链可达数万 token，重试时毫无收益。

    @param messages 对话消息列表。
    @return 新的消息列表；assistant 消息里的 ``reasoning_content`` 被删掉。
    """
    out = []
    for m in messages:
        if m.get("role") == "assistant" and "reasoning_content" in m:
            m = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(m)
    return out


def _retry_without_reasoning(messages: list[dict]) -> list[dict]:
    """@brief 长度被截断、且回复里没有内核时的追问消息。

    回显的 reasoning 被删掉。只剩空白可见文本的 assistant 轮也一并删掉，这样同一个问题会被重新问一遍，
    而不是从一条空回复继续往下。``experiments/truncation_replay.py`` 拿它和其它尝试对比；
    如果那次回放选出别的赢家，要改的就是这个函数。

    @param messages 上一轮的消息列表。
    @return 删掉 reasoning 与空 assistant 轮之后的新消息列表。
    """
    messages = _strip_reasoning(messages)
    if messages and messages[-1].get("role") == "assistant" and not (messages[-1].get("content") or "").strip():
        messages = messages[:-1]
    return messages


# 种子慢于最强基线时，附在规划与每一路补丁反馈后面的 Instruments 文字。抓取失败或没开 profile_poor 时根本不会用到它，搜索照常进行。
def profiler_evidence_text(body: str) -> str:
    """@brief 种子被 Instruments 抓过痕迹后，附在规划与每一路补丁反馈后面的文字。

    种子慢于最强基线时才用得上；抓取失败或没开 ``profile_poor`` 时根本不会调用它，搜索照常进行。
    ``body`` 就是 ``profiles/r*_<seed>.txt`` 的内容。

    @param body Instruments 痕迹的文本形式。
    @return 追加到反馈末尾的提示文字（含读法说明与 ``body``）。
    """
    return (
        "\n\nProfiler evidence for this kernel (Instruments Metal System Trace; taken because the kernel is slower than the strongest baseline). "
        "Read it with the \"Instruments trace\" section of Skill 2. Name the finding or counter that your diagnosis rests on (spills, the saturated limiter, low occupancy, ...) "
        "and make the changes address it; if the evidence says nothing about your hypothesis, say so instead of inventing numbers:\n" + body
    )


def _notes_for_next(outer: str, notes: str, limit: int = 2500) -> str:
    """@brief 下一场会话 NOTES 块的内容。

    外层实测那一行由搜索写、不由 agent 写，固定在第一行。notes 是追加的，所以文件被截断时留下的是最新几行。

    @param outer 外层实测的一行摘要。
    @param notes 本场会话写下的 notes。
    @param limit notes 最多保留的字符数，超出时从头部截断。
    @return 以 ``外层实测：<outer>`` 开头的文本。
    """
    body = (notes or "").strip()
    if len(body) > limit:
        body = "…\n" + body[-limit:]
    return f"外层实测：{outer}\n{body}".strip()


def _operation_bound(report) -> str:
    """@brief 从报告的 roofline 与基线里取出一行说明。

    两个数缺一个就返回空串。下界高于基线时追加一句：不减少运算量的写法不可能超过它。

    @param report 评测报告。
    @return 一行中文说明，或空串。
    """
    from .feedback_extra import _roof_and_baselines

    roof, bases = _roof_and_baselines(report)
    bound = roof.get("lower_bound_us")
    usable = {k: float(v) for k, v in bases.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0}
    if not isinstance(bound, (int, float)) or isinstance(bound, bool) or bound <= 0 or not usable:
        return ""
    name, us = min(usable.items(), key=lambda kv: kv[1])
    line = f"本算法运算量下界 {float(bound):.0f} µs，最强基线 {name} {us:.0f} µs。"
    if float(bound) > us:
        line += "下界高于基线时，不减少运算量的写法不可能超过它。"
    return line


# 加速比是两次中位数之比。标准误的公式在 tcenv.bench，这里只按报告取出 cv 和 reps。没有逐次样本，也不把它叫成置信区间。
def speedup_se(rep: EvalReport | None) -> float:
    """@brief 一次测得加速比的近似相对标准误。

    加速比是两次中位数之比，公式在 ``tcenv.bench``，这里只按报告取出 cv 和 reps。报告里没有计时统计时返回 0。
    没有逐次样本，也不把它叫成置信区间。

    @param rep 评测报告，可为 None。
    @return 相对标准误；无统计数据时为 0。
    """
    from tcenv.bench import speedup_se_of_report

    return speedup_se_of_report(rep)


class Orchestrator:
    """@brief 一次搜索的全部状态与主循环。

    美元上限放进 ``CostLedger``；补丁的模型请求可以重叠，GPU 评测则共用一把 ``gpu_lock``，不能同时量。
    每次搜索有两个委员会（``first`` 与 ``second``），一轮一场 hmz 会话，共用同一本账和同一份会话记录。
    """

    # 一次搜索的全部状态。美元上限放进 CostLedger；K 路补丁的模型请求可以重叠，GPU 评测则共用一把 gpu_lock，不能同时量。
    def __init__(
        self,
        task_name: str,
        dtype: str,
        size: str,
        llm_cfg: LLMConfig,
        search: SearchConfig,
        *,
        run_dir: Path | None = None,
        evaluate_fn: Callable[..., EvalReport] | None = None,
        client: Any = None,
        log: Callable[[str], None] = print,
        seed: int | None = None,
        resuming: bool = False,
        from_run: Path | None = None,
        committees: list | None = None,
        agent_runner: Callable[..., Any] | None = None,
        external_roots: list[Path] | None = None,
        target_id: str = "apple-m4-max",
        from_catalog: bool = False,
    ) -> None:
        """@brief 装配一次搜索的全部状态，并写出 ``run_config.json`` 与首份 ``state.json``。

        ``resuming``：目录里已经有一次运行（见 ``taic.resume``）；在 ``restore()`` 把它装回来之前，不写其中任何东西。
        ``committees``：恰好两个，first 与 second；省略表示两个都复制 ``llm_cfg``。没有单会话模式。
        新运行不接管已经有文件的目录（目录锁不算文件）；续跑才读既有目录，且不改写 ``run_config.json``、不重新抽 ``run_seed``。
        配置写失败就停在这里：还没有账本、会话记录和 ``state.json``，也不会进入 ``run()``。

        @param task_name 任务名（``tasks.get_task``）。
        @param dtype 数据类型。
        @param size 尺寸。
        @param llm_cfg 模型配置（模型、端点、密钥、价格等）。
        @param search 搜索配置（轮数、预算、噪声门槛、确认批次等）。
        @param run_dir 运行目录；None 时在 ``runs/<target_id>/<时间戳>_<task>_<dtype>_<model>`` 下新建。
        @param evaluate_fn 评测函数；None 时用 ``tcenv.evaluate.evaluate``。
        @param client 注入的 HTTP 客户端或脚本客户端（测试用）。
        @param log 日志输出函数，默认 ``print``。
        @param seed 显式指定的 ``run_seed``；None 时随机抽 64 位。
        @param resuming 目录里已有一次运行，按续跑处理。
        @param from_run ``--from`` 指向的既有运行目录。
        @param committees 委员会列表；None 时由 ``llm_cfg`` 生成 first 与 second。
        @param agent_runner 写核函数（测试注入）；None 时走 hmz。
        @param external_roots 扫「别的运行里的核」的根目录；None 表示默认扫仓库 ``runs/``。
        @param target_id 目标设备 id。
        @param from_catalog 从 kernel catalog 取上一版内核。
        @raises RunStartError 目标不可用、委员会配置非法、``--seed`` 越界、目录已有文件，或 ``run_config.json`` 写失败时。
        """
        cached = get_task(task_name, dtype, size)
        from dataclasses import replace
        from tcenv.target import TargetError, require_active

        try:
            self.target = require_active(target_id)
        except TargetError as e:
            raise RunStartError(str(e)) from e
        self.target_id = self.target.id
        # get_task 有缓存。replace 出一份新的 Task，不改缓存里那份的 contract。
        self.task = replace(cached, contract=replace(cached.contract, gpu=self.target.display_name))
        self.task_name, self.dtype, self.size = task_name, dtype, size
        self.cfg, self.search = llm_cfg, search
        from tcenv.backends.registry import load_backend

        self.backend = load_backend(self.target.backend)
        try:
            self.committees = default_committees(llm_cfg) if committees is None else order_committees(list(committees))
        except LLMError as e:
            raise RunStartError(str(e)) from e
        self._committee_name = self.committees[0].name
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir) if run_dir else REPO_ROOT / "runs" / self.target_id / f"{stamp}_{task_name}_{dtype}_{llm_cfg.model}"
        self.log = log
        hw_feedback = bool(search.hw_feedback and self.backend.capabilities.supports_hw_features)
        self.system = prompts.system_prompt(self.task, use_tools=llm_cfg.use_tools, profiler_evidence=search.profile_poor, hw_feedback=hw_feedback, target_id=self.target_id)
        self.from_run = Path(from_run) if from_run else None
        self.from_catalog = from_catalog
        # 新运行不接管已经有文件的目录。续跑（resuming=True）才读既有目录，并且不改写 run_config.json、不重新抽 run_seed。
        # 目录锁不算「文件」：`taic run --run-dir X` 在构造之前就先占锁，否则每个显式指定的新目录都会被自己拒绝。
        if not resuming and self.run_dir.exists():
            lock_name = control.lock_path(self.run_dir).name
            if any(item.name != lock_name for item in self.run_dir.iterdir()):
                raise RunStartError(f"refused: {self.run_dir} already holds files; resume it with `taic resume` or choose a new directory")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if resuming:
            self.run_seed = _load_run_seed(self.run_dir)
        elif seed is None:
            self.run_seed = secrets.randbits(64)
        else:
            chosen = _valid_run_seed(seed)
            if chosen is None:
                raise RunStartError("refused: --seed must be a decimal integer with 0 <= seed < 2**64")
            self.run_seed = chosen
        self.seed = self.run_seed
        # 配置写失败就停在这里：还没有账本、转录和 state.json，也不会进入 run()。
        if not resuming:
            self._write_run_config()
        # 预算闸门的账本。之后每次 LLM.chat 发请求前，CostLedger.check 先 refresh_budget（读 control.json），再比较已花费。
        # 两个委员会共用这一本账和下面那一份转录。换委员会不新开预算。
        self.ledger = CostLedger(llm_cfg, search.budget_usd)
        # 已存在的 control.json 先被忽略，直到文件再变：那是上一会话的请求，启动者已经折进当前预算。损坏的文件只记一笔，不改上限。
        self.ledger.watch_control(self.run_dir / control.CONTROL_FILE)
        self.ledger.on_budget_change = self._budget_changed
        self.ledger.on_control = self._control_limits
        self.ledger.on_control_problem = lambda msg: self.log(f"  {msg}")
        # 并发度至少为分支数 K（search.branches）。模型请求因此可以同时在飞；占 GPU 的评测不走这个信号量。
        self._llm_slots = max(2, search.branches)
        self._client_injected = client is not None
        self.llm = LLM(llm_cfg, self.ledger, Transcript(self.run_dir / "transcript.jsonl"), client=client, max_concurrency=self._llm_slots)
        self._llms = [self.llm]
        self.evaluate_fn = evaluate_fn or evaluate
        # 同一时刻只允许一次 GPU 评测或 Instruments。两份内核抢同一块卡会把计时测脏，挂死时也不能再有内核在跑。
        self.gpu_lock = asyncio.Lock()
        # 升种子要等复查回来再写。锁盖住比较和写入，另一条分支不能对着旧种子同时通过。
        self._promote_lock = asyncio.Lock()
        self.t0 = time.time()
        self.best: Candidate | None = None
        self.candidates: list[Candidate] = []
        self.cache: dict[str, EvalReport] = {}
        self.trace: list[dict] = []
        self.history: list[str] = []
        self.hangs = 0
        self.crashes_in_a_row = 0
        self.evaluations = 0
        self._phase = "start"
        self._round = 0
        self._confirmations: list = []
        self.profiles: list[dict] = []
        self._profile_text: dict[str, str] = {}
        self.llm.progress_path = self.run_dir / "llm.json"
        self._t_offset = 0.0  # seconds of search time spent by earlier sessions of a resumed run
        self._started_at = self.t0
        self._stagnant = 0
        self._round_plans: list[dict] = []
        self._resume = None  # a ``taic.resume.Snapshot`` while a resumed run has not yet caught up with where it stopped
        self._resumes = 0
        self._branch_fail: dict[str, str] = {}  # branch label -> why its last attempt failed (used when search.rich_history is on)
        self._branch_parent: dict[str, str] = {}
        self._search_seed: Candidate | None = None
        self._confirm_winner: Candidate | None = None
        # 测试注入的写核函数。None 时走 hmz。注入之后不再做 hmz 预检，也不去扫仓库里的 runs。
        if agent_runner is None and callable(getattr(client, "script", None)):
            from .kernel_agent import agent_for_script

            agent_runner = agent_for_script(client.script, self.task.contract.kernel_name)
        self.agent_runner = agent_runner
        # 测试注入了 HTTP 客户端时不扫仓库的 runs，避免把练习核判成别的运行的核。
        if external_roots is None and client is not None:
            self._external_roots = []
        else:
            self._external_roots = external_roots
        self._external_texts: set[str] | None = None
        self._from_text = ""
        self._agent_failures = 0
        self._failure_feedback = ""
        self._notes_summary = ""
        self._evidence = ""
        self._feedback_seq: int | None = None
        self._session_briefs: list = []
        self._session_costs: list[float] = []
        # 续跑时目录里已有上一会话的 state.json。先空写会把看板抹掉，所以等到 restore 把账本和候选装回来再发布。
        # 新运行的初始稿用角色 0。续跑要等知道轮次后再绑，不能在这里先绑成角色 0。
        if not resuming:
            self._activate_round(0)
            self._publish_state()

    # 续跑要把上一会话已经花掉的墙钟算进去，否则轨迹上的时间从 0 重新开始，和账本对不上。
    def _elapsed(self) -> float:
        """@brief 整场搜索的耗时，含续跑前各会话已经花掉的墙钟。

        @return 秒数。
        """
        return self._t_offset + (time.time() - self.t0)

    # 第 1 轮和初始稿是 first；之后按轮号在两个委员会之间换。续跑只看轮号，不看上一轮的对话。
    def _activate_round(self, rnd: int) -> None:
        """@brief 绑定第 ``rnd`` 轮使用的委员会。

        第 1 轮和初始稿用 first；之后按轮号在两个委员会之间换，模型 id 没变时名字也照样换。
        端点、密钥、模型都相同时沿用现有 HTTP 客户端；模型 id 变了就同步账本里的价格。
        端点或密钥不同则在同一个账本、同一份会话记录上另开一个客户端。测试注入的客户端被两个委员会共用，
        所以一条脚本队列能服务整场运行。续跑只看轮号，不看上一轮的对话。

        @param rnd 轮号（0 或 1 都落到 first）。
        """
        committee = self.committees[(max(int(rnd), 1) - 1) % 2]
        self._committee_name = committee.name
        nxt = committee.members[0].cfg
        cur = self.llm.cfg
        if cur is nxt:
            return
        same_model = cur.model == nxt.model
        same_endpoint = cur.base_url == nxt.base_url and cur.api_key == nxt.api_key
        if self._client_injected or same_endpoint:
            if not same_model:
                self._use_cfg(nxt)
            return
        self._open_llm(nxt, update_prices=not same_model)

    def _use_cfg(self, cfg: LLMConfig) -> None:
        """@brief 沿用当前 HTTP 客户端，只把配置换成 ``cfg``。

        这是模型 id 变了的情形，所以账本的价格也跟着换；``cfg.api_key`` 会被登记进会话记录的脱敏列表。

        @param cfg 新的模型配置。
        """
        if cfg.api_key:
            self.llm.transcript._secrets = list({*self.llm.transcript._secrets, cfg.api_key})
        self.llm.cfg = cfg
        self.ledger.cfg = cfg

    def _open_llm(self, cfg: LLMConfig, *, update_prices: bool) -> None:
        """@brief 端点或密钥不同时另开一个客户端，账本与会话记录不变。

        新客户端和其它客户端一样在 ``run`` 结束时关闭。

        @param cfg 新的模型配置。
        @param update_prices 是否同步账本里的价格（模型 id 变过时为 True）。
        """
        fresh = LLM(cfg, self.ledger, self.llm.transcript, client=None, max_concurrency=self._llm_slots)
        fresh.progress_path = self.llm.progress_path
        self._llms.append(fresh)
        self.llm = fresh
        if update_prices:
            self.ledger.cfg = cfg

    def _handoff(self, rnd: int) -> str:
        """@brief 轮次边界上的交接提示。

        初始稿与第 1 轮返回空串。之后每一轮都点名它所属的委员会，模型 id 没变时也点名。

        @param rnd 轮号。
        @return 交接文本；第 1 轮及以前为空串。
        """
        if rnd <= 1:
            return ""
        return handoff_for(self.committees[(rnd - 1) % 2].name)

    async def _close_clients(self) -> None:
        """@brief 关闭本次运行开过的所有 HTTP 客户端。

        同一个客户端只关一次；脚本客户端没有东西要关。
        """
        seen: set[int] = set()
        for llm in self._llms:
            ident = id(llm.client)
            if ident in seen:
                continue
            seen.add(ident)
            await llm.aclose()

    # 给 `taic resume` 留的设置。故意不写 API key 和端点。写失败向上抛，调用方不得进入 run()。
    def _write_run_config(self) -> None:
        """@brief 写下这次运行的设置，供 ``taic.cli resume`` 使用。

        故意不写凭据：API key 与端点都不进这个文件。文件写不出去时抛 ``RunStartError``，调用方不得进入 ``run()``。
        这次尝试新建的目录可能留空；既有的旧运行目录永不删除或覆盖（那种情况在本方法之前就被拒绝）。
        内容含运行种子、委员会、系统提示与契约的哈希、git 身份，以及「成本可能不完整」的说明。

        @raises RunStartError 当 ``run_config.json`` 或 ``system_prompt.txt`` 写失败时。
        """
        llm = {k: v for k, v in asdict(self.cfg).items() if k not in ("api_key", "base_url")}
        contract_text = self.task.contract.describe()
        triton_text = self.task.triton_source
        commit, dirty = _git_identity(REPO_ROOT)
        body = {
                "schema_version": RUN_CONFIG_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "target": {
                    "id": self.target_id,
                    "backend": self.target.backend,
                    "backend_version": self.target.backend_version,
                    "display_name": self.target.display_name,
                },
                "task": self.task_name, "dtype": self.dtype, "size": self.size, "seed": self.seed, "run_seed": self.run_seed,
                "llm": llm, "search": asdict(self.search),
                "system_prompt_sha1": hashlib.sha1(self.system.encode()).hexdigest(),
                "contract_description_sha256": hashlib.sha256(contract_text.encode("utf-8")).hexdigest(),
                "triton_source_sha256": hashlib.sha256(triton_text.encode("utf-8")).hexdigest(),
                "task_fingerprint": task_fingerprint(self.task),
                "hash_scope": "task_fingerprint hashes the contract, the triton source, the naive kernel, cases, tolerances, flops, min_bytes and the ieee/domain meta. The two hashes above are contract.describe() and task.triton_source. reference and baseline implementations are not included",
                "git_commit": commit,
                "git_dirty": dirty,
                "settings_source": "llm and search values passed into Orchestrator at start (config file, preset and command-line overrides already applied)",
                "cost_may_be_incomplete": True,
                "cost_note": "A process can die after the provider bills a request and before transcript.jsonl records it, so a resumed ledger can under-count the real bill. A budget_events.json line that fails to land is dropped; that file is not a complete bill. This is not a hard dollar cap. Reconciling with the provider bill needs separate authorization.",
                "created_at": time.time(),
                "launch_protocol": 1,
                "loop_protocol": "flame_chase_1",
                "agent_guidance": self._agent_guidance_record(),
        }
        # 两个委员会和各自唯一的成员。没有密钥，也没有端点。
        body["committees"] = committees_record(self.committees)
        try:
            control.write_json_atomic(self.run_dir / control.CONFIG_FILE, body)
            control.write_text_atomic(self.run_dir / "system_prompt.txt", self.system)
        except OSError as e:
            raise RunStartError(f"cannot start: run_config.json was not written ({e})") from e

    def _agent_guidance_record(self) -> dict:
        """@brief 冻结每场会话都会用到的指引文件，并记下 skill 的哈希。

        ``device.md``、``experience.md`` 与后端的 reference skill 被抄进 ``agent_guidance/``；续跑读这份目录，
        不再从仓库抄一次。skill 文件留在 hmz 加载它的位置，只记哈希。

        @return 文件名到 sha256 的映射（含 ``SKILL.md``）。
        """
        from .kernel_agent import FLOW
        from .target import skill_path
        from tcenv.backends.registry import load_backend

        backend = load_backend(self.target.backend)
        dest = self.run_dir / "agent_guidance"
        dest.mkdir(parents=True, exist_ok=True)
        hashes = {}
        reference_name = f"{backend.reference_skill}.md"
        for name, key in (("device.md", "device"), ("experience.md", "experience"), (reference_name, backend.reference_skill)):
            src = skill_path(key, self.target_id)
            data = src.read_bytes() if src.is_file() else b""
            (dest / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        skill = FLOW / "skills" / "taic-tools" / "SKILL.md"
        data = skill.read_bytes() if skill.is_file() else b""
        hashes["SKILL.md"] = hashlib.sha256(data).hexdigest()
        return hashes

    def _cache_this(self, source: str) -> bool:
        """@brief 这份源码的报告能不能进评测缓存。

        ``launch_protocol`` 之前存下的报告不是 taic-launch 内核的判决。本代码启动的运行会写 ``launch_protocol: 1``；
        缺这个键的续跑目录是单内核路径评出来的，所以带 marker 的源码下次被搜索要它时会重新评一遍。

        @param source 内核正文。
        @return 可以缓存为 True；带 ``taic-launch`` 标记且目录协议不符时为 False。
        """
        if "taic-launch" not in source:
            return True
        path = self.run_dir / control.CONFIG_FILE
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return body.get("launch_protocol") == 1

    # control.json 在下一次预算闸门被接受之后通知这里。上限只升不降到已花费之下（这是 refresh_budget 的不变量）；本函数把新上限同步到 search.budget_usd，搜索继续，而不是停掉。
    def _budget_changed(self, old: float, new: float, doc: dict) -> None:
        """@brief 账本从运行目录的 control 文件拿到了一份新的总预算。

        ``control.json`` 在下一次预算闸门被接受之后才通知这里。上限只升不降到已花费之下（这是 ``refresh_budget``
        的不变量）；本函数把新上限同步到 ``search.budget_usd``，搜索继续而不是停掉，并记一条 budget event、刷新 ``state.json``。

        @param old 旧上限（美元）。
        @param new 新上限（美元）。
        @param doc control 文件内容（读其中的 ``who``）。
        """
        self.search.budget_usd = new
        who = str(doc.get("who") or "dashboard")
        self.log(f"  budget ${old:.2f} -> ${new:.2f} (requested by {who}; spent ${self.ledger.total_usd:.3f})")
        written = control.record_event(self.run_dir, "applied", old, new, who=who, spent_usd=self.ledger.total_usd)
        if not written.get("persisted"):
            self.log("  budget event was not saved; budget_events.json is not a complete record")
        self._publish_state()

    # 控制文件里的 max_rounds / patience 只升不降。仪表盘的「重置」用它去掉次数上限；搜索仍会在新预算花完时停下。
    def _control_limits(self, doc: dict) -> None:
        """@brief control 文件要求更大值时抬高 ``max_rounds`` 与 ``patience``。

        这两个上限只升不降。仪表盘的「重置」用它去掉次数上限；搜索仍会在新预算花完时停下。
        生效后打一行日志并刷新 ``state.json``。

        @param doc control 文件内容。
        """
        notes = []
        for key in ("max_rounds", "patience"):
            raw = doc.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            value = int(raw)
            current = int(getattr(self.search, key))
            if value > current:
                setattr(self.search, key, value)
                notes.append(f"{key} {current} -> {value}")
        if notes:
            self.log("  " + ", ".join(notes) + f" (requested by {doc.get('who') or 'dashboard'})")
            self._publish_state()

    # 给外部看板写 state.json。写失败被吞掉，搜索不能因为看板上的一次 IO 错误停下来。
    def _publish_state(self, phase: str | None = None, *, inflight: str | None = None, stop_reason: str = "") -> None:
        """@brief 重写 ``state.json`` 给外部看板。

        写失败被吞掉：搜索不能因为看板上的一次 IO 错误停下来。

        @param phase 阶段名；None 时沿用当前阶段。
        @param inflight 正在飞的会话标签，可为 None。
        @param stop_reason 停止原因，运行中为空串。
        """
        if phase is not None:
            self._phase = phase
        try:
            from tcenv.events import publish

            publish(self.run_dir / "state.json", {
                "schema_version": STATE_SCHEMA_VERSION,
                "pid": os.getpid(),
                "updated_at": time.time(),
                "started_at": self._started_at,
                "phase": self._phase,
                "task": self.task_name,
                "dtype": self.dtype,
                "size": self.size,
                "target_id": self.target_id,
                "backend": self.target.backend,
                "model": self.llm.cfg.model,
                "committee": self._committee_name,
                "round": self._round,
                "max_rounds": self.search.max_rounds,
                "spent_usd": self.ledger.total_usd,
                # 看板上的上限是账本当前值。运行中 control.json 加过预算后，它和 initial_budget_usd 可以不同。
                "budget_usd": self.ledger.budget_usd,
                "initial_budget_usd": self.ledger.initial_budget_usd,
                "control_supported": True,  # this process reads <run_dir>/control.json before every budget gate (live top-up)
                "resumes": self._resumes,
                "stagnant": self._stagnant,
                "history": self.history,
                "round_plans": self._round_plans,
                "best_speedup": self.best.speedup if self.best else None,
                "best_label": self.best.label if self.best else None,
                "search_seed_label": self._search_seed.label if self._search_seed else (self.best.label if self.best else None),
                "search_seed_speedup": self._search_seed.speedup if self._search_seed else (self.best.speedup if self.best else None),
                "confirm_winner_label": self._confirm_winner.label if self._confirm_winner else None,
                "agent_failures": self._agent_failures,
                "hangs": self.hangs,
                "crashes_in_a_row": self.crashes_in_a_row,
                "evaluations": self.evaluations,
                "stop_reason": stop_reason,
                "ledger": self.ledger.summary(),
                "inflight_label": inflight,
                "trace": self.trace,
                "confirmations": self._confirmations,
                "run_dir": str(self.run_dir),
            })
        except Exception:
            return

    # ------------------------------------------------------------------ resume
    # 把上一会话装回来再继续，而不是从头生成。已评过的稳定结论进缓存；timeout 与 crash 不进。上一会话的 hangs 一并恢复，已经触达 max_hangs 的运行，续跑后第一次完整评测仍会停掉整次搜索。
    def restore(self, snap: Any, *, dry_run: bool = False) -> None:
        """@brief 把上一场会话装回来，让 ``run()`` 接着跑而不是从头生成。

        ``snap`` 是 ``taic.resume.Snapshot``。装回账本、每个候选及其报告、最优内核、评测缓存、轨迹、
        已经试过的假设、计数器和 profiler 证据。已评过的稳定结论进缓存；``timeout`` 与 ``crash`` 不进（那是环境故障，
        不是这颗内核的永久判决）。上一会话的 hangs 一并恢复，已经触达 ``max_hangs`` 的运行，续跑后第一次完整评测仍会停掉整次搜索。
        系统提示沿用开跑时那份（前缀缓存与实验条件不变），提示文件后来改了也不换。
        续跑的委员会由将要进入的轮号决定，不把上一轮的对话带进来。

        ``dry_run`` 只把状态装进内存，什么都不写（不写会话记录 marker，也不写 ``state.json``）。

        @param snap 续跑快照。
        @param dry_run 只装载不落盘（跳过快照会话记录与 ``state.json``）。
        """
        self.ledger.restore(snap.requests)
        self.ledger.initial_budget_usd = snap.initial_budget_usd
        self.ledger.t0 = time.time() - (snap.requests[-1].get("t") or 0.0 if snap.requests else 0.0)
        self.candidates, self.cache, self.best = [], {}, None
        by_idx = {t["idx"]: t for t in snap.trace}
        for c in snap.candidates:
            t = by_idx.get(c["idx"], {})
            cand = Candidate(idx=c["idx"], label=c["label"], source=c["source"], report=c["report"], plan=c.get("plan"), cost_usd=t.get("cost_usd", 0.0), t_s=t.get("t_s", 0.0),
                             is_best=False, within_noise=bool(t.get("within_noise")), correctness_only=self.search.correctness_only,
                             recheck_rejected=bool(c.get("recheck_rejected")))
            self.candidates.append(cand)
            # 与 _eval 相同：超时和 worker 崩溃是环境故障，不能当成这颗内核的永久判决。
            if cand.report is not None and cand.report.status not in ("timeout", "crash") and self._cache_this(cand.source):
                self.cache.setdefault(_sha(cand.source), cand.report)
        self.best = self._by_label(snap.best_label)
        if self.best:
            self.best.is_best = True
        self.trace = [dict(t) for t in snap.trace]
        self._session_costs = []
        previous = 0.0
        for item in sorted(self.trace, key=lambda row: int(row.get("idx") or 0)):
            try:
                total = float(item.get("cost_usd") or 0.0)
            except (TypeError, ValueError):
                continue
            delta = total - previous
            if delta > 0:
                self._session_costs.append(delta)
            previous = max(previous, total)
        self._session_costs = self._session_costs[-8:]
        self.history = list(snap.history)
        self.hangs, self.evaluations, self.crashes_in_a_row = snap.hangs, snap.evaluations, snap.crashes_in_a_row
        self.profiles = [dict(p) for p in snap.profiles]
        for label, body in snap.profile_bodies.items():
            cand = self._by_label(label)
            if cand is not None:
                self._profile_text[_sha(cand.source)] = profiler_evidence_text(body)
        # 续跑必须沿用开跑时的系统提示。提示文件后来改了也不换，否则前缀缓存和实验条件都变了。
        if snap.system:
            self.system = snap.system  # the very prompt the run began with (prefix cache, same experiment conditions), whatever the prompt files say now
        self._t_offset = snap.elapsed_s
        self._started_at = snap.started_at or self.t0
        self._resumes = snap.resumes + 1
        self._stagnant = snap.stagnant
        self._agent_failures = int(getattr(snap, "agent_failures", 0) or 0)
        self._restore_handoff()
        self._round = snap.partial.rnd if snap.partial else snap.rounds_done
        self._resume = snap
        # 续跑的角色由将要进入的轮号决定，不把上一轮的对话带进来。
        self._activate_round(snap.partial.rnd if snap.partial else snap.next_round)
        # 试跑只在内存里装好状态。写转录或 state.json 会让一次「看看能不能续」变成真的动了运行目录。
        if dry_run:
            return
        self.llm.transcript.log({"event": "resume", "n": self._resumes, "spent_usd": self.ledger.total_usd, "budget_usd": self.ledger.budget_usd, "round": self._round, "best": snap.best_label, "pid": os.getpid()})
        try:  # requests that were in flight when the earlier session died are not in flight any more
            from tcenv.events import publish

            prev = control.read_json(self.run_dir / "llm.json") or {}
            publish(self.run_dir / "llm.json", {"inflight": {}, "updated_at": time.time(), "last": prev.get("last")})
        except Exception:  # noqa: BLE001
            pass
        self._publish_state("resume")

    # ------------------------------------------------------------------ evaluation
    # 每个评过的候选都留在 run 目录里。搜索中途停下，或稍后 restore，仍能对照当时的反馈。
    def _save(self, c: Candidate) -> None:
        """@brief 把一颗候选的源码、报告与反馈写进它自己的目录。

        每个评过的候选都留在 run 目录里，搜索中途停下或稍后 restore，仍能对照当时的反馈。
        写入用原子替换：写到一半被杀掉时，留下的是整份旧的或整份新的，不会出现「截断的源码配完整报告」。
        写报告前会把当时种子的实测微秒数记进 ``benchmark['seed_gpu_us_median']``。

        @param c 要写盘的候选。
        """
        d = self.run_dir / "candidates" / f"{c.idx:03d}_{c.label}"
        d.mkdir(parents=True, exist_ok=True)
        # 原子替换：写到一半被杀掉时，留下的是整份旧的或整份新的，不会出现「截断的源码配完整报告」。
        control.write_text_atomic(d / self.kernel_name, c.source)
        if c.report:
            if self.best is not None and self.best.report is not None and self.best is not c:
                seed_gpu = self.best.report.benchmark.get("candidate_gpu_us_median")
                if isinstance(seed_gpu, (int, float)) and not isinstance(seed_gpu, bool) and seed_gpu > 0:
                    c.report.benchmark["seed_gpu_us_median"] = float(seed_gpu)
            control.write_json_atomic(d / "report.json", c.report.to_json())
            control.write_text_atomic(d / "feedback.txt", format_feedback(c.report))

    # a 必须比 b 快出这么多倍才算真正变快：至少 min_improvement，并且要盖过两者加速比标准误合成后再乘 noise_z。差一点只记 within_noise，不能当下一轮种子。
    def _required_gain(self, a: Candidate, b: Candidate) -> float:
        """@brief ``a`` 要比 ``b`` 快多少倍才算真正变快。

        至少是 ``min_improvement``，并且要盖过两者加速比标准误合成后再乘 ``noise_z``。差一点只记 ``within_noise``，
        不能当下一轮种子。

        @param a 待比较的候选。
        @param b 参照的候选（通常是当前种子）。
        @return 需要的加速比倍数。
        """
        return required_gain(self.search.min_improvement, self.search.noise_z, speedup_se(a.report), speedup_se(b.report))

    # 一颗候选的完整评测，也是唯一更新当前种子的地方。不安全、算错、或快不过测量噪声，都留在记录里但种子不变。默认 max_hangs 为 1，所以一次 timeout 就抛 StopRun，整次搜索停；连续 3 次 crash 同样停。
    async def _eval(self, source: str, label: str, *, plan: str | None = None) -> Candidate:
        """@brief 一颗候选的完整评测（在 GPU 上串行），也是唯一更新当前种子的地方。

        不安全、算错、或快不过测量噪声的候选都留在记录里，但种子不变。默认 ``max_hangs`` 为 1，所以一次 timeout
        就抛 ``StopRun`` 停掉整次搜索；连续 3 次 crash 同样停。贴回来的内核（与当前种子、本场更早的候选、
        别的运行里的核逐字相同）不送 GPU，直接记 ``unchanged`` / ``duplicate`` / ``external_duplicate``。同一源码的稳定
        结论可复用缓存里的报告，但要沿用当时那份报告的 seed，不能把它说成一次新种子上的测量；``timeout`` 与 ``crash`` 不进缓存。
        ``search.differential_recheck`` 打开时，将要升种子或落入噪声内并列的候选先用另一颗种子做不计时的正确性复查，
        复查不过就不升种子；复查在这把 ``_promote_lock`` 里等待，另一条分支不能同时对着旧种子通过比较。
        评测结束后写盘、追加 ``trace``、刷新 ``state.json`` 并打一行日志。

        @param source 内核正文。
        @param label 候选标签。
        @param plan 写这颗内核时用的规划文本，可为 None。
        @return 评测完（或复用结论）的候选。
        @raises StopRun 当 hangs 已达 ``max_hangs``（排队的评测不再上 GPU），或连续三次 worker 崩溃时。
        """
        cand = Candidate(idx=len(self.candidates), label=label, source=source, plan=plan, correctness_only=self.search.correctness_only)
        copied = self._copied_report(source)
        self.candidates.append(cand)
        self._publish_state("evaluate", inflight=label)
        key = _sha(source)
        phase_seed = self._phase_seed("search", label, source)
        # 贴回来的核不评测。当前种子、本场更早的候选、别的运行里的核，三条都是逐字比较。
        if copied is not None:
            cand.report = copied
            seed_from = copied.status
            used_seed = 0
        elif key in self.cache:
            # 同一源码的稳定结论可复用。复用的是当时那份报告的 seed，不能把它说成一次新种子上的测量。
            cand.report = self.cache[key]
            seed_from = "cached report"
            used_seed = cand.report.seed
        else:
            async with self.gpu_lock:
                # 已经有超时把 hangs 推到 max_hangs 时，还在排队的评测不再上 GPU，整次搜索在这里停下。
                if self.hangs >= self.search.max_hangs:
                    raise StopRun("gpu_hang_circuit_breaker")
                rep = await asyncio.to_thread(
                    self.evaluate_fn, self.task_name, self.dtype, self.size, source,
                    mode="quick" if self.search.correctness_only else "full", seed=phase_seed, reps=self.search.eval_reps,
                    progress_path=self.run_dir / "eval_progress.json",
                    **({"fast_bench": True} if self.search.fast_bench and not self.search.correctness_only else {}),
                    **({"require_contract_launch": True} if self.search.require_contract_launch else {}),
                    measure_serial=self.search.measure_serial,
                    eval_cache=self.search.eval_cache,
                    target_id=self.target_id,
                )
                self.evaluations += 1
            cand.report = rep
            seed_from = "search" if phase_seed is not None else "random"
            used_seed = phase_seed if phase_seed is not None else rep.seed
            # timeout / crash 不进缓存：那是环境故障，不能当成这颗内核的永久判决。
            if rep.status not in ("timeout", "crash"):
                self.cache[key] = rep
            # 默认 max_hangs 为 1：这一次 timeout 就够让函数末尾抛 StopRun。quick_check 的超时也记在同一个 hangs 上。
            if rep.status == "timeout":
                self.hangs += 1
            self.crashes_in_a_row = self.crashes_in_a_row + 1 if rep.status == "crash" else 0
        cand.cost_usd, cand.t_s = self.ledger.total_usd, self._elapsed()
        async with self._promote_lock:
            best = self.best
            relation = relation_to_best(
                cand.eligible, cand.speedup, None if best is None else best.speedup,
                speedup_se(cand.report), 0.0 if best is None else speedup_se(best.report),
                min_improvement=self.search.min_improvement, noise_z=self.search.noise_z,
            )
            would_replace = relation == "replace"
            would_tie = relation == "tie"
            recheck_doc = None
            # 闸门默认关。打开时，将要升种子或落入噪声内并列的候选先用另一颗种子做不计时的正确性复查。复查不过就不升种子。
            # 复查在这把锁里等待，所以另一条分支不能同时对着旧种子通过比较。
            if self.search.differential_recheck and (would_replace or would_tie):
                recheck_doc = await self._differential_recheck(cand, "new_best" if would_replace else "within_noise")
                if recheck_doc is not None:
                    would_replace = False
                    would_tie = False
                    if recheck_doc.get("counts_as_kernel_error"):
                        cand.recheck_rejected = True
            if would_replace:
                if self.best:
                    self.best.is_best = False
                cand.is_best = True
                self.best = cand
            elif would_tie:
                cand.within_noise = True
        self._save(cand)
        if recheck_doc is not None:
            self._store_recheck(cand, recheck_doc)
        r = cand.report
        self.trace.append(
            {"idx": cand.idx, "label": label, "status": r.status, "speedup": r.speedup, "cost_usd": cand.cost_usd, "t_s": cand.t_s, "best_speedup": self.best.speedup if self.best else None, "improved": cand.is_best, "within_noise": cand.within_noise, "eval_seed": used_seed, "eval_seed_from": seed_from, "differential_recheck": None if recheck_doc is None else recheck_doc["class"], "parent_label": self._parent_of(label)}
        )
        self._publish_state(inflight=None)
        sp = f"{r.speedup:.3f}x" if r.speedup else "-"
        tag = "NEW BEST " if cand.is_best else ("within noise of the best " if cand.within_noise else "")
        self.log(f"  [{label}] {r.status:<13s} speedup {sp:>8s}  {tag}(${cand.cost_usd:.3f}, {cand.t_s:.0f}s)  {('' if r.status == 'ok' else r.message[:110].replace(chr(10), ' '))}")
        # 一次超时或连续三次 worker 崩溃：停止整次搜索，不再开新的模型回合。
        if self.hangs >= self.search.max_hangs or self.crashes_in_a_row >= 3:
            raise StopRun("gpu_hang_circuit_breaker" if self.hangs >= self.search.max_hangs else "repeated_worker_crashes")
        return cand

    def _parent_of(self, label: str) -> str | None:
        """@brief 找出一个标签所属分支的父标签。

        在 ``_branch_parent`` 里查找：键既匹配标签本身，也匹配以 ``键 + "."`` 开头的标签（例如 ``r2.p1`` 的
        ``r2.p1.fix1``）。结果写进 trace 的 ``parent_label``。

        @param label 候选标签。
        @return 父标签；没有记录时为 None。
        """
        for key, parent in self._branch_parent.items():
            if label == key or label.startswith(key + "."):
                return parent
        return None

    def _phase_seed(self, phase: str, label: str, source: str) -> int | None:
        """@brief 记录过种子的运行里，某个阶段该用的派生种子。

        没有 ``run_seed`` 的运行返回 None，让评测器自己随机抽。

        @param phase 评测阶段（``search``/``confirm``/``recheck``/``domain``）。
        @param label 评测前就确定的标签。
        @param source 内核正文。
        @return 派生种子；运行没有 ``run_seed`` 时为 None。
        """
        if self.run_seed is None:
            return None
        return eval_seed(self.run_seed, phase, label, source)

    async def _differential_recheck(self, cand: Candidate, speed_relation: str) -> dict | None:
        """@brief 用 ``recheck`` 种子做一次不计时的正确性复查。

        通过时返回 None。返回文档就表示这颗候选不升种子。``timeout`` 与 ``crash`` 只推进既有的挂死/崩溃计数，
        并把 ``counts_as_kernel_error`` 记为 false，所以它们不会被当成「内核算错」缓存下来。
        返回的文档不是 ``VerifyResult``，也不带 ``counterexample``。

        @param cand 待复查的候选。
        @param speed_relation 与当前种子的关系（``new_best`` 或 ``within_noise``），写进记录。
        @return 失败记录；复查通过时为 None。
        @raises StopRun 当 hangs 已达 ``max_hangs`` 时（不再上 GPU）。
        """
        phase_seed = self._phase_seed("recheck", cand.label, cand.source)
        async with self.gpu_lock:
            if self.hangs >= self.search.max_hangs:
                raise StopRun("gpu_hang_circuit_breaker")
            rep = await asyncio.to_thread(
                self.evaluate_fn, self.task_name, self.dtype, self.size, cand.source, mode="quick", seed=phase_seed, reps=3,
                progress_path=self.run_dir / "eval_progress.json",
                **({"require_contract_launch": True} if self.search.require_contract_launch else {}),
                measure_serial=self.search.measure_serial,
                eval_cache=self.search.eval_cache,
                target_id=self.target_id,
            )
        self.evaluations += 1
        if rep.status == "ok":
            return None
        kernel_error = rep.status not in ("timeout", "crash")
        if rep.status == "timeout":
            self.hangs += 1
        elif rep.status == "crash":
            self.crashes_in_a_row += 1
        else:
            self.crashes_in_a_row = 0
        return self._failure_record(cand, rep, phase_seed, speed_relation, kernel_error)

    def _failure_record(self, cand: Candidate, rep: EvalReport, phase_seed: int | None, speed_relation: str, kernel_error: bool) -> dict:
        """@brief 为一次失败的重测整理一份记录。

        重测命令是给人跑的，本方法不运行它（``human_permission_required`` 恒为 True）。记录里不含 ``counterexample``，
        两份报告按 ``_report_without_counterexample`` 清洗，``check_recheck_record`` 要求的键都会写齐。

        @param cand 被判失败的候选。
        @param rep 这次重测的报告。
        @param phase_seed 使用的派生种子，可为 None。
        @param speed_relation 与当前种子的关系（``new_best``/``within_noise``/``confirm``）。
        @param kernel_error 是否算作内核错误（``timeout``/``crash`` 为 False）。
        @return 复查记录字典。
        """
        search_seed = cand.report.seed if cand.report is not None else None
        recheck_seed = phase_seed if phase_seed is not None else rep.seed
        case = None
        for item in ((rep.correctness or {}).get("cases") or []):
            if isinstance(item, dict) and item.get("ok") is False:
                case = item.get("case")
                break
        tolerance = (rep.correctness or {}).get("tolerance") or {"rtol": self.task.rtol, "atol_rel": self.task.atol_rel, "exact": self.task.exact}
        kernel_path = f"candidates/{cand.idx:03d}_{cand.label}/{self.kernel_name}"
        command = f"uv run python -m tcenv.cli eval {self.task_name} {kernel_path} --dtype {self.dtype} --size {self.size} --quick --seed {recheck_seed}"
        kind = rep.status if rep.status in ("incorrect", "unsafe", "timeout", "crash") else "other"
        return {
            "class": kind,
            "task": self.task_name,
            "dtype": self.dtype,
            "size": self.size,
            "label": cand.label,
            "source_sha256": hashlib.sha256(cand.source.encode("utf-8")).hexdigest(),
            "search_seed": search_seed,
            "recheck_seed": recheck_seed,
            "case": case,
            "tolerance": tolerance,
            "speed_relation": speed_relation,
            "counts_as_kernel_error": kernel_error,
            "search_report": _report_without_counterexample(cand.report),
            "recheck_report": _report_without_counterexample(rep),
            "retest_command": command,
            "human_permission_required": True,
        }

    def _store_recheck(self, cand: Candidate, doc: dict) -> None:
        """@brief 把复查记录写进候选目录，并另存一份到 ``rechecks/<label>.json``。

        @param cand 被复查的候选。
        @param doc ``_failure_record`` 产出的记录。
        """
        text = json.dumps(doc, indent=1, default=str)
        control.write_text_atomic(self.run_dir / "candidates" / f"{cand.idx:03d}_{cand.label}" / "differential_recheck.json", text)
        folder = self.run_dir / "rechecks"
        folder.mkdir(exist_ok=True)
        control.write_text_atomic(folder / f"{cand.label}.json", text)


    # ------------------------------------------------------------------ main loop
    # 整次运行的外壳。预算耗尽、GPU 挂死或模型错误在这里收成 stop_reason。挂死时 hangs 已达上限，_confirm 会跳过；预算耗尽仍会复测已有种子，因为复测不再发模型请求。
    async def run(self) -> RunResult:
        """@brief 整次运行的外壳：搜索、control 文件续轮、确认、可选 profiling、收尾。

        预算耗尽、GPU 挂死或模型错误在这里收成 ``stop_reason``。挂死时 ``hangs`` 已达上限，``_confirm`` 会跳过；
        预算耗尽仍会复测已有种子，因为复测不再发模型请求。搜索停下后 ``_control_extends`` 会再读一次 control 文件，
        允许时先补跑轮次再进确认。确认、profiling 和关闭 client 都在收尾面上：任何一处抛异常，仍然必须写出
        ``result.json`` / ``best.metal`` / ``ledger.json``，而不是让一场已付费的搜索只剩 ``state.json``。

        @return 收尾记录 ``RunResult``。
        @raises RunStartError 当写核方式不是 hmz、``branches``/``init_samples`` 不是 1，或 hmz 预检失败时。
        """
        res = RunResult(run_dir=str(self.run_dir), task=self.task_name, dtype=self.dtype, size=self.size, model=self.cfg.model, budget_usd=self.ledger.budget_usd)
        resumed = self._resume is not None
        self._guard_shape()
        if self.agent_runner is None and not self._client_injected:
            from .kernel_agent import AgentError, preflight

            cred = Path(os.path.expanduser("~/.pi/agent/models.json"))
            try:
                preflight(self.committees, cred)
            except AgentError as exc:
                raise RunStartError(str(exc)) from exc
            self._ensure_gpu_lock_file()
        self.log(f"== {'RESUME ' if resumed else ''}{self.task_name} {self.dtype} {self.size} | model {self.cfg.model} | budget ${self.ledger.budget_usd:.2f} | one session a round | run {self.run_dir.name}")
        self.log("   committees: " + " | ".join(f"{c.name} {c.members[0].provider}/{c.members[0].model}" for c in self.committees))
        if resumed:
            self.log(f"   continuing from round {self._resume.partial.rnd if self._resume.partial else self._resume.next_round}, spent so far ${self.ledger.total_usd:.3f}, best {self.best.label if self.best else '-'}")
        try:
            await (self._search_resumed(res) if resumed else self._search(res))
        except BudgetExhausted as e:
            # CostLedger.check 在下一次请求前拒绝了调用。已在飞的请求可能已经略超 budget_usd。
            res.stop_reason = "budget_exhausted"
            res.session_budget_stop = str(e).startswith("session cost")
            self.log(f"== stop: {e}")
        except StopRun as e:
            # 一次完整评测超时（或连续崩溃）触发的熔断：不再开新回合。
            res.stop_reason = str(e)
            self.log(f"== STOP: {e} (the run ends immediately; a runaway GPU kernel can freeze the display)")
        except LLMError as e:
            res.stop_reason = f"llm_error: {e}"
            self.log(f"== stop: {e}")
        self._search_seed = self.best  # the kernel the search would mutate next; confirmation may publish a different one
        while self._control_extends(res):
            self.log("-- control file extended this run; continuing before confirmation")
            try:
                await self._rounds(res, (self._round or 0) + 1, self._stagnant, None)
            except BudgetExhausted as e:
                res.stop_reason = "budget_exhausted"
                # 和上面外层同一个标志：session cost floor 的停止不是「轮数用完」，
                # 缺了它 _control_extends 会把它当成还能继续，再跑一轮并烧掉一个轮号。
                res.session_budget_stop = str(e).startswith("session cost")
                self.log(f"== stop: {e}")
            except StopRun as e:
                res.stop_reason = str(e)
                self.log(f"== STOP: {e} (the run ends immediately; a runaway GPU kernel can freeze the display)")
            except LLMError as e:
                res.stop_reason = f"llm_error: {e}"
                self.log(f"== stop: {e}")
            self._search_seed = self.best
        # 搜索已经把该花的钱花完了。确认、profiling 和关闭 client 都在收尾面：任何一处抛异常，
        # 都必须仍然写出 result.json / best.metal / ledger.json，而不是让一场已付费的搜索只剩 state.json。
        try:
            self.ledger.refresh_budget()
            await self._confirm(res)
        except (StopRun, BudgetExhausted) as e:
            res.stop_reason = res.stop_reason or str(e)
            self.log(f"== stop: {e}")
        except LLMError as e:
            res.stop_reason = res.stop_reason or f"llm_error: {e}"
            self.log(f"== stop: {e}")
        except Exception as e:  # noqa: BLE001 - the run is over; its files still have to be written
            res.stop_reason = res.stop_reason or f"confirm_error: {type(e).__name__}: {e}"
            self.log(f"== stop: confirmation failed: {type(e).__name__}: {e}")
        if self.search.profile_best:
            try:
                await self._profile_best()
            except Exception as e:  # noqa: BLE001 - a missing trace is not a reason to lose the result
                self.log(f"== warning: profile_best failed: {type(e).__name__}: {e}")
        try:
            await self._close_clients()  # every HTTP client this run opened, on the loop that used it (asyncio.run closes the loop afterwards)
        except Exception as e:  # noqa: BLE001 - the result is already on disk; a close error must not hide it
            self.log(f"== warning: closing the HTTP clients failed: {type(e).__name__}: {e}")
        return self._finish(res)

    def _control_extends(self, res: RunResult) -> bool:
        """@brief 搜索停下后 control 文件带来的改动是否允许再跑一轮。

        确认曾经是最后一段 GPU 工作，所以在它期间写下的加预算永远不会被读到，一次已经结束的非预算停止也无法续上。
        文件就在这里读，在确认之前。``session_budget_stop`` 只在余额还够一次最近规模的会话时才继续；
        熔断类原因（挂死、连续崩溃、agent 失败、LLM 错误、成本不完整、会话成本下限）一律不再继续。

        @param res 收尾记录（读它的 ``stop_reason`` 与 ``session_budget_stop``）。
        @return 可以再跑一轮为 True。
        """
        self.ledger.refresh_budget()
        if res.session_budget_stop:
            remaining = self.ledger.remaining
            recent = session_budget_floor(self._session_costs)
            return remaining is not None and recent is not None and remaining >= recent
        reason = res.stop_reason or ""
        if reason.startswith(("gpu_hang", "repeated_worker", "agent_failure", "llm_error", "cost_incomplete", "session cost")):
            return False
        if reason == "budget_exhausted":
            remaining = self.ledger.remaining
            return remaining is not None and remaining >= 0.05
        if reason == "max_rounds" or reason.startswith("converged"):
            if patience_exhausted(self._stagnant, self.search.patience):
                return False
            return self._round < self.search.max_rounds
        return False

    @property
    def kernel_name(self) -> str:
        """@brief 运行目录里内核文件的文件名。

        由目标的 ``file_ext`` 决定（缺省 ``.metal``），名字一律以 ``kernel`` 开头。

        @return 例如 ``kernel.metal``。
        """
        ext = self.target.file_ext or ".metal"
        return f"kernel{ext if str(ext).startswith('.') else '.' + str(ext)}"

    @property
    def best_name(self) -> str:
        """@brief 运行目录里优胜者文件的文件名。

        把 ``kernel_name`` 的 ``kernel`` 前缀换成 ``best``。

        @return 例如 ``best.metal``。
        """
        return "best" + self.kernel_name[len("kernel"):]

    # --from 只在 best.metal 存在、且 result.json 里的题和这次一致时才把源码交回来。对不上或没有文件就返回 None，调用方从头写。
    def _prior_kernel_text(self) -> str | None:
        """@brief ``--from`` 指向上一次运行时的 ``best.metal`` 正文。

        只在 ``best.metal`` 存在、且 ``result.json`` 里的题（task/dtype/size）与这次一致时才把源码交回来；
        对不上或没有文件就返回 None，调用方从头写。``from_catalog`` 时改从 kernel catalog 取。
        文件在但读不了、或版本比这份代码新时抛 ``RunStartError``，不能当成「没有成绩」再从头写一版。
        调用方会重新评测这份源码。

        @return 上一版内核正文；拿不到时为 None。
        @raises RunStartError 当 ``result.json`` 需要人工检查（``NeedsInspection``）时。
        """
        if self.from_catalog:
            from .kernel_catalog import catalog_dir

            folder = catalog_dir(self.target_id, self.task_name, self.dtype, self.size)
            ext = self.target.file_ext or ".metal"
            found = sorted(folder.glob("best.*"))
            metal = folder / self.best_name
            if not metal.is_file() and found:
                metal = found[0]
            if not metal.is_file():
                self.log(f"-- --from @catalog: no kernel for {self.target_id} {self.task_name} {self.dtype} {self.size}")
                return None
            text = metal.read_text(encoding="utf-8", errors="replace")
            if not _KERNEL_DECL.search(text):
                self.log(f"-- --from @catalog: {metal.name} is empty or has no kernel declaration; writing the first version from scratch")
                return None
            return text
        root = self.from_run
        if root is None:
            return None
        if not self._same_target(root):
            self.log(f"-- --from {root} is not target {self.target_id}; writing the first version from scratch")
            return None
        metal = root / self.best_name
        if not metal.is_file():
            self.log(f"-- --from {root}: no {self.best_name}; writing the first version from scratch")
            return None
        recorded = root / "result.json"
        if recorded.is_file():
            # 文件在但读不了，或版本比这份代码新：不能当成「没有成绩」再从头写一版。
            from .artifacts import RESULT_SCHEMA_VERSION, NeedsInspection, read_versioned

            try:
                doc = read_versioned(recorded, RESULT_SCHEMA_VERSION)
            except NeedsInspection as e:
                raise RunStartError(f"--from {root}: {e}; it is not treated as a run that was never done") from e
            mismatch = [f"{key} is {doc.get(key)!r}, this run is {got!r}" for key, got in (("task", self.task_name), ("dtype", self.dtype), ("size", self.size)) if doc.get(key) not in (None, got)]
            if mismatch:
                self.log(f"-- --from {root}: not this task ({'; '.join(mismatch)}); writing the first version from scratch")
                return None
        else:
            self.log(f"-- --from {root}: no result.json; re-evaluating {self.best_name} on {self.task_name} {self.dtype} {self.size}")
        text = metal.read_text(encoding="utf-8", errors="replace")
        if not _KERNEL_DECL.search(text):
            # A kill between truncate and write leaves an empty/partial winner. Re-evaluating it
            # would spend a GPU evaluation on text that cannot compile.
            self.log(f"-- --from {root}: {metal.name} is empty or has no kernel declaration; writing the first version from scratch")
            return None
        return text

    # 初始生成，然后进入逐轮「诊断 → 规划 → K 路补丁」。只有 _eval 认定安全、算对且快过噪声的候选才会变成下一轮种子。
    async def _search(self, res: RunResult) -> None:
        """@brief 全新搜索：先初始生成，再进入逐轮「诊断 → 规划 → 补丁」。

        只有 ``_eval`` 认定安全、算对且快过噪声的候选才会变成下一轮种子。``--from`` 给了上一版内核且它仍然合格时，
        先评测它并跳过初始生成。初始生成最多重开 ``INITIAL_RESTARTS`` 次；一次都没有合格种子就把 ``stop_reason``
        写成 ``no_valid_initial_kernel`` 并返回，后面的轮次不会开始。

        @param res 收尾记录（写 ``stop_reason`` 与 ``rounds``）。
        """
        # 1. initial implementation, unless the user pointed at a previous kernel that is still eligible
        self._round = 0
        self._publish_state("llm")
        if self.from_run is not None:
            source = self._prior_kernel_text()
            if source is not None:
                self.log(f"-- re-evaluating {self.from_run / self.best_name} on {self.task_name} {self.dtype} {self.size}")
                await self._eval(source, "prior")
                if self.best:
                    self.log(f"-- seed is that kernel at {self.best.speedup:.3f}x (measured now); skipping the first draft")
                else:
                    self.log(f"-- {self.from_run / self.best_name} is no longer eligible; writing the first version from scratch")
        # 初始生成最多重开 INITIAL_RESTARTS 次。一次都没有合格种子就停，原因是 no_valid_initial_kernel，后面的轮次不会开始。
        if not self.best:
            await self._initial_generation()
        if not self.best:
            res.stop_reason = "no_valid_initial_kernel"
            return
        # 2..5. rounds
        await self._rounds(res, 1, 0)

    def _copied_report(self, source: str) -> EvalReport | None:
        """@brief 贴回来的内核给出机械拒绝，需要实测时返回 None。

        三条逐字比较：当前种子（``unchanged``）、本场更早的候选（``duplicate``，但那条候选若超时/崩溃则不算）、
        别的运行里的核（``external_duplicate``）。命中任意一条都不上 GPU。``--from`` 带来的那份源码例外，照常实测。

        @param source 待评的内核正文。
        @return 拒绝用的预置报告；需要实测时为 None。
        """
        if self.best is not None and source == self.best.source:
            return EvalReport(
                task=self.task_name, dtype=self.dtype, size=self.size, mode="full",
                status="unchanged", stage="screen",
                message=f"identical to the current seed {self.best.label}. That kernel stays the seed. This branch does not count.",
            )
        for cand in self.candidates:
            # 超时和崩溃不是这颗核的判决。同一份源码还要再评一次。
            if cand.source == source and (cand.report is None or cand.report.status not in ("timeout", "crash")):
                return EvalReport(
                    task=self.task_name, dtype=self.dtype, size=self.size, mode="full",
                    status="duplicate", stage="screen",
                    message=f"identical to earlier candidate {cand.label}. It does not count as an improvement.",
                )
        # Load the external set first: it also records the --from kernel in ``_from_text``.
        # Checking ``_from_text`` before this call made the first --from evaluation look like an
        # external duplicate (its copy under candidates/ is in the set).
        external = self._external_kernel_texts()
        if self._from_text and source == self._from_text:
            return None
        if source in external:
            return EvalReport(
                task=self.task_name, dtype=self.dtype, size=self.size, mode="full",
                status="external_duplicate", stage="screen",
                message="identical to a kernel from another run. It does not count as a result of this search.",
            )
        return None

    def _external_kernel_texts(self) -> set[str]:
        """@brief 别的运行里的内核正文集合，只读一次。

        空的根目录列表（测试）什么都读不到。跳过本运行自己，也跳过不同目标的运行；没显式指定根目录时，
        目录里的 kernel catalog 也会被扫进来。顺手把 ``--from`` 那份源码记到 ``_from_text``。

        @return 内核正文集合（懒加载后缓存）。
        """
        if self._external_texts is not None:
            return self._external_texts
        roots = self._external_roots
        if roots is None:
            runs = REPO_ROOT / "runs"
            roots = [runs] if runs.is_dir() else []
        texts: set[str] = set()
        from_path = (self.from_run / self.best_name) if self.from_run is not None else None
        if from_path is not None and from_path.is_file():
            self._from_text = from_path.read_text()
        for root in roots:
            if not root.is_dir():
                continue
            for path in list(root.glob(f"*/candidates/*/{self.kernel_name}")) + list(root.glob(f"*/*/candidates/*/{self.kernel_name}")):
                if self.run_dir in path.parents or not self._same_target(path.parents[2]):
                    continue
                try:
                    texts.add(path.read_text())
                except OSError:
                    continue
            for path in list(root.glob(f"*/{self.best_name}")) + list(root.glob(f"*/*/{self.best_name}")):
                if self.run_dir in path.parents or not self._same_target(path.parent):
                    continue
                if from_path is not None and path.resolve() == from_path.resolve():
                    continue
                try:
                    texts.add(path.read_text())
                except OSError:
                    continue
        from .kernel_catalog import catalog_root

        catalog = catalog_root() / self.target_id
        if self._external_roots is None and catalog.is_dir():
            for path in catalog.glob("*/*/*/best.*"):
                try:
                    texts.add(path.read_text(encoding="utf-8"))
                except OSError:
                    continue
        self._external_texts = texts
        return texts

    def _same_target(self, run_dir: Path) -> bool:
        """@brief 一个既有运行目录是不是同一个目标设备的。

        ``run_config.json`` 里记录过 ``target.id`` 就按它比；老目录没有这一段时，退回到「默认就是 apple-m4-max」
        或者「目录名的上一级就是目标 id」。文件读不了时返回 False。

        @param run_dir 既有运行目录。
        @return 同一目标为 True。
        """
        cfg = run_dir / "run_config.json"
        if cfg.is_file():
            try:
                doc = json.loads(cfg.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            target = doc.get("target") if isinstance(doc, dict) else None
            recorded = target.get("id") if isinstance(target, dict) else None
            if recorded:
                return recorded == self.target_id
            return self.target_id == "apple-m4-max"
        return self.target_id == "apple-m4-max" or run_dir.parent.name == self.target_id

    def _guard_shape(self) -> None:
        """@brief 在任何会话开始之前拒绝旧的并行形状。

        这条搜索是一轮一场 hmz 会话：``kernel_agent`` 必须是 ``pi``，``branches`` 与 ``init_samples`` 必须是 1。

        @raises RunStartError 当写核方式或并行形状不符时。
        """
        if self.search.kernel_agent != "pi":
            raise RunStartError("kernel writing goes through hmz; kernel_agent must be pi")
        if self.search.branches != 1 or self.search.init_samples != 1:
            raise RunStartError("this search is one session a round: branches and init_samples must be 1")

    def _ensure_gpu_lock_file(self) -> None:
        """@brief 建好整机的 GPU 锁文件，让只读的会话也能 flock 它。

        会话不会被授予写权限。
        """
        from tcenv.gpulock import lock_file

        path = lock_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    async def _one_session(self, tag: str, *, seedless: bool) -> str:
        """@brief 跑一场 hmz 会话，返回 ``agent_failed``、``unchanged`` 或 ``evaluated``。

        崩溃或超时之后没有交出内核的会话算 ``agent_failed``。干净退出但没写 submission 文件也算 ``agent_failed``。
        干净退出但把种子的字节写回该文件算 ``unchanged``，不计一次失败。文件变了就评测，不管进程怎么退出。
        反馈只解析一次、只记录一次，会话提前停下时同样记录。

        开跑前先过预算闸门：余额不足 $0.05、或余额低于最近一场会话的成本（session cost floor）时抛 ``BudgetExhausted``，
        不再开新会话。本场会话的模型请求发给本轮委员会的唯一成员，输出 token 上限由 ``relay_budget_usd`` 折算。
        自检超时计进 ``hangs``，达 ``max_hangs`` 时抛 ``StopRun`` 熔断；会话结束却没有 token 计数时也抛 ``StopRun``，
        因为账本无法继续。

        @param tag 会话标签（同时是 ``agent_work/`` 下的目录名，``/`` 会换成 ``_``）。
        @param seedless 是否为初始生成（不把当前种子交给会话）。
        @return ``agent_failed`` / ``unchanged`` / ``evaluated``。
        @raises BudgetExhausted 余额不足以再开一场会话时。
        @raises StopRun 自检超时把 hangs 推到上限、或会话没有 token 计数时。
        @raises RunStartError 脚本运行时没有注入 ``agent_runner``（不得起 hmz）时。
        """
        from .agent_feedback import parse_feedback
        from .kernel_agent import AgentError, Prices, run_kernel_agent, output_tokens_for

        self.ledger.check()
        remaining = self.ledger.remaining
        if remaining is not None and remaining < 0.05:
            raise BudgetExhausted(f"budget ${self.ledger.budget_usd:.2f} exhausted (spent ${self.ledger.total_usd:.4f})")
        recent_session = session_budget_floor(self._session_costs)
        if remaining is not None and recent_session is not None and remaining < recent_session:
            raise BudgetExhausted(
                f"session cost floor: ${remaining:.4f} remains, below the recent "
                f"session cost ${recent_session:.4f}; not starting another session"
            )
        member = next(c.members[0] for c in self.committees if c.name == self._committee_name)
        slice_usd = self.search.relay_budget_usd if remaining is None else min(remaining, self.search.relay_budget_usd)
        prices = Prices(member.cfg.price_in, member.cfg.price_out, member.cfg.price_cache_read, member.cfg.price_cache_write)
        tokens = output_tokens_for(slice_usd, prices.output)
        work = (self.run_dir / "agent_work" / tag.replace("/", "_")).resolve()
        brief = self._session_brief(seedless, tag, work)
        self._session_briefs.append(brief)
        self._publish_state("llm", inflight=tag)
        self._prepare_session_dir(work)
        started = time.time()
        seed_label = None if self.best is None else self.best.label
        parsed = parse_feedback("")
        outcome = None
        cand = None
        exit_kind = "agent_exception"
        try:
            try:
                if self.agent_runner is None and self._client_injected:
                    raise RunStartError("a scripted run needs an agent_runner; it must not start hmz")
                if self.agent_runner is not None:
                    result = self.agent_runner(brief, tag)
                    if asyncio.iscoroutine(result):
                        result = await result
                else:
                    result = await asyncio.to_thread(
                        run_kernel_agent, brief, work=work, provider=member.provider, model=member.model,
                        effort=member.cfg.reasoning_effort, prices=prices, turn_output_tokens=tokens,
                        turn_minutes=int(self.search.turn_minutes),
                    )
            except (AgentError, LLMError, Exception) as exc:
                if isinstance(exc, (StopRun, BudgetExhausted, RunStartError)):
                    raise
                parsed = self._parsed_feedback(work, None, started)
                self.log(f"  [{tag}] hmz failed: {exc}")
                event = {
                    "event": "agent", "label": tag, "harness": "pi", "provider": member.provider, "model": member.model,
                    "committee": self._committee_name, "cost_usd": None, "cost_incomplete": True, "had_kernel": False,
                    "outcome": "agent_failed", "error": str(exc)[:180], "self_checks": 0, "self_check_timeouts": 0,
                }
                event.update(self._feedback_slim(parsed))
                self.llm.transcript.log(event)
                outcome = None
                exit_kind = "agent_exception"
                return "agent_failed"
            parsed = self._parsed_feedback(work, result, started)
            if result.cost_usd is None:
                self.llm.transcript.log(self._agent_event(tag, member, result, "cost_incomplete", parsed))
                outcome = None
                exit_kind = "cost_incomplete"
                raise StopRun("cost_incomplete: the session finished without a token count, so the ledger cannot continue")
            self.ledger.charge(result.cost_usd, tag, tokens=result.spent, basis=prices.as_dict())
            self._session_costs.append(float(result.cost_usd))
            if result.cost_usd > self.search.relay_budget_usd + 1e-6:
                self.log(
                    f"  [{tag}] session spent ${result.cost_usd:.2f}, above the ${self.search.relay_budget_usd:.2f} output-token guide. "
                    "The run budget still stops the next session."
                )
            self.hangs += int(result.self_check_timeouts or 0)
            if result.status == "timeout" and result.source:
                # 超时后交回来的核不再送上 GPU。没有内核的超时仍只算代理失败。
                self.hangs += 1
            seed_source = None if self.best is None else self.best.source
            if result.status == "timeout" or result.source is None:
                outcome = "agent_failed"
            elif result.status == "failed" and result.source == seed_source:
                outcome = "agent_failed"
            elif seed_source is not None and result.source == seed_source:
                outcome = "unchanged"
            else:
                outcome = "evaluated"
            self.llm.transcript.log(self._agent_event(tag, member, result, outcome, parsed))
            missing = ""
            if result.source is None:
                from .kernel_agent import submission_name

                missing = " " + submission_name(self.task_name, self.dtype, self.size, tag) + " was not written"
            self.log(f"  [{tag}] {self._committee_name} {member.provider}/{member.model} {outcome}{missing}" + (f" ({result.error[:120]})" if result.error else ""))
            if (result.self_check_timeouts or (result.status == "timeout" and result.source)) and self.hangs >= self.search.max_hangs:
                self._publish_state("llm", inflight=None)
                outcome = None
                exit_kind = "hang_circuit_breaker"
                raise StopRun("gpu_hang_circuit_breaker")
            try:
                if outcome == "evaluated":
                    cand = await self._eval(result.source, tag)
                    self._file_session_notes(cand, result)
                    outer = self._outer_line(cand)
                    self._notes_summary = _notes_for_next(outer, result.notes)
                    if cand.report is not None and not cand.eligible and cand.report.status not in ("unchanged", "duplicate", "external_duplicate"):
                        fb = format_feedback(cand.report)[:2000]
                        note = threadgroup_limit_note(cand.report.message)
                        if note:
                            fb += "\n\n" + note
                        self._failure_feedback = fb
                    elif cand.eligible or cand.is_best:
                        # 这一版已经算对。上一场失败反馈不再指向已经不是种子的核。
                        self._failure_feedback = ""
                elif outcome == "unchanged" and result.source:
                    cand = await self._eval(result.source, tag)
                    self._file_session_notes(cand, result)
                    self._notes_summary = _notes_for_next(self._outer_line(cand), result.notes)
                    self._failure_feedback = ""
                elif outcome == "unchanged":
                    self._notes_summary = _notes_for_next("种子没换", result.notes)
                    self._failure_feedback = ""
            except Exception as exc:
                if isinstance(exc, StopRun) and "gpu_hang_circuit_breaker" in str(exc):
                    if outcome == "evaluated":
                        outcome = None
                    cand = None
                    exit_kind = "hang_circuit_breaker"
                elif isinstance(exc, (StopRun, BudgetExhausted, RunStartError)):
                    exit_kind = "normal"
                    if cand is None:
                        cand = self._by_label(tag)
                else:
                    outcome = None
                    cand = None
                    exit_kind = "eval_error"
                raise
            exit_kind = "normal"
            return outcome
        finally:
            try:
                self._record_feedback(tag, member, work, parsed, outcome, cand, exit_kind, seed_label, started=started)
            except OSError:
                pass

    def _prepare_session_dir(self, work: Path) -> None:
        """@brief 会话开始前清掉上一场留下的 notes 与反馈。

        续跑时这个目录是复用的。顺带删掉受保护的自检结果并重建 ``.selfcheck`` 目录。

        @param work 本场会话的工作目录。
        """
        work.mkdir(parents=True, exist_ok=True)
        for name in ("NOTES.md", "FEEDBACK.md", "self_checks.jsonl"):
            try:
                (work / name).unlink(missing_ok=True)
            except OSError:
                pass
        for protected in (
            work / ".selfcheck" / "self_checks.jsonl",
            work.parent.parent / "session_checks" / work.name / "self_checks.jsonl",
        ):
            try:
                protected.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            (work / ".selfcheck").mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _parsed_feedback(self, work: Path, result, started: float) -> dict:
        """@brief 解析本场会话交上来的反馈。

        agent 结果里带了反馈文本就直接用它（这时「文件超长」按文件大小判断）；没有文本时才读工作目录里
        本场新写（mtime 不早于 ``started - 1.0``）的 ``FEEDBACK.md``。之后交给 ``agent_feedback.parse_feedback``
        统一解析，仓库根用于定位被引用的文件。

        @param work 本场会话的工作目录。
        @param result 会话结果，可为 None（hmz 抛异常时）。
        @param started 本场会话的开始时间（秒）。
        @return ``parse_feedback`` 的解析结果。
        """
        from .agent_feedback import FILE_CAP, parse_feedback, read_feedback_text

        text = ""
        if result is not None:
            text = getattr(result, "feedback", "") or ""
        path = work / "FEEDBACK.md"
        file_over = False
        if str(text).strip():
            try:
                file_over = path.is_file() and path.stat().st_size > FILE_CAP
            except OSError:
                file_over = False
        else:
            try:
                fresh = path.is_file() and path.stat().st_mtime >= started - 1.0
            except OSError:
                fresh = False
            if fresh:
                text, file_over = read_feedback_text(path)
        return parse_feedback(text, work=work, repo=REPO_ROOT, file_over=file_over)

    def _feedback_slim(self, parsed: dict | None) -> dict:
        """@brief 反馈的简短摘要，附进会话记录事件。

        没有任何占位条目时保持旧会话记录的字节完全一致：``feedback_placeholder_items`` 只在非零时才出现。

        @param parsed 解析结果，可为 None。
        @return 含 ``feedback_kinds``、``feedback_sections``、``feedback_chars`` 的字典，有占位条目时多个键。
        """
        parsed = parsed or {}
        slim = {
            "feedback_kinds": parsed.get("kinds") or "",
            "feedback_sections": list(parsed.get("section_names") or []),
            "feedback_chars": int(parsed.get("chars") or 0),
        }
        # Keep the old transcript byte-identical when there is nothing empty to report.
        placeholders = int(parsed.get("placeholder_items") or 0)
        if placeholders:
            slim["feedback_placeholder_items"] = placeholders
        return slim

    def _next_feedback_seq(self) -> int:
        """@brief 下一条反馈记录的序号。

        第一次调用时从 ``feedback.jsonl`` 里现有最大序号起步，之后在进程内自增。

        @return 递增后的序号。
        """
        from .agent_feedback import max_seq

        if self._feedback_seq is None:
            self._feedback_seq = max_seq(self.run_dir / "feedback.jsonl")
        self._feedback_seq += 1
        return self._feedback_seq

    def _record_feedback(self, tag: str, member, work: Path, parsed: dict, outcome: str | None, cand, exit_kind: str, seed_label: str | None, *, started: float | None = None) -> None:
        """@brief 把一场会话的反馈追加进 ``feedback.jsonl``，并按需处理结构化提案。

        记录里有 run id、序号、会话开始时间（唯一能证明探针是在本场创建的时戳，``t`` 是会话结束时写的）、
        委员会/模型、outcome、退出类型、候选状态与微秒数、种子标签与加速比，以及解析出的分节与条目。
        结构化提案属于观察，不是钱也不是 GPU 工作：即使 ``harness_auto`` 关着也会记下来供人复查；
        打开时才 ``settle``，``harness_review`` 打开时再走 ``_review_new_tips``。harness 记录失败不能停下搜索。

        @param tag 会话标签。
        @param member 本轮的委员会成员。
        @param work 本场会话的工作目录。
        @param parsed ``_parsed_feedback`` 的解析结果。
        @param outcome ``evaluated`` / ``unchanged`` / ``agent_failed``；提前熔断时为 None。
        @param cand 本场评出的候选，可为 None。
        @param exit_kind 退出类型（``normal``/``agent_exception``/``eval_error``/``hang_circuit_breaker``/``cost_incomplete``）。
        @param seed_label 开跑时的种子标签。
        @param started 会话开始时间（秒）；None 时写 null。
        """
        from .agent_feedback import round_of

        raw = str((parsed or {}).get("raw") or "")
        items = (parsed or {}).get("items") or []
        report = None if cand is None else cand.report
        try:
            run_id = self.run_dir.resolve().relative_to((REPO_ROOT / "runs").resolve()).as_posix()
        except ValueError:
            run_id = self.run_dir.name
        record = {
            "format": "agent_feedback",
            "v": 1,
            "seq": self._next_feedback_seq(),
            "run_id": run_id,
            "t": time.time(),
            # The session start is the only timestamp that can prove a probe was created
            # during this session; ``t`` is written when the session ends.
            "session_started": None if started is None else float(started),
            "task": self.task_name,
            "dtype": self.dtype,
            "size": self.size,
            "target_id": self.target_id,
            "backend": self.target.backend,
            "label": tag,
            "round": round_of(tag),
            "committee": self._committee_name,
            "provider": member.provider,
            "model": member.model,
            "effort": member.cfg.reasoning_effort,
            "outcome": outcome,
            "exit": exit_kind,
            "candidate_status": None if report is None else report.status,
            "candidate_us": None if report is None or not isinstance(report.candidate_us, (int, float)) else report.candidate_us,
            "seed_label": seed_label,
            "best_speedup": None if self.best is None else self.best.speedup,
            "kinds": parsed.get("kinds") or "none",
            "truncated": bool(parsed.get("truncated")),
            "truncated_sections": list(parsed.get("truncated_sections") or []),
            "work_dir": f"agent_work/{tag.replace('/', '_')}",
            "sections": parsed.get("sections") or {},
            "items": items,
            "unverified_premises": int(parsed.get("unverified_premises") or 0),
            "placeholder_items": int(parsed.get("placeholder_items") or 0),
            "raw": parsed.get("raw") or "",
        }
        path = self.run_dir / "feedback.jsonl"
        control.append_jsonl(path, record)
        # Structured proposals are observations, not money or GPU work. They are recorded even
        # when harness_auto is off so a person can review them; settle only runs when enabled.
        try:
            from . import harness
            from tcenv.schema import PROTOCOL_VERSION

            kernel_sha = "" if cand is None else hashlib.sha256(cand.source.encode("utf-8")).hexdigest()
            trusted_report = None
            if cand is not None and cand.report is not None and cand.report.status == "ok":
                trusted_report = {
                    "status": "ok",
                    "candidate_us": cand.report.candidate_us,
                    "speedup": cand.report.speedup,
                    "target_id": cand.report.target_id,
                    "protocol_version": cand.report.protocol_version,
                    "kernel_sha256": kernel_sha,
                    "trusted_source": "outer candidate report",
                }
            proposals = harness.record_proposals(
                self.run_dir, record=record, parsed=parsed,
                target_id=self.target_id, task=self.task_name, dtype=self.dtype,
                work=work, started=started if started is not None else time.time() - 1.0,
                kernel_sha256=kernel_sha, protocol_version=PROTOCOL_VERSION,
                trusted_report=trusted_report,
            )
            if proposals and self.search.harness_auto:
                harness.settle(
                    self.run_dir, target_id=self.target_id, task=self.task_name,
                    protocol_version=PROTOCOL_VERSION, enabled=True,
                    shadow=bool(self.search.harness_shadow),
                )
                if self.search.harness_review:
                    self._review_new_tips(proposals, PROTOCOL_VERSION)
                harness.sync_trust_status(self.run_dir, target_id=self.target_id, task=self.task_name)
        except Exception as exc:  # noqa: BLE001 - a harness record must not stop a search
            self.log(f"  [{tag}] harness proposals skipped: {exc}")
        del work

    def _review_new_tips(self, proposals: list, protocol_version: str) -> None:
        """@brief 在下一场会话之前，用另一个委员会评审一条单源 tip。

        评审是本运行范围内的，不做仓库级晋升。评审费用记在同一本账上；评审者的预算由它的价格推导，不是固定额度。
        被转正的 claim 不能由写下其源文本的模型来评审：先挑不是该 claim 所属委员会的另一个，再退回不是当前委员会的另一个。
        没有其它委员会、评审者与原作者冲突、或余额低于评审额度时直接跳过。失败或超时的评审者照样花了钱，
        ``run_review`` 抛出的异常上挂着已知花费，这里会把它记进账本。评审失败不能停下搜索。

        @param proposals ``harness.record_proposals`` 产出的提案列表。
        @param protocol_version 当前协议版本。
        """
        from . import harness
        from . import review as review_mod
        from .kernel_agent import Prices

        pending = harness.pending_tip_reviews(self.run_dir, proposals)
        if not pending:
            return
        row = pending[0]
        # A converted claim must not be reviewed by the model that wrote its source text.
        # Choose a committee other than the source claim's committee first.
        proposal_committee = harness.proposal_committee(row)
        other = next((committee for committee in self.committees if committee.name != proposal_committee), None)
        if other is None or not other.members:
            other = next((committee for committee in self.committees if committee.name != self._committee_name), None)
        if other is None or not other.members:
            return
        reviewer = other.members[0]
        if harness.reviewer_conflict(row, reviewer.provider, reviewer.model):
            self.log("  [review] skipped: the reviewer is also an author of the source claim")
            return
        bundle = review_mod.evidence_bundle(self.run_dir, row)
        prompt = review_mod.build_review_prompt(row.get("proposal") or {}, run_id=str(row.get("run_id") or self.run_dir.name), bundle=bundle)
        prices = Prices(reviewer.cfg.price_in, reviewer.cfg.price_out, reviewer.cfg.price_cache_read, reviewer.cfg.price_cache_write)
        plan = review_mod.estimate_review_budget(prices, prompt)
        remaining = self.ledger.remaining
        if remaining is not None and remaining < plan["budget_usd"]:
            self.log(f"  [review] skipped: ${remaining:.4f} remains, review allowance is ${plan['budget_usd']:.4f}")
            return
        self.log(f"  [review] {reviewer.provider}/{reviewer.model} on {row.get('proposal_key')} (budget ${plan['budget_usd']:.4f}, expected ${plan['expected_usd']:.4f})")
        try:
            verdict = review_mod.run_review(
                self.run_dir / "harness_reviews" / str(row.get("proposal_key")),
                prompt=prompt, provider=reviewer.provider, model=reviewer.model,
                effort=reviewer.cfg.reasoning_effort, prices=prices,
            )
            if verdict.get("cost_usd") is None:
                # The kernel path stops on a session with no token bill. A review must not be
                # activated and recorded as paid when its cost is unknown; leave the tip unreviewed.
                self.log("  [review] skipped: the reviewer left no token bill; the tip stays unreviewed")
                return
            review_mod.append_review(self.run_dir, row, verdict)
            if verdict.get("cost_usd") is not None:
                self.ledger.charge(
                    float(verdict["cost_usd"]), f"review:{row.get('proposal_key')}",
                    tokens=verdict.get("tokens"), basis=prices.as_dict(),
                )
            harness.settle(
                self.run_dir, target_id=self.target_id, task=self.task_name,
                protocol_version=protocol_version, enabled=True,
                shadow=bool(self.search.harness_shadow),
            )
            self.log(f"  [review] useful={verdict.get('useful')} confidence={verdict.get('confidence')} (${verdict.get('cost_usd')})")
        except Exception as exc:  # noqa: BLE001 - a review failure must not stop the search
            # A failed or timed-out reviewer still spent money. ``run_review`` attaches what it
            # knows so the ledger does not lose it with the exception.
            bill = getattr(exc, "cost_usd", None)
            if isinstance(bill, (int, float)) and not isinstance(bill, bool) and bill > 0:
                self.ledger.charge(
                    float(bill), f"review:{row.get('proposal_key')}",
                    tokens=getattr(exc, "tokens", None), basis=prices.as_dict(),
                )
            self.log(f"  [review] skipped: {exc}")

    def _restore_handoff(self) -> None:
        """@brief 续跑时捡回上一场会话的 NOTES 与失败反馈。

        不这么做，续跑后的会话就看不到上一场写了什么。取 ``agent_work/*/NOTES.md`` 里 mtime 最新的那份；
        找不到对应候选时按「种子没换」处理。顺手把反馈序号重置，让下次重新从盘上取最大序号。
        """
        self._feedback_seq = None
        root = self.run_dir / "agent_work"
        chosen = None
        if root.is_dir():
            for path in root.glob("*/NOTES.md"):
                try:
                    stamp = path.stat().st_mtime
                except OSError:
                    continue
                if chosen is None or stamp >= chosen[0]:
                    chosen = (stamp, path)
        if chosen is None:
            return
        try:
            notes = chosen[1].read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        label = chosen[1].parent.name
        cand = self._by_label(label)
        if cand is None:
            cand = next((item for item in self.candidates if item.label.replace("/", "_") == label), None)
        if cand is None:
            self._notes_summary = _notes_for_next("种子没换", notes)
            return
        self._notes_summary = _notes_for_next(self._outer_line(cand), notes)
        if cand.report is not None and not cand.eligible and cand.report.status not in ("unchanged", "duplicate", "external_duplicate"):
            fb = format_feedback(cand.report)[:2000]
            note = threadgroup_limit_note(cand.report.message)
            if note:
                fb += "\n\n" + note
            self._failure_feedback = fb
        elif cand.is_best:
            self._failure_feedback = ""

    def _agent_event(self, tag: str, member, result, outcome: str, parsed: dict | None = None) -> dict:
        """@brief 组装一条 ``event: agent`` 的会话记录事件。

        含委员会/模型、花费及其是否不完整、是否有内核、outcome、错误前 180 字、自检次数与耗时、token 分项，
        并附 ``_feedback_slim`` 的反馈摘要。

        @param tag 会话标签。
        @param member 本轮委员会成员。
        @param result hmz 的会话结果。
        @param outcome ``evaluated`` / ``unchanged`` / ``agent_failed`` / ``cost_incomplete``。
        @param parsed 反馈解析结果，可为 None。
        @return 可直接写进会话记录的字典。
        """
        spent = result.spent or {}
        event = {
            "event": "agent", "label": tag, "harness": "pi", "provider": member.provider, "model": member.model,
            "committee": self._committee_name, "cost_usd": result.cost_usd, "cost_incomplete": bool(result.cost_incomplete),
            "had_kernel": bool(result.source), "outcome": outcome, "error": (result.error or "")[:180],
            "self_checks": result.self_checks, "self_check_timeouts": result.self_check_timeouts,
            "self_check_crashes": result.self_check_crashes,
            "self_check_gpu_s": result.self_check_gpu_s,
            "tokens": {k: int(v) for k, v in spent.items() if isinstance(v, (int, float))},
        }
        event.update(self._feedback_slim(parsed))
        return event

    def _file_session_notes(self, cand: Candidate, result) -> None:
        """@brief 把会话的 NOTES、自检结果与反馈抄进候选目录。

        NOTES 来自会话结果；自检先看 ``.selfcheck/self_checks.jsonl``，再看工作目录里的副本，最后看受保护的
        ``session_checks/`` 副本。反馈优先用结果里的文本，没有就读工作目录里的 ``FEEDBACK.md``（超过 16 KiB 时
        只留尾部）。候选目录不存在时直接返回。

        @param cand 本场评出的候选。
        @param result hmz 的会话结果。
        """
        folder = self.run_dir / "candidates" / f"{cand.idx:03d}_{cand.label}"
        if not folder.is_dir():
            return
        if result.notes:
            (folder / "NOTES.md").write_text(result.notes)
        work = self.run_dir / "agent_work" / cand.label.replace("/", "_")
        src = work / ".selfcheck" / "self_checks.jsonl"
        if not src.is_file():
            src = work / "self_checks.jsonl"
        protected = work.parent.parent / "session_checks" / work.name / "self_checks.jsonl"
        if protected.is_file():
            src = protected
        if src.is_file():
            (folder / "self_checks.jsonl").write_text(src.read_text())
        feedback = getattr(result, "feedback", "") or ""
        copied = work / "FEEDBACK.md"
        if not feedback and copied.is_file():
            try:
                feedback = copied.read_text(encoding="utf-8", errors="replace")
                if len(feedback) > 16384:
                    feedback = feedback[-16384:]
            except OSError:
                feedback = ""
        if feedback:
            (folder / "FEEDBACK.md").write_text(feedback)

    def _outer_line(self, cand: Candidate) -> str:
        """@brief 外层实测的一行摘要，写在下一场会话 NOTES 的第一行。

        @param cand 本场评出的候选。
        @return 形如「820 µs，状态 ok，换了种子」的中文句子。
        """
        rep = cand.report
        if rep is None:
            return "没有报告"
        us = f"{rep.candidate_us:.0f} µs" if isinstance(rep.candidate_us, (int, float)) else "没有微秒数"
        replaced = "换了种子" if cand.is_best else "没换种子"
        return f"{us}，状态 {rep.status}，{replaced}"

    def _session_brief(self, seedless: bool, label: str = "init", work_dir: Path | None = None):
        """@brief 组装交给一场 hmz 会话的 ``SessionBrief``。

        把任务、契约、Triton 源码、当前种子的报告与记分卡、最近 5 条历史、失败反馈、notes 摘要、运算量下界、
        证据（profiler / hw feedback / 经验提示）、噪声门槛与会话标签一并带上；``guidance_dir`` 指向开跑时冻结的
        ``agent_guidance/``。``harness_auto`` 打开时附上 harness overlay，``peer_tips`` 打开时附上同伴提示
        （序号用进程内下一条，避免一次失败的追加让未经核实的前提多活一场）。

        @param seedless 是否为初始生成（不把当前种子交给会话）。
        @param label 会话标签。
        @param work_dir 本场会话的工作目录；None 时按标签在 ``agent_work/`` 下推。
        @return ``kernel_agent.SessionBrief``。
        """
        from .kernel_agent import SessionBrief

        seed = self.best
        report_json = ""
        scorecard = ""
        bound = ""
        if seed is not None and seed.report is not None and not seedless:
            report_json = json.dumps(seed.report.to_json())
            scorecard = format_feedback(seed.report)[:1000]
            bound = _operation_bound(seed.report)
        history = "\n".join(self.history[-5:])
        tips = ""
        if self.search.peer_tips:
            from .agent_feedback import peer_tips_for

            # A failed append still assigned a seq in this process. Disk max+1 would let an unverified premise live one extra session.
            now = None if self._feedback_seq is None else self._feedback_seq + 1
            tips = peer_tips_for(self.run_dir, now=now)
        work = work_dir if work_dir is not None else (self.run_dir / "agent_work" / label.replace("/", "_"))
        snap = self.run_dir / "agent_guidance"
        guidance = str(snap.resolve()) if (snap / "experience.md").is_file() else ""
        harness_overlay = ""
        if self.search.harness_auto:
            from . import harness
            from tcenv.schema import PROTOCOL_VERSION

            harness_overlay = harness.overlay_text(
                self.run_dir, target_id=self.target_id, task=self.task_name,
                protocol_version=PROTOCOL_VERSION,
            )
        from tcenv.backends.registry import load_backend

        backend = load_backend(self.target.backend)
        return SessionBrief(
            repo=REPO_ROOT, task_name=self.task_name, dtype=self.dtype, size=self.size,
            kernel_name=self.task.contract.kernel_name, triton_source=self.task.triton_source,
            description=self.task.description, contract_text=self.task.contract.describe(),
            abi_stub=self.task.contract.abi_stub(), rtol=self.task.rtol, atol_rel=self.task.atol_rel,
            seed_source=None if seedless or seed is None else seed.source, seed_report_json=report_json,
            scorecard=scorecard, history=history, failure=self._failure_feedback, notes_summary=self._notes_summary,
            bound_line=bound, evidence=self._evidence, reps=self.search.eval_reps,
            require_contract_launch=self.search.require_contract_launch,
            min_improvement=self.search.min_improvement, noise_z=self.search.noise_z, seedless=seedless,
            peer_tips=tips, label=label, work_dir=str(Path(work).resolve()), guidance_dir=guidance,
            target_id=self.target_id, harness_overlay=harness_overlay,
            file_ext=self.target.file_ext, language_label=backend.display_name,
            source_fence=backend.source_fence, reference_skill=backend.reference_skill,
        )

    async def _initial_generation(self) -> None:
        """@brief 初始生成：最多 ``INITIAL_RESTARTS`` 次，两个委员会轮流接手。

        第一次是 first，失败换 second。一场 ``agent_failed`` 让 ``_agent_failures`` 加一并写一条历史；
        两场都失败就抛 ``StopRun("agent_failure")`` 停下整次搜索，否则计数归零、拿到种子就退出。

        @raises StopRun 连续两场会话都没有交出内核时。
        """
        for restart in range(INITIAL_RESTARTS):
            # 第一次是 first。失败换 second。两场都失败由 _one_session 的计数在下面停下。
            self._activate_round(1 if restart == 0 else 2)
            label = "init" if restart == 0 else f"init{restart + 1}"
            self.log(f"-- initial generation (attempt {restart + 1}/{INITIAL_RESTARTS}, {self._committee_name})")
            outcome = await self._one_session(label, seedless=True)
            if outcome == "agent_failed":
                self._agent_failures += 1
                self.history.append(f"initial: {self._committee_name} -> agent_failed")
                if self._agent_failures >= 2:
                    raise StopRun("agent_failure")
                continue
            self._agent_failures = 0
            if self.best:
                break

    # 续跑不重做已经有合格种子的初始生成，只把没做完的轮次交给 _rounds。上一会话没有种子，则和全新搜索一样停在 no_valid_initial_kernel。
    async def _search_resumed(self, res: RunResult) -> None:
        """@brief 续跑已从目录恢复的运行（``restore``）。

        续跑不重做已经有合格种子的初始生成，只把没做完的轮次交给 ``_rounds``，之后走和全新搜索一样的收尾。
        上一场会话没有种子时，和全新搜索一样停在 ``no_valid_initial_kernel``。搜索阶段已经结束、只差一次确认时
        直接返回：再开一场会话会白花钱，也会把轮号推过上限；轮次真用完时沿用 ``max_rounds``，否则写
        ``resumed_for_confirmation``。

        @param res 收尾记录（写 ``rounds`` 与 ``stop_reason``）。
        """
        snap = self._resume
        res.rounds = snap.rounds_done
        if not self.best:
            await self._initial_generation()
        if not self.best:
            res.stop_reason = "no_valid_initial_kernel"
            return
        if snap.search_finished:
            # 搜索阶段已经结束，只差一次确认；再开一场会话会白花钱，也会把轮号推过上限。
            # 轮次真用完时沿用 max_rounds，这样 result.json 的 stop_reason 和中断前一致。
            res.stop_reason = "max_rounds" if snap.next_round >= self.search.max_rounds else "resumed_for_confirmation"
            self.log("-- the search phase was already over; continuing at the confirmation")
            return
        await self._rounds(res, snap.next_round if snap.partial is None else snap.partial.rnd, snap.stagnant, None)

    # 续跑后的候选是新建对象。按标签从后往前找回同一颗内核，身份比较才会和中断前对得上。
    def _by_label(self, label: str | None) -> Candidate | None:
        """@brief 按标签找回候选。

        续跑后的候选是新建对象，按标签从后往前找，身份比较才会和中断前对得上。

        @param label 候选标签，可为 None。
        @return 最后一个同标签的候选；找不到时为 None。
        """
        for c in reversed(self.candidates):
            if c.label == label:
                return c
        return None

    # 每一轮：可选的 Instruments 证据、诊断规划、再 K 路补丁。分支同时启动，所以模型请求重叠；每路进入 _eval 时在 gpu_lock 上排队，GPU 评测串行。只有换过种子才不算停滞。
    async def _rounds(self, res: RunResult, first_round: int, stagnant: int, partial: Any = None) -> None:
        """@brief 每轮一场新的 hmz 会话，两个委员会轮流接手。

        每一轮：可选的 Instruments 证据（profiler、hw feedback、经验提示、目标提示、状态行）、诊断规划，然后交出一版；
        进入 ``_eval`` 时在 ``gpu_lock`` 上排队，GPU 评测串行。只有换过种子才不算停滞（``next_stagnant``），
        停滞到 ``patience`` 就以 ``converged`` 停下。轮数上限每轮重新读：一轮在飞时 control 文件可以把它抬高。
        ``search.stop_at_roofline`` 打开且种子已贴着下界时以 ``at_roofline`` 停下（这是启发式，不是物理最优）。
        一场 ``agent_failed`` 让 ``_agent_failures`` 加一并写历史，连续两场就以 ``agent_failure`` 停下。
        ``partial`` 被忽略：续跑的那一轮重新开一场会话。会话内触发熔断（``StopRun``）或预算不够（``BudgetExhausted``）时
        异常直接向上传，由 ``run`` 收进 ``stop_reason``。

        @param res 收尾记录（写 ``rounds`` 与 ``stop_reason``）。
        @param first_round 本轮起始轮号。
        @param stagnant 进入本轮时的连续停滞轮数。
        @param partial 续跑快照里的未完成轮次信息，忽略。
        """
        del partial
        rnd = first_round
        while rnd <= self.search.max_rounds:
            res.rounds = rnd
            self._round = rnd
            self._stagnant = stagnant
            self._activate_round(rnd)
            seed = self.best
            self.log(f"-- round {rnd}: seed {seed.label} at {seed.speedup:.3f}x, spent ${self.ledger.total_usd:.3f}, {self._committee_name}")
            if self.search.stop_at_roofline and at_roofline(seed.report, self._required_gain(seed, seed)):
                gap = (getattr(seed.report, "roofline", None) or {}).get("time_over_lower_bound")
                res.stop_reason = "at_roofline"
                self.log(f"-- heuristic stop: {seed.label} is {gap:.3f}x its estimated lower bound, inside the noise gate. This is not a physical optimum.")
                return
            evidence = await self._profile_seed(seed, rnd)
            if self.search.hw_feedback and self.backend.capabilities.supports_hw_features:
                from .hwfeedback import hw_feedback

                block = await asyncio.to_thread(hw_feedback, self.task, seed.source, seed.report, log=self.log, backend_id=self.target.backend)
                if block:
                    evidence += "\n\n" + block
            if self.search.experience_index:
                from .experience import load_index, retrieve_experience_hints
                from .symptom import bound_of

                hints = retrieve_experience_hints(load_index(Path(self.search.experience_index)), self.task_name, bound_of(seed.report), exclude_task=True)
                if hints:
                    evidence += "\n\n" + hints
            if self.search.target_note:
                note = target_note(seed.report)
                if note:
                    evidence += "\n\n" + note
            if self.search.status_lines:
                evidence += "\n\n" + status_lines(
                    rnd=rnd, max_rounds=self.search.max_rounds, patience_left=max(0, self.search.patience - stagnant),
                    spent=self.ledger.total_usd, budget=self.ledger.budget_usd, required_gain=self._required_gain(seed, seed),
                )
            self._evidence = evidence
            before = seed
            outcome = await self._one_session(f"r{rnd}.p0", seedless=False)
            if outcome == "agent_failed":
                self._agent_failures += 1
                self.history.append(f"round {rnd}: {self._committee_name} -> agent_failed")
                if self._agent_failures >= 2:
                    res.stop_reason = "agent_failure"
                    return
            else:
                self._agent_failures = 0
                cand = self._by_label(f"r{rnd}.p0")
                if cand is not None and cand.report is not None and cand.speedup:
                    tag = f"{cand.speedup:.3f}x" + (" (new best)" if cand.is_best else " (within measurement noise of the best)" if cand.within_noise else " (not better)")
                elif outcome == "unchanged":
                    tag = "unchanged"
                elif cand is not None and cand.report is not None:
                    tag = failure_phrase(cand.report.status, cand.report.message) if self.search.rich_history else cand.report.status
                else:
                    tag = outcome
                self.history.append(f"round {rnd}: {self._committee_name} -> {tag}")
            stagnant = next_stagnant(self.best is not before, stagnant)
            self._stagnant = stagnant
            if patience_exhausted(stagnant, self.search.patience):
                res.stop_reason = f"converged ({self.search.patience} rounds without improvement)"
                return
            # 这一轮里看板可能加了预算或轮数。下一轮的 while 比的是已经读进来的上限。
            self.ledger.check()
            rnd += 1
        res.stop_reason = "max_rounds"

    # 复测池：当前优胜者，再加上最多 confirm_top-1 颗合格内核。优胜者相对它们的优势没有超过 _required_gain 时，它们算噪声范围内的并列，必须一起重测，不能只信搜索过程中的最大值。
    def _confirm_pool(self) -> list[Candidate]:
        """@brief 复测池：当前优胜者，加上最多 ``confirm_top - 1`` 颗它没有显著快过的合格内核。

        优胜者相对它们的优势没有超过 ``_required_gain`` 时，它们算噪声范围内的并列，必须一起重测，
        不能只信搜索过程中的最大值。不合格、被复查否决、或源码与已在池中的候选相同的都跳过。

        @return 复测池，第一项恒为当前优胜者。
        """
        pool, seen = [self.best], {_sha(self.best.source)}
        for c in sorted(self.candidates, key=lambda c: c.speedup, reverse=True):
            if len(pool) >= self.search.confirm_top:
                break
            if not c.eligible or c.recheck_rejected or _sha(c.source) in seen:
                continue
            seen.add(_sha(c.source))
            # 优胜者并没有显著快过 c，c 就留在并列里。源码相同的候选只进池一次。
            if in_confirm_pool(self.best.speedup, c.speedup, self._required_gain(self.best, c)):
                pool.append(c)
        return pool

    # 搜索结束后重测优胜者与并列者。每一批都要成功，同一批共用一套输入。搜索种子留下，除非另一颗完整复测超过噪声门槛。correctness_only 或已经挂死则不做。复测超时就停下剩余复测。
    async def _domain_sweep(self, win) -> None:
        """@brief 对确认后的优胜者做一次只查正确性的全域检查（``domain`` 用例）。

        任务没有 ``domain_case`` 时直接返回。已经挂死则不再上 GPU。结果写成 ``domain_report.json``，
        失败也不改已经发布的数字：这份 sweep 是证据，不是第二次丢掉确认的机会。

        @param win 确认后的优胜者。
        """
        if not (self.task.meta or {}).get("domain_case"):
            return
        phase_seed = self._phase_seed("domain", win.label, win.source)
        try:
            if self.hangs >= self.search.max_hangs:
                self.log("   domain sweep skipped: a GPU hang already stopped further launches")
                return
            async with self.gpu_lock:
                rep = await asyncio.to_thread(
                    self.evaluate_fn, self.task_name, self.dtype, self.size, win.source, mode="quick", seed=phase_seed, reps=1,
                    cases=("domain",), progress_path=self.run_dir / "eval_progress.json",
                    **({"require_contract_launch": True} if self.search.require_contract_launch else {}),
                    measure_serial=self.search.measure_serial,
                    eval_cache=self.search.eval_cache,
                    target_id=self.target_id,
                )
            if rep.status == "timeout":
                self.hangs += 1
                self._publish_state("confirm", inflight=None)
            doc = rep.to_json()
        except Exception as e:  # noqa: BLE001 - the sweep is evidence, not a second chance to lose the confirmation
            doc = {"status": f"error: {type(e).__name__}: {e}", "correctness": {"coverage": {"kind": "sampling", "scope": ""}}}
            self.log(f"   domain sweep failed: {doc['status']}")
        control.write_json_atomic(self.run_dir / "domain_report.json", doc)
        kind = ((doc.get("correctness") or {}).get("coverage") or {}).get("kind")
        self.log(f"   domain sweep: {doc.get('status')} ({kind})")

    async def _confirm(self, res: RunResult) -> None:
        """@brief 用新种子、更多重复次数重测优胜者与它的统计并列者。

        每一批都必须成功。同一批里的并列候选共用一个输入种子，比的是同一套输入；批次之间换种子，
        种子不掺内核正文，所以换一颗核不会换掉这一批的输入。发布的内核是搜索最优，除非另一颗完整复测
        越过了噪声门槛（``choose_confirmed``）。后面某一批失败不会把前面成功的批次变成一次确认。
        没有种子、已经熔断、或 ``correctness_only`` 时不做复测：熔断后再上 GPU 可能再次冻住显示。
        复测超时就停下剩余的批次与并列者，搜索已经结束，这里只停止继续占 GPU。确认是「再测一次」
        而不是取回上一次的结果，所以 ``eval_cache=False``。确认成功后调用 ``_domain_sweep``。
        评测本身抛 ``SchemaError`` 时直接上抛，不把它当成一次可忽略的批次失败。

        @param res 收尾记录（写 ``confirmations``、``confirmed_*``、``candidate_us`` 与 ``baseline_us``）。
        @raises SchemaError 当评测报告不符合 schema 时（直接上抛）。
        """
        # 没有种子、已经熔断、或只做正确性实验时不复测。熔断后再上 GPU 可能再次冻住显示。
        if not self.best or self.hangs >= self.search.max_hangs or self.search.correctness_only:
            return
        pool = self._confirm_pool()
        if self.run_seed is None:
            self.log(f"-- confirming {', '.join(c.label for c in pool)} with a fresh seed")
        else:
            self.log(f"-- confirming {', '.join(c.label for c in pool)} with one seed per input batch")
        self._confirmations = res.confirmations
        self._publish_state("confirm", inflight=", ".join(c.label for c in pool))
        batches = max(1, int(getattr(self.search, "confirm_batches", 1) or 1))
        # (候选, 中位数, 每一批的值, 与中位数一致的那份报告)
        confirmed: list[tuple[Candidate, float, list[float], EvalReport]] = []
        for c in pool:
            entry: dict[str, Any] = {"label": c.label, "search_speedup": c.speedup, "failed_batches": 0}
            res.confirmations.append(entry)
            self._confirmations = res.confirmations
            self._publish_state("confirm", inflight=c.label)
            values: list[float] = []
            reports: list[EvalReport] = []
            seeds: list[int | None] = []
            stop = False
            for batch in range(batches):
                # 同一批里的并列候选共用一个种子，比的是同一套输入。批次之间换种子。
                # 种子不掺内核正文，所以换一颗核不会换掉这一批的输入。
                seed_label = f"b{batch + 1}"
                try:
                    async with self.gpu_lock:
                        phase_seed = self._phase_seed("confirm", seed_label, "")
                        rep = await asyncio.to_thread(
                            self.evaluate_fn, self.task_name, self.dtype, self.size, c.source, mode="full", seed=phase_seed, reps=self.search.eval_reps * 2,
                            progress_path=self.run_dir / "eval_progress.json",
                            **({"require_contract_launch": True} if self.search.require_contract_launch else {}),
                            measure_serial=self.search.measure_serial,
                            # 确认是「再测一次」，不是取回上一次的结果。确定性种子加上缓存，会让 resume
                            # 或同一个 --seed 的重跑在 2 ms 内复述旧报告，而发布门会把它当成复测。
                            eval_cache=False,
                            target_id=self.target_id,
                        )
                    seeds.append(phase_seed if phase_seed is not None else rep.seed)
                    if batch == 0:
                        entry["eval_seed"] = seeds[-1]
                        entry["eval_seed_from"] = "confirm" if phase_seed is not None else "random"
                except SchemaError:
                    raise
                except Exception as e:  # noqa: BLE001 - a failed confirmation must not lose the run
                    entry["status"] = f"error: {type(e).__name__}: {e}"
                    entry["failed_batches"] = batches - batch
                    self.log(f"   {c.label}: confirmation batch {batch + 1} failed: {type(e).__name__}: {str(e)[:160]}")
                    self._confirmations = res.confirmations
                    self._publish_state("confirm", inflight=c.label)
                    stop = True
                    break
                if rep.status == "ok" and rep.speedup:
                    values.append(float(rep.speedup))
                    reports.append(rep)
                    continue
                failed = self._failure_record(c, rep, phase_seed, "confirm", rep.status not in ("timeout", "crash"))
                folder = self.run_dir / "rechecks"
                folder.mkdir(exist_ok=True)
                (folder / f"confirm_{c.label}{'' if batch == 0 else f'.b{batch + 1}'}.json").write_text(json.dumps(failed, indent=1, default=str))
                entry["failed_batches"] = int(entry["failed_batches"]) + 1
                entry["status"] = rep.status
                self.log(f"   {c.label}: confirmation {rep.status} {rep.message[:120]}")
                if rep.status == "timeout":
                    # 复测超时就不再测剩下的批次与并列者。搜索已经结束，这里只停止继续占 GPU。
                    self.hangs += 1
                stop = True
                break
            entry["eval_seeds"] = seeds
            # 有一批失败就把失败状态留下。前面成功的批次不能把整颗核标成 ok。
            if not stop and len(values) == batches and reports:
                median = float(statistics.median(values))
                entry["status"] = "ok"
                entry["confirmed_speedup"] = median
                entry["confirmed_values"] = values
                entry["confirmed_worst"] = min(values)
                entry["confirmed_best"] = max(values)
                rep = min(reports, key=lambda r: abs(float(r.speedup) - median))
                confirmed.append((c, median, values, rep))
                self.log(f"   {c.label}: confirmed {median:.3f}x (median of {len(values)} batch(es): " + ", ".join(f"{v:.3f}" for v in values) + ")")
            self._confirmations = res.confirmations
            self._publish_state("confirm", inflight=c.label)
            if stop:
                break
        if not confirmed:
            res.confirmed_status = res.confirmations[0].get("status") or "error"
            return
        measured = [
            {"candidate": c, "median": median, "values": values, "report": rep, "se": speedup_se(rep), "search_best": c is self.best}
            for c, median, values, rep in confirmed
        ]
        chosen = choose_confirmed(measured, min_improvement=self.search.min_improvement, noise_z=self.search.noise_z)
        if chosen is None:
            res.confirmed_status = "uncertain"
            self.log("   confirmation did not separate a winner from the noise")
            return
        win, median, values, rep = chosen["candidate"], chosen["median"], chosen["values"], chosen["report"]
        self._confirm_winner = win
        if win is not self.best:
            self.log(f"   {win.label} clears the noise gate against {self.best.label}; it becomes the result")
            self.best.is_best, win.is_best = False, True
            self.best = win
        res.confirmed_status, res.confirmed_speedup = "ok", median
        res.confirmed_values, res.confirmed_worst, res.confirmed_best = list(values), min(values), max(values)
        res.candidate_us, res.baseline_us = rep.candidate_us, rep.baseline_us
        control.write_json_atomic(self.run_dir / "confirm_report.json", rep.to_json())
        if self.hangs >= self.search.max_hangs:
            self.log("   domain sweep skipped: confirmation already hit the hang limit")
            return
        await self._domain_sweep(win)

    # 种子慢于最强基线（speedup < profile_poor_below）时才抓 Instruments，附给规划和每一路补丁。失败、超过 profile_max 或已经挂死都不影响搜索；同一源码只抓一次。痕迹在 gpu_lock 里跑，和评测一样独占 GPU。
    async def _profile_seed(self, seed: Candidate, rnd: int) -> str:
        """@brief 种子慢于最强基线时为它抓一份 Instruments 痕迹。

        只在 ``search.profile_poor`` 打开、种子加速比低于 ``profile_poor_below``、没有挂死、且没超过
        ``profile_max`` 份时才抓，同一源码只抓一次。抓痕迹与评测、quick_check 共用 ``gpu_lock``，独占 GPU。
        痕迹文本与 json 写进 ``profiles/r<轮号>_<种子标签>.txt|json``，同一份文本会附给规划与每一路补丁作者。

        @param seed 本轮的种子。
        @param rnd 轮号。
        @return 附给规划与每一路补丁作者的文本；关闭、已够快、超预算或抓取失败时为空串。
        """
        s = self.search
        # 没开、已经够快、或搜索已熔断：不占 GPU。规划照常进行，只是没有这份证据。
        if not s.profile_poor or seed.speedup >= s.profile_poor_below or self.hangs >= s.max_hangs:
            return ""
        key = _sha(seed.source)
        if key in self._profile_text:
            return self._profile_text[key]
        if len(self.profiles) >= s.profile_max:
            return ""
        from tcenv.backends.metal import xctrace

        self._publish_state("profile")
        self.log(f"-- profiling the seed {seed.label} ({seed.speedup:.3f}x of the best baseline) with xctrace")
        entry: dict[str, Any] = {"round": rnd, "seed": seed.label, "seed_speedup": seed.speedup}
        text = ""
        try:
            # 抓痕迹与评测、quick_check 共用同一把锁，避免和正在跑的内核抢 GPU。
            async with self.gpu_lock:
                summ = await asyncio.to_thread(
                    xctrace.profile_kernel, self.task_name, self.dtype, self.size, seed.source, target_id=self.target_id,
                )
            body = xctrace.format_for_llm(summ)
            pdir = self.run_dir / "profiles"
            pdir.mkdir(exist_ok=True)
            (pdir / f"r{rnd}_{seed.label}.txt").write_text(body)
            (pdir / f"r{rnd}_{seed.label}.json").write_text(json.dumps(summ.to_json(), indent=1))
            entry.update(spilled_bytes=summ.spilled_bytes, shader_code_bytes=summ.shader_code_bytes, gpu_active_fraction=summ.gpu_active_fraction)
            text = profiler_evidence_text(body)
            self.log(f"   spilled {summ.spilled_bytes} B in {summ.spill_events} events; shader {summ.shader_code_bytes} B; GPU active {summ.gpu_active_fraction:.0%}" if summ.gpu_active_fraction is not None else "   trace taken")
        except Exception as e:  # noqa: BLE001 - profiling is a bonus, never a failure
            entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            self.log(f"   profiling failed: {entry['error']}")
        self.profiles.append(entry)
        self._profile_text[key] = text
        return text

    # 搜索循环之外给最终优胜者补一份 Instruments。失败只留下说明，不改变已经选出的内核；已经挂死则不再上 GPU。
    async def _profile_best(self) -> None:
        """@brief 搜索循环之外给最终优胜者补一份 Instruments 痕迹（``search.profile_best``）。

        痕迹写在结果旁边（``xctrace_profile.txt`` 与 ``xctrace_profile.json``）。失败只在文件里留一句说明，
        不改变已经选出的内核；已经挂死则不再上 GPU。
        """
        if not self.best or self.hangs >= self.search.max_hangs:
            return
        from tcenv.backends.metal import xctrace

        self._publish_state("profile")
        self.log("-- profiling the best kernel with xctrace")
        try:
            async with self.gpu_lock:
                summ = await asyncio.to_thread(
                    xctrace.profile_kernel, self.task_name, self.dtype, self.size, self.best.source, target_id=self.target_id,
                )
            (self.run_dir / "xctrace_profile.txt").write_text(xctrace.format_summary(summ))
            (self.run_dir / "xctrace_profile.json").write_text(json.dumps(summ.to_json(), indent=1))
        except Exception as e:  # noqa: BLE001 - profiling is a bonus, never a failure
            (self.run_dir / "xctrace_profile.txt").write_text(f"profiling failed: {type(e).__name__}: {e}")

    def _archive_if_confirmed(self, result_doc: dict) -> None:
        """@brief 只有发布门确认之后才把内核收进 catalog。

        ``experiments.common.conclusion_of_run`` 判定 ``publication`` 不是 ``confirmed``、或没有优胜者源码时直接返回。
        收录的字段包含目标、后端、任务、加速比、确认阶段各批的值与候选/基线微秒数。

        @param result_doc 即将写进 ``result.json`` 的文档。
        """
        from experiments.common import conclusion_of_run

        verdict = conclusion_of_run(self.run_dir, result_doc)
        if verdict.get("publication") != "confirmed" or self.best is None or not self.best.source:
            return
        from .kernel_catalog import archive_confirmed

        report = {}
        path = self.run_dir / "confirm_report.json"
        if path.is_file():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                report = {}
        archive_confirmed(
            target_id=self.target_id,
            backend=self.target.backend,
            task=self.task_name,
            dtype=self.dtype,
            size=self.size,
            source=self.best.source,
            speedup=float(verdict.get("speedup") or 0),
            run_id=self.run_dir.name,
            label=self.best.label,
            device_name=str(report.get("device_name") or ""),
            gpu_family=str(report.get("gpu_family") or ""),
            file_ext=self.target.file_ext or ".metal",
            confirmed_values=list(result_doc.get("confirmed_values") or []),
            candidate_us=result_doc.get("candidate_us"),
            baseline_us=result_doc.get("baseline_us") or {},
        )

    # 把优胜者源码和账本写成 result.json。有 confirmed_speedup 时对外强调复测值，否则用搜索时的 best_speedup。中途加过的预算在 budget_events 里，结束时的上限可以高于开跑时的 initial_budget_usd。
    def _finish(self, res: RunResult) -> RunResult:
        """@brief 把优胜者源码、``result.json``、``ledger.json`` 与 ``state.json`` 收尾写好。

        有 ``confirmed_speedup`` 时对外强调复测值，否则用搜索时的 ``best_speedup``。中途加过的预算在
        ``budget_events`` 里，结束时的上限可以高于开跑时的 ``initial_budget_usd``。``confirmation_source_sha256``
        只在确认状态为 ok 时写，用来把发布门绑到那颗内核。最后调 ``_archive_if_confirmed`` 与
        ``_publish_state("done")``。

        @param res 收尾记录。
        @return 补全后的同一个 ``RunResult``。
        """
        self.ledger.refresh_budget()
        res.spent_usd = self.ledger.total_usd
        res.budget_usd = self.ledger.budget_usd  # the budget the run ended with (it grows with every top-up)
        res.initial_budget_usd = self.ledger.initial_budget_usd
        res.budget_events = [e for e in control.read_events(self.run_dir) if e.get("kind") in ("applied", "resume")]
        res.resumes = self._resumes
        res.evaluations = self.evaluations
        res.hangs = self.hangs
        res.trace = self.trace
        res.candidates = [c.brief() for c in self.candidates]
        res.ledger = self.ledger.summary()
        res.profiles = self.profiles
        if self._search_seed:
            res.search_seed_label, res.search_seed_speedup = self._search_seed.label, self._search_seed.speedup
        if self._confirm_winner:
            res.confirm_winner_label = self._confirm_winner.label
            if res.confirmed_status == "ok":
                res.confirmation_source_sha256 = hashlib.sha256(self._confirm_winner.source.encode("utf-8")).hexdigest()
        if self.best:
            res.best_label, res.best_speedup = self.best.label, self.best.speedup
            control.write_text_atomic(self.run_dir / self.best_name, self.best.source)
        res.target_id = self.target_id
        res.backend = self.target.backend
        res.backend_version = self.target.backend_version
        res.protocol_version = PROTOCOL_VERSION
        result_doc = asdict(res)
        result_doc["schema_version"] = RESULT_SCHEMA_VERSION
        control.write_json_atomic(self.run_dir / "result.json", result_doc)
        self._archive_if_confirmed(result_doc)
        control.write_json_atomic(self.run_dir / "ledger.json", {"schema_version": LEDGER_SCHEMA_VERSION, "summary": self.ledger.summary(), "requests": self.ledger.requests})
        self._confirmations = res.confirmations
        self._publish_state("done", stop_reason=res.stop_reason)
        sp = f"{res.confirmed_speedup:.3f}x confirmed" if res.confirmed_status == "ok" and res.confirmation_source_sha256 else (f"{res.best_speedup:.3f}x (not confirmed)" if res.best_speedup else "no valid kernel")
        self.log(f"== done: {sp} | spent ${res.spent_usd:.3f} | {res.evaluations} evaluations | stop: {res.stop_reason} | {self.run_dir}")
        return res
