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
from .artifacts import LEDGER_SCHEMA_VERSION, PROTOCOL_VERSION, RESULT_SCHEMA_VERSION, STATE_SCHEMA_VERSION
from .feedback_extra import abi_signature, at_roofline, failure_phrase, status_lines, target_note, threadgroup_limit_note
from .llm import LLM, BudgetExhausted, CostLedger, LLMConfig, LLMError, SearchConfig, Transcript
from .policy import candidate_eligible, choose_confirmed, comparison_speedup, in_confirm_pool, next_stagnant, patience_exhausted, relation_to_best, required_gain
from .committees import committees_record, default_committees, handoff_for, order_committees
from .parse import parse_plans
from .symptom import bound_of

REPO_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_CHARS = 6000
INITIAL_RESTARTS = 2


def _git_identity(root: Path) -> tuple[str, bool | None]:
    """(commit hash, dirty). ``("unknown", None)`` when git cannot be read. This does not touch the GPU."""
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
    """48-bit seed for one evaluation, derived from the run seed and the candidate's identity.

    ``source`` is the kernel text before the contract header is prepended. ``phase`` is ``search``, ``confirm``, ``recheck`` or ``domain``.
    Adding ``domain`` does not change the payload of the other phases.
    The label is the one that exists before evaluation (``r2.p1.fix1``, ``init.s2``, ``prior``), not ``Candidate.idx``.
    The same inputs always return the same seed. This does not control the model provider's sampling or the GPU's temperature."""
    if phase not in _EVAL_PHASES:
        raise ValueError(f"phase must be one of {_EVAL_PHASES}")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    payload = json.dumps(["ai_lowerling/eval_seed/v1", int(run_seed), phase, label, source_sha256], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:6], "big")


def _valid_run_seed(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 0 <= value < 2**64:
        return value
    return None


def _load_run_seed(run_dir: Path) -> int | None:
    """The seed recorded for this run, or None for a run that predates ``run_seed`` (those evaluations stay random)."""
    doc = control.read_json(Path(run_dir) / control.CONFIG_FILE) or {}
    recorded = doc.get("run_seed")
    if recorded is None:
        recorded = doc.get("seed")
    return _valid_run_seed(recorded)


_RECHECK_KEYS = ("class", "task", "dtype", "size", "label", "source_sha256", "search_seed", "recheck_seed", "case", "tolerance", "speed_relation", "counts_as_kernel_error", "retest_command", "human_permission_required")


def _strip_counterexample(doc):
    """Drop VerifyResult.counterexample anywhere in a stored report. A recheck file must not carry that field."""
    if isinstance(doc, dict):
        return {key: _strip_counterexample(value) for key, value in doc.items() if key != "counterexample"}
    if isinstance(doc, list):
        return [_strip_counterexample(value) for value in doc]
    return doc


def _report_without_counterexample(report: EvalReport | None):
    if report is None:
        return None
    return _strip_counterexample(report.to_json())


def _contains_counterexample(doc) -> bool:
    if isinstance(doc, dict):
        if "counterexample" in doc:
            return True
        return any(_contains_counterexample(value) for value in doc.values())
    if isinstance(doc, list):
        return any(_contains_counterexample(value) for value in doc)
    return False


def check_recheck_record(doc: dict) -> list[str]:
    """What's missing from a differential-recheck record. Empty means the record is complete enough to read offline.
    This does not run the kernel."""
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
    """Raised when the run must end immediately (GPU hang circuit breaker)."""


class RunStartError(Exception):
    """A new run was refused before any model request: the directory already holds a run, or ``run_config.json`` could not be written."""


# 一颗被评过（或即将评）的内核。能否成为下一轮种子不看标签，只看 eligible 加上是否快过测量噪声。
@dataclass
class Candidate:
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
        report = self.report
        return candidate_eligible(
            report is not None and bool(report.eligible),
            None if report is None else report.speedup,
            correctness_only=self.correctness_only,
        )

    # 不合格时加速比按 0，缺失的计时不可能把当前种子挤掉。和续跑回放共用 policy.comparison_speedup。
    @property
    def speedup(self) -> float:
        report = self.report
        return comparison_speedup(self.eligible, None if report is None else report.speedup)

    def brief(self) -> dict:
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
    protocol_version: str = ""


# 源码的内容身份：评测缓存、复测去重、profiler 缓存都用它。同一份文本不会被当成两个内核。
def _sha(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()


# 回复里抽不出带约定内核名的 Metal 块时追问的那一句。抽失败只消耗这一问，不会产生候选，也不会上 GPU。
def nudge_user(kernel_name: str) -> str:
    """The reminder sent when a reply holds no ```metal block defining the kernel."""
    return (
        f"Your reply did not contain a ```metal code block defining `kernel void {kernel_name}`, "
        "or a ```metal block with a /* taic-launch */ section and at least one `kernel void`. "
        "Reply again with the complete kernel in exactly one ```metal block."
    )


def _strip_reasoning(messages: list[dict]) -> list[dict]:
    """A copy of ``messages`` without ``reasoning_content``. A truncated thinking trace is tens of thousands of tokens and buys nothing on the retry."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and "reasoning_content" in m:
            m = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(m)
    return out


def _retry_without_reasoning(messages: list[dict]) -> list[dict]:
    """The follow-up after a length-capped reply that contained no kernel.

    The echoed reasoning is dropped. An assistant turn with no visible text is dropped too, so the same question is asked again
    instead of being continued from an empty reply. ``experiments/truncation_replay.py`` compares this with the other arms; change
    this function if that replay picks a different winner.
    """
    messages = _strip_reasoning(messages)
    if messages and messages[-1].get("role") == "assistant" and not (messages[-1].get("content") or "").strip():
        messages = messages[:-1]
    return messages


# 种子慢于最强基线时，附在规划与每一路补丁反馈后面的 Instruments 文字。抓取失败或没开 profile_poor 时根本不会用到它，搜索照常进行。
def profiler_evidence_text(body: str) -> str:
    """The text appended to the feedback of the planner and of every patch author when a seed was traced; ``body`` is what ``profiles/r*_<seed>.txt`` holds."""
    return (
        "\n\nProfiler evidence for this kernel (Instruments Metal System Trace; taken because the kernel is slower than the strongest baseline). "
        "Read it with the \"Instruments trace\" section of Skill 2. Name the finding or counter that your diagnosis rests on (spills, the saturated limiter, low occupancy, ...) "
        "and make the changes address it; if the evidence says nothing about your hypothesis, say so instead of inventing numbers:\n" + body
    )


def _notes_for_next(outer: str, notes: str, limit: int = 2500) -> str:
    """The next session's NOTES block. The outer measurement is the first line, written by the search, not by the agent.

    Notes are appended, so a truncated file keeps the newest lines.
    """
    body = (notes or "").strip()
    if len(body) > limit:
        body = "…\n" + body[-limit:]
    return f"外层实测：{outer}\n{body}".strip()


def _operation_bound(report) -> str:
    """One line from the report's roofline and baselines. Empty when either number is missing."""
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
    """Approximate relative standard error of a measured speedup. 0 when the report has no timing statistics.
    Not a confidence interval."""
    from tcenv.bench import speedup_se_of_report

    return speedup_se_of_report(rep)


class Orchestrator:
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
        """``resuming``: the directory already holds a run (see ``taic.resume``); nothing in it is rewritten until ``restore()`` has loaded it.
        ``committees``: exactly first and second. Omitted means two committees that copy ``llm_cfg``. There is no single-conversation mode."""
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
        try:
            self.committees = default_committees(llm_cfg) if committees is None else order_committees(list(committees))
        except LLMError as e:
            raise RunStartError(str(e)) from e
        self._committee_name = self.committees[0].name
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir) if run_dir else REPO_ROOT / "runs" / self.target_id / f"{stamp}_{task_name}_{dtype}_{llm_cfg.model}"
        self.log = log
        self.system = prompts.system_prompt(self.task, use_tools=llm_cfg.use_tools, profiler_evidence=search.profile_poor, hw_feedback=search.hw_feedback, target_id=self.target_id)
        self.from_run = Path(from_run) if from_run else None
        self.from_catalog = from_catalog
        # 新运行不接管已经有文件的目录。续跑（resuming=True）才读既有目录，并且不改写 run_config.json、不重新抽 run_seed。
        if not resuming and self.run_dir.exists() and any(self.run_dir.iterdir()):
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
        # 续跑时目录里已有上一会话的 state.json。先空写会把看板抹掉，所以等到 restore 把账本和候选装回来再发布。
        # 新运行的初始稿用角色 0。续跑要等知道轮次后再绑，不能在这里先绑成角色 0。
        if not resuming:
            self._activate_round(0)
            self._publish_state()

    # 续跑要把上一会话已经花掉的墙钟算进去，否则轨迹上的时间从 0 重新开始，和账本对不上。
    def _elapsed(self) -> float:
        """Search time of the whole run, including earlier sessions of a resumed one."""
        return self._t_offset + (time.time() - self.t0)

    # 第 1 轮和初始稿是 first；之后按轮号在两个委员会之间换。续跑只看轮号，不看上一轮的对话。
    def _activate_round(self, rnd: int) -> None:
        """Bind the committee for round ``rnd``. The name changes even when the model id does not.

        The same base URL, key and model keep the HTTP client. A changed model id updates the ledger's prices.
        A different endpoint or key opens another client on the same ledger and transcript. A client passed
        in by a test is reused for both committees, so one scripted queue serves the whole run."""
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
        """Keep the current HTTP client and follow ``cfg``. The model id changed, so the ledger prices follow it too."""
        if cfg.api_key:
            self.llm.transcript._secrets = list({*self.llm.transcript._secrets, cfg.api_key})
        self.llm.cfg = cfg
        self.ledger.cfg = cfg

    def _open_llm(self, cfg: LLMConfig, *, update_prices: bool) -> None:
        """A different endpoint or key: a new client, the same ledger and the same transcript. Closed with the others at the end of ``run``."""
        fresh = LLM(cfg, self.ledger, self.llm.transcript, client=None, max_concurrency=self._llm_slots)
        fresh.progress_path = self.llm.progress_path
        self._llms.append(fresh)
        self.llm = fresh
        if update_prices:
            self.ledger.cfg = cfg

    def _handoff(self, rnd: int) -> str:
        """Empty on the initial draft and on round 1. Every later round names its committee, including when the model id is unchanged."""
        if rnd <= 1:
            return ""
        return handoff_for(self.committees[(rnd - 1) % 2].name)

    async def _close_clients(self) -> None:
        """Close every HTTP client this run opened. A scripted client has nothing to close."""
        seen: set[int] = set()
        for llm in self._llms:
            ident = id(llm.client)
            if ident in seen:
                continue
            seen.add(ident)
            await llm.aclose()

    # 给 `taic resume` 留的设置。故意不写 API key 和端点。写失败向上抛，调用方不得进入 run()。
    def _write_run_config(self) -> None:
        """The settings of this run, for ``taic.cli resume`` (no credentials: the API key and the endpoint stay out of it).

        Raises ``RunStartError`` when the file cannot be written. The directory created for this attempt may stay empty;
        an older run directory is never deleted or overwritten (that case is refused before this method)."""
        llm = {k: v for k, v in asdict(self.cfg).items() if k not in ("api_key", "base_url")}
        contract_text = self.task.contract.describe()
        triton_text = self.task.triton_source
        commit, dirty = _git_identity(REPO_ROOT)
        body = {
                "schema_version": 2,
                "protocol_version": PROTOCOL_VERSION,
                "target": {
                    "id": self.target_id,
                    "backend": self.target.backend,
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
            (self.run_dir / "system_prompt.txt").write_text(self.system)
        except OSError as e:
            raise RunStartError(f"cannot start: run_config.json was not written ({e})") from e

    def _agent_guidance_record(self) -> dict:
        """Freeze the two files copied into every session, and record the skill hash.

        A resume reads this directory and does not copy the repository again. The skill file
        stays where hmz loads it; only its hash is recorded.
        """
        from .kernel_agent import FLOW
        from .target import skill_path

        dest = self.run_dir / "agent_guidance"
        dest.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name, key in (("device.md", "device"), ("experience.md", "experience"), ("msl_reference.md", "msl_reference")):
            src = skill_path(key, self.target_id)
            data = src.read_bytes() if src.is_file() else b""
            (dest / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        skill = FLOW / "skills" / "taic-tools" / "SKILL.md"
        data = skill.read_bytes() if skill.is_file() else b""
        hashes["SKILL.md"] = hashlib.sha256(data).hexdigest()
        return hashes

    def _cache_this(self, source: str) -> bool:
        """A report saved before launch_protocol is not a verdict for a taic-launch kernel.

        Runs started by this code write ``launch_protocol: 1``. A resumed directory that lacks the key
        was evaluated by the single-kernel path, so a source that contains the marker is evaluated again
        the next time the search asks for it.
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
        """The ledger took a new total budget from the run's control file."""
        self.search.budget_usd = new
        who = str(doc.get("who") or "dashboard")
        self.log(f"  budget ${old:.2f} -> ${new:.2f} (requested by {who}; spent ${self.ledger.total_usd:.3f})")
        written = control.record_event(self.run_dir, "applied", old, new, who=who, spent_usd=self.ledger.total_usd)
        if not written.get("persisted"):
            self.log("  budget event was not saved; budget_events.json is not a complete record")
        self._publish_state()

    # 控制文件里的 max_rounds / patience 只升不降。仪表盘的「重置」用它去掉次数上限；搜索仍会在新预算花完时停下。
    def _control_limits(self, doc: dict) -> None:
        """Raise ``max_rounds`` and ``patience`` when the control file asks for a higher value."""
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
        """Rewrite ``state.json``. A failure here must not stop the search."""
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
        """Load an earlier session of this run (``snap`` is a ``taic.resume.Snapshot``) so that ``run()`` continues it: the cost ledger, every candidate with its
        report, the best kernel, the evaluation cache, the trace, the hypotheses already tried, the counters and the profiler evidence.
        ``dry_run`` loads everything but writes nothing (no transcript marker, no state.json)."""
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
        d = self.run_dir / "candidates" / f"{c.idx:03d}_{c.label}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.metal").write_text(c.source)
        if c.report:
            control.write_json_atomic(d / "report.json", c.report.to_json())
            (d / "feedback.txt").write_text(format_feedback(c.report))

    # a 必须比 b 快出这么多倍才算真正变快：至少 min_improvement，并且要盖过两者加速比标准误合成后再乘 noise_z。差一点只记 within_noise，不能当下一轮种子。
    def _required_gain(self, a: Candidate, b: Candidate) -> float:
        """Factor by which ``a`` must be measured faster than ``b`` before it counts as faster."""
        return required_gain(self.search.min_improvement, self.search.noise_z, speedup_se(a.report), speedup_se(b.report))

    # 一颗候选的完整评测，也是唯一更新当前种子的地方。不安全、算错、或快不过测量噪声，都留在记录里但种子不变。默认 max_hangs 为 1，所以一次 timeout 就抛 StopRun，整次搜索停；连续 3 次 crash 同样停。
    async def _eval(self, source: str, label: str, *, plan: str | None = None) -> Candidate:
        """Full evaluation of a candidate (serialized on the GPU). Updates best/trace and enforces the hang breaker."""
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
        for key, parent in self._branch_parent.items():
            if label == key or label.startswith(key + "."):
                return parent
        return None

    def _phase_seed(self, phase: str, label: str, source: str) -> int | None:
        """Derived seed for a recorded run. None keeps the evaluator's own random draw (runs with no ``run_seed``)."""
        if self.run_seed is None:
            return None
        return eval_seed(self.run_seed, phase, label, source)

    async def _differential_recheck(self, cand: Candidate, speed_relation: str) -> dict | None:
        """Correctness-only recheck on the ``recheck`` seed. Returns None when it passes.

        A returned document means the candidate is not promoted. ``timeout`` and ``crash`` set the existing hang
        counters and are marked ``counts_as_kernel_error`` false, so they are not cached as a wrong kernel.
        The document is not a ``VerifyResult`` and has no ``counterexample``."""
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
        """Metadata for a failed remeasure. The retest command is for a person to run; this method does not run it."""
        search_seed = cand.report.seed if cand.report is not None else None
        recheck_seed = phase_seed if phase_seed is not None else rep.seed
        case = None
        for item in ((rep.correctness or {}).get("cases") or []):
            if isinstance(item, dict) and item.get("ok") is False:
                case = item.get("case")
                break
        tolerance = (rep.correctness or {}).get("tolerance") or {"rtol": self.task.rtol, "atol_rel": self.task.atol_rel, "exact": self.task.exact}
        kernel_path = f"candidates/{cand.idx:03d}_{cand.label}/kernel.metal"
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
        text = json.dumps(doc, indent=1, default=str)
        (self.run_dir / "candidates" / f"{cand.idx:03d}_{cand.label}" / "differential_recheck.json").write_text(text)
        folder = self.run_dir / "rechecks"
        folder.mkdir(exist_ok=True)
        (folder / f"{cand.label}.json").write_text(text)


    # ------------------------------------------------------------------ main loop
    # 整次运行的外壳。预算耗尽、GPU 挂死或模型错误在这里收成 stop_reason。挂死时 hangs 已达上限，_confirm 会跳过；预算耗尽仍会复测已有种子，因为复测不再发模型请求。
    async def run(self) -> RunResult:
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
                self.log(f"== stop: {e}")
            except StopRun as e:
                res.stop_reason = str(e)
                self.log(f"== STOP: {e} (the run ends immediately; a runaway GPU kernel can freeze the display)")
            except LLMError as e:
                res.stop_reason = f"llm_error: {e}"
                self.log(f"== stop: {e}")
            self._search_seed = self.best
        self.ledger.refresh_budget()
        await self._confirm(res)
        if self.search.profile_best:
            await self._profile_best()
        await self._close_clients()  # every HTTP client this run opened, on the loop that used it (asyncio.run closes the loop afterwards)
        return self._finish(res)

    def _control_extends(self, res: RunResult) -> bool:
        """True when a control-file change arrived after the search stopped and the new limits allow another round.

        Confirmation used to be the last GPU work, so a top-up written during it was never read and a finished
        non-budget stop could not be resumed. The file is read here, before that confirmation.
        """
        self.ledger.refresh_budget()
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

    # --from 只在 best.metal 存在、且 result.json 里的题和这次一致时才把源码交回来。对不上或没有文件就返回 None，调用方从头写。
    def _prior_kernel_text(self) -> str | None:
        """The previous run's ``best.metal`` when it belongs to this task, else None. The caller re-evaluates it."""
        if self.from_catalog:
            from .kernel_catalog import catalog_dir

            folder = catalog_dir(self.target_id, self.task_name, self.dtype, self.size)
            ext = self.target.file_ext or ".metal"
            if not str(ext).startswith("."):
                ext = "." + str(ext)
            found = sorted(folder.glob("best.*"))
            metal = folder / f"best{ext}"
            if not metal.is_file() and found:
                metal = found[0]
            if not metal.is_file():
                self.log(f"-- --from @catalog: no kernel for {self.target_id} {self.task_name} {self.dtype} {self.size}")
                return None
            return metal.read_text(encoding="utf-8")
        root = self.from_run
        if root is None:
            return None
        if not self._same_target(root):
            self.log(f"-- --from {root} is not target {self.target_id}; writing the first version from scratch")
            return None
        metal = root / "best.metal"
        if not metal.is_file():
            self.log(f"-- --from {root}: no best.metal; writing the first version from scratch")
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
            self.log(f"-- --from {root}: no result.json; re-evaluating best.metal on {self.task_name} {self.dtype} {self.size}")
        return metal.read_text()

    # 初始生成，然后进入逐轮「诊断 → 规划 → K 路补丁」。只有 _eval 认定安全、算对且快过噪声的候选才会变成下一轮种子。
    async def _search(self, res: RunResult) -> None:
        # 1. initial implementation, unless the user pointed at a previous kernel that is still eligible
        self._round = 0
        self._publish_state("llm")
        if self.from_run is not None:
            source = self._prior_kernel_text()
            if source is not None:
                self.log(f"-- re-evaluating {self.from_run / 'best.metal'} on {self.task_name} {self.dtype} {self.size}")
                await self._eval(source, "prior")
                if self.best:
                    self.log(f"-- seed is that kernel at {self.best.speedup:.3f}x (measured now); skipping the first draft")
                else:
                    self.log(f"-- {self.from_run / 'best.metal'} is no longer eligible; writing the first version from scratch")
        # 初始生成最多重开 INITIAL_RESTARTS 次。一次都没有合格种子就停，原因是 no_valid_initial_kernel，后面的轮次不会开始。
        if not self.best:
            await self._initial_generation()
        if not self.best:
            res.stop_reason = "no_valid_initial_kernel"
            return
        # 2..5. rounds
        await self._rounds(res, 1, 0)

    def _copied_report(self, source: str) -> EvalReport | None:
        """A mechanical refusal for a kernel that was pasted back, or None when the source has to be measured."""
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
        if self._from_text and source == self._from_text:
            return None
        if source in self._external_kernel_texts():
            return EvalReport(
                task=self.task_name, dtype=self.dtype, size=self.size, mode="full",
                status="external_duplicate", stage="screen",
                message="identical to a kernel from another run. It does not count as a result of this search.",
            )
        return None

    def _external_kernel_texts(self) -> set[str]:
        """Kernel texts from other runs, read once. An empty root list (tests) reads nothing."""
        if self._external_texts is not None:
            return self._external_texts
        roots = self._external_roots
        if roots is None:
            runs = REPO_ROOT / "runs"
            roots = [runs] if runs.is_dir() else []
        texts: set[str] = set()
        from_path = (self.from_run / "best.metal") if self.from_run is not None else None
        if from_path is not None and from_path.is_file():
            self._from_text = from_path.read_text()
        for root in roots:
            if not root.is_dir():
                continue
            for path in list(root.glob("*/candidates/*/kernel.metal")) + list(root.glob("*/*/candidates/*/kernel.metal")):
                if self.run_dir in path.parents or not self._same_target(path.parents[2]):
                    continue
                try:
                    texts.add(path.read_text())
                except OSError:
                    continue
            for path in list(root.glob("*/best.metal")) + list(root.glob("*/*/best.metal")):
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
        """Flame chase is one session a round. The old parallel shape is refused before any session starts."""
        if self.search.kernel_agent != "pi":
            raise RunStartError("kernel writing goes through hmz; kernel_agent must be pi")
        if self.search.branches != 1 or self.search.init_samples != 1:
            raise RunStartError("this search is one session a round: branches and init_samples must be 1")

    def _ensure_gpu_lock_file(self) -> None:
        """Create the machine lock so a read-only session can flock it. The session is not granted write access."""
        from tcenv.gpulock import lock_file

        path = lock_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    async def _one_session(self, tag: str, *, seedless: bool) -> str:
        """One hmz session. Returns ``agent_failed``, ``unchanged`` or ``evaluated``.

        A session that ends without a new kernel after a crash or a timeout is ``agent_failed``.
        A clean exit that does not write the submission file is ``agent_failed``.
        A clean exit that writes the seed bytes into that file is ``unchanged`` and does not count as a failure.
        A changed file is evaluated whatever the process exit was.
        Feedback is parsed once and recorded once, including when the session stops early.
        """
        from .agent_feedback import parse_feedback
        from .kernel_agent import AgentError, Prices, run_kernel_agent, output_tokens_for

        self.ledger.check()
        remaining = self.ledger.remaining
        if remaining is not None and remaining < 0.05:
            raise BudgetExhausted(f"budget ${self.ledger.budget_usd:.2f} exhausted (spent ${self.ledger.total_usd:.4f})")
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
        """Drop the previous session's notes and feedback before this one starts. The directory is reused on resume."""
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
        from .agent_feedback import max_seq

        if self._feedback_seq is None:
            self._feedback_seq = max_seq(self.run_dir / "feedback.jsonl")
        self._feedback_seq += 1
        return self._feedback_seq

    def _record_feedback(self, tag: str, member, work: Path, parsed: dict, outcome: str | None, cand, exit_kind: str, seed_label: str | None, *, started: float | None = None) -> None:
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
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
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
        """Review a single-source tip with the other committee before the next session.

        The review is run-scoped, not a repository promotion. Its cost is charged to the
        same ledger; the reviewer's budget is derived from its prices, not a fixed amount.
        """
        from . import harness
        from . import review as review_mod
        from .kernel_agent import Prices

        pending = harness.pending_tip_reviews(self.run_dir, proposals)
        if not pending:
            return
        other = next((committee for committee in self.committees if committee.name != self._committee_name), None)
        if other is None or not other.members:
            return
        reviewer = other.members[0]
        row = pending[0]
        proposer = (str(row.get("provider") or ""), str(row.get("model") or ""))
        if proposer == (reviewer.provider, reviewer.model):
            self.log("  [review] skipped: the other committee uses the same provider/model")
            return
        prompt = review_mod.build_review_prompt(row.get("proposal") or {}, run_id=str(row.get("run_id") or self.run_dir.name))
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
            self.log(f"  [review] skipped: {exc}")

    def _restore_handoff(self) -> None:
        """A resumed run otherwise forgets the previous session's NOTES and failure text."""
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
        spent = result.spent or {}
        event = {
            "event": "agent", "label": tag, "harness": "pi", "provider": member.provider, "model": member.model,
            "committee": self._committee_name, "cost_usd": result.cost_usd, "cost_incomplete": bool(result.cost_incomplete),
            "had_kernel": bool(result.source), "outcome": outcome, "error": (result.error or "")[:180],
            "self_checks": result.self_checks, "self_check_timeouts": result.self_check_timeouts,
            "self_check_gpu_s": result.self_check_gpu_s,
            "tokens": {k: int(v) for k, v in spent.items() if isinstance(v, (int, float))},
        }
        event.update(self._feedback_slim(parsed))
        return event

    def _file_session_notes(self, cand: Candidate, result) -> None:
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
        rep = cand.report
        if rep is None:
            return "没有报告"
        us = f"{rep.candidate_us:.0f} µs" if isinstance(rep.candidate_us, (int, float)) else "没有微秒数"
        replaced = "换了种子" if cand.is_best else "没换种子"
        return f"{us}，状态 {rep.status}，{replaced}"

    def _session_brief(self, seedless: bool, label: str = "init", work_dir: Path | None = None):
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
        )

    async def _initial_generation(self) -> None:
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
        """Continue a run that was restored from its directory (``restore``): the rounds that were not finished, then the usual ending."""
        snap = self._resume
        res.rounds = snap.rounds_done
        if not self.best:
            await self._initial_generation()
        if not self.best:
            res.stop_reason = "no_valid_initial_kernel"
            return
        await self._rounds(res, snap.next_round if snap.partial is None else snap.partial.rnd, snap.stagnant, None)

    # 续跑后的候选是新建对象。按标签从后往前找回同一颗内核，身份比较才会和中断前对得上。
    def _by_label(self, label: str | None) -> Candidate | None:
        for c in reversed(self.candidates):
            if c.label == label:
                return c
        return None

    # 每一轮：可选的 Instruments 证据、诊断规划、再 K 路补丁。分支同时启动，所以模型请求重叠；每路进入 _eval 时在 gpu_lock 上排队，GPU 评测串行。只有换过种子才不算停滞。
    async def _rounds(self, res: RunResult, first_round: int, stagnant: int, partial: Any = None) -> None:
        """One new hmz session per round, alternating committees. ``partial`` is ignored: a resumed round starts a fresh session.

        The limit is read again every round: a control file can raise it while a round is in flight."""
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
            if self.search.hw_feedback:
                from .hwfeedback import hw_feedback

                block = await asyncio.to_thread(hw_feedback, self.task, seed.source, seed.report, log=self.log)
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
        """The best, plus up to ``confirm_top - 1`` other eligible kernels that it does not beat by more than the noise."""
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
        """Correctness-only full-domain check of the confirmed winner. A failure leaves the published number unchanged."""
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
        """Re-measure the winner and its statistical ties. Every batch must succeed.

        Candidates in one batch share one input seed. The published kernel is the search best unless
        another complete remeasure clears the noise gate. A later failed batch does not turn the earlier
        successes into a confirmation."""
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
                            eval_cache=self.search.eval_cache,
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
        """Instruments trace of the round's seed when it is slower than the strongest baseline (``search.profile_poor``).

        Returns the text appended to the feedback of the planner and of every patch author, or "" (disabled, fast enough, over the trace budget, or the trace
        failed: profiling is a bonus and never stops the search). The same source is traced once. Traces run under the exclusive GPU lock."""
        s = self.search
        # 没开、已经够快、或搜索已熔断：不占 GPU。规划照常进行，只是没有这份证据。
        if not s.profile_poor or seed.speedup >= s.profile_poor_below or self.hangs >= s.max_hangs:
            return ""
        key = _sha(seed.source)
        if key in self._profile_text:
            return self._profile_text[key]
        if len(self.profiles) >= s.profile_max:
            return ""
        from tcenv import xctrace

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
        """Optional (search.profile_best): Instruments trace of the winner, stored next to the result. Never inside the search loop."""
        if not self.best or self.hangs >= self.search.max_hangs:
            return
        from tcenv import xctrace

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
        """Copy a kernel into the catalog only after the publication gate binds it."""
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
            (self.run_dir / "best.metal").write_text(self.best.source)
        res.target_id = self.target_id
        res.backend = self.target.backend
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
