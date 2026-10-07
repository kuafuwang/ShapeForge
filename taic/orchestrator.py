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
# 结束时 _confirm 用新种子重测优胜者和噪声范围内的并列者。中断后续跑走 restore / _search_resumed，不把已评过的内核再评一遍。

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import secrets
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from tasks import get_task
from tcenv.evaluate import EvalReport, evaluate
from tcenv.report import format_feedback

from . import control, prompts
from .feedback_extra import abi_signature, at_roofline, failure_phrase, status_lines, target_note, threadgroup_limit_note
from .llm import LLM, BudgetExhausted, CostLedger, LLMConfig, LLMError, SearchConfig, Transcript
from .parse import extract_metal, parse_plans
from .tools import LOOKUP_TOOL, QUICK_CHECK_TOOL, SkillIndex, make_handlers

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


_EVAL_PHASES = ("search", "confirm", "recheck")


def eval_seed(run_seed: int, phase: str, label: str, source: str) -> int:
    """48-bit seed for one evaluation, derived from the run seed and the candidate's identity.

    ``source`` is the kernel text before the contract header is prepended. ``phase`` is ``search``, ``confirm`` or ``recheck``.
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

# 规划回复抽不出方案、或规划请求本身失败时，仍用这三套通用改法把这一轮补丁做完，搜索不因此停。
FALLBACK_PLANS = [
    {"title": "vectorized 16-byte memory access", "change": "Make every thread load/store 16 bytes at a time (uint4 / half4 pairs) over consecutive addresses and unroll so that several loads are in flight before use.", "expected_gain": ""},
    {"title": "single pass over global memory with SIMD-group reductions", "change": "Keep data in registers between phases, reduce with simd_sum/simd_max first and exchange only one value per SIMD group through threadgroup memory.", "expected_gain": ""},
    {"title": "use simdgroup_matrix for matrix products", "change": "If the kernel contains a matrix product, compute it with simdgroup_matrix 8x8 tiles (half inputs, float accumulators) with register blocking; otherwise restructure the main loop to reduce instruction count.", "expected_gain": ""},
]


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

    # 只做正确性恢复时为真：合格不看加速比，结束时也不做 _confirm。
    correctness_only: bool = False

    # 安全且数值正确（EvalReport.status 为 ok）才算合格。正式搜索还要求报告里真有 speedup；缺计时的候选不能换种子。
    @property
    def eligible(self) -> bool:
        if self.correctness_only:
            return self.report is not None and self.report.eligible
        return self.report is not None and self.report.eligible and self.report.speedup is not None

    # 不合格时加速比按 0，缺失的计时不可能把当前种子挤掉。
    @property
    def speedup(self) -> float:
        if not self.eligible:
            return 0.0
        return self.report.speedup if self.report.speedup is not None else 1.0

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


# 源码的内容身份：评测缓存、复测去重、profiler 缓存都用它。同一份文本不会被当成两个内核。
def _sha(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()


# 回复里抽不出带约定内核名的 Metal 块时追问的那一句。抽失败只消耗这一问，不会产生候选，也不会上 GPU。
def nudge_user(kernel_name: str) -> str:
    """The reminder sent when a reply holds no ```metal block defining the kernel."""
    return f"Your reply did not contain a ```metal code block defining `kernel void {kernel_name}`. Reply again with the complete kernel in exactly one ```metal block."


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


# 加速比是两次中位数之比，这里给出它的相对标准误。报告里没有逐次计时（cv、reps）时标准误视为 0，换种子只剩固定的 min_improvement 门槛。
def speedup_se(rep: EvalReport | None) -> float:
    """Relative standard error of a measured speedup (ratio of two medians), from the per-repetition cvs.
    0 when the report has no timing statistics."""
    b = (rep.benchmark if rep else None) or {}
    try:
        c, base = b["candidate"], b["baselines"][b["best_baseline"]]
        n = max(1, min(c["reps"], base["reps"]))
        # 中位数的标准误约为 sqrt(pi/2)·σ/sqrt(n)；候选和基线的误差按独立合成。
        return math.sqrt(math.pi / 2) * math.hypot(c["cv"], base["cv"]) / math.sqrt(n)  # SE of a median ~ sqrt(pi/2) sigma / sqrt(n)
    except (KeyError, TypeError):
        return 0.0


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
    ) -> None:
        """``resuming``: the directory already holds a run (see ``taic.resume``); nothing in it is rewritten until ``restore()`` has loaded it."""
        self.task = get_task(task_name, dtype, size)
        self.task_name, self.dtype, self.size = task_name, dtype, size
        self.cfg, self.search = llm_cfg, search
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir) if run_dir else REPO_ROOT / "runs" / f"{stamp}_{task_name}_{dtype}_{llm_cfg.model}"
        self.log = log
        self.system = prompts.system_prompt(self.task, use_tools=llm_cfg.use_tools, profiler_evidence=search.profile_poor, hw_feedback=search.hw_feedback)
        self.from_run = Path(from_run) if from_run else None
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
        self.ledger = CostLedger(llm_cfg, search.budget_usd)
        # 已存在的 control.json 先被忽略，直到文件再变：那是上一会话的请求，启动者已经折进当前预算。损坏的文件只记一笔，不改上限。
        self.ledger.watch_control(self.run_dir / control.CONTROL_FILE)
        self.ledger.on_budget_change = self._budget_changed
        self.ledger.on_control = self._control_limits
        self.ledger.on_control_problem = lambda msg: self.log(f"  {msg}")
        # 并发度至少为分支数 K（search.branches）。模型请求因此可以同时在飞；占 GPU 的评测不走这个信号量。
        self.llm = LLM(llm_cfg, self.ledger, Transcript(self.run_dir / "transcript.jsonl"), client=client, max_concurrency=max(2, search.branches))
        self.evaluate_fn = evaluate_fn or evaluate
        self.index = SkillIndex("msl_reference")
        # 同一时刻只允许一次 GPU 评测、quick_check 或 Instruments。两份内核抢同一块卡会把计时测脏，挂死时也不能再有内核在跑。
        self.gpu_lock = asyncio.Lock()
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
        self._search_seed: Candidate | None = None
        self._confirm_winner: Candidate | None = None
        # 续跑时目录里已有上一会话的 state.json。先空写会把看板抹掉，所以等到 restore 把账本和候选装回来再发布。
        if not resuming:
            self._publish_state()

    # 续跑要把上一会话已经花掉的墙钟算进去，否则轨迹上的时间从 0 重新开始，和账本对不上。
    def _elapsed(self) -> float:
        """Search time of the whole run, including earlier sessions of a resumed one."""
        return self._t_offset + (time.time() - self.t0)

    # 给 `taic resume` 留的设置。故意不写 API key 和端点。写失败向上抛，调用方不得进入 run()。
    def _write_run_config(self) -> None:
        """The settings of this run, for ``taic.cli resume`` (no credentials: the API key and the endpoint stay out of it).

        Raises ``RunStartError`` when the file cannot be written. The directory created for this attempt may stay empty;
        an older run directory is never deleted or overwritten (that case is refused before this method)."""
        llm = {k: v for k, v in asdict(self.cfg).items() if k not in ("api_key", "base_url")}
        contract_text = self.task.contract.describe()
        triton_text = self.task.triton_source
        commit, dirty = _git_identity(REPO_ROOT)
        try:
            control.write_json_atomic(self.run_dir / control.CONFIG_FILE, {
                "schema_version": 1,
                "task": self.task_name, "dtype": self.dtype, "size": self.size, "seed": self.seed, "run_seed": self.run_seed,
                "llm": llm, "search": asdict(self.search),
                "system_prompt_sha1": hashlib.sha1(self.system.encode()).hexdigest(),
                "contract_description_sha256": hashlib.sha256(contract_text.encode("utf-8")).hexdigest(),
                "triton_source_sha256": hashlib.sha256(triton_text.encode("utf-8")).hexdigest(),
                "hash_scope": "sha256 of contract.describe() and of task.triton_source only; reference and baseline implementations are not included",
                "git_commit": commit,
                "git_dirty": dirty,
                "settings_source": "llm and search values passed into Orchestrator at start (config file, preset and command-line overrides already applied)",
                "cost_may_be_incomplete": True,
                "cost_note": "A process can die after the provider bills a request and before transcript.jsonl records it, so a resumed ledger can under-count the real bill. A budget_events.json line that fails to land is dropped; that file is not a complete bill. This is not a hard dollar cap. Reconciling with the provider bill needs separate authorization.",
                "created_at": time.time(),
            })
        except OSError as e:
            raise RunStartError(f"cannot start: run_config.json was not written ({e})") from e

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
            from observe.publish import publish

            publish(self.run_dir / "state.json", {
                "pid": os.getpid(),
                "updated_at": time.time(),
                "started_at": self._started_at,
                "phase": self._phase,
                "task": self.task_name,
                "dtype": self.dtype,
                "size": self.size,
                "model": self.cfg.model,
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
                "hangs": self.hangs,
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
                             is_best=False, within_noise=bool(t.get("within_noise")), correctness_only=self.search.correctness_only)
            self.candidates.append(cand)
            # 与 _eval 相同：超时和 worker 崩溃是环境故障，不能当成这颗内核的永久判决。
            if cand.report is not None and cand.report.status not in ("timeout", "crash"):
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
        self._round = snap.partial.rnd if snap.partial else snap.rounds_done
        self._resume = snap
        # 试跑只在内存里装好状态。写转录或 state.json 会让一次「看看能不能续」变成真的动了运行目录。
        if dry_run:
            return
        self.llm.transcript.log({"event": "resume", "n": self._resumes, "spent_usd": self.ledger.total_usd, "budget_usd": self.ledger.budget_usd, "round": self._round, "best": snap.best_label, "pid": os.getpid()})
        try:  # requests that were in flight when the earlier session died are not in flight any more
            from observe.publish import publish

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
            (d / "report.json").write_text(json.dumps(c.report.to_json(), indent=1))
            (d / "feedback.txt").write_text(format_feedback(c.report))

    # a 必须比 b 快出这么多倍才算真正变快：至少 min_improvement，并且要盖过两者加速比标准误合成后再乘 noise_z。差一点只记 within_noise，不能当下一轮种子。
    def _required_gain(self, a: Candidate, b: Candidate) -> float:
        """Factor by which ``a`` must be measured faster than ``b`` before it counts as faster."""
        return max(self.search.min_improvement, 1.0 + self.search.noise_z * math.hypot(speedup_se(a.report), speedup_se(b.report)))

    # 一颗候选的完整评测，也是唯一更新当前种子的地方。不安全、算错、或快不过测量噪声，都留在记录里但种子不变。默认 max_hangs 为 1，所以一次 timeout 就抛 StopRun，整次搜索停；连续 3 次 crash 同样停。
    async def _eval(self, source: str, label: str, *, plan: str | None = None) -> Candidate:
        """Full evaluation of a candidate (serialized on the GPU). Updates best/trace and enforces the hang breaker."""
        cand = Candidate(idx=len(self.candidates), label=label, source=source, plan=plan, correctness_only=self.search.correctness_only)
        self.candidates.append(cand)
        self._publish_state("evaluate", inflight=label)
        key = _sha(source)
        phase_seed = self._phase_seed("search", label, source)
        # 同一源码的稳定结论可复用。复用的是当时那份报告的 seed，不能把它说成一次新种子上的测量。
        if key in self.cache:
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
        # 换种子的三道闸：EvalReport.eligible（status 为 ok，即安全且算对）、正式搜索还要有加速比，并且快过 _required_gain。只是略快则标 within_noise，下一轮仍用旧种子。
        # selection: eligible AND faster (relative to the baseline measured in the same run) than the best so far,
        # by more than both the fixed margin and the measurement noise of the two speedups
        if cand.eligible and (self.best is None or cand.speedup > self.best.speedup * self._required_gain(cand, self.best)):
            if self.best:
                self.best.is_best = False
            cand.is_best = True
            self.best = cand
        elif cand.eligible and cand.speedup > self.best.speedup:
            cand.within_noise = True
        self._save(cand)
        r = cand.report
        self.trace.append(
            {"idx": cand.idx, "label": label, "status": r.status, "speedup": r.speedup, "cost_usd": cand.cost_usd, "t_s": cand.t_s, "best_speedup": self.best.speedup if self.best else None, "improved": cand.is_best, "within_noise": cand.within_noise, "eval_seed": used_seed, "eval_seed_from": seed_from}
        )
        self._publish_state(inflight=None)
        sp = f"{r.speedup:.3f}x" if r.speedup else "-"
        tag = "NEW BEST " if cand.is_best else ("within noise of the best " if cand.within_noise else "")
        self.log(f"  [{label}] {r.status:<13s} speedup {sp:>8s}  {tag}(${cand.cost_usd:.3f}, {cand.t_s:.0f}s)  {('' if r.status == 'ok' else r.message[:110].replace(chr(10), ' '))}")
        # 一次超时或连续三次 worker 崩溃：停止整次搜索，不再开新的模型回合。
        if self.hangs >= self.search.max_hangs or self.crashes_in_a_row >= 3:
            raise StopRun("gpu_hang_circuit_breaker" if self.hangs >= self.search.max_hangs else "repeated_worker_crashes")
        return cand

    def _phase_seed(self, phase: str, label: str, source: str) -> int | None:
        """Derived seed for a recorded run. None keeps the evaluator's own random draw (runs with no ``run_seed``)."""
        if self.run_seed is None:
            return None
        return eval_seed(self.run_seed, phase, label, source)

    # quick_check 用的轻量评测：编译、ABI、安全、正确性，不跑基准，所以模型不能靠它在噪声里挑「看起来更快」的版本。与完整评测共用 gpu_lock 和 hangs；这里的 timeout 只记数，停搜发生在下一次 _eval 进锁时。
    # 工具调用没有稳定的 call id，所以这里仍把 seed 留空，由评测器另抽，并记在它返回的报告里。不能说这次自查可以按 run_seed 重放。
    async def _quick(self, source: str) -> str:
        """quick_check tool: compile + ABI + safety + correctness, no benchmark. Serialized on the GPU like everything else."""
        async with self.gpu_lock:
            # 挂死熔断已经触发就拒绝再上 GPU，避免工具回合在搜索该停之后还启动内核。
            if self.hangs >= self.search.max_hangs:
                return "error: evaluation environment disabled after a GPU hang"
            self._publish_state("evaluate", inflight="quick_check")
            rep = await asyncio.to_thread(
                self.evaluate_fn, self.task_name, self.dtype, self.size, source, mode="quick", seed=None, reps=3,
                progress_path=self.run_dir / "eval_progress.json",
            )
            # 记入与 _eval 相同的 hangs。本函数不抛 StopRun，这段反馈仍会交回模型。
            if rep.status == "timeout":
                self.hangs += 1
        return format_feedback(rep)[:FEEDBACK_CHARS]

    # ------------------------------------------------------------------ generation with repair
    # 一个模型回合。美元检查在 LLM.chat 入口的 CostLedger.check：已花费达到预算就抛 BudgetExhausted，本回合发不出去。已经在飞的其它分支可能略超，闸门不预扣。
    async def _ask(self, messages: list[dict], label: str, *, start_turn: int = 0, pending: list[dict] | None = None, quick_used: int = 0) -> tuple[str, list[dict], str | None]:
        """One model turn (with tools when enabled). Returns (reply text, conversation including the reply, finish_reason).
        ``start_turn`` / ``pending`` / ``quick_used`` continue a tool conversation interrupted by a restart (see ``chat_with_tools``)."""
        self._publish_state("llm")
        if self.cfg.use_tools:
            # 续跑时 quick_used 是中断前已经用掉的次数，这里只把剩余额度交给这一段对话，避免熔断被绕过。
            handlers = make_handlers(self.index, self._quick, max(0, self.search.max_quick_checks - quick_used))
            reply, msgs = await self.llm.chat_with_tools(messages, [LOOKUP_TOOL, QUICK_CHECK_TOOL], handlers, label=label, start_turn=start_turn, pending=pending)
            return reply.text(), msgs, reply.finish_reason
        reply = await self.llm.chat(messages, label=label)
        return reply.text(), messages + [reply.message], reply.finish_reason

    # 从一轮提示写出内核，并最多再修补 max_repair_tries 次。抽不出 Metal 只追问、不产生候选；评测不合格就把失败反馈塞回。完整评测一旦 timeout，_eval 会先停掉整次搜索；若 max_hangs 尚未触达，这颗内核不再继续修补。
    async def _develop(self, messages: list[dict], label: str, *, plan: str | None = None, resume: Any = None) -> Candidate | None:
        """Generate a kernel from ``messages`` and repair it for up to ``max_repair_tries`` turns.
        Returns the first eligible candidate (or None). Failed attempts are still recorded as candidates.

        ``resume`` (a ``taic.resume.BranchResume``) continues a conversation that a restart interrupted: ``resume.messages`` is the conversation as it stood,
        ``resume.attempt`` the repair attempt it was in, and ``resume.reply_text`` / ``resume.pending`` say how far the current attempt had got."""
        name = self.task.contract.kernel_name
        last_abi = None
        for attempt in range(resume.attempt if resume is not None else 0, self.search.max_repair_tries + 1):
            tag = label if attempt == 0 else f"{label}.fix{attempt}"
            retried = False
            while True:
                if resume is not None and attempt == resume.attempt and not retried:
                    r, resume = resume, None
                    # 模型已经答过、但答案还没评测：沿用记下的回复，避免再花一次请求得到另一颗内核。
                    if r.reply_text is not None:  # the model had answered; the answer was not evaluated yet
                        text, messages, finish = r.reply_text, r.messages, None
                    else:
                        text, messages, finish = await self._ask(r.messages, tag, start_turn=r.turn, pending=r.pending or None, quick_used=r.quick_used)
                else:
                    text, messages, finish = await self._ask(messages, tag)
                source = extract_metal(text, name)
                # 截断且没有内核：不占用这一次修补。丢掉回传的思考再问一次，问完无论结果如何都回到原来的流程。
                if source is None and finish == "length" and self.cfg.truncation_retry and not retried:
                    messages = _retry_without_reasoning(messages)
                    retried = True
                    self.log(f"  [{tag}] truncated with no kernel; retrying once without the echoed reasoning")
                    continue
                break
            # 没有声明 kernel void <kernel_name> 的代码块不算写出内核。只追问，不进入评测。
            if source is None:
                self._branch_fail[label] = "no code: truncated" if finish == "length" else "no code: no kernel block"
                messages = messages + [{"role": "user", "content": nudge_user(name)}]
                self.log(f"  [{tag}] no code block in reply")
                continue
            cand = await self._eval(source, tag, plan=plan)
            if cand.eligible:
                return cand
            # 超时不再修补：挂死的内核继续改只会再占 GPU。默认 max_hangs 下 _eval 已经抛出 StopRun，走不到这里。
            if cand.report.status in ("timeout",):
                return None
            self._branch_fail[label] = failure_phrase(cand.report.status, cand.report.message)
            # 同一句 ABI 错误连着出现两次就停掉这一支，不再把剩下的修补次数花在同一个超限上。
            if self.search.abi_guard and cand.report.status == "abi_error":
                sig = abi_signature(cand.report.message)
                if last_abi is not None and sig == last_abi:
                    self.log(f"  [{tag}] the same ABI failure repeated; this branch stops")
                    return None
                last_abi = sig
            else:
                last_abi = None
            # 反馈被截到 FEEDBACK_CHARS。模型看不见后半段，修补可能漏掉排在后面的违规。
            fb = format_feedback(cand.report)[:FEEDBACK_CHARS]
            if self.search.abi_guard:
                note = threadgroup_limit_note(cand.report.message)
                if note:
                    fb += "\n\n" + note
            messages = messages + [{"role": "user", "content": prompts.repair_user(fb, attempt + 1, self.search.max_repair_tries)}]
        return None

    # ------------------------------------------------------------------ planning
    # 诊断当前种子的瓶颈并给出 K（search.branches）份互不相同的改法。规划请求失败或 JSON 抽不出方案时，用 FALLBACK_PLANS 顶上，这一轮搜索不因此结束。
    async def _plan(self, seed: Candidate, rnd: int, evidence: str = "", status: str = "") -> list[dict]:
        self._publish_state("plan")
        k = self.search.branches
        fb = format_feedback(seed.report)[:FEEDBACK_CHARS] + evidence
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": prompts.plan_user(self.task, seed.source, fb, self.history, k, mixed=self.search.plan_mix, status=status)}]
        for attempt in range(2):
            try:
                reply = await self.llm.chat(msgs, label=f"r{rnd}.plan{'' if attempt == 0 else '.retry'}")
            except LLMError as e:
                # 规划本身失败不杀搜索，退回通用方案，这一轮补丁仍会做。
                self.log(f"  planner request failed ({str(e)[:160]}); using generic plans")
                return FALLBACK_PLANS[:k]
            analysis, plans = parse_plans(reply.text(), k)
            # plans 为空即抽失败：再要一次纯 JSON。两次都不行就用 FALLBACK_PLANS。
            if plans:
                self.log(f"  plan r{rnd}: {analysis[:160]}")
                for p in plans:
                    self.log(f"    - {p['title']}")
                return plans
            msgs = msgs + [reply.message, {"role": "user", "content": 'Reply with the JSON object only, in a ```json block, with a non-empty "plans" list.'}]
        self.log("  planner returned no parsable plan; using generic plans")
        return FALLBACK_PLANS[:k]

    # ------------------------------------------------------------------ main loop
    # 整次运行的外壳。预算耗尽、GPU 挂死或模型错误在这里收成 stop_reason。挂死时 hangs 已达上限，_confirm 会跳过；预算耗尽仍会复测已有种子，因为复测不再发模型请求。
    async def run(self) -> RunResult:
        res = RunResult(run_dir=str(self.run_dir), task=self.task_name, dtype=self.dtype, size=self.size, model=self.cfg.model, budget_usd=self.ledger.budget_usd)
        resumed = self._resume is not None
        self.log(f"== {'RESUME ' if resumed else ''}{self.task_name} {self.dtype} {self.size} | model {self.cfg.model} | budget ${self.ledger.budget_usd:.2f} | branches {self.search.branches} | run {self.run_dir.name}")
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
        await self._confirm(res)
        if self.search.profile_best:
            await self._profile_best()
        await self.llm.aclose()  # the HTTP client must be closed on the loop that used it (asyncio.run closes the loop afterwards)
        return self._finish(res)

    # --from 只在 best.metal 存在、且 result.json 里的题和这次一致时才把源码交回来。对不上或没有文件就返回 None，调用方从头写。
    def _prior_kernel_text(self) -> str | None:
        """The previous run's ``best.metal`` when it belongs to this task, else None. The caller re-evaluates it."""
        root = self.from_run
        if root is None:
            return None
        metal = root / "best.metal"
        if not metal.is_file():
            self.log(f"-- --from {root}: no best.metal; writing the first version from scratch")
            return None
        recorded = root / "result.json"
        if recorded.is_file():
            try:
                doc = json.loads(recorded.read_text())
            except json.JSONDecodeError:
                self.log(f"-- --from {root}: result.json is not JSON; writing the first version from scratch")
                return None
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

    async def _initial_generation(self) -> None:
        for restart in range(INITIAL_RESTARTS):
            base = [{"role": "system", "content": self.system}, {"role": "user", "content": prompts.initial_user(self.task)}]
            n = max(1, self.search.init_samples)
            self.log(f"-- initial generation (attempt {restart + 1}/{INITIAL_RESTARTS}" + (f", {n} samples in parallel" if n > 1 else "") + ")")
            label = "init" if restart == 0 else f"init{restart + 1}"
            if n == 1:
                await self._develop(base, label)
            else:
                # init_samples 路初始样本的模型请求重叠。其中一路预算耗尽或 StopRun 会结束全部样本；单路 LLMError 只丢掉该样本，全部失败才结束运行。
                outcomes = await asyncio.gather(*[self._develop(base, label if i == 0 else f"{label}.s{i + 1}") for i in range(n)], return_exceptions=True)
                fatal = [o for o in outcomes if isinstance(o, (BudgetExhausted, StopRun))]
                other = [o for o in outcomes if isinstance(o, BaseException) and not isinstance(o, (BudgetExhausted, StopRun, LLMError))]
                if other:
                    raise other[0]
                if fatal:
                    raise fatal[0]
                if all(isinstance(o, LLMError) for o in outcomes):  # one failed request skips its sample; all of them failing ends the run
                    raise outcomes[0]
                for i, o in enumerate(outcomes):
                    if isinstance(o, LLMError):
                        self.log(f"  initial sample {i + 1} skipped: {str(o)[:200]}")
            if self.best:
                break

    # 续跑不重做已经有合格种子的初始生成，只把没做完的轮次交给 _rounds。上一会话没有种子，则和全新搜索一样停在 no_valid_initial_kernel。
    async def _search_resumed(self, res: RunResult) -> None:
        """Continue a run that was restored from its directory (``restore``): the rounds that were not finished, then the usual ending."""
        snap = self._resume
        res.rounds = snap.rounds_done
        if not self.best:
            res.stop_reason = "no_valid_initial_kernel"
            return
        await self._rounds(res, snap.partial.rnd if snap.partial else snap.next_round, snap.stagnant, snap.partial)

    # 续跑后的候选是新建对象。按标签从后往前找回同一颗内核，身份比较才会和中断前对得上。
    def _by_label(self, label: str | None) -> Candidate | None:
        for c in reversed(self.candidates):
            if c.label == label:
                return c
        return None

    # 每一轮：可选的 Instruments 证据、诊断规划、再 K 路补丁。分支同时启动，所以模型请求重叠；每路进入 _eval 时在 gpu_lock 上排队，GPU 评测串行。只有换过种子才不算停滞。
    async def _rounds(self, res: RunResult, first_round: int, stagnant: int, partial: Any = None) -> None:
        """Rounds ``first_round`` to ``max_rounds``. ``partial`` (a ``taic.resume.RoundPartial``) describes the first round of a resumed run that a restart
        interrupted: its seed, its plans if they were already made, the branches that finished and the conversations of those that were part-way.
        The limit is read again every round: a control file can raise it while a round is in flight."""
        rnd = first_round
        while rnd <= self.search.max_rounds:
            res.rounds = rnd
            self._round = rnd
            self._stagnant = stagnant
            part = partial if (partial is not None and partial.rnd == rnd) else None
            # 续跑的第一轮用中断时记下的种子。若改用现在的 self.best，补丁会对着另一颗内核改。
            seed = (self._by_label(part.seed_label) if part is not None else None) or self.best
            self.log(f"-- round {rnd}{' (continued)' if part is not None else ''}: seed {seed.label} at {seed.speedup:.3f}x, spent ${self.ledger.total_usd:.3f}")
            if self.search.stop_at_roofline and at_roofline(seed.report, self._required_gain(seed, seed)):
                gap = (getattr(seed.report, "roofline", None) or {}).get("time_over_lower_bound")
                res.stop_reason = "at_roofline"
                self.log(f"-- stop: {seed.label} is {gap:.3f}x its roofline lower bound, inside the noise gate, and no baseline beats that bound")
                return
            evidence = await self._profile_seed(seed, rnd)
            if self.search.hw_feedback:  # opt-in (search.hw_feedback): the static hardware-feature block goes where the profiler evidence goes; None on any failure
                from .hwfeedback import hw_feedback

                block = await asyncio.to_thread(hw_feedback, self.task, seed.source, seed.report, log=self.log)
                if block:
                    evidence += "\n\n" + block
            if self.search.target_note:
                note = target_note(seed.report)
                if note:
                    evidence += "\n\n" + note
            status = ""
            if self.search.status_lines:
                status = status_lines(
                    rnd=rnd, max_rounds=self.search.max_rounds, patience_left=max(0, self.search.patience - stagnant),
                    spent=self.ledger.total_usd, budget=self.ledger.budget_usd, required_gain=self._required_gain(seed, seed),
                )
            # 规划已经花过请求并记在转录里时不再重做，避免同一轮出现两套方案。
            if part is not None and part.plans:
                plans = part.plans
                self.log(f"  plans of round {rnd} recovered from the transcript: " + "; ".join(p["title"] for p in plans))
            else:
                plans = await self._plan(seed, rnd, evidence, status)
            self._round_plans = plans
            fb = format_feedback(seed.report)[:FEEDBACK_CHARS] + evidence

            async def branch(i: int, plan: dict) -> Candidate | None:
                # 中断前已经评完的分支直接沿用那颗候选，不再发请求、也不再占 GPU。
                if part is not None and i in part.done:  # finished before the restart
                    return self._by_label(part.done[i])
                if part is not None and i in part.resume:  # part-way: continue the recorded conversation
                    br = part.resume[i]
                    self.log(f"  branch r{rnd}.p{i} continues at attempt {br.attempt}")
                    return await self._develop(br.messages, f"r{rnd}.p{i}", plan=plan["title"], resume=br)
                msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": prompts.patch_user(self.task, seed.source, fb, plan, status=status)}]
                return await self._develop(msgs, f"r{rnd}.p{i}", plan=plan["title"])

            before = seed
            # K 路一起启动：模型请求在锁外重叠，GPU 评测在各路的 _eval 里串行。
            outcomes = await asyncio.gather(*[branch(i, p) for i, p in enumerate(plans)], return_exceptions=True)
            fatal = [o for o in outcomes if isinstance(o, (BudgetExhausted, StopRun))]
            llm_errors = [o for o in outcomes if isinstance(o, LLMError)]
            for i, (p, o) in enumerate(zip(plans, outcomes)):
                if isinstance(o, Candidate):
                    tag = f"{o.speedup:.3f}x" + (" (new best)" if o.is_best else " (within measurement noise of the best)" if o.within_noise else " (not better)")
                elif isinstance(o, LLMError):
                    tag = "skipped (the model request failed)"
                    self.log(f"  branch '{p['title']}' skipped: {str(o)[:200]}")
                elif isinstance(o, BaseException):
                    tag = f"aborted ({type(o).__name__})"
                else:
                    tag = self._branch_fail.get(f"r{rnd}.p{i}", "no valid kernel after repair") if self.search.rich_history else "no valid kernel after repair"
                self.history.append(f"round {rnd}: {p['title']} -> {tag}")
            other = [o for o in outcomes if isinstance(o, BaseException) and not isinstance(o, (BudgetExhausted, StopRun, LLMError))]
            if other:
                raise other[0]
            # 任一路预算耗尽或评测超时，整次搜索停。单路 LLMError 只丢掉该分支；K 路全部失败才结束运行。
            if fatal:
                raise fatal[0]
            if llm_errors and len(llm_errors) == len(plans):  # one failed request skips its branch; all of them failing ends the run
                raise llm_errors[0]
            # 停滞看本轮开始时的种子对象是否仍是 self.best。轮内换成另一颗就清零；只是 within_noise 不会换对象，因此算停滞。连续 patience 轮就收敛停止。
            stagnant = 0 if self.best is not before else stagnant + 1
            if stagnant >= self.search.patience:
                res.stop_reason = f"converged ({self.search.patience} rounds without improvement)"
                return
            rnd += 1
        res.stop_reason = "max_rounds"

    # 复测池：当前优胜者，再加上最多 confirm_top-1 颗合格内核。优胜者相对它们的优势没有超过 _required_gain 时，它们算噪声范围内的并列，必须一起重测，不能只信搜索过程中的最大值。
    def _confirm_pool(self) -> list[Candidate]:
        """The best, plus up to ``confirm_top - 1`` other eligible kernels that it does not beat by more than the noise."""
        pool, seen = [self.best], {_sha(self.best.source)}
        for c in sorted(self.candidates, key=lambda c: c.speedup, reverse=True):
            if len(pool) >= self.search.confirm_top:
                break
            if not c.eligible or _sha(c.source) in seen:
                continue
            seen.add(_sha(c.source))
            # 优胜者并没有显著快过 c，c 就留在并列里。源码相同的候选只进池一次。
            if self.best.speedup <= c.speedup * self._required_gain(self.best, c):
                pool.append(c)
        return pool

    # 搜索结束后用新的随机种子（evaluate 在 seed=None 时另抽）和两倍 eval_reps 重测优胜者与并列者，取重测最快的一颗，避免「多次噪声测量里挑最大值」。correctness_only 或已经挂死则不做。重测中再遇 timeout 就停下剩余复测。
    async def _confirm(self, res: RunResult) -> None:
        """Re-measure the winner and its statistical ties with a fresh seed and twice the repetitions; the fastest
        re-measured kernel is the result (the search maximum alone is biased upward by noise)."""
        # 没有种子、已经熔断、或只做正确性实验时不复测。熔断后再上 GPU 可能再次冻住显示。
        if not self.best or self.hangs >= self.search.max_hangs or self.search.correctness_only:
            return
        pool = self._confirm_pool()
        if self.run_seed is None:
            self.log(f"-- confirming {', '.join(c.label for c in pool)} with a fresh seed")
        else:
            self.log(f"-- confirming {', '.join(c.label for c in pool)} with seeds derived for the confirm phase")
        self._confirmations = res.confirmations
        self._publish_state("confirm", inflight=", ".join(c.label for c in pool))
        confirmed: list[tuple[Candidate, EvalReport]] = []
        for c in pool:
            entry: dict[str, Any] = {"label": c.label, "search_speedup": c.speedup}
            res.confirmations.append(entry)
            self._confirmations = res.confirmations
            self._publish_state("confirm", inflight=c.label)
            try:
                async with self.gpu_lock:
                    # run_seed 为空的旧运行仍由评测器另抽。新运行用 confirm 阶段的派生种子，和搜索阶段不是同一个数。
                    phase_seed = self._phase_seed("confirm", c.label, c.source)
                    rep = await asyncio.to_thread(
                        self.evaluate_fn, self.task_name, self.dtype, self.size, c.source, mode="full", seed=phase_seed, reps=self.search.eval_reps * 2,
                        progress_path=self.run_dir / "eval_progress.json",
                    )
                entry["eval_seed"] = phase_seed if phase_seed is not None else rep.seed
                entry["eval_seed_from"] = "confirm" if phase_seed is not None else "random"
            except Exception as e:  # noqa: BLE001 - a failed confirmation must not lose the run
                entry["status"] = f"error: {type(e).__name__}: {e}"
                self._confirmations = res.confirmations
                self._publish_state("confirm", inflight=c.label)
                continue
            entry["status"], entry["confirmed_speedup"] = rep.status, rep.speedup if rep.status == "ok" else None
            self._confirmations = res.confirmations
            self._publish_state("confirm", inflight=c.label)
            self.log(f"   {c.label}: confirmed {rep.speedup:.3f}x" if rep.status == "ok" and rep.speedup else f"   {c.label}: confirmation {rep.status} {rep.message[:120]}")
            if rep.status == "ok" and rep.speedup:
                confirmed.append((c, rep))
            elif rep.status == "timeout":
                # 复测超时就不再测剩下的并列者。搜索已经结束，这里只停止继续占 GPU。
                self.hangs += 1
                break
        if not confirmed:
            res.confirmed_status = res.confirmations[0]["status"]
            return
        # 重测最快的一颗才是对外结果。它若不是搜索时的 self.best 就换上头衔；best_speedup 仍是搜索过程中的测量，复测值记在 confirmed_speedup。
        win, rep = max(confirmed, key=lambda cr: cr[1].speedup)
        self._confirm_winner = win
        if win is not self.best:
            self.log(f"   {win.label} is faster than {self.best.label} when re-measured; it becomes the result")
            self.best.is_best, win.is_best = False, True
            self.best = win
        res.confirmed_status, res.confirmed_speedup = rep.status, rep.speedup
        res.candidate_us, res.baseline_us = rep.candidate_us, rep.baseline_us
        (self.run_dir / "confirm_report.json").write_text(json.dumps(rep.to_json(), indent=1))

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
                summ = await asyncio.to_thread(xctrace.profile_kernel, self.task_name, self.dtype, self.size, seed.source)
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
                summ = await asyncio.to_thread(xctrace.profile_kernel, self.task_name, self.dtype, self.size, self.best.source)
            (self.run_dir / "xctrace_profile.txt").write_text(xctrace.format_summary(summ))
            (self.run_dir / "xctrace_profile.json").write_text(json.dumps(summ.to_json(), indent=1))
        except Exception as e:  # noqa: BLE001 - profiling is a bonus, never a failure
            (self.run_dir / "xctrace_profile.txt").write_text(f"profiling failed: {type(e).__name__}: {e}")

    # 把优胜者源码和账本写成 result.json。有 confirmed_speedup 时对外强调复测值，否则用搜索时的 best_speedup。中途加过的预算在 budget_events 里，结束时的上限可以高于开跑时的 initial_budget_usd。
    def _finish(self, res: RunResult) -> RunResult:
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
        (self.run_dir / "result.json").write_text(json.dumps(asdict(res), indent=1, default=str))
        (self.run_dir / "ledger.json").write_text(json.dumps({"summary": self.ledger.summary(), "requests": self.ledger.requests}, indent=1))
        self._confirmations = res.confirmations
        self._publish_state("done", stop_reason=res.stop_reason)
        sp = f"{res.confirmed_speedup:.3f}x confirmed" if res.confirmed_status == "ok" and res.confirmation_source_sha256 else (f"{res.best_speedup:.3f}x (not confirmed)" if res.best_speedup else "no valid kernel")
        self.log(f"== done: {sp} | spent ${res.spent_usd:.3f} | {res.evaluations} evaluations | stop: {res.stop_reason} | {self.run_dir}")
        return res
