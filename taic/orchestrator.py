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

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from tasks import get_task
from tcenv.evaluate import EvalReport, evaluate
from tcenv.report import format_feedback

from . import prompts
from .llm import LLM, BudgetExhausted, CostLedger, LLMConfig, LLMError, SearchConfig, Transcript
from .parse import extract_metal, parse_plans
from .tools import LOOKUP_TOOL, QUICK_CHECK_TOOL, SkillIndex, make_handlers

REPO_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_CHARS = 6000
INITIAL_RESTARTS = 2

FALLBACK_PLANS = [
    {"title": "vectorized 16-byte memory access", "change": "Make every thread load/store 16 bytes at a time (uint4 / half4 pairs) over consecutive addresses and unroll so that several loads are in flight before use.", "expected_gain": ""},
    {"title": "single pass over global memory with SIMD-group reductions", "change": "Keep data in registers between phases, reduce with simd_sum/simd_max first and exchange only one value per SIMD group through threadgroup memory.", "expected_gain": ""},
    {"title": "use simdgroup_matrix for matrix products", "change": "If the kernel contains a matrix product, compute it with simdgroup_matrix 8x8 tiles (half inputs, float accumulators) with register blocking; otherwise restructure the main loop to reduce instruction count.", "expected_gain": ""},
]


class StopRun(Exception):
    """Raised when the run must end immediately (GPU hang circuit breaker)."""


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
    within_noise: bool = False  # measured faster than the best at the time, but not by more than the measurement noise

    correctness_only: bool = False

    @property
    def eligible(self) -> bool:
        if self.correctness_only:
            return self.report is not None and self.report.eligible
        return self.report is not None and self.report.eligible and self.report.speedup is not None

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
    best_speedup: float | None = None  # as measured when selected
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


def _sha(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()


def speedup_se(rep: EvalReport | None) -> float:
    """Relative standard error of a measured speedup (ratio of two medians), from the per-repetition cvs.
    0 when the report has no timing statistics."""
    b = (rep.benchmark if rep else None) or {}
    try:
        c, base = b["candidate"], b["baselines"][b["best_baseline"]]
        n = max(1, min(c["reps"], base["reps"]))
        return math.sqrt(math.pi / 2) * math.hypot(c["cv"], base["cv"]) / math.sqrt(n)  # SE of a median ~ sqrt(pi/2) sigma / sqrt(n)
    except (KeyError, TypeError):
        return 0.0


class Orchestrator:
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
    ) -> None:
        self.task = get_task(task_name, dtype, size)
        self.task_name, self.dtype, self.size = task_name, dtype, size
        self.cfg, self.search = llm_cfg, search
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir) if run_dir else REPO_ROOT / "runs" / f"{stamp}_{task_name}_{dtype}_{llm_cfg.model}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = CostLedger(llm_cfg, search.budget_usd)
        self.llm = LLM(llm_cfg, self.ledger, Transcript(self.run_dir / "transcript.jsonl"), client=client, max_concurrency=max(2, search.branches))
        self.evaluate_fn = evaluate_fn or evaluate
        self.log = log
        self.seed = seed
        self.system = prompts.system_prompt(self.task, use_tools=llm_cfg.use_tools, profiler_evidence=search.profile_poor)
        self.index = SkillIndex("msl_reference")
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
        self._publish_state()

    def _publish_state(self, phase: str | None = None, *, inflight: str | None = None, stop_reason: str = "") -> None:
        """Rewrite ``state.json``. A failure here must not stop the search."""
        if phase is not None:
            self._phase = phase
        try:
            from observe.publish import publish

            publish(self.run_dir / "state.json", {
                "pid": os.getpid(),
                "updated_at": time.time(),
                "started_at": self.t0,
                "phase": self._phase,
                "task": self.task_name,
                "dtype": self.dtype,
                "size": self.size,
                "model": self.cfg.model,
                "round": self._round,
                "max_rounds": self.search.max_rounds,
                "spent_usd": self.ledger.total_usd,
                "budget_usd": self.search.budget_usd,
                "best_speedup": self.best.speedup if self.best else None,
                "best_label": self.best.label if self.best else None,
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

    # ------------------------------------------------------------------ evaluation
    def _save(self, c: Candidate) -> None:
        d = self.run_dir / "candidates" / f"{c.idx:03d}_{c.label}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.metal").write_text(c.source)
        if c.report:
            (d / "report.json").write_text(json.dumps(c.report.to_json(), indent=1))
            (d / "feedback.txt").write_text(format_feedback(c.report))

    def _required_gain(self, a: Candidate, b: Candidate) -> float:
        """Factor by which ``a`` must be measured faster than ``b`` before it counts as faster."""
        return max(self.search.min_improvement, 1.0 + self.search.noise_z * math.hypot(speedup_se(a.report), speedup_se(b.report)))

    async def _eval(self, source: str, label: str, *, plan: str | None = None) -> Candidate:
        """Full evaluation of a candidate (serialized on the GPU). Updates best/trace and enforces the hang breaker."""
        cand = Candidate(idx=len(self.candidates), label=label, source=source, plan=plan, correctness_only=self.search.correctness_only)
        self.candidates.append(cand)
        self._publish_state("evaluate", inflight=label)
        key = _sha(source)
        if key in self.cache:
            cand.report = self.cache[key]
        else:
            async with self.gpu_lock:
                if self.hangs >= self.search.max_hangs:
                    raise StopRun("gpu_hang_circuit_breaker")
                rep = await asyncio.to_thread(
                    self.evaluate_fn, self.task_name, self.dtype, self.size, source,
                    mode="quick" if self.search.correctness_only else "full", seed=None, reps=self.search.eval_reps,
                    progress_path=self.run_dir / "eval_progress.json",
                )
                self.evaluations += 1
            cand.report = rep
            if rep.status not in ("timeout", "crash"):
                self.cache[key] = rep
            if rep.status == "timeout":
                self.hangs += 1
            self.crashes_in_a_row = self.crashes_in_a_row + 1 if rep.status == "crash" else 0
        cand.cost_usd, cand.t_s = self.ledger.total_usd, time.time() - self.t0
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
            {"idx": cand.idx, "label": label, "status": r.status, "speedup": r.speedup, "cost_usd": cand.cost_usd, "t_s": cand.t_s, "best_speedup": self.best.speedup if self.best else None, "improved": cand.is_best, "within_noise": cand.within_noise}
        )
        self._publish_state(inflight=None)
        sp = f"{r.speedup:.3f}x" if r.speedup else "-"
        tag = "NEW BEST " if cand.is_best else ("within noise of the best " if cand.within_noise else "")
        self.log(f"  [{label}] {r.status:<13s} speedup {sp:>8s}  {tag}(${cand.cost_usd:.3f}, {cand.t_s:.0f}s)  {('' if r.status == 'ok' else r.message[:110].replace(chr(10), ' '))}")
        if self.hangs >= self.search.max_hangs or self.crashes_in_a_row >= 3:
            raise StopRun("gpu_hang_circuit_breaker" if self.hangs >= self.search.max_hangs else "repeated_worker_crashes")
        return cand

    async def _quick(self, source: str) -> str:
        """quick_check tool: compile + ABI + safety + correctness, no benchmark. Serialized on the GPU like everything else."""
        async with self.gpu_lock:
            if self.hangs >= self.search.max_hangs:
                return "error: evaluation environment disabled after a GPU hang"
            self._publish_state("evaluate", inflight="quick_check")
            rep = await asyncio.to_thread(
                self.evaluate_fn, self.task_name, self.dtype, self.size, source, mode="quick", seed=None, reps=3,
                progress_path=self.run_dir / "eval_progress.json",
            )
            if rep.status == "timeout":
                self.hangs += 1
        return format_feedback(rep)[:FEEDBACK_CHARS]

    # ------------------------------------------------------------------ generation with repair
    async def _ask(self, messages: list[dict], label: str) -> tuple[str, list[dict]]:
        """One model turn (with tools when enabled). Returns (reply text, conversation including the reply)."""
        self._publish_state("llm")
        if self.cfg.use_tools:
            handlers = make_handlers(self.index, self._quick, self.search.max_quick_checks)
            reply, msgs = await self.llm.chat_with_tools(messages, [LOOKUP_TOOL, QUICK_CHECK_TOOL], handlers, label=label)
            return reply.text(), msgs
        reply = await self.llm.chat(messages, label=label)
        return reply.text(), messages + [reply.message]

    async def _develop(self, messages: list[dict], label: str, *, plan: str | None = None) -> Candidate | None:
        """Generate a kernel from ``messages`` and repair it for up to ``max_repair_tries`` turns.
        Returns the first eligible candidate (or None). Failed attempts are still recorded as candidates."""
        name = self.task.contract.kernel_name
        for attempt in range(self.search.max_repair_tries + 1):
            tag = label if attempt == 0 else f"{label}.fix{attempt}"
            text, messages = await self._ask(messages, tag)
            source = extract_metal(text, name)
            if source is None:
                nudge = f"Your reply did not contain a ```metal code block defining `kernel void {name}`. Reply again with the complete kernel in exactly one ```metal block."
                messages = messages + [{"role": "user", "content": nudge}]
                self.log(f"  [{tag}] no code block in reply")
                continue
            cand = await self._eval(source, tag, plan=plan)
            if cand.eligible:
                return cand
            if cand.report.status in ("timeout",):
                return None
            fb = format_feedback(cand.report)[:FEEDBACK_CHARS]
            messages = messages + [{"role": "user", "content": prompts.repair_user(fb, attempt + 1, self.search.max_repair_tries)}]
        return None

    # ------------------------------------------------------------------ planning
    async def _plan(self, seed: Candidate, rnd: int, evidence: str = "") -> list[dict]:
        self._publish_state("plan")
        k = self.search.branches
        fb = format_feedback(seed.report)[:FEEDBACK_CHARS] + evidence
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": prompts.plan_user(self.task, seed.source, fb, self.history, k, mixed=self.search.plan_mix)}]
        for attempt in range(2):
            try:
                reply = await self.llm.chat(msgs, label=f"r{rnd}.plan{'' if attempt == 0 else '.retry'}")
            except LLMError as e:
                self.log(f"  planner request failed ({str(e)[:160]}); using generic plans")
                return FALLBACK_PLANS[:k]
            analysis, plans = parse_plans(reply.text(), k)
            if plans:
                self.log(f"  plan r{rnd}: {analysis[:160]}")
                for p in plans:
                    self.log(f"    - {p['title']}")
                return plans
            msgs = msgs + [reply.message, {"role": "user", "content": 'Reply with the JSON object only, in a ```json block, with a non-empty "plans" list.'}]
        self.log("  planner returned no parsable plan; using generic plans")
        return FALLBACK_PLANS[:k]

    # ------------------------------------------------------------------ main loop
    async def run(self) -> RunResult:
        res = RunResult(run_dir=str(self.run_dir), task=self.task_name, dtype=self.dtype, size=self.size, model=self.cfg.model, budget_usd=self.search.budget_usd)
        self.log(f"== {self.task_name} {self.dtype} {self.size} | model {self.cfg.model} | budget ${self.search.budget_usd:.2f} | branches {self.search.branches} | run {self.run_dir.name}")
        try:
            await self._search(res)
        except BudgetExhausted as e:
            res.stop_reason = "budget_exhausted"
            self.log(f"== stop: {e}")
        except StopRun as e:
            res.stop_reason = str(e)
            self.log(f"== STOP: {e} (the run ends immediately; a runaway GPU kernel can freeze the display)")
        except LLMError as e:
            res.stop_reason = f"llm_error: {e}"
            self.log(f"== stop: {e}")
        await self._confirm(res)
        if self.search.profile_best:
            await self._profile_best()
        await self.llm.aclose()  # the HTTP client must be closed on the loop that used it (asyncio.run closes the loop afterwards)
        return self._finish(res)

    async def _search(self, res: RunResult) -> None:
        # 1. initial implementation
        self._round = 0
        self._publish_state("llm")
        for restart in range(INITIAL_RESTARTS):
            base = [{"role": "system", "content": self.system}, {"role": "user", "content": prompts.initial_user(self.task)}]
            n = max(1, self.search.init_samples)
            self.log(f"-- initial generation (attempt {restart + 1}/{INITIAL_RESTARTS}" + (f", {n} samples in parallel" if n > 1 else "") + ")")
            label = "init" if restart == 0 else f"init{restart + 1}"
            if n == 1:
                await self._develop(base, label)
            else:
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
        if not self.best:
            res.stop_reason = "no_valid_initial_kernel"
            return
        # 2..5. rounds
        stagnant = 0
        for rnd in range(1, self.search.max_rounds + 1):
            res.rounds = rnd
            self._round = rnd
            seed = self.best
            self.log(f"-- round {rnd}: seed {seed.label} at {seed.speedup:.3f}x, spent ${self.ledger.total_usd:.3f}")
            evidence = await self._profile_seed(seed, rnd)
            plans = await self._plan(seed, rnd, evidence)
            fb = format_feedback(seed.report)[:FEEDBACK_CHARS] + evidence

            async def branch(i: int, plan: dict) -> Candidate | None:
                msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": prompts.patch_user(self.task, seed.source, fb, plan)}]
                return await self._develop(msgs, f"r{rnd}.p{i}", plan=plan["title"])

            before = self.best
            outcomes = await asyncio.gather(*[branch(i, p) for i, p in enumerate(plans)], return_exceptions=True)
            fatal = [o for o in outcomes if isinstance(o, (BudgetExhausted, StopRun))]
            llm_errors = [o for o in outcomes if isinstance(o, LLMError)]
            for p, o in zip(plans, outcomes):
                if isinstance(o, Candidate):
                    tag = f"{o.speedup:.3f}x" + (" (new best)" if o.is_best else " (within measurement noise of the best)" if o.within_noise else " (not better)")
                elif isinstance(o, LLMError):
                    tag = "skipped (the model request failed)"
                    self.log(f"  branch '{p['title']}' skipped: {str(o)[:200]}")
                elif isinstance(o, BaseException):
                    tag = f"aborted ({type(o).__name__})"
                else:
                    tag = "no valid kernel after repair"
                self.history.append(f"round {rnd}: {p['title']} -> {tag}")
            other = [o for o in outcomes if isinstance(o, BaseException) and not isinstance(o, (BudgetExhausted, StopRun, LLMError))]
            if other:
                raise other[0]
            if fatal:
                raise fatal[0]
            if llm_errors and len(llm_errors) == len(plans):  # one failed request skips its branch; all of them failing ends the run
                raise llm_errors[0]
            stagnant = 0 if self.best is not before else stagnant + 1
            if stagnant >= self.search.patience:
                res.stop_reason = f"converged ({self.search.patience} rounds without improvement)"
                return
        res.stop_reason = "max_rounds"

    def _confirm_pool(self) -> list[Candidate]:
        """The best, plus up to ``confirm_top - 1`` other eligible kernels that it does not beat by more than the noise."""
        pool, seen = [self.best], {_sha(self.best.source)}
        for c in sorted(self.candidates, key=lambda c: c.speedup, reverse=True):
            if len(pool) >= self.search.confirm_top:
                break
            if not c.eligible or _sha(c.source) in seen:
                continue
            seen.add(_sha(c.source))
            if self.best.speedup <= c.speedup * self._required_gain(self.best, c):
                pool.append(c)
        return pool

    async def _confirm(self, res: RunResult) -> None:
        """Re-measure the winner and its statistical ties with a fresh seed and twice the repetitions; the fastest
        re-measured kernel is the result (the search maximum alone is biased upward by noise)."""
        if not self.best or self.hangs >= self.search.max_hangs or self.search.correctness_only:
            return
        pool = self._confirm_pool()
        self.log(f"-- confirming {', '.join(c.label for c in pool)} with a fresh seed")
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
                    rep = await asyncio.to_thread(
                        self.evaluate_fn, self.task_name, self.dtype, self.size, c.source, mode="full", seed=None, reps=self.search.eval_reps * 2,
                        progress_path=self.run_dir / "eval_progress.json",
                    )
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
                self.hangs += 1
                break
        if not confirmed:
            res.confirmed_status = res.confirmations[0]["status"]
            return
        win, rep = max(confirmed, key=lambda cr: cr[1].speedup)
        if win is not self.best:
            self.log(f"   {win.label} is faster than {self.best.label} when re-measured; it becomes the result")
            self.best.is_best, win.is_best = False, True
            self.best = win
        res.confirmed_status, res.confirmed_speedup = rep.status, rep.speedup
        res.candidate_us, res.baseline_us = rep.candidate_us, rep.baseline_us
        (self.run_dir / "confirm_report.json").write_text(json.dumps(rep.to_json(), indent=1))

    async def _profile_seed(self, seed: Candidate, rnd: int) -> str:
        """Instruments trace of the round's seed when it is slower than the strongest baseline (``search.profile_poor``).

        Returns the text appended to the feedback of the planner and of every patch author, or "" (disabled, fast enough, over the trace budget, or the trace
        failed: profiling is a bonus and never stops the search). The same source is traced once. Traces run under the exclusive GPU lock."""
        s = self.search
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
            async with self.gpu_lock:
                summ = await asyncio.to_thread(xctrace.profile_kernel, self.task_name, self.dtype, self.size, seed.source)
            body = xctrace.format_for_llm(summ)
            pdir = self.run_dir / "profiles"
            pdir.mkdir(exist_ok=True)
            (pdir / f"r{rnd}_{seed.label}.txt").write_text(body)
            (pdir / f"r{rnd}_{seed.label}.json").write_text(json.dumps(summ.to_json(), indent=1))
            entry.update(spilled_bytes=summ.spilled_bytes, shader_code_bytes=summ.shader_code_bytes, gpu_active_fraction=summ.gpu_active_fraction)
            text = (
                "\n\nProfiler evidence for this kernel (Instruments Metal System Trace; taken because the kernel is slower than the strongest baseline). "
                "Read it with the \"Instruments trace\" section of Skill 2. Name the finding or counter that your diagnosis rests on (spills, the saturated limiter, low occupancy, ...) "
                "and make the changes address it; if the evidence says nothing about your hypothesis, say so instead of inventing numbers:\n" + body
            )
            self.log(f"   spilled {summ.spilled_bytes} B in {summ.spill_events} events; shader {summ.shader_code_bytes} B; GPU active {summ.gpu_active_fraction:.0%}" if summ.gpu_active_fraction is not None else "   trace taken")
        except Exception as e:  # noqa: BLE001 - profiling is a bonus, never a failure
            entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            self.log(f"   profiling failed: {entry['error']}")
        self.profiles.append(entry)
        self._profile_text[key] = text
        return text

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

    def _finish(self, res: RunResult) -> RunResult:
        res.spent_usd = self.ledger.total_usd
        res.evaluations = self.evaluations
        res.hangs = self.hangs
        res.trace = self.trace
        res.candidates = [c.brief() for c in self.candidates]
        res.ledger = self.ledger.summary()
        res.profiles = self.profiles
        if self.best:
            res.best_label, res.best_speedup = self.best.label, self.best.speedup
            (self.run_dir / "best.metal").write_text(self.best.source)
        (self.run_dir / "result.json").write_text(json.dumps(asdict(res), indent=1, default=str))
        (self.run_dir / "ledger.json").write_text(json.dumps({"summary": self.ledger.summary(), "requests": self.ledger.requests}, indent=1))
        self._confirmations = res.confirmations
        self._publish_state("done", stop_reason=res.stop_reason)
        sp = f"{res.confirmed_speedup:.3f}x confirmed" if res.confirmed_speedup else (f"{res.best_speedup:.3f}x" if res.best_speedup else "no valid kernel")
        self.log(f"== done: {sp} | spent ${res.spent_usd:.3f} | {res.evaluations} evaluations | stop: {res.stop_reason} | {self.run_dir}")
        return res
