"""Read-only evidence analysis of the *plan type* question, on search runs that already exist.

The question (from the user): "the paper's plans range from local instruction-level changes to global changes in thread mapping, data movement
and synchronization; our ``plan_user`` prompt forces every plan to be structural. Big FlashAttention changes tend to fail ABI/correctness, 12 of
24 first attempts of ``fa_flash`` produced no code, and on saturated elementwise ops there is no structural room to win. Can opening the plan
types solve the problem?"

    python -m experiments.plan_type_analysis <run dirs or dirs of run dirs> --out runs/plan_type_analysis [--logs <orchestrator logs>]

writes ``REPORT.md`` (the answer first) and ``data.json`` into ``--out``. Nothing else is touched: run directories and logs are only *read*
(some may be written by live processes while this runs: every file may be missing, empty or half written, and a partial run is reported as
partial; for a reproducible report give a frozen copy of the runs). Standard library only (no ``tcenv`` / ``taic`` import, no GPU, no LLM call).

Report sections: 1 = plan kind (keyword classification of each plan's text: local / thread_mapping / data_movement / synchronization / algorithm)
vs the first-attempt outcome ok / abi_error / incorrect / no_code, per task and per kind, with the classifier checked against the plans the
plan_mix prompt designated as local. 2 = the replies without code: cause (from ``finish_reason`` and the token counts of the last turn),
cost, thinking budget per model and task, and whether plan text, plan scope or seed size predicts it. 3 = size of the change vs outcome.
4 = saturated ops. 5 = conclusions and limits.

Section 3 (``big changes on FlashAttention fail``): every ``r<k>.p<i>`` patch request ("branch") and every kernel it produced ("attempt":
``r<k>.p<i>``, ``r<k>.p<i>.fix1`` ...) is classified by status (ok / compile_error / abi_error / unsafe / incorrect / runtime_error /
timeout / crash / rejected / no_code), with first-attempt status, what repair turns gave, and whether the attempt became a new best.
Size of change = line-level diff between the candidate and the *seed it patched* (the best kernel at the start of that round, reconstructed from
the candidate records and the ``improved`` flags of the run's trace; cross-checked against the seed printed in the transcript's patch prompt
when a transcript exists). Runs with ``plan_mix`` on (state/config key, else the wording of the planner prompt in the transcript) are split
into the designated local plan (the last plan of each round) and the structural ones.

Section 4 (``saturated ops have no structural room``): per finished run, best speedup vs the best baseline, achieved roofline fraction
(binding bound of ``result/confirm_report``: ``lower_bound_us / candidate_us``), rounds, USD, USD after the last improvement and after the
first candidate at >= ``--sat`` of the roofline, stop reason.

Honest limits, repeated in the report: the runs differ in budget, model, protocol and search aids, so there is **no cross-run causal claim**;
every rate carries its n, and n < ``--small-n`` is flagged as anecdotal.
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import glob
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STATUS_ORDER = ["ok", "compile_error", "abi_error", "unsafe", "incorrect", "runtime_error", "timeout", "crash", "rejected", "no_code"]
FAIL_STATUSES = STATUS_ORDER[1:]
DEFAULT_SIZE_EDGES = (0.20, 0.40)  # diff fraction: < 0.20 small, < 0.40 medium, else large (FA median diff is 0.25; see the sensitivity table)
SENSITIVITY_EDGES = ((0.15, 0.35), (0.20, 0.40), (0.25, 0.50), (0.25, 0.60))
BUCKETS = ("small", "medium", "large")
SAT_FRACTION = 0.9
SMALL_N = 10
MIN_IMPROVEMENT = 1.005  # taic.toml / SearchConfig defaults, used only to *replay* the best chain when a run has no trace
NOISE_Z = 2.0
DEFAULT_MAX_REPAIR = 3

# generic plans the orchestrator substitutes when the planner fails (taic/orchestrator.py FALLBACK_PLANS); such a round has no local plan
FALLBACK_TITLES = {
    "vectorized 16-byte memory access",
    "single pass over global memory with SIMD-group reductions",
    "use simdgroup_matrix for matrix products",
}

# ----------------------------------------------------------------------------------------------------------------------
# plan kinds. The paper (Section 3.3) says plans range "from local instruction-level changes to global changes in thread mapping, data
# movement and synchronization". The lexicons below are therefore: (1) the local edits listed in taic/prompts.py MIXED_PLANS ("vector width,
# unroll or tile factor, loop order, hoisting loop-invariant work, fewer barriers or live registers, fast-math intrinsics, address arithmetic,
# reading operands once") plus close synonyms, (2) one lexicon per global axis of the paper, (3) algorithm-level rewrites (Winograd, online
# softmax cadence ...). A plan's kind = the best-scoring one. It is a keyword heuristic, NOT a human label: section 1.1 of the report checks it
# against the plans the prompt itself designated as local (plan_mix runs) and reports the disagreement. Terms were tuned once after reading the
# 12 designated local plans of those runs, so that agreement is optimistic and is reported as such.
# ----------------------------------------------------------------------------------------------------------------------
PLAN_AXES = ("thread_mapping", "data_movement", "synchronization", "algorithm")  # tie-break order: most global first
PLAN_KINDS = ("local", *PLAN_AXES, "generic", "unclassified")
_LOCAL_RX = (
    r"hoist|strength[- ]reduc|\bexp2\b|\bfma\b|reciprocal|\brcp\b|fast[- ]math|fast::|unroll|live[- ]range|live registers|register[- ]lean|register budget|spill"
    r"|cheap(?:er)?\b|instruction (?:diet|count)|instruction-level|branch(?:less|-free)|predicat|\bpeel|address (?:arithmetic|math)|32-bit (?:index|offset|incremental)"
    r"|packed (?:half2|fp16)|\bhalf2\b|(?:delete|drop|remove|kill|cut)\b.{0,20}\b(?:redundant|dead|duplicate|second|residual)|\bdead\b|redundant|one-line|zero-risk|loop-invariant"
    r"|loop (?:order|interchange)|(?:read|load)(?:ing)? (?:each|operands?|every).{0,25}\bonce|\bonce per\b|vector width|vectori[sz]|tile factor|widen the|\bwider\b|max_total_threads"
    r"|declare the real|\bsfu\b|\btanh\b|taylor|minimax|polynomial|sincos|integer-pipe|\bfewer (?:instructions|ops)\b"
)
_DECLARED_LOCAL_RX = r"\blocal\b|structure-preserving|keep(?:ing)? the (?:whole )?(?:kernel'?s? )?(?:structure|kernel as is)|no structural change|instruction-level|zero-risk"
_AXIS_RX = {
    "thread_mapping": (
        r"split[- ]?k\b|\bk[- ]split|key[- ]split|\b\d+-?way (?:key|split|\()|across (?:the |two |four )?(?:simd|warp|thread)|\bper simd|simd[- ]?groups? per|rows? per (?:simd|threadgroup|thread)"
        r"|\bper threadgroup\b|producer|consumer|specializ|repartition|\bteams?\b|thread mapping|launch mapping|persistent|register[- ]blocking|\bm[- ]blocking"
        r"|resident (?:threadgroups?|tgs?)|residency|occupancy|\btgs?/core|threadgroups? per core|block[- ]index swizzle|grid swizzle|tile traversal|one (?:complete )?row per"
        r"|pair(?:ing)? (?:low|rows)|interleav|\bsub-?tiled?\b|work (?:partition|distribution)|balance the"
    ),
    "data_movement": (
        r"\bstag(?:e|ed|ing)\b|double[- ]?buffer|\bring\b|threadgroup[- ]memory|shared[- ]memory|\bsmem\b|\bslab\b|prefetch|register[- ]resident|resident in registers|in registers"
        r"|register[- ]park|(?:directly )?(?:from|to) (?:global|device)|direct(?:-| )(?:from-)?(?:global|device)|straight (?:from|to|into)|layout|swizzl|padd|transpos|coalesc"
        r"|software[- ]pipelin|pipelin|\basync|bank[- ]conflict|fragment (?:order|loads?)|memory[- ]level parallelism|\bmlp\b|outstanding|in flight|\bburst|\bstream(?:ing|ed)?\b"
        r"|overlay|epilogue|footprint"
    ),
    "synchronization": r"barrier|synchroni[sz]|\bsync\b|ping-?pong|\batomic|relaxed|\bfence|simd_shuffle|\bshuffle|lock-free|handshake|rendezvous",
    "algorithm": (
        r"winograd|online[- ]softmax|\bdeferred\b|\blagged\b|recurrence|identity|s\^t|algebra|reformulat|log-depth|tree\b|\bmerge\b|single[- ]pass|one pass|f\(\d|threshold-triggered"
        r"|rescale|fuse[ds]? (?:the )?(?:softmax|epilogue|normali)"
    ),
}
_RX = {k: re.compile(v, re.I) for k, v in _AXIS_RX.items()}
_LOCAL = re.compile(_LOCAL_RX, re.I)
_DECLARED = re.compile(_DECLARED_LOCAL_RX, re.I)


def classify_plan(title: str, change: str, fallback: bool = False) -> dict:
    """Keyword classification of a plan: ``{kind, scope, scores, local_score, declared_local}``. A hit in the title counts double and the
    description counts at most 3 hits per lexicon, so that a long description does not outvote the headline. ``kind`` is ``local`` when
    the plan declares itself local ('Local, structure-preserving ...') or its local score beats every global axis, else the best global axis
    (ties: thread_mapping > data_movement > synchronization > algorithm); ``generic`` = one of the orchestrator's fallback plans."""
    t, c = (title or "").lower(), (change or "").lower()
    scores = {a: 2 * len(rx.findall(t)) + min(3, len(rx.findall(c))) for a, rx in _RX.items()}
    local = 2 * len(_LOCAL.findall(t)) + min(3, len(_LOCAL.findall(c)))
    declared = bool(_DECLARED.search(t) or _DECLARED.search(c[:240]))
    best = max(scores.values())
    if fallback:
        kind = "generic"
    elif declared or local > best:
        kind = "local"
    elif best == 0:
        kind = "unclassified"
    else:
        kind = next(a for a in PLAN_AXES if scores[a] == best)
    scope = "local" if kind == "local" else ("structural" if kind in PLAN_AXES else kind)
    return {"kind": kind, "scope": scope, "scores": scores, "local_score": local, "declared_local": declared}


# task groups for the tables (the paper's three hard kernels + the rest of the suite)
TASK_GROUPS = {
    "flash_attention": "flash_attention",
    "gemm": "matrix (gemm/conv2d/mamba2)", "conv2d": "matrix (gemm/conv2d/mamba2)", "mamba2_chunk_scan": "matrix (gemm/conv2d/mamba2)",
}


def task_group(task: str) -> str:
    return TASK_GROUPS.get(task.split("@")[0], "memory-bound (elementwise/reduction/gemv/...)")


LABEL_RE = re.compile(r"^(?P<branch>init\d*(?:\.s\d+)?|r(?P<round>\d+)\.p(?P<plan>\d+))(?:\.fix(?P<fix>\d+))?$")
PLAN_TAG_RE = re.compile(r"^r(\d+)\.plan(?:\.retry)?$")
PATCH_TAG_RE = re.compile(r"^r\d+\.p\d+$")
PLAN_K_RE = re.compile(r"propose (\d+) DIFFERENT optimization plans")
PATCH_RE = re.compile(r"Implement exactly this optimization plan and nothing else:\n\*\*(?P<title>.*?)\*\*\n(?P<change>.*?)\n\nKeep everything that already works", re.S)
SEED_RE = re.compile(r"```metal\n(?P<src>.*?)\n```\n\nEvaluation feedback", re.S)
MIXED_MARK = "LOCAL, low-risk change"


# ----------------------------------------------------------------------------------------------------------------------
# small statistics helpers
# ----------------------------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value of the 2x2 table [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0:
        return 1.0

    def pmf(x: int) -> float:
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    p0 = pmf(a)
    return min(1.0, sum(p for p in (pmf(x) for x in range(lo, hi + 1)) if p <= p0 * (1 + 1e-9)))


def n_per_arm(p1: float, p2: float, alpha_z: float = 1.96, power_z: float = 0.8416) -> int | None:
    """Sample size per arm of a two-sided two-proportion z test (alpha 0.05, power 0.8)."""
    if abs(p1 - p2) < 1e-9:
        return None
    pb = (p1 + p2) / 2
    num = (alpha_z * math.sqrt(2 * pb * (1 - pb)) + power_z * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def mann_whitney_p(x: list[float], y: list[float]) -> float | None:
    """Two-sided Mann-Whitney U test (normal approximation with tie and continuity correction)."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return None
    pooled = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    n = n1 + n2
    ranks, ties, i = [0.0] * n, 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u1 = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0) - n1 * (n1 + 1) / 2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return 1.0
    z = (abs(u1 - n1 * n2 / 2) - 0.5) / math.sqrt(var)
    return math.erfc(max(0.0, z) / math.sqrt(2))


def cmh(strata: list[tuple[int, int, int, int]]) -> dict:
    """Cochran-Mantel-Haenszel test over 2x2 tables (a, b, c, d) = (A ok, A fail, B ok, B fail), one per stratum (run): common odds ratio
    of A vs B and the continuity-corrected chi-square p-value (1 df). Strata without both groups carry no information and are skipped."""
    num = den = sa = se = sv = 0.0
    used = 0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        used += 1
        num += a * d / n
        den += b * c / n
        sa += a
        se += (a + b) * (a + c) / n
        sv += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    if used == 0 or sv <= 0:
        return {"strata": used, "odds_ratio": None, "p": None}
    chi = (abs(sa - se) - 0.5) ** 2 / sv if abs(sa - se) > 0.5 else 0.0
    return {"strata": used, "odds_ratio": (num / den) if den > 0 else None, "p": math.erfc(math.sqrt(chi / 2))}


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def pct(k: int, n: int) -> str:
    return "-" if n == 0 else f"{100 * k / n:.0f}%"


def rate_cell(k: int, n: int, small_n: int = SMALL_N, ci: bool = False) -> str:
    """``k/n (p%)`` plus ``†`` when n < small_n (anecdotal), optionally with the 95% Wilson interval."""
    if n == 0:
        return "0/0"
    s = f"{k}/{n} ({pct(k, n)})"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:.0f}-{100 * hi:.0f}]"
    return s + ("†" if n < small_n else "")


def fnum(v, d: int = 3, suffix: str = "") -> str:
    return "-" if v is None else f"{v:.{d}f}{suffix}"


def md_table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    if not rows:
        return "_(no data)_\n"
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------------------------------
# reading run directories (tolerant: files may be missing, empty or half written)
# ----------------------------------------------------------------------------------------------------------------------
def load_json(path: Path):
    try:
        text = path.read_text()
        return json.loads(text) if text.strip() else None
    except (OSError, ValueError):
        return None


def parse_label(label: str) -> dict | None:
    m = LABEL_RE.match(label)
    if not m:
        return None
    attempt = int(m["fix"]) if m["fix"] else 0
    if m["round"] is not None:
        return {"kind": "patch", "round": int(m["round"]), "plan": int(m["plan"]), "attempt": attempt, "branch": m["branch"]}
    return {"kind": "init", "round": 0, "plan": None, "attempt": attempt, "branch": m["branch"]}


def roofline_fraction(roof: dict | None) -> float | None:
    """Fraction of the roofline lower bound a kernel reaches = lower_bound_us / candidate_us = bandwidth_fraction (memory-bound)
    or compute_fraction (compute-bound). May exceed 1: the roofline peaks are measured once and are not a hard ceiling."""
    if not roof:
        return None
    t = roof.get("time_over_lower_bound")
    if t:
        return 1.0 / t
    key = "bandwidth_fraction" if roof.get("bound") == "memory" else "compute_fraction"
    return roof.get(key)


def speedup_se(rep: dict | None) -> float:
    """Relative standard error of a measured speedup (same formula as ``taic.orchestrator.speedup_se``); 0 without statistics."""
    b = (rep or {}).get("benchmark") or {}
    try:
        c, base = b["candidate"], b["baselines"][b["best_baseline"]]
        n = max(1, min(c["reps"], base["reps"]))
        return math.sqrt(math.pi / 2) * math.hypot(c["cv"], base["cv"]) / math.sqrt(n)
    except (KeyError, TypeError):
        return 0.0


def find_key(obj, key: str, depth: int = 3):
    """First value stored under ``key`` in a nested dict (trace/candidates/ledger are not searched)."""
    if not isinstance(obj, dict) or depth < 0:
        return None
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if k in ("trace", "candidates", "ledger", "confirmations") or not isinstance(v, dict):
            continue
        r = find_key(v, key, depth - 1)
        if r is not None:
            return r
    return None


@dataclass
class Cand:
    idx: int
    label: str
    kind: str  # init | patch
    round: int
    plan: int | None
    attempt: int
    branch: str
    status: str
    speedup: float | None
    source: str
    roofline: dict
    se: float
    candidate_us: float | None = None
    baseline_us: dict = field(default_factory=dict)
    task: str = ""
    dtype: str = ""
    bound_detail: str = ""  # first ABI violation / first failing correctness case, normalised
    incorrect_kind: str = ""  # marginal | gross (status incorrect only)
    improved: bool | None = None
    within_noise: bool | None = None
    cost_usd: float | None = None
    seed_label: str | None = None
    seed_speedup: float | None = None
    diff_frac: float | None = None
    lines_changed: int | None = None

    @property
    def eligible(self) -> bool:
        return self.status == "ok" and self.speedup is not None

    @property
    def roof_frac(self) -> float | None:
        return roofline_fraction(self.roofline) if self.status == "ok" else None

    @property
    def ratio(self) -> float | None:
        """Speedup relative to the seed it patched (both measured against the baseline of their own evaluation)."""
        if self.eligible and self.seed_speedup:
            return self.speedup / self.seed_speedup
        return None


def _normalise_violation(text: str) -> str:
    return re.sub(r"\d+", "N", str(text))[:110]


def _candidate_from_dir(d: Path) -> Cand | None:
    m = re.match(r"^(\d+)_(.+)$", d.name)
    if not m:
        return None
    label = m.group(2)
    info = parse_label(label)
    rep = load_json(d / "report.json")
    if info is None or not isinstance(rep, dict) or "status" not in rep:
        return None
    try:
        source = (d / "kernel.metal").read_text()
    except OSError:
        source = ""
    status = rep.get("status") or "unknown"
    detail, inc = "", ""
    if status == "abi_error":
        v = (rep.get("abi") or {}).get("violations") or [rep.get("message", "")]
        detail = _normalise_violation(v[0])
    elif status in ("incorrect", "unsafe"):
        detail = _normalise_violation(rep.get("message", ""))
        bad = [c for c in (rep.get("correctness") or {}).get("cases", []) if not c.get("ok", True)]
        if status == "incorrect" and bad:
            c = bad[0]
            n_el = c.get("n_elems") or 0
            inc = "marginal" if n_el and (c.get("n_bad", 0) / n_el) < 1e-4 and (c.get("scaled_err") or 0) < 3 else "gross"
    return Cand(
        idx=int(m.group(1)), label=label, kind=info["kind"], round=info["round"], plan=info["plan"], attempt=info["attempt"], branch=info["branch"],
        status=status, speedup=rep.get("speedup"), source=source, roofline=rep.get("roofline") or {}, se=speedup_se(rep),
        candidate_us=rep.get("candidate_us"), baseline_us=rep.get("baseline_us") or {}, task=rep.get("task") or "", dtype=rep.get("dtype") or "",
        bound_detail=detail, incorrect_kind=inc,
    )


@dataclass
class Transcript:
    present: bool = False
    tags: dict = field(default_factory=dict)  # tag -> {turns, truncated, cost, last_finish}
    plan_rounds: dict = field(default_factory=dict)  # round -> {k, mixed}
    patches: dict = field(default_factory=dict)  # tag (r<k>.p<i>) -> {title, change, seed}
    total_usd: float | None = None
    max_round: int = 0
    model: str = ""
    plan_cost: float = 0.0
    plan_calls: int = 0
    turns: list = field(default_factory=list)  # one dict per chat turn: tag, turn, finish, token counts, reply sizes (see ``turn_row``)


def _tok(usage, key: str) -> int | None:
    """A token count from a logged usage dict (``usage`` may be a plain string on endpoints without model_dump)."""
    if not isinstance(usage, dict):
        return None
    if key == "reasoning":
        v = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    else:
        v = usage.get(key)
    return int(v) if isinstance(v, (int, float)) else None


def turn_row(e: dict, tag: str) -> dict:
    """What one logged chat turn says about *how the reply ended* (no prompt or reply text is kept)."""
    reply = e.get("reply") if isinstance(e.get("reply"), dict) else {}
    usage = e.get("usage")
    content = reply.get("content") or ""
    reasoning = reply.get("reasoning_content") or ""
    m = re.search(r"#t(\d+)$", str(e.get("label") or ""))
    return {
        "tag": tag, "turn": int(m.group(1)) if m else 0, "finish": e.get("finish_reason"), "prompt": _tok(usage, "prompt_tokens"),
        "completion": _tok(usage, "completion_tokens"), "reasoning": _tok(usage, "reasoning"), "content_chars": len(content),
        "reasoning_chars": len(reasoning), "tool_calls": len(reply.get("tool_calls") or []), "has_fence": "```" in content,
        "reasoning_fences": reasoning.count("```") // 2, "cost": e.get("cost_usd") or 0.0, "seconds": e.get("seconds"),
    }


def read_transcript(path: Path) -> Transcript:
    tr = Transcript()
    if not path.exists():
        return tr
    tr.present = True
    try:
        fh = path.open()
    except OSError:
        return tr
    with fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:  # a line being written right now
                continue
            if not isinstance(e, dict) or e.get("event") != "chat":
                continue
            tag = str(e.get("label") or "").split("#")[0]
            t = tr.tags.setdefault(tag, {"turns": 0, "truncated": 0, "cost": 0.0, "truncated_cost": 0.0, "last_finish": None})
            t["turns"] += 1
            t["cost"] += e.get("cost_usd") or 0.0
            t["last_finish"] = e.get("finish_reason")
            row = turn_row(e, tag)
            tr.turns.append(row)
            t["last_turn"] = row
            if t["turns"] == 1:
                t["first_turn"] = row
            tr.model = tr.model or str(e.get("model") or "")
            if e.get("finish_reason") == "length":
                t["truncated"] += 1
                t["truncated_cost"] += e.get("cost_usd") or 0.0
            if e.get("total_usd") is not None:
                tr.total_usd = e["total_usd"]
            info = parse_label(tag)
            rnd = info["round"] if info else None
            pm = PLAN_TAG_RE.match(tag)
            if pm:
                rnd = int(pm.group(1))
            if rnd is not None:
                tr.max_round = max(tr.max_round, rnd)
            msgs = e.get("messages") or []
            last = msgs[-1].get("content") if msgs and isinstance(msgs[-1], dict) else None
            if not isinstance(last, str):
                continue
            if pm:
                tr.plan_cost += e.get("cost_usd") or 0.0
                tr.plan_calls += 1
            if pm and int(pm.group(1)) not in tr.plan_rounds:
                km = PLAN_K_RE.search(last)
                tr.plan_rounds[int(pm.group(1))] = {"k": int(km.group(1)) if km else None, "mixed": MIXED_MARK in last}
            elif PATCH_TAG_RE.match(tag) and tag not in tr.patches:
                pmatch, smatch = PATCH_RE.search(last), SEED_RE.search(last)
                tr.patches[tag] = {
                    "title": pmatch.group("title").strip() if pmatch else None,
                    "change": pmatch.group("change").strip() if pmatch else None,
                    "seed": smatch.group("src") if smatch else None,
                }
    return tr


# ----------------------------------------------------------------------------------------------------------------------
# diff size
# ----------------------------------------------------------------------------------------------------------------------
def norm_lines(src: str) -> list[str]:
    """Source lines with comments, blank lines and whitespace differences removed."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    out = []
    for ln in src.splitlines():
        ln = re.sub(r"\s*//.*$", "", ln)
        ln = re.sub(r"\s+", " ", ln.strip())
        if ln:
            out.append(ln)
    return out


def diff_stats(seed_src: str, cand_src: str) -> tuple[float, int]:
    """(fraction of changed lines, lines changed). Fraction = (lines removed from the seed + lines added) / (lines of both):
    0 = identical, ~0.3 = about a third of the lines differ, 1 = nothing in common. Line-level Ratcliff/Obershelp matching."""
    a, b = norm_lines(seed_src), norm_lines(cand_src)
    if not a and not b:
        return 0.0, 0
    matched = sum(t.size for t in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    changed = (len(a) - matched) + (len(b) - matched)
    return changed / (len(a) + len(b)), changed


def bucket_of(frac: float | None, edges: tuple[float, float]) -> str | None:
    if frac is None:
        return None
    return BUCKETS[0] if frac < edges[0] else BUCKETS[1] if frac < edges[1] else BUCKETS[2]


# ----------------------------------------------------------------------------------------------------------------------
# a run
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Run:
    path: str
    label: str
    source: str  # the command-line root it was found under
    task: str = ""
    dtype: str = ""
    model: str = ""
    budget_usd: float | None = None
    spent_usd: float | None = None
    rounds: int = 0
    stop_reason: str = ""
    finished: bool = False
    live: bool = False
    result: dict | None = None
    state: dict | None = None
    confirm: dict | None = None
    cands: list[Cand] = field(default_factory=list)
    order: list[int] = field(default_factory=list)  # candidate indices in evaluation (trace) order
    transcript: Transcript = field(default_factory=Transcript)
    plan_mix: bool | None = None
    plan_mix_source: str = "unknown"
    k: int | None = None
    trace_source: str = "replay"
    seed_check: dict = field(default_factory=lambda: {"checked": 0, "match": 0})
    unreadable: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def base_task(self) -> str:
        return self.task.split("@", 1)[0]

    @property
    def by_idx(self) -> dict[int, Cand]:
        return {c.idx: c for c in self.cands}

    @property
    def stop_kind(self) -> str:
        if not self.finished:
            return "partial"
        r = self.stop_reason or ""
        if r.startswith("converged"):
            return "patience"
        if r == "max_rounds":
            return "max_rounds"
        if r.startswith("budget"):
            return "budget"
        return r.split(":")[0] or "unknown"


def is_run_dir(p: Path) -> bool:
    return (p / "candidates").is_dir() or ((p / "result.json").exists() and (p / "transcript.jsonl").exists())


def discover(roots: list[str]) -> list[tuple[Path, str, str]]:
    """(run dir, label, source root name) for every run dir below the roots (a root may itself be a run dir; globs are expanded)."""
    found, seen = [], set()
    expanded: list[Path] = []
    for r in roots:
        hits = [Path(h) for h in sorted(glob.glob(r))] if any(ch in r for ch in "*?[") and not Path(r).exists() else [Path(r)]
        expanded += hits
    for root in expanded:
        root = root.resolve() if root.exists() else root
        if not root.exists():
            continue

        def walk(p: Path, depth: int):
            if is_run_dir(p):
                yield p
                return
            if depth <= 0:
                return
            for c in sorted(p.iterdir()):
                if c.is_dir() and not c.name.startswith(".") and c.name not in ("candidates", "profiles"):
                    yield from walk(c, depth - 1)

        for d in walk(root, 3):
            key = d.resolve()
            if key in seen:
                continue
            seen.add(key)
            rel = d.relative_to(root) if d != root else Path()
            found.append((d, str(Path(root.name) / rel), root.name))
    return found


def load_run(path: Path, label: str, source: str, now: float | None = None) -> Run:
    now = time.time() if now is None else now
    run = Run(path=str(path), label=label, source=source)
    run.result, run.state, run.confirm = load_json(path / "result.json"), load_json(path / "state.json"), load_json(path / "confirm_report.json")
    run.result = run.result if isinstance(run.result, dict) else None
    run.state = run.state if isinstance(run.state, dict) else None
    cdir = path / "candidates"
    if cdir.is_dir():
        for d in sorted(cdir.iterdir()):
            if not d.is_dir():
                continue
            c = _candidate_from_dir(d)
            if c is None:
                run.unreadable += 1
            else:
                run.cands.append(c)
    run.cands.sort(key=lambda c: c.idx)
    run.transcript = read_transcript(path / "transcript.jsonl")
    res, st = run.result or {}, run.state or {}
    run.finished = bool(run.result)
    run.live = (not run.finished) and bool(st) and st.get("phase") != "done" and (now - float(st.get("updated_at") or 0)) < 900
    first = run.cands[0] if run.cands else None
    run.task = res.get("task") or st.get("task") or (first.task if first else "") or ""
    run.dtype = res.get("dtype") or st.get("dtype") or (first.dtype if first else "") or ""
    run.model = res.get("model") or st.get("model") or run.transcript.model or ""
    run.budget_usd = res.get("budget_usd", st.get("budget_usd"))
    run.spent_usd = res.get("spent_usd", st.get("spent_usd", run.transcript.total_usd))
    run.rounds = res.get("rounds") if run.finished else st.get("round", max([c.round for c in run.cands], default=0))
    run.rounds = run.rounds or max([c.round for c in run.cands], default=0)
    run.stop_reason = res.get("stop_reason", "") if run.finished else ""
    if not run.finished and st:
        run.notes.append(f"partial: phase={st.get('phase')}, round {st.get('round')}" + (", live" if run.live else ", not updated for >15 min"))
    elif not run.finished:
        run.notes.append("partial: no result.json / state.json")
    # plan_mix: explicit config key first, then the wording of the planner prompt
    for src_name, obj in (("state/result", st), ("state/result", res), ("config", load_json(path / "config.json"))):
        v = find_key(obj, "plan_mix") if obj else None
        if isinstance(v, bool):
            run.plan_mix, run.plan_mix_source = v, src_name
            break
    pr = run.transcript.plan_rounds
    if run.plan_mix is None and pr:
        run.plan_mix, run.plan_mix_source = any(v["mixed"] for v in pr.values()), "transcript"
    ks = [v["k"] for v in pr.values() if v.get("k")]
    run.k = Counter(ks).most_common(1)[0][0] if ks else None
    # trace: improved flags and the ledger total at each evaluation
    trace = res.get("trace") or st.get("trace") or []
    tmap = {t["idx"]: t for t in trace if isinstance(t, dict) and "idx" in t}
    run.trace_source = "result.json" if res.get("trace") else "state.json" if st.get("trace") else "replay"
    by_idx = run.by_idx
    run.order = [t["idx"] for t in trace if isinstance(t, dict) and t.get("idx") in by_idx]
    run.order += [c.idx for c in run.cands if c.idx not in run.order]
    replay_best: Cand | None = None
    for i in run.order:
        c = by_idx[i]
        t = tmap.get(i)
        if t is not None:
            c.improved, c.within_noise, c.cost_usd = bool(t.get("improved")), bool(t.get("within_noise")), t.get("cost_usd")
            if c.improved:
                replay_best = c
        else:  # not in any trace: decide exactly like the orchestrator would
            if c.eligible and (replay_best is None or c.speedup > replay_best.speedup * max(MIN_IMPROVEMENT, 1 + NOISE_Z * math.hypot(c.se, replay_best.se))):
                c.improved, replay_best = True, c
            else:
                c.improved = False
            c.within_noise = bool(c.eligible and replay_best is not None and not c.improved and c.speedup > replay_best.speedup)
    if first is None and not run.live and not run.finished:
        run.notes.append("no candidate evaluated yet")
    assign_seeds(run)
    return run


def assign_seeds(run: Run) -> None:
    """The seed of round r is the best kernel just before the first candidate of round r was evaluated (``Orchestrator._search``
    patches ``self.best`` as it stood at the start of the round). Compute the diff of every patch attempt against it."""
    by_idx = run.by_idx
    best: Cand | None = None
    seed: dict[int, Cand | None] = {}
    for i in run.order:
        c = by_idx[i]
        if c.kind == "patch" and c.round not in seed:
            seed[c.round] = best
        if c.improved:
            best = c
    for c in run.cands:
        if c.kind != "patch":
            continue
        s = seed.get(c.round)
        if s is None:
            continue
        c.seed_label, c.seed_speedup = s.label, s.speedup
        if s.source and c.source:
            c.diff_frac, c.lines_changed = diff_stats(s.source, c.source)
    # cross-check against the seed shown to the model in the patch prompt
    for tag, p in run.transcript.patches.items():
        info = parse_label(tag)
        s = seed.get(info["round"]) if info else None
        if s is None or not p.get("seed") or not s.source:
            continue
        run.seed_check["checked"] += 1
        run.seed_check["match"] += int(norm_lines(p["seed"]) == norm_lines(s.source))


# ----------------------------------------------------------------------------------------------------------------------
# branches (patch requests) and per-attempt records
# ----------------------------------------------------------------------------------------------------------------------
REASONING_SHARE = 0.9  # a truncated reply whose reasoning tokens are >= this share of its completion tokens died while *thinking*


def nocode_cause(tr: Transcript, tag: str) -> str:
    """Why the model turn(s) of ``tag`` produced no kernel, read from how the last logged turn ended:
    ``reasoning_cap`` = finish_reason=length and >= 90% of the completion tokens were reasoning (the cap was spent thinking, nothing was written);
    ``output_cap`` = finish_reason=length with a long visible reply (the code itself did not fit); ``run_ended`` = the last turn asked for a tool and
    the run stopped before answering it (budget); ``no_block`` = a normal reply without a ```metal block; ``block_without_kernel`` = a code block
    that does not define the kernel; ``unknown`` = no transcript."""
    row = (tr.tags.get(tag) or {}).get("last_turn")
    if not row:
        return "unknown"
    fin = row.get("finish")
    if fin == "length":
        comp, reas = row.get("completion"), row.get("reasoning")
        if comp and reas is not None and reas / comp < REASONING_SHARE:
            return "output_cap"
        return "reasoning_cap"
    if fin == "tool_calls":
        return "run_ended"
    if fin == "stop":
        return "block_without_kernel" if row.get("has_fence") else "no_block"
    return "unknown"


def build_branches(run: Run, edges: tuple[float, float]) -> tuple[list[dict], list[dict]]:
    """(branch records, attempt records) of one run, for init and patch branches alike."""
    tr = run.transcript
    cands_by_label = {c.label: c for c in run.cands}
    keys: dict[str, dict] = {}
    for c in run.cands:
        keys[c.branch] = parse_label(c.branch)
    for tag in tr.tags:
        info = parse_label(tag)
        if info:
            keys.setdefault(info["branch"], parse_label(info["branch"]))
    max_round_seen = max([c.round for c in run.cands] + [tr.max_round, 0])
    max_attempt_run = max([c.attempt for c in run.cands] + [DEFAULT_MAX_REPAIR])
    branches, attempts = [], []
    # round -> number of planned branches (for the local / last plan designation)
    planned: dict[int, set[int]] = defaultdict(set)
    for b, info in keys.items():
        if info and info["kind"] == "patch":
            planned[info["round"]].add(info["plan"])
    for base, info in sorted(keys.items(), key=lambda kv: (kv[1]["round"], kv[1]["plan"] if kv[1]["plan"] is not None else -1, kv[0])):
        tags = [base if n == 0 else f"{base}.fix{n}" for n in range(max_attempt_run + 1)]
        present = [n for n, t in enumerate(tags) if t in cands_by_label or t in tr.tags]
        if not present:
            continue
        last_n = max(present)
        seq = []
        for n in range(last_n + 1):
            t = tags[n]
            c = cands_by_label.get(t)
            if c is not None:
                seq.append({"attempt": n, "tag": t, "status": c.status, "cand": c})
            elif t in tr.tags:
                closed = run.finished or n < last_n or max_round_seen > info["round"]
                cause = nocode_cause(tr, t)
                reason = "truncated" if tr.tags[t].get("last_finish") == "length" else ("run_ended" if cause == "run_ended" else "no_block")
                seq.append({"attempt": n, "tag": t, "status": "no_code" if closed else "pending", "cand": None, "reason": reason, "cause": cause})
            else:
                seq.append({"attempt": n, "tag": t, "status": "no_code", "cand": None, "reason": "gap"})
        first, last = seq[0], seq[-1]
        # an unfinished run: the last attempt of the newest round may still get repair turns
        open_ = (not run.finished) and last["status"] != "ok" and (last["status"] == "pending" or (last_n < DEFAULT_MAX_REPAIR and max_round_seen <= info["round"]))
        accepted = [s["cand"] for s in seq if s["cand"] is not None and s["cand"].improved]
        okc = [s["cand"] for s in seq if s["cand"] is not None and s["cand"].eligible]
        final_c = okc[-1] if okc else None
        fc = first["cand"]
        plan = tr.patches.get(base) or {}
        title = plan.get("title")
        fallback = (title or "") in FALLBACK_TITLES
        pk = classify_plan(title or "", plan.get("change") or "", fallback) if title else {"kind": "unknown", "scope": "unknown", "scores": {}, "local_score": 0, "declared_local": False}
        ft = (tr.tags.get(base) or {}).get("first_turn") or {}
        seed_speedup = next((s["cand"].seed_speedup for s in seq if s["cand"] is not None and s["cand"].seed_speedup), None)
        seed_label = next((s["cand"].seed_label for s in seq if s["cand"] is not None and s["cand"].seed_label), None)
        ratio = final_c.ratio if final_c is not None else None
        rec = {
            "run": run.label, "source": run.source, "task": run.base_task, "variant": run.task, "dtype": run.dtype, "model": run.model,
            "run_finished": run.finished, "kind": info["kind"], "round": info["round"], "plan": info["plan"], "base": base,
            "n_attempts": len([s for s in seq if s["status"] != "pending"]), "first_status": first["status"],
            "final_status": last["status"] if last["status"] != "pending" else (seq[-2]["status"] if len(seq) > 1 else "pending"),
            "open": bool(open_) or first["status"] == "pending",
            "needed_repair": first["status"] not in ("ok", "pending"),
            "repaired": first["status"] not in ("ok", "pending") and final_c is not None,
            "ok_final": final_c is not None, "accepted": bool(accepted),
            "faster_than_seed": bool(ratio is not None and ratio > MIN_IMPROVEMENT),
            "ratio": ratio, "gain_pct": max(0.0, (ratio - 1) * 100) if ratio is not None else 0.0,
            "seed": seed_label, "seed_speedup": seed_speedup,
            "diff_first": fc.diff_frac if fc is not None else None, "lines_first": fc.lines_changed if fc is not None else None,
            "bucket_first": bucket_of(fc.diff_frac, edges) if fc is not None else None,
            "plan_title": title, "truncated_turns": sum((tr.tags.get(s["tag"]) or {}).get("truncated", 0) for s in seq),
            "turns": sum((tr.tags.get(s["tag"]) or {}).get("turns", 0) for s in seq),
            "llm_usd": sum((tr.tags.get(s["tag"]) or {}).get("cost", 0.0) for s in seq) if tr.present else None,
            "truncated_usd": sum((tr.tags.get(s["tag"]) or {}).get("truncated_cost", 0.0) for s in seq) if tr.present else None,
            "no_code_reasons": [s.get("reason") for s in seq if s["status"] == "no_code"],
            "plan_kind": pk["kind"], "plan_scope": pk["scope"], "plan_scores": pk["scores"], "plan_local_score": pk["local_score"], "plan_declared_local": pk["declared_local"],
            "plan_fallback": fallback, "plan_chars": len(plan.get("change") or "") or None, "plan_head": (plan.get("change") or "")[:200] or None,
            "prompt_tokens_first": ft.get("prompt"), "reasoning_first": ft.get("reasoning"), "completion_first": ft.get("completion"), "finish_first": ft.get("finish"),
            "first_cause": first.get("cause") if first["status"] == "no_code" else None,
            "first_nocode_cost_usd": (tr.tags.get(base) or {}).get("cost") if first["status"] == "no_code" and tr.present else None,
        }
        branches.append(rec)
        for s in seq:
            c = s["cand"]
            attempts.append({
                "run": run.label, "source": run.source, "task": run.base_task, "variant": run.task, "kind": info["kind"], "round": info["round"], "plan": info["plan"],
                "base": base, "attempt": s["attempt"], "label": s["tag"], "status": s["status"], "idx": c.idx if c else None,
                "speedup": c.speedup if c else None, "improved": bool(c.improved) if c else False, "seed": c.seed_label if c else None,
                "ratio": c.ratio if c else None, "diff_frac": c.diff_frac if c else None, "lines_changed": c.lines_changed if c else None,
                "bucket": bucket_of(c.diff_frac, edges) if c else None, "detail": c.bound_detail if c else "", "incorrect_kind": c.incorrect_kind if c else "",
                "reason": s.get("reason", ""), "cause": s.get("cause", ""), "run_finished": run.finished,
            })
    # roles in the round: designated local plan (plan_mix runs) / last position (control)
    for rec in branches:
        rec["k"] = run.k
        rec["role"], rec["is_last"] = None, False
    rounds: dict[int, list[dict]] = defaultdict(list)
    for rec in branches:
        if rec["kind"] == "patch":
            rounds[rec["round"]].append(rec)
    for r, recs in rounds.items():
        k = (tr.plan_rounds.get(r) or {}).get("k") or run.k or (max(x["plan"] for x in recs) + 1)
        have = {x["plan"] for x in recs}
        full = have == set(range(k)) and k >= 2
        fallback = any((x["plan_title"] or "") in FALLBACK_TITLES for x in recs)
        for x in recs:
            x["k"] = k
            x["is_last"] = bool(full and x["plan"] == k - 1)
            if run.plan_mix and full and not fallback:
                x["role"] = "local" if x["plan"] == k - 1 else "structural"
            elif run.plan_mix:
                x["role"] = "unknown"
    return branches, attempts


# ----------------------------------------------------------------------------------------------------------------------
# Question 1 tables
# ----------------------------------------------------------------------------------------------------------------------
def status_counts(recs: list[dict], key: str) -> Counter:
    return Counter(r[key] for r in recs)


def group_by(recs: list[dict], key) -> dict:
    out: dict = defaultdict(list)
    for r in recs:
        out[key(r)].append(r)
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def task_order(tasks) -> list[str]:
    pri = {"flash_attention": 0, "gemm": 1, "conv2d": 2}
    return sorted(tasks, key=lambda t: (pri.get(t, 3), t))


def present_statuses(recs: list[dict], key: str) -> list[str]:
    present = {r[key] for r in recs} - {"pending"}
    return [s for s in STATUS_ORDER if s in present] + sorted(present - set(STATUS_ORDER))


def outcome_table(branches: list[dict], key: str, label_fn=lambda x: x, order=None) -> tuple[list[str], list[list]]:
    cols = present_statuses(branches, key)
    rows = []
    for g, recs in group_by(branches, label_fn).items():
        cnt = status_counts(recs, key)
        n = len(recs)
        rows.append([g, n, *[cnt.get(s, 0) for s in cols], f"{pct(n - cnt.get('ok', 0), n)}" + ("†" if n < SMALL_N else "")])
    if order:
        rows.sort(key=lambda r: order.get(r[0], 99))
    return ["group", "n", *cols, "fail %"], rows


def gain_stats(recs: list[dict]) -> dict:
    n = len(recs)
    ratios = [r["ratio"] for r in recs if r["ratio"] is not None]
    diffs = [r["diff_first"] for r in recs if r["diff_first"] is not None]
    has_llm = any(r["llm_usd"] is not None for r in recs)
    return {
        "n": n, "ok_first": sum(r["first_status"] == "ok" for r in recs), "ok_final": sum(r["ok_final"] for r in recs),
        "accepted": sum(r["accepted"] for r in recs), "faster_than_seed": sum(r["faster_than_seed"] for r in recs),
        "median_ratio_ok": median(ratios), "max_ratio": max(ratios) if ratios else None,
        "mean_gain_pct": (sum(r["gain_pct"] for r in recs) / n) if n else None, "median_diff": median(diffs), "n_diff": len(diffs),
        "no_code_first": sum(r["first_status"] == "no_code" for r in recs),
        "llm_usd": sum(r["llm_usd"] for r in recs if r["llm_usd"] is not None) if has_llm else None,
        "usd_per_branch": (sum(r["llm_usd"] for r in recs if r["llm_usd"] is not None) / n) if (has_llm and n) else None,
    }


def compare_cell(g: dict, small_n: int) -> list:
    return [
        g["n"], rate_cell(g["ok_first"], g["n"], small_n), rate_cell(g["ok_final"], g["n"], small_n), rate_cell(g["accepted"], g["n"], small_n),
        rate_cell(g["faster_than_seed"], g["n"], small_n), fnum(g["median_ratio_ok"], 3), fnum(g["max_ratio"], 3),
        fnum(g["mean_gain_pct"], 2, "%"), fnum(g["median_diff"], 2),
    ]


COMPARE_HDR = ["group", "n branches", "first attempt ok", "ok after repair", "new best", "faster than seed", "median ratio (ok)", "max ratio", "mean gain/branch", "median diff frac"]


def size_block(recs: list[dict], edges) -> dict:
    """Per bucket counts of attempt records that have a diff: n, ok, new best, faster than the seed (ratio > min_improvement), medians."""
    out = {}
    for b in BUCKETS:
        rs = [a for a in recs if bucket_of(a["diff_frac"], edges) == b]
        ratios = [a["ratio"] for a in rs if a["ratio"] is not None]
        out[b] = {
            "n": len(rs), "ok": sum(a["status"] == "ok" for a in rs), "new_best": sum(bool(a["improved"]) for a in rs),
            "faster": sum(1 for r in ratios if r > MIN_IMPROVEMENT), "median_diff": median([a["diff_frac"] for a in rs]), "median_ratio": median(ratios),
        }
    return out


def size_rows(blk: dict, small_n: int) -> list[list]:
    return [[b, d["n"], fnum(d["median_diff"], 2), rate_cell(d["ok"], d["n"], small_n, True), rate_cell(d["new_best"], d["n"], small_n, True), rate_cell(d["faster"], d["n"], small_n), fnum(d["median_ratio"], 3)] for b, d in blk.items()]


SIZE_HDR = ["bucket", "n", "median diff", "ok", "new best", "faster than seed", "median ratio vs seed (ok)"]


def sensitivity(first: list[dict]) -> list[dict]:
    """The small-vs-large comparison of first attempts under several bucket schemes (the answer must not hinge on one choice of edges)."""
    xs = sorted(a["diff_frac"] for a in first)
    schemes = [(f"fixed {e[0]:.2f} / {e[1]:.2f}", e) for e in SENSITIVITY_EDGES]
    if len(xs) >= 6:
        schemes.append(("terciles of this pool", (xs[len(xs) // 3], xs[2 * len(xs) // 3])))
    out = []
    for name, e in schemes:
        blk = size_block(first, e)
        s, l = blk["small"], blk["large"]
        out.append({
            "scheme": name, "edges": list(e), "blocks": blk,
            "p_ok": fisher_exact(s["ok"], s["n"] - s["ok"], l["ok"], l["n"] - l["ok"]) if s["n"] and l["n"] else None,
            "p_new_best": fisher_exact(s["new_best"], s["n"] - s["new_best"], l["new_best"], l["n"] - l["new_best"]) if s["n"] and l["n"] else None,
        })
    return out


def diff_by_status(first: list[dict]) -> dict:
    ok = [a["diff_frac"] for a in first if a["status"] == "ok"]
    bad = [a["diff_frac"] for a in first if a["status"] != "ok"]
    by = {s: {"n": sum(a["status"] == s for a in first), "median_diff": median([a["diff_frac"] for a in first if a["status"] == s])} for s in present_statuses(first, "status")}
    return {"by_status": by, "n_ok": len(ok), "n_fail": len(bad), "median_ok": median(ok), "median_fail": median(bad), "p_mwu": mann_whitney_p(ok, bad)}


def round_streaks(patch_b: list[dict]) -> dict:
    """Over completed rounds of each run: P(new best in a round) overall, after one and after two rounds without a new best;
    rounds in which no first attempt was ok."""
    tot = won = a1 = a1w = a2 = a2w = no_ok = 0
    for _, recs in group_by(patch_b, lambda b: b["run"]).items():
        seq = []
        for r, rs in group_by(recs, lambda b: b["round"]).items():
            if any(b["open"] for b in rs):
                continue
            seq.append((r, any(b["accepted"] for b in rs)))
            no_ok += all(b["first_status"] != "ok" for b in rs)
        seq.sort()
        for i, (_, w) in enumerate(seq):
            tot += 1
            won += w
            if i >= 1 and not seq[i - 1][1]:
                a1 += 1
                a1w += w
            if i >= 2 and not seq[i - 1][1] and not seq[i - 2][1]:
                a2 += 1
                a2w += w
    return {"rounds": tot, "won": won, "after_1_no_win": a1, "after_1_won": a1w, "after_2_no_win": a2, "after_2_won": a2w, "rounds_no_ok_first": no_ok}


def q1_tables(patch_b: list[dict], init_b: list[dict], patch_a: list[dict], edges, small_n: int) -> dict:
    is_fa = lambda r: r["task"] == "flash_attention"
    out: dict = {"edges": list(edges)}
    out["gain_all"], out["gain_FA"], out["gain_nonFA"] = gain_stats(patch_b), gain_stats([b for b in patch_b if is_fa(b)]), gain_stats([b for b in patch_b if not is_fa(b)])
    out["per_task"] = {}
    for t, recs in group_by(patch_b, lambda b: b["task"]).items():
        cl = [b for b in recs if not b["open"]]
        out["per_task"][t] = {
            "n": len(recs), "first_status": dict(status_counts(recs, "first_status")), "n_closed": len(cl), "final_status": dict(status_counts(cl, "final_status")),
            "needed_repair": sum(b["needed_repair"] for b in cl), "repaired": sum(b["repaired"] for b in cl), "accepted": sum(b["accepted"] for b in recs),
            "init_first_status": dict(status_counts([b for b in init_b if b["task"] == t], "first_status")),
        }
    out["per_task_round"] = {t: {r: {"n": len(rs), "fail": sum(b["first_status"] != "ok" for b in rs), "accepted": sum(b["accepted"] for b in rs)} for r, rs in group_by(recs, lambda b: b["round"]).items()} for t, recs in group_by(patch_b, lambda b: b["task"]).items()}
    out["per_run_round"] = {run: {r: {"n": len(rs), "fail": sum(b["first_status"] != "ok" for b in rs), "accepted": sum(b["accepted"] for b in rs)} for r, rs in group_by(recs, lambda b: b["round"]).items()} for run, recs in group_by(patch_b, lambda b: b["run"]).items()}
    out["per_run"] = {run: gain_stats(recs) for run, recs in group_by(patch_b, lambda b: b["run"]).items()}
    # size of change
    withdiff = [a for a in patch_a if a["diff_frac"] is not None and a["status"] != "pending"]
    pools = {"FA": [a for a in withdiff if is_fa(a)], "nonFA": [a for a in withdiff if not is_fa(a)], "all": withdiff}
    out["size"], out["sensitivity"], out["diff_by_status"], out["deciles"] = {}, {}, {}, {}
    for name, recs in pools.items():
        first = [a for a in recs if a["attempt"] == 0]
        out["size"][name] = {"first": size_block(first, edges), "all": size_block(recs, edges), "first_from_r2": size_block([a for a in first if a["round"] >= 2], edges)}
        out["sensitivity"][name] = sensitivity(first)
        out["diff_by_status"][name] = diff_by_status(first)
        out["deciles"][name] = deciles([a["diff_frac"] for a in recs])
    out["accepted_by_bucket"] = {name: dict(Counter(bucket_of(a["diff_frac"], edges) for a in recs if a["improved"])) for name, recs in pools.items()}
    # failure causes of patch attempts
    causes: dict = defaultdict(list)
    for a in patch_a:
        if a["status"] in ("ok", "pending"):
            continue
        if a["status"] == "incorrect":
            key = (is_fa(a), "incorrect:" + (a["incorrect_kind"] or "?"), "")
        elif a["status"] == "no_code":
            key = (is_fa(a), "no_code:" + (a["reason"] or "?"), "")
        else:
            key = (is_fa(a), a["status"], a["detail"])
        causes[key].append(a["diff_frac"])
    out["failure_causes"] = sorted(({"fa": k[0], "status": k[1], "detail": k[2], "n": len(v), "median_diff": median([x for x in v if x is not None])} for k, v in causes.items()), key=lambda c: -c["n"])
    out["truncation"] = {
        t: {"branches": len(recs), "turns": sum(b["turns"] for b in recs), "truncated_turns": sum(b["truncated_turns"] for b in recs), "with_truncated_turn": sum(b["truncated_turns"] > 0 for b in recs),
            "truncated_usd": sum(b["truncated_usd"] or 0.0 for b in recs), "llm_usd": sum(b["llm_usd"] or 0.0 for b in recs)}
        for t, recs in group_by(patch_b, lambda b: b["task"]).items()
    }
    # plan_mix: designated local plan vs structural plans, and the position control in runs without plan_mix
    mixed = [b for b in patch_b if b["role"] in ("local", "structural")]
    per_run = {run: {"local": gain_stats([b for b in recs if b["role"] == "local"]), "structural": gain_stats([b for b in recs if b["role"] == "structural"])} for run, recs in group_by(mixed, lambda b: b["run"]).items()}
    strata = [(d["local"]["ok_first"], d["local"]["n"] - d["local"]["ok_first"], d["structural"]["ok_first"], d["structural"]["n"] - d["structural"]["ok_first"]) for d in per_run.values()]
    out["mix"] = {
        "runs": sorted(per_run), "local": gain_stats([b for b in mixed if b["role"] == "local"]), "structural": gain_stats([b for b in mixed if b["role"] == "structural"]),
        "local_FA": gain_stats([b for b in mixed if b["role"] == "local" and is_fa(b)]), "structural_FA": gain_stats([b for b in mixed if b["role"] == "structural" and is_fa(b)]),
        "per_run": per_run, "cmh_first_ok": cmh(strata), "unknown_role": sum(b["role"] == "unknown" for b in patch_b),
        "accepted_by_role": dict(Counter(b["role"] for b in mixed if b["accepted"])),
        "first_status_by_role": {role: dict(status_counts([b for b in mixed if b["role"] == role], "first_status")) for role in ("local", "structural")},
    }
    nonmix_runs = {b["run"] for b in patch_b if b["role"] is None}
    cl = [b for b in patch_b if b["run"] in nonmix_runs and b["k"] and b["k"] >= 2 and b["role"] is None]
    out["control_last"] = {
        "last": gain_stats([b for b in cl if b["is_last"]]), "earlier": gain_stats([b for b in cl if not b["is_last"]]),
        "last_FA": gain_stats([b for b in cl if b["is_last"] and is_fa(b)]), "earlier_FA": gain_stats([b for b in cl if not b["is_last"] and is_fa(b)]),
    }
    out["streaks"] = {"FA": round_streaks([b for b in patch_b if is_fa(b)]), "nonFA": round_streaks([b for b in patch_b if not is_fa(b)])}
    return out


def deciles(xs: list[float]) -> list[float]:
    if not xs:
        return []
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]


# ----------------------------------------------------------------------------------------------------------------------
# Plan kind vs outcome (report section 1) and the replies without code (report section 2)
# ----------------------------------------------------------------------------------------------------------------------
CORE_STATUSES = ("ok", "abi_error", "incorrect", "no_code")


def status_split(recs: list[dict], key: str = "first_status") -> dict:
    """Counts of ok / abi_error / incorrect / no_code (everything else is ``other``) of the ``key`` status of ``recs``."""
    c = Counter(r[key] for r in recs)
    out = {s: c.get(s, 0) for s in CORE_STATUSES}
    out["other"] = sum(v for k, v in c.items() if k not in CORE_STATUSES)
    out["n"] = len(recs)
    return out


def kind_block(recs: list[dict]) -> dict:
    g = gain_stats(recs)
    return {**status_split(recs), "ok_final": g["ok_final"], "accepted": g["accepted"], "faster_than_seed": g["faster_than_seed"],
            "mean_gain_pct": g["mean_gain_pct"], "median_ratio_ok": g["median_ratio_ok"], "llm_usd": g["llm_usd"]}


def local_vs_structural(recs: list[dict], pred, restrict=None) -> dict:
    """2x2 comparison (Fisher exact) of ``pred`` between the local and the structural plans of ``recs`` (``restrict`` filters first)."""
    rs = [b for b in recs if restrict is None or restrict(b)]
    lo = [b for b in rs if b["plan_scope"] == "local"]
    st = [b for b in rs if b["plan_scope"] == "structural"]
    a, c = sum(map(bool, map(pred, lo))), sum(map(bool, map(pred, st)))
    return {"local": [a, len(lo)], "structural": [c, len(st)], "p": fisher_exact(a, len(lo) - a, c, len(st) - c) if lo and st else None}


def kind_tables(patch_b: list[dict], patch_a: list[dict]) -> dict:
    out: dict = {}
    known = [b for b in patch_b if b["plan_kind"] != "unknown"]
    # (a) does the keyword classifier recover the plans the prompt itself designated as local? (plan_mix runs, rounds with all K plans, no fallback)
    mixed = [b for b in known if b["role"] in ("local", "structural")]
    dis = [{"run": b["run"], "base": b["base"], "role": b["role"], "kind": b["plan_kind"], "title": b["plan_title"]} for b in mixed if (b["role"] == "local") != (b["plan_scope"] == "local")]
    out["validation"] = {"n": len(mixed), "agree": len(mixed) - len(dis), "disagreements": dis,
                         "confusion": {f"{r}->{s}": v for (r, s), v in sorted(Counter((b["role"], b["plan_scope"]) for b in mixed).items())}}
    # (b) what the default prompt ("each a concrete structural change, not a micro-tweak") yields in the runs that used it
    default = [b for b in known if b["role"] is None]
    out["default_prompt"] = {g: {"n": len(rs), "scope": dict(Counter(b["plan_scope"] for b in rs)), "kind": dict(Counter(b["plan_kind"] for b in rs))}
                             for g, rs in group_by(default, lambda b: task_group(b["task"])).items()}
    out["default_prompt_runs"] = sorted({b["run"] for b in default})
    # (c) first-attempt outcome by task and plan scope / kind
    out["by_task_scope"] = {t: {sc: kind_block(rs) for sc, rs in group_by(recs, lambda b: b["plan_scope"]).items()} for t, recs in group_by(known, lambda b: b["task"]).items()}
    out["by_group_scope"] = {g: {sc: kind_block(rs) for sc, rs in group_by(recs, lambda b: b["plan_scope"]).items()} for g, recs in group_by(known, lambda b: task_group(b["task"])).items()}
    out["by_task_kind"] = {t: {k: kind_block(rs) for k, rs in group_by(recs, lambda b: b["plan_kind"]).items()} for t, recs in group_by(known, lambda b: b["task"]).items()}
    out["by_scope_all"] = {sc: kind_block(rs) for sc, rs in group_by(known, lambda b: b["plan_scope"]).items()}
    # (d) tests: local vs structural (generic fallback plans and unclassified ones are left out)
    fa = [b for b in known if b["task"] == "flash_attention"]
    preds = {"ok": lambda b: b["first_status"] == "ok", "abi_error": lambda b: b["first_status"] == "abi_error", "incorrect": lambda b: b["first_status"] == "incorrect",
             "no_code": lambda b: b["first_status"] == "no_code"}
    out["tests_fa"] = {k: local_vs_structural(fa, f) for k, f in preds.items()}
    out["tests_fa"]["ok given code"] = local_vs_structural(fa, preds["ok"], lambda b: b["first_status"] != "no_code")
    out["tests_fa"]["new best"] = local_vs_structural(fa, lambda b: b["accepted"])
    out["tests_all"] = {k: local_vs_structural(known, f) for k, f in preds.items()}
    out["tests_all"]["ok given code"] = local_vs_structural(known, preds["ok"], lambda b: b["first_status"] != "no_code")
    strata = []
    for _, recs in group_by(known, lambda b: b["run"]).items():
        lo = [b for b in recs if b["plan_scope"] == "local" and b["first_status"] != "no_code"]
        st = [b for b in recs if b["plan_scope"] == "structural" and b["first_status"] != "no_code"]
        strata.append((sum(b["first_status"] == "ok" for b in lo), sum(b["first_status"] != "ok" for b in lo), sum(b["first_status"] == "ok" for b in st), sum(b["first_status"] != "ok" for b in st)))
    out["cmh_ok_given_code"] = cmh(strata)
    # (e) ABI failures of FlashAttention first attempts: which plan kinds, and is it the threadgroup-memory budget?
    kind_of = {(b["run"], b["base"]): b["plan_kind"] for b in known}
    abi = [a for a in patch_a if a["task"] == "flash_attention" and a["attempt"] == 0 and a["status"] == "abi_error"]
    out["fa_abi_by_kind"] = {k: {"abi": len(v), "threadgroup_memory": sum(x["detail"].startswith("static threadgroup memory") for x in v)} for k, v in group_by(abi, lambda a: kind_of.get((a["run"], a["base"]), "unknown")).items()}
    return out


def turn_group(tag: str) -> str:
    """plan / init / repair / patch: what a logged chat turn belongs to."""
    if PLAN_TAG_RE.match(tag):
        return "plan"
    if ".fix" in tag:
        return "repair"
    return "init" if tag.startswith("init") else "patch"


def turn_round(tag: str) -> int | None:
    m = PLAN_TAG_RE.match(tag)
    if m:
        return int(m.group(1))
    info = parse_label(tag)
    return info["round"] if info else None


def dist(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    return {"n": len(xs), "p50": xs[len(xs) // 2], "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))], "max": xs[-1]}


def within_run_ranks(recs: list[dict], key: str) -> dict[int, float]:
    """id(rec) -> percentile rank (0..1, ties averaged) of ``rec[key]`` among the records of its own run: removes what differs between runs
    (task, model, protocol), so that pooled comparisons ask 'within a run, are truncated requests the ones with the longer plan / bigger seed?'."""
    out: dict[int, float] = {}
    for _, rs in group_by([r for r in recs if r.get(key) is not None], lambda r: r["run"]).items():
        vals = sorted(r[key] for r in rs)
        n = len(vals)
        for r in rs:
            lo, hi = bisect.bisect_left(vals, r[key]), bisect.bisect_right(vals, r[key])
            out[id(r)] = ((lo + hi - 1) / 2 + 0.5) / n
    return out


def nocode_tables(runs: list[Run], branches: list[dict], small_n: int) -> dict:
    out: dict = {}
    rows = []
    for r in runs:
        model = r.transcript.model or r.model
        for t in r.transcript.turns:
            rows.append({**t, "run": r.label, "task": r.base_task, "group": task_group(r.base_task), "model": model, "tg": turn_group(t["tag"]), "round": turn_round(t["tag"])})
    lengths = [t for t in rows if t["finish"] == "length"]
    cap = median([t["completion"] for t in lengths if t["completion"]])
    out["cap_tokens"] = cap
    out["turns"], out["length_turns"] = len(rows), len(lengths)
    out["length_cost"], out["total_cost"] = sum(t["cost"] for t in lengths), sum(t["cost"] for t in rows)
    # what a truncated reply looks like, against the replies that carry the kernel
    reas_share = [t["reasoning"] / t["completion"] for t in lengths if t["completion"] and t["reasoning"] is not None]
    code_replies = [t for t in rows if t["finish"] == "stop" and t["has_fence"] and t["tg"] in ("init", "patch", "repair") and t["completion"] and t["reasoning"] is not None]
    visible = [t["completion"] - t["reasoning"] for t in code_replies]
    long_done = [t for t in rows if t["finish"] != "length" and t["reasoning"] is not None and cap and t["reasoning"] >= cap / 2]
    short_done = [t for t in rows if t["finish"] != "length" and t["reasoning"] is not None and cap and t["reasoning"] < cap / 2]
    out["mechanism"] = {
        "truncated": {"n": len(lengths), "completion": dist([t["completion"] for t in lengths]), "reasoning_share_median": median(reas_share),
                      "reasoning_share_min": min(reas_share) if reas_share else None, "pure_thinking": sum(1 for t in lengths if t["content_chars"] == 0 and not t["tool_calls"]),
                      "with_visible_output": sum(1 for t in lengths if t["content_chars"] > 0 or t["tool_calls"]), "fences_median": median([t["reasoning_fences"] for t in lengths])},
        "code_replies": {"n": len(code_replies), "visible_tokens": dist(visible)},
        "reasoning_ge_half_cap_not_truncated": {"n": len(long_done), "fences_median": median([t["reasoning_fences"] for t in long_done])},
        "reasoning_lt_half_cap": {"n": len(short_done), "fences_median": median([t["reasoning_fences"] for t in short_done])},
    }
    # reasoning-token distribution of the first turn of each request, by model and task group
    first = [t for t in rows if t["turn"] == 0 and t["tg"] in ("patch", "plan", "init") and t["reasoning"] is not None]
    dd = []
    for (model, grp, tg), rs in sorted(group_by(first, lambda t: (t["model"], t["group"], t["tg"])).items()):
        d = dist([t["reasoning"] for t in rs])
        dd.append({"model": model, "group": grp, "tg": tg, **d, "length": sum(t["finish"] == "length" for t in rs), "ge_half": sum(1 for t in rs if cap and t["reasoning"] >= cap / 2)})
    out["first_turn_reasoning"] = dd
    # per run
    per_run = []
    for r in runs:
        rt = [t for t in rows if t["run"] == r.label]
        if not rt:
            continue
        pb = [b for b in branches if b["run"] == r.label and b["kind"] == "patch" and b["first_status"] != "pending"]
        nc = [b for b in pb if b["first_status"] == "no_code"]
        per_run.append({
            "run": r.label, "task": r.base_task, "model": r.transcript.model or r.model, "finished": r.finished, "turns": len(rt), "length_turns": sum(t["finish"] == "length" for t in rt),
            "length_usd": sum(t["cost"] for t in rt if t["finish"] == "length"), "llm_usd": sum(t["cost"] for t in rt),
            "patch_requests": len(pb), "no_code_first": len(nc), "causes": dict(Counter(b["first_cause"] or "gap" for b in nc)),
        })
    out["per_run"] = per_run
    # candidate explanations, on the first turn of patch requests (FlashAttention, same model: deepseek-flash)
    pools = {"flash_attention (deepseek-flash)": lambda b: b["task"] == "flash_attention" and "flash" in (b["model"] or ""), "all tasks (deepseek-flash)": lambda b: "flash" in (b["model"] or "")}
    out["predictors"] = {}
    for name, sel in pools.items():
        rs = [b for b in branches if b["kind"] == "patch" and b["finish_first"] is not None and sel(b)]
        tr_, ok_ = [b for b in rs if b["finish_first"] == "length"], [b for b in rs if b["finish_first"] != "length"]
        feats = {}
        for key, label in (("plan_chars", "plan text (characters)"), ("prompt_tokens_first", "prompt (tokens: system + seed kernel + feedback + plan)"), ("round", "round")):
            xt, xo = [b[key] for b in tr_ if b[key] is not None], [b[key] for b in ok_ if b[key] is not None]
            rk = within_run_ranks(rs, key)
            rt, ro = [rk[id(b)] for b in tr_ if id(b) in rk], [rk[id(b)] for b in ok_ if id(b) in rk]
            feats[label] = {"truncated_median": median(xt), "other_median": median(xo), "p": mann_whitney_p(xt, xo),
                            "rank_truncated": median(rt), "rank_other": median(ro), "p_within_run": mann_whitney_p(rt, ro)}
        rb = {}
        for lo_, hi_, lab in ((1, 2, "rounds 1-2"), (3, 4, "rounds 3-4"), (5, 99, "rounds 5+")):
            sub = [b for b in rs if lo_ <= b["round"] <= hi_]
            rb[lab] = [sum(b["finish_first"] == "length" for b in sub), len(sub)]
        sc = {s: [sum(1 for b in tr_ if b["plan_scope"] == s), sum(1 for b in rs if b["plan_scope"] == s)] for s in ("local", "structural", "generic")}
        lo_, st_, ge_ = sc["local"], sc["structural"], sc["generic"]
        rest = [sum(sc[s][0] for s in ("local", "structural")), sum(sc[s][1] for s in ("local", "structural"))]
        out["predictors"][name] = {
            "n": len(rs), "truncated": len(tr_), "features": feats, "by_round": rb, "by_scope": sc,
            "p_local_vs_structural": fisher_exact(lo_[0], lo_[1] - lo_[0], st_[0], st_[1] - st_[0]) if lo_[1] and st_[1] else None,
            "p_generic_vs_planned": fisher_exact(ge_[0], ge_[1] - ge_[0], rest[0], rest[1] - rest[0]) if ge_[1] and rest[1] else None,
        }
    # FlashAttention: the other model on the same task and prompts
    pro = [b for b in branches if b["kind"] == "patch" and b["task"] == "flash_attention" and b["finish_first"] is not None and "pro" in (b["model"] or "")]
    out["fa_pro"] = {"requests": len(pro), "truncated_first_turns": sum(b["finish_first"] == "length" for b in pro), "no_code_first": sum(b["first_status"] == "no_code" for b in pro)}
    # what the next request looks like after a truncated one (the truncated reasoning is echoed back: llm.echo_reasoning_content)
    before, after = [], []
    for r in runs:
        tags = r.transcript.tags
        for tag, t in tags.items():
            lt, ft = t.get("last_turn"), t.get("first_turn")
            if not lt or lt["finish"] != "length" or turn_group(tag) not in ("patch", "repair") or not ft or ft["prompt"] is None:
                continue
            m = re.match(r"^(.*?)(?:\.fix(\d+))?$", tag)
            nxt = f"{m.group(1)}.fix{int(m.group(2) or 0) + 1}"
            nf = (tags.get(nxt) or {}).get("first_turn")
            if nf and nf["prompt"] is not None:
                before.append(ft["prompt"])
                after.append(nf["prompt"])
    out["repair_after_truncation"] = {"n": len(before), "prompt_before": median(before), "prompt_after": median(after)}
    # recovery: patch requests whose first attempt had no code because the model ran out of thinking budget
    pb = [b for b in branches if b["kind"] == "patch" and b["first_status"] not in ("pending",)]

    def recovery(sel) -> dict:
        rc = [b for b in pb if b["first_cause"] == "reasoning_cap" and sel(b)]
        okb = [b for b in pb if b["first_status"] == "ok" and b["llm_usd"] is not None and sel(b)]
        return {"n": len(rc), "ok_final": sum(b["ok_final"] for b in rc), "accepted": sum(b["accepted"] for b in rc),
                "usd_median": median([b["llm_usd"] for b in rc if b["llm_usd"] is not None]), "usd_median_ok_first": median([b["llm_usd"] for b in okb]),
                "usd_total": sum(b["llm_usd"] for b in rc if b["llm_usd"] is not None)}

    out["recovery"] = recovery(lambda b: True)
    out["recovery_fa"] = recovery(lambda b: b["task"] == "flash_attention")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Question 2
# ----------------------------------------------------------------------------------------------------------------------
def q2_row(run: Run, sat: float) -> dict:
    by_idx = run.by_idx
    order = [by_idx[i] for i in run.order]
    res = run.result or {}
    conf = run.confirm or {}
    best_label = res.get("best_label")
    best = next((c for c in run.cands if c.label == best_label), None)
    if best is None:
        okc = [c for c in run.cands if c.eligible]
        best = max(okc, key=lambda c: c.speedup) if okc else None
    roof = conf.get("roofline") or (best.roofline if best else {}) or {}
    frac = roofline_fraction(roof)
    cand_us = conf.get("candidate_us") or (best.candidate_us if best else None)
    base_us = conf.get("baseline_us") or res.get("baseline_us") or (best.baseline_us if best else {}) or {}
    best_base = min(base_us.values()) if base_us else None
    lb = roof.get("lower_bound_us")
    speed = res.get("confirmed_speedup") or res.get("best_speedup") or (best.speedup if best else None)
    improved = [c for c in order if c.improved]
    last_imp = improved[-1] if improved else None
    spent = run.spent_usd
    sat_first = next((c for c in order if c.status == "ok" and (c.roof_frac or 0) >= sat), None)

    def after(c: Cand | None):
        return None if (c is None or c.cost_usd is None or spent is None) else max(0.0, spent - c.cost_usd)

    def rounds_after(c: Cand | None):
        return None if c is None else max(0, (run.rounds or 0) - c.round)

    ceiling_us = min(x for x in (lb, best_base) if x) if (lb or best_base) else None
    return {
        "run": run.label, "source": run.source, "task": run.task, "dtype": run.dtype, "model": run.model, "finished": run.finished, "stop_kind": run.stop_kind, "stop_reason": run.stop_reason,
        "speedup": speed, "best_label": best.label if best else None, "bound": roof.get("bound"), "roof_frac": frac,
        "bw_frac": roof.get("bandwidth_fraction"), "compute_frac": roof.get("compute_fraction"), "achieved_gbps": roof.get("achieved_gbps"), "achieved_tflops": roof.get("achieved_tflops"),
        "peak_gbps": roof.get("peak_gbps_measured"), "peak_tflops": roof.get("peak_tflops_measured"),
        "headroom_pct": max(0.0, (1 / frac - 1) * 100) if frac else None,
        "baseline_roof_frac": (lb / best_base) if (lb and best_base) else None,
        "ceiling_vs_baseline": (best_base / lb) if (lb and best_base) else None,
        "eff_frac": (ceiling_us / cand_us) if (ceiling_us and cand_us) else None,
        "candidate_us": cand_us, "best_baseline_us": best_base, "lower_bound_us": lb,
        "rounds": run.rounds, "candidates": len(run.cands), "budget_usd": run.budget_usd, "spent_usd": spent,
        "last_improvement": last_imp.label if last_imp else None, "last_improvement_round": last_imp.round if last_imp else None,
        "spent_after_last_improvement": after(last_imp), "rounds_after_last_improvement": rounds_after(last_imp),
        "saturated": bool(frac is not None and frac >= sat), "first_sat": sat_first.label if sat_first else None, "first_sat_round": sat_first.round if sat_first else None,
        "first_sat_frac": sat_first.roof_frac if sat_first else None, "spent_at_first_sat": sat_first.cost_usd if sat_first else None,
        "spent_after_first_sat": after(sat_first), "rounds_after_first_sat": rounds_after(sat_first),
    }


def q2_aggregate(rows: list[dict]) -> dict:
    fin = [r for r in rows if r["finished"] and r["spent_usd"] is not None]
    total = sum(r["spent_usd"] for r in fin)
    sat = [r for r in fin if r["saturated"]]
    sat_after = sum(r["spent_after_first_sat"] or 0 for r in sat)
    nogain = sum(r["spent_after_last_improvement"] or 0 for r in fin)
    sat_total = sum(r["spent_usd"] for r in sat)
    return {
        "n_finished": len(fin), "n_saturated": len(sat), "total_usd": total, "saturated_total_usd": sat_total,
        "saturated_share_of_total": sat_total / total if total else None,
        "spent_after_first_sat_usd": sat_after, "share_of_total_after_first_sat": sat_after / total if total else None,
        "share_of_saturated_after_first_sat": sat_after / sat_total if sat_total else None,
        "spent_after_last_improvement_usd": nogain, "share_of_total_after_last_improvement": nogain / total if total else None,
        "saturated_spent_after_last_improvement_usd": sum(r["spent_after_last_improvement"] or 0 for r in sat),
        "n_saturated_at_init": sum(1 for r in sat if r["first_sat_round"] == 0),
        "n_saturated_baseline_faster_than_roofline": sum(1 for r in sat if (r["ceiling_vs_baseline"] or 9) < 1.0),
        "n_saturated_within10_of_effective_ceiling": sum(1 for r in sat if (r["eff_frac"] or 0) >= 0.9),
        "stop_kinds": dict(Counter(r["stop_kind"] for r in fin)), "stop_kinds_saturated": dict(Counter(r["stop_kind"] for r in sat)),
    }


# ----------------------------------------------------------------------------------------------------------------------
# analysis driver
# ----------------------------------------------------------------------------------------------------------------------
def run_summary(run: Run) -> dict:
    return {
        "run": run.label, "source": run.source, "path": run.path, "task": run.task, "dtype": run.dtype, "model": run.model, "budget_usd": run.budget_usd,
        "spent_usd": run.spent_usd, "rounds": run.rounds, "finished": run.finished, "live": run.live, "stop_reason": run.stop_reason, "stop_kind": run.stop_kind,
        "plan_mix": run.plan_mix, "plan_mix_source": run.plan_mix_source, "k": run.k, "n_candidates": len(run.cands), "trace_source": run.trace_source,
        "transcript": run.transcript.present, "plan_usd": run.transcript.plan_cost, "plan_calls": run.transcript.plan_calls, "seed_check": run.seed_check, "unreadable_candidates": run.unreadable, "notes": run.notes,
    }


def analyse(runs: list[Run], *, edges: tuple[float, float] = DEFAULT_SIZE_EDGES, sat: float = SAT_FRACTION, small_n: int = SMALL_N) -> dict:
    branches, attempts = [], []
    for r in runs:
        b, a = build_branches(r, edges)
        branches += b
        attempts += a
    pending = [b for b in branches if b["first_status"] == "pending"]
    patch_b = [b for b in branches if b["kind"] == "patch" and b["first_status"] != "pending"]
    init_b = [b for b in branches if b["kind"] == "init" and b["first_status"] != "pending"]
    patch_a = [a for a in attempts if a["kind"] == "patch" and a["status"] != "pending"]
    q2 = [q2_row(r, sat) for r in runs]
    data = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "size_edges": list(edges), "sat_fraction": sat, "small_n": small_n, "n_runs": len(runs), "pending_branches_excluded": len(pending),
            "diff_definition": "fraction of changed lines between candidate and the seed it patched (comments/blank lines/whitespace removed): (removed + added) / (lines of both)",
        },
        "runs": [run_summary(r) for r in runs],
        "branches": branches,
        "attempts": attempts,
        "q2": {"rows": q2, "aggregate": q2_aggregate(q2), "aggregate_by_source": {s: q2_aggregate([r for r in q2 if r["source"] == s]) for s in sorted({r["source"] for r in q2})}},
    }
    data["q1"] = q1_tables(patch_b, init_b, patch_a, edges, small_n)
    data["kinds"] = kind_tables(patch_b, patch_a)
    data["nocode"] = nocode_tables(runs, branches, small_n)
    data["facts"] = facts(data, small_n)
    return data


def facts(data: dict, small_n: int) -> dict:
    """The numbers the conclusion rests on (kept in data.json so that the text can be checked against them)."""
    q1, q2 = data["q1"], data["q2"]
    sat = data["meta"]["sat_fraction"]
    f: dict = {}
    fa = q1["gain_FA"]
    causes = [c for c in q1["failure_causes"] if c["fa"]]
    abi = sum(c["n"] for c in causes if c["status"] == "abi_error")
    abi_tg = sum(c["n"] for c in causes if c["status"] == "abi_error" and c["detail"].startswith("static threadgroup memory"))
    fat = q1["truncation"].get("flash_attention", {})
    pt = q1["per_task"].get("flash_attention", {})
    fa_runs = [r for r in data["runs"] if r["task"].split("@")[0] == "flash_attention"]
    plan_calls = sum(r["plan_calls"] for r in fa_runs)
    f["fa"] = {
        "n": fa["n"], "ok_first": fa["ok_first"], "ok_final": fa["ok_final"], "accepted": fa["accepted"], "no_code_first": fa["no_code_first"],
        "first_status": pt.get("first_status", {}), "abi_attempts": abi, "abi_threadgroup_memory_attempts": abi_tg,
        "incorrect_marginal": sum(c["n"] for c in causes if c["status"] == "incorrect:marginal"), "incorrect_gross": sum(c["n"] for c in causes if c["status"] == "incorrect:gross"),
        "truncated_usd": fat.get("truncated_usd"), "llm_usd": fat.get("llm_usd"), "truncated_turns": fat.get("truncated_turns"), "turns": fat.get("turns"),
        "needed_repair": pt.get("needed_repair"), "repaired": pt.get("repaired"), "usd_per_branch": fa["usd_per_branch"],
        "plan_usd_per_call": (sum(r["plan_usd"] for r in fa_runs) / plan_calls) if plan_calls else None,
        "round_seeds": len({(b["run"], b["round"]) for b in data["branches"] if b["kind"] == "patch" and b["task"] == "flash_attention"}),
    }
    f["fa_size"] = {
        "default": q1["size"]["FA"]["first"], "diff_by_status": q1["diff_by_status"]["FA"], "other_diff_by_status": q1["diff_by_status"]["nonFA"], "other_default": q1["size"]["nonFA"]["first"],
        "sensitivity": [{"scheme": s["scheme"], "small": s["blocks"]["small"], "large": s["blocks"]["large"], "p_ok": s["p_ok"], "p_new_best": s["p_new_best"]} for s in q1["sensitivity"]["FA"]],
        "accepted_by_bucket": q1["accepted_by_bucket"]["FA"],
    }
    mx = q1["mix"]
    f["mix"] = {"local": mx["local"], "structural": mx["structural"], "local_FA": mx["local_FA"], "structural_FA": mx["structural_FA"], "cmh": mx["cmh_first_ok"], "runs": mx["runs"],
                "no_code_first": {role: v.get("no_code", 0) for role, v in mx["first_status_by_role"].items()}}
    f["control_FA"] = {"last": q1["control_last"]["last_FA"], "earlier": q1["control_last"]["earlier_FA"]}
    f["streaks"] = q1["streaks"]
    agg = q2["aggregate"]
    fin = [r for r in q2["rows"] if r["finished"]]
    sat_rows = [r for r in fin if r["saturated"]]
    eff = [r for r in sat_rows if (r["eff_frac"] or 0) >= sat]
    ceil = sorted(r["ceiling_vs_baseline"] for r in eff if r["ceiling_vs_baseline"])
    f["q2"] = {
        "aggregate": agg, "n_finished": len(fin), "n_saturated": len(sat_rows), "n_effectively_saturated": len(eff),
        "not_effectively_saturated": [{"run": r["run"], "roof_frac": r["roof_frac"], "eff_frac": r["eff_frac"], "speedup": r["speedup"], "baseline_roof_frac": r["baseline_roof_frac"]} for r in sat_rows if r not in eff],
        "ceiling_median": median(ceil), "ceiling_max": max(ceil) if ceil else None, "n_ceiling_le_1_02": sum(c <= 1.02 for c in ceil),
        "speedups_effective": sorted(r["speedup"] for r in eff if r["speedup"] is not None), "roof_fracs": sorted(r["roof_frac"] for r in sat_rows),
        "dtypes": dict(Counter(r["dtype"] or "?" for r in fin)),
    }
    return f


LOG_RUN_RE = re.compile(r"^==.*\|\s*run\s+(\S+)\s*$")
LOG_NOCODE_RE = re.compile(r"^\s*\[(?P<tag>[^\]]+)\]\s+no code block in reply")


def read_log(path: Path) -> list[tuple[str | None, str]]:
    """(run name, tag) of every ``[tag] no code block in reply`` line of an orchestrator log (the run name comes from the preceding ``== ... | run NAME`` header)."""
    run, out = None, []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        m = LOG_RUN_RE.match(line)
        if m:
            run = m.group(1)
            continue
        m = LOG_NOCODE_RE.match(line)
        if m:
            out.append((run, m.group("tag")))
    return out


def log_check(log_paths: list[str], attempts: list[dict]) -> list[dict]:
    """Compare the orchestrator's own 'no code block in reply' lines with the no-code attempts derived from the transcripts, per log file and run name
    (a run name is matched to the last path component of a run label; ambiguous names are skipped)."""
    out = []
    for lp in sorted({str(x) for pat in log_paths for x in (glob.glob(pat) or [pat])}):
        by_run: dict[str, list[str]] = defaultdict(list)
        for run, tag in read_log(Path(lp)):
            if run:
                by_run[run].append(tag)
        for run, tags in by_run.items():
            labels = {a["run"] for a in attempts if a["run"].split("/")[-1] == run}
            if len(labels) != 1:
                continue
            label = next(iter(labels))
            ours = {a["label"] for a in attempts if a["run"] == label and a["status"] == "no_code" and a.get("cause") not in ("run_ended", "unknown")}
            lg = set(tags)
            pending = {a["label"] for a in attempts if a["run"] == label and a["status"] == "pending"}
            out.append({"log": Path(lp).name, "run": label, "log_lines": sorted(lg), "matched": sorted(lg & ours), "only_in_log": sorted(lg - ours),
                        "only_in_log_pending": sorted((lg - ours) & pending), "only_in_transcript": sorted(ours - lg)})
    return out


def analyse_paths(paths: list[str], *, edges=DEFAULT_SIZE_EDGES, sat=SAT_FRACTION, small_n=SMALL_N, now: float | None = None, logs: list[str] | None = None) -> dict:
    runs = [load_run(p, label, source, now) for p, label, source in discover(paths)]
    data = analyse(runs, edges=edges, sat=sat, small_n=small_n)
    data["meta"]["inputs"] = paths
    data["log_check"] = log_check(logs, data["attempts"]) if logs else []
    return data


# ----------------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------------
def fmt_p(p: float | None) -> str:
    return "n/a" if p is None else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


def usd(v: float | None, d: int = 2) -> str:
    return "-" if v is None else f"${v:.{d}f}"


def share(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def section_runs(data: dict) -> str:
    rows = []
    for r in data["runs"]:
        pm = {True: "on", False: "off", None: "?"}[r["plan_mix"]] + (f" ({r['plan_mix_source']})" if r["plan_mix"] is not None else "")
        state = "finished" if r["finished"] else ("partial (live)" if r["live"] else "partial")
        sc = r["seed_check"]
        rows.append([r["run"], r["task"], r["dtype"], r["model"], fnum(r["budget_usd"], 2), fnum(r["spent_usd"], 2), r["rounds"], r["n_candidates"], state, r["stop_kind"], pm, r["k"] or "-", f"{sc['match']}/{sc['checked']}" if sc["checked"] else "-"])
    return md_table(["run", "task", "dtype", "model", "budget $", "spent $", "rounds", "kernels", "state", "stop", "plan_mix", "K", "seed check"], rows, "llllrrrrlllrr")


def section_q1(data: dict, small_n: int) -> str:
    q1, md = data["q1"], []
    edges = q1["edges"]
    patch_b = [b for b in data["branches"] if b["kind"] == "patch" and b["first_status"] != "pending"]
    tasks = task_order({b["task"] for b in patch_b})  # a task whose only requests are still in flight has no row yet
    order = {t: i for i, t in enumerate(tasks)}
    md.append("### 3.1 What a patch request first produces (first kernel of each `r<k>.p<i>`, by task)\n")
    md.append(f"`n` = patch requests (branches); status counts are those of the *first* kernel of the branch (`no_code` = no code block came back: in practice the reply was cut off at max_tokens, see 2 and 3.4). `†` = n < {small_n}: anecdotal. "
              f"{data['meta']['pending_branches_excluded']} requests still in flight in live runs are excluded.\n")
    hdr, rows = outcome_table(patch_b, "first_status", lambda b: b["task"], order)
    md.append(md_table(hdr, rows))
    md.append("### 3.2 After up to 3 repair turns (final status of the branch; open branches of live runs excluded)\n")
    rows = []
    for t in tasks:
        d = q1["per_task"][t]
        cl, fin = d["n_closed"], Counter(d["final_status"])
        fails = ", ".join(f"{s} {fin[s]}" for s in sorted(fin) if s != "ok") or "-"
        rows.append([t, d["n"], rate_cell(fin.get("ok", 0), cl, small_n), rate_cell(d["needed_repair"], cl, small_n), rate_cell(d["repaired"], d["needed_repair"], small_n), fails, rate_cell(d["accepted"], d["n"], small_n)])
    md.append(md_table(["task", "branches", "ok after repair", "needed repair", "repair succeeded", "still failing (final status)", "new best"], rows))
    md.append("### 3.3 Failure rate by round (first kernel not ok / branches; `[k]` = k new bests)\n")
    rounds = sorted({r for t in tasks for r in q1["per_task_round"][t]})

    def cells(d):
        return ["-" if not d.get(r) else f"{d[r]['fail']}/{d[r]['n']}" + ("†" if d[r]["n"] < small_n else "") + (f" [{d[r]['accepted']}]" if d[r]["accepted"] else "") for r in rounds]

    md.append("By task (all runs pooled):\n")
    md.append(md_table(["task", *[f"r{r}" for r in rounds]], [[t, *cells(q1["per_task_round"][t])] for t in tasks], "l" + "r" * len(rounds)))
    fa_runs = [r for r in data["runs"] if r["task"].split("@")[0] == "flash_attention" and r["run"] in q1["per_run_round"]]
    if fa_runs:
        md.append("FlashAttention, run by run (every `†` cell is n < %d):\n" % small_n)
        md.append(md_table(["run", *[f"r{r}" for r in rounds]], [[r["run"] + ("" if r["finished"] else " (partial)"), *cells(q1["per_run_round"][r["run"]])] for r in fa_runs], "l" + "r" * len(rounds)))
        md.append("FlashAttention, per run (patch requests):\n")
        rows = []
        for r in fa_runs:
            g = q1["per_run"][r["run"]]
            rows.append([r["run"] + ("" if r["finished"] else " (partial)"), g["n"], rate_cell(g["ok_first"], g["n"], small_n), rate_cell(g["ok_final"], g["n"], small_n), rate_cell(g["accepted"], g["n"], small_n), fnum(g["median_diff"], 2), fnum(g["usd_per_branch"], 3)])
        md.append(md_table(["run", "branches", "first kernel ok", "ok after repair", "new best", "median diff frac", "LLM $/branch"], rows))
    md.append("### 3.4 Why do kernels fail? (every failed attempt of a patch request)\n")
    rows = [[("FA" if c["fa"] else "other"), c["status"], c["detail"] or "-", c["n"], fnum(c["median_diff"], 2)] for c in q1["failure_causes"]][:20]
    md.append(md_table(["tasks", "status", "dominant message (digits -> N)", "n", "median diff"], rows, "lllrr"))
    md.append("`incorrect:marginal` = fewer than 1e-4 of the elements out of tolerance and worst scaled error < 3 (a tolerance-edge rounding miss); `incorrect:gross` = anything worse. "
              "`no_code:truncated` = the reply ended with `finish_reason=length` (the 48k-token cap was hit while the model was still reasoning), `no_code:no_block` = the reply had no code block, `no_code:gap` = inferred from missing `.fix` numbers.\n")
    rows = [[t, d["branches"], d["turns"], d["truncated_turns"], rate_cell(d["with_truncated_turn"], d["branches"], small_n), usd(d["truncated_usd"]), usd(d["llm_usd"])] for t, d in sorted(q1["truncation"].items(), key=lambda kv: order.get(kv[0], 99)) if d["turns"]]
    md.append("Replies truncated at max_tokens, and what they cost (patch requests; LLM $ from the transcript):\n")
    md.append(md_table(["task", "branches", "LLM turns", "truncated turns", "branches with one", "truncated $", "total LLM $"], rows))
    md.append("### 3.5 Size of the change vs outcome\n")
    md.append(
        f"Size proxy = fraction of changed lines between the candidate and the seed it patched (comments/blank lines/whitespace removed; `(removed + added) / (lines of both)`; 0 = identical, 1 = no line in common). "
        f"Default buckets: small < {edges[0]:.2f} <= medium < {edges[1]:.2f} <= large (`--size-edges`). 'First kernels' = the first kernel of each patch request (independent samples for the same seed and plan); "
        "'all attempts' also counts repair turns. 'new best' = became the run's best (trace flag `improved`); 'faster than seed' = ok and > 0.5% faster than its seed. "
        "Kernels with no code (truncated replies) have no diff and are not in these tables, so size is conditional on a kernel existing. Round 1 patches the unoptimised initial kernel, where almost any change can win; the `rounds >= 2` rows remove that.\n"
    )
    for name, title in (("FA", "FlashAttention (all variants)"), ("nonFA", "all other tasks")):
        d = q1["size"][name]
        for view, vt in (("first", "first kernels"), ("all", "all attempts incl. repairs"), ("first_from_r2", "first kernels, rounds >= 2")):
            if name == "nonFA" and view == "first_from_r2":
                continue
            md.append(f"**{title}** - {vt}:\n")
            md.append(md_table(SIZE_HDR, size_rows(d[view], small_n), "lrrrrrr"))
    md.append("**Threshold-free check and sensitivity to the bucket edges** (first kernels; Fisher exact p of small vs large; the answer should not hinge on one choice of edges):\n")
    for name, title in (("FA", "FlashAttention"), ("nonFA", "other tasks")):
        ds = q1["diff_by_status"][name]
        md.append(f"{title}: median diff fraction of ok first kernels {fnum(ds['median_ok'], 2)} (n={ds['n_ok']}) vs failed first kernels {fnum(ds['median_fail'], 2)} (n={ds['n_fail']}); Mann-Whitney {fmt_p(ds['p_mwu'])}; "
                  "by status: " + ", ".join(f"{s} {fnum(v['median_diff'], 2)} (n={v['n']})" for s, v in ds["by_status"].items()) + ".\n")
        rows = []
        for s in q1["sensitivity"][name]:
            b = s["blocks"]
            rows.append([s["scheme"], *[f"{b[k]['ok']}/{b[k]['n']}" + ("†" if b[k]["n"] < small_n else "") for k in BUCKETS], fmt_p(s["p_ok"]), *[f"{b[k]['new_best']}/{b[k]['n']}" for k in BUCKETS], fmt_p(s["p_new_best"])])
        md.append(md_table([f"{title}: scheme", "ok small", "ok medium", "ok large", "small vs large", "new best small", "new best medium", "new best large", "small vs large"], rows, "lrrrrrrrr"))
    dd = q1["deciles"]
    md.append("Distribution of the diff fraction over all attempts (min, p10, p25, median, p75, p90, max): " + "; ".join(f"{k}: " + ", ".join(f"{x:.2f}" for x in v) for k, v in dd.items() if v) + ".\n")
    md.append("New bests of patch requests by size bucket (default edges): " + "; ".join(f"{k}: " + ", ".join(f"{b} {v.get(b, 0)}" for b in BUCKETS) for k, v in q1["accepted_by_bucket"].items()) + ".\n")
    md.append("### 3.6 Local vs structural plans (runs with `plan_mix`)\n")
    mx = q1["mix"]
    if not mx["runs"]:
        md.append("_No run with plan_mix on and an evaluated mixed round was found._\n")
    else:
        md.append("Designated local plan = the last plan of a round (`r<k>.p<K-1>`) when the planner returned all K plans and did not fall back to the generic plans; the other plans are structural. "
                  "`ratio` = speedup of the branch's final ok kernel / speedup of its seed; `mean gain/branch` counts a failed branch as 0 gain. Runs: " + ", ".join(mx["runs"]) + ". These runs also differ from the others in budget and search aids (see 0).\n")
        rows = [["local (pooled)", *compare_cell(mx["local"], small_n)], ["structural (pooled)", *compare_cell(mx["structural"], small_n)],
                ["local (FlashAttention only)", *compare_cell(mx["local_FA"], small_n)], ["structural (FlashAttention only)", *compare_cell(mx["structural_FA"], small_n)]]
        for run, d in mx["per_run"].items():
            rows += [[f"{run}: local", *compare_cell(d["local"], small_n)], [f"{run}: structural", *compare_cell(d["structural"], small_n)]]
        md.append(md_table(COMPARE_HDR, rows))
        md.append(f"First-attempt status by role: {mx['first_status_by_role']}. New bests by role: {mx['accepted_by_role']}. Branches in rounds whose roles could not be assigned: {mx['unknown_role']}.\n")
        lo, st = mx["local"], mx["structural"]
        if lo["n"] and st["n"]:
            p = fisher_exact(lo["ok_first"], lo["n"] - lo["ok_first"], st["ok_first"], st["n"] - st["ok_first"])
            c = mx["cmh_first_ok"]
            md.append(f"First-attempt ok, local vs structural: pooled Fisher exact {fmt_p(p)}; stratified by run (Cochran-Mantel-Haenszel, {c['strata']} informative strata) odds ratio {fnum(c['odds_ratio'], 2)}, {fmt_p(c['p'])}.\n")
    cl = q1["control_last"]
    if cl["last"]["n"]:
        md.append("**Control - position effect.** The planner orders plans 'from most to least promising' and the last plan is evaluated last, so it may differ from the others even without plan_mix. In runs *without* plan_mix:\n")
        rows = [["last plan", *compare_cell(cl["last"], small_n)], ["earlier plans", *compare_cell(cl["earlier"], small_n)], ["last plan (FA)", *compare_cell(cl["last_FA"], small_n)], ["earlier plans (FA)", *compare_cell(cl["earlier_FA"], small_n)]]
        md.append(md_table(COMPARE_HDR, rows))
    md.append("### 3.7 Rounds without a new best (what an adaptive trigger would see)\n")
    rows = []
    for name in ("FA", "nonFA"):
        s = q1["streaks"][name]
        rows.append([name, rate_cell(s["won"], s["rounds"], small_n), rate_cell(s["after_1_won"], s["after_1_no_win"], small_n), rate_cell(s["after_2_won"], s["after_2_no_win"], small_n), s["rounds_no_ok_first"]])
    md.append("Completed rounds only. 'new best' columns are the share of rounds that produced a new best: overall, right after one round without one, right after two.\n")
    md.append(md_table(["tasks", "rounds with a new best", "... after 1 round without", "... after 2 rounds without", "rounds in which no first kernel was ok"], rows))
    return "\n".join(md)


def section_q2(data: dict, sat: float, small_n: int) -> str:
    rows_all = data["q2"]["rows"]
    fin = [r for r in rows_all if r["finished"]]
    md = []
    md.append(
        f"Roofline fraction = `lower_bound_us / candidate_us` of the confirmed result (= `bandwidth_fraction` for memory-bound, `compute_fraction` for compute-bound tasks; can exceed 1 because the peaks "
        f"are measured once and are not a hard ceiling). Saturated = fraction >= {sat:.2f}. 'Baseline frac' = the same for the best baseline; 'ceiling vs baseline' = best baseline us / roofline lower bound, i.e. the speedup a kernel exactly "
        "*at* the measured roofline would show (below 1: the baseline already beats the measured roofline, so the roofline is not a ceiling there); 'eff. frac' = min(roofline, best baseline) / candidate time. "
        "'After last improvement' = ledger USD at the end minus USD at the last candidate flagged `improved` (>= min_improvement and beyond noise); it includes the patience rounds that detect convergence. "
        "'After first sat.' = USD after the first candidate whose own report is at >= the saturation fraction.\n"
    )

    def row(r):
        spent = r["spent_usd"]
        return [
            r["run"], r["dtype"], fnum(r["speedup"], 3, "x"), r["bound"] or "-", fnum(r["roof_frac"], 3), fnum(r["baseline_roof_frac"], 3), fnum(r["ceiling_vs_baseline"], 3), fnum(r["eff_frac"], 3),
            r["rounds"], fnum(spent, 3), fnum(r["spent_after_last_improvement"], 3), r["stop_kind"],
            ("yes" if r["saturated"] else "no"), (f"{r['first_sat']} (r{r['first_sat_round']})" if r["first_sat"] else "-"), fnum(r["spent_after_first_sat"], 3),
            pct(int(round(r["spent_after_first_sat"] * 1000)), int(round(spent * 1000))) if r["spent_after_first_sat"] is not None and spent else "-",
        ]

    hdr = ["run", "dtype", "speedup", "bound", "roofline frac", "baseline frac", "ceiling vs base", "eff. frac", "rounds", "spent $", "after last impr. $", "stop", "saturated", "first sat. kernel", "after first sat. $", "% of run"]
    for src in sorted({r["source"] for r in fin}):
        md.append(f"**{src}** (finished runs)\n")
        md.append(md_table(hdr, [row(r) for r in fin if r["source"] == src], "llrlrrrrrrrrllrr"))
    part = [r for r in rows_all if not r["finished"]]
    if part:
        md.append("**Partial runs** (no result.json: in no total below; roofline taken from the best kernel so far)\n")
        md.append(md_table(["run", "task", "best speedup", "roofline frac", "rounds", "kernels", "spent so far $"], [[r["run"], r["task"], fnum(r["speedup"], 3, "x"), fnum(r["roof_frac"], 3), r["rounds"], r["candidates"], fnum(r["spent_usd"], 3)] for r in part], "llrrrrr"))
    ag = data["q2"]["aggregate"]
    md.append("**Totals over the finished runs above**\n")
    rows = [
        [f"finished runs / saturated (>= {sat:.2f})", f"{ag['n_finished']} / {ag['n_saturated']}"],
        ["total spend", usd(ag["total_usd"], 3)],
        ["spend of saturated runs", f"{usd(ag['saturated_total_usd'], 3)} ({share(ag['saturated_share_of_total'])} of total)"],
        ["saturated runs whose first saturated kernel was the initial one (r0)", f"{ag['n_saturated_at_init']} of {ag['n_saturated']}"],
        ["saturated runs in which the best baseline is faster than the roofline lower bound", f"{ag['n_saturated_baseline_faster_than_roofline']} of {ag['n_saturated']}"],
        ["spend after the first near-saturated kernel (saturated runs)", f"{usd(ag['spent_after_first_sat_usd'], 3)} = {share(ag['share_of_total_after_first_sat'])} of total spend"],
        ["spend after the last improvement (all finished runs)", f"{usd(ag['spent_after_last_improvement_usd'], 3)} = {share(ag['share_of_total_after_last_improvement'])} of total"],
        ["stop reasons (all / saturated)", f"{ag['stop_kinds']} / {ag['stop_kinds_saturated']}"],
    ]
    md.append(md_table(["", "value"], rows, "ll"))
    srcs = data["q2"]["aggregate_by_source"]
    if len(srcs) > 1:
        rows = [[s, a["n_finished"], a["n_saturated"], usd(a["total_usd"], 3), usd(a["spent_after_first_sat_usd"], 3), share(a["share_of_total_after_first_sat"]), usd(a["spent_after_last_improvement_usd"], 3), share(a["share_of_total_after_last_improvement"])] for s, a in srcs.items()]
        md.append(md_table(["source", "finished", "saturated", "spend", "after first sat.", "% of spend", "after last impr.", "% of spend"], rows))
    return "\n".join(md)


SCOPE_ORDER = ("local", "structural", "generic", "unclassified")
OUTCOME_HDR = ["n", "ok", "abi_error", "incorrect", "no_code", "other", "ok given code", "ok after repair", "new best"]


def status_cells(d: dict, small_n: int) -> list:
    n = d["n"]
    return [n, rate_cell(d["ok"], n, small_n), rate_cell(d["abi_error"], n, small_n), rate_cell(d["incorrect"], n, small_n), rate_cell(d["no_code"], n, small_n), d["other"],
            rate_cell(d["ok"], n - d["no_code"], small_n), rate_cell(d["ok_final"], n, small_n), rate_cell(d["accepted"], n, small_n)]


def section_kinds(data: dict, small_n: int) -> str:
    k, md = data["kinds"], []
    tasks = task_order(k["by_task_scope"])
    md.append("### 1.1 How plans were labelled, and how far the label can be trusted\n")
    md.append(
        "The unit is a *patch request* (`r<k>.p<i>`); its plan text (title + description) is read from the patch prompt in the transcript. **Kind** is assigned by keyword lexicons (module header of "
        "`experiments/plan_type_analysis.py`): `local` = instruction- or parameter-level edits (the list in the plan_mix prompt: vector width, unroll, loop order, hoisting, fewer barriers or live registers, "
        "fast-math, address arithmetic, reading operands once; or a plan that calls itself 'local' / 'structure-preserving'); the paper's three global axes `thread_mapping`, `data_movement`, `synchronization`; "
        "`algorithm` (Winograd, online-softmax cadence ...). **Scope** = local vs structural (the four global kinds together); `generic` = one of the orchestrator's fallback plans (the planner returned "
        "no usable plan), `unclassified` = no keyword hit. This is a heuristic, not a human label.\n"
    )
    v = k["validation"]
    if v["n"]:
        md.append(f"*Check against the prompt's own designation.* In the plan_mix runs the prompt told the planner that the last plan of a round is local and the others structural. For {v['n']} such plans (rounds with all K plans and no fallback) the classifier agrees on {v['agree']}: "
                  + ", ".join(f"{a} {b}" for a, b in v["confusion"].items()) + ". "
                  + ("Disagreements: " + "; ".join(f"`{d['run']}` {d['base']} designated {d['role']}, classified {d['kind']} ('{d['title']}')" for d in v["disagreements"]) + ". " if v["disagreements"] else "")
                  + "The lexicons were tuned once after reading those plans, so this agreement is optimistic; on all other plans the accuracy is unknown.\n")
    dp = k["default_prompt"]
    rows = []
    for g, d in sorted(dp.items(), key=lambda kv: (kv[0] != "flash_attention", kv[0])):
        sc, kd = d["scope"], d["kind"]
        rows.append([g, d["n"], rate_cell(sc.get("local", 0), d["n"], small_n), rate_cell(sc.get("structural", 0), d["n"], small_n),
                     "/".join(str(kd.get(a, 0)) for a in PLAN_AXES), sc.get("generic", 0), sc.get("unclassified", 0)])
    md.append(f"*What the default prompt delivers.* Its wording is \"each a concrete structural change (not a micro-tweak)\", so it should yield no local plan. In the {len(k['default_prompt_runs'])} runs that used it:\n")
    md.append(md_table(["task group", "plans", "local", "structural", "of which thread_mapping/data_movement/synchronization/algorithm", "generic fallback", "unclassified"], rows, "lrrrlrr"))
    md.append("### 1.2 First-attempt outcome by task and plan scope\n")
    md.append(f"First kernel (or its absence) of each patch request. `ok given code` removes the replies that never contained code (section 2), so that it compares kernels that exist. `ok after repair` = the request ended with an ok kernel after up to 3 repair turns. `†` = n < {small_n}.\n")
    rows = []
    for t in tasks:
        for sc in SCOPE_ORDER:
            d = k["by_task_scope"][t].get(sc)
            if d:
                rows.append([t, sc, *status_cells(d, small_n)])
    md.append(md_table(["task", "plan scope", *OUTCOME_HDR], rows, "llrrrrrrrrr"))
    rows = [[g, sc, *status_cells(d, small_n)] for g, by in sorted(k["by_group_scope"].items(), key=lambda kv: (kv[0] != "flash_attention", kv[0])) for sc in SCOPE_ORDER for d in [by.get(sc)] if d]
    md.append("By task group:\n")
    md.append(md_table(["task group", "plan scope", *OUTCOME_HDR], rows, "llrrrrrrrrr"))
    md.append("### 1.3 FlashAttention by plan kind\n")
    fa = k["by_task_kind"].get("flash_attention", {})
    rows = [[kd, *status_cells(fa[kd], small_n)] for kd in PLAN_KINDS if kd in fa]
    md.append(md_table(["plan kind", *OUTCOME_HDR], rows, "lrrrrrrrrr"))
    md.append("`ok given code` is the figure to read for 'does this kind of plan break the kernel'; the `no_code` column is a property of the reply, not of the kernel (section 2).\n")
    md.append("### 1.4 Local vs structural: is the difference more than noise?\n")
    rows = []
    for name in ("ok", "ok given code", "abi_error", "incorrect", "no_code", "new best"):
        f_, a_ = k["tests_fa"].get(name), k["tests_all"].get(name)
        cell = lambda x: "-" if not x else f"{x['local'][0]}/{x['local'][1]} vs {x['structural'][0]}/{x['structural'][1]}, {fmt_p(x['p'])}"
        rows.append([name, cell(f_), cell(a_)])
    md.append("First attempts, local plans vs structural plans (Fisher exact; generic and unclassified plans left out):\n")
    md.append(md_table(["first attempt", "FlashAttention", "all tasks pooled"], rows, "lll"))
    c = k["cmh_ok_given_code"]
    md.append(f"The pooled column mixes tasks (local plans are mostly proposed for memory-bound ops, which almost never fail), so it overstates the local advantage. Stratified by run (Cochran-Mantel-Haenszel, {c['strata']} informative runs), `ok given code`, local vs structural: odds ratio {fnum(c['odds_ratio'], 2)}, {fmt_p(c['p'])}.\n")
    md.append("### 1.5 ABI failures of FlashAttention first attempts\n")
    ab = k["fa_abi_by_kind"]
    rows = [[kd, ab[kd]["abi"], ab[kd]["threadgroup_memory"]] for kd in PLAN_KINDS if kd in ab]
    md.append(md_table(["plan kind", "abi_error first attempts", "of which 'static threadgroup memory exceeds the device limit'"], rows))
    return "\n".join(md)


def _fmt_int(v) -> str:
    return "-" if v is None else f"{int(round(v)):,}"


def section_nocode(data: dict, small_n: int) -> str:
    n, md = data["nocode"], []
    cap = n["cap_tokens"]
    md.append("### 2.1 How often, and what it costs\n")
    md.append(
        f"A first attempt has no code when the final model turn of the request ended without a ```metal block (the orchestrator then sends a nudge and spends a repair turn). Cause is read from the last logged turn: "
        "`reasoning_cap` = `finish_reason=length` with >= 90% of the completion tokens spent on reasoning; `run_ended` = the run stopped (budget) after the model had asked for a tool, before the request finished (not the model's fault); "
        "`no_block` = a normal reply with no code block. `max_tokens` is not logged per request: " + (f"every truncated reply carries {_fmt_int(cap)} completion tokens (median), matching `max_tokens = 48000` in `taic.toml`; `reasoning_effort` is not logged either (`taic.toml` says `high`).\n" if cap else "no reply was truncated.\n")
    )
    rows = []
    for r in sorted(n["per_run"], key=lambda r: (r["task"] != "flash_attention", r["task"], r["run"])):
        if not r["patch_requests"]:
            continue
        causes = ", ".join(f"{c} {v}" for c, v in sorted(r["causes"].items())) or "-"
        rows.append([r["run"] + ("" if r["finished"] else " (partial)"), r["model"], r["patch_requests"], rate_cell(r["no_code_first"], r["patch_requests"], small_n), causes, r["turns"],
                     f"{r['length_turns']} ({pct(r['length_turns'], r['turns'])})", usd(r["length_usd"]), f"{usd(r['llm_usd'])} ({pct(int(round(1000 * r['length_usd'])), int(round(1000 * r['llm_usd'])))})"])
    md.append(md_table(["run", "model", "patch requests", "first attempt without code", "cause", "LLM turns", "truncated turns", "truncated $", "LLM $ (truncated share)"], rows, "llrrlrrrr"))
    md.append(f"All runs together: {n['length_turns']} of {n['turns']} LLM turns ended by the token cap, costing {usd(n['length_cost'])} of {usd(n['total_cost'])} LLM spend ({share(n['length_cost'] / n['total_cost']) if n['total_cost'] else '-'}). "
              "The cost is more than the truncated reply: the next request is a repair turn that carries the whole reply (reasoning echoed back, `echo_reasoning_content = true`): "
              f"prompt {_fmt_int(n['repair_after_truncation']['prompt_before'])} tokens before vs {_fmt_int(n['repair_after_truncation']['prompt_after'])} after a truncated reply (median of {n['repair_after_truncation']['n']} pairs).\n")
    rows = []
    for name, rc in (("FlashAttention", n["recovery_fa"]), ("all tasks", n["recovery"])):
        rows.append([name, rc["n"], rate_cell(rc["ok_final"], rc["n"], small_n), rate_cell(rc["accepted"], rc["n"], small_n), usd(rc["usd_total"]), usd(rc["usd_median"], 3), usd(rc["usd_median_ok_first"], 3)])
    md.append("Recovery of the patch requests whose first attempt died by `reasoning_cap` (the repair turn is a fresh sample after a nudge):\n")
    md.append(md_table(["tasks", "requests", "ok kernel after repair", "new best", "LLM $ of these requests", "median $ per request", "median $ of a request whose first kernel was ok"], rows, "lrrrrrr"))
    lc = data.get("log_check") or []
    if lc:
        md.append("*Cross-check with the orchestrator's own log.* Every `[<tag>] no code block in reply` line of the logs given with `--logs` was matched, tag by tag, with the attempts classified as no-code from the transcripts "
                  "(a request cut by the budget stop never reaches the log, so `run_ended` attempts are not expected there; `pending` = the attempt of a killed run whose outcome was never logged to the transcript):\n")
        rows = [[x["log"], x["run"], len(x["log_lines"]), len(x["matched"]), len(x["only_in_log"]), len(x["only_in_log_pending"]), len(x["only_in_transcript"])] for x in lc]
        rows.append(["total", "", sum(r[2] for r in rows), sum(r[3] for r in rows), sum(r[4] for r in rows), sum(r[5] for r in rows), sum(r[6] for r in rows)])
        md.append(md_table(["log", "run", "log lines", "matched", "only in the log", "of which pending in the transcript", "no-code attempt not in the log"], rows, "llrrrrr"))
    md.append("### 2.2 What a truncated reply is\n")
    m = n["mechanism"]
    t, cr, lg, sh = m["truncated"], m["code_replies"], m["reasoning_ge_half_cap_not_truncated"], m["reasoning_lt_half_cap"]
    rows = [
        ["truncated (`finish_reason=length`)", t["n"], _fmt_int(t["completion"]["p50"]), _fmt_int(t["completion"]["max"]), fnum(t["reasoning_share_median"], 3), "-", t["fences_median"]],
        ["reply that carries a kernel (`stop`, code block)", cr["n"], "-", "-", "-", f"{_fmt_int(cr['visible_tokens']['p50'])} / {_fmt_int(cr['visible_tokens']['p90'])} / {_fmt_int(cr['visible_tokens']['max'])}", "-"],
        [f"finished turn with reasoning >= half the cap ({_fmt_int(cap / 2) if cap else '-'} tokens)", lg["n"], "-", "-", "-", "-", fnum(lg["fences_median"], 1)],
        ["finished turn with less reasoning", sh["n"], "-", "-", "-", "-", fnum(sh["fences_median"], 1)],
    ]
    md.append(md_table(["turns", "n", "completion tokens (median)", "completion tokens (max)", "reasoning / completion (median)", "visible code tokens p50 / p90 / max", "code drafts inside the reasoning (median ``` pairs)"], rows, "lrrrrrr"))
    md.append(f"Of the {t['n']} truncated replies, {t['pure_thinking']} had written nothing visible (no content, no tool call) and {t['with_visible_output']} had started a tool call or a reply; every one had spent >= {fnum(t['reasoning_share_min'], 2)} of its tokens on reasoning. "
              f"A complete kernel reply needs a median of {_fmt_int(cr['visible_tokens']['p50'])} and at most {_fmt_int(cr['visible_tokens']['max'])} visible tokens: "
              f"{fnum(cap / cr['visible_tokens']['max'], 1) if cap and cr['visible_tokens']['max'] else '-'}x less than the cap. So **the length of the code is not what hits the cap; the thinking is.** "
              "The long thinking is the model drafting and re-checking whole kernels in its reasoning (the reasoning of long replies contains a median of "
              f"{fnum(lg['fences_median'], 0)} code blocks, against {fnum(sh['fences_median'], 0)} in short ones: the same behaviour in the truncated and in the long-but-finished replies, i.e. the cap cuts the tail of one heavy-tailed distribution).\n")
    md.append("### 2.3 Thinking budget per request, by model and task (first turn of each request)\n")
    rows = [[d["model"], d["group"], d["tg"], d["n"], _fmt_int(d["p50"]), _fmt_int(d["p90"]), _fmt_int(d["max"]), rate_cell(d["ge_half"], d["n"], small_n), rate_cell(d["length"], d["n"], small_n)] for d in n["first_turn_reasoning"]]
    md.append(md_table(["model", "task group", "request", "n", "reasoning tokens p50", "p90", "max", "reasoning >= half the cap", "hit the cap"], rows, "lllrrrrrr"))
    pro = n["fa_pro"]
    md.append(f"On the same FlashAttention task and prompts, `deepseek-v4-pro` produced {pro['truncated_first_turns']} truncated first turns in {pro['requests']} patch requests and {pro['no_code_first']} first attempts without code (a single run: no repeat; it differs from the flash runs only in the model, but also in run date and trajectory).\n")
    md.append("### 2.4 Is it the plan text, the size of the code, or the state of the search? (first turn of patch requests, deepseek-flash)\n")
    for name, pr in n["predictors"].items():
        md.append(f"**{name}**: {pr['truncated']} of {pr['n']} first turns hit the cap.\n")
        rows = [[lab, fnum(f["truncated_median"], 0), fnum(f["other_median"], 0), fmt_p(f["p"]), f"{fnum(f['rank_truncated'], 2)} vs {fnum(f['rank_other'], 2)}", fmt_p(f["p_within_run"])] for lab, f in pr["features"].items()]
        md.append(md_table(["feature", "median, truncated", "median, not truncated", "Mann-Whitney (pooled)", "percentile within own run, truncated vs not", "Mann-Whitney (within-run ranks)"], rows, "lrrrrr"))
        md.append("Share truncated: " + "; ".join(f"{lab} {rate_cell(a, b, small_n)}" for lab, (a, b) in pr["by_round"].items()) + ". By plan scope: "
                  + "; ".join(f"{sc} {rate_cell(a, b, small_n)}" for sc, (a, b) in pr["by_scope"].items()) + ".\n")
    pr = next(iter(n["predictors"].values()), None)
    if pr and pr["n"]:
        ft = pr["features"]
        cp, pp, rp = (ft["plan text (characters)"]["p_within_run"], ft["prompt (tokens: system + seed kernel + feedback + plan)"]["p_within_run"], ft["round"]["p_within_run"])
        ge, lo_, st_ = pr["by_scope"]["generic"], pr["by_scope"]["local"], pr["by_scope"]["structural"]
        md.append(f"Reading the first pool (within-run ranks): plan text length {fmt_p(cp)}; prompt size {fmt_p(pp)}; round {fmt_p(rp)}; local vs structural plans {rate_cell(*lo_, small_n)} vs {rate_cell(*st_, small_n)}, {fmt_p(pr['p_local_vs_structural'])}; "
                  f"one-line generic fallback plans {rate_cell(*ge, small_n)} truncated. "
                  + ("The plan text does not separate truncated from other requests"
                     + (", what does is how far the search has gone (seed kernel size and round are collinear: seeds grow as optimisations accumulate)" if (pp is not None and pp < 0.05) or (rp is not None and rp < 0.05) else "")
                     + "." if (cp is not None and cp > 0.05) else "The plan text length is associated with truncation here.")
                  + " Local plans are truncated "
                  + ("less often than structural ones, but the difference does not reach significance at this n." if lo_[1] and st_[1] and lo_[0] / lo_[1] < st_[0] / st_[1] and (pr["p_local_vs_structural"] or 1) >= 0.05 else "about as often as structural ones." if lo_[1] and st_[1] else "(too few to compare).")
                  + "\n")
    return "\n".join(md)


def answer(data: dict, sat: float, small_n: int) -> str:
    """The short answer to the question, assembled from the computed tables (every number is read from ``data``)."""
    k, n, q1, f = data["kinds"], data["nocode"], data["q1"], data["facts"]
    md = []
    md.append(
        "**Question.** The paper's plans range \"from local instruction-level changes to global changes in thread mapping, data movement and synchronization\"; our `plan_user` prompt asks every plan to be "
        "\"a concrete structural change (not a micro-tweak)\". Can opening the plan types (also asking for local plans) fix the FlashAttention failures (ABI errors, wrong results, no code) and the "
        "waste on ops that are already at the roofline?\n"
    )
    fa = q1["per_task"].get("flash_attention", {})
    fs = fa.get("first_status", {})
    nfa = sum(fs.values())
    fa_causes: Counter = Counter()
    for r in n["per_run"]:
        if r["task"] == "flash_attention":
            fa_causes.update(r["causes"])
    flash = next((r for r in n["per_run"] if r["run"].endswith("/fa_flash")), None)
    mech, cr = n["mechanism"], n["mechanism"]["code_replies"]
    cap = n["cap_tokens"]
    pr = next(iter(n["predictors"].values()), None)
    pro = n["fa_pro"]
    ts = k["tests_fa"]
    nc, abi = fs.get("no_code", 0), fs.get("abi_error", 0)
    cut = fa_causes.get("run_ended", 0)
    md.append(f"**Short answer: for the {'largest' if nc >= abi else 'a large'} loss (no code), no; for the ABI errors, perhaps a little; for the final speedup, nothing in this data shows it.**\n")
    # 1. no code
    line = (f"1. **Replies without code, and the plan type is not their cause.** {nc} of {nfa} FlashAttention patch requests ({pct(nc, nfa)}) produced no kernel at the first attempt"
            + (f" ({nc - cut} of {nfa - cut} if the {cut} requests cut by the budget stop are left out)" if cut else "")
            + (f"; in `fa_flash` {flash['no_code_first']} of {flash['patch_requests']}, of which {flash['causes'].get('reasoning_cap', 0)} by the cause below" if flash else "") + ". "
            f"**Root cause: `deepseek-flash` spends the whole `max_tokens` ({_fmt_int(cap)}) on reasoning.** All {mech['truncated']['n']} truncated replies in these transcripts ended with `finish_reason=length`; "
            f"{mech['truncated']['pure_thinking']} of them had written nothing visible and >= {fnum(mech['truncated']['reasoning_share_min'], 2)} of their tokens were reasoning. A complete kernel reply needs a median of {_fmt_int(cr['visible_tokens']['p50'])} "
            f"(max {_fmt_int(cr['visible_tokens']['max'])}) visible tokens, so the length of the code is not the limit. ")
    lc = data.get("log_check") or []
    if lc:
        line += (f"The orchestrator's own `no code block in reply` log lines agree with this classification ({sum(len(x['matched']) for x in lc)} of {sum(len(x['log_lines']) for x in lc)} lines matched tag by tag; "
                 f"the rest belong to attempts of killed runs, section 2.1). ")
    if pr and pr["n"]:
        ft = pr["features"]
        line += (f"The plan text does not explain it (plan length, within-run rank: {fmt_p(ft['plan text (characters)']['p_within_run'])}; the one-line generic fallback plans are cut {rate_cell(*pr['by_scope']['generic'], small_n)}); "
                 f"how far the search has gone does (seed prompt size {fmt_p(ft['prompt (tokens: system + seed kernel + feedback + plan)']['p_within_run'])}, round {fmt_p(ft['round']['p_within_run'])}). "
                 f"Local plans are cut {rate_cell(*pr['by_scope']['local'], small_n)} vs {rate_cell(*pr['by_scope']['structural'], small_n)} for structural ones ({fmt_p(pr['p_local_vs_structural'])}). ")
    line += (f"On the same task `deepseek-v4-pro` was cut {pro['truncated_first_turns']} times in {pro['requests']} requests (one run). Truncated replies cost {usd(n['length_cost'])} of {usd(n['total_cost'])} LLM spend over all runs ({share(n['length_cost'] / n['total_cost']) if n['total_cost'] else '-'}). "
             "The lever is the thinking budget (effort, `max_tokens`, model) and what a retry carries, not the plan type (section 2).\n")
    md.append(line)
    # 2. ABI / incorrect
    ok_c, abi_c = ts["ok given code"], ts["abi_error"]
    abi_all = sum(c["n"] for c in q1["failure_causes"] if c["fa"] and c["status"] == "abi_error")
    abi_tg = f["fa"]["abi_threadgroup_memory_attempts"]
    kinds_abi = [kd for kd, v in k["fa_abi_by_kind"].items() if v["abi"]]
    lo_rate = ok_c["local"][0] / ok_c["local"][1] if ok_c["local"][1] else None
    st_rate = ok_c["structural"][0] / ok_c["structural"][1] if ok_c["structural"][1] else None
    direction = "local plans point the right way" if (lo_rate is not None and st_rate is not None and lo_rate > st_rate) else "local plans show no advantage"
    md.append(
        f"2. **ABI errors and wrong results: {direction}, the evidence is thin, and repair absorbs most of it.** FlashAttention first attempts: {fs.get('ok', 0)}/{nfa} ok, {abi} abi_error, {fs.get('incorrect', 0)} incorrect. "
        f"{abi_tg} of the {abi_all} failed ABI attempts are the single message 'static threadgroup memory exceeds the device limit'; it appears under all {len(kinds_abi)} plan kinds that have an ABI failure ({', '.join(kinds_abi)}), so it is not tied to one kind of plan (section 1.5). "
        f"Local vs structural plans (FA): ok among kernels that exist {ok_c['local'][0]}/{ok_c['local'][1]} vs {ok_c['structural'][0]}/{ok_c['structural'][1]} ({fmt_p(ok_c['p'])}), abi_error {abi_c['local'][0]}/{abi_c['local'][1]} vs {abi_c['structural'][0]}/{abi_c['structural'][1]} ({fmt_p(abi_c['p'])}): "
        f"only {ok_c['local'][1]} local plans have a kernel, and the difference is not significant (stratified by run over all tasks: {fmt_p(k['cmh_ok_given_code']['p'])}). After up to 3 repair turns {rate_cell(f['fa']['ok_final'], f['fa']['n'], small_n)} of the requests end with an ok kernel, "
        f"so these failures cost repair turns more than they cost kernels. Local plans have not been shown to give faster kernels: new best {ts['new best']['local'][0]}/{ts['new best']['local'][1]} local vs {ts['new best']['structural'][0]}/{ts['new best']['structural'][1]} structural (FA).\n"
    )
    # 3. saturated
    q2 = f["q2"]
    ag = q2["aggregate"]
    dp = k["default_prompt"].get("memory-bound (elementwise/reduction/gemv/...)")
    md.append(
        f"3. **Saturated ops: a different plan type cannot recover the money; a stop rule could.** {q2['n_saturated']} of {q2['n_finished']} finished runs end at >= {sat:.0%} of the measured roofline and {ag['n_saturated_at_init']} of those already had such a kernel at the initial generation; "
        f"{usd(ag['spent_after_first_sat_usd'])} ({share(ag['share_of_total_after_first_sat'])} of the {usd(ag['total_usd'])} spent by all finished runs) was spent after that point. "
        + (f"The default prompt is not binding there anyway: {rate_cell(dp['scope'].get('local', 0), dp['n'], small_n)} of the plans proposed for memory-bound ops already classify as local (section 1.1). " if dp else "")
        + "The ceiling of these ops is within a few percent of the best baseline (section 4).\n"
    )
    md.append(
        "4. **Proposal, nothing implemented or run:** (i) test the thinking budget on the truncated requests themselves (same seeds and plans; change only effort, `max_tokens` or the retry context), (ii) keep local plans as a cheap option for "
        "rounds whose seed is close to its ceiling (section 5, (d)), (iii) a roofline-aware stop for saturated ops. Limits that apply to every statement above: observational data from single runs that differ in budget, model and protocol "
        f"(section 0); every rate carries its n and `†` marks n < {small_n}; no cross-run causal claim is made.\n"
    )
    return "\n".join(md)


def conclusions(data: dict, sat: float, small_n: int) -> str:
    f, q1 = data["facts"], data["q1"]
    fa, md = f["fa"], []
    # ---- (a)
    md.append("**(a) Could local plans help FlashAttention?**\n")
    if not fa["n"]:
        md.append("- No FlashAttention patch requests were found.\n")
    else:
        fs = fa["first_status"]
        top = ", ".join(f"{s} {fs[s]}" for s in STATUS_ORDER[1:] if fs.get(s))
        md.append(f"- **First-try failure is high; repair recovers most of it.** FlashAttention patch requests: {rate_cell(fa['ok_first'], fa['n'], small_n)} first kernels ok ({top}); {rate_cell(fa['ok_final'], fa['n'], small_n)} ok after repair "
                  f"(repair succeeded in {fa['repaired']}/{fa['needed_repair']} of the requests that needed it); {fa['accepted']} requests produced a new best.\n")
        causes = f"- **What the failures are.** {fa['abi_threadgroup_memory_attempts']} of {fa['abi_attempts']} failed ABI attempts are the single message 'static threadgroup memory exceeds the device limit' (a resource-budget mistake that the feedback names exactly, not a misunderstanding of the contract); "
        causes += f"{fa['incorrect_marginal']} of {fa['incorrect_marginal'] + fa['incorrect_gross']} `incorrect` attempts are marginal tolerance misses (a handful of elements out of 1,048,576), not logic errors; "
        if fa["truncated_turns"]:
            fa_cut = sum(r["causes"].get("run_ended", 0) for r in data["nocode"]["per_run"] if r["task"] == "flash_attention")
            causes += (f"and {fa['no_code_first'] - fa_cut} first kernels never existed because the model spent the whole max_tokens cap on reasoning" + (f" ({fa_cut} more were cut by the budget stop)" if fa_cut else "")
                       + f" (section 2; {fa['truncated_turns']} truncated turns = {usd(fa['truncated_usd'])} of the {usd(fa['llm_usd'])} LLM spend of FA patch requests). ")
            nc = f["mix"]["no_code_first"]
            lo, st = f["mix"]["local"], f["mix"]["structural"]
            if lo["n"] and st["n"]:
                causes += f"In plan_mix runs (all tasks) {nc.get('local', 0)}/{lo['n']} local vs {nc.get('structural', 0)}/{st['n']} structural first kernels were lost that way (anecdotal, dominated by conv2d).\n"
            else:
                causes += "\n"
        else:
            causes += "and no reply was truncated.\n"
        md.append(causes)
        sz = f["fa_size"]
        ds, ods = sz["diff_by_status"], sz["other_diff_by_status"]
        usable = [s for s in sz["sensitivity"] if s["small"]["n"] >= small_n and s["large"]["n"] >= small_n]
        worse = [s for s in usable if s["large"]["ok"] / s["large"]["n"] < s["small"]["ok"] / s["small"]["n"]]
        sig = [s for s in usable if s["p_ok"] is not None and s["p_ok"] < 0.05 and s in worse]
        mwu = ds["p_mwu"] is not None and ds["p_mwu"] < 0.05 and (ds["median_fail"] or 0) > (ds["median_ok"] or 0)
        d = sz["default"]
        if usable:
            strength = "moderate" if (mwu and len(sig) >= 2) else "weak" if (mwu or sig) else "no"
            od = sz["other_default"]
            md.append(f"- **Do large diffs fail more? {strength.capitalize()} evidence, for FA only.** Threshold-free: median diff of ok vs failed FA first kernels {fnum(ds['median_ok'], 2)} (n={ds['n_ok']}) vs {fnum(ds['median_fail'], 2)} (n={ds['n_fail']}), Mann-Whitney {fmt_p(ds['p_mwu'])}. "
                      f"By buckets (default edges) ok rate small {rate_cell(d['small']['ok'], d['small']['n'], small_n)}, medium {rate_cell(d['medium']['ok'], d['medium']['n'], small_n)}, large {rate_cell(d['large']['ok'], d['large']['n'], small_n)}; "
                      f"the large bucket is worse than the small one in {len(worse)} of {len(usable)} bucket schemes that have n >= {small_n} in both buckets (Fisher p < 0.05 in {len(sig)}), and the schemes with a larger edge leave the large bucket with n < {small_n}. "
                      f"For the other tasks there is no such effect (large {rate_cell(od['large']['ok'], od['large']['n'], small_n)} vs small {rate_cell(od['small']['ok'], od['small']['n'], small_n)} ok; Mann-Whitney {fmt_p(ods['p_mwu'])}). "
                      f"Caveats: line-diff size is a coarse proxy (a structural change such as a threadgroup-memory layout can be a few lines), most FA patches are small or medium (median {fnum(q1['deciles']['FA'][3] if q1['deciles']['FA'] else None, 2)}), so there are few real rewrites to learn from.\n")
        else:
            md.append(f"- **Do large diffs fail more?** Not answerable: no bucket scheme has n >= {small_n} in both the small and the large bucket (anecdotal).\n")
        acc = sz["accepted_by_bucket"]
        if sum(acc.values()):
            tot = sum(v["n"] for v in d.values()) or 1
            md.append(f"- **Where do the FA wins come from?** Of {sum(acc.values())} new bests (default edges) {acc.get('small', 0)} came from small, {acc.get('medium', 0)} from medium and {acc.get('large', 0)} from large diffs; "
                      f"the first kernels split {pct(d['small']['n'], tot)} / {pct(d['medium']['n'], tot)} / {pct(d['large']['n'], tot)}. "
                      f"New best per first kernel: small {rate_cell(d['small']['new_best'], d['small']['n'], small_n)}, medium {rate_cell(d['medium']['new_best'], d['medium']['n'], small_n)}, large {rate_cell(d['large']['new_best'], d['large']['n'], small_n)}. "
                      "So FA wins do come mostly from small and medium diffs, even though the current prompt demands structural changes.\n")
        lo, st, cm = f["mix"]["local"], f["mix"]["structural"], f["mix"]["cmh"]
        if lo["n"] or st["n"]:
            loa, sta = f["mix"]["local_FA"], f["mix"]["structural_FA"]
            md.append(f"- **Local vs structural inside plan_mix runs** ({', '.join(f['mix']['runs'])}; all tasks pooled): first kernel ok {rate_cell(lo['ok_first'], lo['n'], small_n)} vs {rate_cell(st['ok_first'], st['n'], small_n)}, "
                      f"ok after repair {rate_cell(lo['ok_final'], lo['n'], small_n)} vs {rate_cell(st['ok_final'], st['n'], small_n)}, new best {rate_cell(lo['accepted'], lo['n'], small_n)} vs {rate_cell(st['accepted'], st['n'], small_n)}, "
                      f"mean gain per branch {fnum(lo['mean_gain_pct'], 2, '%')} vs {fnum(st['mean_gain_pct'], 2, '%')}, median diff {fnum(lo['median_diff'], 2)} vs {fnum(st['median_diff'], 2)}; stratified by run {fmt_p(cm['p'])}. "
                      f"FA alone: first kernel ok {rate_cell(loa['ok_first'], loa['n'], small_n)} vs {rate_cell(sta['ok_first'], sta['n'], small_n)}. "
                      + ("Both groups are anecdotal (n < %d). " % small_n if min(lo["n"], st["n"]) < small_n else "")
                      + "The direction (local plans fail less at the first try) is what the claim predicts; the gain per branch hardly differs, and repair closes most of the first-try gap.\n")
        ca = f["control_FA"]
        if ca["last"]["n"]:
            md.append(f"- **Position control (FA, runs without plan_mix):** the last plan's first kernel is ok {rate_cell(ca['last']['ok_first'], ca['last']['n'], small_n)} vs {rate_cell(ca['earlier']['ok_first'], ca['earlier']['n'], small_n)} for earlier plans, so being last does not by itself change the failure rate.\n")
        nc_fa = f["fa"]["no_code_first"]
        md.append("- **Verdict.** Opening the plan types is **not** the fix for the largest FA loss: replies without code (" + f"{nc_fa} of {fa['n']} requests" + ") come from the thinking budget of the model (section 2) and do not depend on the plan type or on the plan text. "
                  "For the ABI errors it is at best a partial remedy: local plans fail less often in this data, but with few local plans and no significance; the dominant ABI error (threadgroup memory) is a resource mistake that any plan type can make and repair already fixes (ok after repair above). "
                  "The 'big changes fail' claim itself is weak (size effect modest and FA-specific, see above), and most FA wins already come from small or medium diffs. "
                  "That local plans raise the *final* FA speedup is not shown by any data here; it is a hypothesis for (d).\n")
    # ---- (b)
    md.append("\n**(b) Could local plans help saturated (elementwise-like) ops?**\n")
    q2 = f["q2"]
    ag = q2["aggregate"]
    if not q2["n_saturated"]:
        md.append("- No finished run reached the saturation fraction.\n")
    else:
        rf = q2["roof_fracs"]
        md.append(f"- {q2['n_saturated']} of {q2['n_finished']} finished runs end at >= {sat:.0%} of the measured roofline (range {rf[0]:.3f}-{rf[-1]:.3f}; dtypes {q2['dtypes']}). "
                  f"{q2['n_effectively_saturated']} of them are also within {1 - sat:.0%} of min(roofline, best baseline): there a kernel sitting exactly *on* the measured roofline would beat the best baseline by a median {fnum(q2['ceiling_median'], 3)}x and at most {fnum(q2['ceiling_max'], 3)}x "
                  f"({q2['n_ceiling_le_1_02']} of them <= 1.02x, i.e. inside the measurement noise of cv 0.5-2% and the 0.5% min_improvement). Their actual speedups are {fnum(q2['speedups_effective'][0], 3) if q2['speedups_effective'] else '-'}-{fnum(q2['speedups_effective'][-1], 3) if q2['speedups_effective'] else '-'}x. "
                  "So the maximum possible gain is a few percent at best: a structural plan has nothing to win there and a local plan can win at most that.\n")
        for r in q2["not_effectively_saturated"]:
            md.append(f"- Exception to 'no room': `{r['run']}` is saturated by the roofline definition ({fnum(r['roof_frac'], 3)}) but runs at {fnum(r['speedup'], 3)}x of the best baseline, which sits at {fnum(r['baseline_roof_frac'], 3)} of the roofline (the roofline is not its binding ceiling). This data does not show that there is no room there.\n")
        md.append(f"- {ag['n_saturated_at_init']} of the {ag['n_saturated']} roofline-saturated runs already had a >= {sat:.0%} kernel at the initial generation. The search then spent {usd(ag['spent_after_first_sat_usd'])} after the first near-saturated kernel "
                  f"= {share(ag['share_of_total_after_first_sat'])} of the {usd(ag['total_usd'])} spent by all finished runs analysed here ({share(ag['share_of_saturated_after_first_sat'])} of what the saturated runs spent): "
                  + "; ".join(f"{s}: {usd(a['spent_after_first_sat_usd'])} of {usd(a['total_usd'])} ({share(a['share_of_total_after_first_sat'])})" for s, a in data["q2"]["aggregate_by_source"].items() if a["n_saturated"])
                  + ". That money bought no gain. A different plan type does not recover it; a roofline-aware stop would (at the default patience of 4 a converged run always pays four more rounds after its last improvement, which a stop rule keyed on the roofline would skip).\n")
    # ---- (c)
    md.append("\n**(c) What the data cannot tell**\n")
    md.append(
        "- There is no single-factor ablation of plan_mix. The plan_mix runs (" + (", ".join(f["mix"]["runs"]) or "none") + ") also differ from the others in budget ($15 vs $3 in the older p1 runs), rounds/patience, profiler evidence and init_samples (see table 0), "
        "and runs differ in model and protocol. No cross-run causal statement is made anywhere in this report.\n"
        "- Each run is one stochastic LLM trajectory; no FlashAttention run was repeated, so run-to-run variance of the final speedup is unknown.\n"
        "- The local/structural comparison inside a plan_mix run is paired (same seed, feedback and round) but small, and the local plan is also the last plan, evaluated last, so it faces a higher bar for 'new best' than an earlier branch of the same round; the position-control rows only partly address this.\n"
        "- Size of diff is observational: the plan, not an experiment, chose it. Line-diff size is a coarse proxy for 'structural'. Truncated replies have no diff, so the size results are conditional on a kernel existing.\n"
        "- Whether a local-first strategy would reach the same or a better final FlashAttention speedup cannot be read from runs that never asked for local plans.\n"
        "- Saturation was measured on fp16 and on whatever finished; bf16 and the remaining tasks of the suite may differ.\n"
    )
    # ---- (d)
    md.append("\n**(d) Cheapest decisive experiment (proposal only; nothing was run or implemented)**\n")
    n_fa, ok_fa = fa["n"], fa["ok_first"]
    ce = f["control_FA"]["earlier"]
    base = (ce["ok_first"] / ce["n"]) if ce["n"] else (ok_fa / n_fa if n_fa else 0.5)
    hi = min(0.95, base + 0.25)
    npa = n_per_arm(base, hi)
    flash = [r for r in data["runs"] if r["run"].endswith("/fa_flash") and r["finished"] and r["rounds"]]
    fa_rounds = [r["rounds"] for r in (flash or [r for r in data["runs"] if r["task"] == "flash_attention" and r["finished"] and r["rounds"]])]
    rpr = max(1, round(statistics.mean(fa_rounds))) if fa_rounds else 7
    runs_needed = math.ceil(npa / rpr) if npa else None
    lo = f["mix"]["local"]
    nct = data["nocode"]
    md.append(
        f"0. **Thinking budget (the cause of the no-code replies).** Take the {nct['recovery_fa']['n']} FlashAttention requests whose first attempt died by the token cap (seeds and plans are on disk) and re-issue them with only one factor changed: `reasoning_effort` (the config says `high`), "
        f"`max_tokens` above {_fmt_int(nct['cap_tokens'])}, or no echo of the truncated reasoning in the retry. A cut reply costs about {usd(nct['length_cost'] / nct['length_turns'], 3) if nct['length_turns'] else '-'} on average, so one arm costs about "
        f"{usd(nct['recovery_fa']['n'] * (nct['length_cost'] / nct['length_turns'] if nct['length_turns'] else 0.06), 1)} and decides whether the lever works before any plan-type work.\n"
    )
    md.append(
        f"1. **`fa_flash_mix`** = `fa_flash` (same task, model, $3 budget, K=3, patience, no profiler, `init_samples=1`) with **only** `plan_mix=True`: the single-factor ablation that does not exist yet. A run of that size has about {rpr} rounds, i.e. {rpr} designated local plans "
        f"against {2 * rpr} structural ones from the same seeds (paired). To detect a first-kernel ok-rate lift from {base:.0%} (earlier plans, FA, no plan_mix) to {hi:.0%} (an assumed +25 points; the pooled local rate in plan_mix runs is {pct(lo['ok_first'], lo['n'])}, n={lo['n']}) "
        f"with alpha 0.05 and power 0.8 one needs about {npa} local plans = about {runs_needed} such runs (about ${3 * runs_needed if runs_needed else '?'} at $3 each). Two runs (about $6, ~{2 * rpr} local vs ~{4 * rpr} structural plans) already give a first paired table; "
        "an end-to-end comparison of the final speedup needs repeats of both arms (about 3 seeds per arm, about $18 on top of the existing `fa_flash`) because one trajectory per arm is dominated by run-to-run noise.\n"
        f"2. **Cheaper first look - seed replay.** {fa['round_seeds']} FlashAttention round seeds are on disk (`candidates/`, reconstructable exactly: see the seed check). Ask for one local plan per seed and compare with the structural outcomes already logged for the same seed. "
        f"At about {usd(fa['usd_per_branch'], 3)} LLM per FA patch request plus about {usd(fa['plan_usd_per_call'], 3)} per planning call, 30 seeds cost roughly {usd(30 * ((fa['usd_per_branch'] or 0.1) + (fa['plan_usd_per_call'] or 0.05)), 0)} and need no search loop; "
        "it measures per-plan success and gain against a fixed seed, not the end-to-end search.\n"
    )
    st = f["streaks"]
    sfa, sno = st["FA"], st["nonFA"]
    fe = [r["eff_frac"] for r in data["q2"]["rows"] if r["task"].split("@")[0] == "flash_attention" and r["eff_frac"]]
    fa_eff_range = f"{min(fe):.2f}-{max(fe):.2f}" if fe else "-"
    md.append(
        "3. **Adaptive rule (test after 1; do not build yet).** Ask for local plans only when (i) the best kernel is within X% of its ceiling, with the ceiling = min(roofline lower bound, best baseline) so that `eff. frac >= 1 - X` means 'almost nothing structural is left' "
        "(suggested X = 10%; the same test is the natural stop rule for elementwise ops, where most of the money went, section 4), or (ii) the last two rounds produced no new best. "
        f"Empirically (section 3.7) a round that follows two rounds without a new best produced one in {sfa['after_2_won']}/{sfa['after_2_no_win']} cases for FA vs {sno['after_2_won']}/{sno['after_2_no_win']} for the other tasks: "
        f"trigger (ii) separates saturated ops from FA, but for FA it fires rarely and the next structural round still often wins, so for FA the ceiling test (i) is the more relevant one (FA runs sit at eff. frac {fa_eff_range}). "
        f"A trigger on 'no ok candidate' would almost never fire on FA ({sfa['rounds_no_ok_first']} rounds without an ok first kernel; repair recovers about {pct(fa['ok_final'], fa['n'])} of the requests).\n"
        "4. Log `plan_mix`, `k` and the declared type of each plan in `state.json` / `result.json`, so that future analyses need not infer them from the planner prompt in the transcript.\n"
    )
    return "\n".join(md)


def render_report(data: dict) -> str:
    meta = data["meta"]
    small_n, sat = meta["small_n"], meta["sat_fraction"]
    md = [
        "# Plan-type evidence report\n",
        f"_Generated {meta['generated']} by `experiments/plan_type_analysis.py` from {meta['n_runs']} run directories under: {', '.join(meta.get('inputs', []))}. Read-only: the inputs (run directories and logs) were only read. "
        "Live runs change this picture while they write: re-run the script to refresh._\n",
        (f"_{meta['note']}_\n" if meta.get("note") else ""),
        "**Caveats, up front.** The runs differ in budget ($3 vs $15), model, search aids, protocol version and in whether they finished (`partial` = no `result.json` yet). "
        f"Every rate is `k/n (p%)`; `†` marks n < {small_n} (anecdotal). `[lo-hi]` are 95% Wilson intervals. Tests are Fisher exact (2x2), Mann-Whitney (diff sizes), Cochran-Mantel-Haenszel (stratified by run). "
        "All size / plan-type comparisons are observational: **no cross-run causal claim is made.**\n",
        "## Answer\n",
        answer(data, sat, small_n),
        "## 0. Runs analysed\n",
        "`seed check` = how many of the seeds reconstructed from the candidate records are identical to the seed printed in the transcript's patch prompt.\n",
        section_runs(data),
        "## 1. Plan kind vs first-attempt outcome (ok / abi_error / incorrect / no_code), per task\n",
        section_kinds(data, small_n),
        "## 2. Replies without code: why, and what they cost\n",
        section_nocode(data, small_n),
        "## 3. Size of the change vs outcome - \"big changes on FlashAttention fail ABI/correctness\"\n",
        section_q1(data, small_n),
        "## 4. Saturated ops - \"elementwise ops have no structural room\"\n",
        section_q2(data, sat, small_n),
        "## 5. Conclusions\n",
        conclusions(data, sat, small_n),
    ]
    return "\n".join(md)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="run directories or directories containing run directories (globs are expanded)")
    ap.add_argument("--out", default="runs/plan_type_analysis", help="output directory for REPORT.md and data.json")
    ap.add_argument("--size-edges", default=",".join(str(e) for e in DEFAULT_SIZE_EDGES), help="diff-fraction edges small|medium|large (default %(default)s)")
    ap.add_argument("--sat", type=float, default=SAT_FRACTION, help="roofline fraction counted as saturated (default %(default)s)")
    ap.add_argument("--logs", nargs="*", default=[], help="orchestrator log files (globs ok): their 'no code block in reply' lines are cross-checked against the transcripts")
    ap.add_argument("--note", default="", help="free text printed under the title of the report (e.g. where the inputs come from)")
    ap.add_argument("--small-n", type=int, default=SMALL_N, help="rates with n below this are flagged as anecdotal (default %(default)s)")
    a = ap.parse_args(argv)
    edges = tuple(float(x) for x in a.size_edges.split(","))
    if len(edges) != 2 or not edges[0] < edges[1]:
        ap.error("--size-edges needs two increasing numbers")
    data = analyse_paths(a.paths, edges=edges, sat=a.sat, small_n=a.small_n, logs=a.logs)
    data["meta"]["note"] = a.note
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1, default=str))
    (out / "REPORT.md").write_text(render_report(data))
    print(f"{data['meta']['n_runs']} runs, {len(data['branches'])} branches, {len(data['attempts'])} attempts -> {out}/REPORT.md, {out}/data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
