"""Shared helpers of the experiment scripts."""

# 各实验脚本共用的启动、目录和预算拼装：论文美元数怎么取、一次搜索怎么交给编排器、结果目录里读哪几份 json。
# 套件、重复性、具名跑法和语义测试都从这里进同一次搜索，避免每份脚本各写一套 $15。

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from taic.committees import load_committees
from taic.llm import PAPER_BUDGET_USD, SearchConfig, load_config, load_raw_config
from taic.orchestrator import Orchestrator, RunResult

# Per-kernel USD budgets: the paper's, in dollars, whatever the model. Every kernel gets the paper's $15 (Sections 4.3 and 4.4 of
# arXiv 2609.36800; the 12-kernel grid of Figure 4 states no cap, so the same ceiling is used). Runs that converge earlier (``patience``) spend less.
# Earlier versions of this file used 0.5-3.0 USD per kernel (a fraction of the paper's, scaled by price); those results are
# not comparable with these (the dashboard and each result.json show the budget a run actually had).
# The one exception, Table 1 (repeatability, Section 4.3): the elementwise SwiGLU kernel is given $1, the others $15.
# 这张表只有重复性实验（table1=True）会用。套件和具名跑法不读它。
TABLE1_BUDGETS: dict[str, float] = {"swiglu": 1.0}


# 这里只回答「这一次最多花多少美元」。scale 乘的是这笔钱，不改轮数、分支、耐心。
# table1=True 时 SwiGLU 是 $1、其余是论文的 $15；名字里 @ 后面的启动契约跟基算子共用同一笔预算。
def budget_for(task: str, scale: float = 1.0, *, table1: bool = False) -> float:
    """Budget of one run of ``task`` times ``scale``: the paper's $15 (``table1=True``: Table 1's per-kernel budgets, used by the repeatability experiment)."""
    base = task.split("@", 1)[0]  # a contract variant such as gemm@128x64x32t256 has the budget of gemm
    return (TABLE1_BUDGETS.get(base, PAPER_BUDGET_USD) if table1 else PAPER_BUDGET_USD) * scale


# 套件上限看的是「已经花掉的 + 这一次的整份预算」，提前停下来省下的钱不预支。
# max_total_usd 为空就是不设上限；加上 1e-9 只为了刚好卡在上限上时仍放行。
def fits_cap(spent_usd: float, budget_usd: float, max_total_usd: float | None) -> bool:
    """``--max-total-usd``: a run may start only if its whole budget still fits under the experiment-wide cap (``None``: no cap)."""
    return max_total_usd is None or spent_usd + budget_usd <= max_total_usd + 1e-9


CAP_FILE = "cap.json"


def _num(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN is not a commitment


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# 一次搜索占多少钱：活着的进程按「当前预算和已花费里更大的那个」预留（看板加过预算、单独续跑的都算），
# 已经结束的只算实际花掉的。实验脚本自己的 state.json 没有 spent_usd，不算进这笔账。
def run_commitment(run_dir: Path) -> float:
    """Dollars reserved by one search directory: its live budget while a process holds it, otherwise what it spent."""
    from taic.control import pid_alive

    run_dir = Path(run_dir)
    state = _read(run_dir / "state.json")
    state = state if isinstance(state, dict) else {}
    if "spent_usd" not in state and "script" in state:
        return 0.0
    result = _read(run_dir / "result.json")
    result = result if isinstance(result, dict) else {}
    spent = _num(state.get("spent_usd"))
    if spent is None:
        spent = _num(result.get("spent_usd")) or 0.0
    control = _read(run_dir / "control.json")
    budget = _num(control.get("budget_usd")) if isinstance(control, dict) else None
    if budget is None:
        budget = _num(state.get("budget_usd"))
    if budget is None:
        budget = _num(result.get("budget_usd"))
    lock = _read(run_dir / "resume.lock")
    live = pid_alive(lock.get("pid")) if isinstance(lock, dict) else False
    if not live and "spent_usd" in state:
        live = pid_alive(state.get("pid")) and not result
    if live:
        return max(spent, budget or 0.0)
    return float(result.get("spent_usd") if result and _num(result.get("spent_usd")) is not None else spent)


def experiment_commitment(exp_dir: Path) -> float:
    """Reserved dollars of every search under ``exp_dir`` (finished runs count what they spent; live ones count their budget).

    A directory is a search when it holds ``state.json`` or ``result.json``. The experiment's own ``state.json`` (it has ``script`` and no
    ``spent_usd``) contributes nothing, and a directory is counted once even when it has both files.
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        return 0.0
    dirs = {p.parent.resolve() for name in ("state.json", "result.json") for p in exp_dir.rglob(name)}
    return sum(run_commitment(d) for d in dirs)


def write_cap(exp_dir: Path, max_total_usd: float | None) -> None:
    """Record the experiment cap so the dashboard can refuse a top-up that would break it. ``None`` writes nothing."""
    if max_total_usd is None:
        return
    from taic.control import write_json_atomic

    write_json_atomic(Path(exp_dir) / CAP_FILE, {"max_total_usd": float(max_total_usd)})


def read_cap(exp_dir: Path) -> float | None:
    doc = _read(Path(exp_dir) / CAP_FILE)
    return _num(doc.get("max_total_usd")) if isinstance(doc, dict) else None


def find_cap(run_dir: Path) -> tuple[Path, float] | None:
    """The nearest ancestor of ``run_dir`` that has a ``cap.json``, and the cap it records."""
    p = Path(run_dir).resolve().parent
    while True:
        cap = read_cap(p)
        if cap is not None:
            return p, cap
        if p.parent == p:
            return None
        p = p.parent


def cap_allows(run_dir: Path, new_total: float) -> tuple[bool, str]:
    """Whether raising ``run_dir`` to ``new_total`` still fits the experiment cap. No cap file means it is allowed."""
    found = find_cap(run_dir)
    if found is None:
        return True, ""
    exp_dir, cap = found
    this = run_commitment(run_dir)
    reserved = experiment_commitment(exp_dir) - this + max(this, float(new_total))
    if reserved <= cap + 1e-9:
        return True, ""
    return False, f"refused: ${reserved:.2f} would be reserved under the experiment cap of ${cap:.2f} ({exp_dir / CAP_FILE})"


# 单次搜索的唯一入口。新的 run_dir 会启动一次空的搜索，本函数不读上一次的最优核。
# budget 只替换预设里的美元字段；patience、max_rounds 等仍来自预设，除非调用方在参数里另给。
def run_one(
    task: str,
    dtype: str,
    size: str,
    run_dir: Path,
    *,
    budget: float | None = None,
    branches: int | None = None,
    max_rounds: int | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    config_path: str | None = None,
    preset: str | None = None,
    client: Any = None,
    evaluate_fn: Callable | None = None,
    agent_runner: Callable | None = None,
    log: Callable[[str], None] = print,
    use_tools: bool | None = None,
    **search_overrides,
) -> RunResult:
    raw = load_raw_config(config_path, preset)
    cfg, search = load_config(config_path, model=model, preset=preset, budget_usd=budget, branches=branches, max_rounds=max_rounds, **search_overrides)
    if use_tools is not None:
        # 只有调用方显式传入时才覆盖预设里的开关。
        cfg.use_tools = use_tools
    if max_tokens is not None:
        # 单次回复上限，思考也算。不传就用 taic.toml 的 48000。
        cfg.max_tokens = max_tokens
    # 显式 model 让两个委员会都用它。否则用配置里的两名成员。RunStartError（目录已有文件，或 run_config.json 写失败）从这里原样抛给调用方，不进入 run()。
    committees = load_committees(raw, active=cfg, model_override=model)
    orch = Orchestrator(task, dtype, size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn, log=log, committees=committees, agent_runner=agent_runner)
    return asyncio.run(orch.run())


# 各脚本的 --resume 只在 result.json 不存在时才开新的一次搜索。文件在，但损坏或版本更新，要人来看，不能当成还没跑过。
def read_result(run_dir: Path) -> dict | None:
    from taic.artifacts import RESULT_SCHEMA_VERSION, read_versioned

    return read_versioned(Path(run_dir) / "result.json", RESULT_SCHEMA_VERSION)


# 换种子复测写在另一份文件。没有它时「相对 triton-msl」一列为空，搜索本身仍算跑完。文件损坏则不能当成没测过。
def read_confirm(run_dir: Path) -> dict | None:
    from taic.artifacts import read_versioned
    from tcenv.schema import REPORT_SCHEMA_VERSION

    return read_versioned(Path(run_dir) / "confirm_report.json", REPORT_SCHEMA_VERSION)


# 一行里同时留下搜索期加速比、换种子确认后的加速比，以及手写朴素核，后面才能把「基线弱」和「搜索强」分开。
def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _confirmation_files(run_dir: Path, winner_label: str | None) -> tuple[str | None, str | None, str | None]:
    """(best.metal sha256, winner candidate kernel.metal sha256, confirm_report status)."""
    best = _sha256_file(run_dir / "best.metal")
    report = None
    try:
        doc = json.loads((run_dir / "confirm_report.json").read_text())
        if isinstance(doc, dict):
            report = doc.get("status")
    except (OSError, ValueError):
        report = None
    winner = None
    if isinstance(winner_label, str) and winner_label:
        root = run_dir / "candidates"
        if root.is_dir():
            found = [p for p in root.iterdir() if p.is_dir() and len(p.name) > 4 and p.name[3] == "_" and p.name[4:] == winner_label]
            if len(found) == 1:
                winner = _sha256_file(found[0] / "kernel.metal")
    return best, winner, report


def conclusion_of_run(run_dir: Path, result: dict | None) -> dict:
    """The publication verdict for one run directory. A bare ``result.json`` is not enough to call a number confirmed."""
    if not result:
        return published_conclusion(None)
    best, winner, status = _confirmation_files(Path(run_dir), result.get("confirm_winner_label"))
    got = published_conclusion(result, best_metal_sha256=best, confirm_status=status, winner_metal_sha256=winner)
    if got["publication"] == "confirmed":
        try:
            doc = json.loads((Path(run_dir) / "confirm_report.json").read_text())
        except (OSError, ValueError):
            doc = None
        from tcenv.bench import timing_is_uncertain

        if timing_is_uncertain(doc):
            return {"speedup": None, "publication": "不确定"}
        from taic.policy import batches_agree

        # 同一颗核各批极差超过 2% 时，中位数仍留在 result.json，对外不把它当成精确值。
        if not batches_agree(result.get("confirmed_values")):
            return {"speedup": None, "publication": "不确定"}
    return got


def published_conclusion(result: dict | None, *, best_metal_sha256: str | None = None, confirm_status: str | None = None, winner_metal_sha256: str | None = None) -> dict:
    """What a finished run may show as its confirmed speedup.

    A number is returned only when ``confirmed_status`` is ok, the confirm report is ok, ``best.metal`` and the
    winner candidate's ``kernel.metal`` both hash to ``confirmation_source_sha256``, and ``confirm_winner_label``
    named that candidate. A result that has no ``confirmation_source_sha256`` key is an older file: ``旧版未绑定源码``.
    Anything else that is finished is ``未确认`` and does not fall back to ``best_speedup``.
    """
    if not result:
        return {"speedup": None, "publication": "in_progress"}
    if "confirmation_source_sha256" not in result:
        return {"speedup": None, "publication": "旧版未绑定源码"}
    recorded = result.get("confirmation_source_sha256")
    winner = result.get("confirm_winner_label")
    bound = (
        result.get("confirmed_status") == "ok"
        and confirm_status == "ok"
        and isinstance(recorded, str) and recorded != ""
        and recorded == best_metal_sha256 == winner_metal_sha256
        and isinstance(winner, str) and winner != ""
    )
    if bound:
        return {"speedup": result.get("confirmed_speedup"), "publication": "confirmed"}
    return {"speedup": None, "publication": "未确认"}


def publication_role(
    conclusion: dict | None,
    domain: dict | None,
    *,
    source_sha256: str | None,
    task_fingerprint: str | None,
    recorded_fingerprint: str | None,
    min_improvement: float | None = None,
) -> str | None:
    """``primary`` when a confirmed speedup clears the noise gate and a passing domain sweep is bound to that same source and task.

    ``None`` while the run is still in progress. Anything else, including a correct kernel that is slower than the baselines, is ``fallback``.
    The value is derived. It is not written into ``result.json``.
    """
    if conclusion is None or conclusion.get("publication") == "in_progress":
        return None
    if min_improvement is None:
        min_improvement = SearchConfig.min_improvement
    speed = conclusion.get("speedup")
    cov = {}
    if isinstance(domain, dict):
        nested = domain.get("coverage")
        cov = nested if isinstance(nested, dict) else domain
    speed_ok = isinstance(speed, (int, float)) and not isinstance(speed, bool) and speed >= min_improvement
    bound = (
        conclusion.get("publication") == "confirmed"
        and speed_ok
        and cov.get("kind") == "domain_sweep"
        and cov.get("passed") is True
        and isinstance(source_sha256, str) and source_sha256 != ""
        and cov.get("source_sha256") == source_sha256
        and isinstance(task_fingerprint, str) and task_fingerprint != ""
        and cov.get("task_fingerprint") == task_fingerprint == recorded_fingerprint
    )
    return "primary" if bound else "fallback"


def shown_speedup(row: dict):
    """The number a summary table may print. ``None`` when the publication gate did not accept one.

    A row that never went through the gate has no ``published_speedup`` key and prints nothing, rather than
    ``confirmed_speedup`` or ``best_speedup``.
    """
    if "published_speedup" not in row:
        return None
    return row.get("published_speedup")


def summarize(res: dict, confirm: dict | None, naive: dict | None = None, *, run_dir: Path | None = None) -> dict:
    """One row of the suite table. ``published_speedup`` is set only for a bound, quiet confirmation."""
    if run_dir is not None:
        conclusion = conclusion_of_run(Path(run_dir), res)
    else:
        conclusion = published_conclusion(res, confirm_status=(confirm or {}).get("status"))
    row = {
        "task": res["task"], "dtype": res["dtype"], "model": res["model"], "stop_reason": res["stop_reason"], "rounds": res["rounds"],
        "evaluations": res["evaluations"], "spent_usd": res["spent_usd"], "budget_usd": res["budget_usd"],
        "best_speedup": res.get("best_speedup"), "confirmed_speedup": res.get("confirmed_speedup"),
        "published_speedup": conclusion["speedup"], "publication": conclusion["publication"],
        "candidate_us": res.get("candidate_us"), "baseline_us": res.get("baseline_us") or {},
        "profiles": len(res.get("profiles") or []), "spilled_bytes_seen": max([p.get("spilled_bytes") or 0 for p in res.get("profiles") or []], default=0),
    }
    bl = row["baseline_us"]
    # 字典里的值是耗时（微秒），最小的那个才是最强基线。
    row["best_baseline"] = min(bl, key=bl.get) if bl else None
    cb = (confirm or {}).get("compiler_baseline") or {}
    row["naive_speedup"] = (naive or {}).get("speedup")
    row["vs_triton_msl"] = cb.get("speedup")
    row["triton_msl_us"] = cb.get("estimated_us")
    published = row.get("published_speedup")
    naive_speed = row.get("naive_speedup")
    spent = row.get("spent_usd")
    if isinstance(published, (int, float)) and not isinstance(published, bool) and isinstance(naive_speed, (int, float)) and not isinstance(naive_speed, bool) and naive_speed:
        row["gain_over_naive"] = published / naive_speed
    else:
        row["gain_over_naive"] = None
    if row["gain_over_naive"] is not None and isinstance(spent, (int, float)) and not isinstance(spent, bool) and spent > 0:
        row["gain_per_usd"] = (published - naive_speed) / spent
    else:
        row["gain_per_usd"] = None
    return row


def suite_metrics(rows: list[dict], thresholds: tuple[float, ...] = (0.95, 1.0, 1.005)) -> dict:
    """fast_p counts rows whose published speedup is at least p. Search-only speedups are not counted."""
    finished = [row for row in rows if isinstance(row, dict)]
    out: dict[str, Any] = {"n": len(finished), "spent_usd": 0.0, "fast": {}}
    spent = 0.0
    published_values = []
    for row in finished:
        value = row.get("spent_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            spent += float(value)
        speed = row.get("published_speedup")
        if isinstance(speed, (int, float)) and not isinstance(speed, bool):
            published_values.append(float(speed))
    out["spent_usd"] = spent
    out["published_n"] = len(published_values)
    if published_values:
        ordered = sorted(published_values)
        out["published_median"] = ordered[len(ordered) // 2]
    else:
        out["published_median"] = None
    for threshold in thresholds:
        hits = sum(1 for speed in published_values if speed >= threshold)
        out["fast"][f"{threshold:g}"] = {"n": hits, "share": (hits / len(finished)) if finished else None}
    return out


# 手写朴素核不是搜索的起点，只是同一套评测协议下的参照，用来看出加速比里有多少来自基线太弱。
# 按题和小数缓存在套件目录，同一格重跑或 --resume 时不再占 GPU。
def naive_reference(task: str, dtype: str, size: str, cache_dir: Path, *, evaluate_fn: Callable | None = None) -> dict:
    """Evaluate the task's hand-written *naive* kernel under the same protocol. Its speedup over the baselines is the reference
    point that makes weak baselines visible: if a naive kernel already beats MLX/MPS (as for unfused ops such as Lion), part of
    any AI-lowering speedup is baseline weakness, not compiler skill."""
    from tasks import get_task
    from tcenv.evaluate import evaluate

    p = Path(cache_dir) / f"naive_{task}_{dtype}.json"
    if p.exists():
        # 缓存看文件在不在，不看这次搜索的 result.json。参照测量和模型搜索是两条记录。
        return json.loads(p.read_text())
    t = get_task(task, dtype, size)
    rep = (evaluate_fn or evaluate)(task, dtype, size, t.naive_msl, mode="full", seed=None, reps=15)
    out = {"status": rep.status, "speedup": rep.speedup, "candidate_us": rep.candidate_us, "baseline_us": rep.baseline_us}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return out
