"""Shared helpers of the experiment scripts."""

# 各实验脚本共用的启动、目录和预算拼装：论文美元数怎么取、一次搜索怎么交给编排器、结果目录里读哪几份 json。
# 套件、重复性、具名跑法和语义测试都从这里进同一次搜索，避免每份脚本各写一套 $15。

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from taic.llm import PAPER_BUDGET_USD, load_config
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
    config_path: str | None = None,
    preset: str | None = None,
    client: Any = None,
    evaluate_fn: Callable | None = None,
    log: Callable[[str], None] = print,
    use_tools: bool | None = None,
    **search_overrides,
) -> RunResult:
    cfg, search = load_config(config_path, model=model, preset=preset, budget_usd=budget, branches=branches, max_rounds=max_rounds, **search_overrides)
    if use_tools is not None:
        # 只有调用方显式传入时才覆盖预设里的开关。
        cfg.use_tools = use_tools
    orch = Orchestrator(task, dtype, size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn, log=log)
    return asyncio.run(orch.run())


# 各脚本的 --resume 只认这一份文件在不在。不在就当这次还没跑完；candidates/ 里的半成品不会被接上。
def read_result(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "result.json"
    # 只看文件在不在。写到一半仍然算「有」，解析失败会抛出去，看板那种「截断当缺失」不在这里。
    return json.loads(p.read_text()) if p.exists() else None


# 换种子复测写在另一份文件。没有它时「相对 triton-msl」一列为空，搜索本身仍算跑完。
def read_confirm(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "confirm_report.json"
    return json.loads(p.read_text()) if p.exists() else None


# 一行里同时留下搜索期加速比、换种子确认后的加速比，以及手写朴素核，后面才能把「基线弱」和「搜索强」分开。
def summarize(res: dict, confirm: dict | None, naive: dict | None = None) -> dict:
    """One row of the suite table."""
    row = {
        "task": res["task"], "dtype": res["dtype"], "model": res["model"], "stop_reason": res["stop_reason"], "rounds": res["rounds"],
        "evaluations": res["evaluations"], "spent_usd": res["spent_usd"], "budget_usd": res["budget_usd"],
        "best_speedup": res.get("best_speedup"), "confirmed_speedup": res.get("confirmed_speedup"),
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
    return row


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
