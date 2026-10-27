"""Shared helpers of the experiment scripts."""

# 各实验脚本共用的启动、目录和预算拼装：论文美元数怎么取、一次搜索怎么交给编排器、结果目录里读哪几份 json。
# 套件、重复性、具名跑法和语义测试都从这里进同一次搜索，避免每份脚本各写一套 $15。

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
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
    """@brief 算一次运行最多能花多少美元。

    Budget of one run of ``task`` times ``scale``: the paper's $15 (``table1=True``: Table 1's per-kernel budgets, used by the repeatability experiment).

    @param task 任务名；带 ``@启动契约`` 的变体按基算子的预算算。
    @param scale 预算倍数，只乘这笔钱。
    @param table1 为真时用 Table 1 的逐题预算（SwiGLU $1，其余 $15）。
    @return 这次运行的美元上限。
    """
    base = task.split("@", 1)[0]  # a contract variant such as gemm@128x64x32t256 has the budget of gemm
    return (TABLE1_BUDGETS.get(base, PAPER_BUDGET_USD) if table1 else PAPER_BUDGET_USD) * scale


# 套件上限看的是「已经花掉的 + 这一次的整份预算」，提前停下来省下的钱不预支。
# max_total_usd 为空就是不设上限；加上 1e-9 只为了刚好卡在上限上时仍放行。
def fits_cap(spent_usd: float, budget_usd: float, max_total_usd: float | None) -> bool:
    """@brief 判断在实验总额上限下还能不能再开一次运行。

    ``--max-total-usd``: a run may start only if its whole budget still fits under the experiment-wide cap (``None``: no cap).

    @param spent_usd 实验已经花掉的钱。
    @param budget_usd 这次运行的整份预算。
    @param max_total_usd 实验总额上限；为 ``None`` 时不设上限。
    @return 放得下时为 ``True``。
    """
    return max_total_usd is None or spent_usd + budget_usd <= max_total_usd + 1e-9


CAP_FILE = "cap.json"


def _num(value) -> float | None:
    """@brief 把值转成开销账上可用的浮点数。

    @param value 任意数值或字符串。
    @return 转出来的浮点数；布尔、空值、不能转和 NaN 都返回 ``None``。
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN is not a commitment


def _read(path: Path) -> Any:
    """@brief 读一个 JSON 文件，读不出来时返回 ``None``。

    @param path JSON 文件路径。
    @return 解析结果；文件缺失或不是合法 JSON 时为 ``None``。
    """
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# 一次搜索占多少钱：活着的进程按「当前预算和已花费里更大的那个」预留（看板加过预算、单独续跑的都算），
# 已经结束的只算实际花掉的。实验脚本自己的 state.json 没有 spent_usd，不算进这笔账。
def run_commitment(run_dir: Path) -> float:
    """@brief 算一次搜索当前占用的美元数。

    活着时按「当前预算与已花费里更大的那个」预留，已结束时只算实际花掉的钱。

    Dollars reserved by one search directory: its live budget while a process holds it, otherwise what it spent.

    @param run_dir 单个搜索目录。
    @return 这本账上预留的美元数。
    """
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
    """@brief 算整个实验目录下所有搜索占用的美元数。

    Reserved dollars of every search under ``exp_dir`` (finished runs count what they spent; live ones count their budget).

    A directory is a search when it holds ``state.json`` or ``result.json``. The experiment's own ``state.json`` (it has ``script`` and no
    ``spent_usd``) contributes nothing, and a directory is counted once even when it has both files.

    @param exp_dir 实验根目录。
    @return 预留金额合计；目录不存在时为 0。
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        return 0.0
    dirs = {p.parent.resolve() for name in ("state.json", "result.json") for p in exp_dir.rglob(name)}
    return sum(run_commitment(d) for d in dirs)


def write_cap(exp_dir: Path, max_total_usd: float | None) -> None:
    """@brief 把实验的总额上限写进 ``cap.json``，供看板拒绝超额的追加预算。

    Record the experiment cap so the dashboard can refuse a top-up that would break it. ``None`` writes nothing.

    @param exp_dir 实验根目录。
    @param max_total_usd 总额上限；为 ``None`` 时不写文件。
    """
    if max_total_usd is None:
        return
    from taic.control import write_json_atomic

    write_json_atomic(Path(exp_dir) / CAP_FILE, {"max_total_usd": float(max_total_usd)})


def read_cap(exp_dir: Path) -> float | None:
    """@brief 读回实验的总额上限。

    @param exp_dir 实验根目录。
    @return 上限美元数；没有 ``cap.json`` 或读不出数值时返回 ``None``。
    """
    doc = _read(Path(exp_dir) / CAP_FILE)
    return _num(doc.get("max_total_usd")) if isinstance(doc, dict) else None


def find_cap(run_dir: Path) -> tuple[Path, float] | None:
    """@brief 往上找最近的带 ``cap.json`` 的祖先目录。

    The nearest ancestor of ``run_dir`` that has a ``cap.json``, and the cap it records.

    @param run_dir 搜索目录。
    @return ``(带 cap.json 的目录, 上限美元数)``；一直找到根都没有时返回 ``None``。
    """
    p = Path(run_dir).resolve().parent
    while True:
        cap = read_cap(p)
        if cap is not None:
            return p, cap
        if p.parent == p:
            return None
        p = p.parent


def cap_allows(run_dir: Path, new_total: float) -> tuple[bool, str]:
    """@brief 判断把这次运行的预算提到 ``new_total`` 会不会超实验总额。

    Whether raising ``run_dir`` to ``new_total`` still fits the experiment cap. No cap file means it is allowed.

    @param run_dir 搜索目录。
    @param new_total 提议的新预算总额。
    @return ``(是否放行, 拒绝原因)``；放行时原因为空串。
    """
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
    target_id: str | None = None,
    seed: int | None = None,
    **search_overrides,
) -> RunResult:
    """@brief 启动一次搜索的唯一入口，把预算与覆盖项交给编排器并跑完。

    新的 ``run_dir`` 会启动一次空搜索，不读上一次的最优核；目录里已有可续跑的快照时改为续跑。

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模档位。
    @param run_dir 本次运行的输出目录。
    @param budget 本次的美元上限；为 ``None`` 时用预设里的值。
    @param branches 分支数；为 ``None`` 时用预设。
    @param max_rounds 轮数上限；为 ``None`` 时用预设。
    @param model 模型名；显式给出时两个委员会都用它。
    @param max_tokens 单次回复上限；为 ``None`` 时用配置里的值。
    @param config_path 配置文件路径；为 ``None`` 时用默认。
    @param preset 配置预设名；为 ``None`` 时用默认。
    @param client LLM 客户端；为 ``None`` 时由编排器自建。
    @param evaluate_fn 评测函数；为 ``None`` 时用默认实现。
    @param agent_runner 自定义的写内核进程；为 ``None`` 时用默认。
    @param log 进度输出函数。
    @param use_tools 是否开工具；为 ``None`` 时不覆盖预设。
    @param search_overrides 其余透传给 ``load_config`` 的搜索开关。
    @return 这次搜索的结果对象。
    @raises taic.orchestrator.RunStartError 目录已占用而不可续跑，或 ``run_config.json`` 写不下去时。
    """
    raw = load_raw_config(config_path, preset)
    cfg, search = load_config(config_path, model=model, preset=preset, budget_usd=budget, branches=branches, max_rounds=max_rounds, **search_overrides)
    if use_tools is not None:
        # 只有调用方显式传入时才覆盖预设里的开关。
        cfg.use_tools = use_tools
    if max_tokens is not None:
        # 单次回复上限，思考也算。不传就用 taic.toml 的 48000。
        cfg.max_tokens = max_tokens
    # 显式 model 让两个委员会都用它。否则用配置里的两名成员。RunStartError（目录已有文件，或 run_config.json 写失败）从这里原样抛给调用方，不进入 run()。
    if _looks_resumable(Path(run_dir)):
        log(f"(resume interrupted cell {run_dir})")
        return _resume_one(task, dtype, size, Path(run_dir), cfg, search, raw, model, client, evaluate_fn, agent_runner, log)
    committees = load_committees(raw, active=cfg, model_override=model)
    orch = Orchestrator(task, dtype, size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn, log=log, committees=committees, agent_runner=agent_runner, target_id=target_id or "apple-m4-max", seed=seed)
    return asyncio.run(orch.run())


def _looks_resumable(run_dir: Path) -> bool:
    """@brief 判断目录里是否已经有续跑所需的三份文件。

    A non-empty cell with the three files a flame-chase snapshot needs; a stray file dir is still refused.

    @param run_dir 搜索目录。
    @return 三份文件都在时为 ``True``。
    """
    return (run_dir / "run_config.json").is_file() and (run_dir / "state.json").is_file() and (run_dir / "transcript.jsonl").is_file()


def _resume_one(task, dtype, size, run_dir, cfg, search, raw, model, client, evaluate_fn, agent_runner, log) -> RunResult:
    """@brief 接着一份已中断的快照继续跑，而不是拒绝非空目录。

    Continue an interrupted batch cell instead of refusing the non-empty directory.

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模档位。
    @param run_dir 被续跑的目录。
    @param cfg 已解析的配置。
    @param search 搜索开关。
    @param raw 原始配置字典，用于重建委员会。
    @param model 模型名；为 ``None`` 时沿用配置里记录的委员会。
    @param client LLM 客户端。
    @param evaluate_fn 评测函数。
    @param agent_runner 自定义的写内核进程。
    @param log 进度输出函数。
    @return 续跑后的结果对象。
    @raises taic.orchestrator.RunStartError 快照读不出来时。
    """
    from tasks import get_task

    from taic import resume as rs
    from taic.orchestrator import RunStartError

    task_obj = get_task(task, dtype, size)
    try:
        snap = rs.load_snapshot(run_dir, search, task_obj.contract.kernel_name)
    except rs.ResumeError as exc:
        raise RunStartError(f"{run_dir} is not resumable: {exc}") from exc
    recorded = {}
    try:
        recorded = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        recorded = {}
    recorded_committees = recorded.get("committees") if isinstance(recorded, dict) else None
    if isinstance(recorded_committees, list) and recorded_committees and not model:
        committees = load_committees({"llm": raw.get("llm") or {}, "committees": recorded_committees}, active=cfg)
    else:
        committees = load_committees(raw, active=cfg, model_override=model)
    orch = Orchestrator(
        task, dtype, size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn,
        log=log, committees=committees, agent_runner=agent_runner, resuming=True,
    )
    orch.restore(snap, dry_run=False)
    return asyncio.run(orch.run())


# 各脚本的 --resume 只在 result.json 不存在时才开新的一次搜索。文件在，但损坏或版本更新，要人来看，不能当成还没跑过。
def read_result(run_dir: Path) -> dict | None:
    """@brief 读一次运行的 ``result.json``，并按版本校验。

    @param run_dir 搜索目录。
    @return 结果字典；还没跑完时返回 ``None``。
    @raises taic.artifacts.NeedsInspection 文件存在但损坏、版本对不上或新版化时，要人来看。
    """
    from taic.artifacts import RESULT_SCHEMA_VERSION, read_versioned

    return read_versioned(Path(run_dir) / "result.json", RESULT_SCHEMA_VERSION)


# 换种子复测写在另一份文件。没有它时「相对 triton-msl」一列为空，搜索本身仍算跑完。文件损坏则不能当成没测过。
def read_confirm(run_dir: Path) -> dict | None:
    """@brief 读换种子复测的 ``confirm_report.json``。

    @param run_dir 搜索目录。
    @return 复测报告字典；没测过时返回 ``None``。
    @raises taic.artifacts.NeedsInspection 报告损坏或版本对不上时，不能当成没测过。
    """
    from taic.artifacts import read_versioned
    from tcenv.schema import REPORT_SCHEMA_VERSION

    return read_versioned(Path(run_dir) / "confirm_report.json", REPORT_SCHEMA_VERSION)


# 一行里同时留下搜索期加速比、换种子确认后的加速比，以及手写朴素核，后面才能把「基线弱」和「搜索强」分开。
def _sha256_file(path: Path) -> str | None:
    """@brief 算文件的 sha256。

    @param path 文件路径。
    @return 十六进制摘要；文件读不到时返回 ``None``。
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _confirmation_files(run_dir: Path, winner_label: str | None) -> tuple[str | None, str | None, str | None]:
    """@brief 取出确认门需要比对的三份证据。

    (best.metal sha256, winner candidate kernel.metal sha256, confirm_report status).

    @param run_dir 搜索目录。
    @param winner_label 结果里记录的胜者标签。
    @return ``(best.metal 摘要, 胜者候选摘要, 复测报告状态)``，缺的那项为 ``None``。
    """
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
    """@brief 给一个运行目录出发布结论：只有确认门绑住重测内核时才给数字。

    The publication verdict for one run directory. A bare ``result.json`` is not enough to call a number confirmed.

    @param run_dir 搜索目录。
    @param result 该目录的结果字典；可以为 None。
    @return ``{"speedup": ..., "publication": ...}``；缓存命中、计时不稳或各批不一致时标为 ``不确定``。
    """
    return _conclusion_verdict(run_dir, result)[0]


def publication_reason(run_dir: Path, result: dict | None) -> dict:
    """@brief 发布判定 + 原因码（只给观测用；``conclusion_of_run`` 仍只返回两个键）。

    原因码按代码顺序给出“第一个命中的分支”，可直接映射到行动（重新确认、修协议、等安静测量等），
    不发明组合码。旧格式确认报告只报 ``missing_confirmation_block``，协议缺失等其他问题不进 reason。

    @param run_dir 搜索目录。
    @param result 该目录的结果字典；可以为 None。
    @return ``{"publication": ..., "reason": ...}``。
    """
    got, reason = _conclusion_verdict(run_dir, result)
    return {"publication": got["publication"], "reason": reason}


def _conclusion_verdict(run_dir: Path, result: dict | None) -> tuple[dict, str]:
    """@brief ``conclusion_of_run`` 的内部实现：返回（结论, 原因码）。

    @param run_dir 搜索目录。
    @param result 该目录的结果字典；可以为 None。
    @return ``({"speedup": ..., "publication": ...}, reason)``。
    """
    if not result:
        return published_conclusion(None), "no_result_json"
    best, winner, status = _confirmation_files(Path(run_dir), result.get("confirm_winner_label"))
    got, reason = _published_verdict(
        result, best_metal_sha256=best, confirm_status=status, winner_metal_sha256=winner,
    )
    if got["publication"] != "confirmed":
        return got, reason
    try:
        doc = json.loads((Path(run_dir) / "confirm_report.json").read_text())
    except (OSError, ValueError):
        doc = None
    from tcenv.bench import timing_is_uncertain

    from tcenv.schema import PROTOCOL_VERSION

    if isinstance(doc, dict) and doc.get("cache_hit") is True:
        # 确认报告若来自评测缓存，它只是复述了上一次的测量，不是这一次的复测。
        # 缓存键不含设备、macOS 与 MSL 版本，也不能证明是同一台机器上量的。
        return {"speedup": None, "publication": "不确定"}, "cache_hit"
    if not isinstance(doc, dict):
        return {"speedup": None, "publication": "未确认"}, "missing_confirmation_block"
    benchmark = doc.get("benchmark")
    confirmation = benchmark.get("confirmation") if isinstance(benchmark, dict) else None
    if not isinstance(confirmation, dict):
        return {"speedup": None, "publication": "未确认"}, "missing_confirmation_block"
    if confirmation.get("kernel_sha256") != result.get("confirmation_source_sha256"):
        return {"speedup": None, "publication": "未确认"}, "sha_mismatch"
    if confirmation.get("winner_label") != result.get("confirm_winner_label"):
        return {"speedup": None, "publication": "未确认"}, "label_mismatch"
    if confirmation.get("confirmed_values") != result.get("confirmed_values"):
        return {"speedup": None, "publication": "未确认"}, "values_mismatch"
    if confirmation.get("speedup") != result.get("confirmed_speedup") or doc.get("speedup") != result.get("confirmed_speedup"):
        return {"speedup": None, "publication": "未确认"}, "speedup_mismatch"
    if result.get("candidate_us") is not None and confirmation.get("candidate_us") != result.get("candidate_us"):
        return {"speedup": None, "publication": "未确认"}, "candidate_us_mismatch"
    # 数字只在同一个测量协议、同一台目标机上可比。缺字段按不可核对处理。
    if str(doc.get("protocol_version") or "") != str(PROTOCOL_VERSION):
        return {"speedup": None, "publication": "未确认"}, "protocol_mismatch"
    if str(doc.get("target_id") or "") != str(result.get("target_id") or ""):
        return {"speedup": None, "publication": "未确认"}, "target_mismatch"
    if timing_is_uncertain(doc):
        return {"speedup": None, "publication": "不确定"}, "timing_uncertain"
    from taic.policy import batches_agree

    # 同一颗核各批极差超过 2% 时，中位数仍留在 result.json，对外不把它当成精确值。
    if not batches_agree(result.get("confirmed_values")):
        return {"speedup": None, "publication": "不确定"}, "batches_disagree"
    return got, "confirmed"


def _finite_positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def published_conclusion(result: dict | None, *, best_metal_sha256: str | None = None, confirm_status: str | None = None, winner_metal_sha256: str | None = None) -> dict:
    """@brief 只凭已有字段判断这份结果能不能报一个确认加速比。

    What a finished run may show as its confirmed speedup.

    A number is returned only when ``confirmed_status`` is ok, the confirm report is ok, ``best.metal`` and the
    winner candidate's ``kernel.metal`` both hash to ``confirmation_source_sha256``, and ``confirm_winner_label``
    named that candidate. A result that has no ``confirmation_source_sha256`` key is an older file: ``旧版未绑定源码``.
    Anything else that is finished is ``未确认`` and does not fall back to ``best_speedup``.

    @param result 结果字典；可以为 ``None``。
    @param best_metal_sha256 ``best.metal`` 的摘要。
    @param confirm_status 复测报告的状态。
    @param winner_metal_sha256 胜者候选内核的摘要。
    @return ``{"speedup": ..., "publication": ...}``。
    """
    return _published_verdict(
        result, best_metal_sha256=best_metal_sha256, confirm_status=confirm_status,
        winner_metal_sha256=winner_metal_sha256,
    )[0]


def _published_verdict(result: dict | None, *, best_metal_sha256: str | None = None, confirm_status: str | None = None, winner_metal_sha256: str | None = None) -> tuple[dict, str]:
    """@brief ``published_conclusion`` 的内部实现：返回（结论, 原因码）。

    @param result 结果字典；可以为 ``None``。
    @param best_metal_sha256 ``best.metal`` 的摘要。
    @param confirm_status 复测报告的状态。
    @param winner_metal_sha256 胜者候选内核的摘要。
    @return ``({"speedup": ..., "publication": ...}, reason)``。
    """
    if not result:
        return {"speedup": None, "publication": "in_progress"}, "no_result_json"
    if "confirmation_source_sha256" not in result:
        return {"speedup": None, "publication": "旧版未绑定源码"}, "not_confirmed"
    recorded = result.get("confirmation_source_sha256")
    winner = result.get("confirm_winner_label")
    if result.get("confirmed_status") != "ok" or confirm_status != "ok":
        return {"speedup": None, "publication": "未确认"}, "not_confirmed"
    if not isinstance(recorded, str) or recorded == "" or recorded != best_metal_sha256 or recorded != winner_metal_sha256:
        return {"speedup": None, "publication": "未确认"}, "sha_mismatch"
    if not isinstance(winner, str) or winner == "":
        return {"speedup": None, "publication": "未确认"}, "label_mismatch"
    value = result.get("confirmed_speedup")
    if not _finite_positive(value):
        return {"speedup": None, "publication": "未确认"}, "not_confirmed"
    values = result.get("confirmed_values")
    if not isinstance(values, list) or not values:
        return {"speedup": None, "publication": "未确认"}, "values_mismatch"
    numbers = [float(item) for item in values if isinstance(item, (int, float)) and not isinstance(item, bool)]
    if len(numbers) != len(values) or not all(_finite_positive(item) for item in numbers):
        return {"speedup": None, "publication": "未确认"}, "values_mismatch"
    # 公布的数字必须是各批的中位数本身，而不是与它们无关的另一个字段。
    median = statistics.median(numbers)
    if abs(float(value) - median) > 1e-9 * max(1.0, abs(median)):
        return {"speedup": None, "publication": "未确认"}, "speedup_mismatch"
    return {"speedup": median, "publication": "confirmed"}, "confirmed"


def publication_role(
    conclusion: dict | None,
    domain: dict | None,
    *,
    source_sha256: str | None,
    task_fingerprint: str | None,
    recorded_fingerprint: str | None,
    min_improvement: float | None = None,
) -> str | None:
    """@brief 为一个运行定角色：过了噪声门且域扫通过时是 ``primary``，否则是 ``fallback``。

    ``primary`` when a confirmed speedup clears the noise gate and a passing domain sweep is bound to that same source and task.

    ``None`` while the run is still in progress. Anything else, including a correct kernel that is slower than the baselines, is ``fallback``.
    The value is derived. It is not written into ``result.json``.

    @param conclusion ``conclusion_of_run`` 给出的结论。
    @param domain 域扫报告。
    @param source_sha256 当前内核源码摘要。
    @param task_fingerprint 当前任务指纹。
    @param recorded_fingerprint 域扫报告里记录的指纹。
    @param min_improvement 噪声门；为 ``None`` 时用配置里的值。
    @return ``primary`` / ``fallback``；运行还进行中时返回 ``None``。
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
    speed_ok = _finite_positive(speed) and float(speed) >= min_improvement
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
    """@brief 取汇总表可以印的那一个加速比。

    The number a summary table may print. ``None`` when the publication gate did not accept one.

    A row that never went through the gate has no ``published_speedup`` key and prints nothing, rather than
    ``confirmed_speedup`` or ``best_speedup``.

    @param row 一行汇总结果。
    @return ``published_speedup``；没有这个键时返回 ``None``，不回退到搜索期数字。
    """
    if "published_speedup" not in row:
        return None
    return row.get("published_speedup")


def summarize(res: dict, confirm: dict | None, naive: dict | None = None, *, run_dir: Path | None = None) -> dict:
    """@brief 把一次运行压成套件表里的一行。

    One row of the suite table. ``published_speedup`` is set only for a bound, quiet confirmation.

    @param res 运行的 ``result.json`` 内容。
    @param confirm 换种子复测报告；可以为 ``None``。
    @param naive 手写朴素核的参照结果；为 ``None`` 时不填相关列。
    @param run_dir 运行目录；给出时用它跑完整的发布判定。
    @return 含任务、轮数、花费、各种加速比与相对朴素核增益的一行。
    """
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
    """@brief 统计套件里达到各档发布加速比的格数与占比。

    fast_p counts rows whose published speedup is at least p. Search-only speedups are not counted.

    @param rows 套件的结果行。
    @param thresholds 要统计的加速比门槛。
    @return 行数、总花费、发布条数、中位数与各门槛下的命中数。
    """
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
    """@brief 用同一套评测协议量手写朴素核，作为看得出基线太弱的参照。

    Evaluate the task's hand-written *naive* kernel under the same protocol. Its speedup over the baselines is the reference
    point that makes weak baselines visible: if a naive kernel already beats MLX/MPS (as for unfused ops such as Lion), part of
    any AI-lowering speedup is baseline weakness, not compiler skill.

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模档位。
    @param cache_dir 套件目录，参照结果缓存在这里。
    @param evaluate_fn 评测函数；为 ``None`` 时用 ``tcenv.evaluate``。
    @return 含状态、加速比与两侧耗时的字典。
    """
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
