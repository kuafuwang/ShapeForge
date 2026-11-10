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
from taic.policy import confirmation_publication

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
    evaluator: Any = None,
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
    @param evaluator 统一的评测执行器；为 ``None`` 时由 ``evaluate_fn`` 包成 ``LocalEvaluator``。
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
        return _resume_one(task, dtype, size, Path(run_dir), cfg, search, raw, model, client, evaluate_fn, agent_runner, log, evaluator=evaluator)
    committees = load_committees(raw, active=cfg, model_override=model)
    orch = Orchestrator(task, dtype, size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn, evaluator=evaluator, log=log, committees=committees, agent_runner=agent_runner, target_id=target_id or "apple-m4-max", seed=seed)
    return asyncio.run(orch.run())


def _looks_resumable(run_dir: Path) -> bool:
    """@brief 判断目录里是否已经有续跑所需的三份文件。

    A non-empty cell with the three files a flame-chase snapshot needs; a stray file dir is still refused.

    @param run_dir 搜索目录。
    @return 三份文件都在时为 ``True``。
    """
    return (run_dir / "run_config.json").is_file() and (run_dir / "state.json").is_file() and (run_dir / "transcript.jsonl").is_file()


def _resume_one(task, dtype, size, run_dir, cfg, search, raw, model, client, evaluate_fn, agent_runner, log, *, evaluator=None) -> RunResult:
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
    @param evaluator 统一的评测执行器；为 ``None`` 时由 ``evaluate_fn`` 包成 ``LocalEvaluator``。
    @return 续跑后的结果对象。
    @raises taic.orchestrator.RunStartError 快照读不出来、远端上下文损坏，或 outbox 打不开时。
    """
    from tasks import get_task

    from taic import resume as rs
    from taic.artifacts import NeedsInspection
    from taic.orchestrator import RunStartError

    task_obj = get_task(task, dtype, size)
    try:
        snap = rs.load_snapshot(run_dir, search, task_obj.contract.kernel_name)
    except (rs.ResumeError, NeedsInspection) as exc:
        # A snapshot that is damaged (not merely absent) is a refused start, not a traceback.
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
    try:
        orch = Orchestrator(
            task, dtype, size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn,
            evaluator=evaluator, log=log, committees=committees, agent_runner=agent_runner, resuming=True,
        )
        # Same boundary as ``taic.cli.prepare_resume``: a damaged remote evaluation context or a
        # damaged outbox refuses the resume instead of escaping as a traceback.
        orch.restore(snap, dry_run=False)
    except (NeedsInspection, RunStartError) as exc:
        raise RunStartError(f"{run_dir} is not resumable: {exc}") from exc
    return asyncio.run(orch.run())


# 各脚本的 --resume 只在 result.json 不存在时才开新的一次搜索。文件在，但损坏或版本更新，要人来看，不能当成还没跑过。
def _synthetic_native_build(candidate_sha256: str, identity: dict) -> dict:
    """@brief 为合成确认报告组装一份**形状合法**的 native/``full`` build 文档。

    A confirmation report is a schema-4 ``EvalReport`` document, so its ``build`` record must pass
    the same closed-shape validation as a real one.  The record is projected through the production
    projector from a *portable* artifact set (the described SOURCE manifest only): it is structurally
    a real build and makes no claim that anything was compiled or loaded here.

    @param candidate_sha256 胜者源码摘要；报告的 ``build.candidate_sha256`` 必须就是它。
    @param identity ``{"opspec_digest", "lowering_profile_digest"}``。
    @return ``build`` 文档（已通过 ``validate_shape``）。
    """
    from tcenv.build_projection import project_full
    from tcenv.remote.artifact_builder import (
        BuildArtifact, BuildEvidence, BuildOutcome, BuildProfile, BuildProvenance,
    )
    from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry

    profile = BuildProfile(
        build_profile="synthetic-confirmation-fixture",
        builder_id="synthetic-confirmation-fixture", builder_version="0",
        build_host="control_host", toolchain_id="none", toolchain_version="0",
        sdk_version="fixture",
        compile_profile={"fixture": True, "compiled": False}, build_kind="native",
    )
    entry = BundleEntry(
        path="kernel.tx", role="SOURCE", media_type="text/plain; charset=utf-8",
        size_bytes=0, sha256=candidate_sha256, described=True,
    )
    bundle = ArtifactBundle(entries=(entry,))
    artifacts = tuple(BuildArtifact.from_entry(item) for item in bundle.entries)
    provenance = BuildProvenance(
        profile=profile, source_sha256=candidate_sha256,
        artifact_set_digest=bundle.artifact_set_digest,
    )
    evidence = BuildEvidence(
        mode="prebuilt", status="ok", provenance=provenance, artifacts=artifacts,
        placement="control_host",
    )
    outcome = BuildOutcome(bundle=bundle, provenance=provenance, evidence=evidence, payloads={})
    return project_full(outcome, build_kind="native", identity=identity)


def synthetic_confirmation_documents(
    *, task: str, dtype: str, size: str, target_id: str,
    speedup: float, label: str, kernel_sha256: str,
    candidate_us: float | None = None,
    values: list[float] | None = None,
    protocol_version: str | None = None,
) -> tuple[dict, dict]:
    """@brief 造一对能通过**唯一发布谓词**的 ``(result.json, confirm_report.json)`` 合成文档。

    This is a fixture builder for synthetic runs (the scale probe and the observe/read-path tests),
    **not evidence**: it fabricates the per-batch identity a real confirmation would have recorded,
    because a synthetic run never confirmed anything.  Its one hard rule is that the documents pass
    :func:`taic.policy.confirmation_publication` — the same predicate a real run goes through:

    * if the gate tightens (a new required field, a new identity state), this builder raises and the
      fixtures stop working instead of silently keeping a second publication path;
    * the read side (``conclusion_of_run`` / ``observe``) sees exactly the documents a real run
      writes, so the confirmed path it measures is the production one.

    A ``schema_version`` other than 4 is an old-format report and is refused by every reader, so the
    fixture is schema 4 by construction.

    The frozen expected is always **complete**: the current task/registry identity when the registry
    covers the configuration, otherwise a deterministic fixture identity (some observe-analysis
    fixtures use dtypes the registry does not cover to build distinct comparability keys).  It is
    never left with ``None`` identity fields — an incomplete expected is ``*_identity_unknown`` and
    must not publish, and this builder must not be the place that hides it.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param target_id 目标设备 id。
    @param speedup 确认加速比（单批时同时是各批中位数）。
    @param label 胜者候选标签。
    @param kernel_sha256 胜者内核的 sha256（``best.metal`` 与该候选文件必须同哈希）。
    @param candidate_us 确认批的候选微秒数；``None`` 表示不记。
    @param values 逐批加速比；缺省为 ``[speedup]``。
    @param protocol_version 冻结 expected 与报告里的测量协议；缺省为当前 ``PROTOCOL_VERSION``。
    @return ``(result, confirm_report)`` 两份文档。
    @raises ValueError 生成的文档仍被发布谓词拒绝时（谓词收紧后的回归信号）。
    """
    from tcenv.evaluate import confirmation_expected_document
    from tcenv.evaluation import EvaluationError
    from tcenv.opspec.model import OpSpecError
    from tcenv.schema import REPORT_SCHEMA_VERSION, PROTOCOL_VERSION

    protocol = str(protocol_version or PROTOCOL_VERSION)
    batches_values = list(values) if values else [float(speedup)]
    median = float(statistics.median(batches_values))
    identity = {
        "kind": "match",
        "differences": [],
        "shape": "full",
        "build_kind": "native",
        "measurement_kind": "measured",
        "cache_hit": False,
        "timing_uncertain": False,
    }
    # One batch record per confirmed value: the gate pairs the records with the values one to one
    # (§7.2), so a two-batch fixture must carry two records, not one shared record.
    batch = {
        "label": label,
        "shape": "full",
        "build_kind": "native",
        "measurement_kind": "measured",
        "cache_hit": False,
        "timing_uncertain": False,
        "identity": dict(identity),
    }
    batches = [dict(batch) for _value in batches_values]
    entry = {
        "label": label,
        "search_speedup": median,
        "failed_batches": 0,
        "status": "ok",
        "confirmed_speedup": median,
        "confirmed_values": list(batches_values),
        "identity": dict(identity),
    }
    # The frozen expected must be **complete and consistent** with the report it binds (§7.2).
    # The CPU-side expected builder gives the current task/registry identity; a synthetic run for a
    # configuration the registry does not cover (observe-analysis fixtures use dtypes such as
    # ``fp32``/``int8`` to build distinct comparability keys) has no real identity to state, so it
    # gets a deterministic fixture identity instead.  Either way the document is complete: an
    # expected with ``None`` fields is ``*_identity_unknown`` and can never publish, and this
    # builder must not be the place that hides that.
    evaluation_identity = None
    try:
        expected = confirmation_expected_document(task, dtype, size, target_id, "", probe=False)
        evaluation_identity = expected["evaluation"]
    except (EvaluationError, OpSpecError, KeyError, ValueError):
        # The registry cannot state an identity for this configuration (out-of-domain dtype/size,
        # unknown task name in a budget fixture): a fixture identity is the honest stand-in.
        expected = None
    if expected is None:
        import hashlib

        def _fixture_digest(what: str) -> str:
            seed = f"synthetic:{what}:{task}/{dtype}/{size}/{target_id}"
            return hashlib.sha256(seed.encode("utf-8")).hexdigest()

        # Fixture-only identity: complete and consistent, but not a deployable measurement claim
        # (this module's synthetic documents are explicitly *not evidence*).
        expected = {
            "build": {
                "candidate_sha256": kernel_sha256,
                "opspec_digest": _fixture_digest("opspec"),
                "lowering_profile_digest": _fixture_digest("lowering"),
                "provenance": None,
            },
            "evaluation": {
                "evaluation_fingerprint": _fixture_digest("evaluation"),
                "baseline_implementation_digest": _fixture_digest("baseline_implementation"),
                "baseline_environment_digest": _fixture_digest("baseline_environment"),
                "backend": "metal", "backend_version": "1",
                # Required identity in the projection too: the predicate checks both the top-level
                # pair and ``evaluation.protocol_version``/``target_id`` (§7.2).
                "protocol_version": protocol, "target_id": target_id,
                "device_name": None, "gpu_family": None,
            },
        }
        evaluation_identity = expected["evaluation"]
    else:
        expected["build"]["candidate_sha256"] = kernel_sha256
    expected["protocol_version"] = protocol
    expected["target_id"] = target_id
    if any(evaluation_identity.get(field) is None for field in (
        "evaluation_fingerprint", "baseline_implementation_digest",
        "baseline_environment_digest", "backend", "backend_version",
    )):
        raise ValueError(
            "the synthetic confirmation cannot state a current evaluation identity on this host; "
            "an incomplete expected is unknown and must not be fabricated"
        )
    result = {
        "task": task, "dtype": dtype, "size": size, "target_id": target_id,
        "confirmed_speedup": median, "confirmed_status": "ok", "confirm_winner_label": label,
        "confirmation_source_sha256": kernel_sha256, "confirmed_values": list(batches_values),
        "confirmed_worst": min(batches_values), "confirmed_best": max(batches_values),
        "candidate_us": candidate_us, "confirmations": [entry],
    }
    # The confirm report on disk is a whole schema-4 ``EvalReport`` document (the orchestrator
    # writes ``rep.to_json()``), so the fixture is one too: a reduced dictionary would be refused by
    # the publication read boundary, which now runs the full schema closure.
    from tcenv.evaluate import EvalReport

    identity_document = {
        "opspec_digest": expected["build"].get("opspec_digest"),
        "lowering_profile_digest": expected["build"].get("lowering_profile_digest"),
    }
    build_document = _synthetic_native_build(kernel_sha256, identity_document)
    # A publishable frozen expected states the same recomputable BuildProvenance the report carries
    # (a real run gets it from the live executor probe; a CPU fixture has no probe, so it repeats the
    # fixture profile).  ``artifact_set_digest`` is outcome-only and stays out of the expected.
    expected["build"]["provenance"] = {
        key: value for key, value in build_document["provenance"].items() if key != "artifact_set_digest"
    }
    report_object = EvalReport(
        task=task, dtype=dtype, size=size, mode="full", status="ok", stage="done", seed=0,
        build=build_document,
        speedup=median, candidate_us=candidate_us,
        evaluation_fingerprint=evaluation_identity["evaluation_fingerprint"],
        baseline_implementation_digest=evaluation_identity["baseline_implementation_digest"],
        baseline_environment_digest=evaluation_identity["baseline_environment_digest"],
        backend=evaluation_identity["backend"],
        backend_version=evaluation_identity["backend_version"],
        protocol_version=protocol, target_id=target_id,
        benchmark={
            "noisy": False,
            "candidate": {"cv": 0.01, "reps": 15},
            "baselines": {"mlx": {"cv": 0.01, "reps": 15}},
            "best_baseline": "mlx",
        },
    )
    report = report_object.to_json()
    report["benchmark"]["confirmation"] = {
        "kernel_sha256": kernel_sha256, "winner_label": label,
        "confirmed_values": list(batches_values), "speedup": median,
        "candidate_us": candidate_us,
        "expected": expected, "batches": batches,
    }
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    decision = confirmation_publication(
        result=result, confirm_report=report, batches=batches, expected=expected,
        bindings={
            "confirmation_source_sha256": kernel_sha256,
            "confirm_report_status": "ok",
            "best_source_sha256": kernel_sha256,
            "winner_source_sha256": kernel_sha256,
        },
    )
    if not decision["allowed"]:
        raise ValueError(
            "the synthetic confirmation no longer passes the publication gate: "
            f"{decision['status']} ({decision['reason']})"
        )
    return result, report


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
    from taic.artifacts import read_versioned_exact
    from tcenv.schema import REPORT_SCHEMA_VERSION

    # Exactly the current report schema.  A schema-3 confirm report has no per-batch identity and
    # no frozen expected, so it is undecidable evidence — not a slightly older current report.
    return read_versioned_exact(Path(run_dir) / "confirm_report.json", version=REPORT_SCHEMA_VERSION)


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


def _read_bytes(path: Path | None) -> bytes | None:
    """@brief Read a file's raw bytes (never ``read_text``: that normalizes CRLF).

    @param path file path, or None when the caller could not resolve one.
    @return the bytes, or None when there is no path or the file cannot be read.
    """
    if path is None:
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _sha256_bytes(raw: bytes | None) -> str | None:
    """@brief sha256 of raw bytes, or None for a missing file.

    @param raw file bytes.
    @return hex digest or None.
    """
    return None if raw is None else hashlib.sha256(raw).hexdigest()


def _manifest_sha256(raw: bytes | None) -> str | None:
    """@brief The candidate identity of a delivered file.

    For a segmented ``msl-bundle`` submission the identity is the candidate-entry manifest digest,
    not ``sha256(envelope text)``; for a plain single file the two coincide.  Any decode failure is
    a ``sha_mismatch`` at the caller, never an exception.

    @param raw delivered bytes.
    @return hex digest or None when the bytes are not a decodable candidate.
    """
    from tcenv.track.envelope import EnvelopeError, decode_envelope, is_candidate_envelope

    if raw is None:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not is_candidate_envelope(text):
        return hashlib.sha256(raw).hexdigest()
    try:
        return decode_envelope(text).candidate_sha256
    except EnvelopeError:
        return None


def _best_source_path(run_dir: Path) -> Path:
    """Return the recorded candidate envelope for either MSL or a DSL Track."""
    taic = run_dir / "best.taic"
    return taic if taic.is_file() else run_dir / "best.metal"


def _winner_kernel_path(run_dir: Path, winner_label: str | None) -> Path | None:
    """@brief The winner candidate's ``kernel.metal`` path, if the label resolves uniquely.

    @param run_dir run directory.
    @param winner_label recorded winner label.
    @return path or None.
    """
    if not isinstance(winner_label, str) or not winner_label:
        return None
    root = Path(run_dir) / "candidates"
    if not root.is_dir():
        return None
    found = [
        p for p in root.iterdir()
        if p.is_dir() and len(p.name) > 4 and p.name[3] == "_" and p.name[4:] == winner_label
    ]
    if len(found) != 1:
        return None
    taic = found[0] / "kernel.taic"
    return taic if taic.is_file() else found[0] / "kernel.metal"


def _read_confirm_report(run_dir: Path) -> dict | None:
    """@brief 用拒绝重复键的 loader 读确认报告，并要求 schema 恰为**整数** 4。

    A confirm report of any other schema is *old format*: the run is not publishable through the
    current gate, and it is not migrated or double-read.  The version test is
    :func:`taic.artifacts.document_state` — ``4.0 == 4`` in Python, so a plain ``!=`` comparison
    would accept a float version and publish through a document no reader would call current.

    @param run_dir 搜索目录。
    @return 报告字典；缺失、损坏、重复键或版本不是非 bool 的整数 4 时返回 ``None``。
    """
    from taic.artifacts import NeedsInspection, read_versioned_exact
    from tcenv.evaluate import EvalReport
    from tcenv.schema import REPORT_SCHEMA_VERSION, SchemaError

    try:
        # Same reader as ``read_confirm``: strict JSON (duplicate keys refused) and a type-strict
        # integer version.  Undecidable is "not publishable" here, never a guess or a migration.
        document = read_versioned_exact(Path(run_dir) / "confirm_report.json", version=REPORT_SCHEMA_VERSION)
    except NeedsInspection:
        return None
    try:
        # The version check alone accepts any body.  A publication read must use the same schema-4
        # closure every other reader uses (closed key set, status table, build shape, digest/type
        # rules): otherwise a document that ``EvalReport.from_json`` refuses — an integer
        # ``evaluation_fingerprint``, a missing field, a float version — could still publish through
        # the readers that only compare a few keys.
        EvalReport.from_json(document)
    except (SchemaError, TypeError, ValueError):
        # Damaged/unreadable evidence is ``unknown``, exactly like a schema-3 report: nothing here
        # repairs, migrates or partially trusts the document.
        return None
    return document


def _confirmation_files(run_dir: Path, winner_label: str | None) -> tuple[str | None, str | None, str | None]:
    """@brief 取出确认门需要比对的三份证据。

    (best.metal sha256, winner candidate kernel.metal sha256, confirm_report status).

    @param run_dir 搜索目录。
    @param winner_label 结果里记录的胜者标签。
    @return ``(best.metal 摘要, 胜者候选摘要, 复测报告状态)``，缺的那项为 ``None``。
    """
    best = _sha256_file(_best_source_path(run_dir))
    document = _read_confirm_report(run_dir)
    report = document.get("status") if isinstance(document, dict) else None
    winner = None
    if isinstance(winner_label, str) and winner_label:
        root = run_dir / "candidates"
        if root.is_dir():
            found = [p for p in root.iterdir() if p.is_dir() and len(p.name) > 4 and p.name[3] == "_" and p.name[4:] == winner_label]
            if len(found) == 1:
                winner = _sha256_file(_winner_kernel_path(run_dir, winner_label))
    return best, winner, report


def conclusion_of_run(run_dir: Path, result: dict | None) -> dict:
    """@brief 给一个运行目录出发布结论：只有确认门绑住重测内核时才给数字。

    The publication verdict for one run directory. A bare ``result.json`` is not enough to call a number confirmed.

    @param run_dir 搜索目录。
    @param result 该目录的结果字典；可以为 None。
    @return ``{"speedup": ..., "publication": ...}``；缓存命中、计时不稳或各批不一致时标为 ``不确定``。
    """
    return conclusion_verdict(run_dir, result)[0]


def publication_reason(run_dir: Path, result: dict | None) -> dict:
    """@brief 发布判定 + 原因码（只给观测用；``conclusion_of_run`` 仍只返回两个键）。

    原因码按代码顺序给出“第一个命中的分支”，可直接映射到行动（重新确认、修协议、等安静测量等），
    不发明组合码。旧格式（schema ≠ 4）的确认报告根本读不成当前文档，因此只报 ``not_confirmed``；
    缺 ``confirmation`` 块的 schema-4 报告才报 ``missing_confirmation_block``。

    @param run_dir 搜索目录。
    @param result 该目录的结果字典；可以为 None。
    @return ``{"publication": ..., "reason": ...}``。
    """
    got, reason = conclusion_verdict(run_dir, result)
    return {"publication": got["publication"], "reason": reason}


def _track_binding(run_dir: Path, result: dict, best_raw: bytes | None, winner_raw: bytes | None) -> tuple[str | None, str | None]:
    """@brief Cross-check ``result["track"]`` against the registry and the delivered shape.

    The table is decided top to bottom, and every ``track_mismatch`` returns instead of raising:
    an unknown id / version mismatch is ``UnsupportedContractCapability`` from the registry (a
    ``TaskContractError``), not a ``KeyError``.

    @param run_dir run directory (for diagnostics only).
    @param result result document.
    @param best_raw ``best.metal`` bytes, or None.
    @param winner_raw winner ``kernel.metal`` bytes, or None.
    @return ``(reason, delivery)``; ``reason`` is None when the record is usable.
    """
    from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
    from tcenv.track.envelope import EnvelopeError, decode_envelope, is_candidate_envelope
    from tcenv.track.registry import load_track_registry

    def envelope_track(raw: bytes | None) -> tuple[bool, str | None, bool]:
        """Return ``(is_envelope_shaped, decoded_track_id, decodable)``.

        ``decodable`` is False for envelope-shaped bytes that cannot be decoded (non-UTF-8, a non-ASCII
        manifest line, an unknown role).  Those are *identity* failures, not delivery disagreements: the
        read-disk closed set maps them to ``sha_mismatch``.
        """
        if raw is None:
            return False, None, True
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return True, None, False
        if not is_candidate_envelope(text):
            return False, None, True
        try:
            return True, decode_envelope(text).track_id, True
        except (EnvelopeError, TaskContractError):
            return True, None, False

    best_is_envelope, best_track, best_decodable = envelope_track(best_raw)
    winner_is_envelope, winner_track, winner_decodable = envelope_track(winner_raw)
    block = result.get("track", "__missing__")
    if block == "__missing__":
        # Legacy: a v2 record from before the track block could only have produced single files.
        # An envelope best without a track block cannot be proven to belong to a bundle run.
        return ("track_mismatch", None) if (best_is_envelope or winner_is_envelope) else (None, "single_source")
    if not isinstance(block, dict):
        return "track_mismatch", None
    if not {"track_id", "track_version", "delivery"} <= set(block):
        return "track_mismatch", None
    track_id, version, delivery = block["track_id"], block["track_version"], block["delivery"]
    if not isinstance(track_id, str) or not isinstance(delivery, str):
        return "track_mismatch", None
    try:
        spec = load_track_registry().get(track_id, version=int(version))
    except (UnsupportedContractCapability, TypeError, ValueError):
        return "track_mismatch", None
    if delivery != spec.delivery:
        return "track_mismatch", None
    if best_is_envelope and not best_decodable:
        return "sha_mismatch", None
    if winner_is_envelope and not winner_decodable:
        return "sha_mismatch", None
    if spec.delivery in ("candidate_bundle", "dsl_source_pipeline"):
        if not best_is_envelope or not winner_is_envelope:
            return "track_mismatch", None
        if best_track != track_id or winner_track != track_id:
            return "track_mismatch", None
    else:
        if best_is_envelope or winner_is_envelope:
            return "track_mismatch", None
    return None, spec.delivery


def _v2_conclusion_verdict(run_dir: Path, result: dict) -> tuple[dict, str]:
    """@brief v2（portfolio）运行的发布读盘：只认 subject sidecar 的独立 verdict。

    与 v1 一样绑定三份证据：``best.metal``、胜者候选文件、以及 result 记录的
    ``confirmation_source_sha256`` 必须等于 sidecar 里的候选摘要；sidecar 本身由
    ``verify_subject_document`` 重算 identity/verdict/digest，篡改或旧格式一律不发布。

    @param run_dir 运行目录。
    @param result 结果字典。
    @return ``({"speedup","publication"}, reason)``。
    """
    from taic.portfolio_confirmation import verify_subject_document

    run_dir = Path(run_dir)
    best, winner, _status = _confirmation_files(run_dir, result.get("confirm_winner_label"))
    # Identity is read from the *bytes*, never from ``read_text`` (which normalizes CRLF): a legacy
    # single-file run must keep its transport hash and therefore its verdict byte for byte.
    best_raw = _read_bytes(_best_source_path(run_dir))
    winner_raw = _read_bytes(_winner_kernel_path(run_dir, result.get("confirm_winner_label")))
    track_reason, delivery = _track_binding(run_dir, result, best_raw, winner_raw)
    if track_reason is not None:
        return {"speedup": None, "publication": "未确认"}, track_reason
    if delivery in ("candidate_bundle", "dsl_source_pipeline"):
        manifest_best = _manifest_sha256(best_raw)
        if manifest_best is None:
            return {"speedup": None, "publication": "未确认"}, "sha_mismatch"
        # ``confirmation_source_sha256`` stays the transport hash of the submission text; the sidecar
        # carries the candidate *manifest* digest.  They are different by construction for a bundle.
        transport_best = _sha256_bytes(best_raw)
    else:
        manifest_best = _sha256_bytes(best_raw) if best_raw is not None else None
        transport_best = manifest_best
    folder = run_dir / "confirmations"
    documents: list[dict] = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            try:
                documents.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                return {"speedup": None, "publication": "未确认"}, "not_confirmed"
    if not documents:
        return {"speedup": None, "publication": "未确认"}, "not_confirmed"
    reason = "not_confirmed"
    for document in documents:
        try:
            verified = verify_subject_document(document)
        except Exception:  # noqa: BLE001 - 损坏/被改写的证据不是可发布证据
            return {"speedup": None, "publication": "未确认"}, "not_confirmed"
        if verified.get("status") != "confirmed":
            reason = str(verified.get("reason") or "not_confirmed")
            continue
        candidate_sha256 = verified.get("candidate_sha256")
        if result.get("confirmed_status") != "ok":
            reason = "not_confirmed"
            continue
        if manifest_best is None or candidate_sha256 != manifest_best:
            reason = "sha_mismatch"
            continue
        if result.get("confirmation_source_sha256") != transport_best:
            reason = "sha_mismatch"
            continue
        if best is None or winner is None or winner != transport_best:
            reason = "sha_mismatch"
            continue
        values = result.get("confirmed_values") or []
        numbers = [float(item) for item in values if isinstance(item, (int, float)) and not isinstance(item, bool)]
        speedup = statistics.median(numbers) if numbers else None
        return {"speedup": speedup, "publication": "confirmed"}, "confirmed"
    return {"speedup": None, "publication": "未确认"}, reason


def conclusion_verdict(run_dir: Path, result: dict | None) -> tuple[dict, str]:
    """@brief 读盘绑定组 + 唯一发布谓词，返回 ``(结论, 原因码)``。

    The bindings are derived here (``best.metal``, the winner candidate file, the confirm report's
    status and the result's recorded hash) and the decision is made by
    :func:`taic.policy.confirmation_publication` — the same function the live confirmation calls.
    The *frozen* expected and the per-batch identity records come from the confirm report; nothing
    is re-probed on today's host.

    @param run_dir 搜索目录。
    @param result 结果字典；可以为 ``None``。
    @return ``({"speedup": ..., "publication": ...}, reason)``。
    """
    if not result:
        return published_conclusion(None), "no_result_json"
    run_dir = Path(run_dir)
    if result.get("task_mode") == "v2":
        return _v2_conclusion_verdict(run_dir, result)
    best, winner, status = _confirmation_files(run_dir, result.get("confirm_winner_label"))
    document = _read_confirm_report(run_dir)
    expected = None
    batches = None
    if isinstance(document, dict):
        benchmark = document.get("benchmark")
        confirmation = benchmark.get("confirmation") if isinstance(benchmark, dict) else None
        if isinstance(confirmation, dict):
            expected = confirmation.get("expected")
            batches = confirmation.get("batches")
    decision = confirmation_publication(
        result=result,
        confirm_report=document,
        batches=batches,
        expected=expected,
        bindings={
            "confirmation_source_sha256": result.get("confirmation_source_sha256"),
            "confirm_report_status": status,
            "best_source_sha256": best,
            "winner_source_sha256": winner,
        },
    )
    speedup = None
    if decision["allowed"]:
        values = result.get("confirmed_values") or []
        numbers = [float(item) for item in values if isinstance(item, (int, float)) and not isinstance(item, bool)]
        if numbers:
            speedup = statistics.median(numbers)
    return {"speedup": speedup, "publication": decision["status"]}, decision["reason"]


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
    """Deprecated compatibility shim: a result without its confirm report cannot be published.

    The single predicate needs the confirm report (frozen expected + per-batch records).  Callers
    that only hold ``result.json`` therefore get ``未确认`` instead of a bypass.
    """

    decision = confirmation_publication(
        result=result, confirm_report=None, batches=None, expected=None,
        bindings={
            "confirmation_source_sha256": result.get("confirmation_source_sha256") if isinstance(result, dict) else None,
            "confirm_report_status": confirm_status,
            "best_source_sha256": best_metal_sha256,
            "winner_source_sha256": winner_metal_sha256,
        },
    )
    speedup = None
    if decision["allowed"] and isinstance(result, dict):
        values = result.get("confirmed_values") or []
        numbers = [float(item) for item in values if isinstance(item, (int, float)) and not isinstance(item, bool)]
        if numbers:
            speedup = statistics.median(numbers)
    return {"speedup": speedup, "publication": decision["status"]}, decision["reason"]


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
