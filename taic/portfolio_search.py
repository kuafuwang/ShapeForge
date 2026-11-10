"""Drive one ordinary search per benchmark size, then one assemble session.

The state file is ``portfolio_search.json``.  This module does not import
``tcenv.backends`` at module level.  It does not write ``portfolio_run.json``.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from pathlib import Path
from typing import Any, Callable

from tcenv.portfolio_assembly import (
    PortfolioAssemblyError,
    benchmark_sizes,
    check_parent,
    decode_parent,
    domain_for_sizes,
)
from tcenv.track.envelope import EnvelopeError, decode_envelope

from .portfolio_prompt import render_assemble_task

SEARCH_SCHEMA_ID = "taic.portfolio-search.result.1"
_STATE_NAME = "portfolio_search.json"
_TRACKS = ("msl-single", "msl-bundle")
# 单文件 1、bundle 2、DSL 3。组装出来的多变体父包用 4，和前三种证据隔开。
ASSEMBLED_PROTOCOL_EPOCH = 4


class PortfolioSearchStart(Exception):
    """@brief 搜索还没开会话就拒绝启动。"""


def staged_problems(entries, staged: dict[str, dict[str, str]]) -> tuple[str, ...]:
    """@brief 交卷里的变体字节必须等于驱动摆进去的某一档。

    ``variants/<key>/`` 的相对路径和 sha256 要和 ``staged[key]`` 完全一样，``key`` 就是 size 名。
    可以少用某一档。多出来的变体拒绝。``fallback/`` 有文件时，去掉前缀后的集合必须和某一档完全一样。

    @param entries 父包 entries。
    @param staged size 到 ``{相对路径: sha256}``。
    @return 问题列表；空表示这两条规则通过。
    """
    problems: list[str] = []
    grouped: dict[str, dict[str, str]] = {}
    fallback: dict[str, str] = {}
    for entry in entries:
        path = str(entry.path)
        digest = hashlib.sha256(bytes(entry.data)).hexdigest()
        if path.startswith("variants/"):
            rest = path[len("variants/"):]
            key, _sep, relative = rest.partition("/")
            if not key or not relative:
                problems.append(f"{path}: variant path needs a size and a file")
                continue
            bucket = grouped.setdefault(key, {})
            if relative in bucket:
                problems.append(f"{path}: duplicated variant file")
            bucket[relative] = digest
        elif path.startswith("fallback/"):
            relative = path[len("fallback/"):]
            if not relative or relative.endswith("/"):
                problems.append(f"{path}: fallback path needs a file")
                continue
            if relative in fallback:
                problems.append(f"{path}: duplicated fallback file")
            fallback[relative] = digest
    for key, files in grouped.items():
        expected = staged.get(key)
        if expected is None:
            problems.append(f"{key}: 驱动没有摆过这个变体")
        elif files != expected:
            problems.append(f"{key}: 变体文件和摆进去的核不一致")
    if fallback and fallback not in staged.values():
        problems.append("fallback 不是任何一颗已摆核的原样拷贝")
    return tuple(problems)


def run_portfolio_search(
    task: str,
    dtype: str,
    *,
    sizes: tuple[str, ...] | list[str] | None = None,
    track: str = "msl-single",
    budget: float | None = None,
    assemble_budget: float | None = None,
    assemble_attempts: int = 2,
    max_total_usd: float | None = None,
    run_dir: Path | None = None,
    target_id: str = "apple-m4-max",
    seed: int | None = None,
    config: str | None = None,
    preset: str | None = None,
    model: str | None = None,
    dry_run: bool = False,
    shard_runner: Callable | None = None,
    session_runner: Callable | None = None,
    evaluate: Callable | None = None,
    confirm: Callable | None = None,
) -> int:
    """@brief 跑形状组合搜索。``--dry-run`` 只检查并打印计划。

    测试注入 ``shard_runner``、``session_runner``、``evaluate``。缺省的分片搜索会构造
    ``Orchestrator``，缺省的评测会跑真机；本计划的测试都不走那两条缺省。

    @param task 题名。
    @param dtype 数据类型。
    @param sizes 带 benchmark 的 size；缺省是题目里全部这样的档。
    @param track ``msl-single`` 或 ``msl-bundle``。
    @param budget 每个分片的美元上限；缺省取预设里的 ``budget_usd``。
    @param assemble_budget 组装会话的美元上限；缺省是 ``2 * relay_budget_usd``。
    @param assemble_attempts 组装最多几场。
    @param max_total_usd 名义总花费超过它就拒绝启动。
    @param run_dir 运行目录。dry-run 没给时用临时目录。
    @param target_id 目标档案。
    @param seed 运行种子；缺省是一个 64 位随机数。
    @param config 配置文件。
    @param preset 预设名。
    @param model 模型名，只传给真正的分片搜索。
    @param dry_run 只打印计划。
    @param shard_runner 跑一个分片。
    @param session_runner 跑一场组装会话。
    @param evaluate 评测父包；缺省是 timed 的 ``evaluate_parent``。
    @param confirm 每个点的确认函数，签名与 ``confirm_subject`` 相同且不传 track。``None`` 时用真的 ``confirm_subject``。
    @return 启动失败 5；跑完但没得到通过的组合 1；通过或 dry-run 为 0。
    """
    import tempfile

    from . import control

    try:
        opened = _open(task, dtype, target_id, sizes, track, budget, assemble_budget, max_total_usd, config, preset)
    except PortfolioSearchStart as exc:
        print(str(exc), file=_stderr())
        return 5
    if dry_run and run_dir is None:
        with tempfile.TemporaryDirectory(prefix="portfolio-dry-") as temporary:
            return _locked(Path(temporary), opened, seed, dry_run=True, runners=None)
    if run_dir is None:
        from .llm import REPO_ROOT

        stamp = secrets.token_hex(4)
        run_dir = REPO_ROOT / "runs" / target_id / f"portfolio_{task}_{dtype}_{stamp}"
    if assemble_attempts < 1:
        print("assemble_attempts must be at least 1", file=_stderr())
        return 5
    opened.attempts = int(assemble_attempts)
    runners = None if dry_run else _Runners(shard_runner, session_runner, evaluate, confirm, model, config, preset)
    try:
        control.acquire_run_lock(Path(run_dir), who="portfolio-run")
    except control.RunLockError as exc:
        print(str(exc))
        return 5
    try:
        return _locked(Path(run_dir), opened, seed, dry_run=dry_run, runners=runners)
    finally:
        control.release_run_lock(Path(run_dir))


class _Opened:
    """@brief 启动检查通过后的域、预算和定位器。"""

    def __init__(self, locator, target, domain, track, budget, assemble_budget, nominal, relay):
        self.locator = locator
        self.target = target
        self.domain = domain
        self.track = track
        self.budget = budget
        self.assemble_budget = assemble_budget
        self.nominal = nominal
        self.relay = relay
        self.attempts = 2


class _Runners:
    """@brief 三个可替换的回调。缺省实现只在真正用到时才导入后端。"""

    def __init__(self, shard_runner, session_runner, evaluate, confirm, model, config, preset):
        self.shard_runner = shard_runner
        self.session_runner = session_runner
        self.evaluate = evaluate
        self.confirm = confirm
        self.model = model
        self.config = config
        self.preset = preset


def _locked(run_dir: Path, opened: _Opened, seed: int | None, *, dry_run: bool, runners: _Runners | None) -> int:
    """@brief 锁已经拿到。dry-run 只打印，其余往下跑。

    @param run_dir 运行目录。
    @param opened 启动材料。
    @param seed 调用方给的种子。
    @param dry_run 是否只打印。
    @param runners 回调；dry-run 时为 None。
    @return 进程退出码。
    """
    sizes = opened.domain.sizes
    print(f"sizes: {','.join(sizes)}")
    print(f"track: {opened.track}")
    print(f"nominal_usd: {opened.nominal}")
    if dry_run:
        return 0
    assert runners is not None
    run_seed = secrets.randbits(64) if seed is None else int(seed)
    state = {
        "schema_id": SEARCH_SCHEMA_ID,
        "task": opened.locator.op_name,
        "dtype": opened.locator.dtype,
        "target_id": opened.target.id,
        "track": opened.track,
        "sizes": list(sizes),
        "run_seed": run_seed,
        "shards": {},
        "assemble": [],
        "evaluation_status": None,
        "performance": None,
        "confirmation": None,
        "spent_usd": 0.0,
        "stop_reason": None,
        "publication_allowed": False,
    }
    _write_state(run_dir, state)
    shard_spent = _run_shards(run_dir, opened, run_seed, runners, state)
    if state["stop_reason"]:
        state["spent_usd"] = shard_spent
        _write_state(run_dir, state)
        print(f"stop_reason: {state['stop_reason']}")
        return 1
    _assemble(run_dir, opened, runners, state, shard_spent)
    print(f"stop_reason: {state['stop_reason']}")
    return 0 if state["stop_reason"] is None and state["evaluation_status"] == "ok" else 1


def _open(task, dtype, target_id, sizes, track, budget, assemble_budget, max_total_usd, config, preset) -> _Opened:
    """@brief 做启动检查。不读凭据，不开会话。

    @param task 题名。
    @param dtype 数据类型。
    @param target_id 目标档案。
    @param sizes 显式 size 或 None。
    @param track 交付方式。
    @param budget 每片预算或 None。
    @param assemble_budget 组装预算或 None。
    @param max_total_usd 名义上限或 None。
    @param config 配置文件。
    @param preset 预设。
    @return 打开的材料。
    @raises PortfolioSearchStart 任何一条检查失败。
    """
    from taic.task_binding import TaskBindingError, bind_task
    from tcenv.contract_errors import TaskContractError
    from tcenv.opspec.model import OpSpecError, load_opspec
    from tcenv.target import TargetError, load_target
    from tcenv.taskcontract.loader import OPSPEC_FILE
    from tcenv.taskcontract.locator import discover_standard_tasks

    from .llm import LLMError, REPO_ROOT, load_raw_config

    if track not in _TRACKS:
        raise PortfolioSearchStart(f"track must be msl-single or msl-bundle, got {track!r}")
    try:
        raw = load_raw_config(config, preset)
        search = dict(raw.get("search") or {})
        chosen_budget = float(search.get("budget_usd", 15.0) if budget is None else budget)
        relay = float(search.get("relay_budget_usd", 2.0))
        chosen_assemble = float(2 * relay if assemble_budget is None else assemble_budget)
        base = task.partition("@")[0]
        found = [
            item for item in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT)
            if item.op_name == base
        ]
        if len(found) != 1:
            raise TaskBindingError(f"task {task!r}: expected one discovered v2 standard entry, got {len(found)}")
        discovered = found[0]
        declared = set(load_opspec(discovered.task_dir / OPSPEC_FILE).sizes)
        requested = None if sizes is None else tuple(str(size) for size in sizes)
        initial_size = str(discovered.size)
        if requested is not None:
            initial_size = next((size for size in requested if size in declared), initial_size)
        binding = bind_task(task, dtype, initial_size, target_id=target_id)
        if not binding.is_v2:
            raise PortfolioSearchStart(f"task {task!r} is not a v2 standard entry")
        contract = binding.locator.load()
        chosen = tuple(benchmark_sizes(contract.spec, dtype) if requested is None else requested)
        if len(chosen) < 2:
            raise PortfolioSearchStart("a portfolio needs at least two sizes with benchmark usage")
        allowed = set(benchmark_sizes(contract.spec, dtype))
        missing = [size for size in chosen if size not in allowed]
        if missing:
            raise PortfolioSearchStart(f"{missing} do not have benchmark usage")
        if contract.size not in chosen:
            binding = bind_task(task, dtype, chosen[0], target_id=target_id)
            if not binding.is_v2:
                raise PortfolioSearchStart(f"task {task!r} is not a v2 standard entry")
            contract = binding.locator.load()
        target = load_target(target_id)
        domain = domain_for_sizes(contract, target, chosen)
    except (TaskBindingError, OpSpecError, TargetError, TaskContractError, LLMError) as exc:
        raise PortfolioSearchStart(str(exc)) from exc
    if track == "msl-bundle" and not domain.authority.workspace_declared:
        raise PortfolioSearchStart(
            f"{task} has no declared workspace tensor; msl-bundle variants cannot be projected"
        )
    nominal = len(chosen) * chosen_budget + chosen_assemble
    if max_total_usd is not None and nominal > float(max_total_usd):
        raise PortfolioSearchStart(
            f"nominal cost {nominal} exceeds --max-total-usd {max_total_usd}"
        )
    return _Opened(binding.locator, target, domain, track, chosen_budget, chosen_assemble, nominal, relay)


def _run_shards(run_dir: Path, opened: _Opened, run_seed: int, runners: _Runners, state: dict) -> float:
    """@brief 每个 size 一次普通搜索。已有 ``result.json`` 的目录跳过。

    @param run_dir 运行目录。
    @param opened 启动材料。
    @param run_seed 运行种子，传给分片搜索。
    @param runners 回调。
    @param state 状态文档，就地更新。
    @return 各分片 ``spent_usd`` 之和。
    """
    spent = 0.0
    runner = runners.shard_runner or _default_shard_runner(opened, runners, run_seed)
    for size in opened.domain.sizes:
        shard_dir = run_dir / "shards" / size
        record = {
            "directory": str(shard_dir),
            "stop_reason": None,
            "best_label": None,
            "best_speedup": None,
            "confirmed_speedup": None,
            "spent_usd": 0.0,
        }
        state["shards"][size] = record
        result_path = shard_dir / "result.json"
        if shard_dir.is_dir() and any(shard_dir.iterdir()) and not result_path.is_file():
            record["stop_reason"] = f"shard_incomplete:{size}"
            state["stop_reason"] = record["stop_reason"]
            _write_state(run_dir, state)
            return spent
        if not result_path.is_file():
            shard_dir.mkdir(parents=True, exist_ok=True)
            runner(size, shard_dir)
        if not result_path.is_file():
            record["stop_reason"] = f"shard_failed:{size}"
            state["stop_reason"] = record["stop_reason"]
            _write_state(run_dir, state)
            return spent
        document = _read_json(result_path)
        record["best_label"] = document.get("best_label")
        record["best_speedup"] = document.get("best_speedup")
        record["confirmed_speedup"] = document.get("confirmed_speedup")
        record["spent_usd"] = float(document.get("spent_usd") or 0.0)
        spent += record["spent_usd"]
        best = shard_dir / ("best" + opened.target.file_ext)
        if not best.is_file():
            record["stop_reason"] = f"shard_failed:{size}"
            state["stop_reason"] = record["stop_reason"]
            _write_state(run_dir, state)
            return spent
        try:
            best.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            record["stop_reason"] = f"shard_failed:{size}"
            state["stop_reason"] = record["stop_reason"]
            _write_state(run_dir, state)
            return spent
        _write_state(run_dir, state)
    return spent


def _assemble(run_dir: Path, opened: _Opened, runners: _Runners, state: dict, shard_spent: float) -> None:
    """@brief 摆好变体，开组装会话，检查通过后才评测。

    @param run_dir 运行目录。
    @param opened 启动材料。
    @param runners 回调。
    @param state 状态文档。
    @param shard_spent 分片已经花掉的美元。
    """
    from .kernel_agent import submission_name
    from .llm import REPO_ROOT, BudgetExhausted, CostLedger, LLMConfig

    ledger = CostLedger(
        LLMConfig(
            base_url="http://portfolio-search.invalid",
            api_key="portfolio-search-not-a-key",
            model=runners.model or "portfolio-search",
        ),
        opened.assemble_budget,
    )
    session = runners.session_runner
    rejection = ""
    for number in range(1, opened.attempts + 1):
        try:
            ledger.check()
        except BudgetExhausted:
            state["stop_reason"] = "budget_exhausted"
            break
        if ledger.remaining is not None and ledger.remaining < 0.05:
            state["stop_reason"] = "budget_exhausted"
            break
        work = run_dir / "assemble" / f"a{number}"
        staged = _stage_variants(work / "src", opened, run_dir, state)
        _write_side_files(work, opened, state)
        label = f"assemble{number}"
        submission = work / submission_name(state["task"], state["dtype"], "portfolio", label, ".metal")
        task_text = render_assemble_task(
            repo=REPO_ROOT,
            task=state["task"],
            dtype=state["dtype"],
            target_id=state["target_id"],
            sizes=tuple(state["sizes"]),
            shapes=_shapes(opened),
            shards=_shard_rows(state),
            src=(work / "src").resolve(),
            submission=submission.resolve(),
            rejection=rejection,
        )
        (work / "TASK.md").write_text(task_text)
        if session is None:
            session = _default_session_runner(opened, runners, ledger)
        result = session(attempt=number, work=work, task_text=task_text, submission=submission)
        provider = getattr(result, "provider", "injected")
        model = getattr(result, "model", "injected")
        if getattr(result, "cost_incomplete", False):
            state["assemble"].append(_attempt(label, provider, model, None, "cost_incomplete", None))
            state["stop_reason"] = "cost_incomplete"
            break
        cost = float(result.cost_usd or 0.0)
        basis = getattr(result, "price_basis", None) or {
            "input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0,
        }
        ledger.charge(cost, label, tokens=getattr(result, "spent", {}) or {}, basis=basis)
        source = getattr(result, "source", None)
        if source is None:
            state["assemble"].append(_attempt(label, provider, model, cost, "no_submission", None))
            rejection = "没有交卷"
            _write_state(run_dir, state)
            continue
        try:
            entries = decode_parent(source)
        except (EnvelopeError, PortfolioAssemblyError) as exc:
            rejection = str(exc)
            state["assemble"].append(_attempt(label, provider, model, cost, "rejected", rejection))
            _write_state(run_dir, state)
            continue
        problems = list(staged_problems(entries, staged))
        report = check_parent(entries, opened.domain)
        if not report.ok:
            problems.extend(report.problems)
        if problems:
            rejection = "\n".join(problems)
            state["assemble"].append(_attempt(label, provider, model, cost, "rejected", rejection))
            _write_state(run_dir, state)
            continue
        outcome = _evaluate_until_settled(run_dir, opened, runners, state, entries, source, task_text)
        if outcome == "rejected":
            rejection = "\n".join(str(item) for item in state.get("_last_problems") or [])
        state["assemble"].append(_attempt(label, provider, model, cost, outcome, None if outcome == "ok" else rejection))
        _write_state(run_dir, state)
        if outcome == "ok":
            state["stop_reason"] = None
            break
        if outcome == "confirmation_rejected":
            state["stop_reason"] = "confirmation_rejected"
            break
        if outcome in ("timing_uncertain", "hang"):
            break
    else:
        if state["stop_reason"] is None:
            state["stop_reason"] = "assemble_failed"
    state["spent_usd"] = shard_spent + ledger.total_usd
    state.pop("_last_problems", None)
    _write_state(run_dir, state)


def _evaluate_until_settled(run_dir, opened, runners, state, entries, source: str, task_text: str) -> str:
    """@brief 评测一次。``timing_uncertain`` 时原样再评一次，不再开会话。

    @param run_dir 运行目录。
    @param opened 启动材料。
    @param runners 回调。
    @param state 状态文档。
    @param entries 父包 entries。
    @param source 父包文本。
    @param task_text 最后一场组装的 TASK.md。只拿去算 guidance 摘要，不落盘。
    @return ``ok``、``confirmation_rejected``、``timing_uncertain``、``hang`` 或 ``rejected``。
    """
    evaluate = runners.evaluate or _default_evaluate
    for pass_number in (1, 2):
        evaluation = evaluate(
            opened.locator, entries, target=opened.target, sizes=opened.domain.sizes, mode="timed",
        )
        status = str(evaluation.status)
        state["evaluation_status"] = status
        state["performance"] = getattr(evaluation, "performance", None)
        state["_last_problems"] = list(getattr(evaluation, "problems", ()) or [])
        portfolio = run_dir / "portfolio"
        portfolio.mkdir(parents=True, exist_ok=True)
        _write_text(portfolio / f"evaluation-{pass_number}.json", _dump(evaluation.document()))
        if status == "hang":
            state["stop_reason"] = "gpu_hang_circuit_breaker"
            _write_text(portfolio / "parent.envelope", source)
            _write_text(portfolio / "evaluation.json", _dump(evaluation.document()))
            return "hang"
        if status == "timing_uncertain" and pass_number == 1:
            continue
        if status == "timing_uncertain":
            state["stop_reason"] = "timing_uncertain"
            _write_text(portfolio / "parent.envelope", source)
            _write_text(portfolio / "evaluation.json", _dump(evaluation.document()))
            return "timing_uncertain"
        if status == "ok":
            _write_text(portfolio / "parent.envelope", source)
            _write_text(portfolio / "evaluation.json", _dump(evaluation.document()))
            return _record_confirmation(run_dir, opened, runners, state, entries, task_text)
        return "rejected"
    return "rejected"


def _record_confirmation(run_dir, opened, runners, state, entries, task_text: str) -> str:
    """@brief 评测 ``ok`` 之后做 fresh 确认。结果只写进状态和 ``confirmations/``。

    不调用 ``write_checkpoint``，不生成 ``DispatchManifest``，不写激活指针。
    ``publication_allowed`` 保持 false。master seed 只进 ``ValidationPlan.derive``，不写进状态文件。

    @param run_dir 运行目录。
    @param opened 启动材料。
    @param runners 回调。``confirm`` 为 None 时用 ``confirm_subject``。
    @param state 状态文档，就地写入 ``confirmation``。
    @param entries 父包 entries。
    @param task_text 组装 TASK.md。
    @return ``ok`` 或 ``confirmation_rejected``。
    """
    from tcenv.contract_errors import TaskContractError
    from tcenv.portfolio.dispatch import DispatchRequest
    from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan

    points = tuple(opened.domain.plan.instances)
    tune = tuple(point.instance_key for point in points if point.usage == "benchmark")
    required = tuple(point.instance_key for point in points if point.usage == "golden")
    if not required:
        state["confirmation"] = {"status": "skipped", "reason": "no_unseen_point"}
        return "ok"
    guidance_digest, master_seed, evidence_for = _confirmation_imports()
    freeze = PortfolioFreeze(
        protocol_epoch=ASSEMBLED_PROTOCOL_EPOCH,
        authority=opened.domain.authority,
        candidate_entries=tuple(entries),
        task_pack_digest=opened.locator.deployment_pack_digest,
        guidance_digest=guidance_digest(task_text),
    )
    validation = ValidationPlan.derive(
        freeze,
        tune_shards=tune,
        unseen_count=len(required),
        master_seed=master_seed(int(state["run_seed"])),
        required_instance_keys=required,
    )
    portfolio = freeze.portfolio()
    evidence = evidence_for(portfolio)
    by_size = {str(instance.size): instance for instance in opened.domain.instances}
    by_key = {point.instance_key: point for point in points}
    subject = runners.confirm or _default_confirm_subject
    directory = run_dir / "confirmations"
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for key in (*required, *tune):
        point = by_key[key]
        request = DispatchRequest.from_instance(
            by_size[str(point.size)],
            target_abi_digest=point.target_abi_digest,
            usage=point.usage,
        )
        try:
            document = subject(
                locator=opened.locator,
                domain_plan=opened.domain.plan,
                portfolio=portfolio,
                request=request,
                evidence=evidence,
                validation=validation,
                freeze=freeze,
            )
        except TaskContractError as exc:
            rows.append({
                "instance_key": key,
                "status": "rejected",
                "reason": str(exc),
                "coverage_kind": None,
            })
            break
        if not isinstance(document, dict):
            document = {}
        status = str(document.get("status") or "rejected")
        rows.append({
            "instance_key": key,
            "status": status,
            "reason": document.get("reason"),
            "coverage_kind": document.get("coverage_kind"),
        })
        _write_text(directory / f"{key.replace('/', '__')}.json", _dump(document))
    failed = [row for row in rows if row["status"] != "confirmed"]
    state["confirmation"] = {
        "status": "rejected" if failed else "confirmed",
        "points": rows,
        "reasons": [row["reason"] for row in failed],
        "portfolio_freeze_digest": freeze.digest,
        "validation_plan_digest": validation.digest,
    }
    return "ok" if not failed else "confirmation_rejected"


def _confirmation_imports():
    """@brief 确认用到的三个外部函数。都在函数里导入，模块顶层不碰后端。

    @return ``(guidance_fingerprint, confirmation_master_seed, evidence_for)``。
    """
    from tcenv.backends.metal.search_candidate import evidence_for, guidance_fingerprint

    from .orchestrator import confirmation_master_seed

    return guidance_fingerprint, confirmation_master_seed, evidence_for


def _default_confirm_subject(**kwargs):
    """@brief 缺省的逐点确认。不传 ``track_id`` 和 ``track_version``。测试会把它换掉。

    @param kwargs ``confirm_subject`` 的关键字参数。
    @return 该点的确认文档。
    """
    from tcenv.backends.metal.portfolio_confirmation import confirm_subject

    return confirm_subject(**kwargs)


def _stage_variants(src: Path, opened: _Opened, run_dir: Path, state: dict) -> dict[str, dict[str, str]]:
    """@brief 把每个分片的 ``best`` 文件摆进 ``src/variants/<size>/``。

    @param src 源目录。
    @param opened 启动材料。
    @param run_dir 运行目录。
    @param state 状态文档。
    @return 每个 size 的相对路径到 sha256。
    """
    staged: dict[str, dict[str, str]] = {}
    extension = opened.target.file_ext
    for size in opened.domain.sizes:
        best = Path(state["shards"][size]["directory"]) / ("best" + extension)
        text = best.read_bytes().decode("utf-8")
        destination = src / "variants" / size
        if destination.exists():
            _clear_dir(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if opened.track == "msl-bundle":
            artifact = decode_envelope(text)
            for entry in artifact.entries:
                path = destination / entry.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entry.data)
        else:
            (destination / "kernel.metal").write_text(text, encoding="utf-8")
        staged[size] = _fingerprint_tree(destination)
    return staged


def _fingerprint_tree(root: Path) -> dict[str, str]:
    """@brief 目录下每个普通文件的相对路径和 sha256。

    @param root 变体目录。
    @return 映射。
    """
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(root).as_posix()
            found[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def _write_side_files(work: Path, opened: _Opened, state: dict) -> None:
    """@brief 写 ``INSTANCES.json`` 和 ``SHARDS.json``。不写 ``coverage.json``。

    @param work 会话目录。
    @param opened 启动材料。
    @param state 状态文档。
    """
    instances = {}
    for instance in opened.domain.instances:
        instances[str(instance.size)] = {
            "usage": [str(item) for item in instance.size_usage],
            "symbols": {str(name): int(value) for name, value in instance.symbols.items() if isinstance(value, int) and not isinstance(value, bool)},
        }
    shards = {
        size: {
            "best_speedup": row.get("best_speedup"),
            "confirmed_speedup": row.get("confirmed_speedup"),
            "track": state["track"],
        }
        for size, row in state["shards"].items()
    }
    _write_text(work / "INSTANCES.json", _dump(instances))
    _write_text(work / "SHARDS.json", _dump(shards))


def _shapes(opened: _Opened) -> list[dict]:
    """@brief 给 TASK.md 的形状表。

    @param opened 启动材料。
    @return 每个 size 一行。
    """
    rows = []
    for instance in opened.domain.instances:
        rows.append({
            "size": str(instance.size),
            "usage": [str(item) for item in instance.size_usage],
            "symbols": {
                str(name): int(value)
                for name, value in instance.symbols.items()
                if isinstance(value, int) and not isinstance(value, bool)
            },
        })
    return rows


def _shard_rows(state: dict) -> list[dict]:
    """@brief 给 TASK.md 的搜索期成绩表。

    @param state 状态文档。
    @return 每个 size 一行。
    """
    return [
        {
            "size": size,
            "best_speedup": row.get("best_speedup"),
            "confirmed_speedup": row.get("confirmed_speedup"),
            "track": state["track"],
        }
        for size, row in state["shards"].items()
    ]


def _default_shard_runner(opened: _Opened, runners: _Runners, run_seed: int):
    """@brief 缺省分片搜索。会读凭据并构造 ``Orchestrator``。测试不调用它。

    @param opened 启动材料。
    @param runners 带回配置路径的回调容器。
    @param run_seed 运行种子。
    @return ``(size, shard_dir)`` 可调用对象。
    """
    from .committees import load_committees
    from .llm import load_config, load_raw_config
    from .orchestrator import Orchestrator

    raw = load_raw_config(runners.config, runners.preset)
    cfg, search = load_config(runners.config, model=runners.model, preset=runners.preset, budget_usd=opened.budget)
    committees = load_committees(raw, active=cfg, model_override=runners.model)

    def run(size, shard_dir):
        orch = Orchestrator(
            opened.locator.op_name, opened.locator.dtype, size, cfg, search,
            run_dir=shard_dir, committees=committees, target_id=opened.target.id,
            track_id=opened.track, seed=run_seed,
        )
        return asyncio.run(orch.run())

    return run


def _default_session_runner(opened: _Opened, runners: _Runners, ledger):
    """@brief 缺省组装会话。会读凭据并调用 hmz。测试不调用它。

    输出 token 上限用组装账本的剩余额度，并且不超过 ``relay_budget_usd``。

    @param opened 启动材料。
    @param runners 回调容器。
    @param ledger 组装会话的账本。每次调用时读它的剩余额度。
    @return 会话函数。
    """
    from .committees import load_committees
    from .kernel_agent import Prices, SessionBrief, output_tokens_for, run_kernel_agent
    from .llm import REPO_ROOT, load_config, load_raw_config

    raw = load_raw_config(runners.config, runners.preset)
    cfg, search = load_config(runners.config, model=runners.model, preset=runners.preset)
    committees = load_committees(raw, active=cfg, model_override=runners.model)

    def run(*, attempt, work, task_text, submission):
        member = committees[(attempt - 1) % 2].members[0]
        prices = Prices(
            member.cfg.price_in, member.cfg.price_out, member.cfg.price_cache_read, member.cfg.price_cache_write,
        )
        remaining = ledger.remaining
        slice_usd = search.relay_budget_usd if remaining is None else min(remaining, search.relay_budget_usd)
        tokens = output_tokens_for(slice_usd, prices.output)
        label = f"assemble{attempt}"
        brief = SessionBrief(
            repo=REPO_ROOT, task_name=opened.locator.op_name, dtype=opened.locator.dtype, size="portfolio",
            kernel_name="portfolio", triton_source="", description="", contract_text="", abi_stub="",
            rtol=0.0, atol_rel=0.0, seedless=True, label=label, work_dir=str(work),
            target_id=opened.target.id, track_id="msl-bundle", v2=True,
        )
        result = run_kernel_agent(
            brief, work=work, provider=member.provider, model=member.model,
            effort=member.cfg.reasoning_effort, prices=prices, turn_output_tokens=tokens,
            turn_minutes=int(search.turn_minutes), task_text=task_text,
        )
        return _SessionView(result, member.provider, member.model, prices.as_dict())

    return run


def _default_evaluate(locator, entries, *, target, sizes, mode):
    """@brief 缺省评测。导入 Metal 并跑 timed。测试不调用它。

    @param locator 任务定位器。
    @param entries 父包 entries。
    @param target 目标档案。
    @param sizes size 列表。
    @param mode 评测模式。
    @return 评测结果。
    """
    from tcenv.backends.metal.portfolio_eval import evaluate_parent

    return evaluate_parent(locator, entries, target=target, sizes=sizes, mode=mode)


class _SessionView:
    """@brief 把 hmz 结果补上 provider、model 和费率。"""

    def __init__(self, result, provider, model, price_basis):
        self.source = result.source
        self.cost_usd = result.cost_usd
        self.cost_incomplete = result.cost_incomplete
        self.spent = result.spent
        self.provider = provider
        self.model = model
        self.price_basis = price_basis


def _attempt(label, provider, model, cost, result, rejection) -> dict[str, Any]:
    """@brief 一场组装尝试的状态行。

    @param label 尝试标签。
    @param provider provider 名。
    @param model 模型名。
    @param cost 美元；不完整时为 None。
    @param result 结果名。
    @param rejection 拒绝原因。
    @return 字典。
    """
    return {
        "label": label,
        "provider": provider,
        "model": model,
        "cost_usd": cost,
        "result": result,
        "rejection": rejection,
    }


def _write_state(run_dir: Path, state: dict) -> None:
    """@brief 原子重写 ``portfolio_search.json``。不写 ``portfolio_run.json``。

    @param run_dir 运行目录。
    @param state 状态文档。
    """
    from . import control

    payload = {key: value for key, value in state.items() if not str(key).startswith("_")}
    control.write_json_atomic(run_dir / _STATE_NAME, payload)


def _write_text(path: Path, text: str) -> None:
    """@brief 原子写文本。

    @param path 目标。
    @param text 全文。
    """
    from . import control

    control.write_text_atomic(path, text if text.endswith("\n") else text + "\n")


def _dump(document) -> str:
    """@brief 把文档打成 JSON 文本。

    @param document 可序列化对象。
    @return JSON。
    """
    import json

    return json.dumps(document, indent=2, sort_keys=True, default=str)


def _read_json(path: Path) -> dict:
    """@brief 读一个 JSON 对象。坏文件当空对象。

    @param path 文件。
    @return 字典。
    """
    import json

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def _clear_dir(path: Path) -> None:
    """@brief 删掉目录里的普通文件，好让第二次尝试重新摆核。

    @param path 目录。
    """
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()


def _stderr():
    """@brief 标准错误流。启动失败的原因打到这里。

    @return ``sys.stderr``。
    """
    import sys

    return sys.stderr
