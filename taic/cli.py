"""``taic`` command line: run the AI compiler on a task."""

# 命令行进入搜索的地方。`taic run` 开一次新搜索；`taic resume` 从已有 run 目录接着跑，已经花掉的模型请求和 GPU 评测不会再花一遍。
# `taic brief` 只读：没有目录时打印仓库方位，有目录时打印一份短 JSON。不发模型请求，不上 GPU。
# 新搜索的预算和预设不在这里写死。未指定 --config 时，load_config 读 taic.toml，再按预设名叠 taic.<preset>.toml；预设名是 --preset，否则环境变量 TAIC_PRESET，否则 DEFAULT_PRESET（paper，论文的 $15，即 PAPER_BUDGET_USD）。--budget 只在显式给出时覆盖 search.budget_usd。
# 续跑的预算另算：目录里记下的花费和上限先恢复，--budget 改总上限，--add-budget 加在记下的上限上；两者都没有就沿用记下的上限或 control.json 里还没生效的请求，但新上限必须高于已花费。没记下的键才回到预设，默认 base（只有 taic.toml）。

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# 一次新搜索从这里进入。load_config 把预设和命令行里非空的预算、分支数、轮数上限合成 LLMConfig 与 SearchConfig，再交给 Orchestrator.run。有 best_speedup 则进程成功，一颗合格内核都没有则失败。
def _track_version_of(block: object, run_dir: Path) -> int:
    """@brief 严格读取记录里的 ``track_version``（拒绝 bool/浮点/字符串 coercion）。

    @param block 记录块。
    @param run_dir 运行目录（诊断）。
    @return 版本整数。
    @raises ResumeRefused 缺失或不是 ``int >= 1``（bool 不是版本）。
    """
    value = block.get("track_version") if isinstance(block, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResumeRefused(
            f"cannot resume {run_dir}: the recorded track version is {value!r}; a version is an integer >= 1 and "
            "no coercion is applied",
            5,
        )
    return int(value)


def _recorded_track_selection(
    run_dir: Path, *, requested: str | None = None
) -> tuple[str | None, int | None]:
    """@brief 续跑记录里的 ``(track_id, track_version)``（一次解析，两个 getter 共用）。

    A resumed run must restore the *exact* frozen selection: the id alone is not enough once two
    versions of one Track coexist.  ``(None, None)`` is a v1 run; ``("msl-single", 1)`` is the
    recorded selection of a v2 run that predates the block.

    @param run_dir 运行目录。
    @param requested ``--track``（只校验一致性，不改变记录）。
    @return ``(track_id, track_version)``。
    @raises ResumeRefused 记录非法或与 registry/--track 不符。
    """
    from tcenv.contract_errors import UnsupportedContractCapability
    from tcenv.track.registry import load_track_registry

    path = run_dir / "run_config.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeRefused(f"cannot resume {run_dir}: run_config.json is unreadable ({exc})", 5) from exc
    if not isinstance(document, dict):
        raise ResumeRefused(f"cannot resume {run_dir}: run_config.json is not an object", 5)
    v2 = document.get("task_mode") == "v2"
    if not v2:
        if requested and requested != "msl-single":
            raise ResumeRefused(
                f"this v1 run can only resume as msl-single; --track {requested} does not apply", 5
            )
        return None, None
    block = document.get("track", "__missing__")
    if block == "__missing__":
        # schema < 3: the run predates the track block and produced single files, before versioning.
        recorded, version = "msl-single", 1
    else:
        if not isinstance(block, dict):
            raise ResumeRefused(f"cannot resume {run_dir}: the recorded track block is malformed", 5)
        missing = [key for key in ("track_id", "track_version", "delivery") if key not in block]
        if missing:
            raise ResumeRefused(
                f"cannot resume {run_dir}: the recorded track block is missing {missing}", 5
            )
        recorded = block["track_id"]
        if not isinstance(recorded, str) or not recorded:
            raise ResumeRefused(f"cannot resume {run_dir}: the recorded track id is not a string", 5)
        version = _track_version_of(block, run_dir)
        try:
            track = load_track_registry().get(recorded, version=version)
        except (UnsupportedContractCapability, TypeError, ValueError) as exc:
            raise ResumeRefused(f"cannot resume {run_dir}: {exc}", 5) from exc
        if block["delivery"] != track.delivery:
            raise ResumeRefused(
                f"cannot resume {run_dir}: track {recorded!r} is recorded as delivery {block['delivery']!r}, but the "
                f"registry says {track.delivery!r}",
                5,
            )
    if requested and requested != recorded:
        raise ResumeRefused(
            f"this run delivers {recorded}; --track {requested} does not match", 5
        )
    return recorded, version


def _recorded_track(run_dir: Path, *, requested: str | None = None) -> str | None:
    """@brief 续跑时要用的 track id：记录值 × ``--track`` × registry 的交叉校验。

    Compatibility rules, in the order they are decided:

    * a v1 run records ``track=None``: keep ``None`` (the caller passes nothing) and refuse an explicit
      non-single ``--track``;
    * a v2 run that predates the ``track`` block (schema < 3) resumes as ``msl-single`` -- it cannot
      have produced an envelope;
    * a well-formed block is cross-checked against the registry: an unknown id, a version mismatch or
      an internally contradictory ``delivery`` is a refusal, not a traceback;
    * an explicit ``--track`` that differs from the record is a refusal.

    @param run_dir run directory.
    @param requested explicit ``--track`` value, or ``None``.
    @return the track id to hand to ``Orchestrator``, or ``None`` for a v1 run.
    @raises ResumeRefused the record is malformed, unknown, contradictory, or contradicted by the CLI.
    """
    track_id, _version = _recorded_track_selection(run_dir, requested=requested)
    return track_id


def _recorded_track_version(run_dir: Path) -> int | None:
    """@brief 续跑记录里的冻结 Track 版本（v1 运行没有版本）。

    @param run_dir 运行目录。
    @return 版本整数或 None。
    @raises ResumeRefused 记录非法。
    """
    _track_id, version = _recorded_track_selection(run_dir)
    return version


def _recorded_target(run_dir: Path, requested: str | None) -> str:
    """@brief 读出这次运行记录下的目标档，并在任何模型调用之前拒绝不一致的 ``--target``。

    先看 ``run_config.json`` 里的 ``target.id``；文件不存在、读不动或结构不合法时退回 ``apple-m4-max``。

    @param run_dir 运行目录。
    @param requested 命令行给的 ``--target``；为 ``None`` 或与记录一致时不报错。
    @return 记录下来的目标 id，交给续跑时的 Orchestrator。
    @raises ResumeRefused 当 ``requested`` 与记录不一致时，退出码 5。
    """
    import json

    recorded = "apple-m4-max"
    path = Path(run_dir) / "run_config.json"
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            target = doc.get("target") if isinstance(doc, dict) else None
            if isinstance(target, dict) and target.get("id"):
                recorded = str(target["id"])
        except (OSError, json.JSONDecodeError):
            recorded = "apple-m4-max"
    if requested and requested != recorded:
        raise ResumeRefused(f"this run is target {recorded}; --target {requested} does not match", 5)
    return recorded


def _run_track_choices() -> list[str]:
    """Advertise exactly the currently available registered Tracks, in registry order."""
    from tcenv.track.registry import load_track_registry

    return [track.track_id for track in load_track_registry().available()]


def _resume_track_choices() -> list[str]:
    """Resume parses every registered ID; recorded-version/availability checks stay downstream."""
    from tcenv.track.registry import load_track_registry

    return list(load_track_registry().track_ids)


def _requested_track(a: argparse.Namespace, raw: dict | None = None) -> str | None:
    """@brief 本次要用的 track id：CLI ``--track`` > ``taic.toml`` 里的同名项 > 未指定。

    The parser's ``--track`` defaults to ``None`` so "not given" is distinguishable from an explicit
    ``msl-single``; ``resume`` needs that to tell "keep what the run recorded" from "the caller asked
    for something else".  The TOML value is read from the merged raw section (``load_raw_config``) and
    deliberately *not* added to ``SearchConfig``: the delivery mechanism is a task-binding fact, not a
    search knob.

    @param a parsed arguments.
    @param raw merged raw settings sections.
    @return the requested track id, or ``None`` when nothing was requested.
    """
    explicit = getattr(a, "track", None)
    if explicit:
        return str(explicit)
    section = (raw or {}).get("search") if isinstance(raw, dict) else None
    if isinstance(section, dict) and section.get("track"):
        return str(section["track"])
    return None


def _selfcheck_bridge_for(orch, remote, workspace):
    """Decide whether this run installs the per-session host self-check bridge.

    A remote workspace has no self-check channel: the agent runs on a host the bridge does
    not reach, and the transport refuses to pretend otherwise.  Formal evaluation still uses
    the remote evaluator.  The task-fingerprint check is about the evaluator, not the
    workspace, so it runs in both cases.

    @raises ValueError when the remote evaluator targets a different task.
    """
    if remote is None:
        return None
    if remote.task_fingerprint and remote.task_fingerprint != orch.task_fingerprint:
        raise ValueError("remote evaluator task fingerprint differs from this run")
    if workspace is not None:
        orch.log("== note: a remote workspace has no self-check channel; only formal evaluation is enabled")
        return None
    return remote.make_bridge_factory(
        task=orch.task_name, dtype=orch.dtype, size=orch.size,
        task_fingerprint=orch.task_fingerprint, run_id=orch.run_dir.name,
        require_contract_launch=bool(orch.search.require_contract_launch),
        min_improvement=float(orch.search.min_improvement), noise_z=float(orch.search.noise_z),
    )


def _result_code(res) -> int:
    """@brief 已结束的一次搜索对应的进程退出码。

    0：拿到合格内核，确认阶段绑定成功或不适用。6：搜索产出了合格内核，但确认阶段（confirmation）失败，
    所以任何数字都不得发布；只看状态的脚本不能把这种情况读成完全成功。2：搜索跑完了但没有一颗合格内核。

    @param res 编排器返回的结果对象，需带 ``stop_reason`` 与 ``best_speedup``。
    @return ``0`` / ``6`` / ``2`` 之一。
    """
    if str(getattr(res, "stop_reason", "") or "").startswith("confirm_error"):
        return 6
    return 0 if res.best_speedup else 2


def cmd_track_explore(a: argparse.Namespace) -> int:
    """Run or read independent single-Track searches and write an exploration-only summary."""
    from .track_explore import TrackExploreConfig, TrackExploreError, explore_tracks

    try:
        config = TrackExploreConfig.load(a.config)
        summary = explore_tracks(config, summarize_only=bool(a.summarize_only))
    except (TrackExploreError, OSError, ValueError) as exc:
        print(f"track-explore: {exc}", file=sys.stderr)
        return 5
    print(json.dumps({
        "status": summary["status"],
        "winner": summary["winner"],
        "output": str(config.output_path),
        "exploration_only": True,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _validate_run_instance(a: argparse.Namespace):
    """Resolve the requested instance from its actual task declaration before any run setup."""
    from .task_binding import TaskBindingError, bind_task

    try:
        return bind_task(a.task, a.dtype, a.size)
    except (TaskBindingError, KeyError, ValueError) as exc:
        raise ValueError(
            f"task instance {a.task!r} dtype={a.dtype!r} size={a.size!r} is not declared by its OpSpec/task contract: {exc}"
        ) from exc


def cmd_run(a: argparse.Namespace) -> int:
    """@brief 开一次新搜索：把命令行、预设与 ``taic.toml`` 合成配置，装好两个委员会，再交给 ``Orchestrator.run``。

    新搜索从这里进入。``load_raw_config``/``load_config`` 把预设和命令行里显式给出的预算、分支数、轮数上限
    合成 ``LLMConfig`` 与 ``SearchConfig``。预设与预算在这里汇合：``--config`` 有值时整份文件原样使用，不再叠
    预设；``--budget`` 为 ``None`` 时保留预设里的 ``budget_usd``（paper 即 ``PAPER_BUDGET_USD``）。命令行开关
    写在配置之后，所以这些开关优先于 ``taic.toml`` 和预设；没写的可选整数保持配置值，``0`` 也当成没写。
    显式 ``--model`` 让两个委员会都用这个模型，没写时用配置里的两名成员（可以不同）。新 run 同样要拿目录锁：
    「目录是空的」是 check-then-write，两个 ``taic run --run-dir X`` 会同时写 run_config/transcript/candidates；
    锁靠 pid 判断过期，进程退出后自然可接管。有 ``best_speedup`` 则进程成功，一颗合格内核都没有则失败。

    @param a argparse 命名空间，含 ``task``/``dtype``/``size``/``run_dir``/``seed``/``target``/``from_run``
        以及预算、轮数、effort 与各搜索开关。
    @return 进程退出码：跑完为 ``_result_code`` 的 0/6/2；``--kernel-agent off``、尺寸或任务名不合法、配置写不
        进去为 5；另一个进程占着运行目录锁为 4。
    """
    from tcenv.children import install_termination_handlers

    install_termination_handlers()
    try:
        _validate_run_instance(a)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    from .committees import load_committees
    from .llm import LLMError, load_config, load_raw_config
    from .orchestrator import Orchestrator, RunStartError

    # 预设与预算的汇合点。a.config 有值时整份文件原样使用，不再叠预设；a.budget 为 None 时保留预设里的 budget_usd（paper 即 PAPER_BUDGET_USD）。
    raw = load_raw_config(a.config, a.preset)
    cfg, search = load_config(
        a.config, model=a.model, preset=a.preset, budget_usd=a.budget, branches=a.branches,
        max_rounds=a.max_rounds, provider_retries=getattr(a, "provider_retries", None),
        session_cost_cap_usd=getattr(a, "session_cost_cap_usd", None),
    )
    # 写在配置之后，所以这些开关优先于 taic.toml 和预设。没写的可选整数保持配置值；0 也当成没写。
    if a.no_tools:
        cfg.use_tools = False
    if a.profile_best:
        search.profile_best = True
    if a.profile_poor:
        search.profile_poor = True
    if getattr(a, "hw_feedback", False):  # opt-in: static hardware-feature block in the feedback + Skill 2c in the system prompt (taic/hwfeedback.py)
        search.hw_feedback = True
    _apply_search_switches(a, cfg, search)
    if a.plan_mix:
        search.plan_mix = True
    if a.init_samples:
        search.init_samples = a.init_samples
    _apply_output_switches(a, search)
    if a.eval_reps is not None:
        search.eval_reps = a.eval_reps
    if a.correctness_only is not None:
        search.correctness_only = a.correctness_only
    if a.confirm_batches:
        search.confirm_batches = a.confirm_batches
    if a.patience:
        search.patience = a.patience
    if getattr(a, "kernel_agent", None):
        if a.kernel_agent == "off":
            print("kernel writing goes through hmz; --kernel-agent off is not available", file=sys.stderr)
            return 5
        search.kernel_agent = a.kernel_agent
    if a.effort:
        cfg.reasoning_effort = a.effort
    if getattr(a, "plan_effort", None):
        cfg.plan_reasoning_effort = a.plan_effort
    if getattr(a, "patch_effort", None):
        cfg.patch_reasoning_effort = a.patch_effort
    if getattr(a, "truncation_retry_effort", None):
        cfg.truncation_retry_effort = a.truncation_retry_effort
    for flag, key in (("plan_ledger", "plan_ledger"), ("diversity_archive", "diversity_archive"), ("fast_bench", "fast_bench")):
        value = getattr(a, flag, None)
        if value is not None:
            setattr(search, key, value)
    if getattr(a, "experience_index", None):
        search.experience_index = a.experience_index
    _apply_max_tokens(a, cfg)
    from . import control

    # 新 run 也要拿目录锁："目录是空的" 是 check-then-write，两个 `taic run --run-dir X`
    # 会同时写 run_config/transcript/candidates。lock 靠 pid 判断过期，进程退出后自然可接管。
    lock_dir = Path(a.run_dir).resolve() if a.run_dir else None
    if lock_dir is not None:
        try:
            control.acquire_run_lock(lock_dir, who="run")
        except control.RunLockError as e:
            print(f"cannot start {lock_dir}: {e}", file=sys.stderr)
            return 4
    try:
        try:
            # 显式 --model 让两个委员会都用这个模型。没写时用配置里的两名成员，可以不同。
            committees = load_committees(raw, active=cfg, model_override=a.model)
            from .remote_runtime import RemoteEvaluationConfig
            from .workspace_env import RemoteWorkspaceConfig, WorkspaceError
            remote = RemoteEvaluationConfig.from_file(a.remote_evaluator_config) if getattr(a, "remote_evaluator_config", None) else None
            try:
                workspace = RemoteWorkspaceConfig.from_file(a.remote_workspace_config) if getattr(a, "remote_workspace_config", None) else None
            except WorkspaceError as exc:
                raise ValueError(f"remote workspace config: {exc.message}") from exc
            if remote is not None:
                if remote.target_id != getattr(a, "target", "apple-m4-max"):
                    raise ValueError(f"remote evaluator target {remote.target_id!r} differs from --target")
                if not remote.publishable_site_ids:
                    raise ValueError("remote evaluator config has no publishable_site_ids; refusing to start a search")
                if not remote.publishable_epochs:
                    raise ValueError("remote evaluator config has no publishable_epochs; refusing to start a search")
                if remote.selfcheck_token_file is None:
                    raise ValueError("remote evaluator config requires selfcheck_token_file for selfcheck")
            orch = Orchestrator(
                a.task, a.dtype, a.size, cfg, search,
                run_dir=Path(a.run_dir) if a.run_dir else None,
                from_run=None if a.from_run in (None, "@catalog") else Path(a.from_run),
                seed=getattr(a, "seed", None),
                committees=committees,
                target_id=getattr(a, "target", "apple-m4-max"),
                from_catalog=a.from_run == "@catalog",
                evaluator_factory=None if remote is None else remote.make_evaluator,
                publishable_site_ids=None if remote is None else remote.publishable_site_ids,
                publishable_epochs=None if remote is None else remote.publishable_epochs,
                workspace_config=workspace,
                track_id=_requested_track(a, raw),
            )
            bridge_factory = _selfcheck_bridge_for(orch, remote, workspace)
            if bridge_factory is not None:
                orch.selfcheck_bridge_factory = bridge_factory
        except (RunStartError, ValueError, LLMError) as e:
            # 尺寸或任务名不合法是启动失败，和配置写不进去一样，退出码 5。
            print(str(e), file=sys.stderr)
            return 5
        try:
            res = asyncio.run(orch.run())
        except RunStartError as e:
            print(str(e), file=sys.stderr)
            return 5
        print(json.dumps({k: getattr(res, k) for k in ("task", "dtype", "model", "best_speedup", "confirmed_speedup", "spent_usd", "stop_reason", "run_dir")}, indent=2))
        return _result_code(res)
    finally:
        if lock_dir is not None:
            control.release_run_lock(lock_dir)


SEARCH_FIELDS = ("budget_usd", "branches", "max_repair_tries", "max_rounds", "max_quick_checks", "min_improvement", "noise_z", "confirm_top", "eval_reps", "max_hangs",
                 "correctness_only", "profile_best", "patience", "plan_mix", "init_samples", "confirm_batches", "profile_poor", "profile_poor_below", "profile_max", "hw_feedback",
                 "stop_at_roofline", "abi_guard", "target_note", "rich_history", "status_lines", "differential_recheck",
                 "plan_ledger", "diversity_archive", "fast_bench", "experience_index", "kernel_agent", "require_contract_launch",
                 "relay_budget_usd", "turn_minutes", "provider_retries", "session_cost_cap_usd", "peer_tips", "harness_auto", "harness_shadow", "harness_review", "inherit_prompt", "harness_experiment_id", "harness_experiment_arm", "harness_change_sha256", "harness_change_path", "harness_base_commit",
                 "task_ledger", "plateau_after", "sweep_tool")
LLM_FIELDS = ("model", "reasoning_effort", "echo_reasoning_content", "use_tools", "keep_transcript_bodies", "extra_body", "max_tokens", "request_timeout_s", "max_retries", "truncation_retry",
              "truncation_retry_effort", "plan_reasoning_effort", "patch_reasoning_effort")


# 续跑的配置以这次运行自己记下的设置为准，命令行只覆盖显式给出的项。没有 run_config.json 的老运行用 base，避免被后来的 paper 预设改掉当时的预算和轮数。dry_run 即使没有 API 凭证也要能组出配置，否则「先看看」会因为没钥匙而做不成。
def _resume_config(a: argparse.Namespace, settings: dict, info: dict):
    """@brief 组出续跑用的 ``(LLMConfig, SearchConfig, provenance)``：先取运行自己记下的设置，再叠命令行显式给出的项。

    续跑的配置以这次运行自己记下的设置为准，命令行只覆盖显式给出的项。目录里记下的键盖过预设，命令行再盖过
    目录，每次覆盖都记进 provenance。这里 ``0`` 也算显式给出、会盖过目录里的值（新搜索的 ``cmd_run`` 把 ``0``
    当成没写，两边不一样）。没有 ``run_config.json`` 的老运行用 base 预设，只读 ``taic.toml``，不把后来的 paper
    预设里的 ``$15`` 和轮数盖到一次已经跑过的搜索上。``dry_run`` 即使没有 API 凭证也要能组出配置，否则「先看看」
    会因为没钥匙而做不成；真续跑时凭证错误直接失败。目录里没记的键，provenance 一律标成来自预设/默认。

    @param a argparse 命名空间：``config``/``preset``/``model``/``dry_run`` 与各配置开关。
    @param settings ``resume.infer_settings`` 推出来的本次运行设置，含 ``llm``/``search``/``provenance`` 三段。
    @param info ``resume.inspect_run`` 的检查结果，只用来读 ``model``。
    @return ``(cfg, search, prov)``：LLM 配置、搜索配置，以及每个键来源的 provenance 字典。
    @raises LLMError 凭证或配置不合法且不是 ``--dry-run``；续跑时给 ``--kernel-agent off`` 也走这条。
    """
    from .llm import LLMConfig, LLMError, SearchConfig, load_config, load_raw_config

    prov = dict(settings["provenance"])
    model = a.model or settings["llm"].get("model") or info.get("model")
    # 老运行没有预设记录。默认 base 表示只读 taic.toml，不把 paper 的 $15 和轮数盖到一次已经跑过的搜索上。
    preset = a.preset or "base"  # runs without run_config.json predate the presets: taic.toml alone, as they were started
    try:
        cfg, search = load_config(a.config, model=model, preset=preset)
    except LLMError:
        # dry_run 不发请求，缺凭证仍应能看续跑计划。真续跑则凭证错误直接失败。
        if not a.dry_run:
            raise
        raw = load_raw_config(a.config, preset)  # a dry run needs no credentials
        cfg = LLMConfig(base_url="http://dry-run.invalid", api_key="dry-run-not-a-key", model=model or "deepseek-flash", reasoning_effort=raw.get("llm", {}).get("reasoning_effort", "high"),
                        use_tools=bool(raw.get("llm", {}).get("use_tools", True)))
        search = SearchConfig(**{k: v for k, v in raw.get("search", {}).items() if k in SearchConfig.__dataclass_fields__})
    # 目录里记下的键盖过预设。下面命令行上显式给了的项再盖过目录，并记进 provenance。
    for k in LLM_FIELDS:
        if k in settings["llm"]:
            setattr(cfg, k, settings["llm"][k])
    for k in SEARCH_FIELDS:
        if k in settings["search"]:
            setattr(search, k, settings["search"][k])
    for k in SEARCH_FIELDS:
        if f"search.{k}" not in prov:
            prov[f"search.{k}"] = f"preset '{preset}' / default (not recorded by this run)"
    for k in LLM_FIELDS:
        if f"llm.{k}" not in prov:
            prov[f"llm.{k}"] = f"preset '{preset}' / default (not recorded by this run)"
    # 这里 0 也算显式给出，会盖过目录里的值。新搜索的 cmd_run 把 0 当成没写，两边不一样。
    flags = {"patience": a.patience, "branches": a.branches, "max_rounds": a.max_rounds, "init_samples": a.init_samples,
             "provider_retries": getattr(a, "provider_retries", None),
             "session_cost_cap_usd": getattr(a, "session_cost_cap_usd", None)}
    for k, v in flags.items():
        if v is not None:
            setattr(search, k, v)
            prov[f"search.{k}"] = "command line"
    _apply_output_switches(a, search, prov)
    for flag, k in (("plan_mix", "plan_mix"), ("profile_poor", "profile_poor"), ("profile_best", "profile_best")):
        if getattr(a, flag):
            setattr(search, k, True)
            prov[f"search.{k}"] = "command line"
    if getattr(a, "hw_feedback", False):
        search.hw_feedback = True
        prov["search.hw_feedback"] = "command line"
    _apply_search_switches(a, cfg, search, prov)
    if a.no_tools:
        cfg.use_tools = False
        prov["llm.use_tools"] = "command line"
    if getattr(a, "kernel_agent", None):
        if a.kernel_agent == "off":
            raise LLMError("kernel writing goes through hmz; --kernel-agent off is not available")
        search.kernel_agent = a.kernel_agent
        prov["search.kernel_agent"] = "command line"
    if a.effort:
        cfg.reasoning_effort = a.effort
        prov["llm.reasoning_effort"] = "command line"
    if getattr(a, "plan_effort", None):
        cfg.plan_reasoning_effort = a.plan_effort
        prov["llm.plan_reasoning_effort"] = "command line"
    if getattr(a, "patch_effort", None):
        cfg.patch_reasoning_effort = a.patch_effort
        prov["llm.patch_reasoning_effort"] = "command line"
    if getattr(a, "truncation_retry_effort", None):
        cfg.truncation_retry_effort = a.truncation_retry_effort
        prov["llm.truncation_retry_effort"] = "command line"
    for flag, key in (("plan_ledger", "plan_ledger"), ("diversity_archive", "diversity_archive"), ("fast_bench", "fast_bench")):
        value = getattr(a, flag, None)
        if value is not None:
            setattr(search, key, value)
            prov[f"search.{key}"] = "command line"
    if getattr(a, "experience_index", None):
        search.experience_index = a.experience_index
        prov["search.experience_index"] = "command line"
    _apply_max_tokens(a, cfg, prov)
    return cfg, search, prov


def _committees_for_resume(a: argparse.Namespace, cfg, run_dir: Path):
    """@brief 续跑用的两个委员会：目录里记下的成员优先，除非 ``--model`` 让两个委员会都用同一个模型。

    委员会机制出现之前开始的运行会拿到两个照抄它模型的委员会；轮次号仍然决定下一个由谁出场。
    ``--dry-run`` 下即使凭证不合法，也要能根据记录装出委员会。

    @param a argparse 命名空间，读 ``config``/``preset``/``model``/``dry_run``。
    @param cfg 已经组好的 ``LLMConfig``，用于给成员派生配置与取默认模型。
    @param run_dir 运行目录，从中读 ``run_config.json`` 里记下的 ``committees``。
    @return 已按轮次排序的 ``Committee`` 列表；没有记录时是 ``default_committees`` 的结果。
    @raises LLMError 记录里的成员无法解析，且不是 ``--dry-run``。
    """
    from dataclasses import replace

    from . import resume as rs
    from .committees import Committee, Member, default_committees, load_committees, order_committees
    from .llm import LLMError, load_raw_config

    raw = load_raw_config(a.config, a.preset or "base")
    provider = str((raw.get("llm") or {}).get("provider") or "deepseek")
    if getattr(a, "model", None):
        return default_committees(cfg, provider)
    doc = rs._doc(run_dir / "run_config.json") or {}
    recorded = doc.get("committees")
    if not isinstance(recorded, list) or not recorded:
        return default_committees(cfg, provider)
    try:
        return load_committees({"llm": raw.get("llm") or {}, "committees": recorded}, active=cfg)
    except LLMError:
        if not getattr(a, "dry_run", False):
            raise
        built = []
        for item in recorded:
            if not isinstance(item, dict):
                continue
            members = []
            for member in item.get("members") or []:
                if not isinstance(member, dict):
                    continue
                model = str(member.get("model") or cfg.model)
                members.append(Member(str(member.get("provider") or provider), model, replace(cfg, model=model)))
            built.append(Committee(str(item.get("name") or ""), members))
        return order_committees(built)


# 续跑在发请求之前就该停时用它。code 是进程退出码：目录不能续是 3，锁被占用是 4，设置或预算不合法是 5。
class ResumeRefused(Exception):
    """@brief 续跑在发出第一个请求之前就该停下时的拒绝信号；``code`` 是这条命令的进程退出码。

    退出码约定：目录不能续是 3，锁被占用是 4，设置或预算不合法是 5。``cmd_resume`` 捕获它之后把
    ``code`` 原样当作进程退出码返回。

    字段：``code`` 是进程退出码，构造后不再改变；异常消息是给 stderr 看的失败原因。
    """

    def __init__(self, message: str, code: int) -> None:
        """@brief 记下失败原因与进程退出码。

        @param message 打印到 stderr 的失败原因。
        @param code 这条命令的进程退出码（3/4/5）。
        """
        super().__init__(message)
        self.code = code


# 真正续跑之前的全部检查和装载。失败也不写运行目录，中断时的样子还在。预算由 effective_budget 决定：--budget 是新总上限，--add-budget 加在记下的上限上，都没有则沿用记下的值或 control.json 里尚未生效的请求；新上限不高于已花费就拒绝。
def prepare_resume(a: argparse.Namespace, *, client=None, evaluate_fn=None, log=print):
    """@brief 续跑在第一个模型请求之前的全部检查与装载：确认目录能续、重建状态、组出编排器并把它装进去。

    失败也不写运行目录，中断时的样子还在。预算由 ``resume.effective_budget`` 决定：``--budget`` 是新总上限，
    ``--add-budget`` 加在记下的上限上，都没有则沿用记下的值或 ``control.json`` 里尚未生效的请求。``inspect_run``
    用的是 result/state 里的花费，转录可以更高，所以新上限必须盖过重建后的花费；不高于已花费就拒绝。系统提示
    若已和现在的 ``prompts.py`` 不同，仍用转录里的原文（换提示会让同一次实验的前后轮次对着不同契约说话）。
    ``--dry-run`` 只装内存，``restore`` 在这个开关下不写转录、不改 ``state.json``。

    @param a argparse 命名空间：``run_dir``/``budget``/``add_budget``/``target``/``dry_run``/
        ``lift_count_limits`` 与各配置开关。
    @param client 注入的 LLM 客户端；``None`` 时由编排器自建。
    @param evaluate_fn 注入的评测函数；``None`` 时由编排器自建。
    @param log 进度输出函数，默认 ``print``。
    @return ``(orchestrator, snapshot, report, info, new_budget)``：装好状态的编排器、重建出的快照、
        dry-run 要打印的报告（含预算前后、系统提示来源、settings、provenance、未记录项）、``inspect_run``
        的检查结果，以及这次续跑生效的新预算。
    @raises ResumeRefused 目录不能续（退出码 3）、重建或预算不合法（退出码 5）。
    """
    from . import control, prompts, resume as rs
    from .task_binding import bind_task
    from .llm import LLMError
    from .orchestrator import Orchestrator

    run_dir = Path(a.run_dir).resolve()
    info = rs.inspect_run(
        run_dir,
        allow_count_stops=bool(getattr(a, "lift_count_limits", False)),
        allow_cost_incomplete=bool(getattr(a, "accept_cost_gap", False)),
    )
    if not info["resumable"]:
        code = 4 if info.get("kind") == "resuming" else 3
        raise ResumeRefused(f"cannot resume {run_dir}: {info['reason'] or info['kind']}", code)
    events, _ = rs.read_transcript(run_dir)
    settings = rs.infer_settings(run_dir, events)
    doc = rs._doc(run_dir / "result.json") or rs._doc(run_dir / "state.json") or {}
    info["model"] = doc.get("model")
    try:
        cfg, search, prov = _resume_config(a, settings, info)
        new_budget = rs.effective_budget(info, add_usd=a.add_budget, budget_usd=a.budget, control_doc=control.read_json(control.control_path(run_dir)))
        search.budget_usd = new_budget
        # ``specimen=True`` 只来自 v1 机制测试的 ``tests.support.resume.ns``：把归档的 v1 specimen
        # 当作这次续跑的输入，生产 CLI 没有这个名字，绑定仍按目录版本分流。
        if getattr(a, "specimen", False):
            from tests.support.v1_tasks import bind_specimen

            binding = bind_specimen(info["task"], info["dtype"], doc.get("size", "paper"))
        else:
            binding = bind_task(info["task"], info["dtype"], doc.get("size", "paper"))
        snap = rs.load_snapshot(run_dir, search, binding.kernel_name, settings=settings)
        # inspect 用的是 result/state 里的花费，转录可以更高。新上限必须盖过重建后的花费。
        if new_budget <= snap.spent_usd + 1e-9:
            raise rs.ResumeError(
                f"budget ${new_budget:.2f} does not exceed what the transcript has already spent (${snap.spent_usd:.4f}): add budget with --add-budget USD or --budget TOTAL"
            )
        first = snap.partial.rnd if snap.partial else snap.next_round
        if first > search.max_rounds and not snap.search_finished:
            raise rs.ResumeError(f"all {search.max_rounds} rounds were used (the run is past round {first - 1}): raise the limit with --max-rounds")
        committees = _committees_for_resume(a, cfg, run_dir)
    except (rs.ResumeError, rs.NeedsInspection, LLMError, control.ControlError, KeyError, ValueError) as e:
        raise ResumeRefused(f"cannot resume {run_dir}: {e}", 5) from e
    from .remote_runtime import RemoteEvaluationConfig
    from .workspace_env import RemoteWorkspaceConfig, WorkspaceError
    try:
        remote = RemoteEvaluationConfig.from_file(a.remote_evaluator_config) if getattr(a, "remote_evaluator_config", None) else None
    except ValueError as exc:
        raise ResumeRefused(f"cannot load remote evaluator config: {exc}", 5) from exc
    try:
        workspace = RemoteWorkspaceConfig.from_file(a.remote_workspace_config) if getattr(a, "remote_workspace_config", None) else None
    except WorkspaceError as exc:
        raise ResumeRefused(f"cannot load remote workspace config: {exc.message}", 5) from exc
    resume_target = _recorded_target(run_dir, getattr(a, "target", None))
    recorded_track = _recorded_track(run_dir, requested=getattr(a, "track", None))
    recorded_track_version = _recorded_track_version(run_dir)
    if remote is not None:
        if remote.target_id != resume_target:
            raise ResumeRefused(f"remote evaluator target {remote.target_id!r} differs from the run target {resume_target!r}", 5)
        if not remote.publishable_site_ids:
            raise ResumeRefused("remote evaluator config has no publishable_site_ids", 5)
        if not remote.publishable_epochs:
            raise ResumeRefused("remote evaluator config has no publishable_epochs", 5)
        if remote.selfcheck_token_file is None:
            raise ResumeRefused("remote evaluator config requires selfcheck_token_file for selfcheck", 5)
    execution_path = run_dir / "execution.json"
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.is_file() else None
    except (OSError, ValueError) as exc:
        raise ResumeRefused(f"cannot resume: execution.json is unreadable ({exc})", 5) from exc
    if execution is not None:
        if not isinstance(execution, dict) or execution.get("evaluation") not in {"local", "remote"}:
            raise ResumeRefused("cannot resume: execution.json has no valid evaluation mode", 5)
        recorded_workspace = execution.get("workspace")
        if recorded_workspace is not None:
            if workspace is None:
                raise ResumeRefused("cannot resume a remote-workspace run without --remote-workspace-config", 5)
            if not isinstance(recorded_workspace, dict):
                raise ResumeRefused("cannot resume: execution.json has a malformed workspace placement", 5)
            resolved = recorded_workspace.get("resolved") or {}
            current = {"harness": workspace.harness, "runtime": workspace.runtime, "root": workspace.root}
            if not isinstance(resolved, dict) or any(resolved.get(key) != value for key, value in current.items()):
                raise ResumeRefused("remote workspace placement differs from the original run", 5)
            recorded_isolation = resolved.get("isolation")
            if recorded_isolation is not None and not isinstance(recorded_isolation, dict):
                raise ResumeRefused("cannot resume: execution.json has a malformed isolation block", 5)
            if recorded_isolation is None:
                recorded_isolation = {}
            isolation = getattr(workspace, "isolation", None)
            current_isolation: dict = {}
            if isolation is not None:
                import hashlib

                from .agent_isolation import default_trusted_paths

                trusted = sorted(
                    str(path)
                    for path in default_trusted_paths(run_dir, repo_root=REPO_ROOT)
                    + tuple(Path(path) for path in isolation.trusted_evaluator_paths)
                )
                current_isolation = {
                    "credential_seam": isolation.credential_seam,
                    "simulated": isolation.simulated,
                    "agent_kind": isolation.agent_kind,
                    "trusted_paths_sha256": hashlib.sha256("\n".join(trusted).encode("utf-8")).hexdigest(),
                }
            for key, value in current_isolation.items():
                if recorded_isolation.get(key) != value:
                    raise ResumeRefused(f"remote workspace isolation {key} differs from the original run", 5)
            if not current_isolation and recorded_isolation:
                raise ResumeRefused("cannot resume without the original isolation config", 5)
        elif workspace is not None:
            raise ResumeRefused("cannot switch a local run to a remote workspace during resume", 5)
        mode = execution["evaluation"]
        if mode == "remote" and execution.get("transport") in {None, "", "legacy-ssh"}:
            raise ResumeRefused(
                "cannot resume: the legacy-ssh evaluator transport is no longer supported; "
                "remote runs must have an explicit gRPC transport",
                5,
            )
        if mode == "remote" and remote is None:
            raise ResumeRefused("cannot resume a remote evaluation without --remote-evaluator-config", 5)
        if mode == "local" and remote is not None:
            raise ResumeRefused("cannot switch a local run to remote evaluation during resume", 5)
        if mode == "remote" and not execution.get("publishable_epochs"):
            raise ResumeRefused("cannot resume a remote run without an explicit publication epoch allowlist", 5)
        if remote is not None:
            if set(execution.get("publishable_site_ids") or []) != set(remote.publishable_site_ids):
                raise ResumeRefused("remote publication site allowlist differs from the original run", 5)
            recorded_epochs = execution.get("publishable_epochs")
            if recorded_epochs is not None and set(recorded_epochs) != set(remote.publishable_epochs or []):
                raise ResumeRefused("remote publication epoch allowlist differs from the original run", 5)
            observed = execution.get("observed") or {}
            if isinstance(observed, dict) and observed:
                for config_name, observed_name in (("expected_site_id", "site_id"), ("expected_worker_build", "worker_build"), ("expected_task_pack_digest", "task_pack_digest")):
                    expected = getattr(remote, config_name)
                    if expected and observed.get(observed_name) != expected:
                        raise ResumeRefused(f"remote evaluator {observed_name} differs from the original observed manifest", 5)
                observed_epoch = observed.get("epoch")
                if remote.publishable_epochs is not None and observed_epoch and observed_epoch not in remote.publishable_epochs:
                    raise ResumeRefused("observed evaluator epoch is not in the resume publication allowlist", 5)
    from .artifacts import NeedsInspection
    from .orchestrator import RunStartError

    try:
        orch = Orchestrator(
            snap.task, snap.dtype, snap.size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn, log=log,
            resuming=True, committees=committees, target_id=_recorded_target(run_dir, getattr(a, "target", None)),
            evaluator_factory=None if remote is None else remote.make_evaluator,
            publishable_site_ids=None if remote is None else remote.publishable_site_ids,
            publishable_epochs=None if remote is None else remote.publishable_epochs,
            workspace_config=workspace,
            binding=binding,
            track_id=recorded_track,
            track_version=recorded_track_version,
        )
    except RunStartError as exc:
        # A damaged ``receipts.json`` or a missing signature key is refused here, not re-raised as a
        # traceback: the run directory was not touched.
        raise ResumeRefused(f"cannot resume {run_dir}: {exc}", 5) from exc
    try:
        bridge_factory = _selfcheck_bridge_for(orch, remote, workspace)
    except ValueError as exc:
        raise ResumeRefused(str(exc), 5) from exc
    if bridge_factory is not None:
        orch.selfcheck_bridge_factory = bridge_factory
    if orch._v2:
        variants = {prompts.system_prompt_v2(orch.binding, use_tools=cfg.use_tools, target_id=orch.target_id)}
    else:
        variants = {prompts.system_prompt(orch.task, use_tools=cfg.use_tools, profiler_evidence=pe, hw_feedback=hw) for pe in (False, True) for hw in (False, True)}
    # 系统提示若已和现在的 prompts.py 不同，仍用转录里的原文。换提示会让同一次实验的前后轮次对着不同契约说话。
    prompt_changed = bool(snap.system) and snap.system not in variants  # False: the prompt files still produce the prompt the run began with
    # dry_run 只装内存。restore 在这个开关下不写转录、不改 state.json。远端上下文的行/result/HMAC 校验
    # 也在这里发生：它损坏时是「不能续」，不是 traceback，也不是一份可以静默忽略的 unknown。
    try:
        orch.restore(snap, dry_run=True)
    except (NeedsInspection, RunStartError) as exc:
        raise ResumeRefused(f"cannot resume {run_dir}: {exc}", 5) from exc
    report = rs.describe(snap)
    report.update(
        dry_run=bool(a.dry_run), kind=info["kind"], budget_before_usd=info["budget_usd"], budget_after_usd=new_budget, remaining_usd=round(new_budget - snap.spent_usd, 6),
        system_prompt=(("the one recorded in the transcript is reused" + (" (the prompt files were edited since the run started; this keeps the run's conditions)" if prompt_changed
                        else " (identical to what the current prompt files produce)")) if snap.system else "rebuilt from the current prompt files (no transcript)"),
        settings={k: getattr(search, k) for k in SEARCH_FIELDS} | {"model": cfg.model, "use_tools": cfg.use_tools, "reasoning_effort": cfg.reasoning_effort},
        provenance=prov, command=f"uv run python -m taic.cli resume {run_dir}" + (f" --add-budget {a.add_budget}" if a.add_budget is not None else "") + (f" --budget {a.budget}" if a.budget is not None else ""),
        unrecorded_settings=unrecorded_settings(prov),
    )
    return orch, snap, report, info, new_budget


def unrecorded_settings(prov: dict) -> list[dict]:
    """@brief 列出这次运行没有记录、只能由预设或默认值补上的键。

    @param prov 配置来源表：键到来源说明（预设/默认/命令行）的字典。
    @return 每项为 ``{"key", "source"}`` 的列表，按 key 排序；只有来源里带 “not recorded by this run” 的才算。
    """
    return [{"key": key, "source": source} for key, source in sorted(prov.items()) if isinstance(source, str) and "not recorded by this run" in source]


def _announce_unrecorded(report: dict, log) -> None:
    """@brief 把被补上的设置逐条说清楚；能续跑的续跑不会被这些补值拦住。

    @param report ``prepare_resume`` 出的报告，读其中的 ``unrecorded_settings``。
    @param log 输出函数，通常是 ``orchestrator.log``。
    """
    rows = report.get("unrecorded_settings") or []
    if not rows:
        return
    log("settings this run did not record, filled from the named source (the resume is not refused):")
    for row in rows:
        log(f"  {row['key']}: {row['source']}")


# 检查通过之后才占目录锁、把旧结果留档、记下 resume 预算事件，然后进入原来的搜索循环。别的进程占着锁就拒绝，避免两套搜索同时改同一个 run。
def run_prepared(a: argparse.Namespace, orch, snap, info: dict, new_budget: float):
    """@brief 接着跑 ``prepare_resume`` 准备好的那次运行：占锁、留档叠结果、记预算事件，再进入搜索循环。

    检查通过之后才占目录锁，把旧结果留档成 ``*.before_topup_<n>.json``，在 ``budget_events.json`` 里记下
    resume 的预算事件，然后进入原来的搜索循环。别的进程占着锁就拒绝，避免两套搜索同时改同一个 run。
    损坏的 result/confirm_report 在移动任何文件之前就被拒绝；启动失败时把已经搬走的旧结果还原回去。

    @param a argparse 命名空间，读 ``run_dir`` 与 ``who``。
    @param orch ``prepare_resume`` 建好的编排器，状态已装在内存里但还没落盘。
    @param snap 重建出的快照，提供已花费金额与起始轮次。
    @param info ``inspect_run`` 的检查结果，写进预算事件。
    @param new_budget 这次续跑生效的新总上限。
    @return ``orch.run()`` 的结果对象。
    @raises ResumeRefused 另一个进程占着目录锁（退出码 4），或旧结果/预算事件不合法（退出码 5）。
    @raises RunStartError 形状校验或内核代理 preflight 失败（此时旧结果已还原）。
    """
    from . import control, resume as rs

    run_dir = Path(a.run_dir).resolve()
    first = snap.partial.rnd if snap.partial else snap.next_round
    import os

    from .orchestrator import RunStartError

    try:
        control.acquire_run_lock(run_dir, who=a.who)
    except control.RunLockError as e:
        raise ResumeRefused(f"cannot resume {run_dir}: {e}", 4) from e
    moved: list[str] = []
    try:
        try:
            orch._guard_shape()
            if orch.agent_runner is None and not orch._client_injected:
                from .kernel_agent import AgentError, preflight

                preflight(orch.committees, Path(os.path.expanduser("~/.pi/agent/models.json")))
        except (RunStartError, AgentError) as exc:
            raise RunStartError(str(exc)) from exc
        try:
            moved = rs.snapshot_results(run_dir)
        except rs.ResumeError as exc:
            # 损坏的 result/confirm_report 在移动任何文件之前就被拒绝；给调用方一个干净的错误。
            raise ResumeRefused(f"cannot resume {run_dir}: {exc}", 5) from exc
        if moved:
            orch.log(f"kept the earlier results as {', '.join(moved)}")
        # The first (dry-run) restore already validated the remote evaluation contexts; this second
        # one reuses them.  If it still fails, the earlier results are put back before the refusal
        # leaves this function, so a refused resume never leaves the run directory rewritten.
        try:
            orch.restore(snap, dry_run=False)
        except (rs.NeedsInspection, RunStartError) as exc:
            raise ResumeRefused(f"cannot resume {run_dir}: {exc}", 5) from exc
        event = control.record_event(run_dir, "resume", info["budget_usd"], new_budget, who=a.who, note=f"{info['kind']} run continued at round {first}", spent_usd=snap.spent_usd)
        if not event.get("persisted"):
            orch.log("budget event was not saved; budget_events.json is not a complete record")
        return asyncio.run(orch.run())
    except (RunStartError, ResumeRefused):
        rs.restore_snapshot(run_dir, moved)
        raise
    finally:
        control.release_run_lock(run_dir)


# `taic resume` 的入口。dry_run 只打印将要续的状态就退出，不拿锁、也不跑搜索。不能续时退出码来自 ResumeRefused；真的跑起来之后，_result_code 决定 0 / 6 / 2。
def cmd_resume(a: argparse.Namespace) -> int:
    """@brief ``taic resume`` 的入口：续跑一个中断或为预算停下的运行目录。

    ``--dry-run`` 只打印将要续的状态就退出，不拿锁、也不跑搜索。不能续时退出码来自 ``ResumeRefused``；
    真的跑起来之后，``_result_code`` 决定 0 / 6 / 2。

    @param a argparse 命名空间：``run_dir``/``dry_run``/``who``/``budget``/``add_budget`` 与各配置开关。
    @return 进程退出码：``--dry-run`` 为 0；被拒时是 ``ResumeRefused.code``（3/4/5）；启动失败为 5；
        跑完为 ``_result_code`` 的 0/6/2。
    """
    from .orchestrator import RunStartError
    from tcenv.children import install_termination_handlers

    install_termination_handlers()
    try:
        orch, snap, report, info, new_budget = prepare_resume(a)
        if a.dry_run:
            print(json.dumps(report, indent=2, default=str))
            return 0
        _announce_unrecorded(report, orch.log)
        res = run_prepared(a, orch, snap, info, new_budget)
    except ResumeRefused as e:
        print(str(e), file=sys.stderr)
        return e.code
    except RunStartError as e:
        print(str(e), file=sys.stderr)
        return 5
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 5
    print(json.dumps({k: getattr(res, k) for k in ("task", "dtype", "model", "best_speedup", "confirmed_speedup", "spent_usd", "budget_usd", "stop_reason", "run_dir")}, indent=2))
    return _result_code(res)


_SEARCH_SWITCHES = (
    ("stop_at_roofline", "stop when the seed is already inside the noise gate of its roofline and no baseline beats that bound"),
    ("abi_guard", "recorded switch; the live session does not stop a branch on a repeated ABI error"),
    ("target_note", "when a baseline is faster than the roofline bound, tell the model and give that baseline's time as the target"),
    ("rich_history", "a failed branch's history line names the failure instead of one generic sentence"),
    ("status_lines", "tell the model the noise floor and the budget, round and patience that remain"),
    ("peer_tips", "put this run's recent peer tips into the next TASK.md; tips never cross runs"),
)


def _add_search_switches(p: argparse.ArgumentParser) -> None:
    """@brief 注册 ``_SEARCH_SWITCHES`` 里的 ``--flag`` / ``--no-flag``，外加 ``--truncation-retry``。

    不写就是配置里的默认值，这些开关的默认都是开。``--truncation-retry`` 只被记录：现在的 hmz 会话在回复被
    截断后不会再问一次（默认开）。

    @param p 要挂参数的子命令 parser。
    """
    for dest, help_text in _SEARCH_SWITCHES:
        p.add_argument("--" + dest.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None, help=help_text + " (default: on)")
    p.add_argument("--truncation-retry", action=argparse.BooleanOptionalAction, default=None, help="recorded switch; the live hmz session does not ask again after a cut-off reply (default: on)")


def _u64(text: str) -> int:
    """@brief 把十进制文本解析成无符号 64 位种子。

    @param text 命令行给的十进制文本。
    @return 满足 ``0 <= n < 2**64`` 的整数。
    @raises argparse.ArgumentTypeError 不是十进制整数，或超出 ``0 <= seed < 2**64``。
    """
    try:
        n = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a decimal integer") from None
    if n < 0 or n >= 2**64:
        raise argparse.ArgumentTypeError("must satisfy 0 <= seed < 2**64")
    return n


def _positive_int(text: str) -> int:
    """@brief 解析至少为 1 的整数，给 argparse 的 ``type=`` 用。

    @param text 命令行给的文本。
    @return 大于等于 1 的整数。
    @raises argparse.ArgumentTypeError 不是整数，或小于 1。
    """
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _add_role_efforts(p: argparse.ArgumentParser) -> None:
    """@brief 注册按角色的 effort、新的搜索开关与单请求 token 上限。

    只被记下、现在的 hmz 会话不读的项：``--plan-effort`` 与 ``--patch-effort``（会话用 ``reasoning_effort``）、
    ``--truncation-retry-effort``（会话不重问被截断的回复）、``--plan-ledger``（会话不读 plan ledger）、
    ``--diversity-archive``（会话总是改当前种子）、``--max-tokens``（会话的上限由 ``relay_budget_usd`` 折算成输出
    token）。真正影响搜索的：``--fast-bench`` 让搜索计时跳过串行 dispatch 与更慢的基线（默认关）；
    ``--experience-index`` 指定把实测提示拷进会话的 JSON 文件。

    @param p 要挂参数的子命令 parser。
    """
    p.add_argument("--plan-effort", default=None, choices=["off", "low", "high", "max"], help="recorded only; the hmz session uses reasoning_effort")
    p.add_argument("--patch-effort", default=None, choices=["off", "low", "high", "max"], help="recorded only; the hmz session uses reasoning_effort")
    p.add_argument("--truncation-retry-effort", default=None, choices=["low", "off", "same"], help="recorded only; the hmz session does not retry a cut-off reply")
    p.add_argument("--plan-ledger", action=argparse.BooleanOptionalAction, default=None, help="recorded only; the hmz session does not read a plan ledger")
    p.add_argument("--diversity-archive", action=argparse.BooleanOptionalAction, default=None, help="recorded only; the hmz session always edits the current seed")
    p.add_argument("--fast-bench", action=argparse.BooleanOptionalAction, default=None, help="search timing skips serial dispatch and slower baselines (default: off)")
    p.add_argument("--experience-index", default=None, help="JSON file of measured hints copied into the session")
    p.add_argument("--max-tokens", type=_positive_int, default=None, help="recorded only; the hmz session cap is relay_budget_usd turned into output tokens")


def _add_output_switches(p: argparse.ArgumentParser) -> None:
    """@brief 注册默认关闭的产出开关。不写就保持 SearchConfig 的默认。

    ``--task-ledger`` 把方向账本写进有种子的任务书。``--sweep-tool`` 写上 ``sweep-v2``。
    ``--plateau-after 0`` 是显式关闭，和没写一样不提示。

    @param p 要挂参数的子命令 parser。
    """
    p.add_argument("--task-ledger", action=argparse.BooleanOptionalAction, default=None, help="put the direction ledger in the next seeded TASK.md (default: off)")
    p.add_argument("--sweep-tool", action=argparse.BooleanOptionalAction, default=None, help="name sweep-v2 in TASK.md (default: off)")
    p.add_argument("--plateau-after", type=int, default=None, help="after this many rounds without a new seed, ask for another algorithm family (default: 0, off)")


def _apply_output_switches(a: argparse.Namespace, search, prov: dict | None = None) -> None:
    """@brief 把显式给出的产出开关写进搜索配置。没写的保持原值，``0`` 也算写了。

    @param a argparse 命名空间。缺这些属性时不动配置，旧的测试命名空间因此不用补字段。
    @param search 被改写的 ``SearchConfig``。
    @param prov 配置来源表；给了就记一条来自命令行。
    """
    plateau = getattr(a, "plateau_after", None)
    if plateau is not None:
        search.plateau_after = plateau
        if prov is not None:
            prov["search.plateau_after"] = "command line"
    for key in ("task_ledger", "sweep_tool"):
        value = getattr(a, key, None)
        if value is None:
            continue
        setattr(search, key, bool(value))
        if prov is not None:
            prov[f"search.{key}"] = "command line"


def _apply_max_tokens(a: argparse.Namespace, cfg, prov: dict | None = None) -> None:
    """@brief 把 ``--max-tokens`` 写进 ``cfg.max_tokens``；没给就不动。

    @param a argparse 命名空间，读 ``max_tokens``。
    @param cfg 被改写的 ``LLMConfig``。
    @param prov 配置来源表；给了就记一条 ``llm.max_tokens`` 来自命令行。
    """
    n = getattr(a, "max_tokens", None)
    if n is None:
        return
    cfg.max_tokens = n
    if prov is not None:
        prov["llm.max_tokens"] = "command line"


def _add_harness_switches(p: argparse.ArgumentParser) -> None:
    """@brief 注册 harness 的三个开关。

    ``--harness-auto`` 允许结构化 harness 提案落进本次运行的 overlay（默认关）、``--harness-shadow`` 把没复核的
    quorum 条目搁置而不注入（默认开）、``--harness-review`` 让另一个委员会模型给出 advisory 复核（默认关）。

    @param p 要挂参数的子命令 parser。
    """
    p.add_argument("--harness-auto", action=argparse.BooleanOptionalAction, default=None, help="allow structured harness proposals to settle into this run's overlay (default: off)")
    p.add_argument("--harness-shadow", action=argparse.BooleanOptionalAction, default=None, help="hold unreviewed quorum entries without injecting them (default: on)")
    p.add_argument("--harness-review", action=argparse.BooleanOptionalAction, default=None, help="let the other committee model give an advisory review of a single-source tip (default: off)")


def _apply_search_switches(a: argparse.Namespace, cfg, search, prov: dict | None = None) -> None:
    """@brief 应用显式给出的 ``--flag`` / ``--no-flag``；没给的开关保持配置默认。

    @param a argparse 命名空间，读 harness 三开关、``_SEARCH_SWITCHES`` 与 ``truncation_retry``。
    @param cfg 被改写的 ``LLMConfig``，只改其中的 ``truncation_retry``。
    @param search 被改写的 ``SearchConfig``。
    @param prov 配置来源表；给了就为每个被覆盖的键记一条 “command line”。
    """
    for dest, key in (("harness_auto", "harness_auto"), ("harness_shadow", "harness_shadow"), ("harness_review", "harness_review")):
        value = getattr(a, dest, None)
        if value is None:
            continue
        setattr(search, key, value)
        if prov is not None:
            prov[f"search.{key}"] = "command line"
    for dest, _help in _SEARCH_SWITCHES:
        value = getattr(a, dest, None)
        if value is None:
            continue
        setattr(search, dest, value)
        if prov is not None:
            prov[f"search.{dest}"] = "command line"
    value = getattr(a, "truncation_retry", None)
    if value is None:
        return
    cfg.truncation_retry = value
    if prov is not None:
        prov["llm.truncation_retry"] = "command line"


def cmd_brief(a: argparse.Namespace) -> int:
    """@brief ``taic brief``：没有目录时打印仓库方位，有目录时打印一份短 JSON。

    只读：只读文件，不发模型请求、不上 GPU、不启动任何东西。``--kernels`` 时改为列出已确认的内核（``--target``
    可选筛一个档）；只有 ``--gpu`` 才额外带上机器级 GPU 锁的尾巴（会随墙上时钟变化，默认关）。

    @param a argparse 命名空间，含 ``run_dir``/``kernels``/``target``/``gpu``。
    @return ``run_brief`` 的退出码；``--kernels`` 分支恒为 0。
    """
    from .brief import run_brief
    from .kernel_catalog import list_kernels

    if getattr(a, "kernels", False):
        rows = list_kernels(
            getattr(a, "target", None), limit=getattr(a, "limit", None), cursor=getattr(a, "cursor", None),
        )
        print(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
        return 0
    return run_brief(
        a.run_dir, include_gpu=bool(getattr(a, "gpu", False)), expand=getattr(a, "expand", "") or "",
    )


def cmd_query(a: argparse.Namespace) -> int:
    """@brief ``taic query``：对 observe 索引做查询，不发模型请求、不上 GPU。

    但 ``best``/``failures``/``sessions``/``models`` 会先刷新派生索引，``patrol`` 会刷新目录史投影
    （``kernel_history``，不是 ``tick()``），所以它不是严格的只读进程（``capabilities``、``suggestions``、
    ``suggestion`` 则直接读）。``sessions`` 需要 ``--run``，``suggestion`` 需要 ``--run`` + ``--seq`` +
    ``--digest``，缺了只报错。``--jsonl`` 且结果里有行列表时，逐行一个 JSON 对象输出。

    @param a argparse 命名空间：``what`` 选用哪个查询，其余是 ``task``/``target``/``model``/``status``/
        ``limit``/``sort``/``cursor`` 等筛选项。
    @return 成功为 0；``sessions``/``suggestion`` 缺必需参数或查不到那条建议为 5。
    """
    from . import retrieval

    what = a.what
    limit = a.limit
    if what == "capabilities":
        data = retrieval.capabilities()
    elif what == "patrol":
        # §6.4：一次调用给一屏判定；不摄取索引、不解码 row_json。
        data = retrieval.patrol()
    elif what == "runs":
        # L0 是默认：4 个窄字段（run_id/publication/published_speedup/stop_code）。``--wide`` 才给页面那 22 列（旧行为）。
        data = retrieval.runs(
            task=a.task, dtype=a.dtype, model=a.model, target=a.target,
            publication=a.publication, status=a.status,
            min_speedup=a.min_speedup, sort=a.sort, cursor=a.cursor,
            limit=20 if limit is None else limit, compact=not a.wide,
        )
    elif what == "best":
        data = retrieval.best(task=a.task, target=a.target, limit=20 if limit is None else limit)
    elif what == "failures":
        data = retrieval.failures(
            status=a.status, task=a.task, sig=getattr(a, "sig", None),
            run=getattr(a, "run", None), label=getattr(a, "label", None),
            limit=20 if limit is None else limit,
        )
    elif what == "sessions":
        if not a.run:
            print("query sessions needs --run RUN_ID", file=sys.stderr)
            return 5
        data = retrieval.sessions(a.run, limit=a.limit)
    elif what == "suggestions":
        data = retrieval.suggestions(
            task=a.task, target=a.target, model=a.model, status=a.status,
            limit=a.limit, preview=a.preview, offset=a.offset,
        )
    elif what == "suggestion":
        if not a.run or a.seq is None or not a.digest:
            print("query suggestion needs --run RUN_ID --seq N --digest DIGEST", file=sys.stderr)
            return 5
        try:
            data = retrieval.suggestion(a.run, a.seq, a.digest)
        except (FileNotFoundError, KeyError) as exc:
            print(f"suggestion not found: {exc}", file=sys.stderr)
            return 5
    else:
        data = retrieval.models(limit=50 if limit is None else limit)
    if a.jsonl and isinstance(data, dict):
        rows = data.get("runs") or data.get("best") or data.get("failures") or data.get("sessions") or data.get("models") or data.get("groups")
        if isinstance(rows, list):
            for row in rows:
                print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            return 0
    print(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return 0


def cmd_gc(a: argparse.Namespace) -> int:
    """@brief ``taic gc``：报告或删除已归档的旧运行目录，默认只报告（dry-run）。

    只收拾归档目录，两道闸缺一不可：``--root`` 的目录名必须是 ``deleted``，``--older-than-days`` 至少为 1。
    没有这两道闸时 ``--root . --older-than-days 0 --apply`` 会把仓库的 runs/ kernels/ docs/ 一起 rmtree。
    ``--apply`` 还要求显式 ``--confirm``，免得手滑；真删除时把审计行写进 ``<root>/gc_log.jsonl``。

    @param a argparse 命名空间：``root``/``older_than_days``/``apply``/``confirm``。
    @return 成功（含 dry-run 与没有可清目录）为 0；三道闸任一道不满足为 5。
    """
    from . import gc

    root = Path(a.root).resolve()
    # 只收拾归档目录。没有这两道闸时 `--root . --older-than-days 0 --apply` 会把仓库的
    # runs/ kernels/ docs/ 一起 rmtree；--apply 再要求显式 --confirm，免得手滑。
    if root.name != "deleted":
        print(f"refusing gc: --root must be an archive 'deleted' directory, got {root}", file=sys.stderr)
        return 5
    if float(a.older_than_days) < 1.0:
        print("refusing gc: --older-than-days must be at least 1 (0 would delete every archived run)", file=sys.stderr)
        return 5
    if a.apply and not getattr(a, "confirm", False):
        print("refusing gc --apply without --confirm (run it without --apply to see the list first)", file=sys.stderr)
        return 5
    rows = gc.scan(root, older_than_days=a.older_than_days)
    if not rows:
        print(json.dumps({"root": str(root), "rows": [], "mode": "apply" if a.apply else "dry-run"}, ensure_ascii=False))
        return 0
    if a.apply:
        result = gc.apply(rows, log_path=root / "gc_log.jsonl", root=root)
        print(json.dumps({"root": str(root), "mode": "apply", **result, "rows": len(rows)}, ensure_ascii=False))
    else:
        print(json.dumps({
            "root": str(root), "mode": "dry-run", "older_than_days": a.older_than_days,
            "eligible": sum(1 for row in rows if row["eligible"]),
            "eligible_bytes": sum(row["bytes"] for row in rows if row["eligible"]),
            "rows": rows,
        }, ensure_ascii=False))
    return 0


def cmd_evidence_export(a: argparse.Namespace) -> int:
    """@brief ``taic evidence export RUN_DIR --out x.tar.gz``：两遍流式导出证据包。

    @param a argparse 命名空间：``run_dir``/``out``/``format``。
    @return 成功为 0；源目录不存在或写包失败为 5。
    """
    from . import evidence

    out = Path(a.out)
    try:
        manifest = evidence.export(Path(a.run_dir), out, fmt=a.format)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"evidence export failed: {exc}", file=sys.stderr)
        return 5
    print(json.dumps({
        "out": str(out), "format": a.format, "run_id": manifest["run_id"],
        "files": len(manifest["files"]), "excluded": len(manifest["excluded"]),
        "missing": len(manifest["missing"]), "truncated": len(manifest["truncated"]),
        "bytes": sum(int(row["bytes"]) for row in manifest["files"]),
        "inventory_sha256": manifest["inventory_sha256"],
    }, ensure_ascii=False))
    return 0


def cmd_evidence_verify(a: argparse.Namespace) -> int:
    """@brief ``taic evidence verify x.tar.gz [--against RUN_DIR]``：纯标准库校验。

    @param a argparse 命名空间：``package``/``against``。
    @return 校验通过为 0；任一断言失败为 5（失败项在 JSON 里点名）。
    """
    from . import evidence

    result = evidence.verify(Path(a.package), against=Path(a.against) if a.against else None)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 5


def cmd_blobs(a: argparse.Namespace) -> int:
    """@brief ``taic blobs``：报告内容寻址 blob 的保留情况；``--prune`` 默认 dry-run。

    blob 是持久证据（不能从 runs/kernels 重建），所以默认不删；``--apply`` 必须同时给 ``--prune``
    与 ``--confirm``。删除依据是 blob 文件自身 mtime，不是 ``blob_refs.created_at``。

    @param a argparse 命名空间：``root``/``prune``/``days``/``apply``/``confirm``/``json``。
    @return 成功为 0；``--apply`` 缺少 ``--prune`` 或 ``--confirm`` 时为 5。
    """
    from . import blobs

    root = Path(a.root).resolve() if a.root else blobs.blob_root()
    days = float(a.days) if a.days is not None else blobs.retention_days()
    rows = blobs.scan(root, older_than_days=days)
    eligible = [row for row in rows if row["eligible"]]
    if a.apply and not a.prune:
        print("refusing blobs --apply without --prune (nothing says which blobs to remove)", file=sys.stderr)
        return 5
    if a.apply and not getattr(a, "confirm", False):
        print("refusing blobs --apply without --confirm (run it with --prune to see the list first)", file=sys.stderr)
        return 5
    if a.apply:
        result = blobs.apply(rows)
        print(json.dumps({"root": str(root), "mode": "apply", **result, "total": len(rows)}, ensure_ascii=False))
        return 0
    document = {
        "root": str(root),
        "mode": "dry-run" if a.prune else "report",
        "older_than_days": days,
        "total": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "eligible": len(eligible),
        "eligible_bytes": sum(row["bytes"] for row in eligible),
    }
    if a.prune or a.json:
        document["rows"] = rows if a.json else eligible
    print(json.dumps(document, ensure_ascii=False))
    return 0


def cmd_trust(a: argparse.Namespace) -> int:
    """@brief ``taic trust``：按目标管理模型信任授权，并查看审计与指标。

    支持的表面：``grant`` 写一条授权，``revoke``/``suspend`` 把状态改成 revoked/suspended（操作人与原因一并记下），
    ``audit`` 打印审计记录，``metrics`` 先刷新 observe 索引再统计该目标/任务的信任指标，其余动作（含 ``status``）
    打印现有授权。不发模型请求、不上 GPU；``--root`` 可指向别的 targets 根（离线测试用）。

    @param a argparse 命名空间：``trust_action`` 与对应的 ``target``/``task``/``provider``/``model``/
        ``effort``/``scope``/``by``/``reason``/``root``/``evidence_run``/``expires_at`` 等。
    @return 成功为 0；``trust.TrustError``、``OSError``、``ValueError`` 为 5。
    """
    from . import trust

    action = a.trust_action
    root = Path(a.root) if getattr(a, "root", None) else None
    try:
        if action == "grant":
            entry = trust.grant(
                a.target, a.task, a.provider, a.model, effort=a.effort or "",
                scopes=a.scope or ["overlay:tip"], granted_by=a.by, reason=a.reason,
                evidence_run=a.evidence_run or "", expires_at=a.expires_at,
                prompt_sha1=a.prompt_sha1, protocol_version=a.protocol_version, root=root,
            )
            print(json.dumps(entry, ensure_ascii=False, indent=2))
        elif action in ("revoke", "suspend"):
            entry = trust.set_status(
                a.target, task=a.task, provider=a.provider, model=a.model, effort=a.effort or "",
                status="revoked" if action == "revoke" else "suspended", by=a.by, reason=a.reason, root=root,
            )
            print(json.dumps(entry, ensure_ascii=False, indent=2))
        elif action == "audit":
            print(json.dumps(trust.read_audit(a.target, root), ensure_ascii=False, indent=2))
        elif action == "metrics":
            from observe.index import get_index
            from observe.trust import metrics

            index = get_index()
            index.sync_known()
            rows = metrics(index.conn, target_id=a.target, task=a.task)
            print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        else:
            print(json.dumps(trust.load_grants(a.target, root), ensure_ascii=False, indent=2))
    except (trust.TrustError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 5
    return 0


def cmd_harness(a: argparse.Namespace) -> int:
    """@brief ``taic harness``：查看、复核或落地结构化 harness 提案。

    动作：``status``/``disable``/``settle``/``convert``/``patch``/``commit``/``apply``（只动运行目录里的 harness
    数据），以及会真的调模型的 ``implement``（写代码补丁）与 ``review``（复核提案），和 ``apply-code``。人工闸：
    ``convert`` 要 ``--confirm``；``apply``/``apply-code`` 要 ``--confirm`` 或先 ``--check``；``implement`` 要求提案
    已被复核为有用；``review`` 拒绝提案作者本人来复核，并优先挑没有冲突的模型。没指定 provider/model 时从
    ``run_config.json`` 里提案委员会以外的同伴挑，推不出来就要求显式给。

    @param a argparse 命名空间：``harness_action``、``run_dir``、``key``、``target``/``task``/``dtype``、
        ``provider``/``model``、``by``/``reason``、``budget_usd``、``confirm``/``check``/``run-tests`` 等。
    @return 成功为 0；找不到有效提案、缺人工确认、实现者/复核者不在配置的委员会里，或
        ``harness.ProposalError``/``OSError``/``ValueError``/``LLMError`` 为 5。
    """
    from . import harness
    from .llm import LLMError

    run = Path(a.run_dir)
    try:
        if a.harness_action == "adoption":
            from .adoption import append_adoption_record
            print(json.dumps(append_adoption_record(run), ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "shortlist":
            from .adoption import shortlist
            print(json.dumps(shortlist([Path(path) for path in a.runs], limit=a.limit), ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "execute-experiment":
            from experiments.harness_runner import dry_run, execute_plan
            from .harness_experiment import read_plan
            plan = read_plan(json.loads(Path(a.plan).read_text(encoding="utf-8")))
            out = Path(a.out) if a.out else Path(a.plan).resolve().parent / plan["experiment_id"]
            if not a.execute and not out.is_absolute():
                out = Path(__file__).resolve().parents[1] / out
            if a.execute:
                result = execute_plan(plan, out_dir=out, repo=Path(__file__).resolve().parents[1], allow_spend=a.allow_spend, allow_gpu=a.allow_gpu, approved_by=a.by)
            else:
                result = dry_run(plan, out_dir=out, repo=Path(__file__).resolve().parents[1])
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "plan-experiment":
            from .acceptance import make_spec
            from .code_agent import PATCH_DIR as CODE_PATCH_DIR
            from .harness_experiment import ExperimentError, make_plan

            proposals = [row for row in harness.board(run)["proposals"] if row.get("status") == "valid" and row.get("proposal_key") == a.key]
            if not proposals:
                raise ExperimentError("no valid proposal with that key")
            proposal = proposals[-1]
            reviews = [row for row in harness.board(run)["reviews"] if row.get("proposal_key") == a.key and row.get("event") == "review"]
            spec = make_spec(proposal, reviews[-1] if reviews else None)
            patch = run / CODE_PATCH_DIR / f"{a.key}.patch"
            if not patch.is_file():
                raise ExperimentError("an experiment needs a recorded code patch")
            meta_path = run / CODE_PATCH_DIR / f"{a.key}.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            if not meta.get("implementation", {}).get("accepted") or meta["implementation"].get("spec_sha256") != spec["spec_sha256"] or not meta["implementation"].get("host_verified"):
                raise ExperimentError("an experiment needs host-verified implementation acceptance for the same spec")
            import hashlib
            run_cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8")) if (run / "run_config.json").is_file() else {}
            from .harness_experiment import config_template, digest as experiment_digest
            base_commit = a.base_commit or str(meta.get("base_commit") or "")
            if not base_commit:
                raise ExperimentError("experiment planning needs --base-commit or a recorded implementation base")
            if not run_cfg.get("price_snapshot"):
                raw_cfg = __import__("taic.llm", fromlist=["load_raw_config"]).load_raw_config(None, "base")
                committees = __import__("taic.committees", fromlist=["load_committees"]).load_committees(raw_cfg, active=__import__("taic.llm", fromlist=["load_config"]).load_config(None, preset="base")[0])
                run_cfg["price_snapshot"] = {member.model: {"input": member.cfg.price_in, "output": member.cfg.price_out, "cacheRead": member.cfg.price_cache_read, "cacheWrite": member.cfg.price_cache_write} for committee in committees for member in committee.members}
            canonical_config = experiment_digest(config_template(run_cfg, budget_usd=a.budget_usd, base_commit=base_commit))
            from tcenv.schema import PROTOCOL_VERSION
            if not run_cfg:
                raise ExperimentError("an experiment needs the original run_config.json")
            if spec["kind"] == "hardware_premise":
                raise ExperimentError("hardware premise evidence stays in the probe/premise path; A/B plans are for harness changes")
            from .harness_experiment import config_template
            plan = make_plan(spec, change_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(), protocol_version=PROTOCOL_VERSION, config_sha256=canonical_config, seeds=a.seed, budget_usd=a.budget_usd, repetitions=a.repetitions, config_template_document=config_template(run_cfg, budget_usd=a.budget_usd, base_commit=base_commit), rules=__import__("taic.harness_experiment", fromlist=["evaluation_rules"]).evaluation_rules(min_pairs=a.min_pairs, min_effect=a.min_effect, direction=a.direction), base_commit=base_commit)
            from shutil import copyfile
            out = Path(a.out) if a.out else run / "harness_experiments" / f"{plan['experiment_id']}.json"
            if not out.parent.exists():
                out.parent.mkdir(parents=True)
            if plan["change_sha256"]:
                artifact_dir = out.parent / plan["experiment_id"]
                artifact_dir.mkdir(parents=True, exist_ok=True)
                copyfile(patch, artifact_dir / "treatment.patch")
            if a.allow_spend or a.allow_gpu:
                from .harness_experiment import authorize_plan
                plan = authorize_plan(plan, allow_model_spend=a.allow_spend, allow_gpu=a.allow_gpu, approved_by=a.by)
            out = Path(a.out) if a.out else run / "harness_experiments" / f"{plan['experiment_id']}.json"
            from tcenv.schema import write_json_atomic
            write_json_atomic(out, plan)
            print(json.dumps({"plan": str(out), "experiment": plan, "executed": False}, ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "collect-experiment":
            from .harness_experiment import collect_arm, read_plan, summarize_result
            from tcenv.schema import write_json_atomic
            plan = read_plan(json.loads(Path(a.plan).read_text(encoding="utf-8")))
            result = {"schema_version": 1, "plan": plan, "control": collect_arm(plan, Path(a.control), "control"), "treatment": collect_arm(plan, Path(a.treatment), "treatment")}
            summary = summarize_result(result)
            out = Path(a.out) if a.out else run / "harness_experiments" / f"{summary['experiment_id']}.result.json"
            write_json_atomic(out, result)
            print(json.dumps({"result": str(out), "summary": summary, "effect_validated": False}, ensure_ascii=False, indent=2))
        elif a.harness_action == "validate-effect":
            from .harness_experiment import record_effect_validation, summarize_result
            from tcenv.schema import write_json_atomic
            result = json.loads(Path(a.result).read_text(encoding="utf-8"))
            summary = summarize_result(result)
            evidence_path = run / "harness_experiments" / f"{summary['experiment_id']}.result.json"
            if evidence_path.exists() and json.loads(evidence_path.read_text(encoding="utf-8")) != result:
                raise ValueError("this experiment id already has different result evidence")
            write_json_atomic(evidence_path, result)
            verdict = record_effect_validation(run, result, validated=a.valid, approved_by=a.by, reason=a.reason, evidence_path=evidence_path)
            print(json.dumps(verdict, ensure_ascii=False, indent=2))
        elif a.harness_action == "status":
            print(json.dumps(harness.board(run), ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "disable":
            print(json.dumps(harness.set_overlay_control(run, disabled=True, by=a.by, reason=a.reason), ensure_ascii=False))
        elif a.harness_action == "settle":
            from . import trust

            result = harness.settle(
                run, target_id=a.target, task=a.task,
                protocol_version=a.protocol_version or trust.current_protocol_version(),
                enabled=True, shadow=not a.active,
            )
            print(json.dumps({key: len(value) for key, value in result.items() if isinstance(value, list)} | {"result": result}, ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "patch":
            print(json.dumps(harness.write_patch(run, target_id=a.target, proposal_key=a.key), ensure_ascii=False, indent=2))
        elif a.harness_action == "implement":
            from . import code_agent
            from .llm import load_config, load_raw_config
            from .committees import load_committees
            from .kernel_agent import Prices

            proposals = [row for row in harness.board(run)["proposals"] if row.get("status") == "valid" and row.get("proposal_key") == a.key]
            if not proposals:
                print("no valid proposal with that key", file=sys.stderr)
                return 5
            proposal = proposals[0]
            reviews = [row for row in harness.board(run)["reviews"] if row.get("proposal_key") == a.key and row.get("useful") is True]
            if not reviews:
                print("refusing to implement an unreviewed proposal; run `harness review` first", file=sys.stderr)
                return 5
            from .acceptance import make_spec
            spec = make_spec(proposal, reviews[-1])
            run_cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8")) if (run / "run_config.json").is_file() else {}
            recorded = run_cfg.get("committees") if isinstance(run_cfg, dict) else None
            provider, model = a.provider, a.model
            if not provider or not model:
                proposal_committee = harness.proposal_committee(proposal)
                peers = [c for c in (recorded or []) if isinstance(c, dict) and c.get("name") != proposal_committee]
                members = (peers[0].get("members") if peers else None) or []
                if not members or not isinstance(members[0], dict):
                    print("cannot choose an implementer; pass --provider and --model", file=sys.stderr)
                    return 5
                provider, model = str(members[0].get("provider") or ""), str(members[0].get("model") or "")
            raw = load_raw_config(None, "base")
            cfg, _ = load_config(None, preset="base")
            committees = load_committees(raw, active=cfg)
            member = next((m for c in committees for m in c.members if m.provider == provider and m.model == model), None)
            if member is None:
                print(f"implementer {provider}/{model} is not in the configured committees", file=sys.stderr)
                return 5
            prices = Prices(member.cfg.price_in, member.cfg.price_out, member.cfg.price_cache_read, member.cfg.price_cache_write)
            target_id = str((proposal.get("proposal") or {}).get("scope", {}).get("target_id") or a.target)
            prompt = code_agent.build_prompt(proposal.get("proposal") or {}, run_id=str(proposal.get("run_id") or run.name), acceptance_spec=spec)
            plan = code_agent.estimate_code_budget(prices, prompt)
            budget = float(a.budget_usd) if a.budget_usd is not None else plan["budget_usd"]
            print(json.dumps({"implementer": {"provider": provider, "model": model, "effort": member.cfg.reasoning_effort}, "plan": plan, "budget_usd": budget}, ensure_ascii=False), file=sys.stderr)
            worktree = code_agent.prepare_worktree(Path(__file__).resolve().parents[1], run, str(a.key))
            result = code_agent.run_code_agent(
                worktree, prompt=prompt, provider=provider, model=model,
                effort=member.cfg.reasoning_effort, prices=prices, budget_usd=budget,
            )
            meta = code_agent.write_patch(run, str(a.key), result, target_id=target_id, acceptance_spec=spec)
            print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "review-code":
            from .code_agent import review_code_patch

            print(json.dumps(review_code_patch(run, a.key, by=a.by), ensure_ascii=False, indent=2))
        elif a.harness_action in {"verify-code", "verify-legacy"}:
            from .code_agent import DockerPatchTestRunner, verify_code_patch
            if a.harness_action == "verify-legacy" and not a.legacy_text:
                raise ValueError("verify-legacy needs --legacy-text")
            runner = DockerPatchTestRunner(a.docker_image, docker_binary=a.docker_binary) if a.docker_image else None
            print(json.dumps(verify_code_patch(run, a.key, approved_by=a.by, legacy_text=a.legacy_text, check_ids=a.check_id, static=a.static, execute=a.execute, runner=runner), ensure_ascii=False, indent=2))
        elif a.harness_action == "apply-code":
            from . import code_agent

            if not a.confirm and not a.check:
                print("refusing to apply a code patch without --confirm (use --check first)", file=sys.stderr)
                return 5
            print(json.dumps(code_agent.apply_code_patch(
                run, proposal_key=a.key, check_only=a.check, run_tests=a.run_tests, approved_by=a.by,
                require_effect_validation=not a.legacy_unvalidated,
            ), ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "convert":
            if not a.confirm:
                print("converting free text requires --confirm (the human gate)", file=sys.stderr)
                return 5
            run_cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8")) if (run / "run_config.json").is_file() else {}
            target_id = str(a.target or (run_cfg.get("target") or {}).get("id") or "apple-m4-max")
            task = str(a.task or run_cfg.get("task") or "")
            dtype = str(a.dtype or run_cfg.get("dtype") or "")
            if not task:
                print("cannot infer task; pass --task", file=sys.stderr)
                return 5
            row = harness.convert_feedback(
                run, seq=a.seq, digest=a.digest, target_id=target_id, task=task, dtype=dtype,
                surface=a.surface, action=a.action, probe=a.probe, measurement=a.measurement,
                reproduction=a.reproduction, expected=a.expected,
                rationale=a.rationale, by=a.by, confirm=True, mode=a.mode,
                supersede=a.supersede,
            )
            print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "commit":
            try:
                print(json.dumps(harness.commit_local_promotions(run, message=a.message, by=a.by), ensure_ascii=False, indent=2))
            except harness.ProposalError as exc:
                print(str(exc), file=sys.stderr)
                return 5
        elif a.harness_action == "review":
            from . import review as review_mod
            from . import trust

            proposals = [row for row in harness.board(run)["proposals"] if row.get("status") == "valid" and row.get("proposal_key") == a.key]
            if not proposals:
                print("no valid proposal with that key", file=sys.stderr)
                return 5
            proposal = proposals[0]
            run_cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8")) if (run / "run_config.json").is_file() else {}
            recorded = run_cfg.get("committees") if isinstance(run_cfg, dict) else None
            provider, model = a.provider, a.model
            if not provider or not model:
                proposal_committee = harness.proposal_committee(proposal)
                peers = [c for c in (recorded or []) if isinstance(c, dict) and c.get("name") != proposal_committee]
                members = (peers[0].get("members") if peers else None) or []
                if not members or not isinstance(members[0], dict):
                    from .llm import load_config as _load_config, load_raw_config as _load_raw
                    from .committees import load_committees as _load_committees

                    raw = _load_raw(None, "base")
                    cfg, _ = _load_config(None, preset="base")
                    configured = _load_committees(raw, active=cfg)
                    peers = [c for c in configured if c.name != proposal_committee] or configured
                    members = [{"provider": m.provider, "model": m.model} for c in peers for m in c.members]
                if members and isinstance(members[0], dict):
                    independent = [
                        member for member in members
                        if isinstance(member, dict)
                        and not harness.reviewer_conflicts_with_group(proposals, str(member.get("provider") or ""), str(member.get("model") or ""))
                    ]
                    members = independent or members
                if not members or not isinstance(members[0], dict):
                    print("cannot choose a reviewer; pass --provider and --model", file=sys.stderr)
                    return 5
                provider, model = str(members[0].get("provider") or ""), str(members[0].get("model") or "")
            from .llm import load_config, load_raw_config
            from .committees import load_committees
            from .kernel_agent import Prices

            raw = load_raw_config(None, "base")
            cfg, _ = load_config(None, preset="base")
            committees = load_committees(raw, active=cfg)
            member = next((m for c in committees for m in c.members if m.provider == provider and m.model == model), None)
            if member is None:
                print(f"reviewer {provider}/{model} is not in the configured committees", file=sys.stderr)
                return 5
            if harness.reviewer_conflicts_with_group(proposals, provider, model):
                print(f"reviewer {provider}/{model} is an author of this claim; choose an independent model", file=sys.stderr)
                return 5
            prices = Prices(member.cfg.price_in, member.cfg.price_out, member.cfg.price_cache_read, member.cfg.price_cache_write)
            work = run / "harness_reviews" / str(a.key)
            bundle = review_mod.evidence_bundle(run, proposal)
            prompt = review_mod.build_review_prompt(proposal.get("proposal") or {}, run_id=str(proposal.get("run_id") or run.name), bundle=bundle)
            plan = review_mod.estimate_review_budget(prices, prompt)
            print(json.dumps({"reviewer": {"provider": provider, "model": model, "effort": member.cfg.reasoning_effort}, "plan": plan}, ensure_ascii=False), file=sys.stderr)
            verdict = review_mod.run_review(
                work, prompt=prompt, provider=provider, model=model,
                effort=member.cfg.reasoning_effort, prices=prices,
            )
            review_mod.append_review(run, proposal, verdict)
            print(json.dumps(verdict, ensure_ascii=False, indent=2, default=str))
            print(json.dumps(harness.settle(
                run, target_id=a.target, task=a.task, protocol_version=trust.current_protocol_version(),
                enabled=True, shadow=True,
            ), ensure_ascii=False, indent=2, default=str))
        elif a.harness_action == "apply":
            if not a.confirm and not a.check:
                print("refusing to apply without --confirm (use --check first)", file=sys.stderr)
                return 5
            print(json.dumps(harness.apply_patch(run, target_id=a.target, proposal_key=a.key, check_only=a.check, run_tests=a.run_tests, applied_by=a.by), ensure_ascii=False, indent=2))
        else:
            print("unknown harness action", file=sys.stderr)
            return 5
    except (harness.ProposalError, OSError, ValueError, LLMError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 5
    return 0


def cmd_feedback(a: argparse.Namespace) -> int:
    """@brief 预览或原子地写入一批 JSON 数组形式的 claim 复核。

    ``--dry-run`` 只校验并显示将要做的状态迁移，``--confirm`` 才真的写；两者必须给且只能给一个。

    @param a argparse 命名空间：``run_dir``/``file``（``-`` 读 stdin）/``dry_run``/``confirm``/``method``/
        ``batch_id``。
    @return 成功为 0；缺 ``--dry-run``/``--confirm``、两个都给、文件不是 JSON 数组（含读失败），或
        ``observe.actions.Refused`` 为 5。
    """
    from observe.actions import Refused, review_feedback_batch

    if not a.dry_run and not a.confirm:
        print("refusing to write reviews without --confirm (use --dry-run first)", file=sys.stderr)
        return 5
    if a.dry_run and a.confirm:
        print("pass either --dry-run or --confirm, not both", file=sys.stderr)
        return 5
    try:
        text = sys.stdin.read() if a.file == "-" else Path(a.file).read_text(encoding="utf-8")
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("the file must contain a JSON array")
        result = review_feedback_batch(
            a.run_dir, rows, dry_run=bool(a.dry_run), method=a.method, batch_id=a.batch_id,
        )
    except (OSError, json.JSONDecodeError, ValueError, Refused) as exc:
        print(str(exc), file=sys.stderr)
        return 5
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_migrate_runs(a: argparse.Namespace) -> int:
    """@brief ``taic migrate-runs``：把平铺的运行目录搬进 ``runs/<target>/``。

    @param a argparse 命名空间：``target``（默认 ``apple-m4-max``）与 ``root``（缺省为仓库下的 ``runs``）。
    @return 恒为 0，结果以一行 JSON 打印 ``{"target", "moved"}``。
    """
    from .migrate import migrate_runs

    root = Path(a.root) if a.root else None
    moved = migrate_runs(root, a.target)
    print(json.dumps({"target": a.target, "moved": moved}, ensure_ascii=False))
    return 0


# `run` 开新搜索，`resume` 续旧目录。两边的 --budget 含义不同：run 是这一次内核的总上限，resume 是续跑后的新总上限，必须高于已经花掉的金额。
# `brief` 不接预算：没有 RUN_DIR 时是仓库方位，有目录时是一份短 JSON。
def cmd_portfolio_run(a: argparse.Namespace) -> int:
    """@brief 每个带 benchmark 的 size 搜一颗核，再开组装会话拼成组合。

    ``--dry-run`` 只做启动检查并打印计划，不读凭据，不开会话，不建 ``shards/``。

    @param a 已解析命令行参数。
    @return 启动失败 5；组合没通过 1；通过或 dry-run 为 0。
    """
    from .portfolio_search import run_portfolio_search

    sizes = None
    if a.sizes:
        sizes = tuple(part.strip() for part in str(a.sizes).split(",") if part.strip())
    return run_portfolio_search(
        a.task, a.dtype,
        sizes=sizes, track=a.track, budget=a.budget, assemble_budget=a.assemble_budget,
        assemble_attempts=a.assemble_attempts, max_total_usd=a.max_total_usd,
        run_dir=None if a.run_dir is None else Path(a.run_dir),
        target_id=a.target, seed=a.seed, config=a.config, preset=a.preset, model=a.model,
        dry_run=a.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    """@brief 组出全部 argparse 子命令（run/resume/query/gc/brief/migrate-runs/trust/harness/feedback）并分发到它的 ``fn``。

    ``run`` 开新搜索，``resume`` 续旧目录。两边的 ``--budget`` 含义不同：run 是这一次内核的总上限，resume 是续跑后的
    新总上限，必须高于已经花掉的金额。``brief`` 不接预算：没有 RUN_DIR 时是仓库方位，有目录时是一份短 JSON。新搜索
    的预算与预设不在这里写死：未指定 ``--config`` 时 ``load_config`` 读 ``taic.toml``，再按预设名叠
    ``taic.<preset>.toml``；预设名是 ``--preset``，否则环境变量 ``TAIC_PRESET``，否则 ``DEFAULT_PRESET``（paper，
    论文的 ``$15``，即 ``PAPER_BUDGET_USD``）。``--budget`` 只在显式给出时覆盖 ``search.budget_usd``。续跑的预算另算：
    目录里记下的花费和上限先恢复，``--budget`` 改总上限，``--add-budget`` 加在记下的上限上，两者都没有就沿用记下的上限
    或 ``control.json`` 里还没生效的请求，但新上限必须高于已花费；没记下的键才回到预设，默认 base（只有 ``taic.toml``）。

    @param argv 命令行参数列表；``None`` 时由 argparse 取 ``sys.argv[1:]``。
    @return 所选子命令 ``fn`` 的进程退出码。
    """
    ap = argparse.ArgumentParser(prog="taic", description="Triton AI Compiler for Apple GPUs (Triton -> MSL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="lower one task with the LLM agent")
    p.add_argument("task")
    p.add_argument("--dtype", default="fp16", help="instance dtype declared by the task (default fp16)")
    p.add_argument("--size", default="paper", help="instance size declared by the task (default paper)")
    # 新搜索的三层配置：--config 独占整份文件；否则 --preset（或环境变量 TAIC_PRESET，默认 paper）叠在 taic.toml 上。--budget 再覆盖叠好之后的 budget_usd。
    p.add_argument("--config", default=None, help="one config file, used as it is (no preset is layered on it)")
    p.add_argument("--preset", default=None, help="settings layered over taic.toml: 'paper' (default, taic.paper.toml: the paper's $15 budget) or 'base' (taic.toml alone); also $TAIC_PRESET")
    p.add_argument("--model", default=None)
    p.add_argument("--budget", type=float, default=None, help="USD budget for this kernel (default: the preset's budget_usd, the paper's $15)")
    p.add_argument("--target", default="apple-m4-max", help="device profile id under targets/")
    p.add_argument(
        "--track", default=None, choices=_run_track_choices(),
        help="delivery id among tracks currently registered as available; "
             "default from taic.toml [search].track, else msl-single",
    )
    p.add_argument("--remote-evaluator-config", default=None, help="explicit JSON remote evaluator/selfcheck composition; default is local")
    p.add_argument("--remote-workspace-config", default=None, help="explicit JSON SSH workspace placement for the kernel session; default is the local workspace")
    p.add_argument("--branches", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--eval-reps", type=int, default=None, help="interleaved evaluation repetitions, validated by the existing search config")
    p.add_argument("--correctness-only", action=argparse.BooleanOptionalAction, default=None, help="disable/enable timing and baseline comparison")
    p.add_argument("--provider-retries", type=int, default=None, help="retry a structured provider turn failure within the same committee/candidate (0-3, default 0)")
    p.add_argument("--session-cost-cap-usd", type=float, default=None, help="optional USD cap for one hmz attempt; None keeps the existing total-budget gate")
    p.add_argument("--effort", default=None, choices=["off", "low", "high", "max"])
    _add_role_efforts(p)
    p.add_argument("--no-tools", action="store_true", help="recorded only; the hmz session does not use the chat tool loop")
    p.add_argument("--kernel-agent", default=None, choices=["pi", "off"], help="kernel writing is hmz and pi; off refuses to start")
    p.add_argument("--profile-best", action="store_true", help="xctrace profile of the winning kernel after the search")
    p.add_argument("--profile-poor", action="store_true", help="trace a round's seed with Instruments when it is slower than the best baseline and give the findings to the model")
    p.add_argument("--plan-mix", action="store_true", help="recorded only; the hmz session chooses its own direction")
    p.add_argument("--hw-feedback", action="store_true", help="append a short static hardware-feature analysis of the seed to the planner and patch feedback and teach it as Skill 2c (off by default)")
    _add_search_switches(p)
    _add_harness_switches(p)
    p.add_argument("--init-samples", type=int, default=None, help="seedless drafts, 1 to 3. 1 keeps today's one draft plus one retry (default: 1)")
    _add_output_switches(p)
    p.add_argument("--confirm-batches", type=int, default=None, help="independent confirm evaluations per candidate; the published speedup is their median (default 3, 1 = the old single batch)")
    p.add_argument("--patience", type=int, default=None, help="stop after this many rounds without a new best (default 6)")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--seed", type=_u64, default=None, help="unsigned 64-bit run seed (default: a new random one). resume keeps the seed the run recorded")
    p.add_argument("--from", dest="from_run", default=None, help="directory of a previous run: if its best.metal is still eligible for this task, skip writing a first draft and improve that kernel")
    p.set_defaults(fn=cmd_run)
    explore = sub.add_parser("track-explore", help="independent single-Track runs plus an exploration-only deterministic summary")
    explore.add_argument("--config", required=True, help="versioned track_explore.json (schema taic.track-explore.1)")
    explore.add_argument(
        "--summarize-only", action="store_true",
        help="read existing run evidence only; never start taic run, a model, or a device evaluation",
    )
    explore.set_defaults(fn=cmd_track_explore)
    portfolio = sub.add_parser("portfolio-run", help="search one kernel per benchmark size and assemble an exact-dispatch portfolio")
    portfolio.add_argument("task")
    portfolio.add_argument("--dtype", required=True, choices=["fp16", "bf16"])
    portfolio.add_argument("--sizes", default=None, help="comma-separated benchmark sizes; default is every benchmark size")
    portfolio.add_argument("--track", default="msl-single", help="msl-single or msl-bundle")
    portfolio.add_argument("--budget", type=float, default=None, help="USD cap for each shard search")
    portfolio.add_argument("--assemble-budget", type=float, default=None, help="USD cap for assemble sessions; default is 2x relay_budget_usd")
    portfolio.add_argument("--assemble-attempts", type=int, default=2)
    portfolio.add_argument("--max-total-usd", type=float, default=None, help="reject the start when the nominal cost is above this")
    portfolio.add_argument("--run-dir", default=None)
    portfolio.add_argument("--target", default="apple-m4-max")
    portfolio.add_argument("--seed", type=_u64, default=None)
    portfolio.add_argument("--config", default=None)
    portfolio.add_argument("--preset", default=None)
    portfolio.add_argument("--model", default=None)
    portfolio.add_argument("--dry-run", action="store_true", help="print the plan only; no credentials, sessions, or shards")
    portfolio.set_defaults(fn=cmd_portfolio_run)
    # 续跑不重开搜索。--budget 是新的总上限，--add-budget 加在原上限上；预算已经花满时，至少要给其中一个，否则新上限不会高于已花费。
    r = sub.add_parser("resume", help="continue a run that was interrupted or stopped for budget, from its directory (no money is spent again)")
    r.add_argument("run_dir")
    r.add_argument("--add-budget", type=float, default=None, help="USD to add to the run's budget (a run that stopped for budget needs this or --budget)")
    r.add_argument("--budget", type=float, default=None, help="the new TOTAL USD budget (must exceed what the run has already spent)")
    r.add_argument("--dry-run", action="store_true", help="rebuild the state and print what would be continued; write nothing, call no model, use no GPU")
    r.add_argument("--who", default="cli", choices=["cli", "dashboard", "api"], help="recorded in budget_events.json")
    r.add_argument("--config", default=None)
    r.add_argument("--preset", default=None, help="settings for keys the run did not record (default: base = taic.toml, which is what runs without run_config.json were started with)")
    r.add_argument("--model", default=None)
    r.add_argument("--branches", type=int, default=None)
    r.add_argument("--max-rounds", type=int, default=None)
    r.add_argument("--provider-retries", type=int, default=None, help="override the recorded provider retry count (0-3); omitted keeps the run's value")
    r.add_argument("--session-cost-cap-usd", type=float, default=None, help="override the recorded per-attempt session cost cap")
    r.add_argument("--effort", default=None, choices=["off", "low", "high", "max"])
    _add_role_efforts(r)
    r.add_argument("--no-tools", action="store_true", help="recorded only; the hmz session does not use the chat tool loop")
    r.add_argument("--kernel-agent", default=None, choices=["pi", "off"], help="kernel writing is hmz and pi; off refuses to start")
    r.add_argument("--profile-best", action="store_true")
    r.add_argument("--profile-poor", action="store_true")
    r.add_argument("--plan-mix", action="store_true")
    r.add_argument("--hw-feedback", action="store_true")
    _add_search_switches(r)
    _add_harness_switches(r)
    r.add_argument("--init-samples", type=int, default=None, help="seedless drafts, 1 to 3. Omitted keeps the value this run recorded")
    _add_output_switches(r)
    r.add_argument("--patience", type=int, default=None, help="not recorded by runs older than run_config.json: give the value the run was started with")
    r.add_argument("--lift-count-limits", action="store_true", help="also continue a run that stopped because it used its round limit or its patience")
    r.add_argument("--accept-cost-gap", action="store_true", help="continue a run stopped by cost_incomplete after its hmz epics were reconciled into an unaccounted entry (human decision)")
    r.add_argument("--remote-evaluator-config", default=None, help="explicit JSON remote evaluator/selfcheck composition; must be supplied again for a remote resume")
    r.add_argument(
        "--track", default=None, choices=_resume_track_choices(),
        help="optional cross-check: resume keeps the delivery mechanism the run recorded, and a different value is refused",
    )
    r.add_argument("--remote-workspace-config", default=None, help="explicit JSON SSH workspace placement; must be supplied again for a remote-workspace resume")
    r.set_defaults(fn=cmd_resume)
    q = sub.add_parser("query", help="read-only queries over the observe index (no model call, no GPU)")
    q.add_argument("what", choices=["runs", "best", "failures", "sessions", "models", "capabilities", "suggestions", "suggestion", "patrol"])
    q.add_argument("--sig", default=None, help="failures: filter by failure-cluster signature")
    q.add_argument("--label", default=None, help="failures: filter by candidate label")
    q.add_argument("--task", default=None)
    q.add_argument("--dtype", default=None)
    q.add_argument("--target", default=None)
    q.add_argument("--model", default=None)
    q.add_argument("--publication", default=None,
                   help="runs: confirmed|in_progress|unconfirmed|uncertain|legacy（也接受原始短语）")
    q.add_argument("--status", default=None)
    q.add_argument("--run", default=None)
    q.add_argument("--seq", type=int, default=None)
    q.add_argument("--digest", default=None)
    q.add_argument("--preview", type=int, default=120, help="suggestions: preview characters per digest")
    q.add_argument("--offset", type=int, default=0, help="suggestions: group offset for paging")
    q.add_argument("--min-speedup", type=float, default=None)
    q.add_argument("--sort", default="started", choices=["started", "speedup", "spent"])
    q.add_argument("--cursor", default=None)
    q.add_argument("--limit", type=int, default=None, help="rows per query; runs/failures default to 20, models 50")
    q.add_argument("--wide", action="store_true", help="runs: the 22-column row instead of the L0 4-field row")
    q.add_argument("--jsonl", action="store_true", help="one JSON object per row")
    q.set_defaults(fn=cmd_query)
    g = sub.add_parser("gc", help="report or remove old data/deleted directories (default dry-run)")
    g.add_argument("--root", default="data/deleted")
    g.add_argument("--older-than-days", type=float, default=30.0)
    g.add_argument("--apply", action="store_true", help="delete eligible directories; without it only report")
    g.add_argument("--confirm", action="store_true", help="required with --apply: deletes are irreversible")
    g.set_defaults(fn=cmd_gc)
    bl = sub.add_parser("blobs", help="report or prune content-addressed kernel blobs (default report/dry-run)")
    bl.add_argument("--root", default=None, help="blob root; default data/blobs or TAIC_BLOB_ROOT")
    bl.add_argument("--prune", action="store_true", help="list blobs past the retention line")
    bl.add_argument("--days", type=float, default=None, help="retention days; default 365 or TAIC_BLOB_RETENTION_DAYS")
    bl.add_argument("--apply", action="store_true", help="delete eligible blobs; requires --prune and --confirm")
    bl.add_argument("--confirm", action="store_true", help="required with --apply: deletes are irreversible")
    bl.add_argument("--json", action="store_true", help="include every blob row in the report")
    bl.set_defaults(fn=cmd_blobs)
    ev = sub.add_parser("evidence", help="export or verify a run's evidence package (never calls a model or the GPU)")
    evs = ev.add_subparsers(dest="evidence_action", required=True)
    ex = evs.add_parser("export", help="two-pass streaming export of one run")
    ex.add_argument("run_dir")
    ex.add_argument("--out", required=True)
    ex.add_argument("--format", choices=["tar.gz", "json"], default="tar.gz")
    ex.set_defaults(fn=cmd_evidence_export)
    evv = evs.add_parser("verify", help="verify a package with the standard library only")
    evv.add_argument("package")
    evv.add_argument("--against", default=None, help="recompute the source inventory and compare")
    evv.set_defaults(fn=cmd_evidence_verify)
    b = sub.add_parser("brief", help="read-only orientation, or JSON facts for one run directory (no model call, no GPU)")
    b.add_argument("run_dir", nargs="?", default=None, help="run directory; omit for a repo orientation")
    b.add_argument("--kernels", action="store_true", help="list confirmed kernels; optional --target filters one profile")
    b.add_argument("--target", default=None)
    b.add_argument("--limit", type=int, default=20, help="kernels: how many rows; the L1 brief ignores it")
    b.add_argument("--cursor", default=None, help="kernels: a keyset cursor from the previous page")
    b.add_argument("--expand", default="", help="comma list of heavy sections to add: cands,models,failures")
    b.add_argument("--gpu", action="store_true", help="include the machine-wide GPU lock tail (other runs, changes with wall clock; off by default)")
    b.set_defaults(fn=cmd_brief)
    m = sub.add_parser("migrate-runs", help="move flat run directories into runs/<target>/")
    m.add_argument("--target", default="apple-m4-max")
    m.add_argument("--root", default=None)
    m.set_defaults(fn=cmd_migrate_runs)
    t = sub.add_parser("trust", help="per-target model trust grants and metrics (no model call, no GPU)")
    ts = t.add_subparsers(dest="trust_action", required=True)
    for name in ("grant", "revoke", "suspend"):
        g = ts.add_parser(name)
        g.add_argument("target")
        g.add_argument("task")
        g.add_argument("provider")
        g.add_argument("model")
        g.add_argument("--effort", default="")
        g.add_argument("--scope", action="append", default=None, help="overlay:tip, overlay:premise, propose:skill, propose:task_wording")
        g.add_argument("--by", default="cli")
        g.add_argument("--reason", default="")
        g.add_argument("--root", default=None, help="alternate targets root (offline tests)")
    g = ts.choices["grant"]
    g.add_argument("--evidence-run", default="")
    g.add_argument("--expires-at", type=float, default=None)
    g.add_argument("--prompt-sha1", default=None)
    g.add_argument("--protocol-version", default=None)
    st = ts.add_parser("status")
    st.add_argument("target")
    st.add_argument("--root", default=None)
    au = ts.add_parser("audit")
    au.add_argument("target")
    au.add_argument("--root", default=None)
    me = ts.add_parser("metrics")
    me.add_argument("--target", default=None)
    me.add_argument("--task", default=None)
    t.set_defaults(fn=cmd_trust)
    h = sub.add_parser("harness", help="inspect or settle structured harness proposals (no model call)")
    hs = h.add_subparsers(dest="harness_action", required=True)
    hst = hs.add_parser("status")
    hst.add_argument("run_dir")
    ha = hs.add_parser("adoption", help="append an adoption cost/stage review; no model or GPU")
    ha.add_argument("run_dir")
    hlist = hs.add_parser("shortlist", help="select at most N evidence-complete historical tool suggestions; no implementation")
    hlist.add_argument("runs", nargs="+")
    hlist.add_argument("--limit", type=int, default=10)
    hp = hs.add_parser("plan-experiment", help="write a dry-run controlled experiment plan; never executes a model or GPU")
    hp.add_argument("run_dir")
    hp.add_argument("--key", required=True)
    hp.add_argument("--seed", type=int, action="append", required=True, help="paired run seed; repeat for multiple seeds")
    hp.add_argument("--repetitions", type=int, default=1)
    hp.add_argument("--budget-usd", type=float, required=True, help="maximum spend per arm and repetition")
    hp.add_argument("--out", default=None)
    hp.add_argument("--allow-spend", action="store_true", help="record explicit model-spend authorization; still does not execute")
    hp.add_argument("--allow-gpu", action="store_true", help="record explicit GPU authorization; still does not execute")
    hp.add_argument("--by", default="cli")
    hp.add_argument("--base-commit", default=None, help="common source commit for both arms; defaults to the recorded implementation base")
    hp.add_argument("--min-pairs", type=int, default=3)
    hp.add_argument("--min-effect", type=float, default=0.0, help="minimum beneficial log-ratio or metric difference")
    hp.add_argument("--direction", choices=["lower", "higher"], default="lower")
    hx = hs.add_parser("execute-experiment", help="default dry-run; --execute additionally requires explicit resource authorization")
    hx.add_argument("run_dir")
    hx.add_argument("--plan", required=True)
    hx.add_argument("--out", default=None)
    hx.add_argument("--execute", action="store_true", help="actually run cells; default only materializes and prints them")
    hx.add_argument("--allow-spend", action="store_true")
    hx.add_argument("--allow-gpu", action="store_true")
    hx.add_argument("--by", default="cli")
    hc = hs.add_parser("collect-experiment", help="collect paired host measurements only; no model call or GPU")
    hc.add_argument("run_dir")
    hc.add_argument("--plan", required=True)
    hc.add_argument("--control", required=True)
    hc.add_argument("--treatment", required=True)
    hc.add_argument("--out", default=None)
    hv = hs.add_parser("validate-effect", help="record a person's effect verdict for a measured authorized experiment")
    hv.add_argument("run_dir")
    hv.add_argument("--result", required=True)
    hv.add_argument("--valid", action=argparse.BooleanOptionalAction, default=False)
    hv.add_argument("--by", required=True)
    hv.add_argument("--reason", required=True)
    hd = hs.add_parser("disable")
    hd.add_argument("run_dir")
    hd.add_argument("--by", default="cli")
    hd.add_argument("--reason", default="")
    hse = hs.add_parser("settle")
    hse.add_argument("run_dir")
    hse.add_argument("--target", default="apple-m4-max")
    hse.add_argument("--task", required=True)
    hse.add_argument("--protocol-version", default=None)
    hse.add_argument("--active", action="store_true", help="write an active overlay; default is shadow only")
    hv = hs.add_parser("convert")
    hv.add_argument("run_dir")
    hv.add_argument("--seq", type=int, required=True)
    hv.add_argument("--digest", required=True)
    hv.add_argument("--surface", required=True, choices=["tip", "premise", "skill", "task_wording"])
    hv.add_argument("--action", required=True)
    hv.add_argument("--probe", default="")
    hv.add_argument("--measurement", default="")
    hv.add_argument("--reproduction", default="", help="minimal reproduction for a tool/reporting/wording proposal")
    hv.add_argument("--expected", default="", help="observable expected result after the change")
    hv.add_argument("--rationale", default="")
    hv.add_argument("--target", default=None)
    hv.add_argument("--task", default=None)
    hv.add_argument("--dtype", default=None)
    hv.add_argument("--by", default="cli", help="person confirming the conversion")
    hv.add_argument("--mode", default="manual", choices=["manual", "ai"], help="who filled the structured fields")
    hv.add_argument("--supersede", default="", help="existing proposal key to revise; the old row stays")
    hv.add_argument("--confirm", action="store_true", help="human confirmation gate")
    hk = hs.add_parser("commit")
    hk.add_argument("run_dir")
    hk.add_argument("--message", required=True)
    hk.add_argument("--by", default="cli")
    hi = hs.add_parser("implement")
    hi.add_argument("run_dir")
    hi.add_argument("--key", required=True)
    hi.add_argument("--target", default="apple-m4-max")
    hi.add_argument("--provider", default=None)
    hi.add_argument("--model", default=None)
    hi.add_argument("--budget-usd", type=float, default=None, help="optional cap; otherwise derived from model prices")
    hi.add_argument("--by", default="cli")
    hr = hs.add_parser("review-code", help="record a named human diff review of a patch; never executes it")
    hr.add_argument("run_dir")
    hr.add_argument("--key", required=True)
    hr.add_argument("--by", required=True, help="named human who inspected the diff")
    for action in ("verify-code", "verify-legacy"):
        hc = hs.add_parser(action, help="run static acceptance or an isolated dynamic runner; never applies locally")
        hc.add_argument("run_dir")
        hc.add_argument("--key", required=True)
        hc.add_argument("--by", default="cli", help="named person upgrading a legacy acceptance or the reviewer for --execute")
        hc.add_argument("--check-id", action="append", default=None, help="registered host check id; repeat for multiple checks")
        hc.add_argument("--legacy-text", default="", help="original text acceptance to upgrade explicitly")
        hc.add_argument("--static", action="store_true", help="only inspect paths/spec/patch hash/claims; never execute patch code")
        hc.add_argument("--execute", action="store_true", help="execute patched code only after review-code and an isolated runner")
        hc.add_argument("--docker-image", default=None, help="image for the isolated patch-test container; required with --execute")
        hc.add_argument("--docker-binary", default="docker", help="Docker-compatible executable used by the isolated runner")
    hc = hs.add_parser("apply-code")
    hc.add_argument("run_dir")
    hc.add_argument("--key", required=True)
    hc.add_argument("--check", action="store_true")
    hc.add_argument("--run-tests", action=argparse.BooleanOptionalAction, default=True, help="run the offline suite against the patched tree in an isolated worktree (default: on)")
    hc.add_argument("--by", default="cli")
    hc.add_argument("--confirm", action="store_true")
    hc.add_argument("--legacy-unvalidated", action="store_true", help="explicitly apply an older patch with no acceptance/effect evidence; never marks it experiment_validated")
    hr = hs.add_parser("review")
    hr.add_argument("run_dir")
    hr.add_argument("--key", required=True)
    hr.add_argument("--target", default="apple-m4-max")
    hr.add_argument("--task", required=True)
    hr.add_argument("--provider", default=None)
    hr.add_argument("--model", default=None)
    hp = hs.add_parser("patch")
    hp.add_argument("run_dir")
    hp.add_argument("--target", default="apple-m4-max")
    hp.add_argument("--key", required=True)
    ha = hs.add_parser("apply")
    ha.add_argument("run_dir")
    ha.add_argument("--target", default="apple-m4-max")
    ha.add_argument("--key", required=True)
    ha.add_argument("--check", action="store_true")
    ha.add_argument("--run-tests", action="store_true", help="run the feedback/contract offline tests before applying")
    ha.add_argument("--by", default="cli", help="person applying the patch; recorded in the decision log")
    ha.add_argument("--confirm", action="store_true")
    h.set_defaults(fn=cmd_harness)
    fb = sub.add_parser("feedback", help="preview or atomically review agent-feedback claims (no model call, no GPU)")
    fb.add_argument("run_dir", help="run id or run directory")
    fb.add_argument("--file", required=True, help="JSON array of {seq, digest, status, ...}; '-' reads stdin")
    fb.add_argument("--dry-run", action="store_true", help="validate and show the planned transitions only")
    fb.add_argument("--confirm", action="store_true", help="write the batch after all items validate")
    fb.add_argument("--method", default="human", choices=["human", "agent", "script", "api"])
    fb.add_argument("--batch-id", default="", help="optional audit batch id")
    fb.set_defaults(fn=cmd_feedback)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
