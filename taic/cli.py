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


# 一次新搜索从这里进入。load_config 把预设和命令行里非空的预算、分支数、轮数上限合成 LLMConfig 与 SearchConfig，再交给 Orchestrator.run。有 best_speedup 则进程成功，一颗合格内核都没有则失败。
def _recorded_target(run_dir: Path, requested: str | None) -> str:
    """The target stored on the run. A different --target is refused before any model call."""
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


def cmd_run(a: argparse.Namespace) -> int:
    from tcenv.children import install_termination_handlers

    install_termination_handlers()
    from .committees import load_committees
    from .llm import LLMError, load_config, load_raw_config
    from .orchestrator import Orchestrator, RunStartError

    # 预设与预算的汇合点。a.config 有值时整份文件原样使用，不再叠预设；a.budget 为 None 时保留预设里的 budget_usd（paper 即 PAPER_BUDGET_USD）。
    raw = load_raw_config(a.config, a.preset)
    cfg, search = load_config(a.config, model=a.model, preset=a.preset, budget_usd=a.budget, branches=a.branches, max_rounds=a.max_rounds)
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
    try:
        # 显式 --model 让两个委员会都用这个模型。没写时用配置里的两名成员，可以不同。
        committees = load_committees(raw, active=cfg, model_override=a.model)
        orch = Orchestrator(
            a.task, a.dtype, a.size, cfg, search,
            run_dir=Path(a.run_dir) if a.run_dir else None,
            from_run=None if a.from_run in (None, "@catalog") else Path(a.from_run),
            seed=getattr(a, "seed", None),
            committees=committees,
            target_id=getattr(a, "target", "apple-m4-max"),
            from_catalog=a.from_run == "@catalog",
        )
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
    return 0 if res.best_speedup else 2


SEARCH_FIELDS = ("budget_usd", "branches", "max_repair_tries", "max_rounds", "max_quick_checks", "min_improvement", "noise_z", "confirm_top", "eval_reps", "max_hangs",
                 "correctness_only", "profile_best", "patience", "plan_mix", "init_samples", "confirm_batches", "profile_poor", "profile_poor_below", "profile_max", "hw_feedback",
                 "stop_at_roofline", "abi_guard", "target_note", "rich_history", "status_lines", "differential_recheck",
                 "plan_ledger", "diversity_archive", "fast_bench", "experience_index", "kernel_agent", "require_contract_launch",
                 "relay_budget_usd", "turn_minutes", "peer_tips", "harness_auto", "harness_shadow")
LLM_FIELDS = ("model", "reasoning_effort", "echo_reasoning_content", "use_tools", "keep_transcript_bodies", "extra_body", "max_tokens", "request_timeout_s", "max_retries", "truncation_retry",
              "truncation_retry_effort", "plan_reasoning_effort", "patch_reasoning_effort")


# 续跑的配置以这次运行自己记下的设置为准，命令行只覆盖显式给出的项。没有 run_config.json 的老运行用 base，避免被后来的 paper 预设改掉当时的预算和轮数。dry_run 即使没有 API 凭证也要能组出配置，否则「先看看」会因为没钥匙而做不成。
def _resume_config(a: argparse.Namespace, settings: dict, info: dict):
    """(LLMConfig, SearchConfig, provenance) of the continued run: the run's own settings, then what the command line says."""
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
    flags = {"patience": a.patience, "branches": a.branches, "max_rounds": a.max_rounds, "init_samples": a.init_samples}
    for k, v in flags.items():
        if v is not None:
            setattr(search, k, v)
            prov[f"search.{k}"] = "command line"
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
    """The two committees of a continued run. Recorded members win, unless ``--model`` names one model for both.

    A run started before committees existed gets two committees that copy its model. The round number still picks who is next.
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
    """The run cannot be continued; ``code`` is the exit status of the command."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


# 真正续跑之前的全部检查和装载。失败也不写运行目录，中断时的样子还在。预算由 effective_budget 决定：--budget 是新总上限，--add-budget 加在记下的上限上，都没有则沿用记下的值或 control.json 里尚未生效的请求；新上限不高于已花费就拒绝。
def prepare_resume(a: argparse.Namespace, *, client=None, evaluate_fn=None, log=print):
    """Everything ``resume`` does before the first model request: check that the run can be continued, rebuild its state, build the orchestrator and load the
    state into it. Writes nothing in the run directory. Returns ``(orchestrator, snapshot, report, info, new_budget)``; raises ``ResumeRefused``."""
    from tasks import get_task

    from . import control, prompts, resume as rs
    from .llm import LLMError
    from .orchestrator import Orchestrator

    run_dir = Path(a.run_dir).resolve()
    info = rs.inspect_run(run_dir, allow_count_stops=bool(getattr(a, "lift_count_limits", False)))
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
        task = get_task(info["task"], info["dtype"], doc.get("size", "paper"))
        snap = rs.load_snapshot(run_dir, search, task.contract.kernel_name, settings=settings)
        # inspect 用的是 result/state 里的花费，转录可以更高。新上限必须盖过重建后的花费。
        if new_budget <= snap.spent_usd + 1e-9:
            raise rs.ResumeError(
                f"budget ${new_budget:.2f} does not exceed what the transcript has already spent (${snap.spent_usd:.4f}): add budget with --add-budget USD or --budget TOTAL"
            )
        first = snap.partial.rnd if snap.partial else snap.next_round
        if first > search.max_rounds:
            raise rs.ResumeError(f"all {search.max_rounds} rounds were used (the run is past round {first - 1}): raise the limit with --max-rounds")
        committees = _committees_for_resume(a, cfg, run_dir)
    except (rs.ResumeError, rs.NeedsInspection, LLMError, control.ControlError, KeyError, ValueError) as e:
        raise ResumeRefused(f"cannot resume {run_dir}: {e}", 5) from e
    orch = Orchestrator(snap.task, snap.dtype, snap.size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn, log=log, resuming=True, committees=committees, target_id=_recorded_target(run_dir, getattr(a, "target", None)))
    variants = {prompts.system_prompt(orch.task, use_tools=cfg.use_tools, profiler_evidence=pe, hw_feedback=hw) for pe in (False, True) for hw in (False, True)}
    # 系统提示若已和现在的 prompts.py 不同，仍用转录里的原文。换提示会让同一次实验的前后轮次对着不同契约说话。
    prompt_changed = bool(snap.system) and snap.system not in variants  # False: the prompt files still produce the prompt the run began with
    # dry_run 只装内存。restore 在这个开关下不写转录、不改 state.json。
    orch.restore(snap, dry_run=True)
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
    """Keys filled from a preset or a default because this run did not record them. Each item is ``{"key", "source"}``."""
    return [{"key": key, "source": source} for key, source in sorted(prov.items()) if isinstance(source, str) and "not recorded by this run" in source]


def _announce_unrecorded(report: dict, log) -> None:
    """Say which settings were filled in. A resume that can continue still continues."""
    rows = report.get("unrecorded_settings") or []
    if not rows:
        return
    log("settings this run did not record, filled from the named source (the resume is not refused):")
    for row in rows:
        log(f"  {row['key']}: {row['source']}")


# 检查通过之后才占目录锁、把旧结果留档、记下 resume 预算事件，然后进入原来的搜索循环。别的进程占着锁就拒绝，避免两套搜索同时改同一个 run。
def run_prepared(a: argparse.Namespace, orch, snap, info: dict, new_budget: float):
    """Continue the run prepared by ``prepare_resume``: take the run directory's lock, keep the earlier results as ``*.before_topup_<n>.json``, note the
    budget change in ``budget_events.json`` and run the search. Raises ``ResumeRefused`` when another process holds the run."""
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
        moved = rs.snapshot_results(run_dir)
        if moved:
            orch.log(f"kept the earlier results as {', '.join(moved)}")
        orch.restore(snap, dry_run=False)
        event = control.record_event(run_dir, "resume", info["budget_usd"], new_budget, who=a.who, note=f"{info['kind']} run continued at round {first}", spent_usd=snap.spent_usd)
        if not event.get("persisted"):
            orch.log("budget event was not saved; budget_events.json is not a complete record")
        return asyncio.run(orch.run())
    except RunStartError:
        rs.restore_snapshot(run_dir, moved)
        raise
    finally:
        control.release_run_lock(run_dir)


# `taic resume` 的入口。dry_run 只打印将要续的状态就退出，不拿锁、也不跑搜索。不能续时退出码来自 ResumeRefused；真的跑起来之后，有 best_speedup 为 0，否则为 2。
def cmd_resume(a: argparse.Namespace) -> int:
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
    print(json.dumps({k: getattr(res, k) for k in ("task", "dtype", "model", "best_speedup", "confirmed_speedup", "spent_usd", "budget_usd", "stop_reason", "run_dir")}, indent=2))
    return 0 if res.best_speedup else 2


_SEARCH_SWITCHES = (
    ("stop_at_roofline", "stop when the seed is already inside the noise gate of its roofline and no baseline beats that bound"),
    ("abi_guard", "recorded switch; the live session does not stop a branch on a repeated ABI error"),
    ("target_note", "when a baseline is faster than the roofline bound, tell the model and give that baseline's time as the target"),
    ("rich_history", "a failed branch's history line names the failure instead of one generic sentence"),
    ("status_lines", "tell the model the noise floor and the budget, round and patience that remain"),
    ("peer_tips", "put this run's recent peer tips into the next TASK.md; tips never cross runs"),
)


def _add_search_switches(p: argparse.ArgumentParser) -> None:
    """``--flag`` / ``--no-flag``. Omitted means the config default, which is on."""
    for dest, help_text in _SEARCH_SWITCHES:
        p.add_argument("--" + dest.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None, help=help_text + " (default: on)")
    p.add_argument("--truncation-retry", action=argparse.BooleanOptionalAction, default=None, help="recorded switch; the live hmz session does not ask again after a cut-off reply (default: on)")


def _u64(text: str) -> int:
    """An unsigned 64-bit seed written in decimal."""
    try:
        n = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a decimal integer") from None
    if n < 0 or n >= 2**64:
        raise argparse.ArgumentTypeError("must satisfy 0 <= seed < 2**64")
    return n


def _positive_int(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _add_role_efforts(p: argparse.ArgumentParser) -> None:
    """Role efforts, the new search switches, and the per-request token cap."""
    p.add_argument("--plan-effort", default=None, choices=["off", "low", "high", "max"], help="recorded only; the hmz session uses reasoning_effort")
    p.add_argument("--patch-effort", default=None, choices=["off", "low", "high", "max"], help="recorded only; the hmz session uses reasoning_effort")
    p.add_argument("--truncation-retry-effort", default=None, choices=["low", "off", "same"], help="recorded only; the hmz session does not retry a cut-off reply")
    p.add_argument("--plan-ledger", action=argparse.BooleanOptionalAction, default=None, help="recorded only; the hmz session does not read a plan ledger")
    p.add_argument("--diversity-archive", action=argparse.BooleanOptionalAction, default=None, help="recorded only; the hmz session always edits the current seed")
    p.add_argument("--fast-bench", action=argparse.BooleanOptionalAction, default=None, help="search timing skips serial dispatch and slower baselines (default: off)")
    p.add_argument("--experience-index", default=None, help="JSON file of measured hints copied into the session")
    p.add_argument("--max-tokens", type=_positive_int, default=None, help="recorded only; the hmz session cap is relay_budget_usd turned into output tokens")


def _apply_max_tokens(a: argparse.Namespace, cfg, prov: dict | None = None) -> None:
    n = getattr(a, "max_tokens", None)
    if n is None:
        return
    cfg.max_tokens = n
    if prov is not None:
        prov["llm.max_tokens"] = "command line"


def _add_harness_switches(p: argparse.ArgumentParser) -> None:
    p.add_argument("--harness-auto", action=argparse.BooleanOptionalAction, default=None, help="allow structured harness proposals to settle into this run's overlay (default: off)")
    p.add_argument("--harness-shadow", action=argparse.BooleanOptionalAction, default=None, help="record an eligible overlay without injecting it; use --no-harness-shadow only after the shadow pilot (default: on)")


def _apply_search_switches(a: argparse.Namespace, cfg, search, prov: dict | None = None) -> None:
    """Apply an explicit ``--flag`` or ``--no-flag``. An omitted flag leaves the config default."""
    for dest, key in (("harness_auto", "harness_auto"), ("harness_shadow", "harness_shadow")):
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
    """Repo orientation, or one compact JSON object for a run directory. Reads files and starts nothing."""
    from .brief import run_brief
    from .kernel_catalog import list_kernels

    if getattr(a, "kernels", False):
        rows = list_kernels(getattr(a, "target", None))
        print(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
        return 0
    return run_brief(a.run_dir, include_gpu=bool(getattr(a, "gpu", False)))


def cmd_query(a: argparse.Namespace) -> int:
    from . import retrieval

    what = a.what
    if what == "capabilities":
        data = retrieval.capabilities()
    elif what == "runs":
        data = retrieval.runs(
            task=a.task, publication=a.publication, status=a.status,
            min_speedup=a.min_speedup, sort=a.sort, cursor=a.cursor, limit=a.limit,
        )
    elif what == "best":
        data = retrieval.best(task=a.task, limit=a.limit)
    elif what == "failures":
        data = retrieval.failures(status=a.status, task=a.task, limit=a.limit)
    elif what == "sessions":
        if not a.run:
            print("query sessions needs --run RUN_ID", file=sys.stderr)
            return 5
        data = retrieval.sessions(a.run, limit=a.limit)
    else:
        data = retrieval.models()
    if a.jsonl and isinstance(data, dict):
        rows = data.get("runs") or data.get("best") or data.get("failures") or data.get("sessions") or data.get("models")
        if isinstance(rows, list):
            for row in rows:
                print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            return 0
    print(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return 0


def cmd_gc(a: argparse.Namespace) -> int:
    from . import gc

    root = Path(a.root)
    rows = gc.scan(root, older_than_days=a.older_than_days)
    if not rows:
        print(json.dumps({"root": str(root), "rows": [], "mode": "apply" if a.apply else "dry-run"}, ensure_ascii=False))
        return 0
    if a.apply:
        result = gc.apply(rows, log_path=root / "gc_log.jsonl")
        print(json.dumps({"root": str(root), "mode": "apply", **result, "rows": len(rows)}, ensure_ascii=False))
    else:
        print(json.dumps({
            "root": str(root), "mode": "dry-run", "older_than_days": a.older_than_days,
            "eligible": sum(1 for row in rows if row["eligible"]),
            "eligible_bytes": sum(row["bytes"] for row in rows if row["eligible"]),
            "rows": rows,
        }, ensure_ascii=False))
    return 0


def cmd_trust(a: argparse.Namespace) -> int:
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
    from . import harness

    run = Path(a.run_dir)
    try:
        if a.harness_action == "status":
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
        elif a.harness_action == "apply":
            if not a.confirm and not a.check:
                print("refusing to apply without --confirm (use --check first)", file=sys.stderr)
                return 5
            print(json.dumps(harness.apply_patch(run, target_id=a.target, proposal_key=a.key, check_only=a.check), ensure_ascii=False, indent=2))
        else:
            print("unknown harness action", file=sys.stderr)
            return 5
    except (harness.ProposalError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 5
    return 0


def cmd_migrate_runs(a: argparse.Namespace) -> int:
    from .migrate import migrate_runs

    root = Path(a.root) if a.root else None
    moved = migrate_runs(root, a.target)
    print(json.dumps({"target": a.target, "moved": moved}, ensure_ascii=False))
    return 0


# `run` 开新搜索，`resume` 续旧目录。两边的 --budget 含义不同：run 是这一次内核的总上限，resume 是续跑后的新总上限，必须高于已经花掉的金额。
# `brief` 不接预算：没有 RUN_DIR 时是仓库方位，有目录时是一份短 JSON。
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="taic", description="Triton AI Compiler for Apple GPUs (Triton -> MSL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="lower one task with the LLM agent")
    p.add_argument("task")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--size", default="paper", choices=["paper", "small", "plan"])
    # 新搜索的三层配置：--config 独占整份文件；否则 --preset（或环境变量 TAIC_PRESET，默认 paper）叠在 taic.toml 上。--budget 再覆盖叠好之后的 budget_usd。
    p.add_argument("--config", default=None, help="one config file, used as it is (no preset is layered on it)")
    p.add_argument("--preset", default=None, help="settings layered over taic.toml: 'paper' (default, taic.paper.toml: the paper's $15 budget) or 'base' (taic.toml alone); also $TAIC_PRESET")
    p.add_argument("--model", default=None)
    p.add_argument("--budget", type=float, default=None, help="USD budget for this kernel (default: the preset's budget_usd, the paper's $15)")
    p.add_argument("--target", default="apple-m4-max", help="device profile id under targets/")
    p.add_argument("--branches", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
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
    p.add_argument("--init-samples", type=int, default=None, help="must be 1; any other value is refused before a session starts")
    p.add_argument("--confirm-batches", type=int, default=None, help="independent confirm evaluations per candidate; the published speedup is their median (default 3, 1 = the old single batch)")
    p.add_argument("--patience", type=int, default=None, help="stop after this many rounds without a new best (default 6)")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--seed", type=_u64, default=None, help="unsigned 64-bit run seed (default: a new random one). resume keeps the seed the run recorded")
    p.add_argument("--from", dest="from_run", default=None, help="directory of a previous run: if its best.metal is still eligible for this task, skip writing a first draft and improve that kernel")
    p.set_defaults(fn=cmd_run)
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
    r.add_argument("--init-samples", type=int, default=None)
    r.add_argument("--patience", type=int, default=None, help="not recorded by runs older than run_config.json: give the value the run was started with")
    r.add_argument("--lift-count-limits", action="store_true", help="also continue a run that stopped because it used its round limit or its patience")
    r.set_defaults(fn=cmd_resume)
    q = sub.add_parser("query", help="read-only queries over the observe index (no model call, no GPU)")
    q.add_argument("what", choices=["runs", "best", "failures", "sessions", "models", "capabilities"])
    q.add_argument("--task", default=None)
    q.add_argument("--publication", default=None)
    q.add_argument("--status", default=None)
    q.add_argument("--run", default=None)
    q.add_argument("--min-speedup", type=float, default=None)
    q.add_argument("--sort", default="started", choices=["started", "speedup", "spent"])
    q.add_argument("--cursor", default=None)
    q.add_argument("--limit", type=int, default=50)
    q.add_argument("--jsonl", action="store_true", help="one JSON object per row")
    q.set_defaults(fn=cmd_query)
    g = sub.add_parser("gc", help="report or remove old data/deleted directories (default dry-run)")
    g.add_argument("--root", default="data/deleted")
    g.add_argument("--older-than-days", type=float, default=30.0)
    g.add_argument("--apply", action="store_true", help="delete eligible directories; without it only report")
    g.set_defaults(fn=cmd_gc)
    b = sub.add_parser("brief", help="read-only orientation, or JSON facts for one run directory (no model call, no GPU)")
    b.add_argument("run_dir", nargs="?", default=None, help="run directory; omit for a repo orientation")
    b.add_argument("--kernels", action="store_true", help="list confirmed kernels; optional --target filters one profile")
    b.add_argument("--target", default=None)
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
    hp = hs.add_parser("patch")
    hp.add_argument("run_dir")
    hp.add_argument("--target", default="apple-m4-max")
    hp.add_argument("--key", required=True)
    ha = hs.add_parser("apply")
    ha.add_argument("run_dir")
    ha.add_argument("--target", default="apple-m4-max")
    ha.add_argument("--key", required=True)
    ha.add_argument("--check", action="store_true")
    ha.add_argument("--confirm", action="store_true")
    h.set_defaults(fn=cmd_harness)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
