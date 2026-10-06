"""``taic`` command line: run the AI compiler on a task."""

# 命令行进入搜索的地方。`taic run` 开一次新搜索；`taic resume` 从已有 run 目录接着跑，已经花掉的模型请求和 GPU 评测不会再花一遍。
# 新搜索的预算和预设不在这里写死。未指定 --config 时，load_config 读 taic.toml，再按预设名叠 taic.<preset>.toml；预设名是 --preset，否则环境变量 TAIC_PRESET，否则 DEFAULT_PRESET（paper，论文的 $15，即 PAPER_BUDGET_USD）。--budget 只在显式给出时覆盖 search.budget_usd。
# 续跑的预算另算：目录里记下的花费和上限先恢复，--budget 改总上限，--add-budget 加在记下的上限上；两者都没有就沿用记下的上限或 control.json 里还没生效的请求，但新上限必须高于已花费。没记下的键才回到预设，默认 base（只有 taic.toml）。

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


# 一次新搜索从这里进入。load_config 把预设和命令行里非空的预算、分支数、轮数上限合成 LLMConfig 与 SearchConfig，再交给 Orchestrator.run。有 best_speedup 则进程成功，一颗合格内核都没有则失败。
def cmd_run(a: argparse.Namespace) -> int:
    from .llm import load_config
    from .orchestrator import Orchestrator

    # 预设与预算的汇合点。a.config 有值时整份文件原样使用，不再叠预设；a.budget 为 None 时保留预设里的 budget_usd（paper 即 PAPER_BUDGET_USD）。
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
    if a.plan_mix:
        search.plan_mix = True
    if a.init_samples:
        search.init_samples = a.init_samples
    if a.patience:
        search.patience = a.patience
    if a.effort:
        cfg.reasoning_effort = a.effort
    orch = Orchestrator(a.task, a.dtype, a.size, cfg, search, run_dir=Path(a.run_dir) if a.run_dir else None)
    res = asyncio.run(orch.run())
    print(json.dumps({k: getattr(res, k) for k in ("task", "dtype", "model", "best_speedup", "confirmed_speedup", "spent_usd", "stop_reason", "run_dir")}, indent=2))
    return 0 if res.best_speedup else 2


SEARCH_FIELDS = ("budget_usd", "branches", "max_repair_tries", "max_rounds", "max_quick_checks", "min_improvement", "noise_z", "confirm_top", "eval_reps", "max_hangs",
                 "correctness_only", "profile_best", "patience", "plan_mix", "init_samples", "profile_poor", "profile_poor_below", "profile_max", "hw_feedback")
LLM_FIELDS = ("model", "reasoning_effort", "echo_reasoning_content", "use_tools", "extra_body", "max_tokens", "request_timeout_s", "max_retries")


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
    if a.no_tools:
        cfg.use_tools = False
        prov["llm.use_tools"] = "command line"
    if a.effort:
        cfg.reasoning_effort = a.effort
        prov["llm.reasoning_effort"] = "command line"
    return cfg, search, prov


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
    info = rs.inspect_run(run_dir)
    if not info["resumable"]:
        raise ResumeRefused(f"cannot resume {run_dir}: {info['reason'] or info['kind']}", 3)
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
        first = snap.partial.rnd if snap.partial else snap.next_round
        if first > search.max_rounds:
            raise rs.ResumeError(f"all {search.max_rounds} rounds were used (the run is past round {first - 1}): raise the limit with --max-rounds")
    except (rs.ResumeError, LLMError, control.ControlError, KeyError) as e:
        raise ResumeRefused(f"cannot resume {run_dir}: {e}", 5) from e
    orch = Orchestrator(snap.task, snap.dtype, snap.size, cfg, search, run_dir=run_dir, client=client, evaluate_fn=evaluate_fn, log=log, resuming=True)
    variants = {prompts.system_prompt(orch.task, use_tools=cfg.use_tools, profiler_evidence=pe, hw_feedback=hw) for pe in (False, True) for hw in (False, True)}
    # 系统提示若已和现在的 prompts.py 不同，仍用转录里的原文。换提示会让同一次实验的前后轮次对着不同契约说话。
    prompt_changed = bool(snap.system) and snap.system not in variants  # False: the prompt files still produce the prompt the run began with
    # dry_run 只装内存。restore 在这个开关下不写转录、不改 state.json。
    orch.restore(snap, dry_run=bool(a.dry_run))
    report = rs.describe(snap)
    report.update(
        dry_run=bool(a.dry_run), kind=info["kind"], budget_before_usd=info["budget_usd"], budget_after_usd=new_budget, remaining_usd=round(new_budget - snap.spent_usd, 6),
        system_prompt=(("the one recorded in the transcript is reused" + (" (the prompt files were edited since the run started; this keeps the run's conditions)" if prompt_changed
                        else " (identical to what the current prompt files produce)")) if snap.system else "rebuilt from the current prompt files (no transcript)"),
        settings={k: getattr(search, k) for k in SEARCH_FIELDS} | {"model": cfg.model, "use_tools": cfg.use_tools, "reasoning_effort": cfg.reasoning_effort},
        provenance=prov, command=f"uv run python -m taic.cli resume {run_dir}" + (f" --add-budget {a.add_budget}" if a.add_budget is not None else "") + (f" --budget {a.budget}" if a.budget is not None else ""),
    )
    return orch, snap, report, info, new_budget


# 检查通过之后才占目录锁、把旧结果留档、记下 resume 预算事件，然后进入原来的搜索循环。别的进程占着锁就拒绝，避免两套搜索同时改同一个 run。
def run_prepared(a: argparse.Namespace, orch, snap, info: dict, new_budget: float):
    """Continue the run prepared by ``prepare_resume``: take the run directory's lock, keep the earlier results as ``*.before_topup_<n>.json``, note the
    budget change in ``budget_events.json`` and run the search. Raises ``ResumeRefused`` when another process holds the run."""
    from . import control, resume as rs

    run_dir = Path(a.run_dir).resolve()
    first = snap.partial.rnd if snap.partial else snap.next_round
    try:
        control.acquire_run_lock(run_dir, who=a.who)
    except control.RunLockError as e:
        raise ResumeRefused(f"cannot resume {run_dir}: {e}", 4) from e
    try:
        moved = rs.snapshot_results(run_dir)
        if moved:
            orch.log(f"kept the earlier results as {', '.join(moved)}")
        control.record_event(run_dir, "resume", info["budget_usd"], new_budget, who=a.who, note=f"{info['kind']} run continued at round {first}", spent_usd=snap.spent_usd)
        return asyncio.run(orch.run())
    finally:
        control.release_run_lock(run_dir)


# `taic resume` 的入口。dry_run 只打印将要续的状态就退出，不拿锁、也不跑搜索。不能续时退出码来自 ResumeRefused；真的跑起来之后，有 best_speedup 为 0，否则为 2。
def cmd_resume(a: argparse.Namespace) -> int:
    try:
        orch, snap, report, info, new_budget = prepare_resume(a)
        if a.dry_run:
            print(json.dumps(report, indent=2, default=str))
            return 0
        res = run_prepared(a, orch, snap, info, new_budget)
    except ResumeRefused as e:
        print(str(e), file=sys.stderr)
        return e.code
    print(json.dumps({k: getattr(res, k) for k in ("task", "dtype", "model", "best_speedup", "confirmed_speedup", "spent_usd", "budget_usd", "stop_reason", "run_dir")}, indent=2))
    return 0 if res.best_speedup else 2


# `run` 开新搜索，`resume` 续旧目录。两边的 --budget 含义不同：run 是这一次内核的总上限，resume 是续跑后的新总上限，必须高于已经花掉的金额。
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
    p.add_argument("--branches", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--effort", default=None, choices=["off", "low", "high", "max"])
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--profile-best", action="store_true", help="xctrace profile of the winning kernel after the search")
    p.add_argument("--profile-poor", action="store_true", help="trace a round's seed with Instruments when it is slower than the best baseline and give the findings to the model")
    p.add_argument("--plan-mix", action="store_true", help="the last plan of every round is a local, low-risk change; the others are structural")
    p.add_argument("--hw-feedback", action="store_true", help="append a short static hardware-feature analysis of the seed to the planner and patch feedback and teach it as Skill 2c (off by default)")
    p.add_argument("--init-samples", type=int, default=None, help="sample this many initial kernels in parallel; the best eligible one seeds the search")
    p.add_argument("--patience", type=int, default=None, help="stop after this many rounds without a new best (default 4)")
    p.add_argument("--run-dir", default=None)
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
    r.add_argument("--no-tools", action="store_true")
    r.add_argument("--profile-best", action="store_true")
    r.add_argument("--profile-poor", action="store_true")
    r.add_argument("--plan-mix", action="store_true")
    r.add_argument("--hw-feedback", action="store_true")
    r.add_argument("--init-samples", type=int, default=None)
    r.add_argument("--patience", type=int, default=None, help="not recorded by runs older than run_config.json: give the value the run was started with")
    r.set_defaults(fn=cmd_resume)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
