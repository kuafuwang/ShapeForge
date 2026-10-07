"""Turn run directories and the GPU cache into the dashboard's current frame."""

# 看板的只读面：把 runs/ 里已经落下的快照收成一帧。这里不写搜索，也不改结果文件。
# 读不出来的 JSON（写到一半、被截断）当成没有这份快照。结果缺失但 state 仍新鲜或进程还在，就当作仍在跑；state 也没有是未知，心跳过期且进程不在是中断。都不会标成已完成。

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from observe.budget import budget_status, summarize
from observe.publish import gpu_dir, gpu_events_path, observe_dir

# 超过这么久没再写 state.json，心跳才算停。还要 pid 已经不在，才会把运行标成中断。
STALE_S = 15.0
# 总览只带最近这些条 GPU 事件，一帧才跟得上页面的轮询。
EVENT_TAIL = 40

REPO_ROOT = Path(__file__).resolve().parent.parent


# 结果目录可以指到别处拷来的 runs/。默认仍是仓库下的 runs，看板不创建它。
def runs_root() -> Path:
    env = os.environ.get("AI_LOWERLING_RUNS_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "runs"


# state.json 里的 pid 可能已经过期。信号 0 只问进程还在不在：权限不够说明进程在，查找失败说明已经退出。
def pid_alive(pid: object) -> bool:
    try:
        number = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if number <= 0:
        return False
    try:
        os.kill(number, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程在，只是这个用户不能给它发信号。看成还活着，避免把别人的搜索标成中断。
        return True
    except OSError:
        return False
    return True


# 半截 JSON、空文件、或根节点不是对象，都返回 None。调用方分不清「还没写」和「写坏了」，只能按缺失处理。
def _load(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # 文件不存在和写到一半都进这里。截断的 result.json 因此不会被当成一次已经完成的结果。
        return None
    return data if isinstance(data, dict) else None


# 心跳年龄在读的这一刻才算。文件里只存绝对时间，隔很久打开看板时不会把旧年龄当成现在。
def _age(doc: dict | None) -> dict | None:
    if not doc:
        return None
    out = dict(doc)
    at = out.get("heartbeat_at")
    if isinstance(at, (int, float)):
        out["heartbeat_age_s"] = round(max(0.0, time.time() - float(at)), 1)
    return out


# 只接受 runs/ 里面的相对目录，避免查询串用 .. 读到仓库外。两份快照都不存在，就不是一次搜索。
def resolve_run(run_id: str) -> Path | None:
    """A directory under ``runs/`` that holds a search snapshot. Rejects anything that climbs out."""
    if not run_id or "\x00" in run_id:
        return None
    rel = Path(run_id)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    root = runs_root().resolve()
    try:
        path = (root / rel).resolve()
    except OSError:
        return None
    if path != root and root not in path.parents:
        return None
    if not path.is_dir():
        return None
    if not (path / "state.json").exists() and not (path / "result.json").exists():
        # 这里只看文件在不在。内容截断仍算「有这份运行」，解析失败后面另当缺失快照。
        return None
    return path


# 页面上的运行 id 是 runs/ 下的相对路径，详情接口再用 resolve_run 认回来。
def _rel(path: Path) -> str:
    return path.resolve().relative_to(runs_root().resolve()).as_posix()


# 完成先看 result 能否读成对象，或 state 的 phase 已经是 done。结果缺失时：15 秒内有心跳或 pid 还在，当作仍在跑；心跳过期且进程不在是中断；连 state 都没有是未知。
def liveness(state: dict | None, result: dict | None) -> str:
    if result or (state and state.get("phase") == "done"):
        # result 能读出来就是完成。坏掉的 state 不会把已完成的运行打回「还在跑」。
        return "done"
    if not state:
        # 结果解析失败（含截断）且没有可用的 state：残缺，不叫 running。
        return "unknown"
    updated = float(state.get("updated_at") or 0)
    if time.time() - updated > STALE_S and not pid_alive(state.get("pid")):
        # 两个条件都要。只是 15 秒没写心跳但进程还在，仍算在跑；进程已死但时间戳还新，也先算在跑。
        return "interrupted"
    return "running"


# 套件目录自己的 state.json 带 script、没有本目录的 result.json，那是实验队列，交给 experiment_jobs，不进单次核的历史。
# 核运行的 state 有 task、没有 script。result 截断时只要这份 state 还在，目录仍出现在历史里，由 liveness 判成在跑或中断。
def _search_dirs() -> list[tuple[Path, dict | None, dict | None]]:
    root = runs_root()
    if not root.is_dir():
        return []
    dirs: set[Path] = set()
    for name in ("state.json", "result.json"):
        for path in root.rglob(name):
            dirs.add(path.parent)
    found: list[tuple[Path, dict | None, dict | None]] = []
    for directory in dirs:
        state = _load(directory / "state.json")
        result = _load(directory / "result.json")
        if state and state.get("script") and not result:
            # 实验队列。result 读不出来（含半截）时也不能把它误当成一次核搜索。
            continue
        if result or (state and state.get("task") and not state.get("script")):
            # 只有半截 result、又没有可读的搜索 state 时，两个条件都不成立，这个目录不会出现在历史里。
            found.append((directory, state, result))
    return found


# 跑完优先用换种子确认的加速比；确认值缺失才退回搜索期的最优。还在跑时 result 还没有，只能看 state 里的当前最优。
def _speedup(state: dict | None, result: dict | None) -> float | None:
    if result:
        value = result.get("confirmed_speedup")
        if value is None:
            value = result.get("best_speedup")
        return value
    if state:
        return state.get("best_speedup")
    return None


# 排序用的开始时间。state 里没有 started_at 时退回文件修改时间，旧运行仍能按先后出现。
def _when(directory: Path, state: dict | None) -> float:
    if state and isinstance(state.get("started_at"), (int, float)):
        return float(state["started_at"])
    try:
        return (directory / "state.json").stat().st_mtime if (directory / "state.json").exists() else directory.stat().st_mtime
    except OSError:
        return 0.0


# 后面的空值不能盖掉前面已经写上的数。旧运行往往只有其中一份快照带花费。
def _first(*values):
    return next((value for value in values if value is not None), None)


# 花费和预算按 result、state、state 里的账本、ledger.json 这个顺序补。缺的留空，横幅才能走 unknown 而不是当成 0。
def money_fields(directory: Path, state: dict | None, result: dict | None) -> tuple[object, object]:
    """(spent USD, budget USD) of a run, from the best source that has them: result.json, state.json, the ledger inside state.json,
    ledger.json. Older runs lack some of these; a missing value stays ``None``."""
    ledger = (state or {}).get("ledger") or {}
    spent = _first((result or {}).get("spent_usd"), (state or {}).get("spent_usd"), ledger.get("total_usd"))
    budget = _first((result or {}).get("budget_usd"), (state or {}).get("budget_usd"), ledger.get("budget_usd"))
    if spent is None or budget is None:
        doc = _load(directory / "ledger.json") or {}
        # 有的账本把合计放在 summary 里，旧文件则本身就是合计。截断的 ledger 被 _load 丢掉，已经取到的数保留。
        summary = doc.get("summary") if isinstance(doc.get("summary"), dict) else doc
        spent = _first(spent, summary.get("total_usd"))
        budget = _first(budget, summary.get("budget_usd"))
    return spent, budget


# 重置或加预算先写在 budget_events.json，新进程接手后才会改 result.json。告警用已经答应的较高预算，不等那次接手。
def _granted_budget(directory: Path) -> float | None:
    """The highest budget a reset or top-up has already recorded for this run."""
    best = None
    for event in _budget_events(directory):
        if event.get("kind") not in ("requested", "applied", "resume"):
            continue
        amount = event.get("new_budget_usd")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            continue
        if best is None or float(amount) > best:
            best = float(amount)
    return best


def _budget_for_alert(directory: Path, spent, budget, stop: str) -> tuple[object, str]:
    """Budget and stop reason used only for the alert. A granted budget above what was spent retires the old budget_exhausted flag."""
    granted = _granted_budget(directory)
    shown = budget
    if granted is not None and (shown is None or granted > float(shown)):
        shown = granted
    # 只在重置已经把预算抬到花费之上时撤掉「预算耗尽」。原来停在上限之内的旧结果仍要告警。
    if granted is not None and stop == "budget_exhausted" and spent is not None and shown is not None and float(shown) > float(spent):
        return shown, ""
    return shown, stop


# 每一行在读盘时就算好预算档位，NEAR 用的「是否在跑」和这一行的 status 是同一次判断。
# 「加预算」按钮要不要出现，只看已经读进内存的 state / result：不再多读文件。
# 运行中且进程读控制文件 → 在线加；跑完因预算停下、或进程已死且有可续的内核 → 续跑；其余给出原因。
def control_hint(directory: Path, state: dict | None, result: dict | None, status: str) -> dict:
    from taic.resume import STOP_BUDGET, has_best

    stop = (result or {}).get("stop_reason") or (state or {}).get("stop_reason") or ""
    if status == "running":
        if (state or {}).get("control_supported"):
            return {"action": "topup_live", "reason": ""}
        return {"action": None, "reason": "started by an older version: it cannot be topped up live; wait for it to stop, then resume it"}
    if (directory / "resume.lock").exists():
        return {"action": None, "reason": "a resume of this run is starting"}
    if status == "interrupted":
        if has_best(state, result):
            return {"action": "resume", "reason": ""}
        return {"action": None, "reason": "no valid kernel was found before the process stopped: start a new run"}
    if status == "done" and stop == STOP_BUDGET:
        if has_best(state, result):
            return {"action": "resume", "reason": ""}
        return {"action": None, "reason": "stopped for budget without a valid kernel: start a new run"}
    if status == "done":
        return {"action": None, "reason": f"stopped for '{stop or 'unknown reason'}', not for budget: only runs that ran out of budget (or were interrupted) can be continued"}
    return {"action": None, "reason": ""}


# 加过预算的运行不能和固定预算的运行直接比。首次预算取第一条生效事件里的旧值。
def topup_fields(directory: Path) -> dict:
    if not (directory / "budget_events.json").exists():
        return {"topped_up": False, "initial_budget_usd": None}
    from taic import control

    hist = control.budget_history(directory)
    return {"topped_up": hist["topped_up"], "initial_budget_usd": hist["initial_budget_usd"]}


def _budget_events(directory: Path) -> list[dict]:
    if not (directory / "budget_events.json").exists():
        return []
    from taic import control

    return control.read_events(directory)


def history() -> list[dict]:
    rows = []
    for directory, state, result in _search_dirs():
        status = liveness(state, result)
        task = (result or state or {}).get("task") or ""
        dtype = (result or state or {}).get("dtype") or ""
        phase = (state or {}).get("phase") or ""
        stop = (result or {}).get("stop_reason") or (state or {}).get("stop_reason") or ""
        spent, budget = money_fields(directory, state, result)
        alert_budget, alert_stop = _budget_for_alert(directory, spent, budget, stop)
        rows.append({
            "id": _rel(directory),
            "started_at": _when(directory, state),
            "task": task,
            "dtype": dtype,
            "phase": phase,
            "stop_reason": stop,
            "speedup": _speedup(state, result),
            "spent_usd": spent,
            "budget_usd": alert_budget,
            "budget": budget_status(spent, alert_budget, alert_stop, status == "running"),
            "control": control_hint(directory, state, result, status),
            **topup_fields(directory),
            "round": (state or {}).get("round"),
            "max_rounds": (state or {}).get("max_rounds"),
            "inflight_label": (state or {}).get("inflight_label"),
            "status": status,
        })
    rows.sort(key=lambda row: row["started_at"], reverse=True)
    return rows


# 事件日志是追加写的，进程被杀掉时最后一行可能只写了一半。解析失败的那一行丢掉，前面完整的事件仍然留下。
def _tail_events() -> list[dict]:
    path = gpu_events_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-EVENT_TAIL:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            # 半行 JSON 不是一条事件。跳过它，避免整段日志因为最后一次追加没写完而作废。
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


# 只展示 pid 还活着的持有者和等待者。进程退出后留下的 json 不再占着「正在用 GPU」。
def gpu_snapshot() -> dict:
    holders: list[dict] = []
    waiters: list[dict] = []
    directory = gpu_dir()
    if directory.is_dir():
        for path in directory.glob("*.json"):
            doc = _load(path)
            if not doc or not pid_alive(doc.get("pid")):
                continue
            if path.name.endswith(".wait.json") or doc.get("role") == "waiter":
                waiters.append(doc)
            else:
                holders.append(doc)
    return {"holders": holders, "waiters": waiters, "events": _tail_events()}


# 评测心跳在进程退出后还会留在缓存目录。finished 或 pid 已死的不再算作正在评测。
def live_evals() -> list[dict]:
    directory = observe_dir()
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("tcenv-*.json")):
        doc = _age(_load(path))
        if not doc:
            continue
        try:
            pid = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        doc["pid"] = pid
        if doc.get("finished") or not pid_alive(pid):
            continue
        out.append(doc)
    return out


# 长实验脚本写下的是带 script 的 state.json。phase 已是 done 或 aborted 就按文件说了算；否则和单次运行一样，用心跳加 pid 区分还在跑和中断。
def experiment_jobs() -> list[dict]:
    root = runs_root()
    if not root.is_dir():
        return []
    rows = []
    for path in root.rglob("state.json"):
        state = _load(path)
        if not state or not state.get("script"):
            continue
        phase = state.get("phase") or "run"
        if phase in ("done", "aborted"):
            status = phase
        else:
            # 和 liveness 同一条规则：心跳过期且进程不在才是中断。截断的 state.json 在上面 _load 失败，这条作业不会出现。
            status = "interrupted" if time.time() - float(state.get("updated_at") or 0) > STALE_S and not pid_alive(state.get("pid")) else "running"
        rows.append({
            "id": _rel(path.parent),
            "script": state.get("script"),
            "phase": phase,
            "status": status,
            "current": state.get("current"),
            "queue": state.get("queue") or [],
            "done": state.get("done") or [],
            "updated_at": state.get("updated_at"),
        })
    rows.sort(key=lambda row: float(row.get("updated_at") or 0), reverse=True)
    return rows


# 详情页只留每个用例的对错和截短的消息。整份正确性报告太大，不进每一帧。
def _cases(report: dict) -> list[dict]:
    cases = (report.get("correctness") or {}).get("cases") or []
    out = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        out.append({"case": case.get("case"), "ok": case.get("ok"), "message": (case.get("message") or "")[:160]})
    return out


# 核源码只带开头一小段，页面用来认出是哪份核。文件读不到时留空，不让详情整页失败。
def _excerpt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")[:400]
    except OSError:
        return ""


# 详情要列出每次候选的评测结论。report.json 截断时这一格仍在，只是状态字段为空。
def _candidates(directory: Path) -> list[dict]:
    root = directory / "candidates"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        report = _load(path / "report.json") or {}
        prefix, _, label = path.name.partition("_")
        try:
            idx = int(prefix)
        except ValueError:
            # 目录名不是「序号_标签」时仍展示，只是没有序号。
            idx = None
            label = path.name
        roof = report.get("roofline") or {}
        bench = report.get("benchmark") or {}
        rows.append({
            "idx": idx,
            "label": label,
            "status": report.get("status"),
            "stage": report.get("stage"),
            "mode": report.get("mode"),
            "speedup": report.get("speedup"),
            "candidate_us": report.get("candidate_us"),
            "message": (report.get("message") or "")[:300],
            "cases": _cases(report),
            "roofline": {key: roof.get(key) for key in ("achieved_tflops", "achieved_gbps", "compute_fraction", "bandwidth_fraction", "bound")},
            "benchmark": {key: bench.get(key) for key in ("noisy", "attempts", "noise_cv")},
            "excerpt": _excerpt(path / "kernel.metal"),
        })
    return rows


# 没有心跳字母就不当作还活着。评测进度和 tcenv 文件的形状不同，这里只抽出页面要的三个字段。
def _heartbeat(progress: dict | None) -> dict | None:
    if not progress or not progress.get("heartbeat"):
        return None
    return {"letter": progress.get("heartbeat"), "age_s": progress.get("heartbeat_age_s"), "stage": progress.get("stage")}


# 传入的 rows 和历史表是同一批时，横幅不再扫一遍盘。单独调用才重新读 runs/。
def budget_report(rows: list[dict] | None = None) -> dict:
    """The banner of the page: counts of runs over budget / exhausted / near, who they are, and the total spend."""
    return summarize(history() if rows is None else rows)


# 总览的心跳来自最近启动、且仍在跑的那次搜索的 eval_progress.json；没有的话用仍活着的 tcenv 评测。
def overview() -> dict:
    from tasks import list_tasks

    hist = history()
    live = [row for row in hist if row["status"] == "running"]
    progress = None
    if live:
        directory = resolve_run(live[0]["id"])
        if directory is not None:
            progress = _age(_load(directory / "eval_progress.json"))
    evals = live_evals()
    beat = _heartbeat(progress) or (_heartbeat(evals[0]) if evals else None)
    return {
        "gpu": gpu_snapshot(),
        "live_runs": live,
        "live_evals": evals,
        "heartbeat": beat,
        "tasks": list(list_tasks(include_phase2=True)),
        "experiments": experiment_jobs(),
        "now": time.time(),
    }


# 详情把 state 和 result 叠在一起。截断的那一份是 None，另一份仍能提供轨迹、花费和停止原因。
def run_detail(run_id: str) -> dict | None:
    directory = resolve_run(run_id)
    if directory is None:
        return None
    state = _load(directory / "state.json")
    result = _load(directory / "result.json")
    trace = (state or {}).get("trace") or (result or {}).get("trace") or []
    confirmations = (state or {}).get("confirmations") or (result or {}).get("confirmations") or []
    status = liveness(state, result)
    spent, budget = money_fields(directory, state, result)
    stop = (result or {}).get("stop_reason") or (state or {}).get("stop_reason") or ""
    alert_budget, alert_stop = _budget_for_alert(directory, spent, budget, stop)
    return {
        "id": _rel(directory),
        "status": status,
        "budget": budget_status(spent, alert_budget, alert_stop, status == "running"),
        "control": control_hint(directory, state, result, status),
        **topup_fields(directory),
        "budget_events": _budget_events(directory),
        "state": state,
        "result": result,
        "trace": trace,
        "confirmations": confirmations,
        "candidates": _candidates(directory),
        "eval_progress": _age(_load(directory / "eval_progress.json")),
        "llm": _load(directory / "llm.json"),
    }


# 一帧就是页面要的全部数。查询串里的 run 已经不在 runs/ 下时，改选仍在跑的第一条，否则最近一条。
def frame(run_id: str | None) -> dict:
    hist = history()
    chosen = run_id or ""
    if chosen and resolve_run(chosen) is None:
        # 目录被删掉或路径不合法时不要把坏 id 传给详情，否则页面会停在 404。
        chosen = ""
    if not chosen:
        live = next((row for row in hist if row["status"] == "running"), None)
        chosen = live["id"] if live else (hist[0]["id"] if hist else "")
    return {
        "overview": overview(),
        "history": hist,
        "budget": budget_report(hist),
        "selected": chosen,
        "run": run_detail(chosen) if chosen else None,
    }
