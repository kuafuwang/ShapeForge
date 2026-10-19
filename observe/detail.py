"""Turn run directories and the GPU cache into the dashboard's current frame."""

# 看板的只读面：把 runs/ 里已经落下的快照收成一帧。这里不写搜索，也不改结果文件。
# 读不出来的 JSON（写到一半、被截断）当成没有这份快照。结果缺失但 state 仍新鲜或进程还在，就当作仍在跑；state 也没有是未知，心跳过期且进程不在是中断。都不会标成已完成。

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from experiments.common import conclusion_of_run
from observe.budget import budget_status
from observe.publish import gpu_dir, gpu_events_path, observe_dir

# 超过这么久没再写 state.json，心跳才算停。还要 pid 已经不在，才会把运行标成中断。
STALE_S = 15.0
# 总览只带最近这些条 GPU 事件，一帧才跟得上页面的轮询。
EVENT_TAIL = 40
# 小于这个数的 t 是从本次搜索开始算的秒，不是墙上的 unix 时间。
_UNIX = 1_000_000_000
# 旧运行的 state.json 没有 committee 这一键。页面只留这一句，不补两个委员会。
NO_COMMITTEE_NOTE = "这次搜索没有委员会记录"
_gpu_cache: dict = {"sig": None, "events": []}

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


# 跑完优先用换种子确认的加速比；确认值缺失才退回搜索期的最优。还在跑时 result 还没有，只能看 state 里的当前最优。
def _speedup(directory: Path, state: dict | None, result: dict | None) -> float | None:
    """Headline speedup. A finished run contributes a number only when its confirmation is bound to the source on disk.
    A run that is still in progress shows the search's current best."""
    if result:
        return conclusion_of_run(directory, result)["speedup"]
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
def _as_float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _granted_budget(directory: Path) -> float | None:
    """The allowance a reset or top-up recorded, including a later cut.

    A trailing ``requested`` event counts only while a resume is still starting or has not left a dead lock.
    A failed resume (lock file, dead pid) does not keep the promised budget.
    """
    from taic import control

    applied = None
    pending = None
    for event in _budget_events(directory):
        amount = _as_float(event.get("new_budget_usd"))
        if amount is None:
            continue
        kind = event.get("kind")
        if kind in ("applied", "resume"):
            applied = amount
            pending = None
        elif kind == "requested":
            pending = amount
    if pending is None:
        return applied
    lock = directory / "resume.lock"
    if lock.exists() and control.lock_holder(directory) is None:
        return applied
    return pending


def _budget_for_alert(directory: Path, spent, budget, stop: str) -> tuple[object, str]:
    """Budget and stop reason used only for the alert.

    A grant that is still above the budget written on the result retires an old
    budget_exhausted flag: the reset was recorded and the run has not adopted it.
    A finished run whose own budget already equals that grant keeps the flag,
    even when it stopped a little under the cap.
    """
    granted = _granted_budget(directory)
    on_disk = _as_float(budget)
    shown = on_disk
    granted_n = _as_float(granted)
    spent_n = _as_float(spent)
    if granted_n is not None and (shown is None or granted_n != shown):
        shown = granted_n
    grant_not_on_result = (
        granted_n is not None
        and on_disk is not None
        and granted_n > on_disk + 1e-9
    )
    if grant_not_on_result and stop == "budget_exhausted" and spent_n is not None and shown is not None and shown > spent_n:
        return shown, ""
    return shown if shown is not None else budget, stop


# 每一行在读盘时就算好预算档位，NEAR 用的「是否在跑」和这一行的 status 是同一次判断。
# 「加预算」按钮要不要出现，只看已经读进内存的 state / result：不再多读文件。
# 运行中且进程读控制文件 → 在线加；跑完因预算停下、或进程已死且有可续的内核 → 续跑；其余给出原因。
def control_hint(directory: Path, state: dict | None, result: dict | None, status: str) -> dict:
    from taic import control
    from taic.resume import STOP_BUDGET, has_best

    stop = (result or {}).get("stop_reason") or (state or {}).get("stop_reason") or ""
    if status == "running":
        if (state or {}).get("control_supported"):
            return {"action": "topup_live", "reason": ""}
        return {"action": None, "reason": "started by an older version: it cannot be topped up live; wait for it to stop, then resume it"}
    if control.lock_holder(directory):
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


def row_from_files(directory: Path, state: dict | None, result: dict | None) -> dict:
    """One history row from files already loaded. Does not walk ``runs/``."""
    status = liveness(state, result)
    stop = (result or {}).get("stop_reason") or (state or {}).get("stop_reason") or ""
    spent, budget = money_fields(directory, state, result)
    alert_budget, alert_stop = _budget_for_alert(directory, spent, budget, stop)
    conclusion = conclusion_of_run(directory, result) if result else {"speedup": None, "publication": "in_progress"}
    return {
        "id": _rel(directory),
        "started_at": _when(directory, state),
        "task": (result or state or {}).get("task") or "",
        "dtype": (result or state or {}).get("dtype") or "",
        "phase": (state or {}).get("phase") or "",
        "stop_reason": stop,
        "speedup": conclusion["speedup"] if result else (state or {}).get("best_speedup"),
        "publication": conclusion["publication"],
        "published_speedup": conclusion["speedup"],
        "spent_usd": spent,
        "budget_usd": alert_budget,
        "disk_budget_usd": budget,
        "budget": budget_status(spent, alert_budget, alert_stop, status == "running"),
        "control": control_hint(directory, state, result, status),
        **topup_fields(directory),
        "round": (state or {}).get("round"),
        "max_rounds": (state or {}).get("max_rounds"),
        "inflight_label": (state or {}).get("inflight_label"),
        "status": status,
        "state": state,
        "result": result,
    }


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


def experiment_job(directory: Path) -> dict | None:
    """One experiment queue, when this directory's ``state.json`` is a driver script rather than a search."""
    state = _load(directory / "state.json")
    if not state or not state.get("script"):
        return None
    phase = state.get("phase") or "run"
    if phase in ("done", "aborted"):
        status = phase
    else:
        status = "interrupted" if time.time() - float(state.get("updated_at") or 0) > STALE_S and not pid_alive(state.get("pid")) else "running"
    return {
        "id": _rel(directory),
        "script": state.get("script"),
        "phase": phase,
        "status": status,
        "current": state.get("current"),
        "queue": state.get("queue") or [],
        "done": state.get("done") or [],
        "updated_at": state.get("updated_at"),
        "pid": state.get("pid"),
    }


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


# 聊天行里只留标签和时刻。messages、reply、usage 不进这一帧，提示词和密钥不会出现在时间线上。
def _chat_events(directory: Path) -> list[dict]:
    path = directory / "transcript.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    kept_keys = ("seconds", "t", "started_at", "ended_at", "at")
    out = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or item.get("event") != "chat":
            continue
        kept: dict = {}
        label = item.get("label")
        if isinstance(label, str) and label:
            kept["label"] = label[:80]
        for key in kept_keys:
            number = _as_float(item.get(key))
            if number is not None:
                kept[key] = number
        out.append(kept)
    return out


# ledger.json 里的 t 是请求结束时距账本起点的秒。转录只有耗时 seconds，没有墙上时刻时用它把条放到同一条钟上。
def _ledger_elapsed(directory: Path) -> list[dict]:
    doc = _load(directory / "ledger.json") or {}
    reqs = doc.get("requests")
    if not isinstance(reqs, list):
        return []
    out = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        elapsed = _as_float(req.get("t"))
        if elapsed is None:
            continue
        label = req.get("label") if isinstance(req.get("label"), str) else ""
        out.append({"label": label, "t": elapsed})
    return out


def _as_unix(value, origin: float | None) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    if number >= _UNIX:
        return number
    if origin is not None and number >= 0:
        return origin + number
    return None


# 一条聊天的起止。seconds 是这一次请求花了多久，不是钟上的位置。t / ended_at / at 才是时刻。
def _chat_window(chat: dict, origin: float | None) -> tuple[float, float] | None:
    duration = _as_float(chat.get("seconds"))
    if duration is not None and duration < 0:
        duration = None
    started = _as_unix(chat.get("started_at"), origin) if "started_at" in chat else None
    ended = None
    for key in ("ended_at", "at", "t"):
        if key not in chat:
            continue
        ended = _as_unix(chat.get(key), origin)
        if ended is not None:
            break
    if started is not None and ended is None and duration is not None:
        ended = started + duration
    if ended is not None and started is None and duration is not None:
        started = ended - duration
    if started is None or ended is None:
        return None
    if ended < started:
        ended = started
    return started, ended


def _take_ledger(reqs: list[dict], label: str | None, cursor: int) -> tuple[float | None, int]:
    if not label:
        if cursor >= len(reqs):
            return None, cursor
        return reqs[cursor]["t"], cursor + 1
    for index in range(cursor, len(reqs)):
        if reqs[index]["label"] == label:
            return reqs[index]["t"], index + 1
    return None, cursor


def _model_spans(directory: Path, origin: float | None) -> list[dict]:
    chats = _chat_events(directory)
    if not chats:
        return []
    ledger = _ledger_elapsed(directory)
    cursor = 0
    spans = []
    for chat in chats:
        window = _chat_window(chat, origin)
        elapsed, cursor = _take_ledger(ledger, chat.get("label"), cursor)
        if window is None and elapsed is not None and origin is not None:
            end = elapsed if elapsed >= _UNIX else origin + elapsed
            duration = _as_float(chat.get("seconds"))
            start = end - duration if duration is not None and duration >= 0 else end
            if end < start:
                end = start
            window = (start, end)
        if window is None:
            continue
        span = {"start": window[0], "end": window[1]}
        if chat.get("label"):
            span["label"] = chat["label"]
        spans.append(span)
    return spans


def _gpu_events_all() -> list[dict]:
    path = gpu_events_path()
    try:
        stat = path.stat()
    except OSError:
        return []
    sig = (str(path), stat.st_mtime_ns, stat.st_size)
    if _gpu_cache["sig"] == sig:
        return _gpu_cache["events"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and _as_float(item.get("t")) is not None:
            out.append(item)
    _gpu_cache["sig"] = sig
    _gpu_cache["events"] = out
    return out


def _token(text: str, word: str) -> bool:
    if not text or not word:
        return False
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", text) is not None


def _same_pid(event: dict, pid) -> bool:
    try:
        return int(event.get("pid")) == int(pid)
    except (TypeError, ValueError):
        return False


# 等锁条只收这一次搜索的等待：时间落在这次运行里，并且用途写着这个算子，或者 pid 就是这次搜索自己。
def _wait_belongs(event: dict, task: str, dtype: str, pid) -> bool:
    if _same_pid(event, pid):
        return True
    what = event.get("what") if isinstance(event.get("what"), str) else ""
    if not task or not _token(what, task):
        return False
    if dtype and not _token(what, dtype):
        return False
    return True


def _run_bounds(state: dict | None, result: dict | None, progress: dict | None, status: str) -> tuple[float | None, float | None]:
    start = _as_float((state or {}).get("started_at"))
    if start is None:
        start = _as_float((result or {}).get("started_at"))
    ends = []
    for doc in (state, result, progress):
        if not isinstance(doc, dict):
            continue
        for key in ("updated_at", "entered_at"):
            number = _as_float(doc.get(key))
            if number is not None and number >= _UNIX:
                ends.append(number)
    end = max(ends) if ends else None
    if status == "running":
        end = max(end or 0.0, time.time())
    elif end is None:
        end = start
    if start is not None and end is not None and end < start:
        end = start
    return start, end


def _lock_spans(state: dict | None, result: dict | None, progress: dict | None, status: str) -> list[dict]:
    task = str((result or state or {}).get("task") or "")
    dtype = str((result or state or {}).get("dtype") or "")
    pid = (state or {}).get("pid")
    start, end = _run_bounds(state, result, progress, status)
    if start is None or end is None:
        return []
    lo, hi = start - 2.0, end + 2.0
    pending: dict = {}
    spans = []

    def _emit(wait: dict, stop: float) -> None:
        begin = float(wait["t"])
        if stop < begin:
            stop = begin
        if stop < lo or begin > hi:
            return
        if not _wait_belongs(wait, task, dtype, pid):
            return
        # 同号 pid 以后会被别的进程再用。等到那一次拿到锁，会把这条等锁拉过整段空档，所以收到本次运行结束为止。
        if stop > hi:
            stop = hi
        spans.append({"start": begin, "end": stop})

    for event in sorted(_gpu_events_all(), key=lambda item: float(item["t"])):
        kind = event.get("kind")
        epid = event.get("pid")
        if kind == "wait":
            previous = pending.get(epid)
            if previous is not None:
                _emit(previous, float(previous["t"]))
            pending[epid] = event
            continue
        if kind in ("acquire", "give_up") and epid in pending:
            _emit(pending.pop(epid), float(event["t"]))
    for wait in pending.values():
        _emit(wait, float(wait["t"]))
    return spans


# 评测进度文件每次覆盖，盘上只有最近这一段。没有这份文件时这一道是空的。
def _eval_spans(progress: dict | None) -> list[dict]:
    if not progress:
        return []
    entered = _as_float(progress.get("entered_at"))
    updated = _as_float(progress.get("updated_at"))
    if entered is None and updated is None:
        return []
    start = entered if entered is not None else updated
    end = updated if updated is not None else entered
    if start is None or end is None:
        return []
    if end < start:
        end = start
    return [{"start": start, "end": end}]


def _timeline(directory: Path, state: dict | None, result: dict | None, progress: dict | None, status: str) -> dict:
    origin = _as_float((state or {}).get("started_at"))
    if origin is None:
        origin = _as_float((result or {}).get("started_at"))
    return {
        "lanes": [
            {"label": "问模型", "spans": _model_spans(directory, origin)},
            {"label": "等锁", "spans": _lock_spans(state, result, progress, status)},
            {"label": "评测", "spans": _eval_spans(progress)},
        ]
    }


_RECHECK_PASSED = {"ok", "passed", "pass"}


def _recheck_failed(value) -> bool:
    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, str):
        text = value.strip().lower()
        return bool(text) and text not in _RECHECK_PASSED
    if isinstance(value, dict):
        inner = value.get("class") if "class" in value else value.get("status")
        return _recheck_failed(inner)
    return False


# 轨迹上已经写着 within_noise 和 differential_recheck。这两句放到状态旁边，不用打开评测面板。
def _candidate_marks(trace) -> list[dict]:
    rows = []
    if not isinstance(trace, list):
        return rows
    for row in trace:
        if not isinstance(row, dict):
            rows.append({"label": None, "marks": []})
            continue
        marks = []
        if row.get("within_noise") is True:
            marks.append("噪声内没换种子")
        if _recheck_failed(row.get("differential_recheck")):
            marks.append("正确性复查没过")
        rows.append({"label": row.get("label"), "marks": marks})
    return rows


# 确认点只用发布闸门放行的那个数。闸门没放行时这里是 None，不用搜索期的 best_speedup 顶上。
def _confirmed_point(result: dict | None, trace, conclusion: dict) -> dict | None:
    if not result or "confirmation_source_sha256" not in result:
        return None
    recorded = result.get("confirmation_source_sha256")
    if not isinstance(recorded, str) or recorded == "":
        return None
    if conclusion.get("publication") != "confirmed":
        return None
    speed = conclusion.get("speedup")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        return None
    cost = _as_float(result.get("spent_usd"))
    if cost is None and isinstance(trace, list):
        for row in reversed(trace):
            if isinstance(row, dict):
                cost = _as_float(row.get("cost_usd"))
                if cost is not None:
                    break
    return {"cost_usd": cost, "speedup": speed, "label": "确认"}


def _committee_models(directory: Path) -> dict[str, str]:
    doc = _load(directory / "run_config.json") or {}
    listed = doc.get("committees")
    models: dict[str, str] = {}
    if not isinstance(listed, list):
        return models
    for item in listed:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        members = item.get("members")
        if name not in ("first", "second") or not isinstance(members, list) or not members:
            continue
        member = members[0]
        if isinstance(member, dict) and isinstance(member.get("model"), str) and member["model"].strip():
            models[name] = member["model"].strip()
    return models


def _round_number(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number >= 0 else None


def _highest_trace_round(trace) -> int:
    highest = 0
    if not isinstance(trace, list):
        return highest
    for row in trace:
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        if not isinstance(label, str):
            continue
        match = re.match(r"r(\d+)\b", label)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


# 初稿和第 1 轮是 first，之后第 r 轮是 (r - 1) % 2。没有 committee 这一键的旧运行不编交接。
def _handoff(directory: Path, state: dict | None, result: dict | None, trace) -> dict:
    source = None
    if isinstance(state, dict) and "committee" in state:
        source = state
    elif isinstance(result, dict) and "committee" in result:
        source = result
    if source is None:
        return {"recorded": False, "note": NO_COMMITTEE_NOTE, "rounds": []}
    models = _committee_models(directory)
    current_name = source.get("committee") if source.get("committee") in ("first", "second") else None
    current_model = source.get("model") if isinstance(source.get("model"), str) and source.get("model").strip() else None
    current_round = _round_number(source.get("round"))
    if current_round is None:
        current_round = _round_number((state or {}).get("round"))
    if current_round is None:
        current_round = _round_number((result or {}).get("round"))
    highest = current_round if current_round is not None else 0
    highest = max(highest, _highest_trace_round(trace))
    if current_round is None:
        current_round = highest
    rounds = []
    for rnd in range(0, highest + 1):
        name = ("first", "second")[(max(rnd, 1) - 1) % 2]
        model = models.get(name)
        if rnd == current_round and current_name == name and current_model:
            model = current_model
        rounds.append({
            "round": rnd,
            "label": "初稿" if rnd == 0 else f"第 {rnd} 轮",
            "committee": name,
            "model": model,
        })
    return {"recorded": True, "note": "", "rounds": rounds}


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
    progress = _age(_load(directory / "eval_progress.json"))
    conclusion = conclusion_of_run(directory, result) if result else {"speedup": None, "publication": "in_progress"}
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
        "publication": conclusion["publication"],
        "published_speedup": conclusion["speedup"],
        "eval_progress": progress,
        "llm": _load(directory / "llm.json"),
        "timeline": _timeline(directory, state, result, progress, status),
        "candidate_marks": _candidate_marks(trace),
        "confirmed_point": _confirmed_point(result, trace, conclusion),
        "handoff": _handoff(directory, state, result, trace),
    }

