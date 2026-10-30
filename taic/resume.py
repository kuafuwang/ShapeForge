"""Continue a run from what its directory holds: after a crash, a closed window, or a stop for budget.

``python -m taic.cli resume <run_dir> [--add-budget USD | --budget USD] [--dry-run]`` rebuilds the search state from the files and goes on with the
rounds that were not finished. Nothing here spends money or touches the GPU except ``Orchestrator.run()`` after ``restore()``.

What the directory holds, and what is rebuilt from it
-----------------------------------------------------
* cost ledger        every model request is in ``transcript.jsonl`` with its usage, its cost and the running total, so the money spent so far is exact
                     (``ledger.json`` exists only when a run ended, ``state.json`` only knows the total as of its last write).
* candidates         ``candidates/NNN_<label>/{kernel.metal,report.json}``; the best kernel, the evaluation cache and the trace are rebuilt from them
                     (``state.json`` keeps the trace too; the selection is replayed and compared with it).
* finished rounds    ``state.json`` keeps the history lines and the round number. A round that already has a candidate is finished.
                     A round with no candidate starts again as a new session. The previous conversation is not continued.
* settings           ``run_config.json`` must say ``loop_protocol: flame_chase_1``. A directory without that mark is refused.
                     The system prompt is read back from ``system_prompt.txt`` when that file is there.

What cannot be recovered: a request that was in flight when the process died (the provider may have billed it, the ledger never saw it),
Instruments traces that failed (not listed), the ledger's per-request wall-clock times of older sessions (``t`` is ``null`` for them).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import control
from .artifacts import LEDGER_SCHEMA_VERSION, NeedsInspection, RESULT_SCHEMA_VERSION, STATE_SCHEMA_VERSION, read_versioned
from .policy import candidate_eligible, comparison_speedup, relation_to_best

STOP_BUDGET = "budget_exhausted"
SUGGESTED_AMOUNTS = (3.0, 5.0, 15.0)
LABEL_RE = re.compile(r"^(?P<base>r(?P<round>\d+)\.p(?P<plan>\d+))(?:\.fix(?P<fix>\d+))?(?:#t(?P<turn>\d+))?$")
CAND_DIR_RE = re.compile(r"^(?P<idx>\d+)_(?P<label>.+)$")
PROFILE_RE = re.compile(r"^r(?P<round>\d+)_(?P<label>.+)\.(?P<ext>json|txt)$")


class ResumeError(RuntimeError):
    """@brief 运行目录无法续跑。

    异常消息本身说明原因与下一步动作（补预算、换目录、重开一轮等），调用方（CLI、仪表盘）把它当作可展示的
    拒绝理由，而不是内部错误。
    """


# ---------------------------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------------------------


@dataclass
class BranchResume:
    """@brief 一个中途分支（某轮某支计划）在进程停下时所处的位置。

    字段就是持久化的分支进度：``attempt`` 是进行中的修复次数（0 = 第一次回答），``messages`` 是继续用的对话，
    ``turn`` 是最后一条 assistant 消息（带 ``pending`` 时）或下一次请求（不带时）的工具循环轮号，``pending`` 是
    还没有结果的 tool call，``reply_text`` 是已收到但尚未评测的最终答案，``quick_used`` 是本次请求已用掉的
    quick_check 次数，``note`` 说明恢复依据。
    """

    attempt: int  # repair attempt in progress (0 = the first answer)
    messages: list  # the conversation to go on from
    turn: int = 0  # tool-loop turn of the last assistant message (with ``pending``) or of the next request (without)
    pending: list = field(default_factory=list)  # tool calls of the last assistant message that have no result yet
    reply_text: str | None = None  # a final answer that had been received but not evaluated
    quick_used: int = 0  # quick_check calls this request already used
    note: str = ""


@dataclass
class RoundPartial:
    """@brief 已经进入的最后一轮；它可能已结束、只走了一半，或尚未开始。

    ``plans`` 为 None 表示还得再问一次 planner；``done`` 记录每支分支返回的候选标签（None 表示该支没交出合法
    kernel），``resume`` 记录仍需继续的分支及其 ``BranchResume``，``plan_source`` 说明计划是从哪里找回来的。
    """

    rnd: int
    seed_label: str
    plans: list | None  # None: the planner has to be asked
    done: dict = field(default_factory=dict)  # branch index -> label of the candidate it returned (None: it ended without a valid kernel)
    resume: dict = field(default_factory=dict)  # branch index -> BranchResume
    plan_source: str = ""  # how the plans were found, for the report


@dataclass
class Snapshot:
    """@brief 从运行目录重建出的完整搜索状态，供 ``Orchestrator.restore()`` 消费。

    ``requests`` 是逐次请求的账目（花费与 token 拆分），``candidates`` 是候选字典（idx、label、source、report、plan），
    ``trace`` / ``history`` 是重算后的选择痕迹与历史行。``best_label`` 是重算出的种子标签，``partial`` 只在旧协议里
    才有值。``search_finished`` 为真表示搜索阶段已经结束，本次续跑只差一次确认；``notes`` 记录重建过程中的近似与
    取舍，``checks`` 记录各来源之间的一致性核对结果。
    """

    run_dir: str
    task: str
    dtype: str
    size: str
    model: str
    requests: list
    candidates: list  # dicts: idx, label, source, report (EvalReport), plan
    trace: list
    history: list
    best_label: str | None
    rounds_done: int
    next_round: int
    stagnant: int
    partial: RoundPartial | None
    hangs: int = 0
    evaluations: int = 0
    crashes_in_a_row: int = 0
    profiles: list = field(default_factory=list)
    profile_bodies: dict = field(default_factory=dict)
    system: str | None = None
    elapsed_s: float = 0.0
    started_at: float | None = None
    initial_budget_usd: float | None = None
    resumes: int = 0
    agent_failures: int = 0
    provider_attempts: dict = field(default_factory=dict)  # candidate label -> next provider retry attempt to run
    candidate_origins: dict = field(default_factory=dict)  # candidate label -> provider/model/committee/origin
    # The phase recorded in ``state.json`` when the process stopped. ``confirm`` means the search
    # phase had ended and the resume exists to finish the remeasure, not to start another session.
    phase: str = ""
    # True when no search session is left: the run was killed during confirmation, or its last
    # round was already written and the round cap is reached. The resume then only has to confirm.
    search_finished: bool = False
    notes: list = field(default_factory=list)  # what was rebuilt, what is approximate
    checks: dict = field(default_factory=dict)  # consistency checks between the sources

    @property
    def spent_usd(self) -> float:
        """@brief 已花费的美元总额，取最后一条请求的累计值。

        @return 累计花费；尚未发生过请求时为 0.0
        """
        return float(self.requests[-1]["total_usd"]) if self.requests else 0.0


# ---------------------------------------------------------------------------------------------
# reading the directory (light: no tcenv, no openai)
# ---------------------------------------------------------------------------------------------


def _first(*values):
    """@brief 按优先级取第一个不为 None 的值。

    @param values 候选值，从左到右优先级递减
    @return 第一个非 None 值；全为 None 时返回 None
    """
    return next((v for v in values if v is not None), None)


def _doc(path: Path) -> dict | None:
    """@brief 读一个 JSON 文件，不是对象（或读不到）就当不存在。

    @param path JSON 文件路径
    @return 字典内容；文件缺失、损坏或根不是对象时为 None
    """
    d = control.read_json(path)
    return d if isinstance(d, dict) else None


def read_transcript(run_dir: Path) -> tuple[list[dict], int]:
    """@brief 读 ``transcript.jsonl``，并容忍写到一半的残行。

    进程在写入中途被杀时，最后一行往往不完整；这类行计数后跳过，不影响已完整的请求记录。
    （非 JSON 行计入损坏数，非 dict 的 JSON 行静默丢弃。）

    @param run_dir 运行目录
    @return ``(事件列表, 损坏行数)``；文件不存在时返回 ``([], 0)``
    """
    path = Path(run_dir) / "transcript.jsonl"
    events: list[dict] = []
    bad = 0
    if not path.exists():
        return events, bad
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(e, dict):
                events.append(e)
    return events, bad


def has_best(state: dict | None, result: dict | None) -> bool:
    """@brief 判断运行是否已经找到过有效 kernel。

    先看 ``best_label``，再看 ``trace`` 里是否出现过 ``improved`` 的条目；没找到任何有效 kernel 的运行没有可继续的
    基础，续跑会直接建议重开。

    @param state ``state.json`` 的内容，可为 None
    @param result ``result.json`` 的内容，可为 None
    @return 两个文档任一证明有 best 时为 True
    """
    for doc in (state, result):
        if doc and (doc.get("best_label") or any(t.get("improved") for t in doc.get("trace") or [] if isinstance(t, dict))):
            return True
    return False


def _same_pid(a, b) -> bool:
    """@brief 严格比较两个 pid（都必须是正整数且相等）。

    @param a 第一个 pid，可能是字符串或 None
    @param b 第二个 pid
    @return 可解析为相同的正整数时为 True
    """
    try:
        return int(a) > 0 and int(a) == int(b)
    except (TypeError, ValueError):
        return False


def _launching_us(holder: dict) -> bool:
    """@brief 判断锁文件是不是本次续跑自己写下的。

    锁的 ``pid`` 等于本进程或父进程时，说明它是拉起本次续跑的进程（仪表盘的 ``uv run``）留下的，属于我们自己的启动，
    而不是另一个正在跑的续跑；此时不应报“已经在续跑中”。

    @param holder ``control.lock_holder`` 读到的锁内容
    @return 是本次启动留下的锁时返回 True；``holder`` 不是 dict 时返回 False
    """
    if not isinstance(holder, dict):
        return False
    return _same_pid(holder.get("pid"), os.getpid()) or _same_pid(holder.get("pid"), os.getppid())


def _count_stop(stop: str) -> bool:
    """@brief 判断停止原因是不是“轮次计数”一类，而不是故障。

    只认两种：硬性轮数上限（``max_rounds``），或耐心用完（``converged (...)``）。只有这两类允许在仪表盘“重置”
    后被继续；显卡挂死、模型报错等停止原因不在其中。

    @param stop ``result.json`` / ``state.json`` 里的停止原因
    @return 属于轮次计数一类时为 True
    """
    return stop == "max_rounds" or stop.startswith("converged (")


def inspect_run(run_dir: Path, *, pid_check=control.pid_alive, allow_count_stops: bool = False, allow_cost_incomplete: bool = False) -> dict:
    """@brief 只看小文件，判定 ``run_dir`` 能不能续跑、以哪种方式续跑。

    仪表盘每次点击都会调一次，所以这里不读 transcript、不导入 tcenv。``kind`` 取值：
    ``running``（有活进程持有；``can_topup_live`` 说明它是否会轮询控制文件）、``resuming``（已有续跑持有锁）、
    ``interrupted``（无 result、无活进程）、``budget_exhausted``（已结束，因预算停止）、``finished``（其它停止原因）、
    ``count_limited``（轮数或耐心用完，仅在 ``allow_count_stops`` 下）、``old_protocol``（协议过期，只能读）、
    ``unknown``。

    @param run_dir 运行目录
    @param pid_check pid 存活判定函数，便于测试注入
    @param allow_count_stops 是否把轮数/耐心耗尽也视为可续（仪表盘“重置”才打开）
    @param allow_cost_incomplete 是否允许续跑 ``cost_incomplete`` 的运行；只在人确认过 epic 对账后由 CLI 打开
    @return 描述该目录的字典：``kind``、``resumable``、``reason``、``needs_budget``、花费与预算、轮次、best 加速比等
    """
    run_dir = Path(run_dir)
    try:
        state, result = read_versioned(run_dir / "state.json", STATE_SCHEMA_VERSION), read_versioned(run_dir / "result.json", RESULT_SCHEMA_VERSION)
    except NeedsInspection as e:
        info: dict = {
            "run_dir": str(run_dir), "kind": "unknown", "stop_reason": "", "phase": "", "resumable": False, "reason": str(e), "needs_budget": False,
            "can_topup_live": False, "live_reason": "", "spent_usd": None, "budget_usd": None, "initial_budget_usd": None,
            "pid": None, "round": None, "max_rounds": None, "best_speedup": None, "control_supported": False, "history": control.budget_history(run_dir, None),
            "suggested_amounts": list(SUGGESTED_AMOUNTS), "task": None, "dtype": None,
        }
        return info
    ledger = (state or {}).get("ledger") or {}
    spent = _first((result or {}).get("spent_usd"), (state or {}).get("spent_usd"), ledger.get("total_usd"))
    budget = _first((result or {}).get("budget_usd"), (state or {}).get("budget_usd"), ledger.get("budget_usd"))
    stop = (result or {}).get("stop_reason") or (state or {}).get("stop_reason") or ""
    phase = (state or {}).get("phase") or ""
    pid = (state or {}).get("pid")
    holder = control.lock_holder(run_dir)
    hist = control.budget_history(run_dir, budget)
    info: dict = {
        "run_dir": str(run_dir), "kind": "unknown", "stop_reason": stop, "phase": phase, "resumable": False, "reason": "", "needs_budget": False,
        "can_topup_live": False, "live_reason": "", "spent_usd": spent, "budget_usd": budget, "initial_budget_usd": hist["initial_budget_usd"],
        "pid": pid, "round": (state or {}).get("round"), "max_rounds": (state or {}).get("max_rounds"),
        "best_speedup": _first((result or {}).get("confirmed_speedup"), (result or {}).get("best_speedup"), (state or {}).get("best_speedup")),
        "control_supported": bool((state or {}).get("control_supported")), "history": hist, "suggested_amounts": list(SUGGESTED_AMOUNTS),
        "task": (result or state or {}).get("task"), "dtype": (result or state or {}).get("dtype"),
    }
    try:
        _check_run_config_version(run_dir)
    except NeedsInspection as e:
        info.update(kind="unknown", reason=str(e), resumable=False)
        return info
    try:
        info["needs_budget"] = (
            bool((result or {}).get("session_budget_stop"))
            or (spent is not None and budget is not None and float(spent) >= float(budget) - 1e-9)
        )
    except (TypeError, ValueError):
        pass
    if holder and not _same_pid(holder.get("pid"), pid) and not _launching_us(holder):
        info.update(kind="resuming", reason=f"already being resumed by pid {holder.get('pid')}")
        return info
    if state is None and result is None:
        info.update(reason="no state.json or result.json in this directory: not a search run")
        return info
    finished = result is not None or phase == "done"
    if not finished:
        if pid_check(pid):
            info.update(kind="running", reason=f"still running (pid {pid})")
            if info["control_supported"]:
                info["can_topup_live"] = True
            else:
                info["live_reason"] = "started by an older version that does not read the control file: wait for it to stop, then resume it"
            return info
        info["kind"] = "interrupted"
        if not has_best(state, result):
            info["reason"] = "no valid kernel was found before the process stopped: there is nothing to continue from, start a new run"
            return info
        if _refuse_old_protocol(info, run_dir):
            return info
        info["resumable"] = True
        return info
    if stop == STOP_BUDGET:
        info["kind"] = "budget_exhausted"
        if not has_best(state, result):
            info["reason"] = "stopped for budget without a valid kernel: start a new run"
            return info
        if _refuse_old_protocol(info, run_dir):
            return info
        info["resumable"] = True
        return info
    if allow_count_stops and _count_stop(stop):
        # 仪表盘「重置」才打开这条路：轮数或耐心次数用完的运行可以续，显卡挂死和模型报错仍然不行。
        info["kind"] = "count_limited"
        if not has_best(state, result):
            info["reason"] = "stopped on a round limit without a valid kernel: start a new run"
            return info
        if _refuse_old_protocol(info, run_dir):
            return info
        info["resumable"] = True
        return info
    if allow_cost_incomplete and stop.startswith("cost_incomplete"):
        # A session ended without a token bill. The CLI flag is the human acknowledgement that the
        # epic reconciliation ran; the resume itself adds the unbilled hmz spend as 'unaccounted'.
        info["kind"] = "cost_incomplete"
        if not has_best(state, result):
            info["reason"] = "stopped on an incomplete cost record without a valid kernel: start a new run"
            return info
        if _refuse_old_protocol(info, run_dir):
            return info
        info["resumable"] = True
        return info
    info["kind"] = "finished"
    why = stop or "unknown reason"
    info["reason"] = f"stopped for '{why}', not for budget: only runs that ran out of budget (or were interrupted) can be continued"
    return info


# ---------------------------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------------------------


def infer_settings(run_dir: Path, events: list[dict] | None = None) -> dict:
    """@brief 复原这次运行的设置。

    返回 ``{"search": {...}, "llm": {...}, "provenance": {key: 来源}, "system": str | None}``。
    ``run_config.json`` 存在时它是权威来源；老运行则从 ``state.json`` 与记录下来的提示词反推（例如 Skill 2b/2c 的
    是否存在、planner 要求的计划数、修复次数上限、并行初样本数），每个推导出的键都在 ``provenance`` 里标明依据。

    @param run_dir 运行目录
    @param events 已读出的 transcript 事件；None 时自行读取
    @return 设置字典，含 search / llm / provenance / system 四个键
    """
    run_dir = Path(run_dir)
    prov: dict[str, str] = {}
    cfg = _doc(run_dir / control.CONFIG_FILE)
    if events is None:
        events, _ = read_transcript(run_dir)
    chats = [e for e in events if e.get("event") == "chat"]
    system = None
    if chats:
        msgs = chats[0].get("messages") or []
        if msgs and msgs[0].get("role") == "system":
            system = msgs[0].get("content")
    if cfg:
        search = dict(cfg.get("search") or {})
        llm = dict(cfg.get("llm") or {})
        for k in search:
            prov[f"search.{k}"] = "run_config.json"
        for k in llm:
            prov[f"llm.{k}"] = "run_config.json"
        return {"search": search, "llm": llm, "provenance": prov, "system": system}
    state = _doc(run_dir / "state.json") or {}
    result = _doc(run_dir / "result.json") or {}
    search: dict = {}
    llm: dict = {}
    first_budget = _first((result.get("initial_budget_usd")), control.budget_history(run_dir)["initial_budget_usd"], result.get("budget_usd"), state.get("initial_budget_usd"), state.get("budget_usd"))
    if first_budget is not None:
        search["budget_usd"], prov["search.budget_usd"] = float(first_budget), "state.json / result.json"
    if state.get("max_rounds") is not None:
        search["max_rounds"], prov["search.max_rounds"] = int(state["max_rounds"]), "state.json"
    if state.get("model") or result.get("model"):
        llm["model"], prov["llm.model"] = state.get("model") or result.get("model"), "state.json"
    if system:
        llm["use_tools"] = "# Skill 3: MSL reference (retrieval)" in system
        prov["llm.use_tools"] = "recorded system prompt"
        # profile_poor: Skill 2b is in the system prompt of runs started after it existed; earlier runs show it in the planner feedback or in profiles/
        seen = "# Skill 2b:" in system or (run_dir / "profiles").is_dir() and any((run_dir / "profiles").iterdir()) or any(
            "Profiler evidence for this kernel" in m.get("content", "") for e in chats if str(e.get("label", "")).startswith("r") for m in e.get("messages", []) if m.get("role") == "user"
        )
        search["profile_poor"], prov["search.profile_poor"] = bool(seen), "recorded prompts / profiles/"
        search["hw_feedback"], prov["search.hw_feedback"] = "# Skill 2c:" in system, "recorded system prompt"  # Skill 2c is only in the prompt of runs started with search.hw_feedback
    # planner prompt: number of plans, and whether the last plan is the local one
    for e in chats:
        if str(e.get("label", "")).startswith("r1.plan"):
            users = [m.get("content", "") for m in e.get("messages", []) if m.get("role") == "user"]
            text = users[0] if users else ""
            m = re.search(r"propose (\d+) DIFFERENT", text)
            if m:
                search["branches"], prov["search.branches"] = int(m.group(1)), "recorded plan prompt"
            search["plan_mix"], prov["search.plan_mix"] = "LOCAL, low-risk change" in text, "recorded plan prompt"
            break
    # repair prompt: "repair attempt k/N"
    for e in chats:
        for msg in e.get("messages", []):
            m = re.search(r"repair attempt \d+/(\d+)", msg.get("content", "") if msg.get("role") == "user" else "")
            if m:
                search["max_repair_tries"], prov["search.max_repair_tries"] = int(m.group(1)), "recorded repair prompt"
                break
        if "max_repair_tries" in search:
            break
    # initial samples: the parallel first attempts are labelled init, init.s2, init.s3, ...
    labels = {str(e.get("label", "")) for e in chats}
    samples = {int(m.group(1)) for lab in labels for m in [re.match(r"^init(?:\d+)?(?:\.s(\d+))?(?:\.fix\d+)?(?:#t\d+)?$", lab)] if m and m.group(1)}
    if labels and any(l.startswith("init") for l in labels):
        search["init_samples"], prov["search.init_samples"] = (max(samples) if samples else 1), "recorded labels"
    return {"search": search, "llm": llm, "provenance": prov, "system": system}


# ---------------------------------------------------------------------------------------------
# rebuilding the state
# ---------------------------------------------------------------------------------------------


def _requests_from_transcript(events: list[dict], notes: list[str]) -> list[dict]:
    """@brief 从 transcript 重建逐次请求的账目（token 与花费）。

    ``chat`` 事件按 ``split_usage`` 拆分 token，``agent`` 事件按 ``token_counts``；累计总额取事件里记录的
    ``total_usd``，与逐条相加不符时取较大偏差并记一条 note（早前会话的 transcript 可能缺失）。
    老会话没有墙钟时间，``t`` 统一为 None。

    @param events ``read_transcript`` 读出的事件列表
    @param notes 收集重建说明与异常情况的列表，会被就地追加
    @return 请求记录列表，每项含 t / label / hit / miss / write / out / cost_usd / total_usd
    """
    from .llm import split_usage, token_counts

    reqs: list[dict] = []
    running = 0.0
    drift = 0.0
    for e in events:
        if e.get("event") == "agent":
            cost = e.get("cost_usd")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                continue
            hit, miss, write, out = token_counts(e.get("tokens"))
            running += float(cost)
            reqs.append({
                "t": None, "label": e.get("label", ""), "hit": hit, "miss": miss, "write": write, "out": out,
                "cost_usd": float(cost), "total_usd": running,
            })
            continue
        if e.get("event") != "chat":
            continue
        hit, miss, write, out = split_usage(e.get("usage") if isinstance(e.get("usage"), dict) else {})
        cost = float(e.get("cost_usd") or 0.0)
        running += cost
        recorded_total = float(e.get("total_usd", running))
        total = max(running, recorded_total)
        drift = max(drift, abs(recorded_total - running))
        running = total
        reqs.append({"t": None, "label": e.get("label", ""), "hit": hit, "miss": miss, "write": write, "out": out, "cost_usd": cost, "total_usd": total})
    if drift > 1e-6:
        notes.append(f"transcript totals differ from the sum of request costs by up to ${drift:.6f} (an earlier session's transcript is missing?): the recorded running totals are used")
    return reqs


def _label_info(label: str) -> dict | None:
    """@brief 把候选标签拆成结构化字段。

    @param label 形如 ``r3.p1.fix2#t5`` 的标签
    @return 含 base / round / plan / attempt / turn 的字典；标签不合 ``LABEL_RE`` 时返回 None
    """
    m = LABEL_RE.match(label or "")
    if not m:
        return None
    return {"base": m["base"], "round": int(m["round"]), "plan": int(m["plan"]), "attempt": int(m["fix"] or 0), "turn": int(m["turn"]) if m["turn"] is not None else None}


def _kernel_name(run_dir: Path) -> str:
    """@brief 这颗运行的候选源文件名（含后端扩展名，如 ``kernel.metal``）。

    从 ``run_config.json`` 的 target 解析后端扩展名；目标缺失或解析失败时回退到 ``.metal``，
    以便老目录（没有 target 字段）仍然能读到 kernel 文本。

    @param run_dir 运行目录
    @return 候选文件名，形如 ``kernel`` + 扩展名
    """
    try:
        from tcenv.backends.registry import load_backend
        from tcenv.target import load_target

        doc = control.read_json(Path(run_dir) / "run_config.json") or {}
        target_id = str((doc.get("target") or {}).get("id") or doc.get("target_id") or "apple-m4-max")
        ext = load_backend(load_target(target_id).backend).file_ext or ".metal"
    except Exception:
        ext = ".metal"
    return "kernel" + (ext if str(ext).startswith(".") else "." + str(ext))


def _read_candidates(run_dir: Path, notes: list[str]) -> list[dict]:
    """@brief 读出 ``candidates/*`` 里的全部候选及其报告，以及远端评测的上下文投影。

    目录名须匹配 ``<idx>_<label>`` 且含候选源文件；缺 ``report.json`` 的候选记 note 后跳过。
    测量协议与当前 ``PROTOCOL_VERSION`` 不一致时拒绝续跑：加速比只在同一协议内可比。
    ``differential_recheck.json`` 存在且标记 ``counts_as_kernel_error``（或已损坏）时，该候选视为已被判错，
    不会在续跑时回到种子位。

    远端评测的候选旁边有一份 ``evaluation_context.json`` 投影（固定 schema v1）。这里只**解析并挂载**
    它，再用 outbox 的公开只读 path helper 看那一行是否 completed（D43）：不构造 outbox、不校验
    HMAC、不重建请求。缺投影文件、缺行或行还 pending 都是**未知**上下文（``evaluation_context``
    为 ``None``、``evaluation_context_state`` 为 ``unknown``），记一条 note 而不是拒绝续跑；投影或行
    文件损坏是 ``NeedsInspection``，因为那说明有人改过它们，不是“这次没做远端评测”。

    @param run_dir 运行目录
    @param notes 收集重建说明的列表，会被就地追加
    @return 按 idx 升序的候选字典列表（idx、label、source、report、plan、recheck_rejected、
        evaluation_context、evaluation_context_state）
    @raises NeedsInspection 报告/投影/outbox 行损坏、schema 不符或测量协议与当前版本不一致时
    """
    from tcenv.evaluate import EvalReport
    from tcenv.schema import SchemaError, read_report_json

    from .artifacts import EVALUATION_CONTEXT_FILE_NAME, read_evaluation_context
    from .evaluation_outbox import row_state

    out = []
    root = Path(run_dir) / "candidates"
    kernel_name = _kernel_name(run_dir)
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        m = CAND_DIR_RE.match(d.name)
        if not m or not (d / kernel_name).exists():
            continue
        report_path = d / "report.json"
        if not report_path.exists():
            notes.append(f"candidate {d.name} has no report.json: skipped")
            continue
        try:
            rep = read_report_json(report_path.read_text(encoding="utf-8"), what=str(report_path))
        except (OSError, ValueError) as e:
            raise NeedsInspection(f"{report_path} is damaged and needs human inspection") from e
        try:
            report = EvalReport.from_json(rep)
        except SchemaError as e:
            raise NeedsInspection(f"{report_path} needs human inspection: {e}") from e
        from tcenv.schema import PROTOCOL_VERSION

        recorded_protocol = str(getattr(report, "protocol_version", "") or "")
        if recorded_protocol and recorded_protocol != PROTOCOL_VERSION:
            # 加速比只在同一个测量协议里可比。跨代续跑会把两个 epoch 的数放进同一条比较链，
            # 所以宁可拒绝，也不让旧报告参与种子选择或确认池。
            raise NeedsInspection(
                f"{report_path} was measured under protocol {recorded_protocol}, not {PROTOCOL_VERSION}: "
                "the run cannot be continued across a protocol change"
            )
        rejected = False
        pending = False
        recheck_path = d / "differential_recheck.json"
        if recheck_path.is_file():
            try:
                rec = json.loads(recheck_path.read_text(encoding="utf-8"))
                rejected = bool(isinstance(rec, dict) and rec.get("counts_as_kernel_error"))
                # 任何存在的复查记录都不是通过记录；false 只表示不把环境故障
                # 当成核错，恢复时仍必须保持未决，不能提升这颗候选。
                pending = not rejected
            except (OSError, ValueError):
                # A recheck record nobody can read is not a passed recheck: fail closed, or a
                # kernel already judged wrong would come back as a seed after the next resume.
                rejected = True
                pending = True
                notes.append(f"candidate {d.name}: differential_recheck.json is damaged; it is treated as rejected")
        # Remote evaluations leave a projection next to the candidate.  It is read here only as a
        # fixed-schema document plus a row-existence probe: the HMAC/result checks belong to
        # ``Orchestrator.restore`` -> ``EvaluationOutbox.load_completed_context``, which runs after
        # the outbox (and its signature key) exists.  A local candidate simply has no projection.
        context = read_evaluation_context(d / EVALUATION_CONTEXT_FILE_NAME)
        context_state = "local"
        if context is not None:
            # ``row_state`` reports a damaged row as NeedsInspection; a missing or pending row is
            # an unknown context, never a silent local candidate.  The projection is *not* mounted
            # in that case: a pointer without a completed row is evidence of nothing, and the next
            # resume re-reads the file and re-probes the row anyway.
            state = row_state(run_dir, context["request_id"])
            if state == "completed":
                context_state = "completed"
            else:
                context_state = "unknown"
                notes.append(
                    f"candidate {d.name}: the remote evaluation row is {state}; its evaluation context "
                    "is unknown and the candidate cannot be confirmed from it"
                )
                context = None
        out.append({
            "idx": int(m["idx"]), "label": m["label"], "source": (d / kernel_name).read_text(),
            "report": report, "plan": None, "recheck_rejected": rejected, "recheck_pending": pending,
            "evaluation_context": context, "evaluation_context_state": context_state,
        })
    out.sort(key=lambda c: c["idx"])
    return out


def _eligible(report, correctness_only: bool) -> bool:
    """@brief 判断一份报告是否算作合格候选（委托给 policy）。

    @param report ``EvalReport``，可为 None
    @param correctness_only 只看正确性、忽略加速比门槛时为 True
    @return 合格时为 True；``report`` 为 None 时返回 False
    """
    if report is None:
        return False
    return candidate_eligible(bool(report.eligible), report.speedup, correctness_only=correctness_only)


def _speedup(report, correctness_only: bool) -> float:
    """@brief 取用于比较的加速比（不合格候选按 policy 折算）。

    @param report ``EvalReport``，可为 None
    @param correctness_only 只看正确性时为 True
    @return 参与比较的加速比数值
    """
    recorded = None if report is None else report.speedup
    return comparison_speedup(_eligible(report, correctness_only), recorded)


def _replay(cands: list[dict], search) -> list[dict]:
    """@brief 在已存报告上重放编排器的选择逻辑（对应 ``Orchestrator._eval``）。

    逐个候选按 ``relation_to_best`` 判定与当前 best 的关系（replace / tie / keep）；被复查判错的候选强制为 keep，
    即使它的加速比更高。

    @param cands ``_read_candidates`` 读出的候选列表
    @param search 搜索配置（用到 ``correctness_only``、``min_improvement``、``noise_z``）
    @return 每项含 idx / label / status / speedup / best_speedup / improved / within_noise 的痕迹列表
    """
    from .orchestrator import speedup_se

    best = None
    out = []
    for c in cands:
        rep = c["report"]
        sp = _speedup(rep, search.correctness_only)
        eligible = _eligible(rep, search.correctness_only)
        best_sp = None if best is None else best["sp"]
        best_se = 0.0 if best is None else speedup_se(best["report"])
        if c.get("recheck_rejected"):
            relation = "keep"
        else:
            relation = relation_to_best(
                eligible, sp, best_sp, speedup_se(rep), best_se,
                min_improvement=search.min_improvement, noise_z=search.noise_z,
            )
        improved = relation == "replace"
        within = relation == "tie"
        if improved:
            best = {"report": rep, "sp": sp, "label": c["label"]}
        out.append({"idx": c["idx"], "label": c["label"], "status": rep.status, "speedup": rep.speedup, "best_speedup": best["sp"] if best else None, "improved": improved, "within_noise": within})
    return out



def _refuse_old_protocol(info: dict, run_dir: Path) -> bool:
    """@brief 判断目录是否早于 ``flame_chase_1``（那样 ``load_snapshot`` 会拒绝续跑）。

    目录本身仍然可读（brief、仪表盘、运行详情），只是不能显示续跑按钮。没有这一步，仪表盘会从一个
    ``taic resume`` 必然拒绝的运行里读出 ``resumable: true``。

    @param info ``inspect_run`` 正在构造的结果字典，过期时被就地改写
    @param run_dir 运行目录
    @return 协议过期时为 True（并已把 ``kind`` 改成 ``old_protocol`` 并填写 ``reason``）
    """
    protocol = _loop_protocol(run_dir)
    if protocol == "flame_chase_1":
        return False
    info.update(
        kind="old_protocol",
        reason=f"loop_protocol is {protocol!r}: this code continues only flame_chase_1 runs, so this one can be read but not resumed",
    )
    return True


def _loop_protocol(run_dir: Path) -> str | None:
    """@brief 读 ``run_config.json`` 里记录的搜索协议名。

    @param run_dir 运行目录
    @return 协议名字符串；文件缺失、字段缺失或类型不对时返回 None
    """
    doc = control.read_json(Path(run_dir) / "run_config.json") or {}
    value = doc.get("loop_protocol") if isinstance(doc, dict) else None
    return value if isinstance(value, str) else None


def _check_run_config_version(run_dir: Path) -> None:
    """Refuse a run whose file format or report epoch this code cannot continue.

    The report epoch is a run-level marker written into ``run_config.json`` before any
    evaluation or outbox row exists.  A pre-R4 run (missing, non-integer or != current epoch)
    is refused here — the first step of ``_load_flame_chase`` — so rejection happens before
    ``Orchestrator.__init__`` / ``EvaluationOutbox.__init__`` and before any replay or
    ``formal_resume``.  It never rewrites the directory.
    """
    from .artifacts import RUN_CONFIG_SCHEMA_VERSION, read_versioned
    from tcenv.schema import REPORT_SCHEMA_VERSION

    path = Path(run_dir) / "run_config.json"
    document = read_versioned(path, RUN_CONFIG_SCHEMA_VERSION)
    if not isinstance(document, dict) or document.get("loop_protocol") != "flame_chase_1":
        # A run of another loop protocol is already refused by the protocol gate (readable, not
        # continuable); the report epoch only governs runs this code would otherwise continue.
        return
    epoch = document.get("report_schema_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != REPORT_SCHEMA_VERSION:
        raise NeedsInspection(
            f"{path} has report_schema_epoch {epoch!r}, not {REPORT_SCHEMA_VERSION}; "
            "a run from another report schema is not continued"
        )


def _load_flame_chase(run_dir: Path, search, kernel_name: str) -> Snapshot:
    """@brief 重建一场 ``flame_chase_1`` 运行的状态。

    已有候选的那一轮算完成，没候选的开放轮次以一个全新会话重开；不存在需要接着用的半截对话，
    kernel 文本完全以候选文件为准（``kernel_name`` 参数未使用，只为兼容旧签名）。

    重建时以 ``transcript.jsonl`` 为计费唯一依据：``ledger.json`` 若带请求列表而与 transcript 不符会被拒绝；
    ``state.json`` 报出的花费高于 transcript 时，差额作为一条 ``unaccounted`` 记录补上，绝不低估花费。
    搜索阶段已结束的运行（phase 为 confirm，或下一轮已越过上限）会把 ``search_finished`` 置真并留在确认环节。

    @param run_dir 运行目录
    @param search 搜索配置（用到 ``max_rounds``、``min_improvement``、``noise_z``）
    @param kernel_name 候选源文件名，已弃用
    @return 重建出的 ``Snapshot``
    @raises ResumeError 目录既无 ``state.json`` 也无 ``result.json``，或 ``candidates/`` 里有目录却没有一个可读候选
    @raises NeedsInspection ``ledger.json`` 有请求列表而 transcript 没有时（需人看）
    """
    del kernel_name
    notes: list[str] = []
    _check_run_config_version(run_dir)
    state, result = read_versioned(run_dir / "state.json", STATE_SCHEMA_VERSION), read_versioned(run_dir / "result.json", RESULT_SCHEMA_VERSION)
    if state is None and result is None:
        raise ResumeError(f"{run_dir} has neither state.json nor result.json")
    events, bad = read_transcript(run_dir)
    if bad:
        notes.append(f"{bad} damaged line(s) in transcript.jsonl skipped")
    candidate_origins: dict[str, dict] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("event") != "agent":
            continue
        label = str(event.get("candidate_label") or event.get("label") or "")
        if not label:
            continue
        provider = str(event.get("provider") or "")
        model = str(event.get("model") or "")
        committee = str(event.get("committee") or "")
        if not (provider or model or committee):
            continue
        attempt = event.get("attempt")
        candidate_origins[label] = {
            "origin": "session", "provider": provider or None, "model": model or None,
            "committee": committee or None, "session_label": event.get("label"),
            "attempt": attempt if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0 else 0,
        }
    for row in (result or {}).get("candidates") or []:
        if not isinstance(row, dict) or not isinstance(row.get("origin"), dict):
            continue
        label = str(row.get("label") or "")
        if label and label not in candidate_origins:
            candidate_origins[label] = dict(row["origin"])
    ref = result or state or {}
    task, dtype, size, model = ref.get("task", ""), ref.get("dtype", ""), ref.get("size", ""), ref.get("model", "")
    requests = _requests_from_transcript(events, notes)
    ledger_doc = read_versioned(run_dir / "ledger.json", LEDGER_SCHEMA_VERSION) or {}
    if isinstance(ledger_doc.get("requests"), list) and ledger_doc["requests"]:
        if not requests:
            from .artifacts import NeedsInspection

            raise NeedsInspection(f"{run_dir / 'ledger.json'} has a request list but transcript.jsonl does not; a checkpoint cannot replace the billed transcript and needs human inspection")
        notes.append("ledger.json holds a request list; it was not applied. transcript.jsonl is the billed-request record")
    recorded_spends = [
        value
        for value in (
            (state or {}).get("spent_usd"),
            ((state or {}).get("ledger") or {}).get("total_usd"),
            (result or {}).get("spent_usd"),
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    total = float(requests[-1]["total_usd"]) if requests else 0.0
    # hmz epic 记录可能包含被杀的父进程留下的会话；它们没有 transcript 事件，只有 workspace 标签。
    # 按标签取「epic 合计 - 该标签已记账」的正差，避免把已入账的会话再加一次，也避免只取 max 时
    # 因为某些 epic 的 usage 为 0 而漏掉整段孤儿花费。
    from .kernel_agent import epic_spend_by_label

    label_costs: dict[str, float] = {}
    for row in requests:
        label = str(row.get("label") or "")
        cost = row.get("cost_usd")
        if not label or label == "unaccounted" or isinstance(cost, bool) or not isinstance(cost, (int, float)):
            continue
        label_costs[label] = label_costs.get(label, 0.0) + float(cost)
    epic_gap = 0.0
    for label, cost in (epic_spend_by_label(run_dir) or {}).items():
        epic_gap += max(0.0, float(cost) - label_costs.get(label, 0.0))
    if epic_gap > 1e-9:
        notes.append(f"hmz epics report ${epic_gap:.4f} of spend that transcript.jsonl has no agent event for (a session outlived its parent); it is carried as one 'unaccounted' entry")
        recorded_spends.append(total + epic_gap)
    from_state = max(recorded_spends) if recorded_spends else None
    if isinstance(from_state, (int, float)) and not isinstance(from_state, bool) and float(from_state) > total + 1e-6:
        gap = float(from_state) - total
        notes.append(f"a stored checkpoint or hmz epic reports ${float(from_state):.4f} but the transcript only accounts for ${total:.4f}: the ${gap:.4f} difference is carried as one 'unaccounted' entry so the spend is never under-counted")
        requests.append({"t": None, "label": "unaccounted", "hit": 0, "miss": 0, "out": 0, "cost_usd": gap, "total_usd": float(from_state)})
    cands = _read_candidates(run_dir, notes)
    cand_root = run_dir / "candidates"
    if cand_root.is_dir() and any(cand_root.iterdir()) and not cands:
        raise ResumeError("no candidates/ were stored: nothing to continue from, start a new run")
    recorded = max([d.get("trace") for d in (state, result) if d and isinstance(d.get("trace"), list)] or [[]], key=len)
    trace = [dict(t) for t in recorded if isinstance(t, dict)]
    best_label = (state or {}).get("search_seed_label") or (state or {}).get("best_label") or (result or {}).get("search_seed_label")
    rejected_labels = {c["label"] for c in cands if c.get("recheck_rejected") or c.get("recheck_pending")}

    def last_improved() -> str | None:
        # 被复查判错的候选不能回种子位，即使它在 trace 里带着 improved。
        return next((t["label"] for t in reversed(trace) if t.get("improved") and t.get("label") not in rejected_labels), None)

    if best_label and not any(c["label"] == best_label for c in cands):
        best_label = last_improved()
    if best_label is not None and best_label in rejected_labels:
        notes.append(f"search_seed_label {best_label} was rejected by its differential recheck; it is not the resumed seed")
        best_label = last_improved()
    seen = {t.get("label") for t in trace}
    for cand in cands:
        if cand["label"] in seen:
            continue
        report = cand.get("report")
        speed = getattr(report, "speedup", None)
        if getattr(report, "status", None) == "ok" and isinstance(speed, (int, float)):
            from .orchestrator import speedup_se

            best_sp = None
            best_report = None
            for earlier in cands:
                if earlier["label"] == best_label:
                    best_report = earlier.get("report")
                    break
            for row in reversed(trace):
                if row.get("label") == best_label and isinstance(row.get("speedup"), (int, float)):
                    best_sp = float(row["speedup"])
                    break
            relation = relation_to_best(
                not bool(cand.get("recheck_rejected") or cand.get("recheck_pending")), float(speed), best_sp, speedup_se(report), speedup_se(best_report),
                min_improvement=search.min_improvement, noise_z=search.noise_z,
            )
            improved = relation == "replace"
            notes.append(f"{cand['label']} was rebuilt by replaying the selection")
            if improved:
                notes.append(
                    f"search_seed_label {best_label} is older than {cand['label']} evaluated after state.json: "
                    "the resume continues from the newer kernel"
                )
                best_label = cand["label"]
            trace.append({
                "idx": cand["idx"], "label": cand["label"], "improved": improved, "within_noise": relation == "tie",
                "speedup": speed, "status": "ok", "cost_usd": total, "t_s": 0.0,
            })
            seen.add(cand["label"])
    rnd = int(_first((state or {}).get("round"), (result or {}).get("rounds"), 0) or 0)
    finished = any(c["label"] == f"r{rnd}.p0" or c["label"].startswith(f"r{rnd}.p0.") for c in cands)
    if rnd <= 0:
        next_round = 1
    elif finished:
        next_round = rnd + 1
    else:
        next_round = rnd
        notes.append(f"round {rnd} has no candidate; it starts again as a new session")
    # 搜索阶段已经结束的两种形状：state.json 里写着 confirm，或下一轮号已经越过上限而最后一轮有候选。
    # 这两种续跑只差一次确认，不该被「轮次用完」拒绝，也不该再开一场会话。
    phase = str((state or {}).get("phase") or "")
    search_finished = phase == "confirm" or (rnd > 0 and next_round > search.max_rounds)
    if search_finished:
        # 只差一次确认。轮号留在上限以内，`taic resume` 才不会被「轮次用完」拒绝；
        # _search_resumed 看到这个标志会跳过会话，直接进 _confirm。
        next_round = min(max(rnd, 0), search.max_rounds)
        notes.append("the search phase had already ended; this resume continues at the confirmation")
    history = list((state or {}).get("history") or [])
    stagnant = int((state or {}).get("stagnant") or 0)
    agent_failures = int((state or {}).get("agent_failures") or 0)
    raw_attempts = (state or {}).get("provider_attempts")
    provider_attempts = {}
    if isinstance(raw_attempts, dict):
        for label, attempt in raw_attempts.items():
            if isinstance(label, str) and label and isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0:
                provider_attempts[label] = attempt
    hangs = int(_first((state or {}).get("hangs"), 0) or 0)
    evaluations = int(_first((state or {}).get("evaluations"), len(trace)) or 0)
    crashes = int((state or {}).get("crashes_in_a_row") or 0)
    started = _first((state or {}).get("started_at"))
    elapsed = max([float(t.get("t_s") or 0.0) for t in trace] + [0.0])
    system_path = run_dir / "system_prompt.txt"
    system = system_path.read_text() if system_path.is_file() else None
    if system is None:
        notes.append("system_prompt.txt is missing; the continued run keeps the prompt this process builds")
    resumes = int(_first((state or {}).get("resumes"), 0) or 0)
    hist = control.budget_history(run_dir)
    initial = _first(hist["initial_budget_usd"], (state or {}).get("initial_budget_usd"), (state or {}).get("budget_usd"))
    return Snapshot(
        run_dir=str(run_dir), task=task, dtype=dtype, size=size, model=model, requests=requests, candidates=cands, trace=trace,
        history=history, best_label=best_label, rounds_done=max(0, rnd if search_finished else next_round - 1), next_round=next_round, stagnant=stagnant,
        partial=None, hangs=hangs, evaluations=evaluations, crashes_in_a_row=crashes, system=system, elapsed_s=elapsed,
        started_at=started, initial_budget_usd=initial, resumes=resumes, agent_failures=agent_failures,
        provider_attempts=provider_attempts, candidate_origins=candidate_origins, phase=phase, search_finished=search_finished, notes=notes,
        checks={"loop_protocol": "flame_chase_1"},
    )


def load_snapshot(run_dir: Path, search, kernel_name: str, *, settings: dict | None = None) -> Snapshot:
    """@brief 重建 ``run_dir`` 的搜索状态（只读，不花钱、不碰 GPU）。

    只继续 ``flame_chase_1``：没有这个标记的目录一律拒绝；旧的 chat-and-branches 协议不再续跑。
    真正花时间与 GPU 的只有调用方在 ``Orchestrator.restore()`` 之后的 ``run()``。

    @param run_dir 运行目录
    @param search 搜索配置
    @param kernel_name 候选源文件名（透传，当前未使用）
    @param settings 旧签名的设置字典，当前未使用
    @return 重建出的 ``Snapshot``
    @raises ResumeError ``loop_protocol`` 不是 ``flame_chase_1`` 时
    """
    del settings
    run_dir = Path(run_dir)
    protocol = _loop_protocol(run_dir)
    if protocol != "flame_chase_1":
        raise ResumeError(
            f"{run_dir} cannot be resumed: loop_protocol is {protocol!r}. This code only continues flame_chase_1. The search protocol changed."
        )
    return _load_flame_chase(run_dir, search, kernel_name)



def describe(snap: Snapshot) -> dict:
    """@brief 把重建结果压成一份可 JSON 序列化的小报告（``--dry-run`` 打印它）。

    @param snap ``load_snapshot`` 得到的快照
    @return 含花费、请求数、候选数、best 标签与加速比、完成轮数、待续分支等字段的字典
    """
    p = snap.partial
    return {
        "run_dir": snap.run_dir, "task": snap.task, "dtype": snap.dtype, "model": snap.model,
        "spent_usd": round(snap.spent_usd, 6), "requests": len(snap.requests), "candidates": len(snap.candidates), "trace_entries": len(snap.trace),
        "best_label": snap.best_label, "best_speedup": next((t["best_speedup"] for t in reversed(snap.trace) if t.get("best_speedup") is not None), None),
        "rounds_completed": snap.rounds_done, "history_lines": len(snap.history), "stagnant_rounds": snap.stagnant,
        "evaluations": snap.evaluations, "hangs": snap.hangs, "profiles": len(snap.profiles), "elapsed_s": round(snap.elapsed_s, 1),
        "system_prompt_restored": bool(snap.system),
        "continues_in_round": p.rnd if p else snap.next_round,
        "provider_attempts": dict(snap.provider_attempts),
        "candidate_origins": dict(snap.candidate_origins),
        "round": None if not p else {
            "seed": p.seed_label, "plans": [x["title"] for x in (p.plans or [])] or None, "plans_from": p.plan_source,
            "finished_branches": {f"p{i}": lab for i, lab in p.done.items()},
            "continuing_branches": {f"p{i}": {"attempt": b.attempt, "turn": b.turn, "pending_tool_calls": len(b.pending), "reply_to_evaluate": b.reply_text is not None, "why": b.note} for i, b in p.resume.items()},
            "branches_to_start": [f"p{i}" for i in range(len(p.plans or [])) if i not in p.done and i not in p.resume],
        },
        "notes": snap.notes, "checks": snap.checks,
    }


# ---------------------------------------------------------------------------------------------
# preparing the directory for the continued run
# ---------------------------------------------------------------------------------------------


def _next_index(run_dir: Path, stem: str) -> int:
    """@brief 找一个还没被占用的备份序号。

    同时检查 ``<stem>.before_topup_<n>.json`` 与 ``<stem>.before_resume_<n>.json`` 两类备份名。

    @param run_dir 运行目录
    @param stem 备份文件名前缀（``result`` 或 ``confirm_report``）
    @return 最小的未占用序号，从 1 开始
    """
    n = 1
    while any(run_dir.glob(f"{stem}.before_topup_{n}.json")) or (run_dir / f"{stem}.before_resume_{n}.json").exists():
        n += 1
    return n


def snapshot_results(run_dir: Path) -> list[str]:
    """@brief 续跑开始前把旧结果文件改成备份名，免得被覆盖。

    ``result.json`` 与 ``confirm_report.json`` 改名为 ``<name>.before_topup_<n>.json``，旧的 ``state.json`` 改名为
    ``state.before_resume_<n>.json``。文件是“移动”而不是复制，所以没有 ``result.json`` 的目录就是一场进行中的运行。

    两个文档都在移动任何东西之前先解析完。以前若 ``confirm_report.json`` 已损坏，要到 ``result.json`` 已经被移走后
    才发现，而调用方因为异常不是 ``RunStartError`` 而什么都没恢复：运行就被扫成了完全没有 ``result.json`` 的样子。

    @param run_dir 运行目录
    @return 已被移动的备份文件名列表；``state.json`` 损坏时只是不做它的备份
    @raises ResumeError ``result.json`` 或 ``confirm_report.json`` 损坏：此时一个文件都没动
    """
    run_dir = Path(run_dir)
    pending = []
    for name in ("result", "confirm_report"):
        src = run_dir / f"{name}.json"
        if not src.exists():
            continue
        try:
            pending.append((name, src, json.loads(src.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as exc:
            raise ResumeError(f"{src} is damaged and was left in place; nothing was moved: {exc}") from exc
    moved = []
    # One generation number for both files: the loop below writes result and confirm_report with
    # the same n, so n must skip a generation that either stem already uses.
    n = max(_next_index(run_dir, "result"), _next_index(run_dir, "confirm_report"))
    for name, src, doc in pending:
        dst = run_dir / f"{name}.before_topup_{n}.json"
        control.write_json_atomic(dst, doc)
        src.unlink()
        moved.append(dst.name)
    st = run_dir / "state.json"
    if st.exists():
        dst = run_dir / f"state.before_resume_{n}.json"
        try:
            control.write_json_atomic(dst, json.loads(st.read_text()))
            moved.append(dst.name)
        except ValueError:
            pass
    return moved


def restore_snapshot(run_dir: Path, moved: list[str]) -> None:
    """@brief 启动失败时把 ``result.json`` 与 ``confirm_report.json`` 放回原位。

    只处理 ``result.*`` / ``confirm_report.*`` 备份（``state.`` 开头的跳过）；目标已存在则不覆盖，
    以免把新一次运行的结果抹掉。

    @param run_dir 运行目录
    @param moved ``snapshot_results`` 返回的备份文件名列表
    """
    run_dir = Path(run_dir)
    for name in moved:
        if name.startswith("state."):
            continue
        src = run_dir / name
        stem, marker, _rest = name.partition(".before_topup_")
        if not marker or not src.is_file():
            continue
        dest = run_dir / f"{stem}.json"
        if not dest.exists():
            src.replace(dest)


def effective_budget(info: dict, *, add_usd: float | None, budget_usd: float | None, control_doc: dict | None = None) -> float:
    """@brief 算出续跑后的总预算，并确认它的确高于已花费。

    优先级：``--budget`` 给出的总数 > 已记录预算 + ``--add-budget`` > 已记录预算与 control.json 里待处理请求中较大者。
    因预算停下的运行既没给 ``--add-budget`` 也没给 ``--budget``，且没有待处理请求时直接拒绝：一次正常会话的钱不够。

    @param info ``inspect_run`` 的结果（用到 ``needs_budget``、``spent_usd``、``budget_usd``）
    @param add_usd 追加的预算美元数
    @param budget_usd 新的预算总数
    @param control_doc 控制文件内容，可为 None
    @return 新的总预算（正数）
    @raises ResumeError 无法确定预算、预算非正非有限、或不超过已花费时
    """
    if info.get("needs_budget") and add_usd is None and budget_usd is None:
        pending = control.parse_control(control_doc)[0] if control_doc else None
        if pending is None:
            raise ResumeError(
                "the run stopped with too little budget for another normal session: "
                "add budget with --add-budget USD or --budget TOTAL"
            )
    spent = float(info.get("spent_usd") or 0.0)
    recorded = info.get("budget_usd")
    recorded = float(recorded) if recorded is not None else None
    if budget_usd is not None:
        new = float(budget_usd)
    elif add_usd is not None:
        if recorded is None:
            raise ResumeError("the run does not record its budget: give the new total with --budget")
        new = recorded + float(add_usd)
    else:
        pending = control.parse_control(control_doc)[0] if control_doc else None
        new = max([x for x in (recorded, pending) if x is not None], default=None)
        if new is None:
            raise ResumeError("the run does not record its budget: give it with --budget")
    if not math.isfinite(new) or new <= 0:
        raise ResumeError("the budget must be a positive number")
    if new <= spent + 1e-9:
        raise ResumeError(f"budget ${new:.2f} does not exceed what the run has already spent (${spent:.4f}): add budget with --add-budget USD or --budget TOTAL")
    return new
