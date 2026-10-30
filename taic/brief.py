"""Read-only facts for the next reader of a run. No model call and no GPU kernel.

With no directory, ``run_brief`` prints a short orientation. With a run directory it prints
one compact JSON object built only from files on disk.
"""

# 失败时先看这份摘要，而不是整份 transcript.jsonl、state.json 或看板页面。
# 只读已知文件。消息正文、内核源码、密钥都不进入输出。缺文件就是 null，不补一个数。

from __future__ import annotations

import hashlib
from collections import deque
import json
import re
import sys
from pathlib import Path

from .modelcost import Totals
from .reads import iter_jsonl, parse_jsonl, tail_text, tail_valid_rows_across
from .redact import redact_text
_REPO = Path(__file__).resolve().parent.parent

CAP = 8
_TEXT = 80
_GPU_TAIL = 262_144
_NO_KERNEL_OUTCOMES = frozenset({"agent_failed", "provider_failed", "agent_no_submission"})

ORIENTATION = """\
taic brief
Read-only. No model request, no GPU kernel, no lock taken.

Where a search starts
  uv run python -m taic.cli run TASK
  taic/cli.py cmd_run builds the config and calls Orchestrator.run in taic/orchestrator.py.
  Layers: taic.toml, then a preset (flag --preset, else TAIC_PRESET, else paper = taic.paper.toml).
  --config uses that file alone. --budget, when given, replaces budget_usd.
  uv run python -m taic.cli resume RUN_DIR continues a directory and does not resend billed requests.
  --target names a profile under targets/. The default is apple-m4-max. A new run is stored in runs/<target>/.
  uv run python -m taic.cli brief --kernels lists confirmed kernels under kernels/<target>/.
  uv run python -m taic.cli migrate-runs moves flat run directories into runs/apple-m4-max/.

Two committees
  Every search is exactly two committees, named first and second. Each has one member (provider and model).
  Another count or other names is refused before any model call. A longer member list is not reduced to the first.
  If the config has no committees, both copy the [llm] model. --model assigns that one model to both.
  The same model on both seats still starts a new session. There is no handoff sentence in TASK.md.
  The initial draft and round 1 bind first. From round 1, round r binds (r - 1) % 2. The switch is only at a round boundary.
  Each round is one new hmz session by default. search.provider_retries (0-3, default 0) adds bounded
  same-committee attempts only for a structured provider turn_failed without a submission; each attempt is billed.
  A normal no-submission answer, timeout/process failure, or available checkpoint is not retried.
  A final no-kernel outcome (agent_failed, provider_failed, agent_no_submission) hands the next session to the other committee.
  Two final no-kernel rounds stop with agent_failure. An incorrect kernel, or a file identical to the seed, clears that count.
  A session that ends without a token bill stops the whole search (cost_incomplete).
  Both share one ledger, one transcript.jsonl, and one budget. Keys and endpoints stay out of the run directory.

Evaluation
  tcenv order: screen, compile, ABI, guards and inputs, numeric check, benchmark. verify() runs first and stays unsupported.
  Evaluations inside one search take an asyncio lock in Orchestrator, so those measurements do not overlap.
  The machine lock is a flock in tcenv/gpulock.py. Timing holds it exclusively; compile and correctness hold it shared.
  Default path: ~/.cache/ai_lowerling/gpu.lock (TCENV_GPU_LOCK_FILE overrides it).
  A Metal kernel cannot be preempted and also drives the display. max_hangs is 1, so the first timeout stops the run.
  The next seed must be safe, correct, and faster by min_improvement and by noise_z standard errors.
  A within_noise candidate stays in the trace. The seed stays the previous kernel.

Publication gate
  experiments.common.conclusion_of_run decides the published speedup.
  The number is the remeasure, and only when that confirmation is bound to the same kernel and the timing is quiet.
  best_speedup stays on search_best. An empty gate leaves published null.

GPU lock cache
  Events append to ~/.cache/ai_lowerling/observe/gpu_events.jsonl.
  AI_LOWERLING_OBSERVE_DIR replaces the cache root. This command prints the last events only when that file is present.

Open next
  taic/cli.py
  taic/committees.py
  taic/orchestrator.py: _guard_shape, _one_session, _rounds, _eval, _confirm
  taic/kernel_agent.py: render_contract, run_kernel_agent
  experiments/common.py: conclusion_of_run
  tcenv/gpulock.py
  tcenv/evaluate.py
  On a failure, run: uv run python -m taic.cli brief RUN_DIR
  Read that JSON before transcript.jsonl, state.json, or the HTML dashboard.
"""


def expand_set(value) -> frozenset[str]:
    """@brief 解析 ``--expand`` 那个逗号列表，丢掉未知词。

    @param value 逗号分隔的重面板名，也可以是已经切好的集合。
    @return 只有 ``cands``/``models``/``failures`` 的集合。
    """
    if isinstance(value, (set, frozenset, list, tuple)):
        parts = [str(item) for item in value]
    else:
        parts = str(value or "").split(",")
    return frozenset(part.strip().lower() for part in parts if part.strip().lower() in EXPANDABLE)


def run_brief(run_dir: str | None, *, include_gpu: bool = False, expand="") -> int:
    """@brief 打印导览文字，或一个 JSON 对象。

    @param run_dir 运行目录；为空时打印导览。
    @param include_gpu 是否追加上机全局 GPU 锁事件尾巴。
    @param expand 逗号分隔的重面板名（``cands``/``models``/``failures``）；默认只给 L1。
    @return 退出码：成功为 0，``run_dir`` 不是目录时为 2。
    """
    if not run_dir:
        print(ORIENTATION, end="")
        return 0
    path = Path(run_dir)
    if not path.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 2
    doc = summarize_run(path, include_gpu=include_gpu, expand=expand_set(expand))
    print(json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def summarize_run(run_dir: Path, *, include_gpu: bool = False, expand: frozenset[str] = frozenset()) -> dict:
    """@brief 生成一个可直接序列化成 JSON 的摘要对象（默认是 §6.1 的 L1 形状）。

    缺文件就是 null。消息正文与内核源码不进入输出。transcript 由 ``_transcript`` 做**一次**流式遍历，
    同时产出计数、agent 失败、候选来源与模型费用（§5.5.1）；机全局的 GPU 锁尾巴是 opt-in，
    因为它属于别的运行，且随墙钟时间变化。

    默认只给 L1：``cands``（一场最多 8 条候选）、``model_totals`` 与完整的 ``agent_failures``
    改成 ``expand`` 里的 ``cands``/``models``/``failures`` 才输出；L1 里只留**最近一条**失败的
    短投影（带会话指纹），够回答“这场为什么停”。

    @param run_dir 运行目录。
    @param include_gpu 是否把机全局 GPU 锁事件带上（最多最后 8 条）。
    @param expand 需要额外输出的重面板集合（``cands``/``models``/``failures``）。
    @return 摘要字典。
    """
    run = Path(run_dir)
    state = _read_obj(run / "state.json")
    result, result_err = _load_result(run)
    docs = [d for d in (state, result) if isinstance(d, dict)]
    published, publication = _published(run, result, result_err)
    cands = _cands(run, state, result)
    transcript = _transcript(
        run / "transcript.jsonl",
        labels={row["label"] for row in cands if isinstance(row.get("label"), str) and row.get("label")},
    )
    chat, retry, truncation, transcript_stage, agent_failures = transcript["stats"]
    origins = transcript["origins"]
    for row in cands:
        if not row.get("origin") and row.get("label") in origins:
            row["origin"] = origins[row["label"]]
    doc = {
        "schema_version": "1",
        "protocol_version": _first(docs, "protocol_version"),
        "task": _first(docs, "task"),
        "dtype": _first(docs, "dtype"),
        "size": _first(docs, "size"),
        "committee": _first(docs, "committee"),
        "model": _first(docs, "model"),
        "winner_label": _first(docs, "winner_label") or _first(docs, "confirm_winner_label") or _first(docs, "best_label"),
        "winner_provider": _first(docs, "winner_provider"),
        "winner_model": _first(docs, "winner_model"),
        "winner_committee": _first(docs, "winner_committee"),
        "winner_origin": _first(docs, "winner_origin"),
        "phase": _first(docs, "phase"),
        "stop_reason": _first(docs, "stop_reason"),
        "pending_evaluation": _pending_evaluation(run),
        "round": _round_of(state, result),
        "spent_usd": _first(docs, "spent_usd"),
        "budget_usd": _first(docs, "budget_usd"),
        "published": published,
        "publication": publication,
        "search_best": _first(docs, "best_speedup"),
        # 复测与基线耗时取自确认报告本身：L1 里这两个数回答“快了多少、相对谁”。
        **_confirmed_timings(run),
        "last_failure": _last_failure(agent_failures),
        "counts": {"chat": chat, "retry": retry, "truncation_retries": truncation, "eval_stage": _eval_stage(run, transcript_stage)},
        "paths": _paths(run),
        "stamp": _stamp(run),
    }
    if "models" in expand:
        doc["model_totals"] = transcript["model_totals"]
    if "cands" in expand:
        doc["cands"] = cands
    if "failures" in expand:
        doc["agent_failures"] = agent_failures
    winner_label = doc.get("winner_label")
    if not doc.get("winner_provider") and isinstance(winner_label, str):
        found = next((row for row in cands if row.get("label") == winner_label), None)
        if found and isinstance(found.get("origin"), dict):
            origin = found["origin"]
            doc["winner_provider"] = origin.get("provider")
            doc["winner_model"] = origin.get("model")
            doc["winner_committee"] = origin.get("committee")
            doc["winner_origin"] = origin.get("origin")
    if include_gpu:
        gpu = _gpu_events()
        if gpu is not None:
            doc["gpu"] = gpu
    return doc


EXPANDABLE = ("cands", "models", "failures")
LAST_FAILURE_CHARS = 120


def _baseline_us(value):
    """@brief 基线耗时按原 schema 返回：真实文件是 ``{基线名: 微秒}`` 字典，旧文件可能是标量。

    实测 36/36 份真实 ``confirm_report.json`` 的 ``baseline_us`` 都是 ``{"mlx": …, "torch_mps": …}``，
    不是标量；把它丢成 None 会让 L1 少掉“相对谁”的一半。这里逐项保留（键名与数值都来自文件，
    不替读者挑一个“权威基线”；``benchmark.best_baseline`` 仍在 confirm 文件里）。非数值项丢掉，
    一个都不剩时返回 None。

    @param value ``confirm_report.json`` 里的 ``baseline_us``。
    @return 标量、``{名字: 微秒}`` 字典或 None。
    """
    if isinstance(value, dict):
        cleaned = {}
        for name, item in value.items():
            number = _number(item)
            if number is not None and isinstance(name, str) and name:
                cleaned[name] = number
        return cleaned or None
    return _number(value)


def _confirmed_timings(run: Path) -> dict:
    """@brief 取确认报告里的两次计时，供 L1 直接回答“快了多少、相对谁”。

    数字来自确认门复测（``confirm_report.json`` 顶层），不是搜索最优；没有确认报告时两个都是 None。
    ``candidate_us`` 是标量，``baseline_us`` 按真实 schema 是字典（见 :func:`_baseline_us`）。

    @param run 运行目录。
    @return 含 ``candidate_us`` 与 ``baseline_us`` 的字典。
    """
    # Schema 4 only (type-strict): an older confirm report has no per-batch identity and no frozen
    # expected, so its timings are not current evidence.  The L1 panel then simply has no number.
    from taic.artifacts import document_state

    confirm = _read_obj(run / "confirm_report.json")
    # Type-strict currentness (``4.0 == 4`` must not pass) and no numbers from an old report.
    if document_state(confirm) != "current":
        confirm = {}
    return {"candidate_us": _number(confirm.get("candidate_us")), "baseline_us": _baseline_us(confirm.get("baseline_us"))}


def _last_failure(failures: list[dict]) -> dict | None:
    """@brief L1 里的最近一条失败：失败类、会话指纹与截短后的脱敏错误。

    完整失败列表（最多 4 条）留给 ``--expand failures``；这一条够判断“这场为什么停”。

    @param failures ``_transcript`` 收集的失败条目（最早在前）。
    @return 短字典；没有失败时为 None。
    """
    if not failures:
        return None
    last = failures[-1]
    return {
        "label": last.get("label"),
        "outcome": last.get("outcome"),
        "failure_class": last.get("failure_class"),
        "sig": last.get("sig"),
        "error": redact_text(last.get("error") or "", limit=LAST_FAILURE_CHARS, one_line=True) or None,
    }


def _pending_evaluation(run: Path) -> dict:
    """@brief 汇总 outbox 中尚未完成的远程评测，不读取源码或秘密。"""
    root = run / "evaluation_outbox"
    rows = []
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            if path.name.endswith(".result.json") or path.name == "receipts.json":
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(document, dict) or document.get("status") not in {"pending", "unavailable"}:
                continue
            error = document.get("error") if isinstance(document.get("error"), dict) else {}
            rows.append({
                "request_id": document.get("request_id"),
                "status": document.get("status"),
                "error_code": error.get("code"),
                "retryable": error.get("retryable") is True,
            })
    return {"count": len(rows), "requests": rows[-CAP:]}


def _first(docs: list[dict], key: str):
    """@brief 按顺序取第一个带该字段的文档里的值。

    @param docs 待查文档列表，前面的优先。
    @param key 要取的字段名。
    @return 字段值；都没这个字段时为 ``None``。
    """
    for doc in docs:
        if key in doc:
            return doc[key]
    return None


def _round_of(state: dict | None, result: dict | None):
    """@brief 取当前轮次，兼容 ``round`` 与旧的 ``rounds`` 字段。

    @param state ``state.json`` 内容。
    @param result ``result.json`` 内容。
    @return 轮次值；两个文件都没有时为 ``None``。
    """
    if isinstance(state, dict) and "round" in state:
        return state["round"]
    if isinstance(result, dict) and "round" in result:
        return result["round"]
    if isinstance(result, dict) and "rounds" in result:
        return result["rounds"]
    if isinstance(state, dict) and "rounds" in state:
        return state["rounds"]
    return None


def _load_result(run: Path) -> tuple[dict | None, str | None]:
    """@brief 读 ``result.json``，并把「不可读」与「不存在」区分开。

    @param run 运行目录。
    @return ``(报告, 错误标记)``；文件不存在时为 ``(None, None)``，读不懂时为 ``(None, "unreadable")``。
    """
    path = run / "result.json"
    if not path.is_file():
        return None, None
    from experiments.common import read_result
    from taic.artifacts import NeedsInspection

    try:
        doc = read_result(run)
    except NeedsInspection:
        return None, "unreadable"
    if not isinstance(doc, dict):
        return None, "unreadable"
    return doc, None


def _published(run: Path, result: dict | None, result_err: str | None):
    """@brief 取公布层数字。

    数字来自确认门绑定的复测；门为空时公布值为 ``None``，搜索最优另算。

    @param run 运行目录。
    @param result ``result.json`` 内容。
    @param result_err ``_load_result`` 给出的错误标记。
    @return ``(公布加速比, publication 说明)``。
    """
    if result_err:
        return None, result_err
    if result is None:
        return None, None
    from experiments.common import conclusion_of_run

    got = conclusion_of_run(run, result)
    if not isinstance(got, dict):
        return None, "unreadable"
    speed = got.get("speedup")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        speed = None
    return speed, got.get("publication")


def _cands(run: Path, state: dict | None, result: dict | None) -> list[dict]:
    """@brief 列出最近的候选。

    优先用状态/结果里的 trace；没有 trace 时回退到扫描 ``candidates/`` 目录。

    @param run 运行目录。
    @param state ``state.json`` 内容。
    @param result ``result.json`` 内容。
    @return 最多 ``CAP`` 条候选摘要。
    """
    trace = _trace(state, result)
    if trace is not None:
        # A trace row is only evidence when the candidate's own report is current (§8): the trace
        # carries its own ``status``/``speedup``, so without this binding an old report's numbers
        # would come back through the trace even though the report itself is undecidable.
        by_label = _candidate_states(run)
        rows = []
        for item in trace:
            if not isinstance(item, dict):
                continue
            row = _cand_from_trace(item)
            row.update(_state_fields(by_label.get(row["label"], "missing")))
            rows.append(row)
        return rows[-CAP:]
    root = run / "candidates"
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if _cand_dir(p)]
    dirs.sort(key=lambda p: p.name)
    return [_cand_from_dir(p) for p in dirs[-CAP:]]


def _candidate_states(run: Path) -> dict[str, str]:
    """@brief 每个候选标签对应报告的当前性（``taic.artifacts.report_state``）。

    只在 trace 主路径上用：trace 自带 ``status``/``speedup``，必须与候选自己的报告绑定，
    旧格式/损坏/缺失的报告不能借 trace 把测量结论带回来。

    @param run 运行目录。
    @return ``{label: state}``；没有 ``candidates/`` 时为空表。
    """
    root = run / "candidates"
    if not root.is_dir():
        return {}
    from taic.artifacts import report_state

    states: dict[str, str] = {}
    for folder in root.iterdir():
        if not _cand_dir(folder):
            continue
        path = folder / "report.json"
        states[folder.name[4:]] = report_state(path, _read_obj(path))
    return states


def _state_fields(state: str) -> dict:
    """@brief 把报告当前性映射成候选摘要里该显示/该清空的字段。

    @param state ``report_state`` 的分类结果。
    @return 直接 ``update`` 进摘要行的字典。
    """
    if state == "current":
        return {"report_state": "current"}
    # Undecidable: the format marker replaces the status, and no measurement survives.
    return {"status": state, "speedup": None, "report_state": state}


def _trace(state: dict | None, result: dict | None) -> list | None:
    """@brief 取文档里记录的运行轨迹。

    @param state ``state.json`` 内容。
    @param result ``result.json`` 内容。
    @return ``trace`` 列表；两个文档都没有时为 ``None``。
    """
    for doc in (state, result):
        if isinstance(doc, dict) and isinstance(doc.get("trace"), list):
            return doc["trace"]
    return None


def _cand_dir(path: Path) -> bool:
    """@brief 判断目录名是否符合 ``NNN_label`` 的候选目录命名。

    @param path 候选目录。
    @return 符合命名时为 ``True``。
    """
    name = path.name
    return path.is_dir() and len(name) > 4 and name[:3].isdigit() and name[3] == "_"


def _cand_from_trace(item: dict) -> dict:
    """@brief 把轨迹里的一条记录改写成候选摘要。

    @param item 轨迹条目。
    @return 含 ``label``/``status``/``speedup``/``within_noise``/``differential_recheck`` 的摘要。
    """
    recheck = item.get("differential_recheck")
    if isinstance(recheck, dict):
        recheck = recheck.get("class")
    return {
        "label": _text(item.get("label")),
        "status": _text(item.get("status")),
        "speedup": _number(item.get("speedup")),
        "within_noise": item.get("within_noise") if isinstance(item.get("within_noise"), bool) else None,
        "differential_recheck": _text(recheck),
        "origin": _origin(item.get("origin")),
    }


def _cand_from_dir(path: Path) -> dict:
    """@brief 从候选目录里的 ``report.json`` 与 ``differential_recheck.json`` 生成摘要。

    Only a schema-4 report is current evidence (§8).  An older one still carries its own historical
    ``status``/``speedup``, but that is not a measurement this reader can check: the row says
    ``report_state="old_format"`` and drops the status/speedup instead of presenting ``ok 9.9x`` as
    if it were today's.  A damaged file is ``unreadable``, a missing one ``missing``.

    @param path 候选目录。
    @return 含 ``label``/``status``/``speedup``/``within_noise``/``differential_recheck``/``report_state`` 的摘要。
    """
    from taic.artifacts import report_state

    report_path = path / "report.json"
    report = _read_obj(report_path)
    state = report_state(report_path, report)
    current = state == "current"
    document = report if current and isinstance(report, dict) else {}
    recheck = _read_obj(path / "differential_recheck.json") or {}
    return {
        "label": _text(path.name[4:]),
        "status": _text(document.get("status")) if current else state,
        "speedup": _number(document.get("speedup")) if current else None,
        "within_noise": None,
        "differential_recheck": _text(recheck.get("class")),
        "origin": {},
        **_state_fields(state),
    }


def _origin(value) -> dict:
    """Keep only bounded, displayable origin fields from a transcript/result row."""
    if not isinstance(value, dict):
        return {}
    out = {}
    for key in ("origin", "provider", "model", "committee", "session_label"):
        text = _text(value.get(key))
        if text:
            out[key] = text
    attempt = value.get("attempt")
    if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0:
        out["attempt"] = attempt
    return out


def _transcript(path: Path, labels: set[str] | None = None) -> dict:
    """@brief 一次流式遍历 ``transcript.jsonl``：计数、agent 失败、候选来源与模型费用。

    三个消费者（统计、候选来源、模型费用）合并成一次 ``iter_jsonl`` pass：agent 行逐条喂给
    ``modelcost.Totals`` 增量聚合，**不保留事件列表**；失败只留最后 4 条（deque），来源只保留
    ``labels`` 里会被输出的候选，内存不随全部标签/失败行增长。坏行由 ``iter_jsonl`` 跳过。

    @param path ``transcript.jsonl`` 的路径。
    @param labels 需要来源的候选标签集合；None 表示保留全部（诊断用）。
    @return ``{"stats": (chat, retry, truncated, stage, failures), "origins": {...}, "model_totals": [...]}``；
        文件不存在时 ``stats`` 前四项为 None，其余为空。
    """
    if not path.is_file():
        return {"stats": (None, None, None, None, []), "origins": {}, "model_totals": []}
    from observe.failure_sig import session_failure_signature

    chat = retry = truncated = 0
    stage = None
    failures: deque = deque(maxlen=4)
    origins: dict[str, dict] = {}
    totals = Totals()
    for ev in iter_jsonl(path):
        kind = ev.get("event")
        if kind == "chat":
            chat += 1
        elif kind == "retry":
            retry += 1
        elif kind == "agent" and ev.get("outcome") in _NO_KERNEL_OUTCOMES:
            runtime = ev.get("runtime") if isinstance(ev.get("runtime"), dict) else {}
            failed_turns = runtime.get("failed_turns")
            failures.append({
                # 与 ``sessions.failure_sig`` 同源的会话指纹：索引摄取用的是同一条规则、同一份原文，
                # 所以 ``query failures --sig`` 能直接对上 brief 里的 last_failure.sig。
                "sig": session_failure_signature(
                    ev.get("outcome"), runtime.get("failure_class"), ev.get("error") or runtime.get("last_error"),
                ),
                "label": _text(ev.get("label")),
                "candidate_label": _text(ev.get("candidate_label") or ev.get("label")),
                "attempt": ev.get("attempt") if isinstance(ev.get("attempt"), int) and not isinstance(ev.get("attempt"), bool) and ev.get("attempt") >= 0 else 0,
                "provider": _text(ev.get("provider")),
                "model": _text(ev.get("model")),
                "committee": _text(ev.get("committee")),
                "outcome": _text(ev.get("outcome")),
                "failure_class": _text(runtime.get("failure_class")),
                "failed_turns": failed_turns if isinstance(failed_turns, int) and not isinstance(failed_turns, bool) and failed_turns >= 0 else None,
                "last_terminal_kind": _text(runtime.get("last_terminal_kind")),
                "error": _text(ev.get("error") or runtime.get("last_error")),
                "had_kernel": ev.get("had_kernel") if isinstance(ev.get("had_kernel"), bool) else None,
                "cost_usd": _number(ev.get("cost_usd")),
                "diagnostics_path": _text(ev.get("diagnostics_path")),
            })
        if kind == "agent":
            label = ev.get("candidate_label") or ev.get("label")
            if isinstance(label, str) and label and (labels is None or label in labels):
                origin = _origin({
                    "origin": "session", "provider": ev.get("provider"), "model": ev.get("model"),
                    "committee": ev.get("committee"), "session_label": ev.get("label"), "attempt": ev.get("attempt", 0),
                })
                if origin:
                    origins[label] = origin
            totals.add(ev)
        if ev.get("event") == "warning" and ev.get("message") == "reply truncated by max_tokens":
            truncated += 1
        elif ev.get("finish_reason") == "length":
            truncated += 1
        found = ev.get("stage")
        if isinstance(found, str):
            stage = _text(found)
    return {
        "stats": (chat, retry, truncated, stage, list(failures)),
        "origins": origins,
        "model_totals": totals.result(),
    }


def _eval_stage(run: Path, transcript_stage: str | None):
    """@brief 取当前评测阶段。

    有 ``eval_progress.json`` 就以它为准，否则退回 transcript 里的阶段。

    @param run 运行目录。
    @param transcript_stage 从 transcript 里读到的最后阶段。
    @return 阶段名；两边都没有时为 ``None``。
    """
    path = run / "eval_progress.json"
    if path.is_file():
        doc = _read_obj(path)
        if isinstance(doc, dict):
            return _text(doc.get("stage"))
        return None
    return transcript_stage


def _gpu_events() -> list[dict] | None:
    """@brief 取机全局 GPU 锁事件的最后几条。

    只读文件尾部一段，不把整份事件日志读进内存。

    @return 最多 ``CAP`` 条事件；事件文件不存在时为 ``None``。
    """
    from tcenv.events import gpu_events_path

    path = gpu_events_path()
    try:
        shards = sorted(
            (item for item in path.parent.iterdir() if item.name.startswith(path.name + ".") and item.is_file()),
            key=lambda item: (item.stat().st_mtime_ns, item.name),
        )
    except OSError:
        shards = []
    if not shards and not path.is_file():
        return None
    rows = []
    for ev in tail_valid_rows_across(shards + [path], keep=CAP, max_bytes=_GPU_TAIL):
        what = ev.get("what")
        rows.append({
            "t": _number(ev.get("t")),
            "pid": ev.get("pid") if isinstance(ev.get("pid"), int) and not isinstance(ev.get("pid"), bool) else None,
            "kind": _text(ev.get("kind")),
            "mode": _text(ev.get("mode")),
            "what": _text(what),
        })
    return rows[-CAP:]


def _best_file(run: Path) -> Path:
    """@brief 定位本次运行目标的最优内核文件。

    先按后端扩展名找 ``best<ext>``，找不到时退回通配匹配作为兼容。

    @param run 运行目录。
    @return ``best`` 文件路径（不保证存在）。
    """
    try:
        import json

        from tcenv.backends.registry import load_backend
        from tcenv.target import load_target

        doc = json.loads((run / "run_config.json").read_text())
        target_id = str((doc.get("target") or {}).get("id") or doc.get("target_id") or "")
        ext = load_backend(load_target(target_id or None).backend).file_ext or ".metal"
        found = run / f"best{ext}"
        if found.is_file():
            return found
    except Exception:
        pass
    return next(iter(sorted(run.glob("best.*"))), run / "best.metal")


def _paths(run: Path) -> dict:
    """@brief 列出四个关键文件的路径（仓库内路径用相对形式）。

    @param run 运行目录。
    @return 键为 ``result``/``confirm``/``metal``/``transcript`` 的路径字典；文件不存在时该键为 ``None``。
    """
    names = {
        "result": run / "result.json",
        "confirm": run / "confirm_report.json",
        "metal": _best_file(run),
        "transcript": run / "transcript.jsonl",
    }
    out = {}
    for key, path in names.items():
        if not path.is_file():
            out[key] = None
            continue
        try:
            out[key] = path.resolve().relative_to(_REPO).as_posix()
        except ValueError:
            out[key] = str(path.resolve())
    return out


def _stamp(run: Path) -> str:
    """@brief 对本次阅读最在意的几个文件算一个便宜的身份串。

    四个文件任一发生改变，戳记就变。

    @param run 运行目录。
    @return 16 字符十六进制摘要。
    """
    parts = []
    for name in ("state.json", "result.json", "confirm_report.json", "transcript.jsonl"):
        path = run / name
        try:
            stat = path.stat()
            parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            parts.append(f"{name}:-")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _read_obj(path: Path) -> dict | None:
    """@brief 读一个 JSON 对象文件。

    缺失、不可读或不是对象时都返回 ``None``。

    @param path 文件路径。
    @return 解析后的字典；不可用时为 ``None``。
    """
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _jsonl(path: Path, *, tail: bool = False) -> list[dict]:
    """@brief 读一份 JSONL，坏行跳过；``tail`` 只读末尾一段（用于很大的 GPU 事件日志）。

    @param path 文件路径。
    @param tail 是否只读末尾一段。
    @return 解析成功的字典列表；文件不可读时为空列表。
    """
    if tail:
        return parse_jsonl(tail_text(path, limit=_GPU_TAIL))
    return list(iter_jsonl(path))


def _tail_text(path: Path) -> str:
    """@brief 读文件末尾至多 ``_GPU_TAIL`` 字节，并丢掉第一个半行。

    实现在 :mod:`taic.reads`；``brief`` 保留这个名字给既有调用方。

    @param path 文件路径。
    @return 文本内容；文件不可读时为空字符串。
    """
    return tail_text(path, limit=_GPU_TAIL)


def _text(value) -> str | None:
    """@brief 把短字符串过一遍密钥脱敏后输出。

    带换行或超长的不收，避免把正文带进摘要。

    @param value 待处理的值。
    @return 脱敏后的短字符串；不是短字符串时为 ``None``。
    """
    if not isinstance(value, str) or "\n" in value or len(value) > _TEXT:
        return None
    return redact_text(value, limit=_TEXT)


def _number(value):
    """@brief 只接受真正的数值，bool 不算。

    @param value 待检查的值。
    @return 原值；不是数值（或为 bool）时为 ``None``。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value
