"""Read-only facts for the next reader of a run. No model call and no GPU kernel.

With no directory, ``run_brief`` prints a short orientation. With a run directory it prints
one compact JSON object built only from files on disk.
"""

# 失败时先看这份摘要，而不是整份 transcript.jsonl、state.json 或看板页面。
# 只读已知文件。消息正文、内核源码、密钥都不进入输出。缺文件就是 null，不补一个数。

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

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


def run_brief(run_dir: str | None, *, include_gpu: bool = False) -> int:
    """@brief 打印导览文字，或一个 JSON 对象。

    @param run_dir 运行目录；为空时打印导览。
    @param include_gpu 是否追加上机全局 GPU 锁事件尾巴。
    @return 退出码：成功为 0，``run_dir`` 不是目录时为 2。
    """
    if not run_dir:
        print(ORIENTATION, end="")
        return 0
    path = Path(run_dir)
    if not path.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 2
    print(json.dumps(summarize_run(path, include_gpu=include_gpu), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def summarize_run(run_dir: Path, *, include_gpu: bool = False) -> dict:
    """@brief 生成一个可直接序列化成 JSON 的摘要对象。

    缺文件就是 null。消息正文与内核源码不进入输出。transcript 只读一遍；机全局的 GPU 锁尾巴是 opt-in，
    因为它属于别的运行，且随墙钟时间变化。

    @param run_dir 运行目录。
    @param include_gpu 是否把机全局 GPU 锁事件带上（最多最后 8 条）。
    @return 摘要字典。
    """
    run = Path(run_dir)
    state = _read_obj(run / "state.json")
    result, result_err = _load_result(run)
    docs = [d for d in (state, result) if isinstance(d, dict)]
    published, publication = _published(run, result, result_err)
    chat, retry, truncation, transcript_stage, agent_failures = _transcript_stats(run / "transcript.jsonl")
    doc = {
        "schema_version": "1",
        "protocol_version": _first(docs, "protocol_version"),
        "task": _first(docs, "task"),
        "dtype": _first(docs, "dtype"),
        "size": _first(docs, "size"),
        "committee": _first(docs, "committee"),
        "model": _first(docs, "model"),
        "phase": _first(docs, "phase"),
        "stop_reason": _first(docs, "stop_reason"),
        "round": _round_of(state, result),
        "spent_usd": _first(docs, "spent_usd"),
        "budget_usd": _first(docs, "budget_usd"),
        "published": published,
        "publication": publication,
        "search_best": _first(docs, "best_speedup"),
        "cands": _cands(run, state, result),
        "agent_failures": agent_failures,
        "counts": {"chat": chat, "retry": retry, "truncation_retries": truncation, "eval_stage": _eval_stage(run, transcript_stage)},
        "paths": _paths(run),
        "stamp": _stamp(run),
    }
    if include_gpu:
        gpu = _gpu_events()
        if gpu is not None:
            doc["gpu"] = gpu
    return doc


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
        rows = [_cand_from_trace(item) for item in trace if isinstance(item, dict)]
        return rows[-CAP:]
    root = run / "candidates"
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if _cand_dir(p)]
    dirs.sort(key=lambda p: p.name)
    return [_cand_from_dir(p) for p in dirs[-CAP:]]


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
    }


def _cand_from_dir(path: Path) -> dict:
    """@brief 从候选目录里的 ``report.json`` 与 ``differential_recheck.json`` 生成摘要。

    @param path 候选目录。
    @return 含 ``label``/``status``/``speedup``/``within_noise``/``differential_recheck`` 的摘要。
    """
    report = _read_obj(path / "report.json") or {}
    recheck = _read_obj(path / "differential_recheck.json") or {}
    return {
        "label": _text(path.name[4:]),
        "status": _text(report.get("status")),
        "speedup": _number(report.get("speedup")),
        "within_noise": None,
        "differential_recheck": _text(recheck.get("class")),
    }


def _transcript_stats(path: Path) -> tuple[int | None, int | None, int | None, str | None, list[dict]]:
    """@brief 一次遍历统计对话、重试、截断、阶段与最近几场 agent 失败。

    @param path ``transcript.jsonl`` 的路径。
    @return ``(chat 数, retry 数, 截断数, 最后阶段, agent 失败摘要)``。
        文件不存在时前四项为 ``None``，失败列表为空。响应正文不进入摘要。
    """
    if not path.is_file():
        return None, None, None, None, []
    chat = retry = truncated = 0
    stage = None
    failures: list[dict] = []
    for ev in _jsonl(path):
        kind = ev.get("event")
        if kind == "chat":
            chat += 1
        elif kind == "retry":
            retry += 1
        elif kind == "agent" and ev.get("outcome") in _NO_KERNEL_OUTCOMES:
            runtime = ev.get("runtime") if isinstance(ev.get("runtime"), dict) else {}
            failed_turns = runtime.get("failed_turns")
            failures.append({
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
        if ev.get("event") == "warning" and ev.get("message") == "reply truncated by max_tokens":
            truncated += 1
        elif ev.get("finish_reason") == "length":
            truncated += 1
        found = ev.get("stage")
        if isinstance(found, str):
            stage = _text(found)
    return chat, retry, truncated, stage, failures[-4:]


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
    if not path.is_file():
        return None
    rows = []
    for ev in _jsonl(path, tail=True):
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
    """@brief 读一份 JSONL，坏行跳过。

    @param path 文件路径。
    @param tail 是否只读末尾一段（用于很大的 GPU 事件日志）。
    @return 解析成功的字典列表；文件不可读时为空列表。
    """
    try:
        if tail:
            text = _tail_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            rows.append(ev)
    return rows


def _tail_text(path: Path) -> str:
    """@brief 读文件末尾至多 ``_GPU_TAIL`` 字节，并丢掉第一个半行。

    @param path 文件路径。
    @return 文本内容；文件不可读时为空字符串。
    """
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, size - _GPU_TAIL))
        data = handle.read().decode("utf-8", errors="replace")
    if size > _GPU_TAIL:
        cut = data.find("\n")
        if cut >= 0:
            data = data[cut + 1 :]
    return data


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
