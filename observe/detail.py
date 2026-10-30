"""Turn run directories and the GPU cache into the dashboard's current frame."""

# 看板的只读面：把 runs/ 里已经落下的快照收成一帧。这里不写搜索，也不改结果文件。
# 读不出来的 JSON（写到一半、被截断）当成没有这份快照。结果缺失但 state 仍新鲜或进程还在，就当作仍在跑；state 也没有是未知，心跳过期且进程不在是中断。都不会标成已完成。

from __future__ import annotations

import functools
import json
import os
import re
import time
from pathlib import Path

from experiments.common import conclusion_of_run
from observe.budget import budget_status
from taic.artifacts import REPORT_STATES, report_state
from taic.reads import head_text, iter_jsonl, tail_rows, tail_rows_across, tail_text
from taic.redact import redact_text
from observe.publish import gpu_dir, gpu_events_path, observe_dir

# 超过这么久没再写 state.json，心跳才算停。还要 pid 已经不在，才会把运行标成中断。
STALE_S = 15.0
# 总览只带最近这些条 GPU 事件，一帧才跟得上页面的轮询。
EVENT_TAIL = 40
# 小于这个数的 t 是从本次搜索开始算的秒，不是墙上的 unix 时间。
_UNIX = 1_000_000_000
# 旧运行的 state.json 没有 committee 这一键。页面只留这一句，不补两个委员会。
NO_COMMITTEE_NOTE = "这次搜索没有委员会记录"

REPO_ROOT = Path(__file__).resolve().parent.parent


# 结果目录可以指到别处拷来的 runs/。默认仍是仓库下的 runs，看板不创建它。
def runs_root() -> Path:
    """@brief 返回 ``runs/`` 根目录；可用 ``AI_LOWERLING_RUNS_DIR`` 指向别处拷贝的目录。看板不创建它。

    @return 运行根目录。
    """
    env = os.environ.get("AI_LOWERLING_RUNS_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "runs"


_ROOT_CACHE: dict[str, tuple[str, Path, str]] = {}


def root_path() -> tuple[Path, str]:
    """@brief 取（只解析一次的）``runs/`` 根：解析后的 Path 与规整后的字符串。

    ``Path.resolve()`` 在 macOS 上逐级 ``lstat``；一份详情里 400 颗候选会调很多次 ``_rel``。
    这里缓存解析后的根，并用字符串前缀剥离算相对路径，不再逐个目录 ``realpath``。
    原始英文说明：The resolved runs root, cached by the raw ``AI_LOWERLING_RUNS_DIR`` value.

    @return ``(resolved Path, normpath 文本)``。
    """
    raw = os.fspath(runs_root())
    cached = _ROOT_CACHE.get("root")
    if cached is not None and cached[0] == raw:
        return cached[1], cached[2]
    resolved = Path(raw).resolve()
    text = os.path.normpath(os.fspath(resolved))
    _ROOT_CACHE["root"] = (raw, resolved, text)
    return resolved, text


# state.json 里的 pid 可能已经过期。信号 0 只问进程还在不在：权限不够说明进程在，查找失败说明已经退出。
def pid_alive(pid: object) -> bool:
    """@brief 用信号 0 判断一个 pid 是否还活着；权限不够也算活着。

    @param pid 进程 id，可为任意类型。
    @return 进程仍在时为 True；pid 非法或已退出时返回 False。
    """
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
    """@brief 宽容地读取 JSON 对象；半截 JSON、空文件或根节点不是对象时返回 None。

    @param path JSON 文件路径。
    @return 解析后的字典；读不了或格式不对时返回 None。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # 文件不存在和写到一半都进这里。截断的 result.json 因此不会被当成一次已经完成的结果。
        return None
    return data if isinstance(data, dict) else None


# 心跳年龄在读的这一刻才算。文件里只存绝对时间，隔很久打开看板时不会把旧年龄当成现在。
def _age(doc: dict | None) -> dict | None:
    """@brief 在读取的这一刻补上 ``heartbeat_age_s``，避免旧年龄被当成现在。

    @param doc 心跳快照，可为 None。
    @return 补了心跳年龄的新字典；``doc`` 为空时返回 None。
    """
    if not doc:
        return None
    out = dict(doc)
    at = out.get("heartbeat_at")
    if isinstance(at, (int, float)):
        out["heartbeat_age_s"] = round(max(0.0, time.time() - float(at)), 1)
    return out


# 只接受 runs/ 里面的相对目录，避免查询串用 .. 读到仓库外。两份快照都不存在，就不是一次搜索。
def resolve_run(run_id: str, *, root: Path | None = None) -> Path | None:
    """@brief 把 run id 解析为 ``runs/`` 下确实含有搜索快照的目录，拒绝任何越界路径。

    原始英文说明：A directory under ``runs/`` that holds a search snapshot. Rejects anything that climbs out.

    @param run_id 相对 ``runs/`` 的运行 id。
    @param root 已解析的 ``runs/`` 根；索引器传入缓存值，避免每帧重复 ``realpath``。
    @return 解析后的目录；越界、不存在或两份快照都没有时返回 None。
    """
    if not run_id or "\x00" in run_id:
        return None
    rel = Path(run_id)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    root = Path(root) if root is not None else root_path()[0]
    try:
        path = (root / rel).resolve()
    except OSError:
        return None
    if path != root and root not in path.parents:
        return None
    try:
        if not path.is_dir():
            return None
    except OSError:
        # 名字超过文件系统上限（ENAMETOOLONG）之类的情况：这条 run 不可能在盘上存在。
        return None
    if not (path / "state.json").exists() and not (path / "result.json").exists():
        # 这里只看文件在不在。内容截断仍算「有这份运行」，解析失败后面另当缺失快照。
        return None
    return path


# 取目标身份要整读 ``run_config.json`` 与目标档案。面板 stamp 只允许 stat：同一目录在一次请求里
# 会被多个面板问到，所以按 ``(mtime_ns,size)`` 记住上一次的解析结果，配置改了键就变。
# 每个 run 一条 (目录, run_config 指纹) 记录，约 200B。/best 与面板 stamp 在每个请求里都要问它，
# 上限太小会在几千个已确认运行上反复逐出、把 stat 变成读文件。
@functools.lru_cache(maxsize=8192)
def _target_identity_cached(directory: str, signature: str) -> tuple[str, str, str]:
    """@brief ``target_identity`` 的有缓存版本；缓存的键是目录与 ``run_config.json`` 的 stat。

    @param directory 运行目录的文本路径。
    @param signature ``run_config.json`` 的 ``(mtime_ns,size)`` 指纹。
    @return 三元组：目标 id、后端 id 与内核文件扩展名。
    """
    return _target_identity_uncached(Path(directory))


def _config_signature(directory: Path) -> str:
    """@brief 取 ``run_config.json`` 的 ``(mtime_ns,size)`` 指纹，不改动文件。

    @param directory 运行目录。
    @return 指纹字符串；文件不存在时为 ``0:0``。
    """
    try:
        stat = (Path(directory) / "run_config.json").stat()
    except OSError:
        return "0:0"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


# 页面上的运行 id 是 runs/ 下的相对路径，详情接口再用 resolve_run 认回来。
def target_identity(directory: Path) -> tuple[str, str, str]:
    """@brief 取一个运行的 ``(target_id, backend_id, file_ext)``，旧运行回退到 Metal。

    原始英文说明：``(target_id, backend_id, file_ext)`` for a run; legacy Metal fallback.

    @param directory 运行目录。
    @return 三元组：目标 id、后端 id 与内核文件扩展名（带点）。
    """
    return _target_identity_cached(str(Path(directory)), _config_signature(directory))


def _target_identity_uncached(directory: Path) -> tuple[str, str, str]:
    """@brief 现读 ``run_config.json`` 与目标档案算出目标身份。

    @param directory 运行目录。
    @return 三元组：目标 id、后端 id 与内核文件扩展名（带点）。
    """
    config = _load(Path(directory) / "run_config.json") or {}
    target = config.get("target") if isinstance(config.get("target"), dict) else {}
    target_id = str(target.get("id") or config.get("target_id") or Path(directory).parent.name or "")
    backend_id = str(target.get("backend") or "")
    ext = ".metal"
    try:
        from tcenv.backends.registry import load_backend
        from tcenv.target import load_target

        profile = load_target(target_id or None)
        backend_id = backend_id or profile.backend
        ext = load_backend(profile.backend).file_ext or ".metal"
    except Exception:
        pass
    if not str(ext).startswith("."):
        ext = "." + str(ext)
    return target_id, backend_id, ext


def target_file_names(directory: Path) -> tuple[str, str]:
    """@brief 取该运行目标对应的 ``(内核文件名, 最优文件名)``，旧运行回退到 Metal。

    原始英文说明：``(kernel file, best file)`` for the run's target; legacy Metal fallback.

    @param directory 运行目录。
    @return 二元组：``kernel<ext>`` 与 ``best<ext>``。
    """
    _target_id, _backend_id, ext = target_identity(directory)
    return f"kernel{ext}", f"best{ext}"


def candidate_kernel_files(candidates_dir: Path) -> list[str]:
    """@brief 列出一颗候选目录里所有 ``kernel.*`` 文件名；只列目录，不读配置。

    面板 stamp 只允许 stat，而权威文件名要读 ``run_config.json``。只取“字典序第一个”会在
    混入 ``kernel.aaa`` 时选错文件；stamp 覆盖全部 ``kernel.*``，payload 另按 target 选权威文件。

    @param candidates_dir 一颗候选目录。
    @return 文件名列表，按名字排序。
    """
    try:
        return sorted(
            entry.name for entry in os.scandir(candidates_dir)
            if entry.name.startswith("kernel.") and entry.is_file()
        )
    except OSError:
        return []


def candidate_report_state(report_path: Path) -> str:
    """@brief 读一次 ``report.json`` 并返回它的当前性（不返回报告本身）。

    @param report_path ``report.json`` 路径。
    @return ``report_state`` 的分类结果。
    """
    return report_state(report_path, _load(report_path))


def candidate_row(directory: Path, path: Path) -> dict:
    """@brief 把一颗候选目录读成面板/详情的一行（报告字段 + 权威内核片段）。

    内核片段按 target 的权威文件名取（``target_file_names``），不按字典序猜。``report_state`` 与
    ``build`` 只在 schema-4 报告上出现；旧格式报告标成 ``old_format`` 且不产生 build/评测身份。

    @param directory 运行目录。
    @param path 候选目录。
    @return 候选行字典。
    """
    report_path = path / "report.json"
    report_document = _load(report_path)
    state = report_state(report_path, report_document)
    current = state == "current"
    # Only a current (schema-4) report is evidence.  An old/damaged one contributes no status,
    # speedup, cases or roofline: the row says *which* state it is instead of quoting a measurement
    # this reader cannot check (the raw bytes stay behind the explicit ``/raw`` route).
    report = report_document if current and isinstance(report_document, dict) else {}
    raw_build = report.get("build")
    raw_build = raw_build if isinstance(raw_build, dict) else {}
    prefix, _, label = path.name.partition("_")
    try:
        idx = int(prefix)
    except ValueError:
        # 目录名不是「序号_标签」时仍展示，只是没有序号。
        idx = None
        label = path.name
    roof = report.get("roofline") or {}
    bench = report.get("benchmark") or {}
    kernel = target_file_names(directory)[0]
    return {
        "idx": idx,
        "label": label,
        "status": report.get("status") if current else state,
        "stage": report.get("stage"),
        "mode": report.get("mode"),
        "speedup": report.get("speedup"),
        "candidate_us": report.get("candidate_us"),
        "message": (report.get("message") or "")[:300],
        "cases": _cases(report),
        "roofline": {key: roof.get(key) for key in ("achieved_tflops", "achieved_gbps", "compute_fraction", "bandwidth_fraction", "bound")},
        "benchmark": {key: bench.get(key) for key in ("noisy", "attempts", "noise_cv")},
        # Schema 4 only: an old report has no build identity and must not look like it does.
        "report_state": state,
        "build": (
            {
                "shape": raw_build.get("shape"),
                "build_kind": raw_build.get("build_kind"),
                "native_digest": raw_build.get("native_digest"),
                "artifact_set_digest": raw_build.get("artifact_set_digest"),
            }
            if current else {}
        ),
        "evaluation_fingerprint": report.get("evaluation_fingerprint") if current else None,
        "excerpt": _excerpt(path / kernel) if (path / kernel).is_file() else "",
    }


def _rel(path: Path) -> str:
    """@brief 取运行目录相对 ``runs/`` 根目录的 POSIX 路径。

    @param path 运行目录。
    @return 相对路径文本，用作页面上的运行 id。
    """
    text = os.fspath(path)
    if not os.path.isabs(text):
        text = os.fspath(Path(text).resolve())
    text = os.path.normpath(text)
    _, root = root_path()
    if text == root:
        return ""
    prefix = root + os.sep
    if not text.startswith(prefix):
        raise ValueError(f"{text} is not inside {root}")
    return text[len(prefix):].replace(os.sep, "/")


# 完成先看 result 能否读成对象，或 state 的 phase 已经是 done。结果缺失时：15 秒内有心跳或 pid 还在，当作仍在跑；心跳过期且进程不在是中断；连 state 都没有是未知。
def liveness(state: dict | None, result: dict | None, directory: Path | None = None) -> str:
    """@brief 判定一个运行是在跑、已完成、已中断还是未知。

    完成看 result 能否读成对象，或 state 的 phase 已是 done。结果缺失时：15 秒内有心跳或 pid 还在算仍在跑；
    心跳过期且进程不在是中断；连 state 也没有是未知。

    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @param directory 运行目录，用于检查续跑锁，可为 None。
    @return ``running`` / ``done`` / ``interrupted`` / ``unknown``。
    """
    if directory is not None:
        from taic import control as run_control

        holder = run_control.lock_holder(directory)
        if holder and run_control.pid_alive(holder.get("pid")):
            # 续跑已经拿锁，result.json 可能还没挪走。这段时间仍是在跑，不能当成可以删的已完成记录。
            return "running"
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
    """@brief 取头条加速比；已完成的运行只给绑定到磁盘源码的确认值，进行中则给搜索当前最优。

    原始英文说明：Headline speedup. A finished run contributes a number only when its confirmation is bound to the source on disk.
    A run that is still in progress shows the search's current best.

    @param directory 运行目录。
    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @return 加速比；无数据时返回 None。
    """
    if result:
        return conclusion_of_run(directory, result)["speedup"]
    if state:
        return state.get("best_speedup")
    return None


# 排序用的开始时间。state 里没有 started_at 时退回文件修改时间，旧运行仍能按先后出现。
def _when(directory: Path, state: dict | None) -> float:
    """@brief 取排序用的开始时间；没有 ``started_at`` 时退回文件修改时间。

    @param directory 运行目录。
    @param state 解析后的 ``state.json``，可为 None。
    @return unix 时间戳；读不到时返回 0.0。
    """
    if state and isinstance(state.get("started_at"), (int, float)):
        return float(state["started_at"])
    try:
        return (directory / "state.json").stat().st_mtime if (directory / "state.json").exists() else directory.stat().st_mtime
    except OSError:
        return 0.0


# 后面的空值不能盖掉前面已经写上的数。旧运行往往只有其中一份快照带花费。
def _first(*values):
    """@brief 返回第一个非 None 的值；后面的空值不会盖掉前面已有的数。

    @param values 按优先级排列的候选值。
    @return 第一个非 None 的值；全为 None 时返回 None。
    """
    return next((value for value in values if value is not None), None)


# 花费和预算按 result、state、state 里的账本、ledger.json 这个顺序补。缺的留空，横幅才能走 unknown 而不是当成 0。
def money_fields(directory: Path, state: dict | None, result: dict | None) -> tuple[object, object]:
    """@brief 按 result、state、state 内账本、``ledger.json`` 的顺序取（已花费, 预算），缺失留空。

    原始英文说明：(spent USD, budget USD) of a run, from the best source that has them: result.json, state.json, the ledger inside state.json,
    ledger.json. Older runs lack some of these; a missing value stays ``None``.

    @param directory 运行目录。
    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @return 二元组：已花费美元与预算美元，未知为 None。
    """
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
    """@brief 把数字转成有限浮点数；布尔、非数字、NaN 与无穷返回 None。

    @param value 待转换的值。
    @return 有限浮点数或 None。
    """
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
    """@brief 取重置或加预算已答应的额度，包括之后的削减。

    末尾的 ``requested`` 事件只在续跑仍处于启动中、或尚未留下死锁时计数；失败的续跑（有锁文件但 pid 已死）
    不再保留承诺的预算。
    原始英文说明：The allowance a reset or top-up recorded, including a later cut.

    @param directory 运行目录。
    @return 已答应或待生效的预算额度；没有记录时返回 None。
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
    """@brief 只给告警用的预算与停止原因。

    仍高于 result 上预算的授予会撤销旧的 ``budget_exhausted`` 标记（重置已记录但运行尚未采纳）；
    已结束的运行若自己的预算就等于该授予，则保留标记，即使它略低于上限就停了。
    原始英文说明：Budget and stop reason used only for the alert.

    @param directory 运行目录。
    @param spent 已花费金额。
    @param budget 磁盘上的预算。
    @param stop 磁盘上的停止原因。
    @return 二元组：用于告警的预算（可能为授予值）与要展示的停止原因（撤销时为空串）。
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
    """@brief 判断一个运行能不能加预算，以及如何加。

    运行中且进程读控制文件 → 在线加；因预算停下、或进程已死但留有可续的内核 → 续跑；其余给出原因。

    @param directory 运行目录。
    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @param status ``liveness`` 给出的状态。
    @return 含 ``action``（``topup_live`` / ``resume`` / None）与 ``reason`` 的字典。
    """
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
    """@brief 取“是否已加过预算”与首次预算，供页面区分固定预算的运行。

    @param directory 运行目录。
    @return 含 ``topped_up`` 与 ``initial_budget_usd`` 的字典。
    """
    if not (directory / "budget_events.json").exists():
        return {"topped_up": False, "initial_budget_usd": None}
    from taic import control

    hist = control.budget_history(directory)
    return {"topped_up": hist["topped_up"], "initial_budget_usd": hist["initial_budget_usd"]}


def _budget_events(directory: Path) -> list[dict]:
    """@brief 读取 ``budget_events.json`` 里的事件；文件不存在时为空列表。

    @param directory 运行目录。
    @return 预算事件字典列表。
    """
    if not (directory / "budget_events.json").exists():
        return []
    from taic import control

    return control.read_events(directory)


PENDING_EVAL_REQUESTS = 8


def _pending_requests(directory: Path) -> tuple[list[dict], float | None]:
    """@brief 扫一遍 outbox，列出 pending/unavailable 的请求（按文件名顺序）与最新文件 mtime。

    ``receipts.json`` 与 ``*.result.json`` 是回执与结果，不算待评测请求；坏 JSON 跳过。
    mtime 作为窄投影的 ``updated_at``：它来自磁盘，删库重建后得到同一个值（不用 ``time.time()``）。
    原始英文说明：One scan of the outbox for the pending/unavailable requests that still need an evaluation.

    @param directory 运行目录。
    @return 二元组：请求字典列表（未截断）与最新文件 mtime（秒）；没有 outbox 时为 None。
    """
    root = directory / "evaluation_outbox"
    requests = []
    newest = None
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
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = None
            if mtime is not None and (newest is None or mtime > newest):
                newest = mtime
            error = document.get("error") if isinstance(document.get("error"), dict) else {}
            requests.append({"request_id": document.get("request_id"), "status": document.get("status"), "error_code": error.get("code"), "retryable": error.get("retryable") is True})
    return requests, newest


def pending_evaluation(directory: Path) -> dict:
    """@brief 读取 outbox 的 pending/unavailable 计数与最近请求，供看板只读展示。

    形状是 ``{"count": n, "requests": [...最多 8 条...]}``；索引把它窄投影进 ``run_pending_eval``。

    @param directory 运行目录。
    @return 计数与最近 8 条请求。
    """
    requests, _newest = _pending_requests(directory)
    return {"count": len(requests), "requests": requests[-PENDING_EVAL_REQUESTS:]}


def pending_evaluation_table(directory: Path) -> dict:
    """@brief 摄入用的 outbox 窄投影：聚合计数、最后一条错误码与最近 8 条请求。

    与 :func:`pending_evaluation` 读同一批文件，但保留 ``pending``/``unavailable`` 的**全量**计数
    （旧形状只有总数），供 ``run_pending_eval`` 表逐字段对拍。``updated_at`` 取最新请求文件的 mtime。

    @param directory 运行目录。
    @return 含 ``pending``/``unavailable``/``last_error_code``/``retryable``/``requests``/``updated_at`` 的字典。
    """
    requests, newest = _pending_requests(directory)
    kept = requests[-PENDING_EVAL_REQUESTS:]
    last = kept[-1] if kept else {}
    return {
        "pending": sum(1 for item in requests if item.get("status") == "pending"),
        "unavailable": sum(1 for item in requests if item.get("status") == "unavailable"),
        "last_error_code": last.get("error_code"),
        "retryable": last.get("retryable") is True,
        "requests": kept,
        "updated_at": newest,
    }


# ``query runs`` 的 L0 行用机器码而不是完整停止原因：实测最长一条 stop_reason 有 224 字符
# （一条 HTTP 402 provider 错误），原样放进窄行必然超出 200B 预算。完整文本留在 ``--wide``
# 与详情里。
STOP_CODE_CHARS = 24


def stop_code(stop_reason) -> str:
    """@brief L0 的停止码：停止原因里冒号前的第一段，再取其中的第一个词，最多 24 字符。

    “哪一类停止”就够机器筛选了（``budget_exhausted``/``llm_error``/``converged``）。取第一个词
    是因为 ``converged (12 rounds without improvement)`` 这类文本里轮数会变：按字符截断会把同一类
    停止切成 ``converged (12 rounds wit`` 与 ``converged (4 rounds with`` 两个码，机器筛不出来。
    完整文本仍留在 ``--wide`` 与详情里。

    @param stop_reason 原始的停止原因，可为 None。
    @return 短的机器码；没有停止原因时为空串。
    """
    segment = str(stop_reason or "").split(":", 1)[0].strip()
    words = segment.split()
    return (words[0] if words else "")[:STOP_CODE_CHARS]


def row_from_files(directory: Path, state: dict | None, result: dict | None) -> dict:
    """@brief 用已加载的文件拼出一行历史行，不遍历 ``runs/``。

    原始英文说明：One history row from files already loaded. Does not walk ``runs/``.

    @param directory 运行目录。
    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @return 含状态、花费、预算、发布结果与控制提示的历史行。
    """
    status = liveness(state, result, directory)
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
        # 窄码随 S1c 的窄投影一起写在摄入时（不靠读时截断），L0 行只 SELECT。
        "stop_code": stop_code(stop),
        "speedup": conclusion["speedup"] if result else (state or {}).get("best_speedup"),
        "publication": conclusion["publication"],
        "published_speedup": conclusion["speedup"],
        "spent_usd": spent,
        "budget_usd": alert_budget,
        "disk_budget_usd": budget,
        "budget": budget_status(spent, alert_budget, alert_stop, status == "running"),
        "control": control_hint(directory, state, result, status),
        "pending_evaluation": pending_evaluation(directory),
        **topup_fields(directory),
        "round": (state or {}).get("round"),
        "max_rounds": (state or {}).get("max_rounds"),
        "inflight_label": (state or {}).get("inflight_label"),
        "status": status,
        "state": state,
        "result": result,
        "winner_label": (result or {}).get("winner_label") or (result or {}).get("confirm_winner_label"),
        "winner_provider": (result or {}).get("winner_provider"),
        "winner_model": (result or {}).get("winner_model"),
        "winner_committee": (result or {}).get("winner_committee"),
        "winner_origin": (result or {}).get("winner_origin"),
    }


# 事件日志是追加写的，进程被杀掉时最后一行可能只写了一半。解析失败的那一行丢掉，前面完整的事件仍然留下。
def gpu_event_paths() -> list[Path]:
    """@brief GPU 事件日志的逻辑顺序：轮转保留的分片（旧→新）再加活动文件。

    @return 路径列表；没有分片时只有活动文件。
    """
    path = gpu_events_path()
    try:
        shards = sorted(
            (item for item in path.parent.iterdir() if item.name.startswith(path.name + ".") and item.is_file()),
            key=lambda item: (item.stat().st_mtime_ns, item.name),
        )
    except OSError:
        shards = []
    return shards + [path]


def _tail_events() -> list[dict]:
    """@brief 读 GPU 事件日志的逻辑尾（跨轮转分片），跳过被杀时写了一半的行。

    @return 事件字典列表，最多 ``EVENT_TAIL`` 条。
    """
    # 尾读一段就够画最近的事件；被杀时写了一半的行由解析器丢掉。
    return tail_rows_across(gpu_event_paths(), keep=EVENT_TAIL)


# 只展示 pid 还活着的持有者和等待者。进程退出后留下的 json 不再占着「正在用 GPU」。
def _gpu_locks() -> tuple[list[dict], list[dict]]:
    """@brief 列出仍在运行的 GPU 锁持有者与等待者；不读事件日志。

    @return ``(holders, waiters)`` 两个字典列表。
    """
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
    return holders, waiters


def gpu_snapshot() -> dict:
    """@brief 列出仍在运行的 GPU 持有者与等待者，以及最近的事件。

    @return 含 ``holders``、``waiters`` 与 ``events`` 的字典。
    """
    holders, waiters = _gpu_locks()
    return {"holders": holders, "waiters": waiters, "events": _tail_events()}


def gpu_lock_counts() -> dict:
    """@brief 只给锁的两个计数：巡检只要数字，不读机全局事件日志。

    @return 含 ``holders`` 与 ``waiters`` 计数的字典。
    """
    holders, waiters = _gpu_locks()
    return {"holders": len(holders), "waiters": len(waiters)}


# 评测心跳在进程退出后还会留在缓存目录。finished 或 pid 已死的不再算作正在评测。
def live_evals() -> list[dict]:
    """@brief 列出仍在进行的评测心跳；已 finished 或 pid 已死的忽略。

    @return 心跳字典列表（补了 pid 与心跳年龄）。
    """
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
    """@brief 当该目录的 ``state.json`` 是驱动脚本而不是搜索时，返回一个实验队列条目。

    原始英文说明：One experiment queue, when this directory's ``state.json`` is a driver script rather than a search.

    @param directory 实验目录。
    @return 队列字典；不是驱动脚本时返回 None。
    """
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
    """@brief 只保留每个正确性用例的对错与截短的消息。

    @param report 解析后的 ``report.json``。
    @return 每项含 ``case``、``ok`` 与 ``message`` 的列表。
    """
    cases = (report.get("correctness") or {}).get("cases") or []
    out = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        out.append({"case": case.get("case"), "ok": case.get("ok"), "message": (case.get("message") or "")[:160]})
    return out


# 核源码只带开头一小段，页面用来认出是哪份核。文件读不到时留空，不让详情整页失败。
# 400 字符的 UTF-8 上限是 1,600 字节，读 2,048 字节不会截断可见文本。
_EXCERPT_BYTES = 2048


def _excerpt(path: Path) -> str:
    """@brief 取内核源码开头一小段用于辨认；文件读不到时留空。

    只读开头的字节：400 个字符在 UTF-8 里最多 1,600 字节，读 2,048 字节足够，不必整份加载。

    @param path 内核文件路径。
    @return 前 400 个字符，或空串。
    """
    return head_text(path, limit=_EXCERPT_BYTES)[:400]


# 详情要列出每次候选的评测结论。report.json 截断时这一格仍在，只是状态字段为空。
def _candidates(directory: Path) -> list[dict]:
    """@brief 列出每次候选的评测结论；报告截断时状态字段为空，但这一格仍保留。

    @param directory 运行目录。
    @return 候选行字典列表。
    """
    root = directory / "candidates"
    if not root.is_dir():
        return []
    return [candidate_row(directory, path) for path in sorted(item for item in root.iterdir() if item.is_dir())]


# 没有心跳字母就不当作还活着。评测进度和 tcenv 文件的形状不同，这里只抽出页面要的三个字段。
def _heartbeat(progress: dict | None) -> dict | None:
    """@brief 从评测进度里抽出心跳字母、年龄与阶段；没有心跳字母时返回 None。

    @param progress 解析后的 ``eval_progress.json``，可为 None。
    @return 含 ``letter``、``age_s`` 与 ``stage`` 的字典，或 None。
    """
    if not progress or not progress.get("heartbeat"):
        return None
    return {"letter": progress.get("heartbeat"), "age_s": progress.get("heartbeat_age_s"), "stage": progress.get("stage")}


# 聊天行里只留标签和时刻。messages、reply、usage 不进这一帧，提示词和密钥不会出现在时间线上。
def _retry_kind(error: str) -> str:
    """@brief 按错误文本归类重试原因。

    @param error 错误文本。
    @return ``timeout`` / ``connection`` / ``rate_limit`` / ``server`` / ``retry``。
    """
    text = error.lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text:
        return "connection"
    if "rate" in text or "429" in text:
        return "rate_limit"
    if "internalserver" in text or "500" in text or "502" in text or "503" in text:
        return "server"
    return "retry"


# 网页只留最近这么多行。更早的留在 pi 自己的会话文件里。
_PI_KEEP = 40
# 只读文件末尾。一场很长的会话不会每次轮询都整份解析。
_PI_TAIL_BYTES = 256 * 1024


def _clip(text: str, limit: int) -> str:
    """@brief 压成一行短文本；形似密钥的 token 不展示。

    原始英文说明：One short line. A key-shaped token is not shown.

    @param text 原始文本。
    @param limit 字符上限。
    @return 折叠空白并截断后的文本，超长时以省略号结尾。
    """
    cleaned = redact_text(text, one_line=True)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _pi_hint(arguments) -> str:
    """@brief 从工具调用里取一个路径或命令，而不是整份内核。

    原始英文说明：A path or a command from a tool call, never the whole kernel.

    @param arguments 工具参数，可为 JSON 文本或字典。
    @return 截短后的提示文本；没有可用字段时为空串。
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return _clip(arguments, 120)
    if not isinstance(arguments, dict):
        return ""
    for key in ("command", "path", "file_path", "file", "target"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return _clip(value, 160)
    return ""


def _tail_text(path: Path) -> str:
    """@brief 读取会话文件末尾一段；丢掉不完整的第一行，避免被当成 JSON 解析。

    实现在 :func:`taic.reads.tail_text`，这里只固定本视图的字节上限。

    原始英文说明：The end of a session file. A partial first line is dropped so it is not parsed as JSON.

    @param path 会话文件路径。
    @return 末尾文本；读不了时为空串。
    """
    return tail_text(path, limit=_PI_TAIL_BYTES)


def _pi_lines_in(path: Path) -> list[dict]:
    """@brief 从一个 pi 会话末尾取出可展示的行；思考签名与任务提示词不出现。

    原始英文说明：Readable lines from the tail of one pi session. Thinking signatures and the task prompt stay out.

    @param path 会话文件路径。
    @return 每项含 ``who`` 与 ``text`` 的行列表（最多 40 条）。
    """
    lines = []
    for raw_line in _tail_text(path).splitlines():
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        message = item.get("message") if isinstance(item, dict) else None
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        for part in message.get("content") or []:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if role == "assistant" and kind == "text":
                text = _clip(str(part.get("text") or ""), 400)
                if text:
                    lines.append({"who": "pi", "text": text})
            elif role == "assistant" and kind == "thinking":
                text = _clip(str(part.get("thinking") or ""), 160)
                if text:
                    lines.append({"who": "思考", "text": text})
            elif role == "assistant" and kind == "toolCall":
                name = str(part.get("name") or "tool")
                hint = _pi_hint(part.get("arguments"))
                lines.append({"who": "工具", "text": name + (f" {hint}" if hint else "")})
            elif role == "toolResult" and kind == "text":
                text = _clip(str(part.get("text") or ""), 200)
                if text:
                    lines.append({"who": "结果", "text": text})
    return lines[-_PI_KEEP:]


def _pi_session_files(directory: Path, root: Path) -> list[Path]:
    """@brief 找出目录名包含该运行目录名的 pi 会话文件。

    原始英文说明：Pi sessions whose folder name contains this run's directory name.

    @param directory 运行目录。
    @param root pi 会话根目录。
    @return 按修改时间升序的会话文件列表。
    """
    token = directory.name
    if not token or not root.is_dir():
        return []
    found = []
    for path in root.glob("*/*.jsonl"):
        if token in path.parent.name:
            found.append(path)
    found.sort(key=lambda item: item.stat().st_mtime)
    return found


def _pi_stamp(paths: list[Path]) -> str:
    """@brief 用文件名、mtime 与大小拼出文件集合的变更戳。

    @param paths 文件路径列表。
    @return 变更戳字符串；全部读不到时为空串。
    """
    parts = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


_EPIC_HOW = {"done": "完成", "failed": "失败", "stopped": "停下"}


def _epic_token(token: str) -> str:
    """@brief 把运行目录名压成 hmz 的 epic 目录里出现的样子。

    hmz 用路径拼目录名，把所有非字母数字的字符（含 ``_``）压成一个连字符；运行目录名里有
    ``20261025-070409_conv2d_fp16_...`` 这样的下划线，直接拿原名去 glob 永远匹配不上。

    @param token 运行目录名。
    @return 只含字母数字与连字符的名字；已经如此时原样返回。
    """
    return re.sub(r"[^A-Za-z0-9]+", "-", token).strip("-")


def _epic_files(directory: Path, root: Path) -> list[Path]:
    """@brief 找出目录名包含该运行的 ``epic.jsonl``，只留最近 8 个。

    原始英文说明：epic.jsonl files whose directory name contains this run.

    @param directory 运行目录。
    @param root epic 根目录。
    @return 按修改时间升序的文件列表。
    """
    token = directory.name
    if not token or not root.is_dir():
        return []
    tokens = {token, _epic_token(token)}
    found = []
    for needle in tokens:
        found.extend(path for path in root.glob(f"*{needle}*/**/epic.jsonl") if path.is_file())
    found.sort(key=lambda item: item.stat().st_mtime)
    return found[-8:]


def _epic_pi_sessions(epic_files: list[Path]) -> list[Path]:
    """@brief 取出 epic 目录里 pi 自己写的会话文件；没有就是空。

    hmz 把 pi 的会话目录收在 ``<epic>/<session>/sessions/pi/sessions/<cwd>/<时间>.jsonl``，
    所以 ``~/.pi/agent/sessions`` 下看不到这些运行，只有这里能看到会话当时写的东西。

    @param epic_files 该运行的 ``epic.jsonl`` 列表。
    @return 按修改时间升序的会话文件列表。
    """
    found = []
    for path in epic_files:
        base = path.parent
        found.extend(item for item in base.glob("sessions/pi/sessions/*/*.jsonl") if item.is_file())
    found.sort(key=lambda item: item.stat().st_mtime)
    return found


def _epic_lines_in(path: Path) -> list[dict]:
    """@brief 按 hmz 的读法还原一次 epic：开始、会话、花费、结束；不含任务文本。

    原始英文说明：One epic the way hmz reads it: start, session, spend, end. Not the task text.

    @param path ``epic.jsonl`` 路径。
    @return 每项含 ``who`` 与 ``text`` 的展示行列表。
    """
    lines = []
    for item in tail_rows(path, keep=80):
        event = item.get("event")
        if event == "began":
            agents = item.get("agents") if isinstance(item.get("agents"), list) else []
            first = agents[0] if agents and isinstance(agents[0], dict) else {}
            backend = str(first.get("backend") or "")
            model = str(first.get("model") or "")
            budget = item.get("budget") if isinstance(item.get("budget"), dict) else {}
            cost = budget.get("cost")
            money = f" 预算 ${float(cost):.2f}" if isinstance(cost, (int, float)) and not isinstance(cost, bool) else ""
            who = str(first.get("agent") or "assistant")
            lines.append({"who": "开始", "text": _clip(f"{who} {backend}/{model}{money}".strip(), 180)})
        elif event == "opened":
            session = str(item.get("session") or "")
            lines.append({"who": "打开", "text": _clip(f"{item.get('backend') or ''} 会话 {session[:8]}", 120)})
        elif event == "usage":
            cost = item.get("cost")
            tokens = item.get("output_tokens")
            seconds = item.get("seconds")
            bits = []
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                bits.append(f"${float(cost):.3f}")
            if isinstance(tokens, int) and not isinstance(tokens, bool):
                bits.append(f"输出 {tokens} token")
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                bits.append(f"{float(seconds):.0f} 秒")
            if bits:
                lines.append({"who": "花费", "text": " ".join(bits)})
        elif event == "ended":
            how = str(item.get("how") or "")
            lines.append({"who": "结束", "text": _EPIC_HOW.get(how, how or "结束")})
        elif event in ("reopened", "called", "returned"):
            lines.append({"who": "流程", "text": str(event)})
    return lines


def _epic_feed(directory: Path, root: Path) -> list[dict]:
    """@brief 汇总该运行全部 epic 的展示行。

    @param directory 运行目录。
    @param root epic 根目录。
    @return 最后 40 条展示行。
    """
    lines = []
    for path in _epic_files(directory, root):
        lines.extend(_epic_lines_in(path))
    return lines[-_PI_KEEP:]


def _chat_rows(item: dict) -> list[dict]:
    """@brief 把一个转录事件的发出与收到文本转成展示行；系统提示词与内核草稿不出现。

    原始英文说明：Sent and received text for one transcript event. The system prompt and kernel drafts stay out.

    @param item 一条 ``transcript.jsonl`` 事件。
    @return 每项含 ``who`` 与 ``text`` 的行列表。
    """
    event = item.get("event")
    label = str(item.get("label") or "")
    if event == "retry":
        return [{"who": "重试", "text": _clip(f"{label} {item.get('error') or ''}", 200)}]
    if event == "error":
        return [{"who": "失败", "text": _clip(f"{label} HTTP {item.get('status') or ''} {item.get('error') or ''}", 200)}]
    if event != "chat":
        return []
    rows = []
    messages = item.get("messages") if isinstance(item.get("messages"), list) else []
    sent = ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            sent = content
            break
    if sent:
        rows.append({"who": "发出", "text": _clip(f"{label} {sent}", 400)})
    reply = item.get("reply") if isinstance(item.get("reply"), dict) else {}
    reasoning = reply.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        rows.append({"who": "思考", "text": _clip(reasoning, 160)})
    content = reply.get("content")
    if isinstance(content, str) and content.strip():
        rows.append({"who": "收到", "text": _clip(content, 400)})
    for call in reply.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "tool")
        hint = ""
        raw_args = function.get("arguments")
        if name == "lookup_msl" and isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("topic"), str):
                hint = parsed["topic"]
        rows.append({"who": "工具", "text": _clip(f"{name} {hint}".strip(), 180)})
    if not any(row["who"] == "收到" or row["who"] == "工具" for row in rows):
        rows.append({"who": "收到", "text": _clip(f"{label} 空回复 {item.get('finish_reason') or ''}".strip(), 160)})
    return rows


def _chat_lines(path: Path) -> list[dict]:
    """@brief 从 ``transcript.jsonl`` 末尾取展示行，不整份加载文件。

    原始英文说明：Display lines from the tail of ``transcript.jsonl``. The file itself is not loaded whole.

    @param path 转录文件路径。
    @return 最后 40 条展示行。
    """
    rows = []
    for raw_line in _tail_text(path).splitlines():
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.extend(_chat_rows(item))
    return rows[-_PI_KEEP:]


def _chat_inflight(path: Path) -> list[dict]:
    """@brief 为尚未写入转录的请求生成一行提示。

    原始英文说明：One line for a request that has not been written to the transcript yet.

    @param path ``llm.json`` 路径。
    @return 每项含 ``who``（正在接收/正在等待）与 ``text`` 的行列表。
    """
    doc = _load(path) or {}
    inflight = doc.get("inflight") if isinstance(doc.get("inflight"), dict) else {}
    rows = []
    for label, entry in inflight.items():
        status = entry.get("status") if isinstance(entry, dict) else ""
        who = "正在接收" if status == "streaming" else "正在等待"
        rows.append({"who": who, "text": str(label)})
    return rows


def chat_feed(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 返回直调模型对话记录的末尾；运行不存在时返回 ``None``。

    ``stamp`` 仍然匹配时 ``same`` 为真，页面可跳过重绘。
    原始英文说明：``same`` is true when ``stamp`` still matches, so the page can skip drawing.

    @param run_id 运行 id。
    @param stamp 客户端持有的变更戳，空串表示不复用。
    @return 含 ``same``、``stamp`` 与 ``lines`` 的字典；运行不存在时为 None。
    """
    directory = resolve_run(run_id)
    if directory is None:
        return None
    watched = [path for path in (directory / "transcript.jsonl", directory / "llm.json") if path.is_file()]
    current = _pi_stamp(watched)
    if stamp and stamp == current:
        return {"same": True, "stamp": current, "lines": []}
    lines = _chat_lines(directory / "transcript.jsonl")
    lines.extend(_chat_inflight(directory / "llm.json"))
    return {"same": False, "stamp": current, "lines": lines[-_PI_KEEP:]}


def recorded_kernel_writer(directory: Path, config: dict | None = None) -> str | None:
    """@brief 只凭配置与 ``agent_work`` 判断写核方式；不打开 transcript。

    冷建用它代替 :func:`kernel_writer`：旧运行没有 ``search.kernel_agent`` 时返回 None，让索引器把
    这行留到下一次对账/读取时再用 transcript 判断（§11.4：冷建读 transcript 字节 = 0）。

    原始英文说明：The recorded writer mode, from config and ``agent_work`` only.

    @param directory 运行目录。
    @param config 已解析的 ``run_config.json``；None 时现读。
    @return ``"pi"`` / ``"chat"``；无法从便宜来源判断时返回 None。
    """
    cfg = config if isinstance(config, dict) else (_load(directory / "run_config.json") or {})
    search = cfg.get("search") if isinstance(cfg.get("search"), dict) else {}
    recorded = search.get("kernel_agent") if isinstance(search, dict) else None
    if recorded == "pi":
        return "pi"
    if isinstance(recorded, str):
        return "chat"
    work = directory / "agent_work"
    try:
        if work.is_dir() and any(work.iterdir()):
            return "pi"
    except OSError:
        return None
    return None


def kernel_writer(directory: Path) -> str:
    """@brief 判断该运行是通过 hmz 写核（``pi``）还是自己调模型（``chat``）。

    ``run_config.json`` 优先。该字段出现之前的旧运行，只有已存在 hmz 工作区或 agent 事件时才算 ``pi``。
    原始英文说明：``pi`` when this run writes kernels through hmz. ``chat`` when the search calls the model itself.

    @param directory 运行目录。
    @return ``"pi"`` 或 ``"chat"``。
    """
    cfg = _load(directory / "run_config.json") or {}
    search = cfg.get("search") if isinstance(cfg.get("search"), dict) else {}
    recorded = search.get("kernel_agent") if isinstance(search, dict) else None
    if recorded == "pi":
        return "pi"
    if isinstance(recorded, str):
        return "chat"
    work = directory / "agent_work"
    if work.is_dir() and any(work.iterdir()):
        return "pi"
    path = directory / "transcript.jsonl"
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if '"event": "agent"' in line or '"event":"agent"' in line:
                    return "pi"
    except OSError:
        return "chat"
    return "chat"


def pi_feed(run_id: str, *, sessions_root: Path | None = None, epics_root: Path | None = None, stamp: str = "") -> dict | None:
    """@brief 返回最新 pi 会话的末尾；运行不存在时返回 ``None``。

    ``stamp`` 仍与文件匹配时 ``same`` 为真，页面可跳过重绘。
    原始英文说明：The newest session's tail. ``same`` is true when ``stamp`` still matches the file.

    @param run_id 运行 id。
    @param sessions_root pi 会话根目录，None 表示 ``~/.pi/agent/sessions``。
    @param epics_root epic 根目录，None 表示 ``~/.hmz/epics``。
    @param stamp 客户端持有的变更戳，空串表示不复用。
    @return 含 ``same``、``stamp``、``lines``、``epics`` 与 ``kernel_writer`` 的字典。
    """
    directory = resolve_run(run_id)
    if directory is None:
        return None
    if kernel_writer(directory) != "pi":
        # 直调模型的一场没有 pi 会话。页面不画那两块，这里也不去扫家目录。
        return {"same": stamp == "chat", "stamp": "chat", "lines": [], "epics": [], "kernel_writer": "chat"}
    root = sessions_root if sessions_root is not None else Path.home() / ".pi" / "agent" / "sessions"
    epics_root = epics_root if epics_root is not None else Path.home() / ".hmz" / "epics"
    epic_files = _epic_files(directory, epics_root)
    # 经 hmz 跑的 pi 把会话写进 epic 目录，``~/.pi/agent/sessions`` 里没有这场；两边都找。
    files = _pi_session_files(directory, root) + _epic_pi_sessions(epic_files)
    files.sort(key=lambda item: item.stat().st_mtime)
    current = _pi_stamp(files[-1:]) + "||" + _pi_stamp(epic_files)
    if stamp and stamp == current:
        return {"same": True, "stamp": current, "lines": [], "epics": [], "kernel_writer": "pi"}
    lines = []
    for path in files[-1:]:
        lines.extend(_pi_lines_in(path))
    return {
        "same": False,
        "stamp": current,
        "lines": lines[-_PI_KEEP:],
        "epics": _epic_feed(directory, epics_root),
        "kernel_writer": "pi",
    }


def _number(value) -> float | None:
    """@brief 把数字转成浮点；布尔与非数字返回 None。

    @param value 待转换的值。
    @return 浮点数或 None。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _whole(value) -> int | None:
    """@brief 取整数；布尔与非整数返回 None。

    @param value 待检查的值。
    @return 整数或 None。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def project_turn(item: dict) -> dict | None:
    """@brief 为轮次表生成一行精简记录；系统提示词、内核源码与密钥都不出现。

    原始英文说明：One slim row for the turns table. The system prompt, kernel source and secrets stay out.

    @param item 一条转录事件。
    @return 精简后的轮次行；事件类型不相关时返回 None。
    """
    event = item.get("event")
    if event not in ("chat", "retry", "error", "warning", "agent"):
        return None
    label = item.get("label") if isinstance(item.get("label"), str) else ""
    model = item.get("model") if isinstance(item.get("model"), str) else ""
    row: dict = {
        "label": label,
        "kind": event,
        "model": model,
        "seconds": _number(item.get("seconds")),
        "finish_reason": item.get("finish_reason") if isinstance(item.get("finish_reason"), str) else "",
        "error": "",
        "sent": "",
        "reply": "",
        "thinking": "",
        "tools": [],
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": _number(item.get("cost_usd")),
        "had_kernel": False,
    }
    if event == "retry":
        row["error"] = _clip(str(item.get("error") or ""), 180)
        return row
    if event == "error":
        status = item.get("status")
        err = str(item.get("error") or "")
        row["error"] = _clip(f"HTTP {status}: {err}" if status else err, 180)
        return row
    if event == "warning":
        row["error"] = _clip(str(item.get("message") or ""), 180)
        return row
    if event == "agent":
        parts = [item.get("committee") or "", item.get("harness") or "pi", item.get("provider") or "", item.get("model") or ""]
        row["model"] = "/".join(str(part) for part in parts if part)
        row["error"] = _clip(str(item.get("error") or ""), 180)
        row["had_kernel"] = bool(item.get("had_kernel"))
        tokens = item.get("tokens") if isinstance(item.get("tokens"), dict) else {}
        prompt = 0
        completion = 0
        for kind, value in tokens.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if kind == "output":
                completion += int(value)
            else:
                prompt += int(value)
        row["prompt_tokens"] = prompt or None
        row["completion_tokens"] = completion or None
        row["self_checks"] = _whole(item.get("self_checks"))
        row["self_check_timeouts"] = _whole(item.get("self_check_timeouts"))
        row["outcome"] = item.get("outcome") if isinstance(item.get("outcome"), str) else ""
        return row
    messages = item.get("messages") if isinstance(item.get("messages"), list) else []
    sent = ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            sent = content
            break
    if sent:
        row["sent"] = _clip(sent, 400)
    reply = item.get("reply") if isinstance(item.get("reply"), dict) else {}
    reasoning = reply.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        row["thinking"] = _clip(reasoning, 300)
    content = reply.get("content")
    if isinstance(content, str) and content.strip():
        row["reply"] = _clip(content, 800)
    tools = []
    for call in reply.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        tools.append(str(function.get("name") or "tool"))
    row["tools"] = tools
    usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
    row["prompt_tokens"] = _whole(usage.get("prompt_tokens"))
    row["completion_tokens"] = _whole(usage.get("completion_tokens"))
    return row


def _present_turn(group: dict) -> dict:
    """@brief 把一个事件组转成页面行；同标签且相邻的事件已合并为一组。

    原始英文说明：One page row. Events that share a label and sit next to each other are already one group.

    @param group ``_turn_groups`` 生成的组。
    @return 含标签、区间、类型、指标与文本的页面行。
    """
    events = group["events"]
    def last(kind: str) -> dict | None:
        found = [event for event in events if event.get("kind") == kind]
        return found[-1] if found else None

    chat = last("chat")
    agent = last("agent")
    error = last("error")
    retry = last("retry")
    warning = last("warning")
    model = ""
    for event in reversed(events):
        if event.get("model"):
            model = str(event["model"])
            break
    retries = [str(event.get("error") or "") for event in events if event.get("kind") == "retry" and event.get("error")]
    kind = "retry"
    err = ""
    sent = thinking = reply = ""
    tools: list[str] = []
    seconds = None
    cost = None
    tokens = None
    if chat:
        finish = chat.get("finish_reason") or ""
        sent = str(chat.get("sent") or "")
        thinking = str(chat.get("thinking") or "")
        reply = str(chat.get("reply") or "")
        tools = [str(name) for name in (chat.get("tools") or [])]
        seconds = chat.get("seconds")
        cost = chat.get("cost_usd")
        prompt = chat.get("prompt_tokens")
        completion = chat.get("completion_tokens")
        if isinstance(prompt, int) or isinstance(completion, int):
            tokens = int(prompt or 0) + int(completion or 0)
        if finish == "length" or warning is not None:
            kind = "truncated"
        elif tools and not reply:
            kind = "tool"
        elif not reply and not tools:
            kind = "empty"
        else:
            kind = "reply"
        if error is not None:
            err = str(error.get("error") or "")
    elif agent is not None:
        runtime = agent.get("runtime") if isinstance(agent.get("runtime"), dict) else {}
        failure_class = str(runtime.get("failure_class") or "")
        failed_turns = runtime.get("failed_turns")
        if agent.get("error") or failure_class == "no_submission_after_result":
            kind = "agent_error"
        elif agent.get("had_kernel"):
            kind = "agent"
        else:
            kind = "agent_empty"
        err = _clip(str(agent.get("error") or runtime.get("last_error") or ""), 180)
        cost = agent.get("cost_usd")
        seconds = agent.get("seconds")
        prompt = agent.get("prompt_tokens")
        completion = agent.get("completion_tokens")
        if isinstance(prompt, int) or isinstance(completion, int):
            tokens = int(prompt or 0) + int(completion or 0)
        checks = agent.get("self_checks")
        timeouts = agent.get("self_check_timeouts")
        outcome = agent.get("outcome") or ""
        sent = " ".join(part for part in (
            f"自测 {checks} 次" if isinstance(checks, int) else "",
            f"超时 {timeouts} 次" if isinstance(timeouts, int) and timeouts else "",
            str(outcome),
            f"诊断 {failure_class}" if failure_class else "",
            f"hmz 失败回合 {failed_turns} 次" if isinstance(failed_turns, int) and not isinstance(failed_turns, bool) and failed_turns else "",
            f"诊断文件 {agent.get('diagnostics_path')}" if agent.get("diagnostics_path") else "",
        ) if part)
    elif error is not None:
        kind = "http"
        err = str(error.get("error") or "")
    elif retry is not None:
        err = str(retry.get("error") or "")
        kind = _retry_kind(err)
    elif warning is not None:
        kind = "truncated"
        err = str(warning.get("error") or "")
    return {
        "label": group["label"],
        "min_seq": group["min_seq"],
        "max_seq": group["max_seq"],
        "kind": kind,
        "model": model,
        "seconds": seconds,
        "tokens": tokens,
        "cost_usd": cost,
        "sent": sent,
        "thinking": thinking,
        "reply": reply,
        "tools": tools,
        "retries": retries,
        "error": err,
        "preview": "",
        "preview_thinking": "",
        "stale": False,
        **({"truncated_events": int(group["truncated_events"])} if group.get("truncated_events") else {}),
    }


def _turn_groups(events: list[dict]) -> list[dict]:
    """@brief 把相邻同标签的事件合成一组；同名标签之后再出现算另一组。

    旧实现将就地在 Python 里分组整段事件。现在分组由索引投影的 ``turns.grp`` 决定，
    这里只保留作逐组等价的参考实现（测试用它当独立的期望值）。
    原始英文说明：Consecutive events with the same label are one group. A later repeat of that label is another group.

    @param events 已带 ``seq`` 的事件列表。
    @return 每组含 ``label``、``min_seq``、``max_seq`` 与 ``events`` 的列表。
    """
    groups: list[dict] = []
    for event in events:
        label = str(event.get("label") or "")
        if groups and label and groups[-1]["label"] == label:
            groups[-1]["events"].append(event)
            groups[-1]["max_seq"] = event["seq"]
            continue
        groups.append({"label": label, "min_seq": event["seq"], "max_seq": event["seq"], "events": [event]})
    return groups


def _before_seq(before) -> int | None:
    """@brief 把 ``before`` 参数解析成整数 seq；缺失或非法时返回 None。

    @param before 客户端传入的 seq 边界。
    @return 整数 seq 或 None。
    """
    if before is None or before == "":
        return None
    try:
        return int(before)
    except (TypeError, ValueError):
        return None


def _grp_param(grp) -> int | None:
    """@brief 把 ``grp`` 参数解析成整数；缺失或非法时返回 None。

    @param grp 客户端传入的组号。
    @return 整数组号或 None。
    """
    if grp is None or grp == "":
        return None
    try:
        return int(grp)
    except (TypeError, ValueError):
        return None


def _turn_grp_bound(conn, run_id: str, before_seq: int | None, grp_cursor: int | None) -> int | None:
    """@brief 把 ``grp``/``before`` 换算成排他上界（只取 ``grp < bound`` 的组）。

    ``before`` 是旧契约（页面最旧一组的最小 seq）：先按 seq 查出组号，再排它。
    若行已不在（客户端拿着旧游标），退回“seq 更小的最新一行”所在的组并包含那组。
    ``grp_cursor`` 是新的组游标，本身就是排他上界。
    原始英文说明：Turn the seq/group cursors into the exclusive upper bound of the group page.

    @param conn 只读连接。
    @param run_id 运行 id。
    @param before_seq 旧契约的 seq 游标。
    @param grp_cursor 新的组游标。
    @return 排他上界；不设上界时为 None。
    """
    if grp_cursor is not None:
        return grp_cursor
    if before_seq is None:
        return None
    row = conn.execute("SELECT grp FROM turns WHERE run_id = ? AND seq = ?", (run_id, before_seq)).fetchone()
    if row is not None:
        return int(row["grp"])
    row = conn.execute(
        "SELECT grp FROM turns WHERE run_id = ? AND seq < ? ORDER BY seq DESC LIMIT 1", (run_id, before_seq)
    ).fetchone()
    if row is None:
        return 0
    return int(row["grp"]) + 1


def _parse_turn_rows(rows) -> list[dict]:
    """@brief 把 ``(seq, row_json)`` 行解成事件，坏 JSON 与半行照旧丢掉。

    @param rows sqlite 行序列（含 ``seq`` 与 ``row_json``）。
    @return 按输入顺序的事件列表，每个事件带 ``seq``。
    """
    events = []
    for item in rows:
        try:
            event = json.loads(item["row_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event["seq"] = int(item["seq"])
        events.append(event)
    return events


def _group_rows(conn, run_id: str, grp: int, *, limit: int | None) -> list:
    """@brief 取一组的行；``limit`` 给定时只取最新 ``limit`` 条（再反转成展示顺序）。

    截断下推到 SQL：大组不会先把整组 ``fetchall()`` 进内存。
    原始英文说明：The rows of one group, newest-first truncated in SQL when a limit is given.

    @param conn 只读连接。
    @param run_id 运行 id。
    @param grp 组号。
    @param limit 最多取多少条（None 表示全量，只有显式 ``all=1`` 才用）。
    @return 按 seq 升序的 ``(seq, row_json)`` 行列表。
    """
    if limit is None:
        return conn.execute(
            "SELECT seq, row_json FROM turns WHERE run_id = ? AND grp = ? ORDER BY seq", (run_id, grp)
        ).fetchall()
    fetched = conn.execute(
        "SELECT seq, row_json FROM turns WHERE run_id = ? AND grp = ? ORDER BY seq DESC LIMIT ?",
        (run_id, grp, limit),
    ).fetchall()
    return list(reversed(fetched))


def _page_groups(conn, run_id: str, *, bound: int | None, limit: int) -> tuple[list[dict], bool]:
    """@brief 取一页组（新→旧）并判断更早还有多少组。

    两条组查询都走 ``turns_run_grp(run_id, grp, seq)``；本页的行用闭区间 ``grp BETWEEN`` 取，
    不会因为把最大 grp 当排他上界而整组丢掉。
    原始英文说明：One page of groups (newest first) plus whether older groups exist.

    @param conn 只读连接。
    @param run_id 运行 id。
    @param bound 排他上界；None 表示首页。
    @param limit 最多几组。
    @return 二元组：新→旧的组列表（含 min_seq/max_seq/label）与是否还有更早的组。
    """
    sql = "SELECT grp, MIN(seq) AS min_seq, MAX(seq) AS max_seq, label FROM turns WHERE run_id = ?"
    args: list = [run_id]
    if bound is not None:
        sql += " AND grp < ?"
        args.append(bound)
    sql += " GROUP BY grp ORDER BY grp DESC LIMIT ?"
    args.append(limit + 1)
    fetched = [dict(item) for item in conn.execute(sql, args)]
    return fetched[:limit], len(fetched) > limit


def _group_counts(conn, run_id: str, grps: list[int]) -> dict[int, int]:
    """@brief 一页各组的事件数（SQL 聚合），用于判定哪些组要截断。

    @param conn 只读连接。
    @param run_id 运行 id。
    @param grps 本页的组号。
    @return 组号到事件数的映射。
    """
    if not grps:
        return {}
    lo, hi = min(grps), max(grps)
    return {
        int(row["grp"]): int(row["n"])
        for row in conn.execute(
            "SELECT grp, COUNT(*) AS n FROM turns WHERE run_id = ? AND grp BETWEEN ? AND ? GROUP BY grp",
            (run_id, lo, hi),
        )
    }


def _turns_stamp(turn_bytes: int, directory: Path, epoch: int = 0) -> str:
    """@brief 拼出轮次数据的变更戳：转录与 ``llm.json`` 的大小与 mtime、已存字节数与投影 epoch。

    ``epoch`` 覆盖“字节偏移不变但投影内容变了”的情况：旧 ``grp=0`` 行整份重投影时
    ``turn_bytes`` 可能不变，只有 epoch 能证明旧的 ``same`` 已经失效。

    @param turn_bytes 索引中已存的轮次字节数。
    @param directory 运行目录。
    @param epoch 该 run 的轮次投影 epoch。
    @return 变更戳字符串。
    """
    def part(path: Path) -> tuple[int, int]:
        try:
            stat = path.stat()
        except OSError:
            return (0, 0)
        return (int(stat.st_size), int(stat.st_mtime_ns))

    transcript = part(directory / "transcript.jsonl")
    live = part(directory / "llm.json")
    return f"{transcript[0]}.{transcript[1]}.{int(turn_bytes)}.{int(epoch)}.{live[0]}.{live[1]}"


def _stopped_turns(directory: Path) -> list[dict]:
    """@brief 把搜索中止的分支转成展示行；这些行在 history 里，不在转录里。

    原始英文说明：A branch the search aborted. These lines live in history, not in the transcript.

    @param directory 运行目录。
    @return ``kind=stopped`` 的页面行列表。
    """
    rows = []
    for doc in (_load(directory / "state.json"), _load(directory / "result.json")):
        history = (doc or {}).get("history") if isinstance(doc, dict) else None
        if not isinstance(history, list):
            continue
        for line in history:
            if not isinstance(line, str):
                continue
            if "aborted" not in line and "gave up" not in line and "llm_error" not in line:
                continue
            rows.append({
                "label": "",
                "min_seq": None,
                "max_seq": None,
                "kind": "stopped",
                "model": "",
                "seconds": None,
                "tokens": None,
                "cost_usd": None,
                "sent": "",
                "thinking": "",
                "reply": "",
                "tools": [],
                "retries": [],
                "error": _clip(line, 180),
                "preview": "",
                "preview_thinking": "",
                "stale": False,
            })
        break
    return rows


def _attach_preview(rows: list[dict], directory: Path, *, running: bool, tail: bool) -> None:
    """@brief 把在飞文本挂到同标签的行上；还没有行的标签在尾部新增一行。

    原始英文说明：The in-flight text sits on the row with that label. A label with no row yet is a new line at the end of the tail.

    @param rows 页面行列表（就地修改）。
    @param directory 运行目录，用于读取 ``llm.json``。
    @param running 运行是否仍在跑，决定 ``stale`` 标记。
    @param tail 是否在尾部新增行。
    """
    doc = _load(directory / "llm.json") or {}
    inflight = doc.get("inflight") if isinstance(doc.get("inflight"), dict) else {}
    for label, entry in inflight.items():
        if not isinstance(entry, dict):
            continue
        preview = entry.get("preview") if isinstance(entry.get("preview"), str) else ""
        thinking = entry.get("thinking") if isinstance(entry.get("thinking"), str) else ""
        status = entry.get("status") if isinstance(entry.get("status"), str) else ""
        target = None
        for row in reversed(rows):
            if row.get("label") == label and row.get("min_seq") is not None:
                target = row
                break
        if target is None:
            if not tail:
                continue
            target = {
                "label": str(label),
                "min_seq": None,
                "max_seq": None,
                "kind": "streaming" if status == "streaming" else "waiting",
                "model": "",
                "seconds": None,
                "tokens": None,
                "cost_usd": None,
                "sent": "",
                "thinking": "",
                "reply": "",
                "tools": [],
                "retries": [],
                "error": "",
                "preview": "",
                "preview_thinking": "",
                "stale": not running,
            }
            rows.append(target)
        target["preview"] = preview
        target["preview_thinking"] = thinking
        target["stale"] = not running


TURN_GROUP_EVENTS = 400
# §10.4 组级上限：回合一页 40 组。前端“更早”按这个页大小翻，服务端夹到 1..200。
TURNS_PAGE = 40


def turns_feed(run_id: str, *, before=None, limit: int = TURNS_PAGE, stamp: str = "", grp=None, everything: bool = False) -> dict | None:
    """@brief 返回一段按组切页的轮次窗口；``before``/``grp`` 是页面上最旧那一组的游标。

    组由索引投影的 ``turns.grp`` 决定：相邻同标签一组，同名标签再次出现另起一组，空标签每行一组。
    翻页是组 keyset：先取一页组，再用闭区间 ``grp BETWEEN`` 取这些组的行；单组超过
    ``TURN_GROUP_EVENTS`` 条时只取最新 400 条（SQL 下推）并标 ``truncated_events``。
    ``grp=<g>&all=1`` 显式读回该组全量聚合。
    ``stamp`` 仍匹配时 ``same`` 为真，此时不读轮次表。
    原始英文说明：A window of turns paged by group. ``same`` is true when ``stamp`` still matches, and then the turns table is not read.

    @param run_id 运行 id。
    @param before 旧契约的 seq 游标（页面最旧一组的最小 seq），可为 None。
    @param limit 最大组数，会被夹到 1..200。
    @param stamp 客户端持有的变更戳，空串表示不复用。
    @param grp 新的组游标（排他上界），可为 None。
    @param everything 为 True 且给了 ``grp`` 时返回该组全量聚合（不截断）。
    @return 含 ``same``、``stamp``、``cursor``、``grp_cursor`` 与 ``rows`` 的字典；运行不存在时为 None。
    """
    directory = resolve_run(run_id)
    if directory is None:
        return None
    from observe.index import get_index

    index = get_index()
    index.ensure_turns(directory)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = TURNS_PAGE
    limit = max(1, min(limit, 200))
    before_seq = _before_seq(before)
    grp_cursor = _grp_param(grp)
    full = bool(everything) and grp_cursor is not None
    conn = index.read_only()
    try:
        stored = conn.execute("SELECT turn_bytes, status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        turn_bytes = int(stored["turn_bytes"] or 0) if stored else 0
        running = bool(stored) and stored["status"] == "running"
        epoch_row = conn.execute("SELECT value FROM meta WHERE key = ?", (f"turns_epoch:{run_id}",)).fetchone()
        epoch = int(epoch_row[0]) if epoch_row is not None and str(epoch_row[0]).isdigit() else 0
        current = _turns_stamp(turn_bytes, directory, epoch)
        if stamp and stamp == current and before_seq is None and grp_cursor is None:
            return {"same": True, "stamp": current}
        index.stats["turn_reads"] = index.stats.get("turn_reads", 0) + 1
        if full:
            fetched = conn.execute(
                "SELECT grp, MIN(seq) AS min_seq, MAX(seq) AS max_seq, label FROM turns"
                " WHERE run_id = ? AND grp = ? GROUP BY grp",
                (run_id, grp_cursor),
            ).fetchone()
            groups = [] if fetched is None else [dict(fetched)]
            rows = []
            if groups:
                group = {
                    "label": groups[0].get("label") or "",
                    "min_seq": int(groups[0]["min_seq"]),
                    "max_seq": int(groups[0]["max_seq"]),
                    "events": _parse_turn_rows(_group_rows(conn, run_id, grp_cursor, limit=None)),
                }
                rows = [_present_turn(group)]
            return {"same": False, "stamp": current, "cursor": None, "grp_cursor": None,
                    "all": True, "grp": grp_cursor, "rows": rows}
        bound = _turn_grp_bound(conn, run_id, before_seq, grp_cursor)
        page, has_older = _page_groups(conn, run_id, bound=bound, limit=limit)
        window = list(reversed(page))  # 展示顺序：旧→新
        counts = _group_counts(conn, run_id, [int(item["grp"]) for item in window])
        truncated = {key: value for key, value in counts.items() if value > TURN_GROUP_EVENTS}
        by_grp: dict[int, list[dict]] = {}
        small = [int(item["grp"]) for item in window if int(item["grp"]) not in truncated]
        if small:
            if len(small) == len(window):
                # 常见情况：本页没有超上限的组，一条闭区间查询取全部行（走 turns_run_grp）。
                fetched = conn.execute(
                    "SELECT grp, seq, row_json FROM turns WHERE run_id = ? AND grp BETWEEN ? AND ? ORDER BY seq",
                    (run_id, small[0], small[-1]),
                ).fetchall()
            else:
                placeholders = ",".join("?" for _ in small)
                fetched = conn.execute(
                    "SELECT grp, seq, row_json FROM turns WHERE run_id = ? AND grp IN (" + placeholders + ")"
                    " ORDER BY seq",
                    (run_id, *small),
                ).fetchall()
            for item in fetched:
                by_grp.setdefault(int(item["grp"]), []).append(item)
        for grp_number in truncated:
            by_grp[grp_number] = _group_rows(conn, run_id, grp_number, limit=TURN_GROUP_EVENTS)
    finally:
        conn.close()
    rows = []
    for item in window:
        number = int(item["grp"])
        events = _parse_turn_rows(by_grp.get(number, []))
        group = {
            "label": item.get("label") or "",
            "min_seq": int(item["min_seq"]),
            "max_seq": int(item["max_seq"]),
            "events": events,
        }
        if number in truncated:
            group["truncated_events"] = truncated[number] - TURN_GROUP_EVENTS
        rows.append(_present_turn(group))
    cursor = int(window[0]["min_seq"]) if window and has_older else None
    grp_cursor_out = int(window[0]["grp"]) if window and has_older else None
    tail = bound is None
    if tail:
        rows.extend(_stopped_turns(directory))
    _attach_preview(rows, directory, running=running, tail=tail)
    return {"same": False, "stamp": current, "cursor": cursor, "grp_cursor": grp_cursor_out, "rows": rows}


def _agent_note(item: dict, label: str) -> dict:
    """@brief 生成一条 hmz 轮次说明；编码 agent 写了文件，这不是聊天工具调用，本身也不是异常。

    原始英文说明：One hmz turn. The coding agent wrote a file; this is not a chat-tool call and not an exception by itself.

    @param item 一条 ``event=agent`` 的转录事件。
    @param label 该轮次的标签。
    @return 含 ``label``、``model``、``kind``、``detail`` 与 ``seconds`` 的说明字典。
    """
    parts = [item.get("harness") or "pi", item.get("provider") or "", item.get("model") or ""]
    shown = "/".join(str(part) for part in parts if part)
    runtime = item.get("runtime") if isinstance(item.get("runtime"), dict) else {}
    failure_class = str(runtime.get("failure_class") or "")
    failed_turns = runtime.get("failed_turns")
    error = _clip(str(item.get("error") or runtime.get("last_error") or ""), 180)
    had = bool(item.get("had_kernel"))
    if error or failure_class == "no_submission_after_result":
        kind, detail = "agent_error", error or "正常回答但没有交卷"
    elif had:
        kind, detail = "agent", "候选内核文件里有约定的内核"
    else:
        kind, detail = "agent_empty", "候选内核文件里没有约定的内核"
    if failure_class == "no_submission_after_result":
        detail += "；正常回答但没有交卷"
    if isinstance(failed_turns, int) and not isinstance(failed_turns, bool) and failed_turns > 0:
        detail += f"；{failed_turns} 个 hmz 失败回合"
    cost = item.get("cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        detail = f"{detail}。这场 ${float(cost):.3f}"
    return {"label": label, "model": shown, "kind": kind, "detail": detail, "seconds": None,
            "diagnostics_path": item.get("diagnostics_path") or None}


def _model_notes(directory: Path) -> list[dict]:
    """@brief 列出没有变成候选的模型轮次。

    包括思考时超时、连接或限流重试、HTTP 错误、被长度截断的回复、只调了工具的回复、空的最终回复，
    以及被搜索中止的分支。提示词、推理文本与内核源码都不进这个列表。
    原始英文说明：Model turns that did not become a candidate.

    @param directory 运行目录。
    @return 每项含 ``label``、``model``、``kind``、``detail`` 与 ``seconds`` 的列表。
    """
    notes = []
    path = directory / "transcript.jsonl"
    for item in iter_jsonl(path):
        event = item.get("event")
        label = item.get("label") if isinstance(item.get("label"), str) else ""
        model = item.get("model") if isinstance(item.get("model"), str) else ""
        if event == "retry":
            err = str(item.get("error") or "")
            notes.append({"label": label, "model": model, "kind": _retry_kind(err), "detail": err[:180], "seconds": None})
            continue
        if event == "error":
            err = str(item.get("error") or "")
            status = item.get("status")
            detail = f"HTTP {status}: {err}" if status else err
            notes.append({"label": label, "model": model, "kind": "http", "detail": detail[:180], "seconds": None})
            continue
        if event == "warning" and item.get("message") == "reply truncated by max_tokens":
            if any(note["label"] == label and note["kind"] == "truncated" for note in notes):
                continue
            notes.append({"label": label, "model": model, "kind": "truncated", "detail": "reply truncated by max_tokens", "seconds": None})
            continue
        if event == "agent":
            notes.append(_agent_note(item, label))
            continue
        if event != "chat":
            continue
        reply = item.get("reply") if isinstance(item.get("reply"), dict) else {}
        content = reply.get("content") if isinstance(reply.get("content"), str) else ""
        calls = [call for call in (reply.get("tool_calls") or []) if isinstance(call, dict)]
        if item.get("finish_reason") == "tool_calls" and not content.strip():
            names = [str((call.get("function") or {}).get("name") or "tool") for call in calls]
            notes.append({"label": label, "model": model, "kind": "tool", "detail": ", ".join(names), "seconds": item.get("seconds")})
        elif not content.strip() and not calls:
            notes.append({"label": label, "model": model, "kind": "empty", "detail": str(item.get("finish_reason") or ""), "seconds": item.get("seconds")})
        elif item.get("finish_reason") == "length":
            notes.append({"label": label, "model": model, "kind": "truncated", "detail": f"正文仍有 {len(content.strip())} 字", "seconds": item.get("seconds")})
        else:
            notes.append({"label": label, "model": model, "kind": "reply", "detail": f"正文 {len(content.strip())} 字", "seconds": item.get("seconds")})
    for doc in (_load(directory / "state.json"), _load(directory / "result.json")):
        history = (doc or {}).get("history") if isinstance(doc, dict) else None
        if not isinstance(history, list):
            continue
        for line in history:
            if not isinstance(line, str):
                continue
            if "aborted" not in line and "gave up" not in line and "llm_error" not in line:
                continue
            notes.append({"label": "", "model": "", "kind": "stopped", "detail": line[:180], "seconds": None})
        break
    return notes


def _chat_events(directory: Path) -> list[dict]:
    """@brief 从转录中抽出每个 chat 事件的时间字段与标签。

    @param directory 运行目录。
    @return 每项含 ``label`` 及 ``seconds`` / ``t`` / ``started_at`` / ``ended_at`` / ``at`` 中可用的数值。
    """
    path = directory / "transcript.jsonl"
    kept_keys = ("seconds", "t", "started_at", "ended_at", "at")
    out = []
    for item in iter_jsonl(path):
        if item.get("event") != "chat":
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
    """@brief 取账本里每个请求结束时刻（距账本起点的秒数）；转录只有耗时，用它对齐同一条钟。

    @param directory 运行目录。
    @return 每项含 ``label`` 与 ``t`` 的列表。
    """
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
    """@brief 把可能是墙钟秒数也可能是相对秒数的值统一成 unix 时间。

    @param value 待转换的值。
    @param origin 相对秒的起点（运行开始时间），可为 None。
    @return unix 时间戳；无法判定时返回 None。
    """
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
    """@brief 算出一条聊天请求的起止时刻；``seconds`` 是耗时，不是钟上位置。

    @param chat ``_chat_events`` 给出的一条事件。
    @param origin 运行开始时间，用于把相对秒换成墙钟时间。
    @return 二元组（开始, 结束）；信息不足时返回 None。
    """
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
    """@brief 按标签（无标签时按顺序）从账本里取下一个请求的结束时刻。

    @param reqs 账本请求列表。
    @param label 目标标签，可为 None。
    @param cursor 当前搜索位置。
    @return 二元组：结束秒数（找不到为 None）与新的游标。
    """
    if not label:
        if cursor >= len(reqs):
            return None, cursor
        return reqs[cursor]["t"], cursor + 1
    for index in range(cursor, len(reqs)):
        if reqs[index]["label"] == label:
            return reqs[index]["t"], index + 1
    return None, cursor


def _model_spans(directory: Path, origin: float | None) -> list[dict]:
    """@brief 把落盘的聊天请求转成时间线上的区间。

    @param directory 运行目录。
    @param origin 运行开始时间，用于把相对秒换成墙钟时间。
    @return 每项含 ``start``、``end`` 与可选 ``label`` 的区间列表。
    """
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


def _token(text: str, word: str) -> bool:
    """@brief 判断 ``word`` 是否以完整 token 形式出现在 ``text`` 中（不匹配子串）。

    @param text 待搜索文本。
    @param word 目标词。
    @return 命中时为 True；任一为空时返回 False。
    """
    if not text or not word:
        return False
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", text) is not None


def _same_pid(event: dict, pid) -> bool:
    """@brief 判断事件里的 pid 是否等于给定 pid。

    @param event GPU 事件字典。
    @param pid 目标 pid。
    @return 相同时为 True；任一不可解析时返回 False。
    """
    try:
        return int(event.get("pid")) == int(pid)
    except (TypeError, ValueError):
        return False


# 等锁条只收这一次搜索的等待：时间落在这次运行里，并且用途写着这个算子，或者 pid 就是这次搜索自己。
def _wait_belongs(event: dict, task: str, dtype: str, pid) -> bool:
    """@brief 判断一条等锁事件是否属于本次搜索。

    pid 是自己的，或 ``what`` 里写着这个算子（有 dtype 时也要写着 dtype）。

    @param event GPU 事件字典。
    @param task 运行的任务名。
    @param dtype 运行的 dtype。
    @param pid 本次运行的 pid。
    @return 属于本次搜索时为 True。
    """
    if _same_pid(event, pid):
        return True
    what = event.get("what") if isinstance(event.get("what"), str) else ""
    if not task or not _token(what, task):
        return False
    if dtype and not _token(what, dtype):
        return False
    return True


def _run_bounds(state: dict | None, result: dict | None, progress: dict | None, status: str) -> tuple[float | None, float | None]:
    """@brief 算出一个运行的（开始, 结束）时间边界。

    仍在跑时结束取到当前时刻；已结束但没有结束时间时退回开始时间。

    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @param progress 解析后的 ``eval_progress.json``，可为 None。
    @param status ``liveness`` 给出的状态。
    @return 二元组（开始, 结束），未知部分为 None。
    """
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


def _run_timing(directory: Path, state: dict | None, result: dict | None, progress: dict | None, status: str) -> dict:
    """@brief 生成详情头部的开始、时长与结束时间。

    运行中的运行没有结束时间，时长按本次请求时刻现算；已结束的运行用最后落盘的时间戳作结束，
    中断的运行把它标为估计值（进程可能在写下最终状态前就死了）。
    原始英文说明：Start, elapsed and end for the detail header.

    @param directory 运行目录。
    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @param progress 解析后的 ``eval_progress.json``，可为 None。
    @param status ``liveness`` 给出的状态。
    @return 含 ``started_at``、``ended_at``、``elapsed_s``、``running`` 与 ``ended_at_estimated`` 的字典。
    """
    start, end = _run_bounds(state, result, progress, status)
    if start is None:
        start = _when(directory, state)
    now = time.time()
    if status == "running":
        ended = None
        elapsed = max(0.0, now - start) if start is not None else None
    else:
        ended = end if status in ("done", "interrupted") else None
        elapsed = max(0.0, ended - start) if ended is not None and start is not None else None
    return {
        "started_at": start,
        "ended_at": ended,
        "elapsed_s": elapsed,
        "running": status == "running",
        "ended_at_estimated": status == "interrupted",
    }


def _lock_spans(state: dict | None, result: dict | None, progress: dict | None, status: str) -> list[dict]:
    """@brief 把本次运行的等锁事件转成时间区间。

    只收时间落在本次运行内、且归属本次搜索的等待；同号 pid 被复用时，区间收到本次运行结束为止。

    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @param progress 解析后的 ``eval_progress.json``，可为 None。
    @param status ``liveness`` 给出的状态。
    @return 每项含 ``start`` 与 ``end`` 的区间列表。
    """
    task = str((result or state or {}).get("task") or "")
    dtype = str((result or state or {}).get("dtype") or "")
    pid = (state or {}).get("pid")
    start, end = _run_bounds(state, result, progress, status)
    if start is None or end is None:
        return []
    lo, hi = start - 2.0, end + 2.0
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

    # 区间已经由索引里的增量摄取物化：这里只查与本次运行窗口相交的 span 和仍未配对的 wait。
    # 输出顺序与旧重放一致：先按“关闭事件”的时刻（overwrite 用新 wait、acquire 用 acquire），
    # 最后才是仍未配对的 wait（各自零长，按 pending 的插入顺序）。
    # 未配对 wait 只会以自身时间零长出现，所以可以按窗口在 SQL 里过滤（10 万个窗口外的 pending
    # wait 不应该被取回来）。
    from observe.index import borrow, get_index

    get_index().ensure_gpu_events()
    with borrow() as held:
        closed, waits = _lock_rows(held.conn, lo, hi)
    for row in closed:
        _emit({"pid": row["pid"], "t": row["start"], "mode": row["mode"], "what": row["what"]}, float(row["end"]))
    for row in waits:
        _emit({"pid": row["pid"], "t": row["t"], "mode": row["mode"], "what": row["what"]}, float(row["t"]))
    return spans


# 区间查询的有界读：桶索引由摄取侧维护（见 Index._index_span）。普通 B-tree 只能在一侧开区间，
# 另一侧会扫全部历史/未来行；桶把候选集限制在窗口覆盖的 10 秒桶，超长 span 单独一张小表。
LOCK_SPAN_BUCKET_SECONDS = 10.0
LOCK_SPAN_LONG_SECONDS = 3600.0
LOCK_SPAN_SLOT_SQL = (
    "SELECT l.rowid AS rowid, l.pid, l.start, l.end, l.closed_at, l.mode, l.what"
    " FROM lock_spans l"
    " WHERE l.rowid IN (SELECT span_id FROM lock_span_slots WHERE bucket BETWEEN ? AND ?)"
    " AND l.start <= ? AND l.end >= ?"
)
LOCK_SPAN_LONG_SQL = (
    "SELECT l.rowid AS rowid, l.pid, l.start, l.end, l.closed_at, l.mode, l.what"
    " FROM lock_spans_long g JOIN lock_spans l ON l.rowid = g.span_id"
    " WHERE l.start <= ? AND l.end >= ?"
)
LOCK_SPAN_SQL = (
    "SELECT rowid, pid, start, end, closed_at, mode, what FROM lock_spans "
    "WHERE end >= ? AND start <= ? ORDER BY closed_at, rowid"
)
PENDING_WAIT_SQL = (
    "SELECT pid, t, mode, what, seq FROM gpu_waits WHERE t >= ? AND t <= ? ORDER BY seq"
)


def _has_span_slots(conn) -> bool:
    """@brief 库里有没有桶索引（旧库尚未补建时退回普通索引）。

    @param conn 索引连接。
    @return 存在时为 True。
    """
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'lock_span_slots'"
        ).fetchone() is not None
    except Exception:  # noqa: BLE001 - 只读连接异常时退回普通索引
        return False


def _lock_rows(conn, lo: float, hi: float):
    """@brief 取与窗口相交的已配对区间与窗口内的未配对 wait。

    未配对 wait 只以自身时间零长出现，所以窗口外的 pending wait 不必取回（10 万个也不会进内存）。
    已配对区间走摄取时维护的桶索引：只读窗口覆盖的桶（+ 极少数超长 span），不沿 ``end`` 索引
    扫全部未来行。结果按旧重放的 ``(closed_at, rowid)`` 排序。

    @param conn 索引连接。
    @param lo 窗口下界。
    @param hi 窗口上界。
    @return ``(closed_rows, pending_rows)``，都已按旧重放的输出顺序排好。
    """
    if _has_span_slots(conn):
        first = int(lo // LOCK_SPAN_BUCKET_SECONDS)
        last = int(hi // LOCK_SPAN_BUCKET_SECONDS)
        closed = list(conn.execute(LOCK_SPAN_SLOT_SQL, (first, last, hi, lo)).fetchall())
        closed.extend(conn.execute(LOCK_SPAN_LONG_SQL, (hi, lo)).fetchall())
        closed.sort(key=lambda row: (float(row["closed_at"] or 0.0), int(row["rowid"])))
    else:
        closed = conn.execute(LOCK_SPAN_SQL, (lo, hi)).fetchall()
    waits = conn.execute(PENDING_WAIT_SQL, (lo, hi)).fetchall()
    return closed, waits


def gpu_shard_stamps() -> list[str]:
    """@brief 机全局事件日志保留分片的 ``(名字, mtime_ns, size)`` 指纹。

    时间线戳要包含分片：分片单独被删/被改时活动文件不变，只盯活动文件会一直回 ``same``。

    @return 指纹字符串列表，按分片名排序。
    """
    path = gpu_events_path()
    try:
        items = [item for item in path.parent.iterdir() if item.name.startswith(path.name + ".") and item.is_file()]
    except OSError:
        return []
    parts = []
    for item in sorted(items, key=lambda entry: entry.name):
        try:
            stat = item.stat()
        except OSError:
            continue
        parts.append(f"{item.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return parts


def gpu_history_truncated() -> bool:
    """@brief 机全局 GPU 事件历史是否被截断过（源文件被删除或替换且没有保留分片）。

    @return 标记为截断时为 True。
    """
    from observe.index import Index, borrow

    try:
        if Index._gpu_marker(gpu_events_path()).is_file():
            return True
    except Exception:  # noqa: BLE001 - 标记只是辅助事实，读不到时退回 meta
        pass
    with borrow() as held:
        row = held.conn.execute("SELECT value FROM meta WHERE key = 'gpu_events_truncated'").fetchone()
    return bool(row is not None and str(row[0]) == "1")


# 评测进度文件每次覆盖，盘上只有最近这一段。没有这份文件时这一道是空的。
def _eval_spans(progress: dict | None) -> list[dict]:
    """@brief 用评测进度文件生成一个评测区间；没有这份文件时为空。

    @param progress 解析后的 ``eval_progress.json``，可为 None。
    @return 最多一项的区间列表。
    """
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
    """@brief 组装详情页的三条时间线：问模型、等锁、评测。

    @param directory 运行目录。
    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @param progress 解析后的 ``eval_progress.json``，可为 None。
    @param status ``liveness`` 给出的状态。
    @return 含 ``lanes`` 的字典。
    """
    origin = _as_float((state or {}).get("started_at"))
    if origin is None:
        origin = _as_float((result or {}).get("started_at"))
    return {
        "lanes": [
            {"label": "问模型", "spans": _model_spans(directory, origin)},
            {"label": "等锁", "spans": _lock_spans(state, result, progress, status)},
            {"label": "评测", "spans": _eval_spans(progress)},
        ],
        # 源日志被人为删除/替换过时明确标出来，不假装历史完整。
        "truncated_history": gpu_history_truncated(),
    }


_RECHECK_PASSED = {"ok", "passed", "pass"}


def _recheck_failed(value) -> bool:
    """@brief 判断正确性复查字段是否表示失败。

    @param value 复查字段值，可能是布尔、字符串或含 ``class`` / ``status`` 的字典。
    @return 失败时为 True；None、False 或通过的取值返回 False。
    """
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
    """@brief 从轨迹里取出“噪声内没换种子”“正确性复查没过”两个标记，放在状态旁。

    @param trace 轨迹列表。
    @return 每项含 ``label`` 与 ``marks`` 的列表；轨迹不是列表时为空。
    """
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
    """@brief 取发布门放行的确认点；门没放行时为 None，不用搜索期 ``best_speedup`` 顶替。

    原始英文说明：If the gate did not pass, this is None rather than the search's best_speedup.

    @param result 解析后的 ``result.json``，可为 None。
    @param trace 轨迹列表，用于在缺花费时回退。
    @param conclusion ``conclusion_of_run`` 的结论。
    @return 含花费、加速比、批次与最坏/最好值的确认点；无确认时返回 None。
    """
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
    values = [float(v) for v in (result.get("confirmed_values") or []) if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {
        "cost_usd": cost,
        "speedup": speed,
        "search_speedup": _as_float(result.get("best_speedup")),
        "batches": values,
        "worst": _as_float(result.get("confirmed_worst")),
        "best": _as_float(result.get("confirmed_best")),
        "label": "确认",
    }


def _committee_models(directory: Path) -> dict[str, str]:
    """@brief 从 ``run_config.json`` 里取 first / second 两个委员会各自主成员的模型名。

    @param directory 运行目录。
    @return 委员会名到模型名的映射（缺的省略）。
    """
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
    """@brief 把轮次值转成非负整数；非法时返回 None。

    @param value 轮次字段值。
    @return 非负整数或 None。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number >= 0 else None


def _highest_trace_round(trace) -> int:
    """@brief 从轨迹标签（形如 ``r3``）里取出最大轮次。

    @param trace 轨迹列表。
    @return 最大轮次；没有可用标签时返回 0。
    """
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
    """@brief 推算每个轮次由哪个委员会接手：初稿与第 1 轮是 first，之后第 r 轮是 (r - 1) % 2。

    没有 ``committee`` 这一键的旧运行不编造交接记录。

    @param directory 运行目录。
    @param state 解析后的 ``state.json``，可为 None。
    @param result 解析后的 ``result.json``，可为 None。
    @param trace 轨迹列表。
    @return 含 ``recorded``、``note`` 与 ``rounds`` 的字典。
    """
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
# trace 与 confirmations 不再拷贝：它们与 state.trace / result.confirmations 逐字节相同，走 /raw 取。
def run_detail(run_id: str, *, include: frozenset[str] | None = None) -> dict | None:
    """@brief 把一个运行的 state 与 result 叠成详情负载；截断的那份为 None，另一份仍提供轨迹、花费与停止原因。

    ``include`` 为 None 时装配全部面板（旧行为）；给定集合时只装配集合里的重面板，
    轻量字段（状态、预算、发布、胜者、计时、确认点、交接）始终保留。传空集即得 L1 摘要。

    @param run_id 运行 id。
    @param include 需要装配的重面板名（``candidates``/``budget_events``/``timeline``/``candidate_marks``/
        ``llm``/``state``/``result``/``pending_evaluation``）。
    @return 详情字典；运行不存在时返回 None。
    """
    directory = resolve_run(run_id)
    if directory is None:
        return None
    state = _load(directory / "state.json")
    result = _load(directory / "result.json")
    trace = (state or {}).get("trace") or (result or {}).get("trace") or []
    status = liveness(state, result, directory)
    spent, budget = money_fields(directory, state, result)
    stop = (result or {}).get("stop_reason") or (state or {}).get("stop_reason") or ""
    alert_budget, alert_stop = _budget_for_alert(directory, spent, budget, stop)
    progress = _age(_load(directory / "eval_progress.json"))
    conclusion = conclusion_of_run(directory, result) if result else {"speedup": None, "publication": "in_progress"}
    config = _load(directory / "run_config.json") or {}
    named = config.get("target") if isinstance(config.get("target"), dict) else {}
    target_id = named.get("id") or (state or {}).get("target_id") or (result or {}).get("target_id") or ""
    backend = named.get("backend") or (state or {}).get("backend") or (result or {}).get("backend") or ""
    backend_version = named.get("backend_version") or (state or {}).get("backend_version") or (result or {}).get("backend_version") or ""
    document = {
        "id": _rel(directory),
        "status": status,
        "target_id": target_id,
        "backend": backend,
        "backend_version": backend_version,
        "target_line": f"目标: {target_id} ({backend}{('/' + backend_version) if backend_version else ''})" if target_id else "",
        "budget": budget_status(spent, alert_budget, alert_stop, status == "running"),
        "control": control_hint(directory, state, result, status),
        **topup_fields(directory),
        "stop_reason": stop,
        "task": (result or {}).get("task") or (state or {}).get("task") or "",
        "best_label": (state or {}).get("best_label") or (result or {}).get("best_label") or "",
        "confirm_winner_label": (result or {}).get("confirm_winner_label") or "",
        "inflight_label": (state or {}).get("inflight_label") or "",
        "kernel_writer": kernel_writer(directory),
        "winner": {
            "label": (result or {}).get("winner_label") or (result or {}).get("confirm_winner_label") or (result or {}).get("best_label"),
            "provider": (result or {}).get("winner_provider"),
            "model": (result or {}).get("winner_model"),
            "committee": (result or {}).get("winner_committee"),
            "origin": (result or {}).get("winner_origin"),
        },
        "model_totals": (result or {}).get("model_totals") or [],
        "publication": conclusion["publication"],
        "published_speedup": conclusion["speedup"],
        "eval_progress": progress,
        "timing": _run_timing(directory, state, result, progress, status),
        "confirmed_point": _confirmed_point(result, trace, conclusion),
        "handoff": _handoff(directory, state, result, trace),
    }
    if include is None or "pending_evaluation" in include:
        document["pending_evaluation"] = pending_evaluation(directory)
    if include is None or "budget_events" in include:
        document["budget_events"] = _budget_events(directory)
    if include is None or "state" in include:
        document["state"] = state
    if include is None or "result" in include:
        document["result"] = result
    if include is None or "candidates" in include:
        document["candidates"] = _candidates(directory)
    if include is None or "llm" in include:
        document["llm"] = _load(directory / "llm.json")
    if include is None or "timeline" in include:
        document["timeline"] = _timeline(directory, state, result, progress, status)
    if include is None or "candidate_marks" in include:
        document["candidate_marks"] = _candidate_marks(trace)
    return document

