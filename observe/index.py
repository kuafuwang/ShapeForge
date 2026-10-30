"""Rebuildable SQLite index of ``runs/``. One writer. Readers open the file read-only.

The database is a derivative. Deleting it and opening it again rebuilds the rows from ``runs/``,
``data/books.sqlite`` (read-only, and only to seed the carry-over) and ``ledger_carryover.json``.
A schema version this process does not know is dropped and built again. There is no migration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat as statlib
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from observe import detail, failure_sig, legacy
from observe.budget import budget_status
from observe.failure_sig import MESSAGE_CHARS
from observe.ledger import month_of, operator_name
from taic.redact import redact_text
from tcenv import index_journal

INDEX_SCHEMA_VERSION = 13

#: Per-candidate report state.  Only ``current`` (schema 4) is current evidence; an older report is
#: marked ``old_format`` and an unreadable one ``unreadable`` — this index never infers the build or
#: evaluation identity of a report it cannot validate, and it never upgrades one.
from taic.artifacts import REPORT_STATES  # noqa: F401 - re-exported for the index schema docs/tests

# ``stop_code`` 的重算规则版本：规则变了就改这个串，打开索引时会把旧行重算一次（只改这一列）。
STOP_CODE_RULE = "first-word-24"


def _diagnostic(value, limit: int = 400) -> str:
    """One bounded, single-line diagnostic; response bodies are not indexed."""
    if not isinstance(value, str):
        return ""
    return redact_text(value, limit=limit, one_line=True)
_SKIP_DIRS = {"candidates", "__pycache__"}


class _WriteLock:
    """@brief 全局写锁：跨线程互斥、同线程可重入，并保留 ``locked()`` 自省。

    ``tick()`` 持锁时会顺手调 ``_sync_kernels()``，而 HTTP 线程也可能直接调公开的
    ``ensure_kernel_history()`` / ``publications()``。同一个线程重入必须放行（否则自己把自己锁死），
    跨线程仍然串行。``locked()`` 只回答“当前线程是否持锁”，与读路径测试的口径一致。
    """

    def __init__(self) -> None:
        """@brief 建一把可重入锁。
        """
        self._lock = threading.RLock()

    def __enter__(self):
        """@brief 进入临界区。

        @return 锁本身。
        """
        self._lock.acquire()
        return self

    def __exit__(self, *exc_info) -> bool:
        """@brief 离开临界区。

        @param exc_info 异常三元组，未使用。
        @return 永不吞异常。
        """
        self._lock.release()
        return False

    def locked(self) -> bool:
        """@brief 当前线程是否持有这把锁。

        @return 持有时为 True。
        """
        return self._lock._is_owned()


_WRITE = _WriteLock()
_SINGLE: "Index | None" = None


class CarryoverError(RuntimeError):
    """@brief 结转文件与记录的校验和不符时抛出，需要人来查看。

    原始英文说明：The carry-over file does not match its recorded checksum. A person has to look at it.
    """


def db_path() -> Path:
    """@brief 返回索引数据库路径，可用 ``OBSERVE_DB`` 覆盖。

    @return 默认 ``data/observe.sqlite`` 或环境变量指定的路径。
    """
    override = os.environ.get("OBSERVE_DB")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "observe.sqlite"


def books_path() -> Path:
    """@brief 返回只读的 ``books.sqlite`` 路径，可用 ``AI_LOWERLING_BOOKS`` 覆盖。

    @return 旧账本数据库路径。
    """
    override = os.environ.get("AI_LOWERLING_BOOKS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "books.sqlite"


def carryover_path() -> Path:
    """@brief 返回结转文件 ``ledger_carryover.json`` 的路径，可用 ``OBSERVE_CARRYOVER`` 覆盖。

    @return 结转文件路径。
    """
    override = os.environ.get("OBSERVE_CARRYOVER")
    if override:
        return Path(override)
    return db_path().with_name("ledger_carryover.json")


def preserved_ledger_path(db: Path | None = None) -> Path:
    """@brief 返回即将被替换的数据库把花费行暂存到的文件路径。

    ``drop_archived`` 故意保留 ``ledger`` 行，删除不能降低已记录的花费；旧版重建会随文件一起丢掉它们，
    所以先复制到这里，再回放到新库。``db`` 是正在使用的库（``Index`` 可能建在别的路径上）。
    原始英文说明：Where the spend rows of a database that is about to be replaced are kept.

    @param db 数据库路径，None 表示 ``db_path()``。
    @return ``ledger_preserved.json`` 的路径。
    """
    return (Path(db) if db is not None else db_path()).with_name("ledger_preserved.json")


def get_index() -> "Index":
    """@brief 返回进程内唯一的写者；``OBSERVE_DB`` 变化时会新建数据库。

    原始英文说明：The process-wide writer. A new ``OBSERVE_DB`` opens a new database.

    @return 当前进程的 ``Index`` 实例。
    """
    global _SINGLE
    path = db_path()
    with _WRITE:
        if _SINGLE is None or _SINGLE.path != path:
            _SINGLE = Index(path)
        return _SINGLE


def reset_singleton() -> None:
    """@brief 丢掉进程内的写者单例（不删库），只用于测试与重建路径。

    关掉写者连接前先关掉本线程的只读连接：只读连接不分线程清理，重建时留着会看见旧文件的
    已删索引页。
    """
    global _SINGLE
    with _WRITE:
        _SINGLE = None
    conn = getattr(_READERS, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    _READERS.conn = None
    _READERS.path = ""


@contextmanager
def borrow():
    """@brief 借出唯一的写连接，让请求线程能读它。

    原始英文说明：The one writer connection, held so a request thread can read it.

    @return 上下文管理器，产出 ``Index`` 实例。
    """
    index = get_index()
    with _WRITE:
        yield index


_READERS = threading.local()


def reader() -> sqlite3.Connection:
    """@brief 返回当前线程的只读连接（WAL 并发读），给后台预热这类长任务用。

    与 :func:`borrow` 的区别是**不占全局写锁**：预热每次要算几百毫秒到一秒，拿着写锁会让所有
    请求线程排队。只读连接看得到已提交的数据；换 ``OBSERVE_DB``、库被删重建或连接失效时自动
    重开一次。

    @return 只读 sqlite3 连接（行工厂为 ``Row``）。
    @raises sqlite3.Error 库还不存在或打不开时抛出，由调用方决定重试。
    """
    path = str(db_path())
    conn = getattr(_READERS, "conn", None)
    if conn is not None and getattr(_READERS, "path", "") == path:
        try:
            conn.execute("SELECT 1").fetchone()
            return conn
        except sqlite3.Error:
            try:
                conn.close()
            except sqlite3.Error:
                pass
    elif conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    _READERS.conn = conn
    _READERS.path = path
    return conn


def _canonical(doc: dict) -> str:
    """@brief 把字典序列化成稳定的 JSON 文本，用于计算校验和。

    @param doc 待序列化的字典。
    @return 键排序、无多余空白的 JSON 文本。
    """
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(path: Path) -> str:
    """@brief 取文件或目录的指纹（mtime 与大小）。

    @param path 路径。
    @return 指纹字符串；读不到时为空串。
    """
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _best_fingerprints(directory: Path) -> str:
    """@brief 顶层所有 ``best.*`` 的 ``(文件名, mtime_ns, size)``，按文件名排序。

    发布门固定读 ``best.metal``，所以只盯着“字典序第一个 best 文件”会在混入 ``best.aaa``
    时监视错文件。把全部候选的名字与 stat 一起纳入，增删改名与就地改写都能触发重摄取。

    @param directory 运行目录。
    @return 以 ``|`` 分隔的指纹文本；没有 best 文件时为空串。
    """
    try:
        entries = sorted(
            (item for item in os.scandir(directory) if item.name.startswith("best.") and item.is_file()),
            key=lambda item: item.name,
        )
    except OSError:
        return ""
    return "|".join(f"{item.name}:{_fingerprint(directory / item.name)}" for item in entries)


def _outbox_fingerprint(directory: Path) -> str:
    """@brief ``evaluation_outbox/`` 里每个请求 json 的 ``(名字, mtime_ns, size)``。

    远程回补报告会**就地改写**请求文件，目录 mtime 不变；只看目录会让 ``run_pending_eval``
    对着它要报告的那个事件陈旧。一次 ``scandir`` + 逐文件 ``stat``（请求数本身有界），不打开文件。
    原始英文说明：Which outbox requests changed, from names and stats only.

    @param directory 运行目录。
    @return 以 ``|`` 分隔的指纹；没有 outbox 时为空串。
    """
    root = directory / "evaluation_outbox"
    try:
        entries = sorted(
            (
                entry for entry in os.scandir(root)
                if entry.is_file() and entry.name.endswith(".json")
                and not entry.name.endswith(".result.json") and entry.name != "receipts.json"
            ),
            key=lambda entry: entry.name,
        )
    except OSError:
        return ""
    parts = []
    for entry in entries:
        try:
            stat = entry.stat()
        except OSError:
            continue
        parts.append(f"{entry.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _candidate_files_fingerprint(directory: Path) -> str:
    """@brief 每颗候选 ``report.json`` 与全部 ``kernel.*`` 的 ``(mtime_ns,size)``。

    候选报告可能被就地改写（远程评测回补、人工修报告），目录 mtime 不变；只指纹 candidates
    目录会让旧 message/failure_sig 永远留着。这里一次 scandir + 每颗候选 1 个报告 + 内核 stat。
    原始英文说明：Per-candidate report/kernel stats, so an in-place rewrite is noticed.

    @param directory 运行目录。
    @return 以 ``|`` 分隔的指纹；没有候选目录时为空串。
    """
    root = directory / "candidates"
    try:
        entries = sorted(
            (entry for entry in os.scandir(root) if entry.is_dir() and not entry.name.startswith(".")),
            key=lambda entry: entry.name,
        )
    except OSError:
        return ""
    parts = []
    for entry in entries:
        path = Path(entry.path)
        kernels = ",".join(
            f"{name}:{_fingerprint(path / name)}" for name in detail.candidate_kernel_files(path)
        )
        parts.append(f"{entry.name}:{_fingerprint(path / 'report.json')}={kernels}")
    return "|".join(parts)


# 摘要戳（``query._detail_stamp``）依赖的文件：预算/账本/锁/评测进度，加 ``_outbox_fingerprint``。
# 这批指纹很小（4 次 stat + 一次 outbox scandir），可以进摘要戳；候选指纹不行——它随候选数无界增长，
# 而且一颗非 winner 的报告被改写不该让首屏摘要换戳。
SUMMARY_EXTRA_FILES = ("budget_events.json", "ledger.json", "resume.lock", "eval_progress.json")

# ``fp_extra`` 把「摘要依赖」与「其余摄取新鲜度」拼成一列：前半段可安全地进摘要戳，后半段只用于摄取判定。
EXTRA_SEPARATOR = "\x1e"


def summary_extra_fingerprint(directory: Path) -> str:
    """@brief ``fp_extra`` 里摘要戳依赖的那半段：四个小文件 stat + outbox 逐文件 stat。

    摘要在首屏展示预算（``budget_events.json``/``ledger.json``）、能否续跑（``resume.lock``）、
    评测进度（``eval_progress.json``）与待评测计数（``evaluation_outbox``），所以摘要戳必须含这五项；
    它们都是就地改写而不动运行目录 mtime 的来源。
    原始英文说明：The part of the extra fingerprint the L1 summary depends on.

    @param directory 运行目录。
    @return 以 ``|`` 分隔的指纹。
    """
    return "|".join([
        *(_fingerprint(directory / name) for name in SUMMARY_EXTRA_FILES),
        _outbox_fingerprint(directory),
    ])


_WRITER_MARKERS = (b'"event": "agent"', b'"event":"agent"')
_WRITER_CHUNK = 65_536


def _scan_agent_events(path: Path, offset: int = 0) -> tuple[bool, int]:
    """@brief 从 ``offset`` 起按块扫描 transcript 里的 ``event=agent``，返回（是否出现, 可前移到的偏移）。

    与 ``detail.kernel_writer`` 同义：**未终止的尾行也参与判断**（一条没有换行结尾的 agent 记录同样
    说明这是 hmz 写核的运行）。按 64KB 块读、块间保留“标记长度减一”的重叠，所以标记跨块也能命中，
    也不会因为一条超长尾行把整段内容读进内存；一旦命中就停。返回值里的偏移只前移到最后一个完整行
    之后，尾行下次还会重读。

    原始英文说明：Does the transcript contain an agent event? Chunked, tail-line aware, early exit.

    @param path transcript 路径。
    @param offset 从这里开始读（只增扫描用）；0 表示从头。
    @return ``(是否出现 agent 事件, 可以安全前移到的字节偏移)``。
    """
    overlap = max(len(marker) for marker in _WRITER_MARKERS) - 1
    found = False
    carry = b""
    base = offset
    last_line_end = None
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            while True:
                chunk = handle.read(_WRITER_CHUNK)
                if not chunk:
                    break
                if any(marker in carry + chunk for marker in _WRITER_MARKERS):
                    found = True
                    break
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    last_line_end = base + newline + 1
                base += len(chunk)
                carry = chunk[-overlap:] if overlap else b""
    except OSError:
        return False, offset
    return found, (last_line_end if last_line_end is not None else offset)


def _recorded_kernel_agent(directory: Path) -> str | None:
    """@brief ``run_config.json`` 里记下的 ``search.kernel_agent``；没有记录时为 None。

    与 ``detail.kernel_writer`` 的优先级逐字一致：**任何字符串都算已记录**（包括空串——它同样是
    “自己调模型”，只取 ``chat``），只有键缺失或不是字符串才回退到 ``agent_work``/transcript。
    以前用真值判断会把 ``''`` 当成“没记录”，随后被兜底来源覆盖成 ``pi``。
    原始英文说明：The recorded ``search.kernel_agent`` (any string, including empty), or None.

    @param directory 运行目录。
    @return 记下的取值，或 None。
    """
    try:
        doc = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    search = doc.get("search") if isinstance(doc, dict) else None
    recorded = search.get("kernel_agent") if isinstance(search, dict) else None
    return recorded if isinstance(recorded, str) else None


def split_extra_fingerprint(extra: str) -> tuple[str, str]:
    """@brief 把已存的 ``fp_extra`` 拆成（摘要依赖部分, 其余摄取新鲜度部分）。

    旧库里的 ``fp_extra`` 没有分隔符（升级前写入的行），此时整串都当摄取新鲜度、摘要部分为空串：
    这类行最多在一个摄取周期后自愈，不会把候选指纹误当摘要依赖。
    原始英文说明：The stored extra fingerprint, split into its summary part and the ingestion-only rest.

    @param extra 索引里存的 ``fp_extra``，可为 None 或空串。
    @return ``(摘要依赖部分, 其余部分)``。
    """
    summary, separator, rest = str(extra or "").partition(EXTRA_SEPARATOR)
    return (summary, rest) if separator else ("", summary)


def _target_id(config: dict | None) -> str:
    """@brief 从运行配置取目标硬件 id，缺省为 ``apple-m4-max``。

    @param config 解析后的 ``run_config.json``，可为 None。
    @return 目标硬件 id。
    """
    target = (config or {}).get("target") if isinstance(config, dict) else None
    if isinstance(target, dict) and target.get("id"):
        return str(target["id"])
    return "apple-m4-max"


def _num(value) -> float | None:
    """@brief 把值转成浮点；None、布尔与非数值返回 None。

    @param value 待转换的值。
    @return 浮点数或 None。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _read_json_file(path: Path) -> dict | None:
    """@brief 读一个 JSON 对象；文件不存在或内容损坏时返回 None（不计数，供工作进程用）。

    @param path 文件路径。
    @return 解析后的字典或 None。
    """
    return detail._load(path) if path.exists() else None


def _candidate_payload(directory: Path, result: dict | None, state: dict | None,
                       reader=None) -> list[dict]:
    """@brief 把 ``candidates/`` 读成待写入行（纯读，无数据库访问）。

    串行摄取与并行冷建共用这一份逻辑：读报告的字段、脱敏消息、失败指纹与 ``failure_members``
    的成员字段都在这里决定，写库只按返回的顺序插入。

    @param directory 运行目录。
    @param result 解析后的 ``result.json``，可为 None。
    @param state 解析后的 ``state.json``，可为 None。
    @param reader 读 JSON 的函数；串行路径传入带 ``json_reads`` 计数的版本。
    @return 每个候选一项的字典列表。
    """
    read = reader or _read_json_file
    root = directory / "candidates"
    if not root.is_dir():
        return []
    trace = []
    for doc in (result, state):
        if isinstance(doc, dict) and isinstance(doc.get("trace"), list):
            trace = doc["trace"]
            break
    flags = {str(item.get("label")): bool(item.get("within_noise")) for item in trace if isinstance(item, dict)}
    improved = {str(item.get("label")): bool(item.get("improved")) for item in trace if isinstance(item, dict)}
    kernel_name = detail.target_file_names(directory)[0]
    rows = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        name = path.name
        idx = int(name[:3]) if len(name) >= 3 and name[:3].isdigit() else -1
        label = name[4:] if len(name) > 4 and name[3] == "_" else name
        report_path = path / "report.json"
        report = read(report_path)
        if not isinstance(report, dict) and not (path / kernel_name).is_file():
            continue
        # One definition of "which reports are current evidence", shared with the detail panel.
        report_state = detail.report_state(report_path, report)
        current = report_state == "current"
        build = report.get("build") if current and isinstance(report, dict) else None
        build = build if isinstance(build, dict) else {}
        # An old/damaged report states no measurement: ``status`` is the format marker, the numeric
        # columns stay empty and it joins no failure cluster (a signature would be a claim about a
        # report this reader cannot validate).  The message stays as redacted diagnostic text.
        raw_status = str(report.get("status") or "") if isinstance(report, dict) else ""
        status = raw_status if current else report_state
        stage = str(report.get("stage") or "") if current and isinstance(report, dict) else ""
        benchmark = report.get("benchmark") if current and isinstance(report, dict) else None
        candidate = benchmark.get("candidate") if isinstance(benchmark, dict) else None
        reps = candidate.get("reps") if isinstance(candidate, dict) else None
        cache_hit = report.get("cache_hit") if current and isinstance(report, dict) else None
        failed = current and isinstance(report, dict) and raw_status not in ("", "ok")
        rows.append({
            "idx": idx,
            "label": label,
            "status": status,
            "stage": stage,
            "speedup": _num(report.get("speedup")) if current and isinstance(report, dict) else None,
            "within_noise": 1 if flags.get(label) else 0,
            "eligible": 1 if current and isinstance(report, dict) and report.get("eligible") is True else 0,
            "sha": _sha256_file(path / kernel_name),
            "protocol_version": str(report.get("protocol_version") or "") if current and isinstance(report, dict) else "",
            # Only a schema-4 report can state its build/evaluation identity.  Older ones keep the
            # columns empty and are marked ``old_format``; nothing is inferred from ``status`` or
            # ``result.json``.
            "report_state": report_state,
            "build_shape": (str(build.get("shape") or "") or None) if current else None,
            "build_kind": (str(build.get("build_kind") or "") or None) if current else None,
            "evaluation_fingerprint": (
                (str(report.get("evaluation_fingerprint") or "") or None) if current else None
            ),
            "updated": _safe_mtime(path),
            # message 对所有候选都存（脱敏、有界）：失败簇与 §10.3 的候选搜索共用这一列。
            "message": redact_text(report.get("message"), limit=MESSAGE_CHARS) if isinstance(report, dict) else "",
            "sig": failure_sig.signature(status, stage, report.get("message")) if failed else None,
            "cv": _num(candidate.get("cv")) if isinstance(candidate, dict) else None,
            "reps": int(reps) if isinstance(reps, int) and not isinstance(reps, bool) and reps >= 0 else None,
            "cache_hit": 1 if cache_hit is True else (0 if cache_hit is False else None),
            "improved": 1 if improved.get(label) else 0,
        })
    return rows


def _safe_mtime(path: Path) -> float | None:
    """@brief 取路径的 mtime；读不到时返回 None。

    @param path 任意路径。
    @return mtime 秒或 None。
    """
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _find_manifest_at(directory: Path, root_text: str | None = None) -> tuple[Path, dict] | None:
    """@brief 向上查找最近的 ``experiment.json``（纯读，供工作进程用）。

    @param directory 起始目录。
    @param root_text 已解析的 ``runs/`` 根文本；None 时取本进程缓存。
    @return ``(实验根目录, 清单)``；到根仍未找到时返回 None。
    """
    root_text = root_text or os.path.normpath(os.fspath(detail.root_path()[0]))
    current = os.path.normpath(os.fspath(directory))
    if not (current == root_text or current.startswith(root_text + os.sep)):
        return None
    while True:
        path = Path(current) / "experiment.json"
        if path.is_file():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                doc = None
            if isinstance(doc, dict):
                return Path(current), doc
        parent = os.path.dirname(current)
        if current == root_text or parent == current:
            return None
        current = parent


def _manifest_fingerprint_at(directory: Path, root_text: str | None = None) -> str:
    """@brief 最近的 ``experiment.json`` 的 ``(mtime_ns,size)`` 指纹（纯读）。

    @param directory 起始目录。
    @param root_text 已解析的 ``runs/`` 根文本。
    @return 指纹字符串；找不到时为空串。
    """
    root_text = root_text or os.path.normpath(os.fspath(detail.root_path()[0]))
    current = os.path.normpath(os.fspath(directory))
    if not (current == root_text or current.startswith(root_text + os.sep)):
        return ""
    while True:
        found = _fingerprint(Path(current) / "experiment.json")
        if found:
            return found
        parent = os.path.dirname(current)
        if current == root_text or parent == current:
            return ""
        current = parent


def _fingerprints_at(directory: Path, winner_label: str = "", winner_rel: str = "") -> tuple[str, str, str, str, str]:
    """@brief 只 stat 的五元指纹（纯读，供工作进程与实例方法共用）。

    @param directory 运行目录。
    @param winner_label 获胜候选标签。
    @param winner_rel 获胜内核相对路径。
    @return 五元组。
    """
    from taic.agent_feedback import PARSER_VERSION

    confirm = Index._confirmation_fingerprint(directory, winner_label, winner_rel)
    extra = EXTRA_SEPARATOR.join((
        summary_extra_fingerprint(directory),
        "|".join((
            _fingerprint(directory / "feedback.jsonl"),
            _fingerprint(directory / "feedback_review.json"),
            _fingerprint(directory / "candidates"),
            _candidate_files_fingerprint(directory),
            _manifest_fingerprint_at(directory),
            PARSER_VERSION,
        )),
    ))
    return (
        Index._state_fingerprint(directory),
        _fingerprint(directory / "result.json"),
        _fingerprint(directory / "run_config.json"),
        confirm,
        extra,
    )


def _payload_read_key(directory: Path) -> tuple:
    """@brief 并行 payload 的读取窗口键：内容读取前后必须相同。

    覆盖会被 payload 读取的文件（state/result/config/候选报告与内核/上行清单）。Writers 用原子替换，
    所以前后两次 stat/文件集相同就说明读到的字节属于同一份快照；不同就重试或退回串行。

    @param directory 运行目录。
    @return 可比较的元组。
    """
    return (
        _fingerprint(directory / "state.json"),
        _fingerprint(directory / "result.json"),
        _fingerprint(directory / "run_config.json"),
        _fingerprint(directory / "candidates"),
        _candidate_files_fingerprint(directory),
        _manifest_fingerprint_at(directory),
    )


def _read_run_payload(directory_text: str) -> dict | None:
    """@brief 并行冷建的工作单元：读一个 run 目录的全部纯读输入。

    只读文件、不碰数据库；主进程再按返回内容串行写 SQLite（保持单写者）。读取前后各算一次窗口键，
    不一致就重试；三次仍不稳时返回 None，主进程退回串行 ``_ingest``（它有自己的 stat/read/post-stat
    校验）。这样旧内容永远不会被绑到新指纹上。

    @param directory_text 运行目录的绝对路径文本。
    @return 含 ``state``/``result``/``config``/``fingerprints``/``manifest``/``candidates`` 的字典。
    """
    directory = Path(directory_text)
    root_text = os.path.normpath(os.fspath(detail.root_path()[0]))
    try:
        current = os.path.normpath(os.path.realpath(os.fspath(directory)))
    except OSError:
        current = os.path.normpath(os.fspath(directory))
    if not (current == root_text or current.startswith(root_text + os.sep)):
        return None
    directory = Path(current)
    for _attempt in range(3):
        before = _payload_read_key(directory)
        state = _read_json_file(directory / "state.json")
        result = _read_json_file(directory / "result.json")
        if not isinstance(state, dict) and not isinstance(result, dict):
            return None
        label = result.get("confirm_winner_label") if isinstance(result, dict) else ""
        if not isinstance(label, str):
            label = ""
        rel = _winner_rel(directory, label)
        config = _read_json_file(directory / "run_config.json")
        manifest = _find_manifest_at(directory, root_text)
        candidates = _candidate_payload(directory, result, state)
        fingerprints = _fingerprints_at(directory, label, rel)
        after = _payload_read_key(directory)
        if after == before:
            return {
                "state": state,
                "result": result,
                "config": config,
                "fingerprints": fingerprints,
                "manifest": manifest,
                "candidates": candidates,
                "read_key": before,
            }
    return None


def _rollup_day(started_at) -> str:
    """@brief 把开始时间折成 UTC 日（相对秒无法成天，记空串）。

    @param started_at unix 秒。
    @return ``YYYY-MM-DD`` 或空串。
    """
    value = _num(started_at)
    if value is None or value < 1_000_000_000:
        return ""
    return time.strftime("%Y-%m-%d", time.gmtime(value))


def _rollup_epoch(protocol_version) -> str:
    """@brief 协议纪元；``flame_chase_1`` 按既有的读侧规则折成 ``2``。

    @param protocol_version 运行行里的协议版本。
    @return 纪元字符串。
    """
    text = str(protocol_version or "")
    return "2" if text == "flame_chase_1" else text


def _rollup_fields(row: dict) -> tuple[str, str, str, str, str, str] | None:
    """@brief 一行运行对应的汇总桶键。

    缺失的 task/dtype 等按空串入桶（与 ``runs`` 行的投影一致），不丢行。

    @param row 含 ``task``/``dtype``/``size``/``target_id``/``protocol_version``/``started_at`` 的字典。
    @return 六元组或 None（行不存在）。
    """
    if not row:
        return None
    return (
        _rollup_day(row.get("started_at")),
        str(row.get("task") or ""),
        str(row.get("dtype") or ""),
        str(row.get("size") or ""),
        str(row.get("target_id") or ""),
        _rollup_epoch(row.get("protocol_version")),
    )


class Index:
    """@brief ``runs/`` 的可重建索引：唯一写者，读方以只读方式打开同一文件。

    数据库是导生产物。删掉再打开会从 ``runs/``、``data/books.sqlite``（只读，仅用于播种结转）与
    ``ledger_carryover.json`` 重建；版本不认识的库直接丢弃重建，没有迁移。
    """

    def __init__(self, path: Path) -> None:
        """@brief 记住路径并打开（必要时重建）数据库。

        @param path 数据库文件路径。
        """
        self.path = Path(path)
        self.stats = {"json_reads": 0, "dir_lists": 0, "turn_parses": 0}
        self.changed: list[str] = []
        self._in_reanchor = False
        self._last_full = 0.0
        # 一帧内只摄取同一目录一次：目录 mtime 与写者日志经常同时点名同一个 run。
        self._frame_ingested: set[str] = set()
        self._tables: dict[str, bool] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open()

    def _open(self) -> None:
        """@brief 校验版本后打开连接，建表并回放结转与保留的账本行。
        """
        if self.path.exists() and not self._version_ok():
            self._discard()
        self.conn = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._create()
        self._normalize_stop_codes()
        self._load_carryover()
        self._load_preserved_ledger()

    def _normalize_stop_codes(self) -> None:
        """@brief 一次性把旧的 ``stop_code`` 重算成当前规则（派生物，只改这一列）。

        ``stop_code`` 是摄入时写的窄列；规则从“冒号前第一段按字符截断”改成“冒号前第一段的
        第一个词”后，旧库里的 ``converged (12 rounds wit`` 与 ``converged (4 rounds with`` 会
        长期留在 L0 里，机器筛选看不出是同一类。索引是派生库，重建本来就会得到新值；这里用
        ``meta.stop_code_rule`` 记一次版本，避免每次打开都 O(N) 重算。规则再变时改
        :data:`STOP_CODE_RULE` 即可。
        """
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'stop_code_rule'").fetchone()
        if row is not None and str(row[0]) == STOP_CODE_RULE:
            return
        rows = self.conn.execute(
            "SELECT run_id, stop_reason, COALESCE(stop_code, '') AS stop_code"
            " FROM runs WHERE COALESCE(stop_reason, '') <> ''"
        ).fetchall()
        changed = [
            (detail.stop_code(item["stop_reason"]), item["run_id"])
            for item in rows
            if detail.stop_code(item["stop_reason"]) != item["stop_code"]
        ]
        if changed:
            self.conn.executemany("UPDATE runs SET stop_code = ? WHERE run_id = ?", changed)
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('stop_code_rule', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (STOP_CODE_RULE,),
        )
        self.conn.commit()

    def _version_ok(self) -> bool:
        """@brief 判断该文件是否由当前代码建立。

        被锁、忙或暂时读不到的数据库**不算**版本不符：丢弃它会删掉花费账本，包括 ``drop_archived``
        故意保留的行。只有缺表（旧版）或文件损坏才可以重建。
        原始英文说明：Whether this file was built by this code.

        @return 版本一致时为 True。
        """
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'index_schema_version'").fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return False
            raise
        except sqlite3.DatabaseError:
            return False
        finally:
            conn.close()
        return row is not None and str(row[0]) == str(INDEX_SCHEMA_VERSION)

    def _discard(self) -> None:
        """@brief 先保留花费行，再删除数据库文件及其 ``-wal`` / ``-shm``。

        保全失败时抛 :class:`CarryoverError` 并**不动旧库**：宁可这一轮建不出索引，也不能把
        已经发生的花费丢掉。
        """
        if not self._preserve_ledger():
            raise CarryoverError(
                f"cannot preserve the spend ledger of {self.path}; refusing to replace it"
            )
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix) if suffix else self.path
            try:
                candidate.unlink()
            except OSError:
                pass

    def _preserve_ledger(self) -> bool:
        """@brief 在删旧库前把花费行复制到旁路文件。

        返回 False 表示**有损失风险**：读不出账本、或有行但写不了旁路。调用方必须中止重建并保留
        旧库，钱的数据不能因为一次权限/磁盘故障消失。没有账本表或账本为空时返回 True（没有行
        可存），并清掉旧的旁路文件，避免把更早的状态回放进新库。
        原始英文说明：Copy the spend rows out before the file is unlinked; False means the caller must keep the old DB.

        @return 可以继续替换数据库时为 True。
        """
        try:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error:
            return False
        try:
            try:
                rows = [list(row) for row in conn.execute(
                    "SELECT run_id, operator, month, started_at, spent_max_usd, budget_usd FROM ledger"
                ).fetchall()]
            except sqlite3.OperationalError as exc:
                # 更老的库没有账本表：确实没有行可丢；其它读错误（锁、损坏）不能当作空账本。
                if "no such table" not in str(exc).lower():
                    return False
                rows = []
            except sqlite3.Error:
                return False
        finally:
            conn.close()
        if not rows:
            self._drop_preserved(preserved_ledger_path(self.path))
            return True
        path = preserved_ledger_path(self.path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
            try:
                tmp.write_text(json.dumps({"saved_at": time.time(), "rows": rows}, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, path)
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
        except OSError:
            return False
        return True

    def _load_preserved_ledger(self) -> None:
        """@brief 回放 :meth:`_preserve_ledger` 保存的行；upsert 只会抬高合计，不会降低。

        原始英文说明：Replay the rows saved by :meth:`_preserve_ledger`. The upsert never lowers a total.
        """
        path = preserved_ledger_path(self.path)
        if not path.is_file():
            return
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Unreadable: nothing can be restored, and keeping it would retry forever.
            self._drop_preserved(path)
            return
        rows = doc.get("rows") if isinstance(doc, dict) else None
        if not isinstance(rows, list):
            self._drop_preserved(path)
            return
        try:
            for row in rows:
                if not isinstance(row, list) or len(row) != 6:
                    continue
                self.conn.execute(
                    """
                    INSERT INTO ledger (run_id, operator, month, started_at, spent_max_usd, budget_usd)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        operator = COALESCE(excluded.operator, ledger.operator),
                        month = COALESCE(ledger.month, excluded.month),
                        started_at = COALESCE(ledger.started_at, excluded.started_at),
                        spent_max_usd = MAX(COALESCE(ledger.spent_max_usd, excluded.spent_max_usd), COALESCE(excluded.spent_max_usd, ledger.spent_max_usd)),
                        budget_usd = COALESCE(excluded.budget_usd, ledger.budget_usd)
                    """,
                    tuple(row),
                )
            self.conn.commit()
        except sqlite3.Error:
            # A locked or busy database is transient: keep the file for the next open to retry.
            return
        self._drop_preserved(path)

    @staticmethod
    def _drop_preserved(path: Path) -> None:
        """@brief 删除旁路暂存文件，不存在或删不掉都不报错。

        @param path 暂存文件路径。
        """
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _create(self) -> None:
        """@brief 建表、建索引并补齐旧库缺的列；同时写入 schema 版本等 meta。
        """
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                arm_id TEXT,
                trial INTEGER,
                kind TEXT,
                task TEXT,
                operator TEXT,
                dtype TEXT,
                size TEXT,
                target_id TEXT,
                backend TEXT,
                backend_version TEXT,
                kernel_sha256 TEXT,
                protocol_version TEXT,
                committees_json TEXT,
                model TEXT,
                preset TEXT,
                switch_delta_json TEXT,
                contract TEXT,
                status TEXT,
                phase TEXT,
                stop_reason TEXT,
                started_at REAL,
                updated_at REAL,
                spent_usd REAL,
                spent_max_usd REAL,
                budget_usd REAL,
                granted_budget_usd REAL,
                publication TEXT,
                published_speedup REAL,
                naive_speedup REAL,
                search_best_speedup REAL,
                stop_code TEXT,
                control_action TEXT,
                control_reason TEXT,
                topped_up INTEGER,
                initial_budget_usd REAL,
                kernel_writer TEXT,
                rounds INTEGER,
                evaluations INTEGER,
                inflight_label TEXT,
                pid INTEGER,
                source TEXT,
                binding TEXT,
                fp_state TEXT,
                fp_result TEXT,
                fp_config TEXT,
                fp_confirm TEXT,
                fp_extra TEXT,
                turn_bytes INTEGER NOT NULL DEFAULT 0,
                feedback_bytes INTEGER NOT NULL DEFAULT 0,
                feedback_review_fp TEXT NOT NULL DEFAULT '',
                feedback_parser TEXT NOT NULL DEFAULT '',
                row_json TEXT
            );
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                kind TEXT,
                driver TEXT,
                created_at REAL,
                git_commit TEXT,
                frozen_json TEXT,
                config_hash TEXT,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS arms (
                experiment_id TEXT,
                arm_id TEXT,
                delta_json TEXT,
                planned_n INTEGER,
                task_fingerprint TEXT,
                PRIMARY KEY (experiment_id, arm_id)
            );
            CREATE TABLE IF NOT EXISTS ledger (
                run_id TEXT PRIMARY KEY,
                operator TEXT,
                month TEXT,
                started_at REAL,
                spent_max_usd REAL,
                budget_usd REAL
            );
            CREATE TABLE IF NOT EXISTS ledger_carryover (
                id INTEGER PRIMARY KEY,
                created_at REAL,
                rows INTEGER,
                total_usd REAL,
                origin TEXT,
                sha256 TEXT
            );
            CREATE TABLE IF NOT EXISTS watched (
                path TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL
            );
            -- 直接包含 run 的目录。新 run 一定先改它父目录的 mtime；只看这一小撮目录就能发现
            -- 新运行，不必扫描全部 watched（O(全部历史 run) 的成本留给周期对账）。
            CREATE TABLE IF NOT EXISTS run_parents (
                path TEXT PRIMARY KEY
            );
            CREATE INDEX IF NOT EXISTS runs_experiment ON runs(experiment_id);
            CREATE INDEX IF NOT EXISTS runs_arm ON runs(arm_id);
            CREATE INDEX IF NOT EXISTS runs_target ON runs(target_id);
            CREATE INDEX IF NOT EXISTS runs_status ON runs(status);
            CREATE INDEX IF NOT EXISTS runs_started ON runs(started_at);
            CREATE INDEX IF NOT EXISTS runs_binding ON runs(binding);
            CREATE TABLE IF NOT EXISTS candidates (
                run_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                label TEXT NOT NULL,
                status TEXT,
                stage TEXT,
                speedup REAL,
                message TEXT,
                failure_sig TEXT,
                cv REAL,
                reps INTEGER,
                cache_hit INTEGER,
                within_noise INTEGER NOT NULL DEFAULT 0,
                eligible INTEGER NOT NULL DEFAULT 0,
                kernel_sha256 TEXT,
                protocol_version TEXT,
                report_state TEXT,
                build_shape TEXT,
                build_kind TEXT,
                evaluation_fingerprint TEXT,
                updated_at REAL,
                PRIMARY KEY (run_id, idx)
            );
            CREATE INDEX IF NOT EXISTS candidates_run ON candidates(run_id);
            CREATE INDEX IF NOT EXISTS candidates_status ON candidates(status);
            CREATE INDEX IF NOT EXISTS candidates_sha ON candidates(kernel_sha256);
            CREATE TABLE IF NOT EXISTS sessions (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                label TEXT,
                candidate_label TEXT,
                attempt INTEGER NOT NULL DEFAULT 0,
                committee TEXT,
                provider TEXT,
                model TEXT,
                outcome TEXT,
                cost_usd REAL,
                failure_sig TEXT,
                session_id TEXT,
                failed_turns INTEGER NOT NULL DEFAULT 0,
                last_kind TEXT,
                failure_class TEXT,
                error TEXT,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                diagnostics_path TEXT,
                tokens_input INTEGER NOT NULL DEFAULT 0,
                tokens_output INTEGER NOT NULL DEFAULT 0,
                tokens_cache_read INTEGER NOT NULL DEFAULT 0,
                tokens_cache_write INTEGER NOT NULL DEFAULT 0,
                t REAL,
                PRIMARY KEY (run_id, seq)
            );
            CREATE INDEX IF NOT EXISTS sessions_run ON sessions(run_id);
            CREATE INDEX IF NOT EXISTS sessions_model ON sessions(model);
            CREATE INDEX IF NOT EXISTS sessions_committee ON sessions(committee);
            CREATE INDEX IF NOT EXISTS runs_live_started ON runs(COALESCE(started_at, -1) DESC, run_id DESC) WHERE status <> 'missing';
            CREATE INDEX IF NOT EXISTS runs_live_speedup ON runs(COALESCE(published_speedup, -1) DESC, run_id DESC) WHERE status <> 'missing';
            CREATE INDEX IF NOT EXISTS runs_live_spent ON runs(COALESCE(spent_usd, -1) DESC, run_id DESC) WHERE status <> 'missing';
            CREATE INDEX IF NOT EXISTS runs_confirmed ON runs(published_speedup DESC, run_id) WHERE publication = 'confirmed';
            CREATE TABLE IF NOT EXISTS turns (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                label TEXT,
                kind TEXT,
                grp INTEGER NOT NULL DEFAULT 0,
                row_json TEXT,
                PRIMARY KEY (run_id, seq)
            );
            CREATE INDEX IF NOT EXISTS turns_run_seq ON turns(run_id, seq);
            CREATE TABLE IF NOT EXISTS feedback_items (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                t REAL,
                task TEXT,
                dtype TEXT,
                size TEXT,
                label TEXT,
                committee TEXT,
                provider TEXT,
                model TEXT,
                outcome TEXT,
                exit TEXT,
                kinds TEXT,
                line_at INTEGER,
                has_system INTEGER NOT NULL DEFAULT 0,
                pending_claims INTEGER NOT NULL DEFAULT 0,
                unverified_premises INTEGER NOT NULL DEFAULT 0,
                placeholder_items INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, seq)
            );
            CREATE INDEX IF NOT EXISTS feedback_task ON feedback_items(task);
            CREATE INDEX IF NOT EXISTS feedback_model ON feedback_items(model);
            CREATE TABLE IF NOT EXISTS feedback_claims (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                digest TEXT NOT NULL,
                section TEXT,
                audience TEXT,
                premise TEXT,
                status TEXT,
                verification TEXT,
                topic TEXT,
                change_ref TEXT,
                reviewed_at REAL,
                rev INTEGER NOT NULL DEFAULT 0,
                subclaims_sha256 TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (run_id, seq, digest)
            );
            CREATE INDEX IF NOT EXISTS feedback_digest ON feedback_claims(digest);
            CREATE INDEX IF NOT EXISTS feedback_topic ON feedback_claims(topic);
            CREATE INDEX IF NOT EXISTS feedback_status ON feedback_claims(status);
            CREATE TABLE IF NOT EXISTS feedback_subclaims (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                parent_digest TEXT NOT NULL,
                sub_index INTEGER NOT NULL,
                text TEXT,
                text_sha256 TEXT,
                digest TEXT,
                status TEXT,
                verification TEXT,
                evidence_source TEXT,
                evidence TEXT,
                note TEXT,
                change_ref TEXT,
                backend TEXT,
                target_id TEXT,
                derived_from TEXT,
                rev INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, seq, parent_digest, sub_index)
            );
            CREATE INDEX IF NOT EXISTS feedback_sub_status ON feedback_subclaims(status);
            CREATE INDEX IF NOT EXISTS feedback_sub_digest ON feedback_subclaims(digest);
            CREATE INDEX IF NOT EXISTS feedback_sub_sha ON feedback_subclaims(text_sha256);
            CREATE TABLE IF NOT EXISTS feedback_claim_removed (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                digest TEXT NOT NULL,
                rev INTEGER NOT NULL,
                section TEXT,
                audience TEXT,
                premise TEXT,
                status TEXT,
                reason TEXT,
                PRIMARY KEY (run_id, seq, digest)
            );
            CREATE INDEX IF NOT EXISTS feedback_removed_rev ON feedback_claim_removed(rev);
            -- schema 12 的新表。每个 PR 只写自己那张；建表放在这里，删库重建后逐字段一致。
            CREATE TABLE IF NOT EXISTS failure_members (
                run_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                sig TEXT,
                status TEXT,
                stage TEXT,
                observed_at REAL,
                PRIMARY KEY (run_id, idx)
            );
            CREATE INDEX IF NOT EXISTS failure_members_sig ON failure_members(sig);
            CREATE TABLE IF NOT EXISTS kernel_history (
                target_id TEXT,
                task TEXT,
                dtype TEXT,
                size TEXT,
                seq INTEGER,
                run_id_raw TEXT,
                run_id TEXT,
                winner_label TEXT,
                source_sha256 TEXT,
                confirmed_speedup REAL,
                confirmed_values_json TEXT,
                candidate_us REAL,
                baseline_us_json TEXT,
                task_fingerprint TEXT,
                calibration_sha256 TEXT,
                updated_at REAL,
                source_unknown INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (target_id, task, dtype, size, seq)
            );
            CREATE TABLE IF NOT EXISTS run_rollup (
                day TEXT NOT NULL,
                task TEXT NOT NULL,
                dtype TEXT NOT NULL,
                size TEXT NOT NULL,
                target_id TEXT NOT NULL,
                protocol_epoch TEXT NOT NULL,
                runs INTEGER NOT NULL DEFAULT 0,
                confirmed INTEGER NOT NULL DEFAULT 0,
                spent_usd REAL NOT NULL DEFAULT 0,
                published_sum REAL NOT NULL DEFAULT 0,
                published_max REAL,
                PRIMARY KEY (day, task, dtype, size, target_id, protocol_epoch)
            );
            CREATE TABLE IF NOT EXISTS confirmed_best (
                run_id TEXT PRIMARY KEY,
                task TEXT,
                dtype TEXT,
                size TEXT,
                target_id TEXT,
                protocol_epoch TEXT,
                published_speedup REAL,
                kernel_sha256 TEXT,
                confirmed_at REAL
            );
            CREATE INDEX IF NOT EXISTS confirmed_best_order ON confirmed_best(published_speedup DESC);
            CREATE TABLE IF NOT EXISTS run_pending_eval (
                run_id TEXT PRIMARY KEY,
                pending INTEGER NOT NULL DEFAULT 0,
                unavailable INTEGER NOT NULL DEFAULT 0,
                last_error_code TEXT,
                retryable INTEGER NOT NULL DEFAULT 0,
                requests_json TEXT,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS gpu_events (
                t REAL,
                pid INTEGER,
                kind TEXT,
                mode TEXT,
                what TEXT
            );
            CREATE INDEX IF NOT EXISTS gpu_events_t ON gpu_events(t);
            -- §5.5.2：按 pid 维护未配对的 wait，并把已配对的区间物化成 lock_spans，
            -- 这样时间线只查与 [lo,hi] 相交的 span，不再整读机全局事件日志。
            CREATE TABLE IF NOT EXISTS gpu_waits (
                pid INTEGER PRIMARY KEY,
                t REAL,
                mode TEXT,
                what TEXT,
                seq INTEGER
            );
            CREATE TABLE IF NOT EXISTS lock_spans (
                pid INTEGER,
                start REAL,
                end REAL,
                closed_at REAL,
                mode TEXT,
                what TEXT
            );
            CREATE INDEX IF NOT EXISTS lock_spans_window ON lock_spans(start, end);
            CREATE INDEX IF NOT EXISTS gpu_events_pid ON gpu_events(pid, t);
            CREATE INDEX IF NOT EXISTS gpu_waits_t ON gpu_waits(t);
            CREATE INDEX IF NOT EXISTS lock_spans_end ON lock_spans(end);
            -- §5.6 的窄投影必须有覆盖索引：否则汇总仍要扫每行的 row_json 页（10k 约 21ms）。
            CREATE INDEX IF NOT EXISTS runs_budget_cover ON runs(
                status, started_at, run_id, task, dtype, stop_reason, spent_usd,
                granted_budget_usd, budget_usd, control_action, control_reason, topped_up
            );
            -- 区间查询的有界读：普通 B-tree 只能在一侧开区间，另一侧会扫全部历史/未来行。
            -- 摄取时把每个 span 复制到它覆盖的 10 秒桶（超长 span 只进 lock_spans_long），
            -- 读时只查窗口覆盖的桶 + 很小的一张长 span 表。
            CREATE TABLE IF NOT EXISTS lock_span_slots (
                bucket INTEGER NOT NULL,
                span_id INTEGER NOT NULL,
                PRIMARY KEY (bucket, span_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS lock_spans_long (
                span_id INTEGER PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS blob_refs (
                sha256 TEXT NOT NULL,
                ref_kind TEXT NOT NULL,
                ref_id TEXT NOT NULL,
                created_at REAL,
                PRIMARY KEY (sha256, ref_kind, ref_id)
            );
            """
        )
        self._ensure_lock_span_slots()
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('index_schema_version', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(INDEX_SCHEMA_VERSION),),
        )
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('revision', '0') ON CONFLICT(key) DO NOTHING"
        )
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('turns_revision', '0') ON CONFLICT(key) DO NOTHING"
        )
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('feedback_revision', '0') ON CONFLICT(key) DO NOTHING"
        )
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(feedback_claims)")}
        if "rev" not in columns:
            self.conn.execute("ALTER TABLE feedback_claims ADD COLUMN rev INTEGER NOT NULL DEFAULT 0")
        if "subclaims_sha256" not in columns:
            self.conn.execute("ALTER TABLE feedback_claims ADD COLUMN subclaims_sha256 TEXT NOT NULL DEFAULT ''")
        self.conn.execute("CREATE INDEX IF NOT EXISTS feedback_claim_rev ON feedback_claims(rev)")
        run_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(runs)")}
        # §6.4：巡检的发布告警必须是索引投影，不能逐行读 run 目录。发布原因码与
        # ``result.confirmed_speedup`` 在摄入时写成列；旧库补列后清一次已存指纹，让下一次 tick
        # 重读（否则要等到文件变化才会填上，巡检会一直拿到空 reason）。
        backfill_reason = False
        for name, declaration in (("publication_reason", "TEXT"), ("result_confirmed_speedup", "REAL")):
            if name not in run_columns:
                self.conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {declaration}")
                backfill_reason = True
        if backfill_reason:
            self.conn.execute(
                "UPDATE runs SET fp_state = '', fp_result = '', fp_confirm = '', fp_extra = ''"
                " WHERE status != 'missing'"
            )
        if "feedback_bytes" not in run_columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN feedback_bytes INTEGER NOT NULL DEFAULT 0")
        if "feedback_review_fp" not in run_columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN feedback_review_fp TEXT NOT NULL DEFAULT ''")
        if "feedback_parser" not in run_columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN feedback_parser TEXT NOT NULL DEFAULT ''")
        if "backend" not in run_columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN backend TEXT")
        if "backend_version" not in run_columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN backend_version TEXT")
        session_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(sessions)")}
        for name, declaration in (
            ("candidate_label", "TEXT"),
            ("attempt", "INTEGER NOT NULL DEFAULT 0"),
            ("session_id", "TEXT"),
            ("failed_turns", "INTEGER NOT NULL DEFAULT 0"),
            ("last_kind", "TEXT"),
            ("failure_class", "TEXT"),
            ("error", "TEXT"),
            ("tool_calls", "INTEGER NOT NULL DEFAULT 0"),
            ("diagnostics_path", "TEXT"),
        ):
            if name not in session_columns:
                self.conn.execute(f"ALTER TABLE sessions ADD COLUMN {name} {declaration}")
        item_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(feedback_items)")}
        if "line_at" not in item_columns:
            self.conn.execute("ALTER TABLE feedback_items ADD COLUMN line_at INTEGER")
        if "placeholder_items" not in item_columns:
            self.conn.execute("ALTER TABLE feedback_items ADD COLUMN placeholder_items INTEGER NOT NULL DEFAULT 0")
        removed_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(feedback_claim_removed)")}
        for name in ("section", "audience", "premise", "status", "reason"):
            if name not in removed_columns:
                self.conn.execute(f"ALTER TABLE feedback_claim_removed ADD COLUMN {name} TEXT")
        # 同一版本号内的防御性补列。跨版本会删库重建，所以这里只处理“12 版旧代码建的库”。
        candidate_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(candidates)")}
        backfill_improved = False
        for name, declaration in (
            ("message", "TEXT"), ("failure_sig", "TEXT"), ("cv", "REAL"), ("reps", "INTEGER"), ("cache_hit", "INTEGER"),
            # §8.3：种子链从 trace 的 ``improved`` 序列推导；它在摄入时写进窄列，读时不必再翻
            # ``result.json`` / ``state.json``（否则详情面板会重新读整份轨迹）。
            ("improved", "INTEGER"),
            # §8：只有 schema 4 报告能给出 build/评测身份；旧报告标 ``old_format``，四列留空。
            ("report_state", "TEXT"), ("build_shape", "TEXT"), ("build_kind", "TEXT"),
            ("evaluation_fingerprint", "TEXT"),
        ):
            if name not in candidate_columns:
                self.conn.execute(f"ALTER TABLE candidates ADD COLUMN {name} {declaration}")
                backfill_improved = True
        if backfill_improved:
            # 旧行的新列是 NULL，而指纹没变时 ``_ingest`` 会早退；清一次指纹让下一次 tick 重读。
            self.conn.execute(
                "UPDATE runs SET fp_state = '', fp_result = '', fp_confirm = '', fp_extra = ''"
                " WHERE status != 'missing'"
            )
        for name, declaration in (
            ("search_best_speedup", "REAL"), ("stop_code", "TEXT"), ("control_action", "TEXT"),
            ("control_reason", "TEXT"), ("topped_up", "INTEGER"), ("initial_budget_usd", "REAL"),
            # 摘要字段 kernel_writer 的摄入窄投影：旧 run 上它来自 transcript/agent_work 扫描，
            # 不能在每次摘要请求里重扫，也不能让戳与负载各算一遍。
            ("kernel_writer", "TEXT"),
        ):
            if name not in run_columns:
                self.conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {declaration}")
        if "failure_sig" not in session_columns:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN failure_sig TEXT")
        turn_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(turns)")}
        if "grp" not in turn_columns:
            self.conn.execute("ALTER TABLE turns ADD COLUMN grp INTEGER NOT NULL DEFAULT 0")
        self.conn.execute("CREATE INDEX IF NOT EXISTS gpu_waits_seq ON gpu_waits(seq)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS candidates_sig ON candidates(failure_sig) WHERE failure_sig IS NOT NULL")
        self.conn.execute("CREATE INDEX IF NOT EXISTS sessions_sig ON sessions(failure_sig) WHERE failure_sig IS NOT NULL")
        self.conn.execute("CREATE INDEX IF NOT EXISTS turns_run_grp ON turns(run_id, grp, seq)")
        span_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(lock_spans)")}
        if "closed_at" not in span_columns:
            self.conn.execute("ALTER TABLE lock_spans ADD COLUMN closed_at REAL")
        wait_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(gpu_waits)")}
        if "seq" not in wait_columns:
            self.conn.execute("ALTER TABLE gpu_waits ADD COLUMN seq INTEGER")
        # FTS 投影与同步触发器要先于建表完成校验/修复，见 _ensure_fts()。
        self._ensure_fts()
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('built_at', ?) ON CONFLICT(key) DO NOTHING",
            (str(time.time()),),
        )
        self.conn.commit()

    def _fts_columns(self, name: str) -> tuple[str, ...]:
        """@brief 取 FTS 投影声明的列；不存在时返回空元组。

        @param name 虚拟表名。
        @return 列名元组。
        """
        try:
            return tuple(row[1] for row in self.conn.execute(f"PRAGMA table_info({name})"))
        except sqlite3.Error:
            return ()

    def _fts_trigger_sql(self, table: str, name: str, columns: tuple[str, ...]) -> dict[str, str]:
        """@brief 生成 FTS 同步触发器的规范 SQL。

        索引列就是投影声明的那几列，按同一顺序写进 INSERT/DELETE。

        @param table 内容表名（``runs`` / ``candidates``）。
        @param name FTS 表名。
        @param columns 投影列。
        @return 触发器名到 SQL 的映射。
        """
        listed = ", ".join(columns)
        new_values = ", ".join(f"new.{column}" for column in columns)
        old_values = ", ".join(f"old.{column}" for column in columns)
        return {
            f"{name}_ai": (
                f"CREATE TRIGGER {name}_ai AFTER INSERT ON {table} BEGIN "
                f"INSERT INTO {name}(rowid, {listed}) VALUES (new.rowid, {new_values}); END"
            ),
            f"{name}_ad": (
                f"CREATE TRIGGER {name}_ad AFTER DELETE ON {table} BEGIN "
                f"INSERT INTO {name}({name}, rowid, {listed}) "
                f"VALUES ('delete', old.rowid, {old_values}); END"
            ),
            f"{name}_au": (
                f"CREATE TRIGGER {name}_au AFTER UPDATE ON {table} BEGIN "
                f"INSERT INTO {name}({name}, rowid, {listed}) "
                f"VALUES ('delete', old.rowid, {old_values}); "
                f"INSERT INTO {name}(rowid, {listed}) VALUES (new.rowid, {new_values}); END"
            ),
        }

    def _fts_structure_ok(self, name: str, table: str, columns: tuple[str, ...]) -> bool:
        """@brief 校验一个 FTS5 投影：列形状、``content``/``content_rowid`` 声明、触发器与内容一致性。

        ``integrity-check`` 会对照内容表检查每一行；缺失的索引行或错误的声明都会在这里报错，
        而不是等到下一次写入时才发现。

        @param name 虚拟表名。
        @param table 内容表名。
        @param columns 期望的投影列。
        @return 一切正常时为 True。
        """
        if self._fts_columns(name) != columns:
            return False
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        declaration = "".join(str(row[0] or "").lower().split()) if row is not None else ""
        if f"content='{table}'" not in declaration or "content_rowid='rowid'" not in declaration:
            return False
        expected = {key: "".join(sql.split()) for key, sql in self._fts_trigger_sql(table, name, columns).items()}
        stored = {
            item[0]: "".join(str(item[1] or "").split())
            for item in self.conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name IN (?, ?, ?)",
                tuple(expected),
            )
        }
        if stored != expected:
            return False
        return True

    def _ensure_fts(self) -> None:
        """@brief 校验并修复两张 FTS5 投影：列/声明/触发器/内容任何一项不对就重建。

        §10.3 要求 ``runs_fts`` 投影 ``run_id/task/dtype/target_id/model/stop_reason``。上一版把
        ``operator`` 写进去过，所以同版本旧库也要能自己修好，而不是只在“表不存在时”补一次。

        原始英文说明：Validate and repair both FTS projections, including triggers and content.
        """
        specs = (
            ("runs_fts", "runs", ("run_id", "task", "dtype", "target_id", "model", "stop_reason")),
            ("cands_fts", "candidates", ("label", "message")),
        )
        for name, table, columns in specs:
            if not self._fts_structure_ok(name, table, columns):
                triggers = self._fts_trigger_sql(table, name, columns)
                for key in triggers:
                    self.conn.execute(f"DROP TRIGGER IF EXISTS {key}")
                self.conn.execute(f"DROP TABLE IF EXISTS {name}")
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE {name} USING fts5({', '.join(columns)}, "
                    f"content='{table}', content_rowid='rowid', tokenize='unicode61')"
                )
                for key, sql in triggers.items():
                    self.conn.execute(sql)
            # external-content 的索引内容无法廉价对拍（本构建没有 fts5vocab，integrity-check
            # 也不校验同 rowid 的 token）。每次进程打开都从源表整体重建一次：这是 fail-closed
            # 的等价保证，任何越权改写都会在下次打开时被抹掉。
            try:
                self.conn.execute(f"INSERT INTO {name}({name}) VALUES ('rebuild')")
            except sqlite3.DatabaseError:
                self.conn.execute(f"DROP TABLE IF EXISTS {name}")
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE {name} USING fts5({', '.join(columns)}, "
                    f"content='{table}', content_rowid='rowid', tokenize='unicode61')"
                )
                for key, sql in self._fts_trigger_sql(table, name, columns).items():
                    self.conn.execute(f"DROP TRIGGER IF EXISTS {key}")
                    self.conn.execute(sql)
                self.conn.execute(f"INSERT INTO {name}({name}) VALUES ('rebuild')")

    def _load_carryover(self) -> None:
        """@brief 有结转文件就校验并导入；没有则从旧 ``books.sqlite`` 现算一次并写出。
        """
        path = carryover_path()
        if path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            body = {key: doc[key] for key in ("created_at", "rows", "total_usd", "origin")}
            digest = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
            if digest != doc.get("sha256"):
                raise CarryoverError(f"{path} checksum does not match; refusing to build the index")
            self._insert_carryover(doc)
            return
        books = books_path()
        if not books.is_file():
            return
        self._import_books(books)

    def _import_books(self, books: Path) -> None:
        """@brief 只读打开旧账本库，汇总历史花费并写出结转文件。

        @param books ``books.sqlite`` 路径。
        """
        try:
            conn = sqlite3.connect(f"file:{books}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error:
            return
        try:
            rows = conn.execute("SELECT spent_usd FROM search").fetchall()
        except sqlite3.Error:
            return
        finally:
            conn.close()
        total = 0.0
        for (spent,) in rows:
            number = _num(spent)
            if number is not None:
                total += number
        doc = {
            "created_at": time.time(),
            "rows": len(rows),
            "total_usd": round(total, 6),
            "origin": str(books),
        }
        doc["sha256"] = hashlib.sha256(_canonical({k: doc[k] for k in ("created_at", "rows", "total_usd", "origin")}).encode("utf-8")).hexdigest()
        path = carryover_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        self._insert_carryover(doc)

    def _insert_carryover(self, doc: dict) -> None:
        """@brief 当表里还没有结转行时插入一条。

        @param doc 结转文档，需含 ``created_at`` / ``rows`` / ``total_usd`` / ``origin`` / ``sha256``。
        """
        if self.conn.execute("SELECT 1 FROM ledger_carryover LIMIT 1").fetchone():
            return
        self.conn.execute(
            "INSERT INTO ledger_carryover (created_at, rows, total_usd, origin, sha256) VALUES (?, ?, ?, ?, ?)",
            (doc["created_at"], doc["rows"], doc["total_usd"], doc["origin"], doc["sha256"]),
        )
        self.conn.commit()

    def read_only(self) -> sqlite3.Connection:
        """@brief 新开一个只读连接供读方使用；调用方负责关闭。

        @return 只读的 sqlite3 连接（行工程为 ``Row``）。
        """
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def revision(self) -> int:
        """@brief 取详情 revision。

        @return meta 里的 ``revision``，缺失时为 0。
        """
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()
        return int(row[0] if row else 0)

    def catalog_revision(self) -> int:
        """@brief 取目录史（``kernel_history``）代际；只有投影内容真的变了才抬高。

        目录史不是 ``runs`` 的一行，所以不能靠 ``revision`` 表达“目录史变了”。发布史面板把它
        放进戳，任何目录史追加/就地改写都会让旧戳失效。

        @return meta 里的 ``catalog_revision``，缺失时为 0。
        """
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'catalog_revision'").fetchone()
        return int(row[0] if row else 0)

    def _bump_catalog(self) -> None:
        """@brief 目录史投影内容发生变化时抬高代际。
        """
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('catalog_revision', '1')"
            " ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
        )

    def _bump(self) -> None:
        """@brief 把详情 revision 加一。
        """
        self.conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'"
        )

    def _mark_changed(self, run_id: str) -> None:
        """@brief 记录“这场运行变了”：``changed`` 与 revision bump 必须成对。

        精准刷新靠这一对：SSE 只推变化 run id，客户端按 id 只刷那一场。拆开写会让某个写点
        只改内容不报变化（前端停在旧值），或只报变化不抬 revision（帧被丢掉）。所以这里是
        changed 列表追加的唯一调用点，守卫测试按源码扫描锁住这条。
        原始英文说明：Mark a run changed and bump the revision together.

        @param run_id 运行 id。
        """
        self.changed.append(run_id)
        self._bump()

    def turns_revision(self) -> int:
        """@brief 取轮次 revision。

        @return meta 里的 ``turns_revision``，缺失时为 0。
        """
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'turns_revision'").fetchone()
        return int(row[0] if row else 0)

    def _bump_turns_epoch(self, run_id: str) -> None:
        """@brief 抬高该 run 的轮次投影 epoch；组号重投影不改 ``turn_bytes`` 时靠它让 stamp 失效。

        @param run_id 运行 id。
        """
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, '1')"
            " ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
            (f"turns_epoch:{run_id}",),
        )

    def _bump_turns(self) -> None:
        """@brief 新增了一行转录时抬高轮次 revision，详情 revision 不动。

        原始英文说明：A new transcript row. The detail revision stays where it is.
        """
        self.conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'turns_revision'"
        )

    def feedback_revision(self) -> int:
        """@brief 取反馈 revision。

        @return meta 里的 ``feedback_revision``，缺失时为 0。
        """
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'feedback_revision'").fetchone()
        return int(row[0]) if row else 0

    def _full_interval(self) -> float:
        """@brief 取全量重扫间隔，来自 ``OBSERVE_FULL_TICK_INTERVAL``。

        @return 间隔秒数，不小于 0；环境变量非法时返回 10。
        """
        try:
            return max(0.0, float(os.environ.get("OBSERVE_FULL_TICK_INTERVAL", "10")))
        except ValueError:
            return 10.0

    def tick(self) -> tuple[int, list[str]]:
        """@brief 推进一帧增量：根目录 mtime 发现新运行，每 10 秒做一次全量扫描。

        原始英文说明：One incremental frame. Root mtime detects new runs; a full watched scan runs every 10s.

        @return 二元组：当前详情 revision 与本帧有变化的 run id 列表。
        """
        with _WRITE:
            self._prepare_frame()
            root = detail.runs_root()
            if root.is_dir():
                self._note_dir(root, force=False)
            # 写者日志每帧消费：已结束 run 的就地改写不必等到 10 秒对账周期。
            _healthy, forced = self._consume_journal()
            now = time.monotonic()
            if forced or now - self._last_full >= self._full_interval():
                self._refresh_watched()
                self._sync_kernels()
                self._last_full = now
            self._refresh_running()
            self.conn.commit()
            self._frame_ingested.clear()
        # 机全局 GPU 事件日志独立于 runs/：每帧追平新字节，供时间线按区间查询。
        self.ensure_gpu_events()
        return self.revision(), list(self.changed)

    def _index_workers(self) -> int:
        """@brief 冷建并行度：``OBSERVE_INDEX_WORKERS`` 优先，默认 ``min(8, cpu)``。

        @return 至少 1 的工作进程数。
        """
        raw = os.environ.get("OBSERVE_INDEX_WORKERS")
        if raw not in (None, ""):
            try:
                return max(1, int(raw))
            except ValueError:
                return 1
        return min(8, os.cpu_count() or 1)

    def _collect_tree(self, root: Path) -> list[tuple[Path, int, bool]]:
        """@brief 只列目录、不摄取：收集所有 run/实验目录并记下 ``watched``。

        并行冷建先要知道有哪些目录；这份遍历不读任何 JSON，只 ``scandir`` 与 ``stat``。

        @param root ``runs/`` 根。
        @return ``(目录, mtime_ns, 是否 run/实验)`` 的列表；顺序按遍历先后。
        """
        found: list[tuple[Path, int, bool]] = []
        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                mtime = directory.stat().st_mtime_ns
            except OSError:
                continue
            is_run = (
                (directory / "state.json").exists()
                or (directory / "result.json").exists()
                or (directory / "experiment.json").exists()
            )
            found.append((directory, mtime, is_run))
            try:
                with os.scandir(directory) as entries:
                    children = [(entry.name, entry.is_dir()) for entry in entries]
            except OSError:
                continue
            for name, is_dir in children:
                if not is_dir or name in _SKIP_DIRS or name.startswith("."):
                    continue
                child = Path(directory, name)
                if not self._within_runs(child):
                    continue
                stack.append(child)
        found.sort(key=lambda item: self._rel(item[0]))
        return found

    def _parallel_payloads(self, directories: list[Path], workers: int) -> list[dict | None]:
        """@brief 用进程池并行读目录；池不可用或抛错时整批退回串行。

        @param directories 待读目录。
        @param workers 工作进程数。
        @return 与 ``directories`` 同长的 payload 列表（失败项为 None，交给串行路径）。
        """
        if len(directories) < 64:
            return [None] * len(directories)
        from concurrent.futures import ProcessPoolExecutor

        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(_read_run_payload, [str(item) for item in directories], chunksize=16))
        except Exception:  # noqa: BLE001 - 进程池不可用时必须退回串行，冷建不能因此失败
            return [None] * len(directories)

    def sync_known(self, *, force: bool = False) -> None:
        """@brief 发现已知运行的改动与新子目录；空闲帧用 :meth:`tick`，不调它。

        有写者日志时只消费日志点名的 run（O(changed)）；日志缺失、轮转或调用方要求
        ``force`` 时退回逐 run 指纹对账（今天的行为，也是删掉日志后的兜底）。

        原始英文说明：See edits and new child directories. The idle frame uses :meth:`tick` and does not call this.

        @param force 为 True 时即使日志健康也逐 run 对账（手工全量 refresh 用）。
        """
        with _WRITE:
            self._prepare_frame()
            root, root_text = self._root_path()
            watched = list(self.conn.execute("SELECT path, mtime_ns FROM watched"))
            if root.is_dir() and not watched:
                workers = self._index_workers()
                tree = self._collect_tree(root) if workers > 1 else []
                runs = [directory for directory, _mtime, is_run in tree if is_run]
                if workers > 1 and len(runs) >= 64:
                    # 冷建：先只读列出目录，再让工作进程并行做“目录 → 纯读结果”，主进程串行写库。
                    for directory, mtime, is_run in tree:
                        self.conn.execute(
                            "INSERT INTO watched (path, mtime_ns) VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET mtime_ns = excluded.mtime_ns",
                            (self._rel(directory), mtime),
                        )
                        if not is_run and directory != root and self._has_table("run_parents"):
                            try:
                                self.conn.execute("INSERT OR IGNORE INTO run_parents (path) VALUES (?)", (self._rel(directory),))
                            except ValueError:
                                pass
                    payloads = self._parallel_payloads(runs, workers)
                    for directory, payload in zip(runs, payloads):
                        self._ingest(directory, once=False, payload=payload)
                else:
                    self._note_dir(root, force=True)
            healthy, forced = self._consume_journal()
            if force or not healthy or forced:
                # 没有写者日志（或日志轮转/被删）：退回逐目录 mtime 与逐 run 五元指纹对账。
                for item in watched:
                    path = root if item["path"] in ("", ".") else Path(root_text, item["path"])
                    try:
                        mtime = path.stat().st_mtime_ns
                    except OSError:
                        continue
                    if mtime != item["mtime_ns"]:
                        self._note_dir(path, force=True)
                self._reconcile_rows()
            else:
                # 健康日志：只 stat 根与“可能装新 run 的父目录”做新目录发现，不再逐 watched stat。
                # 新 run 由写者日志点名；带外新建的嵌套目录由这里的父目录 mtime 发现。
                self._discover_new_dirs(root, watched)
            self._last_full = time.monotonic()
            self.conn.commit()
            self._frame_ingested.clear()

    def _discover_new_dirs(self, root: Path, watched: list) -> None:
        """@brief 健康日志帧的新目录发现：只 stat 根与可能装新 run 的父目录。

        父目录集合来自 ``run_parents``（每个 run 的直接父目录）与其祖先，真实语料里是 target 目录与
        实验 arm 目录，数量很小。mtime 变了才进去扫一层，不会逐 watched 目录 stat。

        @param root ``runs/`` 根。
        @param watched 已记录的 ``(path, mtime_ns)`` 行。
        """
        known = {str(item["path"]): int(item["mtime_ns"]) for item in watched}
        candidates = {""}
        if self._has_table("run_parents"):
            candidates.update(str(row["path"]) for row in self.conn.execute("SELECT path FROM run_parents"))
        expanded = {""}
        for item in candidates:
            if item in ("", ".", "./"):
                continue
            relative = Path(item)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            expanded.add(item)
            expanded.update(str(parent) for parent in relative.parents if str(parent) != ".")
        for rel in sorted(expanded, key=lambda item: (item.count("/"), item)):
            path = root if rel in ("", ".") else root / rel
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                continue
            if known.get(rel, known.get(".", -1) if rel in ("", ".") else None) != mtime:
                self._note_dir(path, force=False)

    def _consume_journal(self) -> tuple[bool, bool]:
        """@brief 消费写者日志的新增行，只摄取它点名的 run。

        分片按 ``(name, offset, inode)`` 记在 ``meta``；inode 变化或 ``size < offset`` 视为轮转/截断，
        从 0 重读并报告 ``forced``，让调用方补一次全量对账。

        @return ``(日志是否可用, 是否要求全量对账)``。
        """
        root, _ = self._root_path()
        shards = index_journal.shards(root)
        if not shards:
            return False, False
        forced = False
        names: list[str] = []
        experiments: list[str] = []
        for shard in shards:
            key = shard.name
            try:
                offset = int(self._meta_get(f"journal_offset:{key}") or 0)
                inode = int(self._meta_get(f"journal_inode:{key}") or 0)
            except ValueError:
                offset, inode = 0, 0
            entries, new_offset, new_inode, reset = index_journal.read_new(shard, offset, inode)
            if reset:
                forced = True
            for item in entries:
                name = item.get("run")
                if not isinstance(name, str) or not name:
                    continue
                if name not in names:
                    names.append(name)
                if item.get("path") == "experiment.json" and name not in experiments:
                    experiments.append(name)
            self._meta_set(f"journal_offset:{key}", str(new_offset))
            self._meta_set(f"journal_inode:{key}", str(new_inode))
        for name in names:
            path = self._journal_run_path(name)
            if path is not None and not self._already_ingested(path):
                self._ingest(path)
        # ``experiment.json`` 的 run 值是实验根；改 arm 归属只改清单指纹，子 run 的目录 mtime 不动，
        # 所以必须把该实验下已索引的 run 一起复核（只 stat 它们，指纹没变就短路）。
        for name in experiments:
            self._reingest_children(name)
        return True, forced

    def _reingest_children(self, name: str) -> None:
        """@brief 重新摄取某个实验根下所有已索引的 run。

        @param name 实验根相对 ``runs/`` 的路径。
        """
        prefix = name.rstrip("/") + "/"
        pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        rows = self.conn.execute(
            "SELECT run_id FROM runs WHERE run_id LIKE ? ESCAPE '\\'", (pattern,)
        ).fetchall()
        root, _ = self._root_path()
        for row in rows:
            path = detail.resolve_run(row["run_id"], root=root)
            if path is not None:
                # arm 归属变了：同一帧早先摄取过的子 run 也要按新清单重算 binding。
                self._ingest(path, once=False)
            else:
                self._mark_missing(row["run_id"])

    def _journal_run_path(self, name: str) -> Path | None:
        """@brief 把日志里的 run 相对路径变成一个确实存在的目录；越界或不存在时返回 None。

        @param name 日志行里的 ``run`` 值。
        @return 运行目录或 None。
        """
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            return None
        root, _ = self._root_path()
        path = Path(root, relative)
        try:
            if not path.is_dir():
                return None
        except OSError:
            return None
        return path

    def _reconcile_path(self, run_id: str) -> Path | None:
        """@brief 对账用的便宜 run 路径：不逐级 ``resolve``，但仍跟随符号链接判边界。

        ``resolve_run`` 每次要逐级 ``realpath``（约 10 次 lstat）；热对账每个 run 都做一次就成了
        主要成本。run_id 来自索引（写入时已是 canonical 相对路径），这里只做越界检查、``is_dir``
        与快照存在性检查，路径归属交给缓存的 ``_within_runs``。

        @param run_id 相对 ``runs/`` 的运行 id。
        @return 运行目录；越界、不存在、无快照或解析到根外时返回 None。
        """
        if not run_id or "\x00" in run_id:
            return None
        relative = Path(run_id)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        root, _ = self._root_path()
        path = Path(root, relative)
        try:
            if not path.is_dir():
                return None
            if not (path / "state.json").exists() and not (path / "result.json").exists():
                return None
        except OSError:
            return None
        if not self._within_runs(path):
            return None
        return path

    def _reconcile_rows(self) -> None:
        """@brief 逐 run 的兜底对账：目录消失就标 missing，还在就 ``_ingest`` 一次。

        ``_ingest`` 先查已存指纹，没变的 run 一个字节都不读。
        """
        for row in self.conn.execute("SELECT run_id FROM runs").fetchall():
            path = self._reconcile_path(row["run_id"])
            if path is not None:
                if not self._already_ingested(path):
                    self._ingest(path)
                continue
            # 目录已经不在了。留着一行 status='done' 会让它的花费和公布成绩永远被
            # runs_page / best_confirmed / 预算统计算进去；migrate 之后还会出现同名两行。
            self._mark_missing(row["run_id"])

    def _prepare_frame(self) -> None:
        """@brief 重置本帧统计与变更列表；``runs_root`` 变了就清掉全部派生表。
        """
        self.stats = {"json_reads": 0, "dir_lists": 0, "turn_parses": 0, "root_resolves": 0}
        self.changed = []
        self._frame_ingested = set()
        self._resolved_cache = {}
        self._linkfree_cache = {}
        root = self._root_path()[1]
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'runs_root'").fetchone()
        if row is not None and row[0] != root:
            # schema-11 结构没有 confirmed_best/run_parents；只清实际存在的表，避免 no such table
            # 让多步清理半途中断。
            tables = (
                "runs", "candidates", "confirmed_best", "run_parents", "run_pending_eval", "failure_members",
                "run_rollup", "kernel_history", "gpu_events", "gpu_waits", "lock_spans", "lock_spans_long", "lock_span_slots",
                "sessions", "experiments",
                "arms", "ledger", "watched", "feedback_items", "feedback_claims", "feedback_subclaims",
            )
            for table in tables:
                if self._has_table(table):
                    self.conn.execute(f"DELETE FROM {table}")
            self.conn.execute("DELETE FROM meta WHERE key LIKE 'sessions_fp:%'")
            self.conn.execute("DELETE FROM meta WHERE key LIKE 'sessions_bytes:%'")
            self.conn.execute("DELETE FROM meta WHERE key LIKE 'transcript_bytes:%'")
            self.conn.execute("DELETE FROM meta WHERE key LIKE 'turns_epoch:%'")
            self.conn.execute("DELETE FROM meta WHERE key LIKE 'reconcile_fp:%'")
            self.conn.execute("DELETE FROM meta WHERE key LIKE 'kw_src:%'")
            self.conn.commit()
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('runs_root', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (root,),
        )

    def _watched(self, rel: str) -> int | None:
        """@brief 取某个相对路径已记录的 mtime。

        @param rel 相对 ``runs/`` 的路径。
        @return mtime 纳秒数；没有记录时返回 None。
        """
        row = self.conn.execute("SELECT mtime_ns FROM watched WHERE path = ?", (rel,)).fetchone()
        return None if row is None else int(row[0])

    def _note_dir(self, directory: Path, *, force: bool) -> None:
        """@brief 记录目录 mtime，必要时摄取该目录，并递归处理子目录。

        @param directory 待检查目录。
        @param force 为 True 时忽略已记录的 mtime 强制处理。
        """
        try:
            mtime = directory.stat().st_mtime_ns
        except OSError:
            return
        rel = self._rel(directory)
        if not force and self._watched(rel) == mtime:
            return
        if (directory / "state.json").exists() or (directory / "result.json").exists() or (directory / "experiment.json").exists():
            if not self._already_ingested(directory):
                self._ingest(directory)
        self.stats["dir_lists"] += 1
        self.conn.execute(
            "INSERT INTO watched (path, mtime_ns) VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET mtime_ns = excluded.mtime_ns",
            (rel, mtime),
        )
        # 只数“目录里有哪些孩子”，不把子项做成 Path 再逐个 resolve；entry.is_dir() 等价于 follow 的
        # Path.is_dir()，符号链接仍从这里进入 ``_within_runs`` 的越界检查。
        try:
            with os.scandir(directory) as entries:
                children = [(entry.name, entry.is_dir()) for entry in entries]
        except OSError:
            return
        for name, is_dir in children:
            if not is_dir or name in _SKIP_DIRS or name.startswith("."):
                continue
            child = Path(directory, name)
            if not self._within_runs(child):
                continue
            self._consider(child)

    def _consider(self, directory: Path) -> None:
        """@brief 子目录看起来是运行或实验时摄取它，然后继续往下看。

        是否摄取交给 :meth:`_note_dir` 的目录 mtime 判断：父目录 mtime 变了不代表每个孩子都变了。
        写完一个文件会改它所在目录的 mtime，所以“孩子自己的 mtime 没变”就说明这一帧不必重读它；
        就地改写由写者日志与周期对账兜住。这样 ``.index_journal/`` 刚创建引起的根 mtime 变化
        不会让热帧重摄取一万个 run。

        @param directory 子目录路径。
        """
        is_run = (directory / "state.json").exists() or (directory / "result.json").exists() or (directory / "experiment.json").exists()
        if not is_run and self._has_table("run_parents"):
            # 预先存在的空容器（例如 runs/conv2d/exp/）：先记成 sentinel，之后在里面建 run
            # 只会改容器 mtime，发现路径必须 stat 它。
            try:
                self.conn.execute("INSERT OR IGNORE INTO run_parents (path) VALUES (?)", (self._rel(directory),))
            except ValueError:
                pass
        self._note_dir(directory, force=False)

    def _refresh_watched(self) -> None:
        """@brief 发现新子目录或续跑：建嵌套目录只会改父目录的 mtime，不会改 ``runs/``。

        目录 mtime 不变但内容变的情况（候选报告/outbox 就地改写）由 ``_reconcile_fingerprints``
        按完整指纹逐个 run 复核；它只在全量 tick 周期跑，且只 stat/列目录，不读文件内容。
        原始英文说明：Notice a new child or a resumed run. Creating a nested directory changes its parent, not ``runs/``.
        """
        root = detail.runs_root()
        watched = list(self.conn.execute("SELECT path, mtime_ns FROM watched"))
        for item in watched:
            path = root if item["path"] in ("", ".") else root / item["path"]
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                continue
            if mtime != item["mtime_ns"]:
                self._note_dir(path, force=True)
        self._reconcile_fingerprints()

    def _reconcile_fingerprints(self) -> None:
        """@brief 全量 tick 周期逐 run 复核：只比“便宜键”，命中差异才重新摄取。

        便宜键覆盖候选报告/内核与 outbox 的逐文件 stat。没变的 run 不读任何文件内容；完整指纹
        的其余部分（state/result/config/确认文件）仍由 ``sync_known``、``/best`` 与运行中刷新覆盖。
        """
        rows = self.conn.execute(
            "SELECT run_id, fp_state, fp_result, fp_config, fp_confirm, fp_extra "
            "FROM runs WHERE status NOT IN ('running', 'missing')"
        ).fetchall()
        for row in rows:
            run_id = row["run_id"]
            path = self._reconcile_path(run_id)
            if path is None:
                self._mark_missing(run_id)
                continue
            stored = (row["fp_state"], row["fp_result"], row["fp_config"], row["fp_confirm"], row["fp_extra"])
            # 逐 run 比五元指纹（而不是只看目录 mtime/候选文件）：confirm_report.json、
            # state.json、result.json 的就地改写也必须在这条兜底路径上被发现。
            label, rel = _confirm_target(stored[3])
            if self._cheap_fingerprints(path, label, rel) == stored:
                # 指纹一致也要增量复核 kernel_writer 的兜底来源：摘要戳含这个摄入窄投影，
                # 只调 tick 时旧 run 从 chat 翻成 pi 也必须让摘要换戳（与 eval_progress 同一类要求）。
                self._refresh_kernel_writer(path)
                continue
            self._ingest(path)

    def _refresh_running(self) -> None:
        """@brief 逐行复查仍在跑的运行：指纹没变只补轮次与存活状态，变了就重新摄取。
        """
        for row in self.conn.execute("SELECT run_id, updated_at, pid, phase, status, row_json FROM runs WHERE status = 'running'").fetchall():
            path = detail.resolve_run(row["run_id"], root=self._root_path()[0])
            if path is None:
                self._mark_missing(row["run_id"])
                continue
            stored = self._stored_fingerprints(path)
            label, rel = _confirm_target(stored[3])
            if self._fingerprints(path, label, rel) == stored:
                # transcript.jsonl is not part of the detail fingerprint. A new line still has to land in the turns table,
                # and that must not redraw the candidate list.
                if self._catch_up_turns(path):
                    self._bump_turns()
                    self._bump_turns_epoch(row["run_id"])
                self._refresh_kernel_writer(path)
                state = {"updated_at": row["updated_at"], "pid": row["pid"], "phase": row["phase"]}
                status = detail.liveness(state, None)
                if status != row["status"]:
                    self._write_liveness(row, status)
                    self._mark_changed(row["run_id"])
                continue
            self._ingest(path)

    def _stored_fingerprints(self, directory: Path) -> tuple[str, str, str, str, str]:
        """@brief 取索引里已存的一个运行的五个指纹。

        @param directory 运行目录。
        @return 五元组；索引里没有该行时全为空串。
        """
        row = self.conn.execute(
            "SELECT fp_state, fp_result, fp_config, fp_confirm, fp_extra FROM runs WHERE run_id = ?",
            (self._rel(directory),),
        ).fetchone()
        if row is None:
            return ("", "", "", "", "")
        return tuple(row)

    @staticmethod
    def _winner_matches(directory: Path, winner_label: str) -> str:
        """@brief 与胜者标签同名的候选目录名（只要名字，不读文件）。

        ``conclusion_of_run`` 在 winner 不唯一时不发布；新增一颗同名候选目录可能不改任何
        已存指纹，所以必须把这份“名字集合”纳入定向确认核对。

        @param directory 运行目录。
        @param winner_label 胜者标签。
        @return 排序后的目录名，以 ``|`` 分隔；没有标签时为空串。
        """
        if not winner_label:
            return ""
        root = directory / "candidates"
        try:
            names = sorted(
                item.name for item in root.iterdir()
                if item.is_dir() and item.name.endswith("_" + winner_label)
            )
        except OSError:
            return ""
        return "|".join(names)

    @staticmethod
    def _confirmation_fingerprint(directory: Path, winner_label: str = "", winner_rel: str = "") -> str:
        """@brief 现算确认部分指纹：只 stat 确认报告、best 内核与胜者内核。

        ``best`` 的指纹来自顶层所有 ``best.*`` 的名字与 stat（``_best_fingerprints``），不解析
        ``run_config.json``：这个函数每次 ``/best`` 都会对已确认运行跑一遍，冷缓存时也不能打开配置。

        原始英文说明：The confirmation part of a run's fingerprint, from stats only.

        @param directory 运行目录。
        @param winner_label 获胜候选标签，可为空串。
        @param winner_rel 获胜内核的相对路径，可为空串。
        @return 以 ``\x1f`` 分隔的确认指纹。
        """
        return "\x1f".join((
            _fingerprint(directory / "confirm_report.json"),
            _best_fingerprints(directory),
            winner_label,
            winner_rel,
            _fingerprint(directory / winner_rel) if winner_rel else "",
            Index._winner_matches(directory, winner_label),
        ))

    def confirmation_fingerprints(self) -> list[str]:
        """@brief 已确认运行的确认指纹差异集；只 stat，不打开文件，也不持全局写锁。

        指纹没变的行一个字节都不用读。``/best`` 只要拿到这个集合，再对集合里的行取写锁做一次
        ``_ingest``，就既不会长期展示旧的确认数字，也不会每请求重摄取全部已确认运行。
        原始英文说明：Which confirmed runs changed, judged by stats only and outside the write lock.

        @return 需要重新摄取的 run id 列表；目录消失的已确认行也在其中。
        """
        # 只读连接：这段是纯读，拿全局写锁会让 /best 与在飞的 _ingest 互相排队。
        rows = [
            dict(row) for row in reader().execute(
                "SELECT run_id, fp_state, fp_result, fp_config, fp_confirm "
                "FROM runs WHERE publication = 'confirmed'"
            )
        ]
        root = detail.runs_root()
        stale = []
        for row in rows:
            run_id = row["run_id"] or ""
            # 存下来的 run_id 是相对 runs/ 的路径；这里不走 resolve_run，避免每个已确认运行多做
            # 一次 resolve + is_dir + 两次 exists。越界值一律当作过期，交给 _ingest 去拒绝。
            relative = Path(run_id)
            if not run_id or relative.is_absolute() or ".." in relative.parts:
                stale.append(row["run_id"])
                continue
            directory = root / relative
            winner_label, winner_rel = _confirm_target(row["fp_confirm"])
            confirm = self._confirmation_fingerprint(directory, winner_label, winner_rel)
            # /best 返回的 started_at/size/target_id/model 来自 state 与 run_config，它们变了也必须重摄取。
            changed = (
                confirm != (row["fp_confirm"] or "")
                or _fingerprint(directory / "result.json") != (row["fp_result"] or "")
                or self._state_fingerprint(directory) != (row["fp_state"] or "")
                or _fingerprint(directory / "run_config.json") != (row["fp_config"] or "")
            )
            if changed:
                stale.append(row["run_id"])
        return stale

    @staticmethod
    def _state_fingerprint(directory: Path) -> str:
        """@brief 取运行快照的指纹：有 ``state.json`` 用文件，没有 result-only run 用目录。

        没有 state.json 时 ``detail._when`` 用目录 mtime 作 ``started_at``；摄取与定向确认核对
        必须用同一条规则，否则已存目录指纹永远不等于现算的空文件指纹，造成每次 ``/best`` 都重摄取。

        @param directory 运行目录。
        @return 指纹字符串。
        """
        try:
            stat = (directory / "state.json").stat()
        except OSError:
            return _fingerprint(directory)
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def _cheap_fingerprints(self, directory: Path, winner_label: str = "", winner_rel: str = "") -> tuple[str, str, str, str, str]:
        """@brief 只 stat/列目录的快路径指纹：不打开任何文件，供 ``_ingest`` 开头判是否需要重读。

        这是“按白名单 stat”的那一步：state/result/config/确认文件、``best.*``、胜者内核与
        候选报告/内核的 ``(mtime_ns,size)``，以及上行 ``experiment.json``。它不读 JSON 内容，
        也不读 transcript；``_ingest`` 拿到相同五元组就直接短路。

        @param directory 运行目录。
        @param winner_label 获胜候选标签，可为空串。
        @param winner_rel 获胜内核的相对路径，可为空串。
        @return 五元组指纹。
        """
        return _fingerprints_at(directory, winner_label, winner_rel)

    def _fingerprints(self, directory: Path, winner_label: str = "", winner_rel: str = "") -> tuple[str, str, str, str, str]:
        """@brief 现算一个运行的五个指纹（state、result、config、确认、其余杂项）。

        @param directory 运行目录。
        @param winner_label 获胜候选标签，可为空串。
        @param winner_rel 获胜内核的相对路径，可为空串。
        @return 五元组指纹，用于判断是否需要重新摄取。
        """
        return self._cheap_fingerprints(directory, winner_label, winner_rel)

    def _read_json(self, path: Path) -> dict | None:
        """@brief 读一个 JSON 对象并计入本帧读盘次数；文件不存在时返回 None。

        @param path 文件路径。
        @return 解析后的字典，或 None。
        """
        if not path.exists():
            return None
        self.stats["json_reads"] += 1
        return detail._load(path)

    def _catch_up_turns(self, directory: Path) -> bool:
        """@brief 把 ``transcript.jsonl`` 新增的完整行投影进 turns 表。

        字节偏移存在 ``runs.turn_bytes``；文件变短则从头重投影；尾部半行留到下一帧。返回偏移或表是否
        变化，不抬高详情 revision。
        原始英文说明：Project new complete lines of ``transcript.jsonl`` into the turns table.

        @param directory 运行目录。
        @return 有进展时为 True。
        """
        run_id = self._rel(directory)
        row = self.conn.execute("SELECT turn_bytes FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return False
        turn_bytes = int(row["turn_bytes"] or 0)
        path = directory / "transcript.jsonl"
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            size = 0
        changed = False
        if size < turn_bytes:
            self.conn.execute("DELETE FROM turns WHERE run_id = ?", (run_id,))
            turn_bytes = 0
            changed = True
        previous = self.conn.execute(
            "SELECT label, grp FROM turns WHERE run_id = ? ORDER BY seq DESC LIMIT 1", (run_id,)
        ).fetchone()
        if previous is not None and int(previous["grp"] or 0) == 0:
            # schema-12 刚落地时已投影的行没有 grp。分组边界不可恢复，整份重投影一次：
            # 这是派生表，重投影只花这一场 transcript 的读入；偏移必须先归零，否则
            # “已读完整份”的库会一直跳过这次重建。
            self.conn.execute("DELETE FROM turns WHERE run_id = ?", (run_id,))
            self.conn.execute("UPDATE runs SET turn_bytes = ? WHERE run_id = ?", (0, run_id))
            turn_bytes = 0
            previous = None
            changed = True
        if size == turn_bytes:
            if changed:
                self.conn.execute("UPDATE runs SET turn_bytes = ? WHERE run_id = ?", (0, run_id))
            return changed
        self.stats["turn_seek"] = turn_bytes
        try:
            with path.open("rb") as handle:
                handle.seek(turn_bytes)
                data = handle.read()
        except OSError:
            return changed
        last_nl = data.rfind(b"\n")
        if last_nl < 0:
            return changed
        complete = data[: last_nl + 1]
        new_offset = turn_bytes + last_nl + 1
        seq_row = self.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM turns WHERE run_id = ?", (run_id,)).fetchone()
        seq = int(seq_row[0] or 0)
        prev_label = str(previous["label"] or "") if previous is not None else None
        grp = int(previous["grp"] or 0) if previous is not None else 0
        for raw in complete.splitlines():
            self.stats["turn_parses"] = self.stats.get("turn_parses", 0) + 1
            try:
                item = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(item, dict):
                continue
            projected = detail.project_turn(item)
            if projected is None:
                continue
            seq += 1
            label = str(projected.get("label") or "")
            # 相邻同标签共组；同名标签再次出现、空标签、以及首行都另起一组。
            if prev_label is None or label == "" or label != prev_label:
                grp += 1
            prev_label = label
            self.conn.execute(
                "INSERT INTO turns (run_id, seq, label, kind, grp, row_json) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, seq, label, projected.get("kind") or "", grp,
                 json.dumps(projected, ensure_ascii=False)),
            )
        self.conn.execute("UPDATE runs SET turn_bytes = ? WHERE run_id = ?", (new_offset, run_id))
        return True

    def _finished_without_turns(self, directory: Path) -> bool:
        """@brief 判断该运行是否已结束且从未投影过轮次；打开轮次列表时才会建行。

        原始英文说明：A finished run that has never been projected. Opening its turn list builds the rows.

        @param directory 运行目录。
        @return 已结束且 ``turn_bytes`` 为 0 时为 True。
        """
        row = self.conn.execute(
            "SELECT status, turn_bytes FROM runs WHERE run_id = ?",
            (self._rel(directory),),
        ).fetchone()
        if row is None:
            return False
        return row["status"] != "running" and int(row["turn_bytes"] or 0) == 0

    def _project_turns(self, directory: Path) -> None:
        """@brief 为仍在跑、或已有投影可续的运行补齐轮次。

        原始英文说明：Catch up a run that is still going, or one that already has a projection to extend.

        @param directory 运行目录。
        """
        if self._finished_without_turns(directory):
            return
        if self._catch_up_turns(directory):
            self._bump_turns()
            self._bump_turns_epoch(self._rel(directory))

    def ensure_turns(self, directory: Path) -> None:
        """@brief 按需投影该运行；没有新字节时第二次调用不会再解析文件。

        原始英文说明：Project this run on demand. A second call with no new bytes does not parse the file again.

        @param directory 运行目录。
        """
        with _WRITE:
            if self._catch_up_turns(directory):
                self._bump_turns()
                self._bump_turns_epoch(self._rel(directory))
            self.conn.commit()

    def _catch_up_sessions(self, directory: Path, config: dict | None) -> None:
        """@brief 只投影 transcript 的新增完整行；首次或文件变短时才全量重建。"""
        run_id = self._rel(directory)
        path = directory / "transcript.jsonl"
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            size = 0
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (f"sessions_bytes:{run_id}",)).fetchone()
        offset = int(row[0]) if row is not None and str(row[0]).isdigit() else -1
        if offset < 0 or offset > size:
            self._upsert_sessions(directory, run_id, config)
            return
        if offset == size:
            # ensure_sessions 只在源指纹变过时进来：等长改写不是“没有新增”，必须按重建处理。
            self._upsert_sessions(directory, run_id, config)
            return
        # §11.4：flame_chase_1 也设同一个 >1MB 上限（旧版只对非 flame 生效，会全读大 transcript）。
        if size > 1_000_000:
            self.conn.execute("DELETE FROM sessions WHERE run_id = ?", (run_id,))
            self._record_sessions_projection(directory, size)
            return
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read()
        except OSError:
            return
        last_nl = data.rfind(b"\n")
        if last_nl < 0:
            self._record_sessions_projection(directory, offset)
            return
        complete = data[: last_nl + 1]
        new_offset = offset + last_nl + 1
        seq_row = self.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM sessions WHERE run_id = ?", (run_id,)).fetchone()
        seq = int(seq_row[0] or 0)
        for raw in complete.splitlines():
            if b'"agent"' not in raw:
                continue
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict) or event.get("event") != "agent":
                continue
            seq += 1
            self._insert_session_event(run_id, seq, event)
        self._record_sessions_projection(directory, new_offset)

    def ensure_sessions(self, directory: Path) -> str:
        """@brief 按 transcript 的 ``(mtime_ns,size)`` 增量追平 sessions，并返回该源指纹。

        指纹没变时直接返回，不读文件、不拿锁做 JSON 解析。首次/变短走全量重建；追加时只读新增字节。
        """
        run_id = self._rel(directory)
        path = directory / "transcript.jsonl"
        try:
            stat = path.stat() if path.is_file() else None
        except OSError:
            stat = None
        fingerprint = f"{stat.st_mtime_ns}:{stat.st_size}" if stat is not None else "0:0"
        key = f"sessions_fp:{run_id}"
        with _WRITE:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            if row is not None and row[0] == fingerprint:
                return fingerprint
            config = self._read_json(directory / "run_config.json")
            self._catch_up_sessions(directory, config)
            # 已经为会话读过 transcript：顺手把 kernel_writer 的来源缓存补上，
            # 让巡检/摘要不必为同一份文件再扫一遍。
            self._refresh_kernel_writer(directory)
            self.conn.commit()
        return fingerprint

    def mark_sessions_absent(self, run_id: str) -> None:
        """@brief 记下一个 run 没有可投影的 transcript，让懒投影不再为它占批次。

        ``sessions_fp`` 写 ``ensure_sessions`` 对空目录用的同一枚指纹 ``0:0``，``transcript_bytes``
        写 ``0``。源文件以后出现时，重摄取的 ``_drop_stale_sessions`` 会因指纹不同删掉这两行，
        下一次懒投影重新做。这里只碰 meta，不读 transcript、不动 runs 行。

        @param run_id 运行 id。
        """
        with _WRITE:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (f"sessions_fp:{run_id}", "0:0")
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (f"sessions_bytes:{run_id}", "0")
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (f"transcript_bytes:{run_id}", "0")
            )
            self.conn.commit()

    def _meta_get(self, key: str) -> str:
        """@brief 取一个 meta 值；没有时返回空串。

        @param key meta 键。
        @return 值的文本。
        """
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row is not None else ""

    def _meta_set(self, key: str, value: str) -> None:
        """@brief 写一个 meta 值。

        @param key meta 键。
        @param value 值的文本。
        """
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def _index_span(self, span_id: int, start: float, end: float) -> None:
        """@brief 把一个 span 注册进读侧的有界区间索引。

        正常长度的 span 复制到它覆盖的每个 10 秒桶；超过 1 小时的 span 只进 ``lock_spans_long``
        （这种等待极少，读时扫这张小表即可），避免一个长期等待制造成千上万条桶行。

        @param span_id ``lock_spans.rowid``。
        @param start 区间起点。
        @param end 区间终点。
        """
        if end - start > detail.LOCK_SPAN_LONG_SECONDS:
            self.conn.execute("INSERT OR IGNORE INTO lock_spans_long (span_id) VALUES (?)", (span_id,))
            return
        first = int(start // detail.LOCK_SPAN_BUCKET_SECONDS)
        last = int(end // detail.LOCK_SPAN_BUCKET_SECONDS)
        for bucket in range(first, last + 1):
            self.conn.execute(
                "INSERT OR IGNORE INTO lock_span_slots (bucket, span_id) VALUES (?, ?)",
                (bucket, span_id),
            )

    def _ensure_lock_span_slots(self) -> None:
        """@brief 打开旧库时一次性补建桶索引；之后由 ``_insert_span`` 同步维护。

        桶索引是派生数据：进程在事务中间被杀会随事务回滚，不会与 ``lock_spans`` 脱节。
        """
        if self._meta_get("lock_span_slots_v1") == "1":
            return
        self.conn.execute("DELETE FROM lock_span_slots")
        self.conn.execute("DELETE FROM lock_spans_long")
        for row in self.conn.execute("SELECT rowid, start, end FROM lock_spans").fetchall():
            self._index_span(int(row["rowid"]), float(row["start"]), float(row["end"]))
        self._meta_set("lock_span_slots_v1", "1")

    def _insert_span(self, pid, start, end, closed_at, mode, what) -> None:
        """@brief 写一条已配对区间，并同步读侧的桶索引。

        @param pid 进程号。
        @param start 区间起点。
        @param end 区间终点。
        @param closed_at 关闭该区间的事件时刻（读侧排序键）。
        @param mode 锁模式。
        @param what 事件描述。
        """
        cursor = self.conn.execute(
            "INSERT INTO lock_spans (pid, start, end, closed_at, mode, what) VALUES (?, ?, ?, ?, ?, ?)",
            (pid, start, end, closed_at, mode, what),
        )
        self._index_span(int(cursor.lastrowid), float(start), float(end))

    def _reset_gpu_events(self) -> None:
        """@brief 清空 GPU 事件、未配对 wait 与已物化区间。

        清理派生表不等于删源文件：源日志被截断/替换时，下一次读取会重新扫保留分片。
        """
        self.conn.execute("DELETE FROM gpu_events")
        self.conn.execute("DELETE FROM gpu_waits")
        self.conn.execute("DELETE FROM lock_spans")
        self.conn.execute("DELETE FROM lock_span_slots")
        self.conn.execute("DELETE FROM lock_spans_long")
        self._meta_set("gpu_events_offset", "0")
        self._meta_set("gpu_events_stat", "")
        self._meta_set("gpu_events_tail_sha", "")

    def _gpu_shards(self, path: Path) -> list[Path]:
        """@brief 找出轮转保留的分片文件（``<name>.<后缀>``），按修改时间升序。

        计划里轮转是“改名保留分片，不是删除”；索引在 inode 变化后按分片顺序重放，保证重建等价。

        @param path 当前事件日志。
        @return 分片路径列表。
        """
        try:
            items = [
                item for item in path.parent.iterdir()
                if item != path and item.name.startswith(path.name + ".") and item.is_file()
            ]
        except OSError:
            return []
        return sorted(items, key=lambda item: (item.stat().st_mtime_ns, item.name))

    @staticmethod
    def _gpu_number(value) -> float | None:
        """@brief 只接受有限实数（bool、数字字符串、NaN、Inf 都不算）。

        与旧 ``detail._as_float`` 的判定一致：事件里的 ``t`` 不是有限数时整条丢掉，避免 NaN
        经 SQLite 变成 NULL 后在读侧 ``float(...)`` 炸掉。

        @param value 待校验的值。
        @return 有限浮点数，或 None。
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _gpu_pair(self, pid: int, when: float, kind: str, mode: str, what: str) -> None:
        """@brief 按旧重放规则更新未配对 wait 与已配对区间（不写 gpu_events）。

        新的 ``wait`` 先把旧 wait 以自身时间记成零长区间再覆盖；``acquire``/``give_up`` 把
        pending 与当前时间配成区间；结束时仍未配对的 wait 留到读侧以自身时间记零长。

        @param pid 进程号（缺失时为 0）。
        @param when 事件时间。
        @param kind 事件类型。
        @param mode 锁模式。
        @param what 事件描述。
        """
        if kind == "wait":
            previous = self.conn.execute("SELECT t, mode, what FROM gpu_waits WHERE pid = ?", (pid,)).fetchone()
            if previous is not None:
                self._insert_span(pid, previous["t"], previous["t"], when, previous["mode"], previous["what"])
            # 旧实现最后按 pending 的“首次进入顺序”输出未配对 wait；用单调 seq 记住这个顺序
            # （覆盖 wait 时不改 seq，关闭后重开才拿新号）。
            self.conn.execute(
                "INSERT INTO gpu_waits (pid, t, mode, what, seq) VALUES (?, ?, ?, ?, "
                "(SELECT COALESCE(MAX(seq), 0) + 1 FROM gpu_waits)) "
                "ON CONFLICT(pid) DO UPDATE SET t = excluded.t, mode = excluded.mode, what = excluded.what",
                (pid, when, mode, what),
            )
        elif kind in ("acquire", "give_up"):
            previous = self.conn.execute("SELECT t, mode, what FROM gpu_waits WHERE pid = ?", (pid,)).fetchone()
            if previous is not None:
                self.conn.execute("DELETE FROM gpu_waits WHERE pid = ?", (pid,))
                self._insert_span(pid, previous["t"], when, when, previous["mode"], previous["what"])

    def _gpu_replay(self) -> None:
        """@brief 用已入库事件按全局 ``(t, rowid)`` 顺序重放配对，修复乱序追加。

        旧实现先按 ``t`` 排序再配对；单 pid 重放能修好区间，却修不好“未配对 wait 的进入顺序”
        （旧实现按字典插入顺序输出，等价于 t 序里第一次进入的位置）。所以这里整表重放一次：
        只发生在检测到乱序追加时，正常按时间追加的写者不会走到这里。

        原始英文说明：Replay the whole event index in time order after an out-of-order append.
        """
        self.conn.execute("DELETE FROM lock_spans")
        self.conn.execute("DELETE FROM lock_span_slots")
        self.conn.execute("DELETE FROM lock_spans_long")
        self.conn.execute("DELETE FROM gpu_waits")
        # 流式游标：一条乱序事件不能把整段历史一次读进 Python 内存。
        for row in self.conn.execute("SELECT t, pid, kind, mode, what FROM gpu_events ORDER BY t, rowid"):
            self._gpu_pair(int(row["pid"]), float(row["t"]), str(row["kind"] or ""),
                           str(row["mode"] or ""), str(row["what"] or ""))

    def _gpu_apply(self, raw: bytes, last_seen: list) -> bool:
        """@brief 把一行事件写进派生表，并更新未配对 wait 与已配对区间。

        时间戳必须是有限实数。旧实现先把**全局**事件按 ``t`` 排序再配对；只要新事件的 ``t``
        小于已见的最大 ``t``（不论是否同一 pid），就说明追加顺序与时间顺序不一致，返回 True
        让调用方在这批读完后整表重放。

        @param raw 一行原始字节。
        @param last_seen 单元素列表，保存本批已见的最大 ``t``。
        @return 本事件是否早于已见最大时间（需要重放）。
        """
        try:
            item = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            return None
        if not isinstance(item, dict):
            return None
        when = self._gpu_number(item.get("t"))
        if when is None:
            return None
        raw_pid = self._gpu_number(item.get("pid"))
        pid = int(raw_pid) if raw_pid is not None else 0
        kind = item.get("kind") if isinstance(item.get("kind"), str) else ""
        mode = item.get("mode") if isinstance(item.get("mode"), str) else ""
        what = item.get("what") if isinstance(item.get("what"), str) else ""
        late = last_seen[0] is not None and when < last_seen[0]
        if last_seen[0] is None or when > last_seen[0]:
            last_seen[0] = when
        self.conn.execute(
            "INSERT INTO gpu_events (t, pid, kind, mode, what) VALUES (?, ?, ?, ?, ?)",
            (when, pid, kind, mode, what),
        )
        self._gpu_pair(pid, when, kind, mode, what)
        return late

    def _gpu_stat_state(self, path: Path) -> str:
        """@brief 活动文件的写者元数据：``[ino, mtime_ns, ctime_ns, size]``。

        append 写者每次写都会改 mtime/ctime；三元组不变就说明文件没被写过（含等长改写）。
        这比抽样校验强：任何写入都会让指纹变化。

        @param path 活动事件日志。
        @return JSON 文本；读不到时为空串。
        """
        try:
            stat = path.stat()
        except OSError:
            return ""
        return json.dumps([stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size], separators=(",", ":"))

    @staticmethod
    def _gpu_tail_hash(path: Path, offset: int) -> str:
        """@brief 已消费前缀最后 1KB 的 sha256；追加时用它确认前缀没被换掉。

        @param path 活动事件日志。
        @param offset 已消费到的字节数。
        @return 十六进制摘要；offset<=0 或读失败时为空串。
        """
        if offset <= 0:
            return ""
        try:
            with path.open("rb") as handle:
                handle.seek(max(0, offset - 1024))
                return hashlib.sha256(handle.read(offset - max(0, offset - 1024))).hexdigest()
        except OSError:
            return ""

    def _gpu_prefix_equivalent(self, path: Path, offset: int) -> bool:
        """@brief 已消费前缀是否与库中事件逐条一致（文件大小未变时的完整核对）。

        抽样看不见“只改中段”；这里把整个已消费前缀解析出来，与 ``gpu_events`` 逐字段比较。
        只在 stat 变了但大小没变的可疑帧里做，正常追加不会走到。

        @param path 活动事件日志。
        @param offset 已消费到的字节数。
        @return 一致时为 True。
        """
        try:
            data = path.read_bytes()[:offset]
        except OSError:
            return False
        rows = []
        for raw in data.split(b"\n")[:-1]:
            try:
                item = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if not isinstance(item, dict):
                continue
            when = self._gpu_number(item.get("t"))
            if when is None:
                continue
            raw_pid = self._gpu_number(item.get("pid"))
            rows.append((
                when,
                int(raw_pid) if raw_pid is not None else 0,
                item.get("kind") if isinstance(item.get("kind"), str) else "",
                item.get("mode") if isinstance(item.get("mode"), str) else "",
                item.get("what") if isinstance(item.get("what"), str) else "",
            ))
        stored = [
            (row["t"], row["pid"], row["kind"], row["mode"], row["what"])
            for row in self.conn.execute("SELECT t, pid, kind, mode, what FROM gpu_events ORDER BY rowid")
        ]
        return rows == stored

    def _read_gpu_events(self, path: Path, offset: int) -> None:
        """@brief 从字节偏移增量读一份事件日志，只消费完整行。

        尾部半行留在 offset 之前，下一次追平再读，不会把写到一半的 JSON 当成事件。

        @param path 事件日志路径。
        @param offset 起始字节位置。
        """
        try:
            handle = path.open("rb")
        except OSError:
            return
        consumed = offset
        pending = b""
        row = self.conn.execute("SELECT MAX(t) AS max_t FROM gpu_events").fetchone()
        last_seen = [float(row["max_t"]) if row is not None and row["max_t"] is not None else None]
        late = False
        with handle:
            handle.seek(offset)
            while True:
                chunk = handle.read(65_536)
                if not chunk:
                    break
                consumed += len(chunk)
                pending += chunk
                while True:
                    cut = pending.find(b"\n")
                    if cut < 0:
                        break
                    if self._gpu_apply(pending[:cut], last_seen):
                        late = True
                    pending = pending[cut + 1:]
        if late:
            self._gpu_replay()
        if str(path) == str(detail.gpu_events_path()):
            new_offset = consumed - len(pending)
            self._meta_set("gpu_events_offset", str(new_offset))
            state = self._gpu_stat_state(path)
            if state:
                self._meta_set("gpu_events_stat", state)
            tail = self._gpu_tail_hash(path, new_offset)
            if tail:
                self._meta_set("gpu_events_tail_sha", tail)

    @staticmethod
    def _gpu_marker(path: Path) -> Path:
        """@brief 历史截断的源侧标记文件（``gpu_events.truncated``），与日志同目录。

        派生库里的 meta 删库就没了；把“曾经丢过历史”写成源侧事实，重建后也不会假装完整。
        只有人工恢复历史后显式删除这个文件才能清掉。

        @param path 活动事件日志。
        @return 标记文件路径。
        """
        return Path(path).with_suffix(".truncated")

    def _gpu_truncate(self, path: Path) -> None:
        """@brief 标记 GPU 历史不完整：写 meta 与源侧标记文件。

        @param path 活动事件日志。
        """
        self._meta_set("gpu_events_truncated", "1")
        try:
            marker = self._gpu_marker(path)
            if not marker.is_file():
                marker.write_text("gpu event history is incomplete; remove this file only after restoring shards\n",
                                  encoding="utf-8")
        except OSError:
            return

    def _gpu_shard_state(self) -> list[list]:
        """@brief 读出上次重建时记录的分片 ``[名字, 字节数]`` 列表。

        @return 分片状态列表；没有记录或坏 JSON 时为空。
        """
        raw = self._meta_get("gpu_events_shards")
        if not raw:
            return []
        try:
            state = json.loads(raw)
        except ValueError:
            return []
        return state if isinstance(state, list) else []

    def _record_gpu_shards(self, path: Path) -> None:
        """@brief 记录当前保留分片的名字与字节数，供后续检测分片被删/被改。

        @param path 当前事件日志。
        """
        state = []
        for shard in self._gpu_shards(path):
            try:
                stat = shard.stat()
            except OSError:
                continue
            state.append([shard.name, stat.st_size, stat.st_ino, stat.st_mtime_ns])
        self._meta_set("gpu_events_shards", json.dumps(state))

    def _recorded_shards_lost(self, path: Path, state: list) -> bool:
        """@brief 逐分片核对旧清单：记录过的分片必须还在且字节数不变。

        分片一旦生成就不再追加；缺失、被改短或被同名换成别的内容都说明历史不完整。
        不能用“剩余分片总字节数”替代逐分片核对：删掉一个较大的老分片、同时新轮转一个更大的
        分片，总字节数可能仍然够，但早期历史已经丢了。

        @param path 当前事件日志。
        @param state 上次记录的分片 ``[名字, 字节数]`` 列表。
        @return 有任何记录过分片缺失或字节数不符时为 True。
        """
        for item in state:
            if not isinstance(item, list) or len(item) < 2:
                continue
            name, size = item[0], item[1]
            try:
                stat = path.with_name(str(name)).stat()
            except OSError:
                return True
            if stat.st_size != int(size):
                return True
            if len(item) >= 4:
                # 不可变分片被等长替换：inode/mtime 会变。
                if stat.st_ino != int(item[2]) or stat.st_mtime_ns != int(item[3]):
                    return True
        return False

    def _check_gpu_shards(self, path: Path) -> bool:
        """@brief 同 inode 帧里，之前记录的分片是否还在且没变。

        @param path 当前事件日志。
        @return 有分片丢失或变化时为 True。
        """
        return self._recorded_shards_lost(path, self._gpu_shard_state())

    def _rebuild_gpu_events(self, path: Path, stat, previous_offset: int) -> None:  # noqa: ARG002
        """@brief inode 变化（轮转或替换）后重建派生表：按修改时间重放保留分片，再读当前文件。

        没有分片就说明旧内容已被删除，标记 ``gpu_events_truncated``；分片总字节少于上次消费量也
        标记截断，不假装历史完整。

        @param path 当前事件日志。
        @param stat 当前文件 stat。
        @param previous_offset 上次消费到的字节数。
        """
        had_history = self.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] > 0
        previously_truncated = self._meta_get("gpu_events_truncated") == "1"
        # 必须在覆盖旧清单之前逐分片核对：删掉一个老分片与 inode 变化同时发生时，只看剩余分片
        # 总字节数（甚至只看新清单）会漏掉，历史就会静默丢失且 truncated 仍为 0。
        recorded_lost = self._recorded_shards_lost(path, self._gpu_shard_state())
        shards = self._gpu_shards(path)
        retained = 0
        for shard in shards:
            try:
                retained += shard.stat().st_size
            except OSError:
                continue
        self._reset_gpu_events()
        # 截断是粘性的：除非能证明完整性恢复，否则不把“早期历史已丢”清回 0。
        lost = (
            previously_truncated
            or recorded_lost
            or (had_history and not shards)
            or (bool(shards) and retained < previous_offset)
        )
        self._meta_set("gpu_events_truncated", "0")
        if lost:
            self._gpu_truncate(path)
        for shard in shards:
            self._read_gpu_events(shard, 0)
        if path.is_file():
            self._read_gpu_events(path, 0)
        else:
            # 写者刚把活动文件改名成新分片、还没建新文件：历史仍由分片覆盖，不能标截断。
            self._meta_set("gpu_events_offset", "0")
            self._meta_set("gpu_events_checkpoint", "")
        self._record_gpu_shards(path)

    def ensure_gpu_events(self) -> None:
        """@brief 把机全局 GPU 事件日志追平进派生表；只读新字节。

        tick 与时间线面板都调它。文件被截断（同 inode 变短）时重建并标记 ``gpu_events_truncated``；
        inode 变了走分片重放。没有任何变化时只做一次 stat。
        """
        path = detail.gpu_events_path()
        try:
            stat = path.stat()
        except OSError:
            with _WRITE:
                offset = int(self._meta_get("gpu_events_offset") or 0)
                inode = int(self._meta_get("gpu_events_inode") or 0)
                shards = self._gpu_shards(path)
                if not shards:
                    if self.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] > 0:
                        self._gpu_truncate(path)
                        self.conn.commit()
                    return
                has_rows = self.conn.execute("SELECT COUNT(*) FROM gpu_events").fetchone()[0] > 0
                current = []
                for item in shards:
                    try:
                        stat = item.stat()
                    except OSError:
                        continue
                    current.append([item.name, stat.st_size, stat.st_ino, stat.st_mtime_ns])
                if (inode == 0 and offset == 0 and has_rows and not self._check_gpu_shards(path)
                        and current == self._gpu_shard_state()):
                    # 已经按这批分片重建过，且分片集合没变、活动文件仍未出现：不必每帧重放。
                    self.conn.commit()
                    return
                self._rebuild_gpu_events(path, None, offset)
                self._meta_set("gpu_events_inode", "0")
                self.conn.commit()
            return
        with _WRITE:
            if self._gpu_marker(path).is_file():
                self._meta_set("gpu_events_truncated", "1")
            offset = int(self._meta_get("gpu_events_offset") or 0)
            inode = int(self._meta_get("gpu_events_inode") or 0)
            if inode != stat.st_ino:
                self._rebuild_gpu_events(path, stat, offset)
            elif stat.st_size < offset:
                # 活动文件变短：已消费前缀变了，但仍存在的分片历史必须保留 → 分片感知重建。
                self._rebuild_gpu_events(path, stat, offset)
                self._gpu_truncate(path)
            else:
                current_stat = self._gpu_stat_state(path)
                previous_stat = self._meta_get("gpu_events_stat")
                if offset > 0 and previous_stat and previous_stat != current_stat and stat.st_size == offset:
                    # 大小没变但文件被写过（mtime/ctime 变了）：把整个已消费前缀与库中事件
                    # 逐条对拍。抽样看不见“只改中段”，这里做完整比较。
                    if not self._gpu_prefix_equivalent(path, offset):
                        self._rebuild_gpu_events(path, stat, offset)
                        self._gpu_truncate(path)
                    else:
                        self._meta_set("gpu_events_stat", current_stat)
                elif stat.st_size > offset:
                    # 追加：先用前缀尾部的 1KB 校验已消费内容没被换掉（缩短后再长回来的写者
                    # 竞态会在这里被抓住），再只读新增字节。
                    stored_tail = self._meta_get("gpu_events_tail_sha")
                    if stored_tail and stored_tail != self._gpu_tail_hash(path, offset):
                        self._rebuild_gpu_events(path, stat, offset)
                        self._gpu_truncate(path)
                    else:
                        self._read_gpu_events(path, offset)
            # 分片单独被删/被改短时，活动文件完全没变；不查一下就发现不了历史缺失。
            if self._check_gpu_shards(path):
                self._gpu_truncate(path)
            self._record_gpu_shards(path)
            self._meta_set("gpu_events_inode", str(stat.st_ino))
            self.conn.commit()

    def _already_ingested(self, directory: Path) -> bool:
        """@brief 这一帧是否已经摄取过这个目录。

        @param directory 运行目录。
        @return 已摄取时为 True。
        """
        return self._rel(directory) in self._frame_ingested

    def _ingest(self, directory: Path, *, force: bool = False, once: bool = True, payload: dict | None = None) -> None:
        """@brief 把一个目录（运行或队列）重新读进索引。

        指纹未变且已收录时只补轮次；否则重读快照、清单、候选、会话与反馈。

        @param directory 待摄取目录。
        @param force 为 True 时忽略指纹强制重读。
        @param payload 并行冷建预先读好的纯读输入；给出时不再重读快照与候选报告。
        """
        if not self._within_runs(directory):
            return
        # 用 canonical 目录读写：别名与直名落到同一 run id、同一份文件；根外祖先链接已在上面拒绝。
        directory = Path(self._resolved_text(directory))
        frame_key = self._rel(directory)
        if once and not force and frame_key in self._frame_ingested:
            return
        self._frame_ingested.add(frame_key)
        if payload is not None:
            # Worker 读完之后 state/result/config 又变了：payload 的旧内容不能配新指纹，退回串行。
            # 候选文件不在这里重扫：worker 的读写窗口已经包含它们；worker 返回后才变的候选会在
            # 下一次对账因候选指纹不同而重摄取（旧内容配旧指纹是安全方向）。
            post_key = (
                Index._state_fingerprint(directory),
                _fingerprint(directory / "result.json"),
                _fingerprint(directory / "run_config.json"),
            )
            if post_key != tuple(payload["fingerprints"][:3]):
                payload = None
        stored = self._stored_fingerprints(directory)
        winner_label, winner_rel = _confirm_target(stored[3])
        fingerprints = payload["fingerprints"] if payload is not None else self._cheap_fingerprints(directory, winner_label, winner_rel)
        known = self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (self._rel(directory),)).fetchone()
        tracked = "\x1f" in (stored[3] or "")
        if not force and fingerprints == stored and known and (tracked or not (directory / detail.target_file_names(directory)[1]).exists()):
            self._project_turns(directory)
            self._refresh_kernel_writer(directory)
            return
        # 读快照与算指纹之间文件可能被写者改写。若先读内容、后算指纹，会把“旧内容 + 新指纹”
        # 写进库：定向核对看到指纹已一致，就再也不会重摄取。这里用 stat/read/post-stat 校验，
        # 读取窗口内文件变了就整份重读（最多 3 次；仍不稳时用最后一次，下一次 tick 会自愈）。
        state_path = directory / "state.json"
        result_path = directory / "result.json"
        state = result = None
        if payload is not None:
            state, result = payload["state"], payload["result"]
            manifest = payload["manifest"]
            config = payload["config"]
            candidates = payload["candidates"]
        else:
            candidates = None
            manifest = None
            config = None
        if payload is None and not known:
            # 冷建的新行：顶部已经算过一次指纹（stat 白名单），这里只读一次内容。
            # 读窗口内若写者更新了文件，我们存的是**更旧**的指纹，下一帧必然不相等并重摄取；
            # 这是安全方向，也把冷建的指纹成本砍掉一半。
            state = self._read_json(state_path)
            result = self._read_json(result_path) if result_path.exists() else None
            # 顶部的指纹是在还不知道 winner 标签时算的（新行没有已存 fp_confirm），
            # 确认指纹必须用读到的标签重算一次，否则胜者内核的就地改写永远发现不了。
            label = result.get("confirm_winner_label") if isinstance(result, dict) else ""
            if not isinstance(label, str):
                label = ""
            fingerprints = (
                fingerprints[0], fingerprints[1], fingerprints[2],
                self._confirmation_fingerprint(directory, label, _winner_rel(directory, label)),
                fingerprints[4],
            )
            stable = True
        else:
            if payload is None:
                fingerprints = None
            stable = payload is not None
            for _attempt in range(3 if payload is None else 0):
                before = (_fingerprint(state_path), _fingerprint(result_path))
                state = self._read_json(state_path)
                result = self._read_json(result_path) if result_path.exists() else None
                label = (result or {}).get("confirm_winner_label") if isinstance(result, dict) else ""
                if not isinstance(label, str):
                    label = ""
                rel = _winner_rel(directory, label)
                # 指纹必须在同一次“读取窗口”里算，并且用这一组；读完之后再 stat 一次确认窗口没变。
                candidate_fingerprints = self._fingerprints(directory, label, rel)
                if before == (_fingerprint(state_path), _fingerprint(result_path)):
                    fingerprints = candidate_fingerprints
                    stable = True
                    break
            if not stable:
                # 三次都不稳：不写行，保留旧指纹，让下一次 tick/请求重试；不能把旧内容绑到新指纹上。
                self._project_turns(directory)
                return
        broken = result_path.exists() and result is None
        if payload is None:
            manifest = self._find_manifest(directory)
        if manifest is not None:
            self._upsert_experiment(manifest[0], manifest[1])
        if state and state.get("script") and result is None and not broken:
            self._ingest_queue(directory, state)
            return
        if manifest is not None and self._normalized(directory) == self._normalized(manifest[0]) and not (result or (state and state.get("task") and not state.get("script"))):
            return
        if not result and not (state and state.get("task") and not state.get("script")) and not broken:
            return
        if broken and state is None and result is None:
            self._upsert_missing(directory, fingerprints)
            self._project_turns(directory)
            return
        row = detail.row_from_files(directory, state, result)
        if broken and result is None and state is None:
            row["status"] = "missing"
        if config is None:
            config = self._read_json(directory / "run_config.json")
        self._upsert_run(directory, row, config, fingerprints, manifest)
        self._upsert_candidates(directory, self._rel(directory), row, result, state, candidates)
        # §11.4 选 (a)：冷建不投影 sessions。transcript 变了就丢掉旧投影，第一次读
        # sessions/model_stats 时由 ``ensure_sessions`` 懒投影；冷建读入的 transcript 字节 = 0。
        self._drop_stale_sessions(directory, self._rel(directory))
        self._project_turns(directory)
        from observe.feedback import sync_run_index

        sync_run_index(self.conn, self._rel(directory), directory)

    def _ingest_queue(self, directory: Path, state: dict) -> None:
        """@brief 把一个实验队列脚本的状态写进 experiments 表。

        @param directory 队列目录。
        @param state 解析后的 ``state.json``。
        """
        job = detail.experiment_job(directory)
        if job is None:
            return
        self.conn.execute(
            """
            INSERT INTO experiments (experiment_id, kind, driver, created_at, git_commit, frozen_json, config_hash, status)
            VALUES (?, 'queue', ?, ?, '', '{}', '', ?)
            ON CONFLICT(experiment_id) DO UPDATE SET driver = excluded.driver, status = excluded.status
            """,
            (job["id"], job["script"], state.get("started_at"), job["status"]),
        )
        self._bump()

    def _find_manifest(self, directory: Path) -> tuple[Path, dict] | None:
        """@brief 向上查找最近的 ``experiment.json``。

        @param directory 起始目录。
        @return 二元组（实验根目录, 清单字典）；到 ``runs/`` 根仍未找到时返回 None。
        """
        found = _find_manifest_at(Path(self._resolved_text(directory)), self._root_path()[1])
        if found is not None:
            self.stats["json_reads"] += 1
        return found

    def _upsert_experiment(self, root: Path, doc: dict) -> None:
        """@brief 写入或更新一个实验及其各 arm。

        @param root 实验根目录。
        @param doc ``experiment.json`` 解析后的文档。
        """
        experiment_id = doc.get("experiment_id") or self._rel(root)
        self.conn.execute(
            """
            INSERT INTO experiments (experiment_id, kind, driver, created_at, git_commit, frozen_json, config_hash, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                kind = excluded.kind, driver = excluded.driver, frozen_json = excluded.frozen_json,
                config_hash = excluded.config_hash
            """,
            (
                experiment_id,
                doc.get("kind"),
                doc.get("driver"),
                doc.get("created_at"),
                doc.get("git_commit") or "",
                json.dumps(doc.get("frozen") or {}, sort_keys=True),
                doc.get("config_hash") or "",
                "recorded",
            ),
        )
        for arm in doc.get("arms") or []:
            self.conn.execute(
                """
                INSERT INTO arms (experiment_id, arm_id, delta_json, planned_n, task_fingerprint)
                VALUES (?, ?, ?, ?, COALESCE((SELECT task_fingerprint FROM arms WHERE experiment_id = ? AND arm_id = ?), ''))
                ON CONFLICT(experiment_id, arm_id) DO UPDATE SET
                    delta_json = excluded.delta_json, planned_n = excluded.planned_n
                """,
                (
                    experiment_id,
                    arm.get("arm_id"),
                    json.dumps(arm.get("delta") or {}, sort_keys=True),
                    arm.get("planned_n") or 0,
                    experiment_id,
                    arm.get("arm_id"),
                ),
            )

    def _upsert_candidates(self, directory: Path, run_id: str, row: dict, result: dict | None, state: dict | None,
                           candidates: list[dict] | None = None) -> None:
        """@brief 每个候选目录一行；报告给出状态/阶段/加速比，轨迹给出噪声标记。

        原始英文说明：One row per candidate directory. The report supplies status/stage/speedup; the trace supplies the noise mark.

        @param directory 运行目录。
        @param run_id 运行 id。
        @param row 历史行。
        @param result 解析后的 ``result.json``，可为 None。
        @param state 解析后的 ``state.json``，可为 None。
        @param candidates 并行冷建预先读好的候选行；None 时现读。
        """
        self.conn.execute("DELETE FROM candidates WHERE run_id = ?", (run_id,))
        # §7.2：候选目录整个消失时也要清成员行；这个清理必须在任何早退之前。
        members = self._has_table("failure_members")
        if members:
            self.conn.execute("DELETE FROM failure_members WHERE run_id = ?", (run_id,))
        rows = candidates if candidates is not None else _candidate_payload(directory, result, state, self._read_json)
        if not rows:
            return
        observed_at = _num(row.get("started_at"))
        for item in rows:
            # 用 ON CONFLICT 而不是 INSERT OR REPLACE：REPLACE 的隐式删除在
            # recursive_triggers=OFF 时不会驱动 external-content FTS 的 delete 触发器，投影会损坏。
            self.conn.execute(
                """
                INSERT INTO candidates (
                    run_id, idx, label, status, stage, speedup, within_noise, eligible, kernel_sha256,
                    protocol_version, updated_at, message, failure_sig, cv, reps, cache_hit, improved,
                    report_state, build_shape, build_kind, evaluation_fingerprint
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id, idx) DO UPDATE SET
                    label = excluded.label, status = excluded.status, stage = excluded.stage,
                    speedup = excluded.speedup, within_noise = excluded.within_noise,
                    eligible = excluded.eligible, kernel_sha256 = excluded.kernel_sha256,
                    protocol_version = excluded.protocol_version, updated_at = excluded.updated_at,
                    message = excluded.message, failure_sig = excluded.failure_sig,
                    cv = excluded.cv, reps = excluded.reps, cache_hit = excluded.cache_hit,
                    improved = excluded.improved, report_state = excluded.report_state,
                    build_shape = excluded.build_shape, build_kind = excluded.build_kind,
                    evaluation_fingerprint = excluded.evaluation_fingerprint
                """,
                (run_id, item["idx"], item["label"], item["status"], item["stage"], item["speedup"],
                 item["within_noise"], item["eligible"], item["sha"], item["protocol_version"],
                 item["updated"], item["message"], item["sig"], item["cv"], item["reps"],
                 item["cache_hit"], item["improved"], item["report_state"], item["build_shape"],
                 item["build_kind"], item["evaluation_fingerprint"]),
            )
            if item["sig"] and members:
                self.conn.execute(
                    """
                    INSERT INTO failure_members (run_id, idx, sig, status, stage, observed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, idx) DO UPDATE SET
                        sig = excluded.sig, status = excluded.status, stage = excluded.stage,
                        observed_at = excluded.observed_at
                    """,
                    (run_id, item["idx"], item["sig"], item["status"], item["stage"], observed_at),
                )

    def _insert_session_event(self, run_id: str, seq: int, event: dict) -> None:
        """@brief 把一条 ``event=agent`` 插成 sessions 行；全量与增量投影共用。"""
        tokens = event.get("tokens") if isinstance(event.get("tokens"), dict) else {}
        runtime = event.get("runtime") if isinstance(event.get("runtime"), dict) else {}
        session_ids = runtime.get("session_ids") if isinstance(runtime.get("session_ids"), list) else []
        session_id = _diagnostic(session_ids[-1], 120) if session_ids and isinstance(session_ids[-1], str) else ""

        def count(name: str) -> int:
            value = runtime.get(name, 0)
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

        def token(name: str) -> int:
            value = tokens.get(name, 0)
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

        self.conn.execute(
            """
            INSERT INTO sessions (
                run_id, seq, label, candidate_label, attempt, committee, provider, model, outcome, cost_usd,
                session_id, failed_turns, last_kind, failure_class, error, failure_sig, tool_calls, diagnostics_path,
                tokens_input, tokens_output, tokens_cache_read, tokens_cache_write, t
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id, seq, str(event.get("label") or ""),
                str(event.get("candidate_label") or event.get("label") or ""),
                int(event.get("attempt")) if isinstance(event.get("attempt"), int) and not isinstance(event.get("attempt"), bool) and event.get("attempt") >= 0 else 0,
                str(event.get("committee") or ""),
                str(event.get("provider") or ""), str(event.get("model") or ""), str(event.get("outcome") or ""),
                _num(event.get("cost_usd")),
                session_id, count("failed_turns"), _diagnostic(runtime.get("last_terminal_kind"), 40),
                _diagnostic(runtime.get("failure_class"), 60),
                _diagnostic(event.get("error") or runtime.get("last_error")),
                failure_sig.session_failure_signature(
                    event.get("outcome"), runtime.get("failure_class"),
                    event.get("error") or runtime.get("last_error"),
                ),
                count("tool_calls"), _diagnostic(event.get("diagnostics_path"), 300),
                token("input"), token("output"), token("cache_read"), token("cache_write"), None,
            ),
        )

    def _clear_sessions_projection(self, run_id: str) -> None:
        """@brief 删除/移动/丢失一个 run 时清掉它的 sessions 投影游标与轮次投影 epoch。"""
        self.conn.execute(
            "DELETE FROM meta WHERE key IN (?, ?, ?, ?, ?, ?)",
            (f"sessions_fp:{run_id}", f"sessions_bytes:{run_id}", f"transcript_bytes:{run_id}",
             f"turns_epoch:{run_id}", f"reconcile_fp:{run_id}", f"kw_src:{run_id}"),
        )

    def _record_sessions_projection(self, directory: Path, offset: int) -> None:
        """@brief 记录 sessions 已投影的位置与当时的源指纹；都在 ``meta``，不改表结构。"""
        run_id = self._rel(directory)
        path = directory / "transcript.jsonl"
        try:
            stat = path.stat() if path.is_file() else None
        except OSError:
            stat = None
        fingerprint = f"{stat.st_mtime_ns}:{stat.st_size}" if stat is not None else "0:0"
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (f"sessions_fp:{run_id}", fingerprint))
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (f"sessions_bytes:{run_id}", str(offset)))

    def _drop_stale_sessions(self, directory: Path, run_id: str) -> None:
        """@brief 冷建/重摄取时不读 transcript，只丢掉源已变的旧 sessions 投影。

        @param directory 运行目录。
        @param run_id 运行 id。
        """
        path = directory / "transcript.jsonl"
        try:
            stat = path.stat() if path.is_file() else None
        except OSError:
            stat = None
        fingerprint = f"{stat.st_mtime_ns}:{stat.st_size}" if stat is not None else "0:0"
        # 记一个“有没有 transcript、多大”的便宜标记：巡检据此挑出待懒投影的 run，
        # 不必为没有 transcript 的运行打开目录（§6.4 的“只读索引”仍然成立）。
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (f"transcript_bytes:{run_id}", str(stat.st_size if stat is not None else 0)),
        )
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (f"sessions_fp:{run_id}",)).fetchone()
        if row is not None and row[0] == fingerprint:
            return
        self.conn.execute("DELETE FROM sessions WHERE run_id = ?", (run_id,))
        self.conn.execute(
            "DELETE FROM meta WHERE key IN (?, ?)",
            (f"sessions_fp:{run_id}", f"sessions_bytes:{run_id}"),
        )

    def _upsert_sessions(self, directory: Path, run_id: str, config: dict | None) -> None:
        """@brief 全量重建一个运行的 sessions；首次、重建或协议变化时使用。

        @param directory 运行目录。
        @param run_id 运行 id。
        @param config 解析后的 ``run_config.json``，可为 None。
        """
        self.conn.execute("DELETE FROM sessions WHERE run_id = ?", (run_id,))
        path = directory / "transcript.jsonl"
        if not path.is_file():
            self._record_sessions_projection(directory, 0)
            return
        try:
            size = path.stat().st_size
        except OSError:
            self._record_sessions_projection(directory, 0)
            return
        # §11.4：flame_chase_1 也受 >1MB 上限约束。
        if size > 1_000_000:
            self._record_sessions_projection(directory, size)
            return
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except OSError:
            self._record_sessions_projection(directory, 0)
            return
        seq = 0
        with handle:
            for line in handle:
                if '"agent"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict) or event.get("event") != "agent":
                    continue
                seq += 1
                self._insert_session_event(run_id, seq, event)
        self._record_sessions_projection(directory, size)

    def _set_kernel_writer(self, run_id: str, value: str) -> bool:
        """@brief 写 ``runs.kernel_writer``；值真的变了才抬高详情 revision。

        @param run_id 运行 id。
        @param value ``"pi"``/``"chat"``。
        @return 值是否发生变化。
        """
        row = self.conn.execute("SELECT kernel_writer FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None or str(row["kernel_writer"] or "") == value:
            return False
        self.conn.execute("UPDATE runs SET kernel_writer = ? WHERE run_id = ?", (value, run_id))
        self._mark_changed(run_id)
        return True

    def _kernel_writer_sources(self, directory: Path) -> dict:
        """@brief ``kernel_writer`` 兜底推导的完整来源状态（配置、transcript、工作区）。

        与 ``detail.kernel_writer`` 读取的是同一批源：``run_config.json`` 的 stat（配置内容变了要重解析）、
        ``transcript.jsonl`` 的 ``(mtime_ns,size)``、以及 ``agent_work`` 的三种状态
        （``absent``/``empty``/``nonempty``——``detail`` 只关心“目录存在且非空”）。
        原始英文说明：The full source state behind the derived writer, matching detail.kernel_writer.

        @param directory 运行目录。
        @return 含 ``config_fp``/``transcript_fp``/``size``/``work`` 的字典。
        """
        def stat_or_none(name: str):
            try:
                return os.stat(os.fspath(directory / name))
            except OSError:
                return None

        config = stat_or_none("run_config.json")
        transcript = stat_or_none("transcript.jsonl")
        if transcript is not None and not statlib.S_ISREG(transcript.st_mode):
            transcript = None
        work = "absent"
        try:
            with os.scandir(os.fspath(directory / "agent_work")) as entries:
                work = "nonempty" if any(entries) else "empty"
        except OSError:
            work = "absent"
        return {
            "config_fp": f"{config.st_mtime_ns}:{config.st_size}" if config is not None else "",
            "transcript_fp": f"{transcript.st_mtime_ns}:{transcript.st_size}" if transcript is not None else "0:0",
            "size": transcript.st_size if transcript is not None else 0,
            "work": work,
        }

    def _refresh_kernel_writer(self, directory: Path) -> bool:
        """@brief 复核 ``kernel_writer`` 的窄投影，返回写入值是否变化。

        ``runs.kernel_writer`` 是摘要字段 ``kernel_writer`` 的摄入窄投影（摘要戳含这个值），所以它必须
        与 ``detail.kernel_writer`` 逐字同义：

        1. 配置记了 ``search.kernel_agent``（任何字符串，含空串）就由配置决定；
        2. 否则 ``agent_work`` 非空 → ``pi``；
        3. 否则 transcript 里出现 ``event=agent`` → ``pi``；都没有 → ``chat``。

        来源状态（配置 stat、transcript 的 ``(mtime_ns,size)``、``agent_work`` 三态）变了才重新判定。
        已结束的运行不假设“只追加”：transcript 指纹一动就从头分块重扫（等长改写、变长替换、清空都
        算），因此 ``pi`` 能正确降回 ``chat``；只有仍在跑的运行才沿用已扫前缀（活写者只追加），并且
        尾行每次重读，所以无换行结尾的 agent 记录也能命中。

        原始英文说明：Keep the derived writer identical to detail.kernel_writer; ended runs re-check
        from scratch whenever the transcript fingerprint moves.

        @param directory 运行目录。
        @return 写入值是否发生变化。
        """
        run_id = self._rel(directory)
        row = self.conn.execute(
            "SELECT kernel_writer, status FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return False
        sources = self._kernel_writer_sources(directory)
        # 记录 transcript 是否存在/多大：巡检据此决定要不要为这个 run 懒投影 sessions。
        # 写在这里是因为对账/懒读都会经过它，transcript 后出现也能被跟上。
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (f"transcript_bytes:{run_id}", str(sources.get("size") or 0)),
        )
        key = f"kw_src:{run_id}"
        stored = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        previous: dict = {}
        if stored is not None:
            try:
                decoded = json.loads(stored[0])
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, dict):
                previous = decoded
        if previous and all(previous.get(name) == value for name, value in sources.items()):
            # 来源一个字节都没变：结论不会变，也不重扫。
            return False
        if previous.get("config_fp") == sources["config_fp"] and "recorded" in previous:
            recorded = previous.get("recorded") if isinstance(previous.get("recorded"), str) else None
        else:
            recorded = _recorded_kernel_agent(directory)
        found = False
        scanned = 0
        if recorded is None and sources["work"] != "nonempty":
            transcript = directory / "transcript.jsonl"
            previous_size = int(previous.get("size") or 0)
            if row["status"] == "running" and previous and sources["size"] > previous_size:
                # 活写者只会追加：沿用已扫前缀（尾行由分块扫描重读，命中即停）。
                scanned = int(previous.get("scanned") or 0)
                found = bool(previous.get("found"))
                if not found:
                    found, scanned = _scan_agent_events(transcript, scanned)
            else:
                # 已结束（或文件变短/等长）：不假设纯追加，从头分块复核。
                found, scanned = _scan_agent_events(transcript, 0)
        if recorded is not None:
            value = "pi" if recorded == "pi" else "chat"
        elif sources["work"] == "nonempty" or found:
            value = "pi"
        else:
            value = "chat"
        self._meta_set(key, json.dumps({
            **sources, "recorded": recorded, "scanned": scanned, "found": found,
        }, sort_keys=True))
        return self._set_kernel_writer(run_id, value)

    def _upsert_missing(self, directory: Path, fingerprints: tuple[str, str, str, str, str]) -> None:
        """@brief 把解析不出来的运行记为 ``missing``，并清掉它的候选与会话。

        @param directory 运行目录。
        @param fingerprints 现算的五个指纹。
        """
        run_id = self._rel(directory)
        previous = self.conn.execute(
            "SELECT status, task, dtype, size, target_id, protocol_version, started_at, spent_max_usd,"
            " publication, published_speedup FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if previous is not None:
            self._rollup_remove_row(dict(previous), run_id)
        self._record_run_parent(directory)
        self.conn.execute("DELETE FROM candidates WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM sessions WHERE run_id = ?", (run_id,))
        if self._has_table("run_pending_eval"):
            self.conn.execute("DELETE FROM run_pending_eval WHERE run_id = ?", (run_id,))
        if self._has_table("failure_members"):
            self.conn.execute("DELETE FROM failure_members WHERE run_id = ?", (run_id,))
        self._clear_sessions_projection(run_id)
        self.conn.execute(
            """
            INSERT INTO runs (run_id, status, source, binding, fp_state, fp_result, fp_config, fp_confirm, fp_extra, row_json)
            VALUES (?, 'missing', 'single', 'none', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET status = 'missing', publication = NULL, published_speedup = NULL,
                binding = 'none', fp_state = excluded.fp_state, fp_result = excluded.fp_result,
                fp_config = excluded.fp_config, fp_confirm = excluded.fp_confirm, fp_extra = excluded.fp_extra,
                row_json = excluded.row_json
            """,
            (run_id, *fingerprints, json.dumps({"id": run_id, "status": "missing"})),
        )
        self._mark_changed(run_id)
        self._sync_confirmed_best(run_id)

    def _rollup_adjust(self, fields, *, runs: int, confirmed: int, spent: float,
                       published: float | None, exclude_run_id: str | None = None) -> None:
        """@brief 在同一个事务里给一个汇总桶加减一行运行。

        @param fields ``_rollup_fields`` 给出的六元组。
        @param runs 本次运行的增减（+1 / -1）。
        @param confirmed 确认行增减。
        @param spent 花费增减。
        @param published 已发布加速比；None 表示不参与 ``published_sum``/``published_max``。
        @param exclude_run_id 重算 ``published_max`` 时排除的行（正在删除/正在改写的这一行）。
        """
        if fields is None or not self._has_table("run_rollup"):
            return
        self.conn.execute(
            """
            INSERT INTO run_rollup (day, task, dtype, size, target_id, protocol_epoch,
                                    runs, confirmed, spent_usd, published_sum, published_max)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(day, task, dtype, size, target_id, protocol_epoch) DO UPDATE SET
                runs = run_rollup.runs + excluded.runs,
                confirmed = run_rollup.confirmed + excluded.confirmed,
                spent_usd = run_rollup.spent_usd + excluded.spent_usd,
                published_sum = run_rollup.published_sum + excluded.published_sum,
                published_max = MAX(COALESCE(run_rollup.published_max, excluded.published_max),
                                    COALESCE(excluded.published_max, run_rollup.published_max))
            """,
            (*fields, runs, confirmed, spent, published or 0.0, published),
        )
        if published is not None:
            self._rollup_recompute_max(fields, exclude_run_id)
        self.conn.execute(
            "DELETE FROM run_rollup WHERE day = ? AND task = ? AND dtype = ? AND size = ? AND target_id = ? "
            "AND protocol_epoch = ? AND runs <= 0",
            fields,
        )

    def _rollup_recompute_max(self, fields, exclude_run_id: str | None = None) -> None:
        """@brief 从一个桶的现存行重算 ``published_max``（减法后旧最大值可能已离开）。

        @param fields 六元组桶键。
        @param exclude_run_id 重算时排除的行；删除/改写这一行时它已不该计入。
        """
        if not self._has_table("run_rollup"):
            return
        extra = " AND run_id != ?" if exclude_run_id is not None else ""
        args = [fields[1], fields[2], fields[3], fields[4], fields[0], fields[5]]
        if exclude_run_id is not None:
            args.append(exclude_run_id)
        row = self.conn.execute(
            """
            SELECT MAX(published_speedup) FROM runs
            WHERE COALESCE(task, '') = ? AND COALESCE(dtype, '') = ? AND COALESCE(size, '') = ?
              AND COALESCE(target_id, '') = ?
              AND (CASE WHEN started_at >= 1000000000
                        THEN strftime('%Y-%m-%d', started_at, 'unixepoch') ELSE '' END) = ?
              AND (CASE WHEN protocol_version = 'flame_chase_1' THEN '2' ELSE COALESCE(protocol_version, '') END) = ?
            """ + extra,
            args,
        ).fetchone()
        self.conn.execute(
            "UPDATE run_rollup SET published_max = ? WHERE day = ? AND task = ? AND dtype = ? AND size = ? "
            "AND target_id = ? AND protocol_epoch = ?",
            ((row[0] if row else None), *fields),
        )

    def _rollup_remove_row(self, row: dict, run_id: str | None = None) -> None:
        """@brief 删除/失联一行运行前，把它的贡献从汇总桶里减掉。

        @param row 至少含桶键字段与 ``publication``/``spent_max_usd``/``published_speedup``。
        @param run_id 这一行的 run id；用于重算 ``published_max`` 时把它排除。
        """
        if not row or row.get("status") == "missing":
            return
        published = _num(row.get("published_speedup"))
        self._rollup_adjust(
            _rollup_fields(row),
            runs=-1,
            confirmed=-1 if row.get("publication") == "confirmed" else 0,
            spent=-float(_num(row.get("spent_max_usd")) or 0.0),
            published=-published if published is not None else None,
            exclude_run_id=run_id or str(row.get("run_id") or "") or None,
        )

    def _upsert_run(self, directory: Path, row: dict, config: dict | None, fingerprints: tuple[str, str, str, str, str], manifest: tuple[Path, dict] | None) -> None:
        """@brief 写入或更新一个运行行，并在必要时更新 arm 锚点。

        已花费只增不减（``spent_max`` 取历史最大值），冻结的实验条件用于判定是否漂移。

        @param directory 运行目录。
        @param row ``row_from_files`` 生成的历史行。
        @param config 解析后的 ``run_config.json``，可为 None。
        @param fingerprints 现算的五个指纹。
        @param manifest 找到的实验清单，可为 None。
        """
        run_id = row["id"]
        source, experiment_id, arm_id, trial, kind, delta, binding = self._identity(directory, config, manifest)
        previous = self.conn.execute(
            "SELECT status, task, dtype, size, target_id, protocol_version, started_at, spent_max_usd,"
            " publication, published_speedup, kernel_writer FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        spent = _num(row.get("spent_usd"))
        previous_max = _num(previous["spent_max_usd"]) if previous else None
        if spent is None:
            spent_max = previous_max
        elif previous_max is None:
            spent_max = spent
        else:
            spent_max = max(previous_max, spent)
        search = (config or {}).get("search") or {}
        model = ((config or {}).get("llm") or {}).get("model") or (row.get("result") or {}).get("model") or (row.get("state") or {}).get("model")
        result_doc = row.get("result") if isinstance(row.get("result"), dict) else {}
        kernel_sha = str(result_doc.get("confirmation_source_sha256") or "") or (_sha256_file(directory / detail.target_file_names(directory)[1]) or "")
        config_target = (config or {}).get("target") if isinstance((config or {}).get("target"), dict) else {}
        backend = str(row.get("backend") or config_target.get("backend") or "")
        backend_version = str(row.get("backend_version") or config_target.get("backend_version") or "")
        writer_recorded = detail.recorded_kernel_writer(directory, config)
        # 冷建（新行）不读 transcript：便宜来源判断不了时先按 chat 投影，下一次对账/读取再按内容纠正。
        writer = writer_recorded or ((previous["kernel_writer"] if previous is not None else "") or "chat")
        protocol = str((config or {}).get("protocol_version") or result_doc.get("protocol_version") or "")
        if not protocol:
            # Older runs do not carry an explicit epoch. Keep the four flame_chase_1 runs out of
            # the legacy/chat bucket; the rest are the deprecated chat protocol.
            protocol = "flame_chase_1" if str((config or {}).get("loop_protocol") or "") == "flame_chase_1" else "legacy"
        committees = json.dumps((config or {}).get("committees") or [], sort_keys=True, ensure_ascii=False)
        stored = {key: row.get(key) for key in row if key not in ("state", "result")}
        for key in ("winner_label", "winner_provider", "winner_model", "winner_committee", "winner_origin"):
            if key in result_doc:
                stored[key] = result_doc.get(key)
        fingerprint = (config or {}).get("task_fingerprint")
        stored.update({
            "experiment_id": experiment_id,
            "arm_id": arm_id,
            "trial": trial,
            "kind": kind,
            "source": source,
            "binding": binding,
            "model": model or "",
            "switch_delta": delta,
            "task_fingerprint": fingerprint if isinstance(fingerprint, str) else "",
        })
        # §5.6 的窄投影：control / topped_up 不能在读时靠读盘重推（control 还看 resume.lock 持有者），
        # 所以在摄入时写成列；budget_rows() 只 SELECT。
        control = row.get("control") if isinstance(row.get("control"), dict) else {}
        control_action = control.get("action") or None
        control_reason = str(control.get("reason") or "")
        # §6.4：发布原因码与 ``result.confirmed_speedup`` 是巡检告警的索引投影。在这里算一次，
        # 巡检就只读窄列，不再逐行打开 run 目录。
        from experiments.common import publication_reason

        reason_code = str(publication_reason(directory, result_doc or None).get("reason") or "")
        self.conn.execute(
            """
            INSERT INTO runs (
                run_id, experiment_id, arm_id, trial, kind, task, operator, dtype, size, target_id,
                backend, backend_version, kernel_sha256, protocol_version, committees_json, model, preset,
                switch_delta_json, contract, status, phase, stop_reason, stop_code, started_at, updated_at,
                spent_usd, spent_max_usd, budget_usd, granted_budget_usd, publication, published_speedup,
                naive_speedup, search_best_speedup, control_action, control_reason, topped_up, initial_budget_usd,
                kernel_writer, rounds, evaluations, inflight_label, pid, source, binding,
                publication_reason, result_confirmed_speedup,
                fp_state, fp_result, fp_config, fp_confirm, fp_extra, row_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                experiment_id = excluded.experiment_id, arm_id = excluded.arm_id, trial = excluded.trial,
                kind = excluded.kind, task = excluded.task, operator = excluded.operator, dtype = excluded.dtype,
                size = excluded.size, target_id = excluded.target_id, backend = excluded.backend,
                backend_version = excluded.backend_version, kernel_sha256 = excluded.kernel_sha256,
                protocol_version = excluded.protocol_version, committees_json = excluded.committees_json,
                model = excluded.model, preset = excluded.preset,
                switch_delta_json = excluded.switch_delta_json, contract = excluded.contract,
                status = excluded.status, phase = excluded.phase, stop_reason = excluded.stop_reason,
                stop_code = excluded.stop_code,
                started_at = excluded.started_at, updated_at = excluded.updated_at,
                spent_usd = excluded.spent_usd, spent_max_usd = excluded.spent_max_usd,
                budget_usd = excluded.budget_usd, granted_budget_usd = excluded.granted_budget_usd,
                publication = excluded.publication, published_speedup = excluded.published_speedup,
                naive_speedup = excluded.naive_speedup, search_best_speedup = excluded.search_best_speedup,
                control_action = excluded.control_action,
                control_reason = excluded.control_reason, topped_up = excluded.topped_up,
                initial_budget_usd = excluded.initial_budget_usd, kernel_writer = excluded.kernel_writer,
                rounds = excluded.rounds, evaluations = excluded.evaluations,
                inflight_label = excluded.inflight_label, pid = excluded.pid, source = excluded.source,
                binding = excluded.binding, publication_reason = excluded.publication_reason,
                result_confirmed_speedup = excluded.result_confirmed_speedup,
                fp_state = excluded.fp_state, fp_result = excluded.fp_result,
                fp_config = excluded.fp_config, fp_confirm = excluded.fp_confirm, fp_extra = excluded.fp_extra,
                row_json = excluded.row_json
            """,
            (
                run_id, experiment_id, arm_id, trial, kind, row.get("task") or "", operator_name(row.get("task")),
                row.get("dtype") or "", (config or {}).get("size") or "", _target_id(config),
                backend, backend_version, kernel_sha, protocol, committees, model or "", (config or {}).get("preset") or "",
                json.dumps(delta, sort_keys=True), "", row.get("status"), row.get("phase") or "", row.get("stop_reason") or "",
                row.get("stop_code") or "",
                row.get("started_at"), _num((row.get("state") or {}).get("updated_at")),
                spent, spent_max, _num(row.get("disk_budget_usd")), _num(row.get("budget_usd")),
                row.get("publication"), _num(row.get("published_speedup")), _num((row.get("result") or {}).get("naive_speedup")),
                _num(result_doc.get("best_speedup")),
                control_action, control_reason, 1 if row.get("topped_up") else 0, _num(row.get("initial_budget_usd")),
                writer,
                _num((row.get("state") or {}).get("round")), _num((row.get("result") or {}).get("evaluations")),
                row.get("inflight_label") or "", _num((row.get("state") or {}).get("pid")), source, binding,
                reason_code, _num(result_doc.get("confirmed_speedup")),
                *fingerprints, json.dumps(stored, default=str),
            ),
        )
        self._upsert_pending_eval(run_id, directory)
        # §11.2：同一个事务里先减旧桶、再加新桶；键含 dtype/size/task/target/纪元，不跨桶求和。
        if previous is not None:
            self._rollup_remove_row(dict(previous), run_id)
        new_fields = _rollup_fields({
            "task": row.get("task"),
            "dtype": row.get("dtype"),
            "size": (config or {}).get("size"),
            "target_id": _target_id(config),
            "protocol_version": protocol,
            "started_at": row.get("started_at"),
        })
        self._rollup_adjust(
            new_fields,
            runs=1,
            confirmed=1 if row.get("publication") == "confirmed" else 0,
            spent=float(spent_max or 0.0),
            published=_num(row.get("published_speedup")),
        )
        # 每个有效 run 都要记录父目录：否则同父目录下已有“非确认” run 时，新 run 的父目录
        # 不在发现集合里，/best 会永久看不到它。
        self._record_run_parent(directory)
        # 只有“本来是/现在变成已确认”的行才需要碰 confirmed_best；否则绝大多数运行都白跑一条 DELETE。
        was_confirmed = previous is not None and previous["publication"] == "confirmed"
        if was_confirmed or row.get("publication") == "confirmed":
            self._sync_confirmed_best(run_id)
        self._ledger_row(run_id, row, spent_max, source, binding)
        # 冷建的第一行不读 transcript（便宜来源判断不了时先按 chat 投影）；已存在的行重新摄取、
        # 或便宜来源能判断时才落 ``kw_src`` 并纠正旧运行。
        if previous is not None or writer_recorded is not None:
            self._refresh_kernel_writer(directory)
        self._mark_changed(run_id)
        if source == "manifest" and arm_id:
            self._reanchor(experiment_id, arm_id)

    def _upsert_pending_eval(self, run_id: str, directory: Path) -> None:
        """@brief 把 outbox 的逐 run 窄投影写进 ``run_pending_eval``。

        这张表让摘要与巡检不用解码 ``row_json``，也不用每请求去 glob ``evaluation_outbox``。
        请求列表保留旧形状的最后 ≤8 条（含 ``request_id/status/error_code/retryable``），
        ``pending``/``unavailable`` 是全量计数。
        原始英文说明：Materialise the per-run outbox projection so reads never parse row_json.

        @param run_id 运行 id。
        @param directory 运行目录。
        """
        if not self._has_table("run_pending_eval"):
            return
        fields = detail.pending_evaluation_table(directory)
        self.conn.execute(
            """
            INSERT INTO run_pending_eval
                (run_id, pending, unavailable, last_error_code, retryable, requests_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                pending = excluded.pending, unavailable = excluded.unavailable,
                last_error_code = excluded.last_error_code, retryable = excluded.retryable,
                requests_json = excluded.requests_json, updated_at = excluded.updated_at
            """,
            (
                run_id,
                int(fields["pending"]),
                int(fields["unavailable"]),
                fields["last_error_code"],
                1 if fields["retryable"] else 0,
                json.dumps(fields["requests"], ensure_ascii=False),
                fields["updated_at"],
            ),
        )

    def stored_pending_evaluation(self, run_id: str) -> dict | None:
        """@brief 读 ``run_pending_eval`` 的窄投影，形状与 ``detail.pending_evaluation`` 一致。

        没有这一行（未摄入的 run）或旧结构没有这张表时返回 None，调用方退回读文件。
        原始英文说明：The stored pending-evaluation projection, or None when it is not indexed.

        @param run_id 运行 id。
        @return 含 ``count`` 与 ``requests`` 的字典，或 None。
        """
        if not self._has_table("run_pending_eval"):
            return None
        row = self.conn.execute(
            "SELECT pending, unavailable, requests_json FROM run_pending_eval WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            requests = json.loads(row["requests_json"] or "[]")
        except ValueError:
            requests = []
        if not isinstance(requests, list):
            requests = []
        return {
            "count": int(row["pending"] or 0) + int(row["unavailable"] or 0),
            "requests": [item for item in requests if isinstance(item, dict)],
        }

    def _reanchor(self, experiment_id: str, arm_id: str) -> None:
        """@brief 用最小 trial 的任务指纹作锚点，而不是先扫到的那个目录。

        原始英文说明：The anchor is the task fingerprint of the lowest trial, not whichever directory was scanned first.

        @param experiment_id 实验 id。
        @param arm_id arm id。
        """
        if self._in_reanchor or not experiment_id or not arm_id:
            return
        rows = list(self.conn.execute(
            """
            SELECT run_id, trial, row_json FROM runs
            WHERE experiment_id = ? AND arm_id = ? AND status != 'missing' AND source = 'manifest'
            ORDER BY COALESCE(trial, 1000000000), run_id
            """,
            (experiment_id, arm_id),
        ))
        chosen = ""
        for item in rows:
            try:
                doc = json.loads(item["row_json"] or "{}")
            except ValueError:
                continue
            found = doc.get("task_fingerprint") or ""
            if isinstance(found, str) and found:
                chosen = found
                break
        if not chosen:
            return
        current = self.conn.execute(
            "SELECT task_fingerprint FROM arms WHERE experiment_id = ? AND arm_id = ?",
            (experiment_id, arm_id),
        ).fetchone()
        if current is None or (current["task_fingerprint"] or "") == chosen:
            return
        self.conn.execute(
            "UPDATE arms SET task_fingerprint = ? WHERE experiment_id = ? AND arm_id = ?",
            (chosen, experiment_id, arm_id),
        )
        self._in_reanchor = True
        try:
            for item in rows:
                path = detail.resolve_run(item["run_id"], root=self._root_path()[0])
                if path is not None:
                    self._ingest(path, force=True)
        finally:
            self._in_reanchor = False

    def _has_table(self, name: str) -> bool:
        """@brief 判断库里有没有这张表（schema-11 结构没有 schema-12 的新表）。

        @param name 表名。
        @return 存在时为 True。
        """
        if name not in self._tables:
            row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
            ).fetchone()
            self._tables[name] = row is not None
        return self._tables[name]

    def _record_run_parent(self, directory: Path) -> None:
        """@brief 记住这个 run 的父目录，供 ``discover_new`` 用很小的代价发现新 run。

        这只是加速表：schema-11 结构里没有它时，``discover_new`` 会从 ``runs.run_id`` 现推父目录，
        PR4 的语义不依赖这张表存在。

        @param directory 运行目录。
        """
        if not self._has_table("run_parents"):
            return
        try:
            parent = self._rel(directory.parent)
        except ValueError:
            return
        self.conn.execute("INSERT OR IGNORE INTO run_parents (path) VALUES (?)", (parent,))

    def _mark_missing(self, run_id: str) -> bool:
        """@brief 统一的“运行不见了”清理：清发布字段、候选与 confirmed_best。

        物理删除、running→missing、broken run 都走这里，避免某条路径留下 candidates/FTS 或
        confirmed_best 的陈旧投影。schema-11 结构没有 confirmed_best 时跳过那张表。

        @param run_id 运行 id。
        @return 这次调用是否真的把行改成 missing。
        """
        previous = self.conn.execute(
            "SELECT status, task, dtype, size, target_id, protocol_version, started_at, spent_max_usd,"
            " publication, published_speedup FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        cur = self.conn.execute(
            "UPDATE runs SET status = 'missing', publication = NULL, published_speedup = NULL "
            "WHERE run_id = ? AND status != 'missing'",
            (run_id,),
        )
        self.conn.execute("DELETE FROM candidates WHERE run_id = ?", (run_id,))
        if self._has_table("confirmed_best"):
            self.conn.execute("DELETE FROM confirmed_best WHERE run_id = ?", (run_id,))
        if self._has_table("run_pending_eval"):
            self.conn.execute("DELETE FROM run_pending_eval WHERE run_id = ?", (run_id,))
        if self._has_table("failure_members"):
            self.conn.execute("DELETE FROM failure_members WHERE run_id = ?", (run_id,))
        if cur.rowcount:
            if previous is not None:
                self._rollup_remove_row(dict(previous), run_id)
            # 物理删除后 sessions 行不会自己消失，派生游标也必须一起清掉；否则同一个 id 重建时
            # kw_src/reconcile_fp/sessions_bytes 会沿用已删除目录的来源状态。
            self._clear_sessions_projection(run_id)
            self._mark_changed(run_id)
            return True
        return False

    def discover_new(self) -> None:
        """@brief 只按“可能装着新 run 的父目录”的 mtime 发现新运行，不扫全部 watched。

        新 run 一定先改它父目录的 mtime；父目录集合是历史上每个 run 的 parent（真实语料里是
        target 目录与实验 arm 目录，数量很小）。新父目录本身被创建时，会改它自己父目录的
        mtime，而那个父目录也在集合里（或就是 runs/ 根）。

        原始英文说明：Find new runs by statting only the directories that hold runs.
        """
        with _WRITE:
            self._prepare_frame()
            root = detail.runs_root()
            if not root.is_dir():
                self._frame_ingested.clear()
                return
            if self._has_table("run_parents"):
                parents = [row["path"] for row in self.conn.execute("SELECT path FROM run_parents")]
            else:
                # schema-11 结构：没有加速表时从**历史** run_id（含 missing）现推父目录，语义相同。
                # 某父目录下最后一场被删后，它的 missing 行仍能保住这个父目录，之后新建的 sibling
                # 才不会被漏掉；顶层 watched 目录再兜住“归档把行删光”的情况。
                parents = {
                    str(Path(row["run_id"]).parent)
                    for row in self.conn.execute("SELECT run_id FROM runs")
                }
                parents.update(
                    row["path"] for row in self.conn.execute(
                        "SELECT path FROM watched WHERE path <> '.' AND path NOT LIKE '%/%'"
                    )
                )
            # 被删掉的父目录（如 conv2d/exp）再重建时，改的是它**祖先**的 mtime；把每一级祖先都
            # 放进候选集，才能在一次发现里走到新 run。
            candidates = {""}
            for item in parents:
                if item in ("", "."):
                    continue
                relative = Path(item)
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                candidates.add(item)
                candidates.update(str(parent) for parent in relative.parents if str(parent) != ".")
            ordered = sorted(candidates, key=lambda item: (item.count("/"), item))
            for rel in ordered:
                path = root if rel in ("", ".") else root / rel
                try:
                    path.stat()
                except OSError:
                    # 保留 sentinel：父目录消失后再重建时，它仍要在候选集里。
                    continue
                self._note_dir(path, force=False)
            # 一帧内从“在跑”变成“已确认”的运行：复查仍在跑的少量行（旧 tick 也会做这一步）。
            self._refresh_running()
            self.conn.commit()
            self._frame_ingested.clear()

    def _confirmed_at(self, run_id: str) -> float | None:
        """@brief 确认时刻：``confirm_report.json`` 的 mtime，取不到时为 None。

        ``confirmed_best.confirmed_at`` 以前写的是 ``runs.started_at``，于是“最后确认”被算成开始时间
        （实测会得出两万多天的 stale）；确认报告的 mtime 才是这次确认落盘的时刻。
        原始英文说明：When the confirmation landed (the confirm report's mtime), not when the run started.

        @param run_id 运行 id。
        @return Unix 秒；目录或文件不存在时为 None。
        """
        directory = detail.resolve_run(run_id, root=self._root_path()[0])
        if directory is None:
            return None
        try:
            return (directory / "confirm_report.json").stat().st_mtime
        except OSError:
            return None

    def _sync_confirmed_best(self, run_id: str) -> None:
        """@brief 维护 ``confirmed_best`` 物化行：发布门放行时留一行，其余情况删除。

        这张表只是 ``/best`` 的排序/过滤加速，事实仍来自 ``runs``，由索引在写路径维护；
        读路径（PR4）不需要知道它存在。

        @param run_id 运行 id。
        """
        if not self._has_table("confirmed_best"):
            return
        row = self.conn.execute(
            "SELECT task, dtype, size, target_id, protocol_version, published_speedup, kernel_sha256, "
            "started_at, publication FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None or row["publication"] != "confirmed" or row["published_speedup"] is None:
            self.conn.execute("DELETE FROM confirmed_best WHERE run_id = ?", (run_id,))
            return
        epoch = "2" if str(row["protocol_version"] or "") == "flame_chase_1" else str(row["protocol_version"] or "")
        self.conn.execute(
            """
            INSERT INTO confirmed_best (run_id, task, dtype, size, target_id, protocol_epoch,
                                        published_speedup, kernel_sha256, confirmed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                task = excluded.task, dtype = excluded.dtype, size = excluded.size,
                target_id = excluded.target_id, protocol_epoch = excluded.protocol_epoch,
                published_speedup = excluded.published_speedup, kernel_sha256 = excluded.kernel_sha256,
                confirmed_at = excluded.confirmed_at
            """,
            (
                run_id, row["task"], row["dtype"], row["size"], row["target_id"], epoch,
                row["published_speedup"], row["kernel_sha256"], self._confirmed_at(run_id),
            ),
        )

    def _resolve_history_run(self, raw: str, winner_label: str, source_sha256: str) -> str:
        """@brief 把目录史里的 ``run_id_raw`` 解析成索引里的 run id（§8.1 第 2 步 + §8.2）。

        顺序：① 已含 ``/`` 且能命中索引 → 直接用；② 否则取所有以 ``/`` + raw 结尾的 run；
        恰好 1 个 → 命中；多个 → 先用 ``winner_label`` 过滤，再用 ``source_sha256`` 联合消歧
        （需要读候选内核算 sha，候选数有界）；仍不唯一 → 空串（调用方记 ``source_unknown=1``）。

        @param raw 目录史里写的来源目录名。
        @param winner_label 胜出候选标签。
        @param source_sha256 目录史记下的内核摘要。
        @return 解析到的 run id，或空串。
        """
        raw = str(raw or "").strip()
        if not raw:
            return ""
        if "/" in raw:
            row = self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (raw,)).fetchone()
            if row is not None:
                return raw
        suffix = "/" + raw
        rows = self.conn.execute(
            "SELECT run_id FROM runs WHERE substr(run_id, -?) = ?",
            (len(suffix), suffix),
        ).fetchall()
        candidates = [str(row[0]) for row in rows]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return ""
        labelled = candidates
        if winner_label:
            marked = [
                candidate for candidate in candidates
                if self.conn.execute(
                    "SELECT 1 FROM candidates WHERE run_id = ? AND label = ? LIMIT 1",
                    (candidate, winner_label),
                ).fetchone() is not None
            ]
            if marked:
                labelled = marked
        if len(labelled) == 1:
            return labelled[0]
        if source_sha256:
            # 候选内核摘要已经是索引里的窄列（``candidates.kernel_sha256``），不必再逐份读
            # ``candidates/*/kernel.*`` 重算 sha：解析因此是纯索引操作，挂在 tick 上也不打开 run 目录。
            matched = []
            for candidate in labelled:
                arguments: tuple = (candidate, source_sha256)
                clause = "run_id = ? AND kernel_sha256 = ?"
                if winner_label:
                    clause += " AND label = ?"
                    arguments = (*arguments, winner_label)
                if self.conn.execute(
                    f"SELECT 1 FROM candidates WHERE {clause} LIMIT 1", arguments,
                ).fetchone() is not None:
                    matched.append(candidate)
            unique = sorted(set(matched))
            if len(unique) == 1:
                return unique[0]
        return ""

    _KERNEL_HISTORY_COLUMNS = (
        "target_id", "task", "dtype", "size", "seq", "run_id_raw", "run_id", "winner_label",
        "source_sha256", "confirmed_speedup", "confirmed_values_json", "candidate_us",
        "baseline_us_json", "task_fingerprint", "calibration_sha256", "updated_at", "source_unknown",
    )

    def _sync_kernels(self) -> None:
        """@brief 把 ``kernels/<target>/<task>/<dtype>/<size>/history.jsonl`` 投影进 ``kernel_history``。

        目录史是发布确认时**追加**写的，``updated_at`` 是确认时刻（不是 run 开始时间），发布丢失时也不会
        消失。失效条件由两段组成：目录史文件的 ``(mtime_ns,size)`` 指纹，**加上** 索引的 ``revision``
        （来源 run 的解析依赖 ``runs``/``candidates`` 两张表）。只按文件指纹短路会让“catalog 先到、
        run 后到”的行永远停在 ``source_unknown=1``；把 runs 代际纳入后，下一次 tick 会重解析。
        解析仍然不读 run 目录：候选 sha 取自 ``candidates.kernel_sha256``。

        重建先在内存里算出完整投影，与旧表逐行比较；内容真的变了才 DELETE/INSERT 并抬高
        ``catalog_revision``，所以“runs 变了但目录史行没变”不会让发布史面板的戳无谓失效。

        挂在 10 秒全扫分支上，所以 ``kernels/`` 不存在时只做一次 glob 返回空（CI），不增加
        ``json_reads``。

        @return None
        """
        from taic.kernel_catalog import catalog_root

        with _WRITE:
            root = catalog_root()
            files = []
            if root.is_dir():
                try:
                    files = sorted(
                        (path for path in root.glob("*/*/*/*/history.jsonl") if path.is_file()),
                        key=lambda path: path.as_posix(),
                    )
                except OSError:
                    files = []
            fingerprint_parts = []
            for path in files:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                fingerprint_parts.append(f"{path.relative_to(root).as_posix()}:{stat.st_mtime_ns}:{stat.st_size}")
            key = "\x1f".join([*fingerprint_parts, f"runs:{self.revision()}"])
            row = self.conn.execute("SELECT value FROM meta WHERE key = 'kernel_history_fp_v2'").fetchone()
            if row is not None and row[0] == key:
                self._register_blob_refs()
                self.conn.commit()
                return
            pending: list[tuple] = []
            for path in files:
                relative = path.parent.relative_to(root)
                parts = relative.parts
                if len(parts) != 4:
                    continue
                directory_target, directory_task, directory_dtype, directory_size = parts
                seq = 0
                try:
                    handle = path.open("r", encoding="utf-8", errors="replace")
                except OSError:
                    continue
                with handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(record, dict):
                            continue
                        speed = _num(record.get("confirmed_speedup"))
                        values = record.get("confirmed_values")
                        baselines = record.get("baseline_us")
                        winner = str(record.get("winner_label") or "")
                        sha = str(record.get("source_sha256") or "")
                        run_id = self._resolve_history_run(record.get("run_id"), winner, sha)
                        pending.append((
                            str(record.get("target_id") or directory_target),
                            str(record.get("task") or directory_task),
                            str(record.get("dtype") or directory_dtype),
                            str(record.get("size") or directory_size),
                            seq,
                            str(record.get("run_id") or ""),
                            run_id,
                            winner,
                            sha,
                            speed,
                            json.dumps(values, ensure_ascii=False) if values is not None else None,
                            _num(record.get("candidate_us")),
                            json.dumps(baselines, ensure_ascii=False) if baselines is not None else None,
                            str(record.get("task_fingerprint") or ""),
                            str(record.get("calibration_sha256") or ""),
                            _num(record.get("updated_at")),
                            0 if run_id else 1,
                        ))
                        seq += 1
            columns = ", ".join(self._KERNEL_HISTORY_COLUMNS)
            existing = [
                tuple(row) for row in self.conn.execute(
                    f"SELECT {columns} FROM kernel_history"
                    " ORDER BY target_id, task, dtype, size, seq"
                )
            ]
            changed = pending != existing
            if changed:
                self.conn.execute("DELETE FROM kernel_history")
                if pending:
                    placeholders = ",".join("?" for _ in self._KERNEL_HISTORY_COLUMNS)
                    self.conn.executemany(
                        f"INSERT INTO kernel_history ({columns}) VALUES ({placeholders})", pending,
                    )
                self._bump_catalog()
            self._register_blob_refs(rebuild=changed)
            self._meta_set("kernel_history_fp_v2", key)
            self.conn.commit()

    def _register_blob_refs(self, *, rebuild: bool = False) -> None:
        """@brief 把目录史里已在盘上的 blob 登记进 ``blob_refs``（幂等，不读 run 目录）。

        blob 的写点在确认写路径（``taic.kernel_catalog.archive_confirmed``）；这里只按摘要补登记，
        供 ``taic blobs``/导出定位。目录史行被删时对应的 ref 也一起清掉，避免悬空引用。
        没有变化时只做一次计数比对，不重复写表（发布史/巡检每次请求都会走到这里）。

        @param rebuild 目录史刚整表重建，必须清掉旧登记再补。
        @return None
        """
        from taic.kernel_catalog import blob_path

        if not self._has_table("blob_refs"):
            return
        rows = [
            dict(row) for row in self.conn.execute(
                "SELECT target_id, task, dtype, size, seq, source_sha256 FROM kernel_history"
            )
        ]
        if not rebuild:
            expected = sum(1 for row in rows if self._blob_on_disk(blob_path, row["source_sha256"]))
            have = int(self.conn.execute(
                "SELECT COUNT(*) FROM blob_refs WHERE ref_kind = 'kernel_history'"
            ).fetchone()[0] or 0)
            if have == expected:
                return
        # 先清本 kind 的旧登记：目录史是整表重建，ref 也必须跟着重建，否则删除的行会留下悬空引用。
        self.conn.execute("DELETE FROM blob_refs WHERE ref_kind = 'kernel_history'")
        for row in rows:
            sha = str(row["source_sha256"] or "")
            if not sha:
                continue
            try:
                mtime = blob_path(sha).stat().st_mtime
            except OSError:
                continue
            ref_id = f"{row['target_id']}/{row['task']}/{row['dtype']}/{row['size']}#{row['seq']}"
            self.conn.execute(
                "INSERT OR IGNORE INTO blob_refs (sha256, ref_kind, ref_id, created_at)"
                " VALUES (?, 'kernel_history', ?, ?)",
                (sha, ref_id, mtime),
            )

    @staticmethod
    def _blob_on_disk(blob_path, sha256) -> bool:
        """@brief 摘要对应的 blob 是否已在盘上（只 stat）。

        @param blob_path ``taic.kernel_catalog.blob_path``。
        @param sha256 目录史记下的源码摘要。
        @return 存在时为 True。
        """
        sha = str(sha256 or "")
        if not sha:
            return False
        try:
            return blob_path(sha).is_file()
        except OSError:
            return False

    def ensure_kernel_history(self) -> None:
        """@brief ``_sync_kernels`` 的公开别名（巡检读取前调用，行为一致）。

        @return None
        """
        self._sync_kernels()

    def _ledger_row(self, run_id: str, row: dict, spent_max: float | None, source: str, binding: str) -> None:
        """@brief 维护花费账本行；推断或漂移的运行从账本里删除。

        @param run_id 运行 id。
        @param row 历史行。
        @param spent_max 历史最大花费。
        @param source 来源（manifest / inferred / single）。
        @param binding 绑定状态。
        """
        if source == "inferred" or binding == "drift":
            # 漂移只影响可比性，不抹去已经发生的实际支出；否则总账会下降。
            return
        if spent_max is None:
            return
        self.conn.execute(
            """
            INSERT INTO ledger (run_id, operator, month, started_at, spent_max_usd, budget_usd)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                operator = excluded.operator,
                spent_max_usd = MAX(COALESCE(ledger.spent_max_usd, excluded.spent_max_usd), COALESCE(excluded.spent_max_usd, ledger.spent_max_usd)),
                budget_usd = COALESCE(excluded.budget_usd, ledger.budget_usd),
                started_at = COALESCE(ledger.started_at, excluded.started_at),
                month = CASE WHEN ledger.started_at IS NULL THEN excluded.month ELSE ledger.month END
            """,
            (
                run_id,
                operator_name(row.get("task")),
                month_of(_num(row.get("started_at"))),
                _num(row.get("started_at")),
                spent_max,
                _num(row.get("disk_budget_usd")),
            ),
        )

    def _identity(self, directory: Path, config: dict | None, manifest: tuple[Path, dict] | None) -> tuple:
        """@brief 判定该运行属于哪个实验/arm/trial，以及来源与绑定状态。

        @param directory 运行目录。
        @param config 解析后的 ``run_config.json``，可为 None。
        @param manifest 找到的实验清单，可为 None。
        @return 七元组（source, experiment_id, arm_id, trial, kind, delta, binding）。
        """
        if manifest is not None and self._normalized(directory) != self._normalized(manifest[0]):
            root, doc = manifest
            current = self._normalized(directory)
            manifest_root = self._normalized(root)
            try:
                rel = os.path.relpath(current, manifest_root).replace(os.sep, "/")
            except ValueError:
                rel = directory.name
            arm = None
            for item in doc.get("arms") or []:
                if rel in (item.get("runs") or []):
                    arm = item
                    break
            experiment_id = doc.get("experiment_id") or self._rel(root)
            if arm is None:
                return ("manifest", experiment_id, "", None, doc.get("kind"), {}, "drift")
            delta = arm.get("delta") or {}
            binding, fingerprint = self._binding(config, doc.get("frozen") or {}, delta, experiment_id, arm.get("arm_id"))
            trial = _trial_of(rel)
            return ("manifest", experiment_id, arm.get("arm_id") or "", trial, doc.get("kind"), delta, binding)
        guessed = legacy.infer(directory, detail.runs_root(), resolved_root=self._root_path()[0])
        if guessed is not None:
            return ("inferred", guessed["experiment_id"], guessed["arm_id"], guessed["trial"], guessed["kind"], {}, "none")
        return ("single", self._rel(directory), "", None, "single", {}, "none")

    def _binding(self, config: dict | None, frozen: dict, delta: dict, experiment_id: str, arm_id: str) -> tuple[str, str | None]:
        """@brief 把运行配置与实验冻结条件、arm delta 逐项比对，得出是否漂移。

        @param config 解析后的 ``run_config.json``，可为 None。
        @param frozen 实验的冻结条件。
        @param delta 该 arm 允许的差异。
        @param experiment_id 实验 id。
        @param arm_id arm id。
        @return 二元组：绑定状态（``verified`` / ``drift`` / ``none``）与任务指纹。
        """
        if not config:
            return "none", None
        search = config.get("search") or {}
        for key, expected in delta.items():
            if not _eq(_lookup(config, search, key), expected):
                return "drift", None
        for key in ("dtype", "size"):
            if key in delta or frozen.get(key) in (None, ""):
                continue
            if not _eq(config.get(key), frozen.get(key)):
                return "drift", None
        tasks = frozen.get("tasks")
        task = config.get("task")
        if isinstance(tasks, list) and tasks and task not in tasks and "task" not in delta:
            return "drift", None
        if isinstance(frozen.get("task"), str) and frozen.get("task") and "task" not in delta and task != frozen.get("task"):
            return "drift", None
        budget = frozen.get("budget_usd")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and "budget_usd" not in delta:
            got = _num(search.get("budget_usd"))
            if got is not None and abs(got - float(budget)) > 1e-6:
                return "drift", None
        row = self.conn.execute(
            "SELECT task_fingerprint FROM arms WHERE experiment_id = ? AND arm_id = ?",
            (experiment_id, arm_id),
        ).fetchone()
        expected_fp = row["task_fingerprint"] if row and row["task_fingerprint"] else ""
        got_fp = config.get("task_fingerprint") if isinstance(config.get("task_fingerprint"), str) else ""
        if expected_fp and got_fp and expected_fp != got_fp:
            return "drift", expected_fp
        return "verified", got_fp or expected_fp or None

    def _symlink_free(self, text: str) -> bool:
        """@brief 路径（含祖先）在 ``runs/`` 内是否没有任何符号链接；按目录缓存。

        普通目录不必 ``realpath``：父目录已确认无链接、自己也不是链接，canonical 就等于词法路径。
        这样冷建每场只多一次 ``lstat``，而不是一次逐级 ``realpath``。

        @param text 已规整的绝对路径文本。
        @return 路径与祖先都不是符号链接时为 True。
        """
        cache = getattr(self, "_linkfree_cache", None)
        if cache is None:
            cache = self._linkfree_cache = {}
        cached = cache.get(text)
        if cached is not None:
            return cached
        root = self._root_text()
        if text == root:
            cache[text] = True
            return True
        parent = os.path.dirname(text)
        if not (text == root or text.startswith(root + os.sep)):
            cache[text] = False
            return False
        parent_ok = True if parent == root else self._symlink_free(parent)
        result = parent_ok and not os.path.islink(text)
        if len(cache) < 200_000:
            cache[text] = result
        return result

    def _resolved_text_from_text(self, text: str) -> str:
        """@brief 对已规整的绝对文本做 ``realpath``，按目录缓存。

        @param text 已规整的绝对路径文本。
        @return canonical 文本。
        """
        cache = getattr(self, "_resolved_cache", None)
        if cache is None:
            cache = self._resolved_cache = {}
        cached = cache.get(text)
        if cached is not None:
            return cached
        if self._symlink_free(text):
            resolved = text
        else:
            try:
                resolved = os.path.normpath(os.path.realpath(text))
            except OSError:
                resolved = text
        if len(cache) < 200_000:
            cache[text] = resolved
        return resolved

    def _resolved_text(self, path: Path) -> str:
        """@brief 把路径解析成真实路径文本（跟随符号链接），按目录缓存。

        旧口径是 ``path.resolve()``：根内别名必须给出 canonical run id，指向根外的祖先链接必须被拒绝。
        普通目录没有符号链接时直接返回词法路径（只多一次 ``lstat``）；有链接才 ``realpath``。
        结果按目录缓存，``_prepare_frame`` 清空缓存。

        @param path 待解析路径。
        @return ``realpath`` 后的 ``normpath`` 文本。
        """
        return self._resolved_text_from_text(self._normalized(path))

    def _within_runs(self, path: Path) -> bool:
        """@brief 判断路径（跟随符号链接后）是否落在 ``runs/`` 之内。

        祖先符号链接指向根外时，词法前缀看起来仍在根内，必须按解析后的路径判断；指向根内的别名则
        仍然允许。解析结果按目录缓存，普通热路径每个目录只做一次。

        @param path 待判断路径。
        @return 在 ``runs/`` 之内时为 True。
        """
        return self._relative_text(self._resolved_text(path)) is not None

    def _manifest_fingerprint(self, directory: Path) -> str:
        """@brief 向上查找并取最近的 ``experiment.json`` 指纹。

        @param directory 起始目录。
        @return 指纹字符串；找不到时为空串。
        """
        return _manifest_fingerprint_at(Path(self._resolved_text(directory)), self._root_path()[1])

    def _write_liveness(self, row, status: str) -> None:
        """@brief 只改一行运行行的存活状态与预算告警，不重读磁盘。

        @param row 索引中的运行行（含 ``row_json``）。
        @param status 新的存活状态。
        """
        try:
            payload = json.loads(row["row_json"] or "{}")
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload["status"] = status
        budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else {}
        spent = budget.get("spent", payload.get("spent_usd"))
        cap = budget.get("budget", payload.get("budget_usd"))
        payload["budget"] = budget_status(spent, cap, payload.get("stop_reason") or "", status == "running")
        self.conn.execute(
            "UPDATE runs SET status = ?, row_json = ? WHERE run_id = ?",
            (status, json.dumps(payload, default=str), row["run_id"]),
        )

    def _root_path(self) -> tuple[Path, str]:
        """@brief 取（只解析一次的）``runs/`` 根：解析后的 Path 与规整后的字符串。

        ``Path.resolve()`` 在 macOS 上要逐级 ``lstat``（实测约 10 次/调用）。热对账每帧要判上千次
        路径归属，所以每帧只解析一次根，之后都用字符串前缀比较。根路径的原始值进缓存键：
        测试或别处改了 ``AI_LOWERLING_RUNS_DIR`` 后自动重解析。

        @return ``(resolved Path, normpath 文本)``。
        """
        raw = os.fspath(detail.runs_root())
        cached = getattr(self, "_root_cache", None)
        if cached is not None and cached[0] == raw:
            # 缓存命中也要把文本缓存补上：否则每帧 `_root_text` 会退回 `_root_path` 再查一次环境变量。
            self._root_text_cache = cached[2]
            return cached[1], cached[2]
        resolved = Path(raw).resolve()
        text = os.path.normpath(os.fspath(resolved))
        self._root_cache = (raw, resolved, text)
        self._root_text_cache = text
        self.stats["root_resolves"] = int(self.stats.get("root_resolves", 0)) + 1
        return resolved, text

    def _root_text(self) -> str:
        """@brief 取缓存的 ``runs/`` 根文本；每帧不重复查环境变量。

        @return 规整后的根路径文本。
        """
        text = getattr(self, "_root_text_cache", None)
        if text is None:
            text = self._root_path()[1]
        return text

    def _normalized(self, path: Path) -> str:
        """@brief 把路径规整成绝对字符串，不碰文件系统（无 ``realpath``）。

        @param path 绝对或相对 ``runs/`` 的路径。
        @return ``normpath`` 后的文本。
        """
        text = os.fspath(path)
        if not os.path.isabs(text):
            text = os.path.join(self._root_text(), text)
        return os.path.normpath(text)

    def _relative_text(self, text: str) -> str | None:
        """@brief 字符串前缀剥离：落在 ``runs/`` 内时给 POSIX 相对路径。

        @param text 已规整的绝对路径文本。
        @return 相对路径（根自身为 ``"."``）；不在根内时为 None。
        """
        root = self._root_text()
        if text == root:
            return "."
        prefix = root + os.sep
        if not text.startswith(prefix):
            return None
        return text[len(prefix):].replace(os.sep, "/")

    def _rel(self, path: Path) -> str:
        """@brief 取目录相对 ``runs/`` 根目录的 POSIX 路径（跟随符号链接）。

        根内别名（例如 ``link-in -> inside``）必须给出 canonical id ``inside``；根外路径退回词法
        目录名（与旧 ``path.resolve()`` 口径一致）。解析按目录缓存。

        @param path 目录路径。
        @return 相对路径；不在根目录下时退回目录名，根目录自身为 ``"."``。
        """
        text = os.fspath(path)
        if not os.path.isabs(text):
            text = os.path.join(self._root_text(), text)
        text = os.path.normpath(text)
        resolved = self._resolved_text_from_text(text)
        root = self._root_text()
        if resolved == root:
            return "."
        prefix = root + os.sep
        if resolved.startswith(prefix):
            return resolved[len(prefix):].replace(os.sep, "/")
        return Path(text).name

    def total_usd(self) -> float:
        """@brief 返回含结转的总花费。

        有结转时只统计结转之后开始的运行，避免重复计数。

        @return 总花费美元数（保留 6 位）。
        """
        carry = self.conn.execute("SELECT created_at, total_usd FROM ledger_carryover LIMIT 1").fetchone()
        if carry is None:
            row = self.conn.execute("SELECT COALESCE(SUM(spent_max_usd), 0) FROM ledger").fetchone()
            return round(float(row[0]), 6)
        extra = self.conn.execute(
            "SELECT COALESCE(SUM(spent_max_usd), 0) FROM ledger WHERE started_at >= ?",
            (carry["created_at"],),
        ).fetchone()
        return round(float(carry["total_usd"] or 0) + float(extra[0] or 0), 6)

    def drop_archived(self, run_ids: list[str], experiment_ids: list[str]) -> None:
        """@brief 忘掉已归档的搜索；账本行保留，因此删除不会降低已记录的花费。

        原始英文说明：Forget archived searches. The ledger rows stay, so a deletion cannot lower the recorded spend.

        @param run_ids 要删除的 run id 列表。
        @param experiment_ids 要删除的 experiment id 列表。
        """
        with _WRITE:
            wanted = set(run_ids)
            for experiment_id in experiment_ids:
                for row in self.conn.execute("SELECT run_id FROM runs WHERE experiment_id = ? OR run_id = ?", (experiment_id, experiment_id)):
                    wanted.add(row["run_id"])
                prefix = experiment_id + "/"
                for row in self.conn.execute("SELECT run_id FROM runs"):
                    if str(row["run_id"]).startswith(prefix):
                        wanted.add(row["run_id"])
            from observe.feedback import retire_run_claims

            for run_id in wanted:
                retire_run_claims(self.conn, run_id)
                previous = self.conn.execute(
                    "SELECT status, task, dtype, size, target_id, protocol_version, started_at, spent_max_usd,"
                    " publication, published_speedup FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if previous is not None:
                    self._rollup_remove_row(dict(previous), run_id)
                self.conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM candidates WHERE run_id = ?", (run_id,))
                if self._has_table("confirmed_best"):
                    self.conn.execute("DELETE FROM confirmed_best WHERE run_id = ?", (run_id,))
                if self._has_table("run_pending_eval"):
                    self.conn.execute("DELETE FROM run_pending_eval WHERE run_id = ?", (run_id,))
                if self._has_table("failure_members"):
                    self.conn.execute("DELETE FROM failure_members WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM turns WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM feedback_items WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM feedback_claims WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM feedback_subclaims WHERE run_id = ?", (run_id,))
                self._clear_sessions_projection(run_id)
            watched = [row["path"] for row in self.conn.execute("SELECT path FROM watched")]
            for experiment_id in experiment_ids:
                self.conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
                self.conn.execute("DELETE FROM arms WHERE experiment_id = ?", (experiment_id,))
            prefixes = [item + "/" for item in list(wanted) + list(experiment_ids)]
            exact = wanted | set(experiment_ids)
            for path in watched:
                if path in exact or any(str(path).startswith(prefix) for prefix in prefixes):
                    self.conn.execute("DELETE FROM watched WHERE path = ?", (path,))
            self._bump()
            self.conn.commit()

    def forget_moved(self, run_id: str) -> None:
        """@brief 目录被改名后，删掉旧 run_id 的索引行，包括账本行。

        A move is not a deletion: the spend is carried by the new path, so keeping the old ledger
        row would count it twice. ``drop_archived`` keeps the row because there the run really is
        gone.

        @param run_id 改名前的 run id（`runs/` 下的相对路径）。
        """
        with _WRITE:
            from observe.feedback import retire_run_claims

            retire_run_claims(self.conn, run_id)
            previous = self.conn.execute(
                "SELECT status, task, dtype, size, target_id, protocol_version, started_at, spent_max_usd,"
                " publication, published_speedup FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if previous is not None:
                self._rollup_remove_row(dict(previous), run_id)
            for table in ("runs", "candidates", "sessions", "turns", "feedback_items", "feedback_claims",
                          "feedback_subclaims", "ledger", "confirmed_best", "run_pending_eval", "failure_members"):
                if self._has_table(table):
                    self.conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            self._clear_sessions_projection(run_id)
            self.conn.execute("DELETE FROM watched WHERE path = ?", (run_id,))
            self._bump()
            self.conn.commit()

    def refresh_feedback(self, run_id: str, directory: Path) -> None:
        """@brief 立即重读一个运行的反馈文件；已结束的运行不会再被空闲帧扫描。

        原始英文说明：Re-read one run's feedback files now. A finished run is not scanned again by an idle tick.

        @param run_id 运行 id。
        @param directory 运行目录。
        """
        with _WRITE:
            self._ingest(Path(directory), force=True)
            self.conn.commit()

    def resolve(self, run_id: str) -> Path | None:
        """@brief 把 run id 解析为运行目录，并确保该运行已进索引。

        @param run_id 运行 id。
        @return 运行目录；目录不存在或仍写不进索引时返回 None。
        """
        path = detail.resolve_run(run_id, root=self._root_path()[0])
        if path is None:
            return None
        # 存在性检查必须在写锁内：它读的是共享写连接。放到锁外时，另一个请求线程的写事务
        # 正在改这张表/这条连接，SELECT 可能瞬时看不到刚写好的行（并发 review 的回归
        # test_two_reviews_of_one_run_both_land 因此偶发 404 no such run）。
        with _WRITE:
            self._ingest(path)
            self.conn.commit()
            found = self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if found is None:
            return None
        return path


def _trial_of(relative: str) -> int | None:
    """@brief 从目录名 ``trial_<n>`` 里取 trial 序号。

    @param relative 相对路径。
    @return trial 序号；命名不匹配或无法解析时返回 None。
    """
    name = Path(relative).name
    if name.startswith("trial_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def _sha256_file(path: Path) -> str | None:
    """@brief 计算文件的 sha256。

    @param path 文件路径。
    @return 十六进制摘要；读不了时返回 None。
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _confirm_target(fp_confirm: str) -> tuple[str, str]:
    """@brief 从确认指纹里取获胜候选标签与其内核相对路径。

    @param fp_confirm 确认部分指纹（以 ``\x1f`` 分隔）。
    @return 二元组（标签, 相对路径）；字段不足时两者为空串。
    """
    parts = (fp_confirm or "").split("\x1f")
    if len(parts) >= 4:
        return parts[2], parts[3]
    return "", ""


def _winner_rel(directory: Path, label: str) -> str:
    """@brief 找出唯一匹配该标签的候选内核相对路径。

    @param directory 运行目录。
    @param label 获胜候选标签。
    @return 相对路径；没有标签、目录不存在或匹配不唯一时为空串。
    """
    if not label:
        return ""
    root = directory / "candidates"
    if not root.is_dir():
        return ""
    found = [path for path in root.iterdir() if path.is_dir() and len(path.name) > 4 and path.name[3] == "_" and path.name[4:] == label]
    if len(found) != 1:
        return ""
    return found[0].relative_to(directory).as_posix() + "/" + detail.target_file_names(directory)[0]


def _lookup(config: dict, search: dict, key: str):
    """@brief 按键名在配置的各个层里查值。

    @param config 运行配置。
    @param search 配置里的 ``search`` 段。
    @param key 键名（如 ``model`` / ``task`` / ``dtype``）。
    @return 对应的值；各层都没有时返回 None。
    """
    if key == "model":
        return (config.get("llm") or {}).get("model")
    if key in ("task", "dtype", "size"):
        return config.get(key)
    if key in search:
        return search.get(key)
    llm = config.get("llm") or {}
    if key in llm:
        return llm.get(key)
    return config.get(key)


def _eq(got, expected) -> bool:
    """@brief 比较两个配置值；任一侧是浮点时按 1e-6 容差比较。

    @param got 实际值。
    @param expected 期望值。
    @return 相等（或在容差内）时为 True。
    """
    if isinstance(expected, float) or isinstance(got, float):
        left, right = _num(got), _num(expected)
        if left is None or right is None:
            return got == expected
        return abs(left - right) <= 1e-6
    return got == expected
