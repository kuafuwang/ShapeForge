"""Rebuildable SQLite index of ``runs/``. One writer. Readers open the file read-only.

The database is a derivative. Deleting it and opening it again rebuilds the rows from ``runs/``,
``data/books.sqlite`` (read-only, and only to seed the carry-over) and ``ledger_carryover.json``.
A schema version this process does not know is dropped and built again. There is no migration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from observe import detail, legacy
from observe.budget import budget_status
from observe.ledger import month_of, operator_name

INDEX_SCHEMA_VERSION = 11
_DIAG_SECRET = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")


def _diagnostic(value, limit: int = 400) -> str:
    """One bounded, single-line diagnostic; response bodies are not indexed."""
    if not isinstance(value, str):
        return ""
    return _DIAG_SECRET.sub("<redacted>", " ".join(value.split()))[:limit]
_SKIP_DIRS = {"candidates", "__pycache__"}
_WRITE = threading.Lock()
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


@contextmanager
def borrow():
    """@brief 借出唯一的写连接，让请求线程能读它。

    原始英文说明：The one writer connection, held so a request thread can read it.

    @return 上下文管理器，产出 ``Index`` 实例。
    """
    index = get_index()
    with _WRITE:
        yield index


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
        self._load_carryover()
        self._load_preserved_ledger()

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
        """
        self._preserve_ledger()
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix) if suffix else self.path
            try:
                candidate.unlink()
            except OSError:
                pass

    def _preserve_ledger(self) -> None:
        """@brief 在删文件前把花费行复制到旁路文件；尽力而为，读不了的库没有行可存。

        原始英文说明：Copy the spend rows out before the file is unlinked. Best effort: a database that cannot
        be read has no rows to save.
        """
        try:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5)
            try:
                rows = [list(row) for row in conn.execute(
                    "SELECT run_id, operator, month, started_at, spent_max_usd, budget_usd FROM ledger"
                ).fetchall()]
            finally:
                conn.close()
        except sqlite3.Error:
            return
        if not rows:
            # This database has no spend rows. Any sidecar still around describes an older state
            # and must not be replayed into the next build: the ledger was deliberately emptied.
            self._drop_preserved(preserved_ledger_path(self.path))
            return
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
            return

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
                within_noise INTEGER NOT NULL DEFAULT 0,
                eligible INTEGER NOT NULL DEFAULT 0,
                kernel_sha256 TEXT,
                protocol_version TEXT,
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
                committee TEXT,
                provider TEXT,
                model TEXT,
                outcome TEXT,
                cost_usd REAL,
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
            """
        )
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
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('built_at', ?) ON CONFLICT(key) DO NOTHING",
            (str(time.time()),),
        )
        self.conn.commit()

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

    def _bump(self) -> None:
        """@brief 把详情 revision 加一。
        """
        self.conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'"
        )

    def turns_revision(self) -> int:
        """@brief 取轮次 revision。

        @return meta 里的 ``turns_revision``，缺失时为 0。
        """
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'turns_revision'").fetchone()
        return int(row[0] if row else 0)

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
            now = time.monotonic()
            if now - self._last_full >= self._full_interval():
                self._refresh_watched()
                self._last_full = now
            self._refresh_running()
            self.conn.commit()
            return self.revision(), list(self.changed)

    def sync_known(self) -> None:
        """@brief 发现已知运行的改动与新子目录；空闲帧用 :meth:`tick`，不调它。

        原始英文说明：See edits and new child directories. The idle frame uses :meth:`tick` and does not call this.
        """
        with _WRITE:
            self._prepare_frame()
            root = detail.runs_root()
            watched = list(self.conn.execute("SELECT path, mtime_ns FROM watched"))
            if root.is_dir() and not watched:
                self._note_dir(root, force=True)
            for item in watched:
                path = root if item["path"] in ("", ".") else root / item["path"]
                try:
                    mtime = path.stat().st_mtime_ns
                except OSError:
                    continue
                if mtime != item["mtime_ns"]:
                    self._note_dir(path, force=True)
            for row in self.conn.execute("SELECT run_id FROM runs").fetchall():
                path = detail.resolve_run(row["run_id"])
                if path is not None:
                    self._ingest(path)
                    continue
                # 目录已经不在了。留着一行 status='done' 会让它的花费和公布成绩永远被
                # runs_page / best_confirmed / 预算统计算进去；migrate 之后还会出现同名两行。
                cur = self.conn.execute(
                    "UPDATE runs SET status = 'missing', publication = NULL, published_speedup = NULL "
                    "WHERE run_id = ? AND status != 'missing'",
                    (row["run_id"],),
                )
                if cur.rowcount:
                    self._bump()
            self._last_full = time.monotonic()
            self.conn.commit()

    def _prepare_frame(self) -> None:
        """@brief 重置本帧统计与变更列表；``runs_root`` 变了就清掉全部派生表。
        """
        self.stats = {"json_reads": 0, "dir_lists": 0, "turn_parses": 0}
        self.changed = []
        root = str(detail.runs_root().resolve())
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'runs_root'").fetchone()
        if row is not None and row[0] != root:
            self.conn.executescript(
                "DELETE FROM runs; DELETE FROM candidates; DELETE FROM sessions; DELETE FROM experiments; DELETE FROM arms; DELETE FROM ledger; DELETE FROM watched; DELETE FROM feedback_items; DELETE FROM feedback_claims; DELETE FROM feedback_subclaims;"
            )
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
            self._ingest(directory)
        self.stats["dir_lists"] += 1
        self.conn.execute(
            "INSERT INTO watched (path, mtime_ns) VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET mtime_ns = excluded.mtime_ns",
            (rel, mtime),
        )
        try:
            children = list(directory.iterdir())
        except OSError:
            return
        for child in children:
            if not child.is_dir() or child.name in _SKIP_DIRS or child.name.startswith("."):
                continue
            if not self._within_runs(child):
                continue
            self._consider(child)

    def _consider(self, directory: Path) -> None:
        """@brief 子目录看起来是运行或实验时摄取它，然后继续往下看。

        @param directory 子目录路径。
        """
        if (directory / "state.json").exists() or (directory / "result.json").exists() or (directory / "experiment.json").exists():
            self._ingest(directory)
        self._note_dir(directory, force=False)

    def _refresh_watched(self) -> None:
        """@brief 发现新子目录或续跑：建嵌套目录只会改父目录的 mtime，不会改 ``runs/``。

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

    def _refresh_running(self) -> None:
        """@brief 逐行复查仍在跑的运行：指纹没变只补轮次与存活状态，变了就重新摄取。
        """
        for row in self.conn.execute("SELECT run_id, updated_at, pid, phase, status, row_json FROM runs WHERE status = 'running'").fetchall():
            path = detail.resolve_run(row["run_id"])
            if path is None:
                self.conn.execute(
                    "UPDATE runs SET status = 'missing', publication = NULL, published_speedup = NULL WHERE run_id = ?",
                    (row["run_id"],),
                )
                self.changed.append(row["run_id"])
                self._bump()
                continue
            stored = self._stored_fingerprints(path)
            label, rel = _confirm_target(stored[3])
            if self._fingerprints(path, label, rel) == stored:
                # transcript.jsonl is not part of the detail fingerprint. A new line still has to land in the turns table,
                # and that must not redraw the candidate list.
                if self._catch_up_turns(path):
                    self._bump_turns()
                state = {"updated_at": row["updated_at"], "pid": row["pid"], "phase": row["phase"]}
                status = detail.liveness(state, None)
                if status != row["status"]:
                    self._write_liveness(row, status)
                    self.changed.append(row["run_id"])
                    self._bump()
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

    def _fingerprints(self, directory: Path, winner_label: str = "", winner_rel: str = "") -> tuple[str, str, str, str, str]:
        """@brief 现算一个运行的五个指纹（state、result、config、确认、其余杂项）。

        @param directory 运行目录。
        @param winner_label 获胜候选标签，可为空串。
        @param winner_rel 获胜内核的相对路径，可为空串。
        @return 五元组指纹，用于判断是否需要重新摄取。
        """
        confirm = "\x1f".join((
            _fingerprint(directory / "confirm_report.json"),
            _fingerprint(directory / detail.target_file_names(directory)[1]),
            winner_label,
            winner_rel,
            _fingerprint(directory / winner_rel) if winner_rel else "",
        ))
        extra = "|".join(
            _fingerprint(directory / name) for name in ("budget_events.json", "ledger.json", "resume.lock", "feedback.jsonl", "feedback_review.json")
        )
        from taic.agent_feedback import PARSER_VERSION

        extra += "|" + _fingerprint(directory / "candidates")
        extra += "|" + self._manifest_fingerprint(directory)
        extra += "|" + PARSER_VERSION
        return (
            _fingerprint(directory / "state.json"),
            _fingerprint(directory / "result.json"),
            _fingerprint(directory / "run_config.json"),
            confirm,
            extra,
        )

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
            self.conn.execute(
                "INSERT INTO turns (run_id, seq, label, kind, row_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, seq, projected.get("label") or "", projected.get("kind") or "", json.dumps(projected, ensure_ascii=False)),
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

    def ensure_turns(self, directory: Path) -> None:
        """@brief 按需投影该运行；没有新字节时第二次调用不会再解析文件。

        原始英文说明：Project this run on demand. A second call with no new bytes does not parse the file again.

        @param directory 运行目录。
        """
        with _WRITE:
            if self._catch_up_turns(directory):
                self._bump_turns()
            self.conn.commit()

    def _ingest(self, directory: Path, *, force: bool = False) -> None:
        """@brief 把一个目录（运行或队列）重新读进索引。

        指纹未变且已收录时只补轮次；否则重读快照、清单、候选、会话与反馈。

        @param directory 待摄取目录。
        @param force 为 True 时忽略指纹强制重读。
        """
        if not self._within_runs(directory):
            return
        stored = self._stored_fingerprints(directory)
        winner_label, winner_rel = _confirm_target(stored[3])
        fingerprints = self._fingerprints(directory, winner_label, winner_rel)
        known = self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (self._rel(directory),)).fetchone()
        tracked = "\x1f" in (stored[3] or "")
        if not force and fingerprints == stored and known and (tracked or not (directory / detail.target_file_names(directory)[1]).exists()):
            self._project_turns(directory)
            return
        state = self._read_json(directory / "state.json")
        result_path = directory / "result.json"
        result = self._read_json(result_path) if result_path.exists() else None
        broken = result_path.exists() and result is None
        label = (result or {}).get("confirm_winner_label") if isinstance(result, dict) else ""
        if not isinstance(label, str):
            label = ""
        rel = _winner_rel(directory, label)
        fingerprints = self._fingerprints(directory, label, rel)
        manifest = self._find_manifest(directory)
        if manifest is not None:
            self._upsert_experiment(manifest[0], manifest[1])
        if state and state.get("script") and result is None and not broken:
            self._ingest_queue(directory, state)
            return
        if manifest is not None and directory.resolve() == manifest[0].resolve() and not (result or (state and state.get("task") and not state.get("script"))):
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
        config = self._read_json(directory / "run_config.json")
        self._upsert_run(directory, row, config, fingerprints, manifest)
        self._upsert_candidates(directory, self._rel(directory), row, result, state)
        self._upsert_sessions(directory, self._rel(directory), config)
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
        @return 二元组（实验根目录, 清单字典）；找到 ``runs/`` 根仍未找到时返回 None。
        """
        root = detail.runs_root().resolve()
        current = directory.resolve()
        while True:
            path = current / "experiment.json"
            if path.is_file():
                self.stats["json_reads"] += 1
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    doc = None
                if isinstance(doc, dict):
                    return current, doc
            if current == root or current.parent == current:
                return None
            current = current.parent

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

    def _upsert_candidates(self, directory: Path, run_id: str, row: dict, result: dict | None, state: dict | None) -> None:
        """@brief 每个候选目录一行；报告给出状态/阶段/加速比，轨迹给出噪声标记。

        原始英文说明：One row per candidate directory. The report supplies status/stage/speedup; the trace supplies the noise mark.

        @param directory 运行目录。
        @param run_id 运行 id。
        @param row 历史行。
        @param result 解析后的 ``result.json``，可为 None。
        @param state 解析后的 ``state.json``，可为 None。
        """
        self.conn.execute("DELETE FROM candidates WHERE run_id = ?", (run_id,))
        root = directory / "candidates"
        if not root.is_dir():
            return
        trace = []
        for doc in (result, state):
            if isinstance(doc, dict) and isinstance(doc.get("trace"), list):
                trace = doc["trace"]
                break
        flags = {str(item.get("label")): bool(item.get("within_noise")) for item in trace if isinstance(item, dict)}
        for path in sorted(item for item in root.iterdir() if item.is_dir()):
            name = path.name
            idx = int(name[:3]) if len(name) >= 3 and name[:3].isdigit() else -1
            label = name[4:] if len(name) > 4 and name[3] == "_" else name
            report = self._read_json(path / "report.json")
            kernel_name = detail.target_file_names(directory)[0]
            if not isinstance(report, dict) and not (path / kernel_name).is_file():
                continue
            status = str(report.get("status") or "") if isinstance(report, dict) else ""
            stage = str(report.get("stage") or "") if isinstance(report, dict) else ""
            speedup = _num(report.get("speedup")) if isinstance(report, dict) else None
            eligible = 1 if isinstance(report, dict) and report.get("eligible") is True else 0
            sha = _sha256_file(path / kernel_name)
            try:
                updated = path.stat().st_mtime
            except OSError:
                updated = None
            self.conn.execute(
                """
                INSERT OR REPLACE INTO candidates (
                    run_id, idx, label, status, stage, speedup, within_noise, eligible, kernel_sha256, protocol_version, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (run_id, idx, label, status, stage, speedup, 1 if flags.get(label) else 0, eligible, sha,
                 str(report.get("protocol_version") or "") if isinstance(report, dict) else "", updated),
            )

    def _upsert_sessions(self, directory: Path, run_id: str, config: dict | None) -> None:
        """@brief 每个 hmz 会话索引一行；过大的旧式聊天转录直接跳过。

        原始英文说明：Index one row per hmz session. Legacy chat transcripts are skipped when they are large.

        @param directory 运行目录。
        @param run_id 运行 id。
        @param config 解析后的 ``run_config.json``，可为 None。
        """
        self.conn.execute("DELETE FROM sessions WHERE run_id = ?", (run_id,))
        path = directory / "transcript.jsonl"
        if not path.is_file():
            return
        try:
            size = path.stat().st_size
        except OSError:
            return
        loop = str((config or {}).get("loop_protocol") or "")
        if size > 1_000_000 and loop != "flame_chase_1":
            return
        seq = 0
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except OSError:
            return
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
                        run_id, seq, label, committee, provider, model, outcome, cost_usd,
                        session_id, failed_turns, last_kind, failure_class, error, tool_calls, diagnostics_path,
                        tokens_input, tokens_output, tokens_cache_read, tokens_cache_write, t
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id, seq, str(event.get("label") or ""), str(event.get("committee") or ""),
                        str(event.get("provider") or ""), str(event.get("model") or ""), str(event.get("outcome") or ""),
                        _num(event.get("cost_usd")),
                        session_id, count("failed_turns"), _diagnostic(runtime.get("last_terminal_kind"), 40),
                        _diagnostic(runtime.get("failure_class"), 60),
                        _diagnostic(event.get("error") or runtime.get("last_error")),
                        count("tool_calls"), _diagnostic(event.get("diagnostics_path"), 300),
                        token("input"), token("output"), token("cache_read"), token("cache_write"), None,
                    ),
                )

    def _upsert_missing(self, directory: Path, fingerprints: tuple[str, str, str, str, str]) -> None:
        """@brief 把解析不出来的运行记为 ``missing``，并清掉它的候选与会话。

        @param directory 运行目录。
        @param fingerprints 现算的五个指纹。
        """
        run_id = self._rel(directory)
        self.conn.execute("DELETE FROM candidates WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM sessions WHERE run_id = ?", (run_id,))
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
        self.changed.append(run_id)
        self._bump()

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
        previous = self.conn.execute("SELECT spent_max_usd FROM runs WHERE run_id = ?", (run_id,)).fetchone()
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
        protocol = str((config or {}).get("protocol_version") or result_doc.get("protocol_version") or "")
        if not protocol:
            # Older runs do not carry an explicit epoch. Keep the four flame_chase_1 runs out of
            # the legacy/chat bucket; the rest are the deprecated chat protocol.
            protocol = "flame_chase_1" if str((config or {}).get("loop_protocol") or "") == "flame_chase_1" else "legacy"
        committees = json.dumps((config or {}).get("committees") or [], sort_keys=True, ensure_ascii=False)
        stored = {key: row.get(key) for key in row if key not in ("state", "result")}
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
        self.conn.execute(
            """
            INSERT INTO runs (
                run_id, experiment_id, arm_id, trial, kind, task, operator, dtype, size, target_id,
                backend, backend_version, kernel_sha256, protocol_version, committees_json, model, preset,
                switch_delta_json, contract, status, phase, stop_reason, started_at, updated_at,
                spent_usd, spent_max_usd, budget_usd, granted_budget_usd, publication, published_speedup,
                naive_speedup, rounds, evaluations, inflight_label, pid, source, binding,
                fp_state, fp_result, fp_config, fp_confirm, fp_extra, row_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                experiment_id = excluded.experiment_id, arm_id = excluded.arm_id, trial = excluded.trial,
                kind = excluded.kind, task = excluded.task, operator = excluded.operator, dtype = excluded.dtype,
                size = excluded.size, target_id = excluded.target_id, backend = excluded.backend,
                backend_version = excluded.backend_version, kernel_sha256 = excluded.kernel_sha256,
                protocol_version = excluded.protocol_version, committees_json = excluded.committees_json,
                model = excluded.model, preset = excluded.preset,
                switch_delta_json = excluded.switch_delta_json, contract = excluded.contract,
                status = excluded.status, phase = excluded.phase, stop_reason = excluded.stop_reason,
                started_at = excluded.started_at, updated_at = excluded.updated_at,
                spent_usd = excluded.spent_usd, spent_max_usd = excluded.spent_max_usd,
                budget_usd = excluded.budget_usd, granted_budget_usd = excluded.granted_budget_usd,
                publication = excluded.publication, published_speedup = excluded.published_speedup,
                naive_speedup = excluded.naive_speedup, rounds = excluded.rounds, evaluations = excluded.evaluations,
                inflight_label = excluded.inflight_label, pid = excluded.pid, source = excluded.source,
                binding = excluded.binding, fp_state = excluded.fp_state, fp_result = excluded.fp_result,
                fp_config = excluded.fp_config, fp_confirm = excluded.fp_confirm, fp_extra = excluded.fp_extra,
                row_json = excluded.row_json
            """,
            (
                run_id, experiment_id, arm_id, trial, kind, row.get("task") or "", operator_name(row.get("task")),
                row.get("dtype") or "", (config or {}).get("size") or "", _target_id(config),
                backend, backend_version, kernel_sha, protocol, committees, model or "", (config or {}).get("preset") or "",
                json.dumps(delta, sort_keys=True), "", row.get("status"), row.get("phase") or "", row.get("stop_reason") or "",
                row.get("started_at"), _num((row.get("state") or {}).get("updated_at")),
                spent, spent_max, _num(row.get("disk_budget_usd")), _num(row.get("budget_usd")),
                row.get("publication"), _num(row.get("published_speedup")), _num((row.get("result") or {}).get("naive_speedup")),
                _num((row.get("state") or {}).get("round")), _num((row.get("result") or {}).get("evaluations")),
                row.get("inflight_label") or "", _num((row.get("state") or {}).get("pid")), source, binding,
                *fingerprints, json.dumps(stored, default=str),
            ),
        )
        self._ledger_row(run_id, row, spent_max, source, binding)
        self.changed.append(run_id)
        self._bump()
        if source == "manifest" and arm_id:
            self._reanchor(experiment_id, arm_id)

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
                path = detail.resolve_run(item["run_id"])
                if path is not None:
                    self._ingest(path, force=True)
        finally:
            self._in_reanchor = False

    def _ledger_row(self, run_id: str, row: dict, spent_max: float | None, source: str, binding: str) -> None:
        """@brief 维护花费账本行；推断或漂移的运行从账本里删除。

        @param run_id 运行 id。
        @param row 历史行。
        @param spent_max 历史最大花费。
        @param source 来源（manifest / inferred / single）。
        @param binding 绑定状态。
        """
        if source == "inferred" or binding == "drift":
            self.conn.execute("DELETE FROM ledger WHERE run_id = ?", (run_id,))
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
        if manifest is not None and directory.resolve() != manifest[0].resolve():
            root, doc = manifest
            try:
                rel = directory.resolve().relative_to(root.resolve()).as_posix()
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
        guessed = legacy.infer(directory, detail.runs_root())
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

    def _within_runs(self, path: Path) -> bool:
        """@brief 判断路径是否落在 ``runs/`` 之内。

        @param path 待判断路径。
        @return 在 ``runs/`` 之内时为 True。
        """
        root = detail.runs_root().resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError:
            return False
        return True

    def _manifest_fingerprint(self, directory: Path) -> str:
        """@brief 向上查找并取最近的 ``experiment.json`` 指纹。

        @param directory 起始目录。
        @return 指纹字符串；找不到时为空串。
        """
        root = detail.runs_root().resolve()
        current = directory.resolve()
        while True:
            found = _fingerprint(current / "experiment.json")
            if found:
                return found
            if current == root or current.parent == current:
                return ""
            current = current.parent

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

    def _rel(self, path: Path) -> str:
        """@brief 取目录相对 ``runs/`` 根目录的 POSIX 路径。

        @param path 目录路径。
        @return 相对路径；不在根目录下时退回目录名，根目录自身为 ``"."``。
        """
        root = detail.runs_root().resolve()
        try:
            text = path.resolve().relative_to(root).as_posix()
        except ValueError:
            return path.name
        return text or "."

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
                self.conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM turns WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM feedback_items WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM feedback_claims WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM feedback_subclaims WHERE run_id = ?", (run_id,))
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
            for table in ("runs", "candidates", "sessions", "turns", "feedback_items", "feedback_claims", "feedback_subclaims", "ledger"):
                self.conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
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
        path = detail.resolve_run(run_id)
        if path is None:
            return None
        with _WRITE:
            self._ingest(path)
            self.conn.commit()
        if self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
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
