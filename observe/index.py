"""Rebuildable SQLite index of ``runs/``. One writer. Readers open the file read-only.

The database is a derivative. Deleting it and opening it again rebuilds the rows from ``runs/``,
``data/books.sqlite`` (read-only, and only to seed the carry-over) and ``ledger_carryover.json``.
A schema version this process does not know is dropped and built again. There is no migration.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from observe import detail, legacy
from observe.budget import budget_status
from observe.ledger import month_of, operator_name

INDEX_SCHEMA_VERSION = 9
_SKIP_DIRS = {"candidates", "__pycache__"}
_WRITE = threading.Lock()
_SINGLE: "Index | None" = None


class CarryoverError(RuntimeError):
    """The carry-over file does not match its recorded checksum. A person has to look at it."""


def db_path() -> Path:
    override = os.environ.get("OBSERVE_DB")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "observe.sqlite"


def books_path() -> Path:
    override = os.environ.get("AI_LOWERLING_BOOKS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "books.sqlite"


def carryover_path() -> Path:
    override = os.environ.get("OBSERVE_CARRYOVER")
    if override:
        return Path(override)
    return db_path().with_name("ledger_carryover.json")


def get_index() -> "Index":
    """The process-wide writer. A new ``OBSERVE_DB`` opens a new database."""
    global _SINGLE
    path = db_path()
    with _WRITE:
        if _SINGLE is None or _SINGLE.path != path:
            _SINGLE = Index(path)
        return _SINGLE


@contextmanager
def borrow():
    """The one writer connection, held so a request thread can read it."""
    index = get_index()
    with _WRITE:
        yield index


def _canonical(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _target_id(config: dict | None) -> str:
    target = (config or {}).get("target") if isinstance(config, dict) else None
    if isinstance(target, dict) and target.get("id"):
        return str(target["id"])
    return "apple-m4-max"


def _num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


class Index:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.stats = {"json_reads": 0, "dir_lists": 0, "turn_parses": 0}
        self.changed: list[str] = []
        self._in_reanchor = False
        self._last_full = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open()

    def _open(self) -> None:
        if self.path.exists() and not self._version_ok():
            self._discard()
        self.conn = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._create()
        self._load_carryover()

    def _version_ok(self) -> bool:
        try:
            conn = sqlite3.connect(self.path, timeout=5)
        except sqlite3.Error:
            return False
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'index_schema_version'").fetchone()
        except sqlite3.Error:
            return False
        finally:
            conn.close()
        return row is not None and str(row[0]) == str(INDEX_SCHEMA_VERSION)

    def _discard(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix) if suffix else self.path
            try:
                candidate.unlink()
            except OSError:
                pass

    def _create(self) -> None:
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
                PRIMARY KEY (run_id, seq, digest)
            );
            CREATE INDEX IF NOT EXISTS feedback_digest ON feedback_claims(digest);
            CREATE INDEX IF NOT EXISTS feedback_topic ON feedback_claims(topic);
            CREATE INDEX IF NOT EXISTS feedback_status ON feedback_claims(status);
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
        self.conn.execute("CREATE INDEX IF NOT EXISTS feedback_claim_rev ON feedback_claims(rev)")
        run_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(runs)")}
        if "feedback_bytes" not in run_columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN feedback_bytes INTEGER NOT NULL DEFAULT 0")
        if "feedback_review_fp" not in run_columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN feedback_review_fp TEXT NOT NULL DEFAULT ''")
        if "feedback_parser" not in run_columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN feedback_parser TEXT NOT NULL DEFAULT ''")
        item_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(feedback_items)")}
        if "line_at" not in item_columns:
            self.conn.execute("ALTER TABLE feedback_items ADD COLUMN line_at INTEGER")
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
        if self.conn.execute("SELECT 1 FROM ledger_carryover LIMIT 1").fetchone():
            return
        self.conn.execute(
            "INSERT INTO ledger_carryover (created_at, rows, total_usd, origin, sha256) VALUES (?, ?, ?, ?, ?)",
            (doc["created_at"], doc["rows"], doc["total_usd"], doc["origin"], doc["sha256"]),
        )
        self.conn.commit()

    def read_only(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def revision(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()
        return int(row[0] if row else 0)

    def _bump(self) -> None:
        self.conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'"
        )

    def turns_revision(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'turns_revision'").fetchone()
        return int(row[0] if row else 0)

    def _bump_turns(self) -> None:
        """A new transcript row. The detail revision stays where it is."""
        self.conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'turns_revision'"
        )

    def feedback_revision(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'feedback_revision'").fetchone()
        return int(row[0]) if row else 0

    def _full_interval(self) -> float:
        try:
            return max(0.0, float(os.environ.get("OBSERVE_FULL_TICK_INTERVAL", "10")))
        except ValueError:
            return 10.0

    def tick(self) -> tuple[int, list[str]]:
        """One incremental frame. Root mtime detects new runs; a full watched scan runs every 10s."""
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
        """See edits and new child directories. The idle frame uses :meth:`tick` and does not call this."""
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
            self._last_full = time.monotonic()
            self.conn.commit()

    def _prepare_frame(self) -> None:
        self.stats = {"json_reads": 0, "dir_lists": 0, "turn_parses": 0}
        self.changed = []
        root = str(detail.runs_root().resolve())
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'runs_root'").fetchone()
        if row is not None and row[0] != root:
            self.conn.executescript(
                "DELETE FROM runs; DELETE FROM candidates; DELETE FROM sessions; DELETE FROM experiments; DELETE FROM arms; DELETE FROM ledger; DELETE FROM watched; DELETE FROM feedback_items; DELETE FROM feedback_claims;"
            )
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('runs_root', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (root,),
        )

    def _watched(self, rel: str) -> int | None:
        row = self.conn.execute("SELECT mtime_ns FROM watched WHERE path = ?", (rel,)).fetchone()
        return None if row is None else int(row[0])

    def _note_dir(self, directory: Path, *, force: bool) -> None:
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
        if (directory / "state.json").exists() or (directory / "result.json").exists() or (directory / "experiment.json").exists():
            self._ingest(directory)
        self._note_dir(directory, force=False)

    def _refresh_watched(self) -> None:
        """Notice a new child or a resumed run. Creating a nested directory changes its parent, not ``runs/``."""
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
        row = self.conn.execute(
            "SELECT fp_state, fp_result, fp_config, fp_confirm, fp_extra FROM runs WHERE run_id = ?",
            (self._rel(directory),),
        ).fetchone()
        if row is None:
            return ("", "", "", "", "")
        return tuple(row)

    def _fingerprints(self, directory: Path, winner_label: str = "", winner_rel: str = "") -> tuple[str, str, str, str, str]:
        confirm = "\x1f".join((
            _fingerprint(directory / "confirm_report.json"),
            _fingerprint(directory / "best.metal"),
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
        if not path.exists():
            return None
        self.stats["json_reads"] += 1
        return detail._load(path)

    def _catch_up_turns(self, directory: Path) -> bool:
        """Project new complete lines of ``transcript.jsonl`` into the turns table.

        The byte offset is ``runs.turn_bytes``. A short file is projected again from the start.
        A trailing partial line stays unread until the next frame. Returns true when the offset
        or the table changed. Does not bump the detail revision.
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
        """A finished run that has never been projected. Opening its turn list builds the rows."""
        row = self.conn.execute(
            "SELECT status, turn_bytes FROM runs WHERE run_id = ?",
            (self._rel(directory),),
        ).fetchone()
        if row is None:
            return False
        return row["status"] != "running" and int(row["turn_bytes"] or 0) == 0

    def _project_turns(self, directory: Path) -> None:
        """Catch up a run that is still going, or one that already has a projection to extend."""
        if self._finished_without_turns(directory):
            return
        if self._catch_up_turns(directory):
            self._bump_turns()

    def ensure_turns(self, directory: Path) -> None:
        """Project this run on demand. A second call with no new bytes does not parse the file again."""
        with _WRITE:
            if self._catch_up_turns(directory):
                self._bump_turns()
            self.conn.commit()

    def _ingest(self, directory: Path, *, force: bool = False) -> None:
        if not self._within_runs(directory):
            return
        stored = self._stored_fingerprints(directory)
        winner_label, winner_rel = _confirm_target(stored[3])
        fingerprints = self._fingerprints(directory, winner_label, winner_rel)
        known = self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (self._rel(directory),)).fetchone()
        tracked = "\x1f" in (stored[3] or "")
        if not force and fingerprints == stored and known and (tracked or not (directory / "best.metal").exists()):
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
        """One row per candidate directory. The report supplies status/stage/speedup; the trace supplies the noise mark."""
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
            if not isinstance(report, dict) and not (path / "kernel.metal").is_file():
                continue
            status = str(report.get("status") or "") if isinstance(report, dict) else ""
            stage = str(report.get("stage") or "") if isinstance(report, dict) else ""
            speedup = _num(report.get("speedup")) if isinstance(report, dict) else None
            eligible = 1 if isinstance(report, dict) and report.get("eligible") is True else 0
            sha = _sha256_file(path / "kernel.metal")
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
        """Index one row per hmz session. Legacy chat transcripts are skipped when they are large."""
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

                def token(name: str) -> int:
                    value = tokens.get(name, 0)
                    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

                self.conn.execute(
                    """
                    INSERT INTO sessions (
                        run_id, seq, label, committee, provider, model, outcome, cost_usd,
                        tokens_input, tokens_output, tokens_cache_read, tokens_cache_write, t
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id, seq, str(event.get("label") or ""), str(event.get("committee") or ""),
                        str(event.get("provider") or ""), str(event.get("model") or ""), str(event.get("outcome") or ""),
                        _num(event.get("cost_usd")), token("input"), token("output"), token("cache_read"), token("cache_write"),
                        None,
                    ),
                )

    def _upsert_missing(self, directory: Path, fingerprints: tuple[str, str, str, str, str]) -> None:
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
        kernel_sha = str(result_doc.get("confirmation_source_sha256") or "") or (_sha256_file(directory / "best.metal") or "")
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
                kernel_sha256, protocol_version, committees_json, model, preset,
                switch_delta_json, contract, status, phase, stop_reason, started_at, updated_at,
                spent_usd, spent_max_usd, budget_usd, granted_budget_usd, publication, published_speedup,
                naive_speedup, rounds, evaluations, inflight_label, pid, source, binding,
                fp_state, fp_result, fp_config, fp_confirm, fp_extra, row_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                experiment_id = excluded.experiment_id, arm_id = excluded.arm_id, trial = excluded.trial,
                kind = excluded.kind, task = excluded.task, operator = excluded.operator, dtype = excluded.dtype,
                size = excluded.size, target_id = excluded.target_id, kernel_sha256 = excluded.kernel_sha256,
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
                kernel_sha, protocol, committees, model or "", (config or {}).get("preset") or "",
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
        """The anchor is the task fingerprint of the lowest trial, not whichever directory was scanned first."""
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
        root = detail.runs_root().resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError:
            return False
        return True

    def _manifest_fingerprint(self, directory: Path) -> str:
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
        root = detail.runs_root().resolve()
        try:
            text = path.resolve().relative_to(root).as_posix()
        except ValueError:
            return path.name
        return text or "."

    def total_usd(self) -> float:
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
        """Forget archived searches. The ledger rows stay, so a deletion cannot lower the recorded spend."""
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

    def refresh_feedback(self, run_id: str, directory: Path) -> None:
        """Re-read one run's feedback files now. A finished run is not scanned again by an idle tick."""
        with _WRITE:
            self._ingest(Path(directory), force=True)
            self.conn.commit()

    def resolve(self, run_id: str) -> Path | None:
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
    name = Path(relative).name
    if name.startswith("trial_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _confirm_target(fp_confirm: str) -> tuple[str, str]:
    parts = (fp_confirm or "").split("\x1f")
    if len(parts) >= 4:
        return parts[2], parts[3]
    return "", ""


def _winner_rel(directory: Path, label: str) -> str:
    if not label:
        return ""
    root = directory / "candidates"
    if not root.is_dir():
        return ""
    found = [path for path in root.iterdir() if path.is_dir() and len(path.name) > 4 and path.name[3] == "_" and path.name[4:] == label]
    if len(found) != 1:
        return ""
    return found[0].relative_to(directory).as_posix() + "/kernel.metal"


def _lookup(config: dict, search: dict, key: str):
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
    if isinstance(expected, float) or isinstance(got, float):
        left, right = _num(got), _num(expected)
        if left is None or right is None:
            return got == expected
        return abs(left - right) <= 1e-6
    return got == expected
