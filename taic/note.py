"""Model annotations that answer with what is already known.

``python -m taic.note`` writes a structured annotation and replies:

- a measurement that already exists returns that row's numbers;
- a request that already exists returns its status;
- a repeated suggestion dismisses only the later occurrence; the anchored claim stays.

``NOTES.md`` / ``FEEDBACK.md`` stay the fallback when the note directory cannot
be created. This module does not call a model or a GPU, and it does not invent
numbers that are not already in a measurement file.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
import uuid
from pathlib import Path

SCHEMA = "taic.note.1"
DISMISSAL_SCHEMA = "taic.note.dismissals.1"
SUGGESTION_SECTIONS = frozenset({"对系统的建议", "需要帮助"})
_SHA = re.compile(r"\b[0-9a-f]{16,64}\b", re.I)
_FILE = re.compile(r"[\w./-]+\.(?:metal|py|json|jsonl)")
_NUMBER_KEYS = (
    "candidate_us", "median_us", "cv", "speedup", "conclusion", "ratio",
    "sha_a", "sha_b", "kernel_sha256", "sha256", "source_sha256",
    "median_a_s", "median_b_s", "ratios", "direction", "interval", "rounds",
)
_DIGEST_TOKEN = re.compile(r"^[0-9a-f]{16}$")
_SHA_KEYS = ("sha256", "kernel_sha256", "sha_a", "sha_b", "source_sha256")
_NAME_KEYS = ("name", "kernel", "source", "path_a", "path_b")
_TIMING_KEYS = ("candidate_us", "median_us", "speedup", "median_a_s", "median_b_s")
_FAILED_STATUS = frozenset({
    "timeout", "failed", "error", "skipped", "crash", "refused", "cancelled", "canceled",
})


class NoteError(ValueError):
    """@brief 标注无法写下，调用方没有拿到一次有效回话。"""


def _digest(section: str, text: str) -> str:
    from taic.agent_feedback import _digest as claim_digest

    return claim_digest(section, text)


def _read_json(path: Path) -> dict:
    from observe import safe_read

    document = safe_read.load_json(path)
    return document if isinstance(document, dict) else {}


def _read_jsonl(path: Path) -> list[dict]:
    from observe import safe_read

    return [row for row in safe_read.iter_jsonl(path) if isinstance(row, dict)]


def canonical_directory(path: Path | None) -> Path | None:
    """@brief 相对路径按当前目录展开后再解析。索引的 ``_rel`` 会把相对路径接到 runs 根上。"""
    if path is None:
        return None
    raw = Path(path)
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    try:
        return raw.resolve()
    except OSError:
        return raw


def _atomic_write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _named_digest(text: str) -> str:
    """@brief 整段正文恰好是 16 位 digest 时返回它。嵌在句子里的子串不算。"""
    token = str(text or "").strip().lower()
    return token if _DIGEST_TOKEN.fullmatch(token) else ""


def _as_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_float(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _row_simulated(row: dict) -> bool:
    """@brief 顶层旗标、测量种类、``evidence_level`` 或 ``evidence_provenance`` 任一处声明模拟即为模拟。

    ``declares_simulation`` 覆盖 ``simulated``、``measurement_kind`` / ``measurement_type`` 和 ``provenance``。
    宿主配对写入的 ``evidence_level`` 与 ``evidence_provenance`` 不在那个函数里，这里一并看。
    """
    from taic.evidence_level import SIMULATED, declares_simulation

    if str(row.get("evidence_level") or "") == SIMULATED or declares_simulation(row):
        return True
    provenance = row.get("evidence_provenance")
    if not isinstance(provenance, dict):
        return False
    level = str(provenance.get("evidence_level") or provenance.get("level") or "")
    return level == SIMULATED or declares_simulation(provenance)


def _numbers_from(row: dict, source: str, *, hardware_file: bool) -> dict:
    numbers = {key: row[key] for key in _NUMBER_KEYS if key in row and row[key] not in ("", None)}
    numbers["source"] = source
    numbers["simulated"] = _row_simulated(row)
    numbers["host_evidence"] = bool(hardware_file and row.get("host_evidence") is True)
    numbers["counts_as_hardware"] = numbers["host_evidence"] and not numbers["simulated"]
    if "status" in row and row.get("status") not in ("", None):
        numbers["status"] = row.get("status")
    if "applied" in row:
        numbers["applied"] = row.get("applied") is True
    return numbers


def _has_timing(row: dict) -> bool:
    for key in _TIMING_KEYS:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return True
    ratios = row.get("ratios")
    if isinstance(ratios, (list, tuple)) and len(ratios) > 0:
        return True
    if isinstance(ratios, dict) and len(ratios) > 0:
        return True
    return False


def _counts_as_measurement(row: dict) -> bool:
    """@brief 失败、未应用或没有数字的行不是已测。没有 status 但有数字的旧行仍然算。"""
    status = str(row.get("status") or "").strip().lower()
    if status in _FAILED_STATUS:
        return False
    if "applied" in row and row.get("applied") is False:
        return False
    return _has_timing(row)


def _row_matches(row: dict, shas: set[str], names: set[str]) -> bool:
    for key in _SHA_KEYS:
        value = str(row.get(key) or "").lower()
        if value and value in shas:
            return True
    for key in _NAME_KEYS:
        named = str(row.get(key) or "")
        if named and Path(named).name.lower() in names:
            return True
    return False


def _measurement_files(work: Path) -> list[tuple[Path, str, bool]]:
    """@brief 自测行、宿主自测权威日志与权威宿主配对。

    工作区里的同名配对文件不是证据，提名也不是已测。宿主自测日志在镜像之后读取，
    所以镜像里的同名行不会盖住它。它不是外层 host-process，也不算可发布的硬件证据。
    """
    files: list[tuple[Path, str, bool]] = []
    seen: set[Path] = set()
    for path in (
        work / ".selfcheck" / "self_checks.jsonl",
        work / "self_checks.jsonl",
    ):
        if path.is_file():
            files.append((path, str(path.relative_to(work)), False))
            try:
                seen.add(path.resolve())
            except OSError:
                continue
    from tcenv.selfcheck import authority_log

    self_check = authority_log(work / ".selfcheck")
    try:
        self_check_seen = self_check.resolve() in seen
    except OSError:
        self_check_seen = False
    if self_check.is_file() and not self_check_seen:
        files.append((self_check, "authority/self_checks.jsonl", False))
    from taic.paired_compare import host_evidence_path

    authority = host_evidence_path(work / ".selfcheck")
    workspace_copy = work / ".selfcheck" / "host_paired_comparisons.jsonl"
    try:
        same_file = authority.resolve() == workspace_copy.resolve()
    except OSError:
        same_file = False
    if authority.is_file() and not same_file:
        files.append((authority, "host_paired_comparisons.jsonl", True))
    return files


def find_measurement(work: Path, text: str) -> dict | None:
    """@brief 在已有自测行或权威宿主配对里找这段文字点名的测量。

    工作区 ``paired_comparisons.jsonl`` 是提名，``host_paired_comparisons.jsonl`` 的同名副本也不是证据。
    镜像删掉后仍读 ``authority_log`` 里的宿主自测行。那一行可以回数字，但 ``counts_as_hardware`` 仍为假。
    自测行的文件名键是 ``name``。权威宿主配对还认 ``kernel``、``source``、``path_a`` 和 ``path_b``。
    超时、失败、未应用或没有数字的行不算已测。同一 SHA 有多行时，按文件顺序取最后一条真实测量，早先的失败行不会挡住后面的成功行。

    @param work 会话工作目录。
    @param text 模型写下的文字。
    @return 测量数字和该行的 status/applied；没有真实测量时返回 None。
    """
    shas = {match.group(0).lower() for match in _SHA.finditer(text or "")}
    names = {Path(match.group(0)).name.lower() for match in _FILE.finditer(text or "")}
    if not shas and not names:
        return None
    found = None
    for path, source, hardware_file in _measurement_files(work):
        for row in _read_jsonl(path):
            if not _row_matches(row, shas, names) or not _counts_as_measurement(row):
                continue
            found = _numbers_from(row, source, hardware_file=hardware_file)
    return found


def _load_requests(work: Path) -> list[dict]:
    document = _read_json(work / ".notes" / "requests.json")
    items = document.get("items")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _load_claims(run: Path | None) -> list[dict]:
    if run is None:
        return []
    claims = []
    review = _read_json(run / "feedback_review.json").get("claims")
    review = review if isinstance(review, dict) else {}
    for record in _read_jsonl(run / "feedback.jsonl"):
        seq = record.get("seq")
        if not isinstance(seq, int):
            continue
        for item in record.get("items") or []:
            if not isinstance(item, dict):
                continue
            digest = str(item.get("digest") or "")
            if not digest:
                continue
            saved = review.get(f"{seq}:{digest}")
            status = str(saved["status"]) if isinstance(saved, dict) and saved.get("status") else ""
            claims.append({
                "digest": digest,
                "text": str(item.get("text") or ""),
                "section": str(item.get("section") or ""),
                "seq": seq,
                "status": status or "pending",
                "t": record.get("t"),
                "run_id": str(record.get("run_id") or run.name),
            })
    return claims


def _annotation_rows(work: Path) -> list[dict]:
    rows = _read_jsonl(work / ".notes" / "annotations.jsonl")
    from observe import safe_read

    try:
        text = safe_read.read_text(work / "NOTES.md")
    except OSError:
        return rows
    prefix = "taic.note "
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            row = json.loads(line[len(prefix):])
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _indexed_run_ids(work: Path, run: Path | None) -> set[str]:
    """@brief 这次 note 所属 run 的索引 id。跨 run 查询要排除它们，避免把本场未审 claim 当成另一场的已知请求。"""
    ids: set[str] = set()
    index = None
    try:
        from observe.index import get_index

        index = get_index()
    except (OSError, RuntimeError):
        index = None
    for directory in _runs_touching(work, run):
        ids.add(Path(directory).name)
        if index is None:
            continue
        try:
            ids.add(index._rel(directory))
        except (OSError, RuntimeError, ValueError):
            continue
    ids.discard("")
    return ids


def _rank_known(row: dict) -> tuple:
    """@brief 人工保存过的状态排在默认状态前面；同一类里取最近的时间。"""
    reviewed = row.get("reviewed_at")
    human = isinstance(reviewed, (int, float)) and not isinstance(reviewed, bool)
    when = float(reviewed if human else (row.get("record_t") or 0))
    return (1 if human else 0, when, int(row.get("seq") or 0))


def _known_from_row(row: dict, *, human: bool) -> dict:
    return {
        "status": str(row.get("status") or "pending"),
        "id": "",
        "digest": str(row.get("digest") or ""),
        "seq": int(row["seq"]),
        "run_id": str(row.get("run_id") or ""),
        "section": str(row.get("section") or ""),
        "human": human,
        "cross_run": True,
    }


def _indexed_known(text: str, digest: str, *, skip: set[str]) -> dict | None:
    """@brief 在 observe 索引里按精确 digest 或整句找其他 run 的 claim，并带回最新人工状态。"""
    import sqlite3

    stripped = " ".join(str(text or "").split())
    named = _named_digest(stripped)
    wanted = [item for item in (str(digest or "").lower(), named) if item]
    try:
        from observe.detail import resolve_run
        from observe.index import borrow
        from observe.suggestions import _text_for_digest
    except ImportError:
        return None
    try:
        with borrow() as index:
            if wanted:
                marks = ",".join("?" for _ in wanted)
                digest_rows = [dict(row) for row in index.conn.execute(
                    f"""
                    SELECT c.run_id AS run_id, c.seq AS seq, c.digest AS digest, c.section AS section,
                           c.status AS status, c.reviewed_at AS reviewed_at, i.t AS record_t, i.line_at AS line_at
                    FROM feedback_claims c
                    JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
                    WHERE lower(c.digest) IN ({marks})
                    """,
                    wanted,
                )]
            else:
                digest_rows = []
            text_rows = []
            if stripped and not any(str(row.get("run_id") or "") not in skip for row in digest_rows):
                text_rows = [dict(row) for row in index.conn.execute(
                    """
                    SELECT c.run_id AS run_id, c.seq AS seq, c.digest AS digest, c.section AS section,
                           c.status AS status, c.reviewed_at AS reviewed_at, i.t AS record_t, i.line_at AS line_at
                    FROM feedback_claims c
                    JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
                    WHERE c.digest <> ''
                    """
                )]
    except (OSError, sqlite3.Error, RuntimeError):
        return None
    matches = [row for row in digest_rows if str(row.get("run_id") or "") not in skip]
    if not matches and stripped:
        cache: dict = {}
        known = set(wanted)
        for item in text_rows:
            if str(item.get("run_id") or "") in skip or str(item.get("digest") or "").lower() in known:
                continue
            directory = resolve_run(str(item["run_id"]))
            if directory is None:
                continue
            body = " ".join(_text_for_digest(directory, item.get("line_at"), str(item.get("digest") or ""), cache).split())
            if body and body == stripped:
                matches.append(item)
    if not matches:
        return None
    best = max(matches, key=_rank_known)
    return _known_from_row(best, human=_rank_known(best)[0] == 1)


def find_request(work: Path, text: str, digest: str, *, run: Path | None = None, skip_run_ids: set[str] | None = None) -> dict | None:
    """@brief 精确匹配已知请求。正文包含某个 id 的子串不算已知。

    先看其他 run 在 observe 索引里的精确 digest 或整句。有人工保存状态时，带回最新的那条。
    当前 run 的 ``feedback.jsonl`` 和 ``requests.json`` 仍是本场命中。索引里没有的整句不限小节。

    @param work 会话工作目录。
    @param text 模型写下的文字。
    @param digest 这段文字在该小节下的 digest。
    @param run 可选的 run 目录，用来读已有 claim。
    @param skip_run_ids 索引查询要跳过的 run id，通常是这次 note 所属的 run。
    @return ``{"status", "id", "digest"}``；没有精确命中时返回 None。
    """
    stripped = " ".join(str(text or "").split())
    named = _named_digest(stripped)
    indexed = _indexed_known(stripped, digest, skip=skip_run_ids or set())
    local = None
    for item in _load_requests(work):
        item_digest = str(item.get("digest") or "").lower()
        item_text = " ".join(str(item.get("text") or "").split())
        item_id = str(item.get("id") or "")
        if (item_digest and item_digest in {digest, named}) or (item_text and item_text == stripped) or (item_id and item_id == stripped):
            local = {"status": str(item.get("status") or "open"), "id": item_id, "digest": item_digest or digest, "cross_run": False}
            break
    if local is None:
        for claim in _load_claims(run):
            claim_digest = str(claim["digest"] or "").lower()
            if claim_digest in {digest, named} or " ".join(claim["text"].split()) == stripped:
                local = {
                    "status": claim["status"] or "pending",
                    "id": "",
                    "digest": claim["digest"],
                    "seq": claim["seq"],
                    "run_id": claim["run_id"],
                    "section": claim["section"],
                    "cross_run": False,
                }
                break
    if indexed and indexed.get("human"):
        return indexed
    if local is not None:
        return local
    return indexed


def _prior_suggestions(work: Path, run: Path | None) -> list[dict]:
    priors = []
    for row in _annotation_rows(work):
        if row.get("section") not in SUGGESTION_SECTIONS:
            continue
        if row.get("dismissed") is True:
            continue
        text = str(row.get("text") or "")
        if not text:
            continue
        priors.append({
            "text": text,
            "digest": str(row.get("digest") or ""),
            "section": str(row.get("section") or ""),
            "status": str(row.get("status") or row.get("reply") or "recorded"),
            "seq": None,
            "run_id": "",
            "kind": "annotation",
        })
    for claim in _load_claims(run):
        if claim["section"] not in SUGGESTION_SECTIONS or not claim["text"]:
            continue
        priors.append({**claim, "kind": "claim"})
    return priors


def _near(text: str, prior: str, *, scorer, method: str | None, link_at: float | None) -> bool:
    from observe.suggestion_cluster import is_near_duplicate

    if scorer is None:
        from observe.similarity import lexical_overlap

        return is_near_duplicate(text, prior, scorer=lexical_overlap, method="lexical_overlap", link_at=link_at)
    return is_near_duplicate(text, prior, scorer=scorer, method=method or "lexical_overlap", link_at=link_at)


def find_duplicate(work: Path, text: str, digest: str, section: str, *, run: Path | None = None, scorer=None, method: str | None = None, link_at: float | None = None) -> dict | None:
    """@brief 找同一小节里已经写过的建议。精确 digest 优先。

    同一 digest 既有无序号的标注又有 claim 时，用带 seq 的那条，避免后一次标注把原序号冲掉。其次才是达到聚类阈值的近义句。

    @param work 会话工作目录。
    @param text 新建议正文。
    @param digest 新建议 digest。
    @param section 小节名。
    @param run 可选 run 目录。
    @param scorer 测试注入的打分函数。None 时用词面重叠，避免每次标注都加载句向量。
    @param link_at 覆盖近义阈值。
    @return 先前那条建议；没有重复时返回 None。
    """
    if section not in SUGGESTION_SECTIONS:
        return None
    from observe.suggestion_cluster import request_text

    left = request_text(text).strip()
    exact = None
    near = None
    for prior in _prior_suggestions(work, run):
        if prior["section"] != section:
            continue
        if prior["digest"] and prior["digest"] == digest:
            if exact is None or (exact.get("seq") is None and prior.get("seq") is not None):
                exact = prior
            if prior.get("seq") is not None:
                break
            continue
        if exact is not None or prior["text"] == text:
            continue
        right = request_text(prior["text"]).strip()
        if not left or not right:
            continue
        if left == right or _near(left, right, scorer=scorer, method=method, link_at=link_at):
            near = near or prior
    return exact or near


def _write_annotation(work: Path, row: dict) -> str:
    folder = work / ".notes"
    path = folder / "annotations.jsonl"
    folder.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return str(path)


def _write_fallback(work: Path, row: dict) -> str:
    path = work / "NOTES.md"
    line = "taic.note " + json.dumps(row, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return str(path)


def _record_dismissal(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            document = _read_json(path)
            if document.get("schema") not in (None, "", DISMISSAL_SCHEMA) or not isinstance(document.get("items"), dict):
                document = {"schema": DISMISSAL_SCHEMA, "items": {}}
            document["schema"] = DISMISSAL_SCHEMA
            items = document["items"]
            items[str(row["digest"])] = {
                "status": "info",
                "prior_digest": str(row.get("prior_digest") or ""),
                "prior_seq": row.get("prior_seq"),
                "prior_run": str(row.get("prior_run") or ""),
                "anchor_seq": row.get("anchor_seq"),
                "anchor_t": row.get("anchor_t"),
                "t": row.get("t"),
                "mode": str(row.get("mode") or ""),
                "note": str(row.get("note") or ""),
            }
            _atomic_write(path, document)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _dismissal_paths(directory: Path) -> list[Path]:
    from observe import safe_read

    paths = [directory / ".notes" / "dismissed.json"]
    root = directory / "agent_work"
    if not safe_read.is_directory(root, anchor=directory):
        return paths
    try:
        for child in sorted(root.iterdir()):
            if not safe_read.is_directory(child, anchor=root):
                continue
            paths.append(child / ".notes" / "dismissed.json")
    except OSError:
        pass
    return paths


def _stat_token(path: Path) -> str:
    from observe import safe_read

    info = safe_read.stat_leaf(path)
    if info is None:
        return ""
    return f"{info.st_mtime_ns}:{info.st_size}"


def dismissal_fingerprint(directory: Path) -> str:
    """@brief run 级和 ``agent_work/*/.notes/dismissed.json`` 的 stat 指纹。都没有时为空。"""
    parts = []
    for path in _dismissal_paths(directory):
        token = _stat_token(path)
        if not token:
            continue
        try:
            relative = path.relative_to(directory).as_posix()
        except ValueError:
            relative = path.as_posix()
        parts.append(f"{relative}={token}")
    return "|".join(parts)


def _runs_touching(work: Path, run: Path | None) -> list[Path]:
    found: list[Path] = []
    if run is not None:
        found.append(Path(run))
    current = Path(work)
    try:
        current = current.resolve()
    except OSError:
        pass
    for candidate in (current, *current.parents):
        if (candidate / "state.json").is_file() or (candidate / "result.json").is_file():
            found.append(candidate)
            break
        if candidate.parent == candidate:
            break
    unique: list[Path] = []
    seen: set[str] = set()
    for directory in found:
        try:
            key = str(directory.resolve())
        except OSError:
            key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        unique.append(directory)
    return unique


def _refresh_indexed_runs(work: Path, run: Path | None) -> None:
    """@brief 消重已经写下时，立刻重算受影响且已入索引的 run。未收录的目录跳过。"""
    from observe.index import get_index

    index = get_index()
    for directory in _runs_touching(work, run):
        index.refresh_feedback(index._rel(directory), directory)


def _earliest_occurrence(directory: Path, digest: str) -> tuple[int, float] | None:
    """@brief 这场里这条 digest 最早的一次出现。先比记录时间，再比序号。"""
    best_key: tuple[int, float, int] | None = None
    best: tuple[int, float] | None = None
    for record in _read_jsonl(directory / "feedback.jsonl"):
        seq = _as_int(record.get("seq"))
        if seq is None:
            continue
        items = record.get("items")
        if not isinstance(items, list):
            continue
        if not any(isinstance(item, dict) and str(item.get("digest") or "") == digest for item in items):
            continue
        moment = _as_float(record.get("t"))
        key = (0, moment, seq) if moment is not None else (1, 0.0, seq)
        if best_key is None or key < best_key:
            best_key = key
            best = (seq, moment if moment is not None else 0.0)
    return best


def _keeps_occurrence(directory: Path, item: dict, digest: str, seq: int, record_t: float | None) -> bool:
    """@brief 这次出现是不是该留下的那一条。

    近义句的新 digest、以及已知请求，整段都是重复项。精确重复留下锚定序号；
    note 当时没有序号时，留下这场里记录时间最早的那一次，再用序号打破平局。
    """
    mode = str(item.get("mode") or "")
    if mode in {"near", "known"}:
        return False
    if "anchor_seq" in item:
        anchor_seq = _as_int(item.get("anchor_seq"))
    else:
        anchor_seq = _as_int(item.get("prior_seq"))
    moment = _as_float(record_t)
    if anchor_seq is None:
        earliest = _earliest_occurrence(directory, digest)
        if earliest is None:
            return True
        anchor_seq, anchor_t = earliest
        if seq == anchor_seq:
            return True
        if moment is not None and (moment, seq) < (anchor_t, anchor_seq):
            return True
        return False
    return seq == anchor_seq


def _note_roots(directory: Path) -> list[Path]:
    roots = [directory]
    agent = directory / "agent_work"
    if not agent.is_dir():
        return roots
    try:
        roots.extend(path for path in sorted(agent.iterdir()) if path.is_dir())
    except OSError:
        return roots
    return roots


def _evidence(directory: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """@brief 这场能复核的 claim、标注和请求表。``.notes`` 里的缓存不在这里。"""
    claims = _load_claims(directory)
    notes: list[dict] = []
    requests: list[dict] = []
    for root in _note_roots(directory):
        notes.extend(_annotation_rows(root))
        requests.extend(_load_requests(root))
    return claims, notes, requests


def _skip_run_ids(directory: Path) -> set[str]:
    ids = {directory.name}
    try:
        from observe.index import get_index

        ids.add(get_index()._rel(directory))
    except (OSError, RuntimeError, ValueError):
        pass
    ids.discard("")
    return ids


def _request_still_there(requests: list[dict], digest: str, text: str) -> bool:
    stripped = " ".join(str(text or "").split())
    named = _named_digest(stripped)
    for item in requests:
        item_digest = str(item.get("digest") or "").lower()
        item_text = " ".join(str(item.get("text") or "").split())
        item_id = str(item.get("id") or "")
        if item_digest and item_digest in {digest.lower(), named}:
            return True
        if stripped and item_text and item_text == stripped:
            return True
        if stripped and item_id and item_id == stripped:
            return True
    return False


def _linked_note(notes: list[dict], digest: str, text: str, *, reply: str) -> dict | None:
    stripped = " ".join(str(text or "").split())
    for note in notes:
        prior = note.get("prior") if isinstance(note.get("prior"), dict) else {}
        prior_digest = str(prior.get("digest") or "")
        note_digest = str(note.get("digest") or "")
        note_text = " ".join(str(note.get("text") or "").split())
        linked = note_digest == digest or prior_digest == digest or (stripped and note_text == stripped)
        if not linked:
            continue
        if reply == "known_request" and note.get("reply") == "known_request":
            return note
        if reply == "duplicate" and note.get("reply") == "duplicate":
            return note
    return None


def _occurrence_text(claims: list[dict], digest: str, seq: int, text: str) -> str:
    stripped = " ".join(str(text or "").split())
    if stripped:
        return stripped
    for claim in claims:
        if claim["digest"] == digest and claim["seq"] == seq:
            return " ".join(claim["text"].split())
    return ""


def _dismissal_proven(directory: Path, item: dict, digest: str, seq: int, text: str, section: str) -> bool:
    """@brief 缓存里的 mode 必须能从 claim、标注或宿主索引重新证出来。

    只有时间戳和 mode 的 sidecar 不算数。近义要有先前正文和对应的 duplicate 标注。
    已知请求要有仍在的请求、另一场宿主索引，或本场另一条 claim，并有对应标注。
    精确重复的锚点必须是本场 feedback 里同 digest 的另一次出现。
    """
    mode = str(item.get("mode") or "")
    if mode not in {"near", "known", "later-seq"}:
        return False
    if mode == "near" and section and section not in SUGGESTION_SECTIONS:
        return False
    claims, notes, requests = _evidence(directory)
    body = _occurrence_text(claims, digest, seq, text)
    if mode == "later-seq":
        anchor = _as_int(item.get("anchor_seq"))
        if anchor is None:
            anchor = _as_int(item.get("prior_seq"))
        same = [claim for claim in claims if claim["digest"] == digest]
        if anchor is not None:
            return any(claim["seq"] == anchor for claim in same)
        return any(claim["seq"] != seq for claim in same)
    if mode == "known":
        try:
            indexed = _indexed_known(body or digest, digest, skip=_skip_run_ids(directory))
        except (OSError, RuntimeError, ValueError):
            indexed = None
        if indexed is not None:
            return True
        note = _linked_note(notes, digest, body, reply="known_request")
        if note is None:
            return False
        found = list(requests)
        work = note.get("work")
        if work:
            found.extend(_load_requests(Path(str(work))))
        if _request_still_there(found, digest, body):
            return True
        for claim in claims:
            if claim["seq"] == seq and claim["digest"] == digest:
                continue
            claim_text = " ".join(claim["text"].split())
            if claim["digest"] == digest or (body and claim_text == body):
                return True
        return False
    prior_digest = str(item.get("prior_digest") or "")
    priors: list[dict] = []
    for claim in claims:
        if claim["section"] not in SUGGESTION_SECTIONS or claim["digest"] == digest or not claim["text"]:
            continue
        if prior_digest and claim["digest"] != prior_digest:
            continue
        priors.append(claim)
    for note in notes:
        if note.get("dismissed") is True or note.get("reply") == "duplicate":
            continue
        if str(note.get("section") or "") not in SUGGESTION_SECTIONS:
            continue
        note_digest = str(note.get("digest") or "")
        if not note_digest or note_digest == digest:
            continue
        if prior_digest and note_digest != prior_digest:
            continue
        priors.append({
            "digest": note_digest,
            "text": str(note.get("text") or ""),
            "section": str(note.get("section") or ""),
        })
    matched = None
    for prior in priors:
        right = " ".join(str(prior.get("text") or "").split())
        if not body or not right:
            continue
        if body == right or _near(body, right, scorer=None, method=None, link_at=None):
            matched = prior
            break
    if matched is None:
        return False
    note = _linked_note(notes, digest, body, reply="duplicate")
    if note is None:
        return False
    pointed = str((note.get("prior") or {}).get("digest") or "")
    return not prior_digest or pointed == prior_digest or pointed == str(matched.get("digest") or "")


def dismissal_for(
    directory: Path | None,
    digest: str,
    *,
    seq: int | None = None,
    record_t: float | None = None,
    text: str = "",
    section: str = "",
) -> dict | None:
    """@brief 这次出现要不要因为一次已证实的重复而变成 info。

    ``.notes/dismissed.json`` 只是可重算缓存。应用前要能从本场 claim、标注或宿主索引
    重新证明锚点和重复关系；证不出来就不改队列状态。精确重复只消掉锚定序号之后的坐标。
    第一条、或 note 匹配到的那次出现保持原状态。没有锚定序号时，用这次的记录时间和序号，
    对照这场里最早的一次。近义句只消掉新 digest。已知请求消掉当前 run 上的这次出现。
    已有人工状态的 claim 由调用方跳过，这里不覆盖。

    @param directory run 目录。
    @param digest claim digest。
    @param seq 这次出现的序号。没有序号时无法区分坐标，不消。
    @param record_t 这次出现的记录时间。没有锚定序号时，用来判断它是不是最早的一次。
    @param text 这次出现的正文。空则从本场 feedback 里按序号读取。
    @param section 这次出现的小节。近义证明只用建议小节里的先前正文。
    @return 可并进评审保存对象的字典；这次出现该留下，或关系证不出来时返回 None。
    """
    if directory is None or not digest or seq is None:
        return None
    chosen = None
    for path in _dismissal_paths(directory):
        document = _read_json(path)
        items = document.get("items")
        if not isinstance(items, dict) or digest not in items:
            continue
        item = items[digest]
        if not isinstance(item, dict):
            continue
        try:
            moment = float(item.get("t") or 0)
        except (TypeError, ValueError):
            continue
        if chosen is None or moment >= chosen[0]:
            chosen = (moment, item)
    if chosen is None:
        return None
    item = chosen[1]
    if not _dismissal_proven(directory, item, digest, seq, text, section):
        return None
    if _keeps_occurrence(directory, item, digest, seq, record_t):
        return None
    return {
        "status": "info",
        "verification": "unchecked",
        "note": str(item.get("note") or "duplicate suggestion dismissed at note time"),
    }


def annotate(
    work: Path,
    *,
    text: str,
    section: str = "note",
    run: Path | None = None,
    scorer=None,
    method: str | None = None,
    link_at: float | None = None,
    now: float | None = None,
) -> dict:
    """@brief 写下一条模型标注，并按已有测量、已知请求或重复建议回话。

    @param work 会话工作目录，必须已经存在。
    @param text 标注正文。
    @param section ``对系统的建议``、``需要帮助`` 或其他标注小节。
    @param run 可选 run 目录。重复建议会写到这里的 ``.notes/dismissed.json``。
    @param scorer 近义打分。None 时用词面重叠。
    @param link_at 覆盖近义阈值。
    @param now 测试注入的时间戳。
    @return 回话。``reply`` 是 ``already_measured``、``known_request``、``duplicate`` 或 ``recorded``。
    @raises NoteError 正文为空、工作目录不存在，或结构化文件和 ``NOTES.md`` 都写不了。
    """
    work = canonical_directory(Path(work))
    run = canonical_directory(None if run is None else Path(run))
    if work is None or not work.is_dir():
        raise NoteError("note work directory does not exist")
    body = " ".join(str(text or "").split())
    if not body:
        raise NoteError("note text is empty")
    if len(body) > 4000:
        body = body[:4000]
    section = str(section or "note")
    digest = _digest(section, body)
    moment = float(time.time() if now is None else now)
    measurement = find_measurement(work, body)
    duplicate = find_duplicate(work, body, digest, section, run=run, scorer=scorer, method=method, link_at=link_at)
    request = find_request(work, body, digest, run=run, skip_run_ids=_indexed_run_ids(work, run))
    dismissed = False
    prior = None
    status = ""
    if request is not None and request.get("cross_run"):
        duplicate = None
    if duplicate is not None:
        reply = "duplicate"
        status = str(duplicate.get("status") or "recorded")
        prior = {
            "digest": duplicate.get("digest") or "",
            "status": status,
            "seq": duplicate.get("seq"),
            "t": duplicate.get("t"),
            "run_id": duplicate.get("run_id") or "",
            "kind": duplicate.get("kind") or "",
        }
        dismissed = True
    elif request is not None:
        reply = "known_request"
        status = str(request.get("status") or "open")
        prior = {
            "digest": request.get("digest") or digest,
            "status": status,
            "seq": request.get("seq"),
            "t": request.get("t"),
            "run_id": request.get("run_id") or "",
            "kind": "request",
        }
        claim_section = str(request.get("section") or "")
        dismissed = section in SUGGESTION_SECTIONS and claim_section in {"", *SUGGESTION_SECTIONS}
    elif measurement is not None:
        reply = "already_measured"
        status = "measured"
    else:
        reply = "recorded"
        status = "recorded"
    row = {
        "schema": SCHEMA,
        "t": moment,
        "text": body,
        "section": section,
        "digest": digest,
        "reply": reply,
        "dismissed": dismissed,
        "status": status,
        "numbers": measurement,
        "prior": prior,
        "work": str(work),
        "fallback": False,
    }
    wrote = ""
    fallback = False
    try:
        wrote = _write_annotation(work, row)
    except OSError:
        try:
            row["fallback"] = True
            wrote = _write_fallback(work, row)
            fallback = True
        except OSError as exc:
            raise NoteError("could not write the note or the NOTES.md fallback") from exc
    if dismissed:
        try:
            if run is not None:
                try:
                    same_note = (work / ".notes" / "annotations.jsonl").resolve() == (
                        Path(run) / ".notes" / "annotations.jsonl"
                    ).resolve()
                except OSError:
                    same_note = False
                if not same_note:
                    _write_annotation(Path(run), row)
            prior_digest = str((prior or {}).get("digest") or "")
            raw_seq = (prior or {}).get("seq")
            prior_seq = raw_seq if isinstance(raw_seq, int) and not isinstance(raw_seq, bool) else None
            prior_t = _as_float((prior or {}).get("t"))
            if reply == "known_request":
                mode = "known"
            elif prior_digest and prior_digest != digest:
                mode = "near"
            else:
                mode = "later-seq"
            digests = [digest]
            if mode == "known" and prior_digest and prior_digest != digest:
                digests.append(prior_digest)
            targets = [work / ".notes" / "dismissed.json"]
            if run is not None:
                targets.append(Path(run) / ".notes" / "dismissed.json")
            for item_digest in digests:
                dismissal = {
                    "digest": item_digest,
                    "prior_digest": prior_digest,
                    "prior_seq": prior_seq,
                    "prior_run": str((prior or {}).get("run_id") or ""),
                    "anchor_seq": prior_seq if mode == "later-seq" else None,
                    "anchor_t": prior_t if mode == "later-seq" else None,
                    "t": moment,
                    "mode": mode,
                    "note": f"duplicate of {prior_digest or digest} ({status})",
                }
                for target in targets:
                    _record_dismissal(target, dismissal)
            _refresh_indexed_runs(work, run)
        except OSError as exc:
            raise NoteError("could not persist the dismissal") from exc
    return {**row, "wrote": wrote, "fallback": fallback}


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m taic.note``。成功回话的退出码是 0，空正文或写失败是 5。"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="taic.note")
    parser.add_argument("--work", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--section", default="note")
    parser.add_argument("--run", default=None)
    parser.add_argument("--embed", action="store_true", help="use the platform cluster scorer; default is lexical overlap")
    args = parser.parse_args(argv)
    scorer = None
    link_at = None
    if args.embed:
        from observe.suggestion_cluster import select_scorer

        scorer, method = select_scorer()
    else:
        method = None
    try:
        document = annotate(
            Path(args.work), text=args.text, section=args.section,
            run=None if args.run is None else Path(args.run),
            scorer=scorer, method=method, link_at=link_at,
        )
    except NoteError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
