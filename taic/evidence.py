"""§9.2 证据导出与校验：``.tar.gz`` 两遍流式 + 清单 + 净化 + 截断。

完整性是**相对源清单**证明的，不是自我证明：``inventory_sha256`` 覆盖源目录的 included/excluded/
missing；``verify`` 从包内重算它，``--against RUN_DIR`` 再对源目录重算一遍。``detail_stamp`` 只作为
source 元数据保留，不从包内重算。

默认集是白名单制（``result.json``/``state.json``/``confirm_report.json``/``candidates/*/{report.json,
kernel.metal}``/``feedback.jsonl``，以及源里若有的 ``publications.json``）；其余实际存在的项逐条进
``excluded``（目录带字节数与文件数），这样 ``grep -r "/Users/"`` 为空的验收才有意义。没有源
``publications.json`` 时，导出会派生一份（目录史分片 + 当前发布判定），它是包内容，不进源清单。
"""

from __future__ import annotations

import codecs
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path

SCHEMA_VERSION = 1
MAX_FILE_BYTES = 8 * 1024 * 1024
READ_CHUNK = 1 << 20
TOP_FILES = ("result.json", "state.json", "confirm_report.json", "feedback.jsonl")
CANDIDATE_FILES = ("report.json", "kernel.metal")
MANIFEST_NAME = "MANIFEST.json"
PUBLICATIONS_NAME = "publications.json"
PUBLICATIONS_BYTES = 2048
# 通用绝对路径规则：净化与校验共用同一份。只认真正的绝对路径起点：
# - 文本开头、空白、引号、开括号、`=`/`,`/`;`/`:` 之后，或 `C:`、`file://` 之后；
# - 不把 `/b` 当成路径：`return (a + b)/b;` 的 `/` 前是 `)`，`v[i]/n` 的 `/` 前是 `]`；
# - `//etc/passwd`、`///opt/private` 的连续斜杠整段匹配，不会只漏掉前两个；
# - `http://`/`https://` 明确排除，URL 保持原样。
# 路径段允许 JSON 的 `\uXXXX` 转义（``json.dumps`` 默认就把中文写成这种形式），
# 否则 `/opt/\u79c1\u8d27` 只会替换 `/opt/` 而漏掉后面的段。
_ABS_SEGMENT = r"(?:\\u[0-9a-fA-F]{4}|[^\s/?#\"'<>\\])+"
# 斜杠可能是原样 `/`、JSON 的 solidus 转义 `\/`，或 Unicode 转义 `\u002f`；三种都按同一条路径规则。
_ABS_SLASH = r"(?:/|\\/|\\u002[fF])"
_ABS_SLASHES = _ABS_SLASH + "+"
_ABS_BOUNDARY = r"(?:(?<![\w./\\:)\]}*+%-])|(?<=[A-Za-z]:)|(?<=file://))"
ABS_PATH_RE = re.compile(
    r"(?<!http:)(?<!https:)" + _ABS_BOUNDARY + _ABS_SLASHES + _ABS_SEGMENT +
    r"(?:" + _ABS_SLASHES + _ABS_SEGMENT + r")*"
)
ABS_FAST_MARKERS = (b"/Users/", b"/home/", b"/tmp/", b"/private/", b"/var/folders/")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(data: bytes) -> str:
    """@brief 一段字节的 sha256。

    @param data 字节。
    @return 十六进制摘要。
    """
    return hashlib.sha256(data).hexdigest()


def _tree_size(path: Path) -> tuple[int, int]:
    """@brief 目录下的文件数与总字节（不跟符号链接）。

    @param path 目录。
    @return ``(文件数, 总字节)``。
    """
    files = 0
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink() or not item.is_file():
            continue
        files += 1
        try:
            total += item.stat().st_size
        except OSError:
            pass
    return files, total


def _excluded_dir(root: Path, path: Path, reason: str = "not in the evidence whitelist") -> dict:
    """@brief 一个被排除的目录（相对路径带尾斜杠、递归汇总字节数与文件数）。

    @param root 源 run 目录。
    @param path 目录。
    @param reason 排除原因。
    @return excluded 行。
    """
    files, total = _tree_size(path)
    return {"path": path.relative_to(root).as_posix() + "/", "reason": reason, "bytes": total, "files": files}


def _excluded_file(root: Path, path: Path, reason: str = "not in the evidence whitelist") -> dict:
    """@brief 一个被排除的普通文件/符号链接（相对路径、字节数、文件数）。

    @param root 源 run 目录。
    @param path 文件或链接。
    @param reason 排除原因。
    @return excluded 行。
    """
    if path.is_symlink():
        # 不跟随链接：链接目标可能在 RUN_DIR 之外，不能把外部文件打进证据包。
        return {"path": path.relative_to(root).as_posix(), "reason": "symlink", "bytes": 0, "files": 0}
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {"path": path.relative_to(root).as_posix(), "reason": reason, "bytes": size, "files": 1}


def source_entries(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """@brief 第一遍：按白名单把源目录分成 included/excluded/missing。

    ``excluded`` 枚举实际存在的每一项：顶层非白名单目录的**每个子项**单独一行（子目录递归汇总
    字节数与文件数），不是一条 glob。``candidates`` 是符号链接或普通文件时整条排除，绝不跟随。

    @param root 源 run 目录。
    @return ``(included, excluded, missing)``；included 每行带 ``path`` 与 ``file``。
    """
    included: list[dict] = []
    excluded: list[dict] = []
    missing: list[dict] = []
    for name in TOP_FILES:
        path = root / name
        if path.is_symlink():
            # 顶层符号链接同样只登记一次、不跟随；不能记成 missing 让它从清单里消失。
            excluded.append(_excluded_file(root, path))
        elif path.is_file():
            included.append({"path": name, "file": path})
        elif path.is_dir():
            excluded.append(_excluded_dir(root, path))
        elif path.exists():
            excluded.append(_excluded_file(root, path))
        else:
            missing.append({"path": name, "reason": "not produced" if name == "confirm_report.json" else "not found"})
    source_publications = root / PUBLICATIONS_NAME
    if source_publications.is_symlink():
        # 不跟随链接；也不能随后再派生一个同名 member（会跨 section 重复）。
        excluded.append(_excluded_file(root, source_publications))
    elif source_publications.is_file():
        included.append({"path": PUBLICATIONS_NAME, "file": source_publications})
    elif source_publications.is_dir():
        excluded.append(_excluded_dir(root, source_publications))
    elif source_publications.exists():
        excluded.append(_excluded_file(root, source_publications))
    candidates = root / "candidates"
    if candidates.is_symlink() or (candidates.exists() and not candidates.is_dir()):
        excluded.append(_excluded_file(root, candidates))
    elif candidates.is_dir():
        for child in sorted(candidates.iterdir()):
            if child.is_dir() and not child.is_symlink():
                handled_candidate: set[str] = set()
                for name in CANDIDATE_FILES:
                    path = child / name
                    if path.is_symlink():
                        excluded.append(_excluded_file(root, path))
                        handled_candidate.add(name)
                    elif path.is_file():
                        included.append({"path": f"candidates/{child.name}/{name}", "file": path})
                        handled_candidate.add(name)
                    elif path.is_dir() and not path.is_symlink():
                        # 候选内错类型目录：递归汇总，不能只记目录自身。
                        excluded.append(_excluded_dir(root, path))
                        handled_candidate.add(name)
                    elif path.exists():
                        # 设备等其它错类型：只登记一次，不跟随。
                        excluded.append(_excluded_file(root, path))
                        handled_candidate.add(name)
                    else:
                        missing.append({"path": f"candidates/{child.name}/{name}", "reason": "not produced"})
                for extra in sorted(child.iterdir()):
                    if extra.name in handled_candidate:
                        continue
                    if extra.name in CANDIDATE_FILES and extra.is_file() and not extra.is_symlink():
                        continue
                    if extra.is_dir() and not extra.is_symlink():
                        excluded.append(_excluded_dir(root, extra))
                    else:
                        excluded.append(_excluded_file(root, extra))
            else:
                excluded.append(_excluded_file(root, child))
    handled = {entry["path"] for entry in included}
    for child in sorted(root.iterdir()):
        if child.name in TOP_FILES or child.name in ("candidates", PUBLICATIONS_NAME) or child.name in handled:
            continue
        if child.is_dir() and not child.is_symlink():
            children = sorted(child.iterdir())
            if not children:
                # 空目录没有子项可枚举；它自己就是必须登记的那一项。
                excluded.append(_excluded_dir(root, child))
                continue
            for sub in children:
                if sub.is_dir() and not sub.is_symlink():
                    excluded.append(_excluded_dir(root, sub))
                else:
                    excluded.append(_excluded_file(root, sub))
        else:
            excluded.append(_excluded_file(root, child))
    included.sort(key=lambda row: row["path"])
    excluded.sort(key=lambda row: row["path"])
    missing.sort(key=lambda row: row["path"])
    return included, excluded, missing


def _inventory_entries(included: list[dict], excluded: list[dict], missing: list[dict]) -> list[dict]:
    """@brief 清单的覆盖集合：``(path, kind, bytes)``，按路径排序。

    included 的字节取**源**字节数（``source_bytes``）；派生文件（``derived``）不属于源清单，
    否则外部目录史一追加就会让未变的 RUN_DIR 在 ``--against`` 里失效。

    @param included manifest 的 files 行。
    @param excluded excluded 行。
    @param missing missing 行。
    @return 排序后的条目。
    """
    entries = [
        {"path": row["path"], "kind": "included", "bytes": int(row.get("source_bytes", row.get("bytes", 0)))}
        for row in included if not row.get("derived")
    ]
    entries += [{"path": row["path"], "kind": "excluded", "bytes": int(row.get("bytes", 0))} for row in excluded]
    entries += [{"path": row["path"], "kind": "missing", "bytes": 0} for row in missing]
    return sorted(entries, key=lambda row: (row["path"], row["kind"]))


def inventory_sha256(entries: list[dict]) -> str:
    """@brief 覆盖集合的摘要（不参与 ``generated_at``）。

    @param entries ``_inventory_entries`` 的输出。
    @return 十六进制摘要。
    """
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256(canonical.encode("utf-8"))


def _run_id(root: Path) -> str:
    """@brief 运行 id：在 ``runs/`` 下用相对路径，否则用目录名（不写绝对路径）。

    @param root 源 run 目录。
    @return run id。
    """
    from observe.detail import runs_root

    try:
        relative = root.resolve().relative_to(runs_root().resolve())
        return relative.as_posix()
    except (OSError, ValueError):
        return root.name


def _sanitize(text: str, root: Path, run_id: str) -> str:
    """@brief 把绝对路径改写成可复核的相对值。

    只用 ``resolve()`` 后的绝对根做替换：``RUN_DIR=.`` 时 ``str(root)`` 是 `.`，逐字符替换会把
    JSON 里的 ``1.0`` 变成 ``1conv2d/...`` 并破坏可解析性。

    @param text 原始文本。
    @param root 源 run 目录。
    @param run_id 运行 id。
    @return 净化后的文本。
    """
    from tcenv.target import REPO

    raw = str(root)
    absolute = str(Path(root).resolve())
    replacements: list[tuple[str, str]] = []

    def add(label: str, target: str) -> None:
        # 同一路径可能以原样或 JSON 转义（默认 json.dumps 会把中文写成 \uXXXX）出现。
        if not label or label in (".", "/") or label == target:
            return
        replacements.append((label, target))
        escaped = json.dumps(label)[1:-1]
        if escaped != label:
            replacements.append((escaped, target))

    # 源文件里写的是调用方给的路径；macOS 的 /tmp 是 /private/tmp 的链接，两者都要替换。
    if raw and Path(raw).is_absolute():
        add(raw, run_id)
    add(absolute, run_id)
    add(str(Path.home()), "$HOME")
    add(str(Path(REPO).resolve()), "<repo>")
    add(str(REPO), "<repo>")
    for old, new in replacements:
        text = text.replace(old, new)
    # 相对 RUN_DIR（例如 "."）时，源文件里可能写着另一条绝对别名（macOS 的 /tmp ↔ /private/tmp）。
    # 只要绝对路径以本场的 run id 结尾，就把它整体改写成 run id；边界排除 URL 的 `:`/`/`，
    # 否则 `http://example.com/conv2d/one` 会被错改成 `http:conv2d/one`。
    if "/" in run_id:
        text = re.sub(r'(?<![\w:./])(?:/[^\s"\'\\]+)?/' + re.escape(run_id), run_id, text)
    # 仍残留的绝对路径（别人家目录、/etc、/opt、/root 等）统一抹掉；净化与校验共用这条规则。
    return ABS_PATH_RE.sub("$ABS", text)


def _packed_bytes(data: bytes, root: Path, run_id: str, path: Path | None = None) -> tuple[bytes, bool]:
    """@brief 净化一份字节；不是 UTF-8 就 fail-closed，绝不写出未经净化的二进制。

    @param data 原始字节（可能是文件前缀）。
    @param root 源 run 目录。
    @param run_id 运行 id。
    @param path 源文件路径（只用于错误信息）。
    @return ``(packed, sanitized)``。
    @raises RuntimeError 输入不是 UTF-8 时抛出（导出失败，且不留下包）。
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"source is not UTF-8: {path or '<unknown>'}") from exc
    cleaned = _sanitize(text, root, run_id)
    if cleaned == text:
        return data, False
    return cleaned.encode("utf-8"), True


def _marker_needles(root: Path, run_id: str) -> list[bytes]:
    """@brief 判断“要不要为净化而读整份文件”的字节标记。

    @param root 源 run 目录。
    @param run_id 运行 id。
    @return 标记列表。
    """
    from tcenv.target import REPO

    needles = [str(Path(root).resolve()).encode("utf-8"), str(Path.home()).encode("utf-8"),
               str(Path(REPO).resolve()).encode("utf-8"), str(REPO).encode("utf-8")]
    needles += list(ABS_FAST_MARKERS)
    return [needle for needle in needles if needle]


def _stream_hash(path: Path, needles: list[bytes]) -> tuple[str, bool, int]:
    """@brief 第一遍流式哈希：分块更新 sha256，同时判断是否出现绝对路径标记。

    @param path 源文件。
    @param needles 绝对路径标记。
    @return ``(sha256, 是否命中标记, 字节数)``。
    """
    hasher = hashlib.sha256()
    total = 0
    hit = False
    carry = b""
    decoder = codecs.getincrementaldecoder("utf-8")()
    keep = max(8192, max((len(needle) for needle in needles), default=1))
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(READ_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
            total += len(chunk)
            try:
                decoder.decode(chunk)
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"source is not UTF-8: {path}") from exc
            window = carry + chunk
            if any(needle in window for needle in needles):
                hit = True
            if not hit and ABS_PATH_RE.search(window.decode("utf-8", errors="replace")):
                hit = True
            carry = window[-keep:]
    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"source is not UTF-8: {path}") from exc
    return hasher.hexdigest(), hit, total


def _pack_source(path: Path, root: Path, run_id: str) -> tuple[dict, bytes | None]:
    """@brief 读一份源文件并返回 manifest 字段与要写进包的字节（单文件，不跨文件持有）。

    普通文件先分块哈希；只有出现绝对路径标记时才读入内存净化。读失败不吞成空证据：
    ``OSError`` 向上抛，导出必须失败而不是打一个空成员。

    @param path 源文件。
    @param root 源 run 目录。
    @param run_id 运行 id。
    @return ``(manifest 字段, packed 字节或 None)``。
    """
    raw_size = path.stat().st_size
    needles = _marker_needles(root, run_id)
    if raw_size > MAX_FILE_BYTES:
        with path.open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES)
        packed, sanitized = _packed_bytes(raw, root, run_id, path)
        return {
            "bytes": len(packed),
            "sha256": _sha256(packed),
            "source_sha256": _sha256(raw),
            "source_bytes": raw_size,
            "sanitized": sanitized,
            "truncated": True,
        }, packed
    digest, hit, total = _stream_hash(path, needles)
    if not hit:
        # 没有绝对路径：不读进内存，第二遍直接从源文件流进 tar。
        return {
            "bytes": total,
            "sha256": digest,
            "source_sha256": digest,
            "source_bytes": total,
            "sanitized": False,
        }, None
    data = path.read_bytes()
    if _sha256(data) != digest:
        raise RuntimeError(f"source changed while reading: {path}")
    packed, sanitized = _packed_bytes(data, root, run_id, path)
    return {
        "bytes": len(packed),
        "sha256": _sha256(packed),
        "source_sha256": digest,
        "source_bytes": len(data),
        "sanitized": sanitized,
    }, packed


def _regular_file(path: Path) -> bool:
    """@brief 只接受普通文件；不跟随符号链接（派生元数据不能把外部内容带进包）。

    @param path 路径。
    @return 是普通文件且不是链接时为 True。
    """
    return path.is_file() and not path.is_symlink()


def _derived_verdict(root: Path, result: dict | None) -> dict | None:
    """@brief 派生发布判定；确认门会读的 run 文件只要有一个是链接，就放弃判定而不是跟随。

    @param root 源 run 目录。
    @param result 结果字典。
    @return 判定字典或 None。
    """
    if not isinstance(result, dict):
        return None
    names = ("confirm_report.json",)
    if any((root / name).is_symlink() for name in names):
        return None
    try:
        best_files = list(root.glob("best.*"))
    except OSError:
        best_files = []
    if any(path.is_symlink() for path in best_files):
        return None
    candidates = root / "candidates"
    if candidates.is_symlink():
        return None
    if candidates.is_dir():
        for child in candidates.iterdir():
            if child.is_symlink():
                return None
            if (child / "kernel.metal").is_symlink():
                return None
    try:
        from experiments.common import publication_reason

        return publication_reason(root, result)
    except Exception:  # noqa: BLE001 - derived metadata must not fail the export
        return None


def _run_metadata(root: Path) -> dict:
    """@brief 从 result/state/run_config 取 task/dtype/size/target（只作派生分片的键）。

    @param root 源 run 目录。
    @return 字段字典。
    """
    meta: dict = {}
    for name in ("result.json", "state.json", "run_config.json"):
        path = root / name
        if not _regular_file(path):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        for key in ("task", "dtype", "size"):
            if not meta.get(key) and doc.get(key) not in (None, ""):
                meta[key] = str(doc[key])
        target = doc.get("target_id")
        if not target and isinstance(doc.get("target"), dict):
            target = doc["target"].get("id")
        if not meta.get("target_id") and target:
            meta["target_id"] = str(target)
    return meta


def _publications_shard(root: Path, run_id: str) -> str:
    """@brief 派生 ``publications.json``：该 task 的目录史分片 + 当前发布判定（<2KB，含缩进）。

    @param root 源 run 目录。
    @param run_id 运行 id。
    @return JSON 文本。
    """
    meta = _run_metadata(root)
    records: list[dict] = []
    if all(meta.get(key) for key in ("target_id", "task", "dtype", "size")):
        try:
            from taic.kernel_catalog import catalog_dir

            history = catalog_dir(meta["target_id"], meta["task"], meta["dtype"], meta["size"]) / "history.jsonl"
            lines = history.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        except OSError:
            lines = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append({
                    key: record.get(key) for key in
                    ("updated_at", "confirmed_speedup", "run_id", "winner_label", "source_sha256")
                })
    verdict = None
    result_path = root / "result.json"
    result = None
    if _regular_file(result_path):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = None
    if isinstance(result, dict):
        verdict = _derived_verdict(root, result)
    document = {
        "schema_version": "1",
        "run_id": run_id,
        "target_id": meta.get("target_id", ""),
        "task": meta.get("task", ""),
        "dtype": meta.get("dtype", ""),
        "size": meta.get("size", ""),
        "catalog_history": records,
        "current_publication": verdict,
        "note": "derived at export: catalog shard for this task + the current publication verdict",
    }

    def render() -> str:
        return json.dumps(document, ensure_ascii=False, indent=1)

    while document["catalog_history"] and len(render().encode("utf-8")) > PUBLICATIONS_BYTES:
        document["catalog_history"].pop(0)
    if len(render().encode("utf-8")) > PUBLICATIONS_BYTES:
        document["catalog_history"] = []
        document["note"] = "derived at export"
    return render()


def _second_pass(row: dict, root: Path, run_id: str) -> tuple[bytes, str, int]:
    """@brief 第二遍重新打开源文件（或重算派生文件），返回要写进包的字节与当前源指纹。

    @param row manifest 的 files 行。
    @param root 源 run 目录。
    @param run_id 运行 id。
    @return ``(packed 字节, 源哈希, 源字节数)``。
    """
    if row.get("derived"):
        packed = _sanitize(_publications_shard(root, run_id), root, run_id).encode("utf-8")
        return packed, _sha256(packed), len(packed)
    path = root / str(row["path"])
    source_bytes = int(row["source_bytes"])
    if _source_size(path) != source_bytes:
        raise RuntimeError(f"source size changed between passes: {row['path']}")
    if row.get("truncated"):
        with path.open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES)
        # 截断只写前缀，读完之后必须再核一次大小；追加哪怕一个字节也要失败。
        if _source_size(path) != source_bytes:
            raise RuntimeError(f"source size changed between passes: {row['path']}")
        packed, _sanitized = _packed_bytes(raw, root, run_id, path)
        if _source_size(path) != source_bytes:
            raise RuntimeError(f"source size changed between passes: {row['path']}")
        return packed, _sha256(raw), source_bytes
    data = path.read_bytes()
    if _source_size(path) != source_bytes:
        raise RuntimeError(f"source size changed between passes: {row['path']}")
    packed, _sanitized = _packed_bytes(data, root, run_id, path)
    if _source_size(path) != source_bytes:
        raise RuntimeError(f"source size changed between passes: {row['path']}")
    return packed, _sha256(data), source_bytes


def build(root: Path, *, run_id: str | None = None) -> dict:
    """@brief 第一遍：流式算清单（逐文件读写后丢弃字节，不把整包留在内存）。

    读失败不变成空证据：该文件以 ``unreadable`` 行进 ``files``，``verify`` 会拒绝它；导出也不会为
    它写空成员。

    @param root 源 run 目录。
    @param run_id 覆盖运行 id；``verify --against`` 用包里记录的值，避免环境路径影响派生分片。
    @return manifest。
    """
    root = Path(root)
    run_id = run_id or _run_id(root)
    included, excluded, missing = source_entries(root)
    files: list[dict] = []
    truncated: list[dict] = []
    stamps: dict[str, str] = {}
    for entry in included:
        try:
            row, _packed = _pack_source(entry["file"], root, run_id)
        except OSError as exc:
            row = {
                "bytes": 0, "sha256": _sha256(b""), "source_sha256": _sha256(b""), "source_bytes": 0,
                "sanitized": False,
                "unreadable": f"{type(exc).__name__}: {_sanitize(str(exc), root, run_id)}",
            }
        row["path"] = entry["path"]
        files.append(row)
        if row.get("truncated"):
            truncated.append({
                "path": row["path"],
                "full_bytes": int(row["source_bytes"]),
                "prefix_bytes": row["bytes"],
                "prefix_sha256": row["sha256"],
                "full_sha256": None,
            })
        try:
            stat = entry["file"].stat()
            stamps[row["path"]] = f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            stamps[row["path"]] = "0:0"
    source_publications_exists = (root / PUBLICATIONS_NAME).exists() or (root / PUBLICATIONS_NAME).is_symlink()
    if not source_publications_exists and not any(row["path"] == PUBLICATIONS_NAME for row in files):
        derived = _sanitize(_publications_shard(root, run_id), root, run_id).encode("utf-8")
        files.append({
            "path": PUBLICATIONS_NAME,
            "bytes": len(derived),
            "sha256": _sha256(derived),
            "source_sha256": _sha256(derived),
            "source_bytes": len(derived),
            "sanitized": False,
            "derived": True,
        })
    files.sort(key=lambda row: row["path"])
    entries = _inventory_entries(files, excluded, missing)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": int(time.time()),
        "taic_commit": os.environ.get("TAIC_COMMIT", ""),
        "index_schema_version": _schema_version(),
        "source": {
            "detail_stamp": {"fingerprints": stamps, "note": "_detail_stamp 由 (mtime_ns,size) 指纹拼成，仅供参考，不可从包内重算"},
        },
        "inventory_sha256": inventory_sha256(entries),
        "files": files,
        "excluded": excluded,
        "missing": missing,
        "truncated": truncated,
    }


def _schema_version() -> int:
    """@brief 导出时使用的索引 schema 版本（只作元数据）。

    @return ``INDEX_SCHEMA_VERSION``；导入失败时为 0。
    """
    try:
        from observe.index import INDEX_SCHEMA_VERSION

        return int(INDEX_SCHEMA_VERSION)
    except Exception:  # noqa: BLE001 - 元数据不该让导出失败
        return 0


def _source_size(path: Path) -> int:
    """@brief 第二遍写前核对的当前大小（单独成函数，便于注入“写前已增长”的测试）。

    @param path 源文件。
    @return 字节数。
    """
    return path.stat().st_size


def _add_bytes(handle: tarfile.TarFile, name: str, data: bytes) -> None:
    """@brief 把一个字节块写成确定性的 tar 成员。

    @param handle 打开的 tar。
    @param name 成员名。
    @param data 内容。
    """
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    handle.addfile(info, io.BytesIO(data))


def _sources_unchanged(manifest: dict, root: Path) -> None:
    """@brief 第二遍开始前重算源清单；included/excluded/missing 任一变化都让导出失败。

    @param manifest 第一遍的清单。
    @param root 源 run 目录。
    """
    # 第一遍已读不到的文件会在 files 里带 unreadable；它不会再变成可打包的成员，verify 会拒绝。
    # 这类包已经注定不可验收，不必再比一次源清单，让包照常写完后由 verify 明确报错。
    if any(row.get("unreadable") for row in manifest["files"]):
        return
    included_now, excluded_now, missing_now = source_entries(root)
    if excluded_now != manifest["excluded"] or missing_now != manifest["missing"]:
        raise RuntimeError("source inventory changed between passes")
    manifest_paths = sorted(row["path"] for row in manifest["files"] if not row.get("derived"))
    if sorted(entry["path"] for entry in included_now) != manifest_paths:
        raise RuntimeError("source file set changed between passes")


class _HashingReader:
    """@brief 边读边算 sha256 的只读包装，供 ``tarfile.addfile`` 从源文件直接流进包。

    @param handle 源文件句柄。
    """

    def __init__(self, handle, limit: int) -> None:
        """@brief 记录句柄与要读的字节数。

        @param handle 源文件句柄。
        @param limit 期望字节数（清单里的 bytes）。
        """
        self.handle = handle
        self.limit = limit
        self.count = 0
        self.hasher = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        """@brief 读一段并更新摘要。

        @param size 最多读多少字节；-1 表示剩余全部。
        @return 字节。
        """
        remaining = self.limit - self.count
        if remaining <= 0:
            return b""
        want = remaining if size is None or size < 0 else min(size, remaining)
        chunk = self.handle.read(want)
        self.count += len(chunk)
        self.hasher.update(chunk)
        return chunk


def _stream_member(handle: tarfile.TarFile, row: dict, root: Path) -> None:
    """@brief 第二遍把未净化的源文件直接流进 tar，并在写完时复核摘要。

    @param handle 打开的 tar。
    @param row manifest 行。
    @param root 源 run 目录。
    """
    path = root / str(row["path"])
    size_now = _source_size(path)
    if size_now != int(row["source_bytes"]):
        raise RuntimeError(f"source size changed between passes: {row['path']}")
    info = tarfile.TarInfo(str(row["path"]))
    info.size = int(row["bytes"])
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    trailing = b""
    with path.open("rb") as source:
        reader = _HashingReader(source, int(row["bytes"]))
        handle.addfile(info, reader)
        # 清单只记录到旧 bytes；追加一个字节也必须被发现，不能只靠 --against。
        trailing = source.read(1)
    digest = reader.hasher.hexdigest()
    if trailing or reader.count != int(row["bytes"]) or digest != row["sha256"] or digest != row["source_sha256"]:
        raise RuntimeError(f"bytes changed between passes: {row['path']}")


def _check_member(row: dict, packed: bytes) -> None:
    """@brief 复核一份第二遍字节与清单一致；源在两遍之间变了就让导出失败。

    @param row manifest 行。
    @param packed 第二遍重新读出来的字节。
    """
    path = str(row["path"])
    if row.get("derived"):
        if len(packed) != int(row["bytes"]) or _sha256(packed) != row["sha256"]:
            raise RuntimeError(f"derived evidence changed between passes: {path}")
        return
    if row.get("unreadable"):
        raise RuntimeError(f"source became readable between passes: {path}")
    if _sha256(packed) != row["sha256"] or len(packed) != int(row["bytes"]):
        raise RuntimeError(f"bytes changed between passes: {path}")


def export(root: Path, out: Path, *, fmt: str = "tar.gz") -> dict:
    """@brief 两遍流式导出：第一遍只建清单，第二遍重新打开源文件、边写边复核。

    第二遍拿到的源哈希必须与第一遍一致；包直接流到目标文件（tar 不再在 BytesIO 里拼），
    所以导出中途源被改写时不会留下一个“看起来成功”的包。

    @param root 源 run 目录。
    @param out 输出包路径。
    @param fmt ``tar.gz`` 或 ``json``。
    @return manifest。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"run directory not found: {root}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build(root)
    _sources_unchanged(manifest, root)
    temporary = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    try:
        if fmt == "tar.gz":
            with temporary.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                    with tarfile.open(fileobj=gz, mode="w") as handle:
                        _add_bytes(handle, MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))
                        for row in manifest["files"]:
                            if row.get("unreadable"):
                                continue
                            if not row.get("derived") and not row.get("truncated") and not row.get("sanitized"):
                                # 普通源文件：直接从磁盘流进 tar，边写边算摘要。
                                _stream_member(handle, row, root)
                                continue
                            packed, source_sha, full_bytes = _second_pass(row, root, manifest["run_id"])
                            if not row.get("derived"):
                                if source_sha != row["source_sha256"]:
                                    raise RuntimeError(f"source changed between passes: {row['path']}")
                                if row.get("truncated") and full_bytes != int(row["source_bytes"]):
                                    raise RuntimeError(f"source size changed between passes: {row['path']}")
                            _check_member(row, packed)
                            if not row.get("derived") and _source_size(root / str(row["path"])) != int(row["source_bytes"]):
                                raise RuntimeError(f"source size changed between passes: {row['path']}")
                            _add_bytes(handle, row["path"], packed)
        elif fmt == "json":
            contents: dict[str, str] = {}
            for row in manifest["files"]:
                if row.get("unreadable"):
                    continue
                packed, source_sha, full_bytes = _second_pass(row, root, manifest["run_id"])
                if not row.get("derived"):
                    if source_sha != row["source_sha256"]:
                        raise RuntimeError(f"source changed between passes: {row['path']}")
                    if row.get("truncated") and full_bytes != int(row["source_bytes"]):
                        raise RuntimeError(f"source size changed between passes: {row['path']}")
                _check_member(row, packed)
                contents[row["path"]] = packed.decode("utf-8")
            for row in manifest["files"]:
                if row.get("derived") or row.get("unreadable"):
                    continue
                if _source_size(root / str(row["path"])) != int(row["source_bytes"]):
                    raise RuntimeError(f"source size changed between passes: {row['path']}")
            temporary.write_text(
                json.dumps({"manifest": manifest, "files": contents}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        else:
            raise ValueError(f"unknown format: {fmt}")
        os.replace(temporary, out)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return manifest


def _safe_path(path: object) -> bool:
    """@brief 包内/清单路径必须是相对、非空、无 ``..``，且不能占用 ``MANIFEST.json``。

    @param path 路径。
    @return 安全时为 True。
    """
    if not isinstance(path, str) or not path or path == MANIFEST_NAME:
        return False
    if path.startswith("/") or path.startswith("\\"):
        return False
    parts = Path(path).parts
    return ".." not in parts and "." not in parts


def _hex64(value: object) -> bool:
    """@brief 是否是 64 位小写十六进制摘要。

    @param value 值。
    @return 是时为 True。
    """
    return isinstance(value, str) and bool(HEX64.match(value))


def _validate_manifest(manifest: dict) -> list[str]:
    """@brief 严格校验清单形状；畸形清单不能因为“files 为空”而被当成通过。

    @param manifest 包里的清单。
    @return 失败项列表。
    """
    failures: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not an object"]
    if manifest.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported manifest schema_version")
    if not isinstance(manifest.get("run_id"), str) or not manifest.get("run_id"):
        failures.append("manifest run_id is missing")
    if not _hex64(manifest.get("inventory_sha256")):
        failures.append("manifest inventory_sha256 is not a digest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        failures.append("manifest files are missing")
        return failures
    seen: set[str] = set()
    file_paths: set[str] = set()
    for row in files:
        if not isinstance(row, dict):
            failures.append("manifest files contain a non-object row")
            continue
        path = row.get("path")
        if not _safe_path(path):
            failures.append(f"unsafe file path: {path!r}")
            continue
        if path in seen:
            failures.append(f"duplicate path: {path}")
        seen.add(path)
        file_paths.add(path)
        if not isinstance(row.get("bytes"), int) or isinstance(row.get("bytes"), bool) or row["bytes"] < 0:
            failures.append(f"bad bytes for {path}")
        if not _hex64(row.get("sha256")) or not _hex64(row.get("source_sha256")):
            failures.append(f"bad digest for {path}")
        if not isinstance(row.get("source_bytes"), int) or row["source_bytes"] < 0:
            failures.append(f"bad source_bytes for {path}")
        if not isinstance(row.get("derived", False), bool):
            failures.append(f"bad derived flag for {path}")
        if "unreadable" in row and not isinstance(row["unreadable"], str):
            failures.append(f"bad unreadable marker for {path}")
    for name in ("excluded", "missing"):
        rows = manifest.get(name)
        if not isinstance(rows, list):
            failures.append(f"manifest {name} is not a list")
            continue
        for row in rows:
            if not isinstance(row, dict) or not _safe_path(row.get("path")):
                failures.append(f"unsafe {name} path: {row!r}")
                continue
            path = row["path"]
            if path in seen:
                failures.append(f"duplicate path across manifest sections: {path}")
            seen.add(path)
            if name == "excluded":
                if not isinstance(row.get("bytes"), int) or row["bytes"] < 0:
                    failures.append(f"bad excluded bytes for {path}")
                if not isinstance(row.get("files"), int) or row["files"] < 0:
                    failures.append(f"bad excluded files for {path}")
    truncated = manifest.get("truncated")
    if not isinstance(truncated, list):
        failures.append("manifest truncated is not a list")
    else:
        for row in truncated:
            if not isinstance(row, dict) or row.get("path") not in file_paths:
                failures.append(f"truncated row without file: {row!r}")
        by_path = {row.get("path"): row for row in files if isinstance(row, dict)}
        for row in truncated:
            if not isinstance(row, dict):
                continue
            source = by_path.get(row.get("path"))
            if source is None:
                continue
            if row.get("prefix_bytes") != source.get("bytes") or row.get("prefix_sha256") != source.get("sha256"):
                failures.append(f"truncated prefix mismatch: {row.get('path')}")
            if not isinstance(row.get("full_bytes"), int) or row["full_bytes"] < 0:
                failures.append(f"bad full_bytes: {row.get('path')}")
            if row.get("full_sha256") is not None and not _hex64(row.get("full_sha256")):
                failures.append(f"bad full_sha256: {row.get('path')}")
    serialized = json.dumps(manifest, ensure_ascii=False)
    if ABS_PATH_RE.search(serialized):
        failures.append("absolute path in MANIFEST.json")
    return failures


def _load(package: Path) -> tuple[dict, dict[str, bytes]]:
    """@brief 读一个包：manifest + 成员字节（``.json`` 或 ``.tar.gz``）。

    @param package 包路径。
    @return ``(manifest, {路径: 字节})``。
    """
    members: dict[str, bytes] = {}
    manifest: dict | None = None
    if package.suffix == ".json":
        document = json.loads(package.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("manifest"), dict):
            raise ValueError("json package has no manifest")
        manifest = document["manifest"]
        raw_files = document.get("files") or {}
        if not isinstance(raw_files, dict):
            raise ValueError("json package files are not an object")
        for path, text in raw_files.items():
            if not _safe_path(path) or not isinstance(text, str):
                raise ValueError(f"unsafe json member: {path!r}")
            members[str(path)] = text.encode("utf-8")
        return manifest, members
    with tarfile.open(package, "r:gz") as handle:
        seen_members: set[str] = set()
        for item in handle.getmembers():
            if item.issym() or item.islnk():
                raise ValueError(f"link member is not allowed: {item.name}")
            if not item.isfile():
                continue
            if not _safe_path(item.name) and item.name != MANIFEST_NAME:
                raise ValueError(f"unsafe member: {item.name}")
            if item.name in seen_members:
                raise ValueError(f"duplicate member: {item.name}")
            seen_members.add(item.name)
            data = handle.extractfile(item)
            raw = data.read() if data is not None else b""
            if item.name == MANIFEST_NAME:
                if manifest is not None:
                    raise ValueError("duplicate MANIFEST.json")
                manifest = json.loads(raw.decode("utf-8"))
            else:
                members[item.name] = raw
    if manifest is None:
        raise ValueError(f"{MANIFEST_NAME} missing from {package}")
    return manifest, members


def verify(package: Path, *, against: Path | None = None) -> dict:
    """@brief 校验一个包：成员摘要、清单覆盖、无绝对路径、截断只验前缀。

    @param package 包路径。
    @param against 源 run 目录；给出时再按同一套规则重算源清单并比对。
    @return ``{"ok": bool, "failures": [...], ...}``。
    """
    package = Path(package)
    try:
        manifest, members = _load(package)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        return {
            "ok": False, "failures": [f"unreadable package: {exc}"], "files": 0,
            "verified": "full", "prefix_verified": [],
        }
    failures = _validate_manifest(manifest)
    if failures:
        return {
            "ok": False, "failures": failures, "files": 0, "verified": "full",
            "prefix_verified": [], "inventory_sha256": manifest.get("inventory_sha256") if isinstance(manifest, dict) else None,
        }
    prefix_only: list[str] = []
    files = {str(row["path"]): row for row in manifest["files"]}
    truncated = {str(row.get("path")): row for row in manifest.get("truncated", []) if isinstance(row, dict)}
    for path, row in sorted(files.items()):
        if row.get("unreadable"):
            failures.append(f"unreadable source: {path}")
            continue
        data = members.get(path)
        if data is None:
            failures.append(f"missing member: {path}")
            continue
        digest = _sha256(data)
        if path in truncated:
            item = truncated[path]
            if len(data) != int(item.get("prefix_bytes", -1)):
                failures.append(f"prefix size mismatch: {path}")
            if digest != item.get("prefix_sha256"):
                failures.append(f"prefix digest mismatch: {path}")
            prefix_only.append(path)
            continue
        if len(data) != int(row.get("bytes", -1)):
            failures.append(f"size mismatch: {path}")
        if digest != row.get("sha256"):
            failures.append(f"digest mismatch: {path}")
    for name in sorted(set(members) - set(files)):
        failures.append(f"unlisted member: {name}")
    entries = _inventory_entries(list(files.values()), manifest.get("excluded") or [], manifest.get("missing") or [])
    if inventory_sha256(entries) != manifest.get("inventory_sha256"):
        failures.append("inventory_sha256 mismatch")
    for path, data in members.items():
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"member is not UTF-8: {path}")
            continue
        if ABS_PATH_RE.search(decoded):
            failures.append(f"absolute path in member: {path}")
    state = members.get("state.json")
    if state:
        try:
            document = json.loads(state.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            document = None
        if isinstance(document, dict) and "run_dir" in document and document["run_dir"] != manifest.get("run_id"):
            failures.append("state.json run_dir is not the run id")
    if against is not None:
        try:
            source_manifest = build(Path(against), run_id=manifest.get("run_id"))
        except (OSError, RuntimeError) as exc:
            failures.append(f"source unreadable: {exc}")
        else:
            source_files = {row["path"]: row for row in source_manifest["files"] if not row.get("derived")}
            package_files = {path: row for path, row in files.items() if not row.get("derived")}
            if set(source_files) != set(package_files):
                failures.append("source file set changed since export")
            for path in sorted(set(source_files) & set(package_files)):
                if source_files[path].get("source_sha256") != package_files[path].get("source_sha256"):
                    failures.append(f"source bytes changed: {path}")
            if source_manifest.get("inventory_sha256") != manifest.get("inventory_sha256"):
                failures.append("source inventory changed since export")
    return {
        "ok": not failures,
        "failures": failures,
        "files": len(files),
        # 截断项只承诺前缀一致；输出里必须写清楚，不能让人以为验过全文件。
        "verified": "prefix" if prefix_only else "full",
        "prefix_verified": prefix_only,
        "inventory_sha256": manifest.get("inventory_sha256"),
    }
