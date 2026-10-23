"""Parse one session's FEEDBACK.md and render peer tips for the next session of the same run.

The search records what the agent wrote. It does not treat that text as a measurement.
Peer tips are recomputed from this run's ``feedback.jsonl`` and never read another run.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

FORMAT = "agent_feedback"
VERSION = 1
# Bumped when in-memory repair of old feedback.jsonl lines changes what the board shows.
# observe/index.py appends this to the run fingerprint so a finished run is read again.
PARSER_VERSION = "fbv4"
FILE_CAP = 16384
RAW_CAP = 4096
TIP_CHARS = 250
TIPS_CHARS = 1200
VERIFIED_TIPS_CHARS = 1800
TIP_KEEP = 8
UNVERIFIED_SPAN = 2  # sessions after the one that first wrote an unverified premise; the third session no longer sees it
_PLACEHOLDERS = frozenset({"无", "没有", "空", "none", "na", "n/a"})
# 「无（预算已尽）」「没有阻塞性的事。本轮不需要人介入」这类条目不是请求，也不该进待审。
# 只有缺席句后面还带着明确动作请求时才保留；文件路径和数字不算请求，审计的 7 条都带解释。
_ABSENT = re.compile(
    r"^(?:无|没有|暂无|无阻塞|没有阻塞(?:性)?(?:的)?(?:事情|事|问题)?|不需要|无需)(?:[，,。；;：:\s（(]|$)",
    re.IGNORECASE,
)
_REQUEST = re.compile(
    r"(?:请|建议|希望|应当|应该|能否|能不能|要不要|请加|请改|改成|改一下|修改|修复|增加|去掉|删除|补上|支持)",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[。；;.!！?？\n]")

SECTIONS = (
    "需要帮助",
    "对系统的建议",
    "卡住的原因",
    "给下一棒的避坑提示",
    "被否决的前提",
)
SYSTEM_SECTIONS = ("需要帮助", "对系统的建议")
DIAGNOSTIC_SECTIONS = ("卡住的原因",)
PEER_SECTIONS = ("给下一棒的避坑提示", "被否决的前提")
SECTION_CAPS = {
    "需要帮助": 1200,
    "对系统的建议": 1200,
    "卡住的原因": 800,
    "给下一棒的避坑提示": 500,
    "被否决的前提": 1000,
}
KINDS = ("prompt", "skill", "tool", "rule", "decision", "none")
_HEADER = re.compile(r"^#\s+(" + "|".join(SECTIONS) + r")\s*$")
_KIND = re.compile(r"^kind:\s*(.*)$", re.IGNORECASE)
_VERSION = re.compile(r"^v:\s*(\S+)\s*$", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE = re.compile(r"\s+")
_MEASURED = re.compile(r"\[已实测:\s*([^|\]]+?)\s*\|\s*([^|\]]+?)\s*\|\s*([^\]]+?)\s*\]")
_MARK = re.compile(r"\[(已实测|未验证|分析否决|推理否决)([^\]]*)\]")
_FILE_TOKEN = re.compile(r"[A-Za-z0-9_./-]+\.(?:py|metal|md|json|sh|txt)")
def _wiped_premise(text: str) -> bool:
    """@brief 判断一条前提的正文是否已被降级擦掉。

    被擦掉的前提正文以哨兵 ``[未验证: 探针文件不存在]`` 开头，可从存档小节里恢复。

    @param text 条目正文。
    @return 正文以擦除哨兵开头时为 ``True``。
    """
    return str(text or "").strip().strip("`").strip().startswith("[未验证: 探针文件不存在]")


_ROUND = re.compile(r"^r(\d+)\.p\d+$")


def read_feedback_text(path: Path) -> tuple[str, bool]:
    """@brief 读 FEEDBACK.md 的末尾 16384 个字符，并告知文件是否更长。

    会话是往下追加的。如果切掉末尾保留开头，最新的几行就丢了。

    @param path FEEDBACK.md 的路径。
    @return ``(文本, 是否截断)``；文件不存在时是 ``("", False)``。
    """
    if not path.is_file():
        return "", False
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) > FILE_CAP:
        return data[-FILE_CAP:], True
    return data, False


def empty_feedback() -> dict:
    """@brief 一份空的解析结果，字段与 ``parse_feedback`` 完全一致。

    @return 空反馈字典：没有小节、没有条目、没有截断。
    """
    return {
        "format": FORMAT,
        "v": VERSION,
        "kinds": "none",
        "truncated": False,
        "truncated_sections": [],
        "placeholder_sections": [],
        "placeholder_items": 0,
        "unrecognized_version": False,
        "sections": {},
        "section_names": [],
        "items": [],
        "unverified_premises": 0,
        "raw": "",
        "chars": 0,
    }


def parse_feedback(text: str, *, work: Path | None = None, repo: Path | None = None, file_over: bool = False) -> dict:
    """@brief 切分一份 FEEDBACK.md。

    版本号不认识的文档保留原文、不给任何条目。小节上限在切分之后才生效，于是一个很长的系统小节无法把
    给下一棒的避坑提示小节挤没。

    @param text FEEDBACK.md 的文本（通常由 ``read_feedback_text`` 读出）。
    @param work 本次会话的工作目录，用于验证前提里的探针文件。
    @param repo 仓库根目录，作为探针路径的另一个解析根。
    @param file_over 调用方已知原文件超过上限（例如已经被截断过）。
    @return 解析结果字典：``sections``/``items``/``truncated_sections``/``unverified_premises`` 等。
    """
    body = text or ""
    over = bool(file_over) or len(body) > FILE_CAP
    if len(body) > FILE_CAP:
        body = body[-FILE_CAP:]
    version = _declared_version(body)
    if version is not None and version != VERSION:
        raw = body[:RAW_CAP]
        return {
            "format": FORMAT,
            "v": version,
            "kinds": "none",
            "truncated": over or len(body) > RAW_CAP,
            "truncated_sections": [],
            "placeholder_sections": [],
            "placeholder_items": 0,
            "unrecognized_version": True,
            "sections": {},
            "section_names": [],
            "items": [],
            "unverified_premises": 0,
            "raw": raw,
            "chars": len(body),
        }
    kinds = _kinds(body)
    sections, truncated_sections = _sections(body)
    items = []
    unverified = 0
    placeholder_sections = []
    placeholder_items = 0
    for name in SECTIONS:
        chunk = sections.get(name, "")
        if not chunk.strip():
            continue
        if _is_placeholder(chunk):
            placeholder_sections.append(name)
            placeholder_items += 1
            continue
        for piece in _pieces(chunk):
            if _is_placeholder(piece):
                placeholder_sections.append(name)
                placeholder_items += 1
                continue
            item = _item(name, piece, work=work, repo=repo)
            if item["premise"] in ("unverified", "downgraded"):
                unverified += 1
            items.append(item)
    raw = body[:RAW_CAP]
    truncated = over or bool(truncated_sections) or len(body) > RAW_CAP
    names = [name for name in SECTIONS if sections.get(name, "").strip()]
    return {
        "format": FORMAT,
        "v": VERSION,
        "kinds": kinds,
        "truncated": truncated,
        "truncated_sections": truncated_sections,
        "placeholder_sections": sorted(set(placeholder_sections), key=SECTIONS.index),
        "placeholder_items": placeholder_items,
        "unrecognized_version": False,
        "sections": {name: sections[name] for name in names},
        "section_names": names,
        "items": items,
        "unverified_premises": unverified,
        "raw": raw,
        "chars": len(body),
    }


def max_seq(path: Path) -> int:
    """@brief 取反馈日志里最大的 seq。

    坏行（包括留在文件中间的那种）直接跳过。

    @param path ``feedback.jsonl`` 的路径。
    @return 最大的 seq；文件不存在或一行都读不懂时为 0。
    """
    if not path.is_file():
        return 0
    highest = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        row = _load_line(line)
        if row is None:
            continue
        seq = row.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            highest = max(highest, seq)
    return highest


def load_records(path: Path) -> list[dict]:
    """@brief 读出所有完整记录，坏行跳过。

    文件不会被重写。空 digest、被降级擦掉的前提正文、缺失的 ``truncated_sections`` 都在返回的副本上补齐。

    @param path ``feedback.jsonl`` 的路径。
    @return 修复后的记录列表；文件不存在时为空列表。
    """
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        row = _load_line(line)
        if row is not None and row.get("format") == FORMAT:
            rows.append(_repair_record(row, path.parent))
    return rows


def read_appended_records(path: Path, offset: int) -> tuple[list[tuple[int, dict]], int]:
    """@brief 读出 ``offset`` 之后完整行里的记录，并带上每行的字节位置。

    日志是只追不改的。返回的 offset 停在最后一个换行之后，所以尾巴上的半行会留给下一次调用。
    ``offset`` 超过文件大小意味着文件被替换过，从 0 重新读。

    @param path ``feedback.jsonl`` 的路径。
    @param offset 起始字节位置。
    @return ``(记录列表, 新 offset)``；每项为 ``(行字节位置, 记录)``。
    """
    if not path.is_file():
        return [], 0
    size = path.stat().st_size
    if offset < 0 or offset > size:
        offset = 0
    if offset == size:
        return [], size
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        return [], offset
    rows = []
    pos = 0
    blob = data[: last_nl + 1]
    while pos < len(blob):
        nl = blob.find(b"\n", pos)
        if nl < 0:
            break
        raw = blob[pos:nl]
        line_at = offset + pos
        pos = nl + 1
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode("utf-8", errors="replace")
        row = _load_line(line)
        if row is not None and row.get("format") == FORMAT:
            rows.append((line_at, _repair_record(row, path.parent)))
    return rows, offset + pos


def peer_tips_for(run_dir: Path, *, now: int | None = None) -> str:
    """@brief 为即将开始的会话生成提示。

    ``now`` 就是那一场的序号。每一场结束的会话都有序号，包括一句反馈都没留的那种。未验证或已降级的前提
    只给接下来两场看，第三场就不再见它；重述不会改写它首次出现的那一场。一旦某条被验证过，它就一直是验证
    状态，后来降级的副本不会顶掉那段正文。验证过的行用自己的字符预算，不占普通八条的位置。

    @param run_dir 运行目录。
    @param now 即将开始的会话序号；缺省取已记录最大序号加一。
    @return 多行提示文本；没有任何记录时为空字符串。
    """
    records = load_records(run_dir / "feedback.jsonl")
    if not records:
        return ""
    numbered = [int(row["seq"]) for row in records if isinstance(row.get("seq"), int)]
    if now is None:
        now = (max(numbered) + 1) if numbered else 1
    grouped: dict[str, dict] = {}
    for row in records:
        seq = row.get("seq")
        if not isinstance(seq, int):
            continue
        for index, item in enumerate(row.get("items") or []):
            if not isinstance(item, dict) or item.get("section") not in PEER_SECTIONS:
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            digest = str(item.get("digest") or "")
            if not digest:
                digest = f"undigested-{seq}-{index}"
            premise = item.get("premise") or ""
            current = grouped.get(digest)
            if current is None:
                current = {
                    "first_seq": seq,
                    "seq": seq,
                    "section": item.get("section"),
                    "premise": premise,
                    "text": text,
                    "row": row,
                    "verified_seq": None,
                    "verified_text": "",
                    "verified_row": None,
                }
                grouped[digest] = current
            else:
                current["seq"] = seq
                current["text"] = text
                current["row"] = row
                current["section"] = item.get("section")
                current["premise"] = premise
            if premise == "verified" and item.get("section") == "被否决的前提":
                current["verified_seq"] = seq
                current["verified_text"] = text
                current["verified_row"] = row
    verified = []
    active = []
    ordinary = []
    for entry in grouped.values():
        if entry["verified_row"] is not None:
            verified.append({
                "seq": entry["verified_seq"],
                "text": entry["verified_text"],
                "row": entry["verified_row"],
                "section": "被否决的前提",
                "premise": "verified",
                "first_seq": entry["first_seq"],
            })
            continue
        if entry["section"] == "被否决的前提" and entry["premise"] in ("unverified", "downgraded"):
            # Shown on the next two sessions: ages 1 and 2. Age 3 is the third session and is dropped.
            if 0 < now - entry["first_seq"] <= UNVERIFIED_SPAN:
                active.append(entry)
            continue
        ordinary.append(entry)
    verified.sort(key=lambda entry: (entry["seq"], entry["text"]))
    active.sort(key=lambda entry: (entry["seq"], entry["text"]))
    ordinary.sort(key=lambda entry: entry["seq"], reverse=True)
    if len(active) > TIP_KEEP:
        active = active[-TIP_KEEP:]
    rest = active + ordinary[: max(0, TIP_KEEP - len(active))]
    verified_lines = _fit_lines(verified, VERIFIED_TIPS_CHARS, drop_oldest=True)
    rest_lines = _fit_rest(rest, len(active))
    return "\n".join(verified_lines + rest_lines)


def round_of(label: str) -> int | None:
    """@brief 从 ``rN.pM`` 形式的标签里取出轮次 N。

    @param label 候选标签。
    @return 轮次；标签不是 ``rN.pM`` 形式时为 ``None``。
    """
    match = _ROUND.fullmatch(label or "")
    if not match:
        return None
    return int(match.group(1))


def _declared_version(text: str) -> int | None:
    """@brief 取文档首个非空行声明的版本号。

    首行不是版本行，或版本号不是整数时，都当作「没有声明」处理。

    @param text FEEDBACK.md 文本。
    @return 声明的版本号；没声明或读不懂时为 ``None``。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _VERSION.match(stripped)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _kinds(text: str) -> str:
    """@brief 读 ``kind:`` 行并归一成逗号分隔的合法类别名。

    不认识的取值和 ``none`` 都被丢掉；一个都不剩时返回 ``none``。

    @param text FEEDBACK.md 文本。
    @return 形如 ``prompt,skill`` 的类别串，或 ``none``。
    """
    found = None
    for line in text.splitlines():
        match = _KIND.match(line.strip())
        if match:
            found = match.group(1)
            break
    if found is None:
        return "none"
    kept = []
    for part in re.split(r"[,，\s]+", found.strip()):
        name = part.strip().lower()
        if name in KINDS and name != "none" and name not in kept:
            kept.append(name)
    return ",".join(kept) if kept else "none"


def _sections(text: str) -> tuple[dict[str, str], list[str]]:
    """@brief 按二级标题把文本切成已知的五个小节。

    每个小节各自有字符上限，超长时留下末尾（小节是往下追加的，最新的一条才该进下一场）。

    @param text FEEDBACK.md 文本。
    @return ``(小节正文, 被截断的小节名列表)``。
    """
    current = None
    chunks: dict[str, list[str]] = {name: [] for name in SECTIONS}
    for line in text.splitlines():
        match = _HEADER.match(line.strip())
        if match:
            current = match.group(1)
            continue
        if current is not None:
            chunks[current].append(line)
    out = {}
    truncated = []
    for name, lines in chunks.items():
        body = "\n".join(lines).strip()
        cap = SECTION_CAPS[name]
        if len(body) > cap:
            # 小节是往下追加的，超长时留下末尾，最新的一条才进下一场。
            body = body[-cap:]
            truncated.append(name)
        out[name] = body
    return out, truncated


def _pieces(body: str) -> list[str]:
    """@brief 把一个小节拆成条目。

    有 ``- `` 项目符号时按项目拆（续行归到上一条）；否则整段算一条。

    @param body 小节正文。
    @return 条目文本列表；正文为空时为空列表。
    """
    blocks: list[str] = []
    current: list[str] | None = None
    saw_bullet = False
    for line in body.splitlines():
        if line.startswith("- "):
            saw_bullet = True
            if current is not None:
                blocks.append("\n".join(current).strip())
            current = [line[2:]]
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        blocks.append("\n".join(current).strip())
    if saw_bullet:
        return [block for block in blocks if block]
    stripped = body.strip()
    return [stripped] if stripped else []


def _item(section: str, text: str, *, work: Path | None, repo: Path | None) -> dict:
    """@brief 把一段条目正文变成结构化条目。

    digest 取的是 agent 写下那句话本身；降级只改后面的标签，不得改 digest。

    @param section 所属小节名。
    @param text 条目正文。
    @param work 会话工作目录，用于验证探针文件。
    @param repo 仓库根目录，作为另一个解析根。
    @return 含 ``section``/``audience``/``text``/``digest``/``premise`` 的条目。
    """
    audience = "peer" if section in PEER_SECTIONS else "diagnostic" if section in DIAGNOSTIC_SECTIONS else "system"
    shown = text.strip()
    premise = ""
    # Digest the sentence the agent wrote. Downgrade rewrites the label afterwards and must not change the digest.
    digest = _digest(section, shown)
    if section == "被否决的前提":
        shown, premise = _premise(shown, work=work, repo=repo)
    return {"section": section, "audience": audience, "text": shown, "digest": digest, "premise": premise}


def _premise(text: str, *, work: Path | None, repo: Path | None) -> tuple[str, str]:
    """@brief 判定一条前提的验证状态。

    带 ``[已实测: ...]`` 标记时，探针文件存在为 ``verified``，不存在则降级为 ``downgraded``；没有标记就是
    ``unverified``。

    @param text 条目正文。
    @param work 会话工作目录，用于验证探针文件。
    @param repo 仓库根目录，作为另一个解析根。
    @return ``(正文, 状态)``，状态为 ``verified``/``downgraded``/``unverified``。
    """
    measured = _MEASURED.search(text)
    if measured:
        probe = measured.group(1).strip()
        if _probe_exists(probe, work=work, repo=repo):
            return text, "verified"
        return _downgrade_measured(text), "downgraded"
    return text, "unverified"


def _downgrade_measured(text: str) -> str:
    """@brief 把 ``[已实测: ...]`` 标记改成 ``[未验证: 探针文件不存在 | ...]``。

    命题与数字保留，只换掉标记。

    @param text 条目正文。
    @return 降级后的正文（只替换第一处标记）。
    """

    def repl(match: re.Match) -> str:
        prop = match.group(2).strip()
        number = match.group(3).strip()
        return f"[未验证: 探针文件不存在 | {prop} | {number}]"

    return _MEASURED.sub(repl, text, count=1)


def _probe_roots(work: Path | None, repo: Path | None) -> list[Path]:
    """@brief 列出探针文件允许被解析到的根目录。

    工作目录在 ``agent_work`` 下时，把 ``agent_work`` 也加进去，使兄弟会话留下的探针也能匹配。

    @param work 会话工作目录。
    @param repo 仓库根目录。
    @return 根目录列表，顺序即优先级。
    """
    roots = []
    if work is not None:
        roots.append(work)
        if work.parent.name == "agent_work":
            roots.append(work.parent)
    if repo is not None:
        roots.append(repo)
    return roots


def _probe_exists(token: str, *, work: Path | None, repo: Path | None) -> bool:
    """@brief 判断提到的某个探针文件是否真的可读。

    在工作目录、仓库或兄弟会话目录下任一根里能找到就算。

    @param token 正文里提到的探针片段，可以是一句带路径的话。
    @param work 会话工作目录。
    @param repo 仓库根目录。
    @return 至少一个候选路径存在时为 ``True``。
    """
    if not token or "\x00" in token:
        return False
    roots = _probe_roots(work, repo)
    candidates = [token.strip()]
    candidates.extend(_FILE_TOKEN.findall(token))
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        if _one_probe(name, roots):
            return True
    return False


def _one_probe(token: str, roots: list[Path]) -> bool:
    """@brief 在给定根下找单个探针文件。

    解析后的路径必须仍在某个根之内，防止 ``../`` 之类跑到外面。

    @param token 候选路径（绝对或相对）。
    @param roots 允许的根目录。
    @return 在某个根内找到这个文件时为 ``True``。
    """
    if not token or "\x00" in token:
        return False
    raw = Path(token)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [root / raw for root in roots]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not _inside(resolved, roots):
            continue
        if resolved.is_file():
            return True
    return False


def _inside(path: Path, roots: list[Path]) -> bool:
    """@brief 判断路径是否落在某个根目录之内。

    @param path 待检查的路径（应已 resolve）。
    @param roots 允许的根目录。
    @return 在任一根内时为 ``True``。
    """
    for root in roots:
        try:
            path.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        return True
    return False


def _digest(section: str, text: str) -> str:
    """@brief 算一条主张的身份 digest。

    数字与探针文件名不影响它：括号外的正文非空时，那才是主张，所以一条实测行与同一句被判为未验证的
    句子仍是同一条主张；括号包住整句时只哈希命题位。同一命题配两个探针或两个理由也算同一条。

    @param section 所属小节名。
    @param text 条目正文。
    @return 16 字符十六进制 digest；正文归一后为空时返回空字符串。
    """
    normalized = _digest_text(text or "")
    if not normalized:
        return ""
    payload = f"{section}\n{normalized}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _digest_text(text: str) -> str:
    """@brief 把正文归一成可哈希的形式。

    括号外有正文时只取括号外；否则从标记里挑命题位，没命题位时退回探针位。

    @param text 条目正文。
    @return 去数字、去标点、压空白的文本；空内容时为空字符串。
    """
    match = _MARK.search(text)
    if match is None:
        return _normalize(text)
    outside = _normalize(_MARK.sub(" ", text))
    if outside:
        return outside
    parts = _mark_parts(match.group(2))
    probe = parts[0] if parts else ""
    prop = parts[1] if len(parts) >= 2 else ""
    if len(parts) >= 2 and _normalize(prop):
        return _normalize(prop)
    if len(parts) >= 2:
        return _normalize(f"{probe} {prop}")
    return _normalize(prop or probe)


def _mark_parts(body: str) -> list[str]:
    """@brief 把 ``[...]`` 标记的内容按 ``|`` 拆成若干槽。

    @param body 标记内文本（含或不含前导冒号）。
    @return 去空白的槽列表；内容为空时为空列表。
    """
    body = body.strip()
    if body.startswith(":"):
        body = body[1:].strip()
    return [part.strip() for part in body.split("|")] if body else []


def _is_placeholder(text: str) -> bool:
    """@brief 判断一个小节或条目是否只是在说「没什么可报」。

    完全等同于固定填充词的是占位。以缺席句开头、且后面没有动作请求的也算：``无（预算已尽）`` 是状态说明，
    而 ``无。请修一下 check`` 是请求，保留。光有文件名和数字不构成请求；审计里那七行都解释了为什么是空的，
    仍然归到解析器的占位档。

    @param text 小节或条目正文。
    @return 是占位时为 ``True``。
    """
    raw = (text or "").strip()
    cleaned = _PUNCT.sub("", raw)
    cleaned = _SPACE.sub("", cleaned).strip().lower()
    if cleaned in _PLACEHOLDERS:
        return True
    if not _ABSENT.match(raw):
        return False
    match = _SENTENCE.search(raw)
    rest = raw[match.end():] if match else ""
    return not bool(_REQUEST.search(rest))


def _fit_lines(entries: list[dict], limit: int, *, drop_oldest: bool) -> list[str]:
    """@brief 把若干条提示排成行，直到总字符数落进预算。

    @param entries 条目列表，每项带 ``row`` 与 ``text``。
    @param limit 总字符预算。
    @param drop_oldest 超出预算时先丢最旧的，还是先丢最新的。
    @return 提示行列表，顺序与保留的条目一致。
    """
    lines = [_tip_line(entry["row"], entry["text"]) for entry in entries]
    while lines and sum(len(line) + 1 for line in lines) > limit:
        lines.pop(0 if drop_oldest else -1)
    return lines


def _fit_rest(entries: list[dict], active_count: int) -> list[str]:
    """@brief 排普通提示行：未验证的行留在前面。

    超出 ``TIPS_CHARS`` 时先丢普通提示，丢完再丢最旧的那条未验证行。

    @param entries 条目列表，前面 ``active_count`` 条是未验证前提。
    @param active_count 未验证前提的条数。
    @return 提示行列表。
    """
    lines = [_tip_line(entry["row"], entry["text"]) for entry in entries]
    kept_active = min(active_count, len(lines))
    while lines and sum(len(line) + 1 for line in lines) > TIPS_CHARS:
        if len(lines) > kept_active:
            lines.pop()
        else:
            lines.pop(0)
            kept_active = max(0, kept_active - 1)
    return lines


def _repair_record(row: dict, directory: Path | None = None) -> dict:
    """@brief 补齐旧解析器留下的空字段；不写文件。

    被擦掉的实测前提保留旧 digest，但正文可以从存档的小节里恢复。小节顺序只用来填空 digest，
    然后用 digest 挑回句子；旧记录丢过占位条目，从那时起按位置匹配会恢复错句子。

    @param row 一条反馈记录。
    @param directory 记录所属的运行目录，用于解析探针路径。
    @return 同一个记录对象（已就地补齐）。
    """
    sections = row.get("sections") if isinstance(row.get("sections"), dict) else {}
    if row.get("truncated") and "truncated_sections" not in row:
        row["truncated_sections"] = [
            name for name in SECTIONS
            if isinstance(sections.get(name), str) and len(sections[name]) >= SECTION_CAPS[name]
        ]
    items = [item for item in (row.get("items") or []) if isinstance(item, dict)]
    pieces = _pieces(str(sections.get("被否决的前提") or ""))
    work = repo = None
    if directory is not None:
        work_rel = str(row.get("work_dir") or "")
        candidate = (Path(directory) / work_rel).resolve() if work_rel else None
        if candidate is not None:
            try:
                candidate.relative_to(Path(directory).resolve())
                work = candidate
            except ValueError:
                work = None
        repo = Path(__file__).resolve().parents[1]
    premise_items = [item for item in items if item.get("section") == "被否决的前提"]
    # Older records stored an empty digest. The section pieces are still in order, so fill
    # the empty slots first; never overwrite a digest that a previous rebuild preserved.
    for item, piece in zip(premise_items, pieces):
        if not str(item.get("digest") or "").strip():
            item["digest"] = _digest("被否决的前提", piece)
    source = {
        _digest("被否决的前提", piece): piece
        for piece in pieces
        if _digest("被否决的前提", piece)
    }
    for item in premise_items:
        text = str(item.get("text") or "").strip()
        piece = source.get(str(item.get("digest") or ""))
        if piece is not None and _wiped_premise(text):
            shown, premise = _premise(piece, work=work, repo=repo)
            item["text"] = shown
            item["premise"] = premise
        if not str(item.get("digest") or "").strip() and piece is not None:
            item["digest"] = _digest("被否决的前提", piece)
    for item in items:
        if str(item.get("digest") or "").strip():
            continue
        text = str(item.get("text") or "").strip()
        if text:
            item["digest"] = _digest(str(item.get("section") or ""), text)
    row["items"] = [item for item in items if not _is_placeholder(str(item.get("text") or ""))]
    row["unverified_premises"] = sum(
        1 for item in row["items"] if item.get("premise") in ("unverified", "downgraded")
    )
    if "placeholder_items" not in row:
        count = 0
        for name in SECTIONS:
            chunk = str(sections.get(name) or "")
            if not chunk.strip():
                continue
            if _is_placeholder(chunk):
                count += 1
            else:
                count += sum(1 for piece in _pieces(chunk) if _is_placeholder(piece))
        row["placeholder_items"] = count
    else:
        row["placeholder_items"] = int(row.get("placeholder_items") or 0)
    return row


def check_feedback(path: Path, work: Path, repo: Path) -> str:
    """@brief 打印解析器如何读一份 FEEDBACK.md。

    只读文件，不写任何东西。

    @param path FEEDBACK.md 的路径。
    @param work 会话工作目录，用于验证探针文件。
    @param repo 仓库根目录。
    @return 逐条列出的文字报告（小节截断、占位跳过、每条条目的状态与 digest）。
    """
    text, file_over = read_feedback_text(path)
    parsed = parse_feedback(text, work=work, repo=repo, file_over=file_over)
    lines = []
    cuts = parsed.get("truncated_sections") or []
    skipped = parsed.get("placeholder_sections") or []
    lines.append("truncated_sections: " + (", ".join(cuts) if cuts else "-"))
    lines.append("placeholder_sections: " + (", ".join(skipped) if skipped else "-"))
    lines.append(f"placeholder_items: {int(parsed.get('placeholder_items') or 0)}")
    for index, item in enumerate(parsed.get("items") or [], start=1):
        reason = ""
        if item.get("premise") == "downgraded":
            reason = " reason=探针文件不存在"
        digest = str(item.get("digest") or "")
        lines.append(
            f"[{index}] section={item.get('section') or '-'} premise={item.get('premise') or '-'}"
            f"{reason} digest={digest or '-'} empty_digest={'yes' if not digest else 'no'}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m taic.agent_feedback check FEEDBACK.md --work DIR`` 的入口。

    @param argv 命令行参数；缺省从 ``sys.argv`` 取。
    @return 进程退出码，成功为 0。
    """
    import argparse

    parser = argparse.ArgumentParser(prog="taic.agent_feedback")
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check")
    check.add_argument("path")
    check.add_argument("--work", required=True)
    check.add_argument("--repo", default=None)
    args = parser.parse_args(argv)
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[1]
    sys.stdout.write(check_feedback(Path(args.path), Path(args.work), repo))
    return 0


def _normalize(text: str) -> str:
    """@brief 比较用的归一化：小写、去数字、去标点、压空白。

    @param text 原始文本。
    @return 归一化后的文本。
    """
    lowered = text.lower()
    without_numbers = _NUMBER.sub(" ", lowered)
    without_punct = _PUNCT.sub(" ", without_numbers)
    return _SPACE.sub(" ", without_punct).strip()


def _tip_line(row: dict, text: str) -> str:
    """@brief 把一条提示渲染成一行。

    前缀带标签、委员会/模型、结果与加速比，整行不超过 ``TIP_CHARS``。

    @param row 反馈记录，提供 ``label``/``committee``/``model``/``outcome``/``best_speedup``。
    @param text 提示正文。
    @return 单行提示；正文先压成一行再截断，避免把标题塞进提示面。
    """
    label = row.get("label") or "?"
    committee = row.get("committee") or "?"
    model = row.get("model") or "?"
    outcome = row.get("outcome") or row.get("exit") or "?"
    speed = row.get("best_speedup")
    suffix = ""
    if isinstance(speed, (int, float)) and not isinstance(speed, bool):
        suffix = f" {float(speed):.3f}x"
    prefix = f"- [{label} {committee}/{model}, {outcome}{suffix}] "
    room = TIP_CHARS - len(prefix)
    # 一条 tip 会原样进下一份 TASK.md。正文里的换行能另起一段，把 `# 约束` 之类的标题
    # 塞进契约面，所以先压成一行再截断。
    body = _SPACE.sub(" ", str(text or "")).strip()
    body = body if room >= len(body) else (body[: max(room, 0)] + ("…" if room > 0 else ""))
    line = prefix + body
    return line[:TIP_CHARS]


def _load_line(line: str) -> dict | None:
    """@brief 把一行 JSONL 解成字典。

    空行、坏 JSON、以及不是对象的行都返回 ``None``。

    @param line 单行文本。
    @return 解析后的字典；不可用时为 ``None``。
    """
    if not line.strip():
        return None
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(row, dict):
        return None
    return row


if __name__ == "__main__":
    raise SystemExit(main())
