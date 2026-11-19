"""Read-only scope audit of ``targets/<id>/experience.md``.

The audit lists, for every section, the declared title/Confidence/task/target/
landing, the evidence pointers, whether the boundary names a task/dtype/size,
whether the section carries a simulation declaration or a device-specific term,
and every scope inconsistency the bytes actually prove.

It never edits the experience file, never writes a promotion result, never
changes ``Confidence`` and never copies a section to another target.  A section
that lacks the newer ``题目`` / ``芯片`` / ``落点`` lines is reported as
``legacy``/``incomplete``; missing fields are not evidence of forgery.

Scope mismatches are reported only when a structured field or an evidence path
proves them:

* ``芯片`` / ``落点`` / the file's own ``targets/<id>/`` location disagree;
* a ``runs/`` pointer contains a known target id other than the declared one;
* a ``runs/`` pointer or a ``tasks/<id>/`` citation names a known task other
  than ``题目``.  Only those structured references count: a task name inside the
  prose of ``来源`` ("do not confuse this with conv2d") is background, not proof.

When a path names no known task/target, the dimension stays ``unknown``.  Titles
are never used to guess the task.

The device-term table below is a *mention lexicon* for auditing, not a fact
table: it says "this word appears", never "this word is true of the device".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .experience_draft import valid_host_pointer

SCHEMA_VERSION = 1
KIND = "experience-scope-audit"
READ_ONLY = True
NOTICE = "只读审计；不晋升、不改 Confidence、不写经验库"

_CONFIDENCE = re.compile(r"^Confidence:\s*([A-Za-z_]+)\s*$", re.MULTILINE)
_HOST_LINE = re.compile(r"(?m)^宿主重跑:\s*(.*)$")
_BACKTICK_RUNS = re.compile(r"`(runs/[^`]*)`")
_BACKTICK_SPAN = re.compile(r"`[^`]*`")
#: 来源里不带反引号的指针，例如 ``来源: runs/x/y.json``。中文标点与右括号也停，
#: 尾部的 ``.,;:`` 在取到后去掉，所以英文句子里的 ``runs/x.json.`` 仍是一条路径。
_BARE_RUNS = re.compile(r"runs/[^\s`（）()\[\]{}，。；、,;:!?'\"]+")
_TASK_CITATION = re.compile(r"`?tasks/([A-Za-z0-9][A-Za-z0-9._-]*)/")
# ``_template`` is a reserved target directory, so the leading ``_`` is allowed here even
# though ``experience_draft.target_rel`` only accepts runnable target ids.
_LANDING = re.compile(r"^targets/([A-Za-z0-9_][A-Za-z0-9._-]*)/experience\.md$")
_RELATIVE = re.compile(r"^targets/([A-Za-z0-9_][A-Za-z0-9._-]*)/experience\.md$")
_IDENT_CHAR = re.compile(r"[A-Za-z0-9_]")
_DTYPE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])"
    r"(fp16|fp32|fp64|bf16|f16|f32|f8|i8|u8|i16|u16|i32|u32|i64|u64|"
    r"int8|uint8|half|float16|float32|float64)"
    r"(?![A-Za-z0-9_])"
)
_SIZE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(tiny|small|medium|large|paper|tail|big)(?![A-Za-z0-9_])"
    r"|\b[NHMDKB]\s*=\s*\d+"
    r"|\b\d+\s*[x×]\s*\d+\b"
)

#: Fields introduced by ``experience_draft.render_draft``.  A verified section
#: carrying any of them must also carry exactly one valid host pointer, or the
#: prompt filter drops it silently (``taic.experience_prompt``).
NEW_DRAFT_FIELDS = ("题目", "芯片", "落点", "未验证")

#: ``(group, label, pattern)``.  Mention lexicon only; see the module docstring.
_DEVICE_TERMS = (
    ("metal", "metal", re.compile(r"(?i)(?<![a-z])metal(?![a-z])")),
    ("metal", "msl", re.compile(r"(?i)(?<![a-z])msl(?![a-z])")),
    ("metal", "simdgroup", re.compile(r"(?i)simdgroup")),
    ("metal", "threadgroup", re.compile(r"(?i)threadgroup")),
    ("metal", "mtl", re.compile(r"(?<![A-Za-z])MTL[A-Z]")),
    ("metal", "apple9", re.compile(r"(?i)(?<![a-z0-9])apple9(?![a-z0-9])")),
    ("npu", "npu", re.compile(r"(?i)(?<![a-z])npu(?![a-z])")),
    ("npu", "qingwei", re.compile(r"(?i)qingwei|清微")),
)

#: 每种指针字段对应的告警码：格式无效与路径不存在分开报。
_POINTER_CODES = {
    "宿主重跑": {"invalid": "host_pointer_invalid", "missing": "host_pointer_missing"},
    "来源": {"invalid": "source_pointer_invalid", "missing": "source_pointer_missing"},
    "正文": {"invalid": "body_pointer_invalid", "missing": "body_pointer_missing"},
}

#: Simulation declarations.  A simulated section is never real-hardware advice.
_SIMULATION_TERMS = (
    ("simulated", re.compile(r"(?i)(?<![a-z])simulated(?![a-z])")),
    ("simulation", re.compile(r"(?i)(?<![a-z])simulation(?![a-z])")),
    ("模拟", re.compile(r"模拟")),
)


def discover_tasks(repo: Path) -> list[str]:
    """@brief 仓库里的标准题：``tasks/<id>/opspec.json`` 的目录名。

    @param repo 仓库根。
    @return 排序后的题目 id 列表；没有 ``tasks/`` 时为空。
    """
    root = Path(repo) / "tasks"
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if (path / "opspec.json").is_file())


def discover_targets(repo: Path) -> list[str]:
    """@brief 仓库里登记过的 target：``targets/<id>/target.toml`` 的目录名。

    @param repo 仓库根。
    @return 排序后的 target id 列表；没有 ``targets/`` 时为空。
    """
    root = Path(repo) / "targets"
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if (path / "target.toml").is_file())


def _field(text: str, label: str) -> str:
    """@brief 取一行 ``label: value`` 的值；没有这一行时返回空串。"""
    found = re.search(rf"(?m)^{re.escape(label)}:\s*(.+?)\s*$", text)
    return found.group(1).strip() if found else ""


def _sections(raw: str) -> list[tuple[str, str]]:
    """@brief 按 ``## `` 切小节，返回 ``(标题, 正文)``；标题按原文保留。

    与 ``taic.experience_prompt.filter_experience_for_prompt`` 用同一套切法，
    所以审计口径与提示面口径一致。
    """
    parts = ("\n" + str(raw or "").lstrip("\n")).split("\n## ")
    out = []
    for chunk in parts[1:]:
        title, _, body = chunk.partition("\n")
        out.append((title.strip(), body))
    return out


def _hits(text: str, known: list[str]) -> set[str]:
    """@brief 在文本里找已知 id，要求两侧不是标识符字符。

    ``_`` 算标识符字符，所以 ``gemm`` 不会命中 ``gemm_silu``：两个不同的题目不能
    因为共享前缀就算匹配。
    """
    low = str(text or "").lower()
    found: set[str] = set()
    for name in known:
        target = name.lower()
        start = low.find(target)
        while start >= 0:
            end = start + len(target)
            left = low[start - 1] if start else ""
            right = low[end] if end < len(low) else ""
            if (not left or not _IDENT_CHAR.match(left)) and (not right or not _IDENT_CHAR.match(right)):
                found.add(name)
                break
            start = low.find(target, start + 1)
    return found


def _target_hits(text: str, known: list[str]) -> set[str]:
    """@brief 在路径里找已知 target，只认整段相等，避免 ``runs/hw_features/`` 被当成芯片。"""
    segments = set(re.split(r"[/\s`]+", str(text or "")))
    return {name for name in known if name in segments}


def _pointer_problem(text: str) -> str:
    """@brief 指针不合格的原因，只用于报告，不代替 ``valid_host_pointer``。"""
    if not text:
        return "empty"
    if "://" in text:
        return "url"
    if any(char.isspace() for char in text):
        return "whitespace"
    if not text.startswith("runs/"):
        return "not_under_runs"
    if any(part in ("", ".", "..") for part in text.split("/")):
        return "dot_or_empty_segment"
    if not text.endswith((".json", ".jsonl", ".md")) and len(text.split("/")) < 3:
        return "too_short"
    return "invalid"


def inspect_pointer(text: str, *, repo: Path, field: str) -> dict:
    """@brief 校验一条仓库相对指针：格式用 ``valid_host_pointer``，再查文件存在。

    @param text 指针原文。
    @param repo 仓库根。
    @param field 指针出现的字段（``宿主重跑``/``来源``/``正文``）。
    @return ``field``、``text``、``valid``、``reason``、``exists``、``kind``。
    """
    valid = bool(valid_host_pointer(text))
    doc = {"field": field, "text": text, "valid": valid, "reason": "", "exists": False, "kind": "missing"}
    if not valid:
        doc["reason"] = _pointer_problem(text)
        return doc
    path = Path(repo).joinpath(*text.split("/"))
    if path.is_file():
        doc["exists"], doc["kind"] = True, "file"
    elif path.is_dir():
        doc["exists"], doc["kind"] = True, "directory"
    else:
        doc["reason"] = "not_found"
    return doc


def _looks_like_pointer(text: str) -> bool:
    """@brief 正文里的反引号 token 是不是一条路径；裸 ``runs/`` 是说明文字，不是指针。"""
    parts = str(text or "").split("/")
    return len(parts) >= 2 and parts[0] == "runs" and bool(parts[1])


def _source_pointers(source: str) -> list[str]:
    """@brief 取 ``来源`` 里的 ``runs/`` 指针：反引号包住的与裸写的都算。

    旧条目常把路径裸写在一句话里；只看反引号会漏掉它们，等于没校验。

    @param source ``来源`` 字段的原文。
    @return 按出现顺序的指针文本，已去掉句末标点。
    """
    tokens = _BACKTICK_RUNS.findall(source)
    bare = _BACKTICK_SPAN.sub(" ", source)
    for match in _BARE_RUNS.finditer(bare):
        tokens.append(match.group(0).rstrip(".,;:!?"))
    return tokens


def _pointers(body: str, repo: Path) -> tuple[list[dict], bool]:
    """@brief 收集本节所有 ``runs/`` 指针：``宿主重跑``、``来源``（含裸写）与正文反引号路径。

    同一路径只记一次，先出现的字段优先，所以来源与宿主重跑指同一份证据时不会
    被算成两条指针。

    @return ``(指针列表, 是否出现多行宿主重跑)``。
    """
    found: list[dict] = []
    seen: set[str] = set()
    multiple = False
    host_lines = [match.group(1).strip() for match in _HOST_LINE.finditer(body)]
    if len(host_lines) > 1:
        multiple = True
    for text in host_lines:
        if text in seen:
            continue
        seen.add(text)
        found.append(inspect_pointer(text, repo=repo, field="宿主重跑"))
    source = _field(body, "来源")
    source_tokens = _source_pointers(source)
    rest = body.replace(source, "", 1) if source else body
    for field, tokens in (("来源", source_tokens), ("正文", _BACKTICK_RUNS.findall(rest))):
        for text in tokens:
            if text in seen or (field == "正文" and not _looks_like_pointer(text)):
                continue
            seen.add(text)
            found.append(inspect_pointer(text, repo=repo, field=field))
    return found, multiple


def _boundary(body: str, known_tasks: list[str]) -> dict:
    """@brief 边界是否明确声明了任务、dtype 与 size。

    没有 ``边界`` 行时三项都是 ``None``（unknown），不是 ``False``。
    """
    text = _field(body, "边界")
    if not text:
        return {"present": False, "declares_task": None, "declares_dtype": None, "declares_size": None}
    return {
        "present": True,
        "declares_task": bool(_hits(text, known_tasks)),
        "declares_dtype": bool(_DTYPE.search(text)),
        "declares_size": bool(_SIZE.search(text)),
    }


def _mentions(section_text: str) -> tuple[dict, dict]:
    """@brief 本节的 Metal/NPU 专属词与模拟声明。

    @return ``(device_terms, simulation)``，两者都是出现即记录，不解释设备事实。
    """
    device = {"metal": [], "npu": []}
    for group, label, pattern in _DEVICE_TERMS:
        if pattern.search(section_text) and label not in device[group]:
            device[group].append(label)
    terms = [label for label, pattern in _SIMULATION_TERMS if pattern.search(section_text)]
    return device, {"declared": bool(terms), "terms": terms}


def _dimension(value: str, evidence: set[str], *, kind: str) -> dict:
    """@brief 一个作用域维度的判定：``match`` / ``mismatch`` / ``unknown``。

    只有证据里出现其它已知 id、且没有出现声明值时才是 ``mismatch``；证据里没有已知
    id 时保持 ``unknown``，不靠标题猜。
    """
    ordered = sorted(evidence)
    if not value:
        return {"value": None, "evidence": ordered, "verdict": "unknown", "reason": f"the section does not declare a {kind}"}
    if not ordered:
        return {"value": value, "evidence": [], "verdict": "unknown", "reason": f"no evidence path names a known {kind}"}
    if value in evidence:
        return {"value": value, "evidence": ordered, "verdict": "match", "reason": f"the {kind} appears in the evidence"}
    return {
        "value": value,
        "evidence": ordered,
        "verdict": "mismatch",
        "reason": f"the evidence names only {', '.join(ordered)}, not {value}",
    }


def _section_doc(
    index: int,
    title: str,
    body: str,
    *,
    file_target: str | None,
    repo: Path,
    known_tasks: list[str],
    known_targets: list[str],
) -> dict:
    """@brief 审计一节经验，返回这一节的结构化事实与告警。"""
    section_text = f"{title}\n{body}" if title else body
    declared = {
        "task": _field(body, "题目"),
        "target": _field(body, "芯片"),
        "landing": _field(body, "落点"),
        "source": _field(body, "来源"),
    }
    confidence = ""
    found = _CONFIDENCE.search(section_text)
    if found is not None:
        confidence = found.group(1).lower()
    new_draft = any(re.search(rf"(?m)^{re.escape(label)}:", body) for label in NEW_DRAFT_FIELDS)
    pointers, multiple_host = _pointers(body, repo)
    hosts = [pointer for pointer in pointers if pointer["field"] == "宿主重跑"]

    warnings: list[dict] = []

    def warn(code: str, detail: str) -> None:
        """@brief 记一条结构化告警。"""
        warnings.append({"code": code, "detail": detail})

    if multiple_host:
        warn("multiple_host_pointers", "the section has more than one 宿主重跑 line")
    for pointer in pointers:
        if not pointer["valid"]:
            warn(_POINTER_CODES[pointer["field"]]["invalid"], f"{pointer['text'] or '(empty)'}: {pointer['reason']}")
        elif not pointer["exists"]:
            warn(_POINTER_CODES[pointer["field"]]["missing"], f"{pointer['text']}: path not found under {Path(repo)}")
    if confidence == "verified" and new_draft and not any(p["valid"] for p in hosts):
        warn("verified_without_pointer", "a verified section with new draft fields needs exactly one valid 宿主重跑 pointer")

    landing_target = None
    landing = declared["landing"]
    if landing:
        match = _LANDING.fullmatch(landing)
        if match is None:
            warn("landing_not_chip_layer", f"落点 must be targets/<id>/experience.md, got {landing}")
        else:
            landing_target = match.group(1)
    if declared["target"] and landing_target and declared["target"] != landing_target:
        warn("target_landing_mismatch", f"芯片={declared['target']} but 落点 targets/{landing_target}/experience.md")
    if declared["target"] and file_target and declared["target"] != file_target:
        warn("target_file_mismatch", f"芯片={declared['target']} but this file lives under targets/{file_target}/")
    if landing_target and file_target and landing_target != file_target:
        warn("landing_file_mismatch", f"落点 targets/{landing_target}/experience.md but the file is targets/{file_target}/experience.md")

    # 题目证据只来自实际证据路径与 ``tasks/<id>/`` 这种结构化引用。来源里的说明文字
    # 提到别的题目名称不算错配证明（"别与 conv2d 混淆" 是背景，不是证据）。
    task_evidence = _hits(" ".join(pointer["text"] for pointer in pointers), known_tasks)
    task_evidence |= {name for name in _TASK_CITATION.findall(declared["source"]) if name in known_tasks}
    target_evidence = _target_hits(" ".join(pointer["text"] for pointer in pointers), known_targets)
    task_scope = _dimension(declared["task"], task_evidence, kind="task")
    target_scope = _dimension(declared["target"], target_evidence, kind="target")
    if task_scope["verdict"] == "mismatch":
        warn("evidence_task_mismatch", task_scope["reason"])
    if target_scope["verdict"] == "mismatch":
        warn("evidence_target_mismatch", target_scope["reason"])

    legacy_missing = [
        label for label, key in (("题目", "task"), ("芯片", "target"), ("落点", "landing")) if not declared[key]
    ]
    legacy = bool(legacy_missing)
    device_terms, simulation = _mentions(section_text)

    blockers: list[str] = []
    if simulation["declared"]:
        blockers.append("simulated_evidence")
    if confidence != "verified":
        blockers.append("not_verified")
    if legacy:
        blockers.append("legacy_incomplete")
    if task_scope["verdict"] == "unknown":
        blockers.append("unknown_task")
    if task_scope["verdict"] == "mismatch":
        blockers.append("task_mismatch")
    if target_scope["verdict"] == "mismatch":
        blockers.append("target_mismatch")
    if confidence == "verified":
        if not hosts:
            blockers.append("no_host_pointer")
        else:
            if not any(pointer["valid"] for pointer in hosts):
                blockers.append("invalid_host_pointer")
            elif not any(pointer["valid"] and pointer["exists"] for pointer in hosts):
                blockers.append("missing_host_pointer")
    elif hosts and not any(pointer["valid"] for pointer in hosts):
        blockers.append("invalid_host_pointer")

    return {
        "index": index,
        "title": title,
        "confidence": confidence or None,
        "declared": declared,
        "target": declared["target"] or None,
        "host_rerun": hosts[0] if hosts else None,
        "pointers": pointers,
        "boundary": _boundary(body, known_tasks),
        "device_terms": device_terms,
        "device_specific": bool(device_terms["metal"] or device_terms["npu"]),
        "simulation": simulation,
        "scope": {"task": task_scope, "target": target_scope, "landing_target": landing_target},
        "legacy": legacy,
        "legacy_missing_fields": legacy_missing,
        "new_draft": new_draft,
        "warnings": warnings,
        "promotion_review": {"eligible": not blockers, "blockers": blockers},
    }


def _empty_summary() -> dict:
    """@brief 汇总骨架：只有真正读到的小节才累加，没有文件时不能说成“零条经验”。"""
    return {
        "files": 0,
        "sections": 0,
        "confidence": {"verified": 0, "reported": 0, "other": 0},
        "pointers": {"total": 0, "valid": 0, "invalid": 0, "existing": 0, "missing": 0},
        "legacy_sections": 0,
        "simulated_sections": 0,
        "device_specific_sections": 0,
        "promotion_review_eligible": 0,
        "warnings": {"total": 0, "by_code": {}},
    }


def _summarize(sections: list[dict]) -> dict:
    """@brief 汇总一组小节：Confidence、指针、legacy、模拟、告警与晋升评审输入。"""
    summary = _empty_summary()
    summary["sections"] = len(sections)
    for section in sections:
        level = section["confidence"]
        if level in ("verified", "reported"):
            summary["confidence"][level] += 1
        else:
            summary["confidence"]["other"] += 1
        for pointer in section["pointers"]:
            summary["pointers"]["total"] += 1
            if pointer["valid"]:
                summary["pointers"]["valid"] += 1
                summary["pointers"]["existing" if pointer["exists"] else "missing"] += 1
            else:
                summary["pointers"]["invalid"] += 1
        summary["legacy_sections"] += int(bool(section["legacy"]))
        summary["simulated_sections"] += int(bool(section["simulation"]["declared"]))
        summary["device_specific_sections"] += int(bool(section["device_specific"]))
        summary["promotion_review_eligible"] += int(bool(section["promotion_review"]["eligible"]))
        for warning in section["warnings"]:
            summary["warnings"]["total"] += 1
            code = warning["code"]
            summary["warnings"]["by_code"][code] = summary["warnings"]["by_code"].get(code, 0) + 1
    summary["warnings"]["by_code"] = dict(sorted(summary["warnings"]["by_code"].items()))
    return summary


def _merge(base: dict, extra: dict) -> dict:
    """@brief 把一份小节汇总叠进另一份，字段集合保持相同。"""
    base["files"] += extra["files"]
    base["sections"] += extra["sections"]
    for level in base["confidence"]:
        base["confidence"][level] += extra["confidence"][level]
    for key in base["pointers"]:
        base["pointers"][key] += extra["pointers"][key]
    for key in ("legacy_sections", "simulated_sections", "device_specific_sections", "promotion_review_eligible"):
        base[key] += extra[key]
    base["warnings"]["total"] += extra["warnings"]["total"]
    for code, count in extra["warnings"]["by_code"].items():
        base["warnings"]["by_code"][code] = base["warnings"]["by_code"].get(code, 0) + count
    base["warnings"]["by_code"] = dict(sorted(base["warnings"]["by_code"].items()))
    return base


def _target_of(path: Path, repo: Path) -> str | None:
    """@brief 从 ``targets/<id>/experience.md`` 相对路径取 target id；不在该布局时返回 None。"""
    try:
        relative = Path(path).resolve().relative_to(Path(repo).resolve())
    except ValueError:
        relative = Path(path)
    match = _RELATIVE.fullmatch(str(relative).replace("\\", "/"))
    return match.group(1) if match else None


def _guess_repo(path: Path) -> Path:
    """@brief 由 ``targets/<id>/experience.md`` 反推仓库根；推不出时用当前目录。"""
    parents = path.resolve().parents
    if len(parents) >= 3 and parents[1].name == "targets":
        return parents[2]
    return Path.cwd()


def audit_experience_text(
    raw: str,
    *,
    target_id: str | None,
    repo: Path,
    path: str = "",
    known_tasks: list[str] | None = None,
    known_targets: list[str] | None = None,
) -> dict:
    """@brief 审计一份经验库文本，不改文件、不写晋升结果。

    @param raw ``experience.md`` 全文。
    @param target_id 文件所在的 target；未知时为 None（作用域判定保持 unknown）。
    @param repo 仓库根，用于解析 ``runs/`` 指针。
    @param path 报告中显示的相对路径。
    @param known_tasks 已知题目 id；None 时从 ``repo/tasks`` 发现。
    @param known_targets 已知 target id；None 时从 ``repo/targets`` 发现。
    @return 含 ``sections`` 与 ``summary`` 的字典。
    """
    tasks = discover_tasks(repo) if known_tasks is None else list(known_tasks)
    targets = discover_targets(repo) if known_targets is None else list(known_targets)
    sections = [
        _section_doc(
            index, title, body,
            file_target=target_id, repo=repo, known_tasks=tasks, known_targets=targets,
        )
        for index, (title, body) in enumerate(_sections(raw))
    ]
    return {
        "path": path,
        "target_id": target_id,
        "sections": sections,
        "summary": _summarize(sections),
    }


def audit_experience_file(path: Path, *, repo: Path | None = None) -> dict:
    """@brief 只读审计一个 ``experience.md``。

    @param path 经验库文件。
    @param repo 仓库根；None 时从 ``targets/<id>/experience.md`` 反推。
    @return 与 :func:`audit_experience_text` 相同的字典，``path`` 为相对仓库的路径。
    :raises OSError 文件不可读。
    """
    path = Path(path)
    root = Path(repo) if repo is not None else _guess_repo(path)
    raw = path.read_text(encoding="utf-8")
    try:
        shown = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        shown = str(path)
    return audit_experience_text(
        raw, target_id=_target_of(path, root), repo=root, path=shown.replace("\\", "/")
    )


def audit_repo(repo: Path = Path("."), *, target: str | None = None) -> dict:
    """@brief 审计 ``targets/<id>/experience.md``，默认全部 target。

    @param repo 仓库根。
    @param target 只看这一个 target；None 表示全部。
    @return ``{"schema_version", "kind", "read_only", "notice", "repo", "files", "summary"}``。
    """
    root = Path(repo)
    if target:
        candidates = [root / "targets" / target / "experience.md"]
    else:
        base = root / "targets"
        candidates = sorted(base.glob("*/experience.md")) if base.is_dir() else []
    files = [audit_experience_file(path, repo=root) for path in candidates if Path(path).is_file()]
    total = _empty_summary()
    total["files"] = len(files)
    for item in files:
        _merge(total, item["summary"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "read_only": READ_ONLY,
        "notice": NOTICE,
        "repo": str(root),
        "files": files,
        "summary": total,
    }


def _flag(value) -> str:
    """@brief 三态布尔的人类可读标记：True/False/unknown。"""
    if value is None:
        return "?"
    return "y" if value else "n"


def render_summary(doc: dict) -> str:
    """@brief 把审计结果渲染成人类可读摘要；不重新计算任何事实。"""
    lines = ["经验作用域审计（只读）", f"口径: {NOTICE}", ""]
    for item in doc.get("files", []):
        summary = item["summary"]
        lines.append(f"{item['path']}  [target={item['target_id'] or 'unknown'}]")
        lines.append(
            f"  条目 {summary['sections']} · verified {summary['confidence']['verified']} / "
            f"reported {summary['confidence']['reported']} / 其他 {summary['confidence']['other']} · "
            f"legacy {summary['legacy_sections']} · 模拟 {summary['simulated_sections']} · "
            f"warning {summary['warnings']['total']}"
        )
        pointers = summary["pointers"]
        lines.append(
            f"  指针 {pointers['total']} · 格式有效 {pointers['valid']}（存在 {pointers['existing']}） · "
            f"格式无效 {pointers['invalid']} · 不存在 {pointers['missing']}"
        )
        lines.append(f"  可进入人工晋升评审 {summary['promotion_review_eligible']}（本审计不晋升）")
        lines.append("")
        for section in item["sections"]:
            boundary = section["boundary"]
            lines.append(f"  {section['index'] + 1:02d}. {section['title'] or '(无标题)'}")
            legacy = ""
            if section["legacy"]:
                legacy = f"（legacy 缺 {','.join(section['legacy_missing_fields'])}）"
            lines.append(
                f"      target: {section['target'] or 'unknown'}{legacy} · "
                f"Confidence: {section['confidence'] or 'unknown'} · "
                f"题目: {section['declared']['task'] or 'unknown'}"
            )
            lines.append(f"      来源: {section['declared']['source'] or '无'}")
            pointer = section["host_rerun"]
            if pointer is None:
                shown = "无"
            else:
                shown = f"{pointer['text'] or '(空)'} [{'ok' if pointer['valid'] and pointer['exists'] else pointer['reason'] or 'invalid'}]"
            lines.append(f"      宿主重跑: {shown}")
            lines.append(
                "      作用域 题目:%s target:%s · 边界 任务:%s dtype:%s size:%s" % (
                    section["scope"]["task"]["verdict"],
                    section["scope"]["target"]["verdict"],
                    _flag(boundary["declares_task"]),
                    _flag(boundary["declares_dtype"]),
                    _flag(boundary["declares_size"]),
                )
                + f" · Metal: {', '.join(section['device_terms']['metal']) or '-'}"
                + f" · NPU: {', '.join(section['device_terms']['npu']) or '-'}"
                + f" · 模拟: {'yes' if section['simulation']['declared'] else 'no'}"
            )
            for warning in section["warnings"]:
                lines.append(f"      warning: {warning['code']} — {warning['detail']}")
        lines.append("")
    total = doc.get("summary", _empty_summary())
    pointers = total["pointers"]
    lines.append(
        f"汇总: {total['files']} 个文件 · {total['sections']} 条 · "
        f"verified {total['confidence']['verified']} / reported {total['confidence']['reported']} / "
        f"其他 {total['confidence']['other']} · 指针 {pointers['total']}（有效 {pointers['valid']} / "
        f"不存在 {pointers['missing']} / 无效 {pointers['invalid']}） · legacy {total['legacy_sections']} · "
        f"模拟 {total['simulated_sections']} · warning {total['warnings']['total']} · "
        f"晋升评审输入 {total['promotion_review_eligible']}"
    )
    if total["warnings"]["by_code"]:
        lines.append("warning 分类: " + ", ".join(f"{code}={count}" for code, count in total["warnings"]["by_code"].items()))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """@brief 只读 CLI：``python -m taic.experience_scope [--json]``。

    @param argv 参数列表；None 取 ``sys.argv[1:]``。
    @return 0 正常；2 找不到要审计的经验库。
    """
    parser = argparse.ArgumentParser(
        prog="experience-scope-audit",
        description="read-only scope audit of targets/<id>/experience.md; never edits or promotes",
    )
    parser.add_argument("--repo", default=".", help="repository root (default: current directory)")
    parser.add_argument("--target", default=None, help="audit only this target id")
    parser.add_argument("--json", action="store_true", help="print the machine-readable JSON instead of the summary")
    args = parser.parse_args(argv)
    doc = audit_repo(Path(args.repo), target=args.target)
    if not doc["files"]:
        print(f"no experience file under {Path(args.repo) / 'targets'}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_summary(doc), end="")
    return 0


__all__ = [
    "KIND",
    "NOTICE",
    "READ_ONLY",
    "SCHEMA_VERSION",
    "audit_experience_file",
    "audit_experience_text",
    "audit_repo",
    "discover_targets",
    "discover_tasks",
    "inspect_pointer",
    "main",
    "render_summary",
]


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
