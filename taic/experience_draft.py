"""Chip-level experience drafts.

A draft is a full section with ``Confidence: reported``. The existing prompt
filter keeps reported sections out of the prompt. This module does not write
``targets/<id>/experience.md`` and does not promote a section to ``verified``
unless the caller supplies a host-rerun pointer. Simulated timings and theory
notes are refused. Bare ``- [harness ...]`` bullets are not produced here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_CONFIDENCE = re.compile(r"^Confidence:\s*([A-Za-z_]+)\s*$", re.MULTILINE)
_HOST_RERUN = re.compile(r"^宿主重跑:\s*(\S+)\s*$", re.MULTILINE)
_THEORY_LINE = re.compile(r"^\s*理论\s*[:：]", re.MULTILINE)
_TARGET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LANDING = re.compile(r"^落点:\s*(\S+)\s*$", re.MULTILINE)
_CHIP_LANDING = re.compile(r"^targets/[A-Za-z0-9][A-Za-z0-9._-]*/experience\.md$")
_SOURCE_CITE = re.compile(r"^来源:.*(`runs/|`docs/|`tcenv/|`taic/)", re.MULTILINE)
_BARE_SOURCE = re.compile(r"^((?:runs|docs|tcenv|taic)/\S+)$")
_SECTION_HEAD = re.compile(r"(?m)^## ")
_UNVERIFIED_LINE = re.compile(r"(?m)^(?:未验证:.*|not reproduced:.*)\n", re.IGNORECASE)
_SIMULATED_MARK = re.compile(r"(?i)\bsimulated\b")
_SIMULATION_KEYS = ("evidence_level", "measurement_type", "measurement_kind")
_FIELDS = ("task", "title", "source", "measurement", "boundary", "counterexample", "observation")


class DraftError(ValueError):
    """@brief 草稿不合实践知识合同，调用方不应把它写进经验库。"""


def clean_practice(value) -> dict | None:
    """@brief 留下草稿字段。空字典返回 None，不补任何测量数字。

    @param value 提案上的 ``practice`` 对象。
    @return 清洗后的字段；不是对象时返回 None。
    """
    if not isinstance(value, dict):
        return None
    out: dict = {}
    for key in _FIELDS:
        text = " ".join(str(value.get(key) or "").split())
        if text:
            out[key] = text[:2000]
    from taic.evidence_level import SIMULATED, declares_simulation

    for key in _SIMULATION_KEYS:
        text = str(value.get(key) or "").strip()
        if text == SIMULATED:
            out[key] = SIMULATED
    provenance = value.get("evidence_provenance")
    if isinstance(provenance, dict) and declares_simulation({"evidence_provenance": provenance}):
        out["evidence_provenance"] = {
            key: provenance[key]
            for key in ("level", "evidence_level", "simulated", "measurement_kind", "measurement_type")
            if key in provenance
        }
    declared = (
        value.get("simulated") is True
        or declares_simulation(value)
        or bool(_SIMULATED_MARK.search(out.get("measurement") or ""))
    )
    if declared:
        out["simulated"] = True
    theory = " ".join(str(value.get("theory") or "").split())
    if theory:
        out["theory"] = theory[:2000]
    return out or None


def target_rel(target_id: str) -> str:
    """@brief 芯片层落点。只有 ``targets/<id>/experience.md``。

    @param target_id 芯片标识。
    @return 仓库相对路径。
    @raises DraftError 标识非法时抛出。
    """
    if not _TARGET_ID.fullmatch(str(target_id or "")):
        raise DraftError("invalid target id")
    return f"targets/{target_id}/experience.md"


def practice_identity(practice: dict | None) -> str:
    """@brief 非空 practice 的稳定身份。空 practice 返回空串，不改变旧 proposal key。"""
    if not isinstance(practice, dict) or not practice:
        return ""
    payload = json.dumps(practice, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cite_source(source: str) -> str:
    text = _require("source", source)
    if _SOURCE_CITE.search(f"来源: {text}"):
        return text
    bare = _BARE_SOURCE.fullmatch(text)
    if bare is not None:
        return f"`{bare.group(1)}`"
    raise DraftError("source must cite a runs, docs, tcenv, or taic path")


def _require(name: str, value: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise DraftError(f"{name} is required")
    if _THEORY_LINE.search(text):
        raise DraftError("theory does not belong in an experience draft")
    return text


def render_draft(
    *,
    target_id: str,
    task: str,
    title: str,
    source: str,
    measurement: str,
    boundary: str,
    counterexample: str,
    observation: str = "",
    simulated: bool = False,
    theory: str = "",
    layer: str = "chip",
) -> str:
    """@brief 生成一节 ``Confidence: reported`` 的经验草稿。

    @param target_id 芯片标识。
    @param task 题目。
    @param title 小节标题。
    @param source 来源指针。
    @param measurement 本机实测，必须由调用方给出。
    @param boundary 边界。
    @param counterexample 反例。
    @param observation 观察。空则用标题。
    @param simulated 模拟态计时。为真时拒绝，不生成草稿。
    @param theory 理论解释。非空时拒绝。
    @param layer 只接受 ``chip``。
    @return Markdown 小节。
    @raises DraftError 缺字段、模拟态、理论解释或非芯片层时抛出。
    """
    if layer != "chip":
        raise DraftError("only the chip layer has a landing point")
    if simulated:
        raise DraftError("simulated timing does not generate an experience draft")
    if str(theory or "").strip():
        raise DraftError("theory does not belong in an experience draft")
    chip = target_rel(target_id)
    task_text = _require("task", task)
    title_text = _require("title", title)
    source_text = _cite_source(source)
    measured = _require("measurement", measurement)
    boundary_text = _require("boundary", boundary)
    counter_text = _require("counterexample", counterexample)
    observed = _require("observation", observation or title_text)
    if _SIMULATED_MARK.search(measured):
        raise DraftError("simulated timing does not generate an experience draft")
    return (
        f"## {title_text}\n\n"
        f"Keywords: {task_text}\n"
        "Confidence: reported\n"
        f"题目: {task_text}\n"
        f"芯片: {target_id}\n"
        f"落点: {chip}\n"
        f"观察: {observed}\n"
        f"来源: {source_text}\n"
        f"本机实测: {measured}\n"
        f"边界: {boundary_text}\n"
        f"反例: {counter_text}\n"
        "未验证: 本机还没复现，别当事实用\n"
    )


def valid_host_pointer(pointer: str) -> bool:
    """@brief 宿主重跑指针必须是 run 证据路径，不能是一句测量数字，也不能用 ``.`` / ``..`` 逃出 ``runs/``。"""
    text = str(pointer or "").strip()
    if not text or "://" in text or any(char.isspace() for char in text):
        return False
    if not text.startswith("runs/"):
        return False
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    name = parts[-1]
    if name.endswith((".json", ".jsonl", ".md")):
        return True
    return len(parts) >= 3


def _one_section(text: str) -> str:
    body = str(text or "").strip()
    if not body:
        raise DraftError("experience section needs Confidence")
    if len(_SECTION_HEAD.findall(body)) != 1 or not body.startswith("## "):
        raise DraftError("experience promotion needs exactly one section")
    return body if body.endswith("\n") else body + "\n"


def _field(text: str, label: str) -> str:
    found = re.search(rf"(?m)^{re.escape(label)}:\s*(.+?)\s*$", text)
    return found.group(1) if found else ""


def _assert_draft_contract(text: str, *, confidence: str, pointer: str = "") -> str:
    """@brief 一节芯片层草稿必须自带题目、来源、实测、边界、反例，且指针属于这一节。"""
    levels = _CONFIDENCE.findall(text)
    if levels != [confidence]:
        raise DraftError("experience section needs Confidence")
    if _SIMULATED_MARK.search(text) or _THEORY_LINE.search(text):
        raise DraftError("simulated timing does not generate an experience draft")
    for label in ("题目", "来源", "本机实测", "边界", "反例"):
        if not _field(text, label):
            raise DraftError(f"{label} is required")
    landing = _field(text, "落点")
    if _CHIP_LANDING.fullmatch(landing) is None:
        raise DraftError("only the chip layer has a landing point")
    if _SOURCE_CITE.search(text) is None:
        raise DraftError("source must cite a runs, docs, tcenv, or taic path")
    found = _HOST_RERUN.findall(text)
    if confidence == "reported":
        if "未验证" not in text and "not reproduced" not in text.lower():
            raise DraftError("reported experience must say it is not reproduced")
        if found:
            raise DraftError("reported experience does not take a host rerun pointer")
        return ""
    if len(found) != 1 or not valid_host_pointer(found[0]):
        raise DraftError("verified experience needs a host rerun pointer")
    if pointer and found[0] != pointer:
        raise DraftError("verified experience needs a host rerun pointer")
    return found[0]


_NEW_DRAFT_LINE = re.compile(r"(?m)^(题目|落点|芯片|未验证):")


def verified_section_prompt_ready(text: str) -> bool:
    """@brief 带任一新草稿行的 verified 小节，还要有恰好一条有效宿主重跑指针。

    新草稿行是 ``题目``、``落点``、``芯片``、``未验证``。缺其中一行仍走这套校验。
    旧 verified 小节没有这些行，不补指针也仍然可以进提示。

    @param text 小节正文（不含 ``##`` 标题也行）。
    @return 这一节能否在 ``Confidence: verified`` 之外进入提示面。
    """
    body = str(text or "")
    if _NEW_DRAFT_LINE.search(body) is None:
        return True
    found = _HOST_RERUN.findall(body)
    return len(found) == 1 and valid_host_pointer(found[0])


def validate_section(text: str) -> dict:
    """@brief 检查恰好一节经验。``verified`` 的宿主重跑指针必须落在这一节里。

    不改写文件。``reported`` 可以被现有过滤器挡在提示面外。

    @param text 小节正文。
    @return ``confidence``、``prompt_eligible`` 与 ``host_rerun``。
    @raises DraftError 不止一节、缺字段、模拟态，或 verified 没有属于本节的合格指针。
    """
    section = _one_section(text)
    level = _CONFIDENCE.findall(section)[0] if _CONFIDENCE.findall(section) else ""
    if level not in ("reported", "verified"):
        raise DraftError("experience section needs Confidence")
    pointer = _assert_draft_contract(section, confidence=level)
    return {
        "confidence": level,
        "prompt_eligible": level == "verified" and valid_host_pointer(pointer),
        "host_rerun": pointer if valid_host_pointer(pointer) else "",
    }


def set_verified(text: str, host_rerun: str) -> str:
    """@brief 把恰好一节 ``reported`` 改成 ``verified``，并写上属于这一节的宿主重跑指针。

    重新检查题目、实测、边界、反例、芯片落点、来源、模拟态和理论。不写经验库文件。

    @param text 草稿小节。
    @param host_rerun 宿主重跑证据路径。
    @return 新小节。
    @raises DraftError 指针不合格、不止一节、或原文不满足 reported 合同。
    """
    pointer = str(host_rerun or "").strip()
    if not valid_host_pointer(pointer):
        raise DraftError("verified experience needs a host rerun pointer")
    section = _one_section(text)
    _assert_draft_contract(section, confidence="reported")
    updated = _CONFIDENCE.sub("Confidence: verified", section, count=1)
    updated = _UNVERIFIED_LINE.sub("", updated)
    if _HOST_RERUN.search(updated):
        updated = _HOST_RERUN.sub(f"宿主重跑: {pointer}", updated, count=1)
    else:
        if not updated.endswith("\n"):
            updated += "\n"
        updated += f"宿主重跑: {pointer}\n"
    _assert_draft_contract(updated, confidence="verified", pointer=pointer)
    return updated


def refuse_live_experience(path: Path, repo: Path) -> None:
    """@brief 草稿命令不写仓库里的活经验文件。

    @param path 调用方给出的输出路径。
    @param repo 仓库根。
    @raises DraftError 路径落在 ``targets/<id>/experience.md`` 时抛出。
    """
    try:
        relative = Path(path).resolve().relative_to(Path(repo).resolve())
    except ValueError:
        return
    if len(relative.parts) >= 2 and relative.parts[0] == "targets" and relative.name == "experience.md":
        raise DraftError("draft-experience does not write the live experience file")
