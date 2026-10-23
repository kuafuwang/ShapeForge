"""The dashboard's model picker. Lists model ids from the credentials file and writes the chosen one into ``taic.toml``.

The credentials file holds API keys. This module never returns those keys, the base URL, or prices. A running search keeps the model it
started with; the next search that reads ``taic.toml`` uses the new one, including the next task a live experiment queue has not started yet.
A single choice is written into ``[llm]`` and into every committee member. The committee form writes each
committee's one member on its own, and sets ``[llm]`` to the first committee.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path

from observe.actions import Refused

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "taic.toml"
_SAFE = re.compile(r"[A-Za-z0-9._:-]+")


def _config_path() -> Path:
    """@brief 取配置文件路径，环境变量 ``TAIC_CONFIG`` 可覆盖仓库内的 ``taic.toml``。

    @return 配置文件路径。
    """
    override = os.environ.get("TAIC_CONFIG")
    return Path(override) if override else CONFIG_PATH


def _read_config() -> tuple[Path, str, dict]:
    """@brief 读取并解析 ``taic.toml``，并确认存在 ``[llm]`` 段。

    @return 三元组：配置路径、原文、解析后的文档。
    @raises Refused 文件读不了、TOML 解析失败或缺 ``[llm]`` 段时抛出（HTTP 500）。
    """
    path = _config_path()
    try:
        text = path.read_text(encoding="utf-8")
        doc = tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise Refused(500, f"cannot read {path}: {e}") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("llm"), dict):
        raise Refused(500, f"{path} has no [llm] section")
    return path, text, doc


def _credentials(llm: dict) -> dict:
    """@brief 从 ``[llm]`` 指向的凭据文件里取出 provider 列表（不返回密钥）。

    @param llm ``taic.toml`` 中的 ``[llm]`` 段。
    @return provider 名称到其配置的字典。
    @raises Refused 凭据文件读不了、解析失败或没有 ``providers`` 字段时抛出（HTTP 500）。
    """
    raw = llm.get("credentials_file") or "~/.pi/agent/models.json"
    path = Path(os.path.expanduser(str(raw)))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(500, f"cannot read the model list from {path}: {e}") from e
    providers = doc.get("providers") if isinstance(doc, dict) else None
    if not isinstance(providers, dict):
        raise Refused(500, f"{path} has no providers")
    return providers


def _committee_rows(doc: dict) -> list[dict]:
    """@brief 按文件顺序列出委员会；成员信息不全也仍保留一行，以便表单说明原因。

    原始英文说明：The committees in file order. A missing list is empty; a bad member still leaves a row so the form can say why.

    @param doc 解析后的 ``taic.toml`` 文档。
    @return 每行含 ``name``、``provider``、``model`` 与 ``members`` 数量。
    """
    listed = doc.get("committees")
    if not isinstance(listed, list):
        return []
    rows = []
    for item in listed:
        if not isinstance(item, dict):
            continue
        members = item.get("members") if isinstance(item.get("members"), list) else []
        first = members[0] if members and isinstance(members[0], dict) else {}
        rows.append({
            "name": str(item.get("name") or ""),
            "provider": str(first.get("provider") or ""),
            "model": str(first.get("model") or ""),
            "members": len(members),
        })
    return rows


def model_choices() -> dict:
    """@brief 返回当前默认模型、两个委员会，以及凭据文件中的全部模型 id；不含密钥、URL 或价格。

    原始英文说明：The current default, the two committees, and every model id in the credentials file. No keys, URLs, or prices.

    @return 含 ``provider``、``model``、``choices`` 与 ``committees`` 的字典。
    """
    _path, _text, doc = _read_config()
    llm = doc["llm"]
    choices = []
    for provider, prov in _credentials(llm).items():
        if not isinstance(prov, dict):
            continue
        for model in prov.get("models") or []:
            if not isinstance(model, dict) or not model.get("id"):
                continue
            choices.append({"provider": str(provider), "id": str(model["id"]), "name": str(model.get("name") or model["id"])})
    committees = [
        {"name": row["name"], "provider": row["provider"], "model": row["model"]}
        for row in _committee_rows(doc) if row["name"] in ("first", "second")
    ]
    return {"provider": str(llm.get("provider") or ""), "model": str(llm.get("model") or ""), "choices": choices, "committees": committees}


def _replace_in_llm(text: str, key: str, value: str) -> tuple[str, bool]:
    """@brief 替换 ``[llm]`` 段内第一处 ``key = "..."``，只改到下一个段头之前。

    原始英文说明：Replace the first ``key = "..."`` line inside ``[llm]``, before the next section.

    @param text 配置文件原文。
    @param key 要改的键名，如 ``provider`` 或 ``model``。
    @param value 新值。
    @return 二元组：替换后的文本，以及是否确实替换过。
    """
    lines = text.splitlines(keepends=True)
    in_llm = False
    replaced = False
    out = []
    pattern = re.compile(rf'^(\s*{key}\s*=\s*")[^"]*(")')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_llm = stripped == "[llm]"
        if in_llm and not replaced and pattern.match(line):
            out.append(pattern.sub(rf"\g<1>{value}\2", line, count=1))
            replaced = True
            continue
        out.append(line)
    return "".join(out), replaced


def _replace_in_members(text: str, provider: str, model: str) -> str:
    """@brief 把所有委员会成员指向给定的 ``provider`` / ``model``；没有成员的文件保持不变。

    原始英文说明：Point every committee member at ``provider`` / ``model``. A file with no members is unchanged.

    @param text 配置文件原文。
    @param provider 新的 provider 名。
    @param model 新的模型 id。
    @return 替换后的文本。
    """
    lines = text.splitlines(keepends=True)
    in_member = False
    provider_pat = re.compile(r'^(\s*provider\s*=\s*")[^"]*(")')
    model_pat = re.compile(r'^(\s*model\s*=\s*")[^"]*(")')
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_member = stripped == "[[committees.members]]"
        if in_member and provider_pat.match(line):
            out.append(provider_pat.sub(rf"\g<1>{provider}\2", line, count=1))
            continue
        if in_member and model_pat.match(line):
            out.append(model_pat.sub(rf"\g<1>{model}\2", line, count=1))
            continue
        out.append(line)
    return "".join(out)


def _known(provider: str, model: str, choices: list[dict]) -> None:
    """@brief 校验 provider/model 是列表中的合法 id。

    @param provider provider 名。
    @param model 模型 id。
    @param choices ``model_choices`` 给出的候选列表。
    @raises Refused id 格式非法或不在候选列表里时抛出（HTTP 400）。
    """
    if not isinstance(provider, str) or not isinstance(model, str) or not _SAFE.fullmatch(provider) or not _SAFE.fullmatch(model):
        raise Refused(400, "provider and model must be the ids from the list")
    if not any(row["provider"] == provider and row["id"] == model for row in choices):
        raise Refused(400, f"unknown model {provider}/{model}")


def _assign_members(text: str, assignments: dict[str, tuple[str, str]]) -> tuple[str, dict[str, int]]:
    """@brief 逐个替换指定委员会下的唯一成员，其他委员会不动。

    原始英文说明：Replace the one member under each named committee. Other committees are left alone.

    @param text 配置文件原文。
    @param assignments 委员会名到 ``(provider, model)`` 的映射。
    @return 二元组：替换后的文本，以及每个委员会实际遇到的成员数。
    """
    lines = text.splitlines(keepends=True)
    section = ""
    pending = ""
    target = ""
    seen = {name: 0 for name in assignments}
    provider_pat = re.compile(r'^(\s*provider\s*=\s*")[^"]*(")')
    model_pat = re.compile(r'^(\s*model\s*=\s*")[^"]*(")')
    name_pat = re.compile(r'^\s*name\s*=\s*"([^"]*)"')
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped == "[[committees]]":
            section, pending, target = "committee", "", ""
        elif stripped == "[[committees.members]]":
            section = "member"
            if pending in assignments:
                seen[pending] += 1
                target = pending if seen[pending] == 1 else ""
            else:
                target = ""
        elif stripped.startswith("["):
            section, target = "", ""
        elif section == "committee":
            match = name_pat.match(stripped)
            if match:
                pending = match.group(1)
        if section == "member" and target:
            if provider_pat.match(line):
                line = provider_pat.sub(rf"\g<1>{assignments[target][0]}\2", line, count=1)
            elif model_pat.match(line):
                line = model_pat.sub(rf"\g<1>{assignments[target][1]}\2", line, count=1)
        out.append(line)
    return "".join(out), seen


def _write(path: Path, text: str) -> None:
    """@brief 先写临时文件再原子替换，避免半成品配置。

    @param path 目标文件路径。
    @param text 完整的新内容。
    """
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def set_default_model(provider: str, model: str) -> dict:
    """@brief 将同一个模型写入 ``[llm]`` 及每个委员会成员。

    原始英文说明：Write one model into ``[llm]`` and into every committee member.

    @param provider provider 名。
    @param model 模型 id。
    @return 含写入的 ``provider`` 与 ``model`` 的字典。
    @raises Refused 模型不在候选列表时抛 400；配置缺 ``[llm]`` 的 provider/model 行时抛 500。
    """
    current = model_choices()
    _known(provider, model, current["choices"])
    path, text, _doc = _read_config()
    new, n_provider = _replace_in_llm(text, "provider", provider)
    new, n_model = _replace_in_llm(new, "model", model)
    new = _replace_in_members(new, provider, model)
    if not n_provider or not n_model:
        raise Refused(500, f"{path} has no provider or model line under [llm]")
    _write(path, new)
    return {"provider": provider, "model": model}


def set_committees(rows) -> dict:
    """@brief 分别写入每个委员会的唯一成员，``[llm]`` 跟随第一个委员会；名称必须为 first 与 second。

    原始英文说明：Write each committee's one member. ``[llm]`` follows the first committee. Names must be first and second.

    @param rows 两个委员会行，每行含 ``name``、``provider``、``model``。
    @return 含默认 ``provider`` / ``model`` 与两个 ``committees`` 的字典。
    @raises Refused 行数不是两个、名称非法、模型未知或配置成员数不是一时抛出；写入缺行时抛 500。
    """
    if not isinstance(rows, list) or len(rows) != 2:
        raise Refused(400, "the body needs the first and second committees")
    assignments: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise Refused(400, "each committee needs a name, a provider and a model")
        name = row.get("name")
        if name not in ("first", "second") or name in assignments:
            raise Refused(400, "the two committees must be named first and second")
        assignments[name] = (row.get("provider"), row.get("model"))
    current = model_choices()
    for provider, model in assignments.values():
        _known(provider, model, current["choices"])
    path, text, doc = _read_config()
    counted = {row["name"]: row["members"] for row in _committee_rows(doc)}
    if counted.get("first") != 1 or counted.get("second") != 1:
        raise Refused(400, "each committee in the config must have exactly one member")
    new, seen = _assign_members(text, assignments)
    if seen.get("first") != 1 or seen.get("second") != 1:
        raise Refused(500, f"{path} does not have one member under first and under second")
    first = assignments["first"]
    new, n_provider = _replace_in_llm(new, "provider", first[0])
    new, n_model = _replace_in_llm(new, "model", first[1])
    if not n_provider or not n_model:
        raise Refused(500, f"{path} has no provider or model line under [llm]")
    _write(path, new)
    return {
        "provider": first[0],
        "model": first[1],
        "committees": [{"name": name, "provider": provider, "model": model} for name, (provider, model) in (("first", assignments["first"]), ("second", assignments["second"]))],
    }
