"""Two committees take turns. Each committee has one member today: a provider and a model.

Handoff is between committees. A later round does not receive the previous round's chat. This module does not import
another project. A connection failure is still retried inside ``taic.llm``; it does not switch committees.
"""

# 搜索只有这一种跑法：first 与 second 两个委员会。每个委员会的 members 列表今天必须恰好一名。
# 多于一名就在启动时拒绝，不能悄悄只用第一名。密钥和端点只活在 LLMConfig 里，不进运行目录。

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .llm import LLMConfig, LLMError

_NAMES = ("first", "second")


@dataclass
class Member:
    """@brief 一名成员：一个 provider 加一个 model。

    ``cfg`` 是解析后的客户端设置，不记录进运行目录。
    """

    provider: str
    model: str
    cfg: LLMConfig = field(repr=False)


@dataclass
class Committee:
    """@brief 一个委员会：名字加成员列表。

    ``members`` 是列表，以后改成多个成员时不需要另写一套交棒循环。
    """

    name: str
    members: list[Member]


def handoff_for(name: str) -> str:
    """@brief 接棒时拼在后面轮次计划与补丁前的句子。

    只说委员会名字，不含任何秘密。

    @param name 委员会名（``first`` 或 ``second``）。
    @return 以换行结尾的接棒说明。
    """
    return (
        f"You are the {name} committee; the previous turn's conversation is not included, "
        "so continue from the current kernel and the measured outcomes.\n\n"
    )


def prefix_handoff(prefix: str, text: str) -> str:
    """@brief 拼上接棒句子。

    ``prefix`` 为空时原样返回 ``text``，使第 1 轮与提示函数逐字节一致。

    @param prefix 接棒句子。
    @param text 原本的提示文本。
    @return 拼好的文本。
    """
    if not prefix:
        return text
    return prefix + text


def committees_record(committees: list[Committee]) -> list[dict]:
    """@brief 生成 ``run_config.json`` 里的委员会列表。

    只含名字、provider 与 model，不含密钥和 base URL。

    @param committees 两个委员会。
    @return 可直接写入 JSON 的列表。
    """
    return [
        {"name": c.name, "members": [{"provider": m.provider, "model": m.model} for m in c.members]}
        for c in committees
    ]


def default_committees(active: LLMConfig, provider: str = "deepseek") -> list[Committee]:
    """@brief 生成两个复制 ``active`` 的委员会。

    配置里没有列出 ``committees`` 时使用。

    @param active 当前生效的客户端配置。
    @param provider 两个委员会共用的 provider 名。
    @return ``[first, second]`` 两个委员会。
    """
    return [
        Committee(name, [Member(provider, active.model, replace(active))])
        for name in _NAMES
    ]


def order_committees(committees: list[Committee]) -> list[Committee]:
    """@brief 校验并排成 ``[first, second]`` 顺序。

    成员数不是 1 在这里就被拒绝，发生在这场搜索的任何模型调用之前。

    @param committees 待校验的委员会列表。
    @return 按 ``first``、``second`` 排好的两个委员会。
    @raises LLMError 数量不是两个、名字重复或不叫 first/second、或某个委员会成员数不为 1 时。
    """
    if len(committees) != 2:
        raise LLMError(f"a search needs exactly two committees, named first and second (got {len(committees)})")
    by: dict[str, Committee] = {}
    for committee in committees:
        if committee.name in by:
            raise LLMError(f"duplicate committee {committee.name!r}")
        by[committee.name] = committee
    if set(by) != set(_NAMES):
        raise LLMError(f"committees must be named first and second, got {sorted(by)}")
    ordered = [by[name] for name in _NAMES]
    for committee in ordered:
        count = len(committee.members)
        if count != 1:
            raise LLMError(
                f"committee {committee.name!r} has {count} members; only one member per committee is defined, "
                "and a longer list is not reduced to its first member"
            )
    return ordered


def load_committees(raw: dict | None, *, active: LLMConfig, model_override: str | None = None) -> list[Committee]:
    """@brief 解出一场运行的委员会。

    缺 ``committees`` 键时把 ``active`` 复制两份（即 ``[llm]`` 的 provider 与其模型）。``model_override``（命令行的
    ``--model``）做同样的事，让一个被指定的模型被两个委员会共用，交棒照旧发生。列出的委员会成员数不是 1 时，
    在读凭据文件之前就报错。另一个 provider 从 ``load_config`` 用的同一个凭据文件里解析。

    @param raw 配置文件的原始字典（可为 ``None``）。
    @param active 当前生效的客户端配置。
    @param model_override 命令行指定的模型；给出时两个委员会都用它。
    @return 按 first/second 排好的两个委员会。
    @raises LLMError 委员会结构不合法、成员数不为 1，或另一 provider 缺凭据时。
    """
    doc = raw or {}
    provider = str((doc.get("llm") or {}).get("provider") or "deepseek")
    listed = doc.get("committees") if "committees" in doc else None
    if listed is None:
        return default_committees(active, provider)
    shape = _shape(listed)  # refuses a member count other than 1 before any credentials read
    if model_override:
        return default_committees(active, provider)
    llm = doc.get("llm") if isinstance(doc.get("llm"), dict) else {}
    active_provider = str(llm.get("provider") or "deepseek")
    cred_file = Path(os.path.expanduser(str(llm.get("credentials_file") or "~/.pi/agent/models.json")))
    cache: dict[str, dict] = {}
    built = [
        Committee(name, [ _member(provider, model, active=active, active_provider=active_provider, cred_file=cred_file, cache=cache, reasoning_effort=effort) ])
        for name, provider, model, effort in shape
    ]
    return order_committees(built)


def _shape(listed: object) -> list[tuple[str, str, str]]:
    """@brief 把配置里的 ``committees`` 表解成四元组。

    成员数在读任何凭据之前就检查。

    @param listed 配置里的 ``committees`` 值。
    @return 每项为 ``(name, provider, model, reasoning_effort 或 None)`` 的列表。
    @raises LLMError 结构、名字、成员列表或 provider/model 不合法，或成员数不为 1 时。
    """
    if not isinstance(listed, list):
        raise LLMError("committees must be a list of two committees")
    parsed: list[tuple[str, str, str]] = []
    for item in listed:
        if not isinstance(item, dict):
            raise LLMError("each committee must be a table with a name and a members list")
        name = item.get("name")
        if not isinstance(name, str) or name not in _NAMES:
            raise LLMError(f"committees must be named first and second, got {name!r}")
        members = item.get("members")
        if not isinstance(members, list):
            raise LLMError(f"committee {name!r} members must be a list")
        if len(members) != 1:
            raise LLMError(
                f"committee {name!r} has {len(members)} members; only one member per committee is defined, "
                "and a longer list is not reduced to its first member"
            )
        member = members[0]
        if not isinstance(member, dict):
            raise LLMError(f"committee {name!r} member must be a table with provider and model")
        provider = member.get("provider")
        model = member.get("model")
        if not isinstance(provider, str) or not provider.strip() or not isinstance(model, str) or not model.strip():
            raise LLMError(f"committee {name!r} member needs a provider and a model")
        parsed.append((name, provider.strip(), model.strip(), member.get("reasoning_effort") if isinstance(member.get("reasoning_effort"), str) else None))
    return parsed


def _member(provider: str, model: str, *, active: LLMConfig, active_provider: str, cred_file: Path, cache: dict[str, dict], reasoning_effort: str | None = None) -> Member:
    """@brief 为一个委员会成员构造客户端配置。

    与当前 provider/model 相同时直接复制 ``active``；否则从凭据文件取 base URL、密钥与单价。

    @param provider 成员使用的 provider。
    @param model 成员使用的模型。
    @param active 当前生效的客户端配置。
    @param active_provider 当前生效配置的 provider 名。
    @param cred_file 凭据文件路径。
    @param cache 已解析的凭据表缓存，避免重复读盘。
    @param reasoning_effort 推理强度；给出时覆盖配置里的值。
    @return 构造好的 ``Member``。
    @raises LLMError 另一个 provider 在凭据文件里既没有 base URL 也没有密钥时。
    """
    if provider == active_provider and model == active.model:
        cfg = replace(active)
    else:
        table = _providers(cred_file, cache)
        prov = table.get(provider)
        prov = prov if isinstance(prov, dict) else {}
        price = _model_price(prov, model)
        if provider == active_provider:
            base_url, api_key = active.base_url, active.api_key
        else:
            base_url, api_key = prov.get("baseUrl"), prov.get("apiKey")
        if not base_url or not api_key:
            raise LLMError(f"no credentials: set OPENAI_API_KEY/OPENAI_BASE_URL or provide provider '{provider}' in {cred_file}")
        cfg = _priced(active, base_url=str(base_url), api_key=str(api_key), model=model, price=price)
    if reasoning_effort:
        cfg = replace(cfg, reasoning_effort=reasoning_effort)
    return Member(provider, model, cfg)


def _model_price(prov: dict, model: str) -> dict:
    """@brief 从 provider 表里取出某个模型的单价表。

    @param prov 该 provider 的凭据与模型表。
    @param model 模型 id。
    @return 单价字典；找不到时为空字典。
    """
    for row in prov.get("models") or []:
        if isinstance(row, dict) and row.get("id") == model:
            cost = row.get("cost") or {}
            return cost if isinstance(cost, dict) else {}
    return {}


def _providers(path: Path, cache: dict[str, dict]) -> dict:
    """@brief 读凭据文件里的 ``providers`` 表，结果按路径缓存。

    @param path 凭据文件路径。
    @param cache 进程内缓存，避免同一场搜索反复读盘。
    @return provider 名到其配置的映射；文件缺失或结构不对时为空字典。
    """
    key = str(path)
    if key in cache:
        return cache[key]
    if not path.exists():
        cache[key] = {}
        return {}
    doc = json.loads(path.read_text())
    table = doc.get("providers") if isinstance(doc, dict) else None
    cache[key] = table if isinstance(table, dict) else {}
    return cache[key]


def _priced(active: LLMConfig, *, base_url: str, api_key: str, model: str, price: dict) -> LLMConfig:
    """@brief 用新 provider 的地址、密钥、模型与单价派生一份客户端配置。

    @param active 当前生效的客户端配置，作为其余字段的基底。
    @param base_url 新 provider 的基地址。
    @param api_key 新 provider 的密钥。
    @param model 新模型 id。
    @param price 单价字典（``input``/``output``/``cacheRead``/``cacheWrite``）。
    @return 新的 ``LLMConfig``。
    """
    return replace(
        active,
        base_url=base_url,
        api_key=api_key,
        model=model,
        price_in=float(price.get("input", 0.30)),
        price_out=float(price.get("output", 1.20)),
        price_cache_read=float(price.get("cacheRead", price.get("input", 0.30))),
        # 和 llm.py 的 load_config 用同一个表达式。只为第二委员会换 provider 时不会丢掉 cache-write 单价。
        price_cache_write=float(price.get("cacheWrite", price.get("input", 0.30)) or 0.0),
    )
