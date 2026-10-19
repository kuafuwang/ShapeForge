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
    """One provider and model. ``cfg`` is the resolved client settings and is not recorded."""

    provider: str
    model: str
    cfg: LLMConfig = field(repr=False)


@dataclass
class Committee:
    """Who takes a turn. ``members`` is a list so a later change does not need a new turn-taking loop."""

    name: str
    members: list[Member]


def handoff_for(name: str) -> str:
    """The sentence prepended to a later round's plan and patch. It names the committee and nothing secret."""
    return (
        f"You are the {name} committee; the previous turn's conversation is not included, "
        "so continue from the current kernel and the measured outcomes.\n\n"
    )


def prefix_handoff(prefix: str, text: str) -> str:
    """Return ``text`` itself when ``prefix`` is empty, so round 1 stays byte-identical to the prompt function."""
    if not prefix:
        return text
    return prefix + text


def committees_record(committees: list[Committee]) -> list[dict]:
    """The run_config list: names, providers and models. No keys and no base URLs."""
    return [
        {"name": c.name, "members": [{"provider": m.provider, "model": m.model} for m in c.members]}
        for c in committees
    ]


def default_committees(active: LLMConfig, provider: str = "deepseek") -> list[Committee]:
    """Two committees that copy ``active``. Used when a config file does not list committees."""
    return [
        Committee(name, [Member(provider, active.model, replace(active))])
        for name in _NAMES
    ]


def order_committees(committees: list[Committee]) -> list[Committee]:
    """``[first, second]``, or an error. A member count other than 1 is refused here, before any model call."""
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
    """The two committees for a run.

    A missing ``committees`` key copies ``active`` twice (the ``[llm]`` provider and its model). ``model_override``
    (the command's ``--model``) does the same, so one requested model is used by both committees and handoff still happens.
    A listed committee whose member count is not 1 raises before the credentials file is read. Another provider is
    resolved from the same credentials file ``load_config`` uses.
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
        Committee(name, [ _member(provider, model, active=active, active_provider=active_provider, cred_file=cred_file, cache=cache) ])
        for name, provider, model in shape
    ]
    return order_committees(built)


def _shape(listed: object) -> list[tuple[str, str, str]]:
    """``(name, provider, model)`` for first and second. Member count is checked before any credentials read."""
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
        parsed.append((name, provider.strip(), model.strip()))
    return parsed


def _member(provider: str, model: str, *, active: LLMConfig, active_provider: str, cred_file: Path, cache: dict[str, dict]) -> Member:
    if provider == active_provider and model == active.model:
        return Member(provider, model, replace(active))
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
    return Member(provider, model, cfg)


def _model_price(prov: dict, model: str) -> dict:
    for row in prov.get("models") or []:
        if isinstance(row, dict) and row.get("id") == model:
            cost = row.get("cost") or {}
            return cost if isinstance(cost, dict) else {}
    return {}


def _providers(path: Path, cache: dict[str, dict]) -> dict:
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
    return replace(
        active,
        base_url=base_url,
        api_key=api_key,
        model=model,
        price_in=float(price.get("input", 0.30)),
        price_out=float(price.get("output", 1.20)),
        price_cache_read=float(price.get("cacheRead", price.get("input", 0.30))),
    )
