"""G6 unified dynamic situation block: three layers, one cleaner, one budget.

Overlay remains first, followed by the host ledger and unverified teammate
leads.  No TTL, confidence eviction, host summary, or route-M activation is
introduced.  Long sources are shortened peer-first, then overlay, with the host
ledger last; strict caller budgets and impossible fixed headings stop rendering
with a machine-readable receipt.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# The budget is the sum of the former related source caps (overlay, ledger,
# peer tips, inherited suggestions), not an additional allocation.  Fixed
# headings and cleaning output are charged against this same byte budget.
LEGACY_SOURCE_CAP_BYTES = (1800, 900, 1500, 1200)
DYNAMIC_BUDGET_BYTES = sum(LEGACY_SOURCE_CAP_BYTES)
SCHEMA = "taic.situation_block.1"
INJECTION_ORDER = ("overlay", "host_ledger", "peer_unverified")
_MAIN_HEADING = "\n\n# 本轮态势（动态信息，不是合同）\n"
_OVERLAY_HEADING = "## 已过门 overlay（仍是未验证的模型建议，不是事实）"
_LEDGER_HEADING = "最近的历史（宿主账本）："
_PEER_HEADER = "本任务前序避坑提示（队友交接，仅限本运行参考；凡未附探针文件名与实测数字的硬件/算法上限断言，一律按未验证处理，鼓励你写探针翻案）："
_OVERLAY_SOURCE_HEADING = "# 未验证的模型建议（不是合同，不是事实）"
_INHERIT_SOURCE_HEADING = "# 队友建议（待你复核，不是合同，不是已批准指令）"
_INHERIT_HEADING = "## 队友建议（待你复核，不是合同，不是已批准指令）"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(\s*)#+")


class SituationBudgetError(ValueError):
    """The unified dynamic block exceeded its hard UTF-8 byte budget."""

    def __init__(self, receipt: dict):
        self.receipt = receipt
        super().__init__(
            f"unified situation block is {receipt['candidate_bytes']} bytes; "
            f"the dynamic budget is {receipt['budget_bytes']} bytes"
        )


@dataclass(frozen=True)
class SituationRender:
    text: str
    receipt: dict


def _without_source_heading(text: str, heading: str) -> str:
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.strip() == heading:
            del lines[index]
        break
    return "\n".join(lines)


def _clean_text(text: str) -> str:
    """The single cleaning pipeline used for every dynamic layer."""
    normalized = _CONTROL.sub("", (text or "").replace("\r\n", "\n").replace("\r", "\n"))
    lines = []
    for line in normalized.split("\n"):
        if _FENCE.match(line):
            continue
        # Dynamic prose cannot create a Markdown instruction heading.  The
        # trusted structural headings are added after this pipeline.
        line = _HEADING.sub(lambda match: match.group(1) + "＃", line)
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def clean_situation_layers(*, overlay: str, ledger: str, peer_tips: str, inherit_prompt: str) -> dict[str, str]:
    """Clean all three layers through one pipeline, before adding host headings."""
    overlay_body = _without_source_heading(overlay, _OVERLAY_SOURCE_HEADING)
    inherit_body = _without_source_heading(inherit_prompt, _INHERIT_SOURCE_HEADING)
    cleaned = {
        "overlay": _clean_text(overlay_body),
        "host_ledger": _clean_text(ledger),
        "peer_tips": _clean_text(peer_tips),
        "inherit_prompt": _clean_text(inherit_body),
    }
    peer = cleaned["peer_tips"]
    inherit = cleaned["inherit_prompt"]
    cleaned["peer_unverified"] = "\n\n".join(part for part in (peer, inherit) if part)
    return cleaned


def _clip_utf8(text: str, limit: int) -> str:
    raw = (text or "").encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[: max(0, limit)].decode("utf-8", "ignore")


def render_situation(
    *,
    overlay: str = "",
    ledger: str = "",
    peer_tips: str = "",
    inherit_prompt: str = "",
    budget_bytes: int = DYNAMIC_BUDGET_BYTES,
    strict_budget: bool | None = None,
) -> SituationRender:
    """Clean first, then shorten peer, overlay, and finally the host ledger.

    The default budget never rejects a session merely because its source prose
    is long.  A caller-supplied smaller budget remains a strict contract and
    raises, as does a budget that cannot hold the fixed structural headings.
    """
    budget = int(budget_bytes)
    if budget < 0:
        raise ValueError("dynamic situation budget cannot be negative")
    cleaned = clean_situation_layers(
        overlay=overlay, ledger=ledger, peer_tips=peer_tips, inherit_prompt=inherit_prompt,
    )
    source = {
        "overlay": overlay or "",
        "host_ledger": ledger or "",
        "peer_unverified": "\n\n".join(part for part in (peer_tips or "", inherit_prompt or "") if part),
    }
    prefixes = {
        "overlay": "\n" + _OVERLAY_HEADING + "\n" if cleaned["overlay"] else "",
        "host_ledger": "\n\n" + _LEDGER_HEADING + "\n" if cleaned["host_ledger"] else "",
        "peer_unverified": "\n\n" if cleaned["peer_unverified"] else "",
    }
    if not any(prefixes.values()):
        receipt = {
            "schema": SCHEMA, "status": "empty", "delivery": "empty",
            "budget_bytes": budget, "source_bytes": 0, "sanitized_bytes": 0,
            "candidate_bytes": 0, "output_bytes": 0, "delivered_bytes": 0,
            "omitted_bytes": 0, "over_budget": False,
            "injection_order": list(INJECTION_ORDER), "layers": [],
        }
        return SituationRender("", receipt)

    full_contents = {
        key: cleaned[key]
        for key in ("overlay", "host_ledger", "peer_tips", "inherit_prompt")
    }

    def render_layer(name: str, contents: dict[str, str]) -> str:
        if name == "overlay":
            return prefixes[name] + contents["overlay"] if contents["overlay"] else ""
        if name == "host_ledger":
            return prefixes[name] + contents["host_ledger"] if contents["host_ledger"] else ""
        pieces = [prefixes[name]] if contents["peer_tips"] or contents["inherit_prompt"] else []
        if contents["peer_tips"]:
            pieces.append(_PEER_HEADER + "\n" + contents["peer_tips"])
        if contents["inherit_prompt"]:
            pieces.append(("\n\n" if contents["peer_tips"] else "") + _INHERIT_HEADING + "\n" + contents["inherit_prompt"])
        return "".join(pieces)

    def render_text(contents: dict[str, str]) -> tuple[str, dict[str, str]]:
        segments = {name: render_layer(name, contents) for name in INJECTION_ORDER}
        return _MAIN_HEADING + "".join(segments.values()) + "\n", segments

    full_text, full_segments = render_text(full_contents)
    fixed_text, _fixed_segments = render_text({key: "" for key in full_contents})
    candidate_bytes = len(full_text.encode("utf-8"))
    candidate_text_sha256 = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    fixed_bytes = len(fixed_text.encode("utf-8"))
    strict_mode = (budget < DYNAMIC_BUDGET_BYTES) if strict_budget is None else bool(strict_budget)
    strict_caller_budget = strict_mode and candidate_bytes > budget
    fixed_headings_overflow = budget >= DYNAMIC_BUDGET_BYTES and fixed_bytes > budget

    def stopped_render(reason: str) -> SituationRender:
        layers = []
        for name in INJECTION_ORDER:
            content = cleaned[name]
            segment = full_segments[name]
            rendered_bytes = len(segment.encode("utf-8"))
            layers.append({
                "name": name,
                "source_bytes": len(source[name].encode("utf-8")),
                "sanitized_bytes": len(content.encode("utf-8")),
                "rendered_bytes": rendered_bytes,
                "output_bytes": 0,
                "delivered_bytes": 0,
                "omitted_bytes": rendered_bytes,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest() if content else "",
                "status": "omitted" if segment else "empty",
            })
        receipt = {
            "schema": SCHEMA, "status": "stopped", "delivery": "stopped",
            "budget_bytes": budget,
            "source_bytes": sum(len(value.encode("utf-8")) for value in source.values()),
            "sanitized_bytes": sum(len(cleaned[name].encode("utf-8")) for name in INJECTION_ORDER),
            "candidate_bytes": candidate_bytes, "output_bytes": 0,
            "fixed_bytes": fixed_bytes, "delivered_bytes": 0,
            "omitted_bytes": candidate_bytes, "over_budget": True,
            "error_reason": reason,
            "candidate_text_sha256": candidate_text_sha256,
            "injection_order": list(INJECTION_ORDER),
            "shortening_order": ["peer_unverified", "overlay", "host_ledger"],
            "static_contract_omitted_bytes": 0, "layers": layers,
        }
        return SituationRender("", receipt)

    if strict_caller_budget or fixed_headings_overflow:
        reason = "caller_supplied_smaller_budget" if strict_caller_budget else "fixed_headings_exceed_budget"
        raise SituationBudgetError(stopped_render(reason).receipt)

    def shorten_peer_and_overlay(contents: dict[str, str]) -> int:
        excess = max(0, len(render_text(contents)[0].encode("utf-8")) - budget)
        # Peer leads lose bytes first; overlay is next. Within peer leads, trim
        # from the end (inherit, then tips). The ledger is handled only afterwards.
        for keys in (("peer_tips", "inherit_prompt"), ("overlay",)):
            for key in reversed(keys):
                if excess <= 0:
                    break
                current = len(contents[key].encode("utf-8"))
                if not current:
                    continue
                clipped = _clip_utf8(contents[key], max(0, current - excess))
                reduction = current - len(clipped.encode("utf-8"))
                contents[key] = clipped
                excess = max(0, excess - reduction)
            if excess <= 0:
                break
        return excess

    contents = dict(full_contents)
    excess = shorten_peer_and_overlay(contents)
    if excess > 0 and cleaned["host_ledger"]:
        from .plan_ledger import clip_ledger_text

        without_ledger = dict(contents)
        without_ledger["host_ledger"] = ""
        ledger_fixed_text, _ = render_text(without_ledger)
        # render_layer drops an empty ledger heading. Reserve that heading
        # explicitly because the nonempty clipped ledger will add it back.
        ledger_heading_bytes = len(prefixes["host_ledger"].encode("utf-8"))
        available = budget - len(ledger_fixed_text.encode("utf-8")) - ledger_heading_bytes
        contents["host_ledger"] = (
            clip_ledger_text(cleaned["host_ledger"], available) if available > 0 else ""
        )
    output_text, output_segments = render_text(contents)
    output_bytes = len(output_text.encode("utf-8"))

    # clip_ledger_text deliberately never splits a node line. If not even one
    # line fits, do not let that ledger erase peer/overlay text that was already
    # shortened to empty. Drop the ledger and its heading, then refit the
    # original cleaned peer/overlay layers against the same budget.
    other_layers = ("overlay", "peer_tips", "inherit_prompt")
    ledger_unusable = bool(cleaned["host_ledger"] and not contents["host_ledger"])
    if (ledger_unusable or output_bytes > budget) and any(full_contents[key] for key in other_layers):
        contents = dict(full_contents)
        contents["host_ledger"] = ""
        shorten_peer_and_overlay(contents)
        output_text, output_segments = render_text(contents)
        output_bytes = len(output_text.encode("utf-8"))

    if output_bytes > budget or not any(contents.values()):
        reason = "fixed_headings_and_ledger_do_not_fit" if cleaned["host_ledger"] else "fixed_headings_do_not_fit"
        return stopped_render(reason)

    layers = []
    for name in INJECTION_ORDER:
        content = cleaned[name]
        original = full_segments[name]
        output = output_segments[name]
        rendered_bytes = len(original.encode("utf-8"))
        fitted_bytes = len(output.encode("utf-8"))
        omitted = rendered_bytes - fitted_bytes
        layers.append({
            "name": name,
            "source_bytes": len(source[name].encode("utf-8")),
            "sanitized_bytes": len(content.encode("utf-8")),
            "rendered_bytes": rendered_bytes,
            "output_bytes": fitted_bytes,
            "delivered_bytes": 0,
            "omitted_bytes": omitted,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest() if content else "",
            "status": ("truncated" if omitted else "rendered") if original else "empty",
        })
    overflow = candidate_bytes > budget
    receipt = {
        "schema": SCHEMA,
        "status": "truncated" if overflow else "rendered",
        "delivery": "truncated" if overflow else "pending",
        "budget_bytes": budget,
        "source_bytes": sum(len(value.encode("utf-8")) for value in source.values()),
        "sanitized_bytes": sum(len(cleaned[name].encode("utf-8")) for name in INJECTION_ORDER),
        "candidate_bytes": candidate_bytes,
        "output_bytes": output_bytes,
        "fixed_bytes": fixed_bytes,
        "delivered_bytes": 0,
        "omitted_bytes": candidate_bytes - output_bytes,
        "over_budget": overflow,
        "candidate_text_sha256": candidate_text_sha256,
        "injection_order": list(INJECTION_ORDER),
        "shortening_order": ["peer_unverified", "overlay", "host_ledger"],
        "static_contract_omitted_bytes": 0,
        "layers": layers,
    }
    return SituationRender(output_text, receipt)


def situation_for_brief(
    brief, *, budget_bytes: int = DYNAMIC_BUDGET_BYTES,
    strict_budget: bool | None = None,
) -> SituationRender:
    """Resolve layer inputs while preserving the seedless delivery boundary."""
    seedless = bool(getattr(brief, "seedless", False))
    return render_situation(
        overlay="" if seedless else str(getattr(brief, "harness_overlay", "") or ""),
        ledger=(
            "" if seedless or not getattr(brief, "use_task_ledger", False)
            else str(getattr(brief, "ledger_text", "") or "")
        ),
        peer_tips=str(getattr(brief, "peer_tips", "") or ""),
        inherit_prompt="" if seedless else str(getattr(brief, "inherit_prompt", "") or ""),
        budget_bytes=budget_bytes,
        strict_budget=strict_budget,
    )
