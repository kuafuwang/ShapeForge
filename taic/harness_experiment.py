"""Planning and evidence for controlled harness experiments.

Planning and reading results are offline. This module does not launch a model or a GPU.
A caller that executes a plan must make that authorization explicit and supply identical
control/treatment conditions; the records here make that obligation inspectable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import random
import statistics
from pathlib import Path

from .acceptance import digest, read_spec

SCHEMA_VERSION = 1
METRICS = ("first_eligible_cost_usd", "first_eligible_elapsed_s", "tool_success_rate", "invalid_feedback_rate", "confirmed_speedup")
GUARDRAILS = ("safety_regressions", "correctness_regressions", "agent_failures", "additional_review_cost_usd")


class ExperimentError(ValueError):
    """An experiment plan or result is incomplete, inconsistent or untrusted."""


def _text(value, name: str, default: str = "") -> str:
    result = str(value if value is not None else default).strip()
    if not result:
        raise ExperimentError(f"{name} is required")
    return result[:2000]


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ExperimentError(f"{name} needs a sha256")
    return value


def _review_for_spec(run_dir: Path, proposal_key: str, spec_sha256: str) -> dict:
    path = Path(run_dir) / "harness_reviews.jsonl"
    reviews = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("event") == "review" and row.get("proposal_key") == proposal_key and row.get("useful") is True and row.get("cost_usd") is not None:
                reviews.append(row)
    if not reviews:
        raise ExperimentError("an implementation needs an independent useful review with a known cost")
    from .acceptance import make_spec
    proposals = []
    proposal_path = Path(run_dir) / "harness_proposals.jsonl"
    for line in proposal_path.read_text(encoding="utf-8", errors="replace").splitlines() if proposal_path.is_file() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("status") == "valid" and row.get("proposal_key") == proposal_key:
            proposals.append(row)
    if not proposals:
        raise ExperimentError("the source proposal is missing")
    for review in reviews:
        if make_spec(proposals[-1], review)["spec_sha256"] == spec_sha256:
            return review
    raise ExperimentError("the acceptance contract does not bind the current useful review")


def config_template(run_config: dict, *, budget_usd: float, base_commit: str = "") -> dict:
    """Freeze shared experimental conditions, excluding run and source-tree identity."""
    if not isinstance(run_config, dict):
        raise ExperimentError("run_config must be an object")
    search = run_config.get("search")
    if not isinstance(search, dict):
        raise ExperimentError("run_config has no search settings")
    target = run_config.get("target") if isinstance(run_config.get("target"), dict) else {}
    for key in ("task", "dtype", "size"):
        if not run_config.get(key):
            raise ExperimentError(f"run_config is missing {key}")
    if not target.get("id"):
        raise ExperimentError("run_config has no target id")
    llm = run_config.get("llm") if isinstance(run_config.get("llm"), dict) else {}
    committees = run_config.get("committees")
    if not isinstance(committees, list) or len(committees) != 2:
        raise ExperimentError("an experiment needs both recorded committees")
    shared_search = {key: value for key, value in search.items() if key not in {"harness_base_commit", "harness_experiment_id", "harness_experiment_arm", "harness_change_sha256", "harness_change_path", "eval_cache"}}
    shared_search.update({"budget_usd": float(budget_usd), "eval_cache": False})
    return {
        "schema_version": 1, "run_config_schema_version": run_config.get("schema_version"), "task": run_config["task"], "dtype": run_config["dtype"], "size": run_config["size"],
        "target": {"id": target["id"], "backend": target.get("backend"), "backend_version": target.get("backend_version"), "display_name": target.get("display_name")},
        "llm": llm, "search": shared_search, "committees": committees,
        "task_fingerprint": run_config.get("task_fingerprint"),
        "base_commit": base_commit or run_config.get("base_commit") or "",
        "price_snapshot": run_config.get("price_snapshot") or run_config.get("llm_prices") or {},
        "protocol_version": str(run_config.get("protocol_version") or ""),
        "treatment_guidance": run_config.get("treatment_guidance") or {},
    }


def arm_config(plan: dict, arm: str, seed: int) -> dict:
    """Materialize one offline config for a future, explicitly authorized run."""
    read_plan(plan)
    if plan["status"] != "authorized" or arm not in {"control", "treatment"} or seed not in plan["seeds"]:
        raise ExperimentError("an arm config needs an authorized plan, arm and paired seed")
    config = json.loads(json.dumps(plan["config_template"]))
    config["run_seed"] = seed
    config["seed"] = seed
    config["schema_version"] = config.pop("run_config_schema_version")
    config["protocol_version"] = plan["protocol_version"]
    config["harness_experiment"] = {"experiment_id": plan["experiment_id"], "arm": arm, "change_sha256": plan[arm]["change_sha256"] or "", "change_path": ""}
    config["search"].update({"harness_base_commit": plan["base_commit"], "harness_experiment_id": plan["experiment_id"], "harness_experiment_arm": arm, "harness_change_sha256": plan[arm]["change_sha256"] or "", "harness_change_path": ""})
    return config


def raw_arm_config(plan: dict, arm: str) -> dict:
    """Return the input TOML shape, retaining all committee members and no credentials."""
    read_plan(plan)
    if arm not in {"control", "treatment"}:
        raise ExperimentError("arm must be control or treatment")
    template = plan["config_template"]
    llm = dict(template.get("llm") or {})
    llm.pop("api_key", None)
    llm.pop("base_url", None)
    search = dict(template["search"])
    search.update({"harness_base_commit": plan["base_commit"], "harness_experiment_id": plan["experiment_id"], "harness_experiment_arm": arm, "harness_change_sha256": plan[arm]["change_sha256"] or "", "harness_change_path": ""})
    return {"llm": llm, "search": search, "committees": json.loads(json.dumps(template["committees"]))}


def materialize_arm(plan: dict, arm: str, seed: int) -> dict:
    """Produce inspectable raw config and the exact run_one arguments for one cell."""
    raw = raw_arm_config(plan, arm)
    if seed not in plan["seeds"]:
        raise ExperimentError("cell seed is not in the paired plan")
    return {
        "raw": raw,
        "run_args": {
            "task": plan["config_template"]["task"], "dtype": plan["config_template"]["dtype"], "size": plan["config_template"]["size"],
            "target_id": plan["scope"]["target_id"], "seed": seed, "budget": plan["budget_per_arm_usd"], "preset": None,
            **raw["search"],
        },
        "input_config_sha256": digest(raw),
        "base_commit": plan.get("base_commit") or plan["config_template"].get("base_commit"),
        "change_sha256": plan[arm]["change_sha256"],
    }


def toml_document(raw: dict) -> str:
    """Serialize a raw config without adding a TOML dependency or credentials."""
    def scalar(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return "[" + ", ".join(scalar(item) for item in value if item is not None) + "]"
        if isinstance(value, dict):
            return "{ " + ", ".join(f"{key} = {scalar(item)}" for key, item in value.items() if item is not None) + " }"
        if value is None:
            raise ExperimentError("TOML cannot represent null at this position")
        raise ExperimentError(f"TOML cannot represent {type(value).__name__}")
    lines = []
    for section in ("llm", "search"):
        lines.append(f"[{section}]")
        for key, value in raw.get(section, {}).items():
            if value is not None:
                lines.append(f"{key} = {scalar(value)}")
        lines.append("")
    for committee in raw["committees"]:
        lines.append("[[committees]]")
        lines.append(f"name = {scalar(committee['name'])}")
        for member in committee["members"]:
            lines.append("[[committees.members]]")
            for key, value in member.items():
                if value is not None:
                    lines.append(f"{key} = {scalar(value)}")
            lines.append("")
    return "\n".join(lines)


def evaluation_rules(*, min_pairs: int = 3, min_effect: float = 0.0, direction: str = "lower", max_missing_fraction: float = 0.0, guardrail_limits: dict | None = None, bootstrap_samples: int = 2000) -> dict:
    if not isinstance(min_pairs, int) or min_pairs < 3 or min_pairs > 100:
        raise ExperimentError("min_pairs must be between 3 and 100")
    if direction not in {"lower", "higher"}:
        raise ExperimentError("effect direction must be lower or higher")
    if not isinstance(min_effect, (int, float)) or isinstance(min_effect, bool) or not math.isfinite(min_effect) or min_effect < 0:
        raise ExperimentError("min_effect must be finite and non-negative")
    if not isinstance(max_missing_fraction, (int, float)) or not 0 <= max_missing_fraction < 1:
        raise ExperimentError("max_missing_fraction must be in [0, 1)")
    return {"min_pairs": min_pairs, "min_effect": float(min_effect), "direction": direction, "max_missing_fraction": float(max_missing_fraction), "guardrail_limits": guardrail_limits or {"safety_regressions": 0, "correctness_regressions": 0}, "bootstrap_samples": bootstrap_samples}


def _plan_cells(seeds, repetitions) -> list[dict]:
    """@brief 由冻结的 seeds/repetitions 推出每个 cell 的配对身份。

    身份是 ``(pair_index, seed, repetition)``：``pair_index`` 按 seed 首次出现的顺序分配，
    ``repetition`` 是同一 seed 第几次出现，``order`` 是 cell 在计划里的顺序（只用于报告时间趋势，
    不参与配对）。每个 seed 必须恰好出现 ``repetitions`` 次，于是
    ``len(seeds) == pair_count * repetitions``，重复只来自事前声明的 repetition。

    @param seeds 计划冻结的 seed 列表。
    @param repetitions 每对的重复次数。
    @return 按计划顺序排列的 cell 身份列表。
    @raises ExperimentError seed 数量、重复模式或 repetitions 不合法时。
    """
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or not 1 <= repetitions <= 100:
        raise ExperimentError("repetitions must be between 1 and 100")
    if not isinstance(seeds, list) or not seeds or len(seeds) > 64 or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 2**64 for seed in seeds):
        raise ExperimentError("paired seeds are invalid")
    counts: dict[int, int] = {}
    pair_of: dict[int, int] = {}
    cells = []
    for order, seed in enumerate(seeds):
        repetition = counts.get(seed, 0)
        if repetition >= repetitions:
            raise ExperimentError("a seed repeats more often than the plan's repetitions")
        counts[seed] = repetition + 1
        pair_of.setdefault(seed, len(pair_of))
        cells.append({"seed": seed, "repetition": repetition, "pair_index": pair_of[seed], "order": order})
    if any(count != repetitions for count in counts.values()) or len(seeds) != len(counts) * repetitions:
        raise ExperimentError("seeds must provide one entry per pair and repetition")
    return cells


def plan_cells(plan: dict) -> list[dict]:
    """@brief 返回计划冻结的 cell 身份列表（先读回并校验计划）。

    @param plan 实验计划。
    @return ``{seed, repetition, pair_index, order}`` 列表，顺序即执行顺序。
    """
    read_plan(plan)
    return _plan_cells(plan["seeds"], plan["repetitions"])


def make_plan(spec: dict, *, change_sha256: str, protocol_version: str, config_sha256: str, seeds: list[int], budget_usd: float, repetitions: int = 1, config_template_document: dict | None = None, rules: dict | None = None, base_commit: str = "") -> dict:
    """Bind one proposed treatment to the exact implementation and shared configuration."""
    spec = read_spec(spec)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1 or repetitions > 100:
        raise ExperimentError("repetitions must be between 1 and 100")
    cells = _plan_cells(seeds, repetitions)
    pair_count = len({cell["pair_index"] for cell in cells})
    if isinstance(budget_usd, bool) or not isinstance(budget_usd, (int, float)) or not math.isfinite(budget_usd) or budget_usd <= 0:
        raise ExperimentError("budget_usd must be finite and positive")
    if base_commit and not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise ExperimentError("base_commit must be a full commit hash")
    common = {"target_id": str(spec["scope"].get("target_id") or ""), "task": str(spec["scope"].get("task") or ""), "dtype": str(spec["scope"].get("dtype") or "")}
    if not all(common.values()):
        raise ExperimentError("the acceptance scope needs target_id, task and dtype")
    if not isinstance(config_template_document, dict) or not isinstance(config_template_document.get("search"), dict):
        raise ExperimentError("an experiment needs a canonical run_config template")
    expected_config = digest(config_template_document)
    if config_sha256 != expected_config:
        raise ExperimentError("config_sha256 does not bind the canonical run_config template")
    if spec.get("risk_tier") != "paired_required":
        raise ExperimentError("only a paired_required change may enter a search A/B plan")
    if spec.get("kind") == "hardware_premise":
        raise ExperimentError("hardware premise evidence stays in the probe/premise path")
    plan = {
        "schema_version": 2,
        "proposal_key": spec["proposal_key"], "spec_sha256": spec["spec_sha256"],
        "acceptance_spec": spec,
        "change_sha256": _digest(change_sha256, "change_sha256"),
        "protocol_version": _text(protocol_version, "protocol_version"),
        "config_sha256": _digest(config_sha256, "config_sha256"),
        "config_template": config_template_document,
        "base_commit": base_commit,
        "pair_count": pair_count,
        "primary_metric": spec["effect_metric_id"],
        "evaluation_rules": rules or evaluation_rules(),
        "scope": common,
        "control": {"change_sha256": None, "description": spec["control"]},
        "treatment": {"change_sha256": change_sha256},
        "seeds": list(seeds), "repetitions": repetitions,
        "budget_per_arm_usd": float(budget_usd),
        "max_total_budget_usd": float(budget_usd) * 2 * len(seeds),
        "effect_metric": spec["effect_metric"],
        "effect_metric_id": spec["effect_metric_id"],
        "guardrail_metrics": spec["guardrail_metrics"],
        "rollback": spec["rollback"],
        "authorization": {"allow_model_spend": False, "allow_gpu": False},
        "status": "dry_run",
    }
    plan["experiment_id"] = digest(plan)[:24]
    return plan


def read_plan(document: dict) -> dict:
    if isinstance(document, dict) and document.get("schema_version") == 1:
        if not isinstance(document.get("acceptance_spec"), dict) or document["acceptance_spec"].get("kind") == "hardware_premise":
            raise ExperimentError("legacy hardware premise cannot be a search A/B plan")
        upgraded = json.loads(json.dumps(document))
        upgraded["acceptance_spec"] = read_spec(upgraded["acceptance_spec"])
        upgraded["schema_version"] = 2
        upgraded["base_commit"] = ""
        upgraded.setdefault("repetitions", 1)
        upgraded["pair_count"] = len({cell["pair_index"] for cell in _plan_cells(upgraded.get("seeds"), upgraded["repetitions"])})
        upgraded["primary_metric"] = upgraded["effect_metric_id"]
        upgraded["evaluation_rules"] = evaluation_rules()
        upgraded["experiment_id"] = digest({key: value for key, value in upgraded.items() if key != "experiment_id"})[:24]
        return read_plan(upgraded)
    if not isinstance(document, dict) or document.get("schema_version") != 2:
        raise ExperimentError("unsupported experiment schema")
    spec_keys = ("schema_version", "proposal_key", "spec_sha256", "acceptance_spec", "change_sha256", "protocol_version", "config_sha256", "config_template", "base_commit", "pair_count", "primary_metric", "evaluation_rules", "scope", "control", "treatment", "seeds", "repetitions", "budget_per_arm_usd", "max_total_budget_usd", "effect_metric", "effect_metric_id", "guardrail_metrics", "rollback", "authorization", "status", "experiment_id")
    if set(document) != set(spec_keys):
        raise ExperimentError("experiment plan fields do not match the schema")
    try:
        spec = read_spec(document["acceptance_spec"])
        if document["spec_sha256"] != spec["spec_sha256"] or document["proposal_key"] != spec["proposal_key"]:
            raise ExperimentError("experiment does not bind its acceptance spec")
        if spec.get("risk_tier") != "paired_required":
            raise ExperimentError("this acceptance class does not enter search A/B")
        if document["primary_metric"] != document["effect_metric_id"]:
            raise ExperimentError("primary metric is inconsistent")
        evaluation_rules(**document["evaluation_rules"])
        if document["base_commit"] and not re.fullmatch(r"[0-9a-f]{40}", document["base_commit"]):
            raise ExperimentError("invalid base commit")
        if document["scope"] != spec["scope"] or document["effect_metric"] != spec["effect_metric"] or document["guardrail_metrics"] != spec["guardrail_metrics"] or document["rollback"] != spec["rollback"]:
            raise ExperimentError("experiment terms differ from the acceptance contract")
        _digest(document["change_sha256"], "change_sha256")
        _digest(document["config_sha256"], "config_sha256")
        if not isinstance(document["config_template"], dict) or document["config_sha256"] != digest(document["config_template"]):
            raise ExperimentError("config digest does not bind the run_config template")
        if not isinstance(document["proposal_key"], str) or not re.fullmatch(r"[0-9a-f]{16}", document["proposal_key"]):
            raise ExperimentError("invalid proposal key")
        if not isinstance(document["protocol_version"], str) or not document["protocol_version"]:
            raise ExperimentError("protocol_version is required")
        if not isinstance(document["control"], dict) or not isinstance(document["treatment"], dict):
            raise ExperimentError("control and treatment must be objects")
        if document["control"].get("description") != spec["control"]:
            raise ExperimentError("control description differs from the acceptance contract")
        seeds = document["seeds"]
        repetitions = document["repetitions"]
        if not isinstance(document["repetitions"], int) or isinstance(document["repetitions"], bool) or not 1 <= document["repetitions"] <= 100:
            raise ExperimentError("repetitions must be between 1 and 100")
        cells = _plan_cells(seeds, repetitions)
        pair_count = len({cell["pair_index"] for cell in cells})
        if document["pair_count"] != pair_count or len(seeds) != pair_count * repetitions:
            raise ExperimentError("pair count is inconsistent with the declared seeds and repetitions")
        budget = document["budget_per_arm_usd"]
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0 or document["max_total_budget_usd"] != budget * 2 * len(document["seeds"]):
            raise ExperimentError("experiment budget is inconsistent")
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentError(f"invalid experiment binding: {exc}") from exc
    expected = digest({key: value for key, value in document.items() if key != "experiment_id"})[:24]
    if document["experiment_id"] != expected:
        raise ExperimentError("experiment id does not bind its contents")
    if not isinstance(document["control"], dict) or not isinstance(document["treatment"], dict) or document["control"].get("change_sha256") is not None or document["treatment"].get("change_sha256") != document["change_sha256"]:
        raise ExperimentError("only the treatment may carry the proposed change")
    if document["status"] not in {"dry_run", "authorized", "completed", "inconclusive"}:
        raise ExperimentError("unknown experiment status")
    auth = document["authorization"]
    if not isinstance(auth, dict) or set(auth) - {"allow_model_spend", "allow_gpu", "approved_by"}:
        raise ExperimentError("invalid experiment authorization")
    if document["status"] == "dry_run":
        if auth.get("allow_model_spend") is not False or auth.get("allow_gpu") is not False:
            raise ExperimentError("dry-run plans cannot carry execution authorization")
    elif document["status"] == "authorized":
        if auth.get("allow_model_spend") is not True or auth.get("allow_gpu") is not True or not isinstance(auth.get("approved_by"), str) or auth["approved_by"].strip() in {"", "cli"}:
            raise ExperimentError("authorized experiments need both explicit permissions and a named approver")
    return document


def authorize_plan(plan: dict, *, allow_model_spend: bool, allow_gpu: bool, approved_by: str) -> dict:
    """Return a new plan; never mutate the dry-run artifact in place."""
    read_plan(plan)
    if plan["status"] != "dry_run" or not allow_model_spend or not allow_gpu or approved_by.strip() in {"", "cli"}:
        raise ExperimentError("execution requires a dry-run plan, --allow-spend, --allow-gpu and a named --by")
    out = json.loads(json.dumps(plan))
    out["authorization"] = {"allow_model_spend": True, "allow_gpu": True, "approved_by": approved_by.strip()}
    out["status"] = "authorized"
    out["experiment_id"] = digest({key: value for key, value in out.items() if key != "experiment_id"})[:24]
    return out


CELL_KEYS = ("seed", "repetition", "pair_index", "order", "arm", "run_digest")


def _cell_key(cell: dict) -> tuple[int, int, int]:
    """@brief 配对身份只有 ``pair_index + seed + repetition``，数组位置不是身份。"""
    return (int(cell["pair_index"]), int(cell["seed"]), int(cell["repetition"]))


def _cell_record(value, name: str) -> dict:
    """@brief 校验一条 cell 记录的形状；形状不对就是显式错误，不是可跳过的样本。"""
    if not isinstance(value, dict) or set(value) != set(CELL_KEYS):
        raise ExperimentError(f"{name} needs one cell record per sample with fields {', '.join(CELL_KEYS)}")
    for key in ("seed", "repetition", "pair_index", "order"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ExperimentError(f"{name} cell {key} must be a non-negative integer")
    if value["seed"] >= 2**64 or value["repetition"] > 100 or value["pair_index"] > 64:
        raise ExperimentError(f"{name} cell identity is out of range")
    if value["arm"] not in {"control", "treatment"}:
        raise ExperimentError(f"{name} cell names an unknown arm")
    return {**value, "run_digest": _digest(value["run_digest"], f"{name}.cell.run_digest")}


def _samples_and_cells(value, name: str, *, arm: str, plan: dict) -> tuple[list[float], list[dict]]:
    """@brief 取出一份测量的样本及其 cell 身份，并拒绝缺失、重复、错 arm、错 digest。

    每个样本都必须带自己的 cell 身份；没有身份的样本不能被当成“按位置对齐”的证据。
    计划里不存在的身份、同一个身份出现两次、arm 写错或 digest 不在本臂声明里，都是显式错误。
    """
    if not isinstance(value, dict) or not isinstance(value.get("samples"), list) or not value["samples"]:
        raise ExperimentError(f"{name} needs at least one sample")
    samples = value["samples"]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in samples):
        raise ExperimentError(f"{name} samples must be finite numbers")
    raw_cells = value.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != len(samples):
        raise ExperimentError(f"{name} needs one cell record per sample")
    declared = {_cell_key(cell) for cell in _plan_cells(plan["seeds"], plan["repetitions"])}
    digests = {_digest(item, f"{name}.run_digest") for item in (value.get("run_digests") or [])}
    cells, seen = [], set()
    for raw in raw_cells:
        cell = _cell_record(raw, name)
        key = _cell_key(cell)
        if cell["arm"] != arm:
            raise ExperimentError(f"{name} cell belongs to {cell['arm']}, not {arm}")
        if key not in declared:
            raise ExperimentError(f"{name} cell is not one of the plan's frozen pair and repetition identities")
        if key in seen:
            raise ExperimentError(f"{name} repeats the same cell")
        if cell["run_digest"] not in digests:
            raise ExperimentError(f"{name} cell digest is not declared by the arm")
        seen.add(key)
        cells.append(cell)
    return [float(x) for x in samples], cells


def _measurement(value, name: str, *, arm: str, plan: dict) -> dict:
    samples, cells = _samples_and_cells(value, name, arm=arm, plan=plan)
    return {
        "samples": samples, "unit": _text(value.get("unit"), f"{name}.unit"),
        "run_digests": sorted(_digest(item, f"{name}.run_digest") for item in value.get("run_digests") or []),
        "cells": cells,
    }


def _experiment_feedback_records(run_dir: Path) -> list[dict]:
    records = []
    path = Path(run_dir) / "feedback.jsonl"
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            records.append(row)
    return records


def _reconcile_cost(directory: Path, spent: float) -> tuple[list[dict], float]:
    """@brief 会话费用只读 transcript 明细，总额以 ledger.json 为准。

    ``feedback.jsonl`` 从来不是账单：真实 run 的 feedback 记录里没有 ``cost_usd``。
    会话美元与 token 取 ``modelcost`` 的 ``agent`` 行聚合，逐场明细取 ``modelcost.session_rows``；
    ``ledger.json`` 是含非会话扣费（review 等）的汇总权威。任一项缺失或对不上都是错误，
    不用 0 填充，也不把差额当作零。``session_rows`` 也收 ``chat`` 行，因为 ledger 会为它们计费；
    按模型的 ``model_totals`` 仍只含 ``agent`` 行，就是 ``Totals`` 的既有语义。

    @param directory 已完成的 arm run 目录。
    @param spent ``result.json`` 记录的已花费美元。
    @return ``(按模型的会话汇总, 非会话扣费美元)``。
    @raises ExperimentError transcript/ledger 缺失、某场没有有限完整费用、或与 ``spent`` 对不上时。
    """
    from .artifacts import LEDGER_SCHEMA_VERSION, NeedsInspection, read_versioned
    from .modelcost import from_path, session_rows

    transcript = directory / "transcript.jsonl"
    if not transcript.is_file():
        raise ExperimentError("an experiment needs transcript.jsonl, the billed session record")
    sessions = session_rows(transcript)
    # ``cost_incomplete=True`` is the recorded "unknown bill". An agent row with an explicit
    # ``cost_incomplete=False`` and no cost is the orchestrator's known-zero session.
    for row in sessions:
        if row["cost_incomplete"] is True:
            raise ExperimentError("a billed session has no finite, complete cost")
        if row["cost_usd"] is not None:
            continue
        if row["event"] == "agent" and row["cost_incomplete"] is False:
            row["cost_usd"] = 0.0
            continue
        raise ExperimentError("a billed session has no finite, complete cost")
    if not sessions:
        raise ExperimentError("the run has no billed session in transcript.jsonl")
    session_total = sum(float(row["cost_usd"]) for row in sessions)
    try:
        ledger = read_versioned(directory / "ledger.json", LEDGER_SCHEMA_VERSION)
    except (OSError, ValueError, NeedsInspection) as exc:
        raise ExperimentError(f"cannot read the billed ledger: {exc}") from exc
    summary_doc = ledger.get("summary") if isinstance(ledger, dict) else None
    if not isinstance(summary_doc, dict):
        raise ExperimentError("an experiment needs a complete per-session cost ledger")
    total = summary_doc.get("total_usd")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total < 0:
        raise ExperimentError("the billed ledger has no finite total")
    tolerance = max(0.01, abs(spent) * 0.01)
    if abs(float(total) - spent) > tolerance:
        raise ExperimentError("session cost ledger does not reconcile with run spend")
    if not sessions and float(total) > tolerance:
        raise ExperimentError("the run ledger bills work the transcript does not record")
    requests = ledger.get("requests")
    non_session = float(total) - session_total
    if not isinstance(requests, list) or not requests:
        if non_session < -tolerance:
            raise ExperimentError("the transcript bills more than the run ledger")
        raise ExperimentError("the billed ledger has no request records")
    if isinstance(requests, list) and requests:
        request_total, unbilled = 0.0, list(sessions)
        for row in requests:
            value = row.get("cost_usd") if isinstance(row, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ExperimentError("the billed ledger has a request without a finite cost")
            request_total += float(value)
            label = str(row.get("label") or "")
            # ``CostLedger.charge`` uses the session tag; non-session charges (harness reviews)
            # are the only rows allowed to have no transcript entry.
            if label.startswith("review:"):
                continue
            match = next((item for item in unbilled if item["label"] == label and abs(float(item["cost_usd"]) - float(value)) <= tolerance), None)
            if match is None:
                raise ExperimentError("the ledger bills a session the transcript does not record")
            unbilled.remove(match)
        if abs(request_total - float(total)) > tolerance:
            raise ExperimentError("billed ledger requests do not add up to its summary")
        if unbilled:
            raise ExperimentError("the transcript records a session the ledger does not bill")
        non_session = sum(float(row["cost_usd"]) for row in requests if str(row.get("label") or "").startswith("review:"))
    if non_session < -tolerance:
        raise ExperimentError("the transcript bills more than the run ledger")
    return from_path(transcript), session_total, max(0.0, non_session)


def _resolve_cell(plan: dict, seed, arm: str, *, pair_index, repetition, order, run_digest: str) -> dict:
    """@brief 用计划冻结的 seeds 定位本次 run 的 cell 身份，而不是按调用顺序编号。

    ``repetitions=1`` 时 seed 已足够唯一；``repetitions>1`` 时同一个 seed 对应多个 cell，
    调用方必须给出 ``pair_index`` 与 ``repetition``，否则无法证明是哪一次重复。
    """
    declared = _plan_cells(plan["seeds"], plan["repetitions"])
    for value, name in ((pair_index, "pair_index"), (repetition, "repetition"), (order, "order")):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ExperimentError(f"{name} must be a non-negative integer")
    matches = [cell for cell in declared if cell["seed"] == seed]
    if pair_index is None or repetition is None:
        if len(matches) != 1:
            raise ExperimentError("this run seed is repeated in the plan; collecting it needs pair_index and repetition")
        chosen = matches[0]
        if pair_index is not None and pair_index != chosen["pair_index"]:
            raise ExperimentError("pair_index does not match the plan's seed order")
        if repetition is not None and repetition != chosen["repetition"]:
            raise ExperimentError("repetition does not match the plan's seed order")
    else:
        chosen = next((cell for cell in matches if cell["pair_index"] == pair_index and cell["repetition"] == repetition), None)
        if chosen is None:
            raise ExperimentError("this run is not one of the plan's frozen seed and repetition cells")
    return {"seed": chosen["seed"], "repetition": chosen["repetition"], "pair_index": chosen["pair_index"], "order": chosen["order"] if order is None else order, "arm": arm, "run_digest": run_digest}


def collect_arm(plan: dict, run_dir: Path, arm: str, *, pair_index: int | None = None, repetition: int | None = None, order: int | None = None) -> dict:
    """Extract one arm's metrics from host records; no model call and no GPU.

    ``pair_index`` / ``repetition`` identify which planned cell this run is when the plan
    repeats a seed. They default to the plan's only cell for the run's seed, so a
    ``repetitions=1`` plan still collects from a bare run directory.
    """
    if arm not in {"control", "treatment"}:
        raise ExperimentError("arm must be control or treatment")
    read_plan(plan)
    if plan["status"] != "authorized":
        raise ExperimentError("collecting an experiment arm requires explicit authorization")
    directory = Path(run_dir).resolve()
    from taic.artifacts import NeedsInspection, RESULT_SCHEMA_VERSION, RUN_CONFIG_SCHEMA_VERSION, read_versioned
    try:
        run_cfg = read_versioned(directory / "run_config.json", RUN_CONFIG_SCHEMA_VERSION)
        result = read_versioned(directory / "result.json", RESULT_SCHEMA_VERSION)
    except (OSError, ValueError, NeedsInspection) as exc:
        raise ExperimentError(f"cannot read completed arm records: {exc}") from exc
    if not isinstance(run_cfg, dict) or not isinstance(result, dict):
        raise ExperimentError("arm config and result must be objects")
    scope = plan["scope"]
    target = run_cfg.get("target") if isinstance(run_cfg.get("target"), dict) else {}
    if str(run_cfg.get("task") or result.get("task") or "") != scope["task"] or str(run_cfg.get("dtype") or result.get("dtype") or "") != scope["dtype"] or str(run_cfg.get("target_id") or target.get("id") or result.get("target_id") or "") != scope["target_id"]:
        raise ExperimentError("arm scope differs from the experiment")
    comparison = config_template(run_cfg, budget_usd=float((run_cfg.get("search") or {}).get("budget_usd") or 0), base_commit=str(run_cfg.get("base_commit") or run_cfg.get("git_commit") or plan["base_commit"]))
    config_digest = digest(comparison)
    if config_digest != plan["config_sha256"]:
        raise ExperimentError("arm config differs from the experiment")
    binding = run_cfg.get("harness_experiment")
    expected_binding = {"experiment_id": plan["experiment_id"], "arm": arm, "change_sha256": plan[arm]["change_sha256"] or ""}
    if not isinstance(binding, dict) or any(binding.get(key) != value for key, value in expected_binding.items()):
        raise ExperimentError("run_config does not bind the experiment's change and arm")
    if run_cfg.get("run_seed") not in plan["seeds"] or result.get("run_seed") != run_cfg["run_seed"]:
        raise ExperimentError("arm seed is not one of the paired experiment seeds")
    if run_cfg.get("search", {}).get("harness_experiment_id") != plan["experiment_id"] or run_cfg.get("search", {}).get("harness_experiment_arm") != arm or run_cfg.get("search", {}).get("harness_change_sha256", "") != (plan[arm]["change_sha256"] or ""):
        raise ExperimentError("raw search settings do not bind this experiment and arm")
    if str(run_cfg.get("protocol_version") or result.get("protocol_version") or "") != plan["protocol_version"]:
        raise ExperimentError("arm protocol differs from the experiment")
    spent = result.get("spent_usd")
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not math.isfinite(spent) or spent < 0:
        raise ExperimentError("arm has no finite spend record")
    records = _experiment_feedback_records(directory)
    trace = result.get("trace")
    model_totals, session_cost, non_session_usd = _reconcile_cost(directory, spent)
    if not isinstance(trace, list):
        raise ExperimentError("arm has no candidate trace")
    eligible_trace = [row for row in trace if isinstance(row, dict) and row.get("status") == "ok"]
    if not eligible_trace:
        raise ExperimentError("the arm has no first eligible candidate, so first-eligible metrics are undefined")
    first_trace = min(eligible_trace, key=lambda row: int(row.get("idx", 2**63)))
    for field in ("cost_usd", "t_s"):
        if isinstance(first_trace.get(field), bool) or not isinstance(first_trace.get(field), (int, float)) or not math.isfinite(first_trace[field]) or first_trace[field] < 0:
            raise ExperimentError(f"the first eligible trace has no finite {field}")
    if float(first_trace["cost_usd"]) > spent + max(0.01, abs(spent) * 0.01):
        raise ExperimentError("the first eligible trace costs more than the whole run")
    run_digest = digest({"run_config": run_cfg, "result": result})
    if result.get("harness_experiment_id") != plan["experiment_id"] or result.get("harness_experiment_arm") != arm:
        raise ExperimentError("arm does not bind this experiment and treatment role")
    recorded_base = run_cfg.get("base_commit") or run_cfg.get("git_commit")
    if plan["base_commit"] and recorded_base != plan["base_commit"]:
        raise ExperimentError("arm source tree differs from the experiment base commit")
    if plan["base_commit"] and run_cfg.get("search", {}).get("harness_base_commit") != plan["base_commit"]:
        raise ExperimentError("run search config does not bind the base commit")
    recorded_change = result.get("harness_change_sha256")
    if (recorded_change or "") != (plan[arm]["change_sha256"] or ""):
        raise ExperimentError(f"{arm} result does not bind the expected harness change")
    # The real run writes "" when there is no change (``taic/llm.py`` SearchConfig default),
    # while the plan spells "no change" as ``None``. Return the plan's spelling after the
    # recorded value has been verified, so a real control arm can reach ``summarize_result``.
    change_digest = plan[arm]["change_sha256"]
    cell = _resolve_cell(plan, run_cfg.get("run_seed"), arm, pair_index=pair_index, repetition=repetition, order=order, run_digest=run_digest)

    def measured(samples, unit: str) -> dict:
        return {"samples": [float(x) for x in samples], "unit": unit, "run_digests": [run_digest], "cells": [dict(cell)]}

    metrics = {
        "first_eligible_cost_usd": measured([first_trace["cost_usd"]], "usd"),
        "first_eligible_elapsed_s": measured([first_trace["t_s"]], "s"),
        "invalid_feedback_rate": measured([sum(1 for row in records if row.get("outcome") == "agent_failed") / len(records) if records else 1.0], "ratio"),
    }
    tools = []
    evidence = directory / "tool_evidence"
    if evidence.is_dir():
        for path in evidence.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                tools.append(row["result"]["status"] == "ok")
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                tools.append(False)
    if tools:
        metrics["tool_success_rate"] = measured([sum(tools) / len(tools)], "ratio")
    if isinstance(result.get("confirmed_speedup"), (int, float)) and not isinstance(result.get("confirmed_speedup"), bool):
        metrics["confirmed_speedup"] = measured([result["confirmed_speedup"]], "ratio")
    guards = {}
    for name in ("safety_regressions", "correctness_regressions", "agent_failures"):
        if name == "agent_failures":
            count = sum(1 for row in records if row.get("outcome") == "agent_failed")
        else:
            count = sum(1 for row in trace if isinstance(row, dict) and ((name == "safety_regressions" and row.get("status") in {"unsafe", "crash"}) or (name == "correctness_regressions" and row.get("status") == "incorrect")))
        guards[name] = measured([count], "count")
    review_cost = 0.0
    for row in _read_jsonl(directory / "harness_reviews.jsonl"):
        if not isinstance(row.get("cost_usd"), (int, float)) or isinstance(row.get("cost_usd"), bool) or not math.isfinite(row["cost_usd"]) or row["cost_usd"] < 0:
            raise ExperimentError("a harness review has no finite bill")
        review_cost += float(row["cost_usd"])
    guards["additional_review_cost_usd"] = measured([review_cost], "usd")
    return {"config_sha256": config_digest, "protocol_version": plan["protocol_version"], "seeds": plan["seeds"], "repetitions": plan["repetitions"], "change_sha256": change_digest, "spent_usd": spent, "session_cost_usd": round(session_cost, 6), "non_session_cost_usd": round(non_session_usd, 6), "model_totals": model_totals, "run_digests": [run_digest], **metrics, **guards}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def paired_effect(document: dict) -> dict:
    """Apply frozen effect rules over cells joined by pair identity, never by array position.

    A sample is identified by ``pair_index + seed + repetition``. Repetitions of the same
    pair are averaged into one independent pair, so a legal repetition never inflates the
    pair count. Missing cells are surfaced and limited by ``max_missing_fraction``;
    duplicate cells, a wrong arm, an undeclared identity or an undeclared digest raise
    instead of silently shrinking the sample.
    """
    plan = read_plan(document.get("plan"))
    metric = plan["primary_metric"]
    left = document["control"].get(metric)
    right = document["treatment"].get(metric)
    if left is None or right is None:
        return {"status": "inconclusive", "reason": "primary metric is missing from one arm", "pairs": 0}
    rules = plan["evaluation_rules"]
    ratio_metric = metric in {"first_eligible_cost_usd", "first_eligible_elapsed_s"}
    left_samples, left_cells = _samples_and_cells(left, f"control.{metric}", arm="control", plan=plan)
    right_samples, right_cells = _samples_and_cells(right, f"treatment.{metric}", arm="treatment", plan=plan)
    control_by_cell = {_cell_key(cell): sample for cell, sample in zip(left_cells, left_samples)}
    treatment_by_cell = {_cell_key(cell): sample for cell, sample in zip(right_cells, right_samples)}
    declared = _plan_cells(plan["seeds"], plan["repetitions"])
    pair_count = len({cell["pair_index"] for cell in declared})
    by_pair: dict[int, list[float]] = {}
    missing = 0
    for cell in declared:
        key = _cell_key(cell)
        if key not in control_by_cell or key not in treatment_by_cell:
            missing += 1
            continue
        control, treatment = control_by_cell[key], treatment_by_cell[key]
        if ratio_metric:
            if control <= 0 or treatment <= 0:
                missing += 1
                continue
            value = math.log(treatment / control)
        else:
            value = treatment - control
        if not math.isfinite(value):
            missing += 1
            continue
        by_pair.setdefault(cell["pair_index"], []).append(value)
    # One value per pair: repetitions are repeated measurements, not new independent pairs.
    pairs = [statistics.mean(values) for _, values in sorted(by_pair.items()) if len(values) == plan["repetitions"]]
    result = {
        "metric": metric, "transform": "log_ratio" if ratio_metric else "difference", "pairs": len(pairs),
        "pair_count": pair_count, "repetitions": plan["repetitions"], "missing_cells": missing,
        "min_pairs": rules["min_pairs"], "direction": rules["direction"],
    }
    if len(pairs) < rules["min_pairs"]:
        return {**result, "status": "inconclusive", "reason": "not enough complete pairs", "point_estimate": statistics.mean(pairs) if pairs else None}
    missing_fraction = 1 - len(pairs) / pair_count
    if missing_fraction > rules["max_missing_fraction"]:
        return {**result, "status": "inconclusive", "reason": "too many missing or invalid pairs"}
    rng = random.Random(int(hashlib.sha256(plan["experiment_id"].encode()).hexdigest()[:16], 16))
    means = []
    for _ in range(rules["bootstrap_samples"]):
        means.append(statistics.mean(rng.choice(pairs) for _ in pairs))
    means.sort()
    lo, hi = means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]
    threshold = -rules["min_effect"] if rules["direction"] == "lower" else rules["min_effect"]
    # In the beneficial direction, the whole interval must clear the frozen threshold.
    meets_effect = hi < -rules["min_effect"] if rules["direction"] == "lower" else lo >= rules["min_effect"]
    guards = []
    for name, limit in rules["guardrail_limits"].items():
        if document["treatment"].get(name) is None or document["control"].get(name) is None:
            guards.append(name)
            continue
        treatment_samples, treatment_cells = _samples_and_cells(document["treatment"][name], f"treatment.{name}", arm="treatment", plan=plan)
        control_samples, control_cells = _samples_and_cells(document["control"][name], f"control.{name}", arm="control", plan=plan)
        # The equal-length guard stays; identity coverage below is strictly stronger.
        if len(treatment_samples) != len(control_samples):
            guards.append(name)
            continue
        treatment_by_cell = {_cell_key(cell): sample for cell, sample in zip(treatment_cells, treatment_samples)}
        control_by_cell = {_cell_key(cell): sample for cell, sample in zip(control_cells, control_samples)}
        violated = False
        for cell in declared:
            key = _cell_key(cell)
            if key not in treatment_by_cell or key not in control_by_cell:
                violated = True
                break
            if treatment_by_cell[key] > limit or treatment_by_cell[key] > control_by_cell[key]:
                violated = True
                break
        if violated:
            guards.append(name)
    point = statistics.mean(pairs)
    return {**result, "status": "supports_effect" if meets_effect and not guards else "inconclusive", "reason": "guardrail violation" if guards else "paired interval evaluated against the frozen threshold", "point_estimate": point, "ci95": [lo, hi], "threshold": threshold, "guardrail_violations": guards}


def summarize_result(document: dict) -> dict:
    """Compare arms descriptively; this never turns a small experiment into a promotion."""
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ExperimentError("unsupported experiment result")
    plan = read_plan(document.get("plan"))
    if set(document) != {"schema_version", "plan", "control", "treatment"}:
        raise ExperimentError("experiment result has unknown or missing fields")
    if plan["status"] != "authorized":
        raise ExperimentError("results require an explicitly authorized plan")
    for key in ("control", "treatment"):
        if not isinstance(document.get(key), dict):
            raise ExperimentError(f"missing {key} measurements")
    results = {}
    for metric in METRICS:
        if metric in document["control"] or metric in document["treatment"]:
            if metric not in document["control"] or metric not in document["treatment"]:
                raise ExperimentError(f"{metric} must be measured in both arms")
            results[metric] = {arm: _measurement(document[arm][metric], f"{arm}.{metric}", arm=arm, plan=plan) for arm in ("control", "treatment")}
    if not results:
        raise ExperimentError("at least one paired metric is required")
    if plan["effect_metric_id"] not in results:
        raise ExperimentError("the acceptance contract's effect_metric_id must be measured in both arms")
    if any(not isinstance(document[arm].get("config_sha256"), str) or document[arm]["config_sha256"] != plan["config_sha256"] or document[arm].get("protocol_version") != plan["protocol_version"] or not isinstance(document[arm].get("seeds"), list) or list(document[arm]["seeds"]) != list(plan["seeds"]) or document[arm].get("repetitions") != plan["repetitions"] or document[arm].get("change_sha256") != plan[arm]["change_sha256"] for arm in ("control", "treatment")):
        raise ExperimentError("an arm differs from the authorized config, protocol, seeds, repetitions or change digest")
    for arm in ("control", "treatment"):
        spent = document[arm].get("spent_usd")
        if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not math.isfinite(spent) or spent < 0 or spent > plan["budget_per_arm_usd"] * len(plan["seeds"]):
            raise ExperimentError(f"{arm}.spent_usd exceeds its authorized budget")
    guards = {}
    for metric in GUARDRAILS:
        if metric in document["control"] or metric in document["treatment"]:
            if metric not in document["control"] or metric not in document["treatment"]:
                raise ExperimentError(f"guardrail {metric} must be measured in both arms")
            guards[metric] = {arm: _measurement(document[arm][metric], f"{arm}.{metric}", arm=arm, plan=plan) for arm in ("control", "treatment")}
    required_guards = {name for name in ("safety_regressions", "correctness_regressions") if any(name in text for text in plan["guardrail_metrics"])}
    if required_guards - set(guards):
        raise ExperimentError("declared safety/correctness guardrails are missing")
    effect = paired_effect(document)
    return {
        "schema_version": 1, "event": "experiment_result", "experiment_id": plan["experiment_id"],
        "effect": effect,
        "proposal_key": plan["proposal_key"], "spec_sha256": plan["spec_sha256"], "change_sha256": plan["change_sha256"],
        "config_sha256": plan["config_sha256"], "protocol_version": plan["protocol_version"],
        "metrics": results, "guardrails": guards,
        "effect_validated": False,
        "conclusion": "Measurements are recorded; effect validation and promotion require separate human review.",
    }


def latest_effect_validations(decisions: list[dict], proposal_key: str) -> dict[str, dict]:
    """@brief 每个实验只保留最新 verdict，避免旧的 true 覆盖后续 false。"""
    latest = {}
    for row in decisions:
        if not isinstance(row, dict) or row.get("event") != "effect_validation":
            continue
        if str(row.get("proposal_key") or "") != proposal_key:
            continue
        experiment_id = str(row.get("experiment_id") or "")
        if experiment_id:
            latest[experiment_id] = row
    return latest


def milestones(decisions: list[dict], proposal_key: str) -> dict:
    """Keep review, implementation, validation, local application and commit distinct."""
    rows = [row for row in decisions if isinstance(row, dict) and str(row.get("proposal_key") or "") == proposal_key]
    useful = any(row.get("event") == "review" and row.get("useful") is True and row.get("cost_usd") is not None for row in rows)
    implemented = any(row.get("event") == "implementation" and row.get("accepted") is True and row.get("host_verified") is True for row in rows)
    validated = any(row.get("validated") is True and row.get("approved_by") for row in latest_effect_validations(rows, proposal_key).values())
    local = any(row.get("event") in {"promotion", "code_promotion"} and row.get("status") == "local" for row in rows)
    committed = any(row.get("event") == "promotion_commit" for row in rows)
    return {"reviewed": useful, "implemented": implemented, "experiment_validated": validated, "local": local, "committed": committed}


def record_effect_validation(run_dir: Path, result: dict, *, validated: bool, approved_by: str, reason: str, evidence_path: Path | None = None) -> dict:
    """A named person records the effect verdict; measurement code cannot self-approve."""
    summary = summarize_result(result)
    if approved_by.strip() in {"", "cli"} or not reason.strip():
        raise ExperimentError("effect review needs a named --by and --reason")
    if validated and summary["effect"]["status"] != "supports_effect":
        raise ExperimentError("cannot validate an effect without passing the frozen sample, effect and guardrail rules")
    if validated and result["plan"]["authorization"].get("approved_by") == approved_by.strip():
        raise ExperimentError("the execution approver cannot be the sole effect reviewer")
    spent_values = []
    for arm in ("control", "treatment"):
        value = result.get(arm, {}).get("spent_usd")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            spent_values = []
            break
        spent_values.append(float(value))
    row = {**summary, "event": "effect_validation", "validated": bool(validated), "approved_by": approved_by.strip(), "reason": reason.strip()[:2000]}
    row["result_sha256"] = digest(result)
    if spent_values:
        row["spent_usd"] = sum(spent_values)
    if evidence_path is not None:
        row["evidence_path"] = str(evidence_path)
    path = Path(run_dir) / "harness_decisions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row
