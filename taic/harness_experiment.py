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


def make_plan(spec: dict, *, change_sha256: str, protocol_version: str, config_sha256: str, seeds: list[int], budget_usd: float, repetitions: int = 1, config_template_document: dict | None = None, rules: dict | None = None, base_commit: str = "") -> dict:
    """Bind one proposed treatment to the exact implementation and shared configuration."""
    spec = read_spec(spec)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1 or repetitions > 100:
        raise ExperimentError("repetitions must be between 1 and 100")
    if not isinstance(seeds, list) or not seeds or len(seeds) > 64 or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 2**64 for seed in seeds):
        raise ExperimentError("seeds must be a non-empty list of uint64 values")
    if len(set(seeds)) != len(seeds) and repetitions == 1:
        raise ExperimentError("paired seeds must be distinct unless repetitions intentionally repeats them")
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
        "pair_count": len(seeds) // repetitions,
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
        upgraded["pair_count"] = len(upgraded["seeds"])
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
        if document["primary_metric"] != document["effect_metric_id"] or document["pair_count"] != len(document["seeds"]):
            raise ExperimentError("primary metric or pair count is inconsistent")
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
        if not isinstance(seeds, list) or not seeds or len(seeds) > 64 or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 2**64 for seed in seeds):
            raise ExperimentError("paired seeds are invalid")
        if not isinstance(document["repetitions"], int) or isinstance(document["repetitions"], bool) or not 1 <= document["repetitions"] <= 100:
            raise ExperimentError("repetitions must be between 1 and 100")
        if len(document["seeds"]) != document["pair_count"] * document["repetitions"]:
            raise ExperimentError("seeds must provide one entry per pair and repetition")
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


def _measurement(value, name: str) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("samples"), list) or not value["samples"]:
        raise ExperimentError(f"{name} needs at least one sample")
    samples = value["samples"]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in samples):
        raise ExperimentError(f"{name} samples must be finite numbers")
    return {"samples": [float(x) for x in samples], "unit": _text(value.get("unit"), f"{name}.unit"), "run_digests": sorted(_digest(item, f"{name}.run_digest") for item in value.get("run_digests") or [])}


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


def collect_arm(plan: dict, run_dir: Path, arm: str) -> dict:
    """Extract one arm's metrics from host records; no model call and no GPU."""
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
    if not records and (spent > 0 or trace):
        raise ExperimentError("an experiment needs a complete per-session cost ledger")
    if records:
        cumulative = 0.0
        for row in records:
            value = row.get("cost_usd")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ExperimentError("an experiment needs a complete per-session cost ledger")
            cumulative += float(value)
        if abs(cumulative - spent) > max(0.01, abs(spent) * 0.01):
            raise ExperimentError("session cost ledger does not reconcile with run spend")
    if not isinstance(trace, list):
        raise ExperimentError("arm has no candidate trace")
    eligible_trace = [row for row in trace if isinstance(row, dict) and row.get("status") == "ok"]
    if not eligible_trace:
        raise ExperimentError("the arm has no first eligible candidate, so first-eligible metrics are undefined")
    first_trace = min(eligible_trace, key=lambda row: int(row.get("idx", 2**63)))
    for field in ("cost_usd", "t_s"):
        if isinstance(first_trace.get(field), bool) or not isinstance(first_trace.get(field), (int, float)) or not math.isfinite(first_trace[field]) or first_trace[field] < 0:
            raise ExperimentError(f"the first eligible trace has no finite {field}")
    spent = result.get("spent_usd")
    if isinstance(spent, bool) or not isinstance(spent, (int, float)) or not math.isfinite(spent) or spent < 0:
        raise ExperimentError("arm has no finite spend record")
    run_digest = digest({"run_config": run_cfg, "result": result})
    if result.get("harness_experiment_id") != plan["experiment_id"] or result.get("harness_experiment_arm") != arm:
        raise ExperimentError("arm does not bind this experiment and treatment role")
    recorded_base = run_cfg.get("base_commit") or run_cfg.get("git_commit")
    if plan["base_commit"] and recorded_base != plan["base_commit"]:
        raise ExperimentError("arm source tree differs from the experiment base commit")
    if plan["base_commit"] and run_cfg.get("search", {}).get("harness_base_commit") != plan["base_commit"]:
        raise ExperimentError("run search config does not bind the base commit")
    change_digest = result.get("harness_change_sha256")
    if (change_digest or "") != (plan[arm]["change_sha256"] or ""):
        raise ExperimentError(f"{arm} result does not bind the expected harness change")
    metrics = {
        "first_eligible_cost_usd": {"samples": [float(first_trace["cost_usd"])], "unit": "usd", "run_digests": [run_digest]},
        "first_eligible_elapsed_s": {"samples": [float(first_trace["t_s"])], "unit": "s", "run_digests": [run_digest]},
        "invalid_feedback_rate": {"samples": [sum(1 for row in records if row.get("outcome") == "agent_failed") / len(records) if records else 1.0], "unit": "ratio", "run_digests": [run_digest]},
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
        metrics["tool_success_rate"] = {"samples": [sum(tools) / len(tools)], "unit": "ratio", "run_digests": [run_digest]}
    if isinstance(result.get("confirmed_speedup"), (int, float)) and not isinstance(result.get("confirmed_speedup"), bool):
        metrics["confirmed_speedup"] = {"samples": [result["confirmed_speedup"]], "unit": "ratio", "run_digests": [run_digest]}
    guards = {}
    for name in ("safety_regressions", "correctness_regressions", "agent_failures"):
        if name == "agent_failures":
            count = sum(1 for row in records if row.get("outcome") == "agent_failed")
        else:
            count = sum(1 for row in trace if isinstance(row, dict) and ((name == "safety_regressions" and row.get("status") in {"unsafe", "crash"}) or (name == "correctness_regressions" and row.get("status") == "incorrect")))
        guards[name] = {"samples": [float(count)], "unit": "count", "run_digests": [run_digest]}
    review_cost = 0.0
    for row in _read_jsonl(directory / "harness_reviews.jsonl"):
        if not isinstance(row.get("cost_usd"), (int, float)) or isinstance(row.get("cost_usd"), bool) or not math.isfinite(row["cost_usd"]) or row["cost_usd"] < 0:
            raise ExperimentError("a harness review has no finite bill")
        review_cost += float(row["cost_usd"])
    guards["additional_review_cost_usd"] = {"samples": [review_cost], "unit": "usd", "run_digests": [run_digest]}
    return {"config_sha256": config_digest, "protocol_version": plan["protocol_version"], "seeds": plan["seeds"], "repetitions": plan["repetitions"], "change_sha256": change_digest, "spent_usd": spent, "run_digests": [run_digest], **metrics, **guards}


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
    """Apply frozen effect rules; a small sample never becomes a validation pass."""
    plan = read_plan(document.get("plan"))
    metric = plan["primary_metric"]
    left = document["control"].get(metric)
    right = document["treatment"].get(metric)
    if left is None or right is None:
        return {"status": "inconclusive", "reason": "primary metric is missing from one arm", "pairs": 0}
    rules = plan["evaluation_rules"]
    ratio_metric = metric in {"first_eligible_cost_usd", "first_eligible_elapsed_s"}
    pairs = []
    for control, treatment in zip(left["samples"], right["samples"]):
        if ratio_metric:
            if control <= 0 or treatment <= 0:
                continue
            value = math.log(treatment / control)
        else:
            value = treatment - control
        if math.isfinite(value):
            pairs.append(value)
    result = {"metric": metric, "transform": "log_ratio" if ratio_metric else "difference", "pairs": len(pairs), "min_pairs": rules["min_pairs"], "direction": rules["direction"]}
    if len(pairs) < rules["min_pairs"]:
        return {**result, "status": "inconclusive", "reason": "not enough complete pairs", "point_estimate": statistics.mean(pairs) if pairs else None}
    if len(pairs) != len(left["samples"]) or len(pairs) != len(right["samples"]):
        missing = 1 - len(pairs) / max(len(left["samples"]), len(right["samples"]))
        if missing > rules["max_missing_fraction"]:
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
        treatment_samples = document["treatment"].get(name, {}).get("samples")
        control_samples = document["control"].get(name, {}).get("samples")
        if treatment_samples is None or control_samples is None or len(treatment_samples) != len(control_samples):
            guards.append(name)
            continue
        if any(treatment > limit or treatment > control for treatment, control in zip(treatment_samples, control_samples)):
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
            results[metric] = {arm: _measurement(document[arm][metric], f"{arm}.{metric}") for arm in ("control", "treatment")}
    if not results:
        raise ExperimentError("at least one paired metric is required")
    if plan["effect_metric_id"] not in results:
        raise ExperimentError("the acceptance contract's effect_metric_id must be measured in both arms")
    if any(not isinstance(document[arm].get("config_sha256"), str) or document[arm]["config_sha256"] != plan["config_sha256"] or document[arm].get("protocol_version") != plan["protocol_version"] or not isinstance(document[arm].get("seeds"), list) or set(document[arm]["seeds"]) != set(plan["seeds"]) or document[arm].get("repetitions") != plan["repetitions"] or document[arm].get("change_sha256") != plan[arm]["change_sha256"] for arm in ("control", "treatment")):
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
            guards[metric] = {arm: _measurement(document[arm][metric], f"{arm}.{metric}") for arm in ("control", "treatment")}
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
