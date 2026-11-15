"""P2-U paid-pair recomputation copied from the closed ``P2-U/verify.py``.

The original verifier is closed.  Fix copied-function bugs here and record the
old verifier's drift in a new epoch.  This reads archived runs; it never starts
a search, a model call or a GPU benchmark.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]


EXPERIMENT = ROOT / "runs/p2u-experience-overlay-paid-20261114"


INTERRUPTED = ROOT / "runs/p2u-experience-overlay-paid-20261114-interrupted"


EXPERIENCE = ROOT / "targets/apple-m4-max/experience.md"


OVERLAY = ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2s/evidence/experience-overlay.md"


FILTERED_EXPERIENCE_SHA = "e972450e333550739378afe9362e17470bc9f59fba089a849835e28402a8fea2"


RENDERED_EXPERIENCE_SHA = "8e5adbecd3f764904ceab72c2e35fc7534d08931fde0a24210292cdf23513304"


INIT_CANDIDATE_SHA = "1a3056224676f7fbc3208ba98349781bc0974d3465a68338d9bd0c3a18833cb4"


CELLS = (
    "control/gemm_silu_small_rep1",
    "treatment/gemm_silu_small_rep1",
    "treatment/gemm_silu_small_rep2",
    "control/gemm_silu_small_rep2",
)


SESSIONS = ("init", "r1.p0", "r2.p0", "r3.p0", "r4.p0")


PAIRS = (
    ("control/gemm_silu_small_rep1", "treatment/gemm_silu_small_rep1", 1),
    ("control/gemm_silu_small_rep2", "treatment/gemm_silu_small_rep2", 2),
)


SWITCHES = ("init_samples", "task_ledger", "plateau_after", "sweep_tool")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), f"{path} is not a JSON object")
    return document


def _money(number: float) -> str:
    return str(number)


def _session_archives(cell_dir: Path, mark: str) -> list[dict]:
    scratch = cell_dir / "scratch"
    found = sorted(path.name[: -len(".tar.gz")] for path in scratch.glob("*.tar.gz"))
    require(tuple(found) == SESSIONS, f"{cell_dir.name} session archives are {found}")
    rows = []
    for label in SESSIONS:
        with tarfile.open(scratch / f"{label}.tar.gz", "r:gz") as archive:
            experience_member = next(
                (info for info in archive.getmembers() if info.name == "experience.md" or info.name.endswith("/experience.md")),
                None,
            )
            require(experience_member is not None, f"{cell_dir.name}/{label} has no archived experience.md")
            payload = archive.extractfile(experience_member).read()
            text = payload.decode("utf-8")
            overlay_member = next(
                (info for info in archive.getmembers() if info.name == "experience_overlay.json" or info.name.endswith("/experience_overlay.json")),
                None,
            )
            overlay = None
            if overlay_member is not None:
                overlay = json.loads(archive.extractfile(overlay_member).read().decode("utf-8"))
            rows.append({
                "session": label,
                "experience_bytes": len(payload),
                "experience_sha256": hashlib.sha256(payload).hexdigest(),
                "overlay_copies": text.count(mark),
                "experience_overlay": overlay,
            })
    return rows


def _crossing(requests: list[dict], soft_budget: float) -> dict | None:
    previous = 0.0
    for request in requests:
        total = float(request["total_usd"])
        if previous <= soft_budget + 1e-9 and total > soft_budget + 1e-9:
            return {
                "label": request.get("label"),
                "previous_total_usd": previous,
                "request_cost_usd": float(request["cost_usd"]),
                "total_usd": total,
            }
        previous = total
    return None


def _search_seeds(result: dict) -> list[int]:
    seeds = []
    for row in result.get("trace") or []:
        require(isinstance(row, dict), "a trace row is not an object")
        if row.get("eval_seed_from") != "search":
            continue
        seed = row.get("eval_seed")
        require(type(seed) is int, f"search eval_seed is {seed!r}")
        seeds.append(seed)
    require(seeds, "the result has no search eval_seed")
    return seeds


def build_report() -> dict:
    """Recompute the pair, bill, overlay, recovery, and cap from the run files."""
    from experiments.common import experiment_commitment, fits_cap
    from experiments.search_yield_ab import summarize_experiment
    from taic.experience_overlay import OVERLAY_MARK

    require(EXPERIMENT.is_dir(), f"missing experiment root {EXPERIMENT}")
    require(INTERRUPTED.is_dir(), f"missing interrupted archive {INTERRUPTED}")
    require(not (EXPERIMENT / "recovery-failed-1").exists(), "the failed recovery is inside the billed root")
    result_paths = sorted(path.relative_to(EXPERIMENT).as_posix() for path in EXPERIMENT.rglob("result.json"))
    require(
        result_paths == sorted(f"{cell}/result.json" for cell in CELLS),
        f"official result.json set is {result_paths}",
    )

    plan = load(EXPERIMENT / "plan.json")
    before = load(EXPERIMENT / "plan.before-cap-increase.json")
    approval = load(EXPERIMENT / "approval.json")
    cap_approval = load(EXPERIMENT / "approval-cap-increase.json")
    cap = load(EXPERIMENT / "cap.json")
    summary_on_disk = load(EXPERIMENT / "summary.json")
    summary = summarize_experiment(EXPERIMENT, delta=0.01, allowed_regression=0.0)
    require(summary == summary_on_disk, "summary.json does not match a fresh summarize_experiment")
    require(summary["verdict"] == "no_gain", f"summarizer verdict is {summary['verdict']}")
    require(summary["complete_pairs"] == 2 and summary["expected_pairs"] == 2, "pair count drifted")
    require(summary["excluded_pairs"] == [], f"pairs were excluded: {summary['excluded_pairs']}")
    require(summary["delta"] == 0.01 and summary["allowed_regression"] == 0.0, "delta or allowed regression drifted")

    overlay_sha = sha256(OVERLAY)
    require(overlay_sha == plan["overlay_sha256"], "the frozen overlay file moved")
    global_experience = EXPERIENCE.read_text(encoding="utf-8")
    require(OVERLAY_MARK not in global_experience, "the overlay is installed in the global experience file")
    global_sha = hashlib.sha256(global_experience.encode("utf-8")).hexdigest()

    cells = []
    speeds = {}
    system_sha = ""
    for directory in CELLS:
        cell_dir = EXPERIMENT / directory
        result = load(cell_dir / "result.json")
        ledger = load(cell_dir / "ledger.json")
        config = load(cell_dir / "run_config.json")
        confirm = load(cell_dir / "confirm_report.json")
        subject = load(cell_dir / "confirmations/fp16_small_benchmark.json")
        arm = directory.split("/", 1)[0]
        seeds = _search_seeds(result)
        require(set(seeds) == {7}, f"{directory} search eval_seed values are {sorted(set(seeds))}")
        require(result.get("run_seed") == 7 and config.get("seed") == 7 and config.get("run_seed") == 7, f"{directory} run_seed is not 7")
        require(result.get("confirmed_status") == "ok", f"{directory} confirmed_status is {result.get('confirmed_status')!r}")
        source = result.get("confirmation_source_sha256")
        require(isinstance(source, str) and len(source) == 64, f"{directory} confirmation source is missing")
        require(source == sha256(cell_dir / "best.metal"), f"{directory} best.metal does not match the confirmation source")
        require(source == subject.get("candidate_sha256"), f"{directory} subject candidate does not match the confirmation source")
        require(subject.get("identity") == {"kind": "match", "differences": []}, f"{directory} identity is {subject.get('identity')}")
        require(subject.get("status") == "confirmed" and subject.get("search_seed") == 7, f"{directory} subject is not a seed-7 confirmation")
        require(subject.get("provider") == "candidate" and subject.get("credits_variant") is True, f"{directory} confirmation is not a candidate variant")
        require(subject.get("fallback_id") is None, f"{directory} confirmation fell back")
        confirmation_seed = subject.get("confirmation_seed")
        require(type(confirmation_seed) is int and confirmation_seed != 7, f"{directory} confirmation seed collided with run_seed")
        require(confirm.get("status") == "ok" and confirm.get("eligible") is True, f"{directory} confirm report is not eligible")
        require(confirm.get("speedup") == result.get("confirmed_speedup"), f"{directory} confirm report speedup differs")
        require(confirm.get("seed") == confirmation_seed, f"{directory} confirm report seed differs from the subject")
        require(result.get("confirmed_speedup") != result.get("best_speedup"), f"{directory} used best_speedup as the confirmed number")
        require(result.get("task") == "gemm_silu" and result.get("dtype") == "fp16" and result.get("size") == "small", directory)
        switches = {key: config["search"][key] for key in SWITCHES}
        require(switches == {"init_samples": 1, "task_ledger": False, "plateau_after": 0, "sweep_tool": False}, f"{directory} switches drifted")
        committee_ids = [
            {"name": item["name"], "provider": item["members"][0]["provider"], "model": item["members"][0]["model"]}
            for item in config["committees"]
        ]
        require(committee_ids == plan["committees"], f"{directory} committees differ from the frozen plan")
        prompt = (cell_dir / "system_prompt.txt").read_text(encoding="utf-8")
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        require(OVERLAY_MARK not in prompt, f"{directory} system prompt contains the overlay")
        if system_sha:
            require(prompt_sha == system_sha, "system prompt differs across cells")
        system_sha = prompt_sha
        guidance = (cell_dir / "agent_guidance/experience.md").read_text(encoding="utf-8")
        require(OVERLAY_MARK not in guidance, f"{directory} guidance snapshot contains the overlay")
        require(hashlib.sha256(guidance.encode("utf-8")).hexdigest() == global_sha, f"{directory} guidance snapshot is not the global experience file")
        archives = _session_archives(cell_dir, OVERLAY_MARK)
        spent = float(result["spent_usd"])
        require(float(ledger["summary"]["total_usd"]) == spent, f"{directory} ledger total differs from result spent")
        requests = ledger["requests"]
        require(len(requests) == 5, f"{directory} ledger request count is {len(requests)}")
        running = 0.0
        for request in requests:
            running += float(request["cost_usd"])
            require(abs(running - float(request["total_usd"])) < 1e-9, f"{directory} ledger running total broke")
        require(abs(running - spent) < 1e-9, f"{directory} request costs do not add up")
        if arm == "control":
            require("experience_overlay" not in config, f"{directory} control config has an overlay")
            require(not (cell_dir / "experience_overlay.json").exists(), f"{directory} control has an overlay file")
            for row in archives:
                require(row["overlay_copies"] == 0 and row["experience_bytes"] == 13415, f"{directory}/{row['session']} control archive changed")
                require(row["experience_sha256"] == FILTERED_EXPERIENCE_SHA, f"{directory}/{row['session']} control experience sha changed")
                require(row["experience_overlay"] is None, f"{directory}/{row['session']} control archive has an overlay sidecar")
        else:
            sidecar = load(cell_dir / "experience_overlay.json")
            require(config.get("experience_overlay") == sidecar, f"{directory} run_config overlay differs from the run file")
            require(sidecar["overlay_copies"] == 1 and sidecar["arm"] == "treatment", f"{directory} overlay binding is not one treatment copy")
            require(sidecar["overlay_sha256"] == overlay_sha, f"{directory} overlay sha differs from the frozen file")
            require(sidecar["rendered_experience_sha256"] == RENDERED_EXPERIENCE_SHA, f"{directory} rendered experience sha changed")
            require(sidecar["rendered_experience_bytes"] == 25469, f"{directory} rendered experience length changed")
            require(sidecar["authoritative_task_rules"] is False and sidecar["not_a_task_contract"] is True, directory)
            require(sidecar["installed_in_global_experience"] is False, directory)
            require(sidecar["run_id"] == f"runs/p2u-experience-overlay-paid-20261114/{directory}", directory)
            for row in archives:
                require(row["overlay_copies"] == 1 and row["experience_bytes"] == 25469, f"{directory}/{row['session']} treatment archive is not one copy")
                require(row["experience_sha256"] == RENDERED_EXPERIENCE_SHA, f"{directory}/{row['session']} rendered sha changed")
                require(row["experience_overlay"] == sidecar, f"{directory}/{row['session']} archived sidecar differs")
        speeds[directory] = float(result["confirmed_speedup"])
        cells.append({
            "directory": directory,
            "arm": arm,
            "confirmed_speedup": result["confirmed_speedup"],
            "confirmed_status": "ok",
            "confirmation_source_sha256": source,
            "identity": "match",
            "run_seed": 7,
            "search_eval_seed": 7,
            "confirmation_seed": confirmation_seed,
            "best_speedup": result["best_speedup"],
            "best_label": result["best_label"],
            "confirm_winner_label": result["confirm_winner_label"],
            "spent_usd": result["spent_usd"],
            "spent_usd_text": _money(spent),
            "budget_usd": result["budget_usd"],
            "initial_budget_usd": result["initial_budget_usd"],
            "rounds_field": result["rounds"],
            "ledger_requests": len(requests),
            "session_archives": [{key: value for key, value in row.items() if key != "experience_overlay"} | {"overlay_sidecar": row["experience_overlay"] is not None} for row in archives],
            "soft_budget_crossing": _crossing(requests, 3.0),
            "result_sha256": sha256(cell_dir / "result.json"),
            "ledger_sha256": sha256(cell_dir / "ledger.json"),
            "subject_sha256": sha256(cell_dir / "confirmations/fp16_small_benchmark.json"),
        })

    differences = []
    for control, treatment, rep in PAIRS:
        difference = speeds[treatment] - speeds[control]
        differences.append(difference)
        require(rep in (1, 2), rep)
    ordered = sorted(differences)
    median = (ordered[0] + ordered[1]) / 2
    require(differences == summary["paired_differences"], "manual differences differ from the summarizer")
    require(median == summary["median_difference"], "manual median differs from the summarizer")
    require(differences[0] < 0 and differences[1] < 0, "a pair difference is not negative")
    lift_control = summary["control"]["lift_median"]
    lift_treatment = summary["treatment"]["lift_median"]
    require(lift_treatment > lift_control, "the recorded treatment lift is not the higher secondary number")
    require(summary["verdict"] == "no_gain", "the higher treatment lift changed the primary verdict")

    spent_values = [Decimal(_money(float(load(EXPERIMENT / cell / "result.json")["spent_usd"]))) for cell in CELLS]
    spent_sum = sum(spent_values, Decimal("0"))
    commitment_before_last = sum(spent_values[:3], Decimal("0"))
    require(str(spent_sum).startswith("14.094260376"), f"bill sum is {spent_sum}")
    require(abs(spent_sum - Decimal("14.094260376")) < Decimal("1e-12"), f"bill sum {spent_sum} left the stated total")
    require(str(commitment_before_last).startswith("11.137710888"), f"three-cell commitment is {commitment_before_last}")
    official_commitment = experiment_commitment(EXPERIMENT)
    require(abs(official_commitment - float(spent_sum)) < 1e-9, f"experiment_commitment {official_commitment} does not match the four cells")
    require(experiment_commitment(INTERRUPTED) == 0.0, "the interrupted archive contributes commitment")
    require(experiment_commitment(INTERRUPTED / "recovery-failed-1") == 0.0, "the failed recovery contributes commitment")
    three_float = float(load(EXPERIMENT / CELLS[0] / "result.json")["spent_usd"])
    three_float += float(load(EXPERIMENT / CELLS[1] / "result.json")["spent_usd"])
    three_float += float(load(EXPERIMENT / CELLS[2] / "result.json")["spent_usd"])
    require(fits_cap(three_float, 3.0, 12.0) is False, "the $12 cap would still have started the last cell")
    require(fits_cap(three_float, 3.0, 15.0) is True, "the $15 cap would not have started the last cell")
    require(official_commitment < 15.0, "the actual bill is not under the raised cap")

    recovery = load(EXPERIMENT / "treatment/gemm_silu_small_rep1/recovery.json")
    before_result = load(EXPERIMENT / "treatment/gemm_silu_small_rep1/result.before_topup_1.json")
    before_state = load(EXPERIMENT / "treatment/gemm_silu_small_rep1/state.before_resume_1.json")
    budget_events = json.loads((EXPERIMENT / "treatment/gemm_silu_small_rep1/budget_events.json").read_text(encoding="utf-8"))
    require(isinstance(budget_events, list) and len(budget_events) == 1, "treatment rep1 budget history is not the one restore")
    require(
        budget_events == load(EXPERIMENT / "treatment/gemm_silu_small_rep1/result.json")["budget_events"],
        "treatment rep1 budget_events.json differs from result.json",
    )
    event = budget_events[0]
    require(event["kind"] == "resume" and event["amount_usd"] == recovery["remaining_cell_budget_usd"], "add-budget amount differs from recovery.json")
    require(event["old_budget_usd"] == recovery["cost_usd"] and event["new_budget_usd"] == 3.0, "the restore did not return the cell to $3")
    require(recovery["new_model_call_for_init"] is False, "recovery claims a new init generation")
    require(recovery["candidate_sha256"] == INIT_CANDIDATE_SHA, "recovery candidate sha changed")
    require(recovery["cost_usd"] == before_result["spent_usd"], "recovery cost differs from the pre-resume result")
    first_request = load(EXPERIMENT / "treatment/gemm_silu_small_rep1/ledger.json")["requests"][0]
    require(first_request["label"] == "init" and first_request["cost_usd"] == recovery["cost_usd"], "the recovered bill is not the init ledger row")
    require(before_result["confirmation_source_sha256"] == INIT_CANDIDATE_SHA, "the pre-resume confirmation is not the orphan candidate")
    require(before_result["confirmed_status"] == "ok" and before_result["best_label"] == "init", "the pre-resume result is not the init confirmation")
    require(before_result["confirmed_speedup"] != speeds["treatment/gemm_silu_small_rep1"], "the paired number is the pre-resume confirmation")
    final_source = next(cell["confirmation_source_sha256"] for cell in cells if cell["directory"].endswith("treatment/gemm_silu_small_rep1"))
    require(final_source != INIT_CANDIDATE_SHA, "the final treatment rep1 subject is still the recovered init")
    require(before_state["evaluations"] == 1 and before_state["confirm_winner_label"] == "init", "the pre-resume state has no formal evaluation")
    for location in (
        INTERRUPTED / "treatment/gemm_silu_small_rep1/agent_work/init/gemm_silu-fp16-small-init.metal",
        INTERRUPTED / "recovery-failed-1/agent_work/init/gemm_silu-fp16-small-init.metal",
        EXPERIMENT / "treatment/gemm_silu_small_rep1/agent_work/init/gemm_silu-fp16-small-init.metal",
    ):
        require(sha256(location) == INIT_CANDIDATE_SHA, f"{location} is not the orphan candidate")
    orphan_state = load(INTERRUPTED / "treatment/gemm_silu_small_rep1/state.json")
    require(not (INTERRUPTED / "treatment/gemm_silu_small_rep1/result.json").exists(), "the orphan directory has a result")
    require(orphan_state["spent_usd"] == 0.0 and orphan_state["evaluations"] == 0, "the orphan outer state already had a candidate")
    require(orphan_state["best_label"] is None and orphan_state["trace"] == [], "the orphan outer state recorded an evaluation")
    require((INTERRUPTED / "treatment/gemm_silu_small_rep1/agent_work/init/.selfcheck/self_checks.jsonl").is_file(), "the orphan self-check log is missing")
    failed = load(INTERRUPTED / "recovery-failed-1/result.json")
    failed_ledger = load(INTERRUPTED / "recovery-failed-1/ledger.json")
    require(failed["spent_usd"] == 0.0 and failed["evaluations"] == 0 and failed["confirmed_speedup"] is None, "the failed recovery has a bill or a confirmation")
    require(failed["stop_reason"] == "agent_failure" and failed_ledger["requests"] == [], "the failed recovery is not an agent failure with an empty ledger")
    failed_text = (INTERRUPTED / "recovery-failed-1/transcript.jsonl").read_text(encoding="utf-8")
    require("File exists" in failed_text and ".selfcheck" in failed_text, "the failed transcript does not record the copytree collision")
    require("recovery replay only accepts the first init" in failed_text, "the failed transcript lost the second-committee refusal")
    orphan_experience = (INTERRUPTED / "treatment/gemm_silu_small_rep1/agent_work/init/experience.md").read_bytes()
    failed_experience = (INTERRUPTED / "recovery-failed-1/agent_work/init/experience.md").read_bytes()
    require(orphan_experience == failed_experience, "the failed attempt did not render the same experience file")
    require(hashlib.sha256(orphan_experience).hexdigest() == RENDERED_EXPERIENCE_SHA, "the orphan session experience is not the treatment render")
    require(orphan_experience.decode("utf-8").count(OVERLAY_MARK) == 1, "the orphan session does not contain one overlay")
    interrupted_results = sorted(path.relative_to(INTERRUPTED).as_posix() for path in INTERRUPTED.rglob("result.json"))
    require(interrupted_results == ["recovery-failed-1/result.json"], f"interrupted results are {interrupted_results}")

    require(before["max_total_usd"] == 12.0 and "cap_increase" not in before, "the pre-increase plan does not retain the $12 cap")
    require(before["stop"]["hard_cap_usd"] == 12.0 and before["stop"]["tolerable_overrun_approved"] is None, "the pre-increase stop rule changed")
    require(plan["max_total_usd"] == 15.0 and cap["max_total_usd"] == 15.0, "the live cap is not $15")
    require(plan["cap_increase"]["from_usd"] == 12.0 and plan["cap_increase"]["to_usd"] == 15.0, "the plan cap increase record changed")
    require(plan["cap_increase"]["previous_cells_topped_up"] is False, "the cap increase topped up an earlier cell")
    require(plan["budget_usd"] == 3.0 and before["budget_usd"] == 3.0, "the per-cell soft budget changed")
    require(plan["executed"] is False and plan["paid_search"] == "未授权/未执行" and plan["delta"] is None, "the plan file was rewritten after execution")
    require(approval["executed"] is False and approval["delta"] == 0.01 and approval["allowed_regression"] == 0.0, "the original approval no longer has delta 0.01")
    require(approval["max_total_commitment_usd"] == 12.0 and approval["per_cell_soft_budget_usd"] == 3.0, "the original approval cap changed")
    require(approval["approved_by"] == "kuafu" and cap_approval["approved_by"] == "kuafu", "an approval is unsigned")
    require(cap_approval["from_usd"] == 12.0 and cap_approval["to_usd"] == 15.0, "the cap approval amount changed")
    require(cap_approval["remaining_cell"] == "control/gemm_silu_small_rep2", "the cap approval names another cell")
    require(cap_approval["no_additional_cells"] is True and cap_approval["previous_cells_topped_up"] is False, "the cap approval adds a cell or a top-up")
    require(cap_approval["publication_allowed"] is False and approval["publication_allowed"] is False, "an approval allows publication")
    other_events = []
    for directory in CELLS:
        if directory == "treatment/gemm_silu_small_rep1":
            continue
        require(load(EXPERIMENT / directory / "result.json")["budget_events"] == [], f"{directory} has a budget event")
        other_events.append(directory)
    require(len(other_events) == 3, "a cell other than the recovered one was topped up")

    report = {
        "schema_id": "shapeforge.p2u-paid-pair.1",
        "experiment_id": "p2u-experience-overlay-paid-20261114",
        "experiment_complete": True,
        "complete_pairs": 2,
        "expected_pairs": 2,
        "excluded_pairs": [],
        "verdict": "no_gain",
        "adopt_overlay": False,
        "publication_allowed": False,
        "plan_3_complete": False,
        "plan_3_status": "executed_no_gain",
        "plan_4_started": False,
        "npu_simulated_only": True,
        "delta": 0.01,
        "allowed_regression": 0.0,
        "paired_differences": differences,
        "median_difference": median,
        "primary_metric": "treatment.confirmed_speedup - control.confirmed_speedup",
        "secondary": {"control": summary["control"], "treatment": summary["treatment"]},
        "secondary_does_not_override_primary": True,
        "scope": {
            "task": "gemm_silu",
            "dtype": "fp16",
            "size": "small",
            "pairs": 2,
            "protocol": "frozen experience-overlay pair, shared run_seed 7, search eval_seed literal 7",
        },
        "negative_result_scope": "这次 gemm_silu/fp16/small、两对、冻结协议下未获增益。不是「经验永远无用」的普遍证明。",
        "cells": cells,
        "system_prompt_sha256": system_sha,
        "system_prompt_contains_overlay": False,
        "cost": {
            "cell_spent_usd_text": [str(value) for value in spent_values],
            "sum_decimal": str(spent_sum),
            "stated_total_usd": "14.094260376",
            "commitment_before_last_cell_decimal": str(commitment_before_last),
            "stated_commitment_before_last_usd": "11.137710888",
            "official_experiment_commitment_usd": official_commitment,
            "interrupted_commitment_usd": 0.0,
            "recovery_failed_commitment_usd": 0.0,
            "recovery_cost_usd_text": _money(float(recovery["cost_usd"])),
            "recovery_cost_counted_once_inside_treatment_rep1": True,
            "soft_budget_usd": 3.0,
            "in_flight_overruns": [
                {"directory": cell["directory"], "crossing": cell["soft_budget_crossing"]}
                for cell in cells
                if cell["soft_budget_crossing"] is not None
            ],
        },
        "cap": {
            "original_commitment_cap_usd": 12.0,
            "raised_commitment_cap_usd": 15.0,
            "last_cell_blocked_at_12": True,
            "last_cell_fits_15": True,
            "actual_bill_under_15": True,
            "previous_cells_topped_up": False,
            "no_additional_cells": True,
            "plan_file_executed": False,
            "plan_file_paid_search": "未授权/未执行",
            "plan_file_delta": None,
            "approval_file_executed": False,
            "execution_record": "four confirmed result.json files",
        },
        "seeds": {
            "run_seed": 7,
            "search_eval_seed_literal": 7,
            "confirmation_seeds_are_derived": True,
            "eval_seed_function_result_is_not_the_search_literal": True,
        },
        "recovery": {
            "kind": "orphaned_init_replay",
            "breaks_pairing": False,
            "candidate_sha256": INIT_CANDIDATE_SHA,
            "new_model_call_for_init": False,
            "pre_resume_confirmed_speedup": before_result["confirmed_speedup"],
            "pre_resume_best_speedup": before_result["best_speedup"],
            "pre_resume_is_not_the_paired_metric": True,
            "selfcheck_is_not_the_paired_metric": True,
            "orphan_outer_evaluations": 0,
            "failed_attempt_spent_usd": 0.0,
            "failed_attempt_error": "copytree File exists on .selfcheck; no GPU evaluation and no bill",
            "add_budget_usd_text": _money(float(event["amount_usd"])),
            "token_account": recovery["spent"],
            "why_the_pair_stays": [
                "the orphan, the failed attempt, and the live init submission share candidate sha 1a305622",
                "every treatment session archive has one overlay copy and every control archive has zero",
                "the reconstructed init bill is the first ledger row of treatment rep1 and is not added again",
                "the paired speedup is the post-resume fresh confirmation of r1.p0, not the self-check and not the pre-resume init confirmation",
                "rep2 had no recovery and its difference is also negative",
            ],
        },
        "file_sha256": {
            "plan.json": sha256(EXPERIMENT / "plan.json"),
            "plan.before-cap-increase.json": sha256(EXPERIMENT / "plan.before-cap-increase.json"),
            "approval.json": sha256(EXPERIMENT / "approval.json"),
            "approval-cap-increase.json": sha256(EXPERIMENT / "approval-cap-increase.json"),
            "cap.json": sha256(EXPERIMENT / "cap.json"),
            "summary.json": sha256(EXPERIMENT / "summary.json"),
            "recover_init.py": sha256(EXPERIMENT / "recover_init.py"),
        },
    }
    return report


def canonical(document: dict) -> dict:
    return json.loads(json.dumps(document, sort_keys=True, allow_nan=False))

