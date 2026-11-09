"""Read-only MT4 verifier: real model generation, replay, resume and compatibility evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from taic.trust import current_prompt_sha1
from tcenv.identity import fingerprint
from tcenv.track import decode_envelope, load_track_registry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
IDENTITY = json.loads((HERE / "identity.json").read_text(encoding="utf-8"))
EXPECTED = IDENTITY["deepseek_relu"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(relative: str) -> dict:
    return json.loads((HERE / relative).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_identity() -> None:
    registry = load_track_registry(ROOT / "backends/metal/tracks.json")
    require(registry.get("tilelang-pipeline", version=1).status == "interface_only", "TileLang /1 drifted")
    require(registry.get("tilelang-pipeline", version=2).status == "available", "TileLang /2 is unavailable")
    require(fingerprint(registry.document()) == IDENTITY["track"]["registry_digest"], "registry identity drift")
    prompt = load("prompt-epoch.json")
    require(prompt["current_prompt_sha1"] == current_prompt_sha1() == IDENTITY["prompt"]["current_prompt_sha1"], "MT4 prompt source drift")
    require(prompt["previous_prompt_sha1"] == IDENTITY["prompt"]["previous_prompt_sha1"], "previous prompt epoch was rewritten")
    require(prompt["old_evidence_rewritten"] is False, "old evidence was rewritten")
    require(IDENTITY["reviewer_done"] is True and IDENTITY["done"] is True, "MT4 completion lacks independent review")
    review = load("review.json")
    actor_request = HERE / IDENTITY["review"]["actor_request"]
    require(review["done"] is True, "independent reviewer did not accept MT4")
    require(sha(actor_request) == IDENTITY["review"]["actor_request_sha256"], "actor review request identity drift")
    require(EXPECTED["publication_allowed"] is False and IDENTITY["speedup_publication"] is False, "correctness evidence became performance publication")


def verify_real_model() -> None:
    result = load("evidence/deepseek-relu-r1/result.json")
    require(result["model"] == "deepseek-flash" and result["best_model"] == "deepseek-flash", "wrong model identity")
    require(result["spent_usd"] == EXPECTED["spent_usd"] and result["evaluations"] == 1, "real run cost/evaluation drift")
    require(result["stop_reason"] == "max_rounds" and result["best_label"] == "init", "real run was not one initial slice")
    require(result["confirmed_speedup"] is None and result["candidate_us"] is None, "correctness-only result was turned into a performance result")
    require(result["best_speedup"] == EXPECTED["best_speedup_placeholder"], "the documented placeholder changed")
    config = load("evidence/deepseek-relu-r1/run_config.json")
    require(config["track"] == {"track_id": "tilelang-pipeline", "track_version": 2, "delivery": "dsl_source_pipeline"}, "run_config Track drift")
    require(config["search"]["correctness_only"] is True and config["search"]["max_rounds"] == 0, "run was not correctness-only/initial-only")
    require(config["search"]["budget_usd"] == 1.0 and config["search"]["session_cost_cap_usd"] == 0.5, "run budget drift")
    candidate = decode_envelope((HERE / "evidence/deepseek-relu-r1/best.taic").read_text(encoding="utf-8"))
    require(candidate.candidate_sha256 == EXPECTED["candidate_sha256"], "generated candidate bytes/identity drift")
    require([(entry.path, entry.role) for entry in candidate.entries] == [("host/execution_geometry.json", "schedule"), ("kernel.py", "dsl_source")], "candidate entries drift")
    report = load("evidence/deepseek-relu-r1/report.json")
    require(report["status"] == "ok" and report["abi"]["ok"] and report["correctness"]["ok"], "real model candidate is not correct")
    require(report["safety"]["ok"] and not report["safety"]["violations"], "real model candidate is unsafe")
    require(len(report["correctness"]["cases"]) == EXPECTED["correctness_cases"], "case count drift")
    require(all(case["ok"] for case in report["correctness"]["cases"]), "a real-model case failed")
    require(any(case["case"] == "ieee" and case["ok"] for case in report["correctness"]["cases"]), "IEEE case did not pass")
    require(report["build"]["shape"] == "full" and report["build"]["candidate_sha256"] == EXPECTED["candidate_sha256"], "real build identity drift")
    require(report["build"]["artifacts"]["artifact_set_digest"] == EXPECTED["artifact_set_digest"], "real artifact set drift")
    require(report["build"]["native_digest"] == EXPECTED["original_native_digest"], "original native identity drift")
    require(report["build"]["native"]["descriptor"]["tilelang"]["track"]["track_version"] == 2, "generated candidate ran on the wrong Track")
    require(report["candidate_us"] is None and report["speedup"] is None, "real model run has no benchmark and must not claim speedup")


def verify_replay() -> None:
    replay = load("evidence/deepseek-relu-r1/replay.json")
    result = replay["result"]
    require(result["candidate_sha256"] == EXPECTED["candidate_sha256"], "replay candidate identity drift")
    require(result["artifact_set_digest"] == EXPECTED["artifact_set_digest"], "replay artifact identity drift")
    require(result["native_digest"] == EXPECTED["replay_native_digest"], "replay native identity drift")
    require(result["native_digest"] != EXPECTED["original_native_digest"], "fresh replay reused original native identity")
    require(result["report"]["status"] == "ok", "replay correctness failed")
    dispatch = replay["execution"]["dispatch"]
    require(len(dispatch) == 1, "replay did not have exactly one actual dispatch")
    row = dispatch[0]
    require(row["kernel"] == EXPECTED["dispatch"]["kernel"] and row["grid"] == EXPECTED["dispatch"]["grid"] and row["threads"] == EXPECTED["dispatch"]["threads"], "actual dispatch drift")


def verify_resume() -> None:
    boundary = load("evidence/deepseek-relu-r1/cli-resume-boundary.json")
    require(boundary["ordinary_resume"]["exit_code"] == 3 and boundary["ordinary_resume"]["classification"] == "expected_refusal", "max_rounds refusal was hidden")
    require(boundary["dry_run"]["exit_code"] == 0 and boundary["dry_run"]["model_or_gpu_called"] is False, "resume dry-run was not read-only")
    text = (HERE / "evidence/deepseek-relu-r1/resume-dry-run.txt").read_text(encoding="utf-8")
    for needle in ('"best_label": "init"', '"system_prompt_restored": true', '"continues_in_round": 1', '"dry_run": true', '"candidates": 1'):
        require(needle in text, f"resume dry-run is missing {needle}")
    # The archived original result remains the pre-resume observation.
    require(load("evidence/deepseek-relu-r1/result.json")["best_label"] == "init", "original result was overwritten")


def verify_full_suite() -> None:
    document = load("full-suite-r2.json")
    expected = set(load("../MT1b/p2-epoch-drift.json")["failing_nodes"])
    actual = set(document["failed_nodes"])
    require(document["passed"] == 4963 and document["failed"] == 26 and document["skipped"] == 13, "full-suite totals drift")
    require(actual == expected and document["failed_nodes_match_mt1b_p2_epoch_drift"] is True, "full-suite failure set drift")
    require(not document["unexpected_failures"] and not document["missing_expected_failures"], "a current regression was hidden")
    parallel = document["tolerance_verdict_parallel"]
    require(parallel["nodes"] == 102 and parallel["semantics"] == "same frozen combinations and assertions", "parallel tolerance scope changed")
    source = (ROOT / "tests/backends/metal/test_tolerance_verdict.py").read_text(encoding="utf-8")
    require("snapshot.migrated_cases()" in source and "MIGRATED_COMBINATIONS" in source, "parallel tolerance no longer uses the frozen oracle")
    require("for path in sorted((FRESH_ROOT" not in source, "parallel tolerance broadened to all OpSpec tasks")


def self_test() -> None:
    forged = copy.deepcopy(load("evidence/deepseek-relu-r1/report.json"))
    forged["build"]["artifacts"]["artifact_set_digest"] = "0" * 64
    require(forged["build"]["artifacts"]["artifact_set_digest"] != EXPECTED["artifact_set_digest"], "self-test setup failed")
    try:
        require(forged["build"]["artifacts"]["artifact_set_digest"] == EXPECTED["artifact_set_digest"], "forged artifact was accepted")
    except ValueError:
        pass
    else:
        raise AssertionError("artifact forgery was accepted")
    original = load("evidence/deepseek-relu-r1/report.json")
    require(original["build"]["artifacts"]["artifact_set_digest"] == EXPECTED["artifact_set_digest"], "self-test modified raw evidence")


def verify_manifest() -> None:
    expected: dict[str, str] = {}
    for line in (HERE / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            expected[relative] = digest
    actual = {
        str(path.relative_to(HERE)): sha(path)
        for path in HERE.rglob("*")
        if path.is_file() and path.name != "manifest.sha256" and "__pycache__" not in path.parts
    }
    require(expected == actual, "manifest does not match every MT4 package file")


def main() -> int:
    verify_identity()
    verify_real_model()
    verify_replay()
    verify_resume()
    verify_full_suite()
    self_test()
    verify_manifest()
    print(json.dumps({
        "done": True,
        "verifier_passed": True,
        "reviewer_done": True,
        "stage": "MT4",
        "real_model": "deepseek-flash",
        "candidate_model_calls": 1,
        "real_replay": True,
        "full_suite": {"passed": 4963, "failed": 26, "skipped": 13},
        "parallel_tolerance_nodes": 102,
        "publication_allowed": False,
        "second_dsl_available": False,
        "mt0_overall_blocked": True,
        "manifest": True,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
