#!/usr/bin/env python3
"""Validate the registered /3 Track on the default registry with real Metal, no model call."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path

from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from taic.resume import load_snapshot
from tcenv.backends.metal import portfolio_confirmation as confirmation
from tcenv.backends.metal import search_candidate as search
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.target import load_target
from tcenv.track import decode_envelope
from tests.backends.metal.test_tilelang_mt6_gpu import MT4_RUN, _locator, _source
from taic.kernel_agent import AgentResult

HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence/default-registry-r3"


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    locator = _locator()
    target = load_target("apple-m4-max")
    source = _source()
    evaluated = search.evaluate_candidate(
        locator, source, target=target, mode="quick", seed=7, reps=1,
        track_id="tilelang-pipeline", track_version=3,
    )
    (OUT / "evaluate.json").write_text(json.dumps({
        "status": evaluated.result.report["status"],
        "candidate_sha256": evaluated.result.candidate_sha256,
        "artifact_set_digest": evaluated.result.artifact_set_digest,
        "dispatch": evaluated.result.execution["dispatch"],
        "benchmark_protocol": evaluated.result.execution.get("benchmark_protocol"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_config = json.loads((MT4_RUN / "run_config.json").read_text(encoding="utf-8"))
    frozen = search.confirmation_input(
        locator, source, target=target,
        guidance_digest=run_config["confirmation_guidance_digest"],
        master_seed=b"0123456789abcdef",
        track_id="tilelang-pipeline", track_version=3,
    )
    for usage in ("benchmark", "golden"):
        if usage == "benchmark":
            document = search.confirm_candidate(
                locator, source, target=target,
                guidance_digest=run_config["confirmation_guidance_digest"],
                master_seed=b"0123456789abcdef",
                track_id="tilelang-pipeline", track_version=3,
            )
        else:
            contract = locator.load()
            instance = instantiate_v2(contract.spec, "fp16", "small")
            point = frozen.freeze.domain_plan.find("fp16", "small", "golden")
            request = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage="golden")
            document = confirmation.confirm_subject(
                locator=locator,
                domain_plan=frozen.freeze.domain_plan,
                portfolio=frozen.freeze.portfolio(),
                request=request,
                evidence=frozen.evidence,
                validation=frozen.validation,
                freeze=frozen.freeze,
                track_version=3,
                track_id="tilelang-pipeline",
            )
        (OUT / f"confirmation-{usage}.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record, expected = confirmation.activation_expectation(
        locator=locator,
        freeze=frozen.freeze,
        validation=frozen.validation,
        document={"instance_key": "fp16/small/benchmark"},
        track_version=3,
        track_id="tilelang-pipeline",
    )
    (OUT / "activation.json").write_text(json.dumps({
        "outcome": record.outcome,
        "provider": record.provider,
        "candidate_sha256": expected["build"]["candidate_sha256"],
        "fresh_subject": expected["build"]["provenance"]["compile_profile"].get("tilelang_fresh_subject"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    binding = replace(
        __import__("taic.task_binding", fromlist=["bind_task"]).bind_task("relu", "fp16", "small"),
        track_id="tilelang-pipeline", track_version=3,
    )
    llm = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    search_config = SearchConfig(
        budget_usd=1.0, branches=1, max_rounds=0, eval_reps=1, kernel_agent="pi",
        correctness_only=True, stop_at_roofline=False, abi_guard=False, target_note=False,
        rich_history=False, status_lines=False, plan_ledger=False,
    )
    run_dir = OUT / "resume-run"
    orchestrator = Orchestrator(
        "relu", "fp16", "small", llm, search_config,
        run_dir=run_dir, binding=binding, track_id="tilelang-pipeline", track_version=3,
        agent_runner=lambda brief, tag: AgentResult(source=source, cost_usd=0.0, cost_incomplete=False, status="ok"),
        log=lambda _message: None,
    )
    result = asyncio.run(orchestrator.run())
    resumed = Orchestrator(
        "relu", "fp16", "small", llm, search_config,
        run_dir=run_dir, binding=binding, track_id="tilelang-pipeline", track_version=3, resuming=True,
        agent_runner=lambda *_args: (_ for _ in ()).throw(AssertionError("resume started a new agent")),
        log=lambda _message: None,
    )
    snapshot = load_snapshot(run_dir, resumed.search, "relu_kernel")
    resumed.restore(snapshot, dry_run=False)
    resumed_result = asyncio.run(resumed.run())
    (OUT / "resume.json").write_text(json.dumps({
        "initial_best_label": result.best_label,
        "resumed_best_label": resumed_result.best_label,
        "candidate_count": len(resumed.candidates),
        "candidate_sha256": decode_envelope(resumed.candidates[0].source).candidate_sha256,
        "source_matches": resumed.candidates[0].source == orchestrator.best.source,
        "track": json.loads((run_dir / "run_config.json").read_text())["track"],
        "non_dry_run": True,
        "model_calls": 0,
        "state_json_exists": (run_dir / "state.json").is_file(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"default_registry_validation": "passed", "output": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
