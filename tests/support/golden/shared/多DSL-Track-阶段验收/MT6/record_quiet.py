#!/usr/bin/env python3
"""Record three complete fixed-candidate /3 fresh observations on real Metal.

Explicit GPU authorization is required.  No model call, no candidate generation and no production
registry mutation.  Every verdict is written immediately; timing rejections never stop the series.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from tcenv.backends.metal import portfolio_confirmation as confirmation
from tcenv.backends.metal import search_candidate as search
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.target import load_target
from tcenv.track import tilelang_confirmation as fresh
from tcenv.track.registry import load_track_registry
from tests.backends.metal.test_tilelang_mt6_gpu import MT4_RUN, _locator, _registry_file, _source

HERE = Path(__file__).resolve().parent
RUNS = 3
_RECOVERED = Path("/tmp/shapeforge-frontends/tilelang-0.1.15-recovered")
if _RECOVERED.is_dir():
    os.environ.setdefault("SHAPEFORGE_TILELANG_STAGING", str(_RECOVERED))


def main(output_name: str) -> None:
    if not output_name.startswith("real-metal-quiet-") or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in output_name):
        raise ValueError("output_name must be a safe real-metal-quiet-* evidence epoch")
    out = HERE / "evidence" / output_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite the closed evidence epoch: {out}")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mt6-quiet-registry-") as directory:
        registry_path = _registry_file(Path(directory))
        selected = load_track_registry(registry_path)
        with (
            patch("tcenv.portfolio.candidate.load_track_registry", lambda *args, **kwargs: selected),
            patch("tcenv.taskcontract.runner.load_track_registry", lambda *args, **kwargs: selected),
            patch.object(search, "load_track_registry", lambda *args, **kwargs: selected),
            patch.object(confirmation, "load_track_registry", lambda *args, **kwargs: selected),
            patch.object(fresh, "load_track_registry", lambda *args, **kwargs: selected),
        ):
            locator = _locator()
            target = load_target("apple-m4-max")
            run_config = json.loads((MT4_RUN / "run_config.json").read_text(encoding="utf-8"))
            frozen = search.confirmation_input(
                locator,
                _source(),
                target=target,
                guidance_digest=run_config["confirmation_guidance_digest"],
                master_seed=b"0123456789abcdef",
                track_id="tilelang-pipeline",
                track_version=3,
            )
            contract = locator.load()
            instance = instantiate_v2(contract.spec, "fp16", "small")
            point = frozen.freeze.domain_plan.find("fp16", "small", "golden")
            golden_request = DispatchRequest.from_instance(
                instance, target_abi_digest=point.target_abi_digest, usage="golden"
            )
            rows = []
            for ordinal in range(1, RUNS + 1):
                run_dir = out / f"r{ordinal}"
                run_dir.mkdir(exist_ok=True)
                documents = {}
                for usage, request in (("benchmark", frozen.request), ("golden", golden_request)):
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
                    documents[usage] = document
                    (run_dir / f"fresh-{usage}.json").write_text(
                        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                benchmark = documents["benchmark"]
                golden = documents["golden"]
                search_report = benchmark["search_report"]
                fresh_report = benchmark["confirmation_report"]
                row = {
                    "run": f"r{ordinal}",
                    "status": benchmark["status"],
                    "reason": benchmark["reason"],
                    "confirmation_attempts": benchmark["confirmation_attempts"],
                    "identity_kind": benchmark["identity"]["kind"],
                    "candidate_us": fresh_report["candidate_us"],
                    "baseline_mlx_us": fresh_report["baseline_us"]["mlx"],
                    "speedup": fresh_report["speedup"],
                    "latency_ratio": fresh_report["candidate_us"] / fresh_report["baseline_us"]["mlx"],
                    "noise_cv": fresh_report["benchmark"]["noise_cv"],
                    "noisy": fresh_report["benchmark"]["noisy"],
                    "native_digest_match": search_report["build"]["native_digest"] == fresh_report["build"]["native_digest"],
                    "artifact_set_match": search_report["build"]["artifacts"]["artifact_set_digest"] == fresh_report["build"]["artifacts"]["artifact_set_digest"],
                    "compile_profile_match": search_report["build"]["provenance"]["compile_profile"] == fresh_report["build"]["provenance"]["compile_profile"],
                    "golden_status": golden["status"],
                    "golden_reason": golden["reason"],
                    "golden_identity_kind": golden["identity"]["kind"],
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    stable_timing = all(
        row["status"] == "confirmed"
        and row["golden_status"] == "confirmed"
        and row["identity_kind"] == "match"
        and row["golden_identity_kind"] == "match"
        and row["native_digest_match"]
        and row["artifact_set_match"]
        and row["compile_profile_match"]
        for row in rows
    )
    # A confirmed sidecar proves identity/correctness/quiet timing, not the separate 1.10
    # latency-ratio performance gate.  Keep both predicates explicit.
    performance_gate = stable_timing and all(row["latency_ratio"] <= 1.10 for row in rows)
    summary = {
        "schema_id": "shapeforge.mt6.real-metal-quiet-series.1",
        "series_id": output_name,
        "track_id": "tilelang-pipeline",
        "track_version": 3,
        "candidate_sha256": benchmark["candidate_sha256"],
        "instance_key": "fp16/small/benchmark",
        "real_metal": True,
        "model_executed": False,
        "runs_requested": RUNS,
        "runs_completed": len(rows),
        "runs": rows,
        "stable_fresh_timing": stable_timing,
        "stable_fresh_performance": performance_gate,
        "performance_gate_passed": performance_gate,
        "previous_history_preserved": True,
        "registered": False,
        "selectable": False,
        "publication_allowed": False,
        "done": False,
        "reviewer_done": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "stable_fresh_timing": stable_timing,
        "stable_fresh_performance": performance_gate,
        "runs": len(rows),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    main(args.output_name)
