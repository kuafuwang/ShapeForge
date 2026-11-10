"""V6 real-scale chunked/matrix acceptance: fresh confirmation + active manifests.

This script runs real Metal work and therefore requires explicit GPU authorization.  It does not
invoke paid models, modify P2, or fabricate performance evidence.  The six instances are
D=128, H=64/96, fixed/mixed/uniform, total_tokens=8192.  A subset rerun updates only that shard;
``summary.json`` is confirmed only when all six shard summaries are active.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from taic.task_binding import bind_task
from taic.portfolio_manifest import DispatchManifest
from tcenv.backends.metal.portfolio_confirmation import activation_expectation, confirm_subject
from tcenv.backends.metal.search_candidate import confirm_candidate, confirmation_input, guidance_fingerprint
from tcenv.contract_errors import TaskContractError
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.target import load_target

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "tests/support/golden/metal/varlen/chunked_gform.metal"
SIZES = (
    "v6_h64_fixed",
    "v6_h64_mixed",
    "v6_h64_uniform",
    "v6_h96_fixed",
    "v6_h96_mixed",
    "v6_h96_uniform",
)


def _measurements(document: dict | None) -> dict:
    if not document:
        return {"candidate": None, "baselines": None, "speedup": None}
    report = document["confirmation_report"]
    benchmark = report.get("benchmark") or {}
    return {
        "candidate": benchmark.get("candidate"),
        "baselines": benchmark.get("baselines"),
        "speedup": report.get("speedup"),
    }


def _rejected(size: str, benchmark: dict, golden: dict | None, reason: str) -> dict:
    return {
        "size": size,
        "status": "rejected",
        "reason": reason,
        "manifest": None,
        "benchmark_confirmation": benchmark,
        "golden_confirmation": golden,
    }


def confirm_size(size: str, source: str, target) -> dict:
    binding = bind_task("varlen_kda", "fp16", size)
    master = f"v6-chunked-{size}-fresh-0001".encode()
    guidance = guidance_fingerprint("v6-chunked-gform", size)
    frozen = confirmation_input(binding.locator, source, target=target, guidance_digest=guidance, master_seed=master)
    benchmark = confirm_candidate(binding.locator, source, target=target, guidance_digest=guidance, master_seed=master)
    if benchmark["status"] != "confirmed":
        return _rejected(size, benchmark, None, str(benchmark.get("reason") or "benchmark_not_confirmed"))
    golden_request = DispatchRequest.from_instance(
        binding.contract.instance, target_abi_digest=frozen.request.target_abi_digest, usage="golden"
    )
    golden = confirm_subject(
        locator=binding.locator,
        domain_plan=frozen.freeze.domain_plan,
        portfolio=frozen.freeze.portfolio(),
        request=golden_request,
        evidence=frozen.evidence,
        validation=frozen.validation,
        freeze=frozen.freeze,
    )
    if golden["status"] != "confirmed":
        return _rejected(size, benchmark, golden, str(golden.get("reason") or "golden_not_confirmed"))
    documents = [benchmark, golden]
    expectations = {}
    for document in documents:
        record, expected = activation_expectation(
            locator=binding.locator, freeze=frozen.freeze, validation=frozen.validation, document=document
        )
        expectations[document["instance_key"]] = (record, expected)
    try:
        manifest = DispatchManifest.from_confirmations(
            frozen.freeze, frozen.validation, documents, expectations, locator=binding.locator
        )
        activated = manifest.activate(frozen.freeze, frozen.validation)
    except TaskContractError as exc:
        return _rejected(size, benchmark, golden, str(exc))
    return {
        "size": size,
        "status": activated.status,
        "reason": None,
        "manifest": activated.document(),
        "benchmark_confirmation": benchmark,
        "golden_confirmation": golden,
    }


def _shard_summary(result: dict) -> dict:
    manifest = result.get("manifest")
    benchmark = result["benchmark_confirmation"]
    golden = result.get("golden_confirmation")
    return {
        "size": result["size"],
        "status": result["status"],
        "reason": result.get("reason"),
        "dispatch_manifest_digest": manifest.get("dispatch_manifest_digest") if manifest else None,
        "activation_generation": manifest.get("activation_generation") if manifest else None,
        "candidate_sha256": benchmark.get("candidate_sha256"),
        **_measurements(benchmark),
        "benchmark_confirmation_digest": benchmark.get("confirmation_digest"),
        "golden_confirmation_digest": golden.get("confirmation_digest") if golden else None,
    }


def check_size(size: str, source: str, target, out: Path) -> str:
    """Rebuild activation from archived sidecars without running GPU work."""
    binding = bind_task("varlen_kda", "fp16", size)
    master = f"v6-chunked-{size}-fresh-0001".encode()
    guidance = guidance_fingerprint("v6-chunked-gform", size)
    frozen = confirmation_input(binding.locator, source, target=target, guidance_digest=guidance, master_seed=master)
    documents = [
        json.loads((out / "confirmations" / f"{size}.{usage}.json").read_text(encoding="utf-8"))
        for usage in ("benchmark", "golden")
    ]
    expectations = {}
    for document in documents:
        record, expected = activation_expectation(
            locator=binding.locator, freeze=frozen.freeze, validation=frozen.validation, document=document
        )
        expectations[document["instance_key"]] = (record, expected)
    activated = DispatchManifest.from_confirmations(
        frozen.freeze, frozen.validation, documents, expectations, locator=binding.locator
    ).activate(frozen.freeze, frozen.validation)
    archived = json.loads((out / "manifests" / f"{size}.json").read_text(encoding="utf-8"))
    if activated.document() != archived:
        raise TaskContractError(f"{size}: archived manifest is not the independently rebuilt activation")
    return activated.digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", default=list(SIZES))
    parser.add_argument("--out", default=str(HERE))
    parser.add_argument("--check", action="store_true", help="verify archived sidecars/manifests without GPU work")
    args = parser.parse_args()
    unknown = sorted(set(args.sizes) - set(SIZES))
    if unknown:
        parser.error(f"unknown V6 sizes: {unknown}")
    out = Path(args.out)
    if args.check:
        target = load_target("apple-m4-max")
        source = SOURCE.read_text(encoding="utf-8")
        for size in args.sizes:
            digest = check_size(size, source, target, out)
            print(f"[v6] checked {size}: active {digest}", flush=True)
        return 0
    for directory in (out / "manifests", out / "confirmations", out / "results"):
        directory.mkdir(parents=True, exist_ok=True)
    target = load_target("apple-m4-max")
    source = SOURCE.read_text(encoding="utf-8")
    current = []
    for size in args.sizes:
        print(f"[v6] confirming {size} ...", flush=True)
        result = confirm_size(size, source, target)
        current.append(result)
        benchmark = result["benchmark_confirmation"]
        golden = result.get("golden_confirmation")
        (out / "confirmations" / f"{size}.benchmark.json").write_text(
            json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if golden is not None:
            (out / "confirmations" / f"{size}.golden.json").write_text(
                json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if result.get("manifest") is not None:
            (out / "manifests" / f"{size}.json").write_text(
                json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        shard = _shard_summary(result)
        (out / "results" / f"{size}.json").write_text(
            json.dumps(shard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        measurements = _measurements(benchmark)
        candidate = measurements["candidate"] or {}
        baselines = measurements["baselines"] or {}
        candidate_ms = candidate.get("median_s", 0.0) * 1000.0
        baseline_ms = {name: value.get("median_s", 0.0) * 1000.0 for name, value in baselines.items()}
        print(
            f"[v6] {size}: {result['status']}; reason={result.get('reason')}; "
            f"candidate={candidate_ms:.3f} ms; baselines_ms={baseline_ms}; speedup={measurements['speedup']}",
            flush=True,
        )
    shards = []
    for size in SIZES:
        path = out / "results" / f"{size}.json"
        if path.exists():
            shards.append(json.loads(path.read_text(encoding="utf-8")))
    if any(item["status"] != "active" for item in shards):
        status = "rejected"
    elif len(shards) == len(SIZES):
        status = "confirmed"
    else:
        status = "incomplete"
    summary = {
        "schema_id": "shapeforge.v6.chunked-acceptance.1",
        "status": status,
        "requested_sizes": list(args.sizes),
        "results": shards,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "active_shards": sum(item["status"] == "active" for item in shards)}, indent=2))
    return 0 if all(item["status"] == "active" for item in current) else 1


if __name__ == "__main__":
    sys.exit(main())
