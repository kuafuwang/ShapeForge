"""Read-only MT2 verifier: candidate bytes, canonical identity, real observations and negatives."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

from tcenv.backends.metal.track_factory import adapter_for_track
from tcenv.build_projection import validate_for_status
from tcenv.remote.execution_geometry import verify_evaluator_build
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.pack import task_pack_closure
from tcenv.track import (
    CanonicalBundle, CandidateArtifact, CandidateEntry, REPO_ROOT,
    abi_limits_from_target, candidate_from_single_source, candidate_geometry,
    geometry_pair, load_track_registry,
)
from tcenv.track import tilelang as tilelang_contract
from tcenv.track.abi import derive_tensor_memory_contract
from tcenv.target import load_target

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
IDENTITY = json.loads((HERE / "identity.json").read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(relative: str) -> dict:
    return json.loads((HERE / relative).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(name: str) -> CandidateArtifact:
    if name == "msl":
        return candidate_from_single_source((HERE / "candidates/msl-comparison.metal").read_text(encoding="utf-8"))
    return CandidateArtifact(
        track_id="tilelang-pipeline",
        entries=(
            CandidateEntry(
                path="kernel.py", role="dsl_source", media_type="text/x-python",
                data=(HERE / f"candidates/{name}-kernel.py").read_bytes(),
            ),
            CandidateEntry(
                path="host/execution_geometry.json", role="schedule", media_type="application/json",
                data=(HERE / f"candidates/{name}-geometry.json").read_bytes(),
            ),
        ),
    )


def canonical(name: str) -> CanonicalBundle:
    task = load_task_contract(HERE / "candidates/mt2_scale")
    target = load_target("apple-m4-max")
    abi = derive_tensor_memory_contract(task.instance, target=target)
    limits = abi_limits_from_target(target)
    submitted = candidate(name)
    if name == "msl":
        from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver

        geometry = msl_single_geometry_resolver(task)(submitted)
    else:
        geometry = candidate_geometry(submitted)
    pair = geometry_pair(geometry, None, instance=task.instance, contract=abi, limits=limits)
    return CanonicalBundle(candidate=submitted, pair=pair)


def builder():
    spec = load_track_registry(ROOT / "backends/metal/tracks.json").get("tilelang-pipeline", version=2)
    return adapter_for_track(spec).resolve("builder")(target_id="apple-m4-max")


def verify_identity() -> None:
    registry = load_track_registry(ROOT / "backends/metal/tracks.json")
    v1 = registry.get("tilelang-pipeline", version=1)
    v2 = registry.get("tilelang-pipeline", version=2)
    require(v1.status == "interface_only" and not v1.selectable, "TileLang /1 no longer preserves interface_only")
    require(v2.status == "available" and v2.selectable and v2.supports("metal_dispatch"), "TileLang /2 is not available")
    from tcenv.identity import fingerprint

    require(fingerprint(v1_contract := tilelang_contract.frozen_contract()) == IDENTITY["v1"]["contract_digest"], "/1 contract drift")
    require(fingerprint(tilelang_contract.frozen_contract(track_version=2)) == IDENTITY["track"]["contract_digest"], "/2 contract drift")
    require(fingerprint(registry.document()) == IDENTITY["track"]["registry_digest"], "registry digest drift")
    require(v1_contract["build"]["implemented_modes"] == ["source_only"], "/1 silently gained Metal")
    audit = tilelang_contract.audit_registration(v2)
    require(audit["contract_digest"] == IDENTITY["track"]["contract_digest"], "registered /2 contract differs from frozen contract")
    closure = task_pack_closure(ROOT / "tasks/relu", root=ROOT)
    require(len(closure.code_files) == IDENTITY["task_pack"]["closure_files"], "task pack closure count drift")
    require(closure.digest == IDENTITY["task_pack"]["example_digest"], "task pack digest drift was not recorded")
    require(IDENTITY["reviewer_done"] is True and IDENTITY["done"] is True, "MT2 completion lacks its independent review")
    review = load("review.json")
    require(review["done"] is True and review["independent_sessions"] is True, "MT2 reviewer did not independently accept")
    require(sha(Path(review["log"])) == review["log_sha256"] == IDENTITY["review"]["log_sha256"], "MT2 reviewer log identity drift")
    require(IDENTITY["publication_allowed"] is False, "MT2 mechanism evidence is not publication evidence")


def verify_positive(name: str, relative: str, *, observation: dict | None = None) -> None:
    observation = load(relative) if observation is None else observation
    result = observation["result"]
    report = result["report"]
    expected = IDENTITY["candidates"][name]
    require(observation["real_metal"] is True and observation["model_calls"] == 0, f"{name}: not a fixed real-Metal observation")
    require(observation["paid_search"] is False and observation["publication_allowed"] is False, f"{name}: publication boundary drift")
    require(result["candidate_sha256"] == expected["sha256"], f"{name}: candidate bytes/identity drift")
    require(result["bundle_digest"] == expected["bundle_digest"], f"{name}: canonical bundle digest drift")
    require(result["artifact_set_digest"] == expected["artifact_set_digest"], f"{name}: artifact set identity drift")
    require(report["status"] == "ok" and report["abi"]["ok"] and report["correctness"]["ok"], f"{name}: not correct")
    require(report["backend"] == "metal" and result["track"]["track_id"] == ("msl-single" if name == "msl" else "tilelang-pipeline"), f"{name}: wrong Track")
    require("M4" in report["device_name"] and report["target_id"] == "apple-m4-max", f"{name}: unexpected device identity")
    require(report["benchmark"]["candidate"]["median_us"] > 0 and report["benchmark"]["noisy"] is False, f"{name}: invalid timing")
    require(report["speedup"] is None, f"{name}: a baseline-free mechanism result cannot publish speedup")
    execution = observation["execution"]
    require(execution["benchmark_protocol"]["rotation_sets"] == 16, f"{name}: fresh rotation drift")
    require(execution["benchmark_guard_violations"] == [] and execution["host_runs"] == [], f"{name}: guard/host boundary drift")
    require(execution["worker"]["registry_digest"] == IDENTITY["track"]["registry_digest"], f"{name}: worker registry drift")
    build = observation["build"]
    require(build["shape"] == "full" and build["candidate_sha256"] == expected["sha256"], f"{name}: full build identity drift")
    require(build["provenance"]["artifact_set_digest"] == expected["artifact_set_digest"], f"{name}: build artifact identity drift")
    require(validate_for_status(build, status="ok") == build, f"{name}: schema-4 build is not canonical")
    canonical_bundle = canonical(name)
    require(canonical_bundle.candidate_sha256 == expected["sha256"], f"{name}: packaged candidate digest mismatch")
    require(canonical_bundle.bundle_digest == expected["bundle_digest"], f"{name}: independently rebuilt bundle mismatch")
    require(canonical_bundle.geometry_identity == result["geometry_identity"], f"{name}: geometry identity mismatch")
    task = load_task_contract(HERE / "candidates/mt2_scale")
    require(build["opspec_digest"] == task.spec.digest, f"{name}: opspec identity mismatch")
    identity = {"opspec_digest": build["opspec_digest"], "lowering_profile_digest": build["lowering_profile_digest"]}
    verified = verify_evaluator_build(
        build,
        canonical_bundle=canonical_bundle,
        identity=identity,
        artifact_audit=None if name == "msl" else builder().audit_evaluator_build,
    )
    require(verified.artifact_set_digest == expected["artifact_set_digest"], f"{name}: independent evaluator audit failed")
    if name != "msl":
        require(execution["tilelang_build"]["attempt"]["returncode"] == 0, f"{name}: frontend attempt failed")
        require(execution["tilelang_build"]["attempt"]["online"] is False, f"{name}: frontend was online")
        require(build["native"]["descriptor"]["tilelang"]["frontend_entry_sha256"] == IDENTITY["frontend"]["entry_sha256"], f"{name}: frontend entry drift")
        require(build["provenance"]["builder_id"] == "tilelang-metal-track-builder", f"{name}: wrong builder identity")
        require({entry["path"] for entry in build["artifacts"]["entries"]} == {
            "kernel.py", "host/execution_geometry.json", "candidate_manifest.json", "execution_geometry.json",
            "frontend_identity.json", "kernel.metal", "device_ir.json", "kernel_params.json", "host_module.txt",
        }, f"{name}: generated artifact set is not closed")
        bindings = execution["bindings"]
        require(any(arg["name"] == "acc" and arg["access"] == "read_write" for binding in bindings for arg in binding["arguments"]), f"{name}: read_write acc was not bound")
    if name == "single":
        require([(row["kernel"], row["grid"], row["threads"]) for row in execution["dispatch"]] == [("scale_kernel_kernel", [4, 1, 1], [256, 1, 1])], "single: dispatch drift")
    elif name == "worklist":
        require([(row["kernel"], row["grid"], row["threads"], row["workspace"]) for row in execution["dispatch"]] == [
            ("scale_kernel_kernel", [8, 1, 1], [128, 1, 1], ["tmp"]),
            ("bias_kernel_kernel", [8, 1, 1], [128, 1, 1], ["tmp"]),
        ], "worklist: candidate-owned stage order/dispatch drift")
        require(execution["workspace_bytes"] == {"tmp": 4096}, "worklist: workspace allocation drift")
    else:
        require([(row["kernel"], row["grid"], row["threads"]) for row in execution["dispatch"]] == [("scale_msl", [4, 1, 1], [256, 1, 1])], "msl: dispatch drift")


def verify_negatives() -> None:
    for name in ("invalid_kernel", "external_read"):
        observation = load(f"evidence/r4/negative-{name}.json")
        result = observation["result"]
        report = result["report"]
        require(report["status"] == "compile_error" and report["build"]["shape"] == "partial", f"{name}: not a partial compile failure")
        require(report["build"]["candidate_sha256"] == result["candidate_sha256"], f"{name}: partial candidate identity drift")
        require(report["build"]["provenance"]["artifact_set_digest"] is None, f"{name}: partial build claimed artifacts")
        require(validate_for_status(report["build"], status="compile_error") == report["build"], f"{name}: invalid partial build")
        require(not observation["execution"] or not observation["execution"].get("dispatch"), f"{name}: device dispatch occurred")
        require(observation["publication_allowed"] is False, f"{name}: failure was publishable")
        errors = report["compile"]["errors"]
        require("PermissionError" in errors if name == "external_read" else "prim_func" in errors, f"{name}: wrong failure attribution")
    observation = load("evidence/r4/negative-out-of-bounds.json")
    report = observation["result"]["report"]
    require(report["status"] == "unsafe" and report["build"]["shape"] == "full", "out-of-bounds was not a real-build safety rejection")
    require(validate_for_status(report["build"], status="unsafe") == report["build"], "out-of-bounds build is not schema-4")
    require(observation["publication_allowed"] is False, "out-of-bounds was publishable")
    control = load("evidence/history/system-read-allowed-control.json")
    require(control["result"]["report"]["status"] == "ok", "the declared system-read control was misclassified")
    require(control["publication_allowed"] is False, "system-read control became publication evidence")
    for relative, status in (("evidence/history/worklist-r1-abi-error.json", "abi_error"), ("evidence/history/worklist-r2-poison-unsafe.json", "unsafe")):
        require(load(relative)["report"]["status"] == status, f"failure history was rewritten: {relative}")


def verify_semantic_comparison() -> None:
    single = load("evidence/r4/tilelang-single.json")["result"]["report"]["correctness"]["cases"]
    msl = load("evidence/r4/msl-single-comparison.json")["result"]["report"]["correctness"]["cases"]
    signature = lambda cases: [(row["case"], row["seed"], row["scaled_err"]) for row in cases]
    require(signature(single) == signature(msl), "TileLang and MSL did not produce the same case/seed/error signature")


def verify_no_msl_substitution() -> None:
    for relative in (
        "tcenv/backends/metal/kernel_pipeline.py",
        "tcenv/backends/metal/tilelang_metal.py",
        "tcenv/backends/metal/tilelang_runtime.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        require(not any("track_adapter" in name or "HostTilingPlan" in name for name in imported), f"{relative}: MSL substitution")


def verify_shipped() -> None:
    shipped = load("evidence/r4/shipped-worklist.json")
    main = load("evidence/r4/tilelang-worklist.json")
    require(shipped["result"]["track"]["track_version"] == 2, "shipped registry did not select /2")
    for field in ("candidate_sha256", "bundle_digest", "artifact_set_digest"):
        require(shipped["result"][field] == main["result"][field], f"shipped/main {field} mismatch")
    require(shipped["execution"]["worker"]["registry_digest"] == IDENTITY["track"]["registry_digest"], "shipped registry drift")
    require(shipped["result"]["native_digest"] != main["result"]["native_digest"], "fresh attempts must not reuse the old native identity")
    require(shipped["result"]["report"]["benchmark"]["noisy"] is False, "shipped run was noisy")


def verify_confirmation() -> None:
    """The independent r6 run must reproduce r4 identities; timings stay per-run observations."""
    for name, filename in (("single", "tilelang-single.json"), ("worklist", "tilelang-worklist.json"), ("msl", "msl-single-comparison.json")):
        verify_positive(name, f"evidence/confirmation-r6/{filename}")
    verify_positive("worklist", "evidence/confirmation-r6/shipped-worklist.json")
    for filename in ("tilelang-single.json", "tilelang-worklist.json", "shipped-worklist.json", "msl-single-comparison.json"):
        r4 = load(f"evidence/r4/{filename}")
        r6 = load(f"evidence/confirmation-r6/{filename}")
        require(r4["result"]["artifact_set_digest"] == r6["result"]["artifact_set_digest"], f"r4/r6 artifact drift: {filename}")
        require(r4["result"]["candidate_sha256"] == r6["result"]["candidate_sha256"], f"r4/r6 candidate drift: {filename}")
        if filename != "msl-single-comparison.json":
            require(r4["result"]["native_digest"] != r6["result"]["native_digest"], f"fresh TileLang attempt reused native identity: {filename}")
    for filename, status in (("negative-invalid_kernel.json", "compile_error"), ("negative-external_read.json", "compile_error"), ("negative-out-of-bounds.json", "unsafe")):
        require(load(f"evidence/confirmation-r6/{filename}")["result"]["report"]["status"] == status, f"r6 negative verdict drift: {filename}")


def verify_full_suite() -> None:
    document = load("full-suite-r2.json")
    expected = set(load("../MT1b/p2-epoch-drift.json")["failing_nodes"])
    require(document["passed"] == 4861 and document["failed"] == 26 and document["skipped"] == 13, "full-suite totals drift")
    require(document["failed_nodes_match_mt1b_p2_epoch_drift"] is True, "full-suite failure set was not classified")
    require(not document["unexpected_failures"] and not document["missing_expected_failures"], "a new or missing P2 failure was hidden")
    require(len(expected) == 26 and document["gpu_or_model_executed"] is False, "full-suite scope drift")


def self_test() -> None:
    forged = copy.deepcopy(load("evidence/r4/tilelang-worklist.json"))
    forged["build"]["native"]["descriptor"]["tilelang"]["candidate_sha256"] = "0" * 64
    # Reseal only the native digest so the mutation reaches the Track-specific candidate audit
    # instead of being rejected earlier by the generic native-digest cross-check.
    from tcenv.remote.artifact_bundle import NativeExecutable

    native = forged["build"]["native"]
    forged["build"]["native_digest"] = NativeExecutable(
        backend_id=native["backend_id"], kind=native["kind"], entry_path=native["entry_path"],
        entry_sha256=native["entry_sha256"], descriptor=native["descriptor"],
    ).digest
    try:
        verify_positive("worklist", "evidence/r4/tilelang-worklist.json", observation=forged)
    except ValueError as exc:
        require("candidate" in str(exc), f"forged descriptor was rejected for the wrong reason: {exc}")
    else:
        raise AssertionError("a forged TileLang native descriptor passed independent audit")
    original = load("evidence/r4/tilelang-worklist.json")
    require(original["result"]["candidate_sha256"] == IDENTITY["candidates"]["worklist"]["sha256"], "self-test modified raw evidence")


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
    require(expected == actual, "manifest does not match every MT2 package file")


def main() -> int:
    verify_identity()
    for name, relative in (
        ("single", "evidence/r4/tilelang-single.json"),
        ("worklist", "evidence/r4/tilelang-worklist.json"),
        ("msl", "evidence/r4/msl-single-comparison.json"),
    ):
        verify_positive(name, relative)
    verify_negatives()
    verify_semantic_comparison()
    verify_no_msl_substitution()
    verify_shipped()
    verify_confirmation()
    verify_full_suite()
    self_test()
    verify_manifest()
    print(json.dumps({
        "done": True,
        "verifier_passed": True,
        "reviewer_done": True,
        "stage": "MT2",
        "track": "tilelang-pipeline/2",
        "available_scope": IDENTITY["track"]["scope"],
        "fixed_gpu_nodes": 7,
        "fixed_cpu_nodes": 8,
        "real_metal": True,
        "model_or_paid_search": False,
        "publication_allowed": False,
        "second_dsl_available": False,
        "mt0_overall_blocked": True,
        "manifest": True,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
