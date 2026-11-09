#!/usr/bin/env python3
"""MT1b verifier: Track contract, versioned registry, pack/closure identity, r5 replay, manifest.

CPU/protocol only.  It never imports ``tilelang``, never runs a Metal build, a GPU kernel, a model
or a paid search, and it never edits a closed package.  ``--check`` additionally runs the fixed
node sets (MT1b's own nodes, the compatibility regression, MT1a's fixed nodes and MT5's fixed
nodes) with exact pass counts and no skips/xfails.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
NODES = json.loads((HERE / "nodes.json").read_text(encoding="utf-8"))
R5 = ROOT / "docs/架构/多DSL-Track-阶段验收/MT0-tilelang-probe"


def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _sha(relative: str | Path) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _legacy_reconstruction() -> dict:
    """@brief 重建 MT1a/P0 的 schema-1 registry 身份（replay-only 路径）。"""
    from tcenv.identity import fingerprint
    from tcenv.track.registry import load_track_registry, registry_from_document

    path = HERE / "evidence/tracks-v1-epoch.json"
    frozen = registry_from_document(json.loads(path.read_text(encoding="utf-8")))
    try:
        load_track_registry(path)
    except Exception as exc:  # noqa: BLE001 - the typed refusal is the expected outcome
        refusal = str(exc)
    else:  # pragma: no cover - a live loader that accepts schema 1 would be the bug
        raise ValueError("a schema-1 registry was accepted as a live (new-run) registry")
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "document_digest": fingerprint(frozen.document()),
        "msl_track_digests": {track.track_id: fingerprint(track.document()) for track in frozen.tracks},
        "live_loader_refusal_contains_replay_only": "frozen replay inputs only" in refusal,
    }


def _build_probes() -> dict:
    """@brief 用当前代码真实执行 build 身份的五个不变量（不是读文档）。"""
    from tcenv.track import CandidateArtifact, CandidateEntry
    from tcenv.track import tilelang as contract

    source = (R5 / "evidence/failure-inputs/r5-candidate.py").read_bytes()

    def candidate(grid: int) -> CandidateArtifact:
        return CandidateArtifact(
            track_id=contract.TILELANG_TRACK_ID,
            entries=(
                CandidateEntry(
                    path=contract.GEOMETRY_PATH, role=contract.GEOMETRY_ROLE,
                    media_type=contract.GEOMETRY_MEDIA_TYPE,
                    data=json.dumps({
                        "schema_version": 1, "source": "candidate",
                        "stages": [{"kernel": "scale_kernel", "grid": [grid, 1, 1], "threads": [256, 1, 1]}],
                    }).encode(),
                ),
                CandidateEntry(
                    path=contract.SOURCE_PATH, role=contract.SOURCE_ROLE,
                    media_type=contract.SOURCE_MEDIA_TYPE, data=source,
                ),
            ),
        )

    environment = {
        "python_version": "3.13.1", "python_implementation": "CPython",
        "platform": "macOS-26.7.1-arm64-arm-64bit-Mach-O", "machine": "arm64",
    }
    first = contract.expected_build_subject(candidate=candidate(1), environment=environment)
    second = contract.expected_build_subject(candidate=candidate(999), environment=environment)
    again = contract.expected_build_subject(candidate=candidate(1), environment=environment)

    pinned = {
        "frontend_id": contract.FRONTEND_ID, "frontend_version": contract.FRONTEND_VERSION,
        "entry": contract.FRONTEND_ENTRY, "options_sha256": contract.options_sha256(),
        "package_manifest_sha256": contract.FRONTEND_PACKAGE_MANIFEST_SHA256,
        "dist_info_metadata_sha256": contract.FRONTEND_DIST_INFO_METADATA_SHA256,
        "dist_info_record_sha256": contract.FRONTEND_DIST_INFO_RECORD_SHA256,
        "package_files": contract.FRONTEND_PACKAGE_FILES, "package_bytes": contract.FRONTEND_PACKAGE_BYTES,
    }
    try:
        contract.verify_frontend_observation({**pinned, "package_manifest_sha256": "d" * 64})
        format_valid_drift_rejected = False
    except contract.TileLangIdentityDrift:
        format_valid_drift_rejected = True

    archived = {
        path: (R5 / "evidence/source-only-r5" / path).read_bytes()
        for path in contract.SOURCE_ONLY_ARTIFACT_PATHS
    }
    identity_document = json.loads(archived["frontend_identity.json"])
    identity_document["schema_id"] = "tcenv.track.tilelang.frontend-identity.1"
    identity_document["candidate"] = {
        "source_sha256": hashlib.sha256(source).hexdigest(), "prim_func_name": "scale_kernel",
    }
    production_format = {
        **archived, "frontend_identity.json": json.dumps(identity_document, sort_keys=True).encode()
    }
    attempt = {
        "status": "completed", "returncode": 0, "timed_out": False, "online": False, "hosts": [],
        "route": contract.ISOLATION_ROUTE, "request_sha256": "a" * 64, "hmz_version": "0.2.2",
        "policy_sha256": "b" * 64,
    }
    contract.source_only_build_identity(
        candidate=candidate(1), observed_frontend=pinned, environment=environment,
        artifacts=production_format, attempt=attempt, expected_kernel_names=("scale_kernel",),
    )
    try:
        contract.source_only_build_identity(
            candidate=candidate(1), observed_frontend=pinned, environment=environment,
            artifacts=archived, attempt=attempt,
        )
        self_reported_payload_rejected = False
    except contract.TileLangContractError:
        self_reported_payload_rejected = True
    try:
        contract.source_only_build_identity(
            candidate=candidate(1), observed_frontend=pinned, environment=environment,
            artifacts=production_format, attempt={**attempt, "timed_out": True},
        )
        timed_out_attempt_rejected = False
    except contract.TileLangCodegenError:
        timed_out_attempt_rejected = True

    from tcenv.backends.metal import tilelang_build

    return {
        "same_source_different_geometry_is_distinct": first.cache_key != second.cache_key,
        "same_inputs_reproduce": first.cache_key == again.cache_key and first.digest == again.digest,
        "format_valid_frontend_drift_rejected": format_valid_drift_rejected,
        "self_reported_or_incomplete_payload_rejected": self_reported_payload_rejected,
        "timed_out_attempt_rejected": timed_out_attempt_rejected,
        "frontend_entry_is_harness_owned": (
            tilelang_build.FRONTEND_ENTRY_SHA256
            == hashlib.sha256(tilelang_build.FRONTEND_ENTRY_SOURCE.encode("utf-8")).hexdigest()
        ),
    }


def live_identity() -> dict:
    """@brief 重新计算 MT1b 的可复算身份（不读 identity.json 的任何结论字段）。"""
    from taic.trust import current_prompt_sha1, prompt_fingerprint_parts
    from tcenv.identity import fingerprint
    from tcenv.taskcontract.locator import discover_standard_tasks
    from tcenv.taskcontract.pack import code_import_closure
    from tcenv.track import tilelang as contract
    from tcenv.track.registry import load_track_registry

    registry = load_track_registry()
    closure = code_import_closure(ROOT)
    packs = {locator.task_id: locator.deployment_pack_digest for locator in discover_standard_tasks(ROOT / "tasks")}
    artifacts = {
        path: (R5 / "evidence/source-only-r5" / path).read_bytes()
        for path in contract.SOURCE_ONLY_ARTIFACT_PATHS
    }
    replay = contract.replay_source_only(
        result=json.loads((R5 / "probe-result-r5.json").read_text(encoding="utf-8")),
        artifacts=artifacts,
        candidate_source=(R5 / "evidence/failure-inputs/r5-candidate.py").read_bytes(),
    )
    from tcenv.backends.metal import tilelang_build

    labels = {label for label, _ in prompt_fingerprint_parts()}
    changed = set(_load("identity.json")["changed_files"])
    before = json.loads((HERE / "evidence/closure-before.json").read_text(encoding="utf-8"))["paths"]
    return {
        "track": {
            "track_id": contract.TILELANG_TRACK_ID,
            "track_version": contract.TILELANG_TRACK_VERSION,
            "delivery": contract.TILELANG_DELIVERY,
            "status": registry.get(contract.TILELANG_TRACK_ID).status,
            "contract_digest": fingerprint(contract.frozen_contract()),
            "source": contract.source_spec(),
            "frontend": contract.frontend_identity(),
            "implemented_build_modes": list(contract.IMPLEMENTED_BUILD_MODES),
            "metal_build_implemented": contract.METAL_BUILD_MODE in contract.IMPLEMENTED_BUILD_MODES,
            "selectable": registry.get(contract.TILELANG_TRACK_ID).selectable,
            "registration_audit": contract.audit_registration(registry.get(contract.TILELANG_TRACK_ID)),
        },
        "registry": {
            "schema_version": registry.schema_version,
            "path_sha256": _sha("backends/metal/tracks.json"),
            "document_digest": fingerprint(registry.document()),
            "track_digests": {track.track_id: fingerprint(track.document()) for track in registry.tracks},
            "available": [track.track_id for track in registry.available()],
            "resume_choices": [track.track_id for track in registry.tracks],
            "schema_v2_path_sha256": _sha("tcenv/track/schema/tracks.v2.schema.json"),
            "legacy_epoch": _legacy_reconstruction(),
        },
        "build": {
            "frontend_install": {
                "version": contract.FRONTEND_VERSION,
                "package_manifest_sha256": contract.FRONTEND_PACKAGE_MANIFEST_SHA256,
                "dist_info_metadata_sha256": contract.FRONTEND_DIST_INFO_METADATA_SHA256,
                "dist_info_record_sha256": contract.FRONTEND_DIST_INFO_RECORD_SHA256,
                "package_files": contract.FRONTEND_PACKAGE_FILES,
                "package_bytes": contract.FRONTEND_PACKAGE_BYTES,
                "manifest_rule": contract.FRONTEND_MANIFEST_RULE,
            },
            "frontend_entry_sha256": tilelang_build.FRONTEND_ENTRY_SHA256,
            "probes": _build_probes(),
        },
        "pack": {
            "closure_size": len(closure),
            "closure_added": sorted(set(closure) - set(before)),
            "closure_removed": sorted(set(before) - set(closure)),
            "task_count": len(packs),
            "task_pack_digests": dict(sorted(packs.items())),
        },
        "prompt_surface": {
            "current_prompt_sha1": current_prompt_sha1(),
            "changed_files_that_are_prompt_covered": sorted(changed & labels),
        },
        "archived_r5_replay": {
            "artifact_set_sha256": replay["artifact_set_sha256"],
            "candidate_sha256": replay["candidate_sha256"],
            "artifact_paths": [row["path"] for row in replay["artifacts"]],
            "device_dispatch_calls": replay["device_dispatch_calls"],
            "gpu_npu_executed": replay["gpu_npu_executed"],
            "metal_correctness_or_performance_claim": replay["metal_correctness_or_performance_claim"],
            "production_ready": replay["production_ready"],
            "publication_allowed": replay["publication_allowed"],
        },
    }


def verify_identity(recorded: dict, live: dict) -> None:
    """@brief 记录身份与实时重算身份必须逐项一致。

    @param recorded identity.json。
    @param live live_identity()。
    @raises ValueError 任一字段漂移。
    """
    for relative, digest in recorded.get("code_files", {}).items():
        if _sha(relative) != digest:
            raise ValueError(f"an MT1b-pinned code/contract file changed: {relative}")

    track, live_track = recorded["track"], live["track"]
    for name in ("track_id", "track_version", "delivery", "status", "contract_digest", "source", "frontend",
                 "implemented_build_modes", "metal_build_implemented", "selectable"):
        if track[name] != live_track[name]:
            raise ValueError(f"track identity drifted: {name}")
    if live_track["registration_audit"]["contract_digest"] != track["contract_digest"]:
        raise ValueError("the registered Track does not audit against the frozen contract")
    if live_track["status"] != "interface_only" or live_track["selectable"] is not False:
        raise ValueError("the TileLang Track must stay interface_only and unselectable")
    if live_track["metal_build_implemented"] is not False:
        raise ValueError("this epoch must not claim an implemented Metal build")

    registry, live_registry = recorded["registry"], live["registry"]
    for name in ("schema_version", "path_sha256", "document_digest", "track_digests", "available",
                 "resume_choices", "schema_v2_path_sha256"):
        if registry[name] != live_registry[name]:
            raise ValueError(f"registry identity drifted: {name}")
    if live_registry["available"] != ["msl-single", "msl-bundle"]:
        raise ValueError("only the two MSL Tracks may be selectable")
    if live_registry["resume_choices"] != ["msl-single", "msl-bundle", "tilelang-pipeline"]:
        raise ValueError("resume must parse every registered id")
    legacy, live_legacy = registry["legacy_epoch"], live_registry["legacy_epoch"]
    for name in ("sha256", "document_digest", "msl_track_digests"):
        if legacy[name] != live_legacy[name]:
            raise ValueError(f"frozen schema-1 epoch drifted: {name}")
    if live_legacy["live_loader_refusal_contains_replay_only"] is not True:
        raise ValueError("a schema-1 document was not refused as replay-only")

    build, live_build = recorded["build"], live["build"]
    if live_build["frontend_install"] != build["frontend_install"]:
        raise ValueError("the pinned frontend install identity drifted")
    if live_build["frontend_entry_sha256"] != build["frontend_entry_sha256"]:
        raise ValueError("the frozen in-fence frontend entry drifted")
    for name, value in live_build["probes"].items():
        if value is not True:
            raise ValueError(f"build identity invariant {name!r} is false: {value!r}")
    if live_build["probes"] != build["probes"]:
        raise ValueError("a recorded build identity invariant drifted")

    pack, live_pack = recorded["pack"], live["pack"]
    if live_pack["closure_size"] != pack["closure_size_after"]:
        raise ValueError(f"task pack closure drifted: {live_pack['closure_size']}")
    if live_pack["closure_size"] != 74:
        raise ValueError(f"MT1b's closure is 74 files, got {live_pack['closure_size']}")
    if live_pack["closure_added"] != pack["closure_added"] or live_pack["closure_removed"]:
        raise ValueError("task pack closure membership drifted")
    if live_pack["task_count"] != pack["tasks_after"]:
        raise ValueError("task count drifted")
    if live_pack["task_pack_digests"] != pack["task_pack_digests_after"]:
        raise ValueError("a task deployment pack digest is not the recorded MT1b pack")
    for task, digest in pack["task_pack_digests_before"].items():
        if live_pack["task_pack_digests"][task] == digest:
            raise ValueError(f"{task}: the pack digest did not move although the closure gained schema 2")

    prompt, live_prompt = recorded["prompt_surface"], live["prompt_surface"]
    if live_prompt["current_prompt_sha1"] != prompt["current_prompt_sha1"]:
        raise ValueError("the prompt/TASK source surface drifted")
    covered = sorted(set(recorded.get("changed_files", ())) & set(prompt["prompt_covered_labels"]))
    if covered:
        raise ValueError(f"a prompt-covered file is claimed as an MT1b edit: {covered}")
    if live_prompt["changed_files_that_are_prompt_covered"]:
        raise ValueError(
            f"a prompt-covered file was edited by MT1b: {live_prompt['changed_files_that_are_prompt_covered']}"
        )

    claimed_b = _b_evidence_paths(recorded)
    if claimed_b:
        raise ValueError(f"an MT1b edit may not claim archived B-series evidence: {claimed_b}")

    replay, live_replay = recorded["archived_r5_replay"], live["archived_r5_replay"]
    if replay != live_replay:
        raise ValueError("the archived r5 source-only replay identity drifted")
    if live_replay["publication_allowed"] is not False or live_replay["production_ready"] is not False:
        raise ValueError("a source-only replay must never claim production availability or publication")


def self_test(recorded: dict, live: dict) -> None:
    """@brief 10 个文档级负变异必须全部被拒（不允许“改数字也通过”）。

    @param recorded identity.json。
    @param live live_identity()。
    @raises AssertionError 有一个变异被接受。
    """
    mutations = []

    changed = copy.deepcopy(recorded)
    changed["registry"]["document_digest"] = "0" * 64
    mutations.append(("registry document digest", changed))

    changed = copy.deepcopy(recorded)
    changed["track"]["contract_digest"] = "0" * 64
    mutations.append(("track contract digest", changed))

    changed = copy.deepcopy(recorded)
    changed["pack"]["closure_size_after"] = 73
    mutations.append(("pack closure size", changed))

    changed = copy.deepcopy(recorded)
    changed["registry"]["legacy_epoch"]["sha256"] = "0" * 64
    mutations.append(("frozen schema-1 epoch bytes", changed))

    changed = copy.deepcopy(recorded)
    changed["archived_r5_replay"]["artifact_set_sha256"] = "0" * 64
    mutations.append(("archived r5 artifact-set digest", changed))

    changed = copy.deepcopy(recorded)
    changed["build"]["frontend_install"]["package_manifest_sha256"] = "d" * 64
    mutations.append(("a well-formed but different frontend digest", changed))

    changed = copy.deepcopy(recorded)
    changed["build"]["frontend_entry_sha256"] = "0" * 64
    mutations.append(("the frozen in-fence entry digest", changed))

    changed = copy.deepcopy(recorded)
    changed["build"]["probes"]["same_source_different_geometry_is_distinct"] = False
    mutations.append(("geometry-blind build identity", changed))

    changed = copy.deepcopy(recorded)
    changed["changed_files"] = list(changed["changed_files"]) + [
        "backends/metal/skills/msl_reference.md"  # a prompt-covered text file must never be an MT1b edit
    ]
    mutations.append(("a prompt-covered file claimed as edited", changed))

    changed = copy.deepcopy(recorded)
    changed["changed_files"] = list(changed["changed_files"]) + [
        "docs/架构/跨硬件算子生成分工-阶段验收/B1/evidence/gpu-gemv.json"  # someone else's Metal output
    ]
    mutations.append(("archived B-series evidence claimed as an MT1b edit", changed))

    for name, document in mutations:
        try:
            verify_identity(document, live)
        except ValueError:
            continue
        raise AssertionError(f"MT1b verifier self-test mutation {name!r} was accepted")


def verify_manifest() -> None:
    """@brief 包内文件必须与 manifest.sha256 逐字节一致。"""
    for line in (HERE / "manifest.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = HERE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"manifest mismatch: {relative}")


def _b_evidence_paths(recorded: dict) -> list[str]:
    """@brief ``changed_files`` 里被声称为 MT1b 改动的 B 系列归档证据。

    @param recorded identity.json。
    @return 命中的路径（空列表表示边界正确）。
    """
    return [
        path
        for path in recorded.get("changed_files", ())
        if "/跨硬件算子生成分工-阶段验收/B" in path
    ]


def verify_external_b_evidence(recorded: dict, *, resolution_document: dict | None = None) -> None:
    """@brief 验证 B 漂移原记录、独立新观察与已恢复的封闭归档同时成立。

    ``b-evidence-drift.json`` 保留原漂移事实，``b-evidence-resolution.json`` 再指向一个独立
    处置包。当前 B 文件必须等于 HEAD（已恢复），raw 副本必须等于漂移时的新观察（未丢失），
    原 incident record 的 SHA 不得变化。这样既不允许悄悄覆盖历史，也不允许删除新观察。

    @param recorded identity.json。
    @param resolution_document 测试可注入的处置文档；生产从 b-evidence-resolution.json 读取。
    @raises ValueError 记录过期、raw/归档不一致，或 MT1b 冒认 B 系列证据。
    """
    claimed = _b_evidence_paths(recorded)
    if claimed:
        raise ValueError(f"an MT1b edit may not claim archived B-series evidence: {claimed}")
    document = _load("b-evidence-drift.json")
    resolution = resolution_document or _load("b-evidence-resolution.json")
    if resolution.get("status") != "closed_evidence_restored_and_raw_observations_preserved":
        raise ValueError("the B-evidence resolution must restore closed evidence and preserve raw observations")
    source_record = ROOT / resolution["source_incident_record"]
    if _sha(source_record) != resolution["source_incident_record_sha256"]:
        raise ValueError("the original B-evidence incident record was modified during resolution")
    incident = json.loads((ROOT / resolution["incident"]).read_text(encoding="utf-8"))
    if incident.get("closed_evidence_restored") is not True or incident.get("raw_observations_preserved") is not True:
        raise ValueError("the B-evidence incident has no completed restore/preserve resolution")
    if incident.get("source_incident_record_modified") is not False:
        raise ValueError("the incident resolution modified its source record")
    incident_rows = {row["path"]: row for row in incident.get("files", [])}
    files = document.get("files") or []
    if document.get("count") != len(files) or len(files) != 9 or set(incident_rows) != {row["path"] for row in files}:
        raise ValueError("the B-evidence incident and resolution must cover exactly the same nine files")
    for row in files:
        relative = row["path"]
        resolved = incident_rows[relative]
        raw_path = ROOT / resolved["raw_copy"]
        if not raw_path.is_file() or raw_path.is_symlink() or raw_path.stat().st_nlink != 1:
            raise ValueError(f"{relative}: the preserved raw observation is missing or not an independent regular file")
        if _sha(raw_path) != row["current_sha256"] or _sha(raw_path) != resolved["observed_sha256"]:
            raise ValueError(f"{relative}: the preserved raw observation bytes differ from the incident")
        if _sha(relative) != row["head_sha256"] or _sha(relative) != resolved["restored_sha256"]:
            raise ValueError(f"{relative}: the closed B-evidence file was not restored to its HEAD bytes")
        if list(row["fields_that_differ_from_head"]) != ["median_us"]:
            raise ValueError(f"{relative}: median_us must be the only field that differs from HEAD")
        live_document = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for name, value in row["identity_fields_identical_to_head"].items():
            if live_document.get(name) != value:
                raise ValueError(f"{relative}: recorded identity field {name!r} no longer matches the file")
        if live_document.get("median_us") != row["median_us"]["head"]:
            raise ValueError(f"{relative}: the restored median_us is not the original HEAD value")


def _pytest(nodes: tuple[str, ...], expected: int, *, markers: bool) -> None:
    command = [sys.executable, "-m", "pytest", "-q", "-n", "0", "--tb=short"]
    if markers:
        command.extend(["-m", "not gpu and not llm and not slow"])
    command.extend(nodes)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output[-4000:], end="")
        raise ValueError(f"pytest failed ({completed.returncode}); read logs/pytest/triage/triage_summary.md")
    match = re.search(r"(\d+) passed", output)
    if match is None or int(match.group(1)) != expected:
        raise ValueError(f"unexpected pytest count for {nodes}: expected {expected}, output was {output[-1200:]}")
    if re.search(r"[1-9]\d* (?:skipped|xfailed)", output):
        raise ValueError(f"skipped/xfailed nodes are not acceptable: {output[-1200:]}")


def _prior_nodes() -> tuple[tuple[str, ...], int]:
    mt1a = json.loads((ROOT / "docs/架构/多DSL-Track-阶段验收/MT1a/nodes.json").read_text(encoding="utf-8"))
    mt5 = json.loads((ROOT / "docs/架构/多DSL-Track-阶段验收/MT5/nodes.json").read_text(encoding="utf-8"))
    regression = json.loads((ROOT / "docs/架构/多DSL-Track-阶段验收/MT1a/nodes.json").read_text(encoding="utf-8"))
    nodes = tuple(mt1a["new"] + mt1a["negative"] + mt5["positive"] + mt5["negative"] + regression["regression_files"])
    return nodes, NODES["prior_package_nodes"]["MT1a_new"]["count"] + NODES["prior_package_nodes"]["MT1a_regression"]["count"] + NODES["prior_package_nodes"]["MT5_fixed"]["count"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--check", action="store_true", help="also verify the manifest and run the fixed node sets")
    arguments = parser.parse_args(argv)
    recorded = _load("identity.json")
    live = live_identity()
    verify_identity(recorded, live)
    verify_external_b_evidence(recorded)
    self_test(recorded, live)
    result = {
        "done": True,
        "stage": "MT1b",
        "scope": "TileLang Track contract, versioned registry, build identity and adapter seams; CPU/protocol only",
        "track": f"{live['track']['track_id']}/{live['track']['track_version']}",
        "status": live["track"]["status"],
        "selectable": live["track"]["selectable"],
        "metal_build_implemented": live["track"]["metal_build_implemented"],
        "registry_schema_version": live["registry"]["schema_version"],
        "task_pack_closure": live["pack"]["closure_size"],
        "build_identity_probes": len(live["build"]["probes"]),
        "task_pack_digests_moved": live["pack"]["task_count"],
        "archived_r5_artifact_set_sha256": live["archived_r5_replay"]["artifact_set_sha256"],
        "external_b_evidence_drift_files": len(_load("b-evidence-drift.json")["files"]),
        "production_ready": False,
        "publication_allowed": False,
        "real_metal_gpu_npu_model_paid_search": False,
        "manifest": arguments.check,
        "fixed_nodes": None if not arguments.check else {
            "mt1b_new": NODES["positive_count"],
            "compatibility_regression": NODES["regression_passed"],
            "mt1a_and_mt5_fixed": NODES["prior_package_nodes"]["MT1a_new"]["count"]
            + NODES["prior_package_nodes"]["MT1a_regression"]["count"]
            + NODES["prior_package_nodes"]["MT5_fixed"]["count"],
        },
    }
    if arguments.check:
        verify_manifest()
        _pytest(tuple(NODES["positive"]), NODES["positive_count"], markers=False)
        _pytest(tuple(NODES["regression_files"]), NODES["regression_passed"], markers=True)
        nodes, expected = _prior_nodes()
        _pytest(nodes, expected, markers=True)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
