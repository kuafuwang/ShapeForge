#!/usr/bin/env python3
"""A3 acceptance verifier: registries, discovery, delivery evidence and recovery.

Default run:

* checks ``manifest.sha256`` over the complete A3 deliverable set;
* proves the frozen contract surface (six RPC/field numbers/HMAC/BlobChannel, schema 4, v1 task
  surface, targets, G0/A1/A2 evidence) is untouched since the A2 acceptance head;
* substantively checks the intentional A3 changes to run_config/evaluation/agent-transport
  (schema 3 identity fields, envelope pass-through, v1 self-check refusal);
* scans the A3 production surface for rejected platform names and per-op tiling;
* checks every stable negative node id still exists and runs the fixed A3 + A2 + A1 regression set;
* replays independent negatives for candidate identity, the envelope, pack closure, discovery and
  staged rebuild, track refusal, run_config schema 3 and the v1 self-check.

``--write-manifest`` regenerates ``manifest.sha256`` after an intentional A3 edit.  The verifier
never calls a model, never builds a kernel and never touches a GPU/NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = Path(__file__).resolve().parent / "manifest.sha256"

DELIVERABLES: tuple[str, ...] = (
    # A3 transport and delivery
    "tcenv/track/plane.py",
    "tcenv/track/__init__.py",
    "tcenv/track/delivery.py",
    "tcenv/track/envelope.py",
    "tcenv/evaluation.py",
    "tcenv/evaluate.py",
    "tcenv/remote/execution_geometry.py",
    # A3 registries, discovery and evaluate_task
    "tcenv/taskcontract/__init__.py",
    "tcenv/taskcontract/baselines.py",
    "tcenv/taskcontract/locator.py",
    "tcenv/taskcontract/pack.py",
    "tcenv/taskcontract/runner.py",
    # A3 run_config/resume and the agent transport chain
    "taic/artifacts.py",
    "taic/orchestrator.py",
    "taic/resume.py",
    "taic/parse.py",
    "taic/kernel_agent.py",
    # tests
    "tests/shared/test_track_delivery.py",
    "tests/shared/test_track_envelope.py",
    "tests/shared/test_geometry_artifact.py",
    "tests/shared/test_task_discovery.py",
    "tests/shared/test_evaluate_task.py",
    "tests/shared/test_run_config.py",
    "tests/shared/test_report_schema4.py",
    # acceptance record
    "docs/架构/跨硬件算子生成分工-阶段验收/A3.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A3/verify.py",
)

FIXED_NODE_IDS: tuple[str, ...] = (
    "tests/shared/test_track_delivery.py",
    "tests/shared/test_track_envelope.py",
    "tests/shared/test_geometry_artifact.py",
    "tests/shared/test_task_discovery.py",
    "tests/shared/test_evaluate_task.py",
    "tests/shared/test_run_config.py",
    "tests/shared/test_report_schema4.py",
    "tests/shared/test_track_plane.py",
    "tests/shared/test_task_contract_v2.py",
    "tests/backends/metal/test_track_registry.py",
    "tests/shared/test_cross_hardware_boundary.py",
    # A2/A1 regressions that the A3 wiring must keep green
    "tests/shared/test_track_abi.py",
    "tests/shared/test_track_geometry.py",
    "tests/shared/test_track_protocol.py",
    "tests/shared/test_track_bundle.py",
    "tests/shared/test_track_lifecycle.py",
    "tests/backends/metal/test_stage_binding.py",
    "tests/shared/test_v1_identity_g0.py",
    "tests/shared/test_v2_mapping.py",
    "tests/shared/test_v2_g0_reference_outputs.py",
    "tests/shared/test_op_desc_summary.py",
)

NEGATIVE_NODE_IDS: tuple[str, ...] = (
    "tests/shared/test_track_delivery.py::test_the_candidate_geometry_document_is_closed_and_refuses_harness_shortcuts",
    "tests/shared/test_track_delivery.py::test_a_bundle_owns_exactly_one_geometry_entry",
    "tests/shared/test_track_delivery.py::test_the_bundle_identity_is_the_candidate_manifest_digest_not_the_envelope_hash",
    "tests/shared/test_track_envelope.py::test_a_corrupted_missing_or_extra_segment_is_refused",
    "tests/shared/test_track_envelope.py::test_the_v1_self_check_refuses_a_segmented_source_before_writing_one_file",
    "tests/shared/test_track_envelope.py::test_evaluation_request_uses_the_manifest_identity_on_default_and_from_dict",
    "tests/shared/test_track_envelope.py::test_a_tampered_taic_file_header_sha_is_refused",
    "tests/shared/test_track_envelope.py::test_the_decoder_enforces_the_same_segment_limit_as_the_encoder",
    "tests/shared/test_geometry_artifact.py::test_attaching_a_geometry_from_another_candidate_or_twice_is_refused",
    "tests/shared/test_geometry_artifact.py::test_a_tampered_descriptor_or_geometry_entry_fails_cross_audit",
    "tests/shared/test_task_discovery.py::test_discovery_refuses_a_second_task_for_one_op_with_a_different_pack",
    "tests/shared/test_task_discovery.py::test_a_staged_pack_rebuilds_on_a_worker_without_the_control_host_path",
    "tests/shared/test_task_discovery.py::test_the_baseline_registry_has_one_authority_per_operator_and_name",
    "tests/shared/test_task_discovery.py::test_staged_pack_refuses_absolute_escape_paths_before_writing_anything",
    "tests/shared/test_task_discovery.py::test_load_refuses_a_task_file_or_code_closure_that_drifted_after_discovery",
    "tests/shared/test_task_discovery.py::test_the_pack_closure_runs_in_a_worker_without_the_control_host_repo",
    "tests/shared/test_task_discovery.py::test_a_runtime_schema_change_moves_the_deployment_pack_digest",
    "tests/shared/test_task_discovery.py::test_a_symlinked_runtime_schema_is_refused_by_closure_discovery_and_load",
    "tests/shared/test_task_discovery.py::test_a_symlinked_schema_directory_is_refused",
    "tests/shared/test_task_discovery.py::test_a_nested_symlinked_schema_directory_is_refused",
    "tests/shared/test_evaluate_task.py::test_track_selection_refuses_missing_unknown_and_interface_only_tracks",
    "tests/shared/test_evaluate_task.py::test_evaluate_task_refuses_a_bundle_without_a_lifecycle_or_a_drifting_report",
    "tests/shared/test_evaluate_task.py::test_a_single_source_candidate_uses_the_injected_launch_parser",
    "tests/shared/test_evaluate_task.py::test_evaluate_task_decodes_a_single_submission_envelope_without_splitting_segments",
    "tests/shared/test_evaluate_task.py::test_evaluate_task_refuses_a_non_schema4_report[missing_version]",
    "tests/shared/test_evaluate_task.py::test_evaluate_task_refuses_a_task_that_drifted_after_discovery",
    "tests/shared/test_run_config.py::test_schema3_identity_accepts_v1_and_v2_and_refuses_a_half_filled_pair",
    "tests/shared/test_task_contract_v2.py::test_a_workspace_tensor_enters_the_a2_abi_and_stays_out_of_the_reference_contract",
    "tests/shared/test_track_geometry.py::test_the_fallback_is_independent_and_enters_the_execution_identity",
    "tests/backends/metal/test_track_registry.py::test_the_metal_registry_document_is_closed_and_op_agnostic",
)

#: The checkout A3 started from (A2's committed acceptance head).
A3_BASE_HEAD = "4a93ed2a4934fe526eedfcf4577de5dbd366e24a"

#: Contract surfaces A3 must not change.  A3 intentionally advances three A2-plane files
#: (plane/envelope/evaluation) and the taic run_config path; those are checked substantively
#: below instead of by raw diff.
FROZEN_SURFACE: tuple[str, ...] = (
    "proto",
    "tcenv/remote/v1",
    "tcenv/remote/artifact_bundle.py",
    "tcenv/remote/contracts.py",
    "tcenv/remote/grpc_client.py",
    "tcenv/remote/grpc_contract.py",
    "tcenv/remote/grpc_service.py",
    "tcenv/remote/task_pack.py",
    "tcenv/task.py",
    "tcenv/contract.py",
    "tcenv/build_projection.py",
    "tcenv/report.py",
    "tcenv/schema.py",
    "tcenv/verify.py",
    "tcenv/backends/metal/lowering.py",
    "tcenv/track/schema/track.v1.schema.json",
    "tcenv/track/schema/tracks.v1.schema.json",
    "tcenv/opspec/schema/opspec.v1.schema.json",
    "tasks",
    "targets",
    "backends/metal/backend.toml",
    "backends/metal/tracks.json",
    "tests/shared/test_v1_identity_g0.py",
    "tests/shared/test_v2_mapping.py",
    "tests/shared/test_v2_g0_reference_outputs.py",
    "tests/shared/test_op_desc_summary.py",
    "docs/架构/跨硬件算子生成分工-G0",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1",
    "docs/架构/跨硬件算子生成分工-阶段验收/A2.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A2",
)

REJECTED_PLATFORM = re.compile(
    r"\b(cuda|cann|ascend[a-z]*|msopgen|msprof|easyasc|acl|npu|torch_npu|america|nvidia)\b", re.IGNORECASE
)


def require(condition: bool, message: str) -> None:
    """@brief 断言一个验收条件。

    @param condition 条件。
    @param message 失败说明。
    @raises AssertionError 条件不成立。
    """
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    """@brief 文件字节 sha256。

    @param path 文件。
    @return 十六进制摘要。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    """@brief 运行 git 并返回 stdout。

    @param args git 参数。
    @return stdout。
    @raises AssertionError git 失败。
    """
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def write_manifest() -> None:
    """@brief 重写 A3 manifest.sha256。"""
    lines = [f"{sha256(ROOT / name)}  {name}" for name in DELIVERABLES]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK: wrote {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def verify_manifest() -> None:
    """@brief 校验 manifest 的摘要与文件集合完全一致。

    @raises AssertionError 缺失/多余/摘要不符。
    """
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    seen: dict[str, str] = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"bad digest: {name}")
        require(name not in seen, f"duplicate manifest entry: {name}")
        seen[name] = digest
    expected = set(DELIVERABLES)
    require(
        set(seen) == expected,
        f"manifest file set differs: missing={sorted(expected - set(seen))} extra={sorted(set(seen) - expected)}",
    )
    for name, digest in seen.items():
        require(sha256(ROOT / name) == digest, f"manifest mismatch: {name}")
    print(f"OK: manifest.sha256 ({len(seen)} files)")


def verify_frozen_surface() -> None:
    """@brief 冻结面自 A2 验收头以来没有任何改动。

    @raises AssertionError 冻结面出现 diff/未跟踪文件。
    """
    base = _git("rev-parse", "--verify", f"{A3_BASE_HEAD}^{{commit}}").strip()
    require(base == A3_BASE_HEAD, f"A3 base commit is not available: {A3_BASE_HEAD}")
    changed = _git("diff", "--name-only", A3_BASE_HEAD, "--", *FROZEN_SURFACE).splitlines()
    require(changed == [], "frozen surface changed since the A3 base commit: " + ", ".join(changed))
    dirty = _git(
        "status", "--porcelain", "--", "tasks", "targets", "proto", "backends/metal/backend.toml", "backends/metal/tracks.json"
    ).splitlines()
    require(dirty == [], "frozen task/target/protocol paths have untracked or modified files: " + ", ".join(dirty))
    for name in (
        "docs/架构/跨硬件算子生成分工-阶段验收/A1.md",
        "docs/架构/跨硬件算子生成分工-阶段验收/A2.md",
        "docs/架构/跨硬件算子生成分工-阶段验收/A2/verify.py",
    ):
        require((ROOT / name).is_file(), f"historical acceptance evidence is missing: {name}")
    print(f"OK: frozen surface untouched since {A3_BASE_HEAD[:12]} ({len(FROZEN_SURFACE)} paths)")


def verify_changed_surfaces() -> None:
    """@brief A3 对既有接点的改动是实质性的、可指认的，而不是静默改语义。

    @raises AssertionError 缺少 schema 3 字段、envelope 透传或自检拒绝。
    """
    sys.path.insert(0, str(ROOT))
    from taic.artifacts import RUN_CONFIG_SCHEMA_VERSION

    require(RUN_CONFIG_SCHEMA_VERSION == 3, "run_config schema is not 3")
    orchestrator = (ROOT / "taic/orchestrator.py").read_text(encoding="utf-8")
    for needle in ('"track": None', '"task_pack_digest": None', '"effective_correctness_only"'):
        require(needle in orchestrator, f"run_config schema 3 is missing {needle}")
    resume = (ROOT / "taic/resume.py").read_text(encoding="utf-8")
    require("validate_run_config_identity" in resume, "resume does not validate the schema-3 identity")
    evaluation = (ROOT / "tcenv/evaluation.py").read_text(encoding="utf-8")
    require("decode_envelope" in evaluation and "candidate_sha256" in evaluation, "evaluation lost envelope identity")
    parse = (ROOT / "taic/parse.py").read_text(encoding="utf-8")
    agent = (ROOT / "taic/kernel_agent.py").read_text(encoding="utf-8")
    require("is_candidate_envelope" in parse and "is_candidate_envelope" in agent, "the agent chain lost the envelope")
    evaluate = (ROOT / "tcenv/evaluate.py").read_text(encoding="utf-8")
    require("segmented_source_refusal" in evaluate, "the v1 self-check has no segmented-source refusal")
    plane = (ROOT / "tcenv/track/plane.py").read_text(encoding="utf-8")
    require("sha256(self.manifest_bytes)" in plane, "bundle candidate_sha256 is not the candidate-entry manifest digest")
    evaluation_source = (ROOT / "tcenv/evaluation.py").read_text(encoding="utf-8")
    require("_envelope_candidate_identity" in evaluation_source, "the default/from_dict envelope identity is not decoded")
    locator_source = (ROOT / "tcenv/taskcontract/locator.py").read_text(encoding="utf-8")
    require("_safe_relative_path" in locator_source, "staged paths are not validated")
    require("task_pack_closure(self.task_dir" in locator_source, "locator.load does not recompute the pack digest")
    pack_source = (ROOT / "tcenv/taskcontract/pack.py").read_text(encoding="utf-8")
    require('schema_dir = path.parent / "schema"' in pack_source, "the pack closure does not include runtime schemas")
    require(
        "resource.is_symlink()" in pack_source and "raise PackClosureError" in pack_source,
        "the pack closure silently omits a symlinked/non-regular runtime schema",
    )
    runner_source = (ROOT / "tcenv/taskcontract/runner.py").read_text(encoding="utf-8")
    require("EvalReport.from_json" in runner_source, "evaluate_task does not use the schema-4 reader")
    envelope_source = (ROOT / "tcenv/track/envelope.py").read_text(encoding="utf-8")
    require("header_sha" in envelope_source and "MAX_ENVELOPE_ENTRIES" in envelope_source, "the envelope header/segment limits are missing")
    print("OK: schema-3 run_config, resume gate, envelope identity/limits, staged paths, pack schemas and schema 4 are wired")


def _a3_production_sources() -> dict[str, str]:
    """@brief A3 生产面文本：新文件全文 + 既有文件相对 A3 起点的新增行。

    @return 文件相对路径到待扫描文本。
    """
    tracked = set(_git("ls-files", "--", *DELIVERABLES).splitlines())
    sources: dict[str, str] = {}
    for name in DELIVERABLES:
        if not name.startswith(("tcenv/", "taic/")):
            continue
        path = ROOT / name
        if path.suffix not in {".py", ".json"}:
            continue
        if name in tracked:
            diff = _git("diff", "--unified=0", A3_BASE_HEAD, "--", name)
            added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
            sources[name] = "\n".join(added)
        else:
            sources[name] = path.read_text(encoding="utf-8")
    return sources


def verify_boundary() -> None:
    """@brief A3 生产代码不携带被拒绝平台名或按 op/shape 的 tiling。

    @raises AssertionError 出现禁止词/字段。
    """
    offenders: list[str] = []
    for name, text in _a3_production_sources().items():
        for match in REJECTED_PLATFORM.finditer(text):
            offenders.append(f"{name}: {match.group(0)}")
    require(offenders == [], "\n".join(offenders))
    tracks = json.loads((ROOT / "backends/metal/tracks.json").read_text(encoding="utf-8"))
    for track in tracks["tracks"]:
        require(
            not {"op_name", "shape", "tile", "grid", "threads", "lowering_profile"} & set(track),
            f"track {track['track_id']} has per-op tiling metadata",
        )
    print("OK: no rejected platform or per-op tiling in the A3 production surface")


def verify_negative_nodes() -> None:
    """@brief 固定负例节点仍然存在（不被删除/改名）。

    @raises AssertionError 节点不存在。
    """
    for node in NEGATIVE_NODE_IDS:
        path, _, test = node.partition("::")
        test = test.split("[", 1)[0]
        source = (ROOT / path).read_text(encoding="utf-8")
        require(f"def {test}(" in source, f"negative node id is gone: {node}")
    print(f"OK: {len(NEGATIVE_NODE_IDS)} stable negative node ids present")


def _tracks_document() -> dict:
    """@brief 测试用的可用 track registry（schema 1 闭集）。

    @return 文档。
    """
    def block(name: str, delivery: str, roles: list[str], capabilities: list[str]) -> dict:
        return {
            "track_id": name,
            "track_version": 1,
            "delivery": delivery,
            "status": "available",
            "candidate_roles": roles,
            "capabilities": capabilities,
            "renderer": {"implementation_id": f"metal.track.{name}.renderer", "implementation_version": 1},
            "validator": {"implementation_id": f"metal.track.{name}.validator", "implementation_version": 1},
            "build_adapter": {"implementation_id": f"metal.track.{name}.build", "implementation_version": 1},
            "evidence": ["A3 verifier"],
        }

    return {
        "schema_version": 1,
        "backend_id": "metal",
        "tracks": [
            block("msl-single", "single_source", ["device_kernel"], ["single_file_msl"]),
            block(
                "msl-bundle",
                "candidate_bundle",
                ["device_kernel", "host_tiling", "schedule"],
                ["candidate_host_tiling", "candidate_device_kernel", "workspace", "multi_stage"],
            ),
        ],
    }


def self_test() -> None:
    """@brief 可独立重放的 A3 稳定负例（不依赖 pytest 收集）。

    @raises AssertionError 任一负例未被拒绝或正向行为不成立。
    """
    sys.path.insert(0, str(ROOT))

    import hashlib

    from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
    from tcenv.evaluate import segmented_source_refusal
    from tcenv.evaluation import EvaluationError, EvaluationRequest
    from tcenv.track import EnvelopeError
    from tcenv.taskcontract import (
        PackClosureError,
        StagedTaskPack,
        TaskDiscoveryError,
        discover_standard_tasks,
        task_pack_closure,
    )
    from tcenv.taskcontract.pack import code_import_closure
    from tcenv.taskcontract.runner import _normalize_report, default_geometry_resolver, select_track
    from tcenv.track import (
        CandidateArtifact,
        CandidateEntry,
        candidate_from_single_source,
        candidate_geometry,
        decode_envelope,
        encode_envelope,
        load_track_registry,
    )
    from tests.support.v2_tasks import write_task

    checks: list[str] = []

    def rejected(name: str, exc_type, action) -> None:
        try:
            action()
        except exc_type:
            checks.append(name)
            return
        except Exception as exc:  # noqa: BLE001 - report the wrong exception type
            raise AssertionError(f"{name}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
        raise AssertionError(f"{name}: expected {exc_type.__name__}, nothing was raised")

    def bundle(kernel: bytes = b"kernel void fill() {}\n") -> CandidateArtifact:
        geometry = json.dumps(
            {"schema_version": 1, "source": "candidate", "stages": [{"kernel": "fill", "grid": [2, 1, 1], "threads": [256, 1, 1]}]},
            sort_keys=True,
        ).encode()
        return CandidateArtifact(
            track_id="msl-bundle",
            entries=(
                CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=kernel),
                CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"// t\n"),
                CandidateEntry(path="host/execution_geometry.json", role="schedule", media_type="application/json", data=geometry),
            ),
        )

    candidate = bundle()
    require(
        candidate.candidate_sha256 == hashlib.sha256(candidate.manifest_bytes).hexdigest(),
        "bundle candidate_sha256 is not the candidate-entry manifest digest",
    )
    require(bundle(b"kernel void other() {}\n").candidate_sha256 != candidate.candidate_sha256, "a different kernel kept the identity")
    text = encode_envelope(candidate)
    decoded = decode_envelope(text)
    require(decoded.candidate_sha256 == candidate.candidate_sha256, "envelope decode changed the candidate identity")
    require(decoded.candidate_sha256 != hashlib.sha256(text.encode()).hexdigest(), "the envelope hash became the identity")
    require({entry.path for entry in decoded.entries} == {entry.path for entry in candidate.entries}, "envelope lost a segment")
    require({entry.path: entry.data for entry in decoded.entries} == {entry.path: entry.data for entry in candidate.entries}, "envelope changed a segment")
    single = candidate_from_single_source("kernel void fill() {}\n")
    require(single.candidate_sha256 == hashlib.sha256(b"kernel void fill() {}\n").hexdigest(), "single-source identity changed")
    require(candidate_geometry(decoded).source == "candidate", "the candidate geometry was not parsed")
    checks.append("candidate identity is entries-only; envelope is transport")

    require(segmented_source_refusal(text) is not None, "the v1 self-check accepted a segmented candidate")
    require(segmented_source_refusal("kernel void fill() {}\n") is None, "the v1 self-check refused a plain kernel")
    request = EvaluationRequest(task="rmsnorm", dtype="fp16", size="small", source=text, task_fingerprint="a" * 64)
    require(request.candidate_sha256 == candidate.candidate_sha256, "EvaluationRequest defaulted to the transport hash")
    rejected(
        "transport-hash identity",
        EvaluationError,
        lambda: EvaluationRequest(
            task="rmsnorm", dtype="fp16", size="small", source=text,
            candidate_sha256=hashlib.sha256(text.encode()).hexdigest(), task_fingerprint="a" * 64,
        ),
    )
    checks.append("the v1 self-check refuses an envelope before writing one file; EvaluationRequest cannot use its hash")

    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("taic-file "))
    fields = lines[index].split(" ")
    fields[-1] = "0" * 64
    lines[index] = " ".join(fields)
    rejected("tampered header sha", EnvelopeError, lambda: decode_envelope("\n".join(lines) + "\n"))
    rejected("non-schema4 report", TaskContractError, lambda: _normalize_report({"schema_version": 4}))

    code = code_import_closure(ROOT)
    require("tcenv/taskcontract/reference.py" in code, "the pack closure lost the reference module")
    require(
        "tcenv/opspec/schema/opspec.v2.schema.json" in code
        and "tcenv/taskcontract/schema/baselines.v1.schema.json" in code,
        "the pack closure lost a runtime schema resource",
    )
    require(not any("backends" in name or name.startswith(("taic", "tests", "tasks")) for name in code), "the pack closure crossed a forbidden root")
    checks.append("the static import closure is hardware-neutral and carries runtime schemas")
    checks.append("the envelope header sha, schema-4 reader and transport identity are enforced")

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "tasks"
        task_dir = write_task(root)
        legacy = root / "legacy"
        legacy.mkdir()
        (legacy / "opspec.json").write_text('{"spec_version": 1}', encoding="utf-8")
        (legacy / "track.json").write_text("{}", encoding="utf-8")
        found = discover_standard_tasks(root, code_root=ROOT)
        require([locator.task_id for locator in found] == ["rmsnorm"], f"discovery found {[locator.task_id for locator in found]}")
        locator = found[0]
        require("task_dir" not in locator.document() and "pack_digest" not in locator.document(), "a host path/root-dependent digest entered the identity")
        first = task_pack_closure(task_dir, root=ROOT)
        (task_dir / "guidance_catalog.md").write_text("wiki\n", encoding="utf-8")
        require(task_pack_closure(task_dir, root=ROOT).digest == first.digest, "knowledge entered the task pack digest")
        staged = locator.stage()
        rebuilt = staged.rebuild(base / "worker", code_root=ROOT)
        require(rebuilt.identity == locator.identity, "the staged task did not rebuild to the same fingerprint")
        tampered = tuple((name, data + b"\n# drift\n") if name == "reference.py" else (name, data) for name, data in staged.files)
        rejected(
            "tampered staged task",
            TaskDiscoveryError,
            lambda: StagedTaskPack(locator=staged.locator, files=tampered).rebuild(base / "worker2", code_root=ROOT),
        )
        outside = base / "outside.txt"
        rejected(
            "absolute staged path",
            TaskDiscoveryError,
            lambda: StagedTaskPack(locator=staged.locator, files=staged.files + ((str(outside), b"escape"),)),
        )
        require(not outside.exists(), "an absolute staged path produced a file outside the worker root")
        fake_code = base / "fake-code"
        (fake_code / "tcenv" / "taskcontract").mkdir(parents=True)
        (fake_code / "tcenv" / "opspec" / "schema").mkdir(parents=True)
        (fake_code / "tcenv" / "taskcontract" / "__init__.py").write_text("", encoding="utf-8")
        (fake_code / "tcenv" / "opspec" / "__init__.py").write_text("", encoding="utf-8")
        fake_schema = fake_code / "tcenv" / "opspec" / "schema" / "opspec.v2.schema.json"
        fake_schema.write_text('{"type": "object"}\n', encoding="utf-8")
        fake_locator = discover_standard_tasks(root, code_root=fake_code)[0]
        external_schema = base / "external-schema.json"
        external_schema.write_text('{"type": "object"}\n', encoding="utf-8")
        fake_schema.unlink()
        fake_schema.symlink_to(external_schema)
        rejected("symlinked runtime schema", PackClosureError, lambda: code_import_closure(fake_code))
        rejected("symlinked schema pack", PackClosureError, lambda: task_pack_closure(task_dir, root=fake_code))
        rejected("symlinked schema discovery", TaskDiscoveryError, lambda: discover_standard_tasks(root, code_root=fake_code))
        rejected("symlinked schema locator load", TaskDiscoveryError, fake_locator.load)
        nested_code = base / "nested-code"
        (nested_code / "tcenv" / "taskcontract").mkdir(parents=True)
        (nested_code / "tcenv" / "opspec" / "schema").mkdir(parents=True)
        (nested_code / "tcenv" / "taskcontract" / "__init__.py").write_text("", encoding="utf-8")
        (nested_code / "tcenv" / "opspec" / "__init__.py").write_text("", encoding="utf-8")
        (nested_code / "tcenv" / "opspec" / "schema" / "opspec.v2.schema.json").write_text('{"type": "object"}\n', encoding="utf-8")
        nested_locator = discover_standard_tasks(root, code_root=nested_code)[0]
        outside_tree = base / "outside-schema-tree"
        outside_tree.mkdir()
        (outside_tree / "extra.json").write_text('{"extra": 1}\n', encoding="utf-8")
        (nested_code / "tcenv" / "opspec" / "schema" / "nested").symlink_to(outside_tree, target_is_directory=True)
        rejected("nested symlinked schema tree", PackClosureError, lambda: code_import_closure(nested_code))
        rejected("nested symlinked schema discovery", TaskDiscoveryError, lambda: discover_standard_tasks(root, code_root=nested_code))
        rejected("nested symlinked schema load", TaskDiscoveryError, nested_locator.load)
        (task_dir / "reference.py").write_text((task_dir / "reference.py").read_text() + "\n# drift\n", encoding="utf-8")
        rejected("task drift", TaskDiscoveryError, locator.load)
        tracks_path = base / "tracks.json"
        tracks_path.write_text(json.dumps(_tracks_document()), encoding="utf-8")
        selection = select_track("msl-bundle", registry_path=tracks_path)
        require(selection.track.renderer.implementation_id == "metal.track.msl-bundle.renderer", "the registered renderer is not selected")
        rejected("v2 without a formal track", UnsupportedContractCapability, lambda: select_track(None))
        rejected(
            "interface-only shipped track",
            UnsupportedContractCapability,
            lambda: load_track_registry().get("msl-single").require_available(),
        )
        rejected(
            "single-source has no default launch parser",
            UnsupportedContractCapability,
            lambda: default_geometry_resolver(select_track("msl-single", registry_path=tracks_path).track),
        )
        checks.append("discovery/staged rebuild, knowledge exclusion and track refusal")

    from taic.artifacts import RUN_CONFIG_SCHEMA_VERSION, NeedsInspection, validate_run_config_identity

    require(RUN_CONFIG_SCHEMA_VERSION == 3, "run_config schema is not 3")
    v1 = {"schema_version": 3, "track": None, "task_pack_digest": None, "effective_correctness_only": True}
    require(validate_run_config_identity(v1) is v1, "the v1 schema-3 identity was refused")
    v2 = {"schema_version": 3, "track": {"track_id": "msl-bundle", "track_version": 1}, "task_pack_digest": "a" * 64, "effective_correctness_only": False}
    require(validate_run_config_identity(v2) is v2, "the v2 schema-3 identity was refused")
    rejected("half-filled schema-3 identity", NeedsInspection, lambda: validate_run_config_identity({**v1, "task_pack_digest": "a" * 64}))
    checks.append("run_config schema 3 records v1/v2 identity or refuses a half pair")

    print(f"OK: self-test negatives ({len(checks)})")
    for name in checks:
        print(f"  - {name}")


def run_pytest() -> None:
    """@brief 运行固定 A3 pytest 节点。

    @raises AssertionError 任何用例失败。
    """
    command = [sys.executable, "-m", "pytest", "-q", "-n", "0", *FIXED_NODE_IDS]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join((result.stdout or "").strip().splitlines()[-3:])
    require(result.returncode == 0, f"fixed A3 pytest failed:\n{tail}\n{result.stderr[-4000:]}")
    print(f"OK: fixed A3 pytest\n{tail}")


def main() -> None:
    """@brief 验收入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_frozen_surface()
    verify_changed_surfaces()
    verify_boundary()
    verify_negative_nodes()
    self_test()
    if not args.skip_pytest:
        run_pytest()
    print("OK: A3 acceptance")


if __name__ == "__main__":
    main()
