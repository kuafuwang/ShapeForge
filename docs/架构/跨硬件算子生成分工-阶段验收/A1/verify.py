#!/usr/bin/env python3
"""A1 acceptance verifier: local OpSpec v2 contract, task contract plane and track skeleton.

Default run:

* checks ``manifest.sha256`` over the complete A1 deliverable file set;
* runs the fixed A1 pytest node ids (contract, reference, track, G0 identity, boundary);
* replays the stable negatives (mixed directory, no v1 fallback, semantic conflict,
  unsupported capability, reference boundary, read_write final value, author-free candidate);
* checks the frozen surface (schema 4 / RPC / HMAC / BlobChannel / candidate identity files
  and the shipped v1 tasks are untouched).

``--write-manifest`` regenerates ``manifest.sha256`` after an intentional A1 edit.  The
verifier never calls a model, never builds a kernel and never touches a GPU/NPU.
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

#: The complete A1 deliverable surface.  The plan file is a living document and is not part of
#: this freeze; the G0 corpus is independently frozen by the G0 manifest.
DELIVERABLES: tuple[str, ...] = (
    "tcenv/contract_errors.py",
    "tcenv/task_presentation.py",
    "tcenv/opspec/__init__.py",
    "tcenv/opspec/model.py",
    "tcenv/opspec/op_desc.py",
    "tcenv/opspec/v2.py",
    "tcenv/opspec/schema/opspec.v2.schema.json",
    "tcenv/taskcontract/__init__.py",
    "tcenv/taskcontract/baselines.py",
    "tcenv/taskcontract/display.py",
    "tcenv/taskcontract/errors.py",
    "tcenv/taskcontract/loader.py",
    "tcenv/taskcontract/reference.py",
    "tcenv/taskcontract/sampling.py",
    "tcenv/taskcontract/schema/baselines.v1.schema.json",
    "tcenv/taskcontract/schema/display.v1.schema.json",
    "tcenv/taskcontract/schema/reference.v1.schema.json",
    "tcenv/track/__init__.py",
    "tcenv/track/plane.py",
    "tcenv/track/registry.py",
    "tcenv/track/schema/tracks.v1.schema.json",
    "backends/metal/tracks.json",
    "tests/support/v2_mapping.py",
    "tests/support/v2_tasks.py",
    "tests/shared/test_op_desc_summary.py",
    "tests/shared/test_opspec_v2.py",
    "tests/shared/test_task_contract_v2.py",
    "tests/shared/test_track_plane.py",
    "tests/shared/test_v1_identity_g0.py",
    "tests/shared/test_v2_g0_reference_outputs.py",
    "tests/shared/test_v2_mapping.py",
    "tests/shared/test_cross_hardware_boundary.py",
    "tests/backends/metal/test_track_registry.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1/verify.py",
)

#: Fixed pytest entry points (stable file node ids).
FIXED_NODE_IDS: tuple[str, ...] = (
    "tests/shared/test_opspec_v2.py",
    "tests/shared/test_task_contract_v2.py",
    "tests/shared/test_track_plane.py",
    "tests/shared/test_v1_identity_g0.py",
    "tests/shared/test_v2_g0_reference_outputs.py",
    "tests/shared/test_v2_mapping.py",
    "tests/shared/test_op_desc_summary.py",
    "tests/shared/test_cross_hardware_boundary.py",
    "tests/backends/metal/test_track_registry.py",
)

#: Stable negative node ids that must exist and run (they are part of the fixed set above).
NEGATIVE_NODE_IDS: tuple[str, ...] = (
    "tests/shared/test_task_contract_v2.py::test_a_v2_directory_with_a_v1_marker_is_a_mixed_assembly_error",
    "tests/shared/test_task_contract_v2.py::test_v2_without_reference_does_not_fall_back_to_v1",
    "tests/shared/test_task_contract_v2.py::test_display_spec_text_is_limited_to_the_g0_sem_allowlist",
    "tests/shared/test_task_contract_v2.py::test_display_spec_text_rejects_unregistered_sem_int_suffixes",
    "tests/shared/test_task_contract_v2.py::test_display_formula_must_agree_with_the_opspec_and_carries_no_semantics",
    "tests/shared/test_task_contract_v2.py::test_a_workspace_tensor_needs_the_a2_abi_and_is_refused",
    "tests/shared/test_task_contract_v2.py::test_v2_variants_are_refused_at_assembly",
    "tests/shared/test_task_contract_v2.py::test_the_reference_module_cannot_depend_on_the_harness_or_a_candidate",
    "tests/shared/test_task_contract_v2.py::test_read_write_final_values_are_required_in_the_reference_result",
    "tests/shared/test_task_contract_v2.py::test_default_dtype_and_size_are_applied_only_for_none",
    "tests/shared/test_task_contract_v2.py::test_error_scale_gets_working_copies_and_cannot_mutate_inputs",
    "tests/shared/test_task_contract_v2.py::test_integer_case_override_is_checked_with_the_effective_profile",
    "tests/shared/test_task_contract_v2.py::test_the_reference_cannot_mutate_its_inputs_or_write_through_the_mapping",
    "tests/shared/test_task_contract_v2.py::test_reference_mutation_through_data_or_numpy_is_rejected[data]",
    "tests/shared/test_task_contract_v2.py::test_reference_mutation_through_data_or_numpy_is_rejected[numpy]",
    "tests/shared/test_task_contract_v2.py::test_error_scale_mutation_through_data_or_numpy_is_rejected[data]",
    "tests/shared/test_task_contract_v2.py::test_error_scale_mutation_through_data_or_numpy_is_rejected[numpy]",
    "tests/shared/test_task_contract_v2.py::test_the_reference_registry_is_one_authoritative_oracle_per_op",
    "tests/shared/test_task_contract_v2.py::test_gen_inputs_and_a_sampling_profile_are_mutually_exclusive",
    "tests/shared/test_track_plane.py::test_the_same_candidate_bytes_have_one_verdict_regardless_of_author_source",
    "tests/shared/test_track_plane.py::test_candidate_paths_and_bytes_fail_closed",
    "tests/shared/test_track_plane.py::test_candidate_roles_must_be_registered_on_the_target_track",
    "tests/shared/test_track_plane.py::test_an_op_specific_or_tiling_field_is_refused_at_the_schema_level",
    "tests/shared/test_track_plane.py::test_the_track_renderer_requires_an_abi_and_is_deterministic",
    "tests/shared/test_opspec_v2.py::test_attrs_may_not_smuggle_launch_or_tiling_parameters",
    "tests/shared/test_v1_identity_g0.py::test_the_g0_combo_inventory_is_unchanged",
    "tests/shared/test_cross_hardware_boundary.py::test_a1_production_mentions_no_rejected_hardware_platform",
)

#: Files whose identity/protocol/serialisation must not change in A1.
FROZEN_SURFACE: tuple[str, ...] = (
    "proto",
    "tcenv/remote/v1",
    "tcenv/remote/artifact_bundle.py",
    "tcenv/remote/contracts.py",
    "tcenv/remote/grpc_client.py",
    "tcenv/remote/grpc_contract.py",
    "tcenv/remote/grpc_service.py",
    "tcenv/remote/task_pack.py",
    "tcenv/evaluation.py",
    "tcenv/task.py",
    "tcenv/contract.py",
    "tcenv/build_projection.py",
    "tcenv/report.py",
    "tcenv/schema.py",
    "tcenv/verify.py",
    "tcenv/backends/metal/source_builder.py",
    "tcenv/backends/metal/lowering.py",
    "tcenv/track/schema/track.v1.schema.json",
    "tcenv/opspec/schema/opspec.v1.schema.json",
    "tasks",
    "backends/metal/backend.toml",
    "targets",
    "taic",
)

#: The checkout A1 started from.  The frozen-surface check compares against this commit (not
#: against ``HEAD``), so committing A1 cannot hide a schema 4/RPC/v1-task change.
A1_BASE_HEAD = "dc8670486e5baed5248505b5bdc80d0f21cbcbc2"

REJECTED_PLATFORM = re.compile(
    r"\b(cuda|cann|ascend[a-z]*|msopgen|msprof|easyasc|acl|npu|torch_npu|nvidia)\b", re.IGNORECASE
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


def write_manifest() -> None:
    """@brief 重写 A1 manifest.sha256（覆盖固定交付面）。"""
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
    require(set(seen) == expected, f"manifest file set differs: missing={sorted(expected - set(seen))} extra={sorted(set(seen) - expected)}")
    for name, digest in seen.items():
        require(sha256(ROOT / name) == digest, f"manifest mismatch: {name}")
    print(f"OK: manifest.sha256 ({len(seen)} files)")


def _git(*args: str) -> str:
    """@brief 运行 git 并返回 stdout。

    @param args git 参数。
    @return stdout 文本。
    @raises AssertionError git 失败。
    """
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def verify_frozen_surface() -> None:
    """@brief 证明 schema 4/RPC/HMAC/BlobChannel/单文件身份与 v1 题面都没有被 A1 改动。

    @raises AssertionError 冻结面出现 diff。
    """
    base = _git("rev-parse", "--verify", f"{A1_BASE_HEAD}^{{commit}}").strip()
    require(base == A1_BASE_HEAD, f"A1 base commit is not available: {A1_BASE_HEAD}")
    changed = _git("diff", "--name-only", A1_BASE_HEAD, "--", *FROZEN_SURFACE).splitlines()
    require(changed == [], "frozen surface changed since the A1 base commit: " + ", ".join(changed))
    print(f"OK: frozen surface untouched since {A1_BASE_HEAD[:12]} ({len(FROZEN_SURFACE)} paths)")


def verify_a1_boundary() -> None:
    """@brief A1 生产代码/配置不携带被拒绝平台的名称或专用知识。

    @raises AssertionError 出现禁止词。
    """
    offenders: list[str] = []
    for name in DELIVERABLES:
        if not name.startswith(("tcenv/", "backends/")):
            continue
        path = ROOT / name
        if path.suffix not in {".py", ".json"}:
            continue
        for match in REJECTED_PLATFORM.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{name}: {match.group(0)}")
    require(offenders == [], "\n".join(offenders))
    print("OK: no rejected hardware platform in the A1 production surface")


def run_pytest() -> str:
    """@brief 运行固定 A1 pytest 节点。

    @return pytest 的简要结果行。
    @raises AssertionError 任何用例失败。
    """
    command = [sys.executable, "-m", "pytest", "-q", "-n", "0", *FIXED_NODE_IDS]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join((result.stdout or "").strip().splitlines()[-3:])
    require(result.returncode == 0, f"fixed A1 pytest failed:\n{tail}\n{result.stderr[-4000:]}")
    print(f"OK: fixed A1 pytest\n{tail}")
    return tail


def self_test() -> None:
    """@brief 可独立重放的稳定负例（不依赖 pytest 收集）。

    @raises AssertionError 任一负例未被拒绝。
    """
    sys.path.insert(0, str(ROOT))
    import hashlib as _hashlib

    from tcenv.contract_errors import (
        AssemblyError,
        ReferenceBoundaryError,
        SemanticUnresolved,
        TaskContractError,
        UnsupportedContractCapability,
    )
    from tcenv.identity import fingerprint
    from tcenv.opspec import (
        OP_DESC_TEMPLATE_KEYS,
        OpSpecError,
        instantiate,
        load_opspec,
        opspec_v2_from_document,
        render_op_desc,
        validate_opspec_v2,
    )
    from tcenv.taskcontract import ReferenceRegistry, load_task_contract, spec_text_override_allowed
    from tcenv.track import (
        CandidateArtifact,
        CandidateEntry,
        ExecutionGeometry,
        GeometryStage,
        TensorMemoryContract,
        load_track_registry,
        registry_from_document,
    )
    from tests.support.v2_tasks import (
        RMSNORM_REFERENCE,
        rmsnorm_opspec,
        rmsnorm_read_write_opspec,
        write_task,
    )

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

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)

        def task_dir(name: str = "rmsnorm", **kwargs) -> Path:
            return write_task(base / name, **kwargs)

        document = rmsnorm_opspec()
        document["spec_version"] = 3
        bad_version = base / "rmsnorm" / "opspec.json"
        bad_version.parent.mkdir(parents=True, exist_ok=True)
        bad_version.write_text(json.dumps(document), encoding="utf-8")
        rejected("unsupported spec_version", OpSpecError, lambda: load_opspec(bad_version))

        uses = rmsnorm_opspec()
        uses["uses"] = ["EPS"]
        bad_uses = base / "uses" / "rmsnorm" / "opspec.json"
        bad_uses.parent.mkdir(parents=True, exist_ok=True)
        bad_uses.write_text(json.dumps(uses), encoding="utf-8")
        rejected("v2 uses list", OpSpecError, lambda: load_opspec(bad_uses))

        mixed = task_dir("mixed")
        (mixed / "track.json").write_text("{}", encoding="utf-8")
        rejected("mixed v1/v2 directory", AssemblyError, lambda: load_task_contract(mixed))

        no_reference = write_task(base / "noref", reference=None)
        rejected("v2 without reference", AssemblyError, lambda: load_task_contract(no_reference))

        display = task_dir("display", extra={"display.json": json.dumps({"schema_version": 1, "spec_text": "x"})})
        rejected("display override outside sem_*", SemanticUnresolved, lambda: load_task_contract(display))

        require(spec_text_override_allowed("sem_int_" + "0" * 17 + "7"), "the 18-digit sem_int boundary is rejected")
        for bad_name in (
            "sem_int_not_a_seed",
            "sem_int_",
            "sem_int_-1",
            "sem_int_1e3",
            "sem_int_\u0661",
            "sem_int_\uff11",
            "sem_int_1\u0661",
            "sem_int_" + "0" * 18 + "1",
        ):
            require(not spec_text_override_allowed(bad_name), f"unregistered sem_int spelling accepted: {bad_name!r}")
        bad_sem_display = task_dir(
            "bad_sem_display",
            op_name="sem_int_not_a_seed",
            extra={"display.json": json.dumps({"schema_version": 1, "spec_text": "replacement"})},
        )
        rejected(
            "display sem_int suffix grammar",
            SemanticUnresolved,
            lambda: load_task_contract(bad_sem_display),
        )

        formula = task_dir("formula", extra={"display.json": json.dumps({"schema_version": 1, "formula": "y = x"})})
        rejected("display/OpSpec formula conflict", SemanticUnresolved, lambda: load_task_contract(formula))

        workspace = rmsnorm_opspec()
        workspace["tensors"]["y"]["space"] = "workspace"
        rejected(
            "workspace capability",
            UnsupportedContractCapability,
            lambda: load_task_contract(task_dir("workspace", opspec=workspace)),
        )
        rejected(
            "v2 @variant",
            UnsupportedContractCapability,
            lambda: load_task_contract(task_dir("variant"), variant="128x64x32t256"),
        )
        rejected(
            "empty v2 @variant",
            UnsupportedContractCapability,
            lambda: load_task_contract(task_dir("variant"), variant=""),
        )

        boundary = task_dir("boundary", reference="import known_good\n\ndef reference(inputs, attrs):\n    return {}\n")
        rejected("reference boundary import", ReferenceBoundaryError, lambda: load_task_contract(boundary))

        read_write = task_dir(
            "rmsnorm_rw",
            op_name="rmsnorm_rw",
            opspec=rmsnorm_read_write_opspec(),
            reference="""import torch\n\ndef reference(inputs, attrs):\n    return {"y": inputs["x"]}\n""",
        )
        rw_contract = load_task_contract(read_write)
        rejected(
            "read_write final value",
            SemanticUnresolved,
            lambda: rw_contract.reference.execute(
                rw_contract.instance, rw_contract.reference.generate(rw_contract.instance, "normal", 1)
            ),
        )

        single = CandidateArtifact(
            track_id="msl-single",
            entries=(CandidateEntry(path="k.metal", role="device_kernel", media_type="text/plain", data=b"kernel void k(){}\n"),),
        )
        require(single.candidate_sha256 == _hashlib.sha256(b"kernel void k(){}\n").hexdigest(), "single-source digest is not the byte sha256")
        require(
            single.verdict_key()
            == CandidateArtifact(
                track_id="msl-single",
                entries=(CandidateEntry(path="k.metal", role="device_kernel", media_type="text/plain", data=b"kernel void k(){}\n"),),
            ).verdict_key(),
            "same candidate bytes produced different verdict keys",
        )
        checks.append("author-free candidate identity")

        registry = load_track_registry()
        require(registry.available() == (), "a track is available without real evidence")
        document = json.loads(json.dumps(registry.document()))
        document["tracks"][0]["status"] = "available"
        rejected("available track without evidence", TaskContractError, lambda: registry_from_document(document))

        op_specific = json.loads(json.dumps(registry.document()))
        op_specific["tracks"][0]["op_name"] = "rmsnorm"
        rejected("op-specific track metadata", TaskContractError, lambda: registry_from_document(op_specific))

        for forbidden in ("tile", "grid", "threads", "warps", "fallback"):
            require(not hasattr(TensorMemoryContract, forbidden), f"TensorMemoryContract has {forbidden}")
        checks.append("TensorMemoryContract locks ABI/memory/numerics only")

        import dataclasses

        require(
            {field.name for field in dataclasses.fields(ExecutionGeometry)} == {"source", "stages", "schema_version"},
            "ExecutionGeometry is not the frozen A1 field set",
        )
        require(
            {field.name for field in dataclasses.fields(GeometryStage)}
            == {"kernel", "grid", "threads", "threadgroup_bytes", "workspace", "when", "derived_scalars"},
            "GeometryStage is not the frozen A1 field set",
        )
        checks.append("ExecutionGeometry/GeometryStage are interface-only field sets")

        v2 = opspec_v2_from_document(rmsnorm_opspec(), source="<self-test>")
        require(fingerprint(v2.digest_document()) == v2.digest, "v2 digest_document does not recompute the loaded digest")
        require(v2.digest_document()["fingerprint_version"] == 2, "v2 digest uses the v1 envelope")
        checks.append("v2 digest_document recomputes the digest")
        summary = render_op_desc(instantiate(v2, "fp16", "small"))
        require(tuple(summary) == OP_DESC_TEMPLATE_KEYS, "op_desc summary has an open key set")
        require("op_desc" not in instantiate(v2, "fp16", "small").fingerprint_document(), "op_desc entered the semantic projection")
        checks.append("op_desc summary is closed and non-authoritative")

        for bad_path in ("a/../b.metal", "a/../../b.metal", "a/b/../../c.metal", "a/./b.metal", "."):
            try:
                CandidateEntry(path=bad_path, role="device_kernel", media_type="text/plain", data=b"k")
            except TaskContractError:
                continue
            raise AssertionError(f"candidate path traversal accepted: {bad_path!r}")
        checks.append("candidate paths reject every . or .. segment")

        for bad_attr in ("Block_M", "NumWarps", "BLOCKM", "NUMWARPS", "EPS-value"):
            attr_document = rmsnorm_opspec()
            attr_document["attrs"][bad_attr] = {"dtype": "int", "default": 4}
            try:
                validate_opspec_v2(attr_document, source="<self-test>")
            except OpSpecError:
                continue
            raise AssertionError(f"invalid attr identifier/geometry spelling accepted: {bad_attr!r}")
        checks.append("v2 attr syntax and geometry variants fail closed")

        for kwargs in ({"dtype": ""}, {"size": ""}):
            try:
                load_task_contract(task_dir("empty_defaults"), **kwargs)
            except OpSpecError:
                continue
            raise AssertionError(f"empty default selector was silently replaced: {kwargs!r}")
        checks.append("default dtype/size are applied only for None")

        first = load_task_contract(task_dir("registry_first"))
        versioned = task_dir(
            "registry_versioned",
            extra={"reference.json": json.dumps({"schema_version": 1, "reference_version": 2})},
        )
        registry_v2 = ReferenceRegistry()
        registry_v2.register(first.reference)
        rejected(
            "reference registry identity conflict",
            AssemblyError,
            lambda: registry_v2.register(load_task_contract(versioned).reference),
        )

        inplace = task_dir(
            "inplace",
            reference="import torch\n\ndef reference(inputs, attrs):\n    inputs['x'].mul_(2)\n    return {'y': inputs['x']}\n",
        )
        inplace_contract = load_task_contract(inplace)
        caller_inputs = inplace_contract.reference.generate(inplace_contract.instance, "normal", 1)
        caller_before = {name: tensor.clone() for name, tensor in caller_inputs.items()}
        caller_versions = {name: tensor._version for name, tensor in caller_inputs.items()}
        rejected(
            "reference input mutation",
            ReferenceBoundaryError,
            lambda: inplace_contract.reference.execute(inplace_contract.instance, caller_inputs),
        )
        import torch

        for name, tensor in caller_inputs.items():
            require(torch.equal(tensor, caller_before[name]), f"caller input {name} was mutated")
            require(tensor._version == caller_versions[name], f"caller input {name} version changed")
        checks.append("reference mutation does not leak into caller inputs")

        scale_task = task_dir(
            "scale_boundary",
            reference=RMSNORM_REFERENCE
            + "\n\ndef error_scale(inputs, outputs, instance, tolerances):\n"
            "    inputs['x'].mul_(0)\n"
            "    outputs['y'].mul_(0)\n"
            "    return {name: torch.ones((), dtype=torch.float64) for name in outputs}\n",
        )
        scale_contract = load_task_contract(scale_task)
        scale_inputs = scale_contract.reference.generate(scale_contract.instance, "normal", 1)
        scale_outputs = scale_contract.reference.execute(scale_contract.instance, scale_inputs)
        scale_before_inputs = {name: tensor.clone() for name, tensor in scale_inputs.items()}
        scale_before_outputs = {name: tensor.clone() for name, tensor in scale_outputs.items()}
        rejected(
            "error_scale input mutation",
            ReferenceBoundaryError,
            lambda: scale_contract.reference.error_scale(scale_contract.instance, scale_inputs, scale_outputs),
        )
        for name, tensor in scale_inputs.items():
            require(torch.equal(tensor, scale_before_inputs[name]), f"error_scale mutated caller input {name}")
        for name, tensor in scale_outputs.items():
            require(torch.equal(tensor, scale_before_outputs[name]), f"error_scale mutated caller output {name}")
        checks.append("error_scale receives working copies")

        bundle_registry = load_track_registry()
        bundle_with_launch = CandidateArtifact(
            track_id="msl-bundle",
            entries=(
                CandidateEntry(path="k.metal", role="device_kernel", media_type="text/plain", data=b"k"),
                CandidateEntry(path="host/t.metal", role="host_tiling", media_type="text/plain", data=b"t"),
                CandidateEntry(path="host/s.json", role="schedule", media_type="application/json", data=b"{}"),
                CandidateEntry(path="launch.json", role="launch", media_type="application/json", data=b"{}"),
            ),
        )
        rejected(
            "candidate role outside track candidate_roles",
            TaskContractError,
            lambda: bundle_with_launch.validate_for(bundle_registry.get("msl-bundle")),
        )

        packed = task_dir("pack")
        before = load_task_contract(packed)
        (packed / "guidance_catalog.md").write_text("wiki\n", encoding="utf-8")
        (packed / "skills").mkdir()
        (packed / "skills" / "a.md").write_text("hint\n", encoding="utf-8")
        after = load_task_contract(packed)
        require(after.pack_digest == before.pack_digest, "guidance files entered task_pack_digest")
        require(after.fingerprint() == before.fingerprint(), "guidance files entered the task fingerprint")
        checks.append("knowledge excluded from task pack/fingerprint")

        require(RMSNORM_REFERENCE in before.reference_source, "reference display text is missing")
        checks.append("read-only reference display")

    print(f"OK: self-test negatives ({len(checks)})")
    for name in checks:
        print(f"  - {name}")


def verify_negative_nodes() -> None:
    """@brief 确认固定负例节点仍然存在（不被删除/改名）。

    @raises AssertionError 节点不存在。
    """
    for node in NEGATIVE_NODE_IDS:
        path, _, test = node.partition("::")
        test = test.split("[", 1)[0]
        source = (ROOT / path).read_text(encoding="utf-8")
        require(f"def {test}(" in source, f"negative node id is gone: {node}")
    print(f"OK: {len(NEGATIVE_NODE_IDS)} stable negative node ids present")


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
    verify_a1_boundary()
    verify_negative_nodes()
    self_test()
    if not args.skip_pytest:
        run_pytest()
    print("OK: A1 acceptance")


if __name__ == "__main__":
    main()
