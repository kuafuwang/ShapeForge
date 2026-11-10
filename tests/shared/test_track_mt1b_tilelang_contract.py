"""MT1b: the frozen TileLang Track contract, build identity and archived source-only replay.

Every case here is CPU-only.  Nothing imports ``tilelang``, compiles Metal, creates a device
library or dispatches: the identity checks are pure functions and the replay re-derives digests
from the bytes the MT0 probe already archived.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
from tcenv.identity import fingerprint
from tcenv.track import CandidateArtifact, CandidateEntry, load_track_registry
from tcenv.track import tilelang as contract

ROOT = Path(__file__).resolve().parents[2]
MT0 = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-tilelang-probe"
R5_RESULT = MT0 / "probe-result-r5.json"
R5_ARTIFACTS = MT0 / "evidence/source-only-r5"
R5_CANDIDATE = MT0 / "evidence/failure-inputs/r5-candidate.py"
R5_ARTIFACT_SET_SHA256 = "bc6a7300ee1010be1c7667b187c951cd7ec1e6961b87d288886e3776698e821f"
R5_CANDIDATE_SHA256 = "3ebead1927e3fe4885b98151ee11f92ddd2e8e13314e9d0a530fc11506e32a8c"

#: The real MT0 install identity (not a placeholder): a well-formed digest of other bytes is drift.
GOOD_FRONTEND = {
    "frontend_id": contract.FRONTEND_ID,
    "frontend_version": contract.FRONTEND_VERSION,
    "entry": contract.FRONTEND_ENTRY,
    "options_sha256": contract.options_sha256(),
    "package_manifest_sha256": contract.FRONTEND_PACKAGE_MANIFEST_SHA256,
    "dist_info_metadata_sha256": contract.FRONTEND_DIST_INFO_METADATA_SHA256,
    "dist_info_record_sha256": contract.FRONTEND_DIST_INFO_RECORD_SHA256,
    "package_files": contract.FRONTEND_PACKAGE_FILES,
    "package_bytes": contract.FRONTEND_PACKAGE_BYTES,
}
GOOD_ENVIRONMENT = {
    "python_version": "3.13.1",
    "python_implementation": "CPython",
    "platform": "macOS-26.1-arm64",
    "machine": "arm64",
}


def _tilelang_spec():
    return load_track_registry().get(contract.TILELANG_TRACK_ID, version=1)


def _source_candidate(*, geometry: bool = True, source: bytes = b"import tilelang\n", grid: int = 4) -> CandidateArtifact:
    entries = [
        CandidateEntry(
            path=contract.SOURCE_PATH,
            role=contract.SOURCE_ROLE,
            media_type=contract.SOURCE_MEDIA_TYPE,
            data=source,
        )
    ]
    if geometry:
        entries.append(
            CandidateEntry(
                path=contract.GEOMETRY_PATH,
                role=contract.GEOMETRY_ROLE,
                media_type=contract.GEOMETRY_MEDIA_TYPE,
                data=json.dumps(
                    {
                        "schema_version": 1,
                        "source": "candidate",
                        "stages": [{"kernel": "scale_kernel", "grid": [grid, 1, 1], "threads": [256, 1, 1]}],
                    }
                ).encode(),
            )
        )
    return CandidateArtifact(track_id=contract.TILELANG_TRACK_ID, entries=tuple(entries))


def _artifacts() -> dict[str, bytes]:
    return {path: (R5_ARTIFACTS / path).read_bytes() for path in contract.SOURCE_ONLY_ARTIFACT_PATHS}


def _artifact_rows(artifacts: dict[str, bytes]) -> list[dict]:
    return [
        {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in artifacts.items()
    ]


def _r5_result() -> dict:
    return json.loads(R5_RESULT.read_text(encoding="utf-8"))


ATTEMPT = {
    "status": "completed",
    "returncode": 0,
    "timed_out": False,
    "online": False,
    "hosts": [],
    "route": contract.ISOLATION_ROUTE,
    "request_sha256": "a" * 64,
    "hmz_version": "0.2.2",
    "policy_sha256": "b" * 64,
}


def _production_artifacts(source: bytes = R5_CANDIDATE.read_bytes()) -> dict[str, bytes]:
    """The production entry's artifact format, built on the *real* r5 lowered bytes.

    r5 predates the ``candidate`` block (it is a probe-format archive, replayed by its own
    function); this fixture adds only that block so the production identity path can be exercised
    on real MSL/IR/params/host bytes.
    """
    base = _artifacts()
    identity = json.loads(base["frontend_identity.json"])
    identity["schema_id"] = "tcenv.track.tilelang.frontend-identity.1"
    identity["candidate"] = {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "prim_func_name": "scale_kernel",
    }
    identity["python"] = {
        "version": GOOD_ENVIRONMENT["python_version"],
        "implementation": GOOD_ENVIRONMENT["python_implementation"],
        "platform": GOOD_ENVIRONMENT["platform"],
        "machine": GOOD_ENVIRONMENT["machine"],
    }
    return {**base, "frontend_identity.json": json.dumps(identity, sort_keys=True, indent=2).encode()}


def _identity(source: bytes = R5_CANDIDATE.read_bytes(), *, grid: int = 4, artifacts=None):
    candidate = _source_candidate(source=source, grid=grid)
    return contract.source_only_build_identity(
        candidate=candidate,
        observed_frontend=GOOD_FRONTEND,
        environment=GOOD_ENVIRONMENT,
        artifacts=_production_artifacts(source) if artifacts is None else artifacts,
        attempt=ATTEMPT,
        expected_kernel_names=("scale_kernel",),
    ), candidate


def test_the_frozen_contract_names_a_python_dsl_source_and_kernel_pipeline_geometry():
    """@brief 候选源是 kernel.py/text/x-python，不是 kernel.metal；几何是 kernel_pipeline。"""
    document = contract.frozen_contract()
    assert document["schema_id"] == contract.TRACK_CONTRACT_SCHEMA_ID
    assert document["track_id"] == contract.TILELANG_TRACK_ID == "tilelang-pipeline"
    assert document["track_version"] == contract.TILELANG_TRACK_VERSION == 1
    assert document["delivery"] == contract.TILELANG_DELIVERY == "dsl_source_pipeline"
    assert document["execution_kind"] == "kernel_pipeline"
    assert document["candidate_source"] == {"path": "kernel.py", "media_type": "text/x-python", "role": "dsl_source"}
    assert "kernel.metal" not in json.dumps(document["candidate_source"])
    assert document["geometry"]["path"] == "host/execution_geometry.json"
    assert document["frontend"]["frontend_version"] == "0.1.15"
    assert document["frontend"]["options_sha256"] == contract.options_sha256()
    assert document["build"]["implemented_modes"] == ["source_only"]
    assert document["build"]["metal_build"] == "not implemented" if "metal_build" in document["build"] else True
    assert document["evidence_boundary"]["availability"].startswith("interface_only")


def test_the_registered_track_matches_the_frozen_contract_and_drift_is_typed():
    """@brief 注册声明⇄冻结合同交叉核对；前端/options/实现身份漂移全部 typed。"""
    spec = _tilelang_spec()
    audit = contract.audit_registration(spec)
    assert audit["track"] == "tilelang-pipeline/1"
    assert audit["contract_digest"] == fingerprint(contract.frozen_contract())

    import dataclasses

    def frontend(**changes):
        return dataclasses.replace(spec, frontend=dataclasses.replace(spec.frontend, **changes))

    for build, error, message in (
        (lambda: frontend(frontend_version="0.1.16"), contract.TileLangIdentityDrift, "frontend.frontend_version"),
        (lambda: frontend(options_sha256="0" * 64), contract.TileLangIdentityDrift, "frontend.options_sha256"),
        (lambda: frontend(entry="tilelang.jit"), contract.TileLangIdentityDrift, "frontend.entry"),
        (lambda: dataclasses.replace(
            spec, candidate_source=dataclasses.replace(spec.candidate_source, media_type="text/x-msl")
        ), TaskContractError, "is MSL"),
        (lambda: dataclasses.replace(spec, track_version=2), contract.TileLangContractError, "frozen contract is version 1"),
        (lambda: dataclasses.replace(spec, delivery="candidate_bundle"), contract.TileLangContractError, "delivery"),
        (lambda: dataclasses.replace(
            spec, build_adapter=dataclasses.replace(spec.build_adapter, implementation_version=2)
        ), contract.TileLangIdentityDrift, "build_adapter"),
    ):
        with pytest.raises(error, match=message):
            contract.audit_registration(build())


def test_the_candidate_source_contract_is_enforced_by_the_track_plane():
    """@brief 交付形状：一个 kernel.py + 可选几何；MSL 冒充/未声明条目/错误角色全拒。"""
    spec = _tilelang_spec()
    assert contract.validate_candidate(_source_candidate(), spec) == "accepted"
    assert contract.validate_candidate(_source_candidate(geometry=False), spec) == "accepted"
    # The TrackSpec's own roles are the registry declaration: CandidateArtifact validates them too.
    assert _source_candidate().validate_for(spec) == "accepted"

    def candidate(*entries) -> CandidateArtifact:
        return CandidateArtifact(track_id=contract.TILELANG_TRACK_ID, entries=tuple(entries))

    source = CandidateEntry(path="kernel.py", role="dsl_source", media_type="text/x-python", data=b"x")
    geometry = CandidateEntry(
        path="host/execution_geometry.json", role="schedule", media_type="application/json", data=b"{}"
    )
    negatives = (
        # An MSL-looking candidate can never be relabelled into this Track: an MSL candidate role is
        # not a registered role of this Track, so the refusal happens before any source parse.
        (candidate(CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"k")), "not registered"),
        (candidate(source, CandidateEntry(path="device/x.metal", role="device_kernel", media_type="text/plain", data=b"k")), "not registered"),
        (candidate(source, geometry, CandidateEntry(path="extra.txt", role="launch", media_type="text/plain", data=b"z")), "not registered"),
        # Exactly one dsl_source entry, at the declared path/media type.
        (candidate(source, CandidateEntry(path="other.py", role="dsl_source", media_type="text/x-python", data=b"y")), "exactly one dsl_source"),
        (candidate(CandidateEntry(path="kernel.py", role="dsl_source", media_type="text/plain", data=b"x")), "candidate source as"),
        (candidate(CandidateEntry(path="other.py", role="dsl_source", media_type="text/x-python", data=b"x")), "candidate source as"),
    )
    for artifact, message in negatives:
        with pytest.raises(TaskContractError, match=message):
            contract.validate_candidate(artifact, spec)

    wrong_geometry = CandidateEntry(path="host/geometry.json", role="schedule", media_type="application/json", data=b"{}")
    with pytest.raises(TaskContractError, match="execution geometry must be"):
        contract.validate_candidate(candidate(source, wrong_geometry), spec)
    # Even a ``schedule`` role cannot smuggle an MSL source/library path into a DSL candidate.
    msl_schedule = CandidateEntry(path="device/x.metal", role="schedule", media_type="application/json", data=b"{}")
    with pytest.raises(TaskContractError, match="is an MSL source/library"):
        contract.validate_candidate(candidate(source, msl_schedule), spec)


def test_a_missing_or_drifting_frontend_fails_closed():
    """@brief 缺前端是 backend_capability_unavailable；版本/入口/options/包摘要漂移是 IdentityDrift。"""
    assert contract.verify_frontend_observation(GOOD_FRONTEND) == GOOD_FRONTEND
    with pytest.raises(contract.TileLangFrontendUnavailable) as missing:
        contract.verify_frontend_observation(None)
    assert missing.value.code == "backend_capability_unavailable"
    with pytest.raises(contract.TileLangFrontendUnavailable, match="incomplete"):
        contract.verify_frontend_observation({"frontend_id": "tilelang"})
    # A frontend that is *absent* is not a version drift; it is a capability gap.
    for field, value, message in (
        ("frontend_version", "0.1.16", "not the frozen '0.1.15'"),
        ("frontend_id", "halide", "not the frozen 'tilelang'"),
        ("entry", "tilelang.compile", "not the frozen"),
        ("options_sha256", "0" * 64, "options digest"),
        ("package_manifest_sha256", "xyz", "64-character lowercase hex"),  # malformed digest, not a version drift
        # Well-formed digests of *different* bytes are the dangerous case: a valid-looking hash is
        # not the pinned install, so each one is drift and must be refused.
        ("package_manifest_sha256", "d" * 64, "not the pinned one"),
        ("dist_info_metadata_sha256", "d" * 64, "not the pinned one"),
        ("dist_info_record_sha256", "d" * 64, "not the pinned one"),
        ("package_files", contract.FRONTEND_PACKAGE_FILES + 1, "pinned file count"),
        ("package_bytes", contract.FRONTEND_PACKAGE_BYTES - 1, "pinned byte count"),
    ):
        observed = dict(GOOD_FRONTEND)
        observed[field] = value
        if field == "package_manifest_sha256" and value == "xyz":
            error = TaskContractError
        else:
            error = contract.TileLangIdentityDrift
        with pytest.raises(error, match=message):
            contract.verify_frontend_observation(observed)
    # A boolean/float is never a byte count, however convenient the coercion would be.
    for field, value in (("package_files", True), ("package_bytes", 134219728.0)):
        observed = dict(GOOD_FRONTEND)
        observed[field] = value
        with pytest.raises(contract.TileLangIdentityDrift, match="pinned"):
            contract.verify_frontend_observation(observed)


def test_the_build_request_isolation_is_offline_and_declared_only():
    """@brief online/host/外部读写/第二套 Seatbelt 全拒；合法声明规范化。"""
    good = {
        "route": contract.ISOLATION_ROUTE,
        "online": False,
        "hosts": [],
        "external_reads": [],
        "external_writes": [],
        "declared_write_roots": ["/tmp/attempt/build/work", "/tmp/attempt/build/artifacts"],
        "internal_fence_or_wrap": False,
    }
    assert contract.verify_isolation(good)["online"] is False
    for changed, message in (
        ({**good, "online": True}, "offline"),
        ({**good, "hosts": ["example.com"]}, "empty host allowlist"),
        ({**good, "external_reads": ["/etc/passwd"]}, "declared inputs"),
        ({**good, "external_writes": ["/tmp/outside"]}, "declared artifact"),
        ({**good, "route": "sandbox-exec -p custom"}, "lower-level hmz Profile route"),
        ({**good, "internal_fence_or_wrap": True}, "internal_fence_or_wrap must be False"),
        ({**good, "declared_write_roots": []}, "non-empty list"),
        ({**good, "declared_write_roots": ["relative/dir"]}, "absolute attempt root"),
        ({**good, "online": "false"}, "offline"),
        ({**good, "extra": 1}, "unregistered"),
    ):
        with pytest.raises(TaskContractError, match=message):
            contract.verify_isolation(changed)


def test_source_only_and_metal_build_modes_are_separated_not_switched_silently():
    """@brief Metal build 未实现：请求它是 typed capability 拒绝，不是翻转一个 option。"""
    assert contract.options_document("source_only")["enable_device_compile"] is False
    assert contract.options_document("metal_build")["enable_device_compile"] is True
    assert contract.options_sha256("source_only") != contract.options_sha256("metal_build")
    with pytest.raises(contract.TileLangBuildModeUnavailable, match="not implemented in this epoch"):
        contract.source_only_build_identity(
            candidate=_source_candidate(),
            observed_frontend=GOOD_FRONTEND,
            environment=GOOD_ENVIRONMENT,
            artifacts=_artifacts(),
            attempt=ATTEMPT,
            build_mode="metal_build",
        )
    with pytest.raises(contract.TileLangBuildModeUnavailable, match="unknown TileLang build mode"):
        contract.options_document("jit")


def test_the_build_identity_binds_the_whole_candidate_frontend_options_and_environment():
    """@brief build 身份/缓存键绑定完整候选 manifest（含几何）、前端、options、环境与真实尝试。"""
    identity, candidate = _identity(R5_CANDIDATE.read_bytes())
    document = identity.document()
    assert document["candidate_sha256"] == candidate.candidate_sha256 != R5_CANDIDATE_SHA256
    assert document["dsl_source_sha256"] == R5_CANDIDATE_SHA256
    # The whole manifest enters the identity: the DSL source *and* the candidate-owned geometry.
    assert [row["path"] for row in document["candidate_entries"]] == ["host/execution_geometry.json", "kernel.py"]
    assert document["artifact_set_sha256"] == contract.artifact_set_sha256(document["artifacts"])
    assert identity.full_build is True
    assert document["device_independent"] is True
    assert document["device_dispatch_calls"] == 0
    assert document["attempt"]["status"] == "completed"
    assert document["isolation"]["online"] is False
    assert identity.digest == fingerprint(document)
    # Deterministic.
    again, _ = _identity(R5_CANDIDATE.read_bytes())
    assert again.digest == identity.digest and again.cache_key == identity.cache_key

    # Same source, different execution geometry: the candidate manifest (and therefore the build
    # subject / cache key) must change.  Hashing only kernel.py collided here.
    other_geometry, other_candidate = _identity(R5_CANDIDATE.read_bytes(), grid=999)
    assert other_candidate.candidate_sha256 != candidate.candidate_sha256
    assert other_geometry.candidate_sha256 == other_candidate.candidate_sha256
    assert other_geometry.cache_key != identity.cache_key
    assert other_geometry.digest != identity.digest
    assert other_geometry.artifact_set_sha256 == identity.artifact_set_sha256  # the lowered bytes are the same

    changed_source, _ = _identity(R5_CANDIDATE.read_bytes() + b"\n# edited\n")
    assert changed_source.cache_key != identity.cache_key
    drifted_environment, _ = _identity(R5_CANDIDATE.read_bytes())
    drifted = contract.source_only_build_identity(
        candidate=candidate,
        observed_frontend=GOOD_FRONTEND,
        environment={**GOOD_ENVIRONMENT, "platform": "macOS-27-arm64"},
        artifacts=_production_artifacts(),
        attempt=ATTEMPT,
    )
    assert drifted.cache_key != drifted_environment.cache_key
    assert drifted.artifact_set_sha256 == identity.artifact_set_sha256

    # The expected subject is reproducible *before* a build from independent inputs only.
    subject = contract.expected_build_subject(candidate=candidate, environment=GOOD_ENVIRONMENT)
    assert subject.cache_key == identity.cache_key
    assert subject.full_build is False and subject.artifacts == ()
    assert contract.expected_build_subject(candidate=other_candidate, environment=GOOD_ENVIRONMENT).cache_key != subject.cache_key

    # Closed sets: no extra artifact, no missing artifact, no device fact in the environment.
    with pytest.raises(TaskContractError, match="undeclared"):
        contract.source_only_build_identity(
            candidate=candidate, observed_frontend=GOOD_FRONTEND, environment=GOOD_ENVIRONMENT,
            artifacts={**_production_artifacts(), "extra.metal": b"x"}, attempt=ATTEMPT,
        )
    with pytest.raises(TaskContractError, match="artifact set is closed"):
        contract.source_only_build_identity(
            candidate=candidate, observed_frontend=GOOD_FRONTEND, environment=GOOD_ENVIRONMENT,
            artifacts={path: data for path, data in _production_artifacts().items() if path != "kernel.metal"},
            attempt=ATTEMPT,
        )
    with pytest.raises(TaskContractError, match="environment"):
        contract.source_only_build_identity(
            candidate=candidate, observed_frontend=GOOD_FRONTEND,
            environment={**GOOD_ENVIRONMENT, "device_name": "Apple M4 Max"},
            artifacts=_production_artifacts(), attempt=ATTEMPT,
        )


def test_a_source_only_attempt_must_be_terminal_offline_and_completed():
    """@brief 尝试终态门：超时/非零/未完成/在线/有 host 都不产生 full build。"""
    candidate = _source_candidate(source=R5_CANDIDATE.read_bytes())
    artifacts = _production_artifacts()

    def identity(attempt):
        return contract.source_only_build_identity(
            candidate=candidate, observed_frontend=GOOD_FRONTEND, environment=GOOD_ENVIRONMENT,
            artifacts=artifacts, attempt=attempt,
        )

    assert identity(ATTEMPT).full_build is True
    for changed, error, message in (
        ({**ATTEMPT, "timed_out": True}, contract.TileLangCodegenError, "timed out"),
        ({**ATTEMPT, "returncode": 71}, contract.TileLangCodegenError, "exited 71"),
        ({**ATTEMPT, "status": "timeout"}, contract.TileLangCodegenError, "status"),
        ({**ATTEMPT, "online": True}, contract.TileLangContractError, "offline"),
        ({**ATTEMPT, "hosts": ["example.com"]}, contract.TileLangContractError, "offline"),
        ({**ATTEMPT, "route": "sandbox-exec -p custom"}, contract.TileLangContractError, "lower-level hmz Profile"),
        ({**ATTEMPT, "request_sha256": "not-a-digest"}, TaskContractError, "64-character lowercase hex"),
    ):
        with pytest.raises(error, match=message):
            identity(changed)


def test_garbage_or_self_reported_artifacts_cannot_mint_a_full_build():
    """@brief “有字节”不是 full build：内容契约、候选回指、前端身份与内核名逐项拒绝。"""
    source = R5_CANDIDATE.read_bytes()
    candidate = _source_candidate(source=source)

    def build(artifacts, **kwargs):
        return contract.source_only_build_identity(
            candidate=candidate, observed_frontend=GOOD_FRONTEND, environment=GOOD_ENVIRONMENT,
            artifacts=artifacts, attempt=ATTEMPT, **kwargs,
        )

    good = _production_artifacts()
    build(good, expected_kernel_names=("scale_kernel",))
    junk_host = {**good, "host_module.txt": b"arbitrary bytes, not generated host source"}
    with pytest.raises(TaskContractError, match="host source"):
        build(junk_host)
    with pytest.raises(TaskContractError, match="non-empty lowered device module"):
        build({**good, "device_ir.json": b"{}"})
    with pytest.raises(TaskContractError, match="empty"):
        build({**good, "kernel_params.json": b"[]"})
    with pytest.raises(TaskContractError, match="Metal Shading Language"):
        build({**good, "kernel.metal": b"int main(void) { return 0; }"})
    with pytest.raises(TaskContractError, match="omits the candidate-declared kernel"):
        build(good, expected_kernel_names=("not_the_kernel",))
    identity = json.loads(good["frontend_identity.json"])
    without_candidate = {**identity}
    del without_candidate["candidate"]
    with pytest.raises(TaskContractError, match="missing 'candidate'"):
        build({**good, "frontend_identity.json": json.dumps(without_candidate).encode()})
    other_candidate = {**identity, "candidate": {"source_sha256": "0" * 64, "prim_func_name": "scale_kernel"}}
    with pytest.raises(contract.TileLangIdentityDrift, match="belongs to another candidate"):
        build({**good, "frontend_identity.json": json.dumps(other_candidate).encode()})
    drifted_install = {**identity, "verified": {**identity["verified"], "package_manifest_sha256": "d" * 64}}
    with pytest.raises(contract.TileLangIdentityDrift, match="not the pinned one"):
        build({**good, "frontend_identity.json": json.dumps(drifted_install).encode()})
    dispatch = {**identity, "device_execution": {**identity["device_execution"], "dispatch_calls": 1}}
    with pytest.raises(TaskContractError, match="dispatch"):
        build({**good, "frontend_identity.json": json.dumps(dispatch).encode()})
    device_compile = {**identity, "build_options": {**identity["build_options"], "enable_device_compile": True}}
    with pytest.raises(contract.TileLangIdentityDrift, match="enable_device_compile"):
        build({**good, "frontend_identity.json": json.dumps(device_compile).encode()})


def test_codegen_timeout_nonzero_or_incomplete_artifacts_never_become_a_full_build():
    """@brief codegen 超时/非零/缺工件是 typed 失败，不产生 full build。"""
    from tcenv.track.tilelang import require_codegen_success

    require_codegen_success(returncode=0, timed_out=False, artifacts_present=True)
    for arguments, message in (
        ({"returncode": 0, "timed_out": True, "artifacts_present": False}, "timed out"),
        ({"returncode": 70, "timed_out": False, "artifacts_present": False}, "exited 70"),
        ({"returncode": 0, "timed_out": False, "artifacts_present": False}, "incomplete"),
    ):
        with pytest.raises(contract.TileLangCodegenError, match=message) as failure:
            require_codegen_success(**arguments)
        assert failure.value.code == "codegen_failed"


def test_the_archived_r5_source_only_artifacts_replay_under_the_frozen_identity():
    """@brief 归档 r5 产物只在 CPU 上重放字节/摘要；这些字节属于冻结合同。"""
    replay = contract.replay_source_only(
        result=_r5_result(),
        artifacts=_artifacts(),
        candidate_source=R5_CANDIDATE.read_bytes(),
    )
    assert replay["schema_id"] == contract.SOURCE_ONLY_REPLAY_SCHEMA_ID
    assert replay["artifact_set_sha256"] == R5_ARTIFACT_SET_SHA256
    assert replay["candidate_sha256"] == R5_CANDIDATE_SHA256
    assert [row["path"] for row in replay["artifacts"]] == list(contract.SOURCE_ONLY_ARTIFACT_PATHS)
    assert replay["device_dispatch_calls"] == 0
    assert replay["gpu_npu_executed"] is False
    assert replay["metal_correctness_or_performance_claim"] is False
    assert replay["production_ready"] is False
    assert replay["publication_allowed"] is False
    # The archived probe plan declares the same candidate bytes.
    plan = json.loads((MT0 / "probe-plan.json").read_text(encoding="utf-8"))
    assert plan["candidate"]["sha256"] == replay["candidate_sha256"]


def test_replay_rejects_tampered_bytes_facts_and_frontend_identity():
    """@brief 重放负例：字节、dispatch、online、wheel、artifact set 任一漂移都拒绝。"""
    result = _r5_result()
    artifacts = _artifacts()
    candidate = R5_CANDIDATE.read_bytes()

    tampered = dict(artifacts)
    data = bytearray(tampered["kernel.metal"])
    data[0] = data[0] ^ 0x01
    tampered["kernel.metal"] = bytes(data)
    with pytest.raises(TaskContractError, match="bytes do not match"):
        contract.replay_source_only(result=result, artifacts=tampered, candidate_source=candidate)

    lying = copy.deepcopy(result)
    lying["device_dispatch_calls"] = 1
    with pytest.raises(TaskContractError, match="device dispatches"):
        contract.replay_source_only(result=lying, artifacts=artifacts, candidate_source=candidate)

    lying = copy.deepcopy(result)
    lying["build"]["device_execution"]["dispatch_calls"] = 1
    with pytest.raises(TaskContractError, match="dispatch calls"):
        contract.replay_source_only(result=lying, artifacts=artifacts, candidate_source=candidate)

    online = copy.deepcopy(result)
    online["isolation"]["policy_online"] = True
    with pytest.raises(TaskContractError, match="not an offline build"):
        contract.replay_source_only(result=online, artifacts=artifacts, candidate_source=candidate)

    wheel = copy.deepcopy(result)
    wheel["frontend"]["wheel_sha256"] = "0" * 64
    with pytest.raises(contract.TileLangIdentityDrift, match="wheel sha256"):
        contract.replay_source_only(result=wheel, artifacts=artifacts, candidate_source=candidate)

    declared = copy.deepcopy(result)
    declared["build"]["artifact_set_sha256"] = "0" * 64
    with pytest.raises(TaskContractError, match="artifact_set_sha256 does not match"):
        contract.replay_source_only(result=declared, artifacts=artifacts, candidate_source=candidate)

    wrong_rows = copy.deepcopy(result)
    wrong_rows["build"]["artifact_rows"] = list(reversed(wrong_rows["build"]["artifact_rows"]))
    with pytest.raises(TaskContractError, match="not the frozen set"):
        contract.replay_source_only(result=wrong_rows, artifacts=artifacts, candidate_source=candidate)


def test_the_contract_module_never_imports_a_frontend_or_a_sandbox():
    """@brief 合同模块不在控制机 import TileLang/hmz/subprocess/socket，也不建第二套 Seatbelt。"""
    for relative in ("tcenv/track/tilelang.py", "tcenv/backends/metal/tilelang_adapter.py"):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"tilelang", "torch", "hmz", "subprocess", "socket", "urllib", "http"}, (relative, imported)
        text = path.read_text(encoding="utf-8")
        assert "sandbox-exec" not in text, relative
        assert "Profile(" not in text, relative


def test_an_unsupported_dsl_syntax_is_a_capability_refusal_not_a_compile_error():
    """@brief 未知 DSL 语法/入口属于 unsupported_contract_capability，不表现为编译崩溃。"""
    with pytest.raises(UnsupportedContractCapability) as unavailable:
        contract.source_only_build_identity(
            candidate=_source_candidate(),
            observed_frontend={"frontend_id": contract.FRONTEND_ID},
            environment=GOOD_ENVIRONMENT,
            artifacts=_artifacts(),
            attempt=ATTEMPT,
        )
    assert unavailable.value.code == "backend_capability_unavailable"
