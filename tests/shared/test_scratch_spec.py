"""M0: empty scratch keeps canonical geometry bytes; a real scratch changes identity.

The frozen digests below were recorded at HEAD 72a9c3a before ``tcenv/scratch`` existed.
``scratch == ()`` must reproduce them.  Historical P2/P3/P4 manifests are hashed, not edited.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tcenv.contract_errors import GeometryError
from tcenv.identity import canonical_bytes, fingerprint
from tcenv.portfolio.domain import target_generation_digest
from tcenv.track.abi import abi_limits_from_target
from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
)
from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
from tcenv.remote.execution_geometry import attach_execution_geometry
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    CandidateArtifact,
    CandidateEntry,
    CanonicalBundle,
    ExecutionGeometry,
    GeometryStage,
    TargetAbiLimits,
    candidate_from_single_source,
    derive_tensor_memory_contract,
    geometry_pair,
    resolve_geometry,
)
from tcenv.scratch import ScratchSpec, ScratchUse
from tcenv.track.delivery import parse_candidate_geometry
from taic.trust import current_prompt_sha1
from tests.support.kda_fixed_bundle import geometry_document as kda_geometry_document
from tests.support.shape_portfolio import geometry_document as portfolio_geometry_document
from tests.support.tilelang_mt2 import geometry_document as tilelang_geometry_document
from tests.support.v2_tasks import rmsnorm_workspace_opspec, write_task

ROOT = Path(__file__).resolve().parents[2]
PROMPT_EPOCH = json.loads(
    (ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/prompt-epoch.json").read_text()
)
PROMPT_SHA1 = PROMPT_EPOCH["current_prompt_sha1"]
EMPTY_DOCUMENT = (
    '{"schema_version":1,"source":"candidate","stages":[{"derived_scalars":[],"grid":[2,1,1],'
    '"kernel":"fill","threadgroup_bytes":0,"threads":[256,1,1],"when":null,"workspace":[]}]}'
)
EMPTY_RESOLUTION = (
    '{"resolution_version":1,"source":"candidate","stages":[{"active":true,"derived_scalars":[],'
    '"grid":[2,1,1],"index":0,"kernel":"fill","threadgroup_bytes":0,"threads":[256,1,1],'
    '"when":null,"workspace":[]}]}'
)
EMPTY_PAIR = "a88df7d552d8af97ed2e71cfcba3a0c8f65a2690d16063d51cb54ddece2be495"
EMPTY_BUNDLE = "fd7e056827e081c07987a4bc1d56119cb8f988ea14e1eae8d16cb7303b077a7d"
EMPTY_ARTIFACT = "580e780d83b5f95221de9d894978e8dd712ba447a8781068707cd4281778b145"
EMPTY_NATIVE = "7456db5b016d449aa58a2ed602664cd103c6418ad87388940347396597d5478b"
EMPTY_SINGLE_BUNDLE = "58382f37eed76e7c4890fa064991a84b7d9c231d8c409ebaa1bcad46da8721d6"
EMPTY_SINGLE_ARTIFACT = "f164574165efa64441ac9c75dc604c3f5d2c43a40447f2f4df6ad079262eadeb"
EMPTY_SINGLE_NATIVE = "788c09ee957ad3d7ffe5f0bf86360333d63412df9847daa6bd996f6ac139a840"
TILELANG_SINGLE = (
    '{"schema_version":1,"source":"candidate","stages":[{"derived_scalars":[],"grid":[4,1,1],'
    '"kernel":"scale_kernel_kernel","threadgroup_bytes":0,"threads":[256,1,1],"when":null,"workspace":[]}]}'
)
TILELANG_WORKLIST = (
    '{"schema_version":1,"source":"candidate","stages":[{"derived_scalars":[],"grid":[8,1,1],'
    '"kernel":"scale_kernel_kernel","threadgroup_bytes":0,"threads":[128,1,1],"when":null,'
    '"workspace":[{"name":"tmp","zero":false}]},{"derived_scalars":[],"grid":[8,1,1],'
    '"kernel":"bias_kernel_kernel","threadgroup_bytes":0,"threads":[128,1,1],"when":null,'
    '"workspace":[{"name":"tmp","zero":false}]}]}'
)
PORTFOLIO_GEOMETRY = (
    '{"schema_version":1,"source":"candidate","stages":[{"derived_scalars":[],'
    '"grid":[{"ceil_div":["N",256]},1,1],"kernel":"k","threadgroup_bytes":0,"threads":[256,1,1],'
    '"when":null,"workspace":[{"name":"scratch","zero":true}]}]}'
)
KDA_GEOMETRY = (
    '{"schema_version":1,"source":"candidate","stages":['
    '{"derived_scalars":[],"grid":["H",1,1],"kernel":"kda_piece0","threadgroup_bytes":0,"threads":[1,1,1],'
    '"when":null,"workspace":[{"name":"work_items","zero":false},{"name":"item_counts","zero":false},'
    '{"name":"handoff_states","zero":false}]},'
    '{"derived_scalars":[],"grid":["H",1,1],"kernel":"kda_piece1","threadgroup_bytes":0,"threads":[1,1,1],'
    '"when":null,"workspace":[{"name":"work_items","zero":false},{"name":"item_counts","zero":false},'
    '{"name":"handoff_states","zero":false}]},'
    '{"derived_scalars":[],"grid":["H",1,1],"kernel":"kda_piece2","threadgroup_bytes":0,"threads":[1,1,1],'
    '"when":null,"workspace":[{"name":"work_items","zero":false},{"name":"item_counts","zero":false},'
    '{"name":"handoff_states","zero":false}]}]}'
)
HISTORICAL = {
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-0/golden.json":
        "70ccf5d34e788ee1d74213c216ae758c2d9ca6a9e38893ea4c21f32f4285d86f",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-0/manifest.sha256":
        "26b87710e8c6e4e0bc65eb6117291f1fbee2613d16c3ad2381cd09498f1cad4b",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-0/mutation-report.json":
        "0ff9237c2202ad6dccae58d3fb6abd66f38fae24067f04ef9a39d51180d855e9",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-H/manifest.sha256":
        "ee757baea983a98633511f69f44920361bf4b8966a72991b6990bdafc2786cb5",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/manifest.sha256":
        "943b77478ae141d9500ba76a7085b8ca6f04b5a12f322292ec1945e626edcf4c",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P3/manifest.sha256":
        "4f5d3f7d141f67cc98b77f9000e1434e89f822271e9e7969bce352eef26d3b8b",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P4/manifest.sha256":
        "0e0dca590b9bf12042415040b106f1dd830159ea37aefff416d6973a16ed6cc1",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P4/evidence/simulation.json":
        "c9c7f32d692df4392946beb83d414da8362dbfce88817cf628890c32e46b870f",
}
LIMITS = TargetAbiLimits(max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)
PROFILE = BuildProfile(
    build_profile="metal-msl-control-host-v1",
    builder_id="metal-source-builder",
    builder_version="1",
    build_host="control_host",
    toolchain_id="apple-metal",
    toolchain_version="unknown",
    sdk_version="unknown",
    compile_profile={"language": "msl", "math": "safe", "device": "test", "gpu_family": "test"},
)


def _task(tmp_path: Path, *, workspace: bool = False):
    opspec = rmsnorm_workspace_opspec("rmsnorm") if workspace else None
    contract = load_task_contract(write_task(tmp_path / "task", opspec=opspec))
    abi = derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID))
    return contract, abi


def _pair(tmp_path: Path, geometry: ExecutionGeometry, fallback: ExecutionGeometry | None = None, **kwargs):
    contract, abi = _task(tmp_path)
    if fallback is None:
        fallback = ExecutionGeometry(
            source="fallback", stages=(GeometryStage(kernel="fb", grid=(1, 1, 1), threads=(64, 1, 1)),)
        )
    return geometry_pair(geometry, fallback, instance=contract.instance, contract=abi, limits=LIMITS, **kwargs)


def _bundle(candidate: CandidateArtifact, pair) -> CanonicalBundle:
    return CanonicalBundle(candidate=candidate, pair=pair)


def _attach(candidate: CandidateArtifact, canonical: CanonicalBundle):
    if candidate.single_source:
        entry = candidate.entries[0]
        entries = [
            BundleEntry(
                path=entry.path,
                role="SOURCE",
                media_type=entry.media_type,
                size_bytes=len(entry.data),
                sha256=entry.sha256,
                data=entry.data,
            )
        ]
        native_path = entry.path
    else:
        entries = [
            BundleEntry(
                path=item.path,
                role="AUXILIARY",
                media_type=item.media_type,
                size_bytes=len(item.data),
                sha256=item.sha256,
                data=bytes(item.data),
            )
            for item in canonical.entries
        ]
        native_path = "kernel.metal"
    native = NativeExecutable(
        backend_id="metal",
        kind="compute-pipeline",
        entry_path=native_path,
        entry_sha256=next(entry.sha256 for entry in entries if entry.path == native_path),
        descriptor={"backend": "metal", "kernel_name": "fill"},
        handle=object(),
    )
    bundle = ArtifactBundle(entries=tuple(entries), native=native)
    provenance = BuildProvenance(
        profile=PROFILE, source_sha256=candidate.candidate_sha256, artifact_set_digest=bundle.artifact_set_digest
    )
    evidence = BuildEvidence(
        mode="prebuilt",
        status="ok",
        provenance=provenance,
        artifacts=tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries),
        placement="control_host",
        native=native,
    )
    outcome = BuildOutcome(bundle=bundle, provenance=provenance, evidence=evidence)
    return attach_execution_geometry(outcome, canonical_bundle=canonical)


def _empty_geometry() -> ExecutionGeometry:
    return ExecutionGeometry(source="candidate", stages=(GeometryStage(kernel="fill", grid=(2, 1, 1), threads=(256, 1, 1)),))


def _candidate(single: bool) -> CandidateArtifact:
    if single:
        return candidate_from_single_source("kernel void fill() {}\n")
    geometry_document = {
        "schema_version": 1,
        "source": "candidate",
        "stages": [{"kernel": "fill", "grid": [2, 1, 1], "threads": [256, 1, 1]}],
    }
    return CandidateArtifact(
        track_id="msl-bundle",
        entries=(
            CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"kernel void fill(){}\n"),
            CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"// t\n"),
            CandidateEntry(
                path="host/execution_geometry.json",
                role="schedule",
                media_type="application/json",
                data=json.dumps(geometry_document, sort_keys=True).encode(),
            ),
        ),
    )


def _scratch_geometry(**overrides) -> ExecutionGeometry:
    fields = {"name": "uwin", "dtype": "fp32", "size_terms": ("M",), "init_policy": "uninitialized_poison"}
    fields.update(overrides)
    spec = ScratchSpec(**fields)
    return ExecutionGeometry(
        source="candidate",
        stages=(
            GeometryStage(
                kernel="fill",
                grid=(2, 1, 1),
                threads=(256, 1, 1),
                scratch=(ScratchUse(spec.name, "write"),),
            ),
        ),
        scratch=(spec,),
    )


def test_prompt_surface_matches_the_p2m_epoch():
    """@brief Plan 3 只让提示面前进到登记的 P2-M epoch。"""
    assert current_prompt_sha1() == PROMPT_SHA1
    assert current_prompt_sha1() != PROMPT_EPOCH["previous_prompt_sha1"]


def test_scratch_quota_is_registered_without_moving_the_generation_digest():
    """@brief apple-m4-max 登记 8 块 / 64MiB，但代际身份仍只哈希历史上限表。"""
    profile = load_target(DEFAULT_TARGET_ID)
    limits = abi_limits_from_target(profile)
    assert limits.max_scratch_buffers == 8
    assert limits.max_total_scratch_bytes == 67_108_864
    assert "max_scratch_buffers" not in limits.document()
    historical = {
        str(key): value
        for key, value in dict(profile.raw_limits).items()
        if str(key) not in {"max_scratch_buffers", "max_total_scratch_bytes"}
    }
    assert target_generation_digest(profile) == fingerprint(
        {
            "fingerprint_kind": "target-generation",
            "fingerprint_version": 1,
            "target_id": profile.id,
            "backend_id": profile.backend,
            "backend_version": profile.backend_version,
            "limits": historical,
        }
    )


def test_empty_scratch_matches_the_pre_change_canonical_bytes(tmp_path):
    """@brief scratch=() 时 document、resolution、pair、bundle、artifact、native 与 72a9c3a 一致。"""
    geometry = _empty_geometry()
    assert "scratch" not in geometry.document()
    assert canonical_bytes(geometry.document()).decode() == EMPTY_DOCUMENT
    pair = _pair(tmp_path / "empty", geometry)
    assert pair.candidate is not None
    assert "scratch" not in pair.candidate.document()
    assert "scratch" not in pair.document()["candidate_resolution"]
    assert canonical_bytes(pair.candidate.document()).decode() == EMPTY_RESOLUTION
    assert pair.identity == EMPTY_PAIR
    bundle_candidate = _candidate(False)
    bundle = _bundle(bundle_candidate, pair)
    assert bundle.bundle_digest == EMPTY_BUNDLE
    assert bundle.geometry_identity == EMPTY_PAIR
    attached = _attach(bundle_candidate, bundle)
    assert attached.artifact_set_digest == EMPTY_ARTIFACT
    assert attached.bundle.native.digest == EMPTY_NATIVE
    single = _bundle(_candidate(True), pair)
    assert single.bundle_digest == EMPTY_SINGLE_BUNDLE
    single_attached = _attach(_candidate(True), single)
    assert single_attached.artifact_set_digest == EMPTY_SINGLE_ARTIFACT
    assert single_attached.bundle.native.digest == EMPTY_SINGLE_NATIVE


def _assert_empty_scratch_projection(document: dict) -> None:
    """解析后的几何只补上省略的默认字段，不长出 scratch 通道。"""
    if "scratch" in document or any("scratch" in stage for stage in document["stages"]):
        raise AssertionError("an existing fixture declares a scratch channel")
    parsed = parse_candidate_geometry(document)
    assert parsed.scratch == ()
    emitted = parsed.document()
    assert "scratch" not in emitted
    projected = {
        "schema_version": document["schema_version"],
        "source": document["source"],
        "stages": [
            {
                "kernel": stage["kernel"],
                "grid": list(stage["grid"]),
                "threads": list(stage["threads"]),
                "threadgroup_bytes": stage.get("threadgroup_bytes", 0),
                "workspace": list(stage.get("workspace") or []),
                "when": stage.get("when"),
                "derived_scalars": list(stage.get("derived_scalars") or []),
            }
            for stage in document["stages"]
        ],
    }
    assert canonical_bytes(emitted) == canonical_bytes(projected)


def _archived_dispatches(epoch: str) -> list[list[dict]]:
    found: list[list[dict]] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            dispatch = value.get("dispatch")
            if (
                isinstance(dispatch, list)
                and dispatch
                and all(isinstance(item, dict) and "kernel" in item and "grid" in item for item in dispatch)
            ):
                found.append(dispatch)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    root = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收" / epoch
    for path in sorted(root.rglob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")))
    return found


def test_existing_fixtures_and_archived_geometries_replay_empty_scratch(tmp_path):
    """@brief 现有夹具、P2-H 几何字节、P3/P4 dispatch 和 P2-0 严格投影在 scratch=() 时不长出 scratch。"""
    import importlib.util

    from tcenv.portfolio.domain import normalize_semantic_domain_plan
    from tests.support.shape_portfolio import geometry_document as portfolio_geometry
    from tests.support.shape_portfolio import source_entries
    from tests.support.shape_portfolio_metal import (
        geometry_document as metal_geometry,
        load_p1_task,
        p1_instances,
        parent_entries,
        portfolio_document,
    )

    documents = [
        portfolio_geometry(kernel, workspace=workspace)
        for kernel in ("variant_a", "generic", "k")
        for workspace in (True, False)
    ]
    documents.extend(
        metal_geometry(kernel, workspace=workspace)
        for kernel in ("k", "variant_a")
        for workspace in (True, False)
    )
    documents.append(json.loads(tilelang_geometry_document(worklist=False)))
    documents.append(json.loads(tilelang_geometry_document(worklist=True)))
    documents.append(kda_geometry_document())
    for _path, (_role, media, data) in source_entries().items():
        if media == "application/json" and b'"stages"' in data and b'"schema_version"' in data:
            documents.append(json.loads(data))
    assert len(documents) >= 8
    for document in documents:
        _assert_empty_scratch_projection(document)

    task = load_p1_task(tmp_path / "p1")
    plan = normalize_semantic_domain_plan(task.spec, p1_instances(task), target=load_target("apple-m4-max"))
    for entry in parent_entries(portfolio_document(plan.digest)):
        if not str(entry.path).endswith("geometry.json"):
            continue
        _assert_empty_scratch_projection(json.loads(bytes(entry.data)))

    replayed = 0
    for epoch in ("P3", "P4"):
        for dispatch in _archived_dispatches(epoch):
            document = {
                "schema_version": 1,
                "source": "candidate",
                "stages": [
                    {
                        "kernel": str(item["kernel"]),
                        "grid": list(item["grid"]),
                        "threads": list(item["threads"]),
                        "threadgroup_bytes": int(item.get("threadgroup_bytes") or 0),
                        "workspace": list(item.get("workspace") or []),
                    }
                    for item in dispatch
                ],
            }
            _assert_empty_scratch_projection(document)
            replayed += 1
    assert replayed > 0

    module_path = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-0/verify.py"
    spec = importlib.util.spec_from_file_location("p2i_p20_replay", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    replay_root = tmp_path / "p20"
    replay_root.mkdir()
    module.verify_golden(module.Fixture(replay_root).positives())


def test_existing_single_bundle_tilelang_and_kda_documents_omit_scratch():
    """@brief 单核、bundle、TileLang 与 KDA 夹具在 scratch=() 时 canonical 字节不变。"""
    single = parse_candidate_geometry(json.loads(tilelang_geometry_document(worklist=False)))
    worklist = parse_candidate_geometry(json.loads(tilelang_geometry_document(worklist=True)))
    portfolio = parse_candidate_geometry(portfolio_geometry_document("k"))
    kda = parse_candidate_geometry(kda_geometry_document())
    assert canonical_bytes(single.document()).decode() == TILELANG_SINGLE
    assert canonical_bytes(worklist.document()).decode() == TILELANG_WORKLIST
    assert canonical_bytes(portfolio.document()).decode() == PORTFOLIO_GEOMETRY
    assert canonical_bytes(kda.document()).decode() == KDA_GEOMETRY
    for geometry in (single, worklist, portfolio, kda):
        assert geometry.scratch == ()
        assert "scratch" not in geometry.document()


def test_historical_manifests_and_p4_replay_are_unchanged(tmp_path):
    """@brief P2-0 / P2-G / P2-H / P3 / P4 归档字节不动，P4 模拟重放与归档逐字相同。"""
    for relative, digest in HISTORICAL.items():
        data = (ROOT / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
    from tests.support.npu_sim.shape_portfolio import encode_report, simulate

    archived = (ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P4/evidence/simulation.json").read_text(
        encoding="utf-8"
    )
    assert encode_report(simulate(tmp_path)) == archived


def test_scratch_names_quota_and_lifecycle_fail_closed(tmp_path):
    """@brief 借 y、超配额、未写先读、device_zero/host_written 同名都被拒绝。"""
    contract, abi = _task(tmp_path / "plain")

    def _resolve(geometry: ExecutionGeometry, *, task=contract, memory=abi, **kwargs):
        return resolve_geometry(geometry, instance=task.instance, contract=memory, limits=LIMITS, **kwargs)

    borrowed = _scratch_geometry(name="y")
    with pytest.raises(GeometryError, match="collides"):
        _resolve(borrowed)
    symbol = _scratch_geometry(name="M")
    with pytest.raises(GeometryError, match="collides"):
        _resolve(symbol)
    ws_contract, ws_abi = _task(tmp_path / "ws", workspace=True)
    billed = _scratch_geometry(name="scratch")
    with pytest.raises(GeometryError, match="collides"):
        _resolve(billed, task=ws_contract, memory=ws_abi)
    huge = _scratch_geometry(size_terms=("16777217",))
    with pytest.raises(GeometryError, match="max_total_scratch_bytes"):
        _resolve(huge)
    unread = ExecutionGeometry(
        source="candidate",
        stages=(
            GeometryStage(
                kernel="fill",
                grid=(2, 1, 1),
                threads=(256, 1, 1),
                scratch=(ScratchUse("uwin", "read"),),
            ),
        ),
        scratch=(ScratchSpec("uwin", "fp32", ("M",), "uninitialized_poison"),),
    )
    with pytest.raises(GeometryError, match="first used as read"):
        _resolve(unread)
    both = ExecutionGeometry(
        source="candidate",
        stages=(
            GeometryStage(
                kernel="fill",
                grid=(2, 1, 1),
                threads=(256, 1, 1),
                scratch=(ScratchUse("uwin", "write"),),
            ),
        ),
        scratch=(
            ScratchSpec("uwin", "fp32", ("M",), "device_zero"),
            ScratchSpec("uwin", "fp32", ("M",), "host_written"),
        ),
    )
    with pytest.raises(GeometryError, match="mutually exclusive"):
        _resolve(both)
    hosted = _scratch_geometry(init_policy="host_written")
    with pytest.raises(GeometryError, match="host_written requires"):
        _resolve(hosted)


def test_nonempty_scratch_is_reproducible_and_invalidates_the_old_cache(tmp_path):
    """@brief 同一份 scratch 可复现；改 dtype/形状/init_policy 后旧 bundle/native 缓存不得命中。"""
    contract, abi = _task(tmp_path / "id")
    first = _scratch_geometry()
    second = _scratch_geometry()
    pair_a = geometry_pair(first, instance=contract.instance, contract=abi, limits=LIMITS)
    pair_b = geometry_pair(second, instance=contract.instance, contract=abi, limits=LIMITS)
    assert pair_a.candidate is not None and pair_b.candidate is not None
    assert pair_a.identity == pair_b.identity
    assert "scratch" in pair_a.candidate.document()
    assert pair_a.candidate.scratch[0].nbytes == int(contract.instance.symbols["M"]) * 4
    candidate = _candidate(False)
    bundle_a = CanonicalBundle(candidate=candidate, pair=pair_a)
    bundle_b = CanonicalBundle(candidate=candidate, pair=pair_b)
    assert bundle_a.bundle_digest == bundle_b.bundle_digest
    attached_a = _attach(candidate, bundle_a)
    attached_b = _attach(candidate, bundle_b)
    assert attached_a.artifact_set_digest == attached_b.artifact_set_digest
    assert attached_a.bundle.native.digest == attached_b.bundle.native.digest
    cache = {
        pair_a.identity: "geometry",
        bundle_a.bundle_digest: "bundle",
        attached_a.artifact_set_digest: "artifact",
        attached_a.bundle.native.digest: "native",
    }
    empty = _pair(tmp_path / "base", _empty_geometry())
    assert empty.identity == EMPTY_PAIR
    assert empty.identity not in cache
    for mutated in (
        _scratch_geometry(dtype="fp16"),
        _scratch_geometry(size_terms=("H",)),
        _scratch_geometry(init_policy="device_zero"),
    ):
        pair = geometry_pair(mutated, instance=contract.instance, contract=abi, limits=LIMITS)
        bundle = CanonicalBundle(candidate=candidate, pair=pair)
        attached = _attach(candidate, bundle)
        assert pair.identity != pair_a.identity
        assert bundle.bundle_digest != bundle_a.bundle_digest
        assert attached.artifact_set_digest != attached_a.artifact_set_digest
        assert attached.bundle.native.digest != attached_a.bundle.native.digest
        assert pair.identity not in cache
        assert bundle.bundle_digest not in cache
        assert attached.artifact_set_digest not in cache
        assert attached.bundle.native.digest not in cache
        assert pair.candidate is not None and pair_a.candidate is not None
        assert pair.candidate.document()["scratch"] != pair_a.candidate.document()["scratch"]
