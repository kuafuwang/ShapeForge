"""MT1b: versioned TrackRegistry semantics and the frozen schema-1 replay boundary.

These tests are DSL-neutral: they pin how ``(track_id, track_version)`` selection works, how a
frozen document is reconstructed instead of re-read, and what a schema-2 document may declare.
The TileLang Track is used only as the registered DSL example.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from taic.cli import _resume_track_choices, _run_track_choices
from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
from tcenv.identity import fingerprint
from tcenv.track.registry import (
    LEGACY_TRACKS_SCHEMA_VERSION,
    METAL_TRACKS_PATH,
    TRACKS_SCHEMA_VERSION,
    load_track_registry,
    registry_from_document,
)

ROOT = Path(__file__).resolve().parents[2]
#: The MT1a/P0 epoch registry, archived byte-identically in the MT1b package.
LEGACY_EPOCH = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT1b/evidence/tracks-v1-epoch.json"
LEGACY_EPOCH_SHA256 = "00c1f53c0a51f2a910c490519362e52330fabe9f68e4240b5b33c9490e567f93"
LEGACY_EPOCH_DIGEST = "9d568eec90bab265cdc5d498a7f31baeb0f1406240617e028f9a5da4dd6044cb"
MSL_TRACK_DIGESTS = {
    "msl-single": "7d19a788cb2df0b25816b9f4200d9676f8d7eec317d2411ef0a09c1b8c703af8",
    "msl-bundle": "31b10ee5b33bc6d418106cb85154f6d14d04e08fb92d5382462dbe212610b9ba",
}


def _live() -> dict:
    return json.loads(METAL_TRACKS_PATH.read_text(encoding="utf-8"))


def _msl_track(document: dict, track_id: str = "msl-single") -> dict:
    return next(track for track in document["tracks"] if track["track_id"] == track_id)


def test_the_live_registry_is_schema_2_and_keeps_both_msl_track_documents():
    """@brief live registry 是 schema 2；MSL Track 文档逐字段不变。"""
    document = _live()
    assert document["schema_version"] == TRACKS_SCHEMA_VERSION == 2
    registry = load_track_registry()
    assert registry.schema_version == 2
    assert {track.track_id: fingerprint(track.document()) for track in registry.tracks if track.track_id.startswith("msl-")} == MSL_TRACK_DIGESTS
    # MSL documents are unchanged; MT6 appends the available TileLang /3.
    assert [track.track_id for track in registry.available()] == ["msl-single", "msl-bundle", "tilelang-pipeline"]
    assert [track.track_version for track in registry.available()] == [1, 1, 3]


def test_the_frozen_schema_1_epoch_document_still_reconstructs_its_recorded_digest():
    """@brief schema-1 冻结文档只用于重建旧身份：字节与 digest 不因 MT1b 回填而变。"""
    import hashlib

    assert hashlib.sha256(LEGACY_EPOCH.read_bytes()).hexdigest() == LEGACY_EPOCH_SHA256
    frozen = registry_from_document(json.loads(LEGACY_EPOCH.read_text(encoding="utf-8")))
    assert frozen.schema_version == LEGACY_TRACKS_SCHEMA_VERSION == 1
    assert fingerprint(frozen.document()) == LEGACY_EPOCH_DIGEST
    assert {track.track_id: fingerprint(track.document()) for track in frozen.tracks} == MSL_TRACK_DIGESTS
    # Both readers agree on the MSL specs: only the registry document gained a Track.
    live = load_track_registry()
    for track_id, digest in MSL_TRACK_DIGESTS.items():
        assert fingerprint(frozen.get(track_id).document()) == digest
        assert fingerprint(live.get(track_id).document()) == digest


def test_a_new_run_may_not_select_from_a_replay_only_schema_1_file(tmp_path):
    """@brief 新 run 不得读 schema-1；冻结重建走 registry_from_document。"""
    path = tmp_path / "tracks.json"
    path.write_bytes(LEGACY_EPOCH.read_bytes())
    with pytest.raises(TaskContractError, match="frozen replay inputs only"):
        load_track_registry(path)
    # The same bytes are a legal *reconstruction* input.
    assert registry_from_document(json.loads(path.read_text(encoding="utf-8"))).schema_version == 1


def test_two_versions_of_one_track_coexist_and_selection_is_pinned():
    """@brief 同 id 多版本共存；默认选最高可用版本，显式版本精确匹配。"""
    document = _live()
    first = _msl_track(document)
    second = copy.deepcopy(first)
    second["track_version"] = 2
    second["evidence"] = ["tests/backends/metal/test_track_a4.py"]
    document["tracks"].insert(1, second)
    registry = registry_from_document(document)

    assert registry.versions("msl-single") == (1, 2)
    assert registry.get("msl-single").track_version == 2
    assert registry.get("msl-single", version=1).track_version == 1
    assert registry.get("msl-single", version=2).track_version == 2
    available_single = [track for track in registry.available() if track.track_id == "msl-single"]
    assert [track.track_version for track in available_single] == [2]
    # Registration order is preserved for the other tracks.
    assert [track.track_id for track in registry.tracks] == [
        "msl-single", "msl-single", "msl-bundle", "tilelang-pipeline", "tilelang-pipeline", "tilelang-pipeline"
    ]


def test_a_new_interface_only_version_does_not_hide_the_available_one():
    """@brief 新增 interface_only 版本不改变新 run 的选择；旧 run 的显式版本仍可取。"""
    document = _live()
    future = copy.deepcopy(_msl_track(document))
    future["track_version"] = 2
    future["status"] = "interface_only"
    future.pop("evidence", None)
    document["tracks"].append(future)
    registry = registry_from_document(document)
    assert registry.get("msl-single").track_version == 1
    assert [track.track_id for track in registry.available()] == ["msl-single", "msl-bundle", "tilelang-pipeline"]
    assert registry.get("msl-single", version=2).status == "interface_only"
    with pytest.raises(UnsupportedContractCapability, match="interface_only"):
        registry.require_available("msl-single", version=2)


def test_version_drift_is_a_typed_refusal_not_a_silent_upgrade():
    """@brief 记录/冻结的版本缺失时命名已登记版本并拒绝，绝不静默换版本。"""
    document = _live()
    document["tracks"] = [
        track for track in document["tracks"] if track["track_id"] != "msl-bundle"
    ]
    replacement = copy.deepcopy(_msl_track(document, "msl-single"))
    replacement["track_id"] = "msl-bundle"
    replacement["track_version"] = 2
    replacement["delivery"] = "candidate_bundle"
    replacement["candidate_roles"] = ["device_kernel", "host_tiling", "schedule"]
    replacement["capabilities"] = ["candidate_host_tiling", "candidate_device_kernel", "workspace", "multi_stage"]
    replacement["renderer"] = {"implementation_id": "metal.track.msl-bundle.renderer", "implementation_version": 1}
    replacement["validator"] = {"implementation_id": "metal.track.msl-bundle.validator", "implementation_version": 1}
    replacement["build_adapter"] = {"implementation_id": "metal.track.msl-bundle.build", "implementation_version": 1}
    document["tracks"].append(replacement)
    registry = registry_from_document(document)
    with pytest.raises(UnsupportedContractCapability, match=r"not registered at version 1; registered versions are \[2\]"):
        registry.get("msl-bundle", version=1)
    with pytest.raises(UnsupportedContractCapability, match="registered versions are"):
        registry.require_available("msl-bundle", version=1)


def test_a_duplicate_id_and_version_is_refused_while_a_new_version_is_not():
    """@brief 重复 (id, version) 拒绝；同 id 新版本合法。"""
    document = _live()
    document["tracks"].append(copy.deepcopy(_msl_track(document)))
    with pytest.raises(TaskContractError, match="duplicate track msl-single/1"):
        registry_from_document(document)

    document = _live()
    bumped = copy.deepcopy(_msl_track(document))
    bumped["track_version"] = 3
    document["tracks"].append(bumped)
    assert registry_from_document(document).versions("msl-single") == (1, 3)


def test_the_registry_schema_remains_a_closed_task_agnostic_set():
    """@brief schema 闭集：没有 op/shape/tiling 字段，未知 delivery/role/capability/media 全拒。"""
    document = _live()
    _msl_track(document)["op_name"] = "gemm"
    with pytest.raises(TaskContractError, match="op_name"):
        registry_from_document(document)

    for mutate, message in (
        (lambda doc: _msl_track(doc).__setitem__("delivery", "graph_executable"), "delivery"),
        (lambda doc: _msl_track(doc)["candidate_roles"].append("encoder"), "candidate_roles"),
        (lambda doc: _msl_track(doc)["capabilities"].append("command_encoder"), "capabilities"),
        (lambda doc: _msl_track(doc).__setitem__("candidate_source", {"path": "kernel.py", "media_type": "text/x-python", "role": "dsl_source"}), "candidate_source"),
        (lambda doc: _msl_track(doc)["renderer"].__setitem__("implementation_version", 0), "implementation_version"),
    ):
        changed = _live()
        mutate(changed)
        with pytest.raises(TaskContractError, match=message):
            registry_from_document(changed)


def test_a_dsl_track_cannot_disguise_itself_as_msl():
    """@brief DSL Track 不得借用 MSL 角色/能力，也不得把候选源声明成 kernel.metal。"""
    def dsl_document() -> dict:
        return _live()

    document = dsl_document()
    tilelang = next(track for track in document["tracks"] if track["track_id"] == "tilelang-pipeline")
    # Keeping dsl_source is not enough: an MSL candidate role is never borrowed.
    tilelang["candidate_roles"] = ["dsl_source", "device_kernel", "schedule"]
    with pytest.raises(TaskContractError, match="MSL candidate roles"):
        registry_from_document(document)
    document = dsl_document()
    tilelang = next(track for track in document["tracks"] if track["track_id"] == "tilelang-pipeline")
    tilelang["candidate_roles"] = ["device_kernel", "schedule"]
    with pytest.raises(TaskContractError, match="needs the dsl_source role"):
        registry_from_document(document)

    document = dsl_document()
    tilelang = next(track for track in document["tracks"] if track["track_id"] == "tilelang-pipeline")
    tilelang["capabilities"].append("single_file_msl")
    with pytest.raises(TaskContractError, match="describe MSL candidate delivery"):
        registry_from_document(document)

    for path, media_type, message in (
        ("kernel.metal", "text/x-python", "MSL source path"),
        ("kernel.py", "text/x-metal-source", "is MSL"),
        ("kernel.py", "application/x-metallib", "is MSL"),
    ):
        document = dsl_document()
        tilelang = next(track for track in document["tracks"] if track["track_id"] == "tilelang-pipeline")
        tilelang["candidate_source"] = {"path": path, "media_type": media_type, "role": "dsl_source"}
        with pytest.raises(TaskContractError, match=message):
            registry_from_document(document)
        # The same declaration is refused by the source spec itself (no registry needed).
        from tcenv.track.registry import CandidateSourceSpec

        with pytest.raises(TaskContractError, match="MSL"):
            CandidateSourceSpec(path=path, media_type=media_type, role="dsl_source")


def test_a_dsl_track_must_declare_a_source_a_frontend_and_kernel_pipeline_geometry():
    """@brief DSL Track 的必填身份与 kernel_pipeline 能力闭集。"""
    def tilelang(document: dict) -> dict:
        return next(track for track in document["tracks"] if track["track_id"] == "tilelang-pipeline")

    for mutate, message in (
        (lambda track: track.pop("candidate_source"), "must declare candidate_source"),
        (lambda track: track.pop("frontend"), "must declare its frontend identity"),
        (lambda track: track["capabilities"].remove("kernel_pipeline_geometry"), "kernel_pipeline execution only"),
        (lambda track: track.__setitem__("delivery", "single_source"), "single-source track"),
    ):
        document = _live()
        mutate(tilelang(document))
        with pytest.raises(TaskContractError, match=message):
            registry_from_document(document)


def test_an_interface_only_dsl_track_cannot_claim_device_capability_or_evidence():
    """@brief source-only 证据不能把 DSL Track 变成 available/metal_dispatch。"""
    document = _live()
    track = next(item for item in document["tracks"] if item["track_id"] == "tilelang-pipeline")
    track["capabilities"].append("metal_dispatch")
    with pytest.raises(TaskContractError, match="must not claim metal_dispatch"):
        registry_from_document(document)

    # Even a document that says ``available`` is not selectable without real dispatch evidence.
    document = _live()
    track = next(item for item in document["tracks"] if item["track_id"] == "tilelang-pipeline")
    track["status"] = "available"
    track["evidence"] = ["tests/support/golden/shared/多DSL-Track-阶段验收/MT0-tilelang-probe/acceptance.md"]
    registry = registry_from_document(document)
    spec = registry.get("tilelang-pipeline", version=1)
    assert spec.status == "available" and spec.selectable is False
    assert spec not in registry.available()
    with pytest.raises(UnsupportedContractCapability, match="no metal_dispatch capability"):
        registry.require_available("tilelang-pipeline", version=1)


def test_cli_choices_are_registry_driven_and_never_offer_an_interface_only_track():
    """@brief 新 run 的 choices 只含可用 Track；resume 解析全部已登记 id。"""
    assert _run_track_choices() == ["msl-single", "msl-bundle", "tilelang-pipeline"]
    assert _resume_track_choices() == ["msl-single", "msl-bundle", "tilelang-pipeline"]


def test_a_frozen_track_document_round_trips_and_a_tampered_one_is_refused():
    """@brief 冻结选择按登记版本重建；registry/track/版本漂移全部 typed 拒绝。"""
    from tcenv.taskcontract.runner import select_track

    live = load_track_registry()
    frozen = select_track("msl-bundle").frozen_document(presentation_sha256="0" * 64)
    rebuilt = select_track("msl-bundle", version=1, frozen=frozen)
    assert rebuilt.frozen is True
    assert fingerprint(rebuilt.track.document()) == frozen["track_digest"]
    assert rebuilt.registry_digest == frozen["registry_digest"]

    tampered = copy.deepcopy(frozen)
    tampered["track"]["status"] = "unavailable"
    with pytest.raises(TaskContractError, match="track_digest|drifted"):
        select_track("msl-bundle", version=1, frozen=tampered)

    drifted = copy.deepcopy(frozen)
    drifted["track_version"] = 2
    with pytest.raises(TaskContractError, match="not the requested version"):
        select_track("msl-bundle", version=1, frozen=drifted)

    # A frozen registry that never registered the recorded version cannot be silently upgraded.
    document = live.document()
    document["tracks"] = [track for track in document["tracks"] if track["track_version"] == 1]
    versioned = copy.deepcopy(frozen)
    versioned["registry"] = document
    versioned["registry_digest"] = fingerprint(registry_from_document(document).document())
    versioned["track_version"] = 9
    with pytest.raises(TaskContractError, match="not the requested version"):
        select_track("msl-bundle", version=1, frozen=versioned)


def test_a_forged_frozen_selection_for_an_interface_only_dsl_track_is_refused():
    """@brief 伪造的 TileLang 冻结选择在 Track 边界就被拒绝（不进入任何 build）。"""
    from tcenv.taskcontract.runner import select_track

    live = load_track_registry()
    tilelang = live.get("tilelang-pipeline", version=1)
    document = live.document()
    forged = {
        "schema_id": "tcenv.track.frozen.1",
        "schema_version": 1,
        "track_id": tilelang.track_id,
        "track_version": tilelang.track_version,
        "track": tilelang.document(),
        "track_digest": fingerprint(tilelang.document()),
        "registry": document,
        "registry_digest": fingerprint(document),
        "presentation_sha256": "0" * 64,
    }
    with pytest.raises(UnsupportedContractCapability, match="interface_only"):
        select_track(tilelang.track_id, frozen=forged)


def test_the_registered_dsl_track_is_interface_only_and_never_selectable():
    """@brief 静态登记不等于可用：interface_only 的 DSL Track 不能被选中。"""
    registry = load_track_registry()
    track = registry.get("tilelang-pipeline", version=1)
    assert track.delivery == "dsl_source_pipeline"
    assert track.candidate_source is not None and track.candidate_source.path == "kernel.py"
    assert track.frontend is not None and track.frontend.frontend_version == "0.1.15"
    assert track.selectable is False
    assert track not in registry.available()
    with pytest.raises(UnsupportedContractCapability, match="interface_only"):
        registry.require_available("tilelang-pipeline", version=1)
    with pytest.raises(UnsupportedContractCapability, match="not registered at version"):
        registry.get("tilelang-pipeline", version=4)
