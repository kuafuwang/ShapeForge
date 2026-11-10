"""MT1b: the production source-only build seam (bounded attempt, real observation, verification).

Nothing here is a pure-helper substitute: the builder composes one closed ``AttemptSpec``, runs it
through the existing lower-level hmz Profile, reads the declared artifacts back, verifies their
content and derives the identity from independent inputs.  When the host denies nested
``sandbox-exec`` the honest outcome is the typed capability gap — and the attempt must still have
been really composed and launched, which this file asserts from its own record.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tcenv.backends.metal import tilelang_build
from tcenv.backends.metal.tilelang_adapter import (
    TileLangSourceOnlyBuilder,
    adapter_services,
    expected_build_identity,
)
from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
from tcenv.track import CandidateArtifact, CandidateEntry, load_track_registry
from tcenv.track import tilelang as contract

ROOT = Path(__file__).resolve().parents[3]
MT0 = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-tilelang-probe"
#: The successful MT0-r5 program (the one whose source-only artifacts are archived).
R5_CANDIDATE = MT0 / "evidence/failure-inputs/r5-candidate.py"
#: The historical r3 *failure* input.  It stays byte-identical evidence: only the two textual
#: differences below separate it from r5, and the archive itself is never edited.
R3_HISTORICAL = MT0 / "evidence/failure-inputs/r3-candidate.py"
GOOD_ENVIRONMENT = {
    "python_version": "3.13.1",
    "python_implementation": "CPython",
    "platform": "macOS-26.7.1-arm64-arm-64bit-Mach-O",
    "machine": "arm64",
}


def _spec():
    return load_track_registry().get(contract.TILELANG_TRACK_ID, version=1)


def _candidate(*, grid: int = 4, source: bytes | None = None) -> CandidateArtifact:
    geometry = {
        "schema_version": 1,
        "source": "candidate",
        "stages": [{"kernel": "scale_kernel", "grid": [grid, 1, 1], "threads": [256, 1, 1]}],
    }
    return CandidateArtifact(
        track_id=contract.TILELANG_TRACK_ID,
        entries=(
            CandidateEntry(
                path=contract.GEOMETRY_PATH,
                role=contract.GEOMETRY_ROLE,
                media_type=contract.GEOMETRY_MEDIA_TYPE,
                data=json.dumps(geometry).encode(),
            ),
            CandidateEntry(
                path=contract.SOURCE_PATH,
                role=contract.SOURCE_ROLE,
                media_type=contract.SOURCE_MEDIA_TYPE,
                data=valid_tilelang_source() if source is None else source,
            ),
        ),
    )


def valid_tilelang_source() -> bytes:
    """@brief The MT0-r5 valid program, byte for byte (never the r3 parser-failure fixture).

    r3 is preserved evidence: its ``tilelang.language.tir`` import no longer parses with the
    pinned frontend.  The relation is asserted here instead of editing the archive — repairing
    those two lines yields exactly the r5 program whose source-only artifacts are archived.
    """
    source = R5_CANDIDATE.read_bytes()
    repaired_r3 = R3_HISTORICAL.read_bytes().replace(
        b"from tilelang.language.tir import prim_func\n", b""
    ).replace(b"@prim_func", b"@T.prim_func")
    assert repaired_r3 == source, "the historical r3 fixture drifted, or it is no longer r5 minus the old import"
    return source


def _dsl_source_sha256() -> str:
    return hashlib.sha256(valid_tilelang_source()).hexdigest()


def _request(*, track=None, candidate=None) -> SimpleNamespace:
    spec = _spec() if track is None else track
    return SimpleNamespace(
        track_selection=SimpleNamespace(track=spec),
        candidate=_candidate() if candidate is None else candidate,
        selection=SimpleNamespace(
            geometry=SimpleNamespace(
                source="candidate", stages=(SimpleNamespace(kernel="scale_kernel"),)
            ),
        ),
    )


def test_a_real_attempt_is_composed_and_launched_through_the_hmz_profile(tmp_path):
    """@brief 生产构建真的拼装并启动了闭环尝试；成功则核验工件，能力缺失则断言真实尝试记录。"""
    build = tilelang_build.TileLangSourceOnlyBuild(_spec(), attempt_base=tmp_path)
    try:
        outcome = build.build(_candidate(), expected_kernel_names=("scale_kernel",))
    except contract.TileLangFrontendUnavailable as gap:
        assert gap.code == "backend_capability_unavailable"
        roots = sorted(tmp_path.glob("shapeforge-tilelang-*"))
        assert roots, "the builder must have created a real attempt root"
        root = roots[0]
        request = json.loads((root / "hmz-profile-request.json").read_text(encoding="utf-8"))
        assert request["online"] is False and request["hosts"] == []
        assert request["profile"]["read"] and request["profile"]["write"]
        # The frozen harness entry and the projected candidate really are the attempt inputs.
        assert (root / "input" / tilelang_build.FRONTEND_ENTRY_NAME).read_bytes() == (
            tilelang_build.FRONTEND_ENTRY_SOURCE.encode("utf-8")
        )
        assert (root / "input" / tilelang_build.CANDIDATE_NAME).read_bytes() == valid_tilelang_source()
        pinned = json.loads((root / "input" / tilelang_build.PINNED_NAME).read_text(encoding="utf-8"))
        assert pinned["tilelang_package_manifest_sha256"] == contract.FRONTEND_PACKAGE_MANIFEST_SHA256
        assert pinned["candidate_source_sha256"] == _dsl_source_sha256()
        assert (root / "input" / tilelang_build.FRONTEND_ENTRY_NAME).stat().st_size > 0
        report = json.loads((root / "driver-report.json").read_text(encoding="utf-8"))
        assert report["online"] is False and report["hosts"] == []
        assert report["lower_level_route"].startswith("hmz.")
        assert report["internal_fence_or_wrap_used"] is False
        assert (root / "limits.json").exists()
    else:
        document = outcome.document()
        assert document["full_build"] is True
        assert document["publication_allowed"] is False
        assert document["device_dispatch_calls"] == 0
        assert document["identity"]["candidate_sha256"] == _candidate().candidate_sha256
        assert document["identity"]["dsl_source_sha256"] == _dsl_source_sha256()
        assert document["identity"]["attempt"]["status"] == "completed"
        assert document["identity"]["attempt"]["online"] is False
        assert outcome.entry_sha256 == tilelang_build.FRONTEND_ENTRY_SHA256
        assert [row["path"] for row in document["identity"]["artifacts"]] == list(
            contract.SOURCE_ONLY_ARTIFACT_PATHS
        )


def test_the_staged_frontend_is_observed_byte_exactly_or_reported_as_a_capability_gap():
    """@brief 控制机对 staged 安装做真实字节观测；缺失是 typed 能力缺口，漂移是 IdentityDrift。"""
    try:
        frontend = tilelang_build.observe_staged_frontend()
    except contract.TileLangFrontendUnavailable as gap:
        assert gap.code == "backend_capability_unavailable"
        assert tilelang_build.STAGING_ENV in str(gap) or "staged" in str(gap)
    else:
        observation = frontend.observation()
        assert observation["package_manifest_sha256"] == contract.FRONTEND_PACKAGE_MANIFEST_SHA256
        assert observation["package_files"] == contract.FRONTEND_PACKAGE_FILES
        assert frontend.package.is_dir() and frontend.python.is_file()
        assert contract.verify_frontend_observation(observation) == observation
        environment = tilelang_build.observe_staged_environment(frontend)
        assert set(environment) == set(contract.FRONTEND_ENVIRONMENT_KEYS)
        assert environment["python_implementation"] == "CPython"


def test_a_different_install_is_identity_drift_not_a_valid_looking_observation(tmp_path):
    """@brief 伪造的 staged 安装（字节不同、摘要格式无所谓）必须判 IdentityDrift。"""
    root = tmp_path / "staged"
    site = root / "venv" / "lib" / "python3.13" / "site-packages"
    package = site / "tilelang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"# not the pinned wheel\n")
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "venv" / "bin" / "python3").symlink_to(sys.executable)
    dist_info = site / f"tilelang-{contract.FRONTEND_VERSION}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_bytes(b"Version: 0.1.15\n")
    (dist_info / "RECORD").write_bytes(b"\n")
    with pytest.raises(contract.TileLangIdentityDrift, match="pinned install"):
        tilelang_build.observe_staged_frontend(root)


def test_a_missing_staged_install_is_a_typed_capability_gap(tmp_path):
    """@brief 没有 staged 前端：backend_capability_unavailable，不是 traceback 也不是无隔离回退。"""
    with pytest.raises(contract.TileLangFrontendUnavailable) as gap:
        tilelang_build.observe_staged_frontend(tmp_path / "absent")
    assert gap.value.code == "backend_capability_unavailable"


def test_the_adapter_wires_the_source_only_builder_and_keeps_metal_typed_unavailable(tmp_path):
    """@brief 生产服务表里的 builder 是真构建器；Metal build 仍是 typed 能力拒绝。"""
    spec = _spec()
    services = adapter_services(spec)
    assert set(services) == {
        "validator", "renderer", "builder", "geometry_resolver", "evaluator", "confirmation", "guidance",
    }
    builder = services["builder"]()
    assert isinstance(builder, TileLangSourceOnlyBuilder)
    for name in ("evaluator", "confirmation", "guidance"):
        with pytest.raises(UnsupportedContractCapability, match=name):
            services[name]()
    # Metal build is a typed refusal, never a silent option flip.
    with pytest.raises(contract.TileLangBuildModeUnavailable) as refusal:
        TileLangSourceOnlyBuilder(spec, attempt_base=tmp_path).build_metal(_request())
    assert refusal.value.code == "unsupported_contract_capability"
    # A request that selects another Track is refused before any attempt is created.
    other = load_track_registry().get("msl-single")
    with pytest.raises(TaskContractError, match="not the frozen TileLang Track"):
        TileLangSourceOnlyBuilder(spec, attempt_base=tmp_path).prepare(_request(track=other))
    assert list(tmp_path.iterdir()) == []


def test_the_expected_subject_binds_the_candidate_manifest_not_a_self_report(tmp_path):
    """@brief expected 只由候选字节 + 冻结前端/options + 环境重建；几何不同即不同。"""
    first = expected_build_identity(_candidate(grid=1), environment=GOOD_ENVIRONMENT)
    second = expected_build_identity(_candidate(grid=999), environment=GOOD_ENVIRONMENT)
    assert first.candidate_sha256 != second.candidate_sha256
    assert first.cache_key != second.cache_key
    assert first.full_build is False and first.artifacts == () and first.attempt == {}
    assert first.frontend["package_manifest_sha256"] == contract.FRONTEND_PACKAGE_MANIFEST_SHA256
    same = expected_build_identity(_candidate(grid=1), environment=GOOD_ENVIRONMENT)
    assert same.cache_key == first.cache_key and same.digest == first.digest
    with pytest.raises(TaskContractError, match="environment"):
        expected_build_identity(_candidate(), environment={**GOOD_ENVIRONMENT, "device": "MTLDevice"})


def test_the_build_module_never_imports_the_frontend_on_the_control_host():
    """@brief 构建模块只在围栏内的 entry 文本里 import tilelang；控制机路径不 import 前端。"""
    for relative in ("tcenv/backends/metal/tilelang_build.py", "tcenv/backends/metal/tilelang_adapter.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        top_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".")[0])
        assert "tilelang" not in top_level and "hmz" not in top_level, relative
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "Profile(" not in text, relative
        assert "sandbox-exec" not in text, relative
        assert "wrap._seatbelted" not in text, relative
    # The in-fence entry is harness-owned text and does import the frontend: that is the point.
    assert "import tilelang" in tilelang_build.FRONTEND_ENTRY_SOURCE
    assert "enable_device_compile=False" in tilelang_build.FRONTEND_ENTRY_SOURCE
    assert tilelang_build.FRONTEND_ENTRY_SHA256 == hashlib.sha256(
        tilelang_build.FRONTEND_ENTRY_SOURCE.encode("utf-8")
    ).hexdigest()
