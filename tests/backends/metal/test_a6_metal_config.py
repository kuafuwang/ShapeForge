"""A6: the Metal-owned configuration/data files each have one reader (backend layer).

``tests/shared/test_a6_retirement.py`` covers the backend-free registry and the one-authority
rule; this module covers the two files that live under ``backends/metal/`` and the numeric
verdict the v2 runtime shares with the v1 evaluator.  No GPU: the two live paths are exercised
in ``test_a6_retirement_gpu.py``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
METAL = ROOT / "tcenv" / "backends" / "metal"


def _source(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _module_functions(path) -> set[str]:
    tree = ast.parse(_source(path))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def test_a6_the_build_profiles_have_exactly_one_loader_and_no_python_fallback():
    """@brief ``build_profiles.json`` 是唯一数据源；Python 里没有第二份字面量或回退。"""
    from tcenv.backends.metal import source_builder

    data = ROOT / "backends" / "metal" / "build_profiles.json"
    document = json.loads(data.read_text(encoding="utf-8"))
    assert set(document) == {"schema_version", "profiles"}
    assert document["schema_version"] == 1 and document["profiles"]
    assert source_builder.BUILD_PROFILES_PATH == data
    # The loader materialises exactly ``assemble``/``math`` from the document; the prose
    # ``purpose`` stays in the data file and cannot become a second configuration value.
    assert source_builder.DIAGNOSTIC_BUILD_PROFILES == {
        name: {"assemble": entry["assemble"], "math": entry["math"]}
        for name, entry in document["profiles"].items()
    }
    assert "load_build_profiles" in _module_functions(source_builder.__file__)
    # A second loader would open the file itself; only this module may do that.
    readers = [
        path for path in sorted(METAL.glob("*.py"))
        if "build_profiles.json" in _source(path) and "read_text(" in _source(path)
    ]
    assert [path.name for path in readers] == ["source_builder.py"]
    # The retired Python literal is gone: the module-level constant is *loaded*, not written.
    source = _source(source_builder.__file__)
    assert "DIAGNOSTIC_BUILD_PROFILES: dict[str, dict[str, str]] = load_build_profiles()" in source
    assert "metal.internal-fill" not in source, "a build profile is inlined in Python again"


def test_a6_the_track_registry_file_has_exactly_one_reader():
    """@brief ``backends/metal/tracks.json`` 只由 ``tcenv.track.registry`` 读取。"""
    from tcenv.track.registry import METAL_TRACKS_PATH, load_track_registry

    assert METAL_TRACKS_PATH == ROOT / "backends" / "metal" / "tracks.json"
    registry = load_track_registry(METAL_TRACKS_PATH)
    # MT1b added the interface_only TileLang Track to the same formal registry file and MT6 added
    # its `/3` version; the two MSL delivery mechanisms and their reader stay unchanged.
    assert [track.track_id for track in registry.tracks] == [
        "msl-single", "msl-bundle", "tilelang-pipeline", "tilelang-pipeline", "tilelang-pipeline"
    ]
    readers = []
    for path in sorted((ROOT / "tcenv").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = _source(path)
        if '"tracks.json"' in text and any(opener in text for opener in ("read_text(", "load_strict_json(")):
            readers.append(path.relative_to(ROOT).as_posix())
    assert readers == ["tcenv/track/registry.py"], readers


def test_a6_the_numeric_verdict_is_one_implementation_shared_by_both_paths():
    """@brief 判决只有一个实现：v1 评测与 v2 runtime 都调 ``testing.scaled_error``。"""
    testing = METAL / "testing.py"
    assert {"scaled_error", "_floor"} <= _module_functions(testing)
    owners = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "tcenv").rglob("*.py"))
        if "__pycache__" not in path.parts and "def scaled_error(" in _source(path)
    ]
    assert owners == ["tcenv/backends/metal/testing.py"]
    runtime = _source(METAL / "v2_runtime.py")
    assert "from .testing import" in runtime and "scaled_error" in runtime
    # The v2 seam imports the same helper instead of grading with its own comparison.
    runner = _source(ROOT / "tcenv" / "taskcontract" / "runner.py")
    assert "scaled_error" not in runner, "the v2 runner must not grade numerics itself"


def test_a6_the_candidate_lifecycle_and_geometry_have_one_seam():
    """@brief 候选/几何/构建只有一套 v2 seam；没有按 op 名或 tile 的第二条路径。"""
    for relative in (
        "tcenv/track/lifecycle.py",
        "tcenv/track/geometry.py",
        "tcenv/track/bundle.py",
        "tcenv/track/envelope.py",
        "tcenv/track/delivery.py",
        "tcenv/remote/execution_geometry.py",
        "tcenv/backends/metal/track_adapter.py",
    ):
        assert (ROOT / relative).is_file(), relative
    forbidden = {"StagePartitioner", "TileConnector", "LoweringRuleRegistry", "PatternRegistry"}
    for path in sorted((ROOT / "tcenv").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        assert not (_module_functions(path) & forbidden), path
    # msl-single/msl-bundle are two delivery mechanisms (single source vs candidate bundle),
    # not two generations of the same one: the registry names one implementation per role and
    # the adapter module provides each role's callable.
    from tcenv.track.registry import METAL_TRACKS_PATH, load_track_registry

    registry = load_track_registry(METAL_TRACKS_PATH)
    adapter = _source(METAL / "track_adapter.py")
    roles = set()
    for track in registry.tracks:
        for role in ("renderer", "validator", "build_adapter"):
            spec = getattr(track, role)
            roles.add(spec.implementation_id)
            assert spec.implementation_id.startswith(f"metal.track.{track.track_id}.")
    assert len(roles) == 9, sorted(roles)
    for name in ("msl_single_geometry_resolver", "MetalTrackBuilder", "MetalTrackEvaluator"):
        assert name in adapter, name
    with pytest.raises(ModuleNotFoundError):
        __import__("tasks.generated.elementwise")


def test_a6_the_preserved_lowering_module_exports_only_names_that_exist():
    """@brief ``__all__`` 只列真实存在的 preserved 接口，``import *`` 可用（公开导入回归）。

    A6 deleted nine profiles, two lowerers and the GEMM variant factory but left their names in
    ``__all__``; the registry checks passed while ``from tcenv.backends.metal.lowering import *``
    raised ``AttributeError`` on the first retired name.  Both directions are asserted here:
    every exported name exists, and no retired identity is re-exported.
    """
    from tcenv.backends.metal import lowering as metal_lowering

    retired = {
        "ROPE_PROFILE", "GEMV_PROFILE",  # B1
        "CONV2D_PROFILE", "LION_PROFILE", "LOWERER_ID", "LOWERER_VERSION",  # B2
        "BITDELTA_PROFILE", "MAMBA2_PROFILE", "DEFAULT_FA_VARIANT",  # B3
        "FA_VARIANT_FACTORY_ID", "FA_VARIANT_FACTORY_VERSION",
        "ELEMENTWISE_LOWERER_ID", "ELEMENTWISE_LOWERER_VERSION", "DEFAULT_GEMM_VARIANT",
        "GEMM_PROFILE", "GEMM_SILU_PROFILE", "RELU_PROFILE", "SIGMOID_PROFILE", "SILU_PROFILE",
        "SWIGLU_PROFILE", "REDUCTION_SUM_PROFILE", "RMSNORM_PROFILE", "RMSNORM_PROFILE_NAME",
        "SOFTMAX_PROFILE",
        # B4
        "FLASH_ATTENTION_PROFILE", "SEM_INT_PROFILE", "FA_LOWERER_ID", "FA_LOWERER_VERSION",
        "SEM_INT_LOWERER_ID", "SEM_INT_LOWERER_VERSION", "SEM_INT_BLOCK",
    }
    exported = set(metal_lowering.__all__)
    assert not (exported & retired), sorted(exported & retired)
    missing = sorted(name for name in exported if not hasattr(metal_lowering, name))
    assert missing == [], missing
    namespace: dict[str, object] = {}
    exec("from tcenv.backends.metal.lowering import *", namespace)
    assert set(metal_lowering.__all__) <= set(namespace), set(metal_lowering.__all__) - set(namespace)
    # B4 迁走 sem_attention_odd_scale 与 sem_int 动态族之后，这个模块不再登记任何 profile：
    # 只剩后端的身份常量。（v2 的合同来自 ``tasks/<op>/opspec.json``，几何来自候选。）
    assert exported == {"BACKEND_ID", "TARGET_ID"}


def test_a6_the_opt_in_hang_candidate_uses_the_preserved_lion_abi():
    """@brief 危险 hang 测试的候选是 lion ABI（三个输入、两个输出），不是退役的 relu 形状。

    The dangerous test stays behind ``TCENV_ALLOW_GPU_HANG_TESTS=1`` and is never executed here;
    this is the compile-free construction check that the candidate can only time out, not fail
    with a compile/ABI error (which would make the recovery assertion meaningless).
    """
    from tasks import get_task

    from ...support.hang_candidate import HANG_BODY, LION_BODY, lion_hang_findings, lion_hang_source

    from ...support.v1_specimen import specimen_task

    naive = specimen_task("lion", "fp16", "small").naive_msl  # B2: lion 已迁，用归档夹具
    assert LION_BODY in naive
    hang = lion_hang_source(naive)
    assert lion_hang_findings(naive, hang) == []
    assert HANG_BODY in hang and "0xFFFFFFFFu" in hang
    for output in ("p_out[j] = half(", "m_out[j] = half("):
        assert output in hang
    for retired in ("x[j]", "y[j]"):
        assert retired not in hang
    # A relu-shaped body (the pre-A6 fixture) is what the check rejects.
    relu_shaped = naive.replace(LION_BODY, "if (j < N) { float v = float(x[j]); y[j] = half(v); }")
    assert lion_hang_findings(naive, relu_shaped) != []
