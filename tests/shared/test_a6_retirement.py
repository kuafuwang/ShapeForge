"""A6 retirement boundary: the migrated v1 implementation is gone, the preserved one is intact.

A5 moved the 16 G0 tasks to the v2 standard entry and deleted their ``task.py``/``track.json``
assembly; A6 removes what was left behind: the migrated static names in the registry, the
generated-family parser and its helpers, and the migrated per-task ``LoweringProfile`` /
lowerer / GEMM variant factory.  This module asserts *results*, not intentions:

* the v1 registry registers exactly ``preserved_v1_out_of_scope`` and refuses every migrated
  name by ``spec_version`` (typed refusal, never a fallback or a second registry);
* the deleted modules are absent (not merely unused), and no production module imports them;
* the registered Metal profiles are exactly the preserved ones and are byte-identical to the
  pre-A6 file; the deleted profile names are exactly the migrated set;
* the preserved v1 tasks still assemble with their profile, cases, tolerances, baselines,
  ``@variant`` family and dynamic ``sem_int`` family; the v1 evaluation identity anchor
  (``docs/架构/backend-phase0-baseline.json``) still matches;
* every migrated standard directory has one v2 entry, no v1 assembly marker and no launch
  fact in its document, and the Track registry stays task-agnostic;
* one authority per configuration/responsibility class (G0 ``config-inventory.json`` and
  ``upstream-map.json``): no second task contract loader, track registry, packaging builder,
  or numeric verdict.

No GPU: ``tests/backends/metal/test_a6_metal_config.py`` covers the backend-owned data files,
and ``tests/backends/metal/test_a6_retirement_gpu.py`` covers both live paths on the real Mac.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tasks import get_task, list_tasks
from tasks.semantic import FIXED_NAMES
from tcenv import lowering
from tcenv.contract_errors import TaskContractError

ROOT = Path(__file__).resolve().parents[2]
BOUNDARY = ROOT / "tests/support/golden/shared/跨硬件算子生成分工-G0/migration-boundary.json"
CONFIG_INVENTORY = ROOT / "tests/support/golden/shared/跨硬件算子生成分工-G0/config-inventory.json"
UPSTREAM_MAP = ROOT / "tests/support/golden/shared/跨硬件算子生成分工-G0/upstream-map.json"
PHASE0 = ROOT / "docs/架构/backend-phase0-baseline.json"

from tests.support.boundaries import migrated_tasks as _migrated_tasks
from tests.support.boundaries import preserved_v1 as _preserved_v1

#: The migrated set is the frozen G0 inventory *plus each later batch's own boundary file*
#: (B1 added gemv/rope).  G0's boundary document is closed evidence and is never edited.
MIGRATED: tuple[str, ...] = tuple(_migrated_tasks())
#: The static tasks still on the v1 path, derived the same way.
PRESERVED: tuple[str, ...] = tuple(name for name in _preserved_v1() if not name.startswith("sem_int "))
#: Profiles the migrated tasks used and A6 deleted (proved against the pre-A6 file below).
RETIRED_PROFILES: tuple[str, ...] = (
    "metal-gemm",
    "metal-gemm-silu",
    "metal-reduction-sum",
    "metal-relu",
    "metal-rmsnorm",
    "metal-sigmoid",
    "metal-silu",
    "metal-softmax",
    "metal-swiglu",
)
#: Registration identities that existed only for the migrated family.
RETIRED_LOWERERS: tuple[str, ...] = ("metal.elementwise", "metal.gemm")
#: B2 随 lion/conv2d 退役的 lowerer（metal.rmsnorm 的唯一使用者是 lion）。
B2_RETIRED_LOWERERS: tuple[str, ...] = ("metal.rmsnorm", "metal.conv2d")
B3_RETIRED_LOWERERS: tuple[str, ...] = ("metal.bitdelta-gemm", "metal.mamba2-chunk-scan")
RETIRED_VARIANT_FACTORY = "metal.gemm.variant"
#: Profiles B1 deleted when gemv/rope moved to the v2 standard entry (same rule as A6: the
#: migration and the profile retirement land in one change).
B1_RETIRED_PROFILES: tuple[str, ...] = ("metal-gemv", "metal-rope")
#: B2 退役的 profile（含 lion 独占的共享 lowerer metal.rmsnorm 与 conv2d 的 metal.conv2d）。
B2_RETIRED_PROFILES: tuple[str, ...] = ("metal-conv2d", "metal-lion")
#: B3 退役的 profile（当时 metal-flash-attention 仍由 sem_attention_odd_scale 共享而保留）。
B3_RETIRED_PROFILES: tuple[str, ...] = ("metal-bitdelta-gemm", "metal-mamba2-chunk-scan")
#: B4 退役的 profile：sem_attention_odd_scale 与 sem_int 动态族迁走后，注册表整体为空。
B4_RETIRED_PROFILES: tuple[str, ...] = ("metal-flash-attention", "metal-sem-int")
#: Files/directories A6 deleted (a task or helper, never a historical record).
DELETED_PATHS: tuple[str, ...] = (
    "tasks/generated",
    "tasks/_pointwise.py",
    "tasks/_matmul.py",
)
#: Productions roots scanned for dangling references to the deleted implementation.
PRODUCTION_ROOTS: tuple[str, ...] = ("tcenv", "taic", "tasks", "observe", "experiments")


def _walk(root: Path):
    """@brief 递归列出 Python 模块（跳过缓存）。

    @param root 根目录。
    @return 路径迭代器。
    """
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------- registry names


def test_a6_the_v1_registry_registers_exactly_the_preserved_tasks():
    """@brief v1 registry 只有保留题；16 个迁移名按版本被拒，不是未知名或回退。"""
    import tasks as task_package

    # B4 把最后两道 v1 题迁走：注册表里不再有任何保留题名。
    assert set(task_package.TASK_NAMES + task_package.PHASE2_TASK_NAMES + FIXED_NAMES) == set(PRESERVED) == set()
    assert FIXED_NAMES == ()
    # The two removed name lists must be gone, not emptied under the same name.
    assert not hasattr(task_package, "GENERATED_NAMES")
    assert not hasattr(task_package, "SEMANTIC_NAMES")
    assert list_tasks() == ()  # B3 后 v1 registry 不再枚举任何静态题
    assert list_tasks(True) == ()  # B3 后第二阶段静态题也全部离开 v1
    for task in MIGRATED:
        for spelling in (task, f"{task}@128x64x32t256"):
            with pytest.raises(TaskContractError) as refused:
                get_task(spelling, "fp16", "small")
            assert "v2 standard entry" in str(refused.value), spelling
    # A v2-only operator that was never in a v1 list gets the same typed refusal.
    with pytest.raises(TaskContractError):
        get_task("sigmoid_shift", "fp16", "small")
    # An invalid name is still a lookup miss, not a filesystem error.
    with pytest.raises(KeyError):
        get_task("task-" + "x" * 494, "fp16", "small")


def test_a6_the_generated_family_parser_and_its_helpers_are_gone():
    """@brief 生成族旧解析器与两个只为它/为旧 matmul 存在的 helper 已从仓库删除。"""
    for relative in DELETED_PATHS:
        assert not (ROOT / relative).exists(), relative
    assert not (ROOT / "tasks" / "generated" / "elementwise.toml").exists()
    for name in ("reference", "msl", "build_generated", "parse_expr"):
        assert not hasattr(__import__("tasks"), name)
    # No production module may import the deleted modules or name lists.
    forbidden = ("tasks.generated", "tasks._pointwise", "tasks._matmul", "GENERATED_NAMES", "SEMANTIC_NAMES")
    offenders: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in _walk(ROOT / root):
            text = _source(path)
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}:{token}")
    assert offenders == [], offenders


def test_a6_the_migrated_lowering_entries_are_gone_and_preserved_ones_are_unchanged():
    """@brief 9 份迁移 profile、2 个 lowerer 与 GEMM variant factory 退役；其余与 HEAD 逐字段相同。

    The comparison execs the pre-A6 ``tcenv/backends/metal/lowering.py`` from ``git show HEAD``
    in a subprocess: re-registering a preserved profile with different content raises, so a
    surviving name proves byte-identical content, and the *extra* names are exactly the
    deleted ones.
    """
    script = r'''
import json, subprocess, sys, types
from tcenv import lowering
from tcenv.lowering import LoweringProfile, ensure_lowerings, registered_profiles
ensure_lowerings("metal")
live = {p.name: p.to_dict() for p in registered_profiles("metal")}
lowering.register_lowerer = lambda *a, **k: None
lowering.register_variant_factory = lambda *a, **k: None
# A6 的基线提交（见 tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A5.md、A6.md）：
# 与「当前工作树」对比，而不是与「当前 HEAD」对比——HEAD 已经是 A6 之后的提交，
# 其中 lowering.py 已不含退役 profile，head_only 自然为空。
src = subprocess.check_output(["git", "show", "4a93ed2a4934fe526eedfcf4577de5dbd366e24a:tcenv/backends/metal/lowering.py"], text=True)
mod = types.ModuleType("tcenv.backends.metal._head_lowering")
mod.__package__ = "tcenv.backends.metal"
sys.modules[mod.__name__] = mod
try:
    exec(compile(src, "<HEAD lowering.py>", "exec"), mod.__dict__)
except lowering.LoweringError as exc:
    print(json.dumps({"error": str(exc)}))
    raise SystemExit(0)
head = {v.name: v.to_dict() for v in vars(mod).values() if isinstance(v, LoweringProfile)}
print(json.dumps({
    "live": sorted(live),
    "head_only": sorted(set(head) - set(live)),
    "live_only": sorted(set(live) - set(head)),
    "changed": sorted(name for name in live if name in head and head[name] != live[name]),
}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    assert result.returncode == 0, result.stderr[-4000:]
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert "error" not in payload, payload
    # B4 之后注册表整体为空：pre-A6 文件里的每个 profile 都已退役，既没有新增也没有改动。
    assert payload["live"] == [], payload
    assert payload["live_only"] == [], payload
    assert payload["changed"] == [], payload
    assert payload["head_only"] == sorted(set(RETIRED_PROFILES + B1_RETIRED_PROFILES + B2_RETIRED_PROFILES + B3_RETIRED_PROFILES) | set(B4_RETIRED_PROFILES)), payload
    # The retired identities are not registered under another name either.
    for name in RETIRED_PROFILES + B1_RETIRED_PROFILES + B2_RETIRED_PROFILES + B3_RETIRED_PROFILES:
        with pytest.raises(lowering.LoweringError):
            lowering.profile_for("metal", name)
    for lowerer_id in RETIRED_LOWERERS + B2_RETIRED_LOWERERS + B3_RETIRED_LOWERERS:
        with pytest.raises(lowering.LoweringError):
            lowering.lowerer_version(lowerer_id)
    with pytest.raises(lowering.LoweringError):
        lowering.variant_factory(RETIRED_VARIANT_FACTORY)
    # 全部 variant 族与 lowerer 都已退役（B3 退掉 flash-attention 族，B4 退掉 sem-int 与
    # 共享 FA profile 的最后一个静态题）。
    with pytest.raises(lowering.LoweringError):
        lowering.variant_factory("metal.flash-attention.variant")
    with pytest.raises(lowering.LoweringError):
        lowering.lowerer_version("metal.flash-attention")


# ------------------------------------------------------------------------ preserved v1 path
# B4 把最后两道 v1 题（sem_attention_odd_scale 与 sem_int 动态族）迁到 v2 标准入口，所以
# "8 道保留题仍由 v1 装配并带着评测身份" 这条断言已经没有主语。它守护的不变量现在由
# ``tests/backends/metal/test_task_migration_b4.py`` 承担：归档的 v1 身份与 v2 装配逐字节一致、
# 并且 v1 注册表对每个已迁题名都给出 typed 拒绝（本文件上面的两个测试也仍在验证后者）。

def test_a6_the_migrated_standard_dirs_are_v2_only_and_carry_no_launch_facts():
    """@brief 16 个标准目录只有一个 v2 entry：无 v1 装配、无逐题 profile、无启动事实。"""
    from tcenv.opspec.model import load_opspec
    from tcenv.taskcontract import inspect_standard_entry

    retired = ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py")
    for task in MIGRATED:
        directory = ROOT / "tasks" / task
        entry = inspect_standard_entry(directory)
        assert entry.is_v2 and entry.files[:2] == ("opspec.json", "reference.py"), task
        for name in retired:
            assert not (directory / name).exists(), (task, name)
        document = load_opspec(directory / "opspec.json").document
        assert document["spec_version"] == 2 and document["op_name"] == task
        # No launch/geometry facts may re-enter the task document (candidate-owned since A5).
        keys: list[str] = []

        def collect(value):
            if isinstance(value, dict):
                keys.extend(str(key).lower() for key in value)
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(document)
        assert not ({"grid", "threads", "num_warps", "tile", "block_size", "lowering_profile"} & set(keys)), task
        # A launch variant is not a second v2 contract.
        with pytest.raises(TaskContractError):
            from tcenv.taskcontract import load_task_contract

            load_task_contract(directory, dtype="fp16", size="small", variant="128x64x32t256")
    # A v1 @variant spelling never reaches the v2 loader either.
    with pytest.raises(TaskContractError):
        get_task("gemm@128x64x32t256", "fp16", "small")


def test_a6_the_track_registry_stays_task_agnostic():
    """@brief Track 只有交付机制，没有 op_name/shape/tile/网格；未知 track 被拒绝。"""
    from tcenv.track.registry import METAL_TRACKS_PATH, load_track_registry

    document = json.loads(METAL_TRACKS_PATH.read_text(encoding="utf-8"))
    # MT1b moved the registry to the versioned schema-2 epoch and registered the interface_only
    # TileLang delivery mechanism, MT6 added its `/3` version; the registry is still task-agnostic
    # (asserted below).
    assert document["schema_version"] == 2
    assert [track["track_id"] for track in document["tracks"]] == [
        "msl-single", "msl-bundle", "tilelang-pipeline", "tilelang-pipeline", "tilelang-pipeline"
    ]
    keys: set[str] = set()

    def collect(value):
        if isinstance(value, dict):
            keys.update(str(key).lower() for key in value)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(document)
    assert not ({"op_name", "task", "shape", "tile", "grid", "threads", "lowering_profile"} & keys), sorted(keys)
    registry = load_track_registry(METAL_TRACKS_PATH)
    assert tuple(registry.require_available("msl-single").track_id for _ in (0,)) == ("msl-single",)
    assert registry.require_available("msl-bundle").track_id == "msl-bundle"
    with pytest.raises(TaskContractError):
        registry.require_available("no-such-track")


# ----------------------------------------------------------------- one authority per config


def _definitions(path: Path, kinds=(ast.ClassDef, ast.FunctionDef)) -> set[str]:
    tree = ast.parse(_source(path))
    return {node.name for node in tree.body if isinstance(node, kinds) and not node.name.startswith("_")}


def test_a6_each_configuration_class_has_one_reader():
    """@brief G0 config-inventory 的每个“唯一读取器”仍然唯一，没有第二份解析/回退。

    The check is structural: the authoritative module exists, defines the reader, and no other
    production module opens the same file or defines a second class with that responsibility.
    """
    inventory = json.loads(CONFIG_INVENTORY.read_text(encoding="utf-8"))
    assert {entry["path"] for entry in inventory["entries"]} >= {
        "tasks/<op>/opspec.json",
        "tasks/<op>/track.json",
        "tasks/__init__.py and tasks/_assembly.py",
        "tcenv/backends/metal/lowering.py",
        "tcenv/lowering.py",
        "backends/metal/backend.toml",
    }
    # 1. v2 task contract: exactly one loader class + one load function.
    loader = ROOT / "tcenv" / "taskcontract" / "loader.py"
    assert {"load_task_contract", "TaskContract"} <= _definitions(loader)
    for path in _walk(ROOT / "tcenv"):
        if path == loader:
            continue
        names = _definitions(path)
        assert "load_task_contract" not in names, path
    # 2. Track registry: exactly one class, one reader of tracks.json.
    registry = ROOT / "tcenv" / "track" / "registry.py"
    assert "TrackRegistry" in _definitions(registry)
    assert "METAL_TRACKS_PATH" in _source(registry)
    for path in _walk(ROOT / "tcenv"):
        if path == registry:
            continue
        assert "TrackRegistry" not in _definitions(path), path
        # A second *reader* would open/parse the file, not merely mention it in prose.
        text = _source(path)
        if '"tracks.json"' in text or "'tracks.json'" in text:
            assert not any(opener in text for opener in ("read_text(", "load_strict_json(", "json.loads(", "open(")), path
    # 3. Task registry: v1 discovery lives in tasks/__init__.py only.
    for path in _walk(ROOT / "tasks"):
        text = _source(path)
        assert "def get_task(" not in text or path.name in ("__init__.py", "semantic.py"), path
    # 4. Lowering: tcenv/lowering.py is the unique registration/digest entry; the backend module
    #    only registers.
    assert {"profile_for", "lowering_profile_digest", "register_profile"} <= _definitions(ROOT / "tcenv" / "lowering.py")
    metal_lowering = _source(ROOT / "tcenv" / "backends" / "metal" / "lowering.py")
    assert "def register_profile(" not in metal_lowering and "_PROFILES" not in metal_lowering
    # 5. Packaging/artifact identity: exactly one builder and one registry.
    assert {"ArtifactBuilder", "ArtifactBuilderRegistry"} <= _definitions(ROOT / "tcenv" / "remote" / "artifact_builder.py")
    for path in _walk(ROOT / "tcenv"):
        if path.name == "artifact_builder.py":
            continue
        names = _definitions(path)
        assert "ArtifactBuilderRegistry" not in names and "ArtifactBuilder" not in names, path
    # 6. op_desc is an interoperability summary only, never a contract reader.
    renderer = ROOT / "tcenv" / "opspec" / "op_desc.py"
    assert "render" in _source(renderer) or "summary" in _source(renderer)
    assert "load_task_contract" not in _definitions(renderer)


def test_a6_the_upstream_map_assigns_each_responsibility_one_local_authority():
    """@brief upstream-map 的 local/port/adapt 条目落在唯一模块；没有第二套自研实现。"""
    upstream = json.loads(UPSTREAM_MAP.read_text(encoding="utf-8"))
    assert upstream["selection_policy"]["runtime_dual_read"] is False
    components = {entry["component"] for entry in upstream["entries"]}
    assert {
        "task_contract_schema_and_loader",
        "generic_candidate_and_track_registry",
        "candidate_geometry_and_canonical_bundle",
        "build_evaluation_identity_confirmation_resume",
        "metal_tracks",
        "input_generation_and_numeric_metrics",
        "snapshot_hash_and_python_packaging",
    } <= components
    # Numeric verdict: one implementation (``scaled_error``), reused by the v2 runtime.
    verdict_owners = [
        path for path in _walk(ROOT / "tcenv") if "def scaled_error(" in _source(path)
    ]
    assert [path.relative_to(ROOT).as_posix() for path in verdict_owners] == ["tcenv/backends/metal/testing.py"]
    runtime = _source(ROOT / "tcenv" / "backends" / "metal" / "v2_runtime.py")
    assert "from .testing import" in runtime and "scaled_error" in runtime
    # No KDA/CAKE2-style second path: the v2 seam is one module per responsibility.
    for relative in (
        "tcenv/taskcontract/loader.py",
        "tcenv/track/registry.py",
        "tcenv/track/bundle.py",
        "tcenv/track/geometry.py",
        "tcenv/remote/execution_geometry.py",
        "tcenv/build_projection.py",
    ):
        assert (ROOT / relative).is_file(), relative
    # The generation architecture is not duplicated: no pattern/tile/stage-rule registry.
    forbidden_defs = {"StagePartitioner", "TileConnector", "LoweringRuleRegistry", "PatternRegistry"}
    for root in ("tcenv", "taic"):
        for path in _walk(ROOT / root):
            assert not (_definitions(path) & forbidden_defs), path


def test_a6_the_historical_evidence_is_still_pinned_by_its_own_manifest():
    """@brief A6 只改被 A4 清单登记过的两个测试文件，历史文档/verifier 未被改写。"""
    acceptance = "tests/support/golden/shared/跨硬件算子生成分工-阶段验收"
    # Content pins, not ``git status``: a directory move is not a rewrite, and these
    # digests are the same bytes P0 recorded for A4/A5 and A6 recorded for G0.md.
    pinned = {
        f"{acceptance}/G0.md": "cce9fb17f87828d0c6176e1d5c06b801d50e6c1690acadf6631cd04bdde1b883",
        f"{acceptance}/A5.md": "f08e239c86c44a92b69f3e16e5d32d355ff871be8b94bbb8372ee59e037f172d",
        f"{acceptance}/A5/verify.py": "146c7d6e3e707244d46a49929625a7848d9f64cd0a8257d494dd7664e44e7b85",
        f"{acceptance}/A5/manifest.sha256": "b0668b8786952d1171fa51ae8d90fa0d50678c9ec87d484578a9a128a146d406",
        f"{acceptance}/A4.md": "ca26ca3e0320a8b2f35760ef02b91262c5eb1ab2c3f23bf33738c427a9604b20",
        f"{acceptance}/A4/verify.py": "389398e9bdb54bf132fedfdf06795b6e2e028f02a9affc8aa68c984f61dd3259",
        f"{acceptance}/A4/manifest.sha256": "4690a55e5fa8cd835c1ae364935864858ee96ecc36e6747efe148ef05b9b3bcc",
        "tests/support/golden/shared/跨硬件算子生成分工-G0/verify.py": "6890a312b0c57a40fc54d7778cdbd974789c72ccef46d5109f73d224baa3b76f",
        "tests/support/golden/shared/跨硬件算子生成分工-G0/migration-boundary.json": "057c2bff81d355d9ff4f17ba3e5a4df5092471e97fe4af75f0d325f234be4131",
    }
    for relative, digest in pinned.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    assert not (ROOT / "docs/架构/跨硬件算子生成分工-阶段验收").exists()
    assert not (ROOT / "docs/架构/跨硬件算子生成分工-G0").exists()
    # A5's v1 pre-migration performance records stay the bytes named by its own manifest.
    manifest = {}
    for line in (ROOT / acceptance / "A5/manifest.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            manifest[name] = digest
    legacy = "docs/架构/跨硬件算子生成分工-阶段验收"
    for name in (
        "perf_v1_paper.json", "perf_v1_paper_r2.json", "perf_v1_paper_r3.json",
        "perf_v1_small.json", "perf_v1_small_r2.json", "perf_v1_small_r3.json",
        "perf_gate.json",
    ):
        recorded = f"{legacy}/A5/evidence/{name}"
        path = ROOT / acceptance / "A5/evidence" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[recorded]


def test_a6_a_noisy_measurement_is_uncertain_and_never_a_successful_confirmation():
    """@brief 确认门的“安静”判定是独立负例：噪声大的测量只能得到 timing_uncertain。

    The end-to-end success node (``test_orchestrator_gpu.py``) now requires a real ``ok``
    confirmation on a quiet preserved task.  The rejection path is covered here, deterministically
    and without a GPU: the same ``tcenv.bench.timing_is_uncertain`` predicate the confirmation
    gate uses, plus the published-verdict policy that must report ``timing_uncertain``.
    """
    from tcenv.bench import QUIET_CV, timing_is_uncertain

    def report(candidate_cv: float, baseline_cv: float = 0.02, *, reps: int = 5, noisy: bool = False):
        return {
            "benchmark": {
                "candidate": {"cv": candidate_cv, "reps": reps},
                "baselines": {"mlx": {"cv": baseline_cv, "reps": reps}},
                "best_baseline": "mlx",
                "noisy": noisy,
            }
        }

    assert QUIET_CV == 0.05
    assert timing_is_uncertain(report(0.02)) is False
    # A noisy candidate, a noisy best baseline, the noisy flag, a single rep, a missing cv and a
    # NaN cv all mean "not confirmed", not "quiet because nothing was measured".
    assert timing_is_uncertain(report(QUIET_CV + 0.01)) is True
    assert timing_is_uncertain(report(0.02, QUIET_CV + 0.01)) is True
    assert timing_is_uncertain(report(0.02, noisy=True)) is True
    assert timing_is_uncertain(report(0.02, reps=1)) is True
    assert timing_is_uncertain({"benchmark": {}}) is True
    assert timing_is_uncertain(report(float("nan"))) is True
    # The published verdict for such a batch is the independent negative
    # ``tests/shared/test_r5_policy.py::test_an_explicitly_unstable_batch_stays_uncertain``
    # (A6/verify.py runs it as a fixed noise negative), so the success node never accepts it.
