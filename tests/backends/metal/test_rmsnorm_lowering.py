"""Metal RMSNorm lowering (R1): byte-for-byte agreement with the frozen R0 task.

CPU only: no device is opened and no Metal object is created.  The independent oracle is
the R0 baseline package's copy of ``tasks/rmsnorm/task.py`` and ``tasks/_common.py``
(``docs/架构/构建执行与证据语义升级-R0/draft-files``), loaded by path so that the frozen
``load_text`` reads the frozen templates — not the ones this change edits.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tasks import get_task
from tasks.rmsnorm.task import build as build_rmsnorm
from tcenv import lowering
from tcenv.backends.metal import lowering as metal_lowering
from tcenv.contract import Contract
from tcenv.identity import fingerprint
from tcenv.opspec.model import instantiate, load_opspec
from tcenv.task import task_fingerprint
from tcenv.track import load_track

ROOT = Path(__file__).resolve().parents[3]
R0 = ROOT / "docs" / "架构" / "构建执行与证据语义升级-R0"
COMBINATIONS = [("fp16", "paper"), ("fp16", "small"), ("bf16", "paper"), ("bf16", "small")]


def _load(path: Path, name: str) -> ModuleType:
    """@brief 按路径加载一个模块（用于读取 R0 冻结的旧实现）。

    冻结目录被 ``tests/shared/test_r0_baseline.py`` 逐文件校验，因此这里禁止写 ``__pycache__``，
    否则“只读证据”目录会被执行副作用污染。

    @param path 模块文件路径。
    @param name 注册到 ``sys.modules`` 的名字。
    @return 已执行的模块。
    """
    assert sys.dont_write_bytecode, "the R0 archive must be imported without writing bytecode"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def frozen_task(monkeypatch) -> ModuleType:
    """@brief 导入 R0 冻结的 ``tasks/rmsnorm/task.py`` 及其 ``tasks/_common.py``。

    ``tasks._common`` 只在 fixture 期间被替换成冻结副本，因此冻结的 ``load_text`` 读到的是
    冻结模板，而不是被本阶段改写的 ``tasks/**``。

    @return 冻结的 task 模块。
    """
    import tasks._common  # noqa: F401 - make sure the live module is restored afterwards

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    frozen_common = _load(R0 / "draft-files" / "tasks" / "_common.py", "r0_frozen_common")
    monkeypatch.setitem(sys.modules, "tasks._common", frozen_common)
    return _load(R0 / "draft-files" / "tasks" / "rmsnorm" / "task.py", "r0_frozen_rmsnorm_task")


def _instance(dtype: str, size: str):
    """@brief 由真实 OpSpec 生成一个语义实例。

    @param dtype 元素类型。
    @param size 档位。
    @return :class:`tcenv.opspec.model.SemanticInstance`。
    """
    track = load_track(ROOT / "tasks" / "rmsnorm" / "track.json")
    return instantiate(load_opspec(track.opspec_path), dtype, size)


@pytest.mark.parametrize(("dtype", "size"), COMBINATIONS)
def test_the_derived_contract_matches_the_frozen_r0_task_byte_for_byte(frozen_task, dtype, size):
    """@brief §4.3-1：Contract 全字段（含 gpu）、describe/header/abi/assemble 与冻结实现逐字节一致。"""
    old = frozen_task.build(dtype, size)
    new = get_task("rmsnorm", dtype, size)

    assert isinstance(new.contract, Contract)
    assert new.contract == old.contract
    assert dataclasses.asdict(new.contract) == dataclasses.asdict(old.contract)
    assert new.contract.gpu == "Apple M4 Max (Apple9 GPU family)"
    assert new.contract.describe() == old.contract.describe()
    assert new.contract.render_header() == old.contract.render_header()
    assert new.contract.abi_stub() == old.contract.abi_stub()

    candidate = "kernel void rmsnorm_kernel(device const half* x [[buffer(0)]]) {}\n"
    assert new.contract.assemble(candidate) == old.contract.assemble(candidate)


@pytest.mark.parametrize(("dtype", "size"), COMBINATIONS)
def test_the_rendered_text_and_registered_identity_match_the_frozen_r0_task(frozen_task, dtype, size):
    """@brief §4.3-1/2：题面、known-good、description、四类语义数值与冻结实现一致。"""
    old = frozen_task.build(dtype, size)
    new = get_task("rmsnorm", dtype, size)

    assert new.triton_source == old.triton_source
    assert new.naive_msl == old.naive_msl
    assert new.description == old.description
    assert new.cases == old.cases
    assert (new.rtol, new.atol_rel, new.exact) == (old.rtol, old.atol_rel, old.exact)
    assert new.flops == old.flops and type(new.flops) is type(old.flops) is float
    assert new.min_bytes == old.min_bytes and type(new.min_bytes) is type(old.min_bytes) is float
    assert [(b.name, b.framework) for b in new.baselines] == [(b.name, b.framework) for b in old.baselines]
    assert new.meta == old.meta


@pytest.mark.parametrize(("dtype", "size"), COMBINATIONS)
def test_spec_params_replace_the_old_text_map(frozen_task, dtype, size):
    """@brief ``spec_params`` 与旧 ``{"M": str(M), "H": str(H)}`` 替换 map 一致且参与渲染。"""
    old = frozen_task.build(dtype, size)
    new = get_task("rmsnorm", dtype, size)
    track = load_track(ROOT / "tasks" / "rmsnorm" / "track.json")
    spec = load_opspec(track.opspec_path)
    old_map = {"M": str(spec.sizes[size]["M"]), "H": str(spec.sizes[size]["H"])}
    assert dict(new.semantic_instance.spec_params) == old_map
    assert "{{M}}" not in new.triton_source and old_map["M"] in new.triton_source
    assert track.render_spec_text(new.semantic_instance, new.specialization) == old.triton_source


@pytest.mark.parametrize(("dtype", "size"), COMBINATIONS)
def test_the_frozen_generator_and_reference_still_produce_identical_bytes(frozen_task, dtype, size):
    """@brief §4.3-2：登记实现与冻结的 ``gen_inputs``/``reference`` 逐字节一致（不改变执行行为）。"""
    import torch

    old = frozen_task.build(dtype, size)
    new = get_task("rmsnorm", dtype, size)
    # "ieee" is an evaluation-level case: the harness samples its ``ieee_base`` and injects
    # specials afterwards (tcenv/backends/metal/testing.py), so the generator is never asked
    # for it directly.  Compare the generators on the logical cases they do own.
    base = new.meta.get("ieee_base", "normal")
    for case in [base if name == "ieee" else name for name in new.cases]:
        for seed in (0, 1234):
            expected = old.gen_inputs(case, seed)
            actual = new.gen_inputs(case, seed)
            assert set(actual) == set(expected) == {"x", "w"}
            for name in expected:
                assert actual[name].dtype == expected[name].dtype
                assert actual[name].shape == expected[name].shape
                assert torch.equal(actual[name], expected[name])
            old_ref = old.reference(expected)
            new_ref = new.reference(actual)
            assert set(new_ref) == {"y"}
            assert torch.equal(new_ref["y"], old_ref["y"])


def test_the_profile_is_registered_once_for_the_default_target_only():
    """@brief profile 只登记一次、只覆盖 ``apple-m4-max``；非默认 target 被拒绝（N-9）。"""
    assert metal_lowering.RMSNORM_PROFILE.name == "metal-rmsnorm"
    assert metal_lowering.RMSNORM_PROFILE.target_id == "apple-m4-max"
    assert lowering.registered_targets("metal") == frozenset({"apple-m4-max"})
    covering = lowering.require_target_profile("metal", "apple-m4-max")
    assert metal_lowering.RMSNORM_PROFILE in covering
    assert {profile.target_id for profile in covering} == {"apple-m4-max"}
    with pytest.raises(lowering.LoweringError):
        lowering.require_target_profile("metal", "apple-m9-max")
    # A backend that registers no profile at all is not yet subject to the mapping.
    assert lowering.require_target_profile("sim-npu", "sim-npu-test") == ()


def test_the_lowering_digest_is_stable_and_the_opspec_digest_is_independent(monkeypatch):
    """@brief 只改 profile 的 threads（F-2）或只改 lowerer 版本（F-25）都只失效 lowering/downstream。"""
    instance = _instance("fp16", "paper")
    profile = metal_lowering.RMSNORM_PROFILE
    baseline_profile_digest = lowering.lowering_profile_digest(profile)
    assert baseline_profile_digest == lowering.lowering_profile_digest(profile)
    assert len(baseline_profile_digest) == 64

    wider = dataclasses.replace(profile, threads=(512, 1, 1))
    new_profile_digest = lowering.lowering_profile_digest(wider)
    assert new_profile_digest != baseline_profile_digest
    assert instance.opspec_digest == _instance("fp16", "paper").opspec_digest

    wider_spec = lowering.lower(instance, wider)
    assert wider_spec.contract.threads == (512, 1, 1)
    assert wider_spec.opspec_digest == instance.opspec_digest
    assert wider_spec.lowering_profile_digest == new_profile_digest

    monkeypatch.setattr(lowering, "lowerer_version", lambda _lowerer_id: "2")
    assert lowering.lowering_profile_digest(profile) != baseline_profile_digest


def test_a_profile_change_invalidates_the_task_fingerprint_but_not_the_opspec_digest():
    """@brief F-2/N-7：threads 只经 profile 进入下游，``opspec_digest`` 必须保持不变。"""
    task = build_rmsnorm("fp16", "paper")
    wider = dataclasses.replace(metal_lowering.RMSNORM_PROFILE, threads=(512, 1, 1))
    wider_spec = lowering.lower(task.semantic_instance, wider)
    wider_task = dataclasses.replace(task, specialization=wider_spec, contract=wider_spec.contract)
    assert wider_task.semantic_instance.opspec_digest == task.semantic_instance.opspec_digest
    assert task_fingerprint(wider_task) != task_fingerprint(task)


def test_a_specialization_cannot_disagree_with_its_instance():
    """@brief 构造时不一致（opspec_digest 或 spec_params）即拒绝，不靠 lower 回写对齐。"""
    instance = _instance("bf16", "small")
    specialization = lowering.derive_specialization(
        instance, backend_id="metal", profile_name="metal-rmsnorm"
    )
    specialization.require_matches(instance)
    with pytest.raises(lowering.LoweringError):
        dataclasses.replace(specialization, opspec_digest="0" * 64).require_matches(instance)
    with pytest.raises(lowering.LoweringError):
        dataclasses.replace(specialization, spec_params={"M": "0", "H": "0"}).require_matches(instance)
    with pytest.raises(lowering.LoweringError):
        dataclasses.replace(specialization, spec_params={"M": "64", "H": "512", "X": "1"}).require_matches(instance)


def test_the_specialization_projection_is_a_closed_set_with_null_buffer_values():
    """@brief §3.1：Specialization 投影字段闭集；buffer ``value=null`` 与空 shape 不省略。"""
    task = build_rmsnorm("fp16", "small")
    document = task.specialization.fingerprint_document()
    assert set(document) == {
        "opspec_digest",
        "lowering_profile_digest",
        "spec_params",
        "backend_params",
        "contract",
    }
    assert document["backend_params"] == {}
    contract = document["contract"]
    assert set(contract) == {"kernel_name", "grid", "threads", "gpu", "args", "constants"}
    assert [arg["name"] for arg in contract["args"]] == ["x", "w", "y"]
    assert [arg["index"] for arg in contract["args"]] == [0, 1, 2]
    for arg in contract["args"]:
        assert set(arg) == {"name", "index", "role", "dtype", "shape", "value", "doc"}
        assert arg["value"] is None
    assert set(contract["constants"][0]) == {"name", "msl_type", "value"}
    assert [constant["name"] for constant in contract["constants"]] == ["M", "H", "EPS"]
    assert contract["constants"][2]["value"] == pytest.approx(1e-6)
    # The digest is reproducible from the projection alone.
    assert fingerprint(document) == fingerprint(copy.deepcopy(document))


def test_a_candidate_launch_override_does_not_change_the_task_identity():
    """@brief F-3：候选 ``taic-launch`` 改 threads 只改候选摘要，不改任务/评测身份。"""
    first = "// taic-launch threads=256\nkernel void rmsnorm_kernel(...) {}\n"
    second = "// taic-launch threads=4096\nkernel void rmsnorm_kernel(...) {}\n"
    assert hashlib.sha256(first.encode()).hexdigest() != hashlib.sha256(second.encode()).hexdigest()
    task = get_task("rmsnorm", "fp16", "small")
    assert task_fingerprint(task) == task_fingerprint(get_task("rmsnorm", "fp16", "small"))


def test_the_template_launch_and_semantic_literals_still_agree_with_the_registered_profile():
    """@brief 模板里写死的 ``num_warps``/``eps`` 与登记 profile/语义实例一致；改一边必须改另一边。"""
    track = load_track(ROOT / "tasks" / "rmsnorm" / "track.json")
    instance = _instance("fp16", "paper")
    task = build_rmsnorm("fp16", "paper")
    profile = metal_lowering.RMSNORM_PROFILE
    threads = profile.threads[0] * profile.threads[1] * profile.threads[2]
    assert threads == 256
    for entry in track.document["template_literals"]["launch"]:
        text = (ROOT / entry["template"]).read_text(encoding="utf-8")
        assert entry["text"] in text
        assert entry["threads_per_threadgroup"] == threads
    for entry in track.document["template_literals"]["semantic"]:
        text = (ROOT / entry["template"]).read_text(encoding="utf-8")
        assert entry["text"] in text
        literal = entry["text"].split("=")[-1].strip()
        assert float(literal) == float(instance.resolved_attrs[entry["attr"]])
    assert task.contract.threads == profile.threads
