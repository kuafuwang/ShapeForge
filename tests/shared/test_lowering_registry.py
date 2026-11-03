"""Registered lowering profiles and the ``target_id → LoweringProfile`` rule (N-9, plan §4.2).

CPU only.  The registry is reached through ``tcenv.lowering``; this module never imports a
backend package itself (the metal lowering module is imported on demand by the registry),
so the backend-free test rule still holds.
"""

from __future__ import annotations
from tests.support.v1_tasks import V1_TASK, v1_kernel_name

from pathlib import Path

import pytest

from tcenv import lowering
from tcenv.target import TargetError, TargetProfile

ROOT = Path(__file__).resolve().parents[2]


def _fake_target(tmp_path: Path, target_id: str, backend: str) -> TargetProfile:
    """@brief 造一个不落盘的 target 档案。

    @param tmp_path pytest 临时目录。
    @param target_id 目标 id。
    @param backend 后端 id。
    @return :class:`TargetProfile`。
    """
    return TargetProfile(
        id=target_id,
        backend=backend,
        status="active",
        display_name="Unregistered target",
        match_name_contains="",
        match_family="",
        calibration_device_name="Unregistered target",
        raw_limits={"max_buffer_index": 0, "max_threads_per_threadgroup": 1, "max_threadgroup_memory_bytes": 1, "counter_sample_us": 0},
        root=tmp_path,
        language="metal",
        file_ext=".metal",
        backend_version="1",
    )


def test_only_the_registered_target_and_profile_resolve():
    """@brief R1 只登记 ``apple-m4-max``；未登记的 target 必须拒绝而不是回落 default profile。

    A5 把 16 道 G0 题迁到 v2 标准入口，A6 删除了它们在这里的 9 份 profile（普通算子的
    metal-rmsnorm/-softmax/-reduction-sum/-relu/-sigmoid/-silu/-swiglu/-gemm/-gemm-silu）。
    剩下的必须恰好是 ``preserved_v1_out_of_scope``：v1 registry 只有这些 profile 可读。
    """
    assert lowering.registered_targets("metal") == frozenset({"apple-m4-max"})
    profiles = lowering.require_target_profile("metal", "apple-m4-max")
    assert {profile.name for profile in profiles} == {
        "metal-conv2d",
        "metal-flash-attention",
        "metal-gemv",
        "metal-bitdelta-gemm",
        "metal-mamba2-chunk-scan",
        "metal-lion",
        "metal-rope",
        "metal-sem-int",
    }
    assert {profile.target_id for profile in profiles} == {"apple-m4-max"}
    with pytest.raises(lowering.LoweringError):
        lowering.require_target_profile("metal", "apple-m9-max")
    # A backend that registers no profile has not opted into the mapping yet.
    assert lowering.require_target_profile("no-such-backend", "whatever") == ()


def test_a_migrated_profile_name_and_lowerer_are_gone_not_relocated():
    """@brief 已迁移题的 profile、lowerer 与 GEMM variant factory 都不在登记表里。

    只检查结果：名字未登记、lowerer 未登记、``gemm@…`` 不能再派生。没有第二个读取器或回退。
    """
    for name in (
        "metal-rmsnorm", "metal-softmax", "metal-reduction-sum", "metal-relu",
        "metal-sigmoid", "metal-silu", "metal-swiglu", "metal-gemm", "metal-gemm-silu",
    ):
        with pytest.raises(lowering.LoweringError):
            lowering.profile_for("metal", name)
    for lowerer_id in ("metal.elementwise", "metal.gemm"):
        with pytest.raises(lowering.LoweringError):
            lowering.lowerer_version(lowerer_id)
    with pytest.raises(lowering.LoweringError):
        lowering.variant_factory("metal.gemm.variant")
    # The preserved variant family is untouched: flash_attention still derives its launch.
    assert lowering.lowerer_version("metal.flash-attention")
    assert lowering.variant_factory("metal.flash-attention.variant").version


def test_an_unknown_profile_name_does_not_fall_back():
    """@brief 未知 profile 名不是“用默认的那份”。"""
    with pytest.raises(lowering.LoweringError):
        lowering.profile_for("metal", "metal-not-registered")
    with pytest.raises(lowering.LoweringError):
        lowering.profile_for("no-such-backend", "metal-rmsnorm")


def test_the_orchestrator_refuses_a_target_without_a_registered_profile(tmp_path, monkeypatch):
    """@brief N-9：装配前就拒绝，且不产生任何 run 目录/请求。"""
    import taic.orchestrator as orchestrator_module
    from tests.support.orchestrator import make_orch

    monkeypatch.setattr(
        "tcenv.target.require_active",
        lambda _target_id=None: _fake_target(tmp_path, "apple-m9-max", "metal"),
    )
    with pytest.raises(orchestrator_module.RunStartError) as excinfo:
        make_orch(tmp_path, script=[], calls=[])
    assert "no registered LoweringProfile" in str(excinfo.value)
    assert not (tmp_path / "run").exists()


def test_the_cli_refuses_a_target_without_a_registered_profile(tmp_path, monkeypatch, capsys):
    """@brief ``tcenv show`` 对未登记 target 返回 5，而不是展示 default profile 的契约。"""
    from tcenv import cli

    monkeypatch.setattr(
        "tcenv.target.require_active",
        lambda _target_id=None: _fake_target(tmp_path, "apple-m9-max", "metal"),
    )
    assert cli.main(["show", V1_TASK, "--target", "apple-m9-max"]) == 5
    assert "no registered LoweringProfile" in capsys.readouterr().err


def test_the_default_target_still_resolves_for_the_cli(capsys):
    """@brief 默认 target 仍可展示（回归面：R1 不改变默认目标的行为）。"""
    from tcenv import cli

    assert cli.main(["show", V1_TASK]) == 0
    out = capsys.readouterr().out
    assert "Apple M4 Max (Apple9 GPU family)" in out
    assert v1_kernel_name() in out
