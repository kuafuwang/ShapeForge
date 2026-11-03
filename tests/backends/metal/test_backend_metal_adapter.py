"""Phase 1 adapter delegates to the existing Metal implementation."""

from __future__ import annotations

from tasks import get_task

from tcenv.backends.registry import load_backend
from tests.support.v1_tasks import V1_TASK, v1_kernel_name

V1_KERNEL = v1_kernel_name()


def test_parent_side_screen_and_launch_parse_delegate():
    """@brief 验证父进程侧的 screen 与 parse_launch 直接转调 Metal 后端实现，行为与原实现一致。
    """
    backend = load_backend("metal")
    task = get_task(V1_TASK, "fp16", "paper")
    assert backend.screen("kernel void k() {}", task) == []
    assert backend.screen("kernel void k() { for (;;) {} }", task)
    assert backend.parse_launch("kernel void k() {}", task.contract) is None


def test_metal_declaration_hook_preserves_taic_launch_extraction():
    """The new declaration hook must not disable Metal's launch-block priority."""
    from taic.kernel_agent import kernel_text

    backend = load_backend("metal")
    source = "/* taic-launch\nlaunch other grid=1,1,1 threads=32,1,1\n*/\nkernel void other() {}\n"
    pattern = backend.source_name_pattern("relu_kernel")
    assert kernel_text(source, "relu_kernel", source_fence="metal", source_name_pattern=pattern) == source


def test_check_abi_pipeline_receives_every_kernel_and_buffer_limit(monkeypatch):
    """@brief 验证 check_abi 把每个 kernel 按名传入 ABI 流水线，并透传缓冲区上限与 spec。

    @param monkeypatch pytest 的 monkeypatch，用于替换 metal 的 ABI 流水线。
    """
    backend = load_backend("metal")

    class Kernel:
        def __init__(self, name):
            self.name = name

    seen = {}

    def fake(contract, by_name, spec, **kwargs):
        seen.update({"names": sorted(by_name), "limit": kwargs.get("max_buffer_index"), "spec": spec})
        return []

    monkeypatch.setattr("tcenv.backends.metal.abi.check_pipeline", fake)
    spec = object()
    assert backend.check_abi(
        object(), [Kernel("a"), Kernel("b")],
        max_tg_memory=32768, max_buffer_index=7, target_id="apple-m4-max", spec=spec,
    ) == []
    assert seen == {"names": ["a", "b"], "limit": 7, "spec": spec}


def test_generic_abi_hook_keeps_live_memory_and_profile_buffer_limits(monkeypatch):
    """The refactored call shape must not swap live and profile limit sources."""
    from tcenv.target import load_target

    backend = load_backend("metal")
    seen = {}

    class Kernel:
        name = "relu_kernel"

    def fake(contract, by_name, spec, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("tcenv.backends.metal.abi.check_pipeline", fake)
    backend.check_abi(
        object(), [Kernel()], target=load_target("apple-m4-max"),
        device_info={"max_threadgroup_memory_bytes": 16384}, spec=object(),
    )
    assert seen["max_tg_memory"] == 16384
    assert seen["max_buffer_index"] == 30
    assert seen["target_id"] == "apple-m4-max"


def test_generic_abi_and_launch_hooks_keep_metal_limits():
    """Generic hook inputs still reach the original Metal limit checks."""
    import pytest

    from tcenv.backends.base import LaunchRejected
    from tcenv.target import load_target

    backend = load_backend("metal")
    task = get_task(V1_TASK, "fp16", "paper")
    target = load_target("apple-m4-max")
    info = {
        "max_threads_per_threadgroup_xyz": (1024, 1024, 1024),
        "max_threads_per_threadgroup": 1024,
        "max_threadgroup_memory_bytes": 32768,
        "recommended_working_set_bytes": 1 << 30,
        "max_buffer_length": 1 << 30,
    }
    accepted = backend.parse_launch("/* taic-launch\nlaunch relu_kernel grid=1,1,1 threads=32,1,1\n*/", task.contract)
    backend.validate_launch(accepted, source="", contract=task.contract, task=task, target=target, device_info=info, require_contract_launch=False)
    rejected = backend.parse_launch("/* taic-launch\nlaunch relu_kernel grid=1,1,1 threads=2048,1,1\n*/", task.contract)
    with pytest.raises(LaunchRejected, match="device limit 1024"):
        backend.validate_launch(rejected, source="", contract=task.contract, task=task, target=target, device_info=info, require_contract_launch=False)


def test_contract_rendering_is_delegated_without_changing_text():
    """@brief 验证 assemble、describe 与 abi_stub 都转调合约自身实现，渲染文本逐字不变。
    """
    backend = load_backend("metal")
    contract = get_task(V1_TASK, "fp16", "paper").contract
    source = "kernel void relu_kernel() {}\n"
    full, header = backend.assemble(contract, source)
    assert full == contract.assemble(source)[0]
    assert header == contract.assemble(source)[1]
    assert backend.describe(contract) == contract.describe()
    assert backend.abi_stub(contract) == contract.abi_stub()


def test_moved_tooling_keeps_the_repository_root():
    """@brief 验证搬入后端目录的 triton_baseline 与 xctrace 仍把仓库根与缓存目录指向原位置。
    """
    from pathlib import Path

    from tcenv.backends.metal import triton_baseline, xctrace

    repo = Path(__file__).resolve().parents[3]
    assert triton_baseline.REPO_ROOT == repo
    assert xctrace.REPO_ROOT == repo
    assert triton_baseline.CACHE_DIR == repo / "runs" / "triton_baselines"


def test_rotation_sets_are_delegated():
    """@brief 验证后端的旋转集查询直接转调 Metal 基线模块的 rotation_count。
    """
    backend = load_backend("metal")
    from tcenv.backends.metal.baselines import rotation_count

    task = get_task("flash_attention", "fp16", "paper")
    assert backend.rotation_sets(task) == rotation_count(task)


def test_metal_metadata_drives_task_rendering_names():
    """@brief 验证显示名、源码围栏与参考技能名都由 Metal 元数据决定。
    """
    backend = load_backend("metal")
    assert backend.display_name == "Metal"
    assert backend.source_fence == "metal"
    assert backend.reference_skill == "msl_reference"
    assert backend.skill_names[2] == "msl_reference"


def test_submission_name_accepts_a_backend_extension():
    """@brief 验证提交文件名会拼上后端给出的扩展名，而不是写死 .metal。
    """
    from taic.kernel_agent import submission_name

    assert submission_name("relu", "fp16", "small", "init", ".metal") == "relu-fp16-small-init.metal"
    assert submission_name("relu", "fp16", "small", "init", ".x") == "relu-fp16-small-init.x"


def test_phase4_removed_the_top_level_metal_shims():
    """@brief 验证 tcenv 顶层旧的 Metal 转发模块已全部移除，不能再被导入。

    @raises AssertionError 任一旧路径仍可导入时抛出。
    """
    import importlib

    for name in (
        "screen", "launch", "abi", "sanitize", "testing", "baselines",
        "profile", "pressure", "metal_runtime", "domain", "xctrace", "triton_baseline",
    ):
        try:
            importlib.import_module(f"tcenv.{name}")
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError(f"tcenv.{name} is still importable")


def test_contract_and_verify_have_one_canonical_class():
    """@brief 验证 metal 后端导出的 Contract 与 VerifyResult 就是全局唯一那一份类，且合约可 pickle 往返。
    """
    import pickle

    from tcenv.backends.metal.contract import Contract as MetalContract
    from tcenv.backends.metal.verify import VerifyResult as MetalVerifyResult
    from tcenv.contract import Contract
    from tcenv.verify import VerifyResult

    assert MetalContract is Contract
    assert MetalVerifyResult is VerifyResult
    contract = get_task(V1_TASK, "fp16", "paper").contract
    assert pickle.loads(pickle.dumps(contract)) == contract


def test_profile_command_uses_the_moved_xctrace_module(tmp_path, monkeypatch):
    """@brief 验证 profile CLI 走搬迁后的 xctrace 模块，且在评估被拒时以状态码 1 退出。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的 monkeypatch，用于把评估函数换成直接拒绝的假实现。
    """
    from types import SimpleNamespace

    from tcenv import cli, evaluate

    kernel = tmp_path / "kernel.metal"
    kernel.write_text(f"kernel void {V1_KERNEL}() {{}}\n")
    monkeypatch.setattr(evaluate, "evaluate", lambda *a, **k: SimpleNamespace(status="rejected", message="not profiled"))
    args = SimpleNamespace(kernel=str(kernel), task=V1_TASK, dtype="fp16", size="paper", target="apple-m4-max", json=False)
    assert cli.cmd_profile(args) == 1


def test_triton_baseline_command_uses_the_moved_module(monkeypatch):
    """@brief 验证 triton baseline CLI 走搬迁后的模块，在基线不可用时以状态码 1 退出。

    @param monkeypatch pytest 的 monkeypatch，用于把可用性探测固定为不可用。
    """
    from types import SimpleNamespace

    from tcenv import cli
    from tcenv.backends.metal import triton_baseline

    monkeypatch.setattr(triton_baseline, "available", lambda: False)
    args = SimpleNamespace(task="gemm", dtype="fp16", size="paper", reps=1, wait_quiet=1.0)
    assert cli.cmd_triton_baseline(args) == 1


def test_invalid_metal_limits_are_reported_as_device_check(monkeypatch, tmp_path):
    """Malformed Metal limits fail at device_check, never at candidate compile."""
    from dataclasses import replace
    from types import SimpleNamespace

    from tcenv import evaluate as ev
    from tcenv.target import load_target

    backend = load_backend("metal")
    profile = replace(
        load_target("apple-m4-max"),
        raw_limits={
            "max_buffer_index": "not-an-int",
            "max_threads_per_threadgroup": 1024,
            "max_threadgroup_memory_bytes": 32768,
            "counter_sample_us": 1000,
        },
    )
    monkeypatch.setattr("tcenv.target.require_active", lambda target_id: profile)
    monkeypatch.setattr("tcenv.target.load_target", lambda target_id: profile)
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: backend)
    monkeypatch.setattr(backend, "context", lambda: SimpleNamespace(info=lambda: {}))
    report = ev._evaluate_in_process(
        V1_TASK, "fp16", "small", f"kernel void {V1_KERNEL}() {{}}\n",
        mode="quick", seed=1, reps=1, target_id=profile.id,
    )
    assert report.status == "rejected" and report.stage == "device_check"
    assert "invalid Metal limits" in report.message


def test_heartbeat_callers_pass_a_backend_explicitly():
    """Metal workers must not silently lose launch/compile heartbeat hooks."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    for relative in ("tcenv/xcworker.py", "tcenv/tools/worker.py", "tcenv/worker.py"):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_install_heartbeat"
        ]
        assert calls, relative
        for call in calls:
            assert len(call.args) >= 2 or any(keyword.arg == "backend" for keyword in call.keywords), relative
            if len(call.args) >= 2:
                assert not isinstance(call.args[1], ast.Constant) or call.args[1].value is not None, relative
            for keyword in call.keywords:
                if keyword.arg == "backend":
                    assert not isinstance(keyword.value, ast.Constant) or keyword.value.value is not None, relative
