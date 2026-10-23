"""Phase 1 adapter delegates to the existing Metal implementation."""

from __future__ import annotations

from tasks import get_task

from tcenv.backends.registry import load_backend


def test_parent_side_screen_and_launch_parse_delegate():
    backend = load_backend("metal")
    task = get_task("relu", "fp16", "paper")
    assert backend.screen("kernel void k() {}", task) == []
    assert backend.screen("kernel void k() { for (;;) {} }", task)
    assert backend.parse_launch("kernel void k() {}", task.contract) is None


def test_check_abi_pipeline_receives_every_kernel_and_buffer_limit(monkeypatch):
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


def test_contract_rendering_is_delegated_without_changing_text():
    backend = load_backend("metal")
    contract = get_task("relu", "fp16", "paper").contract
    source = "kernel void relu_kernel() {}\n"
    full, header = backend.assemble(contract, source)
    assert full == contract.assemble(source)[0]
    assert header == contract.assemble(source)[1]
    assert backend.describe(contract) == contract.describe()
    assert backend.abi_stub(contract) == contract.abi_stub()


def test_moved_tooling_keeps_the_repository_root():
    from pathlib import Path

    from tcenv.backends.metal import triton_baseline, xctrace

    repo = Path(__file__).resolve().parents[1]
    assert triton_baseline.REPO_ROOT == repo
    assert xctrace.REPO_ROOT == repo
    assert triton_baseline.CACHE_DIR == repo / "runs" / "triton_baselines"


def test_rotation_sets_are_delegated():
    backend = load_backend("metal")
    from tcenv.backends.metal.baselines import rotation_count

    task = get_task("flash_attention", "fp16", "paper")
    assert backend.rotation_sets(task) == rotation_count(task)


def test_metal_metadata_drives_task_rendering_names():
    backend = load_backend("metal")
    assert backend.display_name == "Metal"
    assert backend.source_fence == "metal"
    assert backend.reference_skill == "msl_reference"
    assert backend.skill_names[2] == "msl_reference"


def test_submission_name_accepts_a_backend_extension():
    from taic.kernel_agent import submission_name

    assert submission_name("relu", "fp16", "small", "init", ".metal") == "relu-fp16-small-init.metal"
    assert submission_name("relu", "fp16", "small", "init", ".x") == "relu-fp16-small-init.x"


def test_phase4_removed_the_top_level_metal_shims():
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
    import pickle

    from tcenv.backends.metal.contract import Contract as MetalContract
    from tcenv.backends.metal.verify import VerifyResult as MetalVerifyResult
    from tcenv.contract import Contract
    from tcenv.verify import VerifyResult

    assert MetalContract is Contract
    assert MetalVerifyResult is VerifyResult
    contract = get_task("relu", "fp16", "paper").contract
    assert pickle.loads(pickle.dumps(contract)) == contract


def test_profile_command_uses_the_moved_xctrace_module(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tcenv import cli, evaluate

    kernel = tmp_path / "kernel.metal"
    kernel.write_text("kernel void relu_kernel() {}\n")
    monkeypatch.setattr(evaluate, "evaluate", lambda *a, **k: SimpleNamespace(status="rejected", message="not profiled"))
    args = SimpleNamespace(kernel=str(kernel), task="relu", dtype="fp16", size="paper", target="apple-m4-max", json=False)
    assert cli.cmd_profile(args) == 1


def test_triton_baseline_command_uses_the_moved_module(monkeypatch):
    from types import SimpleNamespace

    from tcenv import cli
    from tcenv.backends.metal import triton_baseline

    monkeypatch.setattr(triton_baseline, "available", lambda: False)
    args = SimpleNamespace(task="gemm", dtype="fp16", size="paper", reps=1, wait_quiet=1.0)
    assert cli.cmd_triton_baseline(args) == 1
