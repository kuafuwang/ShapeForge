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
