"""Phase B0.5 backend-neutral evaluation, target and fingerprint contracts. CPU only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

import pytest

from taic.kernel_agent import kernel_text
from tasks import get_task
from tcenv import evaluate as ev
from tcenv.evaluate import EvalReport
from tcenv.target import TargetProfile, load_target
from tcenv.task import task_fingerprint

from ..support.npu_sim.fake_backend import FakeNpuBackend


SOURCE = "npu-kernel npu_main\nnpu-launch threads=4096 threadgroup=1073741824\n"


class _BlockMetalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tcenv.backends.metal" or fullname.startswith("tcenv.backends.metal."):
            raise AssertionError(f"generic fake-backend path imported {fullname}")
        return None


@pytest.fixture
def no_metal_imports(monkeypatch):
    before = set(sys.modules)
    monkeypatch.setattr(sys, "meta_path", [_BlockMetalImports(), *sys.meta_path])
    yield
    imported = {
        name for name in set(sys.modules) - before
        if name == "tcenv.backends.metal" or name.startswith("tcenv.backends.metal.")
    }
    assert imported == set()


def _fake_profile(root: Path) -> TargetProfile:
    return TargetProfile(
        id="sim-npu-test",
        backend="sim-npu",
        status="active",
        display_name="Fake NPU test target",
        match_name_contains="",
        match_family="",
        calibration_device_name="Fake NPU",
        raw_limits={
            "max_buffer_index": 0,
            "max_threads_per_threadgroup": 1,
            "max_threadgroup_memory_bytes": 1,
            "counter_sample_us": 0,
        },
        root=root,
        language="fake-npu",
        file_ext=".npu",
        backend_version="1",
    )


def _install_fake(monkeypatch, tmp_path: Path):
    backend = FakeNpuBackend()
    profile = _fake_profile(tmp_path)
    original = get_task("relu", "fp16", "small")
    task = replace(
        original,
        reference_sources={
            "metal": original.reference_for("metal"),
            "sim-npu": "npu-kernel relu_kernel\n",
        },
    )
    monkeypatch.setattr("tcenv.target.require_active", lambda target_id: profile)
    monkeypatch.setattr("tcenv.target.load_target", lambda target_id: profile)
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: backend)
    monkeypatch.setattr("tasks.get_task", lambda name, dtype, size: task)
    monkeypatch.setenv("TCENV_GPU_LOCK_FILE", str(tmp_path / "device.lock"))
    return backend, profile, task


def test_generic_fake_evaluation_never_imports_metal_or_applies_metal_limits(no_metal_imports, monkeypatch, tmp_path):
    """Execute screen/launch/compile/ABI/correctness/bench/baseline through the one fake backend."""
    backend, profile, _task = _install_fake(monkeypatch, tmp_path)
    report = ev._evaluate_in_process(
        "relu", "fp16", "small", SOURCE,
        mode="full", seed=7, reps=2, measure_serial=False, target_id=profile.id,
    )
    assert report.status == "ok", report.message
    assert "kernel void" not in SOURCE
    assert report.backend == "sim-npu" and report.backend_version == "1"
    assert report.target_id == "sim-npu-test" and report.device_name == "Fake NPU"
    assert report.device_warnings == ["simulated device; target limits are not hardware measurements"]
    assert ("validate_launch", SOURCE) in backend.calls
    assert ("check_abi", profile.id) in backend.calls
    assert ("compiler_baseline", "relu") in backend.calls
    assert report.compiler_baseline["simulated"] is True
    assert report.speedup == pytest.approx(2.0)
    assert report.benchmark["rotation_sets"] == 1


def test_parent_evaluation_and_worker_sanitization_use_backend_hooks(no_metal_imports, monkeypatch, tmp_path):
    """The parent path also avoids Metal screen/deadline/validation/sanitize imports."""
    backend, profile, _task = _install_fake(monkeypatch, tmp_path)
    worker_calls = []
    real_run_worker = ev._run_worker

    def fake_run_worker(task, dtype, size, kernel_path, mode, seed, reps, timeout, *, validation, cases, backend, **kwargs):
        worker_calls.append((mode, validation, kernel_path.suffix, backend))
        return EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=seed, status="ok", stage="done")

    monkeypatch.setattr(ev, "_run_worker", fake_run_worker)
    report = ev.evaluate("relu", "fp16", "small", SOURCE, mode="full", seed=11, reps=2, target_id=profile.id)
    assert report.status == "ok"
    assert report.safety["shader_validation"] == "unsupported"
    assert [(mode, validation, suffix) for mode, validation, suffix, _backend in worker_calls] == [("full", False, ".npu")]
    assert worker_calls[0][3] is backend

    # Use the real subprocess driver with an immediate worker stand-in.  This
    # exercises worker_env and validation-stderr parsing without a device.
    monkeypatch.setattr(ev, "_run_worker", real_run_worker)
    source_path = tmp_path / "candidate.npu"
    source_path.write_text(SOURCE, encoding="utf-8")
    driven = ev._run_worker(
        "relu", "fp16", "small", source_path, "validate", 3, 2, 10.0,
        validation=True, cases=None, worker_module="tests.support.fake_worker_success",
        target_id=profile.id, backend=backend,
    )
    assert driven.status == "ok"
    assert ("worker_env", "True") in backend.calls
    assert ("parse_validation_stderr", "") in backend.calls


def test_backend_declaration_pattern_drives_source_extraction(no_metal_imports):
    backend = FakeNpuBackend()
    pattern = backend.source_name_pattern("npu_main")
    fenced = f"```npu\n{SOURCE}```\n"
    assert kernel_text(fenced, "npu_main", source_fence="npu", source_name_pattern=pattern) == SOURCE
    assert kernel_text(SOURCE, "npu_main", source_fence="npu", source_name_pattern=pattern) == SOURCE
    assert kernel_text("kernel void npu_main() {}", "npu_main", source_fence="npu", source_name_pattern=pattern) is None


def _fingerprint_task():
    references = {"metal": "metal-reference-v1", "sim-npu": "npu-reference-v1"}
    return SimpleNamespace(
        contract=SimpleNamespace(describe=lambda: "contract-v1"),
        triton_source="triton-v1",
        cases=("case-v1",),
        rtol=0.125,
        atol_rel=0.25,
        exact=False,
        flops=3.0,
        min_bytes=5.0,
        dtype="fp16",
        size="small",
        meta={"source": "unit-golden"},
        reference_for=lambda backend_id: references.get(backend_id, ""),
    )


def test_task_fingerprint_golden_is_backend_aware_and_metal_compatible():
    task = _fingerprint_task()
    metal = task_fingerprint(task, "metal")
    simulated = task_fingerprint(task, "sim-npu")
    assert metal == task_fingerprint(task) == "c645aa7b004a2dbcb98b67984ccd6fc8d5520483d0fc5d7d3b48ee3b46558a5c"
    assert simulated == "bdf31ed40ace2e3f516c69c49938fa1e16b42558d5f3acc9e9ae8b64efb2ed02"
    assert simulated != metal
    assert task_fingerprint(task, "future-npu") != simulated

    # Independently recompute the pre-B0.5 Metal input tuple as a compatibility golden.
    recorded = {
        "ieee_base": None,
        "ieee_inputs": None,
        "ieee_max": None,
        "domain_case": False,
        "source": "unit-golden",
    }
    import json

    legacy_parts = (
        "contract-v1", "triton-v1", "metal-reference-v1", "('case-v1',)",
        repr((0.125, 0.25, False)), repr((3.0, 5.0)), "fp16", "small",
        json.dumps(recorded, sort_keys=True, ensure_ascii=False),
    )
    digest = hashlib.sha256()
    for part in legacy_parts:
        digest.update(part.encode())
        digest.update(b"\0")
    assert metal == digest.hexdigest()


def test_target_loader_does_not_require_metal_limit_keys(monkeypatch, tmp_path):
    backend = FakeNpuBackend()
    root = tmp_path / "targets" / "fake-target"
    root.mkdir(parents=True)
    (root / "target.toml").write_text(
        """schema_version = 1
id = "fake-target"
backend = "sim-npu"
status = "reserved"
display_name = "Fake target"

[device]
calibration_device_name = "Fake NPU"

[limits]
queue_depth = 8
""",
        encoding="utf-8",
    )
    (root / "device.md").write_text("fake\n", encoding="utf-8")
    (root / "experience.md").write_text("fake\n", encoding="utf-8")
    monkeypatch.setattr("tcenv.target.TARGETS", tmp_path / "targets")
    monkeypatch.setattr("tcenv.backends.registry.load_backend", lambda name: backend)

    profile = load_target("fake-target")
    assert profile.raw_limits == {"queue_depth": 8}
    assert profile.language == "fake-npu" and profile.file_ext == ".npu"
    assert profile.backend_version == "1"


def test_fresh_interpreter_never_imports_metal_on_the_generic_path(tmp_path):
    """The blocker is installed before any project import, not after Metal is preloaded."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "tests.support.backend_neutral_driver", str(tmp_path)],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=90, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "ok"
    assert result["parent_status"] == "ok"
    assert result["worker_status"] == "ok"
    assert result["backend"] == "sim-npu"
    assert result["metal_modules"] == []
    assert result["blocked"] == []
    assert ("validate_launch", SOURCE) in {tuple(call) for call in result["calls"]}
    assert ("worker_env", "True") in {tuple(call) for call in result["calls"]}


def _worker_request(tmp_path: Path, target_id: str) -> tuple[Path, Path, dict]:
    source = tmp_path / "candidate.npu"
    source.write_text(SOURCE, encoding="utf-8")
    out = tmp_path / "worker-result.json"
    raw = {
        "schema_version": 4,
        "task": "relu", "dtype": "fp16", "size": "small",
        "kernel_path": str(source), "mode": "quick", "seed": 3, "reps": 1,
        "cases": [], "out": str(out), "heartbeat": str(tmp_path / "heartbeat"),
        "fast_bench": False, "require_contract_launch": False,
        "measure_serial": False, "target_id": target_id,
    }
    path = tmp_path / "worker-request.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, out, raw


def test_worker_main_resolves_the_backend_and_preserves_invalid_target_behavior(monkeypatch, tmp_path):
    """Exercise the actual tcenv.worker.main, not only the parent _run_worker wrapper."""
    from tcenv import worker as worker_mod

    backend, profile, _task = _install_fake(monkeypatch, tmp_path)
    request_path, out, raw = _worker_request(tmp_path, profile.id)
    seen = {}

    def fake_heartbeat(path, selected):
        seen["path"] = path
        seen["backend"] = selected

    def fake_evaluate(task, dtype, size, source, **kwargs):
        seen["evaluated"] = (task, dtype, size, kwargs["target_id"])
        return EvalReport(task=task, dtype=dtype, size=size, mode=kwargs["mode"], status="ok")

    monkeypatch.setattr("tcenv.worker._install_heartbeat", fake_heartbeat)
    monkeypatch.setattr("tcenv.evaluate.evaluate_in_process", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["tcenv.worker", str(request_path)])
    assert worker_mod.main() == 0
    assert seen["backend"] is backend
    assert seen["path"] == raw["heartbeat"]
    assert seen["evaluated"] == ("relu", "fp16", "small", profile.id)
    assert out.is_file()

    from tcenv.target import TargetError

    seen.clear()
    monkeypatch.setattr("tcenv.target.load_target", lambda target_id: (_ for _ in ()).throw(TargetError("bad target")))
    request_path, out, _raw = _worker_request(tmp_path, "invalid-target")
    monkeypatch.setattr(sys, "argv", ["tcenv.worker", str(request_path)])
    assert worker_mod.main() == 0
    assert "backend" not in seen
    assert seen["evaluated"][-1] == "invalid-target"
    assert out.is_file()


def test_fake_eval_cache_miss_then_hit_does_not_execute_twice(monkeypatch, tmp_path):
    backend, profile, _task = _install_fake(monkeypatch, tmp_path)
    monkeypatch.setenv("TCENV_EVAL_CACHE_DIR", str(tmp_path / "cache")
    )
    worker_calls = []

    def fake_run_worker(*args, **kwargs):
        worker_calls.append((args, kwargs))
        return EvalReport(
            task="relu", dtype="fp16", size="small", mode="quick",
            seed=23, status="ok", speedup=2.0, candidate_us=1.0,
        )

    monkeypatch.setattr(ev, "_run_worker", fake_run_worker)
    first = ev.evaluate(
        "relu", "fp16", "small", SOURCE, mode="quick", seed=23, reps=1,
        eval_cache=True, target_id=profile.id,
    )
    assert first.status == "ok"
    assert len(worker_calls) == 1
    calls_after_miss = len(backend.calls)
    second = ev.evaluate(
        "relu", "fp16", "small", SOURCE, mode="quick", seed=23, reps=1,
        eval_cache=True, target_id=profile.id,
    )
    assert second.status == "ok"
    assert second.speedup == first.speedup
    assert len(worker_calls) == 1
    assert len(backend.calls) == calls_after_miss
