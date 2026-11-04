"""Fresh-interpreter B0.5 import-boundary driver; never imports Metal.

The blocker is installed before any project module is imported.  This is a
support helper, not a second backend or a production composition root.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from tests.support.v1_tasks import V1_TASK


#: The registered LoweringProfile/lowerer registry is *pure data* (no Metal API, no device) and
#: R1/R2 place it under the backend package; a task's track names a registered profile, so task
#: assembly consults it.  Everything else under ``tcenv.backends.metal`` is runtime/device code
#: and stays out of the generic path.
ALLOWED = frozenset({"tcenv.backends.metal", "tcenv.backends.metal.lowering"})


class _BlockMetalImports:
    blocked: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname in ALLOWED:
            return None
        if fullname == "tcenv.backends.metal" or fullname.startswith("tcenv.backends.metal."):
            self.blocked.append(fullname)
            raise AssertionError(f"generic path imported {fullname}")
        return None


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    root = Path(sys.argv[1])
    sys.meta_path.insert(0, _BlockMetalImports())

    from dataclasses import replace

    from tcenv import evaluate as ev
    from tcenv.target import TargetProfile
    from tests.support.npu_sim.fake_backend import FakeNpuBackend
    import tcenv.backends.registry as registry
    import tcenv.target as target_module
    import tasks as task_module

    backend = FakeNpuBackend()
    profile = TargetProfile(
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
    from tests.support.v1_tasks import v1_task

    original = v1_task(V1_TASK, "fp16", "small")
    task = replace(
        original,
        reference_sources={
            "metal": original.reference_for("metal"),
            "sim-npu": "npu-kernel relu_kernel\n",
        },
    )
    target_module.require_active = lambda target_id: profile
    target_module.load_target = lambda target_id: profile
    registry.load_backend = lambda name: backend
    task_module.get_task = lambda name, dtype, size: task
    os.environ["TCENV_GPU_LOCK_FILE"] = str(root / "device.lock")

    source = "npu-kernel npu_main\nnpu-launch threads=4096 threadgroup=1073741824\n"
    report = ev._evaluate_in_process(
        V1_TASK, "fp16", "small", source,
        mode="full", seed=7, reps=2, measure_serial=False, target_id=profile.id,
    )
    source_path = root / "candidate.npu"
    source_path.write_text(source, encoding="utf-8")
    worker_report = ev._run_worker(
        V1_TASK, "fp16", "small", source_path, "validate", 7, 2, 10.0,
        validation=True, cases=None, worker_module="tests.support.fake_worker_success",
        target_id=profile.id, backend=backend,
    )
    original_run_worker = ev._run_worker
    try:
        ev._run_worker = lambda *args, **kwargs: ev.EvalReport(
            task=V1_TASK, dtype="fp16", size="small", mode="full", seed=7, status="ok"
        )
        parent_report = ev.evaluate(
            V1_TASK, "fp16", "small", source, mode="full", seed=7, reps=2,
            eval_cache=False, target_id=profile.id,
        )
    finally:
        ev._run_worker = original_run_worker
    metal_modules = sorted(
        name for name in sys.modules
        if (name == "tcenv.backends.metal" or name.startswith("tcenv.backends.metal."))
        and name not in ALLOWED
    )
    result = {
        "status": report.status,
        "parent_status": parent_report.status,
        "worker_status": worker_report.status,
        "backend": report.backend,
        "target_id": report.target_id,
        "device_warnings": list(report.device_warnings or []),
        "speedup": report.speedup,
        "metal_modules": metal_modules,
        "metal_lowering_loaded": "tcenv.backends.metal.lowering" in sys.modules,
        "blocked": list(_BlockMetalImports.blocked),
        "calls": list(backend.calls),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if (
        report.status == "ok" and parent_report.status == "ok" and worker_report.status == "ok"
        and not metal_modules and not _BlockMetalImports.blocked
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
