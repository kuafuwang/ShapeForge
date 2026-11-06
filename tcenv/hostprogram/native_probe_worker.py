"""Bounded native Host Program probe worker.

The CLI parent sends source and task parameters.  Only this process compiles/loads the dylib.
The result is the ordinary v2-worker result envelope containing a JSON HostRun document.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from tcenv.v2_worker import RESULT_SCHEMA

PROBE_SCHEMA = "tcenv.hostprogram.native-probe.1"


def _result(report: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": RESULT_SCHEMA, "report": dict(report), "execution": None, "build": None}


def _beat(path: Path, marker: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.pwrite(fd, f"{marker}{time.time():020.6f}\n".encode(), 0)
    finally:
        os.close(fd)


def evaluate_probe(payload: Mapping[str, Any], heartbeat: Path) -> dict[str, Any]:
    from taic.task_binding import bind_task
    from tcenv.hostprogram.contract import host_contract_spec
    from tcenv.hostprogram.errors import HostCapabilityUnavailable, HostProgramError
    from tcenv.hostprogram.native_runtime import NativeHostRuntime
    from tcenv.hostprogram.native_toolchain import compile_native_host_program
    from tcenv.taskcontract.runtime_shape import resolved_shape_for_case
    from tcenv.target import load_target
    from tcenv.tensors import round_to, to_raw
    from tcenv.track.abi import derive_tensor_memory_contract

    if payload.get("schema_id") != PROBE_SCHEMA:
        return _result({"status": "crash", "stage": "request", "message": "wrong native probe schema"})
    source = str(payload["source"])
    entry_path = str(payload.get("entry_path") or "host/tiling.c")
    try:
        binding = bind_task(str(payload["task"]), str(payload["dtype"]), str(payload["size"]))
        contract = binding.locator.load()
        target = load_target(str(payload["target_id"]))
        abi = derive_tensor_memory_contract(contract.instance, target=target)
        spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
        _beat(heartbeat, "C")
        build = compile_native_host_program(source, spec=spec, entry_path=entry_path)
        _beat(heartbeat, "E")
        runtime = NativeHostRuntime(build, spec)
        target_values = {
            "max_buffer_index": int(target.raw_limits.get("max_buffer_index", 0)),
            "max_threads_per_threadgroup": int(target.raw_limits.get("max_threads_per_threadgroup", 0)),
            "max_threadgroup_memory_bytes": int(target.raw_limits.get("max_threadgroup_memory_bytes", 0)),
        }
        runtime.set_target(target_values)
        logical = contract.reference.generate(contract.instance, str(payload["case"]), int(payload["seed"]))
        shapes = {tensor.name: tensor.shape for tensor in contract.instance.tensors}
        dtypes = {tensor.name: tensor.dtype for tensor in contract.instance.tensors}
        raw = {
            name: to_raw(round_to(value.reshape(tuple(int(v) for v in shapes[name])), dtypes[name]), dtypes[name])
            for name, value in logical.items()
            if name in shapes
        }
        _beat(heartbeat, "B")
        # An L2 (capacity) instance hands the candidate this call's resolved total/count.  The probe
        # reconstructs it from the declared workload profile, exactly like the evaluation path, so a
        # probe run and a full evaluation cannot disagree about which batch was planned.
        resolved = resolved_shape_for_case(contract.instance, str(payload["case"]))
        # Only an L2 (capacity) instance has runtime resolved scalars.  An L1 instance's resolved shape *is*
        # its declared total/count, so it stays in ``SfHostShapes`` and the contract declares no
        # ``SfHostResolved`` field for it.
        scalars = resolved.resolved_scalars() if resolved is not None and resolved.kind == "l2" else None
        run = runtime.execute(
            lambda name, off, nbytes: raw[name][off : off + nbytes],
            values=None,
            resolved=scalars,
            audit=True,
        )
        _beat(heartbeat, "E")
        document = run.document()
        document.update({
            "status": "ok" if run.ok else "host_program_failed",
            "assembled_host_sha256": build.assembled_sha256,
            "native_sha256": build.native_sha256,
            "native_wall_ns": int(run.host_wall_ns),
            "used_workspaces": list(build.used_workspaces),
            "memory_budget_bytes": build.memory_budget_bytes,
            "execution_backend": build.toolchain.execution_backend,
            "platform": build.toolchain.platform,
            "compiler_version": build.toolchain.clang_version,
        })
        return _result(document)
    except HostCapabilityUnavailable as exc:
        return _result({"status": "backend_capability_unavailable", "stage": "host_toolchain", "message": str(exc)})
    except HostProgramError as exc:
        return _result({"status": "host_program_failed", "stage": "host_program", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - worker returns a typed report, parent still owns native crashes
        return _result({"status": "crash", "stage": "host_program", "message": f"{type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    out = Path(request["out"])
    heartbeat = Path(request["heartbeat"])
    result = evaluate_probe(request["payload"], heartbeat)
    out.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    report = result.get("report", {})
    return 0 if report.get("status") == "ok" else 1


def run_bounded_probe(
    payload: Mapping[str, Any], *, timeout: float = 30.0, launch_deadline_s: float = 10.0
) -> dict[str, Any]:
    """@brief 控制机入口：只返回 JSON，不 import/加载候选 native library。"""
    from tcenv.v2_worker import run_bounded_v2

    return run_bounded_v2(
        dict(payload),
        timeout=float(timeout),
        launch_deadline_s=float(launch_deadline_s),
        compile_deadline_s=float(timeout),
        worker_module="tcenv.hostprogram.native_probe_worker",
    )


__all__ = ["PROBE_SCHEMA", "evaluate_probe", "main", "run_bounded_probe"]


if __name__ == "__main__":
    raise SystemExit(main())
