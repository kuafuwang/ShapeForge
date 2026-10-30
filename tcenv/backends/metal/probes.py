"""Approved probe handlers. Requests name a kernel, never an import or a Python file."""

from __future__ import annotations

import statistics

from tcenv.gpulock import gpu_lock
from tcenv.schema import PROTOCOL_VERSION


def _build_probe_source(source: str, kernel_name: str):
    """Build one typed probe through the backend factory and its registered diagnostic profile.

    The probe has no task contract, so it uses ``metal.raw-source-probe``: the caller's bytes
    are the compilation unit (no contract header), the math mode is the profile's, and the
    result is a ``BuildOutcome`` whose live handle the probe runs.  There is no convenience
    compile entry point any more (R3 §5.2).

    @param source 已通过静态筛查的 MSL 源文本。
    @param kernel_name request 里的核名。
    @return ``BuildOutcome``。
    """
    from tcenv.backends.registry import load_backend
    from tcenv.remote.artifact_builder import SourceBundle

    from .source_builder import DiagnosticContract

    contract = DiagnosticContract(kernel_name=str(kernel_name))
    builder = load_backend("metal").artifact_builder(
        kernel_contract=contract, diagnostic_profile="metal.raw-source-probe"
    )
    return builder.build(
        SourceBundle(source=str(source).encode("utf-8")),
        task_contract=contract,
        deadline_monotonic=None,
    )


def time_msl(request: dict, *, manifest) -> dict:
    """Time a single-buffer MSL probe; correctness is outside this tool's claim."""
    from tcenv.target import device_matches, require_active

    target = require_active(request["target_id"])
    if target.backend != "metal":
        raise ValueError("metal.time_msl requires a Metal target")
    from .runtime import BufferArg, get_context
    from .screen import screen_source

    problems = screen_source(request["source"])
    base = {
        "schema_version": 1, "tool_id": manifest.tool_id, "version": manifest.version,
        "request_digest": "", "tool_manifest_sha256": manifest.digest,
        "handler_source_sha256": manifest.handler_source_sha256,
        "metrics": {}, "units": {}, "target_id": request["target_id"],
        "protocol_version": PROTOCOL_VERSION, "stdout_excerpt": "", "error": "",
    }
    from tcenv.tools.manifest import canonical_digest
    base["request_digest"] = canonical_digest(request)
    if problems:
        return {**base, "status": "rejected", "error": "; ".join(problems)}
    try:
        with gpu_lock("typed Metal probe compile", exclusive=False):
            ctx = get_context()
            info = ctx.info()
            if not device_matches(target, info):
                raise ValueError(f"device {info.get('name')} does not match target {target.id}")
            outcome = _build_probe_source(request["source"], request["kernel_name"])
            handles = outcome.executable_handles()
            if len(handles) != 1:
                raise ValueError("metal.time_msl builds exactly one pipeline")
            kernel = handles[0]
            if len(kernel.bindings) != 1 or kernel.bindings[0].kind != "buffer" or kernel.bindings[0].index != 0:
                raise ValueError("metal.time_msl requires exactly one [[buffer(0)]] parameter and no other bindings")
            if kernel.max_total_threads < request["threads"][0] * request["threads"][1] * request["threads"][2]:
                raise ValueError("threads exceed this pipeline's maxTotalThreadsPerThreadgroup")
            buf = ctx.new_buffer(request["buffer_bytes"])
        with gpu_lock("typed Metal probe timing", exclusive=True):
            samples = [ctx.run(kernel, [[(0, BufferArg(buf))]], tuple(request["grid"]), tuple(request["threads"]), iters=request["iters"]).wall_s for _ in range(3)]
        per_iter = [sample / request["iters"] for sample in samples]
        return {**base, "status": "ok", "metrics": {"wall_per_iter_s": statistics.median(per_iter), "samples_s": samples, "iters": request["iters"]}, "units": {"wall_per_iter_s": "s", "samples_s": "s", "iters": "count"}, "stdout_excerpt": "Timing only. This probe does not establish numerical correctness."}
    except Exception as exc:  # noqa: BLE001 - errors become bounded structured results
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
