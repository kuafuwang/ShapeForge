"""A6: after the v1 retirement, both live paths still run on the real Mac.

The retirement checks are CPU-side (``tests/shared/test_a6_retirement.py`` and
``tests/backends/metal/test_a6_metal_config.py``).  Two things still need a real device:

* the **v2** ``evaluate_task`` path, with a candidate written here (candidate-owned constant and
  ``taic-launch``), must build, dispatch and verify *without* reading any v1
  ``LoweringProfile`` — the profiles A6 deleted must not be on the v2 path at all.

The second historical case (a "preserved v1 task still evaluates with real timing") was retired
with the v1 evaluation entry point; ``tcenv.evaluate.evaluate`` now refuses every call, so that
record can no longer be produced and is not claimed as current coverage.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
gpu = pytest.mark.gpu


def _record(name: str, payload: dict) -> None:
    """@brief A6 verify sets ``A6_EVIDENCE_DIR``; a live slice archives one record."""
    directory = os.environ.get("A6_EVIDENCE_DIR")
    if not directory:
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    identity = {key: body[key] for key in sorted(body) if key not in ("identity_sha256", "median_us", "elapsed_s")}
    body["identity_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (Path(directory) / name).write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")


def _locator(task: str, *, dtype: str = "fp16", size: str = "small"):
    """@brief 为指定实例构造 v2 locator（pack 摘要与部署闭包都来自该实例）。"""
    from tcenv.taskcontract import load_task_contract
    from tcenv.taskcontract.locator import TaskLocator
    from tcenv.taskcontract.pack import task_pack_closure
    from tcenv.track import REPO_ROOT

    directory = ROOT / "tasks" / task
    contract = load_task_contract(directory, dtype=dtype, size=size)
    return (
        TaskLocator(
            task_id=task,
            op_name=contract.op_name,
            dtype=contract.dtype,
            size=contract.size,
            pack_digest=contract.pack_digest,
            deployment_pack_digest=task_pack_closure(directory, root=REPO_ROOT).digest,
            task_dir=directory,
            code_root=REPO_ROOT,
        ),
        contract,
    )


def _sigmoid_candidate(contract) -> str:
    """@brief 候选自有几何的 sigmoid MSL：网格由候选的 ``taic-launch`` 声明，题面不给 grid。

    The runtime prepends the contract header (symbol constants such as ``N``), so the candidate
    must not redeclare them; what it *does* own is the launch block and its own ``BLOCK``.
    """
    n = int(contract.instance.symbols["N"])
    block = 256
    assert n % block == 0
    return (
        f"constant constexpr uint BLOCK = {block};\n"
        "/* taic-launch\n"
        "math safe\n"
        f"launch sigmoid_kernel grid={n // block},1,1 threads={block},1,1\n"
        "*/\n"
        "kernel void sigmoid_kernel(device const half* x [[buffer(0)]],\n"
        "                           device half* y [[buffer(1)]],\n"
        "                           uint i [[thread_position_in_grid]]) {\n"
        "    if (i < N) {  // N comes from the contract header\n"
        "        float v = float(x[i]);\n"
        "        y[i] = half(1.0f / (1.0f + exp(-v)));\n"
        "    }\n"
        "}\n"
    )


@gpu
def test_a6_gpu_the_v2_path_runs_without_any_v1_lowering_profile(monkeypatch):
    """@brief 真实 Mac：v2 候选构建/派发/正确性不调用 v1 profile 读取器（它们已退役）。"""
    from tcenv import lowering
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract = _locator("sigmoid")
    # The retired v1 profile is absent before the patch (so it cannot be read afterwards).
    with pytest.raises(lowering.LoweringError):
        lowering.profile_for("metal", "metal-sigmoid")

    def refuse(*_args, **_kwargs):
        raise AssertionError("the v2 path called a v1 lowering reader")

    for name in ("profile_for", "registered_profiles", "require_target_profile", "ensure_lowerings", "lower"):
        monkeypatch.setattr(lowering, name, refuse)

    source = _sigmoid_candidate(contract)
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="full", seed=3, reps=3, cases=("normal", "wide_range", "outlier")),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=ROOT / "backends" / "metal" / "tracks.json",
    )
    assert result.report["status"] == "ok", result.report
    assert result.report["eligible"] is True and result.report["correctness"]["ok"] is True
    assert result.execution is not None and result.execution["dispatch"], result.execution
    assert result.execution["provider"] == "candidate"
    n = int(contract.instance.symbols["N"])
    assert result.execution["dispatch"][0]["kernel"] == "sigmoid_kernel"
    assert tuple(result.execution["dispatch"][0]["grid"]) == (n // 256, 1, 1)
    assert tuple(result.execution["dispatch"][0]["threads"]) == (256, 1, 1)
    benchmark = result.report["benchmark"]
    assert benchmark["candidate"]["median_us"] > 0
    _record(
        "gpu_v2_sigmoid.json",
        {
            "schema_version": 1,
            "task": "sigmoid",
            "track": "msl-single",
            "dtype": locator.dtype,
            "size": locator.size,
            "status": result.report["status"],
            "provider": result.execution["provider"],
            "correctness_ok": result.report["correctness"]["ok"],
            "effective_correctness_only": result.effective_correctness_only,
            "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "bundle_digest": result.bundle_digest,
            "geometry_identity": result.geometry_identity,
            "native_digest": result.native_digest,
            "artifact_set_digest": result.artifact_set_digest,
            "presentation_sha256": result.presentation_sha256,
            "dispatch": [dict(entry) for entry in result.execution["dispatch"]],
            "median_us": benchmark["candidate"]["median_us"],
            "device_name": result.report["device_name"],
            "gpu_family": result.report["gpu_family"],
        },
    )
