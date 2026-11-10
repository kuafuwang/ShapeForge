"""Fixed MT2 task/candidate fixtures (manual candidates only; no model or production task)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from tcenv.track import CandidateArtifact, CandidateEntry, REPO_ROOT
from .v2_tasks import sem_opspec, write_task

TRACK_ID = "tilelang-pipeline"
TRACK_VERSION = 2
N = 1024

REFERENCE = '''"""MT2 fixed elementwise oracle: y=2*x+1 and read_write acc += y."""

import torch


def gen_inputs(case, seed, instance):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shape = tuple(instance.tensor("x").shape)
    x = torch.randn(shape, generator=generator, dtype=torch.float32)
    acc = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.25
    if case == "wide_range":
        x = x * 100.0
    elif case == "outlier":
        x = torch.where(torch.rand(shape, generator=generator) < 0.01, x * 1000.0, x)
    return {"x": x, "acc": acc}


def reference(inputs, attrs):
    x = inputs["x"].to(torch.float64)
    y = x * 2.0 + 1.0
    return {"y": y, "acc": inputs["acc"].to(torch.float64) + y}
'''


def mt2_opspec(op_name: str = "mt2_scale") -> dict[str, Any]:
    document = sem_opspec(op_name)
    document["formula"] = "y = 2*x + 1; acc = acc + y; tmp is candidate-owned workspace"
    document["type_vars"] = {"T": ["fp32"]}
    document["tensors"]["acc"] = {
        "role": "input", "dims": ["N"], "dtype": "T", "layout": "contiguous",
        "strides": ["1"], "space": "global", "access": "read_write",
    }
    document["tensors"]["tmp"] = {
        "role": "output", "dims": ["N"], "dtype": "T", "layout": "contiguous",
        "strides": ["1"], "space": "workspace", "access": "read_write",
    }
    document["numerics"] = {
        "acc_dtype": "fp32", "rounding": "single_round_at_output",
        "tolerances": {"fp32": {"rtol": 0.00001, "atol_rel": 0.000001}}, "exact": False,
    }
    document["cases"] = {"names": ["normal", "wide_range", "outlier"]}
    document["sizes"] = {"small": {"N": N, "usage": ["golden", "benchmark"]}}
    document["discovery"] = {"phase": 1, "default_instance": {"dtype": "fp32", "size": "small"}}
    document["work_accounting"]["flops"] = {"terms": [{"coefficient": 3, "symbols": ["N"]}]}
    document["work_accounting"]["min_bytes"]["terms"].append({"tensor": "acc"})
    return copy.deepcopy(document)


def write_mt2_task(tmp_path: Path):
    root = tmp_path / "tasks"
    write_task(root, op_name="mt2_scale", opspec=mt2_opspec(), reference=REFERENCE)
    from tcenv.taskcontract import discover_standard_tasks

    return discover_standard_tasks(root, code_root=REPO_ROOT)[0]


def write_mt2_registry(path: Path) -> Path:
    """Write a test-only /2 available registry; shipped /1 stays interface_only."""
    document = json.loads((Path(REPO_ROOT) / "backends/metal/tracks.json").read_text(encoding="utf-8"))
    # MT2 fixtures exercise /1 and /2 only.  The shipped /3 registration must not change their
    # default selection or the frozen /2 audit path.
    document["tracks"] = [
        track for track in document["tracks"]
        if not (track["track_id"] == TRACK_ID and track["track_version"] == 3)
    ]
    v1 = next(track for track in document["tracks"] if track["track_id"] == TRACK_ID and track["track_version"] == 1)
    existing = [track for track in document["tracks"] if track["track_id"] == TRACK_ID and track["track_version"] == TRACK_VERSION]
    if existing:
        v2 = existing[0]
    else:
        v2 = copy.deepcopy(v1)
        document["tracks"].append(v2)
    v2.update(
        track_version=TRACK_VERSION,
        status="available",
        capabilities=list(dict.fromkeys([*v1["capabilities"], "metal_dispatch"])),
        build_adapter={"implementation_id": "metal.track.tilelang-pipeline.build", "implementation_version": 2},
        evidence=[
            "tests/backends/metal/test_tilelang_mt2_gpu.py",
            "tests/support/golden/shared/多DSL-Track-阶段验收/MT2/acceptance.md",
        ],
    )
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


SINGLE_TILELANG = '''from __future__ import annotations
import tilelang
import tilelang.metal.language as T

N = 1024
THREADS = 256

@T.prim_func
def scale_kernel(
    x: T.Tensor((N,), "float32"),
    acc: T.Tensor((N,), "float32"),
    y: T.Tensor((N,), "float32"),
):
    with T.Kernel(N // THREADS, threads=THREADS) as bx:
        for i in T.Parallel(THREADS):
            index = bx * THREADS + i
            value = x[index] * 2.0 + 1.0
            y[index] = value
            acc[index] = acc[index] + value

KERNEL = scale_kernel
'''

WORKLIST_TILELANG = '''from __future__ import annotations
import tilelang
import tilelang.metal.language as T

N = 1024
THREADS = 128

@T.prim_func
def scale_kernel(x: T.Tensor((N,), "float32"), tmp: T.Tensor((N,), "float32")):
    with T.Kernel(N // THREADS, threads=THREADS) as bx:
        for i in T.Parallel(THREADS):
            index = bx * THREADS + i
            tmp[index] = x[index] * 2.0
            tmp[index] = tmp[index] + 0.0

@T.prim_func
def bias_kernel(
    tmp: T.Tensor((N,), "float32"),
    acc: T.Tensor((N,), "float32"),
    y: T.Tensor((N,), "float32"),
):
    with T.Kernel(N // THREADS, threads=THREADS) as bx:
        for i in T.Parallel(THREADS):
            index = bx * THREADS + i
            value = tmp[index] + 1.0
            tmp[index] = value
            y[index] = value
            acc[index] = acc[index] + value

KERNELS = (scale_kernel, bias_kernel)
'''

MSL_COMPARISON = '''kernel void scale_msl(
    device const float* x [[buffer(0)]],
    device float* acc [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < N) {
    float value = x[index] * 2.0f + 1.0f;
    y[index] = value;
    acc[index] = acc[index] + value;
  }
}
/* taic-launch
math safe
launch scale_msl grid=4,1,1 threads=256,1,1
*/
'''


def geometry_document(*, worklist: bool) -> bytes:
    if worklist:
        stages = [
            {
                "kernel": "scale_kernel_kernel", "grid": [8, 1, 1], "threads": [128, 1, 1],
                "workspace": [{"name": "tmp", "zero": False}],
            },
            {
                "kernel": "bias_kernel_kernel", "grid": [8, 1, 1], "threads": [128, 1, 1],
                "workspace": [{"name": "tmp", "zero": False}],
            },
        ]
    else:
        stages = [{"kernel": "scale_kernel_kernel", "grid": [4, 1, 1], "threads": [256, 1, 1]}]
    return (json.dumps({"schema_version": 1, "source": "candidate", "stages": stages}, indent=2) + "\n").encode()


def tilelang_candidate(*, worklist: bool, source: bytes | None = None, geometry: bytes | None = None) -> CandidateArtifact:
    return CandidateArtifact(
        track_id=TRACK_ID,
        entries=(
            CandidateEntry(
                path="kernel.py", role="dsl_source", media_type="text/x-python",
                data=(WORKLIST_TILELANG if worklist else SINGLE_TILELANG).encode() if source is None else bytes(source),
            ),
            CandidateEntry(
                path="host/execution_geometry.json", role="schedule", media_type="application/json",
                data=geometry_document(worklist=worklist) if geometry is None else bytes(geometry),
            ),
        ),
    )


def evaluate_tilelang(locator: Any, registry_path: Path, candidate: CandidateArtifact, *, mode: str = "full", seed: int = 7, reps: int = 3):
    from tcenv.backends.metal.track_factory import adapter_for_track
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import load_track_registry

    spec = load_track_registry(registry_path).get(TRACK_ID, version=TRACK_VERSION)
    adapter = adapter_for_track(spec)
    builder = adapter.resolve("builder")(target_id="apple-m4-max")
    evaluator = adapter.resolve("evaluator")(
        target_id="apple-m4-max", mode=mode, seed=seed, reps=reps,
        cases=("normal", "wide_range", "outlier"),
    )
    return evaluate_task(
        locator,
        track_id=TRACK_ID,
        track_version=TRACK_VERSION,
        candidate=candidate,
        builder=builder,
        evaluator=evaluator,
        geometry_resolver=adapter.resolve("geometry_resolver"),
        track_registry_path=registry_path,
    )


__all__ = [
    "MSL_COMPARISON", "N", "SINGLE_TILELANG", "TRACK_ID", "TRACK_VERSION", "WORKLIST_TILELANG",
    "evaluate_tilelang", "geometry_document", "mt2_opspec", "tilelang_candidate",
    "write_mt2_registry", "write_mt2_task",
]
