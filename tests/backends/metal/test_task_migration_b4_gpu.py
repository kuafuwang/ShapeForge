"""B4 GPU slice: the preserved naive candidates run on the real v2 evaluation path.

CPU equivalence (144 frozen records) is in ``test_task_migration_b1.py``.  This module is the
other half of the migration evidence: the hand-written naive kernel of each migrated task is
compiled, ABI-checked, correctness-checked and timed by ``evaluate_task`` on the real Mac.

The candidate is the archived ``evidence/candidates/<task>-naive.msl`` body plus:

* a ``taic-launch`` block carrying the *v1* launch numbers (grid/threads), because v2 quite
  deliberately never invents a geometry for a candidate;
* only the candidate-owned compile-time constant (``BLOCK_M`` / ``BLOCK_S``).  The contract
  symbols and attrs (``M``/``K``, ``B``/``H``/``S``/``D``/``BASE``) are supplied by the v2
  contract header — declaring them again is a redefinition error, which is exactly the
  candidate/harness boundary working as intended.

Marked ``gpu``: run only with explicit GPU permission, and never as part of the CPU suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.migration_gpu_evidence import write_gpu_observation

ROOT = Path(__file__).resolve().parents[3]
B4 = ROOT / "tests" / "support" / "golden" / "shared" / "跨硬件算子生成分工-阶段验收" / "B4"
gpu = pytest.mark.gpu

#: The v1 launch numbers for the small/fp16 instance, and the one constant the candidate owns.
SLICES = {
    # sem_attention_odd_scale small: S/BLOCK_M = 128/32 = 4, B*H = 2；与 flash_attention 同几何，
    # 只是 scale 由 sum_with_log2e 改为合同 attr SM_SCALE 决定（候选无需自己算 scale）。
    "sem_attention_odd_scale": {
        "kernel": "flash_attention_kernel",
        "grid": (4, 2, 1),
        "threads": (128, 1, 1),
        "constants": "constant uint BLOCK_M = 32;\n",
        "replace": {
            "{{TPR}}": "4",
            "{{SHUFFLES}}": "        s += simd_shuffle_xor(s, 1);\n        s += simd_shuffle_xor(s, 2);\n",
        },
    },
    # sem_int_5 small: N/BLOCK = 4096/256 = 16。K 是标量 buffer（constant int& K [[buffer(2)]]），
    # v2 的 msl-single 没有派生标量绑定，与 bitdelta 一样改用合同头提供的编译常量 SCALAR_K。
    "sem_int_5": {
        "kernel": "sem_int_kernel",
        "grid": (16, 1, 1),
        "threads": (64, 1, 1),  # v1 sem_int profile 是 64 threads（per_thread = 256/64 = 4），不是 128
        "constants": "constant uint BLOCK = 256;\n",
        "replace": {},
        "adapt": "sem_int_scalar_to_const",
        "cases": ("random", "small", "edge"),
    },
}


def _evaluate(task: str):
    """@brief 用归档 naive 候选在真实 v2 路径上评测一题（small/fp16）。

    @param task 任务名。
    @return ``evaluate_task`` 的结果。
    """
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract import load_task_contract
    from tcenv.taskcontract.locator import TaskLocator
    from tcenv.taskcontract.pack import task_pack_closure
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import REPO_ROOT

    spec = SLICES[task]
    directory = ROOT / "tasks" / task
    dtype = "i32" if task.startswith("sem_int_") else "fp16"
    contract = load_task_contract(directory, dtype=dtype, size="small")
    locator = TaskLocator(
        task_id=task,
        op_name=contract.op_name,
        dtype=contract.dtype,
        size=contract.size,
        pack_digest=contract.pack_digest,
        deployment_pack_digest=task_pack_closure(directory, root=REPO_ROOT).digest,
        task_dir=directory,
        code_root=REPO_ROOT,
    )
    body = (B4 / "evidence" / "candidates" / f"{task}-naive.msl").read_text(encoding="utf-8").replace("{{T}}", "half")
    for needle, literal in spec.get("replace", {}).items():
        body = body.replace(needle, literal)
    if spec.get("adapt") == "bitdelta_scalar_to_const":
        body = body.replace("constant float& scale [[buffer(2)]],\n", "")
        body = body.replace("y [[buffer(3)]]", "y [[buffer(2)]]")
        body = body.replace("scale", "SCALE")
    if spec.get("adapt") == "sem_int_scalar_to_const":
        body = body.replace("constant int& K [[buffer(2)]], ", "")  # K 与 out 同一行，连逗号与空格一起删
        body = body.replace("out [[buffer(3)]]", "out [[buffer(2)]]")
        body = body.replace("K", "SCALAR_K")
    grid = ",".join(str(v) for v in spec["grid"])
    threads = ",".join(str(v) for v in spec["threads"])
    launch = f"/* taic-launch\nmath fast\nlaunch {spec['kernel']} grid={grid} threads={threads}\n*/\n"
    source = launch + spec["constants"] + body
    return evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="timed", seed=7, reps=3, cases=spec.get("cases", ("normal", "wide_range", "outlier"))),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=ROOT / "backends" / "metal" / "tracks.json",
    )


@gpu
@pytest.mark.parametrize("task", sorted(SLICES))
def test_b4_gpu_the_naive_candidate_compiles_and_is_correct_on_the_v2_path(task):
    """@brief 真实切片：编译、ABI、正确性、计时与 dispatch 身份全部成立，并写入证据文件。"""
    result = _evaluate(task)
    report = result.report
    assert report["status"] == "ok", report.get("compile", {}).get("errors") or report.get("message")
    assert report["abi"]["ok"] is True
    assert report["correctness"]["ok"] is True
    assert report["benchmark"]["candidate"]["median_us"] > 0
    spec = SLICES[task]
    observed = [
        (record["kernel"], tuple(record["grid"]), tuple(record["threads"]))
        for record in result.execution["dispatch"]
    ]
    assert observed == [(spec["kernel"], spec["grid"], spec["threads"])], observed
    record = {
        "task": task,
        "dtype": "fp16",
        "size": "small",
        "candidate_sha256": result.candidate_sha256,
        "geometry_identity": result.geometry_identity,
        "bundle_digest": result.bundle_digest,
        "artifact_set_digest": result.artifact_set_digest,
        "dispatch": observed,
        "status": report["status"],
        "abi_ok": report["abi"]["ok"],
        "correctness_ok": report["correctness"]["ok"],
        "median_us": report["benchmark"]["candidate"]["median_us"],
    }
    write_gpu_observation(B4, task, record)
