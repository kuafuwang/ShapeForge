"""B2 GPU slice: the preserved naive candidates run on the real v2 evaluation path.

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
B2 = ROOT / "tests" / "support" / "golden" / "shared" / "跨硬件算子生成分工-阶段验收" / "B2"
gpu = pytest.mark.gpu

#: The v1 launch numbers for the small/fp16 instance, and the one constant the candidate owns.
SLICES = {
    # conv2d small: OC/64 = 1, ceil(N*OH*OW/64) = ceil(1024/64) = 16；合同头给 N..K_TOTAL，候选只给 BLOCK_*。
    "conv2d": {
        "kernel": "conv2d_kernel",
        "grid": (1, 16, 1),
        "threads": (128, 1, 1),
        "constants": (
            # M_TOTAL/K_TOTAL 是 v1 LoweringProfile 的派生量，不是 OpSpec 符号：
            # v2 合同头不提供它们，候选必须自己声明（候选自有的派生标量）。
            "constant uint M_TOTAL = 1024;\n"  # N*OH*OW = 4*16*16
            "constant uint K_TOTAL = 288;\n"   # KH*KW*C = 3*3*32
            "constant uint BLOCK_M = 64;\nconstant uint BLOCK_N = 64;\nconstant uint BLOCK_K = 32;\n"
        ),
    },
    # lion small: N/1024 = 64；合同头给 N/LR/WD/BETA1/BETA2，候选只给 BLOCK_SIZE。
    "lion": {"kernel": "lion_kernel", "grid": (64, 1, 1), "threads": (128, 1, 1), "constants": "constant uint BLOCK_SIZE = 1024;\n"},
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
    contract = load_task_contract(directory, dtype="fp16", size="small")
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
    body = (B2 / "evidence" / "candidates" / f"{task}-naive.msl").read_text(encoding="utf-8").replace("{{T}}", "half")
    grid = ",".join(str(v) for v in spec["grid"])
    threads = ",".join(str(v) for v in spec["threads"])
    launch = f"/* taic-launch\nmath fast\nlaunch {spec['kernel']} grid={grid} threads={threads}\n*/\n"
    source = launch + spec["constants"] + body
    return evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="timed", seed=7, reps=3, cases=("normal", "wide_range", "outlier")),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=ROOT / "backends" / "metal" / "tracks.json",
    )


@gpu
@pytest.mark.parametrize("task", sorted(SLICES))
def test_b2_gpu_the_naive_candidate_compiles_and_is_correct_on_the_v2_path(task):
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
    write_gpu_observation(B2, task, record)
