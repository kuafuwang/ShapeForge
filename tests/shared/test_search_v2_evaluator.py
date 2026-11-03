"""V2LocalEvaluator 的候选级拒绝：交付/几何/安检问题是 rejected，不是整次搜索的崩溃。"""

from __future__ import annotations

from taic.task_binding import bind_task
from taic.v2_evaluation import V2LocalEvaluator
from tcenv.evaluation import EvaluationRequest


def _request(source: str) -> EvaluationRequest:
    return EvaluationRequest(task="gemm", dtype="fp16", size="small", source=source, mode="quick", seed=7)


def test_a_candidate_without_a_launch_block_is_rejected_not_a_crash():
    evaluator = V2LocalEvaluator(bind_task("gemm", "fp16", "small"))
    result = evaluator.formal(_request("kernel void gemm_kernel() {}\n"))
    assert result.report.status == "rejected"
    assert result.report.stage == "geometry"
    assert "taic-launch" in result.report.message


def test_a_host_side_include_is_refused_before_any_build():
    evaluator = V2LocalEvaluator(bind_task("gemm", "fp16", "small"))
    source = "#include <stdio.h>\n/* taic-launch\nlaunch gemm_kernel grid=1,1,1 threads=1,1,1\n*/\nkernel void gemm_kernel() {}\n"
    result = evaluator.formal(_request(source))
    assert result.report.status == "rejected"
    assert result.report.stage == "screen"
    assert result.report.build is None
