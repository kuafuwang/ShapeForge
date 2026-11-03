"""v2 搜索的本地评测执行器：``EvaluationRequest`` → ``run_portfolio_task``。

对编排器而言它和 ``LocalEvaluator`` 是同一个协议（``formal``/``selfcheck``/``confirmation_expected``），
所以 ``_evaluate_report`` 的 provenance/outbox/身份校验一行都不用改。差别只在两点：

- 报告来自 ``evaluate_task`` 的 schema-4 文档（候选几何由自己的 ``taic-launch`` 派生）；
- evaluator epoch 的 task-pack 分量是 locator 的部署包摘要，不是 v1 track 登记的 pack。
"""

from __future__ import annotations

from typing import Any, Callable

from tcenv.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    EvaluatorProvenance,
    _backend_identity,
    _epoch,
    request_digest,
)
from tcenv.schema import PROTOCOL_VERSION
from tcenv.target import load_target

from .task_binding import TaskBinding


class V2LocalEvaluator:
    """@brief 单候选 portfolio 驱动的本地 v2 评测器。"""

    def __init__(
        self,
        binding: TaskBinding,
        *,
        evaluate_fn: Callable[..., Any] | None = None,
        expected_identity: Callable[..., dict] | None = None,
    ) -> None:
        if not binding.is_v2:
            raise ValueError("V2LocalEvaluator needs a v2 task binding")
        self.binding = binding
        #: 测试可注入与 ``search_candidate.evaluate_candidate`` 同签名的替身；生产为 None。
        self._evaluate_fn = evaluate_fn
        self._expected_identity = expected_identity

    def _provenance(self, request: EvaluationRequest) -> EvaluatorProvenance:
        backend, backend_version = _backend_identity(request)
        pack_digest = self.binding.task_pack_digest or ""
        epoch = _epoch(
            site_id="local", device_id=request.target_id, backend=backend,
            backend_version=backend_version, target_id=request.target_id,
            worker_build="", task_pack_digest=pack_digest, calibration_digest="",
            protocol_version=PROTOCOL_VERSION,
        )
        return EvaluatorProvenance(
            schema_version=1, epoch=epoch, site_id="local", device_id=request.target_id,
            backend=backend, backend_version=backend_version, target_id=request.target_id,
            worker_build="", task_pack_digest=pack_digest, calibration_digest="",
            protocol_version=PROTOCOL_VERSION, request_digest=request_digest(request),
            measurement_kind="measured", publication_allowed=True,
        )

    def _rejected(self, request: EvaluationRequest, stage: str, message: str) -> EvaluationResult:
        """@brief 候选自己的交付/几何/安检问题：一份 rejected 报告，而不是炸掉整次搜索。

        @param request 原请求。
        @param stage 停在哪一关（screen/geometry/dispatch）。
        @param message 给模型的诊断。
        @return rejected 的评测结果（无 build，BUILD_FREE_STATUSES 覆盖）。
        """
        from tcenv.evaluate import EvalReport

        report = EvalReport(
            task=request.task, dtype=request.dtype, size=request.size, mode=request.mode,
            seed=request.seed or 0, stage=stage, status="rejected", message=str(message)[:2000],
            target_id=request.target_id, protocol_version=PROTOCOL_VERSION,
        )
        return EvaluationResult(report=report, provenance=self._provenance(request))

    def _evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if request.mode == "validate":
            raise ValueError("the v2 local adapter does not support mode='validate'; use quick/timed/full")
        if request.kind not in ("formal_eval", "selfcheck"):
            raise ValueError(f"unknown evaluation kind {request.kind!r}")
        target = load_target(request.target_id)
        from tcenv.backends.metal.screen import screen_v2_source
        from tcenv.contract_errors import GeometryError

        refused = screen_v2_source(request.source)
        if refused:
            return self._rejected(request, "screen", "v2 candidate refused before compilation: " + refused[0])
        # 先用同一个几何解析器做 CPU 预检：缺/坏的 taic-launch 在这里就是 GeometryError，
        # 而不是 dispatch 深处一个类型不那么明确的 TaskContractError。
        from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver
        from tcenv.track.delivery import candidate_from_single_source

        try:
            msl_single_geometry_resolver(self.binding.locator.load())(candidate_from_single_source(request.source))
        except GeometryError as exc:
            return self._rejected(request, "geometry", str(exc))
        try:
            if self._evaluate_fn is not None:
                run = self._evaluate_fn(request)
            else:
                from tcenv.backends.metal.search_candidate import evaluate_candidate

                run = evaluate_candidate(
                    self.binding.locator,
                    request.source,
                    target=target,
                    mode=request.mode,
                    seed=request.seed if request.seed is not None else 7,
                    reps=request.reps,
                    cases=request.cases,
                )
        except GeometryError as exc:
            # 候选自己的 taic-launch/几何不合法：是这颗核不合格，不是评测架故障。
            return self._rejected(request, "geometry", str(exc))
        result = run.result
        if result is None:
            # dispatch/资源在 portfolio 层就是类型化结果（无 fallback 时没有报告）：
            # 对搜索循环它是一次「候选不合格」，而不是基础设施故障。
            outcome = getattr(run.record, "outcome", "unknown")
            return self._rejected(request, "dispatch", f"the candidate did not produce an evaluation (dispatch outcome {outcome})")
        return EvaluationResult(report=dict(result.report), provenance=self._provenance(request))

    def formal(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "formal_eval":
            raise ValueError("formal() requires kind='formal_eval'")
        return self._evaluate(request)

    def selfcheck(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "selfcheck":
            raise ValueError("selfcheck() requires kind='selfcheck'")
        return self._evaluate(request)

    def confirmation_expected(self, *, task: str, dtype: str, size: str, target_id: str, source: str) -> dict:
        """@brief 当前执行器下独立重建的 expected（绝不复制搜索报告）。

        v2 搜索的确认走 ``confirm_subject``（它自己重建 expected），这个方法主要供调用方/测试
        核对同一身份；实现同样只从合同+几何+当前 builder probe 重建。

        @param task 题名。
        @param dtype 数据类型。
        @param size 尺寸档位。
        @param target_id 目标设备 id。
        @param source 候选源码。
        @return build/evaluation 身份文档。
        """
        if self._expected_identity is not None:
            return self._expected_identity(task=task, dtype=dtype, size=size, target_id=target_id, source=source)
        from tcenv.backends.metal.portfolio_confirmation import confirmation_expected as rebuild_expected
        from tcenv.backends.metal.portfolio_confirmation import _resolved
        from tcenv.backends.metal.search_candidate import domain_for, portfolio_for
        from tcenv.portfolio.candidate import project_provider

        target = load_target(target_id)
        contract = self.binding.locator.load()
        plan, authority = domain_for(contract, target)
        portfolio = portfolio_for(source, plan)
        provider = portfolio.variants[0]
        projection = project_provider(portfolio, provider, authority=authority)
        resolved = _resolved(contract, projection, target)
        return rebuild_expected(contract, projection, resolved, locator=self.binding.locator, target_id=target.id)


__all__ = ["V2LocalEvaluator"]
