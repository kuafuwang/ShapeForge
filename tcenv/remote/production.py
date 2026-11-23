"""Production backend adapter for ``python -m tcenv.remote.worker``.

No fake or simulation composition root is imported here.  The adapter is used
only when the service is deployed with the production worker module.
"""

from __future__ import annotations

from .protocol import EvaluationRequest, ProtocolError
from tcenv.evaluation import receipt_id_for


class ProductionBackend:
    """Delegate one request to the existing local ``tcenv.evaluate`` pipeline."""

    def evaluate(self, request: EvaluationRequest) -> dict:
        if request.mode == "validate":
            # Public tcenv.evaluate() accepts only quick/timed/full; its fallback
            # would otherwise silently turn validate into quick.
            raise ProtocolError("production validate mode is not mapped; use quick/timed/full")
        if request.cache_policy != "disabled":
            # V2LocalEvaluator deliberately has no v1 eval_cache knob. Do not silently
            # turn reuse/refresh into a different cache contract; deployments must ask for
            # disabled explicitly until a v2 cache policy is implemented and evidenced.
            raise ProtocolError(
                "the v2 remote worker currently requires cache_policy='disabled'; "
                "reuse/refresh are not implemented"
            )
        from taic.task_binding import bind_task
        from taic.v2_evaluation import V2LocalEvaluator

        binding = bind_task(
            request.task, request.dtype, request.size, target_id=request.target_id,
        )
        evaluator = V2LocalEvaluator(binding)
        result = evaluator.formal(request) if request.kind == "formal_eval" else evaluator.selfcheck(request)
        return {
            "report": dict(result.report),
            "evidence": {"candidate_sha256": request.candidate_sha256},
            "receipt": {"receipt_id": receipt_id_for(request), "request_id": request.request_id, "request_digest": request.request_digest, "measurement_kind": "measured"},
        }


def evaluate_request(request: EvaluationRequest) -> dict:
    """Function form used by the worker entry point."""
    return ProductionBackend().evaluate(request)
