"""Production backend adapter for ``python -m tcenv.remote.worker``.

No fake or simulation composition root is imported here.  The adapter is used
only when the service is deployed with the production worker module.
"""

from __future__ import annotations

from .protocol import EvaluationRequest, ProtocolError


class ProductionBackend:
    """Delegate one request to the existing local ``tcenv.evaluate`` pipeline."""

    def evaluate(self, request: EvaluationRequest) -> dict:
        if request.mode == "validate":
            # Public tcenv.evaluate() accepts only quick/timed/full; its fallback
            # would otherwise silently turn validate into quick.
            raise ProtocolError("production validate mode is not mapped; use quick/timed/full")
        # Import lazily: the protocol/service core must remain usable without
        # importing a device runtime or a backend in the parent process.
        from tcenv.evaluate import evaluate

        report = evaluate(
            request.task,
            request.dtype,
            request.size,
            request.source,
            mode=request.mode,
            seed=request.seed,
            reps=request.reps,
            cases=list(request.cases) if request.cases is not None else None,
            fast_bench=request.fast_bench,
            require_contract_launch=request.require_contract_launch,
            validate=request.validate,
            measure_serial=request.measure_serial,
            eval_cache=request.cache_policy in {"reuse", "refresh"},
            eval_cache_refresh=request.cache_policy == "refresh",
            target_id=request.target_id,
        )
        return {
            "report": report.to_json(),
            "evidence": {"candidate_sha256": request.candidate_sha256},
            "receipt": {"request_id": request.request_id, "request_digest": request.request_digest},
        }


def evaluate_request(request: EvaluationRequest) -> dict:
    """Function form used by the worker entry point."""
    return ProductionBackend().evaluate(request)
