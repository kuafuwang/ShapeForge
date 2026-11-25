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
        from dataclasses import replace
        from taic.task_binding import bind_task
        from taic.v2_evaluation import V2LocalEvaluator
        from tcenv.track.envelope import EnvelopeError, decode_envelope, is_candidate_envelope

        track_id, track_version = "msl-single", None
        if is_candidate_envelope(request.source):
            try:
                candidate = decode_envelope(request.source)
            except (EnvelopeError, UnicodeDecodeError) as exc:
                raise ProtocolError(f"remote candidate envelope is invalid: {exc}") from exc
            # A remote durable row has no out-of-band run_config from which to recover a registry
            # default.  The versioned candidate manifest binds the selection into candidate_sha256;
            # refusing older unversioned envelopes here prevents silent upgrades on replay.
            if candidate.track_version is None:
                raise ProtocolError(
                    "remote extended delivery requires an explicit taic-track-version; "
                    "unversioned envelopes are local-only"
                )
            track_id, track_version = candidate.track_id, candidate.track_version
        binding = replace(
            bind_task(request.task, request.dtype, request.size, target_id=request.target_id),
            track_id=track_id,
            track_version=track_version,
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
