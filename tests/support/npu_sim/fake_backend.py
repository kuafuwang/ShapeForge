"""A deterministic CPU backend for remote-service tests.

It intentionally has no Metal, CUDA, NPU or model dependency.  Test reports are
marked simulated so they can never be mistaken for publishable measurements.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from tcenv.remote.protocol import EvaluationRequest


class FakeNpuBackend:
    """Deterministic evaluator used behind the real remote service core."""

    def evaluate(self, request: EvaluationRequest) -> dict:
        counter = os.environ.get("NPU_SIM_COUNTER")
        if counter:
            path = Path(counter)
            count = int(path.read_text()) if path.is_file() else 0
            path.write_text(str(count + 1))
        # A deliberate slow path exercises worker timeout/cancellation without
        # requiring a device or a real kernel.
        if "REMOTE_SLEEP" in request.source:
            time.sleep(5.0)
        digest = hashlib.sha256((request.source + str(request.seed)).encode("utf-8")).hexdigest()
        receipt_digest = hashlib.sha256((request.source + str(request.seed) + request.request_id).encode("utf-8")).hexdigest()
        receipt_id = "sim-replay" if "REPLAY_RECEIPT" in request.source else f"sim-{receipt_digest[:16]}"
        candidate_us = 100.0 + int(digest[:8], 16) % 1000
        report = {
            "task": request.task,
            "dtype": request.dtype,
            "size": request.size,
            "mode": request.mode,
            "seed": request.seed,
            "status": "ok",
            "stage": "done",
            "candidate_us": candidate_us,
            "speedup": 1.0,
            "elapsed_s": 0.5 if request.mode == "timed" else 0.0,
            "target_id": request.target_id,
            "protocol_version": "3",
            "simulated": True,
        }
        return {
            "report": report,
            "evidence": {
                "request_id": request.request_id,
                "candidate_sha256": request.candidate_sha256,
                "measurement_kind": "simulated",
            },
            "receipt": {
                "receipt_id": receipt_id,
                "request_id": request.request_id,
                "request_digest": request.request_digest,
                "measurement_kind": "simulated",
            },
        }
