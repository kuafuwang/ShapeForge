"""Shared evaluator identity and session-grant business types.

The legacy frame service and the transport-neutral external job engine both use
these types.  Legacy ``epoch``/``provenance`` deliberately retain their original
wire semantics.  External v1 callers must use ``external_epoch`` and
``external_provenance`` so the measurement epoch, API contract and evidence
contract remain separate identities.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .contracts import API_CONTRACT_ID, EVIDENCE_CONTRACT_ID, NPU_API_CONTRACT_IDS, NPU_EVIDENCE_CONTRACT_IDS
from .protocol import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    EvaluatorProvenance,
    provenance_epoch,
)


@dataclass(frozen=True)
class ServiceIdentity:
    """Stable description of the evaluator site and build."""

    site_id: str
    device_id: str
    backend: str
    backend_version: str
    target_id: str
    worker_build: str
    task_pack_digest: str
    calibration_digest: str
    task_fingerprints: frozenset[str] = frozenset()
    # fingerprint -> (task, dtype, size).  When present it is authoritative: a
    # registered fingerprint cannot be reused for a different problem.  Legacy
    # identities that only have ``task_fingerprints`` keep their historical
    # flat-set behavior for resume compatibility.
    task_bindings: Mapping[str, tuple[str, str, str]] | None = None
    measurement_kind: str = "measured"
    publication_allowed: bool = False

    def __post_init__(self) -> None:
        for name in ("site_id", "device_id", "backend", "backend_version", "target_id", "worker_build", "task_pack_digest", "calibration_digest"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"service identity {name} must be non-empty")
        if self.task_bindings is not None:
            if not isinstance(self.task_bindings, Mapping) or not self.task_bindings:
                raise ValueError("service identity task bindings must be a non-empty mapping")
            normalized: dict[str, tuple[str, str, str]] = {}
            triples: set[tuple[str, str, str]] = set()
            for fingerprint, binding in self.task_bindings.items():
                if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                    raise ValueError("service identity task fingerprints must be 64 lowercase hex characters")
                if (
                    not isinstance(binding, (tuple, list)) or len(binding) != 3
                    or any(not isinstance(value, str) or not value for value in binding)
                ):
                    raise ValueError("service identity task bindings must be (task, dtype, size) triples")
                triple = (str(binding[0]), str(binding[1]), str(binding[2]))
                if triple in triples:
                    raise ValueError("service identity registers two fingerprints for the same task")
                triples.add(triple)
                normalized[fingerprint] = triple
            if self.task_fingerprints and set(self.task_fingerprints) != set(normalized):
                raise ValueError("service identity task fingerprints and bindings disagree")
            object.__setattr__(self, "task_bindings", MappingProxyType(normalized))
            object.__setattr__(self, "task_fingerprints", frozenset(normalized))
        if not self.task_fingerprints:
            raise ValueError("service identity must register at least one task fingerprint")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in self.task_fingerprints):
            raise ValueError("service identity task fingerprints must be 64 lowercase hex characters")
        if self.measurement_kind not in {"measured", "simulated"}:
            raise ValueError("measurement_kind must be measured or simulated")

    def accepts_task(self, task: str, dtype: str, size: str, fingerprint: str) -> bool:
        """Bind a registered fingerprint to exactly one (task, dtype, size)."""

        if self.task_bindings is not None:
            return self.task_bindings.get(fingerprint) == (task, dtype, size)
        return fingerprint in self.task_fingerprints

    @property
    def epoch(self) -> str:
        """Legacy evaluator epoch; do not use as the external v1 identity."""

        return provenance_epoch(
            site_id=self.site_id,
            device_id=self.device_id,
            backend=self.backend,
            backend_version=self.backend_version,
            target_id=self.target_id,
            worker_build=self.worker_build,
            task_pack_digest=self.task_pack_digest,
            calibration_digest=self.calibration_digest,
            protocol_version=PROTOCOL_VERSION,
        )

    def external_epoch(self, *, execution_protocol_version: str) -> str:
        """Return the measurement-only epoch used by external v1 jobs."""

        return provenance_epoch(
            site_id=self.site_id,
            device_id=self.device_id,
            backend=self.backend,
            backend_version=self.backend_version,
            target_id=self.target_id,
            worker_build=self.worker_build,
            task_pack_digest=self.task_pack_digest,
            calibration_digest=self.calibration_digest,
            protocol_version=execution_protocol_version,
        )

    def provenance(self, *, request_digest: str = "") -> EvaluatorProvenance:
        """Build the unchanged legacy evidence provenance."""

        return EvaluatorProvenance(
            schema_version=SCHEMA_VERSION,
            epoch=self.epoch,
            site_id=self.site_id,
            device_id=self.device_id,
            backend=self.backend,
            backend_version=self.backend_version,
            target_id=self.target_id,
            worker_build=self.worker_build,
            task_pack_digest=self.task_pack_digest,
            calibration_digest=self.calibration_digest,
            protocol_version=PROTOCOL_VERSION,
            request_digest=request_digest,
            measurement_kind=self.measurement_kind,
            publication_allowed=self.publication_allowed,
        )

    def external_provenance(
        self,
        *,
        execution_protocol_version: str,
        request_digest: str = "",
        attempt_id: str = "",
        api_contract_id: str = API_CONTRACT_ID,
        evidence_contract_id: str = EVIDENCE_CONTRACT_ID,
        npu_execution: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Build the external typed-provenance document.

        Capabilities omit ``attempt_id`` and ``request_digest``; accepted jobs
        always include the digest and results include both fields.  The NPU
        source contract adds the logical device group and execution shape;
        the legacy Metal/source document keeps exactly its original fields.
        """

        document: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "evaluator_epoch": self.external_epoch(execution_protocol_version=execution_protocol_version),
            "site_id": self.site_id,
            "device_id": self.device_id,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "target_id": self.target_id,
            "worker_build": self.worker_build,
            "task_pack_digest": self.task_pack_digest,
            "calibration_digest": self.calibration_digest,
            "protocol_version": execution_protocol_version,
            "api_contract_id": api_contract_id,
            "evidence_contract_id": evidence_contract_id,
            "measurement_kind": self.measurement_kind,
            "publication_allowed": self.publication_allowed,
        }
        if api_contract_id in NPU_API_CONTRACT_IDS:
            if evidence_contract_id not in NPU_EVIDENCE_CONTRACT_IDS:
                raise ValueError("NPU provenance requires the NPU evidence contract")
            if not isinstance(npu_execution, Mapping):
                raise ValueError("NPU provenance requires a validated execution binding")
            from .contracts import npu_execution_document

            execution = npu_execution_document(npu_execution)
            document["device_group"] = execution["device_group"]
            document["artifact_kind"] = execution["artifact_kind"]
            document["launch_kind"] = execution["launch_kind"]
            document["launch_subtype"] = execution["launch_subtype"]
            document["timing_sources"] = execution["required_timing_sources"]
        elif npu_execution is not None:
            raise ValueError("the legacy contract cannot carry an NPU execution binding")
        if attempt_id:
            document["attempt_id"] = attempt_id
        if request_digest:
            document["request_digest"] = request_digest
        return document


@dataclass(frozen=True)
class SessionGrant:
    """Authority carried by one legacy token or external per-RPC grant.

    ``token`` is an in-memory input only.  The durable job engine stores its
    SHA-256 lookup digest plus ``subject``/scope, never the raw token.  New
    external grants should set an explicit stable ``subject`` so a renewed token
    can reattach to the same jobs.  Empty ``allowed_modes`` preserves the legacy
    kind-specific mode defaults.
    """

    token: str
    run_id: str
    session_id: str
    target_id: str
    allowed_kinds: frozenset[str] = frozenset({"formal_eval", "selfcheck"})
    expires_at: float | None = None
    max_timed: int = 20
    max_gpu_s: float = 120.0
    subject: str = ""
    allowed_modes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in ("token", "run_id", "session_id", "target_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"session grant {name} must be non-empty")
        if not self.allowed_kinds or not self.allowed_kinds <= {"formal_eval", "selfcheck"}:
            raise ValueError("session grant kinds are invalid")
        if self.expires_at is not None and (isinstance(self.expires_at, bool) or not isinstance(self.expires_at, (int, float)) or not math.isfinite(self.expires_at)):
            raise ValueError("session grant expiry must be a finite timestamp or null")
        if isinstance(self.max_timed, bool) or not isinstance(self.max_timed, int) or self.max_timed < 1:
            raise ValueError("session grant timed quota must be a positive integer")
        if isinstance(self.max_gpu_s, bool) or not isinstance(self.max_gpu_s, (int, float)) or not math.isfinite(self.max_gpu_s) or self.max_gpu_s <= 0:
            raise ValueError("session grant GPU quota must be finite and positive")
        if not isinstance(self.subject, str):
            raise ValueError("session grant subject must be a string")
        if self.subject and (len(self.subject.encode("utf-8")) > 160 or not re.fullmatch(r"[A-Za-z0-9._:-]+", self.subject)):
            raise ValueError("session grant subject must be a bounded opaque identifier")
        valid_modes = {"quick", "timed", "full", "validate"}
        if not self.allowed_modes <= valid_modes:
            raise ValueError("session grant modes are invalid")
        for kind in self.allowed_kinds:
            kind_modes = self.modes_for(kind)
            if not kind_modes:
                raise ValueError(f"session grant has no usable mode for {kind}")

    @property
    def grant_subject(self) -> str:
        """Stable non-secret subject; derive a digest for legacy callers."""

        if self.subject:
            return self.subject
        return "legacy-" + hashlib.sha256(self.token.encode("utf-8")).hexdigest()[:32]

    def modes_for(self, kind: str) -> frozenset[str]:
        defaults = (
            frozenset({"quick", "full", "validate"})
            if kind == "formal_eval"
            else frozenset({"quick", "timed"})
        )
        modes = self.allowed_modes or defaults
        return frozenset(modes) & defaults

    def scope_document(self) -> dict[str, object]:
        """Canonical scope used for capability and idempotency binding."""

        modes = sorted(set().union(*(self.modes_for(kind) for kind in self.allowed_kinds)))
        return {
            "subject": self.grant_subject,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "kinds": sorted(self.allowed_kinds),
            "target_id": self.target_id,
            "modes": modes,
        }
