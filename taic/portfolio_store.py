"""P2-D: durable portfolio-run sidecar, generation floors and invalidation records."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from taic.portfolio_manifest import DispatchManifest, ManifestSubject
from tcenv.contract_errors import TaskContractError
from tcenv.identity import IdentityError, fingerprint, load_strict_json
from tcenv.portfolio_evidence.projection import _digest, _integer, _json_closure, _keys, _string, strict_json_equal
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan

PORTFOLIO_RUN_SCHEMA_ID = "tcenv.portfolio.run.1"
PORTFOLIO_REVOCATION_FILENAME = "portfolio_revocation.json"
PORTFOLIO_RUN_KEYS = frozenset(
    {
        "schema_id",
        "portfolio_freeze_digest",
        "validation_plan_digest",
        "run_config_digest",
        "dispatch_manifest_digest",
        "activation_generation",
        "status",
        "coverage_kind",
        "subjects",
        "invalidated_artifacts",
    }
)


@dataclass(frozen=True)
class PortfolioRun:
    portfolio_freeze_digest: str
    validation_plan_digest: str
    run_config_digest: str
    dispatch_manifest_digest: str
    activation_generation: int
    status: str
    subjects: tuple[ManifestSubject, ...]
    invalidated_artifacts: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("portfolio_freeze_digest", "validation_plan_digest", "run_config_digest", "dispatch_manifest_digest"):
            _digest(getattr(self, name), where=f"portfolio_run.{name}")
        _integer(self.activation_generation, where="portfolio_run.activation_generation", minimum=1)
        if self.status not in ("active", "revoked"):
            raise TaskContractError("portfolio_run.status must be active or revoked")
        if self.status == "active" and self.invalidated_artifacts:
            raise TaskContractError("an active portfolio_run cannot carry invalidation records")
        if self.status == "revoked" and not self.invalidated_artifacts:
            raise TaskContractError("a revoked portfolio run needs an invalidation record")
        if not self.subjects or any(not isinstance(subject, ManifestSubject) for subject in self.subjects):
            raise TaskContractError("portfolio_run needs the manifest's subject identities")
        keys = [subject.instance_key for subject in self.subjects]
        if len(set(keys)) != len(keys):
            raise TaskContractError("portfolio_run cannot contain duplicate instance keys")

    @classmethod
    def from_manifest(
        cls,
        freeze: PortfolioFreeze,
        validation: ValidationPlan,
        manifest: DispatchManifest,
        *,
        run_config_digest: str,
    ) -> "PortfolioRun":
        if manifest.status != "active":
            raise TaskContractError("only an active DispatchManifest can be checkpointed")
        if not manifest._derived_from_confirmations:
            raise TaskContractError("a document-only manifest cannot be checkpointed")
        if manifest.portfolio_freeze_digest != freeze.digest or manifest.validation_plan_digest != validation.digest:
            raise TaskContractError("checkpoint manifest belongs to another freeze or plan")
        manifest.require_complete_coverage(freeze, validation)
        return cls(
            portfolio_freeze_digest=freeze.digest,
            validation_plan_digest=validation.digest,
            run_config_digest=_digest(run_config_digest, where="run_config.digest"),
            dispatch_manifest_digest=manifest.digest,
            activation_generation=manifest.activation_generation,
            status="active",
            subjects=manifest.subjects,
        )

    @classmethod
    def from_document(cls, value: Any) -> "PortfolioRun":
        _json_closure(value, where="portfolio_run")
        document = _keys(value, PORTFOLIO_RUN_KEYS, where="portfolio_run")
        if document["schema_id"] != PORTFOLIO_RUN_SCHEMA_ID:
            raise TaskContractError("unknown portfolio_run schema")
        if document["coverage_kind"] != "full_finite":
            raise TaskContractError("portfolio_run.coverage_kind must be full_finite")
        from taic.portfolio_manifest import ManifestSubject as Subject

        subjects = document["subjects"]
        if not isinstance(subjects, list):
            raise TaskContractError("portfolio_run.subjects must be an array")
        invalidated = document["invalidated_artifacts"]
        if not isinstance(invalidated, list):
            raise TaskContractError("portfolio_run.invalidated_artifacts must be an array")
        for record in invalidated:
            if not isinstance(record, dict) or set(record) != {"activation_generation", "reason"}:
                raise TaskContractError("each invalidation record has exactly generation and reason")
            _integer(record["activation_generation"], where="invalidated_artifacts.generation", minimum=1)
            _string(record["reason"], where="invalidated_artifacts.reason")
        return cls(
            portfolio_freeze_digest=document["portfolio_freeze_digest"],
            validation_plan_digest=document["validation_plan_digest"],
            run_config_digest=document["run_config_digest"],
            dispatch_manifest_digest=document["dispatch_manifest_digest"],
            activation_generation=_integer(document["activation_generation"], where="portfolio_run.activation_generation", minimum=1),
            status=document["status"],
            subjects=tuple(Subject.from_document(item) for item in subjects),
            invalidated_artifacts=tuple(invalidated),
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema_id": PORTFOLIO_RUN_SCHEMA_ID,
            "portfolio_freeze_digest": self.portfolio_freeze_digest,
            "validation_plan_digest": self.validation_plan_digest,
            "run_config_digest": self.run_config_digest,
            "dispatch_manifest_digest": self.dispatch_manifest_digest,
            "activation_generation": self.activation_generation,
            "status": self.status,
            "coverage_kind": "full_finite",
            "subjects": [subject.document() for subject in self.subjects],
            "invalidated_artifacts": [dict(record) for record in self.invalidated_artifacts],
        }

    @property
    def digest(self) -> str:
        return fingerprint(self.document())

    def _write_atomic(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".tmp-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.document(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def write_atomic(self, run_dir: str | Path) -> None:
        self._write_atomic(Path(run_dir) / "portfolio_run.json")

    def write_revocation_marker(self, run_dir: str | Path) -> None:
        """Persist a fail-closed tombstone before replacing the live sidecar."""
        if self.status != "revoked":
            raise TaskContractError("only a revoked portfolio_run can write a revocation marker")
        self._write_atomic(Path(run_dir) / PORTFOLIO_REVOCATION_FILENAME)

    @classmethod
    def read_revocation_marker(cls, run_dir: str | Path) -> "PortfolioRun | None":
        path = Path(run_dir) / PORTFOLIO_REVOCATION_FILENAME
        if not path.exists():
            return None
        try:
            value = load_strict_json(path)
        except (IdentityError, OSError) as exc:
            raise TaskContractError(f"cannot read portfolio revocation marker: {exc}") from exc
        marker = cls.from_document(value)
        if marker.status != "revoked":
            raise TaskContractError("portfolio revocation marker is not revoked")
        return marker

    @classmethod
    def read(
        cls,
        run_dir: str | Path,
        *,
        freeze: PortfolioFreeze,
        validation: ValidationPlan,
        manifest: DispatchManifest,
        minimum_generation: int = 1,
        run_config_digest: str,
    ) -> "PortfolioRun":
        try:
            value = load_strict_json(Path(run_dir) / "portfolio_run.json")
        except (IdentityError, OSError) as exc:
            raise TaskContractError(f"cannot read portfolio_run.json: {exc}") from exc
        run = cls.from_document(value)
        if manifest.status != "active":
            raise TaskContractError("a non-active DispatchManifest cannot resume execution")
        if manifest.portfolio_freeze_digest != freeze.digest or manifest.validation_plan_digest != validation.digest:
            raise TaskContractError("manifest belongs to another freeze or validation plan")
        if run.portfolio_freeze_digest != freeze.digest or run.validation_plan_digest != validation.digest:
            raise TaskContractError("portfolio_run belongs to another freeze or validation plan")
        validation.validate_for(freeze)
        if run.dispatch_manifest_digest != manifest.digest:
            raise TaskContractError("portfolio_run is not the current DispatchManifest")
        if run.run_config_digest != run_config_digest:
            raise TaskContractError("portfolio_run.run_config_digest does not match the caller's run_config")
        if run.subjects != manifest.subjects:
            raise TaskContractError("portfolio_run subjects differ from the current DispatchManifest")
        if not manifest._derived_from_confirmations:
            raise TaskContractError("a document-only manifest cannot resume execution")
        manifest.require_complete_coverage(freeze, validation)
        if run.status != "active":
            raise TaskContractError("a revoked portfolio_run cannot resume execution")
        if run.activation_generation != manifest.activation_generation:
            raise TaskContractError("portfolio_run generation differs from the current manifest")
        if run.activation_generation < minimum_generation:
            raise TaskContractError("portfolio_run is below the required generation floor")
        return run

    def revoke(self, *, reason: str) -> "PortfolioRun":
        if self.status != "active":
            raise TaskContractError("only an active portfolio_run can be revoked")
        _string(reason, where="portfolio_run.revocation_reason")
        return PortfolioRun(
            portfolio_freeze_digest=self.portfolio_freeze_digest,
            validation_plan_digest=self.validation_plan_digest,
            run_config_digest=self.run_config_digest,
            dispatch_manifest_digest=self.dispatch_manifest_digest,
            activation_generation=self.activation_generation + 1,
            status="revoked",
            subjects=self.subjects,
            invalidated_artifacts=(
                *self.invalidated_artifacts,
                {"activation_generation": self.activation_generation + 1, "reason": reason},
            ),
        )


__all__ = ["PORTFOLIO_REVOCATION_FILENAME", "PORTFOLIO_RUN_SCHEMA_ID", "PortfolioRun"]
