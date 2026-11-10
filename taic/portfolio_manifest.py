"""P2-C: DispatchManifest activation/revocation and the minimal active pointer.

A manifest is derived only from a complete set of currently confirmed subject sidecars.  It never
contains a second copy of the schema-4 evidence and never publishes speedup.  The active pointer
stores only ``dispatch_manifest_digest`` and ``activation_generation``; stale or revoked pointers
are refused by the same production reader used by tests.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from taic.policy import build_identity_view, evaluation_identity_view
from tcenv.contract_errors import TaskContractError
from tcenv.identity import IdentityError, fingerprint, load_strict_json
from tcenv.portfolio_evidence.projection import _digest, _integer, _json_closure, _keys, _string, strict_json_equal
from taic.portfolio_confirmation import validate_confirmation_for_activation
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan

MANIFEST_SCHEMA_ID = "tcenv.portfolio.dispatch-manifest.1"
ACTIVE_POINTER_SCHEMA_ID = "tcenv.portfolio.active-pointer.1"
MANIFEST_STATUSES = frozenset({"validated", "active", "revoked"})
COVERAGE_KINDS = frozenset({"sampled_unseen", "tune_confirmation"})
MANIFEST_KEYS = frozenset(
    {
        "schema_id",
        "portfolio_freeze_digest",
        "validation_plan_digest",
        "status",
        "activation_generation",
        "subjects",
        "revocation_reason",
        "dispatch_manifest_digest",
    }
)
SUBJECT_KEYS = frozenset(
    {
        "instance_key",
        "usage",
        "provider",
        "variant_id",
        "fallback_id",
        "candidate_sha256",
        "credits_variant",
        "stable_confirmation_digest",
        "coverage_kind",
    }
)
POINTER_KEYS = frozenset({"schema_id", "dispatch_manifest_digest", "activation_generation"})


def _stable_confirmation_digest(document: Mapping[str, Any]) -> str:
    return fingerprint(
        {
            "schema_id": "tcenv.portfolio.subject-confirmation.1",
            "instance_key": document["instance_key"],
            "usage": document["usage"],
            "provider": document["provider"],
            "variant_id": document["variant_id"],
            "fallback_id": document["fallback_id"],
            "candidate_sha256": document["candidate_sha256"],
            "credits_variant": document["credits_variant"],
            "search_seed": document["search_seed"],
            "confirmation_seed": document["confirmation_seed"],
            "expected": document["expected"],
            "identity": document["identity"],
            "coverage_kind": document["coverage_kind"],
            "search_build": build_identity_view(document["search_report"]),
            "confirmation_build": build_identity_view(document["confirmation_report"]),
            "search_evaluation": evaluation_identity_view(document["search_report"]),
            "confirmation_evaluation": evaluation_identity_view(document["confirmation_report"]),
        }
    )


@dataclass(frozen=True)
class ManifestSubject:
    instance_key: str
    usage: str
    provider: str
    variant_id: str | None
    fallback_id: str | None
    candidate_sha256: str
    credits_variant: bool
    stable_confirmation_digest: str
    coverage_kind: str

    def __post_init__(self) -> None:
        parts = self.instance_key.split("/")
        if len(parts) != 3 or self.usage != parts[2]:
            raise TaskContractError("manifest subject usage is not the instance_key's usage")
        if self.provider not in ("candidate", "fallback"):
            raise TaskContractError("unknown manifest provider")
        if self.provider == "candidate":
            if self.variant_id is None or self.fallback_id is not None or self.credits_variant is not True:
                raise TaskContractError("a candidate subject has inconsistent variant identity/credit")
            _digest(self.variant_id, where="manifest.variant_id")
        else:
            if self.fallback_id is None or self.variant_id is not None or self.credits_variant is not False:
                raise TaskContractError("a fallback subject must not credit a generated variant")
            _digest(self.fallback_id, where="manifest.fallback_id")
        _digest(self.candidate_sha256, where="manifest.candidate_sha256")
        _digest(self.stable_confirmation_digest, where="manifest.stable_confirmation_digest")
        if self.coverage_kind not in COVERAGE_KINDS:
            raise TaskContractError("unknown subject coverage kind")

    @classmethod
    def from_confirmation(cls, document: Mapping[str, Any]) -> "ManifestSubject":
        if document["status"] != "confirmed":
            raise TaskContractError(f"{document['instance_key']}: {document['reason'] or 'not confirmed'}")
        if (document["identity"] or {}).get("kind") != "match":
            raise TaskContractError(f"{document['instance_key']}: identity does not match")
        return cls(
            instance_key=document["instance_key"],
            usage=document["usage"],
            provider=document["provider"],
            variant_id=document["variant_id"],
            fallback_id=document["fallback_id"],
            candidate_sha256=document["candidate_sha256"],
            credits_variant=document["credits_variant"],
            stable_confirmation_digest=_stable_confirmation_digest(document),
            coverage_kind=document["coverage_kind"],
        )

    @classmethod
    def from_document(cls, value: Any) -> "ManifestSubject":
        document = _keys(value, SUBJECT_KEYS, where="manifest subject")
        return cls(**{key: document[key] for key in SUBJECT_KEYS})

    def document(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in SUBJECT_KEYS}


@dataclass(frozen=True)
class DispatchManifest:
    portfolio_freeze_digest: str
    validation_plan_digest: str
    status: str
    activation_generation: int
    subjects: tuple[ManifestSubject, ...]
    revocation_reason: str | None = None
    _derived_from_confirmations: bool = field(default=False, compare=False, repr=False)
    _digest: str = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        _digest(self.portfolio_freeze_digest, where="manifest.portfolio_freeze_digest")
        _digest(self.validation_plan_digest, where="manifest.validation_plan_digest")
        if self.status not in MANIFEST_STATUSES:
            raise TaskContractError("unknown DispatchManifest status")
        _integer(self.activation_generation, where="manifest.activation_generation", minimum=0)
        if self.status == "validated" and self.activation_generation != 0:
            raise TaskContractError("a validated manifest has generation zero")
        if self.status != "validated" and self.activation_generation < 1:
            raise TaskContractError("active/revoked manifests need a positive generation")
        if self.status == "revoked":
            _string(self.revocation_reason, where="manifest.revocation_reason")
        elif self.revocation_reason is not None:
            raise TaskContractError("only a revoked manifest carries a revocation reason")
        if not isinstance(self.subjects, tuple) or not self.subjects:
            raise TaskContractError("a DispatchManifest needs a non-empty subject tuple")
        if any(not isinstance(subject, ManifestSubject) for subject in self.subjects):
            raise TaskContractError("manifest subjects must be ManifestSubject values")
        keys = [subject.instance_key for subject in self.subjects]
        if len(set(keys)) != len(keys):
            raise TaskContractError("a DispatchManifest cannot contain duplicate instance keys")
        object.__setattr__(self, "_digest", fingerprint(self._identity_document()))

    @classmethod
    def from_confirmations(
        cls,
        freeze: PortfolioFreeze,
        validation: ValidationPlan,
        documents: list[Mapping[str, Any]],
        expectations: Mapping[str, tuple[Any, Mapping[str, Any]]],
        *,
        locator,
        track_version: int | None = None,
    ) -> "DispatchManifest":
        """Build from sidecars plus expectations rebuilt here for this exact freeze/plan/pack.

        Caller-supplied expectations are a cross-check, never a substitute for this gate's own
        dispatch/preflight/probe replay through the authoritative locator.

        @param track_version 该 run 冻结的选择版本；激活重投影与独立 expected 重建都用它，不从
            live registry 重选。``None`` 只用于没有记录版本的归档重放。
        """
        validation.validate_for(freeze)
        if locator.op_name != freeze.authority.spec.op_name:
            raise TaskContractError("activation locator names another task")
        locator.load()
        if locator.deployment_pack_digest != freeze.task_pack_digest:
            raise TaskContractError("activation task pack is not the frozen task pack")
        required = {document["instance_key"] for document in documents}
        if set(expectations) != required:
            raise TaskContractError("activation expectations must cover exactly the confirmation subjects")
        from tcenv.backends.metal.portfolio_confirmation import activation_expectation

        verified_documents = []
        for document in documents:
            instance_key = document["instance_key"]
            supplied_record, supplied_expected = expectations[instance_key]
            expected_record, expected = activation_expectation(
                locator=locator, freeze=freeze, validation=validation, document=document,
                track_version=track_version,
            )
            if not strict_json_equal(supplied_record.document(), expected_record.document()) or not strict_json_equal(
                supplied_expected, expected
            ):
                raise TaskContractError(
                    f"{instance_key}: activation expectation is not the frozen task pack's independent rebuild"
                )
            verified_documents.append(
                validate_confirmation_for_activation(
                    document,
                    freeze,
                    validation,
                    expected_record=expected_record,
                    expected=expected,
                    track_version=track_version,
                )
            )
        subjects = tuple(
            ManifestSubject.from_confirmation(document)
            for document in sorted(verified_documents, key=lambda item: item["instance_key"])
        )
        manifest = cls(
            portfolio_freeze_digest=freeze.digest,
            validation_plan_digest=validation.digest,
            status="validated",
            activation_generation=0,
            subjects=subjects,
            _derived_from_confirmations=True,
        )
        manifest.require_complete_coverage(freeze, validation)
        return manifest

    @classmethod
    def from_document(cls, value: Any) -> "DispatchManifest":
        _json_closure(value, where="dispatch manifest")
        document = _keys(value, MANIFEST_KEYS, where="dispatch manifest")
        if document["schema_id"] != MANIFEST_SCHEMA_ID:
            raise TaskContractError("unknown DispatchManifest schema")
        subjects = document["subjects"]
        if not isinstance(subjects, list):
            raise TaskContractError("manifest.subjects must be an array")
        manifest = cls(
            portfolio_freeze_digest=document["portfolio_freeze_digest"],
            validation_plan_digest=document["validation_plan_digest"],
            status=document["status"],
            activation_generation=_integer(document["activation_generation"], where="manifest.activation_generation", minimum=0),
            subjects=tuple(ManifestSubject.from_document(item) for item in subjects),
            revocation_reason=document["revocation_reason"],
        )
        if document["dispatch_manifest_digest"] != manifest.digest:
            raise TaskContractError("DispatchManifest is not its canonical projection")
        return manifest

    def _validate_subjects(self, freeze: PortfolioFreeze, validation: ValidationPlan) -> None:
        portfolio = freeze.portfolio()
        for subject in self.subjects:
            if subject.provider == "candidate" and subject.variant_id not in portfolio.variant_ids:
                raise TaskContractError(f"{subject.instance_key}: variant_id is not in the frozen portfolio")
            if subject.provider == "fallback" and subject.fallback_id != portfolio.fallback_id:
                raise TaskContractError(f"{subject.instance_key}: fallback_id is not the frozen domain fallback")
            if subject.instance_key in validation.unseen_instance_keys:
                expected_kind = "sampled_unseen"
            elif subject.instance_key in validation.tune_shards:
                expected_kind = "tune_confirmation"
            else:
                raise TaskContractError(f"{subject.instance_key}: not selected by this validation plan")
            if subject.coverage_kind != expected_kind:
                raise TaskContractError(
                    f"{subject.instance_key}: coverage_kind {subject.coverage_kind!r} is not {expected_kind!r}"
                )

    def require_complete_coverage(self, freeze: PortfolioFreeze, validation: ValidationPlan) -> None:
        self._validate_subjects(freeze, validation)
        expected = set(freeze.instance_keys())
        actual = {subject.instance_key for subject in self.subjects}
        if actual != expected:
            raise TaskContractError(
                f"DispatchManifest coverage mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )

    def activate(self, freeze: PortfolioFreeze, validation: ValidationPlan) -> "DispatchManifest":
        if self.status != "validated":
            raise TaskContractError("only a validated manifest can be activated")
        if not self._derived_from_confirmations:
            raise TaskContractError("a manifest cannot be activated from a document without confirmed sidecars")
        if self.portfolio_freeze_digest != freeze.digest or self.validation_plan_digest != validation.digest:
            raise TaskContractError("manifest belongs to another freeze or validation plan")
        self.require_complete_coverage(freeze, validation)
        return DispatchManifest(
            portfolio_freeze_digest=self.portfolio_freeze_digest,
            validation_plan_digest=self.validation_plan_digest,
            status="active",
            activation_generation=self.activation_generation + 1,
            subjects=self.subjects,
            _derived_from_confirmations=True,
        )

    def revoke(self, freeze: PortfolioFreeze, *, reason: str) -> "DispatchManifest":
        if self.status != "active":
            raise TaskContractError("only an active manifest can be revoked")
        if not self._derived_from_confirmations:
            raise TaskContractError("a document-only manifest cannot be revoked as if it came from confirmations")
        if self.portfolio_freeze_digest != freeze.digest:
            raise TaskContractError("manifest belongs to another freeze")
        _string(reason, where="manifest.revocation_reason")
        return DispatchManifest(
            portfolio_freeze_digest=self.portfolio_freeze_digest,
            validation_plan_digest=self.validation_plan_digest,
            status="revoked",
            activation_generation=self.activation_generation + 1,
            subjects=self.subjects,
            revocation_reason=reason,
            _derived_from_confirmations=True,
        )

    def _identity_document(self) -> dict[str, Any]:
        document = self.document()
        document.pop("dispatch_manifest_digest")
        return document

    def document(self) -> dict[str, Any]:
        return {
            "schema_id": MANIFEST_SCHEMA_ID,
            "portfolio_freeze_digest": self.portfolio_freeze_digest,
            "validation_plan_digest": self.validation_plan_digest,
            "status": self.status,
            "activation_generation": self.activation_generation,
            "subjects": [subject.document() for subject in self.subjects],
            "revocation_reason": self.revocation_reason,
            "dispatch_manifest_digest": self._digest if hasattr(self, "_digest") else "0" * 64,
        }

    @property
    def digest(self) -> str:
        return self._digest


def verify_manifest_document(value: Any, freeze: PortfolioFreeze, validation: ValidationPlan) -> DispatchManifest:
    """Check a canonical document binding only; this is not sidecar-backed activation evidence."""
    manifest = DispatchManifest.from_document(value)
    expected = manifest.document()
    if not strict_json_equal(value, expected):
        raise TaskContractError("DispatchManifest is not its canonical projection")
    if manifest.portfolio_freeze_digest != freeze.digest or manifest.validation_plan_digest != validation.digest:
        raise TaskContractError("DispatchManifest belongs to another freeze or validation plan")
    manifest.require_complete_coverage(freeze, validation)
    return manifest


@dataclass(frozen=True)
class ActivePortfolioPointer:
    dispatch_manifest_digest: str
    activation_generation: int

    def __post_init__(self) -> None:
        _digest(self.dispatch_manifest_digest, where="active pointer manifest digest")
        _integer(self.activation_generation, where="active pointer generation", minimum=1)

    def document(self) -> dict[str, Any]:
        return {
            "schema_id": ACTIVE_POINTER_SCHEMA_ID,
            "dispatch_manifest_digest": self.dispatch_manifest_digest,
            "activation_generation": self.activation_generation,
        }

    def write_atomic(self, path: str | Path) -> None:
        """@brief 同目录临时文件 + fsync + os.replace，不写半截 active pointer。"""
        target = Path(path)
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

    @classmethod
    def read(cls, path: str | Path, manifest: DispatchManifest, *, minimum_generation: int = 1) -> "ActivePortfolioPointer":
        try:
            value = load_strict_json(path)
        except (IdentityError, OSError) as exc:
            raise TaskContractError(f"cannot read active portfolio pointer: {exc}") from exc
        document = _keys(value, POINTER_KEYS, where="active portfolio pointer")
        if document["schema_id"] != ACTIVE_POINTER_SCHEMA_ID:
            raise TaskContractError("unknown active pointer schema")
        pointer = cls(
            dispatch_manifest_digest=document["dispatch_manifest_digest"],
            activation_generation=_integer(document["activation_generation"], where="active pointer generation", minimum=1),
        )
        if manifest.status != "active":
            raise TaskContractError("a non-active manifest cannot have an active pointer")
        if not manifest._derived_from_confirmations:
            raise TaskContractError("a document-only active manifest cannot have an active pointer")
        _integer(minimum_generation, where="active pointer minimum generation", minimum=1)
        if pointer.activation_generation < minimum_generation or manifest.activation_generation < minimum_generation:
            raise TaskContractError("the active pointer is below the required generation floor")
        if pointer.dispatch_manifest_digest != manifest.digest or pointer.activation_generation != manifest.activation_generation:
            raise TaskContractError("the active pointer is stale for this manifest")
        return pointer


__all__ = [
    "ACTIVE_POINTER_SCHEMA_ID",
    "ActivePortfolioPointer",
    "DispatchManifest",
    "MANIFEST_SCHEMA_ID",
    "ManifestSubject",
    "verify_manifest_document",
]
