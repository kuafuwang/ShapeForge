"""Canonical candidate bundle: candidate bytes + frozen geometry, deterministically ordered (A2).

A2's delivery object is a *canonical bundle*: the candidate's own source entries plus the resolved
execution geometry that will run them.  It is canonical in the strict sense — the same candidate
bytes and the same geometry document always produce the same bytes and the same digest, regardless
of the order the author wrote the entries in, the author's identity, the committee or the session.

Two identities stay separate, as the plan requires:

* ``candidate_sha256`` is the **submission** identity and considers *only* candidate source
  entries (``CandidateArtifact`` already owns it; ``msl-single`` is the file's SHA-256);
* ``bundle_digest`` is the **build-subject** digest: candidate entries *plus* the frozen geometry
  pair *plus* the registered prepare/run lifecycle when one is attached.  A3 binds this into
  ``provenance.artifact_set_digest``; A2 only has to make it deterministic and geometry-sensitive,
  so a geometry drift cannot keep the same subject digest.

Nothing here appends build products (that is the builder's job) and nothing here is a second
bundle format: the entries are ``CandidateArtifact`` entries, and the geometry is the typed
``GeometryPair`` from :mod:`tcenv.track.geometry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contract_errors import GeometryError
from ..identity import fingerprint
from .geometry import GeometryPair, GeometrySelection, select_geometry
from .lifecycle import CandidateLifecycle
from .plane import CandidateArtifact, CandidateEntry

CANONICAL_BUNDLE_SCHEMA_ID = "tcenv.track.canonical_bundle.1"


@dataclass(frozen=True)
class CanonicalBundle:
    """@brief 候选源文件 + 冻结几何的 canonical 交付对象。"""

    candidate: CandidateArtifact
    pair: GeometryPair
    lifecycle: CandidateLifecycle | None = None
    schema_id: str = CANONICAL_BUNDLE_SCHEMA_ID

    def __post_init__(self) -> None:
        """@brief 校验候选与几何属于同一次交付。

        @raises GeometryError 候选没有可用几何，或生命周期绑定了另一份候选/几何。
        """
        if self.schema_id != CANONICAL_BUNDLE_SCHEMA_ID:
            raise GeometryError(f"unsupported canonical bundle schema {self.schema_id!r}")
        if not self.pair.candidate_document:
            raise GeometryError("a canonical bundle needs the candidate geometry document")
        if self.lifecycle is not None:
            if not isinstance(self.lifecycle, CandidateLifecycle):
                raise GeometryError("canonical bundle lifecycle must be a CandidateLifecycle")
            self.lifecycle.validate_for(self.candidate)
            chosen = self.pair.candidate if self.pair.candidate is not None else self.pair.fallback
            if chosen is None:
                raise GeometryError(
                    "the lifecycle names a resolved geometry but the pair has neither a candidate nor a fallback "
                    "resolution; the build subject would not be reproducible"
                )
            if self.lifecycle.geometry.document() != chosen.document():
                raise GeometryError(
                    "the lifecycle was built for a different resolved geometry than this bundle's pair; prepare/run "
                    "behaviour and the frozen geometry must describe one delivery"
                )

    @property
    def entries(self) -> tuple[CandidateEntry, ...]:
        """@brief 按路径排序的候选源 entries（canonical 顺序）。

        @return entry 元组。
        """
        return tuple(sorted(self.candidate.entries, key=lambda entry: entry.path))

    @property
    def selection(self) -> GeometrySelection:
        """@brief 本次交付选用的几何（候选优先）。

        @return 选择结果。
        @raises GeometryError 候选几何不可用。
        """
        return select_geometry(self.pair)

    @property
    def candidate_sha256(self) -> str:
        """@brief 提交身份（仅候选源 entries）。

        @return 十六进制 sha256。
        """
        return self.candidate.candidate_sha256

    @property
    def geometry_document(self) -> dict[str, Any]:
        """@brief 冻结的几何文档（候选与独立 fallback 都在内）。

        @return 字段字典。
        """
        return self.pair.document()

    @property
    def geometry_identity(self) -> str:
        """@brief 几何执行身份。

        @return 十六进制 sha256。
        """
        return self.pair.identity

    def subject_document(self) -> dict[str, Any]:
        """@brief 构建对象文档：候选清单 + 冻结几何 + 可选的 prepare/run 生命周期。

        @return 字段字典。
        """
        document: dict[str, Any] = {
            "schema_id": self.schema_id,
            "track_id": self.candidate.track_id,
            "candidate_sha256": self.candidate_sha256,
            "candidate_manifest": {
                "schema_id": self.candidate.manifest()["schema_id"],
                "entries": [entry.document() for entry in self.entries],
            },
            "geometry": self.geometry_document,
            "lifecycle": None if self.lifecycle is None else self.lifecycle.document(),
        }
        return document

    @property
    def lifecycle_identity(self) -> str | None:
        """@brief 生命周期的身份（未登记时为 None）。

        @return 十六进制 sha256 或 None。
        """
        return None if self.lifecycle is None else self.lifecycle.identity

    @property
    def bundle_digest(self) -> str:
        """@brief 构建对象摘要（候选 entries + 冻结几何 + 生命周期）。

        @return 十六进制 sha256。
        """
        return fingerprint(
            {
                "fingerprint_kind": "canonical-candidate-bundle",
                "fingerprint_version": 2,
                "subject": self.subject_document(),
            }
        )

    def document(self) -> dict[str, Any]:
        """@brief 交付对象的 JSON 文档。

        @return 字段字典。
        """
        return {**self.subject_document(), "bundle_digest": self.bundle_digest}


__all__ = ["CANONICAL_BUNDLE_SCHEMA_ID", "CanonicalBundle"]
