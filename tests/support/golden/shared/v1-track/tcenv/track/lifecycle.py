"""Hardware-neutral ``prepare``/``run`` candidate lifecycle (A2).

The KDA candidates this project learned from expose a two-phase idiom: ``prepare`` builds the
launch-side work (compiled kernels, host tiling, resource and argument setup) and ``run``
dispatches the device kernels.  Upstream has no generic lifecycle object and no timing identity
around it, so this module is the local, **hardware-neutral** adaptation:

* the host side may only produce geometry, resources and launch arguments — it may not compute
  the reference, preprocess inputs or do the device kernel's math;
* an input-dependent host step is billed to the measured run (``inside_timing``); an
  input-independent reusable step is bound to a cache identity derived from
  ``candidate/bundle`` + ``target`` + ``instance`` + step name, never hand-written;
* the device side is exactly the active stages of the resolved geometry, in order.  A candidate
  bundle that ships Host Tiling cannot "declare" it and then never execute it: the entry role and
  the prepare step must agree;
* the whole description is versioned and hashed, so prepare/run behaviour and timing attribution
  enter the execution/build identity instead of living in prose.

There is deliberately no executor here.  The Metal adapter (A4) implements the two phases; this
module only owns the typed contract, the checks and the identity.  It imports nothing from a
backend, so it cannot smuggle a vendor-specific scheduling rule into the local plane.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..contract_errors import GeometryError
from ..identity import fingerprint
from .geometry import ResolvedGeometry
from .plane import CandidateArtifact
from .protocol import PrepareDeclaration, check_prepare_timing

LIFECYCLE_SCHEMA_VERSION = 1
#: Host-side roles a prepare step may execute.  ``harness`` covers compile/binding/launch-argument
#: assembly the harness owns; the other two are candidate-entry roles.
HOST_STEP_ROLES = frozenset({"host_tiling", "schedule", "harness"})
#: Candidate entry roles that must be executed by a prepare step instead of being ignored.
ENTRY_BOUND_ROLES = ("host_tiling", "schedule")


def cache_scope(candidate_sha256: str, target_id: str, instance_key: str) -> dict[str, Any]:
    """@brief 可复用准备的缓存作用域：候选身份 + 目标 + 实例。

    @param candidate_sha256 候选提交身份。
    @param target_id 目标档案 id。
    @param instance_key 任务实例身份（v2 为任务合同摘要）。
    @return 字段字典。
    """
    return {
        "candidate_sha256": candidate_sha256,
        "target_id": target_id,
        "instance_key": instance_key,
    }


def cache_key(scope: dict[str, Any], step_name: str) -> str:
    """@brief 一个准备步骤在给定作用域下的唯一缓存身份。

    The name is part of the key, so two reusable steps of the same candidate cannot silently
    share one cache slot.

    @param scope :func:`cache_scope` 的结果。
    @param step_name 步骤名。
    @return 十六进制 sha256。
    """
    return fingerprint(
        {
            "fingerprint_kind": "prepare-cache",
            "fingerprint_version": LIFECYCLE_SCHEMA_VERSION,
            "scope": scope,
            "step": step_name,
        }
    )


@dataclass(frozen=True)
class HostStep:
    """@brief 一个 host 侧 prepare 步骤（计时归属显式，缓存身份由生命周期计算）。"""

    declaration: PrepareDeclaration
    role: str = "harness"

    def __post_init__(self) -> None:
        """@brief 校验步骤角色与声明类型。

        @raises GeometryError 角色未登记或声明类型不符。
        """
        if self.role not in HOST_STEP_ROLES:
            raise GeometryError(
                f"host step {self.declaration.name!r}: role {self.role!r} is not registered; host roles are "
                f"{sorted(HOST_STEP_ROLES)} (a device kernel is a run stage, not a prepare step)"
            )
        if not isinstance(self.declaration, PrepareDeclaration):
            raise GeometryError(f"host step {self.role!r}: the declaration must be a PrepareDeclaration")
        if self.declaration.cache_identity is not None:
            raise GeometryError(
                f"host step {self.declaration.name!r}: cache identities are computed from "
                "candidate/target/instance; a hand-written cache_identity is not the single authoritative value"
            )

    def document(self, *, cache_identity: str | None) -> dict[str, Any]:
        """@brief 该步骤的 JSON 文档（含计算出的缓存身份）。

        @param cache_identity 生命周期计算出的缓存身份；不缓存时为 None。
        @return 字段字典。
        """
        document = {"role": self.role, **self.declaration.document()}
        document["cache_identity"] = cache_identity
        return document


@dataclass(frozen=True)
class CandidateLifecycle:
    """@brief 一条候选交付的 prepare/run 生命周期（版本化、进入身份）。"""

    track_id: str
    geometry: ResolvedGeometry
    host_steps: tuple[HostStep, ...]
    candidate_sha256: str
    target_id: str
    instance_key: str
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """@brief 校验身份字段并执行 host/device 分工与计时规则。

        @raises GeometryError 字段非法、输入相关工作藏在计时外或 host/device 分工不一致。
        """
        if self.schema_version != LIFECYCLE_SCHEMA_VERSION:
            raise GeometryError(f"unsupported candidate lifecycle schema {self.schema_version}")
        if not self.track_id or not self.target_id or not self.instance_key:
            raise GeometryError("CandidateLifecycle needs a non-empty track_id, target_id and instance_key")
        digest = str(self.candidate_sha256)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise GeometryError("CandidateLifecycle.candidate_sha256 must be a lowercase hexadecimal sha256")
        if not isinstance(self.geometry, ResolvedGeometry):
            raise GeometryError("CandidateLifecycle needs a resolved geometry")
        seen: set[str] = set()
        for step in self.host_steps:
            name = step.declaration.name
            if not name:
                raise GeometryError("a host prepare step needs a non-empty name")
            if name in seen:
                raise GeometryError(f"duplicate host prepare step {name!r}")
            seen.add(name)
        self._checked_declarations()

    def _checked_declarations(self) -> tuple[PrepareDeclaration, ...]:
        """@brief 计算缓存身份并套用 prepare 计时规则。

        @return 已填入缓存身份的声明元组。
        @raises GeometryError 规则违反。
        """
        scope = self.cache_scope
        filled: list[PrepareDeclaration] = []
        for step in self.host_steps:
            declaration = step.declaration
            identity = cache_key(scope, declaration.name) if declaration.cached else None
            filled.append(replace(declaration, cache_identity=identity))
        return check_prepare_timing(filled)

    @property
    def cache_scope(self) -> dict[str, Any]:
        """@brief 该生命周期的缓存作用域。

        @return 字段字典。
        """
        return cache_scope(self.candidate_sha256, self.target_id, self.instance_key)

    @property
    def device_stages(self) -> tuple[Any, ...]:
        """@brief ``run`` 阶段实际派发的 stage（已求值几何的激活 stage，按顺序）。

        @return ``ResolvedStage`` 元组。
        """
        return self.geometry.active_stages

    @property
    def used_fallback(self) -> bool:
        """@brief 该生命周期绑定的是独立 fallback 几何。

        @return ``geometry.source == "fallback"``。
        """
        return self.geometry.source == "fallback"

    def validate_for(self, candidate: CandidateArtifact) -> str:
        """@brief 校验生命周期与候选交付一致（Host Tiling 不能被声明后忽略）。

        @param candidate 候选字节清单。
        @return ``"accepted"``。
        @raises GeometryError track/身份不符，或 host 角色与候选 entry 不一致。
        """
        if candidate.track_id != self.track_id:
            raise GeometryError(
                f"the lifecycle is for track {self.track_id!r}, the candidate is for {candidate.track_id!r}"
            )
        if candidate.candidate_sha256 != self.candidate_sha256:
            raise GeometryError(
                "the lifecycle's candidate_sha256 does not match the candidate bytes; a prepare declaration "
                "cannot be reused across candidates"
            )
        entry_roles = {entry.role for entry in candidate.entries}
        host_roles = {step.role for step in self.host_steps}
        for role in ENTRY_BOUND_ROLES:
            if role in entry_roles and role not in host_roles:
                raise GeometryError(
                    f"the candidate ships a {role!r} entry but the lifecycle has no {role!r} prepare step; "
                    "candidate-owned host work must be executed, not declared"
                )
            if role in host_roles and role not in entry_roles:
                raise GeometryError(
                    f"the lifecycle declares a {role!r} prepare step but the candidate has no {role!r} entry"
                )
        return "accepted"

    def document(self) -> dict[str, Any]:
        """@brief 生命周期的 JSON 文档（进入构建/执行身份）。

        @return 字段字典。
        """
        declarations = self._checked_declarations()
        prepare = [
            step.document(cache_identity=declaration.cache_identity)
            for step, declaration in zip(self.host_steps, declarations)
        ]
        return {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "candidate_sha256": self.candidate_sha256,
            "target_id": self.target_id,
            "instance_key": self.instance_key,
            "cache_scope": self.cache_scope,
            "geometry": self.geometry.document(),
            "used_fallback": self.used_fallback,
            "prepare": prepare,
            "run": [
                {
                    "stage": stage.index,
                    "kernel": stage.kernel,
                    "grid": list(stage.grid),
                    "threads": list(stage.threads),
                    "threadgroup_bytes": int(stage.threadgroup_bytes),
                    "workspace": [use.name for use in stage.workspace],
                }
                for stage in self.device_stages
            ],
        }

    @property
    def identity(self) -> str:
        """@brief 生命周期身份（prepare/run 行为、缓存与计时归属的摘要）。

        @return 十六进制 sha256。
        """
        return fingerprint(
            {
                "fingerprint_kind": "candidate-lifecycle",
                "fingerprint_version": LIFECYCLE_SCHEMA_VERSION,
                "lifecycle": self.document(),
            }
        )


__all__ = [
    "ENTRY_BOUND_ROLES",
    "HOST_STEP_ROLES",
    "LIFECYCLE_SCHEMA_VERSION",
    "CandidateLifecycle",
    "HostStep",
    "cache_key",
    "cache_scope",
]
