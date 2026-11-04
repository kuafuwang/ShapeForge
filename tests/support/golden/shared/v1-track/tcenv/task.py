"""Task = everything the environment knows about one kernel to be lowered."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable

import torch

from .contract import Contract


@dataclass(frozen=True)
class BaselineImpl:
    """@brief 同一计算的一个厂商调优实现，作为计时的基线。

    A strong, vendor-tuned implementation of the same computation.

    ``fn`` receives framework arrays in the order of the contract's input arguments and
    returns one array (or a tuple, in the order of the contract's output arguments).

    ``implementation_id``/``implementation_version``/``params`` are the registered
    identity of that runner (plan §7.2 ``canonical_callable_identity``): the digest
    covers the registration and canonical params, never the closure's ``repr``.  A task
    that predates that registration leaves them empty and is listed as transitional.
    """

    name: str  # e.g. "mlx", "torch_mps"
    framework: str  # "mlx" | "torch"
    fn: Callable
    implementation_id: str = ""
    implementation_version: int = 0
    params: dict = field(default_factory=dict)


#: ``Task`` fields a migrated task derives from its ``SemanticInstance``/``Specialization``.
#: Writing any of them after construction would let the evaluation facts (launch contract,
#: case enumeration, effective tolerances and work accounting) diverge from the identity
#: ``task_fingerprint`` v2 binds, so ``__setattr__`` refuses it (N-23).
#: ``Task.meta`` keys that are *semantic* (they come from the OpSpec ``cases.inputs`` domain and
#: change what the evaluator sweeps).  A migrated task may not carry any other value for them.
SEMANTIC_META_KEYS = ("domain_case", "ieee_base", "ieee_inputs", "ieee_max")


def _frozen_meta(value):
    """@brief 把 ``Task.meta`` 递归包成只读视图（嵌套的变体块也不能就地改）。

    @param value 任意 JSON 值。
    @return 只读等价物。
    """
    if isinstance(value, dict):
        return MappingProxyType({str(key): _frozen_meta(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_meta(item) for item in value)
    return value
#: ``Task.meta`` keys that are pure diagnostics (registered, not semantic).
DIAGNOSTIC_META_KEYS = ("baseline_note", "program_lines", "triton_baseline_variant")

DERIVED_TASK_FIELDS = (
    "contract",
    "cases",
    "flops",
    "min_bytes",
    "rtol",
    "atol_rel",
    "exact",
    "semantic_instance",
    "specialization",
    "meta",
)


@dataclass
class Task:
    """@brief 一个待 lowering 内核的全部已知信息：契约、规格、参考与基线。"""

    name: str
    dtype: str
    size: str
    contract: Contract
    triton_source: str  # the *specification* shown to the model; never executed
    description: str
    cases: tuple[str, ...]
    gen_inputs: Callable[[str, int], dict[str, torch.Tensor]]
    reference: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    baselines: tuple[BaselineImpl, ...]
    flops: float
    min_bytes: float
    rtol: float
    atol_rel: float
    # Optional per-output error scale (e.g. sum |x| for cancellation-sensitive reductions).
    error_scale: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None
    # A known-good kernel (not necessarily fast) that validates the harness itself.
    # ``reference_sources`` is keyed by backend id; ``naive_msl`` remains the Metal
    # compatibility spelling until Phase 3 callers migrate.
    naive_msl: str = ""
    reference_sources: dict[str, str] = field(default_factory=dict)
    exact: bool = False  # integer tasks compare bit-exactly
    meta: dict = field(default_factory=dict)
    # R1: the derived projections that ``task_fingerprint`` v2 binds.  A task that still
    # has no OpSpec/lowering leaves them None and uses the transitional pre-R2 hash.
    semantic_instance: Any = None
    specialization: Any = None
    # Registered implementation identities for evaluation-side identities (§7.2).
    evaluation_implementations: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """@brief 同步 known-good 源，并把派生字段钉到 ``SemanticInstance`` 上。

        ``flops``/``min_bytes``/``rtol``/``atol_rel``/``exact`` 是语义实例的派生投影：
        评测（``scaled_error``）、roofline 与报告读这些字段，而 ``task_fingerprint`` v2 只读
        ``SemanticInstance``。若允许两者分叉，同一个任务身份就能对应两套评测语义（N-23）。
        构造时不一致一律拒绝，构造后再写一律拒绝；要改容差/记账只能改 OpSpec 并重新实例化。

        @raises ValueError 迁移后的 task 的派生字段与语义实例不一致。
        """
        if self.naive_msl and "metal" not in self.reference_sources:
            self.reference_sources["metal"] = self.naive_msl
        elif "metal" in self.reference_sources and not self.naive_msl:
            self.naive_msl = self.reference_sources["metal"]
        instance = self.semantic_instance
        if instance is not None:
            expected = (
                instance.work_accounting.flops,
                instance.work_accounting.min_bytes,
                instance.tolerances.rtol,
                instance.tolerances.atol_rel,
                instance.tolerances.exact,
            )
            actual = (self.flops, self.min_bytes, self.rtol, self.atol_rel, self.exact)
            if actual != expected:
                raise ValueError(
                    "a migrated task's flops/min_bytes/rtol/atol_rel/exact are derived from its "
                    f"SemanticInstance and cannot diverge from it: instance={expected!r} task={actual!r}"
                )
            specialization = self.specialization
            if specialization is None:
                raise ValueError("a task with a SemanticInstance must also carry its Specialization")
            # Same source: the Specialization was lowered from this exact instance, and the
            # Contract/case list on the Task are its projections (not a second authority).
            specialization.require_matches(instance)
            if self.contract != specialization.contract:
                raise ValueError("Task.contract must be the Contract derived into its Specialization")
            if tuple(self.cases) != tuple(instance.case_domain.cases):
                raise ValueError("Task.cases must be the SemanticInstance's declared case domain")
            # H-17: ``meta`` is a read-only *view*.  The semantic keys must equal the OpSpec's
            # evaluation domain, and nothing outside the registered diagnostic set may appear (a
            # new semantic fact belongs in the OpSpec, where it changes the identity).
            meta = dict(self.meta or {})
            unknown = sorted(set(meta) - set(SEMANTIC_META_KEYS) - set(DIAGNOSTIC_META_KEYS))
            if unknown:
                raise ValueError(
                    f"Task.meta carries unregistered keys {unknown}; semantic facts belong in the "
                    "OpSpec and diagnostics must be one of " + str(sorted(DIAGNOSTIC_META_KEYS))
                )
            projected = instance.evaluation_meta()
            declared = {key: meta[key] for key in SEMANTIC_META_KEYS if key in meta}
            if declared != projected:
                raise ValueError(
                    "Task.meta's semantic keys must be the SemanticInstance's evaluation domain: "
                    f"meta={declared!r} instance={projected!r}"
                )
            object.__setattr__(self, "meta", _frozen_meta(meta))
        # Only a migrated task has a SemanticInstance to fork from; the transitional pre-R2
        # task keeps the old mutable fields until R2 deletes that path.
        object.__setattr__(self, "_derived_fields_sealed", instance is not None)

    def __setattr__(self, name: str, value) -> None:
        """@brief 拒绝事后改写迁移后 task 的派生评测字段。

        @param name 字段名。
        @param value 新值。
        @raises AttributeError 该字段来自 ``SemanticInstance`` 且 task 已完成构造。
        """
        if name in DERIVED_TASK_FIELDS and self.__dict__.get("_derived_fields_sealed", False):
            raise AttributeError(
                f"Task.{name} is derived from the task's SemanticInstance/Specialization; edit the "
                "OpSpec and rebuild the task (e.g. dataclasses.replace with a consistent instance) "
                "instead of mutating the derived projection"
            )
        object.__setattr__(self, name, value)

    @property
    def evaluation_domain(self) -> dict:
        """@brief 评测域视图：``domain_case``/``ieee_*`` 从语义实例物化。

        The evaluation domain lives in the OpSpec ``cases.inputs``; readers use this property
        instead of reading ``Task.meta``, so the OpSpec stays the only authority (H-17).  R2
        deleted the pre-R2 ``meta`` fallback.

        @return 含 ``domain_case`` 与（若声明）``ieee_base``/``ieee_inputs``/``ieee_max`` 的字典。
        """
        instance = self.semantic_instance
        if instance is not None:
            return instance.evaluation_meta()
        # Oracle-only path: a task *without* a SemanticInstance is the frozen R0 oracle (the
        # snapshot's modules build live ``Task`` objects) or a test double.  Every production task
        # comes from ``get_task`` and therefore has an instance, and a migrated task's ``meta`` is
        # a sealed read-only projection validated against that instance — so this branch can never
        # be a second authority for a task that is actually evaluated.
        # ``tests/shared/test_task_fingerprint.py`` asserts the production domain has no such task.
        meta = dict(self.meta or {})
        view: dict = {"domain_case": bool(meta.get("domain_case"))}
        if meta.get("ieee_base") is not None:
            view["ieee_base"] = meta["ieee_base"]
        if meta.get("ieee_inputs") is not None:
            view["ieee_inputs"] = tuple(meta["ieee_inputs"])
        if meta.get("ieee_max") is not None:
            view["ieee_max"] = dict(meta["ieee_max"])
        return view

    def reference_for(self, backend_id: str) -> str:
        """@brief 取该任务在指定后端上的已知正确源码。

        The task's known-good source for one backend; empty when it has none.

        @param backend_id 后端 id，如 ``"metal"``。
        @return 参考内核源码；没有时返回空串。
        """
        if backend_id == "metal" and self.naive_msl:
            return self.naive_msl
        return self.reference_sources.get(backend_id) or ""

    @property
    def key(self) -> str:
        """@brief 任务的三段式键。

        @return ``name_dtype_size`` 形式的字符串。
        """
        return f"{self.name}_{self.dtype}_{self.size}"

    @property
    def total_buffer_bytes(self) -> int:
        """@brief 契约中所有 buffer 参数的总字节数。

        @return 总字节数。
        """
        return sum(a.nbytes for a in self.contract.buffers)


def task_fingerprint(task: Task, backend_id: str = "metal") -> str:
    """@brief 计算被测问题在指定 backend 上的身份。

    Identity of the measured problem, separate from ``tcenv.triton_runner.spec_fingerprint``.

    A task that was migrated to the two-stage derivation carries its ``SemanticInstance``
    and ``Specialization``; its fingerprint is the §3.1 canonical document of
    ``fingerprint_version: 2`` below, which binds the complete derived projections
    (shapes, resolved attrs, tolerances, work accounting, every Contract field including
    ``gpu``), ``opspec_digest``, the triton source, ``description`` and the backend's
    reference kernel.  Changing only the description, only the profile's threads or only
    a rule version therefore invalidates it (F-2/F-24/F-25/N-7).

    R2 migrated every task in the accept domain and deleted the transitional pre-R2 hash,
    so a task without projections is refused instead of hashed twice (see
    ``tests/shared/test_task_fingerprint.py``).

    ``spec_fingerprint`` is the triton-msl baseline cache key. This one also covers the backend's reference kernel,
    the tolerances, the flop and byte counts, and whether a full-domain sweep is declared. The reference function's
    code is not hashed; ``run_config.json`` records ``git_commit`` and ``git_dirty`` for that.

    @param task 要取指纹的任务。
    @param backend_id 目标 backend；缺省 ``metal``。
    @return 十六进制 sha256 指纹。
    """
    from .identity import fingerprint

    backend_id = str(backend_id or "metal")
    instance = getattr(task, "semantic_instance", None)
    specialization = getattr(task, "specialization", None)
    if instance is not None and specialization is not None:
        document = {
            "fingerprint_kind": "task",
            "fingerprint_version": 2,
            "task": task.name,
            "dtype": task.dtype,
            "size": task.size,
            "backend_id": backend_id,
            "semantic_instance": instance.fingerprint_document(),
            "specialization": specialization.fingerprint_document(),
            "triton_source": task.triton_source,
            "description": task.description,
            "reference_source": task.reference_for(backend_id),
        }
        return fingerprint(document)

    # R2 deleted the pre-v2 fallback: every task in the accept domain derives from a
    # ``SemanticInstance``, so a task without one is a construction bug, not a hash variant.
    raise ValueError(
        f"task {task.name!r}: task_fingerprint v2 needs a semantic instance and specialization "
        "(the transitional pre-R2 hash path is gone)"
    )


def default_tolerances(dtype: str) -> tuple[float, float]:
    """@brief 按元素类型给出默认容差。

    (rtol, atol_rel). Tolerance is rtol*|ref| + atol_rel*scale per element.

    @param dtype 元素类型短名。
    @return ``(rtol, atol_rel)`` 二元组；整数类型为 ``(0.0, 0.0)``。
    """
    if dtype == "fp16":
        return 2.0e-3, 1.0e-3
    if dtype == "bf16":
        return 1.6e-2, 8.0e-3
    if dtype == "fp32":
        return 1.0e-5, 1.0e-5
    return 0.0, 0.0
