"""``reference.py``: the independent, read-only correctness oracle of a v2 task.

The standard v2 entry fixes one interface::

    def reference(inputs, attrs):
        return {"y": ...}

    # optional:
    def gen_inputs(case, seed, instance): ...
    def error_scale(inputs, reference_outputs, instance, tolerances): ...

``read_write`` tensors have two sides: their *initial* value is part of the reference input
contract (``instance.reference_inputs``, including an ``output``/``read_write`` buffer) and
their *final* value is part of the result contract (``instance.reference_outputs``).  The
reference may not mutate the inputs it was handed.

The boundary is deliberately narrow, and it is a *static* check, not a Python sandbox:

* a reference module may import the standard library and math libraries (``torch``,
  ``numpy``, ...), but not this repository (``tasks``/``tcenv``/``taic``/...) and not any
  candidate/renderer/known-good module, so the oracle cannot be defined by the thing it
  is supposed to judge;
* file/process/eval primitives are refused at the AST level;
* the returned source text is read-only; candidates never import it (that rule is
  enforced on the candidate side by the track/ABI plane, not by trusting this text).

A packed-varlen (L1) instance adds one runtime invariant the task does not own: the
``cu_seqlens`` tensor a case generates must start at 0, be non-decreasing, end at
``total_tokens``, have ``num_seqs + 1`` ``i32`` offsets and carry at least one token.  Both
:meth:`ReferenceModule.generate` and :meth:`ReferenceModule.execute` route it through
:func:`tcenv.taskcontract.varlen.require_runtime_segments`, so the boundary check is
harness-owned and sits on the same path as the oracle.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..identity import IdentityError, load_strict_json
from ..opspec.v2 import SemanticInstanceV2
from ..opspec.validator import SchemaFailure, validate
from .errors import AssemblyError, ReferenceBoundaryError, SemanticUnresolved, TaskContractError
from .sampling import SamplingPlan, generate_inputs, sampling_plan_from_document
from .runtime_shape import resolved_shape_for_case
from .varlen import require_runtime_segments

REFERENCE_FILE = "reference.py"
REFERENCE_PROFILE_FILE = "reference.json"
REFERENCE_PROFILE_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "reference.v1.schema.json"

#: Repository roots a reference may not import (the oracle must not depend on the harness).
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {"backends", "deploy", "experiments", "kernels", "observe", "proto", "scripts", "taic", "targets", "tasks", "tcenv", "tests"}
)
#: Module segments that identify a candidate/renderer/known-good implementation.
FORBIDDEN_IMPORT_SEGMENTS = frozenset({"candidate", "known_good", "kernel_triton", "lowering", "renderer", "track"})
#: Builtins that turn a reference into a file/process/eval access path.
FORBIDDEN_CALL_NAMES = frozenset({"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"})
#: Attribute calls with the same effect (``os.system``, ``Path.read_text``, ...).
FORBIDDEN_ATTRIBUTES = frozenset(
    {"check_output", "import_module", "open", "popen", "read_bytes", "read_text", "reload", "run", "system"}
)


@dataclass(frozen=True)
class ReferenceModule:
    """@brief 已加载、已做边界检查的 reference oracle。"""

    op_name: str
    path: Path
    source: str
    source_sha256: str
    reference_version: int
    sampling: SamplingPlan | None
    error_scale_kind: str
    _reference: Callable[..., Any] = field(repr=False)
    _gen_inputs: Callable[..., Any] | None = field(default=None, repr=False)
    #: ``gen_inputs`` 是否接受 L2 的 ``resolved=`` 关键字（arity 4）。L2 任务必须接受；固定/L1 任务必须
    #: 保持 3 个位置参数，这样它们的 pack 与调用字节不变。
    gen_inputs_resolved: bool = False
    _error_scale: Callable[..., Any] | None = field(default=None, repr=False)

    @property
    def implementation_id(self) -> str:
        """@brief 已登记的实现 id。

        @return ``<op>.reference``。
        """
        return f"{self.op_name}.reference"

    def document(self) -> dict[str, Any]:
        """@brief reference 身份文档（进入任务身份，不进入 opspec_digest）。

        @return 字段字典。
        """
        return {
            "implementation_id": self.implementation_id,
            "implementation_version": int(self.reference_version),
            "source_sha256": self.source_sha256,
            "sampling": self.sampling.document() if self.sampling is not None else None,
            "gen_inputs": self._gen_inputs is not None,
            "gen_inputs_resolved": bool(self.gen_inputs_resolved),
            "error_scale": self.error_scale_kind,
        }

    def display_source(self) -> str:
        """@brief 只读展示文本（题面中展示的 reference 源）。

        @return 源文本。
        """
        return self.source

    def generate(self, instance: SemanticInstanceV2, case: str, seed: int) -> dict[str, Any]:
        """@brief 生成一个 case/seed 的 reference 初值；优先使用 ``gen_inputs``。

        An L2 (capacity) instance hands the generator the *expected* resolved shape as a keyword
        argument, because the generator must fill the padding and the capacity-sized ``cu_seqlens`` tail
        itself.  The expected value comes from the declared workload profile of this case
        (:func:`tcenv.taskcontract.runtime_shape.resolved_shape_for_case`), not from the generator, so the
        task cannot certify its own batch: a produced input whose prefix/tail disagrees with the declared
        profile is refused below.  Fixed-shape and L1 generators keep the historical three positional
        arguments, so their packs and call bytes do not change.

        @param instance 语义实例。
        @param case 用例名。
        @param seed 种子。
        @return 初值张量名 → ``torch.Tensor``（含 output/read_write 初值）。
        @raises TaskContractError ``gen_inputs`` 返回的键/形状不符合声明。
        @raises VarlenContractError 实际 ``cu_seqlens`` 不满足 L1/L2 分段合同。
        """
        if case not in instance.case_domain.cases:
            raise TaskContractError(f"{self.op_name}: case {case!r} is not declared")
        resolved = resolved_shape_for_case(instance, case)
        if self._gen_inputs is None:
            if self.sampling is None:  # pragma: no cover - loader guarantees one source
                raise TaskContractError(f"{self.path}: no input source is registered")
            if resolved is not None and resolved.kind == "l2":
                raise TaskContractError(
                    f"{self.path}: an L2 capacity instance needs gen_inputs(case, seed, instance, *, resolved); "
                    "sampling cannot fill a capacity-sized cu_seqlens tail with the declared poison sentinel"
                )
            produced = generate_inputs(instance, self.sampling, case, seed)
            where = f"{self.path}: sampling({case!r}, {seed})"
        elif resolved is not None and resolved.kind == "l2":
            if not self.gen_inputs_resolved:
                raise TaskContractError(
                    f"{self.path}: an L2 capacity instance needs gen_inputs(case, seed, instance, resolved=None) "
                    "so the generator can size the capacity buffers and fill the padding; the registered "
                    "gen_inputs takes three positional parameters"
                )
            produced = self._gen_inputs(case, int(seed), instance, resolved=resolved)
            where = f"{self.path}: gen_inputs({case!r}, {seed})"
        else:
            produced = self._gen_inputs(case, int(seed), instance)
            where = f"{self.path}: gen_inputs({case!r}, {seed})"
        # The task owns *which* segmentation a case generates; the harness owns whether that
        # segmentation is legal (first offset 0, non-decreasing, last total_tokens, i32, at least
        # one token).  The cu check therefore runs *before* the generic mapping check: a wrong
        # length/rank/dtype boundary is a varlen contract violation, not a generic
        # ``semantic_unresolved``, and a task generator cannot certify its own boundaries.  The
        # closed key/shape check for every other tensor is unchanged and still runs next.
        if isinstance(produced, Mapping):
            require_runtime_segments(instance, produced, where=where, resolved=resolved, case=case)
        checked = _validate_tensor_mapping(
            produced,
            expected={name: instance.tensor(name) for name in instance.reference_inputs},
            where=where,
            require_torch=True,
        )
        return checked

    def execute(
        self,
        instance: SemanticInstanceV2,
        inputs: Mapping[str, Any],
        attrs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """@brief 执行 reference，并强制 read_write 终值、输入不可变和结果键闭合。

        @param instance 语义实例。
        @param inputs 输入张量。
        @param attrs 已物化 attrs（缺省用实例的 ``resolved_attrs``）。
        @return 张量名 → reference 结果。
        @raises SemanticUnresolved 结果键不闭合（含 read_write 终值缺失）。
        @raises VarlenContractError 实际 ``cu_seqlens`` 不满足 L1 分段合同。
        @raises ReferenceBoundaryError reference 改写了输入。
        """
        # The oracle segments by the *actual* tensor it is handed, so the cu contract runs first:
        # a boundary that is too short, two-dimensional or not i32 must be a typed
        # ``varlen_contract_invalid`` refusal, not the generic closed-key/shape error that the
        # mapping check would report.  Every other tensor keeps that closed mapping check below.
        if isinstance(inputs, Mapping):
            require_runtime_segments(instance, inputs, where=f"{self.path}: inputs")
        expected_inputs = {name: instance.tensor(name) for name in instance.reference_inputs}
        checked = _validate_tensor_mapping(inputs, expected=expected_inputs, where=f"{self.path}: inputs", require_torch=True)
        # The caller's tensors are never handed to the oracle: a reference that mutates in place
        # would otherwise corrupt the evaluation inputs even though we raise afterwards.  The
        # working copies preserve values/dtypes/shapes and let us still detect the violation.
        caller_versions = {name: getattr(tensor, "_version", None) for name, tensor in checked.items()}
        caller_content = {name: _content_digest(tensor) for name, tensor in checked.items()}
        working = {name: tensor.clone() for name, tensor in checked.items()}
        working_versions = {name: getattr(tensor, "_version", None) for name, tensor in working.items()}
        working_content = {name: _content_digest(tensor) for name, tensor in working.items()}
        resolved_attrs = MappingProxyType(dict(instance.resolved_attrs if attrs is None else attrs))
        try:
            produced = self._reference(MappingProxyType(working), resolved_attrs)
        except TypeError as exc:
            raise ReferenceBoundaryError(
                f"{self.path}: reference tried to write through the read-only inputs/attrs mapping: {exc}"
            ) from exc
        except Exception as exc:
            raise TaskContractError(f"{self.path}: reference raised {type(exc).__name__}: {exc}") from exc
        mutated_caller = sorted(
            name for name, tensor in checked.items() if getattr(tensor, "_version", None) != caller_versions[name]
        )
        mutated_working = sorted(
            name for name, tensor in working.items() if getattr(tensor, "_version", None) != working_versions[name]
        )
        mutated_content = sorted(
            name for name in checked
            if _content_digest(checked[name]) != caller_content[name]
            or _content_digest(working[name]) != working_content[name]
        )
        if mutated_caller or mutated_working or mutated_content:
            changed = sorted(set(mutated_caller) | set(mutated_working) | set(mutated_content))
            raise ReferenceBoundaryError(
                f"{self.path}: reference mutated its input tensors {changed}; "
                "the oracle receives pristine working copies and must return new tensors (read_write final values included)"
            )
        expected_outputs = {
            name: instance.tensor(name) for name in instance.reference_outputs
        }
        return _validate_tensor_mapping(
            produced,
            expected=expected_outputs,
            where=f"{self.path}: reference outputs (write outputs + read_write final values)",
            require_torch=True,
        )

    def error_scale(
        self,
        instance: SemanticInstanceV2,
        inputs: Mapping[str, Any],
        reference_outputs: Mapping[str, Any],
        *,
        logical: Mapping[str, tuple[slice, ...]] | None = None,
    ) -> dict[str, Any] | None:
        """@brief 可选 error-scale 钩子（只影响容差尺度，不决定 pass/fail 公式）。

        For an L2 capacity instance the hook is handed the **logical region only** (``logical`` names the
        per-tensor slices).  That is what keeps the scale honest: the unused capacity rows/columns hold
        the declared poison, and a scale computed over them would be inflated by a poison value the
        honest candidate never wrote, hiding a wrong logical result (``output`` all right, ``final_state``
        all wrong) behind a huge denominator.  Slicing *before* the hook also means a task cannot
        aggregate across the padding by accident.

        The returned scale must broadcast to the logical shape of its result tensor; every key must be a
        reference output.  A hook is never allowed to enlarge the comparison region, only to scale it.

        @param instance 语义实例。
        @param inputs 输入张量。
        @param reference_outputs reference 输出。
        @param logical 张量名 → logical 区域切片（L2；固定/L1 省略，等于整张量）。
        @return 张量名 → scale；没有钩子时为 ``None``。
        @raises TaskContractError 返回键不在差分合同内或尺度无法广播到 logical 形状。
        """
        if self._error_scale is None:
            return None
        expected_inputs = {name: instance.tensor(name) for name in instance.reference_inputs}
        expected_outputs = {name: instance.tensor(name) for name in instance.reference_outputs}
        region = dict(logical or {})
        unknown_region = sorted(set(region) - set(expected_inputs) - set(expected_outputs))
        if unknown_region:
            raise TaskContractError(
                f"{self.path}: the error_scale logical region names undeclared tensors {unknown_region}"
            )

        def _project(binding: Any, value: Any) -> Any:
            slices = region.get(binding.name)
            if slices is None:
                return value
            return value[tuple(slices)]

        def _projected_shape(binding: Any) -> tuple[int, ...]:
            slices = region.get(binding.name)
            if slices is None:
                return tuple(int(dim) for dim in binding.shape)
            if len(slices) != len(binding.shape):
                raise TaskContractError(
                    f"{self.path}: the error_scale logical region for {binding.name!r} has {len(slices)} axes, but "
                    f"the tensor has {len(binding.shape)}"
                )
            return tuple(
                len(range(*part.indices(int(dim)))) for part, dim in zip(slices, binding.shape)
            )

        # An undeclared name is *not* projected: it must still reach ``_validate_tensor_mapping`` so the
        # key closure keeps its typed refusal (``SemanticUnresolved``) instead of turning into a bare
        # ``KeyError`` inside this projection (which would be misreported as infrastructure).
        projected_inputs = {
            name: (value if name not in expected_inputs else _project(expected_inputs[name], value))
            for name, value in inputs.items()
        }
        projected_outputs = {
            name: (value if name not in expected_outputs else _project(expected_outputs[name], value))
            for name, value in reference_outputs.items()
        }
        checked_inputs = _validate_tensor_mapping(
            projected_inputs,
            expected=expected_inputs,
            where=f"{self.path}: error_scale inputs",
            require_torch=True,
            shapes={name: _projected_shape(binding) for name, binding in expected_inputs.items()},
        )
        checked_outputs = _validate_tensor_mapping(
            projected_outputs,
            expected=expected_outputs,
            where=f"{self.path}: error_scale reference outputs",
            require_torch=True,
            shapes={name: _projected_shape(binding) for name, binding in expected_outputs.items()},
        )
        # ``error_scale`` is a reference hook, not an implicit mutation channel.  Give it the
        # same pristine working copies as ``execute`` and verify both the caller's tensors and
        # the copies afterwards; a scale hook that writes in place must fail without leaking.
        caller_versions = {
            **{f"input:{name}": getattr(tensor, "_version", None) for name, tensor in checked_inputs.items()},
            **{f"output:{name}": getattr(tensor, "_version", None) for name, tensor in checked_outputs.items()},
        }
        caller_content = {
            **{f"input:{name}": _content_digest(tensor) for name, tensor in checked_inputs.items()},
            **{f"output:{name}": _content_digest(tensor) for name, tensor in checked_outputs.items()},
        }
        working_inputs = {name: tensor.clone() for name, tensor in checked_inputs.items()}
        working_outputs = {name: tensor.clone() for name, tensor in checked_outputs.items()}
        working_versions = {
            **{f"input:{name}": getattr(tensor, "_version", None) for name, tensor in working_inputs.items()},
            **{f"output:{name}": getattr(tensor, "_version", None) for name, tensor in working_outputs.items()},
        }
        working_content = {
            **{f"input:{name}": _content_digest(tensor) for name, tensor in working_inputs.items()},
            **{f"output:{name}": _content_digest(tensor) for name, tensor in working_outputs.items()},
        }
        try:
            produced = self._error_scale(
                MappingProxyType(working_inputs),
                MappingProxyType(working_outputs),
                instance,
                instance.tolerances,
            )
        except TypeError as exc:
            raise ReferenceBoundaryError(
                f"{self.path}: error_scale tried to write through the read-only inputs/outputs mapping: {exc}"
            ) from exc
        caller_after = {
            **{f"input:{name}": getattr(tensor, "_version", None) for name, tensor in checked_inputs.items()},
            **{f"output:{name}": getattr(tensor, "_version", None) for name, tensor in checked_outputs.items()},
        }
        working_after = {
            **{f"input:{name}": getattr(tensor, "_version", None) for name, tensor in working_inputs.items()},
            **{f"output:{name}": getattr(tensor, "_version", None) for name, tensor in working_outputs.items()},
        }
        mutated = sorted(name for name in caller_versions if caller_after[name] != caller_versions[name] or working_after[name] != working_versions[name])
        mutated_content = sorted(
            name for name in caller_content
            if (
                _content_digest((checked_inputs if name.startswith("input:") else checked_outputs)[name.split(":", 1)[1]])
                != caller_content[name]
            ) or (
                _content_digest((working_inputs if name.startswith("input:") else working_outputs)[name.split(":", 1)[1]])
                != working_content[name]
            )
        )
        if mutated or mutated_content:
            changed = sorted(set(mutated) | set(mutated_content))
            raise ReferenceBoundaryError(
                f"{self.path}: error_scale mutated its inputs/reference outputs {changed}; the hook receives "
                "pristine working copies and must return scale values without writing tensors"
            )
        if not isinstance(produced, Mapping):
            raise TaskContractError(f"{self.path}: error_scale must return a mapping")
        unknown = sorted(set(produced) - set(instance.reference_outputs))
        if unknown:
            raise TaskContractError(
                f"{self.path}: error_scale names {unknown}, which are not in the reference diff contract "
                f"{list(instance.reference_outputs)}"
            )
        for name, value in produced.items():
            shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
            target = _projected_shape(expected_outputs[name])
            if not broadcasts_scale(shape, target):
                raise TaskContractError(
                    f"{self.path}: error_scale[{name!r}] has shape {shape}, which does not broadcast to the "
                    f"{'logical' if name in region else 'declared'} result shape {target}; a scale that does not "
                    "match the compared region cannot be honoured without guessing"
                )
        return dict(produced)


def _content_digest(tensor: Any) -> bytes:
    """@brief 返回张量内容的 SHA-256，保留 NaN 位型并捕获别名视图原地写。

    ``Tensor._version`` does not increase for ``.data`` or NumPy-view writes, so the
    boundary also compares exact content before and after every oracle call.
    """
    raw = tensor.detach().to(device="cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).digest()


def broadcasts_scale(shape: tuple[int, ...], target: tuple[int, ...]) -> bool:
    """@brief 一个 scale 形状能否按 numpy 规则广播到目标形状（标量总是可以）。

    @param shape 尺度形状。
    @param target 目标（logical 结果）形状。
    @return 布尔。
    """
    if len(shape) > len(target):
        return False
    for have, want in zip(reversed(shape), reversed(target)):
        if have != want and have != 1:
            return False
    return True


def _validate_tensor_mapping(
    produced: Any,
    *,
    expected: Mapping[str, Any],
    where: str,
    require_torch: bool,
    shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """@brief 校验一个张量映射的键闭合与形状。

    @param produced 被检查的映射。
    @param expected 期望的张量绑定。
    @param where 诊断位置。
    @param require_torch 是否要求值是 torch 张量。
    @param shapes 可选的期望形状覆盖（L2 的 logical 区域）。
    @return 复制出的普通字典。
    @raises SemanticUnresolved 键不闭合或形状不符。
    @raises TaskContractError 不是映射。
    """
    if not isinstance(produced, Mapping):
        raise TaskContractError(f"{where}: expected a mapping of tensor name -> tensor, got {type(produced).__name__}")
    names = set(expected)
    missing = sorted(names - set(produced))
    extra = sorted(set(produced) - names)
    if missing or extra:
        raise SemanticUnresolved(
            f"{where}: tensor keys do not match the OpSpec declaration; missing={missing} extra={extra}; "
            "the OpSpec and reference must describe the same tensor set (read_write initial and final values included)"
        )
    checked: dict[str, Any] = {}
    for name, value in produced.items():
        binding = expected[name]
        if require_torch:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - torch is a hard dependency of evaluation
                raise TaskContractError(f"{where}: torch is required to validate tensor {name!r}") from exc
            if not isinstance(value, torch.Tensor):
                raise TaskContractError(f"{where}: {name!r} must be a torch.Tensor, got {type(value).__name__}")
        shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
        want = tuple(int(dim) for dim in (shapes[name] if shapes is not None and name in shapes else binding.shape))
        if shape != want:
            raise SemanticUnresolved(
                f"{where}: {name!r} has shape {shape}, but the OpSpec declares {want}"
            )
        checked[name] = value
    return checked


def _scan_reference_source(source: str, where: str) -> None:
    """@brief AST 边界检查：不允许仓库/候选 import 与文件/进程/eval 原语。

    @param source reference 源文本。
    @param where 诊断位置。
    @raises ReferenceBoundaryError 命中禁止项。
    """
    try:
        tree = ast.parse(source, filename=where)
    except SyntaxError as exc:
        raise AssemblyError(f"{where}: reference source does not parse: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_import(alias.name, where, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ReferenceBoundaryError(f"{where}:{node.lineno}: relative imports are not allowed in a reference module")
            if node.module:
                _check_import(node.module, where, node.lineno)
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in FORBIDDEN_CALL_NAMES:
                raise ReferenceBoundaryError(
                    f"{where}:{node.lineno}: reference may not call {name}() (file/process/eval boundary)"
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRIBUTES:
                raise ReferenceBoundaryError(
                    f"{where}:{node.lineno}: reference may not call .{node.func.attr}() "
                    "(file/process/eval boundary)"
                )


def _check_import(module: str, where: str, lineno: int) -> None:
    """@brief 检查一个 import 的根与可疑段。

    @param module 模块名。
    @param where 诊断位置。
    @param lineno 行号。
    @raises ReferenceBoundaryError 命中仓库根或候选/renderer/known-good 段。
    """
    segments = module.split(".")
    if segments[0] in FORBIDDEN_IMPORT_ROOTS:
        raise ReferenceBoundaryError(
            f"{where}:{lineno}: reference may not import {module!r}; the oracle must not depend on the harness "
            "or on a candidate implementation"
        )
    bad = sorted(set(segments) & FORBIDDEN_IMPORT_SEGMENTS)
    if bad:
        raise ReferenceBoundaryError(
            f"{where}:{lineno}: reference may not import {module!r} (candidate/renderer/known-good segment {bad})"
        )


def _require_gen_inputs(module: Any, where: str) -> tuple[Callable[..., Any], bool]:
    """@brief 校验 ``gen_inputs`` 的签名，并返回它是否接受 L2 的 ``resolved=``。

    Two forms are registered:

    * ``gen_inputs(case, seed, instance)`` — the historical three-argument generator used by fixed-shape
      and L1 tasks.  Their call bytes and packs do not change.
    * ``gen_inputs(case, seed, instance, resolved=None)`` — the L2 capacity form.  The fourth parameter
      must be keyword-passable (the harness passes ``resolved=``) and may carry a default so a task can
      document the optional reading.

    Which form is *required* depends on the instance and is enforced in
    :meth:`ReferenceModule.generate`: an L2 instance needs the resolved argument, a fixed/L1 instance must
    not receive it.

    @param module 已加载模块。
    @param where 诊断位置。
    @return ``(function, accepts_resolved)``。
    @raises AssemblyError 签名不符。
    """
    function = getattr(module, "gen_inputs", None)
    if not callable(function):  # pragma: no cover - callers check first
        raise AssemblyError(f"{where}: reference.py must define gen_inputs()")
    signature = inspect.signature(function)
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise AssemblyError(f"{where}: gen_inputs() must not use *args/**kwargs, got {signature}")
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(parameter.kind is inspect.Parameter.POSITIONAL_ONLY for parameter in positional[3:]):
        raise AssemblyError(f"{where}: gen_inputs() must accept 'resolved' as a keyword parameter, got {signature}")
    required = [parameter for parameter in positional if parameter.default is inspect.Parameter.empty]
    if len(positional) == 3 and len(required) == 3:
        return function, False
    if len(positional) == 4 and len(required) == 3:
        return function, True
    raise AssemblyError(
        f"{where}: gen_inputs() must be gen_inputs(case, seed, instance) or "
        f"gen_inputs(case, seed, instance, resolved=<default>), got {signature}"
    )


def _require_callable(module: Any, name: str, arity: int, where: str) -> Callable[..., Any]:
    """@brief 校验模块里一个可调用对象的固定位置参数个数。

    @param module 已加载模块。
    @param name 函数名。
    @param arity 位置参数个数。
    @param where 诊断位置。
    @return 该函数。
    @raises AssemblyError 缺失或签名不符。
    """
    function = getattr(module, name, None)
    if not callable(function):
        raise AssemblyError(f"{where}: reference.py must define {name}()")
    signature = inspect.signature(function)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [parameter for parameter in positional if parameter.default is inspect.Parameter.empty]
    if len(positional) != arity or len(required) != arity:
        raise AssemblyError(
            f"{where}: {name}() must take exactly {arity} positional parameters with no defaults, "
            f"got {signature}"
        )
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            raise AssemblyError(f"{where}: {name}() must not use *args/**kwargs, got {signature}")
    return function


def _load_module(path: Path, digest: str) -> Any:
    """@brief 从文件加载一个私有模块（不加入任务目录到 sys.path）。

    @param path reference.py 路径。
    @param digest 源摘要（用于唯一模块名）。
    @return 已执行模块。
    @raises AssemblyError 加载失败。
    """
    module_name = f"_taic_reference_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - file exists
        raise AssemblyError(f"{path}: cannot be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise AssemblyError(f"{path}: reference module failed to import: {exc}") from exc
    return module


def _load_profile(path: Path) -> dict[str, Any]:
    """@brief 读取并校验可选 ``reference.json``。

    @param path reference.json 路径。
    @return 文档。
    @raises TaskContractError 文档非法。
    """
    try:
        document = load_strict_json(path)
        schema = load_strict_json(REFERENCE_PROFILE_SCHEMA_PATH)
    except IdentityError as exc:
        raise TaskContractError(f"{path}: {exc}") from exc
    try:
        validate(document, schema)
    except SchemaFailure as exc:
        raise TaskContractError(f"{path}: {exc}") from exc
    return document


def load_reference(
    task_dir: str | Path,
    *,
    op_name: str,
    instance: SemanticInstanceV2,
) -> ReferenceModule:
    """@brief 加载一个 v2 任务的 reference oracle 与可选 profile。

    @param task_dir 任务目录。
    @param op_name 算子名。
    @param instance 语义实例（用于校验采样/输出名）。
    @return 已加载的 :class:`ReferenceModule`。
    @raises AssemblyError 缺 reference.py 或接口不符。
    @raises ReferenceBoundaryError 越界 import/调用。
    @raises SemanticUnresolved profile 与模块声明冲突。
    """
    root = Path(task_dir)
    path = root / REFERENCE_FILE
    if not path.is_file():
        raise AssemblyError(
            f"{root}: a v2 task requires {REFERENCE_FILE}; there is no fallback to a v1 track/task module"
        )
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssemblyError(f"{path}: reference source must be UTF-8: {exc}") from exc
    _scan_reference_source(source, str(path))
    module = _load_module(path, digest)

    profile_path = root / REFERENCE_PROFILE_FILE
    profile = _load_profile(profile_path) if profile_path.exists() else {}
    reference = _require_callable(module, "reference", 2, str(path))
    gen_inputs, gen_inputs_resolved = (
        _require_gen_inputs(module, str(path)) if callable(getattr(module, "gen_inputs", None)) else (None, False)
    )
    error_scale = (
        _require_callable(module, "error_scale", 4, str(path)) if callable(getattr(module, "error_scale", None)) else None
    )

    declared_scale = (profile.get("error_scale") or {}).get("kind")
    actual_scale = "module" if error_scale is not None else "none"
    if declared_scale is not None and declared_scale != actual_scale:
        raise SemanticUnresolved(
            f"{profile_path}: declares error_scale kind {declared_scale!r}, but reference.py "
            f"{'defines' if error_scale is not None else 'does not define'} error_scale(); the profile and the "
            "oracle must agree"
        )
    declared_sampling = profile.get("sampling")
    if gen_inputs is not None:
        if declared_sampling is not None:
            raise SemanticUnresolved(
                f"{profile_path}: declares a sampling profile while reference.py also defines gen_inputs(); "
                "a task has one input source, not two"
            )
        sampling = None
    else:
        sampling = sampling_plan_from_document(
            declared_sampling,
            instance=instance,
            where=f"{profile_path}.sampling" if profile else "reference.json.sampling",
        )
    reference_version = int(profile.get("reference_version", 1))
    return ReferenceModule(
        op_name=op_name,
        path=path,
        source=source,
        source_sha256=digest,
        reference_version=reference_version,
        sampling=sampling,
        error_scale_kind=actual_scale,
        _reference=reference,
        _gen_inputs=gen_inputs,
        gen_inputs_resolved=gen_inputs_resolved,
        _error_scale=error_scale,
    )


class ReferenceRegistry:
    """@brief op_name → 唯一 reference oracle 的只读注册表。"""

    def __init__(self) -> None:
        """@brief 创建空注册表。"""
        self._entries: dict[str, ReferenceModule] = {}

    def register(self, package: ReferenceModule) -> ReferenceModule:
        """@brief 登记一个 oracle；同一算子只允许一个身份。

        @param package 已加载的 oracle。
        @return 该 oracle。
        @raises AssemblyError 重名且摘要不同。
        """
        existing = self._entries.get(package.op_name)
        if existing is not None and existing.document() != package.document():
            raise AssemblyError(
                f"{package.op_name}: a second reference with a different identity is registered "
                f"(source/version/sampling/error_scale differ); one operator has one authoritative oracle"
            )
        self._entries[package.op_name] = package
        return package

    def get(self, op_name: str) -> ReferenceModule:
        """@brief 取一个已登记的 oracle。

        @param op_name 算子名。
        @return oracle。
        @raises KeyError 未登记。
        """
        return self._entries[op_name]

    def __contains__(self, op_name: object) -> bool:
        """@brief 是否已登记。

        @param op_name 算子名。
        @return 已登记时为 True。
        """
        return op_name in self._entries

    def names(self) -> tuple[str, ...]:
        """@brief 已登记的算子名（排序）。

        @return 名字元组。
        """
        return tuple(sorted(self._entries))


__all__ = [
    "FORBIDDEN_ATTRIBUTES",
    "FORBIDDEN_CALL_NAMES",
    "FORBIDDEN_IMPORT_ROOTS",
    "REFERENCE_FILE",
    "REFERENCE_PROFILE_FILE",
    "REFERENCE_PROFILE_SCHEMA_PATH",
    "ReferenceModule",
    "ReferenceRegistry",
    "load_reference",
]
