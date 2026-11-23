"""Current v2 task fixtures for mechanism tests.

This module deliberately returns production ``TaskContract`` and ``TaskBinding`` objects.  It
does not reconstruct a v1 ``Task``, archived launch geometry, or a historical v1 fingerprint.
Execution geometry and the candidate kernel name belong to each test candidate, not to the task.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from taic.task_binding import TaskBinding, bind_task
from tcenv.evaluation import EvaluationResult
from tcenv.taskcontract import TaskContract, load_task_contract
from taic.v2_evaluation import V2LocalEvaluator

ROOT = Path(__file__).resolve().parents[2]
V2_TASK = "sem_attention_odd_scale"
V2_TASKS: tuple[str, ...] = ("sem_attention_odd_scale", "sem_int_5", "sem_int_3")
V2_DYNAMIC: tuple[str, ...] = ("sem_int_5", "sem_int_3", "sem_int_7")


@functools.lru_cache(maxsize=None)
def v2_contract(
    name: str = V2_TASK,
    dtype: str = "fp16",
    size: str = "paper",
) -> TaskContract:
    """Load one shipped v2 contract without any v1 loader or archived identity projection."""
    directory = ROOT / "tasks" / name
    if not (directory / "opspec.json").is_file():
        raise KeyError(f"{name!r} has no shipped v2 task directory")
    return load_task_contract(directory, dtype=dtype, size=size)


def bind_v2(
    name: str = V2_TASK,
    dtype: str = "fp16",
    size: str = "paper",
    **kwargs,
) -> TaskBinding:
    """Bind through the production directory-version dispatcher, with no specimen override."""
    binding = bind_task(name, dtype, size, **kwargs)
    if not binding.is_v2 or binding.contract is None:
        raise AssertionError(f"{name}/{dtype}/{size}: expected a v2 binding, got {binding.mode}")
    return binding


@dataclasses.dataclass(frozen=True)
class V2EvaluationFixture:
    """A real v2 contract, candidate and production evaluation-task projection."""

    binding: TaskBinding
    contract: TaskContract
    task: Any
    source: str


def v2_candidate_source(
    name: str = V2_TASK,
    dtype: str = "fp16",
    size: str = "small",
    *,
    marker: str = "",
) -> str:
    """Return complete candidate bytes with candidate-owned small-instance geometry."""
    if name != V2_TASK:
        raise KeyError(f"no reusable v2 candidate is registered for {name!r}")
    contract = v2_contract(name, dtype, size)
    symbols = contract.instance.symbols
    source = (ROOT / "tests/support/golden/metal/backend_phase0_kernel.metal").read_text(encoding="utf-8")
    source = re.sub(
        r"launch flash_attention_kernel grid=\S+ threads=\S+",
        f"launch flash_attention_kernel grid={int(symbols['S']) // 32},{int(symbols['B']) * int(symbols['H'])},1 threads=64,1,1",
        source,
        count=1,
    )
    return source.replace("*/\n", f"*/\n// {marker}\n", 1) if marker else source


def v2_evaluation_fixture(
    name: str = V2_TASK,
    dtype: str = "fp16",
    size: str = "small",
    *,
    source: str | None = None,
    target_id: str = "apple-m4-max",
) -> V2EvaluationFixture:
    """Resolve a real candidate; cache only the immutable default fixture, never custom bytes."""
    if source is None:
        return _default_v2_evaluation_fixture(name, dtype, size, target_id)
    return _resolve_v2_evaluation_fixture(name, dtype, size, source=source, target_id=target_id)


@functools.lru_cache(maxsize=None)
def _default_v2_evaluation_fixture(name: str, dtype: str, size: str, target_id: str) -> V2EvaluationFixture:
    return _resolve_v2_evaluation_fixture(name, dtype, size, source=None, target_id=target_id)


def _resolve_v2_evaluation_fixture(
    name: str,
    dtype: str,
    size: str,
    *,
    source: str | None,
    target_id: str,
) -> V2EvaluationFixture:
    """Resolve one real candidate and project it through the production v2 evaluation boundary."""
    from tcenv.backends.metal.portfolio_confirmation import _resolved
    from tcenv.backends.metal.search_candidate import preflight_submission
    from tcenv.backends.metal.kernel_pipeline import v2_evaluation_task
    from tcenv.target import load_target
    from tcenv.track.abi import derive_tensor_memory_contract

    binding = bind_v2(name, dtype, size, target_id=target_id)
    candidate_source = v2_candidate_source(name, dtype, size) if source is None else source
    target = load_target(target_id)
    outcome = preflight_submission(
        binding.contract, candidate_source, target=target,
        track_id=binding.track_id, track_version=binding.track_version,
    )
    if not outcome.ok:
        raise AssertionError(f"the v2 fixture candidate failed preflight: {outcome.stage}: {outcome.rejection}")
    resolved = _resolved(binding.contract, outcome.projection, target)
    abi = derive_tensor_memory_contract(binding.contract.instance, target=target)
    task = v2_evaluation_task(
        SimpleNamespace(
            contract=binding.contract,
            abi=abi,
            target_id=target_id,
            locator=binding.locator,
            selection=SimpleNamespace(geometry=resolved),
        )
    )
    return V2EvaluationFixture(binding, binding.contract, task, candidate_source)


@functools.lru_cache(maxsize=None)
def v2_structural_evaluation_task(
    name: str = V2_TASK,
    dtype: str = "fp16",
    size: str = "small",
    *,
    target_id: str = "apple-m4-max",
):
    """Cheap schema/transport-only v2 projection with explicit structural candidate geometry.

    This does not recompute a deployment closure and is not deployment or GPU evidence.  It is
    for subprocess report fixtures whose queue timeout is intentionally only a few seconds.
    """
    from tcenv.backends.metal.kernel_pipeline import v2_evaluation_task
    from tcenv.target import load_target
    from tcenv.track.abi import derive_tensor_memory_contract
    from tcenv.track.geometry import ResolvedGeometry, ResolvedStage

    contract = v2_contract(name, dtype, size)
    target = load_target(target_id)
    abi = derive_tensor_memory_contract(contract.instance, target=target)
    stage = ResolvedStage(
        index=0, kernel="flash_attention_kernel", grid=(1, 1, 1), threads=(64, 1, 1),
        threadgroup_bytes=0, workspace=(), active=True, when=None, derived_scalars=(),
    )
    geometry = ResolvedGeometry(source="candidate", stages=(stage,))
    return v2_evaluation_task(
        SimpleNamespace(
            contract=contract,
            abi=abi,
            target_id=target_id,
            # Unknown for this schema-only projection; never substitute the task-only pack digest.
            locator=SimpleNamespace(deployment_pack_digest=""),
            selection=SimpleNamespace(geometry=geometry),
        )
    )


class CallbackV2Evaluator(V2LocalEvaluator):
    """CPU-only scripted evaluator with the current v2 executor protocol.

    The callback is a test double, not a production ``evaluate_fn`` fallback.  Reports receive a
    structural native build record derived from the candidate bytes and the real v2 contract;
    no Metal compilation or historical v1 identity is involved.
    """

    def __init__(self, binding: TaskBinding, callback, *, expected_identity=None):
        super().__init__(binding, expected_identity=expected_identity)
        self.callback = callback

    def _evaluate(self, request):
        if self.callback is None:
            raise AssertionError("the scripted v2 evaluator was called without a callback")
        report = self.callback(
            request.task, request.dtype, request.size, request.source,
            mode=request.mode, seed=request.seed, reps=request.reps,
            require_contract_launch=request.require_contract_launch,
        )
        from tests.support.report_build import native_build, native_compile_failure

        if report.status == "ok":
            report.build = native_build(self.binding.contract, request.source.encode("utf-8"))
        elif report.status == "compile_error":
            report.build = native_compile_failure(self.binding.contract, request.source.encode("utf-8"))
        # These optional schema-4 fields belong to the legacy evaluation-task projection.  A
        # scripted v2 transport report leaves them unknown rather than carrying a v1 fingerprint.
        report.evaluation_fingerprint = None
        report.baseline_implementation_digest = None
        report.baseline_environment_digest = None
        return EvaluationResult(report=report, provenance=self._provenance(request))


def v2_session_brief(name: str, dtype: str, size: str, **overrides):
    """Build a SessionBrief with exactly the fields the production v2 orchestrator renders."""
    import json
    from taic.kernel_agent import SessionBrief

    binding = bind_v2(name, dtype, size)
    presentation = binding.presentation()
    tolerances = binding.tolerances()
    values = dict(
        repo=Path("/repo"), task_name=name, dtype=dtype, size=size,
        kernel_name=f"{binding.contract.op_name}_kernel", triton_source="",
        description=presentation.statement,
        contract_text=json.dumps(presentation.abi_document, indent=2, sort_keys=True),
        abi_stub=binding.signature_stub(), rtol=tolerances["rtol"], atol_rel=tolerances["atol_rel"],
        v2=True, reference_source=presentation.reference_source, selfcheck_allow_validate=False,
    )
    values.update(overrides)
    return SessionBrief(**values)


def install_v2_orchestrator(monkeypatch) -> None:
    """Route every constructor in one test module through a real v2 binding/executor."""
    from taic.orchestrator import Orchestrator

    original = Orchestrator.__init__

    def v2_binding(self, *args, **kwargs):
        kwargs["v1_task_loader"] = None
        binding = kwargs.get("binding") or bind_v2(
            args[0], args[1], args[2], target_id=kwargs.get("target_id") or "apple-m4-max",
        )
        kwargs["binding"] = binding
        if kwargs.get("evaluator") is None and kwargs.get("evaluator_factory") is None:
            kwargs["evaluator"] = CallbackV2Evaluator(binding, kwargs.get("evaluate_fn"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(Orchestrator, "__init__", v2_binding)


def v2_pack_digest(name: str = V2_TASK, dtype: str = "fp16", size: str = "paper") -> str:
    """Return the current v2 deployment-pack digest, never an archived v1 track registration."""
    return v2_contract(name, dtype, size).pack_digest


__all__ = [
    "V2_DYNAMIC",
    "V2_TASK",
    "V2_TASKS",
    "CallbackV2Evaluator",
    "V2EvaluationFixture",
    "bind_v2",
    "install_v2_orchestrator",
    "v2_candidate_source",
    "v2_contract",
    "v2_evaluation_fixture",
    "v2_pack_digest",
    "v2_session_brief",
    "v2_structural_evaluation_task",
]
