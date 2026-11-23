"""Current v2 task fixtures for mechanism tests.

This module deliberately returns production ``TaskContract`` and ``TaskBinding`` objects.  It
does not reconstruct a v1 ``Task``, archived launch geometry, or a historical v1 fingerprint.
Execution geometry and the candidate kernel name belong to each test candidate, not to the task.
"""

from __future__ import annotations

import functools
from pathlib import Path

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


class CallbackV2Evaluator(V2LocalEvaluator):
    """CPU-only scripted evaluator with the current v2 executor protocol.

    The callback is a test double, not a production ``evaluate_fn`` fallback.  Reports receive a
    structural native build record derived from the candidate bytes and the real v2 contract;
    no Metal compilation or historical v1 identity is involved.
    """

    def __init__(self, binding: TaskBinding, callback):
        super().__init__(binding)
        self.callback = callback

    def _evaluate(self, request):
        if self.callback is None:
            raise AssertionError("the scripted v2 evaluator was called without a callback")
        report = self.callback(
            request.task, request.dtype, request.size, request.source,
            mode=request.mode, seed=request.seed, reps=request.reps,
        )
        from tests.support.report_build import native_build, native_compile_failure

        if report.status == "ok" and report.build is None:
            report.build = native_build(self.binding.contract, request.source.encode("utf-8"))
        elif report.status == "compile_error" and report.build is None:
            report.build = native_compile_failure(self.binding.contract, request.source.encode("utf-8"))
        return EvaluationResult(report=report, provenance=self._provenance(request))


def v2_pack_digest(name: str = V2_TASK, dtype: str = "fp16", size: str = "paper") -> str:
    """Return the current v2 deployment-pack digest, never an archived v1 track registration."""
    return v2_contract(name, dtype, size).pack_digest


__all__ = [
    "V2_DYNAMIC",
    "V2_TASK",
    "V2_TASKS",
    "CallbackV2Evaluator",
    "bind_v2",
    "v2_contract",
    "v2_pack_digest",
]
