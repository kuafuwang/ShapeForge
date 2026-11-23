"""Frozen names of the retired v1 task family.

This module no longer assembles a v1 ``tcenv.task.Task`` and it is not a task registry.  The v1
evaluation entry points, the worker pipelines and the global specimen hooks are retired; a live
search or evaluation always goes through :mod:`tcenv.taskcontract` and the v2 standard entry.

The three name tuples survive for one reason only: ``docs/架构/清微NPU数据模拟/simulate.py`` is a
closed verifier package whose bytes are pinned by its own ``manifest.sha256``, and that script
imports ``V1_TASK`` as the *fixture label* it writes into its recorded NPU protocol report.  That
label is historical data, not a callable task.  Rewriting the closed package to drop the import
would rewrite an acceptance package, so the label stays here instead.

Nothing may import this module to obtain an evaluable task.  Current mechanism tests use
``tests.support.v2_specimen`` (real ``TaskContract`` / ``TaskBinding``) and historical format
oracles read their frozen JSON directly.
"""

from __future__ import annotations

#: The fixture label the archived NPU protocol report was recorded with.
V1_TASK = "sem_attention_odd_scale"

#: Task names that once had an archived v1 contract (historical inventory only).
V1_TASKS: tuple[str, ...] = ("sem_attention_odd_scale", "sem_int_5", "sem_int_3")

#: Representative seeds of the retired dynamic integer family.
V1_DYNAMIC: tuple[str, ...] = ("sem_int_5", "sem_int_3", "sem_int_007")

__all__ = ["V1_DYNAMIC", "V1_TASK", "V1_TASKS"]
