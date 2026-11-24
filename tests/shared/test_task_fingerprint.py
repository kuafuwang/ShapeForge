"""Current task-fingerprint coverage using a real v2 evaluation projection.

CPU only.  Historical v1 digest goldens and the migration oracle are retired; the tests below
exercise the identity inputs consumed by the current evaluator, not an archived task assembly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.support.v2_specimen import V2_TASK, v2_evaluation_fixture
from tcenv.lowering import contract_document
from tcenv.task import task_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def test_the_v2_document_binds_the_complete_derived_projections():
    """The canonical document binds the complete derived projections and rendered texts."""
    from tcenv.identity import fingerprint

    task = v2_evaluation_fixture(V2_TASK, "fp16", "small").task
    document = {
        "fingerprint_kind": "task",
        "fingerprint_version": 2,
        "task": task.name,
        "dtype": task.dtype,
        "size": task.size,
        "backend_id": "metal",
        "semantic_instance": task.semantic_instance.fingerprint_document(),
        "specialization": task.specialization.fingerprint_document(),
        "triton_source": task.triton_source,
        "description": task.description,
        "reference_source": task.reference_for("metal"),
    }
    assert set(document) == {
        "fingerprint_kind", "fingerprint_version", "task", "dtype", "size", "backend_id",
        "semantic_instance", "specialization", "triton_source", "description", "reference_source",
    }
    assert document["specialization"]["contract"] == contract_document(task.contract)
    assert "gpu" in document["specialization"]["contract"]
    assert fingerprint(document) == task_fingerprint(task, "metal")
    assert task_fingerprint(task) == task_fingerprint(task, "metal")
    assert task_fingerprint(task, "sim-npu") != task_fingerprint(task, "metal")


def test_changing_only_the_description_changes_the_fingerprint():
    task = v2_evaluation_fixture(V2_TASK, "fp16", "small").task
    edited = dataclasses.replace(task, description=task.description + " (rewritten track text)")
    assert edited.description != task.description
    assert task_fingerprint(edited) != task_fingerprint(task)


def test_changing_only_the_backend_reference_kernel_changes_the_fingerprint():
    task = v2_evaluation_fixture(V2_TASK, "bf16", "small").task
    edited = dataclasses.replace(
        task, naive_msl=task.naive_msl + "\n// one more line\n", reference_sources={},
    )
    assert task_fingerprint(edited) != task_fingerprint(task)


def test_a_task_without_semantic_projections_is_rejected():
    """There is one identity path; missing projections are a construction error."""
    stripped = dataclasses.replace(
        v2_evaluation_fixture(V2_TASK, "fp16", "small").task,
        semantic_instance=None,
        specialization=None,
    )
    with pytest.raises(ValueError, match="transitional pre-R2 hash path is gone"):
        task_fingerprint(stripped)
