"""Phase 0 baseline: prompt/contract identity and the recorded GPU comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pytest

from taic import prompts
from taic.trust import current_prompt_sha1
from tasks import get_task
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION, REPORT_SCHEMA_VERSION
from tcenv.task import task_fingerprint

BASELINE = Path("docs/架构/backend-phase0-baseline.json")
REPO = Path(__file__).resolve().parents[1]


def test_the_phase0_baseline_still_matches_the_source_and_prompt():
    if not BASELINE.is_file():
        pytest.skip("phase-0 baseline is absent")
    doc = json.loads(BASELINE.read_text())
    source = doc["source"]
    path = REPO / source["path"]
    if not path.is_file():
        pytest.skip("the recorded GPU source is not present")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    task = get_task(doc["task"], doc["dtype"], doc["size"])
    assert doc["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert doc["protocol_version"] == PROTOCOL_VERSION
    assert doc["top_level_fields"] == [field.name for field in fields(EvalReport)]
    assert doc["prompt_sha1"] == current_prompt_sha1()
    prompt = prompts.system_prompt(task, use_tools=False, target_id=doc["target_id"])
    assert doc["system_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert doc["contract_description_sha256"] == hashlib.sha256(task.contract.describe().encode()).hexdigest()
    assert doc["task_fingerprint"] == task_fingerprint(task)


def test_the_recorded_gpu_delta_is_within_its_declared_gate():
    if not BASELINE.is_file():
        pytest.skip("phase-0 baseline is absent")
    gpu = json.loads(BASELINE.read_text())["gpu"]
    tolerance = gpu["acceptance"]
    delta = gpu["delta_percent"]
    assert abs(delta["candidate_us"]) <= tolerance["candidate_abs_pct"]
    assert abs(delta["candidate_gpu_us_median"]) <= tolerance["gpu_abs_pct"]
    assert abs(delta["mlx_us"]) <= tolerance["mlx_abs_pct"]
    assert abs(delta["speedup"]) <= tolerance["speedup_abs"]
    assert gpu["passes"] is True
