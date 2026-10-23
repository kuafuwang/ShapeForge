"""Phase 0 baseline: prompt/contract identity and the recorded GPU comparison."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import fields
from pathlib import Path

import pytest

from observe.index import INDEX_SCHEMA_VERSION
from taic import prompts
from taic.trust import current_prompt_sha1
from tasks import get_task
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION, REPORT_SCHEMA_VERSION
from tcenv.task import task_fingerprint

BASELINE = Path("docs/架构/backend-phase0-baseline.json")
REPO = Path(__file__).resolve().parents[3]
MAX_CANDIDATE_PCT = 2.0
MAX_GPU_PCT = 2.0
MAX_MLX_PCT = 2.0
MAX_SPEEDUP_DELTA = 0.03


def _doc() -> dict:
    return json.loads(BASELINE.read_text())


def test_the_phase0_baseline_still_matches_the_source_and_prompt():
    doc = _doc()
    source = doc["source"]
    path = REPO / source["path"]
    assert path.is_file(), f"tracked phase-0 source is missing: {path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    task = get_task(doc["task"], doc["dtype"], doc["size"])
    assert doc["report_schema_version"] == REPORT_SCHEMA_VERSION == 3
    assert doc["protocol_version"] == PROTOCOL_VERSION == "3"
    assert doc["index_schema_version"] == INDEX_SCHEMA_VERSION == 11
    assert doc["top_level_fields"] == [field.name for field in fields(EvalReport)]
    assert doc["prompt_sha1"] == current_prompt_sha1()
    prompt = prompts.system_prompt(task, use_tools=False, target_id=doc["target_id"])
    assert doc["system_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert doc["contract_description_sha256"] == hashlib.sha256(task.contract.describe().encode()).hexdigest()
    assert doc["task_fingerprint"] == task_fingerprint(task)


def test_the_recorded_gpu_delta_is_recomputed_and_within_a_fixed_gate():
    gpu = _doc()["gpu"]
    acceptance = gpu["acceptance"]
    assert acceptance["candidate_abs_pct"] == MAX_CANDIDATE_PCT
    assert acceptance["gpu_abs_pct"] == MAX_GPU_PCT
    assert acceptance["mlx_abs_pct"] == MAX_MLX_PCT
    assert acceptance["speedup_abs"] == MAX_SPEEDUP_DELTA
    current = gpu["current_runs"]
    assert len(current) == 3
    candidates = [row["candidate_us"] for row in current]
    gpus = [row["candidate_gpu_us_median"] for row in current]
    mlx = [row["mlx_us"] for row in current]
    speedups = [row["speedup"] for row in current]
    expected = {
        "candidate_us": statistics.median(candidates),
        "candidate_gpu_us_median": statistics.median(gpus),
        "mlx_us": statistics.median(mlx),
        "speedup": statistics.median(speedups),
    }
    for key, value in expected.items():
        assert gpu["current_median"][key] == pytest.approx(value)
    old = gpu["old"]
    delta = gpu["delta_percent"]
    for key, old_key in (("candidate_us", "candidate_us"), ("candidate_gpu_us_median", "candidate_gpu_us_median"), ("mlx_us", "mlx_us")):
        assert delta[key] == pytest.approx((expected[key] - old[old_key]) / old[old_key] * 100, abs=0.01)
    assert delta["speedup"] == pytest.approx(expected["speedup"] - old["speedup"], abs=0.0001)
    assert abs(delta["candidate_us"]) <= MAX_CANDIDATE_PCT
    assert abs(delta["candidate_gpu_us_median"]) <= MAX_GPU_PCT
    assert abs(delta["mlx_us"]) <= MAX_MLX_PCT
    assert abs(delta["speedup"]) <= MAX_SPEEDUP_DELTA
    assert gpu["passes"] is True
