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
    """@brief 读入并解析 phase-0 基线 JSON 文档。

    @return 基线文件的解析结果（dict）。
    """
    return json.loads(BASELINE.read_text())


def test_the_phase0_baseline_still_matches_the_source_and_prompt():
    """@brief 校验记录的 phase-0 基线仍与当前源码、各 schema 版本、prompt 与 task 指纹一致。

    覆盖：源码文件存在且 sha256 未变；report／protocol／index schema 版本号固定在文档记录值；
    EvalReport 顶层字段顺序、prompt sha1、系统提示词与 contract 描述的 sha256、任务指纹均匹配。
    """
    doc = _doc()
    source = doc["source"]
    path = REPO / source["path"]
    assert path.is_file(), f"tracked phase-0 source is missing: {path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    task = get_task(doc["task"], doc["dtype"], doc["size"])
    assert doc["report_schema_version"] == REPORT_SCHEMA_VERSION == 4
    assert doc["protocol_version"] == PROTOCOL_VERSION == "3"
        # Index schema 13 adds the report state and the build/evaluation identity columns (§8):
    # only a schema-4 report is current evidence, so the columns exist and stay NULL for an old one.
    assert doc["index_schema_version"] == INDEX_SCHEMA_VERSION == 13
    assert doc["top_level_fields"] == [field.name for field in fields(EvalReport)]
    assert doc["prompt_sha1"] == current_prompt_sha1()
    prompt = prompts.system_prompt(task, use_tools=False, target_id=doc["target_id"])
    assert doc["system_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert doc["contract_description_sha256"] == hashlib.sha256(task.contract.describe().encode()).hexdigest()
    if task.semantic_instance is None:
        assert doc["task_fingerprint"] == task_fingerprint(task)
    else:
        # R1/R2 deliberately reset the task identity to the versioned canonical document
        # (§3.1): this record is a frozen historical baseline that must not be rewritten or
        # recomputed (plan §2.1), so the recorded pre-v2 hash must *differ* from the live one.
        # The live identity is covered by tests/shared/test_task_fingerprint.py.
        assert doc["task_fingerprint"] != task_fingerprint(task)


def test_the_recorded_gpu_delta_is_recomputed_and_within_a_fixed_gate():
    """@brief 用三次记录运行重算 GPU 中位数与相对旧值的偏差，并核对是否落在固定验收门槛内。

    门槛：candidate／gpu／mlx 时间与 speedup 的绝对偏差分别不超过 2%、2%、2% 与 0.03。
    同时校验记录里的 acceptance 阈值与常量一致、中位数为三次运行的 median、delta 由中位数与旧值算出。
    """
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
