"""The evaluation cache key and the report round-trip. No GPU."""

from __future__ import annotations

import json

from taic.llm import SearchConfig
from tcenv.evaluate import EvalReport, _cache_read, _cache_write, _eval_cache_key
from tests.support.v1_tasks import V1_TASK


def _key(**over):
    """@brief 用默认参数生成评估缓存键，允许覆盖任意字段。

    @param over 覆盖默认键字段。
    @return 缓存键字符串。
    """
    base = dict(
        task=V1_TASK, dtype="fp16", size="small", mode="full", seed=1, reps=15, cases=None,
        fast_bench=False, require_contract_launch=False, validate=False, measure_serial=False,
        target_id="apple-m4-max", kernel_source="kernel void k() {}\n",
    )
    base.update(over)
    return _eval_cache_key(**base)


def test_the_key_separates_seed_serial_protocol_and_source():
    """@brief 缓存键区分 seed、串行测量和 kernel 源码，同输入保持稳定。
    """
    base = _key()
    assert base != _key(seed=2)
    assert base != _key(measure_serial=True)
    assert base != _key(kernel_source="kernel void other() {}\n")
    assert base == _key()


def test_case_order_backend_and_evaluation_semantics_are_part_of_the_key():
    """@brief 乱序用例、后端身份/版本和可执行评测语义变化都不能复用旧缓存。"""
    assert _key(cases=("normal", "edge")) != _key(cases=("edge", "normal"))
    assert _key(backend_id="metal") != _key(backend_id="other")
    assert _key(backend_version="1") != _key(backend_version="2")
    assert _key(evaluation_fingerprint="semantics-a") != _key(evaluation_fingerprint="semantics-b")


def test_the_key_separates_target_device_and_backend():
    """@brief 同一任务在不同设备/后端不得复用同一份缓存报告。"""
    assert _key(target_id="sim-npu-test", backend_id="sim-npu", backend_version="1") != _key()
    assert _key(target_id="other-device", backend_id="metal") != _key()
    assert _key(target_id="sim-npu-test", backend_id="sim-npu", backend_version="1") != _key(
        target_id="sim-npu-test", backend_id="metal", backend_version="1"
    )


def test_a_cached_report_round_trips(tmp_path, monkeypatch):
    """@brief EvalReport 写入缓存后能读回，缓存文件带 schema_version。

    @param tmp_path pytest 提供的临时目录，用作缓存目录。
    @param monkeypatch pytest 补丁工具，用于设置缓存目录环境变量。
    """
    monkeypatch.setenv("TCENV_EVAL_CACHE_DIR", str(tmp_path))
    from tests.support.report_build import attach_identity, ok_build

    from tasks import get_task

    report = EvalReport(
        task=V1_TASK, dtype="fp16", size="small", mode="full", status="ok",
        speedup=1.2, candidate_us=1.0, target_id="apple-m4-max",
    )
    report.benchmark = {"candidate": {"cv": 0.01}}
    report.build = ok_build(V1_TASK, "fp16", "small")
    attach_identity(report, get_task(V1_TASK, "fp16", "small"))
    _cache_write("k", report)
    cached = _cache_read("k")
    assert cached is not None and cached.speedup == 1.2 and cached.status == "ok"
    assert json.loads((tmp_path / "k.json").read_text())["schema_version"] == 4


def test_the_search_defaults_keep_serial_off_and_the_cache_on():
    """@brief SearchConfig 默认关闭串行测量并打开评估缓存。
    """
    config = SearchConfig()
    assert config.measure_serial is False
    assert config.eval_cache is True


def test_the_evaluation_identity_is_the_registered_fields_not_a_closure_repr():
    """@brief 评测身份＝登记的 ``{implementation_id, implementation_version, params}`` + pack 摘要。

    闭包的 ``repr``/源码/默认值不是身份：换一个行为相同、文本不同的函数对象，身份必须不变；
    改登记的 ``params`` 或 task-pack 摘要，身份必须变。
    """
    import dataclasses

    import pytest

    from tcenv.evaluate import _evaluation_fingerprint
    from tcenv.evaluation import EvaluationError
    from tasks import get_task

    task = get_task(V1_TASK, "fp16", "small")
    identity = _evaluation_fingerprint(task)
    assert identity == _evaluation_fingerprint(get_task(V1_TASK, "fp16", "small"))

    def renamed_gen_inputs(case: str, seed: int):  # different code text, same behaviour
        return task.gen_inputs(case, seed)

    rewritten = dataclasses.replace(task, gen_inputs=renamed_gen_inputs)
    assert _evaluation_fingerprint(rewritten) == identity, "the identity must not hash the closure"

    registrations = dict(task.evaluation_implementations)
    changed = dict(registrations["reference"])
    changed["params"] = {**changed["params"], "accumulate": "float32"}
    assert _evaluation_fingerprint(dataclasses.replace(task, evaluation_implementations={**registrations, "reference": changed})) != identity

    moved_pack = dict(registrations["task_pack"])
    moved_pack["digest"] = "0" * 64
    assert _evaluation_fingerprint(dataclasses.replace(task, evaluation_implementations={**registrations, "task_pack": moved_pack})) != identity

    # Fail closed: no registration, or a bogus version, is refused instead of hashed.
    with pytest.raises(EvaluationError, match="no registered evaluation implementations"):
        _evaluation_fingerprint(dataclasses.replace(task, evaluation_implementations={}))
    broken = dict(registrations["gen_inputs"])
    broken["implementation_version"] = 0
    with pytest.raises(EvaluationError, match="implementation_version"):
        _evaluation_fingerprint(dataclasses.replace(task, evaluation_implementations={**registrations, "gen_inputs": broken}))


def test_the_cache_key_is_disabled_when_an_identity_component_is_unknown(monkeypatch):
    """@brief 环境分量不可识别时禁用缓存，绝不把 ``None``/``"unknown"`` 哈希成可复用键。"""
    from tcenv import evaluate as ev

    monkeypatch.setattr(ev, "baseline_environment_digest", lambda *a, **k: None)
    assert _key() is None
    assert _key(backend_id="metal") is None


def test_a_cached_value_must_describe_the_same_measurement_context(tmp_path, monkeypatch):
    """@brief schema 4 合法但身份不同的缓存值是未命中；本地 Metal 还要求 native full。"""
    from tasks import get_task

    from tcenv import evaluate as ev
    from tests.support.report_build import attach_build, attach_identity

    from tests.support.report_build import simulated_build, task_identity

    monkeypatch.setenv("TCENV_EVAL_CACHE_DIR", str(tmp_path))
    task_obj = get_task(V1_TASK, "fp16", "small")
    source = "kernel void k() {}\n"
    expected = ev._eval_cache_identity(
        task=V1_TASK, dtype="fp16", size="small", mode="full", seed=1, reps=15, cases=None,
        fast_bench=False, require_contract_launch=False, validate=False, measure_serial=False,
        target_id="apple-m4-max", kernel_source=source, backend_id="metal", backend_version="1",
    )
    assert expected is not None
    report = EvalReport(
        task=V1_TASK, dtype="fp16", size="small", mode="full", seed=1, status="ok",
        target_id="apple-m4-max", backend="metal", backend_version="1",
    )
    report.build = simulated_build(task_obj, source.encode("utf-8"))
    attach_identity(report, task_obj, backend_id="metal")
    # The simulator build is structurally fine, but a local Metal hit must be native.
    assert report.evaluation_fingerprint == expected["evaluation_fingerprint"]
    assert not ev._cache_value_matches(report, expected, backend_id="metal")
    assert ev._cache_value_matches(report, expected, backend_id="sim-npu")
    # A different identity (for example an oracle/baseline change) is a miss.
    report.evaluation_fingerprint = "0" * 64
    assert not ev._cache_value_matches(report, expected, backend_id="sim-npu")
    # A different candidate is a miss even with the same identity.
    report.evaluation_fingerprint = expected["evaluation_fingerprint"]
    report.build = simulated_build(task_obj, b"kernel void other() {}\n")
    assert not ev._cache_value_matches(report, expected, backend_id="sim-npu")


def test_a_cached_schema_3_file_is_a_miss(tmp_path, monkeypatch):
    """@brief 旧 schema 的缓存文件读成未命中（不迁移、不双读）。"""
    monkeypatch.setenv("TCENV_EVAL_CACHE_DIR", str(tmp_path))
    path = tmp_path / "k.json"
    path.write_text(json.dumps({
        "schema_version": 3, "task": V1_TASK, "dtype": "fp16", "size": "small",
        "mode": "full", "status": "ok",
    }))
    assert _cache_read("k") is None


def test_the_eval_cache_schema_moved_to_three():
    """@brief 缓存键 schema 升到 3；旧键不再命中。"""
    from tcenv.evaluate import EVAL_CACHE_SCHEMA

    assert EVAL_CACHE_SCHEMA == 3
