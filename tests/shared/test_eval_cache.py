"""The evaluation cache key and the report round-trip. No GPU."""

from __future__ import annotations

import json

from taic.llm import SearchConfig
from tcenv.evaluate import EvalReport, _cache_read, _cache_write, _eval_cache_key


def _key(**over):
    """@brief 用默认参数生成评估缓存键，允许覆盖任意字段。

    @param over 覆盖默认键字段。
    @return 缓存键字符串。
    """
    base = dict(
        task="relu", dtype="fp16", size="small", mode="full", seed=1, reps=15, cases=None,
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
    report = EvalReport(task="relu", dtype="fp16", size="small", mode="full", status="ok", speedup=1.2, candidate_us=1.0)
    report.benchmark = {"candidate": {"cv": 0.01}}
    _cache_write("k", report)
    cached = _cache_read("k")
    assert cached is not None and cached.speedup == 1.2 and cached.status == "ok"
    assert json.loads((tmp_path / "k.json").read_text())["schema_version"] >= 1


def test_the_search_defaults_keep_serial_off_and_the_cache_on():
    """@brief SearchConfig 默认关闭串行测量并打开评估缓存。
    """
    config = SearchConfig()
    assert config.measure_serial is False
    assert config.eval_cache is True
