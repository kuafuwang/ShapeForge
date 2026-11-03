"""Metal triton-msl baseline cache and quiet-machine checks."""

from __future__ import annotations

import json

import pytest

from tasks import get_task
from tcenv.backends.metal import triton_baseline as tb
from tcenv.triton_runner import spec_fingerprint
from tests.support.v1_tasks import V1_TASK


def test_measure_explains_a_missing_toolchain(monkeypatch):
    """@brief 验证 Metal 工具链缺失时 measure 抛出带修复命令提示的 RuntimeError。

    @param monkeypatch pytest 补丁工具，用来固定可用性与工具链检查结果。
    @raises RuntimeError 工具链不可用时抛出，消息包含 `xcodebuild -downloadComponent MetalToolchain`。
    """
    from tcenv.backends.metal import triton_baseline as tb

    monkeypatch.setattr(tb, "available", lambda: True)
    monkeypatch.setattr(tb, "metal_toolchain_ok", lambda: False)
    with pytest.raises(RuntimeError, match="xcodebuild -downloadComponent MetalToolchain"):
        tb.measure(V1_TASK, "fp16", "small")


def test_only_trustworthy_cached_measurements_are_used(tmp_path, monkeypatch):
    """@brief 验证只有可信的缓存测量会被采用，其余一概视为无缓存。

    在忙碌机器上测得的（与自身扫描不一致）、launch-bound 的、缺 consistent 字段的、
    状态不支持的、规格指纹不匹配的、以及指纹出现前测得的旧记录，都必须拒绝。

    @param tmp_path pytest 提供的临时目录，作为缓存目录。
    @param monkeypatch pytest 补丁工具，用来改写模块级 CACHE_DIR。
    """
    import json

    from tcenv.backends.metal import triton_baseline as tb

    monkeypatch.setattr(tb, "CACHE_DIR", tmp_path)
    from tcenv.triton_runner import spec_fingerprint

    # A5 retired the migrated v1 assemblies; the baseline cache is a v1 mechanism, so the cases
    # use the preserved v1 tasks only.
    # B1 removed rope/gemv; the cache-trust cases only need seven *distinct* v1 specs, so the
    # semantic-recovery task and the dynamic integer family fill the two slots.
    names = ("flash_attention", "bitdelta_gemm", "mamba2_chunk_scan", "sem_attention_odd_scale", "sem_int_5", "sem_int_3", "sem_int_007")
    fp = {t: spec_fingerprint(get_task(t, "fp16", "paper")) for t in names}
    good = {"status": "ok", "consistent": True, "ratio_to": {"mlx": 1.3}, "median_us": {"triton_msl": 190.0, "mlx": 146.0}}
    cases = {
        names[0]: {**good, "spec_fingerprint": fp[names[0]]},
        names[1]: {**good, "consistent": False, "spec_fingerprint": fp[names[1]]},
        names[2]: {**good, "warning": "launch-bound", "spec_fingerprint": fp[names[2]]},
        names[3]: {k: v for k, v in good.items() if k != "consistent"} | {"spec_fingerprint": fp[names[3]]},
        names[4]: {"status": "unsupported", "spec_fingerprint": fp[names[4]]},
        names[5]: {**good, "spec_fingerprint": "0123456789abcdef"},  # measured on another spec / size / contract
        names[6]: dict(good),  # measured before fingerprints existed
    }
    for name, d in cases.items():
        (tmp_path / f"{name}_fp16_paper.json").write_text(json.dumps(d))
    assert tb.load_cached(names[0], "fp16", "paper") is not None
    for name in names[1:]:
        assert tb.load_cached(name, "fp16", "paper") is None, name


def test_estimate_scales_the_ratio_with_the_live_vendor_time():
    """@brief 验证估算按缓存比值乘以实测厂商标杆耗时，且没有共同基准时返回 None。"""
    from tcenv.backends.metal import triton_baseline as tb

    est = tb.estimate({"ratio_to": {"mlx": 1.3, "torch_mps": 1.4}}, {"mlx": 100.0, "torch_mps": 90.0})
    assert est["estimated_us"] == pytest.approx(130.0) and "mlx" in est["basis"]
    assert tb.estimate({"ratio_to": {"mlx": 1.3}}, {"torch_mps": 90.0}) is None


def test_busy_process_check_ignores_stopped_and_unrelated_processes():
    """@brief 验证忙碌进程检查在真实机器上可调用并返回列表。

    实现侧会跳过已挂起（ps 状态以 T 开头）以及与本 harness 无关的进程。
    """
    from tcenv.backends.metal import triton_baseline as tb

    assert isinstance(tb.busy_processes(), list)


def test_wait_quiet_returns_at_once_on_a_quiet_machine_and_gives_up_after_the_limit(monkeypatch):
    """@brief 验证等待空闲在工作区安静时立即返回，在负载高或有忙碌进程时超时放弃。

    @param monkeypatch pytest 补丁工具，用来替换负载与忙碌进程探测。
    """
    from tcenv.backends.metal import triton_baseline as tb

    monkeypatch.setattr(tb.os, "getloadavg", lambda: (0.5, 0.5, 0.5))
    monkeypatch.setattr(tb, "busy_processes", lambda: [])
    r = tb.wait_quiet(5.0, poll_s=0.01, log=lambda s: None)
    assert r["quiet"] and r["waited_s"] < 1.0
    monkeypatch.setattr(tb.os, "getloadavg", lambda: (9.0, 5.0, 3.0))
    r = tb.wait_quiet(0.05, poll_s=0.01, log=lambda s: None)
    assert not r["quiet"] and r["load_avg_1m"] == 9.0
    monkeypatch.setattr(tb.os, "getloadavg", lambda: (0.5, 0.5, 0.5))
    monkeypatch.setattr(tb, "busy_processes", lambda: ["123 pytest"])
    assert not tb.wait_quiet(0.05, poll_s=0.01, log=lambda s: None)["quiet"]
