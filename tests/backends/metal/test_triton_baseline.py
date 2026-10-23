"""Metal triton-msl baseline cache and quiet-machine checks."""

from __future__ import annotations

import json

import pytest

from tasks import get_task
from tcenv.backends.metal import triton_baseline as tb
from tcenv.triton_runner import spec_fingerprint


def test_measure_explains_a_missing_toolchain(monkeypatch):
    from tcenv.backends.metal import triton_baseline as tb

    monkeypatch.setattr(tb, "available", lambda: True)
    monkeypatch.setattr(tb, "metal_toolchain_ok", lambda: False)
    with pytest.raises(RuntimeError, match="xcodebuild -downloadComponent MetalToolchain"):
        tb.measure("relu", "fp16", "small")


def test_only_trustworthy_cached_measurements_are_used(tmp_path, monkeypatch):
    """A measurement taken on a busy machine (inconsistent with its own sweep) or a launch-bound one must never feed a report."""
    import json

    from tcenv.backends.metal import triton_baseline as tb

    monkeypatch.setattr(tb, "CACHE_DIR", tmp_path)
    from tcenv.triton_runner import spec_fingerprint

    fp = {t: spec_fingerprint(get_task(t, "fp16", "paper")) for t in ("relu", "sigmoid", "silu", "swiglu", "softmax", "rope", "rmsnorm")}
    good = {"status": "ok", "consistent": True, "ratio_to": {"mlx": 1.3}, "median_us": {"triton_msl": 190.0, "mlx": 146.0}}
    cases = {
        "relu": {**good, "spec_fingerprint": fp["relu"]},
        "sigmoid": {**good, "consistent": False, "spec_fingerprint": fp["sigmoid"]},
        "silu": {**good, "warning": "launch-bound", "spec_fingerprint": fp["silu"]},
        "swiglu": {k: v for k, v in good.items() if k != "consistent"} | {"spec_fingerprint": fp["swiglu"]},
        "softmax": {"status": "unsupported", "spec_fingerprint": fp["softmax"]},
        "rope": {**good, "spec_fingerprint": "0123456789abcdef"},  # measured on another spec / size / contract
        "rmsnorm": dict(good),  # measured before fingerprints existed
    }
    for name, d in cases.items():
        (tmp_path / f"{name}_fp16_paper.json").write_text(json.dumps(d))
    assert tb.load_cached("relu", "fp16", "paper") is not None
    for name in ("sigmoid", "silu", "swiglu", "softmax", "rope", "rmsnorm"):
        assert tb.load_cached(name, "fp16", "paper") is None, name


def test_estimate_scales_the_ratio_with_the_live_vendor_time():
    from tcenv.backends.metal import triton_baseline as tb

    est = tb.estimate({"ratio_to": {"mlx": 1.3, "torch_mps": 1.4}}, {"mlx": 100.0, "torch_mps": 90.0})
    assert est["estimated_us"] == pytest.approx(130.0) and "mlx" in est["basis"]
    assert tb.estimate({"ratio_to": {"mlx": 1.3}}, {"torch_mps": 90.0}) is None


def test_busy_process_check_ignores_stopped_and_unrelated_processes():
    from tcenv.backends.metal import triton_baseline as tb

    assert isinstance(tb.busy_processes(), list)


def test_wait_quiet_returns_at_once_on_a_quiet_machine_and_gives_up_after_the_limit(monkeypatch):
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
