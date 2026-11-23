"""Metal profiling advice formatting (xctrace trace summary -> findings).

The orchestrator-side profiling integration (``profile_best``/``profile_poor`` + Skill 2b in the
v1 system prompt) was retired with the v1 execution path: a v2 run refuses those switches at
startup.  Its tests are retired with it.  What remains here is the task-independent part: a
``TraceSummary`` must only produce advice the recorded tables actually support.
"""

from __future__ import annotations

import json

import pytest

from tcenv.backends.metal.xctrace import TraceSummary, advice, format_for_llm


def _trace(spilled=4096, events=3, code_bytes=6000, active=0.95):
    """@brief 构造一个字段固定的 TraceSummary 作为 profile 结果。

    @param spilled 寄存器溢出字节数。
    @param events 溢出事件数。
    @param code_bytes shader 代码字节数。
    @param active GPU 活跃时间占比。
    @return TraceSummary。
    """
    from tcenv.backends.metal.xctrace import TraceSummary

    return TraceSummary(window_s=1.0, n_dispatches=40, dispatch_us_median=50.0, dispatch_us_p10=40.0, dispatch_us_p90=60.0, gpu_active_fraction=active,
                        spilled_bytes=spilled, spill_events=events, shader_code_bytes=code_bytes, thermal_states=["Nominal"])


def test_xctrace_advice_only_reports_what_the_trace_supports():
    """@brief advice 只报告 trace 支持的结论；无问题时明说未发现，format_for_llm 只给出一条 findings。
    """
    from tcenv.backends.metal.xctrace import advice, format_for_llm

    clean = advice(_trace(spilled=0, events=0))
    assert len(clean) == 1 and "Nothing was found in the recorded tables" in clean[0]
    sick = advice(_trace(spilled=100, events=2, code_bytes=100_000, active=0.5))
    text = " ".join(sick)
    assert "spilled 100 bytes" in text and "active only 50%" in text and "compiled shader is large" in text and "Nothing was found" not in text
    t = _trace()
    t.thermal_states = ["Nominal", "Fair"]
    assert any("throttled" in a for a in advice(t))
    assert format_for_llm(_trace()).count("findings:") == 1
