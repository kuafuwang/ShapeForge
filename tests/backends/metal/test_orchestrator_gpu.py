"""Orchestrator + the real evaluation environment, with a scripted (not real) LLM. Every kernel here is safe and terminating."""

from __future__ import annotations

import asyncio
import re

import pytest

from tasks import get_task
from tcenv.evaluate import evaluate

from ...support.fakes import FakeClient, make_response
from ...shared.test_orchestrator import PLAN, make_orch

pytestmark = pytest.mark.gpu


def fenced(src: str) -> str:
    return f"```metal\n{src}\n```"


def test_real_environment_end_to_end_with_scripted_llm(tmp_path):
    naive = get_task("relu", "fp16", "small").naive_msl
    broken = naive.replace("y[j] = half(v < 0.0f ? 0.0f : v)", "y[j] = undefined_symbol")
    wrong = naive.replace("v < 0.0f ? 0.0f : v", "v < 0.0f ? 0.1f * v : v")
    vector = """
kernel void relu_kernel(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                        uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    uint base = tg * BLOCK_SIZE + lid * 8;                       // 128 threads x 8 halves = BLOCK_SIZE
    half4 a = *reinterpret_cast<device const half4*>(x + base);
    half4 b = *reinterpret_cast<device const half4*>(x + base + 4);
    *reinterpret_cast<device half4*>(y + base) = max(a, half4(0));
    *reinterpret_cast<device half4*>(y + base + 4) = max(b, half4(0));
}
"""

    def route(messages, **kw):
        last = messages[-1]["content"]
        if "Translate this Triton kernel" in last:
            return make_response(fenced(broken))
        if "evaluation of your kernel failed" in last:
            if "STATUS: compile_error" in last:
                assert "undefined_symbol" in last and "kernel.metal:" in last  # the real compiler diagnostics reach the model
            else:
                assert "STATUS: incorrect" in last and "out of tolerance" in last
            return make_response(fenced(naive))
        if "Diagnose the dominant bottleneck" in last:
            return make_response(PLAN)
        title = re.search(r"\*\*(.+?)\*\*", last).group(1)
        return make_response(fenced({"A": naive, "B": wrong, "C": vector}[title]))

    assert broken != naive and wrong != naive
    orch = make_orch(tmp_path, route, [], max_rounds=0, evaluate_fn=evaluate)
    orch.search.eval_reps = 5
    res = asyncio.run(orch.run())
    statuses = [c["status"] for c in res.candidates]
    assert statuses[:2] == ["compile_error", "ok"]
    assert res.best_speedup is not None and res.confirmed_status == "ok" and res.confirmed_speedup > 0
    assert res.hangs == 0
