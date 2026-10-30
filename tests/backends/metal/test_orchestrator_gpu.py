"""Orchestrator + the real evaluation environment, with a scripted (not real) LLM. Every kernel here is safe and terminating."""

from __future__ import annotations

import asyncio
import re

import pytest

from tasks import get_task
from tcenv.evaluate import confirmation_expected_document, evaluate

from ...support.fakes import FakeClient, make_response
from ...support.orchestrator import PLAN, make_orch

pytestmark = pytest.mark.gpu


def fenced(src: str) -> str:
    """@brief 把 MSL 源码包成 ```metal 代码块，也就是脚本化模型在回答里给出的内核形状。

    @param src MSL 源码。
    @return 带围栏的文本。
    """
    return f"```metal\n{src}\n```"


def test_real_environment_end_to_end_with_scripted_llm(tmp_path):
    """@brief 用脚本化模型和真实评测环境跑通一次端到端搜索：先落一个编译失败的变体、再修正，最后给出确认结果且没有挂起。

    @param tmp_path pytest 提供的临时目录。
    """
    naive = get_task("relu", "fp16", "paper").naive_msl
    broken = naive.replace("y[j] = half(v < 0.0f ? 0.0f : v)", "y[j] = undefined_symbol")
    wrong = naive.replace("v < 0.0f ? 0.0f : v", "v < 0.0f ? 0.1f * v : v")
    vector = """
kernel void relu_kernel(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                        uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    uint base = tg * BLOCK_SIZE + lid * 8;                       // 128 threads x 8 halves = BLOCK_SIZE
    half4 a = *reinterpret_cast<device const half4*>(x + base);
    half4 b = *reinterpret_cast<device const half4*>(x + base + 4);
    y[base + 0] = half(a[0] < 0.0f ? 0.0f : a[0]);
    y[base + 1] = half(a[1] < 0.0f ? 0.0f : a[1]);
    y[base + 2] = half(a[2] < 0.0f ? 0.0f : a[2]);
    y[base + 3] = half(a[3] < 0.0f ? 0.0f : a[3]);
    y[base + 4] = half(b[0] < 0.0f ? 0.0f : b[0]);
    y[base + 5] = half(b[1] < 0.0f ? 0.0f : b[1]);
    y[base + 6] = half(b[2] < 0.0f ? 0.0f : b[2]);
    y[base + 7] = half(b[3] < 0.0f ? 0.0f : b[3]);
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
            return make_response(fenced(vector))
        if "Diagnose the dominant bottleneck" in last:
            return make_response(PLAN)
        title = re.search(r"\*\*(.+?)\*\*", last).group(1)
        return make_response(fenced({"A": naive, "B": wrong, "C": vector}[title]))

    assert broken != naive and wrong != naive
    orch = make_orch(
        tmp_path, route, [], max_rounds=0, evaluate_fn=evaluate,
        expected_identity=lambda **kw: confirmation_expected_document(**kw, probe=True),
        size="paper",
    )
    orch.search.eval_reps = 50
    res = asyncio.run(orch.run())
    statuses = [c["status"] for c in res.candidates]
    assert statuses[:2] == ["compile_error", "ok"]
    assert res.best_speedup is not None and res.confirmed_status == "ok" and res.confirmed_speedup > 0
    assert res.hangs == 0
