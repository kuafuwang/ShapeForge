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


def test_real_environment_end_to_end_with_scripted_llm(tmp_path, monkeypatch):
    """@brief 用脚本化模型和真实评测环境跑通一次端到端搜索：先落一个编译失败的变体、再修正，最后给出确认结果且没有挂起。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具：把内核目录重定向到临时目录。
    """
    # A successful confirmation archives the winner into ``kernels/<target>/<task>/.../history.jsonl``
    # (a tracked catalog).  Redirect it so a test run cannot append to the repository's real
    # catalog: that polluted ``test_observe_publications.py``'s fixed real-catalog counts.
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "catalog"))
    # A5 retired the migrated relu assembly.  ``flash_attention`` is a preserved v1 task whose
    # paper-size measurement is dominated by real GPU work (812 us per launch), so the evaluator
    # finds a quiet attempt on the first try and the confirmation gate can actually confirm; the
    # weaker elementwise tasks sit close to ``QUIET_CV`` on this Mac.  ``timing_uncertain`` is a
    # rejection path covered by its own verdict negatives, never accepted by this success node.
    naive = get_task("flash_attention", "fp16", "paper").naive_msl
    broken = naive.replace(
        "float alpha = exp2(m - m_new), pj = exp2(s - m_new);",
        "float alpha = undefined_symbol(m - m_new), pj = exp2(s - m_new);",
    )
    wrong = naive.replace(
        "const float inv = 1.0f / l;",
        "const float inv = 0.5f / l;",
    )
    # A correct candidate with different bytes: IEEE multiplication is exactly commutative, so the
    # operand order is swapped in both reductions; every sum stays in the same order.
    rewritten = naive.replace(
        "s += qv[i] * float(k[base + (ulong)j * D + p * DP + i]);",
        "s += float(k[base + (ulong)j * D + p * DP + i]) * qv[i];",
    ).replace(
        "for (uint i = 0; i < DP; ++i) acc[i] = acc[i] * alpha + pj * float(v[base + (ulong)j * D + p * DP + i]);",
        "for (uint i = 0; i < DP; ++i) acc[i] = alpha * acc[i] + float(v[base + (ulong)j * D + p * DP + i]) * pj;",
    )

    def route(messages, **kw):
        last = messages[-1]["content"]
        if "Translate this Triton kernel" in last:
            return make_response(fenced(broken))
        if "evaluation of your kernel failed" in last:
            if "STATUS: compile_error" in last:
                assert "undefined_symbol" in last and "kernel.metal:" in last  # the real compiler diagnostics reach the model
            else:
                assert "STATUS: incorrect" in last and "out of tolerance" in last
            return make_response(fenced(rewritten))
        if "Diagnose the dominant bottleneck" in last:
            return make_response(PLAN)
        title = re.search(r"\*\*(.+?)\*\*", last).group(1)
        return make_response(fenced({"A": naive, "B": wrong, "C": rewritten}[title]))

    assert broken != naive and wrong != naive and rewritten != naive
    orch = make_orch(
        tmp_path, route, [], max_rounds=0, evaluate_fn=evaluate,
        expected_identity=lambda **kw: confirmation_expected_document(**kw, probe=True),
        task_name="flash_attention", size="paper",
    )
    # The gate itself is untouched; two *measurement settings* make this success node stable on a
    # loaded Mac without weakening a single assertion:
    #
    # * ``eval_reps = 20`` (40 samples per confirmation batch) keeps each timed window short, so an
    #   unrelated load spike cannot turn every attempt of a batch noisy.  Measured over both
    #   attention tasks: 0/20 uncertain measurements, worst cv 0.033 against ``QUIET_CV`` 0.05.
    # * ``confirm_batches = 1`` is the minimum legal confirmation (``max(1, confirm_batches)``).  On
    #   this machine one *evaluation* of any preserved task already varies by ~1% run to run, so
    #   three consecutive batches average 1.5-3% and trip the 2% ``CONFIRM_SPREAD`` disagreement
    #   rule about a third of the time -- a property of the machine, not of the kernel.  The
    #   multi-batch disagreement path keeps its deterministic coverage (`test_policy.py`'s
    #   ``batches_agree`` cases, `test_orchestrator.py`'s disagree case and the read-disk
    #   ``batches_disagree`` case in `test_observe_publications.py`), while this node proves the
    #   *success* path: a real ``ok`` with a positive confirmed speedup, and a batch that was neither
    #   noisy nor failed.
    orch.search.eval_reps = 20
    orch.search.confirm_batches = 1
    res = asyncio.run(orch.run())
    statuses = [c["status"] for c in res.candidates]
    assert statuses[:2] == ["compile_error", "ok"]
    assert res.best_speedup is not None and res.best_speedup > 0
    # The end-to-end search must actually *confirm*: the gate remeasures the kept candidate, the
    # measurement is quiet (flash_attention/paper: candidate and baseline cvs ~0.02-0.03 against
    # ``QUIET_CV`` 0.05) and a positive confirmed speedup is recorded.  ``timing_uncertain`` is a
    # rejection path and is asserted by its own verdict negatives (``test_r5_policy`` /
    # ``test_policy`` / ``test_orchestrator`` / ``test_observe_publications``), not accepted here as
    # an alternative success.
    assert res.confirmed_status == "ok", res.confirmations
    assert res.confirmed_speedup is not None and res.confirmed_speedup > 0, res.confirmations
    assert res.confirmations and all(entry.get("status") == "ok" for entry in res.confirmations), res.confirmations
    assert all(entry.get("failed_batches") == 0 for entry in res.confirmations), res.confirmations
    assert all((entry.get("identity") or {}).get("timing_uncertain") is False for entry in res.confirmations), res.confirmations
    assert res.hangs == 0
