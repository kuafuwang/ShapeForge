"""Full-domain sweep inputs and the publication role. No GPU and no model."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import torch

from experiments.common import publication_role
from tcenv.backends.metal.domain import PATTERN_COUNT, coverage_for_report, domain_input, pattern_counts
from tcenv.evaluate import EvalReport
from tcenv.task import task_fingerprint
from tcenv.backends.metal.testing import _floor, prepare_case, scaled_error
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from tasks.relu.task import build as build_relu

from .fakes import FakeClient


def test_every_pattern_appears_and_the_same_seed_repeats():
    small = domain_input(PATTERN_COUNT, 7)
    again = domain_input(PATTERN_COUNT, 7)
    # NaN compares unequal, so the repeat check is on the bit patterns.
    assert torch.equal(small.view(torch.int16), again.view(torch.int16))
    assert int(pattern_counts(small).min()) == 1 and int(pattern_counts(small).max()) == 1
    paper = domain_input(PATTERN_COUNT * 4, 7)
    assert torch.equal(pattern_counts(paper), torch.full((PATTERN_COUNT,), 4))
    assert not torch.equal(domain_input(PATTERN_COUNT, 7).view(torch.int16), domain_input(PATTERN_COUNT, 8).view(torch.int16))


def test_domain_preparation_accepts_the_fp16_maximum_and_rejects_an_all_zero_relu():
    task = build_relu("fp16", "small")
    data = prepare_case(task, "domain", 1)
    finite = data.ref["y"][torch.isfinite(data.ref["y"])]
    assert float(finite.max()) == 65504.0
    zeros = torch.zeros_like(data.ref["y"])
    err = scaled_error(zeros, data.ref["y"], data.scale["y"], task.rtol, task.atol_rel, task.exact, _floor("fp16"))
    assert float(torch.nan_to_num(err, nan=0.0, posinf=1e9).max()) > 1.0
    pos = int(torch.argmax((data.ref["y"] == 65504).to(torch.int64)))
    flipped = data.ref["y"].clone()
    flipped[pos] = 0
    one = scaled_error(flipped, data.ref["y"], data.scale["y"], task.rtol, task.atol_rel, task.exact, _floor("fp16"))
    assert int(torch.argmax(torch.nan_to_num(one, nan=0.0, posinf=1e9))) == pos and float(one[pos]) > 1.0


def test_a_failing_domain_result_is_not_a_domain_sweep():
    task = build_relu("fp16", "small")
    bad = [SimpleNamespace(case="domain", ok=False, seed=1, input_sha256="abc")]
    assert coverage_for_report(task, bad, "kernel", {})["kind"] == "sampling"
    good = [SimpleNamespace(case="domain", ok=True, seed=1, input_sha256="abc")]
    tool = {"device": "cpu", "macos": "test", "msl": "4.0", "math_mode": "fast"}
    got = coverage_for_report(task, good, "kernel", tool)
    assert got["kind"] == "domain_sweep" and got["passed"] is True
    assert got["source_sha256"] and got["task_fingerprint"] == task_fingerprint(task)
    assert got["compare"]["zero_sign"] == "not checked"
    assert "tail paths" in " ".join(got["not_covered"])


def test_changing_the_tolerance_changes_the_task_fingerprint():
    first = build_relu("fp16", "small")
    second = build_relu("fp16", "small")
    assert task_fingerprint(first) == task_fingerprint(second)
    second.rtol = first.rtol + 1
    assert task_fingerprint(first) != task_fingerprint(second)


def test_a_failed_domain_sweep_writes_a_report_and_no_result(tmp_path):
    seen = {}

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        seen["mode"] = mode
        seen["cases"] = kw.get("cases")
        return EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, status="incorrect", stage="correctness", message="bad")

    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=1.0, branches=1, max_rounds=1, eval_reps=3, stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False)
    orch = Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / "run", client=FakeClient([]), evaluate_fn=ev, log=lambda _s: None)
    win = SimpleNamespace(label="init", source="kernel void k() {}\n")
    asyncio.run(orch._domain_sweep(win))
    doc = json.loads((tmp_path / "run" / "domain_report.json").read_text())
    assert seen == {"mode": "quick", "cases": ("domain",)}
    assert doc["status"] == "incorrect"
    assert not (tmp_path / "run" / "result.json").exists()


def test_primary_needs_the_gain_and_a_bound_sweep():
    cov = {"kind": "domain_sweep", "passed": True, "source_sha256": "abc", "task_fingerprint": "fp"}
    confirmed = {"publication": "confirmed", "speedup": 1.2}
    assert publication_role(confirmed, {"coverage": cov}, source_sha256="abc", task_fingerprint="fp", recorded_fingerprint="fp") == "primary"
    assert publication_role({"publication": "confirmed", "speedup": 0.97}, {"coverage": cov}, source_sha256="abc", task_fingerprint="fp", recorded_fingerprint="fp") == "fallback"
    assert publication_role({"publication": "in_progress"}, None, source_sha256=None, task_fingerprint=None, recorded_fingerprint=None) is None
    assert publication_role({"publication": "旧版未绑定源码"}, None, source_sha256=None, task_fingerprint=None, recorded_fingerprint=None) == "fallback"
    other = {**cov, "source_sha256": "nope"}
    assert publication_role(confirmed, {"coverage": other}, source_sha256="abc", task_fingerprint="fp", recorded_fingerprint="fp") == "fallback"
