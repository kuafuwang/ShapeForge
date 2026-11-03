"""Full-domain sweep inputs and the publication role. No GPU and no model."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.common import publication_role
from tcenv.backends.metal.domain import (
    PATTERN_COUNT,
    coverage_for_report,
    domain_coverage,
    domain_input,
    pattern_counts,
)
from tcenv.evaluate import EvalReport
from tcenv.task import task_fingerprint
from tcenv.backends.metal.testing import _floor, prepare_case, scaled_error
from tcenv.taskcontract import load_task_contract
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator

from ...support.fakes import FakeClient
from ...support.v1_tasks import V1_TASK, v1_task

ROOT = Path(__file__).resolve().parents[3]


def test_every_pattern_appears_and_the_same_seed_repeats():
    """@brief 验证全域扫描输入包含每一种特殊值模式，同种子可逐位复现，不同种子则不同。

    NaN 自身不相等，因此复现性比较放在整数位模式上。
    """
    small = domain_input(PATTERN_COUNT, 7)
    again = domain_input(PATTERN_COUNT, 7)
    # NaN compares unequal, so the repeat check is on the bit patterns.
    assert torch.equal(small.view(torch.int16), again.view(torch.int16))
    assert int(pattern_counts(small).min()) == 1 and int(pattern_counts(small).max()) == 1
    paper = domain_input(PATTERN_COUNT * 4, 7)
    assert torch.equal(pattern_counts(paper), torch.full((PATTERN_COUNT,), 4))
    assert not torch.equal(domain_input(PATTERN_COUNT, 7).view(torch.int16), domain_input(PATTERN_COUNT, 8).view(torch.int16))


def test_domain_preparation_accepts_the_fp16_maximum_and_rejects_an_all_zero_relu():
    """@brief 验证 domain 用例准备能产出 fp16 最大值，且全零 relu 输出与单个最大值处错误都被判失败。"""
    contract = load_task_contract(ROOT / "tasks/relu", dtype="fp16", size="small")
    n = int(contract.instance.symbols["N"])
    x = domain_input(n, 1)
    ref = contract.reference.execute(contract.instance, {"x": x})["y"]
    finite = ref[torch.isfinite(ref)]
    assert float(finite.max()) == 65504.0
    scale = torch.zeros((), dtype=torch.float64)
    tolerance = contract.instance.tolerances
    zeros = torch.zeros_like(ref)
    err = scaled_error(zeros, ref, scale, tolerance.rtol, tolerance.atol_rel, tolerance.exact, _floor("fp16"))
    assert float(torch.nan_to_num(err, nan=0.0, posinf=1e9).max()) > 1.0
    pos = int(torch.argmax((ref == 65504).to(torch.int64)))
    flipped = ref.clone()
    flipped[pos] = 0
    one = scaled_error(flipped, ref, scale, tolerance.rtol, tolerance.atol_rel, tolerance.exact, _floor("fp16"))
    assert int(torch.argmax(torch.nan_to_num(one, nan=0.0, posinf=1e9))) == pos and float(one[pos]) > 1.0


def test_a_failing_domain_result_is_not_a_domain_sweep():
    """@brief 验证只有全部通过的 domain 用例才记为 domain_sweep 覆盖，失败时降级为 sampling。"""
    # A5 migrated the tasks that declared a domain sweep; the preserved v1 tasks do not, so
    # the report path must stay ``sampling`` for both a failing and a passing domain result.
    task = v1_task()
    bad = [SimpleNamespace(case="domain", ok=False, seed=1, input_sha256="abc")]
    assert coverage_for_report(task, bad, "kernel", {})["kind"] == "sampling"
    good = [SimpleNamespace(case="domain", ok=True, seed=1, input_sha256="abc")]
    tool = {"device": "cpu", "macos": "test", "msl": "4.0", "math_mode": "fast"}
    assert coverage_for_report(task, good, "kernel", tool)["kind"] == "sampling"
    # The domain-sweep record itself is still constructed with one closed shape.
    got = domain_coverage(
        dtype="fp16",
        contract_sha256="contract",
        source_sha256="abc",
        task_fingerprint=task_fingerprint(task),
        toolchain=tool,
        layout={"seed": 1, "copies_per_pattern": 1, "positions": "random permutation"},
        input_sha256="abc",
    )
    assert got["kind"] == "domain_sweep" and got["passed"] is True
    assert got["source_sha256"] and got["task_fingerprint"] == task_fingerprint(task)
    assert got["compare"]["zero_sign"] == "not checked"
    assert "tail paths" in " ".join(got["not_covered"])


def test_a_forked_effective_tolerance_is_refused_and_a_real_change_changes_identity():
    """@brief 有效容差不得与语义实例分叉；真实容差改变必须换身份。

    ``Task.rtol``/``atol_rel``/``exact`` 是 ``SemanticInstance`` 的派生投影，``scaled_error`` 与
    发布判定直接读它们，而 ``task_fingerprint`` v2 只读语义实例（§3.1）。因此：
    同一语义实例的 task 指纹稳定；事后改写派生字段被拒绝（否则同一身份能对应两套评测语义）；
    构造时不一致被拒绝；由语义实例真实产生的容差变化会改变指纹 **且** 改变实际判决。
    """
    first = v1_task("bitdelta_gemm")  # 输出名为 y 的保留题（默认 flash_attention 的输出是 o）
    second = v1_task("bitdelta_gemm")
    assert task_fingerprint(first) == task_fingerprint(second)
    with pytest.raises(AttributeError):
        second.rtol = first.rtol + 1
    with pytest.raises(AttributeError):
        second.exact = True
    with pytest.raises(ValueError):
        dataclasses.replace(second, rtol=first.rtol + 1)

    # The identity authority itself is sealed too: swapping the SemanticInstance after
    # construction would make ``task_fingerprint`` hash one instance while the task keeps the
    # other's effective tolerance (reviewer reproduction), which is exactly the fork.  The same
    # holds for its Specialization and the projected Contract/case list.
    swapped = dataclasses.replace(
        first.semantic_instance,
        tolerances=dataclasses.replace(first.semantic_instance.tolerances, rtol=first.rtol + 1),
    )
    with pytest.raises(AttributeError):
        second.semantic_instance = swapped
    with pytest.raises(AttributeError):
        second.specialization = first.specialization
    with pytest.raises(AttributeError):
        second.contract = first.contract
    with pytest.raises(AttributeError):
        second.cases = ("normal",)
    assert second.semantic_instance.tolerances.rtol == second.rtol
    assert task_fingerprint(second) == task_fingerprint(first)
    # A fresh, internally consistent construction (the atomic-rebuild path) is allowed and
    # carries the new identity together with the new effective tolerance.
    with pytest.raises(ValueError):
        dataclasses.replace(second, semantic_instance=swapped)

    looser = dataclasses.replace(
        second.semantic_instance,
        tolerances=dataclasses.replace(second.semantic_instance.tolerances, rtol=1.0),
    )
    third = dataclasses.replace(
        second,
        semantic_instance=looser,
        rtol=looser.tolerances.rtol,
        atol_rel=looser.tolerances.atol_rel,
        exact=looser.tolerances.exact,
    )
    assert task_fingerprint(first) != task_fingerprint(third)

    # The two identities are not interchangeable: a 50% high output fails the registered
    # tolerance and passes the loosened one.
    data = prepare_case(first, "normal", 1)
    off = {name: value * 1.5 for name, value in data.ref.items()}

    def passes(task) -> bool:
        err = scaled_error(off["y"], data.ref["y"], data.scale["y"], task.rtol, task.atol_rel, task.exact, _floor("fp16"))
        return bool(torch.nan_to_num(err, nan=0.0, posinf=1e9).max() <= 1.0)

    assert passes(first) is False
    assert passes(third) is True


def test_a_failed_domain_sweep_writes_a_report_and_no_result(tmp_path):
    """@brief 验证前置 domain 扫描失败时只写出 domain_report.json，不产出 result.json。

    @param tmp_path pytest 提供的临时目录，用作 run_dir 的父目录。
    """
    seen = {}

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        seen["mode"] = mode
        seen["cases"] = kw.get("cases")
        return EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, status="incorrect", stage="correctness", message="bad")

    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=1.0, branches=1, max_rounds=1, eval_reps=3, stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False)
    orch = Orchestrator(V1_TASK, "fp16", "small", cfg, search, run_dir=tmp_path / "run", client=FakeClient([]), evaluate_fn=ev, log=lambda _s: None)
    # A5 retired the migrated tasks that declared a domain sweep.  Trigger the sweep branch on
    # the preserved v1 task with an explicit test-only declaration; the evaluator is fake and
    # the point is the report/no-result contract, not the task's mathematics.
    orch.task = SimpleNamespace(evaluation_domain={"domain_case": True})
    win = SimpleNamespace(label="init", source="kernel void k() {}\n")
    asyncio.run(orch._domain_sweep(win))
    doc = json.loads((tmp_path / "run" / "domain_report.json").read_text())
    # 本地评测适配器把请求里的 tuple 转成 list 再交给 evaluate_fn（与远程载荷同一形状），所以这里按 list 断言。
    assert seen == {"mode": "quick", "cases": ["domain"]}
    assert doc["status"] == "incorrect"
    assert not (tmp_path / "run" / "result.json").exists()


def test_primary_needs_the_gain_and_a_bound_sweep():
    """@brief 验证只有提速达标且绑定源码的完整 domain_sweep 才被定为 primary，否则降为 fallback。"""
    cov = {"kind": "domain_sweep", "passed": True, "source_sha256": "abc", "task_fingerprint": "fp"}
    confirmed = {"publication": "confirmed", "speedup": 1.2}
    assert publication_role(confirmed, {"coverage": cov}, source_sha256="abc", task_fingerprint="fp", recorded_fingerprint="fp") == "primary"
    assert publication_role({"publication": "confirmed", "speedup": 0.97}, {"coverage": cov}, source_sha256="abc", task_fingerprint="fp", recorded_fingerprint="fp") == "fallback"
    assert publication_role({"publication": "in_progress"}, None, source_sha256=None, task_fingerprint=None, recorded_fingerprint=None) is None
    assert publication_role({"publication": "旧版未绑定源码"}, None, source_sha256=None, task_fingerprint=None, recorded_fingerprint=None) == "fallback"
    other = {**cov, "source_sha256": "nope"}
    assert publication_role(confirmed, {"coverage": other}, source_sha256="abc", task_fingerprint="fp", recorded_fingerprint="fp") == "fallback"


def test_the_publication_meta_is_a_read_only_view_of_the_opspec():
    """@brief H-17：``Task.meta`` 是 OpSpec 的只读视图，不能就地改、也不能换成别的语义键。"""
    import dataclasses
    from types import MappingProxyType

    import pytest

    from tasks import get_task

    task = v1_task()
    assert isinstance(task.meta, MappingProxyType)
    with pytest.raises(TypeError):
        task.meta["domain_case"] = True
    with pytest.raises(AttributeError):
        task.meta = {}
    # A diagnostic key is allowed, a semantic fact must come from the OpSpec.
    assert "baseline_note" not in task.meta
    with pytest.raises(ValueError, match="unregistered keys"):
        dataclasses.replace(task, meta={"source": "second-copy"})
    with pytest.raises(ValueError, match="semantic keys must be the SemanticInstance"):
        dataclasses.replace(task, meta={"domain_case": True})
    # The semantic keys must equal the instance's evaluation domain (no second copy).
    assert {key: task.meta[key] for key in ("domain_case", "ieee_base") if key in task.meta} == {
        key: value for key, value in task.semantic_instance.evaluation_meta().items() if key in ("domain_case", "ieee_base")
    }
    migrated = get_task("bitdelta_gemm", "fp16", "small")
    assert dict(migrated.meta) == {"ieee_base": "random", "baseline_note": dict(migrated.meta)["baseline_note"]}
