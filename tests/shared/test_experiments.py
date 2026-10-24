"""Experiment scripts end-to-end with a scripted LLM and a scripted evaluator: no network, no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import plots, repeatability, run_suite, semantic_tests

from ..support.fakes import FakeClient, make_response
from tasks import get_task

from ..support.orchestrator import PLAN, code, fake_eval, llm_script


def kernel_of(task: str) -> str:
    """@brief 取出任务 fp16/small 规格下的内核名。

    @param task 任务名。
    @return 该任务合约声明的内核函数名。
    """
    return get_task(task, "fp16", "small").contract.kernel_name


def client_factory(speeds):
    """@brief 构造一个工厂，每次调用都给出全新的脚本化客户端。

    初始内核取 ``speeds[0]``，补丁 A 取 ``speeds[1]``，B 与 C 固定为 speed=0.5。

    @param speeds 依次分配给各次运行的 (初始速度, 补丁 A 速度) 对。
    @return 无参调用即可得到新 FakeClient 的工厂函数。
    """
    it = iter(speeds)

    def make():
        s = next(it)
        return FakeClient(llm_script(code(f"speed={s[0]}"), patches={"A": code(f"speed={s[1]}"), "B": code("speed=0.5"), "C": code("speed=0.5")}))

    return make


def test_suite_runs_resumes_summarizes_and_plots(tmp_path):
    """@brief 验证批量实验能跑完、断点续跑不重复评估、生成 Markdown 表与图表。

    @param tmp_path pytest 提供的临时目录，用作套件输出目录。
    """
    calls = []
    evf = fake_eval(calls)
    rows = []

    # run_suite builds its client inside run_one; patch the scripted client in through the module hook
    import experiments.run_suite as rs

    def one(task, dtype, size, run_dir, **kw):
        kw.pop("client", None)
        from experiments.common import run_one as real

        c = lambda m: code(m, kernel_of(task))
        return real(task, dtype, size, run_dir, client=FakeClient(llm_script(c("speed=0.9"), patches={"A": c("speed=1.2"), "B": c("speed=1.1"), "C": c("speed=0.5")})), **kw)

    rs.run_one = one
    rows = rs.run_suite(["relu", "sigmoid"], ["fp16", "bf16"], "small", tmp_path / "suite", max_rounds=1, evaluate_fn=evf, log=lambda s: None, use_tools=False, with_naive=False)
    assert len(rows) == 4 and all(r["confirmed_speedup"] == pytest.approx(1.2) for r in rows)
    n_calls = len(calls)
    again = rs.run_suite(["relu", "sigmoid"], ["fp16", "bf16"], "small", tmp_path / "suite", max_rounds=1, evaluate_fn=evf, log=lambda s: None, use_tools=False, with_naive=False)
    assert len(calls) == n_calls and len(again) == 4  # resumed: nothing re-run
    md = rs.markdown_table(rows)
    assert "relu" in md and "1.200x" in md
    paths = plots.plot_suite(tmp_path / "suite")
    assert all(Path(p).exists() and Path(p).stat().st_size > 5000 for p in paths)


def test_frontier_contains_only_improvements(tmp_path):
    """@brief 验证前沿曲线只包含标记为 improved 的轨迹点。

    @param tmp_path pytest 提供的临时目录（本用例未使用）。
    """
    res = {"trace": [{"cost_usd": 0.1, "best_speedup": 0.9, "improved": True}, {"cost_usd": 0.2, "best_speedup": 0.9, "improved": False}, {"cost_usd": 0.3, "best_speedup": 1.2, "improved": True}]}
    assert plots.frontier(res) == [(0.1, 0.9), (0.3, 1.2)]


def test_repeatability_table(tmp_path):
    """@brief 验证重复性表按每轮速度汇总出三次试验值、最优值与均值并渲染进 Markdown。

    @param tmp_path pytest 提供的临时目录，用作重复性实验输出目录。
    """
    import experiments.repeatability as rp

    seq = iter([1.0, 1.2, 1.1])

    def one(task, dtype, size, run_dir, **kw):
        kw.pop("client", None)
        from experiments.common import run_one as real

        s = next(seq)
        return real(task, dtype, size, run_dir, client=FakeClient(llm_script(code(f"speed={s}"))), **kw)

    rp.run_one = one
    calls = []
    table = rp.run_repeatability(["relu"], "fp16", "small", tmp_path / "rep", trials=3, max_rounds=0, evaluate_fn=fake_eval(calls), log=lambda s: None, use_tools=False)
    v = table["relu"]
    assert v["trials"] == pytest.approx([1.0, 1.2, 1.1]) and v["best"] == pytest.approx(1.2) and v["mean"] == pytest.approx(1.1)
    assert "trial 3" in rp.markdown(table) and "1.200x" in rp.markdown(table)


def test_semantic_runner_reports_first_try_and_repairs(tmp_path):
    """@brief 验证语义测试运行器区分一次即通过与需修复才通过，并统计修复次数。

    @param tmp_path pytest 提供的临时目录，用作语义测试输出目录。
    """
    import experiments.semantic_tests as st

    def one(name, dtype, size, run_dir, **kw):
        kw.pop("client", None)
        from experiments.common import run_one as real

        # sem_int_1 passes first try; sem_int_2 needs one repair
        c = lambda m: code(m, kernel_of(name))
        script = llm_script(c("speed=1.0"), fixes=[]) if name.endswith("_1") else llm_script(c("status=incorrect"), fixes=[c("speed=1.0")])
        return real(name, dtype, size, run_dir, client=FakeClient(script), **kw)

    st.run_one = one
    rows = st.run_semantic(["sem_int_1", "sem_int_2"], tmp_path / "sem", evaluate_fn=fake_eval([]), log=lambda s: None, use_tools=False)
    assert [r["first_try_ok"] for r in rows] == [True, False]
    assert [r["ok_within_repairs"] for r in rows] == [True, True] and rows[1]["attempts_until_ok"] == 2
    assert "First try: 1/2; within repair budget: 2/2" in st.markdown(rows)


def test_p1_runs_name_buildable_tasks_and_the_report_renders(tmp_path):
    """@brief 验证 P1 实验声明的任务都能构建，且报告能汇总失败、退回与合约三份来源。

    @param tmp_path pytest 提供的临时目录，用来放置 summary 与 gemm 输入。
    """
    from experiments import contract_sensitivity as cs
    from experiments import p1_runs

    for name, spec in p1_runs.RUNS.items():
        t = get_task(spec["task"], spec.get("dtype", "fp16"), "paper")
        assert t.contract.kernel_name, name
    rows = [
        {"run": "fa_flash", "task": "flash_attention", "model": "deepseek-flash", "confirmed_speedup": 0.9, "published_speedup": 0.9, "publication": "confirmed", "best_speedup": 0.92, "candidate_us": 400.0, "best_baseline": "mlx", "naive_speedup": 0.1, "vs_triton_msl": 8.0, "rounds": 7, "evaluations": 30, "spent_usd": 1.5, "stop_reason": "budget"},
        {"run": "fa_pro", "task": "flash_attention", "model": "deepseek-v4-pro", "confirmed_speedup": None, "best_speedup": None, "candidate_us": None, "best_baseline": None, "naive_speedup": None, "vs_triton_msl": None, "rounds": 1, "evaluations": 1, "spent_usd": 0.2, "stop_reason": "x"},
    ]
    md = p1_runs.markdown(rows)
    assert "fa_flash" in md and "0.900x" in md and "Total spent: $1.700" in md
    withheld = {**rows[0], "confirmed_speedup": 1.337, "published_speedup": None, "publication": "不确定", "best_speedup": 2.47}
    hidden = p1_runs.markdown([withheld])
    assert "不确定" in hidden and "1.337" not in hidden
    # the contract report reads the three sources it knows
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "summary.json").write_text(json.dumps([{**rows[0], "task": "flash_attention@64x32t256", "run": "fa_64x32t256"}, rows[0]]))
    (tmp_path / "gemm_fp16_paper.json").write_text(json.dumps([
        {"variant": "64x64x32t128", "status": "ok", "candidate_us": 1360.0, "speedup": 0.87, "baseline_us": {"mlx": 1189.0}, "message": ""},
        {"variant": "128x64x32t256", "status": "ok", "candidate_us": 1420.0, "speedup": 0.83, "baseline_us": {"mlx": 1189.0}, "message": ""},
        {"variant": "128x128x32t128", "status": "abi_error", "abi_violations": ["static threadgroup memory too big"], "message": "x"},
    ]))
    text = cs.report(tmp_path, p1=tmp_path / "p1")
    assert "within 0.0% of the best" in text and "64x32t256" in text and "static threadgroup memory" in text
    s = cs.summarize(json.loads((tmp_path / "gemm_fp16_paper.json").read_text()))
    assert s["best_variant"] == "64x64x32t128" and s["default_over_best"] == pytest.approx(1.0)
    slow_default = [{"variant": "64x64x32t128", "status": "ok", "candidate_us": 1500.0, "speedup": 0.8, "baseline_us": {"mlx": 1200.0}}, {"variant": "128x64x32t256", "status": "ok", "candidate_us": 1200.0, "speedup": 1.0, "baseline_us": {"mlx": 1200.0}}]
    assert "limits GEMM" in cs.summarize(slow_default)["verdict"]


def test_cost_report_sums_spend_per_group_and_model(tmp_path):
    """@brief 验证成本报告按分组与模型累加花费，并把读不了的 result.json 列为待人工检查。

    @param tmp_path pytest 提供的临时目录，用来搭建两个分组目录。
    """
    from experiments import cost_report

    for d, name, usd, model in (("a", "r1", 0.25, "m1"), ("a", "r2", 0.5, "m1"), ("b", "r1", 1.0, "m2")):
        (tmp_path / d / name).mkdir(parents=True)
        (tmp_path / d / name / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "model": model, "spent_usd": usd, "budget_usd": 2.0, "evaluations": 3}))
    (tmp_path / "a" / "junk").mkdir()
    (tmp_path / "a" / "junk" / "result.json").write_text("{not json")
    rows, inspections = cost_report.collect([tmp_path / "a", tmp_path / "b"])
    assert len(rows) == 3
    assert len(inspections) == 1 and "needs human inspection" in inspections[0] and "junk" in inspections[0]
    md = cost_report.markdown(rows)
    assert "| 2 | 6 | 0.750 |" in md and "**1.750**" in md
