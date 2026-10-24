"""Budget alerts of the dashboard (over budget / budget exhausted / near budget) and the paper-level budget defaults. No GPU, no model calls."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from observe import query as scan
from observe.budget import NEAR_FRACTION, budget_status, render_banner_html, summarize


# ------------------------------------------------------------------------------------------------ the classification


def test_over_budget_shows_the_overrun_in_dollars_and_percent():
    """@brief 验证超预算时同时给出 over 与 exhausted 两个标记，并算出超支金额、超支百分比与用量百分比。"""
    st = budget_status(3.123, 3.0, "budget_exhausted", False)
    assert st["level"] == "over" and st["label"] == "OVER BUDGET" and st["flagged"]
    assert st["flags"] == ["over", "exhausted"]  # both apply; over is the worse one
    assert st["over_usd"] == pytest.approx(0.123) and st["over_pct"] == pytest.approx(4.1) and st["used_pct"] == pytest.approx(104.1)
    assert "over by $0.123 (+4.1%)" in st["text"] and "$3.123 / $3.00" in st["text"]


def test_over_budget_without_a_budget_stop_reason():
    """@brief 验证没有预算停止原因、但花费仍超上限时只标 over。"""
    st = budget_status(3.036, 3.0, "converged (4 rounds without improvement)", False)
    assert st["level"] == "over" and st["flags"] == ["over"]


def test_exhausted_under_the_cap_is_flagged_because_money_ended_the_search():
    """@brief 验证花费没超上限、但停止原因是预算耗尽时标 exhausted，超支为 0。"""
    st = budget_status(2.97, 3.0, "budget_exhausted", False)
    assert st["level"] == "exhausted" and st["label"] == "BUDGET EXHAUSTED" and st["flags"] == ["exhausted"]
    assert st["over_usd"] == 0.0 and st["over_pct"] == 0.0 and st["used_pct"] == pytest.approx(99.0)


def test_exactly_the_budget_is_not_over_but_a_budget_stop_is_exhausted():
    """@brief 验证花到刚好等于预算不算超支（含浮点噪声），但预算停止仍算 exhausted。"""
    assert budget_status(3.0, 3.0, "converged", False)["level"] == "ok"
    assert budget_status(3.0, 3.0, "budget_exhausted", False)["level"] == "exhausted"
    assert budget_status(3.0000000001, 3.0, "converged", False)["level"] == "ok"  # float noise


def test_near_budget_only_applies_to_a_run_that_is_still_running():
    """@brief 验证 near 只对仍在运行的运行生效：阈值是 0.8，已结束在 90% 的运行不告警。"""
    assert NEAR_FRACTION == 0.8
    assert budget_status(2.4, 3.0, "", True)["level"] == "near"
    assert budget_status(2.39, 3.0, "", True)["level"] == "ok"
    near = budget_status(2.7, 3.0, "", True)
    assert near["label"] == "NEAR BUDGET" and near["flagged"] and near["used_pct"] == pytest.approx(90.0)
    assert budget_status(2.7, 3.0, "converged", False)["level"] == "ok"  # finished at 90%: nothing to warn about


def test_a_running_run_that_is_already_over_is_over_not_near():
    """@brief 验证已超支的运行即使在跑也判 over，不再降级成 near。"""
    st = budget_status(3.01, 3.0, "", True)
    assert st["level"] == "over" and st["flags"] == ["over"]


@pytest.mark.parametrize("spent,budget", [(None, None), (1.0, None), (1.0, 0), (1.0, -2.0), (None, 3.0), ("abc", 3.0), (1.0, float("nan")), (True, 3.0), (float("inf"), 3.0)])
def test_missing_or_unusable_numbers_never_raise_and_are_not_flagged(spent, budget):
    """@brief 验证缺失或不可用的花费、预算数字不抛异常，一律判 unknown 且不打标记。

    @param spent 花费金额，可能是 None、字符串、bool、nan 或 inf。
    @param budget 预算金额，可能是 None、0、负数或 nan。
    """
    st = budget_status(spent, budget, None, True)
    assert st["level"] == "unknown" and not st["flagged"] and st["flags"] == [] and st["label"] == ""
    assert st["over_usd"] is None and st["over_pct"] is None and st["used_pct"] is None


def test_a_budget_stop_is_flagged_even_when_the_numbers_are_missing():
    """@brief 验证数字缺失时预算停止仍被标记，字符串形式的金额也能解析后归类。"""
    st = budget_status(None, None, "budget_exhausted", False)
    assert st["level"] == "exhausted" and st["flagged"]
    assert budget_status("3.2", "3.0", None, False)["level"] == "over"  # numbers stored as strings still work


def test_old_results_without_a_stop_reason_are_classified_from_the_money_alone():
    """@brief 验证没有停止原因的旧结果只按金额判断是否超支。"""
    assert budget_status(0.2, 0.25, None, False)["level"] == "ok"
    assert budget_status(0.3, 0.25, None, False)["level"] == "over"


# ------------------------------------------------------------------------------------------------ the banner data


def _row(run_id, spent, budget, stop="", status="done", task="gemm"):
    """@brief 造一条历史行，其中的 ``budget`` 字段由 ``budget_status`` 现算。

    @param run_id 运行 id。
    @param spent 花费美元。
    @param budget 预算美元；None 表示该运行没有预算。
    @param stop 停止原因。
    @param status 运行状态；``running`` 时按在跑处理。
    @param task 题目名。
    @return 历史行字典。
    """
    return {"id": run_id, "task": task, "dtype": "fp16", "status": status, "spent_usd": spent, "budget_usd": budget, "stop_reason": stop,
            "budget": budget_status(spent, budget, stop, status == "running")}


def test_summary_counts_names_and_totals_the_flagged_runs_worst_first():
    """@brief 验证汇总按严重程度排序，并统计各状态数量与花费、预算、超支合计。"""
    rows = [
        _row("a/ok", 1.0, 3.0, "converged"),
        _row("a/near", 2.6, 3.0, "", "running"),
        _row("a/exh", 2.97, 3.0, "budget_exhausted"),
        _row("a/over_small", 3.03, 3.0, "converged"),
        _row("a/over_big", 3.3, 3.0, "budget_exhausted"),
        _row("a/nobudget", 0.5, None, "converged"),
    ]
    rep = summarize(rows)
    assert (rep["n_runs"], rep["n_over"], rep["n_exhausted"], rep["n_near"], rep["n_flagged"]) == (6, 2, 2, 1, 4)
    assert [f["id"] for f in rep["flagged"]] == ["a/over_big", "a/over_small", "a/exh", "a/near"]
    assert rep["level"] == "over"
    assert rep["total_spent_usd"] == pytest.approx(1.0 + 2.6 + 2.97 + 3.03 + 3.3 + 0.5)
    assert rep["total_budget_usd"] == pytest.approx(15.0) and rep["total_over_usd"] == pytest.approx(0.33)


def test_summary_of_clean_runs_is_ok():
    """@brief 验证没有异常运行时汇总为 ok、横幅显示预算正常，空输入也不炸。"""
    rep = summarize([_row("a", 1.0, 3.0, "converged"), _row("b", 0.1, 3.0, "converged")])
    assert rep["n_flagged"] == 0 and rep["level"] == "ok" and rep["flagged"] == []
    assert "预算正常" in render_banner_html(rep)
    assert summarize([])["n_runs"] == 0


def test_banner_html_names_the_runs_and_escapes_them():
    """@brief 验证横幅 HTML 列出运行并转义其中的标签与引号，链接参数做百分号编码。"""
    rows = [_row('p1/<b>"x"</b>', 3.1, 3.0, "budget_exhausted", task="<i>")]
    html = render_banner_html(summarize(rows))
    assert "OVER BUDGET" in html and "BUDGET EXHAUSTED" in html and "BUDGET ALERT" in html
    assert "<b>\"x\"</b>" not in html and "&lt;b&gt;" in html and "&lt;i&gt;" in html
    assert "/?run=p1%2F%3Cb%3E%22x%22%3C%2Fb%3E" in html


# ------------------------------------------------------------------------------------------------ from run directories


def _result(root: Path, name: str, *, spent, budget, stop="max_rounds", task="gemm", **extra):
    """@brief 在运行根下写一份 ``result.json``，可按需省略花费或预算字段。

    @param root 运行根目录。
    @param name 运行目录名。
    @param spent 花费美元；None 表示不写该字段。
    @param budget 预算美元；None 表示不写该字段。
    @param stop 停止原因。
    @param task 题目名。
    @param extra 追加或覆盖的顶层字段。
    @return 建好的运行目录。
    """
    directory = root / name
    directory.mkdir(parents=True)
    doc = {"task": task, "dtype": "fp16", "stop_reason": stop, "confirmed_speedup": 1.1, "trace": [], **extra}
    if spent is not None:
        doc["spent_usd"] = spent
    if budget is not None:
        doc["budget_usd"] = budget
    (directory / "result.json").write_text(json.dumps(doc))
    return directory


def _live(root: Path, name: str, *, spent, budget):
    """@brief 写一份 ``state.json``，让这份运行被当成当前进程正在跑的活运行。

    @param root 运行根目录。
    @param name 运行目录名。
    @param spent 花费美元。
    @param budget 预算美元。
    @return 建好的运行目录。
    """
    import os

    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "pid": os.getpid(), "updated_at": time.time(), "started_at": time.time(), "phase": "evaluate", "task": "conv2d", "dtype": "fp16",
        "spent_usd": spent, "budget_usd": budget, "trace": [],
    }))
    return directory


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """@brief 造一组形状贴近真实运行的目录：超预算、预算耗尽、未超预算、旧结果与在跑的五种都有。

    @param tmp_path 临时目录，运行根放在这里。
    @param monkeypatch 把 ``AI_LOWERLING_RUNS_DIR`` 指到临时目录。
    @return 运行根目录。
    """
    runs = tmp_path / "runs"  # the same directory the ``dashboard`` fixture serves
    runs.mkdir(exist_ok=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    # the shapes of real runs: p1/fa_flash (3.123 of 3.00, budget_exhausted), p1/fa_pro (3.036 of 3.00), a run that stopped under its cap
    _result(runs / "p1", "fa_flash", spent=3.1229796840000015, budget=3.0, stop="budget_exhausted", task="flash_attention")
    _result(runs / "p1", "fa_pro", spent=3.036134728000001, budget=3.0, stop="budget_exhausted", task="flash_attention")
    _result(runs / "suite", "relu_fp16", spent=0.2376, budget=0.5, stop="converged (4 rounds without improvement)", task="relu")
    _result(runs / "suite", "rope_fp16", spent=0.79, budget=0.8, stop="budget_exhausted", task="rope")
    _result(runs / "suite", "old_run", spent=0.3, budget=None, stop="max_rounds", task="sigmoid")  # an older run without a budget field
    _live(runs / "p1", "conv2d_paper", spent=13.0, budget=15.0)
    return runs


def test_history_rows_carry_the_budget_status(runs_dir):
    """@brief 验证历史里每一行都带上预算判定，覆盖超支、耗尽、正常、缺字段与在跑 near 五种情况。

    @param runs_dir 由 fixture 造好的运行根目录。
    """
    by_id = {row["id"]: row for row in scan.history()}
    fa = by_id["p1/fa_flash"]
    assert fa["budget_usd"] == 3.0 and fa["budget"]["level"] == "over" and fa["budget"]["over_usd"] == pytest.approx(0.1229796840000015)
    assert by_id["p1/fa_pro"]["budget"]["over_pct"] == pytest.approx(1.2)
    assert by_id["suite/relu_fp16"]["budget"]["level"] == "ok"
    assert by_id["suite/rope_fp16"]["budget"]["level"] == "exhausted"  # stopped by money, ended under the cap
    assert by_id["suite/old_run"]["budget"]["level"] == "unknown" and by_id["suite/old_run"]["budget_usd"] is None
    live = by_id["p1/conv2d_paper"]
    assert live["status"] == "running" and live["budget"]["level"] == "near" and live["budget"]["used_pct"] == pytest.approx(86.7)


def test_a_recorded_reset_clears_the_budget_alert_before_the_next_process_starts(runs_dir):
    """@brief 验证已记录的预算重置在下一个进程启动前就清掉告警，而未重置的运行仍报旧告警。

    @param runs_dir 运行根目录。
    """
    run = runs_dir / "p1" / "fa_flash"
    (run / "budget_events.json").write_text(json.dumps([
        {"kind": "requested", "old_budget_usd": 3.0, "new_budget_usd": 18.122979684, "note": "reset: add $15 and lift the round limit and patience"},
    ]))
    rope = runs_dir / "suite" / "rope_fp16"
    (rope / "budget_events.json").write_text(json.dumps([
        {"kind": "requested", "old_budget_usd": 0.8, "new_budget_usd": 15.79, "note": "reset"},
    ]))
    by_id = {row["id"]: row for row in scan.history()}
    fa = by_id["p1/fa_flash"]
    assert fa["budget_usd"] == pytest.approx(18.122979684) and fa["budget"]["level"] == "ok"
    assert fa["stop_reason"] == "budget_exhausted"  # the old result stays; only the alert uses the granted budget
    assert by_id["suite/rope_fp16"]["budget"]["level"] == "ok"
    assert by_id["p1/fa_pro"]["budget"]["level"] == "over"  # no reset recorded, the old alert remains
    rep = scan.budget_report()
    assert "p1/fa_flash" not in [item["id"] for item in rep["flagged"]]
    assert "suite/rope_fp16" not in [item["id"] for item in rep["flagged"]]


def test_a_finished_undershoot_keeps_the_exhausted_flag(runs_dir):
    """@brief 验证升预算后仍因预算停下、最终花费未到上限时，exhausted 标记保留。

    @param runs_dir 运行根目录。
    """
    run = runs_dir / "suite" / "rope_fp16"
    (run / "result.json").write_text(json.dumps({
        "task": "rope", "dtype": "fp16", "stop_reason": "budget_exhausted",
        "spent_usd": 14.9, "budget_usd": 15.0, "best_label": "init", "trace": [],
    }))
    (run / "budget_events.json").write_text(json.dumps([
        {"kind": "resume", "old_budget_usd": 0.8, "new_budget_usd": 15.0},
    ]))
    row = next(item for item in scan.history() if item["id"] == "suite/rope_fp16")
    assert row["budget_usd"] == 15.0 and row["budget"]["level"] == "exhausted"


def test_a_dead_resume_lock_does_not_keep_an_unapplied_request_or_block_the_button(runs_dir):
    """@brief 验证残留的 ``resume.lock`` 不会让未生效的预算请求生效，也不会让续跑按钮失效。

    @param runs_dir 运行根目录。
    """
    run = runs_dir / "p1" / "fa_flash"
    (run / "budget_events.json").write_text(json.dumps([
        {"kind": "requested", "old_budget_usd": 3.0, "new_budget_usd": 18.0, "note": "reset"},
    ]))
    (run / "result.json").write_text(json.dumps({
        **json.loads((run / "result.json").read_text()),
        "best_label": "init",
    }))
    (run / "resume.lock").write_text(json.dumps({"pid": 2**22 + 9, "starting": True}))
    row = next(item for item in scan.history() if item["id"] == "p1/fa_flash")
    assert row["budget_usd"] == 3.0 and row["budget"]["level"] == "over"
    assert row["control"]["action"] == "resume"


def test_a_later_budget_cut_replaces_the_older_grant(runs_dir):
    """@brief 验证后一次预算下调覆盖先前授予的额度。

    @param runs_dir 运行根目录。
    """
    run = runs_dir / "p1" / "fa_pro"
    (run / "budget_events.json").write_text(json.dumps([
        {"kind": "applied", "old_budget_usd": 3.0, "new_budget_usd": 30.0},
        {"kind": "applied", "old_budget_usd": 30.0, "new_budget_usd": 15.0},
    ]))
    row = next(item for item in scan.history() if item["id"] == "p1/fa_pro")
    assert row["budget_usd"] == 15.0


def test_budget_report_names_the_runs_that_exceeded(runs_dir):
    """@brief 验证预算报告列出所有越界运行，并合计超支金额。

    @param runs_dir 运行根目录。
    """
    rep = scan.budget_report()
    assert rep["n_runs"] == 6 and rep["n_over"] == 2 and rep["n_exhausted"] == 3 and rep["n_near"] == 1
    assert [f["id"] for f in rep["flagged"]] == ["p1/fa_flash", "p1/fa_pro", "suite/rope_fp16", "p1/conv2d_paper"]
    assert rep["total_over_usd"] == pytest.approx(0.1229796840000015 + 0.036134728000001, abs=1e-6)


def test_spend_and_budget_fall_back_to_state_and_to_ledger_json(runs_dir):
    """@brief 验证 ``result.json`` 缺金额时依次回落到 ``state.json`` 的 ledger 与 ``ledger.json`` 的两种形状。

    @param runs_dir 运行根目录。
    """
    d = _result(runs_dir / "x", "no_money_in_result", spent=None, budget=None, stop="budget_exhausted")
    (d / "state.json").write_text(json.dumps({"task": "gemm", "dtype": "fp16", "phase": "done", "ledger": {"total_usd": 1.5, "budget_usd": 1.0}}))
    e = _result(runs_dir / "x", "only_ledger_file", spent=None, budget=None, stop="converged")
    (e / "ledger.json").write_text(json.dumps({"summary": {"total_usd": 2.2, "budget_usd": 2.0}}))
    f = _result(runs_dir / "x", "flat_ledger_file", spent=None, budget=None, stop="converged")
    (f / "ledger.json").write_text(json.dumps({"total_usd": 0.5, "budget_usd": 2.0}))
    by_id = {row["id"]: row for row in scan.history()}
    assert by_id["x/no_money_in_result"]["budget"]["level"] == "over" and by_id["x/no_money_in_result"]["spent_usd"] == 1.5
    assert by_id["x/only_ledger_file"]["budget"]["over_usd"] == pytest.approx(0.2) and by_id["x/only_ledger_file"]["budget_usd"] == 2.0
    assert by_id["x/flat_ledger_file"]["budget"]["level"] == "ok"


def test_run_detail_and_frame_include_the_budget(runs_dir):
    """@brief 验证运行详情与整帧数据都带上预算判定和计数。

    @param runs_dir 运行根目录。
    """
    detail = scan.run_detail("p1/fa_flash")
    assert detail["budget"]["level"] == "over" and detail["budget"]["label"] == "OVER BUDGET"
    frame = scan.frame("p1/fa_pro")
    assert frame["budget"]["n_over"] == 2 and frame["selected"] == "p1/fa_pro" and frame["run"]["budget"]["level"] == "over"


# ------------------------------------------------------------------------------------------------ the served page


def _get(port: int, path: str) -> str:
    """@brief 向被服务的 dashboard 发一个 GET 请求。

    @param port dashboard 监听端口。
    @param path 请求路径。
    @return 响应正文。
    """
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
        return resp.read().decode("utf-8")


def test_the_page_has_the_banner_with_the_over_budget_runs_at_first_paint(dashboard, runs_dir):  # noqa: F811
    """@brief 验证首屏 HTML 里横幅已渲染（占位注释被替掉），列出超支运行、金额与计数，并保留滚动样式。

    @param dashboard 起在临时端口上的 dashboard fixture。
    @param runs_dir 运行根目录。
    """
    port, _ = dashboard
    page = _get(port, "/")
    assert 'id="budget-banner"' in page and "<!--BUDGET_BANNER-->" not in page
    assert "BUDGET ALERT" in page and "2 OVER BUDGET" in page and "3 BUDGET EXHAUSTED" in page and "1 NEAR BUDGET" in page
    for run_id in ("p1/fa_flash", "p1/fa_pro", "suite/rope_fp16", "p1/conv2d_paper"):
        assert f'data-run="{run_id}"' in page
    assert "over by $0.123 (+4.1%)" in page and "over by $0.036 (+1.2%)" in page and "$3.123 / $3.00" in page
    assert 'class="chip over"' in page and 'class="chip exhausted"' in page and 'class="chip near"' in page
    assert "总花费" in page
    assert ".bb-list { max-height: 12rem; overflow-x: hidden; overflow-y: auto;" in page


def test_a_named_run_without_its_own_model_does_not_require_a_null_model():
    """@brief 验证命名运行没有独立模型时不产生 model 字段，并按运行查表补上默认模型。

    顺带核对页面静态脚本仍引用那些关键 id、文案与交互入口。
    """
    from experiments.p1_runs import _arm_delta

    assert "model" not in _arm_delta("fa_flash", "fp16", None)
    assert _arm_delta("fa_pro", "fp16", None)["model"] == "deepseek-v4-pro"
    text = (Path(__file__).resolve().parents[2] / "observe" / "static" / "index.html").read_text(encoding="utf-8")
    for needle in (
        "confirmed_point", "handoff", "budget_events", "candidates",
        "feature_grid", "plan_outcomes",
        "filter-task", "filter-dtype", "filter-model", "filter-status",
        "runs-prev", "runs-next", "by_operator", "by_month",
        "budget-amount", "budget-save", 'method: "POST"', "X-Observe-Token", 'classList.add("open")',
        "function newestFirst", "newestFirst(doc.candidates)", "newestFirst(doc.budget_events)", "newestFirst(handoff.rounds)", "newestFirst(doc.plan_outcomes)", "newestFirst(doc.by_month)", "newestFirst(doc.feature_grid)",
        "安检", "阶段", "代码", 'class="excerpt"', "--blue: #0a66c2", "@keyframes breathe", 'class="breathe"',
        'id="workspace"', 'id="side-toggle"', 'id="runs-toggle"', 'id="detail-toggle"', 'id="detail-task"', 'id="detail-status"', "function paintDetailHead", 'id="view-experiments"', 'id="view-compare"', 'id="view-detail"',
        "<th>目录</th><th>种类</th><th>操作</th>",
        "runs-table-scroll", "detail-scroll", "position: sticky", 'id="budget-toggle"', 'id="term-open"', "validated_win", "untested_impl_fail",
        "已截断：",
        "delete-reason", "/api/records/delete", "删除记录只追加",
        'class="note-list"', ".note-list { max-height: 12rem; overflow: auto;",
        "正常工具调用，不是异常。", "草稿交给本机编译并对答案", "在本机查阅 MSL 资料",
        "hmz 调用 pi 写内核，不是异常", 'id="pi-live"', "/api/v2/pi/",
        "function writerMode", "doc.kernel_writer", "这一场由搜索直接调用模型",
        'id="turns-live"', "/api/v2/turns/", "更早", "function openTurnKeys", "data-turn",
        'id="pi-live"', "/api/v2/pi/",
        'id="harness-badge"', 'id="harness-modal"', "function harnessMarkup", 'id="harness-raw"', "taic harness status",
    ):
        assert needle in text, needle
    # The header entry renders its own panel now; a browser tab of raw JSON was the old behaviour.
    assert 'window.open("/api/v2/harness/' not in text


def test_the_page_script_renders_badges_filters_and_sorting(dashboard):  # noqa: F811
    """@brief 验证页面脚本含徽章、筛选与排序三块渲染逻辑。

    @param dashboard dashboard fixture。
    """
    port, _ = dashboard
    page = _get(port, "/")
    for needle in ("OVER BUDGET", "BUDGET EXHAUSTED", "NEAR BUDGET", "renderBanner", "FILTERS", "SORTS", "badge ", ".badge.over", ".badge.exhausted", ".badge.near", "hist-tools"):
        assert needle in page, needle


def test_the_budget_api_and_the_history_api_expose_the_alerts(dashboard, runs_dir):  # noqa: F811
    """@brief 验证预算、历史与单运行三个接口都暴露出同一套告警。

    @param dashboard dashboard fixture。
    @param runs_dir 运行根目录。
    """
    port, _ = dashboard
    rep = json.loads(_get(port, "/api/budget"))
    assert rep["n_over"] == 2 and {f["id"] for f in rep["flagged"] if f["level"] == "over"} == {"p1/fa_flash", "p1/fa_pro"}
    rows = {row["id"]: row for row in json.loads(_get(port, "/api/history"))}
    assert rows["p1/fa_flash"]["budget"]["label"] == "OVER BUDGET" and rows["suite/rope_fp16"]["budget"]["label"] == "BUDGET EXHAUSTED"
    assert rows["p1/conv2d_paper"]["budget"]["label"] == "NEAR BUDGET"
    one = json.loads(_get(port, "/api/runs/p1/fa_pro"))
    assert one["budget"]["level"] == "over" and one["budget"]["over_usd"] == pytest.approx(0.036134728000001)


def test_a_damaged_run_directory_does_not_break_the_page(dashboard):  # noqa: F811
    """@brief 验证运行目录里出现坏 JSON 时页面仍能渲染横幅。

    @param dashboard fixture 返回的 ``(端口, 运行根目录)``。
    """
    port, runs = dashboard
    (runs / "bad").mkdir()
    (runs / "bad" / "result.json").write_text("{not json")
    page = _get(port, "/")
    assert 'id="budget-banner"' in page and "预算正常" in page


# ------------------------------------------------------------------------------------------------ the paper-level budget defaults


def test_the_default_budget_is_the_papers_15_usd_and_the_round_limit_does_not_end_a_run_first(monkeypatch):
    """@brief 验证默认预算为论文的 15 美元、轮数上限先于预算放开，并核对 base 预设与命令行覆盖。

    @param monkeypatch 用于设置假凭据并清掉预设环境变量。
    """
    from taic.llm import PAPER_BUDGET_USD, SearchConfig, load_config

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    monkeypatch.delenv("TAIC_PRESET", raising=False)
    assert PAPER_BUDGET_USD == 15.0 and SearchConfig().budget_usd == 15.0
    _, s = load_config()
    assert s.budget_usd == 15.0 and s.max_rounds >= 40
    assert s.patience == 12  # the paper preset. `--preset base` is taic.toml, whose patience is 6.
    assert load_config(preset="base")[1].patience == 6
    _, s = load_config(budget_usd=3.0)
    assert s.budget_usd == 3.0  # the command line can still override it


def test_the_base_preset_is_taic_toml_alone_and_taic_toml_is_not_the_paper_file(monkeypatch):
    """@brief 验证 base 预设只读仓库里的 ``taic.toml``，且 ``taic.toml`` 不是论文预设文件；环境变量与参数都能选预设。

    @param monkeypatch 用于设置假凭据与 ``TAIC_PRESET``。
    """
    import tomllib

    from taic.llm import REPO_ROOT, load_config

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    raw = tomllib.loads((REPO_ROOT / "taic.toml").read_text())["search"]
    _, s = load_config(preset="base")
    assert (s.budget_usd, s.max_rounds, s.patience) == (raw.get("budget_usd", 15.0), raw.get("max_rounds", 12), raw.get("patience", 4))
    monkeypatch.setenv("TAIC_PRESET", "base")
    assert load_config()[1].budget_usd == s.budget_usd  # the environment selects it too
    assert load_config(preset="paper")[1].budget_usd == 15.0  # ... and the argument wins


def test_an_explicit_config_file_is_used_as_it_is_and_a_bad_preset_is_an_error(tmp_path, monkeypatch):
    """@brief 验证显式配置文件不被预设叠加，未知或越界的预设名报错。

    @param tmp_path 临时目录，配置文件写在这里。
    @param monkeypatch 用于设置假凭据。
    """
    from taic.llm import LLMError, load_config

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    toml = tmp_path / "mine.toml"
    toml.write_text("[search]\nbudget_usd = 0.75\n")
    assert load_config(toml)[1].budget_usd == 0.75  # no preset is layered over an explicit file
    with pytest.raises(LLMError, match="not found"):
        load_config(preset="nope")
    with pytest.raises(LLMError, match="bad config preset"):
        load_config(preset="../taic")


def test_budget_for_is_15_usd_except_swiglu_in_table_1():
    """@brief 验证答题预算默认 15 美元，只有表 1 的 swiglu 是 1 美元；缩放与总预算上限按比例生效。"""
    from experiments.common import budget_for, fits_cap

    assert budget_for("gemm") == 15.0 and budget_for("softmax") == 15.0 and budget_for("flash_attention@64x32t256") == 15.0
    assert budget_for("swiglu") == 15.0 and budget_for("swiglu", table1=True) == 1.0 and budget_for("conv2d", table1=True) == 15.0
    assert budget_for("gemm", 0.2) == pytest.approx(3.0) and budget_for("swiglu", 2.0, table1=True) == 2.0
    assert fits_cap(0.0, 15.0, None) and fits_cap(10.0, 15.0, 25.0) and not fits_cap(10.1, 15.0, 25.0)


def test_p1_runs_all_default_to_the_paper_budget():
    """@brief 验证 p1 各命名运行默认都用论文预算。"""
    from experiments import p1_runs
    from taic.llm import PAPER_BUDGET_USD

    assert {n: s["budget"] for n, s in p1_runs.RUNS.items()} == {n: PAPER_BUDGET_USD for n in p1_runs.RUNS}


def _stub_run_one(seen, spent):
    """@brief 造假 ``run_one``：记录 ``(题目, 预算)``，写下 ``result.json`` 并返回带停止原因的对象。

    @param seen 记录每次调用的列表。
    @param spent 这次假运行报告的花费。
    @return 可直接替换 ``run_one`` 的函数。
    """
    def one(task, dtype, size, run_dir, *, budget=None, **kw):
        seen.append((task, budget))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps({
            "task": task, "dtype": dtype, "model": "m", "stop_reason": "converged (4 rounds without improvement)", "rounds": 4, "evaluations": 9,
            "spent_usd": spent, "budget_usd": budget, "confirmed_speedup": 1.1, "best_speedup": 1.1, "candidate_us": 5.0, "baseline_us": {"mlx": 5.5},
        }))
        return SimpleNamespace(stop_reason="converged (4 rounds without improvement)")

    return one


def test_run_suite_gives_every_run_the_paper_budget_and_honours_the_optional_total_cap(tmp_path, monkeypatch):
    """@brief 验证 ``run_suite`` 默认给每个运行论文预算，总预算放不下时跳过后续运行，``--budget`` 与 ``--budget-scale`` 可覆盖。

    @param tmp_path 临时目录，结果写在这里。
    @param monkeypatch 用假 ``run_one`` 替换真实运行。
    """
    import experiments.run_suite as rs

    seen, logs = [], []
    monkeypatch.setattr(rs, "run_one", _stub_run_one(seen, 6.0))
    rows = rs.run_suite(["relu", "sigmoid", "silu"], ["fp16"], "small", tmp_path / "a", with_naive=False, log=logs.append)
    assert seen == [("relu", 15.0), ("sigmoid", 15.0), ("silu", 15.0)] and len(rows) == 3  # no cap by default
    seen.clear()
    rows = rs.run_suite(["relu", "sigmoid", "silu"], ["fp16"], "small", tmp_path / "b", max_total_usd=25.0, with_naive=False, log=logs.append)
    assert seen == [("relu", 15.0), ("sigmoid", 15.0)] and len(rows) == 2  # 6 + 6 spent, a third $15 budget would not fit under $25
    assert any("does not fit under --max-total-usd $25.00" in line for line in logs)
    seen.clear()
    rs.run_suite(["relu"], ["fp16"], "small", tmp_path / "c", budget=1.5, with_naive=False, log=logs.append)
    assert seen == [("relu", 1.5)]  # --budget overrides it
    seen.clear()
    rs.run_suite(["relu"], ["fp16"], "small", tmp_path / "d", budget_scale=0.2, with_naive=False, log=logs.append)
    assert seen[0][1] == pytest.approx(3.0)  # --budget-scale 0.2 is a $3 smoke run
    assert "budget USD" in rs.markdown_table(rows) and "15.00" in rs.markdown_table(rows)


def test_repeatability_uses_table_1_budgets_and_the_total_cap(tmp_path, monkeypatch):
    """@brief 验证可重复性实验按表 1 取预算，并在总预算放不下时减少试验次数。

    @param tmp_path 临时目录，结果写在这里。
    @param monkeypatch 用假 ``run_one`` 替换真实运行。
    """
    import experiments.repeatability as rp

    seen, logs = [], []
    monkeypatch.setattr(rp, "run_one", _stub_run_one(seen, 0.5))
    table = rp.run_repeatability(["gemm", "swiglu"], "fp16", "small", tmp_path / "a", trials=2, log=logs.append)
    assert seen == [("gemm", 15.0), ("gemm", 15.0), ("swiglu", 1.0), ("swiglu", 1.0)]
    assert table["gemm"]["budget_usd"] == 15.0 and table["swiglu"]["budget_usd"] == 1.0
    seen.clear()
    table = rp.run_repeatability(["gemm", "swiglu"], "fp16", "small", tmp_path / "b", trials=3, max_total_usd=15.2, log=logs.append)
    # gemm: the first trial fits ($15 <= $15.2), the second would not ($0.5 spent + $15); the $1 swiglu trials still fit
    assert seen == [("gemm", 15.0), ("swiglu", 1.0), ("swiglu", 1.0), ("swiglu", 1.0)]
    assert len(table["gemm"]["trials"]) == 1 and len(table["swiglu"]["trials"]) == 3 and any("does not fit under --max-total-usd" in line for line in logs)
    assert "$15.00" in rp.markdown(table)


def test_p1_runs_cap_budget_override_and_the_note_about_results_at_another_budget(tmp_path, monkeypatch):
    """@brief 验证 p1 命名运行沿用旧预算时跳过并给出说明，总预算上限生效，汇总里记录每个运行的真实预算。

    @param tmp_path 临时目录，输出目录在这里造。
    @param monkeypatch 用假 ``run_one`` 与假基线替换真实运行。
    """
    from experiments import p1_runs

    seen, logs = [], []
    monkeypatch.setattr(p1_runs, "run_one", _stub_run_one(seen, 7.0))
    monkeypatch.setattr(p1_runs, "naive_reference", lambda *a, **kw: {"speedup": 1.0})
    out = tmp_path / "p1"
    (out / "fa_flash").mkdir(parents=True)
    (out / "fa_flash" / "result.json").write_text(json.dumps({
        "task": "flash_attention", "dtype": "fp16", "model": "m", "stop_reason": "budget_exhausted", "rounds": 8, "evaluations": 9, "spent_usd": 3.12, "budget_usd": 3.0,
    }))
    rows = p1_runs.run_named(["fa_flash", "fa_pro", "fa_paper", "gemm_paper"], out, log=logs.append, max_total_usd=20.0)
    text = "\n".join(logs)
    assert "fa_flash was run with a $3.00 budget" in text and "(skip fa_flash: already done, with a $3.00 budget, not $15.00)" in text
    assert [t for t, _ in seen] == ["flash_attention"] and [r["run"] for r in rows] == ["fa_flash", "fa_pro"]
    assert seen[0][1] == 15.0 and "fa_paper: its $15.00 budget does not fit under --max-total-usd $20.00" in text
    summary = json.loads((out / "summary.json").read_text())
    assert {r["run"]: r["budget_usd"] for r in summary} == {"fa_flash": 3.0, "fa_pro": 15.0}  # each row shows the budget its run really had
    assert "budget USD" in p1_runs.markdown(summary) and "| 3.00 |" in p1_runs.markdown(summary)
    seen.clear()
    p1_runs.run_named(["mamba2"], out, log=logs.append, budget=2.0)
    assert seen == [("mamba2_chunk_scan", 2.0)]


def test_remaining_is_the_current_budget_minus_the_spend():
    """@brief 验证剩余额度等于当前预算减花费，缺数时为 None，超支时为负。"""
    assert budget_status(1.5, 3.0)["remaining_usd"] == pytest.approx(1.5)
    assert budget_status(3.123, 3.0, "budget_exhausted", False)["remaining_usd"] == pytest.approx(-0.123)
    assert budget_status(None, 3.0)["remaining_usd"] is None
    assert budget_status(1.0, None)["remaining_usd"] is None

