"""The SSE refresh plan is computed per changed run instead of refetching every panel."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_index_refresh_plan_in_node():
    """@brief 用 Node 执行刷新决策：无关 run 只刷预算，受影响的面板才重取。"""
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()
    start = page.index("function planIndexRefresh(")
    brace = page.index("{", start)
    depth = 0
    for index in range(brace, len(page)):
        if page[index] == "{":
            depth += 1
        elif page[index] == "}":
            depth -= 1
            if depth == 0:
                function = page[start:index + 1]
                break
    else:
        raise AssertionError("unclosed planIndexRefresh")

    script = function + r'''
const base = {reviewsPage: false, selectedRun: "", selectedExperiment: "", visible: "detail", runsPageIds: new Set(), knownRuns: new Map()};
const ctx = (over) => Object.assign({}, base, over);

// 空变化集无法定位，回退到旧的整批刷新。
let plan = planIndexRefresh([], ctx({selectedRun: "a"}));
if (!plan.runs || !plan.experiments || !plan.detail || !plan.banner) throw new Error("empty change set must fall back");

// 别的 run 变化：它不在当前页、没打开、不属于任何实验，只刷预算。
plan = planIndexRefresh(["other"], ctx({knownRuns: new Map([["other", ""]]), selectedRun: "open"}));
if (!plan.banner || plan.runs || plan.experiments || plan.compare || plan.detail) throw new Error("unrelated run must not refetch panels");

// 正在打开的 run 变化：只多刷它的详情。
plan = planIndexRefresh(["open"], ctx({knownRuns: new Map([["open", ""]]), selectedRun: "open"}));
if (!plan.detail || plan.runs || plan.experiments) throw new Error("open run should refresh detail only");

// 新 run：列表与实验聚合都要更新。
plan = planIndexRefresh(["brand-new"], ctx({}));
if (!plan.runs || !plan.experiments) throw new Error("new run must refresh list and experiments");

// 当前页可见且属于当前实验：列表、实验、对照都刷新。
const known = new Map([["v", "exp1"]]);
plan = planIndexRefresh(["v"], ctx({knownRuns: known, runsPageIds: new Set(["v"]), selectedExperiment: "exp1"}));
if (!plan.runs || !plan.experiments || !plan.compare) throw new Error("visible experiment run must refresh compare");

// 同一行但打开的是另一个实验：不刷对照。
plan = planIndexRefresh(["v"], ctx({knownRuns: known, runsPageIds: new Set(["v"]), selectedExperiment: "exp2"}));
if (plan.compare) throw new Error("other experiment must not refresh compare");

// 账本视图下不刷详情。
plan = planIndexRefresh(["open"], ctx({knownRuns: new Map([["open", ""]]), selectedRun: "open", visible: "ledger"}));
if (plan.detail || !plan.ledger) throw new Error("ledger view must not load detail");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def _extract(page: str, name: str) -> str:
    """@brief 从页面源码里抽出整段函数（Node 里直接执行）。

    @param page 页面源码。
    @param name 函数名。
    @return 函数源码。
    """
    start = page.index(f"function {name}(")
    brace = page.index("{", start)
    depth = 0
    for index in range(brace, len(page)):
        if page[index] == "{":
            depth += 1
        elif page[index] == "}":
            depth -= 1
            if depth == 0:
                return page[start:index + 1]
    raise AssertionError(f"unclosed function {name}")


def test_a_live_revision_frame_asks_for_no_requests_in_node():
    """@brief §5.1 验收①：假 frame 里只有 live_revision 变（GPU 锁目录/在飞文件）→ 0 请求、0 重绘。

    走真实的 ``planStreamFrame``（生产 SSE 处理器用的同一个函数）；帧里其它 revision 不变时
    返回的 index/turns 都为空，处理器不会进入任何 load* 分支。

    @raises subprocess.CalledProcessError Node 里的断言失败时抛出。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in ("planStreamFrame", "planIndexRefresh")) + r'''
const seen = {index: "5", turns: "3", reviews: 7};
function loads(plan) {
  const out = [];
  for (const key of ["banner", "experiments", "compare", "runs", "detail", "ledger"]) if (plan[key]) out.push(key);
  return out;
}
// 只有 live 变：处理器两个分支都不进去，等于 0 次请求。
let due = planStreamFrame({index_revision: 5, live_revision: 9, turns_revision: 3, feedback_revision: 7}, seen);
if (due.index || due.turns || due.reviews) throw new Error("a live-only frame scheduled work: " + JSON.stringify(due));
const ctx = {reviewsPage: false, selectedRun: "open", selectedExperiment: "", visible: "detail",
  runsPageIds: new Set(["open"]), knownRuns: new Map([["open", ""]])};
let plan = planIndexRefresh([], ctx);
if (loads(plan).length === 0) throw new Error("sanity: an index frame would have refreshed something");
// index 真的变了才排刷新。
due = planStreamFrame({index_revision: 6, revision: 6, live_revision: 9, runs: ["open"], turns_revision: 3}, seen);
if (due.index !== "6" || due.turns) throw new Error("an index bump was not detected: " + JSON.stringify(due));
plan = planIndexRefresh(["open"], ctx);
if (loads(plan).join(",") !== "banner,runs,detail") throw new Error("index frame plan drifted: " + loads(plan).join(","));
// turns revision 变化只刷回合，不重取详情。
due = planStreamFrame({index_revision: 6, turns_revision: 4}, {index: "6", turns: "3", reviews: 7});
if (due.index || due.turns !== "4") throw new Error("turns frame plan drifted: " + JSON.stringify(due));
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


def test_frontend_tracks_page_membership():
    """@brief 前端在 loadRuns 里记录当前页 id 与 run→experiment 映射，供精准刷新使用。"""
    page = Path("observe/static/index.html").read_text()
    assert "runsPageIds = new Set(rows.map((row) => row.run_id));" in page
    assert 'rows.forEach((row) => knownRuns.set(row.run_id, row.experiment_id || ""));' in page
    assert "planIndexRefresh(changed, {" in page
    assert "frame.live_revision" not in page  # 没有实时面板，live 帧不应触发任何刷新
