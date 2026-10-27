"""§10.4 长内容：组级上限、加载更多、折叠真卸载、keyed patch、大文本窗口与筛选进 URL。

服务端只测“每页多少条、游标与戳”；渲染层在 Node 里跑纯函数/字符串断言（没有布局树，
真实 DOM 节点数不进这里）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from observe import detail, query


def _page(markup: str) -> int:
    """@brief 一段 HTML 里的 ``<tr`` 数量。

    @param markup HTML 字符串。
    @return 行数。
    """
    return markup.count("<tr")


def _run(tmp_path, monkeypatch) -> Path:
    """@brief 一场带 30 条反馈、60 颗候选的运行，用来测分页。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return run 目录。
    """
    runs = tmp_path / "runs"
    directory = runs / "conv2d" / "paged"
    (directory / "candidates").mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "absent.sqlite"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "phase": "done", "started_at": 1.0,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
    }), encoding="utf-8")
    for index in range(60):
        candidate = directory / "candidates" / f"{index:03d}_c{index}"
        candidate.mkdir()
        (candidate / "report.json").write_text(json.dumps({"status": "ok", "stage": "bench"}), encoding="utf-8")
    with (directory / "feedback.jsonl").open("w", encoding="utf-8") as handle:
        for seq in range(1, 31):
            handle.write(json.dumps({"format": "agent_feedback", "seq": seq, "raw": f"note {seq}"}) + "\n")
    return directory


def test_the_group_page_sizes_are_shared_by_the_ui_and_the_server(tmp_path, monkeypatch):
    """@brief §10.4：候选 50/页、反馈 10/页、回合 40 组；缺省请求与常量一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    page = Path("observe/static/index.html").read_text()
    assert "const CANDIDATES_PAGE = 50;" in page
    assert "const FEEDBACK_PAGE = 10;" in page
    assert "const REVIEW_RUN_PAGE = 20;" in page
    assert query.CANDIDATES_PAGE == 50 and query.FEEDBACK_PAGE == 10 and detail.TURNS_PAGE == 40

    _run(tmp_path, monkeypatch)
    candidates = query.candidates_panel("conv2d/paged")
    assert candidates is not None and len(candidates["candidates"]) == query.CANDIDATES_PAGE
    assert candidates["total"] == 60 and candidates["next"]

    first = query.feedback_panel("conv2d/paged")
    assert first is not None and len(first["feedback"]["records"]) == query.FEEDBACK_PAGE
    assert first["feedback"]["total"] == 30 and first["feedback"]["next"]


def test_feedback_records_page_by_ten_and_each_page_has_its_own_stamp(tmp_path, monkeypatch):
    """@brief 反馈一页 10 条；游标走到尾就 ``next=None``；戳按页形状区分。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _run(tmp_path, monkeypatch)
    first = query.feedback_panel("conv2d/paged")
    second = query.feedback_panel("conv2d/paged", cursor=first["feedback"]["next"])
    third = query.feedback_panel("conv2d/paged", cursor=second["feedback"]["next"])
    assert [len(item["feedback"]["records"]) for item in (first, second, third)] == [10, 10, 10]
    assert third["feedback"]["next"] is None
    seqs = [row["seq"] for item in (first, second, third) for row in item["feedback"]["records"]]
    assert seqs == list(range(1, 31))
    assert len({first["stamp"], second["stamp"], third["stamp"]}) == 3
    # 同一页 + 同一戳才短路；换页形状不会共享 same。
    again = query.feedback_panel("conv2d/paged", stamp=first["stamp"])
    assert again is not None and again["same"] is True
    wider = query.feedback_panel("conv2d/paged", stamp=first["stamp"], limit=20)
    assert wider is not None and wider["same"] is False
    assert "raw" not in json.dumps(first, ensure_ascii=False)


def test_the_feedback_endpoint_honours_limit_and_cursor(tmp_path, monkeypatch):
    """@brief ``/feedback?limit=&cursor=`` 走同一个分页；非法 limit 是 400。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    _run(tmp_path, monkeypatch)
    with TestClient(app) as client:
        first = client.get("/api/v2/runs/conv2d/paged/feedback?limit=10").json()
        assert len(first["feedback"]["records"]) == 10 and first["feedback"]["next"]
        second = client.get(
            "/api/v2/runs/conv2d/paged/feedback?limit=10&cursor=" + first["feedback"]["next"]
        ).json()
        assert second["feedback"]["records"][0]["seq"] == 11
        bad = client.get("/api/v2/runs/conv2d/paged/feedback?limit=abc")
        assert bad.status_code == 400
        candidates = client.get("/api/v2/runs/conv2d/paged/candidates").json()
        assert len(candidates["candidates"]) == query.CANDIDATES_PAGE
        assert candidates["next"]


def _node(script: str) -> None:
    """@brief 跑一段 Node 脚本；失败时把 stderr 抛出来。

    @param script 脚本文本。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def _extract(page: str, name: str) -> str:
    """@brief 从页面抽一个函数（含函数体）。

    @param page 页面源码。
    @param name 函数名。
    @return 函数源码。
    """
    prefix = f"async function {name}(" if f"async function {name}(" in page else f"function {name}("
    start = page.index(prefix)
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


def test_the_candidate_group_cap_keeps_the_rendered_html_bounded():
    """@brief 候选数据从 50 涨到 150，首屏 HTML 的 ``<tr`` 与字节数不跟着线性涨。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "esc", "num", "newestCandidates", "paginateGroup", "filterOptions", "candidatesFilters", "candidatesMarkup")) + r'''
const CANDIDATES_PAGE = 50;
const CANDIDATE_STATUSES = [["", "全部状态"], ["ok", "ok"]];
const CANDIDATE_STAGES = [["", "全部阶段"], ["bench", "计时"]];
let candidatePages = 1, candidateStatus = "", candidateStage = "";
let candidateNotice = "";
let selectedCandidate = "";
function seedLabel() { return ""; }
function confirmWinnerLabel() { return ""; }
function marksFor() { return ""; }
const row = (index) => ({idx: index, label: "c" + index, status: "ok", stage: "bench", speedup: index + 1, message: ""});
const rows = (count, from) => Array.from({length: count}, (_, index) => row((from || 0) + index));
// 表头也占一个 <tr>；数据行数是总数减一。
const dataRows = (html) => (html.match(/<tr/g) || []).length - 1;
// 同一页只差在内存里还躺着多少行（小页就是大页最新的 50 行）：首屏 HTML 必须逐字节一样。
const small = candidatesMarkup({candidates: rows(150).slice(100), candidatesTotal: 150, candidatesNext: "cur"});
const grown = candidatesMarkup({candidates: rows(150), candidatesTotal: 150, candidatesNext: "cur"});
if (dataRows(small) !== 50) throw new Error("first page should render 50 rows, got " + dataRows(small));
if (dataRows(grown) !== 50) throw new Error("grown data rendered " + dataRows(grown) + " rows; the group cap is gone");
if (grown.length !== small.length) throw new Error("bytes grew with the data: " + small.length + " -> " + grown.length);
if (!grown.includes("加载更多") || !grown.includes("50 / 150")) throw new Error("load-more footer missing");
candidatePages = 2;
const second = candidatesMarkup({candidates: rows(150), candidatesTotal: 150, candidatesNext: "cur"});
if (dataRows(second) !== 100) throw new Error("an explicit page bump should show 100 rows, got " + dataRows(second));
console.log("ok");
'''
    _node(script)


def test_the_feedback_group_renders_ten_records_per_page():
    """@brief 30 条反馈在默认页只渲染 10 条，翻页后才加到 20 条。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "esc", "num", "paginateGroup", "feedbackCompactMarkup")) + r'''
const FEEDBACK_PAGE = 10;
let feedbackPages = 1;
let feedbackNotice = "";
let currentSeq = 0;
const record = (seq) => ({
  seq: seq, label: "c" + seq, round: 1, committee: "first", provider: "p", model: "m",
  claim_counts: {total: 0}, claims: [], truncated_sections: [],
});
const records = (count) => Array.from({length: count}, (_, index) => record(index + 1));
const sessions = (html) => (html.match(/class="fb-session"/g) || []).length;
const doc = {feedback: {records: records(30), total: 30, next: "cur"}};
const first = feedbackCompactMarkup(doc);
if (sessions(first) !== 10) throw new Error("expected 10 sessions, got " + sessions(first));
if (!first.includes("加载更多") || !first.includes("10 / 30")) throw new Error("feedback load-more footer missing");
feedbackPages = 2;
const second = feedbackCompactMarkup(doc);
if (sessions(second) !== 20) throw new Error("expected 20 sessions after a page bump, got " + sessions(second));
console.log("ok");
'''
    _node(script)


def test_the_review_run_group_page_is_twenty():
    """@brief 每个 run 组一页 20 条；页数只由显式“加载更多”加。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = _extract(page, "reviewRunSlice") + r'''
const REVIEW_RUN_PAGE = 20;
const items = Array.from({length: 45}, (_, index) => ({seq: index}));
if (reviewRunSlice(items, 1).length !== 20) throw new Error("default page is not 20");
if (reviewRunSlice(items, 2).length !== 40) throw new Error("second page is not 40");
if (reviewRunSlice(items, 3).length !== 45) throw new Error("last page must stop at the data");
if (reviewRunSlice([], 3).length !== 0) throw new Error("empty group must stay empty");
if (reviewRunSlice(items, 0).length !== 20) throw new Error("a missing page count must mean one page");
console.log("ok");
'''
    _node(script)


def test_only_changed_detail_blocks_are_replanned():
    """@brief keyed patch 的纯规划：没变的块 0 次重画；变一块只重画一块；换序不重画。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = _extract(page, "planDetailPatch") + r'''
const cache = new Map([["title", "T"], ["budget", "B"], ["turns", "U"]]);
const order = ["title", "budget", "turns"];
const same = [
  {key: "title", html: "T"}, {key: "budget", html: "B"}, {key: "turns", html: "U"},
];
if (planDetailPatch(same, cache, order).length !== 0) throw new Error("identical sections must not be redrawn");
const one = [
  {key: "title", html: "T"}, {key: "budget", html: "B2"}, {key: "turns", html: "U"},
];
const ops = planDetailPatch(one, cache, order);
if (ops.length !== 1 || ops[0] !== "budget") throw new Error("only the changed block may be redrawn: " + JSON.stringify(ops));
const reordered = [
  {key: "budget", html: "B"}, {key: "title", html: "T"}, {key: "turns", html: "U"},
];
if (planDetailPatch(reordered, cache, order).length !== 0) throw new Error("order changes must move nodes, not redraw them");
console.log("ok");
'''
    _node(script)


def test_the_large_text_window_keeps_the_tail_and_the_pi_budget():
    """@brief 大文本只留尾部窗口；pi 的 256KiB 服务端预算不变。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    assert detail._PI_TAIL_BYTES == 256 * 1024
    page = Path("observe/static/index.html").read_text()
    script = _extract(page, "tailWindow") + r'''
const TEXT_WINDOW_BYTES = 256 * 1024;
const long = "0123456789".repeat(30);
const windowed = tailWindow(long, 100);
if (!windowed.truncated || windowed.dropped !== 200) throw new Error("tail window math is wrong: " + JSON.stringify({dropped: windowed.dropped}));
if (windowed.text.length !== 100 || !windowed.text.endsWith("0123456789")) throw new Error("tail window lost the tail");
const short = tailWindow("abc", 100);
if (short.truncated || short.text !== "abc" || short.dropped !== 0) throw new Error("short text must pass through");
if (tailWindow("x".repeat(10), 0).text.length !== 10) throw new Error("a zero limit must fall back to the default budget");
// 非 ASCII：262144 个「界」是 786432B；必须按字节截断且不切坏字符。
const cjk = "界".repeat(262144);
const cjkWindow = tailWindow(cjk);
if (!cjkWindow.truncated) throw new Error("CJK text must be truncated by bytes");
const cjkBytes = new TextEncoder().encode(cjkWindow.text);
if (cjkBytes.length > TEXT_WINDOW_BYTES) throw new Error("CJK window exceeds 256KiB: " + cjkBytes.length);
if (cjkWindow.text.includes("\uFFFD")) throw new Error("CJK window split a character");
if (!cjk.endsWith(cjkWindow.text)) throw new Error("CJK window is not a tail");
const emoji = "😀".repeat(200000);
const emojiWindow = tailWindow(emoji, 1000);
const emojiBytes = new TextEncoder().encode(emojiWindow.text);
if (emojiBytes.length > 1000) throw new Error("emoji window exceeds the byte budget");
if (emojiWindow.text.includes("\uFFFD")) throw new Error("emoji window split a surrogate pair");
if (!emoji.endsWith(emojiWindow.text)) throw new Error("emoji window is not a tail");
console.log("ok");
'''
    _node(script)


def test_the_candidate_row_is_selected_by_one_click_and_the_evidence_button_opens_the_stage():
    """@brief §10.2：候选单击选中、显式「证据」按钮打开阶段/代码，双击保留为快捷键。

    这是候选面板的唯一入口测试：此前 ``bindCandidatePicks`` 在 20 个 Node 套里都被 stub，
    「单击取代双击」的行为没有任何断言。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in ("bindCandidatePicks",)) + r'''
const calls = [];
let selectedCandidate = "";
let additions = 0;
function selectCandidate(label) { selectedCandidate = label; calls.push(["select", label]); }
function openStage(label) { calls.push(["stage", label]); }
function applyCandidateFilter() {}
function loadMoreCandidates() {}
function applyRovingRows(rows) { calls.push(["roving", rows.length]); }
function makeButton(className) {
  const handlers = {};
  const button = {dataset: {}, addEventListener(type, fn) { handlers[type] = fn; additions += 1; }, handlers};
  button.closest = (selector) => (selector === "tr" ? row : null);
  button.className = className;
  return button;
}
const handlers = {};
const row = {
  dataset: {label: "r1.p0"},
  addEventListener(type, fn) { handlers[type] = fn; additions += 1; },
  handlers,
};
const evidence = makeButton("quiet evidence-open");
const seq = makeButton("quiet seq-open");
const host = {
  querySelectorAll(selector) {
    if (selector === "tr.pick") return [row];
    if (selector === "tr.pick button.evidence-open") return [evidence];
    if (selector === "tr.pick button.seq-open") return [seq];
    return [];
  },
  querySelector() { return null; },
};
bindCandidatePicks(host);
const bound = additions;
if (bound !== 3) throw new Error("expected row+evidence+dblclick handlers, got " + bound);
if (!row.dataset.picksBound || !evidence.dataset.picksBound || !seq.dataset.picksBound) throw new Error("nodes were not marked as bound");
// 单击行（非按钮）→ 选中，不开阶段。
let stopped = 0;
row.handlers.click({target: {closest: () => null}});
if (selectedCandidate !== "r1.p0") throw new Error("one click did not select the candidate");
if (calls.some((call) => call[0] === "stage")) throw new Error("a plain row click opened the stage panel");
// 单击行内按钮 → 不选中（按钮自己处理）。
selectedCandidate = "";
row.handlers.click({target: {closest: () => ({})}});
if (selectedCandidate !== "") throw new Error("clicking a button inside the row also selected the row");
// 显式「证据」按钮 → 打开这一颗的阶段/代码，并且不让点击冒泡成选中。
evidence.handlers.click({stopPropagation() { stopped += 1; }});
if (!calls.some((call) => call[0] === "stage" && call[1] === "r1.p0")) throw new Error("the evidence button did not open the candidate stage");
if (stopped !== 1) throw new Error("the evidence button did not stop propagation");
// 双击序号仍是快捷键。
seq.handlers.dblclick();
if (calls.filter((call) => call[0] === "stage").length !== 2) throw new Error("dblclick no longer opens the stage panel");
// keyed patch 复用节点：再绑一次不能叠加监听。
bindCandidatePicks(host);
if (additions !== bound) throw new Error("rebinding duplicated the handlers: " + additions);
console.log("ok");
'''
    _node(script)


def test_the_candidate_filter_lands_in_the_url_and_reloads_only_that_panel():
    """@brief 状态/阶段筛选进 URL，且只重取候选块（不重取详情/运行表）。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    script = fields + "\n" + filter_line + "\n" + "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "applyCandidateFilter")) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0;
let visible = "detail", sideView = "runs";
let selectedRun = "conv2d/paged", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "candidates", currentCluster = "", currentSeq = 0;
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let candidateNotice = "", feedbackNotice = "";
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
let runsSort = "started";
const applied = {task: "relu", dtype: "fp16", model: "", status: "", target: ""};
let detailId = "conv2d/paged", detailDoc = {id: "conv2d/paged"};
let openPanels = new Set(["candidates"]);
let panelStamps = {candidates: "p1"};
const panelLoads = [];
function loadPanel(name, id) { panelLoads.push([name, id]); }
let committed = null;
function commitUrl(state) { committed = state; }
applyCandidateFilter("status", "ok");
if (panelLoads.length !== 1 || panelLoads[0][0] !== "candidates") throw new Error("filter must reload only the candidates panel: " + JSON.stringify(panelLoads));
if (panelStamps.candidates !== "") throw new Error("the old panel stamp must be cleared when the page shape changes");
if (committed.cstatus !== "ok" || committed.cstage) throw new Error("filter state missing from the committed URL state");
const parsed = readUrl("?" + writeUrl(committed));
if (parsed.cstatus !== "ok") throw new Error("cstatus did not round-trip: " + JSON.stringify(parsed));
applyCandidateFilter("stage", "bench");
if (committed.cstage !== "bench" || committed.cstatus !== "ok") throw new Error("stage filter did not merge with status");
if (candidatePages !== 1) throw new Error("changing a filter must go back to page one");
console.log("ok");
'''
    _node(script)


def test_the_review_filter_round_trips_through_the_url():
    """@brief 审查页的筛选进 URL：缺省是 pending，显式空值表示全部。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in ("reviewFilterFromUrl", "reviewFilterQuery")) + r'''
if (reviewFilterFromUrl("") !== "pending") throw new Error("missing rstatus must mean pending");
if (reviewFilterFromUrl("?rstatus=info") !== "info") throw new Error("rstatus was not read");
if (reviewFilterFromUrl("?rstatus=") !== "") throw new Error("an explicit empty rstatus must mean all");
if (reviewFilterQuery("", "adopted") !== "rstatus=adopted") throw new Error("rstatus was not written");
if (reviewFilterQuery("?run=a%2Fb&rstatus=old", "done") !== "run=a%2Fb&rstatus=done") throw new Error("other params must survive");
if (reviewFilterQuery("?rstatus=old", "") !== "rstatus=all") throw new Error("all must be explicit in the URL");
if (reviewFilterFromUrl("?rstatus=all") !== "") throw new Error("all must read back as the empty filter");
console.log("ok");
'''
    _node(script)


def test_the_detail_scroll_is_remembered_per_run():
    """@brief 滚动位置按 run 记住；换场不会被上一场的值污染。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "detailScroller", "rememberDetailScroll", "recallDetailScroll")) + r'''
const detailScrollMemory = new Map();
const box = {scrollTop: 0};
globalThis.document = {getElementById: (id) => (id === "detail-body" ? box : null)};
const host = {closest: () => null};
const store = new Map();
globalThis.sessionStorage = {
  getItem: (key) => (store.has(key) ? store.get(key) : null),
  setItem: (key, value) => store.set(key, String(value)),
};
box.scrollTop = 640;
rememberDetailScroll("run/a", host);
box.scrollTop = 0;
recallDetailScroll("run/a", host);
if (box.scrollTop !== 640) throw new Error("scroll was not restored for the same run: " + box.scrollTop);
box.scrollTop = 111;
rememberDetailScroll("run/b", host);
box.scrollTop = 0;
recallDetailScroll("run/a", host);
if (box.scrollTop !== 640) throw new Error("another run overwrote the remembered scroll");
if (store.get("observe-detail-scroll:run/a") !== "640") throw new Error("scroll was not persisted");
console.log("ok");
'''
    _node(script)


def test_the_first_screen_run_query_asks_for_at_most_sixty_rows():
    """@brief 首屏运行表按 ``RUNS_LIMIT`` 取一页，不会一次拉全量。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    limit_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const RUNS_LIMIT"))
    assert limit_line == "const RUNS_LIMIT = 50;"
    script = limit_line + "\n" + _extract(page, "runQuery") + r'''
let applied = {task: "relu", dtype: "fp16", model: "", status: "", target: ""};
let runsSort = "started", runsCursor = null, selectedExperiment = "";
const params = new URLSearchParams(runQuery());
if (params.get("limit") !== "50") throw new Error("run page limit is not 50: " + params.get("limit"));
if (Number(params.get("limit")) > 60) throw new Error("first screen must stay at 60 rows or fewer");
if (params.get("task") !== "relu" || params.get("dtype") !== "fp16") throw new Error("filters fell out of the run query");
console.log("ok");
'''
    _node(script)


def test_a_collapsed_review_group_unmounts_its_cards():
    """@brief 收起 = 真卸载（0 张卡）；展开只挂一页 20 张，还有剩余才给“加载更多”。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = _extract(page, "reviewFoldAction") + r'''
const REVIEW_RUN_PAGE = 20;
const closed = reviewFoldAction(false, 45, 1);
if (!closed.unmount || closed.mount !== 0) throw new Error("a closed group must unmount everything: " + JSON.stringify(closed));
const first = reviewFoldAction(true, 45, 1);
if (first.unmount || first.mount !== 20 || !first.more) throw new Error("an open group must mount one page: " + JSON.stringify(first));
const last = reviewFoldAction(true, 45, 3);
if (last.mount !== 45 || last.more) throw new Error("the last page must stop at the data: " + JSON.stringify(last));
const empty = reviewFoldAction(true, 0, 1);
if (empty.mount !== 0 || empty.more) throw new Error("an empty group must mount nothing");
console.log("ok");
'''
    _node(script)


def test_the_detail_sections_have_unique_keys_for_every_heavy_block():
    """@brief 详情块的 key 唯一且覆盖所有面板；keyed patch 靠它对齐节点。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    import re

    page = Path("observe/static/index.html").read_text()
    body = page[page.index("function detailSections(doc)"):page.index("function renderDetail(doc)")]
    keys = re.findall(r'\{key: "([^"]+)"', body)
    assert len(keys) == len(set(keys)), sorted(keys)
    for name in ("title", "timing", "budget", "confirm", "handoff", "model-costs", "costs", "turns"):
        assert name in keys, name
    for name in ("candidates", "lineage", "budget-events", "grid", "plan", "feedback"):
        assert f"panel-{name}" in keys, name


def test_the_review_folds_and_scroll_survive_a_refresh():
    """@brief 展开态与滚动位置写进 sessionStorage，刷新后能读回来。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "readReviewFold", "saveReviewFold", "readReviewScroll", "saveReviewScroll")) + r'''
let reviewOpen = new Set(["run\ta", "hw\tapple"]);
let reviewsScroll = 0;
const store = new Map();
globalThis.sessionStorage = {
  getItem: (key) => (store.has(key) ? store.get(key) : null),
  setItem: (key, value) => store.set(key, String(value)),
};
saveReviewFold();
const back = readReviewFold();
if (!back.has("run\ta") || !back.has("hw\tapple")) throw new Error("folds did not round-trip: " + JSON.stringify([...back]));
reviewsScroll = 320;
saveReviewScroll();
if (readReviewScroll() !== 320) throw new Error("scroll did not round-trip");
globalThis.sessionStorage = {getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); }};
if (readReviewFold().size !== 0 || readReviewScroll() !== 0) throw new Error("blocked storage must degrade to empty state");
console.log("ok");
'''
    _node(script)


def test_the_full_url_restore_reopens_every_expanded_panel_but_loads_the_deep_link_once():
    """@brief 刷新恢复所有展开块；深链主块仍由 applyPendingDetail 独占加载。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    apply_block = page[page.index("function applyViewState("):page.index("function applyPendingDetail(")]
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "panelNames")) + "\n" + apply_block + r'''
let navigationEpoch = 0;
let sideView = "runs", visible = "detail";
let selectedRun = "", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "", currentCluster = "", currentSeq = 0;
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0, pendingPanelOwner = "";
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let restoredDetailView = null;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let candidateNotice = "", feedbackNotice = "";
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
let openPanels = new Set();
let runsSort = "started";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
const hosts = {};
globalThis.location = {pathname: "/"};
globalThis.history = {pushState() {}, replaceState() {}};
globalThis.document = {
  getElementById: (id) => (hosts[id] = hosts[id] || {
    value: "", innerHTML: "",
    classList: {add() {}, remove() {}, toggle() {}},
  }),
  querySelector: () => null,
};
function resetPanels() { openPanels = new Set(); }
function showSection() {}
function show() {}
function showRight() {}
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}

applyViewState(readUrl("?run=run&panel=feedback&panels=feedback,candidates,lineage&seq=4"), {full: true});
if (!openPanels.has("candidates") || !openPanels.has("lineage")) throw new Error("expanded panels were not restored: " + JSON.stringify([...openPanels]));
if (openPanels.has("feedback")) throw new Error("the deep-link panel must stay out of openPanels until applyPendingDetail");
if (pendingPanel !== "feedback" || pendingSeq !== 4) throw new Error("the deep-link primary was not kept for applyPendingDetail");
currentPanel = "feedback";
const state = currentViewState();
if (!String(state.panels || "").split(",").includes("candidates")) throw new Error("openPanels did not reach the URL");
const parsed = readUrl("?" + writeUrl(state));
if (parsed.panel !== "feedback") throw new Error("the deep-link panel fell out of the URL");
if (parsed.panels !== "candidates,lineage") throw new Error("panels did not round-trip sorted: " + parsed.panels);
openPanels = new Set();
applyViewState(readUrl("?run=run&panel=feedback"), {full: true});
if (openPanels.size !== 0) throw new Error("a URL without panels must clear them");
console.log("ok");
'''
    _node(script)


def test_the_lazy_folds_round_trip_in_the_url():
    """@brief 回合 / epic-pi / 会话成本的展开态写进 `folds`，读回后仍是同一组。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    apply_block = page[page.index("function applyViewState("):page.index("function applyPendingDetail(")]
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "panelNames")) + "\n" + apply_block + r'''
let navigationEpoch = 0;
let sideView = "runs", visible = "detail";
let selectedRun = "", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "", currentCluster = "", currentSeq = 0;
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0, pendingPanelOwner = "";
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let restoredDetailView = null;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let candidateNotice = "", feedbackNotice = "";
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
let openPanels = new Set();
let runsSort = "started";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
const hosts = {};
globalThis.location = {pathname: "/"};
globalThis.history = {pushState() {}, replaceState() {}};
globalThis.document = {
  getElementById: (id) => (hosts[id] = hosts[id] || {
    value: "", innerHTML: "",
    classList: {add() {}, remove() {}, toggle() {}},
  }),
  querySelector: () => null,
};
function resetPanels() { openPanels = new Set(); turnsOpen = false; piOpen = false; costsOpen = false; }
function showSection() {}
function show() {}
function showRight() {}
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}
function rememberDetailScroll() {}
function recallDetailScroll() {}
function detailScroller() { return null; }
function clearDetailPatch() {}

applyViewState(readUrl("?run=run&panel=&folds=turns,pi,costs"), {full: true});
if (!turnsOpen || !piOpen || !costsOpen) throw new Error("folds were not restored");
const state = currentViewState();
if (state.folds !== "turns,pi,costs") throw new Error("folds did not reach the URL: " + JSON.stringify(state.folds));
const parsed = readUrl("?" + writeUrl(state));
if (parsed.folds !== "turns,pi,costs") throw new Error("folds did not round-trip: " + parsed.folds);
applyViewState(readUrl("?run=run&panel="), {full: true});
if (turnsOpen || piOpen || costsOpen) throw new Error("a URL without folds must close them");
console.log("ok");
'''
    _node(script)


def test_a_cold_open_restores_folds_and_requests_each_lazy_block_once():
    """@brief 冷开带 folds：`loadDetail` 的 resetPanels 不能擦掉恢复态；每块懒加载只请求一次。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    apply_block = page[page.index("function applyViewState("):page.index("function applyPendingDetail(")]
    load_block = page[page.index("async function loadDetail(id)"):page.index("function ledgerTable(")]
    functions = "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "panelNames", "planPanelFetches",
        "resetPanels", "planDetailRequests")) + "\n" + _extract(page, "refreshOpenPanels") + "\n" + apply_block + "\n" + load_block
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + functions + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
let sideView = "runs", visible = "detail";
let selectedRun = "", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "", currentCluster = "", currentSeq = 0;
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0, pendingPanelOwner = "";
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let restoredDetailView = null;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let candidateNotice = "", feedbackNotice = "";
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
let openPanels = new Set();
let runsSort = "started";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
let detailToken = 0, detailDoc = null, detailId = "", detailStamp = "";
let panelStamps = {}, panelTokens = {}, fullFeedback = new Map();
const panelInflight = new Map();
let panelRequestSeq = 0;
let candidatePageToken = 0, feedbackPageToken = 0, feedbackRecordToken = 0, feedbackClaimToken = 0;
const calls = {turns: 0, live: 0, costs: 0, panels: 0};
const hosts = {};
globalThis.location = {pathname: "/"};
globalThis.history = {pushState() {}, replaceState() {}};
globalThis.document = {
  getElementById: (id) => (hosts[id] = hosts[id] || {
    value: "", innerHTML: "",
    classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}},
  }),
  querySelector: () => null,
};
globalThis.fetch = async () => ({ok: true, json: async () => ({
  id: "run", detail_stamp: "d1", candidates: [], feedback: {records: []},
})});
function showSection() {}
function show() {}
function showRight() { loadDetail("run"); }
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}
function rememberDetailScroll() {}
function recallDetailScroll() {}
function detailScroller() { return null; }
function detailPatchedRun() { return ""; }
function clearDetailPatch() {}
function overlayClose() {}
function carryPanels() {}
function patchDetail() {}
function restoreDetailOpenState() {}
function bindPanels() {}
function paintDetailHead() {}
function bindCandidatePicks() {}
function bindFeedback() {}
function openStage() {}
function applyPendingDetail() {}
function loadPanel() { calls.panels += 1; }
function mergePanel() {}
function panelBody() { return ""; }
function panelLabel(name) { return name; }
function selectCandidate() {}
function loadFeedbackRecord() {}
function esc(value) { return String(value ?? ""); }
function refreshTurns() { calls.turns += 1; }
function refreshLive() { calls.live += 1; }
function loadSessionCosts() { calls.costs += 1; }

(async () => {
  applyViewState(readUrl("?run=run&panel=&folds=turns,pi,costs"), {full: true});
  for (let i = 0; i < 16; i += 1) await new Promise((resolve) => setImmediate(resolve));
  if (!turnsOpen || !piOpen || !costsOpen) throw new Error("resetPanels wiped the restored folds");
  if (calls.panels !== 0) throw new Error("no panel should be requested when none is open: " + calls.panels);
  if (calls.turns !== 1 || calls.live !== 1 || calls.costs !== 1) throw new Error("lazy blocks must load once each: " + JSON.stringify(calls));
  if (detailId !== "run") throw new Error("detail was not loaded");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_lazy_fold_toggle_writes_the_url_state():
    """@brief 回合 / epic-pi / 会话成本的开关必须 `commitUrl`，否则刷新即丢。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = _extract(page, "bindPanels") + r'''
function syncDetailSectionCache() {}
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
let detailId = "run", currentPanel = "", openPanels = new Set();
const committed = [];
function commitUrl(state) { committed.push(state); }
function currentViewState() {
  const state = {};
  const folds = [turnsOpen ? "turns" : "", piOpen ? "pi" : "", costsOpen ? "costs" : ""].filter(Boolean);
  if (folds.length) state.folds = folds.join(",");
  return state;
}
function refreshTurns() {}
function refreshLive() {}
function loadSessionCosts() {}
function loadPanel() {}
let handler = null;
const host = {dataset: {}, addEventListener: (_type, fn) => { handler = fn; }};
bindPanels(host);
handler({target: {dataset: {lazy: "turns"}, open: true}});
if (committed.length !== 1 || committed[0].folds !== "turns") throw new Error("turns fold did not reach the URL: " + JSON.stringify(committed));
handler({target: {dataset: {lazy: "pi"}, open: true}});
if (committed[1].folds !== "turns,pi") throw new Error("pi fold did not merge: " + JSON.stringify(committed[1]));
handler({target: {dataset: {lazy: "costs"}, open: false}});
if (committed[2].folds !== "turns,pi") throw new Error("a closed fold must stay out of the URL: " + JSON.stringify(committed[2]));
handler({target: {dataset: {lazy: "costs"}, open: true}});
if (committed[3].folds !== "turns,pi,costs") throw new Error("costs fold did not reach the URL: " + JSON.stringify(committed[3]));
console.log("ok");
'''
    _node(script)


def test_a_same_summary_restore_patches_folds_and_loads_them_once():
    """@brief `doc.same` 分支也要应用 `restoredDetailView`：patch 展开态并各加载一次。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    apply_block = page[page.index("function applyViewState("):page.index("function applyPendingDetail(")]
    load_block = page[page.index("async function loadDetail(id)"):page.index("function ledgerTable(")]
    functions = "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "panelNames", "planPanelFetches",
        "resetPanels", "planDetailRequests")) + "\n" + _extract(page, "refreshOpenPanels") + "\n" + apply_block + "\n" + load_block
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + functions + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
let sideView = "runs", visible = "detail";
let selectedRun = "", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "", currentCluster = "", currentSeq = 0;
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0, pendingPanelOwner = "";
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let restoredDetailView = null;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let candidateNotice = "", feedbackNotice = "";
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
let openPanels = new Set();
let runsSort = "started";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
let detailToken = 0, detailDoc = {id: "run"}, detailId = "run", detailStamp = "d1";
let panelStamps = {}, panelTokens = {}, fullFeedback = new Map();
const panelInflight = new Map();
let panelRequestSeq = 0;
let candidatePageToken = 0, feedbackPageToken = 0, feedbackRecordToken = 0, feedbackClaimToken = 0;
const calls = {turns: 0, live: 0, costs: 0, patches: 0, panels: 0};
const hosts = {};
globalThis.location = {pathname: "/"};
globalThis.history = {pushState() {}, replaceState() {}};
globalThis.document = {
  getElementById: (id) => (hosts[id] = hosts[id] || {
    value: "", innerHTML: "",
    classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}},
  }),
  querySelector: () => null,
};
globalThis.fetch = async () => ({ok: true, json: async () => ({same: true, detail_stamp: "d1"})});
function showSection() {}
function show() {}
function showRight() { loadDetail("run"); }
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}
function rememberDetailScroll() {}
function recallDetailScroll() {}
function detailScroller() { return null; }
function detailPatchedRun() { return "run"; }
function clearDetailPatch() {}
function overlayClose() {}
function carryPanels() {}
function patchDetail() { calls.patches += 1; }
function restoreDetailOpenState() {}
function restoreDetailOpenState() {}
function bindPanels() {}
function paintDetailHead() {}
function bindCandidatePicks() {}
function bindFeedback() {}
function openStage() {}
function applyPendingDetail() {}
function loadPanel() { calls.panels += 1; }
function mergePanel() {}
function panelBody() { return ""; }
function panelLabel(name) { return name; }
function selectCandidate() {}
function loadFeedbackRecord() {}
function esc(value) { return String(value ?? ""); }
function refreshTurns() { calls.turns += 1; }
function refreshLive() { calls.live += 1; }
function loadSessionCosts() { calls.costs += 1; }

(async () => {
  applyViewState(readUrl("?run=run&panel=&folds=turns,pi,costs"), {full: true});
  for (let i = 0; i < 16; i += 1) await new Promise((resolve) => setImmediate(resolve));
  if (!turnsOpen || !piOpen || !costsOpen) throw new Error("the same branch dropped the restored folds");
  if (calls.patches !== 1) throw new Error("the same branch must patch the restored open state, patches=" + calls.patches);
  if (calls.panels !== 0) throw new Error("no panel should be requested: " + calls.panels);
  if (calls.turns !== 1 || calls.live !== 1 || calls.costs !== 1) throw new Error("lazy blocks must load once each: " + JSON.stringify(calls));
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_the_first_candidate_page_is_newest_first_and_label_seq_locators_page(tmp_path, monkeypatch):
    """@brief 首页 newest-first；`label`/`seq` 定位到包含目标的页，深链不再落在页外。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    _run(tmp_path, monkeypatch)
    page = query.candidates_panel("conv2d/paged")
    labels = [row["label"] for row in page["candidates"]]
    assert labels[0] == "c59" and labels[-1] == "c10", labels[:3] + labels[-3:]
    located = query.candidates_panel("conv2d/paged", label="c0")
    assert located["page_index"] == 1
    assert "c0" in [row["label"] for row in located["candidates"]]
    assert len(located["candidates"]) == 10

    feedback = query.feedback_panel("conv2d/paged", seq=15)
    assert feedback["feedback"]["page_index"] == 1
    assert [row["seq"] for row in feedback["feedback"]["records"]] == list(range(11, 21))

    with TestClient(app) as client:
        via_http = client.get("/api/v2/runs/conv2d/paged/candidates?label=c0").json()
        assert via_http["page_index"] == 1 and "c0" in [row["label"] for row in via_http["candidates"]]
        seq_http = client.get("/api/v2/runs/conv2d/paged/feedback?seq=15").json()
        assert seq_http["feedback"]["page_index"] == 1
        assert [row["seq"] for row in seq_http["feedback"]["records"]] == list(range(11, 21))


def test_a_candidate_deep_link_outside_the_first_page_is_fetched_and_selected():
    """@brief `candidate=c0` 不在首页时，按 label 定位页取回、扩到那一页并选中。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "newestCandidates", "ensureCandidateVisible", "applyPendingCandidate")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const CANDIDATES_PAGE = 50;
const row = (index) => ({idx: index, label: "c" + index, status: "ok", stage: "bench", speedup: 1, message: ""});
const firstPage = Array.from({length: 50}, (_, index) => row(59 - index));
let detailDoc = {id: "run", candidates: firstPage, candidatesTotal: 60, candidatesNext: "cur"};
let detailId = "run", openPanels = new Set(["candidates"]), candidatePages = 1;
let selected = "", repainted = 0;
function selectCandidate(label) { selected = label; }
function repaintCandidatePanel() { repainted += 1; }
function panelUrl(id, name) { return "/api/v2/runs/" + id + "/" + name; }
const calls = [];
globalThis.fetch = async (url) => {
  calls.push(String(url));
  const rest = Array.from({length: 10}, (_, index) => row(9 - index));
  return {ok: true, json: async () => ({candidates: rest, total: 60, page_index: 1, candidate_marks: []})};
};
(async () => {
  const found = await ensureCandidateVisible("c0");
  if (!found) throw new Error("the locator did not find c0");
  if (!calls[0].includes("label=c0")) throw new Error("wrong locator request: " + calls[0]);
  if (detailDoc.candidates.length !== 60) throw new Error("locator page was not merged: " + detailDoc.candidates.length);
  if (candidatePages !== 2) throw new Error("candidatePages was not raised to the located page: " + candidatePages);
  if (repainted !== 1) throw new Error("the panel was not repainted");
  await applyPendingCandidate("c0", "run");
  if (selected !== "c0") throw new Error("the deep-linked candidate was not selected");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_feedback_deep_link_outside_the_first_page_locates_renders_and_loads_the_record():
    """@brief `seq=15` 不在前 10 条时：定位页、重绘宿主、再取全文并渲染。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "mergeFeedbackPage", "ensureFeedbackPageFor", "loadFeedbackRecord")) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0, candidateContentEpoch = 0, feedbackContentEpoch = 0;
const FEEDBACK_PAGE = 10;
let feedbackNotice = "";
let currentSeq = 0;
const record = (seq) => ({seq: seq, label: "c" + seq, claims: [], claim_counts: {total: 0}, truncated_sections: []});
function revealFeedbackRecord() {}
let detailDoc = {id: "run", feedback: {records: Array.from({length: 10}, (_, i) => record(i + 1)), total: 30, next: "cur1"}};
let detailId = "run", feedbackPages = 1, openPanels = new Set(["feedback"]);
let panelStamps = {feedback: "s1"}, feedbackRecordToken = 0, reviewsCalibration = null;
const fullFeedback = new Map();
const hosts = {"panel-feedback": {innerHTML: ""}};
let repainted = 0;
function repaintFeedbackPanel(doc) {
  repainted += 1;
  for (const item of doc.feedback.records) hosts["feedback-full-" + item.seq] = hosts["feedback-full-" + item.seq] || {innerHTML: ""};
}
function pickFeedbackFull() { return null; }
function feedbackFullMarkup(item) { return "FULL:" + item.seq; }
function panelUrl(id, name) { return "/api/v2/runs/" + id + "/" + name; }
const calls = [];
globalThis.fetch = async (url) => {
  const text = String(url);
  calls.push(text);
  if (text.includes("/feedback?limit=10&seq=15")) {
    const page = Array.from({length: 10}, (_, i) => record(i + 11));
    return {ok: true, json: async () => ({feedback: {records: page, total: 30, next: "cur2", page_index: 1}, stamp: "s2"})};
  }
  return {ok: true, json: async () => ({record: record(15), calibration: null})};
};
globalThis.document = {getElementById: (id) => hosts[id] || null};
(async () => {
  await loadFeedbackRecord(15, "run");
  if (calls.length !== 2) throw new Error("expected locator + record requests, got " + JSON.stringify(calls));
  if (!calls[0].includes("seq=15")) throw new Error("the locator request is missing: " + calls[0]);
  if (detailDoc.feedback.records.length !== 20) throw new Error("the located page was not merged");
  if (detailDoc.feedback.next !== "cur1") throw new Error("the locator clobbered the load-more cursor: " + detailDoc.feedback.next);
  if (feedbackPages !== 2 || repainted !== 1) throw new Error("the feedback panel was not moved to the located page");
  if (hosts["feedback-full-15"].innerHTML !== "FULL:15") throw new Error("the full record was not rendered: " + hosts["feedback-full-15"].innerHTML);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_collapsed_parent_group_keeps_every_review_card_unmounted():
    """@brief 父 hw/task 收起时，run 自己 open 也不能挂卡片（刷新/upsert 共用这条判断）。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "reviewFoldAction", "reviewRunSlice", "reviewAncestorsOpen", "reviewMountItems")) + r'''
const REVIEW_RUN_PAGE = 20;
const boxes = {"review-hw": {open: false}, "review-task": {open: true}};
const section = (kind, parent) => ({
  parentElement: parent,
  matches: (selector) => selector.split(",").some((part) => part.trim() === "." + kind),
  querySelector: () => boxes[kind],
});
const hw = section("review-hw", null);
const task = section("review-task", hw);
const run = {parentElement: task};
const items = Array.from({length: 45}, (_, i) => ({seq: i}));
if (reviewAncestorsOpen(run)) throw new Error("a collapsed hw must block the run");
if (reviewMountItems(items, true, reviewAncestorsOpen(run), 1).length !== 0) throw new Error("cards mounted under a collapsed hw");
boxes["review-hw"].open = true;
if (!reviewAncestorsOpen(run)) throw new Error("an open hw must allow the run");
boxes["review-task"].open = false;
if (reviewAncestorsOpen(run)) throw new Error("a collapsed task must block the run");
if (reviewMountItems(items, true, reviewAncestorsOpen(run), 1).length !== 0) throw new Error("cards mounted under a collapsed task");
boxes["review-task"].open = true;
if (reviewMountItems(items, true, reviewAncestorsOpen(run), 1).length !== 20) throw new Error("an open chain must mount one page");
if (reviewMountItems(items, false, true, 1).length !== 0) throw new Error("a closed run must mount nothing");
console.log("ok");
'''
    _node(script)
    assert "reviewMountItems(items, !!details.open, reviewAncestorsOpen(run), reviewRunPage(runId))" in page
    assert "if (details && details.open && reviewAncestorsOpen(run))" in page
    assert "for (const runId of touched) paintReviewRun(runId);" in page


def test_paint_review_run_mounts_zero_cards_when_a_parent_is_collapsed():
    """@brief 真跑 `paintReviewRun` 的挂载分支：父组收起时 cards 容器保持零张卡。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "reviewFoldAction", "reviewRunSlice", "reviewAncestorsOpen", "reviewMountItems", "paintReviewRun")) + r'''
const REVIEW_RUN_PAGE = 20;
const boxes = {"review-hw": {open: false}, "review-task": {open: false}};
const detailsOf = (kind) => ({open: kind === "review" ? true : boxes[kind].open});
const section = (kind, parent) => ({
  parentElement: parent,
  matches: (selector) => selector.split(",").some((part) => part.trim() === "." + kind),
  querySelector: (selector) => (selector.includes("details") ? detailsOf(kind) : null),
});
const hw = section("review-hw", null);
const task = section("review-task", hw);
const cards = {children: [], replaceChildren() { this.children = []; }, insertAdjacentHTML() {}};
const run = {
  parentElement: task,
  dataset: {run: "run"},
  querySelector: (selector) => (selector.includes("details") ? detailsOf("review") : (selector === ".review-cards" ? cards : null)),
};
const host = {querySelector: () => run};
globalThis.document = {getElementById: (id) => (id === "reviews-list" ? host : null)};
globalThis.CSS = {escape: (value) => String(value)};
const items = Array.from({length: 45}, (_, index) => ({seq: index}));
function reviewRunItems() { return items; }
function reviewRunPage() { return 1; }
function placeCard(item) { cards.children.push(item); }
function paintReviewCounts() {}
function reviewMoreMarkup() { return "more"; }
const similarLoads = [];
function loadReviewSimilar(runId) { similarLoads.push(runId); }

paintReviewRun("run");
if (cards.children.length !== 0) throw new Error("cards mounted under a collapsed hw: " + cards.children.length);
if (similarLoads.length !== 0) throw new Error("similar must not be fetched under a collapsed hw");
boxes["review-hw"].open = true;
paintReviewRun("run");
if (cards.children.length !== 0) throw new Error("cards mounted under a collapsed task: " + cards.children.length);
if (similarLoads.length !== 0) throw new Error("similar must not be fetched under a collapsed task");
boxes["review-task"].open = true;
paintReviewRun("run");
if (cards.children.length !== 20) throw new Error("an open chain must mount one page, got " + cards.children.length);
if (similarLoads.join(",") !== "run") throw new Error("an open chain must lazily fetch that run's similar once: " + similarLoads);
console.log("ok");
'''
    _node(script)


def test_the_candidate_renderer_keeps_the_server_newest_first_order():
    """@brief 服务端返回 c59…c10 时，`candidatesMarkup` 的首行必须是 c59，不能再被 reverse。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "esc", "num", "newestCandidates", "paginateGroup", "filterOptions", "candidatesFilters", "candidatesMarkup")) + r'''
const CANDIDATES_PAGE = 50;
const CANDIDATE_STATUSES = [["", "全部状态"]];
const CANDIDATE_STAGES = [["", "全部阶段"]];
let candidatePages = 1, candidateStatus = "", candidateStage = "", selectedCandidate = "";
let candidateNotice = "";
function seedLabel() { return ""; }
function confirmWinnerLabel() { return ""; }
function marksFor() { return ""; }
const row = (index) => ({idx: index, label: "c" + index, status: "ok", stage: "bench", speedup: 1, message: ""});
const serverOrder = Array.from({length: 50}, (_, index) => row(59 - index));
const html = candidatesMarkup({candidates: serverOrder, candidatesTotal: 60, candidatesNext: "cur"});
const labels = [...html.matchAll(/data-label="([^"]+)"/g)].map((match) => match[1]);
if (labels[0] !== "c59") throw new Error("first row must be the newest candidate, got " + labels[0] + " (" + labels.slice(0, 3).join(",") + ")");
if (labels[labels.length - 1] !== "c10") throw new Error("first page must end at c10, got " + labels[labels.length - 1]);
console.log("ok");
'''
    _node(script)


def test_a_loaded_but_hidden_candidate_is_expanded_and_rendered():
    """@brief 预载 60 候选/页数 1 时，`c0` 已加载但不可见：必须扩页重绘并渲染，不能直接早退。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "esc", "num", "newestCandidates", "paginateGroup", "filterOptions", "candidatesFilters",
        "candidatesMarkup", "ensureCandidateVisible")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const CANDIDATES_PAGE = 50;
const CANDIDATE_STATUSES = [["", "全部状态"]];
const CANDIDATE_STAGES = [["", "全部阶段"]];
let candidateStatus = "", candidateStage = "", selectedCandidate = "", candidatePages = 1;
let candidateNotice = "";
function seedLabel() { return ""; }
function confirmWinnerLabel() { return ""; }
function marksFor() { return ""; }
const row = (index) => ({idx: index, label: "c" + index, status: "ok", stage: "bench", speedup: 1, message: ""});
const all = Array.from({length: 60}, (_, index) => row(59 - index));
let detailDoc = {id: "run", candidates: all, candidatesTotal: 60, candidatesNext: null};
let detailId = "run", openPanels = new Set(["candidates"]);
let rendered = "";
function repaintCandidatePanel(doc) { rendered = candidatesMarkup(doc); }
let fetches = 0;
globalThis.fetch = async () => { fetches += 1; return {ok: false, json: async () => ({})}; };
function panelUrl() { return "/x"; }
(async () => {
  const found = await ensureCandidateVisible("c0");
  if (!found) throw new Error("the loaded candidate was not found");
  if (fetches !== 0) throw new Error("a loaded candidate must not refetch");
  if (candidatePages !== 2) throw new Error("candidatePages was not expanded: " + candidatePages);
  if (!rendered.includes('data-label="c0"')) throw new Error("c0 is still not rendered");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_loaded_but_hidden_feedback_record_is_rendered_before_the_full_request():
    """@brief 预载 30 反馈/页数 1 时，seq=15 已加载但宿主不存在：扩页重绘后再取全文。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "mergeFeedbackPage", "ensureFeedbackPageFor", "loadFeedbackRecord")) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0, candidateContentEpoch = 0, feedbackContentEpoch = 0;
const FEEDBACK_PAGE = 10;
let feedbackNotice = "";
let currentSeq = 0;
const record = (seq) => ({seq: seq, label: "c" + seq, claims: [], claim_counts: {total: 0}, truncated_sections: []});
function revealFeedbackRecord() {}
const all = Array.from({length: 30}, (_, index) => record(index + 1));
let detailDoc = {id: "run", feedback: {records: all, total: 30, next: "cur1"}};
let detailId = "run", feedbackPages = 1, openPanels = new Set(["feedback"]);
let panelStamps = {feedback: "s1"}, feedbackRecordToken = 0, reviewsCalibration = null;
const fullFeedback = new Map();
const hosts = {"panel-feedback": {innerHTML: ""}};
function repaintFeedbackPanel(doc) {
  for (const item of doc.feedback.records.slice(0, feedbackPages * FEEDBACK_PAGE)) {
    hosts["feedback-full-" + item.seq] = hosts["feedback-full-" + item.seq] || {innerHTML: ""};
  }
}
function pickFeedbackFull() { return null; }
function feedbackFullMarkup(item) { return "FULL:" + item.seq; }
function panelUrl(id, name) { return "/api/v2/runs/" + id + "/" + name; }
const calls = [];
globalThis.fetch = async (url) => {
  calls.push(String(url));
  return {ok: true, json: async () => ({record: record(15), calibration: null})};
};
globalThis.document = {getElementById: (id) => hosts[id] || null};
(async () => {
  await loadFeedbackRecord(15, "run");
  if (calls.length !== 1) throw new Error("expected exactly the full-record request, got " + JSON.stringify(calls));
  if (!calls[0].endsWith("/feedback/15")) throw new Error("wrong request: " + calls[0]);
  if (feedbackPages !== 2) throw new Error("feedbackPages was not expanded: " + feedbackPages);
  if (hosts["feedback-full-15"].innerHTML !== "FULL:15") throw new Error("the record was not rendered");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_seq_deep_link_renders_the_full_record_open_and_does_not_request_twice():
    """@brief seq 深链：全文 details 必须是 open；程序化展开不能再触发第二次全文请求。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    compact = "\n".join(_extract(page, name) for name in (
        "esc", "num", "paginateGroup", "revealFeedbackRecord", "loadFeedbackRecord")) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const FEEDBACK_PAGE = 10;
let feedbackPages = 2, feedbackNotice = "", currentSeq = 15;
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
let detailId = "run", panelStamps = {feedback: "s1"}, feedbackRecordToken = 0, reviewsCalibration = null;
let detailDoc = {id: "run", feedback: {records: [], total: 30, next: null}};
const fullFeedback = new Map();
function pickFeedbackFull() { return null; }
function feedbackFullMarkup(item) { return "FULL:" + item.seq; }
function panelUrl(id) { return "/api/v2/runs/" + id + "/feedback/4"; }
let fetches = 0;
globalThis.fetch = async () => {
  fetches += 1;
  return {ok: true, json: async () => ({record: {seq: 4, claims: []}, calibration: null})};
};
const details = {dataset: {feedbackSeq: "4"}, open: false};
const host = {innerHTML: "", closest: () => details, scrollIntoView() {}};
globalThis.document = {getElementById: (id) => (id === "feedback-full-4" ? host : null)};
(async () => {
  await loadFeedbackRecord(4, "run");
  if (host.innerHTML !== "FULL:4") throw new Error("the full record was not rendered");
  if (fetches !== 1) throw new Error("unexpected number of full-record requests: " + fetches);
  if (!details.open) throw new Error("the deep-linked details was left collapsed");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(compact)


def test_revealing_a_record_suppresses_its_own_toggle_but_not_a_user_open():
    """@brief 程序化展开 details 要抑制自己触发的 toggle；用户手动展开仍走一次加载。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in ("suppressToggle", "revealFeedbackRecord", "bindPanels")) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const suppressedFeedbackToggles = new Map();
let openPanels = new Set(), currentPanel = "", currentSeq = 0, detailId = "run";
let loadCalls = 0;
function loadFeedbackRecord() { loadCalls += 1; }
function commitUrl() {}
function currentViewState() { return {}; }
function loadPanel() {}
function refreshTurns() {}
function refreshLive() {}
function loadSessionCosts() {}
function esc(value) { return String(value ?? ""); }
let toggleHandler = null;
const detailHost = {dataset: {}, addEventListener: (_type, handler) => { toggleHandler = handler; }};
function makeDetails(seq) {
  const details = {
    dataset: {feedbackSeq: String(seq)}, _open: false,
    set open(value) {
      if (this._open === value) return;
      this._open = value;
      queueMicrotask(() => toggleHandler({target: this}));
    },
    get open() { return this._open; },
    closest() { return this; },
  };
  return details;
}
let currentDetails = makeDetails(4);
const host = {innerHTML: "", closest: () => currentDetails, scrollIntoView() {}};
globalThis.document = {getElementById: (id) => (id === "feedback-full-4" ? host : null)};
bindPanels(detailHost);
(async () => {
  revealFeedbackRecord(4);
  for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve));
  if (!currentDetails.open) throw new Error("reveal did not open the details");
  if (loadCalls !== 0) throw new Error("programmatic reveal triggered an extra load: " + loadCalls);
  await new Promise((resolve) => setTimeout(resolve, 0));
  if (suppressedFeedbackToggles.size !== 0) throw new Error("the suppression marker was not consumed");
  currentDetails = makeDetails(4);
  currentDetails.open = true;
  for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve));
  if (loadCalls !== 1) throw new Error("a user open must load once, got " + loadCalls);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_an_unknown_candidate_or_feedback_seq_shows_an_explicit_empty_state():
    """@brief 失效深链要给出明确空态（找不到候选/找不到反馈记录），不能静默返回。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    candidate = "\n".join(_extract(page, name) for name in (
        "esc", "num", "newestCandidates", "paginateGroup", "filterOptions", "candidatesFilters",
        "candidatesMarkup", "setCandidateNotice", "ensureCandidateVisible", "applyPendingCandidate")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const CANDIDATES_PAGE = 50;
const CANDIDATE_STATUSES = [["", "全部状态"]];
const CANDIDATE_STAGES = [["", "全部阶段"]];
let candidateStatus = "", candidateStage = "", selectedCandidate = "", candidatePages = 1;
let candidateNotice = "";
function seedLabel() { return ""; }
function confirmWinnerLabel() { return ""; }
function marksFor() { return ""; }
let detailDoc = {id: "run", candidates: [{idx: 0, label: "c0", status: "ok", stage: "bench", speedup: 1, message: ""}], candidatesTotal: 1, candidatesNext: null};
let detailId = "run", openPanels = new Set(["candidates"]);
let rendered = "";
function repaintCandidatePanel(doc) { rendered = candidatesMarkup(doc); }
function panelUrl() { return "/x"; }
globalThis.fetch = async () => ({ok: true, json: async () => ({candidates: [], total: 1, page_index: null, candidate_marks: []})});
(async () => {
  const found = await ensureCandidateVisible("missing");
  if (found) throw new Error("an unknown candidate must not be treated as found");
  if (!candidateNotice.includes("找不到候选 missing")) throw new Error("missing candidate notice: " + candidateNotice);
  if (!rendered.includes("找不到候选 missing")) throw new Error("the empty state was not rendered");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(candidate)

    feedback = "\n".join(_extract(page, name) for name in (
        "esc", "num", "paginateGroup", "mergeFeedbackPage", "feedbackCompactMarkup", "ensureFeedbackPageFor")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const FEEDBACK_PAGE = 10;
let feedbackPages = 1, feedbackNotice = "", currentSeq = 0;
const record = (seq) => ({seq: seq, label: "c" + seq, claims: [], claim_counts: {total: 0}, truncated_sections: []});
let detailDoc = {id: "run", feedback: {records: [record(1)], total: 1, next: null}};
let detailId = "run", openPanels = new Set(["feedback"]);
let rendered = "";
function repaintFeedbackPanel(doc) { rendered = feedbackCompactMarkup(doc); }
function panelUrl() { return "/x"; }
globalThis.fetch = async () => ({ok: true, json: async () => ({feedback: {records: [], total: 1, next: null, page_index: null}, stamp: "s"})});
(async () => {
  const found = await ensureFeedbackPageFor(99, "run");
  if (found) throw new Error("an unknown seq must not be treated as found");
  if (!feedbackNotice.includes("找不到反馈记录 99")) throw new Error("missing seq notice: " + feedbackNotice);
  if (!rendered.includes("找不到反馈记录 99")) throw new Error("the empty state was not rendered");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(feedback)


def test_the_deep_link_sequence_loads_the_full_record_once_across_panel_and_toggle():
    """@brief 串起 markup → 面板完成 → Chromium 式 toggle：全文只由主路径加载一次。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    functions = "\n".join(_extract(page, name) for name in (
        "esc", "num", "paginateGroup", "panelKey", "consumePendingPanel", "feedbackCompactMarkup", "loadPanel", "bindPanels"))
    script = functions + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const FEEDBACK_PAGE = 10;
let feedbackPages = 1, feedbackNotice = "", currentSeq = 4;
const record = (seq) => ({seq: seq, label: "c" + seq, claims: [], claim_counts: {total: 0}, truncated_sections: []});
let detailDoc = {id: "run", feedback: {records: [record(4)], total: 1, next: null}};
let detailId = "run", panelStamps = {}, panelTokens = {};
const panelInflight = new Map();
let panelRequestSeq = 0;
let pendingPanelOwner = "feedback";
let pendingSeq = 4, pendingCandidate = "", openPanels = new Set(["feedback"]);
let currentPanel = "", fullFeedback = new Map();
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
const loadCalls = [];
function loadFeedbackRecord(seq) { loadCalls.push(seq); }
function bindCandidatePicks() {}
function mergePanel(name, doc, payload) { if (name === "feedback") doc.feedback = payload.feedback || {}; }
function panelLabel(name) { return name; }
function panelQuery() { return "limit=10"; }
function panelUrl(id, name) { return "/api/v2/runs/" + id + "/" + name; }
function panelBody() { return feedbackCompactMarkup(detailDoc); }
function commitUrl() {}
function currentViewState() { return {}; }
function loadPanelStub() {}
let toggleHandler = null;
const detailHost = {dataset: {}, addEventListener: (_type, handler) => { toggleHandler = handler; }};
const host = {
  _html: "", dataset: {}, innerHTML: "",
  classList: {add() {}, remove() {}, toggle() {}, contains() { return false; }},
  get innerHTML() { return this._html; },
  set innerHTML(value) {
    this._html = value;
    // 模拟 Chromium：插入带 open 的 details 会派发 toggle。
    if (/<details[^>]*data-feedback-seq="4"[^>]*\bopen\b/.test(value)) {
      toggleHandler({target: {dataset: {feedbackSeq: "4"}, open: true}});
    }
  },
};
globalThis.document = {
  getElementById: (id) => (id === "panel-feedback" ? host : null),
  querySelector: () => null,
};
globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({stamp: "f1", feedback: {records: [record(4)], total: 1, next: null}}),
});
bindPanels(detailHost);
(async () => {
  await loadPanel("feedback", "run");
  if (loadCalls.length !== 1) throw new Error("the full record was loaded " + loadCalls.length + " times: " + JSON.stringify(loadCalls));
  if (loadCalls[0] !== 4) throw new Error("wrong record: " + JSON.stringify(loadCalls));
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_503_feedback_locator_sets_the_failure_empty_state():
    """@brief 定位请求非 2xx 不能静默：必须写「读取失败」并重绘。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "esc", "num", "paginateGroup", "mergeFeedbackPage", "feedbackCompactMarkup", "ensureFeedbackPageFor")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const FEEDBACK_PAGE = 10;
let feedbackPages = 1, feedbackNotice = "", currentSeq = 0;
const record = (seq) => ({seq: seq, label: "c" + seq, claims: [], claim_counts: {total: 0}, truncated_sections: []});
let detailDoc = {id: "run", feedback: {records: [record(1)], total: 1, next: null}};
let detailId = "run", openPanels = new Set(["feedback"]), rendered = "";
function repaintFeedbackPanel(doc) { rendered = feedbackCompactMarkup(doc); }
function panelUrl() { return "/x"; }
globalThis.fetch = async () => ({ok: false, status: 503, json: async () => ({})});
(async () => {
  const found = await ensureFeedbackPageFor(99, "run");
  if (found) throw new Error("503 must not be treated as found");
  if (!feedbackNotice.includes("读取失败")) throw new Error("missing failure notice: " + feedbackNotice);
  if (!rendered.includes("读取失败")) throw new Error("the failure empty state was not rendered");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_stale_locator_responses_do_not_write_into_a_new_run():
    """@brief 旧 run 的定位响应（含 503/空页）在切 run 后必须直接丢弃，不得污染新 run。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()

    candidate = "\n".join(_extract(page, name) for name in (
        "esc", "num", "newestCandidates", "paginateGroup", "filterOptions", "candidatesFilters",
        "candidatesMarkup", "setCandidateNotice", "ensureCandidateVisible", "applyPendingCandidate")) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0, candidateContentEpoch = 0, feedbackContentEpoch = 0;
const CANDIDATES_PAGE = 50;
const CANDIDATE_STATUSES = [["", "全部状态"]];
const CANDIDATE_STAGES = [["", "全部阶段"]];
let candidateStatus = "", candidateStage = "", selectedCandidate = "old", candidatePages = 1;
let candidateNotice = "";
function seedLabel() { return ""; }
function confirmWinnerLabel() { return ""; }
function marksFor() { return ""; }
let detailDoc = {id: "A", candidates: [], candidatesTotal: 0, candidatesNext: null};
let detailId = "A", openPanels = new Set(["candidates"]), repaints = 0;
function repaintCandidatePanel() { repaints += 1; }
function panelUrl(id) { return "/api/v2/runs/" + id + "/candidates"; }
let resolveFetch = null;
globalThis.fetch = () => new Promise((resolve) => { resolveFetch = resolve; });
(async () => {
  selectedCandidate = "old";
  const pending = applyPendingCandidate("old", "A", navigationEpoch);
  await new Promise((resolve) => setImmediate(resolve));
  detailDoc = {id: "B", candidates: [], candidatesTotal: 0, candidatesNext: null};
  detailId = "B";
  selectedCandidate = "new";
  navigationEpoch += 1;
  resolveFetch({ok: true, json: async () => ({candidates: [{idx: 1, label: "old"}], total: 1, page_index: 0})});
  await pending;
  if (candidateNotice !== "") throw new Error("a stale response wrote a notice into the new run: " + candidateNotice);
  if (repaints !== 0) throw new Error("a stale response repainted the new panel: " + repaints);
  if (selectedCandidate !== "new") throw new Error("a stale response overwrote the new selection: " + selectedCandidate);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(candidate)

    feedback = "\n".join(_extract(page, name) for name in (
        "esc", "num", "paginateGroup", "mergeFeedbackPage", "feedbackCompactMarkup", "ensureFeedbackPageFor")) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0, candidateContentEpoch = 0, feedbackContentEpoch = 0;
const FEEDBACK_PAGE = 10;
let feedbackPages = 1, feedbackNotice = "", currentSeq = 0;
let detailDoc = {id: "A", feedback: {records: [], total: 0, next: null}};
let detailId = "A", openPanels = new Set(["feedback"]), repaints = 0;
function repaintFeedbackPanel() { repaints += 1; }
function panelUrl(id) { return "/api/v2/runs/" + id + "/feedback"; }
let resolveFetch = null;
globalThis.fetch = () => new Promise((resolve) => { resolveFetch = resolve; });
(async () => {
  const pending = ensureFeedbackPageFor(99, "A");
  detailDoc = {id: "B", feedback: {records: [], total: 0, next: null}};
  detailId = "B";
  resolveFetch({ok: false, status: 503, json: async () => ({})});
  const found = await pending;
  if (found) throw new Error("a stale response must not report found");
  if (feedbackNotice !== "") throw new Error("a stale response wrote a notice into the new run: " + feedbackNotice);
  if (repaints !== 0) throw new Error("a stale response repainted the new panel: " + repaints);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(feedback)


def test_restored_panels_and_folds_load_once_across_restore_refresh_and_toggle():
    """@brief 恢复 panel/fold 展开态：restore 的 toggle 被抑制，loadDetail 的刷新各加载一次。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelNames", "planPanelFetches", "refreshOpenPanels", "planDetailRequests",
        "bindPanels", "restoreDetailOpenState")) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
globalThis.CSS = {escape: (value) => String(value)};
let openPanels = new Set(["candidates"]), turnsOpen = true, piOpen = false, costsOpen = true, handoffOpen = false;
let detailId = "run", currentPanel = "", pendingPanel = "", pendingPanelOwner = "";
const calls = {panels: [], turns: 0, live: 0, costs: 0};
function loadPanel(name) { calls.panels.push(name); }
function refreshTurns() { calls.turns += 1; }
function refreshLive() { calls.live += 1; }
function loadSessionCosts() { calls.costs += 1; }
function commitUrl() {}
function currentViewState() { return {}; }
function esc(value) { return String(value ?? ""); }
let toggleHandler = null;
const panelBox = {dataset: {panel: "candidates"}, _open: false};
const turnsBox = {dataset: {lazy: "turns"}, _open: false};
const costsBox = {dataset: {lazy: "costs"}, _open: false};
for (const box of [panelBox, turnsBox, costsBox]) {
  Object.defineProperty(box, "open", {
    get() { return this._open; },
    set(value) {
      if (this._open === value) return;
      this._open = value;
      queueMicrotask(() => toggleHandler({target: this}));
    },
  });
}
const host = {
  querySelector: (selector) => {
    if (selector.includes('data-panel="candidates"')) return panelBox;
    if (selector.includes('data-lazy="turns"')) return turnsBox;
    if (selector.includes('data-lazy="costs"')) return costsBox;
    return null;
  },
  addEventListener: (_type, handler) => { toggleHandler = handler; },
  dataset: {},
};
bindPanels(host);
restoreDetailOpenState(host);
refreshOpenPanels("run");
const plan = planDetailRequests({hasRun: true, turnsOpen, piOpen, costsOpen});
if (plan.turns) refreshTurns();
if (plan.live) refreshLive();
if (plan.costs) loadSessionCosts();
(async () => {
  for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setImmediate(resolve));
  if (!panelBox.open || !turnsBox.open || !costsBox.open) throw new Error("restore did not open the restored blocks");
  if (JSON.stringify(calls.panels) !== JSON.stringify(["candidates"])) throw new Error("panel loaded more than once: " + JSON.stringify(calls.panels));
  if (calls.turns !== 1 || calls.costs !== 1 || calls.live !== 0) throw new Error("lazy blocks loaded more than once: " + JSON.stringify(calls));
  await new Promise((resolve) => setTimeout(resolve, 0));
  if (suppressedFeedbackToggles.size !== 0) throw new Error("restore toggles were not consumed");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_full_restore_closes_old_panels_for_same_summary_and_fresh_detail():
    """@brief 同 run 完整恢复空 panels：same=true/false 都要关闭旧面板；深链面板只加载一次并消费候选/seq。

    宿主模拟 Chromium 的任务合并：同一 tick 的关/开只派发一个 toggle，并在派发时读取最终 open。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    apply_block = page[page.index("function applyViewState("):page.index("function applyPendingDetail(")]
    load_block = page[page.index("async function loadDetail(id)"):page.index("function ledgerTable(")]
    functions = "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "panelNames", "planPanelFetches", "resetPanels",
        "planDetailRequests", "restoreDetailOpenState", "bindPanels", "applyPendingDetail",
        "panelKey", "panelQuery", "panelUrl", "consumePendingPanel", "loadPanel")) + \
        "\n" + _extract(page, "refreshOpenPanels") + "\n" + apply_block + "\n" + load_block
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + functions + r"""
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
globalThis.CSS = {escape: (value) => String(value)};
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let navigationEpoch = 0;
let sideView = "runs", visible = "detail";
let selectedRun = "", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "", currentCluster = "", currentSeq = 0;
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0, pendingPanelOwner = "";
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let restoredDetailView = null;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let candidateNotice = "", feedbackNotice = "";
let openPanels = new Set(["candidates"]);
let runsSort = "started";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
let detailToken = 0, detailDoc = {id: "run"}, detailId = "run", detailStamp = "d1";
let panelStamps = {}, panelTokens = {}, fullFeedback = new Map();
const panelInflight = new Map();
let panelRequestSeq = 0;
let candidatePageToken = 0, feedbackPageToken = 0, feedbackRecordToken = 0, feedbackClaimToken = 0;
let currentDetailMode = "same";
const calls = {turns: 0, live: 0, costs: 0, patches: 0, panelFetches: 0, candidates: [], seqs: []};
let toggleHandler = null;
function makeBox(attrs, open) {
  const box = {dataset: attrs, _open: Boolean(open), _queued: false};
  Object.defineProperty(box, "open", {
    get() { return this._open; },
    set(value) {
      if (this._open === value) return;
      this._open = value;
      // Chromium 合并同一 tick 的多次变化：只调度一次任务，派发时读取最终 open。
      if (this._queued) return;
      this._queued = true;
      queueMicrotask(() => {
        this._queued = false;
        if (toggleHandler) toggleHandler({target: this});
      });
    },
  });
  return box;
}
const boxes = {candidates: makeBox({panel: "candidates"}, true)};
const panelHost = {innerHTML: "", classList: {add() {}, remove() {}, contains() { return false; }}};
const host = {
  dataset: {}, innerHTML: "",
  classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}},
  addEventListener: (_type, handler) => { toggleHandler = handler; },
  querySelectorAll: (selector) => (String(selector).includes("data-panel") ? Object.values(boxes) : []),
  querySelector: (selector) => (String(selector).includes('data-panel="candidates"') ? boxes.candidates : null),
};
const emptyHosts = {};
globalThis.location = {pathname: "/"};
globalThis.history = {pushState() {}, replaceState() {}};
globalThis.document = {
  getElementById: (id) => {
    if (id === "view-detail") return host;
    if (id === "panel-candidates") return panelHost;
    return emptyHosts[id] = emptyHosts[id] || {
      value: "", innerHTML: "", hidden: false,
      classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}}, setAttribute() {},
      addEventListener: (_type, handler) => { emptyHosts[id].on_click = handler; },
    };
  },
  querySelector: (selector) => (String(selector).includes('data-panel="candidates"') ? boxes.candidates : null),
  querySelectorAll: () => [],
};
globalThis.fetch = async (url) => {
  const text = String(url);
  if (text.includes("/candidates?") || text.endsWith("/candidates")) {
    calls.panelFetches += 1;
    return {ok: true, json: async () => ({candidates: [], total: 0, next: null, stamp: "p1", candidate_marks: []})};
  }
  if (currentDetailMode === "same") return {ok: true, json: async () => ({same: true, detail_stamp: "d1"})};
  return {ok: true, json: async () => ({id: "run", detail_stamp: "d1", candidates: [], feedback: {records: []}})};
};
function showSection() {}
function show() {}
function showRight() { loadDetail("run"); }
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}
function rememberDetailScroll() {}
function recallDetailScroll() {}
function detailScroller() { return null; }
function detailPatchedRun() { return ""; }
function clearDetailPatch() {}
function overlayClose() {}
function patchDetail() { calls.patches += 1; }
function carryPanels() {}
function paintDetailHead() {}
function bindCandidatePicks() {}
function bindFeedback() {}
function openStage() {}
function selectCandidate(label) { calls.candidates.push(label); }
function applyPendingCandidate(label) { calls.candidates.push(label); }
function loadFeedbackRecord(seq) { calls.seqs.push(seq); }
function mergePanel(name, doc, payload) { if (name === "candidates") doc.candidates = payload.candidates || []; }
function panelBody() { return "BODY"; }
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
function refreshTurns() { calls.turns += 1; }
function refreshLive() { calls.live += 1; }
function loadSessionCosts() { calls.costs += 1; }
function commitUrl() {}

bindPanels(host);
const settle = async () => {
  for (let i = 0; i < 10; i += 1) await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setTimeout(resolve, 0));
};
(async () => {
  for (const mode of ["same", "fresh"]) {
    currentDetailMode = mode;
    // 1) 无深链面板：旧 candidates 必须关闭，且不触发任何加载。
    detailDoc = {id: "run"};
    detailId = "run";
    detailStamp = "d1";
    openPanels = new Set(["candidates"]);
    boxes.candidates._open = true;
    boxes.candidates._queued = false;
    calls.panelFetches = 0;
    calls.candidates = [];
    calls.seqs = [];
    calls.turns = calls.live = calls.costs = 0;
    applyViewState(readUrl("?run=run&panel=&panels="), {full: true});
    await settle();
    if (boxes.candidates.open) throw new Error(mode + ": the old panel stayed open");
    if (openPanels.size !== 0) throw new Error(mode + ": openPanels was not cleared");
    if (calls.panelFetches !== 0) throw new Error(mode + ": closed panel triggered requests: " + calls.panelFetches);
    if (calls.turns || calls.costs) throw new Error(mode + ": lazy blocks requested while closed");
    if (suppressedFeedbackToggles.size !== 0) throw new Error(mode + ": suppress markers were not consumed");

    // 2) 深链 candidates：restore 关旧节点、applyPendingDetail 再开（浏览器合并为一个 toggle），
    //    主面板明确加载一次，候选与 seq 都由面板完成路径消费。
    detailDoc = {id: "run"};
    detailId = "run";
    detailStamp = "d1";
    openPanels = new Set(["candidates"]);
    boxes.candidates._open = true;
    boxes.candidates._queued = false;
    calls.panelFetches = 0;
    calls.candidates = [];
    calls.seqs = [];
    applyViewState(readUrl("?run=run&panel=candidates&candidate=c1&seq=7&panels="), {full: true});
    await settle();
    if (!boxes.candidates.open) throw new Error(mode + ": the deep-linked panel was not opened");
    if (calls.panelFetches !== 1) throw new Error(mode + ": panel fetched " + calls.panelFetches + " times");
    if (JSON.stringify(calls.candidates) !== JSON.stringify(["c1"])) throw new Error(mode + ": candidate was not consumed: " + JSON.stringify(calls.candidates));
    if (JSON.stringify(calls.seqs) !== JSON.stringify([7])) throw new Error(mode + ": seq was not consumed: " + JSON.stringify(calls.seqs));
    if (suppressedFeedbackToggles.size !== 0) throw new Error(mode + ": deep-link suppress markers were not consumed");
  }
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
"""
    _node(script)


def test_a_task_delayed_toggle_does_not_escape_the_suppression():
    """@brief 浏览器 task 型 toggle 晚于程序化 open 到达时，仍被状态 marker 吃掉，不靠 timer 先后。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in ("suppressToggle", "bindPanels")) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const suppressedFeedbackToggles = new Map();
let loadCalls = 0, detailId = "run", currentSeq = 0, currentPanel = "";
let openPanels = new Set(["feedback"]);
function loadFeedbackRecord() { loadCalls += 1; }
function commitUrl() {}
function currentViewState() { return {}; }
function loadPanel() {}
function refreshTurns() {}
function refreshLive() {}
function loadSessionCosts() {}
function esc(value) { return String(value ?? ""); }
let toggleHandler = null;
const detailHost = {dataset: {}, addEventListener: (_type, handler) => { toggleHandler = handler; }};
const details = {dataset: {feedbackSeq: "4"}, open: false, closest() { return this; }};
globalThis.document = {getElementById: () => null};
bindPanels(detailHost);
suppressToggle(details, true);
details.open = true;
setTimeout(() => {
  // 晚于程序化的 task：派发时读取最终 open（Chromium 语义）。
  toggleHandler({target: details});
  if (loadCalls !== 0) throw new Error("a task-delayed programmatic toggle loaded " + loadCalls + " times");
  if (suppressedFeedbackToggles.size !== 0) throw new Error("the marker was not consumed by the real toggle");
  // 用户随后手动打开仍要加载一次。
  toggleHandler({target: details});
  if (loadCalls !== 1) throw new Error("a user toggle must load once, got " + loadCalls);
  console.log("ok");
}, 20);
'''
    _node(script)


def test_a_main_panel_in_flight_is_not_refetched_by_refresh():
    """@brief 主面板在途时 pendingPanel 已清空；refresh 必须用 pendingPanelOwner/在途表去重。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelNames", "planPanelFetches", "panelKey", "panelQuery", "panelUrl", "consumePendingPanel", "refreshOpenPanels", "loadPanel")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "";
let detailId = "run", detailDoc = {id: "run", candidates: []};
let panelStamps = {}, panelTokens = {}, openPanels = new Set(["candidates"]);
const panelInflight = new Map();
let panelRequestSeq = 0;
let pendingPanel = "", pendingPanelOwner = "candidates", pendingCandidate = "", pendingSeq = 0;
let fullFeedback = new Map();
const hosts = {"panel-candidates": {innerHTML: ""}};
globalThis.document = {getElementById: (id) => hosts[id] || null};
let resolveFetch = null;
let fetches = 0;
globalThis.fetch = () => new Promise((resolve) => { fetches += 1; resolveFetch = resolve; });
function mergePanel(name, doc, payload) { if (name === "candidates") doc.candidates = payload.candidates || []; }
function panelBody() { return "BODY"; }
function bindCandidatePicks() {}
function applyPendingCandidate() {}
function loadFeedbackRecord() {}
function esc(value) { return String(value ?? ""); }
function panelLabel(name) { return name; }
(async () => {
  const pendingLoad = loadPanel("candidates", "run");
  refreshOpenPanels("run");
  if (fetches !== 1) throw new Error("refresh refetched the in-flight owner: " + fetches);
  resolveFetch({ok: true, json: async () => ({candidates: [], total: 0, next: null, stamp: "p"})});
  await pendingLoad;
  if (pendingPanelOwner !== "") throw new Error("the owner was not cleared after completion");
  if (fetches !== 1) throw new Error("unexpected extra fetch after completion: " + fetches);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_same_run_navigation_invalidates_delayed_candidate_selection():
    """@brief 同 run 导航（预算 chip/applyViewState）递增代次；旧候选定位响应不得再选中。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in ("newestCandidates", "ensureCandidateVisible", "applyPendingCandidate")) + r'''
function syncDetailSectionCache() {}
const CANDIDATES_PAGE = 50;
let navigationEpoch = 0, candidateContentEpoch = 0, feedbackContentEpoch = 0, candidateNotice = "", candidatePages = 1, selectedCandidate = "";
let detailDoc = {id: "run", candidates: [], candidatesTotal: 1, candidatesNext: null};
let detailId = "run", openPanels = new Set(["candidates"]);
function selectCandidate(label) { selectedCandidate = label; }
function panelUrl(id) { return "/api/v2/runs/" + id + "/candidates"; }
let resolveFetch = null;
globalThis.fetch = () => new Promise((resolve) => { resolveFetch = resolve; });
(async () => {
  selectedCandidate = "old";
  const pending = applyPendingCandidate("old", "run", navigationEpoch);
  navigationEpoch += 1;  // 同 run 的预算 chip 会走 applyViewState，递增同一代次。
  selectedCandidate = "new";
  resolveFetch({ok: true, json: async () => ({candidates: [{idx: 9, label: "old"}], total: 1, page_index: 0})});
  await pending;
  if (selectedCandidate !== "new") throw new Error("a stale candidate response overwrote the new selection: " + selectedCandidate);
  if (candidateNotice !== "") throw new Error("a stale candidate response wrote a notice: " + candidateNotice);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_same_run_navigation_invalidates_a_delayed_feedback_record():
    """@brief 同 run 导航后，延迟的反馈全文不能写入旧宿主，也不能再请求记录。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "ensureFeedbackPageFor", "mergeFeedbackPage", "loadFeedbackRecord")) + r'''
function syncDetailSectionCache() {}
const FEEDBACK_PAGE = 10;
let navigationEpoch = 0, detailId = "run", feedbackPages = 1, feedbackNotice = "", currentSeq = 15;
let detailDoc = {id: "run", feedback: {records: [], total: 0, next: null}};
let panelStamps = {feedback: "s1"}, feedbackRecordToken = 0, reviewsCalibration = null;
const fullFeedback = new Map();
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
const details = {dataset: {feedbackSeq: "15"}, open: false, closest() { return this; }};
const host = {innerHTML: "before", closest: () => details, scrollIntoView() {}};
globalThis.document = {getElementById: (id) => (id === "feedback-full-15" ? host : null)};
function revealFeedbackRecord() {}
function pickFeedbackFull() { return null; }
function feedbackFullMarkup(record) { return "FULL:" + record.seq; }
function panelUrl(id) { return "/api/v2/runs/" + id + "/feedback"; }
let resolveFetch = null;
let fetches = 0;
globalThis.fetch = () => new Promise((resolve) => { fetches += 1; resolveFetch = resolve; });
(async () => {
  const pending = loadFeedbackRecord(15, "run");
  await new Promise((resolve) => setImmediate(resolve));
  navigationEpoch += 1;  // 同 run 导航。
  resolveFetch({ok: true, json: async () => ({record: {seq: 15, claims: []}, calibration: null})});
  await pending;
  if (host.innerHTML !== "before") throw new Error("a stale feedback record rendered: " + host.innerHTML);
  if (fetches !== 1) throw new Error("unexpected extra record request: " + fetches);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_same_panel_payload_does_not_rewrite_the_loaded_dom():
    """@brief loadPanel 收到 same 且面板已加载时不能写 innerHTML；嵌套展开/滚动必须保留。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelKey", "panelLoaded", "consumePendingPanel", "loadPanel")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let detailId = "run", detailDoc = {id: "run", feedback: {records: []}};
let pendingPanelOwner = "", pendingCandidate = "", pendingSeq = 0, navigationEpoch = 0;
let panelStamps = {}, panelTokens = {};
const panelInflight = new Map();
let panelRequestSeq = 0;
const hosts = {"panel-feedback": {_html: "sentinel", get innerHTML() { return this._html; }, set innerHTML(value) { this._html = value; }}};
globalThis.document = {getElementById: (id) => hosts[id] || null};
let recordCalls = 0;
function loadFeedbackRecord() { recordCalls += 1; }
function applyPendingCandidate() {}
function mergePanel() {}
function panelBody() { return "REWRITTEN"; }
function bindCandidatePicks() {}
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
function panelQuery() { return "limit=10"; }
function panelUrl() { return "/x"; }
globalThis.fetch = async () => ({ok: true, json: async () => ({same: true, stamp: "s"})});
(async () => {
  pendingPanelOwner = "feedback";
  pendingSeq = 4;
  await loadPanel("feedback", "run");
  if (hosts["panel-feedback"].innerHTML !== "sentinel") throw new Error("same payload rewrote the panel DOM");
  if (recordCalls !== 1) throw new Error("pending seq was not consumed exactly once: " + recordCalls);
  if (pendingPanelOwner !== "") throw new Error("pending owner was not cleared");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_collapsing_the_runs_panel_unmounts_rows_and_restores_them_from_cache():
    """@brief 收起运行面板要真卸载 tbody 行；展开从缓存重画，不重新请求。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "readRunsScroll", "saveRunsScroll", "restoreRunsScroll", "renderRunsRows", "setRunsCollapsed")) + r'''
let runsRows = [{run_id: "a/run", task: "relu", publication: "confirmed", published_speedup: 1.5, spent_usd: 1, status: "done"}];
let runsRendered = true, runsScrollTop = 0, selectedRun = "";
const scrollStore = new Map();
globalThis.sessionStorage = {getItem: (key) => (scrollStore.has(key) ? scrollStore.get(key) : null), setItem: (key, value) => scrollStore.set(key, String(value))};
const panel = {classList: {_set: new Set(), toggle(name, on) { if (on) this._set.add(name); else this._set.delete(name); }, contains(name) { return this._set.has(name); }}};
const button = {setAttribute() {}, textContent: "", title: ""};
const body = {innerHTML: "<tr>STALE</tr>", dataset: {}, replaceChildren() { this.innerHTML = ""; }, querySelectorAll() { return []; }, addEventListener() {}};
const scroller = {scrollTop: 120};
globalThis.localStorage = {setItem() {}};
globalThis.document = {
  getElementById: (id) => (id === "runs-panel" ? panel : (id === "runs-toggle" ? button : null)),
  querySelector: (selector) => (selector === "#runs tbody" ? body : (selector === ".runs-table-scroll" ? scroller : null)),
};
function esc(value) { return String(value ?? ""); }
function num(value) { return value == null ? "—" : String(value); }
function money(value) { return "$" + String(value); }
function idText(value) { return String(value); }
function applyRovingRows() {}
function rowKeyAction() { return ""; }
function rovingIndex(current) { return current; }
function openDelete() {}
function applyViewState() {}
function commitUrl() {}
function currentViewState() { return {}; }
setRunsCollapsed(true);
if (body.innerHTML !== "") throw new Error("collapsed panel kept stale rows: " + body.innerHTML);
if (runsRendered) throw new Error("collapsed panel is still marked rendered");
setRunsCollapsed(false);
if (!body.innerHTML.includes("a/run")) throw new Error("expand did not restore rows from cache");
if (scroller.scrollTop !== 120) throw new Error("expand lost the scroll position");
console.log("ok");
'''
    _node(script)


def test_a_normal_panel_refresh_does_not_invalidate_its_own_in_flight_response():
    """@brief 普通面板在途时再次调用 loadPanel 只去重，不递增 token；唯一响应仍要更新 DOM。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelKey", "panelQuery", "panelUrl", "consumePendingPanel", "panelLoaded", "loadPanel")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "";
let detailId = "run", detailDoc = {id: "run", candidates: []};
let panelStamps = {}, panelTokens = {};
const panelInflight = new Map();
let panelRequestSeq = 0, pendingPanelOwner = "", pendingCandidate = "", pendingSeq = 0, navigationEpoch = 0;
const host = {innerHTML: "sentinel"};
globalThis.document = {getElementById: (id) => (id === "panel-candidates" ? host : null)};
let resolveFetch = null, fetches = 0;
globalThis.fetch = () => new Promise((resolve) => { fetches += 1; resolveFetch = resolve; });
function mergePanel(name, doc, payload) { if (name === "candidates") doc.candidates = payload.candidates || []; }
function panelBody() { return "BODY"; }
function bindCandidatePicks() {}
function applyPendingCandidate() {}
function loadFeedbackRecord() {}
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
(async () => {
  const first = loadPanel("candidates", "run");
  await new Promise((resolve) => setImmediate(resolve));
  loadPanel("candidates", "run");  // 在途刷新：必须返回但不碰 token
  if (fetches !== 1) throw new Error("in-flight dedupe issued another fetch: " + fetches);
  resolveFetch({ok: true, json: async () => ({candidates: [], total: 0, next: null, stamp: "p"})});
  await first;
  if (host.innerHTML !== "BODY") throw new Error("the only response was discarded: " + host.innerHTML);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_panel_failure_from_an_old_generation_does_not_overwrite_the_panel():
    """@brief 旧代次面板 reject 不能把「读取失败」写进新面板。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelKey", "panelQuery", "panelUrl", "consumePendingPanel", "panelLoaded", "loadPanel")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "";
let detailId = "run", detailDoc = {id: "run", candidates: []}, navigationEpoch = 0;
let panelStamps = {}, panelTokens = {}, pendingPanelOwner = "", pendingCandidate = "", pendingSeq = 0;
const panelInflight = new Map();
let panelRequestSeq = 0;
const host = {innerHTML: "sentinel"};
globalThis.document = {getElementById: (id) => (id === "panel-candidates" ? host : null)};
let rejectFetch = null;
globalThis.fetch = () => new Promise((_resolve, reject) => { rejectFetch = reject; });
function mergePanel() {}
function panelBody() { return "BODY"; }
function bindCandidatePicks() {}
function applyPendingCandidate() {}
function loadFeedbackRecord() {}
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
(async () => {
  const pending = loadPanel("candidates", "run");
  await new Promise((resolve) => setImmediate(resolve));
  navigationEpoch += 1;
  rejectFetch(new Error("old generation failed"));
  await pending;
  if (host.innerHTML !== "sentinel") throw new Error("a stale failure overwrote the panel: " + host.innerHTML);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_an_owner_in_flight_survives_a_same_run_navigation_generation():
    """@brief owner 在途时同 run 导航：新代次必须重新发一次，旧响应丢弃，候选/seq 只消费一次。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelKey", "panelQuery", "panelUrl", "consumePendingPanel", "panelLoaded", "loadPanel")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "";
let detailId = "run", detailDoc = {id: "run", candidates: []}, navigationEpoch = 0;
let panelStamps = {}, panelTokens = {}, panelRequestSeq = 0, pendingCandidate = "", pendingSeq = 0;
let pendingPanelOwner = "candidates";
const panelInflight = new Map();
const host = {innerHTML: ""};
globalThis.document = {getElementById: (id) => (id === "panel-candidates" ? host : null)};
const pending = new Map();
let fetches = 0;
globalThis.fetch = () => new Promise((resolve) => { fetches += 1; pending.set(fetches, resolve); });
function mergePanel(name, doc, payload) { if (name === "candidates") doc.candidates = payload.candidates || []; }
function panelBody() { return "BODY"; }
function bindCandidatePicks() {}
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
const candidates = [];
const seqs = [];
function applyPendingCandidate(label) { candidates.push(label); }
function loadFeedbackRecord(seq) { seqs.push(seq); }
(async () => {
  pendingCandidate = "c1";
  pendingSeq = 7;
  const oldLoad = loadPanel("candidates", "run");
  await new Promise((resolve) => setImmediate(resolve));
  if (fetches !== 1) throw new Error("first load did not start");
  navigationEpoch += 1;  // 同 run 导航
  const newLoad = loadPanel("candidates", "run");
  await new Promise((resolve) => setImmediate(resolve));
  if (fetches !== 2) throw new Error("the new generation did not issue a request: " + fetches);
  // 新代次先完成，消费 pending。
  pending.get(2)({ok: true, json: async () => ({candidates: [], total: 0, next: null, stamp: "new"})});
  await newLoad;
  if (JSON.stringify(candidates) !== JSON.stringify(["c1"])) throw new Error("new generation did not consume the candidate");
  if (JSON.stringify(seqs) !== JSON.stringify([7])) throw new Error("new generation did not consume the seq");
  // 旧代次迟到（成功或失败）都不能覆盖。
  pending.get(1)({ok: true, json: async () => ({candidates: [{label: "old"}], total: 1, next: null, stamp: "old"})});
  await oldLoad;
  if (host.innerHTML !== "BODY") throw new Error("the old generation overwrote the panel");
  if (candidates.length !== 1 || seqs.length !== 1) throw new Error("the old generation consumed pending again");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_feedback_record_reject_from_an_old_generation_does_not_write_the_error():
    """@brief 旧代次全文 reject 不能把「全文读取失败」写进新宿主。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "ensureFeedbackPageFor", "mergeFeedbackPage", "loadFeedbackRecord")) + r'''
function syncDetailSectionCache() {}
const FEEDBACK_PAGE = 10;
let navigationEpoch = 0, detailId = "run", feedbackPages = 1, feedbackNotice = "", currentSeq = 15;
let detailDoc = {id: "run", feedback: {records: [], total: 0, next: null}};
let panelStamps = {feedback: "s1"}, feedbackRecordToken = 0, reviewsCalibration = null;
const fullFeedback = new Map();
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
const host = {innerHTML: "sentinel", closest: () => null, scrollIntoView() {}};
globalThis.document = {getElementById: (id) => (id === "feedback-full-15" ? host : null)};
function revealFeedbackRecord() {}
function pickFeedbackFull() { return null; }
function feedbackFullMarkup(record) { return "FULL:" + record.seq; }
function panelUrl() { return "/x"; }
let rejectFetch = null;
globalThis.fetch = () => new Promise((_resolve, reject) => { rejectFetch = reject; });
(async () => {
  const pending = loadFeedbackRecord(15, "run");
  await new Promise((resolve) => setImmediate(resolve));
  navigationEpoch += 1;
  rejectFetch(new Error("old generation rejected"));
  await pending;
  if (host.innerHTML !== "sentinel") throw new Error("a stale rejection wrote the error: " + host.innerHTML);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_same_run_budget_chip_hands_the_in_flight_panel_to_a_new_request():
    """@brief 真实 chip→applyViewState 在旧主面板在途时清 owner，并让刷新重新发一次；旧响应丢弃。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    marker = 'document.getElementById("budget-banner").addEventListener("click", (event) => {'
    first = page.index(marker)
    second = page.index(marker, first + 1)
    listener = page[second:page.index("});", second) + 3]
    listener = listener.replace('document.getElementById("budget-banner").addEventListener("click", ', "const budgetChipListener = ", 1)
    assert listener.endswith(");")
    listener = listener[:-2] + ";"
    apply_block = page[page.index("function applyViewState("):page.index("function applyPendingDetail(")]
    functions = "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "panelNames", "planPanelFetches",
        "panelKey", "panelQuery", "panelUrl", "consumePendingPanel", "loadPanel")) + \
        "\n" + _extract(page, "refreshOpenPanels") + "\n" + apply_block + "\n" + listener
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + functions + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "";
let detailDoc = {id: "run", candidates: []}, detailId = "run", navigationEpoch = 0;
let panelStamps = {}, panelTokens = {}, panelRequestSeq = 0;
const panelInflight = new Map();
let pendingPanelOwner = "candidates", pendingPanel = "", pendingCandidate = "", pendingSeq = 0;
let openPanels = new Set(["candidates"]);
let selectedRun = "run", sideView = "detail", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "", currentCluster = "", currentSeq = 0, runsSort = "started";
let turnsOpen = false, piOpen = false, costsOpen = false, restoredDetailView = null, handoffOpen = false;
let candidateNotice = "", feedbackNotice = "";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
let visible = "detail";
const panelHost = {innerHTML: "sentinel"};
const emptyHosts = {};
globalThis.document = {
  getElementById: (id) => {
    if (id === "panel-candidates") return panelHost;
    return emptyHosts[id] = emptyHosts[id] || {
      value: "", innerHTML: "", hidden: false,
      classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}}, setAttribute() {},
      addEventListener: (_type, handler) => { emptyHosts[id].on_click = handler; },
    };
  },
  querySelector: () => null,
  querySelectorAll: () => [],
};
globalThis.history = {pushState() {}, replaceState() {}};
globalThis.location = {pathname: "/", search: ""};
const resolvers = [];
let fetches = 0;
globalThis.fetch = () => new Promise((resolve) => { fetches += 1; resolvers.push(resolve); });
function mergePanel(name, doc, payload) { if (name === "candidates") doc.candidates = payload.candidates || []; }
function panelBody(name, doc) { return "BODY:" + JSON.stringify((doc.candidates || []).map((row) => row.label)); }
function bindCandidatePicks() {}
function applyPendingCandidate() {}
function loadFeedbackRecord() {}
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
function resetPanels() { openPanels = new Set(); pendingPanelOwner = ""; }
function showSection() {}
function show() {}
function showRight() { refreshOpenPanels("run"); }
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}
function commitUrl() {}

(async () => {
  const oldLoad = loadPanel("candidates", "run");
  await new Promise((resolve) => setImmediate(resolve));
  if (fetches !== 1) throw new Error("old panel did not start");
  // 真实预算 chip：同一场导航，旧 owner 还在途。
  budgetChipListener({
    preventDefault() {},
    target: {closest: (selector) => (selector === "a[data-run]" ? {dataset: {run: "run"}} : null)},
  });
  if (pendingPanelOwner !== "") throw new Error("the old owner survived navigation: " + pendingPanelOwner);
  await new Promise((resolve) => setImmediate(resolve));
  if (fetches !== 2) throw new Error("navigation did not hand off a new panel request: " + fetches);
  // 新请求先完成，旧请求随后迟到。
  resolvers[1]({ok: true, json: async () => ({candidates: [{label: "new"}], total: 1, next: null, stamp: "new"})});
  await new Promise((resolve) => setImmediate(resolve));
  resolvers[0]({ok: true, json: async () => ({candidates: [{label: "old"}], total: 1, next: null, stamp: "old"})});
  await oldLoad;
  await new Promise((resolve) => setImmediate(resolve));
  if (!panelHost.innerHTML.includes("new") || panelHost.innerHTML.includes("old")) throw new Error("stale panel overwrote the new one: " + panelHost.innerHTML);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_delayed_same_or_fresh_summaries_wait_for_the_panel_owner_before_consuming_deep_links():
    """@brief 生产 applyPendingDetail/loadDetail：主面板在途时 same=true/false 都不能提前消费 candidate/seq。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    load_block = page[page.index("async function loadDetail(id)"):page.index("function ledgerTable(")]
    functions = "\n".join(_extract(page, name) for name in (
        "panelNames", "planPanelFetches", "panelKey", "panelQuery", "panelUrl",
        "consumePendingPanel", "refreshOpenPanels", "loadPanel", "applyPendingDetail")) + "\n" + load_block
    script = functions + r"""
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "";
let detailDoc = {id: "run", candidates: []}, detailId = "run", detailStamp = "d1";
let navigationEpoch = 0, panelStamps = {}, panelTokens = {}, panelRequestSeq = 0;
const panelInflight = new Map();
let pendingPanelOwner = "", pendingPanel = "", pendingCandidate = "", pendingSeq = 0;
let openPanels = new Set();
let detailToken = 0, selectedCandidate = "";
let turnsOpen = false, piOpen = false, costsOpen = false, restoredDetailView = null, handoffOpen = false;
let visible = "detail";
const panelHost = {innerHTML: "sentinel"};
let panelFetches = 0, panelResolvers = [], summaryResolvers = [], candidateCalls = [], seqCalls = [];
const emptyHosts = {};
globalThis.document = {
  getElementById: (id) => {
    if (id === "panel-candidates") return panelHost;
    if (id === "view-detail") return emptyHosts["view-detail"] = emptyHosts["view-detail"] || {innerHTML: "", dataset: {}, classList: {add() {}, remove() {}, contains() { return false; }}};
    return emptyHosts[id] = emptyHosts[id] || {innerHTML: "", dataset: {}, hidden: false, classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}}, addEventListener() {}, setAttribute() {}};
  },
  querySelector: () => null,
  querySelectorAll: () => [],
};
globalThis.fetch = (url) => {
  const text = String(url);
  if (text.includes("/candidates")) return new Promise((resolve) => { panelFetches += 1; panelResolvers.push(resolve); });
  return new Promise((resolve) => { summaryResolvers.push(resolve); });
};
function mergePanel(name, doc, payload) { if (name === "candidates") doc.candidates = payload.candidates || []; }
function panelBody(name, doc) { return "BODY:" + JSON.stringify((doc.candidates || []).map((row) => row.label)); }
function bindCandidatePicks() {}
function applyPendingCandidate(label) { candidateCalls.push(label); }
function loadFeedbackRecord(seq) { seqCalls.push(seq); }
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
function resetPanels() { openPanels = new Set(); pendingPanelOwner = ""; }
function carryPanels() {}
function patchDetail() {}
function bindPanels() {}
function restoreDetailOpenState() {}
function detailPatchedRun() { return "run"; }
function clearDetailPatch() {}
function overlayClose() {}
function detailScroller() { return null; }
function rememberDetailScroll() {}
function recallDetailScroll() {}
function paintDetailHead() {}
function bindFeedback() {}
function openStage() {}
function planDetailRequests() { return {detail: true, turns: false, live: false, costs: false}; }
function refreshTurns() {}
function refreshLive() {}
function loadSessionCosts() {}
function renderDetail() { return ""; }
const tick = () => new Promise((resolve) => setImmediate(resolve));
(async () => {
  for (const mode of ["same", "fresh"]) {
    detailDoc = {id: "run", candidates: []};
    detailId = "run";
    detailStamp = "d1";
    pendingPanel = "candidates";
    pendingPanelOwner = "";
    pendingCandidate = "r3";
    pendingSeq = 4;
    openPanels = new Set();
    panelStamps = {};
    panelTokens = {};
    panelRequestSeq = 0;
    panelInflight.clear();
    panelFetches = 0;
    panelResolvers = [];
    summaryResolvers = [];
    candidateCalls = [];
    seqCalls = [];
    // 主面板先发出并保持延迟。
    applyPendingDetail();
    await tick();
    if (panelFetches !== 1) throw new Error(mode + ": owner panel did not start: " + panelFetches);
    const detailLoad = loadDetail("run");
    await tick();
    if (summaryResolvers.length !== 1) throw new Error(mode + ": summary request missing");
    summaryResolvers[0]({ok: true, json: async () => (mode === "same"
      ? {same: true, detail_stamp: "d1"}
      : {id: "run", detail_stamp: "d2", candidates: [], feedback: {records: []}})});
    await detailLoad;
    await tick();
    if (candidateCalls.length || seqCalls.length) {
      throw new Error(mode + ": deep link was consumed before the owner panel finished: " + JSON.stringify({candidateCalls, seqCalls}));
    }
    if (mode === "fresh" && panelFetches !== 2) throw new Error("fresh summary did not hand off a new owner request: " + panelFetches);
    if (mode === "same" && panelFetches !== 1) throw new Error("same summary refetched the in-flight owner: " + panelFetches);
    panelResolvers[mode === "fresh" ? 1 : 0]({ok: true, json: async () => ({candidates: [{label: "r3"}], total: 1, next: null, stamp: "p"})});
    await tick();
    if (JSON.stringify(candidateCalls) !== JSON.stringify(["r3"])) throw new Error(mode + ": candidate was not consumed exactly once after owner completion: " + JSON.stringify(candidateCalls));
    if (JSON.stringify(seqCalls) !== JSON.stringify([4])) throw new Error(mode + ": feedback record was not consumed exactly once after owner completion: " + JSON.stringify(seqCalls));
    if (mode === "fresh") {
      panelResolvers[0]({ok: true, json: async () => ({candidates: [{label: "old"}], total: 1, next: null, stamp: "old"})});
      await detailLoad;
      await tick();
      if (candidateCalls.length !== 1 || seqCalls.length !== 1) throw new Error("stale owner response consumed the deep link again");
    }
  }
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
"""
    _node(script)


def test_rapid_candidate_filter_changes_issue_a_request_for_each_shape_and_keep_the_last():
    """@brief 连续 status=ok→failed：请求形状进在途键，两个请求都发，旧响应不覆盖新筛选。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelKey", "panelQuery", "panelUrl", "consumePendingPanel", "loadPanel", "applyCandidateFilter")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let detailId = "run", detailDoc = {id: "run", candidates: []}, navigationEpoch = 0;
let panelStamps = {}, panelTokens = {}, panelRequestSeq = 0, pendingPanelOwner = "", pendingCandidate = "", pendingSeq = 0;
let openPanels = new Set(["candidates"]);
const panelInflight = new Map();
const host = {innerHTML: ""};
globalThis.document = {getElementById: (id) => (id === "panel-candidates" ? host : null)};
const resolvers = [];
let fetches = 0;
globalThis.fetch = () => new Promise((resolve) => { fetches += 1; resolvers.push(resolve); });
function mergePanel(name, doc, payload) { if (name === "candidates") doc.candidates = payload.candidates || []; }
function consumePendingPanel() {}
function syncDetailSectionCache() {}
function panelBody(name, doc) { return "BODY:" + (doc.candidates || []).map((row) => row.status).join(","); }
function bindCandidatePicks() {}
function applyPendingCandidate() {}
function loadFeedbackRecord() {}
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
function commitUrl() {}
function currentViewState() { return {}; }
applyCandidateFilter("status", "ok");
if (candidateStatus !== "ok" || fetches !== 1) throw new Error("first filter did not start one request");
applyCandidateFilter("status", "failed");
if (candidateStatus !== "failed" || fetches !== 2) throw new Error("second filter was swallowed by the in-flight dedupe: " + fetches);
(async () => {
  resolvers[0]({ok: true, json: async () => ({candidates: [{label: "old", status: "ok"}], total: 1, next: null, stamp: "ok"})});
  await new Promise((resolve) => setImmediate(resolve));
  resolvers[1]({ok: true, json: async () => ({candidates: [{label: "new", status: "failed"}], total: 1, next: null, stamp: "failed"})});
  await new Promise((resolve) => setImmediate(resolve));
  if (!host.innerHTML.includes("failed") || host.innerHTML.includes("ok")) throw new Error("old filter response overwrote the new one: " + host.innerHTML);
  if (candidateStatus !== "failed") throw new Error("the final filter state changed");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_stale_candidate_continuation_cannot_merge_after_a_new_filtered_page():
    """@brief 旧筛选的续页晚到时不能合入新筛选首页。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelQuery", "panelUrl", "mergeCandidatePage", "loadMoreCandidates")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "ok", candidateStage = "", candidatePages = 1;
let detailId = "run", detailDoc = {id: "run", candidates: [{label: "new", status: "failed"}], candidatesNext: "cur2", candidatesTotal: 2};
let panelStamps = {}, panelTokens = {}, panelRequestSeq = 0, navigationEpoch = 0;
let selectedRun = "run", detailIdValue = "run";
let candidatePageToken = 0;
function panelBody() { return "BODY"; }
function bindCandidatePicks() {}
let resolveFetch = null, fetches = 0;
globalThis.fetch = () => new Promise((resolve) => { fetches += 1; resolveFetch = resolve; });
(async () => {
  const pending = loadMoreCandidates();
  await new Promise((resolve) => setImmediate(resolve));
  if (fetches !== 1) throw new Error("continuation did not start");
  // 新筛选首页已到达（旧的续页仍在途）。
  candidateStatus = "failed";
  detailDoc.candidates = [{label: "new", status: "failed"}];
  detailDoc.candidatesNext = "cur2";
  resolveFetch({ok: true, json: async () => ({candidates: [{label: "old", status: "ok"}], total: 2, next: null})});
  await pending;
  if (detailDoc.candidates.length !== 1 || detailDoc.candidates[0].label !== "new") {
    throw new Error("a stale continuation merged after the new page: " + JSON.stringify(detailDoc.candidates));
  }
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_carry_panels_keeps_loaded_pages_and_syncs_the_patch_cache():
    """@brief carryPanels 搬旧 payload 保留已加载页数；局部渲染后缓存同步，planner 不再替换。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelKey", "panelNames", "panelLoaded", "mergeCandidatePage", "mergePanel",
        "carryPanels", "planDetailPatch", "syncDetailSectionCache")) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
const detailSectionCache = new Map();
let candidatePages = 2, feedbackPages = 2;
const previous = {id: "run", candidates: [{label: "c1"}], feedback: {records: [{seq: 1}]}};
const fresh = {id: "run"};
carryPanels(previous, fresh);
if (candidatePages !== 2 || feedbackPages !== 2) throw new Error("carryPanels reset the loaded page counts");
const networkDoc = {id: "run"};
mergePanel("candidates", networkDoc, {candidates: [{label: "c2"}]}, false);
mergePanel("feedback", networkDoc, {feedback: {records: [{seq: 2}]}}, false);
if (candidatePages !== 1 || feedbackPages !== 1) throw new Error("a network first page must reset to page one");
function detailSections(doc) { return [{key: "panel-candidates", html: "NEW:" + (doc.candidates || []).length}]; }
syncDetailSectionCache("panel-candidates", networkDoc);
const sections = detailSections(networkDoc);
if (detailSectionCache.get("panel-candidates") !== sections[0].html) throw new Error("cache was not updated after local render");
if (planDetailPatch(sections, detailSectionCache, ["panel-candidates"]).length !== 0) throw new Error("planner would replace the just-updated panel");
console.log("ok");
'''
    _node(script)


def test_a_same_run_budget_chip_cancels_an_in_flight_feedback_claim():
    """@brief 真实 chip 同 run 导航后，旧的反馈 claim 响应不得再 openReview。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    marker = 'document.getElementById("budget-banner").addEventListener("click", (event) => {'
    first = page.index(marker)
    second = page.index(marker, first + 1)
    listener = page[second:page.index("});", second) + 3]
    listener = listener.replace('document.getElementById("budget-banner").addEventListener("click", ', "const budgetChipListener = ", 1)
    listener = listener[:-2] + ";"
    apply_block = page[page.index("function applyViewState("):page.index("function applyPendingDetail(")]
    functions = fields + "\n" + filter_line + "\n" + sort_line + "\n" + _extract(page, "currentViewState") + "\n" + \
        _extract(page, "openFeedbackClaim") + "\n" + apply_block + "\n" + listener
    script = functions + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0, detailId = "run", selectedRun = "run", currentSeq = 0;
let detailDoc = {id: "run", feedback: {records: []}};
let panelStamps = {feedback: "s1"}, feedbackClaimToken = 0, reviewsCalibration = null;
const fullFeedback = new Map();
let visible = "detail", sideView = "runs", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "", currentCluster = "", runsSort = "started";
let turnsOpen = false, piOpen = false, costsOpen = false, restoredDetailView = null, handoffOpen = false;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let candidateNotice = "", feedbackNotice = "";
let openPanels = new Set(), pendingPanel = "", pendingCandidate = "", pendingSeq = 0, pendingPanelOwner = "";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
const emptyHosts = {};
globalThis.location = {pathname: "/", search: ""};
globalThis.history = {pushState() {}, replaceState() {}};
globalThis.document = {
  getElementById: (id) => emptyHosts[id] = emptyHosts[id] || {
    value: "", innerHTML: "", hidden: false,
    classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}}, setAttribute() {},
    addEventListener: (_type, handler) => { emptyHosts[id].on_click = handler; },
  },
  querySelector: () => null,
  querySelectorAll: () => [],
};
let resolveFetch = null, fetchCount = 0, reviewCalls = 0;
globalThis.fetch = () => new Promise((resolve) => { fetchCount += 1; resolveFetch = resolve; });
function openReview() { reviewCalls += 1; }
function pickFeedbackFull() { return null; }
function commitUrl() {}
function resetPanels() { panelStamps = {}; feedbackClaimToken += 1; }
function showSection() {}
function show() {}
function showRight() {}
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}
function loadPanel() {}
function loadFeedbackRecord() {}
function selectCandidate() {}
(async () => {
  const pending = openFeedbackClaim(1, "digest");
  await new Promise((resolve) => setImmediate(resolve));
  if (fetchCount !== 1) throw new Error("claim did not start");
  budgetChipListener({
    preventDefault() {},
    target: {closest: (selector) => (selector === "a[data-run]" ? {dataset: {run: "run"}} : null)},
  });
  if (navigationEpoch === 0) throw new Error("the chip did not start a navigation generation");
  resolveFetch({ok: true, json: async () => ({record: {seq: 1, claims: [{digest: "digest", text: "T"}]}, calibration: null})});
  await pending;
  if (reviewCalls !== 0) throw new Error("a stale claim opened a review after navigation");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_same_shape_candidate_first_page_invalidates_an_old_continuation():
    """@brief 同形状新首页也要推进内容代次；旧续页不能追加旧行或覆盖 next。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelQuery", "panelUrl", "mergeCandidatePage", "mergePanel", "loadMoreCandidates")) + r'''
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let candidateStatus = "", candidateStage = "", candidatePages = 1;
let candidateContentEpoch = 0, feedbackContentEpoch = 0, navigationEpoch = 0;
let detailId = "run", selectedRun = "run", candidatePageToken = 0;
let detailDoc = {id: "run", candidates: [{label: "old1"}], candidatesTotal: 2, candidatesNext: "cur1"};
let panelStamps = {}, panelTokens = {}, panelRequestSeq = 0;
function panelBody() { return "BODY"; }
function bindCandidatePicks() {}
function syncDetailSectionCache() {}
let resolveFetch = null, fetches = 0;
globalThis.fetch = () => new Promise((resolve) => { fetches += 1; resolveFetch = resolve; });
(async () => {
  const pending = loadMoreCandidates();
  await new Promise((resolve) => setImmediate(resolve));
  if (fetches !== 1) throw new Error("continuation did not start");
  // 同形状的新首页到达：内容代次推进，旧续页应失效。
  mergePanel("candidates", detailDoc, {candidates: [{label: "new1"}], total: 1, next: "newcur"}, false);
  resolveFetch({ok: true, json: async () => ({candidates: [{label: "old2"}], total: 2, next: null})});
  await pending;
  if (JSON.stringify(detailDoc.candidates.map((row) => row.label)) !== JSON.stringify(["new1"])) {
    throw new Error("a stale continuation merged old rows: " + JSON.stringify(detailDoc.candidates));
  }
  if (detailDoc.candidatesNext !== "newcur") throw new Error("a stale continuation overwrote next: " + detailDoc.candidatesNext);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_same_shape_feedback_first_page_invalidates_an_old_continuation():
    """@brief 同形状新反馈首页也要推进内容代次；旧续页不能覆盖记录或 next。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "panelQuery", "panelUrl", "mergeFeedbackPage", "mergePanel", "loadMoreFeedback")) + r'''
const CANDIDATES_PAGE = 50, FEEDBACK_PAGE = 10;
let feedbackPages = 1, candidateContentEpoch = 0, feedbackContentEpoch = 0, navigationEpoch = 0;
let detailId = "run", selectedRun = "run", feedbackPageToken = 0;
const record = (seq) => ({seq: seq, label: "c" + seq, claims: [], claim_counts: {total: 0}});
let detailDoc = {id: "run", feedback: {records: [record(1)], total: 2, next: "cur1"}};
let panelStamps = {}, panelTokens = {}, panelRequestSeq = 0;
function repaintFeedbackPanel() {}
function syncDetailSectionCache() {}
let resolveFetch = null, fetches = 0;
globalThis.fetch = () => new Promise((resolve) => { fetches += 1; resolveFetch = resolve; });
(async () => {
  const pending = loadMoreFeedback();
  await new Promise((resolve) => setImmediate(resolve));
  if (fetches !== 1) throw new Error("continuation did not start");
  mergePanel("feedback", detailDoc, {feedback: {records: [record(9)], total: 1, next: "newcur"}}, false);
  resolveFetch({ok: true, json: async () => ({feedback: {records: [record(2)], total: 2, next: null}})});
  await pending;
  if (JSON.stringify(detailDoc.feedback.records.map((row) => row.seq)) !== JSON.stringify([9])) {
    throw new Error("a stale feedback continuation merged old records: " + JSON.stringify(detailDoc.feedback.records));
  }
  if (detailDoc.feedback.next !== "newcur") throw new Error("a stale feedback continuation overwrote next: " + detailDoc.feedback.next);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_first_page_refresh_invalidates_a_stale_locator_page():
    """@brief 定位页与同形状新首页竞争时，内容代次让旧定位页不再合入。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "newestCandidates", "setCandidateNotice", "ensureCandidateVisible", "panelQuery", "mergeCandidatePage", "mergePanel")) + r'''
const CANDIDATES_PAGE = 50;
let candidateStatus = "", candidateStage = "", candidatePages = 1, candidateContentEpoch = 0;
let candidateNotice = "", navigationEpoch = 0, detailId = "run", openPanels = new Set(["candidates"]);
let detailDoc = {id: "run", candidates: [], candidatesTotal: 2, candidatesNext: "cur1"};
function panelUrl() { return "/x"; }
function repaintCandidatePanel() {}
function syncDetailSectionCache() {}
let resolveFetch = null;
globalThis.fetch = () => new Promise((resolve) => { resolveFetch = resolve; });
(async () => {
  const pending = ensureCandidateVisible("c0");
  await new Promise((resolve) => setImmediate(resolve));
  mergePanel("candidates", detailDoc, {candidates: [{label: "new"}], total: 1, next: "newcur"}, false);
  resolveFetch({ok: true, json: async () => ({candidates: [{label: "c0"}], total: 2, page_index: 1})});
  const found = await pending;
  if (found) throw new Error("a stale locator page was treated as merged");
  if (JSON.stringify(detailDoc.candidates.map((row) => row.label)) !== JSON.stringify(["new"])) {
    throw new Error("a stale locator page merged old rows: " + JSON.stringify(detailDoc.candidates));
  }
  if (candidateNotice !== "") throw new Error("a stale locator wrote a notice");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_run_less_navigation_invalidates_an_in_flight_detail_request():
    """@brief 无 run 的 applyViewState 递增 epoch；旧详情成功/404/JSON 迟到都不能填回详情。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    apply_block = page[page.index("function applyViewState("):page.index("function applyPendingDetail(")]
    load_block = page[page.index("async function loadDetail(id)"):page.index("function ledgerTable(")]
    functions = "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "panelNames", "planDetailRequests")) + "\n" + apply_block + "\n" + load_block
    script = functions + r'''
const FILTER_KEYS = ["task", "dtype", "model", "status", "target"];
const SORT_VALUES = ["started", "speedup", "spent"];
let navigationEpoch = 0, selectedRun = "run", selectedCandidate = "", selectedExperiment = "";
let sideView = "detail", visible = "detail", currentPanel = "", currentCluster = "", currentSeq = 0;
let runsSort = "started", turnsOpen = false, piOpen = false, costsOpen = false, restoredDetailView = null, handoffOpen = false;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let candidateNotice = "", feedbackNotice = "";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
let detailToken = 0, detailDoc = {id: "run"}, detailId = "run", detailStamp = "d1";
let pendingPanelOwner = "";
const host = {innerHTML: "OLD", dataset: {}, classList: {add() {}, remove() {}, contains() { return false; }}};
const emptyHosts = {};
globalThis.document = {
  getElementById: (id) => {
    if (id === "view-detail") return host;
    return emptyHosts[id] = emptyHosts[id] || {
      value: "", innerHTML: "", hidden: false,
      classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}}, setAttribute() {},
    };
  },
  querySelector: () => null,
};
globalThis.location = {pathname: "/", search: ""};
globalThis.history = {pushState() {}, replaceState() {}};
let mode = "success";
let jsonResolve = null;
globalThis.fetch = async () => {
  if (mode === "json") {
    return {ok: true, json: () => new Promise((resolve) => { jsonResolve = resolve; })};
  }
  if (mode === "404") return {ok: false, json: async () => ({error: "not found"})};
  return {ok: true, json: async () => ({id: "run", detail_stamp: "d2", candidates: [], feedback: {records: []}})};
};
function resetPanels() {}
function showSection() {}
function show() {}
function showRight() {}
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}
function rememberDetailScroll() {}
function recallDetailScroll() {}
function detailScroller() { return null; }
function detailPatchedRun() { return "run"; }
function clearDetailPatch() {}
function overlayClose() {}
function carryPanels() {}
function patchDetail() {}
function bindPanels() {}
function restoreDetailOpenState() {}
function paintDetailHead() {}
function bindCandidatePicks() {}
function bindFeedback() {}
function openStage() {}
function applyPendingDetail() {}
function planDetailRequests() { return {detail: true, turns: false, live: false, costs: false}; }
function refreshOpenPanels() {}
function refreshTurns() {}
function refreshLive() {}
function loadSessionCosts() {}
function esc(value) { return String(value ?? ""); }
const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
(async () => {
  for (const m of ["success", "404", "json"]) {
    mode = m;
    selectedRun = "run";
    detailDoc = {id: "run"};
    detailId = "run";
    detailStamp = "d1";
    host.innerHTML = "OLD";
    const pending = loadDetail("run");
    await settle();
    applyViewState({view: "runs"}, {full: true});
    if (m === "json") {
      jsonResolve({id: "run", detail_stamp: "d2", candidates: [], feedback: {records: []}});
    }
    await pending;
    await settle();
    if (host.innerHTML !== `<p class="muted">点上面一次运行，这里显示它的详情。</p>`) {
      throw new Error(m + ": stale detail filled the host: " + host.innerHTML);
    }
    if (detailDoc !== null || detailId !== "") throw new Error(m + ": stale detail restored state");
  }
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_the_runs_table_scroll_survives_a_browser_refresh_via_session_storage():
    """@brief 刷新后 loadRuns 重建 tbody 后要从 sessionStorage 恢复运行表滚动。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "esc", "num", "money", "idText", "readRunsScroll", "saveRunsScroll", "restoreRunsScroll",
        "runQuery", "renderRunsRows", "loadRuns")) + r'''
const RUNS_LIMIT = 50;
let runsToken = 0, runsRows = [], runsRendered = false, runsScrollTop = 0;
let runsCursor = null, runsNext = null, selectedRun = "", runsSort = "started", selectedExperiment = "";
const runsStack = [], applied = {task: "", dtype: "", model: "", status: "", target: ""};
let runsPageIds = new Set(), knownRuns = new Map();
const store = new Map([["observe-runs-scroll", "240"]]);
globalThis.sessionStorage = {
  getItem: (key) => (store.has(key) ? store.get(key) : null),
  setItem: (key, value) => store.set(key, String(value)),
};
const body = {innerHTML: "", dataset: {}, querySelectorAll: () => [], addEventListener() {}};
const scroller = {scrollTop: 0};
const panel = {classList: {contains: () => false}};
globalThis.document = {
  getElementById: (id) => (id === "runs-panel" ? panel : {textContent: "", disabled: false, classList: {contains: () => false}}),
  querySelector: (selector) => (selector === "#runs tbody" ? body : (selector === ".runs-table-scroll" ? scroller : null)),
};
globalThis.fetch = async () => ({ok: true, json: async () => ({runs: [{run_id: "a", task: "relu", status: "done", publication: "confirmed"}], total: 1, next: null})});
function applyRovingRows() {}
function rowKeyAction() { return ""; }
function rovingIndex(current) { return current; }
function openDelete() {}
function applyViewState() {}
function commitUrl() {}
function currentViewState() { return {}; }
(async () => {
  runsScrollTop = readRunsScroll();
  await loadRuns();
  if (runsScrollTop !== 240) throw new Error("saved scroll was not read: " + runsScrollTop);
  if (scroller.scrollTop !== 240) throw new Error("refresh did not restore the runs scroll: " + scroller.scrollTop);
  saveRunsScroll(300);
  if (readRunsScroll() !== 300) throw new Error("scroll was not persisted");
  scroller.scrollTop = 0;
  restoreRunsScroll();
  if (scroller.scrollTop !== 300) throw new Error("restore did not apply the remembered scroll");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_generic_inner_scroll_memory_binds_and_restores():
    """@brief 通用 scroll helper：绑定 scroll 事件后写入 sessionStorage，恢复时写回节点。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "readScrollMemory", "writeScrollMemory", "bindScrollMemory", "restoreScrollMemory")) + r'''
const store = new Map();
globalThis.sessionStorage = {
  getItem: (key) => (store.has(key) ? store.get(key) : null),
  setItem: (key, value) => store.set(key, String(value)),
};
const node = {scrollTop: 0, dataset: {}, _listeners: {}, addEventListener: (_type, handler) => { node._listeners.scroll = handler; }};
globalThis.document = {getElementById: (id) => (id === "turns-live" ? node : null)};
bindScrollMemory("turns-live", "turns:run");
node.scrollTop = 180;
node._listeners.scroll();
if (readScrollMemory("turns:run") !== 180) throw new Error("scroll was not saved");
node.scrollTop = 0;
restoreScrollMemory("turns-live", "turns:run");
if (node.scrollTop !== 180) throw new Error("scroll was not restored");
console.log("ok");
'''
    _node(script)


def test_refresh_live_keeps_separate_pi_and_epic_scroll_positions_on_second_refresh():
    """@brief 直接跑生产 refreshLive：非首次、非底部刷新要恢复两个容器各自 scrollTop，且键不互相覆盖。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in (
        "readScrollMemory", "writeScrollMemory", "refreshLive")) + r'''
let selectedRun = "run", detailId = "run", navigationEpoch = 0, visible = "detail", piOpen = true, piStamp = "", piRequestToken = 0;
let detailDoc = {id: "run", kernel_writer: "pi"};
function writerMode() { return "pi"; }
const store = new Map();
globalThis.sessionStorage = {
  getItem: (key) => (store.has(key) ? store.get(key) : null),
  setItem: (key, value) => store.set(key, String(value)),
};
function makeNode(id, top) { return {id, scrollTop: top || 0, scrollHeight: 1000, clientHeight: 100, outerHTML: ""}; }
const hosts = {"pi-live": makeNode("pi-live"), "epic-live": makeNode("epic-live")};
globalThis.document = {getElementById: (id) => hosts[id] || null};
function paintLive(id, html, atBottom) {
  const node = makeNode(id, atBottom ? 1000 : 0);
  hosts[id] = node;
  if (atBottom) node.scrollTop = node.scrollHeight;
}
function epicFeed() { return "EPIC"; }
function piFeed() { return "PI"; }
let fetchStamp = 0;
globalThis.fetch = async () => ({ok: true, json: async () => ({stamp: "s" + (++fetchStamp), lines: [1], epics: [1]})});
(async () => {
  await refreshLive();
  // 用户滚到不同位置，第二次刷新（非底部）必须各自恢复。
  hosts["pi-live"].scrollTop = 180;
  hosts["epic-live"].scrollTop = 90;
  writeScrollMemory("pi:run", 180);
  writeScrollMemory("epic:run", 90);
  await refreshLive();
  if (hosts["pi-live"].scrollTop !== 180) throw new Error("pi scroll was lost: " + hosts["pi-live"].scrollTop);
  if (hosts["epic-live"].scrollTop !== 90) throw new Error("epic scroll was lost: " + hosts["epic-live"].scrollTop);
  if (readScrollMemory("pi:run") !== 180 || readScrollMemory("epic:run") !== 90) throw new Error("pi/epic storage keys overwrote each other");
  // 底部吸附仍保留。
  hosts["pi-live"].scrollTop = 900;
  hosts["epic-live"].scrollTop = 900;
  await refreshLive();
  if (hosts["pi-live"].scrollTop !== 1000 || hosts["epic-live"].scrollTop !== 1000) throw new Error("bottom stick was lost");
  // 首次加载（清空 stamp）才从 sessionStorage 恢复。
  piStamp = "";
  writeScrollMemory("pi:run", 77);
  writeScrollMemory("epic:run", 55);
  await refreshLive();
  if (hosts["pi-live"].scrollTop !== 77 || hosts["epic-live"].scrollTop !== 55) throw new Error("first load did not restore session scroll");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_deferred_a_then_b_turns_and_live_responses_cannot_overwrite_b():
    """@brief A→B 导航、B 先返回、A 迟到：turns/pi/epic 的 rows、stamp、DOM、scroll 都不能被 A 覆盖。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()

    turns = "\n".join(_extract(page, name) for name in ("refreshTurns",)) + r'''
let selectedRun = "A", detailId = "A", navigationEpoch = 0, visible = "detail", turnsOpen = true, handoffOpen = false;
let turnsStamp = "", turnsRows = [], turnsCursor = null, turnsRequestToken = 0;
const host = {innerHTML: "", scrollTop: 0};
globalThis.document = {getElementById: (id) => (id === "turns-live" ? host : null)};
function turnsIdentity(stamp) { return String(stamp || "").split(".").slice(0, 3).join("."); }
function paintTurns() { host.innerHTML = JSON.stringify(turnsRows); }
function bindTurns() {}
function restoreScrollMemory() {}
const pending = new Map();
globalThis.fetch = (url) => new Promise((resolve) => pending.set(String(url), resolve));
(async () => {
  const a = refreshTurns(true);
  await new Promise((resolve) => setImmediate(resolve));
  selectedRun = "B"; detailId = "B"; navigationEpoch += 1;
  const b = refreshTurns(true);
  await new Promise((resolve) => setImmediate(resolve));
  const aUrl = [...pending.keys()].find((key) => key.includes("/turns/A"));
  const bUrl = [...pending.keys()].find((key) => key.includes("/turns/B"));
  pending.get(bUrl)({ok: true, json: async () => ({stamp: "b.1", rows: [{who: "B"}], cursor: 7})});
  await b;
  pending.get(aUrl)({ok: true, json: async () => ({stamp: "a.1", rows: [{who: "A"}], cursor: 9})});
  await a;
  if (JSON.stringify(turnsRows) !== JSON.stringify([{who: "B"}])) throw new Error("late A overwrote B rows: " + JSON.stringify(turnsRows));
  if (turnsStamp !== "b.1" || turnsCursor !== 7) throw new Error("late A overwrote B stamp/cursor");
  if (!host.innerHTML.includes("B") || host.innerHTML.includes("A")) throw new Error("late A rendered into B");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(turns)

    earlier = "\n".join(_extract(page, name) for name in ("loadEarlier",)) + r'''
let selectedRun = "A", detailId = "A", navigationEpoch = 0, visible = "detail", turnsOpen = true, handoffOpen = false;
let turnsRows = [{who: "baseA"}], turnsCursor = "acur", turnsRequestToken = 0;
const host = {scrollTop: 0, scrollHeight: 100, innerHTML: ""};
globalThis.document = {getElementById: (id) => (id === "turns-live" ? host : null)};
function openTurnKeys() { return new Set(); }
function turnsFeed(rows) { host.innerHTML = JSON.stringify(rows); return "FEED"; }
function paintLive(id, html) { host.innerHTML = html; }
function bindTurns() {}
const pending = new Map();
globalThis.fetch = (url) => new Promise((resolve) => pending.set(String(url), resolve));
(async () => {
  const a = loadEarlier();
  await new Promise((resolve) => setImmediate(resolve));
  selectedRun = "B"; detailId = "B"; navigationEpoch += 1; turnsRows = [{who: "baseB"}]; turnsCursor = "bcur";
  const b = loadEarlier();
  await new Promise((resolve) => setImmediate(resolve));
  const aUrl = [...pending.keys()].find((key) => key.includes("/turns/A"));
  const bUrl = [...pending.keys()].find((key) => key.includes("/turns/B"));
  pending.get(bUrl)({ok: true, json: async () => ({rows: [{who: "olderB"}], cursor: "bolder"})});
  await b;
  pending.get(aUrl)({ok: true, json: async () => ({rows: [{who: "olderA"}], cursor: "aolder"})});
  await a;
  if (JSON.stringify(turnsRows.map((row) => row.who)) !== JSON.stringify(["olderB", "baseB"])) {
    throw new Error("late A earlier page merged into B: " + JSON.stringify(turnsRows));
  }
  if (turnsCursor !== "bolder") throw new Error("late A overwrote B cursor: " + turnsCursor);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(earlier)

    live = "\n".join(_extract(page, name) for name in (
        "readScrollMemory", "writeScrollMemory", "refreshLive")) + r'''
let selectedRun = "A", detailId = "A", navigationEpoch = 0, visible = "detail", piOpen = true;
let piStamp = "", piRequestToken = 0;
let detailDoc = {id: "A", kernel_writer: "pi"};
function writerMode() { return "pi"; }
const hosts = {
  "pi-live": {scrollTop: 111, scrollHeight: 1000, clientHeight: 100},
  "epic-live": {scrollTop: 222, scrollHeight: 1000, clientHeight: 100},
};
globalThis.document = {getElementById: (id) => hosts[id] || null};
const store = new Map();
globalThis.sessionStorage = {getItem: (key) => (store.has(key) ? store.get(key) : null), setItem: (key, value) => store.set(key, String(value))};
writeScrollMemory("pi:A", 111);
writeScrollMemory("epic:A", 222);
writeScrollMemory("pi:B", 333);
writeScrollMemory("epic:B", 444);
function paintLive(id, html) { hosts[id] = {scrollTop: 0, scrollHeight: 1000, clientHeight: 100, text: html}; }
function epicFeed(epics) { return "EPIC:" + (epics || []).join(","); }
function piFeed(lines) { return "PI:" + (lines || []).join(","); }
const pending = new Map();
globalThis.fetch = (url) => new Promise((resolve) => pending.set(String(url), resolve));
(async () => {
  const a = refreshLive();
  await new Promise((resolve) => setImmediate(resolve));
  selectedRun = "B"; detailId = "B"; navigationEpoch += 1; detailDoc = {id: "B", kernel_writer: "pi"};
  hosts["pi-live"].scrollTop = 333;
  hosts["epic-live"].scrollTop = 444;
  const b = refreshLive();
  await new Promise((resolve) => setImmediate(resolve));
  const aUrl = [...pending.keys()].find((key) => key.includes("/pi/A"));
  const bUrl = [...pending.keys()].find((key) => key.includes("/pi/B"));
  pending.get(bUrl)({ok: true, json: async () => ({stamp: "b.1", lines: ["B"], epics: ["B"]})});
  await b;
  pending.get(aUrl)({ok: true, json: async () => ({stamp: "a.1", lines: ["A"], epics: ["A"]})});
  await a;
  if (piStamp !== "b.1") throw new Error("late A overwrote B piStamp: " + piStamp);
  if (!hosts["pi-live"].text.includes("PI:B") || hosts["pi-live"].text.includes("PI:A")) throw new Error("late A rendered into B pi DOM: " + hosts["pi-live"].text);
  if (!hosts["epic-live"].text.includes("EPIC:B") || hosts["epic-live"].text.includes("EPIC:A")) throw new Error("late A rendered into B epic DOM: " + hosts["epic-live"].text);
  if (hosts["pi-live"].scrollTop !== 333 || hosts["epic-live"].scrollTop !== 444) throw new Error("B scroll was not restored: " + JSON.stringify([hosts["pi-live"].scrollTop, hosts["epic-live"].scrollTop]));
  if (readScrollMemory("pi:A") !== 111) throw new Error("A pi scroll key changed");
  if (readScrollMemory("epic:A") !== 222) throw new Error("A epic scroll key changed");
  if (readScrollMemory("pi:B") !== 333) throw new Error("B pi scroll key changed");
  if (readScrollMemory("epic:B") !== 444) throw new Error("B epic scroll key changed");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(live)
