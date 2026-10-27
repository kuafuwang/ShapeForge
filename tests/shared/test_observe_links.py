"""§10.1-10.2：首屏三块区、URL 即状态、深链与复制链接的纯函数断言。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observe.index import Index


def _extract(page: str, name: str) -> str:
    """@brief 从页面里抽出一个函数定义（含 async）。

    @param page 页面文本。
    @param name 函数名。
    @return 函数源码。
    """
    marker = f"async function {name}(" if f"async function {name}(" in page else f"function {name}("
    start = page.index(marker)
    # 参数列表里可能有 ``options = {}``：先配平圆括号，再找函数体的第一个 ``{``。
    paren = page.index("(", start)
    depth = 0
    cursor = paren
    while cursor < len(page):
        if page[cursor] == "(":
            depth += 1
        elif page[cursor] == ")":
            depth -= 1
            if depth == 0:
                break
        cursor += 1
    brace = page.index("{", cursor)
    depth = 0
    for index in range(brace, len(page)):
        if page[index] == "{":
            depth += 1
        elif page[index] == "}":
            depth -= 1
            if depth == 0:
                return page[start:index + 1]
    raise AssertionError(f"unclosed function {name}")


def _node(script: str) -> None:
    """@brief 在 Node 里跑一段脚本（没有 node 时跳过）。

    @param script 完整脚本。
    """
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_four_common_questions_are_within_two_state_actions_and_urls_round_trip():
    """@brief 验收：四个常用问题各 ≤2 个动作；``writeUrl``/``readUrl`` 往返一致。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    script = fields + "\n" + "\n".join(_extract(page, name) for name in ("writeUrl", "readUrl", "nextViewState")) + r'''
const STATE = {view: "runs", task: "relu", panel: "candidates", candidate: "r3.p0",
  run: "conv2d/one", cluster: "sig-1", sort: "spent", experiment_id: "exp/1", seq: "4"};
const canonical = (value) => JSON.stringify(Object.keys(value).sort().map((key) => [key, value[key]]));
const round = readUrl("?" + writeUrl(STATE));
if (canonical(round) !== canonical(STATE)) {
  throw new Error("url round trip changed state: " + JSON.stringify(round));
}
if (writeUrl({}) !== "") throw new Error("empty state must give an empty query");

// Q1 某场为什么没发布：选一次运行。
let q1 = {view: "runs"};
q1 = nextViewState(q1, {kind: "select-run", run: "conv2d/one"});
if (q1.view !== "detail" || q1.run !== "conv2d/one") throw new Error("select-run did not reach detail");

// Q2 哪颗候选的证据：展开候选 + 选中（2 个动作）。
let q2 = {view: "runs"};
q2 = nextViewState(q2, {kind: "open-panel", run: "conv2d/one", panel: "candidates"});
q2 = nextViewState(q2, {kind: "pick-candidate", label: "r3.p0"});
if (q2.panel !== "candidates" || q2.candidate !== "r3.p0" || q2.run !== "conv2d/one") throw new Error("candidate evidence path broken");

// Q3 失败簇 → 一场运行（2 个动作）。
let q3 = {view: "runs"};
q3 = nextViewState(q3, {kind: "open-cluster", sig: "sig-1"});
q3 = nextViewState(q3, {kind: "select-run", run: "relu/7"});
if (q3.cluster) throw new Error("select-run must leave the cluster view");
if (q3.run !== "relu/7") throw new Error("cluster sample did not reach the run");

// Q4 同题趋势：打开分析并带题目过滤（2 个动作）。
let q4 = {view: "runs"};
q4 = nextViewState(q4, {kind: "open-view", view: "analysis"});
q4 = nextViewState(q4, {kind: "filter", key: "task", value: "relu"});
if (q4.view !== "analysis" || q4.task !== "relu") throw new Error("analysis filter path broken");
if (!writeUrl(q4).includes("view=analysis") || !writeUrl(q4).includes("task=relu")) throw new Error("analysis state not in url");
console.log("ok");
'''
    _node(script)


def test_the_attention_zones_keep_their_counts_when_collapsed():
    """@brief 验收：三块区（进行中/需要注意/最近结果）计数来自 patrol，折叠只隐藏 body。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    zone_key = next(line.strip() for line in page.splitlines() if line.strip().startswith("const ZONE_KEY"))
    script = zone_key + "\n" + "\n".join(_extract(page, name) for name in (
        "esc", "num", "idText", "attentionRows", "zoneCollapsed", "renderAttention")) + r'''
const hosts = {};
globalThis.document = {
  getElementById: (id) => (hosts[id] = hosts[id] || {innerHTML: ""}),
  querySelector: () => null,
};
globalThis.localStorage = {getItem: (key) => (key === "observe-zone-running" ? "collapsed" : null), setItem: () => {}};
const doc = {
  live: {runs: 3, evals: 2, gpu: {holders: 1, waiters: 4}},
  alerts: [{kind: "publication_lost", run: "conv2d/a", reason: "missing_confirmation_block"},
           {kind: "catalog_mismatch", run: "conv2d/b", reason: "catalog_mismatch"},
           {kind: "confirmation_regressed", run: "conv2d/c", reason: "confirmation_regressed"},
           {kind: "budget_over", run: "conv2d/d"}, {kind: "session_failure", run: "conv2d/e"}],
  failing: {total: 12, clusters: [{sig: "sig-1", name: "ABI 不匹配", n: 133, example: {run: "relu/1"}},
                                   {sig: "sig-2", name: "编译", n: 9, example: {run: "relu/2"}},
                                   {sig: "sig-3", name: "对答案", n: 5, example: {run: "relu/3"}},
                                   {sig: "sig-4", name: "隐藏", n: 1, example: {run: "relu/4"}}]},
  best: [{task: "conv2d", published_speedup: 1.65, stale_days: 0.4},
         {task: "relu", published_speedup: 1.2, stale_days: 2.0},
         {task: "gemm", published_speedup: 1.05, stale_days: 9.0},
         {task: "hidden", published_speedup: 1.0, stale_days: 30.0}],
};
const zones = attentionRows(doc);
if (zones.map((zone) => zone.key).join(",") !== "running,attention,recent") throw new Error("zone order drifted");
if (zones[0].count !== 3) throw new Error("running count wrong");
if (zones[1].count !== 17) throw new Error("attention count must be failing.total + alerts, got " + zones[1].count);
if (!zones[1].body.includes("sig-1") || !zones[1].body.includes("sig-3")) throw new Error("top clusters missing");
if (zones[1].body.includes("sig-4")) throw new Error("only three clusters may be shown");
if (zones[2].count !== 4) throw new Error("recent count wrong");
renderAttention(doc);
const html = hosts["attention-zones"].innerHTML;
if (!html.includes('data-zone="running"') || !html.includes("zone collapsed")) throw new Error("saved collapse state not applied");
if (!html.includes('<span class="zone-count">3</span>')) throw new Error("collapsed zone lost its count");
if (!html.includes('data-cluster="sig-1"')) throw new Error("cluster link missing");
console.log("ok");
'''
    _node(script)


def test_the_copy_link_matches_the_current_view():
    """@brief 验收：复制链接与当前视图一致（stub clipboard），且链接能被 readUrl 读回。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    script = fields + "\n" + filter_line + "\n" + "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "copyText", "copyCurrentLink")) + r'''
let visible = "detail", selectedRun = "conv2d/one", selectedCandidate = "r3.p0";
let selectedExperiment = "", currentPanel = "candidates", currentCluster = "", sideView = "runs";
let currentSeq = 4;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let openPanels = new Set(["candidates"]);
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let restoredDetailView = null;
let runsSort = "started";
const applied = {task: "relu", dtype: "", model: "", status: "", target: ""};
globalThis.location = {origin: "http://127.0.0.1:7777", pathname: "/"};
let copied = "";
const canonical = (value) => JSON.stringify(Object.keys(value).sort().map((key) => [key, value[key]]));
Object.defineProperty(globalThis, "navigator", {
  value: {clipboard: {writeText: async (text) => { copied = text; }}}, configurable: true,
});
globalThis.window = {prompt: () => { throw new Error("clipboard fallback should not run"); }};
copyCurrentLink().then((url) => {
  const state = currentViewState();
  const expected = "http://127.0.0.1:7777/?" + writeUrl(state);
  if (copied !== expected) throw new Error("clipboard got " + copied + " expected " + expected);
  if (!copied.includes("run=conv2d%2Fone") || !copied.includes("panel=candidates") || !copied.includes("task=relu")) {
    throw new Error("link lost part of the view: " + copied);
  }
  if (!copied.includes("seq=4")) throw new Error("copy dropped the current seq: " + copied);
  if (canonical(readUrl(copied.split("?")[1])) !== canonical(state)) throw new Error("copied link does not round-trip");
  console.log("ok");
}).catch((err) => { console.error(err); process.exit(1); });
'''
    _node(script)


def test_the_first_screen_reads_a_five_thousand_run_index_without_listing_directories(tmp_path, monkeypatch):
    """@brief §10.1 验收（合成 5000 场）：首屏读模型不再扫运行目录。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from observe import query
    from observe.index import get_index

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    index = get_index()
    rows = [
        (f"task/{number}", "relu", "fp16", "small", "apple-m4-max", "done", "3", float(number), 1.0, 2.0, "confirmed")
        for number in range(5000)
    ]
    index.conn.executemany(
        "INSERT INTO runs (run_id, task, dtype, size, target_id, status, protocol_version, started_at,"
        " spent_usd, budget_usd, publication) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows,
    )
    index.conn.commit()
    query.tick()
    index._prepare_frame()
    query.patrol()
    query.runs_page(limit=50)
    query.overview()
    query.budget_report()
    assert index.stats["dir_lists"] == 0, "首屏读模型不该列运行目录"
    assert index.stats["json_reads"] == 0
    assert len(json.dumps(query.patrol(), ensure_ascii=False)) < 8192


def test_the_command_palette_reducers_wrap_and_map_actions():
    """@brief §10.3 验收：命令面板的纯 reducer（↑↓ 环绕、Enter/Esc、三种动作映射）。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    assert 'id="search-modal"' in page and "event.metaKey" in page and "setTimeout(runSearch, 120)" in page
    script = "\n".join(_extract(page, name) for name in ("paletteMove", "paletteKey", "searchActions")) + r'''
if (paletteMove(0, -1, 3) !== 2 || paletteMove(2, 1, 3) !== 0) throw new Error("palette must wrap");
if (paletteMove(0, 1, 0) !== 0) throw new Error("empty palette must stay at 0");
const down = paletteKey(0, "ArrowDown", 2);
if (down.index !== 1 || down.action) throw new Error("ArrowDown reducer wrong");
if (paletteKey(0, "Enter", 0).action !== "") throw new Error("Enter on an empty palette must do nothing");
const enter = paletteKey(1, "Enter", 2);
if (enter.index !== 1 || enter.action !== "open") throw new Error("Enter reducer wrong");
if (paletteKey(0, "Escape", 2).action !== "close") throw new Error("Escape reducer wrong");

const candidate = searchActions("candidates", [{run_id: "relu/1", label: "r3.p0", status: "abi_error", message: "ABI mismatch"}])[0];
if (JSON.stringify(candidate.state) !== JSON.stringify({view: "detail", run: "relu/1", panel: "candidates", candidate: "r3.p0"})) {
  throw new Error("candidate action wrong: " + JSON.stringify(candidate.state));
}
if (!candidate.label.includes("r3.p0")) throw new Error("candidate label missing");
const cluster = searchActions("clusters", [{sig: "sig-1", name: "ABI 不匹配", n: 3}])[0];
if (cluster.state.cluster !== "sig-1" || !cluster.label.includes("ABI")) throw new Error("cluster action wrong");
const run = searchActions("runs", [{run_id: "relu/1", task: "relu", stop_reason: "max_rounds"}])[0];
if (run.state.run !== "relu/1" || run.state.view !== "detail") throw new Error("run action wrong");
if (searchActions("runs", []).length !== 0) throw new Error("empty rows must give no actions");
console.log("ok");
'''
    _node(script)


def test_the_banner_script_caps_chips_like_the_server():
    """@brief 评审回归：JS ``bannerHtml`` 也必须只画 10 条并保留总数。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    consts = []
    for prefix in ("const LABELS =", "const LABELS_CN =", "const BANNER_CHIPS ="):
        consts.append(next(line.strip() for line in page.splitlines() if line.strip().startswith(prefix)))
    script = "\n".join(consts) + "\n" + _extract(page, "esc") + "\n" + _extract(page, "idText") + "\n" + _extract(page, "bannerHtml") + r'''
const report = {level: "over", n_runs: 500, n_flagged: 500, total_spent_usd: 1000, total_budget_usd: 500,
  n_over: 500, total_over_usd: 500,
  flagged: Array.from({length: 500}, (_, i) => ({id: "task/" + i, level: "over", task: "relu", dtype: "fp16",
    flags: ["over"], text: "over by $1.000 (+100.0%)"}))};
const html = bannerHtml(report);
if (new TextEncoder().encode(html).length > 8192) throw new Error("banner HTML is too large: " + html.length);
if ((html.match(/class="chip /g) || []).length !== 10) throw new Error("JS banner must cap at 10 chips");
if (!html.includes("共 500 条")) throw new Error("full count missing from the JS banner");
if (!html.includes("还有 490 条")) throw new Error("remaining count missing from the JS banner");
console.log("ok");
'''
    _node(script)


def test_full_url_restore_resets_stale_fields_and_preserves_seq_in_navigation():
    """@brief 评审回归：完整 URL 恢复要复位缺失字段；局部导航保留筛选、换 run 清 seq、选候选保 seq。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + "\n".join(
        _extract(page, name) for name in (
            "writeUrl", "readUrl", "currentViewState", "nextViewState", "commitUrl", "applyViewState", "selectCandidate")
    ) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0;
let visible = "detail", selectedRun = "old/run", selectedCandidate = "r9", selectedExperiment = "old-exp";
let currentPanel = "candidates", currentCluster = "sig-old", currentSeq = 7, sideView = "analysis";
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0;
let pendingPanelOwner = "";
let runsSort = "spent";
const applied = {task: "old", dtype: "bf16", model: "m", status: "done", target: "t"};
const urls = [];
globalThis.location = {pathname: "/"};
globalThis.history = {
  pushState: (_s, _t, url) => urls.push(url),
  replaceState: (_s, _t, url) => urls.push(url),
};
const hosts = {};
globalThis.document = {
  getElementById: (id) => (hosts[id] = hosts[id] || {
    id, value: "", innerHTML: "",
    classList: {toggle() {}, contains() { return false; }, add() {}, remove() {}},
    setAttribute() {},
  }),
  querySelector: () => null,
  querySelectorAll: () => [],
};
let runLoads = 0;
let detailId = "old/run", detailDoc = {id: "old/run"}, detailStamp = "s";
function loadRuns() { runLoads += 1; }
function resetPanels() { openPanels = new Set(); pendingPanel = ""; pendingCandidate = ""; pendingSeq = 0; }
function clearDetailPatch() {}
function showSection() {}
function show() {}
function showRight() {}
function loadCompare() {}
async function openCluster() {}
function loadPanel() {}
function loadFeedbackRecord() {}

// 1) 完整恢复：URL 里没有的筛选/sort/experiment/seq 全部复位。
applyViewState(readUrl("?view=runs"), {full: true});
if (applied.task !== "" || applied.dtype !== "" || applied.model !== "" || applied.status !== "" || applied.target !== "") {
  throw new Error("stale filters survived a full restore: " + JSON.stringify(applied));
}
if (runsSort !== "started") throw new Error("sort was not reset");
if (selectedExperiment !== "") throw new Error("experiment was not reset");
if (currentSeq !== 0 || selectedRun !== "" || selectedCandidate !== "" || currentPanel !== "" || currentCluster !== "") {
  throw new Error("stale run/seq survived a full restore");
}
if (sideView !== "runs") throw new Error("a URL without view must fall back to runs, got " + sideView);
if (openPanels.size !== 0) throw new Error("a URL without panel must clear the open panels");
if (detailDoc !== null || detailId !== "") throw new Error("a run-less URL must clear the old detail");
if (!hosts["view-detail"].innerHTML.includes("点上面一次运行")) throw new Error("stale detail HTML survived a run-less restore");
if (runLoads < 1) throw new Error("a run-less full restore must refetch the runs list");

// 2) 完整恢复带 deep link：seq/候选/筛选都回来。
applyViewState(readUrl("?view=analysis&task=relu&sort=spent&seq=4&experiment_id=exp&run=conv2d/one&panel=candidates&candidate=r3"), {full: true});
if (applied.task !== "relu" || applied.dtype !== "") throw new Error("filters were not restored");
if (runsSort !== "spent" || selectedExperiment !== "exp" || currentSeq !== 4) throw new Error("sort/experiment/seq were not restored");
if (selectedRun !== "conv2d/one" || selectedCandidate !== "r3" || currentPanel !== "candidates") throw new Error("run/candidate/panel were not restored");
if (openPanels.size !== 0) throw new Error("refreshOpenPanels must not pre-load the deep-linked panel");
if (pendingPanel !== "candidates") throw new Error("the deep-linked panel must stay pending until the detail loads");
if (!writeUrl(currentViewState()).includes("seq=4")) throw new Error("seq missing from the restored URL");

// 3) 选候选保留 seq（深链不被改写丢字段）。
selectCandidate("r5.p0");
if (!urls.at(-1).includes("seq=4")) throw new Error("selectCandidate dropped seq: " + urls.at(-1));
if (!urls.at(-1).includes("candidate=r5.p0")) throw new Error("selectCandidate did not record the candidate");

// 4) 换 run 的局部导航：保留筛选，清掉旧 seq，URL 不带 seq。
applyViewState({view: "detail", run: "conv2d/two", panel: ""});
if (currentSeq !== 0) throw new Error("changing the run must clear the old seq");
if (applied.task !== "relu") throw new Error("local navigation must keep filters");
commitUrl(currentViewState(), true);
if (urls.at(-1).includes("seq=")) throw new Error("stale seq reached the URL: " + urls.at(-1));
if (!urls.at(-1).includes("task=relu")) throw new Error("filters were dropped on local navigation");

// 5) 连续导航后 popstate 回到无 task 的 URL：完整恢复要清掉筛选。
applyViewState(readUrl("?view=runs"), {full: true});
if (applied.task !== "" || currentSeq !== 0 || selectedRun !== "") throw new Error("popstate full restore kept stale state");

// 6) select-run 动作本身清 seq，key 仍可 round-trip。
const cleared = nextViewState({view: "runs", seq: "9"}, {kind: "select-run", run: "x"});
if (cleared.seq !== undefined) throw new Error("select-run must clear seq");
const round = readUrl("?" + writeUrl({run: "conv2d/one", seq: "9", task: "relu"}));
if (round.seq !== "9" || round.run !== "conv2d/one" || round.task !== "relu") throw new Error("seq round-trip broke");
console.log("ok");
'''
    _node(script)


def test_a_cold_feedback_panel_loads_once_then_applies_the_seq():
    """@brief 评审回归：``panel=feedback&seq=N`` 冷开只发一个面板请求，面板渲染后再请求记录。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    names = ("panelKey", "panelUrl", "panelQuery", "pickFeedbackFull", "consumePendingPanel", "loadPanel", "loadFeedbackRecord",
             "bindPanels", "applyPendingDetail")
    script = "\n".join(_extract(page, name) for name in names) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
const hosts = {};
function host(id) {
  if (!hosts[id]) hosts[id] = {
    id, value: "", innerHTML: "",
    classList: {toggle() {}, contains() { return false; }, add() {}, remove() {}},
    setAttribute() {}, scrollIntoView() {},
  };
  return hosts[id];
}
let toggleHandler = null;
const box = {
  dataset: {panel: "feedback"}, _open: false,
  set open(value) {
    const changed = this._open !== value;
    this._open = value;
    if (changed) queueMicrotask(() => toggleHandler({target: this}));
  },
  get open() { return this._open; },
};
globalThis.document = {getElementById: host, querySelector: () => box};
const detailHost = {dataset: {}, addEventListener: (_type, handler) => { toggleHandler = handler; }};
const panelCalls = [];
const recordCalls = [];
const order = [];
globalThis.fetch = async (url) => {
  const text = String(url);
  if (text.includes("/feedback/4")) {
    recordCalls.push(text);
    order.push("record");
    return {ok: true, json: async () => ({record: {seq: 4, claims: []}})};
  }
  panelCalls.push(text);
  order.push("panel");
  return {ok: true, json: async () => ({stamp: "s1", feedback: {records: [{seq: 4, claims: []}]}})};
};
let detailId = "run";
let detailDoc = {id: "run"};
let panelStamps = {};
let panelTokens = {};
const panelInflight = new Map();
let panelRequestSeq = 0;
let feedbackRecordToken = 0;
let pendingPanel = "feedback", pendingCandidate = "", pendingSeq = 4;
let pendingPanelOwner = "";
let openPanels = new Set();
let currentPanel = "", currentSeq = 4;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
const CANDIDATES_PAGE = 50;
const FEEDBACK_PAGE = 10;
const fullFeedback = new Map();
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
let reviewsCalibration = null;
function mergePanel(_name, doc, payload) { doc.feedback = payload.feedback; }
function panelBody() { return '<div id="feedback-full-4"></div>'; }
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
function feedbackFullMarkup() { return "full"; }
function revealFeedbackRecord() {}
function commitUrl() {}
function currentViewState() { return {}; }

(async () => {
  bindPanels(detailHost);
  applyPendingDetail();
  for (let i = 0; i < 10; i += 1) await new Promise((resolve) => setImmediate(resolve));
  if (panelCalls.length !== 1) throw new Error("cold feedback panel made " + panelCalls.length + " panel requests");
  if (recordCalls.length !== 1) throw new Error("cold feedback seq made " + recordCalls.length + " record requests");
  if (!panelCalls[0].endsWith("/feedback?limit=10")) throw new Error("wrong panel URL: " + panelCalls[0]);
  if (!recordCalls[0].includes("/feedback/4")) throw new Error("wrong record URL: " + recordCalls[0]);
  if (order.join(",") !== "panel,record") throw new Error("record request must follow the panel render: " + order);
  if (pendingSeq !== 0) throw new Error("pendingSeq must clear after the record request");
  if (host("feedback-full-4").innerHTML !== "full") throw new Error("record was not rendered");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_the_budget_chip_uses_url_state_and_clears_run_scoped_fields():
    """@brief 评审回归：预算 chip 的事件路径必须走 ``applyViewState``/``commitUrl``。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    names = ("writeUrl", "readUrl", "currentViewState", "commitUrl", "applyViewState")
    marker = 'document.getElementById("budget-banner").addEventListener("click", (event) => {'
    first = page.index(marker)
    second = page.index(marker, first + 1)
    end = page.index("});", second) + 3
    listener = page[second:end]
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + "\n".join(
        _extract(page, name) for name in names) + "\n" + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0;
let visible = "detail", sideView = "runs";
let turnsOpen = true, piOpen = true, costsOpen = true, handoffOpen = false;
let restoredDetailView = null;
let selectedRun = "old/run", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "feedback", currentCluster = "sig-old", currentSeq = 7;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0;
let pendingPanelOwner = "";
let openPanels = new Set(["feedback"]);
let runsSort = "started";
const applied = {task: "relu", dtype: "", model: "", status: "", target: ""};
const urls = [];
globalThis.location = {pathname: "/"};
globalThis.history = {
  pushState: (_s, _t, url) => urls.push(url),
  replaceState: (_s, _t, url) => urls.push(url),
};
const hosts = {};
globalThis.document = {
  getElementById: (id) => (hosts[id] = hosts[id] || {
    value: "", innerHTML: "",
    classList: {toggle() {}, contains() { return false; }, add() {}, remove() {}},
    setAttribute() {},
    addEventListener: (_type, handler) => { hosts[id].on_click = handler; },
  }),
  querySelector: () => null,
  querySelectorAll: () => [],
};
function resetPanels() { openPanels = new Set(); pendingPanel = ""; pendingCandidate = ""; pendingSeq = 0; }
function loadRuns() {}
function showSection() {}
function show() {}
function showRight() {}
function loadCompare() {}
async function openCluster() {}
function loadPanel() {}
function loadFeedbackRecord() {}
''' + "\n" + listener + "\n" + r'''
const epochBefore = navigationEpoch;
hosts["budget-banner"].on_click({
  preventDefault() {},
  target: {closest: (selector) => (selector === "a[data-run]" ? {dataset: {run: "new/run"}} : null)},
});
if (selectedRun !== "new/run") throw new Error("chip did not select the run: " + selectedRun);
if (navigationEpoch <= epochBefore) throw new Error("the budget chip did not start a navigation generation");
if (currentSeq !== 0) throw new Error("chip did not clear the old seq");
if (currentPanel !== "" || currentCluster !== "") throw new Error("chip did not clear panel/cluster");
if (applied.task !== "relu") throw new Error("chip must keep the filters");
if (urls.length !== 1) throw new Error("chip did not write history: " + JSON.stringify(urls));
if (!urls[0].includes("run=new%2Frun") || urls[0].includes("seq=") || urls[0].includes("panel=") || urls[0].includes("cluster=")) {
  throw new Error("chip URL still carries run-scoped state: " + urls[0]);
}
// 异 run：上一场的复数展开态（panels / folds）也不能带过去。
if (urls[0].includes("panels=") || urls[0].includes("folds=")) {
  throw new Error("chip URL still carries panels/folds from the old run: " + urls[0]);
}
if (openPanels.size !== 0 || turnsOpen || piOpen || costsOpen) {
  throw new Error("switching runs must clear openPanels/folds");
}

// 同一场的 chip 也要清掉 seq/panel/candidate（不能只在 run 变化时清）；复数展开态保留。
currentSeq = 7;
currentPanel = "feedback";
selectedCandidate = "r9";
openPanels = new Set(["feedback"]);
turnsOpen = true; piOpen = false; costsOpen = false;
hosts["budget-banner"].on_click({
  preventDefault() {},
  target: {closest: (selector) => (selector === "a[data-run]" ? {dataset: {run: "new/run"}} : null)},
});
if (selectedRun !== "new/run") throw new Error("same-run chip lost the run");
if (currentSeq !== 0 || currentPanel !== "" || selectedCandidate !== "") {
  throw new Error("same-run chip must clear seq/panel/candidate");
}
if (urls.length !== 2 || urls[1].includes("seq=") || urls[1].includes("panel=")) {
  throw new Error("same-run chip URL still carries run-scoped state: " + urls[1]);
}
if (!urls[1].includes("panels=feedback") || !urls[1].includes("folds=turns")) {
  throw new Error("same-run navigation must keep the open panels/folds: " + urls[1]);
}
console.log("ok");
'''
    _node(script)


def test_same_run_budget_chip_clears_candidate_highlight_and_open_record_in_dom():
    """@brief 已选候选且已开记录时点同 run 预算 chip：same=true 路径也必须清 DOM 与 patch 缓存。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
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
    names = ("writeUrl", "readUrl", "currentViewState", "clearDetailSelection", "applyViewState")
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + "\n".join(
        _extract(page, name) for name in names) + "\n" + listener + r"""
let navigationEpoch = 0;
let visible = "detail", sideView = "runs";
let selectedRun = "run", selectedCandidate = "r9", selectedExperiment = "";
let currentPanel = "feedback", currentCluster = "", currentSeq = 7;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0, pendingPanelOwner = "";
let openPanels = new Set(["candidates", "feedback"]);
let turnsOpen = false, piOpen = false, costsOpen = false, restoredDetailView = null, handoffOpen = false;
let runsSort = "started";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
const cacheKeys = [];
function syncDetailSectionCache(key) { cacheKeys.push(key); }
function suppressToggle() {}
function resetPanels() {}
function showSection() {}
function show() {}
function showRight() {}
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}
function commitUrl(state) { urls.push(writeUrl(state)); }
const rowClasses = new Set(["pick", "on"]);
const row = {classList: {remove(name) { rowClasses.delete(name); }}};
const closedDetails = {open: true, dataset: {feedbackSeq: "7"}};
const view = {querySelectorAll: (selector) => (selector === "tr.pick.on" ? [row] : (selector === "details[data-feedback-seq][open]" ? [closedDetails] : []))};
const hosts = {};
globalThis.document = {
  getElementById: (id) => (id === "view-detail" ? view : (hosts[id] = hosts[id] || {
    value: "", innerHTML: "", hidden: false,
    classList: {add() {}, remove() {}, contains() { return false; }, toggle() {}}, setAttribute() {},
    addEventListener: (_type, handler) => { hosts[id].on_click = handler; },
  })),
  querySelector: () => null,
  querySelectorAll: () => [],
};
const urls = [];
globalThis.location = {pathname: "/"};
globalThis.history = {pushState(_s, _t, url) { urls.push(url); }, replaceState(_s, _t, url) { urls.push(url); }};
budgetChipListener({
  preventDefault() {},
  target: {closest: (selector) => (selector === "a[data-run]" ? {dataset: {run: "run"}} : null)},
});
if (rowClasses.has("on")) throw new Error("same-run chip left the old candidate highlighted");
if (closedDetails.open) throw new Error("same-run chip left the old feedback record open");
if (!cacheKeys.includes("panel-candidates") || !cacheKeys.includes("panel-feedback")) throw new Error("same-run chip did not sync the patch cache: " + JSON.stringify(cacheKeys));
if (selectedCandidate !== "" || currentSeq !== 0) throw new Error("same-run chip did not clear selection state");
if (urls.length !== 1 || urls[0].includes("seq=") || urls[0].includes("panel=")) throw new Error("same-run chip URL still has run-scoped state: " + JSON.stringify(urls));
console.log("ok");
"""
    _node(script)


def test_same_payload_fast_path_does_not_repaint_after_selection_cleanup():
    """@brief 导航清掉候选高亮/记录后，loadPanel 的 payload.same 快路径不能再把旧 DOM 重画回来。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    functions = "\n".join(_extract(page, name) for name in (
        "panelKey", "panelQuery", "panelUrl", "panelLoaded", "consumePendingPanel",
        "loadPanel", "clearDetailSelection"))
    script = functions + r"""
const CANDIDATES_PAGE = 50;
let candidateStatus = "", candidateStage = "";
let detailDoc = {id: "run", candidates: [{label: "r3"}]}, detailId = "run";
let navigationEpoch = 0, panelStamps = {candidates: "p"}, panelTokens = {}, panelRequestSeq = 0;
const panelInflight = new Map();
let pendingPanelOwner = "", pendingCandidate = "", pendingSeq = 0;
let currentSeq = 0, selectedCandidate = "";
const cacheKeys = [];
function syncDetailSectionCache(key) { cacheKeys.push(key); }
function suppressToggle() {}
function bindCandidatePicks() {}
function mergePanel() {}
function panelBody() { return "REPAINTED"; }
function panelLabel() { return "候选"; }
function esc(value) { return String(value ?? ""); }
const rowClasses = new Set(["pick", "on"]);
const row = {classList: {remove(name) { rowClasses.delete(name); }}};
const details = {open: true, dataset: {feedbackSeq: "7"}};
const view = {querySelectorAll: (selector) => (selector === "tr.pick.on" ? [row] : (selector === "details[data-feedback-seq][open]" ? [details] : []))};
const panelHost = {innerHTML: "OLD"};
globalThis.document = {
  getElementById: (id) => (id === "view-detail" ? view : (id === "panel-candidates" ? panelHost : null)),
  querySelector: () => null,
};
let panelFetches = 0;
globalThis.fetch = async () => { panelFetches += 1; return {ok: true, json: async () => ({same: true})}; };
clearDetailSelection(view);
if (rowClasses.has("on") || details.open) throw new Error("selection cleanup did not clear the old DOM");
if (!cacheKeys.includes("panel-candidates") || !cacheKeys.includes("panel-feedback")) throw new Error("selection cleanup did not sync patch caches");
(async () => {
  await loadPanel("candidates", "run");
  if (panelFetches !== 1) throw new Error("same fast path did not issue its request");
  if (panelHost.innerHTML !== "OLD") throw new Error("same=true repainted the panel: " + panelHost.innerHTML);
  if (rowClasses.has("on") || details.open) throw new Error("same=true restored the stale DOM");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
"""
    _node(script)


def test_compare_ignores_a_stale_response():
    """@brief 评审回归：连续点两个实验时，先发的慢响应不能覆盖后发的。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = "\n".join(_extract(page, name) for name in ("esc", "num", "loadCompare")) + r'''
const host = {innerHTML: ""};
globalThis.document = {getElementById: () => host};
const pending = new Map();
globalThis.fetch = (url) => new Promise((resolve) => pending.set(String(url), resolve));
const doc = (name) => ({ok: true, json: async () => ({verdict: name, reason: "", arms: [], mixed: [], drift: [], inferred: []})});
(async () => {
  const a = loadCompare("exp/A");
  const b = loadCompare("exp/B");
  const aUrl = [...pending.keys()].find((key) => key.includes("exp/A"));
  const bUrl = [...pending.keys()].find((key) => key.includes("exp/B"));
  pending.get(bUrl)(doc("B"));
  await b;
  pending.get(aUrl)(doc("A"));
  await a;
  if (!host.innerHTML.startsWith("<p>B")) throw new Error("stale A overwrote B: " + host.innerHTML);
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_the_analysis_panels_carry_the_task_filter():
    """@brief 评审回归：分析请求必须带上当前的 ``task`` 过滤，且无过滤时不伪造空参数。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    script = _extract(page, "loadAnalysis") + r'''
let analysisState = {epoch: "3", stamps: {gap: "", committees: "", noise: "", publications: ""}};
let applied = {task: "relu"};
const calls = [];
async function analysisFetch(path, params) { calls.push({path, params}); return {same: true}; }
const hosts = {};
globalThis.document = {getElementById: (id) => (hosts[id] = hosts[id] || {innerHTML: ""})};
(async () => {
  await loadAnalysis();
  if (calls.length !== 3) throw new Error("expected three analysis calls, got " + calls.length);
  for (const call of calls) {
    if (call.params.task !== "relu") throw new Error("missing task on " + call.path + ": " + JSON.stringify(call.params));
  }
  calls.length = 0;
  applied = {task: ""};
  analysisState = {epoch: "3", stamps: {gap: "", committees: "", noise: "", publications: ""}};
  await loadAnalysis();
  for (const call of calls) {
    if ("task" in call.params) throw new Error("empty task must not be sent on " + call.path);
  }
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_a_same_run_panel_deep_link_is_loaded_once_by_the_detail_path():
    """@brief 评审回归：同 run 完整恢复时，深链面板只能由 ``applyPendingDetail`` 加载一次。

    生产路径 ``applyViewState`` → ``loadDetail``（``same=true`` 与 ``same=false``）都走一遍；
    ``refreshOpenPanels`` 不能在同一 tick 里再发一次相同的面板请求。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    filter_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const FILTER_KEYS"))
    sort_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const SORT_VALUES"))
    names = ("writeUrl", "readUrl", "currentViewState", "commitUrl", "applyViewState",
             "panelNames", "panelKey", "planPanelFetches", "refreshOpenPanels",
             "applyPendingDetail", "loadDetail", "loadFeedbackRecord", "selectCandidate")
    script = fields + "\n" + filter_line + "\n" + sort_line + "\n" + "\n".join(
        _extract(page, name) for name in names) + r'''
function syncDetailSectionCache() {}
let navigationEpoch = 0;
let visible = "detail", sideView = "runs";
let selectedRun = "", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "", currentCluster = "", currentSeq = 0;
let candidateStatus = "", candidateStage = "", candidatePages = 1, feedbackPages = 1;
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0;
let pendingPanelOwner = "";
let openPanels = new Set(), panelStamps = {}, panelTokens = {};
let runsSort = "started";
const applied = {task: "", dtype: "", model: "", status: "", target: ""};
let detailToken = 0, detailDoc = null, detailId = "", detailStamp = "";
let piStamp = "", turnsStamp = "", turnsRows = [], turnsCursor = null;
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let restoredDetailView = null;
let currentDetail = "same";
const panelLoads = [];
const hosts = {};
function host(id) {
  return hosts[id] = hosts[id] || {
    id, value: "", innerHTML: "",
    classList: {toggle() {}, contains() { return false; }, add() {}, remove() {}},
    setAttribute() {}, scrollIntoView() {},
    addEventListener: (_type, handler) => { hosts[id].on_toggle = handler; },
  };
}
let toggleHandler = null;
const box = {
  dataset: {panel: "feedback"}, _open: false,
  set open(value) {
    const changed = this._open !== value;
    this._open = value;
    if (changed) queueMicrotask(() => toggleHandler({target: this}));
  },
  get open() { return this._open; },
};
globalThis.document = {
  getElementById: host,
  querySelector: () => box,
  querySelectorAll: () => [],
};
globalThis.location = {pathname: "/"};
globalThis.history = {pushState() {}, replaceState() {}};
let rightPromise = null;
const suppressedFeedbackToggles = new Map();
function suppressToggle(box, open) { suppressedFeedbackToggles.set(box, Boolean(open)); }
function showRight() { rightPromise = loadDetail("run"); }
function resetPanels() { openPanels = new Set(); panelStamps = {}; panelTokens = {}; }
function showSection() {}
function show() {}
function loadRuns() {}
function loadCompare() {}
async function openCluster() {}
function loadPanel(name) { panelLoads.push(name); }
function renderDetail() { return '<details class="fold" data-panel="feedback"></details>'; }
function patchDetail() {}
function restoreDetailOpenState() {}
function detailScroller() { return null; }
function rememberDetailScroll() {}
function recallDetailScroll() {}
function carryPanels() {}
function bindPanels(target) { toggleHandler = (event) => {
  const item = event.target;
  if (suppressedFeedbackToggles.has(item)) {
    if (Boolean(item.open) === suppressedFeedbackToggles.get(item)) suppressedFeedbackToggles.delete(item);
    return;
  }
  if (!item.dataset || !item.dataset.panel) return;
  if (item.open) { openPanels.add(item.dataset.panel); currentPanel = item.dataset.panel; loadPanel(item.dataset.panel, detailId); }
  else { openPanels.delete(item.dataset.panel); }
}; target.addEventListener("toggle", toggleHandler); }
function paintDetailHead() {}
function detailPatchedRun(host) { return host && host.dataset ? (host.dataset.patchRun || "") : ""; }
function overlayClose() {}
function detailScroller() { return null; }
function rememberDetailScroll() {}
function recallDetailScroll() {}
function bindCandidatePicks() {}
function bindFeedback() {}
function openStage() {}
function planDetailRequests() { return {detail: true, turns: false, live: false, costs: false}; }
function refreshTurns() {}
function refreshLive() {}
function loadSessionCosts() {}
function esc(value) { return String(value ?? ""); }
globalThis.fetch = async (url) => {
  if (String(url).startsWith("/api/v2/runs/run") && !String(url).includes("/feedback")) {
    if (currentDetail === "same") return {ok: true, json: async () => ({same: true})};
    return {ok: true, json: async () => ({id: "run", detail_stamp: "d1", candidates: [], feedback: {records: []}})};
  }
  return {ok: true, json: async () => ({stamp: "p1", feedback: {records: [{seq: 4, claims: []}]}})};
};

bindPanels({addEventListener: () => {}});
async function settle() {
  for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setImmediate(resolve));
}
(async () => {
  // 冷开/完整恢复：只有一条深链面板请求。
  for (const mode of ["same", "fresh"]) {
    currentDetail = mode;
    panelLoads.length = 0;
    box._open = false;
    openPanels = new Set();
    detailDoc = mode === "same" ? {id: "run"} : null;
    detailId = mode === "same" ? "run" : "";
    detailStamp = mode === "same" ? "d1" : "";
    pendingPanel = "";
    applyViewState(readUrl("?run=run&panel=feedback&seq=4"), {full: true});
    if (!rightPromise) throw new Error("detail was not loaded");
    await rightPromise;
    await settle();
    if (panelLoads.length !== 1) throw new Error(mode + " loaded the panel " + panelLoads.length + " times: " + JSON.stringify(panelLoads));
    if (panelLoads[0] !== "feedback") throw new Error(mode + " loaded the wrong panel: " + panelLoads[0]);
  }

  // 评审回归：feedback 已打开、同 run 已有 detailDoc、full:false；刷新计划必须让开 pendingPanel。
  for (const mode of ["same", "fresh"]) {
    currentDetail = mode;
    panelLoads.length = 0;
    box._open = false;
    openPanels = new Set(["feedback"]);
    detailDoc = {id: "run"};
    detailId = "run";
    detailStamp = "d1";
    pendingPanel = "";
    applyViewState(readUrl("?run=run&panel=feedback&seq=4"));
    await rightPromise;
    await settle();
    if (panelLoads.length !== 1) throw new Error("same-run navigation " + mode + " loaded the panel " + panelLoads.length + " times: " + JSON.stringify(panelLoads));
    if (panelLoads[0] !== "feedback") throw new Error("same-run navigation " + mode + " loaded the wrong panel: " + panelLoads[0]);
  }
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_only_the_primary_deep_link_panel_consumes_the_seq_in_reverse_completion_order():
    """@brief 评审回归：feedback 主深链 + candidates 副面板反序完成，seq 只能由主面板消费。

    旧实现里任何面板完成都会清 `pendingSeq`，candidates 先完成时会把全文请求发在反馈宿主渲染之前。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    page = Path("observe/static/index.html").read_text()
    names = ("panelKey", "panelUrl", "consumePendingPanel", "loadPanel", "applyPendingDetail")
    script = "\n".join(_extract(page, name) for name in names) + r'''
let candidateContentEpoch = 0, feedbackContentEpoch = 0;
function syncDetailSectionCache() {}
let navigationEpoch = 0;
let detailId = "run", detailDoc = {id: "run", candidates: [], feedback: {records: []}};
let panelStamps = {}, panelTokens = {};
const panelInflight = new Map();
let panelRequestSeq = 0;
let pendingPanel = "feedback", pendingCandidate = "", pendingSeq = 4;
let pendingPanelOwner = "";
let openPanels = new Set(["feedback"]);
const fullFeedback = new Map();
const order = [];
const recordCalls = [];
const hosts = {};
function host(id) {
  return hosts[id] = hosts[id] || {id, innerHTML: "", dataset: {}, classList: {add() {}, remove() {}, contains() { return false; }}};
}
globalThis.document = {
  getElementById: host,
  querySelector: () => ({dataset: {panel: "feedback"}, open: true}),
};
function panelQuery() { return "limit=10"; }
function panelBody(name) { order.push("render:" + name); return "BODY:" + name; }
function panelLabel(name) { return name; }
function esc(value) { return String(value ?? ""); }
function bindCandidatePicks() {}
function selectCandidate(label) { order.push("candidate:" + label); }
function loadFeedbackRecord(seq) { recordCalls.push("/feedback/" + seq); order.push("record"); }
function mergePanel(name, doc, payload) {
  if (name === "candidates") doc.candidates = payload.candidates || [];
  if (name === "feedback") doc.feedback = payload.feedback || {};
}
const pending = new Map();
globalThis.fetch = (url) => new Promise((resolve) => pending.set(String(url), resolve));
const respond = (part, doc) => {
  const url = [...pending.keys()].find((key) => key.includes(part));
  if (!url) throw new Error("no pending request for " + part + ": " + JSON.stringify([...pending.keys()]));
  const resolve = pending.get(url);
  pending.delete(url);
  resolve({ok: true, json: async () => doc});
};
const tick = async () => { for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
(async () => {
  applyPendingDetail();
  if (pendingPanelOwner !== "feedback") throw new Error("the deep-link owner was not recorded");
  await tick();
  const candidateLoad = loadPanel("candidates", "run");
  await tick();
  // 副面板先完成。
  respond("/candidates", {candidates: [], total: 0, next: null, stamp: "c"});
  await candidateLoad;
  await tick();
  if (recordCalls.length !== 0) throw new Error("a secondary panel consumed the seq: " + JSON.stringify(recordCalls));
  if (pendingSeq !== 4) throw new Error("pendingSeq was cleared by a secondary panel");
  // 主面板随后完成，这时宿主已经渲染。
  respond("/feedback", {feedback: {records: []}, stamp: "f"});
  await tick();
  if (order.join(",") !== "render:candidates,render:feedback,record") {
    throw new Error("record request order is wrong: " + order.join(","));
  }
  if (recordCalls.length !== 1 || recordCalls[0] !== "/feedback/4") throw new Error("wrong record request: " + JSON.stringify(recordCalls));
  if (pendingSeq !== 0 || pendingPanelOwner !== "") throw new Error("the owner did not consume the seq exactly once");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)
