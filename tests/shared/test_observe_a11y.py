"""§10.5 交互：roving tabindex、键盘序列、模态焦点圈定与 Escape、触摸命中区。

Node 这一层没有浏览器的布局树，所以这里断言纯状态与假元素上的 `activeElement` 序列；
真实命中区像素、焦点是否被 CSS 隐藏、`prefers-reduced-motion` 属于人工清单。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _page() -> str:
    """@brief 页面源码。

    @return HTML 文本。
    """
    return Path("observe/static/index.html").read_text()


def _extract(page: str, name: str) -> str:
    """@brief 从页面抽一个函数（按参数表之后的第一个 `{` 配对）。

    @param page 页面源码。
    @param name 函数名。
    @return 函数源码。
    """
    prefix = f"async function {name}(" if f"async function {name}(" in page else f"function {name}("
    start = page.index(prefix)
    paren = page.index(")", start)
    brace = page.index("{", paren)
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
    """@brief 跑一段 Node 脚本并断言输出 ok。

    @param script 脚本文本。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_roving_tabindex_has_exactly_one_stop_and_moves_with_arrows():
    """@brief roving tabindex：一张表只有一个 0，↑↓ 夹在两端，Home/End 到端点。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in ("rovingIndex", "rovingTabIndexes")) + r'''
if (JSON.stringify(rovingTabIndexes(5, 2)) !== JSON.stringify([-1, -1, 0, -1, -1])) throw new Error("roving tabindex is wrong");
if (JSON.stringify(rovingTabIndexes(3, 99)) !== JSON.stringify([-1, -1, 0])) throw new Error("an out-of-range active index must clamp");
if (rovingIndex(2, "ArrowDown", 5) !== 3) throw new Error("ArrowDown did not move down");
if (rovingIndex(4, "ArrowDown", 5) !== 4) throw new Error("ArrowDown must stop at the last row");
if (rovingIndex(0, "ArrowUp", 5) !== 0) throw new Error("ArrowUp must stop at the first row");
if (rovingIndex(3, "Home", 5) !== 0 || rovingIndex(1, "End", 5) !== 4) throw new Error("Home/End are wrong");
if (rovingIndex(1, "x", 5) !== 1) throw new Error("an unrelated key must not move the stop");
if (rovingIndex(0, "ArrowDown", 0) !== -1) throw new Error("an empty table has no stop");
console.log("ok");
'''
    _node(script)


def test_opening_a_modal_focuses_the_first_control_and_closing_restores_the_opener():
    """@brief 模态打开→首个控件，关闭→原元素；嵌套时 Escape 全部关闭并按栈恢复。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in (
        "focusableElements", "overlayOpen", "overlayClose", "overlayCloseAll")) + r'''
const overlayStack = [];
const overlayOpeners = new Map();
function closeReviewMenu() {}
const active = (id, children) => ({
  id, hidden: false, dataset: {}, _children: children || [],
  classList: {_set: new Set(), add(value) { this._set.add(value); }, remove(value) { this._set.delete(value); }, contains(value) { return this._set.has(value); }},
  focus() { globalThis.document.activeElement = this; },
  querySelectorAll() { return this._children.filter((node) => !node.hidden); },
  querySelector() { return this.querySelectorAll()[0] || null; },
  contains(node) { return node === this || this._children.includes(node); },
});
const opener = active("opener");
const first = active("first");
const last = active("last");
const modal = active("m1", [first, last]);
const first2 = active("first2");
const modal2 = active("m2", [first2]);
const registry = {m1: modal, m2: modal2};
globalThis.document = {
  activeElement: opener,
  getElementById: (id) => registry[id] || null,
};
opener.focus();
overlayOpen("m1");
if (document.activeElement !== first) throw new Error("open must focus the first control, got " + document.activeElement.id);
if (!modal.classList.contains("open")) throw new Error("open did not mark the modal");
overlayOpen("m2");
if (document.activeElement !== first2) throw new Error("the nested modal did not take focus");
overlayCloseAll();
if (document.activeElement !== opener) throw new Error("closing must restore the original opener, got " + document.activeElement.id);
if (document.activeElement === first2 || modal.classList.contains("open") || modal2.classList.contains("open")) throw new Error("Escape did not close every overlay");
if (overlayStack.length !== 0) throw new Error("the overlay stack is not empty");
opener.focus();
overlayOpen("m1");
overlayClose("m1");
if (document.activeElement !== opener) throw new Error("closing one overlay must restore its opener");
overlayClose("m1");
if (document.activeElement !== opener) throw new Error("closing a closed overlay must not steal focus");
console.log("ok");
'''
    _node(script)


def test_the_page_wires_keyboard_and_touch_targets():
    """@brief 结构检查：行可聚焦、模态走 overlay 助记、Escape 关浮层、触摸设备有命中区。

    @raises AssertionError 缺少任一入口时抛出。
    """
    page = _page()
    assert page.count('classList.add("open")') == 1 and page.count('classList.remove("open")') == 1
    assert page.count("overlayOpen(") >= 10 and page.count("overlayClose(") >= 15
    assert "overlayCloseAll()" in page and 'event.key === "Escape"' in page
    assert 'overlayStack[overlayStack.length - 1]' in page
    assert "function applyRovingRows(" in page and "rowKeyAction(" in page
    assert 'applyRovingRows(runRows, 0)' in page
    assert 'host.addEventListener("keydown"' in page
    assert "Shift+F10" in page or 'event.key === "ContextMenu"' in page
    assert "openReviewMenu(" in page
    assert "@media (pointer: coarse)" in page
    assert ".quiet::after" in page and "inset: -10px" in page
    assert "#runs tbody tr:focus-visible" in page


def test_the_copy_entry_point_writes_the_clipboard_and_falls_back():
    """@brief id 复制入口：优先 clipboard，失败回退 prompt；点到行内按钮不冒泡去选中。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in ("copyText", "handleCopyClick")) + r'''
const written = [];
Object.defineProperty(globalThis, "navigator", {
  value: {clipboard: {writeText: async (text) => { written.push(text); }}}, configurable: true,
});
globalThis.window = {prompt: (_label, text) => { written.push("prompt:" + text); }};
const prevented = [];
const button = {dataset: {copy: "conv2d/one"}};
const event = {
  target: {closest: (selector) => (selector === "[data-copy]" ? button : null)},
  preventDefault() { prevented.push("default"); },
  stopPropagation() { prevented.push("propagation"); },
};
(async () => {
  const handled = await handleCopyClick(event);
  if (handled !== true) throw new Error("a copy click must be handled");
  if (written.length !== 1 || written[0] !== "conv2d/one") throw new Error("clipboard did not get the id: " + JSON.stringify(written));
  if (prevented.join(",") !== "default,propagation") throw new Error("a row-level copy click must not select the row");
  const plain = await handleCopyClick({target: {closest: () => null}});
  if (plain !== false) throw new Error("a non-copy click must be ignored");
  Object.defineProperty(globalThis, "navigator", {
    value: {clipboard: {writeText: async () => { throw new Error("blocked"); }}}, configurable: true,
  });
  await copyText("sha");
  if (!written.includes("prompt:sha")) throw new Error("a blocked clipboard must fall back to prompt: " + JSON.stringify(written));
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_the_real_global_key_handler_traps_tab_and_closes_on_escape():
    """@brief 直接调用生产 `handleGlobalKeydown`：Tab 圈定焦点，Escape 关浮层并回原元素。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in (
        "focusableElements", "overlayOpen", "overlayClose", "overlayCloseAll", "handleGlobalKeydown")) + r'''
const overlayStack = [], overlayOpeners = new Map();
const suppressedFeedbackToggles = new Map();
function closeReviewMenu() {}
function makeNode(id) {
  return {id, hidden: false, dataset: {}, classList: {_s: new Set(), add(v) { this._s.add(v); }, remove(v) { this._s.delete(v); }, contains(v) { return this._s.has(v); }},
    focus() { globalThis.document.activeElement = this; }, querySelectorAll() { return this._children || []; },
    querySelector() { return (this._children || [])[0] || null; }, contains(node) { return node === this || (this._children || []).includes(node); }};
}
const first = makeNode("first"), last = makeNode("last");
const modal = makeNode("modal"); modal._children = [first, last];
const opener = makeNode("opener");
globalThis.document = {activeElement: opener, getElementById: (id) => (id === "modal" ? modal : (id === "review-menu" ? {hidden: true} : null))};
overlayOpeners.set("modal", opener);
modal.classList.add("open");
overlayStack.push("modal");
document.activeElement = last;
let prevented = 0;
handleGlobalKeydown({key: "Tab", shiftKey: false, preventDefault() { prevented += 1; }});
if (document.activeElement !== first) throw new Error("Tab did not wrap to the first control");
handleGlobalKeydown({key: "Tab", shiftKey: true, preventDefault() { prevented += 1; }});
if (document.activeElement !== last) throw new Error("Shift+Tab did not wrap to the last control");
handleGlobalKeydown({key: "Escape", preventDefault() { prevented += 1; }});
if (overlayStack.length !== 0) throw new Error("Escape did not close the overlay stack");
if (document.activeElement !== opener) throw new Error("Escape did not restore the opener focus");
if (prevented !== 3) throw new Error("the real handler did not prevent the default for handled keys");
console.log("ok");
'''
    _node(script)


def test_the_retired_detail_drawer_is_gone_and_escape_keeps_closing_overlays():
    """@brief 侧栏抽屉已停用：相关 JS/CSS/DOM 全部清掉，Escape 仍只负责关浮层。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    for gone in ("setDetailAside", "toggleDetailAside", "closeDetailAside", "handleDetailAsideClick",
                 "bindDetailAsideEvents", "aside-open", "detail-aside-toggle", "detail-aside-close",
                 'id="detail-aside"'):
        assert gone not in page, f"已停用的抽屉代码还在：{gone}"
    script = "\n".join(_extract(page, name) for name in (
        "overlayClose", "overlayCloseAll", "handleGlobalKeydown")) + r"""
const overlayStack = ["modal"];
const overlayOpeners = new Map();
const menu = {hidden: true};
const modal = {classList: {add() {}, remove() {}, toggle() {}, contains: () => false}};
globalThis.document = {
  activeElement: null,
  getElementById: (id) => (id === "modal" ? modal : (id === "review-menu" ? menu : null)),
};
let prevented = 0;
handleGlobalKeydown({key: "Escape", preventDefault() { prevented += 1; }});
if (overlayStack.length !== 0) throw new Error("Escape did not close the overlay stack");
if (prevented !== 1) throw new Error("Escape did not prevent the default for a handled overlay");
let untouched = 0;
handleGlobalKeydown({key: "Escape", preventDefault() { untouched += 1; }});
if (untouched !== 0) throw new Error("Escape prevented the default with nothing to close");
console.log("ok");
"""
    _node(script)


def test_real_run_keyboard_sequence_commits_the_final_url_without_a_reducer():
    """@brief 生产 handleRunRowKey→applyViewState→currentViewState→commitUrl：Enter/Space 写最终 URL，方向键不动 URL。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    functions = "\n".join(_extract(page, name) for name in (
        "writeUrl", "readUrl", "currentViewState", "applyViewState", "commitUrl",
        "rovingIndex", "rovingTabIndexes", "applyRovingRows", "rowKeyAction", "handleRunRowKey"))
    script = fields + "\n" + functions + r"""
const FILTER_KEYS = [], SORT_VALUES = ["started"];
let visible = "detail", sideView = "runs", selectedRun = "old", selectedCandidate = "", selectedExperiment = "";
let currentPanel = "", currentSeq = 4, currentCluster = "old-cluster", candidateStatus = "", candidateStage = "";
let runsSort = "started", openPanels = new Set(["lineage"]), turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let pendingPanel = "", pendingCandidate = "", pendingPanelOwner = "", navigationEpoch = 0;
let restoredDetailView = null;
function resetPanels() {}
function showSection() {}
function showRight() {}
function loadCompare() {}
function openCluster() {}
const location = {pathname: "/", search: "?run=old&panel=lineage&seq=4"};
const calls = [];
globalThis.location = location;
globalThis.history = {
  pushState(_state, _title, url) { calls.push({url, push: true}); location.search = url; },
  replaceState(_state, _title, url) { calls.push({url, push: false}); location.search = url; },
};
const rows = [
  {dataset: {run: "r/a"}, tabIndex: 0, focus() { this.focused = true; }},
  {dataset: {run: "r/b"}, tabIndex: -1, focus() { this.focused = true; }},
];
const body = {querySelectorAll: () => rows};
function press(key, row) {
  let prevented = 0;
  const handled = handleRunRowKey({key, target: {closest: (selector) => (selector === "[data-run]" ? row : null)}, preventDefault() { prevented += 1; }}, body);
  return {handled, prevented};
}
let result = press("Enter", rows[1]);
if (!result.handled || result.prevented !== 1) throw new Error("Enter was not handled");
if (calls.length !== 1 || !calls[0].push) throw new Error("Enter did not push a URL");
let params = new URLSearchParams(location.search);
if (params.get("run") !== "r/b" || params.get("panel") !== null || params.get("seq") !== null) throw new Error("Enter URL is wrong: " + location.search);
result = press(" ", rows[1]);
if (!result.handled) throw new Error("Space was not handled");
params = new URLSearchParams(location.search);
if (params.get("run") !== "r/b" || params.get("panel") !== "candidates") throw new Error("Space URL is wrong: " + location.search);
const readBack = readUrl(location.search);
if (readBack.run !== "r/b" || readBack.panel !== "candidates") throw new Error("production readUrl did not round-trip the committed URL");
const beforeArrow = location.search;
result = press("ArrowDown", rows[0]);
if (!result.handled || result.prevented !== 1) throw new Error("ArrowDown must move focus, not commit a URL");
if (location.search !== beforeArrow) throw new Error("ArrowDown changed the URL");
if (!rows[1].focused || rows[1].tabIndex !== 0) throw new Error("ArrowDown did not move the roving focus");
console.log("ok");
"""
    _node(script)


def test_experiment_and_query_rows_have_keyboard_entry_points():
    """@brief experiment/query 行不再只靠鼠标：生产键盘处理器能触发选择/打开。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in (
        "rovingIndex", "rovingTabIndexes", "applyRovingRows", "rowKeyAction",
        "chooseExperiment", "handleExperimentRowKey", "activateQueryRow", "handleQueryRowKey")) + r'''
let selectedExperiment = "", runsCursor = null, runsNext = null;
const runsStack = [];
let compareCalls = 0, runLoads = 0, applied = [], closed = 0;
function syncRunUrl() {}
function loadCompare() { compareCalls += 1; }
function loadRuns() { runLoads += 1; }
function applyViewState(state) { applied.push(state); }
function commitUrl() {}
function currentViewState() { return {}; }
function overlayClose() { closed += 1; }
const expRows = [
  {dataset: {exp: "e1"}, tabIndex: 0, classList: {toggle() {}}, focus() { this.focused = true; }},
  {dataset: {exp: "e2"}, tabIndex: -1, classList: {toggle() {}}, focus() { this.focused = true; }},
];
const expHost = {querySelectorAll: () => expRows};
applyRovingRows(expRows, 0);
handleExperimentRowKey({key: "ArrowDown", target: {closest: (selector) => (selector === "[data-exp]" ? expRows[0] : null)}, preventDefault() {}}, expHost);
if (!expRows[1].focused) throw new Error("experiment ArrowDown did not move focus");
handleExperimentRowKey({key: "Enter", target: {closest: (selector) => (selector === "[data-exp]" ? expRows[1] : null)}, preventDefault() {}}, expHost);
if (selectedExperiment !== "e2" || compareCalls !== 1 || runLoads !== 1) throw new Error("experiment Enter did not choose");
const queryRows = [{dataset: {queryRun: "run/a", queryLabel: "c1"}, tabIndex: 0, focus() {}}];
const queryBody = {querySelectorAll: () => queryRows};
applyRovingRows(queryRows, 0);
handleQueryRowKey({key: "Enter", target: {closest: (selector) => (selector === "[data-query-run]" ? queryRows[0] : null)}, preventDefault() {}}, queryBody);
if (applied.length !== 1 || applied[0].run !== "run/a" || applied[0].panel !== "candidates") throw new Error("query Enter did not open the row");
if (closed !== 1) throw new Error("query Enter did not close the modal");
console.log("ok");
'''
    _node(script)


def test_the_review_menu_restores_focus_to_its_opener():
    """@brief 右键菜单打开时记住 opener，关闭时把焦点还回去。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in ("openReviewMenu", "closeReviewMenu")) + r'''
let reviewMenuOpener = null;
let focused = false;
const opener = {focus() { focused = true; }};
const menu = {hidden: true, innerHTML: "", dataset: {}, style: {}, getBoundingClientRect: () => ({width: 100, height: 80}), querySelector: () => ({focus() {}})};
const section = {
  dataset: {}, classList: {contains: () => false},
  closest: () => null, querySelector: () => ({open: true}), querySelectorAll: () => [],
  matches: () => true,
};
function reviewMenuActions() { return []; }
function esc(value) { return String(value ?? ""); }
globalThis.document = {activeElement: opener, getElementById: (id) => (id === "review-menu" ? menu : null)};
globalThis.window = {innerWidth: 800, innerHeight: 600};
openReviewMenu(section, 10, 10);
if (reviewMenuOpener !== opener || menu.hidden) throw new Error("the menu did not remember its opener");
closeReviewMenu();
if (menu.hidden !== true || !focused) throw new Error("closing the menu did not restore focus");
console.log("ok");
'''
    _node(script)
