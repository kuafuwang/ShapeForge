"""§10.6 视觉：token 表、浅/暗两套对比度、数值列与长 id 的模板片段。

Node/pytest 能测的只有“从源码算出来的颜色与字符串”；容器宽度、真实命中区与
``prefers-reduced-motion`` 的浏览器降级属于人工清单，不在这里假装覆盖。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

def _node(script: str) -> None:
    """@brief 跑一段 Node 脚本。

    @param script 脚本文本。
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def _extract(page: str, name: str) -> str:
    """@brief 从页面抽一个函数（含 async 前缀与完整函数体）。

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


LIGHT_PAIRS = [
    ("ink", "card", 4.5), ("ink", "bg", 4.5),
    ("muted", "card", 4.5), ("muted", "bg", 4.5),
    ("muted", "blue-soft", 4.5), ("muted", "surface-muted", 4.5), ("muted", "surface-info", 4.5),
    ("muted-strong", "amber-soft", 4.5),
    ("blue", "blue-soft", 4.5), ("blue", "card", 4.5),
    ("green", "card", 4.5), ("red", "card", 4.5), ("amber", "card", 4.5),
    ("on-strong", "red", 4.5), ("on-strong", "amber", 4.5),
    ("amber-ink", "amber-soft", 4.5),
    ("green-ink", "green-soft", 4.5),
    ("red-ink", "red-soft", 4.5),
    ("ink-soft", "surface-muted", 4.5), ("ink-soft", "card", 4.5),
    ("muted", "surface-sunken", 4.5), ("ink", "surface-sunken", 4.5),
    ("blue", "surface-sunken", 4.5),
    ("green", "surface-muted", 4.5), ("amber", "surface-muted", 4.5),
    ("focus", "card", 3.0), ("line-strong", "card", 3.0),
]


def _page() -> str:
    """@brief 页面源码。

    @return HTML 文本。
    """
    return Path("observe/static/index.html").read_text()


def _style(page: str) -> str:
    """@brief 取出样式表文本。

    @param page 页面源码。
    @return `<style>` 内的 CSS。
    """
    return page[page.index("<style>") + len("<style>"):page.index("</style>")]


def _declarations(block: str) -> dict[str, str]:
    """@brief 把一段 `--name: value;` 声明读成字典。

    @param block CSS 文本。
    @return 变量名到值（去掉分号）。
    """
    found = {}
    for name, value in re.findall(r"--([\w-]+)\s*:\s*([^;]+);", block):
        found[name.strip()] = value.strip()
    return found


def _root_tokens(page: str) -> dict[str, str]:
    """@brief 浅色 `:root` 的 token 表。

    @param page 页面源码。
    @return 变量字典。
    """
    css = _style(page)
    start = css.index(":root {")
    end = css.index("/* 暗色", start)
    return _declarations(css[start:end])


def _dark_tokens(page: str) -> tuple[dict[str, str], dict[str, str]]:
    """@brief 暗色的两套入口：`[data-theme=dark]` 与媒体查询。

    @param page 页面源码。
    @return ``(attribute, media)`` 两张表。
    """
    css = _style(page)
    attribute = _declarations(css[css.index('[data-theme="dark"] {'):css.index("@media (prefers-color-scheme: dark)")])
    media_start = css.index("@media (prefers-color-scheme: dark)")
    media = _declarations(css[media_start:css.index("@container", media_start) if "@container" in css[media_start:] else len(css)])
    return attribute, media


def _rgb(value: str) -> tuple[int, int, int]:
    """@brief 解析 `#rgb` / `#rrggbb`。

    @param value 颜色文本。
    @return 0..255 三元组。
    """
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    assert len(text) == 6, value
    return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _luminance(value: str) -> float:
    """@brief WCAG 相对亮度。

    @param value 颜色文本。
    @return 0..1。
    """
    def channel(raw: int) -> float:
        ratio = raw / 255
        return ratio / 12.92 if ratio <= 0.03928 else ((ratio + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(item) for item in _rgb(value))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: str, background: str) -> float:
    """@brief 两个颜色的对比度。

    @param foreground 前景。
    @param background 背景。
    @return 对比度倍数。
    """
    high, low = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _assert_pairs(tokens: dict[str, str], label: str) -> None:
    """@brief 按显式表断言每一对前景/背景。

    @param tokens 变量表。
    @param label 浅色/暗色标签。
    @raises AssertionError 任一对不足阈值时抛出。
    """
    for foreground, background, minimum in LIGHT_PAIRS:
        assert foreground in tokens, f"{label} 缺 token --{foreground}"
        assert background in tokens, f"{label} 缺 token --{background}"
        ratio = _contrast(tokens[foreground], tokens[background])
        assert ratio >= minimum, f"{label}: {foreground} on {background} = {ratio:.2f} < {minimum}"


def test_the_muted_tokens_were_raised_without_touching_backgrounds():
    """@brief 三处低于 AA 的小字改用更深的 muted；琥珀底另用 muted-strong。

    @raises AssertionError 值被改回去时抛出。
    """
    tokens = _root_tokens(_page())
    assert tokens["muted"].lower() == "#646b78"
    assert tokens["muted-strong"].lower() == "#5b6472"
    assert tokens["bg"].lower() == "#f4f5f7" and tokens["card"].lower() == "#ffffff"


def test_every_pair_in_the_explicit_table_clears_aa_in_light_and_dark():
    """@brief 显式 26 对前景/背景在浅色与暗色下都过 AA（UI 元素只要求 3.0）。

    @raises AssertionError 任一对不足时抛出。
    """
    page = _page()
    light = _root_tokens(page)
    attribute, media = _dark_tokens(page)
    assert len(LIGHT_PAIRS) >= 20
    _assert_pairs(light, "light")
    _assert_pairs(attribute, "dark[data-theme]")
    _assert_pairs(media, "dark[media]")
    assert {key: value for key, value in attribute.items()} == {key: value for key, value in media.items()}


def test_the_dark_theme_has_both_entry_points_and_overrides_the_same_tokens():
    """@brief 暗色同时支持系统偏好与 `data-theme`，且覆盖浅色定义过的每个颜色 token。

    @raises AssertionError 覆盖不全时抛出。
    """
    page = _page()
    css = _style(page)
    assert '[data-theme="dark"]' in css
    assert "@media (prefers-color-scheme: dark)" in css
    assert ':root:not([data-theme="light"])' in css
    light = _root_tokens(page)
    attribute, _media = _dark_tokens(page)
    colors = {name for name, value in light.items() if value.startswith("#")}
    missing = sorted(colors - set(attribute))
    assert missing == [], f"暗色没有覆盖: {missing}"


def test_no_var_reference_is_undefined():
    """@brief 全样式表的 `var(--…)` 都必须在 `:root` 定义；`--text` 这类笔误立刻被抓。

    @raises AssertionError 出现未定义变量时抛出。
    """
    page = _page()
    css = _style(page)
    used = set(re.findall(r"var\(--([\w-]+)\)", css))
    defined = set(_root_tokens(page))
    assert sorted(used - defined) == []
    assert "var(--text)" not in css
    assert "#6b7280" not in css


def test_the_numeric_columns_and_long_ids_carry_the_right_markup():
    """@brief 数值列带 `numeric`（tabular-nums + 右对齐）；长 id 有省略号与 title。

    @raises AssertionError 模板或样式缺任一项时抛出。
    """
    page = _page()
    css = _style(page)
    assert ".numeric { font-variant-numeric: tabular-nums; text-align: right; }" in css
    assert "font-variant-numeric: tabular-nums" in css
    assert ".id-text" in css and "text-overflow: ellipsis" in css
    assert "function idText(" in page and 'title="${esc(text)}"' in page
    assert "idText(row.run_id)" in page and "idText(doc.id)" in page
    assert "idText(row.run_id)" in page
    assert '<td class="numeric">${esc(num(row.speedup))}</td>' in page
    assert 'class="numeric">${esc(money(row.spent_usd))}' in page


def test_the_container_query_and_motion_fallbacks_are_present():
    """@brief 详情容器查询 <620px 变单列；动效 120–180ms 且 reduced-motion 全部降级。

    @raises AssertionError 缺少任一项时抛出。
    """
    page = _page()
    css = _style(page)
    assert re.search(r"\.detail-panel\s*\{[^}]*container-type: inline-size", css)
    assert "@container (max-width: 620px)" in css
    assert "@media (max-width: 800px)" in css
    assert "--motion-fast: 120ms; --motion: 180ms;" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "transition-duration: .01ms !important" in css
    assert "--font-body:" in css and "--font-mono:" in css
    assert "--s1: 4px" in css and "--r-pill: 999px" in css


def test_every_numeric_table_cell_in_the_templates_is_marked_numeric():
    """@brief 全表 sweep：模板里显示数字的 `<td>` 都要带 `numeric`（纯交互格除外）。

    @raises AssertionError 有数值格漏标时抛出。
    """
    page = _page()
    cells = re.findall(r"<td[^>]*>.*?</td>", page, re.S)
    # 数值字段名也给一层兜底：`node.round`、`row.published_speedup` 这类不走格式化函数。
    numeric_field = re.compile(
        r"(?:^|_)(round|idx|seq|n|count|sessions|runs|speedup|cost|spent|usd|pct|share|median|gain|batches|cv|delta|tokens|confidence|attempts|reps)(?:_|$)",
        re.I,
    )
    offenders = []
    for cell in cells:
        if 'class="numeric"' in cell:
            continue
        if "<button" in cell:
            # 序号/证据按钮/复制按钮是交互格，不是纯数值列。
            continue
        if re.search(r"esc\((?:num|money2?|pctText|number2)\(", cell):
            offenders.append(" ".join(cell.split())[:120])
            continue
        for field in re.findall(r"esc\(\w+\.(\w+)\)", cell):
            if numeric_field.search(field):
                offenders.append(field + " :: " + " ".join(cell.split())[:100])
                break
    assert offenders == [], offenders


def test_the_muted_strong_token_is_used_on_the_amber_small_text():
    """@brief `--muted-strong` 不能只有定义：琥珀底的 chip 小字必须真的用它。

    @raises AssertionError 只定义不使用时抛出。
    """
    page = _page()
    css = _style(page)
    assert css.count("var(--muted-strong)") >= 1
    assert ".bb-list .chip.near" in css
    amber_block = css[css.index(".bb-list .chip.near"):css.index(".bb-list .chip.near") + 400]
    assert "var(--amber-soft)" in amber_block and "var(--muted-strong)" in amber_block
    # 浅色与暗色下这对组合都要过 4.5（表里也含这一对）。
    light = _root_tokens(page)
    attribute, _media = _dark_tokens(page)
    for tokens, label in ((light, "light"), (attribute, "dark")):
        ratio = _contrast(tokens["muted-strong"], tokens["amber-soft"])
        assert ratio >= 4.5, f"{label}: muted-strong on amber-soft = {ratio:.2f}"


def test_long_ids_in_the_real_templates_use_the_ellipsis_and_copy_helper():
    """@brief 落差/发布史/血缘/查询结果里的 run_id 与 sha256 都走 `idText`，并带复制入口。

    @raises AssertionError 仍有模板直接 `esc` 长 id 时抛出。
    """
    page = _page()
    css = _style(page)
    for fragment in (
        "idText(row.run_id)",                      # 落差 / 运行表 / 查询
        "idText(row.run_id || row.run_id_raw)",    # 发布史
        "idText(row.sha256)",                      # 血缘
    ):
        assert fragment in page, fragment
    assert "title=\"${esc(text)}\"" in page and "data-copy=\"${esc(text)}\"" in page
    assert ".id-text" in css and "text-overflow: ellipsis" in css
    assert ".id-copy" in css
    assert "function handleCopyClick(" in page
    assert 'if (event.target.closest && event.target.closest("button")) return;' in page
    # 预算 chip 的 run id 也走长 id + 复制（客户端重绘与服务端首屏各一处）。
    client_banner = page[page.index("function bannerHtml(report)"):page.index("async function refreshBanner()")]
    assert 'idText(item.id, "", false)' in client_banner
    budget = Path("observe/budget.py").read_text()
    assert 'class="id-text"' in budget and 'data-copy=' not in budget
    # 链接内部不能嵌复制 button（会先触发父导航）；反馈历史/相似与评审列表的复制入口保留。
    assert 'idText(cluster.example.run, "", false)' in page
    assert 'idText(item.id, "", false)' in page
    assert 'esc(row.run_id || "")' not in page
    assert 'copy.dataset.copy = runName;' in page
    assert 'if (event.target.closest && event.target.closest("button")) return;' in page
    assert 'addEventListener("click", (event) => {' in page and "handleCopyClick(event)" in page


def test_prior_run_ids_and_report_paths_use_the_long_id_helper():
    """@brief 生产 priorMarkup/feedbackSourceMarkup：此前结论的 run_id 与证据 report_path 都带省略、title、复制。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    functions = "\n".join(_extract(page, name) for name in (
        "esc", "idText", "reviewStatusClass", "priorMarkup", "feedbackSourceMarkup"))
    script = functions + r"""
const runId = "apple-m4-max/20261025-070409_conv2d_fp16_deepseek-flash_g5yab2vc";
const reportPath = "/Users/example/runs/apple-m4-max/20261025-070409_conv2d_fp16_deepseek-flash_g5yab2vc/candidates/000_r1.p0/report.json";
const prior = priorMarkup({prior_summary: "此前结论", prior: [{run_id: runId, seq: 4, status: "done"}]});
if (!prior.includes('class="id-text"') || !prior.includes('title="' + runId + '"') || !prior.includes('data-copy="' + runId + '"')) throw new Error("prior run_id did not use idText: " + prior);
const source = feedbackSourceMarkup({report: {source: "confirm"}, report_path: reportPath});
if (!source.includes('class="id-text"') || !source.includes('title="' + reportPath + '"') || !source.includes('data-copy="' + reportPath + '"')) throw new Error("report_path did not use idText: " + source);
if (feedbackSourceMarkup(null) !== "") throw new Error("empty record produced a source line");
console.log("ok");
"""
    _node(script)


def test_the_detail_is_one_full_width_column_at_every_width():
    """@brief 详情是单列：成本/回合/反馈都按原顺序在主列，没有侧栏节点。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in ("detailSplitMarkup", "renderDetail")) + r'''
function detailSections() {
  return [
    {key: "title", html: "<h2>T</h2>"},
    {key: "panel-candidates", html: "<details data-panel=\"candidates\"></details>"},
    {key: "turns", html: "<details data-lazy=\"turns\"></details>"},
    {key: "handoff", html: "<details data-fold=\"handoff\"></details>"},
    {key: "panel-feedback", html: "<details data-panel=\"feedback\"></details>"},
  ];
}
const html = renderDetail({});
if (!html.includes('class="detail-split"') || !html.includes('class="detail-main"')) {
  throw new Error("renderDetail did not emit the single-column wrapper: " + html);
}
if (html.includes("detail-aside") || html.includes("aside-open")) {
  throw new Error("the single column still renders the retired aside: " + html);
}
const markers = ['<h2>T</h2>', 'data-panel="candidates"', 'data-lazy="turns"', 'data-fold="handoff"', 'data-panel="feedback"'];
let previous = -1;
for (const marker of markers) {
  const at = html.indexOf(marker);
  if (at <= previous) throw new Error("section out of order or missing: " + marker + " in " + html);
  previous = at;
}
console.log("ok");
'''
    _node(script)
    assert "renderDetailInto(host, runId, sections)" in page
    assert "patchDetailColumn(main, sections, detailSectionCache)" in page
    css = _style(page)
    split = re.search(r"\.detail-split\s*\{[^}]*\}", css).group(0)
    assert "grid-template-columns: minmax(0, 1fr);" in split
    assert "gap: var(--s3);" in split
    assert "minmax(240px, 320px)" not in css, "详情不允许再出现主列+侧栏两列轨道"
    assert ".detail-aside" not in css and "aside-open" not in page
    start = css.index("@container (max-width: 620px)")
    block = css[start:css.index("@media (max-width: 800px)", start)]
    assert "position: fixed" not in block and "translateX(100%)" not in block


def test_review_similar_is_fetched_once_per_run_and_merged_into_the_cards():
    """@brief 展开 run 才取相似结论：同一 rev 只取一次，结果按 run/seq/digest 并回卡片并重画该 run。

    @raises subprocess.CalledProcessError Node 断言失败时抛出。
    """
    page = _page()
    script = _extract(page, "loadReviewSimilar") + r'''
const reviewItems = new Map([["r/one\t1\td1", {run_id: "r/one", seq: 1, digest: "d1"}]]);
const reviewSimilar = new Map();
let reviewRev = 7;
const painted = [];
function paintReviewRun(runId) { painted.push(runId); }
const requested = [];
globalThis.fetch = async (url) => {
  requested.push(url);
  return {ok: true, json: async () => ({items: [{seq: 1, digest: "d1", similar: [{run_id: "r/old", score: 0.98}]}]})};
};
loadReviewSimilar("r/one");
loadReviewSimilar("r/one");
setTimeout(() => {
  if (requested.length !== 1) throw new Error("similar fetched more than once: " + requested.length);
  if (requested[0] !== "/api/v2/runs/r/one/similar") throw new Error("bad url " + requested[0]);
  if (reviewItems.get("r/one\t1\td1").similar.length !== 1) throw new Error("similar was not merged into the item");
  if (painted.join(",") !== "r/one") throw new Error("the run was not repainted after merging");
  console.log("ok");
}, 20);
'''
    _node(script)
    assert "loadReviewSimilar(runId)" in page


def test_a_similar_response_from_an_older_rev_cannot_mark_the_new_rev_done():
    """@brief 回归：rev=1 的 similar 响应晚于 rev=2 返回时，不能冒充 rev=2 的结论，且要重新取。

    完成回调过去用全局 ``reviewRev`` 记 done：旧响应会把 ``2:done`` 写上去，rev=2 从此不再请求。

    @raises subprocess.CalledProcessError Node 断言失败时抛出。
    """
    page = _page()
    script = _extract(page, "loadReviewSimilar") + r'''
const key = "r/one\t1\td1";
const reviewItems = new Map([[key, {run_id: "r/one", seq: 1, digest: "d1"}]]);
const reviewSimilar = new Map();
let reviewRev = 1;
const requests = [];
let resolveFirst = null;
globalThis.fetch = (url) => {
  requests.push(url);
  return new Promise((resolve) => { if (requests.length === 1) resolveFirst = resolve; });
};
function paintReviewRun() {}
loadReviewSimilar("r/one");
if (reviewSimilar.get("r/one") !== "1:loading") throw new Error("first request did not start: " + reviewSimilar.get("r/one"));
reviewRev = 2;  // 反馈在请求飞行中前进
resolveFirst({ok: true, json: async () => ({items: [{seq: 1, digest: "d1", similar: [{run_id: "r/old", score: 0.1}]}]})});
setTimeout(() => {
  if (requests.length !== 2) throw new Error("the new rev did not schedule a fresh request: " + requests.length);
  if (reviewItems.get(key).similar !== undefined) throw new Error("a stale rev=1 payload leaked into the rev=2 item");
  if (reviewSimilar.get("r/one") !== "2:loading") throw new Error("the old response was booked as the new rev: " + reviewSimilar.get("r/one"));
  console.log("ok");
}, 20);
'''
    _node(script)


def test_an_incremental_feedback_rev_takes_the_similar_results_for_the_new_rev():
    """@brief 回归：增量 rev 的 paint 必须看到新 rev（rev 在 paint 之前推进），并真的再取一次。

    过去 ``applyReviewFeed`` 先 paint 再写 ``reviewRev``：paint 里的 loadReviewSimilar 看到旧 rev 的
    ``1:done`` 直接返回，rev 更新后也没有人再调度，这一场的相似结论永远停在旧 rev。

    @raises subprocess.CalledProcessError Node 断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in ("loadReviewSimilar", "applyReviewFeed")) + r'''
let reviewRev = 1;
let reviewPending = 0;
let reviewsCalibration = null;
const reviewItems = new Map();
const reviewSimilar = new Map([["r/one", "1:done"]]);
function reviewKey(item) { return `${item.run_id}\t${item.seq}\t${item.digest}`; }
function updateReviewCount() {}
function paintReviewFilters() {}
function paintReviewList() {}
let requests = 0;
globalThis.fetch = () => { requests += 1; return new Promise(() => {}); };
// 生产 patchReviewDom 的 touched-run 路径：每个变化过的 run 都 paintReviewRun；这里只保留
// 该路径里真正会触发相似加载的那一句（paintReviewRun 在展开时调 loadReviewSimilar）。
function paintReviewRun(runId) { loadReviewSimilar(runId); }
function patchReviewDom(upserts) {
  for (const runId of new Set((upserts || []).map((item) => item.run_id))) paintReviewRun(runId);
}
applyReviewFeed({same: false, snapshot: false, rev: 2, pending_total: 1,
                 upserts: [{run_id: "r/one", seq: 1, digest: "d1"}], removed: []});
if (reviewRev !== 2) throw new Error("the feed revision was not advanced");
if (requests !== 1) throw new Error("the rev=2 paint did not fetch similar data: " + requests);
if (reviewSimilar.get("r/one") !== "2:loading") throw new Error("the cache is still keyed by the old rev: " + reviewSimilar.get("r/one"));
console.log("ok");
'''
    _node(script)


def test_the_committee_handoff_is_a_collapsible_fold_without_a_lazy_request():
    """@brief 委员会交接与候选/种子链一样可折叠，但内容在摘要里，不多发请求。

    @raises AssertionError 它不是 fold、没有标题或被挂上懒加载时抛出。
    """
    page = _page()
    match = re.search(r'<details class="fold" data-key="handoff"[^>]*>', page)
    assert match, "委员会交接必须渲染成 details.fold"
    assert "data-lazy" not in match.group(0) and "data-panel" not in match.group(0)
    assert '<summary><span class="fold-title">委员会交接</span></summary>' in page
    assert 'data-fold="handoff"' in page, "交接块要有自己的展开态键（不是 data-lazy，也不发请求）"


def test_the_committee_handoff_keeps_its_open_state_after_a_refresh():
    """@brief 交接块的展开态进 folds、进 URL，刷新后按 URL 恢复；全程不产生请求。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in (
        "suppressToggle", "restoreDetailOpenState")) + r'''
globalThis.CSS = {escape: (value) => value};
const suppressedFeedbackToggles = new Map();
const openPanels = new Set();
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = true;
const handoff = {dataset: {fold: "handoff"}, open: false};
const boxes = {'details[data-fold="handoff"]': handoff};
const host = {querySelectorAll: () => [], querySelector: (selector) => boxes[selector] || null};
restoreDetailOpenState(host);
if (handoff.open !== true) throw new Error("the handoff fold did not reopen from its remembered state");
if (suppressedFeedbackToggles.get(handoff) !== true) throw new Error("reopening the fold was not suppressed against a spurious toggle");
handoffOpen = false;
restoreDetailOpenState(host);
if (handoff.open !== false) throw new Error("the handoff fold stayed open after the remembered state was cleared");
console.log("ok");
'''
    _node(script)
    fields = next(line.strip() for line in page.splitlines() if line.strip().startswith("const URL_FIELDS"))
    url_script = fields + "\n" + "\n".join(_extract(page, name) for name in ("writeUrl", "readUrl", "currentViewState")) + r'''
const FILTER_KEYS = [], SORT_VALUES = ["started"];
const openPanels = new Set();
let turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = true;
let visible = "detail", sideView = "", selectedRun = "", selectedCandidate = "", selectedExperiment = "",
    applied = {}, runsSort = "started", candidateStatus = "", candidateStage = "", currentPanel = "",
    currentCluster = "", currentSeq = 0;
const state = currentViewState();
if (state.folds !== "handoff") throw new Error("the open handoff fold did not reach the view state: " + state.folds);
const round = readUrl("?" + writeUrl(state));
if (round.folds !== "handoff") throw new Error("the handoff fold did not survive a URL round trip: " + JSON.stringify(round));
console.log("ok");
'''
    _node(url_script)
    assert "handoffOpen = folds.includes(\"handoff\")" in page
    assert "planDetailRequests" in page and "handoff" not in page[page.index("function planDetailRequests"):page.index("function pickFeedbackFull")], \
        "交接块不能进入请求计划"

def test_the_single_column_detail_ships_no_sidebar_toggle():
    """@brief 单列详情不再要侧栏开关/抽屉：死代码已清掉，不留隐藏按钮。

    @raises AssertionError 隐藏的侧栏开关、抽屉节点或对应 JS/CSS 还在时抛出。
    """
    page = _page()
    css = _style(page)
    for gone in ("detail-aside-toggle", "detail-aside-close", "detail-aside", "aside-open",
                 "setDetailAside", "toggleDetailAside", "handleDetailAsideClick"):
        assert gone not in page, f"已停用的侧栏代码还在：{gone}"
        assert gone not in css, f"已停用的侧栏样式还在：{gone}"
    panel_rule = re.search(r"\.detail-panel\s*\{[^}]*\}", css)
    assert panel_rule and "container-type: inline-size" in panel_rule.group(0)
    assert "#detail-body { container-type: inline-size; }" not in css
    start = css.index("@container (max-width: 620px)")
    block = css[start:css.index("@media (max-width: 800px)", start)]
    assert "grid-template-columns: minmax(0, 1fr);" in block


def test_design_tokens_are_used_in_real_rules():
    """@brief 字阶/间距/圆角 token 在组件规则里真的被引用。

    @raises AssertionError token 只定义不使用时抛出。
    """
    page = _page()
    css = _style(page)
    body = css[css.index("  * { box-sizing: border-box; }"):]
    for token in ("--fs-lg", "--fs-sm", "--fs-base", "--s1", "--s2", "--s3", "--r-md", "--r-lg", "--r-xl", "--r-pill"):
        assert f"var({token})" in body, token


TABLE_ALIGNMENT_JS = r"""
function tableColumns(html, label) {
  const head = html.match(/<thead>([\s\S]*?)<\/thead>/);
  const body = html.match(/<tbody>([\s\S]*?)<\/tbody>/);
  if (!head || !body) throw new Error(label + " table lacks thead/tbody");
  const headers = [...head[1].matchAll(/<th((?:\s[^>]*)?)>/g)].map((match) => match[1].includes('class="numeric"'));
  const rows = [...body[1].matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/g)].map((match) =>
    [...match[1].matchAll(/<td((?:\s[^>]*)?)>/g)].map((cell) => cell[1].includes('class="numeric"')));
  if (!rows.length) throw new Error(label + " table has no data row");
  rows.forEach((cells, rowIndex) => {
    if (cells.length !== headers.length) throw new Error(label + " row " + rowIndex + " has " + cells.length + " cells for " + headers.length + " headers");
    headers.forEach((numeric, index) => {
      if (numeric !== cells[index]) throw new Error(label + " column " + index + " header/cell numeric mismatch");
    });
  });
  return {headers, rows};
}
"""


def test_render_runs_rows_aligns_every_header_and_cell_numerically():
    """@brief 执行生产 renderRunsRows：发布是文本，加速比/花费是数值，表头与单元格双向对齐。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    static_header = re.search(r'<table id="runs">.*?</thead>', page, re.S)
    assert static_header, "runs table header not found"
    functions = "\n".join(_extract(page, name) for name in (
        "esc", "num", "money", "idText", "renderRunsRows"))
    script = functions + TABLE_ALIGNMENT_JS + r"""
const header = """ + json.dumps(static_header.group(0)) + r""";
let selectedRun = "", runsRendered = false;
const body = {innerHTML: "", dataset: {keysBound: "1"}, querySelectorAll: () => [], addEventListener() {}};
globalThis.document = {querySelector: (selector) => (selector === "#runs tbody" ? body : null)};
function applyRovingRows() {}
renderRunsRows([{run_id: "a/run", task: "relu@fp16", publication: "confirmed",
  published_speedup: 1.5, spent_usd: 0.25, winner_model: "m", stop_reason: "done", binding: "confirmed", status: "done"}]);
const table = header + "<tbody>" + body.innerHTML + "</tbody></table>";
const columns = tableColumns(table, "runs");
const expected = [false, false, false, true, true, false, false, false, false];
if (JSON.stringify(columns.headers) !== JSON.stringify(expected)) throw new Error("runs header numeric columns are wrong: " + JSON.stringify(columns.headers));
if (!body.innerHTML.includes('class="numeric">$0.250</td>')) throw new Error("runs spent cell is not numeric money");
console.log("ok");
"""
    _node(script)


def test_budget_events_markup_aligns_old_and_new_budget_columns():
    """@brief 执行生产 budgetEventsMarkup：旧/新预算表头与单元格都 numeric，金额两位小数。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    functions = "\n".join(_extract(page, name) for name in (
        "esc", "num", "money2", "newestFirst", "budgetEventsMarkup"))
    script = functions + TABLE_ALIGNMENT_JS + r"""
const table = budgetEventsMarkup({budget_events: [
  {time: "2026-10-25T01:02:03", kind: "topup", who: "dashboard", old_budget_usd: 1, new_budget_usd: 2.5, note: "topup"},
]});
const columns = tableColumns(table, "budget");
const expected = [false, false, false, true, true, false];
if (JSON.stringify(columns.headers) !== JSON.stringify(expected)) throw new Error("budget header numeric columns are wrong: " + JSON.stringify(columns.headers));
if (!table.includes('<td class="numeric">$1.00</td><td class="numeric">$2.50</td>')) throw new Error("budget cells are not numeric money");
console.log("ok");
"""
    _node(script)


def test_patch_detail_column_moves_keyed_sections_when_the_order_reverses():
    """@brief 执行生产 patchDetailColumn：缓存相同的 A,B 换序只移动原节点，嵌套/焦点/滚动对象不变。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    functions = "\n".join(_extract(page, name) for name in ("planDetailPatch", "patchDetailColumn"))
    script = functions + r"""
const removed = [];
let children = [];
function makeNode(key, state) {
  const node = {dataset: {key}, ...state, remove() { removed.push(key); }};
  Object.defineProperty(node, "previousSibling", {get() { const index = children.indexOf(node); return index > 0 ? children[index - 1] : null; }});
  return node;
}
const a = makeNode("A", {open: true, focused: true, scrollTop: 37});
const b = makeNode("B", {open: false, focused: false, scrollTop: 11});
children = [a, b];
const column = {
  get children() { return children; },
  get firstChild() { return children[0] || null; },
  querySelector() { return null; },
  insertBefore(node, ref) {
    if (node === ref) return;
    children = children.filter((child) => child !== node);
    const index = children.indexOf(ref);
    children.splice(index < 0 ? children.length : index, 0, node);
  },
};
patchDetailColumn(column, [{key: "B", html: "B"}, {key: "A", html: "A"}], new Map([["A", "A"], ["B", "B"]]));
if (column.children[0] !== b || column.children[1] !== a) throw new Error("keyed sections were replaced instead of moved");
if (!a.open || !a.focused || a.scrollTop !== 37 || b.scrollTop !== 11) throw new Error("moving a section lost nested open/focus/scroll state");
const keys = column.children.map((node) => node.dataset && node.dataset.key);
if (keys[0] !== "B" || keys[1] !== "A") throw new Error("keyed sections stayed in the old order: " + JSON.stringify(keys));
console.log("ok");
"""
    _node(script)


def test_patch_detail_column_drops_removed_sections_and_their_cache():
    """@brief 执行生产 patchDetailColumn：本轮消失的块连缓存一起删，续跑再出现时必须能重画。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    functions = "\n".join(_extract(page, name) for name in ("planDetailPatch", "patchDetailColumn"))
    script = functions + r"""
const removed = [];
let children = [];
const nextNode = {dataset: {key: "panel-grid"}};
const gridNode = {dataset: {key: "panel-grid"}, remove() { removed.push("panel-grid"); }, replaceWith() {}};
const staleNode = {dataset: {key: "stop"}, remove() { removed.push("stop"); const index = children.indexOf(staleNode); if (index >= 0) children.splice(index, 1); }};
children = [gridNode, staleNode];
const column = {
  get children() { return children; },
  get firstChild() { return children[0] || null; },
  querySelector() { return null; },
  insertBefore(node, ref) {
    if (node === ref) return;
    children = children.filter((child) => child !== node);
    const index = children.indexOf(ref);
    children.splice(index < 0 ? children.length : index, 0, node);
  },
};
globalThis.document = {createElement: () => {
  const holder = {};
  Object.defineProperty(holder, "innerHTML", {set() { holder.firstElementChild = nextNode; }});
  return holder;
}};
const cache = new Map([["panel-grid", "<p>grid</p>"], ["stop", "<p>stop</p>"]]);
patchDetailColumn(column, [{key: "panel-grid", html: "<p>grid</p>"}], cache);
if (!removed.includes("stop")) throw new Error("a section that disappeared was left in the DOM");
if (cache.has("stop")) throw new Error("the patch cache kept a section that disappeared");
if (column.children.length !== 1 || column.children[0] !== gridNode) throw new Error("unchanged section was replaced or reordered");
console.log("ok");
"""
    _node(script)


def test_id_text_inside_links_never_embeds_a_copy_button():
    """@brief 链接内的长 id 只渲染 span，复制 button 不能嵌套进锚点触发父导航。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = _extract(page, "esc") + "\n" + _extract(page, "idText") + r'''
const plain = idText("run/a", "", false);
const copied = idText("run/a");
if (plain.includes("<button")) throw new Error("link-safe id rendered a button");
if (!plain.includes('class="id-text"') || !plain.includes("run/a")) throw new Error("link-safe id lost the span");
if (!copied.includes("<button") || !copied.includes("data-copy")) throw new Error("normal id lost its copy entry");
console.log("ok");
'''
    _node(script)
    assert 'idText(cluster.example.run, "", false)' in page
    assert 'idText(item.id, "", false)' in page
    budget = Path("observe/budget.py").read_text()
    assert "data-copy" not in budget


def test_the_first_screen_boot_keeps_its_known_loader_set():
    """@brief 首屏启动只调用已知的三个 loader；视觉改动不得偷偷增加首屏请求。

    @raises AssertionError 启动块新增 load* 调用时抛出。
    """
    page = _page()
    start = page.index("restoreStartupView(readUrl(location.search))")
    end = page.index("async function handleCopyClick", start)
    boot = page[start:end]
    calls = re.findall(r"^\s*(load[A-Za-z]+)\(", boot, re.M)
    assert calls == ["loadExperiments", "loadRuns", "loadPatrol"], calls


def test_refresh_restores_collapsed_side_runs_and_detail_after_the_real_startup_view():
    """@brief 执行生产 restoreStartupView/applyViewState/showSection/showRight：启动展开后按快照重新折叠。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    functions = "\n".join(_extract(page, name) for name in (
        "setSideCollapsed", "readChromeCollapse", "restoreChromeCollapse", "restoreStartupView",
        "setDetailCollapsed", "setRunsCollapsed", "showSection", "showRight", "applyViewState"))
    script = functions + r"""
const FILTER_KEYS = [], SORT_VALUES = ["started"];
const store = new Map([["observe-side", "collapsed"], ["observe-runs", "collapsed"], ["observe-detail", "collapsed"]]);
globalThis.localStorage = {getItem: (key) => (store.has(key) ? store.get(key) : null), setItem: (key, value) => store.set(key, String(value))};
function classes(initial) {
  const values = new Set(initial);
  return {contains: (name) => values.has(name), toggle: (name, on) => { if (on === undefined ? !values.has(name) : on) values.add(name); else values.delete(name); }, add: (name) => values.add(name), remove: (name) => values.delete(name)};
}
const workspace = {classList: classes(["collapsed"])};
const sideButton = {textContent: "", title: "", setAttribute() {}, click() { setSideCollapsed(false); }};
const runsPanel = {classList: classes(["collapsed"])};
const runsButton = {textContent: "", title: "", setAttribute() {}};
const detailPanel = {classList: classes(["collapsed"])};
const detailButton = {textContent: "", title: "", setAttribute() {}};
const detailView = {classList: classes(["on"])};
const ledgerView = {classList: classes([])};
const runsView = {scrollIntoView() {}};
const runsBody = {replaceChildren() {}};
const runsScroller = {scrollTop: 0};
const nodes = {
  workspace, "side-toggle": sideButton, "runs-panel": runsPanel, "runs-toggle": runsButton,
  "detail-panel": detailPanel, "detail-toggle": detailButton, "view-detail": detailView,
  "view-ledger": ledgerView, "view-runs": runsView,
};
globalThis.document = {
  getElementById: (id) => nodes[id] || null,
  querySelector: (selector) => (selector === "#runs tbody" ? runsBody : (selector === ".runs-table-scroll" ? runsScroller : null)),
};
let selectedRun = "", selectedCandidate = "", selectedExperiment = "", currentSeq = 0;
let candidateStatus = "", candidateStage = "", currentCluster = "", currentPanel = "";
let pendingPanel = "", pendingCandidate = "", pendingSeq = 0, visible = "detail", sideView = "runs";
let runsSort = "started", runsRendered = true, runsRows = [], runsToken = 0, detailToken = 0, navigationEpoch = 0;
let openPanels = new Set(), turnsOpen = false, piOpen = false, costsOpen = false, handoffOpen = false;
let candidatePages = 1, feedbackPages = 1, restoredDetailView = null;
function resetPanels() {}
function paintDetailHead() {}
function saveRunsScroll() {}
function renderRunsRows() {}
function restoreRunsScroll() {}
function loadDetail(id) { globalThis.loadedDetail = id; }
function loadCompare() {}
function openCluster() {}
const state = {view: "runs", run: "r/one"};
restoreStartupView(state);
if (!workspace.classList.contains("collapsed")) throw new Error("side lost its collapsed state");
if (!runsPanel.classList.contains("collapsed")) throw new Error("runs lost its collapsed state");
if (!detailPanel.classList.contains("collapsed")) throw new Error("detail lost its collapsed state");
if (store.get("observe-side") !== "collapsed" || store.get("observe-runs") !== "collapsed" || store.get("observe-detail") !== "collapsed") throw new Error("restore wrote the temporary expanded state back to storage");
if (globalThis.loadedDetail !== "r/one") throw new Error("startup did not run the real showRight/loadDetail path");
console.log("ok");
"""
    _node(script)


def test_the_ledger_tables_are_column_described_and_format_money():
    """@brief 执行生产 loadLedger：次数/花费/预算/删除次数逐列 numeric，金额统一两位小数。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    script = "\n".join(_extract(page, name) for name in (
        "esc", "num", "money2", "money4", "newestFirst", "ledgerTable", "loadLedger")) + r'''
const host = {innerHTML: ""};
globalThis.document = {getElementById: (id) => (id === "view-ledger" ? host : null)};
const ledger = {
  spent_usd: 12.345, carryover_usd: 0.5, runs: 7,
  by_operator: [{operator: "relu", runs: 3, spent_usd: 1.2, budget_usd: 2}],
  by_month: [{month: "2026-10", runs: 3, spent_usd: 1.2, operators: [{operator: "relu", spent_usd: 1.2}]}],
};
const deletions = {deletions: [{t: "2026-10-01", kind: "run", name: "x", ip: "1.2.3.4", reason: "r", runs: [1, 2]}]};
globalThis.fetch = async (url) => ({ok: true, json: async () => (String(url).includes("/ledger") ? ledger : deletions)});
(async () => {
  await loadLedger();
  const html = host.innerHTML;
  if (!html.includes("总额 $12.35") || !html.includes("结转 $0.50") || !html.includes("7 次计入账本")) throw new Error("ledger summary money format is wrong: " + html.slice(0, 120));
  for (const label of ["次数", "花费", "预算"]) {
    if (!html.includes(`<th class="numeric">${label}</th>`)) throw new Error("ledger header not numeric: " + label);
  }
  if (!html.includes('<td class="numeric">3</td><td class="numeric">$1.20</td><td class="numeric">$2.00</td>')) throw new Error("operator numeric/money cells are wrong");
  if (!html.includes("relu $1.20")) throw new Error("monthly operator money is not formatted");
  if (!html.includes('<th class="numeric">次数</th>')) throw new Error("deletion count header not numeric");
  if (!html.match(/<td class="numeric">2<\/td><\/tr>\s*<\/tbody><\/table><p class="muted">删除记录只追加/)) throw new Error("deletion count cell not numeric");
  console.log("ok");
})().catch((error) => { console.error(error); process.exit(1); });
'''
    _node(script)


def test_harness_tables_use_column_metadata_for_headers_and_rows():
    """@brief 执行生产 harnessMarkup：建议/提案/决策/overlay/review 的数值列逐列对齐，金额四位统一。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    columns = page[page.index("const SUGGESTION_COLUMNS"):page.index("function harnessRow")]
    surface_line = next(line.strip() for line in page.splitlines() if line.strip().startswith("const HARNESS_SURFACE"))
    decision_block = page[page.index("const HARNESS_DECISION"):page.index("function harnessReviewCell")]
    functions = "\n".join(_extract(page, name) for name in (
        "esc", "num", "money2", "money4", "harnessWho", "harnessStatus", "harnessEvidence",
        "harnessRow", "harnessTable", "harnessReviewCell", "harnessPromotionCell",
        "harnessOverlayCell", "harnessMarkup"))
    script = columns + surface_line + decision_block + functions + r'''
const doc = {
  suggestions: [{seq: 1, label: "r1", text: "text", provider: "p", model: "m", converted_key: ""}],
  proposals: [{seq: 2, proposal_key: "k", status: "valid", origin: "model", provider: "p", model: "m", committee: "first",
    proposal: {surface: "tip", action: "act", evidence: {probe: "probe"}}}],
  reviews: [{proposal_key: "k", reviewer: {provider: "p", model: "m"}, useful: true, confidence: 0.8,
    evidence: "e", risk: "r", cost_usd: 0.123456}],
  decisions: [{event: "promotion", status: "local", decision: "apply", proposal_key: "k", models: ["m"], source_seq: 2}],
  overlay: [{decision: "apply", surface: "tip", text: "t", protocol_version: "3", source_seq: 2}],
};
const html = harnessMarkup(doc);
const tables = [...html.matchAll(/<table>([\s\S]*?)<\/table>/g)].map((match) => match[1]);
if (tables.length !== 5) throw new Error("expected five harness tables, got " + tables.length);
const parse = (table) => {
  const headers = [...table.matchAll(/<th((?:\s[^>]*)?)>([\s\S]*?)<\/th>/g)].map((m) => ({numeric: m[1].includes('class="numeric"'), label: m[2].replace(/<[^>]+>/g, "")}));
  const cells = [...table.matchAll(/<td([^>]*)>/g)].map((m) => m[1].includes('class="numeric"'));
  return {headers, cells};
};
const parsed = tables.map(parse);
const expectColumns = (table, numeric) => {
  if (!table || table.headers.length !== numeric.length || table.cells.length !== numeric.length) {
    throw new Error("harness column count is misaligned: " + JSON.stringify(table && {headers: table.headers, cells: table.cells}));
  }
  table.headers.forEach((header, index) => {
    if (header.numeric !== numeric[index]) throw new Error("header numeric mismatch at " + index + ": " + JSON.stringify(header));
  });
  table.cells.forEach((cell, index) => {
    if (cell !== numeric[index]) throw new Error("cell numeric mismatch at " + index);
  });
};
expectColumns(parsed.find((table) => table.headers.length === 4 && table.headers[0].label === "轮" && table.headers[1].label === "来源"), [true, false, false, false]);
expectColumns(parsed.find((table) => table.headers[0].label === "轮" && table.headers[1].label === "提出者"), [true, false, false, false, false, false, false, false]);
expectColumns(parsed.find((table) => table.headers[0].label === "reviewer"), [false, false, true, false, false, true]);
expectColumns(parsed.find((table) => table.headers[0].label === "决策"), [false, false, false, true]);
expectColumns(parsed.find((table) => table.headers[0].label === "状态"), [false, false, false, false, true]);
if (!html.includes("$0.1235")) throw new Error("review cost is not four-decimal money");
console.log("ok");
'''
    _node(script)


def test_a_section_that_disappears_and_returns_is_redrawn_from_its_original_html():
    """@brief 生产 `patchDetailColumn`：A,B → A → A,B（HTML 不变）时 B 必须回来，缓存也要同步。

    `stop/seed/winner/pi` 是同 run 续跑后才会重新出现的可选块；删除 leftover 节点时若不清缓存，
    下一次 `planDetailPatch` 会因缓存相同返回空，块永远回不来。

    @raises subprocess.CalledProcessError Node 套里的断言失败时抛出。
    """
    page = _page()
    functions = "\n".join(_extract(page, name) for name in ("planDetailPatch", "patchDetailColumn"))
    script = functions + r"""
let children = [];
function makeNode(key, html) {
  const node = {dataset: {key}, html, remove() { const index = children.indexOf(node); if (index >= 0) children.splice(index, 1); }};
  Object.defineProperty(node, "previousSibling", {get() { const index = children.indexOf(node); return index > 0 ? children[index - 1] : null; }});
  Object.defineProperty(node, "nextSibling", {get() { const index = children.indexOf(node); return index >= 0 && index + 1 < children.length ? children[index + 1] : null; }});
  return node;
}
const a = makeNode("A", "A");
const b = makeNode("B", "B");
children = [a, b];
const column = {
  get children() { return children; },
  get firstChild() { return children[0] || null; },
  querySelector() { return null; },
  insertBefore(node, ref) {
    if (node === ref) return;
    children = children.filter((child) => child !== node);
    const index = children.indexOf(ref);
    children.splice(index < 0 ? children.length : index, 0, node);
  },
};
const cache = new Map([["A", "A"], ["B", "B"]]);
const sectionsAB = [{key: "A", html: "A"}, {key: "B", html: "B"}];
patchDetailColumn(column, sectionsAB, cache);
if (children.length !== 2 || children[0] !== a || children[1] !== b) throw new Error("initial keyed patch changed identities");
patchDetailColumn(column, [{key: "A", html: "A"}], cache);
if (children.length !== 1 || children[0] !== a) throw new Error("missing section was not removed");
if (cache.has("B")) throw new Error("removed section kept its patch cache");
const ops = planDetailPatch(sectionsAB, cache);
if (ops.join(",") !== "B") throw new Error("returning section was not planned: " + JSON.stringify(ops));
globalThis.document = {createElement: () => {
  const holder = {};
  Object.defineProperty(holder, "innerHTML", {set(value) { holder.firstElementChild = makeNode(value, value); }});
  return holder;
}};
patchDetailColumn(column, sectionsAB, cache);
const keys = children.map((node) => node.dataset.key);
if (keys.join(",") !== "A,B") throw new Error("returning section did not come back: " + JSON.stringify(keys));
if (children[1] === b) throw new Error("removed node was reused instead of redrawn");
if (planDetailPatch(sectionsAB, cache).length !== 0) throw new Error("patch cache not synced after re-insert");
console.log("ok");
"""
    _node(script)
