"""§8.3 种子链（derived）与跨场同源（matched）：推导规则、根集合、面板与 UI 文案。

``trace.parent_label`` 在真实语料里 0 条非空，所以这里不冒充“谁编辑自谁”：种子链是由
``improved`` 序列推导的，跨场同源只是 ``kernel_sha256`` 相等。两条事实用不同的 ``edge_kind``。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observe import lineage
from observe.index import Index

_SOURCE = "kernel void lineage_fixture() {}\n"


def _index(tmp_path, monkeypatch):
    """@brief 临时 runs 根 + 索引（关掉读时 tick）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(runs 根, Index)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    return runs, Index(tmp_path / "observe.sqlite")


def _run(runs: Path, name: str, *, task: str = "conv2d") -> Path:
    """@brief 造一个能被 ``resolve_run`` 解析的运行目录（不经过摄入）。

    @param runs runs 根目录。
    @param name run id。
    @param task 任务名。
    @return run 目录。
    """
    directory = runs / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "phase": "done", "started_at": 1.0, "updated_at": 2.0,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "size": "paper", "stop_reason": "max_rounds",
        "protocol_version": "3",
    }), encoding="utf-8")
    return directory


def _candidates(index, run_id: str, sequence, *, shas: dict[str, str] | None = None) -> None:
    """@brief 直接写候选窄列（``improved`` 来自 trace 的投影）。

    @param index 索引。
    @param run_id 运行 id。
    @param sequence ``(label, improved)`` 列表，按下标顺序。
    @param shas 标签到源码摘要的映射。
    """
    shas = shas or {}
    for idx, (label, improved) in enumerate(sequence):
        index.conn.execute(
            "INSERT INTO candidates (run_id, idx, label, status, stage, speedup, improved, kernel_sha256)"
            " VALUES (?, ?, ?, 'ok', 'bench', ?, ?, ?)",
            (run_id, idx, label, 1.0 + idx * 0.1, 1 if improved else 0, shas.get(label)),
        )
    index.conn.commit()


def test_an_init_run_with_eleven_candidates_yields_ten_derived_edges(tmp_path, monkeypatch):
    """@brief 验收①：``init:T, r1.p0:T, r2..r10:F`` → 10 条边，且与 ``improved`` 序列一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/init-run")
    sequence = [("init", True), ("r1.p0", True)] + [(f"r{number}.p0", False) for number in range(2, 11)]
    _candidates(index, "conv2d/init-run", sequence)

    document = lineage.seed_chain("conv2d/init-run")
    assert len(document["nodes"]) == 11 and len(document["edges"]) == 10
    assert document["roots"] == ["init"]
    assert document["edges"][0] == {"from": "init", "to": "r1.p0", "kind": "derived_seed"}
    assert all(edge["from"] == "r1.p0" for edge in document["edges"][1:])
    assert all(edge["kind"] == "derived_seed" for edge in document["edges"])
    assert document["edge_counts"]["derived_seed"] == 10
    assert document["notes"]["derived_seed"] == "按轨迹推导的种子关系，不是作者声明"


def test_a_prior_run_without_an_improvement_yields_two_edges(tmp_path, monkeypatch):
    """@brief 验收②：``prior:T, r1.p0:F, r2.p0:T`` → 2 条边，都是 ``prior`` 起。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/prior-run")
    _candidates(index, "conv2d/prior-run", [("prior", True), ("r1.p0", False), ("r2.p0", True)])

    document = lineage.seed_chain("conv2d/prior-run")
    assert document["roots"] == ["prior"]
    assert document["edges"] == [
        {"from": "prior", "to": "r1.p0", "kind": "derived_seed"},
        {"from": "prior", "to": "r2.p0", "kind": "derived_seed"},
    ]


def test_init_retries_are_all_roots_and_never_get_cross_root_edges(tmp_path, monkeypatch):
    """@brief 验收③：``init2`` / ``init3`` 重试场里每个 ``init*`` 都是根，根之间不连边。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/retry-run")
    _candidates(index, "conv2d/retry-run", [
        ("init", True), ("init2", True), ("init3", True),
        ("r1.p0", False), ("r2.p0", True), ("r3.p0", False),
    ])

    document = lineage.seed_chain("conv2d/retry-run")
    assert document["roots"] == ["init", "init2", "init3"]
    assert len(document["edges"]) == len(document["nodes"]) - len(document["roots"]) == 3
    targets = {edge["to"] for edge in document["edges"]}
    assert not targets & {"init", "init2", "init3"}, "根不能是别人的子节点"
    assert {edge["from"] for edge in document["edges"]} == {"init3", "r2.p0"}
    assert document["edges"][0]["from"] == "init3", "最近的根重置当前种子"


def test_cross_run_source_matches_are_named_as_matches_not_derivations(tmp_path, monkeypatch):
    """@brief 验收：同 sha 的其它场候选进 ``cross_run``，``edge_kind`` 恒非空，两条文案不同。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/older")
    _run(runs, "conv2d/newer")
    shared = "a" * 64
    _candidates(index, "conv2d/older", [("prior", True), ("r1.p0", True)],
                shas={"prior": "b" * 64, "r1.p0": shared})
    _candidates(index, "conv2d/newer", [("prior", True), ("r1.p0", True)],
                shas={"prior": shared, "r1.p0": "c" * 64})

    document = lineage.seed_chain("conv2d/newer")
    assert document["edges"] == [{"from": "prior", "to": "r1.p0", "kind": "derived_seed"}]
    assert document["cross_run_total"] == 1
    match = document["cross_run"][0]
    assert match["run_id"] == "conv2d/older" and match["label"] == "r1.p0"
    assert match["from_label"] == "prior"
    assert match["kind"] == "cross_run_source_match"
    assert match["sha256"] == shared[:12]
    assert document["notes"]["cross_run_source_match"] == "源码相同 ≠ 由它衍生"
    assert all(edge["kind"] for edge in document["edges"])

    panel = lineage.lineage_panel("conv2d/newer")
    assert panel["same"] is False and panel["lineage"]["run_id"] == "conv2d/newer"
    assert lineage.lineage_panel("conv2d/newer", stamp=panel["stamp"]) == {
        "run_id": "conv2d/newer", "same": True, "stamp": panel["stamp"],
    }
    assert lineage.seed_chain("conv2d/missing") is None


def test_the_lineage_panel_stays_under_eight_kilobytes_with_many_candidates(tmp_path, monkeypatch):
    """@brief §8.5：种子链面板也受 8KB 预算约束，且截断如实报告。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/many")
    sequence = [("init", True)] + [(f"r{number}.p0-" + "x" * 40, number % 2 == 0) for number in range(1, 260)]
    _candidates(index, "conv2d/many", sequence)

    panel = lineage.lineage_panel("conv2d/many")
    document = panel["lineage"]
    size = len(json.dumps(panel, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert size <= lineage.LINEAGE_BYTES, f"panel wrapper is {size}B"
    assert document["truncated"]["nodes"] is True
    assert document["truncated"]["edges"] is True
    assert len(document["nodes"]) <= lineage.LINEAGE_LIMIT
    # 完整集统计不随展示裁剪：260 颗候选、1 个根 → 259 条真实边。
    assert document["edge_counts"]["derived_seed"] == 259
    assert document["edges_shown"] < 259

    from starlette.testclient import TestClient

    from observe.app import app

    with TestClient(app) as client:
        response = client.get("/api/v2/runs/conv2d/many/lineage")
        assert response.status_code == 200
        assert len(response.content) <= lineage.LINEAGE_BYTES, f"HTTP response is {len(response.content)}B"


def test_the_detail_panel_names_both_relations_in_node():
    """@brief UI 验收：``lineageMarkup`` 把推导与同源分开写，面板文案不混用。

    @raises subprocess.CalledProcessError Node 脚本里的断言失败时抛出。
    """
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = Path("observe/static/index.html").read_text()
    assert 'panelSection("lineage", doc, openPanels.has("lineage"))' in page

    def extract(name):
        marker = f"async function {name}(" if f"async function {name}(" in page else f"function {name}("
        start = page.index(marker)
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

    names = ("esc", "num", "idText", "lineageMarkup", "panelNames", "panelKey", "panelLabel")
    script = "\n".join(extract(name) for name in names) + r'''
if (!panelNames().includes("lineage")) throw new Error("lineage panel is not wired");
if (panelKey("lineage") !== "lineage") throw new Error("panelKey(lineage) is wrong");
if (panelLabel("lineage") !== "种子链") throw new Error("panel label must be 种子链");
const doc = { run_id: "conv2d/one", roots: ["init"], edge_counts: { derived_seed: 1 },
  notes: { derived_seed: "按轨迹推导的种子关系，不是作者声明", cross_run_source_match: "源码相同 ≠ 由它衍生" },
  edges: [{ from: "init", to: "r1.p0", kind: "derived_seed" }],
  cross_run: [{ from_label: "prior", run_id: "conv2d/older", label: "r1.p0", sha256: "aaaaaaaaaaaa" }],
  nodes: [], truncated: { nodes: false, edges: false, cross_run: false } };
const html = lineageMarkup(doc);
for (const text of ["按轨迹推导的种子关系，不是作者声明", "源码相同 ≠ 由它衍生", "按轨迹推导", "跨场同源"]) {
  if (!html.includes(text)) throw new Error("missing wording: " + text);
}
const derived = html.split("跨场同源")[0];
const matched = html.split("跨场同源")[1] || "";
if (matched.includes("按轨迹推导")) throw new Error("cross-run matches must not be labelled derived");
if (!derived.includes("按轨迹推导")) throw new Error("derived edge label missing");
const empty = lineageMarkup(null);
if (!empty.includes("索引里还没有候选")) throw new Error("empty state missing");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def _ingested_run(runs: Path, index, name: str, sequence) -> Path:
    """@brief 造一个走真实摄取路径的 run：候选目录 + result.trace（含 ``improved``）。

    @param runs runs 根目录。
    @param index 索引。
    @param name run id。
    @param sequence ``(label, improved)`` 列表。
    @return run 目录。
    """
    directory = runs / name
    directory.mkdir(parents=True)
    trace = []
    for idx, (label, improved) in enumerate(sequence):
        folder = directory / "candidates" / f"{idx:03d}_{label}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(_SOURCE + f"// {label}\n", encoding="utf-8")
        (folder / "report.json").write_text(json.dumps({
            "status": "ok", "stage": "bench", "speedup": 1.0 + idx * 0.1,
        }), encoding="utf-8")
        trace.append({
            "idx": idx, "label": label, "status": "ok", "speedup": 1.0 + idx * 0.1,
            "improved": improved,
        })
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": 1.0, "updated_at": 2.0,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "size": "paper", "stop_reason": "max_rounds",
        "protocol_version": "3", "trace": trace,
    }), encoding="utf-8")
    index.sync_known()
    return directory


def test_the_improved_flag_is_projected_from_the_trace_at_ingest(tmp_path, monkeypatch):
    """@brief 种子链读索引窄列：``improved`` 必须是摄入时从 ``result.trace`` 写进来的。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    sequence = [("init", True), ("r1.p0", True), ("r2.p0", False), ("r3.p0", False)]
    _ingested_run(runs, index, "conv2d/one", sequence)

    projected = [
        (str(row["label"]), bool(row["improved"])) for row in index.conn.execute(
            "SELECT label, improved FROM candidates WHERE run_id = 'conv2d/one' ORDER BY idx"
        )
    ]
    assert projected == sequence
    document = lineage.seed_chain("conv2d/one")
    assert len(document["edges"]) == 3
    assert [edge["to"] for edge in document["edges"]] == ["r1.p0", "r2.p0", "r3.p0"]


def test_the_lineage_panel_is_reachable_over_http_and_honours_its_stamp(tmp_path, monkeypatch):
    """@brief HTTP 面板：200 + 负载、stamp 短路、运行不存在时 404。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    runs, index = _index(tmp_path, monkeypatch)
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    _ingested_run(runs, index, "conv2d/one", [("init", True), ("r1.p0", False)])
    with TestClient(app) as client:
        first = client.get("/api/v2/runs/conv2d/one/lineage")
        assert first.status_code == 200
        body = first.json()
        assert body["same"] is False
        assert body["lineage"]["edges"] == [{"from": "init", "to": "r1.p0", "kind": "derived_seed"}]
        again = client.get("/api/v2/runs/conv2d/one/lineage?stamp=" + body["stamp"])
        assert again.status_code == 200 and again.json() == {
            "run_id": "conv2d/one", "same": True, "stamp": body["stamp"],
        }
        missing = client.get("/api/v2/runs/conv2d/missing/lineage")
        assert missing.status_code == 404


def test_the_lineage_panel_budgets_root_names_too(tmp_path, monkeypatch):
    """@brief 评审回归：60 个长 ``init*`` 根时，``roots`` 也必须受 8KB 预算与截断标志约束。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _index(tmp_path, monkeypatch)
    _run(runs, "conv2d/roots")
    sequence = [(f"init{'1' * 180}{number}", True) for number in range(60)]
    _candidates(index, "conv2d/roots", sequence)

    panel = lineage.lineage_panel("conv2d/roots")
    document = panel["lineage"]
    size = len(json.dumps(panel, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert size <= lineage.LINEAGE_BYTES, f"panel wrapper is {size}B"
    assert document["roots_total"] == 60
    assert document["truncated"]["roots"] is True
    assert len(document["roots"]) < 60
    assert document["truncated"]["nodes"] is True
