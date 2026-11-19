"""Model health is a read-only descriptive view over existing run/session/feedback data.

No GPU, no model call, no trust mutation. Missing outcome/premise/trace data stays
``None``/``unknown`` and is never rendered as zero.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from observe import health
from observe.index import get_index
from observe.trust import metrics

NOTICE = "历史 run 的描述统计；P1 尚未 active；不参与授权"


def _write_run(runs: Path, run_id: str, events: list[dict], extra: dict[str, str] | None = None) -> Path:
    """@brief 写一份最小 run 目录：state、transcript，以及可选的门迹文件。"""
    directory = runs / run_id
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"task": "relu", "phase": "done"}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({"task": "relu"}), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({"protocol": "v2"}), encoding="utf-8")
    (directory / "transcript.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8"
    )
    for name, text in (extra or {}).items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def _agent(label: str, provider: str, model: str, outcome: str | None, cost: bool = True) -> dict:
    """@brief 一条 ``event=agent`` transcript 行；``cost=False`` 模拟没有成本记录的旧会话。"""
    row = {"event": "agent", "label": label, "committee": "first", "provider": provider, "model": model,
           "tokens": {"output": 2}}
    if cost:
        row["cost_usd"] = 0.1
    if outcome is not None:
        row["outcome"] = outcome
    return row


def _seed(tmp_path: Path, monkeypatch) -> Path:
    """@brief 建两个模型：一个全部交核，一个含未交核/其他失败/无 outcome。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _write_run(runs, "hmz", [
        _agent("a1", "p", "m-submit", "evaluated"),
        _agent("a2", "p", "m-submit", "unchanged"),
        _agent("a3", "p", "m-submit", "evaluated"),
        _agent("b1", "q", "m-miss", "agent_no_submission", cost=False),
        _agent("b2", "q", "m-miss", "agent_failed", cost=False),
        _agent("b3", "q", "m-miss", None, cost=False),
    ])
    from observe import query as scan

    scan.refresh()
    return runs


def _models() -> dict:
    """@brief 按 run 读一次模型行；全局读不主动追平历史 run 的会话投影。"""
    from observe import query as scan

    return {row["model"]: row for row in scan.model_stats(run_id="hmz")["models"]}


def test_submitted_and_no_submission_sessions_stay_distinct(tmp_path, monkeypatch):
    """全部交核模型为 0 未交核；含 no-submission 的模型单列其他失败，无 outcome 记 unknown。"""
    _seed(tmp_path, monkeypatch)
    rows = _models()
    submitted = rows["m-submit"]["health"]
    assert (submitted["sessions"], submitted["known_outcomes"], submitted["evaluated"], submitted["unchanged"]) == (3, 3, 2, 1)
    assert submitted["no_submission"] == 0 and submitted["no_submission_rate"] == 0.0
    assert submitted["unknown_outcomes"] == 0
    assert submitted["other_failures"] == {"agent_failed": 0, "provider_failed": 0,
                                           "delivery_rejected": 0, "cost_incomplete": 0}
    missed = rows["m-miss"]["health"]
    assert (missed["known_outcomes"], missed["no_submission"], missed["no_submission_rate"]) == (2, 1, 0.5)
    assert missed["unknown_outcomes"] == 1  # 无 outcome 那一行不计入未交核，也不计入成功率。
    assert missed["other_failures"]["agent_failed"] == 1
    assert missed["evaluated"] == 0
    # 没有候选投影时 ok 是 unknown/null，只给覆盖信息，不把无关联写成 0 个 ok。
    for row in rows.values():
        health_row = row["health"]
        assert health_row["ok_sessions"] is None and health_row["ok_rate"] is None
        assert health_row["ok_unlinked_sessions"] == health_row["sessions"]
        assert health_row["ok_coverage"] == 0.0
    # 没有成本记录的模型沿用 model_stats 的既有口径：成本覆盖 0，派生成本效率为 null。
    assert rows["m-miss"]["sessions_with_cost"] == 0
    assert rows["m-miss"]["cost_per_session_usd"] is None


def _candidates(run_id: str, rows: list[tuple[int, str, str]]) -> None:
    """@brief 直接写候选投影：标签与 sessions.candidate_label 对齐。"""
    with get_index().conn as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO candidates (run_id, idx, label, status) VALUES (?,?,?,?)",
            [(run_id, idx, label, status) for idx, label, status in rows],
        )


def test_candidate_ok_is_counted_separately_from_outcomes(tmp_path, monkeypatch):
    """ok / 非 ok / 无关联三类分开计；候选 status 不反向推断未交核或 evaluated。"""
    _seed(tmp_path, monkeypatch)
    _candidates("hmz", [(0, "a1", "ok"), (1, "a2", "incorrect"), (2, "b1", "ok")])
    rows = _models()
    submitted = rows["m-submit"]["health"]
    assert (submitted["ok_sessions"], submitted["non_ok_sessions"], submitted["ok_known_sessions"],
            submitted["ok_unlinked_sessions"]) == (1, 1, 2, 1)
    assert submitted["ok_rate"] == pytest.approx(0.5)
    assert submitted["ok_coverage"] == pytest.approx(2 / 3)
    # b1 的 outcome 仍是 agent_no_submission，尽管它关联到一个 ok 候选。
    missed = rows["m-miss"]["health"]
    assert (missed["no_submission"], missed["unknown_outcomes"]) == (1, 1)
    assert (missed["ok_sessions"], missed["non_ok_sessions"], missed["ok_known_sessions"],
            missed["ok_unlinked_sessions"]) == (1, 0, 1, 2)
    assert missed["ok_rate"] == pytest.approx(1.0)
    assert missed["ok_coverage"] == pytest.approx(1 / 3)
    # evaluated 仍只来自显式 outcome，不受候选状态影响。
    assert (submitted["evaluated"], submitted["unchanged"]) == (2, 1)


def test_missing_premise_and_feedback_are_null_not_zero(tmp_path, monkeypatch):
    """无反馈/无前提记录时给 None；never-auto 无门迹时 unavailable，计数为 null。"""
    _seed(tmp_path, monkeypatch)
    document = _models()
    rows = document
    for row in rows.values():
        premise = row["health"]["premise"]
        assert premise == {"verified": None, "unverified": None, "downgraded": None,
                           "total": None, "downgrade_rate": None}
        assert row["health"]["feedback"] is None
        assert row["health"]["never_auto"]["coverage"] == "unavailable"
        assert row["health"]["never_auto"]["traces"] is None
    from observe import query as scan

    document = scan.model_stats()
    assert document["health_notice"] == NOTICE
    assert document["never_auto"]["coverage"] == "unavailable"
    assert document["never_auto"]["traces"] is None
    assert document["never_auto"]["ratio"] is None
    assert document["never_auto"]["runs_with_traces"] is None


def _claims(tmp_path: Path) -> None:
    """@brief 直接写反馈投影行：前提三态与 adopted/rejected/pending 各一条，另加一条无前提行。"""
    index = get_index()
    with index.conn as conn:
        for seq in range(1, 4):
            conn.execute(
                "INSERT OR REPLACE INTO feedback_items (run_id, seq, task, provider, model, placeholder_items)"
                " VALUES (?,?,?,?,?,0)",
                ("hmz", seq, "relu", "p", "m-submit"),
            )
        for digest, premise, status, verification, audience in (
            ("a", "verified", "adopted", "confirmed", "peer"),
            ("b", "unverified", "rejected", "refuted", "peer"),
            ("c", "downgraded", "pending", "unchecked", "peer"),
            ("d", "", "info", "unchecked", "peer"),
        ):
            for seq in (1, 2, 3):
                conn.execute(
                    "INSERT OR REPLACE INTO feedback_claims"
                    " (run_id, seq, digest, premise, status, verification, audience)"
                    " VALUES (?,?,?,?,?,?,?)",
                    ("hmz", seq, digest, premise, status, verification, audience),
                )


def test_premise_and_human_feedback_reuse_trust_definitions(tmp_path, monkeypatch):
    """降级比例只在有前提的记录中算；adopted/refuted 与 trust.metrics 同口径。"""
    _seed(tmp_path, monkeypatch)
    _claims(tmp_path)
    row = _models()["m-submit"]["health"]
    premise = row["premise"]
    assert (premise["verified"], premise["unverified"], premise["downgraded"], premise["total"]) == (3, 3, 3, 9)
    assert premise["downgrade_rate"] == pytest.approx(1 / 3)
    feedback = row["feedback"]
    assert (feedback["adopted_done"], feedback["rejected_refuted"], feedback["pending"], feedback["reviewed"]) == (3, 3, 3, 6)
    assert feedback["adopted_rate"] == pytest.approx(0.5)
    assert feedback["wilson_lower"] is not None and 0 < feedback["wilson_lower"] < 0.5
    # 同一个连接上的既有 trust.metrics 给出同一组计数；健康度没有第二套定义。
    with get_index().conn as conn:
        trust_rows = metrics(conn)
    assert sum(int(item["reasonable"] or 0) for item in trust_rows) == feedback["adopted_done"]
    assert sum(int(item["refuted"] or 0) for item in trust_rows) == feedback["rejected_refuted"]
    assert sum(int(item["pending"] or 0) for item in trust_rows) == feedback["pending"]


def test_trust_metrics_multiple_groups_and_run_filter(tmp_path):
    """多分组不再踩占位符变量遮蔽；run_id 过滤只缩范围，不改口径。"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, target_id TEXT);
        CREATE TABLE sessions (run_id TEXT, seq INTEGER, provider TEXT, model TEXT);
        CREATE TABLE feedback_items (run_id TEXT, seq INTEGER, task TEXT, placeholder_items INTEGER);
        CREATE TABLE feedback_claims (run_id TEXT, seq INTEGER, digest TEXT, premise TEXT, status TEXT, verification TEXT, audience TEXT);
        """
    )
    conn.executemany("INSERT INTO runs VALUES (?, 'apple-m4-max')", [("r1",), ("r2",)])
    conn.executemany("INSERT INTO sessions VALUES (?,1,?,?)", [("r1", "p", "m"), ("r2", "q", "n")])
    conn.executemany("INSERT INTO feedback_items VALUES (?,1,?,?)", [("r1", "relu", 2), ("r2", "conv2d", 0)])
    conn.executemany(
        "INSERT INTO feedback_claims VALUES (?,1,?,?,?,?,'peer')",
        [("r1", "a", "verified", "done", "confirmed"), ("r2", "b", "verified", "rejected", "refuted")],
    )
    rows = metrics(conn)
    # 两个不同分组是关键：旧实现在第二组才把 placeholders 字典替换成 int 并抛 AttributeError。
    assert len(rows) == 2
    assert {(row["task"], row["provider"], row["model"]) for row in rows} == {
        ("relu", "p", "m"), ("conv2d", "q", "n"),
    }
    by_key = {(row["provider"], row["model"]): row for row in rows}
    assert by_key[("p", "m")]["placeholder_items"] == 2
    assert by_key[("p", "m")]["placeholder_rate"] == pytest.approx(0.4)
    assert by_key[("q", "n")]["placeholder_items"] == 0
    assert by_key[("q", "n")]["placeholder_rate"] == 0.0
    filtered = metrics(conn, run_id="r1")
    assert [(row["provider"], row["model"], row["reasonable"], row["refuted"]) for row in filtered] == [("p", "m", 1, 0)]
    assert [(row["reasonable"], row["refuted"]) for row in metrics(conn, run_id="r2")] == [(0, 1)]
def _route_m(provider: str, model: str, status: str) -> dict:
    return {
        "event": "route_m", "route": "M",
        "author": {"provider": provider, "model": model},
        "gates": [
            {"name": "route_proposal", "status": "pass", "passed": True},
            {"name": "never_auto", "status": status, "passed": status == "pass"},
        ],
    }


def _settle(route: str, status: str) -> dict:
    return {
        "event": "settle_gate_trace", "route": route,
        "gates": [{"name": "never_auto", "status": status, "passed": status == "pass"}],
    }


def test_never_auto_counts_only_real_route_m_traces(tmp_path, monkeypatch):
    """有门迹时只报计数与 partial 覆盖；逐模型归属只认绑定作者的路线 M 行。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _write_run(runs, "hmz", [_agent("a1", "p", "m-submit", "evaluated")], extra={
        health.ROUTE_M_TRACE_FILE: "\n".join(json.dumps(row) for row in (
            _route_m("p", "m-submit", "pass"),
            _route_m("p", "m-submit", "fail"),
        )) + "\n",
        health.SETTLE_TRACE_FILE: "\n".join(json.dumps(row) for row in (
            _settle("M", "not_run"),
            _settle("P", "fail"),
        )) + "\n",
    })
    health.clear_cache()
    from observe import query as scan

    scan.refresh()
    document = scan.model_stats(run_id="hmz")
    never = document["never_auto"]
    assert never["coverage"] == "partial" and never["ratio"] is None
    assert (never["traces"], never["gate_observed"], never["gate_passed"], never["gate_failed"], never["gate_not_run"]) == (3, 2, 1, 1, 1)
    assert (never["attributed_traces"], never["unattributed_traces"]) == (2, 1)
    assert never["runs_with_traces"] == 1
    row = document["models"][0]["health"]["never_auto"]
    assert (row["coverage"], row["traces"], row["gate_passed"], row["gate_failed"]) == ("partial", 2, 1, 1)


def test_health_read_is_read_only_and_visible_in_api_json(tmp_path, monkeypatch):
    """调用前后 run 文件字节与 trust 计数不变；API 顶层带说明和覆盖率字段。"""
    runs = _seed(tmp_path, monkeypatch)
    directory = runs / "hmz"
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    with get_index().conn as conn:
        trust_before = metrics(conn)
    from observe import app
    from observe import query as scan

    request = SimpleNamespace(query_params={"run": "hmz"})
    body = json.loads(__import__("asyncio").run(app.api_v2_model_stats(request)).body)
    assert body["health_notice"] == NOTICE
    assert body["never_auto"]["coverage"] == "unavailable"
    assert {"evaluated", "ok_sessions", "ok_coverage", "no_submission", "unknown_outcomes", "never_auto"} <= set(body["models"][0]["health"])
    assert body["models"][0]["health"]["ok_sessions"] is None  # 该合成 run 没有候选投影。
    after = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    assert after == before
    with get_index().conn as conn:
        assert metrics(conn) == trust_before
    assert scan.model_stats(run_id="hmz")["models"][0]["health"]["never_auto"]["coverage"] == "unavailable"


def test_real_index_compatibility_is_read_only(tmp_path):
    """真实索引可稳定读出；不钉条目数，只校验内部一致性与只读。"""
    database = Path(__file__).resolve().parents[2] / "data/observe.sqlite"
    if not database.exists():
        pytest.skip("the working tree has no real observe index")
    before = (database.stat().st_mtime_ns, database.stat().st_size)
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    collected = health.collect(conn)
    conn.close()
    assert collected["sessions"], "the real index should already expose sessions"
    for row in collected["sessions"].values():
        assert row["sessions"] >= row["known_outcomes"] >= 0
        assert row["unknown_outcomes"] == row["sessions"] - row["known_outcomes"]
        assert row["no_submission_rate"] is None or 0 <= row["no_submission_rate"] <= 1
        assert row["no_submission"] <= row["known_outcomes"]
        assert row["ok_unlinked_sessions"] == row["sessions"] - row["ok_known_sessions"]
        assert row["ok_known_sessions"] <= row["sessions"]
        if row["ok_known_sessions"]:
            assert 0 <= row["ok_sessions"] <= row["ok_known_sessions"]
            assert row["non_ok_sessions"] == row["ok_known_sessions"] - row["ok_sessions"]
            assert row["ok_coverage"] == pytest.approx(row["ok_known_sessions"] / row["sessions"])
        else:
            assert row["ok_sessions"] is None and row["non_ok_sessions"] is None and row["ok_rate"] is None
    for row in collected["premise"].values():
        assert row["total"] == row["verified"] + row["unverified"] + row["downgraded"]
        assert row["downgrade_rate"] is None or 0 <= row["downgrade_rate"] <= 1
    for row in collected["feedback"].values():
        assert row["reviewed"] == row["adopted_done"] + row["rejected_refuted"]
        assert row["adopted_rate"] is None or 0 <= row["adopted_rate"] <= 1
    gate, per_model = health.never_auto()
    assert gate["ratio"] is None and gate["coverage"] in {"unavailable", "partial"}
    if gate["traces"] is None:
        assert gate["coverage"] == "unavailable" and gate["gate_observed"] is None
        assert per_model == {}
    else:
        assert gate["coverage"] == "partial" and gate["traces"] > 0
    assert (database.stat().st_mtime_ns, database.stat().st_size) == before


def test_model_cost_table_renders_health_coverage_and_notice(tmp_path):
    """静态页面在 Node 下渲染出说明与 unavailable，不把 null 印成 0/NaN。"""
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    page = (Path(__file__).resolve().parents[2] / "observe/static/index.html").read_text(encoding="utf-8")

    def extract(name: str) -> str:
        start = page.index("function " + name + "(")
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

    functions = "\n".join(extract(name) for name in ("esc", "num", "money2", "number2", "modelCostTable"))
    script = functions + r'''
const html = modelCostTable([{provider:"p",model:"m",committee:"first",sessions:2,sessions_with_cost:1,
  cost_usd:0.5,cache_read_tokens:1,cache_write_tokens:1,output_tokens:1,cache_read_per_output:1,
  usd_per_1m_output:1,cost_per_session_usd:0.5,
  health:{scope:"provider/model",sessions:2,known_outcomes:2,evaluated:1,unchanged:0,evaluated_rate:0.5,
    ok_sessions:1,non_ok_sessions:1,ok_known_sessions:2,ok_unlinked_sessions:0,ok_rate:0.5,ok_coverage:1,
    no_submission:1,no_submission_rate:0.5,unknown_outcomes:0,
    other_failures:{agent_failed:0,provider_failed:0,delivery_rejected:0,cost_incomplete:0},
    premise:{verified:null,unverified:null,downgraded:null,total:null,downgrade_rate:null},
    feedback:null,never_auto:{coverage:"unavailable",traces:null,gate_observed:null,gate_passed:null,gate_failed:null,gate_not_run:null}}}],
  "历史 run 的描述统计；P1 尚未 active；不参与授权");
if (!html.includes("不参与授权")) throw new Error("notice missing");
if (!html.includes("unavailable")) throw new Error("never-auto coverage missing");
if (!html.includes("未交核")) throw new Error("no-submission column missing");
if (!html.includes("候选 ok")) throw new Error("ok column missing");
if (!html.includes("1/2")) throw new Error("ok count missing");
if (html.includes("NaN") || html.includes("Infinity")) throw new Error("non-finite value rendered");
const unknown = modelCostTable([{provider:"p",model:"m",committee:"first",sessions:2,sessions_with_cost:0,
  cost_usd:0,cache_read_tokens:0,cache_write_tokens:0,output_tokens:0,cache_read_per_output:null,
  usd_per_1m_output:null,cost_per_session_usd:null,
  health:{sessions:2,known_outcomes:2,evaluated:1,unchanged:0,evaluated_rate:0.5,
    ok_sessions:null,non_ok_sessions:null,ok_known_sessions:0,ok_unlinked_sessions:2,ok_rate:null,ok_coverage:0,
    no_submission:0,no_submission_rate:0,unknown_outcomes:0,
    other_failures:{},premise:{},feedback:null,never_auto:{coverage:"unavailable",traces:null}}}],
  "历史 run 的描述统计；P1 尚未 active；不参与授权");
if (!unknown.includes("unknown")) throw new Error("unlinked ok must be unknown");
if (!unknown.includes("无候选关联")) throw new Error("unlinked coverage note missing");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"
