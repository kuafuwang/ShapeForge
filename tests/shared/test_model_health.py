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

NOTICE = "历史 run 的描述统计；P1 的激活状态在 overlay 视图里单列；不参与授权"


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


def test_unmeasurable_candidate_states_are_unknown_not_non_ok(tmp_path, monkeypatch):
    """old_format 等读取状态与 unchanged/duplicate 未实测状态都不能进 ok 比例分母。"""
    _seed(tmp_path, monkeypatch)
    _candidates("hmz", [
        (0, "a1", "old_format"),
        (1, "a2", "incorrect"),
        (2, "a3", "unchanged"),
        (3, "b1", "unreadable"),
        (4, "b2", "duplicate"),
        (5, "b3", "missing"),
    ])
    rows = _models()
    submitted = rows["m-submit"]["health"]
    # a1/a3 不可判，只有 a2 是明确非 ok；不能说 0/3 或 0% ok。
    assert (submitted["ok_sessions"], submitted["non_ok_sessions"],
            submitted["ok_known_sessions"], submitted["ok_unknown_sessions"]) == (0, 1, 1, 2)
    assert submitted["ok_unlinked_sessions"] == 0
    assert submitted["ok_rate"] == pytest.approx(0.0)
    assert submitted["ok_coverage"] == pytest.approx(1 / 3)
    missed = rows["m-miss"]["health"]
    # 三个关联候选全都不可判：ok/非 ok 与比例都是 unknown，不是 0/3。
    assert missed["ok_known_sessions"] == 0 and missed["ok_unknown_sessions"] == 3
    assert missed["ok_sessions"] is None and missed["non_ok_sessions"] is None
    assert missed["ok_rate"] is None
    assert missed["ok_unlinked_sessions"] == 0 and missed["ok_coverage"] == 0.0


def test_only_blank_premises_are_unknown_not_zero(tmp_path, monkeypatch):
    """有反馈但全部 premise='' 时，三个计数和 total 都为 None。"""
    _seed(tmp_path, monkeypatch)
    with get_index().conn as conn:
        conn.execute(
            "INSERT OR REPLACE INTO feedback_items"
            " (run_id, seq, task, provider, model, placeholder_items) VALUES (?,?,?,?,?,0)",
            ("hmz", 9, "relu", "p", "m-submit"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO feedback_claims"
            " (run_id, seq, digest, premise, status, verification, audience) VALUES (?,?,?,?,?,?,?)",
            ("hmz", 9, "blank", "", "info", "unchecked", "peer"),
        )
    premise = _models()["m-submit"]["health"]["premise"]
    assert premise == {"verified": None, "unverified": None, "downgraded": None,
                       "total": None, "downgrade_rate": None}


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
        assert row["ok_known_sessions"] <= row["sessions"]
        assert row["ok_unknown_sessions"] >= 0
        assert row["ok_known_sessions"] + row["ok_unknown_sessions"] + row["ok_unlinked_sessions"] == row["sessions"]
        if row["ok_known_sessions"]:
            assert 0 <= row["ok_sessions"] <= row["ok_known_sessions"]
            assert row["non_ok_sessions"] == row["ok_known_sessions"] - row["ok_sessions"]
            assert row["ok_rate"] == pytest.approx(row["ok_sessions"] / row["ok_known_sessions"])
            assert row["ok_coverage"] == pytest.approx(row["ok_known_sessions"] / row["sessions"])
        else:
            assert row["ok_sessions"] is None and row["non_ok_sessions"] is None and row["ok_rate"] is None
    for row in collected["premise"].values():
        if row["total"] is None:
            assert row["verified"] is None and row["unverified"] is None and row["downgraded"] is None
            assert row["downgrade_rate"] is None
        else:
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

    functions = "\n".join(extract(name) for name in ("esc", "num", "money2", "money4", "number2", "overlayActivityTable", "modelCostTable"))
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
  "历史 run 的描述统计；P1 的激活状态在 overlay 视图里单列；不参与授权");
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
  "历史 run 的描述统计；P1 的激活状态在 overlay 视图里单列；不参与授权");
if (!unknown.includes("unknown")) throw new Error("unlinked ok must be unknown");
if (!unknown.includes("无候选关联")) throw new Error("unlinked coverage note missing");
const unreadable = modelCostTable([{provider:"p",model:"m",committee:"first",sessions:36,sessions_with_cost:null,
  cost_usd:0,cache_read_tokens:0,cache_write_tokens:0,output_tokens:0,
  health:{sessions:36,known_outcomes:36,
    ok_sessions:null,non_ok_sessions:null,ok_known_sessions:0,ok_unknown_sessions:36,ok_unlinked_sessions:0,ok_rate:null,ok_coverage:0,
    no_submission:0,no_submission_rate:0,unknown_outcomes:0,
    other_failures:{},premise:{},feedback:null,never_auto:{coverage:"unavailable",traces:null}}}],
  "历史 run 的描述统计；P1 的激活状态在 overlay 视图里单列；不参与授权");
if (!unreadable.includes("候选状态不可判")) throw new Error("old_format must stay visibly undecidable");
if (unreadable.includes("0/36")) throw new Error("old_format must not render as 0/36 non-ok");
const overlay = {notice:"只读：激活不是效果",effect_note:"active 状态不是效果。",
  note:"coverage=partial 表示只统计写过文件的 run",coverage:"partial",runs_scanned:5,runs_with_events:2,
  routes:{premise:{coverage:"partial",proposals:1,proposals_only:0,entries:1,with_shadow:1,with_active:1,
      with_delivery:1,disabled:1,sessions_after_active:1},
    M:{coverage:"unavailable",proposals:null,proposals_only:null,entries:null,with_shadow:null,
      with_active:null,with_delivery:null,disabled:null,sessions_after_active:null},
    tip:{coverage:"partial",proposals:3,proposals_only:3,entries:0,with_shadow:0,with_active:0,
      with_delivery:0,disabled:0,sessions_after_active:0}},
  entries:[{proposal_key:"f142f237",route:"premise",surface:"premise",target_id:"apple-m4-max",
    task:"conv2d",run_id:"apple-m4-max/p1",source_run:"apple-m4-max/p1",source_seq:3,
    evidence_level:"host_process",trusted_record:true,probe_time_verified:true,decisions:["apply","shadow"],
    active_from:3,stages:{proposal:1,shadow:1,active:1,delivered:1,disabled:true,
      sessions_after_active:{sessions:2,outcomes:{evaluated:2},active_from:3}},
    delivery:[{receipt:"sessions/r1.p0/task_delivery.json",kind:"session",status:"delivered",
      delivered_bytes:807,omitted_bytes:0}],unavailable:[]}],effect_rate:null};
const smoke = {coverage:"available",note:"一对样本不是效果证据",effect_rate:null,records:[{
  run_id:"apple-m4-max/p1-premise",pairs:1,total_cost_usd:0.519934296,verdict:"inconclusive",
  publication_allowed:false,cells:[{arm:"control",confirmed_speedup:1.0405464310,confirmed_status:"ok",overlay_delivered:false},
    {arm:"treatment",confirmed_speedup:0.8345098497,confirmed_status:"ok",overlay_delivered:true}]}]};
const withOverlay = modelCostTable([{provider:"p",model:"m",committee:"first",sessions:1,sessions_with_cost:1,
  cost_usd:0.1,cache_read_tokens:0,cache_write_tokens:0,output_tokens:1,cache_read_per_output:null,
  usd_per_1m_output:null,cost_per_session_usd:0.1,health:{sessions:1,known_outcomes:1,premise:{},
  feedback:null,never_auto:{coverage:"unavailable",traces:null}}}], "历史 run 的描述统计；P1 的激活状态在 overlay 视图里单列；不参与授权", overlay, smoke);
if (!withOverlay.includes("active overlay 活动")) throw new Error("overlay block missing");
if (!withOverlay.includes("路线 P（premise）") || !withOverlay.includes("路线 M")) throw new Error("route split missing");
if (!withOverlay.includes("unavailable")) throw new Error("missing stage must render unavailable");
if (!withOverlay.includes("807")) throw new Error("delivered bytes missing");
if (!withOverlay.includes("inconclusive")) throw new Error("smoke verdict missing");
if (!withOverlay.includes("1.0405")) throw new Error("control confirmed missing");
if (withOverlay.includes("NaN") || withOverlay.includes("Infinity")) throw new Error("non-finite overlay value");
if (withOverlay.includes("有用率：")) throw new Error("must not synthesize a usefulness rate");
console.log("ok");
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# active overlay 专属只读指标（G13 第二步）
# --------------------------------------------------------------------------- #

def _jsonl(rows: list[dict]) -> str:
    """@brief 把若干字典写成 JSONL 文本，供 ``_write_run`` 的 extra 用。"""
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"


def _proposal(key: str, *, seq: int = 1, surface: str = "premise", **extra) -> dict:
    """@brief 一条 ``event=proposal`` 行；surface 按真实文件放在内层 ``proposal``。"""
    row = {
        "event": "proposal", "proposal_key": key, "seq": seq, "status": "valid",
        "proposal": {"surface": surface, "action": "x"},
        "target_id": "apple-m4-max", "task": "conv2d",
        "evidence_level": "host_process", "trusted_record": True, "probe_time_verified": True,
    }
    row.update(extra)
    return row


def _overlay_row(key: str, *, shadow: bool = True, active_from: int = 1, surface: str = "premise",
                 route: str | None = None, source_run: str | None = None, **extra) -> dict:
    """@brief 一条 ``event=overlay`` 行；``route='M'`` 时是路线 M 的影子行。

    ``source_run`` 省略时表示这条 overlay 属于本 run（``active_from`` 是本 run 的场次序号）；
    显式给出另一场 run 时，``active_from`` 属于那场 run，不能拿来过滤本地会话。
    """
    row = {
        "event": "overlay", "proposal_key": key, "shadow": shadow, "surface": surface,
        "target_id": "apple-m4-max", "task": "conv2d", "source_seq": 3,
        "evidence_level": "host_process", "active_from": active_from, "active_until": None,
    }
    if source_run:
        row["source_run"] = source_run
    if route:
        row["route"] = route
    row.update(extra)
    return row


def _decision(key: str, decision: str) -> dict:
    return {"event": "decision", "proposal_key": key, "decision": decision,
            "target_id": "apple-m4-max", "task": "conv2d", "source_seq": 3}


def _receipt(*, status: str = "delivered", delivered: int = 807, omitted: int = 0) -> dict:
    """@brief 一份 TASK 送达回执里只有 overlay 层与结论层。"""
    return {
        "schema": "taic.task_delivery.1", "event": "task_delivery",
        "situation": {
            "status": status, "delivery": status,
            "layers": [{
                "name": "overlay", "status": status,
                "delivered_bytes": delivered, "omitted_bytes": omitted,
                "content_sha256": "ab" * 32,
            }],
        },
    }


def _overlay_run(runs: Path, run_id: str, *, proposals=(), overlays=(), decisions=(),
                 control: dict | None = None, render_receipt: dict | None = None,
                 session_receipts: dict[str, dict] | None = None,
                 agents: list[dict] | None = None) -> Path:
    """@brief 写一场带 overlay 证据的 run；没有的环节就不写文件（保持 None 语义）。"""
    extra: dict[str, str] = {}
    if proposals:
        extra[health.PROPOSAL_FILE] = _jsonl(list(proposals))
    if overlays:
        extra[health.OVERLAY_FILE] = _jsonl(list(overlays))
    if decisions:
        extra[health.DECISION_FILE] = _jsonl(list(decisions))
    if control is not None:
        extra[health.CONTROL_FILE] = json.dumps(control)
    directory = _write_run(runs, run_id, agents or [_agent("a1", "p", "m", "evaluated")], extra=extra)
    if render_receipt is not None:
        (directory / "render-check").mkdir()
        (directory / "render-check" / "task_delivery.json").write_text(
            json.dumps(render_receipt), encoding="utf-8")
    for label, receipt in (session_receipts or {}).items():
        (directory / "sessions" / label).mkdir(parents=True)
        (directory / "sessions" / label / "task_delivery.json").write_text(
            json.dumps(receipt), encoding="utf-8")
    return directory


def _refresh():
    from observe import query as scan

    health.clear_cache()
    scan.refresh()


def _sessions(run_id: str) -> list[dict]:
    with get_index().conn as conn:
        return [dict(row) for row in conn.execute(
            "SELECT seq, outcome FROM sessions WHERE run_id = ? ORDER BY seq", (run_id,))]


def test_overlay_activity_without_events_is_unavailable(tmp_path, monkeypatch):
    """没有 overlay/proposal/decision 文件的历史 run：整份视图 unavailable，不写成 0。"""
    _seed(tmp_path, monkeypatch)
    document = health.overlay_activity(run_id="hmz")
    assert document["coverage"] == "unavailable"
    assert document["entries"] == []
    assert document["runs_with_events"] is None
    assert document["effect_rate"] is None and document["effect_note"]
    for route in ("premise", "M", "tip"):
        summary = document["routes"][route]
        assert summary["coverage"] == "unavailable"
        assert summary["proposals"] is None and summary["entries"] is None
        assert summary["with_shadow"] is None and summary["with_active"] is None
        assert summary["with_delivery"] is None and summary["disabled"] is None
    assert health.smoke_summary(run_id="hmz")["coverage"] == "unavailable"
    assert health.smoke_summary(run_id="hmz")["records"] == []


def test_shadow_only_overlay_stays_out_of_active_and_delivery(tmp_path, monkeypatch):
    """只有 shadow 行：active/送达/停用都是 None；有 active 行的条目才谈激活后会话。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _overlay_run(runs, "hmz", proposals=[_proposal("k-shadow")],
                 overlays=[_overlay_row("k-shadow", shadow=True)],
                 decisions=[_decision("k-shadow", "shadow")])
    _refresh()
    document = health.overlay_activity(run_id="hmz")
    entry = document["entries"][0]
    assert entry["route"] == "premise" and entry["proposal_key"] == "k-shadow"
    assert entry["stages"]["proposal"] == 1 and entry["stages"]["shadow"] == 1
    assert entry["stages"]["active"] is None
    assert entry["stages"]["delivered"] is None
    assert entry["stages"]["disabled"] is None
    assert entry["stages"]["sessions_after_active"] is None
    assert entry["decisions"] == ["shadow"]
    assert entry["delivery"] is None
    assert {"active", "delivered", "disabled", "sessions_after_active"} <= set(entry["unavailable"])
    premise = document["routes"]["premise"]
    assert (premise["proposals"], premise["with_shadow"], premise["with_active"]) == (1, 1, 0)
    assert premise["with_delivery"] == 0 and premise["disabled"] == 0
    assert premise["sessions_after_active"] == 0


def test_active_without_receipt_keeps_delivery_unknown(tmp_path, monkeypatch):
    """active 但没有回执：active=1、delivered 保持 None，routes 的 with_delivery 记 0。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _overlay_run(runs, "hmz", proposals=[_proposal("k-active")],
                 overlays=[_overlay_row("k-active", shadow=True),
                           _overlay_row("k-active", shadow=False, active_from=1)],
                 decisions=[_decision("k-active", "shadow"), _decision("k-active", "apply")])
    _refresh()
    document = health.overlay_activity(run_id="hmz")
    entry = document["entries"][0]
    assert (entry["stages"]["shadow"], entry["stages"]["active"]) == (1, 1)
    assert entry["stages"]["delivered"] is None and entry["delivery"] is None
    assert "delivered" in entry["unavailable"]
    assert entry["decisions"] == ["apply", "shadow"]
    assert document["routes"]["premise"]["with_delivery"] == 0

    receipt = _receipt(status="empty", delivered=0)
    (runs / "hmz" / "sessions" / "r1.p0").mkdir(parents=True)
    (runs / "hmz" / "sessions" / "r1.p0" / "task_delivery.json").write_text(json.dumps(receipt))
    health.clear_cache()
    document = health.overlay_activity(run_id="hmz")
    entry = document["entries"][0]
    # 回执存在但 overlay 层没送达：仍然不是 delivered，不能把 0 字节当成功。
    assert entry["stages"]["delivered"] is None
    assert entry["delivery"] == [{
        "receipt": "sessions/r1.p0/task_delivery.json", "kind": "session", "label": "r1.p0",
        "status": "empty", "delivered_bytes": 0, "omitted_bytes": 0, "content_sha256": "ab" * 32,
    }]
    assert "delivered" in entry["unavailable"]


def test_active_delivered_disabled_and_sessions_after_active(tmp_path, monkeypatch):
    """active + 送达 + 停用 + 激活后会话：字节、disabled、后续结局都按盘上事实给出。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _overlay_run(
        runs, "hmz",
        proposals=[_proposal("k-live", seq=2)],
        overlays=[_overlay_row("k-live", shadow=True, active_from=2),
                  _overlay_row("k-live", shadow=False, active_from=2)],
        decisions=[_decision("k-live", "apply")],
        control={"disabled": True, "by": "human", "reason": "smoke complete"},
        render_receipt=_receipt(delivered=807),
        session_receipts={"r1.p0": _receipt(delivered=734, omitted=73)},
        agents=[_agent("r1.p0", "p", "m", "evaluated"),
                _agent("r2.p0", "p", "m", "evaluated"),
                _agent("r3.p0", "p", "m", "agent_no_submission")],
    )
    _refresh()
    before = {path.name: path.read_bytes() for path in (runs / "hmz").iterdir() if path.is_file()}
    document = health.overlay_activity(run_id="hmz")
    entry = document["entries"][0]
    assert (entry["stages"]["proposal"], entry["stages"]["shadow"], entry["stages"]["active"]) == (1, 1, 1)
    assert entry["stages"]["delivered"] == 2
    assert entry["stages"]["disabled"] is True
    # 这条条目由 proposal 行主导：来源 run 是记录它的这场，source_seq 取该 proposal 的 seq。
    assert entry["active_from"] == 2 and entry["source_run"] == "hmz" and entry["source_seq"] == 2
    assert [(item["receipt"], item["kind"], item["delivered_bytes"], item["omitted_bytes"])
            for item in entry["delivery"]] == [
        ("render-check/task_delivery.json", "render_check", 807, 0),
        ("sessions/r1.p0/task_delivery.json", "session", 734, 73),
    ]
    # 激活后的会话结局来自索引：seq >= active_from 的所有会话，不并入候选状态。
    indexed = _sessions("hmz")
    expected = [row for row in indexed if int(row["seq"]) >= 2]
    assert len(expected) == 2  # seq 1 在激活前，seq 2/3 在激活后；边界不靠会话标签猜。
    assert entry["stages"]["sessions_after_active"]["sessions"] == len(expected)
    assert entry["stages"]["sessions_after_active"]["outcomes"] == {"evaluated": 1, "agent_no_submission": 1}
    premise = document["routes"]["premise"]
    assert (premise["with_active"], premise["with_delivery"], premise["disabled"],
            premise["sessions_after_active"]) == (1, 1, 1, 1)
    after = {path.name: path.read_bytes() for path in (runs / "hmz").iterdir() if path.is_file()}
    assert after == before


def test_route_m_and_premise_overlays_are_reported_separately(tmp_path, monkeypatch):
    """路线 M 影子行与 premise 路线 P 分开汇总；M 的 shadow 不因 active 行而升级。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _overlay_run(
        runs, "hmz",
        proposals=[_proposal("k-premise", seq=1), _proposal("k-tip", seq=2, surface="tip")],
        overlays=[_overlay_row("k-premise", shadow=False, active_from=1, surface="premise"),
                  _overlay_row("k-m", shadow=True, surface="tip", route="M", active_from=2)],
    )
    _refresh()
    document = health.overlay_activity(run_id="hmz")
    by_key = {entry["proposal_key"]: entry for entry in document["entries"]}
    assert by_key["k-premise"]["route"] == "premise"
    assert by_key["k-m"]["route"] == "M"
    assert by_key["k-m"]["stages"]["shadow"] == 1 and by_key["k-m"]["stages"]["active"] is None
    assert by_key["k-m"]["stages"]["sessions_after_active"] is None
    premise, route_m, tip = (document["routes"][key] for key in ("premise", "M", "tip"))
    assert (premise["entries"], premise["with_active"]) == (1, 1)
    assert (route_m["entries"], route_m["with_shadow"], route_m["with_active"]) == (1, 1, 0)
    # 只有 proposal、没有 overlay 行的 tip key 不进逐条条目，只计 proposals_only。
    assert (tip["proposals"], tip["proposals_only"], tip["entries"]) == (1, 1, 0)
    assert tip["with_shadow"] == 0 and tip["with_active"] == 0


def test_paired_smoke_summary_is_readable_and_not_a_usefulness_rate(tmp_path, monkeypatch):
    """一对 smoke：成本、两端 confirmed、verdict=inconclusive、publication_allowed=false。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    smoke = {
        "schema": "shapeforge.p1-overlay-paired-smoke.1", "proposal_key": "f142f237a50ab0ef",
        "task": "conv2d", "dtype": "fp16", "size": "small", "pairs": 1, "excluded_pairs": [],
        "total_cost_usd": 0.519934296, "budget_cap_usd": 3.0, "verdict": "inconclusive",
        "publication_allowed": False, "speedup_claim_allowed": False,
        "adopt_overlay_for_effect": False,
        "cells": {
            "control": {"model": "deepseek-flash", "run_seed": 7, "overlay_delivered": False,
                        "confirmed_speedup": 1.0405464310, "confirmed_status": "ok", "cost_usd": 0.28595838},
            "treatment": {"model": "deepseek-flash", "run_seed": 7, "overlay_delivered": True,
                          "confirmed_speedup": 0.8345098497, "confirmed_status": "ok", "cost_usd": 0.233975916},
        },
    }
    _write_run(runs, "hmz", [_agent("a1", "p", "m", "evaluated")],
               extra={health.SMOKE_FILE: json.dumps(smoke)})
    _refresh()
    document = health.smoke_summary(run_id="hmz")
    assert document["coverage"] == "available" and document["effect_rate"] is None
    record = document["records"][0]
    assert record["proposal_key"] == "f142f237a50ab0ef"
    assert (record["pairs"], record["total_cost_usd"], record["verdict"]) == (1, 0.519934296, "inconclusive")
    assert record["publication_allowed"] is False and record["adopt_overlay_for_effect"] is False
    arms = {cell["arm"]: cell for cell in record["cells"]}
    assert arms["control"]["confirmed_speedup"] == pytest.approx(1.0405464310)
    assert arms["treatment"]["confirmed_speedup"] == pytest.approx(0.8345098497)
    assert arms["treatment"]["overlay_delivered"] is True


def test_missing_smoke_fields_stay_unknown(tmp_path, monkeypatch):
    """回执缺字段、smoke 缺结论时留 None；不补 0，也不编 verdict。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _overlay_run(
        runs, "hmz",
        proposals=[_proposal("k-live")],
        overlays=[_overlay_row("k-live", shadow=False, active_from=1)],
        session_receipts={"r1.p0": {"schema": "taic.task_delivery.1", "event": "task_delivery"}},
    )
    _write_run(runs, "hmz2", [_agent("a1", "p", "m", "evaluated")],
               extra={health.SMOKE_FILE: json.dumps({"schema": "shapeforge.p1-overlay-paired-smoke.1"})})
    _refresh()
    entry = health.overlay_activity(run_id="hmz")["entries"][0]
    assert entry["stages"]["delivered"] is None and entry["delivery"] is None
    assert entry["evidence_level"] == "host_process"
    record = health.smoke_summary(run_id="hmz2")["records"][0]
    for name in ("pairs", "total_cost_usd", "verdict", "publication_allowed", "cells"):
        assert record[name] is None
    assert health.smoke_summary(run_id="hmz2")["coverage"] == "available"
    assert health.smoke_summary(run_id="missing")["coverage"] == "unavailable"


def test_overlay_activity_is_read_only_and_visible_in_api_json(tmp_path, monkeypatch):
    """API 顶层带 overlay/smoke；调用前后 run 文件与 trust 计数不变。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    directory = _overlay_run(
        runs, "hmz",
        proposals=[_proposal("k-live", seq=1)],
        overlays=[_overlay_row("k-live", shadow=False, active_from=1)],
        control={"disabled": True, "by": "human", "reason": "x"},
        render_receipt=_receipt(),
    )
    _refresh()
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    with get_index().conn as conn:
        trust_before = metrics(conn)
    from observe import app

    request = SimpleNamespace(query_params={"run": "hmz"})
    body = json.loads(__import__("asyncio").run(app.api_v2_model_stats(request)).body)
    assert body["overlay"]["coverage"] == "partial"
    assert body["overlay"]["entries"][0]["proposal_key"] == "k-live"
    assert body["overlay"]["effect_rate"] is None
    assert body["smoke"]["coverage"] == "unavailable"
    after = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    assert after == before
    with get_index().conn as conn:
        assert metrics(conn) == trust_before


def test_cross_run_active_from_uses_delivery_labels_not_local_seq(tmp_path, monkeypatch):
    """跨 run 复制来的 active_from 属于 source run：本地按送达回执 label 关联，不能报 0 场。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _overlay_run(
        runs, "hmz",
        proposals=[_proposal("k-live", seq=2)],
        overlays=[_overlay_row("k-live", shadow=False, active_from=3, source_run="source/run")],
        decisions=[_decision("k-live", "apply")],
        session_receipts={"r1.p0": _receipt(status="delivered", delivered=807)},
        agents=[_agent("r1.p0", "p", "m", "evaluated")],
    )
    _refresh()
    entry = health.overlay_activity(run_id="hmz")["entries"][0]
    assert entry["source_run"] == "hmz" and entry["active_source_run"] == "source/run"
    assert entry["active_boundary"] == "source_run"
    after = entry["stages"]["sessions_after_active"]
    assert after is not None
    assert (after["sessions"], after["labels"], after["boundary"]) == (1, ["r1.p0"], "delivery_receipt_labels")
    assert after["outcomes"] == {"evaluated": 1}
    # 同一个 run 去掉回执后无法把 source-run 阈值映射到本地会话：保持 None，不写成 0。
    for path in (runs / "hmz" / "sessions").rglob("task_delivery.json"):
        path.unlink()
    health.clear_cache()
    entry = health.overlay_activity(run_id="hmz")["entries"][0]
    assert entry["stages"]["sessions_after_active"] is None
    assert "sessions_after_active" in entry["unavailable"]


def test_control_and_receipt_unknown_values_are_not_fake_zero_or_false(tmp_path, monkeypatch):
    """control 缺字段/null 时 disabled 是 None；回执缺字节或非数字时字节是 None，且不抛异常。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _overlay_run(
        runs, "hmz",
        proposals=[_proposal("k-live")],
        overlays=[_overlay_row("k-live", shadow=False, active_from=1)],
        control={},
        render_receipt={"schema": "taic.task_delivery.1", "situation": {"layers": [
            {"name": "overlay", "status": "delivered", "content_sha256": "cd" * 32}]}},
        session_receipts={"r1.p0": {"schema": "taic.task_delivery.1", "situation": {"layers": [
            {"name": "overlay", "status": "delivered", "delivered_bytes": None,
             "omitted_bytes": "unknown"}]}}},
    )
    _refresh()
    entry = health.overlay_activity(run_id="hmz")["entries"][0]
    # 空的 control 不是“没禁用”，也不是“确认禁用”；缺字段同样保持 unknown。
    assert entry["stages"]["disabled"] is None
    assert "disabled" in entry["unavailable"]
    # status=delivered 是事实，但字节缺失/非数字时不能显示 0/0。
    assert entry["stages"]["delivered"] == 2
    assert [(item["status"], item["delivered_bytes"], item["omitted_bytes"])
            for item in entry["delivery"]] == [
        ("delivered", None, None), ("delivered", None, None),
    ]

    _overlay_run(
        runs, "hmz2",
        proposals=[_proposal("k-null")],
        overlays=[_overlay_row("k-null", shadow=False, active_from=1)],
        control={"disabled": None},
    )
    _refresh()
    assert health.overlay_activity(run_id="hmz2")["entries"][0]["stages"]["disabled"] is None


def test_one_receipt_serving_two_active_keys_is_reported_once(tmp_path, monkeypatch):
    """一条合并回执无法拆到两个 active key：逐 key delivered 为 None，run 级只列一次。"""
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    _overlay_run(
        runs, "hmz",
        proposals=[_proposal("k-a", seq=1), _proposal("k-b", seq=2)],
        overlays=[_overlay_row("k-a", shadow=False, active_from=1),
                  _overlay_row("k-b", shadow=False, active_from=1)],
        session_receipts={"r1.p0": _receipt(status="delivered", delivered=807)},
    )
    _refresh()
    document = health.overlay_activity(run_id="hmz")
    assert len(document["entries"]) == 2
    for entry in document["entries"]:
        assert entry["stages"]["delivered"] is None and entry["delivery"] is None
        assert entry["delivery_ambiguous"] is True
        assert "delivered" in entry["unavailable"]
    assert document["routes"]["premise"]["with_delivery"] == 0
    assert document["routes"]["premise"]["delivery_ambiguous"] == 2
    assert len(document["run_deliveries"]) == 1
    record = document["run_deliveries"][0]
    assert record["run_id"] == "hmz" and record["ambiguous"] is True
    assert record["active_keys"] == ["k-a", "k-b"]
    assert [(item["receipt"], item["delivered_bytes"]) for item in record["receipts"]] == [
        ("sessions/r1.p0/task_delivery.json", 807),
    ]


def test_real_overlay_evidence_is_readable_without_pinned_counts(tmp_path, monkeypatch):
    """真实 runs/ 的 overlay 证据可稳定读出；不钉条目数，只校验内部一致性与只读。"""
    runs = Path(__file__).resolve().parents[2] / "runs"
    if not runs.is_dir():
        pytest.skip("the working tree has no runs directory")
    monkeypatch.delenv("AI_LOWERLING_RUNS_DIR", raising=False)
    health.clear_cache()
    document = health.overlay_activity()
    assert document["effect_rate"] is None and document["effect_note"]
    assert document["coverage"] in {"unavailable", "partial"}
    assert document["runs_scanned"] > 0
    for entry in document["entries"]:
        assert entry["route"] in {"premise", "M", "tip"}
        stages = entry["stages"]
        assert stages["shadow"] is None or stages["shadow"] >= 1
        assert stages["active"] is None or stages["active"] >= 1
        assert stages["disabled"] in {True, False, None}
        if stages["delivered"] is not None:
            assert stages["delivered"] >= 1
            assert any(item["status"] == "delivered" and item["delivered_bytes"] > 0
                       for item in entry["delivery"] or [])
        if stages["sessions_after_active"] is not None:
            assert stages["sessions_after_active"]["sessions"] >= 0
            assert sum(stages["sessions_after_active"]["outcomes"].values()) == \
                stages["sessions_after_active"]["sessions"]
    for route, summary in document["routes"].items():
        grouped = [entry for entry in document["entries"] if entry["route"] == route]
        if summary["coverage"] == "unavailable":
            assert grouped == [] and summary["entries"] is None
            continue
        assert summary["proposals"] == summary["proposals_only"] + summary["entries"]
        assert summary["entries"] == len(grouped)
        assert summary["with_shadow"] == sum(1 for entry in grouped if entry["stages"]["shadow"])
        assert summary["with_active"] == sum(1 for entry in grouped if entry["stages"]["active"])
    treatment = [entry for entry in document["entries"]
                 if entry["run_id"].endswith("p1-overlay-smoke-treatment-20261120")]
    if treatment:
        # 已知事实：这条 treatment 的 overlay 来自 source run（active_from=3），本地唯一会话是
        # r1.p0（seq=1）且回执为 delivered。不钉总条目数，只钉这一条没被漏掉。
        entry = treatment[0]
        assert entry["active_boundary"] == "source_run"
        after = entry["stages"]["sessions_after_active"]
        assert after is not None and after["boundary"] == "delivery_receipt_labels"
        assert after["sessions"] >= 1 and "r1.p0" in after["labels"]
    smoke = health.smoke_summary()
    assert smoke["effect_rate"] is None
    assert smoke["coverage"] in {"unavailable", "available"}
    for record in smoke["records"]:
        assert record["pairs"] is None or record["pairs"] >= 1
        assert record["publication_allowed"] in {True, False, None}
