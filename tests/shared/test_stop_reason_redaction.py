"""A run's ``stop_reason`` is free text; it must not carry credential shapes into storage or a view.

The audit's counterexample is an ordinary run (not a failure-code edge case) whose
``result.json.stop_reason`` carried ``api_key=``, ``sk-``, the JSON ``"api_key"`` spelling and
``ghp_``: all four reached ``runs.stop_reason``, ``row_json``, ``runs_fts``, the ``control.reason``
text that embeds the same string, ``brief`` and ``query runs --wide``.  Only the narrow
``stop_code`` was clean, and only by accident.
"""

from __future__ import annotations

import json

from observe import detail, query
from observe.index import Index
from taic import brief
from taic.redact import STOP_REASON_CHARS

SECRETS = (
    "API-SECRET-1111",
    "sk-abcdefghijklmnopqrstuv",
    "JSON-SECRET-2222",
    "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
)
STOP_REASON = (
    "llm_error: HTTP 401 api_key=API-SECRET-1111 also sk-abcdefghijklmnopqrstuv and "
    '{"api_key": "JSON-SECRET-2222"} plus ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345'
)


def _runs(tmp_path, monkeypatch):
    """Point the readers at a throwaway ``runs/`` tree and a fresh index."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    return runs


def _write_leaky_run(runs, *, stop_reason: str = STOP_REASON):
    run = runs / "relu" / "leaky"
    run.mkdir(parents=True)
    common = {"task": "relu", "dtype": "fp16", "started_at": 10.0, "spent_usd": 0.5, "budget_usd": 2.0}
    (run / "state.json").write_text(
        json.dumps({**common, "phase": "done", "updated_at": 11.0, "stop_reason": stop_reason}),
        encoding="utf-8",
    )
    (run / "result.json").write_text(
        json.dumps({**common, "stop_reason": stop_reason}), encoding="utf-8"
    )
    return run


def _assert_clean(text: str, *, where: str) -> None:
    for secret in SECRETS:
        assert secret not in text, f"{where} still carries {secret!r}: {text!r}"


def test_a_normal_run_stop_reason_is_redacted_in_the_index_and_every_view(tmp_path, monkeypatch):
    """Ordinary run + leaky stop_reason: the column, row_json, FTS, control text and views are clean."""
    runs = _runs(tmp_path, monkeypatch)
    run = _write_leaky_run(runs)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()

    row = index.conn.execute(
        "SELECT stop_reason, stop_code, control_reason, row_json FROM runs WHERE run_id = 'relu/leaky'"
    ).fetchone()
    assert row is not None
    stop_reason, stop_code, control_reason, row_json = row
    _assert_clean(stop_reason, where="runs.stop_reason")
    _assert_clean(control_reason or "", where="runs.control_reason")
    _assert_clean(row_json or "", where="runs.row_json")
    # The narrow projection still says which class of stop this was, and stays bounded.
    assert stop_code == "llm_error"
    assert len(stop_code) <= 24
    # The full text is still there, redacted and bounded.
    assert stop_reason.startswith("llm_error:") and "<redacted>" in stop_reason
    assert len(stop_reason) <= 400

    hits = index.conn.execute(
        "SELECT count(*) FROM runs_fts WHERE runs_fts MATCH ?", ('"API-SECRET-1111"',)
    ).fetchone()[0]
    assert hits == 0

    summary = json.dumps(brief.summarize_run(run), ensure_ascii=False)
    _assert_clean(summary, where="brief")
    assert json.loads(summary)["stop_reason"].startswith("llm_error:")

    wide = json.dumps(query.runs_page(limit=10, compact=False), ensure_ascii=False)
    _assert_clean(wide, where="query runs --wide")

    # The detail view reads the same cleaned text, not the file bytes.
    document = json.dumps(detail.run_detail("relu/leaky"), ensure_ascii=False)
    _assert_clean(document, where="run detail")


def test_the_narrow_stop_code_is_not_cut_out_of_a_token(tmp_path, monkeypatch):
    """A reason that starts with a token must not put 24 characters of it into the narrow column."""
    runs = _runs(tmp_path, monkeypatch)
    directory = runs / "relu" / "token-first"
    directory.mkdir(parents=True)
    # 19 字符：旧代码把它整段切成窄码，新代码先脱敏再取词。
    row = detail.row_from_files(
        directory, {"stop_reason": "sk-short-secret-123: llm_error"}, None
    )
    assert "sk-short-secret-123" not in row["stop_code"], row["stop_code"]
    assert len(row["stop_code"]) <= detail.STOP_CODE_CHARS


def test_a_legacy_file_with_a_leaky_reason_is_cleaned_at_read(tmp_path, monkeypatch):
    """An already-written file cannot be re-written, so the read path must clean it too."""
    runs = _runs(tmp_path, monkeypatch)
    run = _write_leaky_run(runs)
    summary = json.dumps(brief.summarize_run(run), ensure_ascii=False)
    _assert_clean(summary, where="brief over a pre-existing file")
    document = json.dumps(detail.run_detail("relu/leaky"), ensure_ascii=False)
    _assert_clean(document, where="detail over a pre-existing file")


def test_the_control_endpoint_cleans_a_legacy_stop_reason_without_changing_the_action(tmp_path, monkeypatch):
    """``/control`` reads the same old state/result: its text is cleaned, its decision is not.

    The leaky reason contains the JSON ``api_key`` spelling and a ``ghp_`` token.  ``inspect_run``
    decides from the raw text (which is what an old writer left behind) and only the displayed
    ``stop_reason``/``reason_if_not`` are bounded and redacted.
    """
    from observe import actions

    runs = _runs(tmp_path, monkeypatch)
    _write_leaky_run(runs)
    document = actions.describe("relu/leaky")
    text = json.dumps(document, ensure_ascii=False)
    _assert_clean(text, where="/control")
    assert document["stop_reason"].startswith("llm_error:") and "<redacted>" in document["stop_reason"]
    assert len(document["stop_reason"]) <= STOP_REASON_CHARS
    # The old run still stops for a reason other than budget: no budget action is invented.
    assert document["kind"] == "finished"
    assert document["can_resume"] is False and document["can_add_budget"] is False
    assert document["action"] is None

    # The reset planner reads the same ``inspect_run`` document; its copied text is clean too.
    plan = json.dumps(actions._reset_plan_for("relu/leaky"), ensure_ascii=False)
    _assert_clean(plan, where="reset preview")


def test_budget_and_count_semantics_still_authorize_from_the_raw_reason(tmp_path, monkeypatch):
    """A readable old run keeps its resumable decision; only the text that is shown changes."""
    from observe import actions

    runs = _runs(tmp_path, monkeypatch)
    directory = runs / "relu" / "exhausted"
    directory.mkdir(parents=True)
    common = {
        "task": "relu", "dtype": "fp16", "started_at": 10.0, "spent_usd": 2.0, "budget_usd": 2.0,
        "phase": "done", "updated_at": 11.0, "best_label": "r1.p1",
    }
    stop = "budget_exhausted api_key=API-SECRET-1111 ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    (directory / "state.json").write_text(json.dumps({**common, "stop_reason": stop}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({**common, "stop_reason": stop}), encoding="utf-8")
    (directory / "run_config.json").write_text(
        json.dumps({"loop_protocol": "flame_chase_1", "report_schema_epoch": 4}), encoding="utf-8"
    )
    # The raw value is not the exact ``budget_exhausted`` marker, so it stays ``finished``;
    # the redacted text must not turn it into a budget stop either.
    document = actions.describe("relu/exhausted")
    _assert_clean(json.dumps(document, ensure_ascii=False), where="/control for a leaky finished run")
    assert document["kind"] == "finished" and document["can_resume"] is False
    assert "not for budget" in document["reason_if_not"]
    assert "API-SECRET-1111" not in document["reason_if_not"]

    # The exact budget marker (the only value the old writer used for this stop) still authorizes
    # the resume action: cleaning happens after the decision, not instead of it.
    budget = runs / "relu" / "budget"
    budget.mkdir(parents=True)
    (budget / "state.json").write_text(
        json.dumps({**common, "stop_reason": "budget_exhausted"}), encoding="utf-8"
    )
    (budget / "result.json").write_text(
        json.dumps({**common, "stop_reason": "budget_exhausted"}), encoding="utf-8"
    )
    (budget / "run_config.json").write_text(
        json.dumps({"loop_protocol": "flame_chase_1", "report_schema_epoch": 4}), encoding="utf-8"
    )
    exhausted = actions.describe("relu/budget")
    assert exhausted["kind"] == "budget_exhausted"
    assert exhausted["can_resume"] is True and exhausted["action"] == "resume"


def test_an_index_written_before_the_fix_is_repaired_when_it_is_reopened(tmp_path, monkeypatch):
    """An existing DB keeps its old rows until they are rewritten; the rule version does that once."""
    runs = _runs(tmp_path, monkeypatch)
    _write_leaky_run(runs)
    path = tmp_path / "observe.sqlite"
    index = Index(path)
    index.sync_known()
    index.conn.execute(
        "UPDATE runs SET stop_reason = ?, control_reason = ?, stop_code = 'llm_error', row_json = ?"
        " WHERE run_id = 'relu/leaky'",
        (
            STOP_REASON, f"stopped for '{STOP_REASON}'",
            # 旧行的 row_json 缓存了整行，包括 state/result 面板里的同一段原文。
            json.dumps({
                "stop_reason": STOP_REASON,
                "control": {"reason": f"stopped for '{STOP_REASON}'"},
                "state": {"stop_reason": STOP_REASON},
                "result": {"stop_reason": STOP_REASON},
            }),
        ),
    )
    index.conn.execute("DELETE FROM meta WHERE key = 'stop_reason_rule'")
    index.conn.commit()
    index.conn.close()

    reopened = Index(path)
    row = reopened.conn.execute(
        "SELECT stop_reason, stop_code, control_reason, row_json FROM runs WHERE run_id = 'relu/leaky'"
    ).fetchone()
    _assert_clean(row[0], where="repaired runs.stop_reason")
    _assert_clean(row[2] or "", where="repaired runs.control_reason")
    _assert_clean(row[3] or "", where="repaired runs.row_json")
    assert row[1] == "llm_error"
    hits = reopened.conn.execute(
        "SELECT count(*) FROM runs_fts WHERE runs_fts MATCH ?", ('"API-SECRET-1111"',)
    ).fetchone()[0]
    assert hits == 0
    # Idempotent: reopening again does not rewrite or fail.
    reopened.conn.close()
    again = Index(path)
    assert again.conn.execute(
        "SELECT stop_reason FROM runs WHERE run_id = 'relu/leaky'"
    ).fetchone()[0] == row[0]
