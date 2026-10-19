"""The dashboard's 'add budget' control: what it offers per run, who may use it, and what it does. No GPU, no model calls, nothing is spawned."""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from observe import actions, query as scan
from taic import control

from .test_observe import dashboard  # noqa: F401  (pytest fixture: the dashboard served on a free port)

DEAD_PID = 2**22 + 31337


def mk(runs: Path, name: str, kind: str, *, spent: float = 3.1, budget: float = 3.0, stop: str = "budget_exhausted", control_supported: bool = False) -> Path:
    d = runs / name
    d.mkdir(parents=True)
    state = {"pid": os.getpid(), "updated_at": time.time(), "started_at": time.time() - 100, "phase": "llm", "task": "flash_attention", "dtype": "fp16", "round": 4, "max_rounds": 12,
             "spent_usd": spent, "budget_usd": budget, "best_speedup": 0.7, "best_label": "r3.p1", "trace": [{"label": "init", "improved": True, "speedup": 0.5}], "ledger": {"total_usd": spent, "budget_usd": budget}}
    if control_supported:
        state["control_supported"] = True
    if kind in ("exhausted", "finished"):
        state.update(phase="done", pid=DEAD_PID, stop_reason=stop)
        (d / "result.json").write_text(json.dumps({"task": "flash_attention", "dtype": "fp16", "stop_reason": stop, "spent_usd": spent, "budget_usd": budget, "best_label": "r3.p1", "best_speedup": 0.7, "confirmed_speedup": 0.69, "rounds": 4}))
    elif kind == "interrupted":
        state["pid"] = DEAD_PID
        state["updated_at"] = time.time() - 600
    (d / "state.json").write_text(json.dumps(state))
    return d


def call(port: int, method: str, path: str, body=None, *, headers: dict | None = None, host: str | None = None, raw: bytes | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Content-Type": "application/json", **(headers or {})}
    if host:
        h["Host"] = host
    conn.request(method, path, body=raw if raw is not None else (json.dumps(body) if body is not None else None), headers=h)
    r = conn.getresponse()
    data = r.read()
    try:
        doc = json.loads(data) if data else None
    except ValueError:
        doc = data.decode("utf-8", "replace")
    conn.close()
    return r.status, doc


def token_of(port: int) -> str:
    status, page = call(port, "GET", "/")
    assert status == 200
    return re.search(r'<meta name="observe-token" content="([^"]*)"', page).group(1)


def post(port: int, run: str, body, *, token: str | None = None, **kw):
    tok = token_of(port) if token is None else token
    return call(port, "POST", f"/api/runs/{run}/budget", body, headers={actions.TOKEN_HEADER: tok}, **kw)


@pytest.fixture
def runs(dashboard):  # noqa: F811
    port, runs = dashboard
    return port, runs


# ------------------------------------------------------------------------------------------------ GET control: what is possible


def test_control_for_every_situation(runs):
    port, root = runs
    mk(root, "live_new", "running", spent=2.5, control_supported=True)
    mk(root, "live_old", "running", spent=2.5)
    mk(root, "exhausted", "exhausted")
    mk(root, "patience", "finished", stop="converged (4 rounds without improvement)", spent=1.0)
    mk(root, "dead", "interrupted", spent=1.0)
    held = mk(root, "held", "interrupted", spent=1.0)
    other = subprocess.Popen(["sleep", "30"])
    try:
        (held / "resume.lock").write_text(json.dumps({"pid": other.pid, "started_at": time.time()}))
        got = {name: call(port, "GET", f"/api/runs/{name}/control")[1] for name in ("live_new", "live_old", "exhausted", "patience", "dead", "held")}
    finally:
        other.kill()
        other.wait(timeout=5)
    assert got["live_new"]["action"] == "topup_live" and got["live_new"]["can_topup_live"] and not got["live_new"]["can_resume"]
    assert got["live_old"]["action"] is None and "older version" in got["live_old"]["reason_if_not"] and "wait for it to stop" in got["live_old"]["reason_if_not"]
    assert got["exhausted"]["action"] == "resume" and got["exhausted"]["can_resume"] and got["exhausted"]["budget"]["needs_budget"] is True
    assert got["exhausted"]["suggested_amounts"] == [3.0, 5.0, 15.0] and got["exhausted"]["max_topup_usd"] == 25.0
    assert got["patience"]["can_add_budget"] is False and "not for budget" in got["patience"]["reason_if_not"]
    assert got["dead"]["action"] == "resume" and got["dead"]["kind"] == "interrupted" and any("older version" in w for w in got["dead"]["warnings"])
    assert got["held"]["can_add_budget"] is False and "already being resumed" in got["held"]["reason_if_not"]


def test_control_of_an_unknown_or_escaping_run_is_404(runs, tmp_path):
    port, root = runs
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "state.json").write_text(json.dumps({"task": "x"}))
    (root / "link").symlink_to(outside)
    for run in ("nope", "../outside", "..%2foutside", "link", "%2e%2e/outside", "a/../../outside"):
        status, _ = call(port, "GET", f"/api/runs/{run}/control")
        assert status == 404, run
        status, _ = post(port, run, {"amount_usd": 1})
        assert status == 404, run
    assert not (outside / "control.json").exists() and not (outside / "budget_events.json").exists()


# ------------------------------------------------------------------------------------------------ POST: who may ask


def test_a_post_needs_the_token_the_page_carries(runs):
    port, root = runs
    d = mk(root, "live", "running", spent=2.5, control_supported=True)
    assert call(port, "POST", "/api/runs/live/budget", {"amount_usd": 3})[0] == 403  # no token
    assert post(port, "live", {"amount_usd": 3}, token="wrong")[0] == 403
    assert not (d / "control.json").exists()
    assert len(token_of(port)) >= 30
    status, doc = post(port, "live", {"amount_usd": 3})
    assert status == 200 and doc["action"] == "topup_live" and doc["new_budget_usd"] == pytest.approx(6.0)


def test_cross_origin_and_rebound_hosts_are_refused(runs):
    port, root = runs
    d = mk(root, "live", "running", spent=2.5, control_supported=True)
    tok = token_of(port)
    h = {actions.TOKEN_HEADER: tok}
    assert call(port, "POST", "/api/runs/live/budget", {"amount_usd": 3}, headers={**h, "Origin": "http://evil.example"})[0] == 403
    assert call(port, "POST", "/api/runs/live/budget", {"amount_usd": 3}, headers={**h, "Origin": f"http://127.0.0.1:{port + 1}"})[0] == 403
    assert call(port, "POST", "/api/runs/live/budget", {"amount_usd": 3}, headers={**h, "Origin": "null"})[0] == 403
    assert call(port, "POST", "/api/runs/live/budget", {"amount_usd": 3}, headers=h, host=f"evil.example:{port}")[0] == 403  # DNS rebinding
    assert call(port, "GET", "/api/runs/live/control", host="evil.example")[0] == 403
    # a rebound hostname never receives the token
    status, page = call(port, "GET", "/", host=f"evil.example:{port}")
    assert re.search(r'<meta name="observe-token" content=""', page)
    assert not (d / "control.json").exists()
    # the page's own origin is fine
    assert call(port, "POST", "/api/runs/live/budget", {"amount_usd": 3}, headers={**h, "Origin": f"http://127.0.0.1:{port}"})[0] == 200


@pytest.mark.parametrize("body", [{"amount_usd": 0}, {"amount_usd": -2}, {"amount_usd": "lots"}, {"amount_usd": True}, {"amount_usd": None}, {}, {"amount_usd": 1, "new_budget_usd": 9}, [], "x", {"amount_usd": 25.01}, {"new_budget_usd": 2.0}, {"new_budget_usd": 100}])
def test_bad_amounts_are_refused(runs, body):
    port, root = runs
    d = mk(root, "live", "running", spent=2.5, control_supported=True)
    status, doc = post(port, "live", body)
    assert status == 400 and doc["ok"] is False and doc["error"]
    assert not (d / "control.json").exists()


def test_nan_and_garbage_bodies_are_refused(runs):
    port, root = runs
    d = mk(root, "live", "running", spent=2.5, control_supported=True)
    tok = token_of(port)
    assert call(port, "POST", "/api/runs/live/budget", raw=b'{"amount_usd": NaN}', headers={actions.TOKEN_HEADER: tok})[0] == 400
    assert call(port, "POST", "/api/runs/live/budget", raw=b"not json", headers={actions.TOKEN_HEADER: tok})[0] == 400
    assert call(port, "POST", "/api/runs/live/budget", raw=b"x" * 5000, headers={actions.TOKEN_HEADER: tok})[0] == 413
    assert not (d / "control.json").exists()


def test_the_cap_is_configurable(runs, monkeypatch):
    port, root = runs
    mk(root, "live", "running", spent=2.5, control_supported=True)
    monkeypatch.setenv("OBSERVE_MAX_TOPUP_USD", "4")
    assert post(port, "live", {"amount_usd": 4.5})[0] == 400
    status, doc = post(port, "live", {"amount_usd": 4})
    assert status == 200 and doc["new_budget_usd"] == pytest.approx(7.0)


# ------------------------------------------------------------------------------------------------ POST: what it does


def test_a_live_top_up_writes_the_control_file_once(runs):
    port, root = runs
    d = mk(root, "live", "running", spent=2.5, control_supported=True)
    status, doc = post(port, "live", {"amount_usd": 5})
    assert status == 200 and doc["old_budget_usd"] == 3.0 and doc["new_budget_usd"] == 8.0 and doc["dry_run"] is False
    ctl = json.loads((d / "control.json").read_text())
    assert ctl["budget_usd"] == 8.0 and ctl["who"] == "dashboard" and ctl["amount_usd"] == 5.0
    ev = control.read_events(d)
    assert [(e["kind"], e["who"], e["old_budget_usd"], e["new_budget_usd"]) for e in ev] == [("requested", "dashboard", 3.0, 8.0)]
    # the run has not read it yet: a second request waits
    status, doc = post(port, "live", {"amount_usd": 1})
    assert status == 409 and "has not been read" in doc["error"]
    # the run takes it and records that it did: the next request is possible again
    control.record_event(d, "applied", 3.0, 8.0, who="dashboard")
    st = json.loads((d / "state.json").read_text())
    st["budget_usd"] = 8.0
    (d / "state.json").write_text(json.dumps(st))
    assert post(port, "live", {"new_budget_usd": 9})[0] == 200


def test_a_run_started_by_an_older_version_cannot_be_topped_up_live(runs):
    port, root = runs
    d = mk(root, "old", "running", spent=2.5)
    status, doc = post(port, "old", {"amount_usd": 5})
    assert status == 409 and "older version" in doc["error"] and "wait for it to stop" in doc["error"]
    assert not (d / "control.json").exists()


def test_runs_that_did_not_stop_for_budget_are_refused_with_the_reason(runs):
    port, root = runs
    d = mk(root, "patience", "finished", stop="converged (4 rounds without improvement)", spent=1.0)
    status, doc = post(port, "patience", {"amount_usd": 5})
    assert status == 409 and "not for budget" in doc["error"]
    assert not (d / "budget_events.json").exists()
    d2 = mk(root, "hang", "finished", stop="gpu_hang_circuit_breaker", spent=1.0)
    assert post(port, "hang", {"amount_usd": 5})[0] == 409 and not (d2 / "budget_events.json").exists()


def test_dry_run_records_the_command_and_starts_nothing(runs, monkeypatch):
    port, root = runs
    d = mk(root, "exhausted", "exhausted")
    monkeypatch.setenv("OBSERVE_CONTROL_DRYRUN", "1")

    def boom(*a, **k):
        raise AssertionError("a dry run must not start a process")

    monkeypatch.setattr(actions.subprocess, "Popen", boom)
    assert call(port, "GET", "/api/runs/exhausted/control")[1]["dry_run"] is True
    status, doc = post(port, "exhausted", {"amount_usd": 5})
    assert status == 200 and doc["dry_run"] is True and doc["action"] == "resume" and doc["new_budget_usd"] == pytest.approx(8.0)
    cmd = doc["command"]
    assert "resume" in cmd and str(d.resolve()) in cmd and cmd[cmd.index("--budget") + 1] == "8.000000" and cmd[-2:] == ["--who", "dashboard"]
    ev = control.read_events(d)
    assert [e["kind"] for e in ev] == ["dryrun"] and ev[0]["command"] == cmd and "would start" in ev[0]["note"]
    assert not (d / "resume.lock").exists() and not list(d.glob("resume_*.log"))
    # a live run: the control file is not written either
    mk(root, "live", "running", spent=2.5, control_supported=True)
    status, doc = post(port, "live", {"amount_usd": 5})
    assert status == 200 and doc["dry_run"] is True and not (root / "live" / "control.json").exists()


def test_resume_is_started_detached_with_a_log_and_a_second_click_is_refused(runs, monkeypatch):
    port, root = runs
    d = mk(root, "exhausted", "exhausted")
    started = []
    other = subprocess.Popen(["sleep", "30"])

    class FakeProc:
        pid = other.pid  # a live pid that is not this dashboard process

    def fake_popen(cmd, **kw):
        started.append((cmd, kw))
        return FakeProc()

    monkeypatch.setattr(actions.subprocess, "Popen", fake_popen)
    try:
        status, doc = post(port, "exhausted", {"new_budget_usd": 10})
        assert status == 200 and doc["action"] == "resume" and doc["new_budget_usd"] == 10.0 and doc["log"] == "resume_1.log"
        (cmd, kw), = started
        assert kw["start_new_session"] is True and Path(kw["cwd"]) == actions.REPO_ROOT and kw["stdin"] is not None
        assert cmd[cmd.index("--budget") + 1] == "10.000000" and "resume" in cmd and "--who" in cmd
        assert (d / "resume_1.log").exists()
        lock = json.loads((d / "resume.lock").read_text())
        assert lock["pid"] == other.pid and lock["starting"] is True
        assert [e["kind"] for e in control.read_events(d)] == ["requested"]
        status, doc = post(port, "exhausted", {"amount_usd": 5})
        assert status == 409 and "already being resumed" in doc["error"] and len(started) == 1
    finally:
        other.kill()
        other.wait(timeout=5)


def test_the_page_script_and_markup_carry_the_control(runs):
    port, _root = runs
    status, page = call(port, "GET", "/")
    assert status == 200
    for needle in ("X-Observe-Token", "budget-modal", "reset-all", "stop-all", "settings-open", "花费账本", "/api/budget/reset", "/api/stop", "/api/settings/model", "/control", "Add budget"):
        assert needle in page, needle
    status, ledger = call(port, "GET", "/api/v2/ledger")
    assert status == 200 and "by_operator" in ledger and "by_month" in ledger


# ------------------------------------------------------------------------------------------------ what the page then shows


def test_budget_status_is_judged_against_the_current_budget_and_the_original_stays_visible(runs):
    port, root = runs
    d = mk(root, "p1/fa_flash", "exhausted", spent=3.123, budget=3.0)
    row = next(r for r in scan.history() if r["id"] == "p1/fa_flash")
    assert row["budget"]["level"] == "over" and row["topped_up"] is False and row["control"]["action"] == "resume"
    # a resume that took 5 more dollars: the state of the continued run says so, and the events keep the first budget
    control.record_event(d, "resume", 3.0, 8.0, who="dashboard")
    (d / "result.json").unlink()
    st = json.loads((d / "state.json").read_text())
    st.update(phase="llm", pid=os.getpid(), budget_usd=8.0, spent_usd=3.2, stop_reason="", control_supported=True, updated_at=time.time())
    st["ledger"] = {"total_usd": 3.2, "budget_usd": 8.0}
    (d / "state.json").write_text(json.dumps(st))
    row = next(r for r in scan.history() if r["id"] == "p1/fa_flash")
    assert row["status"] == "running" and row["budget_usd"] == 8.0 and row["budget"]["level"] == "ok"
    assert row["topped_up"] is True and row["initial_budget_usd"] == 3.0 and row["control"]["action"] == "topup_live"
    detail = scan.run_detail("p1/fa_flash")
    assert detail["topped_up"] is True and detail["budget_events"][0]["kind"] == "resume" and detail["control"]["action"] == "topup_live"


def test_reset_adds_15_dollars_lifts_count_limits_and_skips_a_hang(runs, monkeypatch):
    port, root = runs
    live = mk(root, "live", "running", spent=2.5, budget=3.0, control_supported=True)
    (live / "resume.lock").write_text(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
    stopped = mk(root, "patience", "finished", stop="converged (4 rounds without improvement)", spent=1.0, budget=3.0)
    hang = mk(root, "hang", "finished", stop="gpu_hang_circuit_breaker", spent=1.0, budget=3.0)
    (root / "cap.json").write_text(json.dumps({"max_total_usd": 5.0}))
    started = []

    class Proc:
        pid = 424242

    def fake_popen(cmd, **_k):
        started.append(cmd)
        return Proc()

    monkeypatch.setattr(actions.subprocess, "Popen", fake_popen)
    tok = token_of(port)
    status, preview = call(port, "GET", "/api/budget/reset")
    assert status == 200
    assert {row["id"] for row in preview["runs"]} == {"live", "patience"}
    assert [row["id"] for row in preview["skipped"]] == ["hang"]
    assert next(row for row in preview["runs"] if row["id"] == "live")["new_budget_usd"] == pytest.approx(18.0)
    status, doc = call(port, "POST", "/api/budget/reset", {}, headers={actions.TOKEN_HEADER: tok})
    assert status == 200 and doc["ok"] is True
    assert {row["id"] for row in doc["runs"]} == {"live", "patience"}
    assert [row["id"] for row in doc["skipped"]] == ["hang"]
    ctl = json.loads((live / "control.json").read_text())
    assert ctl["budget_usd"] == pytest.approx(18.0) and ctl["max_rounds"] == 100_000 and ctl["patience"] == 100_000
    assert started and "--lift-count-limits" in started[0] and "100000" in started[0]
    assert str(stopped.resolve()) in started[0]
    assert not (hang / "control.json").exists() and not (hang / "resume.lock").exists()
    cap = json.loads((root / "cap.json").read_text())["max_total_usd"]
    assert cap > 5.0
    # a plain add-budget still refuses a run that stopped on patience
    assert post(port, "patience", {"amount_usd": 5})[0] == 409


def test_parse_ps_keeps_search_processes_and_drops_the_dashboard():
    text = "\n".join([
        "11 /usr/bin/python -m taic.cli resume runs/p1/fa_paper",
        "12 uv run python -m experiments.search_ab --which fa",
        "13 /usr/bin/python -m observe",
        "14 /usr/bin/python -m tcenv.worker /tmp/req.json",
        "not-a-pid hello",
    ])
    assert actions._parse_ps(text) == [
        (11, "/usr/bin/python -m taic.cli resume runs/p1/fa_paper"),
        (12, "uv run python -m experiments.search_ab --which fa"),
    ]


def test_stop_signals_the_search_process_group_and_not_the_dashboard(runs, monkeypatch):
    port, root = runs
    live = mk(root, "live", "running", spent=1.0, budget=3.0, control_supported=True)
    state = json.loads((live / "state.json").read_text())
    state["pid"] = 4242
    (live / "state.json").write_text(json.dumps(state))
    mk(root, "done", "finished", stop="max_rounds", spent=1.0, budget=3.0)
    monkeypatch.setattr(actions, "_ps_commands", lambda: [(4242, "/usr/bin/python -m taic.cli resume runs/live")])
    monkeypatch.setattr(actions, "_command_of", lambda pid: "/usr/bin/python -m taic.cli resume runs/live" if int(pid) == 4242 else "python -m pytest")

    def alive(pid):
        return int(pid) in (os.getpid(), 4242)

    monkeypatch.setattr(scan, "pid_alive", alive)
    monkeypatch.setattr(actions.os, "getpgid", lambda pid: 9000 if int(pid) == 4242 else os.getpgrp())
    signaled = []
    monkeypatch.setattr(actions.os, "kill", lambda pid, sig: signaled.append((pid, sig)))
    tok = token_of(port)
    status, preview = call(port, "GET", "/api/stop")
    assert status == 200
    assert [group["pgid"] for group in preview["groups"]] == [9000]
    assert preview["groups"][0]["ids"] == ["live"]
    assert all(row.get("pid") != os.getpid() for row in preview["groups"])
    status, doc = call(port, "POST", "/api/stop", {}, headers={actions.TOKEN_HEADER: tok})
    assert status == 200 and doc["ok"] is True
    assert signaled == [(-9000, __import__("signal").SIGTERM)]
    assert not (live / "result.json").exists()  # stopping does not invent a finished result
    # a finished search is not in the stop list
    assert all("done" not in group["ids"] for group in doc["stopped"])


def test_model_picker_saves_the_choice_and_hides_the_key(runs, tmp_path, monkeypatch):
    port, _root = runs
    secret = "sk-SECRET-do-not-leak"
    cred = tmp_path / "models.json"
    cred.write_text(json.dumps({"providers": {"deepseek": {"apiKey": secret, "baseUrl": "https://secret.example", "models": [
        {"id": "deepseek-flash", "name": "DeepSeek V4.1 Flash"},
        {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro"},
    ]}}}))
    cfg = tmp_path / "taic.toml"
    cfg.write_text(f'[llm]\ncredentials_file = "{cred}"\nprovider = "deepseek"\nmodel = "deepseek-flash"  # keep this comment\n[search]\nbudget_usd = 2\n')
    monkeypatch.setenv("TAIC_CONFIG", str(cfg))
    status, doc = call(port, "GET", "/api/settings/model")
    assert status == 200 and doc["model"] == "deepseek-flash" and doc["provider"] == "deepseek"
    assert [row["id"] for row in doc["choices"]] == ["deepseek-flash", "deepseek-v4-pro"]
    blob = json.dumps(doc)
    assert secret not in blob and "secret.example" not in blob and "apiKey" not in blob
    tok = token_of(port)
    status, saved = call(port, "POST", "/api/settings/model", {"provider": "deepseek", "model": "deepseek-v4-pro"}, headers={actions.TOKEN_HEADER: tok})
    assert status == 200 and saved == {"provider": "deepseek", "model": "deepseek-v4-pro"}
    text = cfg.read_text()
    assert 'provider = "deepseek"' in text and 'model = "deepseek-v4-pro"  # keep this comment' in text
    assert secret not in text
    assert call(port, "POST", "/api/settings/model", {"provider": "deepseek", "model": "nope"}, headers={actions.TOKEN_HEADER: tok})[0] == 400
