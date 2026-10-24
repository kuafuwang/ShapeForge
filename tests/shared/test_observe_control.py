"""The dashboard's 'add budget' control: what it offers per run, who may use it, and what it does. No GPU, no model calls, nothing is spawned."""

from __future__ import annotations

import http.client
import json
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from observe import actions, query as scan
from taic import control


DEAD_PID = 2**22 + 31337


def mk(runs: Path, name: str, kind: str, *, spent: float = 3.1, budget: float = 3.0, stop: str = "budget_exhausted", control_supported: bool = False) -> Path:
    """@brief 在运行根目录里手写一个看板可读的运行（state.json，按 kind 补 result.json 或把 pid 改成已死）。

    @param runs 运行根目录。
    @param name 运行目录名，可以带子目录。
    @param kind 运行形态：running、exhausted、finished 或 interrupted。
    @param spent 已花金额。
    @param budget 预算上限。
    @param stop 停止原因，写进 state.json 与 result.json。
    @param control_supported 是否把 state.json 标成支持在线改预算。
    @return 建好的运行目录路径。
    """
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


def _flame_chase(d):
    """@brief 给手搭的运行目录写上 loop_protocol: flame_chase_1，使其成为本代码可续的运行（参见 taic/resume._refuse_old_protocol）。

    @param d 运行目录。
    @return 同一个目录，便于链式使用。
    """
    (d / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1"}))
    return d


def call(port: int, method: str, path: str, body=None, *, headers: dict | None = None, host: str | None = None, raw: bytes | None = None):
    """@brief 向看板发一条 HTTP 请求，并把响应解析成 JSON（或文本）。

    @param port 看板端口。
    @param method HTTP 方法。
    @param path 请求路径。
    @param body 要序列化成 JSON 的请求体，未给则不发正文。
    @param headers 额外的请求头。
    @param host 覆盖 Host 头，用于 DNS rebinding 用例。
    @param raw 直接发送的原始字节，优先于 body。
    @return (状态码, 解析后的响应体) 二元组。
    """
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
    """@brief 从看板首页的 observe-token meta 标签里取出当前控制令牌。

    @param port 看板端口。
    @return 页面携带的令牌字符串。
    """
    status, page = call(port, "GET", "/")
    assert status == 200
    return re.search(r'<meta name="observe-token" content="([^"]*)"', page).group(1)


def post(port: int, run: str, body, *, token: str | None = None, **kw):
    """@brief 向某个运行的加预算接口发一条带令牌的 POST。

    @param port 看板端口。
    @param run 运行 id。
    @param body 请求体。
    @param token 指定的令牌，缺省则从首页读取。
    @param kw 透传给 call 的额外参数（如 host、raw）。
    @return (状态码, 响应体) 二元组。
    """
    tok = token_of(port) if token is None else token
    return call(port, "POST", f"/api/runs/{run}/budget", body, headers={actions.TOKEN_HEADER: tok}, **kw)


@pytest.fixture
def runs(dashboard):  # noqa: F811
    """@brief 把 dashboard fixture 给出的 (端口, 运行根目录) 直接转成本用例要的参数。

    @param dashboard 提供看板端口与临时运行根目录的 fixture。
    @return (端口, 运行根目录) 二元组。
    """
    port, runs = dashboard
    return port, runs


# ------------------------------------------------------------------------------------------------ GET control: what is possible


def test_control_for_every_situation(runs):
    """@brief 验证 GET /control 对每种运行形态给出的动作与理由：可在线加预算、版本太旧、可续、非预算停止、锁被占用。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    """
    port, root = runs
    mk(root, "live_new", "running", spent=2.5, control_supported=True)
    mk(root, "live_old", "running", spent=2.5)
    _flame_chase(mk(root, "exhausted", "exhausted"))
    mk(root, "patience", "finished", stop="converged (4 rounds without improvement)", spent=1.0)
    _flame_chase(mk(root, "dead", "interrupted", spent=1.0))
    held = _flame_chase(mk(root, "held", "interrupted", spent=1.0))
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
    # 没有 loop_protocol 的目录不可续（taic resume 会拒绝），所以「设置靠猜」的警告只在能续的运行上出现过；现在那条路为空。
    assert got["dead"]["action"] == "resume" and got["dead"]["kind"] == "interrupted" and got["dead"]["warnings"] == []
    assert got["held"]["can_add_budget"] is False and "already being resumed" in got["held"]["reason_if_not"]


def test_control_of_an_unknown_or_escaping_run_is_404(runs, tmp_path):
    """@brief 验证不存在或试图越出运行根目录的运行 id（含符号链接与 URL 编码）在看板的 GET 与 POST 上都返回 404，且不写任何文件。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param tmp_path pytest 提供的临时目录，用来放根目录外的目录与链接。
    """
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
    """@brief 验证加预算的 POST 必须带上页面携带的令牌，缺失或错误都返回 403 且不写 control.json。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    """
    port, root = runs
    d = mk(root, "live", "running", spent=2.5, control_supported=True)
    assert call(port, "POST", "/api/runs/live/budget", {"amount_usd": 3})[0] == 403  # no token
    assert post(port, "live", {"amount_usd": 3}, token="wrong")[0] == 403
    assert not (d / "control.json").exists()
    assert len(token_of(port)) >= 30
    status, doc = post(port, "live", {"amount_usd": 3})
    assert status == 200 and doc["action"] == "topup_live" and doc["new_budget_usd"] == pytest.approx(6.0)


def test_cross_origin_and_rebound_hosts_are_refused(runs):
    """@brief 验证跨站 Origin（含 null 与别的端口）和被 DNS rebinding 换掉的 Host 都被拒绝，且换过的主机名拿不到令牌。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    """
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
    """@brief 验证各种非法金额与多余字段的请求体都返回 400，且不产生 control.json。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param body 参数化的请求体。
    """
    port, root = runs
    d = mk(root, "live", "running", spent=2.5, control_supported=True)
    status, doc = post(port, "live", body)
    assert status == 400 and doc["ok"] is False and doc["error"]
    assert not (d / "control.json").exists()


def test_nan_and_garbage_bodies_are_refused(runs):
    """@brief 验证 NaN、非 JSON 与超大请求体分别以 400、400、413 被拒，且不产生 control.json。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    """
    port, root = runs
    d = mk(root, "live", "running", spent=2.5, control_supported=True)
    tok = token_of(port)
    assert call(port, "POST", "/api/runs/live/budget", raw=b'{"amount_usd": NaN}', headers={actions.TOKEN_HEADER: tok})[0] == 400
    assert call(port, "POST", "/api/runs/live/budget", raw=b"not json", headers={actions.TOKEN_HEADER: tok})[0] == 400
    assert call(port, "POST", "/api/runs/live/budget", raw=b"x" * 5000, headers={actions.TOKEN_HEADER: tok})[0] == 413
    assert not (d / "control.json").exists()


def test_the_cap_is_configurable(runs, monkeypatch):
    """@brief 验证 OBSERVE_MAX_TOPUP_USD 能改写单次加预算上限：超限被拒，刚好等于上限则通过。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param monkeypatch pytest 的替换工具，用来改环境变量。
    """
    port, root = runs
    mk(root, "live", "running", spent=2.5, control_supported=True)
    monkeypatch.setenv("OBSERVE_MAX_TOPUP_USD", "4")
    assert post(port, "live", {"amount_usd": 4.5})[0] == 400
    status, doc = post(port, "live", {"amount_usd": 4})
    assert status == 200 and doc["new_budget_usd"] == pytest.approx(7.0)


# ------------------------------------------------------------------------------------------------ POST: what it does


def test_a_live_top_up_writes_the_control_file_once(runs):
    """@brief 验证对运行中的搜索加预算只写一次 control.json 并记下 requested 事件，运行尚未读取前第二次请求返回 409，读走后才允许再改。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    """
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
    """@brief 验证旧版本启动的运行不能在线加预算，而是返回 409 并提示等它停止，且不写 control.json。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    """
    port, root = runs
    d = mk(root, "old", "running", spent=2.5)
    status, doc = post(port, "old", {"amount_usd": 5})
    assert status == 409 and "older version" in doc["error"] and "wait for it to stop" in doc["error"]
    assert not (d / "control.json").exists()


def test_runs_that_did_not_stop_for_budget_are_refused_with_the_reason(runs):
    """@brief 验证不是因预算停止的运行（耐心收敛或 GPU 挂起熔断）加预算被 409 拒绝，并给出原因。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    """
    port, root = runs
    d = mk(root, "patience", "finished", stop="converged (4 rounds without improvement)", spent=1.0)
    status, doc = post(port, "patience", {"amount_usd": 5})
    assert status == 409 and "not for budget" in doc["error"]
    assert not (d / "budget_events.json").exists()
    d2 = mk(root, "hang", "finished", stop="gpu_hang_circuit_breaker", spent=1.0)
    assert post(port, "hang", {"amount_usd": 5})[0] == 409 and not (d2 / "budget_events.json").exists()


def test_dry_run_records_the_command_and_starts_nothing(runs, monkeypatch):
    """@brief 验证 dry-run 模式只把将执行的 resume 命令记进事件，不真的起进程、不写 control.json 与日志，也不留运行锁。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param monkeypatch pytest 的替换工具，用来开启 dry-run 并把 Popen 换成会报错的桩。
    """
    port, root = runs
    d = _flame_chase(mk(root, "exhausted", "exhausted"))
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
    """@brief 验证从看板点续跑会 detached 起进程、写下日志与锁，并记下 requested 事件；第一次点击之后的第二次被 409 拒绝。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param monkeypatch pytest 的替换工具，用来把 Popen 换成返回活 pid 的桩。
    """
    port, root = runs
    d = _flame_chase(mk(root, "exhausted", "exhausted"))
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
    """@brief 验证首页仍带着控制所需的令牌头、预算弹窗、全局重置与停止入口、模型选择与账本接口。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    """
    port, _root = runs
    status, page = call(port, "GET", "/")
    assert status == 200
    for needle in ("X-Observe-Token", "budget-modal", "reset-all", "stop-all", "settings-open", "settings-committees", "data-committee", "花费账本", "/api/budget/reset", "/api/stop", "/api/settings/model", "/control", "Add budget"):
        assert needle in page, needle
    status, ledger = call(port, "GET", "/api/v2/ledger")
    assert status == 200 and "by_operator" in ledger and "by_month" in ledger


# ------------------------------------------------------------------------------------------------ what the page then shows


def test_budget_status_is_judged_against_the_current_budget_and_the_original_stays_visible(runs):
    """@brief 验证预算状态按当前预算判定，而加过预算的运行仍能被识别为 topped_up 并展出最初的预算。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    """
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
    """@brief 验证一键 reset 给每条在跑或停下的运行加 15 美元、抬高峰值与耐心值、跳过挂起熔断的运行，并抬高实验总额上限。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param monkeypatch pytest 的替换工具，用来把 Popen 换成记录命令的桩。
    """
    port, root = runs
    live = mk(root, "live", "running", spent=2.5, budget=3.0, control_supported=True)
    (live / "resume.lock").write_text(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
    stopped = _flame_chase(mk(root, "patience", "finished", stop="converged (4 rounds without improvement)", spent=1.0, budget=3.0))
    hang = _flame_chase(mk(root, "hang", "finished", stop="gpu_hang_circuit_breaker", spent=1.0, budget=3.0))
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
    """@brief 验证 _parse_ps 只留下搜索类进程（taic cli resume 与 experiments.search_ab），丢掉看板自身、评测 worker 与不成形的行。
    """
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
    """@brief 验证停止操作只向搜索进程组发 SIGTERM 而不管看板自身，也不会给停下的运行伪造 result.json。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param monkeypatch pytest 的替换工具，用来桩掉进程表、pid 存活与 kill。
    """
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


def test_stop_also_signals_the_hmz_session_the_search_started(runs, monkeypatch):
    """@brief 验证停止搜索时，由它拉起的 hmz 会话所在进程组（及其相关进程）也一并收到 SIGTERM。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param monkeypatch pytest 的替换工具，用来桩掉进程表、pid 存活与 kill。
    """
    port, root = runs
    live = mk(root, "live", "running", spent=1.0, budget=3.0, control_supported=True)
    state = json.loads((live / "state.json").read_text())
    state["pid"] = 4242
    (live / "state.json").write_text(json.dumps(state))
    run_dir = str(live.resolve())
    monkeypatch.setattr(actions, "_ps_commands", lambda: [(4242, "/usr/bin/python -m taic.cli run conv2d --run-dir " + run_dir)])
    monkeypatch.setattr(actions, "_command_of", lambda pid: "/usr/bin/python -m taic.cli run conv2d" if int(pid) == 4242 else "")
    monkeypatch.setattr(actions, "_process_table", lambda: [
        {"pid": 4242, "ppid": 1, "pgid": 9000, "command": "python -m taic.cli run"},
        {"pid": 5000, "ppid": 4242, "pgid": 5000, "command": "hmz exec"},
        {"pid": 5001, "ppid": 5000, "pgid": 5000, "command": f"hmz internal fence {run_dir}/agent_work/r1.p1 -- pi"},
        {"pid": 5002, "ppid": 5001, "pgid": 5000, "command": "pi --mode rpc"},
    ])

    def alive(pid):
        return int(pid) in (os.getpid(), 4242)

    monkeypatch.setattr(scan, "pid_alive", alive)
    monkeypatch.setattr(actions.os, "getpgid", lambda pid: 9000 if int(pid) == 4242 else os.getpgrp())
    signaled = []
    monkeypatch.setattr(actions.os, "kill", lambda pid, sig: signaled.append((pid, sig)))
    tok = token_of(port)
    status, preview = call(port, "GET", "/api/stop")
    assert status == 200
    assert preview["groups"][0]["related_pgids"] == [5000]
    status, doc = call(port, "POST", "/api/stop", {}, headers={actions.TOKEN_HEADER: tok})
    assert status == 200 and doc["ok"] is True
    assert signaled == [(-5000, __import__("signal").SIGTERM), (-9000, __import__("signal").SIGTERM)]


def test_model_picker_saves_the_choice_and_hides_the_key(runs, tmp_path, monkeypatch):
    """@brief 验证模型选择接口列出候选、保存到 taic.toml 且保留行内注释，同时不回传 API key 与 base url，未知模型被 400 拒绝。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param tmp_path pytest 提供的临时目录，用来放凭据文件与配置。
    @param monkeypatch pytest 的替换工具，用来指定 TAIC_CONFIG。
    """
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


def test_each_committee_can_keep_its_own_model(runs, tmp_path, monkeypatch):
    """@brief 验证每个委员会可以各自保存模型：只改对应委员会段、保留注释与其他段，缺成员的请求被 400 拒绝。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param tmp_path pytest 提供的临时目录，用来放凭据文件与配置。
    @param monkeypatch pytest 的替换工具，用来指定 TAIC_CONFIG。
    """
    port, _root = runs
    secret = "sk-SECRET-do-not-leak"
    cred = tmp_path / "models.json"
    cred.write_text(json.dumps({"providers": {"deepseek": {"apiKey": secret, "baseUrl": "https://secret.example", "models": [
        {"id": "deepseek-flash", "name": "DeepSeek V4.1 Flash"},
        {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro"},
    ]}}}))
    cfg = tmp_path / "taic.toml"
    cfg.write_text(
        f'[llm]\ncredentials_file = "{cred}"\nprovider = "deepseek"\nmodel = "deepseek-flash"  # keep this comment\n'
        '[[committees]]\nname = "first"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "deepseek-flash"\n'
        '[[committees]]\nname = "second"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "deepseek-flash"\n'
    )
    monkeypatch.setenv("TAIC_CONFIG", str(cfg))
    status, doc = call(port, "GET", "/api/settings/model")
    assert status == 200 and [(row["name"], row["model"]) for row in doc["committees"]] == [("first", "deepseek-flash"), ("second", "deepseek-flash")]
    assert secret not in json.dumps(doc)
    tok = token_of(port)
    status, saved = call(port, "POST", "/api/settings/model", {"committees": [
        {"name": "second", "provider": "deepseek", "model": "deepseek-v4-pro"},
        {"name": "first", "provider": "deepseek", "model": "deepseek-flash"},
    ]}, headers={actions.TOKEN_HEADER: tok})
    assert status == 200 and saved["committees"] == [
        {"name": "first", "provider": "deepseek", "model": "deepseek-flash"},
        {"name": "second", "provider": "deepseek", "model": "deepseek-v4-pro"},
    ]
    text = cfg.read_text()
    assert 'model = "deepseek-flash"  # keep this comment' in text
    assert text.count('model = "deepseek-v4-pro"') == 1
    assert text.index('name = "second"') < text.index('model = "deepseek-v4-pro"')
    assert secret not in text
    assert call(port, "POST", "/api/settings/model", {"committees": [{"name": "first", "provider": "deepseek", "model": "deepseek-flash"}]}, headers={actions.TOKEN_HEADER: tok})[0] == 400


def test_deletion_archives_the_directory_and_keeps_the_reason(runs, tmp_path, monkeypatch):
    """@brief 验证删除把运行目录归档到 deleted/、在日志里留下理由与来源 ip，仍在跑或不可删的请求被拒，账本行保留而 runs 表清空。

    @param runs 由 fixture 给出的 (端口, 运行根目录)。
    @param tmp_path pytest 提供的临时目录，用来放观察索引库。
    @param monkeypatch pytest 的替换工具，用来把索引库指到临时路径。
    """
    port, root = runs
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    finished = mk(root, "box/done", "finished")
    live = mk(root, "box/live", "running")
    tok = token_of(port)
    assert call(port, "GET", "/api/v2/runs?limit=50")[0] == 200
    status, refused = call(port, "POST", "/api/records/delete", {"kind": "operator", "name": "flash_attention", "reason": "想清掉"}, headers={actions.TOKEN_HEADER: tok})
    assert status == 409 and "still running" in refused["error"]
    assert finished.is_dir() and live.is_dir()
    from observe.deletions import log_path
    assert not log_path().exists()
    status, saved = call(port, "POST", "/api/records/delete", {"kind": "experiment", "name": "box/done", "reason": "这组做完了，不再看"}, headers={actions.TOKEN_HEADER: tok})
    assert status == 200 and saved["runs"] == ["box/done"]
    assert not finished.exists() and live.is_dir()
    archived = list((tmp_path / "deleted").rglob("state.json"))
    assert len(archived) == 1 and archived[0].is_file()
    lines = log_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["reason"] == "这组做完了，不再看"
    assert call(port, "POST", "/api/records/delete", {"kind": "experiment", "name": "box/done", "reason": ""}, headers={actions.TOKEN_HEADER: tok})[0] == 400
    assert log_path().read_text(encoding="utf-8").splitlines() == lines
    status, listed = call(port, "GET", "/api/records/deletions")
    assert status == 200 and listed["deletions"][0]["reason"] == "这组做完了，不再看"
    with sqlite3.connect(tmp_path / "observe.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM ledger WHERE run_id = ?", ("box/done",)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM runs WHERE run_id = ?", ("box/done",)).fetchone()[0] == 0
    conv = root / "box" / "conv"
    conv.mkdir(parents=True)
    (conv / "state.json").write_text(json.dumps({"pid": DEAD_PID, "updated_at": time.time() - 600, "started_at": time.time() - 800, "phase": "done", "task": "conv2d", "dtype": "fp16", "spent_usd": 1.0, "budget_usd": 3.0}))
    (conv / "result.json").write_text(json.dumps({"task": "conv2d", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 1.0, "budget_usd": 3.0, "confirmed_speedup": 1.1}))
    assert call(port, "GET", "/api/v2/runs?limit=50")[0] == 200
    status, saved = call(port, "POST", "/api/records/delete", {"kind": "operator", "name": "conv2d", "reason": "这个算子的记录不要了"}, headers={actions.TOKEN_HEADER: tok})
    assert status == 200 and saved["runs"] == ["box/conv"]
    assert not conv.exists() and live.is_dir()
    logged = [json.loads(line) for line in log_path().read_text(encoding="utf-8").splitlines()]
    assert logged[0]["reason"] == "这组做完了，不再看"
    assert logged[1]["kind"] == "operator" and logged[1]["ip"] == "127.0.0.1" and logged[1]["reason"] == "这个算子的记录不要了"


def test_delete_refuses_a_path_that_climbs_out_of_runs(tmp_path, monkeypatch):
    """@brief 验证 _under_runs 只接受运行根目录内的路径，越界、绝对路径与符号链接都抛 Refused。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把运行根目录指到临时路径。
    """
    from observe import deletions, detail
    from observe.actions import Refused

    runs = tmp_path / "runs"
    (runs / "inner").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (runs / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(detail, "runs_root", lambda: runs)
    assert deletions._under_runs("inner") == (runs / "inner").resolve()
    for bad in ("../outside", str(outside), "link"):
        with pytest.raises(Refused):
            deletions._under_runs(bad)
    assert outside.is_dir()


def test_a_deletion_that_cannot_be_logged_is_rolled_back(tmp_path, monkeypatch):
    """@brief 验证删除时如果审计日志写不进去，归档会被回滚：运行目录原样保留，deleted/ 下也没有留下东西。

    归档了却没有审计行的运行，和一次没被记录的删除无法区分。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来把运行根目录与日志写入都换成桩。
    """
    from observe import deletions, detail

    runs = tmp_path / "runs"
    run = runs / "box" / "done"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({"pid": 999999, "phase": "done", "task": "relu", "dtype": "fp16", "updated_at": 1}))
    (run / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "stop_reason": "max_rounds"}))
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "missing.sqlite"))
    monkeypatch.setattr(detail, "runs_root", lambda: runs)
    monkeypatch.setattr(deletions, "_live", lambda _path: False)

    def no_log(_entry):
        raise OSError("disk full")

    monkeypatch.setattr(deletions, "_append", no_log)
    with pytest.raises(OSError):
        deletions.delete_records("run", "box/done", "x")
    assert (run / "state.json").is_file() and (run / "result.json").is_file()
    assert not list((tmp_path / "deleted").rglob("state.json"))

