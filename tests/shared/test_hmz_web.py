"""The optional hmz web sidecar; no browser, GPU or model call."""

from __future__ import annotations

import pytest

from observe.hmz_web import HmzWeb, HmzWebError


class _Stdin:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Proc:
    def __init__(self, lines):
        self.stdin = _Stdin()
        self.stdout = iter(lines)
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def test_start_parses_the_loopback_url_and_reuses_it(monkeypatch):
    proc = _Proc(["hmz web: http://127.0.0.1:43210/?key=secret\n"])
    calls = []

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return proc

    monkeypatch.setattr("observe.hmz_web.subprocess.Popen", fake_popen)
    sidecar = HmzWeb(binary="/bin/sh", startup_timeout=1)
    state = sidecar.start()
    assert state["running"] is True and state["url"] == "http://127.0.0.1:43210/?key=secret"
    again = sidecar.start()
    assert again == state
    assert calls[0][0][1:] == ["web", "--no-open", "--port", "0", "--attached"]
    assert calls[0][1]["start_new_session"] is True
    sidecar.stop()
    assert proc.stdin.closed is True


def test_a_missing_binary_is_reported_without_starting(monkeypatch):
    sidecar = HmzWeb(binary="/definitely/missing/hmz")
    assert sidecar.status()["available"] is False
    with pytest.raises(HmzWebError, match="not installed"):
        sidecar.start()


def test_startup_timeout_terminates_the_child(monkeypatch):
    proc = _Proc([])

    def fake_popen(_argv, **_kwargs):
        return proc

    monkeypatch.setattr("observe.hmz_web.subprocess.Popen", fake_popen)
    sidecar = HmzWeb(binary="/bin/sh", startup_timeout=0.1)
    with pytest.raises(HmzWebError, match="did not provide"):
        sidecar.start()
    assert proc.stdin.closed is True


def test_sidecar_uses_the_same_command_as_the_documented_cli(monkeypatch):
    proc = _Proc(["hmz web: http://127.0.0.1:43211/?key=secret\n"])

    def fake_popen(_argv, **_kwargs):
        return proc

    monkeypatch.setattr("observe.hmz_web.subprocess.Popen", fake_popen)
    sidecar = HmzWeb(binary="/bin/sh", startup_timeout=1)
    sidecar.start()
    assert sidecar.status()["url"].startswith("http://127.0.0.1:")
    sidecar.stop()
