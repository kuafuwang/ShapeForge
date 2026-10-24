"""The shared dashboard fixture, loaded as a pytest plugin."""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
import urllib.request

import pytest


def _free_port() -> int:
    """@brief 向系统申请一个空闲的本地端口，避免测试之间抢占。

    @return 已释放但当前空闲的端口号。
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _Events(threading.Thread):
    def __init__(self, url: str) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.events: queue.Queue = queue.Queue()
        self._resp = None

    def run(self) -> None:
        self._resp = urllib.request.urlopen(self.url, timeout=5)
        try:
            while True:
                line = self._resp.readline()
                if not line:
                    break
                if line.startswith(b"data: "):
                    self.events.put(json.loads(line[6:]))
        except Exception:
            return

    def close(self) -> None:
        if self._resp is not None:
            self._resp.close()


def _wait(events: queue.Queue, predicate, timeout=3.0):
    """@brief 从事件队列中阻塞等待第一条满足条件的事件。

    @param events 由 SSE 线程写入的事件队列。
    @param predicate 判定单条事件是否即目标状态的谓词。
    @param timeout 最长等待秒数。
    @return 第一条满足谓词的事件。
    @raises AssertionError 超时仍未出现满足条件的事件时抛出。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            item = events.get(timeout=0.2)
        except queue.Empty:
            continue
        if predicate(item):
            return item
    raise AssertionError("event stream did not reach the expected state")


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    """@brief 在后台线程启动一个指向临时 runs 目录的 dashboard，并交给测试端口与目录。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的 monkeypatch，用于把运行目录等环境变量指向临时目录。
    @return 生成器，产出 (端口, runs 目录)，测试结束后关闭服务线程。
    @raises RuntimeError 服务在限定轮询内未能启动时抛出。
    """
    import uvicorn

    from observe.app import app

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("OBSERVE_POLL_S", "0.05")
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("dashboard did not start")
    yield port, runs
    server.should_exit = True
    thread.join(timeout=2)
