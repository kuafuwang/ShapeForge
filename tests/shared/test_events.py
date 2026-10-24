"""Events stay inside the standard library. The dashboard is a subscriber, not a dependency of judging."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import tcenv.events as events


def test_emit_writes_nothing_until_the_file_log_is_subscribed(tmp_path):
    """@brief 验证未安装文件日志监听器时 ``emit`` 不落盘，安装后同一事件才写出 JSON。

    用例先清空监听器、发出快照确认无文件，再 ``install_file_log`` 后重发并就文件内容校验。
    结束时无论成败都重置监听器并恢复原有订阅（函数显式调用 ``events.reset``，但本身不抛异常）。

    @param tmp_path pytest 提供的临时目录，用作事件写入目标 ``state.json`` 的位置。
    """
    saved = list(events._listeners)
    events.reset()
    try:
        target = tmp_path / "state.json"
        events.emit("snapshot", path=target, doc={"phase": "plan"})
        assert not target.exists()
        events.install_file_log()
        events.emit("snapshot", path=target, doc={"phase": "plan"})
        assert json.loads(target.read_text())["phase"] == "plan"
    finally:
        events.reset()
        for listener in saved:
            events.subscribe(listener)
        events.install_file_log()


def test_atomic_snapshot_writers_do_not_share_a_thread_local_temp_name(tmp_path):
    """@brief 并发 progress writer 不得互相替换临时文件或丢失更新。"""
    target = tmp_path / "eval_progress.json"
    start = threading.Barrier(8)
    errors = []

    def write_many(worker: int) -> None:
        try:
            start.wait()
            for index in range(100):
                events._write_json(target, {"worker": worker, "index": index})
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=write_many, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert set(json.loads(target.read_text())) == {"worker", "index"}


def test_a_gpu_lock_note_is_written(tmp_path, monkeypatch):
    """@brief 验证 GPU 锁事件会写出 holder 文件，并在 ``gpu_events.jsonl`` 末尾追加对应记录。

    @param tmp_path pytest 提供的临时目录，充当 observe 目录的根。
    @param monkeypatch pytest 的环境变量夹具，用于把 ``AI_LOWERLING_OBSERVE_DIR`` 指到临时目录。
    """
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path))
    saved = list(events._listeners)
    events.reset()
    try:
        events.note_gpu("acquire", "ex", "benchmark")
        holder = tmp_path / "gpu.d" / f"{os.getpid()}.json"
        assert json.loads(holder.read_text())["role"] == "holder"
        lines = (tmp_path / "observe" / "gpu_events.jsonl").read_text().splitlines()
        assert json.loads(lines[-1])["kind"] == "acquire"
    finally:
        events.note_gpu("release", "ex", "benchmark")
        events.reset()
        for listener in saved:
            events.subscribe(listener)
        events.install_file_log()


def test_core_imports_do_not_load_the_dashboard_stack(tmp_path):
    """@brief 在拦掉 starlette／uvicorn 的子进程里导入判分、策略与配置模块，确认核心不依赖 dashboard 技术栈。

    新解释器先插一个 meta_path 钩子，令 ``starlette``／``uvicorn`` 导入即抛 ImportError，
    再导入 ``tcenv.evaluate``／``tcenv.verify``／``tcenv.schema``／``taic.policy``／``taic.llm``／
    ``tcenv.gpulock`` 并触发一次事件与 GPU 记录；随后断言 ``sys.modules`` 里没有 ``observe``
    与这两个 UI 包，且关键符号可用。这样保证 dashboard 只是订阅者，不是判分链路的依赖。

    @param tmp_path pytest 提供的临时目录，作为子进程的 observe 目录并接收其事件文件。
    """
    root = Path(__file__).resolve().parents[2]
    code = r"""
import os, sys
os.environ["AI_LOWERLING_OBSERVE_DIR"] = sys.argv[1]

class Block:
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in ("starlette", "uvicorn"):
            raise ImportError("import hook blocked " + fullname)
        return None

sys.meta_path.insert(0, Block())
import tcenv.evaluate
import tcenv.verify
import tcenv.schema
import taic.policy
import taic.llm
import tcenv.gpulock
tcenv.evaluate._note_stage(stage="screen", status="ok", task="relu")
tcenv.gpulock._gpu_note("wait", "sh", "decouple")
from tcenv.events import emit
emit("snapshot", path=os.path.join(sys.argv[1], "unused.json"), doc={"n": 1})
bad = [
    name for name in sys.modules
    if name == "observe" or name.startswith("observe.")
    or name == "starlette" or name.startswith("starlette.")
    or name == "uvicorn" or name.startswith("uvicorn.")
]
if bad:
    sys.stderr.write("ui packages in sys.modules: " + ", ".join(sorted(bad)) + "\n")
    raise SystemExit(1)
assert taic.llm.DEFAULT_PRESET
assert callable(taic.policy.relation_to_best)
assert tcenv.verify.UNSUPPORTED == "unsupported"
print("ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["AI_LOWERLING_OBSERVE_DIR"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert proc.stdout.strip() == "ok"
