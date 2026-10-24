"""Optional ``hmz web`` sidecar for the search dashboard.

The search dashboard remains the source of truth for candidates, confirmation and trust.
This module only opens Humanize's own epic/session explorer on demand.  It never feeds data
back into the dashboard and never treats the sidecar as publication evidence.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ADDRESS = re.compile(r"hmz web:\s+(http://127\.0\.0\.1:\d+/\?key=\S+)")


class HmzWebError(RuntimeError):
    """@brief ``hmz web`` 无法作为 sidecar 启动。"""


class HmzWeb:
    """@brief 按需管理一个 attached 的 ``hmz web`` 子进程。

    该进程只绑定 loopback，URL 自带访问 key；调用方必须只在授权页面里展示它。
    """

    def __init__(self, binary: str = "hmz", *, startup_timeout: float = 10.0, root: Path | None = None) -> None:
        """@brief 记录启动参数；不会在这里启动进程。

        @param binary ``hmz`` 命令或可执行文件路径。
        @param startup_timeout 等待地址输出的秒数。
        @param root 子进程工作目录；默认仓库根目录。
        """
        self._binary = str(binary)
        self._startup_timeout = max(0.1, float(startup_timeout))
        self._root = Path(root) if root is not None else _REPO_ROOT
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._url = ""
        self._tail = ""
        atexit.register(self.stop)

    def _executable(self) -> str | None:
        """@brief 找到要执行的 hmz；绝对路径也允许。"""
        if os.path.sep in self._binary:
            path = Path(self._binary)
            return str(path) if path.is_file() and os.access(path, os.X_OK) else None
        return shutil.which(self._binary)

    def _alive(self) -> bool:
        """@brief 当前 sidecar 是否仍存活。"""
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        """@brief 返回可展示的 sidecar 状态；不会启动进程。

        @return ``{available, running, url, reason}``；URL 只在运行中返回。
        """
        with self._lock:
            if self._proc is not None and not self._alive():
                self._proc = None
                self._url = ""
            running = self._alive()
            return {
                "available": self._executable() is not None,
                "running": running,
                "url": self._url if running else "",
                "reason": "" if self._executable() is not None else "hmz is not installed or not on PATH",
            }

    def start(self) -> dict:
        """@brief 启动 sidecar；已运行则复用现有 URL。

        @return 与 :meth:`status` 相同，running 为 True 且带 URL。
        @raises HmzWebError hmz 不存在、启动超时或子进程提前退出时。
        """
        with self._lock:
            if self._alive() and self._url:
                return {"available": True, "running": True, "url": self._url, "reason": ""}
            binary = self._executable()
            if binary is None:
                raise HmzWebError("hmz is not installed or not on PATH")
            try:
                proc = subprocess.Popen(
                    [binary, "web", "--no-open", "--port", "0", "--attached"],
                    cwd=self._root,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                raise HmzWebError(f"cannot start hmz web: {exc}") from exc
            found: dict[str, str] = {}
            tail: list[str] = []

            def read_output() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    tail.append(line.rstrip())
                    del tail[:-8]
                    if "url" not in found:
                        match = _ADDRESS.search(line)
                        if match:
                            found["url"] = match.group(1)

            reader = threading.Thread(target=read_output, daemon=True, name="observe-hmz-web")
            reader.start()
            deadline = time.monotonic() + self._startup_timeout
            while time.monotonic() < deadline and "url" not in found and proc.poll() is None:
                time.sleep(0.05)
            if "url" not in found:
                self._terminate(proc)
                reason = "; ".join(tail[-3:]) or f"exited with {proc.returncode}"
                raise HmzWebError(f"hmz web did not provide a loopback URL: {reason}")
            self._proc, self._url = proc, found["url"]
            return {"available": True, "running": True, "url": self._url, "reason": ""}

    def stop(self) -> dict:
        """@brief 关闭 sidecar；重复调用安全。"""
        with self._lock:
            proc, self._proc, self._url = self._proc, None, ""
        if proc is not None:
            self._terminate(proc)
        return {"available": self._executable() is not None, "running": False, "url": "", "reason": ""}

    @staticmethod
    def _terminate(proc: subprocess.Popen[str]) -> None:
        """@brief 先关 stdin 让 attached 服务自行收尾，再按进程组兜底清理。"""
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


manager = HmzWeb()
