"""Test-only TCP fault injection for the gRPC/TLS acceptance path.

``reset`` closes the first client connection after a small byte budget (and can
then pass traffic through), ``half_open`` accepts without forwarding, and
``slow`` delays both directions to exercise Watch reconnects and deadlines.
The proxy never inspects TLS plaintext.
"""

from __future__ import annotations

import select
import socket
import struct
import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Connection:
    client: socket.socket
    upstream: socket.socket | None = None
    stop: threading.Event = field(default_factory=threading.Event)


class FaultProxy:
    def __init__(
        self,
        target_host: str,
        target_port: int,
        *,
        mode: str = "pass",
        reset_after_bytes: int = 64,
        reset_once: bool = True,
        delay_s: float = 0.05,
    ) -> None:
        if mode not in {"pass", "reset", "half_open", "slow"}:
            raise ValueError("unknown proxy mode")
        self.target = (target_host, target_port)
        self.mode = mode
        self.reset_after_bytes = int(reset_after_bytes)
        self.reset_once = bool(reset_once)
        self.delay_s = float(delay_s)
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(16)
        self.port = self._listener.getsockname()[1]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connections: list[_Connection] = []
        self._lock = threading.Lock()
        self._reset_used = False

    @property
    def endpoint(self) -> str:
        return f"127.0.0.1:{self.port}"

    def __enter__(self) -> "FaultProxy":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._accept_loop, name="tcenv-fault-proxy", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            connection.stop.set()
            for sock in (connection.client, connection.upstream):
                if sock is None:
                    continue
                try:
                    sock.close()
                except OSError:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._listener], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not readable:
                continue
            try:
                client, _address = self._listener.accept()
            except OSError:
                return
            connection = _Connection(client)
            with self._lock:
                self._connections.append(connection)
            threading.Thread(target=self._handle, args=(connection,), daemon=True).start()

    def _handle(self, connection: _Connection) -> None:
        mode = self.mode
        if mode == "reset" and self.reset_once:
            with self._lock:
                if self._reset_used:
                    mode = "pass"
                else:
                    self._reset_used = True
        if mode == "half_open":
            while not self._stop.is_set() and not connection.stop.is_set():
                time.sleep(0.05)
            return
        try:
            upstream = socket.create_connection(self.target, timeout=5)
        except OSError:
            connection.client.close()
            return
        connection.upstream = upstream
        if mode == "reset":
            # Consume a bounded amount (normally the TLS ClientHello / first
            # gRPC bytes) and then reset instead of closing gracefully.
            try:
                connection.client.settimeout(2)
                connection.client.recv(self.reset_after_bytes)
            except OSError:
                pass
            try:
                connection.client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            except OSError:
                pass
            for sock in (connection.client, upstream):
                try:
                    sock.close()
                except OSError:
                    pass
            return
        delay = self.delay_s if mode == "slow" else 0.0
        try:
            self._pump(connection, delay)
        finally:
            for sock in (connection.client, upstream):
                try:
                    sock.close()
                except OSError:
                    pass

    def _pump(self, connection: _Connection, delay: float) -> None:
        sockets = [connection.client, connection.upstream]
        while not self._stop.is_set() and not connection.stop.is_set():
            try:
                readable, _, _ = select.select(sockets, [], [], 0.1)
            except (OSError, ValueError):
                return
            for sock in readable:
                try:
                    data = sock.recv(16_384)
                except OSError:
                    return
                if not data:
                    return
                other = connection.upstream if sock is connection.client else connection.client
                if delay:
                    time.sleep(delay)
                try:
                    other.sendall(data)
                except OSError:
                    return
