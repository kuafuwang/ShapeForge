"""Run the real Mac -> SSH -> Linux VM -> Docker simulation acceptance.

This is a test composition root only. It uses the production tcenv.remote
protocol/service and the FakeNpuBackend behind it. No model, GPU or NPU is used.
The report is a diagnostic artifact, never a search result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import struct
import subprocess
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

from taic.selfcheck_bridge import SelfCheckClient, SelfCheckConfig, SelfCheckServer
from tcenv.evaluation import EvaluationError, EvaluationRequest
from tcenv.remote.protocol import FrameDecoder, HelloRequest, ProtocolError, encode_frame, read_frame, write_frame
from tcenv.remote.client import CommandTransport, RemoteError, RemoteEvaluator, RemoteEvaluatorPool
from tests.support.npu_sim.service_entry import TASK_FINGERPRINT, TASK_PACK_DIGEST, TOKEN


def _ssh(user: str, host: str, key: str, original: str) -> list[str]:
    return [
        "ssh", "-T", "-q", "-i", key,
        "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        f"{user}@{host}", original,
    ]


def _request(source: str = "kernel void demo() {}", *, request_id: str = "req-1", kind: str = "formal_eval", mode: str = "quick", session_id: str = "session-1", seed: int = 3) -> EvaluationRequest:
    return EvaluationRequest(
        task="relu", dtype="fp16", size="small", source=source,
        candidate_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        kind=kind, mode=mode, seed=seed, reps=2,
        run_id="run-1", session_id=session_id, target_id="sim-npu-test",
        task_fingerprint=TASK_FINGERPRINT, cache_policy="disabled",
        candidate_timeout_s=2, queue_timeout_s=2,
        request_id=request_id,
    )


def _run(command: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)




class _ThrottledReader:
    def __init__(self, inner, delay_s: float) -> None:
        self.inner = inner
        self.delay_s = delay_s

    def read(self, size: int) -> bytes:
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.inner.read(size)

    def read1(self, size: int) -> bytes:
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.inner.read1(size)


class _ThrottledTransport:
    """Transport fault injector; real service/SSH bytes are still used."""

    def __init__(self, inner, delay_s: float) -> None:
        self.inner = inner
        self.delay_s = delay_s
        self.reader = _ThrottledReader(inner.reader, delay_s)
        self.writer = inner.writer

    def start(self) -> None:
        self.inner.start()
        self.reader.inner = self.inner.reader
        self.writer = self.inner.writer

    def close(self) -> None:
        self.inner.close()


def _hello(transport, nonce: str, token: str, pending: list | None = None) -> None:
    hello = HelloRequest(
        protocol_version="1", client_nonce=nonce, session_token=token,
        expected_site_id="sim-site", expected_worker_build="sim-build-1",
        expected_task_pack_digest=TASK_PACK_DIGEST, target_id="sim-npu-test",
    )
    write_frame(transport.writer, "hello", nonce, hello.__dict__)
    while True:
        frame = read_frame(transport.reader)
        if frame.frame_type == "hello":
            return
        if pending is not None and frame.frame_type == "result":
            pending.append(frame.request_id)
            continue
        raise AssertionError(f"hello failed: {frame.frame_type} {frame.payload}")


def _run_frame_limit_matrix(transport_factory, record, check, tag: str) -> None:
    evaluator = RemoteEvaluator(transport_factory(("run-1", "session-1", "sim-npu-test")), token=TOKEN, run_id="run-1", session_id="session-1", target_id="sim-npu-test", expected_site_id="sim-site", expected_worker_build="sim-build-1", expected_task_pack_digest=TASK_PACK_DIGEST, task_fingerprint=TASK_FINGERPRINT, timeout_s=8)
    try:
        near_source = "x" * 999_000
        near = evaluator.formal(_request(near_source, request_id=f"{tag}-limit-near"))
        check("frame_near_source_limit", near.report.get("simulated") is True)
        unicode_source = "🚀" * 250_000  # exactly 1,000,000 UTF-8 bytes
        unicode_result = evaluator.formal(_request(unicode_source, request_id=f"{tag}-limit-unicode-exact"))
        check("frame_unicode_exact_byte_limit", unicode_result.report.get("simulated") is True)
    finally:
        evaluator.close()
    for name, source in (("source_over_limit", "x" * 1_000_001), ("unicode_over_limit", "🚀" * 250_001)):
        try:
            _request(source, request_id=f"{tag}-{name}")
        except EvaluationError:
            record(name)
        else:
            raise AssertionError(f"{name} was accepted")
    try:
        encode_frame("request", "x", {"data": "x" * 2_000_001})
    except ProtocolError:
        record("frame_payload_over_limit")
    else:
        raise AssertionError("oversize frame payload was accepted")
    for name, raw in (
        ("frame_bad_header_length", struct.pack(">I", 1) + b"x"),
        ("frame_bad_payload_digest", encode_frame("request", "x", {"ok": True})[:-1] + b"0"),
    ):
        try:
            FrameDecoder().feed(raw)
        except ProtocolError:
            record(name)
        else:
            raise AssertionError(f"{name} was accepted")
    base = _request(request_id=f"{tag}-limits-base")
    for name, changes in (("reps_over_limit", {"reps": 10_001}), ("candidate_timeout_over_limit", {"candidate_timeout_s": 3601}), ("request_id_over_limit", {"request_id": "x" * 161})):
        try:
            replace(base, **changes)
        except EvaluationError:
            record(name)
        else:
            raise AssertionError(f"{name} was accepted")


def _run_half_open_and_reset_matrix(transport_factory, make_pool, admin, record, check, tag: str) -> None:
    # Slow consumer: the real remote result still arrives after throttled reads.
    inner = transport_factory(("run-1", "session-1", "sim-npu-test"))
    throttled = _ThrottledTransport(inner, 0.03)
    slow = RemoteEvaluator(throttled, token=TOKEN, run_id="run-1", session_id="session-1", target_id="sim-npu-test", expected_site_id="sim-site", expected_worker_build="sim-build-1", expected_task_pack_digest=TASK_PACK_DIGEST, task_fingerprint=TASK_FINGERPRINT, timeout_s=3)
    try:
        result = slow.formal(_request(request_id=f"{tag}-slow-consumer"))
        check("slow_consumer", result.report.get("simulated") is True)
    finally:
        slow.close()

    # Half-open: hello succeeds, then the reader is stalled beyond the deadline.
    inner = transport_factory(("run-1", "session-1", "sim-npu-test"))
    throttled = _ThrottledTransport(inner, 0.0)
    half = RemoteEvaluator(throttled, token=TOKEN, run_id="run-1", session_id="session-1", target_id="sim-npu-test", expected_site_id="sim-site", expected_worker_build="sim-build-1", expected_task_pack_digest=TASK_PACK_DIGEST, task_fingerprint=TASK_FINGERPRINT, timeout_s=0.05)
    try:
        half.connect()
        throttled.delay_s = 0.2
        half.formal(_request(request_id=f"{tag}-half-open"))
    except RemoteError as exc:
        check("half_open_timeout", exc.code in {"transport_timeout", "transport_closed"}, exc.code)
    else:
        raise AssertionError("half-open request unexpectedly succeeded")
    finally:
        half.close()

    # Reset the actual service container while a job is in flight, then recover
    # through a fresh SSH connection and verify a new request can complete.
    reset = RemoteEvaluator(transport_factory(("run-1", "session-1", "sim-npu-test")), token=TOKEN, run_id="run-1", session_id="session-1", target_id="sim-npu-test", expected_site_id="sim-site", expected_worker_build="sim-build-1", expected_task_pack_digest=TASK_PACK_DIGEST, task_fingerprint=TASK_FINGERPRINT, timeout_s=5)
    try:
        reset.connect()
        import threading
        killer = threading.Thread(target=lambda: (time.sleep(0.2), admin("docker ps -q --filter name=taic-npu-sim | xargs -r docker rm -f")), daemon=True)
        killer.start()
        try:
            reset.formal(_request("REMOTE_SLEEP", request_id=f"{tag}-reset-1"))
        except RemoteError as exc:
            check("container_reset_error", exc.code in {"transport_closed", "transport_protocol_error", "worker_failed", "candidate_timeout", "transport_timeout"}, exc.code)
        else:
            raise AssertionError("request unexpectedly survived evaluator container reset")
        finally:
            killer.join(timeout=5)
    finally:
        reset.close()
    after_reset = admin("cat /home/eval/taic-state/counter.txt")
    check("reset_recovery", after_reset.returncode == 0)
    recovered_pool = make_pool()
    try:
        recovered = recovered_pool.formal(_request(request_id=f"{tag}-after-reset"))
        check("reset_recovery", recovered.report.get("simulated") is True)
    finally:
        recovered_pool.close()
    deadline = time.time() + 8
    running = admin("docker ps --filter name=taic-npu-sim --format '{{.Names}}'")
    while running.stdout.strip() and time.time() < deadline:
        time.sleep(0.25)
        running = admin("docker ps --filter name=taic-npu-sim --format '{{.Names}}'")
    check("reset_no_orphan", not running.stdout.strip())


def _run_fairness_matrix(transport_factory, admin, record, check, tag: str) -> None:
    transport = transport_factory(("run-1", "session-1", "sim-npu-test", 0.2))
    a1, b1, a2 = (f"{tag}-fair-a1", f"{tag}-fair-b1", f"{tag}-fair-a2")
    try:
        transport.start()
        _hello(transport, "fair-a-hello", TOKEN)
        first = _request("REMOTE_SLEEP_FAIR", request_id=a1, mode="full", session_id="session-1")
        write_frame(transport.writer, "request", first.request_id, first.to_dict())
        arrivals = []
        # Submit B while A is still running, then queue A2 behind B. Without
        # aging, A2 (formal) would jump ahead of B (selfcheck).
        _hello(transport, "fair-b-hello", "test-alt-token", arrivals)
        second_source = "kernel void demo() {}"
        second = replace(first, request_id=b1, kind="selfcheck", mode="quick", run_id="run-2", source=second_source, candidate_sha256=hashlib.sha256(second_source.encode("utf-8")).hexdigest())
        write_frame(transport.writer, "request", second.request_id, second.to_dict())
        _hello(transport, "fair-a2-hello", TOKEN, arrivals)
        third = replace(first, request_id=a2)
        write_frame(transport.writer, "request", third.request_id, third.to_dict())
        for _ in range(3 - len(arrivals)):
            frame = read_frame(transport.reader)
            if frame.frame_type == "error":
                raise AssertionError(frame.payload)
            if frame.frame_type != "result":
                raise AssertionError(f"unexpected fairness frame {frame.frame_type}")
            arrivals.append(frame.request_id)
        check("fairness_aging_order", arrivals == [a1, b1, a2], str(arrivals))
        order = admin("cat /home/eval/taic-state/order.jsonl")
        check("fairness_order_evidence", order.stdout.index(b1) < order.stdout.index(a2))
    finally:
        transport.close()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.npu_sim.phase25_runner")
    parser.add_argument("--host", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--eval-user", default="eval")
    parser.add_argument("--admin-user", default="kuafu")
    parser.add_argument("--report", default="runs/npu-sim-20261024/report.json")
    args = parser.parse_args(argv)
    results: list[dict] = []
    timings: list[float] = []
    started = time.time()
    pool: RemoteEvaluatorPool | None = None
    bridge: SelfCheckServer | None = None
    failure = ""

    def record(name: str, status: str = "passed", detail: str = "") -> None:
        results.append({"name": name, "status": status, "detail": detail})

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            raise AssertionError(detail or name)
        record(name, detail=detail)

    def transport_factory(key):
        run_id, session_id, _target, *optional = key
        original = f"/usr/local/bin/taic-remote-evaluator --run-id {run_id} --session-id {session_id}"
        if optional:
            original += f" --aging-s {optional[0]}"
        return CommandTransport(_ssh(args.eval_user, args.host, args.key, original), env={})

    def admin(command: str, timeout: int = 30) -> subprocess.CompletedProcess:
        return _run(_ssh(args.admin_user, args.host, args.key, command), timeout=timeout)

    def counter() -> int:
        completed = admin("cat /home/eval/taic-state/counter.txt 2>/dev/null || echo 0")
        try:
            return int(completed.stdout.strip() or "0")
        except ValueError:
            return -1

    def timed_call(label: str, call):
        before = time.perf_counter()
        value = call()
        timings.append(time.perf_counter() - before)
        return value

    try:
        baseline = counter()
        if baseline < 0:
            raise RuntimeError("cannot read the VM counter")
        def make_pool():
            return RemoteEvaluatorPool(
                transport_factory,
                token=TOKEN,
                run_id="run-1",
                target_id="sim-npu-test",
                expected_site_id="sim-site",
                expected_worker_build="sim-build-1",
                expected_task_pack_digest=TASK_PACK_DIGEST,
                task_fingerprint=TASK_FINGERPRINT,
                timeout_s=8,
            )
        pool = make_pool()
        run_tag = "p" + uuid.uuid4().hex[:12]
        request = _request(request_id=f"{run_tag}-req-1")
        first = timed_call("formal", lambda: pool.formal(request))
        check("formal_hello_report", first.report.get("simulated") is True)
        check("formal_first_job_count", counter() == baseline + 1, f"counter={counter()}")
        replay = timed_call("idempotent_replay", lambda: pool.formal(request))
        check("idempotent_replay", replay.report.get("simulated") is True)
        check("idempotent_no_second_job", counter() == baseline + 1, f"counter={counter()}")

        second = timed_call("second_request_id", lambda: pool.formal(replace(request, request_id=f"{run_tag}-req-2")))
        check("second_request_id", second.report.get("simulated") is True)
        check("second_request_id_counts_once", counter() == baseline + 2, f"counter={counter()}")

        try:
            pool.formal(_request(request.source + "\n// changed", request_id=f"{run_tag}-req-2"))
        except RemoteError as exc:
            check("same_id_different_body_rejected", exc.code == "idempotency_conflict", exc.code)
        else:
            raise AssertionError("same request id with a different body was accepted")
        check("conflict_does_not_run_job", counter() == baseline + 2, f"counter={counter()}")

        selfcheck = timed_call("selfcheck_timed", lambda: pool.selfcheck(replace(request, request_id=f"{run_tag}-self-1", kind="selfcheck", mode="timed")))
        check("selfcheck_timed_simulated", selfcheck.report.get("simulated") is True)
        check("selfcheck_timed_count", counter() == baseline + 3, f"counter={counter()}")

        with tempfile.TemporaryDirectory(prefix="phase25-bridge-") as temporary:
            work = Path(temporary)
            config = SelfCheckConfig(
                task="relu", dtype="fp16", size="small", target_id="sim-npu-test",
                task_fingerprint=TASK_FINGERPRINT, run_id="run-1", session_id="session-1",
                token=uuid.uuid4().hex, token_file=work / ".selfcheck" / "token",
                self_check_dir=work / ".selfcheck", usage_path=work / "usage.json",
            )
            bridge = SelfCheckServer(config, pool, Path("/tmp") / f"taic-phase25-{uuid.uuid4().hex}.sock")
            bridge.start()
            checked = timed_call("host_bridge_selfcheck", lambda: SelfCheckClient(bridge.socket_path, token_file=config.token_file).check("kernel void demo() {}", "quick"))
            check("host_bridge_selfcheck", checked.report.get("simulated") is True)
            check("host_bridge_count", counter() == baseline + 4, f"counter={counter()}")
            bridge.stop()
            bridge = None
            check("bridge_token_removed", not config.token_file.exists())

        unicode_result = timed_call("unicode", lambda: pool.formal(_request("// π 🚀\nkernel void demo() {}", request_id=f"{run_tag}-unicode-1")))
        check("unicode_frame", unicode_result.report.get("simulated") is True)
        large_source = "// " + ("x" * 100_000) + "\nkernel void demo() {}"
        large_result = timed_call("large_frame", lambda: pool.formal(_request(large_source, request_id=f"{run_tag}-large-1")))
        check("large_frame", large_result.report.get("simulated") is True)
        check("unicode_large_counts", counter() == baseline + 6, f"counter={counter()}")

        try:
            wrong = RemoteEvaluator(transport_factory(("run-1", "session-1", "sim-npu-test")), token=TOKEN, run_id="run-1", session_id="session-1", target_id="sim-npu-test", expected_worker_build="wrong-build", timeout_s=3)
            wrong.connect()
        except RemoteError as exc:
            check("wrong_build_rejected", "build" in str(exc), str(exc))
        else:
            raise AssertionError("wrong worker build was accepted")

        canary = "/home/eval/phase25-canary"
        admin(f"rm -f {canary}")
        injected = _run(_ssh(args.eval_user, args.host, args.key, f"/bin/touch {canary}"))
        check("shell_injection_rejected", injected.returncode != 0 and not admin(f"test -e {canary}").returncode == 0)
        docker_injection = _run(_ssh(args.eval_user, args.host, args.key, "docker exec taic-npu-sim id"))
        check("docker_exec_rejected", docker_injection.returncode != 0)

        slow = RemoteEvaluator(transport_factory(("run-1", "session-1", "sim-npu-test")), token=TOKEN, run_id="run-1", session_id="session-1", target_id="sim-npu-test", timeout_s=0.2)
        try:
            slow.formal(_request("REMOTE_SLEEP", request_id=f"{run_tag}-timeout-1"))
        except RemoteError as exc:
            check("transport_timeout", exc.code in {"transport_timeout", "candidate_timeout"}, exc.code)
        finally:
            slow.close()
        # Other pool sessions intentionally stay alive for idempotency. Close
        # them before asserting that the timed-out connection left no orphan.
        pool.close()
        pool = None
        time.sleep(1)
        deadline = time.time() + 8
        running = admin("docker ps --filter name=taic-npu-sim --format '{{.Names}}'")
        while running.stdout.strip() and time.time() < deadline:
            time.sleep(0.25)
            running = admin("docker ps --filter name=taic-npu-sim --format '{{.Names}}'")
        check("timeout_no_orphan", not running.stdout.strip(), running.stdout.strip())
        after_timeout_count = counter()

        admin("docker ps -q --filter name=taic-npu-sim | xargs -r docker rm -f")
        pool = make_pool()
        restarted = timed_call("restart_idempotent_replay", lambda: pool.formal(request))
        check("restart_idempotent_replay", restarted.report.get("simulated") is True)
        check("restart_no_duplicate_job", counter() == after_timeout_count, f"counter={counter()}")
        pool.close()
        pool = None
        _run_frame_limit_matrix(transport_factory, record, check, run_tag)
        _run_half_open_and_reset_matrix(transport_factory, make_pool, admin, record, check, run_tag)
        _run_fairness_matrix(transport_factory, admin, record, check, run_tag)
    except Exception as exc:  # noqa: BLE001 - report every failure before returning nonzero
        failure = f"{type(exc).__name__}: {exc}"
        record("failure", "failed", failure)
    finally:
        if bridge is not None:
            bridge.stop()
        if pool is not None:
            pool.close()
        running = admin("docker ps --filter name=taic-npu-sim --format '{{.Names}}'")
        image = admin("docker image inspect --format '{{.Id}}' taic-remote-sim:local")
        report = {
            "schema_version": 1,
            "coverage": "extended_split_control",
            "simulated": True,
            "container_only": False,
            "split_control": not failure,
            "publication_allowed": False,
            "status": "failed" if failure else "passed",
            "base_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "environment": {
                "host": "macOS/arm64",
                "transport": "SSH forced command",
                "vm": "Linux Ubuntu 22.04 ARM64",
                "docker_image": "taic-remote-sim:local",
                "docker_image_id": image.stdout.strip(),
                "docker_platform": "Ubuntu Docker Engine/linux/arm64",
                "network": "none for evaluator containers",
                "read_only_root": True,
            },
            "results": results,
            "counter_final": counter(),
            "control_plane_timings_s": {
                "count": len(timings),
                "p50": statistics.median(timings) if timings else None,
                "p95": max(timings) if timings else None,
                "note": "control-plane only; not candidate_us or a speedup",
            },
            "elapsed_s": round(time.time() - started, 3),
            "remaining_containers": running.stdout.strip(),
            "failure": failure,
            "not_run": ["real_npu", "real_gpu", "paid_model", "npu_sdk_installation"],
        }
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "report": str(path), "results": len(results), "failure": failure}, ensure_ascii=False))
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
