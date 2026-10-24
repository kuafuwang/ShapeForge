"""Tests for the explicit remote composition, using the real service and fake device."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from taic.remote_runtime import RemoteEvaluationConfig
from taic.selfcheck_bridge import SelfCheckClient
from tcenv.evaluation import EvaluationRequest
from tests.support.npu_sim.service_entry import TASK_FINGERPRINT, TASK_PACK_DIGEST, TOKEN

ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path) -> RemoteEvaluationConfig:
    token_file = tmp_path / "grant.token"
    token_file.write_text(TOKEN)
    command = [
        sys.executable, "-m", "tests.support.npu_sim.service_entry",
        "--state-dir", str(tmp_path / "state" / "{session_id}"),
        "--run-id", "{run_id}", "--session-id", "{session_id}", "--token", TOKEN,
    ]
    path = tmp_path / "remote.json"
    path.write_text(json.dumps({
        "command": command,
        "token_file": str(token_file),
        "target_id": "sim-npu-test",
        "expected_site_id": "sim-site",
        "expected_worker_build": "sim-build-1",
        "expected_task_pack_digest": TASK_PACK_DIGEST,
        "task_fingerprint": TASK_FINGERPRINT,
        "publishable_site_ids": [],
        "socket_root": "/tmp/taic-remote-composition",
        "timeout_s": 2,
    }))
    return RemoteEvaluationConfig.from_file(path)


def _request() -> EvaluationRequest:
    source = "kernel void demo() {}"
    return EvaluationRequest(
        task="relu", dtype="fp16", size="small", source=source,
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
        kind="formal_eval", mode="quick", seed=3, reps=2,
        run_id="run-1", session_id="session-1", target_id="sim-npu-test",
        task_fingerprint=TASK_FINGERPRINT, cache_policy="disabled",
        candidate_timeout_s=2, queue_timeout_s=2,
    )


def test_explicit_remote_config_builds_pool_and_per_session_bridge(tmp_path):
    """@brief 配置文件创建真实 pool 和 host bridge，自测保留 simulated 标记并清理 token。"""
    config = _config(tmp_path)
    evaluator = config.make_evaluator("run-1")
    try:
        result = evaluator.formal(_request())
        assert result.report["simulated"] is True
        factory = config.make_bridge_factory(task="relu", dtype="fp16", size="small", task_fingerprint=TASK_FINGERPRINT, run_id="run-1")
        work = tmp_path / "work"
        work.mkdir()
        server, token_file = factory("session-1", work, evaluator)
        assert not server._usage_path.is_relative_to(work)
        try:
            checked = SelfCheckClient(server.socket_path, token_file=token_file).check("kernel void demo() {}", "quick")
            assert checked.report["simulated"] is True
            # The credential lives under the protected .selfcheck directory, so
            # a session prune while the bridge is alive cannot archive it.
            from taic.kernel_agent import prune_work
            (work / "scratch.txt").write_text("x")
            pruned = prune_work(work, label="init", keep={"candidate.metal"})
            assert ".selfcheck" in pruned["kept"]
            assert "token" not in pruned.get("archived", [])
            assert Path(token_file).is_file()
        finally:
            server.stop()
        assert not Path(token_file).exists()
    finally:
        evaluator.close()


def test_remote_config_rejects_unknown_fields(tmp_path):
    """@brief 配置中的未知字段必须在创建连接前拒绝。"""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"command": ["echo"], "token_file": "token", "target_id": "sim-npu-test", "shell": True}))
    try:
        RemoteEvaluationConfig.from_file(path)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown remote config field was accepted")
