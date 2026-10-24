"""Read-only trust/harness command line surfaces. No model calls and no GPU."""

from __future__ import annotations

import json
from pathlib import Path

from taic.cli import main


def test_trust_cli_grants_and_lists_without_network(tmp_path, capsys):
    """@brief trust grant/status/revoke 全在本地完成，输出带状态的 JSON。

    @param tmp_path pytest 提供的临时目录，作为 trust root。
    @param capsys pytest 的输出捕获器。
    """
    assert main(["trust", "grant", "apple-m4-max", "conv2d", "deepseek", "deepseek-flash", "--root", str(tmp_path), "--by", "alice"]) == 0
    granted = json.loads(capsys.readouterr().out)
    assert granted["status"] == "active" and granted["granted_by"] == "alice"
    assert main(["trust", "status", "apple-m4-max", "--root", str(tmp_path)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["model"] == "deepseek-flash"
    assert main(["trust", "revoke", "apple-m4-max", "conv2d", "deepseek", "deepseek-flash", "--root", str(tmp_path), "--by", "bob", "--reason", "pilot over"]) == 0
    revoked = json.loads(capsys.readouterr().out)
    assert revoked["status"] == "revoked"


def test_harness_status_and_disable_are_read_only_except_the_control_file(tmp_path, capsys):
    """@brief harness status 只读；disable 只写控制文件，之后的 status 能看到 disabled。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获器。
    """
    run = tmp_path / "run"
    run.mkdir()
    assert main(["harness", "status", str(run)]) == 0
    board = json.loads(capsys.readouterr().out)
    assert board["proposals"] == [] and board["control"]["disabled"] is False
    assert main(["harness", "disable", str(run), "--by", "alice", "--reason", "pilot pause"]) == 0
    assert json.loads(capsys.readouterr().out)["disabled"] is True
    assert main(["harness", "status", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["control"]["disabled"] is True
