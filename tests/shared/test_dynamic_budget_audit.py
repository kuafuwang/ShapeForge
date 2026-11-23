"""Dynamic-budget analysis separates situation delivery from other-block omissions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taic.dynamic_budget_audit import SCHEMA, audit, render_summary


def _receipt(run: Path, label: str, *, static_omitted=0, other_omitted=100) -> None:
    folder = run / "sessions" / label
    folder.mkdir(parents=True, exist_ok=True)
    other_delivered = 0
    document = {
        "schema": "taic.task_delivery.1",
        "static": {"delivered_bytes": 1000, "omitted_bytes": static_omitted},
        "dynamic": {"delivered_bytes": 500, "omitted_bytes": other_omitted, "budget_bytes": 6000, "over_budget": False},
        "situation": {
            "status": "delivered", "budget_bytes": 5400, "source_bytes": 500,
            "sanitized_bytes": 500, "output_bytes": 500, "delivered_bytes": 500, "omitted_bytes": 0,
            "layers": [{"name": "peer_unverified", "source_bytes": 500, "sanitized_bytes": 500,
                        "rendered_bytes": 500, "output_bytes": 500, "delivered_bytes": 500, "omitted_bytes": 0}],
        },
        "blocks": [
            {"name": "本轮态势", "expected_bytes": 500, "delivered_bytes": 500, "omitted_bytes": 0},
            {"name": "上一轮 NOTES", "expected_bytes": other_omitted, "delivered_bytes": other_delivered,
             "omitted_bytes": other_omitted, "omission_reason": "task_budget"},
        ],
    }
    (folder / "task_delivery.json").write_text(json.dumps(document), encoding="utf-8")


def test_other_block_omissions_are_not_hidden_by_situation_delivery(tmp_path):
    runs = tmp_path / "runs"
    _receipt(runs / "target" / "run-a", "r1")
    _receipt(runs / "target" / "run-b", "init", other_omitted=0)
    document = audit(runs)
    counts = document["counts"]
    assert document["schema"] == SCHEMA
    assert document["receipts"] == 2 and document["runs"] == 2
    assert counts["static_omitted"] == 0
    assert counts["dynamic_receipt_delivered"] == 1000
    assert counts["dynamic_receipt_omitted"] == 100
    assert counts["other_block_omitted"] == 100
    assert counts["receipts_with_dynamic_omission"] == 1
    assert document["dynamic_omission_ratio"] == pytest.approx(1 / 11)
    assert document["verdict"] == "situation_delivered_but_other_blocks_were_cut"
    assert document["blocks"]["上一轮 NOTES"]["omitted_bytes"] == 100
    assert document["omission_reasons_bytes"] == {"task_budget": 100}
    assert document["hot_runs"][0]["run_id"] == "target/run-a"
    summary = render_summary(document)
    assert "上一轮 NOTES" in summary and "situation_delivered_but_other_blocks_were_cut" in summary


def test_static_contract_cut_has_its_own_verdict(tmp_path):
    runs = tmp_path / "runs"
    _receipt(runs / "run-cut", "r1", static_omitted=1)
    document = audit(runs)
    assert document["verdict"] == "static_contract_was_cut"


def test_real_receipts_are_read_only_and_balanced():
    runs = Path(__file__).resolve().parents[2] / "runs"
    if not runs.is_dir():
        pytest.skip("requires the repository's real runs/ corpus")
    files = list(runs.rglob("task_delivery.json"))
    before = {path: path.stat().st_mtime_ns for path in files}
    document = audit(runs)
    after = {path: path.stat().st_mtime_ns for path in files}
    assert after == before
    counts = document["counts"]
    assert counts["static_omitted"] == 0, "the static task contract must never be cut"
    assert counts.get("block_accounting_mismatches", 0) == 0
    assert counts["other_block_omitted"] == counts["dynamic_receipt_omitted"]
    expected = sum(row["expected_bytes"] for row in document["blocks"].values())
    assert expected == counts["dynamic_receipt_delivered"] + counts["dynamic_receipt_omitted"]
    assert document["receipts"] == len(files)
    assert counts["receipts_with_dynamic_omission"] + counts["receipts_without_dynamic_omission"] == len(files)
    assert document["situation_layers"]["peer_unverified"]["omitted_bytes"] == 0
