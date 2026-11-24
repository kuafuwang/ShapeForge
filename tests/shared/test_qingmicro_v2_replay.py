"""Current-code replay of the v2 Qingmicro simulation, bound to a fresh evidence epoch.

The previously registered package report stays frozen.  This test never rewrites it and never
uses a real NPU; it executes the local fake service and compares the complete current report with
the newly recorded evidence document.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/可信harness/qingmicro-v2-replay-20261124-remove-v1.json"
SIMULATOR = ROOT / "docs/架构/清微NPU数据模拟-v2/simulate.py"


def test_current_v2_qingmicro_replay_matches_the_new_epoch():
    document = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert document["schema_id"] == "local.qingmicro-npu-data-simulation-replay.1"
    assert document["epoch"] == "20261124-remove-v1-task-system"
    assert document["simulated"] is True and document["publication_allowed"] is False
    assert document["npu_p0_status"] == "blocked"

    spec = importlib.util.spec_from_file_location("qingmicro_v2_current_replay_test", SIMULATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = module.simulate()
    assert current == document["report"]
    assert current["remote"]["protocol_task"]["spec_version"] == 2
    assert current["remote"]["timing_numbers_included"] is False
    assert current["bundle"]["one_byte_tamper_rejected"] is True
