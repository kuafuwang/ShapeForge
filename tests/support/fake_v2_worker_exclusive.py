"""v2 worker stub: take the exclusive GPU lock, then leave a result envelope."""

import json
import sys
from pathlib import Path

req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
from tcenv.gpulock import gpu_lock

with gpu_lock("fake v2 worker", exclusive=True, max_wait_s=5):
    pass
Path(req["out"]).write_text(
    json.dumps(
        {
            "schema_version": "tcenv.v2.worker.result.1",
            "report": {"status": "ok"},
            "execution": {"dispatch": []},
        }
    ),
    encoding="utf-8",
)
