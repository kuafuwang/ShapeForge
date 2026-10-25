"""Worker stand-in: immediately write a valid successful report."""

import json
import sys
from pathlib import Path

from tcenv.evaluate import EvalReport

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
report = EvalReport(
    task=request["task"],
    dtype=request["dtype"],
    size=request["size"],
    mode=request["mode"],
    seed=request["seed"],
    status="ok",
    stage="done",
)
Path(request["out"]).write_text(json.dumps(report.to_json()), encoding="utf-8")
