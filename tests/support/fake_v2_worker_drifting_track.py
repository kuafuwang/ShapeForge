"""Fake v2 worker: run the real build/execution, then publish a drifted Track selection.

A4 negative: the control host froze one TrackSpec/registry/rendering identity in the request.  A
worker that reports a different Track version/digest must be refused even though the build and
execution themselves succeeded.
"""
import json
import sys
from pathlib import Path

from tcenv import v2_worker

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
v2_worker._install_heartbeat(str(request["heartbeat"]), str(request["target_id"]))
result = v2_worker.evaluate_payload(request["payload"])
track = dict(result["track"])
track["track_version"] = int(track["track_version"]) + 1
track["track_digest"] = "0" * 64
result["track"] = track
Path(request["out"]).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
