"""Fake v2 worker: run the real build/execution, then publish a *self-consistent* drifted build.

A4 negative: the worker tampers its native descriptor's frozen-geometry audit and recomputes
``native_digest``, so the build record is internally valid (``validate_shape`` accepts it and the
report matches it) while its identity no longer describes the control host's canonical geometry.
Only the control host's cross-audit against its own canonical bundle can catch this, so this
worker proves the parent does not silently trust worker-produced build evidence.
"""
import json
import sys
from pathlib import Path

from tcenv import v2_worker
from tcenv.remote.artifact_bundle import NativeExecutable

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
v2_worker._install_heartbeat(str(request["heartbeat"]), str(request["target_id"]))
result = v2_worker.evaluate_payload(request["payload"])
build = dict(result["build"])
native = dict(build["native"])
descriptor = json.loads(json.dumps(native["descriptor"]))
descriptor["execution_geometry"]["sha256"] = "0" * 64
forged = NativeExecutable(
    backend_id=native["backend_id"],
    kind=native["kind"],
    entry_path=native["entry_path"],
    entry_sha256=native["entry_sha256"],
    descriptor=descriptor,
)
build["native"] = dict(forged.document())
build["native_digest"] = forged.digest
result["build"] = build
report = dict(result["report"])
report["build"] = build
result["report"] = report
Path(request["out"]).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
