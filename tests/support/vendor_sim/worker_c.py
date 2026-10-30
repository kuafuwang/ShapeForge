"""Private worker entry for the P1c fake module vendor (test-only).

Two timings share this worker:

* ``prebuilt`` — the request carries the artifact set the control host built.
  The worker reads every entry (inline bytes or chunks from the engine-owned
  blob channel), re-verifies the exact bytes, and hands them to the vendor
  driver, which only loads.  No source is downloaded and nothing is compiled
  here.
* ``evaluator`` — the request carries the source and the registered build
  identity.  The worker runs the *registered* builder in this attempt, then
  loads the produced artifact set.  A typed compile failure becomes a
  ``compile_error`` candidate report with ``compile_failed`` build evidence; it
  is never retried as an infrastructure timeout and never reported as ``ok``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

from tcenv.remote import contracts, worker_protocol
from tcenv.remote.artifact_builder import (
    ArtifactBuilderRegistry,
    BuildError,
    SourceBundle,
)
from tcenv.remote.artifact_bundle import BundleError
from tcenv.remote.blob_channel import BlobReader
from tcenv.schema import PROTOCOL_VERSION

from .module_builder import builder_for_provenance
from .vendor_c import FakeModuleVendorDriver, received_build_evidence

BLOB_ROOT_ENV = "TCENV_BLOB_ROOT"
STATE_DIR_ENV = "TCENV_REMOTE_STATE_DIR"
#: Test fault: die right after a successful build, before anything is loaded.
ABORT_AFTER_BUILD_ENV = "TCENV_VENDOR_C_ABORT_AFTER_BUILD"
#: Test fault: never make progress, so the attempt deadline expires before a
#: build ever starts (the timeout evidence must say ``not_run``, not ``ok``).
SLOW_ENV = "TCENV_VENDOR_C_SLOW"


def _artifact_bytes(
    bundle, root: str | None, *, scratch: Path | None = None,
    scratch_anchor: Path | None = None,
) -> dict[str, bytes]:
    """Read every entry and verify the exact bytes before the driver loads them.

    Inline entries are materialized into this attempt's build scratch directory
    and read back through the verified handles (a path swapped after the write is
    refused).  ``scratch_anchor`` is the engine-owned state directory: the
    ``build`` and attempt components are checked from it with no-follow, so a
    planted ``<state>/build`` link cannot redirect the write.  The scratch
    directory is a real temporary build artifact and the attempt removes it after
    loading; a crash leaves it for the engine's sweep.
    """

    handles = (
        bundle.materialize(scratch, anchor=scratch_anchor)
        if scratch is not None else {}
    )
    reader = BlobReader(Path(root)) if bundle.blob_entries else None
    written: dict[str, bytes] = {}
    for entry in bundle.entries:
        if entry.data is not None:
            data = (
                handles[entry.path].read_bytes() if entry.path in handles
                else bytes(entry.data)
            )
        else:
            if reader is None:
                raise RuntimeError("the prebuilt artifact set needs the blob channel")
            if not root:
                raise RuntimeError("no engine-owned blob root was provided")
            collected = bytearray()
            for chunk in reader.iter_chunks(entry.blob_ref["content_id"]):
                collected += chunk
                if len(collected) > entry.size_bytes:
                    raise RuntimeError("blob-backed entry is larger than its manifest")
            data = bytes(collected)
        if len(data) != entry.size_bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
            raise RuntimeError(f"artifact {entry.path!r} does not match its manifest")
        written[entry.path] = data
    return written


def _compile_failure_report(
    request, attempt_id: str, fence: int, exc: BuildError, *, identity: dict[str, str],
) -> int:
    """Emit the candidate compile failure (no device event, no claim of ok)."""

    build = request.build
    assert build is not None
    evidence = {
        "schema_id": contracts.NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID,
        "candidate_digest": request.npu.candidate_digest.to_dict(),
        "artifact_set_digest": None,
        "device_group": request.npu.execution.device_group.to_dict(),
        "topology": {
            "logical_devices": 1,
            "declared_devices": ["device-0"],
            "required_rank_spans": None,
            "required_die_spans": None,
        },
        "spans": [],
        "timing": {
            "source": "DEVICE_EVENT",
            "clock_domain": contracts.NPU_UNOBSERVED_CLOCK_DOMAIN,
            "synchronization_completed": False,
            "device_elapsed_ns": 0,
            "host_api_elapsed_ns": max(1, exc.duration_ms * 1_000_000),
            "launches": 0,
        },
        "buffer": None,
        "build": {
            "schema_id": contracts.NPU_BUILD_EVIDENCE_SCHEMA_ID,
            "mode": "evaluator",
            "status": "compile_failed",
            "error_class": "compile",
            "provenance": build.provenance_document(),
            "artifacts": None,
            "duration_ms": max(1, exc.duration_ms),
            "log": (
                None if not exc.log
                else {
                    "sha256": hashlib.sha256(exc.log).hexdigest(),
                    "size_bytes": len(exc.log),
                }
            ),
            "loaded": [],
        },
        "measurement_kind": "simulated",
        "publication_allowed": False,
        "complete": False,
        "diagnostics": {"build_error": exc.detail[:300], "compile_in_attempt": True},
    }
    try:
        contracts.npu_evidence_document(
            evidence,
            source_bytes=request.source,
            device_group=request.npu.execution.device_group.to_dict(),
            expected_resources=request.npu.execution.declared_resources.to_dict(),
            expected_build=build.to_dict(),
            request_artifact=request.artifact_document(),
            measurement_kind="simulated", publication_allowed=False,
            report_status="compile_error", dialect=contracts.NPU_BUILD_DIALECT,
        )
    except contracts.ContractError:
        return 26
    from tcenv.build_projection import project_partial
    from tcenv.evaluate import EvalReport

    try:
        failure_builder, _ = builder_for_provenance(build.to_dict()["provenance"])
    except BuildError:
        return 27
    try:
        build_record = project_partial(
            exc, build_kind=failure_builder.build_kind, identity=identity,
        )
    except ValueError:
        return 28
    report_object = EvalReport(
        task=request.task, dtype=request.dtype, size=request.size,
        mode=request.mode, seed=request.seed_int, status="compile_error",
        stage="compile", message=f"fake txcc: {exc.detail}"[:300],
        candidate_us=0.0, speedup=1.0, elapsed_s=0.0,
        target_id=request.target_id, protocol_version=PROTOCOL_VERSION,
    )
    report_object.build = build_record
    report = report_object.to_json()
    worker_protocol.write_result(
        sys.stdout.buffer, request_id=request.request_id, attempt_id=attempt_id,
        lease_fence=fence, report=report, evidence=evidence,
    )
    return 0


def main() -> int:
    if os.environ.get(SLOW_ENV) == "1":
        # The engine owns the candidate deadline; this worker simply never
        # produces a build, so the frontend writes the diagnostic timeout row.
        time.sleep(30)
    attempt_id, request, request_id, fence = worker_protocol.read_request(sys.stdin.buffer)
    if request.npu is None or request.build is None:
        return 12
    build_document = request.build.to_dict()
    from tests.support.report_build import task_identity

    identity = task_identity(request.task, request.dtype, request.size)
    source = request.source
    context = {
        "attempt_id": attempt_id,
        "lease_fence": fence,
        "device_group": request.npu.execution.device_group.to_dict(),
        "candidate_digest": request.npu.candidate_digest.to_dict(),
        "build": build_document,
        "source": source,
    }
    blob_root = os.environ.get(BLOB_ROOT_ENV)
    state_dir = os.environ.get(STATE_DIR_ENV)
    scratch = (
        Path(state_dir) / "build" / attempt_id if state_dir else None
    )
    scratch_anchor = Path(state_dir) if state_dir else None
    deadline = time.monotonic() + request.candidate_timeout_ms / 1000.0
    if request.npu.bundle is not None:
        # Prebuilt: the control host already compiled; this attempt only verifies
        # and loads.  The source branch is empty by contract.
        bundle = request.npu.bundle
        load_started = time.monotonic()
        try:
            loaded = _artifact_bytes(
                bundle, blob_root, scratch=scratch, scratch_anchor=scratch_anchor,
            )
        except Exception:  # noqa: BLE001 - an unreadable artifact is infrastructure
            return 17
        load_ms = max(1, int(round((time.monotonic() - load_started) * 1000)))
        module = next(
            entry for entry in bundle.entries if entry.role == "EXECUTABLE_MODULE"
        )
        try:
            evidence = received_build_evidence(
                bundle=bundle, provenance=build_document["provenance"],
                producer=build_document["producer"], duration_ms=load_ms,
                log=(
                    f"npu: verified {len(loaded)} artifact(s) for device load; "
                    f"{module.path} sha256={module.sha256} in {load_ms}ms\n"
                ).encode("ascii"),
            )
        except (KeyError, ValueError) as exc:  # noqa: BLE001 - inconsistent request
            return 18
        try:
            received_builder, _ = builder_for_provenance(build_document["provenance"])
        except BuildError:
            return 25
        build_kind = received_builder.build_kind
        build_outcome = None
    else:
        # Evaluator build: only a registered profile can run, and the build is
        # part of this attempt (no prebuilt artifact is mixed in).
        try:
            builder, _ = builder_for_provenance(build_document["provenance"])
            registry = ArtifactBuilderRegistry([builder])
            outcome = registry.build(
                source_bundle=SourceBundle(source),
                task_contract={
                    "task_family": request.npu.execution.task_family,
                    "media_type": request.npu.execution.source_media_type,
                },
                provenance=build_document["provenance"],
                deadline_monotonic=deadline,
            )
        except BuildError as exc:
            if exc.error_class == "compile":
                return _compile_failure_report(
                    request, attempt_id, fence, exc, identity=identity,
                )
            # A toolchain/profile/deadline refusal is not a candidate compile
            # failure: no result is reported and the frontend treats it as an
            # infrastructure/attempt failure.
            return 24
        bundle = outcome.bundle
        evidence = outcome.evidence
        build_outcome = outcome  # the BuildOutcome, before ``outcome`` names the device run
        build_kind = builder.build_kind
        try:
            loaded = _artifact_bytes(
            bundle, None, scratch=scratch, scratch_anchor=scratch_anchor,
        )
        except Exception:  # noqa: BLE001 - a freshly built set must verify
            return 19
        if os.environ.get(ABORT_AFTER_BUILD_ENV) == "1" and fence == 1:
            # A worker killed after its build (and after materializing the
            # artifact set for verification) but before any device load: the
            # build scratch stays behind and no artifact/evidence is ever
            # claimed for this attempt.
            os._exit(9)
    driver = FakeModuleVendorDriver()
    try:
        prepared = driver.prepare(
            request.npu.execution, bundle,
            {**context, "loaded_artifacts": loaded, "build_evidence": evidence},
        )
    except (ValueError, BundleError):
        return 14

    def progress(payload) -> None:
        worker_protocol.write_progress(
            sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
            lease_fence=fence, progress=payload,
        )

    try:
        outcome = driver.execute(
            prepared, deadline_monotonic=deadline, progress=progress,
        )
    except Exception:  # noqa: BLE001 - a typed device failure is infrastructure
        driver.close(prepared)
        return 21
    driver.close(prepared)
    if scratch is not None and scratch.exists():
        try:
            shutil.rmtree(scratch)
        except OSError:
            # The build scratch could not be removed: report no result rather
            # than claiming a candidate ran with an unverifiable leftover.
            return 23
        try:
            # The per-attempt directory is this attempt's whole build scratch; an
            # empty ``build`` parent left behind would still be temporary state
            # the job lifecycle claims to have removed.
            scratch.parent.rmdir()
        except OSError:
            # Another attempt still has scratch there: that directory is not ours.
            pass
    device_evidence = dict(outcome.evidence)
    if device_evidence.get("candidate_digest") != request.npu.candidate_digest.to_dict():
        return 14
    timing = device_evidence.get("timing") or {}
    if timing.get("source") != "DEVICE_EVENT":
        return 15
    device_elapsed_ns = int(timing["device_elapsed_ns"])
    # Self-check with the same production validator the frontend and the host
    # outbox use: a build evidence that does not close over the accepted build
    # identity never leaves the worker.
    try:
        contracts.npu_evidence_document(
            device_evidence,
            source_bytes=(source if request.npu.bundle is None else None),
            device_group=request.npu.execution.device_group.to_dict(),
            expected_artifact_set_digest=(
                request.npu.candidate_digest.value
                if request.npu.candidate_digest.algorithm == contracts.BUNDLE_DIGEST_ALGORITHM
                else None
            ),
            expected_resources=request.npu.execution.declared_resources.to_dict(),
            expected_build=build_document,
            request_artifact=request.artifact_document(),
            measurement_kind="simulated", publication_allowed=False,
            report_status="ok", dialect=contracts.NPU_BUILD_DIALECT,
        )
    except contracts.ContractError:
        return 22
    from tcenv.build_projection import project_full, project_received
    from tcenv.evaluate import EvalReport

    if request.npu.bundle is not None:
        # This attempt verified and loaded a prebuilt set; it holds no BuildOutcome.
        build_record = project_received(evidence, build_kind=build_kind, identity=identity)
    else:
        build_record = project_full(build_outcome, build_kind=build_kind, identity=identity)
    report_object = EvalReport(
        task=request.task, dtype=request.dtype, size=request.size,
        mode=request.mode, seed=request.seed_int,
        status="ok", stage="done",
        candidate_us=device_elapsed_ns / 1000.0,
        speedup=1.0, elapsed_s=0.0,
        target_id=request.target_id, protocol_version=PROTOCOL_VERSION,
    )
    report_object.build = build_record
    report = report_object.to_json()
    worker_protocol.write_result(
        sys.stdout.buffer, request_id=request_id, attempt_id=attempt_id,
        lease_fence=fence, report=report, evidence=device_evidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
