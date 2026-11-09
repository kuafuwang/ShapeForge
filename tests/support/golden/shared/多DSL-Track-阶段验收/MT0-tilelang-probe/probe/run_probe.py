#!/usr/bin/env python3
"""Outer-layer trusted runner for the MT0 TileLang source-only frontend probe.

Run this from the repository root on the real outer control machine, where this
process is not itself nested inside a sandbox that denies ``sandbox_apply``.
The runner:

1. re-verifies the staged frontend (wheel-facing METADATA/RECORD and the whole
   ``tilelang`` package manifest) against ``frontend-install.json``;
2. refuses with typed ``backend_capability_unavailable`` when a minimal
   ``sandbox-exec`` capability probe cannot apply a profile in this context;
3. runs the frozen in-fence entry through the single lower-level hmz Profile
   route (``HmzFenceExecutor``), with ``online=false``, an empty host allowlist,
   read-only frontend root, attempt/build-only writes, no mds write root;
4. validates the complete artifact set and build identity; and
5. for ``--fixture`` runs, executes one fixed candidate-side negative fixture
   under the same fence and records the expected denial or contract rejection.

This file never imports ``tilelang`` and never executes generated device code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = HERE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcenv.worker_isolation import (  # noqa: E402
    ArtifactContractError,
    AttemptSpec,
    HmzCapabilityUnavailable,
    HmzFenceExecutor,
    ResourceLimits,
)

FIXTURES = {
    "read_outside": "read_outside.py",
    "write_outside": "write_outside.py",
    "network_connect": "network_connect.py",
    "undeclared_artifact": "undeclared_artifact.py",
}
EXPECTED_FIXTURE_OUTCOME = {
    "read_outside": "denied_no_artifacts",
    "write_outside": "denied_no_artifacts",
    "network_connect": "denied_no_artifacts",
    "undeclared_artifact": "artifact_contract_rejected",
}
CAPABILITY_SCHEMA = "shapeforge.mt0-tilelang-probe.probe-result.1"
PROFILE_PROBE = ("/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true")


class FrontendStateError(RuntimeError):
    """The staged frontend cannot be used; never falls back to an unstaged run."""

    code = "frontend_not_staged"


class FrontendIdentityDrift(FrontendStateError):
    code = "frontend_identity_drift"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("mt0_tilelang_frontend_entry", HERE / "frontend_entry.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = _load_probe_module()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return entry.sha256_file(path)


def verify_staged_frontend(install: dict) -> dict:
    installed = install["installed"]
    recorded = Path(installed["staging_root"])
    if not recorded.is_dir():
        raise FrontendStateError(f"frontend staging root is absent: {recorded}")
    staging = recorded.resolve()
    venv_python = Path(install["python"]["venv_python"]).resolve()
    if not venv_python.is_file() or not os.access(venv_python, os.X_OK):
        raise FrontendStateError(f"frontend python is absent: {venv_python}")
    if venv_python != (staging / "venv" / "bin" / "python").resolve():
        raise FrontendIdentityDrift("frontend python is outside the recorded venv")
    site_packages = staging / "venv" / "lib" / f"python{install['python']['short_version']}" / "site-packages"
    package = site_packages / "tilelang"
    files, total, digest = entry.package_manifest(package)
    if (files, total, digest) != (
        installed["tilelang_package_files"],
        installed["tilelang_package_bytes"],
        installed["tilelang_package_manifest_sha256"],
    ):
        raise FrontendIdentityDrift("staged tilelang package does not match the recorded install identity")
    dist_info = site_packages / f"tilelang-{install['pinned']['version']}.dist-info"
    if _sha256(dist_info / "METADATA") != installed["dist_info_metadata_sha256"]:
        raise FrontendIdentityDrift("staged tilelang METADATA drift")
    if _sha256(dist_info / "RECORD") != installed["dist_info_record_sha256"]:
        raise FrontendIdentityDrift("staged tilelang RECORD drift")
    if _sha256(Path(install["wheel"]["local_path"])) != install["wheel"]["sha256"]:
        raise FrontendIdentityDrift("staged wheel bytes no longer match the pinned wheel sha256")
    return {"staging_root": str(staging), "venv_python": str(venv_python), "package_sha256": digest}


def preflight_seatbelt() -> dict:
    """Observe whether this context can apply a Seatbelt profile at all."""

    try:
        completed = subprocess.run(
            list(PROFILE_PROBE), capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "argv": list(PROFILE_PROBE), "returncode": None,
            "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
            "sandbox_apply_possible": False,
        }
    return {
        "argv": list(PROFILE_PROBE),
        "returncode": completed.returncode,
        "stdout": completed.stdout[:2000],
        "stderr": completed.stderr[:2000],
        "sandbox_apply_possible": completed.returncode == 0,
    }


def pinned_frontend_document(install: dict) -> dict:
    installed = install["installed"]
    return {
        "schema_id": "shapeforge.mt0-tilelang-probe.pinned-frontend.1",
        "tilelang_version": install["pinned"]["version"],
        "upstream_commit": install["pinned"]["upstream_commit"],
        "tvm_commit": install["pinned"]["tvm_commit"],
        "wheel_sha256": install["wheel"]["sha256"],
        "tilelang_package_files": installed["tilelang_package_files"],
        "tilelang_package_bytes": installed["tilelang_package_bytes"],
        "tilelang_package_manifest_sha256": installed["tilelang_package_manifest_sha256"],
        "dist_info_metadata_sha256": installed["dist_info_metadata_sha256"],
        "dist_info_record_sha256": installed["dist_info_record_sha256"],
    }


def build_spec(args, install: dict) -> tuple[AttemptSpec, dict]:
    attempt_root = Path(args.attempt_root)
    venv_python = Path(install["python"]["venv_python"])
    frontend_root = Path(install["installed"]["staging_root"])
    inputs = {
        "frontend_entry.py": (HERE / "frontend_entry.py").read_bytes(),
        "candidate.py": (HERE / "candidate.py").read_bytes(),
        "pinned-frontend.json": (
            json.dumps(pinned_frontend_document(install), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    artifacts = tuple(entry.ARTIFACT_NAMES)
    attempt_artifacts = attempt_root / "build" / "artifacts"
    argv = (
        str(venv_python), "-I", str(attempt_root / "input" / "frontend_entry.py"),
        "--candidate", str(attempt_root / "input" / "candidate.py"),
        "--pinned", str(attempt_root / "input" / "pinned-frontend.json"),
        "--artifacts", str(attempt_artifacts),
    )
    spec = AttemptSpec(
        attempt_root=attempt_root,
        argv=argv,
        artifacts=artifacts,
        input_files=inputs,
        read_only_roots=(frontend_root,),
        extra_env={
            "TILELANG_CACHE_DIR": str(attempt_root / "build" / "cache" / "tilelang"),
            "TILELANG_DISABLE_CACHE": "0",
            "TILELANG_PRINT_ON_COMPILATION": "0",
        },
        limits=ResourceLimits(
            wall_seconds=args.wall_seconds,
            cpu_seconds=int(args.wall_seconds) + 30,
            file_bytes=64 * 1024**2,
            open_files=256,
            artifact_total_bytes=32 * 1024**2,
        ),
    )
    return spec, {"inputs": inputs, "artifacts": artifacts, "attempt_artifacts": str(attempt_artifacts)}


def build_fixture_spec(args, install: dict, fixture_path: Path, fixture_args: list[str]) -> AttemptSpec:
    attempt_root = Path(args.attempt_root)
    inputs = {"fixture.py": fixture_path.read_bytes()}
    argv = (
        str(Path(install["python"]["venv_python"])), "-I",
        str(attempt_root / "input" / "fixture.py"),
        *fixture_args,
        "--artifacts", str(attempt_root / "build" / "artifacts"),
    )
    return AttemptSpec(
        attempt_root=attempt_root,
        argv=argv,
        artifacts=(),
        input_files=inputs,
        read_only_roots=(Path(install["installed"]["staging_root"]),),
        limits=ResourceLimits(
            wall_seconds=args.wall_seconds,
            cpu_seconds=int(args.wall_seconds) + 30,
            file_bytes=64 * 1024**2,
            open_files=128,
            artifact_total_bytes=32 * 1024**2,
        ),
    )


def validate_full_build(isolation, install: dict, spec: AttemptSpec) -> dict:
    if not isolation.ok:
        raise SystemExit(f"probe did not complete successfully: {isolation.status}/{isolation.returncode}")
    rows = {row["path"]: row for row in isolation.document["artifacts"]}
    if set(rows) != set(entry.ARTIFACT_NAMES):
        raise SystemExit("trusted collector returned an unexpected artifact set")
    artifacts_dir = Path(spec.attempt_root) / "build" / "artifacts"
    identity = _json(artifacts_dir / "build_identity.json")
    if identity.get("full_build") is not True:
        raise SystemExit("build identity does not claim a full source-only build")
    candidate = (Path(spec.attempt_root) / "input" / "candidate.py").read_bytes()
    if identity.get("candidate_sha256") != hashlib.sha256(candidate).hexdigest():
        raise SystemExit("build identity candidate digest drift")
    options = identity.get("build_options", {})
    if options.get("target") != {"kind": "metal", "keys": ["metal", "gpu"], "target_host": "c"}:
        raise SystemExit("build identity target drift")
    if options.get("enable_device_compile") is not False:
        raise SystemExit("build identity claims device compilation")
    execution = identity.get("device_execution", {})
    if any(execution.get(key) is not False for key in ("device_library_created", "pipeline_state_created", "buffer_allocated")):
        raise SystemExit("build identity records device object creation")
    if execution.get("dispatch_calls") != 0:
        raise SystemExit("build identity records a device dispatch")
    expected_rows = []
    for name in entry.BUILD_ARTIFACTS:
        data = (artifacts_dir / name).read_bytes()
        expected_rows.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if not data.strip():
            raise SystemExit(f"declared build artifact is empty: {name}")
    if identity.get("artifacts") != expected_rows:
        raise SystemExit("build identity artifact rows were not recomputed consistently")
    digest = hashlib.sha256(
        json.dumps(expected_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if identity.get("artifact_set_sha256") != digest:
        raise SystemExit("artifact set digest drift")
    kernel = (artifacts_dir / "kernel.metal").read_text(encoding="utf-8")
    if "kernel" not in kernel or "metal" not in kernel or "scale_kernel" not in kernel:
        raise SystemExit("kernel.metal is not the expected Metal source artifact")
    if _sha256(artifacts_dir / "build_identity.json") != rows["build_identity.json"]["sha256"]:
        raise SystemExit("collector and on-disk build identity disagree")
    frontend = _json(artifacts_dir / "frontend_identity.json")
    if frontend.get("verified", {}).get("package_manifest_sha256") != install["installed"]["tilelang_package_manifest_sha256"]:
        raise SystemExit("frontend identity does not bind the pinned package manifest")
    if frontend.get("build_options", {}).get("target", {}).get("keys") != ["metal", "gpu"]:
        raise SystemExit("frontend identity target keys drift")
    return {
        "full_build": True,
        "artifact_rows": expected_rows,
        "artifact_set_sha256": digest,
        "kernel_metal_bytes": len(kernel.encode("utf-8")),
        "kernel_metal_sha256": hashlib.sha256(kernel.encode("utf-8")).hexdigest(),
        "frontend_identity": frontend.get("verified"),
        "build_options": options,
        "device_execution": execution,
        "torch_mps_availability_queries": frontend.get("import", {}).get("torch_mps_availability_queries"),
    }


def _result_document(args, install, capability, isolation=None, extra=None) -> dict:
    document = {
        "schema_id": CAPABILITY_SCHEMA,
        "status": "blocked_backend_capability_unavailable",
        "sandbox_capability_probe": capability,
        "typed_code": "backend_capability_unavailable",
        "frontend": {
            "version": install["pinned"]["version"],
            "upstream_commit": install["pinned"]["upstream_commit"],
            "wheel_sha256": install["wheel"]["sha256"],
            "staging_root": install["installed"]["staging_root"],
        },
        "probe_executed": False,
        "full_build": False,
        "gpu_npu_executed": False,
        "publication_allowed": False,
        "production_ready": False,
        "mt0_status": "blocked",
        "mt1b_status": "not_started_blocked",
        "unfenced_fallback_used": False,
    }
    if isolation is not None:
        document["isolation"] = {
            "status": isolation.status,
            "returncode": isolation.returncode,
            "request_sha256": isolation.document.get("request_sha256"),
            "policy_online": isolation.document.get("policy", {}).get("online"),
            "policy_hosts": isolation.document.get("policy", {}).get("hosts"),
            "implicit_mds_write_roots": isolation.document.get("process_boundary", {}).get("implicit_mds_write_roots"),
            "internal_fence_or_wrap_used": isolation.document.get("process_boundary", {}).get("internal_fence_or_wrap_used"),
            "driver_report": isolation.document.get("driver_report"),
            "limits": isolation.document.get("resource_limits"),
            "stdout_path": isolation.document.get("stdout_path"),
            "stderr_path": isolation.document.get("stderr_path"),
        }
    if extra:
        document.update(extra)
    return document


def _write_result(path: Path | None, document: dict) -> None:
    if path is None:
        print(json.dumps(document, indent=2, sort_keys=True))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-install", type=Path, default=PACKAGE / "frontend-install.json")
    parser.add_argument("--attempt-root", type=Path, default=Path("/tmp/shapeforge-mt0-tilelang-attempt"))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--wall-seconds", type=float, default=300.0)
    parser.add_argument("--fixture", choices=sorted(FIXTURES))
    parser.add_argument("--outside-path", type=Path)
    parser.add_argument("--outside-port", type=int)
    args = parser.parse_args(argv)

    install = _json(args.frontend_install)
    try:
        staged = verify_staged_frontend(install)
    except FrontendStateError as exc:
        _write_result(args.result, {
            **_result_document(args, install, {"argv": [], "returncode": None, "stdout": "", "stderr": "", "sandbox_apply_possible": False}),
            "status": f"blocked_{exc.code}",
            "typed_code": exc.code,
            "detail": str(exc),
        })
        return 7
    capability = preflight_seatbelt()
    if not capability["sandbox_apply_possible"]:
        _write_result(args.result, _result_document(args, install, capability, extra={"staged_frontend": staged}))
        print("backend_capability_unavailable: cannot apply a Seatbelt profile in this context", file=sys.stderr)
        return 3

    if args.fixture:
        return _run_fixture(args, install, capability)

    spec, _ = build_spec(args, install)
    executor = HmzFenceExecutor()
    try:
        isolation = executor.run(spec)
    except HmzCapabilityUnavailable as exc:
        _write_result(args.result, {
            **_result_document(args, install, capability),
            "status": "blocked_backend_capability_unavailable",
            "detail": exc.detail,
            "evidence": exc.evidence,
        })
        return 3
    except ArtifactContractError as exc:
        _write_result(args.result, _result_document(
            args, install, capability, extra={"status": "artifact_contract_rejected", "detail": str(exc)},
        ))
        return 4
    try:
        build = validate_full_build(isolation, install, spec)
    except SystemExit as exc:
        _write_result(args.result, _result_document(
            args, install, capability, isolation=isolation, extra={"status": "full_build_rejected", "detail": str(exc)},
        ))
        return 5
    _write_result(args.result, {
        "schema_id": CAPABILITY_SCHEMA,
        "status": "source_only_frontend_probe_completed",
        "sandbox_capability_probe": capability,
        "staged_frontend": staged,
        "frontend": {
            "version": install["pinned"]["version"],
            "upstream_commit": install["pinned"]["upstream_commit"],
            "wheel_sha256": install["wheel"]["sha256"],
        },
        "probe_executed": True,
        "full_build": True,
        "build": build,
        "isolation": _result_document(args, install, capability, isolation=isolation)["isolation"],
        "gpu_npu_executed": False,
        "device_dispatch_calls": 0,
        "publication_allowed": False,
        "production_ready": False,
        "metal_correctness_or_performance_claim": False,
        "mt0_status": "blocked",
        "mt1b_status": "not_started_blocked",
    })
    return 0


def _run_fixture(args, install: dict, capability: dict) -> int:
    fixture_path = HERE / "fixtures" / FIXTURES[args.fixture]
    fixture_args: list[str] = []
    canary = None
    canary_before = None
    if args.fixture in {"read_outside", "write_outside"}:
        if args.outside_path is None:
            print("--outside-path is required for this fixture", file=sys.stderr)
            return 2
        canary = args.outside_path.resolve()
        forbidden = (Path(args.attempt_root).resolve(), ROOT, Path(install["installed"]["staging_root"]).resolve())
        if any(canary.is_relative_to(root) for root in forbidden):
            print("--outside-path must be outside the attempt, repository and frontend root", file=sys.stderr)
            return 2
        canary.parent.mkdir(parents=True, exist_ok=True)
        canary.write_bytes(b"sf-probe-canary\n")
        canary_before = hashlib.sha256(canary.read_bytes()).hexdigest()
        fixture_args = ["--outside-path", str(canary)]
    if args.fixture == "network_connect":
        if args.outside_port is None:
            print("--outside-port is required for this fixture", file=sys.stderr)
            return 2
        fixture_args = ["--outside-port", str(args.outside_port)]

    listener = None
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            connection.close()

    if args.fixture == "network_connect":
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        args.outside_port = listener.getsockname()[1]
        fixture_args = ["--outside-port", str(args.outside_port)]
        threading.Thread(target=serve, daemon=True).start()

    spec = build_fixture_spec(args, install, fixture_path, fixture_args)
    executor = HmzFenceExecutor()
    outcome: dict = {"fixture": args.fixture, "expected": EXPECTED_FIXTURE_OUTCOME[args.fixture]}
    try:
        isolation = executor.run(spec)
    except ArtifactContractError as exc:
        outcome.update({
            "status": "artifact_contract_rejected", "detail": str(exc),
            "matched": EXPECTED_FIXTURE_OUTCOME[args.fixture] == "artifact_contract_rejected",
        })
        _write_result(args.result, _result_document(args, install, capability, extra={"fixture_observation": outcome}))
        return 0 if outcome["matched"] else 4
    except HmzCapabilityUnavailable as exc:
        _write_result(args.result, _result_document(args, install, capability, extra={"fixture_observation": outcome, "detail": exc.detail}))
        return 3
    finally:
        stop.set()
        if listener is not None:
            listener.close()
    outcome.update({
        "status": isolation.status,
        "returncode": isolation.returncode,
        "artifact_count": len(isolation.document["artifacts"]),
        "matched": (
            EXPECTED_FIXTURE_OUTCOME[args.fixture] == "denied_no_artifacts"
            and isolation.returncode == 13
            and isolation.document["artifacts"] == []
        ),
    })
    if canary is not None:
        after = hashlib.sha256(canary.read_bytes()).hexdigest() if canary.exists() else None
        outcome["outside_canary_unchanged"] = after == canary_before
        outcome["outside_canary_sha256"] = canary_before
        if not outcome["outside_canary_unchanged"]:
            outcome["matched"] = False
    _write_result(args.result, _result_document(args, install, capability, isolation=isolation, extra={"fixture_observation": outcome}))
    return 0 if outcome["matched"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
