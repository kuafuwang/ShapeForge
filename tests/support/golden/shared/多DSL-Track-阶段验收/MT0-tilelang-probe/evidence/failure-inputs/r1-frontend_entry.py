#!/usr/bin/env python3
"""In-fence, source-only TileLang frontend probe.

This script is *candidate-side* host code: it is started by the trusted hmz
lower-level Seatbelt ``Profile.command()`` chain, after the Profile has been
applied, and never by the ShapeForge control plane.  It parses one fixed
TileLang candidate, lowers it to host/device IR with the pinned frontend, and
emits Metal Shading Language *source* plus a host artifact and lowered IR text.

Hard boundaries (checked by ``verify.py`` and the fixed tests):

* no ``tilelang.compile``/``tilelang.jit``/JITKernel, no ``tvm.build`` with
  device compilation, no MTLDevice/MTLLibrary/pipeline/encoder/dispatch calls;
* no subprocess, socket, urllib or file write outside the declared artifact
  directory;
* the installed frontend identity is re-verified byte-wise before the import
  that executes the frontend, and MPS availability queries are only counted,
  never initiated here;
* every failure exits non-zero and leaves the artifact directory empty, so the
  trusted collector can never accept a partial build as a full build.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

EXIT_DEVICE_REQUIRED = 70
EXIT_IDENTITY_DRIFT = 71
EXIT_UNSUPPORTED = 72
EXIT_UNEXPECTED = 73

ARTIFACT_NAMES = (
    "frontend_identity.json",
    "kernel.metal",
    "device_ir.json",
    "kernel_params.json",
    "host_module.txt",
    "build_identity.json",
)
BUILD_ARTIFACTS = ARTIFACT_NAMES[:-1]
TARGET_KIND = "metal"
TARGET_KEYS = ("metal", "gpu")
TARGET_HOST = "c"
FRONTEND_ENTRY = "tilelang.engine.lower"


class ProbeError(RuntimeError):
    """One typed, non-recoverable source-only probe failure."""

    exit_code = EXIT_UNEXPECTED

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class IdentityDrift(ProbeError):
    exit_code = EXIT_IDENTITY_DRIFT


class Unsupported(ProbeError):
    exit_code = EXIT_UNSUPPORTED


class DeviceRequired(ProbeError):
    exit_code = EXIT_DEVICE_REQUIRED


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_manifest(root: Path) -> tuple[int, int, str]:
    """Hash every regular file under ``root`` except bytecode caches.

    The formatting rule is part of the pinned identity:
    ``relative/path  bytes  sha256`` joined by ``\\n`` in sorted path order.
    """

    rows: list[tuple[str, int, str]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        total += size
        rows.append((relative.as_posix(), size, sha256_file(path)))
    manifest = "\n".join(f"{name}  {size}  {digest}" for name, size, digest in rows)
    return len(rows), total, sha256_bytes(manifest.encode("utf-8"))


def load_json(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) > 64 * 1024:
        raise Unsupported(f"pinned identity document is too large: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise Unsupported("pinned identity document must be a JSON object")
    return value


def verify_frontend(pinned: dict) -> dict:
    """Locate and hash the installed frontend without executing it."""

    spec = importlib.util.find_spec("tilelang")
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if len(locations) != 1:
        raise IdentityDrift("tilelang package location is absent or ambiguous")
    package = Path(locations[0]).resolve()
    site_packages = package.parent
    if site_packages.name != "site-packages" or package.name != "tilelang":
        raise IdentityDrift(f"tilelang was found outside site-packages: {package}")
    files, total, digest = package_manifest(package)
    if files != pinned["tilelang_package_files"] or total != pinned["tilelang_package_bytes"]:
        raise IdentityDrift(f"tilelang package size/count drift: {files} files / {total} bytes")
    if digest != pinned["tilelang_package_manifest_sha256"]:
        raise IdentityDrift("tilelang package manifest digest drift")
    version = pinned["tilelang_version"]
    dist_info = site_packages / f"tilelang-{version}.dist-info"
    metadata = dist_info / "METADATA"
    record = dist_info / "RECORD"
    if not metadata.is_file() or not record.is_file():
        raise IdentityDrift("tilelang dist-info is incomplete")
    if sha256_file(metadata) != pinned["dist_info_metadata_sha256"]:
        raise IdentityDrift("tilelang dist-info METADATA drift")
    if sha256_file(record) != pinned["dist_info_record_sha256"]:
        raise IdentityDrift("tilelang dist-info RECORD drift")
    declared = [
        line.split(":", 1)[1].strip()
        for line in metadata.read_text(encoding="utf-8").splitlines()
        if line.startswith("Version:")
    ]
    if declared != [version]:
        raise IdentityDrift(f"tilelang dist-info version drift: {declared}")
    return {
        "site_packages": str(site_packages),
        "package": str(package),
        "package_files": files,
        "package_bytes": total,
        "package_manifest_sha256": digest,
        "dist_info_metadata_sha256": pinned["dist_info_metadata_sha256"],
        "dist_info_record_sha256": pinned["dist_info_record_sha256"],
        "tilelang_version": version,
    }


def instrument_mps_counters() -> dict[str, int]:
    """Count availability queries the frontend performs; never initiate one here."""

    import torch

    counters = {"torch.backends.mps.is_available": 0, "torch.mps.is_available": 0}

    def wrap(owner, name: str, key: str) -> None:
        original = getattr(owner, name, None)
        if original is None:
            return

        def counted(*args, **kwargs):
            counters[key] += 1
            return original(*args, **kwargs)

        setattr(owner, name, counted)

    wrap(torch.backends.mps, "is_available", "torch.backends.mps.is_available")
    if hasattr(torch, "mps"):
        wrap(torch.mps, "is_available", "torch.mps.is_available")
    return counters


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("mt0_tilelang_probe_candidate", path)
    if spec is None or spec.loader is None:
        raise Unsupported(f"candidate cannot be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "scale_kernel", None)
    if function is None:
        raise Unsupported("candidate does not define scale_kernel")
    return function


def lower_source_only(candidate_path: Path, frontend: dict) -> dict:
    mps = instrument_mps_counters()
    import tilelang
    from tilelang import tvm
    from tilelang.engine import lower as tilelang_lower
    from tilelang.engine.param import dump_kernel_params

    imported = Path(tilelang.__file__).resolve()
    if imported.parent != Path(frontend["package"]):
        raise IdentityDrift(f"tilelang was imported from an unexpected path: {imported}")
    if str(tilelang.__version__) != frontend["tilelang_version"]:
        raise IdentityDrift(f"tilelang.__version__ drift: {tilelang.__version__}")

    function = load_candidate(candidate_path)
    target = tvm.target.Target({"kind": TARGET_KIND, "keys": list(TARGET_KEYS)})
    try:
        artifact = tilelang_lower(
            function,
            target=target,
            target_host=TARGET_HOST,
            enable_host_codegen=True,
            enable_device_compile=False,
        )
    except ProbeError:
        raise
    except Exception as exc:  # noqa: BLE001 - classified, never swallowed
        message = f"{type(exc).__name__}: {exc}"
        lowered = message.lower()
        if any(token in lowered for token in ("mtl", "metal device", "device", "pipeline", "gpu")):
            raise DeviceRequired(f"frontend requires a device before codegen: {message}") from exc
        raise Unsupported(f"frontend lowering failed: {message}") from exc

    kernel_source = artifact.kernel_source
    if not isinstance(kernel_source, str) or not kernel_source.strip():
        raise Unsupported("frontend returned no device source")
    if "kernel" not in kernel_source or "metal" not in kernel_source:
        raise Unsupported("device source does not look like Metal Shading Language")
    if "scale_kernel" not in kernel_source:
        raise Unsupported("device source omits the candidate global symbol")

    device_ir = tvm.ir.save_json(artifact.device_mod)
    kernel_params = dump_kernel_params(artifact.params or [])

    host_module = artifact.host_mod
    host_artifact = None
    host_format = None
    if host_module is not None:
        for attribute, label in (("inspect_source", "c-source"), ("get_source", "c-source")):
            reader = getattr(host_module, attribute, None)
            if callable(reader):
                value = reader()
                if isinstance(value, str) and value.strip():
                    host_artifact, host_format = value, label
                break
        if host_artifact is None and hasattr(host_module, "functions"):
            value = tvm.ir.save_json(host_module)
            if isinstance(value, str) and value.strip():
                host_artifact, host_format = value, "ir-json"
    if host_artifact is None:
        raise Unsupported("host codegen did not produce a declared host artifact")

    return {
        "kernel_source": kernel_source,
        "device_ir": device_ir,
        "kernel_params": kernel_params,
        "host_artifact": host_artifact,
        "host_format": host_format,
        "mps_probe_calls": dict(mps),
        "target": {
            "kind": target.kind.name,
            "keys": [str(key) for key in target.keys],
            "target_host": TARGET_HOST,
        },
        "frontend_entry": FRONTEND_ENTRY,
        "enable_host_codegen": True,
        "enable_device_compile": False,
        "device_library_created": False,
        "pipeline_state_created": False,
        "buffer_allocated": False,
        "dispatch_calls": 0,
        "imported_tilelang_module": str(imported),
    }


def write_artifact(directory: Path, name: str, data: bytes) -> dict:
    if name not in ARTIFACT_NAMES:
        raise AssertionError(f"undeclared artifact requested: {name}")
    path = directory / name
    path.write_bytes(data)
    return {"path": name, "bytes": len(data), "sha256": sha256_bytes(data)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(argv)

    directory = args.artifacts
    try:
        if not directory.is_dir() or any(directory.iterdir()):
            raise Unsupported("artifact directory must be an existing empty directory")
        pinned = load_json(args.pinned)
        frontend = verify_frontend(pinned)
        result = lower_source_only(args.candidate, frontend)
        frontend_document = {
            "schema_id": "shapeforge.mt0-tilelang-probe.frontend-identity.2",
            "pinned": pinned,
            "verified": frontend,
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "import": {
                "tilelang_module": result["imported_tilelang_module"],
                "torch": True,
                "torch_mps_availability_queries": result["mps_probe_calls"],
            },
            "build_options": {
                "frontend_entry": result["frontend_entry"],
                "target": result["target"],
                "enable_host_codegen": result["enable_host_codegen"],
                "enable_device_compile": result["enable_device_compile"],
                "host_artifact_format": result["host_format"],
            },
            "device_execution": {
                "device_library_created": result["device_library_created"],
                "pipeline_state_created": result["pipeline_state_created"],
                "buffer_allocated": result["buffer_allocated"],
                "dispatch_calls": result["dispatch_calls"],
            },
        }
        artifacts = [
            write_artifact(
                directory, "frontend_identity.json",
                (json.dumps(frontend_document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            ),
            write_artifact(directory, "kernel.metal", result["kernel_source"].encode("utf-8")),
            write_artifact(directory, "device_ir.json", (result["device_ir"] + "\n").encode("utf-8")),
            write_artifact(directory, "kernel_params.json", (result["kernel_params"] + "\n").encode("utf-8")),
            write_artifact(directory, "host_module.txt", result["host_artifact"].encode("utf-8")),
        ]
        assert [row["path"] for row in artifacts] == list(BUILD_ARTIFACTS)
        identity = {
            "schema_id": "shapeforge.mt0-tilelang-probe.build-identity.1",
            "pinned_wheel_sha256": pinned["wheel_sha256"],
            "tilelang_package_manifest_sha256": pinned["tilelang_package_manifest_sha256"],
            "candidate_sha256": sha256_file(args.candidate),
            "frontend_entry": result["frontend_entry"],
            "build_options": frontend_document["build_options"],
            "device_execution": frontend_document["device_execution"],
            "artifacts": artifacts,
            "artifact_set_sha256": sha256_bytes(
                json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
            "full_build": True,
        }
        write_artifact(
            directory, "build_identity.json",
            (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    except ProbeError as exc:
        print(f"frontend-probe: {type(exc).__name__}: {exc.detail}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - any unexpected path must not look like success
        print(f"frontend-probe: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
