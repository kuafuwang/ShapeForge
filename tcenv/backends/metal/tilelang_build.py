"""Production source-only build for the TileLang Track.

The DSL frontend never runs on the control host.  This module

* hashes the staged, read-only frontend install on the control host (no import),
* composes one closed ``AttemptSpec`` and runs it through the existing lower-level
  hmz ``Profile`` route (:class:`~tcenv.worker_isolation.hmz_fence.HmzFenceExecutor`),
* reads the declared artifacts back, verifies their *content* against the frozen
  contract, and derives the build identity from independent inputs (candidate
  manifest bytes, the pinned install, the observed environment, the attempt
  terminal state and the artifact bytes).

A Metal build (``enable_device_compile``) stays a typed refusal: MT2 owns it.
Nothing here imports ``tilelang``; only the in-fence entry does, under the fence.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...track import tilelang as contract
from ...track.registry import TrackSpec
from ...worker_isolation.hmz_fence import (
    ArtifactContractError,
    AttemptSpec,
    HmzCapabilityUnavailable,
    HmzFenceExecutor,
    IsolationResult,
    ResourceLimits,
)

#: The frozen in-fence entry.  It is harness-owned trusted code written as an attempt
#: input; the candidate can never replace it.  Its digest enters the build identity.
FRONTEND_ENTRY_NAME = "frontend_entry.py"
PINNED_NAME = "pinned.json"
CANDIDATE_NAME = contract.SOURCE_PATH

STAGING_ENV = "SHAPEFORGE_TILELANG_STAGING"
#: The MT0 staging root (read-only install); overridable for a fresh epoch.
DEFAULT_STAGING_ROOT = Path("/tmp/shapeforge-frontends/tilelang-0.1.15")

EXIT_DEVICE_REQUIRED = 70
EXIT_IDENTITY_DRIFT = 71
EXIT_UNSUPPORTED = 72
EXIT_UNEXPECTED = 73

FRONTEND_ENTRY_SOURCE = '''#!/usr/bin/env python3
"""Frozen in-fence TileLang source-only frontend entry (harness-owned).

Runs under the staged interpreter inside the hmz lower-level Seatbelt Profile.
It re-verifies the installed package against the pinned identity *before* the
import that executes the frontend, lowers exactly one prim_func with
``enable_device_compile=False``, and writes only the declared source artifacts.
Any failure exits non-zero and leaves the artifact directory empty.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import traceback
from pathlib import Path

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
)


class EntryError(RuntimeError):
    exit_code = EXIT_UNEXPECTED


class IdentityDrift(EntryError):
    exit_code = EXIT_IDENTITY_DRIFT


class Unsupported(EntryError):
    exit_code = EXIT_UNSUPPORTED


class DeviceRequired(EntryError):
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
    rows = []
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
    manifest = "\\n".join(f"{name}  {size}  {digest}" for name, size, digest in rows)
    return len(rows), total, sha256_bytes(manifest.encode("utf-8"))


def verify_frontend(pinned: dict) -> dict:
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


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("shapeforge_tilelang_candidate", path)
    if spec is None or spec.loader is None:
        raise Unsupported(f"candidate cannot be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tilelang import tvm

    declared_many = getattr(module, "KERNELS", None)
    if declared_many is not None:
        if not isinstance(declared_many, (list, tuple)) or not declared_many:
            raise Unsupported("KERNELS must be a non-empty tuple/list of prim_func values")
        functions = list(declared_many)
    else:
        declared = getattr(module, "KERNEL", None)
        if declared is not None:
            functions = [declared]
        else:
            functions = [
                value
                for name, value in vars(module).items()
                if not name.startswith("_") and isinstance(value, tvm.tirx.PrimFunc)
            ]
            if len(functions) != 1:
                raise Unsupported(
                    "the candidate must define exactly one prim_func, declare KERNEL, or declare KERNELS"
                )
    if any(not isinstance(value, tvm.tirx.PrimFunc) for value in functions):
        raise Unsupported("KERNEL/KERNELS must contain only prim_func values")
    names = [str(value.attrs["global_symbol"]) for value in functions]
    if len(set(names)) != len(names):
        raise Unsupported(f"candidate prim_func global_symbol values must be unique: {names}")
    return functions


def normalize_module_source(source: str) -> str:
    """Deduplicate the fixed preamble TileLang repeats for each function in one IRModule."""
    if source.count("// Function:") <= 1:
        return source
    lines = source.splitlines(keepends=True)
    output = []
    seen_preamble = set()
    seen_union = False
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped in {
            "#include <metal_stdlib>",
            "#include <metal_simdgroup>",
            '#define TILELANG_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")',
            "using namespace metal;",
        }:
            if stripped in seen_preamble:
                index += 1
                continue
            seen_preamble.add(stripped)
        if stripped == "union __TVMArgUnion {":
            if seen_union:
                index += 1
                while index < len(lines) and lines[index].strip() != "};":
                    index += 1
                if index >= len(lines):
                    raise Unsupported("unterminated repeated __TVMArgUnion in generated MSL")
                index += 1
                continue
            seen_union = True
        output.append(line)
        index += 1
    return "".join(output)


def lower_source_only(candidate_path: Path, frontend: dict) -> dict:
    import tilelang
    from tilelang import tvm
    from tilelang.engine import lower as tilelang_lower
    from tilelang.engine.param import dump_kernel_params

    imported = Path(tilelang.__file__).resolve()
    if imported.parent != Path(frontend["package"]):
        raise IdentityDrift(f"tilelang was imported from an unexpected path: {imported}")
    if str(tilelang.__version__) != frontend["tilelang_version"]:
        raise IdentityDrift(f"tilelang.__version__ drift: {tilelang.__version__}")
    functions = load_candidate(candidate_path)
    target = tvm.target.Target({"kind": "metal", "keys": ["metal", "gpu"]})
    try:
        # A multi-stage candidate is lowered as one IRModule by the frontend.  This TileLang
        # version repeats fixed include/union preambles per emitted function; the harness-owned
        # normalization below removes only exact duplicates, never a function body.
        if len(functions) == 1:
            function_or_module = functions[0]
        else:
            function_or_module = tvm.IRModule({
                function.attrs["global_symbol"]: function for function in functions
            })
        with target:
            artifact = tilelang_lower(
                function_or_module,
                target=target,
                target_host="c",
                enable_host_codegen=True,
                enable_device_compile=False,
            )
    except Exception as exc:  # noqa: BLE001 - classified, never swallowed
        traceback.print_exc()
        message = f"{type(exc).__name__}: {exc}"
        lowered = message.lower()
        if any(token in lowered for token in ("mtl", "metal device", "pipeline", "device")):
            raise DeviceRequired(f"frontend requires a device before codegen: {message}") from exc
        raise Unsupported(f"frontend lowering failed: {message}") from exc

    kernel_source = artifact.kernel_source
    if not isinstance(kernel_source, str) or not kernel_source.strip():
        raise Unsupported("frontend returned no device source")
    if "kernel" not in kernel_source or "metal" not in kernel_source:
        raise Unsupported("device source does not look like Metal Shading Language")
    kernel_source = normalize_module_source(kernel_source)

    # TVM records the candidate's absolute SourceName.  Attempt roots are intentionally random,
    # so leaving that host path in the artifact would make identical candidates produce different
    # artifact-set digests.  Canonicalize only this source-location metadata; the IR instructions
    # and every other field remain the frontend's own output.
    device_ir_document = json.loads(tvm.ir.save_json(artifact.device_mod))
    candidate_resolved = candidate_path.resolve()

    def canonicalize_source_names(value):
        if isinstance(value, dict):
            if value.get("type") == "ir.SourceName" and isinstance(value.get("data"), str):
                source_name = Path(value["data"])
                if source_name.is_absolute() and source_name.resolve() == candidate_resolved:
                    value["data"] = "kernel.py"
            for item in value.values():
                canonicalize_source_names(item)
        elif isinstance(value, list):
            for item in value:
                canonicalize_source_names(item)

    canonicalize_source_names(device_ir_document)
    device_ir = json.dumps(device_ir_document, indent=2)
    params = artifact.params
    if params is None:
        from tilelang.engine.lower import extrac_params

        params = [param for function in functions for param in extrac_params(function)]
    kernel_params = dump_kernel_params(params)
    host_module = artifact.host_mod
    host_artifact = None
    if host_module is not None:
        for attribute in ("inspect_source", "get_source"):
            reader = getattr(host_module, attribute, None)
            if callable(reader):
                value = reader()
                if isinstance(value, str) and value.strip():
                    host_artifact = value
                break
        if host_artifact is None and hasattr(host_module, "functions"):
            value = tvm.ir.save_json(host_module)
            if isinstance(value, str) and value.strip():
                host_artifact = value
    if host_artifact is None:
        raise Unsupported("host codegen did not produce a declared host artifact")

    declarations = [
        str(function.attrs["global_symbol"])
        for function in functions
        if function.attrs and "global_symbol" in function.attrs
    ]
    return {
        "kernel_source": kernel_source,
        "device_ir": device_ir,
        "kernel_params": kernel_params,
        "host_artifact": host_artifact,
        "prim_func_name": declarations[0] if len(declarations) == 1 else declarations,
        "target": {
            "kind": target.kind.name,
            "keys": [str(key) for key in target.keys],
            "target_host": "c",
        },
        "imported_tilelang_module": str(imported),
    }


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
        pinned = json.loads(args.pinned.read_text(encoding="utf-8"))
        if not isinstance(pinned, dict) or len(pinned) > 64:
            raise Unsupported("pinned identity document must be a small JSON object")
        frontend = verify_frontend(pinned)
        source = args.candidate.read_bytes()
        if sha256_bytes(source) != pinned["candidate_source_sha256"]:
            raise IdentityDrift("the in-fence candidate bytes are not the projected source")
        result = lower_source_only(args.candidate, frontend)
        frontend_document = {
            "schema_id": "tcenv.track.tilelang.frontend-identity.1",
            "verified": frontend,
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "candidate": {
                "source_sha256": sha256_bytes(source),
                "prim_func_name": result["prim_func_name"],
            },
            "build_options": {
                "frontend_entry": pinned["frontend_entry"],
                "target": result["target"],
                "enable_host_codegen": True,
                "enable_device_compile": False,
                "host_artifact_format": "c-source",
            },
            "device_execution": {
                "device_library_created": False,
                "pipeline_state_created": False,
                "buffer_allocated": False,
                "dispatch_calls": 0,
            },
            "imported_tilelang_module": result["imported_tilelang_module"],
        }
        outputs = {
            "frontend_identity.json": json.dumps(frontend_document, sort_keys=True, indent=2).encode("utf-8"),
            "kernel.metal": result["kernel_source"].encode("utf-8"),
            "device_ir.json": result["device_ir"].encode("utf-8"),
            "kernel_params.json": result["kernel_params"].encode("utf-8"),
            "host_module.txt": result["host_artifact"].encode("utf-8"),
        }
        if sorted(outputs) != sorted(ARTIFACT_NAMES):
            raise AssertionError("the entry wrote an undeclared artifact")
        for name in ARTIFACT_NAMES:
            (directory / name).write_bytes(outputs[name])
        return 0
    except EntryError as exc:
        traceback.print_exc()
        print(f"source-only entry refused: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception:  # noqa: BLE001 - every unexpected failure is non-zero
        traceback.print_exc()
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
'''

#: The frozen digest of the entry above; recomputed on import so a text edit cannot silently
#: change what ran inside the fence.
FRONTEND_ENTRY_SHA256 = hashlib.sha256(FRONTEND_ENTRY_SOURCE.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StagedFrontend:
    """A read-only, byte-verified frontend install on the control host."""

    root: Path
    site_packages: Path
    package: Path
    python: Path
    package_files: int
    package_bytes: int
    package_manifest_sha256: str
    dist_info_metadata_sha256: str
    dist_info_record_sha256: str
    version: str

    def observation(self) -> dict[str, Any]:
        """@brief 前端观测（与外部 worker 观测同一闭集身份）。

        @return 观测字典。
        """
        return {
            "frontend_id": contract.FRONTEND_ID,
            "frontend_version": self.version,
            "entry": contract.FRONTEND_ENTRY,
            "options_sha256": contract.options_sha256(),
            "package_manifest_sha256": self.package_manifest_sha256,
            "dist_info_metadata_sha256": self.dist_info_metadata_sha256,
            "dist_info_record_sha256": self.dist_info_record_sha256,
            "package_files": self.package_files,
            "package_bytes": self.package_bytes,
        }

    def pinned_document(self, candidate_source_sha256: str) -> dict[str, Any]:
        """@brief 交给围栏内 entry 的冻结身份（只含需要的字段）。

        @param candidate_source_sha256 候选 ``kernel.py`` 摘要。
        @return JSON 字典。
        """
        return {
            "tilelang_version": self.version,
            "frontend_entry": contract.FRONTEND_ENTRY,
            "tilelang_package_files": self.package_files,
            "tilelang_package_bytes": self.package_bytes,
            "tilelang_package_manifest_sha256": self.package_manifest_sha256,
            "dist_info_metadata_sha256": self.dist_info_metadata_sha256,
            "dist_info_record_sha256": self.dist_info_record_sha256,
            "candidate_source_sha256": candidate_source_sha256,
        }

    def document(self) -> dict[str, Any]:
        """@brief 观测证据。

        @return 字段字典。
        """
        return {
            "schema_id": "tcenv.track.tilelang.staged-frontend.1",
            "root": str(self.root),
            "package": str(self.package),
            "site_packages": str(self.site_packages),
            "python": str(self.python),
            "observation": self.observation(),
            "manifest_rule": contract.FRONTEND_MANIFEST_RULE,
            "imported_by_control_host": False,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_manifest(root: Path) -> tuple[int, int, str]:
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
        rows.append((relative.as_posix(), size, _sha256_file(path)))
    manifest = "\n".join(f"{name}  {size}  {digest}" for name, size, digest in rows)
    return len(rows), total, hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def staging_root(root: Path | None = None) -> Path:
    """@brief 解析 staged frontend 根目录（显式参数 > 环境变量 > MT0 staging）。

    @param root 显式根目录。
    @return 路径。
    """
    if root is not None:
        return Path(root)
    configured = os.environ.get(STAGING_ENV)
    return Path(configured) if configured else DEFAULT_STAGING_ROOT


def observe_staged_frontend(root: Path | None = None) -> StagedFrontend:
    """@brief 在控制机上对 read-only 前端安装做真实观测（不 import 前端）。

    @param root staged frontend 根目录。
    @return 观测结果。
    @raises contract.TileLangFrontendUnavailable 目录/解释器缺失。
    @raises contract.TileLangIdentityDrift 安装字节与冻结身份不符。
    """
    base = staging_root(root)
    site_packages_candidates = sorted((base / "venv" / "lib").glob("python*/site-packages"))
    if not site_packages_candidates:
        raise contract.TileLangFrontendUnavailable(
            f"no staged TileLang install under {base}; a source-only build needs the pinned frontend "
            f"(set {STAGING_ENV})"
        )
    site_packages = site_packages_candidates[0]
    package = site_packages / "tilelang"
    if not package.is_dir():
        raise contract.TileLangFrontendUnavailable(f"the staged install has no tilelang package: {package}")
    python = base / "venv" / "bin" / "python3"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise contract.TileLangFrontendUnavailable(f"the staged install has no executable interpreter: {python}")
    files, total, digest = _package_manifest(package)
    if (files, total, digest) != (
        contract.FRONTEND_PACKAGE_FILES,
        contract.FRONTEND_PACKAGE_BYTES,
        contract.FRONTEND_PACKAGE_MANIFEST_SHA256,
    ):
        raise contract.TileLangIdentityDrift(
            f"the staged tilelang package is {files} files / {total} bytes / {digest}, the pinned install is "
            f"{contract.FRONTEND_PACKAGE_FILES} files / {contract.FRONTEND_PACKAGE_BYTES} bytes / "
            f"{contract.FRONTEND_PACKAGE_MANIFEST_SHA256}; a different install is a new epoch, not a silent reuse"
        )
    dist_info = site_packages / f"tilelang-{contract.FRONTEND_VERSION}.dist-info"
    metadata = _sha256_file(dist_info / "METADATA") if (dist_info / "METADATA").is_file() else ""
    record = _sha256_file(dist_info / "RECORD") if (dist_info / "RECORD").is_file() else ""
    if (metadata, record) != (
        contract.FRONTEND_DIST_INFO_METADATA_SHA256,
        contract.FRONTEND_DIST_INFO_RECORD_SHA256,
    ):
        raise contract.TileLangIdentityDrift(
            "the staged tilelang dist-info (METADATA/RECORD) is not the pinned one"
        )
    return StagedFrontend(
        root=base.resolve(),
        site_packages=site_packages.resolve(),
        package=package.resolve(),
        python=python.resolve(),
        package_files=files,
        package_bytes=total,
        package_manifest_sha256=digest,
        dist_info_metadata_sha256=metadata,
        dist_info_record_sha256=record,
        version=contract.FRONTEND_VERSION,
    )


def observe_staged_environment(frontend: StagedFrontend) -> dict[str, str]:
    """@brief 观测 staged 解释器的设备无关事实（真实子进程，不 import 前端）。

    @param frontend staged 安装。
    @return ``FRONTEND_ENVIRONMENT_KEYS`` 闭集。
    @raises contract.TileLangFrontendUnavailable 解释器不可执行。
    """
    program = (
        "import json, platform, sys; "
        "print(json.dumps({'python_version': platform.python_version(), "
        "'python_implementation': platform.python_implementation(), "
        "'platform': platform.platform(), 'machine': platform.machine(), "
        "'tilelang_module': sys.prefix}))"
    )
    try:
        completed = subprocess.run(
            [str(frontend.python), "-I", "-c", program],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=60,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise contract.TileLangFrontendUnavailable(
            f"the staged interpreter could not be observed: {type(exc).__name__}: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise contract.TileLangFrontendUnavailable(
            f"the staged interpreter exited {completed.returncode}: {completed.stderr.decode('utf-8', 'replace')[-400:]}"
        )
    try:
        document = json.loads(completed.stdout)
    except ValueError as exc:
        raise contract.TileLangFrontendUnavailable(f"the staged interpreter returned no JSON facts: {exc}") from exc
    environment = {name: str(document.get(name, "")) for name in contract.FRONTEND_ENVIRONMENT_KEYS}
    missing = [name for name, value in environment.items() if not value]
    if missing:
        raise contract.TileLangFrontendUnavailable(f"the staged interpreter did not report {missing}")
    return environment


def _artifact_bytes(directory: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for name in contract.SOURCE_ONLY_ARTIFACT_PATHS:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise contract.TileLangContractError(f"the attempt did not produce {name!r}")
        found[name] = path.read_bytes()
    stray = sorted(
        child.name for child in directory.iterdir() if child.is_file() and child.name not in found
    )
    if stray:
        raise contract.TileLangContractError(f"the attempt produced undeclared artifacts {stray}")
    return found


def _attempt_record(result: IsolationResult) -> dict[str, Any]:
    document = dict(result.document)
    return {
        "status": str(document.get("status")),
        "returncode": int(document.get("returncode") or 0),
        "timed_out": document.get("watchdog_reason") not in (None, ""),
        "online": bool(document.get("policy", {}).get("online")),
        "hosts": list(document.get("policy", {}).get("hosts") or ()),
        "route": contract.ISOLATION_ROUTE,
        "request_sha256": str(document.get("request_sha256")),
        "hmz_version": str((document.get("hmz") or {}).get("version")),
        "policy_sha256": hashlib.sha256(
            json.dumps(document.get("policy"), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


@dataclass(frozen=True)
class SourceOnlyBuildOutcome:
    """One verified source-only build (a build record, never a schema-4 evaluation)."""

    identity: contract.SourceOnlyBuildIdentity
    staged_frontend: StagedFrontend
    attempt: Mapping[str, Any]
    entry_sha256: str
    attempt_root: Path

    def document(self) -> dict[str, Any]:
        """@brief 构建记录（无 dispatch、无 schema-4 报告）。

        @return 字段字典。
        """
        return {
            "schema_id": "tcenv.track.tilelang.source-only-build.1",
            "track": {"track_id": contract.TILELANG_TRACK_ID, "track_version": contract.TILELANG_TRACK_VERSION},
            "build_mode": contract.SOURCE_ONLY_BUILD_MODE,
            "identity": self.identity.document(),
            "identity_digest": self.identity.digest,
            "cache_key": self.identity.cache_key,
            "full_build": self.identity.full_build,
            "frontend_entry_sha256": self.entry_sha256,
            "staged_frontend": self.staged_frontend.document(),
            "attempt": dict(self.attempt),
            "device_dispatch_calls": 0,
            "publication_allowed": False,
            "note": (
                "source-only build: no Metal library, pipeline, buffer or dispatch exists; this record is not an "
                "evaluation result and cannot be published as a speedup"
            ),
        }


class TileLangSourceOnlyBuild:
    """Run one real source-only build inside the existing lower-level hmz Profile."""

    def __init__(
        self,
        track: TrackSpec,
        *,
        attempt_base: Path | None = None,
        staging: Path | None = None,
        executor: HmzFenceExecutor | None = None,
        limits: ResourceLimits | None = None,
    ) -> None:
        """@brief 装配一次构建。

        @param track 已登记的 TileLang Track（先做注册⇄合同核对）。
        @param attempt_base 可写尝试根目录的父目录；None 时用系统临时目录。
        @param staging staged frontend 根目录；None 时读环境变量/MT0 默认。
        @param executor 围栏执行器（测试可注入）；None 时用真实 hmz 路线。
        @param limits 资源上限；None 时用本模块的 source-only 默认值。
        """
        contract.audit_registration(track)
        self._track = track
        self._attempt_base = Path(attempt_base) if attempt_base is not None else None
        self._staging = staging
        self._executor = executor
        self._limits = limits or ResourceLimits(
            cpu_seconds=600,
            wall_seconds=600,
            file_bytes=64 * 1024 * 1024,
            artifact_total_bytes=128 * 1024 * 1024,
            open_files=512,
            address_space_bytes=None,
            processes=256,
        )

    @property
    def entry_sha256(self) -> str:
        """@brief 围栏内 entry 的摘要。

        @return 十六进制 sha256。
        """
        return FRONTEND_ENTRY_SHA256

    def build(self, candidate: Any, *, expected_kernel_names: tuple[str, ...] = ()) -> SourceOnlyBuildOutcome:
        """@brief 真实 source-only 构建：观测前端、围栏内 codegen、核验工件、派生身份。

        @param candidate 投影候选（``CandidateArtifact``）。
        @param expected_kernel_names 候选几何声明的内核名。
        @return 构建记录。
        @raises contract.TileLangContractError 候选/工件/尝试不合法。
        @raises contract.TileLangIdentityDrift 前端安装与冻结身份不符。
        @raises contract.TileLangFrontendUnavailable 前端/hmz 能力缺失（typed capability gap）。
        @raises contract.TileLangCodegenError codegen 超时/非零。
        """
        contract.validate_candidate(candidate, self._track)
        source = [entry for entry in candidate.entries if entry.role == contract.SOURCE_ROLE][0].data
        frontend = observe_staged_frontend(self._staging)
        environment = observe_staged_environment(frontend)
        root = Path(tempfile.mkdtemp(prefix="shapeforge-tilelang-", dir=self._attempt_base))
        input_dir = root / "input"
        artifacts_dir = root / "build" / "artifacts"
        spec = AttemptSpec(
            attempt_root=root,
            argv=(
                str(frontend.python),
                str(input_dir / FRONTEND_ENTRY_NAME),
                "--candidate",
                str(input_dir / CANDIDATE_NAME),
                "--pinned",
                str(input_dir / PINNED_NAME),
                "--artifacts",
                str(artifacts_dir),
            ),
            artifacts=contract.SOURCE_ONLY_ARTIFACT_PATHS,
            input_files={
                FRONTEND_ENTRY_NAME: FRONTEND_ENTRY_SOURCE.encode("utf-8"),
                PINNED_NAME: (
                    json.dumps(
                        frontend.pinned_document(hashlib.sha256(bytes(source)).hexdigest()),
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                ).encode("utf-8"),
                CANDIDATE_NAME: bytes(source),
            },
            read_only_roots=(frontend.root,),
            python_path=(frontend.site_packages,),
            limits=self._limits,
        )
        executor = self._executor or HmzFenceExecutor()
        try:
            result = executor.run(spec)
        except HmzCapabilityUnavailable as exc:
            raise contract.TileLangFrontendUnavailable(
                f"the bounded build route is unavailable: {exc}"
            ) from exc
        except ArtifactContractError as exc:
            raise contract.TileLangContractError(str(exc)) from exc
        attempt = _attempt_record(result)
        if attempt["status"] == "timeout" or attempt["timed_out"]:
            raise contract.TileLangCodegenError("the source-only codegen attempt timed out")
        if attempt["returncode"] != 0:
            raise self._codegen_error(attempt["returncode"], root)
        artifacts = _artifact_bytes(artifacts_dir)
        identity = contract.source_only_build_identity(
            candidate=candidate,
            observed_frontend=frontend.observation(),
            environment=environment,
            artifacts=artifacts,
            attempt=attempt,
            expected_kernel_names=expected_kernel_names,
            track_version=self._track.track_version,
        )
        expected = contract.expected_build_subject(
            candidate=candidate, environment=environment, track_version=self._track.track_version
        )
        if identity.cache_key != expected.cache_key:
            raise contract.TileLangIdentityDrift(
                "the built identity does not bind the expected candidate/frontend/environment subject"
            )
        return SourceOnlyBuildOutcome(
            identity=identity,
            staged_frontend=frontend,
            attempt=attempt,
            entry_sha256=FRONTEND_ENTRY_SHA256,
            attempt_root=root,
        )

    @staticmethod
    def _codegen_error(returncode: int, root: Path) -> Exception:
        stderr = ""
        for name in ("stderr.txt",):
            path = root / name
            if path.is_file():
                stderr = path.read_text(encoding="utf-8", errors="replace")[-800:]
        if returncode == EXIT_IDENTITY_DRIFT:
            return contract.TileLangIdentityDrift(f"the in-fence frontend identity drifted: {stderr}")
        if returncode == EXIT_DEVICE_REQUIRED:
            return contract.TileLangCodegenError(
                f"the frontend required a device during source-only codegen: {stderr}"
            )
        if returncode == EXIT_UNSUPPORTED:
            return contract.TileLangContractError(f"the candidate/ frontend combination is unsupported: {stderr}")
        return contract.TileLangCodegenError(f"the source-only codegen exited {returncode}: {stderr}")


__all__ = [
    "FRONTEND_ENTRY_SHA256",
    "FRONTEND_ENTRY_SOURCE",
    "SourceOnlyBuildOutcome",
    "StagedFrontend",
    "TileLangSourceOnlyBuild",
    "observe_staged_environment",
    "observe_staged_frontend",
    "staging_root",
]
