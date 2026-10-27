#!/usr/bin/env python3
"""Reproducibly generate and drift-check the external evaluator Protobuf files.

Pinned generators: protoc 29.3 and grpc_python_plugin 1.69.0.  The 1.69
plugin predates self-reported stub versions, so CI must install that pinned
plugin and this script enforces byte-for-byte output.  Only the three generated
artifacts are replaced; package __init__.py files remain untouched.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROTO_REL = Path("tcenv/remote/v1/evaluator.proto")
PROTOC_VERSION = "29.3"
GRPC_PLUGIN_VERSION = "1.69.0"
#: Every checked-in Protobuf artifact and the proto it is generated from.  The
#: blob channel is a separate proto package on purpose: it must not add an RPC
#: or reuse a message from the six-RPC Evaluator control plane.
PROTO_TARGETS = (
    {
        "proto": "tcenv/remote/v1/evaluator.proto",
        "package_dir": "tcenv/remote/v1",
        "stem": "evaluator",
        "descriptor": "proto/tcenv/remote/v1/evaluator.desc",
    },
    {
        "proto": "tcenv/remote/blob/v1/blob.proto",
        "package_dir": "tcenv/remote/blob/v1",
        "stem": "blob",
        "descriptor": "proto/tcenv/remote/blob/v1/blob.desc",
    },
)


def _tool(name: str, override: str | None = None) -> str:
    path = override or shutil.which(name)
    if not path:
        raise RuntimeError(f"required generator {name!r} was not found on PATH")
    return str(path)


def _generate_one(
    output: Path, protoc: str, plugin: str, target: dict[str, str],
) -> dict[Path, bytes]:
    package_dir = output / target["package_dir"]
    package_dir.mkdir(parents=True, exist_ok=True)
    descriptor = output / f"{target['stem']}.desc"
    command = [
        protoc,
        f"--plugin=protoc-gen-grpc_python={plugin}",
        "-I",
        "proto",
        f"--python_out={output}",
        f"--grpc_python_out={output}",
        f"--descriptor_set_out={descriptor}",
        "--include_imports",
        target["proto"],
    ]
    env = dict(os.environ, LC_ALL="C")
    subprocess.run(command, cwd=ROOT, check=True, env=env)
    stem = target["stem"]
    return {
        ROOT / target["package_dir"] / f"{stem}_pb2.py": (package_dir / f"{stem}_pb2.py").read_bytes(),
        ROOT / target["package_dir"] / f"{stem}_pb2_grpc.py": (package_dir / f"{stem}_pb2_grpc.py").read_bytes(),
        ROOT / target["descriptor"]: descriptor.read_bytes(),
    }


def _generate(output: Path) -> dict[Path, bytes]:
    protoc = _tool("protoc")
    plugin = _tool("grpc_python_plugin", os.environ.get("TCENV_GRPC_PYTHON_PLUGIN"))
    version = subprocess.run(
        [protoc, "--version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if version != f"libprotoc {PROTOC_VERSION}":
        raise RuntimeError(f"need libprotoc {PROTOC_VERSION}, found {version!r}")

    generated: dict[Path, bytes] = {}
    for target in PROTO_TARGETS:
        generated.update(_generate_one(output, protoc, plugin, target))
    stub = generated[ROOT / "tcenv" / "remote" / "v1" / "evaluator_pb2_grpc.py"].decode("utf-8")
    match = re.search(r"^GRPC_GENERATED_VERSION = '([^']+)'$", stub, re.MULTILINE)
    if match is not None and match.group(1) != GRPC_PLUGIN_VERSION:
        raise RuntimeError(
            f"need grpc_python_plugin {GRPC_PLUGIN_VERSION}, generated stub reports {match.group(1)}"
        )
    # grpc_python_plugin 1.69.0 emits no GRPC_GENERATED_VERSION marker.  Its
    # package version is a CI/toolchain requirement; exact output is checked
    # below regardless of whether a future plugin adds a marker.
    return generated


def _report_text_difference(path: Path, expected: bytes, actual: bytes) -> None:
    before = expected.decode("utf-8").splitlines(keepends=True)
    after = actual.decode("utf-8").splitlines(keepends=True)
    sys.stdout.writelines(
        difflib.unified_diff(before, after, fromfile=f"checked-in/{path}", tofile=f"generated/{path}")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if checked-in artifacts differ")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="tcenv-evaluator-proto-") as temporary:
        generated = _generate(Path(temporary))

    drifted = []
    for path, actual in generated.items():
        if args.check:
            if not path.is_file():
                drifted.append(path)
                print(f"missing generated artifact: {path.relative_to(ROOT)}")
                continue
            expected = path.read_bytes()
            if expected == actual:
                continue
            drifted.append(path)
            if path.suffix == ".py":
                _report_text_difference(path.relative_to(ROOT), expected, actual)
            else:
                print(
                    f"descriptor drift: {path.relative_to(ROOT)} "
                    f"checked-in sha256={hashlib.sha256(expected).hexdigest()} "
                    f"generated sha256={hashlib.sha256(actual).hexdigest()}"
                )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(actual)
            print(f"wrote {path.relative_to(ROOT)}")

    if drifted:
        print("evaluator Protobuf artifacts are stale; run scripts/generate_evaluator_proto.py", file=sys.stderr)
        return 1
    if args.check:
        print("evaluator Protobuf generated artifacts are up to date")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"evaluator proto generation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
