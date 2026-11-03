"""Cross-hardware and expressiveness boundary of the A2 production surface (CPU).

A2 only travels on the Mac/清微-neutral path: the ABI deriver, typed geometry, canonical bundle,
lifecycle, protocol and the MSL reflection binding adapter must not mention a rejected vendor's
code/toolchain knowledge and must not import a backend/host harness.  They also must not carry a
v2 loop/``goto`` style review: candidate expressiveness is owned by the candidate, and a runaway
launch is contained by the external watchdog, not by a source-pattern gate in the delivery plane.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tcenv.track import CandidateArtifact, CandidateEntry, load_track_registry

ROOT = Path(__file__).resolve().parents[2]
A2_SOURCES = (
    ROOT / "tcenv" / "track" / "abi.py",
    ROOT / "tcenv" / "track" / "bundle.py",
    ROOT / "tcenv" / "track" / "expr.py",
    ROOT / "tcenv" / "track" / "geometry.py",
    ROOT / "tcenv" / "track" / "lifecycle.py",
    ROOT / "tcenv" / "track" / "protocol.py",
    ROOT / "tcenv" / "backends" / "metal" / "stage_binding.py",
    ROOT / "backends" / "metal" / "build_profiles.json",
)
REJECTED_PLATFORM = re.compile(
    r"\b(cuda|cann|ascend[a-z]*|msopgen|msprof|easyasc|acl|npu|torch_npu|america|nvidia)\b", re.IGNORECASE
)
#: A v2 candidate may write any of these; the delivery plane must not screen them.
CANDIDATE_FREEDOM = (
    b"kernel void k(device half *x) {\n  int i = 0;\nloop:\n  if (i++ < 4) goto loop;\n}\n",
    b"kernel void k(device half *x) {\n  while (true) { if (x[0] > 0) break; }\n}\n",
    b"kernel void k(device half *x) {\n  for (;;) { break; }\n}\n",
    b"kernel void k(device half *x) {\n  // persistent state machine across dispatches\n  switch (x[0]) { default: break; }\n}\n",
)


def test_a2_production_mentions_no_rejected_hardware_platform():
    """@brief A2 生产代码/配置不携带 CUDA/CANN/Ascend 平台名称或专用知识。"""
    offenders: list[str] = []
    for path in A2_SOURCES:
        text = path.read_text(encoding="utf-8")
        for match in REJECTED_PLATFORM.finditer(text):
            offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
    assert offenders == [], "\n".join(offenders)


def test_a2_modules_do_not_import_a_backend_or_the_host_harness():
    """@brief A2 通用层只依赖通用层；Metal 反射适配器不 import taic/tasks/tests/候选。"""
    forbidden_roots = {"backends", "taic", "tasks", "tests", "torch_npu", "acl"}
    offenders: list[str] = []
    for path in A2_SOURCES:
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.lstrip(".").split(".")[0]
                if root in forbidden_roots:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert offenders == [], "\n".join(offenders)


def test_the_v2_delivery_plane_has_no_loop_or_goto_style_review():
    """@brief v2 循环/goto/状态机不被投递面审查：同字节候选一律 accepted。

    The v1 pre-screen (``tcenv.backends.metal.screen``) stays a v1 defence.  The v2 plane must not
    import it and must not grow its own pattern gate: the candidate owns the kernel form, and a
    runaway launch is stopped by the external watchdog.
    """
    registry = load_track_registry()
    for source in CANDIDATE_FREEDOM:
        for track_id, roles in (
            ("msl-single", ("device_kernel",)),
            ("msl-bundle", ("device_kernel", "host_tiling")),
        ):
            entries = tuple(
                CandidateEntry(
                    path=f"{role}.metal", role=role, media_type="text/plain", data=source if index == 0 else b"// t\n"
                )
                for index, role in enumerate(roles)
            )
            candidate = CandidateArtifact(track_id=track_id, entries=entries)
            assert candidate.validate_for(registry.get(track_id)) == "accepted"

    for path in A2_SOURCES:
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "screen_source" not in text, f"{path.relative_to(ROOT)} imports the v1 style screen"
        assert "from .screen" not in text and "import screen" not in text
