"""Test-only fake TSM-style module toolchain and ``ArtifactBuilder``.

This is the P1c stand-in for a real ``TsmModuleBuilder``: a deterministic,
registered build profile that turns a small kernel source into a fake
``TXELF`` module plus the source it consumed.  It is **not** a real compiler, a
real TX ABI or a performance claim: the module is a checksummed container with a
deterministic debug section so the executable entry crosses the inline/blob
boundary (the control-host build is transported through the real BlobChannel).

Kept in the test support tree only.  A production composition root registers
real builders; this module is never imported by one.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import replace
from typing import Any, Mapping

from tcenv.remote.artifact_builder import (
    ArtifactBuilder,
    BuildArtifact,
    BuildCapabilities,
    BuildError,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
    SourceBundle,
)
from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry

BUILDER_ID = "fake-tsm-module-builder"
BUILDER_VERSION = "1.4.0"
TOOLCHAIN_ID = "fake-tx-toolchain"
TOOLCHAIN_VERSION = "0.9.4"
SDK_VERSION = "0.9.4-sim"
MODULE_MEDIA_TYPE = "application/x-tx-elf"
MODULE_MAGIC = b"\x7fTXELF\x02\x01\x01\x00SIMT\x00"
#: The debug build pads the module with a deterministic debug section so the
#: executable entry is larger than the bundle inline limit and must travel
#: through the content-addressed blob channel.
MODULE_DEBUG_SIZE = 327_680
#: Marker a test source can carry to make the fake toolchain fail to compile.
COMPILE_FAIL_MARKER = b"TXCC_FAIL"

CONTROL_HOST_COMPILE_PROFILE = {
    "opt_level": "O2",
    "target_abi": "tx-fake-elf-v1",
    "debug_sections": "on",
}
EVALUATOR_COMPILE_PROFILE = {
    "opt_level": "O2",
    "target_abi": "tx-fake-elf-v1",
    "debug_sections": "off",
}


def control_host_profile(**overrides: Any) -> BuildProfile:
    """The registered profile whose build runs on the control host (prebuilt)."""

    values = {
        "build_profile": "tsm-module-debug-v1",
        "builder_id": BUILDER_ID,
        "builder_version": BUILDER_VERSION,
        "build_host": "control_host",
        "toolchain_id": TOOLCHAIN_ID,
        "toolchain_version": TOOLCHAIN_VERSION,
        "sdk_version": SDK_VERSION,
        "compile_profile": dict(CONTROL_HOST_COMPILE_PROFILE),
    }
    values.update(overrides)
    return BuildProfile(**values)


def evaluator_profile(**overrides: Any) -> BuildProfile:
    """The registered profile whose build runs inside the evaluator attempt."""

    values = {
        "build_profile": "tsm-module-eval-v1",
        "builder_id": BUILDER_ID,
        "builder_version": BUILDER_VERSION,
        "build_host": "evaluator",
        "toolchain_id": TOOLCHAIN_ID,
        "toolchain_version": TOOLCHAIN_VERSION,
        "sdk_version": SDK_VERSION,
        "compile_profile": dict(EVALUATOR_COMPILE_PROFILE),
    }
    values.update(overrides)
    return BuildProfile(**values)


def _debug_padding(size: int, digest: str) -> bytes:
    """Deterministic non-zero debug bytes (a truncation changes the digest)."""

    produced = bytearray()
    seed = digest.encode("ascii")
    while len(produced) < size:
        seed = hashlib.sha256(seed).digest()
        produced += seed
    return bytes(produced[:size])


def compile_module(source: bytes, *, debug_sections: bool) -> bytes:
    """The deterministic fake toolchain output (not a real TX ELF/ABI)."""

    digest = hashlib.sha256(b"fake-tx-module\0" + source).hexdigest()
    body = bytes.fromhex(digest) + source
    header = MODULE_MAGIC + struct.pack("<Q", len(source)) + struct.pack("<Q", len(body))
    if not debug_sections:
        return header + body + bytes.fromhex(digest)
    remaining = MODULE_DEBUG_SIZE - len(header) - len(body) - 32
    if remaining < 0:
        raise ValueError("source is too large for the fake debug module")
    footer = hashlib.sha256(header + body).digest()
    return header + body + footer + _debug_padding(remaining, digest)


def module_sha256(module: bytes) -> str:
    return hashlib.sha256(module).hexdigest()


class FakeTsmModuleBuilder:
    """A registered, deterministic builder for the fake TX toolchain."""

    def __init__(
        self, profile: BuildProfile, *, available: bool = True, detail: str = "",
        reported_version: str | None = None,
        build_reports_version: str | None = None,
    ) -> None:
        self.profile = profile
        self.available = bool(available)
        self.detail = detail
        #: A test seam: advertise a different toolchain identity than the one
        #: the request names, so the profile probe can be proven to gate Submit.
        self.reported_version = reported_version
        #: A test seam: produce an outcome whose provenance claims another
        #: builder version than the registered profile, so the build-side
        #: identity check can be proven to reject a mismatched provenance.
        self.build_reports_version = build_reports_version
        self.builds = 0
        self.sources: list[bytes] = []

    def probe(self) -> BuildCapabilities:
        profile = self.profile
        if self.reported_version is not None:
            profile = replace(profile, builder_version=self.reported_version)
        return BuildCapabilities(profile=profile, available=self.available, detail=self.detail)

    def build(
        self, source_bundle: SourceBundle, *, task_contract: Mapping[str, Any],
        deadline_monotonic: float | None = None,
    ) -> BuildOutcome:
        started = time.monotonic()
        if deadline_monotonic is not None and started >= deadline_monotonic:
            raise BuildError("deadline", "the build deadline expired before compilation")
        self.builds += 1
        self.sources.append(source_bundle.source)
        source = source_bundle.source
        if COMPILE_FAIL_MARKER in source:
            log = (
                b"--txcc: compiling kernel.tx\n"
                b"--txcc: error: undefined symbol 'tx_fail' in kernel.tx\n"
                b"--txcc: 1 error generated\n"
            )
            raise BuildError("compile", "fake txcc rejected the source", log=log, duration_ms=1)
        debug = self.profile.compile_profile.get("debug_sections") == "on"
        module = compile_module(source, debug_sections=debug)
        log = (
            b"--txcc: compiling kernel.tx\n"
            + f"--txcc: emitted module.elf {len(module)} bytes\n".encode("ascii")
        )
        from tcenv.remote.artifact_builder import make_blob_entry

        payloads: dict[str, bytes] = {}
        if len(module) > 262_144:
            # A debug module crosses the inline limit: the artifact set states
            # its content identity and the transport uploads the bytes.
            module_entry = make_blob_entry(
                path="module/kernel.txelf", role="EXECUTABLE_MODULE",
                media_type=MODULE_MEDIA_TYPE, data=module, chunk_size=8192,
            )
            payloads["module/kernel.txelf"] = module
        else:
            module_entry = BundleEntry(
                path="module/kernel.txelf", role="EXECUTABLE_MODULE",
                media_type=MODULE_MEDIA_TYPE, size_bytes=len(module),
                sha256=module_sha256(module), data=module,
            )
        entries = (source_bundle.entry(), module_entry)
        bundle = ArtifactBundle(entries=entries)
        provenance_profile = self.profile
        if self.build_reports_version is not None:
            provenance_profile = replace(
                provenance_profile, builder_version=self.build_reports_version,
            )
        provenance = BuildProvenance(
            profile=provenance_profile,
            source_sha256=source_bundle.sha256,
            artifact_set_digest=bundle.artifact_set_digest,
        )
        evidence = BuildEvidence(
            mode="prebuilt" if self.profile.build_host == "control_host" else "evaluator",
            status="ok",
            provenance=provenance,
            artifacts=tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries),
            duration_ms=max(1, int((time.monotonic() - started) * 1000)),
            log=log,
            placement=(
                "control_host" if self.profile.build_host == "control_host" else "evaluator"
            ),
        )
        outcome = BuildOutcome(
            bundle=bundle, provenance=provenance, evidence=evidence, payloads=payloads,
        )
        if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
            raise BuildError("deadline", "the build did not finish before the attempt deadline")
        return outcome


def builder_for_provenance(
    provenance: Mapping[str, Any],
) -> tuple[FakeTsmModuleBuilder, ArtifactBuilder]:
    """Return the fake builder a test worker uses for one build identity.

    The worker resolves the *registered* profile by identity, exactly like a
    production composition root would; an unknown builder id is a typed failure,
    never an arbitrary toolchain invocation.
    """

    if provenance.get("builder_id") != BUILDER_ID:
        raise BuildError(
            "toolchain_profile", f"builder {provenance.get('builder_id')!r} is not registered"
        )
    host = provenance.get("build_host")
    compile_profile = dict(provenance.get("compile_profile") or {})
    profile = (
        control_host_profile(compile_profile=compile_profile) if host == "control_host"
        else evaluator_profile(compile_profile=compile_profile)
    )
    builder = FakeTsmModuleBuilder(profile)
    return builder, builder


def request_build(
    profile: BuildProfile, *, source: bytes, mode: str, artifact_set_digest: str | None,
    producer: Mapping[str, Any] | None = None,
):
    """The typed NpuBuildRequest for one mode (test helper)."""

    from tcenv.remote.artifact_builder import BuildProvenance
    from tcenv.remote.job_types import NpuBuildRequest

    provenance = BuildProvenance(
        profile=profile, source_sha256=hashlib.sha256(source).hexdigest(),
        artifact_set_digest=artifact_set_digest,
    )
    document = provenance.request_document(mode=mode)
    if producer is not None:
        document["producer"] = dict(producer)
    return NpuBuildRequest.from_dict(document)
