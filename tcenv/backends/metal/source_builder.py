"""The real Metal ``ArtifactBuilder``: MSL -> library + compute pipeline.

Metal is the *local build-and-execute* placement: the control host compiles the
candidate MSL and keeps the resulting ``MTLLibrary``/``MTLComputePipelineState``
in this process.  This builder is a production implementation of the unified
``ArtifactBuilder`` contract, not a mock and not a concept adapter:

* it calls the existing runtime path (``MetalBackend.assemble`` then
  ``MetalBackend.compile`` -> ``MetalContext.compile``), so the compile/pipeline
  behaviour, the ``MetalCompileError`` classification and the measured
  ``compile_s``/``pipeline_s``/``language_version`` values are the ones the Metal
  evaluator already reports;
* a candidate compile failure is a typed ``compile`` failure that keeps the
  compiler's own diagnostic as a bounded log *and* the real elapsed compile
  time, so a failed build is never a bare one-liner with ``duration_ms=0``;
* the executable is expressed as a backend-native artifact
  (``NativeExecutable``: a live pipeline handle plus a canonical descriptor and
  digest), never as fabricated module bytes and never as an NPU artifact set;
* ``candidate_sha256`` is untouched: the ``SOURCE`` entry is the exact candidate
  bytes (``sha256`` of them), while the compiler input that carries the contract
  header is recorded separately in the descriptor.

Nothing here changes the Metal report schema, the local execution path or the
publication rules.  The builder produces provenance/evidence and the local
caller keeps executing through the existing evaluator.
"""

from __future__ import annotations

import hashlib
import platform
import time
from typing import Any, Mapping

from ...remote.artifact_builder import (
    BuildArtifact,
    BuildCapabilities,
    BuildError,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
    SourceBundle,
)
from ...remote.artifact_bundle import ArtifactBundle, NativeExecutable

METAL_BUILDER_ID = "metal-source-builder"
METAL_TOOLCHAIN_ID = "apple-metal"
METAL_PROFILE_NAME = "metal-msl-control-host-v1"
#: A compile-failure log is a diagnostic, not an artifact: it is bounded so a
#: pathological compiler message cannot grow the attempt's memory without limit.
MAX_COMPILE_FAILURE_LOG_BYTES = 64 * 1024


def _compile_failure_log(
    *, kernel_name: str, math: str, device: str, language: str,
    header_lines: int, assembled_sha256: str, error_name: str,
    message: str, raw_message: str, compile_s: float,
) -> bytes:
    """The real compiler diagnostic of a failed Metal build, as a durable log.

    The successful path records ``compile_s``/``pipeline_s``/``language`` and the
    assembled digest in its log; a failed compilation has to carry the same kind
    of provenance or the only record of *why* the candidate failed would be a
    truncated one-line message.  The diagnostic is the compiler's own text (the
    rewritten form the evaluator reports, plus the raw text when the rewrite
    changed it), never a summary invented here.
    """

    lines = [
        f"metal: kernel={kernel_name} math={math} device={device} "
        f"language={language} header_lines={header_lines} "
        f"assembled_sha256={assembled_sha256}\n",
        f"metal: status=compile_failed error={error_name} "
        f"compile_s={compile_s:.6f} pipeline=not_run\n",
        "metal: diagnostic:\n",
        message if message.endswith("\n") else f"{message}\n",
    ]
    if raw_message and raw_message != message:
        lines.append("metal: raw diagnostic:\n")
        lines.append(raw_message if raw_message.endswith("\n") else f"{raw_message}\n")
    log = "".join(lines).encode("utf-8", "replace")
    if len(log) > MAX_COMPILE_FAILURE_LOG_BYTES:
        # Keep the head (identity + first diagnostics) and say it was cut.
        marker = b"\nmetal: diagnostic truncated by the builder\n"
        log = log[: MAX_COMPILE_FAILURE_LOG_BYTES - len(marker)] + marker
    return log


def _binding_document(binding: Any) -> dict[str, Any]:
    """One reflected binding as a canonical, digestible row (no live objects)."""

    return {
        "name": str(getattr(binding, "name", "")),
        "index": int(getattr(binding, "index", 0)),
        "kind": str(getattr(binding, "kind", "")),
        "dtype_code": (
            None if getattr(binding, "dtype_code", None) is None
            else int(binding.dtype_code)
        ),
        "access": str(getattr(binding, "access", "")),
        "elem_size": (
            None if getattr(binding, "elem_size", None) is None
            else int(binding.elem_size)
        ),
        "used": bool(getattr(binding, "used", False)),
        "tg_bytes": (
            None if getattr(binding, "tg_bytes", None) is None
            else int(binding.tg_bytes)
        ),
    }


class MetalSourceBuilder:
    """Compile MSL with the real Metal runtime and describe the built pipeline.

    ``kernel_contract`` is the Metal task contract (``tcenv.contract.Contract``):
    the builder needs ``kernel_name`` to assemble the contract header and to
    create the pipeline, exactly like the evaluator.  A bounded mapping with a
    ``kernel_name`` key is accepted too, so a composition root can hand over the
    same task identity through the generic builder call without inventing a
    second contract type.
    """

    def __init__(
        self, *, backend: Any | None = None, kernel_contract: Any | None = None,
        math: str = "fast",
    ) -> None:
        if math not in ("fast", "safe", "relaxed"):
            raise ValueError("Metal math mode must be fast, safe or relaxed")
        if backend is None:
            from .backend import MetalBackend

            backend = MetalBackend()
        self.backend = backend
        self.kernel_contract = kernel_contract
        self.math = math
        self._profile: BuildProfile | None = None
        self._info: dict[str, Any] | None = None

    # -- probe -------------------------------------------------------------
    def probe(self) -> BuildCapabilities:
        """Report the identity of the Metal toolchain on *this* machine.

        The probe allocates the real context (that is what "can this host build"
        means) but never compiles a candidate.  ``MetalContext.info()`` calls
        ``language_version()``, which compiles one *fixed* two-line MSL library
        in order to pick the highest MSL version this device accepts; that is a
        version probe (library only, no pipeline, no candidate bytes, no kernel
        launch), not a build.  It needs the same Metal device access a build
        needs, so on a host without a device the profile is unavailable, which is
        a pre-Submit refusal, never a compile failure attributed to the candidate.
        """

        try:
            info = self._device_info()
        except Exception as exc:  # noqa: BLE001 - no device is an environment fact
            return BuildCapabilities(
                profile=self._fallback_profile(),
                available=False,
                detail=f"Metal device is unavailable: {type(exc).__name__}: {exc}"[:400],
            )
        return BuildCapabilities(profile=self._profile_for(info), available=True)

    # -- build -------------------------------------------------------------
    def build(
        self, source_bundle: SourceBundle, *, task_contract: Any,
        deadline_monotonic: float | None = None,
    ) -> BuildOutcome:
        """Assemble, compile and create the pipeline through the Metal runtime."""

        started = time.monotonic()
        if deadline_monotonic is not None and started >= deadline_monotonic:
            raise BuildError("deadline", "the Metal compile deadline expired before compilation")
        contract = task_contract if task_contract is not None else self.kernel_contract
        kernel_name = self._kernel_name(contract)
        try:
            text = source_bundle.source.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Metal consumes text; non-UTF-8 source cannot even reach the
            # compiler, so it is a candidate compile failure (the legacy
            # evaluator reports it as compile_error).
            raise BuildError("compile", f"Metal source is not UTF-8 text: {exc}") from exc
        try:
            info = self._device_info()
        except Exception as exc:  # noqa: BLE001 - an absent device is not the candidate's fault
            raise BuildError(
                "toolchain_profile",
                f"the Metal toolchain is unavailable on this host: {type(exc).__name__}: {exc}"[:400],
            ) from exc
        profile = self._profile_for(info)
        self._check_probed_identity(profile)
        try:
            assembled, header_lines = self.backend.assemble(contract, text)
        except Exception as exc:  # noqa: BLE001 - assembly is pure string work
            raise BuildError(
                "toolchain_profile", f"the Metal contract header could not be assembled: {exc}"[:400],
            ) from exc
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise BuildError("deadline", "the Metal compile deadline expired before compiling")
        context = self.backend.context()
        # Only the backend's own compile error is a *candidate* failure.  The
        # legacy evaluator catches exactly ``backend.compile_error()`` too; an
        # unexpected exception (device lost, driver bug, MemoryError) must stay
        # an infrastructure failure and must not be blamed on the candidate.
        compile_error = self.backend.compile_error()
        compile_started = time.monotonic()
        try:
            compiled = self.backend.compile(context, assembled, contract, math=self.math)
        except compile_error as exc:
            # The legacy evaluator turns this same exception into
            # `status=compile_error`, so the unified classification is `compile`
            # and never infrastructure/timeout.  The report text is unchanged
            # (``type: message``, rewritten compiler lines), but the failure now
            # also carries the real diagnostic as a bounded log and the real
            # elapsed time, so a compile failure is not a bare one-liner with
            # ``duration_ms=0``.
            raw_message = str(exc)
            message = raw_message
            rewrite = getattr(self.backend, "rewrite_compiler_lines", None)
            if callable(rewrite):
                try:
                    message = rewrite(contract, message, header_lines)
                except Exception:  # noqa: BLE001 - diagnostics must not mask the failure
                    message = raw_message
            elapsed_s = max(0.0, time.monotonic() - compile_started)
            raise BuildError(
                "compile", f"{type(exc).__name__}: {message}"[:500],
                log=_compile_failure_log(
                    kernel_name=kernel_name, math=self.math,
                    device=str(info.get("name") or "unknown"),
                    language=str(info.get("language_version") or "unknown"),
                    header_lines=header_lines,
                    assembled_sha256=hashlib.sha256(assembled.encode("utf-8")).hexdigest(),
                    error_name=type(exc).__name__, message=message,
                    raw_message=raw_message, compile_s=elapsed_s,
                ),
                # A build that ran is at least one millisecond: ``0`` is the
                # contract's "did not run" duration and must not be reused for
                # a real failure.
                duration_ms=max(1, int(round(elapsed_s * 1000))),
            ) from exc
        compiled_language = str(getattr(compiled, "language_version", "") or "")
        if compiled_language and profile.toolchain_version not in ("unknown", compiled_language):
            # The compiler used a different language version than the identity
            # the probe advertised: the build environment changed, so the result
            # is not comparable and must not be reported as a fresh baseline.
            raise BuildError(
                "toolchain_profile",
                "the Metal compiler used MSL "
                f"{compiled_language} but the probed profile declares {profile.toolchain_version}",
            )
        if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
            raise BuildError("deadline", "the Metal build did not finish before the attempt deadline")

        compile_s = float(getattr(compiled, "compile_s", 0.0) or 0.0)
        pipeline_s = float(getattr(compiled, "pipeline_s", 0.0) or 0.0)
        entry = source_bundle.entry()
        native = NativeExecutable(
            backend_id="metal",
            kind="compute-pipeline",
            entry_path=entry.path,
            entry_sha256=entry.sha256,
            descriptor={
                "backend": "metal",
                "compiler": "apple-metal-msl",
                "kernel_name": kernel_name,
                "language_version": str(getattr(compiled, "language_version", "") or "unknown"),
                "math": self.math,
                "device": str(info.get("name") or "unknown"),
                "gpu_family": str(info.get("gpu_family") or "unknown"),
                "max_total_threads": int(getattr(compiled, "max_total_threads", 0) or 0),
                "execution_width": int(getattr(compiled, "execution_width", 0) or 0),
                "static_tg_bytes": int(getattr(compiled, "static_tg_bytes", 0) or 0),
                "assembled_sha256": hashlib.sha256(assembled.encode("utf-8")).hexdigest(),
                "header_lines": int(header_lines),
                "bindings": [
                    _binding_document(binding)
                    for binding in (getattr(compiled, "bindings", None) or ())
                ],
            },
            handle=compiled,
        )
        bundle = ArtifactBundle(entries=(entry,), native=native)
        provenance = BuildProvenance(
            profile=profile, source_sha256=source_bundle.sha256,
            artifact_set_digest=bundle.artifact_set_digest,
        )
        log = (
            f"metal: kernel={kernel_name} compile_s={compile_s:.6f} "
            f"pipeline_s={pipeline_s:.6f} language={compiled_language or 'unknown'} "
            f"math={self.math}\n"
            f"metal: assembled bytes={len(assembled)} header_lines={header_lines} "
            f"sha256={native.descriptor['assembled_sha256']}\n"
            f"metal: pipeline={native.digest} bindings={len(native.descriptor['bindings'])} "
            f"max_threads={native.descriptor['max_total_threads']} "
            f"thread_execution_width={native.descriptor['execution_width']} "
            f"tg_bytes={native.descriptor['static_tg_bytes']}\n"
        ).encode("utf-8")
        evidence = BuildEvidence(
            # Local placement: the control host builds and then executes in the
            # same process.  `prebuilt` names the build host; `ok` names a build
            # this host performed.  (The NPU validator additionally refuses this
            # combination for a remote row, because there a `prebuilt` artifact
            # must arrive as `reused`.)
            mode="prebuilt",
            status="ok",
            provenance=provenance,
            artifacts=(BuildArtifact.from_entry(entry),),
            duration_ms=max(0, int(round((compile_s + pipeline_s) * 1000))),
            log=log,
            # A producer has not loaded the artifact onto a device; the local
            # executor is the same process and runs the live handle directly.
            loaded=(),
            native=native,
            placement="control_host",
        )
        return BuildOutcome(
            bundle=bundle, provenance=provenance, evidence=evidence, payloads={},
        )

    # -- identity ----------------------------------------------------------
    def _kernel_name(self, contract: Any) -> str:
        name = getattr(contract, "kernel_name", None)
        if name is None and isinstance(contract, Mapping):
            name = contract.get("kernel_name")
        if not isinstance(name, str) or not name:
            raise BuildError(
                "toolchain_profile",
                "the Metal builder needs a task contract that names its kernel",
            )
        return name

    def _device_info(self) -> dict[str, Any]:
        if self._info is None:
            info = self.backend.context().info()
            if not isinstance(info, Mapping):
                raise TypeError("the Metal context did not report device information")
            self._info = dict(info)
        return self._info

    def _macos_version(self) -> str:
        release = platform.mac_ver()[0]
        return f"macos-{release}" if release else "macos-unknown"

    def _profile_for(self, info: Mapping[str, Any]) -> BuildProfile:
        profile = BuildProfile(
            build_profile=METAL_PROFILE_NAME,
            builder_id=METAL_BUILDER_ID,
            builder_version=str(getattr(self.backend, "version", "") or "unknown"),
            build_host="control_host",
            toolchain_id=METAL_TOOLCHAIN_ID,
            toolchain_version=str(info.get("language_version") or "unknown"),
            sdk_version=self._macos_version(),
            compile_profile={
                "language": "msl",
                "math": self.math,
                "device": str(info.get("name") or "unknown"),
                "gpu_family": str(info.get("gpu_family") or "unknown"),
                "unified_memory": bool(info.get("unified_memory", False)),
            },
        )
        if self._profile is None:
            self._profile = profile
        return profile

    def _fallback_profile(self) -> BuildProfile:
        return BuildProfile(
            build_profile=METAL_PROFILE_NAME,
            builder_id=METAL_BUILDER_ID,
            builder_version=str(getattr(self.backend, "version", "") or "unknown"),
            build_host="control_host",
            toolchain_id=METAL_TOOLCHAIN_ID,
            toolchain_version="unknown",
            sdk_version=self._macos_version(),
            compile_profile={"language": "msl", "math": self.math, "device": "unavailable"},
        )

    def _check_probed_identity(self, profile: BuildProfile) -> None:
        """A probe-then-build environment change is not a candidate failure."""

        if self._profile is not None and self._profile != profile:
            raise BuildError(
                "toolchain_profile",
                "the Metal toolchain identity changed between probe and build",
            )


__all__ = ["MetalSourceBuilder"]
