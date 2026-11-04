"""The real Metal ``ArtifactBuilder``: MSL -> library + compute pipeline.

Metal is the *local build-and-execute* placement: the control host compiles the
candidate MSL and keeps the resulting ``MTLLibrary``/``MTLComputePipelineState``
in this process.  This builder is a production implementation of the unified
``ArtifactBuilder`` contract, not a mock and not a concept adapter:

* it calls the existing runtime path (``MetalBackend.assemble`` then
  ``MetalContext.compile``/``compile_all``), so the compile/pipeline behaviour,
  the ``MetalCompileError`` classification and the measured
  ``compile_s``/``pipeline_s``/``language_version`` values are the ones the Metal
  evaluator already reports.  ``MetalContext.compile`` is reached only from this
  builder; the evaluator and profiling paths enter through the builder;
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
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
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

#: The one authoritative registry of **diagnostic** build profiles lives in
#: ``backends/metal/build_profiles.json`` (A2 moved it there from a Python literal).  Probes and
#: bandwidth calibration have no task contract, so they build already-assembled source;
#: ``assemble: raw_source`` means the contract header is *not* prepended, which keeps the probe's
#: byte offsets and diagnostic line numbers exactly what the caller sent.  They are not task
#: LoweringProfiles and a diagnostic outcome never becomes candidate build evidence (plan §5.2).
BUILD_PROFILES_PATH = Path(__file__).resolve().parents[3] / "backends" / "metal" / "build_profiles.json"
BUILD_PROFILES_SCHEMA_VERSION = 1
#: Keys every profile must declare; ``purpose`` is documentation and never a build option.
BUILD_PROFILE_KEYS = frozenset({"assemble", "math", "purpose"})
BUILD_PROFILE_MODES = frozenset({"raw_source", "task_header"})
BUILD_PROFILE_MATH = frozenset({"fast", "safe", "relaxed"})


def load_build_profiles(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """@brief 读取唯一的 ``build_profiles.json`` 并校验闭集 schema。

    @param path 配置文件路径；省略时读 `backends/metal/build_profiles.json`。
    @return profile 名到设置的映射（只含 ``assemble``/``math``）。
    @raises ValueError 文件缺失、版本未知、字段开放或取值非法。
    """
    resolved = Path(path) if path is not None else BUILD_PROFILES_PATH
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover - shipped file
        raise ValueError(f"cannot read the registered build profiles at {resolved}: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"schema_version", "profiles"}:
        raise ValueError("build_profiles.json must have exactly schema_version and profiles")
    if document["schema_version"] != BUILD_PROFILES_SCHEMA_VERSION:
        raise ValueError(f"unsupported build_profiles.json schema_version {document['schema_version']!r}")
    profiles = document["profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("build_profiles.json needs at least one profile")
    loaded: dict[str, dict[str, str]] = {}
    for name, entry in profiles.items():
        if not isinstance(entry, dict) or set(entry) != BUILD_PROFILE_KEYS:
            raise ValueError(f"build profile {name!r} must declare exactly {sorted(BUILD_PROFILE_KEYS)}")
        if entry["assemble"] not in BUILD_PROFILE_MODES:
            raise ValueError(f"build profile {name!r} has an unknown assemble mode {entry['assemble']!r}")
        if entry["math"] not in BUILD_PROFILE_MATH:
            raise ValueError(f"build profile {name!r} has an unknown math mode {entry['math']!r}")
        if not isinstance(entry["purpose"], str) or not entry["purpose"].strip():
            raise ValueError(f"build profile {name!r} needs a non-empty purpose")
        loaded[str(name)] = {"assemble": str(entry["assemble"]), "math": str(entry["math"])}
    return loaded


#: The loaded registry.  There is deliberately no second copy of these values in Python.
DIAGNOSTIC_BUILD_PROFILES: dict[str, dict[str, str]] = load_build_profiles()


@dataclass(frozen=True)
class DiagnosticContract:
    """The minimum a diagnostic build needs: a kernel name and no header.

    It is deliberately *not* a ``Contract``/LoweringProfile: it declares no ABI, no constants
    and no launch facts, so it cannot be mistaken for a task contract.
    """

    kernel_name: str

    def assemble(self, kernel_code: str) -> tuple[str, int]:
        """@brief raw-source assemble：不加契约头，行号偏移为 0。

        @param kernel_code 已经装配好的源文本。
        @return ``(源文本, 0)``。
        """
        return kernel_code, 0

    def render_header(self) -> str:
        """@brief 诊断契约没有头。

        @return 空串。
        """
        return ""


@dataclass(frozen=True)
class MetalKernelSet:
    """Several live pipelines of one multi-stage Metal build.

    ``NativeExecutable.handle`` holds one pipeline for the common single-kernel
    case, so a declared ``taic-launch`` pipeline uses this local container
    instead of pretending the other stages do not exist.  It is never serialized
    and never enters a digest.
    """

    kernels: tuple[Any, ...]

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

    #: This builder compiles with the real toolchain for the real device (§6.2).
    build_kind = "native"

    def __init__(
        self, *, backend: Any | None = None, kernel_contract: Any | None = None,
        math: str = "fast", kernel_names: Any | None = None,
        diagnostic_profile: str | None = None, specialization: Any | None = None,
    ) -> None:
        if math not in ("fast", "safe", "relaxed"):
            raise ValueError("Metal math mode must be fast, safe or relaxed")
        # A diagnostic profile is a *registered* name, never caller-chosen options: it decides
        # whether a contract header is assembled and which math mode is used.
        self.diagnostic_profile = diagnostic_profile
        self.assemble_mode = "task_header"
        # §5.1: a candidate factory receives the Specialization (the same object ``lower``
        # produced), so the contract it assembles and the lowering identity it reports come from
        # one source.  A diagnostic build has no task and must not carry one.
        self.specialization = specialization
        if diagnostic_profile is not None and specialization is not None:
            raise ValueError(
                "a diagnostic build profile has no task: pass either diagnostic_profile or "
                "specialization, not both"
            )
        if specialization is not None and kernel_contract is not None:
            if kernel_contract != specialization.contract:
                raise ValueError(
                    "kernel_contract must be the Specialization's contract (one source, not two)"
                )
        if diagnostic_profile is not None:
            if diagnostic_profile not in DIAGNOSTIC_BUILD_PROFILES:
                raise ValueError(
                    f"unregistered diagnostic build profile {diagnostic_profile!r}; "
                    f"known: {sorted(DIAGNOSTIC_BUILD_PROFILES)}"
                )
            registered = DIAGNOSTIC_BUILD_PROFILES[diagnostic_profile]
            self.assemble_mode = registered["assemble"]
            math = registered["math"]
        if backend is None:
            from .backend import MetalBackend

            backend = MetalBackend()
        self.backend = backend
        self.kernel_contract = kernel_contract
        self.math = math
        names = () if kernel_names is None else tuple(str(name) for name in kernel_names)
        if any(not name for name in names):
            raise ValueError("a Metal build kernel name must not be empty")
        # The builder *configuration* (contract, math, the names of a declared
        # multi-stage pipeline) is chosen by the composition root.  `build` is
        # still the only build entry point: the evaluator does not call the
        # runtime's compile primitive itself, and there is no second path.
        self.kernel_names = tuple(dict.fromkeys(names))
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

    def _compile_assembled(self, context: Any, assembled: str, names: list[str]) -> list[Any]:
        """The Metal runtime primitive: one library, one pipeline per name.

        Only the builder calls this; the evaluator reaches it through
        ``ArtifactBuilder.build``.  Keeping the runtime call here is what makes
        ``MetalContext.compile`` an implementation detail of the builder rather
        than a second production compile path.
        """

        if not names:
            raise BuildError("toolchain_profile", "a Metal build needs at least one kernel name")
        if len(names) == 1:
            return [context.compile(assembled, names[0], math=self.math)]
        return list(context.compile_all(assembled, names, math=self.math))

    @property
    def lowering_identity(self) -> dict[str, str] | None:
        """@brief 候选 build 的语义/启动身份（任务外部/诊断 build 为 ``None``）。

        The candidate composition roots (``evaluate``/``xcworker``) pass the task's
        ``Specialization``; a builder-level test that constructs a synthetic contract has none, and
        gets ``None`` here so it can never claim a task identity it does not have.  Giving a
        Specialization whose contract differs from the build contract is refused in ``build``.

        @return ``{"opspec_digest", "lowering_profile_digest"}`` 或 ``None``。
        """
        if self.specialization is None:
            return None
        return {
            "opspec_digest": self.specialization.opspec_digest,
            "lowering_profile_digest": self.specialization.lowering_profile_digest,
        }

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
        if self.specialization is not None and contract is not self.specialization.contract:
            raise BuildError(
                "toolchain_profile",
                "the build contract is not the Specialization's contract (two sources)",
            )
        kernel_name = self._kernel_name(contract)
        # The typed toolchain profile is established *before* any candidate-level decision, so a
        # candidate compile failure can carry its provenance (plan §5.1).  A host without the
        # toolchain is an infrastructure failure and stays one, whatever the source bytes are.
        try:
            info = self._device_info()
        except Exception as exc:  # noqa: BLE001 - an absent device is not the candidate's fault
            raise BuildError(
                "toolchain_profile",
                f"the Metal toolchain is unavailable on this host: {type(exc).__name__}: {exc}"[:400],
            ) from exc
        profile = self._profile_for(info)
        self._check_probed_identity(profile)
        failed_provenance = BuildProvenance(
            profile=profile, source_sha256=source_bundle.sha256, artifact_set_digest=None,
        )
        try:
            text = source_bundle.source.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Metal consumes text; non-UTF-8 source cannot even reach the compiler, so it is a
            # candidate compile failure (the legacy evaluator reports it as compile_error).  It
            # happens before assembly, so ``assembled_sha256`` is null — the one case §5.1 allows.
            raise BuildError(
                "compile", f"Metal source is not UTF-8 text: {exc}",
                provenance=failed_provenance, assembled_sha256=None,
            ) from exc
        try:
            if self.assemble_mode == "raw_source":
                # Diagnostic build: the caller's bytes are the compilation unit.
                assembled, header_lines = text, 0
            else:
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
        names = list(self.kernel_names) or [kernel_name]
        try:
            kernels = self._compile_assembled(context, assembled, names)
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
                report_detail=message[:500],
                provenance=failed_provenance,
                assembled_sha256=hashlib.sha256(assembled.encode("utf-8")).hexdigest(),
                log=_compile_failure_log(
                    kernel_name=",".join(names), math=self.math,
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
        compiled_language = str(getattr(kernels[0], "language_version", "") or "")
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

        compile_s = float(getattr(kernels[0], "compile_s", 0.0) or 0.0)
        pipeline_s = sum(float(getattr(kernel, "pipeline_s", 0.0) or 0.0) for kernel in kernels)
        handle = kernels[0] if len(kernels) == 1 else MetalKernelSet(tuple(kernels))
        entry = source_bundle.entry()
        descriptor = {
            "backend": "metal",
            "compiler": "apple-metal-msl",
            "kernel_name": kernel_name,
            "language_version": compiled_language or "unknown",
            "math": self.math,
            "device": str(info.get("name") or "unknown"),
            "gpu_family": str(info.get("gpu_family") or "unknown"),
            "max_total_threads": int(getattr(kernels[0], "max_total_threads", 0) or 0),
            "execution_width": int(getattr(kernels[0], "execution_width", 0) or 0),
            "static_tg_bytes": int(getattr(kernels[0], "static_tg_bytes", 0) or 0),
            "assembled_sha256": hashlib.sha256(assembled.encode("utf-8")).hexdigest(),
            "header_lines": int(header_lines),
            "bindings": [
                _binding_document(binding)
                for binding in (getattr(kernels[0], "bindings", None) or ())
            ],
        }
        if len(kernels) > 1:
            # A multi-stage build names every compiled kernel.  This is a local
            # descriptor detail, not a new artifact-set or report field.
            descriptor["kernel_names"] = [str(getattr(kernel, "name", "")) for kernel in kernels]
            descriptor["kernels"] = [
                {
                    "name": str(getattr(kernel, "name", "")),
                    "max_total_threads": int(getattr(kernel, "max_total_threads", 0) or 0),
                    "execution_width": int(getattr(kernel, "execution_width", 0) or 0),
                    "static_tg_bytes": int(getattr(kernel, "static_tg_bytes", 0) or 0),
                    "bindings": [
                        _binding_document(binding)
                        for binding in (getattr(kernel, "bindings", None) or ())
                    ],
                }
                for kernel in kernels
            ]
        native = NativeExecutable(
            backend_id="metal",
            kind="compute-pipeline",
            entry_path=entry.path,
            entry_sha256=entry.sha256,
            descriptor=descriptor,
            handle=handle,
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
            # The compilation unit is the assembled text (contract header + candidate),
            # so this is the digest of what the compiler actually saw.
            assembled_sha256=hashlib.sha256(assembled.encode("utf-8")).hexdigest(),
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


__all__ = ["MetalKernelSet", "MetalSourceBuilder"]
