"""A deterministic CPU backend for remote-service tests.

It intentionally has no Metal, CUDA, NPU or model dependency.  Test reports are
marked simulated so they can never be mistaken for publishable measurements.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tcenv.backends.base import BenchHarness, Capabilities, DeviceInfo
from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildCapabilities,
    BuildError,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
    SourceBundle,
)
from tcenv.remote.artifact_bundle import ArtifactBundle, NativeExecutable
from tcenv.remote.protocol import EvaluationRequest


@dataclass(frozen=True)
class _FakeLaunchStage:
    kernel: str = "npu_main"
    grid: tuple[int, int, int] = (1, 1, 1)
    threads: tuple[int, int, int] = (4096, 1, 1)
    threadgroup_bytes: int = 1 << 30


@dataclass(frozen=True)
class _FakeLaunchSpec:
    stages: tuple[_FakeLaunchStage, ...] = (_FakeLaunchStage(),)
    math: str = "fast"

    def digest(self) -> str:
        return "simulated npu-launch (Metal limits intentionally not applicable)"


@dataclass(frozen=True)
class _FakeKernel:
    name: str
    compile_s: float = 0.0
    pipeline_s: float = 0.0
    language_version: str = "fake-npu-1"
    static_tg_bytes: int = 1 << 30


@dataclass(frozen=True)
class _FakeCaseResult:
    case: str
    ok: bool = True
    message: str = "simulated correctness"

    def to_json(self) -> dict:
        return {"case": self.case, "ok": self.ok, "message": self.message}


class _FakeCompileError(RuntimeError):
    pass


class _FakeRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class _FakeKernelSet:
    kernels: tuple[_FakeKernel, ...]


class _FakeSourceBuilder:
    """Test-only ``ArtifactBuilder`` for the CPU backend.

    It produces the same neutral ``BuildOutcome`` shape as the production Metal
    builder: the SOURCE artifact is the exact source bytes and the executable is
    a backend-native handle.  This lives in tests/support and is never part of a
    production composition root.
    """

    #: A test-only simulator: it never compiles with a vendor toolchain (§6.2).
    build_kind = "simulator"

    def __init__(
        self, backend: "FakeNpuBackend", *, kernel_contract: Any = None, math: str = "fast",
        kernel_names: Any = None, specialization: Any = None,
    ) -> None:
        self.backend = backend
        self.kernel_contract = kernel_contract
        self.math = math
        self.kernel_names = None if kernel_names is None else tuple(kernel_names)
        self.specialization = specialization
        if specialization is not None and kernel_contract is not None:
            if kernel_contract != specialization.contract:
                raise ValueError("kernel_contract must be the Specialization's contract")
        self.profile = BuildProfile(
            build_profile="fake-npu-source-v1",
            builder_id="fake-npu-builder",
            builder_version=backend.version,
            build_host="control_host",
            toolchain_id="fake-npu",
            toolchain_version="1",
            sdk_version="fake",
            compile_profile={"language": "fake-npu", "math": math},
            build_kind="simulator",
        )

    @property
    def lowering_identity(self) -> dict[str, str] | None:
        """The task Specialization identity (``None`` for a taskless fixture)."""
        if self.specialization is None:
            return None
        return {
            "opspec_digest": self.specialization.opspec_digest,
            "lowering_profile_digest": self.specialization.lowering_profile_digest,
        }

    def probe(self) -> BuildCapabilities:
        return BuildCapabilities(profile=self.profile, available=True)

    def build(self, source_bundle: SourceBundle, *, task_contract: Any, deadline_monotonic: float | None = None) -> BuildOutcome:
        name = getattr(task_contract, "kernel_name", None) or "npu_main"
        # A simulated *candidate* compile failure must be shaped like the real one: the typed
        # provenance of the build that ran (no artifact set) plus the assembled digest, so the
        # evaluator can project it without parsing text (R3 §5.1).
        if b"COMPILE_FAIL" in source_bundle.source:
            import hashlib

            raise BuildError(
                "compile", "sim-npu: the fake compiler rejected the kernel",
                report_detail="sim-npu: kernel rejected at line 1",
                provenance=BuildProvenance(
                    profile=self.profile, source_sha256=source_bundle.sha256, artifact_set_digest=None,
                ),
                assembled_sha256=hashlib.sha256(source_bundle.source).hexdigest(),
                duration_ms=1,
            )
        names = list(self.kernel_names or [name])
        kernels = tuple(_FakeKernel(kernel_name) for kernel_name in names)
        entry = source_bundle.entry()
        handle: Any = kernels[0] if len(kernels) == 1 else _FakeKernelSet(kernels)
        native = NativeExecutable(
            backend_id="sim-npu", kind="compute-pipeline",
            entry_path=entry.path, entry_sha256=entry.sha256,
            descriptor={
                "backend": "sim-npu", "kernel_names": names,
                "language_version": kernels[0].language_version, "math": self.math,
            },
            handle=handle,
        )
        bundle = ArtifactBundle(entries=(entry,), native=native)
        provenance = BuildProvenance(
            profile=self.profile, source_sha256=source_bundle.sha256,
            artifact_set_digest=bundle.artifact_set_digest,
        )
        evidence = BuildEvidence(
            mode="prebuilt", status="ok", provenance=provenance,
            artifacts=(BuildArtifact.from_entry(entry),), duration_ms=0,
            log=b"sim-npu: fake build\n", native=native, placement="control_host",
        )
        # A direct-source simulator has no separate assembled representation: the compilation
        # unit is exactly the source bundle it hands to the fake compiler.
        return BuildOutcome(
            bundle=bundle, provenance=provenance, evidence=evidence,
            assembled_sha256=source_bundle.sha256,
        )


class FakeNpuBackend:
    """Deterministic evaluator used behind the real remote service core.

    The same test double also implements the offline generic-evaluation hooks.
    Its deliberately non-MSL launch declares resources above Metal limits; the
    fake accepts them because those limits are not part of its contract.  This
    is a CPU import/behavior fixture, not an NPU contract or a production target.
    """

    id = "sim-npu"
    version = "1"
    language = "fake-npu"
    file_ext = ".npu"
    display_name = "Fake NPU"
    source_fence = "npu"
    reference_skill = ""
    skill_names: tuple[str, ...] = ()
    skill_optimization = ""
    skill_profiling = ""
    skill_profiler = ""
    skill_hardware = ""
    runtime_headers = ""
    capabilities = Capabilities(supports_launch_spec=True)

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def screen(self, source: str, task: Any = None) -> list[str]:
        self.calls.append(("screen", source))
        return ["simulated screen rejection"] if "SCREEN_REJECT" in source else []

    def source_name_pattern(self, kernel_name: str) -> str:
        import re

        return r"\bnpu-kernel\s+" + re.escape(kernel_name) + r"\b"

    def parse_launch(self, source: str, contract: Any) -> _FakeLaunchSpec | None:
        return _FakeLaunchSpec() if "npu-launch" in source else None

    def launch_stage_count(self, source: str, contract: Any) -> int:
        spec = self.parse_launch(source, contract)
        return 1 if spec is None else len(spec.stages)

    def validate_launch(self, spec: Any, *, source: str, contract: Any, task: Any, target: Any, device_info: Mapping[str, Any], require_contract_launch: bool) -> None:
        self.calls.append(("validate_launch", source))
        # No Metal grid/threadgroup/workspace limit is consulted here.

    def validation_env(self, base_env: Mapping[str, str]) -> dict[str, str]:
        return self.worker_env(base_env, validation=True)

    def worker_env(self, base_env: Mapping[str, str], *, validation: bool) -> dict[str, str]:
        self.calls.append(("worker_env", str(validation)))
        env = dict(base_env)
        env["NPU_SIM_VALIDATION"] = "1" if validation else "0"
        return env

    def parse_validation_stderr(self, stderr: str) -> list[str]:
        self.calls.append(("parse_validation_stderr", stderr))
        return []

    def is_instrumented_size_only(self, violations: list[str]) -> bool:
        return False

    def validation_problems(self, violations: list[str], static_bytes: list[int], max_tg_memory: int) -> list[str]:
        return list(violations)

    def validation_summary(self, *, size_only: bool, target: Any) -> str:
        return "unsupported"

    def compiler_baseline(self, task_name: str, dtype: str, size: str, baseline_us: Mapping[str, float], candidate_us: float) -> dict:
        self.calls.append(("compiler_baseline", task_name))
        return {"estimated_us": 2000.0, "speedup": 2000.0 / candidate_us, "simulated": True}

    def device_matches(self, target: Any, info: Mapping[str, Any]) -> bool:
        return info.get("name") == "Fake NPU"

    def limit_notes(self, target: Any, info: Mapping[str, Any]) -> list[str]:
        return ["simulated device; target limits are not hardware measurements"]

    def set_heartbeat_hooks(self, *, on_compile=None, on_launch=None) -> None:
        return None

    def context(self) -> object:
        return object()

    def compile_error(self):
        return _FakeCompileError

    def runtime_error(self):
        return _FakeRuntimeError

    def probe(self, ctx: Any, target: Any = None) -> DeviceInfo:
        # Poison values make accidental generic Metal-limit checks fail the test.
        info = {
            "name": "Fake NPU",
            "device_id": "fake-npu-0",
            "max_threads_per_threadgroup": 1,
            "max_threadgroup_memory_bytes": 1,
            "max_buffer_length": 1,
            "recommended_working_set_bytes": 0,
        }
        matches = self.device_matches(target, info)
        return DeviceInfo(
            backend=self.id,
            name="Fake NPU",
            family="",
            limits=dict(info),
            raw=info,
            matches=matches,
            match_error="" if matches else f"device Fake NPU does not match target {target.id}",
            warnings=tuple(self.limit_notes(target, info)) if matches else (),
        )

    def assemble(self, contract: Any, source: str) -> tuple[str, int]:
        return source, 0

    def rewrite_compiler_lines(self, contract: Any, message: str, header_lines: int) -> str:
        return message

    def describe(self, contract: Any) -> str:
        return f"simulated contract for {contract.kernel_name}"

    def abi_stub(self, contract: Any) -> str:
        return f"npu-kernel {contract.kernel_name}"

    def artifact_builder(
        self, *, kernel_contract: Any = None, math: str = "fast", kernel_names: Any = None,
        diagnostic_profile: str | None = None, specialization: Any = None,
    ) -> _FakeSourceBuilder:
        self.calls.append(("artifact_builder", str(math)))
        return _FakeSourceBuilder(
            self, kernel_contract=kernel_contract, math=math,
            kernel_names=kernel_names, specialization=specialization,
        )

    def check_abi(self, contract: Any, kernels: list[Any], *, target: Any = None, device_info: Mapping[str, Any] | None = None, spec: Any = None, validation: bool = False, **legacy: Any) -> list[str]:
        self.calls.append(("check_abi", target.id if target is not None else ""))
        return []

    def pipeline_stats(self, kernel: Any, contract: Any, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict:
        return {}

    def pipeline_stats_many(self, pairs: list, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict:
        return {}

    def prepare_correctness(self, ctx: Any, contract: Any, *, tight: bool, spec: Any = None) -> object:
        return object()

    def run_correctness(self, ctx: Any, kernel: Any, task: Any, argset: Any, seed: int, *, cases=None, baseline_fns=None, determinism: bool, launcher=None):
        selected = cases or task.cases
        return [_FakeCaseResult(case) for case in selected], [], []

    def prepare_case(self, task: Any, seed: int) -> object:
        return object()

    def baseline_runners(self, task: Any, data0: Any) -> Mapping[str, Any]:
        return {"simulated_baseline": object()}

    def baseline_runner(self, impl: Any, task: Any, data: Any):
        raise AssertionError("the fake generic path must not execute Metal task baselines")

    def rotation_sets(self, task: Any) -> int:
        return 1

    def coverage(self, task: Any, results: list, source: str, toolchain: dict) -> dict:
        return {"kind": "simulated", "passed": True, "cases": len(results)}

    def calibration(self, target_id: str) -> dict:
        return {}

    def roofline(self, task: Any, time_s: float, calibration: dict) -> dict:
        return {}

    def prepare_bench_sets(self, ctx: Any, contract: Any, task: Any, *, spec: Any, data0: Any) -> list[object]:
        return [object()]

    def prepare_bench(self, ctx: Any, kernels: list[Any], contract: Any, task: Any, *, spec: Any, data0: Any, reps: int, measure_serial: bool, sets: list[Any], base_runners: Mapping[str, Any], selected: list[str]) -> BenchHarness:
        baselines = {
            name: (lambda k: 0.002 * k)
            for name in selected
            if name not in {"candidate", "candidate_serial"}
        }
        return BenchHarness(
            rotation_sets=1,
            dispatch_note="simulated CPU timing; not a device measurement",
            protocol_note="deterministic fake backend; not publishable",
            cand_run=lambda k: 0.001 * k,
            cand_serial_run=(lambda k: 0.0015 * k) if measure_serial else None,
            baseline_runners=baselines,
            gpu_samples=[],
            spot_check=lambda: [],
        )

    def evaluate(self, request: EvaluationRequest) -> dict:
        counter = os.environ.get("NPU_SIM_COUNTER")
        if counter:
            path = Path(counter)
            count = int(path.read_text()) if path.is_file() else 0
            path.write_text(str(count + 1))
        # A deliberate slow path exercises worker timeout/cancellation without
        # requiring a device or a real kernel.
        progress_path = os.environ.get("TCENV_REMOTE_PROGRESS_PATH")
        if "REMOTE_PROGRESS" in request.source and progress_path:
            Path(progress_path).write_text(json.dumps({
                "task": request.task, "dtype": request.dtype, "size": request.size,
                "mode": request.mode, "stage": "bench", "status": "running",
                "message": "fake progress", "finished": False, "updated_at": time.time(),
            }), encoding="utf-8")
            time.sleep(0.35)
        if "REMOTE_SLEEP_FAIR" in request.source:
            time.sleep(1.0)
        elif "REMOTE_SLEEP_SHORT" in request.source:
            time.sleep(0.4)
        elif "REMOTE_SLEEP" in request.source:
            time.sleep(5.0)
        if "REMOTE_PROGRESS" in request.source and progress_path:
            # No delay after the terminal write: the service must drain this
            # snapshot during worker cleanup rather than dropping it.
            Path(progress_path).write_text(json.dumps({
                "task": request.task, "dtype": request.dtype, "size": request.size,
                "mode": request.mode, "stage": "done", "status": "ok",
                "message": "fake progress complete", "finished": True, "updated_at": time.time(),
            }), encoding="utf-8")
        order_path = os.environ.get("NPU_SIM_ORDER")
        if order_path:
            with Path(order_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"run_id": request.run_id, "session_id": request.session_id, "kind": request.kind, "request_id": request.request_id}) + "\n")
        digest = hashlib.sha256((request.source + str(request.seed)).encode("utf-8")).hexdigest()
        receipt_digest = hashlib.sha256((request.source + str(request.seed) + request.request_id).encode("utf-8")).hexdigest()
        receipt_id = "sim-replay" if "REPLAY_RECEIPT" in request.source else f"sim-{receipt_digest[:16]}"
        candidate_us = 100.0 + int(digest[:8], 16) % 1000
        report = {
            "task": request.task,
            "dtype": request.dtype,
            "size": request.size,
            "mode": request.mode,
            "seed": request.seed,
            "status": "ok",
            "stage": "done",
            "candidate_us": candidate_us,
            "speedup": 1.0,
            "elapsed_s": 0.5 if request.mode == "timed" else 0.0,
            "target_id": request.target_id,
            "protocol_version": "3",
            "simulated": True,
        }
        return {
            "report": report,
            "evidence": {
                "request_id": request.request_id,
                "candidate_sha256": request.candidate_sha256,
                "measurement_kind": "simulated",
            },
            "receipt": {
                "receipt_id": receipt_id,
                "request_id": request.request_id,
                "request_digest": request.request_digest,
                "measurement_kind": "simulated",
            },
        }
