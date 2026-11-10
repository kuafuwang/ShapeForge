"""Bounded external worker for the v2 task-contract evaluator.

The v2 runtime permits arbitrary candidate control flow (including loops and ``goto``), so an
in-process ``waitUntilCompleted`` cannot be interrupted from inside.  The v2 evaluator therefore
runs the build and execution in this worker process under the existing ``supervise`` parent: a
heartbeat written immediately before a launch lets the parent SIGKILL the worker when one launch
or the whole evaluation exceeds its deadline.  The request is a staged task pack + the exact
candidate entries, never a host path to the control checkout.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

REQUEST_SCHEMA = "tcenv.v2.worker.request.1"
RESULT_SCHEMA = "tcenv.v2.worker.result.1"


def _b64(data: bytes) -> str:
    return base64.b64encode(bytes(data)).decode("ascii")


def _unb64(text: str, what: str) -> bytes:
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{what} is not valid base64") from exc


def encode_request(
    request: Any,
    *,
    mode: str,
    seed: int,
    reps: int,
    cases: tuple[str, ...] | None,
    target_id: str,
    timeout: float,
    launch_deadline_s: float,
) -> dict[str, Any]:
    """@brief 把一次 v2 评测压缩成可交给隔离子进程的 JSON 请求。"""
    build_request = request.build_request
    staged = build_request.locator.stage()
    candidate = build_request.candidate
    lifecycle = request.canonical_bundle.lifecycle
    return {
        "schema_version": REQUEST_SCHEMA,
        "locator": dict(staged.locator),
        "task_files": [[name, _b64(data)] for name, data in staged.files],
        "track_id": build_request.track.track_id,
        # The control host's frozen TrackSpec + registry + rendering identity.  The worker must
        # reconstruct the selection from these bytes and verify both digests; it must not read a
        # default tracks.json that may be a different registry.
        "frozen_track": build_request.track_selection.frozen_document(
            presentation_sha256=build_request.presentation_sha256
        ),
        "candidate": {
            "track_id": candidate.track_id,
            "entries": [
                {
                    "path": entry.path,
                    "role": entry.role,
                    "media_type": entry.media_type,
                    "data": _b64(entry.data),
                }
                for entry in candidate.entries
            ],
        },
        "lifecycle": None if lifecycle is None else lifecycle.document(),
        "mode": mode,
        "seed": int(seed),
        "reps": int(reps),
        "cases": None if cases is None else list(cases),
        "target_id": target_id,
        "timeout": float(timeout),
        "launch_deadline_s": float(launch_deadline_s),
        # The control host's contract fingerprint.  A staged copy spells pack paths differently
        # (repo-relative vs bare names) even when the bytes match, so the worker must publish
        # this identity instead of recomputing it.
        "opspec_digest": build_request.contract.spec.digest,
        "lowering_profile_digest": build_request.contract.fingerprint(),
        # The geometry the control host already resolved.  The worker must not parse the
        # launch text again: derived scalars and when-stages live in this document.
        "candidate_geometry": dict(build_request.pair.candidate_document),
        "fallback_geometry": None if build_request.pair.fallback_document is None else dict(build_request.pair.fallback_document),
    }


def _lifecycle_from_document(document: Mapping[str, Any]) -> Any:
    """@brief 从 build 侧的 canonical lifecycle 文档重建同一生命周期。"""
    from .track import CandidateLifecycle, HostStep, PrepareDeclaration, ResolvedGeometry, ResolvedStage, WorkspaceUse

    raw_geometry = document["geometry"]
    stages = tuple(
        ResolvedStage(
            index=int(item["index"]),
            kernel=str(item["kernel"]),
            grid=tuple(int(value) for value in item["grid"]),
            threads=tuple(int(value) for value in item["threads"]),
            threadgroup_bytes=int(item["threadgroup_bytes"]),
            workspace=tuple(WorkspaceUse(name=str(use["name"]), zero=bool(use["zero"])) for use in item["workspace"]),
            active=bool(item["active"]),
            when=item["when"],
            derived_scalars=tuple((str(pair["name"]), int(pair["value"])) for pair in item["derived_scalars"]),
        )
        for item in raw_geometry["stages"]
    )
    geometry = ResolvedGeometry(
        source=str(raw_geometry["source"]),
        stages=stages,
        resolution_version=int(raw_geometry.get("resolution_version", 1)),
    )
    steps = []
    for item in document.get("prepare", ()):
        fields = {key: item[key] for key in (
            "name", "input_dependent", "cached", "inside_timing", "computes_reference", "preprocesses_inputs",
        ) if key in item}
        # cache_identity is computed by CandidateLifecycle, not a hand-written declaration.
        steps.append(HostStep(PrepareDeclaration(**fields), role=str(item["role"])))
    return CandidateLifecycle(
        track_id=str(document["track_id"]),
        geometry=geometry,
        host_steps=tuple(steps),
        candidate_sha256=str(document["candidate_sha256"]),
        target_id=str(document["target_id"]),
        instance_key=str(document["instance_key"]),
    )


def decode_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 在 worker 内重建任务、候选与生命周期（不读取控制机路径）。"""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("v2 worker request has the wrong schema")
    from .taskcontract import StagedTaskPack
    from .track import CandidateArtifact, CandidateEntry

    files = tuple((str(name), _unb64(str(data), f"task file {name}")) for name, data in payload["task_files"])
    staged = StagedTaskPack(locator=dict(payload["locator"]), files=files)
    work_root = tempfile.mkdtemp(prefix="tcenv_v2_task_")
    locator = staged.rebuild(work_root, code_root=Path(__file__).resolve().parents[1])
    entries = tuple(
        CandidateEntry(
            path=str(item["path"]),
            role=str(item["role"]),
            media_type=str(item["media_type"]),
            data=_unb64(str(item["data"]), f"candidate entry {item['path']}"),
        )
        for item in payload["candidate"]["entries"]
    )
    candidate = CandidateArtifact(track_id=str(payload["candidate"]["track_id"]), entries=entries)
    lifecycle = None
    if payload.get("lifecycle") is not None:
        lifecycle = _lifecycle_from_document(payload["lifecycle"])
    return {
        "locator": locator,
        "candidate": candidate,
        "lifecycle": lifecycle,
        "track_id": str(payload["track_id"]),
        "frozen_track": dict(payload["frozen_track"]),
        "mode": str(payload["mode"]),
        "seed": int(payload["seed"]),
        "reps": int(payload["reps"]),
        "cases": None if payload.get("cases") is None else tuple(str(item) for item in payload["cases"]),
        "target_id": str(payload["target_id"]),
        "frozen_identity": {
            "opspec_digest": str(payload["opspec_digest"]),
            "lowering_profile_digest": str(payload["lowering_profile_digest"]),
        },
        "candidate_geometry": dict(payload["candidate_geometry"]),
        "fallback_geometry": None if payload.get("fallback_geometry") is None else dict(payload["fallback_geometry"]),
    }


def _install_heartbeat(path: str, target_id: str) -> None:
    """@brief 在 worker 内安装编译/启动/等锁心跳；父进程据 B 的年龄 SIGKILL。"""
    from .backends.registry import load_backend
    from .gpulock import set_wait_hooks
    from .target import load_target

    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    compile_hooks = (
        lambda: os.pwrite(fd, f"C{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )
    launch_hooks = (
        lambda: os.pwrite(fd, f"B{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )
    backend = load_backend(load_target(target_id).backend)
    backend.set_heartbeat_hooks(on_compile=compile_hooks, on_launch=launch_hooks)
    # 与 v1 评测同一个组合根：原生批量编码器（launch-bound 小算子的计时不再被 PyObjC 主导）
    # 与内部清零核都在任何计时窗口之前准备好；v2 路径此前漏了这一步，一直走 PyObjC。
    prepare = getattr(backend, "prepare_context", None)
    if callable(prepare):
        prepare(backend.context())
    set_wait_hooks(
        lambda: os.pwrite(fd, f"W{time.time():020.6f}\n".encode(), 0),
        lambda: os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0),
    )


def capability_failure_result(
    locator: Any,
    resolved: Mapping[str, Any],
    error: Any,
) -> dict[str, Any]:
    """@brief 缺 Host Program 工具链：稳定的 ``rejected`` 环境结果，不是 worker crash。

    The native C path needs clang.  When it is absent, the build boundary raises a
    typed ``backend_capability_unavailable``; the worker publishes a build-free ``rejected``
    report with that code in the message (the JSON ``host/tiling.json`` path never reaches this
    branch).  No candidate source is blamed and no build record is fabricated.

    @param locator 任务 locator。
    @param resolved 已解码的 worker 请求。
    @param error 带 ``error_code`` 的构建异常。
    @return worker result document。
    """
    from .evaluate import EvalReport
    from .schema import PROTOCOL_VERSION
    from .taskcontract.runner import select_track

    contract = locator.load()
    selection = select_track(
        resolved["frozen_track"]["track_id"], frozen=resolved["frozen_track"]
    )
    report = EvalReport(
        task=contract.op_name,
        dtype=contract.instance.dtype,
        size=contract.instance.size,
        mode=str(resolved["mode"]),
        seed=int(resolved["seed"]),
        status="rejected",
        stage="host_toolchain",
        message=f"backend_capability_unavailable: {getattr(error, 'detail', str(error))}"[:2000],
        target_id=str(resolved["target_id"]),
        protocol_version=PROTOCOL_VERSION,
        build=None,
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "report": report.to_json(),
        "execution": {"provider": "bounded-v2-worker", "stage": "host_toolchain", "dispatch": []},
        "build": None,
        "track": selection.document(),
        "presentation_sha256": resolved["frozen_track"]["presentation_sha256"],
    }


def evaluate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """@brief worker 主逻辑：重建并执行一次 v2 evaluate_task，返回报告与 execution。"""
    from .remote.artifact_builder import BuildError

    resolved = decode_request(payload)
    from .backends.metal.track_adapter import metal_compile_failure_report
    from .backends.metal.track_factory import adapter_for_track
    from .taskcontract.runner import evaluate_task, select_track

    from .track.plane import execution_geometry_from_document

    locator = resolved["locator"]
    selection = select_track(resolved["frozen_track"]["track_id"], frozen=resolved["frozen_track"])
    adapter = adapter_for_track(selection.track)
    builder_type = adapter.resolve("builder")
    evaluator_type = adapter.resolve("evaluator")
    candidate_geometry = execution_geometry_from_document(resolved["candidate_geometry"])
    fallback_geometry = (
        None if resolved["fallback_geometry"] is None
        else execution_geometry_from_document(resolved["fallback_geometry"])
    )

    def resolver(_candidate: Any) -> Any:
        return candidate_geometry

    evaluator = evaluator_type(
        target_id=resolved["target_id"],
        mode=resolved["mode"],
        seed=resolved["seed"],
        reps=resolved["reps"],
        cases=resolved["cases"],
        worker_child=True,
        frozen_identity=resolved["frozen_identity"],
    )
    from .gpulock import gpu_lock

    try:
        with gpu_lock(f"v2 worker: {locator.task_id} {resolved['mode']}", exclusive=True):
            result = evaluate_task(
                locator,
                track_id=resolved["track_id"],
                candidate=resolved["candidate"],
                builder=builder_type(target_id=resolved["target_id"]),
                evaluator=evaluator,
                geometry_resolver=resolver,
                fallback_geometry=fallback_geometry,
                lifecycle=resolved["lifecycle"],
                frozen_track=resolved["frozen_track"],
                target_id=resolved["target_id"],
            )
    except BuildError as exc:
        # The worker owns the first real compile, so a candidate compile failure is *this*
        # process's report: the typed partial build (real assembled digest/log) and never a
        # ``full`` build.  A missing Host Program toolchain is a typed environment outcome, not a
        # crash; any other build failure class stays an infrastructure crash.
        if getattr(exc, "error_code", None) == "backend_capability_unavailable":
            return capability_failure_result(locator, resolved, exc)
        if getattr(exc, "error_class", None) != "compile":
            raise
        selection = select_track(
            resolved["frozen_track"]["track_id"], frozen=resolved["frozen_track"]
        )
        report, partial = metal_compile_failure_report(
            contract=locator.load(),
            target_id=resolved["target_id"],
            mode=resolved["mode"],
            seed=resolved["seed"],
            error=exc,
            identity=resolved["frozen_identity"],
        )
        return {
            "schema_version": RESULT_SCHEMA,
            "report": report,
            "execution": {"provider": "bounded-v2-worker", "stage": "compile", "dispatch": []},
            "build": partial,
            "track": selection.document(),
            "presentation_sha256": resolved["frozen_track"]["presentation_sha256"],
        }
    return {
        "schema_version": RESULT_SCHEMA,
        "report": result.report,
        "execution": result.execution,
        "build": result.build,
        "track": result.track,
        "presentation_sha256": result.presentation_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m tcenv.v2_worker request.json`` 入口。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: python -m tcenv.v2_worker REQUEST.json\n")
        return 2
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    out = Path(request["out"])
    try:
        _install_heartbeat(str(request["heartbeat"]), str(request["target_id"]))
        if request.get("stage"):
            from .events import bind_progress

            bind_progress(request["stage"])
        result = evaluate_payload(request["payload"])
        out.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        return 0
    except BaseException as exc:  # noqa: BLE001 - worker must leave a machine-readable failure
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "schema_version": RESULT_SCHEMA,
                    "report": {
                        "status": "crash",
                        "stage": "crash",
                        "message": f"v2 worker failed: {type(exc).__name__}: {exc}",
                    },
                    "execution": None,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return 1


def run_bounded_v2(
    payload: Mapping[str, Any],
    *,
    timeout: float,
    launch_deadline_s: float,
    compile_deadline_s: float | None = None,
    worker_module: str = "tcenv.v2_worker",
) -> dict[str, Any]:
    """@brief 在现有 supervise/popen_group 下运行一个 v2 worker。

    The parent, not the worker, owns the deadline.  The worker only writes a launch heartbeat;
    the parent kills its process group when one launch or the whole evaluation is too old.
    """
    from .evaluate import POST_KILL_COOLDOWN_S, REPO_ROOT, supervise
    from .children import forget, kill_group, popen_group, scrubbed_env

    with tempfile.TemporaryDirectory(prefix="tcenv_v2_") as td:
        root = Path(td)
        out_path = root / "report.json"
        hb_path = root / "heartbeat"
        stage_path = root / "stage.json"
        err_path = root / "stderr.txt"
        req = {
            "out": str(out_path),
            "heartbeat": str(hb_path),
            "stage": str(stage_path),
            "target_id": str(payload.get("target_id", "apple-m4-max")),
            "payload": dict(payload),
        }
        req_path = root / "request.json"
        req_path.write_text(json.dumps(req, sort_keys=True), encoding="utf-8")
        env = scrubbed_env()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        base = {"status": "crash", "stage": "crash", "message": ""}
        with open(err_path, "w") as err_f, open(root / "stdout.txt", "w") as out_f:
            proc = popen_group(
                [sys.executable, "-m", worker_module, str(req_path)],
                stdout=out_f,
                stderr=err_f,
                env=env,
                cwd=str(REPO_ROOT),
            )
            from .gpulock import released

            try:
                # The parent only waits.  Pytest holds a shared GPU lock for the whole gpu
                # test; keeping it while this worker asks for the exclusive lock deadlocks
                # the real Metal slices.  v1's worker wait releases the same way.
                with released():
                    if compile_deadline_s is None:
                        reason = supervise(
                            proc,
                            hb_path,
                            launch_deadline_s=launch_deadline_s,
                            timeout=timeout,
                            stage_path=stage_path,
                        )
                    else:
                        reason = supervise(
                            proc,
                            hb_path,
                            launch_deadline_s=launch_deadline_s,
                            timeout=timeout,
                            stage_path=stage_path,
                            compile_deadline_s=float(compile_deadline_s),
                        )
            finally:
                kill_group(proc)
                forget(proc)
        if reason:
            time.sleep(POST_KILL_COOLDOWN_S)
            return {
                "status": "timeout",
                "stage": "timeout",
                "message": f"v2 worker killed by the external deadline ({reason})",
            }
        if not out_path.exists():
            tail = "\n".join(err_path.read_text(errors="replace").strip().splitlines()[-12:])
            return {
                "status": "crash",
                "stage": "crash",
                "message": f"v2 worker exited with code {proc.returncode} without a report; stderr tail:\n{tail}",
            }
        try:
            result = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"status": "crash", "stage": "crash", "message": f"v2 worker left an unreadable report: {exc}"}
        if not isinstance(result, Mapping) or result.get("schema_version") != RESULT_SCHEMA:
            return {"status": "crash", "stage": "crash", "message": "v2 worker returned an invalid result envelope"}
        return dict(result)


__all__ = [
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "decode_request",
    "encode_request",
    "evaluate_payload",
    "main",
    "run_bounded_v2",
]


if __name__ == "__main__":
    raise SystemExit(main())
