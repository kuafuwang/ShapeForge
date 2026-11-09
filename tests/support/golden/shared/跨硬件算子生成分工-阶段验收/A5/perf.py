#!/usr/bin/env python3
"""A5 v1/v2 performance comparison on the same Mac, candidate bytes and benchmark protocol.

The two phases are deliberately separate:

``--phase v1``
    Runs the frozen G0 candidate source through the **v1** task path (``tasks.get_task`` +
    ``tcenv.evaluate``).  It only works before the A5 migration deletes the v1 task
    assembly, and it refuses to run if a task already answers through the v2 standard entry.
    The record is tied to G0 by the combo's ``opspec_digest``/``contract`` facts and by the
    candidate bytes' sha256.

``--phase v2``
    Runs the *same candidate bytes* through the **v2** standard entry
    (``discover_standard_tasks`` + ``evaluate_task`` + ``MetalTrackEvaluator``).  This phase
    is live and must be re-runnable after the migration.

Neither phase calls a model, ``taic run`` or ``Orchestrator.run``.  The candidate is the
frozen G0 ``naive_msl`` text with one ``taic-launch`` block that declares exactly the v1
contract grid/threads, so the kernel bytes and the launch are identical in both phases;
the benchmark is the same ``tcenv.evaluate`` bench code in both cases
(``v2_evaluation_task`` projects the v2 contract back onto that evaluator), which is what
makes ``median_us`` comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
G0 = ROOT / "docs/架构/跨硬件算子生成分工-G0/corpus/index.json"

#: The 16 A5 migration candidates, in G0 order.
TASKS: tuple[str, ...] = (
    "sigmoid",
    "relu",
    "silu",
    "swiglu",
    "rmsnorm",
    "softmax",
    "reduction_sum",
    "gemm",
    "gemm_silu",
    "sem_gemm_neg",
    "sem_relu_gelu_comment",
    "sem_gelu_named_relu",
    "gen_relu",
    "gen_sigmoid",
    "gen_silu",
    "gen_swiglu",
)
DTYPE = "fp16"
SIZE = "small"
REPS = 15
SEED = 7
CASES = ("normal", "wide_range", "outlier")


class EvidenceError(RuntimeError):
    """A performance record cannot support the approved gate.

    This is deliberately not a ``KeyError``: stale evidence from before the baselines were
    interleaved into the v2 measurement (or a record that silently dropped a registered
    baseline) must stop the acceptance with an explicit reason, never look like a passing
    measurement or crash the verifier.
    """


def sha256(text: str) -> str:
    """@brief 文本字节摘要。

    @param text 文本。
    @return 十六进制 sha256。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def head() -> str:
    """@brief 当前 HEAD（v1 阶段在临时副本里测量时用 ``A5_HEAD`` 显式给出）。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    import os

    return os.environ.get("A5_HEAD", "unrecorded")


def tree_digest(paths: "list[Path]", root: Path) -> str:
    """@brief 一棵子树的内容摘要（路径 + 内容，跳过 ``__pycache__``）。

    @param paths 目录/文件列表。
    @param root 相对路径的根。
    @return 十六进制 sha256。
    """
    import hashlib

    digest = hashlib.sha256()
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(item for item in sorted(path.rglob("*")) if item.is_file() and "__pycache__" not in item.parts)
        elif path.is_file():
            files.append(path)
    for item in sorted(files, key=lambda value: str(value.relative_to(root))):
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def source_provenance() -> dict:
    """@brief 本进程是从哪份源码树测量的（v1 是迁移前的只读副本）。

    @return 文档。
    """
    tasks = ROOT / "tasks"
    return {
        "root": str(ROOT),
        "head": head(),
        "tasks_digest": tree_digest([tasks], ROOT),
        "protocol_digest": tree_digest([ROOT / "tcenv" / "bench.py", ROOT / "tcenv" / "evaluate.py"], ROOT),
    }


def index() -> dict:
    """@brief 读取 G0 语料索引。

    @return 文档。
    """
    return json.loads(G0.read_text(encoding="utf-8"))


def v1_task_dir(task: str) -> Path:
    """@brief v1 任务目录（``gen_*`` 在 ``tasks/generated/<stem>/``）。

    @param task 任务名。
    @return 目录。
    """
    if task.startswith("gen_"):
        return ROOT / "tasks" / "generated" / task[len("gen_") :]
    return ROOT / "tasks" / task


def combo_of(document: dict, task: str, variant: str | None = None, *, dtype: str = DTYPE, size: str = SIZE) -> dict:
    """@brief 取一个 ``(task, dtype, size, variant)`` 的 G0 组合。

    @param document 索引文档。
    @param task 任务名。
    @param variant 启动变体（``None`` 为默认）。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @return 组合。
    """
    matches = [
        combo
        for combo in document["combos"]
        if combo["task"] == task and combo["dtype"] == dtype and combo["size"] == size and combo["variant"] == variant
    ]
    if len(matches) != 1:
        raise SystemExit(f"{task}@{variant}/{dtype}/{size}: expected exactly one G0 combo, found {len(matches)}")
    return matches[0]


def frozen_candidate(combo: dict, *, phase: str, contract: object | None = None) -> str:
    """@brief G0 ``naive_msl`` 正文 + 一条声明 v1 契约 grid/threads 的 ``taic-launch``。

    The launch block is what makes the candidate acceptable on the v2 track (which never invents
    a geometry) while v1 resolves exactly the same numbers it would have used by default
    (``math fast`` is the v1 default for a source without a launch block).

    The kernel body is the frozen G0 text in both phases.  The only difference is ownership of
    the contract constants: v1 prepends them itself (``Contract.assemble``), while a v2 candidate
    owns its own compile-time constants, so the v2 source carries the *archived* G0 contract
    header (``contract_text.header``) in front of the same body.  No constant is re-derived.

    @param combo G0 组合。
    @param phase ``v1`` 或 ``v2``。
    @param contract v2 合同（``phase="v2"`` 时用于区分实例符号/attrs 与候选自有的启动常量）。
    @return 候选源码。
    """
    frozen = combo["contract"]
    grid = ",".join(str(int(value)) for value in frozen["grid"])
    threads = ",".join(str(int(value)) for value in frozen["threads"])
    launch = (
        "/* taic-launch\n"
        "math fast\n"
        f"launch {frozen['kernel_name']} grid={grid} threads={threads}\n"
        "*/\n"
    )
    if phase == "v1":
        return launch + combo["naive_msl"]
    if contract is None:
        raise SystemExit("the v2 candidate needs the contract to decide which constants it owns")
    provided = set(contract.instance.symbols) | set(contract.instance.resolved_attrs)
    owned: list[str] = []
    for line in combo["contract_text"]["header"].splitlines():
        if not line.startswith("constant constexpr"):
            continue  # the v2 header already carries the std includes and ``using namespace metal``
        name = line.split()[3]
        if name not in provided:
            owned.append(line)
    prefix = "\n".join(owned) + "\n" if owned else ""
    return prefix + launch + combo["naive_msl"]


def record_of(task: str, combo: dict, source: str, report: dict, *, phase: str, size: str = SIZE) -> dict:
    """@brief 一个任务的性能记录（只取可比较字段），并拒绝缺已登记基线的报告。

    A registered baseline that the evaluator did not measure is a measurement failure, not a
    missing optional field.  It used to surface later as a ``KeyError`` inside the gate; here it
    is an explicit :class:`EvidenceError` naming the task, phase and baseline.

    @param task 任务名。
    @param combo G0 组合。
    @param source 候选源码。
    @param report 报告文档。
    @param phase ``v1`` 或 ``v2``。
    @param size 尺寸档位。
    @return 记录。
    @raises EvidenceError 当候选/已登记基线没有有效计时。
    """
    benchmark = report.get("benchmark") or {}
    candidate = benchmark.get("candidate") or {}
    baselines = benchmark.get("baselines") or {}
    record = {
        "task": task,
        "dtype": DTYPE,
        "size": size,
        "variant": None,
        "phase": phase,
        "status": report.get("status"),
        "stage": report.get("stage"),
        "mode": report.get("mode"),
        "seed": SEED,
        "reps": REPS,
        "cases": list(CASES),
        "candidate_sha256": sha256(source),
        "g0_body_sha256": sha256(combo["naive_msl"]),
        "g0_opspec_digest": combo["specialization"]["opspec_digest"],
        "g0_kernel_name": combo["contract"]["kernel_name"],
        "g0_grid": combo["contract"]["grid"],
        "g0_threads": combo["contract"]["threads"],
        "median_us": candidate.get("median_us"),
        "candidate_cv": candidate.get("cv"),
        "candidate_iters": candidate.get("iters"),
        "benchmark_attempts": benchmark.get("attempts"),
        "benchmark_noise_cv": benchmark.get("noise_cv"),
        "benchmark_rotation_sets": benchmark.get("rotation_sets"),
        "benchmark_protocol": benchmark.get("protocol"),
        "baselines": {
            name: {
                "median_us": (value or {}).get("median_us"),
                "implementation_id": next(
                    (entry["implementation_id"] for entry in combo["baselines"] if entry["name"] == name), None
                ),
            }
            for name, value in baselines.items()
        },
        "baseline_registrations": combo["baselines"],
        "baseline_implementation_digest": report.get("baseline_implementation_digest"),
        "baseline_environment_digest": report.get("baseline_environment_digest"),
        "evaluation_fingerprint": report.get("evaluation_fingerprint"),
        "contract": combo["contract"],
        "device_name": report.get("device_name"),
        "gpu_family": report.get("gpu_family"),
        "target_id": report.get("target_id"),
    }
    validate_record(record, phase=phase, size=size, where=f"{task}/{phase}/{size}")
    return record


def _positive(value: object, where: str) -> float:
    """@brief 要求一个有限正数计时。

    @param value 待检查值。
    @param where 证据位置。
    @return 浮点值。
    @raises EvidenceError 缺失、非数或非正。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{where}: missing or non-numeric timing {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise EvidenceError(f"{where}: invalid timing {value!r}")
    return number


def _registered_names(record: dict, where: str) -> list[str]:
    """@brief 记录里登记的 baseline 名（顺序即 G0 登记顺序）。

    @param record 性能记录。
    @param where 证据位置。
    @return 名字列表。
    @raises EvidenceError 字段缺失/重复/为空名字。
    """
    registrations = record.get("baseline_registrations")
    if not isinstance(registrations, list):
        raise EvidenceError(
            f"{where}: no baseline_registrations; this is pre-interleaving/stale evidence and must be re-measured"
        )
    names: list[str] = []
    for entry in registrations:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
            raise EvidenceError(f"{where}: invalid baseline registration {entry!r}")
        names.append(entry["name"])
    if len(set(names)) != len(names):
        raise EvidenceError(f"{where}: duplicate baseline registrations {names}")
    return names


def validate_record(record: dict, *, phase: str, size: str, where: str) -> None:
    """@brief 一条性能记录必须能支撑同口径的 v1/v2 比较。

    The checks are intentionally about evidence completeness, not about the measured value: the
    gate later applies the approved ratio.  A record without a registered baseline median is
    rejected here with the task/phase/baseline named.

    @param record 性能记录。
    @param phase ``v1`` 或 ``v2``。
    @param size 尺寸档位。
    @param where 证据位置。
    @raises EvidenceError 任一必需字段缺失或自相矛盾。
    """
    if record.get("phase") != phase or record.get("size") != size:
        raise EvidenceError(f"{where}: record declares {record.get('phase')}/{record.get('size')}")
    if record.get("dtype") != DTYPE:
        raise EvidenceError(f"{where}: dtype {record.get('dtype')!r} != {DTYPE!r}")
    if record.get("status") != "ok":
        raise EvidenceError(f"{where}: status {record.get('status')!r}, not 'ok'")
    if record.get("seed") != SEED or record.get("reps") != REPS or record.get("cases") != list(CASES):
        raise EvidenceError(f"{where}: seed/reps/cases differ from the frozen protocol")
    _positive(record.get("median_us"), f"{where}.median_us")
    rotation = record.get("benchmark_rotation_sets")
    if isinstance(rotation, bool) or not isinstance(rotation, int) or rotation < 1:
        raise EvidenceError(f"{where}: missing/invalid benchmark rotation ({rotation!r})")
    attempts = record.get("benchmark_attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise EvidenceError(f"{where}: missing/invalid benchmark attempts ({attempts!r})")
    names = _registered_names(record, where)
    baselines = record.get("baselines")
    if not isinstance(baselines, dict):
        raise EvidenceError(f"{where}: baselines is not an object")
    measured = list(baselines)
    if set(measured) != set(names):
        missing = sorted(set(names) - set(measured))
        extra = sorted(set(measured) - set(names))
        raise EvidenceError(
            f"{where}: registered baselines {names} but measured {sorted(measured)} "
            f"(missing={missing}, extra={extra}); rerun fresh with baseline interleaving"
        )
    for entry in record["baseline_registrations"]:
        name = entry["name"]
        value = baselines[name]
        if not isinstance(value, dict):
            raise EvidenceError(f"{where}.baselines[{name!r}]: not an object")
        _positive(value.get("median_us"), f"{where}.baselines[{name!r}].median_us")
        if value.get("implementation_id") != entry.get("implementation_id"):
            raise EvidenceError(
                f"{where}.baselines[{name!r}]: implementation identity {value.get('implementation_id')!r} "
                f"!= G0 registration {entry.get('implementation_id')!r}"
            )
    if names:
        digest = record.get("baseline_implementation_digest")
        if not isinstance(digest, str) or not digest:
            raise EvidenceError(f"{where}: missing baseline_implementation_digest")
    if not isinstance(record.get("target_id"), str) or not record["target_id"]:
        raise EvidenceError(f"{where}: missing target_id")
    if not isinstance(record.get("device_name"), str) or not record["device_name"]:
        raise EvidenceError(f"{where}: missing device_name")

def run_v1(document: dict, output: Path, *, only: tuple[str, ...], size: str = SIZE) -> None:
    """@brief 用 v1 任务路径测量冻结候选。

    @param document G0 索引。
    @param output 证据文件。
    @param only 只跑这些任务。
    @param size 尺寸档位。
    """
    sys.path.insert(0, str(ROOT))
    # The v1 phase runs in an isolated pre-migration task tree.  A tree assembled with a
    # symlinked ``tcenv`` resolves ``REPO_ROOT`` back to the production checkout, which would
    # silently load the *migrated* tasks and measure the wrong tree.  Pin both roots before the
    # task registry is imported and refuse if the task assembly still points elsewhere.
    import tcenv.track as track_module

    track_module.REPO_ROOT = ROOT
    from tasks import get_task
    import tasks._assembly as task_assembly

    if Path(task_assembly.TASKS_ROOT).resolve() != (ROOT / "tasks").resolve():
        raise SystemExit(
            f"the v1 task assembly resolves {task_assembly.TASKS_ROOT}, not the isolated {ROOT / 'tasks'}; "
            "refusing to measure the wrong task tree"
        )
    try:
        get_task("relu", DTYPE, size)
    except Exception as exc:  # noqa: BLE001 - a migrated task must refuse on the v1 path
        raise SystemExit(f"the v1 task path is already migrated ({exc}); run --phase v1 before the A5 migration")
    from tcenv import evaluate as evaluate_module

    # The worker is spawned with ``PYTHONPATH``/cwd from this module global.  In a fully copied
    # pre-migration tree it is already ``ROOT``; pinning it keeps a symlinked toolchain honest.
    evaluate_module.REPO_ROOT = ROOT
    evaluate = evaluate_module.evaluate

    records = []
    for task in TASKS:
        if only and task not in only:
            continue
        combo = combo_of(document, task, size=size)
        source = frozen_candidate(combo, phase="v1")
        document_v1 = json.loads((v1_task_dir(task) / "opspec.json").read_text(encoding="utf-8"))
        if document_v1.get("spec_version") != 1:
            raise SystemExit(f"{task}: the v1 opspec is already v2; the v1 phase must run before the migration")
        report = evaluate(
            task,
            DTYPE,
            size,
            source,
            mode="timed",
            seed=SEED,
            reps=REPS,
            cases=list(CASES),
            eval_cache=False,
        ).to_json()
        if report["status"] != "ok":
            raise SystemExit(f"{task}: v1 evaluation failed: {report['status']} {report.get('message')}")
        records.append(record_of(task, combo, source, report, phase="v1", size=size))
        print(f"v1 {task}: {records[-1]['median_us']:.3f} us", flush=True)
    payload = {
        "schema_version": 1,
        "phase": "v1",
        "kind": "frozen_g0_naive_candidate_on_the_v1_task_path",
        "head": head(),
        "source": source_provenance(),
        "dtype": DTYPE,
        "size": size,
        "reps": REPS,
        "seed": SEED,
        "cases": list(CASES),
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(records)} records)")


def run_v2(document: dict, output: Path, *, only: tuple[str, ...], size: str = SIZE) -> None:
    """@brief 用 v2 标准目录 + 真实 MetalTrackEvaluator 测量同一份候选字节。

    @param document G0 索引。
    @param output 证据文件。
    @param only 只跑这些任务。
    @param size 尺寸档位。
    """
    sys.path.insert(0, str(ROOT))
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract import discover_standard_tasks
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import REPO_ROOT

    locators = {locator.op_name: locator for locator in discover_standard_tasks(ROOT / "tasks", code_root=REPO_ROOT)}
    tracks = ROOT / "backends/metal/tracks.json"
    records = []
    for task in TASKS:
        if only and task not in only:
            continue
        locator = locators.get(task)
        if locator is None:
            raise SystemExit(f"{task}: the v2 standard entry did not discover this task")
        contract = locator.load()
        if (contract.dtype, contract.size) != (DTYPE, size):
            # The perf comparison uses a non-default instance, so the locator handed to
            # ``evaluate_task`` must describe the *same* instance: the worker rebuilds the task
            # from this locator, and a default-instance locator would run a different size than
            # the parent's ABI/geometry (the report would then not be about the requested combo).
            from tcenv.taskcontract import load_task_contract
            from tcenv.taskcontract.locator import TaskLocator
            from tcenv.taskcontract.pack import task_pack_closure
            from tcenv.track import REPO_ROOT

            contract = load_task_contract(ROOT / "tasks" / task, dtype=DTYPE, size=size)
            locator = TaskLocator(
                task_id=locator.task_id,
                op_name=contract.op_name,
                dtype=contract.dtype,
                size=contract.size,
                pack_digest=contract.pack_digest,
                deployment_pack_digest=task_pack_closure(ROOT / "tasks" / task, root=REPO_ROOT).digest,
                task_dir=ROOT / "tasks" / task,
                code_root=REPO_ROOT,
            )
        combo = combo_of(document, task, size=size)
        source = frozen_candidate(combo, phase="v2", contract=contract)
        result = evaluate_task(
            locator,
            track_id="msl-single",
            candidate_source=source,
            builder=MetalTrackBuilder(),
            evaluator=MetalTrackEvaluator(mode="timed", seed=SEED, reps=REPS, cases=CASES),
            geometry_resolver=msl_single_geometry_resolver(contract),
            track_registry_path=tracks,
        )
        report = result.report
        if report["status"] != "ok":
            raise SystemExit(f"{task}: v2 evaluation failed: {report['status']} {report.get('message')}")
        record = record_of(task, combo, source, report, phase="v2", size=size)
        record.update(
            {
                "track": result.track,
                "worker": (result.execution or {}).get("worker"),
                "geometry_identity": result.geometry_identity,
                "bundle_digest": result.bundle_digest,
                "artifact_set_digest": result.artifact_set_digest,
                "native_digest": result.native_digest,
                "measured_candidate_sha256": result.candidate_sha256,
                "presentation_sha256": result.presentation_sha256,
                "dispatch": [
                    [item["kernel"], item["grid"], item["threads"], item["threadgroup_bytes"], item["workspace"]]
                    for item in (result.execution or {}).get("dispatch", [])
                ],
                "provider": (result.execution or {}).get("provider"),
                "effective_correctness_only": result.effective_correctness_only,
                "reference_implementation": contract.reference.document(),
                "baseline_registrations_v2": [entry.document() for entry in contract.baselines],
            }
        )
        if record["measured_candidate_sha256"] != record["candidate_sha256"]:
            raise SystemExit(f"{task}: the measured candidate identity differs from the frozen bytes")
        records.append(record)
        print(f"v2 {task}: {record['median_us']:.3f} us", flush=True)
    payload = {
        "schema_version": 1,
        "phase": "v2",
        "kind": "same_frozen_candidate_on_the_v2_standard_entry",
        "head": head(),
        "source": source_provenance(),
        "dtype": DTYPE,
        "size": size,
        "reps": REPS,
        "seed": SEED,
        "cases": list(CASES),
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(records)} records)")


def evidence_file(phase: str, size: str, round_index: int) -> Path:
    """@brief 一个 ``(phase, size, round)`` 的证据文件。

    Round 1 keeps the canonical ``perf_<phase>_<size>.json`` name; later rounds add ``_r<N>``.
    Two alternating rounds are what cancel the machine-epoch drift at the launch-bound ``small``
    instance: a single v2 session compared against a single v1 session swings by more than 20%
    per task even when the two runtimes do the same work.

    @param phase ``v1`` 或 ``v2``。
    @param size ``small`` 或 ``paper``。
    @param round_index 从 1 开始的轮次。
    @return 路径。
    """
    suffix = "" if int(round_index) == 1 else f"_r{int(round_index)}"
    return EVIDENCE / f"perf_{phase}_{size}{suffix}.json"


def _fold_rounds(per_round: dict[int, dict[str, dict]]) -> dict[str, dict]:
    """@brief 把多轮测量折叠成每任务的中位数（任务内比较，跨轮抵消机器状态漂移）。

    Each task's candidate and baseline medians are the median over the alternating rounds; the
    identity fields (candidate sha, registrations, device) come from round 1 because a drift in
    any of them is a different measurement, not noise.

    @param per_round 轮次到 ``"<phase>_<size>"`` → ``task`` → 记录的映射。
    @return 折叠后的同一结构。
    """
    import statistics

    first = per_round[1]
    folded: dict[str, dict] = {}
    for key, records in first.items():
        folded[key] = {}
        for task, record in records.items():
            item = dict(record)
            item["median_us"] = statistics.median(per_round[index][key][task]["median_us"] for index in per_round)
            baselines = {}
            for name in record["baselines"]:
                value = dict(record["baselines"][name])
                value["median_us"] = statistics.median(
                    per_round[index][key][task]["baselines"][name]["median_us"] for index in per_round
                )
                baselines[name] = value
            item["baselines"] = baselines
            item["rounds"] = sorted(per_round)
            folded[key][task] = item
    return folded


def _same_identity(records: dict[str, dict], fields: tuple[str, ...], where: str) -> None:
    """@brief 同一 ``(phase,size)`` 的跨轮身份必须逐字段相同。

    @param records 轮次到 task→记录 的映射。
    @param fields 身份字段。
    @param where 证据位置。
    @raises EvidenceError 任一身份漂移。
    """
    first = records[sorted(records)[0]]
    for index in sorted(records)[1:]:
        for task, record in records[index].items():
            for field in fields:
                if record.get(field) != first[task].get(field):
                    raise EvidenceError(
                        f"{where} round {index} {task}.{field} drifted from round {sorted(records)[0]}"
                    )


def _expected_dispatch(record: dict) -> list:
    """@brief G0 单阶段默认组合的期望真实 dispatch。

    @param record 性能记录。
    @return ``[kernel, grid, threads, threadgroup_bytes, workspace]``。
    """
    return [
        [
            record["g0_kernel_name"],
            [int(v) for v in record["g0_grid"]],
            [int(v) for v in record["g0_threads"]],
            0,
            [],
        ]
    ]


def summarize_gate() -> dict:
    """@brief 汇总 v1/v2 同机、同候选正文、同测量口径的性能门。

    The approved ratio is ``geomean(v2_us / v1_us)`` over comparable tasks with baselines on
    the same device and baseline.  Both instances (fp16 ``small`` and ``paper``) are part of the
    gate; a paper-only result is not acceptance.  The folds use the median over alternating
    rounds, and every round must also clear the limit so a single lucky round cannot pass.

    Records from before the v2 runtime interleaved the registered baselines have no baseline
    medians.  They are rejected with :class:`EvidenceError` instead of being folded or skipped.

    @return gate 文档。
    @raises EvidenceError 证据缺 baseline/身份字段，或 v1/v2 口径不一致。
    """
    rounds = 0
    while evidence_file("v2", "paper", rounds + 1).is_file():
        rounds += 1
    if rounds == 0:
        raise SystemExit("no v2 evidence; run --phase v2 first")
    corpus = index()
    per_round: dict[int, dict[str, dict[str, dict]]] = {}
    documents: dict[tuple[int, str, str], dict] = {}
    for round_index in range(1, rounds + 1):
        loaded: dict[str, dict[str, dict]] = {}
        for phase in ("v1", "v2"):
            for size in ("small", "paper"):
                path = evidence_file(phase, size, round_index)
                if not path.is_file():
                    raise EvidenceError(f"round {round_index}: missing {path.relative_to(ROOT)}")
                document_evidence = json.loads(path.read_text(encoding="utf-8"))
                if document_evidence.get("phase") != phase or document_evidence.get("size") != size:
                    raise EvidenceError(f"round {round_index}: {path.relative_to(ROOT)} declares another phase or size")
                records = {
                    record.get("task"): record for record in (document_evidence.get("records") or [])
                }
                if set(records) != set(TASKS):
                    raise EvidenceError(
                        f"round {round_index} {phase}/{size}: task set {sorted(records)} != the 16 frozen tasks"
                    )
                for task, record in records.items():
                    validate_record(
                        record, phase=phase, size=size, where=f"round {round_index} {phase}/{size} {task}"
                    )
                loaded[f"{phase}_{size}"] = records
                documents[(round_index, phase, size)] = document_evidence
        per_round[round_index] = loaded

    identity_fields = (
        "candidate_sha256",
        "g0_body_sha256",
        "g0_opspec_digest",
        "g0_kernel_name",
        "g0_grid",
        "g0_threads",
        "baseline_implementation_digest",
        "target_id",
        "device_name",
        "benchmark_rotation_sets",
    )
    for phase in ("v1", "v2"):
        for size in ("small", "paper"):
            records_by_round = {round_index: per_round[round_index][f"{phase}_{size}"] for round_index in per_round}
            _same_identity(records_by_round, identity_fields, f"{phase}/{size}")

    comparable: list[str] = []
    correctness_only: list[str] = []
    for task in TASKS:
        registrations = None
        for size in ("small", "paper"):
            combo = combo_of(corpus, task, size=size)
            expected_names = [entry["name"] for entry in combo["baselines"]]
            for phase in ("v1", "v2"):
                record = per_round[1][f"{phase}_{size}"][task]
                if [entry["name"] for entry in record["baseline_registrations"]] != expected_names:
                    raise EvidenceError(
                        f"{task}/{size}/{phase}: baseline registrations {record['baseline_registrations']} "
                        f"!= G0 {combo['baselines']}"
                    )
                if record["g0_opspec_digest"] != combo["specialization"]["opspec_digest"]:
                    raise EvidenceError(f"{task}/{size}/{phase}: g0_opspec_digest != the frozen G0 corpus")
                if record["g0_body_sha256"] != sha256(combo["naive_msl"]):
                    raise EvidenceError(f"{task}/{size}/{phase}: g0_body_sha256 != the frozen G0 candidate")
                if record["g0_kernel_name"] != combo["contract"]["kernel_name"]:
                    raise EvidenceError(f"{task}/{size}/{phase}: G0 kernel name drifted")
                if [int(v) for v in record["g0_grid"]] != [int(v) for v in combo["contract"]["grid"]]:
                    raise EvidenceError(f"{task}/{size}/{phase}: G0 grid drifted")
                if [int(v) for v in record["g0_threads"]] != [int(v) for v in combo["contract"]["threads"]]:
                    raise EvidenceError(f"{task}/{size}/{phase}: G0 threads drifted")
            one = per_round[1][f"v1_{size}"][task]
            two = per_round[1][f"v2_{size}"][task]
            # The v2 candidate owns its compile-time constants, so the full source bytes differ
            # from the v1 assembly by exactly that header.  The frozen G0 body and launch are the
            # comparable identity.
            if one["g0_body_sha256"] != two["g0_body_sha256"]:
                raise EvidenceError(f"{task}/{size}: v1/v2 measured different frozen G0 bodies")
            if one["baseline_implementation_digest"] != two["baseline_implementation_digest"]:
                raise EvidenceError(f"{task}/{size}: v1/v2 baseline implementation identity differs")
            if one["benchmark_rotation_sets"] != two["benchmark_rotation_sets"]:
                raise EvidenceError(f"{task}/{size}: v1/v2 rotation sets differ")
            if one["target_id"] != two["target_id"] or one["device_name"] != two["device_name"]:
                raise EvidenceError(f"{task}/{size}: v1/v2 ran on different devices")
            dispatch = two.get("dispatch") or []
            if not dispatch:
                raise EvidenceError(f"{task}/{size}: v2 evidence has no real runtime dispatch")
            if dispatch != _expected_dispatch(two):
                raise EvidenceError(f"{task}/{size}: v2 dispatch {dispatch} != the G0 launch")
            if two.get("provider") != "candidate":
                raise EvidenceError(f"{task}/{size}: v2 dispatch provider {two.get('provider')!r} is not candidate")
            if two.get("effective_correctness_only") is not (not expected_names):
                raise EvidenceError(f"{task}/{size}: effective_correctness_only contradicts the registrations")
            track = two.get("track") or {}
            if track.get("track_id") != "msl-single":
                raise EvidenceError(f"{task}/{size}: v2 did not run the frozen msl-single track")
        if registrations is None:
            registrations = per_round[1]["v1_paper"][task]["baseline_registrations"]
        (comparable if registrations else correctness_only).append(task)

    # The two phases must use one timing protocol; only the task tree differs.
    v1_source = documents[(1, "v1", "paper")].get("source") or {}
    v2_source = documents[(1, "v2", "paper")].get("source") or {}
    if v1_source.get("protocol_digest") != v2_source.get("protocol_digest"):
        raise EvidenceError("v1/v2 protocol digests differ; the timing protocol is not the same code")
    if documents[(1, "v1", "paper")].get("head") != documents[(1, "v2", "paper")].get("head"):
        raise EvidenceError("v1/v2 evidence was produced from different repository heads")
    if v1_source.get("root") == v2_source.get("root"):
        raise EvidenceError("the v1 phase did not run in an isolated pre-migration task tree")
    if v1_source.get("tasks_digest") == v2_source.get("tasks_digest"):
        raise EvidenceError("v1/v2 evidence records the same task tree digest; the v1 tree is not pre-migration")

    evidence = _fold_rounds(per_round)

    def geomean(values: dict[str, float]) -> float:
        return math.exp(sum(math.log(value) for value in values.values()) / len(values))

    approved = json.loads((ROOT / "docs/架构/跨硬件算子生成分工-G0/performance-contract.json").read_text(encoding="utf-8"))
    limit = float(approved["mac_performance_regression_gate"]["approved_max_ratio"])
    gates: dict[str, dict] = {}
    for size in ("paper", "small"):
        ratios = {
            task: evidence[f"v2_{size}"][task]["median_us"] / evidence[f"v1_{size}"][task]["median_us"]
            for task in comparable
        }
        measured_baselines = {
            task: {
                name: {
                    "v1_us": evidence[f"v1_{size}"][task]["baselines"][name]["median_us"],
                    "v2_us": evidence[f"v2_{size}"][task]["baselines"][name]["median_us"],
                    "implementation_id": evidence[f"v2_{size}"][task]["baselines"][name]["implementation_id"],
                }
                for name in evidence[f"v1_{size}"][task]["baselines"]
            }
            for task in comparable
        }
        value = geomean(ratios)
        round_geomeans = {}
        for round_index in range(1, rounds + 1):
            one = per_round[round_index]
            round_geomeans[str(round_index)] = geomean(
                {
                    task: one[f"v2_{size}"][task]["median_us"] / one[f"v1_{size}"][task]["median_us"]
                    for task in comparable
                }
            )
        gates[size] = {
            "size": size,
            "rounds": rounds,
            "round_geomeans": round_geomeans,
            "ratios": {task: round(ratio, 6) for task, ratio in ratios.items()},
            "geomean": value,
            # The approved metric is the median-over-rounds geomean, not the fastest round and
            # not a paper-only instance.  Round geomeans are evidence of stability, not a second
            # threshold that could reject a valid measurement on a cold first round.
            "pass": value <= limit,
            "round_pass": all(item <= limit for item in round_geomeans.values()),
            "measured_baselines": measured_baselines,
        }
    comparable_tasks = {
        task: {
            "dtype": DTYPE,
            "sizes": ["paper", "small"],
            "g0_opspec_digest": evidence["v1_paper"][task]["g0_opspec_digest"],
            "g0_body_sha256": evidence["v1_paper"][task]["g0_body_sha256"],
            "candidate_sha256": {
                "v1": evidence["v1_paper"][task]["candidate_sha256"],
                "v2": evidence["v2_paper"][task]["candidate_sha256"],
            },
            "kernel_name": evidence["v1_paper"][task]["g0_kernel_name"],
            "grid": evidence["v1_paper"][task]["g0_grid"],
            "threads": evidence["v1_paper"][task]["g0_threads"],
            "baseline_implementations": sorted(
                entry["implementation_id"] for entry in evidence["v2_paper"][task]["baseline_registrations"]
            ),
            "gpu_family": evidence["v2_paper"][task]["gpu_family"],
            "device_name": evidence["v2_paper"][task]["device_name"],
        }
        for task in comparable
    }
    result = {
        "schema_version": 1,
        "metric": "geomean(v2_median_us / v1_median_us) over comparable tasks with baselines, per instance",
        "approved_max_ratio": limit,
        "gate_instances": [gates["paper"], gates["small"]],
        "gate_pass": all(gates[size]["pass"] for size in ("paper", "small")),
        "gate_geomean": gates["paper"]["geomean"],
        "gate_ratios": gates["paper"]["ratios"],
        "small_geomean": gates["small"]["geomean"],
        "small_ratios": gates["small"]["ratios"],
        "comparable_tasks": comparable_tasks,
        "correctness_only_tasks": [task for task in TASKS if task not in comparable],
        "measurement": {
            "rounds": rounds,
            "order": "v1 then v2, alternating per round (task-level medians over rounds)",
            "reps": REPS,
            "seed": SEED,
            "cases": list(CASES),
            "protocol": "tcenv.bench interleaved, median per launch, first-quiet-attempt selection",
            "protocol_digest": v1_source.get("protocol_digest"),
        },
        "v1_phase": {
            "reason": (
                "the v1 assembly is deleted by A5, so the v1 numbers are measured from the preserved pre-migration "
                "read-only checkout with the same shared timing protocol (tcenv/bench.py + tcenv/evaluate.py), the "
                "same candidate body/launch and the same registered baselines; nothing is restored in production"
            ),
            "head": documents[(1, "v1", "paper")].get("head"),
            "source": v1_source,
            "evidence": [
                str(evidence_file("v1", size, round_index).relative_to(ROOT))
                for size in ("small", "paper")
                for round_index in range(1, rounds + 1)
            ],
        },
        "v2_phase": {
            "reason": "live, re-runnable migrated-path measurement with the same candidate body, launch and baselines",
            "head": documents[(1, "v2", "paper")].get("head"),
            "source": v2_source,
            "evidence": [
                str(evidence_file("v2", size, round_index).relative_to(ROOT))
                for size in ("small", "paper")
                for round_index in range(1, rounds + 1)
            ],
        },
    }
    if set(comparable) | set(correctness_only) != set(TASKS) or set(comparable) & set(correctness_only):
        raise EvidenceError("the comparable/correctness-only partition is not a partition of the 16 tasks")
    (EVIDENCE / "perf_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

def main() -> None:
    """@brief 命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("v1", "v2", "gate"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--size", choices=("small", "paper"), default=SIZE)
    parser.add_argument("--round", type=int, default=1, help="alternating measurement round (1-based)")
    args = parser.parse_args()
    if args.phase == "gate":
        result = summarize_gate()
        for gate in result["gate_instances"]:
            print(
                f"gate {gate['size']}: geomean {gate['geomean']:.6f} <= {result['approved_max_ratio']} "
                f"({len(gate['ratios'])} comparable tasks): {'pass' if gate['pass'] else 'FAIL'}"
            )
        print(f"gate overall: {'pass' if result['gate_pass'] else 'FAIL'}")
        return
    document = index()
    output = args.output or evidence_file(args.phase, args.size, args.round)
    if args.phase == "v1":
        run_v1(document, output, only=tuple(args.only), size=args.size)
    else:
        run_v2(document, output, only=tuple(args.only), size=args.size)


if __name__ == "__main__":
    main()
