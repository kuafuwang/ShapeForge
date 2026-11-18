"""N-way interleaved timing of candidate kernels inside one GPU window.

Why this exists: a kernel session has to answer "is variant X faster than the
bytes I submitted". One ``--timed`` run answers that only against a recorded
seed stamp from an earlier window, and the same bytes drift by a few percent
between windows, so a 1-2 % edit is not decidable that way. The field reports
that every round therefore hand-copied an in-process probe, and that those
copies disagreed with each other. This module is the repository-owned version
of that probe.

Two properties it must not lose:

* It times the **same physical quantity** the scored benchmark reports
  (``candidate_gpu_us_median``): the fused command-buffer GPU timestamp, not
  the wall clock. A wall clock sample on a 3 us kernel is about 150 us of
  encode/commit/wait, which hides every difference this tool is asked to rank.
* It is **read-only**. It does not write the self-check ledger, does not debit
  the ``--timed`` quota, and cannot promote anything. A ranking is a session's
  own working measurement, never publication evidence.

Correctness is taken from the task's own quick evaluation, not from a second
comparison written here. A numerically wrong candidate is still timed, because
ablation variants (cut a loop, disable an audit) are deliberately wrong and
their cost structure is the thing being measured; such a row is marked
``correct: false`` and kept out of the ranking. A candidate that is *unsafe* is
never timed: a memory violation inside a timing loop is how the machine froze
on 2026-11-17, and this tool has no authority to replay that.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from taic.paired_compare import (
    PRODUCTION_KMIN,
    PRODUCTION_REPS,
    PRODUCTION_WARMUP_REPS,
    PairedCompareError,
)

#: 这份文档的 schema。只读计时，不是评测报告，也不是发布证据。
SCHEMA = "tcenv.ab.1"
#: 一次 ``ab`` 最多接受的文件数。现场反馈里的上限是 16。
MAX_FILES = 16


def _digests(paths) -> list[str]:
    out = []
    for path in paths:
        out.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return out


def _paired_ratio_to_first(samples: dict[str, tuple[float, ...]], first: str, name: str) -> float | None:
    """@brief 逐轮配对比的中位数：第 i 轮的两侧样本按下标相除。

    ``interleaved`` 每轮把顺序打乱后每个 runner 各测一批，所以同一个下标就是同一轮。
    这比两个独立中位数相除更稳，也是现场反馈要求的形状。
    """
    left = samples[first]
    right = samples[name]
    size = min(len(left), len(right))
    if size == 0:
        return None
    ratios = [left[index] / right[index] for index in range(size) if right[index] > 0.0]
    if not ratios:
        return None
    ratios.sort()
    middle = len(ratios) // 2
    if len(ratios) % 2:
        return float(ratios[middle])
    return float((ratios[middle - 1] + ratios[middle]) / 2.0)


#: 会被计时的评测状态。``incorrect`` 在内，因为它就是消融要看的东西。
TIMABLE = frozenset({"ok", "incorrect"})


def compare_many(
    runners: dict[str, Any],
    *,
    verdicts: dict[str, dict] | None = None,
    seed: int = 7,
    reps: int = PRODUCTION_REPS,
    warmup_reps: int = PRODUCTION_WARMUP_REPS,
    kmin: int = PRODUCTION_KMIN,
) -> dict:
    """@brief 在一段 GPU 窗口里交错计时多颗 runner，给出每颗的中位数与配对比。

    @param runners 标签到 ``run_k(k) -> 整批秒数``。第一个标签是对照，必须通过正确性。
    @param verdicts 标签到 quick 评测结论（``status``/``ok``/``worst_scaled_err``）。
        没有结论时按未检查处理，行仍然产出但不可排序。
    @param seed 只打乱计时顺序的种子。
    @param reps 计入统计的轮数。
    @param warmup_reps 丢弃的预热轮数。
    @param kmin 每批至少启动几次，让一批转过全部缓冲区组。
    @return ``tcenv.ab.1`` 文档。``timed_by`` 说明这些数字是 GPU 时间戳还是墙钟。
    """
    from tcenv.bench import interleaved

    labels = list(runners)
    if len(labels) < 2:
        raise PairedCompareError("ab needs at least two candidate files to rank")
    if len(labels) > MAX_FILES:
        raise PairedCompareError(f"ab takes at most {MAX_FILES} candidate files, got {len(labels)}")
    decisions = {label: dict((verdicts or {}).get(label) or {}) for label in labels}
    first = labels[0]
    control = decisions[first]
    if not control.get("ok"):
        raise PairedCompareError(
            "ab needs a correct control: the first file must pass the task's quick evaluation "
            f"(got {control.get('status') or 'not checked'})"
        )
    started = time.perf_counter()
    stats = interleaved(
        runners,
        reps=int(reps),
        seed=int(seed),
        warmup_reps=int(warmup_reps),
        kmin=int(kmin),
    )
    elapsed = time.perf_counter() - started
    samples = {name: tuple(stats[name].samples_s) for name in labels}
    clocks = set()
    for runner in runners.values():
        reader = getattr(runner, "clock_of", None)
        clocks.add(str(reader()) if callable(reader) else "wall_clock")
    timed_by = "gpu_timestamp" if clocks == {"gpu_timestamp"} else "wall_clock"
    rows = []
    for name in labels:
        row = stats[name]
        decision = decisions[name]
        rankable = bool(decision.get("ok"))
        rows.append({
            "label": name,
            "median_us": float(row.median_s) * 1e6,
            "min_us": float(row.min_s) * 1e6,
            "mean_us": float(row.mean_s) * 1e6,
            "cv": float(row.cv),
            "reps": int(row.reps),
            "iters": int(row.iters),
            "paired_ratio_to_first": 1.0 if name == first else _paired_ratio_to_first(samples, first, name),
            "correct": rankable,
            "status": str(decision.get("status") or "not_checked"),
            "worst_scaled_err": decision.get("worst_scaled_err"),
            "rankable": rankable,
        })
    return {
        "schema": SCHEMA,
        "readonly": True,
        "control": first,
        "timed_by": timed_by,
        "seed": int(seed),
        "reps": int(reps),
        "warmup_reps": int(warmup_reps),
        "kmin": int(kmin),
        "elapsed_s": elapsed,
        "rows": rows,
        # 只有通过正确性的行能参与「哪版更快」。算错的行被计时、被标出来，但不排名。
        "ranking": _ranking(rows, first),
        "promotable": False,
        "note": (
            "read-only in-process ranking; it does not debit the --timed quota and is not "
            "publication evidence"
        ),
    }


def _ranking(rows: list[dict], control: str) -> dict:
    """@brief 只让通过正确性的行参与排序，并说清算错的行为什么不在里面。

    @param rows ``compare_many`` 产出的行。
    @param control 对照标签。
    @return ``{"control", "order", "excluded"}``。可排序的行不足两行时 ``order`` 为空。
    """
    rankable = [row for row in rows if row.get("rankable")]
    excluded = [
        {"label": row["label"], "status": row.get("status"), "worst_scaled_err": row.get("worst_scaled_err")}
        for row in rows if not row.get("rankable")
    ]
    if len(rankable) < 2:
        return {"control": control, "order": [], "excluded": excluded,
                "reason": "fewer than two files passed the quick correctness gate"}
    # paired_ratio_to_first 是 control/文件，所以数值越大表示这份文件越快。
    ordered = sorted(
        rankable,
        key=lambda row: (row["paired_ratio_to_first"] is None, -(row["paired_ratio_to_first"] or 0.0)),
    )
    return {
        "control": control,
        # 比值是 control/文件：越大越快，所以排在前面的先出现。
        "order": [row["label"] for row in ordered],
        "excluded": excluded,
    }


def _quick_verdict(source: str, *, task: str, dtype: str, size: str, target_id: str, seed: int) -> dict:
    """@brief 用任务自己的 quick 评测给出这份候选的正确性结论。

    这里不另写一份比较：容差、scaled error 和用例集都由任务合同决定。
    正确性只由仓库的 ``tolerances + scaled_error`` 判定，这条不变量不因为
    多了一个计时工具而改变。

    @return ``{"status", "ok", "worst_scaled_err", "allow", "message"}``。
    """
    from taic.task_binding import bind_task
    from tcenv.backends.metal.search_candidate import evaluate_candidate
    from tcenv.target import load_target

    from tcenv.contract_errors import TaskContractError

    binding = bind_task(task, dtype, size, target_id=target_id)
    if not binding.is_v2:
        raise PairedCompareError(f"task {task!r} is not a v2 standard entry")
    try:
        run = evaluate_candidate(
            binding.locator, source, target=load_target(target_id), mode="quick",
            seed=int(seed), reps=1, cases=None, track_id=binding.track_id,
        )
    except TaskContractError as exc:
        # 交付形态或启动契约不合规是那一份候选自己的事，不该让整次排序崩掉。
        return {"status": "rejected", "ok": False, "message": str(exc)[:300]}
    result = run.result
    if result is None:
        return {"status": "rejected", "ok": False, "message": "the candidate did not reach an evaluation"}
    report = dict(result.report)
    correctness = report.get("correctness") if isinstance(report.get("correctness"), dict) else {}
    cases = [case for case in (correctness.get("cases") or []) if isinstance(case, dict)]
    worst = None
    allow = None
    for case in cases:
        value = case.get("scaled_err")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            worst = max(worst or 0.0, float(value))
        cap = case.get("allow")
        if isinstance(cap, (int, float)) and not isinstance(cap, bool):
            allow = max(allow or 0.0, float(cap))
    return {
        "status": str(report.get("status") or ""),
        "ok": bool(correctness.get("ok")) and str(report.get("status")) == "ok",
        "worst_scaled_err": worst,
        "allow": allow,
        "message": str(report.get("message") or "")[:300],
    }



def _worker_main(request: dict) -> dict:
    """@brief 有界 worker：独占 GPU 锁，绑定 N 颗候选，在一段窗口里交错计时。

    心跳和锁与 ``tcenv.v2_worker`` 的评测 worker 相同，父进程不创建 ``MetalContext``。
    """
    from tcenv.backends.metal.paired_compare import assemble_v2_group, runner_for_prepared
    from tcenv.backends.metal.runtime import get_context
    from tcenv.gpulock import gpu_lock
    from tcenv.v2_worker import RESULT_SCHEMA, _install_heartbeat

    payload = request["payload"]
    target_id = str(request.get("target_id") or payload.get("target_id") or "apple-m4-max")
    out = Path(request["out"])
    try:
        heartbeat = request.get("heartbeat")
        if not heartbeat:
            raise PairedCompareError("the ab worker needs the supervisor heartbeat path")
        _install_heartbeat(str(heartbeat), target_id)
        with gpu_lock(
            f"ab timing: {payload.get('task')} {payload.get('dtype')} {payload.get('size')}",
            exclusive=True,
        ):
            labels = list(payload["labels"])
            paths = list(payload["paths"])
            # 正确性先问任务自己的 quick 评测。算错的仍然计时（消融要那个时间），
            # 不安全的绝不计时：计时循环里重放一次内存越界就是 2026-11-17 那次冻结。
            verdicts = {}
            rejected = []
            for label, path in zip(labels, paths, strict=True):
                verdict = _quick_verdict(
                    Path(path).read_text(encoding="utf-8"),
                    task=payload["task"], dtype=payload["dtype"], size=payload["size"],
                    target_id=target_id, seed=int(payload.get("seed", 7)),
                )
                verdicts[label] = verdict
                if str(verdict.get("status")) not in TIMABLE:
                    rejected.append({
                        "file": label,
                        "status": verdict.get("status"),
                        "reason": verdict.get("message") or "the quick evaluation did not pass",
                    })
            timable = [label for label in labels if label not in {item["file"] for item in rejected}]
            if len(timable) < 2:
                document = {
                    "schema": SCHEMA, "readonly": True, "timed_by": "gpu_timestamp",
                    "control": "", "seed": int(payload.get("seed", 7)),
                    "reps": int(payload.get("reps", PRODUCTION_REPS)),
                    "warmup_reps": PRODUCTION_WARMUP_REPS, "kmin": PRODUCTION_KMIN,
                    "elapsed_s": 0.0, "rows": [], "rejected": rejected,
                    "ranking": {"control": "", "order": [], "excluded": [], "reason": "fewer than two timable files"},
                    "promotable": False,
                    "note": "read-only in-process ranking; not publication evidence",
                }
            else:
                keep = [paths[labels.index(label)] for label in timable]
                document = None
                context = get_context()
                bound = assemble_v2_group(
                    keep, task=payload["task"], dtype=payload["dtype"],
                    size=payload["size"], seed=int(payload.get("seed", 7)),
                    track_id=str(payload.get("track_id") or "msl-single"), context=context,
                    target_id=target_id,
                )
                runners = {
                    label: runner_for_prepared(prepared)
                    for label, prepared in zip(timable, bound, strict=True)
                }
                document = compare_many(
                    runners, verdicts={label: verdicts[label] for label in timable},
                    seed=int(payload.get("seed", 7)),
                    reps=int(payload.get("reps", PRODUCTION_REPS)),
                )
                document["rejected"] = rejected
        document["sha256"] = list(payload.get("sha256") or [])
        envelope = {
            "schema_version": RESULT_SCHEMA,
            "report": {"status": "ok", "stage": "ab", "message": ""},
            "execution": {"provider": "bounded-v2-worker", "ab": True},
            "ab": document,
        }
    except Exception as exc:  # noqa: BLE001 - the parent reads this envelope instead of a traceback
        envelope = {
            "schema_version": RESULT_SCHEMA,
            "report": {"status": "crash", "stage": "ab", "message": f"{type(exc).__name__}: {exc}"[:500]},
            "execution": None,
            "ab": {
                "schema": SCHEMA,
                "readonly": True,
                "status": "crash",
                "reason": f"{type(exc).__name__}: {exc}"[:500],
                "rows": [],
                "promotable": False,
            },
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
    return envelope


def bench_paths(paths, *, task: str, dtype: str, size: str, seed: int = 7,
                track_id: str = "msl-single", reps: int = PRODUCTION_REPS,
                target_id: str = "apple-m4-max") -> dict:
    """@brief 父进程入口：把手里的候选文件交给有界 worker 做 N 路交错计时。"""
    from tcenv.backends.metal.kernel_pipeline import configured_compile_deadline
    from tcenv.backends.metal.paired_compare import WORKER_LAUNCH_DEADLINE_S, WORKER_TIMEOUT_S
    from tcenv.v2_worker import run_bounded_v2

    ordered = [Path(item) for item in paths]
    if len(ordered) < 2:
        raise PairedCompareError("ab needs at least two candidate files to rank")
    if len(ordered) > MAX_FILES:
        raise PairedCompareError(f"ab takes at most {MAX_FILES} candidate files, got {len(ordered)}")
    if len({str(item.resolve()) for item in ordered}) != len(ordered):
        raise PairedCompareError("the same candidate file appears twice in one ab call")
    for item in ordered:
        if not item.is_file():
            raise PairedCompareError(f"no such candidate file: {item}")
    labels = [item.name for item in ordered]
    if len(set(labels)) != len(labels):
        raise PairedCompareError("two candidate files share a name; ab labels rows by file name")
    payload = {
        "task": task,
        "dtype": dtype,
        "size": size,
        "track_id": track_id,
        "target_id": target_id,
        "seed": int(seed),
        "reps": int(reps),
        "paths": [str(item.resolve()) for item in ordered],
        "labels": labels,
        "sha256": _digests(ordered),
    }
    result = run_bounded_v2(
        payload,
        timeout=WORKER_TIMEOUT_S,
        launch_deadline_s=WORKER_LAUNCH_DEADLINE_S,
        compile_deadline_s=configured_compile_deadline(worker_timeout=WORKER_TIMEOUT_S),
        worker_module="tcenv.backends.metal.ab_bench",
    )
    document = result.get("ab") if isinstance(result, dict) else None
    if isinstance(document, dict):
        report = result.get("report") if isinstance(result.get("report"), dict) else {}
        if not document.get("status") and report.get("status") and report["status"] != "ok":
            document = dict(document)
            document["status"] = report.get("status")
        return document
    message = ""
    status = "crash"
    if isinstance(result, dict):
        message = str(result.get("message") or "")
        status = str(result.get("status") or status)
    return {
        "schema": SCHEMA,
        "readonly": True,
        "status": status,
        "reason": (message or "the bounded ab worker returned no measurement")[:500],
        "rows": [],
        "promotable": False,
    }


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m tcenv.backends.metal.ab_bench REQUEST.json``。"""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: python -m tcenv.backends.metal.ab_bench REQUEST.json\n")
        return 2
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    _worker_main(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
