"""Hardware-feature report over the runs on disk (CLI). The detector itself lives in ``taic/hwfeatures.py``.

What it does: for every run directory it reads each candidate's kernel and ``report.json``, builds a ``HardwareFeatureReport`` (source, compiled AIR and
measured evidence, verdicts per feature) and writes ``REPORT.md``, ``data.json`` and ``kernels.jsonl``; it also analyses the validation set (the generated
reference GEMMs, the naive kernels, negative cases) and computes win rates of the features over accepted wins and the rest. See ``docs/hardware_features.md``.

    uv run python -m experiments.hw_features runs/apple-m4-max/p1 runs/apple-m4-max/suite_full --out runs/hw_features
    uv run python -m experiments.hw_features --source kernel.metal --task gemm        # one file

Only imports from ``tcenv`` / ``tasks`` happen lazily and read-only (contract header, task constants); nothing is written outside ``--out``.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


from taic.hwfeatures import *  # noqa: F401,F403 - the detector API stays importable from here
from taic.hwfeatures import _SIMD_REDUCE, _get_task, _read_json  # noqa: F401

REPO = Path(__file__).resolve().parent.parent


# ======================================================================================================================================
# 7b. run discovery
# ======================================================================================================================================


@dataclass
class KernelRef:
    kid: str
    run: str
    label: str
    kind: str  # candidate | best | naive | reference | synthetic
    source: str
    task: str | None
    dtype: str | None
    size: str = "paper"
    report: dict | None = None
    trace: dict | None = None
    partial: bool = False
    idx: int | None = None
    meta: dict = field(default_factory=dict)


def _infer_task(run_dir: Path) -> tuple[str | None, str | None]:
    """@brief 从目录名猜出任务名与数据类型。

    @param run_dir 运行目录。
    @return ``(任务名, dtype)``；猜不出时对应项为 ``None``。
    """
    names = [run_dir.name, run_dir.parent.name]
    for n in names:
        m = re.match(r"^(?:\d{8}-\d{6}_)?(.+?)_(fp16|bf16|fp32)(?:_.*)?$", n)
        if m:
            return m.group(1), m.group(2)
    for n in names:
        if n.startswith("sem_") or n in EXPECTED:
            return n, "fp16"
    return None, None


def discover_runs(roots: Iterable[Path]) -> list[Path]:
    """@brief 找出所有含 ``candidates/`` 的运行目录。

    Directories holding a ``candidates/`` folder (a run), skipping ``_interrupted*`` backups.

    @param roots 起始目录或目录列表。
    @return 去重后的运行目录列表，目录不存在时跳过。
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        cands = [root] if (root / "candidates").is_dir() else sorted(p.parent for p in root.rglob("candidates") if p.is_dir())
        for d in cands:
            if any(part.startswith("_interrupted") for part in d.parts) or d in seen:
                continue
            seen.add(d)
            found.append(d)
    return found


def load_run(run_dir: Path, runs_root: Path) -> tuple[dict, list[KernelRef]]:
    """@brief 读一次运行的概况与每个候选内核的引用。

    (run info, kernels) of one run directory. ``partial`` is True when the run has no ``result.json`` (still running or interrupted).

    @param run_dir 运行目录。
    @param runs_root 运行根目录，用来算相对名字。
    @return ``(运行概况字典, KernelRef 列表)``。
    """
    res = _read_json(run_dir / "result.json")
    state = _read_json(run_dir / "state.json")
    info = res or state or {}
    task = info.get("task")
    dtype = info.get("dtype")
    if not task:
        task, dt2 = _infer_task(run_dir)
        dtype = dtype or dt2
    size = info.get("size") or "paper"
    partial = res is None
    trace = {t["idx"]: t for t in (info.get("trace") or []) if isinstance(t, dict) and "idx" in t}
    try:
        rel = str(run_dir.relative_to(runs_root))
    except ValueError:
        rel = run_dir.name
    best_label = info.get("best_label")
    run = {"run": rel, "task": task, "dtype": dtype, "size": size, "partial": partial, "model": info.get("model"), "best_label": best_label, "stop_reason": info.get("stop_reason"),
           "budget_usd": info.get("budget_usd"), "spent_usd": info.get("spent_usd"), "confirmed_speedup": info.get("confirmed_speedup"), "best_speedup": info.get("best_speedup"),
           "rounds": info.get("rounds", info.get("round")), "has_trace": bool(trace)}
    refs: list[KernelRef] = []
    best_running = -1.0
    for cd in sorted((run_dir / "candidates").iterdir()) if (run_dir / "candidates").is_dir() else []:
        km = cd / "kernel.metal"
        if not km.exists():
            continue
        m = re.match(r"^(\d+)_(.+)$", cd.name)
        idx, label = (int(m.group(1)), m.group(2)) if m else (None, cd.name)
        try:
            src = km.read_text()
        except OSError:
            continue
        rep = _read_json(cd / "report.json")
        tr = dict(trace.get(idx) or {})
        sp = (rep or {}).get("speedup")
        if not tr and rep is not None:  # no trace on disk: reconstruct the acceptance order
            ok = rep.get("status") == "ok" and sp is not None
            improved = bool(ok and sp > best_running * 1.0)
            if ok and sp > best_running:
                best_running = sp
            tr = {"status": rep.get("status"), "speedup": sp, "improved": improved, "reconstructed": True}
        refs.append(KernelRef(kid=f"{rel}/{cd.name}", run=rel, label=label, kind="best" if label == best_label else "candidate", source=src, task=task, dtype=dtype, size=size, report=rep,
                              trace=tr or None, partial=partial, idx=idx))
    return run, refs


def load_naive_measurements(runs_root: Path) -> dict[tuple[str, str], dict]:
    """@brief 从各套件目录收集手写朴素核的缓存测量。

    task, dtype -> the stored summary of the naive kernel (``naive_<task>_<dtype>.json``) from any suite directory.

    @param runs_root 运行根目录。
    @return ``{(任务, dtype): 朴素核测量字典}``。
    """
    out: dict[tuple[str, str], dict] = {}
    for p in sorted(runs_root.rglob("naive_*_*.json")):
        if any(part.startswith("_interrupted") for part in p.parts):
            continue
        m = re.match(r"naive_(.+)_(fp16|bf16)\.json$", p.name)
        d = _read_json(p)
        if m and d and d.get("candidate_us"):
            out[(m.group(1), m.group(2))] = {**d, "_file": str(p)}
    return out


def ceilings_for(task: str | None, dtype: str, size: str, naive: dict, calib: dict | None, runs_root: Path) -> Ceilings:
    """@brief 算一个任务的``天花板``：峰值与朴素核达到的水平。

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模档位。
    @param naive 朴素核测量表。
    @param calib 硬件标定；为 ``None`` 时只用朴素核推。
    @param runs_root 运行根目录。
    @return 填好峰值与朴素核吞吐的 ``Ceilings``；任务不存在时是空的天花板。
    """
    c = Ceilings(source="")
    if not task:
        return c
    try:
        t = _get_task(task, dtype, size)
    except Exception:  # noqa: BLE001
        return c
    if calib:
        c.peak_gbps = calib.get("peak_gbps")
        c.peak_tflops = (calib.get("peak_tflops") or {}).get(dtype)
    nv = naive.get((base_task(task), dtype)) if "@" not in task else None
    if nv and nv.get("status", "ok") == "ok" and nv.get("candidate_us"):
        us = nv["candidate_us"]
        c.naive_gbps = t.min_bytes / (us * 1e-6) / 1e9
        c.naive_tflops = t.flops / (us * 1e-6) / 1e12 if t.flops else None
        c.source = nv.get("_file", "")
    return c


# ======================================================================================================================================
# 8. validation set: reference GEMMs, naive kernels, negative cases
# ======================================================================================================================================

NEGATIVES = {
    "comment_only": ("kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
                     "    // TODO: use simdgroup_multiply_accumulate with simdgroup_float8x8 and simd_sum, threadgroup_barrier, half4 loads\n"
                     "    /* simdgroup_load(a, tile, 8); #pragma unroll */\n"
                     "    y[i] = x[i];\n}\n"),
    "string_only": ('kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n'
                    '    const char* s = "simdgroup_multiply_accumulate(c, a, b, c); simd_sum(v); threadgroup half buf[4096];";\n    y[i] = x[i];\n}\n'),
    "scalar_naive": ("kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n    y[i] = half(float(x[i]) * 2.0f);\n}\n"),
}


def validation_sets(runs_root: Path) -> list[KernelRef]:
    """@brief 造出验证集：参考 GEMM、朴素核与负例。

    Reference GEMMs of ``experiments.contract_sensitivity`` (sources are generated, never run) and the 16 naive kernels.

    @param runs_root 运行根目录，用其中的契约实测结果补上计时。
    @return 用于检验检测器本身的 ``KernelRef`` 列表。
    """
    refs: list[KernelRef] = []
    sys.path.insert(0, str(REPO))
    try:
        from experiments.contract_sensitivity import VARIANTS, gemm_source

        rows = {r["variant"]: r for r in (_read_json(runs_root / "contract_sensitivity" / "gemm_fp16_paper.json") or [])}
        for v in VARIANTS:
            try:
                src = gemm_source(v, "fp16")
            except ValueError:
                continue
            name = "gemm" if v == "64x64x32t128" else f"gemm@{v}"
            row = rows.get(v)
            rep = None
            if row and row.get("status") == "ok":
                rep = {"status": "ok", "speedup": row.get("speedup"), "candidate_us": row.get("candidate_us"), "baseline_us": row.get("baseline_us"),
                       "roofline": {"achieved_tflops": row.get("tflops")}, "pipeline": {"max_total_threads_per_threadgroup": (row.get("pipeline") or {}).get("max_threads"),
                                                                                          "static_threadgroup_bytes": (row.get("pipeline") or {}).get("static_threadgroup_memory")}}
            refs.append(KernelRef(kid=f"reference/gemm@{v}", run="reference", label=v, kind="reference", source=src, task=name, dtype="fp16", report=rep))
    except Exception as e:  # noqa: BLE001
        refs.append(KernelRef(kid="reference/_error", run="reference", label=str(e)[:80], kind="error", source="", task=None, dtype=None))
    try:
        from tasks import list_tasks

        for t in list_tasks(True):
            try:
                tk = _get_task(t, "fp16", "paper")
            except Exception:  # noqa: BLE001
                continue
            if tk.naive_msl:
                refs.append(KernelRef(kid=f"naive/{t}", run="naive", label=t, kind="naive", source=tk.naive_msl, task=t, dtype="fp16"))
    except Exception:  # noqa: BLE001
        pass
    for n, s in NEGATIVES.items():
        refs.append(KernelRef(kid=f"negative/{n}", run="negative", label=n, kind="synthetic", source=s, task=None, dtype=None))
    return refs


# ======================================================================================================================================
# 9. analysis driver, aggregation, report
# ======================================================================================================================================


def analyze_refs(refs: list[KernelRef], *, naive: dict, calib: dict | None, runs_root: Path, compile_evidence: bool, cache_dir: Path | None, jobs: int = 2, progress: bool = True) -> list[tuple[KernelRef, HardwareFeatureReport]]:
    """@brief 并行给每个内核生成硬件特征报告。

    @param refs 要分析的内核引用。
    @param naive 朴素核测量表。
    @param calib 硬件标定；可以为 ``None``。
    @param runs_root 运行根目录。
    @param compile_evidence 为真时额外收编译期证据。
    @param cache_dir 编译缓存目录；可以为 ``None``。
    @param jobs 并行编译数。
    @param progress 为真时每 25 个打一行进度。
    @return ``(内核引用, 报告)`` 列表，顺序与入参一致。
    """
    # tasks must be built in the main thread (imports); compile evidence is the slow part and runs in a small pool
    for r in refs:
        if r.task:
            try:
                _get_task(r.task, r.dtype or "fp16", r.size)
            except Exception:  # noqa: BLE001
                pass
    ceil_cache: dict[tuple, Ceilings] = {}

    def one(r: KernelRef):
        key = (r.task, r.dtype or "fp16", r.size)
        if key not in ceil_cache:
            ceil_cache[key] = ceilings_for(r.task, r.dtype or "fp16", r.size, naive, calib, runs_root)
        tobj = None
        if r.task:
            try:
                tobj = _get_task(r.task, r.dtype or "fp16", r.size)
            except Exception:  # noqa: BLE001
                tobj = None
        rep = r.report
        if rep is None and r.kind == "naive":
            nv = naive.get((r.task, r.dtype or "fp16"))
            rep = {"status": nv.get("status", "ok"), "speedup": nv.get("speedup"), "candidate_us": nv.get("candidate_us"), "baseline_us": nv.get("baseline_us")} if nv else None
        meas = measured_from_report(rep, tobj, calib)
        return r, build_report(r.source, label=r.kid, task=r.task, dtype=r.dtype, size=r.size, measured=meas, ceilings=ceil_cache[key], compile_evidence=compile_evidence, cache_dir=cache_dir,
                               partial=r.partial, extra={"run": r.run, "kind": r.kind, "idx": r.idx, "trace": r.trace})

    out: list[tuple[KernelRef, HardwareFeatureReport]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        for i, res in enumerate(ex.map(one, refs), 1):
            out.append(res)
            if progress and (i % 25 == 0 or i == len(refs)):
                print(f"  analysed {i}/{len(refs)} kernels ({time.time() - t0:.0f} s)", flush=True)
    return out


def _med(xs: list[float]) -> float | None:
    """@brief 取中位数。

    @param xs 数值列表。
    @return 中位数；列表为空时返回 ``None``。
    """
    return statistics.median(xs) if xs else None


def winrate_table(items: list[tuple[KernelRef, HardwareFeatureReport]], feature_ids: Iterable[str]) -> dict:
    """@brief 按特征统计``带它``与``不带它``的胜率与正确率。

    Per feature: accepted-win rate and correctness rate of candidates with vs without the feature (pooled over runs that have a trace).

    @param items ``analyze_refs`` 返回的条目。
    @param feature_ids 要统计的特征 id。
    @return ``{特征: {"with": ..., "without": ...}}``。
    """
    cands = [(r, h) for r, h in items if r.kind in ("candidate", "best") and r.trace and r.run != "negative" and not (r.task or "").startswith("sem_")]
    tab = {}
    for fid in feature_ids:
        rows = {}
        for has in (True, False):
            sel = [(r, h) for r, h in cands if h.features[fid]["present"] == has]
            ok = [x for x in sel if (x[0].trace or {}).get("status") == "ok"]
            wins = [x for x in sel if (x[0].trace or {}).get("improved")]
            sps = [x[0].trace["speedup"] for x in ok if isinstance((x[0].trace or {}).get("speedup"), (int, float))]
            rows["with" if has else "without"] = {"n": len(sel), "correct": len(ok), "wins": len(wins), "win_rate": (len(wins) / len(sel)) if sel else None, "median_speedup_ok": _med(sps)}
        tab[fid] = rows
    return tab


def best_kernel_rows(items: list[tuple[KernelRef, HardwareFeatureReport]], runs: dict[str, dict]) -> list[dict]:
    """@brief 把每次运行的最优内核整理成报告行。

    @param items ``analyze_refs`` 返回的条目。
    @param runs 运行概况表。
    @return 每个最优内核一行：摘要、缺失特征、实测值与特征判定。
    """
    rows = []
    for r, h in items:
        if r.kind != "best":
            continue
        info = runs.get(r.run, {})
        rows.append({"run": r.run, "task": r.task, "label": r.label, "partial": info.get("partial", True), "confirmed_speedup": info.get("confirmed_speedup"), "best_speedup": info.get("best_speedup"),
                     "kid": r.kid, "summary": h.summary, "missing": h.missing, "perf": h.performance_test, "measured": {k: (h.measured or {}).get(k) for k in ("candidate_us", "speedup", "achieved_tflops", "compute_fraction", "achieved_gbps", "bandwidth_fraction", "max_total_threads", "static_threadgroup_bytes")},
                     "features": {fid: f["verdict"] for fid, f in h.features.items() if f["present"] or f["expectation"] in ("required", "recommended")}, "present": sorted(fid for fid, f in h.features.items() if f["present"])})
    return rows


def discovery_table(items: list[tuple[KernelRef, HardwareFeatureReport]], runs: dict[str, dict], feature_ids: Iterable[str], tasks: set[str]) -> dict:
    """@brief 统计特征在``首个正确候选``与``最优候选``里出现的次数。

    Per feature over runs of ``tasks``: how many runs had it in their first correct candidate, in their best, and gained / lost it during the search.

    @param items ``analyze_refs`` 返回的条目。
    @param runs 运行概况表。
    @param feature_ids 要统计的特征 id。
    @param tasks 限定在哪些任务上统计。
    @return ``{特征: {runs, first, best, gained, lost}}``。
    """
    by_run: dict[str, list[tuple[KernelRef, HardwareFeatureReport]]] = {}
    for r, h in items:
        if r.kind in ("candidate", "best") and base_task(r.task) in tasks and r.idx is not None:
            by_run.setdefault(r.run, []).append((r, h))
    out = {f: {"runs": 0, "first": 0, "best": 0, "gained": 0, "lost": 0} for f in feature_ids}
    for run, lst in by_run.items():
        lst.sort(key=lambda x: x[0].idx)
        ok = [x for x in lst if (x[0].trace or {}).get("status") == "ok"]
        best = next((x for x in lst if x[0].kind == "best"), None)
        if not ok or best is None:
            continue
        first = ok[0][1]
        for f in feature_ids:
            d = out[f]
            d["runs"] += 1
            a, b = first.features[f]["present"], best[1].features[f]["present"]
            d["first"] += a
            d["best"] += b
            d["gained"] += (b and not a)
            d["lost"] += (a and not b)
    return out


def width_txt(n: int | None) -> str:
    """@brief 把字节宽度写成表格里的文字。

    @param n 宽度字节数。
    @return 形如 ``16 B`` 的文本；不足 4 字节或为空时写 ``scalar``。
    """
    return f"{n} B" if n and n >= 4 else "scalar"


def cell(x: Any) -> str:
    """@brief 把值转成不会撑破 markdown 表格的单元文本。

    @param x 任意值。
    @return 转义过竖线、去掉换行的文本。
    """
    return str(x).replace("|", "\\|").replace("\n", " ")


def fmt(x: Any, nd: int = 2) -> str:
    """@brief 按固定小数位格式化数值。

    @param x 任意值。
    @param nd 浮点数保留的小数位。
    @return 格式化后的文本；``None`` 写成 ``-``。
    """
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def agreement_table(items: list[tuple[KernelRef, HardwareFeatureReport]]) -> list[dict]:
    """@brief 比较源码证据与编译期（AIR）证据，自检检测器是否一致。

    Source evidence vs compiled (AIR) evidence over every kernel that compiled: the detector's own consistency check.

    @param items ``analyze_refs`` 返回的条目。
    @return 每一项检查一行：参与数、一致数与前几条不一致明细。
    """
    checks: dict[str, dict] = {k: {"n": 0, "agree": 0, "disagree": []} for k in ("matrix unit present", "float accumulators", "device load width (max bytes)", "device store width (max bytes)",
                                                                                   "SIMD reductions present", "SIMD shuffles present", "threadgroup bytes (declared == AIR globals)", "accumulator fragment count (informational)")}

    def rec(name: str, kid: str, a: Any, b: Any):
        c = checks[name]
        c["n"] += 1
        if a == b:
            c["agree"] += 1
        elif len(c["disagree"]) < 4:
            c["disagree"].append(f"{kid}: source {a} vs AIR {b}")

    for r, h in items:
        c = h.compiled
        if not c or not c.get("ok"):
            continue
        s, cm = h.source, c["matrix"]
        rec("matrix unit present", r.kid, s["matrix"]["present"], cm["multiply_accumulate"] + cm["multiply"] > 0)
        if s["matrix"]["present"] or cm["multiply_accumulate"]:
            rec("float accumulators", r.kid, s["matrix"]["acc_types"] == ["float"], cm["acc_types"] == ["f32"])
            rec("accumulator fragment count (informational)", r.kid, s["matrix"]["acc_tiles"], max(c["fragment_tiles_f32"], cm.get("init_f32", 0)))
        rec("device load width (max bytes)", r.kid, s["memory"]["device_load"]["max_bytes"], max((int(b) for b in c["device_load_bytes"]), default=0))
        if not (s["memory"]["device_store"]["max_bytes"] == 0 and "device" in s["matrix"]["store_to"]):
            rec("device store width (max bytes)", r.kid, s["memory"]["device_store"]["max_bytes"], max((int(b) for b in c["device_store_bytes"]), default=0))
        rec("SIMD reductions present", r.kid, bool(s["simd"]["reduce"]), any(k.split("_", 1)[-1] in _SIMD_REDUCE or k.startswith("simd_prefix") for k in c["simd"]))
        rec("SIMD shuffles present", r.kid, bool(s["simd"]["shuffle"]), any("shuffle" in k or "broadcast" in k for k in c["simd"]))
        if s["threadgroup"]["declared_bytes"] or c["tg_global_bytes"]:
            rec("threadgroup bytes (declared == AIR globals)", r.kid, s["threadgroup"]["declared_bytes"], c["tg_global_bytes"])
    return [{"check": k, **v} for k, v in checks.items()]


def render_report(items: list[tuple[KernelRef, HardwareFeatureReport]], runs: dict[str, dict], out: Path, *, compiled: bool, generated: str) -> str:
    """@brief 拼出完整的 markdown 报告。

    @param items ``analyze_refs`` 返回的条目。
    @param runs 运行概况表。
    @param out 输出目录。
    @param compiled 为真时报告里标明含编译期证据。
    @param generated 生成时间戳文本。
    @return 报告文本。
    """
    L: list[str] = []
    best = best_kernel_rows(items, runs)
    by_kind = lambda k: [(r, h) for r, h in items if r.kind == k]  # noqa: E731
    L += ["# Hardware features used by the generated kernels", "", f"Generated {generated} by `experiments/hw_features.py` over {len(items)} kernel sources "
          f"({len(by_kind('candidate')) + len(by_kind('best'))} run candidates, {len(by_kind('reference'))} reference GEMMs, {len(by_kind('naive'))} naive kernels, {len(by_kind('synthetic'))} negative cases). "
          f"Compiled evidence: {'AIR from the Metal toolchain, no GPU dispatch' if compiled else 'not collected (--no-compile)'}. Definitions of the features, the paper mapping and the per-task expectations are in `docs/hardware_features.md`.", ""]
    L += ["**Partial runs.** Several runs were still running or had been interrupted when this was generated (marked `partial`); their \"best\" is the best candidate so far, not a confirmed result. "
          "Re-run the tool after they finish; the compile cache makes a second pass cheap.", ""]

    # --- validation ------------------------------------------------------------------------------------------------------------------
    L += ["## 1. Validation of the detector", ""]
    refs_ = by_kind("reference")
    L += ["### 1a. Hand-written reference GEMMs (`experiments/contract_sensitivity.py` generator; sources are generated, never run here)", "",
          "| contract | matrix unit (source / AIR) | acc fragments | acc type | staging | padded | vector ld/st | harness time us | TFLOP/s |", "| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: |"]
    for r, h in refs_:
        f = h.features
        m = h.measured or {}
        L.append(f"| {r.label} | {'yes' if f['matrix_unit']['source'] else 'NO'} / {'yes' if f['matrix_unit']['compiled'] else ('NO' if f['matrix_unit']['compiled'] is False else '-')} | {h.source['matrix']['acc_tiles']} | {','.join(h.source['matrix']['acc_types'])} | "
                 f"{'yes' if f['tg_staging']['present'] else 'no'} | {'yes' if f['tg_padding']['present'] else 'no'} | {width_txt(h.source['memory']['device_load']['max_bytes'])} / {width_txt(h.source['memory']['device_store']['max_bytes'])} | {fmt(m.get('candidate_us'), 1)} | {fmt(m.get('achieved_tflops'), 1)} |")
    n_ok = sum(1 for _, h in refs_ if h.features["matrix_unit"]["present"])
    L += ["", f"**{n_ok} of {len(refs_)} reference GEMMs are detected as matrix-unit users** (expected: all).", ""]
    small = [(h.measured or {}).get("candidate_us") for _, h in refs_ if (h.source["matrix"]["acc_tiles"] or 0) <= 16 and (h.measured or {}).get("candidate_us")]
    big = [(h.measured or {}).get("candidate_us") for _, h in refs_ if (h.source["matrix"]["acc_tiles"] or 0) >= 32 and (h.measured or {}).get("candidate_us")]
    if small and big:
        L += [f"Register blocking is detected correctly but is not monotone: the contracts with <= 16 accumulator fragments per SIMD group take {fmt(min(small), 0)}-{fmt(max(small), 0)} us (median {fmt(statistics.median(small), 0)}), "
              f"those with >= 32 fragments {fmt(min(big), 0)}-{fmt(max(big), 0)} us (median {fmt(statistics.median(big), 0)}): too many live accumulators spill (see `tcenv.cli profile`), so *present* does not mean *effective*.", ""]
    L += ["### 1b. Naive kernels (`tasks/*/naive.metal`) must not claim a matrix unit unless they use one", "", "| task | matrix unit | vector ld/st | simd ops | staging | unroll | note |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r, h in by_kind("naive"):
        L.append(f"| {r.label} | {'yes' if h.features['matrix_unit']['present'] else 'no'} | {width_txt(h.source['memory']['device_load']['max_bytes'])} / {width_txt(h.source['memory']['device_store']['max_bytes'])} | {','.join(sorted(k.replace('simd_', '') for k in h.source['simd']['ops'])) or '-'} | "
                 f"{'yes' if h.features['tg_staging']['present'] else 'no'} | {'yes' if h.features['unroll']['present'] else 'no'} | {h.compiled.get('error', '')[:60] if h.compiled and not h.compiled.get('ok') else ''} |")
    L += [""]
    L += ["### 1c. Negative cases (a feature named only in a comment or a string must not count)", "", "`single_pass` and `few_barriers` hold trivially for a one-line kernel and are not counted as claims.", "", "| case | features claimed | verdict |", "| --- | --- | --- |"]
    for r, h in by_kind("synthetic"):
        pres = [fid for fid, f in h.features.items() if f["present"] and fid not in ("single_pass", "few_barriers")]
        L.append(f"| {r.label} | {', '.join(pres) or 'none'} | {'PASS' if not pres else 'FAIL'} |")
    L += [""]
    agree = agreement_table(items)
    if any(a["n"] for a in agree):
        ncomp = sum(1 for _, h in items if h.compiled and h.compiled.get("ok"))
        nfail = sum(1 for _, h in items if h.compiled and not h.compiled.get("ok") and not h.compiled.get("skipped"))
        L += [f"### 1d. Source evidence vs compiled evidence (consistency of the two levels over the {ncomp} kernels that compiled; {nfail} failed to compile, which the harness also reported as `compile_error`)", "",
              "The fragment count row is informational: the AIR also counts score / scratch fragments and drops arrays the optimizer promoted to SSA values, so equality is not expected.", "",
              "| check | kernels | agree | disagree | first disagreements |", "| --- | ---: | ---: | ---: | --- |"]
        for a in agree:
            L.append(f"| {a['check']} | {a['n']} | {a['agree']} | {a['n'] - a['agree']} | {cell('; '.join(a['disagree'])) or '-'} |")
        L += [""]

    # --- best kernels -----------------------------------------------------------------------------------------------------------------
    L += ["## 2. Best kernel of every run: which hardware features it uses", "",
          "`best` = the run's winning candidate (confirmed result when the run finished, best-so-far when `partial`). One line per run; the summary string is what the dashboard would show. "
          "`missing` lists expected features (required first) that the source does not contain.", "",
          "| run | task | best | partial | confirmed speedup | summary | missing |", "| --- | --- | --- | --- | ---: | --- | --- |"]
    for b in sorted(best, key=lambda b: (b["task"] or "", b["run"])):
        L.append(f"| {b['run']} | {b['task']} | {b['label']} | {'yes' if b['partial'] else 'no'} | {fmt(b.get('confirmed_speedup'), 3)} | {cell(b['summary'])} | {', '.join(x.replace('_', ' ') for x in b['missing']) or '-'} |")
    L += [""]

    # --- effectiveness ----------------------------------------------------------------------------------------------------------------
    L += ["## 3. Measured vs the no-feature ceiling (best kernels)", "",
          "Ceiling = what the same task reaches with the *naive scalar kernel* under the same protocol (fallback: 0.37 x the matrix peak for compute-bound tasks, 0.87 x the bandwidth peak for memory-bound ones). "
          "`above` = at least 1.10x the ceiling (compute) / 1.02x (memory). Primary feature of a family: matrix unit (compute), vector accesses and SIMD reductions (memory); every other feature is judged with the same test and is *joint* (ablation needed).", "",
          "| run | task | best | achieved | ceiling (basis) | ratio | result | peak fraction | pipeline max threads | TG bytes |", "| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |"]
    for b in sorted([x for x in best if not (x["task"] or "").startswith("sem_")], key=lambda b: (b["task"] or "", b["run"])):
        p = b["perf"]
        frac = b["measured"].get("compute_fraction") if p.get("kind") == "compute" else b["measured"].get("bandwidth_fraction")
        L.append(f"| {b['run']} | {b['task']} | {b['label']} | {fmt(p.get('achieved'))} {p.get('unit', '')} | {fmt(p.get('ceiling'))} ({'naive kernel' if 'naive' in (p.get('ceiling_basis') or '') else 'fallback'}) | {fmt(p.get('ratio'))} | {p['result']} | {fmt(frac)} | {b['measured'].get('max_total_threads') or '-'} | {b['measured'].get('static_threadgroup_bytes') or '-'} |")
    L += [""]

    # --- per-feature verdict matrix ------------------------------------------------------------------------------------------------------
    L += ["## 4. Per-feature verdicts of the best kernels", "", "`+` used and effective, `~` present but not effective, `-` expected but missing, `.` not applicable / not expected, `?` present but not measured or effect unknown.", ""]
    cols = [f for f in FEATURES if f not in ("async_copy", "tensor_api")]
    L += ["| run | task | " + " | ".join(c.replace("_", " ") for c in cols) + " |", "| --- | --- | " + " | ".join("---" for _ in cols) + " |"]
    sym = {"used and effective": "+", "present but not effective": "~", "expected but missing": "-", "not applicable": ".", "present, not expected": "+*", "present, not measured": "?", "present, effect unknown": "?"}
    hmap = {r.kid: h for r, h in items}
    for b in sorted([x for x in best if not (x["task"] or "").startswith("sem_")], key=lambda b: (b["task"] or "", b["run"])):
        h = hmap[b["kid"]]
        L.append(f"| {b['run']} | {b['task']} | " + " | ".join(sym.get(h.features[c]["verdict"], "?") for c in cols) + " |")
    L += ["", "`+*` = present although the task does not list the feature as expected (counts as effective when the kernel beats its ceiling).", ""]

    # --- paper feature scoreboard -----------------------------------------------------------------------------------------------------------
    comp_tasks = {"gemm", "gemm_silu", "conv2d", "flash_attention", "bitdelta_gemm", "mamba2_chunk_scan"}
    cb = sorted([b for b in best if base_task(b["task"]) in comp_tasks], key=lambda b: (base_task(b["task"]), b["run"]))
    if cb:
        short = lambda b: re.sub(r"^repeat_[\w-]+/", "rep/", re.sub(r"^pilot_suite_mini_old_protocol/", "pilot/", re.sub(r"^suite_full/", "suite/", re.sub(r"^p1/", "", b["run"]))))  # noqa: E731
        L += ["## 5. Which of the paper's hardware features appear in the best kernels (compute-bound runs)", "",
              "Paper features P1-P16 as listed in `docs/hardware_features.md` table 1, through their Apple analogue (detector feature in the last column). `X` = the run's best kernel has it (source level, and AIR where the compiler can confirm it), `.` = absent. "
              "P17 (input reuse across overlapping windows) is not detectable from text and is not scored; P3 has no analogue on M4 Max (the Metal 4 tensor API is refused by the harness's include screen).", "",
              "| paper feature | " + " | ".join(f"{short(b)}{'*' if b['partial'] else ''}" for b in cb) + " | detector feature(s) |", "| --- | " + " | ".join("---" for _ in cb) + " | --- |"]
        for pid, (title, fids) in PAPER_TO_DETECTOR.items():
            cells = []
            for b in cb:
                h = hmap[b["kid"]]
                cells.append("X" if any(h.features[f]["present"] for f in fids) else ".")
            L.append(f"| {pid} {title} | " + " | ".join(cells) + f" | {', '.join(fids)} |")
        L += ["", "Columns: " + "; ".join(f"`{short(b)}` = {b['run']} ({b['task']}, best {b['label']})" for b in cb) + ". `*` = partial run.", ""]

    # --- headline findings -------------------------------------------------------------------------------------------------------------
    L += ["## 5b. Headline findings", ""]
    L += _headline(best, hmap, items)

    # --- win rates ------------------------------------------------------------------------------------------------------------------------
    compute_tasks = {"gemm", "gemm_silu", "conv2d", "flash_attention", "bitdelta_gemm", "mamba2_chunk_scan"}
    memory_tasks = set(TASK_FAMILY) - compute_tasks
    fams = (("compute-bound tasks (gemm, gemm_silu, conv2d, flash_attention, bitdelta_gemm, mamba2_chunk_scan)", compute_tasks,
             ["matrix_unit", "register_blocking", "tg_staging", "tg_padding", "double_buffering", "register_prefetch", "pipelining", "simd_shuffle", "operand_in_registers", "online_softmax", "epilogue_on_fragments", "unroll", "fast_math"]),
            ("memory-bound tasks (sigmoid, relu, silu, swiglu, lion, rope, rmsnorm, softmax, reduction_sum, gemv)", memory_tasks,
             ["vector_access", "simd_reduce", "simd_shuffle", "single_pass", "few_barriers", "fast_math", "unroll"]))
    L += ["## 6. Does the LLM find the features, and do accepted wins contain them?", "",
          "`win` = the candidate became the new best of its run when it was evaluated; `correct` = it passed every gate (status ok). Counts are over every candidate of runs that have a trace on disk (semantic tasks excluded; partial runs included). "
          "`~` marks n < 10 (anecdotal): fewer than 10 accepted wins in the first column, fewer than 10 candidates in a win-rate column. A feature that (almost) every candidate has cannot discriminate wins from non-wins: read the `without` n before any rate.", ""]
    def wr_rows(sub, keys):
        t = winrate_table(sub, keys)
        rows = []
        pct = lambda k, n: f"{k} / {n} ({100 * k / n:.0f}%)" if n else "-"  # noqa: E731
        for fid, d in t.items():
            w, wo = d["with"], d["without"]
            n_w, n_o = w["wins"] + wo["wins"], (w["n"] - w["wins"]) + (wo["n"] - wo["wins"])
            tag = "~" if n_w < 10 else ""
            rate = lambda x: (fmt(100 * x["win_rate"], 0) + "%" + ("~" if x["n"] < 10 else "")) if x["win_rate"] is not None else "-"  # noqa: E731
            rows.append(f"| {fid} | {pct(w['wins'], n_w)}{tag} | {pct(w['n'] - w['wins'], n_o)} | {w['n']} / {w['correct']} / {w['wins']} | {rate(w)} | {wo['n']} / {wo['correct']} / {wo['wins']} | {rate(wo)} | {fmt(w['median_speedup_ok'])} / {fmt(wo['median_speedup_ok'])} |")
        return rows

    head = ["| feature | accepted wins that have it | rejected / non-winning candidates that have it | with: n / correct / wins | win rate with | without: n / correct / wins | win rate without | median speedup (ok) with / without |", "| --- | --- | --- | --- | ---: | --- | ---: | --- |"]
    for fam, tasks, keys in fams:
        L += [f"### {fam}", ""]
        if tasks is compute_tasks:
            L += ["Compute-bound tasks differ by a factor of 2 in speedup (GEMM ~0.95, FlashAttention ~0.4-0.7), so win rates and median speedups are given **per task**; pooling would credit a feature with its task's speed.", ""]
            for tk in ("gemm", "flash_attention", "conv2d", "mamba2_chunk_scan"):
                sub = [(r, h) for r, h in items if base_task(r.task) == tk]
                nn = sum(1 for r, h in sub if r.kind in ("candidate", "best") and r.trace)
                if nn:
                    L += [f"**{tk}** ({nn} candidates with a trace)", ""] + head + wr_rows(sub, keys) + [""]
        else:
            sub = [(r, h) for r, h in items if base_task(r.task) in tasks]
            L += ["All memory-bound tasks sit at 0.95-1.05x of the baseline, so pooling them does not mix scales.", ""] + head + wr_rows(sub, keys) + [""]
        L += [f"**First correct candidate vs best candidate, per run** ({fam.split(' (')[0]})", "", "| feature | runs | in the first correct candidate | in the best | gained during the search | lost |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for fid, d in discovery_table(items, runs, keys, tasks).items():
            L.append(f"| {fid} | {d['runs']}{'~' if d['runs'] < 10 else ''} | {d['first']} | {d['best']} | {d['gained']} | {d['lost']} |")
        L += [""]

    # --- limits ---------------------------------------------------------------------------------------------------------------------------------
    L += ["## 7. Limits (read before quoting a number)", "",
          "* **Presence is verified, causation is not.** A kernel that has the feature and is fast does not prove the feature made it fast; that needs an ablation (see `ablation_plan` in the module and `docs/hardware_features.md`). Non-primary features are therefore labelled joint.",
          "* **Source heuristics.** Double buffering, register prefetch, padding, online softmax, single pass and bit packing are pattern detectors (documented in the module); they are conservative but not proofs. `tg_padding` does not see a stride hidden in a runtime value.",
          "* **AIR is not the machine code.** The compiled level confirms the compiler kept the matrix intrinsics, vector widths, barriers and threadgroup allocations; static call-site counts do not equal executed instructions, and register allocation / spills happen after AIR (see `tcenv.cli profile` for spills).",
          "* **No GPU dispatch** was performed; pipeline numbers (`max threads`, `TG bytes`) are the ones the harness stored in each candidate's report.",
          "* **Snapshot.** Runs that were still active are partial; counts of wins use the acceptance flags the orchestrator recorded (`trace.improved`), reconstructed from the reports when no trace exists.", ""]
    text = "\n".join(L)
    return text


def _headline(best: list[dict], hmap: dict, items) -> list[str]:
    """@brief 按任务族生成报告里的主要发现一段。

    @param best ``best_kernel_rows`` 返回的行。
    @param hmap ``{kid: 报告}`` 的索引。
    @param items ``analyze_refs`` 返回的条目。
    @return markdown 行列表。
    """
    L = []

    def rows(prefixes: tuple[str, ...]):
        return [b for b in best if any(b["run"].split("/")[-1].startswith(p) or b["run"] == p for p in prefixes)]

    def line(b):
        h = hmap[b["kid"]]
        f = h.features
        mm = h.source["matrix"]
        p = b["perf"]
        eff = f"{p.get('achieved')} {p.get('unit')} vs ceiling {p.get('ceiling')} ({p.get('ratio')}x, {p['result']})" if p.get("achieved") else p["result"]
        return (f"* `{b['run']}` ({b['task']}, best `{b['label']}`{', partial' if b['partial'] else ''}): matrix unit **{'yes' if f['matrix_unit']['present'] else 'NO'}**"
                f"{' (' + str(mm['acc_tiles']) + ' acc fragments, ' + '/'.join(mm['acc_types']) + ' acc)' if f['matrix_unit']['present'] else ''}; {eff}; "
                f"online softmax {'yes' if f['online_softmax']['present'] else 'no'}; double buffer {h.source['threadgroup']['double_buffer']}; register prefetch {'yes' if f['register_prefetch']['present'] else 'no'}; "
                f"vector {h.source['memory']['device_load']['max_bytes'] or 'scalar'}/{h.source['memory']['device_store']['max_bytes'] or 'scalar'} B; missing: {', '.join(b['missing']) or 'nothing'}.")

    for title, pre in (("GEMM", ("gemm_paper", "gemm_fp16")), ("conv2d", ("conv2d",)), ("FlashAttention", ("fa_", "flash_attention")), ("Mamba-2", ("mamba2",))):
        rs = [b for b in best if any(b["run"].split("/")[-1].startswith(p) for p in pre)]
        if rs:
            L += [f"**{title}**", ""] + [line(b) for b in sorted(rs, key=lambda b: b["run"])] + [""]
    el = [b for b in best if TASK_FAMILY.get(base_task(b["task"]) or "") in ("elementwise",)]
    if el:
        L += ["**Saturated elementwise tasks (bandwidth-bound)**", "", "| run | task | vector load / store bytes (source) | AIR widest ld / st | single pass | achieved GB/s | peak fraction | vs naive |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
        for b in sorted(el, key=lambda b: b["task"]):
            h = hmap[b["kid"]]
            c = h.compiled if h.compiled and h.compiled.get("ok") else None
            aw = (max((int(k) for k in c["device_load_bytes"]), default=0), max((int(k) for k in c["device_store_bytes"]), default=0)) if c else ("-", "-")
            L.append(f"| {b['run']} | {b['task']} | {h.source['memory']['device_load']['max_bytes'] or 'scalar'} / {h.source['memory']['device_store']['max_bytes'] or 'scalar'} | {aw[0]} / {aw[1]} | {'yes' if h.features['single_pass']['present'] else 'no'} | {fmt(b['measured'].get('achieved_gbps'), 0)} | {fmt(b['measured'].get('bandwidth_fraction'))} | {fmt(b['perf'].get('ratio'), 3)}x |")
        L += [""]
    return L


# ======================================================================================================================================
# 10. CLI
# ======================================================================================================================================


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m experiments.hw_features`` 的入口，写 ``REPORT.md``、``data.json`` 与 ``kernels.jsonl``。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码，正常结束时为 0。
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="run directories or roots that contain runs (e.g. runs/apple-m4-max/p1 runs/apple-m4-max/suite_full)")
    ap.add_argument("--out", default="runs/hw_features", help="output directory (REPORT.md, data.json, kernels.jsonl, compile cache)")
    ap.add_argument("--runs-root", default=str(REPO / "runs"), help="root used for relative run names and naive_*.json")
    ap.add_argument("--no-compile", action="store_true", help="source evidence only (skip the AIR compile)")
    ap.add_argument("--no-validation", action="store_true", help="skip the reference GEMMs / naive kernels / negative cases")
    ap.add_argument("--jobs", type=int, default=2, help="parallel compiles (each runs under nice -n 15)")
    ap.add_argument("--source", help="analyse one .metal file (candidate source without the contract header) and print its report")
    ap.add_argument("--task", help="task of --source (gives the contract header and constants), e.g. gemm or flash_attention@64x32t256")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--json", action="store_true", help="with --source: print the full JSON report")
    ap.add_argument("--ablate", action="store_true", help="print the (optional, not executed) ablation plan and exit")
    a = ap.parse_args(argv)
    if a.ablate:
        print(ablation_plan)
        return 0
    runs_root = Path(a.runs_root)
    try:
        from tcenv.target import TargetError, read_calibration

        calib, _path, _source = read_calibration()
    except (OSError, json.JSONDecodeError, TargetError):
        calib = None
    out = Path(a.out)
    if a.source:
        rep = build_report(Path(a.source).read_text(), label=a.source, task=a.task, dtype=a.dtype, compile_evidence=not a.no_compile, cache_dir=out / ".cache")
        print(rep.to_json(indent=1) if a.json else rep.summary)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".cache"
    nested = runs_root / "apple-m4-max"

    def _root(name: str) -> Path:
        moved = nested / name
        return moved if moved.is_dir() else runs_root / name

    roots = [Path(p) for p in (a.paths or [str(_root(d)) for d in ("p1", "suite_full", "semantic_main", "pilot_suite_mini_old_protocol", "counter_teaching")] + [str(p) for p in sorted(nested.glob("repeat_*"))] + [str(p) for p in sorted(nested.glob("2026*_fp16_*"))])]
    naive = load_naive_measurements(runs_root)
    runs: dict[str, dict] = {}
    refs: list[KernelRef] = []
    for d in discover_runs(roots):
        info, ks = load_run(d, runs_root)
        runs[info["run"]] = info
        refs += ks
    if not a.no_validation:
        refs += validation_sets(runs_root)
    print(f"{len(runs)} runs, {len(refs)} kernel sources; compile={'off' if a.no_compile else 'on'}", flush=True)
    t0 = time.time()
    items = analyze_refs(refs, naive=naive, calib=calib, runs_root=runs_root, compile_evidence=not a.no_compile, cache_dir=cache, jobs=a.jobs)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with (out / "kernels.jsonl").open("w") as f:
        for r, h in items:
            f.write(json.dumps({"kid": r.kid, "run": r.run, "kind": r.kind, "label": r.label, "task": r.task, "partial": r.partial, "summary": h.summary, "missing": h.missing, "report": h.to_dict()}, default=lambda o: sorted(o) if isinstance(o, set) else str(o)) + "\n")
    best = best_kernel_rows(items, runs)
    wt = winrate_table(items, [f for f in FEATURES if f not in ("async_copy", "tensor_api")])
    (out / "data.json").write_text(json.dumps({"generated": stamp, "runs": runs, "best_kernels": best, "winrate": wt, "features": FEATURES, "expected": EXPECTED, "compiled": not a.no_compile,
                                               "compile_seconds_total": round(time.time() - t0, 1)}, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    text = render_report(items, runs, out, compiled=not a.no_compile, generated=stamp)
    (out / "REPORT.md").write_text(text + "\n")
    print(f"wrote {out}/REPORT.md, data.json, kernels.jsonl  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
