"""Hardware-feature hints for the planner and the patch authors (``search.hw_feedback``, off by default).

``hw_feedback(task, source, report)`` turns the static hardware-feature analysis of ``taic.hwfeatures`` (source, compiled AIR, the candidate's measured
roofline) into a SHORT text block that the orchestrator appends to the evaluation feedback of a correct seed, in the place where the profiler evidence
goes. The block states which Apple-GPU features the kernel uses, which features that are expected for the task class are missing (one concrete hint
each) and says plainly that the missing items are hints, not requirements. Kernels that already sit at about 90 % of the roofline get no list of
missing features: hardware features cannot help beyond the roofline.

The feedback must never break or slow the search: every failure, and every overrun of the time limit, returns ``None`` and logs a warning.

    python -m taic.hwfeedback --task flash_attention --source best.metal --report candidates/012_r3.p0/report.json     # print the block of one candidate
"""

# 本模块只在 search.hw_feedback 打开时被编排器用到；关着时不会被导入，提示词和反馈与以前逐字节相同。
# 块的内容全部来自 taic/hwfeatures.py 的静态分析：源码层、AIR 层（缓存的 xcrun metal 编译，约 40 ms，不碰 GPU）和这颗候选自己的 roofline。

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

from . import hwfeatures as hw

LOG = logging.getLogger("taic.hwfeedback")

SATURATED_BELOW = 1.0 / 0.90  # time over the roofline lower bound: at or below this the kernel is at >= 90 % of the roofline
MAX_MISSING = 3  # items listed per block (required ones first)
TIME_LIMIT_S = 20.0  # the whole analysis, compile included; the compile itself is limited to COMPILE_TIMEOUT_S
COMPILE_TIMEOUT_S = 10.0
SPILL_FRAGMENTS = 32  # reference GEMMs with this many accumulator fragments per SIMD group were 12 to 26 times slower than with 16 or fewer
CACHE_DIR = Path(os.environ.get("TAIC_HW_CACHE", Path.home() / ".cache" / "ai_lowerling" / "hwfeatures"))

# one concrete Apple-GPU hint per feature (all <= ~30 words). {x} is the roofline gap, {frag} the accumulator fragments.
HINTS: dict[str, str] = {
    "matrix_unit": "no simdgroup_matrix: the matrix product runs on scalar FMA ({gap}); 8x8 simdgroup_multiply_accumulate (half in, float accumulate) is the matrix unit",
    "float_accumulate": "matrix accumulators are not float: use simdgroup_float8x8 accumulators (half accumulation loses accuracy on long K)",
    "register_blocking": "{frag} accumulator fragment(s) per SIMD group: a 2x2 to 4x4 grid (4 to 16) lets each loaded operand tile feed several MMAs (more than ~24 risks spills)",
    "tg_staging": "no threadgroup staging: stage each operand tile once per threadgroup with cooperative 16-byte loads and a barrier, instead of re-reading device memory per SIMD group",
    "tg_padding": "threadgroup tile rows are not padded: a row stride of tile width + 8 halves (or an odd word stride) avoids bank conflicts on simdgroup_load",
    "pipelining": "the next tile's loads are not overlapped with the MMAs (Metal has no async copy): issue them into registers before the MMAs and store to threadgroup memory after, or double-buffer",
    "epilogue_on_fragments": "apply the epilogue on the accumulator registers (thread_elements()) before the single store, not after a round trip through memory",
    "online_softmax": "no running max / sum: keep m and l per row and rescale the accumulator by exp2(m_old - m_new) per key tile, so the score matrix is never materialized",
    "simd_reduce": "reduce inside a SIMD group with simd_sum / simd_max first (no barrier), then exchange one value per SIMD group through threadgroup memory",
    "simd_shuffle": "row max / sum of an MMA fragment need simd_shuffle_xor (xor 1 and xor 8) or a threadgroup round trip",
    "operand_in_registers": "fragments come only from simdgroup_load; thread_elements() can decode, convert or pack values straight into a fragment, skipping a threadgroup round trip",
    "bit_packing": "packed weights are not decoded with shifts and masks on loaded words: decode the bits in registers (e.g. (w >> j) & 1) instead of per-element loads",
    "vector_access": "device accesses are narrower than 8 bytes: 16-byte uint4 / half4 accesses on consecutive lanes cut instruction count (scalar loads already reach about 85-90% of the bandwidth, so this matters only if the roofline shows a gap)",
    "single_pass": "an input is read in more than one pass over global memory: keep the data in registers or threadgroup memory between phases, or use an online recurrence",
    "few_barriers": "{barriers} threadgroup barriers per iteration: fewer synchronization points (one per tile with double buffering) shorten the critical path",
}

# fast_math is detected and reported, but never hinted: kernels often avoid the fast exp on purpose (accuracy of the softmax weights), so "use fast::exp" would be a bad hint
NOT_HINTED = {"fast_math"}
NEEDS_MATRIX_UNIT = {"float_accumulate", "register_blocking", "operand_in_registers", "epilogue_on_fragments", "simd_shuffle"}
_CACHE: dict[tuple, str | None] = {}
_CACHE_MAX = 256


def _kib(n: int) -> str:
    return f"{n / 1024:.1f} KiB"


def _uses(hr: "hw.HardwareFeatureReport") -> str:
    s, f = hr.source, hr.features
    m, mem, tg = s["matrix"], s["memory"], s["threadgroup"]
    bits: list[str] = []
    if f["matrix_unit"]["present"]:
        acc = "/".join(m["acc_types"]) or "?"
        opr = "/".join(m["operand_types"]) or "?"
        tiles = m["acc_tiles"]
        bits.append(f"simdgroup_matrix 8x8 ({opr} in, {acc} acc, {tiles if tiles is not None else '?'} fragments per SIMD group)")
    if f["tg_staging"]["present"]:
        bits.append(f"threadgroup staging {_kib(tg['declared_bytes'])}" + (", padded" if f["tg_padding"]["present"] else ""))
    if tg["double_buffer"] == "explicit":
        bits.append("multi-stage buffers")
    if tg["register_prefetch"]:
        bits.append("register prefetch")
    dl, ds = mem["device_load"]["max_bytes"], mem["device_store"]["max_bytes"]
    ld = f"{dl} B" if dl >= 4 else ("simdgroup_load" if "device" in m["loads_from"] else "scalar")
    st = f"{ds} B" if ds >= 4 else ("simdgroup_store" if "device" in m["store_to"] else "scalar")
    bits.append(f"device {ld} loads / {st} stores")
    simd = s["simd"]["ops"]
    if simd:
        bits.append(", ".join(sorted(simd)))
    for fid, text in (("online_softmax", "online softmax"), ("operand_in_registers", "operand fragments in registers"), ("epilogue_on_fragments", "epilogue on registers"), ("bit_packing", "bit packing"), ("unroll", "unroll")):
        if f[fid]["present"]:
            bits.append(text)
    return "; ".join(bits)


def _performance(hr: "hw.HardwareFeatureReport", measured: dict, roof: dict) -> tuple[str, float | None]:
    """(the measured-performance sentence, fraction of the roofline or None when the report has no roofline)."""
    gap = roof.get("time_over_lower_bound")
    frac = (1.0 / gap) if gap else None
    bound = roof.get("bound") or "?"
    if not roof:
        return "", None
    if bound == "compute" and roof.get("achieved_tflops") is not None:
        pf = roof.get("compute_fraction")
        txt = f"Measured: {roof['achieved_tflops']:.1f} TFLOP/s = {100 * pf:.0f}% of the matrix peak ({roof.get('peak_tflops_measured', 0):.1f}); scalar FMA reaches about {100 * hw.SCALAR_OVER_MATRIX:.0f}%"
        if hr.features["matrix_unit"]["present"]:
            txt += "." if (pf or 0) >= 2.0 * hw.SCALAR_OVER_MATRIX * 0.6 else "; the matrix unit is there but not the limiter (look at loads, barriers, spills, occupancy)."
        else:
            txt += "."
    else:
        txt = f"Measured: {roof.get('achieved_gbps', 0):.0f} GB/s = {100 * (roof.get('bandwidth_fraction') or 0):.0f}% of the bandwidth peak."
    if gap:
        txt += f" {max(gap, 1.0):.2f}x the {bound} roofline lower bound."
    return txt, frac


def _warnings(hr: "hw.HardwareFeatureReport", measured: dict) -> list[str]:
    out = []
    tiles = hr.source["matrix"]["acc_tiles"] or 0
    if tiles >= SPILL_FRAGMENTS:
        out.append(f"Warning: {tiles} accumulator fragments per SIMD group; reference GEMMs with 32 or more fragments ran 12 to 26 times slower (register spills). Do not add fragments; fewer may be faster.")
    stages = measured.get("stages")
    if isinstance(stages, list) and stages:
        for stage in stages:
            mt, lt = stage.get("max_total_threads"), stage.get("launch_threads")
            if mt and lt and mt <= lt and mt < 1024:
                out.append(
                    f"Warning: stage {stage.get('kernel')} reports maxTotalThreadsPerThreadgroup={mt}, "
                    f"equal to that stage's launch size {lt}: the pipeline is capped at its launch size. "
                    "This alone does not prove register pressure; verify register/spill data before changing live state."
                )
    else:
        mt, lt = measured.get("max_total_threads"), measured.get("launch_threads")
        if mt and lt and mt <= lt and mt < 1024:
            out.append(f"Warning: the pipeline reports maxTotalThreadsPerThreadgroup={mt}, equal to the launch size {lt}: the pipeline is capped at its launch size. This alone does not prove register pressure; verify register/spill data before changing live state.")
    return out


def _hints(hr: "hw.HardwareFeatureReport", measured: dict, roof: dict) -> list[str]:
    s = hr.source
    gap = roof.get("time_over_lower_bound")
    fmt = {"gap": f"the roofline shows {gap:.1f}x the lower bound" if gap else "no roofline in the report", "frag": s["matrix"]["acc_tiles"] or 0, "barriers": s["sync"]["main_loop_barrier_sites"] or s["sync"]["threadgroup_barrier_sites"]}
    out = []
    # without a matrix unit the items that only make sense around one (float accumulators, fragments, operand fragments, ...) are noise: say the one thing
    missing = [f for f in hr.missing if f not in NOT_HINTED and not ("matrix_unit" in hr.missing and f in NEEDS_MATRIX_UNIT)]
    for fid in missing[:MAX_MISSING]:
        h = HINTS.get(fid)
        if h:
            out.append(f"- {fid.replace('_', ' ')}: " + h.format(**fmt))
    rest = len(missing) - MAX_MISSING
    if rest > 0:
        out.append(f"- (+{rest} lower-priority item(s) omitted)")
    return out


def format_block(hr: "hw.HardwareFeatureReport", measured: dict, roof: dict, *, bound_beaten: str | None = None) -> str:
    """The text block for one analysed kernel (see the module docstring); at most 12 lines.

    ``bound_beaten`` replaces the saturated/missing section when a baseline is already faster than the roofline bound:
    the kernel is not at a ceiling the hardware cannot pass, the bound is the wrong target.
    """
    perf, frac = _performance(hr, measured, roof)
    lines = ["Hardware-feature analysis of this kernel (static analysis of the source and the compiled AIR, not a measurement):", "Uses: " + _uses(hr) + "."]
    if perf:
        lines.append(perf)
    saturated = frac is not None and (1.0 / frac) <= SATURATED_BELOW
    if bound_beaten:
        lines.append(bound_beaten)
    elif saturated:
        lines.append(f"This kernel is at {min(100.0, 100 * frac):.0f}% of its {roof.get('bound', '')} roofline: hardware features cannot help beyond the roofline, so none is listed as missing.")
    else:
        hints = _hints(hr, measured, roof)
        if hints:
            lines.append("Expected for this kind of kernel but not found (hints):")
            lines += hints
        elif frac is not None:
            lines.append("No expected feature is missing: the gap to the roofline is not explained by anything this analysis can see.")
    lines += _warnings(hr, measured)
    lines.append("Missing items are hints from static analysis, not requirements: the measured time decides.")
    return "\n".join(lines)


def _run_with_timeout(fn: Callable[[], Any], seconds: float) -> Any:
    """Run ``fn`` in a daemon thread; raise TimeoutError when it does not finish in time (the thread is abandoned, it cannot block the process exit)."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the caller
            box["error"] = e

    t = threading.Thread(target=target, name="hw-feedback", daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"hardware-feature analysis took more than {seconds:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _compute(task: Any, source: str, report: Any) -> str | None:
    task_obj = task if hasattr(task, "contract") else hw._get_task(str(task), getattr(report, "dtype", "fp16") or "fp16", getattr(report, "size", "paper") or "paper")
    name = task_obj.name
    if hw.base_task(name) not in hw.EXPECTED:
        return None  # no expectations for this task (semantic tests, unknown names): nothing useful to say
    rep = report.to_json() if hasattr(report, "to_json") else dict(report)
    measured = hw.measured_from_report(rep)
    roof = rep.get("roofline") or {}
    ceil = hw.Ceilings(peak_tflops=roof.get("peak_tflops_measured"), peak_gbps=roof.get("peak_gbps_measured"))
    hr = hw.build_report(source, label="candidate", task=name, dtype=task_obj.dtype, size=task_obj.size, measured=measured, ceilings=ceil, compile_evidence=True, cache_dir=CACHE_DIR,
                         task_obj=task_obj, compile_timeout=COMPILE_TIMEOUT_S)
    from .feedback_extra import bound_beaten_line

    return format_block(hr, measured or {}, roof, bound_beaten=bound_beaten_line(rep))


def hw_feedback(task: Any, source: str, report: Any, *, log: Callable[[str], None] | None = None, backend_id: str = "metal") -> str | None:
    """The hardware-feature block for a correct candidate, or None.

    ``task`` is a ``tcenv.task.Task`` (or a task name), ``source`` the candidate's kernel text, ``report`` its ``EvalReport`` (or the dict of ``report.json``).
    Returns None for candidates that are not correct (their feedback stays as it is), for tasks without expectations and on ANY failure or timeout
    (logged as a warning): the feedback must never break or slow the search loop."""
    try:
        from tcenv.backends.registry import load_backend

        if not load_backend(backend_id).capabilities.supports_hw_features:
            return None
        status = getattr(report, "status", None) if not isinstance(report, dict) else report.get("status")
        if status != "ok":
            return None
        sp = getattr(report, "speedup", None) if not isinstance(report, dict) else report.get("speedup")
        key = (backend_id, getattr(task, "name", str(task)), hashlib.sha1(source.encode()).hexdigest(), round(sp, 4) if isinstance(sp, (int, float)) else None)
        if key in _CACHE:
            return _CACHE[key]
        text = _run_with_timeout(lambda: _compute(task, source, report), TIME_LIMIT_S)
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = text
        return text
    except BaseException as e:  # noqa: BLE001 - includes the timeout; the search goes on without the block
        msg = f"hw_feedback skipped: {type(e).__name__}: {str(e)[:200]}"
        if log is not None:
            try:
                log("  " + msg)
            except Exception:  # noqa: BLE001
                pass
        LOG.warning(msg)
        return None


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    sys.path.insert(0, str(hw.REPO))
    from tcenv.evaluate import EvalReport

    ap = argparse.ArgumentParser(description="print the hardware-feature block of one candidate")
    ap.add_argument("--task", required=True)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--size", default="paper")
    ap.add_argument("--source", required=True, help="candidate kernel source (.metal, without the contract header)")
    ap.add_argument("--report", required=True, help="the candidate's report.json from a run directory")
    a = ap.parse_args(argv)
    task = hw._get_task(a.task, a.dtype, a.size)
    text = hw_feedback(task, Path(a.source).read_text(), EvalReport.from_json(json.loads(Path(a.report).read_text())))
    print(text if text is not None else "(no block: the candidate is not correct, the task has no expectations, or the analysis failed; see the warning above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
