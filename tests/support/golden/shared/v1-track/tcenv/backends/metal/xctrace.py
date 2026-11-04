"""Optional deeper profiler feedback: Instruments GPU counters captured with ``xctrace``.

What ``xcrun xctrace record --template "Metal System Trace"`` reliably yields on this machine (measured):

  * per-dispatch GPU execution intervals (``metal-gpu-intervals``),
  * GPU Active/Idle state (``metal-gpu-state-intervals``): the busy fraction exposes launch gaps,
  * compiler register-spill events with spilled bytes (``graphics-compiler-spill-events``),
  * the compiled shader's code size (``metal-shader-profiler-shader-list``, pc range) as an instruction-count proxy,
  * the thermal state (a throttled GPU makes every measurement unreliable).

Hardware *counters* (ALU/occupancy/limiters) are not available through the command line with the default template: the counter
profile is selected in the Instruments GUI, and recording with the ``Metal GPU Counters`` instrument alone fails with "Selected
counter profile is not supported on target device". If you save a template from Instruments with a counter profile selected, point
``TCENV_XCTRACE_TEMPLATE`` at it and the counters (GPU-wide, restricted to the window of our compute work) are summarized too. Only kernels that already passed the full evaluation are profiled, and the profiled
process is supervised with the same heartbeat/kill protocol as every other worker.

This is much slower than the normal evaluation (tens of seconds) and is therefore only used on demand
(``tcenv profile``) and for the winning kernel of a run, not inside the search loop.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = os.environ.get("TCENV_XCTRACE_TEMPLATE", "Metal System Trace")
BATCH = 10  # launches per encoder in tcenv.xcworker; GPU intervals are per encoder


class TraceError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- record


def record(
    task: str, dtype: str, size: str, kernel_source: str, workdir: Path, *, duration_s: float | None = None, record_limit_s: float = 40.0, launch_deadline_s: float = 2.0
) -> Path:
    """@brief 录一段候选核反复启动的 Metal System Trace。

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模名。
    @param kernel_source 候选核源码。
    @param workdir 存放请求、心跳与 trace 的目录。
    @param duration_s 录制时长；省略时按模板选择。
    @param record_limit_s xctrace 的 time-limit 秒数。
    @param launch_deadline_s 单次启动的超时秒数。
    @return .trace 包路径。
    @raises TraceError 当源码被预筛拒绝、缺少工具、超时或没有产出 trace。

    Record a Metal System Trace of ``kernel_source`` launching repeatedly. Returns the .trace bundle path.

    With a counter template the trace holds ~12 000 samples of 85 counters per second (hundreds of MB of XML to export), so it is recorded for 0.4 s instead of 1 s."""
    if duration_s is None:
        duration_s = 1.0 if TEMPLATE == "Metal System Trace" else 0.4
    from ...evaluate import _read_heartbeat
    from .screen import screen_source

    refused = screen_source(kernel_source)
    if refused:
        raise TraceError("refused: " + refused[0])
    if not shutil.which("xcrun"):
        raise TraceError("xcrun not found (Xcode command line tools required)")
    workdir.mkdir(parents=True, exist_ok=True)
    kp, hb, out, rq = workdir / "kernel.metal", workdir / "heartbeat", workdir / "out.json", workdir / "request.json"
    trace = workdir / "capture.trace"
    if trace.exists():
        shutil.rmtree(trace)
    kp.write_text(kernel_source)
    rq.write_text(json.dumps({"task": task, "dtype": dtype, "size": size, "kernel_path": str(kp), "out": str(out), "heartbeat": str(hb), "duration_s": duration_s, "batch": BATCH}))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "xcrun", "xctrace", "record", "--template", TEMPLATE, "--output", str(trace), "--time-limit", f"{int(record_limit_s)}s", "--no-prompt",
        "--launch", "--", sys.executable, "-m", "tcenv.xcworker", str(rq),
    ]
    from ...evaluate import LAUNCH_DEADLINE_S, deadline_for_source

    wait_s = deadline_for_source(task, dtype, size, kernel_source) if launch_deadline_s == LAUNCH_DEADLINE_S else launch_deadline_s
    t0 = time.perf_counter()
    with open(workdir / "xctrace.log", "w") as log:
        from ...children import forget, kill_group, popen_group

        proc = popen_group(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
        try:
            while proc.poll() is None:
                time.sleep(0.2)
                h = _read_heartbeat(hb)
                if h and h[0] == "B" and time.time() - h[1] > wait_s:
                    _kill_tree(proc)
                    raise TraceError(f"a GPU launch ran for more than {wait_s}s while profiling; killed")
                if time.perf_counter() - t0 > record_limit_s + 30:
                    _kill_tree(proc)
                    raise TraceError("xctrace did not finish in time")
        finally:
            kill_group(proc)
            forget(proc)
    if not trace.exists() or not out.exists():
        raise TraceError("no trace produced:\n" + (workdir / "xctrace.log").read_text()[-600:])
    return trace


def _kill_tree(proc: subprocess.Popen) -> None:
    """@brief 结束子进程及其整棵进程树。

    @param proc 子进程对象。
    """
    from ...children import kill_group

    kill_group(proc)


# --------------------------------------------------------------------------------------------- export


def export_table(trace: Path, schema: str, dest: Path, timeout: float = 300.0) -> Path:
    """@brief 用 xctrace 把一张表导出成 XML 文件。

    @param trace .trace 包路径。
    @param schema Instruments 表名。
    @param dest 输出文件路径。
    @param timeout 导出超时秒数。
    @return 输出文件路径。
    @raises TraceError 当导出命令失败。
    """
    xp = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    with open(dest, "w") as f:
        r = subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", xp], stdout=f, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if r.returncode != 0:
        raise TraceError(f"xctrace export {schema} failed: {r.stderr[-300:]}")
    return dest


# --------------------------------------------------------------------------------------------- parse


def _columns(path: Path) -> list[str]:
    """@brief 从导出 XML 的表头读出列名。

    @param path XML 文件路径。
    @return 列 mnemonic 列表。
    """
    with open(path, "r", errors="replace") as f:
        head = f.read(20000)
    sch = head[head.find("<schema") : head.find("</schema>")]
    return re.findall(r"<mnemonic>([^<]+)</mnemonic>", sch)


def iter_rows(path: Path):
    """@brief 逐行产出 ``{列名: (文本, 格式)}`` 的字典。

    @param path 导出 XML 文件路径。
    @return 行字典的生成器，已解析 Instruments 的 id/ref 复用。

    Yield dicts {column mnemonic: (text, fmt)} resolving Instruments' id/ref value sharing.
    """
    cols = _columns(path)
    table: dict[str, tuple[str, str]] = {}
    for _ev, el in ET.iterparse(str(path), events=("end",)):
        if el.tag != "row":
            continue
        row = {}
        for col, ch in zip(cols, list(el)):
            ref, ident = ch.get("ref"), ch.get("id")
            if ref is not None:
                val = table.get(ref, ("", ""))
            else:
                val = ((ch.text or ""), ch.get("fmt", ""))
                if ident is not None:
                    table[ident] = val
            row[col] = val
        yield row
        el.clear()


@dataclass
class CounterStat:
    mean: float
    max: float
    p90: float
    samples: int
    unit: str = "%"  # "%" for the percentage counters (utilization, limiters, occupancy), " GiB/s" / " simdgroups/core" for the plain values
    p50: float | None = None  # the typical value while the kernel runs; defaults to the mean

    def __post_init__(self) -> None:
        """@brief 未给中位数时用均值补齐。"""
        if self.p50 is None:
            self.p50 = self.mean


@dataclass
class TraceSummary:
    window_s: float = 0.0
    n_dispatches: int = 0
    dispatch_group: int = 1  # 1: one interval is one launch. >1: the median sums that many dispatches of one iteration
    dispatch_us_median: float | None = None
    dispatch_us_p10: float | None = None
    dispatch_us_p90: float | None = None
    gpu_active_fraction: float | None = None  # busy fraction inside the window; 1 - this = launch gaps
    spilled_bytes: int = 0
    spill_events: int = 0
    shader_code_bytes: int | None = None
    thermal_states: list[str] = field(default_factory=list)
    counters: dict[str, CounterStat] = field(default_factory=dict)
    target_id: str = "apple-m4-max"
    # Which tables the trace actually recorded. An empty *recorded* table means "none happened";
    # an absent table means "not measured", and neither the summary nor the advice may turn that
    # into a negative ("no spills", "not throttled").
    tables: dict[str, bool] = field(default_factory=dict)
    # Process names exported on the compute channel. Empty means Instruments did not attribute the
    # dispatches, so the window and the dispatch counts are machine-wide, not just our kernel.
    window_processes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        """@brief 转成可写入报告的字典。

        @return 字段字典，counters 展开为统计子字典。
        """
        d = {k: v for k, v in self.__dict__.items() if k != "counters"}
        d["counters"] = {k: {"p50": v.p50, "mean": v.mean, "max": v.max, "p90": v.p90, "samples": v.samples, "unit": v.unit} for k, v in sorted(self.counters.items())}
        return d


def compute_window(intervals_xml: Path) -> tuple[float, float, list[float]]:
    """@brief 求计算通道的时间窗与每次 dispatch 的时长。

    @param intervals_xml metal-gpu-intervals 导出文件。
    @return (起始 ns, 结束 ns, 每次 dispatch 的时长 ns 列表)。
    @raises TraceError 当 trace 里没有计算区间。

    (start_ns, end_ns, per-dispatch durations_ns) of the compute channel in metal-gpu-intervals.
    The render channels belong to the window server; the profiled process is the only one doing compute work.
    """
    starts, ends, durs = [], [], []
    for row in iter_rows(intervals_xml):
        chan = row.get("channel-name", ("", ""))[1]
        if "compute" not in chan.lower():
            continue
        s, d = float(row["start"][0]), float(row["duration"][0])
        starts.append(s)
        ends.append(s + d)
        durs.append(d)
    if not starts:
        raise TraceError("no compute intervals in the trace")
    return min(starts), max(ends), durs


def compute_processes(intervals_xml: Path) -> set[str]:
    """@brief 读出计算通道上被导出的进程名。

    @param intervals_xml metal-gpu-intervals 导出文件。
    @return 进程名集合；Instruments 没有填该列时为空。

    Process names on the compute channel, as exported. Empty when Instruments did not fill the column.
    """
    out: set[str] = set()
    for row in iter_rows(intervals_xml):
        if "compute" not in row.get("channel-name", ("", ""))[1].lower():
            continue
        # Instruments names this mnemonic "process" in some exports and "process-name" in others.
        name = (row.get("process") or row.get("process-name") or ("", ""))[0].strip()
        if name:
            out.add(name)
    return out


def summarize_counters(counters_xml: Path, window: tuple[float, float]) -> dict[str, CounterStat]:
    """@brief 统计时间窗内的百分比计数器。

    @param counters_xml metal-gpu-counter-intervals 导出文件。
    @param window (起始 ns, 结束 ns)。
    @return 计数器名到 CounterStat 的映射。
    """
    lo, hi = window
    vals: dict[str, list[tuple[float, float]]] = {}
    for row in iter_rows(counters_xml):
        s = float(row["start"][0])
        if s < lo or s > hi:
            continue
        name = re.sub(r"^\d+\.", "", row["name"][0] or row["name"][1])
        pv = row["percent-value"][0] or row["value"][0]
        vals.setdefault(name, []).append((float(row["duration"][0]), float(pv)))
    out: dict[str, CounterStat] = {}
    for name, xs in vals.items():
        tot = sum(d for d, _ in xs) or 1.0
        mean = sum(d * v for d, v in xs) / tot
        srt = sorted(v for _, v in xs)
        out[name] = CounterStat(mean=mean, max=srt[-1], p90=srt[int(0.9 * (len(srt) - 1))], samples=len(xs), p50=srt[(len(srt) - 1) // 2])
    return out


def dispatch_spans(intervals_xml: Path) -> list[tuple[float, float]]:
    """@brief 求每次计算 dispatch 的起止时间，排序并合并重叠区间。

    @param intervals_xml metal-gpu-intervals 导出文件。
    @return 合并后的区间列表（ns）。

    (start_ns, end_ns) of every compute dispatch, sorted and merged: the GPU counters are only meaningful while our kernel runs.
    """
    spans = []
    for row in iter_rows(intervals_xml):
        if "compute" not in row.get("channel-name", ("", ""))[1].lower():
            continue
        s, d = float(row["start"][0]), float(row["duration"][0])
        spans.append((s, s + d))
    spans.sort()
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


_CELL = re.compile(r'<(\w[\w-]*) (?:id="(\d+)"(?: fmt="[^"]*")?>([^<]*)</\1>|ref="(\d+)"/>)')


def _counter_unit(name: str, type_: str, description: str) -> str:
    """@brief 从计数器类型与 Apple 的描述推断单位。

    @param name 计数器名。
    @param type_ 计数器类型。
    @param description Apple 的计数器说明。
    @return 单位字符串，无法判断时为空。
    """
    if type_ == "Percentage":
        return "%"
    d = description.lower()
    if "gib/s" in d:
        return " GiB/s"
    if "gigabytes per second" in d:
        return " GB/s"
    if "simdgroups" in d or "simd groups" in name.lower():
        return " simdgroups/core"
    return ""


def counter_defs(info_xml: Path) -> dict[int, tuple[str, str]]:
    """@brief 读出计数器 id 到 (名字, 单位) 的映射。

    @param info_xml gpu-counter-info 导出文件。
    @return 计数器定义字典。

    counter id -> (name, unit), from the ``gpu-counter-info`` table (the unit comes from Apple's description of the counter).
    """
    out = {}
    for row in iter_rows(info_xml):
        name = row["name"][0]
        out[int(row["counter-id"][0])] = (name, _counter_unit(name, row["type"][0], row["description"][0]))
    return out


def summarize_counter_values(values_xml: Path, defs: dict[int, tuple[str, str]], spans: list[tuple[float, float]]) -> dict[str, CounterStat]:
    """@brief 统计落在本次核 dispatch 区间内的计数器采样。

    @param values_xml gpu-counter-value 导出文件。
    @param defs 计数器定义。
    @param spans 本次核的 dispatch 区间列表。
    @return 计数器名到 CounterStat 的映射。

    Statistics of every counter over the samples that fall inside a dispatch of our kernel.

    The ``gpu-counter-value`` table of a trace recorded with a counter profile holds one row per (sample time, counter), hundreds of thousands in a
    second of tracing, so it is read with a regex per row instead of building an XML tree. Instruments shares repeated values through ``id`` / ``ref``."""
    import bisect

    starts = [a for a, _ in spans]
    ends = [b for _, b in spans]
    shared: dict[str, str] = {}
    vals: dict[int, list[float]] = {}
    last_ts, last_in = None, False
    with open(values_xml, "r", errors="replace") as f:
        for line in f:
            k = line.find("<row>")  # the first row follows the schema on the same line
            if k < 0:
                continue
            cells = _CELL.findall(line, k)
            if len(cells) < 3:
                continue
            txt = []
            for j, (_tag, ident, text, ref) in enumerate(cells):  # every cell may define an id that later rows refer to, also the unused ones
                if ref:
                    t = shared.get(ref, "")
                else:
                    shared[ident] = t = text
                if j < 3:
                    txt.append(t)
            ts_s, cid_s, v_s = txt
            if ts_s != last_ts:
                last_ts = ts_s
                ts = float(ts_s)
                i = bisect.bisect_right(starts, ts) - 1
                last_in = i >= 0 and ts <= ends[i]
            if last_in:
                vals.setdefault(int(cid_s), []).append(float(v_s))
    out: dict[str, CounterStat] = {}
    for cid, xs in vals.items():
        if cid not in defs:
            continue
        name, unit = defs[cid]
        srt = sorted(xs)
        out[name] = CounterStat(mean=sum(xs) / len(xs), max=srt[-1], p90=srt[int(0.9 * (len(srt) - 1))], samples=len(xs), unit=unit, p50=srt[(len(srt) - 1) // 2])
    return out


def _pct(xs: list[float], q: float) -> float:
    """@brief 取列表的分位数。

    @param xs 数值列表。
    @param q 分位，0 到 1。
    @return 对应分位值。
    """
    s = sorted(xs)
    return s[int(q * (len(s) - 1))]


def analyze(trace: Path, workdir: Path, *, kernel_name: str | None = None, kernel_names: list[str] | None = None) -> TraceSummary:
    """@brief 导出并汇总 trace 里的区间、状态、溢出、代码大小、温度与计数器。

    @param trace .trace 包路径。
    @param workdir 导出文件与元数据所在目录。
    @param kernel_name 候选核名，用于筛 shader 列表。
    @param kernel_names 多个候选核名；优先用工作目录里记录的名字。
    @return TraceSummary 汇总。
    @raises TraceError 当关键表导出失败或 trace 里没有计算区间。
    """
    iv = export_table(trace, "metal-gpu-intervals", workdir / "gpu_intervals.xml")
    lo, hi, durs = compute_window(iv)
    group, recorded_names = 1, None
    out_path = workdir / "out.json"
    if out_path.exists():
        try:
            meta = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        if isinstance(meta, dict):
            group = int(meta.get("dispatch_group") or 1)
            if isinstance(meta.get("kernels"), list) and meta["kernels"]:
                recorded_names = [str(n) for n in meta["kernels"]]
    raw = len(durs)
    if group > 1 and len(durs) >= group:
        durs = [sum(durs[i : i + group]) for i in range(0, len(durs) - (len(durs) % group), group)]
    summ = TraceSummary(window_s=(hi - lo) / 1e9, n_dispatches=raw, dispatch_group=group)
    summ.dispatch_us_median, summ.dispatch_us_p10, summ.dispatch_us_p90 = (_pct(durs, q) / 1e3 for q in (0.5, 0.1, 0.9))
    kernel_names = recorded_names or kernel_names

    summ.window_processes = sorted(compute_processes(iv))
    # GPU busy fraction inside the window. An absent state table must not read as "0% active":
    # that is a finding about launch gaps, not a measurement of them.
    st = export_table(trace, "metal-gpu-state-intervals", workdir / "gpu_state.xml")
    summ.tables["gpu_state"] = bool(_columns(st))
    if summ.tables["gpu_state"]:
        busy = 0.0
        for row in iter_rows(st):
            s0, d = float(row["start"][0]), float(row["duration"][0])
            if row["state"][1] == "Active" and lo <= s0 <= hi:
                busy += min(d, hi - s0)
        summ.gpu_active_fraction = min(busy / max(hi - lo, 1.0), 1.0)

    sp = export_table(trace, "graphics-compiler-spill-events", workdir / "spills.xml")
    summ.tables["spills"] = bool(_columns(sp))
    for row in iter_rows(sp):
        summ.spill_events += 1
        summ.spilled_bytes += int(float(row["spilled-bytes"][0] or 0))

    sh = export_table(trace, "metal-shader-profiler-shader-list", workdir / "shaders.xml")
    summ.tables["shaders"] = bool(_columns(sh))
    names = kernel_names or ([kernel_name] if kernel_name else None)
    sizes = [
        int(float(r["pc-end"][0])) - int(float(r["pc-start"][0]))
        for r in iter_rows(sh)
        if r["shader-type"][1] == "Compute" and (names is None or any(n in r["name"][1] for n in names))
    ]
    summ.shader_code_bytes = max(sizes) if sizes else None

    th = export_table(trace, "device-thermal-state-intervals", workdir / "thermal.xml")
    summ.tables["thermal"] = bool(_columns(th))
    summ.thermal_states = sorted({r["thermal-state"][1] for r in iter_rows(th)})

    info = export_table(trace, "gpu-counter-info", workdir / "counter_info.xml")
    if sum(1 for _ in iter_rows(info)) >= 3:  # a counter profile was active (custom template); the default template records one useless counter
        # measured: a template saved with the "Performance Limiters" counter set records ~85 counters as raw samples in ``gpu-counter-value``
        # while ``metal-gpu-counter-intervals`` stays empty; the interval table is kept as a fallback for other counter profiles
        cv = export_table(trace, "gpu-counter-value", workdir / "counter_values.xml", timeout=900)
        summ.counters = summarize_counter_values(cv, counter_defs(info), dispatch_spans(iv))
        if not summ.counters:
            ct = export_table(trace, "metal-gpu-counter-intervals", workdir / "counter_intervals.xml", timeout=900)
            summ.counters = summarize_counters(ct, (lo, hi))
    return summ


KEY_COUNTERS = (
    "Kernel Occupancy", "Occupancy Manager Target", "L1 Eviction Rate", "Compute SIMD Groups Inflight",
    "Instruction Throughput Utilization", "ALU Utilization", "F32 Utilization", "F16 Utilization",
    "L1 Cache Utilization", "Buffer L1 Miss Rate", "L1 Register Residency", "L1 Threadgroup Residency", "Last Level Cache Utilization",
    "GPU Read Bandwidth", "GPU Write Bandwidth", "Threadgroup Memory L1 Read Bandwidth", "Threadgroup Memory L1 Write Bandwidth",
)
SUB_PIPE_LIMITERS = ("F32 Limiter", "F16 Limiter", "Integer and Conditional Limiter", "Integer and Complex Limiter", "Control Flow Limiter")
# interval comes from the active target's counter_sample_us


def _counter_line(n: str, c: CounterStat) -> str:
    """@brief 把一条计数器统计格式成报告行。

    @param n 计数器名。
    @param c 计数器统计。
    @return 单行文本。
    """
    u = c.unit
    return f"    {n:<40s} typical {c.p50:7.1f}{u:<4s} p90 {c.p90:7.1f}{u:<4s} max {c.max:7.1f}{u}"


# what a saturated limiter means for a kernel author (measured on Apple GPUs: Instruments "Performance Limiters" counter set)
LIMITER_MEANING = {
    "Instruction Throughput Limiter": "instruction issue: the kernel is bound by how many instructions it executes. Cut the instruction count: hoist invariant work, simplify address arithmetic, vectorize (half2/half4, packed loads), unroll with care.",
    "F32 Limiter": "the FP32 ALUs: do less FP32 arithmetic (use half where the precision allows, fuse multiply-adds, use cheaper approximations or a matrix instruction).",
    "F16 Limiter": "the FP16 ALUs: this is the compute roof for scalar half math; moving the dot products onto simdgroup_matrix operations is the way past it.",
    "Integer and Conditional Limiter": "integer and compare/select work (index math, predicates): simplify indexing, precompute offsets, remove per-element branches and bounds checks.",
    "Integer and Complex Limiter": "integer and complex-function work (multiplies, divisions, transcendental functions): strength-reduce, use shifts and masks, avoid division and modulo in the inner loop.",
    "Control Flow Limiter": "branches and loops: unroll small loops, make control flow uniform across the simdgroup, remove divergent branches.",
    "L1 Cache Limiter": "the shader core L1 (buffer, threadgroup and register traffic): fewer accesses per FLOP, wider loads, register reuse, avoid threadgroup bank conflicts.",
    "Last Level Cache Limiter": "the last-level cache: reduce bytes moved, increase reuse in registers/threadgroup memory, make accesses contiguous.",
    "MMU Limiter": "address translation (page-table walks): the access pattern touches too many pages; keep accesses contiguous and tiles compact.",
    "Compute Shader Launch Limiter": "launching threads into the cores: the grid has too many tiny threadgroups; use fewer, larger threadgroups with more work per thread.",
}


def format_summary(summ: TraceSummary, top: int = 14) -> str:
    """@brief 把 trace 汇总排成人类可读的报告。

    @param summ trace 汇总。
    @param top 保留的顶层计数器条数。
    @return 多行文本。
    """
    L = [f"Instruments Metal System Trace ({summ.n_dispatches} dispatches over {summ.window_s * 1e3:.0f} ms of back-to-back launches):"]
    if summ.tables.get("spills", True):
        L.append(f"  register spills: {summ.spill_events} events, {summ.spilled_bytes} bytes" + ("  <-- the compiler spilled registers to memory: reduce live values / unrolling" if summ.spilled_bytes else ""))
    else:
        L.append("  register spills: not recorded by this trace template (unknown, not zero)")
    if summ.tables.get("shaders", True):
        if summ.shader_code_bytes is not None:
            L.append(f"  compiled shader size: {summ.shader_code_bytes} bytes of GPU code")
    else:
        L.append("  compiled shader size: not recorded by this trace template")
    if summ.tables.get("thermal", True):
        L.append(f"  thermal state: {', '.join(summ.thermal_states) or 'unknown'}" + ("  (throttling: timings are unreliable)" if any(t not in ("Nominal",) for t in summ.thermal_states) else ""))
    else:
        L.append("  thermal state: not recorded by this trace template")
    if len(summ.window_processes) > 1:
        L.append(f"  scope: the window includes compute from {', '.join(summ.window_processes)} (machine-wide; dispatch counts and medians are not ours alone)")
    elif not summ.window_processes:
        L.append("  scope: Instruments did not export a process column, so the window is machine-wide; dispatch counts and medians include any other compute work")
    if summ.dispatch_group > 1:
        interval = (
            f"GPU interval is the sum of {summ.dispatch_group} dispatches in one iteration "
            f"(the stages, plus a zero-fill when the pipeline has workspace)"
        )
    else:
        interval = "GPU interval per dispatch"
    active = f"GPU active {summ.gpu_active_fraction:.0%} of the window" if summ.gpu_active_fraction is not None else "GPU active fraction not recorded"
    stats = []
    if summ.dispatch_us_median is not None:
        stats.append(f"median {summ.dispatch_us_median:.1f} us")
    if summ.dispatch_us_p10 is not None:
        stats.append(f"p10 {summ.dispatch_us_p10:.1f}")
    if summ.dispatch_us_p90 is not None:
        stats.append(f"p90 {summ.dispatch_us_p90:.1f}")
    L.append(
        f"  indicative only (tracing perturbs timing): {interval} "
        + (", ".join(stats) or "no dispatch durations")
        + f"; {active}"
    )
    if summ.counters:
        L.append("  hardware counters (GPU-wide, sampled every ~41 us while our kernel runs; 'typical' is the median sample; the highest limiter is what the kernel waits for):")
        lim = sorted(((n, c) for n, c in summ.counters.items() if "Limiter" in n and not n.startswith(("Vertex", "Fragment", "Texture"))), key=lambda kv: -kv[1].p50)
        L += [_counter_line(n, c) for n, c in lim[:6]]
        L += [_counter_line(n, summ.counters[n]) for n in KEY_COUNTERS if n in summ.counters]
    return "\n".join(L)


def _counter_advice(counters: dict[str, CounterStat], dispatch_us: float | None = None, target_id: str | None = None) -> list[str]:
    """@brief 从 GPU 计数器给出该核受限于哪个单元的结论。

    @param counters 计数器统计。
    @param dispatch_us 单次 dispatch 的中位微秒数。
    @param target_id 目标机型标识。
    @return 结论文本列表。

    Findings from the GPU counters (Performance Limiters): which unit the kernel waits for, or that none is saturated.
    """
    lims = [(n, c) for n, c in counters.items() if n in LIMITER_MEANING]
    if not lims:
        return []
    name, top = max(lims, key=lambda kv: kv[1].p50)
    occ = counters.get("Kernel Occupancy")
    from ...target import DEFAULT_TARGET_ID, load_target

    sample_us = load_target(target_id or DEFAULT_TARGET_ID).limits.counter_sample_us
    out = []
    if sample_us and dispatch_us is not None and dispatch_us < 4 * sample_us:
        out.append(
            f"The dispatches last only {dispatch_us:.0f} us, about {dispatch_us / sample_us:.1f} counter samples each: "
            "the counters are blurred by the ramp-up and ramp-down at the edges, read them as indicative."
        )
    if top.p50 >= 40:
        out.append(f"Counters: the highest limiter is {name} at {top.p50:.0f}% typical (p90 {top.p90:.0f}%). It limits on {LIMITER_MEANING[name]}")
        itl = counters.get("Instruction Throughput Limiter")
        if name in SUB_PIPE_LIMITERS and top.p50 >= 80 and itl and itl.p50 < 0.9 * top.p50:
            out.append(
                f"{name} is saturated ({top.p50:.0f}%) while the instruction throughput limiter is {itl.p50:.0f}%: the other ALU sub-pipelines (F32, F16, integer-and-conditional, integer-and-complex) "
                "have idle capacity and run in parallel with it, so moving work onto them is possible (e.g. integer index math instead of float, half instead of float where precision allows, select instead of branch)."
            )
        if occ and occ.p50 < 25:
            out.append(f"Kernel occupancy is low ({occ.p50:.0f}%) but a unit is already saturated, so more resident simdgroups would not help: reduce the work that unit does.")
    else:
        out.append(
            f"Counters: no limiter is saturated (the highest, {name}, is {top.p50:.0f}% typical)"
            + (f" and kernel occupancy is {occ.p50:.0f}%" if occ else "")
            + ": the GPU is mostly waiting, not working. That points to latency and dependency stalls (barriers, chains of dependent loads or FMAs) or too little parallel work in flight. "
            "Give each thread more independent work (several accumulators), prefetch or double-buffer loads, use fewer barriers, and keep registers and threadgroup memory per thread low enough for more simdgroups to be resident. "
            "If the grid itself is small (the launch contract fixes it), occupancy cannot be raised from the kernel body: make each simdgroup's work faster instead."
        )
    ev = counters.get("L1 Eviction Rate")
    if ev and ev.p50 >= 50:
        tgt = counters.get("Occupancy Manager Target")
        out.append(
            f"L1 eviction rate is {ev.p50:.0f}% typical: the shader core's private memory (registers, stack, spills, threadgroup data) is thrashing the L1, and the hardware occupancy manager "
            + (f"is holding occupancy at a target of {tgt.p50:.0f}% " if tgt else "is throttling occupancy ")
            + "because of it. Shrink the per-thread working set: smaller register tiles, fewer live arrays, no spills."
        )
    return out


def advice(summ: TraceSummary) -> list[str]:
    """@brief 列出 trace 实际支持、作者可以动手的结论。

    @param summ trace 汇总。
    @return 结论文本列表。

    What the trace says that a kernel author can act on. Only findings that the trace actually supports are listed.
    """
    out = []
    if summ.spilled_bytes:
        out.append(
            f"The compiler spilled {summ.spilled_bytes} bytes of registers to memory ({summ.spill_events} events): the kernel needs more registers per thread than the GPU has. "
            "Spilled values are written and re-read through memory on every use, which usually costs more than anything else here. "
            "Cut live state per thread: smaller register tiles or fewer accumulators, no full unrolling of long loops, keep large arrays in threadgroup memory, recompute cheap values instead of keeping them."
        )
    if summ.tables.get("gpu_state", True) and summ.gpu_active_fraction is not None and summ.gpu_active_fraction < 0.8:
        out.append(
            f"The GPU was active only {summ.gpu_active_fraction:.0%} of the window: gaps between launches (launch/dispatch overhead) matter at this problem size; "
            "this is a property of the launch contract and the dispatch pattern, not of the kernel body."
        )
    if summ.tables.get("shaders", True) and summ.shader_code_bytes and summ.shader_code_bytes > 48 * 1024:
        out.append(f"The compiled shader is large ({summ.shader_code_bytes} bytes): heavy unrolling or inlining; instruction fetch can become the limiter.")
    if summ.tables.get("thermal", True) and any(t != "Nominal" for t in summ.thermal_states):
        out.append("The GPU was thermally throttled while tracing: do not read small timing differences into this run.")
    out += _counter_advice(summ.counters, summ.dispatch_us_median, target_id=summ.target_id)
    if not out:
        checked = [label for key, label in (("spills", "register spills"), ("gpu_state", "launch gaps"), ("shaders", "code size"), ("thermal", "throttling")) if summ.tables.get(key, True)]
        missing = [label for key, label in (("spills", "register spills"), ("gpu_state", "launch gaps"), ("shaders", "code size"), ("thermal", "throttling")) if not summ.tables.get(key, True)]
        out.append(
            ("Nothing was found in the recorded tables (" + ", ".join(checked) + ")" if checked else "This trace recorded none of the usual tables")
            + (". Not recorded, so unknown rather than cleared: " + ", ".join(missing) if missing else "")
            + ": the slowness is in the instruction mix, memory access pattern or synchronization of the kernel itself."
        )
    return out


def format_for_llm(summ: TraceSummary) -> str:
    """@brief 生成给模型看的 trace 摘要加结论。

    @param summ trace 汇总。
    @return 摘要与逐条结论拼成的文本。

    The trace summary followed by what it implies, as it is given to the model.
    """
    return format_summary(summ) + "\n  findings:\n" + "\n".join(f"    - {a}" for a in advice(summ))


def get_task_kernel_name(task: str, dtype: str, size: str) -> str:
    """@brief 取任务契约里的核函数名。

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模名。
    @return 契约声明的 kernel 名。
    """
    from tasks import get_task

    return get_task(task, dtype, size).contract.kernel_name


def profile_kernel(task: str, dtype: str, size: str, kernel_source: str, workdir: Path | None = None, keep: bool = False, target_id: str = "apple-m4-max") -> TraceSummary:
    """@brief 对候选核做一次 xctrace 录制并汇总。

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模名。
    @param kernel_source 候选核源码。
    @param workdir 工作目录；省略时用临时目录。
    @param keep 是否保留临时工作目录。
    @param target_id 目标机型标识。
    @return TraceSummary 汇总。
    @raises TraceError 录制阶段失败时向上抛出。
    """
    own = workdir is None
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="tcenv_xc_"))
    from ...gpulock import gpu_lock

    try:
        with gpu_lock(f"xctrace recording of {task} {dtype} {size}"):
            trace = record(task, dtype, size, kernel_source, wd)
        summ = analyze(trace, wd, kernel_name=get_task_kernel_name(task, dtype, size))
        summ.target_id = target_id
        return summ
    finally:
        if own and not keep:
            shutil.rmtree(wd, ignore_errors=True)
