"""Human / LLM readable rendering of an ``EvalReport``."""
# 反馈关，不再判生死。搜索把这里的短文塞进下一轮提示，让模型知道死在哪一关。
# 截断发生在本函数里：编译错误只留前 max_errors_chars 个字符，安全违例、非确定、shader validation 各只留前 6、3、4 条。
# 整份 worker 日志、心跳、每次测速的原始样本都不进这里。全文塞回去，模型读到的是日志，定位不到该改的那一关。搜索还会把本函数的输出再截到固定字数，放进下一轮提示。

from __future__ import annotations

from .evaluate import EvalReport


# 基线比下界快过 5% 时，下界的运算量假设不成立。与 taic.feedback_extra.CLEAR_OF_BOUND 同一条线。
_CLEAR_OF_BOUND = 0.95


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _baseline_beats_bound(rep: EvalReport) -> tuple[str, float, float] | None:
    """(name, microseconds, lower bound) when the fastest baseline is clearly under the roofline bound."""
    roof = rep.roofline if isinstance(rep.roofline, dict) else {}
    bases = rep.baseline_us if isinstance(rep.baseline_us, dict) else {}
    lb = _num(roof.get("lower_bound_us"))
    usable = {k: float(v) for k, v in bases.items() if _num(v) is not None and float(v) > 0}
    if lb is None or lb <= 0 or not usable:
        return None
    name, us = min(usable.items(), key=lambda kv: kv[1])
    return (name, us, lb) if us < lb * _CLEAR_OF_BOUND else None


def _measured_head(rep: EvalReport) -> list[str]:
    """Short measured lines placed before the long sections, so a later 6000-character cut cannot drop them.

    A missing field is omitted. A baseline that beats the bound replaces the "times the lower bound" line
    and adds that a faster baseline is not a reason to switch algorithm.
    """
    lines: list[str] = []
    roof = rep.roofline if isinstance(rep.roofline, dict) else {}
    bound = roof.get("bound")
    if bound in ("memory", "compute"):
        lines.append(f"bound: {bound}")
    beat = _baseline_beats_bound(rep)
    gap = _num(roof.get("time_over_lower_bound"))
    if beat is not None:
        name, us, lb = beat
        lines.append(
            f"lower bound {lb:.1f} us is not the target: fastest baseline ({name}) is already {us:.1f} us. "
            "Every case, including ieee, still has to match the reference."
        )
    elif gap is not None and gap > 0:
        lines.append(f"candidate is {gap:.2f}x the lower bound")
    if bound == "compute":
        frac = _num(roof.get("compute_fraction"))
        if frac is not None:
            lines.append(f"compute utilization: {frac:.0%}")
    elif bound == "memory":
        frac = _num(roof.get("bandwidth_fraction"))
        if frac is not None:
            lines.append(f"bandwidth utilization: {frac:.0%}")
    bench = rep.benchmark if isinstance(rep.benchmark, dict) else {}
    cand = bench.get("candidate") if isinstance(bench.get("candidate"), dict) else None
    cv = _num(cand.get("cv")) if cand else None
    if cv is not None:
        lines.append(f"timing cv: {cv:.1%}")
    pipe = rep.pipeline if isinstance(rep.pipeline, dict) else {}
    static = pipe.get("static_threadgroup_bytes")
    limit = pipe.get("max_threadgroup_memory_bytes")
    if isinstance(static, int) and not isinstance(static, bool) and isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        lines.append(f"threadgroup memory: {static} B / {limit} B limit")
    elif isinstance(static, int) and not isinstance(static, bool):
        lines.append(f"threadgroup memory: {static} B")
    stages = pipe.get("stages") if isinstance(pipe.get("stages"), list) else None
    if stages:
        lines.append(f"pipeline stages: {len(stages)}")
    else:
        cap = pipe.get("max_total_threads_per_threadgroup")
        launch = pipe.get("launch_threads_per_threadgroup")
        if isinstance(cap, int) and not isinstance(cap, bool) and isinstance(launch, int) and not isinstance(launch, bool):
            lines.append(f"threadgroup thread cap {cap}, this launch {launch}")
    if isinstance(rep.compile, dict) and rep.compile.get("launch"):
        lines.append("achieved fractions use the reference algorithm's operation and byte counts")
    if not lines:
        return []
    return ["Measured:", *[f"  {line}" for line in lines]]


# 把已经写好的判决收成短文。编译错误按字符截断；安全、非确定、校验按条数截断。ABI 违例逐条全留。
# 性能和形式验证只在 status 为 ok 时附上：已经判死的候选不再带着速度结论回去。
# 数字头紧跟 STATUS。搜索再把全文截到固定字数时，砍的是后面的正确性散文，不是这几行。
def format_feedback(rep: EvalReport, *, max_errors_chars: int = 2500) -> str:
    L: list[str] = []
    L.append(f"STATUS: {rep.status}  (stage reached: {rep.stage})")
    L.extend(_measured_head(rep))
    if rep.message:
        L.append(rep.message)
    for note in rep.device_warnings:
        L.append(note)

    # 行号已经由契约层映回候选自己的源码。这里再按字符砍掉后半，避免整段编译器输出占满下一轮提示。
    if rep.compile and not rep.compile.get("ok", True):
        L.append("\nMetal compiler errors (line numbers refer to your kernel source):")
        L.append(rep.compile.get("errors", "")[:max_errors_chars])
    if rep.abi and not rep.abi.get("ok", True):
        L.append("\nABI / launch contract violations:")
        L.extend(f"  - {v}" for v in rep.abi["violations"])
    if rep.safety and not rep.safety.get("ok", True):
        L.append("\nSafety problems:")
        for v in rep.safety.get("violations", [])[:6]:
            L.append(f"  - {v}")
        for v in rep.safety.get("nondeterminism", [])[:3]:
            L.append(f"  - {v}")
        for v in rep.safety.get("validation", [])[:4]:
            L.append(f"  - shader validation: {v}")
    if rep.correctness:
        cases = rep.correctness.get("cases", [])
        if cases:
            L.append("\nCorrectness (randomized differential tests vs fp64/fp32 CPU reference):")
            tol = rep.correctness.get("tolerance", {})
            if tol:
                L.append(f"  tolerance: |out-ref| <= {tol.get('rtol')}*|ref| + {tol.get('atol_rel')}*scale (scaled error 1.0 == at tolerance)")
            for c in cases:
                flag = "PASS" if c["ok"] else "FAIL"
                err = c["scaled_err"]
                L.append(f"  {flag} {c['case']:<11s} worst scaled error {err if isinstance(err, str) else format(err, '.3g'):>8}  allowed {c['allow'] if isinstance(c['allow'], str) else format(c['allow'], '.3g')}")
                if not c["ok"] and c.get("message"):
                    L.append(f"       {c['message']}")
    if rep.status == "ok" and isinstance(rep.benchmark, dict) and rep.benchmark.get("candidate"):
        b = rep.benchmark
        L.append("\nPerformance (interleaved benchmark, median per launch):")
        cand = b["candidate"]
        gpu = b.get("candidate_gpu_us_median")
        L.append(f"  candidate : {cand['median_us']:.2f} us   (cv {cand['cv']:.1%}{', GPU-timestamp %.2f us' % gpu if gpu else ''})")
        seed_gpu = b.get("seed_gpu_us_median")
        if isinstance(gpu, (int, float)) and not isinstance(gpu, bool) and gpu > 0 and isinstance(seed_gpu, (int, float)) and not isinstance(seed_gpu, bool) and seed_gpu > 0:
            L.append(f"  GPU-timestamp speedup vs seed: {seed_gpu / gpu:.3f}x (seed {seed_gpu:.2f} us)")
        # 重测后最安静的一次仍超过阈值。候选可以是 ok，但这句告诉模型百分之几的差距没有意义。
        if b.get("noisy"):
            L.append(f"  note: the machine was busy during this measurement (cv up to {b['noise_cv']:.0%} after {b['attempts']} attempts); differences of a few percent are not meaningful")
        for n, s in b.get("baselines", {}).items():
            sp = rep.speedups.get(n)
            L.append(f"  {n:<10s}: {s['median_us']:.2f} us   -> candidate is {sp:.3f}x {'FASTER' if sp and sp > 1 else 'slower'} than this baseline")
        if rep.speedup is not None:
            L.append(f"  speedup vs best baseline ({b.get('best_baseline')}): {rep.speedup:.3f}x")
        cb = rep.compiler_baseline
        if cb:
            L.append(f"  triton_msl (Triton spec compiled to Metal by a community backend, estimated): {cb['estimated_us']:.2f} us -> candidate is {cb['speedup']:.3f}x {'FASTER' if cb['speedup'] > 1 else 'slower'}")
    if rep.status == "ok" and rep.verification:
        L.append(f"\nFormal verification: {rep.verification.get('status')} (eligibility rests on safety checks and randomized differential tests)")
        scope = ((rep.correctness or {}).get("coverage") or {}).get("scope")
        if scope:
            L.append(f"Evidence: {rep.correctness['coverage'].get('kind')}: {scope}")
    if rep.roofline:
        r = rep.roofline
        if _num(r.get("achieved_gbps")) is not None and _num(r.get("peak_gbps_measured")) is not None and _num(r.get("bandwidth_fraction")) is not None:
            L.append("\nRoofline (measured peaks on this machine):")
            L.append(
                f"  bound: {r['bound']}; achieved {r['achieved_gbps']:.0f} GB/s ({r['bandwidth_fraction']:.0%} of {r['peak_gbps_measured']:.0f} GB/s peak)"
                + (f", {r['achieved_tflops']:.2f} TFLOP/s ({r['compute_fraction']:.0%} of {r['peak_tflops_measured']:.1f})" if r.get("flops") and _num(r.get("achieved_tflops")) is not None and _num(r.get("peak_tflops_measured")) is not None else "")
            )
            by_dtype = r.get("peak_tflops_by_dtype")
            if isinstance(by_dtype, dict) and len(by_dtype) > 1:
                peaks = ", ".join(f"{name} {float(value):.1f}" for name, value in by_dtype.items() if isinstance(value, (int, float)))
                if peaks:
                    L.append(f"  matrix peaks measured on this machine, by operand dtype (TFLOP/s): {peaks}. The bound above uses the task's dtype; another dtype's rate is its own measurement, not a fraction of it.")
        # 基线已经快过下界时，这句会把「离峰值还有百分之几」说成目标。数字头已经改写目标，这里不再重复。
        if r.get("time_over_lower_bound") and _baseline_beats_bound(rep) is None:
            L.append(f"  roofline lower bound {r['lower_bound_us']:.1f} us (reference algorithm); the candidate is {r['time_over_lower_bound']:.2f}x above it")
        if r.get("bound_kind") == "reference_algorithm" and isinstance(rep.compile, dict) and rep.compile.get("launch"):
            L.append("  achieved fractions use the reference algorithm's operation and byte counts")
    # 编译通过后就会带上流水线统计。hints 在每线程寄存器把可驻留线程压低，或静态 threadgroup 内存超过设备上限一半时出现。
    if rep.pipeline and all(k in rep.pipeline for k in ("max_total_threads_per_threadgroup", "launch_threads_per_threadgroup", "thread_execution_width", "static_threadgroup_bytes")):
        p = rep.pipeline
        stages = p.get("stages") if isinstance(p.get("stages"), list) else None
        if stages:
            L.append(f"\nPipeline: {len(stages)} stages, SIMD width {p['thread_execution_width']}")
            for stage in stages:
                L.append(
                    f"  {stage.get('kernel')}: grid {stage.get('grid')} threads {stage.get('threads')}, "
                    f"maxTotalThreadsPerThreadgroup={stage.get('max_total_threads_per_threadgroup')}, "
                    f"static threadgroup memory {stage.get('static_threadgroup_bytes')} B"
                )
        else:
            L.append(
                f"\nPipeline: maxTotalThreadsPerThreadgroup={p['max_total_threads_per_threadgroup']} (launching {p['launch_threads_per_threadgroup']}), "
                f"SIMD width {p['thread_execution_width']}, static threadgroup memory {p['static_threadgroup_bytes']} B"
            )
        for h in p.get("hints", []):
            L.append(f"  hint: {h}")
    return "\n".join(L)
