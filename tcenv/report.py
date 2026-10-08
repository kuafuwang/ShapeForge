"""Human / LLM readable rendering of an ``EvalReport``."""
# 反馈关，不再判生死。搜索把这里的短文塞进下一轮提示，让模型知道死在哪一关。
# 截断发生在本函数里：编译错误只留前 max_errors_chars 个字符，安全违例、非确定、shader validation 各只留前 6、3、4 条。
# 整份 worker 日志、心跳、每次测速的原始样本都不进这里。全文塞回去，模型读到的是日志，定位不到该改的那一关。搜索还会把本函数的输出再截到固定字数，放进下一轮提示。

from __future__ import annotations

from .evaluate import EvalReport


# 把已经写好的判决收成短文。编译错误按字符截断；安全、非确定、校验按条数截断。ABI 违例逐条全留。
# 性能和形式验证只在 status 为 ok 时附上：已经判死的候选不再带着速度结论回去。
def format_feedback(rep: EvalReport, *, max_errors_chars: int = 2500) -> str:
    L: list[str] = []
    L.append(f"STATUS: {rep.status}  (stage reached: {rep.stage})")
    if rep.message:
        L.append(rep.message)

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
        L.append("\nRoofline (measured peaks on this machine):")
        L.append(
            f"  bound: {r['bound']}; achieved {r['achieved_gbps']:.0f} GB/s ({r['bandwidth_fraction']:.0%} of {r['peak_gbps_measured']:.0f} GB/s peak)"
            + (f", {r['achieved_tflops']:.2f} TFLOP/s ({r['compute_fraction']:.0%} of {r['peak_tflops_measured']:.1f})" if r["flops"] else "")
        )
        if r.get("time_over_lower_bound"):
            L.append(f"  roofline lower bound {r['lower_bound_us']:.1f} us; the candidate is {r['time_over_lower_bound']:.2f}x above it")
    # 编译通过后就会带上流水线统计。hints 在每线程寄存器把可驻留线程压低，或静态 threadgroup 内存超过设备上限一半时出现。
    if rep.pipeline:
        p = rep.pipeline
        L.append(
            f"\nPipeline: maxTotalThreadsPerThreadgroup={p['max_total_threads_per_threadgroup']} (launching {p['launch_threads_per_threadgroup']}), "
            f"SIMD width {p['thread_execution_width']}, static threadgroup memory {p['static_threadgroup_bytes']} B"
        )
        for h in p.get("hints", []):
            L.append(f"  hint: {h}")
    return "\n".join(L)
