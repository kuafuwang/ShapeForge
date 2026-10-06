"""Human / LLM readable rendering of an ``EvalReport``."""

from __future__ import annotations

from .evaluate import EvalReport


def format_feedback(rep: EvalReport, *, max_errors_chars: int = 2500) -> str:
    L: list[str] = []
    L.append(f"STATUS: {rep.status}  (stage reached: {rep.stage})")
    if rep.message:
        L.append(rep.message)

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
    if rep.status == "ok" and rep.benchmark:
        b = rep.benchmark
        L.append("\nPerformance (interleaved benchmark, median per launch):")
        cand = b["candidate"]
        gpu = b.get("candidate_gpu_us_median")
        L.append(f"  candidate : {cand['median_us']:.2f} us   (cv {cand['cv']:.1%}{', GPU-timestamp %.2f us' % gpu if gpu else ''})")
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
    if rep.roofline:
        r = rep.roofline
        L.append("\nRoofline (measured peaks on this machine):")
        L.append(
            f"  bound: {r['bound']}; achieved {r['achieved_gbps']:.0f} GB/s ({r['bandwidth_fraction']:.0%} of {r['peak_gbps_measured']:.0f} GB/s peak)"
            + (f", {r['achieved_tflops']:.2f} TFLOP/s ({r['compute_fraction']:.0%} of {r['peak_tflops_measured']:.1f})" if r["flops"] else "")
        )
        if r.get("time_over_lower_bound"):
            L.append(f"  roofline lower bound {r['lower_bound_us']:.1f} us; the candidate is {r['time_over_lower_bound']:.2f}x above it")
    if rep.pipeline:
        p = rep.pipeline
        L.append(
            f"\nPipeline: maxTotalThreadsPerThreadgroup={p['max_total_threads_per_threadgroup']} (launching {p['launch_threads_per_threadgroup']}), "
            f"SIMD width {p['thread_execution_width']}, static threadgroup memory {p['static_threadgroup_bytes']} B"
        )
        for h in p.get("hints", []):
            L.append(f"  hint: {h}")
    return "\n".join(L)
