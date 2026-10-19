"""Does the hand-picked launch contract (tile, threads) limit what a kernel can reach?

The compilation contract fixes the grid and the threadgroup size, and the Triton spec fixes the tile. If a better tile/thread count
were available, every candidate would be capped by the contract, not by the compiler. Two measurements settle it for GEMM:

1. **Hand-written reference kernels under many contracts** (this file). One parameterized ``simdgroup_matrix`` GEMM is generated for
   each launch contract ``gemm@<BM>x<BN>x<BK>t<threads>`` and run through the full evaluation protocol (compile, ABI, safety,
   adversarial correctness, interleaved benchmark against MLX / torch-MPS). The same kernel template everywhere, so the only thing
   that changes between rows is the contract. If the best row is within a few percent of the default contract's row, the
   contract does not limit GEMM.
2. **triton-msl's own sweep** over ``num_warps`` (``tcenv.cli triton-baseline``) and, with ``--triton``, over the tile.

    uv run python -m experiments.contract_sensitivity gemm --dtype fp16 --out runs/contract_sensitivity
    uv run python -m experiments.contract_sensitivity gemm --variants 64x64x32t128,128x128x32t256   # a subset
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT = "64x64x32t128"
VARIANTS = (
    "32x32x32t128", "64x32x32t64", "32x64x32t64",
    "64x64x32t64", "64x64x32t128", "64x64x32t256",
    "64x64x16t128", "64x64x64t128",
    "128x64x32t256", "64x128x32t256", "128x128x32t256", "128x128x16t256", "128x128x32t512", "128x128x32t128",
)


def parse(variant: str) -> tuple[int, int, int, int]:
    import re

    m = re.fullmatch(r"(\d+)x(\d+)x(\d+)t(\d+)", variant)
    if not m:
        raise ValueError(variant)
    return tuple(map(int, m.groups()))  # type: ignore[return-value]


def simdgroup_layout(bm: int, bn: int, threads: int) -> tuple[int, int]:
    """(WM, WN): how the ``threads/32`` simdgroups tile the output; the most square per-simdgroup sub-tile whose sides are multiples of 8."""
    w = threads // 32
    best = None
    for wm in range(1, w + 1):
        if w % wm:
            continue
        wn = w // wm
        if bm % wm or bn % wn or (bm // wm) % 8 or (bn // wn) % 8:
            continue
        score = abs(math.log((bm / wm) / (bn / wn)))
        if best is None or score < best[0]:
            best = (score, wm, wn)
    if best is None:
        raise ValueError(f"no simdgroup layout for tile {bm}x{bn} with {threads} threads")
    return best[1], best[2]


GEMM_TEMPLATE = """\
// Hand-written reference GEMM (simdgroup_matrix), tile BLOCK_M x BLOCK_N x BLOCK_K, {W} simdgroups as {WM} x {WN}; every simdgroup owns a
// {SM} x {SN} sub-tile ({FM} x {FN} fragments of 8 x 8 with fp32 accumulators). Single-buffered; threadgroup rows padded by {PAD} elements (0 when the padded tiles would not fit in 32 KB).
constant constexpr uint WM = {WM}, WN = {WN};
constant constexpr uint SM = BLOCK_M / WM, SN = BLOCK_N / WN;
constant constexpr uint FM = SM / 8, FN = SN / 8;
constant constexpr uint LDA = BLOCK_K + {PAD}, LDB = BLOCK_N + {PAD};
constant constexpr uint NTHREADS = {THREADS};

kernel void gemm_kernel(device const {T}* A [[buffer(0)]],
                        device const {T}* B [[buffer(1)]],
                        device {T}* C [[buffer(2)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint lid [[thread_index_in_threadgroup]]) {{
    threadgroup {T} As[BLOCK_M * LDA];
    threadgroup {T} Bs[BLOCK_K * LDB];
    const uint m0 = tg.y * BLOCK_M, n0 = tg.x * BLOCK_N;
    const uint wm = sg / WN, wn = sg % WN;

    simdgroup_matrix<float, 8, 8> acc[FM][FN];
    for (uint i = 0; i < FM; ++i)
        for (uint j = 0; j < FN; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += BLOCK_K) {{
        for (uint i = lid; i < BLOCK_M * BLOCK_K / 4; i += NTHREADS) {{
            const uint r = i / (BLOCK_K / 4), c = (i % (BLOCK_K / 4)) * 4;
            *(threadgroup {T}4*)&As[r * LDA + c] = *(device const {T}4*)&A[(ulong)(m0 + r) * K + k0 + c];
        }}
        for (uint i = lid; i < BLOCK_K * BLOCK_N / 4; i += NTHREADS) {{
            const uint r = i / (BLOCK_N / 4), c = (i % (BLOCK_N / 4)) * 4;
            *(threadgroup {T}4*)&Bs[r * LDB + c] = *(device const {T}4*)&B[(ulong)(k0 + r) * N + n0 + c];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BLOCK_K; kk += 8) {{
            simdgroup_matrix<{T}, 8, 8> a[FM], b[FN];
            for (uint i = 0; i < FM; ++i) simdgroup_load(a[i], &As[(wm * SM + i * 8) * LDA + kk], LDA);
            for (uint j = 0; j < FN; ++j) simdgroup_load(b[j], &Bs[kk * LDB + wn * SN + j * 8], LDB);
            for (uint i = 0; i < FM; ++i)
                for (uint j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);  // mixed precision: in-place form
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // lane -> (row, col) of the two elements each lane holds in an 8 x 8 fragment
    const uint qid = lane / 4;
    const uint fr = (qid & 4) + ((lane / 2) % 4);
    const uint fc = (qid & 2) * 2 + (lane % 2) * 2;
    for (uint i = 0; i < FM; ++i)
        for (uint j = 0; j < FN; ++j) {{
            const ulong row = m0 + wm * SM + i * 8 + fr, col = n0 + wn * SN + j * 8 + fc;
            thread auto& e = acc[i][j].thread_elements();
            *(device {T}2*)&C[row * N + col] = {T}2({T}(e[0]), {T}(e[1]));
        }}
}}
"""


def gemm_source(variant: str, dtype: str = "fp16") -> str:
    bm, bn, bk, th = parse(variant)
    wm, wn = simdgroup_layout(bm, bn, th)
    t = {"fp16": "half", "bf16": "bfloat"}[dtype]
    pad = 8 if 4 * (bm * (bk + 8) + bk * (bn + 8)) <= 32768 else 0
    return GEMM_TEMPLATE.format(T=t, PAD=pad, THREADS=th, W=th // 32, WM=wm, WN=wn, SM=bm // wm, SN=bn // wn, FM=bm // wm // 8, FN=bn // wn // 8)


def fa_source(variant: str, dtype: str = "fp16", size: str = "paper") -> str:
    """A source-level stand-in used by the static screen. It is not a correct attention kernel."""
    import importlib

    mod = importlib.import_module("tasks.flash_attention.task")
    task = mod.build_variant(dtype, size, variant)
    constants = {name: value for name, _ty, value in task.contract.constants}
    bm, bn, threads = int(constants["BLOCK_M"]), int(constants["BLOCK_N"]), int(task.contract.threads[0])
    groups = max(1, threads // 32)
    fragments = max(1, (bm // 8) * (bn // 8) // groups)
    t = {"fp16": "half", "bf16": "bfloat"}[dtype]
    return (
        f"kernel void flash_attention_kernel(device const {t}* q [[buffer(0)]], device const {t}* k [[buffer(1)]], "
        f"device const {t}* v [[buffer(2)]], device {t}* o [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {{\n"
        f"    threadgroup {t} tile[{bm} * {bn}];\n"
        f"    simdgroup_matrix<float, 8, 8> acc[{fragments}];\n"
        f"    for (uint i = 0; i < {fragments}; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n"
        f"    simdgroup_multiply_accumulate(acc[0], acc[0], acc[0], acc[0]);\n"
        f"    o[gid.x] = {t}(acc[0].thread_elements()[0]);\n"
        f"}}\n"
    )


def static_screen_contracts(op: str, variants: list[str], dtype: str = "fp16", size: str = "paper") -> list[dict]:
    """Drop contracts that do not build, that declare 32 KiB or more of threadgroup memory, or that keep 32 or more accumulator fragments.

    No GPU. The fragment count is the source-level count from ``taic.hwfeatures.analyze_source``.
    """
    from taic.hwfeatures import analyze_source

    rows = []
    for variant in variants:
        row = {"variant": variant, "op": op, "kept": False, "reason": ""}
        try:
            if op == "gemm":
                source = gemm_source(variant, dtype)
            elif op == "flash_attention":
                source = fa_source(variant, dtype, size)
            else:
                row["reason"] = f"no static screen for {op}"
                rows.append(row)
                continue
            ev = analyze_source(source)
        except (ValueError, KeyError, ImportError) as e:
            row["reason"] = str(e)
            rows.append(row)
            continue
        tiles = (ev.get("matrix") or {}).get("acc_tiles") or 0
        if op == "gemm":
            bm, bn, _bk, th = parse(variant)
            wm, wn = simdgroup_layout(bm, bn, th)
            tiles = max(int(tiles or 0), (bm // wm // 8) * (bn // wn // 8))
        declared = (ev.get("threadgroup") or {}).get("declared_bytes") or 0
        row["acc_tiles"] = tiles
        row["threadgroup_bytes"] = declared
        if isinstance(tiles, int) and tiles >= 32:
            row["reason"] = f"{tiles} accumulator fragments"
        elif isinstance(declared, int) and declared > 32768:
            row["reason"] = f"threadgroup {declared} B"
        else:
            row["kept"] = True
            row["reason"] = "kept"
        rows.append(row)
    return rows


def run_gemm(variants: list[str], dtype: str, size: str, out: Path, reps: int) -> list[dict]:
    from tcenv.evaluate import evaluate
    from tcenv.events import install_file_log, job_state

    install_file_log()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    path = out / f"gemm_{dtype}_{size}.json"
    if path.exists():
        rows = json.loads(path.read_text())
    done = {r["variant"]: r for r in rows}
    variants = list(variants)
    for i, v in enumerate(variants):
        job_state(out / "state.json", script="contract_sensitivity", current={"variant": v, "dtype": dtype, "size": size}, queue=[{"variant": x} for x in variants[i + 1 :]], done=list(done.values()))
        if v in done and done[v].get("status") == "ok":
            continue
        name = "gemm" if v == DEFAULT else f"gemm@{v}"
        try:
            src = gemm_source(v, dtype)
            rep = evaluate(name, dtype, size, src, mode="full", reps=reps, progress_path=out / "eval_progress.json")
            row = {
                "variant": v, "task": name, "status": rep.status, "message": rep.message[:300], "abi_violations": [str(x)[:200] for x in (rep.abi or {}).get("violations", [])][:3], "candidate_us": rep.candidate_us,
                "baseline_us": rep.baseline_us, "speedup": rep.speedup, "noisy": (rep.benchmark or {}).get("noisy"),
                "cv": (rep.benchmark or {}).get("cv"), "tflops": (rep.roofline or {}).get("achieved_tflops"),
                "pipeline": {k: rep.pipeline.get(k) for k in ("max_threads", "static_threadgroup_memory")} if rep.pipeline else {},
            }
        except ValueError as e:  # the variant is not representable (tile does not divide the problem, ...)
            row = {"variant": v, "task": name, "status": "invalid", "message": str(e)}
        done[v] = row
        rows = [done[k] for k in done]
        path.write_text(json.dumps(rows, indent=1))
        cu = row.get("candidate_us")
        print(f"  {v:16s} {row['status']:10s} {('%8.1f us' % cu) if cu else '        -   '}  speedup {row.get('speedup') or float('nan'):.3f}x  {(row.get('abi_violations') or [row.get('message', '')])[0][:110]}", flush=True)
    job_state(out / "state.json", script="contract_sensitivity", current=None, queue=[], done=list(done.values()), phase="done")
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("candidate_us")]
    if not ok:
        return {"verdict": "no valid variant"}
    best = min(ok, key=lambda r: r["candidate_us"])
    base = next((r for r in ok if r["variant"] == DEFAULT), None)
    mlx = None
    if base:
        mlx = base["baseline_us"].get("mlx")
    out = {
        "best_variant": best["variant"], "best_us": best["candidate_us"],
        "default_variant": DEFAULT, "default_us": base["candidate_us"] if base else None,
        "default_over_best": (base["candidate_us"] / best["candidate_us"]) if base else None,
        "best_over_baseline": best["speedup"], "default_over_baseline": base["speedup"] if base else None, "mlx_us": mlx,
    }
    if base:
        gap = base["candidate_us"] / best["candidate_us"] - 1
        out["verdict"] = (
            f"the default contract is within {gap:.1%} of the best of {len(ok)} contracts" if gap < 0.05
            else f"another contract is {gap:.1%} faster than the default for the same kernel template: the default contract limits GEMM"
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["gemm", "report", "prescreen"])
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--size", default="paper")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default="runs/contract_sensitivity")
    ap.add_argument("--p1", default="runs/p1", help="directory of experiments.p1_runs results (FlashAttention rows of the report)")
    ap.add_argument("--top", type=int, default=2)
    a = ap.parse_args(argv)
    if a.op == "prescreen":
        rows = static_screen_contracts("gemm", a.variants.split(","), a.dtype, a.size)
        kept = [r for r in rows if r["kept"]][: a.top]
        print(json.dumps({"kept": kept, "rows": rows}, indent=1))
        return 0
    if a.op == "report":
        md = report(Path(a.out), a.dtype, a.size, Path(a.p1))
        (Path(a.out) / "REPORT.md").write_text(md + "\n")
        print(md)
        return 0
    rows = run_gemm(a.variants.split(","), a.dtype, a.size, Path(a.out), a.reps)
    s = summarize(rows)
    (Path(a.out) / f"gemm_{a.dtype}_{a.size}_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------------------------------------------------------------
# report: everything recorded about the contract, in one table per source
# ---------------------------------------------------------------------------------------------------------------------------------


def triton_rows(task_prefix: str, dtype: str = "fp16", size: str = "paper", cache: Path | None = None) -> list[dict]:
    """What triton-msl did with every contract variant of an operator that has a cached measurement."""
    cache = cache or Path(__file__).resolve().parent.parent / "runs" / "triton_baselines"
    rows = []
    for p in sorted(cache.glob(f"{task_prefix}*_{dtype}_{size}.json")):
        stem = p.name[: -len(f"_{dtype}_{size}.json")]
        if stem != task_prefix and not stem.startswith(task_prefix + "@"):
            continue
        d = json.loads(p.read_text())
        sw = d.get("sweep_median_us") or {}
        rows.append({
            "variant": stem.partition("@")[2] or "default", "status": d.get("status"),
            "ratio_to_mlx": (d.get("ratio_to") or {}).get("mlx"), "best_num_warps": d.get("best_num_warps"),
            "sweep_us": {k: round(v, 1) for k, v in sw.items()},
            "spread": (max(sw.values()) / min(sw.values()) - 1) if sw else None,
            "reason": (d.get("error") or "")[:160] if d.get("status") != "ok" else "",
        })
    return rows


def report(out: Path, dtype: str = "fp16", size: str = "paper", p1: Path | None = None) -> str:
    L = ["# Does the hand-picked launch contract limit GEMM / FlashAttention?", ""]
    gp = out / f"gemm_{dtype}_{size}.json"
    if gp.exists():
        rows = json.loads(gp.read_text())
        s = summarize(rows)
        L += ["## 1. GEMM: one simdgroup_matrix kernel template under 14 launch contracts (same protocol as every evaluation)", "",
              "| contract (BMxBNxBK, threads) | status | time us | vs MLX | note |", "| --- | --- | ---: | ---: | --- |"]
        for r in sorted(rows, key=lambda r: r.get("candidate_us") or 1e18):
            note = (r.get("abi_violations") or [r.get("message", "")])[0] if r["status"] != "ok" else ""
            L.append(f"| {r['variant']} | {r['status']} | {('%.1f' % r['candidate_us']) if r.get('candidate_us') else '-'} | {('%.3fx' % r['speedup']) if r.get('speedup') else '-'} | {note[:90]} |")
        L += ["", f"**{s.get('verdict')}** (default {s.get('default_us') and round(s['default_us'], 1)} us, MLX {s.get('mlx_us') and round(s['mlx_us'], 1)} us).", ""]
    for op, title in (("gemm", "GEMM"), ("flash_attention", "FlashAttention")):
        tr = triton_rows(op, dtype, size)
        if tr:
            L += [f"## 2{'a' if op == 'gemm' else 'b'}. {title} through triton-msl: tile and num_warps sweep", "",
                  "| contract | status | time / MLX | best num_warps | time over num_warps 1/2/4/8 (us) | spread | reason |", "| --- | --- | ---: | ---: | --- | ---: | --- |"]
            for r in tr:
                L.append(f"| {r['variant']} | {r['status']} | {('%.3f' % r['ratio_to_mlx']) if r['ratio_to_mlx'] else '-'} | {r['best_num_warps'] or '-'} | {r['sweep_us'] or '-'} | {('%.1f%%' % (100 * r['spread'])) if r['spread'] is not None else '-'} | {r['reason']} |")
            L.append("")
    if p1 and (p1 / "summary.json").exists():
        rows = [r for r in json.loads((p1 / "summary.json").read_text()) if str(r["task"]).startswith("flash_attention")]
        if rows:
            L += ["## 3. FlashAttention: the LLM search under three launch contracts (same budget, same model)", "",
                  "| run | contract | published speedup | candidate us | best baseline | spent USD |", "| --- | --- | ---: | ---: | --- | ---: |"]
            for r in rows:
                published = r.get("published_speedup") if "published_speedup" in r else None
                speed_cell = f"{round(published, 3)}x" if isinstance(published, (int, float)) and not isinstance(published, bool) else (r.get("publication") or "-")
                L.append(f"| {r['run']} | {r['task'].partition('@')[2] or '32x32t128 (default)'} | {speed_cell} | {r['candidate_us'] and round(r['candidate_us'], 1)} | {r['best_baseline']} | {r['spent_usd']:.2f} |")
            L.append("")
    return "\n".join(L)
