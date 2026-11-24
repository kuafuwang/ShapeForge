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
    """@brief 把 ``BMxBNxBKtTHREADS`` 形式的契约字符串拆成四个整数。

    @param variant 契约名，例如 ``64x64x32t128``。
    @return ``(BM, BN, BK, 线程数)``。
    @raises ValueError 字符串不符合该格式时。
    """
    import re

    m = re.fullmatch(r"(\d+)x(\d+)x(\d+)t(\d+)", variant)
    if not m:
        raise ValueError(variant)
    return tuple(map(int, m.groups()))  # type: ignore[return-value]


def simdgroup_layout(bm: int, bn: int, threads: int) -> tuple[int, int]:
    """@brief 算 ``threads/32`` 个 simdgroup 怎么切输出块。

    (WM, WN): how the ``threads/32`` simdgroups tile the output; the most square per-simdgroup sub-tile whose sides are multiples of 8.

    @param bm 输出块的行数。
    @param bn 输出块的列数。
    @param threads 线程数。
    @return ``(WM, WN)``：两个方向的 simdgroup 个数。
    @raises ValueError threads 不是 32 的倍数、或找不到整分块方式时。
    """
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
    """@brief 按一套启动契约生成手写 simdgroup_matrix GEMM 的 MSL 源码。

    @param variant 契约名，交给 ``parse``。
    @param dtype ``fp16`` 或 ``bf16``。
    @return 完整的内核源码；填充宽度按 32 KB 限制自动选。
    @raises KeyError dtype 不是支持的那两种时。
    """
    bm, bn, bk, th = parse(variant)
    wm, wn = simdgroup_layout(bm, bn, th)
    t = {"fp16": "half", "bf16": "bfloat"}[dtype]
    pad = 8 if 4 * (bm * (bk + 8) + bk * (bn + 8)) <= 32768 else 0
    return GEMM_TEMPLATE.format(T=t, PAD=pad, THREADS=th, W=th // 32, WM=wm, WN=wn, SM=bm // wm, SN=bn // wn, FM=bm // wm // 8, FN=bn // wn // 8)




def static_screen_contracts(op: str, variants: list[str], dtype: str = "fp16", size: str = "paper") -> list[dict]:
    """@brief 不跑 GPU，先刷掉建不起来、超线程组内存或累加器太多的契约。

    Drop contracts that do not build, that declare 32 KiB or more of threadgroup memory, or that keep 32 or more accumulator fragments.

    No GPU. The fragment count is the source-level count from ``taic.hwfeatures.analyze_source``.

    @param op ``gemm`` 或 ``flash_attention``。
    @param variants 要筛的契约名列表。
    @param dtype 数据类型。
    @param size 规模档位。
    @return 每个契约一行，``kept`` 标明是否保留，``reason`` 写明原因。
    """
    from taic.hwfeatures import analyze_source

    rows = []
    for variant in variants:
        row = {"variant": variant, "op": op, "kept": False, "reason": ""}
        try:
            if op == "gemm":
                source = gemm_source(variant, dtype)
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




def summarize(rows: list[dict]) -> dict:
    """@brief 比较各契约的实测耗时，判断默认契约是不是限制项。

    @param rows ``run_gemm`` 返回的结果行。
    @return 最快与默认契约的耗时、相对基线的加速比，以及一句结论。
    """
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




if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------------------------------------------------------------
# report: everything recorded about the contract, in one table per source
# ---------------------------------------------------------------------------------------------------------------------------------
