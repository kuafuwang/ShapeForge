import torch
import torch.nn.functional as F

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

KH = KW = 3
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
# (N, H, W, C, OC)
SHAPES = {"paper": (32, 56, 56, 64, 128), "small": (4, 18, 18, 32, 64)}
MAGS = {"normal": 1.0, "wide_range": 0.1}


def _mlx(x, w):
    import mlx.core as mx

    return mx.conv2d(x, w)


def _torch(x, w):
    return F.conv2d(x.permute(0, 3, 1, 2), w.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


def build(dtype: str, size: str) -> Task:
    check_dtype(dtype)
    N, H, W, C, OC = SHAPES[size]
    OH, OW = H - KH + 1, W - KW + 1
    M_TOTAL, K_TOTAL = N * OH * OW, KH * KW * C
    assert M_TOTAL % BLOCK_M == 0 and OC % BLOCK_N == 0 and C % BLOCK_K == 0
    contract = Contract(
        kernel_name="conv2d_kernel",
        grid=(OC // BLOCK_N, M_TOTAL // BLOCK_M, 1),
        threads=(128, 1, 1),
        args=(
            AbiArg("x", 0, "input", dtype, (N, H, W, C), doc="NHWC"),
            AbiArg("w", 1, "input", dtype, (OC, KH, KW, C), doc="[OC, KH, KW, C]"),
            AbiArg("y", 2, "output", dtype, (N, OH, OW, OC), doc="NHWC, stride 1, no padding, no bias"),
        ),
        constants=(
            ("N", "uint", N), ("H", "uint", H), ("W", "uint", W), ("C", "uint", C), ("OC", "uint", OC),
            ("KH", "uint", KH), ("KW", "uint", KW), ("OH", "uint", OH), ("OW", "uint", OW),
            ("M_TOTAL", "uint", M_TOTAL), ("K_TOTAL", "uint", K_TOTAL),
            ("BLOCK_M", "uint", BLOCK_M), ("BLOCK_N", "uint", BLOCK_N), ("BLOCK_K", "uint", BLOCK_K),
        ),
    )

    def gen_inputs(case, seed):
        mag = MAGS.get(case, 1.0)
        return {
            "x": sample(case, (N, H, W, C), seed, dtype, mag=mag, large=20.0, spike=5.0),
            "w": sample(case, (OC, KH, KW, C), seed + 31, dtype, mag=mag, large=20.0, spike=5.0),
        }

    def reference(inp):
        if not (torch.isfinite(inp["x"]).all() and torch.isfinite(inp["w"]).all()):
            # CPU conv2d may use a Winograd transform, which smears one Inf over a whole tile as NaN (Inf * 0); IEEE semantics
            # are those of the sum of products, so for inputs with NaN/Inf take the same fp32 sum tap by tap.
            x, w = inp["x"].float(), inp["w"].float()
            y = torch.zeros((N, OH, OW, OC), dtype=torch.float32)
            for kh in range(KH):
                for kw in range(KW):
                    y += x[:, kh:kh + OH, kw:kw + OW, :] @ w[:, kh, kw, :].T
            return {"y": y.double()}
        x = inp["x"].permute(0, 3, 1, 2).contiguous().float()
        w = inp["w"].permute(0, 3, 1, 2).contiguous().float()
        return {"y": F.conv2d(x, w).permute(0, 2, 3, 1).contiguous().double()}

    rtol, atol_rel = default_tolerances(dtype)
    return Task(
        name="conv2d", dtype=dtype, size=size, contract=contract,
        triton_source=load_text("conv2d", "kernel_triton.py", dtype, {"N": str(N), "H": str(H), "W": str(W), "C": str(C), "OC": str(OC), "OH": str(OH), "OW": str(OW), "M_TOTAL": str(M_TOTAL), "K_TOTAL": str(K_TOTAL)}),
        description="3x3 convolution, stride 1, no padding, no bias, NHWC activations and [OC,KH,KW,C] weights; fp32 accumulation, result rounded once.",
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        meta={"ieee_max": {"x": max(8, M_TOTAL // 100), "w": 4},  # a special in x poisons 9 pixels, one in w a whole output channel; keep most outputs finite
              "triton_baseline_variant": {"source": load_text("conv2d", "kernel_triton_msl.py", dtype, {"N": str(N), "H": str(H), "W": str(W), "C": str(C), "OC": str(OC), "OH": str(OH), "OW": str(OW), "M_TOTAL": str(M_TOTAL), "K_TOTAL": str(K_TOTAL)}), "note": "explicit im2col gather kernel + weight transpose + plain GEMM (the backend refuses the gathered implicit-GEMM operand)"}},
        flops=2.0 * M_TOTAL * OC * K_TOTAL,
        min_bytes=float((N * H * W * C + OC * KH * KW * C + N * OH * OW * OC) * 2),
        rtol=rtol, atol_rel=atol_rel,
        naive_msl=load_text("conv2d", "naive.metal", dtype),
    )
