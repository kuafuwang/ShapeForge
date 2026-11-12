"""Validated F(4x4) Winograd consumer using ``simdgroup_float8x8``.

The candidate is kept as the exact source that passed fresh confirmation.  It uses
an independent ``ScratchSpec`` buffer for the producer's transformed filters; the
output tensor is never borrowed as workspace.  Normal inputs use the MMA path.
The consumer scans the complete 6x6 input patches and every filter tap, so IEEE
inputs take the direct fp32 fallback without depending on a poisonable flag word.

This is a paper-shape candidate fixture, not a task-independent kernel generator.
The generic, reusable part of the change is the scratch lifecycle in ``tcenv``.
"""

from __future__ import annotations

from pathlib import Path
import re


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "support"
    / "golden"
    / "metal"
    / "fixtures"
    / "conv2d_f4_mma_scratch.metal"
)
CONFIRMED_SHA256 = "5fbfea6c501f6439e5abc69a18633d1d1f6c8205759d4add39ce723c749b5be2"
RECOMPUTE_SHA256 = "6f017ede796b74544cf2f70580a4ab888c66d3284afe08774cfd150fd28093ee"
PAPER_SYMBOLS = {
    "N": 32,
    "H": 56,
    "W": 56,
    "C": 64,
    "OC": 128,
    "KH": 3,
    "KW": 3,
    "OH": 54,
    "OW": 54,
}


def metal_source(symbols: dict[str, int], tile: int = 4) -> str:
    """Return the validated paper-shape F(4x4) candidate without changing its bytes."""
    if tile != 4:
        raise ValueError("the validated MMA consumer is F(4x4), not another tile size")
    actual = {name: int(symbols[name]) for name in PAPER_SYMBOLS}
    if actual != PAPER_SYMBOLS:
        raise ValueError("the validated MMA consumer is fixed to the conv2d paper shape")
    return FIXTURE.read_text(encoding="utf-8")


def metal_source_recompute(symbols: dict[str, int]) -> str:
    """Generate the same-MMA control that recomputes U in every threadgroup.

    Only the source of U changes: the transform matrices, consumer tiling, MMA
    instructions, output epilogue, and IEEE direct fallback remain identical.
    This makes the global-reuse measurement an algorithmic A/B rather than a
    comparison with the older scalar consumer.
    """
    source = metal_source(symbols)
    helper = '''static inline float recompute_u1(uint p,uint c,uint oc,device const half* w){
  const uint pr=p/6u,pc=p-pr*6u;
  float acc=0.0f;
  for(uint i=0u;i<3u;++i){
    const float gi=GCF[pr][i];
    if(gi==0.0f)continue;
    for(uint j=0u;j<3u;++j){
      const float cf=gi*GCF[pc][j];
      if(cf==0.0f)continue;
      acc+=cf*float(w[oc*K_TOTAL+(i*KW+j)*C+c]);
    }
  }
  return acc;
}
static inline float4 recompute_u4(uint p,uint cb,uint oc0,uint lrow,uint lcp,device const half* w){
  const uint c=cb*16u+lrow;
  const uint oc=oc0+lcp*2u;
  return float4(
    recompute_u1(p,c,oc,w),
    recompute_u1(p,c,oc+1u,w),
    recompute_u1(p,c+8u,oc,w),
    recompute_u1(p,c+8u,oc+1u,w));
}
'''
    start = source.index("static inline float4 read_u4")
    end = source.index("[[max_total_threads_per_threadgroup(256)]]", start)
    source = source[:start] + helper + source[end:]
    pattern = re.compile(
        r"read_u4\(uwin,\((\d+)u\*4u\+(\d+)u\)\*2048u"
        r"\+\(oc0>>3u\)\*128u\+lrow\*16u\+lcp\*4u\)"
    )
    source, count = pattern.subn(
        lambda match: f"recompute_u4({match.group(1)}u,{match.group(2)}u,oc0,lrow,lcp,w)",
        source,
    )
    if count != 144:
        raise RuntimeError(f"expected 144 interleaved-U loads, found {count}")
    source = source.replace("device const half*w[[buffer(3)]]", "device const half*w[[buffer(1)]]")
    source = source.replace("device const float* uwin [[buffer(1)]]", "")
    source = source.replace("]],,", "]],")
    old_launch = (
        "scratch uwin fp32 36*C*OC zero\n"
        "launch conv2d_wgrid grid=288,1,1 threads=128,1,1 scratch=uwin:write\n"
        "launch conv2d_kernel grid=392,2,1 threads=256,1,1 scratch=uwin:read"
    )
    new_launch = "launch conv2d_kernel grid=392,2,1 threads=256,1,1"
    if old_launch not in source:
        raise RuntimeError("the confirmed two-stage launch block drifted")
    source = source.replace(old_launch, new_launch)
    if "uwin" in source:
        raise RuntimeError("recompute control still refers to global U")
    return source
