"""Launch-contract variants (``gemm@<BM>x<BN>x<BK>t<threads>``, ``flash_attention@<BM>x<BN>t<threads>``): the same operator under another
tile / thread count, used to test whether the hand-picked contract limits what a kernel can reach."""

from __future__ import annotations

import pytest

from tasks import get_task
from tcenv.evaluate import evaluate


def test_default_variant_is_the_default_contract():
    d, v = get_task("gemm", "fp16", "small"), get_task("gemm@64x64x32t128", "fp16", "small")
    assert d.contract.grid == v.contract.grid and d.contract.threads == v.contract.threads
    assert d.contract.constants == v.contract.constants
    assert d.naive_msl == v.naive_msl
    f, w = get_task("flash_attention", "fp16", "small"), get_task("flash_attention@32x32t128", "fp16", "small")
    assert f.contract.grid == w.contract.grid and f.contract.threads == w.contract.threads and f.naive_msl == w.naive_msl


def test_a_variant_changes_grid_threads_constants_and_specs_consistently():
    t = get_task("gemm@128x64x32t256", "fp16", "paper")
    assert t.name == "gemm@128x64x32t256" and t.contract.kernel_name == "gemm_kernel"
    assert t.contract.threads == (256, 1, 1) and t.contract.grid == (2048 // 64, 2048 // 128, 1)
    consts = {n: v for n, _, v in t.contract.constants}
    assert (consts["BLOCK_M"], consts["BLOCK_N"], consts["BLOCK_K"]) == (128, 64, 32)
    assert "BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 32" in t.triton_source and "num_warps = 8" in t.triton_source
    assert "{{" not in t.triton_source and "{{" not in t.naive_msl
    a = get_task("flash_attention@64x32t256", "fp16", "paper")
    assert a.contract.grid == (1024 // 64, 16, 1) and a.contract.threads == (256, 1, 1)
    assert "BLOCK_M = 64, BLOCK_N = 32" in a.triton_source and "num_warps = 8" in a.triton_source
    assert "{{" not in a.naive_msl and "simd_shuffle_xor(s, 2)" in a.naive_msl  # 4 threads per row


@pytest.mark.parametrize("name", [
    "gemm@100x64x32t128", "gemm@64x64x32t100", "gemm@64x64x32t96", "gemm@64x64", "gemm@axb",
    "flash_attention@32x32t100", "flash_attention@32x32t96", "flash_attention@48x32t96", "flash_attention@32x32",
    "relu@1x1x1t32", "gemm_silu@64x64x32t128",
])
def test_unrepresentable_variants_are_refused(name):
    with pytest.raises((ValueError, KeyError)):
        get_task(name, "fp16", "small")


@pytest.mark.parametrize("name", ["gemm@32x32x32t128", "gemm@128x64x32t256", "gemm@64x64x16t64", "flash_attention@64x32t256", "flash_attention@16x32t64", "flash_attention@32x16t128", "flash_attention@32x32t32"])
def test_the_naive_kernel_is_correct_under_every_variant_contract(name):
    rep = evaluate(name, "fp16", "small", get_task(name, "fp16", "small").naive_msl, mode="quick")
    assert rep.status == "ok", (name, rep.message)


def test_the_hand_written_simdgroup_gemm_is_correct_and_fits_every_listed_contract():
    from experiments.contract_sensitivity import VARIANTS, gemm_source, simdgroup_layout

    for v in ("32x32x32t128", "128x64x32t256", "64x64x32t64"):
        rep = evaluate(f"gemm@{v}", "fp16", "small", gemm_source(v), mode="quick")
        assert rep.status == "ok", (v, rep.message)
    for v in VARIANTS:  # every listed variant has a valid simdgroup layout whose sub-tiles are multiples of 8
        from experiments.contract_sensitivity import parse

        bm, bn, _, th = parse(v)
        wm, wn = simdgroup_layout(bm, bn, th)
        assert wm * wn == th // 32 and (bm // wm) % 8 == 0 and (bn // wn) % 8 == 0


def test_task_reference_sources_keep_the_metal_alias():
    from tasks import get_task

    task = get_task("relu", "fp16", "small")
    assert task.reference_for("metal") == task.naive_msl
    assert task.reference_sources["metal"] == task.naive_msl
    assert task.reference_for("other") == ""
