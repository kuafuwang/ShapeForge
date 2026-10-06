"""Hardware-feature detector (``experiments/hw_features.py``): synthetic MSL snippets, the generated reference GEMM, verdict logic and run loading.

No GPU: the compiled-evidence tests only call the Metal compiler front end (``xcrun metal -S -emit-llvm``) and are skipped when the toolchain
component is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import hw_features as H

CONSTS = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "M": 2048, "N": 2048, "K": 2048}


def feats(src: str, **kw) -> dict:
    return H.analyze_source(src, consts=kw.pop("consts", CONSTS), **kw)


def present(src: str, task: str | None = None) -> set[str]:
    rep = H.build_report(src, label="t", task=task, compile_evidence=False)
    return {f for f, v in rep.features.items() if v["present"]}


# ---------------------------------------------------------------------------------------------------------------------------------------
# synthetic kernels
# ---------------------------------------------------------------------------------------------------------------------------------------
MATRIX = """
constant constexpr uint FM = 2, FN = 2;
kernel void gemm_kernel(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]], device half* C [[buffer(2)]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half As[32 * (32 + 8)];
    threadgroup half Bs[32 * (64 + 8)];
    simdgroup_float8x8 acc[FM][FN];
    for (uint i = 0; i < FM; ++i)
        for (uint j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0);
    for (uint k = 0; k < 64; k += 32) {
        *reinterpret_cast<threadgroup uint4*>(&As[lid * 8]) = *reinterpret_cast<device const uint4*>(A + k + lid * 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 a[FM], b[FN];
        for (uint i = 0; i < FM; ++i) simdgroup_load(a[i], &As[i * 8 * 40], 40);
        for (uint j = 0; j < FN; ++j) simdgroup_load(b[j], &Bs[j * 8], 72);
        for (uint i = 0; i < FM; ++i)
            for (uint j = 0; j < FN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    thread auto& e = acc[0][0].thread_elements();
    C[lane] = half(e[0]);
}
"""

SCALAR = """
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    y[i] = half(float(x[i]) * 2.0f);
}
"""

COMMENT_ONLY = """
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    // use simdgroup_multiply_accumulate(c, a, b, c) with simdgroup_float8x8, simd_sum, threadgroup_barrier, half4 loads, #pragma unroll
    /* threadgroup half buf[4096]; simdgroup_load(a, buf, 8); exp2(m_i - m_new) */
    y[i] = x[i];
}
"""

STRING_ONLY = """
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    const constant char* s = "simdgroup_multiply_accumulate(c, a, b, c); simd_sum(v); threadgroup half buf[4096]; // not a comment";
    y[i] = x[i];
}
"""

VEC = """
kernel void k(device const half4* x4 [[buffer(0)]], device const half* x [[buffer(1)]], device half* y [[buffer(2)]], device float4* z [[buffer(3)]],
              uint i [[thread_position_in_grid]]) {
    half4 a = x4[i];
    uint4 b = *reinterpret_cast<device const uint4*>(x + i * 8);
    *reinterpret_cast<device uint2*>(y + i * 4) = as_type<uint2>(a);
    z[i] = float4(a) + float4(as_type<half4>(b.xy));
}
"""

CCAST = """
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    half4 v = *(device const half4*)&x[i * 4];
    *(device half4*)&y[i * 4] = v * 2.0h;
}
"""

DB_TERNARY = """
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup half Ka[256];
    threadgroup half Kb[256];
    for (uint it = 0; it < 8; ++it) {
        threadgroup half* cur = (it & 1u) ? Ka : Kb;
        cur[lid] = x[it * 256 + lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        y[lid] = cur[(lid + 1) % 256];
    }
}
"""

DB_STAGE_DIM = """
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup half buf[2][256];
    for (uint it = 0; it < 8; ++it) {
        buf[it & 1u][lid] = x[it * 256 + lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        y[lid] = buf[it & 1u][(lid + 1) % 256];
    }
}
"""

TWO_OPERAND_TILES = """
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup half As[256];
    threadgroup half Bs[256];
    As[lid] = x[lid]; Bs[lid] = x[256 + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    y[lid] = As[(lid + 1) % 256] * Bs[lid];
}
"""

ONLINE = """
kernel void softmax_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint lid [[thread_index_in_threadgroup]]) {
    float m_i = -INFINITY, l_i = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        float v = x[lid * 8 + j];
        float m_new = max(m_i, v);
        float alpha = exp2(m_i - m_new);
        l_i = l_i * alpha + exp2(v - m_new);
        m_i = m_new;
    }
    y[lid] = l_i;
}
"""

TWO_PASS = """
kernel void softmax_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint lid [[thread_index_in_threadgroup]]) {
    float m = -INFINITY;
    for (uint j = 0; j < 8; ++j) { float v = x[lid * 8 + j]; m = max(m, v); }
    float s = 0.0f;
    for (uint j = 0; j < 8; ++j) { float v = x[lid * 8 + j]; s += exp2(v - m); }
    y[lid] = s;
}
"""

PIPELINED = """
inline void ld(device const half* A, uint k0, uint lid, thread uint4& x) { x = *reinterpret_cast<device const uint4*>(A + k0 + lid * 8); }
inline void st(threadgroup half* S, uint lid, uint4 x) { *reinterpret_cast<threadgroup uint4*>(S + lid * 8) = x; }
kernel void gemm_kernel(device const half* A [[buffer(0)]], device half* C [[buffer(2)]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup half S[1024];
    simdgroup_float8x8 acc = simdgroup_float8x8(0);
    uint4 pa;
    ld(A, 0, lid, pa);
    for (uint k = 0; k < 64; ++k) {
        st(S, lid, pa);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ld(A, k * 32, lid, pa);                      // next tile's loads are in flight while the MMAs run
        simdgroup_half8x8 a, b;
        simdgroup_load(a, S, 8);
        simdgroup_load(b, S + 64, 8);
        simdgroup_multiply_accumulate(acc, a, b, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"""

SERIAL = PIPELINED.replace("""        st(S, lid, pa);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ld(A, k * 32, lid, pa);                      // next tile's loads are in flight while the MMAs run
""", """        ld(A, k * 32, lid, pa);
        st(S, lid, pa);
        threadgroup_barrier(mem_flags::mem_threadgroup);
""").replace("    ld(A, 0, lid, pa);\n", "")

BIT_DECODE = """
kernel void k(device const uint* bits [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    uint w = bits[i];
    half s = ((w >> 3u) & 1u) ? 1.0h : -1.0h;
    y[i] = s;
}
"""


# ---------------------------------------------------------------------------------------------------------------------------------------
# lexical / constants
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_strip_comments_and_strings_keeps_layout():
    src = 'int a; // c1\n/* c2\n c3 */ const char* s = "// not a comment /* x */";\nint b;\n'
    out = H.strip_comments_and_strings(src)
    assert len(out) == len(src) and out.count("\n") == src.count("\n")
    assert "c1" not in out and "c2" not in out and "c3" not in out and "not a comment" not in out
    assert "int a;" in out and "int b;" in out and '"' in out


def test_eval_expr_and_constants():
    c = {"BLOCK_M": 64, "BLOCK_K": 32}
    assert H.eval_expr("BLOCK_M * (BLOCK_K + 8)", c) == 2560
    assert H.eval_expr("(uint)BLOCK_M / 8u", c) == 8
    assert H.eval_expr("1u << 4", c) == 16
    assert H.eval_expr("sizeof(half) * BLOCK_K", c) == 64
    assert H.eval_expr("UNKNOWN * 2", c) is None
    assert H.eval_expr("a ? b : c", c) is None
    code = "constant constexpr uint WM = 2, WN = 2;\nconstant constexpr uint SM = BLOCK_M / WM;\nconstexpr uint LDA = BLOCK_K + 8;\n#define TILE 16\n"
    got = H.collect_constants(code, c)
    assert got["SM"] == 32 and got["LDA"] == 40 and got["WN"] == 2 and got["TILE"] == 16


def test_expand_calls_substitutes_arguments():
    funcs = {f.name: f for f in H.parse_functions("inline void put(threadgroup half* d, uint i, uint4 v) { d[i] = v.x; }\nkernel void k() { }")}
    out = H.expand_calls("put(buf, lid * 2, pa0);", {"put": funcs["put"]})
    assert "buf[(lid * 2)] = pa0.x" in out.replace("  ", " ")


# ---------------------------------------------------------------------------------------------------------------------------------------
# detection and the negative cases
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_matrix_kernel_features():
    e = feats(MATRIX, kernel_name="gemm_kernel")
    m = e["matrix"]
    assert m["present"] and m["mma_sites"] == 1
    assert m["acc_tiles"] == 4 and m["acc_dims"] == [2, 2] and m["acc_types"] == ["float"] and m["operand_types"] == ["half"]
    assert m["loads_from"] == ["threadgroup"]
    assert e["threadgroup"]["staging"] and e["threadgroup"]["declared_bytes"] == 2 * (32 * 40 + 32 * 72)
    assert e["threadgroup"]["padded"]
    assert e["sync"]["threadgroup_barrier_sites"] == 2 and e["sync"]["main_loop_barrier_sites"] == 2
    assert e["memory"]["device_load"]["max_bytes"] == 16
    p = present(MATRIX, "gemm")
    assert {"matrix_unit", "float_accumulate", "register_blocking", "tg_staging", "tg_padding", "vector_access", "simdgroup_tiling"} <= p
    assert not {"online_softmax", "pipelining", "double_buffering", "simd_reduce"} & p


def test_scalar_kernel_has_no_hardware_features():
    p = present(SCALAR)
    assert not (p - {"single_pass", "few_barriers"})
    e = feats(SCALAR)
    assert not e["matrix"]["present"] and e["memory"]["device_load"]["max_bytes"] == 2


@pytest.mark.parametrize("src", [COMMENT_ONLY, STRING_ONLY])
def test_mentions_in_comments_and_strings_do_not_count(src):
    p = present(src)
    assert not (p - {"single_pass", "few_barriers"}), p
    e = feats(src)
    assert not e["matrix"]["present"] and not e["simd"]["ops"] and not e["threadgroup"]["arrays"] and e["unroll"]["pragmas"] == 0
    assert not e["online_softmax"]["present"]


def test_vector_widths_loads_and_stores_are_told_apart():
    m = feats(VEC)["memory"]
    assert m["device_load"]["widths"].get("8") and m["device_load"]["widths"].get("16")
    assert m["device_store"]["widths"].get("8") and m["device_store"]["widths"].get("16")
    assert m["device_load"]["max_bytes"] == 16 and m["device_store"]["max_bytes"] == 16
    assert "vector_access" in present(VEC)


def test_c_style_casts_are_vector_accesses():
    m = feats(CCAST)["memory"]
    assert m["device_load"]["max_bytes"] == 8 and m["device_store"]["max_bytes"] == 8


def test_parenthesised_dereference_store_is_a_store():
    src = """
    kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        const uint4 v = *(reinterpret_cast<device const uint4*>(x + i * 8));
        *(reinterpret_cast<device uint4*>(y + i * 8)) = v;
    }"""
    m = feats(src)["memory"]
    assert m["device_load"]["widths"] == {"16": 1} and m["device_store"]["widths"] == {"16": 1}


def test_typed_pointers_are_scoped_per_function():
    src = """
    inline void a(device const half* x, thread uint4& r) { const device uint4* p = reinterpret_cast<const device uint4*>(x); r = p[0]; }
    inline void b(threadgroup half* t, uint4 r) { threadgroup uint4* p = reinterpret_cast<threadgroup uint4*>(t); p[0] = r; }
    kernel void k(device const half* x [[buffer(0)]]) { threadgroup half t[64]; uint4 r; a(x, r); b(t, r); }"""
    m = feats(src)["memory"]
    assert m["device_load"]["max_bytes"] == 16 and m["threadgroup_store"]["max_bytes"] == 16 and m["device_store"]["max_bytes"] == 0


def test_macro_bodies_are_analysed():
    src = """
    #define MMA(ACC, A, B) simdgroup_multiply_accumulate(ACC, A, B, ACC)
    kernel void k(device half* y [[buffer(0)]]) {
        simdgroup_float8x8 c = simdgroup_float8x8(0); simdgroup_half8x8 a, b;
        MMA(c, a, b);
    }"""
    assert feats(src)["matrix"]["mma_sites"] == 1


def test_double_buffering_patterns():
    assert feats(DB_TERNARY)["threadgroup"]["double_buffer"] == "explicit"
    assert feats(DB_STAGE_DIM)["threadgroup"]["double_buffer"] == "explicit"
    # A and B operand tiles of equal size are two buffers, not two stages
    assert feats(TWO_OPERAND_TILES)["threadgroup"]["double_buffer"] == "no"
    assert "double_buffering" not in present(TWO_OPERAND_TILES)
    assert "pipelining" in present(DB_TERNARY)


def test_online_softmax_versus_two_pass():
    assert feats(ONLINE)["online_softmax"]["present"]
    assert not feats(TWO_PASS)["online_softmax"]["present"]
    assert feats(TWO_PASS)["global_read_passes"] == {"x": 2}
    assert feats(ONLINE)["global_read_passes"] == {"x": 1}
    assert "single_pass" in present(ONLINE) and "single_pass" not in present(TWO_PASS)


def test_register_prefetch_requires_loads_in_flight_across_the_barrier():
    assert feats(PIPELINED, kernel_name="gemm_kernel")["threadgroup"]["register_prefetch"]
    assert "pipelining" in present(PIPELINED)
    assert not feats(SERIAL, kernel_name="gemm_kernel")["threadgroup"]["register_prefetch"]
    assert "pipelining" not in present(SERIAL)


def test_bit_packing_needs_bit_ops_on_loaded_data_not_index_math():
    assert feats(BIT_DECODE)["bit_packing"]["present"]
    assert not feats(VEC)["bit_packing"]["present"]
    assert not feats(SCALAR)["bit_packing"]["present"]
    assert not feats(PIPELINED, kernel_name="gemm_kernel")["bit_packing"]["present"]


def test_simd_and_math_detection():
    src = """
    kernel void k(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint lid [[thread_index_in_threadgroup]]) {
        float v = simd_sum(x[lid]) + simd_max(x[lid]) + simd_shuffle_xor(x[lid], 1);
        y[lid] = fast::exp(v) + exp2(v * 1.4426950408889634f);
    }"""
    e = feats(src)
    assert set(e["simd"]["reduce"]) == {"simd_sum", "simd_max"} and e["simd"]["shuffle"] == ["simd_shuffle_xor"]
    assert "fast::exp" in e["math"]["fast_namespace"] and e["math"]["exp2_with_log2e_fold"]
    p = present(src)
    assert {"simd_reduce", "simd_shuffle", "fast_math"} <= p


def test_tensor_api_and_async_copy_are_recognized():
    src = "#include <metal_tensor>\nkernel void k(device half* y [[buffer(0)]]) { simdgroup_event e; }\n"
    e = feats(src)
    assert e["matrix"]["tensor_api"] and e["async_copy"]


def test_unreachable_helper_is_not_counted():
    src = "inline float dead(float v) { return simd_sum(v); }\nkernel void k(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] = 1.0f; }\n"
    e = feats(src)
    assert e["helpers_unused"] == ["dead"] and not e["simd"]["ops"]


# ---------------------------------------------------------------------------------------------------------------------------------------
# the generated reference GEMMs and the naive kernels of the repository
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_reference_gemm_is_detected_as_matrix_unit_user():
    from experiments.contract_sensitivity import gemm_source

    src = gemm_source("64x64x32t128", "fp16")
    e = feats(src, consts={**CONSTS})
    m = e["matrix"]
    assert m["present"] and m["acc_types"] == ["float"] and m["operand_types"] == ["half"]
    assert m["acc_tiles"] == 16 and m["acc_dims"] == [4, 4]
    assert e["threadgroup"]["staging"] and e["threadgroup"]["padded"] and e["threadgroup"]["double_buffer"] == "no"
    assert e["memory"]["device_load"]["max_bytes"] == 8
    rep = H.build_report(src, label="ref", task="gemm", dtype="fp16", compile_evidence=False)
    assert rep.features["matrix_unit"]["present"] and rep.features["register_blocking"]["present"] and rep.features["tg_padding"]["present"]
    assert "matrix unit: yes" in rep.summary and "16 tile" in rep.summary


def test_naive_kernels_do_not_claim_a_matrix_unit():
    from tasks import get_task, list_tasks

    for name in list_tasks(True):
        t = get_task(name, "fp16", "paper")
        rep = H.build_report(t.naive_msl, label=f"naive/{name}", task=name, dtype="fp16", compile_evidence=False)
        assert not rep.features["matrix_unit"]["present"], name
        assert not rep.features["vector_access"]["present"] or name == "bitdelta_gemm", (name, rep.summary)


# ---------------------------------------------------------------------------------------------------------------------------------------
# compiled evidence (AIR text, no GPU); skipped without the Metal toolchain
# ---------------------------------------------------------------------------------------------------------------------------------------
needs_metal = pytest.mark.skipif(H.metal_toolchain() is None, reason="Metal toolchain (xcrun metal) not available")


@needs_metal
def test_compiled_evidence_confirms_matrix_intrinsics_and_widths(tmp_path):
    from experiments.contract_sensitivity import gemm_source
    from tasks import get_task

    t = get_task("gemm", "fp16", "paper")
    comp = H.compile_air(t.contract.render_header() + gemm_source("64x64x32t128", "fp16"), cache_dir=tmp_path)
    assert comp["ok"], comp.get("error")
    assert comp["matrix"]["multiply_accumulate"] >= 1 and comp["matrix"]["acc_types"] == ["f32"] and comp["matrix"]["operand_types"] == ["f16"]
    assert comp["matrix"]["load_spaces"] == ["threadgroup"]
    assert comp["fragment_tiles_f32"] == 16
    assert comp["barriers"]["workgroup"] >= 2 and comp["tg_global_bytes"] > 0
    assert any(int(b) >= 8 for b in comp["device_load_bytes"])
    again = H.compile_air(t.contract.render_header() + gemm_source("64x64x32t128", "fp16"), cache_dir=tmp_path)  # served from the cache
    assert again == comp


@needs_metal
def test_compiled_evidence_of_scalar_and_vector_kernels(tmp_path):
    from tasks import get_task

    t = get_task("relu", "fp16", "paper")
    head = t.contract.render_header()
    scalar = H.compile_air(head + t.naive_msl, cache_dir=tmp_path)
    assert scalar["ok"] and scalar["matrix"]["multiply_accumulate"] == 0 and not scalar["simd"]
    assert max((int(b) for b in scalar["device_load_bytes"]), default=0) <= 2
    vec = H.compile_air(head + "kernel void relu_kernel(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
                               "    *reinterpret_cast<device uint4*>(y + i * 8) = *reinterpret_cast<device const uint4*>(x + i * 8); }\n", cache_dir=tmp_path)
    assert vec["ok"] and "16" in vec["device_load_bytes"] and "16" in vec["device_store_bytes"]


@needs_metal
def test_compile_error_is_reported_not_raised(tmp_path):
    comp = H.compile_air("kernel void k( { }", cache_dir=tmp_path)
    assert not comp["ok"] and "error" in comp["error"]


@needs_metal
def test_full_report_has_all_three_evidence_levels(tmp_path):
    from experiments.contract_sensitivity import gemm_source

    measured = {"status": "ok", "candidate_us": 1361.0, "achieved_tflops": 12.6, "peak_tflops_measured": 14.77, "speedup": 0.87, "candidate_us_": 1}
    rep = H.build_report(gemm_source("64x64x32t128", "fp16"), label="ref", task="gemm", dtype="fp16", measured=measured, ceilings=H.Ceilings(peak_tflops=14.77), cache_dir=tmp_path)
    mu = rep.features["matrix_unit"]
    assert mu["source"] is True and mu["compiled"] is True
    assert mu["verdict"] == "used and effective" and mu["attribution"] == "primary"
    assert rep.performance_test["result"] == "above" and rep.compiled["ok"]
    data = json.loads(rep.to_json())
    assert data["summary"] == rep.summary and set(data["features"]) == set(H.detect_features(rep.source, rep.compiled))


# ---------------------------------------------------------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------------------------------------------------------
def _meas(**kw):
    return {"status": "ok", "candidate_us": 100.0, **kw}


def test_perf_test_uses_the_no_feature_ceiling():
    c = H.Ceilings(naive_tflops=5.5, peak_tflops=14.8, naive_gbps=440.0, peak_gbps=500.0)
    assert H.perf_test("gemm", _meas(achieved_tflops=14.0), c)["result"] == "above"
    assert H.perf_test("gemm", _meas(achieved_tflops=5.6), c)["result"] == "at"
    assert H.perf_test("relu", _meas(achieved_gbps=441.0), c)["result"] == "at"
    assert H.perf_test("relu", _meas(achieved_gbps=470.0), c)["result"] == "above"
    assert H.perf_test("gemm", {"status": "incorrect"}, c)["result"] == "unmeasured"
    fallback = H.Ceilings(peak_tflops=14.8)
    r = H.perf_test("gemm", _meas(achieved_tflops=14.0), fallback)
    assert r["result"] == "above" and abs(r["ceiling"] - H.SCALAR_OVER_MATRIX * 14.8) < 0.01 and "matrix peak" in r["ceiling_basis"]
    assert H.perf_test("flash_attention@64x32t256", _meas(achieved_tflops=14.0), fallback)["result"] == "above"  # variants use the base task's family


def test_judge_assigns_the_four_verdicts():
    src = H.analyze_source(SCALAR)
    f = H.detect_features(src, None)
    f, missing = H.judge("gemm", f, {"result": "unknown"})
    assert f["matrix_unit"]["verdict"] == "expected but missing" and "matrix_unit" in missing and missing.index("matrix_unit") < missing.index("tg_padding")
    assert f["online_softmax"]["verdict"] == "not applicable"
    # present + above -> effective; present + at -> not effective; no benchmark -> not measured
    for pt, want in (({"result": "above"}, "used and effective"), ({"result": "at"}, "present but not effective"), ({"result": "unmeasured"}, "present, not measured")):
        g = H.detect_features(H.analyze_source(MATRIX, consts=CONSTS), None)
        g, _ = H.judge("gemm", g, pt)
        assert g["matrix_unit"]["verdict"] == want, pt
        assert g["matrix_unit"]["attribution"] == "primary" and g["tg_padding"]["attribution"] == "joint"
    # the elementwise family: vector accesses are the primary feature
    g = H.detect_features(H.analyze_source(VEC), None)
    g, _ = H.judge("relu", g, {"result": "at"})
    assert g["vector_access"]["verdict"] == "present but not effective" and g["vector_access"]["attribution"] == "primary"


def test_docs_table3_matches_the_expectations_in_code():
    doc = (Path(__file__).resolve().parent.parent / "docs" / "hardware_features.md").read_text().splitlines()
    start = next(i for i, ln in enumerate(doc) if ln.startswith("| task | matrix unit |"))
    header = [c.strip().replace(" ", "_") for c in doc[start].strip("|").split("|")][1:]
    sym = {"R": "required", "r": "recommended", "": None}
    seen = set()
    for ln in doc[start + 2 :]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        task, marks = cells[0], cells[1:]
        want = {f: sym[m] for f, m in zip(header, marks) if sym[m]}
        assert want == H.EXPECTED[task], (task, want, H.EXPECTED[task])
        seen.add(task)
    assert seen == set(H.EXPECTED)


def test_every_task_has_expectations_and_a_family():
    from tasks import list_tasks

    for name in list_tasks(True):
        assert name in H.EXPECTED and name in H.TASK_FAMILY, name
        assert set(H.EXPECTED[name]) <= set(H.FEATURES), (name, set(H.EXPECTED[name]) - set(H.FEATURES))
        assert set(H.EXPECTED[name].values()) <= {"required", "recommended"}
    assert "matrix_unit" in H.EXPECTED["gemm"] and "matrix_unit" not in H.EXPECTED["softmax"]
    assert H.EXPECTED["flash_attention"]["online_softmax"] == "required" and H.EXPECTED["gemm_silu"]["epilogue_on_fragments"] == "required"


# ---------------------------------------------------------------------------------------------------------------------------------------
# run loading and the CLI on a fake runs directory
# ---------------------------------------------------------------------------------------------------------------------------------------
def make_run(root: Path, name: str, *, finished: bool) -> Path:
    run = root / name
    for i, (label, src, sp, status) in enumerate([("init", SCALAR, 0.5, "ok"), ("r1.p0", MATRIX, 0.9, "ok"), ("r1.p1", SCALAR, None, "incorrect")]):
        d = run / "candidates" / f"{i:03d}_{label}"
        d.mkdir(parents=True)
        (d / "kernel.metal").write_text(src)
        (d / "report.json").write_text(json.dumps({"status": status, "speedup": sp, "candidate_us": 100.0 if sp else None}))
    trace = [{"idx": 0, "label": "init", "status": "ok", "speedup": 0.5, "improved": True}, {"idx": 1, "label": "r1.p0", "status": "ok", "speedup": 0.9, "improved": True},
             {"idx": 2, "label": "r1.p1", "status": "incorrect", "speedup": None, "improved": False}]
    base = {"task": "gemm", "dtype": "fp16", "size": "paper", "trace": trace, "best_label": "r1.p0", "best_speedup": 0.9}
    (run / ("result.json" if finished else "state.json")).write_text(json.dumps(base))
    return run


def test_load_run_marks_partial_runs_and_picks_the_best(tmp_path):
    done = make_run(tmp_path, "done", finished=True)
    part = make_run(tmp_path, "part", finished=False)
    info, refs = H.load_run(done, tmp_path)
    assert info["partial"] is False and [r.kind for r in refs] == ["candidate", "best", "candidate"] and refs[1].label == "r1.p0"
    assert refs[1].trace["improved"] is True and refs[0].task == "gemm"
    info2, refs2 = H.load_run(part, tmp_path)
    assert info2["partial"] is True and all(r.partial for r in refs2)


def test_discover_runs_skips_interrupted_backups(tmp_path):
    make_run(tmp_path, "a", finished=True)
    make_run(tmp_path / "_interrupted_x", "b", finished=True)
    found = H.discover_runs([tmp_path])
    assert [d.name for d in found] == ["a"]


def test_cli_end_to_end_on_a_fake_runs_dir(tmp_path, capsys):
    runs = tmp_path / "runs"
    make_run(runs, "fake_gemm", finished=True)
    out = tmp_path / "out"
    rc = H.main([str(runs), "--runs-root", str(runs), "--out", str(out), "--no-compile", "--no-validation", "--jobs", "1"])
    assert rc == 0
    report = (out / "REPORT.md").read_text()
    assert "fake_gemm" in report and "matrix unit: yes" in report
    data = json.loads((out / "data.json").read_text())
    assert data["best_kernels"][0]["label"] == "r1.p0" and "matrix_unit" in data["best_kernels"][0]["present"]
    rows = [json.loads(line) for line in (out / "kernels.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all("summary" in r for r in rows)
    # the win-rate table counts the accepted wins of the two matrix candidates (one has it, two do not)
    wt = data["winrate"]["matrix_unit"]
    assert wt["with"]["n"] == 1 and wt["with"]["wins"] == 1 and wt["without"]["n"] == 2


def test_cli_single_source(tmp_path, capsys):
    p = tmp_path / "k.metal"
    p.write_text(VEC)
    assert H.main(["--source", str(p), "--no-compile"]) == 0
    assert "vector: 16B ld / 16B st" in capsys.readouterr().out
    assert H.main(["--ablate"]) == 0
