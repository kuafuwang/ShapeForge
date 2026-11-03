"""Hardware-feature detector (``experiments/hw_features.py``): synthetic MSL snippets, the generated reference GEMM, verdict logic and run loading.

No GPU: the compiled-evidence tests only call the Metal compiler front end (``xcrun metal -S -emit-llvm``) and are skipped when the toolchain
component is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import hw_features as H
from tests.support.v1_tasks import V1_TASK as _UNUSED_V1_TASK  # noqa: F401

# B1 moved rope/gemv (the previous elementwise/rowwise specimens) to v2.  These tests exercise
# the elementwise *family* rules, so they name lion explicitly instead of taking the global
# default v1 task (conv2d, which belongs to the compute family).
V1_TASK = "lion"

CONSTS = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "M": 2048, "N": 2048, "K": 2048}


def feats(src: str, **kw) -> dict:
    """@brief 用默认常量参数分析一段源码，取得各硬件特性的证据。

    @param src MSL 源码文本。
    @param kw 其它分析参数，其中 consts 覆盖默认常量表。
    @return analyze_source 的分析结果。
    """
    return H.analyze_source(src, consts=kw.pop("consts", CONSTS), **kw)


def present(src: str, task: str | None = None) -> set[str]:
    """@brief 构建不带编译证据的报告，取出判定为存在的特性名集合。

    @param src MSL 源码文本。
    @param task 任务名，决定期望特性。
    @return 存在（present）的特性名集合。
    """
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
    """@brief 验证去注释与去字符串只清空内容、保留原长度与换行，使行号仍对得上。
    """
    src = 'int a; // c1\n/* c2\n c3 */ const char* s = "// not a comment /* x */";\nint b;\n'
    out = H.strip_comments_and_strings(src)
    assert len(out) == len(src) and out.count("\n") == src.count("\n")
    assert "c1" not in out and "c2" not in out and "c3" not in out and "not a comment" not in out
    assert "int a;" in out and "int b;" in out and '"' in out


def test_eval_expr_and_constants():
    """@brief 验证常量表达式求值支持四则运算、强制转换、移位与 sizeof，无法求值则返回 None，并能收集 constexpr/#define。
    """
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
    """@brief 验证内联辅助函数的调用会被实参代入展开，使辅助函数体内的访存也能被检测到。
    """
    funcs = {f.name: f for f in H.parse_functions("inline void put(threadgroup half* d, uint i, uint4 v) { d[i] = v.x; }\nkernel void k() { }")}
    out = H.expand_calls("put(buf, lid * 2, pa0);", {"put": funcs["put"]})
    assert "buf[(lid * 2)] = pa0.x" in out.replace("  ", " ")


# ---------------------------------------------------------------------------------------------------------------------------------------
# detection and the negative cases
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_matrix_kernel_features():
    """@brief 验证矩阵核被识别出 MMA 点、2x2 float 累加分片、half 操作数、threadgroup 载入、填充与屏障，且不误报其他特性。
    """
    e = feats(MATRIX, kernel_name="gemm_kernel")
    m = e["matrix"]
    assert m["present"] and m["mma_sites"] == 1
    assert m["acc_tiles"] == 4 and m["acc_dims"] == [2, 2] and m["acc_types"] == ["float"] and m["operand_types"] == ["half"]
    assert m["loads_from"] == ["threadgroup"]
    assert e["threadgroup"]["staging"] and e["threadgroup"]["declared_bytes"] == 2 * (32 * 40 + 32 * 72)
    assert e["threadgroup"]["padded"]
    assert e["sync"]["threadgroup_barrier_sites"] == 2 and e["sync"]["main_loop_barrier_sites"] == 2
    assert e["memory"]["device_load"]["max_bytes"] == 16
    p = present(MATRIX, "bitdelta_gemm")
    assert {"matrix_unit", "float_accumulate", "register_blocking", "tg_staging", "tg_padding", "vector_access", "simdgroup_tiling"} <= p
    assert not {"online_softmax", "pipelining", "double_buffering", "simd_reduce"} & p


def test_scalar_kernel_has_no_hardware_features():
    """@brief 验证标量核不会被判为用上矩阵、向量等硬件特性，最多只有单趟、屏障少这类基础判定。
    """
    p = present(SCALAR)
    assert not (p - {"single_pass", "few_barriers"})
    e = feats(SCALAR)
    assert not e["matrix"]["present"] and e["memory"]["device_load"]["max_bytes"] == 2


@pytest.mark.parametrize("src", [COMMENT_ONLY, STRING_ONLY])
def test_mentions_in_comments_and_strings_do_not_count(src):
    """@brief 验证只在注释或字符串里出现的特性名不会被当成真的用了该特性。

    @param src 参数化的 MSL 片段，一次是注释里的特性名，一次是字符串里的。
    """
    p = present(src)
    assert not (p - {"single_pass", "few_barriers"}), p
    e = feats(src)
    assert not e["matrix"]["present"] and not e["simd"]["ops"] and not e["threadgroup"]["arrays"] and e["unroll"]["pragmas"] == 0
    assert not e["online_softmax"]["present"]


def test_vector_widths_loads_and_stores_are_told_apart():
    """@brief 验证向量访存会区分载入与存储两种方向，各自统计宽度与最大字节数。
    """
    m = feats(VEC)["memory"]
    assert m["device_load"]["widths"].get("8") and m["device_load"]["widths"].get("16")
    assert m["device_store"]["widths"].get("8") and m["device_store"]["widths"].get("16")
    assert m["device_load"]["max_bytes"] == 16 and m["device_store"]["max_bytes"] == 16
    assert "vector_access" in present(VEC)


def test_c_style_casts_are_vector_accesses():
    """@brief 验证 C 风格指针强转的访存也被归为向量访问，并给出 8 字节宽度。
    """
    m = feats(CCAST)["memory"]
    assert m["device_load"]["max_bytes"] == 8 and m["device_store"]["max_bytes"] == 8


def test_parenthesised_dereference_store_is_a_store():
    """@brief 验证括号包住的解引用赋值仍被计为一次 16 字节存储，而不是漏算。
    """
    src = """
    kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        const uint4 v = *(reinterpret_cast<device const uint4*>(x + i * 8));
        *(reinterpret_cast<device uint4*>(y + i * 8)) = v;
    }"""
    m = feats(src)["memory"]
    assert m["device_load"]["widths"] == {"16": 1} and m["device_store"]["widths"] == {"16": 1}


def test_typed_pointers_are_scoped_per_function():
    """@brief 验证类型化指针的角色（device 载入、threadgroup 存储）按函数作用域分别记账，不会跨函数串味。
    """
    src = """
    inline void a(device const half* x, thread uint4& r) { const device uint4* p = reinterpret_cast<const device uint4*>(x); r = p[0]; }
    inline void b(threadgroup half* t, uint4 r) { threadgroup uint4* p = reinterpret_cast<threadgroup uint4*>(t); p[0] = r; }
    kernel void k(device const half* x [[buffer(0)]]) { threadgroup half t[64]; uint4 r; a(x, r); b(t, r); }"""
    m = feats(src)["memory"]
    assert m["device_load"]["max_bytes"] == 16 and m["threadgroup_store"]["max_bytes"] == 16 and m["device_store"]["max_bytes"] == 0


def test_macro_bodies_are_analysed():
    """@brief 验证宏展开体里的 MMA 调用同样被计为矩阵指令。
    """
    src = """
    #define MMA(ACC, A, B) simdgroup_multiply_accumulate(ACC, A, B, ACC)
    kernel void k(device half* y [[buffer(0)]]) {
        simdgroup_float8x8 c = simdgroup_float8x8(0); simdgroup_half8x8 a, b;
        MMA(c, a, b);
    }"""
    assert feats(src)["matrix"]["mma_sites"] == 1


def test_double_buffering_patterns():
    """@brief 验证显式双缓冲的两种写法都能识别，而等大的 A/B 操作数缓冲不会被误判为双缓冲。
    """
    assert feats(DB_TERNARY)["threadgroup"]["double_buffer"] == "explicit"
    assert feats(DB_STAGE_DIM)["threadgroup"]["double_buffer"] == "explicit"
    # A and B operand tiles of equal size are two buffers, not two stages
    assert feats(TWO_OPERAND_TILES)["threadgroup"]["double_buffer"] == "no"
    assert "double_buffering" not in present(TWO_OPERAND_TILES)
    assert "pipelining" in present(DB_TERNARY)


def test_online_softmax_versus_two_pass():
    """@brief 验证在线 softmax 与两趟写法能区分，并且能从全局读取次数与单趟特性印证。
    """
    assert feats(ONLINE)["online_softmax"]["present"]
    assert not feats(TWO_PASS)["online_softmax"]["present"]
    assert feats(TWO_PASS)["global_read_passes"] == {"x": 2}
    assert feats(ONLINE)["global_read_passes"] == {"x": 1}
    assert "single_pass" in present(ONLINE) and "single_pass" not in present(TWO_PASS)


def test_padding_through_a_named_pad_constant_is_found():
    """@brief 验证填充通过命名的 PAD 常量间接表达时仍能识别，并附上推导依据；无关常量不算填充。
    """
    src = """
    kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint lid [[thread_index_in_threadgroup]]) {
        constexpr uint PAD = 8;
        constexpr uint ST = 64 + PAD;
        threadgroup half tile[32 * ST];
        tile[(lid / 64) * ST + lid % 64] = x[lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        y[lid] = tile[(lid / 64) * ST + lid % 64];
    }"""
    ev = feats(src)["threadgroup"]
    assert ev["padded"] and any("ST = 64 + PAD" in e for e in ev["padding_evidence"])
    unrelated = "kernel void k(device half* y [[buffer(0)]], uint i [[thread_position_in_grid]]) { constexpr uint N2 = 7 + 1; y[i * N2] = 0; }"
    assert not feats(unrelated)["threadgroup"]["padded"]


def test_online_softmax_is_found_through_an_exp_wrapper_helper():
    """@brief 验证在线 softmax 能穿透 exp 包装辅助函数识别，而名字近似的 expand 不算指数运算。
    """
    src = ONLINE.replace("exp2(", "exp2_acc(").replace("kernel void softmax_kernel", "inline float exp2_acc(float x) { return exp2(x); }\nkernel void softmax_kernel")
    assert feats(src)["online_softmax"]["present"]
    assert not feats(TWO_PASS.replace("exp2(", "exp2_acc("))["online_softmax"]["present"]
    assert not feats(ONLINE.replace("exp2(", "expand("))["online_softmax"]["present"]  # 'expand' is not an exponential


def test_register_prefetch_requires_loads_in_flight_across_the_barrier():
    """@brief 验证寄存器预取要求载入跨越屏障提前发出，串行版本与流水线版本因此给出相反结果。
    """
    assert feats(PIPELINED, kernel_name="gemm_kernel")["threadgroup"]["register_prefetch"]
    assert "pipelining" in present(PIPELINED)
    assert not feats(SERIAL, kernel_name="gemm_kernel")["threadgroup"]["register_prefetch"]
    assert "pipelining" not in present(SERIAL)


def test_bit_packing_needs_bit_ops_on_loaded_data_not_index_math():
    """@brief 验证位打包只在位运算作用于载入数据时成立，索引算术与普通向量访存都不算。
    """
    assert feats(BIT_DECODE)["bit_packing"]["present"]
    assert not feats(VEC)["bit_packing"]["present"]
    assert not feats(SCALAR)["bit_packing"]["present"]
    assert not feats(PIPELINED, kernel_name="gemm_kernel")["bit_packing"]["present"]


def test_simd_and_math_detection():
    """@brief 验证 simd 归约与洗牌、fast 命名空间数学函数以及 exp2 的 log2(e) 折换都能被识别。
    """
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
    """@brief 验证引入 metal_tensor 或使用 simdgroup_event 时会标记张量 API 与异步拷贝。
    """
    src = "#include <metal_tensor>\nkernel void k(device half* y [[buffer(0)]]) { simdgroup_event e; }\n"
    e = feats(src)
    assert e["matrix"]["tensor_api"] and e["async_copy"]


def test_unreachable_helper_is_not_counted():
    """@brief 验证 kernel 从未调用的辅助函数里的 simd 调用不计入特性，同时被列为未使用 helper。
    """
    src = "inline float dead(float v) { return simd_sum(v); }\nkernel void k(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] = 1.0f; }\n"
    e = feats(src)
    assert e["helpers_unused"] == ["dead"] and not e["simd"]["ops"]


# ---------------------------------------------------------------------------------------------------------------------------------------
# the generated reference GEMMs and the naive kernels of the repository
# ---------------------------------------------------------------------------------------------------------------------------------------
def test_reference_gemm_is_detected_as_matrix_unit_user():
    """@brief 验证生成的参考 GEMM 被判为矩阵单元使用者：16 个 4x4 float 分片、threadgroup 载入与填充、无双缓冲，并写进汇总。
    """
    from experiments.contract_sensitivity import gemm_source

    src = gemm_source("64x64x32t128", "fp16")
    e = feats(src, consts={**CONSTS})
    m = e["matrix"]
    assert m["present"] and m["acc_types"] == ["float"] and m["operand_types"] == ["half"]
    assert m["acc_tiles"] == 16 and m["acc_dims"] == [4, 4]
    assert e["threadgroup"]["staging"] and e["threadgroup"]["padded"] and e["threadgroup"]["double_buffer"] == "no"
    assert e["memory"]["device_load"]["max_bytes"] == 8
    rep = H.build_report(src, label="ref", task="bitdelta_gemm", dtype="fp16", compile_evidence=False)
    assert rep.features["matrix_unit"]["present"] and rep.features["register_blocking"]["present"] and rep.features["tg_padding"]["present"]
    assert "matrix unit: yes" in rep.summary and "16 tile" in rep.summary


def test_naive_kernels_do_not_claim_a_matrix_unit():
    """@brief 验证仓库里每个 naive kernel 都不会被误判为用上矩阵单元，也不会虚假声称向量访存（bitdelta_gemm 除外）。
    """
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


def test_the_air_cache_key_moves_with_the_toolchain(monkeypatch, tmp_path):
    """@brief 验证 AIR 缓存键随工具链变化：同一源码在旧前端编出的证据不能在新工具链的键下被复用。

    Xcode 升级会改变同一源码的 AIR；缓存键同时绑定源码文本、std 与 fast_math 选项，
    以及工具链指纹，任一变化都会得到新键。

    @param monkeypatch pytest 的 monkeypatch，用于把工具链探测固定为临时文件。
    @param tmp_path pytest 提供的临时目录。
    """
    import taic.hwfeatures as HF

    tool = tmp_path / "metal"
    tool.write_text("x")
    monkeypatch.setattr(HF, "metal_toolchain", lambda: str(tool))
    HF._TOOLCHAIN.pop("fingerprint", None)
    first = HF.air_cache_key("kernel void k() {}", std="metal4.0", fast_math=True)
    assert first == HF.air_cache_key("kernel void k() {}", std="metal4.0", fast_math=True)
    assert first != HF.air_cache_key("kernel void k() {}", std="metal3.2", fast_math=True)
    assert first != HF.air_cache_key("kernel void k() { }", std="metal4.0", fast_math=True)
    import os

    os.utime(tool, (1_600_000_000, 1_600_000_000))
    HF._TOOLCHAIN.pop("fingerprint", None)
    moved = HF.air_cache_key("kernel void k() {}", std="metal4.0", fast_math=True)
    assert moved != first


@needs_metal
def test_compiled_evidence_confirms_matrix_intrinsics_and_widths(tmp_path):
    """@brief 验证编译证据从 AIR 里读出矩阵指令、f32 累加与 f16 操作数、屏障、共享内存字节与至少 8 字节的访存，且二次编译命中缓存。

    @param tmp_path pytest 提供的临时目录，同时充当 AIR 缓存目录。
    """
    from experiments.contract_sensitivity import gemm_source
    from tasks import get_task

    t = get_task("bitdelta_gemm", "fp16", "paper")
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
    """@brief 验证编译证据能区分标量核（无 MMA、无 simd、最多 2 字节载入）与手写向量核（16 字节载入与存储）。

    @param tmp_path pytest 提供的临时目录，同时充当 AIR 缓存目录。
    """
    from tasks import get_task

    t = get_task(V1_TASK, "fp16", "paper")
    head = t.contract.render_header()
    scalar = H.compile_air(head + t.naive_msl, cache_dir=tmp_path)
    assert scalar["ok"] and scalar["matrix"]["multiply_accumulate"] == 0 and not scalar["simd"]
    assert max((int(b) for b in scalar["device_load_bytes"]), default=0) <= 2
    vec = H.compile_air(head + "kernel void relu_kernel(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
                               "    *reinterpret_cast<device uint4*>(y + i * 8) = *reinterpret_cast<device const uint4*>(x + i * 8); }\n", cache_dir=tmp_path)
    assert vec["ok"] and "16" in vec["device_load_bytes"] and "16" in vec["device_store_bytes"]


@needs_metal
def test_compile_error_is_reported_not_raised(tmp_path):
    """@brief 验证编译失败只作为结果里的错误字段返回，不会向调用方抛异常。

    @param tmp_path pytest 提供的临时目录，同时充当 AIR 缓存目录。
    """
    comp = H.compile_air("kernel void k( { }", cache_dir=tmp_path)
    assert not comp["ok"] and "error" in comp["error"]


@needs_metal
def test_full_report_has_all_three_evidence_levels(tmp_path):
    """@brief 验证完整报告能叠加源码、编译与实测三层证据，给出“用上且有效”的结论，且 JSON 与内部分析一致。

    @param tmp_path pytest 提供的临时目录，同时充当 AIR 缓存目录。
    """
    from experiments.contract_sensitivity import gemm_source

    measured = {"status": "ok", "candidate_us": 1361.0, "achieved_tflops": 12.6, "peak_tflops_measured": 14.77, "speedup": 0.87, "candidate_us_": 1}
    rep = H.build_report(gemm_source("64x64x32t128", "fp16"), label="ref", task="bitdelta_gemm", dtype="fp16", measured=measured, ceilings=H.Ceilings(peak_tflops=14.77), cache_dir=tmp_path)
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
    """@brief 构造一份最简实测记录，默认状态 ok、耗时 100 微秒。

    @param kw 要覆盖的实测字段，例如 achieved_tflops。
    @return 实测字典。
    """
    return {"status": "ok", "candidate_us": 100.0, **kw}


def test_perf_test_uses_the_no_feature_ceiling():
    """@brief 验证性能判定以不用该特性的上限为基准，覆盖高于/持平/未测三档，缺 naive 基线时按标量 FMA 相对矩阵峰值的比例回退，且变体任务沿用基础族。
    """
    c = H.Ceilings(naive_tflops=5.5, peak_tflops=14.8, naive_gbps=440.0, peak_gbps=500.0)
    assert H.perf_test("gemm", _meas(achieved_tflops=14.0), c)["result"] == "above"
    assert H.perf_test("gemm", _meas(achieved_tflops=5.6), c)["result"] == "at"
    assert H.perf_test(V1_TASK, _meas(achieved_gbps=441.0), c)["result"] == "at"
    assert H.perf_test(V1_TASK, _meas(achieved_gbps=470.0), c)["result"] == "above"
    assert H.perf_test("gemm", {"status": "incorrect"}, c)["result"] == "unmeasured"
    fallback = H.Ceilings(peak_tflops=14.8)
    r = H.perf_test("gemm", _meas(achieved_tflops=14.0), fallback)
    assert r["result"] == "above" and abs(r["ceiling"] - H.SCALAR_OVER_MATRIX * 14.8) < 0.01 and "matrix peak" in r["ceiling_basis"]
    assert H.perf_test("flash_attention@64x32t256", _meas(achieved_tflops=14.0), fallback)["result"] == "above"  # variants use the base task's family


def test_judge_assigns_the_four_verdicts():
    """@brief 验证判定能给出四种结论，缺失特性按优先级排序，且归因区分 primary 与 joint、按任务族选主特性。
    """
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
    g, _ = H.judge(V1_TASK, g, {"result": "at"})
    assert g["vector_access"]["verdict"] == "present but not effective" and g["vector_access"]["attribution"] == "primary"


def test_docs_table3_matches_the_expectations_in_code():
    """@brief 验证文档表 3 声明的期望特性与代码里的 EXPECTED 表逐项一致，且两边覆盖的任务集合相同。
    """
    doc = (Path(__file__).resolve().parents[3] / "docs" / "hardware_features.md").read_text().splitlines()
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
    """@brief 验证每个任务都有期望特性与所属族，且期望名都在特性表内、等级取值为 required 或 recommended。
    """
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
    """@brief 造一个假 run 目录，含三个候选与对应的 trace，并按是否结束写 result.json 或 state.json。

    @param root runs 根目录。
    @param name run 目录名。
    @param finished 为真时写 result.json，否则写 state.json。
    @return 新建的 run 目录。
    """
    run = root / name
    for i, (label, src, sp, status) in enumerate([("init", SCALAR, 0.5, "ok"), ("r1.p0", MATRIX, 0.9, "ok"), ("r1.p1", SCALAR, None, "incorrect")]):
        d = run / "candidates" / f"{i:03d}_{label}"
        d.mkdir(parents=True)
        (d / "kernel.metal").write_text(src)
        # A current candidate report: only schema 4 is evidence for the measured verdict.
        (d / "report.json").write_text(json.dumps({
            "schema_version": 4, "status": status, "speedup": sp, "candidate_us": 100.0 if sp else None,
        }))
    trace = [{"idx": 0, "label": "init", "status": "ok", "speedup": 0.5, "improved": True}, {"idx": 1, "label": "r1.p0", "status": "ok", "speedup": 0.9, "improved": True},
             {"idx": 2, "label": "r1.p1", "status": "incorrect", "speedup": None, "improved": False}]
    base = {"task": "gemm", "dtype": "fp16", "size": "paper", "trace": trace, "best_label": "r1.p0", "best_speedup": 0.9}
    (run / ("result.json" if finished else "state.json")).write_text(json.dumps(base))
    return run


def test_load_run_marks_partial_runs_and_picks_the_best(tmp_path):
    """@brief 验证载入 run 时只写 state.json 的 run 被标为不完整，且引用顺序为首个候选、最优候选、其余候选。

    @param tmp_path pytest 提供的临时目录。
    """
    done = make_run(tmp_path, "done", finished=True)
    part = make_run(tmp_path, "part", finished=False)
    info, refs = H.load_run(done, tmp_path)
    assert info["partial"] is False and [r.kind for r in refs] == ["candidate", "best", "candidate"] and refs[1].label == "r1.p0"
    assert refs[1].trace["improved"] is True and refs[0].task == "gemm"
    info2, refs2 = H.load_run(part, tmp_path)
    assert info2["partial"] is True and all(r.partial for r in refs2)


def test_discover_runs_skips_interrupted_backups(tmp_path):
    """@brief 验证发现 run 时会跳过 _interrupted_ 开头的备份目录。

    @param tmp_path pytest 提供的临时目录。
    """
    make_run(tmp_path, "a", finished=True)
    make_run(tmp_path / "_interrupted_x", "b", finished=True)
    found = H.discover_runs([tmp_path])
    assert [d.name for d in found] == ["a"]


def test_cli_end_to_end_on_a_fake_runs_dir(tmp_path, capsys):
    """@brief 验证 CLI 在假 runs 目录上能产出 REPORT.md、data.json 与 kernels.jsonl，且胜率表按是否存在该特性统计。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获。
    """
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
    """@brief 验证 CLI 的单源码模式会打印向量访存宽度，且 --ablate 能无参取得退出码 0。

    @param tmp_path pytest 提供的临时目录。
    @param capsys pytest 的输出捕获，用于读取向量宽度结果。
    """
    p = tmp_path / "k.metal"
    p.write_text(VEC)
    assert H.main(["--source", str(p), "--no-compile"]) == 0
    assert "vector: 16B ld / 16B st" in capsys.readouterr().out
    assert H.main(["--ablate"]) == 0


def test_an_old_candidate_report_forms_no_measured_evidence(tmp_path):
    """旧候选报告（schema 3）不形成 MEASURED：分析结果标 ``old_format``、无实测数字。

    The reviewer's repro: ``measured_from_report({'schema_version': 3, 'status': 'ok', 'speedup': 9.9,
    'candidate_us': 1.0})`` returned a measured dict with ``speedup=9.9``.  A schema-3 report has no
    build/evaluation identity, so it cannot support a measured verdict or an experience conclusion.
    """
    from taic.hwfeatures import measured_from_report

    old = {"schema_version": 3, "status": "ok", "speedup": 9.9, "candidate_us": 1.0}
    assert measured_from_report(old) is None
    current = {"schema_version": 4, **{k: v for k, v in old.items() if k != "schema_version"}}
    assert measured_from_report(current)["speedup"] == pytest.approx(9.9)
    # A hand-written naive summary has no schema at all: it is a *different* input path, not a
    # looser candidate gate.
    assert measured_from_report({"status": "ok", "candidate_us": 1.0}, source="naive")["candidate_us"] == 1.0
    assert measured_from_report({"status": "ok", "candidate_us": 1.0}) is None

    run = tmp_path / "runs" / "old_gemm"
    folder = run / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text(MATRIX)
    (folder / "report.json").write_text(json.dumps(old))
    (run / "result.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "size": "paper", "best_label": "r1.p0",
    }))
    info, refs = H.load_run(run, tmp_path / "runs")
    assert refs[0].meta["report_state"] == "old_format"
    rows = H.analyze_refs(refs, naive={}, calib=None, runs_root=tmp_path / "runs",
                          compile_evidence=False, cache_dir=None, jobs=1, progress=False)
    _ref, report = rows[0]
    assert report.measured is None, "旧报告不得形成 MEASURED 数字"
    assert report.extra["report_state"] == "old_format"
    assert report.performance_test["result"] == "unmeasured"


def test_an_old_report_cannot_reach_the_winrate_or_discovery_tables(tmp_path):
    """非 current 候选不得通过「重建 trace」或「既有 trace」进入 correct/win/discovery 统计。

    The reviewer's repro: a schema-3 candidate with ``speedup=9.9`` produced
    ``measured=None`` but still ``n=1, correct=1, wins=1, median_speedup_ok=9.9`` in the win-rate
    table, because ``load_run`` rebuilt a trace from the old report and the tables only looked at
    the trace.  Both routes (reconstructed and recorded) must be neutralised.
    """
    from experiments.hw_features import KernelRef, analyze_refs, discovery_table, winrate_table

    def write(name: str, report: dict | None, result: dict | None) -> Path:
        run = tmp_path / "runs" / name
        folder = run / "candidates" / "000_r1.p0"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(MATRIX)
        if report is not None:
            (folder / "report.json").write_text(json.dumps(report))
        (run / "result.json").write_text(json.dumps({
            "task": "gemm", "dtype": "fp16", "size": "paper", "best_label": "r1.p0",
            "best_speedup": 9.9, "confirmed_speedup": 9.9, "confirmed_status": "ok", **(result or {}),
        }))
        return run

    old_report = {"schema_version": 3, "status": "ok", "speedup": 9.9}
    # 1) no trace on disk → the old report must not be rebuilt into one
    rebuilt = write("old_rebuilt", old_report, None)
    _info, refs = H.load_run(rebuilt, tmp_path / "runs")
    assert refs[0].trace is None and refs[0].report is None
    assert refs[0].meta["report_state"] == "old_format"

    # 2) a recorded trace must not make the old candidate count either
    recorded = write("old_recorded", old_report, {"trace": [
        {"idx": 0, "label": "r1.p0", "status": "ok", "speedup": 9.9, "improved": True},
    ]})
    _info2, refs2 = H.load_run(recorded, tmp_path / "runs")
    assert refs2[0].trace is None, "旧报告的 trace 不作为证据"

    items = analyze_refs(refs + refs2, naive={}, calib=None, runs_root=tmp_path / "runs",
                         compile_evidence=False, cache_dir=None, jobs=1, progress=False)
    table = winrate_table(items, ["matrix_unit"])
    for side in ("with", "without"):
        assert table["matrix_unit"][side]["n"] == 0, table
        assert table["matrix_unit"][side]["median_speedup_ok"] is None
    assert discovery_table(items, {}, ["matrix_unit"], {"gemm"})["matrix_unit"]["runs"] == 0

    # 3) A current report still counts: the gate is about the format, not about hiding data.
    current = write("current", {"schema_version": 4, "status": "ok", "speedup": 1.1}, None)
    _info3, refs3 = H.load_run(current, tmp_path / "runs")
    assert refs3[0].trace and refs3[0].trace["reconstructed"] is True
    items3 = analyze_refs(refs3, naive={}, calib=None, runs_root=tmp_path / "runs",
                          compile_evidence=False, cache_dir=None, jobs=1, progress=False)
    assert winrate_table(items3, ["matrix_unit"])["matrix_unit"]["with"]["n"] == 1


def test_the_confirmed_column_is_the_publication_value_not_the_result_field(tmp_path):
    """`confirmed_speedup` 只能来自共同发布谓词；result 里的伪装值不得显示成 confirmed。

    The reviewer's repro: ``load_run`` copied ``result.confirmed_speedup`` and ``render_report``
    printed it as “confirmed speedup” with no gate at all.  A run whose confirmation report is old
    format (or whose kernel/report binding does not hold) publishes nothing, even when
    ``result.confirmed_status == "ok"`` and ``confirmed_speedup == 9.9``.
    """
    from experiments.hw_features import best_kernel_rows

    run = tmp_path / "runs" / "fake"
    folder = run / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text(MATRIX)
    (folder / "report.json").write_text(json.dumps({"schema_version": 4, "status": "ok", "speedup": 1.1}))
    (run / "best.metal").write_text(MATRIX)
    (run / "test.metal").write_text(MATRIX)
    # A schema-3 confirmation report plus a result that claims success.
    (run / "confirm_report.json").write_text(json.dumps({"status": "ok", "speedup": 9.9}))
    (run / "result.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "size": "paper", "best_label": "r1.p0",
        "best_speedup": 1.1, "confirmed_speedup": 9.9, "confirmed_status": "ok",
        "confirmation_source_sha256": "a" * 64, "confirm_winner_label": "r1.p0",
    }))
    info, refs = H.load_run(run, tmp_path / "runs")
    assert info["published_speedup"] is None
    assert info["publication"] != "confirmed"
    assert info["search_best_speedup"] == pytest.approx(1.1)
    assert info["publication_reason"] == "not_confirmed"
    rows = H.analyze_refs(refs, naive={}, calib=None, runs_root=tmp_path / "runs",
                          compile_evidence=False, cache_dir=None, jobs=1, progress=False)
    best = best_kernel_rows(rows, {info["run"]: info})
    assert best and best[0]["published_speedup"] is None
    assert best[0]["search_best_speedup"] == pytest.approx(1.1)
    assert "confirmed_speedup" not in best[0]


def test_reference_summaries_keep_their_own_measured_path(tmp_path):
    """参考 GEMM 的实测行不是候选报告：它走 `source="reference"`，不得被 schema-4 门误伤。"""
    from experiments.hw_features import KernelRef, analyze_refs

    summary = {"status": "ok", "speedup": 1.4, "candidate_us": 100.0,
               "roofline": {"achieved_tflops": 12.0}}
    ref = KernelRef(kid="reference/gemm", run="reference", label="gemm", kind="reference",
                    source=MATRIX, task="gemm", dtype="fp16", report=summary)
    refs = [ref]
    items = analyze_refs(refs, naive={}, calib=None, runs_root=tmp_path,
                         compile_evidence=False, cache_dir=None, jobs=1, progress=False)
    _r, report = items[0]
    assert report.measured is not None, "参考摘要没有 schema，但它是独立输入路径"
    assert report.measured["candidate_us"] == pytest.approx(100.0)
    assert report.extra["report_state"] == "reference"
