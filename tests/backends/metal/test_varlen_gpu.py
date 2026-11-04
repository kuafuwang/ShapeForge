"""varlen KDA 的真实 Metal 垂直切片：分段 recurrence、逐段终态与零状态作弊的拒绝。

三个长度分布档分别是不同的语义实例（``seq_lens`` 派生 total_tokens/num_seqs/NSP1，分段边界进
身份）。这里在真实 Metal 上证明：

* 每档编译/ABI/安全/正确性都过（分段 recurrence 与逐段 ``final_state`` 都被 fp64 分段 oracle 核对）；
* ``log_decay_tab`` workspace 是真实可用的：候选可以声明它、写入它、再在 recurrence 里读回来；
* 丢掉 ``initial_state``（零状态续传）、忽略 ``cu_seqlens`` 的候选会被正确性拒绝——这正是
  ``hacking_example`` 的作弊面；
* L1：把 canonical 边界写死的候选会在同 total/count 的 permuted/fresh 分段上失败；output 正确但
  ``final_state`` 写错的候选也会被拒绝（终态真的参与比较）。
"""

from __future__ import annotations

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.target import load_target

pytestmark = pytest.mark.gpu

TIERS = ("fixed", "mixed", "uniform")

# 每个 (sequence, head) 一个线程；fp32 V-first 状态留在寄存器里，段末写 final_state。
NAIVE = """#include <metal_stdlib>
using namespace metal;

kernel void varlen_kda(device const half* q [[buffer(0)]],
                       device const half* k [[buffer(1)]],
                       device const half* v [[buffer(2)]],
                       device const half* g [[buffer(3)]],
                       device const half* beta [[buffer(4)]],
                       device const float* A_log [[buffer(5)]],
                       device const float* dt_bias [[buffer(6)]],
                       device const float* initial_state [[buffer(7)]],
                       device const int* cu_seqlens [[buffer(8)]],
                       device const float* scale [[buffer(9)]],
                       device half* output [[buffer(10)]],
                       device float* final_state [[buffer(11)]],
                       uint i [[thread_position_in_grid]]) {
    uint seq = i / H;
    uint head = i % H;
    int begin = cu_seqlens[seq];
    int end = cu_seqlens[seq + 1];

    float state[D * D];
    for (int vi = 0; vi < D; vi++) {
        for (int ki = 0; ki < D; ki++) {
            state[vi * D + ki] = INITIAL(vi, ki);
        }
    }

    float scale_value = scale[0];
    float decay_scale = exp(A_log[head]);

    for (int t = begin; t < end; t++) {
        float qv[D], kv[D];
        float q_norm = 0.0f, k_norm = 0.0f;
        for (int ki = 0; ki < D; ki++) {
            qv[ki] = float(q[(t * H + head) * D + ki]);
            kv[ki] = float(k[(t * H + head) * D + ki]);
            q_norm += qv[ki] * qv[ki];
            k_norm += kv[ki] * kv[ki];
        }
        float q_scale = rsqrt(q_norm + 1e-6f) * scale_value;
        float k_scale = rsqrt(k_norm + 1e-6f);
        for (int ki = 0; ki < D; ki++) {
            qv[ki] *= q_scale;
            kv[ki] *= k_scale;
        }

        for (int ki = 0; ki < D; ki++) {
            float gate = float(g[(t * H + head) * D + ki]) + dt_bias[head * D + ki];
            float log_decay = -5.0f * (1.0f / (1.0f + exp(-decay_scale * gate)));
            for (int vi = 0; vi < D; vi++) {
                state[vi * D + ki] *= exp(log_decay);
            }
        }

        float beta_value = 1.0f / (1.0f + exp(-float(beta[t * H + head])));
        for (int vi = 0; vi < D; vi++) {
            float residual = float(v[(t * H + head) * D + vi]);
            for (int ki = 0; ki < D; ki++) {
                residual -= state[vi * D + ki] * kv[ki];
            }
            residual *= beta_value;
            for (int ki = 0; ki < D; ki++) {
                state[vi * D + ki] += residual * kv[ki];
            }
        }

        for (int vi = 0; vi < D; vi++) {
            float out = 0.0f;
            for (int ki = 0; ki < D; ki++) {
                out += state[vi * D + ki] * qv[ki];
            }
            output[(t * H + head) * D + vi] = half(out);
        }
    }

    for (int vi = 0; vi < D; vi++) {
        for (int ki = 0; ki < D; ki++) {
            final_state[(seq * H + head) * D * D + vi * D + ki] = state[vi * D + ki];
        }
    }
}
"""

LAUNCH = "/* taic-launch\nmath safe\nlaunch varlen_kda grid=num_seqs*H,1,1 threads=1,1,1\n*/\n"

# 声明 workspace 的变体：先在 workspace 里物化 per-token per-channel log-decay，再在 recurrence 里读回来。
WORKSPACE_LAUNCH = (
    "/* taic-launch\nmath safe\nworkspace log_decay_tab total_tokens*H*D*4\n"
    "launch varlen_kda grid=num_seqs*H,1,1 threads=1,1,1 workspace=log_decay_tab\n*/\n"
)


def _source(initial: str = "initial_state[(seq * H + head) * D * D + vi * D + ki]") -> str:
    """@brief naive 候选源码。

    @param initial 状态初值表达式（零状态作弊把它换成 ``0.0f``）。
    @return 完整 MSL 源码。
    """
    return (NAIVE + LAUNCH).replace("INITIAL(vi, ki)", initial)


def _evaluate(tmp_path, size: str, source: str):
    """@brief 在真实 Metal 上评估一个候选。

    @param tmp_path 临时目录（未用；保留供将来写入辅助文件）。
    @param size 长度分布档。
    @param source 候选源码（提交文本）。
    @return 组合运行结果。
    """
    binding = bind_task("varlen_kda", "fp16", size)
    return evaluate_candidate(
        binding.locator, source, target=load_target("apple-m4-max"), mode="quick", reps=3
    )


@pytest.mark.parametrize("size", TIERS)
def test_every_length_distribution_is_a_real_instance(tmp_path, size):
    """@brief 三档（fixed/mixed/uniform）各自编译、ABI、安全、正确性都过。"""
    run = _evaluate(tmp_path, size, _source())

    assert run.result.report["status"] == "ok", run.result.report["summary"]
    assert run.result.report["correctness"]["ok"] is True
    instance = bind_task("varlen_kda", "fp16", size).contract.instance
    assert instance.cu_seqlens is not None
    assert instance.cu_seqlens[-1] == instance.symbols["total_tokens"]
    assert len(instance.cu_seqlens) == instance.symbols["num_seqs"] + 1


def test_the_distributions_are_distinct_instances(tmp_path):
    """@brief 三档的实例指纹互不相同（长度分布进身份）。"""
    from tcenv.identity import fingerprint

    digests = {
        fingerprint(bind_task("varlen_kda", "fp16", size).contract.instance.fingerprint_document())
        for size in TIERS
    }
    assert len(digests) == len(TIERS)


def test_a_declared_workspace_is_bound_and_usable(tmp_path):
    """@brief ``log_decay_tab`` 能按 ABI 槽绑定、被候选写入、再在 recurrence 里读回来。"""
    source = (NAIVE + WORKSPACE_LAUNCH).replace(
        "INITIAL(vi, ki)",
        "initial_state[(seq * H + head) * D * D + vi * D + ki]",
    ).replace(
        "                       device float* final_state [[buffer(11)]],",
        "                       device float* final_state [[buffer(11)]],\n"
        "                       device float* log_decay_tab [[buffer(12)]],",
    ).replace(
        """    float scale_value = scale[0];""",
        """    for (int t = begin; t < end; t++) {
        for (int ki = 0; ki < D; ki++) {
            float gate = float(g[(t * H + head) * D + ki]) + dt_bias[head * D + ki];
            log_decay_tab[(t * H + head) * D + ki] = -5.0f * (1.0f / (1.0f + exp(-exp(A_log[head]) * gate)));
        }
    }

    float scale_value = scale[0];""",
    ).replace(
        """        for (int ki = 0; ki < D; ki++) {
            float gate = float(g[(t * H + head) * D + ki]) + dt_bias[head * D + ki];
            float log_decay = -5.0f * (1.0f / (1.0f + exp(-decay_scale * gate)));""",
        """        for (int ki = 0; ki < D; ki++) {
            float log_decay = log_decay_tab[(t * H + head) * D + ki];""",
    )

    run = _evaluate(tmp_path, "mixed", source)

    assert run.result.report["status"] == "ok", run.result.report["summary"]
    assert run.result.report["correctness"]["ok"] is True
    assert run.evaluation["workspace_bytes"] == {"log_decay_tab": 14 * 2 * 8 * 4}


def test_dropping_the_initial_state_is_refused(tmp_path):
    """@brief 零状态续传（丢掉 ``initial_state``）必须被正确性拒绝。"""
    run = _evaluate(tmp_path, "mixed", _source("0.0f"))

    assert run.result.report["status"] == "incorrect"
    assert run.result.report["correctness"]["ok"] is False
    # 丢掉初值的偏差是量级级的（不是容差边界），oracle 有真实的判别力
    assert run.result.report["correctness"]["cases"][0]["scaled_err"] > 10.0


def test_a_candidate_that_ignores_the_segment_boundaries_is_refused(tmp_path):
    """@brief 把整包当成一条序列（忽略 cu_seqlens 分段）必须被正确性拒绝。"""
    source = _source().replace(
        "    int begin = cu_seqlens[seq];\n    int end = cu_seqlens[seq + 1];",
        "    int begin = 0;\n    int end = total_tokens;",
    )
    run = _evaluate(tmp_path, "mixed", source)

    assert run.result.report["status"] == "incorrect"
    assert run.result.report["correctness"]["ok"] is False
    assert run.result.report["correctness"]["cases"][0]["scaled_err"] > 10.0


# --- L1: 任意合法分段 -------------------------------------------------------------------
# 这几条负例依赖 `mode="quick"` 跑全部声明 case：canonical 本身一定过，能识别“写死 canonical
# 边界”的是 permuted / fresh 这些同 total/count 的其他合法分段。

#: ``mixed`` 档（seq_lens = [4, 8, 2]）的 canonical cu_seqlens。一个把它写死的候选在 permuted/
#: fresh 上必然算错——这正是 L1 要拒绝的过拟合。
CANONICAL_TABLE = "    const int CANONICAL[NSP1] = {0, 4, 12, 14};\n"


def _hardcoded(size: str, cases: tuple[str, ...] | None = None):
    """@brief 把 canonical 边界写成常量的候选（不再读 ``cu_seqlens``）。

    @param size 长度分布档。
    @param cases 只跑指定用例；None 跑合同的全部用例。
    @return 组合运行结果。
    """
    source = _source().replace(
        "    int begin = cu_seqlens[seq];\n    int end = cu_seqlens[seq + 1];",
        CANONICAL_TABLE + "    int begin = CANONICAL[seq];\n    int end = CANONICAL[seq + 1];",
    )
    binding = bind_task("varlen_kda", "fp16", size)
    return evaluate_candidate(
        binding.locator,
        source,
        target=load_target("apple-m4-max"),
        mode="quick",
        reps=3,
        cases=cases,
    )


def test_a_candidate_that_hardcodes_the_canonical_boundaries_is_refused(tmp_path):
    """@brief 写死 canonical 分段的候选在 permuted/skewed 等合法分段上必须失败。"""
    run = _hardcoded("mixed")

    assert run.result.report["status"] == "incorrect"
    assert run.result.report["correctness"]["ok"] is False
    bad = run.result.report["correctness"]["cases"][-1]
    assert bad["ok"] is False
    # canonical 本身能过：落败的是另一个合法分段，而不是容差或编译问题。
    assert bad["case"] != "canonical"
    assert bad["scaled_err"] > 10.0


def test_a_candidate_that_hardcodes_the_canonical_boundaries_is_refused_by_a_permuted_case(tmp_path):
    """@brief 只跑 canonical + permuted：写死 canonical 的候选必须在 permuted 单独失败。

    这一条把 permuted 从“全部 case 里最后一个失败者”中拆出来单独钉住：即使未来 skewed 等
    后续 case 仍会失败，也不能用它们掩盖 permuted 意外通过。
    """
    run = _hardcoded("mixed", cases=("canonical", "permuted"))

    assert run.result.report["status"] == "incorrect"
    assert run.result.report["correctness"]["ok"] is False
    cases = run.result.report["correctness"]["cases"]
    assert [case["case"] for case in cases] == ["canonical", "permuted"]
    assert cases[0]["ok"] is True
    assert cases[1]["ok"] is False
    assert cases[1]["scaled_err"] > 10.0


def test_a_candidate_that_hardcodes_the_canonical_boundaries_is_refused_by_a_fresh_draw(tmp_path):
    """@brief 即使只跑 canonical + fresh_distribution，写死边界也必须被 fresh 拒绝。"""
    run = _hardcoded("mixed", cases=("canonical", "fresh_distribution"))

    assert run.result.report["status"] == "incorrect"
    assert run.result.report["correctness"]["ok"] is False
    cases = run.result.report["correctness"]["cases"]
    assert [case["case"] for case in cases] == ["canonical", "fresh_distribution"]
    assert cases[0]["ok"] is True
    assert cases[1]["ok"] is False


def test_output_correct_but_final_state_wrong_is_refused(tmp_path):
    """@brief output 全对、``final_state`` 写错的候选必须被拒绝（终态真的在比较）。"""
    source = _source().replace(
        "            final_state[(seq * H + head) * D * D + vi * D + ki] = state[vi * D + ki];",
        "            final_state[(seq * H + head) * D * D + vi * D + ki] = "
        "initial_state[(seq * H + head) * D * D + vi * D + ki];",
    )
    run = _evaluate(tmp_path, "mixed", source)

    assert run.result.report["status"] == "incorrect"
    assert run.result.report["correctness"]["ok"] is False
    first = run.result.report["correctness"]["cases"][0]
    assert first["case"] == "canonical"
    assert first["worst"]["output"] == "final_state"
    assert first["scaled_err"] > 10.0
