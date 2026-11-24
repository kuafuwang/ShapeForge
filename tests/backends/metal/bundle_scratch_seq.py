"""Device candidate: legal poison ping-pong on two fp32 scratch textures.

P2-Q's ping-pong keeps ``device float*`` parameters.  On this toolchain every non-const
device pointer reflects ``readwrite``, so that candidate is refused at reflection even after
the worker keeps scratch.  This source gives each buffer its own texture and a first use of
``write``:

* stage 0 writes ``state_a`` (``texture2d<float, access::write>``);
* stage 1 reads ``state_a`` and writes ``state_b``;
* stage 2 reads ``state_b`` and writes ``final_state``.

The launch block declares only the OpSpec workspaces the kernels bind.  ``handoff_states``
stays in the task ABI and is not moved onto scratch.  The row width is the layout
``scratch_texture_layout`` picks for every registered ``H*D*D`` (4).  Linear index ``i`` is
``uint2(i % 4, i / 4)``.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.support.kda_fixed_bundle import PLANNER_SOURCE, bundle_source
from tests.support.seq_handoff import kda_geometry

SCRATCH_ROW = 4

_PING_PONG_NOTE = (
    Path(__file__).resolve().parents[3]
    / "tests/support/golden/shared/acceptance-epochs/fixtures/p2r/device-comment.txt"
).read_text(encoding="utf-8").rstrip("\n")
_DEVICE = r"""#include <metal_stdlib>
using namespace metal;

__PING_PONG_NOTE__
// The row width is the R32Float layout for every registered kda H*D*D.
/* taic-launch
math safe
workspace work_items 3*2*4
workspace item_counts 1*4
scratch state_a fp32 H*D*D
scratch state_b fp32 H*D*D
launch kda_piece0 grid=H,1,1 threads=1,1,1 workspace=work_items,item_counts scratch=state_a:write,state_b:write
launch kda_piece1 grid=H,1,1 threads=1,1,1 workspace=work_items,item_counts scratch=state_a:read,state_b:write
launch kda_piece2 grid=H,1,1 threads=1,1,1 workspace=work_items,item_counts scratch=state_b:read
*/

constant constexpr uint SCRATCH_ROW = 4u;

inline float scratch_load(texture2d<float, access::read> tex, uint linear) {
    return tex.read(uint2(linear % SCRATCH_ROW, linear / SCRATCH_ROW)).x;
}

inline void scratch_store(texture2d<float, access::write> tex, uint linear, float value) {
    tex.write(float4(value), uint2(linear % SCRATCH_ROW, linear / SCRATCH_ROW));
}

inline void kda_tokens(device const half* q,
                       device const half* k,
                       device const half* v,
                       device const half* g,
                       device const half* beta,
                       device const float* A_log,
                       device const float* dt_bias,
                       device const float* scale,
                       device half* output,
                       thread float* state,
                       uint head,
                       uint begin,
                       uint count) {
    const float scale_value = scale[0];
    const float decay_scale = exp(A_log[head]);
    for (uint token = begin; token < begin + count; ++token) {
        float qv[D], kv[D];
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (uint j = 0; j < D; ++j) {
            const uint index = (token * H + head) * D + j;
            qv[j] = float(q[index]);
            kv[j] = float(k[index]);
            q_norm += qv[j] * qv[j];
            k_norm += kv[j] * kv[j];
        }
        const float q_scale = rsqrt(q_norm + 1e-6f) * scale_value;
        const float k_scale = rsqrt(k_norm + 1e-6f);
        for (uint j = 0; j < D; ++j) {
            qv[j] *= q_scale;
            kv[j] *= k_scale;
            const float gate = float(g[(token * H + head) * D + j]) + dt_bias[head * D + j];
            const float log_decay = -5.0f / (1.0f + exp(-decay_scale * gate));
            for (uint vi = 0; vi < D; ++vi) {
                state[vi * D + j] *= exp(log_decay);
            }
        }
        const float beta_value = 1.0f / (1.0f + exp(-float(beta[token * H + head])));
        for (uint vi = 0; vi < D; ++vi) {
            float residual = float(v[(token * H + head) * D + vi]);
            for (uint j = 0; j < D; ++j) {
                residual -= state[vi * D + j] * kv[j];
            }
            residual *= beta_value;
            for (uint j = 0; j < D; ++j) {
                state[vi * D + j] += residual * kv[j];
            }
        }
        for (uint vi = 0; vi < D; ++vi) {
            float result = 0.0f;
            for (uint j = 0; j < D; ++j) {
                result += state[vi * D + j] * qv[j];
            }
            output[(token * H + head) * D + vi] = half(result);
        }
    }
}

kernel void kda_piece0(device const half* q [[buffer(0)]],
                       device const half* k [[buffer(1)]],
                       device const half* v [[buffer(2)]],
                       device const half* g [[buffer(3)]],
                       device const half* beta [[buffer(4)]],
                       device const float* A_log [[buffer(5)]],
                       device const float* dt_bias [[buffer(6)]],
                       device const float* initial_state [[buffer(7)]],
                       device const float* scale [[buffer(8)]],
                       device half* output [[buffer(9)]],
                       device float* final_state [[buffer(10)]],
                       device uint* work_items [[buffer(11)]],
                       device int* item_counts [[buffer(12)]],
                       texture2d<float, access::write> state_a [[texture(0)]],
                       uint head [[thread_position_in_grid]]) {
    const int published = item_counts[0];
    if (published < 1 || published > 3 || 0 >= published) {
        return;
    }
    float state[D * D];
    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            const uint index = head * D * D + vi * D + ki;
            state[vi * D + ki] = initial_state[index];
        }
    }
    kda_tokens(q, k, v, g, beta, A_log, dt_bias, scale, output, state, head, work_items[0], work_items[1]);
    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            const uint index = head * D * D + vi * D + ki;
            if (published == 1) {
                final_state[index] = state[vi * D + ki];
            } else {
                scratch_store(state_a, index, state[vi * D + ki]);
            }
        }
    }
}

kernel void kda_piece1(device const half* q [[buffer(0)]],
                       device const half* k [[buffer(1)]],
                       device const half* v [[buffer(2)]],
                       device const half* g [[buffer(3)]],
                       device const half* beta [[buffer(4)]],
                       device const float* A_log [[buffer(5)]],
                       device const float* dt_bias [[buffer(6)]],
                       device const float* initial_state [[buffer(7)]],
                       device const float* scale [[buffer(8)]],
                       device half* output [[buffer(9)]],
                       device float* final_state [[buffer(10)]],
                       device uint* work_items [[buffer(11)]],
                       device int* item_counts [[buffer(12)]],
                       texture2d<float, access::read> state_a [[texture(0)]],
                       texture2d<float, access::write> state_b [[texture(1)]],
                       uint head [[thread_position_in_grid]]) {
    const int published = item_counts[0];
    if (published < 1 || published > 3 || 1 >= published) {
        return;
    }
    float state[D * D];
    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            const uint index = head * D * D + vi * D + ki;
            state[vi * D + ki] = scratch_load(state_a, index);
        }
    }
    kda_tokens(q, k, v, g, beta, A_log, dt_bias, scale, output, state, head, work_items[2], work_items[3]);
    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            const uint index = head * D * D + vi * D + ki;
            if (published == 2) {
                final_state[index] = state[vi * D + ki];
            } else {
                scratch_store(state_b, index, state[vi * D + ki]);
            }
        }
    }
}

kernel void kda_piece2(device const half* q [[buffer(0)]],
                       device const half* k [[buffer(1)]],
                       device const half* v [[buffer(2)]],
                       device const half* g [[buffer(3)]],
                       device const half* beta [[buffer(4)]],
                       device const float* A_log [[buffer(5)]],
                       device const float* dt_bias [[buffer(6)]],
                       device const float* initial_state [[buffer(7)]],
                       device const float* scale [[buffer(8)]],
                       device half* output [[buffer(9)]],
                       device float* final_state [[buffer(10)]],
                       device uint* work_items [[buffer(11)]],
                       device int* item_counts [[buffer(12)]],
                       texture2d<float, access::read> state_b [[texture(0)]],
                       uint head [[thread_position_in_grid]]) {
    const int published = item_counts[0];
    if (published < 1 || published > 3 || 2 >= published) {
        return;
    }
    float state[D * D];
    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            const uint index = head * D * D + vi * D + ki;
            state[vi * D + ki] = scratch_load(state_b, index);
        }
    }
    kda_tokens(q, k, v, g, beta, A_log, dt_bias, scale, output, state, head, work_items[4], work_items[5]);
    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            final_state[head * D * D + vi * D + ki] = state[vi * D + ki];
        }
    }
}
""".replace("__PING_PONG_NOTE__", _PING_PONG_NOTE)


def texture_geometry() -> dict:
    """@brief 两块 poison scratch。每块的第一次激活使用都是 write，且只出现在真正写入的 stage。"""
    document = json.loads(json.dumps(kda_geometry("seq_three_piece")))
    uses = (
        (("state_a", "write"),),
        (("state_a", "read"), ("state_b", "write")),
        (("state_b", "read"),),
    )
    for stage, spec in zip(document["stages"], uses):
        stage["workspace"] = [item for item in stage["workspace"] if item["name"] != "handoff_states"]
        stage["scratch"] = [{"name": name, "access": access} for name, access in spec]
    document["scratch"] = [
        {"name": name, "dtype": "fp32", "size_terms": ["H*D*D"], "init_policy": "uninitialized_poison"}
        for name in ("state_a", "state_b")
    ]
    return document


def texture_source() -> str:
    """@brief 可执行的 bundle：纹理乒乓 + 原来的 Host Program。"""
    if _DEVICE.count("SCRATCH_ROW = 4u") != 1:
        raise AssertionError("the texture kernel must hardcode the layout width once")
    return bundle_source(device_source=_DEVICE, geometry=texture_geometry())


def nonzero_host_source() -> str:
    """@brief 同一份纹理核，Host Program 返回 1。求值必须在提交前拒绝。"""
    needle = "return SF_HOST_OK;"
    if PLANNER_SOURCE.count(needle) != 1:
        raise AssertionError("the planner's success return is not unique")
    return bundle_source(
        planner_source=PLANNER_SOURCE.replace(needle, "return 1;", 1),
        device_source=_DEVICE,
        geometry=texture_geometry(),
    )
