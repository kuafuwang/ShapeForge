#include <metal_stdlib>
using namespace metal;

// Candidate-owned fixed-length KDA device pipeline.  The native Host Program writes at most three
// [begin,count] work items.  Serial stages carry fp32 state through handoff_states; Metal buffer
// barriers provide ordering, so no cross-threadgroup spin or host-side KDA-specific rule is needed.

// The launch block is an executor cross-audit only; the bundle's execution_geometry.json remains
// authoritative for dispatch.  Keeping the two projections aligned is part of the delivery contract.
/* taic-launch
math safe
workspace work_items 3*2*4
workspace item_counts 1*4
workspace handoff_states 2*H*D*D*4
launch kda_piece0 grid=H,1,1 threads=1,1,1 workspace=work_items,item_counts,handoff_states
launch kda_piece1 grid=H,1,1 threads=1,1,1 workspace=work_items,item_counts,handoff_states
launch kda_piece2 grid=H,1,1 threads=1,1,1 workspace=work_items,item_counts,handoff_states
*/

template <uint STAGE>
inline void kda_piece_body(device const half* q,
                           device const half* k,
                           device const half* v,
                           device const half* g,
                           device const half* beta,
                           device const float* A_log,
                           device const float* dt_bias,
                           device const float* initial_state,
                           device const float* scale,
                           device half* output,
                           device float* final_state,
                           device uint* work_items,
                           device int* item_counts,
                           device float* handoff_states,
                           uint head) {
    const int published = item_counts[0];
    if (published < 1 || published > 3 || int(STAGE) >= published) {
        return;
    }
    const uint begin = work_items[STAGE * 2];
    const uint count = work_items[STAGE * 2 + 1];
    float state[D * D];
    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            const uint index = head * D * D + vi * D + ki;
            state[vi * D + ki] = STAGE == 0
                ? initial_state[index]
                : handoff_states[(STAGE - 1) * H * D * D + index];
        }
    }

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

    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            const uint index = head * D * D + vi * D + ki;
            if (int(STAGE) + 1 == published) {
                final_state[index] = state[vi * D + ki];
            } else {
                handoff_states[STAGE * H * D * D + index] = state[vi * D + ki];
            }
        }
    }
}

#define KDA_PIECE_KERNEL(STAGE)                                                        \
kernel void kda_piece##STAGE(device const half* q [[buffer(0)]],                     \
                             device const half* k [[buffer(1)]],                     \
                             device const half* v [[buffer(2)]],                     \
                             device const half* g [[buffer(3)]],                     \
                             device const half* beta [[buffer(4)]],                  \
                             device const float* A_log [[buffer(5)]],                \
                             device const float* dt_bias [[buffer(6)]],              \
                             device const float* initial_state [[buffer(7)]],        \
                             device const float* scale [[buffer(8)]],                \
                             device half* output [[buffer(9)]],                      \
                             device float* final_state [[buffer(10)]],               \
                             device uint* work_items [[buffer(11)]],                 \
                             device int* item_counts [[buffer(12)]],                 \
                             device float* handoff_states [[buffer(13)]],            \
                             uint head [[thread_position_in_grid]]) {                 \
    kda_piece_body<STAGE>(q, k, v, g, beta, A_log, dt_bias, initial_state, scale,    \
                          output, final_state, work_items, item_counts,              \
                          handoff_states, head);                                     \
}

KDA_PIECE_KERNEL(0)
KDA_PIECE_KERNEL(1)
KDA_PIECE_KERNEL(2)
