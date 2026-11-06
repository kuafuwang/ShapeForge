#include <metal_stdlib>
using namespace metal;

// Independent fixed-shape reference candidate.  One thread owns one head's complete fp32 state.
kernel void kda_forward(device const half* q [[buffer(0)]],
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
                        uint head [[thread_position_in_grid]]) {
    float state[D * D];
    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            state[vi * D + ki] = initial_state[head * D * D + vi * D + ki];
        }
    }
    const float scale_value = scale[0];
    const float decay_scale = exp(A_log[head]);
    for (uint t = 0; t < N; ++t) {
        float qv[D], kv[D];
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (uint j = 0; j < D; ++j) {
            const uint index = (t * H + head) * D + j;
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
            const float gate = float(g[(t * H + head) * D + j]) + dt_bias[head * D + j];
            const float log_decay = -5.0f * (1.0f / (1.0f + exp(-decay_scale * gate)));
            for (uint vi = 0; vi < D; ++vi) {
                state[vi * D + j] *= exp(log_decay);
            }
        }
        const float beta_value = 1.0f / (1.0f + exp(-float(beta[t * H + head])));
        for (uint vi = 0; vi < D; ++vi) {
            float residual = float(v[(t * H + head) * D + vi]);
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
            output[(t * H + head) * D + vi] = half(result);
        }
    }
    for (uint vi = 0; vi < D; ++vi) {
        for (uint ki = 0; ki < D; ++ki) {
            final_state[head * D * D + vi * D + ki] = state[vi * D + ki];
        }
    }
}

/* taic-launch
math safe
launch kda_forward grid=H,1,1 threads=1,1,1
*/
