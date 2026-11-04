#include <metal_stdlib>
using namespace metal;

// varlen_kda: naive token-by-token delta rule, one thread per (sequence, head).
// The fp32 V-first recurrent state lives in registers; each [begin, end) segment recurs
// independently and writes final_state[seq, head] at the segment end.
/* taic-launch
launch varlen_kda grid=num_seqs*H,1,1 threads=1,1,1
*/
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
            state[vi * D + ki] = initial_state[(seq * H + head) * D * D + vi * D + ki];
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

        // per-channel log-decay, then state *= exp(log_decay) along the key dim
        for (int ki = 0; ki < D; ki++) {
            float gate = float(g[(t * H + head) * D + ki]) + dt_bias[head * D + ki];
            float sig = 1.0f / (1.0f + exp(-decay_scale * gate));
            float log_decay = -5.0f * sig;
            for (int vi = 0; vi < D; vi++) {
                state[vi * D + ki] *= exp(log_decay);
            }
        }

        // delta residual (v - state @ k) * sigmoid(beta), then state += residual ⊗ k
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

        // output = state @ q
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
