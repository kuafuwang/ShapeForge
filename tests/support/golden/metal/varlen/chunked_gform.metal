#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// Candidate-owned D_DIM=128 G-form KDA chunker.  One threadgroup owns one (sequence, head, v-tile)
// chain and walks its chunks serially, so no cross-threadgroup handoff is needed.  The matrix
// algebra is the generic G-form identity, not a CUDA/CANN ABI or a harness scheduling rule.
constant constexpr uint D_DIM = 128;
constant constexpr uint BT = 8;
constant constexpr uint VT = 8;
constant constexpr uint VGROUPS = D_DIM / VT;

/* taic-launch
math safe
launch kda_chunked grid=num_seqs*H*16,1,1 threads=32,1,1
*/

static inline void mma8(threadgroup const float *A,
                        threadgroup const float *B,
                        threadgroup float *C,
                        uint lane) {
    (void)lane;
    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
    for (uint kb = 0; kb < D_DIM; kb += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, &A[kb], D_DIM);
        simdgroup_load(b, &B[kb], D_DIM, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    simdgroup_store(acc, C, 8);
}

kernel void kda_chunked(device const half* q [[buffer(0)]],
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
                        uint tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float state[VT][D_DIM];
    threadgroup float qt[BT][D_DIM];
    threadgroup float kt[BT][D_DIM];
    threadgroup float vt[BT][D_DIM];
    threadgroup float ld[BT][D_DIM];
    threadgroup float kd[BT][D_DIM];
    threadgroup float rawk[BT][D_DIM];
    threadgroup float rawq[BT][D_DIM];
    threadgroup float A[BT][BT];
    threadgroup float G[BT][BT];
    threadgroup float Base[VT][BT];
    threadgroup float BaseQ[VT][BT];
    threadgroup float residual[VT][BT];
    threadgroup float beta_value[BT];
    threadgroup float qnorm[BT];
    threadgroup float knorm[BT];

    const uint vgroup = tg % VGROUPS;
    const uint head = (tg / VGROUPS) % H;
    const uint seq = tg / (VGROUPS * H);
    const uint state_base = ((seq * H + head) * D_DIM + vgroup * VT) * D_DIM;
    for (uint index = lane; index < VT * D_DIM; index += 32) {
        state[index / D_DIM][index % D_DIM] = initial_state[state_base + index];
    }

    const int begin = cu_seqlens[seq];
    const int end = cu_seqlens[seq + 1];
    const float scale_value = scale[0];
    const float decay_scale = exp(A_log[head]);

    for (int chunk = begin; chunk < end; chunk += BT) {
        const uint valid = min(uint(BT), uint(end - chunk));
        for (uint t = 0; t < BT; ++t) {
            const int token = chunk + int(t);
            if (lane == 0) {
                float qs = 0.0f;
                float ks = 0.0f;
                if (t < valid) {
                    for (uint j = 0; j < D_DIM; ++j) {
                        const uint index = (uint(token) * H + head) * D_DIM + j;
                        const float qraw = float(q[index]);
                        const float kraw = float(k[index]);
                        qs += qraw * qraw;
                        ks += kraw * kraw;
                    }
                }
                qnorm[t] = qs;
                knorm[t] = ks;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = lane; j < D_DIM; j += 32) {
                if (t < valid) {
                    const uint index = (uint(token) * H + head) * D_DIM + j;
                    const float qraw = float(q[index]);
                    const float kraw = float(k[index]);
                    const float qv = qraw * rsqrt(qnorm[t] + 1e-6f) * scale_value;
                    const float kv = kraw * rsqrt(knorm[t] + 1e-6f);
                    const float gate = float(g[index]) + dt_bias[head * D_DIM + j];
                    const float log_decay = -5.0f * (1.0f / (1.0f + exp(-decay_scale * gate)));
                    const float previous = t == 0 ? 0.0f : ld[t - 1][j];
                    ld[t][j] = previous + log_decay;
                    // Kbar = K exp(-L); KD/QD = K/Q exp(L) (the oracle decays
                    // before evaluating the residual, so the current token is included).
                    rawk[t][j] = kv;
                    rawq[t][j] = qv;
                    kt[t][j] = kv * exp(-ld[t][j]);
                    kd[t][j] = kv * exp(ld[t][j]);
                    qt[t][j] = qv * exp(ld[t][j]);
                    vt[t][j] = float(v[index]);
                } else {
                    qt[t][j] = 0.0f;
                    kt[t][j] = 0.0f;
                    kd[t][j] = 0.0f;
                    vt[t][j] = 0.0f;
                    ld[t][j] = 0.0f;
                }
            }
            if (lane == 0) {
                beta_value[t] = 0.0f;
                if (t < valid) {
                    const float b = float(beta[uint(token) * H + head]);
                    beta_value[t] = 1.0f / (1.0f + exp(-b));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma8(&kt[0][0], &kd[0][0], &A[0][0], lane);
        mma8(&state[0][0], &kd[0][0], &Base[0][0], lane);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Complete lower-triangular recurrence, not an output-only shortcut:
        // r_t = beta_t * (v_t - Base_t - sum_{s<t} A_st r_s).
        for (uint row = lane; row < VT; row += 32) {
            for (uint t = 0; t < BT; ++t) {
                float history = 0.0f;
                float compensation = 0.0f;
                for (uint s = 0; s < t; ++s) {
                    const float term = residual[row][s] * A[s][t];
                    const float adjusted = term - compensation;
                    const float next = history + adjusted;
                    compensation = (next - history) - adjusted;
                    history = next;
                }
                residual[row][t] = beta_value[t] * (vt[t][vgroup * VT + row] - Base[row][t] - history);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply the G-form residuals with the original fp32 recurrence.  This avoids carrying
        // another independently rounded G-form state sum: state/output now follow the oracle's
        // exact token order, while A/Base above still use real simdgroup matrix products.
        for (uint t = 0; t < valid; ++t) {
            for (uint index = lane; index < VT * D_DIM; index += 32) {
                const uint row = index / D_DIM;
                const uint j = index % D_DIM;
                const float log_decay = ld[t][j] - (t == 0 ? 0.0f : ld[t - 1][j]);
                state[row][j] = state[row][j] * exp(log_decay) + residual[row][t] * rawk[t][j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint row = lane; row < VT; row += 32) {
                float result = 0.0f;
                for (uint j = 0; j < D_DIM; ++j) {
                    result += state[row][j] * rawq[t][j];
                }
                output[(uint(chunk + int(t)) * H + head) * D_DIM + vgroup * VT + row] = half(result);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint index = lane; index < VT * D_DIM; index += 32) {
        final_state[state_base + index] = state[index / D_DIM][index % D_DIM];
    }
}
