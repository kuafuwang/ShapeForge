#include <metal_stdlib>
using namespace metal;

// V2 Dyn-0 fixture A: a preprocessing stage plus one sequential chain per (sequence, head).
//
// This fixture proves *multi-stage preprocessing* + partial tail + per-sequence terminal state.
// It is deliberately NOT a piece-handoff fixture: stage 2 runs the whole segment
// [cu_seqlens[seq], cu_seqlens[seq+1]) in one threadgroup, so no recurrent state ever crosses a
// stage boundary.  Fixture B (`piece_handoff.metal`) is the one that hands state between stages.
//
// Geometry is static (Dyn-0): the grid, the two stages and their argument slots are fixed at
// build time.  Nothing here binds a runtime scalar or asks the harness for a dynamic launch.
//
// stage 1 `varlen_prep` (one thread per (token, head)):
//   token_stats[t,h]     = { |q|^2, |k|^2, sigmoid(beta) }        (fp32, 3 values)
//   log_decay_tab[t,h,:] = LOWER_BOUND * sigmoid(exp(A_log[h]) * (g + dt_bias[h,:]))
//   the first num_seqs*H threads also build the work list from the *actual* cu_seqlens:
//     word 0 = token begin, word 1 = valid_rows<<16 | head<<8 | seq
//   and thread 0 publishes item_counts[0] = num_seqs*H, the runtime count stage 2 reads.
//
// stage 2 `varlen_chain` (static grid = num_seqs*H*MAX_SPLIT, the ABI capacity):
//   every threadgroup reads the runtime count and returns when it has no item, so the two extra
//   threadgroups per chain (capacity has room for three pieces) exit immediately.  The item table
//   is copied into threadgroup memory when it fits and read from global memory otherwise.  The
//   recurrence walks exactly [begin, begin+valid), so a segment whose length is not a multiple of
//   the fixture's chunk never advances state past cu_seqlens[seq+1].
//
// LOWER_BOUND is this task's declared attr value (-5); the fixture is a KDA-task candidate, not a
// general task-agnostic planner.

//: Capacity of the threadgroup item-table cache, in items (2 u32 words each = 512 B).
constant constexpr uint TG_ITEM_CAP = 64u;
//: Threads per threadgroup in stage 2.  One thread owns the recurrence; all threads cooperate on
//: the table copy so the cache path is a real threadgroup copy, not a comment.
constant constexpr uint CHAIN_THREADS = 32u;

/* taic-launch
math safe
workspace token_stats total_tokens*H*3*4
workspace log_decay_tab total_tokens*H*D*4
workspace work_items num_seqs*H*3*2*4
workspace item_counts num_seqs*H*3*4
launch varlen_prep grid=total_tokens*H,1,1 threads=1,1,1 workspace=token_stats,log_decay_tab,work_items,item_counts
launch varlen_chain grid=num_seqs*H*3,1,1 threads=32,1,1 workspace=work_items,item_counts,token_stats,log_decay_tab
*/

kernel void varlen_prep(device const half* q [[buffer(0)]],
                        device const half* k [[buffer(1)]],
                        device const half* g [[buffer(2)]],
                        device const half* beta [[buffer(3)]],
                        device const float* A_log [[buffer(4)]],
                        device const float* dt_bias [[buffer(5)]],
                        device const int* cu_seqlens [[buffer(6)]],
                        device float* token_stats [[buffer(7)]],
                        device float* log_decay_tab [[buffer(8)]],
                        device uint* work_items [[buffer(9)]],
                        device int* item_counts [[buffer(10)]],
                        uint i [[thread_position_in_grid]]) {
    uint t = i / H;
    uint head = i % H;
    uint lane = t * H + head;

    // Per-token, per-head statistics used by the recurrence stage.
    float q_norm = 0.0f, k_norm = 0.0f;
    for (uint d = 0; d < D; d++) {
        float qv = float(q[lane * D + d]);
        float kv = float(k[lane * D + d]);
        q_norm += qv * qv;
        k_norm += kv * kv;
    }
    token_stats[lane * 3 + 0] = q_norm;
    token_stats[lane * 3 + 1] = k_norm;
    token_stats[lane * 3 + 2] = 1.0f / (1.0f + exp(-float(beta[t * H + head])));
    float decay_scale = exp(A_log[head]);
    for (uint d = 0; d < D; d++) {
        float gate = float(g[lane * D + d]) + dt_bias[head * D + d];
        log_decay_tab[lane * D + d] = -5.0f * (1.0f / (1.0f + exp(-decay_scale * gate)));
    }

    // Device-side work list: chain i is (seq = i / H, head = i % H) for i < num_seqs*H.  Every
    // writer owns its own slot, so this is not a cross-threadgroup dependency.
    if (i == 0u) {
        item_counts[0] = int(num_seqs * H);
    }
    if (i < num_seqs * H) {
        uint seq = i / H;
        int begin = cu_seqlens[seq];
        int end = cu_seqlens[seq + 1];
        work_items[i * 2 + 0] = uint(begin);
        work_items[i * 2 + 1] = (uint(end - begin) << 16) | (head << 8) | seq;
    }
}

kernel void varlen_chain(device const half* q [[buffer(0)]],
                         device const half* k [[buffer(1)]],
                         device const half* v [[buffer(2)]],
                         device const float* initial_state [[buffer(3)]],
                         device const float* scale [[buffer(4)]],
                         // Workspace tensors are declared `access: write` in the OpSpec (they are
                         // scratch, never a reference result); a consuming stage still binds them
                         // as non-const `device` pointers, which the ABI accepts.
                         device float* token_stats [[buffer(5)]],
                         device float* log_decay_tab [[buffer(6)]],
                         device uint* work_items [[buffer(7)]],
                         device int* item_counts [[buffer(8)]],
                         device half* output [[buffer(9)]],
                         device float* final_state [[buffer(10)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]]) {
    // Dyn-0: the grid is static (the ABI has room for MAX_SPLIT pieces per chain); a threadgroup
    // with no item reads the runtime count written by stage 1 and returns.
    int count = item_counts[0];
    if (int(tg) >= count) {
        return;
    }

    // Small table -> threadgroup cache; large table -> global fallback.  All threads of the
    // threadgroup take the same branch (count is uniform), so the barrier is reachable by every
    // thread.
    threadgroup uint table[TG_ITEM_CAP * 2];
    bool cached = count <= int(TG_ITEM_CAP);
    if (cached) {
        for (uint slot = tid; slot < uint(count) * 2; slot += CHAIN_THREADS) {
            table[slot] = work_items[slot];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid != 0u) {
        return;
    }

    uint word0 = cached ? table[tg * 2 + 0] : work_items[tg * 2 + 0];
    uint word1 = cached ? table[tg * 2 + 1] : work_items[tg * 2 + 1];
    uint seq = word1 & 0xFFu;
    uint head = (word1 >> 8) & 0xFFu;
    uint valid = word1 >> 16;

    float state[D * D];
    for (uint vi = 0; vi < D; vi++) {
        for (uint ki = 0; ki < D; ki++) {
            state[vi * D + ki] = initial_state[(seq * H + head) * D * D + vi * D + ki];
        }
    }

    float scale_value = scale[0];
    int begin = int(word0);
    int end = begin + int(valid);  // exactly this segment's tokens: no rounding into the next one
    for (int t = begin; t < end; t++) {
        float qv[D], kv[D];
        float q_scale = rsqrt(token_stats[(uint(t) * H + head) * 3 + 0] + 1e-6f) * scale_value;
        float k_scale = rsqrt(token_stats[(uint(t) * H + head) * 3 + 1] + 1e-6f);
        for (uint d = 0; d < D; d++) {
            qv[d] = float(q[(uint(t) * H + head) * D + d]) * q_scale;
            kv[d] = float(k[(uint(t) * H + head) * D + d]) * k_scale;
        }
        for (uint ki = 0; ki < D; ki++) {
            float decay = exp(log_decay_tab[(uint(t) * H + head) * D + ki]);
            for (uint vi = 0; vi < D; vi++) {
                state[vi * D + ki] *= decay;
            }
        }
        float beta_value = token_stats[(uint(t) * H + head) * 3 + 2];
        for (uint vi = 0; vi < D; vi++) {
            float residual = float(v[(uint(t) * H + head) * D + vi]);
            for (uint ki = 0; ki < D; ki++) {
                residual -= state[vi * D + ki] * kv[ki];
            }
            residual *= beta_value;
            for (uint ki = 0; ki < D; ki++) {
                state[vi * D + ki] += residual * kv[ki];
            }
        }
        for (uint vi = 0; vi < D; vi++) {
            float out = 0.0f;
            for (uint ki = 0; ki < D; ki++) {
                out += state[vi * D + ki] * qv[ki];
            }
            output[(uint(t) * H + head) * D + vi] = half(out);
        }
    }

    for (uint vi = 0; vi < D; vi++) {
        for (uint ki = 0; ki < D; ki++) {
            final_state[(seq * H + head) * D * D + vi * D + ki] = state[vi * D + ki];
        }
    }
}
