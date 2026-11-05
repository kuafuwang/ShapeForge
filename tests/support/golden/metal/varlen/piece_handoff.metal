#include <metal_stdlib>
using namespace metal;

// V2 Dyn-0 fixture B: a *real* recurrent-state handoff between stages.
//
// Unlike fixture A (`prep_chain.metal`), the recurrence here is cut into pieces and the state of a
// piece is handed to the next stage through the declared `handoff_states` workspace, with a
// ready/consumed flag protocol in `handoff_flags`.  Stage boundaries are the real Metal serial
// encoder plus `memoryBarrierWithScope_(MTLBarrierScopeBuffers)`; nothing here spins on a flag from
// another threadgroup of the same launch.
//
// Storage is bounded by the OpSpec capacity, not by the number of chunks: a chain has at most
// MAX_SPLIT = 3 pieces and handoff_slots = num_seqs*H*(MAX_SPLIT-1), so
// handoff_bytes = slots*D*D*4 (see `tasks/varlen_kda/opspec.json`).  A candidate that stored one
// D-by-D state per 64-token chunk would exceed the declared capacity instead.
//
// stage 0 `varlen_plan` (single threadgroup, static grid 1):
//   reads the *actual* cu_seqlens, cuts each (sequence, head) chain into P = min(MAX_SPLIT, len)
//   pieces of span = ceil(len / P), writes the piece table, publishes item_counts[0] = #pieces,
//   resets the chain's handoff flags to FLAG_EMPTY, and materialises
//   final_state[seq,head] = initial_state[seq,head] for an empty segment (an empty sequence never
//   advances state).
//   Item layout (2 u32 words, candidate-owned):
//     word0 = token_begin | valid_rows << 16
//     word1 = chain | piece << 8 | piece_count << 16        (chain < 256, piece_count <= 3)
// stage k `varlen_piece<k>` (k = 0,1,2; static grid = num_seqs*H*MAX_SPLIT):
//   every threadgroup reads the runtime item count and returns when it has no item; it also
//   returns unless the item belongs to piece k, so a chain with fewer pieces simply leaves the
//   later stages idle (Dyn-0: static stage dispatch, runtime data).
//   * first piece: state <- initial_state[seq, head]
//   * middle piece: requires FLAG_READY on its incoming slot, loads the handoff state, then clears
//     the flag (a consumed handoff must not survive), and publishes its own terminal state
//   * last piece: consumes likewise, verifies that every flag of the chain is FLAG_EMPTY after
//     consumption, and writes final_state[seq, head]
//   The recurrence walks exactly [begin, begin+valid) in BT-token chunks with a masked partial
//   tail, so the tail can neither advance state nor write output past cu_seqlens[seq+1].
//
// LOWER_BOUND is this task's declared attr value (-5); the fixture is a KDA-task candidate.

//: Maximum pieces per chain, matching the declared `handoff_states` / `work_items` capacity.
constant constexpr uint MAX_SPLIT = 3u;
//: Fixture chunk size; the last chunk of a piece is masked to `valid` rows.
constant constexpr uint BT = 4u;
//: Handoff flag values (candidate-owned encoding; the plan publishes EMPTY, consumers clear it).
constant constexpr int FLAG_EMPTY = 0;
constant constexpr int FLAG_READY = 0x5A5A5A5A;
//: Value a missing publish / uncleared flag is turned into, so the correctness oracle rejects it.
constant constexpr float PROTOCOL_POISON = 1.0e30f;

/* taic-launch
math safe
workspace work_items num_seqs*H*3*2*4
workspace item_counts num_seqs*H*3*4
workspace handoff_flags num_seqs*H*4*4
workspace handoff_states num_seqs*H*2*D*D*4
launch varlen_plan grid=1,1,1 threads=1,1,1 workspace=work_items,item_counts,handoff_flags
launch varlen_piece0 grid=num_seqs*H*3,1,1 threads=1,1,1 workspace=work_items,item_counts,handoff_flags,handoff_states
launch varlen_piece1 grid=num_seqs*H*3,1,1 threads=1,1,1 workspace=work_items,item_counts,handoff_flags,handoff_states
launch varlen_piece2 grid=num_seqs*H*3,1,1 threads=1,1,1 workspace=work_items,item_counts,handoff_flags,handoff_states
*/

kernel void varlen_plan(device const float* initial_state [[buffer(0)]],
                        device const int* cu_seqlens [[buffer(1)]],
                        device float* final_state [[buffer(2)]],
                        device uint* work_items [[buffer(3)]],
                        device int* item_counts [[buffer(4)]],
                        device int* handoff_flags [[buffer(5)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) {
        return;
    }
    int item = 0;
    for (uint chain = 0; chain < num_seqs * H; chain++) {
        uint seq = chain / H;
        uint head = chain % H;
        int begin = cu_seqlens[seq];
        int end = cu_seqlens[seq + 1];
        uint len = uint(end - begin);
        // One handoff flag per possible boundary of this chain.
        for (uint slot = 0; slot < MAX_SPLIT - 1u; slot++) {
            handoff_flags[chain * (MAX_SPLIT - 1u) + slot] = FLAG_EMPTY;
        }
        if (len == 0u) {
            // An empty sequence keeps its initial state; no piece is planned for it.
            for (uint vi = 0; vi < D; vi++) {
                for (uint ki = 0; ki < D; ki++) {
                    final_state[(seq * H + head) * D * D + vi * D + ki] =
                        initial_state[(seq * H + head) * D * D + vi * D + ki];
                }
            }
            continue;
        }
        uint pieces = min(MAX_SPLIT, len);
        uint span = (len + pieces - 1u) / pieces;
        for (uint piece = 0; piece < pieces; piece++) {
            uint piece_begin = uint(begin) + piece * span;
            uint piece_end = min(uint(end), uint(begin) + (piece + 1u) * span);
            work_items[item * 2 + 0] = piece_begin | ((piece_end - piece_begin) << 16);
            work_items[item * 2 + 1] = chain | (piece << 8) | (pieces << 16);
            item++;
        }
    }
    item_counts[0] = item;
}

template <uint STAGE>
inline void piece_body(device const half* q,
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
                       device int* handoff_flags,
                       device float* handoff_states,
                       uint tg) {
    // Dyn-0: static grid, runtime count.  A threadgroup without an item exits immediately.
    int count = item_counts[0];
    if (int(tg) >= count) {
        return;
    }
    uint word0 = work_items[tg * 2 + 0];
    uint word1 = work_items[tg * 2 + 1];
    uint begin = word0 & 0xFFFFu;
    uint valid = word0 >> 16;
    uint chain = word1 & 0xFFu;
    uint piece = (word1 >> 8) & 0xFFu;
    uint pieces = (word1 >> 16) & 0xFFu;
    if (piece != STAGE) {
        // Another stage owns this item; this stage stays idle instead of waiting on it.
        return;
    }
    uint seq = chain / H;
    uint head = chain % H;

    float state[D * D];
    if (piece == 0u) {
        // First piece: the source of the state is the contract input.
        for (uint vi = 0; vi < D; vi++) {
            for (uint ki = 0; ki < D; ki++) {
                state[vi * D + ki] = initial_state[(seq * H + head) * D * D + vi * D + ki];
            }
        }
    } else {
        uint slot = chain * (MAX_SPLIT - 1u) + piece - 1u;
        if (handoff_flags[slot] != FLAG_READY) {
            // A missing publish (or a consumer dispatched before its producer) is a hard error.
            for (uint vi = 0; vi < D; vi++) {
                for (uint ki = 0; ki < D; ki++) {
                    state[vi * D + ki] = PROTOCOL_POISON;
                }
            }
        } else {
            for (uint vi = 0; vi < D; vi++) {
                for (uint ki = 0; ki < D; ki++) {
                    state[vi * D + ki] = handoff_states[slot * D * D + vi * D + ki];
                }
            }
            // Consume: the flag must not survive this stage.
            handoff_flags[slot] = FLAG_EMPTY;
        }
    }

    float scale_value = scale[0];
    int t0 = int(begin);
    int t1 = int(begin + valid);
    for (int base = t0; base < t1; base += int(BT)) {
        int hi = min(base + int(BT), t1);  // masked tail: never past this segment's end
        for (int t = base; t < hi; t++) {
            float qv[D], kv[D];
            float q_norm = 0.0f, k_norm = 0.0f;
            for (uint d = 0; d < D; d++) {
                qv[d] = float(q[(uint(t) * H + head) * D + d]);
                kv[d] = float(k[(uint(t) * H + head) * D + d]);
                q_norm += qv[d] * qv[d];
                k_norm += kv[d] * kv[d];
            }
            float q_scale = rsqrt(q_norm + 1e-6f) * scale_value;
            float k_scale = rsqrt(k_norm + 1e-6f);
            for (uint d = 0; d < D; d++) {
                qv[d] *= q_scale;
                kv[d] *= k_scale;
            }
            float decay_scale = exp(A_log[head]);
            for (uint ki = 0; ki < D; ki++) {
                float gate = float(g[(uint(t) * H + head) * D + ki]) + dt_bias[head * D + ki];
                float log_decay = -5.0f * (1.0f / (1.0f + exp(-decay_scale * gate)));
                float decay = exp(log_decay);
                for (uint vi = 0; vi < D; vi++) {
                    state[vi * D + ki] *= decay;
                }
            }
            float beta_value = 1.0f / (1.0f + exp(-float(beta[uint(t) * H + head])));
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
    }

    if (piece + 1u < pieces) {
        // Middle piece: publish the terminal state, then the flag that marks it ready.
        uint slot = chain * (MAX_SPLIT - 1u) + piece;
        for (uint vi = 0; vi < D; vi++) {
            for (uint ki = 0; ki < D; ki++) {
                handoff_states[slot * D * D + vi * D + ki] = state[vi * D + ki];
            }
        }
        handoff_flags[slot] = FLAG_READY;
    } else {
        // Last piece: every handoff of this chain must have been consumed by now.  An uncleared
        // flag means a consumer skipped its clear, so the terminal state is refused.
        bool clean = true;
        for (uint slot = 0; slot < MAX_SPLIT - 1u; slot++) {
            if (handoff_flags[chain * (MAX_SPLIT - 1u) + slot] != FLAG_EMPTY) {
                clean = false;
            }
        }
        for (uint vi = 0; vi < D; vi++) {
            for (uint ki = 0; ki < D; ki++) {
                final_state[(seq * H + head) * D * D + vi * D + ki] =
                    clean ? state[vi * D + ki] : PROTOCOL_POISON;
            }
        }
    }
}

kernel void varlen_piece0(device const half* q [[buffer(0)]],
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
                          device int* handoff_flags [[buffer(13)]],
                          device float* handoff_states [[buffer(14)]],
                          uint tg [[thread_position_in_grid]]) {
    piece_body<0u>(q, k, v, g, beta, A_log, dt_bias, initial_state, scale, output, final_state,
                   work_items, item_counts, handoff_flags, handoff_states, tg);
}

kernel void varlen_piece1(device const half* q [[buffer(0)]],
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
                          device int* handoff_flags [[buffer(13)]],
                          device float* handoff_states [[buffer(14)]],
                          uint tg [[thread_position_in_grid]]) {
    piece_body<1u>(q, k, v, g, beta, A_log, dt_bias, initial_state, scale, output, final_state,
                   work_items, item_counts, handoff_flags, handoff_states, tg);
}

kernel void varlen_piece2(device const half* q [[buffer(0)]],
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
                          device int* handoff_flags [[buffer(13)]],
                          device float* handoff_states [[buffer(14)]],
                          uint tg [[thread_position_in_grid]]) {
    piece_body<2u>(q, k, v, g, beta, A_log, dt_bias, initial_state, scale, output, final_state,
                   work_items, item_counts, handoff_flags, handoff_states, tg);
}
