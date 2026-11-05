/*
 * Candidate-owned generic varlen Host+Device example for the independent native64 ABI.
 *
 * This is not a harness planner and contains no CUDA/CANN mechanism.  It learns only the generic
 * ideas: cost-aware range partitioning, LPT chain assignment, bounded pieces, partial tails and
 * explicit state handoff.  A simple balanced mode is kept as a candidate-owned comparison.
 */
#define MAX_SPLIT 3u
#define WORKERS 4u
#define START_COST 230
#define HANDOFF_COST 102
#define COST_SHIFT 8
#define FLAG_EMPTY 0
#define FLAG_READY 0x5A5A5A5A
#ifndef SF_PLANNER_MODE
#define SF_PLANNER_MODE 2
#endif

static int piece_costs[18];
static int piece_chains[18];
static int chain_costs[6];
static int chain_order[6];
static int worker_load[4];
static uint32_t sorted_work_items[72];

static int validate(const SfNativeInputs *in, const SfNativeShapes *sh) {
    if (sh->num_seqs == 0u || sh->num_seqs > 6u || in->cu_seqlens_numel != sh->num_seqs + 1u) return 1;
    if (in->cu_seqlens[0] != 0 || in->cu_seqlens[sh->num_seqs] != sh->total_tokens) return 2;
    for (uint32_t s = 0; s < sh->num_seqs; ++s) {
        if (in->cu_seqlens[s + 1] < in->cu_seqlens[s]) return 3;
    }
    return 0;
}

static int assign_least_loaded(uint32_t chain) {
    uint32_t best = 0;
    for (uint32_t w = 1; w < WORKERS; ++w) {
        if (worker_load[w] < worker_load[best]) best = w;
    }
    worker_load[best] += chain_costs[chain];
    return (int)best;
}

int32_t host_tiling(const SfNativeInputs *in, const SfNativeShapes *sh,
                    const SfNativeTarget *target, SfNativeWorkspaces *ws,
                    SfNativeResolved *resolved) {
    (void)target;
    (void)resolved;
    int check = validate(in, sh);
    if (check != 0) return check;
    uint32_t seqs = sh->num_seqs;
    if (seqs * MAX_SPLIT * 4u > ws->work_items_capacity) return 10;
    if (WORKERS > ws->item_counts_capacity || WORKERS > ws->worker_offsets_capacity ||
        WORKERS > ws->worker_loads_capacity) return 11;
    if (seqs * (MAX_SPLIT - 1u) > ws->handoff_states_capacity ||
        seqs * (MAX_SPLIT - 1u) > ws->handoff_flags_capacity) return 12;

    for (uint32_t w = 0; w < WORKERS; ++w) {
        worker_load[w] = 0;
        ws->item_counts[w] = 0;
        ws->worker_offsets[w] = 0;
        ws->worker_loads[w] = 0;
    }

    /* Range-cut: no empty piece, at most MAX_SPLIT pieces, final piece absorbs the partial tail. */
    uint32_t pieces_total = 0;
    for (uint32_t s = 0; s < seqs; ++s) {
        uint32_t begin = (uint32_t)in->cu_seqlens[s];
        uint32_t len = (uint32_t)(in->cu_seqlens[s + 1] - in->cu_seqlens[s]);
        uint32_t pieces = len < MAX_SPLIT ? len : MAX_SPLIT;
        chain_costs[s] = 0;
        for (uint32_t slot = 0; slot < MAX_SPLIT - 1u; ++slot) {
            ws->handoff_flags[s * (MAX_SPLIT - 1u) + slot] = FLAG_EMPTY;
            ws->handoff_states[s * (MAX_SPLIT - 1u) + slot] = 0;
        }
        uint32_t base = pieces == 0u ? 0u : len / pieces;
        uint32_t rem = pieces == 0u ? 0u : len % pieces;
        uint32_t cursor = begin;
        for (uint32_t p = 0; p < pieces; ++p) {
            uint32_t count = base + (p < rem ? 1u : 0u);
            uint32_t item = pieces_total++;
            ws->work_items[item * 4u + 0u] = cursor;
            ws->work_items[item * 4u + 1u] = count;
            ws->work_items[item * 4u + 2u] = s | (p << 8) | (pieces << 16);
            ws->work_items[item * 4u + 3u] = 0u; /* worker assigned below */
            int cost = ((int)count << COST_SHIFT) + START_COST + (p + 1u < pieces ? HANDOFF_COST : 0);
            piece_costs[item] = cost;
            piece_chains[item] = (int)s;
            chain_costs[s] += cost;
            cursor += count;
        }
    }

    if (SF_PLANNER_MODE == 2) {
        /* LPT: longest cost chain first, whole chain to one least-loaded worker. */
        for (uint32_t s = 0; s < seqs; ++s) chain_order[s] = (int)s;
        for (uint32_t i = 1; i < seqs; ++i) {
            int value = chain_order[i];
            uint32_t j = i;
            while (j > 0u && chain_costs[chain_order[j - 1u]] < chain_costs[value]) {
                chain_order[j] = chain_order[j - 1u];
                --j;
            }
            chain_order[j] = value;
        }
        for (uint32_t i = 0; i < seqs; ++i) {
            uint32_t chain = (uint32_t)chain_order[i];
            if (chain_costs[chain] == 0) continue;
            int worker = assign_least_loaded(chain);
            for (uint32_t item = 0; item < pieces_total; ++item) {
                if (piece_chains[item] == (int)chain) ws->work_items[item * 4u + 3u] = (uint32_t)worker;
            }
        }
    } else {
        /* Candidate baseline: canonical chain order, round-robin whole-chain assignment. */
        for (uint32_t s = 0; s < seqs; ++s) {
            if (chain_costs[s] == 0) continue;
            int worker = assign_least_loaded(s);
            for (uint32_t item = 0; item < pieces_total; ++item) {
                if (piece_chains[item] == (int)s) ws->work_items[item * 4u + 3u] = (uint32_t)worker;
            }
        }
    }

    /* Stable worker partition; each chain's pieces retain their order within its worker. */
    uint32_t out = 0;
    for (uint32_t w = 0; w < WORKERS; ++w) {
        ws->worker_offsets[w] = (int32_t)out;
        uint32_t count = 0;
        for (uint32_t item = 0; item < pieces_total; ++item) {
            if (ws->work_items[item * 4u + 3u] != w) continue;
            for (uint32_t word = 0; word < 4u; ++word) {
                sorted_work_items[out * 4u + word] = ws->work_items[item * 4u + word];
            }
            ++out;
            ++count;
        }
        ws->item_counts[w] = (int32_t)count;
        ws->worker_loads[w] = worker_load[w];
    }
    for (uint32_t word = 0; word < out * 4u; ++word) ws->work_items[word] = sorted_work_items[word];
    ws->work_items_offset = 0;
    ws->work_items_count = pieces_total * 4u;
    ws->item_counts_offset = 0;
    ws->item_counts_count = WORKERS;
    ws->worker_offsets_offset = 0;
    ws->worker_offsets_count = WORKERS;
    ws->worker_loads_offset = 0;
    ws->worker_loads_count = WORKERS;
    ws->handoff_states_offset = 0;
    ws->handoff_states_count = 0; /* device-owned state, never host-published */
    ws->handoff_flags_offset = 0;
    ws->handoff_flags_count = seqs * (MAX_SPLIT - 1u);
    return 0;
}

/* A backend-neutral device consumer.  Metal will replace only this execution mechanism later. */
int32_t device_execute(const SfNativeInputs *in, const SfNativeShapes *sh,
                       const SfNativeWorkspaces *ws, int32_t *output, int32_t *final_state) {
    for (uint32_t s = 0; s < sh->num_seqs; ++s) final_state[s] = in->initial_state[s];
    for (uint32_t w = 0; w < WORKERS; ++w) {
        int32_t begin_item = ws->worker_offsets[w];
        int32_t count = ws->item_counts[w];
        for (int32_t local = 0; local < count; ++local) {
            uint32_t item = (uint32_t)(begin_item + local);
            uint32_t token = ws->work_items[item * 4u + 0u];
            uint32_t valid = ws->work_items[item * 4u + 1u];
            uint32_t packed = ws->work_items[item * 4u + 2u];
            uint32_t seq = packed & 0xffu;
            uint32_t piece = (packed >> 8) & 0xffu;
            uint32_t pieces = (packed >> 16) & 0xffu;
            int32_t state = piece == 0u ? in->initial_state[seq] : 0;
            if (piece != 0u) {
                uint32_t slot = seq * (MAX_SPLIT - 1u) + piece - 1u;
                if (ws->handoff_flags[slot] != FLAG_READY) return 100;
                state = ws->handoff_states[slot];
                ws->handoff_flags[slot] = FLAG_EMPTY;
            }
            /* Chunked loop with a real partial tail; no token escapes its piece/segment. */
            for (uint32_t base = 0; base < valid; base += 4u) {
                uint32_t hi = base + 4u < valid ? base + 4u : valid;
                for (uint32_t i = base; i < hi; ++i) {
                    state += in->tokens[token + i];
                    output[token + i] = state;
                }
            }
            if (piece + 1u < pieces) {
                uint32_t slot = seq * (MAX_SPLIT - 1u) + piece;
                ws->handoff_states[slot] = state;
                ws->handoff_flags[slot] = FLAG_READY;
            } else {
                final_state[seq] = state;
            }
        }
    }
    for (uint32_t s = 0; s < sh->num_seqs; ++s) {
        for (uint32_t slot = 0; slot < MAX_SPLIT - 1u; ++slot) {
            if (ws->handoff_flags[s * (MAX_SPLIT - 1u) + slot] != FLAG_EMPTY) return 101;
        }
    }
    return 0;
}
