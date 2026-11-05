/*
 * Candidate-owned KDA-style host planner for the restricted C99 / Wasm Host Program (V3).
 *
 * This file is a *candidate fixture*, not a harness rule.  It has no libc, no includes and no
 * host pointers: the harness prepends the generated `host_contract.h` (typed inputs/target/
 * workspaces + the whitelisted read imports) and compiles it with standard Clang + wasm-ld.
 *
 * Two candidate modes share one ABI (a build-time constant owned by the candidate):
 *
 *   SF_PLANNER_MODE 1 (default, the device delivery)
 *     balanced split: `pieces = min(MAX_SPLIT, len)`, `base, rem = divmod(len, pieces)`, first
 *     `rem` pieces get one extra token; every piece is non-empty.  Items are emitted in canonical
 *     `(chain, piece)` order and packed as two u32 words exactly like the Dyn-0 Metal fixture:
 *       word0 = token_begin | valid_rows << 16
 *       word1 = chain | piece << 8 | pieces << 16
 *     It also resets the chain's handoff flags to FLAG_EMPTY.  The device stages then run with
 *     the same static grid/parity as V2; this is the real host+device example.
 *
 *   SF_PLANNER_MODE 2 (planner exercise)
 *     a fixed-point cost model (cost = tokens*256 + START_COST + HANDOFF_COST for a non-last
 *     piece), **range-cut** into at most MAX_SPLIT pieces by a cost target, then **LPT** of whole
 *     chains onto at most 8 workers with a binary max-heap and first-fit tie-breaking.  Items are
 *     emitted grouped by worker, so one worker owns a chain's whole piece sequence.  This mode is
 *     never fed to the device fixture: it demonstrates the planner algorithms the ABI can carry,
 *     and the CPU test compares its makespan with a round-robin assignment.
 *
 * The returned span is the only thing the harness sees.  On any capacity/bounds problem the
 * planner returns a non-zero code *before* the harness copies anything; a failed run has no
 * partial effect.
 */

/* Candidate-owned constants (part of the candidate's algorithm, not the ABI). */
#define SF_PLANNER_MAX_SPLIT 3u
#define SF_PLANNER_START_COST 230 /* 0.9 in 1/256 fixed point */
#define SF_PLANNER_HANDOFF_COST 102 /* 0.4 in 1/256 fixed point */
#define SF_PLANNER_COST_SHIFT 8
#define SF_PLANNER_MAX_WORKERS 8u
/* Dyn-0 fixture flag values: the host publishes the empty state, consumers clear after consuming. */
#define SF_PLANNER_FLAG_EMPTY 0

#ifndef SF_PLANNER_MODE
#define SF_PLANNER_MODE 1
#endif

/* Static planner scratch, sized from the generated contract macros (the ABI capacities). */
#define SF_MAX_ITEMS (SF_WS_WORK_ITEMS_CAPACITY_ELEMS / 2u)
#define SF_MAX_CHAINS (SF_WS_ITEM_COUNTS_CAPACITY_ELEMS / SF_PLANNER_MAX_SPLIT)
#define SF_MAX_WORKERS_EFFECTIVE \
    (SF_PLANNER_MAX_WORKERS < SF_MAX_CHAINS ? SF_PLANNER_MAX_WORKERS : SF_MAX_CHAINS)

static uint32_t sf_piece_word0[SF_MAX_ITEMS];
static uint32_t sf_piece_word1[SF_MAX_ITEMS];
static int64_t sf_piece_cost[SF_MAX_ITEMS];
static uint32_t sf_piece_chain[SF_MAX_ITEMS];
static uint32_t sf_piece_index[SF_MAX_ITEMS];
static uint32_t sf_piece_worker[SF_MAX_ITEMS];
static int64_t sf_chain_cost[SF_MAX_CHAINS];
static int64_t sf_worker_load[SF_MAX_WORKERS_EFFECTIVE];
static uint32_t sf_chain_heap[SF_MAX_CHAINS];

static uint32_t sf_min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

/* Strictly validate the runtime boundaries.  The harness validates them too; a candidate that
 * skips the check would still be refused, but returning early keeps the planner's own arithmetic
 * from depending on garbage. */
static int32_t sf_validate(const SfHostInputs *inputs, const SfHostShapes *shapes) {
    int32_t *cu = inputs->cu_seqlens;
    uint32_t num_seqs = shapes->num_seqs;
    uint32_t h = shapes->H;
    uint32_t token = 0u;
    if (num_seqs == 0u || h == 0u) {
        return SF_HOST_INVALID_BOUNDS;
    }
    if (inputs->cu_seqlens_numel != num_seqs + 1u) {
        return SF_HOST_INVALID_BOUNDS;
    }
    if (cu[0] != 0) {
        return SF_HOST_INVALID_BOUNDS;
    }
    for (uint32_t s = 0u; s < num_seqs; s++) {
        if (cu[s] < 0 || cu[s + 1] < cu[s]) {
            return SF_HOST_INVALID_BOUNDS;
        }
        token += (uint32_t)(cu[s + 1] - cu[s]);
    }
    if (token != shapes->total_tokens) {
        return SF_HOST_INVALID_BOUNDS;
    }
    return SF_HOST_OK;
}

static int32_t sf_check_capacity(uint32_t chains, uint32_t items) {
    if (chains > SF_MAX_CHAINS) {
        return SF_HOST_CAPACITY_EXCEEDED;
    }
    if (items > SF_MAX_ITEMS) {
        return SF_HOST_CAPACITY_EXCEEDED;
    }
    if (chains * (SF_PLANNER_MAX_SPLIT - 1u) > SF_WS_HANDOFF_FLAGS_CAPACITY_ELEMS) {
        return SF_HOST_CAPACITY_EXCEEDED;
    }
    return SF_HOST_OK;
}

/* Publish the valid span of a workspace.  `items` is in elements, never in bytes. */
static void sf_publish(SfHostWorkspaces *workspaces, uint32_t item_count,
                       uint32_t flag_slots) {
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = item_count * 2u;
    workspaces->item_counts_offset = 0u;
    workspaces->item_counts_count = 1u;
    workspaces->handoff_flags_offset = 0u;
    workspaces->handoff_flags_count = flag_slots;
}

static int32_t sf_plan_canonical(const SfHostInputs *inputs, const SfHostShapes *shapes,
                                 SfHostWorkspaces *workspaces) {
    int32_t *cu = inputs->cu_seqlens;
    uint32_t chains = shapes->num_seqs * shapes->H;
    uint32_t item = 0u;
    if (chains > 0x100u) {
        /* the fixture's word packing stores chain in 8 bits; refuse instead of truncating */
        return SF_HOST_CAPACITY_EXCEEDED;
    }
    for (uint32_t chain = 0u; chain < chains; chain++) {
        uint32_t seq = chain / shapes->H;
        uint32_t begin = (uint32_t)cu[seq];
        uint32_t len = (uint32_t)(cu[seq + 1] - cu[seq]);
        for (uint32_t slot = 0u; slot < SF_PLANNER_MAX_SPLIT - 1u; slot++) {
            workspaces->handoff_flags[chain * (SF_PLANNER_MAX_SPLIT - 1u) + slot] = SF_PLANNER_FLAG_EMPTY;
        }
        if (len == 0u) {
            continue;
        }
        uint32_t pieces = sf_min_u32(SF_PLANNER_MAX_SPLIT, len);
        uint32_t base = len / pieces;
        uint32_t rem = len % pieces;
        uint32_t cursor = begin;
        if (item + pieces > SF_MAX_ITEMS) {
            return SF_HOST_CAPACITY_EXCEEDED;
        }
        for (uint32_t p = 0u; p < pieces; p++) {
            uint32_t size = base + (p < rem ? 1u : 0u);
            if (cursor >= 0x10000u || size >= 0x10000u || pieces >= 0x100u) {
                return SF_HOST_CAPACITY_EXCEEDED;
            }
            workspaces->work_items[item * 2u + 0u] = cursor | (size << 16);
            workspaces->work_items[item * 2u + 1u] = chain | (p << 8) | (pieces << 16);
            cursor += size;
            item++;
        }
    }
    workspaces->item_counts[0] = (int32_t)item;
    sf_publish(workspaces, item, chains * (SF_PLANNER_MAX_SPLIT - 1u));
    return SF_HOST_OK;
}

/* --- mode 2: fixed-point range-cut + LPT ------------------------------------------------- */

static void sf_heap_push(uint32_t *heap, uint32_t *size, uint32_t value) {
    /* max-heap by sf_chain_cost, ties broken by the lower chain id (deterministic). */
    uint32_t at = *size;
    heap[at] = value;
    while (at > 0u) {
        uint32_t parent = (at - 1u) / 2u;
        int64_t a = sf_chain_cost[heap[parent]];
        int64_t b = sf_chain_cost[heap[at]];
        if (a > b || (a == b && heap[parent] < heap[at])) {
            break;
        }
        uint32_t tmp = heap[parent];
        heap[parent] = heap[at];
        heap[at] = tmp;
        at = parent;
    }
    *size += 1u;
}

static uint32_t sf_heap_pop(uint32_t *heap, uint32_t *size) {
    uint32_t top = heap[0];
    *size -= 1u;
    heap[0] = heap[*size];
    uint32_t at = 0u;
    for (;;) {
        uint32_t left = at * 2u + 1u;
        uint32_t right = left + 1u;
        uint32_t best = at;
        if (left < *size) {
            int64_t a = sf_chain_cost[heap[left]];
            int64_t b = sf_chain_cost[heap[best]];
            if (a > b || (a == b && heap[left] < heap[best])) {
                best = left;
            }
        }
        if (right < *size) {
            int64_t a = sf_chain_cost[heap[right]];
            int64_t b = sf_chain_cost[heap[best]];
            if (a > b || (a == b && heap[right] < heap[best])) {
                best = right;
            }
        }
        if (best == at) {
            break;
        }
        uint32_t tmp = heap[at];
        heap[at] = heap[best];
        heap[best] = tmp;
        at = best;
    }
    return top;
}

static int32_t sf_plan_lpt(const SfHostInputs *inputs, const SfHostShapes *shapes,
                           SfHostWorkspaces *workspaces) {
    int32_t *cu = inputs->cu_seqlens;
    uint32_t chains = shapes->num_seqs * shapes->H;
    uint32_t workers = sf_min_u32(SF_MAX_WORKERS_EFFECTIVE, chains);
    uint32_t item = 0u;
    if (workers == 0u) {
        return SF_HOST_INVALID_BOUNDS;
    }
    for (uint32_t w = 0u; w < workers; w++) {
        sf_worker_load[w] = 0;
    }
    /* 1. range-cut each chain by a fixed-point cost target; remember each piece's cost. */
    for (uint32_t chain = 0u; chain < chains; chain++) {
        uint32_t seq = chain / shapes->H;
        uint32_t begin = (uint32_t)cu[seq];
        uint32_t len = (uint32_t)(cu[seq + 1] - cu[seq]);
        for (uint32_t slot = 0u; slot < SF_PLANNER_MAX_SPLIT - 1u; slot++) {
            workspaces->handoff_flags[chain * (SF_PLANNER_MAX_SPLIT - 1u) + slot] = SF_PLANNER_FLAG_EMPTY;
        }
        sf_chain_cost[chain] = 0;
        if (len == 0u) {
            continue;
        }
        uint32_t pieces = sf_min_u32(SF_PLANNER_MAX_SPLIT, len);
        int64_t target = ((int64_t)len << SF_PLANNER_COST_SHIFT) / (int64_t)pieces
                         + SF_PLANNER_START_COST;
        uint32_t remaining = len;
        uint32_t cursor = begin;
        for (uint32_t p = 0u; p < pieces; p++) {
            uint32_t slots_left = pieces - p - 1u;
            uint32_t size;
            if (p + 1u == pieces) {
                size = remaining;
            } else {
                int64_t want = (target - SF_PLANNER_START_COST) >> SF_PLANNER_COST_SHIFT;
                if (want < 1) {
                    want = 1;
                }
                if ((uint32_t)want > remaining - slots_left) {
                    want = (int64_t)(remaining - slots_left);
                }
                size = (uint32_t)want;
            }
            uint32_t last = (p + 1u == pieces) ? 1u : 0u;
            int64_t cost = ((int64_t)size << SF_PLANNER_COST_SHIFT) + SF_PLANNER_START_COST
                           + (last ? 0 : SF_PLANNER_HANDOFF_COST);
            if (item >= SF_MAX_ITEMS || cursor >= 0x10000u || size >= 0x10000u || pieces >= 0x100u
                || chain >= 0x100u) {
                return SF_HOST_CAPACITY_EXCEEDED;
            }
            sf_piece_word0[item] = cursor | (size << 16);
            sf_piece_word1[item] = chain | (p << 8) | (pieces << 16);
            sf_piece_cost[item] = cost;
            sf_piece_chain[item] = chain;
            sf_piece_index[item] = p;
            sf_piece_worker[item] = 0u;
            sf_chain_cost[chain] += cost;
            cursor += size;
            remaining -= size;
            item++;
        }
    }
    /* 2. LPT: heaviest chain first onto the least-loaded worker (ties: lower worker id). */
    uint32_t heap_size = 0u;
    for (uint32_t chain = 0u; chain < chains; chain++) {
        if (sf_chain_cost[chain] > 0) {
            sf_heap_push(sf_chain_heap, &heap_size, chain);
        }
    }
    while (heap_size > 0u) {
        uint32_t chain = sf_heap_pop(sf_chain_heap, &heap_size);
        uint32_t best = 0u;
        for (uint32_t w = 1u; w < workers; w++) {
            if (sf_worker_load[w] < sf_worker_load[best]) {
                best = w;
            }
        }
        sf_worker_load[best] += sf_chain_cost[chain];
        for (uint32_t i = 0u; i < item; i++) {
            if (sf_piece_chain[i] == chain) {
                sf_piece_worker[i] = best;
            }
        }
    }
    /* 3. Emit grouped by worker, preserving the chain/piece order inside a worker. */
    uint32_t out = 0u;
    for (uint32_t w = 0u; w < workers; w++) {
        for (uint32_t i = 0u; i < item; i++) {
            if (sf_piece_worker[i] == w) {
                workspaces->work_items[out * 2u + 0u] = sf_piece_word0[i];
                workspaces->work_items[out * 2u + 1u] = sf_piece_word1[i];
                out++;
            }
        }
    }
    workspaces->item_counts[0] = (int32_t)out;
    sf_publish(workspaces, out, chains * (SF_PLANNER_MAX_SPLIT - 1u));
    return SF_HOST_OK;
}

#if SF_PLANNER_MODE == 3
/* Target-dependent emission order: the same canonical items, reversed when the target exposes a
 * large threadgroup limit.  It is a legal planner (the device reads item_counts and never assumes
 * an order) and exists to prove that cross-target parity compares the item *set* and semantic
 * dependency edges, not the emission order or byte layout. */
static int32_t sf_plan_target_order(const SfHostInputs *inputs, const SfHostShapes *shapes,
                                    const SfHostTarget *target, SfHostWorkspaces *workspaces) {
    int32_t status = sf_plan_canonical(inputs, shapes, workspaces);
    if (status != SF_HOST_OK) {
        return status;
    }
    uint32_t count = workspaces->work_items_count / 2u;
    if (target->max_threads_per_threadgroup >= 2048u) {
        for (uint32_t i = 0u; i < count / 2u; i++) {
            uint32_t j = count - 1u - i;
            uint32_t w0 = workspaces->work_items[2u * i];
            uint32_t w1 = workspaces->work_items[2u * i + 1u];
            workspaces->work_items[2u * i] = workspaces->work_items[2u * j];
            workspaces->work_items[2u * i + 1u] = workspaces->work_items[2u * j + 1u];
            workspaces->work_items[2u * j] = w0;
            workspaces->work_items[2u * j + 1u] = w1;
        }
    }
    return SF_HOST_OK;
}
#endif

int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
#if SF_PLANNER_MODE != 3
    (void)target;
#endif
    (void)resolved;
    int32_t status = sf_validate(inputs, shapes);
    if (status != SF_HOST_OK) {
        return status;
    }
    status = sf_check_capacity(shapes->num_seqs * shapes->H, 0u);
    if (status != SF_HOST_OK) {
        return status;
    }
#if SF_PLANNER_MODE == 2
    return sf_plan_lpt(inputs, shapes, workspaces);
#elif SF_PLANNER_MODE == 3
    return sf_plan_target_order(inputs, shapes, target, workspaces);
#else
    return sf_plan_canonical(inputs, shapes, workspaces);
#endif
}
