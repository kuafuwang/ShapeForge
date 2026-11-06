/*
 * Candidate-owned fixed-length KDA planner for the native64 Host Program ABI.
 * The generated host_contract.h is prepended by the harness; no libc or harness scheduler is used.
 */

int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs;
    (void)target;
    (void)resolved;
    const uint32_t tokens = shapes->N;
    if (tokens == 0u || shapes->H == 0u || shapes->D == 0u) {
        return SF_HOST_INVALID_BOUNDS;
    }
    if (SF_WS_WORK_ITEMS_CAPACITY_ELEMS < 6u || SF_WS_ITEM_COUNTS_CAPACITY_ELEMS < 1u) {
        return SF_HOST_CAPACITY_EXCEEDED;
    }
    const uint32_t pieces = tokens < 3u ? tokens : 3u;
    const uint32_t base = tokens / pieces;
    const uint32_t remainder = tokens % pieces;
    uint32_t begin = 0u;
    for (uint32_t piece = 0u; piece < pieces; ++piece) {
        const uint32_t count = base + (piece < remainder ? 1u : 0u);
        workspaces->work_items[piece * 2u] = begin;
        workspaces->work_items[piece * 2u + 1u] = count;
        begin += count;
    }
    if (begin != tokens) {
        return SF_HOST_INVALID_BOUNDS;
    }
    workspaces->item_counts[0] = (int32_t)pieces;
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = pieces * 2u;
    workspaces->item_counts_offset = 0u;
    workspaces->item_counts_count = 1u;
    return SF_HOST_OK;
}
