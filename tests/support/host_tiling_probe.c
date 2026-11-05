#include <stdint.h>

// V3 toolchain probe only: read a packed boundary vector and return segment lengths.
// It deliberately has no libc/WASI imports and no candidate scheduling algorithm.
__attribute__((export_name("host_tiling")))
int32_t host_tiling(const int32_t *cu, int32_t count, int32_t *out) {
    if (count < 2 || cu[0] != 0) return 1;
    for (int32_t i = 1; i < count; ++i) {
        if (cu[i] < cu[i - 1]) return 2;
        out[i - 1] = cu[i] - cu[i - 1];
    }
    return 0;
}

// A separate export used by the probe to prove fuel exhaustion is a typed trap,
// not an unbounded host loop. It is never a candidate planner.
__attribute__((export_name("busy")))
void busy(void) {
    for (;;) { }
}
