Keywords: xctrace instruments profiler evidence spill register spills shader size thermal state counters limiter utilization occupancy eviction bandwidth latency-bound

This supersedes what Skill 2 says about hardware counters: when the trace has a `hardware counters` section they are available; without it only the fields that are always there are.

The feedback may end with "Profiler evidence for this kernel": a Metal System Trace of THIS kernel launching back to back. It is the only place where real hardware
signals appear, so use it to name the bottleneck instead of guessing. If the block is absent, there is no profiler evidence: do not invent counter values.

**Fields that are always there**
- `register spills: N events, B bytes`: non-zero means the compiler ran out of registers and moves values through memory on every use: shrink register tiles, unrolling or live ranges. This usually outweighs everything else.
- `compiled shader size`: a proxy for instruction count; much larger than the algorithm needs (tens of KB) means over-unrolling or inlining.
- `thermal state`: anything but Nominal means timings are unreliable.
- `GPU interval per dispatch` and `GPU active X%`: indicative only (tracing perturbs timing). A low GPU-active share means gaps between launches: a property of the launch contract and the problem size, not of the kernel body.

**Hardware counters (Instruments "Performance Limiters")** are sampled every ~41 us across the whole GPU while our kernel runs (nothing else is on the GPU during the trace).
Each line shows `typical` (median sample), `p90` and `max`; `typical` is the steady state of the kernel, `max` is often a one-sample edge effect: do not quote it. Dispatches shorter than ~160 us give
fewer than four samples and are blurred. Percentages are relative to the unit's peak.

Read them in this order:

1. **Limiters** (`... Limiter`): the share of time a unit was asked to work, relative to its peak. 100% means that unit is full and cannot take more work; **the highest limiter is the bottleneck**.
   (`... Utilization` is the share of time the unit actually executed; it never exceeds its limiter. Limiters are not additive: several can be high at once.)

   | highest limiter (typical >= 40%) | the kernel is bound by | what to change |
   | --- | --- | --- |
   | Instruction Throughput | the number of instructions executed | hoist loop-invariant work, simplify address arithmetic, vectorize (half2/half4, packed loads), fewer passes |
   | F32 / F16 | scalar floating-point arithmetic | fewer FP ops per output, FMA chains, half where precision allows, move the math onto `simdgroup_matrix` if it is a matrix product |
   | Integer and Conditional / Integer and Complex | index math, compares, selects, divisions, transcendental functions | precompute offsets, shifts and masks instead of `/` and `%`, `select()` instead of branches, cheaper approximations |
   | Control Flow | branches and loops | unroll small loops, make branches uniform across the simdgroup |
   | L1 Cache | traffic through the shader core cache (buffers, threadgroup memory, registers) | fewer accesses per FLOP, wider loads, reuse in registers, no threadgroup bank conflicts |
   | Last Level Cache / MMU | memory system | fewer bytes moved, contiguous accesses, compact tiles (MMU: too many pages touched) |
   | Compute Shader Launch | launching threads into the cores | fewer, larger threadgroups, more work per thread |

   If `Instruction Throughput Limiter` is clearly lower than the highest sub-pipeline limiter (F32, F16, Integer and Conditional, Integer and Complex), the other sub-pipelines have spare capacity
   and run in parallel: spread the instructions across them (index math as integers, half and float mixed where precision allows, `select` instead of branches).

2. **No limiter above ~40%: the kernel is latency-bound, not throughput-bound.** The units are idle because they wait. Then look at `Kernel Occupancy` and `Compute SIMD Groups Inflight` (average resident simdgroups per shader core):
   - occupancy low because the grid is small (few threadgroups; the launch contract fixes the grid, you cannot add threads): make each simdgroup's work faster; give every thread several independent accumulators / loads in flight (instruction-level parallelism), unroll the inner loop, use wider loads.
   - occupancy low although the grid is large: each thread uses too many registers or too much threadgroup memory (see `maxTotalThreadsPerThreadgroup` and `static threadgroup memory` in the pipeline statistics, and spills): shrink them.
   - occupancy fine but nothing busy: dependent chains (each load or FMA waits for the previous), barriers, or one thread doing serial work while the rest of the simdgroup waits: restructure so independent work is issued before the result is needed, merge or remove barriers.

3. **`L1 Eviction Rate` and `Occupancy Manager Target`**: an eviction rate near 100% means the shader core's private memory (registers, stack, spills, threadgroup data) is thrashing the L1; the hardware then lowers occupancy on its own
   (the target falls below 100%). Remedy: a smaller per-thread working set (smaller register tiles, fewer live arrays, no spills).

4. **Memory counters**: `GPU Read/Write Bandwidth` (GB/s to device memory) against the measured roofline peak (449-527 GB/s) says how close a memory-bound kernel is to the roof; `Last Level Cache` and `MMU` limiters
   say whether the memory system is busy; `L1 Register Residency` high means registers dominate the L1 (register pressure); `Threadgroup Memory L1 ... Bandwidth` in GiB/s shows how hard threadgroup memory is used.
   A counter only matters when it is high for its unit: a high `Buffer L1 Miss Rate` next to near-zero bandwidth and low limiters is NOT evidence of a memory problem (very few accesses are happening), and a low limiter is not a "symptom" of anything.

Limits of the evidence: this document does not tell how work done by `simdgroup_matrix` instructions is counted, so judge matrix-unit use from the roofline ratio, not from the ALU counters. A counter can motivate a
change but never prove a gain: the evaluation is the judge. State in your diagnosis which finding or counter your hypothesis rests on.

Quick reference for the diagnosis:

| profiler evidence | likely cause | typical change |
| --- | --- | --- |
| spills > 0 | live state exceeds the register file | smaller tiles/accumulators, no full unrolling, recompute cheap values |
| highest limiter >= 40% | that unit is the bottleneck | see the limiter table above |
| every limiter low, occupancy low | latency-bound (small grid or heavy per-thread resources) | independent work per thread (ILP), wider loads, shrink registers/threadgroup memory |
