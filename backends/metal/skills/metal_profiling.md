# Interpreting the evaluation feedback (the Nsight Compute analogue)

Metal does not expose NCU-style hardware counters to programs, so the environment reports what can be measured reliably.
This document explains each field of the feedback you receive after a candidate is evaluated and how to turn it into the next optimization.

## Status line
Keywords: status stage ok rejected compile_error abi_error unsafe incorrect runtime_error timeout crash

`STATUS: <status> (stage reached: <stage>)`. Stages run in order: screen, compile, abi, correctness (includes safety), bench.

- `rejected`: refused before compilation (unbounded loop, non-Metal include). Fix the source construct named in the message.
- `compile_error`: the Metal compiler's diagnostics follow, with line numbers relative to *your* kernel text.
- `abi_error`: the kernel's parameters/launch do not match the contract (names, slots, const-ness, extra parameters, dynamic threadgroup memory,
  or the pipeline cannot run the contract's threads per threadgroup).
- `unsafe`: out-of-bounds write (guard bands), input modified, output depends on initial contents, results differ between identical launches,
  or Metal shader validation flagged an invalid access.
- `incorrect`: numerically wrong. The failing case name, the number of out-of-tolerance elements, the worst element (index, got, want) and whether elements
  were never written are listed.
- `runtime_error` / `timeout` / `crash`: the GPU reported an error, a single launch ran longer than seconds and the worker was killed, or the worker died.
  Look for unbounded loops, barriers inside divergent branches, or huge out-of-bounds accesses.
- `ok`: eligible. The performance section is then meaningful.

## Correctness section
Keywords: tolerance scaled error cases normal wide_range large tiny outlier cancel const

Each case is a randomized adversarial input distribution compared with a float64/float32 CPU reference.
`worst scaled error` is `max |out - ref| / (rtol*|ref| + atol_rel*scale)`: **1.0 means exactly at the tolerance**, PASS needs <= the allowed value
(1.0, or up to 2x the vendor baseline's own error when the baseline is itself less accurate than the nominal tolerance).
Cases: `normal` (Gaussian), `wide_range` (six decades of magnitude), `large` (values near the dtype's safe edge: overflow bugs), `tiny` (subnormal / near-underflow),
`outlier` (rare 100x spikes), `cancel` (large +/- pairs: accumulation bugs), `const` (all equal).
A failure only in `large` usually means an fp16 intermediate or a missing max-subtraction; only in `cancel`/`wide_range` usually means a low-precision accumulator;
only in `tiny` usually means a division/rsqrt without an epsilon or flush-to-zero sensitivity.

## Performance section
Keywords: median cv gpu timestamp speedup baseline interleaved rotation

- `candidate: X us (cv Y%, GPU-timestamp Z us)`: median wall time per launch of `k` back-to-back launches (serial dispatch), over interleaved repetitions
  with the baselines, rotating over several buffer sets so caches are cold. `cv` is the run-to-run coefficient of variation: differences smaller than a few cv are noise.
  The GPU-timestamp is the on-device time of the same command buffer (excludes submission latency).
- `mlx` / `torch_mps`: the vendor-tuned implementations of the same operator under the same protocol. The speedup that matters is against the **faster** of them.
  A candidate becomes the new best only if it is eligible and faster.

## Roofline section
Keywords: roofline bound memory compute achieved GB/s TFLOP/s lower bound

The roofline uses peaks measured on this chip (a read-only kernel and a copy kernel for bandwidth, the vendor GEMM for compute), not data-sheet numbers. The bandwidth peak is interpolated on the operator's read/write mix. The numbers are on the scorecard and in `device.md`.

- `bound: memory|compute`: which side of the ridge point the operator is on given its minimal bytes and FLOPs.
- `achieved ... GB/s (N% of peak)`: counted with the *minimal* traffic (each input and output byte once). Values near 85-95% of the measured peak are as good as it gets;
  far below means redundant traffic, poor vectorization, too few bytes in flight, or a latency-bound reduction.
- `roofline lower bound X us; candidate is Rx above it`: the best conceivable time. R close to 1.0 means little is left; R of 2-3 for a compute-bound GEMM means the matrix unit is
  not being used well.

## Pipeline statistics
Keywords: maxTotalThreadsPerThreadgroup register pressure static threadgroup memory occupancy

`maxTotalThreadsPerThreadgroup` is the pipeline's threadgroup launch limit. It is not a reliable register-pressure measurement on its own: without an explicit attribute it commonly reports the device cap, and a lower value can come from threadgroup memory or an explicit limit. Use it with a real spill/register signal and the measured time.
and fewer resident threads, which hurts latency hiding. `static threadgroup memory` is what the kernel declares; more than half of the 32 KiB limits residency.
Hints are printed when either looks problematic.

## Instruments trace (optional, on demand)
Keywords: xctrace instruments spill register spills shader size thermal state counters

For a kernel that already passes evaluation, `tcenv profile` records a Metal System Trace with `xctrace` and reports the signals that are reliable on this machine:
**register spill bytes** (non-zero means the compiler ran out of registers and now moves values through memory: shrink tiles, unrolling or live ranges),
**compiled shader size** (a proxy for instruction count: much larger than expected means over-unrolling), and the **thermal state** (anything but Nominal means
timings are unreliable). Per-dispatch GPU times and the GPU-active fraction are printed as *indicative only* because tracing perturbs timing. Hardware counters
(ALU utilization, occupancy, limiters) are only available if a counter profile was saved in an Instruments template and passed via `TCENV_XCTRACE_TEMPLATE`.
This trace is never part of the inner search loop.

## Diagnose -> plan checklist
Keywords: diagnosis plan next step bottleneck

1. Is the status `ok`? If not, fix that first; do not optimize.
2. Compare the candidate to the faster baseline and to the roofline lower bound.
3. Name the dominant bottleneck from the table below, state the hypothesis, make one structural change.

| Observation | Likely cause | Typical change |
| --- | --- | --- |
| memory-bound, < 70% of peak GB/s | scalar loads, few bytes in flight | 16-byte vector loads, unroll to issue several loads before use |
| memory-bound, ~2x traffic | second pass over global memory, unfused ops | keep the row in registers, online softmax, fuse |
| compute-bound, 2-3x above bound | scalar FMA instead of matrix unit | `simdgroup_matrix` with float accumulators |
| compute-bound, matrix unit used but slow | no register blocking / operand reuse, bank conflicts | 4x4 accumulator tiles per SIMD group, padding, double buffering |
| `maxTotalThreadsPerThreadgroup` dropped | too many live registers (over-unrolling) | reduce unroll / tile sizes, narrower temporaries |
| small reduction kernels | latency of cross-SIMD-group barriers | `simd_sum` first, one barrier, fewer passes |
