# Metal Shading Language reference (compiler: MSL 4.0 on an Apple M4 Max, Apple9 GPU family)

This is the equivalent of the PTX ISA documentation: facts about the target language that you need to
write correct, fast kernels. Every fenced block tagged `metal` is a complete kernel that was compiled with
the real compiler on this machine; `metal-snippet` blocks are fragments.

The harness already provides `#include <metal_stdlib>`, `<metal_simdgroup>`, `<metal_simdgroup_matrix>` and
`using namespace metal;` in the contract header. Do not include anything else.

## Thread, threadgroup and SIMD-group indices
Keywords: attributes thread_position_in_grid threadgroup_position_in_grid thread_index_in_threadgroup simdgroup lane

A threadgroup (TG) is the analogue of a Triton program / CUDA block; a SIMD group (32 threads) is a warp.
Without a `/* taic-launch */` block the harness launches `dispatchThreadgroups(grid, threadsPerThreadgroup)` with the sizes given in the contract.
With that block, each `launch` line supplies its own grid and threads. Request any of these as extra kernel parameters (they are not part of the buffer ABI):

```metal
kernel void k(device float* y [[buffer(0)]],
              uint3 gid  [[thread_position_in_grid]],           // global thread id (x,y,z)
              uint3 tg   [[threadgroup_position_in_grid]],      // TG coordinate (x,y,z)
              uint3 ntg  [[threadgroups_per_grid]],
              uint3 tpt  [[threads_per_threadgroup]],
              uint  lid  [[thread_index_in_threadgroup]],       // 0 .. threads_per_threadgroup-1 (linearized)
              uint  lane [[thread_index_in_simdgroup]],         // 0..31
              uint  sg   [[simdgroup_index_in_threadgroup]],    // 0 .. simdgroups_per_threadgroup-1
              uint  nsg  [[simdgroups_per_threadgroup]]) {
    y[gid.x] = float(tg.x + ntg.x + tpt.x + lid + lane + sg + nsg);
}
```

**Dimension rule (a frequent compile error):** the attributes `thread_position_in_grid`, `threadgroup_position_in_grid`, `threadgroups_per_grid`,
`thread_position_in_threadgroup`, `threads_per_threadgroup` and `threads_per_grid` must *all* be declared as scalars (`uint`) or *all* as vectors with the same
number of components (`uint2`, `uint3`). Mixing `uint gid` with `uint3 tg` fails with "expecting input declarations with either all scalar types or all vector
types with the same number of elements". The `*_index_*` / `simdgroup*` attributes (`thread_index_in_threadgroup`, `thread_index_in_simdgroup`,
`simdgroup_index_in_threadgroup`, `simdgroups_per_threadgroup`) are always scalar and can be combined freely with either choice.

For a 2-D grid declare `uint2 tg [[threadgroup_position_in_grid]]` (and no scalar grid-position attribute next to it); `tg.x` runs over grid x and `tg.y` over grid y.
Use `ushort` for indices when you want to save registers. `[[max_total_threads_per_threadgroup(N)]]` before `kernel`
tells the compiler the largest TG you will launch, which can let it allocate registers more freely.

## Address spaces and buffer arguments
Keywords: device constant threadgroup thread address space pointer const restrict

- `device`: global memory (all buffers). Inputs must be `device const T*`, outputs `device T*` (ABI rule).
- `constant`: read-only, cached; used for scalar arguments `constant uint& n [[buffer(i)]]`.
- `threadgroup`: on-chip memory shared by one TG, **32 KiB per TG on this GPU**. Declare it as a *static array inside the
  kernel body*, or, on a stage whose `launch` line says `threadgroup=BYTES`, as one dynamic `threadgroup T* x [[threadgroup(0)]]` parameter. The byte count is the launch line, not a size the kernel picks at runtime. A dynamic length is checked against the device limit on its own; the shader-validation exemption for a doubled static size does not apply to it.
- `thread`: private registers / stack.
- Compile-time sizes come from the `constant constexpr` values of the contract header (e.g. `BLOCK_SIZE`), usable in array bounds.

```metal
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint lid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 256;
    threadgroup half tile[TILE];          // static allocation: 512 bytes
    tile[lid % TILE] = x[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    y[lid] = tile[(lid + 1) % TILE];
}
```

Pointers into `device` memory may be reinterpreted (`reinterpret_cast<device const uint4*>(x)`), as long as the address is
aligned for the new type. Buffers are at least 16-byte aligned (the harness places them on 16 KiB boundaries plus guard margins).

## Scalar and vector types, bit casts, vectorized loads
Keywords: half bfloat float half4 float4 uint4 packed_half4 as_type vector load 16 byte convert

- `half` (fp16), `bfloat` (bf16, MSL >= 3.1), `float`, `int`, `uint`, `short`, `ushort`, `char`, `uchar`, `ulong`.
- Vector types: `half2/3/4`, `bfloat2/3/4`, `float2/3/4`, `uint2/3/4`, ... (`packed_half4` has alignment 2, no padding).
- Convert with constructors: `float4(h4)`, `half4(f4)`, `bfloat(f)`; `float(bf)`. fp16/bf16 values should be widened to float for
  accumulation and narrowed once at the end.
- Reinterpret bits with `as_type<T>(v)`, e.g. `as_type<half2>(uint_value)`; sizes must match.
- A thread should load 16 bytes at a time when it can: `half4`/`uint4` over a half buffer = 8 elements per load. This is the main way to
  saturate memory bandwidth (see apple_gpu_optimization).

```metal
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    uint4 v = *reinterpret_cast<device const uint4*>(x + i * 8);       // 8 halves in one 16-byte load
    half4 lo = as_type<half4>(v.xy);
    half4 hi = as_type<half4>(v.zw);
    float4 a = float4(lo) * 2.0f, b = float4(hi) * 2.0f;
    *reinterpret_cast<device half4*>(y + i * 8) = half4(a);
    *reinterpret_cast<device half4*>(y + i * 8 + 4) = half4(b);
}
```

`bfloat` arithmetic works but is converted through float internally; do the math in `float`.

## Math functions and numerical precision
Keywords: exp exp2 log sin cos tanh rsqrt fma fast precise fast-math select clamp isnan denormal

- The kernel is compiled with **fast math on** (default). `exp`, `log`, `sin`, `cos`, `tanh`, division, etc. are the fast variants:
  fine for moderate arguments, **inaccurate for large trigonometric arguments**. Use `precise::sin/cos/exp/log` when accuracy matters
  (RoPE angles up to ~1e3 rad) and `fast::` explicitly when you want speed.
- `exp2(x)` is the native primitive: `exp(x) == exp2(x * 1.4426950408889634f)`. Folding the constant into a scale is the usual trick.
- `fma(a,b,c)`, `rsqrt`, `sqrt`, `abs`/`fabs`, `min`/`max`, `clamp`, `select(a, b, cond)` (returns b where cond), `isnan`, `isinf`, `copysign`.
- fast-math may assume no NaN/Inf: do not rely on `x != x` tricks or on `inf` propagating in comparisons; subtract the row max before `exp`.
- Subnormals: **fp32 (and therefore bf16) subnormals are flushed to zero** by the GPU; fp16 subnormals are preserved.
- Overflow: a fp16 intermediate overflows at 65504. Accumulate sums of squares, dot products and softmax denominators in `float`.

```metal
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float v = float(x[i]);
    float e = exp2(-fabs(v) * 1.4426950408889634f);            // exp(-|v|) via the native exp2
    float s = v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);   // sigmoid without overflow for large |v|
    y[i] = half(s);
}
```

## SIMD-group (warp) operations
Keywords: simd_sum simd_max simd_min simd_shuffle simd_shuffle_xor simd_shuffle_down simd_broadcast prefix reduction

All 32 lanes of a SIMD group must execute these together (no divergent calls). They work on `half`, `float`, `int`, `uint`, ... and vectors.

- Reductions: `simd_sum(v)`, `simd_max(v)`, `simd_min(v)`, `simd_product(v)`, `simd_and/or/xor` (integers). The result is broadcast to all lanes.
- Scans: `simd_prefix_inclusive_sum(v)`, `simd_prefix_exclusive_sum(v)`, `..._product`.
- Data movement: `simd_shuffle(v, lane)`, `simd_shuffle_xor(v, mask)`, `simd_shuffle_down(v, delta)`, `simd_shuffle_up`, `simd_broadcast(v, lane)`, `simd_broadcast_first(v)`.
- Votes: `simd_all(p)`, `simd_any(p)`, `simd_ballot(p)`.
- Cross-SIMD-group reduction goes through threadgroup memory: each SIMD group writes one partial, barrier, then every thread (or one SIMD group) combines them.

```metal
kernel void k(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
              uint lid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];                 // one slot per SIMD group (up to 32 SIMD groups per TG)
    float v = simd_sum(x[lid]);                    // stage 1: inside each SIMD group, no memory, no barrier
    if (lane == 0) partial[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < 4; ++s) total += partial[s];   // stage 2 (4 SIMD groups in a 128-thread TG)
    if (lid == 0) y[0] = total;
}
```

## Threadgroup memory and barriers
Keywords: threadgroup_barrier simdgroup_barrier mem_threadgroup mem_device bank conflict double buffering

- `threadgroup_barrier(mem_flags::mem_threadgroup)` makes writes to threadgroup memory visible to the whole TG. It must be reached by **every
  thread of the TG** under uniform control flow (never inside a branch that only some threads take, never in a loop whose trip count differs per thread).
- `simdgroup_barrier(mem_flags::mem_threadgroup)` synchronizes only the 32 lanes of one SIMD group (cheap).
- `mem_flags::mem_device` orders device-memory accesses within the TG; there is no grid-wide barrier.
- Threadgroup memory is organized in 32 banks of 4 bytes; threads of a SIMD group that access different addresses in the same bank serialize.
  Pad row strides (e.g. `+ 1` or `+ 2` elements) to avoid column-access conflicts.
- A data race (reading what another thread writes without a barrier in between) makes results nondeterministic. The harness runs every
  kernel several times and rejects kernels whose output changes.
- Total static threadgroup memory per TG must stay <= 32768 bytes; more reduces how many TGs can be resident.

## simdgroup_matrix: 8x8 matrix multiply-accumulate (the tensor-core analogue)
Keywords: simdgroup_matrix simdgroup_half8x8 simdgroup_float8x8 simdgroup_bfloat8x8 simdgroup_load simdgroup_store simdgroup_multiply_accumulate mma tensor core gemm

The Apple GPU's matrix unit works on 8x8 tiles owned cooperatively by the 32 lanes of one SIMD group. **All 32 lanes must execute
these calls together.** Types: `simdgroup_half8x8`, `simdgroup_bfloat8x8`, `simdgroup_float8x8`. The accumulator may be wider than the inputs
(half inputs, float accumulator is supported and is what you need for accuracy: a half accumulator loses ~3 digits).

- `simdgroup_load(mat, ptr, elements_per_row [, ulong2 matrix_origin [, bool transpose]])` loads from `device` or `threadgroup` memory.
  `elements_per_row` is the leading dimension (row stride in elements) of the *source*; the pointer addresses element (0,0) of the 8x8 tile.
  `transpose = true` loads the transposed tile.
- `simdgroup_store(mat, ptr, elements_per_row [, origin [, transpose]])`.
- `simdgroup_multiply_accumulate(D, A, B, C)` computes `D = A * B + C` (D may alias C). `simdgroup_multiply(D, A, B)` computes `D = A * B`.
- `simdgroup_float8x8(0)` builds a zero tile. Keep several accumulator tiles live (e.g. a 4x4 grid of them = a 32x32 output block per SIMD group)
  and reuse each loaded A/B tile across multiple multiply-accumulates; that is what raises arithmetic intensity.

```metal
// C(8x8, float) = A(8x64, half) @ B(64x8, half): one SIMD group, mixed precision. Launch with 32 threads.
kernel void k(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device float* c [[buffer(2)]]) {
    simdgroup_half8x8 A, B;
    simdgroup_float8x8 C = simdgroup_float8x8(0);
    for (uint k = 0; k < 64; k += 8) {
        simdgroup_load(A, a + k, 64);        // A is row-major 8x64: leading dimension 64
        simdgroup_load(B, b + k * 8, 8);     // B is row-major 64x8: leading dimension 8
        simdgroup_multiply_accumulate(C, A, B, C);
    }
    simdgroup_store(C, c, 8);
}
```

Tiles can also be loaded from threadgroup memory that the whole TG filled cooperatively (stage a tile of A and B with all threads, barrier,
then each SIMD group multiplies its part). Loading straight from device memory is allowed too.
Metal 4 "tensor" APIs (cooperative tensors / matmul2d) target newer neural-accelerator GPUs and are not usable here; stay with `simdgroup_matrix`.

## simdgroup_matrix element access and epilogues
Keywords: thread_elements lane mapping accumulator epilogue activation fused silu fragment

Each lane holds two consecutive elements of an 8x8 tile, reachable through `mat.thread_elements()` (a 2-vector, readable and writable).
Verified on this GPU for `simdgroup_float8x8` loaded with `simdgroup_load(..., 8)`: for lane `l`

```metal-snippet
ushort qid = l / 4;
ushort row = (qid & 4) + ((l / 2) % 4);     // 0..7
ushort col = (qid & 2) * 2 + (l % 2) * 2;   // 0,2,4,6 ; the lane owns (row, col) and (row, col + 1)
```

This lets you apply an element-wise epilogue (activation, scaling, bias, rounding) directly on the accumulator registers before the store,
without a round trip through memory:

```metal
kernel void k(device float* c [[buffer(0)]]) {
    simdgroup_float8x8 C = simdgroup_float8x8(1.5f);
    auto e = C.thread_elements();                       // 2 elements per lane
    e[0] = e[0] / (1.0f + exp(-e[0]));                  // silu on the accumulator
    e[1] = e[1] / (1.0f + exp(-e[1]));
    C.thread_elements() = e;
    simdgroup_store(C, c, 8);
}
```

The mapping is an implementation detail rather than documented API; if you depend on it, keep a fallback of storing the tile to threadgroup
memory and applying the epilogue there.

## Atomics
Keywords: atomic atomic_fetch_add_explicit atomic_float memory_order_relaxed nondeterminism

`device atomic_uint*`, `atomic_int`, `atomic_float` exist (`atomic_fetch_add_explicit(p, v, memory_order_relaxed)`), but the contract has no
scratch buffers and the harness rejects kernels whose results differ between identical launches, so floating-point atomics (unordered sums)
will usually be rejected. Prefer a deterministic reduction tree.

```metal
kernel void k(device atomic_uint* counter [[buffer(0)]]) {
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
}
```

## Common compiler errors and what they mean
Keywords: error compile cookbook undeclared implicit conversion ambiguous address space threadgroup variable

- `use of undeclared identifier 'X'`: a contract constant (`N`, `BLOCK_SIZE`, ...) or a helper you forgot to define; contract constants are listed in the task description.
- `no matching function for call to 'simdgroup_load'`: pointer address space or type does not match the matrix type (`simdgroup_half8x8` needs a `half` pointer).
- `cannot initialize ... with an lvalue of type 'const device ...'`: assigning a `device const` pointer to a non-const pointer; keep `const`.
- `threadgroup variables are only allowed in kernel function scope`/`... must be declared in kernel`: declare `threadgroup T name[N];` inside the kernel body, not in helper functions or at file scope.
- `expecting input declarations with either all scalar types or all vector types`: you mixed scalar and vector *grid position* attributes (see the dimension rule at the top).
- `call to 'max' is ambiguous`: mixed `half`/`float`/`uint` arguments; cast so both operands have the same type.
- `invalid operands to binary expression` with `bfloat`: convert to `float` first.
- A helper function that uses `simd_sum` / `simdgroup_*` must be called by all lanes; mark helpers `inline`.
- Declaring extra `[[buffer(n)]]` / `[[threadgroup(n)]]` parameters changes the ABI and is rejected by the harness.
