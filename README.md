# AI lowering on Apple GPUs: Triton -> Metal Shading Language

A re-implementation, for an Apple M4 Max, of the setup of *"AI as a compiler: compiling Triton kernels without the Triton compiler"*.
An LLM translates a Triton kernel directly into a Metal (MSL) kernel; an evaluation environment compiles, safety-checks,
differentially tests, benchmarks and profiles every candidate, and the search keeps only candidates that are safe, correct and faster.

| paper | here |
| --- | --- |
| Triton -> PTX | Triton (as specification text) -> MSL |
| TCEnv (environment) | `tcenv/` |
| TAIC (agent) | `taic/` |
| ptxas + CUDA driver | `MTLDevice.newLibraryWithSource` (PyObjC) |
| compute-sanitizer | guard bands + NaN-poisoned outputs + determinism + `MTL_SHADER_VALIDATION` |
| Nsight Compute | pipeline statistics, GPU timestamps, measured roofline, optional `xctrace` |
| autotuned Triton baseline | faster of MLX and PyTorch-MPS; optionally Triton compiled by `triton-msl` (see below) |
| FP8 | not available on Apple GPUs; FP16 and BF16 instead |
| GPT-6 Astra, USD budget | any OpenAI-compatible endpoint (default `deepseek-flash`), cache-aware USD ledger |

## Quick start

```bash
uv sync
uv run python -m tcenv.cli tasks                       # the 12 kernels (+ phase 2: flash_attention, lion, bitdelta_gemm, mamba2_chunk_scan)
uv run python -m tcenv.cli naive gemm                  # harness self-check: a hand-written kernel through the whole pipeline
uv run python -m tcenv.cli eval softmax my_kernel.metal
uv run python -m taic.cli run softmax --budget 1.0     # let the LLM lower one kernel (credentials: ~/.pi/agent/models.json or OPENAI_*)
uv run pytest -q                                       # environment, orchestrator (scripted LLM), experiment scripts
```

When a test fails, read `logs/pytest/triage/triage_summary.md`, then the markdown file it points at. The JSON block between `---BEGIN_AI_FAILURE_HANDOFF---` and `---END_AI_FAILURE_HANDOFF---` is the ground truth for that failure (status, feedback, and a single-process reproduce command). Re-run with the command at the top of that file. Leave it single-process: the GPU also drives the display, so the suite does not use `-n`.

Configuration is in `taic.toml`; credentials are never stored in the repository (they are read from the file named there, or from
`OPENAI_API_KEY` / `OPENAI_BASE_URL`) and are redacted in every transcript.

## GPU safety (read this)

The Apple GPU also drives the display and **nothing can preempt a running Metal kernel**. A non-terminating kernel froze this machine
badly enough to need a hard reset. Therefore:

* never run a kernel with an unbounded loop by hand, and never add a test that does;
* the harness refuses `while (true)` / `for (;;)` / non-Metal `#include` before anything reaches the GPU (`tcenv/screen.py`);
* every evaluation runs in a worker process that publishes a heartbeat around each GPU launch; the parent `SIGKILL`s the worker after a
  single launch has been in flight for 2 s (4 s under shader validation) and enforces an overall timeout;
* the orchestrator stops the *whole run* after the first timed-out evaluation (`max_hangs = 1`);
* the real-hang test is skipped unless `TCENV_ALLOW_GPU_HANG_TESTS=1`; the supervisor is covered by fake workers instead.

## Measurement hygiene

Concurrent jobs distort timings: a triton-msl measurement taken while the suite ran came out 2-4x too slow, and a GEMM candidate measured
at cv 16% (quiet runs: 0.5-1.5%) displaced a better kernel. Therefore:

* every timing in this project (candidate benchmarks, baselines, triton-msl, calibration, xctrace recordings) holds a machine-wide lock
  exclusively, and all other GPU work of the evaluator holds it shared, so separate experiments can run at the same time without ever
  overlapping a measurement (`tcenv/gpulock.py`, `flock` on `~/.cache/ai_lowerling/gpu.lock`, override with `TCENV_GPU_LOCK_FILE`).
  Time spent waiting for the lock does not count toward the evaluation timeout. Programs outside this project are not covered;
* the candidate is dispatched the way MLX dispatches the independent calls of a graph (concurrent encoder, buffer barrier between passes over the
  rotating buffer sets). A strictly serial dispatch penalizes the same kernel by 0.5% (sigmoid) to 38% (gemv), median 5.8%, over the 15 tasks
  (`experiments/dispatch_ab.py`, `runs/dispatch_ab/`); the serial time is still reported as `candidate_serial_us`;
* a benchmark whose per-repetition cv exceeds 5% is repeated (up to 3 attempts, the quietest is kept, `benchmark.noisy` flags the rest);
* a candidate replaces the best only if it is faster by `min_improvement` *and* by `noise_z` standard errors of the two speedups;
* at the end, the best and up to `confirm_top - 1` candidates tied with it within noise are re-measured; the fastest becomes the result
  (`result.json: confirmations`).

## What a candidate has to survive

Each evaluation runs these stages in order; the first failing one ends it with feedback for the LLM.

1. **Screen and ABI.** Source rules (no unbounded loops, only Metal includes), then `MTLComputePipelineReflection` must show exactly the
   buffer names, address spaces and thread-position attributes of the launch contract.
2. **Correctness on randomized adversarial inputs.** Cases `normal`, `wide_range`, `large`, `tiny`, `outlier` and `ieee`. `ieee` injects NaN, +-Inf,
   -0 and denormals into float inputs (capped per task so one special does not poison every output; conv2d scales the cap with the image size).
   Outputs are compared with an fp32 reference through a NaN/Inf-aware scaled error: a NaN must appear exactly where the reference has one.
3. **Safety.** Guard bands around every buffer, NaN-poisoned outputs (an element the kernel never writes is an error), and a *determinism check on every
   case* (same inputs twice, bitwise equal), because a race can hide behind one input distribution.
4. **Benchmark** (below) and, optionally, `tcenv.verify` (a stub for a formal-verification backend; `verify(task, source)` returns `unsupported` today,
   the interface and the report line exist so a backend can be added without touching the evaluator).

**What the model is told about a slow kernel.** Every evaluation report carries a roofline (bound, achieved GB/s and TFLOP/s as a fraction of the peaks measured
on this machine, the lower bound) and pipeline statistics (`maxTotalThreadsPerThreadgroup` as a register-pressure hint, static threadgroup memory). That is what
the planner reasons from. A deeper trace is optional because it is slow (tens of seconds under the exclusive GPU lock): with `search.profile_poor = true`
(`taic run ... --profile-poor`), the seed of a round whose speedup is below `profile_poor_below` is traced with Instruments (`xctrace`, Metal System Trace) and the
findings go to the planner and to every patch author of that round: compiler register spills (events, bytes), compiled shader size, GPU idle fraction, thermal state,
each with what to do about it. Each source is traced once, at most `profile_max` times per run, and a failed trace never stops the search. Apple does not expose
hardware counters (ALU, occupancy, limiters) to `xctrace` on the command line, so those are not available. The traces are stored in `<run>/profiles/`.

Baselines are the fastest of MLX, PyTorch-MPS and, where no fused library kernel exists, compiled fused versions: the `mlx` baseline of `silu`, `swiglu`,
`gemm_silu` and Lion is an `mx.compile`d function; BitDelta and Mamba-2 add `mlx_compile` / `mlx_half_compile`; Lion, BitDelta and Mamba-2 add
`torch_compile` (Inductor on Metal, i.e. a compiler-generated fused Metal kernel). The strongest baseline is always the minimum over the ones that ran.

## Problem sizes

`--size paper` is the size of every experiment (the paper's shapes where it gives them; conv2d has N=32). `--size small` is for tests. `--size plan`
exists only where `paper` deviates from the implementation plan: `reduction_sum` is 4096x4096 at `paper` because the plan's 256x4096 problem takes about
4 us, which is bounded by Metal launch overhead (it measures dispatch, not the kernel); `plan` keeps the plan's shape available for comparison.

## Launch-contract variants

The contract (tile sizes, threads per group, buffer ABI) is fixed by us, not by the LLM, so it could cap what the search finds. Contract variants are
addressable as `gemm@BMxBNxBKtT` and `flash_attention@BMxBNtT` (for example `gemm@128x64x32t256`, `flash_attention@64x32t256`) anywhere a task name is
accepted. `experiments/contract_sensitivity.py` sweeps them (results below).

## Experiments (paper Section 4)

```bash
uv run python -m experiments.run_suite --tasks all --dtypes fp16,bf16     # Figure 4: speedup vs cost, 12 kernels
uv run python -m experiments.repeatability --tasks conv2d,gemm,swiglu --trials 5   # Table 1
uv run python -m experiments.semantic_tests --n-random 10                 # Section 4.1
uv run python -m tcenv.cli profile gemm best.metal                        # xctrace: spills, code size, thermal state
uv run python -m observe                                                  # local dashboard at http://127.0.0.1:8765
```

How to read the numbers: the reported speedup of a run is the *confirmed* one (the winner and its ties re-measured with a fresh seed and
twice the repetitions); the suite table also lists the speedup of the task's naive hand-written kernel over the same baselines, because some
baselines are weak (for example Lion and the BitDelta GEMM have no fused library implementation), and a naive kernel that already beats
the baseline means part of any gain is baseline weakness.

## Compiler baseline (optional): Triton -> Metal with triton-msl

Upstream Triton has no Metal backend. The specification text of every task is real Triton, so it can be compiled by the community backend
[triton-msl](https://pypi.org/project/triton-msl/) (alpha). It lives in an isolated environment (`.venvs/triton-msl`, Python 3.14) and is only used
through `tcenv.triton_runner`; the main environment never imports Triton.

```bash
scripts/install_triton_msl.sh                          # once: isolated Python 3.14 env (resumable downloads, pinned hashes)
xcodebuild -downloadComponent MetalToolchain           # once: Apple's Metal toolchain, an optional Xcode component triton-msl requires
uv run python -m tcenv.cli triton-baseline gemm        # or: all
```

Provenance: `triton-msl` is the PyPI wheel with a pinned sha256; Triton 3.7.0 (upstream commit `4da2e268`) is the *unofficial prebuilt wheel* published by
the triton-msl author because upstream has no macOS wheel (its sha256 is stored in `.venvs/triton-msl/TRITON_WHEEL.sha256`). Building Triton from source
instead needs a 1.7 GB LLVM download from the Triton CDN, which is the slow path.

The result (time relative to MLX, measured interleaved in one process, `num_warps` swept, correctness checked on the same adversarial cases including `ieee`) is
cached in `runs/triton_baselines/<task>_<dtype>_<size>.json` and shown in evaluation reports as an *estimated* comparison. "Faster than triton-msl" is a much
weaker statement than the paper's "faster than NVIDIA Triton".

How it is used: triton-msl is **correct-or-refuse**. When it cannot compile a kernel faithfully it raises instead of emitting wrong code, and a refusal is
recorded as such rather than as a number. Some tasks need a *variant spec* because the task's own Triton text is not in the form the compiler accepts
(`task.meta["triton_baseline_variant"]`: the FlashAttention tutorial form, a two-kernel `gemm` + `silu` pipeline for `gemm_silu`, `im2col` + `gemm` for
`conv2d`). Each cache entry carries a fingerprint of the spec text, the compiler version and the sizes; an entry whose fingerprint no longer matches is
treated as missing, so a fixed spec cannot be hidden behind an old number. `triton-baseline ... --wait-quiet MIN` waits up to MIN minutes for a quiet machine
(1-minute load <= 6 and no pytest/benchmark process) before each measurement and records the wait.

Time relative to MLX (below 1 = faster than MLX; paper size; one quiet-ish machine, see the cache entries for load and cv):

| task | fp16 | bf16 | task | fp16 | bf16 |
| --- | ---: | ---: | --- | ---: | ---: |
| sigmoid | 1.19 | 1.24 | gemv | 1.43 | 1.51 |
| relu | 1.21 | 1.21 | gemm | 1.16 | 1.16 |
| silu | 1.25 | 1.21 | gemm_silu | 1.20 | 1.19 |
| swiglu | 1.14 | 1.13 | conv2d | 1.66 | 9.94 |
| rmsnorm | 1.26 | 1.36 | flash_attention | 10.06 | refused |
| softmax | 1.39 | 1.41 | lion | 1.02 | 0.92 |
| reduction_sum | 1.60 | 1.64 | bitdelta_gemm | refused (quantized matmul) | refused |
| rope | 1.45 | 20.47 | mamba2_chunk_scan | refused (shared-memory staging) | refused |

The bf16 outliers (rope, conv2d) are real slow paths of the alpha compiler (the same 10-20x for every `num_warps`; MLX's bf16 rope is the noisier side at cv 22%,
which cannot explain a 20x gap), not measurement noise; FlashAttention
through triton-msl is 10x slower than MLX's fused attention and is refused for bf16.

## Layout

`tcenv/` environment (contract, Metal runtime, ABI check, sanitizers, testing, benchmark, profile, worker, xctrace) -
`taic/` agent (LLM client + ledger, tools, prompts, orchestrator, skills) - `tasks/` kernels (Triton spec, contract, reference, baselines, naive MSL) -
`experiments/` - `observe/` (local read-only dashboard) - `tests/` - `runs/` (artifacts, git-ignored).
