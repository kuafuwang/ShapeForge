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
uv run python -m taic.cli run softmax --budget 1.0     # let the LLM lower one kernel with a $1 smoke budget (default: the paper's $15, see Budget)
uv run pytest -q                                       # environment, orchestrator (scripted LLM), experiment scripts
```

When a test fails, read `logs/pytest/triage/triage_summary.md`, then the markdown file it points at. The JSON block between `---BEGIN_AI_FAILURE_HANDOFF---` and `---END_AI_FAILURE_HANDOFF---` is the ground truth for that failure (status, feedback, and a single-process reproduce command). Re-run with the command at the top of that file. Leave it single-process: the GPU also drives the display, so the suite does not use `-n`.

Configuration is in `taic.toml`, with the paper's search budget layered on top by the `paper` preset `taic.paper.toml` (see Budget); credentials
are never stored in the repository (they are read from the file named there, or from `OPENAI_API_KEY` / `OPENAI_BASE_URL`) and are redacted in
every transcript.

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

## Budget

The paper caps every search at a fixed USD amount (arXiv 2609.36800v1, `docs/2609.36800v1.pdf`): Section 4.3 and Table 1, "a maximum API
budget of $15 per run for convolution, matrix multiplication, and FlashAttention, and $1 for the simpler elementwise SwiGLU kernel"; Section 4.4
and Figure 5, "under a $15 budget" for the ten published kernels. The search repeats until the budget is exhausted (Section 3.3); the gate is
checked before each LLM request, so a request already in flight can finish a little above the cap. This repository does the same, and
**the default budget is $15 per run** (`SearchConfig`, `taic.cli run`, `run_suite`, `repeatability`, `p1_runs`). The 12-kernel grid of Figure 4
states no cap, so it uses $15 too; the repeatability experiment (Table 1) gives SwiGLU $1, as the paper does.

* **Where it lives.** `taic.paper.toml` is a preset read on top of `taic.toml` (it sets `budget_usd = 15` and `max_rounds = 100`: at $15 the
  budget ends a run, not the round count, whereas the old limit of 12 rounds would have stopped it at about a third of the money). `--preset base`
  or `TAIC_PRESET=base` reads `taic.toml` alone; `--config FILE` is used as it is. the preset raises `patience` (stop after N rounds without a new best) from 4 to 12, so a short plateau does not end a $15 run but a converged one still stops early
  and still ends runs that converged early, so a kernel that stops improving spends less than $15.
* **Override.** `--budget USD` everywhere, `--budget-scale F` on `run_suite` and `repeatability` (for example `0.2` for a smoke run). The
  semantic tests (`$0.15` per run), `counter_teaching` (`$1`) and the unit tests keep their small budgets on purpose.
* **Worst case.** If every run used its whole budget: `run_suite --tasks all --dtypes fp16,bf16` is 24 runs, $360; `repeatability` is 5 trials
  x ($15 + $15 + $1) = $155 (with `flash_attention` $230); `p1_runs` is 11 runs, $165. `--max-total-usd USD` (on those three drivers, default off)
  skips any run whose budget would not fit under a cap on the whole invocation. Nothing has come close so far: the largest spend of a finished run
  is $3.12 (an 8-round FlashAttention search that was stopped by its $3 budget).
* **Earlier results are not comparable.** They were produced with smaller budgets, which the dashboard shows per run (and each `result.json`
  records as `budget_usd`): `runs/suite_full` $0.5-$1.2 per kernel, `runs/repeat_*` $3 per trial, `runs/p1` $3 for most runs (the `*_paper`
  runs started later have $15), `runs/semantic_main` $0.15, `runs/pilot_suite_mini_old_protocol` $1.25-$1.5. A run that stopped on
  `budget_exhausted` there would have continued under the paper's budget.

## Experiments (paper Section 4)

```bash
uv run python -m experiments.run_suite --tasks all --dtypes fp16,bf16     # Figure 4: speedup vs cost, 12 kernels ($15 per run; --budget-scale 0.2 for a smoke run)
uv run python -m experiments.repeatability --tasks conv2d,gemm,swiglu --trials 5   # Table 1 ($15, swiglu $1)
uv run python -m experiments.semantic_tests --n-random 10                 # Section 4.1
uv run python -m tcenv.cli profile gemm best.metal                        # xctrace: spills, code size, thermal state
uv run python -m observe                                                  # local dashboard at http://127.0.0.1:8765
```

How to read the numbers: the reported speedup of a run is the *confirmed* one (the winner and its ties re-measured with a fresh seed and
twice the repetitions); the suite table also lists the speedup of the task's naive hand-written kernel over the same baselines, because some
baselines are weak (for example Lion and the BitDelta GEMM have no fused library implementation), and a naive kernel that already beats
the baseline means part of any gain is baseline weakness.

## Hardware-feature feedback (optional, off by default)

`search.hw_feedback = true` (`taic run ... --hw-feedback`, `taic resume ... --hw-feedback`, or `hw_feedback=True` in the search options of a driver, as for `profile_poor`)
appends a short block (at most 12 lines, about 150 words) to the evaluation feedback of every *correct* seed that the planner and the patch authors read, in the place where the
profiler evidence goes, and adds `taic/skills/hardware_feedback.md` to the system prompt as Skill 2c. The block is computed by `taic/hwfeedback.py` from the static detector of
`docs/hardware_features.md` (`taic/hwfeatures.py`: the source, the compiled AIR, and the roofline measured for this candidate) and says:

* which Apple-GPU features the kernel uses (matrix unit with its accumulator type and fragment count, threadgroup staging and padding, double buffering or register prefetch, width of the
  device loads and stores, SIMD reductions and shuffles, online softmax, ...) and how far it is from the roofline;
* which features that are expected for this kind of task are missing, at most three, each with one concrete hint (for example "no simdgroup_matrix: the matrix product runs on scalar FMA;
  the roofline shows 3.0x the lower bound"), always with the caveat that these are hints from static analysis, not requirements, and that the measured time decides;
* nothing as missing for a kernel that is already at about 90% of its roofline (hardware features cannot help beyond it; the elementwise tasks are in that state, and scalar loads already
  reach most of the bandwidth there: `docs/hardware_features.md` section 6), and a warning when a kernel has 32 or more accumulator fragments (the reference GEMMs were 12 to 26 times slower)
  or sits at the register-pressure limit.

Why off by default: it is an experiment and nothing yet shows that it helps. With the switch off, every system prompt, user message and feedback string is **byte-identical** to before; `tests/test_hw_feedback.py`
checks that against hashes recorded from the code before the feature existed (176 system prompts, 110 user messages, 12 feedback texts and every request of four scripted searches), so runs that are
running or resumed are not affected. `taic.toml` and `taic.paper.toml` do not mention it. It is recorded in `run_config.json` and kept by `taic resume`. The analysis never breaks or slows a search: a timeout
(20 s), a failure or a missing Metal toolchain drops the block and logs a warning. A/B experiment: `experiments.p1_runs` run `fa_flash_hw` is `fa_flash` with only this switch changed
(n=1 on a stochastic search: an anecdote); the run was started on 2026-10-07 06:08, stopped once and continued with `taic resume` after a detector fix (round 1 saw the earlier block, `docs/hardware_features.md` section 9), and its result is not in yet (look at `runs/p1/fa_flash_hw`; the comparison is `fa_flash`: 0.735x confirmed, $3.123, 8 rounds, stopped by budget).

When the hardware-feature block is on and a baseline is already more than 5% faster than the roofline lower bound, the block does not say the kernel is at the ceiling. That bound counts the direct algorithm's operations, and a baseline that beats it is using fewer of them (the measured case is conv2d: torch-mps at 777 us against a 932 us direct-convolution bound, about 1.2x the measured matrix peak). The reachable target is that baseline's time.

## Search switches (optional, all off by default)

Each one is a flag on `taic run` / `taic resume` and a field of the search or LLM config (`truncation_retry` is on the LLM config; the others are on the search config). A driver passes them like `profile_poor`, including `truncation_retry=True`. They are recorded in `run_config.json` and kept by `taic resume`. With all of them off, prompts and feedback stay byte-identical to before.

| switch | what it changes |
| --- | --- |
| `--truncation-retry` | a reply cut off at `max_tokens` with no kernel is asked once more, without the echoed reasoning, and that retry does not use up a repair turn |
| `--stop-at-roofline` | stop when the seed is already inside the noise gate of its roofline and no baseline beats that bound (`stop_reason=at_roofline`) |
| `--abi-guard` | stop a branch that hits the same ABI failure twice, and tell the repair the threadgroup limit (32768 B) and the size it just declared |
| `--target-note` | when a baseline is more than 5% faster than the roofline bound, tell the model the bound does not hold and the target is that baseline's time |
| `--rich-history` | a failed branch's history line names the failure (`abi: ...`, `incorrect: ...`, `no code: ...`) instead of one generic sentence |
| `--status-lines` | the planner and the patch author are told the noise floor ("at least X% faster") and the round, patience and budget that remain |

`result.json` and `state.json` also record `search_seed_label` (the kernel the search was mutating) and `confirm_winner_label` (the kernel the fresh-seed remeasure kept). A resume continues from the search seed.

`--max-total-usd` on the suite, the repeatability experiment and `p1_runs` is checked against the dollars reserved on disk: a live run counts its current budget (so a dashboard top-up and a run resumed on the side both count), a finished run counts what it spent. The cap is written to `cap.json` in the experiment directory, and the dashboard refuses a top-up that would break it.

## Dashboard and budget alerts

`uv run python -m observe` serves a read-only dashboard (default `http://127.0.0.1:8765`; restart it after updating the code, the page is
read at start-up). It shows every run's budget next to what it spent, and raises an alert for runs that crossed it. The rule is one pure
function, `observe.budget.budget_status(spent, budget, stop_reason, running)`:

| level | colour | when |
|---|---|---|
| `OVER BUDGET` (超预算) | red | spent > budget; the overrun is shown in dollars and percent |
| `BUDGET EXHAUSTED` (预算耗尽) | orange | the search stopped with `stop_reason == "budget_exhausted"` (can be within budget: the gate fires before the next request) |
| `NEAR BUDGET` (接近预算) | yellow | still running and at least 80 % of the budget used |

A run can be both over budget and exhausted; it then carries both badges. A run whose budget or spend is unknown (older results) is shown as
"budget —" and never flagged. What you see:

* a sticky banner at the top with the counts per level, the total spend against the summed budgets, the summed overrun and one clickable chip
  per flagged run (it is also rendered into the first HTML response, so it is there without JavaScript);
* in the history list: budget, spent and percent used for every run, a coloured badge on flagged runs, filter pills (all / flagged / over /
  exhausted / near) and sort pills (time / share of budget used), so the over-budget runs can be pulled to the top;
* on the selected run, the status card and the search panel repeat the badge with the exact amounts.

`/api/budget` returns the same summary as JSON, and `/?run=p1%2Ffa_pro` opens a given run directly. Note that the figures come from each run's
own `state.json` / `result.json` / `ledger.json`, so a run launched before this change is judged by the budget it actually had.

## Adding budget and resuming runs

A run that ran out of money (`stop_reason == "budget_exhausted"`) or whose process died (a crash, a closed editor window) can be continued
from its own directory; nothing already spent is spent again. Two ways, one idea:

* **Live top-up** of a run that is still going: `<run_dir>/control.json` (`{"budget_usd": <new TOTAL>}`, written atomically). The search reads it
  before every budget gate (one `stat`), raises its budget to it, never below what is already spent, and logs `budget $a -> $b`. Every change is
  appended to `<run_dir>/budget_events.json` (`requested` by whoever asked, `applied` by the run, `resume` for a restart; each with time, old and
  new budget, amount and who: `dashboard` / `cli`). Only processes started after this feature read the file: `state.json` of such a process says
  `"control_supported": true`; a run started by an older version cannot be topped up live (wait for it to stop, then resume it).
* **Resume** of a stopped run: `uv run python -m taic.cli resume <run_dir> [--add-budget USD | --budget TOTAL] [--dry-run]`. It rebuilds the state
  from the directory and goes on with the rounds that were not finished, then confirms and finishes as usual. Before it overwrites anything,
  `result.json` / `confirm_report.json` are kept as `*.before_topup_<n>.json` (and `state.json` as `state.before_resume_<n>.json`), so the earlier
  result is never lost. `--dry-run` rebuilds everything in memory, prints what would be continued (including the checks between the sources) and writes
  nothing, calls no model, uses no GPU. A run that stopped for any other reason (converged, `max_rounds`, GPU hang, model errors) is **not** resumable
  and there is no override; a run still alive, or one that is being resumed (`<run_dir>/resume.lock` of a live pid), is refused.

What a resume rebuilds, and from where (`taic/resume.py` has the details):

| state | source | exact? |
|---|---|---|
| money spent, tokens, request list | `transcript.jsonl` (usage, cost and running total of every request); `ledger.json` / `state.json` only cross-check it | yes, for every request that was answered |
| candidates, reports, best kernel, evaluation cache | `candidates/*/{kernel.metal,report.json}`; selection replayed and compared with the trace in `state.json` | yes |
| hypotheses tried (`history`), stagnation counter | the planner's replies in the transcript plus each branch's candidates; checked against the history block of every recorded planner prompt | yes (a branch whose model request *failed* is worded as "no valid kernel after repair") |
| the interrupted round | its plans come from the transcript (no new planner request); finished branches stay finished; a part-way branch continues from its recorded conversation: tool calls without a result are answered locally, the next request carries the label that was due (`r4.p0.fix1#t1` ...) | yes, up to the requests that were in flight |
| system prompt | the one in the transcript is reused, even if the prompt files were edited since | yes |
| settings | `run_config.json` (new runs, no credentials in it); older runs: budget and round limit from `state.json`; tool use, profiler skill, plan mix, branches, repair tries and initial samples from the recorded prompts and labels | `patience` and `reasoning_effort` are not recorded by older runs: pass `--patience` (default: `taic.toml`, i.e. what runs older than the presets used) |
| profiler evidence of earlier rounds | `profiles/r*_<seed>.{txt,json}` | yes; traces that failed are not listed |

Not recoverable: model requests that were in flight when the process died (the provider may have billed them, the ledger never saw them: the
resume notes list them), and the per-request wall-clock times of earlier sessions (`t` is `null` there). The dry run lists both.

**Dashboard.** On a budget-exhausted / over-budget / interrupted run (and on a running run that reached NEAR BUDGET) the run's card, its banner chip
and the search panel carry an **Add budget** button. It opens a dialog with the current budget, what is spent, the best speedup so far, `+$3 / +$5 /
+$15` and a custom amount, the resulting total, and the sentence "this spends real LLM money, up to $X more"; after Confirm the page keeps showing
the live state of the run, with a TOPPED UP badge (such a run is not directly comparable with fixed-budget runs) and its budget history
(`$3.00 -> $8.00 (+$5.00, 04:35, dashboard)`). A run that cannot be continued shows the reason instead of the button. The page is read-only
otherwise. The write path (`POST /api/runs/<run>/budget` with `{"amount_usd": 5}` or `{"new_budget_usd": 8}`; `GET /api/runs/<run>/control`
says what is possible) spends money and starts GPU work, so: the server only answers on a loopback `Host`, a browser `Origin` must be the page's
own, every POST needs the per-process random token the served page carries (`X-Observe-Token`), the amount must be positive and at most
`OBSERVE_MAX_TOPUP_USD` (default 25), the run must resolve inside `runs/` (no `..`, no symlink out), and one request per run is in flight at a time.
A live run gets `control.json`; a stopped run gets a detached `uv run python -m taic.cli resume ...` (own session, log in `<run_dir>/resume_<n>.log`).
`OBSERVE_CONTROL_DRYRUN=1` performs every check, records the command that *would* run in `budget_events.json` (kind `dryrun`) and starts nothing.

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
