# AI lowering on Apple GPUs: Triton -> Metal Shading Language

The model writes kernel text. This repository compiles, checks, times, and decides what to keep. A number you can quote is the remeasured one after search, with the source hash bound and the timing quiet. The fastest candidate seen during search is a different number. A default run spends about $15 and alternates two models, `deepseek-flash` then `kimi-k3`. `taic run` and `taic resume` spend money and use the GPU that also drives the display. `uv run python -m taic.cli brief` only reads.

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
| GPT-6 Astra, USD budget | any OpenAI-compatible endpoint. Default committees alternate `deepseek-flash` and `kimi-k3`. Cache-aware USD ledger |

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
* every evaluation runs in a worker process that publishes a heartbeat around each GPU launch; the parent `SIGKILL`s the worker after one
  command buffer has been in flight too long: 2 s for one kernel (4 s under shader validation), or 2 s times the stage count and at most 8 s
  for a pipeline (validation doubles that, at most 16 s), and enforces an overall timeout;
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
* a benchmark whose per-repetition cv exceeds 5% on the candidate or a library baseline is repeated (up to 3 attempts, the quietest is kept, `benchmark.noisy` flags the rest). `candidate_serial` does not trigger the repeat;
* a candidate replaces the best only if it is faster by `min_improvement` *and* by `noise_z` standard errors of the two speedups;
* at the end, the best and up to `confirm_top - 1` candidates tied with it within noise are re-measured on the same inputs. The search seed stays the result unless another complete remeasure clears the noise gate
  (`result.json: confirmations`).

## What a candidate has to survive

Each evaluation runs these stages in order; the first failing one ends it with feedback for the LLM.

1. **`verify()` first.** It returns `unsupported` and does not decide the verdict.
2. **Screen, compile, and ABI.** Source rules (no unbounded loops, only Metal includes), then `MTLComputePipelineReflection` must show exactly the
   buffer names, address spaces and thread-position attributes of the launch contract.
3. **Guards, then numbers.** Guard bands around every buffer. An element the kernel never writes is `incorrect`, not a safety failure. A case that
   already matches is launched three times (poisoned outputs, outputs filled with zero, poisoned again) and the three results must be bitwise equal.
   Cases are `normal`, `wide_range`, `large`, `tiny`, `outlier` and `ieee` on most tasks. `ieee` injects NaN, +-Inf and -0 into the inputs that task
   names, not denormals (those are in `tiny`) and not every float input. A NaN must appear exactly where the reference has one, and an infinity must
   keep the reference sign. Writing +0 where the reference has -0 still passes. The reference is the task's CPU function, float64 or float32.
4. **Benchmark** (below).

`tcenv.verify` stays a stub so a backend can be added without touching the evaluator.

**What the model is told about a slow kernel.** Every evaluation report carries a roofline (bound, achieved GB/s and TFLOP/s as a fraction of the peaks measured
on this machine, the lower bound) and pipeline statistics (`maxTotalThreadsPerThreadgroup` as a register-pressure hint, static threadgroup memory). With
`search.profile_poor = true` (`taic run ... --profile-poor`), the seed of a round whose speedup is below `profile_poor_below` is traced with Instruments
(`xctrace`, Metal System Trace) and the findings go into that round's `TASK.md`: compiler register spills (events, bytes), compiled shader size, GPU idle fraction,
thermal state, each with what to do about it. Each source is traced once, at most `profile_max` times per run, and a failed trace never stops the search. The
default Metal System Trace template does not include ALU, occupancy, or limiter counters. A template named by `TCENV_XCTRACE_TEMPLATE` can. The traces are stored in `<run>/profiles/`.

Baselines are the fastest of the ones that ran. `silu`, `swiglu`, and Lion use an `mx.compile`d `mlx` baseline. `gemm_silu` compiles the SiLU and leaves the matmul outside it. BitDelta adds `mlx_compile` and `torch_compile`; the compiled decode is fused and the matmul stays a separate call, and there is no `mlx_half_compile`. Mamba-2 adds `mlx_compile`, `mlx_half_compile`, and `torch_half`, and does not add `torch_compile`. `rope` has only MLX. Lion does add `torch_compile`. The strongest baseline is always the minimum over the ones that ran.

## Problem sizes

`--size paper` is the size of every experiment (the paper's shapes where it gives them; conv2d has N=32). `--size small` is for tests. `--size plan`
exists only where `paper` deviates from the implementation plan: `reduction_sum` is 4096x4096 at `paper` because the plan's 256x4096 problem takes about
4 us, which is bounded by Metal launch overhead (it measures dispatch, not the kernel); `plan` keeps the plan's shape available for comparison.

## Launch-contract variants

Without a `/* taic-launch */` block, the grid and the threads per group are the task contract. With that block, each stage of the same search picks its own grid and threads. `gemm@BMxBNxBKtT` and `flash_attention@BMxBNtT` (for example `gemm@128x64x32t256`, `flash_attention@64x32t256`) change the contract constants, and a contract-menu search rejects `taic-launch`. No other task has an `@` variant. `experiments/contract_sensitivity.py` sweeps the contract variants.

## Budget

The paper caps every search at a fixed USD amount (arXiv 2609.36800v1, `docs/2609.36800v1.pdf`): Section 4.3 and Table 1, "a maximum API
budget of $15 per run for convolution, matrix multiplication, and FlashAttention, and $1 for the simpler elementwise SwiGLU kernel"; Section 4.4
and Figure 5, "under a $15 budget" for the ten published kernels. The search repeats until the budget is exhausted (Section 3.3); the gate is
checked before each session, so a session already in flight can finish a little above the cap. `relay_budget_usd` only sizes that session's output-token guide; input and cache can spend more, and that extra is charged to the run budget instead of ending the search. This repository does the same for the run cap, and
**the default budget is $15 per run** (`SearchConfig`, `taic.cli run`, `run_suite`, `repeatability`, `p1_runs`). The 12-kernel grid of Figure 4
states no cap, so it uses $15 too; the repeatability experiment (Table 1) gives SwiGLU $1, as the paper does.

* **Where it lives.** `taic.paper.toml` is a preset read on top of `taic.toml` (it sets `budget_usd = 15` and `max_rounds = 100`: at $15 the
  budget ends a run, not the round count, whereas the old limit of 12 rounds would have stopped it at about a third of the money). `--preset base`
  or `TAIC_PRESET=base` reads `taic.toml` alone. That file's own ceiling is $50 per run, with `max_rounds = 12` and `patience = 6`; the $15 default is the paper preset, not `taic.toml` by itself. `--config FILE` is used as it is. The paper preset raises `patience` (stop after N rounds without a new best) from 6 to 12, so a short plateau does not end a $15 run, and a kernel that stops improving still spends less than $15.
* **Override.** `--budget USD` everywhere, `--budget-scale F` on `run_suite` and `repeatability` (for example `0.2` for a smoke run). The
  semantic tests (`$0.15` per run), `counter_teaching` (`$1`) and the unit tests keep their small budgets on purpose.
* **Worst case.** If every run used its whole budget: `run_suite --tasks all --dtypes fp16,bf16` is 24 runs, $360 (the default `--dtypes` is fp16 only: 12 runs, $180); `repeatability` is 5 trials
  x ($15 + $15 + $1) = $155 (with `flash_attention` $230); `p1_runs` is 12 named runs, $180. `--max-total-usd USD` (on those three drivers, default off)
  skips any run whose budget would not fit under a cap on the whole invocation. Finished runs on disk have spent more than the old $3.12 note: `runs/apple-m4-max/conv2d-two-models-20261019` spent about $46 under a $50 ceiling. The $3.12 figure is still the spend of `runs/apple-m4-max/p1/fa_flash`.
* **Earlier results are not comparable.** They were produced with smaller budgets, which the dashboard shows per run (and each `result.json`
  records as `budget_usd`): `runs/apple-m4-max/suite_full` is mostly $0.5-$1.2 per kernel, except `gemm_fp16`, whose budget field is now $15; `runs/apple-m4-max/repeat_*` $3 per trial, `runs/apple-m4-max/p1` $3 for most runs (the `*_paper`
  runs started later have $15), `runs/semantic_main` $0.15, `runs/pilot_suite_mini_old_protocol` $1.25-$1.5. A run that stopped on
  `budget_exhausted` there would have continued under the paper's budget.

## Experiments (paper Section 4)

```bash
uv run python -m experiments.run_suite --tasks all --dtypes fp16,bf16     # Figure 4: speedup vs cost, 12 kernels ($15 per run; --budget-scale 0.2 for a smoke run)
uv run python -m experiments.repeatability --tasks conv2d,gemm,swiglu --trials 5   # Table 1 ($15, swiglu $1)
uv run python -m experiments.semantic_tests --n-random 10                 # Section 4.1: 10 random kernels plus 4 misleading ones, $0.15 each
uv run python -m tcenv.cli profile gemm best.metal                        # xctrace: spills, code size, thermal state
uv run python -m observe                                                  # local dashboard at http://127.0.0.1:8765
```

How to read the numbers: the reported speedup of a run is the *confirmed* one (the winner and its ties re-measured with twice the repetitions;
a run that has `run_seed` uses the seed derived for the confirm phase, not a fresh random draw); the suite table also lists the speedup of the task's naive hand-written kernel over the same baselines, because some
baselines are weak (for example Lion and the BitDelta GEMM have no fused library implementation), and a naive kernel that already beats
the baseline means part of any gain is baseline weakness.

## Hardware-feature feedback (optional, off by default)

`search.hw_feedback = true` (`taic run ... --hw-feedback`, `taic resume ... --hw-feedback`, or `hw_feedback=True` in the search options of a driver, as for `profile_poor`)
appends a short block (at most 12 lines, about 150 words) to the evidence written into that round's `TASK.md`, in the place where the
profiler evidence goes, and adds `backends/metal/skills/hardware_feedback.md` to the stored system prompt as Skill 2c. The `pi` session reads the task file, not that prompt. The block is computed by `taic/hwfeedback.py` from the static detector of
`docs/hardware_features.md` (`taic/hwfeatures.py`: the source, the compiled AIR, and the roofline measured for this candidate) and says:

* which Apple-GPU features the kernel uses (matrix unit with its accumulator type and fragment count, threadgroup staging and padding, double buffering or register prefetch, width of the
  device loads and stores, SIMD reductions and shuffles, online softmax, ...) and how far it is from the roofline;
* which features that are expected for this kind of task are missing, at most three, each with one concrete hint (for example "no simdgroup_matrix: the matrix product runs on scalar FMA;
  the roofline shows 3.0x the lower bound"), always with the caveat that these are hints from static analysis, not requirements, and that the measured time decides;
* nothing as missing for a kernel that is already at about 90% of its roofline (hardware features cannot help beyond it; the elementwise tasks are in that state, and scalar loads already
  reach most of the bandwidth there: `docs/hardware_features.md` section 6), and a warning when a kernel has 32 or more accumulator fragments (the reference GEMMs were 12 to 26 times slower)
  or sits at the register-pressure limit.

Why off by default: it is an experiment and nothing yet shows that it helps. With the switch off, `tests/test_hw_feedback.py` checks prompts and feedback against `tests/golden/hw_off.json` (176 system prompts, 110 user messages, 12 feedback texts). That file was recorded again on 2026-10-20, so it is not a snapshot from before the feature existed, and it covers more than four scripted searches. `taic.toml` and `taic.paper.toml` do not mention the switch. It is recorded in `run_config.json` and kept by `taic resume`. The analysis never breaks or slows a search: a timeout
(20 s), a failure or a missing Metal toolchain drops the block and logs a warning. A/B experiment: `experiments.p1_runs` run `fa_flash_hw` is `fa_flash` with only this switch changed
(n=1 on a stochastic search: an anecdote); the run was started on 2026-10-07 06:08, stopped once and continued with `taic resume` after a detector fix (round 1 saw the earlier block; see `docs/hardware_features.md` section 9). The result is on disk: `runs/apple-m4-max/p1/fa_flash_hw` confirmed about 0.622x, spent about $2.89, and stopped after 8 rounds because four rounds brought no new seed. The comparison `fa_flash` confirmed about 0.735x, spent $3.12, 8 rounds, stopped by its $3 budget. Neither file has the source hash the publication gate now requires, so neither number is a published conclusion about the switch.

When the hardware-feature block is on and a baseline is already more than 5% faster than the roofline lower bound, the block does not say the kernel is at the ceiling. That bound counts the direct algorithm's operations, and a baseline that beats it is using fewer of them (the measured case is conv2d: torch-mps at 777 us against a 932 us direct-convolution bound, about 1.2x the measured matrix peak). The reachable target is that baseline's time.

## Search switches (on by default)

Each one is on unless a run passes `--no-<flag>` (`--no-stop-at-roofline`, and the same for the others) or sets the field to false. `truncation_retry` is on the LLM config; the others are on the search config. They are recorded in `run_config.json`. `taic resume` keeps a recorded value; a run that never recorded the switch (started before it existed) is continued with the default, which is on. The six `--no-*` flags turn off the first table. The plan ledger is a separate switch and stays on unless `--no-plan-ledger` is also passed.

| switch | what it changes (on unless `--no-<flag>`) |
| --- | --- |
| `--truncation-retry` | the field stays on and `truncation_retry_effort` still defaults to `low`. The live session does not ask again. The retry lives in the unused `_develop` path |
| `--stop-at-roofline` | stop when the seed is already inside the noise gate of its roofline and no baseline beats that bound (`stop_reason=at_roofline`) |
| `--abi-guard` | the field stays on. The live session does not stop a branch after a repeated ABI error. That stop lives in the unused `_develop` path |
| `--target-note` | when a baseline is more than 5% faster than the roofline bound, the task file says the bound does not hold and the target is that baseline's time |
| `--rich-history` | a failed round's history line names the failure (`abi: ...`, `incorrect: ...`, `no code: ...`) instead of one generic sentence |
| `--status-lines` | the task file is told the noise floor ("at least X% faster") and the round, patience and budget that remain |

In the next table, only the plan ledger is on by default. The live task file still gets the last 8 plain history lines, not the ledger. The other three stay off until a run turns them on. With them off, a search-stage benchmark still times the serial dispatch and every baseline.

| switch | default | what it changes when on |
| --- | --- | --- |
| `--plan-ledger` / `--no-plan-ledger` | on | the dashboard can recompute whether a direction was timed and lost. The live task file does not include that ledger |
| `--diversity-archive` | off | the selection function remains. The live loop does not call it, and `parent_label` stays empty |
| `--fast-bench` | off | search-stage timing skips the serial dispatch and, once this task's fastest baseline name is remembered, times only that baseline. Confirmation still times every baseline and the serial dispatch |
| `--experience-index PATH` | empty | up to three hint lines from measured pairs go into the task file, excluding pairs from the same task. The repeatability experiment forces this empty |

`result.json` and `state.json` also record `search_seed_label` (the kernel the search was mutating) and `confirm_winner_label` (the kernel the fresh-seed remeasure kept). A resume continues from the search seed.

`--max-total-usd` on the suite, the repeatability experiment and `p1_runs` is checked against the dollars reserved on disk: a live run counts its current budget (so a dashboard top-up and a run resumed on the side both count), a finished run counts what it spent. The cap is written to `cap.json` in the experiment directory, and the dashboard refuses a top-up that would break it.

## Dashboard and budget alerts

`uv run python -m observe` serves a local dashboard (default `http://127.0.0.1:8765`; restart it after updating the code, the page is
read at start-up). Views: experiments, arm comparison (published speedup, `fast_p`, dollars per unit of gain), a paged run list, one run, and a spend ledger. The page can add budget, reset budgets, stop runs, change the models used by the next search, and delete a record.
Runs grouped from an old directory name are marked inferred and are not given a winner. The read API is `/api/v2`. It shows every run's budget next to what it spent, and raises an alert for runs that crossed it. The rule is one pure
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
* in the run list: columns are run, task, publication, speedup, binding, and status. Filters are task, dtype, model, and status. The old filter and sort pills are not drawn;
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
* **Resume** of a stopped run: `uv run python -m taic.cli resume <run_dir> [--add-budget USD | --budget TOTAL] [--dry-run]`. Only a directory whose `run_config.json` says `loop_protocol: flame_chase_1` continues; anything else exits 5. It rebuilds the state
  from the directory and goes on with the rounds that were not finished, then confirms and finishes as usual. A round that already has a candidate is not redone. A round with no candidate starts a new session. Before it overwrites anything,
  `result.json` / `confirm_report.json` are kept as `*.before_topup_<n>.json` (and `state.json` as `state.before_resume_<n>.json`), so the earlier
  result is never lost. `--dry-run` rebuilds everything in memory, prints what would be continued (including the checks between the sources) and writes
  nothing, calls no model, uses no GPU. A run that stopped because it used its round limit or its patience can be continued with `--lift-count-limits` when it already has an eligible kernel. A GPU hang or a model error is still not resumable. A run still alive, or one that is being resumed (`<run_dir>/resume.lock` of a live pid), is refused. The dashboard "reset budgets" action adds $15 to every resumable run, raises `max_rounds` and `patience` to 100000, and passes `--lift-count-limits`.

What a resume rebuilds, and from where (`taic/resume.py` has the details):

| state | source | exact? |
|---|---|---|
| money spent | `transcript.jsonl` `agent` events: `cost_usd` is summed. The rebuilt hit, miss, and output token counts are 0 | dollars yes; token split no |
| candidates, reports, best kernel, evaluation cache | `candidates/*/{kernel.metal,report.json}`; selection replayed and compared with the trace in `state.json` | yes |
| hypotheses tried (`history`), stagnation counter | `state.json` | the lines that were saved |
| the interrupted round | a round that already has a candidate is finished. A round with no candidate starts again as a new session and is billed again. `partial` is always empty | yes, for `flame_chase_1` only |
| system prompt | `system_prompt.txt` when that file exists. The next session still gets a newly rendered `TASK.md` | the file, not the transcript text |
| settings | `run_config.json` (new runs, no credentials in it); older runs: budget and round limit from `state.json`; tool use, profiler skill, plan mix, branches, repair tries and initial samples from the recorded prompts and labels | `patience` and `reasoning_effort` are not recorded by older runs: pass `--patience` (default: `taic.toml`, i.e. what runs older than the presets used) |
| profiler evidence of earlier rounds | `profiles/r*_<seed>.{txt,json}` | yes; traces that failed are not listed |

Not recoverable: model requests that were in flight when the process died (the provider may have billed them, the ledger never saw them: the
resume notes list them), and the per-request wall-clock times of earlier sessions (`t` is `null` there). The dry run lists both.

**Dashboard.** On a budget-exhausted / over-budget / interrupted run (and on a running run that reached NEAR BUDGET) the run's card, its banner chip
and the search panel carry an **Add budget** button. The dialog asks for a dollar amount (the field starts at 1). The same page can reset budgets, stop every search, change the two committees used by the next search, and delete a finished record with a reason. After Confirm the page keeps showing
the live state of the run and its budget history. A run that cannot be continued shows the reason instead of the button. The write path (`POST /api/runs/<run>/budget` with `{"amount_usd": 5}` or `{"new_budget_usd": 8}`; `GET /api/runs/<run>/control`
says what is possible) spends money and starts GPU work, so: the server only answers on a loopback `Host`, an `Origin` header is checked only when the request sends one,
every POST needs the per-process random token the served page carries (`X-Observe-Token`), the amount must be positive and at most
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
`experiments/` - `observe/` (local dashboard; it can spend money and stop runs) - `tests/` - `runs/` (artifacts, git-ignored).

`docs/` (Chinese notes): `docs/架构说明.md` how the parts divide the work - `docs/架构/可观测.md` how experiments are grouped and how the dashboard reads them - `docs/诚实说明.md` what the pipeline does **not** prove, and how to check every claim - `docs/算子工程师边界.md` what this route replaces and does not replace in human roles (with the evidence paths). Index of the directory: `docs/README.md`.
