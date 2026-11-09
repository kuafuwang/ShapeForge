# MT5：轻量多 Track 批量探索与结果汇总（fake agent / CPU fixture 完成）

## 1. 范围结论

本包完成 MT5 的 **版本化输入、串行外层启动器、独立结果读取/校验、单/多实例确定性汇总、预算/超时/搜索成本记录和完整 CPU fixture 负例**。

**没有运行真实 `taic run` / `taic resume` / 生产 `track-explore`，没有真实模型 kernel 搜索、付费实验、GPU/NPU 执行或性能结论。** `SubprocessSingleTrackRunner` 只是对既有 `taic run` 的生产适配；本轮测试均注入确定性 fake runner。MT0 仍 blocked，未开始 MT1b/MT2–MT7。

## 2. 接口与信任边界

### 2.1 输入与产物

`taic/track_explore.py` 实现 `taic.track-explore.1`；`taic/cli.py` 提供：

```bash
# 生产接口，真实执行等同 taic run，需另行模型/GPU/费用授权
uv run python -m taic.cli track-explore --config track_explore.json

# 只读已有证据，本包固定安全入口
uv run python -m taic.cli track-explore --config track_explore.json --summarize-only
```

输入绑定：task/target、参与 Track/version/run_dir/build timeout/generation、非空去重 benchmarks、run/master seed、输入生成器与派生版本、`comparability_digest`、shared generation/correctness/baseline/evaluation/confirmation、串行策略、统一 `max_agent_turns`、`budget_scope`、USD/wall 上限、输出路径和 selection policy version。

每条 Track 的 `run_dir` 是独立根目录，每个实例使用确定性的 `<root>/<dtype>__<size>__<usage>` 子目录。不同 Track 根目录在路径解析（含 `..` 与 symlink）后不得相同或嵌套。输出旁路 `*-observations.json` 只记外层 wall/returncode/build timeout/budget/实际花费，不进入 run_dir，避免让既有 `taic run` 把目录判为非空。

生产 runner 的 `--dtype/--size` 不再经过 `taic run` 的静态 fp16/bf16、paper/small/plan 白名单：`taic.cli` 在启动前用实际任务 OpSpec 装配并 typed refusal，因而 `kda single/small/tail`、`varlen_kda fixed/mixed/uniform/...` 都可进入同一 CLI。`SubprocessSingleTrackRunner.argv()` / `environment()` 可独立做无设备测试；固定 `kda single` 与 `varlen_kda mixed` 的 argv 均通过真实 parser 和 OpSpec 装配，未声明实例则 typed refusal。

每个 track×instance 都注入 `TCENV_EVAL_CACHE_DIR=<run_dir>/cache`，不再共享 `~/.cache/ai_lowerling/evals`。summary 的 `cache_identity` 记录实际目录、逐搜索报告 cache_hit/hit_count/miss_count/hits 及 confirmation cache_hit；目录不匹配、缺 cache identity、搜索 cache_hit 都 fail closed。独立 run_dir 并不被当作独立 cache 的替代证明。

### 2.2 外层只做四件事

1. 串行启动缺失的单 Track run；
2. 读取 result/run_config/schema-4/subject sidecar；
3. 复核身份、独立性和唯一发布谓词；
4. 写出探索 summary。

没有中央 Host Scheduler、统一 DSL IR、第二套 correctness/baseline/timing/evaluator、TrackRegistry 自动选路或跨 Track runtime portfolio。生产 runner 只拼现有 `taic run` 参数；不重新实现 Orchestrator 内部搜索。

`generation_policy='parallel'` 在 schema 1 被 typed 拒绝：当前单 Track run 没有分离 generation 与 quiet confirmation 的接口，不能安全地并行生成后又串行确认。默认且唯一实现的是串行，未用并发全确认冒充“并行 generation”。

### 2.3 方案 A，而不是同 fresh 输入

- run_seed 相同；master_seed 必须等于现有 `confirmation_master_seed(run_seed)` 的 hex，派生算法不变。
- 各 Track 走自己的现有 fresh confirmation。MT1a 的 adapter 服务独立重建 freeze/ValidationPlan，并核验实际 `confirmation_seed`。
- 每个实例分别记录 freeze/validation digest、实际 seed、coverage kind，以及 fresh input identity。fixed-shape 报告的 `input_sha256` 本来就为空，因此身份是 `task_pack_digest + instance_key + case + seed` 的 `derived_input_identity`，并在 `identity_scope` 明确“没有 per-case 字节摘要”，不伪造成输入字节 hash；variable-length（L1/L2）实例必须保留非空 `input_sha256` 或 `evaluation_input_digest`，缺失才是 `input_identity_unavailable` 并淘汰对应 Track bundle，而不是把 fixed-shape 错杀。
- summary 永远写 `comparison_mode=scheme_A_independent_fresh_confirmation_same_distribution`、`same_fresh_input=false`、`formal_publication=false`。
- 不实施方案 B，不派生候选无关的新 seed，不创建 graph 或新 confirmation epoch。

候选有效性先经过现有 `conclusion_verdict`；subject sidecar 再经 `verify_subject_document`。另检查 candidate provider/credit、full/native/fresh/quiet、transport 与 candidate identity、Track/version/delivery、run/master、task/instance/target、build/evaluation identity、task pack 和 metric 互为倒数。自报 search best、fallback、cache hit、incorrect/unsafe、timing uncertain、失败 batch 或 Track 漂移均不得入选。

### 2.4 Comparability 与独立性

共同 canonical 元组包含 opspec/reference、排序后的实例、baseline implementation/environment、target/device、protocol/timing/eval_reps、输入生成器与派生版本。输入中的 digest 必须等于重算值；每个 run 又从实际报告/合同重建，不一致仅展示，不参与选优。每条 run 的 build/evaluation identity 仍分别保留。显式 `shared.baseline=null` 表示 correctness-only 无 baseline，此时从 comparability 元组中省略 baseline 字段，不要求用户编造 digest；有 baseline 时两个 digest 必须成对且合法。

除独立目录外，还做：

- result/run_config/confirm/subject 的 symlink 拒绝；
- `(st_dev,st_ino)` 跨 run 复用检测，硬链接 confirmation 也算 cache 串用；
- run_config/result Track 一致性与 frozen task-pack/guidance 检查；
- fresh confirmation 的 cache_hit/native/full 检查。

## 3. 冻结的选择规则

Selection policy 为 `track-explore-selection.1`，固定 digest：

`22e4283765d35cba939699a10cfac70592f8f6147d2ec8100fb0ac06cd3ca5f8`

- `latency_ratio = candidate/baseline`，`speedup = baseline/candidate`；实际报告三者必须一致。
- 单实例：先过现有发布/identity/fresh/quiet 门，再按确认的 ratio 升序；`1.10` 恰好允许。没有额外 `speedup > 1.0` 门。
- 多实例：每条 Track 各取自己的 per-instance best 组成完整向量；缺实例或任一实例 ratio > 1.10 淘汰整条 Track。过门后按等权 geomean 升序，worst-shard 只做并列 tie-break，之后是候选 identity vector 与 Track identity。
- `sum(candidate)/sum(baseline)` 仅展示，不参与排名；固定测试证明长耗时实例不再垄断。
- winner 是 `(track_id,track_version,per-instance candidate identity vector)`；`exploration_bundle_digest` 由 `track_id + track_version + per-instance candidate identity vector` 共同 canonical 计算，不能只哈希向量。向量按 instance_key 排序，不能跨 Track 拼冠军。
- 没有有效 Track 时为 `status=no_winner`、`winner=null`，不降门制造胜者。golden/correctness-only 行只展示，不产生性能 winner。

## 4. 预算、超时与搜索成本

- `per_track_instance`：每个 track×instance 独立预算；`per_track`：同 Track 后续实例只获得剩余预算，两种 scope 不混算。
- 模型在飞请求可能超过既有软预算，summary 保留实际超支，不把它伪装成硬 cap，也不删除已产生的证据。
- 每个 run 在任何 eligibility return 前都挂载 run_identity、candidate/build/evaluation identity（若存在）、confirmation_status、agent turns、schema-4 candidate + independent confirmation 构建耗时及计算依据、外层 wall、spent USD、build timeout、budget allocation、returncode/timeout；incorrect/unsafe/no-baseline/timing_uncertain/cache_reuse/track_mismatch 和 golden 行也保留这些字段，不是只汇总 wall/USD。`agent_turns` 的语义是每个 `event=agent` 的 billable hmz session attempt，包含 initial、后续 rounds、initial restart 和 provider retry，不是内部 tool message；`max_rounds=max_agent_turns-1` 仅是正常路径上限，retry/restart 超额会淘汰，不能悄悄多烧轮次。
- `shared.evaluation.eval_reps` 通过现有 `--eval-reps` 传入；`shared.correctness.correctness_only` 通过现有 BooleanOptionalAction 开关传入；`shared.confirmation.confirm_batches` 通过现有 `--confirm-batches` 传入。当前 v2 subject 固定 `confirm_batches=1`、search reps=3、confirmation reps=15；summary 同时核对 run_config.confirm_batches、subject.confirmation_attempts 和 benchmark.attempts，不能用自报配置顶替实际协议。
- pre-MT1a run_config 缺少 guidance digest 时，从该 run 的 `system_prompt.txt` 与 presentation statement 重算，复用 Orchestrator 的同一 guidance 规则；文件也缺失则为 typed `confirmation_guidance_unavailable`，不再以笼统 `seed_derivation_unavailable` 淘汰。
- 构建成本汇总所有可用 compile/build duration，逐颗保留 `failed_reports` 与 `missing_duration_reports`。compile_error 不再吞掉其它成功构建成本；只有确实缺数据才 `cost_complete=false`。
- 总 wall/USD 对所有已启动的 track×instance 求和，包含 golden/correctness-only，与最佳性能分开；没有任何行或确实缺数据时才 `cost_complete=false`，不拿候选速度代替搜索成本。
- `TAIC_TRACK_BUILD_TIMEOUT_SEC` 只在生产子进程内设置，未设置时旧行为不变；非法/非有限/超过 worker timeout 均拒绝。
- 外层总 wall 超时、runner 非零返回和非空 partial run 都有处理。partial run 进入 resume 互锁，不自动覆盖重跑；重复完整运行不会再次启动付费工作，summary 字节保持幂等。

## 5. 结构化 fixture 证据

`evidence/` 包含完整全成功示例：版本化 input、observations、summary 和 trust-scope metadata。固定候选来自既有 P2 variant 字节：

- `msl-single`：prefix `1209a6d886` 的单源交付；
- `msl-bundle`：prefix `22c5880e60` 的精确 envelope 交付。

每个 fixture run 仍通过真实读盘发布谓词与 subject verifier。示例 winner 为 `msl-bundle`（fixture ratio 0.8 对 0.9），两条 Track 具有不同 freeze、confirmation seed 和 fresh input identity。示例 wall=21.0、USD=0.75 都是脚本成本，不是账单或性能测量。路径仅把临时根替换为 `<fixture>`，身份摘要不变。

固定 45 个 node id 及正/负分类见 `nodes.json`。覆盖：fixed-shape 身份、golden 成本、全失败 run 成本、confirm_batches 负例、pre-MT1a guidance，以及全成功、真实 parser/OpSpec 的 runner argv、shared 配置进入现有 run、独立 cache 目录与 cache_hit、空 varlen input identity、Track 绑定的 bundle digest、路径别名/symlink、agent-turn 超额，以及全成功、幂等、incorrect/unsafe、显式 no baseline、golden unranked、timing uncertain、目录/hardlink cache 串用、Track 漂移、不同 baseline、非 fresh seed、1.10 边界、geomean、worst/tie-break、缺实例、局部失败不可被 geomean 掩盖、版本化输入、per-track 预算、observation budget/run_seed 篡改、timeout/runner failure、selection policy golden 和安全 CLI summarize-only。

`verify.py` 还固定 4 个 schema 负例（master、comparability、parallel、共享目录），独立重放全成功示例与幂等，并要求精确 45 passed 及 manifest 一致。

共同起点：`../start-state.json` / `../start-worktree.sha256`，HEAD `efb8218587a3deb9752a0f9dc70035afe5fb26e2`。没有修改 R0–R6/G0/A1–A6/P0–P2、六 RPC、field number、HMAC 或 BlobChannel，也没有回退其他会话的文档改动。全量 CPU 回归及环境失败分类见 `../cpu-regression.json`。

## 6. 未做事项与下一授权点

- 未做真实 Metal 批量验收（MT6）、真实 DSL/LLM 接入（MT2–MT4）、方案 B、graph、跨 Track runtime portfolio 或正式发布。
- 真实生产 runner 若要执行，必须另行授权模型/GPU/费用，并设置预算与时间上限；CPU fixture 不能写成真实 Metal 或真实模型验收。
- 并行 generation 需要先有单 Track 的 generation/confirmation 分离接口及独立评审；不能由本外层脚本伪造。
- 最终 reviewer 结论只通过本轮合法 `{done, notes}` JSON 返回，不伪造独立 RLAR 签名。
