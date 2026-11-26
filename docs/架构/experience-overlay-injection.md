# 经验叠加层的 treatment 注入

P2-T 把 P2-S 冻结的经验叠加层接进实验组（加上要验证的改动的一侧）的真实会话。默认搜索和对照组（不加这项改动的一侧）看不到这份文字。本阶段不启动 `taic run`、`taic resume` 或 `taic track-explore --config`，也不跑 GPU benchmark。实弹搜索未授权、未执行。Plan 3 和 Plan 4 都没有完成。`publication_allowed=false`。NPU 仍是 simulation-only。

`docs/架构/shape-portfolio实施计划.md` 被 P2-I 钉住。本文和 `AGENTS.md` 记录这一阶段。P2-S 的 manifest、证据和合同保持原字节。

## 注入落在哪

写核代理读到的经验是 `prepare_session` 放进会话目录的 `experience.md`。`stage_guidance` 先取这一轮 `agent_guidance/experience.md` 的原字节，再做原来的过滤。

对照组和默认搜索停在这一步。过滤结果是 13415 字节，SHA-256 `e972450e333550739378afe9362e17470bc9f59fba089a849835e28402a8fea2`。

实验组在过滤前把 P2-S 的 `evidence/experience-overlay.md` 接在原文后面。渲染结果是 25469 字节，SHA-256 `8e5adbecd3f764904ceab72c2e35fc7534d08931fde0a24210292cdf23513304`。标记句在会话文件里恰好出现一次。冻结快照本身保持仓库原文，全局 `targets/apple-m4-max/experience.md` 也不写入。

绑定记在这一轮目录的 `experience_overlay.json`。`run_config.json` 已经存在时，同一个不含正文的记录写进顶层 `experience_overlay`。会话目录里的 `experience_overlay.json` 再记一次路径、字节 SHA-256、渲染后 SHA-256、`arm` 和 run。环境变量不构成绑定。

写会话文件之前，这三份记录必须一致：run 目录的 `experience_overlay.json`、`run_config.json` 里的同名对象，以及这次从冻结快照渲染出的摘要。核对包含 `rendered_experience_sha256` 和字节数。只改渲染后摘要也会拒绝，不会留下一份摘要正确、run 记录却是旧值的会话 sidecar。两份文件都在时，删掉 `run_config.json` 里的 `experience_overlay` 键同样拒绝，而且发生在 `device.md` 和会话经验写出之前。`--render-only` 不写 `run_config.json`，只有 run 文件的实验组仍然合法。没有绑定的会话仍只做原来的过滤，并且不读取 `brief.repo`。

`taic/orchestrator.py` 保持 P2-I manifest 里的字节。搜索入口是 `experiments.common.run_one`：编排器写完 `run_config.json` 和冻结快照之后，实验组才调用 `install_run_overlay`。随后的 `prepare_session` 从这一轮目录读绑定。续跑同一目录会再核对这份记录；run 身份对不上就拒绝。另一轮对照目录没有这个文件。

叠加层不是第二份任务合同。正文里出现 `## 任务合同` 或「这是权威任务规则」会被拒绝。空文件、摘要不符、路径越出仓库、标记不是恰好一份，同样拒绝。

## 身份

| 项 | 注入后 |
| --- | --- |
| current_prompt_sha1 | `e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`，`prompt_changed=false` |
| experience.md | `3bc1760a4373329dc633a5193286ca41dbee82f8930ecccab959017675b81dba` |
| 任务包 | 79 个文件，`b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e` |
| 叠加层文件 | `8e3b9f975cb3988e03af09a12e131bfdb8a699631c8780d544d3836f4bdb97cc` |
| gemm_silu fp16 small 的 msl-single 系统提示 | 36031 字节，`b8446c3bfcececdd708561bc8d3e34365044320c9c291efed3812b0da8965f35`，两组相同，不含标记 |

`SearchConfig` 没有新字段。默认 `run_config.json` 不出现 `experience_overlay`。两组的 `init_samples`、`task_ledger`、`plateau_after`、`sweep_tool` 都是对照默认值。委员会仍是 `taic.toml` 的 first/second。

经验计划的 `search_eval_seed` 是 7。`--execute` 把这颗值作为 `run_one(..., seed=7)` 传给两组，编排器写成 `run_config.json` 的 `seed` 和 `run_seed`。两组的确认 master key 都是 `confirmation_master_seed(7)`。四开关计划不传 `seed`，每次运行仍各自抽取 `run_seed`。

v2 搜索期的评测请求是编排器里的字面量 7（`phase_seed = 7 if self._v2`）。trace 里 `eval_seed_from=search` 的 `eval_seed` 记的是这颗字面量，汇总器要求它等于 7。这不是 `eval_seed(run_seed, phase, label, source)` 的返回值。那个函数给 v1 搜索以及 confirm、recheck、domain 派生 48 位种子，`eval_seed(7, "search", ...)` 不是 7。确认行不要求等于 7。

经验计划的配对因此看两件事：搜索期 `eval_seed` 是字面量 7，并且两边 `result.json` 的 `run_seed` 都是 7。只改计划字段，或只在 `--render-only` 的 manifest 里抄上 7，都不会写出 `run_config.json`。四开关汇总不读 `run_seed`。

执行前还核对冻结矩阵。每个 pair 的 task、dtype、size、rep 必须相同，control 和 treatment 各一格，奇数 pair 是 AB，偶数 pair 是 BA，格子数等于 `tasks × pairs × 2`。`committees` 必须等于当时从 `taic.toml` 读出的 first/second，不能只当展示字段。格子或委员会对不上，就在调用 runner 之前拒绝。

规范化掉本轮绝对路径之后，两组的 `TASK.md` 相同。系统提示相同。差别只在会话里的 `experience.md` 和那份摘要 sidecar。

## 渲染与批准

`experiments/search_yield_ab.py --factor experience-overlay` 的 dry-run 写出 `injection_ready: true`。这表示注入路径已经接上，不表示搜索已经获批。

`--render-only` 对四个 cell 调用 `prepare_session`，写出 `render-only.json`。它不产生候选，不调用模型，不是搜索结果。实弹搜索仍是未授权/未执行。

`--execute` 仍要四道门禁：`--allow-spend`、`--allow-gpu`、非空 `--approved-by`、有限正数 `--delta`。缺任何一道就在创建目录之前返回 2。四道都在时，实验组把叠加层路径和字节摘要交给 runner，对照组不带这两项。本阶段的批准人、δ 和可容忍超支是空的，验收不调用这个 runner 的生产实现。

P2-S 的 `manifest.sha256` 保持原字节。现场 `experiments/search_yield_ab.py` 和 `tests/shared/test_experience_search_prep.py` 相对那份 manifest 的摘要已经不同。其余 P2-S 交付物仍与 manifest 一致。

## 仍未授权

- 批准人、δ、可容忍超支、执行记录都是空的。配对结果即使以后跑出来，也还不能下结论。
- $12 仍是「下一格 cell 能不能启动」的承诺上限，不是账单天花板。
- 全局 `experience.md` 不能写入这份叠加层，否则 `current_prompt_sha1` 会移动。
- Plan 3 没有完成。Plan 4 没有开始。
- `publication_allowed=false`。NPU 仍是 simulation-only。
