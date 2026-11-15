# 按任务检索的紧凑经验 overlay

P2-V 是 P2-U `verdict=no_gain` 之后的下一版经验文本。它只为 `gemm_silu` / `fp16` / `small` / `apple-m4-max` 准备一份短叠加层，并沿用 P2-T 已经验证的 treatment 注入。本阶段不安装为默认经验，不启动 `taic run`、`taic resume` 或 `taic track-explore --config`，也不跑 GPU benchmark。这是一次新的未批准实验。实弹搜索未授权、未执行。Plan 4 未开始。`adopt_compact_overlay=false`。`publication_allowed=false`。NPU 仍是 simulation-only。

`docs/架构/shape-portfolio实施计划.md` 被 P2-I 钉住。本文和 `AGENTS.md` 记录这一阶段。P2-S 的完整叠加层、P2-U 的 `no_gain` 结论和 P2-I 到 P2-U 的 manifest 保持原字节。

## 怎么选中

选择器是精确四元组，不是关键词过滤。唯一投影是：

| 字段 | 值 |
| --- | --- |
| task | `gemm_silu` |
| dtype | `fp16` |
| size | `small` |
| target_id | `apple-m4-max` |
| lesson_id | `gemm-silu-reg` |
| projection_id | `gemm-silu-fp16-small` |

目录里 `gemm-silu-reg` 同时有 small 和 paper。正文只取 small 这一行。paper、`reg_speedup_vs_best` 和 `hard_lower_bound_over_reg` 只出现在投影记录的省略字段里，数值不抄进叠加层。其余课按 id 省略：`conv2d-paper-shared-u`、`conv2d-paper-scalar-ablation`、`conv2d-small-single-stage`、`gemm-silu-occupied-retry`、`bitdelta-reg`、`bitdelta-small-search-best`、`gemm-splitk-stop`、`seq-serial-handoff`、`seq-tail-decision-measurement`、`seq-single-self-check`、`p2r-worker-geometry`。大小写不同、子串相同或换成 paper / 别的任务，都拒绝。

## 正文留了什么

冻结文件是 `tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/evidence/compact-overlay.md`，974 字节，SHA-256 `fd72abf078d9d5a7cd2118ee5263a3c83e5695c044c0bc0cc6ca86758a0aa3d1`。

留下的观察只有这一条，数字来自 P2-O 的 fresh subject 和 probe，写正文之前要和 `decision.json`、`scratch-probe.json`、`reg-small-confirmation.json` 以及 `docs/架构/gemm-silu-scratch-headroom.md` 对上：

- 一次性 SiLU 中间结果留在寄存器。
- GLOBAL scratch 不省 FLOPs，还增加 fp32 写读、barrier、producer/consumer 和 dispatch。没有写 GLOBAL 候选。
- REG fresh 1.8252 µs。原值是 1.8252153147444679 µs，确认种子 3775173618909035，identity `match`，搜索种子 7。
- 两阶段 no-fill floor 8.7713 µs。原值是 8.771250140853226 µs，`fill_bytes_per_iteration=0`，`stages=2`，kernel 是 `probe_touch_producer` 和 `probe_touch_consumer`，`not_a_candidate=true`。
- 这两个微秒数是已测结构边界，不是发布 speedup。候选仍自有算法与 tiling。正文不指定候选 tiling。正确性与 fresh identity 仍按原合同。`publication_allowed=false`。

`Confidence: verified` 是既有过滤器留下这一节的标记，和完整叠加层使用的是同一种标记。标记句仍然是 P2-S 那一句，恰好一份。叠加层不是第二份任务合同。

## 渲染

注入面仍是 `prepare_session` 写进会话的 `experience.md`。没有第二条通道。审计字段集保持 P2-T 的原样；紧凑和完整靠路径、字节 SHA-256 和渲染后 SHA-256 区分。

| 文本 | 字节 | SHA-256 |
| --- | --- | --- |
| 对照会话 | 13415 | `e972450e333550739378afe9362e17470bc9f59fba089a849835e28402a8fea2` |
| 完整叠加层文件 | 12059 | `8e3b9f975cb3988e03af09a12e131bfdb8a699631c8780d544d3836f4bdb97cc` |
| 完整处理会话 | 25469 | `8e5adbecd3f764904ceab72c2e35fc7534d08931fde0a24210292cdf23513304` |
| 紧凑叠加层文件 | 974 | `fd72abf078d9d5a7cd2118ee5263a3c83e5695c044c0bc0cc6ca86758a0aa3d1` |
| 紧凑处理会话 | 14389 | `93b65a30a3b1de5addaabda1e1249cf72295e7396d6627981baaa5511c748c81` |

P2-U 测到的会话增量是 25469 − 13415 = 12054。紧凑处理会话的增量是 14389 − 13415 = 974，和紧凑文件字节数相同。去掉的是 11080 字节，占原来 12054 字节增量的 0.9191969470715116。完整文件比那次会话增量多 5 字节；那份 12059 字节的 P2-S 文件保持原字节，本阶段不改它。

对照臂 0 份。处理臂恰好 1 份。系统提示两臂相同，36031 字节里的摘要仍是 `b8446c3bfcececdd708561bc8d3e34365044320c9c291efed3812b0da8965f35`，不含叠加层。篡改渲染摘要、错臂、把记录拷进另一轮，都在写会话文件之前拒绝。错任务和 paper 形状在写 `runs/` 目录之前拒绝。

## 新的 dry-run

计划在 `tests/support/golden/shared/shape-portfolio-阶段验收/P2-V/evidence/dry-run-plan.json`。schema 是 `shapeforge.p2v-compact-overlay-plan.1`，epoch 是 `p2v-compact-experience-overlay-1`，实验 id 是 `p2v-compact-experience-overlay`。内层仍是 P2-T 的经验计划：`gemm_silu`、fp16、small、两对、AB/BA、每格软预算 $3、$12 仍是下一格能不能启动的承诺上限。`approved_by`、`delta`、`tolerable_overrun`、`execution_record` 都是空的，`executed=false`。

P2-U 的结论仍是 `verdict=no_gain`、`adopt_overlay=false`。那次实验的批准人、δ = 0.01 和后来提到 $15 的提额留在 P2-U，`inherits_p2u_approval=false`。`experiments/experience_compact.py --execute` 以及 `experiments/search_yield_ab.py --execute` 带上 `--allow-spend`、`--allow-gpu`、批准人和正数 δ，仍然在创建目录和调用 runner 之前返回 2。完整叠加层的四道门禁行为保持 P2-T：门禁齐备时仍会进入既有 runner 路径。紧凑文件不进入那条路径。

`--render-only` 写出四个 cell 的真实会话。它不产生候选，不调用模型，不是搜索结果。实弹搜索仍是未授权/未执行。

## 身份

| 项 | P2-V |
| --- | --- |
| current_prompt_sha1 | `e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`，`prompt_changed=false` |
| experience.md | `3bc1760a4373329dc633a5193286ca41dbee82f8930ecccab959017675b81dba` |
| 任务包 | 79 个文件，`b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e` |
| parent HEAD | `d3388497aabaf2a687f3f1d65801c87edd9fdd9c` |
| 上一份 parent HEAD | `a9c1dccce636107c2852995578b05611bf19865b` |
| orchestrator.py | `458cd29093fd36fd87fb2f500d8bcbe935bfa2d1d6a0fb583bde4b7c868795bd` |
| 上一份 orchestrator.py | `35832a11e22bf3a5a222f541157ac85e02a5a2d17d2ad925dc575fb2511e9ef1` |
| 现场 `experiments/search_yield_ab.py` | `413df71af15af76524aaea28ed80e581543b56365b3ff0be4185ed25734fbca6` |
| 现场 `taic/experience_overlay.py` | `55ebc524f4c0b0feb6c23325b1e3cc77512ad0519b83ab122cc46b3cb8211ec5` |
| 两臂归一化 TASK.md | `7ad7dfe8a3be4ed0245f07ffa8bd27abc50e52507a41bb47f2d8b7fb73d028bb` |

`d3388497aabaf2a687f3f1d65801c87edd9fdd9c` 的提交说明是 `fix: stop v2 task text from requiring eval-v2 --validate`。它只改了已提交的 `taic/orchestrator.py` 和 `tests/shared/test_search_v2_bundle_prompt.py`：v2 的 `brief.selfcheck_allow_validate=False`，TASK.md 不再要求不存在的 `eval-v2 --validate`。这次提交不改 experience 渲染、`current_prompt_sha1` 或 79 文件包。对照会话仍是 13415 字节，完整处理会话仍是 25469 字节，紧凑处理会话仍是 14389 字节，渲染 SHA-256 仍是 `93b65a30a3b1de5addaabda1e1249cf72295e7396d6627981baaa5511c748c81`。系统提示摘要仍是 `b8446c3bfcececdd708561bc8d3e34365044320c9c291efed3812b0da8965f35`。

`seedless_kernel_brief` 原来把 `selfcheck_allow_validate` 留在默认的 True，渲染出来的任务书仍要求 `--validate`，和编排器的 seedless brief 不再逐字段相等。最小修复是 v2 也写成 False。experience.md 的字节不因此改变。未付费渲染里，对照臂和处理臂的归一化 TASK.md 都是 `7ad7dfe8a3be4ed0245f07ffa8bd27abc50e52507a41bb47f2d8b7fb73d028bb`，两份都不含 `--validate`，都含 `eval-v2`。

未来 compact 搜索的对照臂和处理臂走同一条编排器路径，共享这份修正后的 TASK.md。配对内部仍然只有 overlay 差别。若拿这次尚未执行的方案和 P2-U 已经跑过的会话比，任务书也变了，不能把 overlay 长度说成唯一变量。

P2-T manifest 里的 `experiments/search_yield_ab.py` 仍钉在 `1ce450fb9804e1efdd624bb3d51542d10f6ea35fd68d0051170e4a46686cd7f8`，`taic/experience_overlay.py` 仍钉在 `982439a7233b1f3261a7cacd342dee877d3cbce37247c9c1c592e2408f7df4db`。P2-I manifest 里的 `taic/orchestrator.py` 仍钉在 `35832a11e22bf3a5a222f541157ac85e02a5a2d17d2ad925dc575fb2511e9ef1`。P2-M manifest 里的 `tests/shared/test_search_v2_bundle_prompt.py` 仍钉在 `1ecefb12391d0faf02083afaa11d02050d1a8eec802fb6a9fa131245bd31bf6b`。P2-S 的已知漂移集合仍是 `experiments/search_yield_ab.py` 和 `tests/shared/test_experience_search_prep.py`。`taic/kernel_agent.py` 仍与 P2-T manifest 的 `c5ca855cb3869744c279483bd68c9109e7f024a6ef0154325eacca1dfb052ecf` 一致。这些历史 manifest 和 P2-I 到 P2-U 的证据没有改写。

现在重跑 P2-T 或 P2-U 的 `verify.py` 会失败。这是历史锚点漂移：两份 acceptance-baseline 的 parent HEAD 仍是 `a9c1dccce636107c2852995578b05611bf19865b`，orchestrator pin 仍是 `35832a11e22bf3a5a222f541157ac85e02a5a2d17d2ad925dc575fb2511e9ef1`，而现场 HEAD 和 orchestrator 已经是上一张表里的新值。P2-T 的 dry-run `git_head` 也仍是旧 HEAD。P2-U 自己的 history 检查仍要求 P2-I 只漂 `tcenv/v2_worker.py`、其余已钉文件逐字节相同。这些失败不能靠改写 P2-T 或 P2-U 消掉。

## 仍未授权

- 新实验的批准人、δ、可容忍超支和执行记录都是空的。
- P2-U 的批准不能启动这一次。
- 就算补上四道门禁，紧凑实验现在也会拒绝执行。要执行必须另开阶段，并改掉这道拒绝。
- 全局 `experience.md` 不能写入这份叠加层，否则 `current_prompt_sha1` 会移动。
- `adopt_compact_overlay=false`。Plan 4 未开始。
- `publication_allowed=false`。NPU 仍是 simulation-only。
- 和 P2-U 已经执行的会话比较时，任务书也变了，不能把 overlay 长度说成唯一变量。
- `review-r1-superseded.json` 原样保留 d338849 之前的 `done=true`。那份签名不覆盖当前 HEAD。
