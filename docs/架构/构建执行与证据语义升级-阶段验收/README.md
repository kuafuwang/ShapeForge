# 阶段验收总表：构建、执行与证据语义升级

状态：R1、R2 已完成并通过硬验收；R3 的 CPU 侧硬验收（§5.3-1…4）已通过、第 5 条真机验收 blocked；**R4 通过（CPU 侧）**；**R6 已通过（最终验收见 `R6.md` §13）**；**R5 已通过（最终验收见 `R5.md` §26）**——§18 的最终验收之后，评审又用可复现负例否掉了「finalize 冻结的 expected 与各批比较用的那份不同」这条旁路，修复与负例见 `R5.md` §19；随后评审又用负例否掉了「冻结 expected 的不完整字段仍可发布」，修复见 `R5.md` §20；第三次评审又发现 `experiments/plan_type_analysis.py::_published_speed` 另有一套独立发布判定，修复见 `R5.md` §21；第四次评审又发现 `protocol_version`/`target_id` 两边同时缺失（`None == None`）被当成一致，修复见 `R5.md` §22；第五次评审一次给出五条硬阻断（读盘未绑定报告候选/构建资格、读盘边界只查版本号、provenance 双缺失、`result.target_id` 脱离身份、outbox 行/result 身份矛盾），修复见 `R5.md` §23；第六次评审又发现三条（保留 result 报告未过 schema-4 闭包、历史 epoch 双缺失、确认阶段未分类异常被吞），修复见 `R5.md` §24；第七次评审又发现 retained result 用普通 `json.loads` 读，重复键在 schema 闭包前丢失，修复见 `R5.md` §25；最终验收又发现本次引入的唯一未登记回归（`fake_worker_exclusive` 报告无 build），修复见 `R5.md` §26：全量 33 failed / 3158 passed / 7 skipped，R0 node-id 双向差集为空，**R5 已通过**。此前八轮评审每轮都用可复现负例否掉了当时的「已完成」：①谓词接受批次记录不全、确认异常仍是自由文本、analysis/brief 把旧格式当当前证据（`R5.md` §15、`R6.md` §7）；②候选面板原样返回旧报告正文、`hwfeatures` 无 schema 门（§8）；③`hw_features` 的 trace/发布值/参考摘要、`hwfeedback` 的 roofline、brief 的 trace 主路径（§9）；
④`hwfeedback` 的缓存键在 schema 门之前命中（§10）；⑤`4.0 == 4` 的类型旁路（确认门 + 四个读取入口）与
`hw_feedback_ab` 的宽松读取（§11）；⑥`timing_uncertain` 的缺失/null 值曾可发布（`R5.md` §16）；⑦顶层 `cache_hit` 的缺失/null 值曾可发布（`R5.md` §17）。这些修复与负例都保留在对应小节；最终验收逐条复核了 §9 的 R5/R6 硬验收，并补上「已确认候选不被后来失败候选推翻」的直接用例。R7 未授权（blocked）。

本目录只记录**实际执行**的阶段验收：实现/删除的文件与符号、逐条硬验收映射、真实命令与结果、
与基线失败集的差异、未决事项。它不是规范（规范见 `../构建执行与证据语义升级-plan.md`），也不是
R0 的冻结证据（那在 `../构建执行与证据语义升级-R0/`，被 `manifest.sha256` 与 `verify.py` 封条，
本流程不修改其中任何文件）。

| 阶段 | 状态 | 验收文档 | 关键结论 |
| --- | --- | --- | --- |
| R0 | 已完成（冻结+登记，未实现） | `../构建执行与证据语义升级-R0/README.md` | `verify.py` OK；20 归档文件 / 309 未跟踪分类 |
| R1 | **已通过** | [`R1.md`](R1.md) | 四个 `fp16`/`bf16` × `paper`/`small` 组合与 R0 冻结实现逐字节一致；新增 78 个 CPU 用例；全量快速套件与基线失败集完全相同 |
| R2 | **已通过（含复核修正 §15）** | [`R2.md`](R2.md) | 全部静态入口（12 `TASK_NAMES` + 4 phase2 + 4 固定 `sem_*` + 4 `gen_*`）与 R0 冻结实现逐字段/逐字节等价；`sem_int_<seed>` 与 GEMM/FA variant 两个动态族按域验收；旧路径（`_elementwise.py`、`elementwise.toml`、闭包 `repr` 身份、pre-v2 指纹、两处 `gpu` 覆盖）全部删除 |
| R3 | **CPU 侧通过（含复核修正 §6–§9）；真机 blocked** | [`R3.md`](R3.md) | 唯一构建入口：`compile_source`/`_compile_probe_source` 删除，probes/标定/内部 fill 全部走 `artifact_builder` + 登记诊断 profile，`BuildError` 带 typed provenance/`assembled_sha256`，`BuildError(deadline)`→timeout；AST 结构检查替换正则；14 个基线失败消失 |
| R4 | **通过（CPU 侧）** | [`R4.md`](R4.md) | schema 4 唯一读写（`migrate_report` 删除）、`build` 闭集（full/partial，native/simulator/source_only，received 预构建）、`report_schema_epoch` 续跑闸门、`EVAL_CACHE_SCHEMA=3` 与身份缺失禁用缓存、descriptor×payload 两端校验、`simulated` 键退场、Go/评估向量与 metal fixtures 重建；R7 真机项 blocked |
| R5 | **已通过（§26 最终验收；§19–§25 的阻断修复全部关闭）** | [`R5.md`](R5.md) | 唯一发布谓词与单一身份比较器（`scope=local|remote`）；本地确认门与逐批证据；读盘共用谓词；远端上下文固定 schema v1 投影、`_save` 写入、`load_completed_context` 只读校验、`restore` 投影↔行/result 逐项对账；每批确认带自己的已校验上下文（远端字段不再因缺上下文一律 unknown）；远端 expected 的冻结环境分量；恢复边界与二次 restore 复用；§19：每颗候选只冻结一份 expected、finalize 重算只作校验（漂移即 `build_identity_mismatch`/`evaluation_identity_mismatch`），谓词新增「冻结 expected↔报告身份」一致性检查，读盘同拒 |
| R6 | **已通过（最终验收 `R6.md` §13）** | [`R6.md`](R6.md) | 只有 schema 4 是当前证据：全部合成确认文档经 `synthetic_confirmation_documents` 必须通过唯一谓词；索引升到 13 并新增 `report_state`/`build_shape`/`build_kind`/`evaluation_fingerprint`（旧报告标 `old_format`、三列 NULL）；detail/query 展示；analysis/feedback/brief/review/experience 拒绝旧格式；死掉的旧发布谓词删除；NPU 离线向量与 Go 一致性 232 用例通过 |
| R7 | **blocked（未授权）** | 见下表 | 真实设备工作一件未做 |

## R7 未授权事项（只登记，不得宣称完成）

| 事项 | 需要的授权 | 当前状态 |
| --- | --- | --- |
| 真实 Mac 上有效/无效 MSL 编译、单核/多核 pipeline、诊断行号 | 打开 GPU | blocked |
| `candidate_sha256`/`assembled_sha256`/`native_digest` 与 live handle 绑定（真机） | 打开 GPU | blocked |
| 固定版本 probe 只建临时 library、无 pipeline（真机核对） | 打开 GPU | blocked |
| `tests/backends/metal/` 的 `gpu` 标记用例 | 打开 GPU | blocked（本流程一律 `-m 'not gpu and not llm and not slow'`） |
| 真实 NPU 工具链/SDK/驱动/设备与真实评测发布 | 设备 + 厂商 SDK + 付费授权 | blocked |
| `taic run` / 真正 `taic resume` / 付费搜索 | 付费授权 | blocked |

## 基线（供后续阶段归因）

2026-10-29 在当前工作树（HEAD `3fb9ac85bfc7` + R0 冻结的未提交草稿）实测
`uv run pytest -q -m 'not gpu and not llm and not slow'`：

| 时点 | 结果 |
| --- | --- |
| 本流程开始时（R1 之前） | 47 failed / 2708 passed / 6 skipped |
| R1 之后 | 47 failed / 2786 passed / 6 skipped |
| R2 增量（harness 第三次修正 + 容差判决用例） | 47 failed / 2803 passed / 6 skipped |
| R2 通过时（`gen_*`/bitdelta/mamba2/variant/删除项全部落地） | 47 failed / 2846 passed / 6 skipped（失败 node-id 集与基线 `comm` 双向为空） |
| R3 CPU 侧完成时（唯一构建路径 + 结构检查替换） | 33 failed / 2868 passed / 6 skipped（14 个基线失败消失：13 `tool_protocol` + 1 `test_structure`；无新增） |
| R3 复核修正后（§7 null digest、§8–§9 内部 fill 行为 spy 与 GPU 接线） | 33 failed / 2879 passed / 6 skipped（`comm -13` 仍为空） |
| R4 实现完成（schema 4 + 身份 + 网关；R5 尚未开始时的定位点） | 37 failed / 2928 passed / 6 skipped |
| R4 通过、R5 本地确认门落地（确认门收紧后） | 90 failed / 2902 passed / 6 skipped |
| R5 复核修正（删除 simulator→native 重标、expected 改为执行器当前值） | 85 failed / 2910 passed / 6 skipped：34 个 R0 沙箱基线（含 1 个 bundle 草稿）、50 个 observe 面 + 1 个 patrol（R6 未开始）；无 R5 自身失败 |
| R5 收尾 + R6 第一切面（投影写入、每批上下文、对账；合成确认文档统一走谓词） | 34 failed / 3065 passed / 6 skipped（observe 50 + patrol 1 + experiments 2 全部消失，失败集与 R0 沙箱基线完全相同） |
| R6 完成（报告当前性、索引 13、下游读取面） | 34 failed / 3072 passed / 6 skipped（失败 node-id 仍是那 34 个 R0 沙箱项：`comm` 双向差集为空） |
| 评审后三个缺口修复（批次记录一一对应、确认异常出口、analysis/brief 的 schema 门） | 34 failed / 3083 passed / 7 skipped（第 7 个 skip 是真实语料索引存在但 0 行；失败集与 R0 基线双向差集为空） |
| 评审第二轮两处旧证据旁路（候选面板正文、hwfeatures 实测门）+ 一条规则收口（§8.3） | 34 failed / 3086 passed / 7 skipped（失败 node-id 与 R0 沙箱基线双向差集为空） |
| 评审第三轮五处（hw_features trace/发布值/参考摘要、hwfeedback roofline、brief trace） | 34 failed / 3091 passed / 7 skipped（失败集仍与 R0 基线双向差集为空） |
| 评审第四轮（hwfeedback 缓存键绕过当前性门） | 34 failed / 3093 passed / 7 skipped（同上；并行全量偶见容器/gRPC 时序失败，单跑通过，见 `R6.md` §10.3） |
| 评审第五轮（`4.0 == 4` 类型旁路、四个读取入口、`hw_feedback_ab`） | 34 failed / 3102 passed / 7 skipped（失败集仍与 R0 基线双向差集为空） |
| 评审第六轮（修 §11.3 引入的 would-be 回归：`from_json` 直接吃 dict） | 34 failed / 3103 passed / 7 skipped（同上） |
| 评审第七轮（`timing_uncertain` 缺失/null 曾可发布） | 34 failed / 3113 passed / 7 skipped（失败集仍与 R0 基线双向差集为空） |
| 评审第八轮（顶层 `cache_hit` 缺失/null 曾可发布） | 34 failed / 3125 passed / 7 skipped（同上） |
| R5/R6 最终验收（逐条硬验收复核 + 「已确认候选不被后来失败候选推翻」用例） | 34 failed / 3126 passed / 7 skipped（失败 node-id 仍与 R0 沙箱基线双向差集为空；多出的 1 个 passed 是新增的确认池用例） |
| R5 评审阻断修复（冻结 expected 一致性 + 漂移校验，`R5.md` §19） | 34 failed / 3129 passed / 7 skipped（同样与 R0 沙箱基线双向差集为空；多出的 3 个 passed 是 §18/§19 新增用例） |
| R5 评审阻断修复第二轮（冻结 expected 必须完整：缺必需身份即 `*_identity_unknown`，`R5.md` §20） | 34 failed / 3129 passed / 7 skipped（失败 node-id 仍与 R0 沙箱基线双向差集为空；同批另一次全量只多一个已登记的 xdist 墙钟抖动，`-n 0` 通过） |
| R5 评审阻断修复第三轮（`plan_type_analysis` 改用共同谓词，删除本地发布判定，`R5.md` §21） | 34 failed / 3132 passed / 7 skipped（失败 node-id 仍与 R0 沙箱基线双向差集为空；多出的 3 个 passed 是 §21 新增用例；NPU/Go 离线集合 232 passed） |
| R5 评审阻断修复第五轮（`R5.md` §23：报告候选/资格绑定、schema-4 读盘闭包、provenance 必填、`result.target_id`、outbox epoch/candidate） | 见 §23.5 的全量结果 |
| R5 评审阻断修复第六轮（`R5.md` §24：retained report schema-4 闭包、历史 epoch 必填、confirm_error 传播） | 35 failed / 3153 passed / 7 skipped（34 个与 R0 基线相同，多出的一个是已登记 xdist 抖动，`-n 0` 通过） |
| R5 评审阻断修复第七轮（`R5.md` §25：retained result 改用 `loads_strict`，登记进严格 reader 表 + AST 检查） | 34 failed / 3157 passed / 7 skipped（失败 node-id 与 R0 沙箱基线双向差集为空） |
| R5 最终验收（`R5.md` §26：`fake_worker_exclusive` 改用 schema-4 helper） | 33 failed / 3158 passed / 7 skipped；当前失败 ⊄ 基线的为空集，基线里另有 14 个已合法修复 |

两次运行失败的 node id 集合**完全相同**（`comm` 双向差集为空）。47 个失败全部先于本计划，
归因见 `../../构建执行与证据语义升级-R1-R6-工作指令.md` §1.1：18 沙箱 `OSError`、5 沙箱
`PermissionError`、13 由工作树里已判丢弃的 `probes.py` 草稿引起（R3 修）、1 跨层 import 的未跟踪
草稿（R3 改写）、8 沙箱 assertion、1 并发安全草稿漏改 `materialize(anchor=)`、1 沙箱 `git apply`。
