---
name: Feedback docs batch 2 (from real entries)
overview: 用 runs/conv2d-feedback-20261021 的真实反馈条目定型第二批交付：给写核代理的教程（含 system_tutorial.md 快照机制）与代理反馈章程/拼读合册，并入 docs/README.md。数据里暴露的三个代码缺陷先修，否则文档会写成假的：TASK.md 的 eval 命令工作目录错、实测前提被 8 条上限挤掉、56 条条目没有 digest。
todos:
  - id: fix-eval-command
    content: TASK.md 的自测命令用绝对 kernel 路径（--directory 会换 cwd，相对路径必 FileNotFoundError；两个会话各自踩过）
  - id: keep-measured-premises
    content: 被否决的前提里 verified 的按 digest 去重后全留；只有 unverified 才两记录过期（r12.p0 报告 r9 的实测数字被 8 条上限挤掉）
  - id: digest-robustness
    content: 约定层已改（没有内容就不写、整份为空不建文件、无建议不写 kind 行）；解析层仍要规范化后取 digest（反引号/全角引号/`[已实测: …]` 变体）消灭 56 条空 digest，并让存量文件里的纯「无」条目不进 items
  - id: truncation-signal
    content: 16 KB 上限被触碰 10/20；提高上限或在条目里记下被截位置，让审查的人知道漏了什么
  - id: tutorial
    content: docs/给写核代理的教程.md，六节，每节引一条真实反馈；明确 system_tutorial.md 的运行开始快照、每场复制、续跑读快照、版本行
  - id: handbook
    content: docs/代理反馈.md = 章程 + 拼读合册；写清五节、premise 三态、kind 词表、审查四态三态与证据硬约束、追溯链、去重聚合、版本登记表
  - id: seed-labels-in-ui
    content: 看板分清三种种子（详情头加「种子核」与「确认胜者」、候选表加种子核徽章、反馈 meta 写「当时的种子核」、术语面板加三条）— 已完成
  - id: docs-index
    content: docs/README.md 并入两篇的索引条目，并在 docs/架构/搜索.md 与 docs/架构/可观测.md 各加一句指向
  - id: tests
    content: 三个修复各一条离线用例；docs/README.md 链接存在性检查
isProject: false
---

# 第二批：用真实反馈定型教程与章程

## 一、数据（`runs/conv2d-feedback-20261021`，2026-10-21，**运行仍在进行**）

| 项 | 值 |
| --- | --- |
| 状态 | pid 7230 在跑，第 20 轮，$13.51 / $15，无 result.json |
| 会话 / 记录 | `agent_work/` 21 个会话，`feedback.jsonl` **20 条**（seq 1–20 连续；缺的是正在跑的 `r20.p0`） |
| 结局 | `evaluated` 10、`unchanged` 10；委员会 first 11 / second 9；deepseek-flash 11 / kimi-k3 9 |
| 候选 | 20 个：`ok` 9、`unchanged` 10、`duplicate` 1 |
| 搜到的速度 | init 1.179× → r5 1.552× → r15 1.608× → **r19 1.728×**（搜索口径，尚未确认；上一版最好 0.83） |
| 条目总数 | **168**（每记录中位 12 条，最多 17，7 条记录 items 为空但 raw 非空） |
| 小节 | 被否决的前提 72、给下一棒的避坑提示 52、对系统的建议 16、卡住的原因 16、需要帮助 12 |
| premise 三态 | `downgraded` 36、`verified` 22、`unverified` 14 |
| kinds | `none` 10，其余 rule/prompt/tool/decision 组合 10 |
| 截断 | **10/20 条记录的 FEEDBACK 被 16 KB 上限截到**（raw 都在 4 KB 内） |
| digest | 108 个不同值；**56 条 digest 为空**；重复出现的只有 `909aed32…`（6 次，内容是「无。」） |
| 需要帮助 | **11/12 是「无」**（另一条写「没有」），即这一节目前几乎全是噪声 |

16 条「对系统的建议」归纳成 8 类（原文清单见附录）：

1. **自测命令的工作目录错**：`uv run --directory <repo> … eval conv2d kernel.metal` 会把 cwd 换到仓库，相对路径直接 `FileNotFoundError`。r1.p0 与 r5.p0 各自独立报告。
2. **`--timed` 的绝对读数不稳定**：同一份核三行 `self_checks.jsonl` 是 548.85 / 577.14 / 547.68 µs；建议改成配对 A/B + GPU 时间戳（r3.p0、r12.p0、r17.p0、r19.p0 都指向这一件事）。
3. **消融会给出假数**：删掉折叠语句后编译器把上游 MMA 一起 DCE；把 sink 放进永不成立的分支会把整段沉进去（r5.p0、r7.p0、r13.p0、r15.p0）。
4. **时间去向缺自洽检查**：分项相加应当等于总时间（305+163+62+44 vs 483），这条比任何单项微秒数都有信息量（r7.p0）。
5. **交接标签混淆**：「已实测」可能只测了「会不会算错」，没测「误差多大」（r9.p0）。
6. **提示只留最近两轮，丢掉了关键实测**：r12.p0 明确说 r9 那条「V 存 fp16 → scaled error 6.7 倍容差」没进提示，而 fp16 操作数正是当时唯一没算完的账。
7. **本地仪器应当入库**：`bench.py` / `bench2.py`（配对 + GPU 戳）口径与外层 `--timed` 有 1–2% 系统差，每轮手抄（r12.p0、r17.p0）。
8. **上一轮 NOTES 与脚本没进新工作目录**：§5–§8 全丢，agent 要自己跑去兄弟目录翻（r16.p0）。

**结论**：这条通道确实接住了系统盲区——16 条建议里 12 条是系统层面、只有 1 条是「这题我拿不到」。同时它也当场指出三个代码缺陷（第 1、6、以及 56 条空 digest）。所以第二批文档必须先让这三处行为成真，否则教程会教一个不成立的东西。

## 二、先修三处（文档要描述的行为必须真实）

**1. 自测命令的工作目录**（`taic/kernel_agent.py` 的 `_check_command`）
现在发出的是 `uv run --directory {repo} python -m tcenv.cli eval {task} {name}`，`{name}` 是相对的 `kernel.metal`；`--directory` 会改变工作目录，于是两个会话照着抄都拿到 `FileNotFoundError`，各自浪费一步。
改：命令里用**绝对路径**（`brief.kernel_path`），并在 `TASK.md` 加一句「这两条命令里的路径是绝对的，不要改成相对路径」。

**2. 实测前提不该被条数上限挤掉**
`peer_tips_for` 把 `verified` 的「被否决的前提」也塞进普通提示池、只留最近 8 条，于是 r9 的实测数字被挤掉（r12.p0 的报告）。
改：`verified`（含 `downgraded`？见下）按 digest 去重后**全留**，只受总字符数 `TIPS_CHARS` 限制；`unverified` 才按「两记录」过期。这条正是这条通道存在的理由：实测过的否决是最贵的信息。

**3. 空占位与空 digest**（**首要改约定，已在合同里改掉**）
先说结论：`无。`／`（空）` 是**模型真的写出来的**（原文在 `agent_work/<label>/FEEDBACK.md`，代码里没有任何地方生成它），因为合同原文写的是「缺的可以留空」，模型选择显式声明"这一节为空"。这个 run 里 168 条里有 8 条是这种占位（7 条在 `需要帮助`、1 条在 `卡住的原因`），而且 `需要帮助` 12 条**全都没有请求**。

改法分两层：

- **约定（已改，`taic/kernel_agent.py`）**：合同现在写「**没有内容就不要写**：你的输出按 token 计费，某一节没话说就整节省略，不要写「无」「（空）」「none」这类占位符；整份 `FEEDBACK.md` 都没有内容时不要创建这个文件」；`kind:` 那行也只在有系统建议或求助时才写。输出是收费的，占位符是纯成本，而且还会变成看板上一条要人清理的待审查条目。
- **解析（仍要做，为存量文件）**：规范化后若某节只剩否定词（`无 / 没有 / 空 / none / n/a / （空）`），不进 `items`、只留 `raw`；`pending` 计数也就不再包含它。另外 56 条空 digest 仍要修：原文形如 `` `[已实测: bench3.py | “…” | 805–8…`` 的反引号与全角引号让 `_digest` 退化成空串，去重、过期、聚合三件事同时失效。
这些条目的原文形如 `` `[已实测: bench3.py | “组覆盖全部 128 通道…” | 805–8…`` ——反引号与全角引号让 `_digest` 退化成空串，落到 `undigested-{seq}-{index}`，于是**去重、过期、聚合三件事同时失效**（168 条里三分之一）。
改：解析前先规范化（去反引号、全角引号折半角、折叠空白）再算 digest；纯「无」「没有」「none」「n/a」的条目不进 `items`（只留 raw），消掉 `909aed32…`×6 那类噪声。

**4.（同批小改）截断要有信号**：10/20 条被 16 KB 截到，说明 agent 愿意写、上限偏紧。把 `FILE_CAP` 提到 32 KB，或在 `truncated` 为真时把「被截在第几节」记进条目，审查的人才知道漏了什么。

## 三、交付物一：`docs/给写核代理的教程.md`

六节，每节都要能用本 run 的真实条目做例子（写的时候引用 `runs/conv2d-feedback-20261021/feedback.jsonl` 的 seq 与 label）：

1. **你是谁**：一场接力的一棒；两委员会轮流；这场结束对话就没了；模型与题目由配置定。产出三件：`kernel.metal`、`NOTES.md`、`FEEDBACK.md`。
2. **系统全景**：`taic` 排轮次、`tcenv` 编译/对答案/计时、hmz+pi 是你干活的工具、看板是人看的地方。附一张与 plan 同形的图，标出三件产出各去哪。
3. **一轮怎么走**：合同 → 写核 → `--timed` 自测（同一个协议、同一个噪声门槛）→ 外层全量评测 → 换不换种子 → NOTES/FEEDBACK 往下传。**用 r19.p0 那条「跨会话比较」举例**，说明自测只是给自己看的。
4. **什么不能动、什么可以质疑**：合同签名/容差/围栏/预算固定；笔记、文档、`experience.md`、旧提示里的前提可以质疑，办法是探针。**用 r5.p0 的 `experience.md`「32 片段溢出」被过度推广举例**，再用 `probe_fmma.py` 的 15.42 / 15.09 TFLOP/s 说明"实测能翻案"。
5. **已知情况去哪查**：成绩单各类型峰值、`runs/calibration.json`、仓库 `docs/`、旧 run 的 NOTES。写清"先查再问"。
6. **怎样写有用的反馈 / 你的反馈后来会怎样**：五个小节各给一个好例子和坏例子——好例子直接用本 run 的 r7.p0（分项和=总时间）、r13.p0（消融假数）、r16.p0（NOTES 被截断）；坏例子用「需要帮助：无。」（11/12 条这样）。说明人逐条复核、无证据不采纳、采纳后由人改仓库、你不会因此拿到新权限。

**快照机制**（写在教程开头，并在实现里落地）：
- 运行开始时编排器把教程写进 `run_dir/system_tutorial.md`（与 `system_prompt.txt` 并列冻结），带版本行；
- 每场会话把它复制进工作目录，并在合同的系统说明里要求先读；
- **续跑读快照**，不读仓库里已更新的版本；新教程只对新搜索生效；
- 改教程要与改系统的同一个提交里升版本号（版本登记表在章程里）。

## 四、交付物二：`docs/代理反馈.md`（章程 + 拼读合册）

一篇两半，前半是规则、后半是读法。必须写死的（都已在本 run 的数据里出现过）：

- **条目身份**：追溯键 = 运行目录名 + `seq`；每条自带题目/dtype/规模/label/轮次/委员会/provider/模型/effort/结局/exit/种子 label/当时最好成绩。
- **术语表：三种「种子」必须分开写**（本 run 的条目里三种都出现过，混用会直接教错读者）：
  | 中文 | 字段 | 是什么 |
  | --- | --- | --- |
  | 种子核 | `state.best_label` / `result.best_label`、反馈条目的 `seed_label` | 搜索期当前最好的一颗核，每场从它开始改；只有快过噪声门槛的候选才换掉它 |
  | 评测种子 | `run_config.run_seed`、`eval_seed(run_seed, phase, label, source)` | 决定这次评测用哪一批随机输入；确认阶段换一批 |
  | 搜索期种子 / 确认胜者 | `result.search_seed_label` / `result.confirm_winner_label` | 搜索停下时在改的那颗 / 换输入重测后活下来的那颗；可以不是同一颗 |
  术语表还要写一句反例：「噪声内没换种子」说的是**种子核**没换，与评测种子无关。
- **五个小节与 `kind` 词表**：`kind: prompt|skill|tool|rule|decision|none`（可多值）；`无` 不算内容。
- **premise 三态**：`verified`（探针文件 + 命题 + 数字）、`downgraded`（给了探针名但文件不可读/路径逃逸，按未验证算）、`unverified`。**过期规则按 §二.2 的新政策写**：verified 全留、unverified 两记录过期、复述不续命。
- **审查四态/三态**：`pending|adopted|rejected|done` × `unchecked|confirmed|refuted`；`adopted` 必须 `confirmed` + 证据，`done` 必须 `confirmed` + 改动位置；每次审查追加历史，不覆盖旧结论。
- **追溯链双向**：采纳写改动落点（提交/文件）；因反馈而改系统的提交引用追溯键。
- **去重与聚合**：按 digest 合并同题同断言，按题目/按模型定期汇总（本 run 已能看出：two committees 各 9–11 条，建议集中在工具与测量口径）。
- **保留与删除**：三个文件随运行目录归档，不单独删。
- **版本登记表 v1**：`feedback.jsonl` 每行 `format`/`v`、`feedback_review.json` 顶层 `v`、本文件与教程的版本行；同版只许加字段；历史文件永不回写。
- **拼读五步**（后半）：运行结局 → 当场候选 `report.json` + 同目录 `FEEDBACK.md` → 条目来源（模型/轮次/当时种子）→ 复核状态与证据 → 处置与改动落点。给一个真实例子的完整拼读：用本 run 的 r7.p0 那条「分项和=总时间」。

## 五、界面上的种子（已完成，随这次提交）

看板不能只写「种子」两个字，要让人一眼知道是哪一种：

- **详情头部**加一行「种子核（搜索期当前最好的一颗）：`r15.p0`」，并在 `confirm_winner_label` 与它不同时追加「确认胜者（换一批输入重测后活下来的）：`r19.p0`」。
- **候选表**里给 `label == 种子核` 的那一行加徽章「种子核」，与已有的「噪声内没换种子」标记并列。
- **反馈块**的每场 meta 写成「当时的种子核 `r15.p0` · 种子成绩 1.608」，因为反馈条目里的 `seed_label` 指的是种子核，不是随机种子。
- **术语面板**（顶栏的 `TERMS` 搜索）加三条：种子核、评测种子、搜索期种子与确认胜者；每条一句话说清它和另外两条的区别。
- 实现是三处小改：`observe/feedback.py` 导出 `seed_label`、`observe/static/index.html` 的详情头与候选表与 TERMS。没有新增接口、没有改索引。

## 六、文档索引与交叉链接

- `docs/README.md` 加两条（教程、代理反馈），措辞照现有条目「回答什么问题」的风格。
- `docs/架构/搜索.md` 的反馈通道那段补一句指向教程；`docs/架构/可观测.md` 的反馈块补一句指向章程。
- 教程与章程互相链接；章程里引用 plan（`docs/human-in-loop_rsi_feedback_20f84da7.plan.md`）作为设计来源。

## 七、测试与验收

- `tests/test_kernel_agent.py`：`TASK.md` 的两条 eval 命令里 kernel 路径是**绝对**的（回归 r1.p0/r5.p0 的 FileNotFoundError）。
- `tests/test_agent_feedback.py`：verified 前提不被 8 条上限挤掉；`unverified` 仍是两记录过期；反引号/全角引号条目也能拿到稳定 digest；纯「无」不进 items。
- `tests/test_observe.py` 或新增一条：`docs/README.md` 里出现的相对链接都存在（防死链，含这次两篇）。
- 离线门禁：`uv run pytest -q -m "not gpu and not llm"` 全绿；不跑搜索、不花钱、不上 GPU。

## 八、不做的

- 不把「建议」自动变成改动；教程只教怎么提，章程只管怎么审。
- 不在文档里写死会漂的数字（例如「--timed 漂 5%」必须注明样本、机器、日期，或改写成「用配对测量」）。
- 不因为 r12.p0 的抱怨就把提示改成全量历史（那会把 168 条塞进下一场）；只把 **verified 的实测否决**保留下来。
- 不改换种子/预算/停机规则。

## 附录：16 条「对系统的建议」原文索引

| # | label | 摘要 |
| --- | --- | --- |
| 1 | init | 无；指向 experience.md 的 32 片段说法被过度推广 |
| 2 | r1.p0 | `--directory` 换 cwd，命令里的相对 `kernel.metal` 找不到 |
| 3 | r3.p0 | 同一份核三次 `--timed` 548.85/577.14/547.68 µs，绝对读数会漂 |
| 4 | r5.p0 | `maxTotalThreadsPerThreadgroup` 提示把人引向寄存器/占用率，实测（probe_regs.py）方向不对 |
| 5 | r5.p0 | 同 2，命令 cwd 问题（第二次独立报告） |
| 6 | r5.p0 | 消融会被编译器 DCE 顺手带走，建议写进技能页 |
| 7 | r7.p0 | `abl8.py` 的 noyfold/gsink2 给出假数：sink 只活在永不成立的分支里 |
| 8 | r7.p0 | 时间去向应要求分项相加等于总时间 |
| 9 | r9.p0 | 「已实测」标签混淆：测了「会不会错」不等于测了「误差多大」 |
| 10 | r11.p0 | 建议把「矩阵单元 ≈ SIMT ALU 吞吐且严格相加」写进 experience.md（12.62–14.34 vs 13.7 TFLOP/s） |
| 11 | r12.p0 | 避坑提示只留最近两轮，r9 的实测数字没进提示 |
| 12 | r13.p0 | 本地 bench.py 应入库；口径与外层差 1–2% |
| 13 | r15.p0 | 分段消融的成本账会骗人（砍掉折叠后编译器释放 24 个累加器寄存器） |
| 14 | r16.p0 | 上一轮 NOTES 与工具脚本没进新工作目录（§5–§8 全丢） |
| 15 | r17.p0 | bench2.py（配对 A/B + GPU 戳）应进技能；单跑漂移 ±2.5% |
| 16 | r19.p0 | `--timed` 的候选 vs 种子是跨会话比较，建议外层同时跑种子并打印配对差 |
