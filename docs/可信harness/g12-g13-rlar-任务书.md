# G12 / G13 分阶段实施任务书（2026-11-20）

用户授权：用 rlar 逐个实现，在用户休息期间完成可安全落地的部分。

## 总原则

1. 按顺序执行：G13 只读模型健康度 → G12 经验作用域审计 → 文档与状态同步。
2. 不跑 GPU，不调用付费模型，不运行 `taic run` / `taic resume` / `taic track-explore --config`。
3. 不 push，除非用户另行明确要求。
4. 不自动暂停授权、不改委员会、不改搜索、不做自动跨芯片/跨后端经验晋升。
5. 不把模拟 NPU 当真实硬件；不把历史缺失字段当作 0；不把 G13 原型冒充 P1 正式信任系统。
6. 遵守 `AGENTS.md`、`tests/README.md`、分层 README 和 `tests/shared/test_structure.py`。shared 测试不得 import backend。
7. 不手工限制 pytest `-n`；默认 `-n auto`。每个阶段只跑一次必要的全量，失败按本任务书基线分类，不为变绿删测试或削弱断言。
8. reviewer 只能对代码缺陷、违反本任务书和缺失验收提出 blocking；不得把“全量只能跑一次”变成不可撤销的 done=false。重复同一异议两轮后，记录并结束本轮，报告实际状态。
9. 每轮最多 30 分钟。到时就停止，保留可复核提交和真实报告，不无限循环。
10. 若遇到必须用户决策的边界，停止该项，记录原因，继续做不依赖该项的后续工作。

## 阶段一：G13 只读模型健康度原型

### 目标

在现有观测数据上提供按 provider/model 的只读健康度统计和展示，供人判断模型表现，不影响授权。

### 必做

- 汇总会话数、 evaluated/ok 会话数、未交核比例。未交核口径必须来自现有明确 outcome（至少包括 `agent_no_submission`），不能用候选 status 猜测；没有 outcome 的行显示 unknown。
- 汇总反馈 premise：`downgraded`、`unverified`、`verified`，并计算可判前提中的降级比例；无反馈或不适用时返回 null/unknown，不返回 0。
- 汇总人工反馈状态：adopted/done、rejected/refuted、pending，沿用 `observe/trust.py` 的既有口径，避免两套定义。
- 展示成本、token 等 `observe/query.py::model_stats` 已有信息，可以复用而不是重算。
- never-auto：只统计真实存在的 G1-M 门迹。没有门迹的历史 run 显示 `coverage=unavailable` / null，不得写成 0%。如果无法得到可靠全量覆盖率，本字段第一版可以只显示门迹计数和覆盖率说明，不算比例。
- 提供只读入口，优先在 `observe` 现有 API/页面增加模型健康度；也可提供 `taic query models` 的健康字段，但不要新增重复数据层。
- 页面/输出明确标注：`历史 run 的描述统计；P1 尚未 active；不参与授权`。
- 性能遵守现有面板预算；不要每个模型重复全表扫描或 N+1 文件读取。

### 测试

- 合成索引/会话数据，覆盖两个模型：一个全部交核，一个含 no-submission。
- 无反馈、无门迹、无成本数据必须是 null/unknown，不得伪装为 0。
- 验证 adopted/done、rejected/refuted 口径。
- 验证输出是只读且不改变 trust/run 配置。
- 验证 incomplete/coverage 字段在 UI/JSON 中可见。

### 不做

- 不自动暂停、不调整模型权重、不推荐“永久淘汰模型”。
- 不把 never-auto 缺失补成历史 0。
- 不宣称 G13 正式完成；完成后标为“只读原型已完成，正式 P1 健康度仍等 active”。

## 阶段二：G12 经验作用域审计（只读）

### 目标

实现经验库的离线/只读审计，识别 target、backend、任务、证据和模拟声明的错配，为将来跨芯片晋升提供输入；不执行晋升。

### 必做

- 对 `targets/<id>/experience.md` 的每个 section 输出：标题、target、Confidence、题目、来源、宿主重跑指针、边界是否声明任务/dtype/size、是否含模拟声明、是否含 Metal 专属或 NPU 专属词。
- 校验所有 `runs/` 指针存在，格式沿用 `taic/experience_draft.py::valid_host_pointer`；禁止 `.`、`..`、URL、空白。
- 检查 target 与 `落点` 是否一致；缺 target 落点的旧条目标记 legacy/incomplete，不擅自补 target。
- 识别 scope 错配：正文标某任务/芯片，证据路径明确属于另一任务/target 时给 warning，不能仅凭路径猜不出就判错。
- 识别模拟声明：`simulated`、NPU simulation 等必须标记，不允许把模拟证据列为可晋升真实硬件经验。
- 输出机器可读 JSON 和人类可读摘要；统计 verified/reported、有效/缺失指针、legacy、warning、simulation。
- 只读，不改 `experience.md`，不写晋升结果，不改 prompt 指纹。

### 测试

- 有效芯片层条目。
- 缺指针、指针越界、指针不存在。
- target/落点错配。
- 明确任务错配与无法判定任务的区别。
- Metal 专属/NPU 模拟声明。
- 老条目缺新字段时的 legacy，不误报为伪造。
- 当前真实 experience.md 的审计应稳定运行；不要钉死今天条目数，防止经验更新后无意义漂移。

### 不做

- 不创建后端层/通用层经验库。
- 不自动或半自动跨 target 复制、修改 Confidence、生成 verified。
- 不用 Qingwei 模拟数据满足 G12 触发条件。

## 阶段三：文档同步

- 更新 `docs/代理反馈.md`：G13 为只读原型，G12 为作用域审计，正式触发条件仍保留。
- 在相关实现文档写清命令/API、字段定义、覆盖率和非目标。
- 如果改动了系统提示/任务书/experience.md，才走 prompt epoch；只读观察功能通常不应改 `current_prompt_sha1`。
- 不要改写已关闭纪元、退役包或历史 manifest。

## 全量基线（2026-11-20 HEAD 6164960）

最近一次完整非 GPU 测试为 **7 failed / 5609 passed / 18 skipped**，之后只改经验文档、提示锚点和相关测试。允许环境/并行差异，但必须分类。

已知实质失败：

- `tests/backends/metal/test_track_mt1b_build_seam.py`：缺 staged TileLang frontend。
- `tests/shared/test_acceptance_governance.py::test_status_is_clean_and_d3_is_an_approved_document_update`：G0 worktree_head 未登记，保留给用户决策。
- `tests/shared/test_experience_search_prep.py` 与 `test_experience_overlay_injection.py`：钉住付费实验历史 prompt 边界；G9 后不能沿用，不能伪造刷新。

可能并行敏感：remote_npu_bundle_cleanup、remote_job_worker_bridge、observe_scale 等；先用 `-n 0` 复现分类，不要改阈值。

## 交付

每个阶段独立提交，提交信息写清实现、测试、未验证项。最终保留一份进度报告，列明每个阶段提交、测试数字、未完成事项。不要 push。
