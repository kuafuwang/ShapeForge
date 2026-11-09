# msl-bundle 接入计划（搜索支持多文件候选 bundle）

状态：**B0–B6 已实施（见文末 §7 实施记录与偏差）**，第 24 版。第 23 版是计划原文，保留不改；本版只在文末追加
实施记录。目标读者是真正要改 `taic/`、`experiments/common.py`、
`taic/kernel_catalog.py` 与 `tcenv/backends/metal/search_candidate.py` 的人。
每个缺口带 `文件:行` 锚点；每阶段带验收节点与负例，并要求验收断言**真实运行对象**而不是中间变量
（第 4 版就栽在这一点：resume 的 track 只改了局部 binding）。

**本版（v23）相对 v22 的改动**

1. **删掉正文里最后一处自相矛盾的措辞**：v22 已在 §2.5/§4/B3 定案"不新增模块级常量、替换文本是
   `_rules_for`/`system_prompt_v2` 函数体内的局部量"，但 §B3 的"实现方式"段仍写着
   "`HARNESS_RULES` 拆成可寻址的片段（四条冲突句各是一个具名常量）"。改成唯一方案：
   **`HARNESS_RULES` 整体原文保留（只做清零句纠错）**，`_rules_for` 在 bundle 分支对渲染文本执行
   "旧句 → 函数体内局部新句"的替换——是运行期字符串操作，不是拆分常量。
2. **受信面验收的编号与计数改正**：原来有两个 "3."，且结尾写"这三步"却实际有四步。现在 1/2/3/4 顺序
   无重复，"这四步都在同一个测试里"，并把"标签拼错 → 显式报错"独立成第 4 步（不再塞在第 2 步末尾）。

**v22 相对 v21 的改动（全部经代码核对）**

1. **新增提示文本必须落在已哈希的源码里**：`trust.py:154-172` 只哈希 `_PROMPT_FUNCTIONS`/
   `_PROMPT_CONSTANTS` 里登记的符号。v21 一边要求把 `HARNESS_RULES` 拆成"具名片段"并在 bundle 渲染时
   替换，一边只登记既有符号——如果这些替换句被放进**未登记**的模块级常量，改它们就不会改变
   `current_prompt_sha1()`，旧授权会静默继续有效。定案：**四条替换句与 geometry/workspace 指引全部作为
   `_rules_for` 函数体内的具名局部量**，bundle 段全部写在 `system_prompt_v2` 函数体内（两者都已/将登记，
   `inspect.getsource` 会把函数体一起哈希）；**不允许**提到模块级新常量，除非同时加进 `_PROMPT_CONSTANTS`。
   §2.5 白名单与 §4 指纹清单同步。
2. **B3 判别力验收覆盖到新片段**：新增两条可执行判定——(a) 把 `prompt_fingerprint_parts()` 的字节拼成
   blob，断言**每一条**新增/修改的模型可见文本都能在里面找到（找不到说明它跑到未登记常量里了）；
   (b) 对 `_rules_for` 与 `system_prompt_v2` **各做一次** `inspect.getsource` monkeypatch 判别力断言
   （不只测 `signature_stub`）；若实现选择把片段提到模块级，则每个常量各自覆盖一次并先登记。

**v21 相对 v20 的改动（全部经代码核对）**

1. **受信面"判别力"验收改成可执行机制**：`_symbol_bytes`（`trust.py:154-172`）用
   `inspect.getsource(value)` 取文本，所以"monkeypatch 改源码/替换函数对象"并不会改变磁盘文件、也证明
   不了指纹会变。改为 monkeypatch `inspect.getsource`：仅当对象是
   `TaskBinding.signature_stub` 时返回原文改一个字符、其余委托真实实现 → 断言
   `current_prompt_sha1()` 变化；`monkeypatch.undo()` 后回到原值；再只改无关符号
   （`task_binding.bind_task`）的返回文本 → 指纹**不变**；标签拼错 → 显式报错。
2. **B3 验收⑦ 真正断言第二个调用点**：改成从**真实 `Orchestrator._session_brief(...)`**
   （`orchestrator.py:3178`，内部 `:3234` 用 `self.binding.signature_stub()` 填 `abi_stub`）取 brief，
   断言 `brief.track_id` 与 `brief.abi_stub`，再把**这个真实 brief** 交给 `render_contract`；
   手工 `dataclasses.replace` 只保留为纯 renderer 单元测试。single 对照也要**重新生成** brief/`abi_stub`，
   不能只改 `binding.track_id` 而复用旧 bundle stub。
3. **未知 track id 的闭集补齐**：`TrackRegistry.get`（`tcenv/track/registry.py:229-246`）对未知 id /
   版本不符抛的是 `UnsupportedContractCapability`（`contract_errors.py:29`，`TaskContractError` 子类），
   **不是** `KeyError`；§2.6 判定表新增该行并明确"窄捕获 → `(未确认, track_mismatch)`、不抛"，
   B4b 增加 `track_id="no-such-track"` 的读盘负例（`conclusion_of_run` 与 `publication_reason` 都不抛，
   resume 准备阶段 → `ResumeRefused`）。
4. **计数纠正**：B6 实际列了五份文档（`v2协议参考.md`、`搜索-v2接线.md`、本文、`experience.md`、
   `数据层.md`），§5 的"四份文档"改成五份并列出名字。

**v20 相对 v19 的改动（全部经代码核对）**

1. **`signature_stub` 不再新增参数**：`TaskBinding` 本身已有 `track_id`（`task_binding.py:38`），
   `signature_stub()`（`:123`）是无额外参数的方法，`presentation()` 已经按 `self.track_id` 选 track
   （`:110`）；两个现有调用点（`prompts.py:176`、`orchestrator.py:3234`）都不传参。改为**方法内读
   `self.track_id`、签名不变**，避免多一条可能与 binding 分裂的入参；B3 验收改为断言**两个真实调用点**
   的输出 + `inspect.signature` 只有一个 `self` + 改 `binding.track_id` 后两处一起回到 single 形状。
2. **`baselines.json` 的 schema 锚点改正**：实际路径是
   `tcenv/taskcontract/schema/baselines.v1.schema.json`（由 `baselines.py:32` 的
   `BASELINES_SCHEMA_PATH` 引用），v19 写的 `schema/baselines.v1.schema.json` 会让实施者找不到闭集。
3. **B6 补原因码文档同步**：B4b 新增 `track_mismatch`，B6 必须同步
   `docs/架构/v2协议参考.md:480-499` 的 reason 闭集表与 `docs/架构/数据层.md:519` 的
   `publication_reason` 枚举（按 `taic/policy.py` 的代码顺序），并加"两份文档与代码闭集逐项相等"的断言。

**v19 相对 v18 的改动（全部经代码核对）**

1. **交付拒绝的落盘方案定案**：`Candidate` 没有 `kernel_path` 字段（`orchestrator.py:227-270`），
   `_save`（`:1469-1482`）无条件写 `candidates/NNN_<label>/<kernel_name>`；而
   `resume._read_candidates`（`resume.py:553-590`）会跳过没有内核文件的目录、`resume.py:867-870`
   还会以 `no candidates/ were stored` 拒绝续跑。所以删掉"不写 kernel 文件 / `kernel_path` 未设"，
   改成**保留 `_save` + 明确空串占位文件**；并补一条硬要求：**空 source 永不进 cache**（新路径不写、
   restore 路径 `orchestrator.py:1405-1413` 用 `_cache_this("")` 今天会返回 `True`，必须加守卫）。
   `_one_session_attempt` 里明确 `cand = self._record_delivery_rejection(...)`，
   否则 `finally` 的 `_record_feedback(..., cand, ...)` 拿不到候选。B2 加"rejection 落盘后 resume"
   验收。
2. **`SessionBrief.track_id` 的字段新增移到 B0**：`collect_session(brief, ...)` 收不到 binding
   （`kernel_agent.py:1313`），`SessionBrief` 今天也没有 `track_id`，所以"替换 binding"无法驱动 B0 的
   collect 分支。改为：**字段在 B0 新增**（默认 `"msl-single"`），B0 测试用
   `dataclasses.replace(brief, track_id="msl-bundle")`；**B2 只负责填值**（由真实 `Orchestrator`）。
3. **B2 不再验收 B3 的 prompt 内容**：`orch.system` 含 bundle 段、冲突句消失等断言全部留在 B3；
   B2 只断言三个对象（`orch.binding`/`orch.evaluator.binding`/`SessionBrief.track_id`）以及
   "渲染函数确实收到了这个 track"（spy 入参，不断言文本）。
4. **发布 legacy 规则改成判定表**：原来"缺 `track` 或畸形 → `track_mismatch`"与紧随其后的 legacy
   例外自相矛盾。现在按行取唯一结论：缺 `track` + 非 envelope best → legacy；缺 `track` + envelope best
   → `track_mismatch`；存在但畸形/自相矛盾/与 best 形态不符 → `track_mismatch`；兼容承诺只覆盖
   "缺 track 的旧单文件记录"与"well-formed 当前 single-track 记录"两行。

**v18 相对 v17 的改动（全部经代码核对）**

1. **费用完整性纠正**：v17 让 `collect_session` 硬编码 `cost_incomplete=False`，而它今天的写法是
   `not invocation.saw_tokens`（`kernel_agent.py:1377`），且 `_one_session_attempt` 先看这个字段再入账
   （`orchestrator.py:2677-2688`）——硬编码会把未知费用按 0 入账，正好造成要避免的丢失。改为原样保留
   `cost_usd`/`spent`/`not invocation.saw_tokens`；验收拆成"已知费用 → 入账 + rejected + 不 StopRun"
   与"未知费用 → 仍 `StopRun("cost_incomplete")`、不得伪造零费用"两个用例。
2. **状态语义纠正**：v17 硬编码 `status="ok"` 会把 timeout/process failed 改写成正常完成。改为保留
   `invocation.status`/`error` 与既有 diagnostics；`candidate_delivery` **只在 `status == "ok"` 且写了
   提交文件但字节非法**时设置，不吞掉 timeout/process/turn failure。
3. **把"同一个 rejected 构造器"落实成可照做的对象**：新增 `tcenv/evaluate.py::rejected_report(...)`
   纯函数（`V2LocalEvaluator._rejected` 改调它），以及 `Orchestrator._record_delivery_rejection(...)`
   的无 GPU 落盘路径——写清 `Candidate.source=""`、**空串占位文件（`_save` 照常写）**、
   `trace`/`origin`/`cost_usd`/`_failure_feedback` 分别怎么记，明确不进 `cache`、不动
   `best`/`hangs`/`crashes`、不建 outbox；白名单补 `tcenv/evaluate.py` 与 `taic/session_supervisor.py`。
4. **依赖顺序修正**：B0 的 G21 验收原本要求贯穿真实 `_one_session_attempt`，而那需要
   `SessionBrief.track_id`（B2）——循环依赖。改为：**B0 只做单元级**（直接 `bind_task` +
   `replace(track_id=)` 构造 bundle binding/brief），**真实 orchestrator 端到端移到 B2**。

**v17 相对 v16 的改动（全部经代码核对）**

1. **G21 的闭合机制改正**：`collect_session` 跑在会话执行阶段（`kernel_agent.py:1428`、
   `session_runtime.py:484`），在它里面抛 `CandidateDeliveryError` 会被 `_one_session_attempt` 的
   `except Exception`（`orchestrator.py:2661-2676`）记成 `agent_failed` + `cost_usd=None` +
   `cost_incomplete=True`——既到不了请求构造处的 rejected 转换，又丢掉已计费成本（`kernel_agent.py:1314-1315`
   已明说）。改为：**`collect_session` 不抛**，先算完费用，再返回 `source=None` + 已计费 +
   `diagnostics={"failure_class": "candidate_delivery", ...}` 的 `AgentResult`；
   `classify_result` 新增 `candidate_delivery → delivery_rejected`（在 `agent_failed` 兜底之前）；
   `_one_session_attempt` 在 `ledger.charge(...)` 之后用 §2.3 同一个 rejected 构造器生成
   `stage="delivery"` 的拒绝（不调 `_eval`、不占 GPU、不写 outbox）。G21、§2.5、B0、B2 同步；
   负例改为**贯穿 `_one_session_attempt`**，断言 rejected + 费用入账 + 不是 `agent_failed`。
2. **受信面措辞纠正**：`system_prompt_v2` 与 `_check_command` **不在** `_PROMPT_FUNCTIONS`
   （`trust.py:77-88`）里，所以它们和 `signature_stub` 一样都是**新增登记项**；正文里"本来就在指纹里"
   的说法已改掉。

**v16 相对 v15 的改动（全部经代码核对）**

1. **`signature_stub` 的受信面登记改成可执行机制**：它是 `TaskBinding` 的**方法**
   （`task_binding.py:123`），模块里没有同名符号；`trust.py:155-172` 的 `_symbol_bytes` 直接
   `getattr(module, symbol)`，照 v15 的写法登记会让 `prompt_fingerprint_parts()`/`current_prompt_sha1()`
   抛 `AttributeError`。改为：`_symbol_bytes` 支持**点分属性路径**、只哈希
   `TaskBinding.signature_stub` 的方法源码、解析失败显式报错；`taic/trust.py` 的这项 helper 改动进
   §2.5 白名单；验收先调真实 `prompt_fingerprint_parts()`/`current_prompt_sha1()` 证明不崩，
   再证明"只改该方法源码 → 指纹变、改无关函数 → 指纹不变"。
2. **B4 的确认调用次数写错**：`confirm_subject` 是 **1 次 search + 最多 3 次 confirmation**
   （`portfolio_confirmation.py:260-296` 的 `for attempt in range(1, BENCH_ATTEMPTS+1)`，
   `BENCH_ATTEMPTS == 3`，`tcenv/bench.py:157`），不是"各调一次/两次返回"。改为捕获 1 次 search +
   **全部** confirmation runs，每个 run 断言自身 `dispatch/_stages == expected_*`，每个 confirmation
   再与 search 比序列与 `variant_id`，并用 sidecar 的 `confirmation_attempts` 核对条数。
3. **补上真实交卷链路的字节破坏**：`collect_session`（`kernel_agent.py:1313-1316`）用
   `read_text(errors="replace")` + `kernel_text()` 抽块，会做 CRLF→LF 归一与 U+FFFD 替换，导致一份
   device entry 含 CRLF 的合法 envelope 在 decode 时炸成 `a taic-file body is not followed by its end
   marker`。新增缺口 G21；§2.5 白名单加 `collect_session` 的 bundle 分支（`read_bytes()` + 不归一解码 +
   跳过 `kernel_text()`，非法 UTF-8 → typed rejected，single 不变）；B0 验收加"CRLF envelope 从 collect
   到 decode/preflight 的字节保真正例 + 非法 UTF-8 负例 + single 回归"。

**v15 相对 v14 的改动（全部经代码核对）**

1. **TASK.md 自己的 launch 旧指引补进 B3**：`kernel_agent.py:672-674` 的 `render_contract` v2 分支写着
   "外层不会替你选 grid/threads。写成 `launch NAME grid=... threads=...`"，这与 bundle 的"dispatch 以
   geometry 为准"直接冲突，而且它就在**同一个 TASK.md** 里。冲突表新增该行、§2.5 白名单新增
   `render_contract` 的 bundle 分支；验收⑧ 改为**在 `render_contract` 的实际返回值上断言这段中文旧句
   不存在**（只查英文句会让这个渲染面空过），并断言替换句存在、single 仍保留该中文句。
2. **B4 的"dispatch stage 序列一致"补上可读对象**：`confirm_subject` 的 sidecar
   （`portfolio_confirmation.py:267-296`）只存两份 report/identity/seed/`confirmation_attempts`/
   `credits_variant`，不存任何 `PortfolioRunResult.evaluation`，schema-4 report 也没有 `dispatch_stages`
   → 验收改为**替换 `portfolio_confirmation._evaluate`**（`:122`）捕获 **1 次 search + 全部 confirmation**
   （`v15` 曾误写成"两次"；真实是 `:260-296` 的 `for attempt in range(1, BENCH_ATTEMPTS+1)`，
   `BENCH_ATTEMPTS == 3`，见 `tcenv/bench.py:157`），每个 run 断言自身 `dispatch/_stages` 与 `expected_*`
   一致，每个 confirmation 再与 search 比序列与 `variant_id`，并用 sidecar 的 `confirmation_attempts`
   核对捕获条数；或另做一次明确的新鲜 replay。明确禁止从 sidecar/report 读这些字段。

**v14 相对 v13 的改动（全部经代码核对）**

1. **B3 的"不改常量、只追加覆盖"方案自相矛盾，已换成唯一可实现的方案**：`system_prompt_v2` 开头就是
   `_rules_for(backend)`（`prompts.py:157`），Metal 返回完整 `HARNESS_RULES`，所以"只追加 ignore earlier"
   之后旧句**必然仍在最终 prompt**，而验收⑧又禁止它们出现——没有任何实现能同时满足。改为：
   `_rules_for(backend, delivery)` 在 `candidate_bundle` 时**删除并替换**四条冲突句（launch 决定
   grid/threads ×2、可选动态 threadgroup bytes、恰好一个代码块），`single_source` 输出今天的正文
   （具体落地见 v22：替换文本是 `_rules_for` 函数体内的具名局部量，不新增模块级常量）。
2. **清零句的性质定案**：它属于**允许修改的常量内容**（与 track 无关，v1 与 v2 两个引擎都错），
   所以 `HARNESS_RULES` 会被改；single 的"逐字节不变"承诺**收窄**为
   「除这一句纠错外与今天逐字节相等」，验收⑧给出可执行判定
   （`single_text.replace(NEW, OLD) == TODAY_TEXT`）。
3. **验收与指纹说明同步**：验收⑧改为分别在 `system_prompt_v2` 与 `render_contract` 两个实际返回值上断言
   （bundle：四条冲突句逐条不存在 + 四条替换句在；single：还原纠错句后与今天逐字节相等）；
   §4 的指纹条目改为列出 `HARNESS_RULES`/`_rules_for`/`system_prompt_v2`/`render_contract`/
   `signature_stub`/`_check_command`/SKILL.md 与"所有 track 一起重签"。

**v13 相对 v12 的改动（全部经代码核对）**

1. **`baselines.json` 形状写错、装载入口不存在**：`baselines.v1.schema.json` 是
   `additionalProperties: false` 且没有 `schema_id`（用计划里的文档实际调 `validate` 会得到
   `SchemaFailure: $: unknown field 'schema_id'`）→ 删掉该键，只留 `schema_version + baselines`；
   仓库里没有 `load_baseline_package`，真实入口是 `load_baselines(task_dir, op_name=...)`
   （`baselines.py:145`）→ 测试改成真的调它（这样 manifest 错误会被测试抓到）。
2. **zero-count 正例与运行语义矛盾**：`v2_runtime.py:199-201` 无论 count 是否为零都拒绝 `stride < 1`
   → 正例限定为 `count=0, stride>=1`，保留 `stride=0` 负例；并明确 `offset >= 0` 是**本期新加**的限制
   （今天放行 `offset=-1, count=2, stride=2`），所以不能声称"逐条对齐"，且白名单必须显式允许把
   `HostTilingPlan.execute()` 换成共用 helper。
3. **真正生成"签名骨架"的函数被漏掉**：`TaskBinding.signature_stub()`（`task_binding.py:123-149`）
   显式跳过 workspace，同时进系统提示（`prompts.py:176`）与 TASK.md（`orchestrator.py:3234`）→
   纳入 B3：bundle 分支保留 workspace 参数与可写性、single 逐字节不变；`taic/task_binding.py` 进白名单；
   该**方法**登记进受信面（v16 补上点分路径机制，见下）；验收断言两处骨架都含 workspace 参数。
4. **旧提示里的矛盾没被覆盖**：`HARNESS_RULES` 仍说 launch 块决定各 stage 的 grid/threads、允许
   optional dynamic threadgroup bytes、要求恰好一个 Metal 代码块（`prompts.py:57-74`），
   `system_prompt_v2:167-171` 也仍把 grid/threads 全归给 launch 块 → 给出逐句表格与**渲染期替换**方案
   （v14 已把这里从"追加覆盖"改成"按 delivery 渲染时删除/替换"，见 v14 第 1 条）。

**v12 相对 v11 的改动（保留）**

1. **B5-b 的 baseline 落地闭合**：补 `BASELINES = {"torch_mps": run}` 与闭合的 `baselines.json`
   （`baselines.py:194-207` 要求 `BASELINES` 是 name→callable 且键集合与 manifest **完全一致**），
   题目测试覆盖这条闭合关系与 `SEG/BASE/STEP` 常量一致。
2. **承重验收改成真实存在的字段**：`report["dispatch"]` 不存在 → 改 `run.evaluation["dispatch"]` /
   `["expected_dispatch"]` / `["dispatch_stages"]`（`portfolio_runner.py:1066-1077`）；
   `workspace_fills` 只在 `run.result.execution`（`track_adapter.py:456-459`）；
   `median_us` 取 `report["benchmark"]["candidate"]["median_us"]`；
   `base_runners` 是局部变量 → timed 自测改查 `report["benchmark"]["baselines"]` 与 `report["baseline_us"]`；
   并把 **eval-v2 timed 自测** 与 **真实搜索的 `confirmed_values`/`conclusion_of_run`** 拆成两段。
3. **Host Tiling span 公式对齐执行语义**：`count == 0 → span = 0`（朴素公式会把空写误判成 -1），
   补 `offset >= 0` 的负首地址校验，抽成预检与 `execute()` **共用的纯 helper**，并加 zero-count 正例。
4. **深嵌套 JSON 的三条源头**：envelope → `EnvelopeError`、geometry（`delivery.py`/共享 `schema.py`）→
   `GeometryError`、Host Tiling（`v2_runtime.py:132-136`）→ `TaskContractError`；白名单与 B0 负例同步。
5. **动态 threadgroup 题面统一**：本期**不支持** `[[threadgroup(k)]]` 参数、静态
   `threadgroup T name[N]` 可用、`threadgroup_bytes` 写 0、非零数字不算证据；B3 的 prompt 验收同步。
6. **两处残留清理**：§6 里残留的 `contract.known_good_sha256` 改成
   `fingerprint_document()["known_good_sha256"]`；`cmd_eval_v2` 从 `taic/cli.py` 白名单项移到
   `tcenv/cli.py`（它实际在 `tcenv/cli.py:87`）。

**v11 相对 v10 的改动（保留）**

1. **B5-b 的 oracle 与舍入判别都写错了**：真实路径里 `x` 已经是 fp32（`round_to` 先舍入到声明 dtype
   再加宽，`tensors.py:25-36`；`v2_runtime.py:261-266` 在调 reference 前就这么做），所以
   `.to(x.dtype)` 不会输出 T；改为**独立 fp64 oracle**（ramp 与乘积都在 fp64）。判别测试改成
   **确定性输入上的 bitwise 比较**（good/bad 同一次最终 fp16 cast）；删掉「测试固定了
   `single_round_at_output`」的暗示（`rounding` 不进 `SemanticInstanceV2`，判定只有
   `tolerances + scaled_error`）。
2. **baseline 合同与性能验收写实**：baseline 只收到 `contract.inputs`（`baselines.py:89-100`），
   workspace 又被 `track_adapter.py:82-106` 排除 → 改成一参数 runner 自己构造 fp32 ramp，并加
   **需 GPU 的 baseline 数值测试**；`--quick` 在 bench 前返回（`evaluate.py:623-632`）→ B5-b 增加
   full/timed 证据与五项断言（baseline 名字存在、时间有限为正、speedup 有限、`confirmed_values` 非空、
   published speedup 非 `None`）；证据措辞改成 **steady-state table-reuse**。
3. **承重证据不再诱导挑算法，也不假装有落盘对象**：负例固定为 `start=0, step=0`（全零表 → 固定 seed
   必 incorrect），避免"改错一点仍在容差内"；不再要求搜索 winner 必须读表；`execution` 不进 schema-4
   document（`runner.py:404-415`）、`V2LocalEvaluator` 只留 report（`v2_evaluation.py:106-124`），
   所以改成 winner/known-good envelope 的 **fresh GPU replay**，从 `run.evaluation` 断言。
4. **B5-a 的动态 threadgroup 正例是空证据**：真实 `[[threadgroup(n)]]` 参数被
   `stage_binding.py:200-204` 拒，PyObjC 只在有 binding 时分配（`runtime.py:638-640`），原生路径见
   `native_batch.py:180-185,263-275` → 本期明确**不支持**，改成负例，不把非零数字称作动态内存正例。
5. **B3 补 workspace 的真实 MSL ABI**：`prompts.py:170` 的 "workspace tensors are not function
   parameters" 在 bundle 里是错的；给出按名字绑定、可写性必须匹配 `access`（`stage_binding.py:181-197,
   248-255`）、完整签名/launch 块/geometry 对照，并说明 bf16 需要 `bfloat` 变体。
6. **身份与异常闭集继续补洞**：`retest_command`（不是 `record["command"]`）；resume 与发布都要求
   `delivery == registry.get(track_id, version=track_version).delivery`；`result.track` 缺失/畸形
   fail-closed 且定义 legacy 规则；seed 判定复用 preflight 而不是 `startswith`。
7. **预检补 span 校验与四类漏网异常**：Host Tiling 的 count/stride/offset span 提前到预检；
   `RecursionError`、`json.loads` 超长整数、launch 超长十进制、plain source 的 `UnicodeEncodeError`
   在源头规范化；`cmd_eval_v2` 的 `read_text()` 改成窄边界内 `read_bytes()+decode`。
8. **小项**：`fingerprint_document()["known_good_sha256"]` + `read_bytes()` 逐字节比较；
   §5 依赖补 B5-a←B4、B4b←B2；`min_bytes` 排除 `scale_tab` 的理由改成"算法可自行重算"。

**v10 相对 v9 的改动（保留）**

1. **`reference.py` 示例缺 `import torch`**：`load_reference`（`taskcontract/reference.py:471-501`）把模块当
   普通模块执行、不注入任何名字，照抄会 `NameError`。已补 `import torch`。
2. **示例多舍入了一次**：`tab[idx].to(x.dtype) * x.float()` 先把 scale 降到 fp16，与
   `acc_dtype=fp32` / `single_round_at_output` 及 fp32 workspace 的设计矛盾。改为
   `(tab[idx] * x.to(torch.float32)).to(x.dtype)`；并把 `BASE` 定为 2049（>2048，fp16 间隔 2，奇数项不可
   精确表示），使「预舍入 scale」与「仅输出舍入」在本题实例上可判别；baseline 与 `known_good.metal`
   要求同一语义；新题测试加舍入语义判别用例。
3. **`_failure_record` 只进了 §2.1、没进 G15/B2**（v9 的总结说进了，正文没落实）：G15 行、B2 改动句、
   B2 验收现在都覆盖第二条命令（字段名见 v11 更正：是 `retest_command`，v10 误写成 `record["command"]`）。

**v9 相对 v8 的改动（保留）**

1. **B5-b 合同改成 oracle 能实现的形状**：reference 只有 `reference(inputs, attrs)`
   （`taskcontract/reference.py:5,133-159`），拿不到 `instance.symbols`、也拿不到 workspace，所以
   `SEG/BASE/STEP` 由 symbol 改为**固定整数 attr**（v2 attr 只来自 `default`，天然不随 size 变），
   只留 `N/NS` 为 symbol、`spec_params=["N","NS"]`；并补齐 `numerics`/`work_accounting`/`discovery`、
   `formula`/`type_vars`/`cases`、reference 与 baselines 的可直接落地内容。
2. **track 的命令传播补第二处**：`_failure_record`（`orchestrator.py:2069`）拼的 differential-recheck
   人工复测命令也要带 `--track msl-bundle`；进 G15 与 B2 的改动面和验收。
3. **计数纠正**：B5-b 验收写「上面三条承重证据」实际是四条。

**v8 相对 v7 的改动（保留）**

1. **改动面白名单补 `taic/cli.py`**，并把 TOML 的 `track` 读取方式钉死（只用现有 `load_raw_config()`
   的原始 section，不改 `taic/llm.py`、不进 `SearchConfig`）。
2. **B3 补 geometry 闭集合同**：顶层三键、stage 必填/可选字段、多阶段有序、`threadgroup_bytes` 的语义，
   以及「dispatch 以 geometry 为准、`taic-launch` 只负责 math 与 workspace 交叉检查」的真实分工。
3. **track 接线补两个入口**：只有 v2 才重建带 track 的 binding（v1 + `--track msl-bundle` → 启动拒绝）；
   注入的 `V2LocalEvaluator` 必须核 binding；`tcenv eval-v2` 加 `--track`、重建 binding 并透传。
4. **请求构造前的超限路径入闭集**：`EvaluationRequest` 的 source 超限（`evaluation.py:168-170`）改为
   `CandidateDeliveryError`，并加两条端到端负例。
5. **发布读盘的异常闭集写全**：`raw.decode("utf-8")`/envelope decode/entry 构造的异常统一 → `sha_mismatch`，
   并加「envelope 形状但非法 UTF-8」负例。

**v7 相对 v6 的改动（保留）**

1. **pack 闭包不递归**：`V2_STANDARD_FILES`（`loader.py:48-56`）与 `STANDARD_TASK_FILES`（`pack.py:37-45`）
   都是**固定顶层清单**，`task_pack_files()`（`pack.py:246-258`）只收其中存在的顶层文件；v6 声称
   `selfcheck/` 会随 `known_good.metal` 一同进 pack 是错的。改为「自测素材不进 pack」+ 明确的证据绑定（§B5-b）。
2. **B5-b 合同的符号没定义**：`base/step` 在 v6 里既不是 symbol 也不是 attr，`HostTilingPlan` 的名称求值
   （`v2_runtime.py:69-80,169-174`）会得到未定义名称。改为把常量声明成 **symbols**（`BASE/STEP`）并把
   `x/y` 的 role/space/dtype/shape/access 写全（§B5-b）。
3. **track 接线漏了两个局部消费者**：`orchestrator.py:693` 的 `system_prompt_v2(binding)` 与 `:743` 的
   `V2LocalEvaluator(binding)` 用的是原局部变量。改为「先重建唯一局部 `binding`，再赋给 `self.binding`」，
   并在 B2/B3 验收里断言实际 system prompt 与 evaluator 的 binding（§2.1/B2）。

**v6 相对 v5 的改动（保留）**

1. **envelope 没有 `track_version`**：v5 的 §2.6/B4b 要求 decode 出它，但 `envelope.py:101-109,265-271`
   只传 `track_id`，`plane.py:368-374` 的 `CandidateArtifact` 也没有该字段；改为 envelope 只校 `track_id`，
   `track_version`/`delivery` 由 `result["track"]` × registry 交叉校验（§2.6/B4b）。
2. **`@catalog` 修复漏了 `None` 路径运算**：门放宽后 `self.from_run` 仍是 `None`，`orchestrator.py:2368,2373`
   的 `self.from_run / self.best_name` 会 `TypeError`；B4b 要求先引入统一的 prior 来源显示名/路径。
3. **删掉「不读表就判不合格」**：`workspace_probe`（`v2_runtime.py:945`）只看最终字节，无法证明 device
   是否读过；且 device 重算公式是合法算法。承重证据只保留「指定正例必须读表」+「同 device/破坏 host plan
   后 correctness 必错」，不新增任务级拒绝语义。
4. **`known_good.metal` 的形态定清**：它是 pack 文件（`taskcontract/loader.py:55,64,418-432`），
   保持**原始 MSL**；bundle 自测另交一份由打包器生成的 envelope + 命令，host 素材作测试夹具并在证据里记 sha
   （§B5-b；pack 闭包是固定顶层清单，不递归）。
5. **resume 的旧配置兼容**：`artifacts.py:386-392` 明确放行 schema<3；缺失 `track` 的 v2 旧运行按
   `msl-single` 恢复，v1 仍为 `track=None`，只有**显式存在但非法/**版本不符才 `ResumeRefused`。

**v5 已修、仍保留的（不变）**：

1. resume 的 track 传播改成「给 `Orchestrator` 加参数」，并定义新运行优先级与省略哨兵（§2.1/B2）。
2. `_prior_kernel_text`/`@catalog` 的现状判断写反了：envelope 的 device body 是**明文**，
   `_KERNEL_DECL` 会匹配，所以真风险是「不解码就接受」；且 `@catalog` 的 seed 路径整条是死的（§1.7/B4b）。
3. selfcheck 的验收与 typed preflight 自相矛盾：统一成 `ok=true` + `report.status="rejected"`（§2.4）。
4. 异常闭集补 `UnicodeDecodeError`、`LaunchRejected`、`TaskContractError`、`parse_launch_spec` 的 `None`，
   并在源头把两类泄漏规范化（§2.3/§2.5）。
5. B4b 纳入 catalog 持久化、`read_bytes()` 字节身份、从 `result.track` 反查 delivery 的绑定（§1.3/§2.6）。
6. 受信面补 `system_prompt_v2` 与 `_check_command`；remote workspace 的 bundle 走启动期拒绝（§B3）。
7. B5 的验收改成可执行：新题自己的 discovery/instantiation/reference 测试、`eval-v2` 的 MSL 与命令、
   Host Tiling 承重的负例；`workspace_bytes` 断言 ABI span；打包器负例修正、`1,000,000` 字节上限。

---

## 0. 结论先行

**主线不变**：评测/几何/lifecycle/确认这四层已经支持 bundle 并有真机用例；本次只在搜索侧接交付、
包装、预检、题面与自测，不新设计评测或确认机制，不引入 KDA/CAKE2 的 host 派生几何模式。

**七个前提必须先补，否则「只接搜索」跑不出第一颗真正发布成功的 bundle 候选：**

1. **题**：bundle 可用性由 OpSpec 决定——投影要求
   `_require_bundle_shape(..., workspace_declared=authority.workspace_declared)`
   （`tcenv/portfolio/candidate.py:1290-1311`），即权威 OpSpec 必须声明 `space: workspace`；
   而 `tasks/*/opspec.json` 现在**一个 workspace 张量都没有**（含 `flash_attention`）。→ B5 必须造题，
   且只能**新增题**（改既有题会改 `opspec/pack` digest，动到已关闭证据）。
2. **发布身份**：bundle 下 `sha256(提交文本)` 与 `candidate_sha256`（manifest）必然不同，而唯一发布谓词
   要求 sidecar 的候选摘要同时等于 `result.confirmation_source_sha256`、`best.metal` 与胜者
   `kernel.metal` 的哈希（`experiments/common.py:730-760`）。不改就永远 `sha_mismatch`（§1.3/§2.6/B4b）。
3. **CPU 预检漏两项**：跨 device 的 launch 唯一性只在合并源码后检查；`workspace_size_problems`
   只在 runtime bind 调用（§1.4）。
4. **agent 闭环**：envelope 解析链路已通，但无工具生成合法 envelope，`tcenv eval-v2` 也只评单文件，
   且 `_check_command` 不注入 track（§2.1/B0/B2）。
5. **一处崩溃缺陷**：多 device 源拼接用 `sorted(CandidateEntry)` → `TypeError`（§1.5）。
6. **非法交付的边界**：`EvaluationRequest.__post_init__` 先于评测器；且 `decode_envelope` 会抛
   `EnvelopeError`/裸 `TaskContractError`/`UnicodeDecodeError` 三类（§1.6）。
7. **上一版内核的 seed 路径**：`@catalog` 今天完全取不到 seed，且 dup 豁免只为 `--from` 设置（§1.7）。

五条设计约束：

- **一次运行一种 delivery**；`track_id` 由 `Orchestrator` 参数统一决定并向下传播（§2.1）。
- **单一包装 + 单一预检**：四条消费路径共用 `portfolio_for_submission()` / `preflight_submission()`。
- **单文件父包字节不变**；**非 envelope 文本文档的发布判定逐字节不变**（§2.6，含 CRLF 回归）。
- **harness 权威不动**：父包 manifest 由 harness 生成，候选只贡献 `entries` 与 `geometry_entry`。
- **不改 R5 字段与 `verify_subject_document`**：发布身份靠读盘 decode + track 反查绑定。

---

## 1. 现状复核

### 1.1 已验证可用

| 层 | 事实 | 锚点 |
|---|---|---|
| Track | `msl-single`（`device_kernel`）与 `msl-bundle`（`device_kernel`/`host_tiling`/`schedule`，能力 `workspace`+`multi_stage`） | `backends/metal/tracks.json` |
| 交付形状 | 子包：恰一个 `host_tiling`、≥1 device、几何 entry 固定 `host/execution_geometry.json`(role `schedule`)、device 源必须保留 `taic-launch` 块 | `track_adapter.py:398-420`、`candidate.py:1351-1388` |
| 父包→子包投影 | 父包几何 entry role 是 `host_tiling`（`candidate.py:83`），投影后改为固定路径的 `schedule`（media type 也被改写，见 §2.2 负例） | `candidate.py:83-96,1466-1495` |
| Host work | `HostTilingPlan`（闭集 JSON，只支持 `kind: linear`）在 host 真执行；`start/step/count/stride/offset` 全部按实例符号求值为**整数** | `v2_runtime.py:113-260`（`_integer` 在 `:169-174`） |
| workspace 张量形状 | `space: workspace` 需要 `role: output` + 显式 `dtype/layout/strides/access`；fp32 让 span 是 4 的倍数（launch 语法要求） | `tests/support/shape_portfolio_metal.py:175-188` |
| 计时归属 | `inside_timing` 由 harness 派生（任一激活 stage 声明 `{name, zero}` → 进计时窗）；runtime 硬拒「写 zero workspace 却在计时外」 | `portfolio_runner.py:_lifecycle_factory`、`v2_runtime.py:604-614` |
| `inside_timing=true` 真机 | 已有直接 `evaluate_task` 正例 | `tests/backends/metal/test_track_a4.py:1149-1176` |
| 清零语义 | v1 与 v2 都只清**声明 `zero`** 的 workspace | `backend.py:633`、`v2_runtime.py:641`、`evaluate.py:660` |
| 正确性 workspace | 正确性三试次对每个 workspace 都填：`out2` 零、`out1/out3` poison | `tcenv/track/protocol.py:107-140` |
| Envelope | `taic-envelope 1`；只有 manifest/path/media_type 是 base64，body 明文；上限**精确 1,000,000 bytes**、≤64 entries | `tcenv/track/envelope.py:30-40,87-215` |
| 唯一 launch 块 | `unique_launch_block` 拒绝第二个块/伪标记；`parse_launch_spec` 无块返回 `None`、非法抛 `LaunchRejected(ValueError)` | `launch.py:68-143`、`backend/base.py:27` |
| workspace 字节核对 | `workspace_size_problems(abi, geometry, spec)`；今天唯一调用点是 runtime bind | `v2_runtime.py:522-547,586` |
| 组合证据 | 每期强制 geometry identity、逐 stage dispatch、bundle 必须真跑过 host plan | `portfolio_runner.py:575-612,900-1032` |
| 确认 | `confirm_subject` 用 freeze 父包 + 每期重跑 `run_portfolio_task` | `portfolio_confirmation.py:130-145` |
| 发布谓词（v2） | sidecar 的候选摘要必须 == `result.confirmation_source_sha256` == `best.metal` 哈希 == 胜者 `kernel.metal` 哈希；原因码是**闭集** | `experiments/common.py:664-683,730-760`、`taic/policy.py:791-798` |
| catalog | `archive_confirmed` 只记 `source_sha256`（提交文本），blob 按它寻址；`best.*` 与 `meta.json` 各自原子写 | `taic/kernel_catalog.py:149-180` |
| 受信提示面 | `_PROMPT_FUNCTIONS` 含 `system_prompt`/`render_contract`/`_cap_task` 等，**不含** `system_prompt_v2`/`_check_command` | `taic/trust.py:77-88` |
| 单文件真机 | host tiling 真执行、workspace 回读、单 dispatch、lifecycle 不计时 | `tests/backends/metal/test_shape_portfolio_gpu.py:269-351` |

### 1.2 前几版写错的地方（本版已改）

| 说法（v5 及更早） | 代码事实 | 后果 |
|---|---|---|
| 「envelope decode 出 `track_id`/`track_version` 与 `result["track"]` 比对」 | envelope 只传输 `track_id`（`envelope.py:101-109,265-271`），`CandidateArtifact` 无 `track_version`（`plane.py:368-374`）；而且还与「不改 envelope 格式」矛盾 | 无法实现；改为只校 `track_id` + registry 交叉校版本 |
| 「放宽 `_search` 的门就够了」 | `orchestrator.py:2368,2373` 在门内执行 `self.from_run / self.best_name`，`@catalog` 时 `from_run is None` → `TypeError` | 端到端验收会在 prior 评测前就崩 |
| 「不读表的另一颗 bundle 判不合格」 | `workspace_probe`（`v2_runtime.py:945`）只记录最终 workspace 字节，观察不到 device 是否读过；且重算是合法算法 | 侵犯候选自由、且证据不支持 |
| 「`known_good.metal` 直接用作 `eval-v2 --track msl-bundle` 的提交」 | `known_good.metal` 是 pack 文件、纯 MSL 文本（`loader.py:55,64,418-432`）；bundle 收到非 envelope 必须拒 | 自相矛盾 |
| 「resume 一律从 `track.track_id` 取值」 | `artifacts.py:386-392` 放行 schema<3，旧 v2 配置可能没有 `track` | 旧目录无法 resume |
| 「resume 在 `cli.py:426` 重建 binding 时 `replace(track_id=)`」 | 那个局部 `binding` 只用于 `load_snapshot`；真正运行的 `Orchestrator` 在 `cli.py:539-550` 重新构造，并在 `orchestrator.py:640` 再 `bind_task` 一次；`cmd_run`（`cli.py:185-199`）也没传 track | 接线无效，仍落回 `msl-single` |
| G13「envelope 必然不匹配 `_KERNEL_DECL`，所以被当成没有内核」 | `_KERNEL_DECL = r"\bkernel\s+void\b"`（`orchestrator.py:433`），而 envelope 的 device body 是明文，**会匹配** | 方向写反：真风险是「不解码就接受」，含残缺/被改的 envelope |
| 「`--from`/`from_catalog` 只是静默跳过」 | `_search` 只在 `self.from_run is not None` 时调 `_prior_kernel_text`（`orchestrator.py:2362-2365`），而 `@catalog` 被转成 `from_run=None`（`cli.py:177-179`）；`_from_text` 与 dup 豁免也只为 `from_run` 设置（`orchestrator.py:2410,2435-2437`），catalog 的 `best.*` 还会进 `_external_kernel_texts`（`:2452-2460`） | `@catalog` 根本取不到 seed，即便能取也会被判 `external_duplicate` |
| 「bundle 会话交单文件 → bridge 返回 `invalid_request`」 | 单文件能构造出合法请求；按 §2.3 它在 preflight 变成 rejected `EvaluationResult`，bridge 因此返回 `ok=true, result=...`（`selfcheck_bridge.py:411-414`） | 与本计划自己的 typed preflight 设计冲突 |
| 异常闭集只有 `EnvelopeError`/`TaskContractError`/`GeometryError`/`LaunchRejected` | `envelope.py:180` 的 `line(b"taic-manifest ").decode("ascii")` 对非 ASCII 直接抛 `UnicodeDecodeError`；`HostTilingPlan.from_candidate`（`v2_runtime.py:145`）对 `schema_version=null` 抛 `TypeError`、字符串抛 `ValueError` | 两类都能在预检/请求构造处炸掉搜索 |
| B4b「只改 `experiments/common.py` 与 `_prior_kernel_text`」 | catalog 只持久化 `source_sha256`（`kernel_catalog.py:167-174`），`_archive_if_confirmed`（`orchestrator.py:4042-4054`）也没传 manifest/track | §2.6 的「同时保留两种摘要」在实现上无处存放 |
| 「transport 哈希用 `read_text()`」 | `read_text()` 会做换行归一（`\r\n`→`\n`），与原始字节不同 | CRLF 旧单文件的「判定不变」不成立 |
| 「受信面只有 `HARNESS_RULES`/`render_contract`/SKILL.md」 | `system_prompt_v2` 与 `_check_command` 都不在 `_PROMPT_FUNCTIONS`；本计划两者都要改 | 以后对这两个模型可见函数的独立修改会绕过指纹 |
| 「GPU 许可只在 B5」 | B4 已有 `confirm_subject` GPU 验收 | 许可见 §4 总则 |
| 「上限 1 MiB」 | 精确 `MAX_ENVELOPE_BYTES = 1_000_000` 字节（`envelope.py:35`） | 预算口径要写准 |
| 打包器负例「重复 role」 | 多个 `device_kernel` 是**合法**的（B5-a 正例就是多源） | 负例自相矛盾，需删除 |
| 「`test_structure.py`/`test_opspec_v2.py`/`test_v2_g0_reference_outputs.py` 覆盖新题」 | `test_opspec_v2.py` 用固定 fixture；`test_v2_g0_reference_outputs.py:96-102` 只参数化 RMSNorm/sigmoid | 不能当作新题证据 |

### 1.3 代码缺口 A：发布身份与 catalog 持久化

- `orchestrator.py:4093`：`confirmation_source_sha256 = sha256(self._confirm_winner.source)` —— 提交文本哈希；
- sidecar 的 `candidate_sha256` 来自 `verify_subject_document`，是候选 **manifest** digest；
- `experiments/common.py:747-755` 要求两者相等，还要求等于 `best.metal` 与胜者 `kernel.metal` 的哈希。

bundle 下前两者必然不同 → `sha_mismatch` → 永不发布、永不归档。另外：
`_confirmation_files`（`common.py:664-683`）硬编码读 `best.metal`/`candidates/*/kernel.metal`（保留现有后缀即可）；
catalog 只存 `source_sha256`，没有 manifest、没有 track，所以「读者能分辨打包提交」缺少落盘字段；
而 `orchestrator._prior_kernel_text` 用 `read_text()` 读旧内核，换行归一后与原始字节的 transport 哈希不一致。

### 1.4 代码缺口 B：CPU 预检缺 launch 唯一性与 launch↔ABI workspace 核对

`_require_launch_retained(single=False)`（`candidate.py:1384-1388`）只要求「至少一个 carrier」，每个 device
单独检查；两个 device 各带一个块 → 投影通过，直到 `MetalV2Executor.__init__` 对合并且 `parse_launch_spec`
才炸。`workspace_size_problems` 今天唯一调用点是 `v2_runtime.py:586`（runtime bind），`resolve_geometry`
只核 workspace 名在 ABI 里。两项都是纯 CPU 函数，补进预检后「不 build」才成立。

### 1.5 需顺手修的崩溃缺陷

`track_adapter.py:198-199` 的 `sorted(CandidateEntry)` 缺 `key=` → `TypeError`（对照 `plane.py:409` 已写
`key=lambda entry: entry.path`）。修法：两处补 `key`；回归放 `tests/backends/metal/`：反序双 device 拼接确定性、
换序不改 `candidate_sha256`。

### 1.6 交付错误的三类泄漏（本版修正）

1. `decode_envelope` 构造 `CandidateEntry`/`CandidateArtifact` 时，未知 role / 空 body / 非法路径 /
   大小写碰撞会由 `plane.py:334-340,383-397` 抛**裸 `TaskContractError`**（探针：manifest 与 header 同为
   未知 role 即复现）；
2. `envelope.py:180` 对**非 ASCII** 的 `taic-manifest` 行直接 `UnicodeDecodeError`；
3. `HostTilingPlan.from_candidate`（`v2_runtime.py:145`）对 `schema_version: null` → `TypeError`、
   字符串 → `ValueError`（经投影泄漏）。

4. **大小超限**：`EvaluationRequest._validate` 对 `len(source_bytes) > MAX_SOURCE_BYTES`（**精确
   1,000,000**，`evaluation.py:168-170`）抛的是**普通 `EvaluationError`**，既不区分「envelope 坏」也不区分
   「单文件太大」；bundle 会话里一个超限的单文件或不完整的超限 envelope 都会走这里。

`tcenv/evaluation.py:73-80` 只捕 `EnvelopeError`；orchestrator 在请求构造处不捕 `ValueError` → 上面四类
（`TaskContractError`/`UnicodeDecodeError`/oversize `EvaluationError`）都会终止整次运行（CLI 只捕
`TaskBindingError` → traceback）。bridge 今天恰好安全（它们都是 `ValueError`，被
`selfcheck_bridge.py:415-420` 捕成 `invalid_request`），但那是巧合，不是设计。

### 1.7 代码缺口 C：上一版内核作为 seed 的路径

- `_search` 的门是 `if self.from_run is not None`（`orchestrator.py:2362`），而 `@catalog` 被转成
  `from_run=None` + `from_catalog=True`（`cli.py:177-179`）→ `_prior_kernel_text()` 的 catalog 分支
  永远不执行；**并且门内 `:2368,2373` 直接算 `self.from_run / self.best_name`**，所以只放宽门会立刻
  `TypeError`——必须先引入统一的 prior 来源显示名/路径（运行目录或 catalog 的 `best.*`）；
- 即便修了门，`_from_text` 与 dup 豁免也只为 `from_run` 设置（`orchestrator.py:2410,2435-2437`），
  而 catalog 的 `best.*` 会进 `_external_kernel_texts()`（`:2452-2460`）→ 种子会被判 `external_duplicate`；
- `_prior_kernel_text` 用 `_KERNEL_DECL` 判定「是不是一颗核」，envelope 因 body 明文而**通过**，
  于是残缺/被改的 envelope 也会被当成 seed 交给评测（没有 decode、没有 manifest 校验、没有 track 校验）；
- 读盘用 `read_text()`（`:2437`），换行归一后与原始字节身份不一致。

### 1.8 缺口清单

| # | 缺口 | 锚点 | 要补什么 |
|---|---|---|---|
| G1 | 无 envelope 生成工具 | — | `scripts/msl_bundle_pack.py`（唯一生产者 `encode_envelope`，原子替换） |
| G2 | 父包包装单文件专用 | `search_candidate.py:94-133` | `portfolio_for_submission()`（返回已解析对象） |
| G3 | 评测入口固定 single | `search_candidate.py:164-199` | 预检共用；按 provider 形态决定注入 launch 解析器 |
| G4 | 预检链不存在 | `taic/v2_evaluation.py:79-116` | `preflight_submission()`（含 §1.4 两项） |
| G5 | 交付错误在请求构造处变成基础设施故障 | `evaluation.py:73-80,140-145`、`orchestrator.py:1664-1682` | `CandidateDeliveryError` + 调用点窄转（§1.6） |
| G6 | CLI 无错误分类 | `tcenv/cli.py:120-134` | 共用分类入口 |
| G7 | 多 device 源 TypeError | `track_adapter.py:198-199` | 补 `key=` + 回归 |
| G8 | 跨 device launch 唯一性 / launch↔ABI workspace 只在 build 期检查 | `candidate.py:1384-1388`、`v2_runtime.py:586` | 预检里显式调用 |
| G9 | freeze 单文件专用 | `search_candidate.py:220-306` | 共用包装函数 |
| G10 | `confirmation_expected` 仍走 single-only `portfolio_for` | `taic/v2_evaluation.py:151-160` | 切成同一包装函数 |
| G11 | track 传播链不存在（含 resume 与 Orchestrator） | `cli.py:185-199,426,539-550`、`orchestrator.py:640,1104,3249-3267,4119` | `Orchestrator(track_id=)` + 优先级 + 记录/恢复（§2.1） |
| G12 | **发布身份冲突** | `orchestrator.py:4093`、`common.py:664-683,730-760` | 读盘 decode 绑定（§2.6） |
| G13 | catalog/seed 路径（含 `@catalog` 死路、残缺 envelope 被接受、`read_text` 归一） | `orchestrator.py:2362,2410,2435-2437,2452-2460`、`kernel_catalog.py:149-180` | §1.7 / B4b |
| G14 | workspace 清零文案错（v1/v2 都只清声明 `zero`） | `prompts.py:73-74`、`kernel_agent.py:662` | 全局改正 + 必做重签 |
| G15 | 题面无 bundle 段；**两处**命令生成都不带 track：`_check_command`（自测）与 `_failure_record`（differential-recheck 人工复测） | `prompts.py:142-200`、`kernel_agent.py:524-585`、`orchestrator.py:2069` | v2 条件段 + 两处 `--track` 注入 |
| G16 | 两个模型可见函数不在受信面 | `taic/trust.py:77-88` | 登记 `system_prompt_v2`、`_check_command` |
| G17 | 无「搜索派生 lifecycle」正例、无多阶段/多源组合 | 测试 | B5-a |
| G18 | 题库无 workspace 题 | `tasks/*/opspec.json` | 新增题（B5-b，含自带测试与 `eval-v2` 素材） |
| G19 | 无 bundle 新题的 evidence 支撑（现有三个测试都不覆盖新题） | `test_opspec_v2.py`、`test_v2_g0_reference_outputs.py:96-102` | 新题自己的 discovery/instantiation/reference 全尺寸测试 |
| G20 | 原因码闭集不含 track 不匹配 | `taic/policy.py:791-798` | 增 `track_mismatch`（追加，不改既有码） |
| G21 | **交卷读取会破坏 envelope 字节**：`collect_session` 用 `read_text(..., errors="replace")`（CRLF→LF、非法 UTF-8→U+FFFD），再经 `kernel_text()` 抽代码块。**并且它不能抛异常**：它跑在会话执行阶段（`kernel_agent.py:1428`、`session_runtime.py:484`），`_one_session_attempt` 的 `except Exception`（`orchestrator.py:2661-2676`）会把它记成 `agent_failed` + `cost_usd=None` + `cost_incomplete=True`，既到不了请求构造处的 rejected 转换，又会丢掉已计费的成本（`kernel_agent.py:1314-1315` 已明说） | `taic/kernel_agent.py:1313-1316`、`taic/session_supervisor.py:38-58`、`orchestrator.py:2661-2695` | bundle 交卷：`read_bytes()` + 不归一解码 + 跳过 `kernel_text()`；**非法 UTF-8 不抛**，返回**保留 `cost_usd`/`spent`/`cost_incomplete(=not saw_tokens)`/`status`/`error`** 并追加 `diagnostics.delivery` 的 `AgentResult`（`failure_class="candidate_delivery"` **仅当 `status=="ok"`**）；`classify_result` 新增 `candidate_delivery → delivery_rejected`（不吞 timeout/process/turn）；`_one_session_attempt` **在既有费用分支之后**调 `_record_delivery_rejection(...)` 生成 typed rejected（无 GPU/outbox，未知费用仍 `StopRun("cost_incomplete")`）；single 旧行为不变 |

---

## 2. 接入设计

### 2.0 几何派生策略（保底 + 可覆盖 + 可选强制）

**偏差：** 代码里没有 `search.geometry_mode`。本节表格是第 23 版计划原文，没有落地。现行规则见 §7.3 第 8 条：msl-single 必须自带 `/* taic-launch */`，harness 不派生 grid 或 threads。

下面到本节结束是未落地原文，不是现行规则。原文当时打算把可机械推导的几何交给 harness 做保底，并允许 LLM 覆盖；只有配置标明强制时才锁死。

| 模式 | 行为 | 配置 |
|---|---|---|
| **默认（auto，不特别标明）** | harness 派生几何作为**保底**写进题面（`grid/threads/workspace` 的默认实现），并**明确告诉 LLM「这是 harness 算的，你可以用也可以不用」**：LLM 可以直接采用（不写 `taic-launch`），也可以自己写 `taic-launch` 覆盖（想优化时自己选启动配置）。 | `search.geometry_mode = "auto"`（默认） |
| **强制（forced，用户特别标明）** | 必须按 harness 派生的几何执行，LLM **不能覆盖**（写了 `taic-launch` 也会被忽略/拒绝）。 | `search.geometry_mode = "forced"` |

要点：

- **保底不是限制**：harness 派生的是一个**正确的默认几何**（`grid = shape/tile`、`threads = warp*32`、workspace 按声明求和），保证「不写启动配置也能跑对」。
- **可覆盖不是放纵**：LLM 想写非标准但更快的启动配置时，**允许并鼓励**它自己写 `taic-launch` 覆盖默认——这正是「不限制 LLM 后续内化能力」：可推导的机械化（保底），需要创造力的留给 LLM（覆盖）。
- **强制是用户的选择**：只有当用户通过配置特别标明强制时，才锁死为 harness 派生（比如做对照实验、固定几何以排除启动配置变量）。平时不强制。
- **题面必须说清**：`TASK.md`/系统提示里明确写「harness 已为你算出默认几何 `grid=…, threads=…`，你可以直接采用，也可以自己写 `taic-launch` 覆盖；除非任务被配置为强制」，避免 LLM 误以为「只能照抄」。

这条原则同时吸收 KDA/CAKE2「把可推导的机械化」的经验，又不锁死 LLM 发现非标准好配置的空间（比如 init 那版 16 TG 就不是 `M/BM` 规则能推出来的）。

**为什么这样设计合理（本机实测，Apple M4 Max，`gemm` small 档 M=256,N=256,K=128，fp16）**：

- **保底降低门槛**：不写启动配置时，harness 按 `grid=(N/BN,M/BM)=(4,4,1)`、`threads=(32,2,1)` 派生，kernel 正确且接近 MLX 水平——不写配置也能跑对，但不锁死。
- **可覆盖保住创造力**：搜索里 LLM 自选 **16 个线程组 × 256 线程**（不是 `M/BM` 规则能推出来的），实测比默认 4×4 更快。若 harness 全包几何，这个更优解根本不会出现。
- **强制用于对照**：比较两种算法在同一 grid/threads 下的纯算法差异时，`geometry_mode=forced` 锁死为 harness 派生，LLM 不能覆盖，对照才公平。
- **机械化省掉重复学习成本**：让 LLM 学会选启动配置要「读文档 + 写探针 + 测量 + 修正」，每个候选每个 shape 都重来一遍；harness 派生是零成本机械计算，一次写对、所有候选受益。这是「把可推导的机械化」的收益，不是「LLM 不行」。

这四条合起来：保底降低门槛、可覆盖保住创造力、强制保证对照、机械化省掉重复劳动——是被实测反复验证过的取舍，不是闭门造车。

### 2.1 track：取值、优先级与传播（一次说清）

两个名字不许混：**`track_id`** = `"msl-single"` / `"msl-bundle"`；**`delivery`** = registry 的
`"single_source"` / `"candidate_bundle"`（只用于校验，不单独传输）。

**新运行的优先级**：CLI `--track` > `taic.toml` 同名项 > `msl-single`。parser 的 `--track` 必须
`default=None` 作为**省略哨兵**（这样「未指定」与「显式 `msl-single`」可区分，resume 才谈得上校验）。

**唯一入口是 Orchestrator 参数**（v4 的局部 `replace` 是无效接线）：

| 环节 | 改动 |
|---|---|
| `Orchestrator.__init__` | 增加 `track_id: str = "msl-single"`；内部：`track = load_track_registry().get(track_id)` + `require_available()`；`track_id=="msl-bundle"` 且题目无 `space: workspace` → `RunStartError`；然后**先重建唯一的局部变量** `binding = replace(binding, track_id=track_id)`，**再** `self.binding = binding`（`bind_task` 本身不加参数，`task_binding.py:178-187`）。顺序不能反：`system_prompt_v2(binding)`（`orchestrator.py:693`）与 `V2LocalEvaluator(binding)`（`:743`）读的是**局部变量**，只改 `self.binding` 时题面与评测器仍会落回 `msl-single` |
| `cmd_run`（`cli.py:185-199`） | 把解析出的 track 传给 `Orchestrator(...)`；**不传就不生效** |
| `prepare_resume`（`cli.py:426,539-550`） | 从 `run_config.json` 的 `track` 块取记录值传给 `Orchestrator(track_id=recorded)`；**旧配置兼容**：`artifacts.py:386-392` 放行 schema<3，所以 (a) v2 `run_config` 缺失 `track`（旧目录）→ 按 `msl-single` 恢复；(b) v1 保持 `track=None`（不传）；(c) 只有 `track` **显式存在但非法**才 `ResumeRefused`。判据包含**内部一致性**：`delivery` 必须等于 `registry.get(track_id, version=track_version).delivery`，否则（例如 `track_id=msl-bundle` + `delivery=single_source`）→ `ResumeRefused`；`track` 块畸形（非 dict、缺字段、类型错）→ `ResumeRefused`（不抛裸异常）。`taic resume` 的 parser 加 `--track`（`default=None`）：给了且与记录不同 → `ResumeRefused` |
| run_config/result | `track` 块由 `load_track_registry().get(self.binding.track_id).document()` 派生（含 `track_version`/`delivery`），替掉硬写的 `"single_source"`（`orchestrator.py:1104,4119`） |
| 评测 | `V2LocalEvaluator(binding)` 用 `self.binding.track_id`；`evaluate_candidate(..., track_id=...)` 显式传参 |
| 确认 | `_confirm_v2` → `confirm_candidate(..., track_id=self.binding.track_id)` → `confirmation_input` 选包装分支 |
| 会话 | `SessionBrief.track_id`：**字段新增归 B0**（`kernel_agent.py:84-143` 今天没有；`collect_session` 只拿得到 brief，收不到 binding），**填值归 B2**（`orchestrator.py:3249-3267` 用它构造 brief 时传 `self.binding.track_id`）→ `_check_command` 追 `--track`；`render_contract` 条件段用它 |
| 人工复测命令 | `_failure_record`（`orchestrator.py:2069`）写出的 **`retest_command`**（键名见 `_RECHECK_KEYS`，`orchestrator.py:144`；赋值处 `:2086`）也必须在 bundle 运行里追加 `--track msl-bundle`（否则 differential-recheck 记录里那条命令对 envelope 文件会按 single 处理并拒绝）；single 运行该字段逐字符不变 |
| 自测 bridge | **不新增参数**：bridge 用的 `V2LocalEvaluator` 的 binding 已带 track；只要求 `preflight_submission()` 在 bridge 路径生效 |

- **只有 v2 才重建带 track 的 binding**：v1 的 `--track` 只允许 `msl-single`（等价于不传）；
  v1 上显式给 `--track msl-bundle` → **启动期 `RunStartError`**，不得静默跑成 v1 single。
- **注入执行器也要核**：调用方注入 `V2LocalEvaluator`（测试或远端）时，`Orchestrator` 必须校验
  `evaluator.binding.track_id == self.binding.track_id`；不一致 → `RunStartError`
  （否则题面 bundle、评测 single 会静默分裂）。
- **一次运行一种 delivery**：实交不匹配 → typed rejected（bundle 运行交单文件 / single 运行交 envelope）。

**验收必须断言运行对象**（v4 的教训）：① `orch.binding.track_id`；② 真实 evaluator 的
`orch.evaluator.binding.track_id`（v2 本地执行器）；③ `orchestrator.py:3249-3267` 生成的
`SessionBrief.track_id`。另加：resume 不带 `--track` 时上述三者仍是 bundle；带不同值 → `ResumeRefused`。

**后缀不改**：`file_ext` 由后端决定，牵动 `SessionBrief.file_ext`、`kernel_name`/`best_name`、
`resume._kernel_name`、`kernel_catalog`/`from_catalog`。bundle 的提交文件仍是 `<...>.metal`，靠 magic 自识别。

**seed（定案）**：`seed_name` 是只读 property（`kernel_agent.py:144-149`），`prepare_session`（`:1118-1123`）
直接 `work / brief.seed_name`；所以不改名字、不传 `None`：首场 bundle 会话（无合法 envelope seed）
用 `seedless=True`；后续/续跑把 **envelope 原文**放进 `seed_source`（文件仍叫 `kernel.metal`）。
任何情况都不把 envelope 拆成单文件。

### 2.2 唯一包装函数（返回已解析对象）

```python
portfolio_for_submission(submission: str, plan, *, authority, track_id: str)
    -> tuple[CandidateArtifact, PortfolioCandidate]
```

- **single 分支**：先做形态检查——`is_candidate_envelope(text)` 或「含 `taic-envelope 1` 行但不完整」
  → `CandidateDeliveryError`；然后复用 `envelope_entries()` 的产物（`device/candidate.metal` +
  `coverage_plan/portfolio.json`、`geometry_entry=None`、planner `{"id":"taic.search.single","version":1}`、
  `PROTOCOL_EPOCH=1`）并 `parse_portfolio()` 一次，保证单文件父包逐字节不变。
- **bundle 分支**：`is_candidate_envelope()` 为假或 `decode_envelope()` 失败 → `CandidateDeliveryError`；
  断言 `candidate.track_id == "msl-bundle"`（不静默重贴 track），再机械映射：

| 子包 entry（role） | 父包 entry（role） | 附加校验 |
|---|---|---|
| `device_kernel`（≥1，路径不变） | `device_kernel`，同路径同字节 | 多源需先修 §1.5；**允许 ≥2 个** |
| `host_tiling`（恰 1 = Host Tiling plan） | `host_tiling`，同路径同字节 | 不得落在 `host/execution_geometry.json` |
| `schedule`（`host/execution_geometry.json`） | **`host_tiling`**，同路径同字节 | 路径固定 **且 `media_type == "application/json"`** |
| （无） | `coverage_plan/portfolio.json`（role `launch`） | harness 生成：单 variant、恒真 guard、`fallback=None` |

manifest 差异（仅 bundle）：`variants[0].entries` = 全部非几何 entry；`variants[0].geometry_entry` = 几何路径；
planner `{"id":"taic.search.bundle","version":1}`；`freeze.protocol_epoch = 2`（`validation.py:130` 只校验 `>=1`）。
函数返回**已解析** `PortfolioCandidate` + 解码后 `CandidateArtifact`，调用方不再二次解析。

拒绝清单（可行动诊断，全部 CPU）：① bundle 运行收到非 envelope/残缺 envelope；② single 运行收到 envelope；
③ 解码后 `track_id != "msl-bundle"`；④ `schedule` 数量/路径/media_type 不符；⑤ Host Tiling plan 与几何同路径；
⑥ 几何 entry 同时列在 `variants[0].entries`；⑦ `{device_kernel, host_tiling, schedule}` 之外的 role；
⑧ 题目无 workspace 张量；⑨ device 源无 `taic-launch`、含引号 `#include`。顺序可对齐
`evaluate_task`（`runner.py:614-625`）。

### 2.3 预检分层与异常闭集

`preflight_submission(binding, submission, *, target) -> PreflightOutcome`（`.candidate`/`.portfolio`/
`.rejection(stage, message)`），全部 CPU、**不 build**：

1. 形态与解码（§2.2，含 track_id、schedule 路径/media_type）；
2. 父包构造 + `parse_portfolio()`（一次）；
3. `project_provider()`（含 role/workspace/`HostTilingPlan`）；
4. **跨文件 launch 唯一性**：`unique_launch_block(combined_candidate_source(projection.candidate))`；
5. `resolve_geometry(候选几何, instance, abi, limits)`；
6. **launch↔ABI workspace**：`parse_launch_spec(combined, launch_constants(instance))` 为 `None` → 缺陷；
   `workspace_size_problems(abi, resolved, spec)` 非空 → 缺陷；
7. **Host Tiling plan 的 span 校验（纯 CPU，抽成共用 helper，必须在预检里）**：把
   `v2_runtime.py:198-206` 的规则抽成一个**纯函数**（例如 `host_write_span(write, abi_nbytes) ->
   problems: list[str]`），预检与 `execute()` **共用同一份**，避免两份规则漂移。规则必须与今天的执行
   语义逐条对齐：
   - `stride >= 1` **无条件**成立（`count == 0` 也不放行 `stride < 1`——这正是今天
     `v2_runtime.py:199-201` 的行为）；
   - `count == 0` 时 `span = 0`（**不是** `offset + (count-1)*stride + 1`；例如
     `count=0, stride=2, offset=0` 在朴素公式下会算成 `-1`，把运行时接受的空写误判为非法）；
   - `count > 0` 时 `span = offset + (count-1)*stride + 1`，并要求 `offset >= 0`；
   - `count >= 0`，且 `span * nbytes <= abi_workspace_bytes(workspace)`。
   **`offset >= 0` 是本期新加的限制，不是"与今天逐条对齐"**：今天只查 `count < 0 or stride < 1 or
   span < 0`（`v2_runtime.py:199-201`），所以 `offset=-1, count=2, stride=2` 的 span 是 3、会被放行，
   但第一次写入落在负地址。要兑现「预检拦住越界 host write」，就必须显式加这条；因此它同时要求
   **允许改 `HostTilingPlan.execute()`**（把执行侧的同一段规则换成共用 helper），这不是纯规范化，
   §2.5 白名单必须显式写出（见下）。

**异常闭集（漏一个就会让新负例变成搜索崩溃）**

| 来源 | 类型 | 处理 |
|---|---|---|
| decode（`envelope.py`） | `EnvelopeError`、候选引起的 `TaskContractError`、`UnicodeDecodeError` | 在**调用点**转 `CandidateDeliveryError`；`UnicodeDecodeError` 优先在 `envelope.py:180` 规范化成 `EnvelopeError`（§2.5 允许的最小硬化） |
| 父包 parse | `TaskContractError` | 归 `stage="delivery"`；「我们生成的 manifest 自身不合法」= harness bug，由 B0 自检用例先拓出来 |
| 投影 / 几何 | `GeometryError`、`TaskContractError`、`HostTilingPlan` 的 `TypeError/ValueError` | 前者原样归 `geometry`；后两者**在源头**规范化成 `TaskContractError`（§2.5），预检只认 `TaskContractError` |
| launch | `LaunchRejected`（`backend/base.py:27`） | `stage="launch"`；`parse_launch_spec` 的 `None` 也算缺陷 |
| harness 不变量 | —— | `run_portfolio_task` 内部的断言保持致命 |
| 大小 | `EvaluationRequest` 的 source 超限（`evaluation.py:168-170`） | 这是**提交文本自身**的属性（模型写了超大文件/超大 envelope），不是跑架故障 → 在该处抛 `CandidateDeliveryError`（内部保持 1,000,000 字节上限不变），让它和 decode 错误走同一条 rejected 路径 |
| 深嵌套 JSON（**三条源头**） | `RecursionError`：`envelope.py:181-184` 的 `loads_strict`、`delivery.py:180-186` 的 geometry `loads_strict`、以及 `v2_runtime.py:132-136` 的 `HostTilingPlan` 裸 `json.loads`——三者都只捕 `UnicodeDecodeError`/`SchemaError`/`JSONDecodeError`，深嵌套输入会漏出裸 `RecursionError` | 各自在源头规范化：envelope → `EnvelopeError`；geometry → `GeometryError`（`delivery.py`，或把两者的 `loads_strict` 调用统一到一个捕 `RecursionError` 的共享包装里，`schema.py` 同时被列入白名单）；Host Tiling → `TaskContractError`。调用点仍只认这三个类型，不用 broad `except` |
| 超长整数 JSON | `json.loads` 的裸 `ValueError`（Python 的 4300 位限制；`v2_runtime.py:132-136` 只捕 `UnicodeDecodeError/JSONDecodeError`） | 在 `HostTilingPlan.from_candidate` 里补 `ValueError`（含 `JSONDecodeError` 之外的那类）→ `TaskContractError`；几何 entry 的同款路径一起处理 |
| launch 十进制 | `int(tok)` 的裸 `ValueError`（`launch.py:536-541`，超长数字串） | 在 `launch.py` 源头转 `LaunchRejected`（它只继承 `ValueError`，预检只认它） |
| plain source 编码 | `self.source.encode("utf-8")` 的 `UnicodeEncodeError`（`evaluation.py:130-145` 的 `__post_init__` 与 `:166-170` 的 `_validate`；lone surrogate） | 在 `evaluation.py` 源头转 `CandidateDeliveryError`（与超限同一条路径） |
| 身份 | `candidate_sha256` 不符 | **永远 fail-closed**，不降级 |

**不要**用 `except ValueError` 一把捕——那会连 `TaskContractError` 之外的调用方错误一起吞掉。

三个使用点：`V2LocalEvaluator._evaluate`、`tcenv/cli.py:cmd_eval_v2`、`evaluate_candidate`。
orchestrator 请求构造边界只做一件事：`except CandidateDeliveryError` → `status="rejected"`,
`stage="delivery"` 的 `EvaluatedReport`（不建 outbox、不占 GPU、可确定性重现）。

**这条路只覆盖"已经拿到提交文本、进入评测请求"之后**。交卷读取阶段的交付错误（G21：非法 UTF-8）
发生在**会话执行**里，不能靠抛 `CandidateDeliveryError`——那里被 `_one_session_attempt` 的宽 `except`
接住后会变成 `agent_failed` + `cost_incomplete=True`。它走另一条闭合路径：
`collect_session` 返回已计费的 `AgentResult(diagnostics.failure_class="candidate_delivery")` →
`classify_result` → `delivery_rejected` → `_one_session_attempt` **入账之后**调用**同一个** rejected
构造器（`status="rejected"`、`stage="delivery"`）。两条路径必须产出同一种记录形状，只是入口不同。

### 2.4 消费路径同源（含 bridge 的行为）

| 路径 | 今天 | 改为 |
|---|---|---|
| 搜索期正式评测 | `V2LocalEvaluator._evaluate` → `evaluate_candidate` | `preflight_submission` + 包装分支 |
| agent 自测 | `tcenv/cli.py:cmd_eval_v2`（`:121-134` 用默认 binding、不传 track）→ `evaluate_candidate` | parser 加 `--track`（`default=None`）→ `replace(bind_task(...), track_id=...)` → `evaluate_candidate(..., track_id=...)`；**文件读取也要进同一条窄边界**：现在是 `src = Path(a.kernel).read_text()`（`tcenv/cli.py:111`）在分类 `try`（`:120`）**之前**，非法 UTF-8 直接 traceback → 改成 `read_bytes()` + 在 `try` 内 `decode("utf-8")`（`UnicodeDecodeError` → 退出码 2）；错误分类共用（退出码 2 = 参数/交付非法，1 = 候选判负） |
| selfcheck bridge | `evaluator.selfcheck(request)` | 同上。**交付不匹配的验收是 `ok=true` + `report.status="rejected"` + `stage="delivery"` + 无 GPU**（`selfcheck_bridge.py:411-414` 会把 rejected 结果原样返回，不会变成 `invalid_request`） |
| 确认 freeze | `confirmation_input` → `envelope_entries` | `portfolio_for_submission(..., track_id=...)` |
| 确认 expected | `V2LocalEvaluator.confirmation_expected` → single-only `portfolio_for` | 同一包装函数取 `variants[0]` |

`evaluate_candidate` 的 `geometry_resolver_factory`：只有父包含**没有** `geometry_entry` 的 provider 时才注入。

### 2.5 允许的改动面（改成白名单，避免「不改」与实现冲突）

允许（且只允许这些语义改动）：

- `tcenv/backends/metal/track_adapter.py:198-199`：`key=`；
- `tcenv/track/envelope.py:180`：非 ASCII manifest 行 → `EnvelopeError`（**不改格式**）；
- `tcenv/backends/metal/v2_runtime.py::HostTilingPlan.from_candidate:145`：`int(...)` 的
  `TypeError/ValueError` → `TaskContractError`（不改解析语义）；
- `tcenv/backends/metal/v2_runtime.py::HostTilingPlan.execute:198-206`：把 span 规则抽成**预检与
  `execute()` 共用的纯 helper**（预检侧归 `stage="delivery"`），并新增 `offset >= 0` 限制；
  这是行为改动（今天放行负 offset），必须由 §B0 的正负例覆盖；
- `tcenv/evaluation.py`：`CandidateDeliveryError` + decode 调用点窄转 + 源码超限与 `UnicodeEncodeError`
  的源头分类；
- `tcenv/track/envelope.py`、`tcenv/track/delivery.py`（或共享的 `tcenv/schema.py::loads_strict`）：
  `RecursionError` → `EnvelopeError` / `GeometryError`（与既有的非 ASCII manifest 同处）；
- `tcenv/backends/metal/launch.py`：`int(tok)` 的 `ValueError` → `LaunchRejected`；
- `tcenv/cli.py`：**`cmd_eval_v2` 在这里**（`tcenv/cli.py:87`，不在 `taic/cli.py`）——parser 加 `--track`、
  重建 binding、透传 `evaluate_candidate(..., track_id=...)`，并把它前面的
  `Path(a.kernel).read_text()`（`:111`）改成 `read_bytes()` + 窄边界内 `decode`（§2.4）；
- `tcenv/backends/metal/search_candidate.py`：包装/预检/`evaluate_candidate`/`confirmation_input`；
- `taic/cli.py`：`cmd_run`/`prepare_resume` 传 track、run/resume 两个 parser 加 `--track`（省略哨兵）。
  **TOML 的 `track` 只在 CLI 读**：用已有的 `load_raw_config()`（`taic/llm.py:350-361`，返回合并后的原始
  section 字典）取 `raw.get("search", {}).get("track")`；**不改 `taic/llm.py`，也不把 `track` 加进
  `SearchConfig`**（它是任务绑定事实，不是搜索旋钮）；
- `tcenv/evaluate.py`：新增纯函数 `rejected_report(...)`（共用 rejected `EvalReport` 工厂）；
- `taic/v2_evaluation.py`、`tcenv/cli.py`、`taic/orchestrator.py`、`taic/kernel_agent.py`、
  `taic/session_supervisor.py`、`taic/selfcheck_bridge.py`（只改文案/分类/新增无 GPU 落盘 helper，
  不改协议）；
- `taic/prompts.py`：`HARNESS_RULES` 的清零句纠错（常量正文允许改；**不为拆分而新建模块级常量**），
  `_rules_for(backend, delivery)` 增加 `delivery` 参数并在 bundle 分支**替换**四条冲突句，
  `system_prompt_v2` 的条件段与 `prompts.py:170` 改正。**替换文本与 bundle 段一律写在
  `_rules_for`/`system_prompt_v2` 的函数体内（具名局部量）**，不要新增未登记的模块级提示常量；
  若新增，则 `_PROMPT_CONSTANTS`（`trust.py:90-95`）必须同步追加，并纳入 B3 的判别力覆盖；
- `taic/kernel_agent.py::SessionBrief`：**新增字段 `track_id: str = "msl-single"`**（`kernel_agent.py:84-143`
  今天没有它，而 `collect_session(brief, work, invocation)`（`:1313`）只拿得到 `brief`——它**收不到
  binding**，所以"替换 binding"不可能让 collect 分支知道自己是 bundle）。字段在 **B0** 落地，
  B0 测试用 `dataclasses.replace(brief, track_id="msl-bundle")`；**B2 只负责由真实 `Orchestrator`
  填它**（`orchestrator.py` 构造 brief 处传 `self.binding.track_id`）。
- `taic/kernel_agent.py::collect_session:1313-1316`：**bundle 交卷读取**改为
  `submit_path.read_bytes()` 后**不做换行归一**地严格 `decode("utf-8")`，并**跳过** `kernel_text()`
  抽块（envelope 必须逐字节原样进入 preflight/decode）；single 仍走今天的 `read_text(errors="replace")`
  + `kernel_text()`，逐字节不变。
  **`collect_session` 绝不因为交付非法而抛异常**（它跑在会话执行阶段，见 G21），并且**不许改写费用/
  状态语义**：
  1. **费用字段原样保留**：`cost_usd=invocation.cost_usd`、`spent=<今天的统计>`、
     `cost_incomplete=not invocation.saw_tokens`（`kernel_agent.py:1377` 的既有写法）。
     **不许硬编码 `cost_incomplete=False`**：`_one_session_attempt` 在 `orchestrator.py:2677-2688`
     先看这个字段再入账，硬编码会把"没有 token 计数"的未知费用按 0 入账——正是要避免的成本丢失。
  2. **状态/错误字段原样保留**：`status=invocation.status`、`error=invocation.error`，并保留今天算出的
     `diagnostics`（`failure_class` 的既有分支不动）。**不许硬编码 `status="ok"`**，否则会把
     `timeout`/`process_failed` 改写成"正常完成"、再被新分支归成 `delivery_rejected`，掩盖
     `classify_result` 现有的 agent/provider 失败语义。
  3. **`candidate_delivery` 只用于一种情形**：会话**正常完成**（`status == "ok"`）、确实写了提交文件、
     但字节不是合法 UTF-8。此时在既有 diagnostics 上**追加**
     `diagnostics["delivery"] = {"stage": "delivery", "reason": "invalid_utf8", "path": ...,
     "message": ...}`，并把 `failure_class` 设为 `"candidate_delivery"`；`source=None`。
     若 `status != "ok"`，**不加** `delivery`、不改 `failure_class`，走今天的 timeout/process/turn 语义。
- `taic/session_supervisor.py::classify_result:38-58`：新增一条分支
  `failure_class == "candidate_delivery"` → 新 outcome `"delivery_rejected"`；它只在
  `status == "ok"` 且 `source is None` 时可达（见上一条），因此不会吞掉 timeout/process/turn failure；
- `taic/orchestrator.py::_one_session_attempt:2677-2700`：费用分支**不动**
  （`cost_incomplete` 仍然先 `StopRun`，已知费用仍然 `ledger.charge(...)` 入账）；
  `outcome == "delivery_rejected"` 时，在 `ledger.charge(...)` **之后**调用下面这个**无 GPU 落盘**
  helper（不是 `_eval`），然后照常进入下一轮：
  - **共用的 `EvalReport` rejected 工厂**：在 `tcenv/evaluate.py` 里加一个纯函数
    `rejected_report(task, dtype, size, mode, *, stage, message, seed, target_id) -> EvalReport`
    （`status="rejected"`、无 `build`、`BUILD_FREE_STATUSES` 覆盖）；`V2LocalEvaluator._rejected`
    （`taic/v2_evaluation.py:62-76`）改为调它（它仍然需要合法的 `EvaluationRequest`，所以**不能**用于
    非法 UTF-8 路径——那是本计划要写清的点）；
  - **`Orchestrator._record_delivery_rejection(label, *, stage, message, plan, origin, diagnostics_path)
    -> Candidate`**：照 `_eval` 的记录形状手工做这几件事（**不要复制半套 `_eval`**，也不要只剩事件没有候选）：
    `cand = Candidate(idx=len(self.candidates), label=label, source="", plan=plan,
    correctness_only=self.search.correctness_only, origin=dict(origin or {}))`
    → `cand.report = rejected_report(...)`（`stage="delivery"`，`message` 里带上 diagnostics 路径与
    "重新打包并按字节提交"的可行动提示）→ `cand.cost_usd = self.ledger.total_usd`、
    `cand.t_s = self._elapsed()` → `self.candidates.append(cand)` → `self._save(cand)` →
    `self.trace.append({"idx": cand.idx, "label": label, "status": "rejected", "speedup": None,
    "cost_usd": cand.cost_usd, "t_s": cand.t_s, "improved": False, "within_noise": False,
    "delivery": {"stage": "delivery", "reason": ..., "diagnostics_path": ...}, ...})` →
    `self._publish_state(inflight=None)` → 把 `cand.report.message` 放进 `self._failure_feedback`，
    让下一场看到"上次交卷不是合法 UTF-8"。
    明确**不做**：改 `self.best`、动 `hangs`/`crashes`、建 outbox、调 `_eval`/`_evaluate_report`。
    另外 `_one_session_attempt` 必须把返回值接住：`cand = self._record_delivery_rejection(...)`，
    否则 `finally` 里的 `_record_feedback(..., cand, ...)` 拿不到刚落的候选（会变成只有事件、没有候选）。
  - **`Candidate.source` 记为空串**（不是 U+FFFD 替换后的文本）。**`_save` 照常调用**，因此磁盘上会有
    `candidates/NNN_<label>/<kernel_name>` 这个**空串占位文件**——这是刻意的，不是"没有内核文件"：
    `Candidate` 根本没有 `kernel_path` 字段（`orchestrator.py:227-270`），`_save`
    （`:1469-1482`）无条件写 `d / self.kernel_name`；而 `resume._read_candidates`
    （`taic/resume.py:553-590`）会**跳过没有内核文件的目录**，`resume.py:867-870` 还会在
    "目录非空但一条候选都读不出"时以 `no candidates/ were stored` 拒绝续跑。所以占位文件必须存在，
    计划里不再出现"不写 kernel 文件"或"`kernel_path` 未设"的说法。
  - **空 source 永不进 cache**（两条路都要守）：新路径不往 `self.cache` 里放；restore 路径
    （`orchestrator.py:1405-1413`）用 `self._cache_this(cand.source) and ...`
    把每颗候选按 `_sha(source)` 塞进 cache，而 `_cache_this("")` 今天返回 `True`
    （`:1182-1192` 只看 `"taic-launch" in source`）——必须加一条 `if not cand.source: 不进 cache`
    （或让 `_cache_this` 对空串返回 `False`）。否则续跑后 `_sha("")` 会成为一条"可信缓存"，
    与"这颗候选没有可信源码"直接矛盾。
- `taic/kernel_agent.py::render_contract:672-674`：TASK.md 的 v2 中文 launch 指引按
  `brief.track_id == "msl-bundle"` 换成"dispatch 由 geometry 决定；`taic-launch` 只承担 math 与
  workspace↔ABI 交叉检查"（single/v1 分支除清零句外保持原文）；
- `taic/task_binding.py`：`signature_stub()` **不加参数**——方法内读已有的 `self.track_id`
  （`task_binding.py:38`；`presentation()` 已经是这么做的，`:110`），`delivery == "candidate_bundle"`
  时保留 workspace 参数与可写性、single 输出逐字节不变。**两个现有调用点都不改签名**：
  `prompts.py:176`（系统提示）与 `orchestrator.py:3234`（`SessionBrief.abi_stub`）继续无参调用——
  新增一个 `track_id=` 参数只会制造"binding 的 track 与方法参数的 track 分裂"的可能；
- `taic/trust.py`：登记 `system_prompt_v2`、`_check_command` **与
  `task_binding.TaskBinding.signature_stub`** 到 `_PROMPT_FUNCTIONS`；并让 `_symbol_bytes`
  （`:155-172`）支持**点分属性路径**（只哈希 `TaskBinding.signature_stub` 的方法源码，解析失败显式报错）；
- `taic/kernel_catalog.py` + `orchestrator._archive_if_confirmed`：持久化 transport/manifest/track；
- `experiments/common.py`（v2 读盘）、`taic/policy.py`（追加原因码）；
- 新增 `scripts/msl_bundle_pack.py`、新题 `tasks/<op>/`、新测试。

不允许：R5 比较字段集合、`CONFIRMATION_KEYS`、`verify_subject_document`、envelope 格式、
六方 RPC/字段号、portfolio/track 平面语义、任何 per-task tiling/registry。

### 2.6 发布与归档的身份绑定（实施阶段 = B4b）

```python
raw = path.read_bytes()                       # 必须用原始字节，不能用 read_text()
transport_sha = sha256(raw)
manifest_sha  = decode_envelope(raw.decode("utf-8")).candidate_sha256 if 完整 envelope else transport_sha
```

0. **读 `result["track"]` 的判定表（先在 CPU 上做，不碰 GPU；无歧义，按行取唯一结论）**：

   | `result["track"]` 的状态 | best/winner 的形态 | 结论 |
   |---|---|---|
   | **缺失** | 非 envelope 文本（旧 v2 单文件记录） | **legacy**：按 `msl-single`/`single_source` 处理，与今天逐字段相同（今天 `_v2_conclusion_verdict` 根本不读 `track`） |
   | **缺失** | envelope | `track_mismatch`（无法证明这是 bundle 运行） |
   | **存在且 well-formed，但 `registry.get(track_id, version=track_version)` 失败**（未知 id / 版本不符） | 任意 | `track_mismatch`——`TrackRegistry.get`（`tcenv/track/registry.py:229-246`）对这两种情况抛 `UnsupportedContractCapability`（`tcenv/contract_errors.py:29`，`TaskContractError` 的子类），**不是** `KeyError`；必须用窄捕获 `except UnsupportedContractCapability` 接住后返回 `(未确认, track_mismatch)`，不得让 `conclusion_of_run` 抛 |
   | **存在且 well-formed，`registry.get` 成功** | 与 `delivery` 一致（`single_source`↔非 envelope、`candidate_bundle`↔envelope），且 `delivery == spec.delivery`、`track_version == spec.track_version` | 按该 track 判定 |
   | **存在但畸形**（非 dict、缺 `track_id`/`track_version`/`delivery`、类型错） | 任意 | `track_mismatch` |
   | **存在但自相矛盾**（`track_id=msl-bundle` + `delivery=single_source` 等） | 任意 | `track_mismatch` |
   | **存在但与 best/winner 形态不符**（`single_source` + envelope best，或反之） | — | `track_mismatch` |

   所有 `track_mismatch` 都是 `(未确认, track_mismatch)`，**不抛异常**；兼容承诺**只覆盖两行**——
   "缺失 + 非 envelope best"（旧单文件记录）与"存在且 well-formed、`registry.get` 成功、形态一致"
   （当前 single-track 记录），不是"所有既有单文件判定不变"这种全称说法。
   **查表顺序就是上表自上而下**：先看 `track` 是否存在，再做 registry lookup（窄捕获
   `UnsupportedContractCapability`），最后才比 `delivery` 与 best/winner 形态。

1. `_v2_conclusion_verdict` 的绑定：
   - `sidecar["candidate_sha256"] == manifest_sha(best 原始字节)`；
   - `result["confirmation_source_sha256"] == transport_sha(best) == transport_sha(winner)`；
   - **track 绑定（只用 envelope 里真实存在的字段）**：从 `result["track"]` 经 registry 得到期望
     `delivery` 与该 registry 当前 `track_version`（lookup 失败按上表：窄捕获 →
     `(未确认, track_mismatch)`）；`candidate_bundle` 时要求 best/winner 都是**完整
     envelope** 且 `decode_envelope(...).track_id == result["track"]["track_id"]`；`single_source` 时要求
     **不是** envelope。`track_version` 只做「`result["track"]["track_version"]` == registry 当前版本」的
     交叉校（envelope 没有这个字段，也不新增）。不匹配 → `track_mismatch`；
   - **读盘的异常闭集**：`raw.decode("utf-8")` 的 `UnicodeDecodeError`、`decode_envelope` 的
     `EnvelopeError`、以及从 entry 构造泄漏的 `TaskContractError`（非 ASCII manifest、未知 role、空 body、
     非法路径）全部在**同一个窄 `try`** 里捕获 → `sha_mismatch`；它连同「两种 transport 不等」与
     「sidecar 与 decode 不等」一起，保证 `conclusion_of_run` 在读坏文件时**不抛异常**、只给
     `未确认 / sha_mismatch`；
   - 非 envelope 文本时 `manifest_sha == transport_sha`，三条退化为今天的三方相等 →
     **既有运行判定逐字节不变**（含 CRLF 旧文件，见 B4b 回归）。
2. `taic/policy.py:791-798` 的 `PUBLICATION_REASONS` 追加 `"track_mismatch"`（只追加；既有码与既有运行
   的原因码都不变）。若实施时发现 observe 侧有闭集映射，同一提交里补上。
3. catalog（`kernel_catalog.archive_confirmed` + `_archive_if_confirmed`）：`meta.json`/`history.jsonl` 增加
   `transport_sha256`、`candidate_sha256`（manifest，可为空）、`track`（id/version/delivery）；
   **blob 仍按 transport 寻址**（不改索引键，旧 blob 可复用）。
4. `_prior_kernel_text`/`_search`/`_from_text`（§1.7）：
   - **先引入统一的 prior 来源显示名/路径**：新增一个 helper（例如 `self._prior_source_label` /
     `self._prior_source_path`）返回 `from_run / best_name`（--from）或 `catalog_dir(...) / best_name`（@catalog），
     并把 `orchestrator.py:2368,2373` 的 `self.from_run / self.best_name` **两处**都换掉——否则门放宽后
     `from_run is None` 会直接 `TypeError`；`self.from_run` 本身的语义不变（仍是「用户给了运行目录」）；
   - `_search` 的门改成 `self.from_run is not None or self.from_catalog`；
   - 取到的文本一律先过**与提交路径同一个** delivery-shape/preflight 判定（`preflight_submission()`），
     **不要**再写一个 `startswith("taic-envelope 1")`：magic 前有空白/多行/截断头时那种开头判断会漏掉
     损坏的 envelope；判定的结果只有两种——合法的 bundle envelope（且 `candidate.track_id ==
     self.binding.track_id`）或合法的单文件文本；其余一律 typed `RunStartError`（不静默当作「没有内核」，
     也不把残缺 envelope 交给评测）；
   - `_from_text` 与 dup 豁免对 `from_run` 与 `from_catalog` **都**设置（`:2410,2435-2437`），
     否则 catalog 种子会被自己的 `_external_kernel_texts()` 判成 `external_duplicate`；
   - 读盘用 `read_bytes()` + `decode("utf-8")`（失败 → typed 错误），保证 transport 身份与文件字节一致。

---

## 3. 阶段

### B0 交付形态、包装、预检（含 §1.4/§1.5 修复与源头规范化）

改动：

1. `track_adapter.py:198-199` 补 `key=lambda entry: entry.path`。
2. `envelope.py:180` 非 ASCII → `EnvelopeError`；`HostTilingPlan.from_candidate` 的 `int(...)` →
   `TaskContractError`（§1.2/§1.6）；两者各加一条负例。
3. **打包脚本**：`scripts/msl_bundle_pack.py`
   - 命令：`uv run --directory <repo> python scripts/msl_bundle_pack.py --src <dir> --out <submission>`；
   - `--src` 固定布局与 role/media_type 映射：`device/**/*.metal` → `device_kernel`/`text/plain`
     （**允许 ≥1 个，多个是合法多源**）；`host/tiling.json` → `host_tiling`/`application/json`；
     `host/execution_geometry.json` → `schedule`/`application/json`；
   - 缺 `host/tiling.json`、缺几何、`device/` 为空、出现未映射路径（例如 `host/notes.md`）→ 非零退出 +
     一行原因（作者错误，不是候选判负）；**不做「重复 role」检查**（多 device 合法）；
   - 唯一生产者是 `encode_envelope`；脚本不得自己拼 manifest/base64；
   - 输出**原子替换** `--out`（同目录临时文件 + `os.replace`），因为 `--out` 就是 `SessionBrief` 指定的
     交卷文件；总体积受 `MAX_ENVELOPE_BYTES = 1_000_000` 限制，超限由 `encode_envelope` 报错并原样转出。
4. `search_candidate.portfolio_for_submission()` / `preflight_submission()`；`CandidateDeliveryError`
   与两处 decode 调用点窄转（§2.3）；`orchestrator._evaluate_report` 窄捕；`cmd_eval_v2` 分类；
   `v2_evaluation._evaluate` 走预检。
5. `collect_session` 的 bundle 读取分支 + `classify_result` 的 `candidate_delivery` 分支 +
   `tcenv/evaluate.py::rejected_report(...)` 纯工厂 + `Orchestrator._record_delivery_rejection(...)`
   的无 GPU 落盘路径（§2.5）。这些代码在 B0 落地，但**真实 `_one_session_attempt` 的端到端验收在 B2**
   （它要 `SessionBrief.track_id` / `Orchestrator(track_id=)`）；B0 只做单元级验收。

验收（CPU、不 build；**Metal 相关全部放 `tests/backends/metal/`**）：

- `tests/backends/metal/test_search_v2_portfolio.py`：bundle envelope → `track_id=="msl-bundle"`、父包几何
  role/media_type、planner 与父包 manifest（**不含 `protocol_epoch`**，它在 `PortfolioFreeze`，归 B4）；
  单文件 → 父包逐字节不变（对既有常量断言 `portfolio_digest`）。
- `tests/backends/metal/test_search_v2_bundle_preflight.py`（新）：§2.2 九条负例 +
  §1.4 两项（两个 device 各一个 launch 块；launch workspace 字节 ≠ ABI span）+
  §1.6 四类（未知 role、空 body、非 ASCII manifest、`schema_version: null` 的 host plan），
  全部断言 typed rejection（`stage` 明确），无一抛异常。
- `tests/backends/metal/test_track_bundle_compose.py`（新或并入 a4）：§1.5 反序双 device 拼接回归。
- `tests/backends/metal/test_msl_bundle_pack.py`（新）：单 device 与**双 device** 各一条正例（role/media_type
  逐项断言）；缺 `host/tiling.json`、缺几何、`device/` 为空、未映射路径各一条非零退出；`--out` 已存在时被
  原子替换；产物与 `encode_envelope` 逐字节相同；输出含明文 body。
- `tests/shared/test_search_v2_orchestrator.py`：非法 envelope 候选 → 运行继续、报告可读、**无 outbox 行、
  无 GPU 调用**、`result.json` 有 rejected 记录；其中必须有四条**贯穿 orchestrator 的端到端负例**：
  未知 role、空 body、**bundle 会话提交超过 1,000,000 字节的单文件**、**超过 1,000,000 字节的残缺 envelope**
  （后两条走 `EvaluationRequest` 的超限路径，必须同样是 rejected 而非基础设施故障）；
- **异常源头负例（全部 CPU、无 GPU、无 build）**：深嵌套 envelope（`RecursionError` 源）、
  **深嵌套的 Host Tiling `tiling.json`**（→ `TaskContractError`）、**深嵌套的 geometry entry**
  （→ `GeometryError`）、
  超长整数 host tiling（`json.loads` 的 `ValueError` 源）、`launch` 里的超长十进制串、
  plain source 含 lone surrogate（`UnicodeEncodeError` 源）、`cmd_eval_v2` 读到非法 UTF-8 文件
  —— 每条都必须给 typed rejected / 退出码 2，且都**不得** traceback；
- **交卷链路的字节保真（B0 只做单元级；真实 `Orchestrator` 端到端在 B2）**：
  B0（两个对象都自己造，**不需要** `Orchestrator(track_id=)`——那是 B2 的接线）：
  - **binding**：`replace(bind_task(...), track_id="msl-bundle")`（preflight/`portfolio_for_submission`
    要它）；
  - **brief**：`dataclasses.replace(brief, track_id="msl-bundle")`（`collect_session` 只拿得到 brief，
    字段在 B0 新增，见 §2.5）；
  - 正例——`encode_envelope()` 造一份 **device entry 含 CRLF** 的 envelope，写盘后走
    `collect_session` → `portfolio_for_submission`/`preflight_submission` → `decode_envelope`：必须通过，
    且 `decode` 出的每个 entry 的**字节**、`transport_sha256`（文件原始字节）与 `candidate_sha256`（manifest）
    **与写盘前完全相同**（今天会在 `read_text` 的 CRLF 归一后炸成
    `a taic-file body is not followed by its end marker`）；
  - `collect_session` 的返回语义（**两个费用用例，不能只测一个**）：
    已知费用（有 token 计数）+ 非法 UTF-8 → `status=invocation.status`、`cost_incomplete is False`、
    `diagnostics.failure_class == "candidate_delivery"`；**未知费用**（`invocation.saw_tokens is False`）
    + 非法 UTF-8 → `cost_incomplete is True` 且 `cost_usd is None`/原值，**不得**被改写成 0 或 False；
    另加：`status="timeout"`/"failed" 的会话即使字节非法，也**不得**出现 `failure_class ==
    "candidate_delivery"`（保留 timeout/process_failed）；
  - single 回归——单文件交卷（含 CRLF 与非法 UTF-8）仍走今天的 `read_text(errors="replace")` +
    `kernel_text()`，行为逐字节不变；
  - **依赖顺序**：B0 自己就能满足前提——`SessionBrief.track_id`（默认 `"msl-single"`）的**字段新增在
    B0**（见 §2.5），测试用 `dataclasses.replace(brief, track_id="msl-bundle")` 就能驱动 collect 分支；
    `Orchestrator(track_id=)` 与"由真实编排器填这个字段"仍在 B2，所以**贯穿
    `Orchestrator._one_session_attempt` 的端到端负例放在 B2**（见 B2 验收），B0 不产生循环依赖。
- **Host Tiling span 负例（预检阶段，不 build）**：`count < 0`、`stride = 0`、
  `offset = -1`（负首地址）、`offset` 越出 ABI span 四种 plan → 预检给 `stage="delivery"` 的缺陷，
  **不是**等到执行时的 `GeometryError`；保留 `stride = 0` 单独一条（`count = 0` 时也**不**放行，
  与 `v2_runtime.py:199-201` 一致）。**正例**：`count = 0, stride = 1, offset = 0`（等价地
  `count = 0, stride = 2, offset = 0`）必须被接受（span 当 0），证明抽出的 helper 没有把空写误拒；
  反例明确不含「`count=0` + `stride=0`」。

### B1 投影与 lifecycle 复用证明（只加测试）

验收（CPU）：`projection.track_id=="msl-bundle"`、`HostTilingPlan.from_candidate` 可解析、
`_lifecycle_factory` 步骤 `("host_tiling","schedule")`；`inside_timing` 两条（无 `zero` → `False`；
`{"name":"scratch","zero":true}` → `True`）；三处一致性负例（`stage_binding.py:203`、`geometry.py:519-523`、
`v2_runtime.py:150-160`）；权威性用例（源里 grid/threads 与几何不一致仍按几何执行）。

### B2 track 接线与记录（验收断言运行对象）

按 §2.1 完成 `Orchestrator(track_id=)`、`cmd_run`、`prepare_resume`、run_config/result、seed 策略，
`SessionBrief.track_id` 的**填值**（字段本身在 B0 新增）+ `kernel_agent._check_command()` 的 `--track`
注入，**以及 `_failure_record`
（`orchestrator.py:2069`）那条 differential-recheck 人工复测命令的 `--track` 注入**（§2.1 最后两行与
「人工复测命令」行）。

验收：

- 三对象断言：`orch.binding.track_id`、`orch.evaluator.binding.track_id`、`SessionBrief.track_id`
  （三者都在 `orchestrator.py:693`/`:743`/`:3249-3267` 的局部 `binding` 上重建，只改 `self.binding`
  时会静默落回 `msl-single`）；
  **prompt 内容的断言不在这里**（bundle 段、冲突句替换、workspace ABI 与签名骨架都在 B3）：
  B2 只断言"渲染函数确实收到了这个 track"——即 `prompts.system_prompt_v2` /
  `kernel_agent.render_contract` 被调用时拿到的 `binding.track_id`/`brief.track_id` 是 `msl-bundle`
  （用 spy/包装断言入参，不断言文本内容）；文本断言见 B3 验收①–⑨；
- 优先级：CLI > TOML > 默认；`--track` 省略时 `args.track is None`；
- 启动拒绝：`msl-bundle` + 无 workspace 题 → `RunStartError`；
- resume：parser 层接受 `--track`；`prepare_resume` 不带 `--track` 时仍是 bundle（不是落回 single）；
  带不同值 → `ResumeRefused`；记录的 `track_version` 与 registry 不符 → `ResumeRefused`；
  **旧配置兼容**：一份 schema<3、没有 `track` 块的 v2 目录能 resume 并按 `msl-single` 运行；
  v1 目录 resume 后 `track` 仍为 `None`；`track` 存在但未知 id / 版本不符 → `ResumeRefused`；
- 自测命令：bundle 会话的 TASK.md 检查命令含 `--track msl-bundle`，single 会话命令逐字符不变；
- **differential-recheck 的人工复测命令**：断言对象是**真实的 `_failure_record(...)` 返回值**，字段名是
  **`retest_command`**（`_RECHECK_KEYS`，`orchestrator.py:144`；写入处 `:2086`）——不是 `record["command"]`。
  bundle 运行的 `record["retest_command"]` 含 `--track msl-bundle`，且把它原样喂给 CLI 时对 envelope
  文件**可执行**（不再按 single 拒绝）；single 运行的同一字段与今天**逐字符相等**；
- `tcenv eval-v2` 自身：parser 接受 `--track`；`--track msl-bundle` 时重建的 binding 与传给
  `evaluate_candidate(track_id=)` 的值都是 `msl-bundle`（bundle 正例能评过 preflight）；不带 `--track` 时
  行为与今天逐字符相同；
- v1：显式 `--track msl-bundle` → 启动期 `RunStartError`（不是静默 v1 single）；
- 注入执行器：`V2LocalEvaluator` 的 `binding.track_id` 与 `orch.binding.track_id` 不一致 → `RunStartError`；
- bridge：bundle 会话交单文件 → **`ok=true` + `report.status="rejected"` + `stage="delivery"` + 无 GPU**
  （§2.4，不是 `invalid_request`）；
- seed：首场 bundle 会话 `seedless=True` 且 seed 文件不存在；后续会话 `seed_source` 是 envelope 原文、
  文件仍叫 `kernel.metal` 且能被 `decode_envelope` 解开；
- **交卷读取（G21）在真实会话路径上端到端**（**归 B2**，因为要用 `SessionBrief.track_id` 与
  `Orchestrator(track_id=)`；断言对象是 `_one_session_attempt` 的结果，不只是 `collect_session` 的返回值）：
  正例——`encode_envelope` 写一份 **device entry 含 CRLF** 的 envelope，假 runner 让会话"交出"这个文件 →
  `_one_session_attempt` 走到评测：`decode_envelope` 通过，且 entry 字节、`transport_sha256`、
  `candidate_sha256` 与写盘前一致；
  **负例 A（已知费用）**——同一份文件注入一个非法 UTF-8 字节，且 runner 报了 token 计数 → 该场最终是
  `status="rejected"` + `stage="delivery"`，**无 GPU、无 outbox**，`ledger` 入账本场 `cost_usd`
  且 `cost_incomplete is False`；`self.candidates[-1]` 存在（`source == ""`、报告 `stage="delivery"`、
  `cost_usd` 已记）、`trace` 有 `status="rejected"` 行、`self._failure_feedback` 非空；
  事件不是 `agent_failed`，也不抛 `StopRun`；
  **负例 B（未知费用）**——同一份非法字节文件，但 runner **没有** token 计数
  （`invocation.saw_tokens is False`）→ 仍然抛 `StopRun("cost_incomplete: ...")`，
  **不得**伪造 `cost_incomplete=False` / 把费用按 0 入账（这是费用完整性的底线）；
  **落盘 + resume（新增，覆盖 §2.5 的占位文件选择）**——跑完负例 A 后：`candidates/NNN_<label>/`
  里存在 `<kernel_name>`（**空串占位文件**）与 `report.json`；用 `taic resume` 读该目录时
  `_read_candidates` 能恢复这条 rejected 记录（不会被"缺内核文件"跳过），
  **且**它既不是 `best`、也不在 `self.cache` 里（`_sha("")` 不得成为可信缓存；这条正是
  `_cache_this("")` 今天返回 `True` 会踩的坑）；若这是唯一候选，resume 也不得以
  `no candidates/ were stored` 拒绝（`resume.py:867-870`）；
  single——同一份文件走 single 分支（`read_text(errors="replace")` + `kernel_text()`）时行为与今天
  逐字节相同。

### B3 题面与 agent 引导（含必做重签与受信面登记）

- **改正三处实际模型可见文本**：
  1. `prompts.py:73-74`（`HARNESS_RULES` 的 "Every workspace is filled with zeros inside the timed command
     buffer" 句）与 `kernel_agent.py:662`：v1 `backend.py:633` 与 v2 `v2_runtime.py:641` 都**只清声明
     `zero` 的 workspace**，而正确性对每个 workspace 都填；原文对两个引擎都错 → **允许改这句常量正文**
     （它与 track 无关，两个引擎、所有 track 都错）。因此 single 的"逐字节不变"承诺**收窄为**：
     「除这一句纠错外与今天逐字节相等」（见下面的验收⑧，用
     `single_text.replace(NEW_SENTENCE, OLD_SENTENCE) == TODAY_TEXT` 这种可执行判定）；
  2. `prompts.py:170` 的 "workspace tensors are not function parameters"：在 bundle 里 workspace
     **是**按名字绑定的 kernel 参数（见下条），这句必须改掉，否则 agent 会把 workspace 从签名里删掉、
     直接被 `stage_binding.py:181-197` 拒。
  3. **真正生成「Required signature skeleton」的函数也必须 track-aware**：
     `TaskBinding.signature_stub()`（`taic/task_binding.py:123-149`）现在**显式跳过所有 workspace**
     （`if tensor["space"] == "workspace": continue`），而它同时进入系统提示（`prompts.py:176`）和
     TASK.md 的 `SessionBrief.abi_stub`（`orchestrator.py:3234`）。只补一段完整示例没用：这两个标着
     "签名骨架" 的代码块会继续引导 agent 漏掉 workspace 参数。改法：**方法签名不变**（仍是无参
     `signature_stub()`），方法内读 `self.track_id`（`TaskBinding.track_id`，`task_binding.py:38`；
     `presentation()` 已按同一字段选 track，`:110`），在 `delivery == "candidate_bundle"` 时**保留**
     workspace 槽位（`device float* scale_tab [[buffer(2)]]`，可写性与 `access` 一致），single 时输出
     **逐字节不变**。两个现有调用点（`prompts.py:176`、`orchestrator.py:3234`）保持无参调用，
     **不要**新增 `track_id=` 参数：那会多出一条可能与 `binding.track_id` 分裂的入参。
     它也必须进受信面，但**不能按模块函数登记**：`signature_stub` 是 `TaskBinding` 的**方法**
     （`taic/task_binding.py:123`），模块里没有同名符号；`_symbol_bytes`（`taic/trust.py:155-172`）直接
     `getattr(module, symbol)`，照写会在 `prompt_fingerprint_parts()` 里抛 `AttributeError`，把
     `current_prompt_sha1()` 一起带崩。确切机制（写进 B3）：
     1. `_symbol_bytes` 支持**点分属性路径**：`label="task_binding.TaskBinding.signature_stub"`、
        第二项仍是模块名 `taic.task_binding`、第三项是 `"TaskBinding.signature_stub"`；
        解析时先 `getattr(module, "TaskBinding")`，再 `getattr(cls, "signature_stub")`，对得到的函数
        `inspect.getsource(...)`（方法源码可被 `inspect` 取到），**只哈希该方法源码**；
     2. `_PROMPT_FUNCTIONS` 里追加 `("task_binding.TaskBinding.signature_stub", "taic.task_binding",
        "TaskBinding.signature_stub")`；
     3. `taic/trust.py` 的这项 helper 改动进 §2.5 白名单（它只影响解析，不改其它符号的字节）；
     4. 兜底：点分解析失败必须**显式报错**（`TypeError`/`ValueError` 带标签），不能静默返回空字节
        ——否则一个拼错的标签会让指纹变弱而没人发现。
- **共享文案必须按 track 条件化，并逐句给出保留边界**（只加一个 bundle 段不够：旧句会和新段互相矛盾）：
  | 现文案 | 位置 | bundle 下的问题 | 处理 |
  |---|---|---|---|
  | "With one `/* taic-launch */` block, that block is the only launch description: several `kernel` functions, each with its own grid, threads, **optional dynamic threadgroup bytes** and optional workspace buffers" | `prompts.py:57-60`（`HARNESS_RULES`） | ① grid/threads 在 bundle 里由 geometry 决定；② 动态 threadgroup 参数本期**不支持** | bundle 渲染时**删除并替换**为 "`host/execution_geometry.json` 决定 dispatch；`threadgroup_bytes` 必须是 0；`[[threadgroup(k)]]` 函数参数本期不支持" |
  | "With the block, each stage's grid and threads come from the block" | `prompts.py:62-63` | 同上 | 同上（替换为 "each stage's grid and threads come from the geometry entry"） |
  | "Return the complete source text of everything you define." / "exactly one Metal code block" | `prompts.py:71-74` | bundle 交的是 **envelope**（多文件、打包），不是单个代码块 | 替换为 "package the bundle and submit the envelope; do not paste one code block" |
  | "Your candidate MUST carry exactly one `taic-launch` block (grid/threads, optional threadgroup bytes...)" | `prompts.py:167-171`（`system_prompt_v2`） | 把 grid/threads 完全归给 launch 块 | 替换为 "the single `taic-launch` block supplies `math` and the launch↔ABI workspace cross-check only" |
  | "必须自带且只能带一个 `/* taic-launch */` 块：**外层不会替你选 grid/threads**。写成 `launch NAME grid=16,1,1 threads=128,1,1`" | `kernel_agent.py:672-674`（`render_contract` 的 v2 分支，**TASK.md**） | 这句直接把作者引向 launch 块选 grid/threads，而 bundle 段说 dispatch 以 geometry 为准——两处**在同一个 TASK.md 里** | bundle 分支替换为「dispatch 由 `host/execution_geometry.json` 决定（grid/threads/stage 顺序/workspace 名单）；`taic-launch` 块仍必须恰好一个，但它只承担 `math` 与 workspace↔ABI 交叉检查」；single/v1 分支除清零句外保持原文 |

  **实现方式（唯一方案，不是"追加 ignore earlier"，也**不**拆模块级常量）**：
  - `HARNESS_RULES` **整体原文保留**，只做上面第 1 条要求的**清零句纠错**；不把它拆成若干具名常量
    （拆成模块级常量会让新文本落到未登记的符号上，见下条）。
    `_rules_for(backend, delivery)`（`prompts.py:30-44`）按 `delivery` 渲染：
    `single_source` → 今天的正文（**只**含清零纠错，无任何 track 分支）；
    `candidate_bundle` → **在函数体内**对上一步渲染出的文本做"旧句 → 函数体内局部新句"的替换
    （上表右列那几句），再把结果放进 `system_prompt_v2`。也就是说：**旧句常量不动，新句是
    `_rules_for` 的局部量**，替换是运行期字符串操作，不是"把常量拆开"。
  - 之所以必须这样：`system_prompt_v2` 开头就是 `_rules_for(backend)`（`prompts.py:157`），
    只追加"ignore earlier"的话，旧句**必然仍在最终 prompt 里**，而验收⑧又禁止它们出现——那种写法
    没有任何实现能满足。所以冲突句必须**在渲染阶段被删掉**，不能靠后加句子压制。
  - `HARNESS_RULES` 的**清零纠错**属于允许修改的常量内容（与 track 无关、两个引擎都错）；
    四条 bundle 冲突句**不删除常量原文**，只在 bundle 渲染路径被替换。
  - **每条替换文本必须落在"已被哈希的源码"里**（否则改它不会让 `current_prompt_sha1()` 变，旧授权
    会静默继续有效）。`trust.py:154-172` 只哈希 `_PROMPT_FUNCTIONS`/`_PROMPT_CONSTANTS` 里登记的符号，
    所以本期**采用的方式**是：四条替换句与 geometry/workspace 指引**全部作为 `_rules_for` 函数体内的
    具名局部量**（`inspect.getsource(_rules_for)` 会把函数体一起哈希），bundle 专属段落全部写在
    `system_prompt_v2` 函数体内（它已在 B3 登记进 `_PROMPT_FUNCTIONS`）。
    **不允许**为了"好读"把这些文本提到模块级新常量：一旦提到模块级又不登记，就绕过了指纹。
    若将来确实需要模块级常量，必须**同时**把每个常量加进 `_PROMPT_CONSTANTS`（`trust.py:90-95`），
    并在下面 B3 的判别力验收里逐个覆盖。
  - 判定标准（可执行）：把 `prompt_fingerprint_parts()` 的所有字节拼起来，**每一条**替换句都必须能在
    这个 blob 里找到（说明它确实被哈希）；详见 B3 验收的"判别力"部分。
  - **`render_contract` 也要按 track 分支**（它渲染 TASK.md，是 agent 的第二个渲染面）：
    `kernel_agent.py:672-674` 的 v2 中文句在 `brief.track_id == "msl-bundle"` 时替换成上表最后一行右列；
    其余（v1 分支、single 的 v2 分支）除清零句外保持原文。判定用 `brief.track_id`（B2 已经把它填进
    `SessionBrief`），**不要**再读别的地方。
  - 受信面：`HARNESS_RULES`（`_PROMPT_CONSTANTS`，`taic/trust.py:91-95`）、`system_prompt`、`_rules_for`、
    `render_contract`（`kernel_agent.render_contract`）本来就都在指纹里；**`system_prompt_v2` 与
    `_check_command` 不在**（`_PROMPT_FUNCTIONS`，`trust.py:77-88` 里没有它们），和
    `signature_stub` 一样是**新增登记项**。
- **题面必须给出可照写的 `host/execution_geometry.json` 示例与字段语义**（这是 agent 唯一失败率最高的文件：
  它是闭集，多一个字段就拒）。要写清：
  - 顶层**恰好**三个键：`schema_version`（当前 `1`，`plane.py:38`）、`source`（**必须是 `"candidate"`**）、
    `stages`（非空有序列表）；
  - 每个 stage 的字段：必填 `kernel`/`grid`/`threads`；可选 `threadgroup_bytes`（整数，默认 0）、
    `workspace`（名字列表或 `{"name":..., "zero":true}`）、`when`（激活表达式）、`derived_scalars`；
    `grid`/`threads` 各三个整数项，可用合同符号的整除表达式；
  - **多阶段 = 有序 stage 列表**，执行顺序就是列表顺序，`when` 决定某 stage 在该实例上是否激活；
  - **动态 threadgroup 参数本期不支持**（必须如实写进题面）：`threadgroup T name[N] [[threadgroup(k)]]`
    形式的**函数参数**会被 `stage_binding.py:199-204` 直接拒（"dynamic threadgroup parameter is not part
    of the ABI"）；**静态** `threadgroup T name[N];` 声明在 kernel 体内是允许的。`threadgroup_bytes`
    字段保留（闭集需要），但本期必须写 **0**；非零数字**不是**动态内存证据（没有 `threadgroup` binding 时
    根本没有分配，`runtime.py:638-640`），源里 `taic-launch` 的 `threadgroup=` 也不参与 bundle 的 dispatch。
  - 示例（可直接改成作者的 kernel 名）：
    ```json
    {
      "schema_version": 1,
      "source": "candidate",
      "stages": [
        {"kernel": "affine_ramp", "grid": ["N", 1, 1], "threads": [1, 1, 1],
         "threadgroup_bytes": 0, "workspace": ["scale_tab"]}
      ]
    }
    ```
    多阶段示例只需在同级再加一个对象（例如 `{"kernel": "affine_ramp_tail", "grid": ["N", 1, 1],
    "threads": [32, 1, 1], "workspace": ["scale_tab"]}`），并在题面说明它是第二个 stage。
- **必须改正一处会误导的既有提示并给出 workspace 的真实 MSL ABI**：`prompts.py:170` 现在写
  "workspace tensors are not function parameters"——**在 bundle 里是错的**：读表的 kernel 必须按名字
  绑定 workspace，例如 `device float* scale_tab [[buffer(2)]]`。规则是：
  - buffer 槽位顺序 = ABI 声明顺序（含 workspace），kernel 参数名必须与 ABI 张量名相同；
  - `stage_binding.py:181-197` 要求该 workspace 参数必须被**本 stage** 的 `workspace` 计划声明；
  - 可写性必须与 `access` 一致：本题 `scale_tab.access = "write"`，写成 `device const float*` 会被
    `stage_binding.py:248-255` 拒（"output ... is read-only in the kernel; it must be writable"）。
  - known-good 的完整形状（fp16 档）：
    ```metal
    #include <metal_stdlib>
    using namespace metal;
    kernel void affine_ramp(device const half* x   [[buffer(0)]],
                            device half* y         [[buffer(1)]],
                            device float* scale_tab [[buffer(2)]],   // access: write → 非 const
                            uint i [[thread_position_in_grid]]) {
      y[i] = half(scale_tab[i / SEG] * float(x[i]));   // SEG 来自 attrs 注入的头常量
    }
    /* taic-launch
    math safe
    workspace scale_tab NS*4
    launch affine_ramp grid=N,1,1 threads=1,1,1 workspace=scale_tab
    */
    ```
    对应 `host/execution_geometry.json`：`{"schema_version":1,"source":"candidate","stages":[
    {"kernel":"affine_ramp","grid":["N",1,1],"threads":[1,1,1],"threadgroup_bytes":0,
    "workspace":["scale_tab"]}]}`（launch 块的 grid/threads 不要求等于 geometry，见下）。
  - **bf16**：`half` 与 `bfloat` 是不同类型，不能复用同一签名。若声称 bf16 的 GPU 覆盖，必须再加一个
    `device const bfloat* x` / `device bfloat* y` 的变体并跑同一数值断言；只跑 fp16 自测就不要在
    证据里写 bf16 GPU 覆盖（bf16 仍由 CPU 的 `instantiate_v2`/reference 测试覆盖）。
- **必须解释「双份声明」的真实分工**（否则作者会以为两者要相等）：
  - **实际 dispatch 以 geometry entry 为准**：`grid`/`threads`/`threadgroup_bytes`/stage 顺序/workspace 名单
    都来自 `host/execution_geometry.json`；
  - device 源里的**唯一** `/* taic-launch */` 块仍然必需，它负责两件事：`math fast|safe`（`launch_math`），
    以及 launch↔ABI 的 workspace 交叉检查（`workspace_size_problems`：块里声明的每个 workspace 必须在 ABI 里、
    且字节数与 ABI span 相同）；
  - 所以**不要**要求源里的 `grid`/`threads` 与 geometry 相等——B1 的权威性用例正是「两者不一致时仍按
    geometry 执行」；源里写错 workspace 名单/字节才是硬错。
- **agent 端到端清单**（TASK.md 与 `taic/flows/kernel_once/skills/taic-tools/SKILL.md` 各一份）：
  1) 建 `device/`、`host/`（路径与 role 映射见 B0）；2) 跑 TASK.md 里那条打包命令直接覆盖交卷文件；
  3) 用 TASK.md 里的 `tcenv eval-v2 ... --track msl-bundle`（已带参数）看 correctness/`dispatch`/`host_runs`；
  4) 不手写 base64/manifest、不建第二个 `taic-launch` 块、不 `#include "x"`。
- **受信面登记**：把 `system_prompt_v2`、`kernel_agent._check_command` 与
  `task_binding.TaskBinding.signature_stub`（三处都是模型可见文本）加进 `taic/trust.py:77-88` 的
  `_PROMPT_FUNCTIONS`；**注意 `signature_stub` 是方法不是模块符号**，所以同时要让 `_symbol_bytes`
  支持点分属性路径（机制见上一条第 3 项），否则 `prompt_fingerprint_parts()` 会 `AttributeError`。B3 交付清单包含重签
  `docs/架构/backend-phase0-baseline.json`（`prompt_sha1`、`system_prompt_sha256` 等
  `test_backend_phase0.py:55-57` 检查的摘要全部更新），旧授权按设计失效。
- **remote workspace**：`workspace_prompt.py` 明确不打印控制机路径命令（`:23`），远端会话拿不到打包脚本；
  因此本阶段对「bundle + remote workspace」在**启动期 typed 拒绝**（`RunStartError`，说明需要本地工作区），
  远端支持列为后续扩展，不在本期渲染半可执行的指引。

验收：`tests/shared/test_search_v2_prompt.py` 至少九条：
① bundle 会话含 bundle 段与打包/自测命令；② single 会话不含；
③ bundle 会话含 **geometry 示例全文**（三个顶层键 + stage 必填/可选字段名 + `"source": "candidate"`）；
④ bundle 会话含 **多阶段有序**、**「动态 threadgroup 参数不支持、静态 `threadgroup T name[N]` 可用、
`threadgroup_bytes` 本期写 0」**、**dispatch 以 geometry 为准而 `taic-launch` 只负责 math 与 workspace
交叉检查** 这三句规则（并显式**不含**「源里 grid/threads 必须等于 geometry」「动态 threadgroup 由
geometry 驱动」这两类会引导 agent 写出必拒 kernel 的说法）；⑤ bundle 会话含 **workspace 的真实 MSL ABI**：按名字绑定 `[[buffer(n)]]`、
可写性必须匹配 `access`（`write` 不能写 `const`）、`stage_binding.py:181-197` 的「必须被本 stage 声明」
规则、以及 fp16 档的完整签名/launch 块/geometry 对照示例；⑥ 并显式断言 bundle 文案里**不再出现**旧句
"workspace tensors are not function parameters"；
⑦ **签名骨架（断言两个真实调用点）**：
  - **调用点 1（系统提示）**：`prompts.system_prompt_v2(orch.binding, use_tools=..., target_id=...)`
    的返回值里骨架块**含 workspace 参数**（名字、`[[buffer(k)]]` 槽位、`access` 对应的可写性）——
    走的就是 `prompts.py:176` 的无参调用；
  - **调用点 2（TASK.md）**：从**真实 `Orchestrator`** 取 brief —— `brief = orch._session_brief(
    seedless=True, label="init", work_dir=tmp)`（`orchestrator.py:3178`，它内部在 `:3234` 用
    `self.binding.signature_stub()` 填 `abi_stub`，并填 `track_id`），断言
    `brief.track_id == "msl-bundle"` **且** `brief.abi_stub` 含 workspace 参数；再把**这个真实的
    brief** 交给 `kernel_agent.render_contract(brief, ...)`，断言 TASK.md 里也有同一份骨架。
    **不要**用 `dataclasses.replace(brief, abi_stub=binding.signature_stub())` 手工重建输入——那样抓不到
    "`_session_brief` 仍从旧 binding 取 `abi_stub`"或"新字段 `track_id` 没接上"这两类回归；
    （手工 `replace` 可以留作 `render_contract` 的纯 renderer 单元测试，但不能叫真实调用点验收。）
  - **single 对照必须重新生成**：换一个 `track_id="msl-single"` 的 `Orchestrator`/binding，重新调
    `_session_brief(...)` 与 `system_prompt_v2(...)`，断言两处骨架都回到 single 形状（不含 workspace
    参数）；**不能**只改 `binding.track_id` 却继续复用旧 bundle 的 `abi_stub`（那会把"重新生成"这条
    路径也一起跳过）；
  - 两个调用点都以 `binding.track_id` 为准：断言 `signature_stub()` 的签名**没有新增参数**
    （`inspect.signature` 只有一个 `self`），证明没有第二条 track 来源；
  - single 会话的这两处，把唯一的清零句纠错还原后与今天逐字节相等；
⑧ **两个渲染面分别断言**（`system_prompt_v2(...)` 与 `kernel_agent.render_contract(...)` 的实际返回值）：
  - **bundle 的 `system_prompt_v2`**：四条英文冲突句**逐条不存在**（`assert "each stage's grid and threads
    come from the block" not in text` 等四条），且替换后的四条 bundle 句都在；即"渲染阶段删除"，不是
    "再追加 ignore earlier"；
  - **bundle 的 `render_contract`（TASK.md）**：**中文旧句**逐条不存在——至少断言
    `"外层不会替你选 grid/threads" not in task_md` 与
    `"必须自带且只能带一个 \`/* taic-launch */\` 块：外层不会替你选 grid/threads" not in task_md`，
    且替换句（"dispatch 由 `host/execution_geometry.json` 决定" + "`taic-launch` 只承担 `math` 与
    workspace↔ABI 交叉检查"）都在；**这一面不能只查英文句**（那四句英文本来就不在 TASK.md 里，
    只查英文会让这个渲染面空过）；
  - **single 的两个面**：`system_prompt_v2` 与 `render_contract` 各自
    `text.replace(NEW_ZERO_SENTENCE, OLD_ZERO_SENTENCE) == TODAY_TEXT`
    （把唯一的清零纠错还原后必须与今天逐字节相等；`TODAY_TEXT` 取 B3 记录下来的当前文本常量，
    prompt 与 TASK.md 各一份），并断言 `NEW_ZERO_SENTENCE in text`；single 的 TASK.md 必须仍**保留**
    中文旧句（它是 single 的正确指引）；
  - 两个面上都断言：不会同时出现 (a) "launch 块决定 grid/threads" 与 "dispatch 以 geometry 为准"、
    (b) "optional dynamic threadgroup bytes" 与 "动态 threadgroup 参数不支持"、
    (c) "exactly one Metal code block" 与 "交打包后的 envelope"。
§ **受信面验收（必须先证明"不崩"，再证明"有判别力"）**：
  1. 直接调用真实的 `taic.trust.prompt_fingerprint_parts()` 与 `current_prompt_sha1()` —— 两者都必须
     正常返回（今天若按模块函数登记会抛 `AttributeError`）；并断言返回的标签集合里含
     `task_binding.TaskBinding.signature_stub`，且它的字节**等于**
     `inspect.getsource(TaskBinding.signature_stub).encode()`（不是整个模块、不是空串）；
  2. 判别力（**可执行的写法**：改的是 `inspect.getsource` 的返回，而不是磁盘上的源文件）：
     `_symbol_bytes`（`trust.py:154-172`）是通过 `inspect.getsource(value)` 取文本的，所以
     "monkeypatch 改源码"或"替换函数对象"都不会改变磁盘文件、也证明不了指纹会变。做法：
     ```python
     real = inspect.getsource
     def fake(obj, *a, **k):
         text = real(obj, *a, **k)
         return text.replace("def signature_stub", "def signature_stub ", 1) \
             if getattr(obj, "__qualname__", "").endswith("TaskBinding.signature_stub") else text
     monkeypatch.setattr(inspect, "getsource", fake)          # _symbol_bytes 内部 `import inspect` 取的就是它
     assert current_prompt_sha1() != baseline_sha1            # 摘要必须变
     monkeypatch.undo(); assert current_prompt_sha1() == baseline_sha1   # 恢复后必须回到原值
     ```
     再做一个反向用例：同样的手法只改**无关**符号（`taic.task_binding.bind_task`）的返回文本，
     `current_prompt_sha1()` 必须**不变**。
     （也可以用临时模块文件 + `importlib.reload`，但不要写成含义不清的"monkeypatch 改源码"。）
  3. **新增提示文本必须逐个被覆盖（不能只测 `signature_stub`）**：
     - **覆盖判定**：把 `prompt_fingerprint_parts()` 的全部字节拼成一个 blob，断言**每一条**新增/修改的
       模型可见文本都能在这个 blob 里找到——四条 bundle 替换句、geometry 字段指引、workspace MSL ABI
       指引、清零纠错句、bundle 交付说明（envelope/打包命令/`--track`）。任何一条找不到，说明它跑到
       未登记的模块级常量里了，必须移回函数体或登记进 `_PROMPT_CONSTANTS`；
     - **判别力**：对 `_rules_for` 与 `system_prompt_v2` **各做一次**上面的 `inspect.getsource`
       monkeypatch（只改各自函数体里的一个字符）→ `current_prompt_sha1()` 必须变化，`undo()` 后回到原值；
       四个 bundle 替换句与 bundle 段都在这两个函数体内，所以这两次 monkeypatch 就覆盖了它们；
     - 若实现选择把某些片段提到模块级常量（本计划不推荐），则**每个**这样的常量都要各自做一次
       monkeypatch 判别力断言，且必须先被加进 `_PROMPT_CONSTANTS`。
  4. 反向：把标签拼错（`TaskBinding.no_such_method`）时必须**显式报错**，不能静默给出空字节。
  这四步都在同一个测试里，且**不是**"断言元组里出现了一个名字"。
`test_backend_phase0.py` 在新 baseline 下通过；`bundle + remote workspace` 启动被拒。

### B4 确认门（CPU + 需许可的 GPU）

改动：`confirmation_input` 与 `V2LocalEvaluator.confirmation_expected` 切到同一包装函数并接收 `track_id`。
不改 `PortfolioFreeze`/`confirm_subject`/R5（含 `BENCH_ATTEMPTS` 的重试循环：1 次 search + 最多 3 次
confirmation）。

验收：CPU——bundle freeze 可复算（**`protocol_epoch==2`**、planner id）、`confirmation_expected` 对 bundle
走 `candidate_geometry`；GPU（**需显式 GPU 许可**）——bundle 优胜者 `confirm_subject` → `confirmed`
（identity match、fresh seed ≠ 7、`credits_variant is True`）；
负例——改一个 device 字节 → manifest 变；文件顺序变化 → manifest 不变。无许可时只能标「CPU 部分完成」。

**「dispatch stage 序列一致」怎么断言（必须写清，否则这条读不到对象）**：
`confirm_subject` 返回的 sidecar（`portfolio_confirmation.py:267-296`）只存 `search_report`、
`confirmation_report`、identity、seed、`confirmation_attempts` 与 `credits_variant`；它**不保存**任何
`PortfolioRunResult.evaluation`，schema-4 report 里也没有 `dispatch_stages`。
**调用次数不是两次**：`confirm_subject` 在 `:251` 调一次 search（`SEARCH_SEED`），确认期在
`:260-296` 的 `for attempt in range(1, BENCH_ATTEMPTS + 1)` 里**最多调 3 次**（`BENCH_ATTEMPTS == 3`，
`tcenv/bench.py:157`；`timing_uncertain` 会重试，`confirmed` 或别的原因就提前退出）。所以测试必须：

- **替换 `tcenv.backends.metal.portfolio_confirmation._evaluate`**（模块级函数，`:122`）为包装器，
  捕获 **1 次 search run + 全部 confirmation runs**（放进列表，不要假设长度）；被测调用结束后用
  sidecar 的 `confirmation_attempts` 校验捕获到的 confirmation 条数（`len(confirmations) ==
  document["confirmation_attempts"]` 且 `1 <= confirmation_attempts <= BENCH_ATTEMPTS`）；
- **每个 run 各自**断言 `run.evaluation["dispatch_stages"] == run.evaluation["expected_dispatch_stages"]`
  且 `run.evaluation["dispatch"] == run.evaluation["expected_dispatch"]`
  （这些键只在 `run.evaluation` 里，`portfolio_runner.py:1066-1077`）；
- **每个 confirmation 都与 search 比**：`search_run.evaluation["dispatch_stages"] ==
  confirmation.evaluation["dispatch_stages"]`，并且 `search.record.variant_id ==
  confirmation.record.variant_id`（同一颗变体）——不是"两期各一次"的对比；
- **不要**从 sidecar 的 `search_report`/`confirmation_report` 或 schema-4 report 里读 `dispatch`、
  `dispatch_stages`、`host_runs` —— 那些字段不在那里；若不想 monkeypatch，就另做一次**明确的新鲜
  replay**（拿 winner 的 envelope 原始字节直接 `evaluate_candidate`）来产生这些对象。

### B4b 发布/归档身份与 catalog（必须在 B5-b 之前，无需 GPU）

改动：§2.6 全部（`experiments/common.py` v2 读盘、`taic/policy.py` 原因码、`kernel_catalog` +
`_archive_if_confirmed` 持久化、`_prior_kernel_text`/`_search` 门/`_from_text`/`read_bytes`），
**包括 `orchestrator.py:2368,2373` 的 prior 来源显示名/路径重构**（不重构就是 `TypeError`）。

验收（CPU，合成 run 目录 + 合成 catalog）：

- bundle confirmed 正例：`conclusion_of_run` → `confirmed` + 中位 speedup；
- 错 envelope（改 body）→ `sha_mismatch`；`best`/`winner` 不一致 → `sha_mismatch`；
  sidecar manifest 与 decode 不符 → `sha_mismatch`；
- **坏的读写输入不崩**：一份 envelope 形状但含**非法 UTF-8 字节**的 `best.metal`、一份非 ASCII 的
  `taic-manifest` 行、一份未知 role 的 envelope → 三者都返回 `(未确认, sha_mismatch)`，
  `conclusion_of_run`/`publication_reason` 都不抛异常；
- **未登记的 track id（well-typed）**：`result.track = {"track_id": "no-such-track", "track_version": 1,
  "delivery": "single_source"}` → `conclusion_of_run(...)["publication"] is False` 且
  `publication_reason(...)["reason"] == "track_mismatch"`，**两者都不抛异常**（registry 抛的是
  `UnsupportedContractCapability`，必须窄捕获）；同一记录喂给 `taic resume` 的准备阶段 →
  `ResumeRefused`（不是 traceback）；
- **内部矛盾记录**：`track_id=msl-bundle` + `track_version` 合法 + `delivery="single_source"` →
  `track_mismatch`；同一矛盾出现在 `run_config.json` 上、resume 时 → `ResumeRefused`；
  `result.track` 缺失但 best 是 envelope → `track_mismatch`（不抛异常）；
  `result.track` 是畸形（字符串/缺 `track_id`）→ `track_mismatch`（不抛异常）；
  旧 v2 记录（无 `track`）且 best 是单文件文本 → 与今天逐字段相同（legacy 规则）；
- **track 绑定**：`result.track` 声明 `single_source` 但 best 是 envelope → `track_mismatch`；
  反之亦然（single 文本 + `candidate_bundle`）；envelope decode 出的 `track_id` 与 `result.track.track_id`
  不符 → `track_mismatch`；`result.track.track_version` 与 registry 当前版本不符 → `track_mismatch`
  （**envelope 没有 `track_version`，不对它做任何断言，也不升级格式**）；
- **单文件判定不变（含 CRLF）**：一份既有单文件 run 目录，`best.metal` 用 CRLF 写盘，
  改动前后 `(publication, reason, speedup)` 逐字段相等；`transport_sha == sha256(read_bytes())`；
- **catalog 持久化**：确认后 `meta.json`/`history.jsonl` 同时有 transport 与 manifest 摘要 + track；
  blob 仍按 transport 寻址（旧 blob 命中率不变）；
- **端到端的 seed 路径**（不是只测 `_prior_kernel_text()` 的返回值）：在合成 catalog 里放一份 envelope
  best，跑 `_search`（`from_catalog=True`）→ 实际完成 `prior` 评测、`self.best` 被设置、日志出现
  `-- re-evaluating <catalog best 路径>`（**证明 prior 来源显示名重构已生效、没有 `None` 路径运算**），
  且**没有** `external_duplicate`；残缺 envelope → typed `RunStartError`。

### B5 真实 GPU 垂直切片

**范围**：`inside_timing=true` 已有直接 `evaluate_task` 正例（`test_track_a4.py:1149-1176`）；缺的是
**搜索/portfolio 派生 lifecycle** 路径与**多阶段/多源组合**。

#### B5-a 搜索路径（多阶段与多源放这里）

`tests/backends/metal/test_search_v2_bundle_gpu.py`，用 `tests/support/shape_portfolio_metal.py` 的夹具题：
`replace(bind_task(..., tasks_root=tmp_path), track_id="msl-bundle")`，经 `evaluate_candidate` /
`V2LocalEvaluator` + `confirm_candidate`：

1. **派生 lifecycle 的 host tiling**：`host_runs` 非空、`workspace_bytes == {"scratch": 4*N}`（**等于 ABI
   span**，不是「与几何一致」）、`dispatch == expected_dispatch`、`inside_timing is False`；
2. **派生 lifecycle 的 `inside_timing=true`**：几何声明 `{"scratch", zero:true}` →
   `run.evaluation["workspace_zero"] == {"scratch": True}`、
   `run.result.execution["workspace_fills"]` 非空（**注意**：`workspace_fills` 只在 evaluator 侧的
   `ExecutionResult.execution` 里产生，`track_adapter.py:456-459`；portfolio 的
   `run.evaluation` 没有这个键）、`run.result.report["benchmark"]["candidate"]["median_us"] > 0`
   （**不要**写裸 `median_us`：报告里没有这个名字，`evaluate.py:646-655` 的 bench 段把它放在
   `benchmark.candidate` 下）；
3. **多阶段（无动态 threadgroup 参数）+ 动态 threadgroup 的负例**：两 stage、两个非零 buffer 参数、
   `threadgroup_bytes` 全 0 → `dispatch_stages == [0,1]`、签名里每段 `threadgroup_bytes == 0`；
   再加一条**负例**：kernel 声明 `threadgroup float scratch[64] [[threadgroup(0)]]` →
   typed reject。核实过为什么本期不做动态 threadgroup：真实 `[[threadgroup(n)]]` 参数会被
   `stage_binding.py:200-204` 直接拒（"dynamic threadgroup parameter is not part of the ABI"），
   PyObjC 仅在存在 `threadgroup` binding 时才调 `setThreadgroupMemoryLength`（`runtime.py:638-640`），
   原生批量路径一旦有 `threadgroup` binding 就退回 PyObjC（`native_batch.py:263-275`，且 `:180-185`
   要求该路径的字节必须为 0）。所以「geometry 里写一个非零 `threadgroup_bytes`」**没有绑定**时只是
   一个数字，不构成"动态内存工作"的证据——**不能**把非零数字称作动态 threadgroup 正例。
   要真做正例，必须把 `stage_binding`/`runtime`/`native_batch` 的支持和"真实内存被分配并被写"的断言
   一起纳入改动面（本期**不做**，属于"只接搜索"之外的运行时扩展）。
4. **多 device 源**（§1.5 修复后）：两个 device entry、确定性拼接、跑通一颗真实 bundle。

写进证据：`inside_timing=true` 走 PyObjC 且每轮一次 launch（`v2_runtime.py:915-933`），计时根与批量路径不同。

#### B5-b 新增一道带 workspace 的标准题（可发布切片）

- **为什么不是 flash_attention**：它的 OpSpec 没有 `space: workspace`，投影直接拒；加张量会改该题的
  `opspec/pack` digest，动到 P1/P2/P3 已关闭证据。所以新增题。
- **单 stage**（多阶段由 B5-a 覆盖）：`y[n] = scale_tab[n // SEG] * x[n]`，其中
  `scale_tab[s] = BASE + STEP * s`（fp32）。**具体合同（可直接落地）**：

  > **oracle 接口约束（v9 修正）**：reference 的签名是 `reference(inputs, attrs)`
  > （`taskcontract/reference.py:5,133-159`），`attrs` = `instance.resolved_attrs`；它**看不到**
  > `instance.symbols`，而 workspace 又不在 `reference_inputs`/`reference_outputs` 里。所以
  > `SEG/BASE/STEP` **不能**做 symbol——必须做**整数 attr**（v2 的 attr 值只来自 `default`，
  > 不能逐 size 覆盖，正好对应「这些常量对所有 size 相同」的语义）。`N`/`NS` 仍是 symbol
  > （shape/ABI 需要它们）。`HostTilingPlan._names`（`v2_runtime.py:69-80`）同时吃 symbols 与整数
  > `resolved_attrs`，所以 host 侧两边都能求值——**但 oracle 只能靠 attrs**，这就是分界的理由。

  - 顶层：`spec_version: 2`、`op_name: "<新题名>"`、`category: "elementwise"`。
  - `symbols`：仅 `N`、`NS`（逐档给值，`{"min": 1}`）；**不出现** `SEG/BASE/STEP`。
  - `attrs`（整数、所有 size 同值）：`SEG: {"dtype":"int","default":64,"domain":{"gt":0}}`、
    `BASE: {"dtype":"int","default":2049}`、`STEP: {"dtype":"int","default":2}`。
    `BASE` 取 **> 2048** 是刻意的：fp16 在 2048 以上间隔为 2，`tab` 的奇数项**不可精确表示**，
    于是「先把 scale 舍入到 fp16 再乘」与「fp32 乘、只在输出舍入一次」在本题实例上**可区分**
    （否则 `1,3,5,…` 全可精确表示，两种语义结果相同、测试抓不到）。`tab` 范围
    `2049..2049+2*(NS-1)`（paper `2049..2303`，small `2049..2079`），远小于 fp16 上限 65504。
    名称刻意不用 `BLOCK`（`_GEOMETRY_ATTR_TOKENS`，`v2.py:66-87` 是 attr 的禁用词）；
    `SEG/BASE/STEP` 都不在该集合里。
  - `spec_params`：`["N", "NS"]`（必须都是已声明 symbol）。
  - `tensors`：
    - `x`：`role: input`、`dims: ["N"]`、`dtype: "T"`（`type_vars.T = [fp16, bf16]`）、
      `layout: contiguous`、`strides: ["1"]`、`space: global`、`access: read`；
    - `y`：`role: output`、`dims: ["N"]`、`dtype: "T"`、`layout: contiguous`、`strides: ["1"]`、
      `space: global`、`access: write`；
    - `scale_tab`：`role: output`、`dims: ["NS"]`、`dtype: fp32`、`layout: contiguous`、`strides: ["1"]`、
      `space: workspace`、`access: write`（fp32 让 span `4*NS` 是 4 的倍数；不进 reference 输入/输出）。
  - `sizes`（两档，都声明 `usage: ["golden","benchmark"]`）：`paper: {N: 8192, NS: 128}`、
    `small: {N: 1000, NS: 16}`（`small` 是 tail 档：`ceil(1000/64) = 16`）。
    测试固定 `NS == ceil(N/SEG)`（`SEG` 从 `instance.resolved_attrs["SEG"]` 取）。
  - `formula`：`"y[n] = (BASE + STEP * (n // SEG)) * x[n]; the same ramp is materialised into the
    workspace scale_tab[NS] by Host Tiling"`。
  - `type_vars`：`{"T": ["fp16", "bf16"]}`；`category`：`"elementwise"`。
  - `numerics`：`{"acc_dtype": "fp32", "rounding": "single_round_at_output",
    "tolerances": {"fp16": {"rtol": 0.002, "atol_rel": 0.001},
                   "bf16": {"rtol": 0.016, "atol_rel": 0.008}}, "exact": false}`
    （`tolerances` 必须覆盖 `type_vars.T` 的每个 dtype，否则 `instantiate_v2` 直接拒）。
  - `cases`：`{"names": ["normal"], "inputs": {}}`（没有额外 case 输入；`gen_inputs` 可选）。
  - `work_accounting`：`{"flops": {"terms": [{"coefficient": 1, "symbols": ["N"]}]},
    "min_bytes": {"terms": [{"tensor": "x"}, {"tensor": "y"}]}}`
    （`_validate_accounting` 要求每项「恰好一种量」，这里的量是 `symbols` 或 `tensor`）。
    为什么 `min_bytes` 只算 `x`/`y`：`min_bytes` **不会**自动排除 workspace，这里显式不算
    `scale_tab` 的理由是**算法层面**的——合法候选可以自己重算 ramp、完全不需要读那张表，
    所以表的流量不是本题的强制工作；写成三项会变成一个我们无权要求的"最低工作量"。
  - `discovery`（可选但本计划要求写，`default_instance` 决定 `bind_task` 的初始 locator）：
    `{"phase": 1, "default_instance": {"dtype": "fp16", "size": "paper"},
    "tags": ["workspace", "host-tiling", "bundle"]}`。
  - `reference.py`（只用 `inputs` 与 `attrs`；**独立、不读 symbols、不按 N 特判、不做输出舍入**）：
    ```python
    import torch

    def reference(inputs, attrs):
        # 真实路径里 x 已经是 float32：round_to 会把每个浮点张量先舍入到声明 dtype 再加宽回
        # float32（tensors.py:25-36），v2_runtime.py:261-266 在调 reference 之前就这么做。
        # 所以绝不能拿 x.dtype 推断「声明输出类型」——那是 float32，不是 T。
        x = inputs["x"].to(torch.float64)                # [N]
        seg, base, step = int(attrs["SEG"]), int(attrs["BASE"]), int(attrs["STEP"])
        n = x.shape[0]
        ns = (n + seg - 1) // seg                        # = 声明的 NS；reference 自己算，不读 symbol
        tab = base + step * torch.arange(ns, dtype=torch.float64, device=x.device)
        idx = torch.arange(n, device=x.device) // seg
        return {"y": tab[idx] * x}                       # fp64，交给 harness 用 tolerances 判
    ```
    `import torch` 必须写：`load_reference`（`taskcontract/reference.py:471-501`）把模块当普通模块
    执行（`_load_module`），**不注入任何名字**。oracle 保持 fp64 未舍入；输出类型由 `tensors` 的
    `dtype: "T"` 声明，**不由 `x.dtype` 推断**。`y` 是唯一 `reference_outputs`；`scale_tab` 是
    workspace → **不进 reference 结果**（`_validate` 只要求 `reference_outputs` 键闭合）。
    **不要**在 reference 里做 `.to(torch.float16)`：`rounding` 没有进入 `SemanticInstanceV2`，
    harness 的判定只有 `tolerances + scaled_error`（`docs/架构/评测.md`），reference 自己舍入只会
    把「语义分歧」藏进比较里，而不会让判定更严。
  - `baselines.json`/`baselines.py`：**一参数 runner + 必需 `BASELINES` 映射，不能读候选的 workspace**。核实：
    `BaselineRunner`（`baselines.py:89-100`）只为 `task.contract.inputs` 里的每个张量传参，本题
    contract 的 `inputs` 只有 `x`（`track_adapter.py:82-106` 把 `space=="workspace"` 的张量从 `args`
    里排除），所以 `scale_tab` 根本到不了 baseline。写法固定在 `baselines.py` 模块级常量：
    ```python
    import torch
    SEG, BASE, STEP = 64, 2049, 2          # 与 opspec attrs 一致；测试断言相等，防止漂移

    def run(x):                            # 唯一参数：contract.inputs = [x]
        n = x.shape[0]
        ns = (n + SEG - 1) // SEG
        tab = BASE + STEP * torch.arange(ns, dtype=torch.float32, device=x.device)   # 独立构造 ramp
        idx = torch.arange(n, device=x.device) // SEG
        return (tab[idx] * x.to(torch.float32)).to(x.dtype)

    # 必须登记：load_baselines（baselines.py:145）要求 BASELINES 是 name -> callable 的 Mapping，
    # 且键集合与 baselines.json 的登记集合**完全相等**，否则新题在 discovery/load 阶段就被拒。
    BASELINES = {"torch_mps": run}
    ```
    与之闭合的 `baselines.json`（**只能**是 `schema_version + baselines` 两键：
    `tcenv/taskcontract/schema/baselines.v1.schema.json`（由 `tcenv/taskcontract/baselines.py:32` 的
    `BASELINES_SCHEMA_PATH` 引用）是 `additionalProperties: false`，多一个键就 `SchemaFailure`；
    `framework` 必须与 `to_framework` 支持的 `torch`/`mlx` 对齐）：
    ```json
    {"schema_version": 1,
     "baselines": [{"name": "torch_mps", "framework": "torch",
                    "implementation_id": "torch_mps_affine_ramp", "implementation_version": 1,
                    "params": {}}]}
    ```
    闭合关系由 `baselines.py:194-207` 强制：`BASELINES` 不是 `Mapping`、或 `missing`/`extra` 非空 →
    `AssemblyError`。**真实装载入口是 `load_baselines(task_dir, op_name=...)`（`baselines.py:145`）**，
    仓库里没有 `load_baseline_package`；题目测试必须**真的调用它**（这样上面的 manifest 错误会在测试里
    直接暴露，而不只是检查 `BASELINES` 映射）。
    这里 `x` 到 baseline 时**已经是声明 dtype**：`BaselineRunner` 用 `to_framework(..., a.dtype, ...)`
    把输入转成 `fp16`/`bf16`（`baselines.py:50-73`），所以最后那次 `.to(x.dtype)` 就是唯一的输出舍入
    （与候选「fp32 乘、输出舍入一次」对齐）。
    要求：ramp 由 baseline **自己**构造（fp32）、fp32 相乘、只在最后 `.to(x.dtype)` 一次；
    `baselines.json` 声明该 runner 为目标 baseline；缺 `baselines.py` `_confirm_v2` 会跳过确认。
    **GPU 数值测试是必需的**：v2 计时只判候选 correctness，不检查 baseline 输出（`evaluate.py`
    的 bench 段只收集 `baseline_runners` 的时间），所以要有一条**需 GPU 许可**的测试，用固定输入
    断言 baseline `run(x)` 与 reference 在 `tolerances` 内一致——否则一个输出全错的 baseline 也能
    当分母，speedup 会失真。
  - `known_good.metal` 也必须用同一语义（读 fp32 `scale_tab[s]` → fp32 相乘 → 输出时舍入一次），
    不能把表项先降到 `half` 再乘；否则参考核与 oracle 的偏差会随 `BASE` 放大，量到的是语义分歧而不是性能。
  - Host Tiling（`host/tiling.json`）：
    ```json
    {"schema_id": "tcenv.metal.host-tiling.1", "schema_version": 1,
     "writes": [{"workspace": "scale_tab", "kind": "linear", "start": "BASE", "step": "STEP",
                 "count": "NS", "stride": 1, "offset": 0, "dtype": "fp32"}]}
    ```
    `BASE/STEP` 来自 `resolved_attrs`（整数 attr），`NS` 来自 `symbols`——两处都由
    `_names(instance)`（`v2_runtime.py:69-80`）解析，正是这两个来源的并集；写出的第 `s` 个
    fp32 = `BASE + STEP*s`，`s = 0..NS-1`。
- **该题自己的测试**（不能拿固定 fixture 的测试充当证据：`test_opspec_v2.py` 用固定 fixture，
  `test_v2_g0_reference_outputs.py:96-102` 只参数化 RMSNorm/sigmoid）：
  `tests/shared/test_<op>_task.py` 覆盖 discovery → `load_task_contract` → 每个 (dtype,size) 的
  `instantiate_v2` → reference 与手算/float64 对照（含 tail 档）→ `NS == ceil(N/SEG)`
  （`SEG` 从 `instance.resolved_attrs` 取，**不从 symbol**）→ ABI workspace span（`4*NS`）→
  「reference 只读 `inputs`/`attrs`，且在 fp16/bf16 两档、paper/small 两档都对齐手算」→
  **baseline 闭合**：`load_baselines(task_dir, op_name=...)`（`baselines.py:145`）能装上该题（它的
  `validate` 会先跑 `baselines.v1.schema.json`，所以文档多/少一个键就失败；随后 `BASELINES` 的键集合必须与
  登记的 `torch_mps` 完全一致，`baselines.py:194-207`），且 `runners["torch_mps"]` 可调用、
  `baselines.py` 的 `SEG/BASE/STEP` 与 `instance.resolved_attrs` 相等 →
  **舍入语义判别用例（bitwise，不用相对误差）**：在**确定性输入**上比较两条构造路径的**最终
  `float16` 位模式**，good/bad 两边都做同一次最终 cast，只差 scale 是否被预先舍入：
  ```python
  x = torch.tensor([(i * 37 % 17 - 8) / 8 for i in range(64)])       # 精确可表示的定值输入
  tab = BASE + STEP * torch.arange(NS, dtype=torch.float32)          # 2049, 2051, ...
  idx = torch.arange(64) // SEG
  good = (tab[idx] * x).to(torch.float16)                            # 先 fp32 乘、只在最后舍入
  bad = (tab[idx].to(torch.float16) * x).to(torch.float16)           # 预先把 scale 舍入到 fp16
  assert not torch.equal(good.view(torch.int16), bad.view(torch.int16))   # 至少一个元素位模式不同
  ```
  `BASE=2049` 让这条成立：fp16 在 >2048 区间步长为 2，`2049 → 2048`，于是 bad 的乘积整批偏一个
  ulp 量级；`1/2049 = 4.8804e-4 < 2**-11 = 4.8828e-4`，**相对误差口径抓不到**这个差异
  （v9 的写法因此是错的）。若另加相对误差检查，必须写明适用 dtype 与「有限正常输出」前提：
  fp16 用 `2**-11`，**bf16 用自己的 `2**-8`**，且不把它当成判定规则的替代。
  **不要声称这条测试固定了 harness 的 `single_round_at_output`**：`rounding` 没有进入
  `SemanticInstanceV2`，判定仍然只有 `tolerances + scaled_error`；当前容差会**接受**预舍入 scale
  的候选。这条测试固定的是**本题参考核/baseline 的口径**（证据卫生），不是判定语义。
- **`eval-v2` 素材、证据绑定与命令（形态定清）**：`tasks/<op>/known_good.metal` 保持**原始 MSL 文本**
  （它是 pack 文件：`taskcontract/loader.py:55,64,418-432` 把它读成 `contract.known_good` 并算 sha，
  并进 `contract.fingerprint()` 与任务 pack），所以它**不能**直接交给 `eval-v2 --track msl-bundle`
  （bundle 收到非 envelope 必须拒）。
  **自测素材不进 pack**：`V2_STANDARD_FILES`（`loader.py:48-56`）与 `STANDARD_TASK_FILES`
  （`pack.py:37-45`）都是**固定顶层清单**，`task_pack_files()`（`pack.py:246-258`）不递归，
  所以往题目录里塞 `selfcheck/` **不会**进入 task/deployment pack，也不会改变 pack digest
  ——不能拿它当身份保证。本期做法（不扩包闭包）：

  | 素材 | 位置 | 身份绑定方式 |
  |---|---|---|
  | 读表的 device kernel | `tasks/<op>/known_good.metal`（在 pack 内） | 打包时**逐字节复用**它；测试用 `known_good_path.read_bytes()` 与解码出的 device entry **字节**比较（**不要** `read_text()`，CRLF 会被归一），身份是 `contract.fingerprint_document()["known_good_sha256"]`（`TaskContract` 没有 `known_good_sha256` 属性，`loader.py:334-360`） |
  | `host/tiling.json`、`host/execution_geometry.json` | `tests/support/msl_bundle_<op>/host/`（测试夹具，不在 pack） | B5-b 的证据 JSON 记录两者 `sha256`；夹具改动 = 证据重录 |
  | 打包出的提交文件 | 临时 work 目录 | 证据 JSON 记录 `transport_sha256` 与 `candidate_sha256`（manifest） |

  可重现的命令（写进 B5-b 证据）：

  ```
  # 1) 组装 --src（device 直接取 pack 内的 known_good.metal，保证身份绑定）
  <work>/selfcheck-src/device/<kern>.metal          <- tasks/<op>/known_good.metal
  <work>/selfcheck-src/host/tiling.json             <- tests/support/msl_bundle_<op>/host/tiling.json
  <work>/selfcheck-src/host/execution_geometry.json <- tests/support/msl_bundle_<op>/host/execution_geometry.json
  # 2) 打包 → 3) 自测
  uv run --directory <repo> python scripts/msl_bundle_pack.py \
      --src <work>/selfcheck-src --out <work>/selfcheck.envelope.metal
  uv run --directory <repo> python -m tcenv.cli eval-v2 <op> <work>/selfcheck.envelope.metal \
      --dtype fp16 --size <size> --quick --track msl-bundle
  ```

  `--quick` **只能当 compile/ABI/safety/correctness 自测**：`evaluate.py:623-632` 在
  `mode in ("validate","quick")` 时于 bench 之前 `finish("ok", "compiled, ABI ok, safe and correct")`
  返回，没有时间、没有 speedup。性能证据必须另跑 **full/timed**（`--timed`，需 GPU 许可）。

  （后缀仍 `.metal`，靠 magic 自识别。若将来确实要求自测素材进 task pack，必须同时改两个固定清单、
  locator/staging 闭包与 `test_task_contract_v2`/`test_structure` 的对应断言——本期**不做**。）
- **Host Tiling 承重的验收（只绑定固定 known-good bundle，不绑定搜索 winner）**
  1. **正例（CPU 侧身份 + GPU 侧 fresh replay）**：把上面打包出的 **known-good envelope** 的原始字节
     直接喂 `evaluate_candidate(...)` 做一次**新的 GPU replay**，从 `run.evaluation`
     （`runner.py:404-415` 的 evaluator 侧字段）断言：`host_runs[-1]["writes"] == 4*NS`、
     `source_sha256` 是 `host/tiling.json` 的摘要、该 stage 的 `workspace == ["scale_tab"]`、
     `workspace_zero == {"scale_tab": False}`；dispatch 证据也只能从 `run.evaluation` 取：
     `run.evaluation["dispatch"] == run.evaluation["expected_dispatch"]`，且
     `run.evaluation["dispatch_stages"] == run.evaluation["expected_dispatch_stages"]`
     （`portfolio_runner.py:1066-1077`；`report` 是 schema-4 `EvalReport`，**没有**顶层 `dispatch`
     ——写 `report["dispatch"]` 会直接 `KeyError`。现有 GPU 正例就是这么断言的：
     `test_shape_portfolio_gpu.py:283-299`）。
     **不能**声称「从现有搜索报告里读 `winner.host_runs`」：`execution` 不进 schema-4 document
     （`runner.py:404-415`），`V2LocalEvaluator` 只保留 `result.report`（`v2_evaluation.py:106-124`），
     落盘报告里没有 host/dispatch 明细 → 必须 replay。
  2. **负例（确定性，不是"改错一点"）**：同一 device 源 + `start=0, step=0` 的 host plan（表全零）
     组成第二颗 bundle → 固定 seed 下 correctness 必须是 `incorrect`。用全零表而不是"把 start/step
     改错一位"，是为了避免改写后的值仍落在 `tolerances` 内、把负例变成噪声。
  3. **不要要求真实搜索 winner 必须读表**：device 自己重算 ramp 是完全合法的算法，要求它就等于
     「只有挑到偏好的算法才通过」，直接违反候选自由。真实 winner 的验收只到：
     其 envelope 真实字节 → correctness/ABI/安全 → `confirm_subject` → 发布身份
     （`transport_sha`/`manifest_sha`/`track`）。
  4. **本条只约束本题验收素材**，不新增任务级拒绝语义、也不加 per-task 门禁；
     `workspace_probe` 只看最终字节，无法判定 device 是否读过。
- **许可**：见 §4 总则（GPU 需许可；真实搜索 + 确认还需单独付费许可）。

验收（B5-b 分三段，且都要 GPU 许可）：

**第一段：`tcenv eval-v2 --quick`（无时间）**
- `correctness/ABI/安全` 全过（`evaluate.py:623-632` 在 bench 前就返回，没有时间、没有 speedup）。

**第二段：`tcenv eval-v2 --timed`（有时间，但**不会**产生确认值或 run conclusion）**
- `report["benchmark"]["baselines"]` 里**确实有** `"torch_mps"` 一项（**不要**断言 `base_runners`：它只是
  `evaluate.py:630-634` 的局部变量，CLI/report/搜索结果都不暴露）；
- `report["baseline_us"]["torch_mps"]` 有限且 > 0；
- `report["benchmark"]["candidate"]["median_us"]` 有限且 > 0；
- `report["speedup"]` 非空有限（`report["speedups"]["torch_mps"]` 同）。

**第三段：真实搜索 + 确认（`confirmed_values` / `conclusion_of_run` 只在这里出现）**
- `confirmed_values` 非空；`conclusion_of_run` 的 published speedup **不为 `None`**
  （防止 baseline 被静默丢弃后仍得到 `confirmed + speedup=None`）；
- baseline 数值：GPU 测试断言 `run(x)` 与 reference 在 `tolerances` 内一致（§B5-b 上文）。
- 证据措辞：本题 `zero=false` → Host Tiling 在**计时外**，而 baseline 每次调用都自己构造 ramp，所以
  证据必须标为 **steady-state table-reuse**；**不得**写成「两种实现端到端做了相同的准备工作」。
- 身份：`result.json` 的 `track.track_id == "msl-bundle"`、`track.delivery == "candidate_bundle"`、
  `confirmation_source_sha256` 是 envelope 文本哈希、sidecar `candidate_sha256` 是 manifest digest，
  且上面**四条**承重证据成立。
- B5-a：四个节点全绿（其中第 3 项按下面改过的「不支持动态 threadgroup 参数」负例实现）。

### B6 回归与文档

- 快速安全套件 `uv run pytest -q -m 'not gpu and not llm and not slow'`；相关 CPU 子集
  `-m 'not gpu and not llm'`；Metal GPU 套件需许可。失败先读 `logs/pytest/triage/triage_summary.md`。
- 文档：`docs/架构/v2协议参考.md` 加「bundle 交付」段（entry/role/固定几何路径与 media_type、
  `taic-launch` 在 bundle 里的含义、Host Tiling 的 `linear` 与整数约束、`zero` 的代价、
  正确性 zero/poison 语义与「只清声明 zero」、1,000,000 字节上限）；`docs/架构/搜索-v2接线.md` 更新状态；
  本文标 completed；`experience.md` 只加实测结论，未测写 unverified。
- **原因码文档必须同步（B4b 新增了 `track_mismatch`）**：
  - `docs/架构/v2协议参考.md:480-499` 的「L. 发布 verdict / reason 闭集」表加一行
    `track_mismatch`（含义："`result.track` 与 best/winner 的交付形态、`delivery` 或 `track_version`
    不符/畸形"），并注明它与 `sha_mismatch` 的分工（字节对不上是 `sha_mismatch`，身份/交付形态对不上是
    `track_mismatch`）；
  - `docs/架构/数据层.md:519` 的 `publication_reason` 原因码枚举列表按 `taic/policy.py` 的**代码顺序**
    插入 `track_mismatch`（并保持"按代码顺序取第一个命中"的说明成立）；
  - 验收：两份文档里的 reason 列表与 `taic/policy.py` 的闭集**逐项相等**（一条测试断言集合与顺序，
    不是人眼对照）。

---

## 4. 边界与风险

### 4.0 与 v1→v2 迁移的协调（2026-11-03 讨论定案）

另一路正在把所有 v1 任务迁移到 v2，完成后移除 v1 支持。**这与本 plan 方向一致、不是障碍**：

- 本 plan 的接入点（`taic/v2_evaluation.py`、`tcenv/backends/metal/search_candidate.py`、`orchestrator.py` 的 v2 分支、`prompts.system_prompt_v2`、`kernel_agent` 的 v2 brief）**全部落在 v2 路径上**，不依赖 v1。
- 本 plan 里的 **v1 残留**（`task_binding` 的 v1 分派、`orchestrator` 的 v1 分支、`prompts` 的 v1 system_prompt、`kernel_agent` 的 v1 brief）会在 v1 移除时被删掉——**正是那一路要做的事，与本接入不冲突，反而让接入点更纯粹**。

**唯一要做的协调**：迁移如果改了 v2 的接口（`bind_task`、`TaskContract`、`evaluate_task`、`run_portfolio_task`、envelope、几何/identity），接入时必须**对齐迁移后的最新接口**，不能按本计划调研时的旧接口写死。建议二选一：

1. **等迁移收尾再按本 plan 接入**（最稳，接口冻结后直接做）。
2. **先对齐当前 v2 接口再接入**（确认没撞车就开始；变了的接口按新接口调整）。

倾向等迁移收尾后动手，避免在迁移进行中改同一批文件造成返工。

- **GPU 与付费许可（总则）**：所有 GPU 节点——B4 的 `confirm_subject`、B5-a 四节点、B5-b 的参考核与真实
  搜索——都需**显式 GPU 许可**；B5-b 的真实搜索还需**单独付费许可**。无许可时该阶段只能标「CPU 部分完成」。
- **workspace 是硬门**：无 `space: workspace` 的题收不了 bundle；放宽需单独立项改 `_require_bundle_shape`。
- **Host Tiling 只有 `kind: linear` 且全部整数**：复杂 host 循环、多阶段 host work、`input_dependent=True`
  都不在本期；题面要写「`base/step` 是整数」。
- **`zero` 的三重代价**：计时窗内每迭代一个 fill kernel + 退掉原生批量编码器（`native_batch.py:268-275`）
  + 把 host_tiling 拖进计时窗；走 PyObjC 且每轮一次 launch。
- **清零语义**：v1/v2 都只清声明 `zero` 的；正确性对每个 workspace 都填（zero/poison 不对称）→
  依赖初始零的候选必须被 poison 抓到。
- **多源 = 单编译单元**：符号不可重复、只能一个 `taic-launch` 块（预检提前拦）、不能引号 `#include`。
- **动态 threadgroup 参数本期不支持**：`[[threadgroup(k)]]` 函数参数被 `stage_binding.py:199-204` 拒；
  静态 `threadgroup T name[N];` 可用；`threadgroup_bytes` 必须写 0，非零数字不算动态内存证据
  （PyObjC 只在存在 `threadgroup` binding 时分配，`runtime.py:638-640`；原生批量路径也要求 0，
  `native_batch.py:180-185,263-275`）。
- **fallback 语义**：搜索父包恒 `fallback=None`、manifest 由 harness 生成；候选不能声明 fallback。
- **身份记账**：`transport_sha`（提交原始字节）与 `manifest_sha`（候选）不可互换；发布按 §2.6 分别绑定，
  R5 与 `verify_subject_document` 不动。envelope 顺序变化只影响前者；**一切哈希用 `read_bytes()`**。
- **原因是闭集**：新增 `track_mismatch` 是追加；既有原因码与既有运行判定不变。
- **remote workspace**：本期 bundle + remote workspace 在启动期拒绝（`workspace_prompt.py:23` 不打印
  控制机命令，远端无法运行打包脚本）。
- **prompt 指纹**：本计划会改 `HARNESS_RULES`（**只**含清零句纠错；原文保留，不含新片段）、
  `_rules_for`（新增 `delivery` 参数、**函数体内**的四条替换句与 geometry/workspace 指引）、
  `system_prompt_v2`（**函数体内**的 bundle 段）、`render_contract`、
  `task_binding.TaskBinding.signature_stub`、`_check_command` 与 SKILL.md；
  `signature_stub`、`system_prompt_v2`、`_check_command` 是**新增**登记（`_PROMPT_FUNCTIONS` 里原本
  没有后两者；`signature_stub` 还需要 `_symbol_bytes` 支持点分路径，因为它是方法）；
  `HARNESS_RULES`/`_rules_for`/`render_contract` 本来就在 `_PROMPT_CONSTANTS`/`_PROMPT_FUNCTIONS` 里，
  而**新增的替换文本全部放在这两个已哈希函数的函数体内**，所以 `_PROMPT_CONSTANTS` 本期不新增条目
  ——除非真要把某段提到模块级，那就必须同时登记。→ `current_prompt_sha1()` 与
  `docs/架构/backend-phase0-baseline.json` 必须重签，旧授权按设计失效；**所有 track 一起变**，
  因为清零纠错对所有 track 生效，而 bundle 分支只改渲染结果、不改 single 的字节（除那一句）。

---

## 5. 交付顺序与工作量

| 阶段 | 内容 | 依赖 | 量级 |
|---|---|---|---|
| B0 | §1.5 修复 + 源头规范化 + 打包脚本 + 包装/预检 + 交付错误分类 + `SessionBrief.track_id` 字段 + `rejected_report()`/`_record_delivery_rejection()`（单元级验收） | — | 24% |
| B1 | 投影/lifecycle 复用证明（只加测试） | B0 | 7% |
| B2 | `Orchestrator(track_id=)` 全链 + resume + `_check_command`/`retest_command` 注入 + run_config/result + seed | B0,B1 | 12% |
| B3 | 三处文案（`prompts.py:73-74,170`、`kernel_agent.py:662`）+ v2 bundle 段（geometry 闭集 + workspace MSL ABI）+ agent 清单 + 受信面登记 + 重签 + remote 拒绝 | B0,B2 | 10% |
| B4 | freeze + `confirmation_expected`（带 `track_id`） | B1,B2 | 6% |
| B4b | 发布/归档身份 + track 绑定 + catalog 持久化 + prior 来源重构/seed 路径（无 GPU；seed 与 catalog 验收要用真实 `Orchestrator`，所以依赖 B2） | B0,B2 | 11% |
| B5-a | 派生 lifecycle：host tiling / `inside_timing=true` / 多阶段（+ 动态 threadgroup 负例）/ 多源 | B2,B4 | 12% |
| B5-b | 新题（自带测试 + 打包素材/命令/证据绑定）+ 承重证据 + 真实搜索/确认 | B2,B3,B4,B4b,B5-a | 11% |
| B6 | 全链回归（含 envelope 端到端 run 目录）+ 五份文档（`v2协议参考.md`、`搜索-v2接线.md`、本文、`experience.md`、`数据层.md`） | 全部 | 7% |

**B4b 必须在 B5-b 之前完成**（B5-b 的验收含 published speedup 与归档）；B3 依赖 B2 是因为条件段要用
`brief.track_id`；B5-b 依赖 B3 是因为没有题面指引的 bundle 会话跑不出来；**B5-a 依赖 B4** 是因为它要跑
`confirm_candidate`/freeze 路径；**B4b 依赖 B2** 是因为它的 seed/catalog 验收要驱动真实 `Orchestrator`。B0 是唯一「必须先全部做完
才能继续」的阶段。B5-b 单独评审（新题落地即新的关闭边界）。

---

## 6. 为什么这样改（v23）

**一句话（v23）**：v22 把"新文本必须落在已哈希源码里"定成了规则，但正文还留着一句旧措辞
（"`HARNESS_RULES` 拆成可寻址片段，四条冲突句各是一个具名常量"），它和同页的定案直接冲突——照它实施
就又会把新句推到模块级常量上；另外受信面验收里出现了两个 "3."、结尾却写"这三步"。v23 把这段改成
唯一方案（常量整体保留、`_rules_for` 里做旧句→局部新句的替换）并把验收编号与计数对齐。

**v22 的结论（保留）**：v21 把三处锚点修对了，但漏掉一个受信面自洽性问题——B3 新写的 bundle 替换文本如果
放进**未登记**的模块级常量，`current_prompt_sha1()` 不会变（它只哈希登记过的符号），旧授权就会静默
继续有效；而 v21 的判别力验收只测了 `signature_stub`。v22 把"替换文本必须落在已哈希的函数体内"定为
硬规则，并把验收扩成"每条新文本都能在指纹 blob 里找到"+"对 `_rules_for`/`system_prompt_v2` 各做一次
monkeypatch 判别力"。

**v21 的结论（保留）**：v20 修对了三处锚点，但其中两处的**验收**仍然不可执行或会空过：
`_symbol_bytes` 是从 `inspect.getsource` 取文本的，"monkeypatch 改源码"既不改变磁盘文件、也证明不了
指纹会变；B3 用 `dataclasses.replace` 手工造 brief，抓不到"`_session_brief` 仍用旧 binding 取
`abi_stub`"和"新字段 `track_id` 没接上"这两类回归。另外 §2.6 的判定表漏了"well-typed 但未登记的
track id"——`TrackRegistry.get` 抛的是 `UnsupportedContractCapability`，不补进窄捕获就会让
`conclusion_of_run` 抛异常，与"所有 `track_mismatch` 都不抛"直接冲突。v21 把这三处改成可执行、
可证伪的写法。

**v20 的结论（保留）**：v19 的三处结构性修复都成立，剩下三处是"细节锚点"问题：给一个**已经持有
`track_id`** 的 `TaskBinding` 方法硬加 `track_id=` 参数（只会制造第二种 track 来源，而两个现有调用点
都不传参）；把 `baselines.json` 的 schema 路径写成仓库里不存在的 `schema/...`；以及 B4b 新增了
`track_mismatch` 却忘了让 B6 同步两份现行文档的原因码闭集——那会让代码与文档在实施完成后立刻不一致。

**v19 的结论（保留）**：v18 把费用与状态语义修对了，但它写的"不写 kernel 文件"与"`kernel_path` 未设"在仓库里
并不成立（`Candidate` 没这个字段，`_save` 无条件写文件，而 `resume` 又靠这个文件才能读回候选，唯一候选
缺失时还会直接拒绝续跑）；它还让 B0 依赖一个当时不存在的 `SessionBrief.track_id`，让 B2 验收 B3 才写的
prompt 文本，并在发布 legacy 规则上留了一句自相矛盾的全称否定。v19 把落盘方案、字段归属、验收边界与
legacy 判定表逐条钉死。

**v18 的结论（保留）**：v17 的方向（不抛、走 `delivery_rejected`）对，但三处细节会自己拆掉刚建好的语义：
硬编码 `cost_incomplete=False` 会把未知费用按 0 入账（它今天的值是 `not invocation.saw_tokens`，
而且 `_one_session_attempt` 先看这个字段再入账）；硬编码 `status="ok"` 会把 timeout/process failed
改写成正常完成；"调用同一个 rejected 构造器"在非法字节路径上并不存在——`V2LocalEvaluator._rejected`
要求合法的 `EvaluationRequest`。v18 把这三处落成：原样保留费用/状态字段、新增
`rejected_report()` 纯工厂与 `_record_delivery_rejection()` 的无 GPU 落盘路径（写清候选、trace、origin、
费用、失败反馈各记什么）；并把 G21 的真实 orchestrator 端到端从 B0 移到 B2，消掉
`SessionBrief.track_id` 的循环依赖。

**v17 的结论（保留）**：v16 找到了正确的缺口（`collect_session` 的 `read_text`），但开出的处置方式会走进
另一条错路——它跑在会话执行阶段，抛异常只会被 `_one_session_attempt` 的宽 `except` 记成 `agent_failed`、
`cost_usd=None`、`cost_incomplete=True`，既拿不到 rejected，又丢掉已计费的成本。

**v16 的结论（保留）**：v15 之后剩的三处都是"照着写会当场崩或漏掉真实路径"：受信面把一个**方法**当模块函数
登记（`getattr(module, "signature_stub")` 会 `AttributeError`，把整个指纹带崩）；B4 把确认期写成"各调一次"
（真实是 1 次 search + 最多 3 次 confirmation，`BENCH_ATTEMPTS == 3`）；最要命的是**真实交卷链路**
`collect_session` 用 `read_text(errors="replace")`，CRLF 归一与 U+FFFD 替换会让一份本来合法的 envelope
在 decode 时炸掉——前面所有关于"发布/seed 都用 `read_bytes()`"的要求都绕过了这条路。

**v15 的结论（保留）**：v14 把 `HARNESS_RULES` 的矛盾理顺了，但仍漏了两个"读不到/查不到"的洞：TASK.md 自己
那段中文 launch 指引（`kernel_agent.py:672-674`）根本没进冲突表，而验收⑧声称检查 `render_contract`，
查的却是四句**本来不会出现在 TASK.md 里**的英文——那个渲染面必然空过；B4 的"两期 dispatch stage 序列
一致"也没有可读对象：sidecar 只存两份 report，schema-4 report 没有 `dispatch_stages`。这一版把
TASK.md 的中文句纳入渲染期替换并给出中文断言，把 B4 改成捕获 `PortfolioRunResult` 后断言
（v15 把它写成"捕获两次"是错的：确认期是最多 3 次重试，v16 已改为"1 次 search + 全部 confirmation"）。

**v14 的结论（保留）**：v13 的那张表诊断对了，但开出的方子是自相矛盾的——`system_prompt_v2` 的第一段就是
完整的 `HARNESS_RULES`（`prompts.py:157`），"不改常量、只在后面追加 ignore earlier"根本不可能让旧句
从最终 prompt 里消失，而验收⑧又要求它们不出现；同时"不改常量"与"全局改正清零句""本计划会改
`HARNESS_RULES`"两处直接冲突。v14 只保留一种方案：**清零句是允许改的常量正文**（与 track 无关），
**四条 bundle 冲突句在渲染期按 `delivery` 删除/替换**，single 的承诺收窄为"除那一句外逐字节相等"。

**v13 的结论（保留）**：v12 换对了字段名，但仍是"按记忆写接口"：`baselines.json` 多了一个 schema 里不存在的
`schema_id`（实际 `validate` 直接 `SchemaFailure`）、装载函数名不存在；zero-count 正例放行了运行时明确
拒绝的 `stride=0`；而**真正生成"签名骨架"的 `signature_stub()` 仍在跳过 workspace**（它同时进系统提示和
TASK.md，只补一段示例根本没用）；共享 `HARNESS_RULES` 里的三条旧话（launch 决定 grid/threads、可选动态
threadgroup、恰好一个代码块）与 bundle 语义冲突，却因为"只改一句 + 追加一段"而继续生效。这一版把
baseline 文档、span 规则、签名骨架与共享文案都改到**能被测试抓到**的程度。

**v12 的结论（保留）**：v11 把 B5-b 的口径改对了，但它引用的证据对象还有几个是"想象出来的"：baseline 少了
必填的 `BASELINES` 映射（会在 discovery 就被拒）、`report["dispatch"]` 与 `workspace_fills` 在报告里根本
没有、`base_runners` 只是局部变量、timed 自测被写成了能产出确认值；Host Tiling 的新 span 公式还把
`count=0` 的空写算成负 span、并放过负首地址；深嵌套 JSON 只封了 envelope 一条路；题面一边说"不支持
动态 threadgroup"一边又叫 agent 用 geometry 驱动它。这一版把这些全部换成**真实存在的字段与共用的
校验**，并把两处残留（`contract.known_good_sha256`、`cmd_eval_v2` 的归属文件）改对。

**v11 的结论（保留）**：v10 的骨架对，但 B5-b 那一节几乎每一项都踩在真实接口的边上——oracle 的输入在
真实路径上已是 fp32（`round_to` 加宽），所以「只在输出舍入」的写法根本没生效；判据用相对误差抓不到
预舍入 scale（`1/2049 < 2**-11`）；baseline 拿不到 workspace；`--quick` 根本不测时间；承重证据要么
诱导挑算法、要么读一个不落盘的对象；动态 threadgroup 的"正例"没有真实内存；题面还说错了 workspace
的 MSL ABI。加上 `retest_command` 字段名、`delivery` 一致性、四类漏网异常与 Host Tiling span，
这一版把 B5-b 从"读起来合理"改成"照着能实现、能证伪"。

**v10 的结论（保留）**：v9 那三处的方向对，但落地时留了三个「照抄就崩 / 口径不一致 / 总结超前于正文」的洞：
reference 示例没有 `import torch`（模块是裸执行的）；示例把 scale 先舍入到 fp16，等于多舍入一次，
和 `single_round_at_output` 冲突、还会与参考核的 fp32 读表语义分叉；`_failure_record` 的复测命令只写进
了 §2.1 的设计段，没落进 G15/B2 的改动面与验收（而上一轮总结却声称已落实）。三处已补齐。

**v9 的结论（保留）**：v8 之后还剩三处**写到具体合同才暴露**的问题：B5-b 把常量放进 symbol，而 oracle 的
`reference(inputs, attrs)` 看不到 symbols（只有 attrs），也没法读 workspace —— 这样那道题**根本写不出
reference**；track 的命令传播只覆盖了题面自测命令，漏了 `_failure_record` 生成的
differential-recheck 复测命令；另有一处验收计数写错（三条 → 四条）。三处都已改正。

**v8 的结论（保留）**：v7 之后剩下五处**边界没闭上**：白名单漏了实际要改的 `taic/cli.py`；题面没给 geometry
闭集合同（agent 最易写错的文件）；track 还漏 v1 拒绝、注入执行器核对与 `eval-v2` 入口；请求构造前的
**超限**路径仍是普通 `EvaluationError`；发布读盘的 `raw.decode("utf-8")` 不在窄 `try` 里。五处都已补进
正文与验收，且都不新增任务级语义。

**v7 的结论（保留）**：v6 之后还有三处「照着写会落地失败」的细节——一处把不递归的 pack 闭包当成递归
（`selfcheck/` 根本进不了 pack）、一处把 `BASE/STEP` 当成已有符号（Host Tiling 名称求值会报未定义）、
一处只改 `self.binding` 而漏了两个读局部 `binding` 的消费者（题面与评测器仍落回 `msl-single`）。
三处都已改成可直接落地的写法，并在 B2/B3/B5-b 的验收里加了对应断言。

**v6 的结论（保留）**：v5 把发布链路、受信面和预检闭集都补上了，但它有五处「写进验收就会当场崩或越界」的东西：
一个不存在的字段（envelope 的 `track_version`）、一处漏掉的 `None` 路径运算（`@catalog` 的日志行）、
一条证据不支持的判定（用 `workspace_probe` 判「没读表」）、一处形态矛盾的验收素材（`known_good.metal`），
以及旧配置的 resume 兼容。v6 逐条改正，并把原文已改而总结还主张旧结论的地方一并清掉。

1. **不存在的字段不能进验收**。envelope 只传输 `track_id`（`envelope.py:101-109,265-271`），
   `CandidateArtifact` 也没有 `track_version`；v5 要求 decode 出它并与 `result["track"]` 比对，
   要么无法实现，要么偷偷升级格式——而「不改 envelope 格式」是底线。所以只看 `track_id`，
   版本与 delivery 由 `result["track"]` × registry 交叉校。
2. **放宽 gate 只是第一步**。`orchestrator.py:2368,2373` 在门内就用 `self.from_run / self.best_name`
   做日志，`@catalog` 时 `from_run is None` —— 门一放宽就是 `TypeError`，端到端验收会在 prior 评测
   之前就崩。所以 B4b 要求先引入统一的 prior 来源显示名/路径，再改门、dup 豁免与读字节。
3. **承重可以证明，但「不读表」不能定罪**。`workspace_probe`（`v2_runtime.py:945`）只看最终字节，
   观察不到 device 是否读过；device 自己重算公式也是完全合法的算法。所以只保留两条真的能证明承重的证据：
   指定正例读表（`host_runs` 字节数/摘要 + stage 的 `workspace` 声明），以及「同 device + 破坏 host plan
   → correctness 必错」。不读表的候选只能「不作为本条承重证据」，**不能**变成任务级拒绝语义或
   per-task 门禁——那会直接碰撞「候选自由」这条不变量。
4. **验收素材的形态要与预检规则一致，身份要绑在真实存在的东西上**。`known_good.metal` 是 pack 文件、
   纯 MSL（`loader.py:55,64,418-432`），而 bundle 收到非 envelope 必须拒——两者放在一起就矛盾。所以它继续
   做 pack 文件并在打包时被逐字节复用（测试用 `known_good_path.read_bytes()` 与解码出的 device entry
   字节比较，于是 `contract.fingerprint_document()["known_good_sha256"]` 就是它的身份）；host 侧的 `tiling.json`/`execution_geometry.json` 作测试夹具放在
   `tests/support/`，由 B5-b 的证据 JSON 记录 sha256 绑定，**不进** task/deployment pack：`V2_STANDARD_FILES`
   与 `STANDARD_TASK_FILES` 是固定顶层清单、`task_pack_files()` 不递归（`loader.py:48-56`、`pack.py:37-45,246-258`），
   声称「塞进题目录就会进 pack」是错的。
5. **旧目录必须能 resume**。`artifacts.py:386-392` 明确放行 schema<3，所以「一律从 `track.track_id` 取值」
   会把旧 v2 目录卡死。规则改成：缺失 `track` 的 v2 旧运行按 `msl-single` 恢复，v1 仍 `track=None`，
   只有显式存在但非法/版本不符才 `ResumeRefused`；并加一条「旧配置 resume 不回归」的验收。

以下六条是 v5 已经改对、本版保留的：

1. **接线必须改在运行对象上**。resume 去 `replace` 一个只喂给 `load_snapshot` 的局部 binding 是无效的：
   真正跑的 `Orchestrator` 在 `cli.py:539-550` 重新构造、再在 `orchestrator.py:640` `bind_task` 一次。
   唯一正确的入口是给 `Orchestrator` 加 `track_id`，由 `cmd_run` 与 `prepare_resume` 分别传入，
   验收断言 `orch.binding`、真实 evaluator 的 binding、`SessionBrief.track_id` 三个对象。
2. **G13 的方向按 `_KERNEL_DECL` 的事实写**。envelope 的 body 是明文，`\bkernel\s+void\b` 会匹配，
   所以真风险是「不解码就接受」（含残缺/被改的 envelope）；而 `@catalog` 死在三处：门、dup 豁免、
   外部集合。验收要跑到「实际完成 prior 评测并建立 seed」。
3. **bridge 的验收跟 typed preflight 一致**。单文件在 bundle 会话仍能建构合法请求，于是它是 rejected
   `EvaluationResult` → `ok=true`。写 `invalid_request` 就得另写请求构造前校验；本期选前者。
4. **异常闭集按源头补**。`envelope.py:180` 的 `.decode("ascii")` 与 `HostTilingPlan` 的 `int(...)`
   是两处真实泄漏；在源头规范化成 `EnvelopeError`/`TaskContractError`，调用点只认这两类，
   不用 `except ValueError` 一把捕。
5. **发布链路缺的不只是谓词**。catalog 只存 `source_sha256`、`_archive_if_confirmed` 不传 manifest/track；
   `read_text()` 的换行归一又会让 CRLF 旧文件的 transport 哈希对不上。B4b 把 catalog 与调用方一起纳入，
   哈希一律来自 `read_bytes()`，加 CRLF 回归与 single/bundle 交叉重贴标签的负例；`track_mismatch` 是
   **追加**到原因码闭集，不改变任何旧运行。
6. **受信面要说全**。`system_prompt_v2` 与 `_check_command` 都是模型可见却不在 `_PROMPT_FUNCTIONS` 里；
   不登记，以后单独改这两个函数就能绕过指纹。B3 包含「登记进受信面 + 重签 baseline 受检摘要」。

`workspace_bytes` 断言 ABI span（`4*N`）而不是「与几何一致」，是因为后者在几何写错时会一起错、
失去判别力。

---

## 7. 实施记录与偏差（第 24 版）

实施顺序按 §5：B0 → B1 → B2 → B4b → B4 → B3 → B5 → B6。下面是每一阶段的落点、验收命令，以及与计划原文的
偏差（**遇到冲突时以当时的代码为准**）。

### 7.1 每阶段的代码与测试

| 阶段 | 代码 | 测试 |
|---|---|---|
| B0 | `track_adapter.py`（`sorted(key=path)`）、`v2_runtime.py`（`host_write_span()` 共用 helper + `from_candidate` 的 `int()`/`RecursionError` 规范化）、`schema.py`（`RecursionError` → `SchemaError`）、`envelope.py`（非 ASCII manifest 行 → `EnvelopeError`）、`evaluation.py`（`CandidateDeliveryError` + 超限/surrogate 归类）、`launch.py`（超长十进制 → `LaunchRejected`）、`search_candidate.py`（`portfolio_for_submission` / `preflight_submission` / `PreflightOutcome` / `bundle_parent_entries`）、`scripts/msl_bundle_pack.py`、`evaluate.py::rejected_report`、`v2_evaluation.py`（走预检）、`kernel_agent.py`（`SessionBrief.track_id` + bundle 交卷读字节）、`session_supervisor.py`（`delivery_rejected`）、`orchestrator.py::_record_delivery_rejection`、`tcenv/cli.py`（`eval-v2 --track`、`read_bytes()`+窄解码） | `test_search_v2_bundle_preflight.py`（32 条：九条交付负例、§1.4 两项、Host Tiling span 正负例、四类异常源头）、`test_msl_bundle_pack.py`（11 条）、`test_track_bundle_compose.py`（4 条）、`test_search_v2_bundle_session.py`（8 条） |
| B1 | 无（只加测试） | `test_search_v2_bundle_reuse.py`（8 条：投影 track、`HostTilingPlan`、lifecycle `("host_tiling","schedule")`、`inside_timing` 两条、几何权威性、三处一致性负例） |
| B2 | `orchestrator.py`（`track_id=` 参数、`_resolve_run_track`、`_track_document`、局部 binding 重建在 `self.binding` **之前**、注入 evaluator 核对、`_failure_record.retest_command`）、`cli.py`（`--track` 两个 parser、`_requested_track`、`_recorded_track`、`cmd_run`/`prepare_resume` 透传）、`kernel_agent.py`（`_check_command --track`） | `test_search_v2_bundle_track.py`（21 条：三对象断言、启动拒绝、优先级、resume 判定表、两条命令、G21 端到端正负例 + 落盘 + 空 source 不进 cache） |
| B4b | `experiments/common.py`（`_track_binding`、`_manifest_sha256`、`_read_bytes`、`_winner_kernel_path`、`_v2_conclusion_verdict` 双身份绑定）、`policy.py`（`track_mismatch`）、`kernel_catalog.py` + `_archive_if_confirmed`（transport/manifest/track）、`orchestrator.py`（`_prior_source_path/_prior_source_label/_read_prior_source`、`_search` 门、dup 豁免、`read_bytes`） | `test_search_v2_bundle_publish.py`（19 条）、`test_publication_reason_closed_set.py`（3 条） |
| B4 | `search_candidate.confirmation_input/confirm_candidate(track_id=)`、`_confirm_v2` 传 track、`V2LocalEvaluator.confirmation_expected` 走同一包装 | CPU：`test_search_v2_bundle_reuse.py` 与 publish 用例；GPU：`test_search_v2_bundle_gpu.py::test_the_confirmation_freezes_the_bundle_subject_and_replays_the_same_dispatch`（1 次 search + ≤3 次 confirmation，逐 run 断言 `dispatch`/`dispatch_stages` 与 `variant_id`） |
| B3 | `prompts.py`（`_rules_for(backend, delivery)` 的替换句、`system_prompt_v2` 的 bundle 段、清零句纠错）、`kernel_agent.py`（`render_contract` 的 bundle 分支 + 中文清零句）、`task_binding.py::signature_stub`（bundle 保留 workspace 槽位）、`trust.py`（点分路径 + 三个新登记项）、`docs/架构/backend-phase0-baseline.json` 重签 | `test_search_v2_bundle_prompt.py`（17 条：两个渲染面、签名骨架、受信面 blob 覆盖、三次 `inspect.getsource` 判别力、反向用例） |
| B5 | 新题 `tasks/affine_ramp/`（`opspec.json`/`reference.py`/`baselines.{json,py}`/`known_good.metal`）+ `tests/support/msl_bundle_affine_ramp/`（host 夹具 + `evidence.json`） | `test_affine_ramp_task.py`（15 条 CPU）、`test_search_v2_bundle_gpu.py`（8 条 GPU，本机全绿） |
| B6 | 本文、`v2协议参考.md`（L.1 bundle 交付 + `track_mismatch`）、`搜索-v2接线.md`（§8 接线）、`数据层.md`（原因码）、`experience.md`（一条 `reported`） | `test_publication_reason_closed_set.py`、全量快速套件 |

### 7.2 实测验收

```
uv run pytest -q -p no:randomly tests/backends/metal/test_search_v2_bundle_preflight.py      # 32 passed
uv run pytest -q -p no:randomly tests/backends/metal/test_msl_bundle_pack.py                # 11 passed
uv run pytest -q -p no:randomly tests/backends/metal/test_track_bundle_compose.py           #  4 passed
uv run pytest -q -p no:randomly tests/backends/metal/test_search_v2_bundle_reuse.py         #  8 passed
uv run pytest -q -p no:randomly tests/backends/metal/test_search_v2_bundle_publish.py       # 19 passed
uv run pytest -q -p no:randomly tests/shared/test_search_v2_bundle_track.py                 # 21 passed
uv run pytest -q -p no:randomly tests/shared/test_search_v2_bundle_session.py               #  8 passed
uv run pytest -q -p no:randomly tests/shared/test_search_v2_bundle_prompt.py                # 17 passed
uv run pytest -q -p no:randomly tests/shared/test_affine_ramp_task.py                       # 15 passed
uv run pytest -q -n 0 -m gpu tests/backends/metal/test_search_v2_bundle_gpu.py              #  8 passed
```

B5-b 的三段自测（真实 Metal）：

```
uv run python scripts/msl_bundle_pack.py --src <work>/selfcheck-src --out <work>/selfcheck.envelope.metal
uv run python -m tcenv.cli eval-v2 affine_ramp <work>/selfcheck.envelope.metal --dtype fp16 --size small --quick --track msl-bundle
  -> STATUS ok / compiled, ABI ok, safe and correct（Host Tiling 真跑，workspace 回读）
uv run python -m tcenv.cli eval-v2 affine_ramp <work>/selfcheck.envelope.metal --dtype fp16 --size paper --timed --track msl-bundle --reps 5
  -> candidate 22.14 µs (cv 10.5%) / torch_mps 83.45 µs / 3.769x
```

**第三段（真实搜索 + 确认的 published speedup）未跑**：那是付费动作，需要单独许可。本题的
`evidence.json` 里 `stage_3_search.status = "not_run"`，`experience.md` 的对应条目是 `reported` 而不是
`verified`，不计入 13 条 verified。

### 7.3 与计划原文的偏差（以代码为准）

1. **§2.2 的 bundle 父包文档不需要 `fallback` 键**。计划写「harness 生成：单 variant、恒真 guard、
   `fallback=None`」，但 `candidate-portfolio.v1.json` 的 `fallback` 是 object 类型（不接受 `null`）且非必填，
   所以实现是**省略该键**（与 single 分支一致），语义仍是「没有 fallback」。
2. **`_symbol_bytes` 的点分路径落地成 `_resolve_symbol`**（新增 helper），而不是在原函数里内联；语义与计划一致：
   只哈希 `TaskBinding.signature_stub` 的方法源码，标签解析失败抛 `ValueError`。
3. **§2.6 判定表新增一行实现细节**：envelope 形状但**解不开**（非 UTF-8、非 ASCII manifest 行、未知 role）
   在 `_track_binding` 里返回 `sha_mismatch`，不是 `track_mismatch`。理由是按计划 B4b 验收里明确的
   「三者都返回 `(未确认, sha_mismatch)`」，与 §2.6「读盘异常闭集 → `sha_mismatch`」一致；`track_mismatch`
   只用于**能解开但身份/形态不符**的行。
4. **`EvaluationRequest` 的超限与 surrogate 归类**：`CandidateDeliveryError` 继承 `EvaluationError`，
   所以既有 `except EvaluationError` 的调用方行为不变；新增的窄捕获只在三个消费点。
5. **`tcenv/evaluate.py`、`tcenv/evaluation.py`、`tcenv/track/envelope.py`、`tcenv/schema.py` 都在
   deployment code closure 里**（`task_pack_closure(code_import_closure)`），改它们会改变**每个**任务的
   `deployment_pack_digest`。实施前已核实：`tests/support/golden/shared/shape-portfolio-阶段验收/P2/evidence/` 里记录的
   `portfolio_freeze_digest`（`a9ce4c31…`）在当时的 HEAD 上已经不能复现（当时算出来是 `20a89efd…`），
   把这些文件的改动全部还原后仍然不能复现。那一批 `test_shape_portfolio_manifest.py` /
   `test_shape_portfolio_resume.py` 失败是本次改动之前就存在的，不是本计划引入的回归。
   后来的 P2-F 已在新 epoch 重录；现行 `tests/support/golden/shared/shape-portfolio-阶段验收/P2/golden.json` 的 `portfolio_freeze_digest` 是 `64afe993…`。本文不改已关闭证据。P3-C 的最终验收已在之后完成
   （`tests/support/golden/shared/shape-portfolio-阶段验收/P3/C.md`）。
6. **`Orchestrator(_session_brief)` 的 `track_id` 在 `SessionBrief(...)` 构造里赋值**（计划写「B2 填值」），
   字段默认值是 `"msl-single"`，这样旧调用点（测试直接构造 brief）不需要改。
7. **B5-a 的多阶段正例需要两条 device entry 里只有一条带 `taic-launch` 块**：`unique_launch_block` 在合并
   源码后检查唯一性，所以两源各带一块会被预检拒（这正是 §1.4 的负例）。
8. **§2.0 的 `search.geometry_mode` 没有实现。** msl-single 缺 `/* taic-launch */` 时由
   `tcenv/backends/metal/track_adapter.py` 拒绝，harness 不派生 grid 或 threads。bundle 的 dispatch
   来自 `host/execution_geometry.json`。
9. **§2.1 把 `track_id` 写成只有 `msl-single` / `msl-bundle`。** 现行 `backends/metal/tracks.json` 还有 `tilelang-pipeline`，delivery 为 `dsl_source_pipeline`。新 run 的优先级仍是 CLI `--track`，然后 `taic.toml` 的 `[search].track`，再然后 `msl-single`。`--track` 的可选项来自 `load_track_registry().available()`。
