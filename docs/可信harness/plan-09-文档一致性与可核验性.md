# plan-09：文档一致性与可核验性

状态：设计计划，尚未实施；核对日期：2026-10-26。本轮只改 `docs/可信harness/` 下的 Markdown，未改代码、配置、PDF 或测试；未运行 `taic`、模型或 GPU。下文的新文件、命令和检查均为规划，不能写成「已通过」。

## 当前决定

- **本批只做**：相对链接、README 与 `plan-NN` 集合对账、(a)–(d) 文档修正，以及 `MAX_OVERLAY`、`TASK_MAX_BYTES`、评测 `PROTOCOL_VERSION`、当前委员会 provider/model 和各 flow `_permission` 与 `docs/security.md` 的窄核对。
- **输入**：本目录 Markdown、README/plan 文件集合、`taic.toml` 与四个常量所在源码。
- **本批不做**：成百条 `path:line` 的 blocking/默认 warning、联网检查、大型 PDF registry 或自动改写；`AGENTS.md` 的模型名漂移作为单独文档修正提出。
- **完成产物**：只读检查报告和少量明确文档 patch；围栏写面必须按实际实现解释，不能只读 `ALL` 字样。行号漂移只供人审，冻结设计移入 `archive/` 减少维护面。
- **失败去向**：先读 `logs/pytest/triage/triage_summary.md` 及 handoff；checker 未实现前不得把规划命令写成已运行。

## 1. 角度与问题

文档是这个方向的「合同」：代理从它读路径、开关和安全边界，人从它判断哪些数字已经验证。问题在于文档、代码和配置会各自漂移；一个相对链接仍然存在，不代表链接文字、路径层级、行为声明和配置值仍然正确。
本方向目前没有一个覆盖 doc↔code/config、claim 出处和链接/路径的统一自动检查。已有的局部范例是把测试布局写成断言（`tests/shared/test_structure.py:24-80`），以及要求 `verified` 经验带来源、实测和边界（`tests/shared/test_wiki_experience.py:42-70`），应扩展这种思路而不是另造文档信仰。
本次在 `scripts/` 与 `tests/` 未找到通用 Markdown/doc 漂移检查器；目录合并后通过的相对链接检查也没有在仓库内找到可重复命令。外部 CI 是否另有检查：**未验证**。因此「曾经人工检查过」不能冒充常规回归。

## 2. 目标 / 非目标

| 目标 | 非目标 |
| --- | --- |
| 给当前实现、设计基线、历史快照和未验证声明不同的核验规则 | 把所有散文改成模板，或让正则判断自然语言真伪 |
| 先做只读报告和具体修正清单，再增加离线检查，最后进常规 CPU 套件 | 本计划直接批量改文档、改运行配置或改写历史 run |
| 对链接、路径、符号、关键常量和配置快照提供可追溯检查 | 默认联网查外链、自动下载文献、调用模型或使用 GPU |
| 给数字、行为声明和论文引用记录来源、版本与日期 | 用检查通过替代行为测试、人审、证据签名或效果实验 |
| 建立本文件夹和论文索引的轻量维护例程 | 把历史模型名、旧阈值和历史结果全局替换成当前值 |

## 3. 现状证据与具体修正清单

### (a) pi session 路径多了一层
- 现状：`docs/架构/hmz.md:63` 写 `<epic>/<场次>/sessions/pi/sessions/`；实际布局由 `humanize@19b3dc59:src/hmz/runtime/epic.py:145-146,902-903` 与 `src/hmz/coganchor/backends.py:2671-2676` 合出，没有 per-turn 层级。`observe/detail.py:866` 的 docstring 同样多余一层，而 `observe/detail.py:875` 的 glob 正确。
- 建议文本：「pi 会话位于 `<epic>/sessions/<cli>/sessions/<workspace>/<started>_<id>.jsonl`；当前 `cli=pi`，没有按场次再嵌一层。」后续只改 `hmz.md` 与该 docstring 的说明，保留正确 glob；其他 hmz 版本未验证。

### (b) 把 `hmz web` 写成「只读」
- 现状：`docs/架构/hmz.md:85` 的「只读 Humanize 的会话视图」可被理解为服务只读，但 `docs/架构/可观测.md:233` 已说明它能启动或 steer epic；源码还有 start/stop 与设置 POST（`humanize@19b3dc59:src/hmz/web/held.py:86-100,753-790`、`src/hmz/web/settings.py:123-127`）。
- 建议文本：「它只把 epic/session 作为辅助展示的数据来源，不读搜索候选、确认门或 trust；但服务本身不是只读，含启动、停止、steer 和设置写入。loopback、每次启动 key、同源检查与 SameSite cookie 是防线，不是绕过 `taic` 的授权；会启动真实运行的页面操作需人工授权。」

### (c) `online=NONE` 易被读成整场物理离线
- 现状：`docs/security.md:20` 只说「不提供 web 搜索/抓取」；实际是在 CLI 之下切网，并由 loopback proxy 只放行白名单主机，模型 API 仍需可用（`humanize@19b3dc59:src/hmz/coganchor/fence/proxy.py:1-18,131-156`）。
- 建议文本：「`_permission.online=NONE` 表示禁止 CLI 直接外连，仅经 loopback proxy 访问模型/登录相关白名单主机；不提供通用 web 搜索/抓取，但不等于整个 session 物理离线。」能否经允许主机或本机服务外传内容仍标 **未验证**，不写成绝对防泄漏。

### (d) 缓存「不可写」与「重定向」混为一谈
- 现状：`docs/架构/hmz.md:62` 写 `~/.cache` 和 Clang 默认模块缓存不可写；围栏实际把不能写原位置的缓存环境变量放进本场 scratch（`humanize@19b3dc59:src/hmz/coganchor/fence/wrap.py:64-75,95-116`），这是更精确的声明。
- 建议文本：「`TMPDIR` 指向本场临时目录；相关缓存环境变量按可写性重定向到本场 scratch，包括文档所称的默认缓存/Clang 模块缓存场景。」不要由此反推原路径一律可写或直接路径写能力已全部验证；这些另列 **未验证**。

### (e) 委员会配置与「当前默认」文档漂移
- 现状：至少三处仍把 second 写成 `kimi-k3`：`docs/README.md:7`、`AGENTS.md:16`（项目指令）与历史试点报告。当前配置是 `kimi-code/k3.1-pro`（`taic.toml:52-57`），而 `targets/apple-m4-max/trust.toml` 的 grant 是 `moonshot/*`。三处漂移的后果不同：README/AGENTS 只是描述错；trust 的字面量不匹配是真实的 fail-closed 拒绝（归 plan-03 处理，含它发现的 task 维度问题）。这是检查设计的样例，本计划不现在修，也不把历史试点里的 `kimi-k3` 当错误。
- 建议文本（仅用于描述当前的页面）：「委员会以 `taic.toml` 的 `[[committees.members]]` 为准；截至 `<实际核对日期>`，second 为 `kimi-code/k3.1-pro`。」历史报告保留当时值并标快照日期；检查不得全局禁用旧模型名。配置名与授权名的映射只能由人批准的别名解决（plan-03），不能在文档里把一个名字直接当成另一个。**可立即单独提交的一行修正是**：把 `AGENTS.md` 的 second 描述改为上述「以 `taic.toml` 为准 + 核对日期 + 当前值」，并注明 grant 字符串不自动等于部署身份；该行由人单独审，不等待 checker，也不在本轮批量改写。

### (f) 下载失败文献只有简短备注，缺少可持续状态
- 现状：`docs/可信harness/papers/导读.md` 记录「Trustworthy LLM Harnesses…」预印本因 HTTP 403 未下载，并声明未经同行评审；这是正确起点，但后续补取、版本和复核状态没有统一字段。
- 建议追加文本：「全文状态=未获取；最近尝试=`<实际日期>`，HTTP 403；核验状态=未读原文，仅作书目线索。」合法取得后记录实际来源 URL、版本、下载日期、文件名和 SHA-256，再改状态；未读前不得据标题转述其框架或数字。

## 4. 可核验性的分层

| 层 | 机器能可靠检查什么 | 必须保留的人审/边界 |
| --- | --- | --- |
| Markdown 链接与图片 | 相对目标存在、URL 解码、越界、`file` 与 anchor 分离 | 链接文字是否贴切；外部 URL 默认只查语法，不联网判活 |
| 行内路径与文件存在性 | 登记的 repo 相对文件/目录、行号范围、显式 `--run-config` 文件 | `<id>`、`RUN_DIR`、命令输出、已归档产物不能一刀切判错 |
| 术语与开关名 | 登记词表、TOML key、CLI flag、配置布尔值是否仍存在 | 「只读」「离线」「默认」的语义不能只靠关键词 |
| 常量与配置快照 | AST/TOML 提取值，和登记文档值比较；区分 base、paper 与历史 run | 文档是否选对了该写的层级，历史差异是否合理 |
| 文件中的代码符号 | Python AST 中函数/类/赋值符号存在，无需 import 或执行 | 符号存在不证明调用链、版本部署或行为正确 |
| 行为声明与代码 | 可核路由/常量/测试引用是否存在并记录版本 | 「不可写」「无网络」「会启动」须看实际实现；必要时另做离线行为测试，外部版本未测就标未验证 |

## 5. 具体机制建议

### 5.1 离线 checker 的放置与规则
- 核心放 `taic/doc_audit.py`，只用标准库读文本、AST 和 TOML；`scripts/check_docs.py` 只做 CLI。两者与 `docs/doc_audit.toml`、`tests/support/doc_cases/` 都是规划产物，当前不存在。默认 stdout，退出前不改文档、不刷新 SQLite、不读凭据、不联网。
- backend 无关断言放 `tests/shared/test_doc_integrity.py`；最小 Markdown/TOML/AST fixture 放 `tests/support/doc_cases/`，loader/helper 才可放 `tests/support/`，support 不放 `test_*.py`。只有 Metal 专属文档行为才进 `tests/backends/metal/`，遵循 `tests/README.md:71-99,116-152`。
- 静态检查不加 `gpu`/`llm` marker；先按快速 CPU 用例设计。是否需 `slow` 必须由实测决定，不能凭文档数量预设（marker 规则见 `tests/README.md:154-190`）。
- 全量严格检查真实 Markdown 相对链接；忽略 fenced code、行内 code、HTML comment 中的示例链接。anchor 在 slug 规则未固定前只 warning；外链不默认请求。
- **索引与目录双向对账。** `docs/可信harness/README.md` 第 2 节的 plan 链接集合必须等于 `docs/可信harness/plan-NN-*.md` 的实际集合（历史文件 `human-in-loop…plan.md` 不算 plan-NN）；缺一行、多一行、文件名或编号不符都是 error。README 的 plan 索引只登记链接与一句话摘要，不复制 plan 的结论数字。`papers/导读.md` 则是唯一书目与评述来源，其 §2 详介是数字与口径的权威位置；§1 速览和 §3 机制表若重复结论，必须能定位到同一详介，不得在索引单独新增或改写数字。checker 对索引独有或与详介不一致的数字给 warning，而不是禁止导读写评述。
以下 `repo_current`、外部 checkout、引文 lint 与快照规则均为后续，不属本批：

- 行内 `path:line` / `path::symbol` 先只检查 `docs/doc_audit.toml` 登记的 claim。路径或符号消失是 error；行号漂移只 warning 并建议新行号，避免无关插入造成文档锁死。
- **外部 checkout 的写法。** 本仓库文档会引用 sibling checkout（如 `humanize@19b3dc59:src/…:line`、`kda-260927-kda/bench.py:55`），这些路径不在本仓库内，链接/路径检查器无法直接解析。登记这类 claim 时必须用 `external` 类，并把 `locator` 写成 `../<checkout>@<rev>:<repo相对路径>:<line>`；只有 sibling checkout 存在且 HEAD 与 `<rev>` 相符时才实际比对，否则报 `unverified` 而不是 error。未按此格式写的外部路径一律报「无法定位」warning，禁止当成本仓库文件去判错。
- manifest 每条 claim 必含 `claim_class=repo_current|repo_historical|external|design|theory`、`source`、`source_version`、`locator`、`checked_at`、`scope`；`external` 另含 `evidence=author_report|secondary_report|author_repetition|local_reproduction|independent_reproduction`，缺项即为 `unverified`，不得写成当前事实。
- 论文引用可登记 `citation_locator`：`section|appendix|table|figure` 的标签集合，取自离线抽取的标题/TOC（`pdftotext` 文本即可，不联网），用于**可选的 dangling-label lint**：登记的标签在文中找不到报 `citation_unresolved`，只报告、默认不阻塞。**它只能发现「引了一个不存在的标签」**；对「标签存在但内容不支持结论」完全无效——本轮查出的 RRSI `B.6`/`E.2`、Harmony `App. C`/`K.6`、HarnessDev `§5` 都真实存在，都会被它放过。高影响引文仍须人审抽查：`claim → 本地 PDF 文件名 + 节/表/页 + supporting passage`，并记录复核人与日期。
- `theory` 另含 `assumptions_locator`；综述的 corpus 数量/比例另含 `corpus_locator`、检索窗口与 cutoff。`verified_by_checker` 只表示字段和可提取值已核对，不能把作者报告、形式论证或 PDF 哈希升级为本地/独立复现；`independent_reproduction` 仅用于有独立主体与可核 run 的第三方复现。

### 5.2 常量单源化、快照与 run_config 漂移
- 代码/配置是当前值的唯一权威；manifest 只登记「文档定位 + 提取规则 + 源码定位 + claim 类型」，不再复制第三份 expected 值。新表格可直接生成，存量叙事用快照测试，不自动覆盖散文。
- 首批登记：`MAX_OVERLAY=1800`（`taic/harness.py:86`，字符）、`TASK_MAX_BYTES=12_000`（`taic/kernel_agent.py:56`，UTF-8 字节；「12KB」不能解释成 12×1024）、评测 `PROTOCOL_VERSION="3"`（`tcenv/schema.py:16`）与当前委员会的 provider/model（`taic.toml:52-57`）；远程协议 `"1"`（`tcenv/remote/protocol.py:21`）只作区分说明。预算、preset 合并、run_config 与外部 benchmark 快照均为后续，不在本批验收。
- 预算必须写层级：paper 常量 `15.0`（`taic/llm.py:212,286`）、base 配置 `50.0`（`taic.toml:17`）、`relay_budget_usd=2.0`（`taic.toml:19`、`taic/llm.py:323`）。这些是当前源码事实，不是本计划新测的性能或成本数字。
- 预设合并只复用纯文件函数 `load_raw_config`（`taic/llm.py:348-377`）；不得为文档检查调用会读凭据的 `load_config`（`taic/llm.py:381-408`）。
- `--config-drift` 分别报告 base/paper 当前配置与登记文档的差异；显式 `--run-config RUN_DIR/run_config.json` 才把某次冻结 run 与其历史文档对账。机器先报 `match / mismatch / expected_historical_difference / unknown`；`doc_stale` 与 `config_changed` 只能由人结合日期和 git 历史判定，不让 checker 猜原因。绝不把旧 run_config 改成当前值，也不默认扫描 `runs/`。
- 引用外部 benchmark 数字须有关联快照：`benchmark@version/split`、`task_role=development|feedback|heldout|OOD`、metric/完整分母与失败处理、harness/seed commit、model route、evaluator/judge、`evidence` 和原文定位；缺项只能标 `unverified`。
- 用于横向比较或 SOTA 时另需 `comparison_scope=paired_local|same_window|external_reference`、trials/aggregation（原文未充分说明时字段值写 `unclear`，正文标「未充分说明」，报告状态仍只能是 `unverified`，不得升级为 `verified_by_checker`）、评测窗口、model effort/decoding、budget/timeout/step cap、memory/delegation/network/evolution 开关；Creation/Evolution 还必须分别登记 creator-or-proposer、development env、executor 和 frozen artifact。比较字段不全时只能单列「作者报告」，不得排名。
- token/cost 必须声明 creator/proposer、executor、judge、probe 的包含范围；checker 只核对快照字段与文档值/来源，不声称复现实验，也不证明不同论文的配置可比。

### 5.3 误报控制与不锁死文档
- blocking 范围只含真实相对链接和显式登记的 `repo_current` claim；其余先报告。placeholder、历史快照、外部版本、设计示例分别用机器可读状态豁免，豁免须带原因与复核日期，禁止全局忽略某个词或某篇文档。
- 报告区分 `verified_by_checker`、`stale`、`unverified`、`not_applicable`；未检查/未联网不能计入通过。允许人修改措辞，只要登记的机读值、来源和状态仍一致。
- 代码行号、章节号、篇名和数字变动应输出修正建议而非自动 rewrite。任何自动生成都只限明确标记的生成区，生成区外的人审结论不动。

## 6. 文档写作规范与新增 plan 模板

1. 每篇文档在文首标类型，并在正文明确区分 `设计基线 / 当前实现 / 未验证`；历史快照须标快照日期，混写时必须分节，不把目标句写成已上线。`docs/可信harness/架构与理论.md:411-419` 的「设计基线｜当前实现｜风险」是既有范例。
2. 事实性数字（阈值、金额、性能、样本数、测试计数）须随句或表注给出处与日期；章节编号不在此列。当前常量给 `path::symbol + 核对日期`，实测给 `run/result 或命令 + 日期 + 范围`，外部数字给论文版本与页/节。没跑就不填结果，不根据旧测试数推算当前数。
3. 统一术语：`默认` 要指明 base/paper/有效 run；`当前` 指当前 checkout 并带核对日期；`只读` 要分清数据来源与服务路由；`离线` 要分清无 web 工具、网络围栏和不联网检查；`已验证` 只表示所列检查在所述范围通过。
4. 论文条目至少写标题、作者/机构（未知就标未核）、持久 ID/URL、版本与版本日期、出版状态、检索日期、全文位置/未获取状态；本地 PDF 另写实际 `source_url`、下载日期和 `sha256`，引用结论或数字另给页/节。预印本固定写「预印本，未经同行评审；引用前回原文」，有正式 venue 也须给官方来源，不凭 arXiv 猜测。
5. 新文件名用 `plan-NN-简短中文标题.md`，NN 从两位开始；取本文件夹现有最大编号加一，编号由维护者登记后创建，不复用、不覆盖，历史 `human-in-loop…plan.md` 不占序列。并发冲突由后创建者换号，不能改旧 plan 来让路。
6. 转述 acceptance/reliability 时回原文定义，分开 `evaluated/admissible/selected` 与 evaluation uncertainty/search repetition；作者重抽随机性重复搜索只记 `author_repetition`，不得写成第三方复现。`显著/稳定/复现` 必须带对应区间、重复次数和范围，缺证不得使用。
7. 引用 raw/audited 两个口径时须并列审计者、覆盖范围、调整规则和分母，不得互换；「审计未发现」不得改成「不存在」。`SOTA/最佳` 只写成带比较集合和截至日期的作者/榜单声明，不能省略 `author_report` 或 `secondary_report`。

新 plan 最小模板如下；各节没有内容时写「无/未验证」，不得删除状态边界：

```text
# plan-NN：标题
状态：设计计划（不是当前实现）；核对日期：YYYY-MM-DD；默认离线。
## 当前决定（本批做 / 不做 / 完成产物 / 失败去向）
1 目标与非目标
2 现状证据（稳定符号优先，path:line 只作当前快照）
3 分阶段步骤（每阶段输入、产物与退出条件）
4 离线验收（只列该阶段真实可运行项）
5 风险、回滚与其他 plan 边界
附录：文献与变更索引（可选；不重复正文）
```

本文件夹后续可增加索引，登记每篇编号、角度、状态和最近核对日期；`docs/README.md` 只做导航，不复制各 plan 结论。代码/配置变更与受影响文档应在同一提交更新，或显式留下带日期的 `stale/unverified` 修正项。

## 7. 论文索引与资料维护（后续，本阶段不实施）

- 补下载按 DOI/出版方或预印本官方页、作者公开主页、机构库的顺序人工查找；记录每次来源与日期，不绕过 403、验证码或付费墙。仍不可得就保留「未获取/未读原文」，不能用标题或二手摘要冒充已读。
- 补下载是单独联网操作，不属于默认 checker；如需付费取得资料则 **需人工授权**，绝不作为验收步骤。PDF 取得后计算 SHA-256 并记录文件版本；哈希只证明字节一致，不证明来源、同行评审或内容正确。
- `docs/可信harness/papers/导读.md` 保持唯一的书目与评述来源，**计划新增**机读 registry 字段（当前 `导读.md` 没有 `translations`/SHA 字段，checker 也尚不存在，见 §1），速览链接不算第二书目；计划的 checker 双向核对详介记录与 `papers/` 根目录 PDF：登记文件缺失、未登记文件或同一文件对应多条详介均报 error，并校验已登记 SHA。`<id>_<name>.zh.pdf` 按 basename 归到同一篇的 `translations`，不单独要求详介，但必须能在 registry 里找到对应英文条目并登记译本 SHA；没有英文原文的译本不能被当成「已读原文」。`2026-10-25` 的索引快照（首批 17 篇，后追补 harness 测量 3 篇与 RSI 专题 14 篇，共 34 篇英文原文 PDF + 中译若干（截至 2026-10-25 21:36 为 34 篇中译、共 68 个 PDF，以 `find docs/可信harness/papers -maxdepth 1 -name '*.zh.pdf' | wc -l` 与 `find docs/可信harness/papers -maxdepth 1 -type f -name '*.pdf' | wc -l` 为准；中译与总数是变动量）+ 2 篇博客；中译不计入篇数），不是永久定额。
- 文件名中的 arXiv ID 必须等于条目持久 ID；同 ID+version 对应多个 SHA 或未标版本时不得静默覆盖，转人审。PDF 存在、可读取或 SHA 一致只证明工件登记，不证明版本内容、来源、同行评审或本仓库复现；不要再手工维护第二份同义清单。
- PDF 入库逐篇权衡：公开许可、需要离线逐页引用、版本固定的原文适合入库；许可不清、重复译本、体积压力大或无人引用的文件可只留持久链接、版本、检索日期和已知 SHA。受限资料与凭据不得入库；删除当前文件也不会清除 git 历史。
- 维护例程：合并前跑离线检查；每月一次人工只读漂移报告并复核失败下载/新版本。发现新版时新增版本记录，不静默覆盖旧版引用；是否改采用某篇结论仍由人审。

## 7.1 新增文献后的窄核对（2026-10-27）

本节只记录已经完成的文档修正与后续核对点，**不启动 registry、自动下载或大规模 citation lint**。

- `papers/` 现有 **50 篇英文原文 + 34 篇中译，共 84 个 PDF**；2026-10-27 新迁入 16 篇英文原文，已在[导读](papers/导读.md)的速览与 §2I 分组登记，新增中译标为待补。数量是当次快照，不是永久定额。
- `docs/paper/` 是公众号与 NPU 补充资料的下载索引，不是可信 harness 的第二书目；可信 harness 的书目仍以 `papers/导读.md` 为唯一来源。`论文价值评估.md`、`NPU与昇腾资料.md` 和 `LLM与NPU-DSL：为什么被选中、如何为LLM设计.md` 都是内部学习材料，不能作为独立外部证据。
- NKI-Agent、NPUEval、JAXBench 目前只有 arXiv HTML/阅读记录，**没有本地 PDF，不得伪造本地链接**。新增引用必须区分“本地 PDF”“arXiv HTML”“本地工程/文档”“二手博客或笔记”。
- CANNBot-DSL 的公开仓是样例/文档/精度测试，后端由 wheel 提供；AscendCraft 截至检索未找到官方实现；CAKE、CANNBot-CAKE、CANNBot-DSL 是不同对象，书名相近不能合并引用。
- 当前可重复核对仍是相对链接、表格列数、fenced code、PDF 是否登记及是否存在。链接存在或 PDF 可读不证明论文结论、版本、同行评审或本仓库复现。

## 8. 分阶段步骤

- **Phase 0（本批）：窄检查与明确修正。** 做相对链接、README/plan 集合对账、(a)–(d) 文档修正、四个窄常量/身份核对，以及各 flow `_permission` 与 `docs/security.md` 的边界对照；`AGENTS.md` 的模型名漂移单独提出文档 patch，不等待 checker。冻结设计只保留归档链接，不把成百条行号当默认 warning。
- **后续阶段。** 登记的 `repo_current` 符号、run_config、PDF registry、citation lint、CI 纳入和联网资料维护分别评审。成百条 `path:line` 不进 blocking，也不进默认 warning；只保留稳定符号与当前快照供人审。

## 9. 验收标准（均为离线命令）

本批只验收：中文/空格相对链接、fenced code 假链接、README 少/多/错登 plan、四个窄常量与当前 provider/model、(a)–(d) 的建议文本。`AGENTS.md` 漂移单独处置，历史模型名不做全局替换；行号漂移不进 blocking 或默认 warning。失败先读 `logs/pytest/triage/triage_summary.md` 及 handoff。

```bash
# 最小 checker 实现后才可运行
uv run python scripts/check_docs.py --root . --format text
uv run pytest -q -n 0 tests/shared/test_doc_integrity.py tests/shared/test_structure.py
```

PDF registry、`citation_locator`、run_config drift、外部 benchmark 快照和联网检查均为后续；不得把规划命令写成已通过。

## 10. 风险、回滚、边界与参考

- **误报/锁文：** 正则可能误读 Markdown，行号必然漂移。以真实链接 + 登记 claim 限界，行号只警告；回滚时从常规套件移除新测试、保留只读 CLI，不恢复已知错误文案。
- **单源变多源：** manifest、生成表和散文可能再次漂移。expected 只从代码/配置或历史文件读取；回滚生成区即可，不自动改人审文本。
- **虚假保证：** 路径存在、hash 一致或「有测试引用」不证明行为/论文结论为真。报告必须显示核验层级和未验证项，不能用全绿替代安全或效果证据。
- **资料风险：** PDF 可能带来版权、体积和陈旧版本问题；入库前审查，回滚只移除当前不合适的二进制并保留书目，不假装能抹掉历史。
- **边界：** plan-01/02 管验收行为和效果，plan-03/04 管授权与执行证据，plan-05/06 管评审质量与运行时安全，plan-07/08 管知识生命周期与采纳成本；本文只修文档声明、出处和可重复检查，不改变那些机制，也不把 claim provenance 冒充 plan-04 的证据签名。

## 附录：文献与变更索引

只作查阅，不再逐轮重述正文；论文数字均未在本仓库复现。

| 日期 | 影响位置 | 证据文件 |
| --- | --- | --- |
| 2026-10-25 | 文献对应的正文规则与边界 | `2609.01437_harnessdev.pdf`、`2608.08311_ouroboros-reviewed-core-evolution.pdf`、`2609.24972_rrsi-regularized-harness-rsi.pdf`、`2607.07663_rsi-survey-bounded-to-autonomous.pdf` |
| 2026-10-26 | 当前决定、阶段、验收与跨 plan 归属 | `审核意见-plan-01至09.md` |
| 2026-10-27 | 新增文献后的窄核对 | 50 英文 + 34 中译 = 84 PDF；补书目登记、HTML/本地 PDF 区分与内部材料边界，不启动 registry |
