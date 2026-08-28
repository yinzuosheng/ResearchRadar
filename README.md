# 水色遥感预测科研助手

这是一个面向秋招展示的两周制硕士个人 AI 应用：围绕水色遥感与水质参数预测，构建合法开放文献的本地知识库，并提供科研对话 Agent、带证据问答、研究路线、可恢复种子采集和可复现实验评估。项目强调可解释、可验证的工程闭环，不把联网搜索或模型输出包装成事实。

## 架构与数据流

```text
检索词 → OpenAlex 发现 → Crossref 元数据校正
                    ↓
          Unpaywall OA DOI 解析 → CORE 仓储回退
                    ↓
        PDF 安全下载/校验 → 按页解析 → 证据分块
                    ↓
      SQLite 元数据+FTS / 向量索引 / 结构化论文画像
                    ↓
       混合检索 → 引用校验 → 证据问答、研究路线
                         ↑
       有界 LangGraph Agent → evidence_qa / research_plan
       最近 8 条对话 + 摘要 + 结构化研究范围
```

Streamlit 使用左侧两组导航组织科研工作台：工作台包含科研助手、知识库、文献库和研究方案；维护包含知识库维护、任务与日志和数据源设置。聊天只在科研助手页面的中心工作区出现，不占用其他页面。

数据源职责明确：OpenAlex 用于发现候选文献；Unpaywall、Semantic Scholar（仅在配置 `SEMANTIC_SCHOLAR_API_KEY` 时启用）和 CORE 作为 DOI 级开放全文回退；Crossref 只做 DOI、标题、作者等元数据校正。系统只下载明确可合法开放访问的内容，不绕过付费墙、不抓取需要订阅或登录的全文。

## 核心技术主线（面试口径）

本项目的核心技术主线是：**受控科研 Agent + 证据型 RAG + 可审计后端工作流**。完整路径为：合法 OA 摄入 → 证据分层 → SQLite/FTS5 与 FAISS 混合检索 → 引用校验 → 结构化回答。

论文画像和 Streamlit 导航属于辅助能力，用于展示工作流复用，不作为独立平台模块宣传。当前本地快照中，704 是元数据目录，116 是可定位全文；摘要证据单独标记。未完成真实人工标注和对应运行前，不写 Recall、MRR 或问答质量百分比。

这是一个个人 MVP；Semantic Scholar 只作为本轮唯一新增的单一 DOI 全文 fallback，后续不新增全文 provider、OCR、FastAPI、多用户或云部署。优先保证证据链、失败恢复、边界控制和可复现评估能够经得住面试追问。

## 本地安装与配置

推荐 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在本地 `.env` 中填写以下名称，README 和版本库不提供任何示例密钥：

```text
OPENALEX_API_KEY
CORE_API_KEY
UNPAYWALL_EMAIL
SEMANTIC_SCHOLAR_API_KEY
OPENAI_API_KEY
MODEL_NAME
EMBEDDINGS_MODEL
```

`.env`、SQLite、PDF、向量索引与生成报告均被 Git 忽略。请使用自行申请且可轮换的凭据，禁止提交或截图泄露。CLI 只在选中需要外部依赖的命令后读取环境变量。

## 命令

```powershell
python app.py provider-health
python app.py seed --config config/seed_queries.yml
python app.py sync
python app.py stats
python app.py ask "哪些传感器用于叶绿素 a 预测？"
python app.py rebuild-index
python app.py evaluate --dataset data/evaluation/questions.jsonl
python app.py corpus-curate --limit 600
python app.py profile-abstracts --limit 500
python app.py collect-papers --provider openalex --max-results 10 --include-pdf
python app.py schedule
streamlit run web/app.py
```

`provider-health` 对 OpenAlex、CORE、Unpaywall、Crossref 做有界检查；配置 S2 key 时额外检查 Semantic Scholar。输出仅包含 provider、标准化状态、HTTP 状态、延迟与安全数值配额，不显示 URL、查询、邮箱、密钥、响应体或异常文本。

## 种子知识库：选择公式与恢复行为

首轮 seed 的目标是约 480 条元数据（允许实际结果落在 450–500），补充检索可扩展到约 1200 条；全文抓取按合法开放获取地址分批重试。近期文献配额为 100 条，代表性高被引文献配额为 60 条；两类配额会经过去重和总量上限裁剪。`stats` 和工作台概览分别报告元数据、PDF、解析、画像、索引、仅摘要与失败数量，不把元数据记录承诺为可下载或成功解析的 PDF。

候选文献在本地按确定性公式排序：

```text
score = 0.50 × relevance + 0.20 × recency
      + 0.20 × log-normalized citations + 0.10 × usable OA
```

`config/seed_queries.yml` 同时保留近期文献和高被引代表作配额。候选先整体持久化，再逐篇执行合法 OA 摄入；重复 DOI（或规范化标题+年份）不会重复建库。再次运行同一 seed 命令会跳过指纹未变化且已完成的条目，只重试失败/可重试项，`stats` 提供计数和稳定失败原因。

## 引用与证据保证

- 检索以 SQLite FTS 与向量搜索做加权 reciprocal-rank fusion；检索结果保留 `paper_id`、`chunk_id`、页码和标题。
- 模型提交的引用不是信任源。系统回到本地 chunk store 校验 chunk ID，并验证引用片段确实存在于证据正文。
- 有效证据不足时返回明确 fallback，不生成“看似完整”的答案；摘要建库的页码为 `0`，界面标记为“摘要证据”，不伪装成 PDF 页码。
- 对比与趋势结论区分直接证据、跨文献综合和研究建议，避免把推断当原文结论。

## 科研对话 Agent 与研究路线

Streamlit 的“科研 Agent”页面支持连续追问。路由节点结合最近 8 条消息、最多 1500 字符的摘要，以及主题、预测目标、传感器、研究区、年份和方法约束，将本轮问题改写为可独立检索的查询；用户消息会先显示，再展示处理状态。

Agent 单轮只允许一次路由和一次受控调用。领域问题开放三个受控 Skill：

- `evidence_qa`：调用引用安全问答工作流；
- `research_plan`：基于本地论文证据生成研究现状和待验证的起步步骤。

普通问候和与水色遥感无关的问题走 `general_chat`，不调用本地 RAG，也不生成论文引用；一旦问题涉及本项目的论文事实、方法或研究路线，路由会切回证据模式。

当领域问题缺少本地直接证据时，Agent 不再只返回空拒答：它会明确标注“本地知识库未找到足够的直接证据”，再给出不带论文引用的通用解释或通用研究建议，避免把通用知识冒充成本地文献结论。

研究路线中的文献事实必须携带能够在 SQLite chunk store 中复核的 `chunk_id` 和原文片段；无合法引用的事实会被删除。建议步骤统一标记为“待验证步骤”，不会被包装成论文结论。Agent 不拥有下载、索引或引用校验的绕过权限。

## 20 问评估与六项指标

`data/evaluation/questions.jsonl` 是 20 问诚实注释模板，覆盖 prediction target、sensor、dataset、model、metric、limitation 与 temporal trend。每行结构为：

```json
{"question_id":"q01","question":"...","relevant_paper_ids":["..."],"relevant_chunk_ids":["..."]}
```

当前模板使用显式 `__ANNOTATE_FROM_LOCAL_SEED__`，正常 evaluator 会以 `evaluation_dataset_unannotated` 拒绝它，防止 placeholder 被误报成测量结果。完成真实 seed 后，从本地数据库逐问核对证据，替换为至少一个真实 paper ID/chunk ID；不要猜造 ID。

当前已新增一份独立的本地标注集 `data/evaluation/questions-annotated.jsonl`，包含 48 个经过 paper/chunk 回查的问题，全部来自核心 RAG 池。每个问题使用带 rationale 的 evidence groups：同组 chunk 可相互替代，不同组代表回答必须覆盖的独立事实。它不覆盖 20 问模板，用于验证真实检索链路：

```powershell
python app.py evaluate --dataset data/evaluation/questions-annotated.jsonl --retrieval-only
```

标注集按 `exact_term`、`natural_language`、`cross_paper` 三类记录问题分布，每类 16 问；评估报告会输出数据集规模、类别计数和逐类 Evidence-group Recall@5、MRR 与 Paper Recall@5。48 问仍是当前本地语料版本的离线评估，不代表通用领域准确率；新增或替换论文后必须重新人工回查真实 `paper_id/chunk_id`。

这 48 问的标注是当前语料版本的离线基线，不代表领域完整 gold standard；新增或替换论文后应重新人工复核并更新数据集指纹。

评估对完全相同的问题集分别运行 keyword、vector、hybrid 和 two-stage。two-stage 只用于离线消融；默认问答继续使用 hybrid，除非它通过文档化的推广门槛。

1. Evidence-group Recall@5：前 5 个结果覆盖的独立证据组比例；同组替代 chunk 不重复计分；
2. MRR：首个相关 chunk 排名的倒数；
3. Paper Recall@5：辅助的论文层召回；
4. Citation precision：被本地可信 chunk store 与相关性注释共同确认的引用/全部引用；
5. Evidence coverage：至少存在一个有效引用的“问题×模式”比例；
6. Unsupported-claim rate：无有效支持引用的 evidence-sufficient 答案/全部 evidence-sufficient 答案。证据不足 fallback 不算自信的无支持声明。

运行 `python app.py evaluate --dataset <annotated-dataset>` 后，会在 `data/reports/evaluation/` 原子生成 UTC 命名的 JSON 与 Markdown。报告含四模式指标、数据集规模和类别统计、逐类 Evidence-group Recall@5/MRR/Paper Recall@5、每题检索 trace、QA retrieval/model 耗时和 fallback 状态、全部 question ID、检索配置/权重、数据集指纹、时间戳和验收结果，不保存全文、prompt、模型原始输出、URL、绝对路径或凭据。Hybrid Evidence-group Recall@5 若严格同时低于 keyword 和 vector，验收码为 `hybrid_recall_regression`；two-stage 必须同时满足 cross-paper 提升且 natural_language 不回退才会通过推广门槛。只能修改文档化的权重并重新测量，不能在代码中悄悄调参。

### 最近一次本地运行快照（2026-08-27）

以下数字来自开发机上的 `knowledge-audit --json` 快照。SQLite、PDF 和 FAISS
均被 `.gitignore` 排除，因此干净克隆默认显示 0；按“继续真实 seed 与评估”步骤运行后，
应以你自己的审计输出替换这些数字。

快照按 `knowledge-audit --json` 的分层口径记录，不把元数据数量冒充全文数量：

| 层级 | 数量 | 含义 |
|---|---:|---|
| 目录层 | 704 | SQLite 中的论文元数据记录 |
| 摘要字段 | 669 | 论文记录含摘要字段 |
| 证据层 | 672 | 至少有一个全文或摘要 chunk |
| 全文证据 | 116 | 存在 `page_number > 0` 的 PDF 文本块 |
| 摘要证据 | 559 | 仅存在 `page_number = 0` 的摘要块 |
| 结构化画像 | 11 | `paper_profiles` 中已保存的结构化画像 |
| FAISS 文档 | 17992 | 与 SQLite chunks 一一对应的向量文档 |

工作台维护页同时展示三层证据口径（metadata catalog、abstract evidence、page-addressable full-text）和画像覆盖率；画像只作为证据问答和研究路线的辅助数据，不代表全部 704 篇元数据。

2026-08-25 已用 S2 对 588 篇待检查 DOI 做串行低速检查（低于 1 req/s），得到 255 个 OA 候选，最终 11 篇通过 PDF 校验并完成按页解析。当前运行审计为 116 篇页码全文、17992 个 chunks，向量缺失和孤立均为 0。上表是 2026-08-27 本机快照；运行数据仍被 Git 忽略，干净克隆不会默认包含这些记录。

向量审计结果：`missing_chunk_ids = 0`、`orphan_vector_ids = 0`。当前项目没有声称线上 Recall、MRR 或问答准确率；48 问标注集仅用于当前本地语料版本的 retrieval-only 离线检索诊断。

真实问答质量、citation precision、evidence coverage 和 unsupported-claim rate：**尚未完成统计有效的联网生成式评测**，不会用模板或离线单测数字替代。`evaluate-answers` 的 10 条样本只验证人工整理的引用链；它不等于模型现场回答质量。当前已完成 48 问 retrieval-only 评估，结果见下方。

### 当前已标注集检索结果（2026-08-27）

使用 `questions-annotated.jsonl --retrieval-only` 得到以下 48 问离线结果；该模式不调用 LLM，因此只报告检索指标。每类 16 问，问题和相关 chunk 均已从当前本地数据库逐项回查：

| 模式 | Evidence-group Recall@5 | MRR | Paper Recall@5 |
|---|---:|---:|---:|
| keyword | 0.6563 | 0.5979 | 0.8229 |
| vector | 0.5938 | 0.4670 | 0.7500 |
| hybrid | 0.6563 | 0.6267 | 0.8229 |
| two-stage | 0.6354 | 0.5969 | 0.8542 |

这组数字只对应 2026-08-27 的数据库快照，且 scope 为 retrieval-only。向量索引使用本地 `BAAI/bge-m3`，输入格式为“标题 + 章节/页码 + 正文”，共 17992 个 1024 维归一化向量。Vector Recall@5 相比旧远程模型的 0.0972 提升到 0.5694；经权重消融后，Hybrid 使用 BM25 20.0、Dense 1.0 的保守 RRF，在 48 问集上与 Keyword 的 Evidence-group Recall@5 持平，并将 MRR 从 0.5979 提升到 0.6267。two-stage 提升 cross-paper 论文级召回，但没有提升段落级证据召回且降低 natural_language 结果，未通过推广门槛。不能宣称 Hybrid 或 two-stage 已改善线上问答；当前证据只支持“Dense 语义召回明显改善，Hybrid 保持证据召回并改善排序”。若论文重新摄入、去重或数据库被重建，旧 `paper_id/chunk_id` 可能失效，运行前必须先逐题回查标注。

### Answer-level 引用评估

`data/evaluation/answers-annotated.jsonl` 提供 10 条人工整理的答案/引用样本，运行：

```powershell
python app.py evaluate-answers --dataset data/evaluation/answers-annotated.jsonl
```

该命令不调用 LLM，只评估样本中的 citation precision、evidence coverage 和 unsupported-claim rate；它是引用链路 smoke set，不代表领域通用问答准确率。

## 离线测试证明的失败模式（不是线上实测）

- 非 PDF 内容、错误文件签名或超过大小限制会得到 `invalid_pdf`/`pdf_too_large` 等稳定错误，并清理 `.part` 文件，不进入解析和索引。
- 检索或模型证据不足会返回 evidence-insufficient fallback；它不会被评估为“自信但无引用”的主张。
- Provider 单项不可达不会阻断另外三项健康检查，输出不会回显异常中的 URL 或凭据。
- 周报推送失败时，本地原子保存的报告仍保留，返回可重试 `delivery_failed`，不泄露 webhook 异常文本。

这些是可复现的离线单元/集成测试结果，与后续真实 provider 连通性、知识库规模和评估指标严格分开。

## 为什么使用确定性工作流而不是完全自主 ReAct

对话 Agent 只负责识别科研意图、延续会话约束和选择受控工具。采集、合法 OA 判断、下载、解析、去重、证据索引、引用校验与评估仍采用显式状态机和稳定错误码。这样可以复现每个步骤、控制联网边界、避免 Agent 自行扩大数据源或绕过版权规则，也便于测试幂等性和失败恢复。项目的可信核心不依赖模型“自行决定”事实或安全边界。

## 两周 MVP 未实现

- 扫描版 PDF OCR；
- 多用户账户、权限与云端隔离；
- 自动执行研究实验或训练预测模型。

这些能力需要更长周期的数据治理、算力、安全与评测设计，不在个人两周 MVP 中伪装实现。

## 安全、隐私与 Windows 重建取舍

运行时数据全部留在本机并被 Git 忽略；日志、CLI、健康检查和评估报告只使用稳定错误码与安全字段。向量重建只允许仓库 `data/vector_store` 边界内的验证路径。Windows 在缺少安全 symlink 删除能力时，旧索引会先移动到边界内 quarantine 而不递归删除：代价是可能遗留少量待人工检查的目录，换取不跟随 junction/reparse point 误删仓库外数据。

## 继续真实 seed 与评估

1. 用户在本机创建 `.env`，使用新申请/已轮换的 provider 和模型配置；不要把文件内容发到终端或聊天。
2. 运行离线测试：`.\.venv\Scripts\python.exe -m pytest -v`。
3. 运行 `python app.py provider-health`，仅确认标准化状态。
4. 连续运行两次 `python app.py seed --config config/seed_queries.yml`，再运行 `python app.py stats`，验证恢复与去重。
5. 查询本地 SQLite/chunk store，为 20 问写入真实 relevant paper/chunk ID。
6. 运行 evaluation，检查 JSON/Markdown 与 hybrid acceptance，再将真实数字和日期写入“实测结果”。

## 截图

仓库不提供伪造截图。待本地凭据配置后生成实际界面，并在本地运行 `streamlit run web/app.py`；截图只应展示不含密钥、绝对路径和版权全文的状态、问答引用和研究路线页面。
