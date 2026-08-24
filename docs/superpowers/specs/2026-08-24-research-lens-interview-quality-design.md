# Research Lens 秋招面试质量增强设计

## 背景

Research Lens 是面向 AI 应用/大模型工程岗位的硕士个人项目。当前系统已经具备合法 OA 文献摄入、证据型 RAG、受控 Agent、Streamlit 工作台和离线评估能力，但功能面较宽，评估集较小，Agent 的内部运行信息没有在演示界面中清晰呈现。本次工作聚焦于让现有能力更容易被面试官验证，而不是继续增加平台功能。

## 目标

1. 将项目主线固定为“受控科研 Agent + 证据型 RAG + 可审计后端工作流”，并在 README/面试文档中区分核心能力与辅助能力。
2. 让评估报告支持问题类别、数据集规模和逐类指标，明确标注 retrieval-only、离线基线和真实问答评估边界。
3. 在科研 Agent 页面展示本轮 skill、路由方式、检索候选数、最终证据数、引用数、fallback 状态和耗时，形成可解释的演示闭环。
4. 保持个人 MVP 范围：不新增全文 provider、OCR、FastAPI、多用户、云部署或自动实验执行。

## 非目标

- 不伪造或自动生成 gold 标注，不把现有 12 问结果升级为通用准确率。
- 不改变 OA 版权边界、数据库状态机或向量索引数据口径。
- 不把 Agent 改造成多轮自主 ReAct；仍保持一次路由和一次受控 skill 调用。

## 设计

### 1. 项目主线和叙述

README 与 `docs/interview/ai-engineer-project-positioning.md` 增加“面试主线”小节：

- 核心路径：候选发现/合法摄入 -> 证据分层 -> FTS5 + FAISS 检索 -> 引用校验 -> 受控 Agent 输出。
- 辅助能力：论文画像、论文对比、趋势报告和 Streamlit 导航，用于展示复用能力，不作为独立平台模块宣传。
- 明确数据口径：704 是元数据目录，96 是可定位全文，摘要证据单独计数；未完成人工标注前不写 Recall、MRR 或问答质量百分比。

同一文档补充三条最终简历 bullet、30 秒项目介绍和常见追问回答，所有量化数字注明来源和日期。

### 2. 评测诊断增强

扩展 `evaluation.dataset.EvaluationQuestion`，支持可选 `category` 字段。缺省类别为 `uncategorized`，保留现有数据集兼容性。类别只接受短 ASCII 标识（例如 `exact_term`、`natural_language`、`cross_paper`），非法值返回稳定的 `evaluation_dataset_invalid_category`。

扩展 `evaluation.run.run_evaluation`：

- 保持现有 keyword/vector/hybrid 三模式和 Recall@5、MRR、Paper Recall@5、citation precision、evidence coverage、unsupported-claim rate；
- 在报告中增加 `dataset_summary`（问题总数、类别计数、evaluation scope）；
- 增加 `metrics_by_category`，每类至少输出问题数、Recall@5、MRR；问答指标沿用现有 overall 口径；
- 报告 markdown 增加“不可宣传项”提示：当数据集小于 20 问或为 retrieval-only 时，明确说明结果只代表当前本地语料版本；
- 现有 acceptance 规则不变：hybrid Recall@5 严格低于 keyword 和 vector 时仍返回 `hybrid_recall_regression`。

不在本次代码中自动调权或引入 reranker；先把问题分层和误差定位能力做实，避免为追求数字增加不可解释依赖。

### 3. Agent 运行诊断

在 `domain.models` 增加 `AgentDiagnostics` 数据契约，并作为 `ResearchAgentReply.diagnostics` 可选字段。字段只包含安全、有限的数值/枚举：

- `route_mode`: `model` 或 `fallback`；
- `skill_id`；
- `retrieval_candidates`、`evidence_chunks`、`citation_count`；
- `fallback`: 布尔值；
- `retrieval_ms`、`model_ms`、`total_ms`。

`ResearchAgentService` 从受控 workflow 的 `last_trace` 读取诊断，读取不到时使用 0，不把 query、prompt、原文、URL、绝对路径或异常详情写入诊断。路由节点记录模型结构化输出失败时的 fallback，工具节点记录总耗时。诊断只进入当前对话状态和页面，不写入运行报告或数据库，避免扩大敏感数据留存面。

Streamlit 科研 Agent 页面在每个 assistant 消息下增加一个折叠的“运行诊断”区域，展示 skill、route、证据数、引用数、fallback 和耗时；缺少诊断时不显示空面板。现有引用渲染、证据不足提示和补充检索行为保持不变。

## 错误处理

- 评测数据集类别错误：在数据集加载阶段抛出 `DatasetError("evaluation_dataset_invalid_category")`，CLI 返回稳定 JSON 错误。
- workflow 未提供 trace：诊断字段使用零值，不影响问答结果。
- trace 值不是有限数字或超出边界：归一化为 0，页面不显示异常文本。
- 页面诊断渲染异常：只显示 `ui_agent_diagnostics_failed`，不得影响回答和引用展示。

## 测试策略

- 先为类别解析、缺省兼容、按类别指标和报告提示编写失败测试，再实现最小逻辑。
- 为 Agent diagnostics 编写模型路由成功、路由 fallback、workflow 无 trace 三类测试，断言不包含 prompt/原文/URL/凭据。
- 为 Streamlit presenter 增加诊断字段的安全渲染测试，保持已有导航和引用测试通过。
- 完成后运行针对性测试、完整 pytest 和 `knowledge-audit --json`；不以测试数量替代真实评测指标。

## 验收标准

1. README 和面试定位文档只保留一条核心技术主线，明确辅助能力和数据口径。
2. 旧的 `questions-annotated.jsonl` 无需修改即可通过评估；新增带类别数据集能生成 `dataset_summary` 和 `metrics_by_category`。
3. Agent 页面能在成功和 evidence-insufficient 两种路径显示安全诊断，回答内容和引用行为不回归。
4. 全部自动化测试通过，知识库审计仍为 `missing_chunk_ids = 0`、`orphan_vector_ids = 0`。
