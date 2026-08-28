# AI Engineer 面试定位

## 一句话定位

面向水色遥感与水质参数预测的受控科研 Agent：将合法开放文献采集、PDF 解析、SQLite/FTS5、FAISS 混合检索、结构化论文画像和引用校验串成一个可恢复、可审计的 RAG 工作流。

## 核心技术主线

面试时只围绕一条主线展开：**合法 OA 摄入 → 证据分层 → FTS5/FAISS 混合检索 → 引用校验 → 受控 Agent 输出**。论文画像和 Streamlit 工作台是复用这条主线的辅助能力。

30 秒介绍：

> 我做了一个面向水色遥感文献的受控科研 Agent。系统从合法开放来源构建本地证据库，用 SQLite FTS5 和 FAISS 做混合检索，再由服务端回查 chunk、页码和 quote 校验引用。Agent 只负责识别意图并调用受控 skill，下载、索引和引用安全由确定性后端工作流负责。这个项目重点解决的是 RAG 应用里的证据可信、失败可恢复和运行可审计问题。

## 可以写进简历的三条成果

1. 设计 OpenAlex、Crossref、Unpaywall、CORE 与 Semantic Scholar fallback 的开放文献摄入流水线，完成 DOI 去重、OA 地址解析、PDF 文件头/大小校验、按页解析和断点恢复；当前运行目录包含 704 篇元数据，其中 116 篇形成可定位页码的开放全文证据。
2. 构建 SQLite FTS5 + FAISS + RRF 混合检索，使用稳定 `chunk_id`、页码和本地 chunk 回查校验引用；当前 17992 个文本块与向量文档通过一致性审计，缺失向量和孤立向量均为 0。
3. 实现有界 Agent 与结构化论文画像：Agent 单轮只执行一次路由和一次受控工具调用，问答/研究路线输出可回查引用；摄入、检索、引用安全和失败恢复由自动化测试覆盖。

4. 增强 RAG 可观测性与可复现性：混合检索记录候选规模、融合结果、论文多样性和延迟；项目记忆保存用户确认来源；FAISS 写入 `vector_store_manifest.json` 固化 embedding、切块指纹和文档数，避免索引配置漂移。

5. 增加索引兼容性保护和评测诊断：FAISS 加载前校验 embedding/chunking manifest，评测报告保留每题 retrieval trace，支持定位 hybrid 与 keyword 的差异。

6. 增加 QA 运行时诊断：记录检索耗时、模型耗时、canonical chunk 数、引用数和 fallback 状态，并写入安全评测报告，不记录 prompt、原文或凭据。

建议最终只选择下面三条中的 2–3 条，避免简历变成功能清单：

1. 设计并实现面向水色遥感文献的受控科研 Agent，基于 LangGraph 完成意图路由、结构化输出和单轮调用预算控制，仅开放证据问答和研究路线等受控技能，并通过确定性 fallback 降低模型失控风险。
2. 构建合法开放文献的 RAG 摄入与检索链路，串联 OpenAlex、Crossref、Unpaywall/CORE/Semantic Scholar、PDF 校验、按页解析、SQLite/FTS5、FAISS 和 RRF 混合检索；本地运行快照包含 704 篇元数据、116 篇可定位全文、17992 个证据向量，向量一致性审计缺失/孤立均为 0。
3. 实现 citation-safe QA 和可审计评估：服务端回查 `paper_id/chunk_id/page_number/quote` 验证引用，证据不足时自动降级；评估报告记录 Recall@5、MRR、引用精度、证据覆盖率、unsupported-claim rate、检索 trace 和延迟，并由 311 个通过、1 个跳过的自动化测试覆盖核心摄入、检索、引用和失败恢复路径。

这些数字分别来自本机 `stats`、`knowledge-audit --json` 和最近一次测试快照；重新运行后应以最新输出替换。36 问 Recall/MRR 仅属于当前本地语料版本的 retrieval-only 离线结果，不把它写成线上准确率；问答质量数字仍待单独评估。

## 面试时的范围取舍

- 核心：受控 Agent、证据型 RAG、引用校验、可恢复摄入和一致性审计。
- 辅助：论文画像和 Streamlit 页面，用于演示复用能力。
- 明确不做：继续扩展全文 provider（Semantic Scholar 已作为唯一新增 DOI fallback）、OCR、FastAPI、多用户、云部署和自动科研实验执行。

## 不能写的说法

- 不要把元数据目录数量写成全文论文数量；当前 704 是元数据目录，116 篇是可定位页码的全文证据。
- 不要写“RAG 准确率 XX%”；当前真实评测集尚未完成标注。
- 不要写“自动获取所有论文全文”；合法 OA 地址和仓储稳定性决定全文命中率。
- 不要写“Agent 自主完成科研”；Agent 只负责受控路由，事实、索引和引用由程序控制。

## 面试时主动说明的限制

- 全文开放获取命中率低于元数据发现量，当前全文证据为 116 篇；系统保留摘要证据并明确标记，不把摘要冒充全文。
- pypdf 对扫描版 PDF、复杂表格和公式支持有限，OCR 和版面级解析尚未纳入 MVP。
- 已完成 36 问 retrieval-only 离线评估，按 `exact_term`、`natural_language`、`cross_paper` 各 12 问配平；2026-08-27 当前数据库快照的 Recall@5 为 keyword 0.6806、vector 0.5694、hybrid 0.6806，Paper Recall@5 分别为 0.8750、0.7500、0.8750。将旧远程 Embedding 替换为本地 BGE-M3，并以“标题 + 章节/页码 + 正文”重建 17992 个向量后，Vector Recall@5 从 0.0972 提升到 0.5694；经消融选择 BM25 20.0、Dense 1.0 的保守 RRF，Hybrid 召回与 Keyword 持平，MRR 从 0.6389 提升到 0.6690。不能宣称 Hybrid 提升了召回率。
- citation precision、evidence coverage、unsupported-claim rate 和问答质量尚未运行联网生成式评测；论文重新摄入或去重后，旧 `paper_id/chunk_id` 可能失效，必须重新人工回查标注。

## 后续唯一关键实验

在保持 36 问标注集的前提下，优先扩充同一结论的多个有效 Chunk 标注，并针对 `cross_paper` 增加论文级召回后再做段落定位；之后再评估轻量 reranker 和问答质量、citation precision、evidence coverage、unsupported-claim rate。简历只写可复现报告中的真实数字，不写模板或离线单测数字。

## 面试追问的回答主线

**为什么不用纯向量检索？** 论文问题包含大量精确术语、传感器名、模型名和指标名；FTS5 保证词项命中，FAISS 补充语义相似度，RRF 负责融合，最后按论文和页码去重。

**如何避免模型编造引用？** 模型只能提交候选 `chunk_id`；服务端回查 SQLite，验证论文 ID、页码和 quote 子串，非法引用被丢弃，证据不足则降级为 fallback。

**为什么不把所有元数据放进 RAG？** 目录层服务于发现和追踪，证据层服务于问答；默认检索只使用经过相关性、证据类型和被引排序的核心池，降低偏题文献污染。
