# Research Lens 交接文档

更新时间：2026-08-27
项目：水色遥感与水质参数预测科研助手
工作目录：`C:\Users\Administrator\Desktop\实习项目\document\intel_agent`

## 1. 项目定位与边界

这是用于秋招展示的硕士个人 AI 项目，目标岗位是 AI 应用/大模型工程，项目主线固定为：

> 受控科研 Agent + 证据型 RAG + 可审计后端摄入流程。

核心能力：

- 从 OpenAlex 等公开来源发现水色遥感文献，并通过合法 OA 地址补充全文；
- 校验 PDF、按页解析、证据切块，并写入 SQLite/FTS5 与 FAISS；
- 通过意图识别和查询扩展执行关键词、向量和混合检索；
- 科研问答必须绑定论文、年份、页码/章节、Chunk 和原文引用；
- 普通聊天可使用 GPT，但涉及知识库事实时必须进入证据检索流程；
- 保存检索 trace、证据等级和稳定失败码，便于判断失败发生在摄入、切块、检索还是回答阶段。

明确不做：

- 不绕过付费墙、登录或版权限制；
- 不把摘要、HTML 或错误响应伪装成全文；
- 不扩张成多用户平台，不引入 Redis、云部署、OCR、复杂多 Agent 编排；
- 不继续为了指标盲目更换 Embedding 模型；
- 不把元数据数量、单元测试或 retrieval-only 指标包装成线上问答准确率；
- `.env`、API key、邮箱、SQLite、PDF、FAISS 和运行报告不提交 GitHub。

## 2. 当前运行配置

对话模型和 Embedding 已分离：

| 用途 | 当前配置 | 边界 |
|---|---|---|
| 普通对话与答案生成 | OpenAI 兼容中转，`gpt-5.5` | 已通过最小对话连通性验证；没有完成联网答案质量评测 |
| Dense Embedding | 本地 `BAAI/bge-m3` | RTX 4090 CUDA，1024 维，L2 归一化 |
| 关键词检索 | SQLite FTS5/BM25 | 精确术语、论文名、传感器名和方法名的主召回通道 |
| 混合检索 | Weighted RRF | BM25 `20.0`、Dense `1.0`、`rrf_k=60` |

GPT 中转不提供 `text-embedding-3-small` 或 `text-embedding-3-large`，调用 Embedding 端点会返回不支持。因此 GPT 中转只负责对话，不能用于向量库重建。

本地 `.env` 当前配置了对话中转和 BGE-M3，但交接、日志和 Git 文件中不得记录密钥。

本地环境当前为 PyTorch `2.7.1+cu128`，CUDA 已识别 RTX 4090。BGE-M3 模型缓存和 CUDA 环境占用较大，系统盘剩余空间约 7 GB；下一步不要再下载其他大型 Embedding 模型。

## 3. 最新本地数据快照

以下数据来自 2026-08-27 的 `knowledge-audit --json`。运行数据被 Git 忽略，干净克隆不会包含这些记录。

| 指标 | 数量 | 口径 |
|---|---:|---|
| 论文元数据 | 704 | SQLite `papers` 表 |
| 含摘要论文 | 669 | `abstract` 字段非空 |
| 可定位页码的真实全文 | 116 | 存在 `page_number > 0` 的文本块 |
| 摘要文本证据 | 559 | `page_number = 0` 的摘要证据层 |
| 含任意本地文本论文 | 675 | 全文证据或摘要证据 |
| SQLite 文本块 | 17992 | 全文块与摘要块合计 |
| FAISS 向量 | 17992 | 与 SQLite Chunk 一一对应 |
| 已保存结构化画像 | 11 | `paper_profiles` 表 |
| 向量缺失/孤立 | 0 / 0 | `knowledge-audit` 结果 |

当前论文状态：

```text
indexed: 102
parsed: 14
abstract_only: 577
discovered: 1
metadata_ready: 6
failed: 4
```

全文数量只能按 `page_number > 0` 统计，不能用 `papers_with_chunks`、`vector_indexed_papers` 或 `indexed` 代替。

## 4. 为什么会排查 Embedding

### 4.1 最初异常指标

36 问 retrieval-only 评测集按 `exact_term`、`natural_language`、`cross_paper` 各 12 问配平。旧索引结果为：

| 模式 | Recall@5 | MRR | Paper Recall@5 |
|---|---:|---:|---:|
| keyword | 0.6806 | 0.6389 | 0.8750 |
| vector | 0.0972 | 0.1111 | 0.2639 |
| hybrid | 0.6806 | 0.5833 | 0.8750 |

真正触发排查的是 **Vector Recall@5 只有 0.0972**。同时 Hybrid MRR 低于 Keyword，说明旧 Dense 分支几乎没有提供有效语义召回，融合还可能拖累排序。

### 4.2 排查排除了什么

已确认旧索引不存在以下基础配置错误：

- 索引和查询使用同一个 Embedding 模型；
- FAISS 维度与模型输出一致；
- 使用 L2 距离，距离排序方向正确；
- 存储向量范数约为 1.0，没有明显归一化错配；
- SQLite Chunk 数与 FAISS 向量数一致；
- 36 个问题均为英文，不是简单的中英文不匹配；
- Chunk 并非普遍过短。

发现的主要表示问题：

- 旧向量只编码 `chunk.text`，标题未参与 Embedding；
- `17433 / 17992` 个 Chunk 的 section 为空，结构信息弱；
- 评测只标注一个正确 Chunk 时，可能把同论文的另一处有效证据误判为失败；
- 旧远程 Embedding 在当前水色遥感语料上的实测召回确实很低。

因此问题不是单一“代码写错”，而是 **向量输入表示不足 + 旧模型在当前语料上效果差 + 评测标注偏严格**。

### 4.3 为什么最后选择 BGE-M3

处理过程曾考虑 OpenAI Embedding，但用户提供的 GPT 中转只开放对话模型，不开放 Embedding 模型。继续依赖旧远程服务还存在全量重建吞吐很慢的问题。

最终选择本地 `BAAI/bge-m3`，原因是：

- 免费，不产生持续 API 成本；
- 支持中英文和跨语言语义检索；
- 本机 RTX 4090 24GB 可以稳定运行；
- 面试演示和项目复现不依赖第三方 Embedding 服务可用性；
- 适合个人项目，不需要额外部署独立向量服务。

这不是为了“追新模型”，而是由旧 Vector Recall@5 过低、GPT 中转不支持 Embedding、远程重建过慢共同推动的一次有评测依据的替换。

## 5. BGE-M3 改造与最终结果

### 5.1 已完成改造

- 向量文本 schema 改为 `title-location-text-v1`：

```text
标题：论文标题
位置：章节或页码
正文：Chunk 原文
```

- 引用和检索结果仍返回 `canonical_text`，不会把拼接后的向量文本当作原文；
- 本地 BGE-M3 输出 1024 维归一化向量；
- 已原子重建 17992 个向量，SQLite 缺失/FAISS 孤立均为 0；
- manifest 固化 Embedding provider、模型、文本 schema、切块指纹和文档数；
- 普通加载遇到模型/切块漂移仍拒绝旧索引；
- 显式 `rebuild-index` 可以跳过不兼容旧索引，在 staging 中完成后原子发布；
- 多查询向量融合最终严格截断为 `k`，避免 `k=5` 返回十几条结果；
- 评测 CLI 与线上 RAG 读取同一份 `rag.yml` 权重，避免报告参数与实际运行参数漂移。

### 5.2 最终正式评测

正式报告：`evaluation-20260827T042419Z-0e37ac53.json`（本地报告目录，Git 忽略）。

| 模式 | Recall@5 | MRR | Paper Recall@5 |
|---|---:|---:|---:|
| keyword | 0.6806 | 0.6389 | 0.8750 |
| vector | 0.5694 | 0.4514 | 0.7500 |
| hybrid | 0.6806 | 0.6690 | 0.8750 |

关键结论：

- Vector Recall@5 从 `0.0972` 提升到 `0.5694`，Dense 分支已从基本失效变成有效补充；
- Hybrid Recall@5 与 Keyword 持平，不能写“Hybrid 提升召回率”；
- Hybrid MRR 从 Keyword 的 `0.6389` 提升到 `0.6690`，可以写“保持召回的同时改善排序”；
- 权重消融覆盖 `1:1、7:3、2:1、3:1、5:1、8:1、10:1、20:1`，最终选择保守的 BM25 `20.0`、Dense `1.0`；
- 真实叶绿素中文问题已能召回叶绿素浓度估算、Case-II 水体三波段模型、Sentinel-2/Landsat-8 反演等直接相关论文。

评测报告中的 `accepted` 只是程序的粗粒度回归状态，不能代替人工验收。当前项目的实际门槛是：Hybrid Recall@5 不得低于 Keyword，并同时检查 MRR 和 Paper Recall@5。

### 5.3 证据组标注与两阶段检索消融（2026-08-27）

- 评测集已扩展为 48 问，`exact_term`、`natural_language`、`cross_paper` 各 16 问；相关 Chunk 改为带 rationale 的 evidence groups，同组 Chunk 表示可替代证据，不同组表示必须分别覆盖的事实。
- 新增 `evidence_group_recall_at_5`，避免同一问题标注多个有效 Chunk 时把替代证据误判为失败；所有 48 条论文和 Chunk ID 均通过本地 SQLite 校验。
- Two-stage 先聚合 BM25/BGE-M3 的论文级候选，再在候选论文内运行现有 Hybrid 定位证据，不创建新索引。

| 模式 | Evidence-group Recall@5 | MRR | Paper Recall@5 |
|---|---:|---:|---:|
| keyword | 0.6563 | 0.5979 | 0.8229 |
| vector | 0.5938 | 0.4670 | 0.7500 |
| hybrid | 0.6563 | 0.6267 | 0.8229 |
| two_stage | 0.6354 | 0.5969 | 0.8542 |

分类 Evidence-group Recall@5 / Paper Recall@5：`exact_term` 为 Keyword `0.6563 / 0.9375`、Vector `0.6875 / 0.8750`、Hybrid `0.6563 / 0.9375`、Two-stage `0.6563 / 0.9375`；`natural_language` 为 `0.8750 / 0.9375`、`0.8125 / 0.8750`、`0.8750 / 0.9375`、`0.8125 / 0.9375`；`cross_paper` 为 `0.4375 / 0.5938`、`0.2813 / 0.5000`、`0.4375 / 0.5938`、`0.4375 / 0.6875`（顺序均为 Keyword、Vector、Hybrid、Two-stage）。

Two-stage 的 Paper Recall@5 有提升，但 Evidence-group Recall@5 和 `natural_language` 分类均下降，未通过推广门槛。因此默认 QA 继续使用 Hybrid；Two-stage 仅保留为可复现实验路径，不宣称线上收益。

### 5.4 分类指标

| 类别 | Keyword Recall@5 | Vector Recall@5 | Hybrid Recall@5 |
|---|---:|---:|---:|
| exact_term | 0.6250 | 0.6667 | 0.6250 |
| natural_language | 0.9167 | 0.7500 | 0.9167 |
| cross_paper | 0.5000 | 0.2917 | 0.5000 |

当前最明显的检索短板已经不再是“Embedding 完全不可用”，而是 **跨论文综合问题的论文级召回和段落定位不足**。

## 6. 当前完成边界

已经完成并有证据支持：

- 36 问 retrieval-only 离线评测；
- 48 问 evidence-group retrieval-only 离线评测与论文/Chunk ID 审计；
- Keyword、Vector、Hybrid 三路指标和分类指标；
- Two-stage 论文级召回与段落定位消融，明确未通过推广门槛；
- BGE-M3 全量重建与索引一致性审计；
- RRF 权重消融；
- 查询扩展、同义词/中英文术语转换和多查询 Dense 召回；
- 证据分级、邻接 Chunk 上下文、引用安全校验和检索 trace；
- GPT 中转最小对话连通性验证；
- 完整自动化测试：`336 passed, 1 skipped`。

尚未完成，不能写成项目成果：

- 没有完成联网生成式问答的 citation precision、evidence coverage、unsupported-claim rate 和答案可验证率评测；
- 现有 10 条 answer/citation 数据只是人工 smoke set，不代表真实问答质量；
- 没有证明 Reranker 能提升当前数据集；
- 两阶段检索实验已完成，但未通过 natural_language 不下降与证据组 Recall 门槛，未接入默认 QA；
- 评测集只有 36 问，且有效 Chunk 标注可能不完整；
- 趋势和论文对比只基于 11 条结构化画像，不能代表全部 704 篇论文。

## 7. 已摒弃或暂停的方向

以下内容不再作为当前下一步：

- 继续更换 Embedding 模型：BGE-M3 已显著改善 Vector 指标，先解决跨论文检索和评测标注；
- 继续接 OpenAI Embedding：现有 GPT 中转不支持，个人项目也没有必要增加持续成本；
- 继续堆全文 provider：S2、OpenAlex、Unpaywall、CORE 已覆盖当前 MVP，新增 provider 的复杂度高于面试收益；
- 直接增加复杂 Reranker：必须先有基于当前 36 问或扩展集的消融证据；
- OCR、表格/公式解析、多用户、Redis、云部署和复杂 Agent 编排：不属于当前个人项目边界；
- 用“展示思考过程”暴露模型隐式推理：页面只展示意图、查询改写、工具、候选数、引用数、fallback 和耗时等可审计 trace。

## 8. 下一步计划

按优先级执行，不再从“换模型”开始：

### P0：修正和扩展检索评测标注

- 为同一问题补充多个有效 `chunk_id`，避免“找到正确论文另一处证据”被误判；
- 重点复查 `cross_paper` 12 问的相关论文和有效段落；
- 保持 `exact_term / natural_language / cross_paper` 分类统计；
- 数据规模从 36 问扩展到约 45–50 问即可，不追求大型 benchmark。

验收：评测标注可解释，每个问题至少说明相关论文和证据段落为什么有效。

### P1：两阶段检索解决 cross_paper

- 第一阶段以标题、摘要和论文级聚合分数召回相关论文；
- 第二阶段只在候选论文内部检索 Chunk；
- 命中 Chunk 后补充同章节标题、前后邻接 Chunk 和页码；
- 与当前单阶段 Keyword/Vector/Hybrid 做同集消融。

验收：优先提升 `cross_paper` Paper Recall@5 和 Recall@5，且不得降低 `natural_language` 的现有结果。

### P2：联网答案级评测

- 从已有问题中整理 10–15 个可人工核验的问题；
- 运行真实 GPT 回答，人工核对引用、原文支持和确定性措辞；
- 统计 citation precision、evidence coverage、answer verifiability 和 unsupported-claim rate；
- 记录证据不足是检索失败、段落定位失败还是回答模型失败。

验收：简历只使用真实报告数字，不使用模板、单元测试或人工 smoke set 冒充答案质量。

### P3：条件满足后再考虑轻量 Reranker

只有 P0/P1 后 Hybrid 仍存在明确排序错误，才增加 CrossEncoder 或 API Rerank，并与无 Rerank 基线比较。没有显著收益就不保留。

## 9. 全文获取现状

2026-08-25 已使用 Semantic Scholar key 对 588 篇待检查 DOI 低速串行检查，得到 255 个 OA 候选，最终 11 篇通过 PDF 校验并按页解析。当前失败主要为：

```text
pdf_download_failed: 478
invalid_pdf: 36
no_open_full_text: 63
pdf_parse_failed: 1
vector_index_failed: 3  # 历史失败记录
```

重复执行旧的全文重试命令收益很低。后续只做小批次、可恢复重试，或将人工合法获取的 PDF 按 DOI/`paper_id` 走统一摄入流水线。暂不新增全文 provider。

## 10. 关键文件

```text
app.py                              CLI、DefaultServices、运行时/评测配置对齐
web/app.py                          Streamlit 工作台
retrieval/query_expansion.py        意图识别、同义词、中英文和多查询扩展
retrieval/keyword_index.py          SQLite FTS5/BM25
retrieval/hybrid.py                 Weighted RRF、去重和检索 trace
rag/vector_store.py                 BGE-M3、FAISS、manifest、原子重建
storage/database.py                 SQLite、FTS5、chunks、profiles
workflows/qa.py                     证据分级、引用和邻接上下文
evaluation/run.py                   三路检索与分类评测
data/evaluation/questions-annotated.jsonl
tests/                              自动化测试
```

## 11. 常用命令与接手顺序

新对话先执行：

```cmd
cd /d "C:\Users\Administrator\Desktop\实习项目\document\intel_agent"
.venv\Scripts\python.exe app.py stats
.venv\Scripts\python.exe app.py knowledge-audit --json
.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-next
```

只有修改检索、标注或索引后才重新评测：

```cmd
.venv\Scripts\python.exe app.py evaluate --dataset data\evaluation\questions-annotated.jsonl --retrieval-only
```

只有 manifest 明确不兼容或数据库 Chunk 发生变化时才重建：

```cmd
.venv\Scripts\python.exe app.py rebuild-index
```

不要为了“试一下”重复重建 17992 个 BGE-M3 向量。

旧的 `.vector_store.rebuild-delete-*` 索引隔离备份已从项目目录移出并归档到项目外的 `_intel_agent_cleanup_20260828`，不参与当前加载。当前活动索引只在 `data\vector_store`。

新对话可直接发送：

> 请先读取 `docs/research-lens-handoff-2026-08-22.md`，在 `C:\Users\Administrator\Desktop\实习项目\document\intel_agent` 继续工作。当前 BGE-M3 向量索引已完成，Vector Recall@5 为 0.5694，不要继续更换 Embedding。先运行 `stats` 和 `knowledge-audit --json`，下一步从 cross_paper 标注复查和两阶段检索开始。不要输出 `.env`、API key、PDF、SQLite、FAISS 或本地运行报告。

## 12. 安全与工程约束

- 不输出或提交 API key、邮箱、`.env` 内容和凭据；
- 不绕过付费墙、登录或版权限制；
- 不将 HTML、登录页或错误响应保存为 PDF；
- 不把摘要块写成全文页码证据；
- 不手工修改论文状态或评测结果；
- 正常加载必须保留 manifest 兼容性校验；
- 修改检索行为必须先补测试，再跑同一评测集；
- 删除数据库、PDF、当前 FAISS 或评测标注前必须明确确认。

## 13. 面试与简历叙述

推荐主线：

> 面向水色遥感文献构建受控科研 Agent，通过合法 OA 摄入、页码级证据切块、BM25 + BGE-M3 + Weighted RRF 混合检索和引用校验，实现可追溯的科研问答；使用 36 问离线集按问题类型评测并通过消融定位 Dense 召回问题。

当前可写的量化点：

- 管理 704 篇元数据、116 篇页码全文和 17992 个证据 Chunk，SQLite/FAISS 缺失与孤立向量均为 0；
- 将向量输入从纯正文改为标题、位置和正文，并以 BGE-M3 原子重建索引，使 Vector Recall@5 从 0.0972 提升到 0.5694；
- 对 RRF 权重做离线消融，在保持 Hybrid Recall@5 为 0.6806 的同时，将 MRR 从 Keyword 基线的 0.6389 提升到 0.6690；
- 将意图、查询改写、检索置信度、证据等级、引用和 fallback 分开记录，避免一次检索失败被误判为知识库无答案。

必须主动说明：这组数字来自当前本地语料的 36 问 retrieval-only 评测，不代表线上通用问答准确率；答案级引用指标尚未完成。
