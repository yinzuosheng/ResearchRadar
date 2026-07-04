# ResearchRadar — 垂直领域情报收集与日报系统

基于 LangChain ReAct Agent + FAISS 向量检索构建的自动化情报 Agent，覆盖多源采集 → 语义摄入 → RAG 简报生成 → 多通道推送全链路，支持学术论文与网页信息自动追踪与定时自动化运行。

## 功能

- **多源论文采集** — Semantic Scholar、OpenAlex、arXiv 三大学术源，支持 OpenAlex 倒排索引摘要重建
- **网页搜索** — Tavily / Bing 双搜索引擎，trafilatura 正文提取 + BeautifulSoup fallback
- **RAG 双链检索** — summary 链（知识问答）+ report 链（结构化日报），PromptTemplate 解耦
- **FAISS 向量库** — 持久化存储，支持 OpenAI / HuggingFace 双 Embedding 后端切换
- **定时自动化** — APScheduler 每日定时采集 → 摄入 → 生成全自动
- **多通道推送** — 飞书 Webhook、钉钉机器人、SMTP 邮件，YAML 配置驱动
- **CLI 五命令** — `collect` / `collect-papers` / `brief` / `ask` / `schedule`

## 快速开始

### 1. 安装依赖

```bash
cd intel_agent
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入 Key：

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
# 以下可选
SEMANTIC_SCHOLAR_API_KEY=
BING_API_KEY=
FEISHU_WEBHOOK=
```

### 3. 采集论文

```bash
python app.py collect-papers
```

### 4. 生成日报

```bash
python app.py brief
```

### 5. 启动定时任务

```bash
python app.py schedule
```

### 6. 交互问答

```bash
python app.py ask "agentic workflow 最新进展"
```

## 技术栈

Python · LangChain · FAISS · APScheduler · Semantic Scholar API · OpenAlex API · arXiv API · Tavily API · pypdf · trafilatura

## 配置说明

编辑 `config/agent.yml` 可自定义 topics、paper_queries、定时时间、推送通道等。

## 项目结构

```
intel_agent/
├── agent/            # ReAct Agent + 8工具
├── rag/              # FAISS 向量库 + RAG 双链
├── model/            # LLM 工厂（OpenAI）
├── utils/            # 采集 Pipeline / 搜索 / 推送 / 调度器
├── config/           # YAML 配置（agent / tools / rag）
├── prompts/          # System / RAG / Report 模板
├── data/             # 缓存 / 向量库 / 日报输出
└── app.py            # CLI 入口（argparse 五命令）
```
