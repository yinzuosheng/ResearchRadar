# GitHub 发布说明

## 可提交内容

- Python 源码：`agent/`, `domain/`, `evaluation/`, `ingestion/`, `providers/`, `rag/`, `retrieval/`, `storage/`, `web/`, `workflows/`, `utils/`
- 配置模板：`config/`、`.env.example`
- Prompt：`prompts/`
- 自动化测试：`tests/`
- 文档：`README.md`、`docs/`
- 不含凭据的评测输入：`data/evaluation/*.jsonl`

## 不可提交内容

`.env`、SQLite 数据库、PDF、FAISS 索引、缓存、评测报告、虚拟环境、`.pytest-tmp*` 和 worktree 均被 `.gitignore` 排除。评测报告包含本地运行时间和数据集指纹，发布前应只提交不含本地路径、URL、prompt 和模型原始输出的示例报告（如需展示）。

## 发布前检查

```powershell
git status --short --ignored
git diff --check
python -m pytest -q
python app.py knowledge-audit --json
```

确认 `git status` 中没有 `.env`、`research.db`、`index.faiss`、PDF 或 API key。项目默认使用空状态启动；运行真实采集和问答前，在本机复制 `.env.example` 为 `.env` 并填写凭据。

## 推荐仓库结构

```text
intel_agent/
├── agent/ domain/ evaluation/ ingestion/
├── providers/ rag/ retrieval/ storage/ workflows/
├── web/ utils/ model/ prompts/ config/
├── tests/
├── data/evaluation/
├── docs/
├── app.py
├── README.md
├── requirements.txt
└── .env.example
```

本项目是本地可复现的硕士个人 AI 应用，不把本地知识库快照或第三方受版权保护全文上传到 GitHub。面试演示使用公开元数据、用户自行配置的开放获取来源和本地生成的审计结果。
