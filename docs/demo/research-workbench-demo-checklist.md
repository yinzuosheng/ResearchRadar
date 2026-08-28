# 科研工作台演示清单

这份清单用于准备秋招项目演示，不把未实际运行的联网结果写成项目指标。

## 运行前

- [ ] 在本机 `.env` 配置已轮换的 provider 和模型凭据，不在截图或终端输出中展示文件内容。
- [ ] 运行离线测试：`..\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp-demo`。
- [ ] 运行 `..\.venv\Scripts\python.exe app.py provider-health`，只记录标准化状态。
- [ ] 确认 `data/research.db`、`data/papers/`、`data/vector_store/` 和 `data/reports/` 不进入 Git。

## 知识库准备

- [ ] 运行 `..\.venv\Scripts\python.exe app.py seed --config config/seed_queries.yml`。
- [ ] 再次运行相同 seed，确认不会重复新增 DOI、规范化标题或向量。
- [ ] 运行 `..\.venv\Scripts\python.exe app.py stats`，记录 metadata、PDF、解析、画像、索引、仅摘要和失败数量。
- [ ] 只有在本地 seed 完成后，才把 20 问评估集中的占位符替换为真实 `paper_id/chunk_id`。

## 页面演示顺序

1. `科研助手`：先展示“叶绿素是什么”这类证据不足问题，确认页面给出带边界的通用解释；再展示一次 evidence QA，说明 Agent 只路由到受控 Skill。
2. `运行诊断`：展开本轮 route、skill、检索候选数、证据数、引用数、fallback 和耗时。
3. `知识库维护`：展示 704 篇元数据、116 篇可定位全文和向量一致性审计结果。
4. `文献库`：按标题、DOI 和状态筛选 indexed、abstract-only 或 failed 论文，说明证据层级。
5. 论文对比与趋势报告已从工作台入口移除，不作为演示链路；重点展示证据问答与研究路线。
6. 普通问候走 `general_chat`，科研问题走受控技能并展示引用；两条路径都在聊天区即时显示用户消息。
7. `任务与日志`、`数据源设置`：展示本地同步记录和点击触发的标准化健康状态，不是通用任务平台。

## 截图要求

- [ ] 截图只包含页面、脱敏标题和稳定状态码。
- [ ] 遮挡 API key、邮箱、绝对路径、完整本地文件名和未公开全文。
- [ ] 不使用伪造的 450–500 条统计数字、召回率或问答质量数字。
- [ ] 空数据库截图可以作为安装后的初始状态，真实数据截图必须来自本机实际运行。
