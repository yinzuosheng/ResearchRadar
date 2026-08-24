# Research Lens Interview Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing RAG/Agent project easier to evaluate in an AI-engineering interview by narrowing its narrative, adding honest evaluation diagnostics, and exposing safe Agent runtime diagnostics.

**Architecture:** Preserve the current deterministic ingestion and bounded route-to-skill Agent. Add optional evaluation categories at the dataset/report boundary and add a non-persistent diagnostics contract assembled from existing retrieval/QA traces. Keep UI rendering pure and sanitized.

**Tech Stack:** Python 3.12, Pydantic, LangGraph, SQLite/FTS5, FAISS, Streamlit, pytest.

---

### Task 1: Align interview narrative and project scope

**Files:**
- Modify: `README.md`
- Modify: `docs/interview/ai-engineer-project-positioning.md`
- Modify: `docs/demo/research-workbench-demo-checklist.md`
- Test: `tests/test_evaluation.py` (existing README truthfulness test)

- [ ] **Step 1: Write the documentation assertions first**

Extend `test_readme_handoff_is_truthful_and_contains_required_boundaries` with these required strings: `核心技术主线`, `受控科研 Agent`, `辅助能力`, `704 是元数据目录`, `96 是可定位全文`, and `不写 Recall、MRR 或问答质量百分比`. Add an assertion that README contains `不新增全文 provider、OCR、FastAPI`.

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluation.py::test_readme_handoff_is_truthful_and_contains_required_boundaries -q --basetemp=.pytest-tmp-plan-readme`

Expected: FAIL because the new narrative markers are not yet present.

- [ ] **Step 3: Update the narrative documents**

Add one concise section to README immediately after the architecture/data-flow section:

```markdown
## 核心技术主线（面试口径）

受控科研 Agent + 证据型 RAG + 可审计后端工作流：合法 OA 摄入 -> 证据分层 -> FTS5/FAISS 检索 -> 引用校验 -> 结构化回答。

论文画像、对比、趋势和 Streamlit 导航是辅助能力。704 是元数据目录，96 是可定位全文；摘要证据单独标记。未完成人工标注和真实运行前，不写 Recall、MRR 或问答质量百分比。个人 MVP 不新增全文 provider、OCR、FastAPI、多用户或云部署。
```

Update the interview positioning document with three final bullets, a 30-second pitch, and a “面试官追问” section. Update the demo checklist so the first demo flow is Agent -> evidence citation -> runtime diagnostics -> knowledge audit, and label comparison/trends as optional.

- [ ] **Step 4: Run the focused test and verify it passes**

Run the same pytest command; expected: PASS.

### Task 2: Add optional evaluation categories

**Files:**
- Modify: `evaluation/dataset.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing dataset tests**

Add tests:

```python
def test_dataset_accepts_optional_category_and_defaults_to_uncategorized(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"question_id": "q1", "question": "q", "relevant_paper_ids": ["p1"], "relevant_chunk_ids": ["p1:c1"]}) + "\n", encoding="utf-8")
    assert load_questions(path)[0].category == "uncategorized"

def test_dataset_rejects_invalid_category_without_echoing_value(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"question_id": "q1", "question": "q", "category": "自然语言", "relevant_paper_ids": ["p1"], "relevant_chunk_ids": ["p1:c1"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="^evaluation_dataset_invalid_category$"):
        load_questions(path)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -k "optional_category or invalid_category" -q --basetemp=.pytest-tmp-plan-category`

Expected: FAIL because `EvaluationQuestion` has no `category` field and the parser allow-list rejects `category`.

- [ ] **Step 3: Implement minimal category parsing**

Add `category: str = "uncategorized"` to `EvaluationQuestion`. Extend the allow-list with `category`, validate non-empty ASCII tokens matching `^[a-z][a-z0-9_]{0,31}$`, and raise `DatasetError("evaluation_dataset_invalid_category")` for invalid values. Include the normalized category in the dataclass and in the dataset fingerprint canonical payload.

- [ ] **Step 4: Run the focused tests and the existing dataset tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -k "category or dataset" -q --basetemp=.pytest-tmp-category-green`

Expected: all selected tests PASS.

### Task 3: Add per-category evaluation reporting

**Files:**
- Modify: `evaluation/run.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing report assertions**

Update the existing evaluator fixture so q1 has `category="exact_term"` and q2 has `category="natural_language"`. Assert the JSON report contains:

```python
assert payload["dataset_summary"] == {
    "question_count": 2,
    "category_counts": {"exact_term": 1, "natural_language": 1},
    "evaluation_scope": "answer_and_retrieval",
}
assert payload["metrics_by_category"]["exact_term"]["keyword"]["recall_at_5"] == 1.0
```

Also add a retrieval-only assertion that `dataset_summary["evaluation_scope"] == "retrieval_only"` and the markdown contains `仅代表当前本地语料版本` for a two-question dataset.

- [ ] **Step 2: Run the report tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -k "category or report" -q --basetemp=.pytest-tmp-plan-report`

Expected: FAIL because the report payload has no `dataset_summary` or `metrics_by_category`.

- [ ] **Step 3: Implement category aggregation**

After loading questions, build `category_counts`. During each mode loop, keep per-category recall, reciprocal rank, and paper recall lists. Add `metrics_by_category` to `metrics` using the same `deterministic_mean` helpers and add `dataset_summary` to the JSON payload. Keep the existing overall metrics and acceptance calculation unchanged. Add the small-dataset/retrieval-only caution line in `_markdown` without including question text, raw evidence, paths, or secrets.

- [ ] **Step 4: Run focused and full evaluation tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -q --basetemp=.pytest-tmp-evaluation-green`

Expected: PASS with all existing report shape and sanitization assertions intact.

### Task 4: Add safe Agent runtime diagnostics

**Files:**
- Modify: `domain/models.py`
- Modify: `agent/research_agent.py`
- Modify: `tests/test_research_agent.py`

- [ ] **Step 1: Write failing diagnostics tests**

Add tests that construct fake QA/retrieval services with `last_trace`, then assert a successful turn has `reply.diagnostics.skill_id == "evidence_qa"`, numeric counts, and no query/prompt text. Add a fallback-router test asserting `route_mode == "fallback"`. Add a no-trace test asserting zero-valued diagnostics and no exception.

- [ ] **Step 2: Run diagnostics tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_research_agent.py -k diagnostics -q --basetemp=.pytest-tmp-plan-agent`

Expected: FAIL because `ResearchAgentReply` has no diagnostics field.

- [ ] **Step 3: Implement the diagnostics contract and assembly**

Add `AgentDiagnostics` with bounded fields and `diagnostics: AgentDiagnostics | None = None` to `ResearchAgentReply`. In the route node, track `route_mode` and pass it through graph state. In `use_tool`, read `last_trace` from the selected workflow/retriever, normalize finite non-negative numbers, and assemble counts for candidates, canonical evidence, and citations. Measure total turn duration in `chat`. Do not persist diagnostics to SQLite or include raw query/evidence in the model.

- [ ] **Step 4: Run Agent tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_research_agent.py -q --basetemp=.pytest-tmp-agent-green`

Expected: PASS, including all prior budget, memory, and citation behavior tests.

### Task 5: Render diagnostics in the Streamlit workbench

**Files:**
- Modify: `web/app.py`
- Modify: `web/presenters.py`
- Test: `tests/test_workbench_presenters.py`
- Test: `tests/test_workbench_navigation.py`

- [ ] **Step 1: Write failing presenter test**

Add a pure helper test for `render_agent_diagnostics` using a mapping containing `skill_id`, `route_mode`, `retrieval_candidates`, `evidence_chunks`, `citation_count`, `fallback`, and finite timings. Assert the rendered text contains labels and never contains a URL, Windows path, or arbitrary unknown key.

- [ ] **Step 2: Run the presenter test and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_workbench_presenters.py -k diagnostics -q --basetemp=.pytest-tmp-plan-ui`

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement pure sanitized rendering and wire the page**

Implement `render_agent_diagnostics(value)` in `web/presenters.py`. Accept mappings or Pydantic objects, whitelist the seven fields, use existing `escape_untrusted`, clamp invalid numeric values to `0.0`, and return a short Markdown block. In `render_research_agent_page`, render an expander below each assistant message only when `message.diagnostics` is present. Preserve existing citation and evidence-insufficient branches.

- [ ] **Step 4: Run UI-focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_workbench_presenters.py tests/test_workbench_navigation.py -q --basetemp=.pytest-tmp-ui-green`

Expected: PASS.

### Task 6: Final verification and interview artifact review

**Files:**
- Verify: all modified files and `docs/superpowers/specs/2026-08-24-research-lens-interview-quality-design.md`

- [ ] **Step 1: Run the complete test suite**

Run: `.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp-interview-quality`

Expected: zero failures; report the exact passed/skipped counts.

- [ ] **Step 2: Run live local audits**

Run: `.venv\Scripts\python.exe app.py stats` and `.venv\Scripts\python.exe app.py knowledge-audit --json`.

Expected: no credentials printed; `missing_chunk_ids` and `orphan_vector_ids` remain empty.

- [ ] **Step 3: Review the final diff for scope and data claims**

Run: `git diff -- README.md docs/interview/ai-engineer-project-positioning.md docs/demo/research-workbench-demo-checklist.md evaluation/dataset.py evaluation/run.py domain/models.py agent/research_agent.py web/presenters.py web/app.py tests`

Confirm no new provider, OCR, cloud, account, PDF, SQLite, FAISS, `.env`, URL, or fabricated evaluation number was added to tracked files.
