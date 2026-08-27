# Evidence Groups And Two-Stage Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flat Chunk relevance with auditable evidence groups, expand the balanced retrieval set to 48 questions, and measure a paper-first Two-stage retriever against the existing three retrieval modes.

**Architecture:** `evaluation.dataset` owns the evidence-group contract and exposes derived paper/Chunk unions. `evaluation.metrics` measures group recall, while `evaluation.run` evaluates four retrievers and applies the promotion gate. A new `retrieval.two_stage` wrapper aggregates existing BM25/BGE-M3 Chunk rankings into candidate papers and delegates scoped evidence ranking to `HybridRetriever`; it creates no new index.

**Tech Stack:** Python 3.12, dataclasses, Pydantic evidence models, SQLite FTS5, FAISS/BGE-M3, pytest, YAML configuration.

---

## File Map

- Modify `evaluation/dataset.py`: canonical evidence-group schema, validation, and derived ID unions.
- Modify `evaluation/metrics.py`: evidence-group Recall@k.
- Modify `evaluation/run.py`: four-mode evaluation, trace projection, metric labels, and promotion gate.
- Create `retrieval/two_stage.py`: paper aggregation and scoped second-stage retrieval.
- Modify `app.py`: construct Two-stage for evaluation and conditionally use it online only after the measured gate passes.
- Modify `config/rag.yml`: bounded Two-stage candidate settings.
- Modify `data/evaluation/questions.jsonl`: migrate the placeholder template to the canonical group shape.
- Modify `data/evaluation/questions-annotated.jsonl`: reviewed 48-question evidence-group dataset.
- Modify `tests/test_evaluation.py`: schema, metrics, four-mode reporting, gate, CLI wiring, and dataset integrity tests.
- Create `tests/test_two_stage_retrieval.py`: isolated paper aggregation, scope, fallback, bounds, and trace tests.
- Modify `docs/research-lens-handoff-2026-08-22.md`: dated P0/P1 outcome and truthful metrics after the run.

### Task 1: Make Evidence Groups The Canonical Dataset Contract

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `evaluation/dataset.py`
- Modify: `data/evaluation/questions.jsonl`

- [ ] **Step 1: Replace the dataset helper and add failing evidence-group validation tests**

Use this canonical fixture shape in `tests/test_evaluation.py`:

```python
def _annotated(question_id="q01", question="Which sensor?"):
    return {
        "question_id": question_id,
        "question": question,
        "category": "exact_term",
        "evidence_groups": [
            {
                "paper_id": "paper-1",
                "chunk_ids": ["paper-1:p3:c1", "paper-1:p3:c2"],
                "rationale": "Either passage directly identifies the sensor.",
            }
        ],
    }
```

Add tests asserting that `load_questions()`:

```python
question = load_questions(path)[0]
assert question.relevant_paper_ids == ("paper-1",)
assert question.relevant_chunk_ids == ("paper-1:p3:c1", "paper-1:p3:c2")
assert question.evidence_groups[0].rationale.startswith("Either passage")
```

Parameterize invalid cases for an empty group list, empty rationale, duplicate Chunk IDs, a Chunk whose prefix differs from `paper_id`, duplicate groups, unknown top-level fields, and legacy `relevant_paper_ids`/`relevant_chunk_ids`. Require stable codes `evaluation_dataset_invalid` or `evaluation_dataset_invalid_evidence_group` without echoing rejected content.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -q --basetemp .pytest-tmp-evidence-schema-red
```

Expected: failures show that `evidence_groups` is currently rejected and `EvaluationQuestion` has no group contract.

- [ ] **Step 3: Implement the minimal immutable evidence-group model and parser**

In `evaluation/dataset.py`, introduce:

```python
@dataclass(frozen=True)
class EvidenceGroup:
    paper_id: str
    chunk_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    question: str
    evidence_groups: tuple[EvidenceGroup, ...]
    category: str = "uncategorized"
    placeholder: bool = False

    @property
    def relevant_paper_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(group.paper_id for group in self.evidence_groups))

    @property
    def relevant_chunk_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                chunk_id
                for group in self.evidence_groups
                for chunk_id in group.chunk_ids
            )
        )
```

Add `_parse_evidence_groups()` that accepts only `paper_id`, `chunk_ids`, and `rationale`, applies existing sensitive/placeholder checks, checks `chunk_id.startswith(f"{paper_id}:")`, and rejects duplicate normalized group tuples. Change `_parse_row()` allowed keys to `question_id`, `question`, `category`, and `evidence_groups`. Include group contents in duplicate-row detection and `_fingerprint()`.

- [ ] **Step 4: Migrate the 20-row placeholder template**

Rewrite each `data/evaluation/questions.jsonl` row from the two flat arrays to one `evidence_groups` entry while retaining its current placeholder IDs. Keep `template_mode=True` as the only path that accepts those placeholders.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the same focused command. Expected: dataset and metric-independent evaluation tests pass; tests still using flat fixtures fail only where their migration is intentionally deferred to Task 2.

- [ ] **Step 6: Commit the schema change**

```powershell
git add -- evaluation/dataset.py data/evaluation/questions.jsonl tests/test_evaluation.py
git commit -m "feat: model evaluation evidence groups"
```

### Task 2: Measure Required Evidence Groups Instead Of Flat IDs

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `evaluation/metrics.py`
- Modify: `evaluation/run.py`

- [ ] **Step 1: Add failing metric tests for alternatives and independent requirements**

Import `evidence_group_recall_at_k` and add:

```python
def test_evidence_group_recall_treats_chunks_within_a_group_as_alternatives():
    groups = [("p1:c1", "p1:c2"), ("p2:c1",)]
    assert evidence_group_recall_at_k(["p1:c2", "p2:c1"], groups, 5) == 1.0
    assert evidence_group_recall_at_k(["p1:c1"], groups, 5) == 0.5


def test_evidence_group_recall_deduplicates_ranked_ids_and_validates_groups():
    assert evidence_group_recall_at_k(["p1:c1", "p1:c1"], [("p1:c1",)], 5) == 1.0
    with pytest.raises(ValueError, match="^evaluation_metric_invalid$"):
        evidence_group_recall_at_k(["p1:c1"], [()], 5)
```

- [ ] **Step 2: Run the metric tests and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -q --basetemp .pytest-tmp-group-metric-red
```

Expected: import failure for `evidence_group_recall_at_k`.

- [ ] **Step 3: Implement group recall**

Add to `evaluation/metrics.py`:

```python
def evidence_group_recall_at_k(
    ranked_ids: Sequence[str],
    relevant_groups: Iterable[Iterable[str]],
    k: int,
) -> float:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("evaluation_metric_invalid")
    ranked = set(_validated_ids(ranked_ids)[:k])
    groups = [set(_validated_ids(group)) for group in relevant_groups]
    if any(not group for group in groups):
        raise ValueError("evaluation_metric_invalid")
    return sum(bool(ranked & group) for group in groups) / len(groups) if groups else 0.0
```

- [ ] **Step 4: Migrate evaluator fixtures and metric labels**

Change all question fixtures in `tests/test_evaluation.py` to `evidence_groups`. In `evaluation/run.py`, compute:

```python
groups = [group.chunk_ids for group in question.evidence_groups]
group_recall = evidence_group_recall_at_k(ranked_chunk_ids, groups, retrieval_k)
```

Replace every retrieval `recall_at_5` key and Markdown heading with `evidence_group_recall_at_5` / `Evidence-group Recall@5`. Keep `reciprocal_rank()` over the derived Chunk union and `recall_at_k()` for distinct paper IDs.

- [ ] **Step 5: Run focused tests and verify GREEN**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -q --basetemp .pytest-tmp-group-metric-green
```

Expected: all evaluation unit tests pass except tests intentionally awaiting the fourth mode or 48-row dataset.

- [ ] **Step 6: Commit the metric migration**

```powershell
git add -- evaluation/metrics.py evaluation/run.py tests/test_evaluation.py
git commit -m "feat: score retrieval by evidence groups"
```

### Task 3: Review And Expand The Dataset To 48 Questions

**Files:**
- Modify: `data/evaluation/questions-annotated.jsonl`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Add the failing checked-in dataset integrity test**

Update the existing dataset test to require:

```python
assert len(questions) == 48
assert {
    category: sum(question.category == category for question in questions)
    for category in {"exact_term", "natural_language", "cross_paper"}
} == {"exact_term": 16, "natural_language": 16, "cross_paper": 16}
assert all(group.rationale.strip() for question in questions for group in question.evidence_groups)
```

Use `ResearchDatabase(default_database_path())` to assert that every derived paper ID resolves with `get_paper()` and every derived Chunk ID resolves with `get_chunks_by_ids()`.

- [ ] **Step 2: Run the dataset test and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py::test_annotated_dataset_contains_concrete_local_evidence_ids -q --basetemp .pytest-tmp-dataset48-red
```

Expected: failure because the file still has 36 flat-schema rows.

- [ ] **Step 3: Audit the current 36 questions against trusted local evidence**

For each current question, load every annotated Chunk plus immediate neighbors through `ResearchDatabase.get_chunks_with_context(chunk_ids, window=1)`. Convert each independently required fact into its own group. Put genuinely interchangeable neighboring passages in the same group. Write a one-sentence rationale that states the paper's role and the fact supported; do not copy raw evidence into the rationale.

Review all 12 existing `cross_paper` rows first. Verify that comparison questions retain one required group per compared contribution and do not collapse two papers into one group.

- [ ] **Step 4: Add four exact-term questions grounded in unused local passages**

Use direct terms present in the local corpus, covering these four distinct targets: phycocyanin absorption/reflectance wavelengths, XGBoost versus random-forest TSI metrics, atmospheric-correction dependence of water reflectance, and Secchi-depth inversion terminology. Each question must have one or more evidence groups whose IDs and rationale are verified from the local database.

- [ ] **Step 5: Add four natural-language questions grounded in unused local passages**

Cover these four distinct concepts without copying the source wording: why atmospheric correction is harder over inland waters, why field spectra reveal cyanobacteria features better than coarse multispectral data, what a Secchi measurement represents, and how spectral band combinations support trophic-state estimation. Verify each answerable contribution against the selected Chunk and neighbors.

- [ ] **Step 6: Add four cross-paper questions grounded in at least two contributions**

Cover these comparisons: Secchi-based transparency monitoring versus turbidity instrumentation; field hyperspectral cyanobacteria evidence versus satellite spectral criteria; atmospheric-correction evaluation versus constituent/depth retrieval; and model-performance evidence for trophic-state estimation versus water-quality-index estimation. Each row must contain at least two evidence groups and normally at least two distinct relevant papers.

- [ ] **Step 7: Run dataset validation and inspect the deterministic fingerprint**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py::test_annotated_dataset_contains_concrete_local_evidence_ids -q --basetemp .pytest-tmp-dataset48-green
.venv\Scripts\python.exe -c "from evaluation.dataset import load_questions; q=load_questions('data/evaluation/questions-annotated.jsonl'); print(len(q), {c:sum(x.category==c for x in q) for c in sorted({x.category for x in q})})"
```

Expected: the test passes and the command prints `48` with `16` for each category.

- [ ] **Step 8: Commit the reviewed dataset**

```powershell
git add -- data/evaluation/questions-annotated.jsonl tests/test_evaluation.py
git commit -m "test: expand reviewed retrieval benchmark"
```

### Task 4: Implement Paper-First Two-Stage Retrieval

**Files:**
- Create: `retrieval/two_stage.py`
- Create: `tests/test_two_stage_retrieval.py`

- [ ] **Step 1: Write failing paper aggregation tests**

Create recording retrievers and test that repeated Chunks from one paper collapse before fusion:

```python
def test_stage_one_collapses_chunk_rankings_to_unique_papers_before_rrf():
    keyword = RecordingRetriever([chunk("p1:c1"), chunk("p1:c2"), chunk("p2:c1")])
    vector = RecordingRetriever([chunk("p2:c2"), chunk("p3:c1")])
    retriever = TwoStageRetriever(
        keyword,
        vector,
        keyword_weight=1.0,
        vector_weight=1.0,
        paper_candidate_k=10,
        paper_k=2,
        chunk_candidate_k=8,
        max_chunks_per_paper=2,
    )

    retriever.search("water quality", k=5)

    assert retriever.last_trace.candidate_paper_ids == ["p2", "p1"]
    assert keyword.calls[0] == ("water quality", 10, None)
    assert vector.calls[0] == ("water quality", 10, None)
```

- [ ] **Step 2: Write failing scope, fallback, bounds, and trace tests**

Cover these exact contracts:

```python
assert retriever.search("q", k=0) == []
assert retriever.search("q", k=5, paper_ids=[]) == []
assert all(item.paper_id in {"p2"} for item in retriever.search("q", k=5, paper_ids=["p2"]))
assert len(retriever.search("q", k=3)) <= 3
assert retriever.last_trace.fallback_used is False
```

Add a no-candidate case proving unscoped search calls Stage 2 once with `paper_ids=None`, while a caller scope with no Stage 1 hit returns empty and is never widened. Assert the trace contains IDs/counts only and no Chunk text.

- [ ] **Step 3: Run the new test file and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_two_stage_retrieval.py -q --basetemp .pytest-tmp-two-stage-red
```

Expected: import failure because `retrieval.two_stage` does not exist.

- [ ] **Step 4: Implement stable paper RRF and bounded trace**

Create `retrieval/two_stage.py` with:

```python
@dataclass(frozen=True)
class TwoStageTrace:
    query: str
    stage1_keyword_candidates: int = 0
    stage1_vector_candidates: int = 0
    candidate_paper_ids: list[str] = field(default_factory=list)
    stage2_candidates: int = 0
    selected_count: int = 0
    selected_chunk_ids: list[str] = field(default_factory=list)
    selected_paper_ids: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    fallback_used: bool = False
    latency_ms: float = 0.0
```

Implement `_unique_papers(chunks)` and `_paper_rrf(keyword_ids, vector_ids, *, keyword_weight, vector_weight, rrf_k)` with one-based ranks and stable paper-ID tie breaking. Validate finite non-negative weights, non-negative `rrf_k`, and positive candidate limits.

- [ ] **Step 5: Implement `TwoStageRetriever.search()` minimally**

Construct an internal `HybridRetriever` using the same weights and `rrf_k`, but `candidate_k=chunk_candidate_k` and the Two-stage `max_chunks_per_paper`. Search both Stage 1 branches with `paper_candidate_k`, fuse unique paper lists, restrict to `paper_k`, and delegate to internal Hybrid with those IDs. Preserve explicit caller scopes and the unscoped no-candidate fallback.

- [ ] **Step 6: Run focused tests and verify GREEN**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_two_stage_retrieval.py tests/test_hybrid_retrieval.py -q --basetemp .pytest-tmp-two-stage-green
```

Expected: both files pass with no change to current Hybrid behavior.

- [ ] **Step 7: Commit the retriever**

```powershell
git add -- retrieval/two_stage.py tests/test_two_stage_retrieval.py
git commit -m "feat: add paper-first two-stage retrieval"
```

### Task 5: Add Four-Mode Evaluation And The Promotion Gate

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `evaluation/run.py`

- [ ] **Step 1: Add a failing fourth-mode evaluation test**

Extend `_evaluation_fixture()` with a `two_stage` ranking and pass `two_stage_retriever`. Assert event order includes all four modes and report keys are exactly `keyword`, `vector`, `hybrid`, and `two_stage` plus `overall`.

- [ ] **Step 2: Add failing promotion-gate tests**

Use a three-category fixture and assert:

```python
assert result.acceptance["accepted"] is True
assert result.acceptance["code"] == "two_stage_promotion_gate_passed"
```

Add parameterized failures for lower cross-paper paper recall, lower cross-paper evidence-group recall, no strict cross-paper improvement, and lower natural-language evidence-group recall. Require `two_stage_promotion_gate_failed` and individual boolean checks in the report.

- [ ] **Step 3: Run evaluation tests and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -q --basetemp .pytest-tmp-four-mode-red
```

Expected: failures because the evaluator still has three hard-coded modes and the old acceptance rule.

- [ ] **Step 4: Generalize mode handling and safe trace projection**

Require either `two_stage_retriever` or `two_stage_factory`, build it after Hybrid, and define one ordered modes tuple used by loops and Markdown. Extend `_safe_retrieval_trace()` with bounded optional Two-stage fields:

```python
"stage1_keyword_candidates": int(values.get("stage1_keyword_candidates", 0)),
"stage1_vector_candidates": int(values.get("stage1_vector_candidates", 0)),
"candidate_paper_ids": [str(item) for item in values.get("candidate_paper_ids", [])[:16]],
"stage2_candidates": int(values.get("stage2_candidates", 0)),
```

- [ ] **Step 5: Implement the promotion gate as a pure helper**

Add `_two_stage_acceptance(metrics_by_category)` that compares Two-stage with Hybrid and returns:

```python
{
    "accepted": all(checks.values()),
    "code": (
        "two_stage_promotion_gate_passed"
        if all(checks.values())
        else "two_stage_promotion_gate_failed"
    ),
    "checks": checks,
}
```

The checks are the four conditions in the approved specification, with strict improvement represented as one named boolean.

- [ ] **Step 6: Run evaluation tests and verify GREEN**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -q --basetemp .pytest-tmp-four-mode-green
```

Expected: all evaluation tests pass, reports contain no evidence text or local paths, and all metric labels use evidence-group terminology.

- [ ] **Step 7: Commit four-mode evaluation**

```powershell
git add -- evaluation/run.py tests/test_evaluation.py
git commit -m "feat: evaluate two-stage retrieval promotion"
```

### Task 6: Wire Configuration, Run The Ablation, And Apply The Gate

**Files:**
- Modify: `config/rag.yml`
- Modify: `app.py`
- Modify: `tests/test_evaluation.py`
- Conditionally modify: `app.py` QA retriever constructors only when the real gate passes.
- Modify: `docs/research-lens-handoff-2026-08-22.md`

- [ ] **Step 1: Add failing configuration and factory-wiring tests**

Assert that `DefaultServices.evaluate()` passes both factories and this exact bounded configuration shape:

```python
"two_stage": {
    "paper_candidate_k": 50,
    "paper_k": 12,
    "chunk_candidate_k": 40,
    "max_chunks_per_paper": 2,
}
```

Add invalid-config cases for zero/fractional/oversized limits and unknown keys, all returning `evaluation_config_invalid` without writing reports.

- [ ] **Step 2: Run wiring tests and verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -q --basetemp .pytest-tmp-two-stage-config-red
```

Expected: factory/config assertions fail because the CLI does not construct Two-stage yet.

- [ ] **Step 3: Add Two-stage configuration and CLI construction**

Append to `config/rag.yml`:

```yaml
two_stage:
  paper_candidate_k: 50
  paper_k: 12
  chunk_candidate_k: 40
  max_chunks_per_paper: 2
```

Extend `_validated_config()` with exact-key and integer bounds: `paper_candidate_k` and `chunk_candidate_k` from 5 through 200, `paper_k` from 2 through 50, and `max_chunks_per_paper` from 1 through 5. In `DefaultServices.evaluate()`, pass the settings and a `_evaluation_two_stage` factory that imports `TwoStageRetriever` lazily.

- [ ] **Step 4: Run focused and full automated tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py tests/test_two_stage_retrieval.py tests/test_hybrid_retrieval.py -q --basetemp .pytest-tmp-p1-focused
.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-p1-full
```

Expected focused result: all pass. Expected full result: no failures beyond the previously observed `test_readme_handoff_is_truthful_and_contains_required_boundaries`; if that baseline test now passes, the full suite must be completely green.

- [ ] **Step 5: Run the real 48-question retrieval-only ablation once**

```powershell
.venv\Scripts\python.exe app.py evaluate --dataset data\evaluation\questions-annotated.jsonl --retrieval-only
```

Read the generated local JSON report and record Keyword, Vector, Hybrid, and Two-stage overall/category metrics plus every promotion check. Do not add the report file to Git.

- [ ] **Step 6: Apply the promotion decision exactly**

If the report code is `two_stage_promotion_gate_passed`, change the two default QA construction sites in `app.py` to use `TwoStageRetriever` with the validated `rag.yml` values and add tests showing scoped and ordinary QA receive it. If the code is `two_stage_promotion_gate_failed`, leave both QA construction sites on `HybridRetriever`; add a test or assertion documenting that evaluation construction includes Two-stage while default QA construction does not.

- [ ] **Step 7: Re-run verification after the conditional wiring**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evaluation.py tests/test_two_stage_retrieval.py tests/test_qa.py -q --basetemp .pytest-tmp-promotion-green
.venv\Scripts\python.exe app.py knowledge-audit --json
git diff --check
```

Expected: focused tests pass; audit reports `chunks_total=17992`, `vector_indexed=17992`, and empty missing/orphan ID lists; diff check emits no errors.

- [ ] **Step 8: Update the handoff with only measured claims**

Record the new dataset size/fingerprint, explain the evidence-group metric break from the old flat-ID Recall, list the exact four-mode overall/category metrics, and state whether Two-stage was promoted. Preserve the existing warning that retrieval-only metrics are not online answer accuracy.

- [ ] **Step 9: Commit the integration and measured handoff**

```powershell
git add -- app.py config/rag.yml tests/test_evaluation.py docs/research-lens-handoff-2026-08-22.md
git commit -m "feat: integrate evaluated two-stage retrieval"
```

## Final Verification

- [ ] Confirm `git status --short` contains no newly generated report, SQLite, PDF, FAISS, `.env`, or credential file staged by this work.
- [ ] Confirm the 48-question dataset loads with 16 questions in each category and every evidence group resolves to trusted local Chunk IDs.
- [ ] Confirm the report contains all four modes and the promotion decision matches its component checks.
- [ ] Confirm no BGE-M3 model download or FAISS rebuild occurred.
- [ ] Report the known README baseline failure separately if it remains; do not claim a fully green suite unless the final command proves it.
