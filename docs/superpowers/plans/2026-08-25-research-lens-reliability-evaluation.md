# Research Lens Reliability And Evaluation Implementation Plan

Status: completed and verified on 2026-08-25 (`311 passed, 1 skipped`).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make snapshot documentation truthful, FAISS publication recoverable, S2 retries reproducible, and answer-level citation evaluation measurable within the personal MVP.

**Architecture:** Keep SQLite as the source of paper/chunk truth. Build FAISS in a verified temporary directory and publish it as one snapshot. Add a provider-scoped retry mode at the existing CLI/workflow boundary, and keep answer-level evaluation separate from retrieval-only evaluation.

**Tech Stack:** Python, pytest, SQLite, FAISS/LangChain, Pydantic, existing argparse CLI.

---

### Task 1: Add regression tests for the reliability contracts

**Files:**
- Modify: `tests/test_hybrid_retrieval.py`
- Modify: `tests/test_retry_fulltext.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_seed.py`

- [x] Write tests that assert a vector rebuild publishes only after the temporary build succeeds and that the audit exposes vector coverage separately from paper status.
- [x] Write a retry test asserting a Semantic Scholar-only scope invokes one resolver request per DOI and forwards rate/timeout settings through the CLI service boundary.
- [x] Write answer-level metric tests with two synthetic answers covering valid citations, irrelevant citations, and unsupported confident claims.
- [x] Run the focused tests and confirm they fail for the expected missing contracts.

### Task 2: Make FAISS publication recoverable

**Files:**
- Modify: `rag/vector_store.py`
- Modify: `workflows/knowledge_audit.py`
- Modify: `tests/test_hybrid_retrieval.py`

- [x] Add a temporary-directory publication helper that saves a complete FAISS snapshot and manifest before switching the active directory.
- [x] Preserve the existing active index if embedding or save fails.
- [x] Add a load-time consistency error with a stable project error code instead of exposing the raw FAISS read exception.
- [x] Report `vector_indexed` counts and missing/orphan IDs independently of `papers.status`.
- [x] Run the focused vector/audit tests.

### Task 3: Add Semantic Scholar provider scope to retry CLI

**Files:**
- Modify: `providers/registry.py`
- Modify: `workflows/retry_fulltext.py`
- Modify: `app.py`
- Modify: `tests/test_providers.py`
- Modify: `tests/test_retry_fulltext.py`
- Modify: `tests/test_seed.py`

- [x] Add an explicit provider scope with `semantic_scholar` as the first supported value.
- [x] Ensure the scoped resolver does not invoke the same Semantic Scholar DOI resolver again as a fallback during one attempt.
- [x] Add CLI flags for provider, request rate, and download timeout with bounded validation.
- [x] Preserve stable failure codes and existing resumability by selecting only eligible DOI records.
- [x] Run CLI and provider tests.

### Task 4: Add answer-level citation evaluation

**Files:**
- Create: `data/evaluation/answers-annotated.jsonl`
- Modify: `evaluation/dataset.py`
- Modify: `evaluation/metrics.py`
- Modify: `evaluation/run.py`
- Modify: `tests/test_evaluation.py`

- [x] Define a sanitized answer-evaluation row containing question ID, expected relevant chunk IDs, answer claims, and citations.
- [x] Validate the row size, IDs, quotes, and sensitive-content boundary.
- [x] Calculate citation precision, evidence coverage, and unsupported-claim rate from supplied answer rows only.
- [x] Keep `--retrieval-only` output explicitly free of answer-quality claims.
- [x] Add a small set of manually reviewed rows based on current local evidence without storing full paper text.

### Task 5: Synchronize snapshot documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/interview/ai-engineer-project-positioning.md`
- Modify: `docs/research-lens-handoff-2026-08-22.md`

- [x] Replace stale 96/97/14630 snapshot values with the latest verified 116/17992 values and timestamp them.
- [x] Add the latest 36-question retrieval-only result and explicitly state that keyword currently beats hybrid.
- [x] Document the reproducible S2 CLI and answer-level evaluation boundary.
- [x] Scan for stale historical claims and run `git diff --check`.

### Task 6: Full verification

- [x] Run focused tests for each changed boundary.
- [x] Run the full pytest suite.
- [x] Run `app.py knowledge-audit --json` and verify missing/orphan vectors are empty.
- [x] Run the 36-question retrieval-only evaluation and record its current report name without committing runtime reports.
