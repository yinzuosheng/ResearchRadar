# Research Lens Reliability And Evaluation Design

**Goal:** Make the current local snapshot truthful, make vector indexing recoverable, expose a reproducible Semantic Scholar retry command, and add a small answer-level citation evaluation boundary without expanding the MVP.

## Scope

1. Synchronize README, interview positioning, and handoff documents to one dated local snapshot and the latest 36-question retrieval-only result.
2. Make FAISS rebuild publication recoverable by constructing a complete temporary index before replacing the active index. Expose vector coverage separately from paper ingestion status.
3. Add a Semantic Scholar-only retry scope to the CLI with explicit request rate and download timeout controls. The scope must be resumable through existing paper statuses and must not query the same DOI twice in one attempt.
4. Add a bounded answer/citation evaluation dataset and evaluator path. It must report citation precision, evidence coverage, and unsupported-claim rate only when answer data is supplied; retrieval-only runs must remain retrieval-only.

## Non-goals

- No new provider, OCR, async queue, Redis, FastAPI, cloud deployment, or multi-user features.
- No credentials, PDFs, SQLite databases, FAISS files, prompts, or model outputs in tracked files.
- No claim that hybrid retrieval is best; current keyword results remain the baseline until a measured change improves them.

## Contracts

- `papers.status` describes the ingestion lifecycle. Vector coverage is reported independently by `knowledge-audit` using chunk IDs.
- A published FAISS directory must have matching index/docstore/manifest files. An interrupted temporary build must not replace a valid active directory.
- Semantic Scholar retry defaults remain below 1 request/second. A provider-scoped run makes at most one S2 metadata request per DOI attempt.
- Answer-level evaluation consumes sanitized JSONL records with question ID, answer claims, citations, and expected relevant chunk IDs. It never infers QA quality from retrieval-only output.

## Verification

- Unit tests cover atomic rebuild publication, vector coverage reporting, provider-scoped CLI dispatch, one-request S2 scope, and answer-level metric calculations.
- Full pytest must pass.
- `app.py knowledge-audit --json` must report no missing or orphan vectors after rebuild.
- The current 36-question retrieval-only evaluation must run and document the measured keyword/vector/hybrid comparison.
