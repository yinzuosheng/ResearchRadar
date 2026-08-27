# Evidence Groups And Two-Stage Retrieval Design

**Goal:** Make retrieval evaluation tolerant of valid alternative evidence, then test a paper-first retrieval path that improves cross-paper questions without weakening the current natural-language baseline.

## Scope

1. Replace flat relevant Chunk annotations with auditable evidence groups for the retrieval dataset.
2. Review the existing 12 `cross_paper` questions and expand the balanced dataset from 36 to 48 questions: 16 `exact_term`, 16 `natural_language`, and 16 `cross_paper`.
3. Add a two-stage experimental retriever that reuses the current BM25 and BGE-M3 indexes.
4. Compare Keyword, Vector, Hybrid, and Two-stage on the same annotated dataset and enforce an explicit promotion gate.

## Non-Goals

- Do not change or download an Embedding model.
- Do not rebuild FAISS unless the existing index becomes incompatible for an unrelated reason.
- Do not add a reranker, a separate paper vector index, a new SQLite table, or a new external service.
- Do not change answer generation or claim answer-level quality from retrieval-only results.
- Do not automatically enable Two-stage for online QA unless it passes the promotion gate.

## Evaluation Annotation Contract

Each JSONL question uses evidence groups as its canonical relevance annotation:

```json
{
  "question_id": "b13",
  "question": "How do global lake bloom monitoring and bloom forecasting studies differ in scale and input variables?",
  "category": "cross_paper",
  "evidence_groups": [
    {
      "paper_id": "paper-id-1",
      "chunk_ids": ["paper-id-1:abstract:c0"],
      "rationale": "Supports the global monitoring scale and observations."
    },
    {
      "paper_id": "paper-id-2",
      "chunk_ids": ["paper-id-2:abstract:c0"],
      "rationale": "Supports forecasting inputs and prediction scope."
    }
  ]
}
```

An evidence group represents one required evidence contribution. Its `chunk_ids` are alternatives: retrieving any one of them satisfies the group. Separate facts required to answer a comparison remain separate groups, even when they come from the same paper.

The loader validates that:

- every question has at least one group;
- every group has exactly one `paper_id`, one or more unique `chunk_ids`, and a concise non-empty `rationale`;
- every Chunk ID belongs to the group's paper;
- question IDs and rows remain unique;
- categories and sensitive-content restrictions retain their existing behavior.

`EvaluationQuestion` exposes derived unions of paper IDs and Chunk IDs for compatibility with existing report and citation code. The checked-in dataset uses only the new canonical schema; the loader does not maintain two competing annotation formats.

## Annotation Review Workflow

For every question, the reviewer loads the annotated Chunk, its immediate neighbors, and the paper metadata from the trusted local database. A group is accepted only when at least one listed Chunk directly supports the rationale. Nearby alternatives are added only when they independently support the same required contribution.

The four new questions per category must be based on existing local evidence and must not duplicate an existing question by paraphrase. The finished dataset remains balanced at 48 questions. Dataset tests verify that all referenced papers and Chunks exist and that every rationale is present; the review does not alter the SQLite corpus or FAISS index.

## Metrics

- **Evidence-group Recall@5:** number of required evidence groups with at least one alternative Chunk in the top five, divided by total required groups.
- **MRR:** reciprocal rank of the first retrieved Chunk that belongs to any evidence group.
- **Paper Recall@5:** number of distinct relevant papers represented in the top five results, divided by total distinct relevant papers.

The report labels the first metric `evidence_group_recall_at_5`. It must not reuse the old `recall_at_5` label because the relevance unit has changed. Category-level metrics remain mandatory.

## Two-Stage Retrieval

`TwoStageRetriever` wraps the existing keyword and vector retrievers and returns the same `EvidenceChunk` interface as `HybridRetriever`.

### Stage 1: Paper Recall

1. Search BM25 and BGE-M3 independently with a larger, fixed candidate pool.
2. Collapse each ranking to the first occurrence of each paper. This prevents papers with many near-duplicate Chunks from gaining an unfair frequency advantage.
3. Fuse the two paper rankings with the configured Weighted RRF weights.
4. Keep a bounded number of candidate paper IDs.

Stage 1 uses the title already present in FTS and in the `title-location-text-v1` vector representation. It creates no new index.

### Stage 2: Evidence Location

Run the existing `HybridRetriever` with `paper_ids` restricted to the Stage 1 candidates and with a larger internal Chunk candidate pool. Preserve existing duplicate removal and `max_chunks_per_paper`, then return exactly the requested `k` results at most.

The final ordering is the Stage 2 evidence score; Stage 1 controls eligibility but does not overwrite Chunk scores. This keeps the implementation explainable and isolates paper recall from paragraph location.

### Fallback And Trace

If Stage 1 produces no candidate paper, run the existing unscoped Hybrid retrieval once. An explicitly empty caller scope returns no result and does not fall back. A caller-provided `paper_ids` scope is intersected with Stage 1 candidates and is never widened.

The bounded trace records Stage 1 keyword/vector candidate counts, candidate paper IDs, Stage 2 candidate/result counts, fallback use, selected Chunk/paper IDs, query variants, and latency. It must not include Chunk text, model output, paths, or credentials.

## Integration Boundary

Evaluation adds `two_stage` as a fourth retrieval mode while retaining the three current baselines. CLI construction reads the same BM25/Dense weights from `config/rag.yml`; Two-stage-only candidate limits are explicit configuration values with bounded positive validation.

The default QA service continues using the current `HybridRetriever` during the experiment. Two-stage becomes the default only when a completed report passes the promotion gate and focused tests confirm scoped-search and fallback behavior.

## Promotion Gate

Compared with Hybrid on the same 48-question dataset, Two-stage must satisfy all of the following:

- `cross_paper` Paper Recall@5 does not decrease;
- `cross_paper` evidence-group Recall@5 does not decrease;
- at least one of those two `cross_paper` metrics strictly improves;
- `natural_language` evidence-group Recall@5 does not decrease;
- no evaluation service or full-suite regression is introduced.

If the gate fails, the report remains valid experimental evidence, but Two-stage is not connected to online QA and no improvement claim is made.

## Error Handling

- Invalid evidence-group records fail with stable, non-echoing dataset error codes.
- Invalid Two-stage limits or weights fail during construction/config validation.
- Underlying retrieval failures retain the evaluator's stable `evaluation_service_failed` boundary.
- Empty results are valid retrieval outcomes and are reported as zero recall rather than converted into errors.

## Testing And Verification

Implementation follows test-first development.

- Dataset tests cover schema validation, derived ID unions, alternative-Chunk group matching, paper ownership, balanced category counts, and existence of all local IDs.
- Metric tests distinguish alternative Chunks within one group from multiple required groups.
- Retriever tests cover paper-rank collapsing, Weighted RRF ordering, scoped Stage 2 calls, result bounds, duplicate limits, fallback, and trace sanitization.
- Evaluation tests cover the fourth mode, new metric labels, category output, configuration validation, and promotion-gate outcomes.
- Focused tests run after each behavior change, followed by the full suite.
- Run `knowledge-audit --json` to confirm the existing 17,992-Chunk index remains consistent.
- Run one retrieval-only evaluation on the reviewed 48-question dataset. Record exact results from the generated local report without committing the report.

The pre-existing README contract test failure is treated as a known baseline until separately corrected; this work must introduce no additional failures.
