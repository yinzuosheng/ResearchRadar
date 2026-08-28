"""Read-only health report for the local research knowledge base."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any


class KnowledgeAuditService:
    """Compare catalog evidence with the persisted vector docstore."""

    def __init__(self, database: Any, vector_store: Any | None = None) -> None:
        self.database = database
        self.vector_store = vector_store

    def run(self) -> dict[str, Any]:
        papers = self.database.list_papers(limit=1_000_000)
        chunks = self.database.list_chunks()
        chunk_ids = {str(chunk.chunk_id) for chunk in chunks}
        paper_ids_with_chunks = {str(chunk.paper_id) for chunk in chunks}
        metadata_total = len(papers)
        abstract_evidence_papers = {
            str(paper.paper_id)
            for paper in papers
            if any(
                chunk.paper_id == paper.paper_id and chunk.page_number == 0
                for chunk in chunks
            )
        }
        fulltext_evidence_papers = {
            str(chunk.paper_id) for chunk in chunks if chunk.page_number > 0
        }
        profile_stats = self._profile_stats(fulltext_evidence_papers)
        profiled_ids = profile_stats["ids"]
        # Evidence-layer reporting follows both persisted page evidence and the
        # ingestion state.  A record can be marked abstract_only even when its
        # source had no abstract text to persist, so do not under-count it by
        # relying on page-0 chunks alone.
        abstract_only_ids = {
            str(paper.paper_id)
            for paper in papers
            if str(paper.status) == "abstract_only"
            and str(paper.paper_id) not in fulltext_evidence_papers
        }
        fulltext_profiled = profile_stats["fulltext"]

        status_counts = dict(sorted(Counter(str(paper.status) for paper in papers).items()))
        provider_counts = dict(sorted(Counter(str(paper.source) for paper in papers).items()))
        failure_counts = dict(
            sorted(
                Counter(str(paper.last_error) for paper in papers if paper.last_error).items()
            )
        )
        vector_report = self._vector_report(chunk_ids)
        vector_ids = set(vector_report["document_ids"])
        vector_paper_ids = set(vector_report["paper_ids"])
        paper_ids = {str(paper.paper_id) for paper in papers}
        parsed_statuses = {"parsed", "profiled", "indexed"}
        parsed_paper_ids = {
            str(paper.paper_id) for paper in papers if str(paper.status) in parsed_statuses
        }
        indexed_status_ids = {
            str(paper.paper_id) for paper in papers if str(paper.status) == "indexed"
        }

        return {
            "metadata_total": metadata_total,
            "papers_with_abstract": sum(bool((paper.abstract or "").strip()) for paper in papers),
            "papers_with_chunks": len(paper_ids_with_chunks),
            "papers_without_chunks": len({str(paper.paper_id) for paper in papers} - paper_ids_with_chunks),
            "chunks_total": len(chunks),
            "profiled_papers": len(profiled_ids),
            "abstract_profiled_papers": profile_stats["abstract"],
            "fulltext_profiled_papers": profile_stats["fulltext"],
            "vector_indexed": len(vector_ids),
            "vector_indexed_papers": len(vector_paper_ids),
            "abstract_only_papers": sum(paper.status == "abstract_only" for paper in papers),
            "evidence_layers": {
                "metadata_catalog": metadata_total,
                "abstract_evidence": len(abstract_evidence_papers),
                "page_addressable_fulltext": len(fulltext_evidence_papers),
                "abstract_only": len(abstract_only_ids),
            },
            "profile_coverage": {
                "profiled_papers": len(profiled_ids),
                "metadata_total": metadata_total,
                "coverage_ratio": _ratio(len(profiled_ids), metadata_total),
                "fulltext_profiled_papers": fulltext_profiled,
                "fulltext_evidence_papers": len(fulltext_evidence_papers),
                "fulltext_coverage_ratio": _ratio(
                    fulltext_profiled, len(fulltext_evidence_papers)
                ),
            },
            "status_counts": status_counts,
            "ingestion_layers": {
                "parsed_papers": len(parsed_paper_ids),
                "indexed_papers": len(indexed_status_ids),
                "vector_indexed_papers": len(vector_paper_ids),
                "parsed_not_indexed": len(parsed_paper_ids - indexed_status_ids),
                "vector_papers_without_catalog": len(vector_paper_ids - paper_ids),
            },
            "provider_counts": provider_counts,
            "failure_counts": failure_counts,
            "vector_index": {
                "available": vector_report["available"],
                "document_count": len(vector_ids),
                "paper_count": len(vector_paper_ids),
                "missing_chunk_ids": sorted(chunk_ids - vector_ids),
                "orphan_vector_ids": sorted(vector_ids - chunk_ids),
            },
        }

    def _profiled_ids(self) -> set[str]:
        with self.database._connect() as connection:
            rows = connection.execute("SELECT paper_id FROM paper_profiles").fetchall()
        return {str(row["paper_id"]) for row in rows}

    def _profile_stats(self, fulltext_evidence_papers: set[str]) -> dict[str, object]:
        with self.database._connect() as connection:
            rows = connection.execute(
                "SELECT paper_id, profile_json FROM paper_profiles"
            ).fetchall()
        abstract = fulltext = 0
        ids: set[str] = set()
        for row in rows:
            ids.add(str(row["paper_id"]))
            try:
                payload = json.loads(row["profile_json"])
            except (TypeError, ValueError):
                abstract += 1
                continue
            pages = []
            for value in payload.values() if isinstance(payload, dict) else []:
                fields = value if isinstance(value, list) else [value]
                for field in fields:
                    for evidence in (field.get("evidence", []) if isinstance(field, dict) else []):
                        if isinstance(evidence, dict) and isinstance(evidence.get("page_number"), int):
                            pages.append(evidence["page_number"])
            if any(page > 0 for page in pages) or str(row["paper_id"]) in fulltext_evidence_papers:
                fulltext += 1
            else:
                abstract += 1
        return {"ids": ids, "abstract": abstract, "fulltext": fulltext}

    def _vector_report(self, chunk_ids: set[str]) -> dict[str, Any]:
        del chunk_ids  # Kept in the signature to make the comparison boundary explicit.
        store = getattr(self.vector_store, "_store", self.vector_store)
        mapping = getattr(store, "index_to_docstore_id", None)
        docstore = getattr(store, "docstore", None)
        if not isinstance(mapping, dict) or docstore is None:
            return {"available": False, "document_ids": [], "paper_ids": []}

        document_ids: list[str] = []
        paper_ids: list[str] = []
        for document_id in mapping.values():
            document_id = str(document_id)
            document = docstore.search(document_id)
            if document is None:
                continue
            document_ids.append(document_id)
            metadata = getattr(document, "metadata", {})
            if isinstance(metadata, dict) and metadata.get("paper_id") is not None:
                paper_ids.append(str(metadata["paper_id"]))
        return {
            "available": True,
            "document_ids": document_ids,
            "paper_ids": paper_ids,
        }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
