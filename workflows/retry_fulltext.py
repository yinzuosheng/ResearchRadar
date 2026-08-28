"""Resumable lawful full-text retries for DOI-backed abstract records."""

from __future__ import annotations

from collections import Counter

from domain.models import IngestionResult, PaperCandidate, RetryFullTextReport
from domain.statuses import ABSTRACT_ONLY, DISCOVERED, FAILED, INDEXED


STABLE_FAILURE_CODES = {
    "full_text_resolution_failed",
    "ingestion_failed",
    "invalid_pdf",
    "metadata_enrichment_failed",
    "no_open_full_text",
    "pdf_download_failed",
    "pdf_parse_failed",
    "pdf_too_large",
    "profile_extraction_failed",
    "vector_index_failed",
}


class RetryFullTextService:
    """Retry only eligible records and continue when one item fails."""

    def __init__(self, database, ingestor) -> None:
        self.database = database
        self.ingestor = ingestor

    def run(
        self, *, limit: int, discovered_only: bool = False, provider: str | None = None
    ) -> RetryFullTextReport:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("retry_fulltext_invalid_limit")
        if provider not in {None, "semantic_scholar"}:
            raise ValueError("retry_fulltext_invalid_provider")
        registry = getattr(self.ingestor, "registry", None)
        if provider is not None and registry is not None:
            setter = getattr(registry, "set_full_text_provider", None)
            if setter is not None:
                setter(provider)

        records = self._eligible_records(limit, discovered_only=discovered_only)
        attempted = indexed = abstract_only = failed = 0
        failures: Counter[str] = Counter()
        for record in records:
            attempted += 1
            candidate = self._candidate(record)
            try:
                result = self.ingestor.ingest(candidate)
            except Exception:
                failed += 1
                failures["ingestion_failed"] += 1
                continue
            if result.status == INDEXED:
                indexed += 1
            elif result.status == ABSTRACT_ONLY:
                abstract_only += 1
                failures[self._failure_code(result.paper_id)] += 1
            elif result.status == FAILED:
                failed += 1
                failures[self._failure_code(result.paper_id)] += 1
            else:
                failed += 1
                failures["ingestion_failed"] += 1

        return RetryFullTextReport(
            attempted=attempted,
            indexed=indexed,
            abstract_only=abstract_only,
            failed=failed,
            failures=dict(sorted(failures.items())),
        )

    def _eligible_records(self, limit: int, *, discovered_only: bool = False):
        list_with_doi = getattr(self.database, "list_papers_with_doi", None)
        if list_with_doi is not None:
            records = []
            seen: set[str] = set()
            statuses = (DISCOVERED,) if discovered_only else (ABSTRACT_ONLY, DISCOVERED)
            for status in statuses:
                for record in list_with_doi(status=status, limit=limit):
                    if record.paper_id in seen:
                        continue
                    seen.add(record.paper_id)
                    records.append(record)
            records.sort(
                key=lambda record: (
                    not bool(record.pdf_url and record.pdf_url.strip()),
                    (
                        0
                        if (
                            record.status == DISCOVERED
                            and bool(record.pdf_url and record.pdf_url.strip())
                        )
                        else 1
                    ),
                    record.first_seen_at,
                    record.paper_id,
                )
            )
            return records[:limit]
        records = []
        seen: set[str] = set()
        statuses = (DISCOVERED,) if discovered_only else (ABSTRACT_ONLY, DISCOVERED)
        for status in statuses:
            for record in self.database.list_papers(status=status, limit=limit * 4):
                if not record.doi or not record.doi.strip() or record.paper_id in seen:
                    continue
                seen.add(record.paper_id)
                records.append(record)
        records.sort(
            key=lambda record: (
                not bool(record.pdf_url and record.pdf_url.strip()),
                (
                    0
                    if (
                        record.status == DISCOVERED
                        and bool(record.pdf_url and record.pdf_url.strip())
                    )
                    else 1
                ),
                record.first_seen_at,
                record.paper_id,
            )
        )
        return records[:limit]

    @staticmethod
    def _candidate(record) -> PaperCandidate:
        fields = PaperCandidate.model_fields
        return PaperCandidate.model_validate(
            {name: getattr(record, name) for name in fields}
        )

    def _failure_code(self, paper_id: str) -> str:
        record = self.database.get_paper(paper_id)
        code = record.last_error if record is not None else None
        return code if code in STABLE_FAILURE_CODES else "ingestion_failed"
