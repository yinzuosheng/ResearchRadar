"""Resumable paper-profile extraction for already parsed local PDFs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from domain.statuses import FAILED, PARSED, PROFILED
from ingestion.pdf_parser import PdfParser
from storage.database import ResearchDatabase


class ProfileService:
    """Extract and persist one structured profile at a time."""

    def __init__(
        self,
        *,
        database: ResearchDatabase,
        pdf_dir: Path,
        parser: PdfParser,
        extractor,
    ) -> None:
        self.database = database
        self.pdf_dir = Path(pdf_dir)
        self.parser = parser
        self.extractor = extractor

    def run(self, *, retry_failed: bool = False) -> dict[str, object]:
        records = self.database.list_papers(limit=10000)
        candidates = [record for record in records if self._eligible(record.status, record.last_error, retry_failed)]
        failed: Counter[str] = Counter()
        processed = 0
        profiled = 0
        for record in candidates:
            pdf_path = self.pdf_dir / f"{record.paper_id}.pdf"
            if not pdf_path.exists():
                failed["pdf_not_found"] += 1
                continue
            processed += 1
            try:
                pages = self.parser.parse(pdf_path)
                if not pages:
                    raise ValueError("empty_pdf")
                profile = self.extractor.extract(pages)
                self.database.save_profile(record.paper_id, profile)
                self.database.update_status(record.paper_id, PROFILED)
                profiled += 1
            except Exception:
                self.database.update_status(record.paper_id, FAILED, "profile_extraction_failed")
                failed["profile_extraction_failed"] += 1
        return {
            "processed": processed,
            "profiled": profiled,
            "failed": dict(sorted(failed.items())),
        }

    @staticmethod
    def _eligible(status: str, last_error: str | None, retry_failed: bool) -> bool:
        return status == PARSED or (
            retry_failed and status == FAILED and last_error == "profile_extraction_failed"
        )
