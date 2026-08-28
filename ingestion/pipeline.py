"""Idempotent ingestion state machine for research papers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from domain.models import EvidenceChunk, IngestionResult, PageText, PaperCandidate, PaperProfile
from domain.statuses import (
    ABSTRACT_ONLY,
    FAILED,
    INDEXED,
    METADATA_READY,
    PARSED,
    PDF_READY,
    PROFILED,
)
from ingestion.downloader import PdfDownloader
from ingestion.pdf_parser import PdfParser
from providers.registry import ProviderRegistry
from storage.database import ResearchDatabase, source_fingerprint
from utils.text_splitter import split_pages


class ProfileExtractor(Protocol):
    def extract(self, pages: list[PageText]) -> PaperProfile | None: ...


class VectorIndex(Protocol):
    def index(self, chunks: list[EvidenceChunk]) -> None: ...


class DeferredProfileExtractor:
    """Task-4 placeholder that preserves the Task-5 extraction seam."""

    def extract(self, pages: list[PageText]) -> None:
        return None


class NullVectorIndex:
    def index(self, chunks: list[EvidenceChunk]) -> None:
        return None


class VectorStoreIndex:
    """Narrow adapter around the legacy vector store, replaceable in Task 6."""

    def __init__(self, store) -> None:
        self.store = store

    def index(self, chunks: list[EvidenceChunk]) -> None:
        documents = [
            Document(
                page_content=self._embedding_text(chunk),
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "title": chunk.title,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "evidence_label": "摘要证据" if chunk.page_number == 0 else "全文证据",
                    "canonical_text": chunk.text,
                },
            )
            for chunk in chunks
        ]
        self.store.add_documents(documents)

    @staticmethod
    def _embedding_text(chunk: EvidenceChunk) -> str:
        location = chunk.section or ("摘要" if chunk.page_number == 0 else f"第 {chunk.page_number} 页")
        return f"标题：{chunk.title}\n位置：{location}\n正文：{chunk.text}"

    def remove(self, paper_id: str) -> None:
        self.store.remove_paper(paper_id)


class ResearchIngestor:
    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        database: ResearchDatabase,
        download_dir: Path,
        downloader: PdfDownloader | None = None,
        parser: PdfParser | None = None,
        profile_extractor: ProfileExtractor | None = None,
        vector_index: VectorIndex | None = None,
        allow_full_text: bool = True,
    ) -> None:
        self.registry = registry
        self.database = database
        self.download_dir = Path(download_dir)
        self.downloader = downloader or PdfDownloader(registry)
        self.parser = parser or PdfParser()
        self.profile_extractor = profile_extractor or DeferredProfileExtractor()
        self.vector_index = vector_index or NullVectorIndex()
        self.allow_full_text = allow_full_text
        self.offline_only = isinstance(self.profile_extractor, DeferredProfileExtractor) and isinstance(
            self.vector_index, NullVectorIndex
        )

    def ingest(self, candidate: PaperCandidate) -> IngestionResult:
        incoming_fingerprint = source_fingerprint(candidate)
        existing = self.database.find_candidate(candidate)
        paper = self.database.upsert_candidate(candidate)
        if (
            existing is not None
            and existing.status == INDEXED
            and existing.source_fingerprint == incoming_fingerprint
        ):
            return IngestionResult(
                paper_id=paper.paper_id,
                status=INDEXED,
                skipped=True,
                chunks_indexed=len(self.database.get_chunks(paper.paper_id)),
            )

        try:
            enriched = self.registry.enrich(candidate)
            paper = self.database.update_enriched_candidate(paper.paper_id, enriched)
            self.database.update_status(paper.paper_id, METADATA_READY)
        except Exception:
            return self._fail(paper.paper_id, "metadata_enrichment_failed")

        if not self.allow_full_text:
            return self._store_abstract(
                paper.paper_id,
                enriched,
                "no_open_full_text",
            )

        target_path = self.download_dir / paper.paper_id
        download = None
        if existing is not None and existing.source_fingerprint == incoming_fingerprint:
            reuse = getattr(self.downloader, "reuse_existing", None)
            if reuse is not None:
                download = reuse(target_path)
        try:
            download = download or self.downloader.download(
                enriched,
                target_path,
            )
        except Exception:
            return self._store_abstract(
                paper.paper_id,
                enriched,
                "pdf_download_failed",
            )
        if not download.success or download.path is None:
            return self._store_abstract(
                paper.paper_id,
                enriched,
                download.error_code or "no_open_full_text",
            )
        self.database.update_status(paper.paper_id, PDF_READY)

        try:
            pages = self.parser.parse(download.path)
            if not pages:
                raise ValueError("PDF contains no extractable text")
        except Exception:
            return self._fail(paper.paper_id, "pdf_parse_failed")
        self.database.update_status(paper.paper_id, PARSED)

        chunks = split_pages(paper.paper_id, enriched.title, pages)
        self.database.replace_chunks(paper.paper_id, chunks)

        profile_saved = False
        if not isinstance(self.profile_extractor, DeferredProfileExtractor):
            try:
                profile = self.profile_extractor.extract(pages)
                if profile is not None:
                    self.database.save_profile(paper.paper_id, profile)
                    profile_saved = True
            except Exception:
                return self._fail(paper.paper_id, "profile_extraction_failed")
        if profile_saved:
            self.database.update_status(paper.paper_id, PROFILED)

        if isinstance(self.vector_index, NullVectorIndex):
            return IngestionResult(
                paper_id=paper.paper_id,
                status=PROFILED if profile_saved else PARSED,
                chunks_indexed=len(chunks),
            )

        try:
            self.vector_index.index(chunks)
        except Exception:
            return self._fail(paper.paper_id, "vector_index_failed")

        self.database.update_status(paper.paper_id, INDEXED)
        return IngestionResult(
            paper_id=paper.paper_id,
            status=INDEXED,
            chunks_indexed=len(chunks),
        )

    def _store_abstract(
        self,
        paper_id: str,
        candidate: PaperCandidate,
        error_code: str,
    ) -> IngestionResult:
        remove_vectors = getattr(self.vector_index, "remove", None)
        if remove_vectors is not None:
            remove_vectors(paper_id)
        chunks: list[EvidenceChunk] = []
        if candidate.abstract and candidate.abstract.strip():
            chunks = [
                EvidenceChunk(
                    chunk_id=f"{paper_id}:abstract:c0",
                    paper_id=paper_id,
                    title=candidate.title,
                    page_number=0,
                    section="摘要证据",
                    text=candidate.abstract.strip(),
                )
            ]
        self.database.replace_chunks(paper_id, chunks)
        if chunks:
            try:
                self.vector_index.index(chunks)
            except Exception:
                return self._fail(paper_id, "vector_index_failed")
        self.database.update_status(paper_id, ABSTRACT_ONLY, error_code)
        return IngestionResult(
            paper_id=paper_id,
            status=ABSTRACT_ONLY,
            chunks_indexed=len(chunks),
        )

    def _fail(self, paper_id: str, error_code: str) -> IngestionResult:
        self.database.update_status(paper_id, FAILED, error_code)
        return IngestionResult(paper_id=paper_id, status=FAILED)
