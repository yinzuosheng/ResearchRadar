from pathlib import Path

from domain.models import (
    ExtractedField,
    IngestionResult,
    PageText,
    PaperCandidate,
    PaperProfile,
)
from domain.statuses import ABSTRACT_ONLY, FAILED, INDEXED, PARSED
from ingestion.downloader import DownloadResult, PdfDownloader
from ingestion.pdf_parser import PdfParser
from ingestion.pipeline import ResearchIngestor
from providers.base import FullTextLocation
from storage.database import ResearchDatabase, source_fingerprint
from utils.text_splitter import split_pages


def candidate(**changes) -> PaperCandidate:
    values = {
        "source": "openalex",
        "source_id": "W123",
        "title": "Water Colour from Space",
        "doi": "10.1000/example",
        "abstract": "Satellite observations predict chlorophyll-a.",
        "pdf_url": "https://repository.example/paper.pdf",
        "license": "cc-by",
        "source_updated_at": "2026-08-01",
    }
    values.update(changes)
    return PaperCandidate(**values)


def empty_profile() -> PaperProfile:
    return PaperProfile(
        prediction_target=ExtractedField(),
        study_area=ExtractedField(),
        time_span=ExtractedField(),
        sample_size=ExtractedField(),
    )


class FakeRegistry:
    def __init__(self, events=None, locations=None):
        self.events = events
        self.locations = locations or []

    def enrich(self, paper):
        if self.events is not None:
            self.events.append("enrich")
        return paper

    def resolve_full_text(self, paper):
        return self.locations


class FakeDownloader:
    def __init__(self, path: Path | None, events=None, error_code="no_open_full_text"):
        self.path = path
        self.events = events
        self.error_code = error_code
        self.calls = 0

    def download(self, paper, target_path):
        self.calls += 1
        if self.events is not None:
            self.events.append("download")
        if self.path is None:
            return DownloadResult(success=False, error_code=self.error_code)
        return DownloadResult(
            success=True,
            path=self.path,
            location=FullTextLocation(
                provider="fake",
                url="https://example.test/paper.pdf",
                license="cc-by",
                is_oa=True,
                priority=1,
            ),
        )


class FakeParser:
    def __init__(self, pages=None, events=None, error=None):
        self.pages = pages or []
        self.events = events
        self.error = error

    def parse(self, path):
        if self.events is not None:
            self.events.append("parse")
        if self.error is not None:
            raise self.error
        return self.pages


class FakeProfileExtractor:
    def __init__(self, events):
        self.events = events

    def extract(self, pages):
        self.events.append("profile")
        assert [page.page_number for page in pages] == [1, 2]
        return empty_profile()


class FakeVectorIndex:
    def __init__(self, events):
        self.events = events
        self.chunks = []

    def index(self, chunks):
        self.events.append("index")
        self.chunks = list(chunks)

    def remove(self, paper_id):
        self.events.append(("remove", paper_id))
        self.chunks = [chunk for chunk in self.chunks if chunk.paper_id != paper_id]


class TrackingDatabase(ResearchDatabase):
    def __init__(self, path, events):
        self.events = events
        super().__init__(path)

    def replace_chunks(self, paper_id, chunks):
        super().replace_chunks(paper_id, chunks)
        self.events.append("replace_chunks")


def make_ingestor(tmp_path, *, downloader, parser, events=None):
    db = (
        TrackingDatabase(tmp_path / "research.db", events)
        if events is not None
        else ResearchDatabase(tmp_path / "research.db")
    )
    return db, ResearchIngestor(
        registry=FakeRegistry(events),
        database=db,
        download_dir=tmp_path / "pdfs",
        downloader=downloader,
        parser=parser,
        profile_extractor=FakeProfileExtractor(events) if events is not None else None,
        vector_index=FakeVectorIndex(events) if events is not None else None,
    )


def test_ingest_preserves_page_numbers_is_idempotent_and_orders_collaborators(tmp_path):
    events = []
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    downloader = FakeDownloader(pdf_path, events)
    parser = FakeParser(
        [PageText(page_number=1, text="Page one."), PageText(page_number=2, text="Page two.")],
        events,
    )
    db, ingestor = make_ingestor(
        tmp_path, downloader=downloader, parser=parser, events=events
    )

    result1 = ingestor.ingest(candidate())
    result2 = ingestor.ingest(candidate())

    assert result1.status == INDEXED
    assert result2 == IngestionResult(
        paper_id=result1.paper_id,
        status=INDEXED,
        skipped=True,
        chunks_indexed=2,
    )
    assert db.count_papers() == 1
    assert {chunk.page_number for chunk in db.get_chunks(result1.paper_id)} == {1, 2}
    assert events == [
        "enrich",
        "download",
        "parse",
        "replace_chunks",
        "profile",
        "index",
    ]
    assert downloader.calls == 1


def test_changed_source_fingerprint_reprocesses_indexed_paper(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    downloader = FakeDownloader(pdf_path)
    db, ingestor = make_ingestor(
        tmp_path,
        downloader=downloader,
        parser=FakeParser([PageText(page_number=1, text="Body.")]),
    )

    first = ingestor.ingest(candidate())
    changed = ingestor.ingest(candidate(source_updated_at="2026-08-02"))

    assert first.paper_id == changed.paper_id
    assert changed.status == PARSED
    assert changed.skipped is False
    assert downloader.calls == 2


def test_completed_download_is_reused_after_parse_interruption(tmp_path):
    location = FullTextLocation(
        provider="fake",
        url="https://example.test/paper.pdf",
        license="cc-by",
        is_oa=True,
        priority=1,
    )

    class Response:
        headers = {"Content-Type": "application/pdf"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"%PDF-completed"

        def close(self):
            return None

    calls = []

    def http_get(*args, **kwargs):
        calls.append(args[0])
        return Response()

    database = ResearchDatabase(tmp_path / "research.db")
    parser = FakeParser(error=RuntimeError("interrupted after download"))
    ingestor = ResearchIngestor(
        registry=FakeRegistry(locations=[location]),
        database=database,
        download_dir=tmp_path / "pdfs",
        downloader=PdfDownloader(FakeRegistry(locations=[location]), http_get=http_get),
        parser=parser,
    )
    first = ingestor.ingest(candidate())
    assert first.status == FAILED
    parser.error = None
    parser.pages = [PageText(page_number=1, text="Recovered body")]

    second = ingestor.ingest(candidate())

    assert second.status == PARSED
    assert calls == [location.url]


def test_default_offline_adapters_stop_at_parsed_after_full_text(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    database = ResearchDatabase(tmp_path / "research.db")
    ingestor = ResearchIngestor(
        registry=FakeRegistry(),
        database=database,
        download_dir=tmp_path / "pdfs",
        downloader=FakeDownloader(pdf_path),
        parser=FakeParser([PageText(page_number=1, text="Body.")]),
    )

    result = ingestor.ingest(candidate())

    assert result.status == PARSED
    stored = database.get_paper(result.paper_id)
    assert stored.status == PARSED
    assert database.get_chunks(result.paper_id)


def test_enriched_metadata_is_persisted_without_changing_source_fingerprint(tmp_path):
    class EnrichingRegistry(FakeRegistry):
        def enrich(self, paper):
            return paper.model_copy(
                update={"authors": ["Crossref Author"], "venue": "Crossref Journal"}
            )

    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    db = ResearchDatabase(tmp_path / "research.db")
    ingestor = ResearchIngestor(
        registry=EnrichingRegistry(),
        database=db,
        download_dir=tmp_path / "pdfs",
        downloader=FakeDownloader(pdf_path),
        parser=FakeParser([PageText(page_number=1, text="Body.")]),
    )
    incoming = candidate()

    result = ingestor.ingest(incoming)

    stored = db.get_paper(result.paper_id)
    assert stored.authors == ["Crossref Author"]
    assert stored.venue == "Crossref Journal"
    assert stored.source_fingerprint == source_fingerprint(incoming)


def test_ingestor_can_disable_full_text_for_legacy_include_pdf_flag(tmp_path):
    downloader = FakeDownloader(tmp_path / "must-not-be-used.pdf")
    db = ResearchDatabase(tmp_path / "research.db")
    ingestor = ResearchIngestor(
        registry=FakeRegistry(),
        database=db,
        download_dir=tmp_path / "pdfs",
        downloader=downloader,
        parser=FakeParser(),
        allow_full_text=False,
    )

    result = ingestor.ingest(candidate())

    assert result.status == ABSTRACT_ONLY
    assert downloader.calls == 0
    assert db.get_paper(result.paper_id).last_error == "no_open_full_text"


def test_pdf_failure_keeps_abstract_only_record_and_labels_evidence(tmp_path):
    db, ingestor = make_ingestor(
        tmp_path,
        downloader=FakeDownloader(None),
        parser=FakeParser(),
    )

    result = ingestor.ingest(candidate())

    assert result.status == ABSTRACT_ONLY
    assert db.get_paper(result.paper_id).last_error == "no_open_full_text"
    assert db.get_chunks(result.paper_id)[0].page_number == 0
    assert db.get_chunks(result.paper_id)[0].section == "摘要证据"


def test_abstract_fallback_removes_previous_vector_documents(tmp_path):
    events = []
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    downloader = FakeDownloader(pdf_path, events)
    db, ingestor = make_ingestor(
        tmp_path,
        downloader=downloader,
        parser=FakeParser(
            [PageText(page_number=1, text="Body."), PageText(page_number=2, text="Methods.")],
            events,
        ),
        events=events,
    )

    first = ingestor.ingest(candidate())
    downloader.path = None
    second = ingestor.ingest(candidate(source_updated_at="2026-08-02"))

    assert first.status == INDEXED
    assert second.status == ABSTRACT_ONLY
    assert ("remove", first.paper_id) in events


def test_download_exception_is_stored_as_stable_abstract_only_error(tmp_path):
    class RaisingDownloader:
        def download(self, paper, target_path):
            raise RuntimeError("https://provider.test/file.pdf?token=credential")

    db, ingestor = make_ingestor(
        tmp_path,
        downloader=RaisingDownloader(),
        parser=FakeParser(),
    )

    result = ingestor.ingest(candidate())

    assert result.status == ABSTRACT_ONLY
    assert db.get_paper(result.paper_id).last_error == "pdf_download_failed"
    assert "credential" not in db.get_paper(result.paper_id).last_error


def test_parser_failure_marks_failed_with_stable_error_only(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    db, ingestor = make_ingestor(
        tmp_path,
        downloader=FakeDownloader(pdf_path),
        parser=FakeParser(error=RuntimeError("secret URL https://x.test/?key=credential")),
    )

    result = ingestor.ingest(candidate())

    assert result.status == FAILED
    assert db.get_paper(result.paper_id).last_error == "pdf_parse_failed"


def test_split_pages_keeps_page_numbers_and_stable_ids():
    chunks = split_pages(
        "paper-1",
        "A title",
        [PageText(page_number=3, text="First page chunk."), PageText(page_number=7, text="Later chunk.")],
    )

    assert [(chunk.chunk_id, chunk.page_number) for chunk in chunks] == [
        ("paper-1:p3:c0", 3),
        ("paper-1:p7:c0", 7),
    ]


class FakeResponse:
    def __init__(self, content_type, chunks, content_length=None):
        self.headers = {"Content-Type": content_type}
        self.chunks = chunks
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield from self.chunks


def response(content_type, chunks, content_length=None):
    return FakeResponse(content_type, chunks, content_length)


def test_downloader_rejects_html_from_pdf_url_and_removes_part(tmp_path):
    registry = FakeRegistry(
        locations=[
            FullTextLocation(
                provider="fake",
                url="https://example.test/paper.pdf",
                is_oa=True,
                priority=1,
            )
        ]
    )
    downloader = PdfDownloader(
        registry,
        http_get=lambda *args, **kwargs: response("text/html", [b"<html>not pdf</html>"]),
    )
    target = tmp_path / "paper.pdf"

    result = downloader.download(candidate(), target)

    assert result.success is False
    assert result.error_code == "invalid_pdf"
    assert not target.exists()
    assert not target.with_suffix(".part").exists()


def test_downloader_uses_fallback_resolver_after_preferred_location_fails(tmp_path):
    preferred = FullTextLocation(
        provider="openalex",
        url="https://example.test/stale.pdf",
        is_oa=True,
        priority=1,
    )
    fallback = FullTextLocation(
        provider="unpaywall",
        url="https://example.test/current.pdf",
        is_oa=True,
        priority=2,
    )

    class FallbackRegistry(FakeRegistry):
        def resolve_fallback_full_text(self, paper):
            return [fallback]

    calls = []

    def http_get(url, **kwargs):
        calls.append(url)
        if url == preferred.url:
            return response("text/html", [b"<html>stale</html>"])
        return response("application/pdf", [b"%PDF-current"])

    result = PdfDownloader(
        FallbackRegistry(locations=[preferred]), http_get=http_get
    ).download(candidate(), tmp_path / "paper")

    assert result.success is True
    assert result.location.provider == "unpaywall"
    assert calls == [preferred.url, fallback.url]


def test_downloader_sanitizes_full_text_resolution_failure(tmp_path):
    class RaisingRegistry:
        def resolve_full_text(self, paper):
            raise RuntimeError("missing secret credential from https://provider.test")

    result = PdfDownloader(RaisingRegistry()).download(candidate(), tmp_path / "paper")

    assert result == DownloadResult(
        success=False,
        error_code="full_text_resolution_failed",
    )


def test_downloader_rejects_pdf_content_type_without_pdf_magic(tmp_path):
    registry = FakeRegistry(
        locations=[
            FullTextLocation(
                provider="fake",
                url="https://example.test/download",
                is_oa=True,
                priority=1,
            )
        ]
    )
    downloader = PdfDownloader(
        registry,
        http_get=lambda *args, **kwargs: response("application/pdf", [b"not-a-pdf"]),
    )

    result = downloader.download(candidate(), tmp_path / "paper")

    assert result.success is False
    assert result.error_code == "invalid_pdf"


def test_downloader_streams_atomically_and_enforces_maximum_size(tmp_path):
    location = FullTextLocation(
        provider="fake",
        url="https://example.test/download",
        is_oa=True,
        priority=1,
    )
    registry = FakeRegistry(locations=[location])
    valid = PdfDownloader(
        registry,
        http_get=lambda *args, **kwargs: response("application/pdf", [b"%PDF-", b"body"]),
        max_bytes=10,
    ).download(candidate(), tmp_path / "valid")
    oversized = PdfDownloader(
        registry,
        http_get=lambda *args, **kwargs: response("application/pdf", [b"%PDF-", b"123456"]),
        max_bytes=10,
    ).download(candidate(), tmp_path / "large")

    assert valid.success is True
    assert valid.path == tmp_path / "valid.pdf"
    assert valid.path.read_bytes() == b"%PDF-body"
    assert oversized.success is False
    assert oversized.error_code == "pdf_too_large"
    assert not (tmp_path / "large.part").exists()


def test_downloader_accepts_case_insensitive_http_header_names(tmp_path):
    location = FullTextLocation(
        provider="fake",
        url="https://example.test/download",
        is_oa=True,
        priority=1,
    )
    http_response = response("unused", [b"%PDF-body"])
    http_response.headers = {
        "content-type": "application/pdf",
        "content-length": "9",
    }
    result = PdfDownloader(
        FakeRegistry(locations=[location]),
        http_get=lambda *args, **kwargs: http_response,
    ).download(candidate(), tmp_path / "paper")

    assert result.success is True
    assert result.path.read_bytes() == b"%PDF-body"


def test_downloader_accepts_octet_stream_when_pdf_magic_is_valid(tmp_path):
    location = FullTextLocation(
        provider="fake",
        url="https://example.test/download",
        is_oa=True,
        priority=1,
    )
    http_response = response("application/octet-stream", [b"%PDF-body"])
    result = PdfDownloader(
        FakeRegistry(locations=[location]),
        http_get=lambda *args, **kwargs: http_response,
    ).download(candidate(), tmp_path / "paper")

    assert result.success is True


def test_pdf_parser_extracts_each_nonempty_page_once(monkeypatch, tmp_path):
    class Page:
        def __init__(self, text):
            self.text = text
            self.calls = 0

        def extract_text(self):
            self.calls += 1
            return self.text

    pages = [Page("  first  "), Page(None), Page("second")]
    monkeypatch.setattr("ingestion.pdf_parser.PdfReader", lambda path: type("Reader", (), {"pages": pages})())

    result = PdfParser().parse(tmp_path / "paper.pdf")

    assert result == [PageText(page_number=1, text="first"), PageText(page_number=3, text="second")]
    assert [page.calls for page in pages] == [1, 1, 1]


def test_legacy_collect_command_delegates_candidates_to_new_ingestor(monkeypatch):
    from utils import pipeline

    papers = [candidate(source_id="W1"), candidate(source_id="W2", doi="10.1000/two")]

    class Registry:
        def __init__(self):
            self.calls = []

        def discover(self, query, *, from_year, max_results):
            self.calls.append((query, from_year, max_results))
            return papers

    class Ingestor:
        def __init__(self):
            self.calls = []

        def ingest(self, paper):
            self.calls.append(paper)
            return IngestionResult(paper_id=paper.source_id, status=INDEXED, chunks_indexed=1)

    registry = Registry()
    ingestor = Ingestor()
    saved = []
    monkeypatch.setattr(pipeline, "save_paper_sources", lambda sources: saved.extend(sources))

    summary = pipeline.collect_papers_and_ingest(
        queries=["water colour"],
        max_results=2,
        registry=registry,
        ingestor=ingestor,
    )

    assert summary == "collected=2 ingested=2"
    assert registry.calls == [("water colour", 2023, 2)]
    assert ingestor.calls == papers
    assert [item["source_id"] for item in saved] == ["W1", "W2"]
