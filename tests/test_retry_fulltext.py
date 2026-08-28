from __future__ import annotations

import pytest

from domain.models import IngestionResult, PaperCandidate
from domain.statuses import ABSTRACT_ONLY, FAILED, INDEXED
from storage.database import ResearchDatabase
from workflows.retry_fulltext import RetryFullTextService


def candidate(source_id: str, *, doi: str | None) -> PaperCandidate:
    return PaperCandidate(
        source="openalex",
        source_id=source_id,
        title=f"Paper {source_id}",
        doi=doi,
        abstract="abstract",
    )


class FakeIngestor:
    def __init__(self, database, outcomes):
        self.database = database
        self.outcomes = outcomes
        self.calls: list[str] = []

    def ingest(self, item):
        self.calls.append(item.source_id)
        record = self.database.find_candidate(item)
        outcome = self.outcomes.get(item.source_id, INDEXED)
        if outcome == INDEXED:
            self.database.update_status(record.paper_id, INDEXED)
        elif outcome == ABSTRACT_ONLY:
            self.database.update_status(record.paper_id, ABSTRACT_ONLY, "no_open_full_text")
        else:
            self.database.update_status(record.paper_id, FAILED, "pdf_parse_failed")
        return IngestionResult(paper_id=record.paper_id, status=outcome)


def seed_abstract_only(database, item):
    record = database.upsert_candidate(item)
    database.update_status(record.paper_id, ABSTRACT_ONLY, "no_open_full_text")


def test_retries_only_doi_backed_abstracts_and_respects_limit(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    seed_abstract_only(database, candidate("W1", doi="10.1000/1"))
    seed_abstract_only(database, candidate("W2", doi=None))
    seed_abstract_only(database, candidate("W3", doi="10.1000/3"))
    ingestor = FakeIngestor(database, {})

    report = RetryFullTextService(database, ingestor).run(limit=1)

    assert report.attempted == 1
    assert report.indexed == 1
    assert ingestor.calls == ["W1"]


def test_retries_doi_backed_discovered_records_after_abstract_records(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    seed_abstract_only(database, candidate("W1", doi="10.1000/1"))
    discovered = database.upsert_candidate(candidate("W2", doi="10.1000/2"))
    assert discovered.status != ABSTRACT_ONLY

    ingestor = FakeIngestor(database, {})
    report = RetryFullTextService(database, ingestor).run(limit=2)

    assert report.attempted == 2
    assert ingestor.calls == ["W1", "W2"]


def test_prioritizes_direct_pdf_candidates_across_statuses(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    seed_abstract_only(database, candidate("W1", doi="10.1000/1"))
    direct = candidate("W2", doi="10.1000/2").model_copy(
        update={"pdf_url": "https://repo.test/w2.pdf"}
    )
    database.upsert_candidate(direct)
    ingestor = FakeIngestor(database, {})

    report = RetryFullTextService(database, ingestor).run(limit=1)

    assert report.attempted == 1
    assert ingestor.calls == ["W2"]


def test_continues_after_failure_and_reports_stable_codes(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    for source_id in ("W1", "W2"):
        seed_abstract_only(database, candidate(source_id, doi=f"10.1000/{source_id}"))
    ingestor = FakeIngestor(database, {"W1": FAILED, "W2": ABSTRACT_ONLY})

    report = RetryFullTextService(database, ingestor).run(limit=2)

    assert report.attempted == 2
    assert report.failed == 1
    assert report.abstract_only == 1
    assert report.failures == {"no_open_full_text": 1, "pdf_parse_failed": 1}


@pytest.mark.parametrize("limit", [0, -1])
def test_limit_must_be_positive(tmp_path, limit):
    with pytest.raises(ValueError, match="^retry_fulltext_invalid_limit$"):
        RetryFullTextService(ResearchDatabase(tmp_path / "research.db"), object()).run(limit=limit)


def test_provider_scope_is_restricted_to_supported_values(tmp_path):
    with pytest.raises(ValueError, match="^retry_fulltext_invalid_provider$"):
        RetryFullTextService(ResearchDatabase(tmp_path / "research.db"), object()).run(
            limit=1, provider="unknown"
        )
