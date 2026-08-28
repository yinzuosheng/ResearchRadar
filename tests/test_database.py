from datetime import UTC, datetime, timedelta

import pytest

from domain.models import (
    EvidenceChunk,
    EvidenceRef,
    ExtractedField,
    PaperCandidate,
    PaperProfile,
)
from domain.statuses import DISCOVERED, INDEXED
from storage.database import PaperIdentityConflictError, ResearchDatabase


def test_fts_projection_needs_backfill_only_when_chunk_counts_differ(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    paper = db.upsert_candidate(candidate())
    db.replace_chunks(
        paper.paper_id,
        [EvidenceChunk(chunk_id="chunk", paper_id=paper.paper_id, title=paper.title, page_number=1, text="text")],
    )
    assert db._fts_projection_needs_backfill() is False
    with db._connect() as connection:
        connection.execute("DELETE FROM chunk_fts")
    assert db._fts_projection_needs_backfill() is True


def candidate(doi="10.1234/test", title="Chlorophyll Prediction", year=2024):
    return PaperCandidate(
        source="openalex",
        source_id="W1",
        title=title,
        doi=doi,
        authors=["A. Researcher"],
        year=year,
        venue="Remote Sensing",
        abstract="Predicting chlorophyll-a.",
        landing_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        license="cc-by",
        cited_by_count=8,
    )


def profile() -> PaperProfile:
    evidence = [EvidenceRef(page_number=3, quote="Chlorophyll-a was predicted.")]
    return PaperProfile(
        prediction_target=ExtractedField(value="chlorophyll-a", evidence=evidence),
        study_area=ExtractedField(value="Lake Taihu"),
        time_span=ExtractedField(value="2019-2022"),
        sample_size=ExtractedField(value="120 samples"),
        models=[ExtractedField(value="random forest", evidence=evidence)],
    )


def test_upsert_deduplicates_normalized_doi(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    first = db.upsert_candidate(candidate(doi="https://doi.org/10.1234/TEST"))
    second = db.upsert_candidate(candidate(doi="10.1234/test"))
    assert first.paper_id == second.paper_id
    assert db.count_papers() == 1
    assert second.normalized_doi == "10.1234/test"


def test_upsert_without_doi_deduplicates_title_and_year(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    first = db.upsert_candidate(
        candidate(doi=None, title="  Water-Color   Prediction ")
    )
    second = db.upsert_candidate(candidate(doi=None, title="water color prediction"))
    assert first.paper_id == second.paper_id
    assert db.count_papers() == 1


def test_upsert_with_doi_upgrades_matching_no_doi_record_without_resetting_state(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    original = db.upsert_candidate(
        candidate(doi=None, title="  Water-Color   Prediction ")
    )
    db.update_status(original.paper_id, INDEXED, error="awaiting reindex")
    before_upgrade = db.get_paper(original.paper_id)

    upgraded = db.upsert_candidate(
        candidate(doi="10.5678/UPGRADED", title="water color prediction")
    )

    assert before_upgrade is not None
    assert upgraded.paper_id == original.paper_id
    assert upgraded.normalized_doi == "10.5678/upgraded"
    assert upgraded.status == INDEXED
    assert upgraded.last_error == "awaiting reindex"
    assert upgraded.first_seen_at == before_upgrade.first_seen_at
    assert db.count_papers() == 1


def test_upsert_rejects_conflicting_doi_and_title_year_matches_without_changes(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    title_match = db.upsert_candidate(candidate(doi=None, title="Water Color Prediction"))
    doi_match = db.upsert_candidate(
        candidate(doi="10.5678/conflict", title="A Different Paper")
    )

    with pytest.raises(
        PaperIdentityConflictError,
        match="DOI and title/year match different papers",
    ):
        db.upsert_candidate(
            candidate(doi="10.5678/conflict", title="water-color prediction")
        )

    assert db.count_papers() == 2
    assert db.get_paper(title_match.paper_id) == title_match
    assert db.get_paper(doi_match.paper_id) == doi_match


def test_status_and_discovery_time_are_persisted_as_utc_datetimes(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    paper = db.upsert_candidate(candidate())

    db.update_status(paper.paper_id, INDEXED, error="index retry")
    stored = db.get_paper(paper.paper_id)

    assert stored is not None
    assert stored.status == INDEXED
    assert stored.last_error == "index retry"
    assert stored.first_seen_at.tzinfo is not None
    assert db.list_papers(status=INDEXED) == [stored]
    assert db.list_papers_discovered_after(datetime.now(UTC) - timedelta(minutes=1)) == [stored]


def test_replacing_chunks_removes_old_chunks_and_preserves_order(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    paper = db.upsert_candidate(candidate())
    db.replace_chunks(
        paper.paper_id,
        [
            EvidenceChunk(
                chunk_id="old",
                paper_id=paper.paper_id,
                title=paper.title,
                page_number=1,
                text="old text",
            )
        ],
    )
    db.replace_chunks(
        paper.paper_id,
        [
            EvidenceChunk(
                chunk_id="second",
                paper_id=paper.paper_id,
                title=paper.title,
                page_number=2,
                section="Methods",
                text="second text",
                score=0.7,
            ),
            EvidenceChunk(
                chunk_id="first",
                paper_id=paper.paper_id,
                title=paper.title,
                page_number=1,
                text="first text",
            ),
        ],
    )

    assert db.get_chunks(paper.paper_id) == [
        EvidenceChunk(
            chunk_id="second",
            paper_id=paper.paper_id,
            title=paper.title,
            page_number=2,
            section="Methods",
            text="second text",
            score=0.7,
        ),
        EvidenceChunk(
            chunk_id="first",
            paper_id=paper.paper_id,
            title=paper.title,
            page_number=1,
            text="first text",
        ),
    ]


def test_get_chunks_with_context_expands_adjacent_parent_chunks(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    paper = db.upsert_candidate(candidate())
    db.replace_chunks(
        paper.paper_id,
        [
            EvidenceChunk(chunk_id=f"c{i}", paper_id=paper.paper_id, title=paper.title, page_number=i + 1, text=f"text {i}")
            for i in range(4)
        ],
    )

    expanded = db.get_chunks_with_context(["c1"], window=1)

    assert [chunk.chunk_id for chunk in expanded] == ["c0", "c1", "c2"]


def test_profile_round_trips_through_json(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    paper = db.upsert_candidate(candidate())

    db.save_profile(paper.paper_id, profile())

    assert db.get_profile(paper.paper_id) == profile()


def test_sync_records_return_most_recent_completed_utc_time(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    sync_id = db.start_sync("daily")
    assert db.last_successful_sync("daily") is None

    db.finish_sync(sync_id, discovered=3, downloaded=2, indexed=1)

    completed_at = db.last_successful_sync("daily")
    assert completed_at is not None
    assert completed_at.tzinfo is not None
    assert completed_at <= datetime.now(UTC)


def test_recent_sync_runs_returns_bounded_display_records(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    sync_id = db.start_sync("daily")
    db.finish_sync(sync_id, discovered=3, downloaded=2, indexed=1)

    rows = db.recent_sync_runs(limit=1)

    assert len(rows) == 1
    assert rows[0]["kind"] == "daily"
    assert rows[0]["status"] == "success"
    assert rows[0]["indexed"] == 1
