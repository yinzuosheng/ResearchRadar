from domain.models import EvidenceChunk, PaperCandidate
from storage.database import ResearchDatabase
from workflows.corpus_curator import CorpusCurator


def paper(source_id, title, abstract):
    return PaperCandidate(source="openalex", source_id=source_id, title=title, abstract=abstract)


def test_curator_prefers_relevant_evidence_and_reports_tiers(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    relevant = db.upsert_candidate(
        paper("W1", "Sentinel-2 chlorophyll-a prediction", "machine learning water quality")
    )
    weak = db.upsert_candidate(paper("W2", "Satellite image classification", "land cover"))
    db.replace_chunks(
        relevant.paper_id,
        [EvidenceChunk(chunk_id=f"{relevant.paper_id}:p1:c0", paper_id=relevant.paper_id, title=relevant.title, page_number=1, text="full")],
    )
    db.replace_chunks(
        weak.paper_id,
        [EvidenceChunk(chunk_id=f"{weak.paper_id}:abstract:c0", paper_id=weak.paper_id, title=weak.title, page_number=0, text="abstract")],
    )
    report = CorpusCurator(db).report(limit=10)
    assert report["selected_papers"] == 1
    assert report["selected_fulltext_papers"] == 1
    assert report["paper_ids"] == [relevant.paper_id]
