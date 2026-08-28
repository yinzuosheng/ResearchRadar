from langchain_core.documents import Document
from io import StringIO

import app
from domain.models import EvidenceChunk, PaperCandidate, PaperProfile, ExtractedField
from storage.database import ResearchDatabase
from workflows.knowledge_audit import KnowledgeAuditService


def _candidate(paper_id: str, *, abstract: str | None = "abstract"):
    return PaperCandidate(
        source="test",
        source_id=paper_id,
        title=f"Paper {paper_id}",
        doi=f"10.1234/{paper_id}",
        year=2024,
        venue="Journal",
        abstract=abstract,
    )


class _Docstore:
    def __init__(self, documents):
        self.documents = documents

    def search(self, document_id):
        return self.documents.get(document_id)


class _Store:
    def __init__(self, documents):
        self.docstore = _Docstore(documents)
        self.index_to_docstore_id = {
            index: document_id for index, document_id in enumerate(documents)
        }


def _document(chunk_id: str, paper_id: str):
    return Document(
        page_content="evidence",
        metadata={
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "title": "Paper",
            "page_number": 1,
            "section": "Results",
        },
    )


def test_audit_separates_profiles_evidence_and_vectors(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    paper_with_evidence = database.upsert_candidate(_candidate("p1"))
    paper_abstract_only = database.upsert_candidate(_candidate("p2"))
    database.update_status(paper_abstract_only.paper_id, "abstract_only")
    database.replace_chunks(
        paper_with_evidence.paper_id,
        [EvidenceChunk(chunk_id="c1", paper_id=paper_with_evidence.paper_id, title="Paper", page_number=1, text="evidence")],
    )
    database.save_profile(
        paper_with_evidence.paper_id,
        PaperProfile(
            prediction_target=ExtractedField(),
            study_area=ExtractedField(),
            time_span=ExtractedField(),
            sample_size=ExtractedField(),
        ),
    )
    report = KnowledgeAuditService(
        database,
        vector_store=_Store({"c1": _document("c1", paper_with_evidence.paper_id)}),
    ).run()

    assert report["metadata_total"] == 2
    assert report["papers_with_abstract"] == 2
    assert report["papers_with_chunks"] == 1
    assert report["chunks_total"] == 1
    assert report["profiled_papers"] == 1
    assert report["evidence_layers"] == {
        "metadata_catalog": 2,
        "abstract_evidence": 0,
        "page_addressable_fulltext": 1,
        "abstract_only": 1,
    }
    assert report["profile_coverage"] == {
        "profiled_papers": 1,
        "metadata_total": 2,
        "coverage_ratio": 0.5,
        "fulltext_profiled_papers": 1,
        "fulltext_evidence_papers": 1,
        "fulltext_coverage_ratio": 1.0,
    }
    assert report["vector_indexed"] == 1
    assert report["ingestion_layers"] == {
        "parsed_papers": 0,
        "indexed_papers": 0,
        "vector_indexed_papers": 1,
        "parsed_not_indexed": 0,
        "vector_papers_without_catalog": 0,
    }
    assert report["abstract_only_papers"] == 1
    assert report["vector_index"]["missing_chunk_ids"] == []
    assert report["vector_index"]["orphan_vector_ids"] == []


def test_audit_reports_vector_gaps_without_changing_database(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    paper = database.upsert_candidate(_candidate("p1"))
    database.replace_chunks(
        paper.paper_id,
        [
            EvidenceChunk(chunk_id="c1", paper_id=paper.paper_id, title="Paper", page_number=1, text="one"),
            EvidenceChunk(chunk_id="c2", paper_id=paper.paper_id, title="Paper", page_number=2, text="two"),
        ],
    )
    report = KnowledgeAuditService(
        database,
        vector_store=_Store({"c1": _document("c1", paper.paper_id), "orphan": _document("orphan", "missing")}),
    ).run()

    assert report["vector_index"]["missing_chunk_ids"] == ["c2"]
    assert report["vector_index"]["orphan_vector_ids"] == ["orphan"]
    assert database.get_paper(paper.paper_id).status == "discovered"


def test_knowledge_audit_cli_dispatches_to_service():
    calls = []

    class Services:
        def knowledge_audit(self):
            calls.append("audit")
            return {"metadata_total": 2}

    output = StringIO()
    assert app.run(["knowledge-audit"], services=Services(), stdout=output) == 0
    assert calls == ["audit"]
    assert '"metadata_total": 2' in output.getvalue()
