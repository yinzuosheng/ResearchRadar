from domain.models import EvidenceRef, ExtractedField, PaperCandidate, PaperProfile
from storage.database import ResearchDatabase
from web.presenters import (
    distribution_series,
    knowledge_metrics,
    paper_table_rows,
    render_agent_diagnostics,
)


def _candidate(paper_id, *, year=2024, venue="Remote Sensing"):
    return PaperCandidate(
        source="test",
        source_id=paper_id,
        title=f"Paper {paper_id}",
        doi=f"10.1234/{paper_id}",
        year=year,
        venue=venue,
        abstract="abstract",
    )


def _field(value=None):
    return ExtractedField(
        value=value,
        evidence=[] if value is None else [EvidenceRef(page_number=1, quote=value)],
    )


def _profile(target="chlorophyll-a", sensor="Sentinel-2", model="random forest"):
    return PaperProfile(
        prediction_target=_field(target),
        sensors=[_field(sensor)] if sensor else [],
        study_area=_field("Taihu"),
        time_span=_field(None),
        sample_size=_field(None),
        models=[_field(model)] if model else [],
    )


def test_database_statistics_split_status_and_profile_distributions(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    records = {}
    for paper_id, status in [("p1", "indexed"), ("p2", "abstract_only"), ("p3", "failed")]:
        paper = database.upsert_candidate(
            _candidate(paper_id, year=2024 if paper_id != "p3" else 2023)
        )
        records[paper_id] = paper
        database.update_status(
            paper.paper_id,
            status,
            "pdf_parse_failed" if status == "failed" else None,
        )
    database.save_profile(records["p1"].paper_id, _profile())

    snapshot = database.knowledge_statistics()

    assert snapshot["stats"] == {
        "metadata_total": 3,
        "pdf_ready": 0,
        "parsed": 0,
        "profiled": 1,
        "indexed": 1,
        "abstract_only": 1,
        "failed": 1,
    }
    assert snapshot["years"] == {2023: 1, 2024: 2}
    assert snapshot["venues"] == {"Remote Sensing": 3}
    assert snapshot["prediction_targets"] == {"chlorophyll-a": 1, "未分类": 2}
    assert snapshot["sensors"] == {"Sentinel-2": 1, "未分类": 2}
    assert snapshot["methods"] == {"random forest": 1, "未分类": 2}
    assert snapshot["chunks_total"] == 0


def test_presenters_keep_empty_distributions_and_escape_paper_rows():
    snapshot = {"stats": {}, "years": {}, "venues": {}, "prediction_targets": {}}

    assert knowledge_metrics(snapshot) == {}
    assert distribution_series(snapshot, "years") == []
    rows = paper_table_rows([{"title": "<script>", "status": "indexed"}])
    assert rows[0]["title"] != "<script>"


def test_paper_rows_include_source_profile_and_evidence_coverage():
    rows = paper_table_rows(
        [{
            "title": "Paper",
            "source": "crossref",
            "status": "abstract_only",
            "profiled": True,
            "evidence_chunks": 12,
        }]
    )

    assert rows[0]["source"] == "crossref"
    assert rows[0]["profiled"] == "True"
    assert rows[0]["evidence_chunks"] == "12"


def test_knowledge_metrics_keeps_profiled_count_for_overview_cards():
    metrics = knowledge_metrics({"stats": {"metadata_total": 8, "profiled": 3}})

    assert metrics == {"metadata_total": 8, "profiled": 3}


def test_agent_diagnostics_renderer_whitelists_and_escapes_runtime_fields():
    rendered = render_agent_diagnostics(
        {
            "skill_id": "evidence_qa",
            "route_mode": "model",
            "route_reason": "model_structured_route",
            "skill_version": "1.0",
            "evidence_sufficient": True,
            "retrieval_candidates": 20,
            "evidence_chunks": 5,
            "citation_count": 2,
            "fallback": False,
            "retrieval_ms": 3.5,
            "model_ms": 8.25,
            "total_ms": 11.75,
            "query": "https://secret.example/path",
            "unknown": "should not render",
        }
    )

    assert "evidence\\_qa" in rendered
    assert "模型结构化路由" in rendered
    assert "1\\.0" in rendered
    assert "本地证据：充足" in rendered
    assert "20" in rendered
    assert "11.750" in rendered
    assert "secret.example" not in rendered
    assert "should not render" not in rendered
