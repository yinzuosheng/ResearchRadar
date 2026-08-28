from domain.models import EvidenceChunk, PaperCandidate
from retrieval.keyword_index import KeywordIndex
from retrieval.query_expansion import classify_question, expand_query, plan_query
from storage.database import ResearchDatabase


def test_expand_query_maps_chinese_water_quality_terms_to_english_variants():
    variants = expand_query("叶绿素a 水质 遥感")

    assert variants[0] == "叶绿素a 水质 遥感"
    assert any("chlorophyll-a" in item for item in variants)
    assert any("water quality" in item for item in variants)
    assert any("remote sensing" in item for item in variants)
    assert len(variants) <= 6


def test_classify_question_distinguishes_general_concept_and_research_intents():
    assert classify_question("你好，今天怎么样") == "general_chat"
    assert classify_question("叶绿素是什么") == "concept_explanation"
    assert classify_question("What is chlorophyll?") == "concept_explanation"
    assert classify_question("哪种传感器用于叶绿素a预测？") == "evidence_qa"
    assert classify_question("如何设计 Sentinel-2 的基线实验？") == "research_plan"


def test_plan_query_returns_semantic_intent_and_bounded_subqueries():
    plan = plan_query("如何用卫星反演湖泊叶绿素？")

    assert plan.intent == "research_plan"
    assert plan.needs_local_evidence is True
    assert plan.confidence >= 0.8
    assert any("chlorophyll-a" in query for query in plan.queries)
    assert any("retrieval" in query or "prediction" in query for query in plan.queries)
    assert len(plan.queries) <= 6
    assert plan.keyword_queries[0] == plan.normalized_query
    assert plan.semantic_queries[0] == plan.normalized_query
    assert any("chlorophyll-a" in query for query in plan.semantic_queries)
    assert plan.metadata_queries


def test_plan_query_normalizes_synonyms_for_retrieval_language():
    plan = plan_query("湖泊叶绿素反演")

    assert any("lake" in query.lower() for query in plan.queries)
    assert any("retrieval" in query.lower() or "estimation" in query.lower() for query in plan.queries)


def test_plan_query_expands_domain_abbreviations_for_metadata_lookup():
    plan = plan_query("TSS 与 Chl-a 的预测")

    assert "total suspended solids" in plan.abbreviation_expansions
    assert any("chlorophyll-a" in query for query in plan.semantic_queries)


def test_keyword_index_uses_expanded_variants_for_chinese_question(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db")
    paper = db.upsert_candidate(
        PaperCandidate(
            source="test", source_id="p1", title="Chlorophyll-a retrieval",
            abstract="chlorophyll-a water quality remote sensing", year=2024,
        )
    )
    db.replace_chunks(
        paper.paper_id,
        [EvidenceChunk(
            chunk_id=f"{paper.paper_id}:p1:c0", paper_id=paper.paper_id,
            title="Chlorophyll-a retrieval", page_number=1,
            text="Chlorophyll-a is a water quality parameter retrieved from remote sensing.",
        )],
    )

    rows = KeywordIndex(db).search("叶绿素a 水质 遥感", 5)

    assert rows
    assert rows[0].paper_id == paper.paper_id


def test_vector_search_runs_multiple_expanded_queries_for_cjk_input(tmp_path):
    from rag.vector_store import VectorStoreService

    class Backing:
        def __init__(self):
            self.search_calls = []

        def similarity_search_with_score(self, query, **kwargs):
            self.search_calls.append(query)
            return []

    backing = Backing()
    service = VectorStoreService(
        embeddings=object(), store=backing,
        store_path=tmp_path / "vector_store",
        repository_root=tmp_path,
    )

    service.search("叶绿素a 水质遥感", k=3)

    assert len(backing.search_calls) >= 2
    assert any("chlorophyll-a" in query for query in backing.search_calls)


def test_vector_search_fuses_query_variants_by_rank_instead_of_first_hit_order(tmp_path):
    from langchain_core.documents import Document
    from rag.vector_store import VectorStoreService

    first = Document(
        page_content="original evidence",
        metadata={"chunk_id": "first", "paper_id": "p1", "title": "First", "page_number": 1, "section": None},
    )
    expanded = Document(
        page_content="expanded evidence",
        metadata={"chunk_id": "expanded", "paper_id": "p2", "title": "Expanded", "page_number": 1, "section": None},
    )

    class Backing:
        def similarity_search_with_score(self, query, **kwargs):
            if query == "叶绿素":
                return [(expanded, 0.1), (first, 0.9)]
            return [(first, 0.1), (expanded, 0.9)]

    service = VectorStoreService(
        embeddings=object(), store=Backing(), store_path=tmp_path / "vector_store", repository_root=tmp_path
    )

    results = service.search("叶绿素", k=2)

    assert {item.chunk_id for item in results} == {"first", "expanded"}
    assert results[0].chunk_id == "expanded"
    assert results[0].score < 0.1  # RRF score, not the raw FAISS distance score


def test_vector_search_limits_multi_query_fusion_to_requested_k(tmp_path):
    from langchain_core.documents import Document
    from rag.vector_store import VectorStoreService

    documents = [
        Document(
            page_content=f"evidence {index}",
            metadata={
                "chunk_id": f"chunk-{index}",
                "paper_id": f"paper-{index}",
                "title": f"Title {index}",
                "page_number": 1,
                "section": None,
            },
        )
        for index in range(3)
    ]

    class Backing:
        def similarity_search_with_score(self, query, **kwargs):
            return [(document, float(index)) for index, document in enumerate(documents)]

    service = VectorStoreService(
        embeddings=object(),
        store=Backing(),
        store_path=tmp_path / "vector_store",
        repository_root=tmp_path,
    )

    assert len(service.search("叶绿素", k=2)) == 2
