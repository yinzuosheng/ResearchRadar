"""Backward-compatible rendering over citation-safe question answering."""

from __future__ import annotations

from domain.models import CitedAnswer


class RagSummarizeService:
    def __init__(self, *, cited_qa=None) -> None:
        self._cited_qa = cited_qa or self._build_default_service()

    @staticmethod
    def _build_default_service():
        from model.factory import build_chat_model
        from rag.vector_store import VectorStoreService
        from retrieval.hybrid import HybridRetriever
        from retrieval.keyword_index import KeywordIndex
        from storage.database import ResearchDatabase
        from storage.paths import default_database_path
        from utils.config import load_rag_config
        from workflows.qa import CitedQaService

        config = load_rag_config()
        database = ResearchDatabase(default_database_path())
        vector = VectorStoreService(database=database)
        hybrid = HybridRetriever(
            KeywordIndex(database),
            vector,
            keyword_weight=float(config.get("keyword_weight", 2.0)),
            vector_weight=float(config.get("vector_weight", 0.5)),
            rrf_k=int(config.get("rrf_k", 60)),
            candidate_k=int(config.get("candidate_k", 20)),
            max_chunks_per_paper=int(config.get("max_chunks_per_paper", 4)),
        )
        return CitedQaService(
            hybrid,
            build_chat_model(),
            chunk_store=database,
            answer_k=int(config.get("answer_k", 8)),
        )

    def cited_answer(self, query: str) -> CitedAnswer:
        return self._cited_qa.answer(query)

    def rag_summarize(self, query: str) -> str:
        answer = self.cited_answer(query)
        citations = "\n".join(
            f"[{citation.title}, p. {citation.page_number}]"
            for citation in answer.citations
        )
        return (
            f"{answer.answer_markdown}\n\n{citations}"
            if citations
            else answer.answer_markdown
        )

    def rag_report(self, query: str) -> str:
        return self.rag_summarize(query)
