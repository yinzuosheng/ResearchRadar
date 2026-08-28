"""Select a bounded, evidence-backed core corpus without deleting the catalog."""

from __future__ import annotations

import math

from workflows.relevance import relevance_groups


class CorpusCurator:
    """Rank papers with evidence and return IDs for the active RAG pool."""

    def __init__(self, database) -> None:
        self.database = database

    def select_ids(self, *, limit: int = 600) -> list[str]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("corpus_invalid_limit")
        papers = {paper.paper_id: paper for paper in self.database.list_papers(limit=10000)}
        chunks_by_paper: dict[str, list] = {}
        for chunk in self.database.list_chunks():
            chunks_by_paper.setdefault(chunk.paper_id, []).append(chunk)
        ranked = []
        for paper_id, chunks in chunks_by_paper.items():
            paper = papers.get(paper_id)
            if paper is None:
                continue
            groups = len(relevance_groups(paper))
            fulltext = any(chunk.page_number > 0 for chunk in chunks)
            evidence = 1.0 if fulltext else 0.5
            relevance = groups / 3.0
            citations = math.log1p(max(int(paper.cited_by_count or 0), 0)) / 10.0
            ranked.append((evidence + relevance + min(citations, 1.0), paper_id, groups, fulltext))

        ranked.sort(key=lambda item: (-item[0], -item[2], not item[3], item[1]))
        preferred = [item for item in ranked if item[2] >= 2]
        fallback = [item for item in ranked if item[2] < 2 and item[3]]
        return [item[1] for item in (preferred + fallback)[:limit]]

    def report(self, *, limit: int = 600) -> dict[str, object]:
        ids = self.select_ids(limit=limit)
        chunks = {chunk.paper_id for chunk in self.database.list_chunks() if chunk.paper_id in ids}
        fulltext = {
            chunk.paper_id
            for chunk in self.database.list_chunks()
            if chunk.paper_id in ids and chunk.page_number > 0
        }
        return {
            "limit": limit,
            "selected_papers": len(ids),
            "selected_fulltext_papers": len(fulltext),
            "selected_abstract_evidence_papers": len(chunks - fulltext),
            "paper_ids": ids,
        }
