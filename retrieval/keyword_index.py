"""SQLite FTS5 keyword retrieval over stored evidence chunks."""

from __future__ import annotations

import sqlite3
import unicodedata

from domain.models import EvidenceChunk
from storage.database import ResearchDatabase
from retrieval.query_expansion import plan_query


class KeywordSearchError(RuntimeError):
    """Stable boundary for database/index failures unrelated to user syntax."""


def _safe_fts_query(query: str, *, operator: str = "AND") -> str:
    terms: list[str] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFKC", query):
        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or character == "_":
            current.append(character)
        elif current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    terms = [
        term
        for term in terms
        if len(term) > 1 or not term.isascii() or term.isdigit()
    ]
    if operator not in {"AND", "OR"}:
        raise ValueError("keyword_operator_invalid")
    return f" {operator} ".join(f'"{term}"' for term in terms)


class KeywordIndex:
    """Search the transactionally synchronized FTS projection."""

    def __init__(self, database: ResearchDatabase) -> None:
        self.database = database

    def search(
        self,
        query: str,
        k: int,
        paper_ids: list[str] | None = None,
    ) -> list[EvidenceChunk]:
        if k <= 0 or paper_ids == []:
            return []
        # BM25 keeps the original wording plus constrained keyword variants;
        # dense-only semantic rewrites are left to the vector branch.
        queries = plan_query(query).keyword_queries
        if not queries:
            return []
        sql = """
            SELECT chunks.*, bm25(chunk_fts) AS rank
            FROM chunk_fts
            JOIN chunks ON chunks.chunk_id = chunk_fts.chunk_id
            WHERE chunk_fts MATCH ?
        """
        rows_by_id = {}
        try:
            with self.database._connect() as connection:
                for expanded in queries:
                    fts_query = _safe_fts_query(expanded)
                    if not fts_query:
                        continue
                    parameters: list[object] = [fts_query]
                    scoped_sql = sql
                    if paper_ids is not None:
                        placeholders = ", ".join("?" for _ in paper_ids)
                        scoped_sql += f" AND chunks.paper_id IN ({placeholders})"
                        parameters.extend(paper_ids)
                    scoped_sql += " ORDER BY rank ASC, chunks.chunk_id ASC LIMIT ?"
                    parameters.append(k)
                    rows = connection.execute(scoped_sql, parameters).fetchall()
                    if not rows and " AND " in fts_query:
                        fallback_query = _safe_fts_query(expanded, operator="OR")
                        fallback_parameters = [fallback_query, *parameters[1:]]
                        rows = connection.execute(scoped_sql, fallback_parameters).fetchall()
                    for row in rows:
                        existing = rows_by_id.get(row["chunk_id"])
                        if existing is None or float(row["rank"]) < float(existing["rank"]):
                            rows_by_id[row["chunk_id"]] = row
        except sqlite3.Error:
            raise KeywordSearchError("keyword_search_failed") from None
        rows = sorted(rows_by_id.values(), key=lambda row: (float(row["rank"]), row["chunk_id"]))[:k]
        return [
            EvidenceChunk(
                chunk_id=row["chunk_id"],
                paper_id=row["paper_id"],
                title=row["title"],
                page_number=row["page_number"],
                section=row["section"],
                text=row["text"],
                score=-float(row["rank"]),
            )
            for row in rows
        ]
