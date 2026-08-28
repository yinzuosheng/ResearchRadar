"""Transparent local ranking and resumable seed-corpus collection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any

from domain.models import PaperCandidate, SeedReport
from domain.statuses import ABSTRACT_ONLY, DISCOVERED, FAILED, INDEXED, PARSED
from storage.database import normalize_doi, normalize_title, source_fingerprint
from workflows.relevance import is_relevant


TERMINAL_STATUSES = {INDEXED, ABSTRACT_ONLY}
STABLE_FAILURE_CODES = {
    "full_text_resolution_failed",
    "ingestion_failed",
    "invalid_pdf",
    "metadata_enrichment_failed",
    "no_open_full_text",
    "pdf_download_failed",
    "pdf_parse_failed",
    "pdf_too_large",
    "profile_extraction_failed",
    "vector_index_failed",
}


@dataclass(frozen=True)
class _RankedCandidate:
    candidate: PaperCandidate
    relevance: float
    recency: float
    citation: float
    usable_oa: bool

    @property
    def score(self) -> float:
        return (
            0.50 * self.relevance
            + 0.20 * self.recency
            + 0.20 * self.citation
            + 0.10 * float(self.usable_oa)
        )


def _positive_integer(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"seed_invalid_{name}")
    return value


def _nonnegative_integer(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(code)
    return value


def _usable_oa(candidate: PaperCandidate) -> bool:
    if candidate.source == "crossref" and candidate.doi:
        # Crossref discovery has DOI metadata; Unpaywall resolves OA status later.
        return True
    return bool(
        candidate.pdf_url
        and candidate.pdf_url.strip()
        and candidate.license
        and candidate.license.strip()
    )


def _identity(candidate: PaperCandidate) -> tuple[object, ...]:
    doi = normalize_doi(candidate.doi)
    return (
        ("doi", doi)
        if doi
        else ("title", normalize_title(candidate.title), candidate.year)
    )


def _title_identity(candidate: PaperCandidate) -> tuple[object, ...]:
    return ("title", normalize_title(candidate.title), candidate.year)


def _stable_identity(candidate: PaperCandidate) -> str:
    return "|".join("" if part is None else str(part) for part in _identity(candidate))


class SeedService:
    """Discover once per query, rank locally, persist, then ingest legal OA items."""

    def __init__(self, registry, ingestor, database) -> None:
        self.registry = registry
        self.ingestor = ingestor
        self.database = database

    def collect(self, config: dict[str, Any]) -> SeedReport:
        (
            target_metadata,
            target_fulltext,
            queries,
            from_year,
            recent_year,
            recent_target,
            representative_target,
            minimum_relevance_groups,
        ) = self._validate(config)
        discovered = self._discover(
            queries, from_year, target_metadata, minimum_relevance_groups
        )
        ranked = self._rank(discovered, from_year)
        selected = self._select(
            ranked,
            target_metadata=target_metadata,
            recent_year=recent_year,
            recent_target=recent_target,
            representative_target=representative_target,
        )
        terminal_statuses = set(TERMINAL_STATUSES)
        if getattr(self.ingestor, "offline_only", False):
            terminal_statuses.add(PARSED)

        # Persist the complete selection before any fallible download/ingestion work.
        prior = {id(item): self.database.find_candidate(item.candidate) for item in selected}
        records = {id(item): self.database.upsert_candidate(item.candidate) for item in selected}
        for item in selected:
            existing = prior[id(item)]
            if (
                existing is not None
                and existing.status in terminal_statuses
                and existing.source_fingerprint != source_fingerprint(item.candidate)
            ):
                self.database.update_status(records[id(item)].paper_id, DISCOVERED)

        attempts = 0
        indexed = 0
        failures: Counter[str] = Counter()
        for item in selected:
            if attempts >= target_fulltext:
                break
            if not item.usable_oa:
                continue
            existing = prior[id(item)]
            if (
                existing is not None
                and existing.status in terminal_statuses
                and existing.source_fingerprint == source_fingerprint(item.candidate)
            ):
                continue
            attempts += 1
            try:
                result = self.ingestor.ingest(item.candidate)
            except Exception:
                failures["ingestion_failed"] += 1
                continue
            if result.status == INDEXED:
                indexed += 1
            elif result.status in {FAILED, ABSTRACT_ONLY}:
                record = self.database.get_paper(result.paper_id)
                proposed = record.last_error if record else None
                code = proposed if proposed in STABLE_FAILURE_CODES else "ingestion_failed"
                failures[code] += 1

        return SeedReport(
            metadata_count=len(records),
            fulltext_count=attempts,
            indexed_count=indexed,
            failures=dict(sorted(failures.items())),
        )

    @staticmethod
    def _validate(config: dict[str, Any]):
        target_metadata = _positive_integer(config, "target_metadata")
        target_fulltext = _positive_integer(config, "target_fulltext")
        if target_fulltext > target_metadata:
            raise ValueError("seed_fulltext_exceeds_metadata")
        from_year = config.get("from_year")
        if isinstance(from_year, bool) or not isinstance(from_year, int) or from_year <= 0:
            raise ValueError("seed_invalid_from_year")
        if from_year > datetime.now().year:
            raise ValueError("seed_from_year_in_future")
        queries = config.get("queries")
        if not isinstance(queries, list) or not queries or not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError("seed_queries_required")
        recent = config.get("recent_queries") or {}
        representative = config.get("representative_queries") or {}
        recent_year = recent.get("from_year", from_year)
        if isinstance(recent_year, bool) or not isinstance(recent_year, int) or recent_year <= 0:
            raise ValueError("seed_invalid_recent_from_year")
        recent_target = _nonnegative_integer(recent.get("target", 0), "seed_invalid_recent_target")
        representative_target = _nonnegative_integer(
            representative.get("target", 0), "seed_invalid_representative_target"
        )
        if representative.get("sort", "cited_by_count") != "cited_by_count":
            raise ValueError("seed_invalid_representative_sort")
        minimum_relevance_groups = config.get("minimum_relevance_groups", 0)
        if (
            isinstance(minimum_relevance_groups, bool)
            or not isinstance(minimum_relevance_groups, int)
            or minimum_relevance_groups < 0
            or minimum_relevance_groups > 3
        ):
            raise ValueError("seed_invalid_relevance_groups")
        return (
            target_metadata,
            target_fulltext,
            [q.strip() for q in queries],
            from_year,
            recent_year,
            min(recent_target, target_metadata),
            min(representative_target, target_metadata),
            minimum_relevance_groups,
        )

    def _discover(
        self,
        queries: list[str],
        from_year: int,
        target: int,
        minimum_relevance_groups: int = 0,
    ):
        unique: list[dict[str, Any]] = []
        denominator = max(1, target - 1)
        for query in queries:
            results = self.registry.discover(query, from_year=from_year, max_results=target)
            for rank, candidate in enumerate(results, start=1):
                if not is_relevant(candidate, minimum_groups=minimum_relevance_groups):
                    continue
                relevance = max(0.0, min(1.0, 1.0 - (rank - 1) / denominator))
                doi = normalize_doi(candidate.doi)
                title_key = _title_identity(candidate)
                same_title = [
                    item
                    for item in unique
                    if _title_identity(item["candidate"]) == title_key
                ]
                existing = next(
                    (
                        item
                        for item in same_title
                        if doi and normalize_doi(item["candidate"].doi) == doi
                    ),
                    None,
                )
                if existing is None and doi:
                    existing = next(
                        (item for item in same_title if normalize_doi(item["candidate"].doi) is None),
                        None,
                    )
                if existing is None and doi is None:
                    no_doi = [item for item in same_title if normalize_doi(item["candidate"].doi) is None]
                    doi_groups = [item for item in same_title if normalize_doi(item["candidate"].doi)]
                    existing = no_doi[0] if no_doi else doi_groups[0] if len(doi_groups) == 1 else None
                if existing is None:
                    existing = {"candidate": candidate, "relevance": relevance}
                    unique.append(existing)
                else:
                    existing["relevance"] = max(existing["relevance"], relevance)
                    stored_doi = normalize_doi(existing["candidate"].doi)
                    if (doi and not stored_doi) or (
                        doi == stored_doi and relevance > existing.get("candidate_relevance", -1.0)
                    ):
                        existing["candidate"] = candidate
                existing["candidate_relevance"] = max(
                    existing.get("candidate_relevance", -1.0), relevance
                )
        return unique

    @staticmethod
    def _rank(discovered, from_year: int) -> list[_RankedCandidate]:
        current_year = datetime.now().year
        year_span = current_year - from_year
        max_log_citations = max(
            (math.log1p(max(0, item["candidate"].cited_by_count)) for item in discovered),
            default=0.0,
        )
        ranked = []
        for item in discovered:
            candidate = item["candidate"]
            if candidate.year is None:
                recency = 0.0
            elif year_span == 0:
                recency = 1.0 if candidate.year >= current_year else 0.0
            else:
                recency = max(0.0, min(1.0, (candidate.year - from_year) / year_span))
            citation_log = math.log1p(max(0, candidate.cited_by_count))
            citation = citation_log / max_log_citations if max_log_citations else 0.0
            ranked.append(
                _RankedCandidate(
                    candidate=candidate,
                    relevance=item["relevance"],
                    recency=max(0.0, min(1.0, recency)),
                    citation=max(0.0, min(1.0, citation)),
                    usable_oa=_usable_oa(candidate),
                )
            )
        return ranked

    @staticmethod
    def _score_key(item: _RankedCandidate):
        candidate = item.candidate
        return (
            -item.score,
            -int(item.usable_oa),
            -(candidate.year or 0),
            -candidate.cited_by_count,
            _stable_identity(candidate),
        )

    @classmethod
    def _select(
        cls,
        ranked: list[_RankedCandidate],
        *,
        target_metadata: int,
        recent_year: int,
        recent_target: int,
        representative_target: int,
    ) -> list[_RankedCandidate]:
        if not ranked:
            return []
        representative = sorted(
            ranked,
            key=lambda item: (
                -item.candidate.cited_by_count,
                -int(item.usable_oa),
                -(item.candidate.year or 0),
                _stable_identity(item.candidate),
            ),
        )
        recent = sorted(
            (item for item in ranked if (item.candidate.year or 0) >= recent_year),
            key=cls._score_key,
        )
        representative_limit = min(representative_target, target_metadata)
        recent_limit = min(recent_target, target_metadata)
        if representative and recent and representative_limit + recent_limit > target_metadata:
            total = representative_limit + recent_limit
            representative_limit = max(
                1, round(target_metadata * representative_limit / total)
            )
            recent_limit = max(1, target_metadata - representative_limit)
            representative_limit = target_metadata - recent_limit
        representative_reserved = representative[:representative_limit]
        recent_reserved = recent[:recent_limit]
        selected = []
        seen = set()
        for item in representative_reserved + recent_reserved + sorted(ranked, key=cls._score_key):
            identity = _stable_identity(item.candidate)
            if identity in seen:
                continue
            if len(selected) >= target_metadata:
                break
            seen.add(identity)
            selected.append(item)
        return selected
