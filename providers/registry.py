"""Composition root for discovery, OA resolution, and metadata enrichment."""

from __future__ import annotations

import os

from domain.models import PaperCandidate
from providers.base import FullTextLocation
from providers.core import CoreProvider
from providers.crossref import CrossrefEnricher
from providers.openalex import OpenAlexProvider
from providers.semantic_scholar import SemanticScholarResolver
from providers.unpaywall import UnpaywallResolver
from storage.database import normalize_doi, normalize_title
from utils.http import RetryableHttpError


class ProviderRegistry:
    def __init__(
        self,
        *,
        openalex=None,
        unpaywall=None,
        core=None,
        crossref=None,
        semantic_scholar=None,
        use_core: bool | None = None,
        use_semantic_scholar: bool | None = None,
    ) -> None:
        supplied_core = core is not None
        supplied_semantic_scholar = semantic_scholar is not None
        self.openalex = openalex or OpenAlexProvider()
        self.unpaywall = unpaywall or UnpaywallResolver()
        self.core = core or CoreProvider()
        self.crossref = crossref or CrossrefEnricher()
        self.semantic_scholar = semantic_scholar or SemanticScholarResolver()
        self.use_core = (
            use_core
            if use_core is not None
            else supplied_core or bool(os.environ.get("CORE_API_KEY", "").strip())
        )
        self.use_semantic_scholar = (
            use_semantic_scholar
            if use_semantic_scholar is not None
            else supplied_semantic_scholar
            or bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip())
        )
        self.full_text_provider: str | None = None

    def set_full_text_provider(self, provider: str | None) -> None:
        if provider not in {None, "semantic_scholar"}:
            raise ValueError("retry_fulltext_invalid_provider")
        self.full_text_provider = provider

    def discover(
        self, query: str, *, from_year: int | None, max_results: int
    ) -> list[PaperCandidate]:
        try:
            results = self.openalex.search(
                query, from_year=from_year, max_results=max_results
            )
        except RetryableHttpError:
            fallback = getattr(self.crossref, "search", None)
            if fallback is None:
                raise
            results = fallback(
                query, from_year=from_year, max_results=max_results
            )
        if self.use_core and len(results) < max_results:
            results.extend(
                self.core.search(
                    query,
                    from_year=from_year,
                    max_results=max_results,
                )
            )
        return self._deduplicate_candidates(results)[:max_results]

    def resolve_full_text(self, candidate: PaperCandidate) -> list[FullTextLocation]:
        if self.full_text_provider == "semantic_scholar":
            if not self.use_semantic_scholar:
                return []
            return self.semantic_scholar.resolve(candidate)
        locations: list[FullTextLocation] = []
        # OpenAlex candidates commonly contain a direct repository PDF.  Once
        # that resolver supplies an OA location, querying DOI fallbacks adds
        # latency and rate-limit pressure without improving the first attempt.
        # Fallbacks remain available for records where OpenAlex has no usable
        # location (for example CORE/Crossref-discovered records).
        openalex_resolver = getattr(self.openalex, "resolve", None)
        if openalex_resolver is not None:
            locations.extend(openalex_resolver(candidate))
        has_openalex_oa = any(
            location.is_oa and location.url
            for location in locations
            if location.provider == "openalex"
        )
        if not has_openalex_oa:
            fallback_providers = [self.unpaywall]
            if self.use_semantic_scholar:
                fallback_providers.append(self.semantic_scholar)
            if self.use_core:
                fallback_providers.append(self.core)
            for provider in fallback_providers:
                resolver = getattr(provider, "resolve", None)
                if resolver is not None:
                    locations.extend(resolver(candidate))
        unique: list[FullTextLocation] = []
        seen: set[str] = set()
        for location in locations:
            if location.url not in seen:
                seen.add(location.url)
                unique.append(location)
        return unique

    def resolve_fallback_full_text(self, candidate: PaperCandidate) -> list[FullTextLocation]:
        """Resolve slower DOI fallbacks after the preferred locations fail."""
        if self.full_text_provider == "semantic_scholar":
            return []
        locations: list[FullTextLocation] = []
        providers = [self.unpaywall]
        if self.use_semantic_scholar:
            providers.append(self.semantic_scholar)
        if self.use_core:
            providers.append(self.core)
        for provider in providers:
            resolver = getattr(provider, "resolve", None)
            if resolver is not None:
                locations.extend(resolver(candidate))
        unique: list[FullTextLocation] = []
        seen: set[str] = set()
        for location in locations:
            if location.url not in seen:
                seen.add(location.url)
                unique.append(location)
        return unique

    def enrich(self, candidate: PaperCandidate) -> PaperCandidate:
        if candidate.source == "crossref":
            return candidate
        if normalize_doi(candidate.doi) is None:
            return candidate
        # Discovery providers already return the fields needed for ingestion;
        # avoid a second DOI metadata request for complete records.  Crossref
        # remains the enrichment path for genuinely incomplete candidates.
        if candidate.title.strip() and candidate.year and candidate.venue and candidate.authors:
            return candidate
        return self.crossref.enrich(candidate)

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[PaperCandidate],
    ) -> list[PaperCandidate]:
        unique: list[PaperCandidate] = []
        seen: set[tuple[object, ...]] = set()
        for candidate in candidates:
            doi = normalize_doi(candidate.doi)
            identity = (
                ("doi", doi)
                if doi
                else ("title", normalize_title(candidate.title), candidate.year)
            )
            if identity not in seen:
                seen.add(identity)
                unique.append(candidate)
        return unique
