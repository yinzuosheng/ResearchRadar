"""OpenAlex discovery and lawful full-text location normalization."""

from __future__ import annotations

import os
import time
from urllib.parse import quote
from typing import Any

from domain.models import PaperCandidate
from providers.base import FullTextLocation, is_pdf_url, require_environment_variable
from storage.database import normalize_doi
from utils.config import load_tools_config
from utils.http import get_with_retry


def _strip_openalex_id(value: object) -> str:
    text = str(value or "").strip()
    prefix = "https://openalex.org/"
    return text[len(prefix) :] if text.startswith(prefix) else text


def _abstract_from_inverted_index(index: object) -> str | None:
    if not isinstance(index, dict) or not index:
        return None
    positions: list[tuple[int, str]] = []
    for word, indexes in index.items():
        if not isinstance(indexes, list):
            continue
        positions.extend((position, str(word)) for position in indexes if isinstance(position, int))
    if not positions:
        return None
    return " ".join(word for _, word in sorted(positions))


def _usable_license(location: object) -> str | None:
    if not isinstance(location, dict):
        return None
    license_value = location.get("license")
    if not isinstance(license_value, str) or not license_value.strip():
        return None
    return license_value.strip()


class OpenAlexProvider:
    def __init__(self) -> None:
        config = load_tools_config().get("openalex", {})
        self.endpoint = config.get("endpoint", "https://api.openalex.org/works")
        self.content_endpoint = config.get(
            "content_endpoint", "https://content.openalex.org/works"
        )
        self.api_key_env = config.get("api_key_env", "OPENALEX_API_KEY")
        self.requests_per_second = float(config.get("requests_per_second", 0) or 0)
        self._next_request_at = 0.0
        self._works: dict[str, dict[str, Any]] = {}

    def search(
        self, query: str, *, from_year: int | None, max_results: int
    ) -> list[PaperCandidate]:
        filters = ["open_access.is_oa:true"]
        if from_year is not None:
            filters.append(f"from_publication_date:{from_year}-01-01")
        params = {
            "search": query,
            "filter": ",".join(filters),
            "sort": "relevance_score:desc",
            "per_page": min(max_results, 100),
        }
        api_key = os.environ.get(self.api_key_env, "").strip()
        if api_key:
            params["api_key"] = api_key
        # Anonymous access is supported; mailto improves the polite pool.
        email = os.environ.get("UNPAYWALL_EMAIL", "").strip()
        if email:
            params["mailto"] = email
        candidates: list[PaperCandidate] = []
        cursor = None
        while len(candidates) < max_results:
            request_params = dict(params)
            if cursor:
                request_params["cursor"] = cursor
            self._throttle()
            payload = get_with_retry(self.endpoint, params=request_params).json()
            page_results = payload.get("results", [])
            if not isinstance(page_results, list) or not page_results:
                break
            for work in page_results:
                if not isinstance(work, dict):
                    continue
                candidate = self._candidate_from_work(work)
                if not candidate.title or not candidate.source_id:
                    continue
                self._works[candidate.source_id] = work
                candidates.append(candidate)
                if len(candidates) >= max_results:
                    break
            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
        return candidates

    def _throttle(self) -> None:
        if self.requests_per_second <= 0:
            return
        now = time.monotonic()
        delay = self._next_request_at - now
        if delay > 0:
            time.sleep(delay)
        self._next_request_at = time.monotonic() + (1.0 / self.requests_per_second)

    def resolve(self, candidate: PaperCandidate) -> list[FullTextLocation]:
        work = self._works.get(candidate.source_id)
        # A rehydrated candidate may already carry a validated direct OA PDF.
        # Use it before DOI refresh so offline/retry paths remain bounded.
        if work is None and is_pdf_url(candidate.pdf_url):
            return [
                FullTextLocation(
                    provider="openalex",
                    url=candidate.pdf_url,
                    license=candidate.license,
                    is_oa=True,
                    priority=10,
                )
            ]
        if work is None and candidate.doi:
            work = self._refresh_by_doi(candidate.doi)
        if work is None:
            if candidate.source != "openalex" or not is_pdf_url(candidate.pdf_url):
                return []
            return [
                FullTextLocation(
                    provider="openalex",
                    url=candidate.pdf_url,
                    license=candidate.license,
                    is_oa=True,
                    priority=10,
                )
            ]
        return self._locations_from_work(work, candidate.source_id)

    def _refresh_by_doi(self, doi: str) -> dict[str, Any] | None:
        normalized = normalize_doi(doi)
        if not normalized:
            return None
        url = f"{self.endpoint.rstrip('/')}/{quote(f'https://doi.org/{normalized}', safe='')}"
        params = {}
        api_key = os.environ.get(self.api_key_env, "").strip()
        if api_key:
            params["api_key"] = api_key
        self._throttle()
        try:
            payload = get_with_retry(url, params=params).json()
        except Exception:
            return None
        work = payload
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            work = next((item for item in payload["results"] if isinstance(item, dict)), None)
        if not isinstance(work, dict):
            return None
        source_id = _strip_openalex_id(work.get("id"))
        if source_id:
            self._works[source_id] = work
        return work

    def _candidate_from_work(self, work: dict[str, Any]) -> PaperCandidate:
        source_id = _strip_openalex_id(work.get("id"))
        authors = [
            name
            for authorship in work.get("authorships", [])
            if isinstance(authorship, dict)
            for name in [(authorship.get("author") or {}).get("display_name")]
            if isinstance(name, str) and name.strip()
        ]
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        locations = self._locations_from_work(work, source_id)
        direct = next((location for location in locations if location.priority < 30), None)
        return PaperCandidate(
            source="openalex",
            source_id=source_id,
            title=str(work.get("title") or work.get("display_name") or "").strip(),
            doi=normalize_doi(work.get("doi")),
            authors=authors,
            year=work.get("publication_year"),
            venue=source.get("display_name"),
            abstract=_abstract_from_inverted_index(work.get("abstract_inverted_index")),
            landing_url=primary.get("landing_page_url") or work.get("doi") or work.get("id"),
            pdf_url=direct.url if direct else None,
            license=direct.license if direct else None,
            cited_by_count=int(work.get("cited_by_count") or 0),
            source_updated_at=work.get("updated_date"),
        )

    def _locations_from_work(
        self, work: dict[str, Any], source_id: str
    ) -> list[FullTextLocation]:
        locations: list[FullTextLocation] = []
        seen: set[str] = set()

        best = work.get("best_oa_location")
        best_license = _usable_license(best)
        if (
            isinstance(best, dict)
            and best.get("is_oa") is True
            and is_pdf_url(best.get("pdf_url"))
            and best_license
        ):
            url = best["pdf_url"].strip()
            seen.add(url)
            locations.append(
                FullTextLocation(
                    provider="openalex",
                    url=url,
                    license=best_license,
                    is_oa=True,
                    priority=10,
                )
            )

        licensed_oa_location = best_license if isinstance(best, dict) and best.get("is_oa") is True else None
        for location in work.get("locations", []):
            if not isinstance(location, dict) or location.get("is_oa") is not True:
                continue
            license_value = _usable_license(location)
            if license_value and licensed_oa_location is None:
                licensed_oa_location = license_value
            url = location.get("pdf_url")
            if not is_pdf_url(url) or url.strip() in seen:
                continue
            seen.add(url.strip())
            locations.append(
                FullTextLocation(
                    provider="openalex",
                    url=url.strip(),
                    license=license_value,
                    is_oa=True,
                    priority=20,
                )
            )

        if licensed_oa_location:
            content_url = work.get("content_url")
            if not is_pdf_url(content_url):
                content_url = f"{self.content_endpoint.rstrip('/')}/{source_id}.pdf"
            if content_url not in seen:
                locations.append(
                    FullTextLocation(
                        provider="openalex",
                        url=content_url,
                        license=licensed_oa_location,
                        is_oa=True,
                        priority=30,
                    )
                )
        return locations
