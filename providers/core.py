"""CORE v3 discovery fallback and repository-PDF resolver."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from domain.models import PaperCandidate
from providers.base import FullTextLocation, is_pdf_url, require_environment_variable
from storage.database import normalize_doi, normalize_title
from utils.config import load_tools_config
from utils.http import get_with_retry


def _core_pdf_url(work: dict[str, Any]) -> str | None:
    download_url = work.get("downloadUrl")
    if isinstance(download_url, str) and download_url.strip():
        return download_url.strip()
    full_text = work.get("fullTextIdentifier")
    if is_pdf_url(full_text):
        return full_text.strip()
    for link in work.get("links", []):
        if isinstance(link, dict):
            link = link.get("url")
        if is_pdf_url(link):
            return link.strip()
    return None


def _is_core_download_url(value: object) -> bool:
    """Return whether a URL is a CORE-hosted download endpoint."""
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme in {"http", "https"}
        and (host == "core.ac.uk" or host.endswith(".core.ac.uk"))
        and parsed.path.lower().startswith("/download/")
    )


class CoreProvider:
    def __init__(self) -> None:
        config = load_tools_config().get("core", {})
        self.endpoint = config.get("endpoint", "https://api.core.ac.uk/v3")
        self.api_key_env = config.get("api_key_env", "CORE_API_KEY")
        self._locations: dict[tuple[object, ...], FullTextLocation] = {}

    def search(
        self, query: str, *, from_year: int | None, max_results: int
    ) -> list[PaperCandidate]:
        api_key = require_environment_variable(self.api_key_env)
        payload = get_with_retry(
            f"{self.endpoint.rstrip('/')}/search/works",
            params={"q": query, "limit": max_results},
            headers={"Authorization": f"Bearer {api_key}"},
        ).json()
        candidates: list[PaperCandidate] = []
        for work in payload.get("results", []):
            if not isinstance(work, dict):
                continue
            candidate = self._candidate_from_work(work)
            # CORE v3's required search boundary has no from-year parameter;
            # applying it here keeps the registry contract consistent.
            if from_year is not None and candidate.year is not None and candidate.year < from_year:
                continue
            candidates.append(candidate)
            if candidate.pdf_url:
                self._locations[self._identity(candidate)] = FullTextLocation(
                    provider="core",
                    url=candidate.pdf_url,
                    license=candidate.license,
                    is_oa=True,
                    priority=60,
                )
        return candidates

    def resolve(self, candidate: PaperCandidate) -> list[FullTextLocation]:
        cached = self._locations.get(self._identity(candidate))
        if cached:
            return [cached]
        if (
            candidate.source == "core"
            and (
                is_pdf_url(candidate.pdf_url)
                or _is_core_download_url(candidate.pdf_url)
            )
        ):
            return [
                FullTextLocation(
                    provider="core",
                    url=candidate.pdf_url.strip(),
                    license=candidate.license,
                    is_oa=True,
                    priority=60,
                )
            ]
        return []

    @staticmethod
    def _identity(candidate: PaperCandidate) -> tuple[object, ...]:
        doi = normalize_doi(candidate.doi)
        return ("doi", doi) if doi else ("title", normalize_title(candidate.title), candidate.year)

    @staticmethod
    def _candidate_from_work(work: dict[str, Any]) -> PaperCandidate:
        authors = []
        for author in work.get("authors", []):
            if isinstance(author, str):
                name = author
            elif isinstance(author, dict):
                name = author.get("name") or author.get("displayName")
            else:
                name = None
            if isinstance(name, str) and name.strip():
                authors.append(name.strip())
        pdf_url = _core_pdf_url(work)
        return PaperCandidate(
            source="core",
            source_id=str(work.get("id") or work.get("coreId") or ""),
            title=str(work.get("title") or "").strip(),
            doi=normalize_doi(work.get("doi")),
            authors=authors,
            year=work.get("yearPublished") or work.get("publishedYear"),
            venue=work.get("publisher") or work.get("journals"),
            abstract=work.get("abstract"),
            landing_url=work.get("fullTextIdentifier") if not pdf_url else None,
            pdf_url=pdf_url,
            license=work.get("license") or None,
        )
