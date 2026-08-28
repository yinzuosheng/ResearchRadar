"""Crossref metadata enrichment for DOI-bearing candidates."""

from __future__ import annotations

import re
from urllib.parse import quote

from domain.models import PaperCandidate
from providers.base import require_environment_variable
from storage.database import normalize_doi
from utils.config import load_tools_config
from utils.http import get_with_retry


def _first(value: object) -> object | None:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _authors(message: dict) -> list[str]:
    names: list[str] = []
    for author in message.get("author", []):
        if not isinstance(author, dict):
            continue
        name = author.get("name")
        if not name:
            name = " ".join(
                part.strip()
                for part in (author.get("given", ""), author.get("family", ""))
                if isinstance(part, str) and part.strip()
            )
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _year(message: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        date_parts = (message.get(key) or {}).get("date-parts", [])
        if date_parts and date_parts[0] and isinstance(date_parts[0][0], int):
            return date_parts[0][0]
    return None


class CrossrefEnricher:
    def __init__(self) -> None:
        config = load_tools_config().get("crossref", {})
        self.endpoint = config.get("endpoint", "https://api.crossref.org/works")
        self.email_env = config.get("email_env", "UNPAYWALL_EMAIL")

    def search(
        self, query: str, *, from_year: int | None, max_results: int
    ) -> list[PaperCandidate]:
        """Discover DOI-bearing works when the primary catalog is unavailable."""
        email = require_environment_variable(self.email_env)
        params = {
            "query.bibliographic": query,
            "rows": min(max_results, 100),
            "mailto": email,
        }
        if from_year is not None:
            params["filter"] = f"from-pub-date:{from_year}-01-01"
        payload = get_with_retry(self.endpoint, params=params).json()
        items = (payload.get("message") or {}).get("items", [])
        candidates: list[PaperCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            doi = normalize_doi(item.get("DOI"))
            title = _first(item.get("title"))
            if not doi or not isinstance(title, str) or not title.strip():
                continue
            candidates.append(
                PaperCandidate(
                    source="crossref",
                    source_id=doi,
                    title=title.strip(),
                    doi=doi,
                    authors=_authors(item),
                    year=_year(item),
                    venue=_first(item.get("container-title")),
                    abstract=_abstract(item.get("abstract")),
                    landing_url=item.get("URL"),
                    cited_by_count=int(item.get("is-referenced-by-count") or 0),
                )
            )
        return candidates

    def enrich(self, candidate: PaperCandidate) -> PaperCandidate:
        doi = normalize_doi(candidate.doi)
        if doi is None:
            return candidate
        email = require_environment_variable(self.email_env)
        payload = get_with_retry(
            f"{self.endpoint.rstrip('/')}/{quote(doi, safe='')}",
            params={"mailto": email},
        ).json()
        message = payload.get("message") or {}
        abstract = message.get("abstract")
        if isinstance(abstract, str):
            abstract = re.sub(r"<[^>]+>", "", abstract).strip() or None
        updates = {
            "title": _first(message.get("title")),
            "authors": _authors(message),
            "year": _year(message),
            "venue": _first(message.get("container-title")),
            "abstract": abstract,
            "landing_url": message.get("URL"),
        }
        merged = candidate.model_dump()
        for field, value in updates.items():
            if merged.get(field) in (None, "", []) and value not in (None, "", []):
                merged[field] = value
        return PaperCandidate(**merged)


def _abstract(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return re.sub(r"<[^>]+>", "", value).strip() or None


CrossrefProvider = CrossrefEnricher
