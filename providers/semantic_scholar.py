"""Semantic Scholar DOI resolver for declared open-access PDFs."""

from __future__ import annotations

import os
import time
from urllib.parse import quote

import requests

from domain.models import PaperCandidate
from providers.base import FullTextLocation, require_environment_variable
from storage.database import normalize_doi
from utils.config import load_tools_config
from utils.http import RetryableHttpError, get_once


_OPEN_ACCESS_STATUSES = frozenset({"GREEN", "GOLD", "HYBRID", "BRONZE"})


class SemanticScholarResolver:
    """Resolve a candidate DOI through Semantic Scholar's OA metadata."""

    def __init__(self) -> None:
        config = load_tools_config().get("semantic_scholar", {})
        self.endpoint = config.get(
            "endpoint", "https://api.semanticscholar.org/graph/v1"
        )
        self.api_key_env = config.get("api_key_env", "SEMANTIC_SCHOLAR_API_KEY")
        self.requests_per_second = float(config.get("requests_per_second", 0.5) or 0.5)
        self._next_request_at = 0.0

    def resolve(self, candidate: PaperCandidate) -> list[FullTextLocation]:
        doi = normalize_doi(candidate.doi)
        if doi is None:
            return []
        api_key = require_environment_variable(self.api_key_env)
        self._throttle()
        try:
            payload = get_once(
                f"{self.endpoint.rstrip('/')}/paper/{quote(f'DOI:{doi}', safe=':')}",
                headers={"x-api-key": api_key},
                params={"fields": "openAccessPdf"},
            ).json()
        except (RetryableHttpError, requests.RequestException):
            # A stale DOI or a temporary provider limit must not abort the
            # lawful resolver chain; the caller can continue with other OA
            # sources and retain a stable ingestion failure code if none work.
            return []
        raw = payload.get("openAccessPdf") if isinstance(payload, dict) else None
        if not isinstance(raw, dict):
            return []
        url = raw.get("url")
        if not isinstance(url, str) or not url.strip():
            return []
        status = raw.get("status")
        if isinstance(status, str) and status.strip().upper() not in _OPEN_ACCESS_STATUSES:
            return []
        return [
            FullTextLocation(
                provider="semantic_scholar",
                url=url.strip(),
                license=None,
                is_oa=True,
                priority=50,
            )
        ]

    def _throttle(self) -> None:
        if self.requests_per_second <= 0:
            return
        now = time.monotonic()
        delay = self._next_request_at - now
        if delay > 0:
            time.sleep(delay)
        self._next_request_at = time.monotonic() + (1.0 / self.requests_per_second)


SemanticScholarProvider = SemanticScholarResolver
