"""Shared contracts and small boundary helpers for literature providers."""

from __future__ import annotations

import os
from typing import Protocol
from urllib.parse import urlparse

from pydantic import BaseModel

from domain.models import PaperCandidate


class DiscoveryProvider(Protocol):
    def search(
        self, query: str, *, from_year: int | None, max_results: int
    ) -> list[PaperCandidate]: ...


class OaResolver(Protocol):
    def resolve(self, candidate: PaperCandidate) -> list["FullTextLocation"]: ...


class MetadataEnricher(Protocol):
    def enrich(self, candidate: PaperCandidate) -> PaperCandidate: ...


class FullTextLocation(BaseModel):
    provider: str
    url: str
    license: str | None = None
    is_oa: bool
    priority: int


def require_environment_variable(name: str) -> str:
    """Read a provider credential without including its value in errors."""
    try:
        value = os.environ[name]
    except KeyError as error:
        raise ValueError(f"Missing required environment variable: {name}") from error
    if not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def is_pdf_url(value: object) -> bool:
    """Return whether a URL explicitly identifies a PDF resource."""
    if not isinstance(value, str) or not value.strip():
        return False
    return urlparse(value.strip()).path.lower().endswith(".pdf")
