"""Unpaywall DOI-based open-access resolver."""

from __future__ import annotations

from domain.models import PaperCandidate
from providers.base import FullTextLocation, require_environment_variable
from storage.database import normalize_doi
from utils.config import load_tools_config
from utils.http import get_with_retry


class UnpaywallResolver:
    def __init__(self) -> None:
        config = load_tools_config().get("unpaywall", {})
        self.endpoint = config.get("endpoint", "https://api.unpaywall.org/v2")
        self.email_env = config.get("email_env", "UNPAYWALL_EMAIL")

    def resolve(self, candidate: PaperCandidate) -> list[FullTextLocation]:
        doi = normalize_doi(candidate.doi)
        if doi is None:
            return []
        email = require_environment_variable(self.email_env)
        payload = get_with_retry(
            f"{self.endpoint.rstrip('/')}/{doi}", params={"email": email}
        ).json()
        raw_locations = [payload.get("best_oa_location"), *payload.get("oa_locations", [])]
        locations: list[FullTextLocation] = []
        seen: set[str] = set()
        for raw in raw_locations:
            if not isinstance(raw, dict) or raw.get("is_oa") is not True:
                continue
            url = raw.get("url_for_pdf")
            if not isinstance(url, str) or not url.strip() or url.strip() in seen:
                continue
            seen.add(url.strip())
            locations.append(
                FullTextLocation(
                    provider="unpaywall",
                    url=url.strip(),
                    license=raw.get("license") or None,
                    is_oa=True,
                    priority=40 + len(locations),
                )
            )
        return locations


UnpaywallProvider = UnpaywallResolver
