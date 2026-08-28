"""Lazy live provider probes used only by the selected CLI command."""

from __future__ import annotations

import os
from collections.abc import Mapping

from evaluation.health import run_provider_health


def _probe(url, *, environ, request_get, required_env=(), headers=None, params=None):
    missing = [name for name in required_env if not environ.get(name, "").strip()]
    if missing:
        return {"status": "missing_configuration"}
    response = request_get(
        url, headers=headers, params=params, timeout=10, allow_redirects=False
    )
    remaining = response.headers.get("X-RateLimit-Remaining")
    return {"status_code": response.status_code, "remaining_quota": remaining}


def default_provider_health(
    *,
    request_get=None,
    environ: Mapping[str, str] | None = None,
    dotenv_loader=None,
):
    """Read environment values only after command selection; never print them."""
    if dotenv_loader is None:
        from dotenv import load_dotenv

        dotenv_loader = load_dotenv
    dotenv_loader()
    if request_get is None:
        import requests

        request_get = requests.get
    environ = os.environ if environ is None else environ
    probes = {
        "openalex": lambda: _probe(
            "https://api.openalex.org/works",
            environ=environ,
            request_get=request_get,
            params={
                **(
                    {"api_key": environ["OPENALEX_API_KEY"]}
                    if environ.get("OPENALEX_API_KEY", "").strip()
                    else {}
                ),
                "per-page": 1,
            },
        ),
        "core": lambda: _probe(
            "https://api.core.ac.uk/v3/search/works",
            environ=environ,
            request_get=request_get,
            required_env=("CORE_API_KEY",),
            headers={"Authorization": f"Bearer {environ.get('CORE_API_KEY', '')}"},
            params={"q": "water", "limit": 1},
        ),
        "unpaywall": lambda: _probe(
            "https://api.unpaywall.org/v2/10.1038/nature12373",
            environ=environ,
            request_get=request_get,
            required_env=("UNPAYWALL_EMAIL",),
            params={"email": environ.get("UNPAYWALL_EMAIL")},
        ),
        "crossref": lambda: _probe(
            "https://api.crossref.org/works",
            environ=environ,
            request_get=request_get,
            params={"query.title": "water remote sensing", "rows": 1},
        ),
    }
    if environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip():
        probes["semantic_scholar"] = lambda: _probe(
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/nature12373",
            environ=environ,
            request_get=request_get,
            required_env=("SEMANTIC_SCHOLAR_API_KEY",),
            headers={"x-api-key": environ.get("SEMANTIC_SCHOLAR_API_KEY", "")},
            params={"fields": "openAccessPdf"},
        )
    return run_provider_health(probes)
